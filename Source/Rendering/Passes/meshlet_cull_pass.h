#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_driven_helpers.h"
#include "gpu_cull_resources.h"
#include "hzb_constants.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

class MeshletCullPass : public RenderPass {
public:
    MeshletCullPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~MeshletCullPass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void setFrameContext(const FrameContext* ctx) override {
        RenderPass::setFrameContext(ctx);
        syncFrameContextFlags();
    }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource cullResult;
    FGResource visibleMeshlets;
    FGResource cullCounter;
    FGResource instanceData;

    FGResource getOutput(const std::string& name) const override {
        if (name == "cullResult") return cullResult;
        if (name == "visibleMeshlets") return visibleMeshlets;
        if (name == "cullCounter") return cullCounter;
        if (name == "instanceData") return instanceData;
        if (name == "visibilityWorklist") return visibleMeshlets;
        if (name == "visibilityIndirectArgs") return cullCounter;
        if (name == "visibilityInstances") return instanceData;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        cullResult = builder.createToken("cullResult");

        m_maxMeshlets = std::max<uint32_t>(1u, computeMaxMeshletCapacity());
        m_maxInstances = std::max<uint32_t>(1u, computeMaxInstanceCapacity());

        const auto visibleWorklist =
            GpuDriven::createIndirectWorklist<MeshletDrawInfo>(builder,
                                                               "visibleMeshlets",
                                                               "VisibleMeshletBuffer",
                                                               m_maxMeshlets,
                                                               "cullCounter",
                                                               "CullCounterBuffer",
                                                               true);
        visibleMeshlets = visibleWorklist.payload;
        cullCounter = visibleWorklist.state;

        FGBufferDesc instanceDataDesc =
            GpuDriven::makeStructuredBufferDesc<GPUInstanceData>(m_maxInstances,
                                                                 "InstanceDataBuffer",
                                                                 true);
        instanceData = builder.create("instanceData", instanceDataDesc);

        m_hzbHistoryRead.clear();
        m_hzbLevelCount = computeHzbLevelCount(static_cast<uint32_t>(m_width),
                                               static_cast<uint32_t>(m_height));
        m_hzbHistoryRead.reserve(m_hzbLevelCount);
        for (uint32_t level = 0; level < m_hzbLevelCount; ++level) {
            const std::string resourceName = hzbHistoryResourceName(level);
            m_hzbHistoryRead.push_back(
                builder.readHistory(resourceName.c_str(),
                                    makeHzbTextureDesc(static_cast<uint32_t>(m_width),
                                                       static_cast<uint32_t>(m_height),
                                                       level)));
        }
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        ZoneScopedN("MeshletCullPass");
        MICROPROFILE_SCOPEI("RenderPass", "MeshletCullPass", 0xffff8800);
        if (!m_frameContext || !m_runtimeContext) return;
        if (!m_frameContext->gpuDrivenCulling) return;

        auto cullIt = m_runtimeContext->computePipelinesRhi.find("MeshletCullPass");
        auto buildIt = m_runtimeContext->computePipelinesRhi.find("BuildIndirectPass");
        if (cullIt == m_runtimeContext->computePipelinesRhi.end() || !cullIt->second.nativeHandle()) return;
        if (buildIt == m_runtimeContext->computePipelinesRhi.end() || !buildIt->second.nativeHandle()) return;

        const auto& visibleNodes = m_frameContext->visibleMeshletNodes;
        uint32_t instanceCount = std::min<uint32_t>(
            m_frameContext->visibilityInstanceCount,
            static_cast<uint32_t>(visibleNodes.size()));
        if (instanceCount == 0) return;

        std::vector<uint32_t> validVisibleNodes;
        validVisibleNodes.reserve(instanceCount);
        for (uint32_t i = 0; i < instanceCount; ++i) {
            const uint32_t nodeID = visibleNodes[i];
            if (nodeID < m_ctx.sceneGraph.nodes.size()) {
                validVisibleNodes.push_back(nodeID);
            }
        }
        if (validVisibleNodes.empty()) return;
        instanceCount = static_cast<uint32_t>(validVisibleNodes.size());

        // Compute total meshlet count across all visible instances
        uint32_t totalMeshlets = 0;
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[validVisibleNodes[i]];
            totalMeshlets += node.meshletCount;
        }
        if (totalMeshlets == 0) return;

        m_totalMeshlets = totalMeshlets;

        if (!m_frameGraph) return;
        RhiBuffer* visibleMeshletBuffer = m_frameGraph->getBuffer(visibleMeshlets);
        RhiBuffer* counterBuffer = m_frameGraph->getBuffer(cullCounter);
        RhiBuffer* instanceDataBuffer = m_frameGraph->getBuffer(instanceData);
        if (!visibleMeshletBuffer || !counterBuffer || !instanceDataBuffer) {
            return;
        }
        GpuDriven::ensureIndirectGridCommandBufferInitialized(counterBuffer, m_initializedCounterBuffer);

        // Read back previous frame's visible count (1-frame delayed).
        // build_indirect writes count into the indirect-dispatch X slot, then resets the counter.
        // By this point the previous frame's GPU work is complete.
        m_lastVisibleCount = GpuDriven::readBuiltIndirectGridCount(counterBuffer);

        // Upload instance data (CPU PU, StorageModeShared)
        auto* instPtr = static_cast<GPUInstanceData*>(instanceDataBuffer->mappedData());
        uint32_t dispatchStart = 0u;
        for (uint32_t i = 0; i < instanceCount; i++) {
            const auto& node = m_ctx.sceneGraph.nodes[validVisibleNodes[i]];
            float4x4 nodeModelView = m_frameContext->view * node.transform.worldMatrix;
            float4x4 nodeMVP = m_frameContext->proj * nodeModelView;

            GPUInstanceData& inst = instPtr[i];
            inst.mvp = transpose(nodeMVP);
            inst.modelView = transpose(nodeModelView);
            inst.worldMatrix = transpose(node.transform.worldMatrix);
            inst.meshletStart = node.meshletStart;
            inst.meshletCount = node.meshletCount;
            inst.dispatchStart = dispatchStart;
            inst.instanceID = i;
            dispatchStart += node.meshletCount;
        }

        // Build CullUniforms
        float4x4 viewProj = m_frameContext->proj * m_frameContext->view;
        float4x4 prevViewProj = m_frameContext->prevCullProj * m_frameContext->prevCullView;
        CullUniforms cullUni{};
        cullUni.viewProj = transpose(viewProj);
        cullUni.prevViewProj = transpose(prevViewProj);
        cullUni.prevView = transpose(m_frameContext->prevCullView);
        cullUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        cullUni.prevCameraWorldPos = m_frameContext->prevCameraWorldPos;
        cullUni.prevProjScale = float2(std::abs(m_frameContext->prevCullProj[0].x),
                                       std::abs(m_frameContext->prevCullProj[1].y));
        cullUni.totalDispatchCount = totalMeshlets;
        cullUni.instanceCount = instanceCount;
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1 : 0;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1 : 0;

        std::array<const RhiTexture*, kHzbMaxLevels> hzbTextures{};
        uint32_t hzbTextureCount = 0;
        const bool historyValid =
            m_enableOcclusionCull &&
            !m_hzbHistoryRead.empty() &&
            m_frameGraph &&
            m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
        if (historyValid) {
            const uint32_t maxLevels =
                std::min<uint32_t>(static_cast<uint32_t>(m_hzbHistoryRead.size()), kHzbMaxLevels);
            for (uint32_t level = 0; level < maxLevels; ++level) {
                const RhiTexture* historyTexture = m_frameGraph->getTexture(m_hzbHistoryRead[level]);
                if (!historyTexture) {
                    break;
                }
                hzbTextures[hzbTextureCount++] = historyTexture;
            }
        }

        cullUni.enableOcclusionCull = hzbTextureCount > 0 ? 1u : 0u;
        cullUni.hzbLevelCount = hzbTextureCount;
        if (hzbTextureCount > 0) {
            cullUni.hzbTextureSize =
                float2(static_cast<float>(hzbTextures[0]->width()),
                       static_cast<float>(hzbTextures[0]->height()));
        }
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;

        // --- Dispatch 1: Meshlet cull ---
        encoder.setComputePipeline(cullIt->second);
        encoder.setBytes(&cullUni, sizeof(cullUni), GpuDriven::MeshletCullBindings::kUniforms);
        encoder.setBuffer(instanceDataBuffer, 0, GpuDriven::MeshletCullBindings::kInstanceData);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletCullBindings::kBounds);
        encoder.setBuffer(visibleMeshletBuffer, 0, GpuDriven::MeshletCullBindings::kCompactionOutput);
        encoder.setBuffer(counterBuffer, 0, GpuDriven::MeshletCullBindings::kCounter);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kHzbTextureBase,
                                hzbTextureCount);
        }

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (totalMeshlets + threadgroupSize - 1) / threadgroupSize;
        encoder.dispatchThreadgroups({threadgroups, 1, 1}, {threadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // --- Dispatch 2: Build indirect args ---
        encoder.setComputePipeline(buildIt->second);
        encoder.setBuffer(counterBuffer, 0, GpuDriven::BuildDispatchBindings::kCounter);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        static bool sLoggedGpuPublish = false;
        if (!sLoggedGpuPublish) {
            spdlog::info(
                "MeshletCullPass produced FG cull buffers: instances={} meshlets={} visibleBuf={} counterBuf={} instanceBuf={}",
                instanceCount,
                totalMeshlets,
                fmt::ptr(visibleMeshletBuffer),
                fmt::ptr(counterBuffer),
                fmt::ptr(instanceDataBuffer));
            sLoggedGpuPublish = true;
        }

        // m_lastVisibleCount is set at the top of this function via 1-frame-delayed readback
    }

    void renderUI() override {
        ImGui::Text("Total Meshlets: %u", m_totalMeshlets);
        ImGui::Text("Visible Meshlets: %u", m_lastVisibleCount);
        if (m_totalMeshlets > 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(m_totalMeshlets);
            ImGui::Text("Cull Rate: %.1f%%", cullRate * 100.0f);
        }
        if (ImGui::Checkbox("Frustum Cull", &m_enableFrustumCull)) {
            syncFrameContextFlags();
        }
        if (ImGui::Checkbox("Cone Cull", &m_enableConeCull)) {
            syncFrameContextFlags();
        }
        ImGui::Checkbox("HZB Occlusion Cull", &m_enableOcclusionCull);
        ImGui::SliderFloat("HZB Depth Bias", &m_occlusionDepthBias, 0.0f, 0.05f, "%.4f");
        ImGui::SliderFloat("HZB Bounds Scale", &m_occlusionBoundsScale, 1.0f, 1.5f, "%.2f");
        if (m_frameContext) {
            ImGui::Text("Instances: %u", m_frameContext->visibilityInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
        const bool historyValid =
            m_frameGraph && !m_hzbHistoryRead.empty() && m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
        ImGui::Text("HZB History: %s (%u levels)", historyValid ? "Ready" : "Warming Up", m_hzbLevelCount);
    }

private:
    void syncFrameContextFlags() {
        if (!m_frameContext) {
            return;
        }
        auto* frameContext = const_cast<FrameContext*>(m_frameContext);
        frameContext->enableFrustumCull = m_enableFrustumCull;
        frameContext->enableConeCull = m_enableConeCull;
    }

    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Meshlet Cull";

    uint32_t m_maxMeshlets = 0;
    uint32_t m_maxInstances = 0;
    uint32_t m_totalMeshlets = 0;
    uint32_t m_lastVisibleCount = 0;
    uint32_t m_hzbLevelCount = 0;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;
    bool m_enableOcclusionCull = true;
    float m_occlusionDepthBias = 0.0015f;
    float m_occlusionBoundsScale = 1.1f;
    const RhiBuffer* m_initializedCounterBuffer = nullptr;
    std::vector<FGResource> m_hzbHistoryRead;

    uint32_t computeMaxMeshletCapacity() const {
        uint64_t totalMeshlets = 0;
        for (const auto& node : m_ctx.sceneGraph.nodes) {
            totalMeshlets += node.meshletCount;
        }
        return static_cast<uint32_t>(std::max<uint64_t>(1u, totalMeshlets));
    }

    uint32_t computeMaxInstanceCapacity() const {
        uint32_t instanceCapacity = 0;
        for (const auto& node : m_ctx.sceneGraph.nodes) {
            if (node.meshletCount > 0) {
                ++instanceCapacity;
            }
        }
        return std::max(1u, instanceCapacity);
    }
};



