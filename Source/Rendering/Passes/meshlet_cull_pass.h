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
        if (name == "visibilityWorklistState") return cullCounter;
        if (name == "visibilityIndirectArgs") return cullCounter;
        if (name == "visibilityInstances") return instanceData;
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        cullResult = builder.createToken("cullResult");

        m_maxMeshlets = std::max<uint32_t>(1u, computeMaxMeshletCapacity());

        const auto visibleWorklist =
            GpuDriven::createTypedIndirectWorklist<MeshletDrawInfo,
                                                   GpuDriven::MeshDispatchCommandLayout>(
                builder,
                "visibleMeshlets",
                "VisibleMeshletBuffer",
                m_maxMeshlets,
                "cullCounter",
                "VisibilityWorklistStateBuffer",
                true);
        visibleMeshlets = visibleWorklist.payload;
        cullCounter = visibleWorklist.state;

        // Keep the legacy output slot alive for authored pipeline compatibility.
        instanceData = builder.createToken("instanceData");

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

        const GpuSceneTables& gpuScene = m_ctx.gpuScene;
        if (!gpuScene.instanceBuffer.nativeHandle() ||
            !gpuScene.geometryBuffer.nativeHandle() ||
            gpuScene.instanceCount == 0 ||
            gpuScene.totalMeshletDispatchCount == 0) {
            return;
        }

        m_totalMeshlets = gpuScene.totalMeshletDispatchCount;

        if (!m_frameGraph) return;
        RhiBuffer* visibleMeshletBuffer = m_frameGraph->getBuffer(visibleMeshlets);
        RhiBuffer* worklistStateBuffer = m_frameGraph->getBuffer(cullCounter);
        if (!visibleMeshletBuffer || !worklistStateBuffer) {
            return;
        }
        GpuDriven::ensureWorklistStateBufferInitialized<GpuDriven::MeshDispatchCommandLayout>(
            worklistStateBuffer,
            m_initializedCounterBuffer);

        // Read back the previous frame's published worklist count. The build step copies the
        // producer cursor into the persistent produced-count slot before resetting the cursor.
        m_lastVisibleCount =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);

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
        cullUni.totalDispatchCount = gpuScene.totalMeshletDispatchCount;
        cullUni.instanceCount = gpuScene.instanceCount;
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
        encoder.setBuffer(&gpuScene.instanceBuffer, 0, GpuDriven::MeshletCullBindings::kInstances);
        encoder.setBuffer(&gpuScene.geometryBuffer, 0, GpuDriven::MeshletCullBindings::kGeometries);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletCullBindings::kBounds);
        encoder.setBuffer(visibleMeshletBuffer, 0, GpuDriven::MeshletCullBindings::kCompactionOutput);
        encoder.setBuffer(worklistStateBuffer, 0, GpuDriven::MeshletCullBindings::kCounter);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kHzbTextureBase,
                                hzbTextureCount);
        }

        uint32_t threadgroupSize = 256;
        uint32_t threadgroups = (gpuScene.totalMeshletDispatchCount + threadgroupSize - 1) / threadgroupSize;
        encoder.dispatchThreadgroups({threadgroups, 1, 1}, {threadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // --- Dispatch 2: Publish mesh-dispatch args from the worklist cursor ---
        encoder.setComputePipeline(buildIt->second);
        encoder.setBuffer(worklistStateBuffer, 0, GpuDriven::BuildWorklistBindings::kState);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        static bool sLoggedGpuPublish = false;
        if (!sLoggedGpuPublish) {
            spdlog::info(
                "MeshletCullPass produced FG cull buffers: sceneInstances={} visibleInstances={} meshlets={} visibleBuf={} counterBuf={} sceneInstanceBuf={}",
                gpuScene.instanceCount,
                gpuScene.visibleInstanceCount,
                gpuScene.totalMeshletDispatchCount,
                fmt::ptr(visibleMeshletBuffer),
                fmt::ptr(worklistStateBuffer),
                fmt::ptr(gpuScene.instanceBuffer.nativeHandle()));
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
            ImGui::Text("Scene Instances: %u", m_ctx.gpuScene.instanceCount);
            ImGui::Text("Visible Scene Instances: %u", m_ctx.gpuScene.visibleInstanceCount);
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
        return std::max(1u, m_ctx.gpuScene.totalMeshletDispatchCount);
    }
};



