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

        m_maxVisibleInstances = std::max<uint32_t>(1u, m_ctx.gpuScene.instanceCount);
        m_maxMeshlets = std::max<uint32_t>(1u, computeMaxMeshletCapacity());

        const auto visibleInstanceWorklist =
            GpuDriven::createTypedIndirectWorklist<VisibleInstanceInfo,
                                                   GpuDriven::ComputeDispatchCommandLayout>(
                builder,
                "instanceData",
                "VisibleInstanceBuffer",
                m_maxVisibleInstances,
                "VisibleInstanceState",
                "VisibleInstanceStateBuffer",
                true);
        instanceData = visibleInstanceWorklist.payload;
        m_visibleInstanceState = visibleInstanceWorklist.state;

        const auto visibleMeshletWorklist =
            GpuDriven::createTypedIndirectWorklist<MeshletDrawInfo,
                                                   GpuDriven::MeshDispatchCommandLayout>(
                builder,
                "visibleMeshlets",
                "VisibleMeshletBuffer",
                m_maxMeshlets,
                "cullCounter",
                "VisibilityWorklistStateBuffer",
                true);
        visibleMeshlets = visibleMeshletWorklist.payload;
        cullCounter = visibleMeshletWorklist.state;

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

        auto classifyIt = m_runtimeContext->computePipelinesRhi.find("InstanceClassifyPass");
        auto cullIt = m_runtimeContext->computePipelinesRhi.find("MeshletCullPass");
        auto buildIt = m_runtimeContext->computePipelinesRhi.find("BuildIndirectPass");
        if (classifyIt == m_runtimeContext->computePipelinesRhi.end() ||
            !classifyIt->second.nativeHandle()) {
            return;
        }
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
        RhiBuffer* visibleInstanceBuffer = m_frameGraph->getBuffer(instanceData);
        RhiBuffer* visibleInstanceStateBuffer = m_frameGraph->getBuffer(m_visibleInstanceState);
        RhiBuffer* visibleMeshletBuffer = m_frameGraph->getBuffer(visibleMeshlets);
        RhiBuffer* worklistStateBuffer = m_frameGraph->getBuffer(cullCounter);
        if (!visibleInstanceBuffer || !visibleInstanceStateBuffer ||
            !visibleMeshletBuffer || !worklistStateBuffer) {
            return;
        }

        GpuDriven::ensureWorklistStateBufferInitialized<GpuDriven::ComputeDispatchCommandLayout>(
            visibleInstanceStateBuffer,
            m_initializedInstanceStateBuffer);
        GpuDriven::ensureWorklistStateBufferInitialized<GpuDriven::MeshDispatchCommandLayout>(
            worklistStateBuffer,
            m_initializedMeshletStateBuffer);

        m_lastVisibleInstanceCount =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::ComputeDispatchCommandLayout>(
                visibleInstanceStateBuffer);
        m_lastVisibleCount =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);

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

        InstanceClassifyUniforms classifyUni{};
        classifyUni.viewProj = transpose(m_frameContext->proj * m_frameContext->view);
        classifyUni.prevViewProj = transpose(m_frameContext->prevCullProj * m_frameContext->prevCullView);
        classifyUni.prevView = transpose(m_frameContext->prevCullView);
        classifyUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        classifyUni.prevCameraWorldPos = m_frameContext->prevCameraWorldPos;
        classifyUni.prevProjScale = float2(std::abs(m_frameContext->prevCullProj[0].x),
                                           std::abs(m_frameContext->prevCullProj[1].y));
        classifyUni.instanceCount = gpuScene.instanceCount;
        classifyUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1u : 0u;
        classifyUni.enableOcclusionCull = hzbTextureCount > 0 ? 1u : 0u;
        classifyUni.hzbLevelCount = hzbTextureCount;
        if (hzbTextureCount > 0) {
            classifyUni.hzbTextureSize =
                float2(static_cast<float>(hzbTextures[0]->width()),
                       static_cast<float>(hzbTextures[0]->height()));
        }
        classifyUni.occlusionDepthBias = m_occlusionDepthBias;
        classifyUni.occlusionBoundsScale = m_occlusionBoundsScale;

        CullUniforms cullUni{};
        cullUni.viewProj = classifyUni.viewProj;
        cullUni.prevViewProj = classifyUni.prevViewProj;
        cullUni.prevView = classifyUni.prevView;
        cullUni.cameraWorldPos = classifyUni.cameraWorldPos;
        cullUni.prevCameraWorldPos = classifyUni.prevCameraWorldPos;
        cullUni.prevProjScale = classifyUni.prevProjScale;
        cullUni.hzbTextureSize = classifyUni.hzbTextureSize;
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1u : 0u;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1u : 0u;
        cullUni.enableOcclusionCull = classifyUni.enableOcclusionCull;
        cullUni.hzbLevelCount = classifyUni.hzbLevelCount;
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;

        // Dispatch 1: coarse instance classification from scene tables.
        encoder.setComputePipeline(classifyIt->second);
        encoder.setBytes(&classifyUni, sizeof(classifyUni), GpuDriven::InstanceClassifyBindings::kUniforms);
        encoder.setBuffer(&gpuScene.instanceBuffer, 0, GpuDriven::InstanceClassifyBindings::kInstances);
        encoder.setBuffer(&gpuScene.geometryBuffer, 0, GpuDriven::InstanceClassifyBindings::kGeometries);
        encoder.setBuffer(visibleInstanceBuffer, 0, GpuDriven::InstanceClassifyBindings::kOutput);
        encoder.setBuffer(visibleInstanceStateBuffer, 0, GpuDriven::InstanceClassifyBindings::kState);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::InstanceClassifyBindings::kHzbTextureBase,
                                hzbTextureCount);
        }
        constexpr uint32_t kClassifyThreadgroupSize = 64u;
        const uint32_t classifyThreadgroups =
            (gpuScene.instanceCount + kClassifyThreadgroupSize - 1u) / kClassifyThreadgroupSize;
        encoder.dispatchThreadgroups({classifyThreadgroups, 1, 1}, {kClassifyThreadgroupSize, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Dispatch 2: publish indirect compute args from visible-instance count.
        encoder.setComputePipeline(buildIt->second);
        encoder.setBuffer(visibleInstanceStateBuffer, 0, GpuDriven::BuildWorklistBindings::kState);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Dispatch 3: expand visible instances into visible meshlets.
        encoder.setComputePipeline(cullIt->second);
        encoder.setBytes(&cullUni, sizeof(cullUni), GpuDriven::MeshletCullBindings::kUniforms);
        encoder.setBuffer(&gpuScene.instanceBuffer, 0, GpuDriven::MeshletCullBindings::kInstances);
        encoder.setBuffer(&gpuScene.geometryBuffer, 0, GpuDriven::MeshletCullBindings::kGeometries);
        encoder.setBuffer(&m_ctx.meshletData.boundsBuffer, 0, GpuDriven::MeshletCullBindings::kBounds);
        encoder.setBuffer(visibleInstanceBuffer, 0, GpuDriven::MeshletCullBindings::kVisibleInstances);
        encoder.setBuffer(visibleMeshletBuffer, 0, GpuDriven::MeshletCullBindings::kCompactionOutput);
        encoder.setBuffer(worklistStateBuffer, 0, GpuDriven::MeshletCullBindings::kCounter);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kHzbTextureBase,
                                hzbTextureCount);
        }
        encoder.dispatchThreadgroupsIndirect(*visibleInstanceStateBuffer,
                                             GpuDriven::ComputeDispatchCommandLayout::kIndirectArgsOffset,
                                             {64, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        // Dispatch 4: publish mesh-dispatch args from the visible meshlet cursor.
        encoder.setComputePipeline(buildIt->second);
        encoder.setBuffer(worklistStateBuffer, 0, GpuDriven::BuildWorklistBindings::kState);
        encoder.dispatchThreadgroups({1, 1, 1}, {1, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

        static bool sLoggedGpuPublish = false;
        if (!sLoggedGpuPublish) {
            spdlog::info(
                "MeshletCullPass front-end ready: sceneInstances={} sceneVisibleFlags={} maxMeshlets={} visibleInstanceBuf={} visibleInstanceState={} visibleMeshletBuf={} meshletState={}",
                gpuScene.instanceCount,
                gpuScene.visibleInstanceCount,
                gpuScene.totalMeshletDispatchCount,
                fmt::ptr(visibleInstanceBuffer),
                fmt::ptr(visibleInstanceStateBuffer),
                fmt::ptr(visibleMeshletBuffer),
                fmt::ptr(worklistStateBuffer));
            sLoggedGpuPublish = true;
        }
    }

    void renderUI() override {
        ImGui::Text("Classified Instances: %u", m_lastVisibleInstanceCount);
        ImGui::Text("Total Meshlets: %u", m_totalMeshlets);
        ImGui::Text("Visible Meshlets: %u", m_lastVisibleCount);
        if (m_ctx.gpuScene.instanceCount > 0) {
            float coarseCullRate =
                1.0f - float(m_lastVisibleInstanceCount) / float(m_ctx.gpuScene.instanceCount);
            ImGui::Text("Instance Cull Rate: %.1f%%", coarseCullRate * 100.0f);
        }
        if (m_totalMeshlets > 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(m_totalMeshlets);
            ImGui::Text("Meshlet Cull Rate: %.1f%%", cullRate * 100.0f);
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
            ImGui::Text("Scene Visible Flags: %u", m_ctx.gpuScene.visibleInstanceCount);
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

    uint32_t m_maxVisibleInstances = 0;
    uint32_t m_maxMeshlets = 0;
    uint32_t m_totalMeshlets = 0;
    uint32_t m_lastVisibleInstanceCount = 0;
    uint32_t m_lastVisibleCount = 0;
    uint32_t m_hzbLevelCount = 0;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;
    bool m_enableOcclusionCull = true;
    float m_occlusionDepthBias = 0.0015f;
    float m_occlusionBoundsScale = 1.1f;
    FGResource m_visibleInstanceState;
    const RhiBuffer* m_initializedInstanceStateBuffer = nullptr;
    const RhiBuffer* m_initializedMeshletStateBuffer = nullptr;
    std::vector<FGResource> m_hzbHistoryRead;

    uint32_t computeMaxMeshletCapacity() const {
        return std::max(1u, m_ctx.gpuScene.totalMeshletDispatchCount);
    }
};
