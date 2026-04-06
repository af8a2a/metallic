#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "frame_context.h"
#include "gpu_driven_helpers.h"
#include "gpu_cull_resources.h"
#include "cluster_lod_builder.h"
#include "hzb_constants.h"
#include "pass_registry.h"
#include "imgui.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
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

        m_clusterTraversalStats = builder.create("ClusterTraversalStats",
                                                 makeTraversalStatsBufferDesc());
        m_dummyLodNodes = builder.create("DummyClusterLodNodes",
                                         makeSingleElementBufferDesc<GPULodNode>("DummyClusterLodNodes"));
        m_dummyLodGroups = builder.create("DummyClusterLodGroups",
                                          makeSingleElementBufferDesc<GPUClusterGroup>("DummyClusterLodGroups"));
        m_dummyLodGroupMeshletIndices =
            builder.create("DummyClusterLodGroupMeshletIndices",
                           makeSingleElementBufferDesc<uint32_t>("DummyClusterLodGroupMeshletIndices"));
        m_dummyLodBounds = builder.create("DummyClusterLodBounds",
                                          makeSingleElementBufferDesc<GPUMeshletBounds>("DummyClusterLodBounds"));
        m_dummyLodNodeResidency =
            builder.create("DummyClusterLodNodeResidency",
                           makeSingleElementBufferDesc<uint32_t>("DummyClusterLodNodeResidency"));
        m_dummyLodGroupPageTable =
            builder.create("DummyClusterLodGroupPageTable",
                           makeSingleValueBufferDesc<uint32_t>(kClusterLodGroupPageInvalidAddress,
                                                               "DummyClusterLodGroupPageTable"));
        m_dummyResidencyRequests =
            builder.create("DummyClusterLodResidencyRequests",
                           makeSingleElementBufferDesc<ClusterResidencyRequest>(
                               "DummyClusterLodResidencyRequests"));
        m_dummyResidencyRequestState =
            builder.create("DummyClusterLodResidencyRequestState",
                           GpuDriven::makeWorklistStateBufferDesc<GpuDriven::ComputeDispatchCommandLayout>(
                               "DummyClusterLodResidencyRequestState",
                               false));

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
        RhiBuffer* clusterTraversalStatsBuffer = m_frameGraph->getBuffer(m_clusterTraversalStats);
        if (!visibleInstanceBuffer || !visibleInstanceStateBuffer ||
            !visibleMeshletBuffer || !worklistStateBuffer || !clusterTraversalStatsBuffer) {
            return;
        }

        m_lastTraversalStats = readTraversalStats(clusterTraversalStatsBuffer);
        if (clusterTraversalStatsBuffer->mappedData()) {
            std::memset(clusterTraversalStatsBuffer->mappedData(), 0, sizeof(ClusterTraversalStats));
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

        const ClusterLODData& clusterLodData = m_ctx.clusterLodData;
        const bool clusterLodAvailable =
            clusterLodData.nodeBuffer.nativeHandle() &&
            clusterLodData.groupBuffer.nativeHandle() &&
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle() &&
            clusterLodData.boundsBuffer.nativeHandle();
        RhiBuffer* dummyLodNodesBuffer = m_frameGraph->getBuffer(m_dummyLodNodes);
        RhiBuffer* dummyLodGroupsBuffer = m_frameGraph->getBuffer(m_dummyLodGroups);
        RhiBuffer* dummyLodGroupMeshletIndicesBuffer =
            m_frameGraph->getBuffer(m_dummyLodGroupMeshletIndices);
        RhiBuffer* dummyLodBoundsBuffer = m_frameGraph->getBuffer(m_dummyLodBounds);
        RhiBuffer* dummyLodNodeResidencyBuffer = m_frameGraph->getBuffer(m_dummyLodNodeResidency);
        RhiBuffer* dummyLodGroupPageTableBuffer = m_frameGraph->getBuffer(m_dummyLodGroupPageTable);
        RhiBuffer* dummyResidencyRequestBuffer = m_frameGraph->getBuffer(m_dummyResidencyRequests);
        RhiBuffer* dummyResidencyRequestStateBuffer =
            m_frameGraph->getBuffer(m_dummyResidencyRequestState);
        const RhiBuffer* lodNodeBuffer =
            clusterLodAvailable ? &clusterLodData.nodeBuffer : dummyLodNodesBuffer;
        const RhiBuffer* lodGroupBuffer =
            clusterLodAvailable ? &clusterLodData.groupBuffer : dummyLodGroupsBuffer;
        const RhiBuffer* lodGroupMeshletIndicesBuffer =
            clusterLodAvailable ? &clusterLodData.groupMeshletIndicesBuffer
                                : dummyLodGroupMeshletIndicesBuffer;
        const RhiBuffer* lodBoundsBuffer =
            clusterLodAvailable ? &clusterLodData.boundsBuffer : dummyLodBoundsBuffer;
        ensureResidencyResources(std::max<uint32_t>(1u, clusterLodData.totalNodeCount),
                                 std::max<uint32_t>(1u, clusterLodData.totalGroupCount));
        processResidencyRequests(clusterLodData);
        const bool residencyStreamingResourcesReady =
            m_lodNodeResidencyBuffer &&
            m_lodGroupPageTableBuffer &&
            m_residencyRequestBuffer &&
            m_residencyRequestStateBuffer;
        const RhiBuffer* lodNodeResidencyBuffer =
            residencyStreamingResourcesReady ? m_lodNodeResidencyBuffer.get()
                                             : dummyLodNodeResidencyBuffer;
        const RhiBuffer* lodGroupPageTableBuffer =
            residencyStreamingResourcesReady ? m_lodGroupPageTableBuffer.get()
                                             : dummyLodGroupPageTableBuffer;
        const RhiBuffer* residencyRequestBuffer =
            residencyStreamingResourcesReady ? m_residencyRequestBuffer.get()
                                             : dummyResidencyRequestBuffer;
        const RhiBuffer* residencyRequestStateBuffer =
            residencyStreamingResourcesReady ? m_residencyRequestStateBuffer.get()
                                             : dummyResidencyRequestStateBuffer;

        std::array<const RhiTexture*, kHzbMaxLevels> hzbTextures{};
        uint32_t hzbTextureCount = 0;
        const bool historyValid =
            m_enableOcclusionCull &&
            m_frameContext &&
            !m_frameContext->historyReset &&
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

        const float4x4 currentCullProj = m_frameContext->unjitteredProj;

        InstanceClassifyUniforms classifyUni{};
        classifyUni.viewProj = transpose(currentCullProj * m_frameContext->view);
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
        cullUni.projScale =
            float2(std::abs(currentCullProj[0].x), std::abs(currentCullProj[1].y));
        cullUni.renderTargetSize = float2(static_cast<float>(m_width), static_cast<float>(m_height));
        cullUni.prevProjScale = classifyUni.prevProjScale;
        cullUni.hzbTextureSize = classifyUni.hzbTextureSize;
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1u : 0u;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1u : 0u;
        cullUni.enableOcclusionCull = classifyUni.enableOcclusionCull;
        cullUni.hzbLevelCount = classifyUni.hzbLevelCount;
        cullUni.lodReferencePixels = m_lodReferencePixels;
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;
        cullUni.clusterLodEnabled = clusterLodAvailable ? 1u : 0u;
        cullUni.enableResidencyStreaming =
            (clusterLodAvailable && residencyStreamingResourcesReady && m_enableResidencyStreaming) ? 1u : 0u;

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
        encoder.setBuffer(lodNodeBuffer, 0, GpuDriven::MeshletCullBindings::kLodNodes);
        encoder.setBuffer(lodGroupBuffer, 0, GpuDriven::MeshletCullBindings::kLodGroups);
        encoder.setBuffer(lodGroupMeshletIndicesBuffer, 0, GpuDriven::MeshletCullBindings::kLodGroupMeshletIndices);
        encoder.setBuffer(lodBoundsBuffer, 0, GpuDriven::MeshletCullBindings::kLodBounds);
        encoder.setBuffer(clusterTraversalStatsBuffer, 0, GpuDriven::MeshletCullBindings::kTraversalStats);
        encoder.setBuffer(lodNodeResidencyBuffer, 0, GpuDriven::MeshletCullBindings::kLodNodeResidency);
        encoder.setBuffer(lodGroupPageTableBuffer, 0, GpuDriven::MeshletCullBindings::kLodGroupPageTable);
        encoder.setBuffer(residencyRequestBuffer, 0, GpuDriven::MeshletCullBindings::kResidencyRequests);
        encoder.setBuffer(residencyRequestStateBuffer, 0, GpuDriven::MeshletCullBindings::kResidencyRequestState);
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
        ImGui::Text("LOD Traversal Instances: %u", m_lastTraversalStats.lodTraversalInstanceCount);
        ImGui::Text("Fallback Instances: %u", m_lastTraversalStats.fallbackInstanceCount);
        ImGui::Text("Traversed Nodes: %u", m_lastTraversalStats.traversedNodeCount);
        ImGui::Text("HZB-Culled Nodes: %u", m_lastTraversalStats.occludedNodeCount);
        ImGui::Text("Candidate Groups: %u", m_lastTraversalStats.candidateGroupCount);
        ImGui::Text("Selected Groups: %u", m_lastTraversalStats.selectedGroupCount);
        ImGui::Text("HZB-Culled Groups: %u", m_lastTraversalStats.occludedGroupCount);
        ImGui::Text("Candidate LOD Meshlets: %u", m_lastTraversalStats.candidateClusterMeshletCount);
        ImGui::Text("LOD Meshlets: %u", m_lastTraversalStats.emittedClusterMeshletCount);
        ImGui::Text("Candidate Fallback Meshlets: %u", m_lastTraversalStats.candidateFallbackMeshletCount);
        ImGui::Text("Fallback Meshlets: %u", m_lastTraversalStats.emittedFallbackMeshletCount);
        ImGui::Text("Max Selected LOD: %u", m_lastTraversalStats.maxSelectedLodLevel);
        if (m_ctx.gpuScene.instanceCount > 0) {
            float coarseCullRate =
                1.0f - float(m_lastVisibleInstanceCount) / float(m_ctx.gpuScene.instanceCount);
            ImGui::Text("Instance Cull Rate: %.1f%%", coarseCullRate * 100.0f);
        }
        if (m_totalMeshlets > 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(m_totalMeshlets);
            ImGui::Text("Meshlet Cull Rate: %.1f%%", cullRate * 100.0f);
        }
        if (m_lastTraversalStats.traversedNodeCount > 0) {
            float nodeRejectRate =
                float(m_lastTraversalStats.occludedNodeCount) /
                float(m_lastTraversalStats.traversedNodeCount);
            ImGui::Text("Node HZB Reject Rate: %.1f%%", nodeRejectRate * 100.0f);
        }
        if (m_lastTraversalStats.candidateGroupCount > 0) {
            float groupKeepRate =
                float(m_lastTraversalStats.selectedGroupCount) /
                float(m_lastTraversalStats.candidateGroupCount);
            ImGui::Text("Group Keep Rate: %.1f%%", groupKeepRate * 100.0f);
        }
        if (m_lastTraversalStats.candidateClusterMeshletCount > 0) {
            float clusterMeshletKeepRate =
                float(m_lastTraversalStats.emittedClusterMeshletCount) /
                float(m_lastTraversalStats.candidateClusterMeshletCount);
            ImGui::Text("LOD Meshlet Keep Rate: %.1f%%", clusterMeshletKeepRate * 100.0f);
        }
        if (m_lastTraversalStats.candidateFallbackMeshletCount > 0) {
            float fallbackMeshletKeepRate =
                float(m_lastTraversalStats.emittedFallbackMeshletCount) /
                float(m_lastTraversalStats.candidateFallbackMeshletCount);
            ImGui::Text("Fallback Meshlet Keep Rate: %.1f%%", fallbackMeshletKeepRate * 100.0f);
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
        ImGui::SliderFloat("LOD Reference Pixels", &m_lodReferencePixels, 8.0f, 256.0f, "%.1f");
        if (ImGui::Checkbox("Virtual Residency Streaming", &m_enableResidencyStreaming)) {
            m_residencyStateDirty = true;
            requestVisibilityHistoryReset();
        }
        int streamingBudgetNodes = static_cast<int>(m_streamingBudgetNodes);
        const int maxStreamingBudget =
            static_cast<int>(std::max<uint32_t>(m_lastAlwaysResidentNodeCount + 1u,
                                                std::max<uint32_t>(m_activeResidencyNodeCount, 1u)));
        if (ImGui::SliderInt("Streaming Budget (LOD Nodes)",
                             &streamingBudgetNodes,
                             0,
                             maxStreamingBudget,
                             "%d")) {
            m_streamingBudgetNodes = static_cast<uint32_t>(std::max(streamingBudgetNodes, 0));
            requestVisibilityHistoryReset();
        }
        if (ImGui::Button("Reset Residency State")) {
            m_residencyStateDirty = true;
            requestVisibilityHistoryReset();
        }
        if (m_frameContext) {
            ImGui::Text("Scene Instances: %u", m_ctx.gpuScene.instanceCount);
            ImGui::Text("Scene Visible Flags: %u", m_ctx.gpuScene.visibleInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
        ImGui::Text("Residency Nodes: %u active / %u resident",
                    m_activeResidencyNodeCount,
                    m_lastResidentNodeCount);
        ImGui::Text("Always Resident Nodes: %u", m_lastAlwaysResidentNodeCount);
        ImGui::Text("Resident Groups: %u / %u",
                    m_lastResidentGroupCount,
                    m_activeResidencyGroupCount);
        ImGui::Text("Always Resident Groups: %u", m_lastAlwaysResidentGroupCount);
        ImGui::Text("Dynamic Resident Nodes: %zu", m_dynamicResidentNodes.size());
        ImGui::Text("Pending Residency Requests: %zu", m_pendingResidencyNodes.size());
        ImGui::Text("GPU Residency Requests (last frame): %u", m_lastResidencyRequestCount);
        ImGui::Text("Promoted / Evicted (last frame): %u / %u",
                    m_lastResidencyPromotedCount,
                    m_lastResidencyEvictedCount);
        const bool historyValid =
            m_frameGraph && !m_hzbHistoryRead.empty() && m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
        ImGui::Text("HZB History: %s (%u levels)", historyValid ? "Ready" : "Warming Up", m_hzbLevelCount);
        for (uint32_t level = 0; level < kClusterTraversalStatsHistogramSize; ++level) {
            ImGui::Text("LOD %u Hits: %u", level, m_lastTraversalStats.selectedLodLevelHistogram[level]);
        }
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

    void requestVisibilityHistoryReset() {
        if (!m_frameContext) {
            return;
        }
        auto* frameContext = const_cast<FrameContext*>(m_frameContext);
        frameContext->historyReset = true;
    }

    void ensureResidencyResources(uint32_t nodeCapacity, uint32_t groupCapacity) {
        if (!m_runtimeContext || !m_runtimeContext->resourceFactory) {
            return;
        }

        nodeCapacity = std::max(1u, nodeCapacity);
        groupCapacity = std::max(1u, groupCapacity);
        const bool needsRecreate =
            !m_lodNodeResidencyBuffer ||
            !m_lodGroupPageTableBuffer ||
            !m_residencyRequestBuffer ||
            !m_residencyRequestStateBuffer ||
            m_residencyNodeCapacity != nodeCapacity ||
            m_residencyGroupCapacity != groupCapacity;
        if (!needsRecreate) {
            return;
        }

        RhiBufferDesc residencyDesc{};
        residencyDesc.size = size_t(nodeCapacity) * sizeof(uint32_t);
        residencyDesc.hostVisible = true;
        residencyDesc.debugName = "ClusterLodNodeResidency";
        m_lodNodeResidencyBuffer = m_runtimeContext->resourceFactory->createBuffer(residencyDesc);

        RhiBufferDesc groupPageTableDesc{};
        groupPageTableDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        groupPageTableDesc.hostVisible = true;
        groupPageTableDesc.debugName = "ClusterLodGroupPageTable";
        m_lodGroupPageTableBuffer = m_runtimeContext->resourceFactory->createBuffer(groupPageTableDesc);

        RhiBufferDesc requestDesc{};
        requestDesc.size = size_t(nodeCapacity) * sizeof(ClusterResidencyRequest);
        requestDesc.hostVisible = true;
        requestDesc.debugName = "ClusterLodResidencyRequests";
        m_residencyRequestBuffer = m_runtimeContext->resourceFactory->createBuffer(requestDesc);

        RhiBufferDesc requestStateDesc{};
        requestStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        requestStateDesc.hostVisible = true;
        requestStateDesc.debugName = "ClusterLodResidencyRequestState";
        m_residencyRequestStateBuffer = m_runtimeContext->resourceFactory->createBuffer(requestStateDesc);

        m_residencyNodeCapacity = nodeCapacity;
        m_residencyGroupCapacity = groupCapacity;
        m_residencyStateDirty = true;
        m_dynamicResidentNodes.clear();
        m_pendingResidencyNodes.clear();
        m_residencyNodeLeafGroups.clear();
        m_residencySourceNodeBufferHandle = nullptr;

        if (m_lodNodeResidencyBuffer && m_lodNodeResidencyBuffer->mappedData()) {
            std::memset(m_lodNodeResidencyBuffer->mappedData(), 0, m_lodNodeResidencyBuffer->size());
        }
        invalidateGroupPageTable();
        if (m_residencyRequestBuffer && m_residencyRequestBuffer->mappedData()) {
            std::memset(m_residencyRequestBuffer->mappedData(), 0, m_residencyRequestBuffer->size());
        }
        seedResidencyRequestQueue();
        updateResidencyDebugCounts();
    }

    void invalidateGroupPageTable() {
        uint32_t* pageTable = groupPageTableWords();
        if (!pageTable) {
            return;
        }

        std::fill(pageTable,
                  pageTable + m_residencyGroupCapacity,
                  kClusterLodGroupPageInvalidAddress);
    }

    void seedResidencyRequestQueue() {
        if (!m_residencyRequestStateBuffer) {
            return;
        }
        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            m_residencyRequestStateBuffer.get());
    }

    uint32_t* residencyStateWords() {
        if (!m_lodNodeResidencyBuffer || !m_lodNodeResidencyBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(m_lodNodeResidencyBuffer->mappedData());
    }

    const uint32_t* residencyStateWords() const {
        return const_cast<MeshletCullPass*>(this)->residencyStateWords();
    }

    uint32_t* groupPageTableWords() {
        if (!m_lodGroupPageTableBuffer || !m_lodGroupPageTableBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(m_lodGroupPageTableBuffer->mappedData());
    }

    const uint32_t* groupPageTableWords() const {
        return const_cast<MeshletCullPass*>(this)->groupPageTableWords();
    }

    ClusterResidencyRequest* residencyRequests() {
        if (!m_residencyRequestBuffer || !m_residencyRequestBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<ClusterResidencyRequest*>(m_residencyRequestBuffer->mappedData());
    }

    bool isResidencyNodeResident(uint32_t nodeIndex) const {
        const uint32_t* words = residencyStateWords();
        return words &&
               nodeIndex < m_residencyNodeCapacity &&
               (words[nodeIndex] & kClusterLodNodeResidencyResident) != 0u;
    }

    bool isResidencyNodeAlwaysResident(uint32_t nodeIndex) const {
        const uint32_t* words = residencyStateWords();
        return words &&
               nodeIndex < m_residencyNodeCapacity &&
               (words[nodeIndex] & kClusterLodNodeResidencyAlwaysResident) != 0u;
    }

    void buildResidencyNodeLeafGroupsRecursive(uint32_t nodeIndex,
                                               const ClusterLODData& clusterLodData,
                                               std::vector<uint8_t>& builtNodes) {
        if (nodeIndex >= m_residencyNodeLeafGroups.size() || builtNodes[nodeIndex] != 0u) {
            return;
        }

        builtNodes[nodeIndex] = 1u;
        const GPULodNode& node = clusterLodData.nodes[nodeIndex];
        auto& leafGroups = m_residencyNodeLeafGroups[nodeIndex];
        if (node.isLeaf != 0u) {
            leafGroups.reserve(node.childCount);
            for (uint32_t childIndex = 0; childIndex < node.childCount; ++childIndex) {
                leafGroups.push_back(node.childOffset + childIndex);
            }
            return;
        }

        for (uint32_t childIndex = 0; childIndex < node.childCount; ++childIndex) {
            const uint32_t childNodeIndex = node.childOffset + childIndex;
            if (childNodeIndex >= clusterLodData.nodes.size()) {
                continue;
            }

            buildResidencyNodeLeafGroupsRecursive(childNodeIndex, clusterLodData, builtNodes);
            const auto& childLeafGroups = m_residencyNodeLeafGroups[childNodeIndex];
            leafGroups.insert(leafGroups.end(), childLeafGroups.begin(), childLeafGroups.end());
        }
    }

    void rebuildResidencyNodeLeafGroups(const ClusterLODData& clusterLodData) {
        m_residencyNodeLeafGroups.clear();
        m_residencyNodeLeafGroups.resize(clusterLodData.totalNodeCount);
        std::vector<uint8_t> builtNodes(clusterLodData.totalNodeCount, 0u);
        for (uint32_t nodeIndex = 0; nodeIndex < clusterLodData.totalNodeCount; ++nodeIndex) {
            buildResidencyNodeLeafGroupsRecursive(nodeIndex, clusterLodData, builtNodes);
        }
    }

    void patchResidentGroupsForNode(uint32_t nodeIndex,
                                    const ClusterLODData& clusterLodData,
                                    bool resident) {
        uint32_t* pageTable = groupPageTableWords();
        if (!pageTable || nodeIndex >= m_residencyNodeLeafGroups.size()) {
            return;
        }

        for (uint32_t groupIndex : m_residencyNodeLeafGroups[nodeIndex]) {
            if (groupIndex >= m_residencyGroupCapacity ||
                groupIndex >= clusterLodData.groups.size()) {
                continue;
            }

            pageTable[groupIndex] = resident
                ? clusterLodData.groups[groupIndex].clusterStart
                : kClusterLodGroupPageInvalidAddress;
        }
    }

    void touchDynamicResidentNode(uint32_t nodeIndex) {
        auto it = std::find(m_dynamicResidentNodes.begin(), m_dynamicResidentNodes.end(), nodeIndex);
        if (it != m_dynamicResidentNodes.end()) {
            m_dynamicResidentNodes.erase(it);
        }
        m_dynamicResidentNodes.push_back(nodeIndex);
    }

    void enqueuePendingResidencyNode(uint32_t nodeIndex) {
        if (std::find(m_pendingResidencyNodes.begin(), m_pendingResidencyNodes.end(), nodeIndex) ==
            m_pendingResidencyNodes.end()) {
            m_pendingResidencyNodes.push_back(nodeIndex);
        }
    }

    void evictOldestDynamicResidencyNode(const ClusterLODData& clusterLodData) {
        if (m_dynamicResidentNodes.empty()) {
            return;
        }

        uint32_t* words = residencyStateWords();
        if (!words) {
            m_dynamicResidentNodes.clear();
            return;
        }

        const uint32_t nodeIndex = m_dynamicResidentNodes.front();
        m_dynamicResidentNodes.erase(m_dynamicResidentNodes.begin());
        if (nodeIndex < m_residencyNodeCapacity) {
            words[nodeIndex] &= ~(kClusterLodNodeResidencyResident |
                                  kClusterLodNodeResidencyRequested);
        }
        patchResidentGroupsForNode(nodeIndex, clusterLodData, false);
        ++m_lastResidencyEvictedCount;
    }

    void promotePendingResidencyNodes(const ClusterLODData& clusterLodData) {
        uint32_t* words = residencyStateWords();
        if (!words) {
            return;
        }

        size_t pendingIndex = 0;
        while (pendingIndex < m_pendingResidencyNodes.size()) {
            const uint32_t nodeIndex = m_pendingResidencyNodes[pendingIndex];
            if (nodeIndex >= m_residencyNodeCapacity) {
                m_pendingResidencyNodes.erase(m_pendingResidencyNodes.begin() +
                                              static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if ((words[nodeIndex] & kClusterLodNodeResidencyResident) != 0u) {
                words[nodeIndex] &= ~kClusterLodNodeResidencyRequested;
                if ((words[nodeIndex] & kClusterLodNodeResidencyAlwaysResident) == 0u) {
                    touchDynamicResidentNode(nodeIndex);
                }
                m_pendingResidencyNodes.erase(m_pendingResidencyNodes.begin() +
                                              static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (m_streamingBudgetNodes == 0u) {
                break;
            }

            while (m_dynamicResidentNodes.size() >= size_t(m_streamingBudgetNodes) &&
                   !m_dynamicResidentNodes.empty()) {
                evictOldestDynamicResidencyNode(clusterLodData);
            }
            if (m_dynamicResidentNodes.size() >= size_t(m_streamingBudgetNodes)) {
                break;
            }

            words[nodeIndex] |= kClusterLodNodeResidencyResident;
            words[nodeIndex] &= ~kClusterLodNodeResidencyRequested;
            patchResidentGroupsForNode(nodeIndex, clusterLodData, true);
            touchDynamicResidentNode(nodeIndex);
            ++m_lastResidencyPromotedCount;
            m_pendingResidencyNodes.erase(m_pendingResidencyNodes.begin() +
                                          static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
        }
    }

    void rebuildResidencyState(const ClusterLODData& clusterLodData) {
        uint32_t* words = residencyStateWords();
        if (!words) {
            return;
        }

        std::memset(words, 0, m_lodNodeResidencyBuffer->size());
        invalidateGroupPageTable();
        rebuildResidencyNodeLeafGroups(clusterLodData);
        m_dynamicResidentNodes.clear();
        m_pendingResidencyNodes.clear();
        m_lastResidencyPromotedCount = 0;
        m_lastResidencyEvictedCount = 0;

        for (uint32_t lodRootNode : clusterLodData.primitiveGroupLodRoots) {
            if (lodRootNode == UINT32_MAX || lodRootNode >= clusterLodData.nodes.size()) {
                continue;
            }

            const GPULodNode& lodRoot = clusterLodData.nodes[lodRootNode];
            if (lodRoot.childCount == 0u) {
                continue;
            }

            uint32_t alwaysResidentNode = lodRootNode;
            if (lodRoot.isLeaf == 0u) {
                alwaysResidentNode = lodRoot.childOffset + lodRoot.childCount - 1u;
            }
            if (alwaysResidentNode >= m_residencyNodeCapacity) {
                continue;
            }

            words[alwaysResidentNode] =
                kClusterLodNodeResidencyResident | kClusterLodNodeResidencyAlwaysResident;
            patchResidentGroupsForNode(alwaysResidentNode, clusterLodData, true);
        }

        seedResidencyRequestQueue();
        m_residencySourceNodeBufferHandle = clusterLodData.nodeBuffer.nativeHandle();
        m_residencyStateDirty = false;
        updateResidencyDebugCounts();
    }

    void updateResidencyDebugCounts() {
        const uint32_t* words = residencyStateWords();
        const uint32_t* pageTable = groupPageTableWords();
        m_lastResidentNodeCount = 0;
        m_lastAlwaysResidentNodeCount = 0;
        m_lastResidentGroupCount = 0;
        m_lastAlwaysResidentGroupCount = 0;
        if (!words) {
            m_activeResidencyNodeCount = 0;
            m_activeResidencyGroupCount = 0;
            return;
        }

        for (uint32_t nodeIndex = 0; nodeIndex < m_activeResidencyNodeCount; ++nodeIndex) {
            const uint32_t state = words[nodeIndex];
            if ((state & kClusterLodNodeResidencyResident) != 0u) {
                ++m_lastResidentNodeCount;
            }
            if ((state & kClusterLodNodeResidencyAlwaysResident) != 0u) {
                ++m_lastAlwaysResidentNodeCount;
            }
        }

        if (!pageTable) {
            return;
        }

        for (uint32_t groupIndex = 0; groupIndex < m_activeResidencyGroupCount; ++groupIndex) {
            if (pageTable[groupIndex] != kClusterLodGroupPageInvalidAddress) {
                ++m_lastResidentGroupCount;
            }
        }

        for (uint32_t nodeIndex = 0; nodeIndex < m_activeResidencyNodeCount; ++nodeIndex) {
            if (!isResidencyNodeAlwaysResident(nodeIndex) ||
                nodeIndex >= m_residencyNodeLeafGroups.size()) {
                continue;
            }

            m_lastAlwaysResidentGroupCount +=
                static_cast<uint32_t>(m_residencyNodeLeafGroups[nodeIndex].size());
        }
    }

    void processResidencyRequests(const ClusterLODData& clusterLodData) {
        m_activeResidencyNodeCount = clusterLodData.totalNodeCount;
        m_activeResidencyGroupCount = clusterLodData.totalGroupCount;
        if (!m_lodNodeResidencyBuffer ||
            !m_lodGroupPageTableBuffer ||
            !m_residencyRequestBuffer ||
            !m_residencyRequestStateBuffer) {
            return;
        }

        const bool sourceBufferChanged =
            m_residencySourceNodeBufferHandle != clusterLodData.nodeBuffer.nativeHandle();
        if (m_residencyStateDirty || sourceBufferChanged) {
            requestVisibilityHistoryReset();
        }

        if (m_residencyStateDirty || sourceBufferChanged) {
            rebuildResidencyState(clusterLodData);
        }

        m_lastResidencyRequestCount = 0;
        m_lastResidencyPromotedCount = 0;
        m_lastResidencyEvictedCount = 0;

        const uint32_t requestCapacity = static_cast<uint32_t>(
            m_residencyRequestBuffer->size() / sizeof(ClusterResidencyRequest));
        m_lastResidencyRequestCount = std::min<uint32_t>(
            GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                m_residencyRequestStateBuffer.get()),
            requestCapacity);

        ClusterResidencyRequest* requests = residencyRequests();
        if (requests) {
            for (uint32_t requestIndex = 0; requestIndex < m_lastResidencyRequestCount; ++requestIndex) {
                const ClusterResidencyRequest& request = requests[requestIndex];
                if (request.targetNodeIndex >= m_activeResidencyNodeCount) {
                    continue;
                }
                if (isResidencyNodeAlwaysResident(request.targetNodeIndex)) {
                    continue;
                }
                if (isResidencyNodeResident(request.targetNodeIndex)) {
                    touchDynamicResidentNode(request.targetNodeIndex);
                    continue;
                }
                enqueuePendingResidencyNode(request.targetNodeIndex);
            }
        }

        if (m_enableResidencyStreaming) {
            while (m_dynamicResidentNodes.size() > size_t(m_streamingBudgetNodes)) {
                evictOldestDynamicResidencyNode(clusterLodData);
            }
            promotePendingResidencyNodes(clusterLodData);
        }

        updateResidencyDebugCounts();
        seedResidencyRequestQueue();
    }

    const RenderContext& m_ctx;
    int m_width, m_height;
    std::string m_name = "Meshlet Cull";

    uint32_t m_maxVisibleInstances = 0;
    uint32_t m_maxMeshlets = 0;
    uint32_t m_totalMeshlets = 0;
    uint32_t m_lastVisibleInstanceCount = 0;
    uint32_t m_lastVisibleCount = 0;
    ClusterTraversalStats m_lastTraversalStats{};
    uint32_t m_hzbLevelCount = 0;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;
    bool m_enableOcclusionCull = true;
    bool m_enableResidencyStreaming = false;
    bool m_residencyStateDirty = true;
    float m_lodReferencePixels = 96.0f;
    float m_occlusionDepthBias = 0.0015f;
    float m_occlusionBoundsScale = 1.1f;
    uint32_t m_streamingBudgetNodes = 64u;
    uint32_t m_residencyNodeCapacity = 0;
    uint32_t m_residencyGroupCapacity = 0;
    uint32_t m_activeResidencyNodeCount = 0;
    uint32_t m_activeResidencyGroupCount = 0;
    uint32_t m_lastResidencyRequestCount = 0;
    uint32_t m_lastResidencyPromotedCount = 0;
    uint32_t m_lastResidencyEvictedCount = 0;
    uint32_t m_lastResidentNodeCount = 0;
    uint32_t m_lastAlwaysResidentNodeCount = 0;
    uint32_t m_lastResidentGroupCount = 0;
    uint32_t m_lastAlwaysResidentGroupCount = 0;
    FGResource m_visibleInstanceState;
    FGResource m_clusterTraversalStats;
    FGResource m_dummyLodNodes;
    FGResource m_dummyLodGroups;
    FGResource m_dummyLodGroupMeshletIndices;
    FGResource m_dummyLodBounds;
    FGResource m_dummyLodNodeResidency;
    FGResource m_dummyLodGroupPageTable;
    FGResource m_dummyResidencyRequests;
    FGResource m_dummyResidencyRequestState;
    std::unique_ptr<RhiBuffer> m_lodNodeResidencyBuffer;
    std::unique_ptr<RhiBuffer> m_lodGroupPageTableBuffer;
    std::unique_ptr<RhiBuffer> m_residencyRequestBuffer;
    std::unique_ptr<RhiBuffer> m_residencyRequestStateBuffer;
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const RhiBuffer* m_initializedInstanceStateBuffer = nullptr;
    const RhiBuffer* m_initializedMeshletStateBuffer = nullptr;
    std::vector<FGResource> m_hzbHistoryRead;
    std::vector<uint32_t> m_dynamicResidentNodes;
    std::vector<uint32_t> m_pendingResidencyNodes;
    std::vector<std::vector<uint32_t>> m_residencyNodeLeafGroups;

    uint32_t computeMaxMeshletCapacity() const {
        return std::max(1u, m_ctx.gpuScene.totalMeshletDispatchCount);
    }

    template <typename T>
    FGBufferDesc makeSingleElementBufferDesc(const char* debugName) const {
        static const T kZero{};
        FGBufferDesc desc;
        desc.size = sizeof(T);
        desc.initialData = &kZero;
        desc.hostVisible = false;
        desc.debugName = debugName;
        return desc;
    }

    template <typename T>
    FGBufferDesc makeSingleValueBufferDesc(T value, const char* debugName) const {
        static const T kValue = value;
        FGBufferDesc desc;
        desc.size = sizeof(T);
        desc.initialData = &kValue;
        desc.hostVisible = false;
        desc.debugName = debugName;
        return desc;
    }

    FGBufferDesc makeTraversalStatsBufferDesc() const {
        static const ClusterTraversalStats kZeroStats{};
        FGBufferDesc desc;
        desc.size = sizeof(ClusterTraversalStats);
        desc.initialData = &kZeroStats;
        desc.hostVisible = true;
        desc.debugName = "ClusterTraversalStats";
        return desc;
    }

    ClusterTraversalStats readTraversalStats(RhiBuffer* buffer) const {
        ClusterTraversalStats stats{};
        if (!buffer || !buffer->mappedData() || buffer->size() < sizeof(ClusterTraversalStats)) {
            return stats;
        }

        std::memcpy(&stats, buffer->mappedData(), sizeof(stats));
        return stats;
    }
};
