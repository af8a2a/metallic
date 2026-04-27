#pragma once

#include "render_pass.h"
#include "render_uniforms.h"
#include "cluster_streaming_service.h"
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

    METALLIC_PASS_TYPE_INFO(MeshletCullPass, "Meshlet Cull", "Geometry",
        (std::vector<PassSlotInfo>{
            makeInputSlot("streamingSync", "Streaming Sync", true),
            makeInputSlot("visibleMeshletsInput", "Visible Meshlets Input", true),
            makeInputSlot("visibilityWorklistInput", "Visibility Worklist Input", true),
            makeInputSlot("visibilityWorklistStateInput", "Visibility Worklist State Input", true),
            makeInputSlot("cullCounterInput", "Cull Counter Input", true),
            makeInputSlot("currentHzb0", "Current HZB 0", true),
            makeInputSlot("currentHzb1", "Current HZB 1", true),
            makeInputSlot("currentHzb2", "Current HZB 2", true),
            makeInputSlot("currentHzb3", "Current HZB 3", true),
            makeInputSlot("currentHzb4", "Current HZB 4", true),
            makeInputSlot("currentHzb5", "Current HZB 5", true),
            makeInputSlot("currentHzb6", "Current HZB 6", true),
            makeInputSlot("currentHzb7", "Current HZB 7", true),
            makeInputSlot("currentHzb8", "Current HZB 8", true),
            makeInputSlot("currentHzb9", "Current HZB 9", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("cullResult", "Cull Result", true),
            makeOutputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeOutputSlot("cullCounter", "Cull Counter", true),
            makeOutputSlot("instanceData", "Instance Data", true),
            makeOutputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeOutputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeOutputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
            makeOutputSlot("visibilityInstances", "Visibility Instances", true)
        }),
        PassTypeInfo::PassType::Compute);

    METALLIC_PASS_EDITOR_TYPE_INFO(MeshletCullPass, "Meshlet Cull", "Geometry",
        (std::vector<PassSlotInfo>{
            makeHiddenInputSlot("streamingSync", "Streaming Sync", true),
            makeHiddenInputSlot("visibleMeshletsInput", "Visible Meshlets Input", true),
            makeHiddenInputSlot("visibilityWorklistInput", "Visibility Worklist Input", true),
            makeHiddenInputSlot("visibilityWorklistStateInput", "Visibility Worklist State Input", true),
            makeHiddenInputSlot("cullCounterInput", "Cull Counter Input", true),
            makeHiddenInputSlot("currentHzb0", "Current HZB 0", true),
            makeHiddenInputSlot("currentHzb1", "Current HZB 1", true),
            makeHiddenInputSlot("currentHzb2", "Current HZB 2", true),
            makeHiddenInputSlot("currentHzb3", "Current HZB 3", true),
            makeHiddenInputSlot("currentHzb4", "Current HZB 4", true),
            makeHiddenInputSlot("currentHzb5", "Current HZB 5", true),
            makeHiddenInputSlot("currentHzb6", "Current HZB 6", true),
            makeHiddenInputSlot("currentHzb7", "Current HZB 7", true),
            makeHiddenInputSlot("currentHzb8", "Current HZB 8", true),
            makeHiddenInputSlot("currentHzb9", "Current HZB 9", true)
        }),
        (std::vector<PassSlotInfo>{
            makeOutputSlot("cullResult", "Cull Result"),
            makeHiddenOutputSlot("visibleMeshlets", "Visible Meshlets", true),
            makeHiddenOutputSlot("cullCounter", "Cull Counter", true),
            makeHiddenOutputSlot("instanceData", "Instance Data", true),
            makeHiddenOutputSlot("visibilityWorklist", "Visibility Worklist", true),
            makeHiddenOutputSlot("visibilityWorklistState", "Visibility Worklist State", true),
            makeHiddenOutputSlot("visibilityIndirectArgs", "Visibility Indirect Args", true),
            makeHiddenOutputSlot("visibilityInstances", "Visibility Instances", true)
        }),
        PassTypeInfo::PassType::Compute);

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void setFrameContext(const FrameContext* ctx) override {
        RenderPass::setFrameContext(ctx);
        syncFrameContextFlags();
    }

    void configure(const PassConfig& config) override {
        m_name = config.name;
        if (config.config.is_object()) {
            if (config.config.contains("cullPassIndex")) {
                m_cullPassIndex = config.config["cullPassIndex"].get<uint32_t>();
            } else if (config.config.contains("phase")) {
                const std::string phase = config.config["phase"].get<std::string>();
                m_cullPassIndex = (phase == "second" || phase == "late") ? 1u : 0u;
            }
            if (config.config.contains("enableOcclusionCull")) {
                m_enableOcclusionCull = config.config["enableOcclusionCull"].get<bool>();
            }
            if (config.config.contains("directCurrentHzbCull")) {
                m_occlusionCullMode =
                    config.config["directCurrentHzbCull"].get<bool>() ? 1u : 0u;
            }
            if (config.config.contains("occlusionMode")) {
                const std::string mode = config.config["occlusionMode"].get<std::string>();
                if (mode == "current" || mode == "currentDirect" || mode == "directCurrent") {
                    m_occlusionCullMode = 1u;
                } else {
                    m_occlusionCullMode = 0u;
                }
            }
            if (config.config.contains("occlusionDepthBias")) {
                m_occlusionDepthBias = config.config["occlusionDepthBias"].get<float>();
            }
            if (config.config.contains("occlusionBoundsScale")) {
                m_occlusionBoundsScale = config.config["occlusionBoundsScale"].get<float>();
            }
        }
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
        FGResource streamingSyncInput = getInput("streamingSync");
        if (streamingSyncInput.isValid()) {
            builder.read(streamingSyncInput);
        }

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

        FGResource visibleMeshletsInput = getInput("visibleMeshletsInput");
        if (!visibleMeshletsInput.isValid()) {
            visibleMeshletsInput = getInput("visibilityWorklistInput");
        }
        FGResource cullCounterInput = getInput("visibilityWorklistStateInput");
        if (!cullCounterInput.isValid()) {
            cullCounterInput = getInput("cullCounterInput");
        }
        m_appendToExistingWorklist = visibleMeshletsInput.isValid() && cullCounterInput.isValid();
        if (m_appendToExistingWorklist) {
            visibleMeshlets = builder.write(visibleMeshletsInput,
                                            FGResourceUsage::StorageRead |
                                                FGResourceUsage::StorageWrite);
            cullCounter = builder.write(cullCounterInput,
                                        FGResourceUsage::StorageRead |
                                            FGResourceUsage::StorageWrite);
        } else {
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
        }

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
        m_dummyGroupResidency =
            builder.create("DummyClusterLodGroupResidency",
                           makeSingleElementBufferDesc<uint32_t>("DummyClusterLodGroupResidency"));
        m_dummyLodGroupPageTable =
            builder.create("DummyClusterLodGroupPageTable",
                           makeSingleValueBufferDesc<uint64_t>(makeClusterLodGroupPageInvalidAddress(),
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
        m_dummyGroupAge =
            builder.create("DummyClusterLodGroupAge",
                           makeSingleElementBufferDesc<uint32_t>("DummyClusterLodGroupAge"));

        m_hzbHistoryRead.clear();
        m_currentHzbRead.clear();
        m_hzbLevelCount = kHzbMaxLevels;
        m_hzbHistoryRead.reserve(m_hzbLevelCount);
        for (uint32_t level = 0; level < m_hzbLevelCount; ++level) {
            const std::string resourceName = hzbHistoryResourceName(level);
            m_hzbHistoryRead.push_back(
                builder.readHistory(resourceName.c_str(),
                                    makeHzbTextureDesc(static_cast<uint32_t>(m_width),
                                                       static_cast<uint32_t>(m_height),
                                                       level)));
        }

        m_currentHzbRead.reserve(kHzbMaxLevels);
        for (uint32_t level = 0; level < kHzbMaxLevels; ++level) {
            FGResource currentHzbInput = getInput(currentHzbInputSlotName(level));
            if (!currentHzbInput.isValid()) {
                break;
            }
            m_currentHzbRead.push_back(builder.read(currentHzbInput, FGResourceUsage::Sampled));
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

        m_lastVisibleInstanceCount =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::ComputeDispatchCommandLayout>(
                visibleInstanceStateBuffer);
        m_lastVisibleCount =
            GpuDriven::readPublishedWorkItemCount<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            visibleInstanceStateBuffer);
        if (!m_appendToExistingWorklist) {
            GpuDriven::seedWorklistStateBuffer<GpuDriven::MeshDispatchCommandLayout>(
                worklistStateBuffer);
        }

        const ClusterLODData& clusterLodData = m_ctx.clusterLodData;
        ClusterStreamingService* streamingService =
            m_runtimeContext ? m_runtimeContext->clusterStreamingService : nullptr;
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
        RhiBuffer* dummyGroupResidencyBuffer = m_frameGraph->getBuffer(m_dummyGroupResidency);
        RhiBuffer* dummyLodGroupPageTableBuffer = m_frameGraph->getBuffer(m_dummyLodGroupPageTable);
        RhiBuffer* dummyResidencyRequestBuffer = m_frameGraph->getBuffer(m_dummyResidencyRequests);
        RhiBuffer* dummyResidencyRequestStateBuffer =
            m_frameGraph->getBuffer(m_dummyResidencyRequestState);
        RhiBuffer* dummyGroupAgeBuffer = m_frameGraph->getBuffer(m_dummyGroupAge);
        const RhiBuffer* lodNodeBuffer =
            clusterLodAvailable ? &clusterLodData.nodeBuffer : dummyLodNodesBuffer;
        const RhiBuffer* lodGroupBuffer =
            clusterLodAvailable ? &clusterLodData.groupBuffer : dummyLodGroupsBuffer;
        const RhiBuffer* lodBoundsBuffer =
            clusterLodAvailable ? &clusterLodData.boundsBuffer : dummyLodBoundsBuffer;
        const RhiBuffer* sourceLodGroupMeshletIndicesBuffer =
            clusterLodAvailable ? &clusterLodData.groupMeshletIndicesBuffer
                                : dummyLodGroupMeshletIndicesBuffer;
        const bool residencyStreamingResourcesReady =
            clusterLodAvailable &&
            streamingService &&
            streamingService->ready();
        const bool residencyStreamingEnabled =
            residencyStreamingResourcesReady &&
            streamingService &&
            streamingService->streamingEnabled();
        const RhiBuffer* lodGroupMeshletIndicesBuffer =
            clusterLodAvailable
                ? (residencyStreamingEnabled && streamingService->residentGroupMeshletIndicesBuffer()
                       ? streamingService->residentGroupMeshletIndicesBuffer()
                       : &clusterLodData.groupMeshletIndicesBuffer)
                : dummyLodGroupMeshletIndicesBuffer;
        const RhiBuffer* groupResidencyBuffer =
            residencyStreamingResourcesReady ? streamingService->groupResidencyBuffer()
                                             : dummyGroupResidencyBuffer;
        const RhiBuffer* lodGroupPageTableBuffer =
            residencyStreamingResourcesReady ? streamingService->lodGroupPageTableBuffer()
                                             : dummyLodGroupPageTableBuffer;
        const RhiBuffer* residencyRequestBuffer =
            residencyStreamingResourcesReady ? streamingService->residencyRequestBuffer()
                                             : dummyResidencyRequestBuffer;
        const RhiBuffer* residencyRequestStateBuffer =
            residencyStreamingResourcesReady ? streamingService->residencyRequestStateBuffer()
                                             : dummyResidencyRequestStateBuffer;
        const RhiBuffer* groupAgeBuffer =
            residencyStreamingResourcesReady ? streamingService->groupAgeBuffer()
                                             : dummyGroupAgeBuffer;

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

        std::array<const RhiTexture*, kHzbMaxLevels> currentHzbTextures{};
        uint32_t currentHzbTextureCount = 0;
        if (m_enableOcclusionCull && m_cullPassIndex > 0u && !m_currentHzbRead.empty()) {
            const uint32_t maxLevels =
                std::min<uint32_t>(static_cast<uint32_t>(m_currentHzbRead.size()), kHzbMaxLevels);
            for (uint32_t level = 0; level < maxLevels; ++level) {
                const RhiTexture* currentTexture = m_frameGraph->getTexture(m_currentHzbRead[level]);
                if (!currentTexture) {
                    break;
                }
                currentHzbTextures[currentHzbTextureCount++] = currentTexture;
            }
        }
        m_currentHzbLevelCount = currentHzbTextureCount;

        const float4x4 currentCullProj = m_frameContext->unjitteredProj;
        const float4x4 currentCullView = m_frameContext->view;
        const float4x4 previousCullProj = m_frameContext->prevCullProj;
        const float4x4 previousCullView = m_frameContext->prevCullView;
        const float2 currentProjScale =
            float2(std::abs(currentCullProj[0].x), std::abs(currentCullProj[1].y));
        const float2 previousProjScale =
            float2(std::abs(previousCullProj[0].x), std::abs(previousCullProj[1].y));
        const bool classifyWithCurrentHzb =
            m_enableOcclusionCull && m_cullPassIndex > 0u && currentHzbTextureCount > 0u;
        const RhiTexture* const* classifyHzbTextures =
            classifyWithCurrentHzb ? currentHzbTextures.data() : hzbTextures.data();
        const uint32_t classifyHzbTextureCount =
            classifyWithCurrentHzb ? currentHzbTextureCount : hzbTextureCount;

        InstanceClassifyUniforms classifyUni{};
        classifyUni.viewProj = transpose(currentCullProj * currentCullView);
        classifyUni.prevViewProj = classifyWithCurrentHzb
            ? classifyUni.viewProj
            : transpose(previousCullProj * previousCullView);
        classifyUni.prevView = classifyWithCurrentHzb
            ? transpose(currentCullView)
            : transpose(previousCullView);
        classifyUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        classifyUni.prevCameraWorldPos = classifyWithCurrentHzb
            ? m_frameContext->cameraWorldPos
            : m_frameContext->prevCameraWorldPos;
        classifyUni.prevProjScale = classifyWithCurrentHzb
            ? currentProjScale
            : previousProjScale;
        classifyUni.instanceCount = gpuScene.instanceCount;
        classifyUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1u : 0u;
        classifyUni.enableOcclusionCull =
            (m_enableOcclusionCull && classifyHzbTextureCount > 0) ? 1u : 0u;
        classifyUni.hzbLevelCount = classifyHzbTextureCount;
        if (classifyHzbTextureCount > 0) {
            classifyUni.hzbTextureSize =
                float2(static_cast<float>(classifyHzbTextures[0]->width()),
                       static_cast<float>(classifyHzbTextures[0]->height()));
        }
        classifyUni.occlusionDepthBias = m_occlusionDepthBias;
        classifyUni.occlusionBoundsScale = m_occlusionBoundsScale;

        // Classify may temporarily project current HZB through the "previous" helper;
        // meshlet cull still needs both real history and current-frame HZB spaces.
        CullUniforms cullUni{};
        cullUni.viewProj = transpose(currentCullProj * currentCullView);
        cullUni.view = transpose(currentCullView);
        cullUni.prevViewProj = transpose(previousCullProj * previousCullView);
        cullUni.prevView = transpose(previousCullView);
        cullUni.cameraWorldPos = m_frameContext->cameraWorldPos;
        cullUni.prevCameraWorldPos = m_frameContext->prevCameraWorldPos;
        cullUni.projScale = currentProjScale;
        cullUni.renderTargetSize = float2(static_cast<float>(m_width), static_cast<float>(m_height));
        cullUni.prevProjScale = previousProjScale;
        if (hzbTextureCount > 0u) {
            cullUni.hzbTextureSize =
                float2(static_cast<float>(hzbTextures[0]->width()),
                       static_cast<float>(hzbTextures[0]->height()));
        }
        if (currentHzbTextureCount > 0) {
            cullUni.currentHzbTextureSize =
                float2(static_cast<float>(currentHzbTextures[0]->width()),
                       static_cast<float>(currentHzbTextures[0]->height()));
        }
        cullUni.enableFrustumCull = m_frameContext->enableFrustumCull ? 1u : 0u;
        cullUni.enableConeCull = m_frameContext->enableConeCull ? 1u : 0u;
        cullUni.enableOcclusionCull =
            (m_enableOcclusionCull && (hzbTextureCount > 0u || currentHzbTextureCount > 0u))
                ? 1u
                : 0u;
        cullUni.hzbLevelCount = hzbTextureCount;
        cullUni.lodReferencePixels = m_lodReferencePixels;
        cullUni.occlusionDepthBias = m_occlusionDepthBias;
        cullUni.occlusionBoundsScale = m_occlusionBoundsScale;
        cullUni.clusterLodEnabled = clusterLodAvailable ? 1u : 0u;
        cullUni.enableResidencyStreaming = residencyStreamingEnabled ? 1u : 0u;
        cullUni.residencyRequestFrameIndex = m_frameContext ? m_frameContext->frameIndex : 0u;
        cullUni.cullPassIndex = m_cullPassIndex;
        cullUni.currentHzbLevelCount = currentHzbTextureCount;
        cullUni.occlusionCullMode = m_occlusionCullMode;

        // Dispatch 1: coarse instance classification from scene tables.
        encoder.setComputePipeline(classifyIt->second);
        encoder.setBytes(&classifyUni, sizeof(classifyUni), GpuDriven::InstanceClassifyBindings::kUniforms);
        encoder.setBuffer(&gpuScene.instanceBuffer, 0, GpuDriven::InstanceClassifyBindings::kInstances);
        encoder.setBuffer(&gpuScene.geometryBuffer, 0, GpuDriven::InstanceClassifyBindings::kGeometries);
        encoder.setBuffer(visibleInstanceBuffer, 0, GpuDriven::InstanceClassifyBindings::kOutput);
        encoder.setBuffer(visibleInstanceStateBuffer, 0, GpuDriven::InstanceClassifyBindings::kState);
        if (classifyHzbTextureCount > 0) {
            encoder.setTextures(classifyHzbTextures,
                                GpuDriven::InstanceClassifyBindings::kHzbTextureBase,
                                classifyHzbTextureCount);
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
        encoder.setBuffer(groupResidencyBuffer, 0, GpuDriven::MeshletCullBindings::kGroupResidency);
        encoder.setBuffer(lodGroupPageTableBuffer, 0, GpuDriven::MeshletCullBindings::kLodGroupPageTable);
        encoder.setBuffer(residencyRequestBuffer, 0, GpuDriven::MeshletCullBindings::kResidencyRequests);
        encoder.setBuffer(residencyRequestStateBuffer, 0, GpuDriven::MeshletCullBindings::kResidencyRequestState);
        encoder.setBuffer(sourceLodGroupMeshletIndicesBuffer,
                          0,
                          GpuDriven::MeshletCullBindings::kLodGroupMeshletIndicesSource);
        encoder.setBuffer(groupAgeBuffer, 0, GpuDriven::MeshletCullBindings::kGroupAge);
        if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kHzbTextureBase,
                                hzbTextureCount);
        }
        if (currentHzbTextureCount > 0) {
            encoder.setTextures(currentHzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kCurrentHzbTextureBase,
                                currentHzbTextureCount);
        } else if (hzbTextureCount > 0) {
            encoder.setTextures(hzbTextures.data(),
                                GpuDriven::MeshletCullBindings::kCurrentHzbTextureBase,
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
        ImGui::Text("Cull Pass: %u", m_cullPassIndex);
        ImGui::Text("Append Worklist: %s", m_appendToExistingWorklist ? "Yes" : "No");
        ImGui::Text("HZB Occlusion: %s", m_enableOcclusionCull ? "Enabled" : "Disabled");
        ImGui::Text("HZB Mode: %s", m_occlusionCullMode == 1u ? "Current Direct" : "Two Phase");
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
        uint32_t totalCandidates =
            m_lastTraversalStats.candidateClusterMeshletCount +
            m_lastTraversalStats.candidateFallbackMeshletCount;
        if (totalCandidates > 0) {
            float cullRate = 1.0f - float(m_lastVisibleCount) / float(totalCandidates);
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
        ClusterStreamingService* streamingService =
            m_runtimeContext ? m_runtimeContext->clusterStreamingService : nullptr;
        const ClusterStreamingService::DebugStats* streamingStats =
            streamingService ? &streamingService->debugStats() : nullptr;
        bool enableResidencyStreaming =
            streamingService ? streamingService->streamingEnabled() : false;
        if (ImGui::Checkbox("Virtual Residency Streaming", &enableResidencyStreaming) &&
            streamingService) {
            streamingService->setStreamingEnabled(enableResidencyStreaming);
            requestVisibilityHistoryReset();
        }
        if (streamingService &&
            ImGui::BeginCombo("Streaming Budget Preset",
                              ClusterStreamingService::budgetPresetLabel(
                                  streamingService->budgetPreset()))) {
            for (uint32_t presetIndex = 0u;
                 presetIndex < ClusterStreamingService::kBudgetPresetCount;
                 ++presetIndex) {
                const auto preset =
                    static_cast<ClusterStreamingService::BudgetPreset>(presetIndex);
                const bool selected = streamingService->budgetPreset() == preset;
                if (ImGui::Selectable(ClusterStreamingService::budgetPresetLabel(preset),
                                      selected)) {
                    streamingService->setBudgetPreset(preset);
                    requestVisibilityHistoryReset();
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        int streamingBudgetGroups =
            streamingService ? static_cast<int>(streamingService->streamingBudgetGroups()) : 0;
        const uint32_t activeResidencyGroupCount =
            streamingStats ? streamingStats->activeResidencyGroupCount : 0u;
        const uint32_t alwaysResidentGroupCount =
            streamingStats ? streamingStats->lastAlwaysResidentGroupCount : 0u;
        const uint64_t sceneStorageBytes =
            uint64_t(std::max<size_t>(1u, m_ctx.clusterLodData.groupMeshletIndices.size())) *
            sizeof(uint32_t);
        const uint64_t sceneStorageKb =
            std::max<uint64_t>(1ull, (sceneStorageBytes + 1023ull) / 1024ull);
        int streamingStorageCapacityKb =
            streamingService
                ? static_cast<int>(std::max<uint64_t>(
                      1ull,
                      (streamingService->effectiveStreamingStorageCapacityBytes() + 1023ull) / 1024ull))
                : static_cast<int>(sceneStorageKb);
        int streamingTransferCapacityKb =
            streamingService
                ? static_cast<int>(std::max<uint64_t>(
                      1ull,
                      (streamingService->effectiveStreamingTransferCapacityBytes() + 1023ull) /
                          1024ull))
                : static_cast<int>(sceneStorageKb);
        const int maxStreamingStorageCapacityKb =
            static_cast<int>(std::max<uint64_t>(sceneStorageKb,
                                                uint64_t(std::max(streamingStorageCapacityKb, 1))));
        const int maxStreamingTransferCapacityKb =
            static_cast<int>(std::max<uint64_t>(sceneStorageKb,
                                                uint64_t(std::max(streamingTransferCapacityKb, 1))));
        const uint32_t maxDynamicGroupBudget =
            activeResidencyGroupCount > alwaysResidentGroupCount
                ? activeResidencyGroupCount - alwaysResidentGroupCount
                : 1u;
        if (ImGui::SliderInt("Streaming Target (Dynamic Groups)",
                             &streamingBudgetGroups,
                             0,
                             static_cast<int>(std::max(1u, maxDynamicGroupBudget)),
                             "%d") &&
            streamingService) {
            streamingService->setStreamingBudgetGroups(
                static_cast<uint32_t>(std::max(streamingBudgetGroups, 0)));
            requestVisibilityHistoryReset();
        }
        if (ImGui::SliderInt("Streaming Storage Pool (KB)",
                             &streamingStorageCapacityKb,
                             1,
                             maxStreamingStorageCapacityKb,
                             "%d") &&
            streamingService) {
            streamingService->setStreamingStorageCapacityBytes(
                uint64_t(std::max(streamingStorageCapacityKb, 1)) * 1024ull);
            requestVisibilityHistoryReset();
        }
        if (ImGui::SliderInt("Streaming Transfer Cap (KB)",
                             &streamingTransferCapacityKb,
                             1,
                             maxStreamingTransferCapacityKb,
                             "%d") &&
            streamingService) {
            streamingService->setMaxStreamingTransferBytes(
                uint64_t(std::max(streamingTransferCapacityKb, 1)) * 1024ull);
            requestVisibilityHistoryReset();
        }
        int maxLoadsPerFrame =
            streamingService ? static_cast<int>(streamingService->maxLoadsPerFrame()) : 1;
        if (ImGui::SliderInt("Streaming Load Cap",
                             &maxLoadsPerFrame,
                             1,
                             static_cast<int>(std::max(1u, activeResidencyGroupCount)),
                             "%d") &&
            streamingService) {
            streamingService->setMaxLoadsPerFrame(
                static_cast<uint32_t>(std::max(maxLoadsPerFrame, 1)));
        }
        int maxUnloadsPerFrame =
            streamingService ? static_cast<int>(streamingService->maxUnloadsPerFrame()) : 1;
        if (ImGui::SliderInt("Streaming Unload Cap",
                             &maxUnloadsPerFrame,
                             1,
                             static_cast<int>(std::max(1u, activeResidencyGroupCount)),
                             "%d") &&
            streamingService) {
            streamingService->setMaxUnloadsPerFrame(
                static_cast<uint32_t>(std::max(maxUnloadsPerFrame, 1)));
        }
        bool adaptiveBudgetEnabled =
            streamingService ? streamingService->adaptiveBudgetEnabled() : false;
        if (ImGui::Checkbox("Adaptive Unload Age", &adaptiveBudgetEnabled) &&
            streamingService) {
            streamingService->setAdaptiveBudgetEnabled(adaptiveBudgetEnabled);
        }
        int ageThreshold =
            streamingService ? static_cast<int>(streamingService->configuredAgeThreshold()) : 16;
        if (ImGui::SliderInt("Streaming Unload Age",
                             &ageThreshold,
                             1,
                             256,
                             "%d") &&
            streamingService) {
            streamingService->setAgeThreshold(
                static_cast<uint32_t>(std::max(ageThreshold, 1)));
        }
        if (streamingService && streamingService->adaptiveBudgetEnabled()) {
            ImGui::Text("Effective Streaming Unload Age: %u",
                        streamingService->ageThreshold());
        }
        bool compactAgeFilterDispatch =
            streamingService ? streamingService->compactAgeFilterDispatchEnabled() : true;
        if (ImGui::Checkbox("Compact GPU Age Filter", &compactAgeFilterDispatch) &&
            streamingService) {
            streamingService->setCompactAgeFilterDispatchEnabled(compactAgeFilterDispatch);
        }
        if (ImGui::Button("Reset Residency State") && streamingService) {
            streamingService->markStateDirty();
            requestVisibilityHistoryReset();
        }
        if (m_frameContext) {
            ImGui::Text("Scene Instances: %u", m_ctx.gpuScene.instanceCount);
            ImGui::Text("Scene Visible Flags: %u", m_ctx.gpuScene.visibleInstanceCount);
            ImGui::Text("GPU Culling: %s", m_frameContext->gpuDrivenCulling ? "On" : "Off");
        }
        ImGui::Text("LOD Nodes: %u active",
                    streamingStats ? streamingStats->activeResidencyNodeCount : 0u);
        ImGui::Text("Resident Groups: %u / %u",
                    streamingStats ? streamingStats->lastResidentGroupCount : 0u,
                    activeResidencyGroupCount);
        ImGui::Text("Always Resident Groups: %u",
                    alwaysResidentGroupCount);
        ImGui::Text("Resident Storage: %u / %u indices",
                    streamingStats ? streamingStats->residentHeapUsed : 0u,
                    streamingStats ? streamingStats->residentHeapCapacity : 0u);
        ImGui::Text("Upload Staging (frame): %.2f / %.2f KB",
                    streamingService
                        ? float(double(streamingService->streamingUploadBytesUsed()) / 1024.0)
                        : 0.0f,
                    streamingService
                        ? float(double(streamingService->effectiveStreamingTransferCapacityBytes()) /
                                1024.0)
                        : 0.0f);
        ImGui::Text("Dynamic Resident Groups: %u",
                    streamingStats ? streamingStats->dynamicResidentGroupCount : 0u);
        ImGui::Text("Pending Residency Groups: %u",
                    streamingStats ? streamingStats->pendingResidencyGroupCount : 0u);
        ImGui::Text("GPU Residency Requests (last frame): %u",
                    streamingStats ? streamingStats->lastResidencyRequestCount : 0u);
        ImGui::Text("GPU Unload Requests (last frame): %u",
                    streamingStats ? streamingStats->lastUnloadRequestCount : 0u);
        ImGui::Text("Promoted / Evicted (last frame): %u / %u",
                    streamingStats ? streamingStats->lastResidencyPromotedCount : 0u,
                    streamingStats ? streamingStats->lastResidencyEvictedCount : 0u);
        ImGui::Text("Load / Unload Cap: %u / %u",
                    streamingStats ? streamingStats->maxLoadsPerFrame : 0u,
                    streamingStats ? streamingStats->maxUnloadsPerFrame : 0u);
        ImGui::Text("Unload Pending / Confirmed: %u / %u",
                    streamingStats ? streamingStats->pendingUnloadGroupCount : 0u,
                    streamingStats ? streamingStats->confirmedUnloadGroupCount : 0u);
        ImGui::Text("Unload Age Threshold: %u",
                    streamingStats ? streamingStats->ageThreshold : 0u);
        const bool historyValid =
            m_frameGraph && !m_hzbHistoryRead.empty() && m_frameGraph->isHistoryValid(m_hzbHistoryRead[0]);
        ImGui::Text("HZB History: %s (%u levels)", historyValid ? "Ready" : "Warming Up", m_hzbLevelCount);
        ImGui::Text("Current HZB Inputs: %u levels", m_currentHzbLevelCount);
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
    uint32_t m_currentHzbLevelCount = 0;
    uint32_t m_cullPassIndex = 0;
    bool m_appendToExistingWorklist = false;
    bool m_enableFrustumCull = false;
    bool m_enableConeCull = false;
    bool m_enableOcclusionCull = true;
    uint32_t m_occlusionCullMode = 0;
    float m_lodReferencePixels = 96.0f;
    float m_occlusionDepthBias = 0.0015f;
    float m_occlusionBoundsScale = 1.1f;
    FGResource m_visibleInstanceState;
    FGResource m_clusterTraversalStats;
    FGResource m_dummyLodNodes;
    FGResource m_dummyLodGroups;
    FGResource m_dummyLodGroupMeshletIndices;
    FGResource m_dummyLodBounds;
    FGResource m_dummyGroupResidency;
    FGResource m_dummyLodGroupPageTable;
    FGResource m_dummyResidencyRequests;
    FGResource m_dummyResidencyRequestState;
    FGResource m_dummyGroupAge;
    std::vector<FGResource> m_hzbHistoryRead;
    std::vector<FGResource> m_currentHzbRead;

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

    static std::string currentHzbInputSlotName(uint32_t level) {
        return "currentHzb" + std::to_string(level);
    }
};

METALLIC_REGISTER_PASS(MeshletCullPass);
