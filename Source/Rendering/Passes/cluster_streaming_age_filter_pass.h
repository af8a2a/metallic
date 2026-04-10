#pragma once

#include "cluster_streaming_service.h"
#include "gpu_cull_resources.h"
#include "pass_registry.h"
#include "render_pass.h"

#ifdef _WIN32
#include "rhi_resource_utils.h"
#include "vulkan_upload_service.h"
#include "vulkan_resource_handles.h"
#include <array>
#endif

class ClusterStreamingAgeFilterPass : public RenderPass {
public:
    ClusterStreamingAgeFilterPass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~ClusterStreamingAgeFilterPass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    void setup(FGBuilder& builder) override {
        FGResource cullInput = getInput("cullResult");
        if (cullInput.isValid()) {
            m_cullResultRead = builder.read(cullInput);
        }
    }

    void prepareResources(RhiCommandBuffer&) override {
#ifdef _WIN32
        if (!m_runtimeContext || !m_runtimeContext->clusterStreamingService ||
            !m_runtimeContext->clusterStreamingService->gpuStatsReadbackEnabled()) {
            resetGpuStatsReadbacks();
            return;
        }
        consumeReadyGpuStatsReadbacks();
#endif
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        if (!m_runtimeContext || !m_frameContext) {
            return;
        }

        auto pipelineIt =
            m_runtimeContext->computePipelinesRhi.find("ClusterStreamingAgeFilterPass");
        if (pipelineIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipelineIt->second.nativeHandle()) {
            return;
        }

        ClusterStreamingService* streamingService = m_runtimeContext->clusterStreamingService;
        if (!streamingService ||
            !streamingService->ready() ||
            !streamingService->streamingEnabled() ||
            m_ctx.clusterLodData.totalGroupCount == 0u) {
            return;
        }

        const RhiBuffer* groupResidencyBuffer = streamingService->groupResidencyBuffer();
        const RhiBuffer* groupAgeBuffer = streamingService->groupAgeBuffer();
        const RhiBuffer* unloadRequestBuffer = streamingService->unloadRequestBuffer();
        const RhiBuffer* unloadRequestStateBuffer = streamingService->unloadRequestStateBuffer();
        const RhiBuffer* statsBuffer = streamingService->streamingStatsBuffer();
        if (!groupResidencyBuffer || !groupAgeBuffer || !unloadRequestBuffer ||
            !unloadRequestStateBuffer || !statsBuffer) {
            return;
        }

        StreamingAgeFilterUniforms uniforms{};
        uniforms.groupCount = m_ctx.clusterLodData.totalGroupCount;
        uniforms.ageThreshold = streamingService->ageThreshold();
        uniforms.requestFrameIndex = m_frameContext ? m_frameContext->frameIndex : 0u;

        encoder.setComputePipeline(pipelineIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), GpuDriven::StreamingAgeFilterBindings::kUniforms);
        encoder.setBuffer(groupResidencyBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kGroupResidency);
        encoder.setBuffer(groupAgeBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kGroupAge);
        encoder.setBuffer(unloadRequestBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kUnloadRequests);
        encoder.setBuffer(unloadRequestStateBuffer,
                          0,
                          GpuDriven::StreamingAgeFilterBindings::kUnloadRequestState);
        encoder.setBuffer(statsBuffer, 0, GpuDriven::StreamingAgeFilterBindings::kStats);

        constexpr uint32_t kThreadCount = 64u;
        const uint32_t dispatchX = (uniforms.groupCount + kThreadCount - 1u) / kThreadCount;
        encoder.dispatchThreadgroups({dispatchX, 1, 1}, {kThreadCount, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);

#ifdef _WIN32
        if (streamingService->gpuStatsReadbackEnabled()) {
            scheduleGpuStatsReadback(encoder, statsBuffer);
        }
#endif
    }

private:
    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Streaming Age Filter";
    FGResource m_cullResultRead;

#ifdef _WIN32
    static constexpr uint32_t kGpuStatsReadbackRingSize = 4u;

    bool tryConsumeGpuStatsReadback(VulkanReadbackService::ReadbackRequest& request) {
        if (!request.valid() || !m_runtimeContext || !m_runtimeContext->readbackService ||
            !m_runtimeContext->clusterStreamingService || !m_frameContext) {
            return false;
        }

        VulkanReadbackService* readbackService = m_runtimeContext->readbackService;
        if (!readbackService->isReady(request, m_frameContext->frameIndex)) {
            return false;
        }

        ClusterStreamingGpuStats stats{};
        if (readbackService->readData(request, &stats, sizeof(stats))) {
            m_runtimeContext->clusterStreamingService->ingestGpuStreamingStats(stats);
        }
        request = {};
        return true;
    }

    void consumeReadyGpuStatsReadbacks() {
        for (VulkanReadbackService::ReadbackRequest& request : m_gpuStatsReadbacks) {
            tryConsumeGpuStatsReadback(request);
        }
    }

    void resetGpuStatsReadbacks() {
        for (VulkanReadbackService::ReadbackRequest& request : m_gpuStatsReadbacks) {
            request = {};
        }
    }

    void scheduleGpuStatsReadback(RhiComputeCommandEncoder& encoder, const RhiBuffer* statsBuffer) {
        if (!m_runtimeContext || !m_runtimeContext->readbackService || !m_frameContext ||
            !statsBuffer) {
            return;
        }

        VulkanReadbackService* readbackService = m_runtimeContext->readbackService;
        const VkBuffer vkStatsBuffer = getVulkanBufferHandle(statsBuffer);
        const VkCommandBuffer commandBuffer = static_cast<VkCommandBuffer>(encoder.nativeHandle());
        if (vkStatsBuffer == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE) {
            return;
        }

        VulkanReadbackService::ReadbackRequest& slot =
            m_gpuStatsReadbacks[m_frameContext->frameIndex % kGpuStatsReadbackRingSize];
        if (slot.valid() && !tryConsumeGpuStatsReadback(slot)) {
            return;
        }

        VkBufferMemoryBarrier2 statsReadbackBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        statsReadbackBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        statsReadbackBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        statsReadbackBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        statsReadbackBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        statsReadbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        statsReadbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        statsReadbackBarrier.buffer = vkStatsBuffer;
        statsReadbackBarrier.offset = 0u;
        statsReadbackBarrier.size = sizeof(ClusterStreamingGpuStats);

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &statsReadbackBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);

        slot = readbackService->scheduleBufferReadback(vkStatsBuffer,
                                                       0u,
                                                       sizeof(ClusterStreamingGpuStats));
        readbackService->recordPendingReadbacks(commandBuffer);
    }

    std::array<VulkanReadbackService::ReadbackRequest, kGpuStatsReadbackRingSize>
        m_gpuStatsReadbacks = {};
#endif
};
