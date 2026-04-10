#pragma once

#include "cluster_streaming_service.h"
#include "gpu_cull_resources.h"
#include "pass_registry.h"
#include "render_pass.h"
#include "rhi_resource_utils.h"

#include <cstring>

#ifdef _WIN32
#include "vulkan_upload_service.h"
#include "vulkan_resource_handles.h"
#include <vector>
#endif

class ClusterStreamingUpdatePass : public RenderPass {
public:
    ClusterStreamingUpdatePass(const RenderContext& ctx, int w, int h)
        : m_ctx(ctx), m_width(w), m_height(h) {}

    ~ClusterStreamingUpdatePass() override = default;

    FGPassType passType() const override { return FGPassType::Compute; }
    const char* name() const override { return m_name.c_str(); }

    void configure(const PassConfig& config) override {
        m_name = config.name;
    }

    FGResource getOutput(const std::string& name) const override {
        if (name == "streamingSync") {
            return m_streamingSync;
        }
        return FGResource{};
    }

    void setup(FGBuilder& builder) override {
        m_streamingSync = builder.createToken("ClusterStreamingSync");
    }

    void prepareResources(RhiCommandBuffer&) override {
        if (!m_runtimeContext || !m_runtimeContext->clusterStreamingService) {
            return;
        }

        m_runtimeContext->clusterStreamingService->runUpdateStage(m_ctx.clusterLodData,
                                                                  *m_runtimeContext,
                                                                  m_frameContext);
    }

    void executeCompute(RhiComputeCommandEncoder& encoder) override {
        if (!m_runtimeContext || !m_runtimeContext->clusterStreamingService) {
            return;
        }

        ClusterStreamingService* streamingService = m_runtimeContext->clusterStreamingService;
        if (!streamingService->streamingEnabled() || !streamingService->ready()) {
            return;
        }

        const RhiBuffer* statsBuffer = streamingService->streamingStatsBuffer();
        if (!statsBuffer) {
            return;
        }

        initializeStreamingStatsBuffer(encoder,
                                       statsBuffer,
                                       m_frameContext ? m_frameContext->frameIndex : 0u,
                                       streamingService->activeUpdateTransferBytes());

        auto pipelineIt = m_runtimeContext->computePipelinesRhi.find("ClusterStreamingUpdatePass");
        if (pipelineIt == m_runtimeContext->computePipelinesRhi.end() ||
            !pipelineIt->second.nativeHandle()) {
            return;
        }

        const uint32_t patchCount = streamingService->streamingPatchCount();
        if (patchCount == 0u) {
            return;
        }

        const RhiBuffer* sourceGroupMeshletIndicesBuffer =
            m_ctx.clusterLodData.groupMeshletIndicesBuffer.nativeHandle()
                ? &m_ctx.clusterLodData.groupMeshletIndicesBuffer
                : nullptr;
        const RhiBuffer* residentGroupMeshletIndicesBuffer =
            streamingService->residentGroupMeshletIndicesBuffer();
        const RhiBuffer* lodGroupPageTableBuffer = streamingService->lodGroupPageTableBuffer();
        const RhiBuffer* patchBuffer = streamingService->streamingPatchBuffer();
        if (!sourceGroupMeshletIndicesBuffer ||
            !residentGroupMeshletIndicesBuffer ||
            !lodGroupPageTableBuffer ||
            !patchBuffer ||
            !statsBuffer) {
            return;
        }

#ifdef _WIN32
        const std::vector<StreamingStorage::CopyRegion>& copyRegions =
            streamingService->streamingUploadCopyRegions();
        if (!copyRegions.empty()) {
            const uint64_t transferWaitValue =
                submitStreamingDataCopiesAsync(streamingService->streamingUploadStagingBuffer(),
                                               residentGroupMeshletIndicesBuffer,
                                               copyRegions);
            if (transferWaitValue != 0u) {
                streamingService->completeTransferTask(transferWaitValue);
            } else if (!recordStreamingDataCopies(encoder,
                                                  streamingService->streamingUploadStagingBuffer(),
                                                  residentGroupMeshletIndicesBuffer,
                                                  copyRegions)) {
                return;
            } else {
                streamingService->completeTransferTask(0u);
            }
        }

        if (rhiBufferContents(*patchBuffer) == nullptr) {
            const StreamingPatch* patchData = streamingService->streamingPatchData();
            if (!uploadStreamingPatches(encoder, patchBuffer, patchData, patchCount)) {
                return;
            }
        }
#endif

        StreamingUpdateUniforms uniforms{};
        uniforms.patchCount = patchCount;
#ifdef _WIN32
        uniforms.copySourceData = 0u;
#else
        uniforms.copySourceData = 1u;
#endif
        uniforms.sourceGroupMeshletIndexCount = static_cast<uint32_t>(
            sourceGroupMeshletIndicesBuffer->size() / sizeof(uint32_t));
        uniforms.residentGroupMeshletIndexCount = static_cast<uint32_t>(
            residentGroupMeshletIndicesBuffer->size() / sizeof(uint32_t));
        uniforms.groupPageTableCount = static_cast<uint32_t>(
            lodGroupPageTableBuffer->size() / sizeof(uint64_t));

        encoder.setComputePipeline(pipelineIt->second);
        encoder.setBytes(&uniforms, sizeof(uniforms), GpuDriven::StreamingUpdateBindings::kUniforms);
        encoder.setBuffer(sourceGroupMeshletIndicesBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kSourceGroupMeshletIndices);
        encoder.setBuffer(residentGroupMeshletIndicesBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kResidentGroupMeshletIndices);
        encoder.setBuffer(lodGroupPageTableBuffer,
                          0,
                          GpuDriven::StreamingUpdateBindings::kGroupPageTable);
        encoder.setBuffer(patchBuffer, 0, GpuDriven::StreamingUpdateBindings::kPatches);
        encoder.setBuffer(statsBuffer, 0, GpuDriven::StreamingUpdateBindings::kStats);

        constexpr uint32_t kThreadCount = 64u;
        const uint32_t dispatchX = (patchCount + kThreadCount - 1u) / kThreadCount;
        encoder.dispatchThreadgroups({dispatchX, 1, 1}, {kThreadCount, 1, 1});
        encoder.memoryBarrier(RhiBarrierScope::Buffers);
        const uint64_t graphicsCompletionSerial =
            m_runtimeContext->rhi ? m_runtimeContext->rhi->nextGraphicsSubmissionSerial() : 0u;
        streamingService->setPendingUpdateGraphicsCompletionSerial(graphicsCompletionSerial);
        streamingService->markUpdateTaskQueued();
    }

private:
    const RenderContext& m_ctx;
    int m_width = 0;
    int m_height = 0;
    std::string m_name = "Cluster Streaming Update";
    FGResource m_streamingSync;

    static ClusterStreamingGpuStats makeInitialStreamingGpuStats(uint32_t frameIndex,
                                                                 uint64_t copiedBytes) {
        ClusterStreamingGpuStats stats{};
        stats.frameIndex = frameIndex;
        stats.copiedBytesLow = static_cast<uint32_t>(copiedBytes & 0xFFFFFFFFull);
        stats.copiedBytesHigh = static_cast<uint32_t>(copiedBytes >> 32u);
        return stats;
    }

    static bool initializeStreamingStatsBuffer(RhiComputeCommandEncoder& encoder,
                                               const RhiBuffer* statsBuffer,
                                               uint32_t frameIndex,
                                               uint64_t copiedBytes) {
        if (!statsBuffer) {
            return false;
        }

        const ClusterStreamingGpuStats stats =
            makeInitialStreamingGpuStats(frameIndex, copiedBytes);
        if (void* mappedStats = rhiBufferContents(*statsBuffer)) {
            std::memcpy(mappedStats, &stats, sizeof(stats));
            return true;
        }

#ifdef _WIN32
        VulkanUploadService* uploadService = vulkanGetUploadService();
        const VkBuffer vkStatsBuffer = getVulkanBufferHandle(statsBuffer);
        const VkCommandBuffer commandBuffer = static_cast<VkCommandBuffer>(encoder.nativeHandle());
        if (!uploadService || vkStatsBuffer == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE) {
            return false;
        }

        if (!uploadService->stageBuffer(vkStatsBuffer, 0u, &stats, sizeof(stats))) {
            return false;
        }

        uploadService->recordPendingUploads(commandBuffer);

        VkBufferMemoryBarrier2 statsUploadBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        statsUploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        statsUploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        statsUploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        statsUploadBarrier.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        statsUploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        statsUploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        statsUploadBarrier.buffer = vkStatsBuffer;
        statsUploadBarrier.offset = 0u;
        statsUploadBarrier.size = sizeof(ClusterStreamingGpuStats);

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &statsUploadBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
        return true;
#else
        (void)encoder;
        (void)frameIndex;
        (void)copiedBytes;
        return false;
#endif
    }

#ifdef _WIN32
    static bool recordStreamingDataCopies(
        RhiComputeCommandEncoder& encoder,
        const RhiBuffer* stagingBuffer,
        const RhiBuffer* residentBuffer,
        const std::vector<StreamingStorage::CopyRegion>& copyRegions) {
        if (copyRegions.empty()) {
            return true;
        }
        if (!stagingBuffer || !residentBuffer) {
            return false;
        }

        const VkBuffer vkStagingBuffer = getVulkanBufferHandle(stagingBuffer);
        const VkBuffer vkResidentBuffer = getVulkanBufferHandle(residentBuffer);
        const VkCommandBuffer commandBuffer = static_cast<VkCommandBuffer>(encoder.nativeHandle());
        if (vkStagingBuffer == VK_NULL_HANDLE ||
            vkResidentBuffer == VK_NULL_HANDLE ||
            commandBuffer == VK_NULL_HANDLE) {
            return false;
        }

        std::vector<VkBufferCopy> bufferCopies;
        bufferCopies.reserve(copyRegions.size());
        for (const StreamingStorage::CopyRegion& copyRegion : copyRegions) {
            if (copyRegion.sizeBytes == 0u) {
                continue;
            }

            VkBufferCopy region{};
            region.srcOffset = copyRegion.srcOffsetBytes;
            region.dstOffset = copyRegion.dstOffsetBytes;
            region.size = copyRegion.sizeBytes;
            bufferCopies.push_back(region);
        }
        if (bufferCopies.empty()) {
            return true;
        }

        vkCmdCopyBuffer(commandBuffer,
                        vkStagingBuffer,
                        vkResidentBuffer,
                        static_cast<uint32_t>(bufferCopies.size()),
                        bufferCopies.data());

        VkBufferMemoryBarrier2 residentUploadBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        residentUploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        residentUploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        residentUploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        residentUploadBarrier.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        residentUploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        residentUploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        residentUploadBarrier.buffer = vkResidentBuffer;
        residentUploadBarrier.offset = 0u;
        residentUploadBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &residentUploadBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
        return true;
    }

    static uint64_t submitStreamingDataCopiesAsync(
        const RhiBuffer* stagingBuffer,
        const RhiBuffer* residentBuffer,
        const std::vector<StreamingStorage::CopyRegion>& copyRegions) {
        if (copyRegions.empty() || !stagingBuffer || !residentBuffer) {
            return 0u;
        }

        VulkanUploadService* uploadService = vulkanGetUploadService();
        if (!uploadService || !uploadService->hasTransferQueue()) {
            return 0u;
        }

        const VkBuffer vkStagingBuffer = getVulkanBufferHandle(stagingBuffer);
        const VkBuffer vkResidentBuffer = getVulkanBufferHandle(residentBuffer);
        if (vkStagingBuffer == VK_NULL_HANDLE || vkResidentBuffer == VK_NULL_HANDLE) {
            return 0u;
        }

        std::vector<VkBufferCopy> bufferCopies;
        bufferCopies.reserve(copyRegions.size());
        for (const StreamingStorage::CopyRegion& copyRegion : copyRegions) {
            if (copyRegion.sizeBytes == 0u) {
                continue;
            }

            VkBufferCopy region{};
            region.srcOffset = copyRegion.srcOffsetBytes;
            region.dstOffset = copyRegion.dstOffsetBytes;
            region.size = copyRegion.sizeBytes;
            bufferCopies.push_back(region);
        }
        if (bufferCopies.empty()) {
            return 0u;
        }

        return uploadService->submitAsyncBufferCopies(vkStagingBuffer,
                                                      vkResidentBuffer,
                                                      bufferCopies.data(),
                                                      static_cast<uint32_t>(bufferCopies.size()));
    }

    static bool uploadStreamingPatches(RhiComputeCommandEncoder& encoder,
                                       const RhiBuffer* patchBuffer,
                                       const StreamingPatch* patchData,
                                       uint32_t patchCount) {
        if (!patchBuffer || !patchData || patchCount == 0u) {
            return false;
        }

        VulkanUploadService* uploadService = vulkanGetUploadService();
        if (!uploadService) {
            return false;
        }

        const VkBuffer vkPatchBuffer = getVulkanBufferHandle(patchBuffer);
        const VkCommandBuffer commandBuffer = static_cast<VkCommandBuffer>(encoder.nativeHandle());
        if (vkPatchBuffer == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE) {
            return false;
        }

        const VkDeviceSize uploadSize = VkDeviceSize(patchCount) * sizeof(StreamingPatch);
        if (!uploadService->stageBuffer(vkPatchBuffer, 0u, patchData, uploadSize)) {
            return false;
        }

        uploadService->recordPendingUploads(commandBuffer);

        VkBufferMemoryBarrier2 patchUploadBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        patchUploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        patchUploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        patchUploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        patchUploadBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        patchUploadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        patchUploadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        patchUploadBarrier.buffer = vkPatchBuffer;
        patchUploadBarrier.offset = 0u;
        patchUploadBarrier.size = uploadSize;

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &patchUploadBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
        return true;
    }
#endif
};
