#pragma once

#include "cluster_lod_builder.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "gpu_driven_helpers.h"
#include "rhi_backend.h"
#include "rhi_resource_utils.h"
#include "streaming_storage.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

class ClusterStreamingService {
public:
    static constexpr uint32_t kStreamingAgeHistogramBucketCount = 16u;

    struct StreamingStats {
        uint32_t residentGroupCount = 0;
        uint32_t residentClusterCount = 0;
        uint32_t alwaysResidentGroupCount = 0;
        uint32_t dynamicResidentGroupCount = 0;

        uint64_t storagePoolCapacityBytes = 0u;
        uint64_t storagePoolUsedBytes = 0u;
        uint32_t residentHeapCapacity = 0;
        uint32_t residentHeapUsed = 0;

        uint32_t loadRequestsThisFrame = 0;
        uint32_t unloadRequestsThisFrame = 0;
        uint32_t loadsExecutedThisFrame = 0;
        uint32_t unloadsExecutedThisFrame = 0;
        uint32_t loadsDeferredThisFrame = 0;

        uint64_t transferBytesThisFrame = 0u;
        float transferUtilization = 0.0f;

        uint32_t failedAllocations = 0;
        bool gpuStatsValid = false;
        uint32_t gpuStatsFrameIndex = UINT32_MAX;
        uint32_t gpuUnloadRequestCount = 0;
        float gpuAverageUnloadAge = 0.0f;
        uint32_t gpuAppliedPatchCount = 0;
        uint64_t gpuCopiedBytes = 0u;
        uint32_t ageHistogramBucketWidth = 1;
        uint32_t ageHistogramMaxAge = 0;
        std::array<uint32_t, kStreamingAgeHistogramBucketCount> ageHistogram = {};
        std::vector<uint32_t> residentGroupsPerLod;
        std::vector<uint32_t> totalGroupsPerLod;
    };

    struct DebugStats {
        uint32_t activeResidencyNodeCount = 0;
        uint32_t activeResidencyGroupCount = 0;
        uint32_t lastResidencyRequestCount = 0;
        uint32_t lastUnloadRequestCount = 0;
        uint32_t lastResidencyPromotedCount = 0;
        uint32_t lastResidencyEvictedCount = 0;
        uint32_t lastResidentGroupCount = 0;
        uint32_t lastAlwaysResidentGroupCount = 0;
        uint32_t residentHeapCapacity = 0;
        uint32_t residentHeapUsed = 0;
        uint32_t dynamicResidentGroupCount = 0;
        uint32_t pendingResidencyGroupCount = 0;
        uint32_t pendingUnloadGroupCount = 0;
        uint32_t confirmedUnloadGroupCount = 0;
        uint32_t maxLoadsPerFrame = 0;
        uint32_t maxUnloadsPerFrame = 0;
        uint32_t ageThreshold = 0;
        uint32_t streamingTaskCapacity = 0;
        uint32_t freeStreamingTaskCount = 0;
        uint32_t preparedStreamingTaskCount = 0;
        uint32_t transferSubmittedTaskCount = 0;
        uint32_t updateQueuedTaskCount = 0;
        uint32_t selectedTransferTaskIndex = UINT32_MAX;
        uint32_t selectedUpdateTaskIndex = UINT32_MAX;
        uint32_t selectedUpdatePatchCount = 0;
        uint64_t selectedTransferBytes = 0u;
        uint64_t selectedUpdateTransferWaitValue = 0u;
        bool resourcesReady = false;
    };

    void setStreamingEnabled(bool enabled) {
        if (m_enableStreaming == enabled) {
            return;
        }

        m_enableStreaming = enabled;
        clearGpuStreamingStats();
        m_stateDirty = true;
    }

    bool streamingEnabled() const { return m_enableStreaming; }

    void setGpuStatsReadbackEnabled(bool enabled) {
        if (m_enableGpuStatsReadback == enabled) {
            return;
        }

        m_enableGpuStatsReadback = enabled;
        clearGpuStreamingStats();
    }

    bool gpuStatsReadbackEnabled() const { return m_enableGpuStatsReadback; }

    void setStreamingBudgetGroups(uint32_t budgetGroups) {
        m_streamingBudgetGroups = budgetGroups;
    }

    uint32_t streamingBudgetGroups() const { return m_streamingBudgetGroups; }

    void setMaxLoadsPerFrame(uint32_t maxLoadsPerFrame) {
        m_maxLoadsPerFrame = std::max(1u, maxLoadsPerFrame);
    }

    uint32_t maxLoadsPerFrame() const { return m_maxLoadsPerFrame; }

    void setMaxUnloadsPerFrame(uint32_t maxUnloadsPerFrame) {
        m_maxUnloadsPerFrame = std::max(1u, maxUnloadsPerFrame);
    }

    uint32_t maxUnloadsPerFrame() const { return m_maxUnloadsPerFrame; }

    void setAgeThreshold(uint32_t ageThreshold) {
        m_ageThreshold = ageThreshold;
    }

    uint32_t ageThreshold() const { return m_ageThreshold; }

    void setStreamingStorageCapacityBytes(uint64_t capacityBytes) {
        capacityBytes = std::max<uint64_t>(capacityBytes, sizeof(uint32_t));
        if (m_streamingStorageCapacityBytes == capacityBytes) {
            return;
        }

        m_streamingStorageCapacityBytes = capacityBytes;
        m_stateDirty = true;
    }

    uint64_t streamingStorageCapacityBytes() const { return m_streamingStorageCapacityBytes; }
    uint64_t effectiveStreamingStorageCapacityBytes() const {
        return uint64_t(m_streamingStorage.capacityElements()) * sizeof(uint32_t);
    }

    void setMaxStreamingTransferBytes(uint64_t maxTransferBytes) {
        maxTransferBytes = std::max<uint64_t>(maxTransferBytes, sizeof(uint32_t));
        if (m_maxStreamingTransferBytes == maxTransferBytes) {
            return;
        }

        m_maxStreamingTransferBytes = maxTransferBytes;
        m_stateDirty = true;
    }

    uint64_t maxStreamingTransferBytes() const { return m_maxStreamingTransferBytes; }
    uint64_t effectiveStreamingTransferCapacityBytes() const {
        return m_streamingStorage.maxUploadBytesPerFrame();
    }

    void markStateDirty() { m_stateDirty = true; }

    const DebugStats& debugStats() const { return m_debugStats; }
    const StreamingStats& streamingStats() const { return m_streamingStats; }
    void ingestGpuStreamingStats(const ClusterStreamingGpuStats& stats) {
        if (stats.frameIndex == UINT32_MAX) {
            m_lastGpuStreamingStats = {};
            m_hasGpuStreamingStats = false;
            applyGpuStreamingStatsToTelemetry();
            return;
        }

        if (m_hasGpuStreamingStats && stats.frameIndex < m_lastGpuStreamingStats.frameIndex) {
            return;
        }

        m_lastGpuStreamingStats = stats;
        m_hasGpuStreamingStats = true;
        applyGpuStreamingStatsToTelemetry();
    }

    bool ready() const {
        for (const FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer ||
                !frameBuffers.groupAgeBuffer ||
                !frameBuffers.residencyRequestBuffer ||
                !frameBuffers.residencyRequestStateBuffer ||
                !frameBuffers.unloadRequestBuffer ||
                !frameBuffers.streamingStatsBuffer ||
                !frameBuffers.streamingPatchBuffer ||
                !frameBuffers.unloadRequestStateBuffer) {
                return false;
            }
        }
        return m_lodGroupPageTableBuffer && m_streamingStorage.ready() &&
               m_streamingStorage.uploadReady();
    }

    bool useResidentHeap() const {
        return ready() && m_enableStreaming;
    }

    const RhiBuffer* groupResidencyBuffer() const {
        return activeFrameBuffers().groupResidencyBuffer.get();
    }
    const RhiBuffer* groupAgeBuffer() const {
        return activeFrameBuffers().groupAgeBuffer.get();
    }
    const RhiBuffer* streamingStatsBuffer() const {
        return activeFrameBuffers().streamingStatsBuffer.get();
    }
    const RhiBuffer* lodGroupPageTableBuffer() const {
        return m_lodGroupPageTableBuffer.get();
    }
    const RhiBuffer* residentGroupMeshletIndicesBuffer() const {
        return m_streamingStorage.buffer();
    }
    const RhiBuffer* streamingUploadStagingBuffer() const {
        return validTaskIndex(m_transferTaskIndex) ? m_streamingStorage.uploadBuffer(m_transferTaskIndex)
                                                   : nullptr;
    }
    const std::vector<StreamingStorage::CopyRegion>& streamingUploadCopyRegions() const {
        static const std::vector<StreamingStorage::CopyRegion> kEmpty;
        return validTaskIndex(m_transferTaskIndex) ? m_streamingStorage.copyRegions(m_transferTaskIndex)
                                                   : kEmpty;
    }
    uint64_t streamingUploadBytesUsed() const {
        return validTaskIndex(m_transferTaskIndex) ? m_streamingStorage.uploadBytesUsed(m_transferTaskIndex)
                                                   : 0u;
    }
    uint64_t activeUpdateTransferBytes() const {
        return validTaskIndex(m_updateTaskIndex) ? m_streamingStorage.uploadBytesUsed(m_updateTaskIndex)
                                                 : 0u;
    }
    const RhiBuffer* residencyRequestBuffer() const {
        return activeFrameBuffers().residencyRequestBuffer.get();
    }
    const RhiBuffer* residencyRequestStateBuffer() const {
        return activeFrameBuffers().residencyRequestStateBuffer.get();
    }
    const RhiBuffer* unloadRequestBuffer() const {
        return activeFrameBuffers().unloadRequestBuffer.get();
    }
    const RhiBuffer* unloadRequestStateBuffer() const {
        return activeFrameBuffers().unloadRequestStateBuffer.get();
    }
    const RhiBuffer* streamingPatchBuffer() const {
        return activeFrameBuffers().streamingPatchBuffer.get();
    }
    uint32_t streamingPatchCount() const {
        return m_activeUpdatePatchCount;
    }
    const StreamingPatch* streamingPatchData() const {
        if (!validTaskIndex(m_updateTaskIndex)) {
            return nullptr;
        }

        const std::vector<StreamingPatch>& patches = m_streamingTasks[m_updateTaskIndex].patches;
        return patches.empty() ? nullptr : patches.data();
    }
    void completeTransferTask(uint64_t waitValue) {
        if (!validTaskIndex(m_transferTaskIndex)) {
            return;
        }

        StreamingTask& task = m_streamingTasks[m_transferTaskIndex];
        task.state = StreamingTaskState::TransferSubmitted;
        task.transferWaitValue = waitValue;
        task.transferSubmitFrame = m_frameIndex;
        m_transferTaskIndex = kInvalidTaskIndex;
    }
    void markUpdateTaskQueued() {
        if (!validTaskIndex(m_updateTaskIndex)) {
            m_pendingUpdateGraphicsCompletionSerial = 0u;
            return;
        }

        StreamingTask& task = m_streamingTasks[m_updateTaskIndex];
        task.state = StreamingTaskState::UpdateQueued;
        task.updateQueuedFrame = m_frameIndex;
        if (m_pendingUpdateGraphicsCompletionSerial != 0u) {
            task.graphicsCompletionSerial = m_pendingUpdateGraphicsCompletionSerial;
            m_pendingUpdateGraphicsCompletionSerial = 0u;
        }
        m_updateTaskIndex = kInvalidTaskIndex;
    }
    void setPendingUpdateGraphicsCompletionSerial(uint64_t completionSerial) {
        m_pendingUpdateGraphicsCompletionSerial = completionSerial;
    }
    uint64_t consumePendingTransferWaitValue() {
        const uint64_t consumed = m_activeUpdateTransferWaitValue;
        m_activeUpdateTransferWaitValue = 0u;
        return consumed;
    }

    void runUpdateStage(const ClusterLODData& clusterLodData,
                        const PipelineRuntimeContext& runtimeContext,
                        const FrameContext* frameContext) {
        m_debugStats.activeResidencyNodeCount = clusterLodData.totalNodeCount;
        m_debugStats.activeResidencyGroupCount = clusterLodData.totalGroupCount;
        m_activeFrameSlot = frameContext ? (frameContext->frameIndex % kBufferedFrameCount) : 0u;
        m_frameIndex = frameContext ? frameContext->frameIndex : 0u;
        m_prepareTaskIndex = kInvalidTaskIndex;
        m_transferTaskIndex = kInvalidTaskIndex;
        m_updateTaskIndex = kInvalidTaskIndex;
        m_activeUpdatePatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;
        m_loadRequestsThisFrame = 0u;
        m_unloadRequestsThisFrame = 0u;
        m_failedAllocationsThisFrame = 0u;
        m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;

        const bool clusterLodAvailable =
            clusterLodData.nodeBuffer.nativeHandle() &&
            clusterLodData.groupBuffer.nativeHandle() &&
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle() &&
            clusterLodData.boundsBuffer.nativeHandle();
        if (!clusterLodAvailable) {
            resetDebugStats();
            return;
        }

        ensureStreamingResources(clusterLodData, runtimeContext);
        if (!ready()) {
            updateDebugStats(&clusterLodData);
            return;
        }
        recycleCompletedTasks(runtimeContext);

        const bool sourceBufferChanged =
            m_residencySourceNodeBufferHandle != clusterLodData.nodeBuffer.nativeHandle() ||
            m_residencySourceGroupBufferHandle != clusterLodData.groupBuffer.nativeHandle() ||
            m_residencySourceGroupMeshletIndicesHandle !=
                clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        if (m_stateDirty || sourceBufferChanged) {
            requestHistoryReset(frameContext);
            resetStreamingTasks();
            if (beginPrepareTask()) {
                // The page table is device-local. Clear it immediately on rebuild so the
                // cull shader never dereferences stale resident heap offsets for one frame.
                queueInvalidateAllGroups();
                finalizePrepareTask();
            }
            if (beginPrepareTask()) {
                rebuildStreamingState(clusterLodData);
                finalizePrepareTask();
            } else {
                rebuildStreamingState(clusterLodData);
            }
        } else {
            runRequestReadbackStage(clusterLodData);
            if (beginPrepareTask()) {
                runResidencyUpdateStage(clusterLodData);
                finalizePrepareTask();
            }
        }

        selectTransferTask();
        // Intentionally update from an older transfer submission so the graphics queue
        // never has to wait on uploads recorded earlier in the same frame.
        selectUpdateTask();
        uploadCanonicalStateToActiveFrame();
        updateDebugStats(&clusterLodData);
    }

private:
    static constexpr uint32_t kInvalidResidentHeapOffset = UINT32_MAX;
    static constexpr uint32_t kInvalidTaskIndex = UINT32_MAX;
    static constexpr uint32_t kInvalidFrameIndex = UINT32_MAX;
    static constexpr uint32_t kBufferedFrameCount = 2u;
    static constexpr uint32_t kStreamingTaskCount = 3u;
    static constexpr uint64_t kDefaultStreamingStorageCapacityBytes = 512ull * 1024ull * 1024ull;
    static constexpr uint64_t kDefaultMaxStreamingTransferBytes = 32ull * 1024ull * 1024ull;

    enum class StreamingTaskState : uint8_t {
        Free,
        Prepared,
        TransferSubmitted,
        UpdateQueued,
    };

    struct StreamingTask {
        StreamingTaskState state = StreamingTaskState::Free;
        uint32_t transferSubmitFrame = UINT32_MAX;
        uint32_t updateQueuedFrame = UINT32_MAX;
        uint64_t transferWaitValue = 0u;
        uint64_t serial = 0u;
        uint64_t graphicsCompletionSerial = 0u;
        std::vector<StreamingPatch> patches;
    };

    struct GroupResidentAllocation {
        uint32_t heapOffset = kInvalidResidentHeapOffset;
        uint32_t heapCount = 0;
    };

    struct FrameBuffers {
        std::unique_ptr<RhiBuffer> groupResidencyBuffer;
        std::unique_ptr<RhiBuffer> groupAgeBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestStateBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestStateBuffer;
        std::unique_ptr<RhiBuffer> streamingStatsBuffer;
        std::unique_ptr<RhiBuffer> streamingPatchBuffer;
        uint32_t submittedFrameIndex = kInvalidFrameIndex;
    };

    static void resetFrameBuffers(FrameBuffers& frameBuffers) {
        frameBuffers.groupResidencyBuffer.reset();
        frameBuffers.groupAgeBuffer.reset();
        frameBuffers.residencyRequestBuffer.reset();
        frameBuffers.residencyRequestStateBuffer.reset();
        frameBuffers.unloadRequestBuffer.reset();
        frameBuffers.unloadRequestStateBuffer.reset();
        frameBuffers.streamingStatsBuffer.reset();
        frameBuffers.streamingPatchBuffer.reset();
        frameBuffers.submittedFrameIndex = kInvalidFrameIndex;
    }

    FrameBuffers& activeFrameBuffers() {
        return m_frameBuffers[m_activeFrameSlot % kBufferedFrameCount];
    }

    const FrameBuffers& activeFrameBuffers() const {
        return m_frameBuffers[m_activeFrameSlot % kBufferedFrameCount];
    }

    static bool validTaskIndex(uint32_t taskIndex) {
        return taskIndex < kStreamingTaskCount;
    }

    bool taskHasTransferWork(uint32_t taskIndex) const {
        return validTaskIndex(taskIndex) &&
               !m_streamingStorage.copyRegions(taskIndex).empty();
    }

    void releaseTask(uint32_t taskIndex) {
        if (!validTaskIndex(taskIndex)) {
            return;
        }

        m_streamingStorage.resetUploadFrame(taskIndex);
        StreamingTask& task = m_streamingTasks[taskIndex];
        task.state = StreamingTaskState::Free;
        task.transferSubmitFrame = UINT32_MAX;
        task.updateQueuedFrame = UINT32_MAX;
        task.transferWaitValue = 0u;
        task.serial = 0u;
        task.graphicsCompletionSerial = 0u;
        task.patches.clear();

        if (m_prepareTaskIndex == taskIndex) {
            m_prepareTaskIndex = kInvalidTaskIndex;
        }
        if (m_transferTaskIndex == taskIndex) {
            m_transferTaskIndex = kInvalidTaskIndex;
        }
        if (m_updateTaskIndex == taskIndex) {
            m_updateTaskIndex = kInvalidTaskIndex;
        }
    }

    void resetStreamingTasks() {
        for (uint32_t taskIndex = 0u; taskIndex < kStreamingTaskCount; ++taskIndex) {
            releaseTask(taskIndex);
        }
        m_prepareTaskIndex = kInvalidTaskIndex;
        m_transferTaskIndex = kInvalidTaskIndex;
        m_updateTaskIndex = kInvalidTaskIndex;
        m_activeUpdatePatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;
        m_pendingUpdateGraphicsCompletionSerial = 0u;
    }

    void recycleCompletedTasks(const PipelineRuntimeContext& runtimeContext) {
        const uint64_t completedGraphicsSerial =
            runtimeContext.rhi ? runtimeContext.rhi->completedGraphicsSubmissionSerial() : 0u;
        for (uint32_t taskIndex = 0u; taskIndex < kStreamingTaskCount; ++taskIndex) {
            const StreamingTask& task = m_streamingTasks[taskIndex];
            if (task.state != StreamingTaskState::UpdateQueued) {
                continue;
            }

            const bool completedBySerial =
                task.graphicsCompletionSerial != 0u &&
                completedGraphicsSerial >= task.graphicsCompletionSerial;
            const bool completedByFrameDelay =
                task.updateQueuedFrame != UINT32_MAX &&
                m_frameIndex >= task.updateQueuedFrame + kBufferedFrameCount;
            if (!completedBySerial && !completedByFrameDelay) {
                continue;
            }

            releaseTask(taskIndex);
        }
    }

    bool beginPrepareTask() {
        for (uint32_t taskIndex = 0u; taskIndex < kStreamingTaskCount; ++taskIndex) {
            if (m_streamingTasks[taskIndex].state != StreamingTaskState::Free) {
                continue;
            }

            m_prepareTaskIndex = taskIndex;
            m_streamingStorage.resetUploadFrame(taskIndex);

            StreamingTask& task = m_streamingTasks[taskIndex];
            task.transferSubmitFrame = UINT32_MAX;
            task.updateQueuedFrame = UINT32_MAX;
            task.transferWaitValue = 0u;
            task.serial = 0u;
            task.graphicsCompletionSerial = 0u;
            task.patches.clear();
            task.state = StreamingTaskState::Prepared;
            task.serial = m_nextTaskSerial++;
            return true;
        }

        m_prepareTaskIndex = kInvalidTaskIndex;
        return false;
    }

    static uint32_t* mappedUint32(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<uint32_t*>(rhiBufferContents(*buffer));
    }

    static ClusterResidencyRequest* mappedRequests(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<ClusterResidencyRequest*>(rhiBufferContents(*buffer));
    }

    static ClusterUnloadRequest* mappedUnloadRequests(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<ClusterUnloadRequest*>(rhiBufferContents(*buffer));
    }

    static StreamingPatch* mappedPatches(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<StreamingPatch*>(rhiBufferContents(*buffer));
    }

    void resetDebugStats() {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastUnloadRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = 0;
        m_debugStats.residentHeapUsed = 0;
        m_debugStats.dynamicResidentGroupCount = 0;
        m_debugStats.pendingResidencyGroupCount = 0;
        m_debugStats.pendingUnloadGroupCount = 0;
        m_debugStats.confirmedUnloadGroupCount = 0;
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
        m_debugStats.ageThreshold = m_ageThreshold;
        m_debugStats.streamingTaskCapacity = kStreamingTaskCount;
        m_debugStats.freeStreamingTaskCount = kStreamingTaskCount;
        m_debugStats.preparedStreamingTaskCount = 0;
        m_debugStats.transferSubmittedTaskCount = 0;
        m_debugStats.updateQueuedTaskCount = 0;
        m_debugStats.selectedTransferTaskIndex = kInvalidTaskIndex;
        m_debugStats.selectedUpdateTaskIndex = kInvalidTaskIndex;
        m_debugStats.selectedUpdatePatchCount = 0;
        m_debugStats.selectedTransferBytes = 0u;
        m_debugStats.selectedUpdateTransferWaitValue = 0u;
        m_debugStats.resourcesReady = false;
        m_streamingStats = {};
        m_loadRequestsThisFrame = 0u;
        m_unloadRequestsThisFrame = 0u;
        m_failedAllocationsThisFrame = 0u;
        m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
        clearGpuStreamingStats();
    }

    void requestHistoryReset(const FrameContext* frameContext) const {
        if (!frameContext) {
            return;
        }

        auto* mutableFrameContext = const_cast<FrameContext*>(frameContext);
        mutableFrameContext->historyReset = true;
    }

    void createFrameBufferSet(FrameBuffers& frameBuffers,
                              const PipelineRuntimeContext& runtimeContext,
                              uint32_t frameSlot,
                              uint32_t groupCapacity) {
        RhiBufferDesc residencyDesc{};
        residencyDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        residencyDesc.hostVisible = true;
        const std::string residencyName = "ClusterLodGroupResidency[" + std::to_string(frameSlot) + "]";
        residencyDesc.debugName = residencyName.c_str();
        frameBuffers.groupResidencyBuffer = runtimeContext.resourceFactory->createBuffer(residencyDesc);

        RhiBufferDesc ageDesc{};
        ageDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        ageDesc.hostVisible = true;
        const std::string ageName = "ClusterLodGroupAge[" + std::to_string(frameSlot) + "]";
        ageDesc.debugName = ageName.c_str();
        frameBuffers.groupAgeBuffer = runtimeContext.resourceFactory->createBuffer(ageDesc);

        RhiBufferDesc requestDesc{};
        requestDesc.size = size_t(groupCapacity) * sizeof(ClusterResidencyRequest);
        requestDesc.hostVisible = true;
        const std::string requestName = "ClusterLodResidencyRequests[" + std::to_string(frameSlot) + "]";
        requestDesc.debugName = requestName.c_str();
        frameBuffers.residencyRequestBuffer = runtimeContext.resourceFactory->createBuffer(requestDesc);

        RhiBufferDesc requestStateDesc{};
        requestStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        requestStateDesc.hostVisible = true;
        const std::string requestStateName =
            "ClusterLodResidencyRequestState[" + std::to_string(frameSlot) + "]";
        requestStateDesc.debugName = requestStateName.c_str();
        frameBuffers.residencyRequestStateBuffer =
            runtimeContext.resourceFactory->createBuffer(requestStateDesc);

        RhiBufferDesc unloadDesc{};
        unloadDesc.size = size_t(groupCapacity) * sizeof(ClusterUnloadRequest);
        unloadDesc.hostVisible = true;
        const std::string unloadName = "ClusterLodUnloadRequests[" + std::to_string(frameSlot) + "]";
        unloadDesc.debugName = unloadName.c_str();
        frameBuffers.unloadRequestBuffer = runtimeContext.resourceFactory->createBuffer(unloadDesc);

        RhiBufferDesc unloadStateDesc{};
        unloadStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        unloadStateDesc.hostVisible = true;
        const std::string unloadStateName =
            "ClusterLodUnloadRequestState[" + std::to_string(frameSlot) + "]";
        unloadStateDesc.debugName = unloadStateName.c_str();
        frameBuffers.unloadRequestStateBuffer =
            runtimeContext.resourceFactory->createBuffer(unloadStateDesc);

        RhiBufferDesc patchDesc{};
        patchDesc.size = size_t(groupCapacity) * sizeof(StreamingPatch);
#ifdef _WIN32
        patchDesc.hostVisible = false;
#else
        patchDesc.hostVisible = true;
#endif
        const std::string patchName = "ClusterLodStreamingPatches[" + std::to_string(frameSlot) + "]";
        patchDesc.debugName = patchName.c_str();
        frameBuffers.streamingPatchBuffer = runtimeContext.resourceFactory->createBuffer(patchDesc);

        RhiBufferDesc statsDesc{};
        statsDesc.size = sizeof(ClusterStreamingGpuStats);
#ifdef _WIN32
        statsDesc.hostVisible = false;
#else
        statsDesc.hostVisible = true;
#endif
        const std::string statsName = "ClusterLodStreamingStats[" + std::to_string(frameSlot) + "]";
        statsDesc.debugName = statsName.c_str();
        frameBuffers.streamingStatsBuffer = runtimeContext.resourceFactory->createBuffer(statsDesc);
    }

    void ensureStreamingResources(const ClusterLODData& clusterLodData,
                                  const PipelineRuntimeContext& runtimeContext) {
        if (!runtimeContext.resourceFactory) {
            return;
        }

        const uint32_t groupCapacity = std::max(1u, clusterLodData.totalGroupCount);
        const uint32_t storageCapacity = computeStreamingStorageCapacity(clusterLodData);
        const uint64_t transferCapacityBytes = computeStreamingTransferCapacityBytes(clusterLodData);
        const bool hasExistingResources =
            m_residencyGroupCapacity != 0u || m_lodGroupPageTableBuffer || m_streamingStorage.ready() ||
            m_streamingStorage.uploadReady();
        const bool needsRecreate =
            !ready() ||
            m_residencyGroupCapacity != groupCapacity ||
            m_streamingStorage.capacityElements() != storageCapacity ||
            m_streamingStorage.maxUploadBytesPerFrame() != transferCapacityBytes;
        if (!needsRecreate) {
            return;
        }

        if (hasExistingResources && runtimeContext.rhi) {
            runtimeContext.rhi->waitIdle();
        }

        for (uint32_t frameSlot = 0; frameSlot < kBufferedFrameCount; ++frameSlot) {
            resetFrameBuffers(m_frameBuffers[frameSlot]);
            createFrameBufferSet(m_frameBuffers[frameSlot],
                                 runtimeContext,
                                 frameSlot,
                                 groupCapacity);
        }

        RhiBufferDesc groupPageTableDesc{};
        groupPageTableDesc.size = size_t(groupCapacity) * sizeof(uint64_t);
        groupPageTableDesc.hostVisible = false;
        groupPageTableDesc.debugName = "ClusterLodGroupPageTable";
        m_lodGroupPageTableBuffer = runtimeContext.resourceFactory->createBuffer(groupPageTableDesc);

        m_streamingStorage.ensureBuffer(*runtimeContext.resourceFactory,
                                        storageCapacity,
                                        "ClusterLodResidentGroupMeshletStorage");
        m_streamingStorage.ensureUploadBuffers(*runtimeContext.resourceFactory,
                                              kStreamingTaskCount,
                                              transferCapacityBytes,
                                              "ClusterLodStreamingUpload");

        m_residencyGroupCapacity = groupCapacity;
        m_groupResidencyState.assign(groupCapacity, 0u);
        m_groupAgeState.assign(groupCapacity, 0u);
        m_groupResidentSinceFrame.assign(groupCapacity, kInvalidFrameIndex);
        m_groupPendingUnloadState.assign(groupCapacity, 0u);
        m_pendingResidencyRequestFrames.assign(groupCapacity, kInvalidFrameIndex);
        m_residentTouchSeenScratch.assign(groupCapacity, 0u);
        m_unloadRequestSeenScratch.assign(groupCapacity, 0u);
        m_patchLastWriteIndexScratch.assign(groupCapacity, UINT32_MAX);
        m_patchTouchedGroupsScratch.clear();
        m_stateDirty = true;
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        resetStreamingTasks();
        m_residencySourceNodeBufferHandle = nullptr;
        m_residencySourceGroupBufferHandle = nullptr;
        m_residencySourceGroupMeshletIndicesHandle = nullptr;
        resetResidentHeapAllocator();
        uploadCanonicalStateToAllFrames();
        updateDebugStats(&clusterLodData);
    }

    void seedResidencyRequestQueue(FrameBuffers& frameBuffers) {
        if (!frameBuffers.residencyRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            frameBuffers.residencyRequestStateBuffer.get());
    }

    void seedUnloadRequestQueue(FrameBuffers& frameBuffers) {
        if (!frameBuffers.unloadRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            frameBuffers.unloadRequestStateBuffer.get());
    }

    void uploadCanonicalStateToFrame(FrameBuffers& frameBuffers,
                                     bool uploadAgeState,
                                     bool markSubmittedFrame) {
        if (uint32_t* words = mappedUint32(frameBuffers.groupResidencyBuffer.get())) {
            std::memcpy(words,
                        m_groupResidencyState.data(),
                        size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        }
        if (uploadAgeState) {
            if (uint32_t* ages = mappedUint32(frameBuffers.groupAgeBuffer.get())) {
                std::memcpy(ages,
                            m_groupAgeState.data(),
                            size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
            }
        }
        if (frameBuffers.residencyRequestBuffer) {
            if (void* requests = rhiBufferContents(*frameBuffers.residencyRequestBuffer)) {
                std::memset(requests, 0, frameBuffers.residencyRequestBuffer->size());
            }
        }
        if (frameBuffers.unloadRequestBuffer) {
            if (void* unloadRequests = rhiBufferContents(*frameBuffers.unloadRequestBuffer)) {
                std::memset(unloadRequests, 0, frameBuffers.unloadRequestBuffer->size());
            }
        }
        seedResidencyRequestQueue(frameBuffers);
        seedUnloadRequestQueue(frameBuffers);
        frameBuffers.submittedFrameIndex = markSubmittedFrame ? m_frameIndex : kInvalidFrameIndex;
    }

    void uploadSelectedUpdateTaskToActiveFrame() {
        m_activeUpdatePatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;

        if (!validTaskIndex(m_updateTaskIndex)) {
            return;
        }

        const StreamingTask& task = m_streamingTasks[m_updateTaskIndex];
        const uint32_t patchCount =
            std::min<uint32_t>(static_cast<uint32_t>(task.patches.size()), m_residencyGroupCapacity);
        m_activeUpdatePatchCount = patchCount;
        m_activeUpdateTransferWaitValue = task.transferWaitValue;
        if (patchCount == 0u) {
            return;
        }

        StreamingPatch* patches = mappedPatches(activeFrameBuffers().streamingPatchBuffer.get());
        if (patches) {
            std::memcpy(patches,
                        task.patches.data(),
                        size_t(patchCount) * sizeof(StreamingPatch));
        }
    }

    void uploadCanonicalStateToAllFrames() {
        for (FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer) {
                continue;
            }
            uploadCanonicalStateToFrame(frameBuffers, true, false);
        }
    }

    void uploadCanonicalStateToActiveFrame() {
        uploadCanonicalStateToFrame(activeFrameBuffers(), true, true);
        uploadSelectedUpdateTaskToActiveFrame();
    }

    void resetResidentHeapAllocator() {
        m_streamingStorage.resetAllocator();
        m_groupResidentAllocations.assign(m_residencyGroupCapacity, {});
    }

    bool isGroupResident(uint32_t groupIndex) const {
        return groupIndex < m_groupResidencyState.size() &&
               (m_groupResidencyState[groupIndex] & kClusterLodGroupResidencyResident) != 0u;
    }

    bool isGroupAlwaysResident(uint32_t groupIndex) const {
        return groupIndex < m_groupResidencyState.size() &&
               (m_groupResidencyState[groupIndex] & kClusterLodGroupResidencyAlwaysResident) != 0u;
    }

    void collectLeafGroupsForNode(uint32_t nodeIndex,
                                  const ClusterLODData& clusterLodData,
                                  std::vector<uint32_t>& outLeafGroups) const {
        if (nodeIndex >= clusterLodData.nodes.size()) {
            return;
        }

        std::vector<uint32_t> nodeStack;
        nodeStack.push_back(nodeIndex);
        while (!nodeStack.empty()) {
            const uint32_t currentNodeIndex = nodeStack.back();
            nodeStack.pop_back();
            if (currentNodeIndex >= clusterLodData.nodes.size()) {
                continue;
            }

            const GPULodNode& node = clusterLodData.nodes[currentNodeIndex];
            if (node.isLeaf != 0u) {
                for (uint32_t childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                    outLeafGroups.push_back(node.childOffset + childIndex);
                }
                continue;
            }

            for (uint32_t childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                nodeStack.push_back(node.childOffset + childIndex);
            }
        }
    }

    void appendStreamingPatch(const StreamingPatch& patch) {
        if (!validTaskIndex(m_prepareTaskIndex)) {
            return;
        }

        m_streamingTasks[m_prepareTaskIndex].patches.push_back(patch);
    }

    void finalizePrepareTask() {
        if (!validTaskIndex(m_prepareTaskIndex)) {
            return;
        }

        StreamingTask& task = m_streamingTasks[m_prepareTaskIndex];
        if (m_patchLastWriteIndexScratch.size() != m_residencyGroupCapacity) {
            m_patchLastWriteIndexScratch.assign(m_residencyGroupCapacity, UINT32_MAX);
        }

        m_patchTouchedGroupsScratch.clear();
        for (uint32_t patchIndex = 0u;
             patchIndex < static_cast<uint32_t>(task.patches.size());
             ++patchIndex) {
            const StreamingPatch& patch = task.patches[patchIndex];
            if (patch.groupIndex >= m_residencyGroupCapacity) {
                continue;
            }
            if (m_patchLastWriteIndexScratch[patch.groupIndex] == UINT32_MAX) {
                m_patchTouchedGroupsScratch.push_back(patch.groupIndex);
            }
            m_patchLastWriteIndexScratch[patch.groupIndex] = patchIndex;
        }

        if (!m_patchTouchedGroupsScratch.empty()) {
            std::vector<StreamingPatch> dedupedPatches;
            dedupedPatches.reserve(m_patchTouchedGroupsScratch.size());
            for (uint32_t patchIndex = 0u;
                 patchIndex < static_cast<uint32_t>(task.patches.size());
                 ++patchIndex) {
                const StreamingPatch& patch = task.patches[patchIndex];
                if (patch.groupIndex >= m_residencyGroupCapacity ||
                    m_patchLastWriteIndexScratch[patch.groupIndex] != patchIndex) {
                    continue;
                }
                dedupedPatches.push_back(patch);
            }
            task.patches.swap(dedupedPatches);
        }

        for (uint32_t groupIndex : m_patchTouchedGroupsScratch) {
            m_patchLastWriteIndexScratch[groupIndex] = UINT32_MAX;
        }
        m_patchTouchedGroupsScratch.clear();

        if (task.patches.empty() && !taskHasTransferWork(m_prepareTaskIndex)) {
            releaseTask(m_prepareTaskIndex);
            return;
        }

        if (!taskHasTransferWork(m_prepareTaskIndex)) {
            task.state = StreamingTaskState::TransferSubmitted;
            task.transferWaitValue = 0u;
            task.transferSubmitFrame = m_frameIndex;
        }

        m_prepareTaskIndex = kInvalidTaskIndex;
    }

    void queueInvalidateAllGroups() {
        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            queueUnloadPatch(groupIndex);
        }
    }

    uint32_t findOldestTask(StreamingTaskState state, bool requirePriorFrame = false) const {
        uint32_t bestTaskIndex = kInvalidTaskIndex;
        uint64_t bestSerial = UINT64_MAX;
        for (uint32_t taskIndex = 0u; taskIndex < kStreamingTaskCount; ++taskIndex) {
            const StreamingTask& task = m_streamingTasks[taskIndex];
            if (task.state != state) {
                continue;
            }
            if (requirePriorFrame &&
                (task.transferSubmitFrame == UINT32_MAX || task.transferSubmitFrame >= m_frameIndex)) {
                continue;
            }
            if (task.serial < bestSerial) {
                bestSerial = task.serial;
                bestTaskIndex = taskIndex;
            }
        }
        return bestTaskIndex;
    }

    void selectTransferTask() {
        m_transferTaskIndex = findOldestTask(StreamingTaskState::Prepared);
    }

    bool taskReadyForUpdate(const StreamingTask& task) const {
        if (task.state != StreamingTaskState::TransferSubmitted ||
            task.transferSubmitFrame == UINT32_MAX) {
            return false;
        }

        if (task.transferWaitValue == 0u) {
            return true;
        }

        return task.transferSubmitFrame < m_frameIndex;
    }

    void selectUpdateTask() {
        m_updateTaskIndex = kInvalidTaskIndex;
        uint64_t bestSerial = UINT64_MAX;
        for (uint32_t taskIndex = 0u; taskIndex < kStreamingTaskCount; ++taskIndex) {
            const StreamingTask& task = m_streamingTasks[taskIndex];
            if (!taskReadyForUpdate(task) || task.serial >= bestSerial) {
                continue;
            }

            bestSerial = task.serial;
            m_updateTaskIndex = taskIndex;
        }
    }

    void queueLoadPatch(uint32_t groupIndex,
                       uint32_t heapOffset,
                       uint32_t clusterStart,
                       uint32_t clusterCount) {
        StreamingPatch patch{};
        patch.groupIndex = groupIndex;
        patch.residentHeapOffset = makeClusterLodGroupPageResidentAddress(heapOffset);
        patch.clusterStart = clusterStart;
        patch.clusterCount = clusterCount;
        appendStreamingPatch(patch);
    }

    void queueUnloadPatch(uint32_t groupIndex) {
        StreamingPatch patch{};
        patch.groupIndex = groupIndex;
        patch.residentHeapOffset = makeClusterLodGroupPageInvalidAddress(m_frameIndex);
        patch.clusterStart = 0u;
        patch.clusterCount = 0u;
        appendStreamingPatch(patch);
    }

    bool ensureResidentHeapSliceForGroup(uint32_t groupIndex,
                                         const ClusterLODData& clusterLodData) {
        if (groupIndex >= m_groupResidentAllocations.size() ||
            groupIndex >= clusterLodData.groups.size()) {
            return false;
        }

        GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        if (allocation.heapOffset != kInvalidResidentHeapOffset) {
            return true;
        }

        const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
        if (size_t(group.clusterStart) + group.clusterCount > clusterLodData.groupMeshletIndices.size()) {
            return false;
        }

        uint32_t heapOffset = 0u;
        if (!m_streamingStorage.allocate(group.clusterCount, heapOffset)) {
            ++m_failedAllocationsThisFrame;
            return false;
        }
        if (heapOffset + group.clusterCount > m_streamingStorage.capacityElements()) {
            ++m_failedAllocationsThisFrame;
            m_streamingStorage.release(heapOffset, group.clusterCount);
            return false;
        }

        allocation.heapOffset = heapOffset;
        allocation.heapCount = group.clusterCount;
        return true;
    }

    bool queueLoadPatchForGroup(uint32_t groupIndex, const ClusterLODData& clusterLodData) {
        if (groupIndex >= m_groupResidentAllocations.size() ||
            groupIndex >= clusterLodData.groups.size()) {
            return false;
        }

        GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        const bool hadAllocation = allocation.heapOffset != kInvalidResidentHeapOffset;
        if (!ensureResidentHeapSliceForGroup(groupIndex, clusterLodData)) {
            return false;
        }

        const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
        if (size_t(group.clusterStart) + group.clusterCount > clusterLodData.groupMeshletIndices.size()) {
            if (!hadAllocation) {
                invalidateResidentGroup(groupIndex);
            }
            return false;
        }

        const uint64_t dstOffsetBytes =
            uint64_t(m_groupResidentAllocations[groupIndex].heapOffset) * sizeof(uint32_t);
        const uint64_t uploadSizeBytes = uint64_t(group.clusterCount) * sizeof(uint32_t);
        const uint32_t* srcData = clusterLodData.groupMeshletIndices.data() + group.clusterStart;
        if (!validTaskIndex(m_prepareTaskIndex) ||
            !m_streamingStorage.stageUpload(m_prepareTaskIndex,
                                            srcData,
                                            uploadSizeBytes,
                                            dstOffsetBytes)) {
            if (!hadAllocation) {
                invalidateResidentGroup(groupIndex);
            }
            return false;
        }

        queueLoadPatch(groupIndex,
                       m_groupResidentAllocations[groupIndex].heapOffset,
                       group.clusterStart,
                       group.clusterCount);
        return true;
    }

    void invalidateResidentGroup(uint32_t groupIndex) {
        if (groupIndex >= m_groupResidentAllocations.size()) {
            return;
        }

        GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        m_streamingStorage.release(allocation.heapOffset, allocation.heapCount);
        allocation = {};
    }

    void clearPendingUnloadCandidate(uint32_t groupIndex) {
        if (groupIndex < m_groupPendingUnloadState.size()) {
            m_groupPendingUnloadState[groupIndex] = 0u;
        }
        m_confirmedUnloadGroups.erase(
            std::remove(m_confirmedUnloadGroups.begin(),
                        m_confirmedUnloadGroups.end(),
                        groupIndex),
            m_confirmedUnloadGroups.end());
    }

    void clearPendingResidencyRequestFrame(uint32_t groupIndex) {
        if (groupIndex < m_pendingResidencyRequestFrames.size()) {
            m_pendingResidencyRequestFrames[groupIndex] = kInvalidFrameIndex;
        }
    }

    void advanceResidentGroupAges(uint32_t sourceFrameIndex) {
        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
                m_groupAgeState[groupIndex] = 0u;
                continue;
            }
            if (m_groupResidentSinceFrame[groupIndex] == kInvalidFrameIndex ||
                m_groupResidentSinceFrame[groupIndex] > sourceFrameIndex ||
                m_residentTouchSeenScratch[groupIndex] != 0u) {
                continue;
            }

            m_groupAgeState[groupIndex] = std::min(m_groupAgeState[groupIndex] + 1u, 0xFFFFu);
        }
    }

    void touchDynamicResidentGroup(uint32_t groupIndex) {
        clearPendingUnloadCandidate(groupIndex);
        if (groupIndex < m_groupAgeState.size()) {
            m_groupAgeState[groupIndex] = 0u;
        }
        auto it = std::find(m_dynamicResidentGroups.begin(), m_dynamicResidentGroups.end(), groupIndex);
        if (it != m_dynamicResidentGroups.end()) {
            m_dynamicResidentGroups.erase(it);
        }
        m_dynamicResidentGroups.push_back(groupIndex);
    }

    void enqueuePendingResidencyGroup(uint32_t groupIndex, uint32_t requestFrameIndex) {
        clearPendingUnloadCandidate(groupIndex);
        if (groupIndex < m_groupResidencyState.size()) {
            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyRequested;
        }
        if (groupIndex < m_pendingResidencyRequestFrames.size()) {
            m_pendingResidencyRequestFrames[groupIndex] = requestFrameIndex;
        }
        if (std::find(m_pendingResidencyGroups.begin(), m_pendingResidencyGroups.end(), groupIndex) ==
            m_pendingResidencyGroups.end()) {
            m_pendingResidencyGroups.push_back(groupIndex);
        }
    }

    bool evictResidentGroup(uint32_t groupIndex) {
        if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
            return false;
        }

        auto evictIt = std::find(m_dynamicResidentGroups.begin(), m_dynamicResidentGroups.end(), groupIndex);
        if (evictIt != m_dynamicResidentGroups.end()) {
            m_dynamicResidentGroups.erase(evictIt);
        }
        if (groupIndex < m_residencyGroupCapacity) {
            m_groupResidencyState[groupIndex] &= ~(kClusterLodGroupResidencyResident |
                                                   kClusterLodGroupResidencyRequested);
            m_groupAgeState[groupIndex] = 0u;
            m_groupResidentSinceFrame[groupIndex] = kInvalidFrameIndex;
        }
        if (groupIndex < m_groupPendingUnloadState.size()) {
            m_groupPendingUnloadState[groupIndex] = 0u;
        }
        invalidateResidentGroup(groupIndex);
        queueUnloadPatch(groupIndex);
        ++m_debugStats.lastResidencyEvictedCount;
        return true;
    }

    void promotePendingResidencyGroups(const ClusterLODData& clusterLodData,
                                       uint32_t& remainingLoads) {
        if (remainingLoads == 0u) {
            return;
        }

        size_t pendingIndex = 0;
        while (pendingIndex < m_pendingResidencyGroups.size() && remainingLoads > 0u) {
            const uint32_t groupIndex = m_pendingResidencyGroups[pendingIndex];
            if (groupIndex >= m_residencyGroupCapacity ||
                groupIndex >= clusterLodData.totalGroupCount) {
                clearPendingResidencyRequestFrame(groupIndex);
                m_pendingResidencyGroups.erase(
                    m_pendingResidencyGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (isGroupResident(groupIndex)) {
                m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
                m_groupAgeState[groupIndex] = 0u;
                if (!isGroupAlwaysResident(groupIndex)) {
                    touchDynamicResidentGroup(groupIndex);
                }
                clearPendingResidencyRequestFrame(groupIndex);
                m_pendingResidencyGroups.erase(
                    m_pendingResidencyGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (m_streamingBudgetGroups == 0u) {
                break;
            }

            if (m_dynamicResidentGroups.size() >= size_t(m_streamingBudgetGroups)) {
                break;
            }

            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyResident;
            m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
            m_groupAgeState[groupIndex] = 0u;
            m_groupResidentSinceFrame[groupIndex] = m_frameIndex;
            if (!queueLoadPatchForGroup(groupIndex, clusterLodData)) {
                m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyResident;
                m_groupResidentSinceFrame[groupIndex] = kInvalidFrameIndex;
                break;
            }
            touchDynamicResidentGroup(groupIndex);
            ++m_debugStats.lastResidencyPromotedCount;
            --remainingLoads;
            clearPendingResidencyRequestFrame(groupIndex);
            m_pendingResidencyGroups.erase(
                m_pendingResidencyGroups.begin() +
                static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
        }
    }

    void rebuildStreamingState(const ClusterLODData& clusterLodData) {
        std::fill(m_groupResidencyState.begin(), m_groupResidencyState.end(), 0u);
        std::fill(m_groupAgeState.begin(), m_groupAgeState.end(), 0u);
        std::fill(m_groupResidentSinceFrame.begin(), m_groupResidentSinceFrame.end(), kInvalidFrameIndex);
        std::fill(m_groupPendingUnloadState.begin(), m_groupPendingUnloadState.end(), 0u);
        std::fill(m_pendingResidencyRequestFrames.begin(),
                  m_pendingResidencyRequestFrames.end(),
                  kInvalidFrameIndex);
        std::fill(m_residentTouchSeenScratch.begin(), m_residentTouchSeenScratch.end(), 0u);
        std::fill(m_unloadRequestSeenScratch.begin(), m_unloadRequestSeenScratch.end(), 0u);
        std::fill(m_patchLastWriteIndexScratch.begin(), m_patchLastWriteIndexScratch.end(), UINT32_MAX);
        m_patchTouchedGroupsScratch.clear();
        resetResidentHeapAllocator();
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        resetDebugStats();

        std::vector<uint32_t> alwaysResidentGroups;
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

            alwaysResidentGroups.clear();
            collectLeafGroupsForNode(alwaysResidentNode, clusterLodData, alwaysResidentGroups);
            for (uint32_t groupIndex : alwaysResidentGroups) {
                if (groupIndex >= m_residencyGroupCapacity) {
                    continue;
                }
                m_groupResidencyState[groupIndex] =
                    kClusterLodGroupResidencyResident | kClusterLodGroupResidencyAlwaysResident;
                m_groupAgeState[groupIndex] = 0u;
                m_groupResidentSinceFrame[groupIndex] = 0u;
                if (!queueLoadPatchForGroup(groupIndex, clusterLodData)) {
                    m_groupResidencyState[groupIndex] = 0u;
                    m_groupAgeState[groupIndex] = 0u;
                    m_groupResidentSinceFrame[groupIndex] = kInvalidFrameIndex;
                    continue;
                }
            }
        }

        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex)) {
                queueUnloadPatch(groupIndex);
            }
        }

        m_residencySourceNodeBufferHandle = clusterLodData.nodeBuffer.nativeHandle();
        m_residencySourceGroupBufferHandle = clusterLodData.groupBuffer.nativeHandle();
        m_residencySourceGroupMeshletIndicesHandle =
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        m_stateDirty = false;
    }

    void collectUnloadRequestCandidates() {
        for (uint32_t groupIndex = 0; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
                clearPendingUnloadCandidate(groupIndex);
                continue;
            }

            if (m_unloadRequestSeenScratch[groupIndex] == 0u) {
                clearPendingUnloadCandidate(groupIndex);
                continue;
            }

            if (m_groupPendingUnloadState[groupIndex] != 0u) {
                if (std::find(m_confirmedUnloadGroups.begin(),
                              m_confirmedUnloadGroups.end(),
                              groupIndex) == m_confirmedUnloadGroups.end()) {
                    m_confirmedUnloadGroups.push_back(groupIndex);
                }
            } else {
                m_groupPendingUnloadState[groupIndex] = 1u;
            }
        }
    }

    void runRequestReadbackStage(const ClusterLODData& clusterLodData) {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastUnloadRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        std::fill(m_residentTouchSeenScratch.begin(), m_residentTouchSeenScratch.end(), 0u);
        std::fill(m_unloadRequestSeenScratch.begin(), m_unloadRequestSeenScratch.end(), 0u);

        FrameBuffers& frameBuffers = activeFrameBuffers();
        const uint32_t expectedFrameIndex = frameBuffers.submittedFrameIndex;
        if (expectedFrameIndex == kInvalidFrameIndex) {
            return;
        }

        bool requestReadbackValid = true;
        if (frameBuffers.residencyRequestBuffer && frameBuffers.residencyRequestStateBuffer) {
            const uint32_t requestCapacity = static_cast<uint32_t>(
                frameBuffers.residencyRequestBuffer->size() / sizeof(ClusterResidencyRequest));
            const uint32_t requestCount = std::min<uint32_t>(
                GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                    frameBuffers.residencyRequestStateBuffer.get()),
                requestCapacity);

            ClusterResidencyRequest* requests = mappedRequests(frameBuffers.residencyRequestBuffer.get());
            if (requestCount > 0u && !requests) {
                requestReadbackValid = false;
            } else if (requests && requestCount > 0u) {
                m_requestReadbackScratch.assign(requests, requests + requestCount);
                for (const ClusterResidencyRequest& request : m_requestReadbackScratch) {
                    if (request.requestFrameIndex != expectedFrameIndex) {
                        requestReadbackValid = false;
                        break;
                    }
                }
                if (!requestReadbackValid) {
                    m_requestReadbackScratch.clear();
                }
            }

            m_debugStats.lastResidencyRequestCount =
                static_cast<uint32_t>(m_requestReadbackScratch.size());
            m_lastProcessedRequestFrameIndex = expectedFrameIndex;
            for (const ClusterResidencyRequest& request : m_requestReadbackScratch) {
                if (request.targetGroupIndex >= clusterLodData.totalGroupCount) {
                    continue;
                }
                if (isGroupAlwaysResident(request.targetGroupIndex)) {
                    continue;
                }
                if (isGroupResident(request.targetGroupIndex)) {
                    m_residentTouchSeenScratch[request.targetGroupIndex] = 1u;
                    continue;
                }
                ++m_loadRequestsThisFrame;
                enqueuePendingResidencyGroup(request.targetGroupIndex, request.requestFrameIndex);
            }
        }

        if (!requestReadbackValid) {
            m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
            m_loadRequestsThisFrame = 0u;
            return;
        }

        advanceResidentGroupAges(expectedFrameIndex);
        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (m_residentTouchSeenScratch[groupIndex] == 0u || !isGroupResident(groupIndex) ||
                isGroupAlwaysResident(groupIndex)) {
                continue;
            }

            m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
            touchDynamicResidentGroup(groupIndex);
        }

        bool unloadReadbackValid = true;
        if (frameBuffers.unloadRequestBuffer && frameBuffers.unloadRequestStateBuffer) {
            const uint32_t unloadCapacity = static_cast<uint32_t>(
                frameBuffers.unloadRequestBuffer->size() / sizeof(ClusterUnloadRequest));
            const uint32_t unloadCount = std::min<uint32_t>(
                GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                    frameBuffers.unloadRequestStateBuffer.get()),
                unloadCapacity);

            ClusterUnloadRequest* unloadRequests =
                mappedUnloadRequests(frameBuffers.unloadRequestBuffer.get());
            if (unloadCount > 0u && !unloadRequests) {
                unloadReadbackValid = false;
            } else if (unloadRequests && unloadCount > 0u) {
                m_unloadRequestReadbackScratch.assign(unloadRequests, unloadRequests + unloadCount);
                for (const ClusterUnloadRequest& request : m_unloadRequestReadbackScratch) {
                    if (request.requestFrameIndex != expectedFrameIndex) {
                        unloadReadbackValid = false;
                        break;
                    }
                }
                if (!unloadReadbackValid) {
                    m_unloadRequestReadbackScratch.clear();
                }
            }

            m_debugStats.lastUnloadRequestCount =
                static_cast<uint32_t>(m_unloadRequestReadbackScratch.size());
            for (const ClusterUnloadRequest& request : m_unloadRequestReadbackScratch) {
                const uint32_t groupIndex = request.targetGroupIndex;
                if (groupIndex >= clusterLodData.totalGroupCount || !isGroupResident(groupIndex) ||
                    isGroupAlwaysResident(groupIndex)) {
                    continue;
                }
                ++m_unloadRequestsThisFrame;
                m_unloadRequestSeenScratch[groupIndex] = 1u;
            }
        }

        if (!unloadReadbackValid) {
            m_unloadRequestsThisFrame = 0u;
            return;
        }

        collectUnloadRequestCandidates();
    }

    void runResidencyUpdateStage(const ClusterLODData& clusterLodData) {
        if (!m_enableStreaming) {
            return;
        }

        uint32_t remainingLoads = m_maxLoadsPerFrame;
        uint32_t remainingUnloads = m_maxUnloadsPerFrame;

        size_t confirmedIndex = 0;
        while (confirmedIndex < m_confirmedUnloadGroups.size()) {
            const uint32_t groupIndex = m_confirmedUnloadGroups[confirmedIndex];
            if (groupIndex >= m_residencyGroupCapacity ||
                !isGroupResident(groupIndex) ||
                isGroupAlwaysResident(groupIndex)) {
                m_confirmedUnloadGroups.erase(
                    m_confirmedUnloadGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(confirmedIndex));
                continue;
            }

            if (remainingUnloads == 0u) {
                break;
            }

            if (evictResidentGroup(groupIndex)) {
                --remainingUnloads;
            }

            m_confirmedUnloadGroups.erase(
                m_confirmedUnloadGroups.begin() +
                static_cast<std::vector<uint32_t>::difference_type>(confirmedIndex));
        }

        promotePendingResidencyGroups(clusterLodData, remainingLoads);
    }

    uint32_t configuredStreamingStorageCapacityElements() const {
        const uint64_t capacityElements =
            std::max<uint64_t>(1ull, m_streamingStorageCapacityBytes / sizeof(uint32_t));
        return static_cast<uint32_t>(std::min<uint64_t>(capacityElements, uint64_t(UINT32_MAX)));
    }

    uint64_t computeStreamingTransferCapacityBytes(const ClusterLODData& clusterLodData) const {
        const uint64_t sceneBytes =
            uint64_t(std::max<size_t>(1u, clusterLodData.groupMeshletIndices.size())) * sizeof(uint32_t);
        const uint64_t alwaysResidentBytes =
            uint64_t(computeAlwaysResidentClusterCapacity(clusterLodData)) * sizeof(uint32_t);
        return std::max(alwaysResidentBytes, std::min(sceneBytes, m_maxStreamingTransferBytes));
    }

    uint32_t computeAlwaysResidentClusterCapacity(const ClusterLODData& clusterLodData) const {
        const uint32_t totalGroupCount =
            std::max<uint32_t>(1u, static_cast<uint32_t>(clusterLodData.groups.size()));
        std::vector<uint8_t> alwaysResidentGroupSeen(totalGroupCount, 0u);
        std::vector<uint32_t> alwaysResidentGroups;
        uint64_t totalClusterCount = 0u;

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

            alwaysResidentGroups.clear();
            collectLeafGroupsForNode(alwaysResidentNode, clusterLodData, alwaysResidentGroups);
            for (uint32_t groupIndex : alwaysResidentGroups) {
                if (groupIndex >= clusterLodData.groups.size() ||
                    alwaysResidentGroupSeen[groupIndex] != 0u) {
                    continue;
                }

                alwaysResidentGroupSeen[groupIndex] = 1u;
                totalClusterCount += clusterLodData.groups[groupIndex].clusterCount;
            }
        }

        return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(1ull, totalClusterCount),
                                                        uint64_t(UINT32_MAX)));
    }

    uint32_t computeStreamingStorageCapacity(const ClusterLODData& clusterLodData) const {
        const uint32_t sceneClusterCapacity =
            std::max<uint32_t>(1u, static_cast<uint32_t>(clusterLodData.groupMeshletIndices.size()));
        const uint32_t configuredCapacity = configuredStreamingStorageCapacityElements();
        const uint32_t alwaysResidentCapacity = computeAlwaysResidentClusterCapacity(clusterLodData);
        return std::max(alwaysResidentCapacity, std::min(sceneClusterCapacity, configuredCapacity));
    }

    void applyGpuStreamingStatsToTelemetry() {
        m_streamingStats.gpuStatsValid = m_hasGpuStreamingStats;
        m_streamingStats.gpuStatsFrameIndex =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.frameIndex : UINT32_MAX;
        m_streamingStats.gpuUnloadRequestCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.unloadRequestCount : 0u;
        m_streamingStats.gpuAverageUnloadAge =
            (m_hasGpuStreamingStats && m_lastGpuStreamingStats.unloadRequestCount != 0u)
                ? float(double(m_lastGpuStreamingStats.unloadAgeSum) /
                        double(m_lastGpuStreamingStats.unloadRequestCount))
                : 0.0f;
        m_streamingStats.gpuAppliedPatchCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.appliedPatchCount : 0u;
        m_streamingStats.gpuCopiedBytes =
            m_hasGpuStreamingStats ? clusterStreamingGpuStatsCopiedBytes(m_lastGpuStreamingStats)
                                   : 0u;
    }

    void clearGpuStreamingStats() {
        m_lastGpuStreamingStats = {};
        m_hasGpuStreamingStats = false;
        applyGpuStreamingStatsToTelemetry();
    }

    void updateDebugStats(const ClusterLODData* clusterLodData = nullptr) {
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = m_streamingStorage.capacityElements();
        m_debugStats.residentHeapUsed = m_streamingStorage.usedElements();
        m_debugStats.dynamicResidentGroupCount =
            static_cast<uint32_t>(m_dynamicResidentGroups.size());
        m_debugStats.pendingResidencyGroupCount =
            static_cast<uint32_t>(m_pendingResidencyGroups.size());
        m_debugStats.pendingUnloadGroupCount = 0;
        for (uint8_t pendingState : m_groupPendingUnloadState) {
            m_debugStats.pendingUnloadGroupCount += pendingState != 0u ? 1u : 0u;
        }
        m_debugStats.confirmedUnloadGroupCount =
            static_cast<uint32_t>(m_confirmedUnloadGroups.size());
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
        m_debugStats.ageThreshold = m_ageThreshold;
        m_debugStats.streamingTaskCapacity = kStreamingTaskCount;
        m_debugStats.freeStreamingTaskCount = 0;
        m_debugStats.preparedStreamingTaskCount = 0;
        m_debugStats.transferSubmittedTaskCount = 0;
        m_debugStats.updateQueuedTaskCount = 0;
        m_debugStats.selectedTransferTaskIndex = m_transferTaskIndex;
        m_debugStats.selectedUpdateTaskIndex = m_updateTaskIndex;
        m_debugStats.selectedUpdatePatchCount = m_activeUpdatePatchCount;
        m_debugStats.selectedTransferBytes = streamingUploadBytesUsed();
        m_debugStats.selectedUpdateTransferWaitValue = m_activeUpdateTransferWaitValue;
        m_debugStats.resourcesReady = ready();

        for (const StreamingTask& task : m_streamingTasks) {
            switch (task.state) {
            case StreamingTaskState::Free:
                ++m_debugStats.freeStreamingTaskCount;
                break;
            case StreamingTaskState::Prepared:
                ++m_debugStats.preparedStreamingTaskCount;
                break;
            case StreamingTaskState::TransferSubmitted:
                ++m_debugStats.transferSubmittedTaskCount;
                break;
            case StreamingTaskState::UpdateQueued:
                ++m_debugStats.updateQueuedTaskCount;
                break;
            }
        }

        const uint32_t groupCount =
            std::min(m_debugStats.activeResidencyGroupCount, m_residencyGroupCapacity);
        uint32_t maxResidentAge = 0u;
        for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            const uint32_t state = m_groupResidencyState[groupIndex];
            if ((state & kClusterLodGroupResidencyResident) != 0u) {
                ++m_debugStats.lastResidentGroupCount;
                if (groupIndex < m_groupAgeState.size()) {
                    maxResidentAge = std::max(maxResidentAge, m_groupAgeState[groupIndex]);
                }
            }
            if ((state & kClusterLodGroupResidencyAlwaysResident) != 0u) {
                ++m_debugStats.lastAlwaysResidentGroupCount;
            }
        }

        const uint32_t residentClusterCount = m_streamingStorage.usedElements();
        const uint64_t storagePoolCapacityBytes =
            uint64_t(m_streamingStorage.capacityElements()) * sizeof(uint32_t);
        const uint64_t storagePoolUsedBytes =
            uint64_t(residentClusterCount) * sizeof(uint32_t);
        const uint64_t transferCapacityBytes = effectiveStreamingTransferCapacityBytes();

        m_streamingStats.residentGroupCount = m_debugStats.lastResidentGroupCount;
        m_streamingStats.residentClusterCount = residentClusterCount;
        m_streamingStats.alwaysResidentGroupCount = m_debugStats.lastAlwaysResidentGroupCount;
        m_streamingStats.dynamicResidentGroupCount = m_debugStats.dynamicResidentGroupCount;
        m_streamingStats.storagePoolCapacityBytes = storagePoolCapacityBytes;
        m_streamingStats.storagePoolUsedBytes = storagePoolUsedBytes;
        m_streamingStats.residentHeapCapacity = m_debugStats.residentHeapCapacity;
        m_streamingStats.residentHeapUsed = residentClusterCount;
        m_streamingStats.loadRequestsThisFrame = m_loadRequestsThisFrame;
        m_streamingStats.unloadRequestsThisFrame = m_unloadRequestsThisFrame;
        m_streamingStats.loadsExecutedThisFrame = m_debugStats.lastResidencyPromotedCount;
        m_streamingStats.unloadsExecutedThisFrame = m_debugStats.lastResidencyEvictedCount;
        m_streamingStats.loadsDeferredThisFrame = 0u;
        if (m_lastProcessedRequestFrameIndex != kInvalidFrameIndex) {
            for (uint32_t groupIndex : m_pendingResidencyGroups) {
                if (groupIndex < m_pendingResidencyRequestFrames.size() &&
                    m_pendingResidencyRequestFrames[groupIndex] == m_lastProcessedRequestFrameIndex) {
                    ++m_streamingStats.loadsDeferredThisFrame;
                }
            }
        }
        m_streamingStats.transferBytesThisFrame = m_debugStats.selectedTransferBytes;
        m_streamingStats.transferUtilization =
            transferCapacityBytes != 0u
                ? float(double(m_streamingStats.transferBytesThisFrame) /
                        double(transferCapacityBytes))
                : 0.0f;
        m_streamingStats.failedAllocations = m_failedAllocationsThisFrame;
        applyGpuStreamingStatsToTelemetry();
        m_streamingStats.ageHistogram.fill(0u);
        m_streamingStats.ageHistogramMaxAge = maxResidentAge;
        m_streamingStats.ageHistogramBucketWidth =
            std::max(1u,
                     (maxResidentAge + kStreamingAgeHistogramBucketCount) /
                         kStreamingAgeHistogramBucketCount);
        for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            if (!isGroupResident(groupIndex) || groupIndex >= m_groupAgeState.size()) {
                continue;
            }

            const uint32_t bucketIndex = std::min<uint32_t>(
                m_groupAgeState[groupIndex] / m_streamingStats.ageHistogramBucketWidth,
                kStreamingAgeHistogramBucketCount - 1u);
            ++m_streamingStats.ageHistogram[bucketIndex];
        }

        m_streamingStats.residentGroupsPerLod.clear();
        m_streamingStats.totalGroupsPerLod.clear();
        if (clusterLodData && !clusterLodData->levels.empty()) {
            uint32_t maxLodDepth = 0u;
            for (const ClusterLODLevel& level : clusterLodData->levels) {
                maxLodDepth = std::max(maxLodDepth, level.depth);
            }

            m_streamingStats.residentGroupsPerLod.assign(size_t(maxLodDepth) + 1u, 0u);
            m_streamingStats.totalGroupsPerLod.assign(size_t(maxLodDepth) + 1u, 0u);
            for (const ClusterLODLevel& level : clusterLodData->levels) {
                if (level.depth >= m_streamingStats.totalGroupsPerLod.size()) {
                    continue;
                }

                const uint32_t levelGroupBegin = std::min(level.groupStart, groupCount);
                const uint32_t levelGroupEnd = static_cast<uint32_t>(
                    std::min<uint64_t>(uint64_t(level.groupStart) + uint64_t(level.groupCount),
                                       uint64_t(groupCount)));
                for (uint32_t groupIndex = levelGroupBegin; groupIndex < levelGroupEnd; ++groupIndex) {
                    ++m_streamingStats.totalGroupsPerLod[level.depth];
                    if (isGroupResident(groupIndex)) {
                        ++m_streamingStats.residentGroupsPerLod[level.depth];
                    }
                }
            }
        }
    }

    bool m_enableStreaming = false;
    bool m_stateDirty = true;
    uint32_t m_activeFrameSlot = 0u;
    uint32_t m_streamingBudgetGroups = 256u;
    uint32_t m_residencyGroupCapacity = 0;
    uint32_t m_maxLoadsPerFrame = 128u;
    uint32_t m_maxUnloadsPerFrame = 256u;
    uint32_t m_ageThreshold = 16u;
    bool m_enableGpuStatsReadback = true;
    uint32_t m_frameIndex = 0u;
    uint32_t m_prepareTaskIndex = kInvalidTaskIndex;
    uint32_t m_transferTaskIndex = kInvalidTaskIndex;
    uint32_t m_updateTaskIndex = kInvalidTaskIndex;
    uint32_t m_activeUpdatePatchCount = 0u;
    uint64_t m_streamingStorageCapacityBytes = kDefaultStreamingStorageCapacityBytes;
    uint64_t m_maxStreamingTransferBytes = kDefaultMaxStreamingTransferBytes;
    uint64_t m_activeUpdateTransferWaitValue = 0u;
    uint64_t m_pendingUpdateGraphicsCompletionSerial = 0u;
    uint64_t m_nextTaskSerial = 1u;
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const void* m_residencySourceGroupBufferHandle = nullptr;
    const void* m_residencySourceGroupMeshletIndicesHandle = nullptr;
    std::array<FrameBuffers, kBufferedFrameCount> m_frameBuffers;
    std::unique_ptr<RhiBuffer> m_lodGroupPageTableBuffer;
    StreamingStorage m_streamingStorage;
    std::array<StreamingTask, kStreamingTaskCount> m_streamingTasks;
    std::vector<uint32_t> m_groupResidencyState;
    std::vector<uint32_t> m_groupAgeState;
    std::vector<uint32_t> m_groupResidentSinceFrame;
    std::vector<uint8_t> m_groupPendingUnloadState;
    std::vector<uint32_t> m_pendingResidencyRequestFrames;
    std::vector<uint8_t> m_residentTouchSeenScratch;
    std::vector<uint8_t> m_unloadRequestSeenScratch;
    std::vector<uint32_t> m_dynamicResidentGroups;
    std::vector<uint32_t> m_pendingResidencyGroups;
    std::vector<uint32_t> m_patchLastWriteIndexScratch;
    std::vector<uint32_t> m_patchTouchedGroupsScratch;
    std::vector<ClusterResidencyRequest> m_requestReadbackScratch;
    std::vector<ClusterUnloadRequest> m_unloadRequestReadbackScratch;
    std::vector<uint32_t> m_confirmedUnloadGroups;
    std::vector<GroupResidentAllocation> m_groupResidentAllocations;
    ClusterStreamingGpuStats m_lastGpuStreamingStats{};
    bool m_hasGpuStreamingStats = false;
    StreamingStats m_streamingStats;
    DebugStats m_debugStats;
    uint32_t m_loadRequestsThisFrame = 0u;
    uint32_t m_unloadRequestsThisFrame = 0u;
    uint32_t m_failedAllocationsThisFrame = 0u;
    uint32_t m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
};
