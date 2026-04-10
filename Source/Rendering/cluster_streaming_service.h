#pragma once

#include "cluster_lod_builder.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "gpu_driven_helpers.h"
#include "rhi_backend.h"
#include "rhi_resource_utils.h"
#include "streaming_storage.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class ClusterStreamingService {
public:
    static constexpr uint32_t kStreamingAgeHistogramBucketCount = 16u;
    static constexpr uint32_t kBudgetPresetCount = 5u;

    enum class BudgetPreset : uint8_t {
        Auto = 0,
        Low,
        Medium,
        High,
        Custom,
    };

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
        bool adaptiveBudgetEnabled = false;
        uint32_t configuredAgeThreshold = 0;
        uint32_t effectiveAgeThreshold = 0;
        float smoothedFailedAllocations = 0.0f;
        float smoothedStorageUtilization = 0.0f;
        uint32_t adaptiveBudgetAdjustmentCount = 0;
        bool gpuStatsValid = false;
        uint32_t gpuStatsFrameIndex = UINT32_MAX;
        uint32_t gpuUnloadRequestCount = 0;
        float gpuAverageUnloadAge = 0.0f;
        uint32_t gpuAppliedPatchCount = 0;
        uint64_t gpuCopiedBytes = 0u;
        uint32_t gpuErrorUpdateCount = 0u;
        uint32_t gpuErrorAgeFilterCount = 0u;
        uint32_t gpuErrorAllocationCount = 0u;
        uint32_t gpuErrorPageTableCount = 0u;
        bool gpuAgeFilterDispatchMissing = false;
        uint32_t gpuAgeFilterDispatchMissingFrameIndex = UINT32_MAX;
        bool cpuUnloadFallbackActive = false;
        uint32_t cpuUnloadFallbackGroupCount = 0u;
        uint32_t cpuUnloadFallbackFrameIndex = UINT32_MAX;
        bool graphicsTransferFallbackActive = false;
        uint32_t graphicsTransferFallbackFrameIndex = UINT32_MAX;
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

    struct MemoryBudgetInfo {
        bool available = false;
        uint32_t heapCount = 0;
        uint64_t totalBudgetBytes = 0u;
        uint64_t totalUsageBytes = 0u;
        uint64_t totalHeadroomBytes = 0u;
        uint64_t deviceLocalBudgetBytes = 0u;
        uint64_t deviceLocalUsageBytes = 0u;
        uint64_t deviceLocalHeadroomBytes = 0u;
        uint64_t targetStorageBytes = 0u;
    };

    void setStreamingEnabled(bool enabled) {
        if (m_enableStreaming == enabled) {
            return;
        }

        m_enableStreaming = enabled;
        resetAdaptiveBudgetState(true);
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
        setStreamingBudgetGroupsInternal(budgetGroups, true);
    }

    uint32_t streamingBudgetGroups() const { return m_streamingBudgetGroups; }

    void setBudgetPreset(BudgetPreset preset) {
        if (m_budgetPreset == preset && preset != BudgetPreset::Auto) {
            return;
        }

        m_budgetPreset = preset;
        if (preset != BudgetPreset::Custom) {
            applyBudgetPreset();
        }
    }

    BudgetPreset budgetPreset() const { return m_budgetPreset; }

    static const char* budgetPresetLabel(BudgetPreset preset) {
        switch (preset) {
        case BudgetPreset::Auto:
            return "Auto";
        case BudgetPreset::Low:
            return "Low";
        case BudgetPreset::Medium:
            return "Medium";
        case BudgetPreset::High:
            return "High";
        case BudgetPreset::Custom:
            return "Custom";
        }
        return "Custom";
    }

    void setMaxLoadsPerFrame(uint32_t maxLoadsPerFrame) {
        m_maxLoadsPerFrame = std::max(1u, maxLoadsPerFrame);
    }

    uint32_t maxLoadsPerFrame() const { return m_maxLoadsPerFrame; }

    void setMaxUnloadsPerFrame(uint32_t maxUnloadsPerFrame) {
        m_maxUnloadsPerFrame = std::max(1u, maxUnloadsPerFrame);
    }

    uint32_t maxUnloadsPerFrame() const { return m_maxUnloadsPerFrame; }

    void setAgeThreshold(uint32_t ageThreshold) {
        const uint32_t clampedAgeThreshold =
            std::clamp(ageThreshold, kMinStreamingAgeThreshold, kMaxStreamingAgeThreshold);
        m_configuredAgeThreshold = clampedAgeThreshold;
        m_ageThreshold = clampedAgeThreshold;
        resetAdaptiveBudgetState(false);
    }

    uint32_t ageThreshold() const { return m_ageThreshold; }
    uint32_t configuredAgeThreshold() const { return m_configuredAgeThreshold; }

    void setAdaptiveBudgetEnabled(bool enabled) {
        if (m_adaptiveBudgetEnabled == enabled) {
            return;
        }

        m_adaptiveBudgetEnabled = enabled;
        resetAdaptiveBudgetState(true);
    }

    bool adaptiveBudgetEnabled() const { return m_adaptiveBudgetEnabled; }

    void setStreamingStorageCapacityBytes(uint64_t capacityBytes) {
        setStreamingStorageCapacityBytesInternal(capacityBytes, true);
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

    void applyAutoMemoryBudget(MemoryBudgetInfo memoryBudgetInfo) {
        if (memoryBudgetInfo.available && memoryBudgetInfo.targetStorageBytes == 0u) {
            memoryBudgetInfo.targetStorageBytes =
                computeAutoStreamingStorageCapacityBytes(memoryBudgetInfo.deviceLocalHeadroomBytes);
        }

        m_memoryBudgetInfo = memoryBudgetInfo;
        if (m_budgetPreset == BudgetPreset::Auto) {
            applyBudgetPreset();
        }
    }

    const MemoryBudgetInfo& memoryBudgetInfo() const { return m_memoryBudgetInfo; }

    void markStateDirty() { m_stateDirty = true; }

    void resetForPipelineReload() {
        resetStreamingTasks();
        std::fill(m_groupPendingUnloadState.begin(), m_groupPendingUnloadState.end(), 0u);
        std::fill(m_pendingResidencyRequestFrames.begin(),
                  m_pendingResidencyRequestFrames.end(),
                  kInvalidFrameIndex);
        std::fill(m_residentTouchSeenScratch.begin(), m_residentTouchSeenScratch.end(), 0u);
        std::fill(m_unloadRequestSeenScratch.begin(), m_unloadRequestSeenScratch.end(), 0u);
        std::fill(m_patchLastWriteIndexScratch.begin(), m_patchLastWriteIndexScratch.end(), UINT32_MAX);
        m_patchTouchedGroupsScratch.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        m_loadRequestsThisFrame = 0u;
        m_unloadRequestsThisFrame = 0u;
        m_failedAllocationsThisFrame = 0u;
        m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
        m_residencySourceNodeBufferHandle = nullptr;
        m_residencySourceGroupBufferHandle = nullptr;
        m_residencySourceGroupMeshletIndicesHandle = nullptr;
        uploadCanonicalStateToAllFrames();
        clearGpuStreamingStats();
        m_stateDirty = true;
        updateDebugStats();
    }

    const DebugStats& debugStats() const { return m_debugStats; }
    const StreamingStats& streamingStats() const { return m_streamingStats; }
    void ingestGpuStreamingStats(const ClusterStreamingGpuStats& stats) {
        if (stats.frameIndex == UINT32_MAX) {
            m_lastGpuStreamingStats = {};
            m_lastLoggedGpuErrorStats = {};
            m_hasGpuStreamingStats = false;
            applyGpuStreamingStatsToTelemetry();
            return;
        }

        if (m_hasGpuStreamingStats && stats.frameIndex < m_lastGpuStreamingStats.frameIndex) {
            return;
        }

        m_lastGpuStreamingStats = stats;
        m_hasGpuStreamingStats = true;
        maybeLogGpuStreamingErrors(stats);
        applyGpuStreamingStatsToTelemetry();
    }

    bool ready() const {
        for (const FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer ||
                !frameBuffers.groupAgeBuffer ||
                !frameBuffers.activeResidentGroupsBuffer ||
                !frameBuffers.activeResidentPatchBuffer ||
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

    void markGpuAgeFilterDispatched(uint32_t frameIndex) {
        activeFrameBuffers().gpuAgeFilterDispatchFrameIndex = frameIndex;
    }

    void markGraphicsTransferFallbackUsed(uint32_t frameIndex) {
        m_graphicsTransferFallbackActive = true;
        m_graphicsTransferFallbackFrameIndex = frameIndex;
        m_streamingStats.graphicsTransferFallbackActive = true;
        m_streamingStats.graphicsTransferFallbackFrameIndex = frameIndex;
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
    const RhiBuffer* activeResidentGroupsBuffer() const {
        return activeFrameBuffers().activeResidentGroupsBuffer.get();
    }
    uint32_t activeResidentGroupCount() const {
        return m_activeFrameResidentGroupCount;
    }
    const RhiBuffer* activeResidentPatchBuffer() const {
        return activeFrameBuffers().activeResidentPatchBuffer.get();
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
    uint32_t activeResidentPatchCount() const {
        return m_activeActiveResidentPatchCount;
    }
    const StreamingPatch* streamingPatchData() const {
        if (!validTaskIndex(m_updateTaskIndex)) {
            return nullptr;
        }

        const std::vector<StreamingPatch>& patches = m_streamingTasks[m_updateTaskIndex].patches;
        return patches.empty() ? nullptr : patches.data();
    }
    const ActiveResidentGroupPatch* activeResidentPatchData() const {
        if (!validTaskIndex(m_updateTaskIndex)) {
            return nullptr;
        }

        const std::vector<ActiveResidentGroupPatch>& patches =
            m_streamingTasks[m_updateTaskIndex].activeResidentPatches;
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
        m_activeActiveResidentPatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;
        m_activeFrameResidentGroupCount = 0u;
        m_loadRequestsThisFrame = 0u;
        m_unloadRequestsThisFrame = 0u;
        m_failedAllocationsThisFrame = 0u;
        m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
        resetDegradationTelemetryForFrame();
        switchSceneBudgetState(clusterLodData);

        const bool clusterLodAvailable =
            clusterLodData.nodeBuffer.nativeHandle() &&
            clusterLodData.groupBuffer.nativeHandle() &&
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle() &&
            clusterLodData.boundsBuffer.nativeHandle();
        if (!clusterLodAvailable) {
            resetAdaptiveBudgetState(true);
            resetDebugStats();
            return;
        }

        ensureStreamingResources(clusterLodData, runtimeContext);
        if (!ready()) {
            resetAdaptiveBudgetState(true);
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
        updateAdaptiveBudget();
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
    static constexpr uint64_t kAutoStreamingStorageAlignmentBytes = 16ull * 1024ull * 1024ull;
    static constexpr uint64_t kAutoStreamingStorageMaxBytes = 2ull * 1024ull * 1024ull * 1024ull;
    static constexpr uint64_t kLowPresetStreamingStorageCapacityBytes = 256ull * 1024ull * 1024ull;
    static constexpr uint64_t kMediumPresetStreamingStorageCapacityBytes = 512ull * 1024ull * 1024ull;
    static constexpr uint64_t kHighPresetStreamingStorageCapacityBytes = 1024ull * 1024ull * 1024ull;
    static constexpr uint32_t kLowPresetStreamingBudgetGroups = 64u;
    static constexpr uint32_t kMediumPresetStreamingBudgetGroups = 256u;
    static constexpr uint32_t kHighPresetStreamingBudgetGroups = 1024u;
    static constexpr uint32_t kMinStreamingAgeThreshold = 1u;
    static constexpr uint32_t kMaxStreamingAgeThreshold = 256u;
    static constexpr uint32_t kAdaptiveBudgetSmoothingWindowFrames = 24u;
    static constexpr uint32_t kAdaptiveBudgetEvaluationIntervalFrames = 8u;
    static constexpr float kAdaptiveBudgetFailedAllocationThreshold = 0.10f;
    static constexpr float kAdaptiveBudgetRelaxUtilizationThreshold = 0.50f;

    struct SceneBudgetSettings {
        BudgetPreset preset = BudgetPreset::Auto;
        uint32_t streamingBudgetGroups = kMediumPresetStreamingBudgetGroups;
        uint64_t streamingStorageCapacityBytes = kDefaultStreamingStorageCapacityBytes;
    };

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
        uint32_t activeResidentGroupCountAfter = 0u;
        std::vector<StreamingPatch> patches;
        std::vector<ActiveResidentGroupPatch> activeResidentPatches;
        std::vector<uint32_t> activeResidentGroupsBefore;
    };

    struct GroupResidentAllocation {
        uint32_t heapOffset = kInvalidResidentHeapOffset;
        uint32_t heapCount = 0;
    };

    struct FrameBuffers {
        std::unique_ptr<RhiBuffer> groupResidencyBuffer;
        std::unique_ptr<RhiBuffer> groupAgeBuffer;
        std::unique_ptr<RhiBuffer> activeResidentGroupsBuffer;
        std::unique_ptr<RhiBuffer> activeResidentPatchBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestStateBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestStateBuffer;
        std::unique_ptr<RhiBuffer> streamingStatsBuffer;
        std::unique_ptr<RhiBuffer> streamingPatchBuffer;
        uint32_t submittedFrameIndex = kInvalidFrameIndex;
        uint32_t gpuAgeFilterDispatchFrameIndex = kInvalidFrameIndex;
    };

    static void resetFrameBuffers(FrameBuffers& frameBuffers) {
        frameBuffers.groupResidencyBuffer.reset();
        frameBuffers.groupAgeBuffer.reset();
        frameBuffers.activeResidentGroupsBuffer.reset();
        frameBuffers.activeResidentPatchBuffer.reset();
        frameBuffers.residencyRequestBuffer.reset();
        frameBuffers.residencyRequestStateBuffer.reset();
        frameBuffers.unloadRequestBuffer.reset();
        frameBuffers.unloadRequestStateBuffer.reset();
        frameBuffers.streamingStatsBuffer.reset();
        frameBuffers.streamingPatchBuffer.reset();
        frameBuffers.submittedFrameIndex = kInvalidFrameIndex;
        frameBuffers.gpuAgeFilterDispatchFrameIndex = kInvalidFrameIndex;
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
        task.activeResidentGroupCountAfter = 0u;
        task.patches.clear();
        task.activeResidentPatches.clear();
        task.activeResidentGroupsBefore.clear();

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
        m_activeActiveResidentPatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;
        m_activeFrameResidentGroupCount = 0u;
        m_pendingUpdateGraphicsCompletionSerial = 0u;
        m_prepareTaskActiveResidentGroupsScratch.clear();
        std::fill(m_prepareTaskActiveResidentGroupIndexScratch.begin(),
                  m_prepareTaskActiveResidentGroupIndexScratch.end(),
                  UINT32_MAX);
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
            assignCurrentActiveResidentGroups(task.activeResidentGroupsBefore);
            task.activeResidentGroupCountAfter =
                static_cast<uint32_t>(std::min<size_t>(task.activeResidentGroupsBefore.size(),
                                                      size_t(m_residencyGroupCapacity)));
            task.patches.clear();
            task.activeResidentPatches.clear();
            initializePrepareTaskActiveResidentState(task.activeResidentGroupsBefore);
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

    static ActiveResidentGroupPatch* mappedActiveResidentPatches(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<ActiveResidentGroupPatch*>(rhiBufferContents(*buffer));
    }

    static ClusterStreamingGpuStats* mappedStreamingStats(RhiBuffer* buffer) {
        if (!buffer) {
            return nullptr;
        }
        return static_cast<ClusterStreamingGpuStats*>(rhiBufferContents(*buffer));
    }

    void assignCurrentActiveResidentGroups(std::vector<uint32_t>& outGroups) const {
        outGroups.clear();
        const size_t totalCount = std::min<size_t>(m_alwaysResidentGroups.size() +
                                                       m_dynamicResidentGroups.size(),
                                                   size_t(m_residencyGroupCapacity));
        outGroups.reserve(totalCount);

        for (uint32_t groupIndex : m_alwaysResidentGroups) {
            if (outGroups.size() >= totalCount) {
                break;
            }
            outGroups.push_back(groupIndex);
        }
        for (uint32_t groupIndex : m_dynamicResidentGroups) {
            if (outGroups.size() >= totalCount) {
                break;
            }
            outGroups.push_back(groupIndex);
        }
    }

    void initializePrepareTaskActiveResidentState(const std::vector<uint32_t>& activeResidentGroups) {
        m_prepareTaskActiveResidentGroupsScratch = activeResidentGroups;
        if (m_prepareTaskActiveResidentGroupIndexScratch.size() != m_residencyGroupCapacity) {
            m_prepareTaskActiveResidentGroupIndexScratch.assign(m_residencyGroupCapacity, UINT32_MAX);
        } else {
            std::fill(m_prepareTaskActiveResidentGroupIndexScratch.begin(),
                      m_prepareTaskActiveResidentGroupIndexScratch.end(),
                      UINT32_MAX);
        }

        const uint32_t activeResidentGroupCount = static_cast<uint32_t>(std::min<size_t>(
            m_prepareTaskActiveResidentGroupsScratch.size(), size_t(m_residencyGroupCapacity)));
        for (uint32_t activeResidentIndex = 0u; activeResidentIndex < activeResidentGroupCount;
             ++activeResidentIndex) {
            const uint32_t groupIndex = m_prepareTaskActiveResidentGroupsScratch[activeResidentIndex];
            if (groupIndex >= m_prepareTaskActiveResidentGroupIndexScratch.size()) {
                continue;
            }
            m_prepareTaskActiveResidentGroupIndexScratch[groupIndex] = activeResidentIndex;
        }
    }

    void uploadActiveResidentGroupsToFrame(FrameBuffers& frameBuffers,
                                           const std::vector<uint32_t>& activeResidentGroups) {
        uint32_t* activeResidentWords = mappedUint32(frameBuffers.activeResidentGroupsBuffer.get());
        if (!activeResidentWords) {
            return;
        }

        std::memset(activeResidentWords,
                    0xFF,
                    size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        const uint32_t activeResidentGroupCount = static_cast<uint32_t>(std::min<size_t>(
            activeResidentGroups.size(), size_t(m_residencyGroupCapacity)));
        if (activeResidentGroupCount == 0u) {
            return;
        }

        std::memcpy(activeResidentWords,
                    activeResidentGroups.data(),
                    size_t(activeResidentGroupCount) * sizeof(uint32_t));
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
        resetDegradationTelemetryForFrame();
        m_cpuUnloadFallbackWasActive = false;
        applyAdaptiveBudgetTelemetry();
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

        RhiBufferDesc activeResidentGroupsDesc{};
        activeResidentGroupsDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        activeResidentGroupsDesc.hostVisible = true;
        const std::string activeResidentGroupsName =
            "ClusterLodActiveResidentGroups[" + std::to_string(frameSlot) + "]";
        activeResidentGroupsDesc.debugName = activeResidentGroupsName.c_str();
        frameBuffers.activeResidentGroupsBuffer =
            runtimeContext.resourceFactory->createBuffer(activeResidentGroupsDesc);

        RhiBufferDesc activeResidentPatchDesc{};
        activeResidentPatchDesc.size = size_t(groupCapacity) * sizeof(ActiveResidentGroupPatch);
        activeResidentPatchDesc.hostVisible = true;
        const std::string activeResidentPatchName =
            "ClusterLodActiveResidentPatches[" + std::to_string(frameSlot) + "]";
        activeResidentPatchDesc.debugName = activeResidentPatchName.c_str();
        frameBuffers.activeResidentPatchBuffer =
            runtimeContext.resourceFactory->createBuffer(activeResidentPatchDesc);

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
        m_alwaysResidentGroups.clear();
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        m_prepareTaskActiveResidentGroupsScratch.clear();
        m_prepareTaskActiveResidentGroupIndexScratch.assign(groupCapacity, UINT32_MAX);
        m_activeFrameResidentGroupCount = 0u;
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
                                     bool markSubmittedFrame,
                                     const std::vector<uint32_t>* activeResidentGroupsOverride = nullptr) {
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
        std::vector<uint32_t> activeResidentGroupsStorage;
        const std::vector<uint32_t>* activeResidentGroups = activeResidentGroupsOverride;
        if (!activeResidentGroups) {
            assignCurrentActiveResidentGroups(activeResidentGroupsStorage);
            activeResidentGroups = &activeResidentGroupsStorage;
        }
        uploadActiveResidentGroupsToFrame(frameBuffers, *activeResidentGroups);
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
        frameBuffers.gpuAgeFilterDispatchFrameIndex = kInvalidFrameIndex;
    }

    void uploadSelectedUpdateTaskToActiveFrame() {
        m_activeUpdatePatchCount = 0u;
        m_activeActiveResidentPatchCount = 0u;
        m_activeUpdateTransferWaitValue = 0u;

        if (!validTaskIndex(m_updateTaskIndex)) {
            return;
        }

        const StreamingTask& task = m_streamingTasks[m_updateTaskIndex];
        const uint32_t patchCount =
            std::min<uint32_t>(static_cast<uint32_t>(task.patches.size()), m_residencyGroupCapacity);
        const uint32_t activeResidentPatchCount = std::min<uint32_t>(
            static_cast<uint32_t>(task.activeResidentPatches.size()), m_residencyGroupCapacity);
        m_activeUpdatePatchCount = patchCount;
        m_activeActiveResidentPatchCount = activeResidentPatchCount;
        m_activeUpdateTransferWaitValue = task.transferWaitValue;
        if (patchCount == 0u && activeResidentPatchCount == 0u) {
            return;
        }

        StreamingPatch* patches = mappedPatches(activeFrameBuffers().streamingPatchBuffer.get());
        if (patches && patchCount != 0u) {
            std::memcpy(patches,
                        task.patches.data(),
                        size_t(patchCount) * sizeof(StreamingPatch));
        }

        ActiveResidentGroupPatch* activeResidentPatches =
            mappedActiveResidentPatches(activeFrameBuffers().activeResidentPatchBuffer.get());
        if (activeResidentPatches && activeResidentPatchCount != 0u) {
            std::memcpy(activeResidentPatches,
                        task.activeResidentPatches.data(),
                        size_t(activeResidentPatchCount) * sizeof(ActiveResidentGroupPatch));
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
        const std::vector<uint32_t>* activeResidentGroupsOverride = nullptr;
        m_activeFrameResidentGroupCount = 0u;
        if (validTaskIndex(m_updateTaskIndex)) {
            const StreamingTask& task = m_streamingTasks[m_updateTaskIndex];
            activeResidentGroupsOverride = &task.activeResidentGroupsBefore;
            m_activeFrameResidentGroupCount = task.activeResidentGroupCountAfter;
        } else {
            std::vector<uint32_t> currentActiveResidentGroups;
            assignCurrentActiveResidentGroups(currentActiveResidentGroups);
            m_activeFrameResidentGroupCount = static_cast<uint32_t>(std::min<size_t>(
                currentActiveResidentGroups.size(), size_t(m_residencyGroupCapacity)));
            uploadCanonicalStateToFrame(activeFrameBuffers(),
                                        true,
                                        true,
                                        &currentActiveResidentGroups);
            uploadSelectedUpdateTaskToActiveFrame();
            return;
        }
        uploadCanonicalStateToFrame(activeFrameBuffers(),
                                    true,
                                    true,
                                    activeResidentGroupsOverride);
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

        std::stable_partition(
            task.patches.begin(),
            task.patches.end(),
            [](const StreamingPatch& patch) {
                return !isClusterLodGroupPageAddressValid(patch.residentHeapOffset);
            });
        task.activeResidentPatches.clear();
        task.activeResidentGroupCountAfter = static_cast<uint32_t>(std::min<size_t>(
            m_prepareTaskActiveResidentGroupsScratch.size(), size_t(m_residencyGroupCapacity)));
        const uint32_t activeResidentGroupCountBefore = static_cast<uint32_t>(std::min<size_t>(
            task.activeResidentGroupsBefore.size(), size_t(m_residencyGroupCapacity)));
        const uint32_t diffCount =
            std::max(activeResidentGroupCountBefore, task.activeResidentGroupCountAfter);
        task.activeResidentPatches.reserve(diffCount);
        for (uint32_t activeResidentIndex = 0u; activeResidentIndex < diffCount; ++activeResidentIndex) {
            const uint32_t beforeGroupIndex =
                activeResidentIndex < activeResidentGroupCountBefore
                    ? task.activeResidentGroupsBefore[activeResidentIndex]
                    : UINT32_MAX;
            const uint32_t afterGroupIndex =
                activeResidentIndex < task.activeResidentGroupCountAfter
                    ? m_prepareTaskActiveResidentGroupsScratch[activeResidentIndex]
                    : UINT32_MAX;
            if (beforeGroupIndex == afterGroupIndex) {
                continue;
            }

            ActiveResidentGroupPatch activeResidentPatch{};
            activeResidentPatch.activeResidentIndex = activeResidentIndex;
            activeResidentPatch.groupIndex = afterGroupIndex;
            task.activeResidentPatches.push_back(activeResidentPatch);
        }

        for (uint32_t groupIndex : m_patchTouchedGroupsScratch) {
            m_patchLastWriteIndexScratch[groupIndex] = UINT32_MAX;
        }
        m_patchTouchedGroupsScratch.clear();
        m_prepareTaskActiveResidentGroupsScratch.clear();
        std::fill(m_prepareTaskActiveResidentGroupIndexScratch.begin(),
                  m_prepareTaskActiveResidentGroupIndexScratch.end(),
                  UINT32_MAX);

        if (task.patches.empty() &&
            task.activeResidentPatches.empty() &&
            !taskHasTransferWork(m_prepareTaskIndex)) {
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
        if (groupIndex < m_prepareTaskActiveResidentGroupIndexScratch.size() &&
            m_prepareTaskActiveResidentGroupIndexScratch[groupIndex] == UINT32_MAX &&
            m_prepareTaskActiveResidentGroupsScratch.size() < size_t(m_residencyGroupCapacity)) {
            m_prepareTaskActiveResidentGroupIndexScratch[groupIndex] =
                static_cast<uint32_t>(m_prepareTaskActiveResidentGroupsScratch.size());
            m_prepareTaskActiveResidentGroupsScratch.push_back(groupIndex);
        }
        appendStreamingPatch(patch);
    }

    void queueUnloadPatch(uint32_t groupIndex) {
        StreamingPatch patch{};
        patch.groupIndex = groupIndex;
        patch.residentHeapOffset = makeClusterLodGroupPageInvalidAddress(m_frameIndex);
        patch.clusterStart = 0u;
        patch.clusterCount = 0u;
        if (groupIndex < m_prepareTaskActiveResidentGroupIndexScratch.size()) {
            const uint32_t activeResidentIndex =
                m_prepareTaskActiveResidentGroupIndexScratch[groupIndex];
            if (activeResidentIndex != UINT32_MAX &&
                activeResidentIndex < m_prepareTaskActiveResidentGroupsScratch.size()) {
                const uint32_t lastActiveResidentIndex = static_cast<uint32_t>(
                    m_prepareTaskActiveResidentGroupsScratch.size() - 1u);
                if (activeResidentIndex != lastActiveResidentIndex) {
                    const uint32_t movedGroupIndex =
                        m_prepareTaskActiveResidentGroupsScratch[lastActiveResidentIndex];
                    if (movedGroupIndex < m_prepareTaskActiveResidentGroupIndexScratch.size()) {
                        m_prepareTaskActiveResidentGroupIndexScratch[movedGroupIndex] =
                            activeResidentIndex;
                    }
                    m_prepareTaskActiveResidentGroupsScratch[activeResidentIndex] = movedGroupIndex;
                }
                m_prepareTaskActiveResidentGroupIndexScratch[groupIndex] = UINT32_MAX;
                m_prepareTaskActiveResidentGroupsScratch.pop_back();
            }
        }
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
        m_alwaysResidentGroups.clear();
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
                if ((m_groupResidencyState[groupIndex] & kClusterLodGroupResidencyAlwaysResident) != 0u) {
                    continue;
                }
                m_groupResidencyState[groupIndex] =
                    kClusterLodGroupResidencyResident | kClusterLodGroupResidencyAlwaysResident;
                m_groupAgeState[groupIndex] = 0u;
                m_groupResidentSinceFrame[groupIndex] = 0u;
            }
        }

        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex)) {
                queueUnloadPatch(groupIndex);
            }
        }

        for (uint32_t groupIndex = 0u; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupAlwaysResident(groupIndex)) {
                continue;
            }
            if (!queueLoadPatchForGroup(groupIndex, clusterLodData)) {
                m_groupResidencyState[groupIndex] = 0u;
                m_groupAgeState[groupIndex] = 0u;
                m_groupResidentSinceFrame[groupIndex] = kInvalidFrameIndex;
                continue;
            }
            m_alwaysResidentGroups.push_back(groupIndex);
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

    bool shouldUseCpuFifoUnloadFallback(const FrameBuffers& frameBuffers,
                                        uint32_t expectedFrameIndex) const {
        if (!m_enableStreaming ||
            expectedFrameIndex == kInvalidFrameIndex ||
            frameBuffers.gpuAgeFilterDispatchFrameIndex == expectedFrameIndex ||
            m_dynamicResidentGroups.empty()) {
            return false;
        }

        const bool overBudget =
            m_streamingBudgetGroups != 0u &&
            m_dynamicResidentGroups.size() > size_t(m_streamingBudgetGroups);
        const bool atBudget =
            m_streamingBudgetGroups != 0u &&
            m_dynamicResidentGroups.size() >= size_t(m_streamingBudgetGroups);
        const bool pendingLoadPressure = !m_pendingResidencyGroups.empty() && atBudget;
        const bool storagePressure =
            !m_pendingResidencyGroups.empty() &&
            m_streamingStorage.capacityElements() != 0u &&
            m_streamingStorage.usedElements() >= m_streamingStorage.capacityElements();
        const bool recentAllocationPressure =
            !m_pendingResidencyGroups.empty() &&
            (m_failedAllocationsThisFrame != 0u || m_smoothedFailedAllocations > 0.0f);

        return overBudget || pendingLoadPressure || storagePressure || recentAllocationPressure;
    }

    uint32_t collectCpuFifoUnloadFallbackCandidates(const FrameBuffers& frameBuffers,
                                                   uint32_t expectedFrameIndex) {
        m_gpuAgeFilterDispatchMissing =
            frameBuffers.gpuAgeFilterDispatchFrameIndex != expectedFrameIndex;
        m_gpuAgeFilterDispatchMissingFrameIndex =
            m_gpuAgeFilterDispatchMissing ? expectedFrameIndex : kInvalidFrameIndex;
        if (!m_gpuAgeFilterDispatchMissing) {
            m_cpuUnloadFallbackWasActive = false;
            return 0u;
        }

        if (!shouldUseCpuFifoUnloadFallback(frameBuffers, expectedFrameIndex)) {
            return 0u;
        }

        const uint32_t pendingLoadCount =
            static_cast<uint32_t>(std::min<size_t>(m_pendingResidencyGroups.size(), UINT32_MAX));
        const uint32_t overBudgetCount =
            m_dynamicResidentGroups.size() > size_t(m_streamingBudgetGroups)
                ? static_cast<uint32_t>(std::min<size_t>(
                      m_dynamicResidentGroups.size() - size_t(m_streamingBudgetGroups),
                      UINT32_MAX))
                : 0u;
        const uint32_t desiredUnloadCount =
            std::min(m_maxUnloadsPerFrame, std::max(1u, std::max(pendingLoadCount, overBudgetCount)));

        uint32_t queuedUnloadCount = 0u;
        for (uint32_t groupIndex : m_dynamicResidentGroups) {
            if (queuedUnloadCount >= desiredUnloadCount) {
                break;
            }
            if (groupIndex >= m_residencyGroupCapacity ||
                !isGroupResident(groupIndex) ||
                isGroupAlwaysResident(groupIndex)) {
                continue;
            }
            if (groupIndex < m_residentTouchSeenScratch.size() &&
                m_residentTouchSeenScratch[groupIndex] != 0u) {
                continue;
            }
            if (std::find(m_confirmedUnloadGroups.begin(),
                          m_confirmedUnloadGroups.end(),
                          groupIndex) != m_confirmedUnloadGroups.end()) {
                continue;
            }

            if (groupIndex < m_groupPendingUnloadState.size()) {
                m_groupPendingUnloadState[groupIndex] = 1u;
            }
            m_confirmedUnloadGroups.push_back(groupIndex);
            ++queuedUnloadCount;
        }

        if (queuedUnloadCount == 0u) {
            return 0u;
        }

        m_cpuUnloadFallbackActive = true;
        m_cpuUnloadFallbackFrameIndex = expectedFrameIndex;
        m_cpuUnloadFallbackGroupCount = queuedUnloadCount;
        m_unloadRequestsThisFrame += queuedUnloadCount;
        m_debugStats.lastUnloadRequestCount += queuedUnloadCount;

        if (!m_cpuUnloadFallbackWasActive) {
            spdlog::warn(
                "Cluster streaming age-filter dispatch missing for frame {}; using CPU FIFO unload fallback",
                expectedFrameIndex);
        }
        m_cpuUnloadFallbackWasActive = true;
        return queuedUnloadCount;
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

        ingestMappedGpuStreamingStats(frameBuffers, expectedFrameIndex);

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

        if (collectCpuFifoUnloadFallbackCandidates(frameBuffers, expectedFrameIndex) != 0u) {
            return;
        }
        if (frameBuffers.gpuAgeFilterDispatchFrameIndex != expectedFrameIndex) {
            return;
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

    SceneBudgetSettings captureCurrentSceneBudgetSettings() const {
        SceneBudgetSettings settings;
        settings.preset = m_budgetPreset;
        settings.streamingBudgetGroups = m_streamingBudgetGroups;
        settings.streamingStorageCapacityBytes = m_streamingStorageCapacityBytes;
        return settings;
    }

    static SceneBudgetSettings defaultSceneBudgetSettings() {
        return {};
    }

    void applySceneBudgetSettings(const SceneBudgetSettings& settings) {
        if (settings.preset == BudgetPreset::Custom) {
            const bool wasApplyingBudgetPreset = m_applyingBudgetPreset;
            m_applyingBudgetPreset = true;
            setStreamingBudgetGroupsInternal(settings.streamingBudgetGroups, false);
            setStreamingStorageCapacityBytesInternal(settings.streamingStorageCapacityBytes, false);
            m_applyingBudgetPreset = wasApplyingBudgetPreset;
            m_budgetPreset = BudgetPreset::Custom;
            return;
        }

        m_budgetPreset = settings.preset;
        applyBudgetPreset();
    }

    void restoreSceneBudgetSettings(uint64_t sceneSignature) {
        SceneBudgetSettings settings = defaultSceneBudgetSettings();
        if (sceneSignature != 0u) {
            auto it = m_sceneBudgetSettings.find(sceneSignature);
            if (it != m_sceneBudgetSettings.end()) {
                settings = it->second;
            }
        }
        applySceneBudgetSettings(settings);
    }

    void switchSceneBudgetState(const ClusterLODData& clusterLodData) {
        const uint64_t sceneSignature = clusterLodData.sourceSceneSignature;
        if (sceneSignature == m_activeSceneSignature) {
            return;
        }

        if (m_activeSceneSignature != 0u) {
            m_sceneBudgetSettings[m_activeSceneSignature] = captureCurrentSceneBudgetSettings();
        }

        m_activeSceneSignature = sceneSignature;
        restoreSceneBudgetSettings(sceneSignature);
    }

    void setStreamingBudgetGroupsInternal(uint32_t budgetGroups, bool markCustomPreset) {
        if (m_streamingBudgetGroups == budgetGroups) {
            return;
        }

        m_streamingBudgetGroups = budgetGroups;
        if (markCustomPreset && !m_applyingBudgetPreset) {
            m_budgetPreset = BudgetPreset::Custom;
        }
    }

    void setStreamingStorageCapacityBytesInternal(uint64_t capacityBytes, bool markCustomPreset) {
        capacityBytes = std::max<uint64_t>(capacityBytes, sizeof(uint32_t));
        if (m_streamingStorageCapacityBytes == capacityBytes) {
            return;
        }

        m_streamingStorageCapacityBytes = capacityBytes;
        if (markCustomPreset && !m_applyingBudgetPreset) {
            m_budgetPreset = BudgetPreset::Custom;
        }
        resetAdaptiveBudgetState(false);
        m_stateDirty = true;
    }

    static uint32_t computeAutoStreamingBudgetGroups(uint64_t targetStorageBytes) {
        static constexpr uint64_t kLowToMediumThresholdBytes =
            (kLowPresetStreamingStorageCapacityBytes + kMediumPresetStreamingStorageCapacityBytes) / 2ull;
        static constexpr uint64_t kMediumToHighThresholdBytes =
            (kMediumPresetStreamingStorageCapacityBytes + kHighPresetStreamingStorageCapacityBytes) / 2ull;
        if (targetStorageBytes < kLowToMediumThresholdBytes) {
            return kLowPresetStreamingBudgetGroups;
        }
        if (targetStorageBytes < kMediumToHighThresholdBytes) {
            return kMediumPresetStreamingBudgetGroups;
        }
        return kHighPresetStreamingBudgetGroups;
    }

    void applyBudgetPreset() {
        uint64_t targetStorageBytes = m_streamingStorageCapacityBytes;
        uint32_t targetBudgetGroups = m_streamingBudgetGroups;

        switch (m_budgetPreset) {
        case BudgetPreset::Auto:
            targetStorageBytes =
                m_memoryBudgetInfo.targetStorageBytes != 0u
                    ? m_memoryBudgetInfo.targetStorageBytes
                    : kDefaultStreamingStorageCapacityBytes;
            targetBudgetGroups = computeAutoStreamingBudgetGroups(targetStorageBytes);
            break;
        case BudgetPreset::Low:
            targetStorageBytes = kLowPresetStreamingStorageCapacityBytes;
            targetBudgetGroups = kLowPresetStreamingBudgetGroups;
            break;
        case BudgetPreset::Medium:
            targetStorageBytes = kMediumPresetStreamingStorageCapacityBytes;
            targetBudgetGroups = kMediumPresetStreamingBudgetGroups;
            break;
        case BudgetPreset::High:
            targetStorageBytes = kHighPresetStreamingStorageCapacityBytes;
            targetBudgetGroups = kHighPresetStreamingBudgetGroups;
            break;
        case BudgetPreset::Custom:
            return;
        }

        m_applyingBudgetPreset = true;
        setStreamingBudgetGroupsInternal(targetBudgetGroups, false);
        setStreamingStorageCapacityBytesInternal(targetStorageBytes, false);
        m_applyingBudgetPreset = false;
    }

    static uint64_t computeAutoStreamingStorageCapacityBytes(uint64_t deviceLocalHeadroomBytes) {
        uint64_t targetBytes = std::min(deviceLocalHeadroomBytes / 2u, kAutoStreamingStorageMaxBytes);
        if (targetBytes >= kAutoStreamingStorageAlignmentBytes) {
            targetBytes =
                (targetBytes / kAutoStreamingStorageAlignmentBytes) *
                kAutoStreamingStorageAlignmentBytes;
        }
        return std::max<uint64_t>(targetBytes, sizeof(uint32_t));
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
        m_streamingStats.gpuErrorUpdateCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.errorUpdate : 0u;
        m_streamingStats.gpuErrorAgeFilterCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.errorAgeFilter : 0u;
        m_streamingStats.gpuErrorAllocationCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.errorAllocation : 0u;
        m_streamingStats.gpuErrorPageTableCount =
            m_hasGpuStreamingStats ? m_lastGpuStreamingStats.errorPageTable : 0u;
    }

    void resetDegradationTelemetryForFrame() {
        m_gpuAgeFilterDispatchMissing = false;
        m_gpuAgeFilterDispatchMissingFrameIndex = kInvalidFrameIndex;
        m_cpuUnloadFallbackActive = false;
        m_cpuUnloadFallbackGroupCount = 0u;
        m_cpuUnloadFallbackFrameIndex = kInvalidFrameIndex;
        m_graphicsTransferFallbackActive = false;
        m_graphicsTransferFallbackFrameIndex = kInvalidFrameIndex;
    }

    void clearGpuStreamingStats() {
        m_lastGpuStreamingStats = {};
        m_lastLoggedGpuErrorStats = {};
        m_hasGpuStreamingStats = false;
        applyGpuStreamingStatsToTelemetry();
    }

    static bool sameGpuStreamingErrorCounts(const ClusterStreamingGpuStats& lhs,
                                            const ClusterStreamingGpuStats& rhs) {
        return lhs.errorUpdate == rhs.errorUpdate &&
               lhs.errorAgeFilter == rhs.errorAgeFilter &&
               lhs.errorAllocation == rhs.errorAllocation &&
               lhs.errorPageTable == rhs.errorPageTable;
    }

    void maybeLogGpuStreamingErrors(const ClusterStreamingGpuStats& stats) {
        if (!clusterStreamingGpuStatsHasErrors(stats)) {
            m_lastLoggedGpuErrorStats = {};
            return;
        }

        if (sameGpuStreamingErrorCounts(stats, m_lastLoggedGpuErrorStats)) {
            return;
        }

        spdlog::warn(
            "Cluster streaming GPU errors at frame {}: update={}, ageFilter={}, allocation={}, pageTable={}",
            stats.frameIndex,
            stats.errorUpdate,
            stats.errorAgeFilter,
            stats.errorAllocation,
            stats.errorPageTable);
        m_lastLoggedGpuErrorStats = stats;
    }

    void ingestMappedGpuStreamingStats(FrameBuffers& frameBuffers, uint32_t expectedFrameIndex) {
        if (!m_enableGpuStatsReadback || !frameBuffers.streamingStatsBuffer) {
            return;
        }

        ClusterStreamingGpuStats* stats =
            mappedStreamingStats(frameBuffers.streamingStatsBuffer.get());
        if (!stats || stats->frameIndex != expectedFrameIndex) {
            return;
        }

        ingestGpuStreamingStats(*stats);
    }

    uint32_t adaptiveMinAgeThreshold() const {
        return std::max(kMinStreamingAgeThreshold, m_configuredAgeThreshold / 4u);
    }

    uint32_t adaptiveMaxAgeThreshold() const {
        const uint64_t maxAgeThreshold =
            std::max<uint64_t>(m_configuredAgeThreshold,
                               uint64_t(m_configuredAgeThreshold) * 4ull);
        return static_cast<uint32_t>(
            std::min<uint64_t>(maxAgeThreshold, kMaxStreamingAgeThreshold));
    }

    void resetAdaptiveBudgetState(bool resetEffectiveAge) {
        m_adaptiveBudgetSmoothingInitialized = false;
        m_adaptiveBudgetEvaluationFrameCount = 0u;
        m_smoothedFailedAllocations = 0.0f;
        m_smoothedStorageUtilization = 0.0f;

        if (resetEffectiveAge) {
            m_ageThreshold = m_configuredAgeThreshold;
        } else {
            m_ageThreshold =
                std::clamp(m_ageThreshold, adaptiveMinAgeThreshold(), adaptiveMaxAgeThreshold());
        }

        applyAdaptiveBudgetTelemetry();
    }

    void updateAdaptiveBudget() {
        if (!m_adaptiveBudgetEnabled || !m_enableStreaming || !ready()) {
            resetAdaptiveBudgetState(true);
            return;
        }

        const uint32_t capacityElements = m_streamingStorage.capacityElements();
        const float storageUtilization =
            capacityElements != 0u
                ? float(double(m_streamingStorage.usedElements()) / double(capacityElements))
                : 0.0f;
        const float failedAllocations = static_cast<float>(m_failedAllocationsThisFrame);

        if (!m_adaptiveBudgetSmoothingInitialized) {
            m_smoothedFailedAllocations = failedAllocations;
            m_smoothedStorageUtilization = storageUtilization;
            m_adaptiveBudgetSmoothingInitialized = true;
        } else {
            constexpr float kSmoothingAlpha =
                1.0f / static_cast<float>(kAdaptiveBudgetSmoothingWindowFrames);
            m_smoothedFailedAllocations +=
                (failedAllocations - m_smoothedFailedAllocations) * kSmoothingAlpha;
            m_smoothedStorageUtilization +=
                (storageUtilization - m_smoothedStorageUtilization) * kSmoothingAlpha;
        }

        const uint32_t minAgeThreshold = adaptiveMinAgeThreshold();
        const uint32_t maxAgeThreshold = adaptiveMaxAgeThreshold();
        m_ageThreshold = std::clamp(m_ageThreshold, minAgeThreshold, maxAgeThreshold);

        uint32_t nextAgeThreshold = m_ageThreshold;
        if (m_failedAllocationsThisFrame != 0u) {
            constexpr uint32_t kImmediateFailureStep = 2u;
            nextAgeThreshold =
                m_ageThreshold > minAgeThreshold + kImmediateFailureStep
                    ? m_ageThreshold - kImmediateFailureStep
                    : minAgeThreshold;
            m_adaptiveBudgetEvaluationFrameCount = 0u;
        } else {
            ++m_adaptiveBudgetEvaluationFrameCount;
            if (m_adaptiveBudgetEvaluationFrameCount < kAdaptiveBudgetEvaluationIntervalFrames) {
                return;
            }
            m_adaptiveBudgetEvaluationFrameCount = 0u;

            if (m_smoothedFailedAllocations > kAdaptiveBudgetFailedAllocationThreshold) {
                nextAgeThreshold =
                    m_ageThreshold > minAgeThreshold ? m_ageThreshold - 1u : minAgeThreshold;
            } else if (m_smoothedStorageUtilization < kAdaptiveBudgetRelaxUtilizationThreshold) {
                nextAgeThreshold = std::min(maxAgeThreshold, m_ageThreshold + 1u);
            }
        }

        if (nextAgeThreshold != m_ageThreshold) {
            m_ageThreshold = nextAgeThreshold;
            ++m_adaptiveBudgetAdjustmentCount;
        }
    }

    void applyAdaptiveBudgetTelemetry() {
        m_streamingStats.adaptiveBudgetEnabled = m_adaptiveBudgetEnabled;
        m_streamingStats.configuredAgeThreshold = m_configuredAgeThreshold;
        m_streamingStats.effectiveAgeThreshold = m_ageThreshold;
        m_streamingStats.smoothedFailedAllocations = m_smoothedFailedAllocations;
        m_streamingStats.smoothedStorageUtilization = m_smoothedStorageUtilization;
        m_streamingStats.adaptiveBudgetAdjustmentCount = m_adaptiveBudgetAdjustmentCount;
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
        m_streamingStats.gpuAgeFilterDispatchMissing = m_gpuAgeFilterDispatchMissing;
        m_streamingStats.gpuAgeFilterDispatchMissingFrameIndex =
            m_gpuAgeFilterDispatchMissingFrameIndex;
        m_streamingStats.cpuUnloadFallbackActive = m_cpuUnloadFallbackActive;
        m_streamingStats.cpuUnloadFallbackGroupCount = m_cpuUnloadFallbackGroupCount;
        m_streamingStats.cpuUnloadFallbackFrameIndex = m_cpuUnloadFallbackFrameIndex;
        m_streamingStats.graphicsTransferFallbackActive = m_graphicsTransferFallbackActive;
        m_streamingStats.graphicsTransferFallbackFrameIndex = m_graphicsTransferFallbackFrameIndex;
        applyGpuStreamingStatsToTelemetry();
        applyAdaptiveBudgetTelemetry();
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
    BudgetPreset m_budgetPreset = BudgetPreset::Auto;
    uint32_t m_configuredAgeThreshold = 16u;
    uint32_t m_ageThreshold = 16u;
    bool m_adaptiveBudgetEnabled = true;
    bool m_enableGpuStatsReadback = true;
    uint32_t m_frameIndex = 0u;
    uint32_t m_prepareTaskIndex = kInvalidTaskIndex;
    uint32_t m_transferTaskIndex = kInvalidTaskIndex;
    uint32_t m_updateTaskIndex = kInvalidTaskIndex;
    uint32_t m_activeUpdatePatchCount = 0u;
    uint32_t m_activeActiveResidentPatchCount = 0u;
    uint32_t m_activeFrameResidentGroupCount = 0u;
    uint64_t m_streamingStorageCapacityBytes = kDefaultStreamingStorageCapacityBytes;
    uint64_t m_maxStreamingTransferBytes = kDefaultMaxStreamingTransferBytes;
    uint64_t m_activeUpdateTransferWaitValue = 0u;
    uint64_t m_pendingUpdateGraphicsCompletionSerial = 0u;
    uint64_t m_nextTaskSerial = 1u;
    uint64_t m_activeSceneSignature = 0u;
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const void* m_residencySourceGroupBufferHandle = nullptr;
    const void* m_residencySourceGroupMeshletIndicesHandle = nullptr;
    std::array<FrameBuffers, kBufferedFrameCount> m_frameBuffers;
    std::unique_ptr<RhiBuffer> m_lodGroupPageTableBuffer;
    StreamingStorage m_streamingStorage;
    std::array<StreamingTask, kStreamingTaskCount> m_streamingTasks;
    std::vector<uint32_t> m_alwaysResidentGroups;
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
    std::vector<uint32_t> m_prepareTaskActiveResidentGroupsScratch;
    std::vector<uint32_t> m_prepareTaskActiveResidentGroupIndexScratch;
    std::vector<ClusterResidencyRequest> m_requestReadbackScratch;
    std::vector<ClusterUnloadRequest> m_unloadRequestReadbackScratch;
    std::vector<uint32_t> m_confirmedUnloadGroups;
    std::vector<GroupResidentAllocation> m_groupResidentAllocations;
    ClusterStreamingGpuStats m_lastGpuStreamingStats{};
    ClusterStreamingGpuStats m_lastLoggedGpuErrorStats{};
    bool m_hasGpuStreamingStats = false;
    StreamingStats m_streamingStats;
    DebugStats m_debugStats;
    MemoryBudgetInfo m_memoryBudgetInfo;
    std::unordered_map<uint64_t, SceneBudgetSettings> m_sceneBudgetSettings;
    bool m_applyingBudgetPreset = false;
    bool m_adaptiveBudgetSmoothingInitialized = false;
    uint32_t m_adaptiveBudgetEvaluationFrameCount = 0u;
    uint32_t m_adaptiveBudgetAdjustmentCount = 0u;
    float m_smoothedFailedAllocations = 0.0f;
    float m_smoothedStorageUtilization = 0.0f;
    uint32_t m_loadRequestsThisFrame = 0u;
    uint32_t m_unloadRequestsThisFrame = 0u;
    uint32_t m_failedAllocationsThisFrame = 0u;
    uint32_t m_lastProcessedRequestFrameIndex = kInvalidFrameIndex;
    bool m_gpuAgeFilterDispatchMissing = false;
    uint32_t m_gpuAgeFilterDispatchMissingFrameIndex = kInvalidFrameIndex;
    bool m_cpuUnloadFallbackActive = false;
    bool m_cpuUnloadFallbackWasActive = false;
    uint32_t m_cpuUnloadFallbackGroupCount = 0u;
    uint32_t m_cpuUnloadFallbackFrameIndex = kInvalidFrameIndex;
    bool m_graphicsTransferFallbackActive = false;
    uint32_t m_graphicsTransferFallbackFrameIndex = kInvalidFrameIndex;
};
