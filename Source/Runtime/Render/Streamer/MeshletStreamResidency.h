#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/Profiling/RenderGraphProfile.h"
#include "Runtime/Render/Streamer/MeshletStreamLatency.h"
#include "Runtime/Render/Streamer/MeshletStreamPageLoader.h"
#include "Runtime/Render/Streamer/StreamingTaskQueue.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace metallic::render {

struct CpuProfileRecorder;

inline constexpr uint32_t kInvalidStreamDeviceOffsetBytes = UINT32_MAX;
inline constexpr uint64_t kMeshletStreamStorageAlignment = 256;

enum class MeshletStreamPageResidencyState : uint8_t {
    Unloaded,
    PendingUpload,
    Resident,
    LockedFallback,
    PendingUnload,
};

inline constexpr uint32_t kStreamPageStateMask = 0x00000007u;
inline constexpr uint32_t kStreamPageDeviceOffsetMask = ~kStreamPageStateMask;
inline constexpr uint64_t kStreamPageTableOffsetAlignment = kStreamPageStateMask + 1u;

struct StreamPageTableEntry {
    uint32_t deviceOffsetAndState = 0;
    uint32_t lastRequestFrame = 0;
};

inline constexpr uint32_t packStreamPageTableEntry(
    uint32_t deviceOffsetBytes,
    MeshletStreamPageResidencyState state)
{
    return (state == MeshletStreamPageResidencyState::Unloaded
            ? 0u
            : deviceOffsetBytes & kStreamPageDeviceOffsetMask) |
        static_cast<uint32_t>(state);
}

inline constexpr uint32_t streamPageTableDeviceOffset(const StreamPageTableEntry& entry)
{
    return (entry.deviceOffsetAndState & kStreamPageStateMask) ==
            static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded)
        ? kInvalidStreamDeviceOffsetBytes
        : entry.deviceOffsetAndState & kStreamPageDeviceOffsetMask;
}

inline constexpr MeshletStreamPageResidencyState streamPageTableState(const StreamPageTableEntry& entry)
{
    return static_cast<MeshletStreamPageResidencyState>(
        entry.deviceOffsetAndState & kStreamPageStateMask);
}

struct StreamPageTablePatch {
    uint32_t pageId = 0;
    uint32_t deviceOffsetAndState = 0;
};

inline constexpr uint32_t streamPageTablePatchDeviceOffset(const StreamPageTablePatch& patch)
{
    const StreamPageTableEntry entry{.deviceOffsetAndState = patch.deviceOffsetAndState};
    return streamPageTableDeviceOffset(entry);
}

inline constexpr MeshletStreamPageResidencyState streamPageTablePatchState(
    const StreamPageTablePatch& patch)
{
    const StreamPageTableEntry entry{.deviceOffsetAndState = patch.deviceOffsetAndState};
    return streamPageTableState(entry);
}

inline constexpr uint32_t kStreamRequestHeaderWordCount = 16;
inline constexpr uint32_t kStreamPrefetchPageTag = 1u << 31;
inline constexpr uint32_t kStreamUpdateHeaderWordCount = 16;

struct StreamRequestBufferHeader {
    uint32_t maxLoadRequests = 0;
    uint32_t maxUnloadRequests = 0;
    uint32_t loadCounter = 0;
    uint32_t unloadCounter = 0;
    uint32_t frameIndex = 0;
    uint32_t loadOverflowCounter = 0;
    uint32_t unloadOverflowCounter = 0;
    uint32_t invalidPageCounter = 0;
    uint32_t lastLoadOverflowFrame = 0;
    uint32_t lastUnloadOverflowFrame = 0;
    uint32_t lastInvalidPageFrame = 0;
    uint32_t loadPriorityOffset = 0; // Word offsets; zero preserves the legacy request ABI.
    uint32_t priorityTableOffset = 0;
    uint32_t prefetchRequestLimit = 0;
    uint32_t prefetchRequestCounter = 0;
    uint32_t prefetchDroppedCounter = 0;
};

struct StreamUpdateBufferHeader {
    uint32_t patchUnloadPageCount = 0;
    uint32_t patchPageCount = 0;
    uint32_t frameIndex = 0;
    uint32_t patchOverflowCounter = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
    uint32_t padding3 = 0;
    uint32_t padding4 = 0;
    uint32_t padding5 = 0;
    uint32_t padding6 = 0;
    uint32_t padding7 = 0;
    uint32_t padding8 = 0;
    uint32_t padding9 = 0;
    uint32_t padding10 = 0;
    uint32_t padding11 = 0;
};

static_assert(sizeof(StreamPageTableEntry) == 8);
static_assert(sizeof(StreamPageTablePatch) == 8);
static_assert(sizeof(StreamRequestBufferHeader) == kStreamRequestHeaderWordCount * sizeof(uint32_t));
static_assert(sizeof(StreamUpdateBufferHeader) == kStreamUpdateHeaderWordCount * sizeof(uint32_t));

struct StreamGpuRequestBatch {
    std::span<const uint32_t> loadPageIds;
    std::span<const uint32_t> unloadPageIds;
    uint32_t loadRequestCounter = 0;
    uint32_t unloadRequestCounter = 0;
    uint32_t loadOverflowCounter = 0;
    uint32_t unloadOverflowCounter = 0;
    uint32_t invalidPageCounter = 0;
    uint32_t frameIndex = 0;
    // The unload list enumerates resident pages unused by this GPU view. Keep
    // them cached and use the feedback for budget eviction, not eager unload.
    bool residentDemandFeedback = false;
    std::span<const float> loadPriorities; // Maximum screen benefit across instances, before byte cost.
    bool taggedPrefetchRequests = false;
};

struct MeshletStreamStorageAllocation {
    uint64_t offset = UINT64_MAX;
    uint64_t requestedSize = 0;
    uint64_t allocatedSize = 0;

    bool valid() const { return offset != UINT64_MAX && allocatedSize != 0; }
};

class MeshletStreamStorage {
public:
    bool initialize(
        uint64_t capacityBytes,
        uint64_t alignmentBytes,
        std::string& reason,
        uint64_t maxCapacityBytes = UINT32_MAX);
    void reset();

    MeshletStreamStorageAllocation allocate(uint64_t byteSize);
    void release(const MeshletStreamStorageAllocation& allocation);

    uint64_t allocationSize(uint64_t byteSize) const;
    bool canAllocate(uint64_t byteSize) const;

    uint64_t capacityBytes() const { return capacityBytes_; }
    uint64_t usedBytes() const { return usedBytes_; }
    uint64_t freeBytes() const { return capacityBytes_ >= usedBytes_ ? capacityBytes_ - usedBytes_ : 0; }
    uint64_t largestFreeBlockBytes() const;
    uint64_t alignmentBytes() const { return alignmentBytes_; }
    uint32_t allocationCount() const { return allocationCount_; }
    uint32_t freeBlockCount() const { return static_cast<uint32_t>(freeBlocks_.size()); }

private:
    struct FreeBlock {
        uint64_t offset = 0;
        uint64_t size = 0;
    };
    void updateFreeBlockBounds() const;

    uint64_t capacityBytes_ = 0;
    uint64_t alignmentBytes_ = kMeshletStreamStorageAlignment;
    uint64_t usedBytes_ = 0;
    uint32_t allocationCount_ = 0;
    std::vector<FreeBlock> freeBlocks_;
    mutable bool freeBlockBoundsValid_ = false;
    mutable uint64_t largestFreeBlockBytes_ = 0;
    mutable uint64_t largestAllocatableBytes_ = 0;
};

struct MeshletStreamResidencyDesc {
    const scene::MeshletStreamAsset* asset = nullptr;
    uint64_t maxResidentBytes = 0;
    uint32_t maxResidentPages = 0;
    uint32_t queuedFrameCount = 3;
    uint64_t pageStride = 0;
    uint64_t storageAlignment = kMeshletStreamStorageAlignment;
    uint32_t unloadDelayFrames = 1;
    uint32_t evictionAgeThresholdFrames = 1;
    uint32_t pageLoadConcurrency = 0;
    uint32_t maxPageLoadsInFlight = 0;
    bool measurePageLatency = false;
    bool immediateGpuRequests = false;
    bool completionDrivenUploads = true;
};

struct MeshletStreamResidencyStats {
    StreamCpuWorkCounters cpuWork;
    uint64_t frameIndex = 0;
    uint32_t pageCount = 0;
    uint32_t trackedPageCount = 0;
    uint32_t maxResidentPages = 0;
    uint64_t maxResidentBytes = 0;
    uint64_t usedResidentBytes = 0;
    uint64_t freeResidentBytes = 0;
    uint64_t largestFreeBlockBytes = 0;
    uint32_t storageAllocationCount = 0;
    uint32_t storageFreeBlockCount = 0;
    uint32_t usedSlotCount = 0;
    uint32_t freeSlotCount = 0;
    uint32_t activePageCount = 0;
    uint32_t residentPageCount = 0;
    uint32_t pendingPageCount = 0;
    uint32_t queuedUploadCount = 0;
    uint32_t pageLoadConcurrency = 0;
    uint32_t pendingPageLoadCount = 0;
    uint32_t activePageLoadCount = 0;
    uint32_t completedPageLoadCount = 0;
    uint32_t preparedPageLoadCount = 0;
    uint32_t queuedRequestTaskCount = 0;
    uint32_t availableRequestTaskCount = 0;
    uint32_t queuedStorageTaskCount = 0;
    uint32_t availableStorageTaskCount = 0;
    uint32_t queuedUnloadTaskCount = 0;
    uint32_t availableUnloadTaskCount = 0;
    uint32_t queuedUpdateTaskCount = 0;
    uint32_t availableUpdateTaskCount = 0;
    uint32_t pendingPatchCount = 0;
    uint32_t frameGpuRequestCount = 0;
    uint32_t frameUniqueGpuRequestCount = 0;
    uint32_t frameGpuUnloadRequestCount = 0;
    uint32_t frameUniqueGpuUnloadRequestCount = 0;
    uint32_t frameScheduledRequestTaskCount = 0;
    uint32_t frameCompletedRequestTaskCount = 0;
    uint32_t frameDroppedRequestTaskCount = 0;
    uint32_t frameRequestTaskFailureCount = 0;
    uint32_t frameConsumedGpuRequestCount = 0;
    uint32_t frameConsumedGpuUnloadRequestCount = 0;
    uint32_t frameGpuRequestOverflowCount = 0;
    uint32_t frameGpuUnloadRequestOverflowCount = 0;
    uint32_t frameGpuInvalidRequestCount = 0;
    uint32_t frameQueuedUploadCount = 0;
    uint32_t frameScheduledUploadCount = 0;
    uint32_t frameCompletedStorageTaskCount = 0;
    uint32_t frameScheduledUpdateCount = 0;
    uint32_t frameCompletedUpdateCount = 0;
    uint32_t frameCompletedUploadCount = 0;
    uint32_t frameStorageTaskFailureCount = 0;
    uint32_t frameUpdateTaskFailureCount = 0;
    uint32_t frameScheduledUnloadCount = 0;
    uint32_t frameCompletedUnloadCount = 0;
    uint32_t frameUnloadTaskFailureCount = 0;
    uint32_t frameDelayedFreeCount = 0;
    uint32_t frameEvictionAgeRejectedCount = 0;
    uint32_t frameResidentBudgetFailureCount = 0;
    uint32_t frameTransferBudgetFailureCount = 0;
    uint32_t frameEvictedPageCount = 0;
    uint32_t frameAllocationFailureCount = 0;
    uint32_t frameEvictionScanCount = 0;
    uint32_t frameEvictionCandidateTests = 0;
    uint32_t frameAllocationDeferredCount = 0;
    uint32_t frameCachedUnusedPageCount = 0;
    uint32_t frameResidentDemandCount = 0;
    uint64_t frameUploadBytes = 0;
    uint64_t totalUploadBytes = 0;
    uint32_t frameAdmissionDeferredCount = 0;
    uint32_t frameCancelledQueuedLoadCount = 0;
    uint64_t totalCancelledQueuedLoadCount = 0;
    uint32_t frameScheduledPageLoadCount = 0;
    uint32_t frameCompletedPageLoadCount = 0;
    uint32_t framePageLoadFailureCount = 0;
    uint64_t totalGpuRequestCount = 0;
    uint64_t totalUniqueGpuRequestCount = 0;
    uint64_t totalGpuUnloadRequestCount = 0;
    uint64_t totalUniqueGpuUnloadRequestCount = 0;
    uint64_t totalScheduledRequestTaskCount = 0;
    uint64_t totalCompletedRequestTaskCount = 0;
    uint64_t totalDroppedRequestTaskCount = 0;
    uint64_t totalRequestTaskFailureCount = 0;
    uint64_t totalConsumedGpuRequestCount = 0;
    uint64_t totalConsumedGpuUnloadRequestCount = 0;
    uint64_t totalGpuRequestOverflowCount = 0;
    uint64_t totalGpuUnloadRequestOverflowCount = 0;
    uint64_t totalGpuInvalidRequestCount = 0;
    uint64_t totalQueuedUploadCount = 0;
    uint64_t totalScheduledUploadCount = 0;
    uint64_t totalCompletedStorageTaskCount = 0;
    uint64_t totalScheduledUpdateCount = 0;
    uint64_t totalCompletedUpdateCount = 0;
    uint64_t totalCompletedUploadCount = 0;
    uint64_t totalStorageTaskFailureCount = 0;
    uint64_t totalUpdateTaskFailureCount = 0;
    uint64_t totalScheduledUnloadCount = 0;
    uint64_t totalCompletedUnloadCount = 0;
    uint64_t totalUnloadTaskFailureCount = 0;
    uint64_t totalDelayedFreeCount = 0;
    uint64_t totalEvictionAgeRejectedCount = 0;
    uint64_t totalResidentBudgetFailureCount = 0;
    uint64_t totalTransferBudgetFailureCount = 0;
    uint64_t totalEvictedPageCount = 0;
    uint64_t totalAllocationFailureCount = 0;
    uint64_t totalScheduledPageLoadCount = 0;
    uint64_t totalCompletedPageLoadCount = 0;
    uint64_t totalPageLoadFailureCount = 0;
    uint64_t oldestActiveAge = 0;
    uint64_t oldestResidentAge = 0;
    uint64_t oldestPendingAge = 0;
    uint64_t totalPrefetchAdmitted = 0;
    uint64_t totalPrefetchUsed = 0;
    uint64_t totalPrefetchDeferred = 0;
    uint64_t totalCancelledUploads = 0;
    uint64_t totalCompletionDrivenUploads = 0;
};

struct MeshletStreamColdPageReclaimDesc {
    uint64_t clasUsedBytes = 0;
    uint64_t clasCapacityBytes = 0;
    uint64_t clasRetiringBytes = 0;
    uint32_t retentionFrames = 120;
    uint32_t pressureAgeFrames = 16;
    uint32_t maxPages = 256;
    std::function<uint64_t(uint32_t)> clasPageBytes;
};

class MeshletStreamResidencyManager {
public:
    bool initialize(const MeshletStreamResidencyDesc& desc, std::string& reason);
    void reset();

    void beginFrame(CpuProfileRecorder* profiler = nullptr);
    bool lockFallbackPages(std::span<const uint32_t> pageIndices, std::string& reason);
    bool requestPage(uint32_t pageIndex);
    bool unloadPage(uint32_t pageIndex);
    uint32_t consumeGpuRequests(std::span<const uint32_t> pageIds);
    uint32_t consumeGpuRequests(const StreamGpuRequestBatch& requests, CpuProfileRecorder* profiler = nullptr);
    // Called once after an upload is admitted, before the decoded payload is released.
    using UploadObserver = std::function<void(uint32_t, std::span<const uint8_t>)>;
    uint32_t processUploads(Streamer& streamer, Buffer& destination, uint32_t maxUploads,
        const UploadObserver& observer = {}, CpuProfileRecorder* profiler = nullptr,
        uint64_t maxUploadBytesPerFrame = 0);

    // Uses confirmed unused feedback; geometry and its CLAS share one victim list.
    uint32_t reclaimColdPages(const MeshletStreamColdPageReclaimDesc& desc, CpuProfileRecorder* profiler = nullptr);

    void buildInitialPageTable(std::span<StreamPageTableEntry> outEntries) const;
    std::span<const StreamPageTablePatch> pendingPatches() const { return patches_; }
    void clearPendingPatches() { patches_.clear(); }
    void rebuildPendingPatches();
    // GPU visibility only: CPU residency and allocation retirement still wait
    // for completion. Never use these patches in a different recording.
    uint32_t buildOrderedUploadPatches(const CommandBuffer& commands,
        std::vector<StreamPageTablePatch>& outPatches) const;

    MeshletStreamPageResidencyState pageState(uint32_t pageIndex) const;
    uint64_t deviceOffsetForPage(uint32_t pageIndex) const;
    uint32_t deviceSizeForPage(uint32_t pageIndex) const;
    bool pageAllocated(uint32_t pageIndex) const;
    bool pageResident(uint32_t pageIndex) const;
    uint64_t pageAge(uint32_t pageIndex) const;

    uint32_t maxResidentPages() const { return maxResidentPages_; }
    uint64_t maxResidentBytes() const { return storage_.capacityBytes(); }
    uint32_t residentPageCount() const;
    uint32_t pendingPageCount() const;
    uint32_t queuedUploadCount() const;
    uint32_t trackedPageCount() const { return static_cast<uint32_t>(pages_.size()); }
    uint64_t pageBufferSize() const { return storage_.capacityBytes(); }
    std::span<const uint32_t> requestedPages() const { return requestedPages_; }
    std::span<const uint32_t> unloadRequestedPages() const { return unloadRequestedPages_; }
    std::span<const uint32_t> activePages() const { return activePages_; }
    std::span<const uint32_t> residentPages() const { return residentPages_; }
    std::span<const uint32_t> pendingPages() const { return pendingPages_; }
    std::span<const uint32_t> newlyResidentPages() const { return newlyResidentPages_; }
    std::span<const uint32_t> newlyUnloadedPages() const { return newlyUnloadedPages_; }
    const MeshletStreamStorage& storage() const { return storage_; }
    // Detailed mode scans page ages and allocator free blocks.
    MeshletStreamResidencyStats stats(bool detailed = true) const;
    MeshletStreamLatencySnapshot latencySnapshot() const { return latency_ ? latency_->snapshot() : MeshletStreamLatencySnapshot{}; }
    uint32_t availablePrefetchRequests() const
    {
        const uint32_t limit = pageLoader_.ready() ? std::max(maxPageLoadsInFlight_ / 4u, 1u) : 32u;
        const uint32_t queued = queuedUploadCount();
        return queued < limit ? limit - queued : 0u;
    }

private:
    struct PageRequest {
        uint32_t pageIndex;
        float screenBenefit = -1.0f;
        double schedulingPriority = 0.0;
        bool prefetch = false;
    };

    struct PageEntry {
        uint64_t lastUsedFrame = 0;
        uint64_t firstRequestFrame = 0;
        uint64_t residentSinceFrame = 0;
        uint32_t deviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
        uint32_t allocationBytes = 0;
        uint32_t deviceSizeBytes = 0;
        uint32_t taskIndex = kInvalidStreamingTaskIndex;
        uint32_t activeTablePosition = UINT32_MAX;
        uint32_t stateTablePosition = UINT32_MAX;
        bool lockedFallback = false;
        bool queued = false;
        MeshletStreamPageResidencyState state = MeshletStreamPageResidencyState::Unloaded;
        bool gpuUnused = false;
        float screenBenefit = -1.0f;
        bool prefetch = false;
    };
    static_assert(sizeof(PageEntry) == 64);

    using PagePositionMember = uint32_t PageEntry::*;

    size_t prepareEvictionCandidates(CpuProfileRecorder* profiler = nullptr, uint32_t minimumAge = 0);
    bool allocatePageStorage(uint32_t pageIndex);
    bool scheduleUnload(uint32_t pageIndex, bool eviction);
    void completeUnloadTask(uint32_t taskIndex);
    void completeUploadPages(std::span<const uint32_t> pageIndices, uint32_t taskIndex);
    void releasePageStorage(uint32_t pageIndex);
    void setPageState(uint32_t pageIndex, MeshletStreamPageResidencyState state);
    void queueUpload(uint32_t pageIndex);
    void recordPatch(uint32_t pageIndex);
    void addToTable(std::vector<uint32_t>& table, PagePositionMember positionMember, uint32_t pageIndex);
    void removeFromTable(std::vector<uint32_t>& table, PagePositionMember positionMember, uint32_t pageIndex);
    void updateStateTables(
        uint32_t pageIndex,
        MeshletStreamPageResidencyState oldState,
        MeshletStreamPageResidencyState newState);
    uint64_t oldestAge(std::span<const uint32_t> pageIndices) const;
    void resetFrameStats();
    void consumeReadyRequestTasks(CpuProfileRecorder* profiler = nullptr);

    const scene::MeshletStreamAsset* asset_ = nullptr;
    MeshletStreamStorage storage_;
    std::unordered_map<uint32_t, PageEntry> pages_;
    std::deque<uint32_t> uploadQueue_;
    MeshletStreamPageLoader pageLoader_;
    std::unique_ptr<MeshletStreamLatencyTracker> latency_;
    std::deque<MeshletStreamPageLoadResult> preparedPageLoads_;
    StreamingTaskQueue requestTaskQueue_;
    std::array<std::vector<PageRequest>, kStreamingMaxActiveTasks> requestTaskPages_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> requestTaskUnloadPages_;
    StreamingTaskQueue storageTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> storageTaskPages_;
    std::array<std::shared_ptr<StreamUploadCompletion>, kStreamingMaxActiveTasks> storageCompletions_;
    StreamingTaskQueue unloadTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> unloadTaskPages_;
    StreamingTaskQueue updateTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> updateTaskPages_;
    std::vector<uint32_t> requestedPages_;
    std::vector<uint32_t> unloadRequestedPages_;
    std::vector<uint32_t> activePages_;
    std::vector<uint32_t> residentPages_;
    // unordered_map nodes survive rehash; remove these pointers before erasing a page.
    std::vector<PageEntry*> residentPageEntries_;
    std::vector<uint32_t> pendingPages_;
    std::vector<uint32_t> newlyResidentPages_;
    std::vector<uint32_t> newlyUnloadedPages_;
    struct EvictionCandidate {
        uint64_t lastUsedFrame;
        uint32_t pageIndex;
    };
    std::vector<EvictionCandidate> evictionCandidates_;
    size_t evictionCandidateCursor_ = 0;
    size_t evictionSortedCount_ = 0;
    uint64_t evictionSortedMinimumAge_ = UINT64_MAX;
    bool evictionCandidatesBuilt_ = false;
    bool evictionAgeRejected_ = false;
    bool residentDemandFeedback_ = false;
    bool geometryReclaimPressure_ = false;
    bool clasReclaimPressure_ = false;
    uint32_t frameUnloadTaskIndex_ = kInvalidStreamingTaskIndex;
    std::unordered_map<uint32_t, size_t> requestMarks_;
    // One bit per logical page; clear only words touched by the previous batch.
    std::vector<uint64_t> unloadRequestBits_;
    std::vector<uint32_t> unloadRequestTouchedWords_;
    std::vector<StreamPageTablePatch> patches_;
    MeshletStreamResidencyStats stats_;
    uint64_t frameIndex_ = 0;
    uint32_t pageCount_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t queuedFrameCount_ = 3;
    uint32_t unloadDelayFrames_ = 1;
    uint32_t evictionAgeThresholdFrames_ = 1;
    uint32_t maxPageLoadsInFlight_ = 0;
    bool immediateGpuRequests_ = false;
    bool completionDrivenUploads_ = true;
};

} // namespace metallic::render
