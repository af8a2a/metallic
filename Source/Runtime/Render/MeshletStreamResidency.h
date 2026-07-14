#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/MeshletStreamPageLoader.h"
#include "Runtime/Render/StreamingTaskQueue.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kInvalidStreamDeviceOffsetBytes = UINT32_MAX;
inline constexpr uint64_t kMeshletStreamStorageAlignment = 256;

enum class MeshletStreamPageResidencyState : uint8_t {
    Unloaded,
    PendingUpload,
    Resident,
    LockedFallback,
    PendingUnload,
};

struct StreamPageTableEntry {
    uint32_t deviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
    uint32_t deviceSizeBytes = 0;
    uint32_t state = static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded);
    uint32_t lastRequestFrame = 0;
    uint32_t lodLevel = 0;
    uint32_t payloadBytes = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct StreamPageTablePatch {
    uint32_t pageId = 0;
    uint32_t deviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
    uint32_t deviceSizeBytes = 0;
    uint32_t state = static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded);
};

inline constexpr uint32_t kStreamRequestHeaderWordCount = 16;
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
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
    uint32_t padding3 = 0;
    uint32_t padding4 = 0;
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

static_assert(sizeof(StreamPageTableEntry) == 32);
static_assert(sizeof(StreamPageTablePatch) == 16);
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
};

struct MeshletStreamStorageAllocation {
    uint64_t offset = UINT64_MAX;
    uint64_t requestedSize = 0;
    uint64_t allocatedSize = 0;

    bool valid() const { return offset != UINT64_MAX && allocatedSize != 0; }
};

class MeshletStreamStorage {
public:
    bool initialize(uint64_t capacityBytes, uint64_t alignmentBytes, std::string& reason);
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

    uint64_t capacityBytes_ = 0;
    uint64_t alignmentBytes_ = kMeshletStreamStorageAlignment;
    uint64_t usedBytes_ = 0;
    uint32_t allocationCount_ = 0;
    std::vector<FreeBlock> freeBlocks_;
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
    uint32_t pageLoadWorkerCount = 0;
    uint32_t maxPageLoadsInFlight = 0;
};

struct MeshletStreamResidencyStats {
    uint64_t frameIndex = 0;
    uint32_t pageCount = 0;
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
    uint32_t pageLoadWorkerCount = 0;
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
};

class MeshletStreamResidencyManager {
public:
    bool initialize(const MeshletStreamResidencyDesc& desc, std::string& reason);
    void reset();

    void beginFrame();
    bool lockFallbackPages(std::span<const uint32_t> pageIndices, std::string& reason);
    bool requestPage(uint32_t pageIndex);
    bool unloadPage(uint32_t pageIndex);
    uint32_t consumeGpuRequests(std::span<const uint32_t> pageIds);
    uint32_t consumeGpuRequests(const StreamGpuRequestBatch& requests);
    uint32_t processUploads(Streamer& streamer, Buffer& destination, uint32_t maxUploads);

    void buildInitialPageTable(std::span<StreamPageTableEntry> outEntries) const;
    std::span<const StreamPageTablePatch> pendingPatches() const { return patches_; }
    void clearPendingPatches() { patches_.clear(); }

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
    uint64_t pageBufferSize() const { return storage_.capacityBytes(); }
    std::span<const uint32_t> requestedPages() const { return requestedPages_; }
    std::span<const uint32_t> unloadRequestedPages() const { return unloadRequestedPages_; }
    std::span<const uint32_t> activePages() const { return activePages_; }
    std::span<const uint32_t> residentPages() const { return residentPages_; }
    std::span<const uint32_t> pendingPages() const { return pendingPages_; }
    std::span<const uint32_t> newlyResidentPages() const { return newlyResidentPages_; }
    std::span<const uint32_t> newlyUnloadedPages() const { return newlyUnloadedPages_; }
    const MeshletStreamStorage& storage() const { return storage_; }
    MeshletStreamResidencyStats stats() const;

private:
    struct PageEntry {
        uint64_t deviceOffsetBytes = UINT64_MAX;
        uint64_t allocationBytes = 0;
        uint32_t deviceSizeBytes = 0;
        uint32_t storageTaskIndex = kInvalidStreamingTaskIndex;
        uint32_t updateTaskIndex = kInvalidStreamingTaskIndex;
        uint32_t unloadTaskIndex = kInvalidStreamingTaskIndex;
        uint64_t lastUsedFrame = 0;
        bool lockedFallback = false;
        bool queued = false;
        MeshletStreamPageResidencyState state = MeshletStreamPageResidencyState::Unloaded;
    };

    bool allocatePageStorage(uint32_t pageIndex);
    bool scheduleUnload(uint32_t pageIndex, bool eviction);
    void completeUnloadTask(uint32_t taskIndex);
    void releasePageStorage(uint32_t pageIndex);
    void setPageState(uint32_t pageIndex, MeshletStreamPageResidencyState state);
    void queueUpload(uint32_t pageIndex);
    void recordPatch(uint32_t pageIndex);
    void addToTable(std::vector<uint32_t>& table, std::vector<uint32_t>& positions, uint32_t pageIndex);
    void removeFromTable(std::vector<uint32_t>& table, std::vector<uint32_t>& positions, uint32_t pageIndex);
    void updateStateTables(
        uint32_t pageIndex,
        MeshletStreamPageResidencyState oldState,
        MeshletStreamPageResidencyState newState);
    uint64_t oldestAge(std::span<const uint32_t> pageIndices) const;
    void resetFrameStats();

    const scene::MeshletStreamAsset* asset_ = nullptr;
    MeshletStreamStorage storage_;
    std::vector<PageEntry> pages_;
    std::deque<uint32_t> uploadQueue_;
    MeshletStreamPageLoader pageLoader_;
    std::deque<MeshletStreamPageLoadResult> preparedPageLoads_;
    StreamingTaskQueue requestTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> requestTaskPages_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> requestTaskUnloadPages_;
    StreamingTaskQueue storageTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> storageTaskPages_;
    StreamingTaskQueue unloadTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> unloadTaskPages_;
    StreamingTaskQueue updateTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> updateTaskPages_;
    std::vector<uint32_t> requestedPages_;
    std::vector<uint32_t> unloadRequestedPages_;
    std::vector<uint32_t> activePages_;
    std::vector<uint32_t> residentPages_;
    std::vector<uint32_t> pendingPages_;
    std::vector<uint32_t> newlyResidentPages_;
    std::vector<uint32_t> newlyUnloadedPages_;
    std::vector<uint32_t> activePagePositions_;
    std::vector<uint32_t> residentPagePositions_;
    std::vector<uint32_t> pendingPagePositions_;
    std::vector<uint8_t> requestMarks_;
    std::vector<uint8_t> unloadRequestMarks_;
    std::vector<StreamPageTablePatch> patches_;
    MeshletStreamResidencyStats stats_;
    uint64_t frameIndex_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t queuedFrameCount_ = 3;
    uint32_t unloadDelayFrames_ = 1;
    uint32_t evictionAgeThresholdFrames_ = 1;
    uint32_t maxPageLoadsInFlight_ = 0;
};

} // namespace metallic::render
