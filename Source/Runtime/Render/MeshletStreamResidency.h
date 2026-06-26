#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/StreamingTaskQueue.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

enum class MeshletStreamPageResidencyState : uint8_t {
    Unloaded,
    PendingUpload,
    Resident,
    LockedFallback,
};

struct StreamPageTableEntry {
    uint32_t slot = UINT32_MAX;
    uint32_t state = static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded);
    uint32_t lastRequestFrame = 0;
    uint32_t lodLevel = 0;
    uint32_t payloadBytes = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct StreamPageTablePatch {
    uint32_t pageId = 0;
    uint32_t slot = UINT32_MAX;
    uint32_t state = static_cast<uint32_t>(MeshletStreamPageResidencyState::Unloaded);
    uint32_t padding0 = 0;
};

struct StreamRequestBufferHeader {
    uint32_t loadCounter = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

struct StreamUpdateBufferHeader {
    uint32_t patchCounter = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(StreamPageTableEntry) == 32);
static_assert(sizeof(StreamPageTablePatch) == 16);
static_assert(sizeof(StreamRequestBufferHeader) == 16);
static_assert(sizeof(StreamUpdateBufferHeader) == 16);

struct MeshletStreamResidencyDesc {
    const scene::MeshletStreamAsset* asset = nullptr;
    uint32_t maxResidentPages = 0;
    uint32_t queuedFrameCount = 3;
    uint64_t pageStride = 0;
};

struct MeshletStreamResidencyStats {
    uint64_t frameIndex = 0;
    uint32_t pageCount = 0;
    uint32_t maxResidentPages = 0;
    uint32_t usedSlotCount = 0;
    uint32_t freeSlotCount = 0;
    uint32_t activePageCount = 0;
    uint32_t residentPageCount = 0;
    uint32_t pendingPageCount = 0;
    uint32_t queuedUploadCount = 0;
    uint32_t queuedRequestTaskCount = 0;
    uint32_t availableRequestTaskCount = 0;
    uint32_t queuedStorageTaskCount = 0;
    uint32_t availableStorageTaskCount = 0;
    uint32_t queuedUpdateTaskCount = 0;
    uint32_t availableUpdateTaskCount = 0;
    uint32_t pendingPatchCount = 0;
    uint32_t frameGpuRequestCount = 0;
    uint32_t frameUniqueGpuRequestCount = 0;
    uint32_t frameScheduledRequestTaskCount = 0;
    uint32_t frameCompletedRequestTaskCount = 0;
    uint32_t frameDroppedRequestTaskCount = 0;
    uint32_t frameRequestTaskFailureCount = 0;
    uint32_t frameConsumedGpuRequestCount = 0;
    uint32_t frameQueuedUploadCount = 0;
    uint32_t frameScheduledUploadCount = 0;
    uint32_t frameCompletedStorageTaskCount = 0;
    uint32_t frameScheduledUpdateCount = 0;
    uint32_t frameCompletedUpdateCount = 0;
    uint32_t frameCompletedUploadCount = 0;
    uint32_t frameStorageTaskFailureCount = 0;
    uint32_t frameUpdateTaskFailureCount = 0;
    uint32_t frameEvictedPageCount = 0;
    uint32_t frameAllocationFailureCount = 0;
    uint64_t totalGpuRequestCount = 0;
    uint64_t totalUniqueGpuRequestCount = 0;
    uint64_t totalScheduledRequestTaskCount = 0;
    uint64_t totalCompletedRequestTaskCount = 0;
    uint64_t totalDroppedRequestTaskCount = 0;
    uint64_t totalRequestTaskFailureCount = 0;
    uint64_t totalConsumedGpuRequestCount = 0;
    uint64_t totalQueuedUploadCount = 0;
    uint64_t totalScheduledUploadCount = 0;
    uint64_t totalCompletedStorageTaskCount = 0;
    uint64_t totalScheduledUpdateCount = 0;
    uint64_t totalCompletedUpdateCount = 0;
    uint64_t totalCompletedUploadCount = 0;
    uint64_t totalStorageTaskFailureCount = 0;
    uint64_t totalUpdateTaskFailureCount = 0;
    uint64_t totalEvictedPageCount = 0;
    uint64_t totalAllocationFailureCount = 0;
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
    uint32_t consumeGpuRequests(std::span<const uint32_t> pageIds);
    uint32_t processUploads(Streamer& streamer, Buffer& destination, uint32_t maxUploads);

    void buildInitialPageTable(std::span<StreamPageTableEntry> outEntries) const;
    std::span<const StreamPageTablePatch> pendingPatches() const { return patches_; }
    void clearPendingPatches() { patches_.clear(); }

    MeshletStreamPageResidencyState pageState(uint32_t pageIndex) const;
    uint32_t slotForPage(uint32_t pageIndex) const;
    bool pageResident(uint32_t pageIndex) const;
    uint64_t pageAge(uint32_t pageIndex) const;

    uint32_t maxResidentPages() const { return maxResidentPages_; }
    uint32_t residentPageCount() const;
    uint32_t pendingPageCount() const;
    uint32_t queuedUploadCount() const { return static_cast<uint32_t>(uploadQueue_.size()); }
    uint64_t pageStride() const { return pageStride_; }
    uint64_t pageBufferSize() const { return pageStride_ * maxResidentPages_; }
    std::span<const uint32_t> requestedPages() const { return requestedPages_; }
    std::span<const uint32_t> activePages() const { return activePages_; }
    std::span<const uint32_t> residentPages() const { return residentPages_; }
    std::span<const uint32_t> pendingPages() const { return pendingPages_; }
    std::span<const uint32_t> slotToPageTable() const { return slotToPage_; }
    MeshletStreamResidencyStats stats() const;

private:
    struct PageEntry {
        uint32_t slot = UINT32_MAX;
        uint32_t storageTaskIndex = kInvalidStreamingTaskIndex;
        uint32_t updateTaskIndex = kInvalidStreamingTaskIndex;
        uint64_t lastUsedFrame = 0;
        bool lockedFallback = false;
        bool queued = false;
        MeshletStreamPageResidencyState state = MeshletStreamPageResidencyState::Unloaded;
    };

    uint32_t allocateSlot(uint32_t pageIndex);
    void releaseSlot(uint32_t pageIndex);
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
    std::vector<PageEntry> pages_;
    std::vector<uint32_t> slotToPage_;
    std::vector<uint32_t> freeSlots_;
    std::vector<uint32_t> uploadQueue_;
    StreamingTaskQueue requestTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> requestTaskPages_;
    StreamingTaskQueue storageTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> storageTaskPages_;
    StreamingTaskQueue updateTaskQueue_;
    std::array<std::vector<uint32_t>, kStreamingMaxActiveTasks> updateTaskPages_;
    std::vector<uint32_t> requestedPages_;
    std::vector<uint32_t> activePages_;
    std::vector<uint32_t> residentPages_;
    std::vector<uint32_t> pendingPages_;
    std::vector<uint32_t> activePagePositions_;
    std::vector<uint32_t> residentPagePositions_;
    std::vector<uint32_t> pendingPagePositions_;
    std::vector<uint8_t> requestMarks_;
    std::vector<StreamPageTablePatch> patches_;
    MeshletStreamResidencyStats stats_;
    uint64_t frameIndex_ = 0;
    uint64_t pageStride_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t queuedFrameCount_ = 3;
};

} // namespace metallic::render
