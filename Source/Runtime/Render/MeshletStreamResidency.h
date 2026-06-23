#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

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

    uint32_t maxResidentPages() const { return maxResidentPages_; }
    uint32_t residentPageCount() const;
    uint32_t pendingPageCount() const;
    uint32_t queuedUploadCount() const { return static_cast<uint32_t>(uploadQueue_.size()); }
    uint64_t pageStride() const { return pageStride_; }
    uint64_t pageBufferSize() const { return pageStride_ * maxResidentPages_; }

private:
    struct PageEntry {
        uint32_t slot = UINT32_MAX;
        uint32_t pendingFrames = 0;
        uint64_t lastUsedFrame = 0;
        bool lockedFallback = false;
        bool queued = false;
        MeshletStreamPageResidencyState state = MeshletStreamPageResidencyState::Unloaded;
    };

    uint32_t allocateSlot(uint32_t pageIndex);
    void releaseSlot(uint32_t pageIndex);
    void setPageState(uint32_t pageIndex, MeshletStreamPageResidencyState state);
    void recordPatch(uint32_t pageIndex);

    const scene::MeshletStreamAsset* asset_ = nullptr;
    std::vector<PageEntry> pages_;
    std::vector<uint32_t> slotToPage_;
    std::vector<uint32_t> freeSlots_;
    std::vector<uint32_t> uploadQueue_;
    std::vector<uint8_t> requestMarks_;
    std::vector<StreamPageTablePatch> patches_;
    uint64_t frameIndex_ = 0;
    uint64_t pageStride_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t queuedFrameCount_ = 3;
};

} // namespace metallic::render
