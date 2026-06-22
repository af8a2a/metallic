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
    uint32_t processUploads(Streamer& streamer, Buffer& destination, uint32_t maxUploads);

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

    const scene::MeshletStreamAsset* asset_ = nullptr;
    std::vector<PageEntry> pages_;
    std::vector<uint32_t> slotToPage_;
    std::vector<uint32_t> freeSlots_;
    std::vector<uint32_t> uploadQueue_;
    uint64_t frameIndex_ = 0;
    uint64_t pageStride_ = 0;
    uint32_t maxResidentPages_ = 0;
    uint32_t queuedFrameCount_ = 3;
};

} // namespace metallic::render
