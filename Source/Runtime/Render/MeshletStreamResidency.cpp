#include "Runtime/Render/MeshletStreamResidency.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace metallic::render {
namespace {

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

bool MeshletStreamResidencyManager::initialize(
    const MeshletStreamResidencyDesc& desc,
    std::string& reason)
{
    reset();
    reason.clear();
    if (desc.asset == nullptr || !desc.asset->valid()) {
        reason = "MeshletStreamResidencyManager requires a valid asset";
        return false;
    }
    if (desc.maxResidentPages == 0) {
        reason = "MeshletStreamResidencyManager requires at least one resident page slot";
        return false;
    }
    if (desc.asset->pageCount() == 0 || desc.asset->maxPagePayloadBytes() == 0) {
        reason = "MeshletStreamResidencyManager asset has no streamable pages";
        return false;
    }

    const uint64_t defaultStride = alignUp(desc.asset->maxPagePayloadBytes(), 256);
    pageStride_ = desc.pageStride != 0 ? desc.pageStride : defaultStride;
    if (pageStride_ < desc.asset->maxPagePayloadBytes()) {
        reason = "MeshletStreamResidencyManager page stride is smaller than the largest payload";
        return false;
    }

    asset_ = desc.asset;
    maxResidentPages_ = desc.maxResidentPages;
    queuedFrameCount_ = std::max(desc.queuedFrameCount, 1u);
    pages_.resize(asset_->pageCount());
    slotToPage_.assign(maxResidentPages_, UINT32_MAX);
    requestMarks_.assign(asset_->pageCount(), 0);
    freeSlots_.reserve(maxResidentPages_);
    for (uint32_t slot = 0; slot < maxResidentPages_; ++slot) {
        freeSlots_.push_back(maxResidentPages_ - slot - 1u);
    }
    return true;
}

void MeshletStreamResidencyManager::reset()
{
    asset_ = nullptr;
    pages_.clear();
    slotToPage_.clear();
    freeSlots_.clear();
    uploadQueue_.clear();
    requestMarks_.clear();
    patches_.clear();
    frameIndex_ = 0;
    pageStride_ = 0;
    maxResidentPages_ = 0;
    queuedFrameCount_ = 3;
}

void MeshletStreamResidencyManager::beginFrame()
{
    ++frameIndex_;
    for (PageEntry& page : pages_) {
        if (page.state != MeshletStreamPageResidencyState::PendingUpload) {
            continue;
        }
        if (page.pendingFrames > 0) {
            --page.pendingFrames;
        }
        if (page.pendingFrames == 0) {
            const uint32_t pageIndex = static_cast<uint32_t>(&page - pages_.data());
            setPageState(
                pageIndex,
                page.lockedFallback
                    ? MeshletStreamPageResidencyState::LockedFallback
                    : MeshletStreamPageResidencyState::Resident);
        }
    }
}

bool MeshletStreamResidencyManager::lockFallbackPages(
    std::span<const uint32_t> pageIndices,
    std::string& reason)
{
    reason.clear();
    if (asset_ == nullptr) {
        reason = "MeshletStreamResidencyManager is not initialized";
        return false;
    }
    if (pageIndices.size() > freeSlots_.size()) {
        reason = "not enough resident page slots for locked fallback pages";
        return false;
    }

    for (uint32_t pageIndex : pageIndices) {
        if (pageIndex >= pages_.size()) {
            reason = "fallback page index is out of range";
            return false;
        }
        PageEntry& page = pages_[pageIndex];
        page.lockedFallback = true;
        if (page.slot == UINT32_MAX && allocateSlot(pageIndex) == UINT32_MAX) {
            reason = "failed to allocate a fallback page slot";
            return false;
        }
        if (page.state == MeshletStreamPageResidencyState::Unloaded && !page.queued) {
            uploadQueue_.push_back(pageIndex);
            page.queued = true;
        }
    }
    return true;
}

bool MeshletStreamResidencyManager::requestPage(uint32_t pageIndex)
{
    if (asset_ == nullptr || pageIndex >= pages_.size()) {
        return false;
    }

    PageEntry& page = pages_[pageIndex];
    page.lastUsedFrame = frameIndex_;
    if (page.state == MeshletStreamPageResidencyState::Resident ||
        page.state == MeshletStreamPageResidencyState::LockedFallback ||
        page.state == MeshletStreamPageResidencyState::PendingUpload) {
        return true;
    }

    if (page.slot == UINT32_MAX && allocateSlot(pageIndex) == UINT32_MAX) {
        return false;
    }
    if (!page.queued) {
        uploadQueue_.push_back(pageIndex);
        page.queued = true;
    }
    return false;
}

uint32_t MeshletStreamResidencyManager::consumeGpuRequests(std::span<const uint32_t> pageIds)
{
    if (asset_ == nullptr || requestMarks_.size() != pages_.size()) {
        return 0;
    }

    std::vector<uint32_t> uniqueRequests;
    uniqueRequests.reserve(pageIds.size());
    for (uint32_t pageIndex : pageIds) {
        if (pageIndex >= pages_.size() || requestMarks_[pageIndex] != 0) {
            continue;
        }
        requestMarks_[pageIndex] = 1;
        uniqueRequests.push_back(pageIndex);
    }

    uint32_t consumed = 0;
    for (uint32_t pageIndex : uniqueRequests) {
        (void)requestPage(pageIndex);
        ++consumed;
        requestMarks_[pageIndex] = 0;
    }
    return consumed;
}

uint32_t MeshletStreamResidencyManager::processUploads(
    Streamer& streamer,
    Buffer& destination,
    uint32_t maxUploads)
{
    if (asset_ == nullptr || maxUploads == 0) {
        return 0;
    }

    uint32_t uploadCount = 0;
    for (size_t queueIndex = 0; queueIndex < uploadQueue_.size() && uploadCount < maxUploads;) {
        const uint32_t pageIndex = uploadQueue_[queueIndex];
        PageEntry& page = pages_[pageIndex];
        page.queued = false;

        if (page.state != MeshletStreamPageResidencyState::Unloaded || page.slot == UINT32_MAX) {
            uploadQueue_.erase(uploadQueue_.begin() + static_cast<std::ptrdiff_t>(queueIndex));
            continue;
        }

        const std::span<const uint8_t> payload = asset_->pagePayload(pageIndex);
        if (payload.empty() || payload.size() > pageStride_) {
            uploadQueue_.erase(uploadQueue_.begin() + static_cast<std::ptrdiff_t>(queueIndex));
            continue;
        }

        const StreamDataChunk chunk{
            .data = payload.data(),
            .size = static_cast<uint64_t>(payload.size()),
        };
        const BufferOffset streamed = streamer.streamBufferData(StreamBufferDataDesc{
            .dataChunks = &chunk,
            .dataChunkCount = 1,
            .placementAlignment = 16,
            .dstBuffer = &destination,
            .dstOffset = static_cast<uint64_t>(page.slot) * pageStride_,
        });
        if (!streamed.valid()) {
            page.queued = true;
            break;
        }

        setPageState(pageIndex, MeshletStreamPageResidencyState::PendingUpload);
        page.pendingFrames = std::max(streamer.desc().queuedFrameCount, queuedFrameCount_);
        ++uploadCount;
        uploadQueue_.erase(uploadQueue_.begin() + static_cast<std::ptrdiff_t>(queueIndex));
    }
    return uploadCount;
}

void MeshletStreamResidencyManager::buildInitialPageTable(std::span<StreamPageTableEntry> outEntries) const
{
    if (asset_ == nullptr || outEntries.size() < pages_.size()) {
        return;
    }

    const std::span<const scene::MeshletStreamPageInfo> assetPages = asset_->pages();
    for (uint32_t pageIndex = 0; pageIndex < pages_.size(); ++pageIndex) {
        const PageEntry& page = pages_[pageIndex];
        const scene::MeshletStreamPageInfo& assetPage = assetPages[pageIndex];
        outEntries[pageIndex] = StreamPageTableEntry{
            .slot = page.state == MeshletStreamPageResidencyState::Unloaded ? UINT32_MAX : page.slot,
            .state = static_cast<uint32_t>(page.state),
            .lastRequestFrame = 0,
            .lodLevel = assetPage.lodLevel,
            .payloadBytes = static_cast<uint32_t>(assetPage.payloadSize),
        };
    }
}

MeshletStreamPageResidencyState MeshletStreamResidencyManager::pageState(uint32_t pageIndex) const
{
    if (pageIndex >= pages_.size()) {
        return MeshletStreamPageResidencyState::Unloaded;
    }
    return pages_[pageIndex].state;
}

uint32_t MeshletStreamResidencyManager::slotForPage(uint32_t pageIndex) const
{
    if (pageIndex >= pages_.size()) {
        return UINT32_MAX;
    }
    return pages_[pageIndex].slot;
}

bool MeshletStreamResidencyManager::pageResident(uint32_t pageIndex) const
{
    const MeshletStreamPageResidencyState state = pageState(pageIndex);
    return state == MeshletStreamPageResidencyState::Resident ||
        state == MeshletStreamPageResidencyState::LockedFallback;
}

uint32_t MeshletStreamResidencyManager::residentPageCount() const
{
    uint32_t count = 0;
    for (const PageEntry& page : pages_) {
        if (page.state == MeshletStreamPageResidencyState::Resident ||
            page.state == MeshletStreamPageResidencyState::LockedFallback) {
            ++count;
        }
    }
    return count;
}

uint32_t MeshletStreamResidencyManager::pendingPageCount() const
{
    uint32_t count = 0;
    for (const PageEntry& page : pages_) {
        if (page.state == MeshletStreamPageResidencyState::PendingUpload) {
            ++count;
        }
    }
    return count;
}

uint32_t MeshletStreamResidencyManager::allocateSlot(uint32_t pageIndex)
{
    if (pageIndex >= pages_.size()) {
        return UINT32_MAX;
    }
    PageEntry& page = pages_[pageIndex];
    if (page.slot != UINT32_MAX) {
        return page.slot;
    }

    uint32_t slot = UINT32_MAX;
    if (!freeSlots_.empty()) {
        slot = freeSlots_.back();
        freeSlots_.pop_back();
    } else {
        uint64_t oldestFrame = std::numeric_limits<uint64_t>::max();
        uint32_t evictPage = UINT32_MAX;
        for (uint32_t candidate = 0; candidate < pages_.size(); ++candidate) {
            const PageEntry& entry = pages_[candidate];
            if (entry.lockedFallback ||
                entry.slot == UINT32_MAX ||
                entry.state == MeshletStreamPageResidencyState::PendingUpload) {
                continue;
            }
            if (entry.lastUsedFrame < oldestFrame) {
                oldestFrame = entry.lastUsedFrame;
                evictPage = candidate;
            }
        }
        if (evictPage == UINT32_MAX) {
            return UINT32_MAX;
        }
        slot = pages_[evictPage].slot;
        releaseSlot(evictPage);
    }

    page.slot = slot;
    slotToPage_[slot] = pageIndex;
    return slot;
}

void MeshletStreamResidencyManager::releaseSlot(uint32_t pageIndex)
{
    if (pageIndex >= pages_.size()) {
        return;
    }
    PageEntry& page = pages_[pageIndex];
    if (page.slot == UINT32_MAX) {
        return;
    }
    if (page.slot < slotToPage_.size()) {
        slotToPage_[page.slot] = UINT32_MAX;
    }
    const MeshletStreamPageResidencyState oldState = page.state;
    page.slot = UINT32_MAX;
    page.pendingFrames = 0;
    page.queued = false;
    setPageState(pageIndex, MeshletStreamPageResidencyState::Unloaded);
    if (oldState == MeshletStreamPageResidencyState::Unloaded) {
        recordPatch(pageIndex);
    }
}

void MeshletStreamResidencyManager::setPageState(uint32_t pageIndex, MeshletStreamPageResidencyState state)
{
    if (pageIndex >= pages_.size()) {
        return;
    }
    PageEntry& page = pages_[pageIndex];
    if (page.state == state) {
        return;
    }
    page.state = state;
    recordPatch(pageIndex);
}

void MeshletStreamResidencyManager::recordPatch(uint32_t pageIndex)
{
    if (pageIndex >= pages_.size()) {
        return;
    }
    const PageEntry& page = pages_[pageIndex];
    patches_.push_back(StreamPageTablePatch{
        .pageId = pageIndex,
        .slot = page.state == MeshletStreamPageResidencyState::Unloaded ? UINT32_MAX : page.slot,
        .state = static_cast<uint32_t>(page.state),
    });
}

} // namespace metallic::render
