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

constexpr uint32_t kInvalidTablePosition = UINT32_MAX;

bool residentState(MeshletStreamPageResidencyState state)
{
    return state == MeshletStreamPageResidencyState::Resident ||
        state == MeshletStreamPageResidencyState::LockedFallback;
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
    unloadRequestMarks_.assign(asset_->pageCount(), 0);
    activePagePositions_.assign(asset_->pageCount(), kInvalidTablePosition);
    residentPagePositions_.assign(asset_->pageCount(), kInvalidTablePosition);
    pendingPagePositions_.assign(asset_->pageCount(), kInvalidTablePosition);
    requestedPages_.reserve(asset_->pageCount());
    unloadRequestedPages_.reserve(asset_->pageCount());
    activePages_.reserve(maxResidentPages_);
    residentPages_.reserve(maxResidentPages_);
    pendingPages_.reserve(maxResidentPages_);
    freeSlots_.reserve(maxResidentPages_);
    for (uint32_t slot = 0; slot < maxResidentPages_; ++slot) {
        freeSlots_.push_back(maxResidentPages_ - slot - 1u);
    }
    stats_.pageCount = asset_->pageCount();
    stats_.maxResidentPages = maxResidentPages_;
    return true;
}

void MeshletStreamResidencyManager::reset()
{
    asset_ = nullptr;
    pages_.clear();
    slotToPage_.clear();
    freeSlots_.clear();
    uploadQueue_.clear();
    requestTaskQueue_.reset();
    for (std::vector<uint32_t>& taskPages : requestTaskPages_) {
        taskPages.clear();
    }
    for (std::vector<uint32_t>& taskPages : requestTaskUnloadPages_) {
        taskPages.clear();
    }
    storageTaskQueue_.reset();
    for (std::vector<uint32_t>& taskPages : storageTaskPages_) {
        taskPages.clear();
    }
    updateTaskQueue_.reset();
    for (std::vector<uint32_t>& taskPages : updateTaskPages_) {
        taskPages.clear();
    }
    requestedPages_.clear();
    unloadRequestedPages_.clear();
    activePages_.clear();
    residentPages_.clear();
    pendingPages_.clear();
    activePagePositions_.clear();
    residentPagePositions_.clear();
    pendingPagePositions_.clear();
    requestMarks_.clear();
    unloadRequestMarks_.clear();
    patches_.clear();
    stats_ = {};
    frameIndex_ = 0;
    pageStride_ = 0;
    maxResidentPages_ = 0;
    queuedFrameCount_ = 3;
}

void MeshletStreamResidencyManager::beginFrame()
{
    ++frameIndex_;
    requestedPages_.clear();
    unloadRequestedPages_.clear();
    resetFrameStats();

    while (updateTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = updateTaskQueue_.pop();
        if (taskIndex < updateTaskPages_.size()) {
            for (uint32_t pageIndex : updateTaskPages_[taskIndex]) {
                if (pageIndex >= pages_.size()) {
                    continue;
                }
                PageEntry& page = pages_[pageIndex];
                if (page.state != MeshletStreamPageResidencyState::PendingUpload ||
                    page.updateTaskIndex != taskIndex) {
                    continue;
                }
                page.updateTaskIndex = kInvalidStreamingTaskIndex;
                setPageState(
                    pageIndex,
                    page.lockedFallback
                        ? MeshletStreamPageResidencyState::LockedFallback
                        : MeshletStreamPageResidencyState::Resident);
                ++stats_.frameCompletedUpdateCount;
                ++stats_.frameCompletedUploadCount;
                ++stats_.totalCompletedUpdateCount;
                ++stats_.totalCompletedUploadCount;
            }
            updateTaskPages_[taskIndex].clear();
        }
        updateTaskQueue_.releaseTaskIndex(taskIndex);
    }

    if (storageTaskQueue_.canPop(frameIndex_, true)) {
        uint32_t dependentIndex = kInvalidStreamingTaskIndex;
        const uint32_t taskIndex = storageTaskQueue_.popWithDependent(dependentIndex);
        uint32_t updateTaskIndex = dependentIndex;
        if (updateTaskIndex == kInvalidStreamingTaskIndex) {
            updateTaskIndex = updateTaskQueue_.acquireTaskIndex();
            if (updateTaskIndex == kInvalidStreamingTaskIndex) {
                ++stats_.frameUpdateTaskFailureCount;
                ++stats_.totalUpdateTaskFailureCount;
            }
        }

        if (updateTaskIndex != kInvalidStreamingTaskIndex) {
            uint32_t updateCount = 0;
            if (taskIndex < storageTaskPages_.size()) {
                std::vector<uint32_t>& updatePages = updateTaskPages_[updateTaskIndex];
                updatePages.clear();
                for (uint32_t pageIndex : storageTaskPages_[taskIndex]) {
                    if (pageIndex >= pages_.size()) {
                        continue;
                    }
                    PageEntry& page = pages_[pageIndex];
                    if (page.state != MeshletStreamPageResidencyState::PendingUpload ||
                        page.storageTaskIndex != taskIndex) {
                        continue;
                    }
                    page.storageTaskIndex = kInvalidStreamingTaskIndex;
                    page.updateTaskIndex = updateTaskIndex;
                    updatePages.push_back(pageIndex);
                    ++updateCount;
                }
                storageTaskPages_[taskIndex].clear();
            }

            ++stats_.frameCompletedStorageTaskCount;
            ++stats_.totalCompletedStorageTaskCount;
            if (updateCount > 0) {
                ++stats_.frameScheduledUpdateCount;
                ++stats_.totalScheduledUpdateCount;
                updateTaskQueue_.push(updateTaskIndex, frameIndex_ + 1u);
            } else {
                updateTaskPages_[updateTaskIndex].clear();
                updateTaskQueue_.releaseTaskIndex(updateTaskIndex);
            }
        }
        storageTaskQueue_.releaseTaskIndex(taskIndex);
    }

    if (requestTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = requestTaskQueue_.pop();
        uint32_t latestTaskIndex = taskIndex;
        while (requestTaskQueue_.canPop(frameIndex_, false)) {
            if (latestTaskIndex < requestTaskPages_.size()) {
                requestTaskPages_[latestTaskIndex].clear();
                requestTaskUnloadPages_[latestTaskIndex].clear();
            }
            requestTaskQueue_.releaseTaskIndex(latestTaskIndex);
            ++stats_.frameDroppedRequestTaskCount;
            ++stats_.totalDroppedRequestTaskCount;
            latestTaskIndex = requestTaskQueue_.pop();
        }

        uint32_t consumed = 0;
        uint32_t consumedUnloads = 0;
        if (latestTaskIndex < requestTaskPages_.size()) {
            for (uint32_t pageIndex : requestTaskUnloadPages_[latestTaskIndex]) {
                if (pageIndex >= pages_.size()) {
                    continue;
                }
                unloadRequestedPages_.push_back(pageIndex);
                if (unloadPage(pageIndex)) {
                    ++consumedUnloads;
                }
            }
            for (uint32_t pageIndex : requestTaskPages_[latestTaskIndex]) {
                if (pageIndex >= pages_.size()) {
                    continue;
                }
                requestedPages_.push_back(pageIndex);
                (void)requestPage(pageIndex);
                ++consumed;
            }
            requestTaskPages_[latestTaskIndex].clear();
            requestTaskUnloadPages_[latestTaskIndex].clear();
        }
        requestTaskQueue_.releaseTaskIndex(latestTaskIndex);

        ++stats_.frameCompletedRequestTaskCount;
        ++stats_.totalCompletedRequestTaskCount;
        stats_.frameConsumedGpuRequestCount += consumed;
        stats_.totalConsumedGpuRequestCount += consumed;
        stats_.frameConsumedGpuUnloadRequestCount += consumedUnloads;
        stats_.totalConsumedGpuUnloadRequestCount += consumedUnloads;
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
            queueUpload(pageIndex);
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
        queueUpload(pageIndex);
    }
    return false;
}

bool MeshletStreamResidencyManager::unloadPage(uint32_t pageIndex)
{
    if (asset_ == nullptr || pageIndex >= pages_.size()) {
        return false;
    }

    PageEntry& page = pages_[pageIndex];
    if (page.lockedFallback) {
        return false;
    }
    if (page.slot == UINT32_MAX && page.state == MeshletStreamPageResidencyState::Unloaded) {
        page.lastUsedFrame = frameIndex_;
        return true;
    }

    page.lastUsedFrame = frameIndex_;
    releaseSlot(pageIndex);
    return true;
}

uint32_t MeshletStreamResidencyManager::consumeGpuRequests(std::span<const uint32_t> pageIds)
{
    return consumeGpuRequests(StreamGpuRequestBatch{
        .loadPageIds = pageIds,
        .loadRequestCounter = static_cast<uint32_t>(std::min<uint64_t>(
            pageIds.size(),
            std::numeric_limits<uint32_t>::max())),
    });
}

uint32_t MeshletStreamResidencyManager::consumeGpuRequests(const StreamGpuRequestBatch& requests)
{
    if (asset_ == nullptr ||
        requestMarks_.size() != pages_.size() ||
        unloadRequestMarks_.size() != pages_.size()) {
        return 0;
    }

    stats_.frameGpuRequestCount += static_cast<uint32_t>(std::min<uint64_t>(
        requests.loadRequestCounter != 0 ? requests.loadRequestCounter : requests.loadPageIds.size(),
        std::numeric_limits<uint32_t>::max()));
    stats_.totalGpuRequestCount += requests.loadRequestCounter != 0
        ? requests.loadRequestCounter
        : requests.loadPageIds.size();
    stats_.frameGpuUnloadRequestCount += static_cast<uint32_t>(std::min<uint64_t>(
        requests.unloadRequestCounter != 0 ? requests.unloadRequestCounter : requests.unloadPageIds.size(),
        std::numeric_limits<uint32_t>::max()));
    stats_.totalGpuUnloadRequestCount += requests.unloadRequestCounter != 0
        ? requests.unloadRequestCounter
        : requests.unloadPageIds.size();
    stats_.frameGpuRequestOverflowCount += requests.loadOverflowCounter;
    stats_.totalGpuRequestOverflowCount += requests.loadOverflowCounter;
    stats_.frameGpuUnloadRequestOverflowCount += requests.unloadOverflowCounter;
    stats_.totalGpuUnloadRequestOverflowCount += requests.unloadOverflowCounter;
    stats_.frameGpuInvalidRequestCount += requests.invalidPageCounter;
    stats_.totalGpuInvalidRequestCount += requests.invalidPageCounter;

    std::vector<uint32_t> uniqueRequests;
    uniqueRequests.reserve(requests.loadPageIds.size());
    for (uint32_t pageIndex : requests.loadPageIds) {
        if (pageIndex >= pages_.size() || requestMarks_[pageIndex] != 0) {
            if (pageIndex >= pages_.size()) {
                ++stats_.frameGpuInvalidRequestCount;
                ++stats_.totalGpuInvalidRequestCount;
            }
            continue;
        }
        requestMarks_[pageIndex] = 1;
        uniqueRequests.push_back(pageIndex);
        requestedPages_.push_back(pageIndex);
    }
    stats_.frameUniqueGpuRequestCount += static_cast<uint32_t>(uniqueRequests.size());
    stats_.totalUniqueGpuRequestCount += uniqueRequests.size();

    for (uint32_t pageIndex : uniqueRequests) {
        requestMarks_[pageIndex] = 0;
    }

    std::vector<uint32_t> uniqueUnloadRequests;
    uniqueUnloadRequests.reserve(requests.unloadPageIds.size());
    for (uint32_t pageIndex : requests.unloadPageIds) {
        if (pageIndex >= pages_.size() || unloadRequestMarks_[pageIndex] != 0) {
            if (pageIndex >= pages_.size()) {
                ++stats_.frameGpuInvalidRequestCount;
                ++stats_.totalGpuInvalidRequestCount;
            }
            continue;
        }
        unloadRequestMarks_[pageIndex] = 1;
        uniqueUnloadRequests.push_back(pageIndex);
        unloadRequestedPages_.push_back(pageIndex);
    }
    stats_.frameUniqueGpuUnloadRequestCount += static_cast<uint32_t>(uniqueUnloadRequests.size());
    stats_.totalUniqueGpuUnloadRequestCount += uniqueUnloadRequests.size();

    for (uint32_t pageIndex : uniqueUnloadRequests) {
        unloadRequestMarks_[pageIndex] = 0;
    }
    if (uniqueRequests.empty() && uniqueUnloadRequests.empty()) {
        return 0;
    }

    const uint32_t taskIndex = requestTaskQueue_.acquireTaskIndex();
    if (taskIndex == kInvalidStreamingTaskIndex) {
        ++stats_.frameRequestTaskFailureCount;
        ++stats_.totalRequestTaskFailureCount;
        return 0;
    }

    std::vector<uint32_t>& taskPages = requestTaskPages_[taskIndex];
    taskPages = std::move(uniqueRequests);
    std::vector<uint32_t>& taskUnloadPages = requestTaskUnloadPages_[taskIndex];
    taskUnloadPages = std::move(uniqueUnloadRequests);
    requestTaskQueue_.push(taskIndex, frameIndex_ + 1u);
    ++stats_.frameScheduledRequestTaskCount;
    ++stats_.totalScheduledRequestTaskCount;
    return static_cast<uint32_t>(taskPages.size() + taskUnloadPages.size());
}

uint32_t MeshletStreamResidencyManager::processUploads(
    Streamer& streamer,
    Buffer& destination,
    uint32_t maxUploads)
{
    if (asset_ == nullptr || maxUploads == 0) {
        return 0;
    }

    const uint32_t taskIndex = storageTaskQueue_.acquireTaskIndex();
    if (taskIndex == kInvalidStreamingTaskIndex) {
        ++stats_.frameStorageTaskFailureCount;
        ++stats_.totalStorageTaskFailureCount;
        return 0;
    }

    const uint32_t updateTaskIndex = updateTaskQueue_.acquireTaskIndex();
    if (updateTaskIndex == kInvalidStreamingTaskIndex) {
        storageTaskQueue_.releaseTaskIndex(taskIndex);
        ++stats_.frameUpdateTaskFailureCount;
        ++stats_.totalUpdateTaskFailureCount;
        return 0;
    }

    std::vector<uint32_t>& taskPages = storageTaskPages_[taskIndex];
    taskPages.clear();
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
        page.storageTaskIndex = taskIndex;
        taskPages.push_back(pageIndex);
        ++uploadCount;
        ++stats_.frameScheduledUploadCount;
        ++stats_.totalScheduledUploadCount;
        uploadQueue_.erase(uploadQueue_.begin() + static_cast<std::ptrdiff_t>(queueIndex));
    }

    if (uploadCount == 0) {
        taskPages.clear();
        storageTaskQueue_.releaseTaskIndex(taskIndex);
        updateTaskQueue_.releaseTaskIndex(updateTaskIndex);
        return 0;
    }

    const uint32_t frameDelay = std::max(streamer.desc().queuedFrameCount, queuedFrameCount_);
    storageTaskQueue_.push(taskIndex, frameIndex_ + frameDelay, updateTaskIndex);
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
    return residentState(pageState(pageIndex));
}

uint64_t MeshletStreamResidencyManager::pageAge(uint32_t pageIndex) const
{
    if (pageIndex >= pages_.size()) {
        return 0;
    }
    const uint64_t lastUsedFrame = pages_[pageIndex].lastUsedFrame;
    return frameIndex_ >= lastUsedFrame ? frameIndex_ - lastUsedFrame : 0;
}

uint32_t MeshletStreamResidencyManager::residentPageCount() const
{
    return static_cast<uint32_t>(residentPages_.size());
}

uint32_t MeshletStreamResidencyManager::pendingPageCount() const
{
    return static_cast<uint32_t>(pendingPages_.size());
}

MeshletStreamResidencyStats MeshletStreamResidencyManager::stats() const
{
    MeshletStreamResidencyStats result = stats_;
    result.frameIndex = frameIndex_;
    result.pageCount = static_cast<uint32_t>(pages_.size());
    result.maxResidentPages = maxResidentPages_;
    result.usedSlotCount = static_cast<uint32_t>(activePages_.size());
    result.freeSlotCount = static_cast<uint32_t>(freeSlots_.size());
    result.activePageCount = static_cast<uint32_t>(activePages_.size());
    result.residentPageCount = static_cast<uint32_t>(residentPages_.size());
    result.pendingPageCount = static_cast<uint32_t>(pendingPages_.size());
    result.queuedUploadCount = static_cast<uint32_t>(uploadQueue_.size());
    result.queuedRequestTaskCount = requestTaskQueue_.queuedTaskCount();
    result.availableRequestTaskCount = requestTaskQueue_.availableTaskCount();
    result.queuedStorageTaskCount = storageTaskQueue_.queuedTaskCount();
    result.availableStorageTaskCount = storageTaskQueue_.availableTaskCount();
    result.queuedUpdateTaskCount = updateTaskQueue_.queuedTaskCount();
    result.availableUpdateTaskCount = updateTaskQueue_.availableTaskCount();
    result.pendingPatchCount = static_cast<uint32_t>(patches_.size());
    result.oldestActiveAge = oldestAge(activePages_);
    result.oldestResidentAge = oldestAge(residentPages_);
    result.oldestPendingAge = oldestAge(pendingPages_);
    return result;
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
            ++stats_.frameAllocationFailureCount;
            ++stats_.totalAllocationFailureCount;
            return UINT32_MAX;
        }
        slot = pages_[evictPage].slot;
        ++stats_.frameEvictedPageCount;
        ++stats_.totalEvictedPageCount;
        releaseSlot(evictPage, false);
    }

    page.slot = slot;
    slotToPage_[slot] = pageIndex;
    addToTable(activePages_, activePagePositions_, pageIndex);
    return slot;
}

void MeshletStreamResidencyManager::releaseSlot(uint32_t pageIndex, bool returnToFreeList)
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
    const uint32_t releasedSlot = page.slot;
    const MeshletStreamPageResidencyState oldState = page.state;
    page.slot = UINT32_MAX;
    page.storageTaskIndex = kInvalidStreamingTaskIndex;
    page.updateTaskIndex = kInvalidStreamingTaskIndex;
    page.queued = false;
    removeFromTable(activePages_, activePagePositions_, pageIndex);
    setPageState(pageIndex, MeshletStreamPageResidencyState::Unloaded);
    if (returnToFreeList && releasedSlot < maxResidentPages_) {
        freeSlots_.push_back(releasedSlot);
    }
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
    const MeshletStreamPageResidencyState oldState = page.state;
    page.state = state;
    updateStateTables(pageIndex, oldState, state);
    recordPatch(pageIndex);
}

void MeshletStreamResidencyManager::queueUpload(uint32_t pageIndex)
{
    if (pageIndex >= pages_.size()) {
        return;
    }
    PageEntry& page = pages_[pageIndex];
    if (page.queued) {
        return;
    }
    uploadQueue_.push_back(pageIndex);
    page.queued = true;
    ++stats_.frameQueuedUploadCount;
    ++stats_.totalQueuedUploadCount;
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

void MeshletStreamResidencyManager::addToTable(
    std::vector<uint32_t>& table,
    std::vector<uint32_t>& positions,
    uint32_t pageIndex)
{
    if (pageIndex >= positions.size() || positions[pageIndex] != kInvalidTablePosition) {
        return;
    }
    positions[pageIndex] = static_cast<uint32_t>(table.size());
    table.push_back(pageIndex);
}

void MeshletStreamResidencyManager::removeFromTable(
    std::vector<uint32_t>& table,
    std::vector<uint32_t>& positions,
    uint32_t pageIndex)
{
    if (pageIndex >= positions.size() || positions[pageIndex] == kInvalidTablePosition) {
        return;
    }

    const uint32_t position = positions[pageIndex];
    const uint32_t movedPage = table.back();
    table[position] = movedPage;
    positions[movedPage] = position;
    table.pop_back();
    positions[pageIndex] = kInvalidTablePosition;
}

void MeshletStreamResidencyManager::updateStateTables(
    uint32_t pageIndex,
    MeshletStreamPageResidencyState oldState,
    MeshletStreamPageResidencyState newState)
{
    if (residentState(oldState) && !residentState(newState)) {
        removeFromTable(residentPages_, residentPagePositions_, pageIndex);
    }
    if (!residentState(oldState) && residentState(newState)) {
        addToTable(residentPages_, residentPagePositions_, pageIndex);
    }
    if (oldState == MeshletStreamPageResidencyState::PendingUpload &&
        newState != MeshletStreamPageResidencyState::PendingUpload) {
        removeFromTable(pendingPages_, pendingPagePositions_, pageIndex);
    }
    if (oldState != MeshletStreamPageResidencyState::PendingUpload &&
        newState == MeshletStreamPageResidencyState::PendingUpload) {
        addToTable(pendingPages_, pendingPagePositions_, pageIndex);
    }
}

uint64_t MeshletStreamResidencyManager::oldestAge(std::span<const uint32_t> pageIndices) const
{
    uint64_t oldest = 0;
    for (uint32_t pageIndex : pageIndices) {
        oldest = std::max(oldest, pageAge(pageIndex));
    }
    return oldest;
}

void MeshletStreamResidencyManager::resetFrameStats()
{
    stats_.frameGpuRequestCount = 0;
    stats_.frameUniqueGpuRequestCount = 0;
    stats_.frameGpuUnloadRequestCount = 0;
    stats_.frameUniqueGpuUnloadRequestCount = 0;
    stats_.frameScheduledRequestTaskCount = 0;
    stats_.frameCompletedRequestTaskCount = 0;
    stats_.frameDroppedRequestTaskCount = 0;
    stats_.frameRequestTaskFailureCount = 0;
    stats_.frameConsumedGpuRequestCount = 0;
    stats_.frameConsumedGpuUnloadRequestCount = 0;
    stats_.frameGpuRequestOverflowCount = 0;
    stats_.frameGpuUnloadRequestOverflowCount = 0;
    stats_.frameGpuInvalidRequestCount = 0;
    stats_.frameQueuedUploadCount = 0;
    stats_.frameScheduledUploadCount = 0;
    stats_.frameCompletedStorageTaskCount = 0;
    stats_.frameScheduledUpdateCount = 0;
    stats_.frameCompletedUpdateCount = 0;
    stats_.frameCompletedUploadCount = 0;
    stats_.frameStorageTaskFailureCount = 0;
    stats_.frameUpdateTaskFailureCount = 0;
    stats_.frameEvictedPageCount = 0;
    stats_.frameAllocationFailureCount = 0;
}

} // namespace metallic::render
