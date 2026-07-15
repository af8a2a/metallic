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

bool streamableEvictionState(MeshletStreamPageResidencyState state)
{
    return state == MeshletStreamPageResidencyState::Resident;
}

} // namespace

bool MeshletStreamStorage::initialize(
    uint64_t capacityBytes,
    uint64_t alignmentBytes,
    std::string& reason,
    uint64_t maxCapacityBytes)
{
    reset();
    reason.clear();
    if (capacityBytes == 0) {
        reason = "MeshletStreamStorage requires a non-zero byte budget";
        return false;
    }
    if (maxCapacityBytes == 0 || capacityBytes > maxCapacityBytes) {
        reason = "MeshletStreamStorage byte budget exceeds the configured address limit";
        return false;
    }
    alignmentBytes_ = std::max<uint64_t>(alignmentBytes, 1u);
    if (capacityBytes > std::numeric_limits<uint64_t>::max() - (alignmentBytes_ - 1u)) {
        reason = "MeshletStreamStorage aligned byte budget overflowed";
        reset();
        return false;
    }
    capacityBytes_ = alignUp(capacityBytes, alignmentBytes_);
    if (capacityBytes_ > maxCapacityBytes) {
        reason = "MeshletStreamStorage aligned byte budget exceeds the configured address limit";
        reset();
        return false;
    }
    freeBlocks_.push_back(FreeBlock{
        .offset = 0,
        .size = capacityBytes_,
    });
    return true;
}

void MeshletStreamStorage::reset()
{
    capacityBytes_ = 0;
    alignmentBytes_ = kMeshletStreamStorageAlignment;
    usedBytes_ = 0;
    allocationCount_ = 0;
    freeBlocks_.clear();
}

MeshletStreamStorageAllocation MeshletStreamStorage::allocate(uint64_t byteSize)
{
    const uint64_t alignedSize = allocationSize(byteSize);
    if (alignedSize == 0) {
        return {};
    }

    for (size_t index = 0; index < freeBlocks_.size(); ++index) {
        FreeBlock& block = freeBlocks_[index];
        const uint64_t alignedOffset = alignUp(block.offset, alignmentBytes_);
        if (alignedOffset < block.offset) {
            continue;
        }
        const uint64_t prefixBytes = alignedOffset - block.offset;
        if (prefixBytes > block.size || alignedSize > block.size - prefixBytes) {
            continue;
        }

        const uint64_t suffixOffset = alignedOffset + alignedSize;
        const uint64_t suffixBytes = block.offset + block.size - suffixOffset;
        if (prefixBytes != 0 && suffixBytes != 0) {
            block.size = prefixBytes;
            freeBlocks_.insert(
                freeBlocks_.begin() + static_cast<std::ptrdiff_t>(index + 1u),
                FreeBlock{
                    .offset = suffixOffset,
                    .size = suffixBytes,
                });
        } else if (prefixBytes != 0) {
            block.size = prefixBytes;
        } else if (suffixBytes != 0) {
            block.offset = suffixOffset;
            block.size = suffixBytes;
        } else {
            freeBlocks_.erase(freeBlocks_.begin() + static_cast<std::ptrdiff_t>(index));
        }

        usedBytes_ += alignedSize;
        ++allocationCount_;
        return MeshletStreamStorageAllocation{
            .offset = alignedOffset,
            .requestedSize = byteSize,
            .allocatedSize = alignedSize,
        };
    }

    return {};
}

void MeshletStreamStorage::release(const MeshletStreamStorageAllocation& allocation)
{
    if (!allocation.valid() ||
        allocation.offset >= capacityBytes_ ||
        allocation.allocatedSize > capacityBytes_ - allocation.offset) {
        return;
    }

    FreeBlock block{
        .offset = allocation.offset,
        .size = allocation.allocatedSize,
    };
    auto iter = std::lower_bound(
        freeBlocks_.begin(),
        freeBlocks_.end(),
        block.offset,
        [](const FreeBlock& lhs, uint64_t offset) {
            return lhs.offset < offset;
        });
    iter = freeBlocks_.insert(iter, block);

    if (iter != freeBlocks_.begin()) {
        auto previous = iter - 1;
        if (previous->offset + previous->size == iter->offset) {
            previous->size += iter->size;
            iter = freeBlocks_.erase(iter);
            iter = previous;
        }
    }
    auto next = iter + 1;
    if (next != freeBlocks_.end() && iter->offset + iter->size == next->offset) {
        iter->size += next->size;
        freeBlocks_.erase(next);
    }

    usedBytes_ = allocation.allocatedSize <= usedBytes_ ? usedBytes_ - allocation.allocatedSize : 0;
    if (allocationCount_ != 0) {
        --allocationCount_;
    }
}

uint64_t MeshletStreamStorage::allocationSize(uint64_t byteSize) const
{
    if (byteSize == 0) {
        return 0;
    }
    return alignUp(byteSize, alignmentBytes_);
}

bool MeshletStreamStorage::canAllocate(uint64_t byteSize) const
{
    const uint64_t alignedSize = allocationSize(byteSize);
    if (alignedSize == 0) {
        return false;
    }
    for (const FreeBlock& block : freeBlocks_) {
        const uint64_t alignedOffset = alignUp(block.offset, alignmentBytes_);
        if (alignedOffset < block.offset) {
            continue;
        }
        const uint64_t prefixBytes = alignedOffset - block.offset;
        if (prefixBytes <= block.size && alignedSize <= block.size - prefixBytes) {
            return true;
        }
    }
    return false;
}

uint64_t MeshletStreamStorage::largestFreeBlockBytes() const
{
    uint64_t largest = 0;
    for (const FreeBlock& block : freeBlocks_) {
        largest = std::max(largest, block.size);
    }
    return largest;
}

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
    if (desc.asset->pageCount() == 0 || desc.asset->maxPagePayloadBytes() == 0) {
        reason = "MeshletStreamResidencyManager asset has no streamable pages";
        return false;
    }
    if (desc.pageLoadWorkerCount > kMeshletStreamMaxPageLoadWorkers) {
        reason = "MeshletStreamResidencyManager page load worker count exceeds the supported limit";
        return false;
    }

    const uint64_t defaultStride = alignUp(desc.asset->maxPagePayloadBytes(), 256);
    const uint64_t legacyStride = desc.pageStride != 0 ? desc.pageStride : defaultStride;
    uint64_t maxResidentBytes = desc.maxResidentBytes;
    if (maxResidentBytes == 0) {
        if (desc.maxResidentPages == 0) {
            reason = "MeshletStreamResidencyManager requires maxResidentBytes or at least one legacy resident page";
            return false;
        }
        if (legacyStride < desc.asset->maxPagePayloadBytes()) {
            reason = "MeshletStreamResidencyManager legacy page stride is smaller than the largest payload";
            return false;
        }
        if (legacyStride > std::numeric_limits<uint64_t>::max() / desc.maxResidentPages) {
            reason = "MeshletStreamResidencyManager resident byte budget overflowed";
            return false;
        }
        maxResidentBytes = legacyStride * desc.maxResidentPages;
    }
    if (!storage_.initialize(maxResidentBytes, desc.storageAlignment, reason)) {
        return false;
    }

    asset_ = desc.asset;
    maxResidentPages_ = desc.maxResidentPages;
    queuedFrameCount_ = std::max(desc.queuedFrameCount, 1u);
    unloadDelayFrames_ = std::max(desc.unloadDelayFrames, 1u);
    evictionAgeThresholdFrames_ = desc.evictionAgeThresholdFrames;
    pageCount_ = asset_->pageCount();
    if (maxResidentPages_ != 0) {
        const uint32_t residentReserve = std::min(maxResidentPages_, pageCount_);
        pages_.reserve(residentReserve);
        activePages_.reserve(residentReserve);
        residentPages_.reserve(residentReserve);
        pendingPages_.reserve(residentReserve);
    }
    if (desc.pageLoadWorkerCount != 0) {
        const uint32_t pageLoadCapacity = maxResidentPages_ != 0
            ? std::min(maxResidentPages_, pageCount_)
            : pageCount_;
        maxPageLoadsInFlight_ = std::min(
            std::max(desc.maxPageLoadsInFlight, desc.pageLoadWorkerCount),
            pageLoadCapacity);
        if (!pageLoader_.initialize(*asset_, desc.pageLoadWorkerCount, reason)) {
            const std::string loaderReason = reason;
            reset();
            reason = loaderReason;
            return false;
        }
    }
    stats_.pageCount = pageCount_;
    stats_.maxResidentPages = maxResidentPages_;
    stats_.maxResidentBytes = storage_.capacityBytes();
    return true;
}

void MeshletStreamResidencyManager::reset()
{
    pageLoader_.reset();
    asset_ = nullptr;
    storage_.reset();
    pages_.clear();
    uploadQueue_.clear();
    preparedPageLoads_.clear();
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
    unloadTaskQueue_.reset();
    for (std::vector<uint32_t>& taskPages : unloadTaskPages_) {
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
    newlyResidentPages_.clear();
    newlyUnloadedPages_.clear();
    requestMarks_.clear();
    unloadRequestMarks_.clear();
    patches_.clear();
    stats_ = {};
    frameIndex_ = 0;
    pageCount_ = 0;
    maxResidentPages_ = 0;
    queuedFrameCount_ = 3;
    unloadDelayFrames_ = 1;
    evictionAgeThresholdFrames_ = 1;
    maxPageLoadsInFlight_ = 0;
}

void MeshletStreamResidencyManager::beginFrame()
{
    ++frameIndex_;
    requestedPages_.clear();
    unloadRequestedPages_.clear();
    newlyResidentPages_.clear();
    newlyUnloadedPages_.clear();
    resetFrameStats();

    while (updateTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = updateTaskQueue_.pop();
        if (taskIndex < updateTaskPages_.size()) {
            for (uint32_t pageIndex : updateTaskPages_[taskIndex]) {
                auto pageIter = pages_.find(pageIndex);
                if (pageIter == pages_.end()) {
                    continue;
                }
                PageEntry& page = pageIter->second;
                if (page.state != MeshletStreamPageResidencyState::PendingUpload ||
                    page.taskIndex != taskIndex) {
                    continue;
                }
                page.taskIndex = kInvalidStreamingTaskIndex;
                setPageState(
                    pageIndex,
                    page.lockedFallback
                        ? MeshletStreamPageResidencyState::LockedFallback
                        : MeshletStreamPageResidencyState::Resident);
                newlyResidentPages_.push_back(pageIndex);
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
                    auto pageIter = pages_.find(pageIndex);
                    if (pageIter == pages_.end()) {
                        continue;
                    }
                    PageEntry& page = pageIter->second;
                    if (page.state != MeshletStreamPageResidencyState::PendingUpload ||
                        page.taskIndex != taskIndex) {
                        continue;
                    }
                    page.taskIndex = updateTaskIndex;
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

    while (unloadTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = unloadTaskQueue_.pop();
        completeUnloadTask(taskIndex);
        unloadTaskQueue_.releaseTaskIndex(taskIndex);
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
                if (pageIndex >= pageCount_) {
                    continue;
                }
                unloadRequestedPages_.push_back(pageIndex);
                if (unloadPage(pageIndex)) {
                    ++consumedUnloads;
                }
            }
            std::vector<uint32_t>& taskPages = requestTaskPages_[latestTaskIndex];
            for (auto iter = taskPages.rbegin(); iter != taskPages.rend(); ++iter) {
                const uint32_t pageIndex = *iter;
                if (pageIndex >= pageCount_) {
                    continue;
                }
                requestedPages_.push_back(pageIndex);
                (void)requestPage(pageIndex);
                ++consumed;
            }
            taskPages.clear();
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
    uint64_t requiredBytes = 0;
    uint32_t requiredPages = 0;
    std::vector<uint32_t> uniqueNewPages;
    uniqueNewPages.reserve(pageIndices.size());
    for (uint32_t pageIndex : pageIndices) {
        if (pageIndex >= pageCount_) {
            reason = "fallback page index is out of range";
            return false;
        }
        if (!pageAllocated(pageIndex)) {
            if (std::find(uniqueNewPages.begin(), uniqueNewPages.end(), pageIndex) != uniqueNewPages.end()) {
                continue;
            }
            uniqueNewPages.push_back(pageIndex);
            ++requiredPages;
            const uint64_t pageBytes = asset_->pages()[pageIndex].uncompressedSize;
            const uint64_t allocationBytes = storage_.allocationSize(pageBytes);
            if (allocationBytes == 0 ||
                allocationBytes > std::numeric_limits<uint64_t>::max() - requiredBytes) {
                reason = "locked fallback page byte budget overflowed";
                return false;
            }
            requiredBytes += allocationBytes;
        }
    }
    if (requiredBytes > storage_.freeBytes()) {
        reason = "not enough resident byte budget for locked fallback pages";
        return false;
    }
    if (maxResidentPages_ != 0 &&
        (activePages_.size() >= maxResidentPages_ ||
        requiredPages > maxResidentPages_ - static_cast<uint32_t>(activePages_.size()))) {
        reason = "not enough resident page budget for locked fallback pages";
        return false;
    }

    for (uint32_t pageIndex : pageIndices) {
        PageEntry& page = pages_.try_emplace(pageIndex).first->second;
        page.lockedFallback = true;
        if (!pageAllocated(pageIndex) && !allocatePageStorage(pageIndex)) {
            reason = "failed to allocate fallback page storage";
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
    if (asset_ == nullptr || pageIndex >= pageCount_) {
        return false;
    }

    auto [pageIter, inserted] = pages_.try_emplace(pageIndex);
    PageEntry& page = pageIter->second;
    page.lastUsedFrame = frameIndex_;
    if (page.state == MeshletStreamPageResidencyState::Resident ||
        page.state == MeshletStreamPageResidencyState::LockedFallback ||
        page.state == MeshletStreamPageResidencyState::PendingUpload) {
        return true;
    }
    if (page.state == MeshletStreamPageResidencyState::PendingUnload) {
        ++stats_.frameResidentBudgetFailureCount;
        ++stats_.totalResidentBudgetFailureCount;
        return false;
    }

    if (!pageAllocated(pageIndex) && !allocatePageStorage(pageIndex)) {
        if (inserted) {
            pages_.erase(pageIter);
        }
        return false;
    }
    if (!page.queued) {
        queueUpload(pageIndex);
    }
    return false;
}

bool MeshletStreamResidencyManager::unloadPage(uint32_t pageIndex)
{
    if (asset_ == nullptr || pageIndex >= pageCount_) {
        return false;
    }

    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return true;
    }
    PageEntry& page = pageIter->second;
    if (page.lockedFallback) {
        return false;
    }
    if (!pageAllocated(pageIndex) && page.state == MeshletStreamPageResidencyState::Unloaded) {
        page.lastUsedFrame = frameIndex_;
        return true;
    }

    page.lastUsedFrame = frameIndex_;
    return scheduleUnload(pageIndex, false);
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
    if (asset_ == nullptr) {
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
    requestMarks_.clear();
    requestMarks_.reserve(requests.loadPageIds.size());
    for (uint32_t pageIndex : requests.loadPageIds) {
        if (pageIndex >= pageCount_ || !requestMarks_.insert(pageIndex).second) {
            if (pageIndex >= pageCount_) {
                ++stats_.frameGpuInvalidRequestCount;
                ++stats_.totalGpuInvalidRequestCount;
            }
            continue;
        }
        uniqueRequests.push_back(pageIndex);
        requestedPages_.push_back(pageIndex);
    }
    stats_.frameUniqueGpuRequestCount += static_cast<uint32_t>(uniqueRequests.size());
    stats_.totalUniqueGpuRequestCount += uniqueRequests.size();

    std::vector<uint32_t> uniqueUnloadRequests;
    uniqueUnloadRequests.reserve(requests.unloadPageIds.size());
    unloadRequestMarks_.clear();
    unloadRequestMarks_.reserve(requests.unloadPageIds.size());
    for (uint32_t pageIndex : requests.unloadPageIds) {
        if (pageIndex >= pageCount_ || !unloadRequestMarks_.insert(pageIndex).second) {
            if (pageIndex >= pageCount_) {
                ++stats_.frameGpuInvalidRequestCount;
                ++stats_.totalGpuInvalidRequestCount;
            }
            continue;
        }
        uniqueUnloadRequests.push_back(pageIndex);
        unloadRequestedPages_.push_back(pageIndex);
    }
    stats_.frameUniqueGpuUnloadRequestCount += static_cast<uint32_t>(uniqueUnloadRequests.size());
    stats_.totalUniqueGpuUnloadRequestCount += uniqueUnloadRequests.size();

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
    if (asset_ == nullptr) {
        return 0;
    }

    const bool asynchronousLoads = pageLoader_.ready();
    auto schedulePageLoads = [this]() {
        while (!uploadQueue_.empty() &&
            static_cast<uint64_t>(pageLoader_.outstandingCount()) + preparedPageLoads_.size() <
                maxPageLoadsInFlight_) {
            const uint32_t pageIndex = uploadQueue_.front();
            uploadQueue_.pop_front();
            auto pageIter = pages_.find(pageIndex);
            if (pageIter == pages_.end()) {
                continue;
            }
            PageEntry& page = pageIter->second;
            if (page.state != MeshletStreamPageResidencyState::Unloaded ||
                !pageAllocated(pageIndex) ||
                !page.queued) {
                page.queued = false;
                continue;
            }
            if (!pageLoader_.enqueue(pageIndex)) {
                uploadQueue_.push_front(pageIndex);
                break;
            }
            ++stats_.frameScheduledPageLoadCount;
            ++stats_.totalScheduledPageLoadCount;
        }
    };

    if (asynchronousLoads) {
        schedulePageLoads();
        MeshletStreamPageLoadResult loadedPage;
        while (pageLoader_.tryPop(loadedPage)) {
            ++stats_.frameCompletedPageLoadCount;
            ++stats_.totalCompletedPageLoadCount;
            auto pageIter = pages_.find(loadedPage.pageIndex);
            if (loadedPage.pageIndex >= pageCount_ || pageIter == pages_.end()) {
                ++stats_.framePageLoadFailureCount;
                ++stats_.totalPageLoadFailureCount;
                continue;
            }
            PageEntry& page = pageIter->second;
            const bool validPayload = loadedPage.success() &&
                page.state == MeshletStreamPageResidencyState::Unloaded &&
                pageAllocated(loadedPage.pageIndex) &&
                page.queued &&
                loadedPage.payload.size() == page.deviceSizeBytes &&
                loadedPage.payload.size() <= page.allocationBytes;
            if (!validPayload) {
                page.queued = false;
                releasePageStorage(loadedPage.pageIndex);
                ++stats_.framePageLoadFailureCount;
                ++stats_.totalPageLoadFailureCount;
                continue;
            }
            preparedPageLoads_.push_back(std::move(loadedPage));
        }
    }

    if (maxUploads == 0) {
        if (queuedUploadCount() != 0) {
            ++stats_.frameTransferBudgetFailureCount;
            ++stats_.totalTransferBudgetFailureCount;
        }
        return 0;
    }
    if ((asynchronousLoads && preparedPageLoads_.empty()) ||
        (!asynchronousLoads && uploadQueue_.empty())) {
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
    std::vector<uint8_t> decompressedPayload;
    uint32_t uploadCount = 0;
    while (uploadCount < maxUploads &&
        (asynchronousLoads ? !preparedPageLoads_.empty() : !uploadQueue_.empty())) {
        const uint32_t pageIndex = asynchronousLoads
            ? preparedPageLoads_.front().pageIndex
            : uploadQueue_.front();
        if (!asynchronousLoads) {
            uploadQueue_.pop_front();
        }
        auto pageIter = pages_.find(pageIndex);
        if (pageIter == pages_.end()) {
            if (asynchronousLoads) {
                preparedPageLoads_.pop_front();
            }
            continue;
        }
        PageEntry& page = pageIter->second;

        if (page.state != MeshletStreamPageResidencyState::Unloaded ||
            !pageAllocated(pageIndex) ||
            !page.queued) {
            page.queued = false;
            if (asynchronousLoads) {
                preparedPageLoads_.pop_front();
            }
            continue;
        }

        std::span<const uint8_t> devicePayload;
        if (asynchronousLoads) {
            devicePayload = preparedPageLoads_.front().payload;
        } else {
            const scene::MeshletStreamPageInfo& assetPage = asset_->pages()[pageIndex];
            const std::span<const uint8_t> storedPayload = asset_->pagePayload(pageIndex);
            std::string decodeReason;
            if (!scene::decodeMeshletStreamPayloadForDevice(
                    assetPage,
                    storedPayload,
                    decompressedPayload,
                    devicePayload,
                    decodeReason) ||
                devicePayload.empty() ||
                devicePayload.size() != page.deviceSizeBytes ||
                devicePayload.size() > page.allocationBytes ||
                devicePayload.size() > std::numeric_limits<uint32_t>::max()) {
                page.queued = false;
                releasePageStorage(pageIndex);
                ++stats_.frameTransferBudgetFailureCount;
                ++stats_.totalTransferBudgetFailureCount;
                ++stats_.framePageLoadFailureCount;
                ++stats_.totalPageLoadFailureCount;
                continue;
            }
        }

        const StreamDataChunk chunk{
            .data = devicePayload.data(),
            .size = static_cast<uint64_t>(devicePayload.size()),
        };
        const BufferOffset streamed = streamer.streamBufferData(StreamBufferDataDesc{
            .dataChunks = &chunk,
            .dataChunkCount = 1,
            .placementAlignment = 16,
            .dstBuffer = &destination,
            .dstOffset = page.deviceOffsetBytes,
        });
        if (!streamed.valid()) {
            if (!asynchronousLoads) {
                uploadQueue_.push_front(pageIndex);
            }
            ++stats_.frameTransferBudgetFailureCount;
            ++stats_.totalTransferBudgetFailureCount;
            break;
        }

        page.queued = false;
        if (asynchronousLoads) {
            preparedPageLoads_.pop_front();
        }
        setPageState(pageIndex, MeshletStreamPageResidencyState::PendingUpload);
        page.taskIndex = taskIndex;
        taskPages.push_back(pageIndex);
        ++uploadCount;
        ++stats_.frameScheduledUploadCount;
        ++stats_.totalScheduledUploadCount;
    }

    if (uploadCount == 0) {
        taskPages.clear();
        storageTaskQueue_.releaseTaskIndex(taskIndex);
        updateTaskQueue_.releaseTaskIndex(updateTaskIndex);
        return 0;
    }
    if (queuedUploadCount() != 0 && uploadCount >= maxUploads) {
        ++stats_.frameTransferBudgetFailureCount;
        ++stats_.totalTransferBudgetFailureCount;
    }

    const uint32_t frameDelay = std::max(streamer.desc().queuedFrameCount, queuedFrameCount_);
    storageTaskQueue_.push(taskIndex, frameIndex_ + frameDelay, updateTaskIndex);
    if (asynchronousLoads) {
        schedulePageLoads();
    }
    return uploadCount;
}

void MeshletStreamResidencyManager::buildInitialPageTable(std::span<StreamPageTableEntry> outEntries) const
{
    if (asset_ == nullptr || outEntries.size() < pageCount_) {
        return;
    }

    std::fill_n(outEntries.begin(), pageCount_, StreamPageTableEntry{});
    for (const auto& [pageIndex, page] : pages_) {
        const bool tableResident = page.state != MeshletStreamPageResidencyState::Unloaded && pageAllocated(pageIndex);
        outEntries[pageIndex] = StreamPageTableEntry{
            .deviceOffsetBytes = tableResident
                ? static_cast<uint32_t>(page.deviceOffsetBytes)
                : kInvalidStreamDeviceOffsetBytes,
            .metadata = packStreamPageTableMetadata(
                tableResident ? page.deviceSizeBytes : 0u,
                page.state),
            .lastRequestFrame = 0,
        };
    }
}

MeshletStreamPageResidencyState MeshletStreamResidencyManager::pageState(uint32_t pageIndex) const
{
    if (pageIndex >= pageCount_) {
        return MeshletStreamPageResidencyState::Unloaded;
    }
    const auto pageIter = pages_.find(pageIndex);
    return pageIter != pages_.end()
        ? pageIter->second.state
        : MeshletStreamPageResidencyState::Unloaded;
}

uint64_t MeshletStreamResidencyManager::deviceOffsetForPage(uint32_t pageIndex) const
{
    if (pageIndex >= pageCount_) {
        return UINT64_MAX;
    }
    const auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return UINT64_MAX;
    }
    const uint32_t deviceOffset = pageIter->second.deviceOffsetBytes;
    return deviceOffset == kInvalidStreamDeviceOffsetBytes ? UINT64_MAX : deviceOffset;
}

uint32_t MeshletStreamResidencyManager::deviceSizeForPage(uint32_t pageIndex) const
{
    if (pageIndex >= pageCount_) {
        return 0;
    }
    const auto pageIter = pages_.find(pageIndex);
    return pageIter != pages_.end() ? pageIter->second.deviceSizeBytes : 0u;
}

bool MeshletStreamResidencyManager::pageAllocated(uint32_t pageIndex) const
{
    if (pageIndex >= pageCount_) {
        return false;
    }
    const auto pageIter = pages_.find(pageIndex);
    return pageIter != pages_.end() &&
        pageIter->second.deviceOffsetBytes != kInvalidStreamDeviceOffsetBytes;
}

bool MeshletStreamResidencyManager::pageResident(uint32_t pageIndex) const
{
    return residentState(pageState(pageIndex));
}

uint64_t MeshletStreamResidencyManager::pageAge(uint32_t pageIndex) const
{
    if (pageIndex >= pageCount_) {
        return 0;
    }
    const auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return 0;
    }
    const uint64_t lastUsedFrame = pageIter->second.lastUsedFrame;
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

uint32_t MeshletStreamResidencyManager::queuedUploadCount() const
{
    const uint64_t count = static_cast<uint64_t>(uploadQueue_.size()) +
        pageLoader_.outstandingCount() +
        preparedPageLoads_.size();
    return count > std::numeric_limits<uint32_t>::max()
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(count);
}

MeshletStreamResidencyStats MeshletStreamResidencyManager::stats() const
{
    MeshletStreamResidencyStats result = stats_;
    result.frameIndex = frameIndex_;
    result.pageCount = pageCount_;
    result.trackedPageCount = static_cast<uint32_t>(pages_.size());
    result.maxResidentPages = maxResidentPages_;
    result.maxResidentBytes = storage_.capacityBytes();
    result.usedResidentBytes = storage_.usedBytes();
    result.freeResidentBytes = storage_.freeBytes();
    result.largestFreeBlockBytes = storage_.largestFreeBlockBytes();
    result.storageAllocationCount = storage_.allocationCount();
    result.storageFreeBlockCount = storage_.freeBlockCount();
    result.usedSlotCount = static_cast<uint32_t>(activePages_.size());
    const uint32_t slotCapacity = maxResidentPages_ != 0
        ? maxResidentPages_
        : pageCount_;
    result.freeSlotCount = slotCapacity >= result.usedSlotCount
        ? slotCapacity - result.usedSlotCount
        : 0u;
    result.activePageCount = static_cast<uint32_t>(activePages_.size());
    result.residentPageCount = static_cast<uint32_t>(residentPages_.size());
    result.pendingPageCount = static_cast<uint32_t>(pendingPages_.size());
    result.queuedUploadCount = queuedUploadCount();
    result.pageLoadWorkerCount = pageLoader_.workerCount();
    result.pendingPageLoadCount = pageLoader_.pendingCount();
    result.activePageLoadCount = pageLoader_.activeCount();
    result.completedPageLoadCount = pageLoader_.completedCount();
    result.preparedPageLoadCount = static_cast<uint32_t>(preparedPageLoads_.size());
    result.queuedRequestTaskCount = requestTaskQueue_.queuedTaskCount();
    result.availableRequestTaskCount = requestTaskQueue_.availableTaskCount();
    result.queuedStorageTaskCount = storageTaskQueue_.queuedTaskCount();
    result.availableStorageTaskCount = storageTaskQueue_.availableTaskCount();
    result.queuedUnloadTaskCount = unloadTaskQueue_.queuedTaskCount();
    result.availableUnloadTaskCount = unloadTaskQueue_.availableTaskCount();
    result.queuedUpdateTaskCount = updateTaskQueue_.queuedTaskCount();
    result.availableUpdateTaskCount = updateTaskQueue_.availableTaskCount();
    result.pendingPatchCount = static_cast<uint32_t>(patches_.size());
    result.oldestActiveAge = oldestAge(activePages_);
    result.oldestResidentAge = oldestAge(residentPages_);
    result.oldestPendingAge = oldestAge(pendingPages_);
    return result;
}

bool MeshletStreamResidencyManager::allocatePageStorage(uint32_t pageIndex)
{
    if (pageIndex >= pageCount_) {
        return false;
    }
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return false;
    }
    PageEntry& page = pageIter->second;
    if (pageAllocated(pageIndex)) {
        return true;
    }

    const scene::MeshletStreamPageInfo& assetPage = asset_->pages()[pageIndex];
    const uint64_t requiredBytes = storage_.allocationSize(assetPage.uncompressedSize);
    const bool pageBudgetReached = maxResidentPages_ != 0 && activePages_.size() >= maxResidentPages_;
    const bool storageBudgetReached = !storage_.canAllocate(assetPage.uncompressedSize);
    if (pageBudgetReached || storageBudgetReached) {
        uint64_t oldestFrame = std::numeric_limits<uint64_t>::max();
        uint32_t evictPage = UINT32_MAX;
        bool rejectedByAge = false;
        for (uint32_t candidate : residentPages_) {
            const auto candidateIter = pages_.find(candidate);
            if (candidateIter == pages_.end()) {
                continue;
            }
            const PageEntry& entry = candidateIter->second;
            if (entry.lockedFallback ||
                !pageAllocated(candidate) ||
                !streamableEvictionState(entry.state)) {
                continue;
            }
            if (storageBudgetReached && storage_.freeBytes() + entry.allocationBytes < requiredBytes) {
                continue;
            }
            const uint64_t age = pageAge(candidate);
            if (age < evictionAgeThresholdFrames_) {
                rejectedByAge = true;
                continue;
            }
            if (entry.lastUsedFrame < oldestFrame) {
                oldestFrame = entry.lastUsedFrame;
                evictPage = candidate;
            }
        }
        if (evictPage == UINT32_MAX) {
            if (rejectedByAge) {
                ++stats_.frameEvictionAgeRejectedCount;
                ++stats_.totalEvictionAgeRejectedCount;
            }
            ++stats_.frameResidentBudgetFailureCount;
            ++stats_.totalResidentBudgetFailureCount;
            ++stats_.frameAllocationFailureCount;
            ++stats_.totalAllocationFailureCount;
            return false;
        }
        if (!scheduleUnload(evictPage, true)) {
            ++stats_.frameResidentBudgetFailureCount;
            ++stats_.totalResidentBudgetFailureCount;
            ++stats_.frameAllocationFailureCount;
            ++stats_.totalAllocationFailureCount;
            return false;
        }
        ++stats_.frameEvictedPageCount;
        ++stats_.totalEvictedPageCount;
        return false;
    }

    MeshletStreamStorageAllocation allocation = storage_.allocate(assetPage.uncompressedSize);
    if (!allocation.valid()) {
        ++stats_.frameResidentBudgetFailureCount;
        ++stats_.totalResidentBudgetFailureCount;
        ++stats_.frameAllocationFailureCount;
        ++stats_.totalAllocationFailureCount;
        return false;
    }

    if (allocation.offset > std::numeric_limits<uint32_t>::max() ||
        allocation.allocatedSize > std::numeric_limits<uint32_t>::max()) {
        storage_.release(allocation);
        ++stats_.frameAllocationFailureCount;
        ++stats_.totalAllocationFailureCount;
        return false;
    }
    page.deviceOffsetBytes = static_cast<uint32_t>(allocation.offset);
    page.allocationBytes = static_cast<uint32_t>(allocation.allocatedSize);
    page.deviceSizeBytes = static_cast<uint32_t>(allocation.requestedSize);
    addToTable(activePages_, &PageEntry::activeTablePosition, pageIndex);
    return true;
}

bool MeshletStreamResidencyManager::scheduleUnload(uint32_t pageIndex, bool eviction)
{
    if (pageIndex >= pageCount_) {
        return false;
    }

    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return false;
    }
    PageEntry& page = pageIter->second;
    if (page.lockedFallback ||
        !pageAllocated(pageIndex) ||
        (page.state == MeshletStreamPageResidencyState::Unloaded && !page.queued)) {
        return false;
    }
    if (page.state == MeshletStreamPageResidencyState::PendingUnload) {
        return true;
    }

    const uint32_t taskIndex = unloadTaskQueue_.acquireTaskIndex();
    if (taskIndex == kInvalidStreamingTaskIndex) {
        ++stats_.frameUnloadTaskFailureCount;
        ++stats_.totalUnloadTaskFailureCount;
        if (eviction) {
            ++stats_.frameResidentBudgetFailureCount;
            ++stats_.totalResidentBudgetFailureCount;
        }
        return false;
    }

    std::vector<uint32_t>& taskPages = unloadTaskPages_[taskIndex];
    taskPages.clear();
    taskPages.push_back(pageIndex);
    page.taskIndex = taskIndex;
    page.queued = false;
    setPageState(pageIndex, MeshletStreamPageResidencyState::PendingUnload);
    unloadTaskQueue_.push(taskIndex, frameIndex_ + unloadDelayFrames_);
    ++stats_.frameScheduledUnloadCount;
    ++stats_.totalScheduledUnloadCount;
    return true;
}

void MeshletStreamResidencyManager::completeUnloadTask(uint32_t taskIndex)
{
    if (taskIndex >= unloadTaskPages_.size()) {
        return;
    }

    for (uint32_t pageIndex : unloadTaskPages_[taskIndex]) {
        auto pageIter = pages_.find(pageIndex);
        if (pageIter == pages_.end()) {
            continue;
        }
        PageEntry& page = pageIter->second;
        if (page.state != MeshletStreamPageResidencyState::PendingUnload ||
            page.taskIndex != taskIndex) {
            continue;
        }
        page.taskIndex = kInvalidStreamingTaskIndex;
        releasePageStorage(pageIndex);
        newlyUnloadedPages_.push_back(pageIndex);
        ++stats_.frameCompletedUnloadCount;
        ++stats_.totalCompletedUnloadCount;
        ++stats_.frameDelayedFreeCount;
        ++stats_.totalDelayedFreeCount;
    }
    unloadTaskPages_[taskIndex].clear();
}

void MeshletStreamResidencyManager::releasePageStorage(uint32_t pageIndex)
{
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }
    PageEntry& page = pageIter->second;
    if (!pageAllocated(pageIndex)) {
        return;
    }
    const MeshletStreamPageResidencyState oldState = page.state;
    storage_.release(MeshletStreamStorageAllocation{
        .offset = page.deviceOffsetBytes,
        .requestedSize = page.deviceSizeBytes,
        .allocatedSize = page.allocationBytes,
    });
    page.deviceOffsetBytes = kInvalidStreamDeviceOffsetBytes;
    page.allocationBytes = 0;
    page.deviceSizeBytes = 0;
    page.taskIndex = kInvalidStreamingTaskIndex;
    page.queued = false;
    removeFromTable(activePages_, &PageEntry::activeTablePosition, pageIndex);
    setPageState(pageIndex, MeshletStreamPageResidencyState::Unloaded);
    if (oldState == MeshletStreamPageResidencyState::Unloaded) {
        recordPatch(pageIndex);
    }
    if (!page.lockedFallback) {
        pages_.erase(pageIndex);
    }
}

void MeshletStreamResidencyManager::setPageState(uint32_t pageIndex, MeshletStreamPageResidencyState state)
{
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }
    PageEntry& page = pageIter->second;
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
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }
    PageEntry& page = pageIter->second;
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
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }
    const PageEntry& page = pageIter->second;
    const bool tableResident = page.state != MeshletStreamPageResidencyState::Unloaded && pageAllocated(pageIndex);
    patches_.push_back(StreamPageTablePatch{
        .pageId = pageIndex,
        .deviceOffsetBytes = tableResident
            ? static_cast<uint32_t>(page.deviceOffsetBytes)
            : kInvalidStreamDeviceOffsetBytes,
        .deviceSizeBytes = tableResident ? page.deviceSizeBytes : 0u,
        .state = static_cast<uint32_t>(page.state),
    });
}

void MeshletStreamResidencyManager::addToTable(
    std::vector<uint32_t>& table,
    PagePositionMember positionMember,
    uint32_t pageIndex)
{
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }
    uint32_t& position = pageIter->second.*positionMember;
    if (position != kInvalidTablePosition) {
        return;
    }
    position = static_cast<uint32_t>(table.size());
    table.push_back(pageIndex);
}

void MeshletStreamResidencyManager::removeFromTable(
    std::vector<uint32_t>& table,
    PagePositionMember positionMember,
    uint32_t pageIndex)
{
    auto pageIter = pages_.find(pageIndex);
    if (pageIter == pages_.end()) {
        return;
    }

    uint32_t& position = pageIter->second.*positionMember;
    if (position == kInvalidTablePosition) {
        return;
    }
    const uint32_t movedPage = table.back();
    table[position] = movedPage;
    auto movedIter = pages_.find(movedPage);
    if (movedIter != pages_.end()) {
        movedIter->second.*positionMember = position;
    }
    table.pop_back();
    position = kInvalidTablePosition;
}

void MeshletStreamResidencyManager::updateStateTables(
    uint32_t pageIndex,
    MeshletStreamPageResidencyState oldState,
    MeshletStreamPageResidencyState newState)
{
    if (residentState(oldState) && !residentState(newState)) {
        removeFromTable(residentPages_, &PageEntry::stateTablePosition, pageIndex);
    }
    if (oldState == MeshletStreamPageResidencyState::PendingUpload &&
        newState != MeshletStreamPageResidencyState::PendingUpload) {
        removeFromTable(pendingPages_, &PageEntry::stateTablePosition, pageIndex);
    }
    if (!residentState(oldState) && residentState(newState)) {
        addToTable(residentPages_, &PageEntry::stateTablePosition, pageIndex);
    }
    if (oldState != MeshletStreamPageResidencyState::PendingUpload &&
        newState == MeshletStreamPageResidencyState::PendingUpload) {
        addToTable(pendingPages_, &PageEntry::stateTablePosition, pageIndex);
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
    stats_.frameScheduledUnloadCount = 0;
    stats_.frameCompletedUnloadCount = 0;
    stats_.frameUnloadTaskFailureCount = 0;
    stats_.frameDelayedFreeCount = 0;
    stats_.frameEvictionAgeRejectedCount = 0;
    stats_.frameResidentBudgetFailureCount = 0;
    stats_.frameTransferBudgetFailureCount = 0;
    stats_.frameEvictedPageCount = 0;
    stats_.frameAllocationFailureCount = 0;
    stats_.frameScheduledPageLoadCount = 0;
    stats_.frameCompletedPageLoadCount = 0;
    stats_.framePageLoadFailureCount = 0;
}

} // namespace metallic::render
