#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Render/GAPI/StreamUploadCompletion.h"
#include "Runtime/Render/Profiling/CpuProfile.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
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

// Screen benefit per resident byte with bounded request aging. Zero-benefit
// dependencies still progress when visible requests no longer consume admission.
double pageBenefitPerByte(float benefit, uint64_t bytes, uint64_t age)
{
    return std::max(benefit, 0.0f) * (1.0 + std::min<uint64_t>(age, 120u) / 30.0) /
        double(std::max<uint64_t>(bytes, 1024u));
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
    freeBlockBoundsValid_ = false;
}

MeshletStreamStorageAllocation MeshletStreamStorage::allocate(uint64_t byteSize)
{
    const uint64_t alignedSize = allocationSize(byteSize);
    if (alignedSize == 0 || !canAllocate(byteSize)) {
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
        // Other blocks cannot become larger when allocating. Preserve a valid
        // bound unless this allocation consumes one of its maximizers.
        if (block.size == largestFreeBlockBytes_ || block.size - prefixBytes == largestAllocatableBytes_) {
            freeBlockBoundsValid_ = false;
        }
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
    freeBlockBoundsValid_ = false;
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
    if (byteSize == 0 || byteSize > UINT64_MAX - (alignmentBytes_ - 1u)) {
        return 0;
    }
    return alignUp(byteSize, alignmentBytes_);
}

bool MeshletStreamStorage::canAllocate(uint64_t byteSize) const
{
    const uint64_t alignedSize = allocationSize(byteSize);
    if (alignedSize == 0) { return false; }
    updateFreeBlockBounds();
    return alignedSize <= largestAllocatableBytes_;
}

void MeshletStreamStorage::updateFreeBlockBounds() const
{
    if (freeBlockBoundsValid_) { return; }
    largestFreeBlockBytes_ = 0;
    largestAllocatableBytes_ = 0;
    for (const FreeBlock& block : freeBlocks_) {
        largestFreeBlockBytes_ = std::max(largestFreeBlockBytes_, block.size);
        const uint64_t alignedOffset = alignUp(block.offset, alignmentBytes_);
        if (alignedOffset < block.offset) {
            continue;
        }
        const uint64_t prefixBytes = alignedOffset - block.offset;
        if (prefixBytes <= block.size) { largestAllocatableBytes_ = std::max(largestAllocatableBytes_, block.size - prefixBytes); }
    }
    freeBlockBoundsValid_ = true;
}

uint64_t MeshletStreamStorage::largestFreeBlockBytes() const
{
    updateFreeBlockBounds();
    return largestFreeBlockBytes_;
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
    if (desc.pageLoadConcurrency > kMeshletStreamMaxPageLoadConcurrency) {
        reason = "MeshletStreamResidencyManager page load concurrency exceeds the supported limit";
        return false;
    }
    if (desc.storageAlignment < kStreamPageTableOffsetAlignment ||
        desc.storageAlignment % kStreamPageTableOffsetAlignment != 0) {
        reason = "MeshletStreamResidencyManager storage alignment cannot pack page state into offsets";
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
    immediateGpuRequests_ = desc.immediateGpuRequests;
    completionDrivenUploads_ = desc.completionDrivenUploads;
    gpuDecompression_ = desc.gpuDecompression && completionDrivenUploads_;
    gpuDecompressionMinBatchBytes_ = desc.gpuDecompressionMinBatchBytes;
    throughput_.sample(meshletStreamTimeMicroseconds(), traffic_);
    maxResidentPages_ = desc.maxResidentPages;
    queuedFrameCount_ = std::max(desc.queuedFrameCount, 1u);
    if (desc.measurePageLatency) { latency_ = std::make_unique<MeshletStreamLatencyTracker>(uint64_t(queuedFrameCount_) * 2u + 2u); }
    unloadDelayFrames_ = std::max(desc.unloadDelayFrames, 1u);
    evictionAgeThresholdFrames_ = desc.evictionAgeThresholdFrames;
    pageCount_ = asset_->pageCount();
    if (latency_) { latencyExcludedPages_.assign((uint64_t(pageCount_) + 63) / 64, 0); }
    unloadRequestBits_.assign((uint64_t(pageCount_) + 63) / 64, 0);
    requestIndexBlocks_.resize((uint64_t(pageCount_) + kRequestIndexBlockSize - 1) / kRequestIndexBlockSize);
    if (maxResidentPages_ != 0) {
        const uint32_t residentReserve = std::min(maxResidentPages_, pageCount_);
        pages_.reserve(residentReserve);
        activePages_.reserve(residentReserve);
        residentPages_.reserve(residentReserve);
        residentPageEntries_.reserve(residentReserve);
        pendingPages_.reserve(residentReserve);
    }
    if (desc.pageLoadConcurrency != 0) {
        const uint32_t pageLoadCapacity = maxResidentPages_ != 0
            ? std::min(maxResidentPages_, pageCount_)
            : pageCount_;
        maxPageLoadsInFlight_ = std::min(
            std::max(desc.maxPageLoadsInFlight, desc.pageLoadConcurrency),
            pageLoadCapacity);
        if (!pageLoader_.initialize(*asset_, desc.pageLoadConcurrency, reason, gpuDecompression_)) {
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
    latency_.reset();
    latencyExcludedPages_.clear();
    immediateGpuRequests_ = false;
    completionDrivenUploads_ = true;
    gpuDecompression_ = false;
    gpuDecompressionMinBatchBytes_ = 0;
    traffic_ = {};
    throughput_ = {};
    asset_ = nullptr;
    storage_.reset();
    residentPageEntries_.clear();
    pages_.clear();
    uploadQueue_.clear();
    preparedPageLoads_.clear();
    requestTaskQueue_.reset();
    residentDemandFeedback_ = false;
    geometryReclaimPressure_ = clasReclaimPressure_ = false;
    for (auto& taskPages : requestTaskPages_) {
        taskPages.clear();
    }
    for (std::vector<uint32_t>& taskPages : requestTaskUnloadPages_) {
        taskPages.clear();
    }
    storageTaskQueue_.reset();
    storageCompletions_ = {};
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
    evictionCandidates_.clear();
    evictionCandidateCursor_ = 0;
    evictionCandidatesBuilt_ = false;
    evictionAgeRejected_ = false;
    budgetAdmissionExhausted_ = false;
    frameUnloadTaskIndex_ = kInvalidStreamingTaskIndex;
    requestIndexBlocks_.clear();
    requestScratch_.clear();
    unloadRequestBits_.clear();
    unloadRequestTouchedWords_.clear();
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

void MeshletStreamResidencyManager::beginFrame(CpuProfileRecorder* profiler)
{
    CpuProfileScope profile(profiler, "Reset / latency tracking");
    ++frameIndex_;
    throughput_.sample(meshletStreamTimeMicroseconds(), traffic_);
    if (latency_) {
        latency_->frameTimes[frameIndex_ % latency_->frameTimes.size()] = {frameIndex_, meshletStreamTimeMicroseconds()};
        latency_->expire(frameIndex_, [this](uint32_t id) {
            const auto page = pages_.find(id);
            return page != pages_.end() && (page->second.queued ||
                page->second.state == MeshletStreamPageResidencyState::PendingUpload);
        });
    }
    requestedPages_.clear();
    unloadRequestedPages_.clear();
    newlyResidentPages_.clear();
    newlyUnloadedPages_.clear();
    resetFrameStats();
    evictionCandidates_.clear();
    evictionCandidateCursor_ = 0;
    evictionCandidatesBuilt_ = false;
    evictionAgeRejected_ = false;
    budgetAdmissionExhausted_ = false;
    frameUnloadTaskIndex_ = kInvalidStreamingTaskIndex;

    profile.next("Complete update tasks");
    while (updateTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = updateTaskQueue_.pop();
        if (taskIndex < updateTaskPages_.size()) {
            completeUploadPages(updateTaskPages_[taskIndex], taskIndex);
            updateTaskPages_[taskIndex].clear();
        }
        updateTaskQueue_.releaseTaskIndex(taskIndex);
    }

    profile.next("Complete storage tasks");
    while (!storageTaskQueue_.empty()) {
        const uint32_t frontIndex = storageTaskQueue_.frontTaskIndex();
        const auto completion = storageCompletions_[frontIndex];
        if (completion) {
            const bool cancelled = completion->isCancelled();
            if (!cancelled && !completion->isComplete()) {
                break;
            }
            uint32_t updateTaskIndex = kInvalidStreamingTaskIndex;
            const uint32_t taskIndex = storageTaskQueue_.popWithDependent(updateTaskIndex);
            if (cancelled) {
                for (uint32_t pageIndex : storageTaskPages_[taskIndex]) {
                    auto pageIter = pages_.find(pageIndex);
                    if (pageIter == pages_.end() ||
                        pageIter->second.state != MeshletStreamPageResidencyState::PendingUpload ||
                        pageIter->second.taskIndex != taskIndex) {
                        continue;
                    }
                    // No copy was submitted. Keep the bounded allocation and
                    // request priority, and retry both demand and locked roots.
                    pageIter->second.taskIndex = kInvalidStreamingTaskIndex;
                    setPageState(pageIndex, MeshletStreamPageResidencyState::Unloaded);
                    queueUpload(pageIndex);
                    ++stats_.totalCancelledUploads;
                }
            } else {
                stats_.totalCompletionDrivenUploads += storageTaskPages_[taskIndex].size();
                completeUploadPages(storageTaskPages_[taskIndex], taskIndex);
                ++stats_.frameCompletedStorageTaskCount;
                ++stats_.totalCompletedStorageTaskCount;
                ++stats_.frameScheduledUpdateCount;
                ++stats_.totalScheduledUpdateCount;
            }
            storageTaskPages_[taskIndex].clear();
            storageCompletions_[taskIndex].reset();
            storageTaskQueue_.releaseTaskIndex(taskIndex);
            updateTaskQueue_.releaseTaskIndex(updateTaskIndex);
            continue;
        }
        if (!storageTaskQueue_.canPop(frameIndex_, true)) {
            break;
        }
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
        // Preserve the frame-delayed protocol for untracked legacy callers.
        break;
    }

    profile.next("Complete unload tasks");
    while (unloadTaskQueue_.canPop(frameIndex_, true)) {
        const uint32_t taskIndex = unloadTaskQueue_.pop();
        completeUnloadTask(taskIndex);
        unloadTaskQueue_.releaseTaskIndex(taskIndex);
    }

    profile.next("Consume queued requests");
    consumeReadyRequestTasks(profiler);
}

void MeshletStreamResidencyManager::completeUploadPages(
    std::span<const uint32_t> pageIndices, uint32_t taskIndex)
{
    for (uint32_t pageIndex : pageIndices) {
        auto pageIter = pages_.find(pageIndex);
        if (pageIter == pages_.end()) { continue; }
        PageEntry& page = pageIter->second;
        if (page.state != MeshletStreamPageResidencyState::PendingUpload || page.taskIndex != taskIndex) {
            continue;
        }
        page.taskIndex = kInvalidStreamingTaskIndex;
        setPageState(pageIndex, page.lockedFallback
            ? MeshletStreamPageResidencyState::LockedFallback
            : MeshletStreamPageResidencyState::Resident);
        newlyResidentPages_.push_back(pageIndex);
        ++traffic_.geometryReadyPages;
        traffic_.geometryReadyBytes += page.deviceSizeBytes;
        page.residentSinceFrame = frameIndex_;
        if (latency_) { latency_->complete(pageIndex, frameIndex_); }
        ++stats_.frameCompletedUpdateCount;
        ++stats_.frameCompletedUploadCount;
        ++stats_.totalCompletedUpdateCount;
        ++stats_.totalCompletedUploadCount;
    }
}

void MeshletStreamResidencyManager::consumeReadyRequestTasks(CpuProfileRecorder* profiler)
{
    if (requestTaskQueue_.canPop(frameIndex_, true)) {
        CpuProfileScope profile(profiler, "Select latest request task");
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
            profile.next("Apply explicit unloads");
            for (uint32_t pageIndex : requestTaskUnloadPages_[latestTaskIndex]) {
                if (pageIndex >= pageCount_) {
                    continue;
                }
                unloadRequestedPages_.push_back(pageIndex);
                if (unloadPage(pageIndex)) {
                    ++consumedUnloads;
                }
            }
            auto& taskPages = requestTaskPages_[latestTaskIndex];
            profile.next("Refresh request priorities");
            // Refresh ALL demand before cancelling queued I/O or choosing victims.
            // Allocated/in-flight requests need no sorting or repeated admission.
            const bool prioritized = std::any_of(taskPages.begin(), taskPages.end(),
                [](const PageRequest& request) { return request.screenBenefit >= 0.0f || request.prefetch; });
            size_t candidateCount = 0;
            for (auto request : taskPages) {
                requestedPages_.push_back(request.pageIndex);
                const auto found = pages_.find(request.pageIndex);
                if (found != pages_.end()) {
                    auto& page = found->second;
                    page.lastUsedFrame = frameIndex_;
                    page.screenBenefit = request.screenBenefit;
                    if (!request.prefetch && page.prefetch) {
                        ++stats_.totalPrefetchUsed;
                        page.prefetch = false;
                    }
                    request.needsAllocation = page.deviceOffsetBytes == kInvalidStreamDeviceOffsetBytes;
                    if (residentState(page.state) || page.state == MeshletStreamPageResidencyState::PendingUpload ||
                        page.state == MeshletStreamPageResidencyState::PendingUnload ||
                        (!request.needsAllocation && page.queued)) {
                        if (page.state == MeshletStreamPageResidencyState::PendingUnload) {
                            ++stats_.frameResidentBudgetFailureCount;
                            ++stats_.totalResidentBudgetFailureCount;
                        }
                        ++stats_.cpuWork.admissionBypassed;
                        ++consumed;
                        continue;
                    }
                }
                request.payloadBytes = scene::meshletStreamDevicePayloadSize(asset_->pages()[request.pageIndex]);
                if (prioritized) {
                    const uint64_t age = found == pages_.end() ? 0u : frameIndex_ - found->second.firstRequestFrame;
                    request.schedulingPriority = pageBenefitPerByte(request.screenBenefit, request.payloadBytes, age);
                    if (request.prefetch) {
                        request.prefetchLod = asset_->groups()[asset_->pages()[request.pageIndex].lodGroupIndex].lodLevel;
                    }
                }
                taskPages[candidateCount++] = request;
            }
            taskPages.resize(candidateCount);
            stats_.cpuWork.admissionCandidates += static_cast<uint32_t>(candidateCount);
            profile.next("Cancel stale queued loads");
            std::erase_if(uploadQueue_, [this](uint32_t pageIndex) {
                const auto found = pages_.find(pageIndex);
                if (found == pages_.end()) { return true; }
                if (!found->second.lockedFallback &&
                    pageAge(pageIndex) > uint64_t(queuedFrameCount_) * 2u + 2u) {
                    releasePageStorage(pageIndex);
                    ++stats_.frameCancelledQueuedLoadCount;
                    ++stats_.totalCancelledQueuedLoadCount;
                    return true;
                }
                return false;
            });
            // The heap produces exactly the old total priority order, but only
            // selects entries that can execute. No full N log N sort each frame.
            const auto lowerPriority = [](const PageRequest& a, const PageRequest& b) {
                if (a.prefetch != b.prefetch) { return a.prefetch; }
                if (a.prefetch && a.prefetchLod != b.prefetchLod) { return a.prefetchLod < b.prefetchLod; }
                return a.schedulingPriority != b.schedulingPriority
                    ? a.schedulingPriority < b.schedulingPriority : a.pageIndex > b.pageIndex;
            };
            enum class DeferredReason { None, Prefetch, Io, Budget };
            const auto deferredReason = [&](const PageRequest& request) {
                if (!request.needsAllocation) { return DeferredReason::None; }
                if (request.prefetch &&
                    (storage_.usedBytes() + storage_.allocationSize(request.payloadBytes) > storage_.capacityBytes() * 3u / 4u ||
                        !storage_.canAllocate(request.payloadBytes) ||
                        (pageLoader_.ready() && queuedUploadCount() >= std::max(maxPageLoadsInFlight_ / 4u, 1u)) ||
                        (maxResidentPages_ != 0 && activePages_.size() + 1 > uint64_t(maxResidentPages_) * 3u / 4u))) {
                    return DeferredReason::Prefetch;
                }
                if (pageLoader_.ready() && queuedUploadCount() >= uint64_t(maxPageLoadsInFlight_) * 2u) {
                    return DeferredReason::Io;
                }
                if (budgetAdmissionExhausted_ &&
                    ((maxResidentPages_ != 0 && activePages_.size() >= maxResidentPages_) ||
                        !storage_.canAllocate(request.payloadBytes))) { return DeferredReason::Budget; }
                return DeferredReason::None;
            };
            const auto filterBlocked = [&] {
                std::erase_if(taskPages, [&](const PageRequest& request) {
                    const auto reason = deferredReason(request);
                    if (reason == DeferredReason::None) { return false; }
                    ++stats_.cpuWork.admissionBypassed;
                    if (reason == DeferredReason::Prefetch) { ++stats_.totalPrefetchDeferred; }
                    else {
                        ++consumed;
                        if (reason == DeferredReason::Io) { ++stats_.frameAdmissionDeferredCount; }
                        else {
                            ++stats_.cpuWork.budgetRetrySuppressed;
                            ++stats_.frameAllocationDeferredCount;
                        }
                    }
                    return true;
                });
            };
            profile.next("Filter blocked admission");
            filterBlocked();
            profile.next("Prepare admission heap");
            if (prioritized) { std::make_heap(taskPages.begin(), taskPages.end(), lowerPriority); }
            profile.next("Admit demand / prefetch");
            while (!taskPages.empty()) {
                const auto& next = prioritized ? taskPages.front() : taskPages.back();
                if (deferredReason(next) != DeferredReason::None) {
                    // Eligibility can only tighten in this loop: frees/publication
                    // run outside it. Remove the blocked remainder in one linear
                    // pass; retain smaller fits and their deterministic priority.
                    filterBlocked();
                    if (prioritized) { std::make_heap(taskPages.begin(), taskPages.end(), lowerPriority); }
                    continue;
                }
                if (prioritized) {
                    std::pop_heap(taskPages.begin(), taskPages.end(), lowerPriority);
                    ++stats_.cpuWork.admissionPriorityPops;
                }
                const auto request = taskPages.back();
                taskPages.pop_back();
                ++stats_.cpuWork.admissionCalls;
                (void)requestPage(request.pageIndex);
                const auto admitted = pages_.find(request.pageIndex);
                if (admitted != pages_.end()) {
                    admitted->second.screenBenefit = request.screenBenefit;
                    if (request.needsAllocation) {
                        admitted->second.prefetch = request.prefetch;
                        stats_.totalPrefetchAdmitted += request.prefetch;
                    }
                }
                ++consumed;
            }
            profile.next("Release request task");
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
            const uint64_t pageBytes = scene::meshletStreamDevicePayloadSize(asset_->pages()[pageIndex]);
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
        updateLatencyEligibility(pageIndex, page);
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

    auto pageIter = pages_.find(pageIndex);
    const bool allocated = pageIter != pages_.end() &&
        pageIter->second.deviceOffsetBytes != kInvalidStreamDeviceOffsetBytes;
    if (!allocated) {
        // Do not create and erase a hash node for every over-budget request.
        // This is a capacity gate, not a per-ID cooldown: new priorities are
        // honored immediately and a large rejection cannot block smaller pages.
        if (pageLoader_.ready() && queuedUploadCount() >= uint64_t(maxPageLoadsInFlight_) * 2u) {
            ++stats_.frameAdmissionDeferredCount;
            return false;
        }
        if (budgetAdmissionExhausted_ &&
            ((maxResidentPages_ != 0 && activePages_.size() >= maxResidentPages_) ||
                !storage_.canAllocate(scene::meshletStreamDevicePayloadSize(asset_->pages()[pageIndex])))) {
            ++stats_.cpuWork.budgetRetrySuppressed;
            ++stats_.frameAllocationDeferredCount;
            return false;
        }
    }
    const bool inserted = pageIter == pages_.end();
    if (inserted) { pageIter = pages_.try_emplace(pageIndex).first; }
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

    if (!allocated && !allocatePageStorage(pageIndex)) {
        if (inserted) {
            pages_.erase(pageIter);
        }
        return false;
    }
    if (!page.queued) {
        page.firstRequestFrame = frameIndex_;
        if (latency_) {
            const auto sample = latency_->find(pageIndex);
            if (sample != nullptr && sample->admissionTime == 0) {
                sample->admissionTime = meshletStreamTimeMicroseconds();
                latency_->observe(MeshletStreamLatencyStage::Admission, sample->feedbackTime, sample->admissionTime);
            }
        }
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

uint32_t MeshletStreamResidencyManager::consumeGpuRequests(const StreamGpuRequestBatch& requests, CpuProfileRecorder* profiler)
{
    CpuProfileScope profile(profiler, "Deduplicate loads");
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

    auto& uniqueRequests = requestScratch_;
    CpuProfileScope detail(profiler, "Prepare load containers");
    uniqueRequests.clear();
    uniqueRequests.reserve(requests.loadPageIds.size());
    detail.next("Validate / merge loads");
    for (size_t index = 0; index < requests.loadPageIds.size(); ++index) {
        const uint32_t encodedPage = requests.loadPageIds[index];
        const bool prefetch = requests.taggedPrefetchRequests && (encodedPage & kStreamPrefetchPageTag) != 0;
        const uint32_t pageIndex = requests.taggedPrefetchRequests ? encodedPage & ~kStreamPrefetchPageTag : encodedPage;
        if (pageIndex >= pageCount_) {
            ++stats_.frameGpuInvalidRequestCount;
            ++stats_.totalGpuInvalidRequestCount;
            continue;
        }
        float benefit = index < requests.loadPriorities.size() ? requests.loadPriorities[index] : -1.0f;
        if (!std::isfinite(benefit)) { benefit = 0.0f; }
        benefit = std::clamp(benefit, -1.0f, 1e9f);
        auto& block = requestIndexBlocks_[pageIndex / kRequestIndexBlockSize];
        if (!block) {
            block = std::make_unique<RequestIndexBlock>();
            block->fill(UINT32_MAX);
        }
        uint32_t& slot = (*block)[pageIndex % kRequestIndexBlockSize];
        if (slot < uniqueRequests.size() && uniqueRequests[slot].pageIndex == pageIndex) {
            auto& merged = uniqueRequests[slot];
            merged.screenBenefit = std::max(merged.screenBenefit, benefit);
            merged.prefetch = merged.prefetch && prefetch;
            ++stats_.cpuWork.requestDuplicatesMerged;
            continue;
        }
        slot = static_cast<uint32_t>(uniqueRequests.size());
        uniqueRequests.push_back({.pageIndex = pageIndex, .screenBenefit = benefit, .prefetch = prefetch});
        requestedPages_.push_back(pageIndex);
    }
    // Track the final merged demand once, including prefetch -> demand promotion.
    // Resident checks and latency tracking do not run for each duplicate.
    detail.next("Track merged demand");
    if (latency_) {
        const uint64_t feedbackTime = meshletStreamTimeMicroseconds();
        for (const auto& request : uniqueRequests) {
            const uint64_t bit = uint64_t(1) << (request.pageIndex % 64);
            if ((latencyExcludedPages_[request.pageIndex / 64] & bit) == 0) {
                latency_->request(request.pageIndex, requests.frameIndex, frameIndex_, request.prefetch, feedbackTime);
            }
        }
    }
    stats_.frameUniqueGpuRequestCount += static_cast<uint32_t>(uniqueRequests.size());
    stats_.totalUniqueGpuRequestCount += uniqueRequests.size();

    detail.end();
    profile.next("Deduplicate unloads");
    std::vector<uint32_t> uniqueUnloadRequests;
    CpuProfileScope unloadDetail(profiler, "Prepare unused-page set");
    uniqueUnloadRequests.reserve(requests.unloadPageIds.size());
    for (uint32_t word : unloadRequestTouchedWords_) { unloadRequestBits_[word] = 0; }
    unloadRequestTouchedWords_.clear();
    unloadRequestTouchedWords_.reserve(std::min(requests.unloadPageIds.size(), unloadRequestBits_.size()));
    unloadDetail.next("Validate / insert unused pages");
    for (uint32_t pageIndex : requests.unloadPageIds) {
        if (pageIndex >= pageCount_) {
            ++stats_.frameGpuInvalidRequestCount;
            ++stats_.totalGpuInvalidRequestCount;
            continue;
        }
        const uint32_t wordIndex = pageIndex / 64;
        const uint64_t bit = uint64_t(1) << (pageIndex % 64);
        uint64_t& word = unloadRequestBits_[wordIndex];
        if (word & bit) { continue; }
        if (!word) { unloadRequestTouchedWords_.push_back(wordIndex); }
        word |= bit;
        uniqueUnloadRequests.push_back(pageIndex);
        unloadRequestedPages_.push_back(pageIndex);
    }
    stats_.frameUniqueGpuUnloadRequestCount += static_cast<uint32_t>(uniqueUnloadRequests.size());
    stats_.totalUniqueGpuUnloadRequestCount += uniqueUnloadRequests.size();

    unloadDetail.end();
    profile.next("Update resident demand");
    if (requests.residentDemandFeedback) {
        residentDemandFeedback_ = true;
        bool newColdPage = false;
        const bool complete = requests.unloadOverflowCounter == 0 && requests.invalidPageCounter == 0 &&
            requests.unloadRequestCounter <= requests.unloadPageIds.size();
        for (size_t i = 0; i < residentPages_.size(); ++i) {
            ++stats_.cpuWork.demandVisited;
            const uint32_t pageIndex = residentPages_[i];
            PageEntry& page = *residentPageEntries_[i];
            // This page was not in an older frame's resident list. Its absence
            // from that frame's unused list cannot imply a prefetch hit.
            if (requests.frameIndex != 0 && page.residentSinceFrame > requests.frameIndex) {
                ++stats_.cpuWork.demandNewerThanFeedback;
                continue;
            }
            // Only explicitly unused pages may be budget victims. Truncated
            // feedback must not infer that an omitted page is cold.
            const bool unused = (unloadRequestBits_[pageIndex / 64] & (uint64_t(1) << (pageIndex % 64))) != 0;
            newColdPage |= unused && !page.gpuUnused;
            page.gpuUnused = unused;
            if (page.gpuUnused) {
                ++stats_.cpuWork.demandUnused;
                ++stats_.frameCachedUnusedPageCount;
            } else if (complete) {
                ++stats_.cpuWork.demandRefreshed;
                page.lastUsedFrame = frameIndex_;
                if (page.prefetch) { ++stats_.totalPrefetchUsed; page.prefetch = false; }
                ++stats_.frameResidentDemandCount;
            } else {
                ++stats_.cpuWork.demandIncompleteProtected;
            }
        }
        // New feedback can expose victims after admission exhausted the snapshot.
        if (newColdPage && evictionCandidatesBuilt_) {
            evictionCandidates_.clear();
            evictionCandidateCursor_ = 0;
            evictionCandidatesBuilt_ = false;
            evictionAgeRejected_ = false;
            budgetAdmissionExhausted_ = false;
        }
        uniqueUnloadRequests.clear();
    }

    profile.next("Admit request batch");
    if (uniqueRequests.empty() && uniqueUnloadRequests.empty()) {
        return 0;
    }

    CpuProfileScope admission(profiler, "Queue request task");
    const uint32_t taskIndex = requestTaskQueue_.acquireTaskIndex();
    if (taskIndex == kInvalidStreamingTaskIndex) {
        ++stats_.frameRequestTaskFailureCount;
        ++stats_.totalRequestTaskFailureCount;
        return 0;
    }

    auto& taskPages = requestTaskPages_[taskIndex];
    taskPages.swap(uniqueRequests);
    uniqueRequests.clear();
    std::vector<uint32_t>& taskUnloadPages = requestTaskUnloadPages_[taskIndex];
    taskUnloadPages = std::move(uniqueUnloadRequests);
    requestTaskQueue_.push(taskIndex, frameIndex_ + (immediateGpuRequests_ ? 0u : 1u));
    ++stats_.frameScheduledRequestTaskCount;
    ++stats_.totalScheduledRequestTaskCount;
    const uint32_t count = static_cast<uint32_t>(taskPages.size() + taskUnloadPages.size());
    if (immediateGpuRequests_) {
        admission.next("Consume ready request tasks");
        consumeReadyRequestTasks(profiler);
    }
    return count;
}

uint32_t MeshletStreamResidencyManager::processUploads(
    Streamer& streamer,
    Buffer& destination,
    uint32_t maxUploads,
    const UploadObserver& observer, CpuProfileRecorder* profiler,
    uint64_t maxUploadBytesPerFrame, const GpuUploadObserver& gpuObserver)
{
    CpuProfileScope profile(profiler, "Sort queued loads");
    if (asset_ == nullptr) {
        return 0;
    }

    const bool asynchronousLoads = pageLoader_.ready();
    // Roots first, then recently demanded pages, with aging within a live
    // request batch. For equal age, prefer the cheaper payload. stable_sort
    // preserves request order when priorities are equal.
    const auto higherPriority = [this](uint32_t a, uint32_t b) {
        const auto ia = pages_.find(a), ib = pages_.find(b);
        if (ia == pages_.end() || ib == pages_.end()) { return ia != pages_.end(); }
        const PageEntry& pa = ia->second; const PageEntry& pb = ib->second;
        if (pa.lockedFallback != pb.lockedFallback) { return pa.lockedFallback; }
        if (pa.prefetch != pb.prefetch) { return !pa.prefetch; }
        if (pa.prefetch) {
            const auto la = asset_->groups()[asset_->pages()[a].lodGroupIndex].lodLevel;
            const auto lb = asset_->groups()[asset_->pages()[b].lodGroupIndex].lodLevel;
            if (la != lb) { return la > lb; }
        }
        if (pa.lastUsedFrame != pb.lastUsedFrame) { return pa.lastUsedFrame > pb.lastUsedFrame; }
        if (pa.screenBenefit >= 0.0f || pb.screenBenefit >= 0.0f) {
            const double left = pageBenefitPerByte(pa.screenBenefit, pa.deviceSizeBytes, frameIndex_ - pa.firstRequestFrame);
            const double right = pageBenefitPerByte(pb.screenBenefit, pb.deviceSizeBytes, frameIndex_ - pb.firstRequestFrame);
            if (left != right) { return left > right; }
        }
        if (pa.firstRequestFrame != pb.firstRequestFrame) { return pa.firstRequestFrame < pb.firstRequestFrame; }
        return pa.deviceSizeBytes < pb.deviceSizeBytes;
    };
    std::stable_sort(uploadQueue_.begin(), uploadQueue_.end(), higherPriority);
    const auto useGpuForCohort = [&]() {
        if (!gpuDecompression_) { return false; }
        if (gpuDecompressionMinBatchBytes_ == 0) { return true; }
        const uint64_t outstanding = pageLoader_.outstandingCount() + preparedPageLoads_.size();
        const uint64_t capacity = asynchronousLoads
            ? (outstanding < maxPageLoadsInFlight_ ? maxPageLoadsInFlight_ - outstanding : 0)
            : maxUploads;
        uint64_t bytes = 0, encodedBytes = 0, count = 0;
        // Preview only the admissible, priority-ordered cohort. No waiting to
        // fill a batch; decisions travel with each worker request. The upload
        // completion order can still split a large cohort over several frames.
        for (uint32_t id : uploadQueue_) {
            if (count >= std::min<uint64_t>(capacity, maxUploads)) { break; }
            const auto found = pages_.find(id);
            if (found == pages_.end()) { continue; }
            const auto& page = found->second;
            if (page.state != MeshletStreamPageResidencyState::Unloaded || !page.queued || !pageAllocated(id)) { continue; }
            if (asynchronousLoads && page.prefetch && outstanding + count >= std::max(maxPageLoadsInFlight_ / 4u, 1u)) { break; }
            if (maxUploadBytesPerFrame != 0 && (bytes != 0 || stats_.frameUploadBytes != 0) &&
                (stats_.frameUploadBytes >= maxUploadBytesPerFrame ||
                    bytes >= maxUploadBytesPerFrame - stats_.frameUploadBytes ||
                    page.deviceSizeBytes > maxUploadBytesPerFrame - stats_.frameUploadBytes - bytes)) { break; }
            bytes += page.deviceSizeBytes;
            ++count;
            if (asset_->pages()[id].compressionMode == uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles)) {
                encodedBytes += page.deviceSizeBytes;
            }
            if (encodedBytes >= gpuDecompressionMinBatchBytes_) { return true; }
        }
        return false;
    };
    auto schedulePageLoads = [&]() {
        const bool useGpu = useGpuForCohort();
        while (!uploadQueue_.empty() &&
            static_cast<uint64_t>(pageLoader_.outstandingCount()) + preparedPageLoads_.size() <
                maxPageLoadsInFlight_) {
            const uint32_t pageIndex = uploadQueue_.front();
            const auto next = pages_.find(pageIndex);
            if (next != pages_.end() && next->second.prefetch &&
                uint64_t(pageLoader_.outstandingCount()) + preparedPageLoads_.size() >= std::max(maxPageLoadsInFlight_ / 4u, 1u)) {
                break;
            }
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
            const uint64_t enqueueTime = latency_ ? meshletStreamTimeMicroseconds() : 0;
            if (!pageLoader_.enqueue(pageIndex, useGpu)) {
                uploadQueue_.push_front(pageIndex);
                break;
            }
            if (latency_) {
                const auto sample = latency_->find(pageIndex);
                if (sample != nullptr) { sample->enqueueTime = enqueueTime; }
            }
            ++stats_.frameScheduledPageLoadCount;
            ++stats_.totalScheduledPageLoadCount;
        }
    };

    profile.next("Schedule page I/O");
    if (asynchronousLoads) {
        schedulePageLoads();
        profile.next("Collect decoded pages");
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
            if (latency_) {
                const auto sample = latency_->find(loadedPage.pageIndex);
                if (sample != nullptr) {
                    latency_->observe(MeshletStreamLatencyStage::IoQueue, sample->enqueueTime, loadedPage.startedMicroseconds);
                    latency_->observe(MeshletStreamLatencyStage::Decode, loadedPage.startedMicroseconds, loadedPage.completedMicroseconds);
                }
            }
            const bool validPayload = loadedPage.success() &&
                page.state == MeshletStreamPageResidencyState::Unloaded &&
                pageAllocated(loadedPage.pageIndex) &&
                page.queued &&
                loadedPage.payload.size() == (loadedPage.gpuEncoded
                    ? asset_->pages()[loadedPage.pageIndex].payloadSize : page.deviceSizeBytes) &&
                page.deviceSizeBytes <= page.allocationBytes;
            if (!validPayload) {
                page.queued = false;
                releasePageStorage(loadedPage.pageIndex);
                ++stats_.framePageLoadFailureCount;
                ++stats_.totalPageLoadFailureCount;
                continue;
            }
            ++traffic_.loadedPages;
            traffic_.loadedStoredBytes += asset_->pages()[loadedPage.pageIndex].payloadSize;
            traffic_.preparedDeviceBytes += page.deviceSizeBytes;
            if (gpuDecompression_ && !loadedPage.gpuEncoded &&
                asset_->pages()[loadedPage.pageIndex].compressionMode == uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles)) {
                ++traffic_.smallBatchCpuPages;
            }
            preparedPageLoads_.push_back(std::move(loadedPage));
        }
    }

    profile.next("Sort ready uploads");
    if (asynchronousLoads) {
        std::stable_sort(preparedPageLoads_.begin(), preparedPageLoads_.end(),
            [&](const auto& a, const auto& b) { return higherPriority(a.pageIndex, b.pageIndex); });
    }

    profile.next("Prepare transfer batch");
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
    const bool synchronousGpu = !asynchronousLoads && useGpuForCohort();
    profile.next("Stage payloads / CLAS plans");
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

        // Use device bytes, including legacy float4 -> float3 conversion. A
        // single oversized page may make progress in an otherwise empty frame;
        // subsequent calls in the same frame share the already spent budget.
        if (maxUploadBytesPerFrame != 0 && stats_.frameUploadBytes != 0 &&
            (stats_.frameUploadBytes >= maxUploadBytesPerFrame ||
                page.deviceSizeBytes > maxUploadBytesPerFrame - stats_.frameUploadBytes)) {
            if (!asynchronousLoads) { uploadQueue_.push_front(pageIndex); }
            ++stats_.frameTransferBudgetFailureCount;
            ++stats_.totalTransferBudgetFailureCount;
            break;
        }

        std::span<const uint8_t> devicePayload;
        scene::MeshletStreamGpuPage synchronousGpuPage;
        const scene::MeshletStreamGpuPage* gpuPage = nullptr;
        if (asynchronousLoads) {
            devicePayload = preparedPageLoads_.front().payload;
            if (preparedPageLoads_.front().gpuEncoded) { gpuPage = &preparedPageLoads_.front().gpuPage; }
        } else {
            const scene::MeshletStreamPageInfo& assetPage = asset_->pages()[pageIndex];
            const std::span<const uint8_t> storedPayload = asset_->pagePayload(pageIndex);
            std::string decodeReason;
            const bool encoded = synchronousGpu &&
                assetPage.compressionMode == uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles);
            const bool decoded = encoded
                ? scene::inspectMeshletStreamGpuPage(assetPage, storedPayload, synchronousGpuPage, decodeReason)
                : scene::decodeMeshletStreamPayloadForDevice(
                    assetPage,
                    storedPayload,
                    decompressedPayload,
                    devicePayload,
                    decodeReason);
            if (encoded && decoded) {
                devicePayload = storedPayload;
                gpuPage = &synchronousGpuPage;
            }
            if (!decoded || devicePayload.empty() ||
                (!encoded && devicePayload.size() != page.deviceSizeBytes) ||
                page.deviceSizeBytes > page.allocationBytes ||
                devicePayload.size() > std::numeric_limits<uint32_t>::max()) {
                page.queued = false;
                releasePageStorage(pageIndex);
                ++stats_.frameTransferBudgetFailureCount;
                ++stats_.totalTransferBudgetFailureCount;
                ++stats_.framePageLoadFailureCount;
                ++stats_.totalPageLoadFailureCount;
                continue;
            }
            ++traffic_.loadedPages;
            traffic_.loadedStoredBytes += assetPage.payloadSize;
            traffic_.preparedDeviceBytes += page.deviceSizeBytes;
            if (gpuDecompression_ && !encoded &&
                assetPage.compressionMode == uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles)) { ++traffic_.smallBatchCpuPages; }
        }

        const StreamDataChunk chunk{
            .data = devicePayload.data(),
            .size = static_cast<uint64_t>(devicePayload.size()),
        };
        bool staged = false;
        if (gpuPage) {
            std::vector<StreamDecompressionTile> tiles;
            tiles.reserve(gpuPage->tiles.size());
            for (const auto& tile : gpuPage->tiles) {
                tiles.push_back({tile.sourceOffset, tile.destinationOffset, tile.storedBytes, tile.decodedBytes, tile.codec != 0});
            }
            staged = streamer.streamDecompressedBufferData(devicePayload, tiles, destination, page.deviceOffsetBytes);
        } else {
            staged = streamer.streamBufferData(StreamBufferDataDesc{
                .dataChunks = &chunk,
                .dataChunkCount = 1,
                .placementAlignment = 16,
                .dstBuffer = &destination,
                .dstOffset = page.deviceOffsetBytes,
            }).valid();
        }
        if (!staged) {
            if (!asynchronousLoads) {
                uploadQueue_.push_front(pageIndex);
            }
            ++stats_.frameTransferBudgetFailureCount;
            ++stats_.totalTransferBudgetFailureCount;
            break;
        }

        if (gpuPage) {
            if (gpuObserver) { gpuObserver(pageIndex, *gpuPage); }
            for (const auto& tile : gpuPage->tiles) { traffic_.transferPayloadBytes += tile.storedBytes; }
            ++stats_.frameGpuDecompressedPages;
            ++stats_.totalGpuDecompressedPages;
        } else {
            traffic_.transferPayloadBytes += devicePayload.size();
            if (observer) { observer(pageIndex, devicePayload); }
        }
        if (latency_) {
            const auto sample = latency_->find(pageIndex);
            if (sample != nullptr) {
                sample->uploadTime = meshletStreamTimeMicroseconds();
                if (asynchronousLoads) {
                    latency_->observe(MeshletStreamLatencyStage::ReadyToUpload,
                        preparedPageLoads_.front().completedMicroseconds, sample->uploadTime);
                }
            }
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
        stats_.frameUploadBytes += page.deviceSizeBytes;
        stats_.totalUploadBytes += page.deviceSizeBytes;
        stats_.frameStoredUploadBytes += chunk.size;
        stats_.totalStoredUploadBytes += chunk.size;
    }

    profile.next("Submit upload tracking");
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
    storageCompletions_[taskIndex] = completionDrivenUploads_ ? streamer.pendingCopyCompletion() : nullptr;
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
            .deviceOffsetAndState = packStreamPageTableEntry(
                tableResident
                    ? static_cast<uint32_t>(page.deviceOffsetBytes)
                    : kInvalidStreamDeviceOffsetBytes,
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

MeshletStreamResidencyStats MeshletStreamResidencyManager::stats(bool detailed) const
{
    MeshletStreamResidencyStats result = stats_;
    result.frameIndex = frameIndex_;
    result.pageCount = pageCount_;
    result.trackedPageCount = static_cast<uint32_t>(pages_.size());
    result.maxResidentPages = maxResidentPages_;
    result.maxResidentBytes = storage_.capacityBytes();
    result.usedResidentBytes = storage_.usedBytes();
    result.freeResidentBytes = storage_.freeBytes();
    result.largestFreeBlockBytes = detailed ? storage_.largestFreeBlockBytes() : 0;
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
    result.pageLoadConcurrency = pageLoader_.concurrency();
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
    if (detailed) {
        result.oldestActiveAge = oldestAge(activePages_);
        result.oldestResidentAge = oldestAge(residentPages_);
        result.oldestPendingAge = oldestAge(pendingPages_);
    }
    return result;
}

size_t MeshletStreamResidencyManager::prepareEvictionCandidates(CpuProfileRecorder* profiler, uint32_t minimumAge)
{
    if (!evictionCandidatesBuilt_) {
        CpuProfileScope profile(profiler, "Scan resident candidates");
        evictionCandidatesBuilt_ = true;
        evictionSortedCount_ = 0;
        evictionSortedMinimumAge_ = UINT64_MAX;
        ++stats_.frameEvictionScanCount;
        for (size_t i = 0; i < residentPages_.size(); ++i) {
            const uint32_t candidate = residentPages_[i];
            ++stats_.frameEvictionCandidateTests;
            const PageEntry& entry = *residentPageEntries_[i];
            if (entry.lockedFallback || !streamableEvictionState(entry.state) ||
                (residentDemandFeedback_ && !entry.gpuUnused)) { continue; }
            const uint64_t age = frameIndex_ >= entry.lastUsedFrame ? frameIndex_ - entry.lastUsedFrame : 0;
            if (age < evictionAgeThresholdFrames_) {
                evictionAgeRejected_ = true;
                continue;
            }
            evictionCandidates_.push_back({entry.lastUsedFrame, candidate});
            ++stats_.cpuWork.coldCandidates;
        }
    }
    CpuProfileScope profile(profiler, "Sort cold candidates");
    const auto due = [this, minimumAge](const EvictionCandidate& candidate) {
        return minimumAge == 0 || (frameIndex_ >= candidate.lastUsedFrame &&
            frameIndex_ - candidate.lastUsedFrame >= minimumAge);
    };
    const auto older = [](const EvictionCandidate& a, const EvictionCandidate& b) {
        return a.lastUsedFrame != b.lastUsedFrame ? a.lastUsedFrame < b.lastUsedFrame : a.pageIndex < b.pageIndex;
    };
    // Keep young candidates for later allocation pressure, but sort only the
    // due prefix. A lower age threshold expands it without another page scan.
    if (minimumAge < evictionSortedMinimumAge_) {
        auto middle = evictionCandidates_.begin() + evictionSortedCount_;
        auto end = std::partition(middle, evictionCandidates_.end(), due);
        std::sort(middle, end, older);
        if (middle != evictionCandidates_.begin() && middle != end) {
            std::inplace_merge(evictionCandidates_.begin(), middle, end, older);
        }
        evictionSortedCount_ = static_cast<size_t>(end - evictionCandidates_.begin());
        evictionSortedMinimumAge_ = minimumAge;
    }
    return static_cast<size_t>(std::partition_point(evictionCandidates_.begin(),
        evictionCandidates_.begin() + evictionSortedCount_, due) - evictionCandidates_.begin());
}

uint32_t MeshletStreamResidencyManager::reclaimColdPages(const MeshletStreamColdPageReclaimDesc& desc, CpuProfileRecorder* profiler)
{
    CpuProfileScope profile(profiler, "Credit pending frees");
    if (!residentDemandFeedback_ || !desc.retentionFrames || !desc.maxPages) { return 0; }
    const auto subtract = [](uint64_t a, uint64_t b) { return a > b ? a - b : 0; };
    uint64_t geometry = storage_.usedBytes();
    uint64_t clas = subtract(desc.clasUsedBytes, desc.clasRetiringBytes);
    // Credit already scheduled frees before selecting more victims. A delayed
    // free must not drive repeated evictions while its GPU readers drain.
    for (uint32_t taskIndex = 0; taskIndex < unloadTaskPages_.size(); ++taskIndex) {
        for (uint32_t id : unloadTaskPages_[taskIndex]) {
            const auto found = pages_.find(id);
            if (found == pages_.end()) { continue; }
            const auto& page = found->second;
            if (page.state != MeshletStreamPageResidencyState::PendingUnload || page.taskIndex != taskIndex) { continue; }
            ++stats_.cpuWork.pendingFreePages;
            geometry = subtract(geometry, page.allocationBytes);
            if (desc.clasPageBytes) { clas = subtract(clas, desc.clasPageBytes(id)); }
        }
    }
    profile.next("Evaluate budget pressure");
    const uint64_t geometryTarget = storage_.capacityBytes() * 70 / 100;
    const uint64_t clasTarget = desc.clasCapacityBytes * 70 / 100;
    const auto pressure = [](bool current, uint64_t used, uint64_t capacity) {
        return capacity && (current ? used > capacity * 70 / 100 : used >= capacity * 85 / 100);
    };
    geometryReclaimPressure_ = pressure(geometryReclaimPressure_, geometry, storage_.capacityBytes());
    clasReclaimPressure_ = pressure(clasReclaimPressure_, clas, desc.clasCapacityBytes);
    const auto minimumPossibleAge = [&]() {
        return (geometryReclaimPressure_ && geometry > geometryTarget) ||
            (clasReclaimPressure_ && clas > clasTarget)
            ? std::min(desc.pressureAgeFrames, desc.retentionFrames) : desc.retentionFrames;
    };
    profile.next("Prepare cold candidates");
    const size_t dueCount = prepareEvictionCandidates(profiler, minimumPossibleAge());
    profile.next("Schedule cold evictions");
    uint32_t reclaimed = 0;
    for (size_t i = 0; i < dueCount; ++i) {
        const EvictionCandidate& candidate = evictionCandidates_[i];
        const uint32_t id = candidate.pageIndex;
        if (reclaimed >= desc.maxPages || stats_.frameEvictedPageCount >= 256) { break; }
        const uint32_t possibleAge = minimumPossibleAge();
        // Pressure can end after a victim is credited. Only the sorted snapshot
        // key proves that every following candidate is too young as well.
        if (possibleAge && (frameIndex_ < candidate.lastUsedFrame ||
            frameIndex_ - candidate.lastUsedFrame < possibleAge)) { break; }
        ++stats_.cpuWork.coldVisited;
        const auto& page = pages_.at(id);
        if (page.lockedFallback || !page.gpuUnused || !streamableEvictionState(page.state)) {
            ++stats_.cpuWork.coldStateRejected;
            continue;
        }
        const uint64_t age = pageAge(id);
        if (age < possibleAge || frameIndex_ - page.residentSinceFrame < desc.pressureAgeFrames) {
            ++stats_.cpuWork.coldAgeRejected;
            continue;
        }
        stats_.cpuWork.coldClasLookups += bool(desc.clasPageBytes);
        const uint64_t clasBytes = desc.clasPageBytes ? desc.clasPageBytes(id) : 0;
        const bool needGeometry = geometryReclaimPressure_ && geometry > geometryTarget;
        const bool needClas = clasReclaimPressure_ && clas > clasTarget && clasBytes > 0;
        const uint64_t minimumAge = needGeometry || needClas ? desc.pressureAgeFrames : desc.retentionFrames;
        // Recently uploaded / recently used pages must survive delayed feedback.
        if (age < minimumAge) {
            ++stats_.cpuWork.coldAgeRejected;
            continue;
        }
        const uint64_t geometryBytes = page.allocationBytes;
        if (!scheduleUnload(id, true)) {
            ++stats_.cpuWork.coldScheduleFailed;
            break;
        }
        if (needGeometry || needClas) { ++stats_.cpuWork.coldPressureScheduled; }
        else { ++stats_.cpuWork.coldRetentionScheduled; }
        geometry = subtract(geometry, geometryBytes); clas = subtract(clas, clasBytes);
        ++reclaimed; ++stats_.frameEvictedPageCount; ++stats_.totalEvictedPageCount;
    }
    return reclaimed;
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
    ++stats_.cpuWork.allocationAttempts;
    const bool pageBudgetReached = maxResidentPages_ != 0 && activePages_.size() >= maxResidentPages_;
    const bool storageBudgetReached = !storage_.canAllocate(scene::meshletStreamDevicePayloadSize(assetPage));
    if (pageBudgetReached || storageBudgetReached) {
        uint32_t evictPage = UINT32_MAX;
        prepareEvictionCandidates();
        // At most 256 evictions per frame; one delayed-free task batches them.
        // Retrying after an exhausted scan/task budget is constant time.
        while (evictionCandidateCursor_ < evictionCandidates_.size() && stats_.frameEvictedPageCount < 256u) {
            const uint32_t candidate = evictionCandidates_[evictionCandidateCursor_++].pageIndex;
            const auto candidateIter = pages_.find(candidate);
            if (candidateIter == pages_.end()) {
                continue;
            }
            const PageEntry& entry = candidateIter->second;
            if (entry.lockedFallback ||
                !pageAllocated(candidate) ||
                !streamableEvictionState(entry.state) || (residentDemandFeedback_ && !entry.gpuUnused)) {
                continue;
            }
            const uint64_t age = pageAge(candidate);
            if (age < evictionAgeThresholdFrames_) {
                evictionAgeRejected_ = true;
                continue;
            }
            evictPage = candidate;
            break;
        }
        if (evictPage == UINT32_MAX) {
            budgetAdmissionExhausted_ = true;
            ++stats_.frameAllocationDeferredCount;
            if (evictionAgeRejected_) {
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
            budgetAdmissionExhausted_ = true;
            evictionCandidateCursor_ = evictionCandidates_.size();
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

    MeshletStreamStorageAllocation allocation = storage_.allocate(scene::meshletStreamDevicePayloadSize(assetPage));
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
    page.gpuUnused = false;
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

    const bool newTask = frameUnloadTaskIndex_ == kInvalidStreamingTaskIndex;
    const uint32_t taskIndex = newTask ? unloadTaskQueue_.acquireTaskIndex() : frameUnloadTaskIndex_;
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
    if (newTask) { taskPages.clear(); }
    taskPages.push_back(pageIndex);
    page.taskIndex = taskIndex;
    page.queued = false;
    setPageState(pageIndex, MeshletStreamPageResidencyState::PendingUnload);
    if (newTask) {
        frameUnloadTaskIndex_ = taskIndex;
        unloadTaskQueue_.push(taskIndex, frameIndex_ + unloadDelayFrames_);
    }
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
    if (latency_) { latency_->abandon(pageIndex); }
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
    updateLatencyEligibility(pageIndex, page);
    updateStateTables(pageIndex, oldState, state);
    recordPatch(pageIndex);
}

void MeshletStreamResidencyManager::updateLatencyEligibility(uint32_t pageIndex, const PageEntry& page)
{
    if (!latency_) { return; }
    auto& word = latencyExcludedPages_[pageIndex / 64];
    const uint64_t bit = uint64_t(1) << (pageIndex % 64);
    if (residentState(page.state) || page.lockedFallback) { word |= bit; }
    else { word &= ~bit; }
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

void MeshletStreamResidencyManager::rebuildPendingPatches()
{
    patches_.clear();
    for (const auto& [page, entry] : pages_) { recordPatch(page); }
}

uint32_t MeshletStreamResidencyManager::buildOrderedUploadPatches(
    const CommandBuffer& commands, std::vector<StreamPageTablePatch>& outPatches) const
{
    outPatches = patches_;
    std::unordered_map<uint32_t, size_t> indices;
    // Keep only the latest state for each page: parallel GPU patch writes must
    // never race a PendingUpload entry against its drawable replacement.
    size_t count = 0;
    for (const auto patch : patches_) {
        const auto [entry, inserted] = indices.try_emplace(patch.pageId, count);
        outPatches[entry->second] = patch;
        if (inserted) { ++count; }
    }
    outPatches.resize(count);
    uint32_t ordered = 0;
    for (uint32_t task = 0; task < storageCompletions_.size(); ++task) {
        const auto& receipt = storageCompletions_[task];
        if (!receipt || !receipt->isRecordedBefore(commands)) { continue; }
        for (uint32_t pageIndex : storageTaskPages_[task]) {
            const auto found = pages_.find(pageIndex);
            if (found == pages_.end()) { continue; }
            const auto& page = found->second;
            if (page.state != MeshletStreamPageResidencyState::PendingUpload || page.taskIndex != task) { continue; }
            const StreamPageTablePatch patch{pageIndex, packStreamPageTableEntry(page.deviceOffsetBytes,
                page.lockedFallback ? MeshletStreamPageResidencyState::LockedFallback : MeshletStreamPageResidencyState::Resident)};
            const auto [entry, inserted] = indices.try_emplace(pageIndex, outPatches.size());
            if (inserted) { outPatches.push_back(patch); }
            else { outPatches[entry->second] = patch; }
            ++ordered;
        }
    }
    return ordered;
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
        .deviceOffsetAndState = packStreamPageTableEntry(
            tableResident
                ? static_cast<uint32_t>(page.deviceOffsetBytes)
                : kInvalidStreamDeviceOffsetBytes,
            page.state),
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
    if (&table == &residentPages_) {
        residentPageEntries_.push_back(&pageIter->second);
    }
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
    if (&table == &residentPages_) {
        residentPageEntries_[position] = residentPageEntries_.back();
        residentPageEntries_.pop_back();
    }
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
    stats_.frameEvictionScanCount = 0;
    stats_.frameEvictionCandidateTests = 0;
    stats_.cpuWork = {};
    stats_.frameAllocationDeferredCount = 0;
    stats_.frameCachedUnusedPageCount = 0;
    stats_.frameResidentDemandCount = 0;
    stats_.frameUploadBytes = 0;
    stats_.frameStoredUploadBytes = 0;
    stats_.frameGpuDecompressedPages = 0;
    stats_.frameAdmissionDeferredCount = 0;
    stats_.frameCancelledQueuedLoadCount = 0;
    stats_.frameScheduledPageLoadCount = 0;
    stats_.frameCompletedPageLoadCount = 0;
    stats_.framePageLoadFailureCount = 0;
}

} // namespace metallic::render
