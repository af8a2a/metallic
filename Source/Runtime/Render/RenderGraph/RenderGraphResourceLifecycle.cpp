#include "Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>

namespace metallic::render {
namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hashBytes(uint64_t& hash, std::span<const std::byte> bytes)
{
    for (std::byte value : bytes) {
        hash = (hash ^ std::to_integer<uint8_t>(value)) * kFnvPrime;
    }
}

void hashString(uint64_t& hash, std::string_view text)
{
    const uint64_t size = text.size();
    hashBytes(hash, std::as_bytes(std::span{&size, size_t{1}}));
    if (!text.empty()) {
        hashBytes(hash, std::as_bytes(std::span{text.data(), text.size()}));
    }
}

bool sameActualQueue(const Queue* lhs, const Queue* rhs)
{
    return lhs != nullptr && rhs != nullptr && lhs->sameQueue(*rhs);
}

void appendUniqueActiveCompletion(std::vector<GPUCompletionPoint>& destination, const GPUCompletionPoint& completion)
{
    if (!completion.valid() || completion.isCancelled() || !completion.isSubmitted() || completion.isComplete()) {
        return;
    }
    const bool exists = std::any_of(destination.begin(), destination.end(),
        [&](const GPUCompletionPoint& existing) { return existing.sameSubmission(completion); });
    if (!exists) {
        destination.push_back(completion);
    }
}

void appendUniqueSubmittedCompletion(std::vector<GPUCompletionPoint>& destination, const GPUCompletionPoint& completion)
{
    if (!completion.valid() || completion.isCancelled() || !completion.isSubmitted()) {
        return;
    }
    const bool exists = std::any_of(destination.begin(), destination.end(),
        [&](const GPUCompletionPoint& existing) { return existing.sameSubmission(completion); });
    if (!exists) {
        destination.push_back(completion);
    }
}

bool textureDescEqual(const TextureDesc& lhs, const TextureDesc& rhs)
{
    return lhs.type == rhs.type &&
        lhs.usage == rhs.usage &&
        lhs.format == rhs.format &&
        lhs.width == rhs.width &&
        lhs.height == rhs.height &&
        lhs.depth == rhs.depth &&
        lhs.mipCount == rhs.mipCount &&
        lhs.layerCount == rhs.layerCount &&
        lhs.memoryLocation == rhs.memoryLocation &&
        lhs.queueAccess == rhs.queueAccess;
}

bool textureViewDescEqual(const TextureViewDesc& lhs, const TextureViewDesc& rhs)
{
    return lhs.format == rhs.format &&
        lhs.range.baseMip == rhs.range.baseMip &&
        lhs.range.mipCount == rhs.range.mipCount &&
        lhs.range.baseLayer == rhs.range.baseLayer &&
        lhs.range.layerCount == rhs.range.layerCount &&
        lhs.forceArray == rhs.forceArray;
}

bool bufferDescEqual(const BufferDesc& lhs, const BufferDesc& rhs)
{
    return lhs.size == rhs.size &&
        lhs.structureStride == rhs.structureStride &&
        lhs.usage == rhs.usage &&
        lhs.memoryLocation == rhs.memoryLocation &&
        lhs.queueAccess == rhs.queueAccess;
}

bool bufferViewDescEqual(const std::optional<BufferViewDesc>& lhs, const std::optional<BufferViewDesc>& rhs)
{
    if (lhs.has_value() != rhs.has_value()) {
        return false;
    }
    if (!lhs.has_value()) {
        return true;
    }
    return lhs->type == rhs->type &&
        lhs->range.offset == rhs->range.offset &&
        lhs->range.size == rhs->range.size &&
        lhs->structureStride == rhs->structureStride;
}

void retireOwnedBacking(
    uint64_t backingAllocationId,
    std::shared_ptr<void> owner,
    const CrossFrameResourceTracker& tracker,
    DeferredReleaseQueue& deferredReleases)
{
    if (!owner) {
        return;
    }
    const auto completions = tracker.activeCompletions(backingAllocationId);
    for (const GPUCompletionPoint& completion : completions) {
        deferredReleases.retire(completion, owner);
    }
}

} // namespace

const char* renderGraphResourceOwnershipName(RenderGraphResourceOwnership ownership)
{
    switch (ownership) {
    case RenderGraphResourceOwnership::GraphOwned:
        return "Graph";
    case RenderGraphResourceOwnership::TemporalOwned:
        return "Temporal";
    case RenderGraphResourceOwnership::ProviderOwned:
        return "Provider";
    case RenderGraphResourceOwnership::ExternallyOwned:
        return "External";
    case RenderGraphResourceOwnership::OpaqueSDK:
        return "OpaqueSDK";
    }
    return "Unknown";
}

const char* renderGraphMemoryPolicyName(RenderGraphMemoryPolicy policy)
{
    switch (policy) {
    case RenderGraphMemoryPolicy::MinimizeMemory:
        return "MinimizeMemory";
    case RenderGraphMemoryPolicy::MaximizeOverlap:
        return "MaximizeOverlap";
    }
    return "Unknown";
}

LogicalResourceId makeLogicalResourceId(
    std::string_view ownerDomain,
    std::string_view name,
    uint64_t viewId,
    uint64_t contentKey)
{
    uint64_t hash = kFnvOffsetBasis;
    hashString(hash, ownerDomain);
    hashString(hash, name);
    hashBytes(hash, std::as_bytes(std::span{&viewId, size_t{1}}));
    hashBytes(hash, std::as_bytes(std::span{&contentKey, size_t{1}}));
    if (hash == 0) {
        hash = 1;
    }
    return LogicalResourceId{
        .value = hash,
        .viewId = viewId,
        .contentKey = contentKey,
        .ownerDomain = std::string(ownerDomain),
        .name = std::string(name),
    };
}

bool ResourceFrontierAccess::isActive() const
{
    if (status == FrontierEntryStatus::Complete) {
        return false;
    }
    if (status == FrontierEntryStatus::Recording) {
        return true;
    }
    return completion.valid() && completion.isSubmitted() && !completion.isCancelled() && !completion.isComplete();
}

bool ResourceFrontierAccess::hasSubmittedCompletion() const
{
    return completion.valid() && completion.isSubmitted() && !completion.isCancelled();
}

bool ResourceFrontierAccess::isVisibleTo(const Queue* targetQueue) const
{
    if (targetQueue == nullptr) {
        return true;
    }
    if (sameActualQueue(queue, targetQueue)) {
        return true;
    }
    return std::any_of(visibleQueues.begin(), visibleQueues.end(),
        [&](const Queue* visible) { return sameActualQueue(visible, targetQueue); });
}

bool ResourceFrontierAccess::needsVisibilityOn(const Queue* targetQueue) const
{
    return hasSubmittedCompletion() && !isVisibleTo(targetQueue);
}

void ResourceFrontierAccess::markVisibleTo(Queue* targetQueue)
{
    if (targetQueue == nullptr || isVisibleTo(targetQueue)) {
        return;
    }
    visibleQueues.push_back(targetQueue);
}

void ResourceAccessFrontier::pruneCompleted()
{
    if (lastWriter && lastWriter->status == FrontierEntryStatus::Accepted && lastWriter->completion.isComplete()) {
        lastWriter->status = FrontierEntryStatus::Complete;
    }
    if (layoutAnchor && layoutAnchor->status == FrontierEntryStatus::Accepted && layoutAnchor->completion.isComplete()) {
        layoutAnchor->status = FrontierEntryStatus::Complete;
    }
    std::erase_if(outstandingReaders, [](const ResourceFrontierAccess& reader) {
        return reader.status == FrontierEntryStatus::Complete ||
            (reader.status == FrontierEntryStatus::Accepted && reader.completion.isComplete());
    });
}

bool ResourceAccessFrontier::isPhysicalBackingIdle() const
{
    if (lastWriter && lastWriter->isActive()) {
        return false;
    }
    if (layoutAnchor && layoutAnchor->isActive()) {
        return false;
    }
    return std::none_of(outstandingReaders.begin(), outstandingReaders.end(),
        [](const ResourceFrontierAccess& reader) { return reader.isActive(); });
}

size_t ResourceAccessFrontier::activeReaderCount() const
{
    return static_cast<size_t>(std::count_if(outstandingReaders.begin(), outstandingReaders.end(),
        [](const ResourceFrontierAccess& reader) { return reader.isActive(); }));
}

std::vector<GPUCompletionPoint> ResourceAccessFrontier::activeCompletions() const
{
    std::vector<GPUCompletionPoint> completions;
    if (lastWriter && lastWriter->isActive()) {
        appendUniqueActiveCompletion(completions, lastWriter->completion);
    }
    if (layoutAnchor && layoutAnchor->isActive()) {
        appendUniqueActiveCompletion(completions, layoutAnchor->completion);
    }
    for (const ResourceFrontierAccess& reader : outstandingReaders) {
        if (reader.isActive()) {
            appendUniqueActiveCompletion(completions, reader.completion);
        }
    }
    return completions;
}

void ResourceAccessFrontier::collectWaitsForRead(
    Queue* consumerQueue,
    bool layoutChange,
    std::vector<GPUCompletionPoint>& outWaits,
    std::vector<std::string>* outReasons) const
{
    const std::string resourceLabel = logicalId.name.empty()
        ? ("backing#" + std::to_string(backingAllocationId))
        : logicalId.name;
    // Wait whenever consumerQueue has not yet established a device-side
    // semaphore dependency against the writer/layout anchor, even if the host
    // already observed the producer's timeline value as complete.
    if (lastWriter && lastWriter->needsVisibilityOn(consumerQueue)) {
        appendUniqueSubmittedCompletion(outWaits, lastWriter->completion);
        if (outReasons != nullptr) {
            outReasons->push_back("Cross-frame RAW on " + resourceLabel +
                " (frame " + std::to_string(lastWriter->frameIndex) + ")");
        }
    }
    if (layoutAnchor && layoutAnchor->needsVisibilityOn(consumerQueue)) {
        appendUniqueSubmittedCompletion(outWaits, layoutAnchor->completion);
        if (outReasons != nullptr) {
            outReasons->push_back("Cross-frame layout RAW on " + resourceLabel +
                " (frame " + std::to_string(layoutAnchor->frameIndex) + ")");
        }
    }
    if (layoutChange) {
        size_t remoteReaders = 0;
        for (const ResourceFrontierAccess& reader : outstandingReaders) {
            if (reader.isActive() && !sameActualQueue(reader.queue, consumerQueue)) {
                appendUniqueActiveCompletion(outWaits, reader.completion);
                ++remoteReaders;
            }
        }
        if (remoteReaders > 0 && outReasons != nullptr) {
            outReasons->push_back("Cross-frame layout WAR on " + resourceLabel +
                " (" + std::to_string(remoteReaders) + " active reader(s))");
        }
    }
}

void ResourceAccessFrontier::collectWaitsForWrite(
    Queue* writerQueue,
    std::vector<GPUCompletionPoint>& outWaits,
    std::vector<std::string>* outReasons) const
{
    const std::string resourceLabel = logicalId.name.empty()
        ? ("backing#" + std::to_string(backingAllocationId))
        : logicalId.name;
    if (lastWriter && lastWriter->needsVisibilityOn(writerQueue)) {
        appendUniqueSubmittedCompletion(outWaits, lastWriter->completion);
        if (outReasons != nullptr) {
            outReasons->push_back("Cross-frame WAW on " + resourceLabel +
                " (frame " + std::to_string(lastWriter->frameIndex) + ")");
        }
    }
    if (layoutAnchor && layoutAnchor->needsVisibilityOn(writerQueue)) {
        appendUniqueSubmittedCompletion(outWaits, layoutAnchor->completion);
        if (outReasons != nullptr) {
            outReasons->push_back("Cross-frame layout WAW on " + resourceLabel +
                " (frame " + std::to_string(layoutAnchor->frameIndex) + ")");
        }
    }
    size_t remoteReaders = 0;
    for (const ResourceFrontierAccess& reader : outstandingReaders) {
        if (reader.isActive() && !sameActualQueue(reader.queue, writerQueue)) {
            appendUniqueActiveCompletion(outWaits, reader.completion);
            ++remoteReaders;
        }
    }
    if (remoteReaders > 0 && outReasons != nullptr) {
        outReasons->push_back("Cross-frame WAR on " + resourceLabel +
            " (" + std::to_string(remoteReaders) + " active reader(s))");
    }
}

void CrossFrameResourceTracker::clear()
{
    frontiers_.clear();
}

void CrossFrameResourceTracker::pruneCompleted()
{
    for (auto& [id, frontier] : frontiers_) {
        (void)id;
        frontier.pruneCompleted();
    }
}

CrossFrameAccessPlanResult CrossFrameResourceTracker::planAccess(
    uint64_t backingAllocationId,
    std::string_view debugName,
    RenderGraphResourceType type,
    Queue* targetQueue,
    ResourceState nextState,
    bool writes,
    bool exclusiveQueueOwnership) const
{
    CrossFrameAccessPlanResult result;
    const auto found = frontiers_.find(backingAllocationId);
    if (found == frontiers_.end()) {
        return result;
    }
    const ResourceAccessFrontier& frontier = found->second;
    result.initialState = frontier.state;
    const bool isExclusive = exclusiveQueueOwnership || frontier.exclusiveQueueOwnership;
    if (isExclusive) {
        auto crossesQueue = [&](const std::optional<ResourceFrontierAccess>& access) {
            return access && access->queue != nullptr && !sameActualQueue(access->queue, targetQueue);
        };
        const bool readerCrossesQueue = std::any_of(frontier.outstandingReaders.begin(), frontier.outstandingReaders.end(),
            [&](const ResourceFrontierAccess& reader) {
                return reader.queue != nullptr && !sameActualQueue(reader.queue, targetQueue);
            });
        if (crossesQueue(frontier.lastWriter) || crossesQueue(frontier.layoutAnchor) || readerCrossesQueue) {
            result.requiresConservativeFallback = true;
            result.conservativeReason = "Exclusive queue-family ownership on " +
                std::string(debugName.empty() ? frontier.logicalId.name : std::string(debugName)) +
                " (#60 conservative boundary)";
        }
    }

    const bool texture = type == RenderGraphResourceType::Texture2D;
    const bool layoutChange = texture && frontier.state != ResourceState::Undefined && frontier.state != nextState;
    if (writes) {
        frontier.collectWaitsForWrite(targetQueue, result.waits, &result.waitReasons);
    } else {
        frontier.collectWaitsForRead(targetQueue, layoutChange, result.waits, &result.waitReasons);
    }

    // Per #58, never assume a prior frame's narrow read barrier established
    // writer visibility for every consumer stage in the new frame.
    if (frontier.state != ResourceState::Undefined &&
        (frontier.lastWriter.has_value() || frontier.layoutAnchor.has_value() || !frontier.outstandingReaders.empty())) {
        result.initialScope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite};
    }
    return result;
}

void CrossFrameResourceTracker::commitAcceptedAccess(const AcceptedCrossFrameAccess& access)
{
    if (access.backingAllocationId == 0) {
        return;
    }
    ResourceAccessFrontier& frontier = frontiers_[access.backingAllocationId];
    frontier.pruneCompleted();
    frontier.backingAllocationId = access.backingAllocationId;
    if (access.logicalId.valid()) {
        frontier.logicalId = access.logicalId;
    }
    if (access.version.valid()) {
        frontier.version = access.version;
    }
    frontier.type = access.type;
    frontier.exclusiveQueueOwnership = frontier.exclusiveQueueOwnership || access.exclusiveQueueOwnership;

    const FrontierEntryStatus status =
        (!access.completion.valid() || access.completion.isComplete())
            ? FrontierEntryStatus::Complete
            : FrontierEntryStatus::Accepted;
    ResourceFrontierAccess record{
        .frameIndex = access.frameIndex,
        .queue = access.queue,
        .scope = access.scope,
        .completion = access.completion,
        .status = status,
        .passName = access.passName,
    };
    if (access.queue != nullptr) {
        record.visibleQueues.push_back(access.queue);
    }

    if (access.layoutChanged && access.type == RenderGraphResourceType::Texture2D) {
        frontier.layoutAnchor = record;
    } else if (frontier.layoutAnchor && access.queue != nullptr) {
        frontier.layoutAnchor->markVisibleTo(access.queue);
    }
    if (access.writes) {
        frontier.lastWriter = record;
        frontier.outstandingReaders.clear();
    } else {
        if (frontier.lastWriter && access.queue != nullptr) {
            frontier.lastWriter->markVisibleTo(access.queue);
        }
        auto existing = std::find_if(frontier.outstandingReaders.begin(), frontier.outstandingReaders.end(),
            [&](const ResourceFrontierAccess& reader) {
                return sameActualQueue(reader.queue, record.queue) &&
                    reader.completion.sameSubmission(record.completion);
            });
        if (existing != frontier.outstandingReaders.end()) {
            existing->scope.stages = existing->scope.stages | record.scope.stages;
            existing->scope.access = existing->scope.access | record.scope.access;
            existing->status = record.status;
        } else {
            frontier.outstandingReaders.push_back(record);
        }
    }
    frontier.state = access.state;
}

void CrossFrameResourceTracker::updateBackingState(uint64_t backingAllocationId, ResourceState state)
{
    if (backingAllocationId == 0) {
        return;
    }
    if (auto found = frontiers_.find(backingAllocationId); found != frontiers_.end()) {
        found->second.state = state;
    }
}

void CrossFrameResourceTracker::forgetBacking(uint64_t backingAllocationId)
{
    frontiers_.erase(backingAllocationId);
}

bool CrossFrameResourceTracker::isBackingIdle(uint64_t backingAllocationId)
{
    const auto found = frontiers_.find(backingAllocationId);
    if (found == frontiers_.end()) {
        return true;
    }
    found->second.pruneCompleted();
    return found->second.isPhysicalBackingIdle();
}

bool CrossFrameResourceTracker::isBackingIdle(uint64_t backingAllocationId) const
{
    const auto found = frontiers_.find(backingAllocationId);
    if (found == frontiers_.end()) {
        return true;
    }
    return found->second.isPhysicalBackingIdle();
}

std::vector<GPUCompletionPoint> CrossFrameResourceTracker::activeCompletions(uint64_t backingAllocationId) const
{
    const auto found = frontiers_.find(backingAllocationId);
    if (found == frontiers_.end()) {
        return {};
    }
    return found->second.activeCompletions();
}

const ResourceAccessFrontier* CrossFrameResourceTracker::findFrontier(uint64_t backingAllocationId) const
{
    const auto found = frontiers_.find(backingAllocationId);
    return found == frontiers_.end() ? nullptr : &found->second;
}

void FrameExecutionResourceSnapshot::beginBuild(uint64_t frameIndex, uint64_t executionGeneration)
{
    frameIndex_ = frameIndex;
    executionGeneration_ = executionGeneration;
    frozen_ = false;
    bindings_.clear();
    bindingIndex_.clear();
    resourceSets_.clear();
    retainedOwners_.clear();
}

Result<> FrameExecutionResourceSnapshot::bindLogicalResource(std::string_view key, ResolvedLogicalBinding binding)
{
    if (frozen_ || key.empty()) {
        return makeError(Error::InvalidArgument);
    }
    if (binding.owner) {
        retainedOwners_.push_back(binding.owner);
    }
    const std::string keyString(key);
    const auto existing = bindingIndex_.find(keyString);
    if (existing != bindingIndex_.end()) {
        bindings_[existing->second] = std::move(binding);
        return {};
    }
    bindingIndex_.emplace(keyString, bindings_.size());
    bindings_.push_back(std::move(binding));
    return {};
}

Result<> FrameExecutionResourceSnapshot::bindResourceSetRaw(
    std::type_index type,
    std::shared_ptr<const void> data,
    std::shared_ptr<void> owner,
    LogicalResourceId setId,
    ResourceVersionId version)
{
    if (frozen_ || !data) {
        return makeError(Error::InvalidArgument);
    }
    if (owner) {
        retainedOwners_.push_back(owner);
    }
    resourceSets_[type] = ResourceSetEntry{
        .setId = std::move(setId),
        .version = std::move(version),
        .data = std::move(data),
        .owner = std::move(owner),
    };
    return {};
}

Result<> FrameExecutionResourceSnapshot::retainOwner(std::shared_ptr<void> owner)
{
    if (frozen_) {
        return makeError(Error::InvalidArgument);
    }
    if (owner) {
        retainedOwners_.push_back(std::move(owner));
    }
    return {};
}

void FrameExecutionResourceSnapshot::freeze()
{
    frozen_ = true;
}

const ResolvedLogicalBinding* FrameExecutionResourceSnapshot::findLogicalResource(std::string_view key) const
{
    const auto found = bindingIndex_.find(std::string(key));
    return found == bindingIndex_.end() ? nullptr : &bindings_[found->second];
}

const void* FrameExecutionResourceSnapshot::findResourceSetRaw(std::type_index type) const
{
    const auto found = resourceSets_.find(type);
    return found == resourceSets_.end() ? nullptr : found->second.data.get();
}

Result<RenderGraphResourceVersionPool::AcquiredTexture> RenderGraphResourceVersionPool::acquireTexture(
    Device& device,
    uint64_t logicalId,
    const TextureDesc& desc,
    const TextureViewDesc& viewDesc,
    uint64_t frameIndex,
    const CrossFrameResourceTracker& tracker,
    DeferredReleaseQueue& deferredReleases)
{
    if (logicalId == 0) {
        return makeError(Error::InvalidArgument);
    }
    TexturePoolEntry& entry = textures_[logicalId];
    if (!entry.backings.empty() &&
        (!textureDescEqual(entry.desc, desc) || !textureViewDescEqual(entry.viewDesc, viewDesc))) {
        for (auto& backing : entry.backings) {
            retireOwnedBacking(backing.backingAllocationId, backing.texture, tracker, deferredReleases);
            retireOwnedBacking(backing.backingAllocationId, backing.view, tracker, deferredReleases);
        }
        entry.backings.clear();
    }
    entry.desc = desc;
    entry.viewDesc = viewDesc;

    size_t selectedIndex = SIZE_MAX;
    uint64_t oldestIdleFrame = UINT64_MAX;
    for (size_t index = 0; index < entry.backings.size(); ++index) {
        const auto& backing = entry.backings[index];
        if (tracker.isBackingIdle(backing.backingAllocationId) && backing.lastUsedFrame < oldestIdleFrame) {
            oldestIdleFrame = backing.lastUsedFrame;
            selectedIndex = index;
        }
    }

    const uint32_t maxVersions = policy_.mode == RenderGraphMemoryPolicy::MaximizeOverlap
        ? std::max(1u, policy_.maxPhysicalVersionsPerResource)
        : 1u;

    if (selectedIndex == SIZE_MAX) {
        const bool needInitial = entry.backings.empty();
        bool canAllocateExtra = !needInitial &&
            policy_.mode == RenderGraphMemoryPolicy::MaximizeOverlap &&
            entry.backings.size() < maxVersions;
        if (canAllocateExtra) {
            uint64_t requiredBytes = entry.backings.front().sizeBytes;
            if (requiredBytes == 0) {
                if (auto queried = device.textureAllocationSize(desc)) {
                    requiredBytes = *queried;
                }
            }
            if (requiredBytes == 0 || currentExtraBytes() + requiredBytes > policy_.extraVersionBudgetBytes) {
                canAllocateExtra = false;
            }
        }
        if (needInitial || canAllocateExtra) {
            std::unique_ptr<Texture> texture;
            Result<> result = device.createTexture(desc).transform([&](auto value) { texture = std::move(value); });
            if (!result || !texture) {
                return makeError(result ? Error::Failure : result.error());
            }
            std::unique_ptr<TextureView> view;
            result = device.createTextureView(*texture, viewDesc).transform([&](auto value) { view = std::move(value); });
            if (!result || !view) {
                return makeError(result ? Error::Failure : result.error());
            }
            const auto mem = texture->memoryInfo();
            const uint64_t allocationId = mem.backingAllocationId != 0 ? mem.backingAllocationId : mem.allocationId;
            const uint64_t sizeBytes = mem.backingSizeBytes != 0 ? mem.backingSizeBytes : mem.sizeBytes;
            TextureBacking backing{
                .texture = std::shared_ptr<Texture>(std::move(texture)),
                .view = std::shared_ptr<TextureView>(std::move(view)),
                .backingAllocationId = allocationId,
                .allocationGeneration = nextAllocationGeneration_++,
                .sizeBytes = sizeBytes,
                .lastUsedFrame = frameIndex,
                .state = ResourceState::Undefined,
            };
            selectedIndex = entry.backings.size();
            entry.backings.push_back(std::move(backing));
            updatePeakManagedBytes();
        } else {
            uint64_t oldestFrame = UINT64_MAX;
            for (size_t index = 0; index < entry.backings.size(); ++index) {
                if (entry.backings[index].lastUsedFrame <= oldestFrame) {
                    oldestFrame = entry.backings[index].lastUsedFrame;
                    selectedIndex = index;
                }
            }
        }
    }

    auto& chosen = entry.backings[selectedIndex];
    chosen.lastUsedFrame = frameIndex;
    struct TextureOwnerPair {
        std::shared_ptr<Texture> texture;
        std::shared_ptr<TextureView> view;
    };
    auto owner = std::make_shared<TextureOwnerPair>(TextureOwnerPair{chosen.texture, chosen.view});
    return AcquiredTexture{
        .texture = chosen.texture.get(),
        .view = chosen.view.get(),
        .backingAllocationId = chosen.backingAllocationId,
        .allocationGeneration = chosen.allocationGeneration,
        .sizeBytes = chosen.sizeBytes,
        .physicalVersionCount = static_cast<uint32_t>(entry.backings.size()),
        .extraVersion = entry.backings.size() > 1,
        .state = chosen.state,
        .owner = std::move(owner),
    };
}

Result<RenderGraphResourceVersionPool::AcquiredBuffer> RenderGraphResourceVersionPool::acquireBuffer(
    Device& device,
    uint64_t logicalId,
    const BufferDesc& desc,
    const BufferViewDesc* viewDesc,
    uint64_t frameIndex,
    const CrossFrameResourceTracker& tracker,
    DeferredReleaseQueue& deferredReleases)
{
    if (logicalId == 0) {
        return makeError(Error::InvalidArgument);
    }
    const std::optional<BufferViewDesc> requestedView = viewDesc != nullptr
        ? std::optional<BufferViewDesc>(*viewDesc)
        : std::nullopt;
    BufferPoolEntry& entry = buffers_[logicalId];
    if (!entry.backings.empty() &&
        (!bufferDescEqual(entry.desc, desc) || !bufferViewDescEqual(entry.viewDesc, requestedView))) {
        for (auto& backing : entry.backings) {
            retireOwnedBacking(backing.backingAllocationId, backing.buffer, tracker, deferredReleases);
            retireOwnedBacking(backing.backingAllocationId, backing.view, tracker, deferredReleases);
        }
        entry.backings.clear();
    }
    entry.desc = desc;
    entry.viewDesc = requestedView;

    size_t selectedIndex = SIZE_MAX;
    uint64_t oldestIdleFrame = UINT64_MAX;
    for (size_t index = 0; index < entry.backings.size(); ++index) {
        const auto& backing = entry.backings[index];
        if (tracker.isBackingIdle(backing.backingAllocationId) && backing.lastUsedFrame < oldestIdleFrame) {
            oldestIdleFrame = backing.lastUsedFrame;
            selectedIndex = index;
        }
    }

    const uint32_t maxVersions = policy_.mode == RenderGraphMemoryPolicy::MaximizeOverlap
        ? std::max(1u, policy_.maxPhysicalVersionsPerResource)
        : 1u;

    if (selectedIndex == SIZE_MAX) {
        const bool needInitial = entry.backings.empty();
        bool canAllocateExtra = !needInitial &&
            policy_.mode == RenderGraphMemoryPolicy::MaximizeOverlap &&
            entry.backings.size() < maxVersions;
        if (canAllocateExtra) {
            const uint64_t requiredBytes = entry.backings.front().sizeBytes != 0
                ? entry.backings.front().sizeBytes
                : desc.size;
            if (requiredBytes == 0 || currentExtraBytes() + requiredBytes > policy_.extraVersionBudgetBytes) {
                canAllocateExtra = false;
            }
        }
        if (needInitial || canAllocateExtra) {
            std::unique_ptr<Buffer> buffer;
            Result<> result = device.createBuffer(desc).transform([&](auto value) { buffer = std::move(value); });
            if (!result || !buffer) {
                return makeError(result ? Error::Failure : result.error());
            }
            std::unique_ptr<BufferView> view;
            if (requestedView.has_value()) {
                result = device.createBufferView(*buffer, *requestedView).transform([&](auto value) { view = std::move(value); });
                if (!result || !view) {
                    return makeError(result ? Error::Failure : result.error());
                }
            }
            const auto mem = buffer->memoryInfo();
            const uint64_t allocationId = mem.backingAllocationId != 0 ? mem.backingAllocationId : mem.allocationId;
            const uint64_t sizeBytes = mem.backingSizeBytes != 0
                ? mem.backingSizeBytes
                : (mem.sizeBytes != 0 ? mem.sizeBytes : desc.size);
            BufferBacking backing{
                .buffer = std::shared_ptr<Buffer>(std::move(buffer)),
                .view = std::shared_ptr<BufferView>(std::move(view)),
                .backingAllocationId = allocationId,
                .allocationGeneration = nextAllocationGeneration_++,
                .sizeBytes = sizeBytes,
                .lastUsedFrame = frameIndex,
                .state = ResourceState::Undefined,
            };
            selectedIndex = entry.backings.size();
            entry.backings.push_back(std::move(backing));
            updatePeakManagedBytes();
        } else {
            uint64_t oldestFrame = UINT64_MAX;
            for (size_t index = 0; index < entry.backings.size(); ++index) {
                if (entry.backings[index].lastUsedFrame <= oldestFrame) {
                    oldestFrame = entry.backings[index].lastUsedFrame;
                    selectedIndex = index;
                }
            }
        }
    }

    auto& chosen = entry.backings[selectedIndex];
    chosen.lastUsedFrame = frameIndex;
    struct BufferOwnerPair {
        std::shared_ptr<Buffer> buffer;
        std::shared_ptr<BufferView> view;
    };
    auto owner = std::make_shared<BufferOwnerPair>(BufferOwnerPair{chosen.buffer, chosen.view});
    return AcquiredBuffer{
        .buffer = chosen.buffer.get(),
        .view = chosen.view.get(),
        .backingAllocationId = chosen.backingAllocationId,
        .allocationGeneration = chosen.allocationGeneration,
        .sizeBytes = chosen.sizeBytes,
        .physicalVersionCount = static_cast<uint32_t>(entry.backings.size()),
        .extraVersion = entry.backings.size() > 1,
        .state = chosen.state,
        .owner = std::move(owner),
    };
}

void RenderGraphResourceVersionPool::updateBackingState(uint64_t backingAllocationId, ResourceState state)
{
    for (auto& [id, entry] : textures_) {
        (void)id;
        for (auto& backing : entry.backings) {
            if (backing.backingAllocationId == backingAllocationId) {
                backing.state = state;
                return;
            }
        }
    }
    for (auto& [id, entry] : buffers_) {
        (void)id;
        for (auto& backing : entry.backings) {
            if (backing.backingAllocationId == backingAllocationId) {
                backing.state = state;
                return;
            }
        }
    }
}

void RenderGraphResourceVersionPool::retireAll(
    const CrossFrameResourceTracker& tracker,
    DeferredReleaseQueue& deferredReleases)
{
    for (auto& [id, entry] : textures_) {
        (void)id;
        for (auto& backing : entry.backings) {
            retireOwnedBacking(backing.backingAllocationId, backing.texture, tracker, deferredReleases);
            retireOwnedBacking(backing.backingAllocationId, backing.view, tracker, deferredReleases);
        }
    }
    for (auto& [id, entry] : buffers_) {
        (void)id;
        for (auto& backing : entry.backings) {
            retireOwnedBacking(backing.backingAllocationId, backing.buffer, tracker, deferredReleases);
            retireOwnedBacking(backing.backingAllocationId, backing.view, tracker, deferredReleases);
        }
    }
    textures_.clear();
    buffers_.clear();
}

void RenderGraphResourceVersionPool::clear()
{
    textures_.clear();
    buffers_.clear();
    peakManagedBytes_ = 0;
}

uint64_t RenderGraphResourceVersionPool::currentExtraBytes() const
{
    uint64_t bytes = 0;
    for (const auto& [id, entry] : textures_) {
        (void)id;
        for (size_t i = 1; i < entry.backings.size(); ++i) {
            bytes += entry.backings[i].sizeBytes;
        }
    }
    for (const auto& [id, entry] : buffers_) {
        (void)id;
        for (size_t i = 1; i < entry.backings.size(); ++i) {
            bytes += entry.backings[i].sizeBytes;
        }
    }
    return bytes;
}

uint64_t RenderGraphResourceVersionPool::currentTotalBytes() const
{
    uint64_t bytes = 0;
    for (const auto& [id, entry] : textures_) {
        (void)id;
        for (const auto& backing : entry.backings) {
            bytes += backing.sizeBytes;
        }
    }
    for (const auto& [id, entry] : buffers_) {
        (void)id;
        for (const auto& backing : entry.backings) {
            bytes += backing.sizeBytes;
        }
    }
    return bytes;
}

void RenderGraphResourceVersionPool::updatePeakManagedBytes()
{
    peakManagedBytes_ = std::max(peakManagedBytes_, currentTotalBytes());
}

RenderGraphResourceVersionPool::Stats RenderGraphResourceVersionPool::stats() const
{
    Stats out;
    out.pooledLogicalResources = static_cast<uint32_t>(textures_.size() + buffers_.size());
    for (const auto& [id, entry] : textures_) {
        (void)id;
        out.totalPhysicalVersions += static_cast<uint32_t>(entry.backings.size());
        if (entry.backings.size() > 1) {
            out.extraVersionCount += static_cast<uint32_t>(entry.backings.size() - 1);
            for (size_t i = 1; i < entry.backings.size(); ++i) {
                out.extraVersionBytes += entry.backings[i].sizeBytes;
            }
        }
        for (const auto& backing : entry.backings) {
            out.totalManagedBytes += backing.sizeBytes;
        }
    }
    for (const auto& [id, entry] : buffers_) {
        (void)id;
        out.totalPhysicalVersions += static_cast<uint32_t>(entry.backings.size());
        if (entry.backings.size() > 1) {
            out.extraVersionCount += static_cast<uint32_t>(entry.backings.size() - 1);
            for (size_t i = 1; i < entry.backings.size(); ++i) {
                out.extraVersionBytes += entry.backings[i].sizeBytes;
            }
        }
        for (const auto& backing : entry.backings) {
            out.totalManagedBytes += backing.sizeBytes;
        }
    }
    out.peakManagedBytes = std::max(peakManagedBytes_, out.totalManagedBytes);
    return out;
}

} // namespace metallic::render
