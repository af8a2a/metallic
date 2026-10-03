#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Streamer/MeshletStreamCompactCLASPool.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace metallic::render {
namespace {
uint64_t compactAlign(uint64_t bytes, uint64_t alignment)
{
    return (bytes + alignment - 1) / alignment * alignment;
}
constexpr auto kCompactBufferUsage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                                     BufferUsageBits::AccelerationStructureStorage |
                                     BufferUsageBits::AccelerationStructureBuildInput | BufferUsageBits::TransferSource |
                                     BufferUsageBits::TransferDestination;
Result<> publicationBarrier(CommandBuffer& cmd, Buffer& buffer, ResourceState before, ResourceState after)
{
    const BufferBarrierDesc barrier{
        .buffer = &buffer,
        .before = resourceSyncScope(before, PipelineStageBits::AllCommands),
        .after = resourceSyncScope(after, PipelineStageBits::AllCommands),
    };
    if (auto commandResult = cmd.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    return {};
}
} // namespace

struct MeshletStreamCompactCLASPool::Impl {
    enum class State { Pending, Active, Retiring };
    enum class Phase { Free, Building, Sized, Moving };
    enum class Lifetime { Unassigned, Persistent, Transient };
    struct FrameUse {
        GPUCompletionPoint completion;
        std::vector<std::shared_ptr<Buffer>> buffers;
        bool completed = false;
    };
    using FrameUses = std::vector<std::shared_ptr<FrameUse>>;
    struct Chunk {
        std::shared_ptr<Buffer> buffer;
        MeshletStreamStorage storage;
        FrameUses uses;
        uint64_t emptySinceFrame = UINT64_MAX;
        Lifetime lifetime = Lifetime::Unassigned;
    };
    struct Page {
        State state = State::Pending;
        bool wanted = true;
        uint64_t retireFrame = 0, encodedBytes = 0;
        Chunk* chunk = nullptr;
        MeshletStreamStorageAllocation allocation, addresses;
        FrameUses retirementUses;
        std::vector<uint32_t> offsets;
    };
    struct Item {
        uint32_t page = 0, first = 0;
        bool moving = false;
        std::vector<uint32_t> sizes;
    };
    struct Batch {
        Phase phase = Phase::Free;
        std::unique_ptr<MeshletStreamCLASPool> builder;
        std::unique_ptr<Buffer> sizes, moveSources, moveDestinations;
        GPUCompletionPoint completion;
        std::shared_ptr<SubmissionTransaction> submission;
        std::vector<Item> items;
        std::vector<uint32_t> stagedPages;
    };
    Device* device = nullptr;
    const scene::MeshletStreamAsset* asset = nullptr;
    MeshletStreamStorage addressStorage;
    std::vector<std::unique_ptr<Chunk>> chunks;
    std::vector<bool> persistentPages;
    FrameUses frameUses;
    std::unique_ptr<Buffer> scratch, addresses, pageTable;
    uint64_t chunkBytes = 0, persistentChunkBytes = 0, retryGrowthFrame = 0;
    uint32_t emptyChunkRetentionFrames = 0;
    std::unordered_map<uint32_t, Page> pages;
    struct Retirement {
        uint64_t deadline;
        uint32_t pageIndex;
    };
    std::deque<Retirement> retirements, waitingRetirements;
    std::vector<Batch> batches;
    MeshletStreamCLASPoolStats stats;
    uint64_t frame = 0, alignment = 0, stride = 0, scratchOffset = 0;
    uint32_t maxBuild = 0, queuedFrames = 0;
    bool initialized = false;
    std::string error;
    struct Publication {
        uint64_t revision = 0;
        MeshletStreamCLASPageEntry entry;
        uint32_t addressOffset = 0;
        std::vector<uint64_t> addresses;
    };
    using Publications = std::unordered_map<uint32_t, Publication>;
    // Shared with cancellation callbacks; never capture a pool's lifetime.
    std::shared_ptr<Publications> publications = std::make_shared<Publications>();

    Result<> buffer(uint64_t bytes, MemoryLocation location, std::unique_ptr<Buffer>& output,
        MemoryBudgetDomain domain = MemoryBudgetDomain::CLASScratch)
    {
        return device->createBuffer({.size = std::max(bytes, uint64_t(8)), .usage = kCompactBufferUsage, .memoryLocation = location,
                .memoryDomain = domain}).transform([&](auto rhiValue) { output = std::move(rhiValue); });
    }
    static bool completed(const std::shared_ptr<FrameUse>& use)
    {
        return use->completed;
    }
    void collectFrameUses()
    {
        // Query each submission once, not once per chunk or retired page.
        for (auto& use : frameUses) {
            const auto& point = use->completion;
            use->completed = point.isCancelled() || (point.isSubmitted() && point.isComplete());
            // Command wrappers may survive completion until their frame slot is
            // reused. Release backing here, so an empty chunk really frees its
            // allocation before another chunk consumes the pool budget.
            if (use->completed) { use->buffers.clear(); }
        }
        std::erase_if(frameUses, completed);
    }
    Result<> retain(CommandBuffer& cmd, Chunk& chunk)
    {
        const auto point = metallic::render::RenderFrameContext::from(cmd)->completion();
        if (!chunk.uses.empty() && chunk.uses.back()->completion.sameSubmission(point)) { return {}; }
        if (frameUses.empty() || !frameUses.back()->completion.sameSubmission(point)) {
            frameUses.push_back(std::make_shared<FrameUse>(FrameUse{.completion = point}));
        }
        auto& use = frameUses.back();
        auto result = cmd.retainResource(use);
        if (!result) { return result; }
        use->buffers.push_back(chunk.buffer);
        chunk.uses.push_back(use);
        return {};
    }
    void collectChunks(bool pressure = false)
    {
        for (auto& chunk : chunks) { std::erase_if(chunk->uses, completed); }
        std::erase_if(chunks, [&](const auto& chunk) {
            if (chunk->storage.usedBytes() || !chunk->uses.empty()) {
                chunk->emptySinceFrame = UINT64_MAX;
                return false;
            }
            if (chunk->emptySinceFrame == UINT64_MAX) { chunk->emptySinceFrame = frame; }
            if (!pressure && frame - chunk->emptySinceFrame < emptyChunkRetentionFrames) { return false; }
            const auto bytes = chunk->buffer->desc().size;
            stats.storageBytes -= bytes;
            stats.totalStorageReleasedBytes += bytes;
            ++stats.frameStorageReleaseCount;
            retryGrowthFrame = 0;
            return true;
        });
        stats.storageChunkCount = uint32_t(chunks.size());
    }
    Result<> grow(uint64_t capacity)
    {
        auto chunk = std::make_unique<Chunk>();
        std::string reason;
        if (!chunk->storage.initialize(capacity, alignment, reason, UINT64_MAX)) {
            error = reason;
            return makeError(Error::InvalidArgument);
        }
        std::unique_ptr<Buffer> backing;
        auto result = buffer(capacity, MemoryLocation::Device, backing, MemoryBudgetDomain::CLAS);
        if (!result) { return result; }
        chunk->buffer = std::shared_ptr<Buffer>(std::move(backing));
        chunk->emptySinceFrame = frame;
        stats.storageBytes += capacity;
        ++stats.totalStorageGrowthCount;
        ++stats.frameStorageGrowthCount;
        chunks.push_back(std::move(chunk));
        stats.storageChunkCount = uint32_t(chunks.size());
        return {};
    }
    bool allocate(uint32_t pageIndex, Page& page, uint64_t bytes)
    {
        const auto lifetime = persistentPages[pageIndex] ? Lifetime::Persistent : Lifetime::Transient;
        // Pack into the fullest compatible block, then consume its smallest
        // suitable hole. Root pages never pin blocks containing evictable pages.
        Chunk* selected = nullptr;
        uint64_t leastFree = UINT64_MAX;
        for (auto& chunk : chunks) {
            if (chunk->storage.usedBytes() && chunk->lifetime != lifetime) { continue; }
            if (chunk->storage.freeBytes() >= leastFree || !chunk->storage.canAllocate(bytes)) { continue; }
            selected = chunk.get();
            leastFree = chunk->storage.freeBytes();
        }
        if (selected) {
            page.allocation = selected->storage.allocate(bytes, true);
            page.chunk = selected;
            selected->lifetime = lifetime;
            selected->emptySinceFrame = UINT64_MAX;
            return true;
        }
        bytes = compactAlign(bytes, alignment);
        uint64_t remaining = stats.storageBudgetBytes - stats.storageBytes;
        if (bytes > remaining) {
            // A retained empty block must not prevent a larger page from fitting
            // under max. This still requires every GPU reader to be complete.
            collectChunks(true);
            remaining = stats.storageBudgetBytes - stats.storageBytes;
        }
        if (bytes > remaining || frame < retryGrowthFrame) { return false; }
        const auto quantum = lifetime == Lifetime::Persistent ? persistentChunkBytes : chunkBytes;
        const uint64_t capacity = std::min(std::max(quantum, bytes), remaining);
        auto result = grow(capacity);
        if (!result) {
            if (hasError(result, Error::OutOfMemory)) { retryGrowthFrame = frame + 30; }
            else { error = "Compact CLAS backing allocation failed: " + std::string(resultToString(result)); }
            return false;
        }
        auto& chunk = chunks.back();
        page.allocation = chunk->storage.allocate(bytes);
        page.chunk = chunk.get();
        chunk->lifetime = lifetime;
        chunk->emptySinceFrame = UINT64_MAX;
        return true;
    }
    void publish(uint32_t id, const Page* page, bool orderedMove = false)
    {
        Publication update;
        update.revision = ++stats.publicationRevision;
        update.entry.publicationGeneration = uint32_t(update.revision);
        if (page && (page->state != State::Pending || orderedMove)) {
            update.entry.addressOffsetAndState = packMeshletStreamClasPageEntry(
                uint32_t(page->addresses.offset), page->state == State::Active ? MeshletStreamCLASPageState::Active
                    : orderedMove ? MeshletStreamCLASPageState::Active : MeshletStreamCLASPageState::Retiring);
            update.addressOffset = uint32_t(page->addresses.offset);
            for (uint32_t offset : page->offsets) {
                update.addresses.push_back(page->chunk->buffer->deviceAddress() + page->allocation.offset + offset);
            }
        }
        (*publications)[id] = std::move(update);
    }
    Result<> flushPublications(CommandBuffer& cmd, std::string& log)
    {
        if (publications->empty()) { return {}; }
        auto updates = std::make_shared<Publications>(*publications);
        uint64_t bytes = 0;
        for (const auto& [id, update] : *updates) { bytes += update.addresses.size() * 8u + 8u; }
        std::unique_ptr<Buffer> staging;
        // This is only a copy source, not AS storage or shader input. Inheriting
        // kCompactBufferUsage places it in device-local BAR memory and makes
        // even retirement/publication fail when capture consumes VRAM headroom.
        auto result = device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload, .memoryDomain = MemoryBudgetDomain::Upload})
            .transform([&](auto rhiValue) { staging = std::move(rhiValue); });
        if (!result) {
            log = "Compact CLAS publication upload allocation failed: " + std::string(resultToString(result));
            return result;
        }
        auto upload = std::shared_ptr<Buffer>(std::move(staging));
        auto* mapped = static_cast<uint8_t*>(upload->map());
        if (!mapped) { return makeError(Error::Failure); }
        uint64_t offset = 0;
        for (const auto& [id, update] : *updates) {
            const uint64_t size = update.addresses.size() * 8u;
            if (size) { std::memcpy(mapped + offset, update.addresses.data(), size); }
            std::memcpy(mapped + offset + size, &update.entry, sizeof(update.entry));
            offset += size + 8u;
        }
        upload->flush();
        upload->unmap();
        result = cmd.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(nullptr,
            [pending = publications, updates] {
                for (const auto& [id, update] : *updates) {
                    auto it = pending->find(id);
                    if (it == pending->end() || it->second.revision < update.revision) { (*pending)[id] = update; }
                }
            }));
        if (!result) { return result; }
        result = cmd.retainResource(upload);
        if (!result) { return result; }
        for (Buffer* target : {addresses.get(), pageTable.get()}) {
            if (auto commandResult = publicationBarrier(cmd, *target, ResourceState::General, ResourceState::TransferDestination); !commandResult) { return commandResult; }
        }
        if (auto commandResult = publicationBarrier(cmd, *upload, ResourceState::Undefined, ResourceState::TransferSource); !commandResult) { return commandResult; }
        offset = 0;
        for (const auto& [id, update] : *updates) {
            const uint64_t size = update.addresses.size() * 8u;
            if (size) {
                {
                    auto sourceSlice = upload.get()->slice({offset, size});
                    if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                    auto destinationSlice = addresses.get()->slice({uint64_t(update.addressOffset) * 8u, size});
                    if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                    if (auto commandResult = cmd.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
                }
            }
            {
                auto sourceSlice = upload.get()->slice({offset + size, sizeof(MeshletStreamCLASPageEntry)});
                if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                auto destinationSlice = pageTable.get()->slice({uint64_t(id) * sizeof(MeshletStreamCLASPageEntry), sizeof(MeshletStreamCLASPageEntry)});
                if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                if (auto commandResult = cmd.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
            }
            offset += size + 8u;
        }
        for (Buffer* target : {addresses.get(), pageTable.get()}) {
            if (auto commandResult = publicationBarrier(cmd, *target, ResourceState::TransferDestination, ResourceState::General); !commandResult) { return commandResult; }
        }
        publications->clear();
        return {};
    }
    void release(Page& page)
    {
        if (!page.allocation.valid()) {
            return;
        }
        stats.encodedStorageBytes -= page.encodedBytes;
        stats.worstCaseStorageBytes -= uint64_t(page.offsets.size()) * stride;
        page.chunk->storage.release(page.allocation);
        page.chunk = nullptr;
        addressStorage.release(page.addresses);
        page.allocation = {};
        page.addresses = {};
        page.encodedBytes = 0;
        page.offsets.clear();
    }
    void discard(uint32_t id)
    {
        auto it = pages.find(id);
        if (it == pages.end()) {
            return;
        }
        release(it->second);
        publish(id, nullptr);
        pages.erase(it);
    }
    void reset(Batch& batch)
    {
        // The batch's final move (or cancellation before submission) is complete.
        batch.builder->retirePages(batch.stagedPages);
        batch.builder->beginFrame();
        batch.items.clear();
        batch.stagedPages.clear();
        batch.completion = {};
        batch.submission.reset();
        batch.phase = Phase::Free;
    }
    Result<> track(CommandBuffer& cmd, Batch& batch)
    {
        batch.completion = metallic::render::RenderFrameContext::from(cmd)->completion();
        batch.submission = std::make_shared<SubmissionTransaction>(nullptr, nullptr);
        return cmd.addSubmissionTransaction(batch.submission);
    }
    void collect()
    {
        for (auto& batch : batches) {
            if (batch.phase != Phase::Building && batch.phase != Phase::Moving) {
                continue;
            }
            const bool cancelled = batch.submission->cancelled() || batch.completion.isCancelled();
            if (!cancelled &&
                (!batch.submission->resolved() || !batch.completion.isSubmitted() || !batch.completion.isComplete())) {
                continue;
            }
            if (batch.phase == Phase::Building) {
                if (cancelled) {
                    for (const auto& item : batch.items) {
                        discard(item.page);
                    }
                    reset(batch);
                    continue;
                }
                batch.sizes->invalidate();
                const auto* sizes = static_cast<const uint32_t*>(batch.sizes->map());
                if (!sizes) {
                    error = "Compact CLAS size readback map failed";
                    return;
                }
                for (auto& item : batch.items) {
                    const uint32_t count = asset->pages()[item.page].clusterCount;
                    item.sizes.assign(sizes + item.first, sizes + item.first + count);
                    for (uint32_t size : item.sizes) {
                        if (!size || size > stride) {
                            error = "GPU returned invalid CLAS encoded size";
                        }
                    }
                }
                batch.sizes->unmap();
            } else {
                for (auto& item : batch.items) {
                    if (!item.moving) {
                        continue;
                    }
                    auto& page = pages.at(item.page);
                    if (cancelled) {
                        release(page);
                        publish(item.page, nullptr);
                        item.moving = false;
                        continue;
                    }
                    if (!page.wanted) {
                        discard(item.page);
                        continue;
                    }
                    // GPU publication was ordered directly after MOVE. Only CPU
                    // ownership/statistics wait for the completion here.
                    page.state = State::Active;
                    ++stats.builtPageCount;
                    stats.builtClusterCount += uint32_t(page.offsets.size());
                }
                std::erase_if(batch.items, [](const auto& item) { return item.moving; });
            }
            if (batch.items.empty()) {
                reset(batch);
            } else {
                batch.phase = Phase::Sized;
                batch.submission.reset();
                batch.completion = {};
            }
        }
    }
};

MeshletStreamCompactCLASPool::MeshletStreamCompactCLASPool() : impl_(std::make_unique<Impl>())
{
}
MeshletStreamCompactCLASPool::~MeshletStreamCompactCLASPool() = default;

Result<> MeshletStreamCompactCLASPool::initialize(Device& device, const MeshletStreamCLASPoolDesc& desc, std::string& log)
{
    log.clear();
    auto& p = *impl_;
    p.device = &device;
    p.asset = desc.asset;
    if (!desc.asset || !desc.asset->valid() || !desc.maxStorageBytes || !desc.maxBuildClusters ||
        !desc.queuedFrameCount || !desc.growStorageBytes || desc.startStorageBytes > desc.maxStorageBytes) {
        log = "Compact CLAS requires a valid asset, nonzero max/grow/build/queue limits, and start <= max";
        return makeError(Error::InvalidArgument);
    }
    p.persistentPages.assign(desc.asset->pageCount(), false);
    for (uint32_t id : desc.persistentPages) {
        if (id >= p.persistentPages.size()) {
            log = "Compact CLAS persistent page index is out of range";
            return makeError(Error::InvalidArgument);
        }
        p.persistentPages[id] = true;
    }
    ClusterAccelerationStructureProperties properties;
    auto result = device.queryClusterAccelerationStructureProperties().transform([&](auto rhiValue) { properties = std::move(rhiValue); });
    if (!result) {
        return result;
    }
    if (!properties.clusterStorageAlignment || !properties.scratchAlignment) {
        return makeError(Error::Unsupported);
    }
    p.alignment = properties.clusterStorageAlignment;
    p.maxBuild = desc.maxBuildClusters;
    p.queuedFrames = desc.queuedFrameCount;
    ClusterAccelerationStructureBuildSizes single, move;
    if (!(result = device.queryClusterAccelerationStructureTriangleBuildSizes({.maxClusterTriangleCount = p.asset->maxClusterTriangles(),
               .maxClusterVertexCount = p.asset->maxClusterVertices(),
               .maxTotalTriangleCount = p.asset->maxClusterTriangles(),
               .maxTotalVertexCount = p.asset->maxClusterVertices()}).transform([&](auto rhiValue) { single = std::move(rhiValue); }))) {
        return result;
    }
    p.stride = compactAlign(single.accelerationStructureSize, p.alignment);
    if (!(result = device.queryClusterAccelerationStructureMoveSizes(p.maxBuild, p.stride * p.maxBuild).transform([&](auto rhiValue) { move = std::move(rhiValue); }))) {
        return result;
    }
    const uint64_t capacity = desc.maxStorageBytes / p.alignment * p.alignment;
    const uint64_t slots =
        std::min(capacity / p.alignment, uint64_t(p.asset->pageCount()) * p.asset->maxPageClusters());
    if (!capacity || slots > kMeshletStreamCLASPageAddressOffsetMask ||
        !p.addressStorage.initialize(slots, 1, log, UINT64_MAX)) {
        return makeError(Error::InvalidArgument);
    }
    // MOVE_OBJECTS uses updateScratchSize; buildScratchSize may be zero.
    // Keep the alignment padding inside the allocation as for triangle builds.
    if (!(result = p.buffer(move.updateScratchSize + properties.scratchAlignment, MemoryLocation::Device, p.scratch)) ||
        !(result = p.buffer(slots * 8, MemoryLocation::HostUpload, p.addresses)) ||
        !(result = p.buffer(uint64_t(p.asset->pageCount()) * sizeof(MeshletStreamCLASPageEntry), MemoryLocation::HostUpload, p.pageTable))) {
        return result;
    }
    for (Buffer* buffer : {p.addresses.get(), p.pageTable.get()}) {
        void* mapped = buffer->map();
        if (!mapped) {
            return makeError(Error::Failure);
        }
        std::memset(mapped, 0, buffer->desc().size);
        buffer->flush();
        buffer->unmap();
    }
    p.scratchOffset =
        compactAlign(p.scratch->deviceAddress(), properties.scratchAlignment) - p.scratch->deviceAddress();
    p.batches.resize(desc.queuedFrameCount);
    p.stats.scratchBytes = p.scratch->desc().size;
    for (auto& batch : p.batches) {
        batch.builder = std::make_unique<MeshletStreamCLASPool>();
        if (!(result = batch.builder->initialize(device,
                                                 {.asset = desc.asset,
                                                  .maxStorageBytes = p.stride * p.maxBuild,
                                                  .maxBuildClusters = p.maxBuild,
                                                  .queuedFrameCount = 1,
                                                  .compactStorage = false},
                                                 log)) ||
            !(result = p.buffer(uint64_t(p.maxBuild) * 4, MemoryLocation::HostReadback, batch.sizes)) ||
            !(result = p.buffer(uint64_t(p.maxBuild) * 8, MemoryLocation::HostUpload, batch.moveSources)) ||
            !(result = p.buffer(uint64_t(p.maxBuild) * 8, MemoryLocation::HostUpload, batch.moveDestinations))) {
            return result;
        }
        p.stats.scratchBytes += batch.builder->stats().storageBytes + batch.builder->stats().scratchBytes +
                                batch.builder->clusterAddressBuffer()->desc().size +
                                batch.builder->pageTableBuffer()->desc().size + batch.sizes->desc().size +
                                batch.moveSources->desc().size + batch.moveDestinations->desc().size +
                                uint64_t(p.maxBuild) * (properties.triangleBuildInfoSize + sizeof(uint64_t));
    }
    p.stats.pageCapacity = p.asset->pageCount();
    p.stats.clusterSlotCapacity = uint32_t(slots);
    p.stats.clusterStrideBytes = p.stride;
    p.stats.storageBudgetBytes = capacity;
    p.chunkBytes = std::min(capacity, compactAlign(std::min(desc.growStorageBytes, capacity), p.alignment));
    const auto persistentCount = std::count(p.persistentPages.begin(), p.persistentPages.end(), true);
    if (persistentCount && persistentCount < p.persistentPages.size()) {
        // An oversized growth quantum must not reserve the entire small pool
        // for the first lifetime. This caps block size, not either class's quota.
        const uint64_t sharedQuantum = capacity / 2 / p.alignment * p.alignment;
        if (sharedQuantum) { p.chunkBytes = std::min(p.chunkBytes, sharedQuantum); }
    }
    p.persistentChunkBytes = desc.persistentGrowStorageBytes
        ? std::min(p.chunkBytes, compactAlign(std::min(desc.persistentGrowStorageBytes, capacity), p.alignment))
        : p.chunkBytes;
    p.emptyChunkRetentionFrames = desc.emptyChunkRetentionFrames;
    p.stats.growStorageBytes = p.chunkBytes;
    p.stats.persistentGrowStorageBytes = p.persistentChunkBytes;
    p.stats.startStorageBytes = std::min(capacity, compactAlign(std::min(desc.startStorageBytes, capacity), p.alignment));
    while (p.stats.storageBytes < p.stats.startStorageBytes) {
        if (!(result = p.grow(std::min(p.chunkBytes, p.stats.startStorageBytes - p.stats.storageBytes)))) {
            log = "Compact CLAS initial backing allocation failed: " + std::string(resultToString(result));
            // Failed initialization never holds a partial initial commitment.
            p.chunks.clear();
            p.stats.storageBytes = 0;
            p.stats.storageChunkCount = 0;
            return result;
        }
    }
    p.initialized = true;
    return {};
}

void MeshletStreamCompactCLASPool::beginFrame(CPUProfileRecorder* profiler)
{
    if (!ready()) {
        return;
    }
    auto& p = *impl_;
    ++p.frame;
    p.stats.frameBuiltPageCount = p.stats.frameBuiltClusterCount = p.stats.frameRejectedPageCount = 0;
    p.stats.frameMovedPageCount = p.stats.frameMovedClusterCount = 0;
    p.stats.frameStorageGrowthCount = p.stats.frameStorageReleaseCount = 0;
    CPUProfileScope profile(profiler, "Collect completed CLAS");
    p.collectFrameUses();
    p.collect();
    profile.next("Expire retired CLAS");
    // The queued-frame delay is fixed, so retirement deadlines arrive in order.
    while (!p.retirements.empty() && p.retirements.front().deadline <= p.frame) {
        p.waitingRetirements.push_back(p.retirements.front());
        p.retirements.pop_front();
    }
    const size_t retirementCount = p.waitingRetirements.size();
    for (size_t i = 0; i < retirementCount; ++i) {
        const auto retirement = p.waitingRetirements.front();
        p.waitingRetirements.pop_front();
        const auto it = p.pages.find(retirement.pageIndex);
        if (it == p.pages.end()) { continue; }
        auto& page = it->second;
        // A revived page may have been retired again with a later deadline.
        if (page.state == Impl::State::Retiring && page.retireFrame == retirement.deadline) {
            if (!std::all_of(page.retirementUses.begin(), page.retirementUses.end(), Impl::completed)) {
                p.waitingRetirements.push_back(retirement);
                continue;
            }
            --p.stats.retiringPageCount;
            p.stats.retiringClusterCount -= uint32_t(page.offsets.size());
            p.stats.retiringStorageBytes -= page.allocation.allocatedSize;
            p.release(page);
            p.publish(it->first, nullptr);
            p.pages.erase(it);
        }
    }
    p.collectChunks();
}

Result<> MeshletStreamCompactCLASPool::cmdBuildPages(CommandBuffer& cmd, Buffer& geometry,
                                                   std::span<const MeshletStreamCLASPageBuild> requests,
                                                   std::string& log)
{
    auto& p = *impl_;
    log = p.error;
    if (!ready() || !log.empty()) {
        return makeError(Error::Failure);
    }
    if (!metallic::render::RenderFrameContext::from(cmd) || !metallic::render::RenderFrameContext::from(cmd)->recording()) {
        log = "Compact CLAS requires tracked frame completion";
        return makeError(Error::InvalidArgument);
    }
    // These addresses can be consumed later in this frame by BLAS/TLAS work.
    // Retain physical backing even if the CPU subsequently retires every page.
    for (auto& chunk : p.chunks) {
        if (!chunk->storage.usedBytes()) { continue; }
        if (auto result = p.retain(cmd, *chunk); !result) { return result; }
    }
    for (const auto& request : requests) {
        if (request.pageIndex >= p.asset->pageCount()) {
            return makeError(Error::InvalidArgument);
        }
        auto found = p.pages.find(request.pageIndex);
        if (found == p.pages.end()) {
            if (!request.plan || request.plan->pageIndex != request.pageIndex || request.plan->clusters.empty() ||
                request.deviceOffsetBytes > geometry.desc().size ||
                request.plan->payloadByteSize > geometry.desc().size - request.deviceOffsetBytes) {
                log = "Invalid compact CLAS page request";
                return makeError(Error::InvalidArgument);
            }
            continue;
        }
        auto& page = found->second;
        page.wanted = true;
        if (page.state == Impl::State::Retiring) {
            page.state = Impl::State::Active;
            page.retirementUses.clear();
            --p.stats.retiringPageCount;
            p.stats.retiringClusterCount -= uint32_t(page.offsets.size());
            p.stats.retiringStorageBytes -= page.allocation.allocatedSize;
            ++p.stats.builtPageCount;
            p.stats.builtClusterCount += uint32_t(page.offsets.size());
            p.publish(request.pageIndex, &page);
        }
    }
    // Only completed build output can become a relocation source. One bounded
    // move batch per frame keeps both allocation work and command cost bounded.
    for (auto& batch : p.batches) {
        if (batch.phase != Impl::Phase::Sized) {
            continue;
        }
        std::erase_if(batch.items, [&](const auto& item) {
            if (p.pages.at(item.page).wanted) {
                return false;
            }
            p.discard(item.page);
            return true;
        });
        if (batch.items.empty()) {
            p.reset(batch);
            continue;
        }
        std::vector<ClusterAccelerationStructureMoveInfo> moves;
        for (auto& item : batch.items) {
            auto& page = p.pages.at(item.page);
            uint64_t bytes = 0;
            for (auto size : item.sizes) {
                bytes += compactAlign(size, p.alignment);
            }
            p.allocate(item.page, page, bytes);
            if (page.allocation.valid()) {
                page.addresses = p.addressStorage.allocate(item.sizes.size());
            }
            if (!page.allocation.valid() || !page.addresses.valid()) {
                if (page.allocation.valid()) {
                    page.chunk->storage.release(page.allocation);
                    page.chunk = nullptr;
                    page.allocation = {};
                }
                ++p.stats.frameRejectedPageCount;
                ++p.stats.totalRejectedPageCount;
                continue;
            }
            uint64_t offset = 0;
            for (uint32_t i = 0; i < item.sizes.size(); ++i) {
                page.offsets.push_back(uint32_t(offset));
                moves.push_back({.sourceBuffer = batch.builder->storageBuffer(),
                                 .sourceOffset = batch.builder->clusterAddress(item.page, i) -
                                                 batch.builder->storageBuffer()->deviceAddress(),
                                 .destinationBuffer = page.chunk->buffer.get(),
                                 .destinationOffset = page.allocation.offset + offset,
                                 .size = item.sizes[i]});
                offset += compactAlign(item.sizes[i], p.alignment);
                page.encodedBytes += item.sizes[i];
            }
            p.stats.encodedStorageBytes += page.encodedBytes;
            p.stats.worstCaseStorageBytes += uint64_t(item.sizes.size()) * p.stride;
            item.moving = true;
        }
        if (moves.empty()) {
            if (!p.error.empty()) { log = p.error; return makeError(Error::Failure); }
            continue;
        }
        Result<> result;
        for (auto& item : batch.items) {
            if (item.moving) {
                result = p.retain(cmd, *p.pages.at(item.page).chunk);
                if (!result) { break; }
            }
        }
        if (!p.error.empty()) { result = makeError(Error::Failure); log = p.error; }
        if (result) {
            result = cmd.moveClusterAccelerationStructures({
                .objects = moves,
                .sourceAddressBuffer = batch.moveSources.get(),
                .destinationAddressBuffer = batch.moveDestinations.get(),
                .scratchBuffer = p.scratch.get(),
                .scratchBufferOffset = p.scratchOffset,
            });
        }
        if (result) {
            result = p.track(cmd, batch);
        }
        if (!result) {
            for (auto& item : batch.items) {
                if (item.moving) {
                    p.release(p.pages.at(item.page));
                    item.moving = false;
                }
            }
            log = "Compact CLAS move recording failed";
            return result;
        }
        for (const auto& item : batch.items) {
            if (item.moving) {
                p.publish(item.page, &p.pages.at(item.page), true);
                ++p.stats.frameMovedPageCount;
                p.stats.frameMovedClusterCount += uint32_t(item.sizes.size());
            }
        }
        batch.phase = Impl::Phase::Moving;
        break;
    }
    auto freeBatch =
        std::find_if(p.batches.begin(), p.batches.end(), [](const auto& b) { return b.phase == Impl::Phase::Free; });
    if (freeBatch == p.batches.end()) {
        return p.flushPublications(cmd, log);
    }
    auto& batch = *freeBatch;
    std::vector<MeshletStreamCLASPageBuild> builds;
    std::unordered_set<uint32_t> unique;
    uint32_t count = 0;
    for (const auto& request : requests) {
        if (p.pages.contains(request.pageIndex) || !unique.insert(request.pageIndex).second) {
            continue;
        }
        const uint32_t clusters = uint32_t(request.plan->clusters.size());
        if (clusters != p.asset->pages()[request.pageIndex].clusterCount || clusters > p.maxBuild) {
            log = "Compact CLAS page exceeds staging build limit";
            return makeError(Error::InvalidArgument);
        }
        if (clusters > p.maxBuild - count) {
            continue;
        }
        builds.push_back(request);
        batch.items.push_back({.page = request.pageIndex, .first = count});
        batch.stagedPages.push_back(request.pageIndex);
        count += clusters;
    }
    if (builds.empty()) {
        return p.flushPublications(cmd, log);
    }
    batch.builder->beginFrame();
    auto result = batch.builder->cmdBuildPages(cmd, geometry, builds, log, batch.sizes.get());
    if (!result) {
        p.reset(batch);
        return result;
    }
    if (batch.builder->stats().frameBuiltClusterCount != count) {
        log = "Compact CLAS staging batch did not fit its reserved capacity";
        return makeError(Error::Failure);
    }
    if (!(result = p.track(cmd, batch))) {
        return result;
    }
    for (const auto& build : builds) {
        p.pages.emplace(build.pageIndex, Impl::Page{});
    }
    batch.phase = Impl::Phase::Building;
    p.stats.frameBuiltPageCount += uint32_t(builds.size());
    p.stats.frameBuiltClusterCount += count;
    p.stats.totalBuiltPageCount += builds.size();
    p.stats.totalBuiltClusterCount += count;
    return p.flushPublications(cmd, log);
}

void MeshletStreamCompactCLASPool::retirePages(std::span<const uint32_t> ids)
{
    auto& p = *impl_;
    for (uint32_t id : ids) {
        auto found = p.pages.find(id);
        if (found == p.pages.end()) {
            continue;
        }
        auto& page = found->second;
        page.wanted = false;
        if (page.state != Impl::State::Active) {
            continue;
        }
        page.state = Impl::State::Retiring;
        page.retireFrame = p.frame + p.queuedFrames;
        page.retirementUses = page.chunk->uses;
        p.retirements.push_back({page.retireFrame, id});
        --p.stats.builtPageCount;
        p.stats.builtClusterCount -= uint32_t(page.offsets.size());
        ++p.stats.retiringPageCount;
        p.stats.retiringClusterCount += uint32_t(page.offsets.size());
        p.stats.retiringStorageBytes += page.allocation.allocatedSize;
        p.publish(id, &page);
    }
}

bool MeshletStreamCompactCLASPool::ready() const
{
    return impl_->initialized;
}
bool MeshletStreamCompactCLASPool::pageHasClas(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && it->second.state != Impl::State::Pending;
}
bool MeshletStreamCompactCLASPool::pageBuildPending(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && it->second.state == Impl::State::Pending;
}
uint64_t MeshletStreamCompactCLASPool::pageStorageBytes(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it == impl_->pages.end() ? 0 : it->second.allocation.allocatedSize;
}
uint32_t MeshletStreamCompactCLASPool::pageClasAddressOffset(uint32_t id) const
{
    return pageHasClas(id) ? uint32_t(impl_->pages.at(id).addresses.offset) : kInvalidMeshletStreamCLASAddressOffset;
}
uint64_t MeshletStreamCompactCLASPool::clusterAddress(uint32_t id, uint32_t cluster) const
{
    if (!pageHasClas(id)) {
        return 0;
    }
    const auto& page = impl_->pages.at(id);
    return cluster < page.offsets.size()
               ? page.chunk->buffer->deviceAddress() + page.allocation.offset + page.offsets[cluster]
               : 0;
}
Buffer* MeshletStreamCompactCLASPool::pageStorageBuffer(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && it->second.chunk ? it->second.chunk->buffer.get() : nullptr;
}
Buffer* MeshletStreamCompactCLASPool::clusterAddressBuffer() const
{
    return impl_->addresses.get();
}
Buffer* MeshletStreamCompactCLASPool::pageTableBuffer() const
{
    return impl_->pageTable.get();
}
MeshletStreamCLASPoolStats MeshletStreamCompactCLASPool::stats() const
{
    auto result = impl_->stats;
    result.trackedPageCount = uint32_t(impl_->pages.size());
    for (const auto& chunk : impl_->chunks) {
        result.usedStorageBytes += chunk->storage.usedBytes();
        if (!chunk->storage.usedBytes()) { result.emptyStorageBytes += chunk->buffer->desc().size; }
        if (chunk->lifetime == Impl::Lifetime::Persistent) {
            result.persistentStorageBytes += chunk->buffer->desc().size;
            result.persistentUsedBytes += chunk->storage.usedBytes();
        } else if (chunk->lifetime == Impl::Lifetime::Transient) {
            result.transientStorageBytes += chunk->buffer->desc().size;
            result.transientUsedBytes += chunk->storage.usedBytes();
        }
        result.fragmentedFreeBytes += chunk->storage.freeBytes() - chunk->storage.largestFreeBlockBytes();
    }
    return result;
}
} // namespace metallic::render
