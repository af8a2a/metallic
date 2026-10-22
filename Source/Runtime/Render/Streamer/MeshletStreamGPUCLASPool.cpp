#include "Runtime/Render/Streamer/MeshletStreamGPUCLASPool.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace metallic::render {
namespace {

uint64_t aligned(uint64_t bytes, uint64_t alignment)
{
    return (bytes + alignment - 1u) / alignment * alignment;
}

constexpr auto kDataUsage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
    BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination |
    BufferUsageBits::AccelerationStructureBuildInput;
constexpr auto kArenaUsage = kDataUsage | BufferUsageBits::AccelerationStructureStorage;

struct Operation {
    uint32_t page, kind, count, addressOffset;
    uint32_t sourceOffset, sizeOffset, generation, reserved = 0;
};
struct Outcome { uint32_t success, bytes, encodedBytes, kind; };
struct Parameters {
    uint64_t state, gaps, allocations, pageTable, addresses, operations, sizes;
    uint64_t sourceAddresses, moveSources, moveDestinations, moveCount, work, outcomes, arena;
    uint32_t operationCount, pageCount, alignment, stride;
    uint32_t committedUnits, nodeCapacity, initialize, reserved = 0;
};
struct Push { uint64_t parameters; uint32_t phase, reserved = 0; };
static_assert(sizeof(Operation) == 32 && sizeof(Outcome) == 16 && sizeof(Parameters) == 144 && sizeof(Push) == 16);

bool finished(const GPUCompletionPoint& point)
{
    return point.isCancelled() || (point.isSubmitted() && point.isComplete());
}

Result<> barrier(CommandBuffer& cmd, Buffer& buffer, ResourceState before = ResourceState::General,
    ResourceState after = ResourceState::General)
{
    const BufferBarrierDesc value{.buffer = &buffer,
        .before = resourceSyncScope(before, PipelineStageBits::AllCommands),
        .after = resourceSyncScope(after, PipelineStageBits::AllCommands)};
    return cmd.synchronize({.buffers = {&value, 1}});
}

} // namespace

struct MeshletStreamGPUCLASPool::Impl {
    struct Page {
        MeshletStreamStorageAllocation addresses;
        uint32_t count = 0;
        uint64_t bytes = 0, encoded = 0, deadline = 0, requestedFrame = 0;
        bool ready = false, pending = true, wanted = true, dirty = false, freeing = false;
        std::vector<GPUCompletionPoint> retirementUses;
    };
    struct Batch {
        std::unique_ptr<MeshletStreamCLASPool> builder;
        std::unique_ptr<Buffer> sizes, operations, parameters, work, outcomes;
        std::unique_ptr<Buffer> sources, destinations, count;
        GPUCompletionPoint completion;
        std::shared_ptr<SubmissionTransaction> transaction;
        std::vector<Operation> ops;
        std::vector<uint32_t> staged;
        uint64_t recordedFrame = 0;
    };
    // Submission callbacks never capture the pool. Accepted binding is separate
    // from allocator visibility; a cancelled frame retries the GPU initialization.
    struct Journal { bool initialized = false; uint64_t capacity = 0; };
    std::shared_ptr<Journal> journal = std::make_shared<Journal>();
    Device* device = nullptr;
    const scene::MeshletStreamAsset* asset = nullptr;
    std::unique_ptr<Buffer> arena, state, gaps, allocations, addresses, table, scratch;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::unique_ptr<BindlessHeap> heap;
    std::vector<Batch> batches;
    std::unordered_map<uint32_t, Page> pages;
    // Only dirty or retiring pages need per-frame maintenance. A queued free
    // leaves this set until cancellation makes it eligible again.
    std::unordered_set<uint32_t> maintenancePages;
    std::unordered_map<uint32_t, uint64_t> firstRequests;
    std::vector<GPUCompletionPoint> uses;
    MeshletStreamStorage addressStorage;
    MeshletStreamCLASPoolStats counters;
    std::vector<bool> persistent;
    uint64_t frame = 0, alignment = 0, stride = 0, scratchOffset = 0, capacity = 0, growth = 0;
    uint64_t requestedCapacity = 0, retryGrowthFrame = 0, reservedBytes = 0;
    uint32_t maxBuild = 0, maxOperations = 0, queuedFrames = 0, nodeCapacity = 0;
    bool initialized = false, sparse = false, allocationPressure = false;
    std::string error;

    Result<> buffer(uint64_t bytes, MemoryLocation memory, std::unique_ptr<Buffer>& out,
        BufferUsageBits usage = kDataUsage, MemoryBudgetDomain domain = MemoryBudgetDomain::CLASScratch)
    {
        return device->createBuffer({.size = std::max(bytes, uint64_t(16)), .usage = usage,
            .memoryLocation = memory, .memoryDomain = domain})
            .transform([&](auto value) { out = std::move(value); });
    }

    void refreshMaintenance(uint32_t id, const Page& page)
    {
        if (!page.freeing && (!page.wanted || page.dirty)) { maintenancePages.insert(id); }
        else { maintenancePages.erase(id); }
    }

    template<bool Add>
    void accountPage(uint32_t id, const Page& page)
    {
        // Pending pages have zero byte counts until a successful outcome sets ready.
        if (!page.ready) { return; }
        const auto adjust = [](auto& value, auto amount) {
            if constexpr (Add) { value += amount; }
            else { value -= amount; }
        };
        adjust(counters.usedStorageBytes, page.bytes);
        adjust(counters.encodedStorageBytes, page.encoded);
        adjust(counters.worstCaseStorageBytes, uint64_t(page.count) * stride);
        adjust(persistent[id] ? counters.persistentUsedBytes : counters.transientUsedBytes, page.bytes);
        if (page.wanted) {
            adjust(counters.builtPageCount, 1u);
            adjust(counters.builtClusterCount, page.count);
        } else {
            adjust(counters.retiringPageCount, 1u);
            adjust(counters.retiringClusterCount, page.count);
            adjust(counters.retiringStorageBytes, page.bytes);
        }
    }

    void releasePage(uint32_t id)
    {
        auto it = pages.find(id);
        if (it == pages.end()) { return; }
        accountPage<false>(id, it->second);
        // Retiring/freeing pages stay charged until their free completes;
        // pending builds retain the worst-case charge until outcomes arrive.
        reservedBytes -= it->second.pending ? uint64_t(it->second.count) * stride : it->second.bytes;
        addressStorage.release(it->second.addresses);
        maintenancePages.erase(id);
        pages.erase(it);
    }

    void collect()
    {
        std::erase_if(uses, finished);
        for (auto& batch : batches) {
            if (!batch.transaction || !batch.transaction->resolved() || !finished(batch.completion)) { continue; }
            const bool cancelled = batch.transaction->cancelled() || batch.completion.isCancelled();
            const Outcome* outcomes = nullptr;
            if (!cancelled) {
                batch.outcomes->invalidate();
                outcomes = static_cast<const Outcome*>(batch.outcomes->map());
                if (!outcomes) { error = "GPU CLAS page-outcome readback failed"; return; }
            }
            for (size_t index = 0; index < batch.ops.size(); ++index) {
                const auto& op = batch.ops[index];
                auto it = pages.find(op.page);
                if (it == pages.end()) { continue; }
                auto& page = it->second;
                if (op.kind == 0) {
                    if (cancelled || outcomes[index].success != 1u) {
                        if (!cancelled && outcomes[index].success == 2u) {
                            error = "GPU CLAS builder returned an invalid encoded size";
                        }
                        if (!cancelled) { ++counters.frameRejectedPageCount; ++counters.totalRejectedPageCount; }
                        if (!cancelled && outcomes[index].success == 0u) { allocationPressure = true; }
                        releasePage(op.page);
                    } else {
                        reservedBytes -= uint64_t(page.count) * stride;
                        reservedBytes += outcomes[index].bytes;
                        page.pending = false;
                        page.ready = true;
                        page.bytes = outcomes[index].bytes;
                        page.encoded = outcomes[index].encodedBytes;
                        accountPage<true>(op.page, page);
                        ++counters.frameBuiltPageCount;
                        counters.frameBuiltClusterCount += page.count;
                        ++counters.totalBuiltPageCount;
                        counters.totalBuiltClusterCount += page.count;
                        const uint64_t latency = batch.recordedFrame - page.requestedFrame;
                        ++counters.readyLatencySamples;
                        counters.readyLatencyFrameTotal += latency;
                        counters.readyLatencyFrameMax = std::max(counters.readyLatencyFrameMax, latency);
                        ++counters.readyLatencyHistogram[std::min(latency, uint64_t(32))];
                        firstRequests.erase(op.page);
                        ++counters.frameMovedPageCount;
                        counters.frameMovedClusterCount += page.count;
                        refreshMaintenance(op.page, page);
                    }
                } else if (op.kind == 1) {
                    if (cancelled) {
                        page.freeing = false;
                        page.dirty = true;
                        refreshMaintenance(op.page, page);
                    }
                    else { releasePage(op.page); }
                } else if (cancelled) {
                    page.dirty = true;
                    refreshMaintenance(op.page, page);
                }
            }
            if (outcomes) { batch.outcomes->unmap(); }
            batch.builder->retirePages(batch.staged);
            batch.builder->beginFrame();
            batch.staged.clear();
            batch.ops.clear();
            batch.transaction.reset();
            batch.completion = {};
        }
    }

    Result<> retain(CommandBuffer& cmd)
    {
        const auto point = RenderFrameContext::from(cmd)->completion();
        if (uses.empty() || !uses.back().sameSubmission(point)) { uses.push_back(point); }
        for (Buffer* resource : {arena.get(), state.get(), gaps.get(), allocations.get(), addresses.get(), table.get(), scratch.get()}) {
            if (auto result = cmd.retainResource(resource->retainAllocation()); !result) { return result; }
        }
        return {};
    }

    Result<> dispatch(CommandBuffer& cmd, Batch& batch, uint32_t phase, uint32_t threads)
    {
        const Push push{batch.parameters->deviceAddress(), phase};
        if (auto result = cmd.bindBindlessHeap(*heap); !result) { return result; }
        if (auto result = cmd.bindExecution(pipeline->execution(), &push, sizeof(push)); !result) { return result; }
        const uint32_t groups = (std::max(threads, 1u) + 63u) / 64u;
        return cmd.dispatch(std::min(groups, 65535u), (groups + 65534u) / 65535u, 1);
    }
};

MeshletStreamGPUCLASPool::MeshletStreamGPUCLASPool() : impl_(std::make_unique<Impl>())
{
}
MeshletStreamGPUCLASPool::~MeshletStreamGPUCLASPool() = default;

Result<> MeshletStreamGPUCLASPool::initialize(Device& device, const MeshletStreamCLASPoolDesc& desc, std::string& log)
{
    auto& p = *impl_;
    log.clear();
    if (!desc.asset || !desc.asset->valid() || !desc.maxStorageBytes || !desc.maxBuildClusters ||
        !desc.queuedFrameCount || !desc.growStorageBytes || desc.startStorageBytes > desc.maxStorageBytes) {
        log = "GPU CLAS requires a valid asset and nonzero storage/build/queue limits";
        return makeError(Error::InvalidArgument);
    }
    p.device = &device;
    p.asset = desc.asset;
    p.sparse = desc.sparseStorage;
    p.maxBuild = desc.maxBuildClusters;
    p.maxOperations = p.maxBuild * 2u;
    p.queuedFrames = desc.queuedFrameCount;
    auto properties = device.queryClusterAccelerationStructureProperties();
    if (!properties) { return std::unexpected(properties.error()); }
    p.alignment = properties->clusterStorageAlignment;
    if (!p.alignment || !properties->scratchAlignment) { return makeError(Error::Unsupported); }
    auto single = device.queryClusterAccelerationStructureTriangleBuildSizes({
        .maxClusterTriangleCount = p.asset->maxClusterTriangles(), .maxClusterVertexCount = p.asset->maxClusterVertices(),
        .maxTotalTriangleCount = p.asset->maxClusterTriangles(), .maxTotalVertexCount = p.asset->maxClusterVertices()});
    if (!single) { return std::unexpected(single.error()); }
    p.stride = aligned(single->accelerationStructureSize, p.alignment);
    auto move = device.queryClusterAccelerationStructureMoveSizes(p.maxBuild, p.stride * p.maxBuild);
    if (!move) { return std::unexpected(move.error()); }
    p.capacity = desc.maxStorageBytes / p.alignment * p.alignment;
    const uint64_t slotCount = std::min(p.capacity / p.alignment,
        uint64_t(p.asset->pageCount()) * p.asset->maxPageClusters());
    if (!p.capacity || p.capacity / p.alignment > UINT32_MAX || p.stride > UINT32_MAX ||
        slotCount > kMeshletStreamCLASPageAddressOffsetMask || !slotCount || p.maxBuild > UINT32_MAX / 2u) {
        log = "GPU CLAS capacity exceeds address or allocator index range";
        return makeError(Error::InvalidArgument);
    }
    p.nodeCapacity = uint32_t(std::min(uint64_t(p.asset->pageCount()), slotCount)) * 2u + 1u;
    if (!p.addressStorage.initialize(slotCount, 1, log, UINT64_MAX)) { return makeError(Error::InvalidArgument); }
    p.persistent.assign(p.asset->pageCount(), false);
    for (uint32_t id : desc.persistentPages) {
        if (id >= p.persistent.size()) { return makeError(Error::InvalidArgument); }
        p.persistent[id] = true;
    }
    Result<> result;
    const BufferDesc arenaDesc{.size = p.capacity, .usage = kArenaUsage,
        .memoryLocation = MemoryLocation::Device, .memoryDomain = MemoryBudgetDomain::CLAS};
    if (p.sparse) {
        result = device.createSparseBuffer(arenaDesc).transform([&](auto value) { p.arena = std::move(value); });
    } else {
        result = device.createBuffer(arenaDesc).transform([&](auto value) { p.arena = std::move(value); });
    }
    if (!result) { log = "GPU CLAS arena creation failed: " + std::string(resultToString(result)); return result; }
    const uint64_t quantum = p.sparse ? p.arena->sparseBlockAlignment() : p.alignment;
    p.growth = aligned(desc.growStorageBytes, quantum);
    p.requestedCapacity = p.sparse ? std::min(p.capacity, aligned(desc.startStorageBytes, quantum)) : p.capacity;
    if (!(result = p.buffer(64u * 4u, MemoryLocation::Device, p.state)) ||
        !(result = p.buffer(uint64_t(p.nodeCapacity) * 32u, MemoryLocation::Device, p.gaps)) ||
        !(result = p.buffer(uint64_t(p.asset->pageCount()) * 16u, MemoryLocation::Device, p.allocations)) ||
        !(result = p.buffer(slotCount * 8u, MemoryLocation::Device, p.addresses)) ||
        !(result = p.buffer(uint64_t(p.asset->pageCount()) * 8u, MemoryLocation::Device, p.table)) ||
        !(result = p.buffer(move->updateScratchSize + properties->scratchAlignment, MemoryLocation::Device, p.scratch, kArenaUsage))) {
        return result;
    }
    p.scratchOffset = aligned(p.scratch->deviceAddress(), properties->scratchAlignment) - p.scratch->deviceAddress();
    ShaderCompileResult compiled;
    result = ShaderRegistry::instance().getShader({.moduleName = "Features/GPUDriven/StreamCLASAllocator", .entryPointName = "allocatorMain",
        .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
        .transform([&](auto value) { compiled = std::move(value); });
    if (!result) { log = compiled.diagnostics; return result; }
    result = ShaderRegistry::instance().getShaderModule(device, {.spirv = compiled.spirv,
        .debugName = "StreamCLASAllocator.allocatorMain"}).transform([&](auto value) { p.shader = std::move(value); });
    if (!result) { return result; }
    result = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {p.shader.get(), "main"},
        .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(Push)})
        .transform([&](auto value) { p.pipeline = std::move(value); });
    if (!result) { return result; }
    result = device.createBindlessHeap({.maxBuffers = 1}).transform([&](auto value) { p.heap = std::move(value); });
    if (!result) { return result; }
    p.batches.resize(p.queuedFrames);
    for (auto& batch : p.batches) {
        batch.builder = std::make_unique<MeshletStreamCLASPool>();
        result = batch.builder->initialize(device, {.asset = p.asset, .maxStorageBytes = p.stride * p.maxBuild,
            .maxBuildClusters = p.maxBuild, .queuedFrameCount = 1, .compactStorage = false}, log);
        if (!result) { return result; }
        if (!(result = p.buffer(uint64_t(p.maxBuild) * 4u, MemoryLocation::Device, batch.sizes, kArenaUsage)) ||
            !(result = p.buffer(uint64_t(p.maxOperations) * sizeof(Operation), MemoryLocation::HostUpload, batch.operations)) ||
            !(result = p.buffer(sizeof(Parameters), MemoryLocation::HostUpload, batch.parameters)) ||
            !(result = p.buffer(uint64_t(p.maxOperations) * 16u, MemoryLocation::Device, batch.work)) ||
            !(result = p.buffer(uint64_t(p.maxOperations) * sizeof(Outcome), MemoryLocation::HostReadback, batch.outcomes)) ||
            !(result = p.buffer(uint64_t(p.maxBuild) * 8u, MemoryLocation::Device, batch.sources)) ||
            !(result = p.buffer(uint64_t(p.maxBuild) * 8u, MemoryLocation::Device, batch.destinations, kArenaUsage)) ||
            !(result = p.buffer(4u, MemoryLocation::Device, batch.count, kDataUsage | BufferUsageBits::Indirect))) { return result; }
        p.counters.scratchBytes += batch.builder->stats().storageBytes + batch.builder->stats().scratchBytes;
        for (Buffer* resource : {batch.sizes.get(), batch.operations.get(), batch.parameters.get(), batch.work.get(),
                 batch.outcomes.get(), batch.sources.get(), batch.destinations.get(), batch.count.get()}) {
            p.counters.scratchBytes += resource->desc().size;
        }
    }
    p.counters.scratchBytes += p.scratch->desc().size + p.state->desc().size + p.gaps->desc().size + p.allocations->desc().size;
    p.counters.pageCapacity = p.asset->pageCount();
    p.counters.clusterSlotCapacity = uint32_t(slotCount);
    p.counters.clusterStrideBytes = p.stride;
    p.counters.storageBudgetBytes = p.capacity;
    p.counters.virtualStorageBytes = p.capacity;
    p.counters.startStorageBytes = p.requestedCapacity;
    p.counters.growStorageBytes = p.growth;
    p.initialized = true;
    return {};
}

void MeshletStreamGPUCLASPool::beginFrame(CPUProfileRecorder* profiler)
{
    if (!ready()) { return; }
    auto& p = *impl_;
    ++p.frame;
    p.counters.frameBuiltPageCount = p.counters.frameBuiltClusterCount = p.counters.frameRejectedPageCount = 0;
    p.counters.frameMovedPageCount = p.counters.frameMovedClusterCount = 0;
    p.counters.frameStorageGrowthCount = p.counters.frameStorageReleaseCount = 0;
    p.counters.frameAllocatorPageCount = 0;
    CPUProfileScope profile(profiler, "Collect GPU CLAS page outcomes");
    p.collect();
}

Result<> MeshletStreamGPUCLASPool::cmdBuildPages(CommandBuffer& cmd, Buffer& geometry,
    std::span<const MeshletStreamCLASPageBuild> requests, std::string& log,
    const std::function<void(std::string_view)>& checkpoint)
{
    auto& p = *impl_;
    log = p.error;
    if (!ready() || !log.empty()) { return makeError(Error::Failure); }
    if (!RenderFrameContext::from(cmd) || !RenderFrameContext::from(cmd)->recording()) {
        log = "GPU CLAS requires tracked frame completion";
        return makeError(Error::InvalidArgument);
    }
    p.collect();
    if (!p.error.empty()) { log = p.error; return makeError(Error::Failure); }
    if (auto result = p.retain(cmd); !result) { return result; }
    for (const auto& request : requests) {
        if (request.pageIndex >= p.asset->pageCount()) { return makeError(Error::InvalidArgument); }
        auto it = p.pages.find(request.pageIndex);
        if (it == p.pages.end() && (!request.plan || request.plan->pageIndex != request.pageIndex || request.plan->clusters.empty() ||
            request.plan->clusters.size() != p.asset->pages()[request.pageIndex].clusterCount ||
            request.plan->clusters.size() > p.maxBuild || request.deviceOffsetBytes > geometry.desc().size ||
            request.plan->payloadByteSize > geometry.desc().size - request.deviceOffsetBytes)) {
            log = "Invalid GPU CLAS page request";
            return makeError(Error::InvalidArgument);
        }
        // A request may wait for an already-recorded free before a replacement
        // page can be built. Include that wait in its first-request latency.
        if (it == p.pages.end() || it->second.freeing) {
            p.firstRequests.try_emplace(request.pageIndex, p.frame);
        }
    }
    // A second recording must not initialize over an earlier, unsubmitted
    // initialization. Its requests remain eligible on the next recording.
    if (!p.journal->initialized && std::any_of(p.batches.begin(), p.batches.end(),
            [](const auto& batch) { return batch.transaction && !batch.transaction->resolved(); })) { return {}; }
    auto freeBatch = std::find_if(p.batches.begin(), p.batches.end(), [](const auto& batch) { return !batch.transaction; });
    if (freeBatch == p.batches.end()) {
        if (p.sparse) { return cmd.commitSparseBuffer(*p.arena, p.arena->sparseCommittedBytes()); }
        return {};
    }
    auto& batch = *freeBatch;
    for (const auto& request : requests) {
        auto it = p.pages.find(request.pageIndex);
        if (it == p.pages.end() || it->second.wanted || it->second.freeing) { continue; }
        auto& page = it->second;
        // Residency demand is CPU scheduling state, independent of whether this
        // command is submitted. Cancellation preserves demand and retries only
        // the GPU state publication; retirePages is its explicit revocation.
        p.accountPage<false>(request.pageIndex, page);
        page.wanted = true;
        p.accountPage<true>(request.pageIndex, page);
        page.dirty = true;
        page.retirementUses.clear();
        p.refreshMaintenance(request.pageIndex, page);
    }
    // Free only after every reader that could retain the old address is done.
    // Address slots stay reserved until the free submission completes as well.
    for (uint32_t id : p.maintenancePages) {
        if (batch.ops.size() >= p.maxBuild) { break; }
        const auto& page = p.pages.at(id);
        uint32_t kind = UINT32_MAX;
        if (!page.wanted && !page.pending && p.frame >= page.deadline &&
            std::all_of(page.retirementUses.begin(), page.retirementUses.end(), finished)) {
            kind = 1;
        } else if (page.dirty) { kind = page.wanted ? 3 : 2; }
        if (kind != UINT32_MAX) {
            batch.ops.push_back({id, kind, page.count, uint32_t(page.addresses.offset), 0, 0, 0});
        }
    }
    std::vector<MeshletStreamCLASPageBuild> builds;
    std::unordered_set<uint32_t> unique;
    uint32_t clusters = 0;
    for (const auto& request : requests) {
        if (p.pages.contains(request.pageIndex) || !unique.insert(request.pageIndex).second) { continue; }
        const uint32_t count = uint32_t(request.plan->clusters.size());
        if (count > p.maxBuild - clusters) { continue; }
        auto slots = p.addressStorage.allocate(count);
        if (!slots.valid()) { ++p.counters.frameRejectedPageCount; ++p.counters.totalRejectedPageCount; continue; }
        p.pages.emplace(request.pageIndex, Impl::Page{.addresses = slots, .count = count,
            .requestedFrame = p.firstRequests.at(request.pageIndex)});
        p.reservedBytes += uint64_t(count) * p.stride;
        batch.ops.push_back({request.pageIndex, 0, count, uint32_t(slots.offset), 0, clusters, 0});
        builds.push_back(request);
        batch.staged.push_back(request.pageIndex);
        clusters += count;
    }
    if (p.sparse) {
        // Delayed per-page observations only regulate physical growth. They do
        // not select offsets or gate this frame's GPU publication.
        const uint64_t previous = p.arena->sparseCommittedBytes();
        const uint64_t pressureTarget = p.allocationPressure ? std::min(p.capacity, previous + p.growth) : previous;
        const uint64_t target = p.frame < p.retryGrowthFrame ? previous :
            std::min(p.capacity, aligned(std::max({p.reservedBytes, p.requestedCapacity, pressureTarget}), p.growth));
        auto result = cmd.commitSparseBuffer(*p.arena, target);
        if (!result && hasError(result, Error::OutOfMemory)) {
            p.retryGrowthFrame = p.frame + 30;
            result = cmd.commitSparseBuffer(*p.arena, p.arena->sparseCommittedBytes());
        }
        if (!result) {
            for (uint32_t id : batch.staged) { p.releasePage(id); }
            batch.ops.clear(); batch.staged.clear();
            log = "GPU CLAS sparse commitment failed: " + std::string(resultToString(result));
            return result;
        }
        p.requestedCapacity = p.arena->sparseCommittedBytes();
        if (p.requestedCapacity >= pressureTarget) { p.allocationPressure = false; }
        if (p.requestedCapacity > previous) { ++p.counters.totalStorageGrowthCount; ++p.counters.frameStorageGrowthCount; }
    }
    const uint64_t committed = p.sparse ? p.arena->sparseCommittedBytes() : p.capacity;
    if (batch.ops.empty() && p.journal->initialized && p.journal->capacity == committed) { return {}; }
    const bool initialize = !p.journal->initialized;
    ++p.counters.publicationRevision;
    p.counters.frameAllocatorPageCount += uint32_t(batch.ops.size());
    const uint32_t generation = uint32_t(p.counters.publicationRevision);
    for (auto& op : batch.ops) { op.generation = generation; }
    batch.completion = RenderFrameContext::from(cmd)->completion();
    batch.recordedFrame = p.frame;
    batch.transaction = std::make_shared<SubmissionTransaction>([journal = p.journal, committed] {
        journal->initialized = true;
        journal->capacity = committed;
    }, nullptr);
    auto result = cmd.addSubmissionTransaction(batch.transaction);
    if (!result) {
        for (uint32_t id : batch.staged) { p.releasePage(id); }
        batch.ops.clear(); batch.staged.clear(); batch.transaction.reset();
        return result;
    }
    for (const auto& op : batch.ops) {
        auto& page = p.pages.at(op.page);
        page.dirty = false;
        if (op.kind == 1) { page.freeing = true; }
        p.refreshMaintenance(op.page, page);
    }
    if (!builds.empty()) {
        if (checkpoint) { checkpoint("BeforeCLASScratchBuild"); }
        batch.builder->beginFrame();
        result = batch.builder->cmdBuildPages(cmd, geometry, builds, log, batch.sizes.get());
        if (!result) { return result; }
        if (batch.builder->stats().frameBuiltClusterCount != clusters) {
            log = "GPU CLAS staging batch exceeded its reserved capacity";
            return makeError(Error::Failure);
        }
        for (auto& op : batch.ops) {
            if (op.kind == 0) { op.sourceOffset = batch.builder->pageClasAddressOffset(op.page); }
        }
    }
    const Parameters parameters{
        p.state->deviceAddress(), p.gaps->deviceAddress(), p.allocations->deviceAddress(), p.table->deviceAddress(),
        p.addresses->deviceAddress(), batch.operations->deviceAddress(), batch.sizes->deviceAddress(),
        batch.builder->clusterAddressBuffer()->deviceAddress(), batch.sources->deviceAddress(), batch.destinations->deviceAddress(),
        batch.count->deviceAddress(), batch.work->deviceAddress(), batch.outcomes->deviceAddress(), p.arena->deviceAddress(),
        uint32_t(batch.ops.size()), p.asset->pageCount(), uint32_t(p.alignment), uint32_t(p.stride),
        uint32_t(committed / p.alignment), p.nodeCapacity, initialize ? 1u : 0u};
    for (const auto [buffer, data, bytes] : std::array{
             std::tuple{batch.parameters.get(), static_cast<const void*>(&parameters), sizeof(parameters)},
             std::tuple{batch.operations.get(), static_cast<const void*>(batch.ops.data()), batch.ops.size() * sizeof(Operation)}}) {
        void* mapped = buffer->map();
        if (!mapped) { return makeError(Error::Failure); }
        if (bytes) { std::memcpy(mapped, data, bytes); }
        buffer->flush(); buffer->unmap();
    }
    for (Buffer* resource : {batch.sizes.get(), batch.operations.get(), batch.parameters.get(), batch.work.get(),
             batch.outcomes.get(), batch.sources.get(), batch.destinations.get(), batch.count.get()}) {
        if (!(result = cmd.retainResource(resource->retainAllocation()))) { return result; }
        if (!(result = barrier(cmd, *resource))) { return result; }
    }
    if (checkpoint) { checkpoint("BeforeCLASAllocate"); }
    for (Buffer* resource : {p.state.get(), p.gaps.get(), p.allocations.get(), p.table.get(), p.addresses.get(),
             batch.builder->clusterAddressBuffer()}) {
        if (!(result = barrier(cmd, *resource))) { return result; }
    }
    if (initialize) {
        if (!(result = p.dispatch(cmd, batch, 0, std::max(p.asset->pageCount(), 64u)))) { return result; }
        for (Buffer* resource : {p.state.get(), p.allocations.get(), p.table.get()}) {
            if (!(result = barrier(cmd, *resource))) { return result; }
        }
    }
    if (!(result = p.dispatch(cmd, batch, 1, uint32_t(batch.ops.size()))) ||
        !(result = barrier(cmd, *batch.work)) ||
        !(result = p.dispatch(cmd, batch, 2, 1))) { return result; }
    for (Buffer* resource : {p.state.get(), p.gaps.get(), p.allocations.get(), p.table.get(), batch.work.get(), batch.count.get()}) {
        if (!(result = barrier(cmd, *resource))) { return result; }
    }
    if (!(result = p.dispatch(cmd, batch, 3, uint32_t(batch.ops.size())))) { return result; }
    for (Buffer* resource : {batch.sources.get(), batch.destinations.get(), batch.count.get(), p.addresses.get()}) {
        if (!(result = barrier(cmd, *resource))) { return result; }
    }
    if (clusters) {
        if (checkpoint) { checkpoint("BeforeCLASMove"); }
        auto sources = batch.sources->slice();
        auto destinations = batch.destinations->slice();
        auto count = batch.count->slice();
        auto scratch = p.scratch->slice({p.scratchOffset});
        auto sourceStorage = batch.builder->storageBuffer()->slice();
        auto destinationStorage = p.arena->slice();
        if (!sources || !destinations || !count || !scratch || !sourceStorage || !destinationStorage) {
            return makeError(Error::InvalidArgument);
        }
        result = cmd.moveClusterAccelerationStructuresIndirect({.maxObjectCount = clusters,
            .maxMovedBytes = uint64_t(clusters) * p.stride, .sourceAddressBuffer = *sources,
            .destinationAddressBuffer = *destinations, .objectCountBuffer = *count, .scratchBuffer = *scratch,
            .sourceStorageBuffer = *sourceStorage, .destinationStorageBuffer = *destinationStorage});
        if (!result) {
            log = "GPU CLAS indirect move recording failed: " + std::string(resultToString(result));
            return result;
        }
    }
    if (checkpoint) { checkpoint("BeforeCLASPublish"); }
    if (!(result = barrier(cmd, *p.arena)) || !(result = p.dispatch(cmd, batch, 4, uint32_t(batch.ops.size())))) { return result; }
    for (Buffer* resource : {p.table.get(), p.addresses.get()}) {
        if (!(result = barrier(cmd, *resource))) { return result; }
    }
    const BufferBarrierDesc observationBarrier{.buffer = batch.outcomes.get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    return cmd.synchronize({.buffers = {&observationBarrier, 1}});
}

void MeshletStreamGPUCLASPool::retirePages(std::span<const uint32_t> ids)
{
    auto& p = *impl_;
    for (uint32_t id : ids) {
        p.firstRequests.erase(id);
        auto it = p.pages.find(id);
        if (it == p.pages.end() || !it->second.wanted) { continue; }
        auto& page = it->second;
        p.accountPage<false>(id, page);
        page.wanted = false;
        p.accountPage<true>(id, page);
        page.dirty = true;
        page.deadline = p.frame + p.queuedFrames;
        page.retirementUses = p.uses;
        p.refreshMaintenance(id, page);
    }
}

bool MeshletStreamGPUCLASPool::ready() const { return impl_->initialized; }
bool MeshletStreamGPUCLASPool::pageHasClas(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && it->second.ready && !it->second.freeing;
}
bool MeshletStreamGPUCLASPool::pageBuildPending(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && (it->second.pending || it->second.freeing);
}
uint64_t MeshletStreamGPUCLASPool::pageStorageBytes(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() ? it->second.bytes : 0;
}
uint32_t MeshletStreamGPUCLASPool::pageClasAddressOffset(uint32_t id) const
{
    const auto it = impl_->pages.find(id);
    return it != impl_->pages.end() && !it->second.freeing
        ? uint32_t(it->second.addresses.offset) : kInvalidMeshletStreamCLASAddressOffset;
}
uint64_t MeshletStreamGPUCLASPool::clusterAddress(uint32_t, uint32_t) const
{
    // Compact offsets exist only on the GPU. Callers must consume the published
    // GPU address table, not mistake a delayed CPU mirror for its authority.
    return 0;
}
Buffer* MeshletStreamGPUCLASPool::pageStorageBuffer(uint32_t id) const
{
    return impl_->pages.contains(id) ? impl_->arena.get() : nullptr;
}
Buffer* MeshletStreamGPUCLASPool::clusterAddressBuffer() const { return impl_->addresses.get(); }
Buffer* MeshletStreamGPUCLASPool::pageTableBuffer() const { return impl_->table.get(); }
uint64_t MeshletStreamGPUCLASPool::publicationRevision() const
{
    return impl_->counters.publicationRevision;
}

MeshletStreamCLASPoolStats MeshletStreamGPUCLASPool::stats() const
{
    const auto& p = *impl_;
    auto result = p.counters;
    uint64_t cumulative = 0;
    bool foundMedian = false;
    for (uint32_t bin = 0; bin < result.readyLatencyHistogram.size(); ++bin) {
        cumulative += result.readyLatencyHistogram[bin];
        if (result.readyLatencySamples && cumulative >= (result.readyLatencySamples + 1u) / 2u && !foundMedian) {
            result.readyLatencyFramesP50 = bin;
            foundMedian = true;
        }
        if (result.readyLatencySamples && cumulative >= (result.readyLatencySamples * 95u + 99u) / 100u) {
            result.readyLatencyFramesP95 = bin;
            break;
        }
    }
    result.storageBytes = p.sparse ? p.arena->sparseCommittedBytes() : p.capacity;
    result.storageChunkCount = result.storageBytes ? 1u : 0u;
    result.trackedPageCount = uint32_t(p.pages.size());
    result.emptyStorageBytes = result.usedStorageBytes == 0 ? result.storageBytes : 0;
    return result;
}

} // namespace metallic::render
