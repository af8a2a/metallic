#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamBLASAllocator.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <json.hpp>

namespace metallic::tests {
namespace {
using namespace render;

template<typename T>
T checked(Result<T> result)
{
    if (!result) { throw std::runtime_error(resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

void require(bool condition, const char* reason)
{
    if (!condition) { throw std::runtime_error(reason); }
}

// Exercises the production allocator with deterministic driver-size inputs.
// ClusterBLASMoveTests separately proves that real implicit-build size/address
// outputs are consumed by the native BLAS Move operation in one submission.
struct AllocatorFixture {
    Device& device;
    Queue& queue;
    std::unique_ptr<BindlessHeap> heap;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<Fence> fence;
    std::vector<std::unique_ptr<Buffer>> buffers;
    bool submitted = false;

    explicit AllocatorFixture(Device& value) : device(value), queue(*value.getQueue(QueueType::Graphics))
    {
        heap = checked(device.createBindlessHeap({.maxBuffers = 1}));
        pool = checked(device.createCommandPool(queue));
        commands = checked(pool->createCommandBuffer());
        fence = checked(device.createFence(false));
        std::string diagnostics;
        auto result = compileSlangShaderToSpirv({
            .moduleName = "Features/GPUDriven/StreamBLASAllocator", .entryPointName = "allocatorMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, diagnostics);
        if (!result) { throw std::runtime_error(diagnostics); }
        auto compiled = std::move(*result);
        shader = checked(device.createShaderModule({.spirv = compiled.spirv}));
        pipeline = checked(device.createComputePipeline({.computeShader = {shader.get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(MeshletStreamBLASAllocatorPush)}));
    }

    ~AllocatorFixture()
    {
        (void)queue.waitIdle();
    }

    Buffer& buffer(uint64_t size)
    {
        buffers.push_back(checked(device.createBuffer({.size = size,
            .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress |
                BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostUpload})));
        auto& result = *buffers.back();
        void* mapped = result.map();
        require(mapped != nullptr, "Allocator buffer map failed");
        std::memset(mapped, 0, size_t(size)); result.flush(); result.unmap();
        return result;
    }

    template<typename T>
    void write(Buffer& target, const T& data)
    {
        void* mapped = target.map(); require(mapped != nullptr, "Allocator input map failed");
        std::memcpy(mapped, &data, sizeof(data)); target.flush(); target.unmap();
    }

    template<typename T>
    T read(Buffer& source)
    {
        source.invalidate(); const void* mapped = source.map();
        require(mapped != nullptr, "Allocator result map failed");
        T result; std::memcpy(&result, mapped, sizeof(result)); source.unmap(); return result;
    }

    void begin()
    {
        if (submitted) { checked(fence->reset()); checked(pool->reset()); }
        checked(commands->begin()); commands->hostWriteBarrier();
        checked(commands->bindBindlessHeap(*heap));
    }

    void barrier()
    {
        std::vector<BufferBarrierDesc> entries;
        for (auto& buffer : buffers) {
            entries.push_back({.buffer = buffer.get(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}});
        }
        checked(commands->synchronize({.buffers = entries}));
    }

    void finish(bool cancel = false)
    {
        std::vector<BufferBarrierDesc> host;
        for (auto& buffer : buffers) {
            host.push_back({.buffer = buffer.get(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}});
        }
        checked(commands->synchronize({.buffers = host}));
        checked(commands->end());
        if (cancel) { checked(pool->reset()); return; }
        CommandBuffer* list[]{commands.get()};
        checked(queue.submit({.commandBuffers = list, .signalFence = fence.get()}));
        checked(fence->wait()); submitted = true;
    }
};

// More than two workgroups, with active owners on both sides of a group boundary
// and a partial tail, exercises the production parallel gather/publish passes.
constexpr uint32_t kOwners = 137;
constexpr uint32_t kStableOwner = kOwners - 1;
constexpr uint32_t kNodes = 4 * kOwners + 1;
constexpr uint32_t kDirty = 8;
constexpr uint32_t kAlignment = 256;
using Owners = std::array<MeshletStreamGPUInstanceBLAS, kOwners>;
using Allocations = std::array<MeshletStreamBLASAllocatorAllocation, kOwners>;
using Gaps = std::array<MeshletStreamBLASAllocatorGap, kNodes>;
using State = std::array<uint32_t, kBLASAllocatorStateWords>;
using PriorityOrder = std::array<uint32_t, kOwners>;
using PriorityImportance = std::array<std::array<uint32_t, 6>, kOwners>;
enum class GatherOrder { Native, Ascending, Descending, Rotated };

void reorderGather(std::array<uint32_t, kOwners>& ids, uint32_t count, GatherOrder order)
{
    require(count <= kOwners, "Gather count exceeds the fixture capacity");
    std::sort(ids.begin(), ids.begin() + count);
    if (order == GatherOrder::Descending) {
        std::reverse(ids.begin(), ids.begin() + count);
    } else if (order == GatherOrder::Rotated) {
        std::rotate(ids.begin(), ids.begin() + count / 2, ids.begin() + count);
    }
}

struct Frame {
    Owners owners;
    Allocations allocations;
    Gaps gaps;
    State state;
    MeshletStreamGPUBLASHeader header;
    std::array<uint32_t, 2> moves;
};

struct AllocatorScenario {
    AllocatorFixture fixture;
    Buffer &state, &gaps, &allocations, &owners, &header, &buildInfos, &buildOwners;
    Buffer &sizes, &sourceAddresses, &moveSources, &moveDestinations, &moveCounts, &parameters, &priorityOrder, &importance, &priorityCacheControl;
    MeshletStreamBLASAllocatorParameters params{};
    std::array<uint32_t, kOwners> encoded{};
    nlohmann::json frames = nlohmann::json::array();
    // Keep the large readback payload out of each caller's return-value slots.
    // Tests take value copies explicitly when comparing across submissions.
    std::unique_ptr<Frame> lastFrame = std::make_unique<Frame>();

    AllocatorScenario(Device& device, uint32_t storageBytes, uint32_t relocateBudget,
        uint32_t scanCount = kOwners) : fixture(device),
        state(fixture.buffer(sizeof(State))), gaps(fixture.buffer(sizeof(Gaps))),
        allocations(fixture.buffer(sizeof(Allocations))), owners(fixture.buffer(sizeof(Owners))),
        header(fixture.buffer(sizeof(MeshletStreamGPUBLASHeader))),
        buildInfos(fixture.buffer(kOwners * sizeof(MeshletStreamGPUBLASBuildInfo))),
        buildOwners(fixture.buffer(kOwners * 4)), sizes(fixture.buffer(kOwners * 4)),
        sourceAddresses(fixture.buffer(kOwners * 8)), moveSources(fixture.buffer(kOwners * 2 * 8)),
        moveDestinations(fixture.buffer(kOwners * 2 * 8)), moveCounts(fixture.buffer(8)),
        parameters(fixture.buffer(sizeof(params))), priorityOrder(fixture.buffer(sizeof(PriorityOrder))),
        importance(fixture.buffer(sizeof(PriorityImportance))), priorityCacheControl(fixture.buffer(8))
    {
        params = {state.deviceAddress(), gaps.deviceAddress(), allocations.deviceAddress(), owners.deviceAddress(),
            header.deviceAddress(), buildInfos.deviceAddress(), buildOwners.deviceAddress(), sizes.deviceAddress(),
            sourceAddresses.deviceAddress(), moveSources.deviceAddress(), moveDestinations.deviceAddress(), moveCounts.deviceAddress(),
            0x100000000ull, 0x200000000ull, kOwners, kAlignment, storageBytes, relocateBudget, kNodes, scanCount, 1, 1,
            0x600000000ull};
        fixture.write(parameters, params);
        std::array<uint64_t, kOwners> sources{};
        for (uint32_t i = 0; i < kOwners; ++i) { sources[i] = 0x400000000ull + uint64_t(i) * 65536; }
        fixture.write(sourceAddresses, sources);
        fixture.begin(); dispatch(0); fixture.finish();
    }

    void priority(const PriorityImportance& values, uint32_t referenceCapacity, uint32_t buildCapacity)
    {
        PriorityOrder order;
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (values[a][0] != values[b][0]) { return values[a][0] > values[b][0]; }
            if (values[a][1] != values[b][1]) { return values[a][1] < values[b][1]; }
            if (values[a][2] != values[b][2]) { return values[a][2] > values[b][2]; }
            return a < b;
        });
        fixture.write(priorityOrder, order); fixture.write(importance, values);
        params.priorityOrder = priorityOrder.deviceAddress(); params.importance = importance.deviceAddress();
        params.referenceCapacity = referenceCapacity; params.buildCapacity = buildCapacity;
        fixture.write(parameters, params);
    }

    void dispatch(uint32_t phase)
    {
        const MeshletStreamBLASAllocatorPush push{parameters.deviceAddress(), phase, 0};
        checked(fixture.commands->bindExecution(fixture.pipeline->execution(), &push, sizeof(push)));
        checked(fixture.commands->dispatch((kOwners + 63) / 64)); fixture.barrier();
    }

    void admissionCache(bool enabled, bool sortChanged = false)
    {
        fixture.write(priorityCacheControl, std::array<uint32_t, 2>{sortChanged ? 1u : 0u, sortChanged ? 0u : 1u});
        params.priorityCacheControl = enabled ? priorityCacheControl.deviceAddress() : 0;
        fixture.write(parameters, params);
    }

    Owners inputs()
    {
        auto values = fixture.read<Owners>(owners);
        for (auto& value : values) {
            value.flags = value.selectedClusterCount ? kMeshletStreamBLASInstanceDynamic : 0;
            value.insertedClusterCount = 0;
        }
        return values;
    }

    Frame snapshot()
    {
        return {fixture.read<Owners>(owners), fixture.read<Allocations>(allocations), fixture.read<Gaps>(gaps),
            fixture.read<State>(state), fixture.read<MeshletStreamGPUBLASHeader>(header),
            fixture.read<std::array<uint32_t, 2>>(moveCounts)};
    }

    const Frame& frame(const Owners& input, bool cancel = false, bool inspectPendingMoves = false,
        GatherOrder gatherOrder = GatherOrder::Native)
    {
        fixture.write(owners, input);
        fixture.write(header, MeshletStreamGPUBLASHeader{});
        // Driver-size synthesis belongs only to this state-machine fixture.
        // Real build/allocate/Move ordering is exercised by the native fixture.
        fixture.begin(); dispatch(5);
        if (gatherOrder != GatherOrder::Native && !cancel) {
            require(params.priorityOrder == 0, "Gather permutation is a baseline-only regression control");
            fixture.finish();
            auto releases = fixture.read<std::array<uint32_t, kOwners>>(buildOwners);
            // Phase 5's transient release count is allocator state word 101.
            reorderGather(releases, fixture.read<State>(state)[101], gatherOrder);
            fixture.write(buildOwners, releases);
            fixture.begin();
        }
        dispatch(1); dispatch(6);
        if (cancel) { dispatch(2); dispatch(8); dispatch(3); }
        fixture.finish(cancel);
        if (cancel) { *lastFrame = snapshot(); return *lastFrame; }
        const auto prepared = snapshot();
        auto ids = fixture.read<std::array<uint32_t, kOwners>>(buildOwners);
        std::array<uint32_t, kOwners> bytes{};
        auto inserted = prepared.owners;
        if (gatherOrder != GatherOrder::Native) {
            const auto originalInfos = fixture.read<std::array<MeshletStreamGPUBLASBuildInfo, kOwners>>(buildInfos);
            auto reorderedInfos = originalInfos;
            reorderGather(ids, prepared.header.blasBuildCount, gatherOrder);
            for (uint32_t build = 0; build < prepared.header.blasBuildCount; ++build) {
                auto& owner = inserted[ids[build]];
                reorderedInfos[build] = originalInfos[owner.blasBuildIndex];
                owner.blasBuildIndex = build;
            }
            fixture.write(buildInfos, reorderedInfos);
            fixture.write(buildOwners, ids);
        }
        for (uint32_t i = 0; i < prepared.header.blasBuildCount; ++i) {
            bytes[i] = encoded[ids[i]];
            // Model the production reference-insertion stage before Build.
            inserted[ids[i]].insertedClusterCount = inserted[ids[i]].selectedClusterCount;
        }
        fixture.write(owners, inserted);
        fixture.write(sizes, bytes);
        fixture.begin(); dispatch(2);
        if (inspectPendingMoves) {
            fixture.finish();
            const auto pending = snapshot();
            const auto sources = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveSources);
            const auto destinations = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveDestinations);
            require(pending.moves[1] != 0, "Pending Move inspection did not exercise relocation");
            for (uint32_t owner = 0; owner < kOwners; ++owner) {
                const auto& allocation = pending.allocations[owner];
                if (allocation.oldNode == UINT32_MAX) { continue; }
                require(allocation.pendingNode == UINT32_MAX && allocation.node == allocation.oldNode &&
                    !pending.gaps[allocation.oldNode].free &&
                    pending.owners[owner].storageOffset == pending.gaps[allocation.oldNode].offset,
                    "Staging plan released or published the old version before copying it to scratch");
            }
            for (uint32_t move = 0; move < pending.moves[1]; ++move) {
                const uint64_t source = sources[kOwners + move], destination = destinations[kOwners + move];
                require(source >= params.arena && source < params.arena + params.storageBytes &&
                    destination >= params.relocationArena && destination < params.relocationArena + params.relocateBudget,
                    "Clean staging Move did not preserve its source in the separate relocation arena");
                for (uint32_t other = 0; other < pending.moves[1]; ++other) {
                    require(destination != sources[kOwners + other], "Move destinations alias a live source");
                }
            }
            fixture.begin();
        }
        // The fixture synthesizes driver sizes and does not execute native
        // Moves. Phase 8 models completion of CopyToStage; the native regression
        // proves the two real copies preserve ray-visible geometry.
        dispatch(8);
        if (inspectPendingMoves) {
            fixture.finish();
            const auto returning = snapshot();
            const auto sources = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveSources);
            const auto destinations = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveDestinations);
            for (uint32_t move = 0; move < returning.moves[1]; ++move) {
                require(sources[kOwners + move] >= params.relocationArena &&
                    sources[kOwners + move] < params.relocationArena + params.relocateBudget &&
                    destinations[kOwners + move] >= params.arena &&
                    destinations[kOwners + move] < params.arena + params.storageBytes,
                    "Clean return Move does not copy staged bytes back to persistent storage");
            }
            for (uint32_t owner = 0; owner < kOwners; ++owner) {
                const auto& allocation = returning.allocations[owner];
                if (allocation.status != 2) { continue; }
                require(allocation.pendingNode != UINT32_MAX && !returning.gaps[allocation.pendingNode].free &&
                    allocation.pendingOffset < returning.owners[owner].storageOffset,
                    "Scratch-assisted relocation failed to reserve a lower persistent destination before publication");
            }
            fixture.begin();
        }
        dispatch(3); fixture.finish();
        *lastFrame = snapshot();
        const auto& output = *lastFrame;
        check(output);
        if (gatherOrder != GatherOrder::Native) {
            const auto sources = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveSources);
            const auto destinations = fixture.read<std::array<uint64_t, 2 * kOwners>>(moveDestinations);
            uint32_t move = 0;
            for (uint32_t owner = 0; owner < kOwners; ++owner) {
                if (output.allocations[owner].status != 1) { continue; }
                require(sources[move] == 0x400000000ull + uint64_t(output.owners[owner].blasBuildIndex) * 65536 &&
                    destinations[move] == params.arena + output.owners[owner].storageOffset,
                    "Stable owner admission lost the permuted native Build source/address mapping");
                ++move;
            }
            require(move == output.moves[0], "Deterministic admission published an unexpected Build Move count");
        }
        frames.push_back({{"builds", output.header.blasBuildCount}, {"repack", output.header.arenaRepack},
            {"encodedBytes", output.state[kBLASAllocatorEncodedBytesWord]},
            {"allocatedBytes", output.state[kBLASAllocatorAllocatedBytesWord]},
            {"largestFreeBytes", output.state[kBLASAllocatorLargestFreeWord]},
            {"freeBytes", output.state[kBLASAllocatorFreeBytesWord]},
            {"relocatedCount", output.state[kBLASAllocatorRelocatedCountWord]},
            {"relocatedBytes", output.state[kBLASAllocatorRelocatedBytesWord]},
            {"suppressedBuilds", output.state[kBLASAllocatorSuppressedBuildsWord]},
            {"pressureTargetBytes", output.state[kBLASAllocatorPressureTargetWord]},
            {"referenceBudgetRejected", output.header.referenceBudgetRejected},
            {"buildBudgetRejected", output.header.buildBudgetRejected},
            {"priorityEvicted", output.state[kBLASAllocatorPriorityEvictedWord]},
            {"postBuildBudgetRejected", output.state[kBLASAllocatorPostBuildBudgetRejectedWord]},
            {"physicalAllocationFailed", output.state[kBLASAllocatorPhysicalAllocationFailedWord]},
            {"waitingClaimBytes", output.state[kBLASAllocatorWaitingClaimBytesWord]},
            {"maxWaitingBytes", output.state[kBLASAllocatorMaxWaitingBytesWord]},
            {"unknownBuildCount", output.state[kBLASAllocatorUnknownBuildCountWord]},
            {"projectedRetryDeferredCount", output.state[kBLASAllocatorProjectedRetryDeferredCountWord]},
            {"projectedDirtyClaimBytes", output.state[kBLASAllocatorProjectedDirtyClaimBytesWord]},
            {"sortReused", output.state[kBLASAllocatorSortReusedWord]},
            {"admissionReused", output.state[kBLASAllocatorAdmissionReusedWord]},
            {"storageRejected", output.header.storageRejected}});
        return output;
    }

    void check(const Frame& output)
    {
        require(output.header.arenaRepack == 0, "Persistent allocator requested a full repack");
        require(output.state[kBLASAllocatorRelocatedBytesWord] <= params.relocateBudget,
            "Clean relocation exceeded the byte budget");
        require(output.moves[1] == output.state[kBLASAllocatorRelocatedCountWord], "Relocation count disagrees with Move list");
        uint32_t allocated = 0, encodedBytes = 0;
        std::vector<std::pair<uint32_t, uint32_t>> intervals;
        for (uint32_t i = 0; i < kOwners; ++i) {
            const auto& allocation = output.allocations[i];
            if (allocation.node == UINT32_MAX) { continue; }
            require(allocation.node < kNodes, "Allocation points outside node capacity");
            const auto& gap = output.gaps[allocation.node];
            require(!gap.free && gap.size != 0 && gap.offset % kAlignment == 0 && gap.size % kAlignment == 0 &&
                gap.offset <= params.storageBytes && gap.size <= params.storageBytes - gap.offset,
                "Published allocation is free, misaligned or outside storage");
            require(output.owners[i].storageOffset == gap.offset && output.owners[i].storageCapacity == gap.size,
                "Published owner does not match its allocation");
            require(gap.size >= allocation.encodedBytes && gap.size - allocation.encodedBytes < kAlignment,
                "Persistent allocation retained bucket-sized internal waste");
            intervals.emplace_back(gap.offset, gap.offset + gap.size);
            allocated += gap.size; encodedBytes += allocation.encodedBytes;
        }
        std::sort(intervals.begin(), intervals.end());
        for (size_t i = 1; i < intervals.size(); ++i) {
            require(intervals[i - 1].second <= intervals[i].first, "Published persistent allocations overlap");
        }
        require(allocated == output.state[kBLASAllocatorAllocatedBytesWord] &&
            encodedBytes == output.state[kBLASAllocatorEncodedBytesWord] &&
            allocated + output.state[kBLASAllocatorFreeBytesWord] == params.storageBytes,
            "Live/free byte accounting leaks or double-counts storage");
    }
};

// Keep snapshots on the heap and this regression out of the main test's large
// stack frame. The fixture deliberately changes atomic gather order instead of
// relying on the GPU scheduler to reproduce an unlucky ordering.
nlohmann::json runDeterministicBaselineRegression(Device& device, std::string& phase)
{
    AllocatorScenario scenario(device, 4096, 0);
    auto value = std::make_unique<Frame>();
    std::array<std::unique_ptr<Frame>, 3> reference;
    const auto reset = [&] {
        scenario.encoded.fill(0);
        scenario.fixture.write(scenario.gaps, Gaps{});
        scenario.fixture.write(scenario.owners, Owners{});
        scenario.fixture.begin(); scenario.dispatch(0); scenario.fixture.finish();
    };
    const auto compare = [&](uint32_t checkpoint) {
        if (!reference[checkpoint]) {
            reference[checkpoint] = std::make_unique<Frame>(*value);
            return;
        }
        const auto& expected = *reference[checkpoint];
        require(std::memcmp(value->allocations.data(), expected.allocations.data(), sizeof(Allocations)) == 0 &&
            std::memcmp(value->gaps.data(), expected.gaps.data(), sizeof(Gaps)) == 0 &&
            std::memcmp(value->state.data(), expected.state.data(), sizeof(State)) == 0,
            "Baseline gather order changed persistent allocation or free topology");
        require(value->header.blasBuildCount == expected.header.blasBuildCount &&
            value->header.storageRejected == expected.header.storageRejected &&
            value->header.reusedInstances == expected.header.reusedInstances,
            "Baseline gather order changed admission accounting");
        for (uint32_t owner = 0; owner < kOwners; ++owner) {
            const auto& actual = value->owners[owner];
            const auto& prior = expected.owners[owner];
            require(actual.cachedValid == prior.cachedValid && actual.flags == prior.flags &&
                actual.storageOffset == prior.storageOffset && actual.storageCapacity == prior.storageCapacity,
                "Baseline gather order changed an owner's published representation");
        }
    };
    constexpr std::array orders{GatherOrder::Ascending, GatherOrder::Descending, GatherOrder::Rotated};
    constexpr std::array coldOwners{0u, 7u, 63u, 64u, 65u, 127u, 136u};
    constexpr std::array coldSizes{1281u, 256u, 257u, 1793u, 511u, 769u, 256u};
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        for (const auto order : orders) {
            phase = "baseline deterministic cold admission and static convergence";
            reset();
            Owners input{};
            for (uint32_t index = 0; index < coldOwners.size(); ++index) {
                const uint32_t owner = coldOwners[index];
                input[owner].selectedClusterCount = index + 1;
                input[owner].clusterReferenceOffset = index * 16;
                input[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                scenario.encoded[owner] = coldSizes[index];
            }
            *value = scenario.frame(input, false, false, order);
            uint32_t remaining = scenario.params.storageBytes;
            for (const uint32_t owner : coldOwners) {
                const uint32_t bytes = (scenario.encoded[owner] + kAlignment - 1) & ~(kAlignment - 1);
                const bool admitted = bytes <= remaining;
                require(value->owners[owner].cachedValid == (admitted ? 1u : 2u),
                    "Baseline cold admission does not follow stable owner order with oversized-item backfill");
                if (admitted) { remaining -= bytes; }
            }
            require(remaining == 0 && value->header.blasBuildCount == coldOwners.size() &&
                value->header.storageRejected == 1, "Cold deterministic fixture did not exercise tight storage");
            compare(0);
            for (uint32_t frame = 0; frame < 32; ++frame) {
                *value = scenario.frame(scenario.inputs(), false, false, order);
                require(value->header.blasBuildCount == 0 && value->header.reusedInstances == 6 &&
                    value->header.storageRejected == 1 && value->state[kBLASAllocatorSuppressedBuildsWord] == 1,
                    "Deterministic baseline rebuilt a stable resident or unchanged rejected cut");
            }
        }
    }
    for (const auto order : orders) {
        phase = "baseline deterministic dirty/dead release and cancellation";
        reset();
        Owners input{};
        constexpr std::array residentOwners{0u, 7u, 32u, 63u, 64u, 95u, 127u, 136u};
        for (const uint32_t owner : residentOwners) {
            input[owner].selectedClusterCount = 1;
            input[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            scenario.encoded[owner] = 512;
        }
        *value = scenario.frame(input, false, false, order);
        input = scenario.inputs();
        input[0] = {};
        for (const uint32_t owner : {32u, 64u, 127u}) {
            input[owner].flags |= kDirty;
            scenario.encoded[owner] = owner == 64 ? 768 : 256;
        }
        auto cancelled = std::make_unique<Frame>(scenario.frame(input, true, false, order));
        require(std::memcmp(cancelled->allocations.data(), value->allocations.data(), sizeof(Allocations)) == 0 &&
            std::memcmp(cancelled->gaps.data(), value->gaps.data(), sizeof(Gaps)) == 0 &&
            std::memcmp(cancelled->state.data(), value->state.data(), sizeof(State)) == 0,
            "Cancelled baseline release/build recording changed persistent state");
        *value = scenario.frame(input, false, false, order);
        require(value->header.blasBuildCount == 3 && value->header.storageRejected == 1 &&
            value->header.reusedInstances == 4 && value->owners[64].cachedValid == 2,
            "Dirty/dead release fixture did not exercise fragmented baseline rejection");
        compare(1);
        for (uint32_t frame = 0; frame < 32; ++frame) {
            *value = scenario.frame(scenario.inputs(), false, false, order);
            require(value->header.blasBuildCount == 0 && value->header.reusedInstances == 6 &&
                value->state[kBLASAllocatorSuppressedBuildsWord] == 1,
                "Fragmented baseline negative cache did not converge without rebuilding");
        }
        // Releasing this intervening survivor joins two 512-byte holes. The
        // same previously rejected owner must retry once under every ordering.
        input = scenario.inputs(); input[7] = {};
        *value = scenario.frame(input, false, false, order);
        require(value->header.blasBuildCount == 1 && value->owners[64].cachedValid == 1 &&
            value->header.storageRejected == 0, "Deterministic baseline failed to retry after a fitting gap appeared");
        compare(2);
        for (uint32_t frame = 0; frame < 32; ++frame) {
            *value = scenario.frame(scenario.inputs(), false, false, order);
            require(value->header.blasBuildCount == 0 && value->header.reusedInstances == 6 &&
                value->header.storageRejected == 0, "Recovered deterministic baseline did not remain stable");
        }
    }
    return std::move(scenario.frames);
}

// Consume an actual allocator frame's header through the next frame's
// production representation planner. The cache becomes idle only here; it was
// an active clean survivor in the allocator frame that produced the header.
struct IdleCacheAdmissionProbe {
    enum Input { Params, Primitives, RTState, Header, Active, Representations, InputCount };
    static constexpr uint32_t kCapacity = 8, kRTStateWords = 16 + 3 * 12 + kMeshletStreamRTPriorityControlWords;
    AllocatorFixture& fixture;
    std::unique_ptr<BindlessHeap> heap;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::array<Buffer*, InputCount> buffers;
    std::array<BindlessHandle, InputCount> handles;

    explicit IdleCacheAdmissionProbe(AllocatorFixture& value) : fixture(value)
    {
        heap = checked(fixture.device.createBindlessHeap({.maxBuffers = InputCount}));
        const uint64_t sizes[] = {sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUPrimitive),
            kRTStateWords * 4, sizeof(MeshletStreamGPUBLASHeader) + (kCapacity + 1) * 16,
            sizeof(MeshletStreamGPUActiveHeader), 3 * sizeof(MeshletStreamGPUInstanceBLAS)};
        for (uint32_t input = 0; input < InputCount; ++input) {
            buffers[input] = &fixture.buffer(sizes[input]);
            handles[input] = checked(heap->allocate(BindlessHandleKind::Buffer));
            checked(heap->writeStorageBuffer(handles[input], checked(buffers[input]->slice())));
        }
        std::string diagnostics;
        auto compiled = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamBLASInputEntryPoint, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, diagnostics);
        if (!compiled) { throw std::runtime_error(diagnostics); }
        shader = checked(fixture.device.createShaderModule({.spirv = compiled->spirv}));
        pipeline = checked(fixture.device.createComputePipeline({.computeShader = {shader.get()},
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}));
    }

    void check(MeshletStreamGPUBLASHeader precedingHeader, bool priority, bool expectEviction)
    {
        MeshletStreamGPUParams params{};
        params.scenePrimitiveCount = params.sceneLodLevelCount = 1;
        params.rtRepresentationCount = 3;
        params.rtRepresentationGroupCapacity = kCapacity;
        params.rtRepresentationStateBuffer = handles[RTState].shaderIndex;
        params.rtRepresentationHeaderBuffer = handles[Active].shaderIndex;
        params.rtRepresentationBuffer = handles[Representations].shaderIndex;
        params.lodTopologyBuffer = params.lodStateBuffer = handles[RTState].shaderIndex;
        params.enableBlasRepresentations = params.enableBlasGPUAllocator = 1;
        params.enableBlasPriority = priority;
        params.blasClusterReferenceCapacity = params.blasStorageBytes = params.blasBuildCapacity = UINT32_MAX;
        MeshletStreamGPUPrimitive primitive{};
        primitive.lodLevelCount = 1;
        std::array<MeshletStreamGPUInstanceBLAS, 3> representations{};
        representations[2].previousGroupCount = 1;
        std::array<uint32_t, kRTStateWords> rtState{};
        rtState[0] = rtState[10] = 3;
        rtState[1] = 3 * 64;
        rtState[15] = 1u << 29; // The discrete cache became idle in this frame.
        for (uint32_t owner = 0; owner < 3; ++owner) { rtState[16 + owner * 6 + 1] = 0x7f7fffffu; }
        rtState[16] = 1;
        rtState[19] = rtState[20] = 1;
        // Representation counts belong to this probe scene. All allocator
        // rejection/overflow counters are passed through without modification.
        precedingHeader.representationCount = 2;
        precedingHeader.representationGroupCount = precedingHeader.representationGroupOverflow = 0;
        MeshletStreamGPUActiveHeader active{};
        active.activeGroupCapacity = kCapacity;
        fixture.write(*buffers[Params], params);
        fixture.write(*buffers[Primitives], primitive);
        fixture.write(*buffers[RTState], rtState);
        fixture.write(*buffers[Header], precedingHeader);
        fixture.write(*buffers[Active], active);
        fixture.write(*buffers[Representations], representations);
        MeshletStreamUserPush push{.pageTableBuffer = handles[RTState].shaderIndex,
            .paramsBuffer = handles[Params].shaderIndex, .instanceBuffer = handles[Primitives].shaderIndex,
            .primitiveBuffer = handles[Primitives].shaderIndex, .groupBuffer = handles[Primitives].shaderIndex,
            .blasHeaderBuffer = handles[Header].shaderIndex};
        fixture.begin();
        checked(fixture.commands->bindBindlessHeap(*heap));
        const auto dispatch = [&](uint32_t phase) {
            push.activeBuildPhase = phase;
            checked(fixture.commands->bindExecution(pipeline->execution(), &push, sizeof(push)));
            checked(fixture.commands->dispatch(1)); fixture.barrier();
        };
        if (priority) { dispatch(35); dispatch(37); }
        dispatch(31); dispatch(26);
        if (priority) { dispatch(38); dispatch(39); dispatch(40); }
        dispatch(32); dispatch(33); dispatch(34);
        fixture.finish();
        rtState = fixture.read<decltype(rtState)>(*buffers[RTState]);
        const auto header = fixture.read<MeshletStreamGPUBLASHeader>(*buffers[Header]);
        active = fixture.read<MeshletStreamGPUActiveHeader>(*buffers[Active]);
        require(((rtState[15] & (1u << 30)) != 0) == expectEviction &&
            header.representationPadding == uint32_t(expectEviction) &&
            header.representationCount == (expectEviction ? 1u : 2u) &&
            rtState[10] == (expectEviction ? 2u : 3u) && active.activeGroupCount == rtState[10],
            "Next-frame production representation planning confused projected retry delay with actual storage pressure");
        require(header.storageRejected == precedingHeader.storageRejected,
            "Idle-cache pressure probe changed the allocator rejection counter it was meant to consume");
    }
};

// Keep large frame snapshots in a separate stack frame from the main allocator
// suite. A Frame contains all 137 owners and 549 gap records.
nlohmann::json runAdmissionCacheRegression(Device& device, std::string& phase)
{
    Owners owners{};
    Frame value{};
    phase = "exact stationary admission cache invalidation and current consumer metrics";
    AllocatorScenario cachedAdmission(device, 1024, 0);
    PriorityImportance cachePriorities{};
    for (auto& priority : cachePriorities) { priority[1] = 0x7f7fffffu; }
    owners = cachedAdmission.inputs();
    for (const uint32_t owner : {0u, 64u, 136u}) {
        cachePriorities[owner] = {std::bit_cast<uint32_t>(float(owner + 1)) + 1,
            std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
        owners[owner].selectedClusterCount = 4;
        owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        cachedAdmission.encoded[owner] = 512;
    }
    cachedAdmission.priority(cachePriorities, 64, kOwners);
    cachedAdmission.admissionCache(true, true);
    value = cachedAdmission.frame(owners);
    cachedAdmission.admissionCache(true);
    value = cachedAdmission.frame(cachedAdmission.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0 && value.header.blasBuildCount == 0,
        "Native builds armed an admission cache before a complete quiet selection");
    for (uint32_t frame = 0; frame < 32; ++frame) {
        value = cachedAdmission.frame(cachedAdmission.inputs());
        require(value.state[kBLASAllocatorAdmissionReusedWord] == 1 && value.header.blasBuildCount == 0 &&
            value.header.reusedInstances == 2 && value.header.storageRejected == 1 &&
            value.state[kBLASAllocatorSuppressedBuildsWord] == 1,
            "Stable exact negative and resident admission did not reuse its decision for 32 frames");
    }
    cachePriorities[0][3] = 11; cachePriorities[0][4] = 7; cachePriorities[0][5] = 4;
    cachedAdmission.priority(cachePriorities, 64, kOwners);
    value = cachedAdmission.frame(cachedAdmission.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1 &&
        value.state[kBLASAllocatorRequestedVisibleWord] == 9 && value.state[kBLASAllocatorRejectedVisibleWord] == 7 &&
        value.state[kBLASAllocatorRequestedOffscreenWord] == 4 && value.state[kBLASAllocatorRejectedOffscreenWord] == 4,
        "Admission reuse cached stale consumer feedback instead of aggregating current consumers");
    cachePriorities[0][0] = std::bit_cast<uint32_t>(1000.f) + 1;
    cachedAdmission.priority(cachePriorities, 64, kOwners);
    cachedAdmission.admissionCache(true, true);
    owners = cachedAdmission.inputs();
    const auto beforeCacheCancel = cachedAdmission.snapshot();
    const auto afterCacheCancel = cachedAdmission.frame(owners, true);
    require(std::memcmp(beforeCacheCancel.allocations.data(), afterCacheCancel.allocations.data(), sizeof(Allocations)) == 0 &&
        std::memcmp(beforeCacheCancel.state.data(), afterCacheCancel.state.data(), sizeof(State)) == 0,
        "Cancelled sorted admission changed its input history or cached decision");
    value = cachedAdmission.frame(owners);
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0 && value.header.blasBuildCount == 1 &&
        value.owners[0].cachedValid == 1 && value.owners[64].cachedValid == 2,
        "Changed exact priority order reused stale storage admission");
    cachedAdmission.admissionCache(true);
    cachedAdmission.frame(cachedAdmission.inputs());
    value = cachedAdmission.frame(cachedAdmission.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1, "New priority order did not converge to a reusable decision");
    owners = cachedAdmission.inputs(); owners[136].flags |= kDirty;
    value = cachedAdmission.frame(owners);
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0 && value.header.blasBuildCount == 1,
        "A dirty cut with unchanged counts/order incorrectly reused admission");
    cachedAdmission.frame(cachedAdmission.inputs());
    cachedAdmission.frame(cachedAdmission.inputs());
    owners = cachedAdmission.inputs(); owners[64].selectedClusterCount = 8;
    value = cachedAdmission.frame(owners);
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0 && value.header.blasBuildCount == 0,
        "Changed rejected-owner demand reused an old decision");
    cachedAdmission.priority(cachePriorities, 4, 1);
    value = cachedAdmission.frame(cachedAdmission.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0, "Changed work quotas did not invalidate admission");
    value = cachedAdmission.frame(cachedAdmission.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1, "Unchanged new work quotas did not converge");

    phase = "cached waiting admission follows useful compaction changes, not address-only epochs";
    AllocatorScenario cachedWaiting(device, 4096, 0);
    PriorityImportance compactPriorities{};
    for (auto& priority : compactPriorities) { priority[1] = 0x7f7fffffu; }
    owners = cachedWaiting.inputs();
    for (uint32_t owner = 0; owner < 8; ++owner) {
        compactPriorities[owner] = {std::bit_cast<uint32_t>(float(100 - owner)) + 1,
            std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
        owners[owner].selectedClusterCount = 1;
        owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        cachedWaiting.encoded[owner] = 512;
    }
    cachedWaiting.priority(compactPriorities, kOwners, kOwners);
    cachedWaiting.admissionCache(true, true);
    cachedWaiting.frame(owners);
    owners = cachedWaiting.inputs();
    for (uint32_t owner = 0; owner < 8; owner += 2) { owners[owner] = {}; }
    compactPriorities[136] = {std::bit_cast<uint32_t>(1000.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
    owners[136].selectedClusterCount = 1;
    owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
    cachedWaiting.encoded[136] = 1024;
    cachedWaiting.priority(compactPriorities, kOwners, kOwners);
    cachedWaiting.frame(owners);
    cachedWaiting.admissionCache(true);
    cachedWaiting.frame(cachedWaiting.inputs());
    value = cachedWaiting.frame(cachedWaiting.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1 && value.header.blasBuildCount == 0 &&
        value.state[kBLASAllocatorWaitingCountWord] == 1 && value.state[kBLASAllocatorWaitingClaimBytesWord] == 1024,
        "Known fragmented waiting state did not reuse its exact claim without native builds");
    auto cursorState = value.state;
    cursorState[100] = 7; // First compact the last live block: largest gap stays 512.
    cachedWaiting.fixture.write(cachedWaiting.state, cursorState);
    cachedWaiting.params.relocateBudget = 512;
    cachedWaiting.fixture.write(cachedWaiting.parameters, cachedWaiting.params);
    const auto epochBeforeMove = value.state[kBLASAllocatorEpochWord];
    value = cachedWaiting.frame(cachedWaiting.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1 && value.moves[1] == 1 &&
        value.state[kBLASAllocatorLargestFreeWord] == 512 && value.state[kBLASAllocatorEpochWord] != epochBeforeMove,
        "A cache hit suppressed bounded clean compaction or changed an address-only gap result");
    value = cachedWaiting.frame(cachedWaiting.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 1 && value.moves[1] == 1 &&
        value.state[kBLASAllocatorLargestFreeWord] >= 1024,
        "An address-only epoch change needlessly invalidated admission or stopped compaction progress");
    value = cachedWaiting.frame(cachedWaiting.inputs());
    require(value.state[kBLASAllocatorAdmissionReusedWord] == 0 && value.header.blasBuildCount == 1 &&
        value.owners[136].cachedValid == 1,
        "A newly fitting largest gap did not invalidate the cached wait and admit its owner");

    phase = "zero-build exact allocation skip preserves an exhausted eviction cap";
    nlohmann::json exhaustedCacheFrames = nlohmann::json::array();
    for (bool cacheEnabled : {false, true}) {
        AllocatorScenario exhaustedCache(device, 4096, 0, 1);
        PriorityImportance exhaustPriorities{};
        for (auto& priority : exhaustPriorities) { priority[1] = 0x7f7fffffu; }
        owners = exhaustedCache.inputs();
        for (uint32_t owner = 0; owner < 8; ++owner) {
            exhaustPriorities[owner] = {std::bit_cast<uint32_t>(float(100 + owner)) + 1,
                std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            owners[owner].selectedClusterCount = 1;
            owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            exhaustedCache.encoded[owner] = 512;
        }
        exhaustPriorities[136] = {std::bit_cast<uint32_t>(1.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
        owners[136].selectedClusterCount = 1; owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        exhaustedCache.encoded[136] = 1024;
        exhaustedCache.priority(exhaustPriorities, kOwners, kOwners);
        exhaustedCache.admissionCache(cacheEnabled, true);
        exhaustedCache.frame(owners);
        exhaustPriorities[136][0] = std::bit_cast<uint32_t>(1000.f) + 1;
        exhaustedCache.priority(exhaustPriorities, kOwners, kOwners);
        value = exhaustedCache.frame(exhaustedCache.inputs());
        require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 7 &&
            value.state[kBLASAllocatorPriorityEvictedWord] == 1 && value.state[kBLASAllocatorWaitingCountWord] == 1 &&
            value.state[kBLASAllocatorAllocatedBytesWord] == 3584 && value.state[kBLASAllocatorFreeBytesWord] == 512,
            "Skipping a redundant no-build exact pass changed cap-limited eviction or waiting claims");
        value = exhaustedCache.frame(exhaustedCache.inputs());
        require(value.header.blasBuildCount == 1 && value.owners[136].cachedValid == 1 &&
            value.state[kBLASAllocatorAllocatedBytesWord] == 4096,
            "A cap-limited waiting owner failed to progress after the next bounded eviction");
        exhaustedCacheFrames.push_back({{"enabled", cacheEnabled}, {"frames", exhaustedCache.frames}});
    }

    return {{"cachedAdmission", cachedAdmission.frames}, {"cachedWaiting", cachedWaiting.frames},
        {"exhaustedEvictionCache", exhaustedCacheFrames}};
}

struct MixedTileRegressionResult {
    nlohmann::json frames;
    std::vector<Frame> checkpoints;
};

MixedTileRegressionResult runMixedPriorityTileCases(Device& device, std::string& phase, bool fastPath)
{
    MixedTileRegressionResult result;
    phase = "interleaved resident and oversized negative priority tiles";
    auto scenario = std::make_unique<AllocatorScenario>(device, 4096, 0);
    scenario->params.reserved = uint32_t(fastPath);
    PriorityImportance priorities{};
    // Retain inactive ranks, so the live/negative pairs straddle both full
    // tiles and the partial tail instead of being compacted into one tile.
    for (uint32_t owner = 0; owner < kOwners; ++owner) {
        priorities[owner] = {std::bit_cast<uint32_t>(float(kOwners - owner)) + 1,
            std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
    }
    scenario->priority(priorities, kOwners, kOwners);
    auto owners = scenario->inputs();
    for (const uint32_t owner : {0u, 63u, 64u, 127u, 136u, 1u, 62u, 65u, 126u}) {
        owners[owner].selectedClusterCount = 1;
        owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        scenario->encoded[owner] = owner == 1 || owner == 62 || owner == 65 || owner == 126 ? 8192 : 256;
    }
    auto value = scenario->frame(owners);
    result.checkpoints.push_back(value);
    require(value.header.blasBuildCount == 9 && value.header.storageRejected == 4 &&
        value.state[kBLASAllocatorAllocatedBytesWord] == 1280,
        "Mixed priority fixture did not establish five residents and four measured negative entries");
    const auto residentOffsets = value.owners;
    for (const bool cached : {false, true}) {
        scenario->admissionCache(cached, true);
        result.checkpoints.push_back(scenario->frame(scenario->inputs()));
        scenario->admissionCache(cached);
        for (uint32_t frame = 0; frame < 32; ++frame) {
            value = scenario->frame(scenario->inputs());
            result.checkpoints.push_back(value);
            require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 5 &&
                value.header.storageRejected == 4 && value.state[kBLASAllocatorSuppressedBuildsWord] == 4 &&
                value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0 &&
                value.state[kBLASAllocatorPriorityEvictedWord] == 0 &&
                value.state[kBLASAllocatorAdmissionReusedWord] == uint32_t(cached),
                "Mixed resident/negative tiles changed rejection accounting or stationary cache behavior");
            for (const uint32_t owner : {0u, 63u, 64u, 127u, 136u}) {
                require(value.owners[owner].cachedValid == 1 &&
                    value.owners[owner].storageOffset == residentOffsets[owner].storageOffset,
                    "Mixed negative fast rejection overwrote or moved a resident owner");
            }
        }
    }
    phase = "mixed priority tile dirty and unknown cuts invalidate the summary";
    owners = scenario->inputs();
    owners[63].flags |= kDirty;
    owners[2].selectedClusterCount = 1;
    owners[2].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
    scenario->encoded[2] = 256;
    value = scenario->frame(owners);
    result.checkpoints.push_back(value);
    require(value.header.blasBuildCount == 2 && value.owners[2].cachedValid == 1 &&
        value.owners[63].cachedValid == 1 && value.state[kBLASAllocatorAllocatedBytesWord] == 1536 &&
        value.state[kBLASAllocatorAdmissionReusedWord] == 0,
        "Mixed tile summary bypassed an unknown or dirty cut");

    phase = "mixed negative exact remaining-budget boundary must retry";
    auto equality = std::make_unique<AllocatorScenario>(device, 4096, 0);
    equality->params.reserved = uint32_t(fastPath);
    auto equalityPriorities = priorities;
    equalityPriorities[1][0] = 1;
    equality->priority(equalityPriorities, kOwners, kOwners);
    owners = equality->inputs();
    for (const uint32_t owner : {0u, 63u, 64u, 127u, 136u, 1u}) {
        owners[owner].selectedClusterCount = 1;
        owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        equality->encoded[owner] = owner == 1 ? 3840 : 256;
    }
    value = equality->frame(owners);
    result.checkpoints.push_back(value);
    require(value.owners[1].cachedValid == 2 && value.state[kBLASAllocatorAllocatedBytesWord] == 1280,
        "Exact-boundary fixture lacks a measured negative cut");
    equality->priority(priorities, kOwners, kOwners);
    value = equality->frame(equality->inputs());
    result.checkpoints.push_back(value);
    require(value.header.blasBuildCount == 1 && value.owners[0].cachedValid == 1 &&
        value.owners[1].cachedValid == 1 && value.state[kBLASAllocatorAllocatedBytesWord] == 4096 &&
        value.state[kBLASAllocatorPriorityEvictedWord] == 4,
        "A negative exactly equal to remaining storage was incorrectly rejected by the tile shortcut");

    phase = "mixed resident sum exceeding a preceding waiting claim stays serialized";
    auto exhausted = std::make_unique<AllocatorScenario>(device, 32768, 0, 1);
    exhausted->params.reserved = uint32_t(fastPath);
    exhausted->priority(priorities, kOwners, kOwners);
    owners = exhausted->inputs();
    for (uint32_t owner = 0; owner < 128; ++owner) {
        owners[owner].selectedClusterCount = 1;
        owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
        exhausted->encoded[owner] = owner == 80 ? 65536 : 256;
    }
    owners[136].selectedClusterCount = 1;
    owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
    exhausted->encoded[136] = 8192;
    value = exhausted->frame(owners);
    result.checkpoints.push_back(value);
    require(value.state[kBLASAllocatorAllocatedBytesWord] == 127 * 256 && value.owners[136].cachedValid == 2,
        "Resident-sum fixture did not retain its waiting candidate and oversized negative");
    auto exhaustedPriorities = priorities;
    exhaustedPriorities[136][0] = std::bit_cast<uint32_t>(1000.f) + 1;
    exhausted->priority(exhaustedPriorities, kOwners, kOwners);
    value = exhausted->frame(exhausted->inputs());
    result.checkpoints.push_back(value);
    require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 126 &&
        value.state[kBLASAllocatorPriorityEvictedWord] == 1 &&
        value.state[kBLASAllocatorWaitingCountWord] == 1 &&
        value.state[kBLASAllocatorWaitingClaimBytesWord] == 8192 &&
        value.state[kBLASAllocatorAllocatedBytesWord] == 126 * 256 &&
        value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0,
        "Mixed tile resident overflow skipped the eviction cap or corrupted waiting claims");
    result.frames = {{"mixedTiles", scenario->frames}, {"exactBoundary", equality->frames},
        {"residentSumExhaustion", exhausted->frames}};
    return result;
}

nlohmann::json runMixedPriorityTileRegression(Device& device, std::string& phase)
{
    const auto serial = runMixedPriorityTileCases(device, phase, false);
    const auto optimized = runMixedPriorityTileCases(device, phase, true);
    phase = "mixed tile fast path preserves every serial allocator output";
    require(serial.checkpoints.size() == optimized.checkpoints.size(), "Mixed tile comparison frame counts differ");
    for (size_t frame = 0; frame < serial.checkpoints.size(); ++frame) {
        const auto& a = serial.checkpoints[frame];
        const auto& b = optimized.checkpoints[frame];
        require(std::memcmp(a.owners.data(), b.owners.data(), sizeof(Owners)) == 0 &&
            std::memcmp(a.allocations.data(), b.allocations.data(), sizeof(Allocations)) == 0 &&
            std::memcmp(a.gaps.data(), b.gaps.data(), sizeof(Gaps)) == 0 &&
            std::memcmp(a.state.data(), b.state.data(), sizeof(State)) == 0 &&
            std::memcmp(&a.header, &b.header, sizeof(a.header)) == 0 && a.moves == b.moves,
            "Mixed tile fast path changed owner, allocator, gap, budget, or Move state");
    }
    return {{"serial", serial.frames}, {"optimized", optimized.frames},
        {"identicalFrames", serial.checkpoints.size()}};
}

class StreamBLASAllocatorTest final : public RHITest {
public:
    StreamBLASAllocatorTest()
    {
        type = RHITestType::Rendering;
        name = "stream_blas_gpu_allocator";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::atomic_uint validationErrors = 0;
        auto created = createDevice({.applicationName = "Persistent BLAS allocation",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}});
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Bindless unavailable")
                : RHITestResult::fail(resultToString(created));
        }
        auto device = std::move(*created);
        std::string phase = "exact sizes and stationary frames";
        try {
            AllocatorScenario boundary(*device, 16384, 0);
            auto owners = boundary.inputs();
            owners[0].selectedClusterCount = 1024;
            owners[kStableOwner].selectedClusterCount = 1025;
            owners[0].flags = owners[kStableOwner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            boundary.encoded[0] = 1001; boundary.encoded[kStableOwner] = 1257;
            auto value = boundary.frame(owners);
            require(value.header.blasBuildCount == 2 && value.moves[0] == 2 &&
                value.owners[0].storageCapacity == 1024 && value.owners[kStableOwner].storageCapacity == 1280,
                "1024/1025 cluster boundary ignored GPU actual sizes");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = boundary.frame(boundary.inputs());
                require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 2 &&
                    value.moves[0] == 0 && value.moves[1] == 0, "Stationary owners rebuilt or moved");
            }

            phase = "grow and shrink";
            for (uint32_t frame = 0; frame < 32; ++frame) {
                owners = boundary.inputs(); owners[0].flags |= kDirty;
                owners[0].selectedClusterCount = frame % 2 ? 1024 : 1025;
                boundary.encoded[0] = frame % 2 ? 1001 : 1513;
                const uint32_t unchangedOffset = value.owners[kStableOwner].storageOffset;
                value = boundary.frame(owners);
                require(value.header.blasBuildCount == 1 && value.header.reusedInstances == 1 &&
                    value.owners[kStableOwner].storageOffset == unchangedOffset, "Local grow/shrink rebuilt or moved an unchanged owner");
            }

            phase = "cancelled recording and retry";
            const auto beforeCancel = boundary.snapshot();
            owners = boundary.inputs(); owners[kStableOwner].flags |= kDirty;
            boundary.encoded[kStableOwner] = 2023;
            const auto cancelled = boundary.frame(owners, true);
            require(std::memcmp(cancelled.allocations.data(), beforeCancel.allocations.data(), sizeof(Allocations)) == 0 &&
                std::memcmp(cancelled.state.data(), beforeCancel.state.data(), sizeof(State)) == 0,
                "Discarded command recording mutated persistent GPU allocation state");
            value = boundary.frame(owners);
            require(value.header.blasBuildCount == 1 && value.owners[kStableOwner].storageCapacity == 2048,
                "Cancelled dirty owner did not rebuild correctly on retry");

            phase = "stable rejection and free-space retry";
            AllocatorScenario rejection(*device, 2048, 0);
            owners = rejection.inputs();
            for (uint32_t i = 0; i < 3; ++i) {
                owners[i].selectedClusterCount = 1; owners[i].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                rejection.encoded[i] = 1024;
            }
            value = rejection.frame(owners);
            require(value.header.blasBuildCount == 3 && value.header.storageRejected == 1, "Tight storage did not reject one complete owner");
            uint32_t rejectedOwner = UINT32_MAX, releasedOwner = UINT32_MAX;
            for (uint32_t i = 0; i < 3; ++i) {
                if (value.owners[i].cachedValid == 2) { rejectedOwner = i; }
                else if (releasedOwner == UINT32_MAX) { releasedOwner = i; }
            }
            require(rejectedOwner != UINT32_MAX && releasedOwner != UINT32_MAX, "Tight storage fixture lacks live/rejected owners");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = rejection.frame(rejection.inputs());
                require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 2 &&
                    value.header.storageRejected == 1 && value.state[kBLASAllocatorSuppressedBuildsWord] == 1,
                    "Static over-budget owner caused a rebuild loop");
            }
            owners = rejection.inputs(); owners[releasedOwner] = {};
            value = rejection.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.storageRejected == 0 && value.owners[rejectedOwner].cachedValid == 1,
                "Freed capacity did not retry the rejected owner");

            phase = "fragmentation and bounded clean moves";
            AllocatorScenario holes(*device, 4096, 512);
            owners = holes.inputs();
            for (uint32_t i = 0; i < 8; ++i) {
                owners[i].selectedClusterCount = 1; owners[i].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                holes.encoded[i] = 512;
            }
            value = holes.frame(owners);
            require(value.header.blasBuildCount == 8 && value.state[kBLASAllocatorFreeBytesWord] == 0, "Fragmentation fixture did not fill arena");
            std::array<uint32_t, 8> byOffset{};
            std::iota(byOffset.begin(), byOffset.end(), 0);
            std::sort(byOffset.begin(), byOffset.end(), [&](uint32_t a, uint32_t b) {
                return value.owners[a].storageOffset < value.owners[b].storageOffset;
            });
            owners = holes.inputs();
            for (uint32_t i : {1u, 3u, 5u, 7u}) { owners[byOffset[i]] = {}; }
            owners[8].selectedClusterCount = 2; owners[8].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            holes.encoded[8] = 1024;
            value = holes.frame(owners, false, true);
            require(value.header.storageRejected == 1 && value.moves[1] == 1,
                "Fragmented arena did not reject the large owner and start one bounded clean move");
            uint32_t relocationFrames = value.moves[1] != 0;
            bool recovered = false;
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = holes.frame(holes.inputs());
                relocationFrames += value.moves[1] != 0;
                require(value.header.blasBuildCount <= 1 && value.moves[1] <= 1,
                    "Incremental defragmentation rebuilt clean owners or exceeded one-move budget");
                recovered |= value.owners[8].cachedValid == 1;
            }
            require(recovered && relocationFrames != 0 && value.header.blasBuildCount == 0 && value.moves[1] == 0,
                "Fragmentation never converged to a live large owner and stationary zero work");

            phase = "stationary holes without allocation pressure";
            AllocatorScenario noPressure(*device, 4096, 512);
            owners = noPressure.inputs();
            for (uint32_t owner = 0; owner < 8; ++owner) {
                owners[owner].selectedClusterCount = 1;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                noPressure.encoded[owner] = 512;
            }
            value = noPressure.frame(owners);
            std::iota(byOffset.begin(), byOffset.end(), 0);
            std::sort(byOffset.begin(), byOffset.end(), [&](uint32_t a, uint32_t b) {
                return value.owners[a].storageOffset < value.owners[b].storageOffset;
            });
            owners = noPressure.inputs();
            for (uint32_t i : {1u, 3u, 5u, 7u}) { owners[byOffset[i]] = {}; }
            value = noPressure.frame(owners);
            require(value.header.blasBuildCount == 0 && value.moves[1] == 0 &&
                value.state[kBLASAllocatorFreeBytesWord] == 2048 &&
                value.state[kBLASAllocatorLargestFreeWord] == 512,
                "Unpressured holes triggered unnecessary clean compaction");
            const auto stationaryAllocations = value.allocations;
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = noPressure.frame(noPressure.inputs());
                require(value.header.blasBuildCount == 0 && value.moves[0] == 0 && value.moves[1] == 0 &&
                    value.state[kBLASAllocatorPressureTargetWord] == 0 &&
                    std::memcmp(value.allocations.data(), stationaryAllocations.data(), sizeof(Allocations)) == 0,
                    "Stationary fragmented storage without pressure rebuilt, moved or changed allocations");
            }

            phase = "persistent pressure survives a scan with no relocation";
            AllocatorScenario deferredPressure(*device, 4096, 512, 1);
            owners = deferredPressure.inputs();
            // Owner zero remains empty. The first pressured scan must miss all
            // live owners, leaving the rejected cut's epoch unchanged.
            for (uint32_t owner = 1; owner <= 8; ++owner) {
                owners[owner].selectedClusterCount = 1;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                deferredPressure.encoded[owner] = 512;
            }
            value = deferredPressure.frame(owners);
            std::iota(byOffset.begin(), byOffset.end(), 1);
            std::sort(byOffset.begin(), byOffset.end(), [&](uint32_t a, uint32_t b) {
                return value.owners[a].storageOffset < value.owners[b].storageOffset;
            });
            owners = deferredPressure.inputs();
            for (uint32_t i : {1u, 3u, 5u, 7u}) { owners[byOffset[i]] = {}; }
            owners[9].selectedClusterCount = 2;
            owners[9].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            deferredPressure.encoded[9] = 1024;
            value = deferredPressure.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.storageRejected == 1 && value.moves[1] == 0 &&
                value.state[kBLASAllocatorPressureTargetWord] == 1024 &&
                value.state[kBLASAllocatorEpochWord] == value.allocations[9].rejectedEpoch,
                "The first pressured scan failed to leave a persistent target without advancing the rejection epoch");
            bool movedWhileSuppressed = false;
            recovered = false;
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = deferredPressure.frame(deferredPressure.inputs());
                require(value.header.blasBuildCount <= 1 && value.moves[1] <= 1,
                    "Deferred pressure recovery exceeded incremental build or relocation bounds");
                if (frame == 0) {
                    require(value.header.blasBuildCount == 0 && value.state[kBLASAllocatorSuppressedBuildsWord] == 1,
                        "The unchanged rejected cut was rebuilt instead of suppressed after the empty scan");
                }
                movedWhileSuppressed |= value.moves[1] != 0 && value.header.blasBuildCount == 0 &&
                    value.state[kBLASAllocatorSuppressedBuildsWord] == 1;
                recovered |= value.owners[9].cachedValid == 1;
            }
            require(movedWhileSuppressed && recovered && value.header.blasBuildCount == 0 && value.moves[1] == 0 &&
                value.state[kBLASAllocatorPressureTargetWord] == 0,
                "Negative build caching lost pending pressure and prevented bounded compaction from converging");

            phase = "scratch-assisted compaction across undersized holes";
            AllocatorScenario smallHoles(*device, 4096, 768);
            // Establish [256,768] four times in address order without relying
            // on cross-lane atomic packing order. After releasing every small
            // allocation, no hole can hold even the smallest live BLAS.
            for (uint32_t owner = 0; owner < 8; ++owner) {
                owners = smallHoles.inputs();
                owners[owner].selectedClusterCount = 1;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                smallHoles.encoded[owner] = owner % 2 ? 768 : 256;
                value = smallHoles.frame(owners);
                require(value.header.blasBuildCount == 1 && value.header.storageRejected == 0,
                    "Undersized-hole fixture failed to establish the alternating physical layout");
            }
            require(value.state[kBLASAllocatorFreeBytesWord] == 0, "Undersized-hole fixture did not fill storage");
            owners = smallHoles.inputs();
            for (uint32_t owner : {0u, 2u, 4u, 6u}) { owners[owner] = {}; }
            owners[8].selectedClusterCount = 2;
            owners[8].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            smallHoles.encoded[8] = 1024;
            value = smallHoles.frame(owners, false, true);
            require(value.header.storageRejected == 1 && value.moves[1] == 1 &&
                value.state[kBLASAllocatorRelocatedBytesWord] == 768,
                "No-fitting-hole allocator failed to stage one clean BLAS within the logical copy budget");
            recovered = false;
            uint32_t suppressedAfterRelocationFrames = 0;
            uint32_t recoveryBuilds = 0;
            for (uint32_t frame = 0; frame < 32; ++frame) {
                const bool rejected = value.owners[8].cachedValid == 2;
                const bool stillTooFragmented = value.state[kBLASAllocatorLargestFreeWord] < smallHoles.encoded[8];
                const bool epochChanged = value.state[kBLASAllocatorEpochWord] != value.allocations[8].rejectedEpoch;
                value = smallHoles.frame(smallHoles.inputs());
                require(value.header.blasBuildCount <= 1 && value.moves[1] <= 1,
                    "Scratch-assisted compaction rebuilt unrelated owners or copied multiple clean BLAS over budget");
                if (rejected && stillTooFragmented) {
                    require(value.header.blasBuildCount == 0 && value.moves[0] == 0 &&
                        value.state[kBLASAllocatorSuppressedBuildsWord] == 1 &&
                        value.state[kBLASAllocatorFailedAllocationsWord] == 0,
                        "A relocation epoch change rebuilt an unchanged rejected cut before any contiguous gap could fit it");
                    suppressedAfterRelocationFrames += epochChanged;
                }
                recoveryBuilds += value.header.blasBuildCount;
                recovered |= value.owners[8].cachedValid == 1;
            }
            require(recovered && value.header.blasBuildCount == 0 && value.moves[1] == 0 &&
                value.state[kBLASAllocatorFreeBytesWord] == 0,
                "Undersized holes did not converge to the exact-fit large allocation and zero stationary work");
            require(suppressedAfterRelocationFrames >= 2 && recoveryBuilds == 1,
                "Fragmented rejection did not survive multiple relocation epochs and retry exactly once when a fitting gap appeared");

            phase = "priority dirty-only reference and build budgets";
            PriorityImportance priorities{};
            for (auto& priority : priorities) { priority[1] = 0x7f7fffffu; }
            for (const uint32_t owner : {0u, 64u, 136u}) {
                priorities[owner] = {std::bit_cast<uint32_t>(float(owner + 1)) + 1,
                    std::bit_cast<uint32_t>(float(kOwners - owner)), std::bit_cast<uint32_t>(.25f), 1, 1, 0};
            }
            priorities[0][3] = 4; priorities[0][5] = 3;
            nlohmann::json dirtyBudgetFrames = nlohmann::json::array();
            for (const bool referenceLimited : {true, false}) {
                AllocatorScenario budget(*device, 4096, 0);
                const uint32_t referenceCapacity = referenceLimited ? 4u : 64u;
                const uint32_t buildCapacity = referenceLimited ? kOwners : 1u;
                budget.priority(priorities, referenceCapacity, buildCapacity);
                owners = budget.inputs();
                for (const uint32_t owner : {0u, 64u, 136u}) {
                    owners[owner].selectedClusterCount = 4;
                    owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                    budget.encoded[owner] = 512;
                }
                for (const uint32_t expected : {136u, 64u, 0u}) {
                    value = budget.frame(owners);
                    require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                        value.owners[expected].cachedValid == 1 && value.owners[expected].clusterReferenceOffset == 0,
                        "Priority did not admit the highest dirty owner or charged a clean cached owner against the build/reference budget");
                    require(value.header.blasBuildCount <= buildCapacity && value.header.clusterReferenceCount <= referenceCapacity,
                        "Dirty priority work exceeded its reference/build capacity");
                    owners = budget.inputs();
                }
                for (uint32_t frame = 0; frame < 32; ++frame) {
                    value = budget.frame(budget.inputs());
                    require(value.header.blasBuildCount == 0 && value.header.clusterReferenceCount == 0 &&
                        value.header.reusedInstances == 3 && value.header.liveClusterReferences == 12 &&
                        value.header.referenceBudgetRejected == 0 && value.header.buildBudgetRejected == 0,
                        "Clean priority owners consumed dirty-only budgets or failed to converge to zero builds");
                }
                dirtyBudgetFrames.push_back({{"referenceLimited", referenceLimited}, {"frames", budget.frames}});
            }

            phase = "known-size priority-only storage replacement";
            AllocatorScenario rankedStorage(*device, 1024, 0);
            rankedStorage.priority(priorities, 64, kOwners);
            owners = rankedStorage.inputs();
            for (const uint32_t owner : {0u, 64u, 136u}) {
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                rankedStorage.encoded[owner] = 512;
            }
            value = rankedStorage.frame(owners);
            require(value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 1 &&
                value.owners[0].cachedValid == 2 && value.header.storageRejected == 1,
                "Actual-size storage did not preserve the highest-priority owners");
            require(value.state[kBLASAllocatorUnknownBuildCountWord] == 3 &&
                value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 1 &&
                value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0 &&
                value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 0,
                "Initial unknown-size budget rejection was confused with fragmentation or a known waiting claim");
            require(value.state[kBLASAllocatorRequestedVisibleWord] == 3 &&
                value.state[kBLASAllocatorRequestedOffscreenWord] == 3 &&
                value.state[kBLASAllocatorRejectedVisibleWord] == 1 &&
                value.state[kBLASAllocatorRejectedOffscreenWord] == 3 &&
                value.state[kBLASAllocatorRejectedMaxErrorWord] == priorities[0][0],
                "Priority feedback did not count all shared visible/offscreen consumers of the rejected owner");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = rankedStorage.frame(rankedStorage.inputs());
                require(value.header.blasBuildCount == 0 && value.header.storageRejected == 1 &&
                    value.state[kBLASAllocatorPriorityEvictedWord] == 0 &&
                    value.state[kBLASAllocatorUnknownBuildCountWord] == 0 &&
                    value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 &&
                    value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0 &&
                    value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 0,
                    "Stable ranked storage rebuilt its negative cache or oscillated clean admission");
            }
            const uint32_t survivorOffset = value.owners[136].storageOffset;
            priorities[0][0] = std::bit_cast<uint32_t>(1000.f) + 1;
            rankedStorage.priority(priorities, 64, kOwners);
            const auto beforePriorityCancel = rankedStorage.snapshot();
            owners = rankedStorage.inputs();
            const auto cancelledPriority = rankedStorage.frame(owners, true);
            require(std::memcmp(cancelledPriority.allocations.data(), beforePriorityCancel.allocations.data(), sizeof(Allocations)) == 0 &&
                std::memcmp(cancelledPriority.state.data(), beforePriorityCancel.state.data(), sizeof(State)) == 0,
                "Cancelled priority admission changed persistent allocations or negative-cache state");
            value = rankedStorage.frame(owners);
            require(value.header.blasBuildCount == 1 && value.owners[0].cachedValid == 1 &&
                value.owners[136].cachedValid == 1 && value.owners[136].storageOffset == survivorOffset &&
                value.owners[64].cachedValid == 2 && value.state[kBLASAllocatorPriorityEvictedWord] == 1,
                "A priority-only change did not replace exactly the lower-priority clean owner while preserving its survivor");
            require(value.state[kBLASAllocatorUnknownBuildCountWord] == 0 &&
                value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 &&
                value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0 &&
                value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 0,
                "Priority-only rebuilding of a measured negative entry was counted as unknown work or a failed allocation");
            require(value.state[kBLASAllocatorRejectedVisibleWord] == 1 &&
                value.state[kBLASAllocatorRejectedOffscreenWord] == 0 &&
                value.state[kBLASAllocatorAdmittedOffscreenWord] == 3 &&
                value.state[kBLASAllocatorRejectedMaxErrorWord] == priorities[64][0],
                "Priority replacement did not publish the new rejected owner and admitted offscreen consumers");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = rankedStorage.frame(rankedStorage.inputs());
                require(value.header.blasBuildCount == 0 && value.header.storageRejected == 1 &&
                    value.moves[0] == 0 && value.moves[1] == 0 && value.state[kBLASAllocatorPriorityEvictedWord] == 0 &&
                    value.owners[0].cachedValid == 1 && value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 2,
                    "Priority-only storage replacement did not settle for 32 consecutive frames");
            }

            phase = "fragmented high-priority wait must not starve fitting dirty work";
            AllocatorScenario fragmentedPriority(*device, 4096, 0);
            PriorityImportance waitingPriorities{};
            for (auto& priority : waitingPriorities) { priority[1] = 0x7f7fffffu; }
            owners = fragmentedPriority.inputs();
            for (uint32_t owner = 0; owner < 8; ++owner) {
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                fragmentedPriority.encoded[owner] = 512;
                waitingPriorities[owner] = {std::bit_cast<uint32_t>(float(100 + owner)) + 1,
                    std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            }
            fragmentedPriority.priority(waitingPriorities, 64, kOwners);
            value = fragmentedPriority.frame(owners);
            require(value.state[kBLASAllocatorFreeBytesWord] == 0, "Priority starvation fixture did not fill its arena");
            std::iota(byOffset.begin(), byOffset.end(), 0);
            std::sort(byOffset.begin(), byOffset.end(), [&](uint32_t a, uint32_t b) {
                return value.owners[a].storageOffset < value.owners[b].storageOffset;
            });
            owners = fragmentedPriority.inputs();
            for (const uint32_t position : {1u, 3u, 5u, 7u}) {
                const uint32_t owner = byOffset[position];
                owners[owner] = {};
                waitingPriorities[owner] = {0, 0x7f7fffffu, 0, 0, 0, 0};
            }
            owners[136].selectedClusterCount = 4;
            owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            fragmentedPriority.encoded[136] = 1024;
            waitingPriorities[136] = {std::bit_cast<uint32_t>(1000.f) + 1,
                std::bit_cast<uint32_t>(.1f), 0, 1, 1, 0};
            fragmentedPriority.priority(waitingPriorities, 4, 1);
            value = fragmentedPriority.frame(owners);
            require(value.header.blasBuildCount == 1 && value.owners[136].cachedValid == 2 &&
                value.state[kBLASAllocatorFreeBytesWord] == 2048 && value.state[kBLASAllocatorLargestFreeWord] == 512,
                "Priority starvation fixture lacks a known high-priority request waiting on fragmented space");
            require(value.state[kBLASAllocatorUnknownBuildCountWord] == 1 &&
                value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 1 &&
                value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 &&
                value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 0,
                "An initially unknown fitting-budget request failed physically without the matching fragmentation counter");
            // The waiting 1024-byte owner retains its storage claim, but must
            // not reserve the sole build slot or four-reference work budget.
            // Four lower-ranked 256-byte owners can fit; a fifth would invade
            // that claim and must remain rejected after its actual size is known.
            for (uint32_t index = 0; index < 5; ++index) {
                const uint32_t owner = 135 - index;
                owners = fragmentedPriority.inputs();
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                fragmentedPriority.encoded[owner] = 256;
                waitingPriorities[owner] = {std::bit_cast<uint32_t>(float(50 - index)) + 1,
                    std::bit_cast<uint32_t>(10.f), 0, 1, 1, 0};
                fragmentedPriority.priority(waitingPriorities, 4, 1);
                value = fragmentedPriority.frame(owners);
                require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                    value.owners[136].cachedValid == 2 && value.moves[1] == 0 &&
                    value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                    "A non-fitting high-priority negative entry starved lower-ranked fitting dirty work");
                require(value.owners[owner].cachedValid == (index < 4 ? 1u : 2u) &&
                    value.state[kBLASAllocatorAllocatedBytesWord] == 2048 + std::min(index + 1, 4u) * 256 &&
                    value.state[kBLASAllocatorPressureTargetWord] == 1024,
                    "Lower-ranked dirty work consumed the waiting owner's persistent storage claim");
                require(value.state[kBLASAllocatorUnknownBuildCountWord] == 1 &&
                    value.state[kBLASAllocatorWaitingClaimBytesWord] == 1024 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 1024 &&
                    value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0 &&
                    value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == (index == 4 ? 1u : 0u),
                    "Waiting claims, newly measured builds and post-build budget failures were not distinguished");
            }
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = fragmentedPriority.frame(fragmentedPriority.inputs());
                require(value.header.blasBuildCount == 0 && value.header.clusterReferenceCount == 0 &&
                    value.header.storageRejected == 2 && value.state[kBLASAllocatorSuppressedBuildsWord] == 2 &&
                    value.state[kBLASAllocatorAllocatedBytesWord] == 3072 &&
                    value.state[kBLASAllocatorFreeBytesWord] == 1024 && value.state[kBLASAllocatorLargestFreeWord] == 512 &&
                    value.state[kBLASAllocatorPressureTargetWord] == 1024 && value.moves[0] == 0 && value.moves[1] == 0,
                    "Disabled compaction failed to retain the high-priority storage claim and converge without rebuilds");
                require(value.state[kBLASAllocatorUnknownBuildCountWord] == 0 &&
                    value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 &&
                    value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0 &&
                    value.state[kBLASAllocatorWaitingClaimBytesWord] == 1024 && value.state[kBLASAllocatorMaxWaitingBytesWord] == 1024,
                    "Stationary waiting state retained stale build/failure counters or lost its reserved claim");
                for (const uint32_t owner : {132u, 133u, 134u, 135u}) {
                    require(value.owners[owner].cachedValid == 1, "Fitting lower-priority work was lost after admission");
                }
            }

            phase = "projected dirty overestimate delays exact negative retry without eviction";
            AllocatorScenario projectedShrink(*device, 1536, 0);
            PriorityImportance projectedPriorities{};
            for (auto& priority : projectedPriorities) { priority[1] = 0x7f7fffffu; }
            for (const uint32_t owner : {0u, 64u, 136u}) {
                projectedPriorities[owner] = {std::bit_cast<uint32_t>(float(owner + 1)) + 1,
                    std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            }
            projectedShrink.priority(projectedPriorities, 64, kOwners);
            owners = projectedShrink.inputs();
            for (const uint32_t owner : {0u, 64u, 136u}) {
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            }
            projectedShrink.encoded[136] = 1280;
            projectedShrink.encoded[64] = 256;
            projectedShrink.encoded[0] = 512;
            value = projectedShrink.frame(owners);
            require(value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 1 &&
                value.owners[0].cachedValid == 2 && value.allocations[0].encodedBytes == 512 &&
                value.allocations[136].pendingBytes == 1280,
                "Projected shrink fixture lacks resident size history and an exact negative cache");
            const auto exactNegative = value.allocations[0];
            const uint32_t projectedSurvivorOffset = value.owners[64].storageOffset;
            projectedPriorities[0][0] = std::bit_cast<uint32_t>(100.f) + 1;
            projectedShrink.priority(projectedPriorities, 64, kOwners);
            projectedShrink.encoded[136] = 256;
            owners = projectedShrink.inputs();
            owners[136].flags |= kDirty;
            const auto beforeProjectedCancel = projectedShrink.snapshot();
            const auto cancelledProjected = projectedShrink.frame(owners, true);
            require(std::memcmp(cancelledProjected.allocations.data(), beforeProjectedCancel.allocations.data(), sizeof(Allocations)) == 0 &&
                std::memcmp(cancelledProjected.state.data(), beforeProjectedCancel.state.data(), sizeof(State)) == 0,
                "Cancelled projected admission changed actual-size history or negative-cache state");
            value = projectedShrink.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                value.state[kBLASAllocatorUnknownBuildCountWord] == 1 && value.owners[136].cachedValid == 1 &&
                value.allocations[136].encodedBytes == 256 && value.allocations[136].pendingBytes == 256 &&
                value.owners[0].cachedValid == 2 && value.allocations[0].encodedBytes == exactNegative.encodedBytes &&
                value.allocations[0].pendingBytes == exactNegative.pendingBytes &&
                value.allocations[0].rejectedEpoch == exactNegative.rejectedEpoch &&
                value.owners[64].cachedValid == 1 && value.owners[64].storageOffset == projectedSurvivorOffset &&
                value.state[kBLASAllocatorAllocatedBytesWord] == 512 && value.state[kBLASAllocatorPriorityEvictedWord] == 0 &&
                value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorPressureTargetWord] == 0 &&
                value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 && value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0,
                "An old-size overestimate rebuilt or claimed storage for a deferred negative entry, evicted a clean owner, or replaced exact history");
            require(value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 1 &&
                value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 1280 &&
                value.header.storageRejected == 0 && value.state[kBLASAllocatorSuppressedBuildsWord] == 1 &&
                value.header.overflowCount != 0,
                "Dirty shrink did not separately account its old-size prediction and deferred exact retry");
            IdleCacheAdmissionProbe idleCacheProbe(projectedShrink.fixture);
            idleCacheProbe.check(value.header, false, false);
            idleCacheProbe.check(value.header, true, false);
            value = projectedShrink.frame(projectedShrink.inputs());
            require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                value.state[kBLASAllocatorUnknownBuildCountWord] == 0 && value.owners[0].cachedValid == 1 &&
                value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 1 &&
                value.state[kBLASAllocatorAllocatedBytesWord] == 1024 && value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                "A deferred exact negative entry did not recover on the next stationary frame after dirty shrink");
            require(value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0 &&
                value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 0,
                "Known retry incorrectly consumed dirty prediction or retained the previous frame's deferral");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = projectedShrink.frame(projectedShrink.inputs());
                require(value.header.blasBuildCount == 0 && value.header.clusterReferenceCount == 0 &&
                    value.header.storageRejected == 0 && value.header.reusedInstances == 3 &&
                    value.state[kBLASAllocatorAllocatedBytesWord] == 1024 && value.moves[0] == 0 && value.moves[1] == 0,
                    "Projected shrink recovery failed to settle for 32 stationary frames");
            }

            phase = "projected dirty underestimate still uses native actual-size admission";
            AllocatorScenario projectedGrow(*device, 1024, 0);
            projectedPriorities[0][0] = std::bit_cast<uint32_t>(1.f) + 1;
            projectedGrow.priority(projectedPriorities, 64, kOwners);
            owners = projectedGrow.inputs();
            for (const uint32_t owner : {0u, 64u, 136u}) {
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            }
            projectedGrow.encoded[136] = 256;
            projectedGrow.encoded[64] = 768;
            projectedGrow.encoded[0] = 512;
            value = projectedGrow.frame(owners);
            require(value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 1 && value.owners[0].cachedValid == 2,
                "Projected growth fixture lacks its clean working set and measured rejected owner");
            projectedPriorities[0][0] = std::bit_cast<uint32_t>(100.f) + 1;
            projectedGrow.priority(projectedPriorities, 64, kOwners);
            projectedGrow.encoded[136] = 1024;
            owners = projectedGrow.inputs();
            owners[136].flags |= kDirty;
            value = projectedGrow.frame(owners);
            require(value.header.blasBuildCount == 2 && value.header.clusterReferenceCount == 8 &&
                value.state[kBLASAllocatorUnknownBuildCountWord] == 1 && value.owners[136].cachedValid == 1 &&
                value.allocations[136].encodedBytes == 1024 && value.allocations[136].pendingBytes == 1024 &&
                value.owners[0].cachedValid == 2 && value.allocations[0].encodedBytes == 512 && value.allocations[0].pendingBytes == 512 &&
                value.owners[64].cachedValid == 2 && value.state[kBLASAllocatorAllocatedBytesWord] == 1024 &&
                value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 1 && value.state[kBLASAllocatorPhysicalAllocationFailedWord] == 0,
                "An underestimated dirty cut bypassed native actual-size budget rejection or poisoned the measured negative cache");
            require(value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0 &&
                value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 256 && value.header.storageRejected != 0,
                "Dirty growth replaced its separate old-size prediction with actual allocation accounting");
            idleCacheProbe.check(value.header, false, true);
            idleCacheProbe.check(value.header, true, true);
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = projectedGrow.frame(projectedGrow.inputs());
                require(value.header.blasBuildCount == 0 && value.header.clusterReferenceCount == 0 &&
                    value.header.storageRejected == 2 && value.state[kBLASAllocatorSuppressedBuildsWord] == 2 &&
                    value.owners[136].cachedValid == 1 && value.owners[0].cachedValid == 2 && value.owners[64].cachedValid == 2 &&
                    value.state[kBLASAllocatorPostBuildBudgetRejectedWord] == 0 && value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                    "Actual-size rejection after underestimated growth did not converge to a stable negative cache");
            }

            phase = "mixed actual rejection and projected deferral retain next-frame idle-cache pressure";
            AllocatorScenario projectedMixed(*device, 1536, 0);
            auto mixedPriorities = projectedPriorities;
            mixedPriorities[0][0] = std::bit_cast<uint32_t>(1.f) + 1;
            mixedPriorities[1] = {std::bit_cast<uint32_t>(.5f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            projectedMixed.priority(mixedPriorities, 64, kOwners);
            owners = projectedMixed.inputs();
            for (const uint32_t owner : {0u, 1u, 64u, 136u}) {
                owners[owner].selectedClusterCount = 4;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            }
            projectedMixed.encoded[136] = 1280;
            projectedMixed.encoded[64] = 256;
            projectedMixed.encoded[0] = 512;
            projectedMixed.encoded[1] = 2048;
            value = projectedMixed.frame(owners);
            require(value.owners[136].cachedValid == 1 && value.owners[64].cachedValid == 1 &&
                value.owners[0].cachedValid == 2 && value.owners[1].cachedValid == 2,
                "Mixed pressure fixture did not establish two independently measured negative entries");
            mixedPriorities[0][0] = std::bit_cast<uint32_t>(100.f) + 1;
            projectedMixed.priority(mixedPriorities, 64, kOwners);
            projectedMixed.encoded[136] = 256;
            owners = projectedMixed.inputs();
            owners[136].flags |= kDirty;
            value = projectedMixed.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.storageRejected == 1 &&
                value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 1 &&
                value.state[kBLASAllocatorSuppressedBuildsWord] == 2 && value.header.overflowCount == 2 &&
                value.owners[64].cachedValid == 1 && value.owners[0].cachedValid == 2 && value.owners[1].cachedValid == 2,
                "Subtracting projected deferral hid real storage pressure or lost deferred-consumer diagnostics");
            idleCacheProbe.check(value.header, false, true);
            idleCacheProbe.check(value.header, true, true);

            phase = "work-budget denied dirty history must not consume projected retry capacity";
            AllocatorScenario projectedWorkBudget(*device, 1024, 0);
            PriorityImportance workPriorities{};
            for (auto& priority : workPriorities) { priority[1] = 0x7f7fffffu; }
            workPriorities[136] = {std::bit_cast<uint32_t>(100.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            workPriorities[0] = {std::bit_cast<uint32_t>(1.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            projectedWorkBudget.priority(workPriorities, 64, kOwners);
            owners = projectedWorkBudget.inputs();
            owners[136].selectedClusterCount = 8;
            owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            owners[0].selectedClusterCount = 4;
            owners[0].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            projectedWorkBudget.encoded[136] = 1024;
            projectedWorkBudget.encoded[0] = 512;
            value = projectedWorkBudget.frame(owners);
            require(value.owners[136].cachedValid == 1 && value.owners[0].cachedValid == 2,
                "Projected work-budget fixture lacks measured resident and negative sizes");
            projectedWorkBudget.priority(workPriorities, 4, 1);
            owners = projectedWorkBudget.inputs();
            owners[136].flags |= kDirty;
            value = projectedWorkBudget.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                value.header.referenceBudgetRejected == 1 && value.header.buildBudgetRejected == 0 &&
                value.state[kBLASAllocatorUnknownBuildCountWord] == 0 && value.owners[0].cachedValid == 1 &&
                value.owners[136].cachedValid == 0 && value.allocations[136].encodedBytes == 0 &&
                value.allocations[136].pendingBytes == 1024 && value.state[kBLASAllocatorAllocatedBytesWord] == 512 &&
                value.state[kBLASAllocatorWaitingClaimBytesWord] == 0 && value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                "Reference-denied dirty work charged its old-size prediction, lost history, or starved a fitting exact retry");
            require(value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0 &&
                value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 0,
                "Work-budget rejection charged a dirty estimate despite submitting no dirty build");
            workPriorities[64] = {std::bit_cast<uint32_t>(1000.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            projectedWorkBudget.priority(workPriorities, 64, 1);
            owners = projectedWorkBudget.inputs();
            owners[64].selectedClusterCount = 4;
            owners[64].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            owners[136].selectedClusterCount = 4;
            owners[136].flags |= kDirty;
            projectedWorkBudget.encoded[64] = 256;
            projectedWorkBudget.encoded[136] = 256;
            value = projectedWorkBudget.frame(owners);
            require(value.header.blasBuildCount == 1 && value.header.clusterReferenceCount == 4 &&
                value.header.buildBudgetRejected == 1 && value.header.referenceBudgetRejected == 0 &&
                value.owners[64].cachedValid == 1 && value.owners[0].cachedValid == 1 && value.owners[136].cachedValid == 0 &&
                value.allocations[136].pendingBytes == 1024 && value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 0 &&
                value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0,
                "Build-slot denial consumed or erased dirty size history despite submitting only a new unmeasured owner");
            value = projectedWorkBudget.frame(projectedWorkBudget.inputs());
            require(value.header.blasBuildCount == 1 && value.owners[136].cachedValid == 1 &&
                value.allocations[136].encodedBytes == 256 && value.allocations[136].pendingBytes == 256 &&
                value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 1024 &&
                value.state[kBLASAllocatorAllocatedBytesWord] == 1024 && value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                "Budget-delayed dirty work lost its historical estimate or used prediction as actual allocation size");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = projectedWorkBudget.frame(projectedWorkBudget.inputs());
                require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 3 &&
                    value.state[kBLASAllocatorProjectedDirtyClaimBytesWord] == 0 &&
                    value.state[kBLASAllocatorProjectedRetryDeferredCountWord] == 0,
                    "Work-budget recovery did not settle or retained stale prediction counters");
            }

            phase = "known-negative tiles must preserve fitting tail backfill";
            nlohmann::json negativeTileFrames = nlohmann::json::array();
            for (const bool fittingTail : {false, true}) {
                AllocatorScenario negativeTile(*device, 256, 0);
                PriorityImportance tilePriorities{};
                for (auto& priority : tilePriorities) { priority[1] = 0x7f7fffffu; }
                owners = negativeTile.inputs();
                for (uint32_t owner = 0; owner < 64; ++owner) {
                    tilePriorities[owner] = {std::bit_cast<uint32_t>(float(owner + 1)) + 1,
                        std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
                    owners[owner].selectedClusterCount = 1;
                    owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                    negativeTile.encoded[owner] = fittingTail && owner == 0 ? 256 : 512;
                }
                tilePriorities[136] = {std::bit_cast<uint32_t>(1000.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
                owners[136].selectedClusterCount = 1;
                owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                negativeTile.encoded[136] = 256;
                negativeTile.priority(tilePriorities, kOwners, kOwners);
                value = negativeTile.frame(owners);
                require(value.owners[136].cachedValid == 1 && value.header.storageRejected == 64,
                    "Negative tile fixture did not produce 64 measured rejected entries");
                owners = negativeTile.inputs();
                owners[136] = {};
                tilePriorities[136] = {0, 0x7f7fffffu, 0, 0, 0, 0};
                negativeTile.priority(tilePriorities, kOwners, kOwners);
                value = negativeTile.frame(owners);
                require(value.header.blasBuildCount == (fittingTail ? 1u : 0u) &&
                    value.header.clusterReferenceCount == (fittingTail ? 1u : 0u) &&
                    value.header.storageRejected == (fittingTail ? 63u : 64u) &&
                    value.state[kBLASAllocatorUnknownBuildCountWord] == 0 &&
                    value.owners[0].cachedValid == (fittingTail ? 1u : 2u),
                    "Known-negative tile skip suppressed a fitting final-ranked owner or rebuilt a non-fitting entry");
                for (uint32_t frame = 0; frame < 32; ++frame) {
                    value = negativeTile.frame(negativeTile.inputs());
                    require(value.header.blasBuildCount == 0 && value.header.clusterReferenceCount == 0 &&
                        value.header.storageRejected == (fittingTail ? 63u : 64u) &&
                        value.state[kBLASAllocatorAllocatedBytesWord] == (fittingTail ? 256u : 0u) &&
                        value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                        "Known-negative tile backfill failed to settle without repeated builds or evictions");
                }
                negativeTileFrames.push_back({{"fittingTail", fittingTail}, {"frames", negativeTile.frames}});
            }

            phase = "clean resident tile exact boundary, actual-size overflow and inactive lanes";
            AllocatorScenario residentTiles(*device, 128 * 256, 0);
            PriorityImportance residentPriorities{};
            for (auto& priority : residentPriorities) { priority[1] = 0x7f7fffffu; }
            owners = residentTiles.inputs();
            for (uint32_t owner = 0; owner < 128; ++owner) {
                residentPriorities[owner] = {std::bit_cast<uint32_t>(float(owner + 1)) + 1,
                    std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
                owners[owner].selectedClusterCount = 1;
                owners[owner].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
                residentTiles.encoded[owner] = 256;
            }
            residentTiles.priority(residentPriorities, kOwners, kOwners);
            value = residentTiles.frame(owners);
            require(value.header.blasBuildCount == 128 && value.state[kBLASAllocatorFreeBytesWord] == 0,
                "Resident tile fixture did not exactly fill two clean owner blocks");
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = residentTiles.frame(residentTiles.inputs());
                require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 128 &&
                    value.header.storageRejected == 0 && value.state[kBLASAllocatorAllocatedBytesWord] == 128 * 256,
                    "An exactly fitting clean resident tile was rejected or rebuilt");
            }
            residentPriorities[136] = {std::bit_cast<uint32_t>(1000.f) + 1, std::bit_cast<uint32_t>(1.f), 0, 1, 1, 0};
            residentTiles.priority(residentPriorities, kOwners, kOwners);
            owners = residentTiles.inputs();
            owners[136].selectedClusterCount = 1;
            owners[136].flags = kMeshletStreamBLASInstanceDynamic | kDirty;
            residentTiles.encoded[136] = 512;
            value = residentTiles.frame(owners);
            require(value.header.blasBuildCount == 1 && value.owners[136].cachedValid == 1 &&
                value.owners[0].cachedValid == 2 && value.owners[1].cachedValid == 2 &&
                value.state[kBLASAllocatorPriorityEvictedWord] == 2 && value.state[kBLASAllocatorAllocatedBytesWord] == 128 * 256,
                "A clean tile exceeding the remaining actual-size budget skipped ranked eviction");
            for (uint32_t owner = 2; owner < 128; ++owner) {
                require(value.owners[owner].cachedValid == 1, "Actual-size tile admission evicted a higher-priority survivor");
            }
            owners = residentTiles.inputs();
            // Keep these owners' sort keys to leave inactive lanes inside both
            // resident tiles, and remove the two negative entries from demand.
            for (const uint32_t owner : {0u, 1u, 31u, 95u}) { owners[owner] = {}; }
            value = residentTiles.frame(owners);
            for (uint32_t frame = 0; frame < 32; ++frame) {
                value = residentTiles.frame(residentTiles.inputs());
                require(value.header.blasBuildCount == 0 && value.header.reusedInstances == 125 &&
                    value.header.storageRejected == 0 && value.state[kBLASAllocatorAllocatedBytesWord] == 126 * 256 &&
                    value.state[kBLASAllocatorPriorityEvictedWord] == 0,
                    "Inactive lanes changed resident-tile budget accounting or static-frame convergence");
            }

            auto admissionCacheFrames = runAdmissionCacheRegression(*device, phase);
            auto deterministicBaselineFrames = runDeterministicBaselineRegression(*device, phase);
            auto mixedPriorityTileFrames = runMixedPriorityTileRegression(*device, phase);

            require(validationErrors.load() == 0, "Vulkan validation errors during allocator regression");
            const auto directory = context.outputDirectory / "BLASAllocator";
            std::filesystem::create_directories(directory);
            std::ofstream report(directory / "Frames.json");
            report << nlohmann::json{{"boundary", boundary.frames}, {"rejection", rejection.frames},
                {"fragmentation", holes.frames}, {"undersizedHoles", smallHoles.frames},
                {"noPressure", noPressure.frames}, {"deferredPressure", deferredPressure.frames},
                {"suppressedAfterRelocationFrames", suppressedAfterRelocationFrames},
                {"fragmentedRecoveryBuilds", recoveryBuilds},
                {"dirtyPriorityBudgets", dirtyBudgetFrames}, {"rankedStorage", rankedStorage.frames},
                {"fragmentedPriorityWait", fragmentedPriority.frames},
                {"projectedDirtyShrink", projectedShrink.frames}, {"projectedDirtyGrow", projectedGrow.frames},
                {"projectedMixedPressure", projectedMixed.frames},
                {"nextFrameIdleCachePressure", "production phase 26 and legacy/priority admission consume actual allocator headers; projection preserves newly idle cache, actual and mixed pressure evict it"},
                {"projectedWorkBudget", projectedWorkBudget.frames},
                {"negativeTiles", negativeTileFrames}, {"residentTiles", residentTiles.frames},
                {"admissionCache", std::move(admissionCacheFrames)},
                {"deterministicBaseline", std::move(deterministicBaselineFrames)},
                {"mixedPriorityTiles", std::move(mixedPriorityTileFrames)},
                {"relocationAccounting", "logical bytes; native scratch round-trip copies twice this amount"},
                {"validationErrors", validationErrors.load()},
                {"sizeInput", "deterministic driver-size synthesis; native Move fixture covers real build outputs"}}.dump(2);
            return RHITestResult::pass("Actual sizes at 1024/1025 clusters; 32 static frames; 32 grow/shrink frames; cancelled recording/retry; stable tight-budget rejection/recovery; ordinary and undersized holes converge through scratch-assisted bounded clean Moves without rebuilds");
        } catch (const std::exception& error) {
            return RHITestResult::fail(phase + ": " + error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamBLASAllocatorTest);

} // namespace
} // namespace metallic::tests
