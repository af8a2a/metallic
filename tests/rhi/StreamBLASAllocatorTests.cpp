#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamBLASAllocator.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <atomic>
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
    Buffer &sizes, &sourceAddresses, &moveSources, &moveDestinations, &moveCounts, &parameters;
    MeshletStreamBLASAllocatorParameters params{};
    std::array<uint32_t, kOwners> encoded{};
    nlohmann::json frames = nlohmann::json::array();

    AllocatorScenario(Device& device, uint32_t storageBytes, uint32_t relocateBudget,
        uint32_t scanCount = kOwners) : fixture(device),
        state(fixture.buffer(sizeof(State))), gaps(fixture.buffer(sizeof(Gaps))),
        allocations(fixture.buffer(sizeof(Allocations))), owners(fixture.buffer(sizeof(Owners))),
        header(fixture.buffer(sizeof(MeshletStreamGPUBLASHeader))),
        buildInfos(fixture.buffer(kOwners * sizeof(MeshletStreamGPUBLASBuildInfo))),
        buildOwners(fixture.buffer(kOwners * 4)), sizes(fixture.buffer(kOwners * 4)),
        sourceAddresses(fixture.buffer(kOwners * 8)), moveSources(fixture.buffer(kOwners * 2 * 8)),
        moveDestinations(fixture.buffer(kOwners * 2 * 8)), moveCounts(fixture.buffer(8)),
        parameters(fixture.buffer(sizeof(params)))
    {
        params = {state.deviceAddress(), gaps.deviceAddress(), allocations.deviceAddress(), owners.deviceAddress(),
            header.deviceAddress(), buildInfos.deviceAddress(), buildOwners.deviceAddress(), sizes.deviceAddress(),
            sourceAddresses.deviceAddress(), moveSources.deviceAddress(), moveDestinations.deviceAddress(), moveCounts.deviceAddress(),
            0x100000000ull, 0x200000000ull, kOwners, kAlignment, storageBytes, relocateBudget, kNodes, scanCount, 1, 0,
            0x600000000ull};
        fixture.write(parameters, params);
        std::array<uint64_t, kOwners> sources{};
        for (uint32_t i = 0; i < kOwners; ++i) { sources[i] = 0x400000000ull + uint64_t(i) * 65536; }
        fixture.write(sourceAddresses, sources);
        fixture.begin(); dispatch(0); fixture.finish();
    }

    void dispatch(uint32_t phase)
    {
        const MeshletStreamBLASAllocatorPush push{parameters.deviceAddress(), phase, 0};
        checked(fixture.commands->bindExecution(fixture.pipeline->execution(), &push, sizeof(push)));
        checked(fixture.commands->dispatch((kOwners + 63) / 64)); fixture.barrier();
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

    Frame frame(const Owners& input, bool cancel = false, bool inspectPendingMoves = false)
    {
        fixture.write(owners, input);
        fixture.write(header, MeshletStreamGPUBLASHeader{});
        // Driver-size synthesis belongs only to this state-machine fixture.
        // Real build/allocate/Move ordering is exercised by the native fixture.
        fixture.begin(); dispatch(5); dispatch(1); dispatch(6);
        if (cancel) { dispatch(2); dispatch(8); dispatch(3); }
        fixture.finish(cancel);
        if (cancel) { return snapshot(); }
        const auto prepared = snapshot();
        const auto ids = fixture.read<std::array<uint32_t, kOwners>>(buildOwners);
        std::array<uint32_t, kOwners> bytes{};
        for (uint32_t i = 0; i < prepared.header.blasBuildCount; ++i) { bytes[i] = encoded[ids[i]]; }
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
        const auto output = snapshot();
        check(output);
        frames.push_back({{"builds", output.header.blasBuildCount}, {"repack", output.header.arenaRepack},
            {"encodedBytes", output.state[kBLASAllocatorEncodedBytesWord]},
            {"allocatedBytes", output.state[kBLASAllocatorAllocatedBytesWord]},
            {"largestFreeBytes", output.state[kBLASAllocatorLargestFreeWord]},
            {"freeBytes", output.state[kBLASAllocatorFreeBytesWord]},
            {"relocatedCount", output.state[kBLASAllocatorRelocatedCountWord]},
            {"relocatedBytes", output.state[kBLASAllocatorRelocatedBytesWord]},
            {"suppressedBuilds", output.state[kBLASAllocatorSuppressedBuildsWord]},
            {"pressureTargetBytes", output.state[kBLASAllocatorPressureTargetWord]},
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

            require(validationErrors.load() == 0, "Vulkan validation errors during allocator regression");
            const auto directory = context.outputDirectory / "BLASAllocator";
            std::filesystem::create_directories(directory);
            std::ofstream report(directory / "Frames.json");
            report << nlohmann::json{{"boundary", boundary.frames}, {"rejection", rejection.frames},
                {"fragmentation", holes.frames}, {"undersizedHoles", smallHoles.frames},
                {"noPressure", noPressure.frames}, {"deferredPressure", deferredPressure.frames},
                {"suppressedAfterRelocationFrames", suppressedAfterRelocationFrames},
                {"fragmentedRecoveryBuilds", recoveryBuilds},
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
