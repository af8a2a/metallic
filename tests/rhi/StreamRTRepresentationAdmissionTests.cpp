#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;

class StreamRTRepresentationAdmissionTest final : public RHITest {
public:
    StreamRTRepresentationAdmissionTest()
    {
        type = RHITestType::Rendering;
        name = "stream_rt_representation_parallel_admission";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto require = [](bool condition, const char* message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Parallel RT representation admission",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Bindless unavailable")
                : RHITestResult::fail("Device creation failed");
        }
        try {
            constexpr uint32_t kPrimitives = 137, kBlocks = (kPrimitives + 63) / 64, kCapacity = 1024;
            constexpr uint32_t kOwners = kPrimitives * 3, kOwnerBlocks = (kOwners + 63) / 64;
            constexpr uint32_t kImportanceBase = kPrimitives * 16;
            constexpr uint32_t kStateWords = kImportanceBase + kOwners * 12 + kMeshletStreamRTPriorityControlWords;
            enum Input { Params, Primitives, State, Header, Active, Representations, InputCount };
            const uint64_t sizes[] = {sizeof(MeshletStreamGPUParams), kPrimitives * sizeof(MeshletStreamGPUPrimitive),
                kStateWords * 4, sizeof(MeshletStreamGPUBLASHeader) + (kCapacity + kOwnerBlocks) * 16,
                sizeof(MeshletStreamGPUActiveHeader), kPrimitives * 3 * sizeof(MeshletStreamGPUInstanceBLAS)};
            const uint32_t strides[] = {sizeof(MeshletStreamGPUParams), sizeof(MeshletStreamGPUPrimitive), 4,
                sizeof(MeshletStreamGPUBLASHeader), sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUInstanceBLAS)};
            std::unique_ptr<BindlessHeap> heap;
            require(bool(device->createBindlessHeap({.maxBuffers = InputCount}).transform([&](auto value) { heap = std::move(value); })), "Heap creation");
            std::array<std::unique_ptr<Buffer>, InputCount> buffers;
            std::array<BindlessHandle, InputCount> handles;
            for (uint32_t input = 0; input < InputCount; ++input) {
                require(bool(device->createBuffer({.size = sizes[input], .structureStride = strides[input],
                    .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto value) { buffers[input] = std::move(value); })), "Input buffer");
                require(bool(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto value) { handles[input] = std::move(value); })), "Handle allocation");
                auto slice = buffers[input]->slice();
                require(bool(slice) && bool(heap->writeStorageBuffer(handles[input], *slice)), "Buffer binding");
            }
            ShaderCompileResult compiled;
            require(bool(compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
                .entryPointName = kMeshletStreamBLASInputEntryPoint, .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, compiled.diagnostics)
                .transform([&](auto value) { compiled = std::move(value); })), compiled.diagnostics.c_str());
            std::unique_ptr<ShaderModule> shader;
            std::unique_ptr<ComputePipeline> pipeline;
            require(bool(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto value) { shader = std::move(value); })) &&
                bool(device->createComputePipeline({.computeShader = {shader.get()}, .usesBindlessHeap = true,
                    .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}).transform([&](auto value) { pipeline = std::move(value); })), "Pipeline creation");
            auto* queue = device->getQueue(QueueType::Graphics);
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Fence> fence;
            std::unique_ptr<Buffer> readback;
            require(bool(device->createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); })) &&
                bool(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); })) &&
                bool(device->createFence(false).transform([&](auto value) { fence = std::move(value); })) &&
                bool(device->createBuffer({.size = sizes[State] + sizeof(MeshletStreamGPUBLASHeader) + sizes[Active],
                    .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto value) { readback = std::move(value); })), "Command resources");
            struct Drain { Queue& queue; ~Drain() { (void)queue.waitIdle(); } } drain{*queue};
            const auto upload = [&](Input input, const void* data, uint64_t bytes) {
                void* mapped = buffers[input]->map(); require(mapped != nullptr, "Map upload");
                std::memset(mapped, 0, sizes[input]);
                if (bytes != 0) { std::memcpy(mapped, data, bytes); }
                buffers[input]->flush(); buffers[input]->unmap();
            };
            std::array<MeshletStreamGPUPrimitive, kPrimitives> primitives{};
            std::array<MeshletStreamGPUInstanceBLAS, kPrimitives * 3> representations{};
            for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
                primitives[primitive].lodLevelOffset = primitive;
                primitives[primitive].lodLevelCount = 1;
                representations[primitive * 3 + 2].previousGroupCount = 1;
            }
            upload(Primitives, primitives.data(), sizeof(primitives));
            upload(Representations, representations.data(), sizeof(representations));
            MeshletStreamUserPush push{.pageTableBuffer = handles[State].shaderIndex,
                .paramsBuffer = handles[Params].shaderIndex,
                .instanceBuffer = handles[Primitives].shaderIndex,
                .primitiveBuffer = handles[Primitives].shaderIndex,
                .groupBuffer = handles[Primitives].shaderIndex, .blasHeaderBuffer = handles[Header].shaderIndex};
            bool submitted = false;
            // Exercise the production prefix/admission phases with their real
            // GPU-generated count ABI, independently of the cut selector.
            for (uint32_t scenario = 0; scenario < 8; ++scenario) {
                const bool priority = scenario >= 5;
                const bool idleCache = scenario == 3 || scenario == 4 || scenario == 7;
                const bool evictCache = scenario == 3 || scenario == 7;
                std::array<uint32_t, kPrimitives> counts{};
                for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) { counts[primitive] = primitive % 5 + 1; }
                uint32_t capacity = std::accumulate(counts.begin(), counts.begin() + 73, 0u);
                if (scenario == 1) { counts.fill(1); capacity = 97; counts[0] = capacity + 1; }
                if (scenario == 2) {
                    std::fill_n(counts.begin(), 64, 0);
                    capacity = std::accumulate(counts.begin(), counts.end(), 0u);
                }
                if (idleCache) { counts.fill(3); capacity = evictCache ? kPrimitives * 2 : kCapacity; }
                if (scenario == 5 || scenario == 6) {
                    counts.fill(1); capacity = 3;
                    if (scenario == 6) { counts[kPrimitives - 1] = capacity + 1; }
                }
                MeshletStreamGPUParams params;
                params.scenePrimitiveCount = kPrimitives;
                params.sceneLodLevelCount = kPrimitives;
                params.rtRepresentationCount = kPrimitives * 3;
                params.rtRepresentationGroupCapacity = capacity;
                params.rtRepresentationStateBuffer = handles[State].shaderIndex;
                params.rtRepresentationHeaderBuffer = handles[Active].shaderIndex;
                params.rtRepresentationBuffer = handles[Representations].shaderIndex;
                params.lodTopologyBuffer = handles[State].shaderIndex;
                params.lodStateBuffer = handles[State].shaderIndex;
                params.enableBlasRepresentations = 1;
                params.enableBlasPriority = priority;
                params.blasClusterReferenceCapacity = evictCache ? kPrimitives * 2 : UINT32_MAX;
                params.blasStorageBytes = UINT32_MAX;
                params.blasBuildCapacity = UINT32_MAX;
                upload(Params, &params, sizeof(params));
                std::vector<uint32_t> previous;
                for (uint32_t frame = 0; frame < 16; ++frame) {
                    std::vector<uint32_t> state(kStateWords);
                    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
                        const uint32_t base = kPrimitives * 4 + primitive * 12;
                        state[primitive * 4] = counts[primitive];
                        state[primitive * 4 + 1] = counts[primitive] * 64;
                        state[base + 6] = counts[primitive];
                        state[base + 11] = idleCache ? 1u << 29 : 0;
                        for (uint32_t mode = 0; mode < 3; ++mode) {
                            const uint32_t importance = kImportanceBase + (primitive * 3 + mode) * 6;
                            state[importance + 1] = 0x7f7fffffu;
                            if (mode == primitive % 3) {
                                state[importance] = std::bit_cast<uint32_t>(float(primitive + 1)) + 1;
                                state[importance + 1] = std::bit_cast<uint32_t>(1.f);
                                state[importance + 3] = state[importance + 4] = 1;
                            }
                        }
                    }
                    MeshletStreamGPUBLASHeader header;
                    header.representationCount = idleCache ? kPrimitives * 2 : kPrimitives;
                    MeshletStreamGPUActiveHeader active;
                    active.activeGroupCapacity = capacity;
                    upload(State, state.data(), state.size() * 4);
                    upload(Header, &header, sizeof(header)); upload(Active, &active, sizeof(active));
                    if (submitted) { require(bool(fence->reset()) && bool(pool->reset()), "Reset commands"); }
                    require(bool(commands->begin()) && bool(commands->bindBindlessHeap(*heap)) &&
                        bool(commands->bindExecution(pipeline->execution())), "Begin production dispatch");
                    commands->hostWriteBarrier();
                    std::array<BufferBarrierDesc, InputCount> barriers;
                    for (uint32_t input = 0; input < InputCount; ++input) {
                        barriers[input] = {.buffer = buffers[input].get(),
                            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
                    }
                    const auto dispatch = [&](uint32_t phase, uint32_t groups, uint32_t traversal = 0) {
                        push.activeBuildPhase = phase;
                        push.traversalPhase = traversal;
                        require(bool(commands->pushBindlessData(&push, sizeof(push))) &&
                            bool(commands->dispatch(groups)) &&
                            bool(commands->synchronize({.buffers = barriers})), "Production admission phase");
                    };
                    if (priority) {
                        dispatch(35, kOwnerBlocks);
                        uint32_t parity = 0;
                        for (uint32_t runLog2 = 6; (1u << runLog2) < kOwners; ++runLog2) {
                            dispatch(36, kOwnerBlocks, runLog2 | (parity << 8)); parity ^= 1;
                        }
                        dispatch(37, kOwnerBlocks, parity << 8);
                    }
                    dispatch(31, kBlocks); dispatch(26, 1);
                    if (priority) { dispatch(38, kOwnerBlocks); dispatch(39, 1); dispatch(40, kOwnerBlocks); }
                    dispatch(32, kBlocks); dispatch(33, 1); dispatch(34, kBlocks);
                    for (auto& barrier : barriers) { barrier.after = {PipelineStageBits::Transfer, AccessBits::TransferRead}; }
                    require(bool(commands->synchronize({.buffers = barriers})), "Readback barrier");
                    uint64_t offset = 0;
                    for (const auto& [input, bytes] : {std::pair{State, sizes[State]},
                            std::pair{Header, uint64_t(sizeof(header))}, std::pair{Active, sizes[Active]}}) {
                        auto source = buffers[input]->slice({0, bytes}); auto destination = readback->slice({offset, bytes});
                        require(bool(source) && bool(destination) && bool(commands->copyBuffer(*source, *destination)), "Readback copy");
                        offset += bytes;
                    }
                    require(bool(commands->end()), "End admission commands");
                    CommandBuffer* submittedCommands[]{commands.get()};
                    require(bool(queue->submit({.commandBuffers = submittedCommands, .signalFence = fence.get()})) && bool(fence->wait()), "Submit admission commands");
                    submitted = true;
                    readback->invalidate(); const auto* data = static_cast<const uint8_t*>(readback->map());
                    require(data != nullptr, "Map admission readback");
                    std::memcpy(state.data(), data, sizes[State]);
                    std::memcpy(&header, data + sizes[State], sizeof(header));
                    std::memcpy(&active, data + sizes[State] + sizeof(header), sizeof(active));
                    readback->unmap();
                    uint32_t prefix = 0, acceptedEnd = 0, rejected = 0;
                    std::array<bool, kPrimitives> rankedAccepted{};
                    if (priority) {
                        uint32_t rankedPrefix = 0;
                        for (uint32_t remaining = kPrimitives; remaining != 0; --remaining) {
                            const uint32_t primitive = remaining - 1;
                            const uint32_t count = counts[primitive] - (evictCache ? 1u : 0u);
                            rankedAccepted[primitive] = count <= capacity && rankedPrefix <= capacity && count <= capacity - rankedPrefix;
                            // Individually impossible cuts do not reserve a prefix.
                            if (count <= capacity) { rankedPrefix += count; }
                        }
                    }
                    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
                        const uint32_t requested = counts[primitive] - (evictCache ? 1u : 0u);
                        const uint32_t count = priority && !rankedAccepted[primitive] ? 0u : requested;
                        const uint32_t base = kPrimitives * 4 + primitive * 12;
                        const bool overflow = priority ? requested != 0 && !rankedAccepted[primitive] :
                            count != 0 && (prefix > capacity || count > capacity - prefix);
                        require(state[base + 6] == count && state[base + 7] == prefix,
                            "Cross-block count/prefix mismatch");
                        require(((state[base + 11] >> 31) != 0) == overflow, "Whole-primitive suffix admission mismatch");
                        require(((state[base + 11] & (1u << 30)) != 0) == evictCache, "Idle-cache eviction mismatch");
                        if (overflow) { ++rejected; }
                        else if (count != 0) { acceptedEnd = prefix + count; }
                        prefix += count;
                    }
                    require(header.representationGroupOverflow == rejected && header.representationGroupCount == acceptedEnd &&
                        active.activeGroupCount == acceptedEnd, "Published admission totals disagree with complete cuts");
                    require(header.representationCount == (scenario == 4 ? kPrimitives * 2 : kPrimitives), "Eviction changed the wrong representation count");
                    require(previous.empty() || previous == state, "Parallel admission changed across identical frames");
                    previous = std::move(state);
                }
            }
            return RHITestResult::pass("137 primitives; legacy exact-fit/suffix/idle-cache cases; priority high-index winners, oversized top cut skipped, unique-primitive charging across owner modes; 128 deterministic production GPU admissions");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamRTRepresentationAdmissionTest);
} // namespace
} // namespace metallic::tests
