#include "RHITest.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"

#include <algorithm>
#include <array>
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
            constexpr uint32_t kStateWords = kPrimitives * 16;
            enum Input { Params, Primitives, State, Header, Active, Representations, InputCount };
            const uint64_t sizes[] = {sizeof(MeshletStreamGPUParams), kPrimitives * sizeof(MeshletStreamGPUPrimitive),
                kStateWords * 4, sizeof(MeshletStreamGPUBLASHeader) + (kCapacity + kBlocks) * 16,
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
            for (uint32_t scenario = 0; scenario < 5; ++scenario) {
                std::array<uint32_t, kPrimitives> counts{};
                for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) { counts[primitive] = primitive % 5 + 1; }
                uint32_t capacity = std::accumulate(counts.begin(), counts.begin() + 73, 0u);
                if (scenario == 1) { counts.fill(1); capacity = 97; counts[0] = capacity + 1; }
                if (scenario == 2) {
                    std::fill_n(counts.begin(), 64, 0);
                    capacity = std::accumulate(counts.begin(), counts.end(), 0u);
                }
                if (scenario >= 3) { counts.fill(3); capacity = scenario == 3 ? kPrimitives * 2 : kCapacity; }
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
                params.blasClusterReferenceCapacity = scenario == 3 ? kPrimitives * 2 : UINT32_MAX;
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
                        state[base + 11] = scenario >= 3 ? 1u << 29 : 0;
                    }
                    MeshletStreamGPUBLASHeader header;
                    header.representationCount = scenario >= 3 ? kPrimitives * 2 : kPrimitives;
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
                    for (uint32_t phase : {31u, 26u, 32u, 33u, 34u}) {
                        push.activeBuildPhase = phase;
                        require(bool(commands->pushBindlessData(&push, sizeof(push))) &&
                            bool(commands->dispatch(phase == 26 || phase == 33 ? 1 : kBlocks)) &&
                            bool(commands->synchronize({.buffers = barriers})), "Production admission phase");
                    }
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
                    for (uint32_t primitive = 0; primitive < kPrimitives; ++primitive) {
                        const uint32_t count = counts[primitive] - (scenario == 3 ? 1u : 0u);
                        const uint32_t base = kPrimitives * 4 + primitive * 12;
                        const bool overflow = count != 0 && (prefix > capacity || count > capacity - prefix);
                        require(state[base + 6] == count && state[base + 7] == prefix,
                            "Cross-block count/prefix mismatch");
                        require(((state[base + 11] >> 31) != 0) == overflow, "Whole-primitive suffix admission mismatch");
                        require(((state[base + 11] & (1u << 30)) != 0) == (scenario == 3), "Idle-cache eviction mismatch");
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
            return RHITestResult::pass("137 primitive owners, three scan blocks and tail; exact fit; oversized first owner rejects suffix; empty block; idle-cache eviction/retention; 80 deterministic production GPU admissions");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamRTRepresentationAdmissionTest);
} // namespace
} // namespace metallic::tests
