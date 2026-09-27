#include "RhiTest.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

#define CANDIDATE_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { \
    return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked) + " " + log); } } while (false)

class StreamClusterCandidatesTest final : public RhiTest {
public:
    StreamClusterCandidatesTest() { type = RhiTestType::Rendering; name = "stream_cluster_candidates_stable_parallel"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::string log;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream candidate regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless heap"); }
        CANDIDATE_REQUIRE(created);
        if (!device->capabilities().shaderBufferInt64Atomics || device->capabilities().subPixelPrecisionBits > 8) {
            return RhiTestResult::skip("Requires hybrid raster capabilities");
        }
        constexpr uint32_t capacity = 65537, groupCapacity = 30001, instances = 97;
        constexpr uint32_t candidateBase = 16 + 5 * capacity;
        constexpr uint32_t retryBase = 16 + 8 * capacity + ((capacity + 127) / 128) * 5;
        constexpr uint32_t poison = 0xa5a5a5a5u;
        VisibilityHybridRasterizer rasterizer;
        CANDIDATE_REQUIRE(rasterizer.initialize(*device, 1, 1, log, 1, capacity));
        std::unique_ptr<BindlessHeap> heap;
        CANDIDATE_REQUIRE(device->createBindlessHeap({.maxBuffers = 7}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        std::array<BindlessHandle, 7> handles;
        for (auto& handle : handles) { CANDIDATE_REQUIRE(heap->allocateBuffer().transform([&](auto rhiValue) { handle = std::move(rhiValue); })); }
        std::array<std::unique_ptr<Buffer>, 5> inputs;
        const uint32_t strides[] = {sizeof(MeshletStreamGpuActiveHeader), sizeof(MeshletStreamGpuActiveGroup),
            sizeof(MeshletStreamGpuRasterBindings), 4, 8};
        const uint32_t counts[] = {1, groupCapacity, 1, instances, capacity};
        for (size_t i = 0; i < inputs.size(); ++i) {
            CANDIDATE_REQUIRE(device->createBuffer({.size = uint64_t(strides[i]) * counts[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { inputs[i] = std::move(rhiValue); }));
            CANDIDATE_REQUIRE(heap->writeStorageBuffer(handles[i], *inputs[i]));
        }
        CANDIDATE_REQUIRE(heap->writeStorageBuffer(handles[5], rasterizer.clusterBuffer()));
        CANDIDATE_REQUIRE(heap->writeStorageBuffer(handles[6], rasterizer.candidateArguments()));
        const auto upload = [&](size_t index, const void* data, size_t size) {
            void* mapped = inputs[index]->map();
            if (!mapped) { return false; }
            std::memcpy(mapped, data, size);
            inputs[index]->flush();
            inputs[index]->unmap();
            return true;
        };
        std::array<std::unique_ptr<ShaderModule>, 2> shaders;
        std::array<std::unique_ptr<ComputePipeline>, 2> pipelines;
        for (size_t i = 0; i < shaders.size(); ++i) {
            ShaderCompileResult compiled;
            const auto result = compileSlangShaderToSpirv({
                .moduleName = i == 0 ? "Features/GPUDriven/GPUDrivenStreamAsset" : "HybridClusterProbe",
                .entryPointName = i == 0 ? "streamClusterPrepareMain" : "seedStreamCandidatesMain",
                .searchPath = i == 0 ? PROJECT_SOURCE_DIR "/Shaders" : PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
            log = compiled.diagnostics;
            CANDIDATE_REQUIRE(result);
            CANDIDATE_REQUIRE(device->createShaderModule({
                .spirv = compiled.spirv,
            }).transform([&](auto rhiValue) { shaders[i] = std::move(rhiValue); }));
            CANDIDATE_REQUIRE(device->createComputePipeline({
                .computeShader = {shaders[i].get()},
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = i == 0 ? uint32_t(sizeof(MeshletStreamUserPush)) : 12u,
            }).transform([&](auto rhiValue) { pipelines[i] = std::move(rhiValue); }));
        }
        std::unique_ptr<Buffer> readback, arguments;
        const uint64_t bytes = rasterizer.clusterBuffer().desc().size;
        CANDIDATE_REQUIRE(device->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
        CANDIDATE_REQUIRE(device->createBuffer({.size = 36, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        CANDIDATE_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        CANDIDATE_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        CANDIDATE_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        struct Case { uint32_t count; uint32_t phase; bool dense = false; bool hidden = false; };
        const Case cases[] = {{0, 0}, {1, 0}, {127, 0}, {128, 1}, {129, 1},
            {16383, 0}, {16384, 1}, {16385, 0}, {30001, 1}, {30001, 0, false, true},
            {3001, 0, true}, {0, 1}, {129, 0}, {3001, 1, true}};
        std::vector<MeshletStreamGpuActiveGroup> groups(groupCapacity);
        std::vector<std::array<uint32_t, 2>> retries(capacity);
        std::array<uint32_t, instances> visibility;
        bool submitted = false;
        size_t caseIndex = 0;
        for (const auto test : cases) {
            for (uint32_t i = 0; i < instances; ++i) {
                visibility[i] = test.hidden ? 0u : test.dense ? 1u : i % 4u;
            }
            for (uint32_t i = 0; i < capacity; ++i) { retries[i] = {0x80000001u | (i * 2654435761u), 0u}; }
            for (uint32_t i = 0; i < groupCapacity; ++i) {
                groups[i].gpuSceneInstanceIndex = (i * 37u + 13u) % instances;
                groups[i].clusterSelectionMask = test.dense ? UINT32_MAX :
                    i % 7u == 0u ? 0u : (0x80000001u | (i * 2246822519u));
            }
            std::vector<uint32_t> expected;
            for (uint32_t i = 0; i < test.count; ++i) {
                const uint32_t state = visibility[groups[i].gpuSceneInstanceIndex];
                for (uint32_t cluster = 0; cluster < 32; ++cluster) {
                    const uint32_t bit = 1u << cluster;
                    const bool phaseVisible = test.phase == 0 ? state == 1 :
                        state == 3 || (state == 1 && (retries[i][0] & bit) != 0);
                    if (phaseVisible && (groups[i].clusterSelectionMask & bit) != 0) { expected.push_back(i * 32 + cluster); }
                }
            }
            const uint32_t total = static_cast<uint32_t>(expected.size());
            expected.resize(std::min(total, capacity));
            MeshletStreamGpuActiveHeader header{.activeGroupCount = test.count, .activeGroupCapacity = groupCapacity,
                .maxActiveGroupClusters = 32};
            MeshletStreamGpuRasterBindings bindings{.instanceVisibilityBuffer = handles[3].shaderIndex};
            if (!upload(0, &header, sizeof(header)) || !upload(1, groups.data(), groups.size() * sizeof(groups[0])) ||
                !upload(2, &bindings, sizeof(bindings)) || !upload(3, visibility.data(), sizeof(visibility)) ||
                !upload(4, retries.data(), retries.size() * sizeof(retries[0]))) {
                return RhiTestResult::fail("Cannot upload candidate input");
            }
            if (submitted) { CANDIDATE_REQUIRE(fence->reset()); CANDIDATE_REQUIRE(pool->reset()); }
            CANDIDATE_REQUIRE(commands->begin());
            CANDIDATE_REQUIRE(rasterizer.beginClusters(*commands, 8, true, 0, capacity, true, true));
            commands->bindBindlessHeap(*heap);
            if (auto commandResult = commands->bindExecution((pipelines[1])->execution()); !commandResult) { return RhiTestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
            const uint32_t seedPush[] = {handles[4].shaderIndex, handles[5].shaderIndex, capacity};
            commands->pushBindlessData(seedPush, sizeof(seedPush));
            commands->dispatch((capacity + 127) / 128);
            BufferBarrierDesc ready{
                .buffer = &rasterizer.clusterBuffer(),
                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            };
            if (auto commandResult = commands->synchronize({.buffers = {&ready, 1}}); !commandResult) { return RhiTestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            MeshletStreamUserPush push{.activeGroupBuffer = handles[1].shaderIndex, .activeHeaderBuffer = handles[0].shaderIndex,
                .traversalPhase = test.phase, .rasterBindingsBuffer = handles[2].shaderIndex,
                .hybridQueueBuffer = handles[6].shaderIndex, .hybridClusterBuffer = handles[5].shaderIndex};
            CANDIDATE_REQUIRE(rasterizer.prepareStreamClusterCandidates(*commands, *pipelines[0], push));
            const BufferBarrierDesc copies[] = {
                {
                    .buffer = &rasterizer.clusterBuffer(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                },
                {
                    .buffer = &rasterizer.candidateArguments(),
                    .before = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                }};
            if (auto commandResult = commands->synchronize({.buffers = {copies, 2}}); !commandResult) { return RhiTestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            {
                auto sourceSlice = (&rasterizer.clusterBuffer())->slice({0, bytes});
                if (!sourceSlice) { return RhiTestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = readback.get()->slice({0, bytes});
                if (!destinationSlice) { return RhiTestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RhiTestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            {
                auto sourceSlice = (&rasterizer.candidateArguments())->slice({0, 36});
                if (!sourceSlice) { return RhiTestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = arguments.get()->slice({0, 36});
                if (!destinationSlice) { return RhiTestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RhiTestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            // Restore state and reuse scratch with real binning kernels. All
            // tags are deliberately culled; the copy retains the pre-bin state.
            const BufferBarrierDesc restore[] = {
                {
                    .buffer = &rasterizer.clusterBuffer(),
                    .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                },
                {
                    .buffer = &rasterizer.candidateArguments(),
                    .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                    .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
                }};
            if (auto commandResult = commands->synchronize({.buffers = {restore, 2}}); !commandResult) { return RhiTestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            CANDIDATE_REQUIRE(rasterizer.finishClusterBins(*commands));
            CANDIDATE_REQUIRE(commands->end());
            CommandBuffer* list[] = {commands.get()};
            CANDIDATE_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
            CANDIDATE_REQUIRE(fence->wait());
            submitted = true;
            readback->invalidate(); arguments->invalidate();
            const auto* bins = static_cast<const uint32_t*>(readback->map());
            const auto* args = static_cast<const uint32_t*>(arguments->map());
            if (!bins || !args) { return RhiTestResult::fail("Cannot read candidate output"); }
            bool valid = bins[12] == expected.size() && bins[13] == 32 && bins[14] == total - expected.size() && bins[15] == test.count;
            for (uint32_t i = 0; i < capacity; ++i) {
                valid = valid && bins[candidateBase + i * 2] == (i < expected.size() ? expected[i] : poison) &&
                    bins[candidateBase + i * 2 + 1] == (i < expected.size() ? UINT32_MAX : poison);
                valid = valid && bins[retryBase + i] == (test.phase == 0 && i < test.count ? 0u : retries[i][0]);
            }
            const uint32_t dispatches[] = {static_cast<uint32_t>(expected.size()),
                (static_cast<uint32_t>(expected.size()) + 127) / 128, (test.count + 127) / 128};
            for (uint32_t i = 0; i < 3; ++i) {
                valid = valid && args[i * 3] == std::min(dispatches[i], 65535u) &&
                    args[i * 3 + 1] == std::max(1u, (dispatches[i] + 65534) / 65535) && args[i * 3 + 2] == 1u;
            }
            readback->unmap(); arguments->unmap();
            if (!valid) { return RhiTestResult::fail("Candidate order, mask, guard or arguments mismatch in case " + std::to_string(caseIndex)); }
            ++caseIndex;
        }
        return RhiTestResult::pass("14 GPU/CPU cases: stable IDs, partial and >128 blocks, early/late recovery, empty/shrinking lists, overflow and 2D classify args");
    }
};

METALLIC_REGISTER_RHI_TEST(StreamClusterCandidatesTest);
#undef CANDIDATE_REQUIRE
} // namespace
} // namespace metallic::tests
