#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

namespace metallic::tests {
namespace {

constexpr uint32_t kWorkers = 8;
constexpr uint32_t kItemsPerWorker = 8;
constexpr uint32_t kItems = kWorkers * kItemsPerWorker;

#define HEAP_REQUIRE(expression) do { \
    const auto result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + render::errorToString(result.error())); } \
} while (false)

class BindlessHeapConcurrentAllocationTest final : public RHITest {
public:
    BindlessHeapConcurrentAllocationTest()
    {
        type = RHITestType::Resource;
        name = "bindless_heap_concurrent_allocation";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"binding.heap.concurrent.allocation"}, bench::Layer::RHI, "binding", "binding");
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().bindlessDescriptorHeap) { return RHITestResult::skip("requires bindless heap"); }
        std::unique_ptr<render::BindlessHeap> heap;
        HEAP_REQUIRE(context.device.createBindlessHeap({.maxSamplers = kItems,
            .maxSampledImages = kItems / 2, .maxStorageImages = kItems / 2, .maxBuffers = kItems})
            .transform([&](auto value) { heap = std::move(value); }));
        std::array<std::array<render::BindlessHandle, kItems>, 3> handles{};
        std::atomic_bool failed = false;
        std::barrier phase(kWorkers);
        std::vector<std::jthread> workers;
        for (uint32_t worker = 0; worker < kWorkers; ++worker) {
            workers.emplace_back([&, worker] {
                for (uint32_t round = 0; round < 64; ++round) {
                    phase.arrive_and_wait();
                    for (uint32_t item = worker; item < kItems; item += kWorkers) {
                        const render::BindlessHandleKind kinds[]{render::BindlessHandleKind::Sampler,
                            (item + round) % 2 ? render::BindlessHandleKind::SampledImage : render::BindlessHandleKind::StorageImage,
                            (item + round) % 2 ? render::BindlessHandleKind::Buffer : render::BindlessHandleKind::AccelerationStructure};
                        for (uint32_t group = 0; group < 3; ++group) {
                            const auto allocated = heap->allocate(kinds[group]);
                            handles[group][item] = allocated ? *allocated : render::BindlessHandle{};
                            if (!allocated) { failed = true; }
                        }
                    }
                    phase.arrive_and_wait();
                    // Interleave frees and allocations on the same full heap,
                    // allowing other workers to acquire each released slot.
                    for (uint32_t item = worker; item < kItems; item += kWorkers) {
                        for (auto& group : handles) {
                            const auto previous = group[item];
                            if (!previous.valid()) { continue; }
                            heap->release(previous);
                            const auto allocated = heap->allocate(previous.kind);
                            group[item] = allocated ? *allocated : render::BindlessHandle{};
                            if (!allocated) { failed = true; }
                        }
                    }
                    phase.arrive_and_wait();
                    if (worker == 0) {
                        for (const auto& group : handles) {
                            std::array<uint32_t, kItems> indices{};
                            for (uint32_t i = 0; i < kItems; ++i) { indices[i] = group[i].index; }
                            std::sort(indices.begin(), indices.end());
                            for (uint32_t i = 0; i < kItems; ++i) { if (indices[i] != i) { failed = true; } }
                        }
                        // Both image kinds and both buffer kinds share capacity.
                        for (const auto kind : {render::BindlessHandleKind::Sampler, render::BindlessHandleKind::SampledImage,
                                render::BindlessHandleKind::StorageImage, render::BindlessHandleKind::Buffer,
                                render::BindlessHandleKind::AccelerationStructure}) {
                            const auto overflow = heap->allocate(kind);
                            if (!render::hasError(overflow, render::Error::OutOfMemory)) { failed = true; }
                            if (overflow) { heap->release(*overflow); }
                        }
                    }
                    phase.arrive_and_wait();
                    for (uint32_t item = worker; item < kItems; item += kWorkers) {
                        for (auto& group : handles) { if (group[item].valid()) { heap->release(group[item]); } }
                    }
                }
            });
        }
        workers.clear(); // Join before inspecting or destroying shared state.
        return failed ? RHITestResult::fail("duplicate/missing slots, lost capacity or incorrect exhaustion") :
            RHITestResult::pass("8 workers, 64 full-capacity allocation/release rounds across all handle kinds");
    }
};

struct HeapPush {
    uint32_t constantBuffer, storageBuffer, bufferView, sampledImage, storageImage, sampler, outputBuffer, item;
};

class BindlessHeapConcurrentWritesTest final : public RHITest {
public:
    BindlessHeapConcurrentWritesTest()
    {
        type = RHITestType::Command;
        name = "bindless_heap_concurrent_writes_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"binding.heap.concurrent.readback"}, bench::Layer::RHI, "binding", "binding", {"readback.bin"});
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto& device = context.device;
        if (!device.capabilities().bindlessDescriptorHeap) { return RHITestResult::skip("requires bindless heap"); }
        std::unique_ptr<render::BindlessHeap> heap;
        HEAP_REQUIRE(device.createBindlessHeap({.maxSamplers = kItems,
            .maxSampledImages = kItems, .maxStorageImages = kItems, .maxBuffers = 3 * kItems + 1})
            .transform([&](auto value) { heap = std::move(value); }));
        struct Item {
            std::unique_ptr<render::Buffer> buffer;
            std::unique_ptr<render::BufferView> view;
            render::BufferSlice slice;
            std::unique_ptr<render::Texture> texture;
            std::unique_ptr<render::TextureView> image;
            std::array<render::BindlessHandle, 6> handles;
        };
        std::array<Item, kItems> items;
        for (uint32_t i = 0; i < kItems; ++i) {
            auto& item = items[i];
            HEAP_REQUIRE(device.createBuffer({.size = 512, .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::Constant,
                .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto value) { item.buffer = std::move(value); }));
            const uint32_t value = 0x12340000u + i;
            const uint32_t viewValue = 0x56780000u + i;
            auto* mapped = item.buffer->map();
            if (!mapped) { return RHITestResult::fail("input map failed"); }
            std::memset(mapped, 0, 512);
            std::memcpy(mapped, &value, sizeof(value));
            std::memcpy(static_cast<uint8_t*>(mapped) + 256, &viewValue, sizeof(viewValue));
            item.buffer->flush({0, 512});
            item.buffer->unmap();
            HEAP_REQUIRE(item.buffer->slice({128, 128}).transform([&](auto value) { item.slice = std::move(value); }));
            HEAP_REQUIRE(device.createBufferView(*item.buffer, {.type = render::BufferViewType::Raw, .range = {256, 256}})
                .transform([&](auto value) { item.view = std::move(value); }));
            HEAP_REQUIRE(device.createTexture({.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::Storage,
                .format = render::Format::RGBA8Unorm, .width = 2}).transform([&](auto value) { item.texture = std::move(value); }));
            HEAP_REQUIRE(device.createTextureView(*item.texture, {}).transform([&](auto value) { item.image = std::move(value); }));
        }
        std::unique_ptr<render::Buffer> output;
        HEAP_REQUIRE(device.createBuffer({.size = kItems * 16, .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto value) { output = std::move(value); }));
        render::BindlessHandle outputHandle;
        HEAP_REQUIRE(heap->allocate(render::BindlessHandleKind::Buffer).transform([&](auto value) { outputHandle = value; }));
        auto outputSlice = output->slice();
        HEAP_REQUIRE(outputSlice);
        HEAP_REQUIRE(heap->writeStorageBuffer(outputHandle, *outputSlice));

        std::atomic_bool failed = false;
        std::barrier start(kWorkers);
        std::vector<std::jthread> workers;
        for (uint32_t worker = 0; worker < kWorkers; ++worker) {
            workers.emplace_back([&, worker] {
                start.arrive_and_wait();
                for (uint32_t i = worker; i < kItems; i += kWorkers) {
                    auto& item = items[i];
                    const render::BindlessHandleKind kinds[]{render::BindlessHandleKind::Buffer, render::BindlessHandleKind::Buffer,
                        render::BindlessHandleKind::Buffer, render::BindlessHandleKind::SampledImage,
                        render::BindlessHandleKind::StorageImage, render::BindlessHandleKind::Sampler};
                    for (uint32_t j = 0; j < 6; ++j) {
                        auto allocated = heap->allocate(kinds[j]);
                        if (!allocated) { failed = true; return; }
                        item.handles[j] = *allocated;
                    }
                    for (uint32_t round = 0; round < 32; ++round) {
                        if (!heap->writeConstantBuffer(item.handles[0], *item.buffer) ||
                            !heap->writeStorageBuffer(item.handles[1], item.slice) ||
                            !heap->writeBufferView(item.handles[2], *item.view) ||
                            !heap->writeSampledImage(item.handles[3], *item.image, render::TextureLayout::General) ||
                            !heap->writeStorageImage(item.handles[4], *item.image) ||
                            !heap->writeSampler(item.handles[5], {.minFilter = render::SamplerFilter::Nearest,
                                .magFilter = render::SamplerFilter::Nearest,
                                .addressU = i % 2 ? render::SamplerAddressMode::Repeat : render::SamplerAddressMode::ClampToEdge})) {
                            failed = true; return;
                        }
                    }
                }
            });
        }
        workers.clear();
        if (failed) { return RHITestResult::fail("concurrent descriptor allocation/write failed"); }

        std::array<std::unique_ptr<render::ShaderModule>, 2> shaders;
        std::array<std::unique_ptr<render::ComputePipeline>, 2> pipelines;
        const char* entries[]{"writeImageMain", "readMain"};
        for (uint32_t i = 0; i < 2; ++i) {
            std::string diagnostics;
            auto compiled = render::compileSlangShaderToSpirv({.moduleName = "BindlessHeapConcurrency",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, diagnostics);
            if (!compiled) { return RHITestResult::fail(diagnostics); }
            HEAP_REQUIRE(device.createShaderModule({.spirv = compiled->spirv}).transform([&](auto value) { shaders[i] = std::move(value); }));
            HEAP_REQUIRE(device.createComputePipeline({.computeShader = {shaders[i].get(), "main"},
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(HeapPush)})
                .transform([&](auto value) { pipelines[i] = std::move(value); }));
        }
        bench::GPUCommands gpu(context.graphicsQueue);
        HEAP_REQUIRE(gpu.initialize(device));
        auto& commands = *gpu.commands;
        HEAP_REQUIRE(commands.bindBindlessHeap(*heap));
        std::array<render::TextureBarrierDesc, kItems> transitions;
        for (uint32_t i = 0; i < kItems; ++i) {
            transitions[i] = {.texture = items[i].texture.get(), .oldLayout = render::TextureLayout::Undefined,
                .newLayout = render::TextureLayout::General,
                .after = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderWrite}};
        }
        HEAP_REQUIRE(commands.synchronize({.textures = transitions}));
        for (uint32_t pass = 0; pass < 2; ++pass) {
            HEAP_REQUIRE(commands.bindExecution(pipelines[pass]->execution()));
            for (uint32_t i = 0; i < kItems; ++i) {
                const auto& h = items[i].handles;
                const HeapPush push{h[0].shaderIndex, h[1].shaderIndex, h[2].shaderIndex,
                    h[3].shaderIndex, h[4].shaderIndex, h[5].shaderIndex, outputHandle.shaderIndex, i};
                HEAP_REQUIRE(commands.pushBindlessData(&push, sizeof(push)));
                HEAP_REQUIRE(commands.dispatch(1, 1, 1));
            }
            if (pass == 0) {
                for (auto& transition : transitions) {
                    transition.oldLayout = render::TextureLayout::General;
                    transition.before = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderWrite};
                    transition.after = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderRead};
                }
                HEAP_REQUIRE(commands.synchronize({.textures = transitions}));
            }
        }
        const render::BufferBarrierDesc hostRead{.buffer = output.get(),
            .before = {render::PipelineStageBits::ComputeShader, render::AccessBits::ShaderWrite},
            .after = {render::PipelineStageBits::Host, render::AccessBits::HostRead}};
        HEAP_REQUIRE(commands.synchronize({.buffers = {&hostRead, 1}}));
        HEAP_REQUIRE(gpu.submitAndWait());
        output->invalidate({0, kItems * 16});
        const auto* mapped = static_cast<const uint32_t*>(output->map());
        if (!mapped) { return RHITestResult::fail("output map failed"); }
        std::array<uint32_t, kItems * 4> readback;
        std::memcpy(readback.data(), mapped, sizeof(readback));
        output->unmap();
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(readback));
        for (uint32_t i = 0; i < kItems; ++i) {
            for (uint32_t component = 0; component < 4; ++component) {
                const uint32_t expected = component == 3 ? (i % 2 ? i + 1 : 200 - i) :
                    (component == 2 ? 0x56780000u : 0x12340000u) + i;
                if (readback[i * 4 + component] != expected) {
                    return RHITestResult::fail("descriptor readback mismatch at item " + std::to_string(i) +
                        " component " + std::to_string(component));
                }
            }
        }
        return RHITestResult::pass("8 workers updated 384 descriptors; GPU verified constant/storage/view and sampled storage-image values");
    }
};

// Opt-in CPU API benchmark. "baseline" runs only externally serialized shared
// access, so the same test can safely measure a backend without internal locks.
// "internal" adds the concurrent RHI path. No GPU execution is timed.
class BindlessHeapThroughputTest final : public RHITest {
public:
    BindlessHeapThroughputTest()
    {
        type = RHITestType::Resource;
        name = "bindless_heap_cpu_throughput";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const char* mode = std::getenv("METALLIC_BINDLESS_BENCHMARK");
        if (!mode || (std::strcmp(mode, "baseline") && std::strcmp(mode, "internal"))) {
            return RHITestResult::skip("set METALLIC_BINDLESS_BENCHMARK=baseline or internal for CPU throughput samples");
        }
        if (!context.device.capabilities().bindlessDescriptorHeap) { return RHITestResult::skip("requires bindless heap"); }
        std::unique_ptr<render::BindlessHeap> heap;
        HEAP_REQUIRE(context.device.createBindlessHeap({.maxSamplers = 64, .maxSampledImages = 64, .maxBuffers = 64})
            .transform([&](auto value) { heap = std::move(value); }));
        std::unique_ptr<render::Buffer> buffer;
        std::unique_ptr<render::BufferView> view;
        HEAP_REQUIRE(context.device.createBuffer({.size = 256, .usage = render::BufferUsageBits::Storage})
            .transform([&](auto value) { buffer = std::move(value); }));
        HEAP_REQUIRE(context.device.createBufferView(*buffer, {.type = render::BufferViewType::Raw})
            .transform([&](auto value) { view = std::move(value); }));
        std::array<render::BindlessHandle, kWorkers> buffers, samplers;
        for (uint32_t i = 0; i < kWorkers; ++i) {
            HEAP_REQUIRE(heap->allocate(render::BindlessHandleKind::Buffer).transform([&](auto value) { buffers[i] = value; }));
            HEAP_REQUIRE(heap->allocate(render::BindlessHandleKind::Sampler).transform([&](auto value) { samplers[i] = value; }));
        }
        std::filesystem::create_directories(context.outputDirectory / name);
        std::ofstream csv(context.outputDirectory / name / "samples.csv");
        csv << "workload,threads,synchronization,sample,iterations_per_thread,operations,elapsed_ms,operations_per_second\n";
        std::mutex externalMutex;
        std::atomic_bool failed = false;
        const bool includeInternal = std::strcmp(mode, "internal") == 0;
        for (const char* workload : {"allocate_release", "buffer_write", "sampler_write", "mixed_write"}) {
            const bool allocate = std::strcmp(workload, "allocate_release") == 0;
            const bool sampler = std::strcmp(workload, "sampler_write") == 0;
            const bool mixed = std::strcmp(workload, "mixed_write") == 0;
            const uint32_t iterations = allocate ? 1000000 : sampler ? 200000 : mixed ? 300000 : 1000000;
            for (uint32_t threadCount : {1u, 2u, 4u, 8u}) {
                for (uint32_t sample = 0; sample < 7; ++sample) {
                    for (bool external : {true, false}) {
                        if (!external && !includeInternal && threadCount != 1) { continue; }
                        std::barrier phase(threadCount + 1);
                        std::vector<std::jthread> workers;
                        for (uint32_t worker = 0; worker < threadCount; ++worker) {
                            workers.emplace_back([&, worker] {
                                auto call = [&](auto&& operation) {
                                    std::unique_lock lock(externalMutex, std::defer_lock);
                                    if (external) { lock.lock(); }
                                    return operation();
                                };
                                auto operation = [&] {
                                    if (allocate) {
                                        const render::BindlessHandleKind kinds[]{render::BindlessHandleKind::Sampler,
                                            render::BindlessHandleKind::SampledImage, render::BindlessHandleKind::Buffer};
                                        auto handle = call([&] { return heap->allocate(kinds[worker % 3]); });
                                        if (!handle) { failed = true; }
                                        else { call([&] { heap->release(*handle); }); }
                                    } else {
                                        auto result = call([&] {
                                            return sampler || (mixed && worker % 2) ? heap->writeSampler(samplers[worker], {}) :
                                                heap->writeBufferView(buffers[worker], *view);
                                        });
                                        if (!result) { failed = true; }
                                    }
                                };
                                for (uint32_t i = 0; i < 10000; ++i) { operation(); }
                                phase.arrive_and_wait(); // Warmup and thread creation excluded.
                                phase.arrive_and_wait();
                                for (uint32_t i = 0; i < iterations; ++i) { operation(); }
                                phase.arrive_and_wait();
                            });
                        }
                        phase.arrive_and_wait();
                        const auto begin = std::chrono::steady_clock::now();
                        phase.arrive_and_wait();
                        phase.arrive_and_wait();
                        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                        workers.clear();
                        const uint64_t operations = uint64_t(iterations) * threadCount * (allocate ? 2 : 1);
                        csv << workload << ',' << threadCount << ',' << (external ? "external" : "direct") << ',' << sample << ','
                            << iterations << ',' << operations << ',' << seconds * 1000 << ',' << operations / seconds << '\n';
                    }
                }
            }
        }
        csv.flush();
        if (!csv || failed) { return RHITestResult::fail("benchmark operation or sample write failed"); }
        return RHITestResult::pass("CPU throughput samples saved; excludes GPU execution, creation and per-sample warmup");
    }
};

METALLIC_REGISTER_RHI_TEST(BindlessHeapConcurrentAllocationTest);
METALLIC_REGISTER_RHI_TEST(BindlessHeapConcurrentWritesTest);
METALLIC_REGISTER_RHI_TEST(BindlessHeapThroughputTest);

#undef HEAP_REQUIRE
} // namespace
} // namespace metallic::tests
