#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/NativeDescriptorHeapSpirv.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

#define NATIVE_REQUIRE(expression) do { \
    const render::Result<> result = (expression); \
    if (!result) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(result) + " " + log); } \
} while (false)

class NativeDescriptorHeapTest final : public RhiTest {
public:
    NativeDescriptorHeapTest()
    {
        type = RhiTestType::Rendering;
        name = "native_descriptor_heap_nested_layout";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::vector<uint32_t> untouched{0x12345678};
        const std::vector<uint32_t> malformed{0x07230203, 0x10600, 0, 10, 0, 0};
        if (render::normalizeNativeDescriptorHeapSpirv(malformed, untouched, log) ||
            untouched != std::vector<uint32_t>{0x12345678}) {
            return RhiTestResult::fail("malformed SPIR-V modified output or was accepted");
        }
        const char* rayCapabilities[] = {"spvRayQueryKHR"};
        render::ShaderCompileResult rejected;
        if (render::compileSlangShaderToSpirv({
                .moduleName = "NativeDescriptorHandles", .entryPointName = "unsafeNativeAsMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .capabilities = rayCapabilities, .capabilityCount = 1,
                .descriptorHeapMode = render::SlangDescriptorHeapMode::Native,
            }, {.enableDiskCache = false}, rejected) || !rejected.spirv.empty() ||
            rejected.diagnostics.find("resolveDescriptor") == std::string::npos) {
            return RhiTestResult::fail("unsafe native AS lowering was not rejected: " + rejected.diagnostics);
        }
        std::unique_ptr<render::Device> device;
        auto setup = render::createDevice({.applicationName = "Native descriptor layout regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("descriptor heaps unavailable"); }
        NATIVE_REQUIRE(setup);
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            render::ShaderCompileResult shader;
            NATIVE_REQUIRE(render::compileSlangShaderToSpirv({
                .moduleName = "NativeDescriptorHandles", .entryPointName = "nestedBufferMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = mode,
            }, {.enableDiskCache = false}, shader));
            std::vector<uint32_t> normalized;
            if (!render::normalizeNativeDescriptorHeapSpirv(shader.spirv, normalized, log) || normalized != shader.spirv) {
                return RhiTestResult::fail("normalization is not idempotent: " + log);
            }
            std::filesystem::create_directories(context.outputDirectory / name);
            std::ofstream binary(context.outputDirectory / name /
                (mode == render::SlangDescriptorHeapMode::Native ? "native.spv" : "mapped.spv"), std::ios::binary);
            binary.write(reinterpret_cast<const char*>(shader.spirv.data()), shader.spirv.size() * sizeof(uint32_t));
            render::ComputeProgram program;
            const render::ComputeProgramBindingDesc layout[] = {{0}, {1}};
            const auto initialized = program.initialize(*device, {
                .spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * sizeof(uint32_t),
                .bindings = layout, .bindingCount = 2, .requiresRayQuery = false,
            }, log);
            if (mode == render::SlangDescriptorHeapMode::Native && render::hasError(initialized, render::Error::Unsupported)) {
                return RhiTestResult::skip("mapped passed; native requires KHR untyped pointers");
            }
            NATIVE_REQUIRE(initialized);
            std::unique_ptr<render::Buffer> records, output;
            NATIVE_REQUIRE(device->createBuffer({.size = 32u * 96u, .structureStride = 96,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::Device}).transform([&](auto rhiValue) { records = std::move(rhiValue); }));
            NATIVE_REQUIRE(device->createBuffer({.size = 32u * 200u, .structureStride = 4,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
            render::QueueSubmissionTracker tracker;
            NATIVE_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            NATIVE_REQUIRE(device->createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
            NATIVE_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
            render::RenderFrameContext frame;
            struct Drain {
                render::RenderFrameContext& frame;
                render::CommandPool& pool;
                ~Drain()
                {
                    if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                    (void)pool.reset();
                    (void)frame.reset();
                }
            } drain{frame, *pool};
            NATIVE_REQUIRE(frame.begin(0));
            NATIVE_REQUIRE(commands->begin(&frame));
            const render::ComputeDispatchBinding bindings[] = {{.binding = 0, .buffer = records.get()}, {.binding = 1, .buffer = output.get()}};
            NATIVE_REQUIRE(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings, .bindingCount = 2}));
            NATIVE_REQUIRE(commands->end());
            render::CommandBuffer* submitted[] = {commands.get()};
            NATIVE_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
            NATIVE_REQUIRE(frame.wait(10'000'000'000ull));
            std::array<uint32_t, 1600> values{};
            const void* mapped = output->map();
            if (mapped == nullptr) { return RhiTestResult::fail("nested output map failed"); }
            output->invalidate();
            std::memcpy(values.data(), mapped, sizeof(values));
            output->unmap();
            for (uint32_t thread = 0; thread < 32; ++thread) {
                const uint32_t base = ((thread + 1) % 32) * 1000;
                for (uint32_t word = 0; word < 48; ++word) {
                    const uint32_t expected = base + word % 24 + 1;
                    if (values[thread * 50 + word] != expected) {
                        return RhiTestResult::fail("nested layout mismatch at thread/word " +
                            std::to_string(thread) + "/" + std::to_string(word));
                    }
                }
                if (values[thread * 50 + 48] != 32 || values[thread * 50 + 49] != 96) {
                    return RhiTestResult::fail("untyped runtime array length/stride mismatch");
                }
            }
        }
        return RhiTestResult::pass("mapped/native aggregate stores, raw/typed nested reads, arrays and dimensions agree; unsafe AS lowering rejected");
    }
};
METALLIC_REGISTER_RHI_TEST(NativeDescriptorHeapTest);

class FinalDescriptorIndicesTest final : public RhiTest {
public:
    FinalDescriptorIndicesTest()
    {
        type = RhiTestType::Rendering;
        name = "final_descriptor_indices_heap_switch";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::unique_ptr<render::Device> device;
        const auto setup = render::createDevice({.applicationName = "Final descriptor indices",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("descriptor heaps unavailable"); }
        NATIVE_REQUIRE(setup);
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            render::ShaderCompileResult compiled;
            NATIVE_REQUIRE(render::compileSlangShaderToSpirv({
                .moduleName = "FinalDescriptorIndices", .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = mode,
            }, compiled));
            std::unique_ptr<render::ShaderModule> shader;
            const auto moduleResult = device->createShaderModule({.code = compiled.spirv.data(),
                .byteSize = compiled.spirv.size() * sizeof(uint32_t)}).transform([&](auto rhiValue) { shader = std::move(rhiValue); });
            if (mode == render::SlangDescriptorHeapMode::Native && render::hasError(moduleResult, render::Error::Unsupported)) {
                return RhiTestResult::skip("mapped passed; native requires KHR untyped pointers");
            }
            NATIVE_REQUIRE(moduleResult);
            struct Push { uint32_t inputBuffer; uint32_t cookie; };
            static_assert(sizeof(Push) == 8);
            std::unique_ptr<render::ComputePipeline> pipeline;
            NATIVE_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .computeEntryPoint = "main",
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(Push)}).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
            std::array<std::unique_ptr<render::BindlessHeap>, 2> heaps;
            std::array<std::unique_ptr<render::Buffer>, 2> inputs, outputs;
            std::array<render::BindlessHandle, 2> inputHandles, outputHandles;
            for (uint32_t i = 0; i < heaps.size(); ++i) {
                // Different image capacities move the buffer partition. Also skip
                // slot zero so neither a local index nor an implicit slot can pass.
                NATIVE_REQUIRE(device->createBindlessHeap({.maxSampledImages = 3u + i * 10u, .maxBuffers = 4}).transform([&](auto rhiValue) { heaps[i] = std::move(rhiValue); }));
                render::BindlessHandle unused;
                NATIVE_REQUIRE(heaps[i]->allocateBuffer().transform([&](auto rhiValue) { unused = std::move(rhiValue); }));
                NATIVE_REQUIRE(heaps[i]->allocateBuffer().transform([&](auto rhiValue) { inputHandles[i] = std::move(rhiValue); }));
                NATIVE_REQUIRE(heaps[i]->allocateBuffer().transform([&](auto rhiValue) { outputHandles[i] = std::move(rhiValue); }));
                if (inputHandles[i].index == 0 || inputHandles[i].shaderIndex == inputHandles[i].index) {
                    return RhiTestResult::fail("test must exercise a nonzero slot and buffer partition");
                }
                NATIVE_REQUIRE(device->createBuffer({.size = 16, .usage = render::BufferUsageBits::Storage,
                    .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { inputs[i] = std::move(rhiValue); }));
                NATIVE_REQUIRE(device->createBuffer({.size = 16, .usage = render::BufferUsageBits::Storage,
                    .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { outputs[i] = std::move(rhiValue); }));
                NATIVE_REQUIRE(heaps[i]->writeStorageBuffer(inputHandles[i], *inputs[i]));
                NATIVE_REQUIRE(heaps[i]->writeStorageBuffer(outputHandles[i], *outputs[i]));
                const std::array<uint32_t, 4> data{outputHandles[i].shaderIndex, 17u + i * 13u, 0, 0};
                void* mapped = inputs[i]->map();
                if (mapped == nullptr) { return RhiTestResult::fail("input map failed"); }
                std::memcpy(mapped, data.data(), sizeof(data));
                inputs[i]->flush();
                inputs[i]->unmap();
            }
            if (inputHandles[0].shaderIndex == inputHandles[1].shaderIndex) {
                return RhiTestResult::fail("test heaps must produce different final indices for the same local slot");
            }
            render::QueueSubmissionTracker tracker;
            NATIVE_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            NATIVE_REQUIRE(device->createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
            NATIVE_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
            render::RenderFrameContext frame;
            struct Drain {
                render::RenderFrameContext& frame;
                render::CommandPool& pool;
                ~Drain()
                {
                    if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                    (void)pool.reset();
                    (void)frame.reset();
                }
            } drain{frame, *pool};
            NATIVE_REQUIRE(frame.begin(0));
            NATIVE_REQUIRE(commands->begin(&frame));
            for (uint32_t i = 0; i < heaps.size(); ++i) {
                const Push push{inputHandles[i].shaderIndex, 0x12340000u + i};
                if (i == 0) {
                    commands->bindBindlessHeap(*heaps[i]);
                    commands->pushBindlessData(&push, sizeof(push));
                    commands->bindComputePipeline(*pipeline);
                } else {
                    // Rebinding the heap must preserve/replay the exact user payload.
                    commands->bindComputePipeline(*pipeline, &push, sizeof(push));
                    commands->bindBindlessHeap(*heaps[i]);
                }
                commands->dispatch(1, 1, 1);
            }
            NATIVE_REQUIRE(commands->end());
            render::CommandBuffer* submitted[] = {commands.get()};
            NATIVE_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
            NATIVE_REQUIRE(frame.wait(10'000'000'000ull));
            for (uint32_t i = 0; i < heaps.size(); ++i) {
                std::array<uint32_t, 4> actual{};
                const void* mapped = outputs[i]->map();
                if (mapped == nullptr) { return RhiTestResult::fail("output map failed"); }
                outputs[i]->invalidate();
                std::memcpy(actual.data(), mapped, sizeof(actual));
                outputs[i]->unmap();
                const std::array<uint32_t, 4> expected{17u + i * 13u, 0x12340000u + i,
                    inputHandles[i].shaderIndex, outputHandles[i].shaderIndex};
                if (actual != expected) {
                    return RhiTestResult::fail("final index or push ABI mismatch for heap " + std::to_string(i));
                }
            }
        }
        return RhiTestResult::pass("mapped/native final indices work across heap layouts, nonzero slots and nested resource indices");
    }
};
METALLIC_REGISTER_RHI_TEST(FinalDescriptorIndicesTest);

class NativeDescriptorAtomicsTest final : public RhiTest {
public:
    NativeDescriptorAtomicsTest()
    {
        type = RhiTestType::Rendering;
        name = "native_descriptor_heap_mixed_atomics";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::unique_ptr<render::Device> device;
        const auto setup = render::createDevice({.applicationName = "Native mixed atomic regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("descriptor heaps unavailable"); }
        NATIVE_REQUIRE(setup);
        if (!device->capabilities().shaderBufferInt64Atomics) { return RhiTestResult::skip("uint64 buffer atomics unavailable"); }
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        constexpr uint32_t count = 256, base = 8;
        constexpr uint64_t canary = 0xa55aa55aa55aa55aull;
        for (auto mode : {render::SlangDescriptorHeapMode::Mapped, render::SlangDescriptorHeapMode::Native}) {
            render::ShaderCompileResult compiled;
            NATIVE_REQUIRE(render::compileSlangShaderToSpirv({.moduleName = "NativeDescriptorAtomics", .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .descriptorHeapMode = mode},
                {.enableDiskCache = false}, compiled));
            std::vector<uint32_t> normalized;
            if (!render::normalizeNativeDescriptorHeapSpirv(compiled.spirv, normalized, log) || normalized != compiled.spirv) {
                return RhiTestResult::fail("mixed atomic normalization is not idempotent: " + log);
            }
            std::filesystem::create_directories(context.outputDirectory / name);
            std::ofstream binary(context.outputDirectory / name /
                (mode == render::SlangDescriptorHeapMode::Native ? "native.spv" : "mapped.spv"), std::ios::binary);
            binary.write(reinterpret_cast<const char*>(compiled.spirv.data()), compiled.spirv.size() * sizeof(uint32_t));
            std::unique_ptr<render::ShaderModule> shader;
            const auto module = device->createShaderModule({.code = compiled.spirv.data(),
                .byteSize = compiled.spirv.size() * sizeof(uint32_t)}).transform([&](auto rhiValue) { shader = std::move(rhiValue); });
            if (mode == render::SlangDescriptorHeapMode::Native && render::hasError(module, render::Error::Unsupported)) {
                return RhiTestResult::skip("mapped passed; native requires KHR untyped pointers");
            }
            NATIVE_REQUIRE(module);
            struct Push { uint32_t atomics, counters, records, count, base; };
            static_assert(sizeof(Push) == 20);
            std::unique_ptr<render::ComputePipeline> pipeline;
            NATIVE_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .computeEntryPoint = "main",
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(Push)}).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
            std::unique_ptr<render::BindlessHeap> heap;
            NATIVE_REQUIRE(device->createBindlessHeap({.maxSampledImages = 7, .maxBuffers = 8}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
            render::BindlessHandle unused;
            NATIVE_REQUIRE(heap->allocateBuffer().transform([&](auto rhiValue) { unused = std::move(rhiValue); }));
            std::array<render::BindlessHandle, 3> handles;
            std::array<std::unique_ptr<render::Buffer>, 3> buffers;
            const std::array<uint64_t, 3> sizes{1024 * 8, 16, count * 96};
            const std::array<uint32_t, 3> strides{8, 4, 96};
            for (uint32_t i = 0; i < buffers.size(); ++i) {
                NATIVE_REQUIRE(heap->allocateBuffer().transform([&](auto rhiValue) { handles[i] = std::move(rhiValue); }));
                NATIVE_REQUIRE(device->createBuffer({.size = sizes[i], .structureStride = strides[i],
                    .usage = render::BufferUsageBits::Storage,
                    .memoryLocation = i == 2 ? render::MemoryLocation::HostUpload : render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { buffers[i] = std::move(rhiValue); }));
                NATIVE_REQUIRE(heap->writeStorageBuffer(handles[i], *buffers[i]));
                if (handles[i].index == 0 || handles[i].index == handles[i].shaderIndex) {
                    return RhiTestResult::fail("atomic test requires nonzero final descriptor indices");
                }
            }
            std::array<uint64_t, 1024> values;
            values.fill(canary);
            values[base] = values[base + 1] = values[base + 3] = 0;
            values[base + 2] = UINT64_MAX;
            std::array<uint32_t, 4> counters{};
            std::vector<uint32_t> records(count * 24);
            for (uint32_t i = 0; i < records.size(); ++i) { records[i] = (i / 24) * 1000 + i % 24 + 1; }
            const std::array<const void*, 3> initial{values.data(), counters.data(), records.data()};
            for (uint32_t i = 0; i < buffers.size(); ++i) {
                void* mapped = buffers[i]->map();
                if (!mapped) { return RhiTestResult::fail("atomic input map failed"); }
                std::memcpy(mapped, initial[i], sizes[i]);
                buffers[i]->flush();
                buffers[i]->unmap();
            }
            render::QueueSubmissionTracker tracker;
            NATIVE_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            NATIVE_REQUIRE(device->createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
            NATIVE_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
            render::RenderFrameContext frame;
            struct Drain {
                render::RenderFrameContext& frame;
                render::CommandPool& pool;
                ~Drain()
                {
                    if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                    (void)pool.reset();
                    (void)frame.reset();
                }
            } drain{frame, *pool};
            NATIVE_REQUIRE(frame.begin(0));
            NATIVE_REQUIRE(commands->begin(&frame));
            const render::BufferBarrierDesc barriers[] = {
                {.buffer = buffers[0].get(), .after = render::ResourceState::General},
                {.buffer = buffers[1].get(), .after = render::ResourceState::General},
                {.buffer = buffers[2].get(), .after = render::ResourceState::ShaderRead}};
            commands->barrier({.buffers = barriers, .bufferCount = 3});
            const Push push{handles[0].shaderIndex, handles[1].shaderIndex, handles[2].shaderIndex, count, base};
            commands->bindBindlessHeap(*heap);
            commands->bindComputePipeline(*pipeline, &push, sizeof(push));
            commands->dispatch(count / 64, 1, 1);
            NATIVE_REQUIRE(commands->end());
            render::CommandBuffer* submitted[] = {commands.get()};
            NATIVE_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
            NATIVE_REQUIRE(frame.wait(10'000'000'000ull));
            const std::array<void*, 2> destination{values.data(), counters.data()};
            for (uint32_t i = 0; i < destination.size(); ++i) {
                const void* mapped = buffers[i]->map();
                if (!mapped) { return RhiTestResult::fail("atomic output map failed"); }
                buffers[i]->invalidate();
                std::memcpy(destination[i], mapped, sizes[i]);
                buffers[i]->unmap();
            }
            if (counters != std::array<uint32_t, 4>{count, 0, 0, 0} ||
                values[base] != uint64_t(count) * 0x100000001ull ||
                values[base + 1] != ((uint64_t(count) << 32) | 0xf00dull) ||
                values[base + 2] != 0x10000f00dull || values[base + 3] != 0xfeed00000000beefull) {
                return RhiTestResult::fail("mixed atomic totals, uint64 dimensions or CPU-authored nested layout mismatch");
            }
            auto first = values.begin() + base + 8;
            std::sort(first, first + count);
            std::sort(first + count, first + 2 * count);
            for (uint32_t i = 0; i < count; ++i) {
                if (first[i] != uint64_t(i) * 0x100000001ull || first[count + i] != i) {
                    return RhiTestResult::fail("atomic return values are not a complete unique sequence");
                }
            }
            uint32_t compareExchangeWinners = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint64_t previous = first[2 * count + i];
                compareExchangeWinners += previous == 0;
                if (previous != 0 && previous != 0xfeed00000000beefull) {
                    return RhiTestResult::fail("compare-exchange returned a torn or unexpected value");
                }
            }
            if (compareExchangeWinners != 1) { return RhiTestResult::fail("compare-exchange must have exactly one winner"); }
            for (uint32_t i = 0; i < values.size(); ++i) {
                if ((i < base || (i >= base + 4 && i < base + 8) || i >= base + 8 + 3 * count) && values[i] != canary) {
                    return RhiTestResult::fail("atomic buffer guard overwritten at " + std::to_string(i));
                }
            }
        }
        return RhiTestResult::pass("mapped/native uint32 + uint64 Add/Min/Max/CAS, return values, guards, dimensions and CPU-authored nested reads agree");
    }
};
METALLIC_REGISTER_RHI_TEST(NativeDescriptorAtomicsTest);

#undef NATIVE_REQUIRE
} // namespace
} // namespace metallic::tests
