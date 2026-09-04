#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

#define FRAME_REQUIRE(expression) do { \
    const render::Result frameResult = (expression); \
    if (!frameResult) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(frameResult)); } \
} while (false)

constexpr uint64_t kWaitTimeout = 5'000'000'000ull;

struct Commands {
    render::RenderFrameContext frame;
    std::unique_ptr<render::CommandPool> pool;
    std::unique_ptr<render::CommandBuffer> buffer;

    explicit Commands(uint32_t slot = 0) : frame(slot) {}
    ~Commands()
    {
        if (frame.completion().isSubmitted()) {
            (void)frame.wait();
        }
        if (pool != nullptr) {
            (void)pool->reset();
        }
        (void)frame.reset();
    }
    render::Result initialize(render::Device& device, render::Queue& queue)
    {
        render::Result result = device.createCommandPool(queue, pool);
        return result ? pool->createCommandBuffer(buffer) : result;
    }
    render::Result begin(uint64_t index)
    {
        render::Result result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? buffer->begin(&frame) : result;
    }
    render::Result submit(render::QueueSubmissionTracker& tracker, render::Semaphore* gate = nullptr)
    {
        render::Result result = buffer->end();
        if (!result) { return result; }
        render::CommandBuffer* buffers[] = {buffer.get()};
        render::SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        return tracker.submit(render::QueueSubmitDesc{
            .waitSemaphores = gate != nullptr ? &wait : nullptr,
            .waitSemaphoreCount = gate != nullptr ? 1u : 0u,
            .commandBuffers = buffers,
            .commandBufferCount = 1,
        }, frame);
    }
};

// Construct after resources: every failure path first unblocks and drains the
// queue, so a regression reports failure instead of deadlocking during cleanup.
struct QueueDrain {
    render::Queue& queue;
    render::Semaphore* gate;
    render::Semaphore* secondGate = nullptr;
    ~QueueDrain()
    {
        if (gate != nullptr && gate->currentValue() < 1) { (void)gate->signal(1); }
        if (secondGate != nullptr && secondGate->currentValue() < 1) { (void)secondGate->signal(1); }
        (void)queue.waitIdle();
    }
};

// A regression that waits on a pending frame during recording must fail the
// overlap assertion instead of hanging the test process indefinitely.
struct GateWatchdog {
    std::mutex mutex;
    std::condition_variable_any wake;
    std::jthread worker;
    explicit GateWatchdog(render::Semaphore& gate)
        : worker([this, &gate](std::stop_token stop) {
            std::unique_lock lock(mutex);
            wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
            if (!stop.stop_requested() && gate.currentValue() == 0) { (void)gate.signal(1); }
        }) {}
};

bool readWords(render::Buffer& buffer, uint32_t* values, size_t count)
{
    buffer.invalidate();
    void* mapped = buffer.map();
    if (mapped == nullptr) { return false; }
    std::memcpy(values, mapped, count * sizeof(uint32_t));
    buffer.unmap();
    return true;
}

void storageBarrier(render::CommandBuffer& commandBuffer, render::Buffer& buffer)
{
    render::BufferBarrierDesc barrier{
        .buffer = &buffer,
        .before = render::ResourceState::General,
        .after = render::ResourceState::General,
    };
    commandBuffer.barrier({.buffers = &barrier, .bufferCount = 1});
}

class FrameCompletionLifecycleTest : public RhiTest {
public:
    FrameCompletionLifecycleTest() { type = RhiTestType::Command; name = "frame_completion_lifecycle"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands;
        render::DeferredReleaseQueue deferred;
        render::RenderSubsystemHost host;
        render::RenderFrameContext abandonedSlot(1);
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore(gate));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(0));
        const render::GpuCompletionPoint cancelled = commands.frame.completion();
        if (cancelled.isComplete() || cancelled.isSubmitted() || cancelled.wait(0)) {
            return RhiTestResult::fail("recording was reported as completed/waitable");
        }
        if (tracker.submit({.signalSemaphoreCount = 1}, commands.frame) || cancelled.isSubmitted()) {
            return RhiTestResult::fail("invalid submission committed a completion point");
        }
        FRAME_REQUIRE(commands.pool->reset());
        commands.frame.cancel();
        if (!cancelled.isCancelled() || !cancelled.isComplete()) {
            return RhiTestResult::fail("cancelled recording did not retire");
        }

        FRAME_REQUIRE(commands.frame.begin(1));
        render::CommandBuffer* staleBuffers[] = {commands.buffer.get()};
        if (tracker.submit({.commandBuffers = staleBuffers, .commandBufferCount = 1}, commands.frame)) {
            return RhiTestResult::fail("cancelled command recording was accepted by a new frame generation");
        }
        FRAME_REQUIRE(commands.buffer->begin(&commands.frame));
        const render::GpuCompletionPoint point = commands.frame.completion();
        auto retained = std::make_shared<uint32_t>(17);
        std::weak_ptr<uint32_t> retainedWeak = retained;
        commands.frame.retain(std::move(retained));
        auto retired = std::make_shared<uint32_t>(23);
        std::weak_ptr<uint32_t> retiredWeak = retired;
        deferred.retire(point, std::move(retired));
        std::string log;
        FRAME_REQUIRE(host.initialize(context.device, 2, log));
        FRAME_REQUIRE(host.beginFrame(1, 0, nullptr, log, &commands.frame));
        host.endFrame();
        auto outsideFrame = std::make_shared<uint32_t>(31);
        std::weak_ptr<uint32_t> outsideWeak = outsideFrame;
        host.retire(std::move(outsideFrame));
        FRAME_REQUIRE(commands.submit(tracker, gate.get()));
        FRAME_REQUIRE(abandonedSlot.begin(2));
        FRAME_REQUIRE(host.beginFrame(2, 1, nullptr, log, &abandonedSlot));
        auto replaced = std::make_shared<uint32_t>(43);
        std::weak_ptr<uint32_t> replacedWeak = replaced;
        host.retire(std::move(replaced));
        host.endFrame();
        abandonedSlot.cancel();
        FRAME_REQUIRE(abandonedSlot.begin(3));
        FRAME_REQUIRE(host.beginFrame(3, 1, nullptr, log, &abandonedSlot));
        host.endFrame();
        abandonedSlot.cancel();
        if (replacedWeak.expired()) {
            return RhiTestResult::fail("cancelling a replacement released a resource still used by another slot");
        }
        deferred.collect();
        if (!point.isSubmitted() || point.isComplete() || point.value() != 1 ||
            commands.frame.begin(2, 0) || retainedWeak.expired() || retiredWeak.expired() || outsideWeak.expired()) {
            return RhiTestResult::fail("pending submission allowed reuse or premature release");
        }
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(point.wait(kWaitTimeout));
        deferred.collect();
        if (!retiredWeak.expired() || retainedWeak.expired()) {
            return RhiTestResult::fail("completion retirement or frame retention was incorrect");
        }
        FRAME_REQUIRE(commands.begin(2));
        FRAME_REQUIRE(host.beginFrame(2, 0, nullptr, log, &commands.frame));
        host.endFrame();
        if (!retainedWeak.expired() || !outsideWeak.expired() || !replacedWeak.expired() || !point.isComplete()) {
            return RhiTestResult::fail("completed resources were not reclaimed on reuse");
        }
        FRAME_REQUIRE(commands.pool->reset());
        commands.frame.cancel();
        host.shutdown();
        return RhiTestResult::pass();
    }
};

class FrameUploadLifetimeTest : public RhiTest {
public:
    FrameUploadLifetimeTest() { type = RhiTestType::Resource; name = "frame_upload_lifetime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands;
        render::RenderFrameContext conflictingSlot;
        std::unique_ptr<render::Streamer> streamer;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore(gate));
        FRAME_REQUIRE(context.device.createStreamer(render::StreamerDesc{
            .constantBufferSize = 4096,
            .dynamicBufferSizePerFrame = 64 * 1024,
            .queuedFrameCount = 1,
        }, streamer));
        constexpr uint32_t kLargeWordCount = 20'000;
        FRAME_REQUIRE(context.device.createBuffer(render::BufferDesc{
            .size = (kLargeWordCount + 1ull) * sizeof(uint32_t),
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback,
        }, output));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(0));
        FRAME_REQUIRE(streamer->beginFrame(commands.frame));
        std::vector<uint32_t> constant(1024, 0x12345678);
        if (streamer->streamConstantData(constant.data(), 4096) != 0 ||
            streamer->streamConstantData(constant.data(), 4) != UINT64_MAX) {
            return RhiTestResult::fail("constant arena overwrote an earlier allocation when full");
        }
        const uint32_t first = 0x11223344;
        std::vector<uint32_t> large(kLargeWordCount, 0x55667788);
        render::StreamDataChunk chunk{.data = &first, .size = sizeof(first)};
        auto firstUpload = streamer->streamBufferData({
            .dataChunks = &chunk, .dataChunkCount = 1, .dstBuffer = output.get(),
        });
        chunk = {.data = large.data(), .size = large.size() * sizeof(uint32_t)};
        auto secondUpload = streamer->streamBufferData({
            .dataChunks = &chunk, .dataChunkCount = 1, .dstBuffer = output.get(), .dstOffset = sizeof(first),
        });
        if (firstUpload.buffer == nullptr || secondUpload.buffer == nullptr || firstUpload.buffer == secondUpload.buffer) {
            return RhiTestResult::fail("upload growth did not create the expected old/new allocations");
        }
        streamer->copyStreamedData(*commands.buffer);
        streamer->endFrame();
        FRAME_REQUIRE(commands.submit(tracker, gate.get()));
        FRAME_REQUIRE(conflictingSlot.begin(1));
        if (streamer->beginFrame(conflictingSlot)) {
            return RhiTestResult::fail("upload arena allowed reuse of an incomplete slot");
        }
        conflictingSlot.cancel();
        // Neither CPU frame advancement nor destruction of the uploader may
        // invalidate copies already recorded/submitted to the GPU.
        for (int index = 0; index < 8; ++index) { streamer->endFrame(); }
        streamer.reset();
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
        std::vector<uint32_t> readback(kLargeWordCount + 1);
        if (!readWords(*output, readback.data(), readback.size()) || readback.front() != first ||
            !std::all_of(readback.begin() + 1, readback.end(), [](uint32_t word) { return word == 0x55667788; })) {
            return RhiTestResult::fail("pending upload data did not survive buffer growth/endFrame/destruction");
        }
        return RhiTestResult::pass();
    }
};

render::Result createProbe(render::Device& device, const char* entry,
    std::span<const render::ComputeProgramBindingDesc> bindings, render::ComputeProgram& program, std::string& log)
{
    render::ShaderCompileResult shader;
    render::Result result = render::compileSlangShaderToSpirv({
        .moduleName = "FrameResourceProbe", .entryPointName = entry,
        .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
    }, shader);
    if (!result) { log = shader.diagnostics; return result; }
    return program.initialize(device, {
        .spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * sizeof(uint32_t),
        .pushConstantSize = sizeof(uint32_t), .bindings = bindings.data(),
        .bindingCount = static_cast<uint32_t>(bindings.size()), .requiresRayQuery = false,
    }, log);
}

class FrameDescriptorSnapshotTest : public RhiTest {
public:
    FrameDescriptorSnapshotTest() { type = RhiTestType::Rendering; name = "frame_descriptor_snapshots"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result setup = render::createDevice({.applicationName = "Frame descriptors",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (!setup && render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("descriptor heap unsupported"); }
        FRAME_REQUIRE(setup);
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        render::QueueSubmissionTracker tracker;
        Commands first, second(1);
        render::ComputeProgram program;
        std::array<std::unique_ptr<render::Buffer>, 3> inputs;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(*device, queue));
        FRAME_REQUIRE(first.initialize(*device, queue));
        FRAME_REQUIRE(second.initialize(*device, queue));
        FRAME_REQUIRE(device->createSemaphore(gate));
        const std::array<uint32_t, 3> expected{17, 83, 197};
        for (size_t index = 0; index < inputs.size(); ++index) {
            FRAME_REQUIRE(device->createBuffer({.size = 4, .structureStride = 4,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostUpload}, inputs[index]));
            void* mapped = inputs[index]->map();
            if (mapped == nullptr) { return RhiTestResult::fail("input map failed"); }
            std::memcpy(mapped, &expected[index], 4);
            inputs[index]->flush(); inputs[index]->unmap();
        }
        FRAME_REQUIRE(device->createBuffer({.size = 12, .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}, output));
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer},
        };
        std::string log;
        FRAME_REQUIRE(createProbe(*device, "copyValue", bindings, program, log));
        QueueDrain drain{queue, gate.get()};
        FRAME_REQUIRE(first.begin(0));
        FRAME_REQUIRE(second.begin(1));
        for (uint32_t index = 0; index < 3; ++index) {
            auto& commands = index < 2 ? first : second;
            storageBarrier(*commands.buffer, *output);
            const render::ComputeDispatchBinding resources[] = {
                {.binding = 0, .buffer = inputs[index].get()}, {.binding = 1, .buffer = output.get()},
            };
            FRAME_REQUIRE(program.dispatch({.commandBuffer = commands.buffer.get(),
                .bindings = resources, .bindingCount = 2, .pushData = &index, .pushDataSize = 4}));
        }
        FRAME_REQUIRE(first.submit(tracker, gate.get()));
        FRAME_REQUIRE(second.submit(tracker));
        program.clear();
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(second.frame.wait(kWaitTimeout));
        std::array<uint32_t, 3> actual{};
        if (!readWords(*output, actual.data(), actual.size()) || actual != expected) {
            return RhiTestResult::fail("dispatch descriptors were overwritten within/across frames or pipeline clear invalidated work");
        }
        return RhiTestResult::pass();
    }
};

class FrameHistoryDependencyTest : public RhiTest {
public:
    FrameHistoryDependencyTest() { type = RhiTestType::Rendering; name = "frame_history_dependencies"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result setup = render::createDevice({.applicationName = "Frame history",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (!setup && render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("descriptor heap unsupported"); }
        FRAME_REQUIRE(setup);
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        render::QueueSubmissionTracker tracker;
        std::array<Commands, 3> frames;
        render::HistoryResourceManager history;
        render::ComputeProgram program;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(*device, queue));
        FRAME_REQUIRE(history.initialize(*device));
        FRAME_REQUIRE(device->createSemaphore(gate));
        FRAME_REQUIRE(device->createBuffer({.size = 12, .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}, output));
        render::TextureDesc textureDesc;
        textureDesc.usage = render::TextureUsageBits::Storage;
        textureDesc.width = 1;
        textureDesc.height = 1;
        textureDesc.format = render::Format::Rgba32Sfloat;
        FRAME_REQUIRE(history.ensureTexture("history", textureDesc));
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::StorageBuffer},
        };
        std::string log;
        FRAME_REQUIRE(createProbe(*device, "accumulateHistory", bindings, program, log));
        QueueDrain drain{queue, gate.get()};
        for (uint32_t index = 0; index < 3; ++index) {
            auto& commands = frames[index];
            FRAME_REQUIRE(commands.initialize(*device, queue));
            FRAME_REQUIRE(commands.begin(index));
            history.beginFrame(index);
            FRAME_REQUIRE(history.transitionTexture(*commands.buffer, "history", render::HistorySlot::Current, render::ResourceState::General));
            FRAME_REQUIRE(history.transitionTexture(*commands.buffer, "history", render::HistorySlot::Previous, render::ResourceState::General));
            storageBarrier(*commands.buffer, *output);
            const render::ComputeDispatchBinding resources[] = {
                {.binding = 0, .textureView = history.texture("history", render::HistorySlot::Previous).view},
                {.binding = 1, .textureView = history.texture("history", render::HistorySlot::Current).view},
                {.binding = 2, .buffer = output.get()},
            };
            FRAME_REQUIRE(program.dispatch({.commandBuffer = commands.buffer.get(), .bindings = resources,
                .bindingCount = 3, .pushData = &index, .pushDataSize = 4}));
            history.markWritten("history");
            FRAME_REQUIRE(commands.submit(tracker, index == 0 ? gate.get() : nullptr));
        }
        textureDesc.width = 2;
        FRAME_REQUIRE(history.ensureTexture("history", textureDesc));
        history.reset();
        program.clear();
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(frames.back().frame.wait(kWaitTimeout));
        std::array<uint32_t, 3> actual{};
        if (!readWords(*output, actual.data(), actual.size()) || actual != std::array<uint32_t, 3>{41, 42, 43}) {
            return RhiTestResult::fail("General history dependencies or resized history lifetime failed");
        }
        return RhiTestResult::pass();
    }
};

class FrameTwoSlotGraphTest : public RhiTest {
public:
    FrameTwoSlotGraphTest() { type = RhiTestType::Rendering; name = "frame_two_slot_graph_reuse"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        std::array<Commands, 2> slots{Commands{0}, Commands{1}};
        render::RenderGraph graph = render::RenderGraph::createDefaultTriangleGraph();
        render::RenderSubsystemHost host;
        render::RenderWorld world;
        render::RenderGraphExecutor executor(host, world);
        std::unique_ptr<render::Streamer> streamer;
        std::unique_ptr<render::Buffer> uploadReadback;
        std::array<std::unique_ptr<render::Buffer>, 6> imageReadbacks;
        std::unique_ptr<render::Semaphore> gate, rebuildGate;
        std::string log;
        constexpr uint32_t kWidth = 32;
        constexpr uint32_t kPixelCount = kWidth * kWidth;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        for (auto& slot : slots) { FRAME_REQUIRE(slot.initialize(context.device, context.graphicsQueue)); }
        FRAME_REQUIRE(context.device.createSemaphore(gate));
        FRAME_REQUIRE(context.device.createSemaphore(rebuildGate));
        FRAME_REQUIRE(context.device.createStreamer({.dynamicBufferSizePerFrame = 64 * 1024,
            .queuedFrameCount = 2}, streamer));
        FRAME_REQUIRE(context.device.createBuffer({.size = 6 * sizeof(uint32_t),
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}, uploadReadback));
        for (auto& image : imageReadbacks) {
            FRAME_REQUIRE(context.device.createBuffer({.size = kPixelCount * sizeof(uint32_t),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}, image));
        }
        render::RenderGraphCompileOptions options;
        options.enablePreviewOutputAccess = true;
        FRAME_REQUIRE(host.initialize(context.device, 2, log));
        FRAME_REQUIRE(executor.compile(context.device, graph, kWidth, kWidth, options, log));
        if (host.frameSlotCount() != 2) {
            return RhiTestResult::fail("graph compile replaced the caller's two-slot capacity");
        }
        QueueDrain drain{context.graphicsQueue, gate.get(), rebuildGate.get()};
        GateWatchdog watchdog(*gate);
        render::GpuCompletionPoint firstPoint;
        for (uint32_t index = 0; index < 6; ++index) {
            Commands& commands = slots[index % slots.size()];
            FRAME_REQUIRE(commands.begin(index));
            FRAME_REQUIRE(streamer->beginFrame(commands.frame));
            const uint32_t value = 100 + index;
            const render::StreamDataChunk chunk{.data = &value, .size = sizeof(value)};
            const auto upload = streamer->streamBufferData({.dataChunks = &chunk, .dataChunkCount = 1,
                .dstBuffer = uploadReadback.get(), .dstOffset = index * sizeof(value)});
            if (upload.buffer == nullptr) { return RhiTestResult::fail("slot upload allocation failed"); }
            streamer->copyStreamedData(*commands.buffer);
            streamer->endFrame();
            FRAME_REQUIRE(executor.execute(*commands.buffer));
            if (executor.streamingStats().streamer.queuedFrameCount != 2) {
                return RhiTestResult::fail("graph uploader did not use the host's two-slot capacity");
            }
            if (index == 1 && firstPoint.isComplete()) {
                return RhiTestResult::fail("recording slot 1 waited for slot 0 instead of overlapping it");
            }
            // Leave the shared attachment in ColorAttachment on alternate frames
            // so the graph also exercises a same-state write-after-write barrier.
            if (index % 2 == 0 || index == 5) {
                FRAME_REQUIRE(executor.transitionOutput(*commands.buffer, "Triangle.color", render::ResourceState::TransferSource));
                commands.buffer->copyTextureToBuffer({
                    .texture = executor.outputResource("Triangle.color")->texture,
                    .buffer = imageReadbacks[index].get(), .width = kWidth, .height = kWidth,
                });
            }
            FRAME_REQUIRE(commands.submit(tracker, index == 0 ? gate.get() : index == 5 ? rebuildGate.get() : nullptr));
            if (index == 0) { firstPoint = commands.frame.completion(); }
            if (index == 1) {
                if (slots[0].frame.begin(2, 0) || slots[1].frame.completion().isComplete()) {
                    return RhiTestResult::fail("two-slot ring reused an incomplete slot");
                }
                FRAME_REQUIRE(gate->signal(1));
                watchdog.worker.request_stop();
            }
        }
        if (executor.waitForSubmittedWork(0) || slots[1].frame.completion().isComplete()) {
            return RhiTestResult::fail("graph did not track its externally submitted pending work");
        }
        FRAME_REQUIRE(rebuildGate->signal(1));
        // Rebuild must wait before replacing graph images, passes and query pools.
        FRAME_REQUIRE(executor.compile(context.device, graph, 48, 48, options, log));
        if (!firstPoint.isComplete() || firstPoint.value() != 1 ||
            !slots[1].frame.completion().isComplete() || slots[1].frame.completion().value() != 6) {
            return RhiTestResult::fail("completion generations were not preserved through slot reuse/rebuild");
        }
        std::array<uint32_t, 6> words{};
        if (!readWords(*uploadReadback, words.data(), words.size()) ||
            words != std::array<uint32_t, 6>{100, 101, 102, 103, 104, 105}) {
            return RhiTestResult::fail("upload data was overwritten while alternating two slots");
        }
        std::array<uint32_t, kPixelCount> reference{}, pixels{};
        if (!readWords(*imageReadbacks[0], reference.data(), reference.size()) ||
            reference[0] == reference[kPixelCount / 2 + kWidth / 2]) {
            return RhiTestResult::fail("triangle readback did not contain rendered geometry");
        }
        for (uint32_t index : {2u, 4u, 5u}) {
            if (!readWords(*imageReadbacks[index], pixels.data(), pixels.size()) || pixels != reference) {
                return RhiTestResult::fail("overlapped graph rendering differed after slot reuse");
            }
        }
        return RhiTestResult::pass();
    }
};

// Signals all gates before waiting on any queue; a failed assertion cannot
// deadlock destruction when queues depend on one another.
struct DeviceDrain {
    render::Device& device;
    render::Semaphore* first = nullptr;
    render::Semaphore* second = nullptr;
    ~DeviceDrain()
    {
        for (auto* gate : {first, second}) {
            if (gate != nullptr && gate->currentValue() < 1) { (void)gate->signal(1); }
        }
        (void)device.waitIdle();
    }
};

class FrameMultiQueueCompletionTest : public RhiTest {
public:
    FrameMultiQueueCompletionTest() { type = RhiTestType::Command; name = "frame_multi_queue_completion"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (copyQueue == nullptr) { return RhiTestResult::skip("independent copy queue unavailable"); }
        render::QueueSubmissionTracker graphics, copy;
        render::RenderFrameContext frame;
        render::DeferredReleaseQueue retired;
        std::unique_ptr<render::Semaphore> graphicsGate, copyGate;
        FRAME_REQUIRE(graphics.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(copy.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(context.device.createSemaphore(graphicsGate));
        FRAME_REQUIRE(context.device.createSemaphore(copyGate));
        DeviceDrain drain{context.device, graphicsGate.get(), copyGate.get()};
        FRAME_REQUIRE(frame.begin(0));
        auto resource = std::make_shared<uint32_t>(17);
        const std::weak_ptr<uint32_t> weak = resource;
        frame.retain(resource);
        retired.retire(frame.completion(), std::move(resource));
        const auto batch = frame.completion();
        render::GpuCompletionPoint graphicsPoint, copyPoint;
        render::SemaphoreSubmitDesc graphicsWait{.semaphore = graphicsGate.get(), .value = 1};
        render::SemaphoreSubmitDesc copyWait{.semaphore = copyGate.get(), .value = 1};
        FRAME_REQUIRE(graphics.submitSegment({.waitSemaphores = &graphicsWait, .waitSemaphoreCount = 1}, frame, graphicsPoint));
        FRAME_REQUIRE(copy.submitSegment({.waitSemaphores = &copyWait, .waitSemaphoreCount = 1}, frame, copyPoint));
        std::vector<render::SemaphoreSubmitDesc> waits;
        if (batch.isSubmitted() || batch.isComplete() || batch.wait(0) || batch.appendWaits(waits) || frame.begin(1, 0)) {
            return RhiTestResult::fail("open submission batch was reusable or waitable");
        }
        // Simulate a later submission rejected before reaching the driver.
        render::GpuCompletionPoint failed;
        if (copy.submitSegment({.commandBufferCount = 1}, frame, failed) || failed.valid()) {
            return RhiTestResult::fail("failed segment acquired a completion value");
        }
        frame.cancel();
        if (!batch.isSubmitted() || batch.isCancelled() || batch.value() != 0) {
            return RhiTestResult::fail("partial multi-queue batch was cancelled instead of sealed");
        }
        FRAME_REQUIRE(batch.appendWaits(waits));
        FRAME_REQUIRE(batch.appendWaits(waits));
        if (waits.size() != 2) { return RhiTestResult::fail("composite waits were not coalesced per queue"); }
        FRAME_REQUIRE(graphicsGate->signal(1));
        FRAME_REQUIRE(graphicsPoint.wait(kWaitTimeout));
        retired.collect();
        if (batch.isComplete() || batch.wait(0) || copyPoint.isComplete() || weak.expired()) {
            return RhiTestResult::fail("graphics completion prematurely retired copy-queue resources");
        }
        FRAME_REQUIRE(copyGate->signal(1));
        FRAME_REQUIRE(batch.wait(kWaitTimeout));
        FRAME_REQUIRE(frame.begin(1));
        retired.collect();
        if (!weak.expired() || !batch.isComplete()) {
            return RhiTestResult::fail("completed partial batch failed to release resources");
        }
        // Two signals from one queue collapse to the final value, while old
        // segment points continue identifying their original submission.
        FRAME_REQUIRE(copy.submitSegment({}, frame, copyPoint));
        const auto oldCopy = copyPoint;
        FRAME_REQUIRE(copy.submitSegment({}, frame, copyPoint));
        FRAME_REQUIRE(frame.finishSubmission());
        FRAME_REQUIRE(frame.wait(kWaitTimeout));
        if (oldCopy.value() != 2 || copyPoint.value() != 3 || frame.completion().value() != 3) {
            return RhiTestResult::fail("failed segment consumed a timeline value or rewrote old completion");
        }
        return RhiTestResult::pass();
    }
};

class FrameGraphTransferPass final : public render::RenderGraphPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    render::QueueType queueType() const override
    {
        const std::string queue = properties().value("queue", std::string("graphics"));
        return queue == "copy" ? render::QueueType::Copy : queue == "compute"
            ? render::QueueType::Compute : render::QueueType::Graphics;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (properties().value("copy", false)) {
            reflection.addBufferInput("source").buffer(16).transferRead();
        }
        reflection.addBufferOutput("data").buffer(16).transferWrite();
        return reflection;
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        if (properties().value("fail", false)) { return render::makeError(render::Error::Failure); }
        auto* output = context.outputBuffer("data").buffer();
        if (properties().value("copy", false)) {
            context.commandBuffer().copyBuffer({.source = context.inputBuffer("source").buffer(),
                .destination = output, .size = 16});
            return {};
        }
        const uint32_t value = 100 + static_cast<uint32_t>(context.frameIndex());
        const std::array<uint32_t, 4> words{value, value + 1, value + 2, value + 3};
        const render::StreamDataChunk chunk{.data = words.data(), .size = sizeof(words)};
        return context.streamer()->streamBufferData({.dataChunks = &chunk, .dataChunkCount = 1,
            .dstBuffer = output}).valid() ? render::Result{} : render::makeError(render::Error::Failure);
    }
};

class FrameGraphHistoryPass final : public render::UnsafePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("source").buffer(16).transferRead();
        reflection.addBufferOutput("data").buffer(16).transferWrite();
        return reflection;
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        auto* history = context.historyResources();
        if (history == nullptr) { return render::makeError(render::Error::InvalidArgument); }
        render::Result result = history->ensureBuffer("self-submit-history", {.size = 16,
            .usage = render::BufferUsageBits::TransferSource | render::BufferUsageBits::TransferDestination});
        if (!result) { return result; }
        result = history->transitionBuffer(context.commandBuffer(), "self-submit-history",
            render::HistorySlot::Current, render::ResourceState::TransferDestination);
        if (!result) { return result; }
        auto* source = context.inputBuffer("source").buffer();
        context.commandBuffer().copyBuffer({.source = source,
            .destination = history->buffer("self-submit-history", render::HistorySlot::Current).buffer, .size = 16});
        if (history->hasPrevious("self-submit-history")) {
            result = history->transitionBuffer(context.commandBuffer(), "self-submit-history",
                render::HistorySlot::Previous, render::ResourceState::TransferSource);
            if (!result) { return result; }
            source = history->buffer("self-submit-history", render::HistorySlot::Previous).buffer;
        }
        context.commandBuffer().copyBuffer({.source = source,
            .destination = context.outputBuffer("data").buffer(), .size = 16});
        history->markWritten("self-submit-history");
        return {};
    }
};

void registerFrameGraphTransferPass()
{
    static const bool registered = [] {
        render::registerRenderGraphPassType("FrameGraphHistoryPass", "Self submission test history",
            [] { return std::make_unique<FrameGraphHistoryPass>(); });
        render::registerRenderGraphPassType("FrameGraphTransferPass", "Frame submission test transfer",
            [] { return std::make_unique<FrameGraphTransferPass>(); });
        return true;
    }();
    (void)registered;
}

class FrameSelfSubmitTwoSlotTest : public RhiTest {
public:
    FrameSelfSubmitTwoSlotTest() { type = RhiTestType::Rendering; name = "frame_self_submit_two_slots"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (copyQueue == nullptr) { return RhiTestResult::skip("independent copy queue unavailable"); }
        registerFrameGraphTransferPass();
        render::RenderGraph graph;
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("FrameGraphTransferPass", "Upload", {{"queue", "copy"}});
        graph.markOutput("Triangle.color");
        graph.markOutput("Upload.data");
        render::RenderGraphExecutor executor;
        Commands blockedGraphics, consumer;
        render::QueueSubmissionTracker blocker, consumerTracker;
        std::unique_ptr<render::Buffer> consumerReadback;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(blocker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(blockedGraphics.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore(gate));
        FRAME_REQUIRE(consumer.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(consumerTracker.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(context.device.createBuffer({.size = 16,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback,
            .queueAccess = render::QueueAccessBits::Copy}, consumerReadback));
        std::string log;
        FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        DeviceDrain drain{context.device, gate.get()};
        GateWatchdog watchdog(*gate);
        FRAME_REQUIRE(blockedGraphics.begin(0));
        FRAME_REQUIRE(blockedGraphics.submit(blocker, gate.get()));
        render::RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
            .copyQueue = copyQueue, .slotWaitTimeoutNanoseconds = 0};
        FRAME_REQUIRE(executor.execute(submit));
        const auto first = executor.lastSubmittedCompletion();
        std::vector<render::SemaphoreSubmitDesc> waits;
        FRAME_REQUIRE(first.appendWaits(waits));
        size_t finished = 0;
        for (const auto& wait : waits) {
            if (wait.semaphore->wait(wait.value, 300'000'000ull)) { ++finished; }
        }
        if (waits.size() != 2 || finished != 1 || first.isComplete()) {
            return RhiTestResult::fail("independent copy branch was blocked by the graphics branch");
        }
        std::array<uint32_t, 4> actual{};
        if (!readWords(*executor.outputResource("Upload.data")->buffer, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{100, 101, 102, 103}) {
            return RhiTestResult::fail("copy-queue upload did not finish while graphics was blocked");
        }
        FRAME_REQUIRE(executor.execute(submit));
        const auto second = executor.lastSubmittedCompletion();
        const uint64_t recorded = executor.streamingStats().frameIndex;
        if (first.isComplete() || second.isComplete() || executor.execute(submit) ||
            !executor.compiled() || executor.streamingStats().frameIndex != recorded || executor.waitForSubmittedWork(0)) {
            return RhiTestResult::fail("self submission failed two-slot overlap/backpressure contract");
        }
        // Consume the pending graph on an externally recorded command buffer.
        // transitionOutput attaches the aggregate wait to Queue::submit.
        FRAME_REQUIRE(consumer.begin(0));
        FRAME_REQUIRE(executor.transitionOutput(*consumer.buffer, "Upload.data", render::ResourceState::TransferSource));
        FRAME_REQUIRE(consumer.buffer->addDependency(second)); // Duplicate is coalesced.
        consumer.buffer->copyBuffer({.source = executor.outputResource("Upload.data")->buffer,
            .destination = consumerReadback.get(), .size = 16});
        FRAME_REQUIRE(consumer.submit(consumerTracker));
        if (consumer.frame.completion().isComplete()) {
            return RhiTestResult::fail("external consumer ignored pending graph completion");
        }
        FRAME_REQUIRE(gate->signal(1));
        watchdog.worker.request_stop();
        FRAME_REQUIRE(consumer.frame.wait(kWaitTimeout));
        if (!readWords(*consumerReadback, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{101, 102, 103, 104}) {
            return RhiTestResult::fail("pending self-to-external handoff copied the wrong frame");
        }
        submit.slotWaitTimeoutNanoseconds = kWaitTimeout;
        for (uint32_t index = 2; index < 6; ++index) { FRAME_REQUIRE(executor.execute(submit)); }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        if (!first.isComplete() || !second.isComplete() ||
            !readWords(*executor.outputResource("Upload.data")->buffer, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{105, 106, 107, 108}) {
            return RhiTestResult::fail("self-submitted upload data was corrupted through slot reuse");
        }
        FRAME_REQUIRE(executor.compile(context.device, graph, 24, 24, log));
        return RhiTestResult::pass();
    }
};

class FrameCrossQueueGraphTest : public RhiTest {
public:
    FrameCrossQueueGraphTest() { type = RhiTestType::Rendering; name = "frame_cross_queue_graph_dependencies"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* compute = context.device.getQueue(render::QueueType::Compute);
        auto* copy = context.device.getQueue(render::QueueType::Copy);
        if (compute == nullptr || copy == nullptr) { return RhiTestResult::skip("compute/copy queue unavailable"); }
        registerFrameGraphTransferPass();
        render::RenderGraph graph;
        graph.addNode("FrameGraphTransferPass", "Upload", {{"queue", "copy"}});
        graph.addNode("FrameGraphTransferPass", "Compute", {{"queue", "compute"}, {"copy", true}});
        graph.addNode("FrameGraphTransferPass", "Graphics", {{"copy", true}});
        graph.addNode("FrameGraphTransferPass", "FanOut", {{"queue", "copy"}, {"copy", true}});
        graph.addNode("FrameGraphHistoryPass", "History");
        graph.addNode("TriangleRasterPass", "Triangle");
        graph.addNode("CopyColorPass", "TextureCopy");
        graph.addEdge("Upload.data", "Compute.source");
        graph.addEdge("Compute.data", "Graphics.source");
        graph.addEdge("Compute.data", "FanOut.source");
        graph.addEdge("Triangle.color", "TextureCopy.source");
        graph.addEdge("Upload.data", "History.source");
        graph.markOutput("Graphics.data");
        graph.markOutput("FanOut.data");
        graph.markOutput("TextureCopy.color");
        graph.markOutput("History.data");
        render::HistoryResourceManager history;
        FRAME_REQUIRE(history.initialize(context.device));
        render::RenderGraphExecutor executor;
        std::string log;
        FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        DeviceDrain drain{context.device};
        const render::RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
            .computeQueue = compute, .copyQueue = copy, .historyResources = &history};
        for (uint32_t index = 0; index < 6; ++index) { FRAME_REQUIRE(executor.execute(submit)); }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        for (const auto* name : {"Graphics.data", "FanOut.data"}) {
            std::array<uint32_t, 4> actual{};
            if (!readWords(*executor.outputResource(name)->buffer, actual.data(), actual.size()) ||
                actual != std::array<uint32_t, 4>{105, 106, 107, 108}) {
                return RhiTestResult::fail("cross-queue dependency chain/fan-out copied stale data");
            }
        }
        std::array<uint32_t, 4> previous{};
        if (!readWords(*executor.outputResource("History.data")->buffer, previous.data(), previous.size()) ||
            previous != std::array<uint32_t, 4>{104, 105, 106, 107}) {
            return RhiTestResult::fail("self-submitted history was not advanced/ordered across frames");
        }
        // Switching to caller-owned graphics commands requires a drain and an
        // acquire barrier for graph outputs last used on the copy queue.
        Commands readback;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::Buffer> pixels;
        FRAME_REQUIRE(readback.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createBuffer({.size = 16 * 16 * 4,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}, pixels));
        FRAME_REQUIRE(readback.begin(6));
        FRAME_REQUIRE(executor.transitionOutput(*readback.buffer, "TextureCopy.color", render::ResourceState::TransferSource));
        readback.buffer->copyTextureToBuffer({.texture = executor.outputResource("TextureCopy.color")->texture,
            .buffer = pixels.get(), .width = 16, .height = 16});
        FRAME_REQUIRE(readback.submit(tracker));
        FRAME_REQUIRE(readback.frame.wait(kWaitTimeout));
        std::array<uint32_t, 256> image{};
        if (!readWords(*pixels, image.data(), image.size()) || image[0] == image[136]) {
            return RhiTestResult::fail("graphics-to-copy texture transition lost rendered contents");
        }
        // A recording failure must cancel the new slot and require recompilation,
        // without replacing a previously returned successful completion point.
        const auto good = executor.lastSubmittedCompletion();
        graph.addNode("FrameGraphTransferPass", "Failure", {{"fail", true}});
        graph.markOutput("Failure.data");
        FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        if (executor.execute(submit) || executor.compiled() || !executor.lastSubmittedCompletion().sameSubmission(good)) {
            return RhiTestResult::fail("failed graph recording remained executable or published a false completion");
        }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        FRAME_REQUIRE(executor.compile(context.device, render::RenderGraph::createDefaultTriangleGraph(), 16, 16, log));
        FRAME_REQUIRE(executor.execute(submit));
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(FrameMultiQueueCompletionTest);
METALLIC_REGISTER_RHI_TEST(FrameSelfSubmitTwoSlotTest);
METALLIC_REGISTER_RHI_TEST(FrameCrossQueueGraphTest);

METALLIC_REGISTER_RHI_TEST(FrameTwoSlotGraphTest);
METALLIC_REGISTER_RHI_TEST(FrameCompletionLifecycleTest);
METALLIC_REGISTER_RHI_TEST(FrameUploadLifetimeTest);
METALLIC_REGISTER_RHI_TEST(FrameDescriptorSnapshotTest);
METALLIC_REGISTER_RHI_TEST(FrameHistoryDependencyTest);

#undef FRAME_REQUIRE
} // namespace
} // namespace metallic::tests
