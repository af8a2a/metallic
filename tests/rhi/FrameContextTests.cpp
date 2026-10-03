#include "Runtime/Render/Streamer/UploadStreamer.h"
#include <stdexcept>
#include <string>

#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/GraphEvidence.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

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
    const render::Result<> frameResult = (expression); \
    if (!frameResult) { return RHITestResult::fail(std::string(#expression) + ": " + toString(frameResult)); } \
} while (false)

constexpr uint64_t kWaitTimeout = 5'000'000'000ull;

struct ScopedTimelineWaitResult {
    PFN_vkWaitSemaphores& entry;
    PFN_vkWaitSemaphores original;
    static inline VkResult result = VK_SUCCESS;
    static inline uint32_t calls = 0;
    static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice, const VkSemaphoreWaitInfo*, uint64_t)
    {
        ++calls;
        return result;
    }
    explicit ScopedTimelineWaitResult(render::Device& device)
        : entry(const_cast<VolkDeviceTable*>(render::vulkan::nativeDevice(device).functions)->vkWaitSemaphores),
          original(entry)
    {
        // Quiescent, test-only fault injection into this device's dispatch.
        calls = 0;
        entry = wait;
    }
    ~ScopedTimelineWaitResult() { entry = original; }
};

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
    render::Result<> initialize(render::Device& device, render::Queue& queue)
    {
        render::Result<> result = device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        return result ? pool->createCommandBuffer().transform([&](auto rhiValue) { buffer = std::move(rhiValue); }) : result;
    }
    render::Result<> begin(uint64_t index)
    {
        render::Result<> result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? buffer->begin(frame.submissionContext()) : result;
    }
    render::Result<> submit(render::QueueSubmissionTracker& tracker, render::Semaphore* gate = nullptr)
    {
        render::Result<> result = buffer->end();
        if (!result) { return result; }
        render::CommandBuffer* buffers[] = {buffer.get()};
        render::SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        return tracker.submit(render::QueueSubmitDesc{
            .waitSemaphores = {gate != nullptr ? &wait : nullptr, gate != nullptr ? 1u : 0u},
            .commandBuffers = {buffers, 1},
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
        .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
    };
    if (auto commandResult = commandBuffer.synchronize({.buffers = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
}

class FrameCompletionLifecycleTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"frame.completion.transaction.contract"}, bench::Layer::RenderGraph, "async", "sync");
    }

    FrameCompletionLifecycleTest() { type = RHITestType::Command; name = "frame_completion_lifecycle"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands;
        render::DeferredReleaseQueue deferred;
        render::RenderSubsystemHost host;
        render::RenderFrameContext abandonedSlot(1);
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(0));
        const render::GPUCompletionPoint cancelled = commands.frame.completion();
        if (cancelled.isComplete() || cancelled.isSubmitted() || cancelled.wait(0)) {
            return RHITestResult::fail("recording was reported as completed/waitable");
        }
        if (tracker.submit({.signalSemaphores = std::array<render::SemaphoreSubmitDesc, 1>{}}, commands.frame) || cancelled.isSubmitted()) {
            return RHITestResult::fail("invalid submission committed a completion point");
        }
        FRAME_REQUIRE(commands.pool->reset());
        commands.frame.cancel();
        if (!cancelled.isCancelled() || !cancelled.isComplete()) {
            return RHITestResult::fail("cancelled recording did not retire");
        }

        FRAME_REQUIRE(commands.frame.begin(1));
        render::CommandBuffer* staleBuffers[] = {commands.buffer.get()};
        if (tracker.submit({.commandBuffers = {staleBuffers, 1}}, commands.frame)) {
            return RHITestResult::fail("cancelled command recording was accepted by a new frame generation");
        }
        FRAME_REQUIRE(commands.buffer->begin(commands.frame.submissionContext()));
        const render::GPUCompletionPoint point = commands.frame.completion();
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
            return RHITestResult::fail("cancelling a replacement released a resource still used by another slot");
        }
        deferred.collect();
        if (!point.isSubmitted() || point.isComplete() || point.value() != 1 ||
            commands.frame.begin(2, 0) || retainedWeak.expired() || retiredWeak.expired() || outsideWeak.expired()) {
            return RHITestResult::fail("pending submission allowed reuse or premature release");
        }
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(point.wait(kWaitTimeout));
        deferred.collect();
        if (!retiredWeak.expired() || retainedWeak.expired()) {
            return RHITestResult::fail("completion retirement or frame retention was incorrect");
        }
        FRAME_REQUIRE(commands.begin(2));
        FRAME_REQUIRE(host.beginFrame(2, 0, nullptr, log, &commands.frame));
        host.endFrame();
        if (!retainedWeak.expired() || !outsideWeak.expired() || !replacedWeak.expired() || !point.isComplete()) {
            return RHITestResult::fail("completed resources were not reclaimed on reuse");
        }
        FRAME_REQUIRE(commands.pool->reset());
        commands.frame.cancel();
        host.shutdown();
        return RHITestResult::pass();
    }
};

class FrameDeviceLostCleanupTest final : public RHITest {
public:
    FrameDeviceLostCleanupTest() { type = RHITestType::Command; name = "frame_device_lost_cleanup"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands;
        render::DeferredReleaseQueue deferred;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(0));
        auto retained = std::make_shared<uint32_t>(17);
        auto retired = std::make_shared<uint32_t>(23);
        const std::weak_ptr<uint32_t> retainedWeak = retained, retiredWeak = retired;
        commands.frame.retain(std::move(retained));
        deferred.retire(commands.frame.completion(), std::move(retired));
        FRAME_REQUIRE(commands.submit(tracker, gate.get()));
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
        FRAME_REQUIRE(commands.pool->reset());

        // Drain actual GPU work first; inject only the host API result so the
        // test covers terminal teardown without deliberately faulting the GPU.
        ScopedTimelineWaitResult injected(context.device);
        ScopedTimelineWaitResult::result = VK_TIMEOUT;
        if (commands.frame.reset() || tracker.reset() || deferred.drain() ||
            !commands.frame.completion().valid() || retainedWeak.expired() || retiredWeak.expired()) {
            return RHITestResult::fail("Retryable wait failure discarded pending lifetimes");
        }
        ScopedTimelineWaitResult::result = VK_ERROR_DEVICE_LOST;
        if (!render::hasError(commands.frame.reset(), render::Error::DeviceLost) ||
            !render::hasError(tracker.reset(), render::Error::DeviceLost) ||
            !render::hasError(deferred.drain(), render::Error::DeviceLost) ||
            commands.frame.completion().valid() || deferred.size() != 0 ||
            !retainedWeak.expired() || !retiredWeak.expired()) {
            return RHITestResult::fail("Device loss did not release lifetimes while preserving the error");
        }
        const uint32_t waits = ScopedTimelineWaitResult::calls;
        FRAME_REQUIRE(commands.frame.reset());
        FRAME_REQUIRE(tracker.reset());
        FRAME_REQUIRE(deferred.drain());
        if (ScopedTimelineWaitResult::calls != waits) {
            return RHITestResult::fail("Repeated teardown accessed a discarded timeline");
        }
        return RHITestResult::pass("Timeout preserves resources; device loss releases them and teardown is idempotent");
    }
};

METALLIC_REGISTER_RHI_TEST(FrameDeviceLostCleanupTest);

class FrameUploadLifetimeTest : public RHITest {
public:
    FrameUploadLifetimeTest() { type = RHITestType::Resource; name = "frame_upload_lifetime"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands;
        render::RenderFrameContext conflictingSlot;
        std::unique_ptr<render::Streamer> streamer;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(createStreamer(context.device, render::StreamerDesc{
            .constantBufferSize = 4096,
            .dynamicBufferSizePerFrame = 64 * 1024,
            .queuedFrameCount = 1,
        }).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        constexpr uint32_t kLargeWordCount = 20'000;
        FRAME_REQUIRE(context.device.createBuffer(render::BufferDesc{
            .size = (kLargeWordCount + 1ull) * sizeof(uint32_t),
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback,
        }).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(0));
        FRAME_REQUIRE(streamer->beginFrame(commands.frame));
        std::vector<uint32_t> constant(1024, 0x12345678);
        if (streamer->streamConstantData(constant.data(), 4096) != 0 ||
            streamer->streamConstantData(constant.data(), 4) != UINT64_MAX) {
            return RHITestResult::fail("constant arena overwrote an earlier allocation when full");
        }
        const uint32_t first = 0x11223344;
        std::vector<uint32_t> large(kLargeWordCount, 0x55667788);
        render::StreamDataChunk chunk{.data = &first, .size = sizeof(first)};
        auto firstUpload = streamer->streamBufferData({
            .dataChunks = {&chunk, 1},
            .dstBuffer = output.get(),
        });
        chunk = {.data = large.data(), .size = large.size() * sizeof(uint32_t)};
        auto secondUpload = streamer->streamBufferData({
            .dataChunks = {&chunk, 1},
            .dstBuffer = output.get(),
            .dstOffset = sizeof(first),
        });
        if (firstUpload.buffer == nullptr || secondUpload.buffer == nullptr || firstUpload.buffer == secondUpload.buffer) {
            return RHITestResult::fail("upload growth did not create the expected old/new allocations");
        }
        if (auto commandResult = streamer->copyStreamedData(*commands.buffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
        streamer->endFrame();
        FRAME_REQUIRE(commands.submit(tracker, gate.get()));
        FRAME_REQUIRE(conflictingSlot.begin(1));
        if (streamer->beginFrame(conflictingSlot)) {
            return RHITestResult::fail("upload arena allowed reuse of an incomplete slot");
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
            return RHITestResult::fail("pending upload data did not survive buffer growth/endFrame/destruction");
        }
        return RHITestResult::pass();
    }
};

class FrameUploadGrowthBurstTest : public RHITest {
public:
    FrameUploadGrowthBurstTest() { type = RHITestType::Resource; name = "frame_upload_growth_burst"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::QueueSubmissionTracker tracker;
        Commands commands(1);
        std::unique_ptr<render::Streamer> streamer;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(createStreamer(context.device, {.dynamicBufferSizePerFrame = 64 * 1024,
            .queuedFrameCount = 2}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        constexpr uint32_t kPages = 64, kWordsPerPage = 5 * 1024;
        constexpr uint64_t kBytes = uint64_t(kPages) * kWordsPerPage * sizeof(uint32_t);
        FRAME_REQUIRE(context.device.createBuffer({.size = kBytes,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        QueueDrain drain{context.graphicsQueue, gate.get()};
        FRAME_REQUIRE(commands.begin(1));
        FRAME_REQUIRE(streamer->beginFrame(commands.frame));
        std::vector<uint32_t> expected(kPages * kWordsPerPage);
        render::Buffer* previous = nullptr;
        uint32_t allocations = 0;
        uint64_t allocatedBytes = 0;
        for (uint32_t page = 0; page < kPages; ++page) {
            auto* words = expected.data() + page * kWordsPerPage;
            std::fill_n(words, kWordsPerPage, 0x12340000u + page);
            render::StreamDataChunk chunk{.data = words, .size = kWordsPerPage * sizeof(uint32_t)};
            const auto upload = streamer->streamBufferData({
                .dataChunks = {&chunk, 1},
                .dstBuffer = output.get(),
                .dstOffset = uint64_t(page) * chunk.size,
            });
            if (!upload.valid()) { return RHITestResult::fail("burst upload failed"); }
            if (upload.buffer != previous) {
                ++allocations;
                allocatedBytes += streamer->stats().dynamicBufferSizePerFrame * 2;
                previous = upload.buffer;
            }
        }
        const auto capacity = streamer->stats().dynamicBufferSizePerFrame;
        if (allocations > 8 || capacity < kBytes || capacity >= 2 * kBytes || allocatedBytes >= capacity * 4) {
            return RHITestResult::fail("burst uploads amplified retained staging allocations");
        }
        if (auto commandResult = streamer->copyStreamedData(*commands.buffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
        streamer->endFrame();
        FRAME_REQUIRE(commands.submit(tracker, gate.get()));
        for (int index = 0; index < 8; ++index) { streamer->endFrame(); }
        streamer.reset();
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
        std::vector<uint32_t> actual(expected.size());
        if (!readWords(*output, actual.data(), actual.size()) || actual != expected) {
            return RHITestResult::fail("slot 1 burst copies lost data across staging growth/destruction");
        }
        // Reject capacities whose alignment or queued-frame multiplication
        // would overflow before attempting any Vulkan allocation.
        if (createStreamer(context.device, {.dynamicBufferSizePerFrame = UINT64_MAX,
                .queuedFrameCount = 2}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); })) {
            return RHITestResult::fail("overflowing staging capacity was accepted");
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(FrameUploadGrowthBurstTest);

render::Result<> createProbe(render::Device& device, const char* entry,
    std::span<const render::ComputeProgramBindingDesc> bindings, render::ComputeProgram& program, std::string& log)
{
    render::ShaderCompileResult shader;
    render::Result<> result = render::compileSlangShaderToSpirv({
        .moduleName = "FrameResourceProbe", .entryPointName = entry,
        .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
    }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
    if (!result) { log = shader.diagnostics; return result; }
    return program.initialize(device, {
        .spirv = shader.spirv,
        .pushConstantSize = sizeof(uint32_t),
        .bindings = bindings,
        .requiresRayQuery = false,
    }, log);
}

class FrameDescriptorSnapshotTest : public RHITest {
public:
    FrameDescriptorSnapshotTest() { type = RHITestType::Rendering; name = "frame_descriptor_snapshots"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> setup = render::createDevice({.applicationName = "Frame descriptors",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!setup && render::hasError(setup, render::Error::Unsupported)) { return RHITestResult::skip("descriptor heap unsupported"); }
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
        FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        const std::array<uint32_t, 3> expected{17, 83, 197};
        for (size_t index = 0; index < inputs.size(); ++index) {
            FRAME_REQUIRE(device->createBuffer({.size = 4, .structureStride = 4,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { inputs[index] = std::move(rhiValue); }));
            void* mapped = inputs[index]->map();
            if (mapped == nullptr) { return RHITestResult::fail("input map failed"); }
            std::memcpy(mapped, &expected[index], 4);
            inputs[index]->flush(); inputs[index]->unmap();
        }
        FRAME_REQUIRE(device->createBuffer({.size = 12, .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
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
            FRAME_REQUIRE(program.dispatch({
                .commandBuffer = commands.buffer.get(),
                .bindings = {resources, 2},
                .pushData = &index,
                .pushDataSize = 4,
            }));
        }
        FRAME_REQUIRE(first.submit(tracker, gate.get()));
        FRAME_REQUIRE(second.submit(tracker));
        program.clear();
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(second.frame.wait(kWaitTimeout));
        std::array<uint32_t, 3> actual{};
        if (!readWords(*output, actual.data(), actual.size()) || actual != expected) {
            return RHITestResult::fail("dispatch descriptors were overwritten within/across frames or pipeline clear invalidated work");
        }
        return RHITestResult::pass();
    }
};

class FrameSampledImageCacheTest : public RHITest {
public:
    FrameSampledImageCacheTest() { type = RHITestType::Rendering; name = "frame_sampled_image_cache"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto setup = render::createDevice({.applicationName = "Sampled image cache",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!setup && render::hasError(setup, render::Error::Unsupported)) { return RHITestResult::skip("descriptor heap unsupported"); }
        FRAME_REQUIRE(setup);
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        render::QueueSubmissionTracker tracker;
        Commands commands, pending(1);
        render::ComputeProgram program, secondProgram;
        std::unique_ptr<render::Buffer> output;
        std::unique_ptr<render::Semaphore> gate;
        struct Images {
            std::array<std::unique_ptr<render::Texture>, 3> textures;
            std::array<std::shared_ptr<render::TextureView>, 3> views;
        };
        auto images = std::make_shared<Images>();
        FRAME_REQUIRE(tracker.initialize(*device, queue));
        FRAME_REQUIRE(commands.initialize(*device, queue));
        FRAME_REQUIRE(pending.initialize(*device, queue));
        FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(device->createBuffer({.size = 8 * sizeof(uint32_t), .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        for (size_t i = 0; i < 3; ++i) {
            render::TextureDesc desc;
            desc.width = 1; desc.height = 1;
            desc.format = render::Format::RGBA32Sfloat;
            desc.usage = render::TextureUsageBits::Sampled | render::TextureUsageBits::TransferDestination;
            FRAME_REQUIRE(device->createTexture(desc).transform([&](auto rhiValue) { images->textures[i] = std::move(rhiValue); }));
            std::unique_ptr<render::TextureView> view;
            FRAME_REQUIRE(device->createTextureView(*images->textures[i], {}).transform([&](auto rhiValue) { view = std::move(rhiValue); }));
            images->views[i] = std::move(view);
        }
        auto a = std::make_shared<const render::ComputeSampledImageSnapshot>(render::ComputeSampledImageSnapshot{
            images, {images->views[0], images->views[1]}});
        auto b = std::make_shared<const render::ComputeSampledImageSnapshot>(render::ComputeSampledImageSnapshot{
            images, {images->views[0], images->views[2]}});
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::SampledImage, .descriptorCount = 2},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer},
        };
        std::string log;
        FRAME_REQUIRE(createProbe(*device, "sampleImages", bindings, program, log));
        FRAME_REQUIRE(createProbe(*device, "sampleImages", bindings, secondProgram, log));
        QueueDrain drain{queue, gate.get()};
        // A raw array and a second program reuse the same canonical descriptors.
        const std::array<uint32_t, 6> writes{2, 0, 1, 0, 0, 0};
        const std::array<uint32_t, 8> expected{30, 30, 40, 30, 30, 40, 30, 40};
        for (uint32_t i = 0; i < 6; ++i) {
            FRAME_REQUIRE(commands.begin(i));
            if (i == 0) {
                for (size_t j = 0; j < 3; ++j) {
                    render::TextureBarrierDesc barrier{
                        .texture = images->textures[j].get(),
                        .oldLayout = render::TextureLayout::Undefined,
                        .newLayout = render::TextureLayout::TransferDestination,
                        .before = {},
                        .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                    };
                    if (auto commandResult = commands.buffer->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    commands.buffer->clearColorTexture(*images->textures[j], render::ResourceState::TransferDestination,
                        {float((j + 1) * 10), 0, 0, 0});
                    barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite};
                    barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
                    if (auto commandResult = commands.buffer->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                }
            }
            storageBarrier(*commands.buffer, *output);
            render::TextureView* raw[] = {images->views[0].get(), images->views[1].get()};
            const render::ComputeDispatchBinding resources[] = {
                {
                    .binding = 0,
                    .textureViews = {raw, 2},
                    .sampledImages = i == 3 ? nullptr : ((i == 2 || i == 5) ? b : a),
                },
                {.binding = 1, .buffer = output.get()},
            };
            render::ComputeDispatchStats stats;
            FRAME_REQUIRE((i == 4 ? secondProgram : program).dispatch({
                .commandBuffer = commands.buffer.get(),
                .bindings = {resources, 2},
                .pushData = &i,
                .pushDataSize = 4,
                .stats = &stats,
            }));
            if (stats.sampledImageWrites != writes[i] || stats.sampledImageCacheHits != 2 - writes[i]) {
                return RHITestResult::fail("incorrect sampled-image generation reuse at step " + std::to_string(i));
            }
            FRAME_REQUIRE(commands.submit(tracker));
            FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
            FRAME_REQUIRE(commands.frame.reset());
        }
        // Keep two generations in flight and release the caller's ownership
        // before execution. Cached weak entries must not own retired images.
        for (uint32_t i = 6; i < 8; ++i) {
            auto& frame = i == 6 ? commands : pending;
            FRAME_REQUIRE(frame.begin(i));
            storageBarrier(*frame.buffer, *output);
            const render::ComputeDispatchBinding resources[] = {
                {.binding = 0, .sampledImages = i == 6 ? a : b}, {.binding = 1, .buffer = output.get()},
            };
            FRAME_REQUIRE(program.dispatch({
                .commandBuffer = frame.buffer.get(),
                .bindings = {resources, 2},
                .pushData = &i,
                .pushDataSize = 4,
            }));
            FRAME_REQUIRE(frame.submit(tracker, i == 6 ? gate.get() : nullptr));
        }
        std::weak_ptr<Images> lifetime = images;
        images.reset(); a.reset(); b.reset();
        if (lifetime.expired()) { return RHITestResult::fail("in-flight sampled images released early"); }
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(pending.frame.wait(kWaitTimeout));
        FRAME_REQUIRE(commands.frame.reset());
        FRAME_REQUIRE(pending.frame.reset());
        if (!lifetime.expired()) { return RHITestResult::fail("completed descriptor cache retains retired images"); }
        std::array<uint32_t, 8> actual{};
        if (!readWords(*output, actual.data(), actual.size()) || actual != expected) {
            return RHITestResult::fail("sampled descriptors returned stale/overwritten image values");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(FrameSampledImageCacheTest);

class FrameHistoryDependencyTest : public RHITest {
public:
    FrameHistoryDependencyTest() { type = RHITestType::Rendering; name = "frame_history_dependencies"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        render::Result<> setup = render::createDevice({.applicationName = "Frame history",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!setup && render::hasError(setup, render::Error::Unsupported)) { return RHITestResult::skip("descriptor heap unsupported"); }
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
        FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(device->createBuffer({.size = 12, .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        render::TextureDesc textureDesc;
        textureDesc.usage = render::TextureUsageBits::Storage;
        textureDesc.width = 1;
        textureDesc.height = 1;
        textureDesc.format = render::Format::RGBA32Sfloat;
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
            FRAME_REQUIRE(program.dispatch({
                .commandBuffer = commands.buffer.get(),
                .bindings = {resources, 3},
                .pushData = &index,
                .pushDataSize = 4,
            }));
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
            return RHITestResult::fail("General history dependencies or resized history lifetime failed");
        }
        return RHITestResult::pass();
    }
};

class FrameTwoSlotGraphTest : public RHITest {
public:
    FrameTwoSlotGraphTest() { type = RHITestType::Rendering; name = "frame_two_slot_graph_reuse"; }
    RHITestResult run(RHITestContext& context) override
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
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { rebuildGate = std::move(rhiValue); }));
        FRAME_REQUIRE(createStreamer(context.device, {.dynamicBufferSizePerFrame = 64 * 1024,
            .queuedFrameCount = 2}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        FRAME_REQUIRE(context.device.createBuffer({.size = 6 * sizeof(uint32_t),
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { uploadReadback = std::move(rhiValue); }));
        for (auto& image : imageReadbacks) {
            FRAME_REQUIRE(context.device.createBuffer({.size = kPixelCount * sizeof(uint32_t),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { image = std::move(rhiValue); }));
        }
        render::RenderGraphCompileOptions options;
        options.enablePreviewOutputAccess = true;
        FRAME_REQUIRE(host.initialize(context.device, 2, log));
        FRAME_REQUIRE(executor.compile(context.device, graph, kWidth, kWidth, options, log));
        if (host.frameSlotCount() != 2) {
            return RHITestResult::fail("graph compile replaced the caller's two-slot capacity");
        }
        QueueDrain drain{context.graphicsQueue, gate.get(), rebuildGate.get()};
        GateWatchdog watchdog(*gate);
        render::GPUCompletionPoint firstPoint;
        for (uint32_t index = 0; index < 6; ++index) {
            Commands& commands = slots[index % slots.size()];
            FRAME_REQUIRE(commands.begin(index));
            FRAME_REQUIRE(streamer->beginFrame(commands.frame));
            const uint32_t value = 100 + index;
            const render::StreamDataChunk chunk{.data = &value, .size = sizeof(value)};
            const auto upload = streamer->streamBufferData({
                .dataChunks = {&chunk, 1},
                .dstBuffer = uploadReadback.get(),
                .dstOffset = index * sizeof(value),
            });
            if (upload.buffer == nullptr) { return RHITestResult::fail("slot upload allocation failed"); }
            if (auto commandResult = streamer->copyStreamedData(*commands.buffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
            streamer->endFrame();
            FRAME_REQUIRE(executor.execute(*commands.buffer));
            if (executor.streamingStats().streamer.queuedFrameCount != 2) {
                return RHITestResult::fail("graph uploader did not use the host's two-slot capacity");
            }
            if (index == 1 && firstPoint.isComplete()) {
                return RHITestResult::fail("recording slot 1 waited for slot 0 instead of overlapping it");
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
                    return RHITestResult::fail("two-slot ring reused an incomplete slot");
                }
                FRAME_REQUIRE(gate->signal(1));
                watchdog.worker.request_stop();
            }
        }
        if (executor.waitForSubmittedWork(0) || slots[1].frame.completion().isComplete()) {
            return RHITestResult::fail("graph did not track its externally submitted pending work");
        }
        FRAME_REQUIRE(rebuildGate->signal(1));
        // Rebuild must wait before replacing graph images, passes and query pools.
        FRAME_REQUIRE(executor.compile(context.device, graph, 48, 48, options, log));
        if (!firstPoint.isComplete() || firstPoint.value() != 1 ||
            !slots[1].frame.completion().isComplete() || slots[1].frame.completion().value() != 6) {
            return RHITestResult::fail("completion generations were not preserved through slot reuse/rebuild");
        }
        std::array<uint32_t, 6> words{};
        if (!readWords(*uploadReadback, words.data(), words.size()) ||
            words != std::array<uint32_t, 6>{100, 101, 102, 103, 104, 105}) {
            return RHITestResult::fail("upload data was overwritten while alternating two slots");
        }
        std::array<uint32_t, kPixelCount> reference{}, pixels{};
        if (!readWords(*imageReadbacks[0], reference.data(), reference.size()) ||
            reference[0] == reference[kPixelCount / 2 + kWidth / 2]) {
            return RHITestResult::fail("triangle readback did not contain rendered geometry");
        }
        for (uint32_t index : {2u, 4u, 5u}) {
            if (!readWords(*imageReadbacks[index], pixels.data(), pixels.size()) || pixels != reference) {
                return RHITestResult::fail("overlapped graph rendering differed after slot reuse");
            }
        }
        return RHITestResult::pass();
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

class FrameMultiQueueCompletionTest : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        auto metadata = bench::gpuMetadata({"frame.partialSubmit.contract"}, bench::Layer::RenderGraph, "async", "sync");
        metadata.requirements.capabilities.push_back(bench::Capability::IndependentCopy);
        metadata.requirements.queues.push_back(render::QueueType::Copy);
        return metadata;
    }

    FrameMultiQueueCompletionTest() { type = RHITestType::Command; name = "frame_multi_queue_completion"; }
    RHITestResult run(RHITestContext& context) override
    {
        auto* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (copyQueue == nullptr) { return RHITestResult::skip("independent copy queue unavailable"); }
        render::QueueSubmissionTracker graphics, copy;
        render::RenderFrameContext frame;
        render::DeferredReleaseQueue retired;
        std::unique_ptr<render::Semaphore> graphicsGate, copyGate;
        FRAME_REQUIRE(graphics.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(copy.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { graphicsGate = std::move(rhiValue); }));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { copyGate = std::move(rhiValue); }));
        DeviceDrain drain{context.device, graphicsGate.get(), copyGate.get()};
        FRAME_REQUIRE(frame.begin(0));
        auto resource = std::make_shared<uint32_t>(17);
        const std::weak_ptr<uint32_t> weak = resource;
        frame.retain(resource);
        retired.retire(frame.completion(), std::move(resource));
        const auto batch = frame.completion();
        render::GPUCompletionPoint graphicsPoint, copyPoint;
        render::SemaphoreSubmitDesc graphicsWait{.semaphore = graphicsGate.get(), .value = 1};
        render::SemaphoreSubmitDesc copyWait{.semaphore = copyGate.get(), .value = 1};
        FRAME_REQUIRE(graphics.submitSegment({.waitSemaphores = {&graphicsWait, 1}}, frame).transform([&](auto value) { graphicsPoint = std::move(value); }));
        FRAME_REQUIRE(copy.submitSegment({.waitSemaphores = {&copyWait, 1}}, frame).transform([&](auto value) { copyPoint = std::move(value); }));
        std::vector<render::SemaphoreSubmitDesc> waits;
        if (batch.isSubmitted() || batch.isComplete() || batch.wait(0) || batch.appendWaits(waits) || frame.begin(1, 0)) {
            return RHITestResult::fail("open submission batch was reusable or waitable");
        }
        // Simulate a later submission rejected before reaching the driver.
        render::GPUCompletionPoint failed;
        if (copy.submitSegment({.commandBuffers = std::array<render::CommandBuffer*, 1>{nullptr}}, frame).transform([&](auto value) { failed = std::move(value); }) || failed.valid()) {
            return RHITestResult::fail("failed segment acquired a completion value");
        }
        frame.cancel();
        if (!batch.isSubmitted() || batch.isCancelled() || batch.value() != 0) {
            return RHITestResult::fail("partial multi-queue batch was cancelled instead of sealed");
        }
        FRAME_REQUIRE(batch.appendWaits(waits));
        FRAME_REQUIRE(batch.appendWaits(waits));
        if (waits.size() != 2) { return RHITestResult::fail("composite waits were not coalesced per queue"); }
        FRAME_REQUIRE(graphicsGate->signal(1));
        FRAME_REQUIRE(graphicsPoint.wait(kWaitTimeout));
        retired.collect();
        if (batch.isComplete() || batch.wait(0) || copyPoint.isComplete() || weak.expired()) {
            return RHITestResult::fail("graphics completion prematurely retired copy-queue resources");
        }
        FRAME_REQUIRE(copyGate->signal(1));
        FRAME_REQUIRE(batch.wait(kWaitTimeout));
        FRAME_REQUIRE(frame.begin(1));
        retired.collect();
        if (!weak.expired() || !batch.isComplete()) {
            return RHITestResult::fail("completed partial batch failed to release resources");
        }
        // Two signals from one queue collapse to the final value, while old
        // segment points continue identifying their original submission.
        FRAME_REQUIRE(copy.submitSegment({}, frame).transform([&](auto value) { copyPoint = std::move(value); }));
        const auto oldCopy = copyPoint;
        FRAME_REQUIRE(copy.submitSegment({}, frame).transform([&](auto value) { copyPoint = std::move(value); }));
        FRAME_REQUIRE(frame.finishSubmission());
        FRAME_REQUIRE(frame.wait(kWaitTimeout));
        if (oldCopy.value() != 2 || copyPoint.value() != 3 || frame.completion().value() != 3) {
            return RHITestResult::fail("failed segment consumed a timeline value or rewrote old completion");
        }
        return RHITestResult::pass();
    }
};

std::vector<int> asyncBranchEvents;
class FrameParallelBranchPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(16).transferWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        const auto result = context.device->createBuffer({.size = 16,
            .usage = render::BufferUsageBits::TransferSource, .memoryLocation = render::MemoryLocation::HostUpload,
            .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}).transform([&](auto rhiValue) { input_ = std::move(rhiValue); });
        if (!result) { return result; }
        const uint32_t words[] = {11, 12, 21, 22};
        void* mapped = input_->map();
        if (!mapped) { return render::makeError(render::Error::Failure); }
        std::memcpy(mapped, words, sizeof(words)); input_->flush(); input_->unmap();
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto parentProfile = context.profileScope("Fork/join");
        const auto transaction = [](render::CommandBuffer& commands, int id) {
            return commands.addSubmissionTransaction(std::make_shared<render::SubmissionTransaction>(
                [id] { asyncBranchEvents.push_back(id); }, [id] { asyncBranchEvents.push_back(-id); }));
        };
        auto result = transaction(context.commandBuffer(), 1);
        if (!result) { return result; }
        auto* output = context.outputBuffer("data").buffer();
        result = context.parallelCompute([&](render::CommandBuffer& commands) -> metallic::render::Result<> {
            auto profile = context.profileScope(commands, "Compute branch");
            auto result = transaction(commands, 2);
            if (!result) { return result; }
            {
                auto sourceSlice = input_.get()->slice({0, 8});
                if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                auto destinationSlice = output->slice({0, 8});
                if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
            }
            return properties().value("fail", 0) == 2 ? render::makeError(render::Error::Failure) : render::Result<>{};
        }, [&](render::CommandBuffer& commands) -> metallic::render::Result<> {
            auto profile = context.profileScope(commands, "Graphics branch");
            auto result = transaction(commands, 3);
            if (!result) { return result; }
            {
                auto sourceSlice = input_.get()->slice({8, 8});
                if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                auto destinationSlice = output->slice({8, 8});
                if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
            }
            return properties().value("fail", 0) == 3 ? render::makeError(render::Error::Failure) : render::Result<>{};
        });
        return result ? transaction(context.commandBuffer(), 4) : result;
    }
private:
    std::unique_ptr<render::Buffer> input_;
};

class FrameParallelBranchTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.forkJoin.cancel.readback"}, bench::Layer::RenderGraph, "async", "sync");
    }

    FrameParallelBranchTest() { type = RHITestType::Rendering; name = "frame_parallel_compute_join_and_cancellation"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::TestDevice device;
        FRAME_REQUIRE(bench::createTestDevice(context, {.applicationName = "Async branch lifetime regression",
            .enableValidation = context.enableValidation, .enableAsyncCompute = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        auto* compute = device->getQueue(render::QueueType::Compute);
        render::registerRenderGraphPassType("FrameParallelBranchPass", "Parallel branch test",
            [] { return std::make_unique<FrameParallelBranchPass>(); });
        render::RenderGraphExecutor executor;
        std::string log;
        for (bool parallel : {true, false}) {
            for (int fail : {2, 3, 0}) {
                render::RenderGraph graph;
                graph.addNode("FrameParallelBranchPass", "Branches", {{"fail", fail}});
                graph.markOutput("Branches.data");
                FRAME_REQUIRE(executor.compile(*device, graph, 1, 1, log));
                asyncBranchEvents.clear();
                const auto result = executor.execute(render::RenderGraphSubmitDesc{.graphicsQueue = graphics,
                    .computeQueue = parallel ? compute : graphics});
                if (fail != 0) {
                    const std::vector<int> expected = fail == 2 ? std::vector<int>{-2, -1} : std::vector<int>{-3, -2, -1};
                    if (result || executor.compiled() || asyncBranchEvents != expected) {
                        return RHITestResult::fail("Failed branch submitted work or did not roll back the entire recording");
                    }
                    continue;
                }
                FRAME_REQUIRE(result);
                FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
                std::vector<render::RenderGraphExecutionStats> timings;
                FRAME_REQUIRE(executor.collectCompletedGpuExecutionStats().transform([&](auto value) { timings = std::move(value); }));
                if (device->capabilities().timestampQueries) {
                    if (timings.size() != 1 || !timings[0].gpuTimingAvailable || timings[0].nodes.size() != 1 ||
                        timings[0].nodes[0].sections.size() != 4) {
                        return RHITestResult::fail("fork/join timings missing or cancelled recording leaked a sample");
                    }
                    const auto& sections = timings[0].nodes[0].sections;
                    const auto expectedQueue = parallel ? compute->type() : graphics->type();
                    if (sections[1].queue != expectedQueue || sections[2].queue != graphics->type() ||
                        sections[0].parent != UINT32_MAX || sections[1].parent != 0 || sections[2].parent != 0 ||
                        sections[3].name != "Upload flush" || sections[3].parent != UINT32_MAX) {
                        return RHITestResult::fail("fork/join profiling lost queue or parent identity");
                    }
                    for (const auto& section : sections) {
                        if (!section.gpuTimingAvailable || section.gpuMilliseconds < 0 ||
                            section.gpuMilliseconds > timings[0].gpuMilliseconds + 0.01) {
                            return RHITestResult::fail("fork/join section timing unavailable or outside graph envelope");
                        }
                    }
                }
                if (asyncBranchEvents != std::vector<int>{1, 2, 3, 4} || executor.executionStats().asyncComputeBranches !=
                    (parallel && device->capabilities().independentComputeQueue ? 1u : 0u)) {
                    return RHITestResult::fail("Parallel/aliased queue topology or transaction commit order was incorrect");
                }
                Commands consumer;
                render::QueueSubmissionTracker tracker;
                std::unique_ptr<render::Buffer> readback;
                FRAME_REQUIRE(consumer.initialize(*device, *graphics));
                FRAME_REQUIRE(tracker.initialize(*device, *graphics));
                FRAME_REQUIRE(device->createBuffer({.size = 16, .usage = render::BufferUsageBits::TransferDestination,
                    .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
                FRAME_REQUIRE(consumer.begin(0));
                FRAME_REQUIRE(executor.transitionOutput(*consumer.buffer, "Branches.data", render::ResourceState::TransferSource));
                {
                    auto sourceSlice = (executor.outputResource("Branches.data")->buffer)->slice({0, 16});
                    if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                    auto destinationSlice = readback.get()->slice({0, 16});
                    if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                    if (auto commandResult = consumer.buffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                }
                FRAME_REQUIRE(consumer.submit(tracker)); FRAME_REQUIRE(consumer.frame.wait(kWaitTimeout));
                readback->invalidate(); const auto* words = static_cast<const uint32_t*>(readback->map());
                const bool correct = words && words[0] == 11 && words[1] == 12 && words[2] == 21 && words[3] == 22;
                if (words) { bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(words, 4)); }
                readback->unmap();
                if (!correct) { return RHITestResult::fail("Join did not make both branch writes visible"); }
            }
        }
        return RHITestResult::pass("Independent and aliased queues, fork/join visibility, compute/HW recording failure and reverse transaction cancellation");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameParallelBranchTest);

// Diagnostic fixture: immutable pass state, graph-owned image and no private uploads.
class FrameGraphDiagnosticClearPass final : public render::UnsafePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").transferWrite().format = render::Format::RGBA8Unorm;
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        context.commandBuffer().clearColorTexture(*context.outputTexture("color").texture(),
            render::ResourceState::TransferDestination, {0.25f, 0.5f, 0.75f, 1.0f});
        return {};
    }
};

class FrameGraphTransferPass final : public render::RenderGraphPass {
public:
    static inline uint32_t overlapQueryCount = 0;
    // This fixture uses only frame-local streamer data; other tests retain their original mode.
    bool supportsPipelinedSubmission() const override { return properties().value("pipelined", false); }
    bool supportsFrameOverlap() const override { ++overlapQueryCount; return true; }
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
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto profile = context.profileScope("Transfer");
        if (properties().value("fail", false)) { return render::makeError(render::Error::Failure); }
        auto* output = context.outputBuffer("data").buffer();
        if (properties().value("copy", false)) {
            {
                auto sourceSlice = (context.inputBuffer("source").buffer())->slice({0, 16});
                if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                auto destinationSlice = output->slice({0, 16});
                if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                if (auto commandResult = context.commandBuffer().copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
            }
            return {};
        }
        const uint32_t value = 100 + static_cast<uint32_t>(context.frameIndex());
        const std::array<uint32_t, 4> words{value, value + 1, value + 2, value + 3};
        const render::StreamDataChunk chunk{.data = words.data(), .size = sizeof(words)};
        return context.streamer()->streamBufferData({
            .dataChunks = {&chunk, 1},
            .dstBuffer = output,
        }).valid() ? render::Result<>{} : render::makeError(render::Error::Failure);
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
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto* history = context.historyResources();
        if (history == nullptr) { return render::makeError(render::Error::InvalidArgument); }
        render::Result<> result = history->ensureBuffer("self-submit-history", {.size = 16,
            .usage = render::BufferUsageBits::TransferSource | render::BufferUsageBits::TransferDestination});
        if (!result) { return result; }
        result = history->transitionBuffer(context.commandBuffer(), "self-submit-history",
            render::HistorySlot::Current, render::ResourceState::TransferDestination);
        if (!result) { return result; }
        auto* source = context.inputBuffer("source").buffer();
        {
            auto sourceSlice = source->slice({0, 16});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = (history->buffer("self-submit-history", render::HistorySlot::Current).buffer)->slice({0, 16});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = context.commandBuffer().copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
        if (history->hasPrevious("self-submit-history")) {
            result = history->transitionBuffer(context.commandBuffer(), "self-submit-history",
                render::HistorySlot::Previous, render::ResourceState::TransferSource);
            if (!result) { return result; }
            source = history->buffer("self-submit-history", render::HistorySlot::Previous).buffer;
        }
        {
            auto sourceSlice = source->slice({0, 16});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = (context.outputBuffer("data").buffer())->slice({0, 16});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = context.commandBuffer().copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
        history->markWritten("self-submit-history");
        return {};
    }
};

void registerFrameGraphTransferPass()
{
    static const bool registered = [] {
        render::registerRenderGraphPassType("FrameGraphDiagnosticClearPass", "Diagnostic immutable clear",
            [] { return std::make_unique<FrameGraphDiagnosticClearPass>(); });
        render::registerRenderGraphPassType("FrameGraphHistoryPass", "Self submission test history",
            [] { return std::make_unique<FrameGraphHistoryPass>(); });
        render::registerRenderGraphPassType("FrameGraphTransferPass", "Frame submission test transfer",
            [] { return std::make_unique<FrameGraphTransferPass>(); });
        return true;
    }();
    (void)registered;
}

// Hold the output reader on a GPU gate while recording the next producer.
// Checking old frame contents catches missing WAR ordering across queues; a
// zero host timeout catches accidentally serializing CPU recording again.
class FrameOutputConsumerOverlapTest final : public RHITest {
public:
    FrameOutputConsumerOverlapTest() { type = RHITestType::Rendering; name = "frame_output_consumer_gpu_dependencies"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        FRAME_REQUIRE(render::createDevice({.applicationName = "Graph output consumer overlap",
            .enableValidation = context.enableValidation, .enableAsyncCompute = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); }));
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        auto* compute = device->getQueue(render::QueueType::Compute);
        registerFrameGraphTransferPass();
        for (auto* readerQueue : {graphics, compute != nullptr ? compute : graphics}) {
            render::RenderGraph graph;
            graph.addNode("FrameGraphTransferPass", "Output");
            graph.markOutput("Output.data");
            render::RenderGraphExecutor executor;
            render::QueueSubmissionTracker readerTracker;
            Commands reader;
            std::unique_ptr<render::Buffer> readback;
            FRAME_REQUIRE(readerTracker.initialize(*device, *readerQueue));
            FRAME_REQUIRE(reader.initialize(*device, *readerQueue));
            FRAME_REQUIRE(device->createBuffer({.size = 16,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
                .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
            std::string log;
            FRAME_REQUIRE(executor.compile(*device, graph, 1, 1, log));
            const render::RenderGraphSubmitDesc submit{.graphicsQueue = graphics,
                .computeQueue = compute, .slotWaitTimeoutNanoseconds = 0};
            for (uint32_t cycle = 0; cycle < 4; ++cycle) {
                std::unique_ptr<render::Semaphore> gate;
                FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
                DeviceDrain drain{*device, gate.get()};
                GateWatchdog watchdog(*gate);
                FRAME_REQUIRE(executor.execute(submit));
                FRAME_REQUIRE(executor.lastSubmittedCompletion().wait(kWaitTimeout));
                const uint32_t expected = 100 + static_cast<uint32_t>(executor.executionStats().executionId);
                FRAME_REQUIRE(reader.begin(cycle));
                FRAME_REQUIRE(executor.transitionOutput(*reader.buffer, "Output.data", render::ResourceState::TransferSource));
                // Duplicate output requests must not multiply dependencies.
                FRAME_REQUIRE(executor.transitionOutput(*reader.buffer, "Output.data", render::ResourceState::TransferSource));
                {
                    auto sourceSlice = (executor.outputResource("Output.data")->buffer)->slice({0, 16});
                    if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                    auto destinationSlice = readback.get()->slice({0, 16});
                    if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                    if (auto commandResult = reader.buffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                }
                FRAME_REQUIRE(reader.submit(readerTracker, gate.get()));
                const uint32_t queries = FrameGraphTransferPass::overlapQueryCount;
                FRAME_REQUIRE(executor.execute(submit));
                if (FrameGraphTransferPass::overlapQueryCount != queries + 1) {
                    return RHITestResult::fail("Preflight evaluated the same overlap contract more than once");
                }
                if (executor.executionStats().drainReasonMask != 0 || executor.executionStats().externalCompletionCount != 1 ||
                    gate->currentValue() != 0 || reader.frame.completion().isComplete()) {
                    return RHITestResult::fail("External consumer blocked CPU recording or dependency generations accumulated");
                }
                if (executor.lastSubmittedCompletion().wait(100'000'000ull) || executor.waitForSubmittedWork(0)) {
                    return RHITestResult::fail("Graph overwrite/drain ignored the gated output reader");
                }
                FRAME_REQUIRE(gate->signal(1));
                watchdog.worker.request_stop();
                FRAME_REQUIRE(executor.lastSubmittedCompletion().wait(kWaitTimeout));
                FRAME_REQUIRE(reader.frame.wait(kWaitTimeout));
                std::array<uint32_t, 4> actual{};
                if (!readWords(*readback, actual.data(), actual.size()) ||
                    actual != std::array<uint32_t, 4>{expected, expected + 1, expected + 2, expected + 3}) {
                    return RHITestResult::fail("Next graph overwrote the output before its consumer read it");
                }
            }
            // Destructive graph changes must still wait even though recording no
            // longer waits. A delayed signal makes a premature rebuild observable.
            std::unique_ptr<render::Semaphore> gate;
            FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
            DeviceDrain drain{*device, gate.get()};
            FRAME_REQUIRE(reader.begin(4));
            FRAME_REQUIRE(executor.transitionOutput(*reader.buffer, "Output.data", render::ResourceState::TransferSource));
            {
                auto sourceSlice = (executor.outputResource("Output.data")->buffer)->slice({0, 16});
                if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = readback.get()->slice({0, 16});
                if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = reader.buffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            FRAME_REQUIRE(reader.submit(readerTracker, gate.get()));
            if (executor.waitForSubmittedWork(0)) { return RHITestResult::fail("Lost pending consumer before rebuild"); }
            std::jthread release([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                (void)gate->signal(1);
            });
            FRAME_REQUIRE(executor.compile(*device, graph, 2, 2, log));
            if (!reader.frame.completion().isComplete()) {
                return RHITestResult::fail("Rebuild released an output still used by an external consumer");
            }
        }
        return RHITestResult::pass("Same/cross queue output reads, two-slot reuse, GPU WAR ordering and rebuild lifetime");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameOutputConsumerOverlapTest);

class FrameSelfSubmitTwoSlotTest : public RHITest {
public:
    explicit FrameSelfSubmitTwoSlotTest(bool joined = false) : joined_(joined)
    {
        type = RHITestType::Rendering;
        name = joined ? "frame_self_submit_two_slots_joined" : "frame_self_submit_two_slots";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "async", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.capabilities = {bench::Capability::IndependentCopy},
                .queues = {render::QueueType::Graphics, render::QueueType::Copy}},
            .coverage = {"graph.independentCopy.progress", "graph.frameCompletion.join", "graph.backend.diagnostics"}, .artifacts = {"readback.bin", "graph.json"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto* copyQueue = context.device.getQueue(render::QueueType::Copy);
        if (copyQueue == nullptr) { return RHITestResult::skip("independent copy queue unavailable"); }
        registerFrameGraphTransferPass();
        render::RenderGraph graph;
        graph.addNode("FrameGraphDiagnosticClearPass", "Graphics");
        graph.addNode("FrameGraphTransferPass", "Upload", {{"queue", "copy"}, {"pipelined", true}});
        graph.markOutput("Graphics.color");
        graph.markOutput("Upload.data");
        render::RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        Commands blockedGraphics, consumer;
        render::QueueSubmissionTracker blocker, consumerTracker;
        std::unique_ptr<render::Buffer> consumerReadback;
        std::unique_ptr<render::Semaphore> gate;
        FRAME_REQUIRE(blocker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(blockedGraphics.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(consumer.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(consumerTracker.initialize(context.device, *copyQueue));
        FRAME_REQUIRE(context.device.createBuffer({.size = 16,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback,
            .queueAccess = render::QueueAccessBits::Copy}).transform([&](auto rhiValue) { consumerReadback = std::move(rhiValue); }));
        std::string log;
        FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        DeviceDrain drain{context.device, gate.get()};
        GateWatchdog watchdog(*gate);
        FRAME_REQUIRE(blockedGraphics.begin(0));
        FRAME_REQUIRE(blockedGraphics.submit(blocker, gate.get()));
        render::RenderGraphSubmitDesc submit{.graphicsQueue = &context.graphicsQueue,
            .copyQueue = copyQueue, .slotWaitTimeoutNanoseconds = 0,
            .submissionMode = joined_ ? render::FrameSubmissionMode::Joined : render::FrameSubmissionMode::Pipelined};
        FRAME_REQUIRE(executor.execute(submit));
        const auto snapshot = executor.executionSnapshot();
        const uint32_t uploadId = graph.findNode("Upload")->id;
        if (!snapshot) { return RHITestResult::fail("missing independent-queue execution snapshot"); }
        if (snapshot->pipelinedSubmission != !joined_) {
            return RHITestResult::fail("actual graph submission mode differs from the requested test path");
        }
        const auto uploadSegment = std::find_if(snapshot->segments.begin(), snapshot->segments.end(),
            [uploadId](const auto& segment) { return segment.passId == uploadId; });
        if (uploadSegment == snapshot->segments.end() || !uploadSegment->predecessors.empty()) {
            return RHITestResult::fail("independent copy segment acquired an unrelated prologue dependency");
        }
        if (context.device.capabilities().timestampQueries) {
            const auto join = std::find_if(snapshot->segments.begin(), snapshot->segments.end(),
                [](const auto& segment) { return segment.role == render::RenderGraphSegmentRole::Epilogue; });
            if (join == snapshot->segments.end() ||
                std::find(join->predecessors.begin(), join->predecessors.end(), uploadSegment->id) == join->predecessors.end()) {
                return RHITestResult::fail("graph timing epilogue no longer joins the independent copy branch");
            }
        }
        if (const auto error = bench::graphEvidence(context, executor, *snapshot); !error.empty()) {
            return RHITestResult::fail(error);
        }
        const auto first = executor.lastSubmittedCompletion();
        std::vector<render::SemaphoreSubmitDesc> waits;
        FRAME_REQUIRE(first.appendWaits(waits));
        size_t finished = 0;
        for (const auto& wait : waits) {
            if (wait.semaphore->wait(wait.value, 300'000'000ull)) { ++finished; }
        }
        if (waits.size() != 2 || finished != 1 || first.isComplete()) {
            return RHITestResult::fail("independent copy branch was blocked by the graphics branch");
        }
        std::array<uint32_t, 4> actual{};
        if (!readWords(*executor.outputResource("Upload.data")->buffer, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{100, 101, 102, 103}) {
            return RHITestResult::fail("copy-queue upload did not finish while graphics was blocked");
        }
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
        FRAME_REQUIRE(executor.execute(submit));
        const auto second = executor.lastSubmittedCompletion();
        const uint64_t recorded = executor.streamingStats().frameIndex;
        if (first.isComplete() || second.isComplete() || executor.execute(submit) ||
            !executor.compiled() || executor.streamingStats().frameIndex != recorded || executor.waitForSubmittedWork(0)) {
            return RHITestResult::fail("self submission failed two-slot overlap/backpressure contract");
        }
        // Consume the pending graph on an externally recorded command buffer.
        // transitionOutput attaches the aggregate wait to Queue::submit.
        FRAME_REQUIRE(consumer.begin(0));
        FRAME_REQUIRE(executor.transitionOutput(*consumer.buffer, "Upload.data", render::ResourceState::TransferSource));
        FRAME_REQUIRE(addCommandDependency(*consumer.buffer, second)); // Duplicate is coalesced.
        {
            auto sourceSlice = (executor.outputResource("Upload.data")->buffer)->slice({0, 16});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = consumerReadback.get()->slice({0, 16});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = consumer.buffer->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        FRAME_REQUIRE(consumer.submit(consumerTracker));
        if (consumer.frame.completion().isComplete()) {
            return RHITestResult::fail("external consumer ignored pending graph completion");
        }
        FRAME_REQUIRE(gate->signal(1));
        watchdog.worker.request_stop();
        FRAME_REQUIRE(consumer.frame.wait(kWaitTimeout));
        if (!readWords(*consumerReadback, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{101, 102, 103, 104}) {
            return RHITestResult::fail("pending self-to-external handoff copied the wrong frame");
        }
        bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
        submit.slotWaitTimeoutNanoseconds = kWaitTimeout;
        for (uint32_t index = 2; index < 6; ++index) { FRAME_REQUIRE(executor.execute(submit)); }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        if (!first.isComplete() || !second.isComplete() ||
            !readWords(*executor.outputResource("Upload.data")->buffer, actual.data(), actual.size()) ||
            actual != std::array<uint32_t, 4>{105, 106, 107, 108}) {
            return RHITestResult::fail("self-submitted upload data was corrupted through slot reuse");
        }
        FRAME_REQUIRE(executor.compile(context.device, graph, 24, 24, log));
        return RHITestResult::pass();
    }
private:
    bool joined_;
};

class FrameSelfSubmitTwoSlotJoinedTest final : public FrameSelfSubmitTwoSlotTest {
public:
    FrameSelfSubmitTwoSlotJoinedTest() : FrameSelfSubmitTwoSlotTest(true) {}
};

class FrameCrossQueueGraphTest : public RHITest {
public:
    FrameCrossQueueGraphTest() { type = RHITestType::Rendering; name = "frame_cross_queue_graph_dependencies"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "async", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.capabilities = {bench::Capability::IndependentCopy, bench::Capability::IndependentCompute, bench::Capability::TimestampQueries},
                .queues = {render::QueueType::Graphics, render::QueueType::Compute, render::QueueType::Copy},
                .timestampQueues = {render::QueueType::Graphics, render::QueueType::Compute, render::QueueType::Copy}},
            .coverage = {"graph.crossQueue.dependencies", "query.ring.reuse", "history.frame.readback"},
            .artifacts = {"query-ring.json", "history.bin", "texture.bin"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        const auto validationBefore = context.validationMessageCount ? context.validationMessageCount->load() : 0u;
        auto* compute = context.device.getQueue(render::QueueType::Compute);
        auto* copy = context.device.getQueue(render::QueueType::Copy);
        if (compute == nullptr || copy == nullptr) { return RHITestResult::skip("compute/copy queue unavailable"); }
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
        std::vector<render::RenderGraphExecutionStats> timings;
        FRAME_REQUIRE(executor.collectCompletedGpuExecutionStats().transform([&](auto value) { timings = std::move(value); }));
        if (context.evidence) {
            bench::Json frames = bench::Json::array();
            for (const auto& frame : timings) {
                bench::Json nodes = bench::Json::array();
                for (const auto& node : frame.nodes) {
                    nodes.push_back({{"queue", int(node.queue)}, {"available", node.gpuTimingAvailable}, {"milliseconds", node.gpuMilliseconds}});
                }
                frames.push_back({{"available", frame.gpuTimingAvailable}, {"milliseconds", frame.gpuMilliseconds}, {"nodes", nodes}});
            }
            context.evidence->json("query-ring.json", frames);
        }
        if (context.device.capabilities().timestampQueries) {
            if (timings.size() != 6) { return RHITestResult::fail("mixed queue query ring lost completed frames"); }
            for (const auto& frame : timings) {
                if (!frame.gpuTimingAvailable || frame.cpuMilliseconds <= 0) {
                    return RHITestResult::fail("mixed queue graph frame timings missing");
                }
                for (const auto& node : frame.nodes) {
                    const auto* queue = context.device.getQueue(node.queue);
                    if (queue->timestampValidBits() && (!node.gpuTimingAvailable || node.gpuMilliseconds > frame.gpuMilliseconds + 0.01)) {
                        return RHITestResult::fail("mixed queue pass timing missing or outside frame envelope");
                    }
                    for (const auto& section : node.sections) {
                        if (queue->timestampValidBits() && !section.gpuTimingAvailable) {
                            return RHITestResult::fail("mixed queue inner scope timing missing");
                        }
                    }
                }
            }
        }
        for (const auto* name : {"Graphics.data", "FanOut.data"}) {
            std::array<uint32_t, 4> actual{};
            if (!readWords(*executor.outputResource(name)->buffer, actual.data(), actual.size()) ||
                actual != std::array<uint32_t, 4>{105, 106, 107, 108}) {
                return RHITestResult::fail("cross-queue dependency chain/fan-out copied stale data");
            }
        }
        std::array<uint32_t, 4> previous{};
        if (!readWords(*executor.outputResource("History.data")->buffer, previous.data(), previous.size()) ||
            previous != std::array<uint32_t, 4>{104, 105, 106, 107}) {
            return RHITestResult::fail("self-submitted history was not advanced/ordered across frames");
        }
        bench::readbackEvidence(context, "history.bin", std::span<const uint32_t>(previous));
        // Switching to caller-owned graphics commands requires a drain and an
        // acquire barrier for graph outputs last used on the copy queue.
        Commands readback;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::Buffer> pixels;
        FRAME_REQUIRE(readback.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        FRAME_REQUIRE(context.device.createBuffer({.size = 16 * 16 * 4,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { pixels = std::move(rhiValue); }));
        FRAME_REQUIRE(readback.begin(6));
        FRAME_REQUIRE(executor.transitionOutput(*readback.buffer, "TextureCopy.color", render::ResourceState::TransferSource));
        readback.buffer->copyTextureToBuffer({.texture = executor.outputResource("TextureCopy.color")->texture,
            .buffer = pixels.get(), .width = 16, .height = 16});
        FRAME_REQUIRE(readback.submit(tracker));
        FRAME_REQUIRE(readback.frame.wait(kWaitTimeout));
        std::array<uint32_t, 256> image{};
        if (!readWords(*pixels, image.data(), image.size()) || image[0] == image[136]) {
            return RHITestResult::fail("graphics-to-copy texture transition lost rendered contents");
        }
        bench::readbackEvidence(context, "texture.bin", std::span<const uint32_t>(image));
        // A recording failure must cancel the new slot and require recompilation,
        // without replacing a previously returned successful completion point.
        const auto good = executor.lastSubmittedCompletion();
        graph.addNode("FrameGraphTransferPass", "Failure", {{"fail", true}});
        graph.markOutput("Failure.data");
        FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        if (executor.execute(submit) || executor.compiled() || !executor.lastSubmittedCompletion().sameSubmission(good)) {
            return RHITestResult::fail("failed graph recording remained executable or published a false completion");
        }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        FRAME_REQUIRE(executor.compile(context.device, render::RenderGraph::createDefaultTriangleGraph(), 16, 16, log));
        FRAME_REQUIRE(executor.execute(submit));
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        if (context.validationMessageCount && context.validationMessageCount->load() != validationBefore) {
            return RHITestResult::fail("mixed queue profiling emitted Vulkan validation messages");
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(FrameMultiQueueCompletionTest);
METALLIC_REGISTER_RHI_TEST(FrameSelfSubmitTwoSlotTest);
METALLIC_REGISTER_RHI_TEST(FrameSelfSubmitTwoSlotJoinedTest);
METALLIC_REGISTER_RHI_TEST(FrameCrossQueueGraphTest);

METALLIC_REGISTER_RHI_TEST(FrameTwoSlotGraphTest);
METALLIC_REGISTER_RHI_TEST(FrameCompletionLifecycleTest);
METALLIC_REGISTER_RHI_TEST(FrameUploadLifetimeTest);
METALLIC_REGISTER_RHI_TEST(FrameDescriptorSnapshotTest);
METALLIC_REGISTER_RHI_TEST(FrameHistoryDependencyTest);

class FrameSubmissionContextLifetimeTest final : public RHITest {
public:
    FrameSubmissionContextLifetimeTest()
    {
        type = RHITestType::Command;
        name = "frame_submission_context_lifetime";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"frame.submission.context.lifetime"}, bench::Layer::Core);
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        Commands commands;
        FRAME_REQUIRE(commands.initialize(context.device, context.graphicsQueue));
        std::shared_ptr<CommandSubmissionContext> stale;
        {
            RenderFrameContext frame;
            if (commands.buffer->begin(frame.submissionContext())) {
                return RHITestResult::fail("Unbegun frame silently became a standalone recording");
            }
            FRAME_REQUIRE(frame.begin(0));
            stale = frame.submissionContext();
            FRAME_REQUIRE(commands.buffer->begin(stale));
            FRAME_REQUIRE(commands.buffer->end());
            FRAME_REQUIRE(frame.reset());
            if (commands.buffer->begin(frame.submissionContext())) {
                return RHITestResult::fail("Reset frame silently became a standalone recording");
            }
            FRAME_REQUIRE(frame.begin(1));
            if (stale == frame.submissionContext() || stale->recording() || stale->canSubmit(true, true) ||
                RenderFrameContext::from(*commands.buffer) || commands.buffer->begin(stale)) {
                return RHITestResult::fail("Old recording generation survived frame reset");
            }
            FRAME_REQUIRE(commands.pool->reset());
            FRAME_REQUIRE(commands.buffer->begin(frame.submissionContext()));
            FRAME_REQUIRE(commands.buffer->end());
            stale = frame.submissionContext();
        }
        CommandBuffer* buffers[]{commands.buffer.get()};
        if (RenderFrameContext::from(*commands.buffer) || stale->recording() || stale->canSubmit(true, true) ||
            context.graphicsQueue.submit({.commandBuffers = buffers}) || commands.buffer->begin(stale)) {
            return RHITestResult::fail("Destroyed frame left a usable recording context");
        }
        FRAME_REQUIRE(commands.pool->reset());
        FRAME_REQUIRE(commands.buffer->begin());
        auto semaphore = context.device.createSemaphore();
        if (!semaphore) { return RHITestResult::fail("Cannot create dependency semaphore"); }
        Semaphore liveSemaphore = std::move(**semaphore);
        SemaphoreSubmitDesc invalidWait{.semaphore = semaphore->get(), .value = 1};
        FRAME_REQUIRE(commands.buffer->addDependency({&invalidWait, 1}));
        FRAME_REQUIRE(commands.buffer->end());
        if (!hasError(context.graphicsQueue.submit({.commandBuffers = buffers}), Error::InvalidArgument)) {
            return RHITestResult::fail("Neutral dependency accepted an empty semaphore wrapper");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(FrameSubmissionContextLifetimeTest);

class FrameSubmissionTransactionsTest final : public RHITest {
public:
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.submission.rollback.contract"}, bench::Layer::RenderGraph, "async", "sync");
    }

    FrameSubmissionTransactionsTest()
    {
        type = RHITestType::Command;
        name = "frame_submission_transactions";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto* queue = context.device.getQueue(render::QueueType::Graphics);
        if (queue == nullptr) { return RHITestResult::skip("requires a graphics queue"); }
        std::vector<int> events;
        Commands commands;
        FRAME_REQUIRE(commands.initialize(context.device, *queue));
        render::RenderSubsystemHost host;
        std::string log;
        FRAME_REQUIRE(host.initialize(context.device, 2, log));
        auto registerEvent = [&](render::CommandBuffer& buffer, int event) {
            return host.deferSubmission(buffer,
                [&, event]() { events.push_back(event); },
                [&, event]() { events.push_back(-event); }).transform([](auto) {});
        };
        render::CommandBuffer* buffers[] = {commands.buffer.get()};
        const render::QueueSubmitDesc submit{.commandBuffers = {buffers, 1}};

        // Direct external submit: validation failure leaves a recording retryable.
        FRAME_REQUIRE(commands.buffer->begin());
        FRAME_REQUIRE(registerEvent(*commands.buffer, 1));
        FRAME_REQUIRE(commands.buffer->end());
        if (queue->submit({.commandBuffers = std::array<render::CommandBuffer*, 1>{nullptr}}) || !events.empty()) {
            return RHITestResult::fail("rejected submit resolved a transaction");
        }
        FRAME_REQUIRE(queue->submit(submit));
        FRAME_REQUIRE(queue->waitIdle());
        FRAME_REQUIRE(commands.pool->reset());
        if (events != std::vector<int>{1}) { return RHITestResult::fail("external submit did not commit exactly once"); }

        // Pool reset and implicit command-buffer re-record discard pending work.
        FRAME_REQUIRE(commands.buffer->begin());
        FRAME_REQUIRE(registerEvent(*commands.buffer, 2));
        FRAME_REQUIRE(registerEvent(*commands.buffer, 3));
        FRAME_REQUIRE(commands.buffer->end());
        FRAME_REQUIRE(commands.pool->reset());
        if (queue->submit(submit) || events != std::vector<int>{1, -3, -2}) {
            return RHITestResult::fail("pool reset did not cancel in reverse order or allowed resubmission");
        }
        FRAME_REQUIRE(commands.buffer->begin());
        FRAME_REQUIRE(registerEvent(*commands.buffer, 4));
        FRAME_REQUIRE(commands.buffer->end());
        FRAME_REQUIRE(commands.buffer->begin());
        FRAME_REQUIRE(commands.buffer->end());
        if (events.back() != -4) { return RHITestResult::fail("re-record left an unresolved publication"); }

        // endFrame does not resolve GPU publication; frame cancellation does.
        FRAME_REQUIRE(commands.begin(0));
        FRAME_REQUIRE(host.beginFrame(0, 0, nullptr, log, &commands.frame));
        FRAME_REQUIRE(registerEvent(*commands.buffer, 5));
        FRAME_REQUIRE(commands.buffer->end());
        host.endFrame();
        if (events.back() != -4) { return RHITestResult::fail("endFrame committed unsubmitted work"); }
        if (host.beginFrame(1, 1, nullptr, log)) {
            return RHITestResult::fail("next CPU frame consumed an unresolved subsystem publication");
        }
        commands.frame.cancel();
        if (events.back() != -5 || queue->submit(submit)) {
            return RHITestResult::fail("cancelled frame remained submittable");
        }

        // A partial batch must commit its accepted prefix and cancel only the tail.
        render::QueueSubmissionTracker tracker;
        FRAME_REQUIRE(tracker.initialize(context.device, *queue));
        FRAME_REQUIRE(commands.begin(1));
        std::unique_ptr<render::CommandBuffer> tail;
        FRAME_REQUIRE(commands.pool->createCommandBuffer().transform([&](auto rhiValue) { tail = std::move(rhiValue); }));
        FRAME_REQUIRE(registerEvent(*commands.buffer, 6));
        FRAME_REQUIRE(commands.buffer->end());
        FRAME_REQUIRE(tail->begin(commands.frame.submissionContext()));
        FRAME_REQUIRE(registerEvent(*tail, 7));
        FRAME_REQUIRE(tail->end());
        render::GPUCompletionPoint prefix;
        FRAME_REQUIRE(tracker.submitSegment(submit, commands.frame).transform([&](auto value) { prefix = std::move(value); }));
        commands.frame.cancel();
        FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
        if (!commands.frame.completion().isSubmitted() || events != std::vector<int>{1, -3, -2, -4, -5, 6, -7}) {
            return RHITestResult::fail("partial batch rolled back its submitted prefix or retained its tail");
        }

        // Host teardown cancels before destroying callback owners, even when an
        // external command buffer outlives the host's active subsystems.
        FRAME_REQUIRE(commands.pool->reset());
        FRAME_REQUIRE(commands.buffer->begin());
        FRAME_REQUIRE(registerEvent(*commands.buffer, 8));
        FRAME_REQUIRE(commands.buffer->end());
        host.shutdown();
        FRAME_REQUIRE(commands.pool->reset());
        if (events.back() != -8 || events.size() != 8 || queue->submit(submit)) {
            return RHITestResult::fail("host shutdown did not cancel exactly once");
        }
        return RHITestResult::pass();
    }
};

class FrameEnvironmentProbePass final : public render::UnsafePass {
public:
    inline static int publicationCount = 0;
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(16, 16).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "FrameEnvironmentProbe",
            .entryPointName = "readEnvironment", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 2},
            .requiresRayQuery = false,
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (properties().value("failRecording", false)) { return render::makeError(render::Error::Failure); }
        const int previous = publicationCount++;
        auto publication = std::make_shared<render::SubmissionTransaction>([] {},
            [previous]() { publicationCount = previous; });
        auto publicationResult = context.commandBuffer().addSubmissionTransaction(publication);
        if (!publicationResult) { return publicationResult; }
        if (properties().value("failSubmission", false)) {
            // An explicitly cancelled transaction makes just this segment invalid
            // at the RHI boundary, after the graph prologue has been submitted.
            auto transaction = std::make_shared<render::SubmissionTransaction>([] {}, [] {});
            auto result = context.commandBuffer().addSubmissionTransaction(transaction);
            if (!result) { return result; }
            transaction->cancel();
        }
        const auto& snapshot = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot();
        render::TextureView* views[] = {snapshot.radianceView};
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = {views, 1}},
            {.binding = 1, .buffer = context.outputBuffer("data").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 2}});
    }
private:
    render::ComputeProgram program_;
};

class FrameEnvironmentRecoveryTest final : public RHITest {
public:
    FrameEnvironmentRecoveryTest()
    {
        type = RHITestType::Rendering;
        name = "frame_environment_submission_recovery";
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto result = render::createDevice({.applicationName = "Environment submission recovery",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptors"); }
        FRAME_REQUIRE(result);
        render::registerRenderGraphPassType("FrameEnvironmentProbePass", "Environment submission probe",
            [] { return std::make_unique<FrameEnvironmentProbePass>(); });
        for (bool partialSubmission : {false, true}) {
            FrameEnvironmentProbePass::publicationCount = 0;
            render::RenderGraph graph;
            graph.addNode("FrameEnvironmentProbePass", "First");
            graph.addNode("FrameEnvironmentProbePass", "Probe",
                {{"failRecording", !partialSubmission}, {"failSubmission", partialSubmission}});
            graph.addNode("FrameEnvironmentProbePass", "Tail");
            graph.markOutput("First.data");
            graph.markOutput("Probe.data");
            graph.markOutput("Tail.data");
            render::RenderGraphExecutor executor;
            std::string log;
            FRAME_REQUIRE(executor.compile(*device, graph, 1, 1, log));
            const render::RenderGraphSubmitDesc submit{.graphicsQueue = device->getQueue(render::QueueType::Graphics)};
            if (executor.execute(submit) || executor.compiled()) {
                return RHITestResult::fail("failed graph remained compiled");
            }
            if (executor.lastSubmittedCompletion().isSubmitted() != partialSubmission) {
                return RHITestResult::fail("graph lost its partial-submission completion");
            }
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            if (FrameEnvironmentProbePass::publicationCount != (partialSubmission ? 1 : 0)) {
                return RHITestResult::fail("graph did not roll back its unsubmitted tail in reverse recording order");
            }
            auto* environment = executor.subsystemHost()->get<render::EnvironmentLightingSubsystem>();
            if (environment->snapshot().valid() != partialSubmission ||
                environment->snapshot().resourceRevision != (partialSubmission ? 1u : 0u)) {
                return RHITestResult::fail("environment publication does not match the accepted graph prefix");
            }
            render::RenderGraph recovered;
            recovered.addNode("FrameEnvironmentProbePass", "Probe");
            recovered.markOutput("Probe.data");
            FRAME_REQUIRE(executor.compile(*device, recovered, 1, 1, log));
            FRAME_REQUIRE(executor.execute(submit));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            auto* buffer = executor.outputResource("Probe.data")->buffer;
            std::array<uint32_t, 4> words{};
            if (!readWords(*buffer, words.data(), words.size()) || words[3] != 0x3f800000u ||
                environment->snapshot().resourceRevision != 1) {
                return RHITestResult::fail("environment retry did not restore the uploaded alpha=1 pixel exactly once");
            }
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(FrameSubmissionTransactionsTest);
METALLIC_REGISTER_RHI_TEST(FrameEnvironmentRecoveryTest);

#undef FRAME_REQUIRE
} // namespace
} // namespace metallic::tests
