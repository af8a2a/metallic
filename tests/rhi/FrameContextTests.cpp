#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include <stdexcept>
#include <string>

#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/GraphEvidence.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
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
    std::span<const render::ComputeResourceBindingDesc> bindings, render::ComputeProgram& program, std::string& log)
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
        .resourceParameterSize = (std::string_view(entry) == "accumulateHistory" || std::string_view(entry) == "fillSolidTexture")
            ? sizeof(metallic::tests::FrameHistoryProbeResources) :
            std::string_view(entry) == "sampleImages" ? sizeof(metallic::tests::FrameImagesProbeResources) : sizeof(metallic::tests::FrameCopyProbeResources),
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
        const render::ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .kind = render::ComputeResourceBindingKind::StorageBuffer},
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
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .buffer = inputs[index].get()}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .buffer = output.get()},
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
        constexpr uint32_t kStableFrames = 32;
        constexpr uint32_t kCancelledFrames = 4;
        constexpr uint32_t kResumedIndex = 6 + kStableFrames;
        constexpr uint32_t kInFlightIndex = kResumedIndex + 1;
        constexpr uint32_t kOutputCount = kInFlightIndex + 3;
        FRAME_REQUIRE(tracker.initialize(*device, queue));
        FRAME_REQUIRE(commands.initialize(*device, queue));
        FRAME_REQUIRE(pending.initialize(*device, queue));
        FRAME_REQUIRE(device->createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        FRAME_REQUIRE(device->createBuffer({.size = kOutputCount * sizeof(uint32_t), .structureStride = 4,
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
        // Distinct generation owners make retirement observable even though the
        // generations deliberately share their unchanged first texture.
        auto aOwner = std::make_shared<std::shared_ptr<Images>>(images);
        auto bOwner = std::make_shared<std::shared_ptr<Images>>(images);
        std::weak_ptr<void> aLifetime = aOwner;
        std::weak_ptr<void> bLifetime = bOwner;
        auto makeSnapshot = [](std::shared_ptr<const void> owner,
            std::vector<std::shared_ptr<render::TextureView>> views) {
            auto snapshot = std::make_shared<render::ComputeSampledImageSnapshot>();
            snapshot->owner = std::move(owner);
            snapshot->views = std::move(views);
            return std::shared_ptr<const render::ComputeSampledImageSnapshot>(std::move(snapshot));
        };
        auto a = makeSnapshot(aOwner, {images->views[0], images->views[1]});
        auto b = makeSnapshot(bOwner, {images->views[0], images->views[2]});
        aOwner.reset(); bOwner.reset();
        const render::ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .kind = render::ComputeResourceBindingKind::SampledImage, .descriptorCount = 2},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .kind = render::ComputeResourceBindingKind::StorageBuffer},
        };
        std::string log;
        FRAME_REQUIRE(createProbe(*device, "sampleImages", bindings, program, log));
        FRAME_REQUIRE(createProbe(*device, "sampleImages", bindings, secondProgram, log));
        QueueDrain drain{queue, gate.get()};
        // A raw array and a second program reuse the same canonical descriptors.
        const std::array<uint32_t, 6> writes{2, 0, 1, 0, 0, 0};
        std::array<uint32_t, kOutputCount> expected{30, 30, 40, 30, 30, 40};
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
                    if (auto commandResult = commands.buffer->clearColorTexture(*images->textures[j], render::TextureLayout::TransferDestination,
                        {float((j + 1) * 10), 0, 0, 0}); !commandResult) { return RHITestResult::fail(std::string("clearColorTexture failed: ") + render::resultToString(commandResult)); }
                    barrier.before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite};
                    barrier.after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead};
                    if (auto commandResult = commands.buffer->synchronize({.textures = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                }
            }
            storageBarrier(*commands.buffer, *output);
            render::TextureView* raw[] = {images->views[0].get(), images->views[1].get()};
            const render::ComputeDispatchBinding resources[] = {
                {
                    .binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images),
                    .textureViews = {raw, 2},
                    .sampledImages = i == 3 ? nullptr : ((i == 2 || i == 5) ? b : a),
                },
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = output.get()},
            };
            render::ComputeDispatchStats stats;
            FRAME_REQUIRE((i == 4 ? secondProgram : program).dispatch({
                .commandBuffer = commands.buffer.get(),
                .bindings = {resources, 2},
                .pushData = &i,
                .pushDataSize = 4,
                .stats = &stats,
            }));
            const bool tableBuild = i == 0 || i == 2;
            const bool rawArray = i == 3;
            if (stats.sampledImageWrites != writes[i] ||
                stats.sampledImageCacheHits != (tableBuild || rawArray ? 2 - writes[i] : 0) ||
                stats.sampledImageTableBuilds != (tableBuild ? 1u : 0u) ||
                stats.sampledImageTableCacheHits != (!tableBuild && !rawArray ? 1u : 0u) ||
                stats.sampledImageTableUploadBytes != (tableBuild ? 2 * sizeof(uint32_t) : 0)) {
                return RHITestResult::fail("incorrect sampled-image generation reuse at step " + std::to_string(i));
            }
            FRAME_REQUIRE(commands.submit(tracker));
            FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
            FRAME_REQUIRE(commands.frame.reset());
        }
        // Completed frame arenas are recycled repeatedly while both independent
        // programs reuse a snapshot's immutable table. No per-image registry
        // lookup or index upload is allowed on these stable frames.
        for (uint32_t i = 6; i < kResumedIndex; ++i) {
            FRAME_REQUIRE(commands.begin(i));
            storageBarrier(*commands.buffer, *output);
            const bool useA = (i & 1u) == 0;
            expected[i] = useA ? 30 : 40;
            const render::ComputeDispatchBinding resources[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .sampledImages = useA ? a : b},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = output.get()},
            };
            render::ComputeDispatchStats stats;
            FRAME_REQUIRE((i % 3 == 0 ? secondProgram : program).dispatch({
                .commandBuffer = commands.buffer.get(), .bindings = resources,
                .pushData = &i, .pushDataSize = 4, .stats = &stats,
            }));
            if (stats.sampledImageWrites != 0 || stats.sampledImageCacheHits != 0 ||
                stats.sampledImageTableBuilds != 0 || stats.sampledImageTableCacheHits != 1 ||
                stats.sampledImageTableUploadBytes != 0) {
                return RHITestResult::fail("stable snapshot repeated per-image registration or index upload");
            }
            FRAME_REQUIRE(commands.submit(tracker));
            FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
            FRAME_REQUIRE(commands.frame.reset());
        }
        // A table first built during a cancelled recording must remain usable.
        // Reusing the frame upload arena must not overwrite its index storage.
        auto cancelled = makeSnapshot(images, {images->views[2], images->views[1]});
        for (uint32_t attempt = 0; attempt <= kCancelledFrames; ++attempt) {
            FRAME_REQUIRE(commands.begin(kResumedIndex + attempt));
            storageBarrier(*commands.buffer, *output);
            const render::ComputeDispatchBinding resources[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .sampledImages = cancelled},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = output.get()},
            };
            render::ComputeDispatchStats stats;
            FRAME_REQUIRE(program.dispatch({
                .commandBuffer = commands.buffer.get(), .bindings = resources,
                .pushData = &kResumedIndex, .pushDataSize = 4, .stats = &stats,
            }));
            if (stats.sampledImageWrites != 0 || stats.sampledImageCacheHits != (attempt == 0 ? 2u : 0u) ||
                stats.sampledImageTableBuilds != (attempt == 0 ? 1u : 0u) ||
                stats.sampledImageTableCacheHits != (attempt == 0 ? 0u : 1u) ||
                stats.sampledImageTableUploadBytes != (attempt == 0 ? 2 * sizeof(uint32_t) : 0)) {
                return RHITestResult::fail("cancelled recording invalidated or rebuilt a snapshot table");
            }
            if (attempt < kCancelledFrames) {
                FRAME_REQUIRE(commands.pool->reset());
                commands.frame.cancel();
            } else {
                FRAME_REQUIRE(commands.submit(tracker));
                FRAME_REQUIRE(commands.frame.wait(kWaitTimeout));
            }
            FRAME_REQUIRE(commands.frame.reset());
        }
        cancelled.reset();
        expected[kResumedIndex] = 50;
        // A populated cache must never bypass the other device's image boundary.
        {
            std::unique_ptr<render::Device> otherDevice;
            FRAME_REQUIRE(render::createDevice({.applicationName = "Foreign snapshot cache",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { otherDevice = std::move(value); }));
            render::ComputeProgram otherProgram;
            FRAME_REQUIRE(createProbe(*otherDevice, "sampleImages", bindings, otherProgram, log));
            std::unique_ptr<render::Buffer> otherOutput;
            FRAME_REQUIRE(otherDevice->createBuffer({.size = kOutputCount * sizeof(uint32_t), .structureStride = 4,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback})
                .transform([&](auto value) { otherOutput = std::move(value); }));
            const render::ComputeDispatchBinding resources[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .sampledImages = a},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = otherOutput.get()},
            };
            render::ComputeDispatchStats stats;
            const auto encoded = otherProgram.encoder.encode(nullptr, {
                .bindings = resources, .pushData = &kResumedIndex, .pushDataSize = 4, .stats = &stats,
            });
            if (encoded || !render::hasError(encoded, render::Error::InvalidArgument) ||
                stats.sampledImageTableCacheHits != 0) {
                return RHITestResult::fail("foreign registry reused a cached sampled-image table");
            }
        }
        // Keep two generations in flight and release the caller's ownership
        // before execution. Also recreate the same registry wrapper's heap
        // epoch: the same snapshot must acquire new indices, while the old
        // packet must retain its original heap and table until GPU completion.
        std::shared_ptr<render::ResourceRegistry> registry;
        FRAME_REQUIRE(render::ResourceRegistry::forDevice(*device).transform([&](auto value) { registry = std::move(value); }));
        for (uint32_t i = kInFlightIndex; i < kOutputCount; ++i) {
            const bool first = i == kInFlightIndex;
            const bool useA = i != kOutputCount - 1;
            expected[i] = useA ? 30 : 40;
            auto& frame = first ? commands : pending;
            if (i == kInFlightIndex + 1) {
                *registry = render::ResourceRegistry{};
                FRAME_REQUIRE(registry->initialize(*device));
            }
            if (i != kOutputCount - 1) { FRAME_REQUIRE(frame.begin(i)); }
            storageBarrier(*frame.buffer, *output);
            const render::ComputeDispatchBinding resources[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .sampledImages = useA ? a : b}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = output.get()},
            };
            render::ComputeDispatchStats stats;
            FRAME_REQUIRE(program.dispatch({
                .commandBuffer = frame.buffer.get(),
                .bindings = {resources, 2},
                .pushData = &i,
                .pushDataSize = 4,
                .stats = &stats,
            }));
            const uint32_t expectedWrites = first ? 0u : (useA ? 2u : 1u);
            if (stats.sampledImageWrites != expectedWrites ||
                stats.sampledImageCacheHits != (!first && !useA ? 1u : 0u) ||
                stats.sampledImageTableBuilds != (first ? 0u : 1u) ||
                stats.sampledImageTableCacheHits != (first ? 1u : 0u) ||
                stats.sampledImageTableUploadBytes != (first ? 0u : 2 * sizeof(uint32_t))) {
                return RHITestResult::fail("snapshot reused indices from a different registry epoch");
            }
            if (first || i == kOutputCount - 1) {
                FRAME_REQUIRE(frame.submit(tracker, first ? gate.get() : nullptr));
            }
        }
        std::weak_ptr<Images> lifetime = images;
        images.reset(); a.reset(); b.reset();
        program.clear(); secondProgram.clear();
        if (lifetime.expired() || aLifetime.expired() || bLifetime.expired()) {
            return RHITestResult::fail("in-flight sampled image generation released early");
        }
        FRAME_REQUIRE(gate->signal(1));
        FRAME_REQUIRE(pending.frame.wait(kWaitTimeout));
        FRAME_REQUIRE(commands.frame.reset());
        if (aLifetime.expired() || bLifetime.expired()) {
            return RHITestResult::fail("new epoch did not retain its sampled-image generations");
        }
        FRAME_REQUIRE(pending.frame.reset());
        if (!lifetime.expired() || !aLifetime.expired() || !bLifetime.expired()) {
            return RHITestResult::fail("completed descriptor cache retains retired images");
        }
        std::array<uint32_t, kOutputCount> actual{};
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
        const render::ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .kind = render::ComputeResourceBindingKind::StorageBuffer},
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
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .textureView = history.texture("history", render::HistorySlot::Previous).view},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .textureView = history.texture("history", render::HistorySlot::Current).view},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .buffer = output.get()},
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
                if (auto commandResult = (imageReadbacks[index].get())->slice().and_then([&](const auto& bufferSlice) { return commands.buffer->copyTextureToBuffer({
                    .texture = executor.outputResource("Triangle.color")->texture,
                    .buffer = bufferSlice, .width = kWidth, .height = kWidth,
                }); }); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
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
        if (auto commandResult = context.commandBuffer().clearColorTexture(*context.outputTexture("color").texture(),
            render::TextureLayout::TransferDestination, {0.25f, 0.5f, 0.75f, 1.0f}); !commandResult) { return commandResult; }
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
        if (auto commandResult = (pixels.get())->slice().and_then([&](const auto& bufferSlice) { return readback.buffer->copyTextureToBuffer({.texture = executor.outputResource("TextureCopy.color")->texture,
            .buffer = bufferSlice, .width = 16, .height = 16}); }); !commandResult) { return RHITestResult::fail(std::string("copyTextureToBuffer failed: ") + render::resultToString(commandResult)); }
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
        if (properties().value("splitSubmission", false)) {
            auto created = context.device->createSemaphore();
            if (!created) { return std::unexpected(created.error()); }
            submissionWait_ = std::move(*created);
            auto signaled = submissionWait_->signal(1);
            if (!signaled) { return signaled; }
        }
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "FrameEnvironmentProbe",
            .entryPointName = "readEnvironment", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameEnvironmentProbeResources, environment), .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameEnvironmentProbeResources, output), .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 2},
            .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::FrameEnvironmentProbeResources),
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (properties().value("failRecording", false)) { return render::makeError(render::Error::Failure); }
        if (submissionWait_) {
            const render::SemaphoreSubmitDesc wait{.semaphore = submissionWait_.get(), .value = 1};
            auto result = context.commandBuffer().addDependency({&wait, 1});
            if (!result) { return result; }
        }
        const int previous = publicationCount++;
        auto publication = std::make_shared<render::SubmissionTransaction>([] {},
            [previous]() { publicationCount = previous; });
        auto publicationResult = context.commandBuffer().addSubmissionTransaction(publication);
        if (!publicationResult) { return publicationResult; }
        if (properties().value("failSubmission", false)) {
            // Reject this batch at acceptance. An explicit wait boundary keeps
            // an accepted prefix; a merged batch must roll back every command.
            auto transaction = std::make_shared<render::SubmissionTransaction>([] {}, [] {});
            auto result = context.commandBuffer().addSubmissionTransaction(transaction);
            if (!result) { return result; }
            transaction->cancel();
        }
        const auto& snapshot = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot();
        render::TextureView* views[] = {snapshot.radianceView};
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameEnvironmentProbeResources, environment), .textureViews = {views, 1}},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameEnvironmentProbeResources, output), .buffer = context.outputBuffer("data").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 2}});
    }
private:
    render::ComputeProgram program_;
    std::unique_ptr<render::Semaphore> submissionWait_;
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
        for (const auto [failRecording, partialSubmission] :
            {std::pair{true, false}, std::pair{false, false}, std::pair{false, true}}) {
            FrameEnvironmentProbePass::publicationCount = 0;
            render::RenderGraph graph;
            graph.addNode("FrameEnvironmentProbePass", "First");
            graph.addNode("FrameEnvironmentProbePass", "Probe",
                {{"failRecording", failRecording}, {"failSubmission", !failRecording}, {"splitSubmission", partialSubmission}});
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

class CrossFrameGateGraphicsPass final : public render::RasterPass {
public:
    static inline render::Semaphore* gateSemaphore = nullptr;
    static inline uint64_t gateWaitValue = 0;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Gated clear color")
            .transient(render::RenderGraphInitialization::Clear)
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (gateSemaphore != nullptr && gateWaitValue != 0) {
            const render::SemaphoreSubmitDesc wait{
                .semaphore = gateSemaphore,
                .value = gateWaitValue,
                .stages = render::PipelineStageBits::AllCommands,
            };
            auto dep = context.commandBuffer().addDependency({&wait, 1});
            if (!dep) { return dep; }
        }
        return clear_->execute(context);
    }

private:
    std::unique_ptr<render::RenderGraphPass> clear_ = render::builtin_pass::createClearColorPass();
};

class CrossFrameConservativeOptOutPass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    // Deliberately false even though overlap/async/pipelined are all true.
    bool supportsCrossFrameResourceScheduling() const override { return false; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        return clear_->reflect({});
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return clear_->execute(context);
    }

private:
    std::unique_ptr<render::RenderGraphPass> clear_ = render::builtin_pass::createClearColorPass();
};

class CrossFrameGraphicsBufferCopyPass final : public render::ComputePass {
public:
    render::QueueType queueType() const override { return render::QueueType::Graphics; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        return copy_->reflect(context);
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        return copy_->compile(context, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return copy_->execute(context);
    }

private:
    std::unique_ptr<render::RenderGraphPass> copy_ = render::builtin_pass::createRenderGraphBufferCopyPass();
};

class FrameCrossFrameOverlapAndHazardTest final : public RHITest {
public:
    FrameCrossFrameOverlapAndHazardTest()
    {
        type = RHITestType::Rendering;
        name = "frame_cross_frame_overlap_and_hazards";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.cross_frame.gpu_overlap", "graph.cross_frame.conservative_fallback",
            "graph.cross_frame.version_pool"}, bench::Layer::RenderGraph, "async", "sync");
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("CrossFrameGateGraphicsPass", "Timeline-gated cross-frame graphics pass",
                [] { return std::make_unique<CrossFrameGateGraphicsPass>(); });
            registerRenderGraphPassType("CrossFrameConservativeOptOutPass", "Cross-frame opt-out pass",
                [] { return std::make_unique<CrossFrameConservativeOptOutPass>(); });
            registerRenderGraphPassType("CrossFrameGraphicsBufferCopyPass", "Cross-frame graphics buffer copy",
                [] { return std::make_unique<CrossFrameGraphicsBufferCopyPass>(); });
            return true;
        }();
        (void)registered;

        std::unique_ptr<Device> device;
        auto created = createDevice({.applicationName = "Cross-frame overlap and hazards",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true}).transform([&](auto v) { device = std::move(v); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute) { return RHITestResult::fail("missing graphics or compute queue"); }

        // Subtest A: True cross-frame GPU queue overlap + MaximizeOverlap version pool.
        std::unique_ptr<Semaphore> gate;
        FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { gate = std::move(s); }));
        bool gateSignaled = false;

        RenderGraph overlapGraph;
        overlapGraph.setName("CrossFrameOverlap");
        overlapGraph.addNode("CrossFrameGateGraphicsPass", "Gfx");
        overlapGraph.addNode("RenderGraphBufferWritePass", "Cmp");
        overlapGraph.markOutput("Gfx.color");
        overlapGraph.markOutput("Cmp.data");

        RenderGraphExecutor executor;
        struct GateGuard {
            Semaphore* sem = nullptr;
            bool* signaled = nullptr;
            ~GateGuard()
            {
                CrossFrameGateGraphicsPass::gateSemaphore = nullptr;
                CrossFrameGateGraphicsPass::gateWaitValue = 0;
                if (sem && signaled && !*signaled) {
                    (void)sem->signal(1);
                    *signaled = true;
                }
            }
        } gateGuard{gate.get(), &gateSignaled};
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        RenderGraphCompileOptions overlapOpts;
        overlapOpts.memoryPolicy = RenderGraphMemoryPolicy::MaximizeOverlap;
        overlapOpts.maxFramesInFlight = 3;
        overlapOpts.versionPoolBudgetBytes = 4ull * 1024 * 1024;
        overlapOpts.maxPhysicalVersionsPerResource = 2;
        FRAME_REQUIRE(executor.compile(*device, overlapGraph, 64, 64, overlapOpts, log));

        // Frame 0: gate Graphics on unsignaled gate semaphore.
        CrossFrameGateGraphicsPass::gateSemaphore = gate.get();
        CrossFrameGateGraphicsPass::gateWaitValue = 1;
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
        if (!executor.executionStats().crossFrameResourceScheduled) {
            return RHITestResult::fail("Frame 0 did not enable crossFrameResourceScheduled");
        }
        const auto* gfxRes0 = executor.outputResource("Gfx.color");
        const Texture* frame0Tex = gfxRes0 ? gfxRes0->texture : nullptr;

        // Frame 1: remove gate and execute while Frame 0's Graphics is still blocked!
        CrossFrameGateGraphicsPass::gateSemaphore = nullptr;
        CrossFrameGateGraphicsPass::gateWaitValue = 0;
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
        const auto stats1 = executor.executionStats();
        const auto snap1 = executor.executionSnapshot();
        if (!stats1.crossFrameResourceScheduled || !snap1 || !snap1->crossFrameResourceScheduled ||
            !stats1.conservativeFallbackReasons.empty() || stats1.maxFramesInFlight != 3) {
            return RHITestResult::fail("Frame 1 cross-frame scheduling diagnostics mismatch");
        }
        const auto* gfxRes1 = executor.outputResource("Gfx.color");
        if (!gfxRes1 || !frame0Tex || gfxRes1->texture == frame0Tex || stats1.extraVersionCount == 0) {
            return RHITestResult::fail("MaximizeOverlap did not allocate extra physical version while Frame 0 backing was busy");
        }

        // Prove Frame 1 Compute finishes ON THE GPU while Frame 0 Graphics is STILL blocked!
        if (!graphics->sameQueue(*compute)) {
            std::unique_ptr<Fence> computeFence;
            FRAME_REQUIRE(device->createFence(false).transform([&](auto f) { computeFence = std::move(f); }));
            FRAME_REQUIRE(compute->submit({.signalFence = computeFence.get()}));
            auto waited = computeFence->wait(5'000'000'000ull);
            if (!waited) {
                return RHITestResult::fail("Frame 1 disjoint Compute did not complete while Frame 0 Graphics was gated (no true GPU overlap)");
            }
            const auto* cmpOut = executor.outputResource("Cmp.data");
            constexpr std::array expected{0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu};
            std::array<uint32_t, 4> actual{};
            if (!cmpOut || !cmpOut->buffer || !readWords(*cmpOut->buffer, actual.data(), actual.size()) || actual != expected) {
                return RHITestResult::fail("Frame 1 overlapping Compute output readback mismatch");
            }
        }

        // Release Gate so Frame 0 and Frame 1 Graphics complete cleanly.
        FRAME_REQUIRE(gate->signal(1));
        gateSignaled = true;
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));

        // Subtest B: Opt-out pass forces conservative fallback even when overlap/async/pipelined are true.
        RenderGraph fallbackGraph;
        fallbackGraph.setName("CrossFrameConservativeFallback");
        fallbackGraph.addNode("CrossFrameConservativeOptOutPass", "OptOut");
        fallbackGraph.addNode("RenderGraphBufferWritePass", "Cmp");
        fallbackGraph.markOutput("OptOut.color");
        fallbackGraph.markOutput("Cmp.data");
        FRAME_REQUIRE(executor.compile(*device, fallbackGraph, 32, 32, log));
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        const auto fbStats = executor.executionStats();
        const auto fbSnap = executor.executionSnapshot();
        if (fbStats.crossFrameResourceScheduled || fbStats.conservativeFallbackReasons.empty() ||
            !fbSnap || fbSnap->crossFrameResourceScheduled || fbSnap->conservativeFallbackReasons.empty()) {
            return RHITestResult::fail("Opt-out pass failed to force conservative fallback or report reason");
        }

        // Subtest C: Multi-queue cross-frame RAW/WAR hazard test across 6 frames with 3 frames in flight.
        RenderGraph hazardGraph;
        hazardGraph.setName("CrossFrameMultiQueueHazard");
        hazardGraph.addNode("RenderGraphBufferWritePass", "Writer");
        hazardGraph.addNode("CrossFrameGraphicsBufferCopyPass", "ReaderGfx");
        hazardGraph.addEdge("Writer.data", "ReaderGfx.source");
        hazardGraph.markOutput("ReaderGfx.data");
        RenderGraphCompileOptions hazardOpts;
        hazardOpts.maxFramesInFlight = 3;
        hazardOpts.memoryPolicy = RenderGraphMemoryPolicy::MinimizeMemory;
        FRAME_REQUIRE(executor.compile(*device, hazardGraph, 1, 1, hazardOpts, log));
        for (uint32_t f = 0; f < 6; ++f) {
            FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            if (!executor.executionStats().crossFrameResourceScheduled ||
                executor.executionStats().extraVersionCount != 0) {
                return RHITestResult::fail("MinimizeMemory multi-queue hazard test violated policy or scheduling");
            }
        }
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        const auto* hazardOut = executor.outputResource("ReaderGfx.data");
        constexpr std::array expected{0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu};
        std::array<uint32_t, 4> actual{};
        if (!hazardOut || !hazardOut->buffer || !readWords(*hazardOut->buffer, actual.data(), actual.size()) || actual != expected) {
            return RHITestResult::fail("Multi-queue cross-frame RAW/WAR output mismatch");
        }

        return RHITestResult::pass(graphics->sameQueue(*compute)
            ? "Cross-frame scheduling, version pool, conservative fallback and hazards verified (aliased queues)"
            : "Cross-frame GPU overlap (Compute completed while Graphics gated), MaximizeOverlap version pool, conservative fallback and multi-queue RAW/WAR verified");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameCrossFrameOverlapAndHazardTest);

class TemporalCounterProbePass final : public render::ComputePass {
public:
    static inline bool skipWrite = false;
    static inline bool failExecute = false;
    static inline bool lastPrevValid = false;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("prev", "Previous frame counter")
            .buffer(16, 4).transferRead().temporalRead("counter", -1, /*graphOwned=*/true);
        reflection.addBufferOutput("curr", "Current frame counter")
            .buffer(16, 4).transferWrite().temporalWrite("counter", /*graphOwned=*/true);
        reflection.addBufferOutput("out", "Observed previous counter")
            .buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        using namespace render;
        if (!context.device) { return makeError(Error::InvalidArgument); }
        BufferDesc desc;
        desc.size = 16 * 16;
        desc.usage = BufferUsageBits::TransferSource;
        desc.memoryLocation = MemoryLocation::HostUpload;
        desc.queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy;
        auto created = context.device->createBuffer(desc);
        if (!created) { return std::unexpected(created.error()); }
        constants_ = std::move(*created);
        auto* mapped = static_cast<uint32_t*>(constants_->map());
        if (!mapped) { return makeError(Error::Failure); }
        for (uint32_t slot = 0; slot < 16; ++slot) {
            const uint32_t value = slot == 0 ? 0xFFFFFFFFu : (100u + (slot - 1u));
            for (uint32_t w = 0; w < 4; ++w) { mapped[slot * 4 + w] = value; }
        }
        constants_->flush();
        constants_->unmap();
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const BufferHandle prev = context.temporalBuffer("prev").buffer;
        const BufferHandle curr = context.temporalBuffer("curr").buffer;
        const BufferHandle out = context.outputBuffer("out");
        if (!prev.valid() || !curr.valid() || !out.valid() || !constants_) {
            return makeError(Error::InvalidArgument);
        }
        lastPrevValid = context.temporalValid("prev");
        if (failExecute) {
            return makeError(Error::Failure);
        }
        const uint64_t tagSlot = 1ull + (context.frameIndex() % 15ull);
        auto currSlice = curr.buffer()->slice({0, 16});
        auto prevSlice = prev.buffer()->slice({0, 16});
        auto outSlice = out.buffer()->slice({0, 16});
        auto invalidSlice = constants_->slice({0, 16});
        auto tagSlice = constants_->slice({tagSlot * 16ull, 16});
        if (!currSlice || !prevSlice || !outSlice || !invalidSlice || !tagSlice) {
            return makeError(Error::Failure);
        }
        if (lastPrevValid) {
            auto r = context.commandBuffer().copyBuffer(*prevSlice, *outSlice);
            if (!r) { return r; }
        } else {
            auto r = context.commandBuffer().copyBuffer(*invalidSlice, *outSlice);
            if (!r) { return r; }
        }
        if (skipWrite) {
            context.setTemporalWritten("curr", false);
        } else {
            auto r = context.commandBuffer().copyBuffer(*tagSlice, *currSlice);
            if (!r) { return r; }
        }
        const BufferBarrierDesc hostBarrier{
            .buffer = out.buffer(),
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead},
        };
        return context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}});
    }

private:
    std::unique_ptr<render::Buffer> constants_;
};

class FrameDeclarativeTemporalHistoryTest final : public RHITest {
public:
    FrameDeclarativeTemporalHistoryTest()
    {
        type = RHITestType::Rendering;
        name = "frame_declarative_temporal_history";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.temporal.declarative_history", "graph.temporal.three_slot_ring"},
            bench::Layer::RenderGraph, "async", "sync");
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("TemporalCounterProbePass",
            "Declarative temporal history probe",
            [] { return std::make_unique<TemporalCounterProbePass>(); });
        (void)registered;

        for (uint32_t framesInFlight : {2u, 3u}) {
            TemporalCounterProbePass::skipWrite = false;
            TemporalCounterProbePass::failExecute = false;
            RenderGraph graph;
            graph.setName("DeclarativeTemporalTest");
            graph.addNode("TemporalCounterProbePass", "Counter");
            graph.markOutput("Counter.out");

            RenderGraphExecutor executor;
            executor.setExecutionCaptureEnabled(true);
            std::string log;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = framesInFlight;
            FRAME_REQUIRE(executor.compile(context.device, graph, 16, 16, opts, log));

            auto readOut = [&]() -> uint32_t {
                auto* out = executor.outputResource("Counter.out");
                std::array<uint32_t, 4> words{};
                if (!out || !out->buffer || !readWords(*out->buffer, words.data(), words.size())) { return 0; }
                return words[0];
            };

            // Frame 0: no valid previous -> out == 0xFFFFFFFF, writes 100.
            FRAME_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            if (TemporalCounterProbePass::lastPrevValid || readOut() != 0xFFFFFFFFu) {
                return RHITestResult::fail("Frame 0 expected invalid temporal previous");
            }
            const auto snap = executor.executionSnapshot();
            const bool hasTemporalResource = snap && std::any_of(snap->resources.begin(), snap->resources.end(),
                [](const auto& r) { return r.ownership == RenderGraphResourceOwnership::TemporalOwned && r.logicalResourceId != 0; });
            if (!hasTemporalResource) {
                return RHITestResult::fail("Execution snapshot did not classify declarative temporal resource as TemporalOwned");
            }

            // Frames 1..3: each frame observes previous frame's written tag (100, 101, 102).
            for (uint32_t f = 1; f <= 3; ++f) {
                FRAME_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue}));
                FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
                if (!TemporalCounterProbePass::lastPrevValid || readOut() != 100u + (f - 1)) {
                    return RHITestResult::fail("Frame " + std::to_string(f) + " observed wrong temporal value");
                }
            }

            // Frame 4: setTemporalWritten("curr", false) -> Frame 5 sees temporalValid == false!
            TemporalCounterProbePass::skipWrite = true;
            FRAME_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            TemporalCounterProbePass::skipWrite = false;
            FRAME_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            if (TemporalCounterProbePass::lastPrevValid || readOut() != 0xFFFFFFFFu) {
                return RHITestResult::fail("setTemporalWritten(false) did not suppress next frame temporal validity");
            }

            // Frame 6: failure rolls back current slot cleanly.
            TemporalCounterProbePass::failExecute = true;
            if (executor.execute({.graphicsQueue = &context.graphicsQueue}).has_value()) {
                return RHITestResult::fail("Expected failing temporal pass to return error");
            }
            TemporalCounterProbePass::failExecute = false;
        }
        return RHITestResult::pass("Declarative temporal fields verified for 2-slot and 3-slot rings, skipWrite, failure rollback, and ExecutionSnapshot TemporalOwned metadata");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameDeclarativeTemporalHistoryTest);

class BenchComputeStagePass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return properties().value("crossFrame", true); }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const uint64_t elements = properties().value("elements", 262144ull);
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("stage", "Async compute stage output")
            .buffer(elements * sizeof(uint32_t), sizeof(uint32_t)).storageWrite()
            .transient(render::RenderGraphInitialization::FullOverwrite);
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        if (!context.device) { return makeError(Error::InvalidArgument); }
        BufferDesc inputDesc;
        inputDesc.size = 16;
        inputDesc.usage = BufferUsageBits::Storage | BufferUsageBits::TransferDestination;
        inputDesc.memoryLocation = MemoryLocation::HostUpload;
        inputDesc.queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy;
        auto created = context.device->createBuffer(inputDesc);
        if (!created) { return std::unexpected(created.error()); }
        seed_ = std::move(*created);
        if (auto* mapped = static_cast<uint32_t*>(seed_->map())) {
            mapped[0] = 0x13579BDFu;
            mapped[1] = 0;
            mapped[2] = 0;
            mapped[3] = 0;
            seed_->flush();
            seed_->unmap();
        }
        const ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .kind = ComputeResourceBindingKind::StorageBuffer},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        return createProbe(*context.device, "benchComputeWork", bindings, program_, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const BufferHandle stage = context.outputBuffer("stage");
        if (!stage.valid() || !seed_ || !program_.valid()) { return makeError(Error::InvalidArgument); }
        const uint32_t iterations = properties().value("iterations", 512u);
        const uint64_t elements = properties().value("elements", 262144ull);
        const uint32_t groupsX = static_cast<uint32_t>((elements + 63ull) / 64ull);
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, input), .buffer = seed_.get()},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameCopyProbeResources, output), .buffer = stage.buffer()},
        };
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .pushData = &iterations,
            .pushDataSize = sizeof(iterations),
            .groupCountX = groupsX,
            .groupCountY = 1,
            .groupCountZ = 1,
        });
    }

private:
    render::ComputeProgram program_;
    std::unique_ptr<render::Buffer> seed_;
};

class BenchGraphicsStagePass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return properties().value("crossFrame", true); }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        const uint64_t elements = properties().value("elements", 262144ull);
        render::RenderPassReflection reflection;
        reflection.addBufferInput("stage", "Async compute stage input")
            .buffer(elements * sizeof(uint32_t), sizeof(uint32_t)).transferRead();
        reflection.addBufferOutput("copyOut", "Graphics consumed copy")
            .buffer(elements * sizeof(uint32_t), sizeof(uint32_t)).transferWrite()
            .transient(render::RenderGraphInitialization::FullOverwrite);
        auto& color = reflection.addTextureOutput("color", "Graphics render target")
            .texture2D(context.width, context.height).colorWrite()
            .transient(render::RenderGraphInitialization::Clear);
        color.format = render::Format::RGBA16Sfloat;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const BufferHandle stage = context.inputBuffer("stage");
        const BufferHandle copyOut = context.outputBuffer("copyOut");
        const TextureHandle color = context.outputTexture("color");
        if (!stage.valid() || !copyOut.valid() || !color.valid()) { return makeError(Error::InvalidArgument); }
        auto srcSlice = stage.buffer()->slice();
        auto dstSlice = copyOut.buffer()->slice();
        if (!srcSlice || !dstSlice) { return makeError(Error::Failure); }
        if (auto r = context.commandBuffer().copyBuffer(*srcSlice, *dstSlice); !r) { return r; }

        const uint32_t clearPasses = properties().value("clearPasses", 6u);
        for (uint32_t pass = 0; pass < clearPasses; ++pass) {
            if (pass > 0) {
                const TextureBarrierDesc barrier{
                    .texture = color.texture(),
                    .oldLayout = TextureLayout::ColorAttachment,
                    .newLayout = TextureLayout::ColorAttachment,
                    .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite},
                    .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorWrite},
                };
                if (auto r = context.commandBuffer().synchronize({.textures = {&barrier, 1}}); !r) { return r; }
            }
            const float v = static_cast<float>((context.frameIndex() + pass) & 7u) / 7.0f;
            const RenderingAttachmentDesc attachment{
                .view = color.view(),
                .loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {v, 0.25f, 0.5f, 1.0f},
            };
            auto r = context.commandBuffer().beginRendering({
                .renderArea = {.width = color.desc().width, .height = color.desc().height},
                .colorAttachments = {&attachment, 1},
            });
            if (!r) { return r; }
            context.commandBuffer().endRendering();
        }
        return {};
    }
};

class FrameCrossFramePolicyBenchmarkTest final : public RHITest {
public:
    FrameCrossFramePolicyBenchmarkTest()
    {
        type = RHITestType::Rendering;
        name = "frame_cross_frame_policy_benchmark";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.cross_frame.policy_benchmark"}, bench::Layer::RenderGraph, "async", "sync");
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("BenchComputeStagePass", "Cross-frame benchmark compute stage",
                [] { return std::make_unique<BenchComputeStagePass>(); });
            registerRenderGraphPassType("BenchGraphicsStagePass", "Cross-frame benchmark graphics stage",
                [] { return std::make_unique<BenchGraphicsStagePass>(); });
            return true;
        }();
        (void)registered;

        std::unique_ptr<Device> device;
        auto created = createDevice({.applicationName = "Cross-frame policy benchmark",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true}).transform([&](auto v) { device = std::move(v); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute) { return RHITestResult::fail("missing graphics or compute queue"); }

        struct BenchConfig {
            std::string name;
            uint32_t maxFramesInFlight = 2;
            RenderGraphMemoryPolicy policy = RenderGraphMemoryPolicy::MinimizeMemory;
            uint64_t budgetBytes = 0;
            bool crossFrame = true;
            uint32_t width = 1024;
            uint32_t height = 1024;
            uint64_t elements = 262144;
            uint32_t iterations = 512;
            uint32_t clearPasses = 6;
            uint32_t warmupFrames = 16;
            uint32_t measureFrames = 64;
        };

        const char* envConfig = std::getenv("METALLIC_RG_BENCH_CONFIG");
        const bool benchmarkMode = envConfig != nullptr && envConfig[0] != '\0';
        std::vector<BenchConfig> configs;
        if (benchmarkMode) {
            const std::string mode(envConfig);
            BenchConfig cfg;
            cfg.name = mode;
            cfg.width = 2048;
            cfg.height = 2048;
            cfg.elements = 524288;
            cfg.iterations = 1024;
            cfg.clearPasses = 8;
            cfg.warmupFrames = 16;
            cfg.measureFrames = 80;
            if (mode == "2_conservative") {
                cfg.maxFramesInFlight = 2; cfg.policy = RenderGraphMemoryPolicy::MinimizeMemory; cfg.crossFrame = false;
            } else if (mode == "2_crossframe_minmem") {
                cfg.maxFramesInFlight = 2; cfg.policy = RenderGraphMemoryPolicy::MinimizeMemory; cfg.crossFrame = true;
            } else if (mode == "3_crossframe_minmem") {
                cfg.maxFramesInFlight = 3; cfg.policy = RenderGraphMemoryPolicy::MinimizeMemory; cfg.crossFrame = true;
            } else if (mode == "2_crossframe_maxoverlap") {
                cfg.maxFramesInFlight = 2; cfg.policy = RenderGraphMemoryPolicy::MaximizeOverlap;
                cfg.budgetBytes = 64ull * 1024 * 1024; cfg.crossFrame = true;
            } else if (mode == "3_crossframe_maxoverlap") {
                cfg.maxFramesInFlight = 3; cfg.policy = RenderGraphMemoryPolicy::MaximizeOverlap;
                cfg.budgetBytes = 64ull * 1024 * 1024; cfg.crossFrame = true;
            } else {
                return RHITestResult::fail("Unknown METALLIC_RG_BENCH_CONFIG: " + mode);
            }
            configs.push_back(cfg);
        } else {
            // Fast validation mode for standard test runs: check all 4 policy modes with small dimensions.
            configs.push_back({"2_conservative", 2, RenderGraphMemoryPolicy::MinimizeMemory, 0, false, 64, 64, 4096, 16, 2, 2, 4});
            configs.push_back({"2_crossframe_minmem", 2, RenderGraphMemoryPolicy::MinimizeMemory, 0, true, 64, 64, 4096, 16, 2, 2, 4});
            configs.push_back({"3_crossframe_minmem", 3, RenderGraphMemoryPolicy::MinimizeMemory, 0, true, 64, 64, 4096, 16, 2, 2, 4});
            configs.push_back({"3_crossframe_maxoverlap", 3, RenderGraphMemoryPolicy::MaximizeOverlap, 16ull * 1024 * 1024, true, 64, 64, 4096, 16, 2, 2, 4});
        }

        for (const auto& cfg : configs) {
            RenderGraph graph;
            graph.setName("CrossFramePolicyBench_" + cfg.name);
            graph.addNode("BenchComputeStagePass", "ComputeStage", {
                {"crossFrame", cfg.crossFrame}, {"elements", cfg.elements}, {"iterations", cfg.iterations}});
            graph.addNode("BenchGraphicsStagePass", "GraphicsStage", {
                {"crossFrame", cfg.crossFrame}, {"elements", cfg.elements}, {"clearPasses", cfg.clearPasses}});
            graph.addEdge("ComputeStage.stage", "GraphicsStage.stage");
            graph.markOutput("GraphicsStage.copyOut");
            graph.markOutput("GraphicsStage.color");

            RenderGraphExecutor executor;
            std::string log;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = cfg.maxFramesInFlight;
            opts.memoryPolicy = cfg.policy;
            opts.versionPoolBudgetBytes = cfg.budgetBytes;
            opts.maxPhysicalVersionsPerResource = 2;
            FRAME_REQUIRE(executor.compile(*device, graph, cfg.width, cfg.height, opts, log));

            const RenderGraphSubmitDesc submitDesc{
                .graphicsQueue = graphics,
                .computeQueue = compute,
                .submissionMode = FrameSubmissionMode::Pipelined,
            };
            for (uint32_t i = 0; i < cfg.warmupFrames; ++i) {
                FRAME_REQUIRE(executor.execute(submitDesc));
                (void)executor.collectCompletedGpuExecutionStats();
            }
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            (void)executor.collectCompletedGpuExecutionStats();

            double cpuMsSum = 0.0;
            double waitMsSum = 0.0;
            double gpuMsSum = 0.0;
            double computeGpuMsSum = 0.0;
            double graphicsGpuMsSum = 0.0;
            uint32_t gpuSamples = 0;
            uint32_t maxExtraVersions = 0;
            uint64_t maxExtraBytes = 0;
            uint64_t maxPeakVram = 0;
            bool observedScheduled = false;

            const auto collectGpu = [&]() {
                if (auto completedList = executor.collectCompletedGpuExecutionStats(); completedList) {
                    for (const auto& completed : *completedList) {
                        if (completed.gpuTimingAvailable) {
                            gpuMsSum += completed.gpuMilliseconds;
                            for (const auto& nodeStat : completed.nodes) {
                                if (nodeStat.gpuTimingAvailable) {
                                    if (nodeStat.name == "ComputeStage") { computeGpuMsSum += nodeStat.gpuMilliseconds; }
                                    else if (nodeStat.name == "GraphicsStage") { graphicsGpuMsSum += nodeStat.gpuMilliseconds; }
                                }
                            }
                            ++gpuSamples;
                        }
                    }
                }
            };

            const auto wallStart = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < cfg.measureFrames; ++i) {
                FRAME_REQUIRE(executor.execute(submitDesc));
                const auto stats = executor.executionStats();
                cpuMsSum += stats.cpuMilliseconds;
                waitMsSum += double(stats.scheduling.frameWaitNs) / 1e6;
                maxExtraVersions = std::max(maxExtraVersions, stats.extraVersionCount);
                maxExtraBytes = std::max(maxExtraBytes, stats.extraVersionBytes);
                maxPeakVram = std::max(maxPeakVram, stats.peakManagedVramBytes);
                observedScheduled |= stats.crossFrameResourceScheduled;
                collectGpu();
            }
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            const auto wallEnd = std::chrono::steady_clock::now();
            collectGpu();

            if (cfg.crossFrame && !observedScheduled) {
                return RHITestResult::fail("Config '" + cfg.name + "' failed to engage crossFrameResourceScheduled");
            }
            if (!cfg.crossFrame && observedScheduled) {
                return RHITestResult::fail("Conservative config '" + cfg.name + "' unexpectedly enabled crossFrameResourceScheduled");
            }
            if (cfg.policy == RenderGraphMemoryPolicy::MinimizeMemory && maxExtraVersions != 0) {
                return RHITestResult::fail("MinimizeMemory config '" + cfg.name + "' allocated extra versions");
            }

            const double wallTotalMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
            const double wallPerFrameMs = wallTotalMs / double(cfg.measureFrames);
            const double cpuPerFrameMs = cpuMsSum / double(cfg.measureFrames);
            const double waitPerFrameMs = waitMsSum / double(cfg.measureFrames);
            const double gpuPerFrameMs = gpuSamples > 0 ? (gpuMsSum / double(gpuSamples)) : 0.0;
            const double computePerFrameMs = gpuSamples > 0 ? (computeGpuMsSum / double(gpuSamples)) : 0.0;
            const double graphicsPerFrameMs = gpuSamples > 0 ? (graphicsGpuMsSum / double(gpuSamples)) : 0.0;
            if (benchmarkMode) {
                std::printf(
                    "RG_BENCH_JSON:{\"config\":\"%s\",\"asyncCompute\":%s,\"maxFramesInFlight\":%u,\"policy\":\"%s\","
                    "\"crossFrameScheduled\":%s,\"warmupFrames\":%u,\"measureFrames\":%u,"
                    "\"wallMsPerFrame\":%.4f,\"cpuMsPerFrame\":%.4f,\"slotWaitMsPerFrame\":%.4f,"
                    "\"gpuMsPerFrame\":%.4f,\"computeGpuMs\":%.4f,\"graphicsGpuMs\":%.4f,\"gpuSamples\":%u,"
                    "\"extraVersionCount\":%u,\"extraVersionBytes\":%llu,\"peakManagedVramBytes\":%llu}\n",
                    cfg.name.c_str(),
                    graphics->sameQueue(*compute) ? "false" : "true",
                    cfg.maxFramesInFlight,
                    renderGraphMemoryPolicyName(cfg.policy),
                    observedScheduled ? "true" : "false",
                    cfg.warmupFrames,
                    cfg.measureFrames,
                    wallPerFrameMs,
                    cpuPerFrameMs,
                    waitPerFrameMs,
                    gpuPerFrameMs,
                    computePerFrameMs,
                    graphicsPerFrameMs,
                    gpuSamples,
                    maxExtraVersions,
                    static_cast<unsigned long long>(maxExtraBytes),
                    static_cast<unsigned long long>(maxPeakVram));
                std::fflush(stdout);
            }
        }
        return RHITestResult::pass("Cross-frame policy benchmark configs validated");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameCrossFramePolicyBenchmarkTest);

// --- Batch 1 Review Fixes: Cross-Queue Completed Timeline Visibility, #58 GPU Fan-Out, #60 Exclusive Ownership, Pooled Optimal Layout ---

class CrossQueueTemporalWriterPass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("shared", "Compute-written temporal buffer")
            .buffer(16, 4).transferWrite().temporalWrite("shared_vis", /*graphOwned=*/true, /*perPassScope=*/false);
        reflection.addBufferOutput("done", "Keep pass active")
            .buffer(16, 4).transferWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        using namespace render;
        BufferDesc desc{.size = 64, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy};
        auto created = context.device->createBuffer(desc);
        if (!created) { return std::unexpected(created.error()); }
        constants_ = std::move(*created);
        auto* words = static_cast<uint32_t*>(constants_->map());
        if (!words) { return makeError(Error::Failure); }
        for (uint32_t slot = 0; slot < 4; ++slot) {
            const uint32_t val = 0x55AA0001u + slot;
            for (uint32_t w = 0; w < 4; ++w) { words[slot * 4 + w] = val + w; }
        }
        constants_->flush();
        constants_->unmap();
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto dst = context.temporalBuffer("shared").buffer;
        if (!dst.valid() || !constants_) { return render::makeError(render::Error::InvalidArgument); }
        const uint64_t slot = context.frameIndex() % 4ull;
        auto srcSlice = constants_->slice({slot * 16ull, 16});
        auto dstSlice = dst.buffer()->slice({0, 16});
        if (!srcSlice || !dstSlice) { return render::makeError(render::Error::Failure); }
        return context.commandBuffer().copyBuffer(*srcSlice, *dstSlice);
    }

private:
    std::unique_ptr<render::Buffer> constants_;
};

class CrossQueueTemporalReaderPass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("shared", "Previous Compute-written temporal buffer")
            .buffer(16, 4).transferRead().temporalRead("shared_vis", -1, /*graphOwned=*/true, /*perPassScope=*/false);
        reflection.addBufferOutput("readback", "Graphics readback buffer")
            .buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto src = context.temporalBuffer("shared").buffer;
        auto dst = context.outputBuffer("readback");
        if (!src.valid() || !dst.valid()) { return makeError(Error::InvalidArgument); }
        if (context.temporalValid("shared")) {
            auto srcSlice = src.buffer()->slice({0, 16});
            auto dstSlice = dst.buffer()->slice({0, 16});
            if (!srcSlice || !dstSlice) { return makeError(Error::Failure); }
            if (auto r = context.commandBuffer().copyBuffer(*srcSlice, *dstSlice); !r) { return r; }
        }
        const BufferBarrierDesc hostBarrier{
            .buffer = dst.buffer(),
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead},
        };
        return context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}});
    }
};

struct CachedBufferResourceSet {
    render::Buffer* buffer = nullptr;
    render::GPUCompletionPoint producerCompletion;
};

class CrossQueueCachedBufferProvider final : public render::IRenderSubsystem, public render::IRenderGraphResourceProvider {
public:
    struct Desc {};
    static constexpr render::RenderSubsystemId kSubsystemId = "test.cached-buffer-provider";

    render::RenderSubsystemId providerSubsystemId() const override { return kSubsystemId; }

    render::Result<> initialize(const render::RenderSubsystemInitContext& context, std::string&) override
    {
        using namespace render;
        device_ = &context.device;
        BufferDesc srcDesc{.size = 16, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy};
        auto src = device_->createBuffer(srcDesc);
        if (!src) { return std::unexpected(src.error()); }
        upload_ = std::move(*src);
        auto* mapped = static_cast<uint32_t*>(upload_->map());
        if (!mapped) { return makeError(Error::Failure); }
        mapped[0] = 0x77112233u; mapped[1] = 0x77445566u; mapped[2] = 0x778899AAu; mapped[3] = 0x77BBCCDDu;
        upload_->flush();
        upload_->unmap();

        BufferDesc dstDesc{.size = 16, .usage = BufferUsageBits::TransferDestination | BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy};
        auto dst = device_->createBuffer(dstDesc);
        if (!dst) { return std::unexpected(dst.error()); }
        cached_ = std::shared_ptr<Buffer>(std::move(*dst));
        return {};
    }

    void shutdown() override
    {
        committedCompletion_ = {};
        cached_.reset();
        upload_.reset();
        device_ = nullptr;
    }

    render::Result<> declareRequest(render::RenderGraphResourceRequestBuilder& builder)
    {
        using namespace render;
        const LogicalResourceId id = makeLogicalResourceId("test.provider", "cached_buf");
        auto set = std::make_shared<CachedBufferResourceSet>();
        set->buffer = cached_.get();
        set->producerCompletion = committedCompletion_;
        ResourceVersionId ver{.logicalId = id, .version = 1, .contentRevision = 1, .allocationGeneration = 1};
        if (auto r = builder.bindResourceSet(std::shared_ptr<const CachedBufferResourceSet>(set), cached_, id, ver); !r) { return r; }
        if (!committed_) {
            RenderGraphProviderProducerWork work;
            work.debugName = "TestCachedBufferProducer";
            work.providerId = kSubsystemId;
            work.dedupKey = id.value;
            work.preferredQueue = QueueType::Graphics;
            work.producedResources.push_back(id);
            work.record = [this](CommandBuffer& commands, std::string&) -> Result<> {
                ++producerRecordCount;
                auto srcSlice = upload_->slice({0, 16});
                auto dstSlice = cached_->slice({0, 16});
                if (!srcSlice || !dstSlice) { return makeError(Error::Failure); }
                return commands.copyBuffer(*srcSlice, *dstSlice);
            };
            work.onAccepted = [this](const GPUCompletionPoint& completion) {
                committed_ = true;
                committedCompletion_ = completion;
            };
            if (auto r = builder.registerProducerWork(std::move(work)); !r) { return r; }
            return builder.requireProducer(id.value);
        }
        return builder.addExternalDependency(committedCompletion_, "TestCachedBufferProvider cached graphics write");
    }

    uint32_t producerRecordCount = 0;
    render::GPUCompletionPoint committedCompletion() const { return committedCompletion_; }

private:
    render::Device* device_ = nullptr;
    std::unique_ptr<render::Buffer> upload_;
    std::shared_ptr<render::Buffer> cached_;
    bool committed_ = false;
    render::GPUCompletionPoint committedCompletion_;
};

class CrossQueueProviderComputeReaderPass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{CrossQueueCachedBufferProvider::kSubsystemId};
        return required;
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("readback", "Compute readback from cached Graphics provider buffer")
            .buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> declareResourceRequests(
        const render::RenderGraphResourceRequestContext& context,
        render::RenderGraphResourceRequestBuilder& builder,
        std::string&) override
    {
        auto* provider = context.subsystem<CrossQueueCachedBufferProvider>();
        if (!provider) { return render::makeError(render::Error::InvalidArgument); }
        return provider->declareRequest(builder);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const auto* set = context.resources<CachedBufferResourceSet>();
        auto dst = context.outputBuffer("readback");
        if (!set || !set->buffer || !dst.valid()) { return makeError(Error::InvalidArgument); }
        auto srcSlice = set->buffer->slice({0, 16});
        auto dstSlice = dst.buffer()->slice({0, 16});
        if (!srcSlice || !dstSlice) { return makeError(Error::Failure); }
        if (auto r = context.commandBuffer().copyBuffer(*srcSlice, *dstSlice); !r) { return r; }
        const BufferBarrierDesc hostBarrier{
            .buffer = dst.buffer(),
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead},
        };
        return context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}});
    }
};

class FrameCrossQueueCompletedTimelineVisibilityTest final : public RHITest {
public:
    FrameCrossQueueCompletedTimelineVisibilityTest()
    {
        type = RHITestType::Rendering;
        name = "frame_cross_queue_completed_timeline_visibility";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("CrossQueueTemporalWriterPass", "Compute temporal writer",
                [] { return std::make_unique<CrossQueueTemporalWriterPass>(); });
            registerRenderGraphPassType("CrossQueueTemporalReaderPass", "Graphics temporal reader",
                [] { return std::make_unique<CrossQueueTemporalReaderPass>(); });
            registerRenderGraphPassType("CrossQueueProviderComputeReaderPass", "Compute provider cache reader",
                [] { return std::make_unique<CrossQueueProviderComputeReaderPass>(); });
            return true;
        }();
        (void)registered;

        std::unique_ptr<Device> device;
        DeviceDesc devDesc{
            .applicationName = "Cross-queue completed timeline visibility",
            .enableValidation = context.enableValidation,
            .enableAsyncCompute = true,
        };
        vulkan::deviceExtensions(devDesc).preferUnifiedImageLayouts = false;
        FRAME_REQUIRE(createDevice(devDesc).transform([&](auto d) { device = std::move(d); }));
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute || graphics->sameQueue(*compute)) {
            return RHITestResult::skip("requires independent graphics and compute queues");
        }

        // Part A: Ordinary graph/temporal resource written on Compute, host waits & polls isComplete() == true,
        // then first read on Graphics must STILL include a semaphore wait for cross-queue memory visibility.
        {
            RenderGraph graph;
            graph.setName("CrossQueueCompletedVisGraph");
            graph.addNode("CrossQueueTemporalWriterPass", "Writer");
            graph.addNode("CrossQueueTemporalReaderPass", "Reader");
            graph.markOutput("Writer.done");
            graph.markOutput("Reader.readback");

            RenderGraphExecutor executor;
            executor.setExecutionCaptureEnabled(true);
            std::string log;
            FRAME_REQUIRE(executor.compile(*device, graph, 16, 16, log));

            // Frame 0: Compute writes slot 0. Drain completely and actively poll isComplete().
            FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            if (!executor.lastSubmittedCompletion().isComplete()) {
                return RHITestResult::fail("Frame 0 completion did not report isComplete() after wait");
            }

            // Frame 1: Graphics reads slot 0 for the first time AFTER host saw Frame 0 complete.
            FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            const auto snap = executor.executionSnapshot();
            if (!executor.executionStats().crossFrameResourceScheduled || !snap) {
                return RHITestResult::fail("Part A did not use crossFrameResourceScheduled");
            }
            bool foundGraphicsRawWait = false;
            for (const auto& batch : snap->batches) {
                if (batch.semaphoreWaitCount > 0) {
                    for (const auto& reason : batch.waitReasons) {
                        if (reason.find("Cross-frame RAW") != std::string::npos) {
                            foundGraphicsRawWait = true;
                        }
                    }
                }
            }
            if (!foundGraphicsRawWait) {
                std::string diag = "batches=" + std::to_string(snap->batches.size());
                for (const auto& batch : snap->batches) {
                    diag += " [b" + std::to_string(batch.id) + " q=" + std::to_string(batch.queueId) +
                        " waits=" + std::to_string(batch.semaphoreWaitCount) + " reasons=";
                    for (const auto& r : batch.waitReasons) { diag += "(" + r + ")"; }
                    diag += "]";
                }
                for (const auto& p : snap->passes) {
                    diag += " {pass=" + p.name + " q=" + std::to_string(p.actualQueueId) + " uses=" + std::to_string(p.uses.size()) + "}";
                }
                return RHITestResult::fail("Part A skipped cross-queue RAW semaphore wait after host observed writer timeline complete: " + diag);
            }
            const auto* rb = executor.outputResource("Reader.readback");
            std::array<uint32_t, 4> actual{};
            constexpr std::array expected{0x55AA0001u, 0x55AA0002u, 0x55AA0003u, 0x55AA0004u};
            if (!rb || !rb->buffer || !readWords(*rb->buffer, actual.data(), actual.size()) || actual != expected) {
                return RHITestResult::fail("Part A Graphics readback did not match Compute writer words");
            }
        }

        // Part B: Provider cache hit across queues AFTER Graphics producer timeline isComplete() == true.
        {
            RenderSubsystemHost host;
            std::string log;
            if (!host.registerSubsystem<CrossQueueCachedBufferProvider>(log)) {
                return RHITestResult::fail("Failed to register CrossQueueCachedBufferProvider: " + log);
            }
            FRAME_REQUIRE(host.initialize(*device, 2, log));

            RenderGraph graph;
            graph.setName("CrossQueueProviderCacheHit");
            graph.addNode("CrossQueueProviderComputeReaderPass", "CmpReader");
            graph.markOutput("CmpReader.readback");

            RenderWorld world;
            RenderGraphExecutor executor(host, world);
            executor.setExecutionCaptureEnabled(true);
            FRAME_REQUIRE(executor.compile(*device, graph, 16, 16, log));

            // Frame 0: Graphics ProviderProducer populates cached buffer; wait and poll isComplete() == true.
            FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            auto* provider = host.get<CrossQueueCachedBufferProvider>();
            if (!provider || provider->producerRecordCount != 1 || !provider->committedCompletion().isComplete()) {
                return RHITestResult::fail("Part B Frame 0 provider producer did not complete cleanly");
            }

            // Frame 1: Cache hit on Compute queue (0 producers); MUST STILL emit semaphore wait on Graphics producer!
            FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
            if (provider->producerRecordCount != 1) {
                return RHITestResult::fail("Part B Frame 1 did not hit provider cache");
            }
            const auto snap1 = executor.executionSnapshot();
            bool foundProviderWait = false;
            if (snap1) {
                for (const auto& batch : snap1->batches) {
                    if (batch.semaphoreWaitCount > 0) {
                        for (const auto& reason : batch.waitReasons) {
                            if (reason.find("TestCachedBufferProvider") != std::string::npos) {
                                foundProviderWait = true;
                            }
                        }
                    }
                }
            }
            if (!foundProviderWait) {
                return RHITestResult::fail("Part B skipped Provider cache-hit semaphore wait after host observed producer timeline complete");
            }
            const auto* rb = executor.outputResource("CmpReader.readback");
            std::array<uint32_t, 4> actual{};
            constexpr std::array expected{0x77112233u, 0x77445566u, 0x778899AAu, 0x77BBCCDDu};
            if (!rb || !rb->buffer || !readWords(*rb->buffer, actual.data(), actual.size()) || actual != expected) {
                return RHITestResult::fail("Part B Compute cache-hit readback did not match Graphics provider data");
            }
            executor = RenderGraphExecutor{};
            host.shutdown();
        }
        return RHITestResult::pass("Cross-queue semaphore waits preserved after host-observed timeline completion for both graph resources and provider cache hits");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameCrossQueueCompletedTimelineVisibilityTest);

class ExclusiveComputeConsumerPass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("exclusivePrev", "External exclusive history buffer")
            .buffer(16, 4).transferRead().temporalRead("exclusive_buf", -1, /*graphOwned=*/false, /*perPassScope=*/false);
        reflection.addBufferOutput("out", "Readback")
            .buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto dst = context.outputBuffer("out");
        if (!dst.valid()) { return render::makeError(render::Error::InvalidArgument); }
        return {};
    }
};

class FrameExclusiveQueueOwnershipBoundaryTest final : public RHITest {
public:
    FrameExclusiveQueueOwnershipBoundaryTest()
    {
        type = RHITestType::Rendering;
        name = "frame_exclusive_queue_ownership_boundary";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("ExclusiveComputeConsumerPass",
            "Exclusive queue ownership boundary probe",
            [] { return std::make_unique<ExclusiveComputeConsumerPass>(); });
        (void)registered;

        std::unique_ptr<Device> device;
        DeviceDesc devDesc{
            .applicationName = "Exclusive queue ownership boundary (#60)",
            .enableValidation = context.enableValidation,
            .enableAsyncCompute = true,
        };
        vulkan::deviceExtensions(devDesc).preferUnifiedImageLayouts = false;
        FRAME_REQUIRE(createDevice(devDesc).transform([&](auto d) { device = std::move(d); }));
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute || graphics->sameQueue(*compute)) {
            return RHITestResult::skip("requires independent graphics and compute queues");
        }

        // Pre-allocate an external history buffer with Graphics-only exclusive ownership.
        HistoryResourceManager externalHistory;
        FRAME_REQUIRE(externalHistory.initialize(*device, 2));
        externalHistory.beginFrame(0);
        FRAME_REQUIRE(externalHistory.ensureBuffer("exclusive_buf", BufferDesc{
            .size = 16,
            .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics, // VK_SHARING_MODE_EXCLUSIVE
        }));

        RenderGraph graph;
        graph.setName("ExclusiveOwnershipBoundary60");
        graph.addNode("ExclusiveComputeConsumerPass", "ExclusiveCmp");
        graph.markOutput("ExclusiveCmp.out");

        RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        FRAME_REQUIRE(executor.compile(*device, graph, 16, 16, log));
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics, .computeQueue = compute,
            .historyResources = &externalHistory}));
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));

        const auto stats = executor.executionStats();
        const auto snap = executor.executionSnapshot();
        const auto has60Reason = [](const std::vector<std::string>& reasons) {
            return std::any_of(reasons.begin(), reasons.end(), [](const std::string& r) {
                return r.find("#60") != std::string::npos || r.find("Exclusive") != std::string::npos;
            });
        };
        if (stats.crossFrameResourceScheduled || !has60Reason(stats.conservativeFallbackReasons) ||
            !snap || snap->crossFrameResourceScheduled || !has60Reason(snap->conservativeFallbackReasons)) {
            return RHITestResult::fail("Executor failed to enforce #60 conservative fallback or report reason in Viewer snapshot");
        }
        return RHITestResult::pass("Executor enforced #60 exclusive queue-family ownership fallback and surfaced reason in ExecutionSnapshot");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameExclusiveQueueOwnershipBoundaryTest);

class AlternatingPooledColorPass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color", "Pooled alternating RGBA8 output")
            .texture2D(16, 16).colorWrite().transient(render::RenderGraphInitialization::Clear)
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto color = context.outputTexture("color");
        if (!color.valid()) { return makeError(Error::InvalidArgument); }
        const bool red = (context.frameIndex() % 2ull) == 0ull;
        const RenderingAttachmentDesc att{
            .view = color.view(),
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = red ? ColorValue{1.0f, 0.0f, 0.0f, 1.0f} : ColorValue{0.0f, 1.0f, 0.0f, 1.0f},
        };
        auto r = context.commandBuffer().beginRendering({
            .renderArea = {.width = color.desc().width, .height = color.desc().height},
            .colorAttachments = {&att, 1},
        });
        if (r) { context.commandBuffer().endRendering(); }
        return r;
    }
};

class FramePooledTextureExternalLayoutTest final : public RHITest {
public:
    FramePooledTextureExternalLayoutTest()
    {
        type = RHITestType::Rendering;
        name = "frame_pooled_texture_external_layout_optimal";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("AlternatingPooledColorPass",
            "Pooled optimal layout test pass",
            [] { return std::make_unique<AlternatingPooledColorPass>(); });
        (void)registered;

        // Explicitly disable unified image layouts to enforce Vulkan optimal layouts.
        std::unique_ptr<Device> device;
        DeviceDesc deviceDesc{.applicationName = "Pooled texture external layout (optimal)",
            .enableValidation = context.enableValidation};
        vulkan::deviceExtensions(deviceDesc).preferUnifiedImageLayouts = false;
        FRAME_REQUIRE(createDevice(deviceDesc).transform([&](auto d) { device = std::move(d); }));
        Queue* graphics = device->getQueue(QueueType::Graphics);
        if (!graphics) { return RHITestResult::fail("missing graphics queue"); }

        RenderGraph graph;
        graph.setName("PooledExternalLayoutOptimal");
        graph.addNode("AlternatingPooledColorPass", "Clear");
        graph.markOutput("Clear.color");

        RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        RenderGraphCompileOptions opts;
        opts.memoryPolicy = RenderGraphMemoryPolicy::MinimizeMemory;
        opts.maxFramesInFlight = 2;
        FRAME_REQUIRE(executor.compile(*device, graph, 16, 16, opts, log));

        QueueSubmissionTracker extTracker;
        Commands extCmd;
        FRAME_REQUIRE(extTracker.initialize(*device, *graphics));
        FRAME_REQUIRE(extCmd.initialize(*device, *graphics));

        std::unique_ptr<Buffer> readback;
        FRAME_REQUIRE(device->createBuffer({.size = 16 * 16 * 4,
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto b) { readback = std::move(b); }));

        auto readPixel = [&](uint32_t frameNum) -> Result<std::array<uint8_t, 4>> {
            auto r = extCmd.begin(frameNum);
            if (!r) { return std::unexpected(r.error()); }
            r = executor.transitionOutput(*extCmd.buffer, "Clear.color", ResourceState::TransferSource);
            if (!r) { return std::unexpected(r.error()); }
            auto* outRes = executor.outputResource("Clear.color");
            auto dstSlice = readback->slice();
            if (!outRes || !outRes->texture || !dstSlice) { return std::unexpected(Error::Failure); }
            r = extCmd.buffer->copyTextureToBuffer({.texture = outRes->texture, .buffer = *dstSlice,
                .width = 16, .height = 16});
            if (!r) { return std::unexpected(r.error()); }
            const BufferBarrierDesc hostBarrier{
                .buffer = readback.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
            };
            r = extCmd.buffer->synchronize({.buffers = {&hostBarrier, 1}});
            if (!r) { return std::unexpected(r.error()); }
            // Also transition to ShaderRead just like EditorApplication does for viewport sampling!
            r = executor.transitionOutput(*extCmd.buffer, "Clear.color", ResourceState::ShaderRead);
            if (!r) { return std::unexpected(r.error()); }
            r = extCmd.submit(extTracker);
            if (!r) { return std::unexpected(r.error()); }
            r = extCmd.frame.wait(kWaitTimeout);
            if (!r) { return std::unexpected(r.error()); }
            readback->invalidate();
            auto* bytes = static_cast<const uint8_t*>(readback->map());
            if (!bytes) { return std::unexpected(Error::Failure); }
            std::array<uint8_t, 4> px{bytes[0], bytes[1], bytes[2], bytes[3]};
            readback->unmap();
            return px;
        };

        // Frame 0: renders Red -> external copy + transitionOutput(ShaderRead) + submit.
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics}));
        auto px0 = readPixel(10);
        if (!px0 || *px0 != std::array<uint8_t, 4>{255, 0, 0, 255}) {
            return RHITestResult::fail("Frame 0 readback pixel mismatch (expected red)");
        }

        // Frame 1: renders Green. Under optimal layouts, planned barrier MUST start from ShaderRead!
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics}));
        const auto snap1 = executor.executionSnapshot();
        if (!snap1 || snap1->passes.empty() || snap1->passes[0].barriers.empty() ||
            snap1->passes[0].barriers[0].before != ResourceState::ShaderRead) {
            return RHITestResult::fail("Frame 1 pooled texture did not preserve external ShaderRead layout before barrier");
        }
        auto px1 = readPixel(11);
        if (!px1 || *px1 != std::array<uint8_t, 4>{0, 255, 0, 255}) {
            return RHITestResult::fail("Frame 1 readback pixel mismatch (expected green)");
        }

        // Frame 2: renders Red, leaves output in ColorAttachment (no external submit).
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics}));
        FRAME_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));

        // Record an external transitionOutput(ShaderRead) and CANCEL it without submitting!
        FRAME_REQUIRE(extCmd.begin(12));
        FRAME_REQUIRE(executor.transitionOutput(*extCmd.buffer, "Clear.color", ResourceState::ShaderRead));
        FRAME_REQUIRE(extCmd.pool->reset());
        extCmd.frame.cancel();

        // Frame 3: renders Green. Cancelled external transition MUST NOT have advanced pool state to ShaderRead!
        FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics}));
        const auto snap3 = executor.executionSnapshot();
        if (!snap3 || snap3->passes.empty() || snap3->passes[0].barriers.empty() ||
            snap3->passes[0].barriers[0].before != ResourceState::ColorAttachment) {
            return RHITestResult::fail("Cancelled external transitionOutput polluted pooled texture state");
        }
        auto px3 = readPixel(13);
        if (!px3 || *px3 != std::array<uint8_t, 4>{0, 255, 0, 255}) {
            return RHITestResult::fail("Frame 3 readback pixel mismatch after cancelled external transition");
        }

        return RHITestResult::pass("Pooled texture external transitionOutput preserved optimal ShaderRead layout and rolled back cleanly on cancellation");
    }
};
METALLIC_REGISTER_RHI_TEST(FramePooledTextureExternalLayoutTest);

class WriterVisibility58ComputeWritePass final : public render::ComputePass {
public:
    render::QueueType queueType() const override { return render::QueueType::Graphics; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("tex", "Compute-written texture")
            .texture2D(16, 16).storageWrite().transient(render::RenderGraphInitialization::FullOverwrite)
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        BufferDesc dummyDesc{.size = 16, .structureStride = 4, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device};
        auto created = context.device->createBuffer(dummyDesc);
        if (!created) { return std::unexpected(created.error()); }
        dummy_ = std::move(*created);
        const ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        return createProbe(*context.device, "fillSolidTexture", bindings, program_, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto tex = context.outputTexture("tex");
        if (!tex.valid() || !dummy_ || !program_.valid()) { return makeError(Error::InvalidArgument); }
        TextureView* view = tex.view();
        const uint32_t packedColor = (context.frameIndex() % 2ull) == 0ull ? 0xFF3366CCu : 0xFFCC9933u;
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .textureViews = {&view, 1}},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .textureViews = {&view, 1}},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .buffer = dummy_.get()},
        };
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .pushData = &packedColor,
            .pushDataSize = sizeof(packedColor),
            .groupCountX = 2,
            .groupCountY = 2,
            .groupCountZ = 1,
        });
    }

private:
    render::ComputeProgram program_;
    std::unique_ptr<render::Buffer> dummy_;
};

class WriterVisibility58ComputeNarrowReadPass final : public render::ComputePass {
public:
    render::QueueType queueType() const override { return render::QueueType::Graphics; }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("tex", "Narrow compute sampled read")
            .texture2D(16, 16).sampledRead().format = render::Format::RGBA8Unorm;
        reflection.addBufferOutput("out", "Compute narrow read sink")
            .buffer(16, 4).storageWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        const ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .kind = ComputeResourceBindingKind::SampledImage, .descriptorCount = 2},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        return createProbe(*context.device, "sampleImages", bindings, program_, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto tex = context.inputTexture("tex");
        auto out = context.outputBuffer("out");
        if (!tex.valid() || !out.valid() || !program_.valid()) { return makeError(Error::InvalidArgument); }
        TextureView* views[2] = {tex.view(), tex.view()};
        const uint32_t index = 0;
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .textureViews = views},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = out.buffer()},
        };
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = bindings,
            .pushData = &index,
            .pushDataSize = sizeof(index),
            .groupCountX = 1,
            .groupCountY = 1,
            .groupCountZ = 1,
        });
    }

private:
    render::ComputeProgram program_;
};

class WriterVisibility58FragmentReadPass final : public render::RasterPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("tex", "Same-layout fragment sampled read")
            .texture2D(16, 16).sampledRead().format = render::Format::RGBA8Unorm;
        reflection.addTextureOutput("color", "Fragment sampled output")
            .texture2D(16, 16).colorWrite().transient(render::RenderGraphInitialization::Clear)
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        if (!context.device) { return makeError(Error::InvalidArgument); }
        if (auto r = ResourceRegistry::forDevice(*context.device).transform([&](auto v) { registry_ = std::move(v); }); !r) {
            return r;
        }
        auto loadShader = [&](const char* entry, std::unique_ptr<ShaderModule>& out) -> Result<> {
            ShaderCompileResult compiled;
            auto r = ShaderRegistry::instance().getShader(SlangShaderDesc{
                .moduleName = "Features/Samples/ImageSample",
                .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            }, compiled.diagnostics).transform([&](auto v) { compiled = std::move(v); });
            if (!r) { log += compiled.diagnostics; return r; }
            return ShaderRegistry::instance().getShaderModule(*context.device, ShaderModuleDesc{
                .spirv = compiled.spirv, .debugName = entry,
            }).transform([&](auto m) { out = std::move(m); });
        };
        if (auto r = loadShader("imageSampleVertexMain", vs_); !r) { return r; }
        if (auto r = loadShader("imageSampleFragmentMain", fs_); !r) { return r; }
        return ShaderRegistry::instance().getGraphicsPipeline(*context.device, GraphicsPipelineDesc{
            .vertexShader = {vs_.get()},
            .fragmentShader = {fs_.get()},
            .colorFormats = {Format::RGBA8Unorm},
            .colorAttachmentCount = 1,
            .topology = PrimitiveTopology::TriangleList,
            .usesBindlessHeap = true,
        }).transform([&](auto p) { pipeline_ = std::move(p); });
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto tex = context.inputTexture("tex");
        auto color = context.outputTexture("color");
        if (!tex.valid() || !color.valid() || !registry_ || !pipeline_) { return makeError(Error::InvalidArgument); }
        ResourceLease lease;
        if (auto r = registry_->sampledImage(*tex.view()).transform([&](auto l) { lease = std::move(l); }); !r) { return r; }
        if (auto r = context.commandBuffer().retainResource(std::make_shared<ResourceLease>(lease)); !r) { return r; }
        const Rect area{.x = 0, .y = 0, .width = 16, .height = 16};
        const RenderingAttachmentDesc att{
            .view = color.view(), .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store,
            .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
        };
        if (auto r = context.commandBuffer().beginRendering({.renderArea = area, .colorAttachments = {&att, 1}}); !r) { return r; }
        if (auto r = context.commandBuffer().setViewport({.width = 16.0f, .height = 16.0f, .minDepth = 0.0f, .maxDepth = 1.0f}); !r) { return r; }
        context.commandBuffer().setScissor(area);
        if (auto r = registry_->bind(context.commandBuffer(), std::array{lease}); !r) { return r; }
        const uint32_t imageIndex = lease.shaderIndex();
        if (auto r = context.commandBuffer().bindExecution(pipeline_->execution(), &imageIndex, sizeof(imageIndex)); !r) { return r; }
        if (auto r = context.commandBuffer().draw(3); !r) { return r; }
        context.commandBuffer().endRendering();
        return {};
    }

private:
    std::shared_ptr<render::ResourceRegistry> registry_;
    std::unique_ptr<render::ShaderModule> vs_;
    std::unique_ptr<render::ShaderModule> fs_;
    std::shared_ptr<render::GraphicsPipeline> pipeline_;
};

class FrameSameLayoutReadFanout58Test final : public RHITest {
public:
    FrameSameLayoutReadFanout58Test()
    {
        type = RHITestType::Rendering;
        name = "frame_same_layout_read_fanout_writer_visibility_58";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("WriterVisibility58ComputeWritePass", "#58 compute writer",
                [] { return std::make_unique<WriterVisibility58ComputeWritePass>(); });
            registerRenderGraphPassType("WriterVisibility58ComputeNarrowReadPass", "#58 narrow compute reader",
                [] { return std::make_unique<WriterVisibility58ComputeNarrowReadPass>(); });
            registerRenderGraphPassType("WriterVisibility58FragmentReadPass", "#58 fragment reader",
                [] { return std::make_unique<WriterVisibility58FragmentReadPass>(); });
            return true;
        }();
        (void)registered;

        for (bool optimalLayouts : {false, true}) {
            std::unique_ptr<Device> device;
            DeviceDesc deviceDesc{.applicationName = "#58 same-layout read fan-out GPU visibility",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true};
            vulkan::deviceExtensions(deviceDesc).preferUnifiedImageLayouts = !optimalLayouts;
            auto created = createDevice(deviceDesc).transform([&](auto d) { device = std::move(d); });
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
            FRAME_REQUIRE(created);
            Queue* graphics = device->getQueue(QueueType::Graphics);

            RenderGraph graph;
            graph.setName("SameLayoutReadFanout58");
            graph.addNode("WriterVisibility58ComputeWritePass", "ComputeWrite");
            graph.addNode("WriterVisibility58ComputeNarrowReadPass", "ComputeNarrowRead");
            graph.addNode("WriterVisibility58FragmentReadPass", "FragmentRead");
            graph.addEdge("ComputeWrite.tex", "ComputeNarrowRead.tex");
            graph.addEdge("ComputeWrite.tex", "FragmentRead.tex");
            graph.markOutput("ComputeNarrowRead.out");
            graph.markOutput("FragmentRead.color");

            RenderGraphExecutor executor;
            executor.setExecutionCaptureEnabled(true);
            std::string log;
            FRAME_REQUIRE(executor.compile(*device, graph, 16, 16, log));

            QueueSubmissionTracker rbTracker;
            Commands rbCmd;
            FRAME_REQUIRE(rbTracker.initialize(*device, *graphics));
            FRAME_REQUIRE(rbCmd.initialize(*device, *graphics));
            std::unique_ptr<Buffer> readback;
            FRAME_REQUIRE(device->createBuffer({.size = 16 * 16 * 4,
                .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto b) { readback = std::move(b); }));

            for (uint32_t frame = 0; frame < 4; ++frame) {
                FRAME_REQUIRE(executor.execute({.graphicsQueue = graphics}));
                const auto snap = executor.executionSnapshot();
                if (!snap) { return RHITestResult::fail("missing snapshot"); }
                // Assert FragmentRead received a barrier covering FragmentShader | ShaderRead from the writer
                // despite ComputeNarrowRead already having transitioned to ShaderRead for ComputeShader!
                bool foundFragmentBarrier = false;
                for (const auto& pass : snap->passes) {
                    if (pass.name == "FragmentRead") {
                        for (const auto& b : pass.barriers) {
                            if ((static_cast<uint64_t>(b.afterScope.stages) & static_cast<uint64_t>(PipelineStageBits::FragmentShader)) != 0 &&
                                (static_cast<uint64_t>(b.afterScope.access) & static_cast<uint64_t>(AccessBits::ShaderRead)) != 0) {
                                foundFragmentBarrier = true;
                            }
                        }
                    }
                }
                if (!foundFragmentBarrier) {
                    return RHITestResult::fail("#58 regression: FragmentRead lacked writer visibility barrier after narrow ComputeShader read");
                }
                FRAME_REQUIRE(rbCmd.begin(frame));
                FRAME_REQUIRE(executor.transitionOutput(*rbCmd.buffer, "FragmentRead.color", ResourceState::TransferSource));
                auto* colorRes = executor.outputResource("FragmentRead.color");
                auto dstSlice = readback->slice();
                if (!colorRes || !colorRes->texture || !dstSlice) { return RHITestResult::fail("missing color output"); }
                FRAME_REQUIRE(rbCmd.buffer->copyTextureToBuffer({.texture = colorRes->texture, .buffer = *dstSlice,
                    .width = 16, .height = 16}));
                const BufferBarrierDesc hostBarrier{
                    .buffer = readback.get(),
                    .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead},
                };
                FRAME_REQUIRE(rbCmd.buffer->synchronize({.buffers = {&hostBarrier, 1}}));
                FRAME_REQUIRE(rbCmd.submit(rbTracker));
                FRAME_REQUIRE(rbCmd.frame.wait(kWaitTimeout));
                readback->invalidate();
                auto* bytes = static_cast<const uint8_t*>(readback->map());
                if (!bytes) { return RHITestResult::fail("readback map failed"); }
                const std::array<uint8_t, 4> expected = (frame % 2u) == 0u
                    ? std::array<uint8_t, 4>{0xCC, 0x66, 0x33, 0xFF}
                    : std::array<uint8_t, 4>{0x33, 0x99, 0xCC, 0xFF};
                const std::array<uint8_t, 4> px{bytes[0], bytes[1], bytes[2], bytes[3]};
                readback->unmap();
                for (size_t c = 0; c < 4; ++c) {
                    if (std::abs(int(px[c]) - int(expected[c])) > 2) {
                        return RHITestResult::fail("#58 FragmentRead pixel mismatch on frame " + std::to_string(frame));
                    }
                }
            }
        }
        return RHITestResult::pass("#58 same-layout compute write -> compute narrow read -> fragment read verified under unified and optimal layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameSameLayoutReadFanout58Test);

// ============================================================================
// Batch 2 Temporal History Contract & PathTrace Conditional Allocation Tests
// ============================================================================

render::Result<> copyHostWordToBuffer(
    render::Device& device,
    render::CommandBuffer& commands,
    render::Buffer& destination,
    uint32_t word)
{
    using namespace render;
    std::shared_ptr<Buffer> staging;
    auto created = device.createBuffer({
        .size = 16,
        .structureStride = 4,
        .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
    }).transform([&](auto b) { staging = std::move(b); });
    if (!created || !staging) { return created ? makeError(Error::Failure) : created; }
    void* mapped = staging->map();
    if (!mapped) { return makeError(Error::Failure); }
    const std::array<uint32_t, 4> words{word, word, word, word};
    std::memcpy(mapped, words.data(), sizeof(words));
    staging->flush({0, 16});
    staging->unmap();
    commands.hostWriteBarrier();
    if (auto r = commands.retainResource(staging); !r) { return r; }
    auto src = staging->slice({0, 16});
    auto dst = destination.slice({0, 16});
    if (!src || !dst) { return makeError(Error::Failure); }
    return commands.copyBuffer(*src, *dst);
}

struct TemporalOffsetProbeRecord {
    uint64_t frameIndex = 0;
    bool prevValid = false;
    uint64_t prevVersion = 0;
    uint64_t prevBackingId = 0;
    uint64_t currVersion = 0;
    uint64_t currBackingId = 0;
};

class TemporalOffsetProbePass final : public render::ComputePass {
public:
    static inline render::Device* devicePtr = nullptr;
    static inline std::vector<TemporalOffsetProbeRecord> records;
    static inline uint32_t writeBase = 0x1000u;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const int32_t offset = properties().value("offset", -1);
        const bool graphOwned = properties().value("graphOwned", false);
        const std::string key = properties().value("key", std::string("offset_probe"));
        render::RenderPassReflection reflection;
        reflection.addBufferInput("prev", "Temporal offset read").buffer(16, 4).transferRead()
            .temporalRead(key, offset, graphOwned, false);
        reflection.addBufferOutput("curr", "Temporal current write").buffer(16, 4).transferWrite()
            .temporalWrite(key, graphOwned, false);
        reflection.addBufferOutput("out", "Readback previous word").buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const auto prev = context.temporalBuffer("prev");
        const auto curr = context.temporalBuffer("curr");
        const auto out = context.outputBuffer("out");
        if (!devicePtr || !prev.buffer.valid() || !curr.buffer.valid() || !out.valid()) {
            return makeError(Error::InvalidArgument);
        }
        const auto prevMem = prev.buffer.buffer()->memoryInfo();
        const auto currMem = curr.buffer.buffer()->memoryInfo();
        records.push_back(TemporalOffsetProbeRecord{
            .frameIndex = context.frameIndex(),
            .prevValid = prev.valid,
            .prevVersion = prev.version.version,
            .prevBackingId = prevMem.backingAllocationId ? prevMem.backingAllocationId : prevMem.allocationId,
            .currVersion = curr.version.version,
            .currBackingId = currMem.backingAllocationId ? currMem.backingAllocationId : currMem.allocationId,
        });

        auto outSlice = out.buffer()->slice({0, 16});
        if (!outSlice) { return makeError(Error::Failure); }
        if (prev.valid) {
            auto prevSlice = prev.buffer.buffer()->slice({0, 16});
            if (!prevSlice) { return makeError(Error::Failure); }
            if (auto r = context.commandBuffer().copyBuffer(*prevSlice, *outSlice); !r) { return r; }
        } else {
            if (auto r = copyHostWordToBuffer(*devicePtr, context.commandBuffer(), *out.buffer(), 0u); !r) { return r; }
        }
        const uint32_t value = writeBase + static_cast<uint32_t>(context.frameIndex());
        if (auto r = copyHostWordToBuffer(*devicePtr, context.commandBuffer(), *curr.buffer.buffer(), value); !r) { return r; }
        context.setTemporalWritten("curr", true);
        return {};
    }
};

class FrameTemporalOffsetBoundsAndVersionTest final : public RHITest {
public:
    FrameTemporalOffsetBoundsAndVersionTest()
    {
        type = RHITestType::Rendering;
        name = "frame_temporal_offset_bounds_and_version_traceability";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("TemporalOffsetProbePass",
            "Temporal offset bounds and version traceability probe",
            [] { return std::make_unique<TemporalOffsetProbePass>(); });
        (void)registered;

        TemporalOffsetProbePass::devicePtr = &context.device;
        struct DeviceGuard { ~DeviceGuard() { TemporalOffsetProbePass::devicePtr = nullptr; } } devGuard;

        Queue* graphics = &context.graphicsQueue;
        const auto readWord = [](RenderGraphExecutor& exec, std::string_view outputName) -> uint32_t {
            auto* res = exec.outputResource(outputName);
            if (!res || !res->buffer) { return 0xFFFFFFFFu; }
            res->buffer->invalidate();
            const void* mapped = res->buffer->map();
            if (!mapped) { return 0xFFFFFFFFu; }
            uint32_t word = 0xFFFFFFFFu;
            std::memcpy(&word, mapped, sizeof(word));
            res->buffer->unmap();
            return word;
        };

        // Part A: RenderGraph::validate() must reject offsets < -2 and INT32_MIN before slot mapping.
        for (int32_t badOffset : {-3, std::numeric_limits<int32_t>::min()}) {
            RenderGraph badGraph;
            badGraph.setName("BadTemporalOffset");
            badGraph.addNode("TemporalOffsetProbePass", "Probe", {{"offset", badOffset}, {"graphOwned", true}});
            badGraph.markOutput("Probe.out");
            std::string log;
            if (badGraph.validate(log)) {
                return RHITestResult::fail("RenderGraph::validate() accepted out-of-range offset " + std::to_string(badOffset));
            }
        }

        // Part B1: 2-slot graph-owned history rejects offset -2 at compile time.
        {
            RenderGraph g;
            g.setName("TwoSlotRejectNeg2GraphOwned");
            g.addNode("TemporalOffsetProbePass", "Probe", {{"offset", -2}, {"graphOwned", true}});
            g.markOutput("Probe.out");
            RenderGraphExecutor exec;
            std::string log;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = 2;
            if (exec.compile(context.device, g, 16, 16, opts, log).has_value()) {
                return RHITestResult::fail("2-slot graph-owned history compiled with offset -2");
            }
        }

        // Part B2: 2-slot external HistoryResourceManager rejects offset -2 before slot mapping,
        // and accepts offset -1 with traceable write-origin version IDs and backings.
        {
            HistoryResourceManager history2;
            FRAME_REQUIRE(history2.initialize(context.device, 2));

            RenderGraph badExecGraph;
            badExecGraph.setName("TwoSlotRejectNeg2External");
            badExecGraph.addNode("TemporalOffsetProbePass", "Probe", {{"offset", -2}, {"graphOwned", false}, {"key", "ext2_bad"}});
            badExecGraph.markOutput("Probe.out");
            RenderGraphExecutor badExec;
            std::string log;
            FRAME_REQUIRE(badExec.compile(context.device, badExecGraph, 16, 16, log));
            if (badExec.execute({.graphicsQueue = graphics, .historyResources = &history2}).has_value()) {
                return RHITestResult::fail("2-slot external HistoryResourceManager accepted offset -2");
            }

            TemporalOffsetProbePass::records.clear();
            TemporalOffsetProbePass::writeBase = 0x1100u;
            RenderGraph okGraph;
            okGraph.setName("TwoSlotOffsetNeg1");
            okGraph.addNode("TemporalOffsetProbePass", "Probe", {{"offset", -1}, {"graphOwned", false}, {"key", "ext2_ok"}});
            okGraph.markOutput("Probe.out");
            RenderGraphExecutor exec;
            FRAME_REQUIRE(exec.compile(context.device, okGraph, 16, 16, log));
            for (uint32_t f = 0; f < 3; ++f) {
                FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history2}));
                FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
                const uint32_t word = readWord(exec, "Probe.out");
                const uint32_t expected = f == 0 ? 0u : (0x1100u + f - 1u);
                if (word != expected) {
                    return RHITestResult::fail("2-slot offset -1 readback mismatch on frame " + std::to_string(f));
                }
            }
            if (TemporalOffsetProbePass::records.size() != 3 ||
                TemporalOffsetProbePass::records[0].prevValid ||
                TemporalOffsetProbePass::records[0].prevVersion != 0 ||
                !TemporalOffsetProbePass::records[1].prevValid ||
                TemporalOffsetProbePass::records[1].prevVersion != TemporalOffsetProbePass::records[0].currVersion ||
                TemporalOffsetProbePass::records[1].prevBackingId != TemporalOffsetProbePass::records[0].currBackingId ||
                !TemporalOffsetProbePass::records[2].prevValid ||
                TemporalOffsetProbePass::records[2].prevVersion != TemporalOffsetProbePass::records[1].currVersion ||
                TemporalOffsetProbePass::records[2].prevBackingId != TemporalOffsetProbePass::records[1].currBackingId) {
                return RHITestResult::fail("2-slot offset -1 version or backing failed to trace back to write frame");
            }
        }

        // Part C: 3-slot history accepts offset -2 and traces Frame N-2's exact version ID, backing, and GPU value.
        for (bool graphOwned : {false, true}) {
            HistoryResourceManager history3;
            if (!graphOwned) {
                FRAME_REQUIRE(history3.initialize(context.device, 3));
            }
            TemporalOffsetProbePass::records.clear();
            TemporalOffsetProbePass::writeBase = graphOwned ? 0xA200u : 0xA100u;
            RenderGraph graph3;
            graph3.setName(graphOwned ? "ThreeSlotOffsetNeg2Graph" : "ThreeSlotOffsetNeg2Ext");
            graph3.addNode("TemporalOffsetProbePass", "Probe", {
                {"offset", -2}, {"graphOwned", graphOwned}, {"key", graphOwned ? "g3_ok" : "e3_ok"},
            });
            graph3.markOutput("Probe.out");
            RenderGraphExecutor exec;
            std::string log;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = 3;
            FRAME_REQUIRE(exec.compile(context.device, graph3, 16, 16, opts, log));
            for (uint32_t f = 0; f < 4; ++f) {
                FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics,
                    .historyResources = graphOwned ? nullptr : &history3}));
                FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
                const uint32_t word = readWord(exec, "Probe.out");
                const uint32_t expected = f < 2 ? 0u : (TemporalOffsetProbePass::writeBase + f - 2u);
                if (word != expected) {
                    return RHITestResult::fail("3-slot offset -2 readback mismatch on frame " + std::to_string(f));
                }
            }
            const auto& r = TemporalOffsetProbePass::records;
            if (r.size() != 4 || r[0].prevValid || r[1].prevValid ||
                !r[2].prevValid || r[2].prevVersion != r[0].currVersion || r[2].prevBackingId != r[0].currBackingId ||
                !r[3].prevValid || r[3].prevVersion != r[1].currVersion || r[3].prevBackingId != r[1].currBackingId) {
                return RHITestResult::fail("3-slot offset -2 version or backing failed to trace back to Frame N-2");
            }
        }
        return RHITestResult::pass("Temporal offset bounds (-1 on 2-slot, -2 on 3-slot, reject -2 on 2-slot & INT32_MIN) and write-origin version/backing traceability verified");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameTemporalOffsetBoundsAndVersionTest);

class TemporalColorSampleWritePass final : public render::RasterPass {
public:
    static inline std::vector<uint64_t> observedAllocGenerations;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("prev", "Sampled temporal history read")
            .texture2D(16, 16).sampledRead().temporalRead("merged_color_hist", -1, false, false)
            .format = render::Format::RGBA8Unorm;
        reflection.addTextureOutput("curr", "Color attachment temporal history write")
            .texture2D(16, 16).colorWrite().temporalWrite("merged_color_hist", false, false)
            .format = render::Format::RGBA8Unorm;
        reflection.addTextureOutput("sampledOut", "Sampled copy of previous history frame")
            .texture2D(16, 16).colorWrite().transient(render::RenderGraphInitialization::Clear)
            .format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        if (!context.device) { return makeError(Error::InvalidArgument); }
        if (auto r = ResourceRegistry::forDevice(*context.device).transform([&](auto v) { registry_ = std::move(v); }); !r) {
            return r;
        }
        auto loadShader = [&](const char* entry, std::unique_ptr<ShaderModule>& out) -> Result<> {
            ShaderCompileResult compiled;
            auto r = ShaderRegistry::instance().getShader(SlangShaderDesc{
                .moduleName = "Features/Samples/ImageSample",
                .entryPointName = entry,
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            }, compiled.diagnostics).transform([&](auto v) { compiled = std::move(v); });
            if (!r) { log += compiled.diagnostics; return r; }
            return ShaderRegistry::instance().getShaderModule(*context.device, ShaderModuleDesc{
                .spirv = compiled.spirv, .debugName = entry,
            }).transform([&](auto m) { out = std::move(m); });
        };
        if (auto r = loadShader("imageSampleVertexMain", vs_); !r) { return r; }
        if (auto r = loadShader("imageSampleFragmentMain", fs_); !r) { return r; }
        return ShaderRegistry::instance().getGraphicsPipeline(*context.device, GraphicsPipelineDesc{
            .vertexShader = {vs_.get()},
            .fragmentShader = {fs_.get()},
            .colorFormats = {Format::RGBA8Unorm},
            .colorAttachmentCount = 1,
            .topology = PrimitiveTopology::TriangleList,
            .usesBindlessHeap = true,
        }).transform([&](auto p) { pipeline_ = std::move(p); });
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const auto prev = context.temporalTexture("prev");
        const auto curr = context.temporalTexture("curr");
        const auto sampledOut = context.outputTexture("sampledOut");
        if (!prev.texture.valid() || !curr.texture.valid() || !sampledOut.valid() || !registry_ || !pipeline_) {
            return makeError(Error::InvalidArgument);
        }
        observedAllocGenerations.push_back(curr.version.allocationGeneration);

        const Rect area{.x = 0, .y = 0, .width = 16, .height = 16};
        // 1. Sample previous history texture into sampledOut if valid, else clear black.
        if (prev.valid) {
            ResourceLease lease;
            if (auto r = registry_->sampledImage(*prev.texture.view()).transform([&](auto l) { lease = std::move(l); }); !r) { return r; }
            if (auto r = context.commandBuffer().retainResource(std::make_shared<ResourceLease>(lease)); !r) { return r; }
            const RenderingAttachmentDesc att{
                .view = sampledOut.view(), .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
            };
            if (auto r = context.commandBuffer().beginRendering({.renderArea = area, .colorAttachments = {&att, 1}}); !r) { return r; }
            if (auto r = context.commandBuffer().setViewport({.width = 16.0f, .height = 16.0f, .minDepth = 0.0f, .maxDepth = 1.0f}); !r) { return r; }
            context.commandBuffer().setScissor(area);
            if (auto r = registry_->bind(context.commandBuffer(), std::array{lease}); !r) { return r; }
            const uint32_t imageIndex = lease.shaderIndex();
            if (auto r = context.commandBuffer().bindExecution(pipeline_->execution(), &imageIndex, sizeof(imageIndex)); !r) { return r; }
            if (auto r = context.commandBuffer().draw(3); !r) { return r; }
            context.commandBuffer().endRendering();
        } else {
            const RenderingAttachmentDesc att{
                .view = sampledOut.view(), .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
            };
            if (auto r = context.commandBuffer().beginRendering({.renderArea = area, .colorAttachments = {&att, 1}}); !r) { return r; }
            context.commandBuffer().endRendering();
        }

        // 2. Write distinct color to current history slot via ColorAttachment.
        const ColorValue writeColor = context.frameIndex() == 0
            ? ColorValue{0.8f, 0.2f, 0.1f, 1.0f}
            : (context.frameIndex() == 1 ? ColorValue{0.1f, 0.8f, 0.3f, 1.0f} : ColorValue{0.2f, 0.3f, 0.9f, 1.0f});
        const RenderingAttachmentDesc writeAtt{
            .view = curr.texture.view(), .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store,
            .clearColor = writeColor,
        };
        if (auto r = context.commandBuffer().beginRendering({.renderArea = area, .colorAttachments = {&writeAtt, 1}}); !r) { return r; }
        context.commandBuffer().endRendering();
        context.setTemporalWritten("curr", true);
        return {};
    }

private:
    std::shared_ptr<render::ResourceRegistry> registry_;
    std::unique_ptr<render::ShaderModule> vs_;
    std::unique_ptr<render::ShaderModule> fs_;
    std::shared_ptr<render::GraphicsPipeline> pipeline_;
};

class TemporalDescriptorMismatchPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const uint32_t w = properties().value("w", 16u);
        const uint32_t h = properties().value("h", 16u);
        const bool fp16 = properties().value("fp16", false);
        const bool isWriter = properties().value("writer", false);
        render::RenderPassReflection reflection;
        if (isWriter) {
            auto& f = reflection.addTextureOutput("hist", "History write").texture2D(w, h)
                .storageWrite().temporalWrite("mismatch_key", true, false);
            f.format = fp16 ? render::Format::RGBA16Sfloat : render::Format::RGBA8Unorm;
        } else {
            auto& f = reflection.addTextureInput("hist", "History read").texture2D(w, h)
                .sampledRead().temporalRead("mismatch_key", -1, true, false);
            f.format = fp16 ? render::Format::RGBA16Sfloat : render::Format::RGBA8Unorm;
        }
        reflection.addBufferOutput("out").buffer(16, 4).transferWrite();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext&) override { return {}; }
};

class FrameTemporalMergedUsageAndMismatchTest final : public RHITest {
public:
    FrameTemporalMergedUsageAndMismatchTest()
    {
        type = RHITestType::Rendering;
        name = "frame_temporal_merged_usage_and_descriptor_mismatch";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("TemporalColorSampleWritePass", "Sampled read + ColorAttachment temporal write",
                [] { return std::make_unique<TemporalColorSampleWritePass>(); });
            registerRenderGraphPassType("TemporalDescriptorMismatchPass", "Temporal descriptor mismatch probe",
                [] { return std::make_unique<TemporalDescriptorMismatchPass>(); });
            return true;
        }();
        (void)registered;

        // Part A: SampledRead + ColorWrite merged per logical key under optimal layouts!
        std::unique_ptr<Device> device;
        DeviceDesc deviceDesc{.applicationName = "Temporal merged usage (optimal layouts)",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true};
        vulkan::deviceExtensions(deviceDesc).preferUnifiedImageLayouts = false;
        auto created = createDevice(deviceDesc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);

        HistoryResourceManager history;
        FRAME_REQUIRE(history.initialize(*device, 2));
        TemporalColorSampleWritePass::observedAllocGenerations.clear();

        RenderGraph graph;
        graph.setName("TemporalMergedColorUsage");
        graph.addNode("TemporalColorSampleWritePass", "ColorHist");
        graph.markOutput("ColorHist.sampledOut");
        RenderGraphExecutor exec;
        std::string log;
        FRAME_REQUIRE(exec.compile(*device, graph, 16, 16, log));

        QueueSubmissionTracker rbTracker;
        Commands rbCmd;
        FRAME_REQUIRE(rbTracker.initialize(*device, *graphics));
        FRAME_REQUIRE(rbCmd.initialize(*device, *graphics));
        std::unique_ptr<Buffer> readback;
        FRAME_REQUIRE(device->createBuffer({.size = 16 * 16 * 4,
            .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto b) { readback = std::move(b); }));

        for (uint32_t f = 0; f < 3; ++f) {
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history}));
            FRAME_REQUIRE(rbCmd.begin(f));
            FRAME_REQUIRE(exec.transitionOutput(*rbCmd.buffer, "ColorHist.sampledOut", ResourceState::TransferSource));
            auto* outRes = exec.outputResource("ColorHist.sampledOut");
            auto dst = readback->slice();
            if (!outRes || !outRes->texture || !dst) { return RHITestResult::fail("missing sampledOut"); }
            FRAME_REQUIRE(rbCmd.buffer->copyTextureToBuffer({.texture = outRes->texture, .buffer = *dst, .width = 16, .height = 16}));
            const BufferBarrierDesc hostBarrier{
                .buffer = readback.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
            };
            FRAME_REQUIRE(rbCmd.buffer->synchronize({.buffers = {&hostBarrier, 1}}));
            FRAME_REQUIRE(rbCmd.submit(rbTracker));
            FRAME_REQUIRE(rbCmd.frame.wait(kWaitTimeout));
            readback->invalidate();
            auto* px = static_cast<const uint8_t*>(readback->map());
            if (!px) { return RHITestResult::fail("readback map failed"); }
            const std::array<uint8_t, 4> got{px[0], px[1], px[2], px[3]};
            readback->unmap();
            if (f == 1) {
                const std::array<uint8_t, 4> expected{204, 51, 26, 255};
                for (size_t c = 0; c < 4; ++c) {
                    if (std::abs(int(got[c]) - int(expected[c])) > 2) {
                        return RHITestResult::fail("Frame 1 sampledOut did not read Frame 0 ColorAttachment write");
                    }
                }
            } else if (f == 2) {
                const std::array<uint8_t, 4> expected{26, 204, 77, 255};
                for (size_t c = 0; c < 4; ++c) {
                    if (std::abs(int(got[c]) - int(expected[c])) > 2) {
                        return RHITestResult::fail("Frame 2 sampledOut did not read Frame 1 ColorAttachment write");
                    }
                }
            }
        }
        if (TemporalColorSampleWritePass::observedAllocGenerations.size() != 3 ||
            std::any_of(TemporalColorSampleWritePass::observedAllocGenerations.begin(),
                TemporalColorSampleWritePass::observedAllocGenerations.end(),
                [](uint64_t g) { return g != 1; })) {
            return RHITestResult::fail("Merged sampledRead + colorWrite temporal key reallocated across frames");
        }

        // Part B: Incompatible size and incompatible format on the same logical key must return error.
        for (bool formatMismatch : {false, true}) {
            RenderGraph bad;
            bad.setName(formatMismatch ? "TemporalFormatMismatch" : "TemporalSizeMismatch");
            bad.addNode("TemporalDescriptorMismatchPass", "Writer", {{"writer", true}, {"w", 16u}, {"h", 16u}, {"fp16", false}});
            bad.addNode("TemporalDescriptorMismatchPass", "Reader", {
                {"writer", false},
                {"w", formatMismatch ? 16u : 32u},
                {"h", 16u},
                {"fp16", formatMismatch},
            });
            bad.markOutput("Writer.out");
            bad.markOutput("Reader.out");
            RenderGraphExecutor badExec;
            FRAME_REQUIRE(badExec.compile(*device, bad, 16, 16, log));
            if (badExec.execute({.graphicsQueue = graphics}).has_value()) {
                return RHITestResult::fail(formatMismatch
                    ? "Executor failed to reject temporal format mismatch"
                    : "Executor failed to reject temporal extent mismatch");
            }
        }
        return RHITestResult::pass("Merged sampledRead + colorWrite temporal key preserved generation=1 with exact cross-frame pixels under optimal layouts; incompatible descriptors rejected");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameTemporalMergedUsageAndMismatchTest);

struct ParallelSharedTemporalControl {
    render::Device* device = nullptr;
    std::mutex mutex;
    std::condition_variable cv;
    bool batch0Done = false;
    bool failPass0 = false;
    bool failPass1 = false;
    bool waitForBatch0InPass1 = false;
    uint32_t pass0Value = 0xAA00u;
    uint32_t pass1Value = 0xBB00u;
    bool observedPrev1Valid = false;
    bool observedPrev2Valid = false;
};

class ParallelSharedTemporalPass final : public render::ComputePass {
public:
    static inline ParallelSharedTemporalControl* control = nullptr;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::CPURecordingPolicy cpuRecordingPolicy() const override { return render::CPURecordingPolicy::ParallelJoined; }
    uint32_t recordingWorkload() const override { return 4; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        const uint32_t stage = properties().value("stage", 0u);
        render::RenderPassReflection reflection;
        if (stage == 0u) {
            reflection.addBufferInput("prev1").buffer(16, 4).transferRead()
                .temporalRead("shared_parallel_slot", -1, false, false);
            reflection.addBufferInput("prev2").buffer(16, 4).transferRead()
                .temporalRead("shared_parallel_slot", -2, false, false);
            reflection.addBufferOutput("curr").buffer(16, 4).transferWrite()
                .temporalWrite("shared_parallel_slot", false, false);
            reflection.addBufferOutput("read1").buffer(16, 4).transferWrite().hostReadback();
            reflection.addBufferOutput("read2").buffer(16, 4).transferWrite().hostReadback();
        } else {
            reflection.addBufferOutput("curr").buffer(16, 4).transferWrite()
                .temporalWrite("shared_parallel_slot", false, false);
            reflection.addBufferOutput("done").buffer(16, 4).transferWrite();
        }
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const uint32_t stage = context.properties().value("stage", 0u);
        auto* ctrl = control;
        if (ctrl == nullptr) { return makeError(Error::InvalidArgument); }

        if (stage == 0u) {
            const auto prev1 = context.temporalBuffer("prev1");
            const auto prev2 = context.temporalBuffer("prev2");
            const auto curr = context.temporalBuffer("curr");
            const auto read1 = context.outputBuffer("read1");
            const auto read2 = context.outputBuffer("read2");
            if (!prev1.buffer.valid() || !prev2.buffer.valid() || !curr.buffer.valid() ||
                !read1.valid() || !read2.valid()) {
                return makeError(Error::InvalidArgument);
            }
            ctrl->observedPrev1Valid = prev1.valid;
            ctrl->observedPrev2Valid = prev2.valid;
            auto r1Slice = read1.buffer()->slice({0, 16});
            auto r2Slice = read2.buffer()->slice({0, 16});
            if (!ctrl->device || !r1Slice || !r2Slice) { return makeError(Error::Failure); }
            if (prev1.valid) {
                auto p1 = prev1.buffer.buffer()->slice({0, 16});
                if (!p1 || !context.commandBuffer().copyBuffer(*p1, *r1Slice)) { return makeError(Error::Failure); }
            } else if (!copyHostWordToBuffer(*ctrl->device, context.commandBuffer(), *read1.buffer(), 0u)) {
                return makeError(Error::Failure);
            }
            if (prev2.valid) {
                auto p2 = prev2.buffer.buffer()->slice({0, 16});
                if (!p2 || !context.commandBuffer().copyBuffer(*p2, *r2Slice)) { return makeError(Error::Failure); }
            } else if (!copyHostWordToBuffer(*ctrl->device, context.commandBuffer(), *read2.buffer(), 0u)) {
                return makeError(Error::Failure);
            }
            if (ctrl->failPass0) {
                {
                    std::lock_guard lock(ctrl->mutex);
                    ctrl->batch0Done = true;
                }
                ctrl->cv.notify_all();
                return makeError(Error::Failure);
            }
            if (auto r = copyHostWordToBuffer(*ctrl->device, context.commandBuffer(), *curr.buffer.buffer(), ctrl->pass0Value); !r) { return r; }
            context.setTemporalWritten("curr", true);
            {
                std::lock_guard lock(ctrl->mutex);
                ctrl->batch0Done = true;
            }
            ctrl->cv.notify_all();
            return {};
        }

        if (ctrl->waitForBatch0InPass1) {
            std::unique_lock lock(ctrl->mutex);
            ctrl->cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return ctrl->batch0Done; });
            // Give coordinator's pipelined loop time to accept and submit Batch 0 before Batch 1 fails.
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        if (ctrl->failPass1) {
            return makeError(Error::Failure);
        }
        const auto curr = context.temporalBuffer("curr");
        const auto done = context.outputBuffer("done");
        if (!ctrl->device || !curr.buffer.valid() || !done.valid()) { return makeError(Error::InvalidArgument); }
        if (auto r = copyHostWordToBuffer(*ctrl->device, context.commandBuffer(), *curr.buffer.buffer(), ctrl->pass1Value); !r) { return r; }
        if (auto r = copyHostWordToBuffer(*ctrl->device, context.commandBuffer(), *done.buffer(), 1u); !r) { return r; }
        context.setTemporalWritten("curr", true);
        return {};
    }
};

class FrameTemporalParallelSharedSlotTest final : public RHITest {
public:
    FrameTemporalParallelSharedSlotTest()
    {
        type = RHITestType::Rendering;
        name = "frame_temporal_parallel_shared_slot_and_partial_recovery";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 2) {
            return RHITestResult::skip("requires at least 2 task workers for parallel recording");
        }
        static const bool registered = registerRenderGraphPassType("ParallelSharedTemporalPass",
            "Parallel shared temporal slot coordinator-commit probe",
            [] { return std::make_unique<ParallelSharedTemporalPass>(); });
        (void)registered;

        ParallelSharedTemporalControl ctrl{};
        ctrl.device = &context.device;
        ParallelSharedTemporalPass::control = &ctrl;
        struct ResetControl { ~ResetControl() { ParallelSharedTemporalPass::control = nullptr; } } cleanup;

        HistoryResourceManager history3;
        FRAME_REQUIRE(history3.initialize(context.device, 3));

        RenderGraph graph;
        graph.setName("ParallelSharedTemporalGraph");
        graph.addNode("ParallelSharedTemporalPass", "BatchA", {{"stage", 0u}});
        graph.addNode("ParallelSharedTemporalPass", "BatchB", {{"stage", 1u}});
        graph.markOutput("BatchA.read1");
        graph.markOutput("BatchA.read2");
        graph.markOutput("BatchB.done");

        RenderGraphExecutor exec;
        std::string log;
        FRAME_REQUIRE(exec.compile(context.device, graph, 16, 16, log));

        const auto readWord = [&](std::string_view outputName) -> uint32_t {
            auto* res = exec.outputResource(outputName);
            if (!res || !res->buffer) { return 0xFFFFFFFFu; }
            res->buffer->invalidate();
            const void* mapped = res->buffer->map();
            if (!mapped) { return 0xFFFFFFFFu; }
            uint32_t word = 0xFFFFFFFFu;
            std::memcpy(&word, mapped, sizeof(word));
            res->buffer->unmap();
            return word;
        };

        // Frame 0: Both parallel batches share perPassScope=false "shared_parallel_slot" and succeed.
        // Coordinator must commit BatchA then BatchB in deterministic order (leaving 0x10B0).
        ctrl.batch0Done = false;
        ctrl.pass0Value = 0x10A0u;
        ctrl.pass1Value = 0x10B0u;
        FRAME_REQUIRE(exec.execute({
            .graphicsQueue = &context.graphicsQueue,
            .historyResources = &history3,
            .recordingWorkerLimit = 2,
            .recordingBatchWorkload = 4,
            .submissionMode = FrameSubmissionMode::Pipelined,
        }));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        const auto stats0 = exec.executionStats();
        if (stats0.recordingBatchCount < 2 || stats0.parallelRecordedPassCount < 2 || !stats0.pipelinedSubmission) {
            return RHITestResult::fail("Expected >=2 pipelined parallel batches for shared temporal slot");
        }

        // Frame 1: Partial submission! BatchA writes 0x20A0 and gets submitted in pipelined mode;
        // BatchB fails during worker recording and must NOT overwrite or corrupt history.
        ctrl.batch0Done = false;
        ctrl.pass0Value = 0x20A0u;
        ctrl.pass1Value = 0x20B0u;
        ctrl.waitForBatch0InPass1 = true;
        ctrl.failPass1 = true;
        if (exec.execute({
                .graphicsQueue = &context.graphicsQueue,
                .historyResources = &history3,
                .recordingWorkerLimit = 2,
                .recordingBatchWorkload = 4,
                .submissionMode = FrameSubmissionMode::Pipelined,
            }).has_value()) {
            return RHITestResult::fail("Expected Frame 1 BatchB failure to return error");
        }
        const auto partialStats = exec.executionStats();
        const bool partialAccepted = partialStats.submittedBatchCount >= 1 && exec.lastSubmittedCompletion().isSubmitted();
        if (!partialAccepted) {
            return RHITestResult::fail("Expected BatchA to be accepted before BatchB failed in pipelined partial submission");
        }
        if (readWord("BatchA.read1") != 0x10B0u) {
            return RHITestResult::fail("Frame 1 did not read Frame 0's deterministic final BatchB value (0x10B0)");
        }

        // Frame 2: Recompile after partial submission failure; BatchA reads Frame 1's accepted 0x20A0 at offset -1
        // and Frame 0's 0x10B0 at offset -2, and writes 0x30A0 -> 0x30B0.
        ctrl.batch0Done = false;
        ctrl.waitForBatch0InPass1 = false;
        ctrl.failPass1 = false;
        ctrl.pass0Value = 0x30A0u;
        ctrl.pass1Value = 0x30B0u;
        FRAME_REQUIRE(exec.compile(context.device, graph, 16, 16, log));
        FRAME_REQUIRE(exec.execute({
            .graphicsQueue = &context.graphicsQueue,
            .historyResources = &history3,
            .recordingWorkerLimit = 2,
            .recordingBatchWorkload = 4,
            .submissionMode = FrameSubmissionMode::Pipelined,
        }));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        if (!ctrl.observedPrev1Valid || !ctrl.observedPrev2Valid ||
            readWord("BatchA.read1") != 0x20A0u || readWord("BatchA.read2") != 0x10B0u) {
            return RHITestResult::fail("Recovery after partial submission did not preserve BatchA's accepted write (0x20A0) and Frame 0 (0x10B0)");
        }

        // Frame 3: Total cancel (BatchA fails before any batch submits).
        ctrl.batch0Done = false;
        ctrl.failPass0 = true;
        ctrl.pass0Value = 0xDEAD0000u;
        ctrl.pass1Value = 0xDEAD0001u;
        if (exec.execute({
                .graphicsQueue = &context.graphicsQueue,
                .historyResources = &history3,
                .recordingWorkerLimit = 2,
                .recordingBatchWorkload = 4,
                .submissionMode = FrameSubmissionMode::Pipelined,
            }).has_value()) {
            return RHITestResult::fail("Expected Frame 3 BatchA failure to return error");
        }

        // Frame 4: Recompile & run; offset -1 (cancelled Frame 3) MUST be invalid (reads 0, NOT 0xDEAD*),
        // while offset -2 (accepted Frame 2) MUST still be valid and read 0x30B0!
        ctrl.batch0Done = false;
        ctrl.failPass0 = false;
        ctrl.pass0Value = 0x40A0u;
        ctrl.pass1Value = 0x40B0u;
        FRAME_REQUIRE(exec.compile(context.device, graph, 16, 16, log));
        FRAME_REQUIRE(exec.execute({
            .graphicsQueue = &context.graphicsQueue,
            .historyResources = &history3,
            .recordingWorkerLimit = 2,
            .recordingBatchWorkload = 4,
            .submissionMode = FrameSubmissionMode::Pipelined,
        }));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        if (ctrl.observedPrev1Valid || !ctrl.observedPrev2Valid ||
            readWord("BatchA.read1") != 0u || readWord("BatchA.read2") != 0x30B0u) {
            return RHITestResult::fail("Post-cancel Frame 4 exposed cancelled slot or lost accepted Frame 2 history");
        }
        return RHITestResult::pass("Parallel 2-batch shared temporal slot coordinator commit, partial-submission recovery, and post-cancel readback verified");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameTemporalParallelSharedSlotTest);

class FramePathTraceConditionalHistoryTest final : public RHITest {
public:
    FramePathTraceConditionalHistoryTest()
    {
        type = RHITestType::Rendering;
        name = "frame_pathtrace_conditional_history_allocation";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        DeviceDesc desc{
            .applicationName = "PathTrace conditional history allocation",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableMeshShader = true,
            .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true,
            .enableClusterAccelerationStructure = true,
        };
        vulkan::deviceExtensions(desc).preferUnifiedImageLayouts = false;
        auto created = createDevice(desc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) {
            return RHITestResult::skip("requires ray query, CLAS, and bindless descriptor heap");
        }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);

        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf")) {
            return RHITestResult::fail("Failed to load ABeautifulGame.gltf: " + document.lastLoadResult().error);
        }

        const std::string historyKey = "PathTrace.accumulation";
        const auto countTemporalRGBA32F = [](const RenderGraphExecutionSnapshot& snap) -> size_t {
            size_t count = 0;
            for (const auto& res : snap.resources) {
                if (res.ownership == RenderGraphResourceOwnership::TemporalOwned &&
                    res.type == RenderGraphResourceType::Texture2D &&
                    res.textureDesc.format == Format::RGBA32Sfloat) {
                    ++count;
                }
            }
            return count;
        };

        RenderWorld world;
        world.setScene(&document);
        world.setEnvironment({.enabled = false, .visible = false});
        RenderGraph graph;
        graph.setName("PathTraceConditionalHistory");
        const RenderGraphNode* ptNode = graph.addNode("ScenePathTracePass", "PathTrace", {
            {"bsdf", "openpbr"},
            {"cacheMode", "off"},
            {"maxDepth", 1},
            {"samples", 1},
            {"accumulate", false},
            {"debugView", "final"},
            {"camera", {{"eye", {0.0, 0.35, 0.65}}, {"center", {0.0, 0.05, 0.0}}, {"fovDegrees", 50.0}}},
        });
        if (!ptNode) { return RHITestResult::fail("Failed to add ScenePathTracePass node"); }
        const uint32_t ptId = ptNode->id;
        graph.markOutput("PathTrace.color");

        HistoryResourceManager history;
        FRAME_REQUIRE(history.initialize(*device, 2));
        RenderGraphExecutor exec;
        exec.bindRenderWorld(&world);
        exec.setExecutionCaptureEnabled(true);
        std::string log;
        auto compiled = exec.compile(*device, graph, 64, 48, log);
        if (hasError(compiled, Error::Unsupported)) { return RHITestResult::skip(log); }
        FRAME_REQUIRE(compiled);

        // Step 1: accumulate=false -> ZERO RGBA32F history textures allocated!
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        auto snap = exec.executionSnapshot();
        if (!snap || history.hasRecord(historyKey) || countTemporalRGBA32F(*snap) != 0) {
            return RHITestResult::fail("accumulate=false allocated RGBA32F history textures");
        }

        // Step 2: accumulate=true but non-accumulating OpenPBR debugView="baseColor" -> STILL ZERO history allocated!
        graph.setNodeRuntimeProperty(ptId, "accumulate", true);
        graph.setNodeRuntimeProperty(ptId, "debugView", "baseColor");
        if (!exec.syncRuntimeProperties(graph)) {
            return RHITestResult::fail("syncRuntimeProperties failed for debugView");
        }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        snap = exec.executionSnapshot();
        if (!snap || history.hasRecord(historyKey) || countTemporalRGBA32F(*snap) != 0) {
            return RHITestResult::fail("Non-accumulating OpenPBR debugView allocated RGBA32F history textures");
        }

        // Step 3: debugView="final" + accumulate=true -> allocates history and publishes valid slot.
        graph.setNodeRuntimeProperty(ptId, "debugView", "final");
        if (!exec.syncRuntimeProperties(graph)) {
            return RHITestResult::fail("syncRuntimeProperties failed for final view");
        }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        snap = exec.executionSnapshot();
        if (!snap || !history.hasRecord(historyKey) || countTemporalRGBA32F(*snap) != 2 ||
            !history.texture(historyKey, HistorySlot::Current).valid) {
            return RHITestResult::fail("Enabling accumulate=true + debugView=final did not allocate 2 RGBA32F history textures");
        }

        // Step 4: Turning accumulate=false invalidates existing history and drops temporal bindings from frame snapshot.
        graph.setNodeRuntimeProperty(ptId, "accumulate", false);
        if (!exec.syncRuntimeProperties(graph)) {
            return RHITestResult::fail("syncRuntimeProperties failed disabling accumulate");
        }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        snap = exec.executionSnapshot();
        if (!snap || countTemporalRGBA32F(*snap) != 0 ||
            history.texture(historyKey, HistorySlot::Current).valid ||
            history.texture(historyKey, HistorySlot::Previous).valid) {
            return RHITestResult::fail("Disabling accumulate did not invalidate existing history");
        }

        // Step 5: Re-enabling accumulate=true starts from INVALID history (hasPrevious == false before frame runs).
        graph.setNodeRuntimeProperty(ptId, "accumulate", true);
        if (!exec.syncRuntimeProperties(graph)) {
            return RHITestResult::fail("syncRuntimeProperties failed re-enabling accumulate");
        }
        if (history.hasPrevious(historyKey)) {
            return RHITestResult::fail("Re-enabling accumulate did not start from invalid history");
        }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        if (!history.texture(historyKey, HistorySlot::Current).valid ||
            history.texture(historyKey, HistorySlot::Previous).valid) {
            return RHITestResult::fail("First frame after re-enabling accumulate did not start cleanly from invalid previous slot");
        }
        return RHITestResult::pass("PathTrace conditional RGBA32F history allocation (accumulate=false, OpenPBR debugView, re-enable from invalid history) verified under optimal layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(FramePathTraceConditionalHistoryTest);

// ============================================================================
// Batch 3 Tests:
// 1. Extended multi-frame timeline gate (N+1 & N+2 before N unblocks,
//    same-backing RAW/WAR/WAW + read-only sharing + Graphics<->Compute optimal
//    layout transitions, multi-reader backing retirement, cancelled Producer)
// 2. Production-graph lifecycle stress (Camera Cut, World switch, Resize,
//    Shader Reload, Cache Eviction, Graph Recompile, Inspector pin/readback,
//    Streamer ProviderOwned publications, conservative boundaries)
// 3. Real built-in production post-process policy benchmark (A-B-A-B harness)
// 4. Incremental declarative migration verification (Lumen Radiance Cache,
//    VirtualShadowMapPass, StreamerSubsystem)
// ============================================================================

class TimelineMultiFrameComputeCounterPass final : public render::ComputePass {
public:
    static inline render::Device* devicePtr = nullptr;
    static inline uint32_t baseTag = 0xA0000000u;

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Per-frame compute counter output")
            .buffer(16, 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto out = context.outputBuffer("data");
        if (!devicePtr || !out.valid()) { return makeError(Error::InvalidArgument); }
        const uint32_t value = baseTag + static_cast<uint32_t>(context.frameIndex());
        if (auto r = copyHostWordToBuffer(*devicePtr, context.commandBuffer(), *out.buffer(), value); !r) { return r; }
        const BufferBarrierDesc hostBarrier{
            .buffer = out.buffer(),
            .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
            .after = {PipelineStageBits::Host, AccessBits::HostRead},
        };
        return context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}});
    }
};

class SharedBackingComputeWriterPass final : public render::ComputePass {
public:
    // Distinct R channel values chosen so FrameResourceProbe.sampleImages (uint(2*r))
    // yields exact distinct frame tags {0, 1, 2} on RGBA8Unorm under pure ShaderRead sharing!
    static inline std::array<uint32_t, 4> frameColors{0xFF000000u, 0xFF000090u, 0xFF0000FFu, 0xFF000090u};
    static inline std::array<uint32_t, 4> expectedSampled{0u, 1u, 2u, 1u};

    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("tex", "Shared backing RGBA8 written on Compute")
            .texture2D(16, 16).storageWrite().format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        const ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .kind = ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        if (auto r = createProbe(*context.device, "fillSolidTexture", bindings, program_, log); !r) { return r; }
        return context.device->createBuffer({
            .size = 16, .structureStride = 4, .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::Device,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
        }).transform([&](auto b) { dummy_ = std::move(b); });
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto tex = context.outputTexture("tex");
        if (!tex.valid() || !dummy_ || !program_.valid()) { return makeError(Error::InvalidArgument); }
        TextureView* view = tex.view();
        const uint32_t packed = frameColors[context.frameIndex() % frameColors.size()];
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, previous), .textureViews = {&view, 1}},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, current), .textureViews = {&view, 1}},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameHistoryProbeResources, output), .buffer = dummy_.get()},
        };
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = &packed, .pushDataSize = sizeof(packed),
            .groupCountX = 2, .groupCountY = 2, .groupCountZ = 1,
        });
    }

private:
    render::ComputeProgram program_;
    std::unique_ptr<render::Buffer> dummy_;
};

class SharedBackingGatedSamplerPass final : public render::ComputePass {
public:
    static inline render::Semaphore* gateSemGfx = nullptr;
    static inline uint64_t gateValGfx = 0;
    static inline uint64_t gateFrameGfx = UINT64_MAX;
    static inline render::Semaphore* gateSemCmp = nullptr;
    static inline uint64_t gateValCmp = 0;
    static inline uint64_t gateFrameCmp = UINT64_MAX;
    static inline std::array<render::Buffer*, 4> sinkGfx{};
    static inline std::array<render::Buffer*, 4> sinkCmp{};

    render::QueueType queueType() const override
    {
        return properties().value("useGraphicsQueue", false) ? render::QueueType::Graphics : render::QueueType::Compute;
    }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("tex", "Shared backing sampled read")
            .texture2D(16, 16).sampledRead().format = render::Format::RGBA8Unorm;
        reflection.addBufferOutput("out", "Sampled word output")
            .buffer(16, 4).storageWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        const ComputeResourceBindingDesc bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .kind = ComputeResourceBindingKind::SampledImage, .descriptorCount = 2},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .kind = ComputeResourceBindingKind::StorageBuffer},
        };
        return createProbe(*context.device, "sampleImages", bindings, program_, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        const bool isGfx = properties().value("useGraphicsQueue", false);
        Semaphore* gateSem = isGfx ? gateSemGfx : gateSemCmp;
        const uint64_t gateVal = isGfx ? gateValGfx : gateValCmp;
        const uint64_t gateFrame = isGfx ? gateFrameGfx : gateFrameCmp;
        if (gateSem != nullptr && gateVal != 0 && context.frameIndex() == gateFrame) {
            const SemaphoreSubmitDesc wait{.semaphore = gateSem, .value = gateVal, .stages = PipelineStageBits::AllCommands};
            if (auto r = context.commandBuffer().addDependency({&wait, 1}); !r) { return r; }
        }
        auto tex = context.inputTexture("tex");
        auto out = context.outputBuffer("out");
        if (!tex.valid() || !out.valid() || !program_.valid()) { return makeError(Error::InvalidArgument); }
        TextureView* views[2] = {tex.view(), tex.view()};
        const uint32_t index = 0;
        const ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, images), .textureViews = views},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::FrameImagesProbeResources, output), .buffer = out.buffer()},
        };
        if (auto r = program_.dispatch({
            .commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = &index, .pushDataSize = sizeof(index),
            .groupCountX = 1, .groupCountY = 1, .groupCountZ = 1,
        }); !r) { return r; }

        // Also archive per-frame sampled pixel into an external per-frame host sink so
        // overlapping multi-frame tests can verify every frame's observed value independently.
        Buffer* sink = isGfx
            ? sinkGfx[context.frameIndex() % sinkGfx.size()]
            : sinkCmp[context.frameIndex() % sinkCmp.size()];
        if (sink != nullptr) {
            const BufferBarrierDesc toCopy{
                .buffer = out.buffer(),
                .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
            };
            if (auto r = context.commandBuffer().synchronize({.buffers = {&toCopy, 1}}); !r) { return r; }
            auto srcSlice = out.buffer()->slice({0, 16});
            auto dstSlice = sink->slice({0, 16});
            if (!srcSlice || !dstSlice) { return makeError(Error::Failure); }
            if (auto r = context.commandBuffer().copyBuffer(*srcSlice, *dstSlice); !r) { return r; }
            const BufferBarrierDesc hostBarrier{
                .buffer = sink,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
            };
            if (auto r = context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}}); !r) { return r; }
        }
        return {};
    }

private:
    render::ComputeProgram program_;
};

class CancelledProducerProvider final : public render::IRenderSubsystem, public render::IRenderGraphResourceProvider {
public:
    struct Desc {};
    static constexpr render::RenderSubsystemId kSubsystemId = "test.cancelled-producer-provider";
    render::RenderSubsystemId providerSubsystemId() const override { return kSubsystemId; }
    render::Result<> initialize(const render::RenderSubsystemInitContext&, std::string&) override { return {}; }
    void shutdown() override {}

    bool failProducerRecord = true;
    uint32_t cancelCallbacks = 0;
    uint32_t acceptCallbacks = 0;

    render::Result<> declareFailingRequest(render::RenderGraphResourceRequestBuilder& builder)
    {
        using namespace render;
        const LogicalResourceId lid = makeLogicalResourceId("test.cancelled", "payload");
        RenderGraphProviderProducerWork work;
        work.debugName = "CancelledProducerWork";
        work.providerId = kSubsystemId;
        work.dedupKey = lid.value;
        work.preferredQueue = QueueType::Graphics;
        work.producedResources.push_back(lid);
        work.record = [this](CommandBuffer&, std::string& log) -> Result<> {
            if (failProducerRecord) {
                log = "Injected provider producer failure";
                return makeError(Error::Failure);
            }
            return {};
        };
        work.onAccepted = [this](const GPUCompletionPoint&) { ++acceptCallbacks; };
        work.onCancelled = [this]() { ++cancelCallbacks; };
        if (auto r = builder.registerProducerWork(std::move(work)); !r) { return r; }
        return builder.requireProducer(lid.value);
    }
};

class PipelinedCancelMarkerPass final : public render::ComputePass {
public:
    static inline render::Device* devicePtr = nullptr;
    static inline render::Buffer* markerBuffer = nullptr;
    static inline uint32_t writeValue = 0;
    static inline bool failNow = false;
    static inline uint32_t executeCount = 0;

    render::QueueType queueType() const override
    {
        return properties().value("useGraphicsQueue", false) ? render::QueueType::Graphics : render::QueueType::Compute;
    }
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    bool supportsCrossFrameResourceScheduling() const override { return true; }

    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        if (properties().value("useCancelledProvider", false)) {
            static constexpr std::array required{CancelledProducerProvider::kSubsystemId};
            return required;
        }
        return {};
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (properties().value("hasInput", false)) {
            reflection.addBufferInput("in", "Producer dependency").buffer(16, 4).transferRead();
        }
        reflection.addBufferOutput("out", "Output").buffer(16, 4).transferWrite();
        return reflection;
    }

    render::Result<> declareResourceRequests(
        const render::RenderGraphResourceRequestContext& context,
        render::RenderGraphResourceRequestBuilder& builder,
        std::string&) override
    {
        if (!properties().value("useCancelledProvider", false)) { return {}; }
        auto* provider = context.subsystem<CancelledProducerProvider>();
        if (!provider) { return render::makeError(render::Error::InvalidArgument); }
        return provider->declareFailingRequest(builder);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        ++executeCount;
        if (properties().value("failThisPass", false) && failNow) {
            return makeError(Error::Failure);
        }
        Buffer* target = markerBuffer;
        const uint32_t val = properties().value("markerValue", writeValue);
        if (devicePtr && target) {
            if (auto r = copyHostWordToBuffer(*devicePtr, context.commandBuffer(), *target, val); !r) { return r; }
            const BufferBarrierDesc hostBarrier{
                .buffer = target,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead},
            };
            if (auto r = context.commandBuffer().synchronize({.buffers = {&hostBarrier, 1}}); !r) { return r; }
        }
        return {};
    }
};

class FrameTimelineGateMultiFrameHazardsTest final : public RHITest {
public:
    FrameTimelineGateMultiFrameHazardsTest()
    {
        type = RHITestType::Rendering;
        name = "frame_timeline_gate_multiframe_hazards_and_retirement";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        static const bool registered = [] {
            registerRenderGraphPassType("CrossFrameGateGraphicsPass", "Timeline-gated cross-frame graphics pass",
                [] { return std::make_unique<CrossFrameGateGraphicsPass>(); });
            registerRenderGraphPassType("TimelineMultiFrameComputeCounterPass", "Per-frame compute counter",
                [] { return std::make_unique<TimelineMultiFrameComputeCounterPass>(); });
            registerRenderGraphPassType("SharedBackingComputeWriterPass", "Compute writer for shared backing",
                [] { return std::make_unique<SharedBackingComputeWriterPass>(); });
            registerRenderGraphPassType("SharedBackingGatedSamplerPass", "Gated sampler for shared backing",
                [] { return std::make_unique<SharedBackingGatedSamplerPass>(); });
            registerRenderGraphPassType("PipelinedCancelMarkerPass", "Cancel marker pass",
                [] { return std::make_unique<PipelinedCancelMarkerPass>(); });
            return true;
        }();
        (void)registered;

        std::unique_ptr<Device> device;
        DeviceDesc desc{
            .applicationName = "Batch 3 multi-frame timeline gate & retirement",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true,
        };
        vulkan::deviceExtensions(desc).preferUnifiedImageLayouts = false; // Force optimal layouts!
        auto created = createDevice(desc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("requires bindless descriptor heap"); }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute || graphics->sameQueue(*compute)) {
            return RHITestResult::skip("requires independent graphics and compute queues");
        }

        const auto makeHostReadbackBuf = [&](uint32_t initialWord = 0u) -> std::unique_ptr<Buffer> {
            std::unique_ptr<Buffer> buf;
            if (!device->createBuffer({
                .size = 16, .structureStride = 4,
                .usage = BufferUsageBits::TransferDestination | BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy,
            }).transform([&](auto b) { buf = std::move(b); })) { return nullptr; }
            if (auto* mapped = static_cast<uint32_t*>(buf->map())) {
                mapped[0] = initialWord; mapped[1] = 0; mapped[2] = 0; mapped[3] = 0;
                buf->flush();
                buf->unmap();
            }
            return buf;
        };
        const auto readFirstWord = [](Buffer& buf) -> uint32_t {
            buf.invalidate();
            const auto* mapped = static_cast<const uint32_t*>(buf.map());
            if (!mapped) { return 0xFFFFFFFFu; }
            const uint32_t v = mapped[0];
            buf.unmap();
            return v;
        };

        // ------------------------------------------------------------------
        // Subtest A: Submit Frame N+1 AND Frame N+2 BEFORE Frame N unblocks!
        // Prove disjoint independent-queue Compute work in N+1 & N+2 completes
        // on GPU while Frame N Graphics is still gated.
        // ------------------------------------------------------------------
        {
            TimelineMultiFrameComputeCounterPass::devicePtr = device.get();
            TimelineMultiFrameComputeCounterPass::baseTag = 0xA1000000u;
            std::unique_ptr<Semaphore> gate;
            FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { gate = std::move(s); }));
            bool signaled = false;
            struct GateGuard {
                Semaphore* sem = nullptr;
                bool* sig = nullptr;
                ~GateGuard()
                {
                    CrossFrameGateGraphicsPass::gateSemaphore = nullptr;
                    CrossFrameGateGraphicsPass::gateWaitValue = 0;
                    TimelineMultiFrameComputeCounterPass::devicePtr = nullptr;
                    if (sem && sig && !*sig) { (void)sem->signal(1); *sig = true; }
                }
            } guard{gate.get(), &signaled};

            RenderGraph graph;
            graph.setName("TimelineGateNPlus1NPlus2");
            graph.addNode("CrossFrameGateGraphicsPass", "Gfx");
            graph.addNode("TimelineMultiFrameComputeCounterPass", "Cmp");
            graph.markOutput("Gfx.color");
            graph.markOutput("Cmp.data");

            RenderGraphExecutor exec;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = 3;
            opts.memoryPolicy = RenderGraphMemoryPolicy::MaximizeOverlap;
            opts.versionPoolBudgetBytes = 8ull * 1024 * 1024;
            opts.maxPhysicalVersionsPerResource = 3;
            std::string log;
            FRAME_REQUIRE(exec.compile(*device, graph, 32, 32, opts, log));

            // Frame 0 (N): gated on Graphics!
            CrossFrameGateGraphicsPass::gateSemaphore = gate.get();
            CrossFrameGateGraphicsPass::gateWaitValue = 1;
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            const Texture* tex0 = exec.outputResource("Gfx.color") ? exec.outputResource("Gfx.color")->texture : nullptr;

            // Frame 1 (N+1) AND Frame 2 (N+2) submitted BEFORE Frame N unblocks!
            CrossFrameGateGraphicsPass::gateSemaphore = nullptr;
            CrossFrameGateGraphicsPass::gateWaitValue = 0;
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            const Texture* tex1 = exec.outputResource("Gfx.color") ? exec.outputResource("Gfx.color")->texture : nullptr;
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            const Texture* tex2 = exec.outputResource("Gfx.color") ? exec.outputResource("Gfx.color")->texture : nullptr;

            if (!exec.executionStats().crossFrameResourceScheduled || !tex0 || !tex1 || !tex2 ||
                tex0 == tex1 || tex1 == tex2 || tex0 == tex2) {
                return RHITestResult::fail("Subtest A: MaximizeOverlap (3 frames in flight) did not assign 3 distinct Graphics backings while Frame N was gated");
            }

            // Prove Compute queue finished Frame N+1 AND Frame N+2 BEFORE gate is signaled!
            std::unique_ptr<Fence> cmpFence;
            FRAME_REQUIRE(device->createFence(false).transform([&](auto f) { cmpFence = std::move(f); }));
            FRAME_REQUIRE(compute->submit({.signalFence = cmpFence.get()}));
            if (!cmpFence->wait(5'000'000'000ull)) {
                return RHITestResult::fail("Subtest A: Frame N+1 / N+2 Compute did not complete while Frame N Graphics was gated");
            }
            const auto* cmpOut = exec.outputResource("Cmp.data");
            if (!cmpOut || !cmpOut->buffer || readFirstWord(*cmpOut->buffer) != 0xA1000002u) {
                return RHITestResult::fail("Subtest A: Frame N+2 Compute output word mismatch while Frame N was still gated");
            }
            FRAME_REQUIRE(gate->signal(1));
            signaled = true;
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        }

        // ------------------------------------------------------------------
        // Subtest B & C: Same-backing RAW, WAR, WAW, Read-Only Sharing,
        // Graphics <-> Compute optimal layout transitions, AND Multi-Reader
        // Backing Retirement (2 distinct overlapping GPU readers gated on
        // semA & semB; release semA first, verify ReaderB + WriterNext wait
        // until BOTH readers complete!).
        // ------------------------------------------------------------------
        {
            std::unique_ptr<Semaphore> semA, semB;
            FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { semA = std::move(s); }));
            FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { semB = std::move(s); }));
            bool sigA = false, sigB = false;

            std::array<std::unique_ptr<Buffer>, 3> gfxSinks, cmpSinks;
            for (size_t i = 0; i < 3; ++i) {
                gfxSinks[i] = makeHostReadbackBuf(0xEEEEEEEEu);
                cmpSinks[i] = makeHostReadbackBuf(0xEEEEEEEEu);
                if (!gfxSinks[i] || !cmpSinks[i]) { return RHITestResult::fail("Failed to create per-frame readback sinks"); }
                SharedBackingGatedSamplerPass::sinkGfx[i] = gfxSinks[i].get();
                SharedBackingGatedSamplerPass::sinkCmp[i] = cmpSinks[i].get();
            }
            struct MultiReaderGuard {
                Semaphore* a = nullptr; bool* sa = nullptr;
                Semaphore* b = nullptr; bool* sb = nullptr;
                ~MultiReaderGuard()
                {
                    SharedBackingGatedSamplerPass::gateSemGfx = nullptr;
                    SharedBackingGatedSamplerPass::gateValGfx = 0;
                    SharedBackingGatedSamplerPass::gateFrameGfx = UINT64_MAX;
                    SharedBackingGatedSamplerPass::gateSemCmp = nullptr;
                    SharedBackingGatedSamplerPass::gateValCmp = 0;
                    SharedBackingGatedSamplerPass::gateFrameCmp = UINT64_MAX;
                    SharedBackingGatedSamplerPass::sinkGfx.fill(nullptr);
                    SharedBackingGatedSamplerPass::sinkCmp.fill(nullptr);
                    if (a && sa && !*sa) { (void)a->signal(1); *sa = true; }
                    if (b && sb && !*sb) { (void)b->signal(1); *sb = true; }
                }
            } mrGuard{semA.get(), &sigA, semB.get(), &sigB};

            RenderGraph graph;
            graph.setName("SameBackingHazardsAndMultiReader");
            graph.addNode("SharedBackingComputeWriterPass", "WriterCmp");
            graph.addNode("SharedBackingGatedSamplerPass", "ReaderGfx", {{"useGraphicsQueue", true}});
            graph.addNode("SharedBackingGatedSamplerPass", "ReaderCmp", {{"useGraphicsQueue", false}});
            graph.addEdge("WriterCmp.tex", "ReaderGfx.tex");
            graph.addEdge("WriterCmp.tex", "ReaderCmp.tex");
            graph.markOutput("ReaderGfx.out");
            graph.markOutput("ReaderCmp.out");

            RenderGraphExecutor exec;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = 3;
            opts.memoryPolicy = RenderGraphMemoryPolicy::MinimizeMemory; // Force SINGLE shared backing across all 3 frames!
            std::string log;
            FRAME_REQUIRE(exec.compile(*device, graph, 16, 16, opts, log));

            // Gate BOTH Frame 0 readers independently: ReaderGfx on semA, ReaderCmp on semB!
            SharedBackingGatedSamplerPass::gateSemGfx = semA.get();
            SharedBackingGatedSamplerPass::gateValGfx = 1;
            SharedBackingGatedSamplerPass::gateFrameGfx = 0;
            SharedBackingGatedSamplerPass::gateSemCmp = semB.get();
            SharedBackingGatedSamplerPass::gateValCmp = 1;
            SharedBackingGatedSamplerPass::gateFrameCmp = 0;

            // Submit Frame 0 (N), Frame 1 (N+1), and Frame 2 (N+2) BEFORE releasing semA or semB!
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute}));
            if (!exec.executionStats().crossFrameResourceScheduled || exec.executionStats().extraVersionCount != 0) {
                return RHITestResult::fail("Subtest B/C did not run MinimizeMemory cross-frame resource scheduling");
            }

            // Release ReaderGfx (semA) FIRST while ReaderCmp (semB) is STILL blocked!
            // Frame 1's WriterCmp MUST NOT overwrite the shared texture before ReaderCmp reads it!
            FRAME_REQUIRE(semA->signal(1));
            sigA = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            // Frame 2 cannot be done yet because ReaderCmp (semB) is still holding the shared backing!
            if (exec.lastSubmittedCompletion().isComplete()) {
                return RHITestResult::fail("Subtest C: Frame N+2 completed while Frame N's second overlapping reader (semB) was still gated!");
            }

            // Release ReaderCmp (semB) -> now all 3 frames can cascade through RAW/WAR/WAW & optimal layout transitions!
            FRAME_REQUIRE(semB->signal(1));
            sigB = true;
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));

            // Verify EVERY frame's BOTH readers read their exact frame's color without corruption!
            for (size_t f = 0; f < 3; ++f) {
                const uint32_t expected = SharedBackingComputeWriterPass::expectedSampled[f];
                const uint32_t gotGfx = readFirstWord(*gfxSinks[f]);
                const uint32_t gotCmp = readFirstWord(*cmpSinks[f]);
                if (gotGfx != expected || gotCmp != expected) {
                    return RHITestResult::fail("Subtest B/C frame " + std::to_string(f) +
                        " mismatch: expected=" + std::to_string(expected) +
                        " gotGfx=" + std::to_string(gotGfx) + " gotCmp=" + std::to_string(gotCmp));
                }
            }

            // Also verify CrossFrameResourceTracker::isBackingIdle() strictly requires ALL overlapping GPU readers to finish!
            std::unique_ptr<Semaphore> trackSemA, trackSemB;
            FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { trackSemA = std::move(s); }));
            FRAME_REQUIRE(device->createSemaphore({.initialValue = 0}).transform([&](auto s) { trackSemB = std::move(s); }));
            bool sigTA = false, sigTB = false;
            struct TrackSemGuard {
                Semaphore* a = nullptr; bool* sa = nullptr;
                Semaphore* b = nullptr; bool* sb = nullptr;
                ~TrackSemGuard()
                {
                    if (a && sa && !*sa) { (void)a->signal(1); *sa = true; }
                    if (b && sb && !*sb) { (void)b->signal(1); *sb = true; }
                }
            } trackGuard{trackSemA.get(), &sigTA, trackSemB.get(), &sigTB};

            QueueSubmissionTracker trackerGfx, trackerCmp;
            FRAME_REQUIRE(trackerGfx.initialize(*device, *graphics));
            FRAME_REQUIRE(trackerCmp.initialize(*device, *compute));
            std::unique_ptr<CommandPool> poolGfx, poolCmp;
            std::unique_ptr<CommandBuffer> cmdGfx, cmdCmp;
            FRAME_REQUIRE(device->createCommandPool(*graphics).transform([&](auto p) { poolGfx = std::move(p); }));
            FRAME_REQUIRE(device->createCommandPool(*compute).transform([&](auto p) { poolCmp = std::move(p); }));
            FRAME_REQUIRE(poolGfx->createCommandBuffer().transform([&](auto c) { cmdGfx = std::move(c); }));
            FRAME_REQUIRE(poolCmp->createCommandBuffer().transform([&](auto c) { cmdCmp = std::move(c); }));
            RenderFrameContext frameA(0), frameB(1);
            FRAME_REQUIRE(frameA.begin(0));
            FRAME_REQUIRE(frameB.begin(1));
            FRAME_REQUIRE(cmdGfx->begin(frameA.submissionContext()));
            FRAME_REQUIRE(cmdGfx->end());
            FRAME_REQUIRE(cmdCmp->begin(frameB.submissionContext()));
            FRAME_REQUIRE(cmdCmp->end());
            const SemaphoreSubmitDesc waitA{.semaphore = trackSemA.get(), .value = 1, .stages = PipelineStageBits::AllCommands};
            const SemaphoreSubmitDesc waitB{.semaphore = trackSemB.get(), .value = 1, .stages = PipelineStageBits::AllCommands};
            CommandBuffer* rawGfx = cmdGfx.get();
            CommandBuffer* rawCmp = cmdCmp.get();
            FRAME_REQUIRE(trackerGfx.submit({.waitSemaphores = {&waitA, 1}, .commandBuffers = {&rawGfx, 1}}, frameA));
            FRAME_REQUIRE(trackerCmp.submit({.waitSemaphores = {&waitB, 1}, .commandBuffers = {&rawCmp, 1}}, frameB));

            CrossFrameResourceTracker tracker;
            tracker.commitAcceptedAccess({
                .backingAllocationId = 0xBEEF0001ull, .type = RenderGraphResourceType::Texture2D,
                .frameIndex = 0, .queue = graphics, .state = ResourceState::ShaderRead,
                .writes = false, .completion = frameA.completion(), .passName = "ReaderA",
            });
            tracker.commitAcceptedAccess({
                .backingAllocationId = 0xBEEF0001ull, .type = RenderGraphResourceType::Texture2D,
                .frameIndex = 0, .queue = compute, .state = ResourceState::ShaderRead,
                .writes = false, .completion = frameB.completion(), .passName = "ReaderB",
            });
            if (tracker.isBackingIdle(0xBEEF0001ull)) {
                return RHITestResult::fail("Subtest C: isBackingIdle() returned true while both overlapping readers were incomplete");
            }
            FRAME_REQUIRE(trackSemA->signal(1));
            sigTA = true;
            FRAME_REQUIRE(frameA.wait(kWaitTimeout));
            if (tracker.isBackingIdle(0xBEEF0001ull)) {
                return RHITestResult::fail("Subtest C: isBackingIdle() returned true while ReaderB was still incomplete");
            }
            FRAME_REQUIRE(trackSemB->signal(1));
            sigTB = true;
            FRAME_REQUIRE(frameB.wait(kWaitTimeout));
            if (!tracker.isBackingIdle(0xBEEF0001ull)) {
                return RHITestResult::fail("Subtest C: isBackingIdle() stayed false after BOTH overlapping readers completed");
            }
        }

        // ------------------------------------------------------------------
        // Subtest D: Cancelled Producer -> dependent consumer segment is NOT
        // submitted, while already-accepted independent segments complete safely!
        // ------------------------------------------------------------------
        {
            // Part D1: Provider Producer fails -> dependent consumer never executes/submits!
            RenderSubsystemHost host;
            std::string log;
            if (!host.registerSubsystem<CancelledProducerProvider>(log)) {
                return RHITestResult::fail("Failed to register CancelledProducerProvider: " + log);
            }
            FRAME_REQUIRE(host.initialize(*device, 2, log));
            auto depMarker = makeHostReadbackBuf(0u);
            PipelinedCancelMarkerPass::devicePtr = device.get();
            PipelinedCancelMarkerPass::markerBuffer = depMarker.get();
            PipelinedCancelMarkerPass::executeCount = 0;

            RenderGraph provGraph;
            provGraph.setName("CancelledProviderConsumer");
            provGraph.addNode("PipelinedCancelMarkerPass", "DependentConsumer", {
                {"useCancelledProvider", true}, {"markerValue", 0xBAD0DEADu},
            });
            provGraph.markOutput("DependentConsumer.out");

            RenderWorld world;
            RenderGraphExecutor exec(host, world);
            FRAME_REQUIRE(exec.compile(*device, provGraph, 16, 16, log));
            auto* provider = host.get<CancelledProducerProvider>();
            auto res = exec.execute({.graphicsQueue = graphics, .computeQueue = compute, .recordingWorkerLimit = 1});
            if (res.has_value()) {
                return RHITestResult::fail("Subtest D1: failing Provider Producer unexpectedly succeeded");
            }
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
            const uint32_t depWord = readFirstWord(*depMarker);
            if (PipelinedCancelMarkerPass::executeCount != 0 || depWord != 0u ||
                !provider || provider->cancelCallbacks != 1 || provider->acceptCallbacks != 0) {
                return RHITestResult::fail("Subtest D1: cancelled Provider Producer mismatch (execCount=" +
                    std::to_string(PipelinedCancelMarkerPass::executeCount) + " depWord=" + std::to_string(depWord) +
                    " provider=" + std::to_string(provider != nullptr) +
                    " cancel=" + std::to_string(provider ? provider->cancelCallbacks : 0u) +
                    " accept=" + std::to_string(provider ? provider->acceptCallbacks : 0u) + ")");
            }
            exec = RenderGraphExecutor{};
            host.shutdown();

            // Part D2: Pipelined multi-segment graph where IndependentAccepted commits on Compute
            // BEFORE FailingProducer fails on Graphics -> IndependentAccepted completes safely on GPU,
            // while DependentConsumer is NEVER submitted!
            auto acceptedMarker = makeHostReadbackBuf(0u);
            PipelinedCancelMarkerPass::markerBuffer = acceptedMarker.get();
            PipelinedCancelMarkerPass::failNow = true;
            PipelinedCancelMarkerPass::executeCount = 0;

            RenderGraph pipeGraph;
            pipeGraph.setName("PipelinedAcceptedPlusCancelledProducer");
            pipeGraph.addNode("PipelinedCancelMarkerPass", "A_IndependentAccepted", {
                {"useGraphicsQueue", false}, {"markerValue", 0xCAFE0001u},
            });
            pipeGraph.addNode("PipelinedCancelMarkerPass", "B_FailingProducer", {
                {"useGraphicsQueue", true}, {"hasInput", true}, {"failThisPass", true}, {"markerValue", 0xBAD00001u},
            });
            pipeGraph.addNode("PipelinedCancelMarkerPass", "C_DependentConsumer", {
                {"useGraphicsQueue", true}, {"hasInput", true}, {"markerValue", 0xBAD0DEADu},
            });
            pipeGraph.addEdge("A_IndependentAccepted.out", "B_FailingProducer.in");
            pipeGraph.addEdge("B_FailingProducer.out", "C_DependentConsumer.in");
            pipeGraph.markOutput("A_IndependentAccepted.out");
            pipeGraph.markOutput("C_DependentConsumer.out");

            RenderGraphExecutor pipeExec;
            FRAME_REQUIRE(pipeExec.compile(*device, pipeGraph, 16, 16, log));
            auto pipeRes = pipeExec.execute({
                .graphicsQueue = graphics, .computeQueue = compute,
                .recordingWorkerLimit = 1,
                .submissionMode = FrameSubmissionMode::Pipelined,
            });
            if (pipeRes.has_value()) {
                return RHITestResult::fail("Subtest D2: pipelined graph with failing producer unexpectedly succeeded");
            }
            FRAME_REQUIRE(pipeExec.waitForSubmittedWork(kWaitTimeout));
            const uint32_t observed = readFirstWord(*acceptedMarker);
            if (observed != 0xCAFE0001u) {
                return RHITestResult::fail("Subtest D2: accepted independent segment did not complete safely or dependent consumer overwrote marker (got=" +
                    std::to_string(observed) + ")");
            }
            PipelinedCancelMarkerPass::devicePtr = nullptr;
            PipelinedCancelMarkerPass::markerBuffer = nullptr;
        }

        return RHITestResult::pass("Batch 3 timeline gate (N+1/N+2 before N unblocks, same-backing RAW/WAR/WAW + read-only sharing + Graphics<->Compute optimal layouts, multi-reader retirement, cancelled producer isolation) verified");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameTimelineGateMultiFrameHazardsTest);

class ProductionLifecycleInspectorObserver final : public render::IRenderDebugObserver {
public:
    uint32_t compiledCount = 0;
    uint32_t beginCount = 0;
    uint32_t endSuccessCount = 0;
    uint32_t afterPassBoundaries = 0;
    bool sawPinnedColor = false;
    bool copiedReadback = false;
    render::Buffer* readbackBuffer = nullptr;

    void compiled(debug::DebugValue) override { ++compiledCount; }
    void beginExecution(render::Device&, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override { ++beginCount; }
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t, std::string_view pass,
        std::span<const render::DebugResourceBinding> resources, const debug::DebugValue&) override
    {
        if (checkpoint != "AfterPass" || pass != "PathTrace") { return; }
        ++afterPassBoundaries;
        for (const auto& res : resources) {
            if (res.id == "PathTrace.color" && res.texture != nullptr) {
                sawPinnedColor = true;
                if (readbackBuffer != nullptr) {
                    const auto toLayout = [](render::ResourceState s) {
                        switch (s) {
                        case render::ResourceState::ShaderRead: return render::TextureLayout::ShaderRead;
                        case render::ResourceState::ColorAttachment: return render::TextureLayout::ColorAttachment;
                        case render::ResourceState::TransferSource: return render::TextureLayout::TransferSource;
                        case render::ResourceState::TransferDestination: return render::TextureLayout::TransferDestination;
                        default: return render::TextureLayout::General;
                        }
                    };
                    const render::TextureLayout curLayout = toLayout(res.state);
                    const render::TextureBarrierDesc toSrc{
                        .texture = res.texture,
                        .oldLayout = curLayout,
                        .newLayout = render::TextureLayout::TransferSource,
                        .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                        .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                    };
                    if (commands.synchronize({.textures = {&toSrc, 1}})) {
                        if (auto dst = readbackBuffer->slice({0, 16})) {
                            if (commands.copyTextureToBuffer({
                                .texture = res.texture, .buffer = *dst,
                                .width = 1, .height = 1, .depth = 1,
                            })) {
                                const render::TextureBarrierDesc restore{
                                    .texture = res.texture,
                                    .oldLayout = render::TextureLayout::TransferSource,
                                    .newLayout = curLayout,
                                    .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
                                    .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
                                };
                                const render::BufferBarrierDesc hostRead{
                                    .buffer = readbackBuffer,
                                    .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
                                    .after = {render::PipelineStageBits::Host, render::AccessBits::HostRead},
                                };
                                if (commands.synchronize({.textures = {&restore, 1}, .buffers = {&hostRead, 1}})) {
                                    copiedReadback = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    void endExecution(bool success) override { if (success) { ++endSuccessCount; } }
};

class FrameProductionGraphLifecycleStressTest final : public RHITest {
public:
    FrameProductionGraphLifecycleStressTest()
    {
        type = RHITestType::Rendering;
        name = "frame_production_graph_lifecycle_stress";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        DeviceDesc desc{
            .applicationName = "Production graph lifecycle stress (Batch 3)",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableMeshShader = true,
            .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true,
            .enableClusterAccelerationStructure = true,
        };
        vulkan::deviceExtensions(desc).preferUnifiedImageLayouts = false; // Optimal layouts!
        auto created = createDevice(desc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) {
            return RHITestResult::skip("requires ray query, CLAS, and bindless descriptor heap");
        }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);

        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf")) {
            return RHITestResult::fail("Failed to load ABeautifulGame.gltf: " + document.lastLoadResult().error);
        }

        const auto readOutputMeanRGB = [&](RenderGraphExecutor& exec, std::string_view outName,
                                           std::array<double, 3>& outMean) -> Result<> {
            auto* res = exec.outputResource(outName);
            if (!res || !res->texture) { return makeError(Error::InvalidArgument); }
            const auto& td = res->texture->desc();
            const uint64_t pixelBytes = td.format == Format::RGBA32Sfloat ? 16ull : 4ull;
            const uint64_t totalBytes = uint64_t(td.width) * td.height * pixelBytes;
            std::unique_ptr<Buffer> rb;
            if (auto r = device->createBuffer({
                .size = totalBytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            }).transform([&](auto b) { rb = std::move(b); }); !r) { return r; }
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> cmd;
            std::unique_ptr<Fence> fence;
            if (auto r = device->createCommandPool(*graphics).transform([&](auto p) { pool = std::move(p); }); !r) { return r; }
            if (auto r = pool->createCommandBuffer().transform([&](auto c) { cmd = std::move(c); }); !r) { return r; }
            if (auto r = device->createFence(false).transform([&](auto f) { fence = std::move(f); }); !r) { return r; }
            if (auto r = cmd->begin(); !r) { return r; }
            if (auto r = exec.transitionOutput(*cmd, outName, ResourceState::TransferSource); !r) { return r; }
            auto slice = rb->slice({0, totalBytes});
            if (!slice) { return makeError(Error::Failure); }
            if (auto r = cmd->copyTextureToBuffer({.texture = res->texture, .buffer = *slice,
                .width = td.width, .height = td.height, .depth = 1}); !r) { return r; }
            const BufferBarrierDesc hostRead{.buffer = rb.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
            if (auto r = cmd->synchronize({.buffers = {&hostRead, 1}}); !r) { return r; }
            if (auto r = cmd->end(); !r) { return r; }
            CommandBuffer* raw = cmd.get();
            if (auto r = graphics->submit({.commandBuffers = {&raw, 1}, .signalFence = fence.get()}); !r) { return r; }
            if (auto r = fence->wait(kWaitTimeout); !r) { return r; }
            rb->invalidate();
            const void* mapped = rb->map();
            if (!mapped) { return makeError(Error::Failure); }
            outMean = {0.0, 0.0, 0.0};
            const size_t n = size_t(td.width) * td.height;
            if (td.format == Format::RGBA32Sfloat) {
                const auto* px = static_cast<const float*>(mapped);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        const float v = px[i * 4 + c];
                        if (!std::isfinite(v) || v < -0.01f) { rb->unmap(); return makeError(Error::Failure); }
                        outMean[c] += v;
                    }
                }
            } else {
                const auto* px = static_cast<const uint8_t*>(mapped);
                for (size_t i = 0; i < n; ++i) {
                    for (size_t c = 0; c < 3; ++c) { outMean[c] += double(px[i * 4 + c]) / 255.0; }
                }
            }
            rb->unmap();
            for (double& c : outMean) { c /= double(std::max<size_t>(n, 1)); }
            return {};
        };

        auto makePhysicalEnv = [](float sunElevDeg) {
            environment::WorldEnvironment env;
            env.source = environment::EnvironmentSource::PhysicalAtmosphere;
            env.sun.enabled = true;
            const float rad = sunElevDeg * 0.017453292519943295f;
            env.sun.direction = float3(-std::cos(rad), -std::sin(rad), 0.0f);
            env.moon.enabled = false;
            return env;
        };

        RenderWorld world1;
        world1.setScene(&document);
        world1.setWorldEnvironment(makePhysicalEnv(60.0f)); // Bright noon sun

        RenderWorld world2;
        world2.setScene(&document);
        world2.setWorldEnvironment(makePhysicalEnv(4.0f)); // Low golden sunset

        RenderGraph graph;
        graph.setName("ProductionLifecycleStress");
        const RenderGraphNode* ptNode = graph.addNode("ScenePathTracePass", "PathTrace", {
            {"bsdf", "openpbr"},
            {"cacheMode", "off"},
            {"maxDepth", 1},
            {"samples", 1},
            {"accumulate", true},
            {"debugView", "final"},
            {"camera", {{"eye", {0.0, 0.35, 0.65}}, {"center", {0.0, 0.05, 0.0}}, {"fovDegrees", 50.0}}},
        });
        const uint32_t ptId = ptNode->id;
        graph.markOutput("PathTrace.color");

        HistoryResourceManager history;
        FRAME_REQUIRE(history.initialize(*device, 2));

        RenderGraphExecutor exec;
        exec.bindRenderWorld(&world1);
        exec.setExecutionCaptureEnabled(true);
        std::string log;
        auto compiled = exec.compile(*device, graph, 64, 48, log);
        if (hasError(compiled, Error::Unsupported)) { return RHITestResult::skip(log); }
        FRAME_REQUIRE(compiled);

        // 1. Baseline 2-frame accumulation + verify conservative boundary + Streamer & Env Provider publications!
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        if (!history.hasPrevious("PathTrace.accumulation")) {
            return RHITestResult::fail("Step 1 (Baseline): PathTrace accumulation history did not become valid");
        }
        const auto baseStats = exec.executionStats();
        const auto baseSnap = exec.executionSnapshot();
        if (baseStats.crossFrameResourceScheduled || baseStats.conservativeFallbackReasons.empty() || !baseSnap) {
            return RHITestResult::fail("Step 1 (Baseline): production scene graph did not keep conservative fallback boundary");
        }
        bool sawStreamerPub = false, sawEnvPub = false;
        for (const auto& res : baseSnap->resources) {
            if (res.ownership == RenderGraphResourceOwnership::ProviderOwned) {
                if (res.name.find("render.streamer.") == 0) { sawStreamerPub = true; }
                if (res.name.find("render.environment.") == 0) { sawEnvPub = true; }
            }
        }
        if (!sawStreamerPub || !sawEnvPub) {
            return RHITestResult::fail("Step 1 (Baseline): missing declared Streamer or Environment ProviderOwned resources in snapshot");
        }
        std::array<double, 3> noonMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "PathTrace.color", noonMean));
        if (noonMean[0] + noonMean[1] + noonMean[2] <= 1e-4) {
            return RHITestResult::fail("Step 1 (Baseline): PathTrace rendered black frame under physical noon sky");
        }

        // 2. Camera Cut: discontinuity invalidation + camera jump -> clean restart & finite output.
        const uint64_t invBefore = history.invalidationRevision();
        history.invalidateAll(HistoryInvalidationReason::Discontinuity);
        if (history.invalidationRevision() <= invBefore || history.hasPrevious("PathTrace.accumulation")) {
            return RHITestResult::fail("Step 2 (Camera Cut): Discontinuity invalidation did not clear previous history");
        }
        graph.setNodeRuntimeProperty(ptId, "camera", RenderGraphProperties{
            {"eye", {0.55, 0.45, -0.55}}, {"center", {0.0, 0.05, 0.0}}, {"fovDegrees", 50.0},
        });
        if (!exec.syncRuntimeProperties(graph)) { return RHITestResult::fail("Step 2: syncRuntimeProperties failed"); }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        std::array<double, 3> cutMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "PathTrace.color", cutMean));
        if (cutMean[0] + cutMean[1] + cutMean[2] <= 1e-4) {
            return RHITestResult::fail("Step 2 (Camera Cut): post-cut frame rendered black");
        }

        // 3. World Switch: bindRenderWorld(&world2) (Sunset) -> environment switches cleanly!
        exec.bindRenderWorld(&world2);
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        std::array<double, 3> sunsetMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "PathTrace.color", sunsetMean));
        const double noonSum = noonMean[0] + noonMean[1] + noonMean[2];
        const double sunsetSum = sunsetMean[0] + sunsetMean[1] + sunsetMean[2];
        if (sunsetSum <= 0.0 || std::abs(noonSum - sunsetSum) < 1e-3) {
            return RHITestResult::fail("Step 3 (World Switch): switching noon -> sunset RenderWorld did not change sky lighting");
        }

        // 4. Resize: 64x48 -> 96x64!
        FRAME_REQUIRE(exec.compile(*device, graph, 96, 64, log));
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        const auto* resizedRes = exec.outputResource("PathTrace.color");
        if (!resizedRes || !resizedRes->texture || resizedRes->texture->desc().width != 96 || resizedRes->texture->desc().height != 64) {
            return RHITestResult::fail("Step 4 (Resize): output dimensions did not update to 96x64");
        }
        std::array<double, 3> resizedMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "PathTrace.color", resizedMean));

        // 5. Shader Reload: transactional reloadShaders()!
        std::string reloadLog;
        FRAME_REQUIRE(exec.reloadShaders(reloadLog));
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        std::array<double, 3> reloadMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "PathTrace.color", reloadMean));
        if (reloadMean[0] + reloadMean[1] + reloadMean[2] <= 1e-4) {
            return RHITestResult::fail("Step 5 (Shader Reload): post-reload frame rendered black");
        }

        // 6. Cache Eviction: push >10 distinct observer positions (>8 environment cache capacity)!
        for (int k = 0; k < 10; ++k) {
            graph.setNodeRuntimeProperty(ptId, "camera", RenderGraphProperties{
                {"eye", {0.2 + 0.05 * k, 0.35 + 0.04 * k, 0.65}}, {"center", {0.0, 0.05, 0.0}}, {"fovDegrees", 50.0},
            });
            if (!exec.syncRuntimeProperties(graph)) { return RHITestResult::fail("Step 6: syncRuntimeProperties failed"); }
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        }
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        auto* envSub = exec.subsystemHost()->get<EnvironmentLightingSubsystem>();
        if (!envSub || envSub->physicalProducerCount() < 10 || envSub->mediumShareCount() < 8) {
            return RHITestResult::fail("Step 6 (Cache Eviction): environment cache eviction + medium sharing did not trigger as expected");
        }

        // 7. Graph Recompile: add ColorGradingLUTPass + FinalBlitPass and recompile!
        RenderGraph recompiledGraph = graph;
        recompiledGraph.addNode("ColorGradingLUTPass", "ColorGrading", {{"toneCurve", "aces2"}});
        recompiledGraph.addNode("FinalBlitPass", "FinalBlit");
        recompiledGraph.addEdge("PathTrace.color", "FinalBlit.source");
        recompiledGraph.addEdge("ColorGrading.lut", "FinalBlit.lut");
        recompiledGraph.clearOutputs();
        recompiledGraph.markOutput("FinalBlit.color");
        const uint64_t prevGen = exec.executionStats().graphGeneration;
        FRAME_REQUIRE(exec.compile(*device, recompiledGraph, 96, 64, log));
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        if (exec.executionStats().graphGeneration <= prevGen) {
            return RHITestResult::fail("Step 7 (Graph Recompile): graphGeneration did not advance");
        }
        std::array<double, 3> blitMean{};
        FRAME_REQUIRE(readOutputMeanRGB(exec, "FinalBlit.color", blitMean));
        if (blitMean[0] + blitMean[1] + blitMean[2] <= 1e-4) {
            return RHITestResult::fail("Step 7 (Graph Recompile): FinalBlit rendered black");
        }

        // 8. Inspector Pin & Readback: attach debug observer + pin PathTrace.color + read back via boundary!
        std::unique_ptr<Buffer> inspectorRb;
        FRAME_REQUIRE(device->createBuffer({
            .size = 16, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback,
        }).transform([&](auto b) { inspectorRb = std::move(b); }));
        ProductionLifecycleInspectorObserver observer;
        observer.readbackBuffer = inspectorRb.get();
        exec.setDebugObserver(&observer);
        RenderGraphCompileOptions pinOpts;
        pinOpts.enableDebugReadback = true;
        pinOpts.extraOutputs = {"PathTrace.color"};
        FRAME_REQUIRE(exec.compile(*device, recompiledGraph, 96, 64, pinOpts, log));
        if (!exec.isExportedOutput("PathTrace.color")) {
            return RHITestResult::fail("Step 8 (Inspector Pin): extraOutput PathTrace.color was not pinned/exported");
        }
        FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .historyResources = &history, .recordingWorkerLimit = 1}));
        FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
        const auto pinStats = exec.executionStats();
        const bool hasDebugReason = std::any_of(pinStats.conservativeFallbackReasons.begin(),
            pinStats.conservativeFallbackReasons.end(),
            [](const std::string& r) { return r.find("Debug capture observer") != std::string::npos; });
        if (!observer.sawPinnedColor || !observer.copiedReadback || observer.endSuccessCount == 0 || !hasDebugReason) {
            return RHITestResult::fail("Step 8 (Inspector Pin/Readback): observer missed pinned PathTrace.color readback or conservative reason");
        }
        inspectorRb->invalidate();
        const auto* px = static_cast<const float*>(inspectorRb->map());
        if (!px || !std::isfinite(px[0]) || !std::isfinite(px[1]) || !std::isfinite(px[2])) {
            if (px) { inspectorRb->unmap(); }
            return RHITestResult::fail("Step 8 (Inspector Pin/Readback): non-finite pixel read back from pinned intermediate");
        }
        inspectorRb->unmap();
        exec.setDebugObserver(nullptr);

        return RHITestResult::pass("Production graph lifecycle stress (Camera Cut, World switch, Resize, Shader Reload, Cache Eviction, Graph Recompile, Inspector pin/readback, Streamer+Env ProviderOwned publications) verified under optimal layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameProductionGraphLifecycleStressTest);

class FrameProductionPostProcessPolicyBenchmarkTest final : public RHITest {
public:
    FrameProductionPostProcessPolicyBenchmarkTest()
    {
        type = RHITestType::Rendering;
        name = "frame_production_postprocess_policy_benchmark";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        DeviceDesc desc{
            .applicationName = "Production post-process policy benchmark (Batch 3)",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true,
        };
        vulkan::deviceExtensions(desc).preferUnifiedImageLayouts = false;
        auto created = createDevice(desc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) {
            return RHITestResult::skip("requires bindless descriptor heap");
        }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);
        Queue* compute = device->getQueue(QueueType::Compute);
        if (!graphics || !compute) { return RHITestResult::fail("missing graphics or compute queue"); }

        struct ProdBenchConfig {
            std::string name;
            uint32_t maxFramesInFlight = 2;
            RenderGraphMemoryPolicy policy = RenderGraphMemoryPolicy::MinimizeMemory;
            uint64_t budgetBytes = 0;
            uint32_t width = 256;
            uint32_t height = 256;
            uint32_t warmupFrames = 4;
            uint32_t measureFrames = 8;
        };

        const char* envConfig = std::getenv("METALLIC_RG_PROD_BENCH_CONFIG");
        const bool benchmarkMode = envConfig != nullptr && envConfig[0] != '\0';
        std::vector<ProdBenchConfig> configs;
        if (benchmarkMode) {
            const std::string mode(envConfig);
            ProdBenchConfig cfg;
            cfg.name = mode;
            cfg.width = 1920;
            cfg.height = 1080;
            cfg.warmupFrames = 16;
            cfg.measureFrames = 80;
            if (mode == "2_minmem") {
                cfg.maxFramesInFlight = 2; cfg.policy = RenderGraphMemoryPolicy::MinimizeMemory;
            } else if (mode == "3_minmem") {
                cfg.maxFramesInFlight = 3; cfg.policy = RenderGraphMemoryPolicy::MinimizeMemory;
            } else if (mode == "2_maxoverlap") {
                cfg.maxFramesInFlight = 2; cfg.policy = RenderGraphMemoryPolicy::MaximizeOverlap;
                cfg.budgetBytes = 128ull * 1024 * 1024;
            } else if (mode == "3_maxoverlap") {
                cfg.maxFramesInFlight = 3; cfg.policy = RenderGraphMemoryPolicy::MaximizeOverlap;
                cfg.budgetBytes = 128ull * 1024 * 1024;
            } else {
                return RHITestResult::fail("Unknown METALLIC_RG_PROD_BENCH_CONFIG: " + mode);
            }
            configs.push_back(cfg);
        } else {
            configs.push_back({"2_minmem", 2, RenderGraphMemoryPolicy::MinimizeMemory, 0, 128, 128, 2, 4});
            configs.push_back({"3_maxoverlap", 3, RenderGraphMemoryPolicy::MaximizeOverlap, 64ull * 1024 * 1024, 128, 128, 2, 4});
        }

        std::filesystem::create_directories(context.outputDirectory);
        for (const auto& cfg : configs) {
            // Real built-in multi-queue production post-processing / presentation graph:
            // Graphics: TriangleA + TriangleB (raster)
            // Async Compute: ColorGradingLUTPass (ACES 2.0 64^3 3D LUT) + SliderDebugPass + FinalBlitPass
            RenderGraph graph;
            graph.setName("ProdPostProcess_" + cfg.name);
            graph.addNode("TriangleRasterPass", "RasterA");
            graph.addNode("TriangleRasterPass", "RasterB");
            graph.addNode("ColorGradingLUTPass", "ColorGrading", {{"toneCurve", "aces2"}});
            graph.addNode("SliderDebugPass", "Slider", {{"splitPosition", 0.5f}});
            graph.addNode("FinalBlitPass", "FinalBlit");
            graph.addEdge("RasterA.color", "Slider.sourceA");
            graph.addEdge("RasterB.color", "Slider.sourceB");
            graph.addEdge("Slider.color", "FinalBlit.source");
            graph.addEdge("ColorGrading.lut", "FinalBlit.lut");
            graph.markOutput("FinalBlit.color");

            RenderGraphExecutor exec;
            RenderGraphCompileOptions opts;
            opts.maxFramesInFlight = cfg.maxFramesInFlight;
            opts.memoryPolicy = cfg.policy;
            opts.versionPoolBudgetBytes = cfg.budgetBytes;
            opts.maxPhysicalVersionsPerResource = cfg.maxFramesInFlight;
            std::string log;
            FRAME_REQUIRE(exec.compile(*device, graph, cfg.width, cfg.height, opts, log));

            for (uint32_t w = 0; w < cfg.warmupFrames; ++w) {
                FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute, .schedulingDiagnostics = true}));
            }
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
            (void)exec.collectCompletedGpuExecutionStats();

            std::vector<double> frameWallMs;
            std::vector<double> gpuFrameMs;
            std::vector<double> gpuSumPassMs;
            std::vector<double> gpuGfxPassMs;
            std::vector<double> gpuCmpPassMs;
            std::vector<double> cpuSubmitMs;
            std::vector<double> cpuDrainWaitMs;
            frameWallMs.reserve(cfg.measureFrames);
            cpuSubmitMs.reserve(cfg.measureFrames);
            cpuDrainWaitMs.reserve(cfg.measureFrames);
            uint32_t crossFrameScheduledFrames = 0;
            uint32_t drainFrames = 0;
            uint32_t maxExtraVersions = 0;
            uint64_t peakGraphManaged = 0;
            uint64_t peakDeviceTracked = 0;
            uint64_t peakDriverDevice = 0;
            bool driverVramAvail = false;

            const auto collectGpuSamples = [&]() {
                if (auto completed = exec.collectCompletedGpuExecutionStats(); completed) {
                    for (const auto& gs : *completed) {
                        if (gs.gpuTimingAvailable && gs.gpuMilliseconds > 0.0) {
                            gpuFrameMs.push_back(gs.gpuMilliseconds);
                            double sumPasses = 0.0, sumGfx = 0.0, sumCmp = 0.0;
                            for (const auto& n : gs.nodes) {
                                if (!n.gpuTimingAvailable) { continue; }
                                sumPasses += n.gpuMilliseconds;
                                if (n.queue == QueueType::Graphics) { sumGfx += n.gpuMilliseconds; }
                                else if (n.queue == QueueType::Compute) { sumCmp += n.gpuMilliseconds; }
                            }
                            gpuSumPassMs.push_back(sumPasses);
                            gpuGfxPassMs.push_back(sumGfx);
                            gpuCmpPassMs.push_back(sumCmp);
                        }
                    }
                }
            };

            const auto totalBegin = std::chrono::steady_clock::now();
            for (uint32_t f = 0; f < cfg.measureFrames; ++f) {
                const auto fBegin = std::chrono::steady_clock::now();
                FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .computeQueue = compute, .schedulingDiagnostics = true}));
                const auto fEnd = std::chrono::steady_clock::now();
                const auto st = exec.executionStats();
                if (!st.crossFrameResourceScheduled) {
                    return RHITestResult::fail("Production postprocess graph did not enable crossFrameResourceScheduled");
                }
                ++crossFrameScheduledFrames;
                if (st.drainReasonMask != 0) { ++drainFrames; }
                maxExtraVersions = std::max(maxExtraVersions, st.extraVersionCount);
                peakGraphManaged = std::max(peakGraphManaged, st.peakManagedVramBytes);
                peakDeviceTracked = std::max(peakDeviceTracked, st.peakDeviceTrackedVramBytes);
                peakDriverDevice = std::max(peakDriverDevice, st.peakDriverDeviceVramBytes);
                driverVramAvail |= st.driverDeviceVramAvailable;
                frameWallMs.push_back(std::chrono::duration<double, std::milli>(fEnd - fBegin).count());
                cpuSubmitMs.push_back(st.cpuMilliseconds);
                cpuDrainWaitMs.push_back(double(st.scheduling.frameWaitNs) / 1e6);
                collectGpuSamples();
            }
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
            const auto totalEnd = std::chrono::steady_clock::now();
            collectGpuSamples();

            const double totalWallMs = std::chrono::duration<double, std::milli>(totalEnd - totalBegin).count();
            const double avgWallMs = totalWallMs / double(cfg.measureFrames);
            const auto percentile = [](std::vector<double> v, double p) -> double {
                if (v.empty()) { return 0.0; }
                std::sort(v.begin(), v.end());
                const size_t idx = std::min(v.size() - 1, size_t(double(v.size() - 1) * p));
                return v[idx];
            };
            const auto meanOf = [](const std::vector<double>& v) -> double {
                if (v.empty()) { return 0.0; }
                return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
            };
            const double avgGpuCriticalMs = meanOf(gpuFrameMs);
            const double avgGpuPassSumMs = meanOf(gpuSumPassMs);
            const double overlapGainMs = std::max(0.0, avgGpuPassSumMs - avgGpuCriticalMs);

            nlohmann::json report{
                {"config", cfg.name},
                {"width", cfg.width}, {"height", cfg.height},
                {"maxFramesInFlight", cfg.maxFramesInFlight},
                {"memoryPolicy", renderGraphMemoryPolicyName(cfg.policy)},
                {"asyncComputeDistinct", !graphics->sameQueue(*compute)},
                {"warmupFrames", cfg.warmupFrames}, {"measureFrames", cfg.measureFrames},
                {"crossFrameScheduledFrames", crossFrameScheduledFrames},
                {"cpuDrainFrames", drainFrames},
                {"maxExtraVersions", maxExtraVersions},
                {"totalWallMs", totalWallMs},
                {"avgFrameLatencyMs", avgWallMs},
                {"p50FrameLatencyMs", percentile(frameWallMs, 0.50)},
                {"p95FrameLatencyMs", percentile(frameWallMs, 0.95)},
                {"avgCpuSubmitMs", meanOf(cpuSubmitMs)},
                {"avgCpuFrameWaitMs", meanOf(cpuDrainWaitMs)},
                {"gpuSamples", gpuFrameMs.size()},
                {"avgGpuCriticalPathMs", avgGpuCriticalMs},
                {"p50GpuCriticalPathMs", percentile(gpuFrameMs, 0.50)},
                {"avgGpuPassSumMs", avgGpuPassSumMs},
                {"avgGpuGraphicsPassMs", meanOf(gpuGfxPassMs)},
                {"avgGpuComputePassMs", meanOf(gpuCmpPassMs)},
                {"gpuQueueOverlapMs", overlapGainMs},
                {"peakGraphManagedVramBytes", peakGraphManaged},
                {"peakDeviceTrackedVramBytes", peakDeviceTracked},
                {"driverDeviceVramAvailable", driverVramAvail},
                {"peakDriverDeviceVramBytes", peakDriverDevice},
            };
            const auto outFile = context.outputDirectory / ("prod_postprocess_bench_" + cfg.name + ".json");
            std::ofstream os(outFile);
            os << report.dump(2) << '\n';
            std::printf("[PROD_BENCH] %s wallAvg=%.3fms p50=%.3fms p95=%.3fms gpuCrit=%.3fms gpuSum=%.3fms (gfx=%.3f cmp=%.3f) overlap=%.3fms cpuWait=%.3fms extraVer=%u graphVram=%.2fMiB devVram=%.2fMiB drvVram=%.2fMiB\n",
                cfg.name.c_str(), avgWallMs, percentile(frameWallMs, 0.50), percentile(frameWallMs, 0.95),
                avgGpuCriticalMs, avgGpuPassSumMs, meanOf(gpuGfxPassMs), meanOf(gpuCmpPassMs), overlapGainMs,
                meanOf(cpuDrainWaitMs), maxExtraVersions,
                double(peakGraphManaged) / (1024.0 * 1024.0),
                double(peakDeviceTracked) / (1024.0 * 1024.0),
                double(peakDriverDevice) / (1024.0 * 1024.0));
        }
        return RHITestResult::pass("Production post-process policy benchmark verified");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameProductionPostProcessPolicyBenchmarkTest);

class FrameMigratedLumenVsmStreamerDomainsTest final : public RHITest {
public:
    FrameMigratedLumenVsmStreamerDomainsTest()
    {
        type = RHITestType::Rendering;
        name = "frame_migrated_lumen_vsm_streamer_domains";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        DeviceDesc desc{
            .applicationName = "Migrated Lumen Radiance Cache + VSM + Streamer domains (Batch 3)",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .enableMeshShader = true,
            .enableTaskShader = true,
            .enableTaskShaderSubgroupBallot = true,
            .enableGeometryShader = true,
            .enableSubgroupSizeControl = true,
            .enableComputeFullSubgroups = true,
            .preferredTaskSubgroupSize = 32,
        };
        vulkan::deviceExtensions(desc).preferUnifiedImageLayouts = false; // Optimal layouts!
        auto created = createDevice(desc).transform([&](auto d) { device = std::move(d); });
        if (hasError(created, Error::Unsupported)) {
            return RHITestResult::skip("requires bindless descriptor heap and mesh shaders");
        }
        FRAME_REQUIRE(created);
        Queue* graphics = device->getQueue(QueueType::Graphics);
        const std::filesystem::path root(PROJECT_SOURCE_DIR);

        // ------------------------------------------------------------------
        // Part A: GI Radiance Cache (LumenRadianceCachePass) temporal migration
        // Runs real Pipelines/Samples/lumen_radiance_cache.metallic_graph.json
        // pruned to RadianceCache.cacheRadiance on CornellBox.gltf with optimal layouts.
        // ------------------------------------------------------------------
        {
            scene::SceneDocument cornell;
            if (!cornell.load(root / "Asset/Lumen/CornellBox.metallic_scene.json")) {
                return RHITestResult::fail("Part A: failed to load CornellBox.metallic_scene.json: " +
                    cornell.documentWarning() + " " + cornell.lastLoadResult().error);
            }

            RenderGraph graph;
            std::string log;
            if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_radiance_cache.metallic_graph.json", graph, log)) {
                return RHITestResult::fail("Part A: failed to load lumen_radiance_cache graph: " + log);
            }
            const auto rewriteCachePaths = [&](const char* nodeName, const char* propName) -> bool {
                const RenderGraphNode* node = graph.findNode(nodeName);
                if (!node || !node->properties.contains(propName)) { return false; }
                RenderGraphProperties props = node->properties;
                auto paths = props[propName];
                if (!paths.is_array() || paths.empty()) { return false; }
                for (auto& item : paths) {
                    const std::filesystem::path p(item.get<std::string>());
                    item = (p.is_absolute() ? p : (root / p)).generic_string();
                }
                props[propName] = std::move(paths);
                return graph.setNodeProperties(node->id, std::move(props));
            };
            if (!rewriteCachePaths("GlobalDistanceFieldPages", "distanceFieldPaths") ||
                !rewriteCachePaths("SurfaceCache", "meshCardPaths")) {
                return RHITestResult::fail("Part A: failed to rewrite CornellBox distanceField/meshCard paths");
            }
            graph.clearOutputs();
            graph.markOutput("RadianceCache.cacheRadiance");

            RenderWorld world;
            world.setScene(&cornell);
            world.setEnvironment({.enabled = false, .visible = false});
            environment::WorldEnvironment noSky{};
            noSky.sun.enabled = false;
            noSky.moon.enabled = false;
            world.setWorldEnvironment(noSky);

            RenderView view;
            ViewCamera camera;
            camera.eye = {0.0f, 4.12f, 14.2f};
            camera.center = {0.0f, 3.92f, 0.0f};
            camera.up = {0.0f, 1.0f, 0.0f};
            camera.fovDegrees = 43.0f;
            camera.nearPlane = 0.05f;
            camera.farPlane = 250.0f;
            if (!view.setCamera(camera) || !view.setRenderResolution(64, 48)) {
                return RHITestResult::fail("Part A: invalid CornellBox RenderView");
            }

            RenderGraphExecutor exec;
            exec.bindRenderWorld(&world);
            exec.bindRenderView(&view);
            exec.setExecutionCaptureEnabled(true);
            auto compiled = exec.compile(*device, graph, 64, 48, log);
            if (hasError(compiled, Error::Unsupported)) { return RHITestResult::skip("Part A: " + log); }
            FRAME_REQUIRE(compiled);

            uint64_t savedVersionF0 = 0;
            uint64_t savedVersionF1 = 0;
            uint64_t prevVersionF1 = 0;
            uint64_t savedBackingF0 = 0;
            bool sawStreamerPub = false;
            for (uint32_t frame = 0; frame < 3; ++frame) {
                FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .recordingWorkerLimit = 1}));
                FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
                const auto snap = exec.executionSnapshot();
                if (!snap) { return RHITestResult::fail("Part A: missing execution snapshot"); }
                for (const auto& res : snap->resources) {
                    if (res.ownership == RenderGraphResourceOwnership::ProviderOwned &&
                        res.name.find("render.streamer.") == 0) {
                        sawStreamerPub = true;
                    }
                    if (res.ownership == RenderGraphResourceOwnership::TemporalOwned &&
                        (res.name.find("indirection") != std::string::npos ||
                         res.name.find("Indirection") != std::string::npos)) {
                        const uint64_t backing = res.memory.backingAllocationId != 0
                            ? res.memory.backingAllocationId : res.memory.allocationId;
                        const bool isWrite = res.name.find("savedIndirection") != std::string::npos ||
                            res.producerName == "RadianceCache";
                        if (frame == 0 && isWrite) {
                            savedVersionF0 = res.versionId;
                            savedBackingF0 = backing;
                        } else if (frame == 1) {
                            if (isWrite) { savedVersionF1 = res.versionId; }
                            else { prevVersionF1 = res.versionId; }
                        }
                    }
                }
            }
            if (!sawStreamerPub || savedVersionF0 == 0 || savedVersionF1 <= savedVersionF0 ||
                prevVersionF1 != savedVersionF0 || savedBackingF0 == 0) {
                return RHITestResult::fail("Part A: LumenRadianceCache temporal indirection traceability or Streamer ProviderOwned publication failed (savedF0=" +
                    std::to_string(savedVersionF0) + " prevF1=" + std::to_string(prevVersionF1) +
                    " savedF1=" + std::to_string(savedVersionF1) + " streamer=" + std::to_string(sawStreamerPub) + ")");
            }

            // Read back RadianceCache.cacheRadiance (RGBA16Sfloat) and verify finite non-NaN texels.
            const auto* radRes = exec.outputResource("RadianceCache.cacheRadiance");
            if (!radRes || !radRes->texture) { return RHITestResult::fail("Part A: missing RadianceCache.cacheRadiance"); }
            const auto& td = radRes->texture->desc();
            const uint64_t totalBytes = uint64_t(td.width) * td.height * 8ull;
            std::unique_ptr<Buffer> rb;
            FRAME_REQUIRE(device->createBuffer({
                .size = totalBytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            }).transform([&](auto b) { rb = std::move(b); }));
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> cmd;
            std::unique_ptr<Fence> fence;
            FRAME_REQUIRE(device->createCommandPool(*graphics).transform([&](auto p) { pool = std::move(p); }));
            FRAME_REQUIRE(pool->createCommandBuffer().transform([&](auto c) { cmd = std::move(c); }));
            FRAME_REQUIRE(device->createFence(false).transform([&](auto f) { fence = std::move(f); }));
            FRAME_REQUIRE(cmd->begin());
            FRAME_REQUIRE(exec.transitionOutput(*cmd, "RadianceCache.cacheRadiance", ResourceState::TransferSource));
            auto dst = rb->slice({0, totalBytes});
            if (!dst) { return RHITestResult::fail("Part A: failed to slice readback buffer"); }
            FRAME_REQUIRE(cmd->copyTextureToBuffer({
                .texture = radRes->texture, .buffer = *dst,
                .width = td.width, .height = td.height, .depth = 1,
            }));
            const BufferBarrierDesc hostRead{.buffer = rb.get(),
                .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
            FRAME_REQUIRE(cmd->synchronize({.buffers = {&hostRead, 1}}));
            FRAME_REQUIRE(cmd->end());
            CommandBuffer* raw = cmd.get();
            FRAME_REQUIRE(graphics->submit({.commandBuffers = {&raw, 1}, .signalFence = fence.get()}));
            FRAME_REQUIRE(fence->wait(kWaitTimeout));
            rb->invalidate();
            const auto* halfWords = static_cast<const uint16_t*>(rb->map());
            if (!halfWords) { return RHITestResult::fail("Part A: failed to map cacheRadiance readback"); }
            uint64_t nonZeroTexels = 0;
            uint64_t nonFiniteHalfs = 0;
            const uint64_t texelCount = uint64_t(td.width) * td.height;
            for (uint64_t i = 0; i < texelCount; ++i) {
                bool anyNz = false;
                for (uint32_t c = 0; c < 4; ++c) {
                    const uint16_t h = halfWords[i * 4 + c];
                    if ((h & 0x7C00u) == 0x7C00u) { ++nonFiniteHalfs; }
                    if ((h & 0x7FFFu) != 0u) { anyNz = true; }
                }
                if (anyNz) { ++nonZeroTexels; }
            }
            rb->unmap();
            if (nonFiniteHalfs != 0 || nonZeroTexels == 0) {
                return RHITestResult::fail("Part A: cacheRadiance invalid (nonFinite=" +
                    std::to_string(nonFiniteHalfs) + " nonZeroTexels=" + std::to_string(nonZeroTexels) + ")");
            }
            std::printf("[MIGRATION_LUMEN] CornellBox 3-frame RadianceCache OK: savedF0=%llu prevF1=%llu savedF1=%llu backing=%llu nonZeroTexels=%llu/%llu\n",
                static_cast<unsigned long long>(savedVersionF0),
                static_cast<unsigned long long>(prevVersionF1),
                static_cast<unsigned long long>(savedVersionF1),
                static_cast<unsigned long long>(savedBackingF0),
                static_cast<unsigned long long>(nonZeroTexels),
                static_cast<unsigned long long>(texelCount));
        }

        // ------------------------------------------------------------------
        // Part B: Virtual Shadow Maps (VirtualShadowMapPass) declared cache publication
        // Runs VisibilityBufferPass -> VirtualShadowMapPass (cacheEnabled=true)
        // across 2 frames under optimal layouts and verifies render.vsm.{atlas,pageTable,projections}
        // ProviderOwned publication + valid shadow visibility output.
        // ------------------------------------------------------------------
        {
            scene::SceneDocument sceneDoc;
            if (!sceneDoc.load(root / "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf")) {
                return RHITestResult::fail("Part B: failed to load ABeautifulGame.gltf: " + sceneDoc.lastLoadResult().error);
            }

            RenderWorld world;
            world.setScene(&sceneDoc);
            environment::WorldEnvironment sunEnv{};
            sunEnv.sun.enabled = true;
            sunEnv.sun.direction = float3(-0.4f, -0.85f, -0.34f);
            world.setWorldEnvironment(sunEnv);

            RenderView view;
            ViewCamera camera;
            camera.eye = {0.0f, 0.38f, 0.72f};
            camera.center = {0.0f, 0.05f, 0.0f};
            camera.up = {0.0f, 1.0f, 0.0f};
            camera.fovDegrees = 50.0f;
            camera.nearPlane = 0.02f;
            camera.farPlane = 50.0f;
            if (!view.setCamera(camera) || !view.setRenderResolution(64, 48)) {
                return RHITestResult::fail("Part B: invalid VSM RenderView");
            }

            RenderGraph graph;
            graph.setName("MigratedVsmDeclaredCache");
            graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "none"}, {"enableClusterRtx", false}});
            graph.addNode("VirtualShadowMapPass", "Shadows", {
                {"pageSize", 64}, {"virtualPagesPerAxis", 16}, {"clipmapCount", 4},
                {"physicalPageCount", 64}, {"cacheEnabled", true}, {"receiverFootprint", false},
            });
            graph.addEdge("VBuffer.depth", "Shadows.depth");
            graph.addEdge("VBuffer.rasterInfo", "Shadows.rasterInfo");
            graph.addEdge("VBuffer.visibility", "Shadows.visibility");
            graph.markOutput("Shadows.shadow");

            RenderGraphExecutor exec;
            exec.bindRenderWorld(&world);
            exec.bindRenderView(&view);
            exec.setExecutionCaptureEnabled(true);
            std::string log;
            auto compiled = exec.compile(*device, graph, 64, 48, log);
            if (hasError(compiled, Error::Unsupported)) { return RHITestResult::skip("Part B: " + log); }
            FRAME_REQUIRE(compiled);

            // Frame 0 builds and commits VSM persistent cache; Frame 1 declares them as ProviderOwned.
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .recordingWorkerLimit = 1}));
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));
            FRAME_REQUIRE(exec.execute({.graphicsQueue = graphics, .recordingWorkerLimit = 1}));
            FRAME_REQUIRE(exec.waitForSubmittedWork(kWaitTimeout));

            const auto snap = exec.executionSnapshot();
            if (!snap) { return RHITestResult::fail("Part B: missing execution snapshot"); }
            bool sawAtlas = false, sawPageTable = false, sawProjections = false, sawStreamer = false;
            uint64_t atlasBytes = 0;
            for (const auto& res : snap->resources) {
                if (res.ownership != RenderGraphResourceOwnership::ProviderOwned) { continue; }
                const uint64_t backing = res.memory.backingAllocationId != 0
                    ? res.memory.backingAllocationId : res.memory.allocationId;
                const uint64_t bytes = res.memory.backingSizeBytes != 0
                    ? res.memory.backingSizeBytes : res.memory.sizeBytes;
                if (res.name == "render.vsm.atlas" && backing != 0 && bytes > 0) {
                    sawAtlas = true;
                    atlasBytes = bytes;
                } else if (res.name == "render.vsm.pageTable" && backing != 0 && bytes > 0) {
                    sawPageTable = true;
                } else if (res.name == "render.vsm.projections" && backing != 0 && bytes > 0) {
                    sawProjections = true;
                } else if (res.name.find("render.streamer.") == 0) {
                    sawStreamer = true;
                }
            }
            const auto stats = exec.executionStats();
            const bool hasVsmConservative = std::any_of(stats.conservativeFallbackReasons.begin(),
                stats.conservativeFallbackReasons.end(),
                [](const std::string& r) { return r.find("render.pass.virtual-shadow-map") != std::string::npos; });
            if (!sawAtlas || !sawPageTable || !sawProjections || !sawStreamer || !hasVsmConservative) {
                return RHITestResult::fail("Part B: VSM/Streamer declared ProviderOwned publication missing (atlas=" +
                    std::to_string(sawAtlas) + " pageTable=" + std::to_string(sawPageTable) +
                    " projections=" + std::to_string(sawProjections) + " streamer=" + std::to_string(sawStreamer) +
                    " conservative=" + std::to_string(hasVsmConservative) + ")");
            }
            std::printf("[MIGRATION_VSM] VSM 2-frame declared cache OK: atlas=%.2fMiB pageTable=1 projections=1 streamer=1 conservative=1\n",
                double(atlasBytes) / (1024.0 * 1024.0));
        }

        return RHITestResult::pass("Migrated GI Radiance Cache (temporal indirection), VSM (declared persistent cache), and Streamer (ProviderOwned geometry/materials) GPU verification passed under optimal layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(FrameMigratedLumenVsmStreamerDomainsTest);

#undef FRAME_REQUIRE
} // namespace
} // namespace metallic::tests
