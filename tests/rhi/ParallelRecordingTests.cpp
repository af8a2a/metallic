#include "RhiTest.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Task/TaskSystem.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace metallic::tests {
namespace {

#define RECORD_REQUIRE(expression) do { \
    const render::Result<> checked = (expression); \
    if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(checked)); } \
} while (false)
#define RECORD_CHECK(condition) do { \
    if (!(condition)) { return RhiTestResult::fail(#condition); } \
} while (false)

constexpr uint64_t kTimeout = 5'000'000'000ull;

struct QueueDrain {
    render::Queue& queue;
    render::Semaphore& gate;
    ~QueueDrain()
    {
        if (gate.currentValue() < 1) { (void)gate.signal(1); }
        (void)queue.waitIdle();
    }
};

class RecordingContextTest final : public RhiTest {
public:
    RecordingContextTest() { type = RhiTestType::Command; name = "parallel_recording_context_lifetime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderFrameContext frame;
        render::CommandRecordingContext first, second;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::Semaphore> gate;
        RECORD_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        QueueDrain drain{context.graphicsQueue, *gate};
        RECORD_REQUIRE(first.initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(second.initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(frame.begin(0));
        render::CommandBuffer* a = nullptr;
        render::CommandBuffer* b = nullptr;
        RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { a = value; }));
        RECORD_REQUIRE(second.prepare(frame).transform([&](auto value) { b = value; }));
        auto accepted = std::make_shared<int>(1), cancelled = std::make_shared<int>(2);
        std::weak_ptr<int> acceptedWeak = accepted, cancelledWeak = cancelled;
        RECORD_REQUIRE(b->retainResource(std::move(cancelled)));
        RECORD_REQUIRE(b->end());
        std::mutex mutex;
        std::condition_variable cv;
        bool entered = false, release = false;
        render::Result<> workerResult;
        std::jthread worker([&] {
            workerResult = first.record([&]() -> render::Result<> {
                auto retained = a->retainResource(std::move(accepted));
                if (first.record([] { return render::Result<>{}; })) { retained = render::makeError(render::Error::Failure); }
                {
                    std::unique_lock lock(mutex);
                    entered = true;
                    cv.notify_all();
                    cv.wait(lock, [&] { return release; });
                }
                return retained ? a->end() : retained;
            });
        });
        {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return entered; });
        }
        const bool exclusive = !first.record([] { return render::Result<>{}; }) && !first.prepare(frame) && !first.reset();
        const bool rejectedEarlySubmit = !tracker.submit({.commandBuffers = &b, .commandBufferCount = 1}, frame);
        {
            std::lock_guard lock(mutex);
            release = true;
        }
        cv.notify_all();
        worker.join();
        RECORD_CHECK(exclusive && rejectedEarlySubmit);
        RECORD_REQUIRE(workerResult);
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        render::GpuCompletionPoint prefix;
        RECORD_REQUIRE(tracker.submitSegment({.waitSemaphores = &wait, .waitSemaphoreCount = 1,
            .commandBuffers = &a, .commandBufferCount = 1}, frame, prefix));
        // An invalid tail after an accepted prefix must keep only accepted owners.
        RECORD_CHECK(!tracker.submitSegment({.commandBuffers = &a, .commandBufferCount = 1}, frame, prefix));
        frame.cancel();
        RECORD_CHECK(frame.completion().isSubmitted() && !frame.completion().isComplete());
        RECORD_CHECK(!acceptedWeak.expired() && cancelledWeak.expired() && !first.reset());
        RECORD_REQUIRE(gate->signal(1));
        RECORD_REQUIRE(frame.wait(kTimeout));
        RECORD_REQUIRE(first.reset());
        RECORD_REQUIRE(second.reset());
        RECORD_CHECK(!acceptedWeak.expired());
        RECORD_REQUIRE(frame.reset());
        RECORD_CHECK(acceptedWeak.expired());
        // Reuse the same pool after completion, and cancel a complete recording.
        RECORD_REQUIRE(frame.begin(1));
        RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { a = value; }));
        auto token = std::make_shared<int>(3);
        std::weak_ptr<int> tokenWeak = token;
        RECORD_REQUIRE(first.record([&]() -> render::Result<> {
            auto result = a->retainResource(std::move(token));
            return result ? a->end() : result;
        }));
        frame.cancel();
        RECORD_CHECK(tokenWeak.expired());
        RECORD_REQUIRE(first.reset());
        // Destruction while a prefix is accepted but the frame is not sealed
        // must wait for that prefix without implicitly publishing the frame.
        RECORD_REQUIRE(frame.begin(2));
        auto unsealed = std::make_unique<render::CommandRecordingContext>();
        RECORD_REQUIRE(unsealed->initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(unsealed->prepare(frame).transform([&](auto value) { a = value; }));
        RECORD_REQUIRE(unsealed->record([&] { return a->end(); }));
        wait.value = 2;
        RECORD_REQUIRE(tracker.submitSegment({.waitSemaphores = &wait, .waitSemaphoreCount = 1,
            .commandBuffers = &a, .commandBufferCount = 1}, frame, prefix));
        std::jthread releaseGate([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            (void)gate->signal(2);
        });
        unsealed.reset();
        RECORD_CHECK(prefix.isComplete() && !frame.completion().isSubmitted());
        frame.cancel();
        return RhiTestResult::pass();
    }
};

struct RecordingProbe {
    std::thread::id coordinator = std::this_thread::get_id();
    std::mutex mutex;
    std::condition_variable cv;
    uint32_t rendezvous = 0;
    uint32_t arrived = 0;
    std::atomic<uint32_t> finished = 0;
    std::atomic<bool> wrongThread = false;
    std::atomic<bool> prematureSubmit = false;
    std::array<std::weak_ptr<void>, 5> owners;
    std::array<bool, 5> recorded{};
    std::vector<int> events;
    uint32_t prepareRendezvous = 0;
    uint32_t prepareArrived = 0;
    std::atomic<uint32_t> prepareFinished = 0;
    bool preparePublished = false;
};
RecordingProbe* recordingProbe = nullptr;

class RecordingProbePass final : public render::RenderGraphPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    render::CpuRecordingPolicy cpuRecordingPolicy() const override
    {
        return properties().value("serial", false) ? render::CpuRecordingPolicy::Serial : render::CpuRecordingPolicy::ParallelJoined;
    }
    uint32_t recordingWorkload() const override { return properties().value("work", 1u); }
    render::QueueType queueType() const override
    {
        return properties().value("copyQueue", false) ? render::QueueType::Copy : render::QueueType::Graphics;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (properties().value("index", 0u) != 0) { reflection.addBufferInput("source").buffer(16).transferRead(); }
        auto& output = reflection.addBufferOutput("data").buffer(16).transferWrite();
        output.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        device_ = context.device;
        return {};
    }
    render::Result<> prepareExecution(render::RenderGraphExecutionContext& context) override
    {
        if (std::this_thread::get_id() != recordingProbe->coordinator) { recordingProbe->wrongThread = true; }
        if (!properties().value("prepareJobs", false)) { return {}; }
        auto& probe = *recordingProbe;
        const uint32_t failure = properties().value("prepareFailure", 0u);
        std::array<uint32_t, 5> outputs{};
        const std::array weights{3u, 1u, 2u, 2u, 1u};
        std::vector<render::RenderPreparationTask> jobs;
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            jobs.push_back({.name = "Prepare " + std::to_string(i), .workload = weights[i], .prepare = [&, i]() -> render::Result<> {
                if (probe.prepareRendezvous && (i == 0 || i == 2 || i == 4)) {
                    std::unique_lock lock(probe.mutex);
                    ++probe.prepareArrived;
                    probe.cv.notify_all();
                    if (!probe.cv.wait_for(lock, std::chrono::seconds(5), [&] { return probe.prepareArrived == probe.prepareRendezvous; })) {
                        return render::makeError(render::Error::Failure);
                    }
                }
                outputs[i] = i + 100;
                ++probe.prepareFinished;
                if (i == 0 && failure == 1) { return render::makeError(render::Error::InvalidArgument); }
                if (i == 0 && failure == 2) { throw std::runtime_error("Preparation failure"); }
                return {};
            }});
        }
        auto result = context.prepareJoined(jobs);
        if (!result) { return result; }
        if (probe.prepareFinished != 5 || outputs != std::array<uint32_t, 5>{100, 101, 102, 103, 104} ||
            std::this_thread::get_id() != probe.coordinator) { return render::makeError(render::Error::Failure); }
        probe.preparePublished = true;
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto& probe = *recordingProbe;
        const uint32_t index = properties().value("index", 0u);
        auto scope = context.profileScope("Record transfer");
        if (probe.rendezvous && (index == 0 || index == 2 || index == 4)) {
            std::unique_lock lock(probe.mutex);
            ++probe.arrived;
            probe.cv.notify_all();
            if (!probe.cv.wait_for(lock, std::chrono::seconds(5), [&] { return probe.arrived >= probe.rendezvous; })) {
                return render::makeError(render::Error::Failure);
            }
        }
        auto token = std::make_shared<int>(int(index));
        probe.owners[index] = token;
        auto& commands = context.commandBuffer();
        auto result = commands.retainResource(std::move(token));
        if (!result) { return result; }
        result = commands.addSubmissionTransaction(std::make_shared<render::SubmissionTransaction>(
            [&probe, index] {
                if (probe.finished != 5) { probe.prematureSubmit = true; }
                if (std::this_thread::get_id() != probe.coordinator) { probe.wrongThread = true; }
                probe.events.push_back(int(index + 1));
            }, [&probe, index] { probe.events.push_back(-int(index + 1)); }));
        if (!result) { return result; }
        probe.recorded[index] = true;
        if (properties().value("fail", false)) { return render::makeError(render::Error::Failure); }
        auto* target = context.outputBuffer("data").buffer();
        render::BufferSlice source;
        if (index == 0) {
            std::unique_ptr<render::Buffer> upload;
            result = device_->createBuffer({.size = 16, .usage = render::BufferUsageBits::TransferSource,
                .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto value) { upload = std::move(value); });
            if (!result) { return result; }
            const std::array<uint32_t, 4> words{17, 23, 42, 99};
            auto* mapped = upload->map();
            if (!mapped) { return render::makeError(render::Error::Failure); }
            std::memcpy(mapped, words.data(), sizeof(words));
            upload->flush();
            upload->unmap();
            result = upload->slice(0, 16).transform([&](auto value) { source = std::move(value); });
        } else {
            result = context.inputBuffer("source").buffer()->slice(0, 16).transform([&](auto value) { source = std::move(value); });
        }
        if (!result) { return result; }
        render::BufferSlice destination;
        result = target->slice(0, 16).transform([&](auto value) { destination = std::move(value); });
        if (result) { result = commands.copyBuffer(source, destination); }
        ++probe.finished;
        return result;
    }
private:
    render::Device* device_ = nullptr;
};

class RecordingGraphTest final : public RhiTest {
public:
    RecordingGraphTest() { type = RhiTestType::Rendering; name = "parallel_recording_workload_and_order"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 3) { return RhiTestResult::skip("three TaskSystem workers required"); }
        static const bool registered = [] {
            render::registerRenderGraphPassType("RecordingProbePass", "Parallel recording regression",
                [] { return std::make_unique<RecordingProbePass>(); });
            return true;
        }();
        (void)registered;
        // Serial, workload batches, cancellation, mixed queues, a serial island,
        // and invocation inside a TaskSystem callback (must not wait recursively).
        for (uint32_t mode = 0; mode < 6; ++mode) {
            RecordingProbe probe;
            recordingProbe = &probe;
            probe.rendezvous = mode == 1 || mode == 2 ? 3 : 0;
            render::RenderGraph graph;
            constexpr std::array workloads{3, 1, 2, 2, 1};
            for (uint32_t i = 0; i < 5; ++i) {
                const auto name = "Probe" + std::to_string(i);
                graph.addNode("RecordingProbePass", name, {{"index", i}, {"work", workloads[i]},
                    {"fail", mode == 2 && i == 2}, {"serial", mode == 4 && i == 2}, {"copyQueue", mode == 3 && i % 2 == 1}});
                if (i) { graph.addEdge("Probe" + std::to_string(i - 1) + ".data", name + ".source"); }
            }
            graph.markOutput("Probe4.data");
            render::RenderGraphExecutor executor;
            std::string log;
            RECORD_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
            render::RenderGraphSubmitDesc desc{.graphicsQueue = &context.graphicsQueue,
                .copyQueue = mode == 3 ? context.device.getQueue(render::QueueType::Copy) : nullptr,
                .recordingWorkerLimit = mode == 0 ? 1u : 4u, .recordingBatchWorkload = 4};
            render::Result<> result;
            if (mode == 5) {
                task::TaskGraph parent("Nested renderer recording");
                parent.addTask({.name = "Execute graph"}, [&] {
                    probe.coordinator = std::this_thread::get_id();
                    result = executor.execute(desc);
                });
                auto run = tasks->submit(std::move(parent));
                RECORD_CHECK(run && run->wait());
            } else { result = executor.execute(desc); }
            if (mode == 2) {
                RECORD_CHECK(!result && !executor.compiled() && !executor.lastSubmittedCompletion().valid());
                std::vector<int> cancelled;
                for (int i = 4; i >= 0; --i) { if (probe.recorded[i]) { cancelled.push_back(-(i + 1)); } }
                RECORD_CHECK(probe.events == cancelled && !cancelled.empty());
                for (const auto& owner : probe.owners) { RECORD_CHECK(owner.expired()); }
                continue;
            }
            RECORD_REQUIRE(result);
            RECORD_CHECK(!probe.wrongThread && !probe.prematureSubmit);
            RECORD_CHECK((probe.events == std::vector<int>{1, 2, 3, 4, 5}));
            const auto& stats = executor.executionStats();
            if (mode == 1) {
                RECORD_CHECK(probe.arrived == 3 && stats.recordingTaskCount == 3 && stats.recordingBatchCount == 3);
                RECORD_CHECK(stats.parallelRecordedPassCount == 5);
            }
            if (mode == 0 || mode == 5) { RECORD_CHECK(stats.recordingTaskCount == 0); }
            RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
            auto* output = executor.outputResource("Probe4.data")->buffer;
            output->invalidate();
            auto* mapped = output->map();
            RECORD_CHECK(mapped != nullptr);
            std::array<uint32_t, 4> actual;
            std::memcpy(actual.data(), mapped, sizeof(actual));
            output->unmap();
            RECORD_CHECK((actual == std::array<uint32_t, 4>{17, 23, 42, 99}));
            for (const auto& owner : probe.owners) { RECORD_CHECK(owner.expired()); }
            std::vector<render::RenderGraphExecutionStats> timings;
            RECORD_REQUIRE(executor.collectCompletedGpuExecutionStats(timings));
            if (context.device.capabilities().timestampQueries) {
                RECORD_CHECK(timings.size() == 1 && timings[0].nodes.size() == 5);
                for (const auto& node : timings[0].nodes) {
                    RECORD_CHECK(node.gpuTimingAvailable && !node.sections.empty() && node.sections[0].gpuTimingAvailable);
                }
            }
        }
        recordingProbe = nullptr;
        return RhiTestResult::pass();
    }
};

class RecordingSlotsTest final : public RhiTest {
public:
    RecordingSlotsTest() { type = RhiTestType::Rendering; name = "parallel_recording_two_slots"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 3) { return RhiTestResult::skip("three TaskSystem workers required"); }
        render::registerRenderGraphPassType("RecordingSlotProbePass", "Parallel recording slot regression",
            [] { return std::make_unique<RecordingProbePass>(); });
        RecordingProbe probe;
        recordingProbe = &probe;
        probe.rendezvous = 3;
        render::RenderFrameContext blockerFrame;
        render::CommandRecordingContext blocker;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::Semaphore> gate;
        render::RenderGraphExecutor executor;
        render::RenderGraph graph;
        constexpr std::array workloads{3, 1, 2, 2, 1};
        for (uint32_t i = 0; i < 5; ++i) {
            const auto name = "Probe" + std::to_string(i);
            graph.addNode("RecordingSlotProbePass", name, {{"index", i}, {"work", workloads[i]}});
            if (i) { graph.addEdge("Probe" + std::to_string(i - 1) + ".data", name + ".source"); }
        }
        graph.markOutput("Probe4.data");
        std::string log;
        RECORD_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
        RECORD_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        QueueDrain drain{context.graphicsQueue, *gate};
        // Bound an accidental CPU/GPU serialization regression without allowing
        // it to pass: successful overlap must leave the gate at zero.
        std::mutex mutex;
        std::condition_variable_any cv;
        std::jthread watchdog([&](std::stop_token stop) {
            std::unique_lock lock(mutex);
            cv.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
            if (!stop.stop_requested() && gate->currentValue() == 0) { (void)gate->signal(1); }
        });
        RECORD_REQUIRE(blocker.initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        RECORD_REQUIRE(blockerFrame.begin(0));
        render::CommandBuffer* commands = nullptr;
        RECORD_REQUIRE(blocker.prepare(blockerFrame).transform([&](auto value) { commands = value; }));
        RECORD_REQUIRE(blocker.record([&] { return commands->end(); }));
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        RECORD_REQUIRE(tracker.submit({.waitSemaphores = &wait, .waitSemaphoreCount = 1,
            .commandBuffers = &commands, .commandBufferCount = 1}, blockerFrame));
        render::RenderGraphSubmitDesc desc{.graphicsQueue = &context.graphicsQueue, .slotWaitTimeoutNanoseconds = 0,
            .recordingWorkerLimit = 4, .recordingBatchWorkload = 4};
        RECORD_REQUIRE(executor.execute(desc));
        const auto first = executor.lastSubmittedCompletion();
        const auto firstOwners = probe.owners;
        probe.finished = 0;
        probe.arrived = 0;
        RECORD_REQUIRE(executor.execute(desc));
        const auto second = executor.lastSubmittedCompletion();
        RECORD_CHECK(gate->currentValue() == 0 && !first.isComplete() && !second.isComplete());
        RECORD_CHECK(!executor.execute(desc) && executor.compiled());
        for (const auto& owner : firstOwners) { RECORD_CHECK(!owner.expired()); }
        for (const auto& owner : probe.owners) { RECORD_CHECK(!owner.expired()); }
        RECORD_CHECK(!probe.wrongThread && !probe.prematureSubmit);
        RECORD_REQUIRE(gate->signal(1));
        RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
        for (const auto& owner : firstOwners) { RECORD_CHECK(owner.expired()); }
        for (const auto& owner : probe.owners) { RECORD_CHECK(owner.expired()); }
        std::vector<render::RenderGraphExecutionStats> timings;
        RECORD_REQUIRE(executor.collectCompletedGpuExecutionStats(timings));
        if (context.device.capabilities().timestampQueries) { RECORD_CHECK(timings.size() == 2); }
        // The same lane pools must also work in the next frame generation.
        probe.finished = 0;
        probe.arrived = 0;
        RECORD_REQUIRE(executor.execute(desc));
        RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
        RECORD_CHECK(probe.events.size() == 15 && !probe.prematureSubmit);
        recordingProbe = nullptr;
        return RhiTestResult::pass();
    }
};

class RecordingBuiltinPixelsTest final : public RhiTest {
public:
    RecordingBuiltinPixelsTest() { type = RhiTestType::Rendering; name = "parallel_recording_builtin_pixels"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 2) { return RhiTestResult::skip("two TaskSystem workers required"); }
        for (uint32_t workers : {1u, 4u}) {
            render::RenderGraph graph;
            graph.addNode("ClearColorPass", "Red", {{"color", {1.0f, 0.0f, 0.0f, 1.0f}}});
            graph.addNode("ClearColorPass", "Green", {{"color", {0.0f, 1.0f, 0.0f, 1.0f}}});
            graph.addNode("CopyColorPass", "CopyRed");
            graph.addNode("CopyColorPass", "CopyGreen");
            graph.addEdge("Red.color", "CopyRed.source");
            graph.addEdge("Green.color", "CopyGreen.source");
            graph.markOutput("CopyRed.color");
            graph.markOutput("CopyGreen.color");
            render::RenderGraphExecutor executor;
            std::string log;
            RECORD_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
            RECORD_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue,
                .copyQueue = context.device.getQueue(render::QueueType::Copy),
                .recordingWorkerLimit = workers, .recordingBatchWorkload = 1}));
            if (workers == 4) {
                RECORD_CHECK(executor.executionStats().parallelRecordedPassCount >= 2);
                RECORD_CHECK(executor.executionStats().recordingBatchCount == 4);
            }
            RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
            std::unique_ptr<render::Buffer> readback;
            RECORD_REQUIRE(context.device.createBuffer({.size = 128, .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto value) { readback = std::move(value); }));
            render::RenderFrameContext frame;
            render::CommandRecordingContext recording;
            render::QueueSubmissionTracker tracker;
            RECORD_REQUIRE(recording.initialize(context.device, context.graphicsQueue));
            RECORD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
            RECORD_REQUIRE(frame.begin(0));
            render::CommandBuffer* commands = nullptr;
            RECORD_REQUIRE(recording.prepare(frame).transform([&](auto value) { commands = value; }));
            RECORD_REQUIRE(recording.record([&]() -> render::Result<> {
                uint32_t offset = 0;
                for (const char* output : {"CopyRed.color", "CopyGreen.color"}) {
                    auto result = executor.transitionOutput(*commands, output, render::ResourceState::TransferSource);
                    if (!result) { return result; }
                    commands->copyTextureToBuffer({.texture = executor.outputResource(output)->texture,
                        .buffer = readback.get(), .bufferOffset = offset, .width = 4, .height = 4});
                    offset += 64;
                }
                return commands->end();
            }));
            RECORD_REQUIRE(tracker.submit({.commandBuffers = &commands, .commandBufferCount = 1}, frame));
            RECORD_REQUIRE(frame.wait(kTimeout));
            readback->invalidate();
            const auto* pixels = static_cast<const uint8_t*>(readback->map());
            RECORD_CHECK(pixels != nullptr);
            bool correct = true;
            for (uint32_t image = 0; image < 2; ++image) {
                for (uint32_t pixel = 0; pixel < 16; ++pixel) {
                    const auto* rgba = pixels + image * 64 + pixel * 4;
                    correct &= rgba[0] == (image == 0 ? 255 : 0) && rgba[1] == (image == 1 ? 255 : 0) &&
                        rgba[2] == 0 && rgba[3] == 255;
                }
            }
            readback->unmap();
            RECORD_CHECK(correct);
        }
        return RhiTestResult::pass();
    }
};

class PreparationGraphTest final : public RhiTest {
public:
    PreparationGraphTest() { type = RhiTestType::Rendering; name = "parallel_preparation_join_failure_and_batching"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* system = task::tryGetTaskSystem();
        if (!system || system->workerCount() < 3) { return RhiTestResult::skip("three preparation workers required"); }
        render::registerRenderGraphPassType("PreparationProbePass", "Pure preparation regression",
            [] { return std::make_unique<RecordingProbePass>(); });
        // Inline, joined batches, Result failure, exception, nested-worker fallback,
        // and a target large enough to coalesce all work into one inline batch.
        for (uint32_t mode = 0; mode < 6; ++mode) {
            RecordingProbe probe;
            recordingProbe = &probe;
            probe.prepareRendezvous = mode >= 1 && mode <= 3 ? 3 : 0;
            render::RenderGraph graph;
            graph.addNode("PreparationProbePass", "Probe", {{"index", 0}, {"serial", true}, {"prepareJobs", true},
                {"prepareFailure", mode == 2 ? 1 : mode == 3 ? 2 : 0}});
            graph.markOutput("Probe.data");
            render::RenderGraphExecutor executor;
            std::string log;
            RECORD_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
            render::RenderGraphSubmitDesc desc{.graphicsQueue = &context.graphicsQueue,
                .recordingWorkerLimit = mode == 0 ? 1u : 3u, .preparationBatchWorkload = mode == 5 ? 32u : 4u};
            render::Result<> result;
            if (mode == 4) {
                task::TaskGraph outer("Nested prepare");
                outer.addTask({.name = "Execute"}, [&] { probe.coordinator = std::this_thread::get_id(); result = executor.execute(desc); });
                auto submitted = system->submit(std::move(outer));
                RECORD_CHECK(submitted && submitted->wait());
            } else { result = executor.execute(desc); }
            RECORD_CHECK(probe.prepareFinished == 5 && !probe.wrongThread);
            if (mode == 2 || mode == 3) {
                RECORD_CHECK(!result && !probe.preparePublished && probe.finished == 0 &&
                    !executor.lastSubmittedCompletion().valid() && !executor.compiled());
            } else {
                RECORD_REQUIRE(result);
                RECORD_CHECK(probe.preparePublished);
                RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
            }
            RECORD_CHECK(executor.executionStats().preparationTaskCount == (mode >= 1 && mode <= 3 ? 3u : 0u));
        }
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(PreparationGraphTest);

METALLIC_REGISTER_RHI_TEST(RecordingContextTest);
METALLIC_REGISTER_RHI_TEST(RecordingGraphTest);
METALLIC_REGISTER_RHI_TEST(RecordingSlotsTest);
METALLIC_REGISTER_RHI_TEST(RecordingBuiltinPixelsTest);

} // namespace
} // namespace metallic::tests
