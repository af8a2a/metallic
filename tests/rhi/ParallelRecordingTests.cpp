#include "RhiTest.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
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
    render::TimestampQueryPool* progress = nullptr;
    std::atomic<bool> gpuProgressObserved = false;
};
RecordingProbe* recordingProbe = nullptr;

class RecordingProbePass final : public render::RenderGraphPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return properties().value("pipeline", false); }
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
        if (properties().value("prepareThrow", false)) { throw std::runtime_error("Late preparation failure"); }
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
        if (probe.progress && index == 2) {
            // The later CPU batch cannot finish until the earlier copy really
            // executes on the GPU. A joined-only implementation times out here.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            render::TimestampQueryResult progress{};
            do {
                result = probe.progress->readResults(0, 1, &progress);
                if (!result) { return result; }
                if (progress.available) { break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (std::chrono::steady_clock::now() < deadline);
            if (!progress.available) { return render::makeError(render::Error::Failure); }
            probe.gpuProgressObserved = true;
        }
        if (properties().value("fail", false)) { return render::makeError(render::Error::Failure); }
        if (properties().value("throw", false)) { throw std::runtime_error("Late recording failure"); }
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
        if (result && probe.progress && index == 0) {
            result = commands.writeTimestamp(*probe.progress, 0, render::PipelineStageBits::BottomOfPipe);
        }
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
            executor.setExecutionCaptureEnabled(true);
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
                const auto capture = executor.executionSnapshot();
                RECORD_CHECK(capture && !capture->success && capture->status == render::RenderGraphExecutionSnapshotStatus::Failed);
                RECORD_CHECK(std::none_of(capture->segments.begin(), capture->segments.end(),
                    [](const auto& segment) { return segment.accepted; }));
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
            const auto recorded = executor.executionSnapshot();
            RECORD_CHECK(recorded && recorded->success && recorded->passes.size() == 5);
            RECORD_CHECK(std::all_of(recorded->passes.begin(), recorded->passes.end(),
                [](const auto& pass) { return pass.recorded; }));
            RECORD_CHECK(std::all_of(recorded->segments.begin(), recorded->segments.end(),
                [](const auto& segment) { return segment.recorded && segment.accepted; }));
            RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
            const auto completed = executor.executionSnapshot();
            RECORD_CHECK(std::all_of(completed->segments.begin(), completed->segments.end(),
                [](const auto& segment) { return segment.completed; }));
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

class PipelinedBatchTest final : public RhiTest {
public:
    PipelinedBatchTest() { type = RhiTestType::Command; name = "pipelined_batch_receipt_and_frame_completion"; }
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
        RECORD_REQUIRE(frame.begin(0, UINT64_MAX, render::FrameSubmissionMode::Pipelined));
        render::CommandBuffer* a = nullptr;
        render::CommandBuffer* b = nullptr;
        RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { a = value; }));
        RECORD_REQUIRE(second.prepare(frame).transform([&](auto value) { b = value; }));
        auto acceptedOwner = std::make_shared<int>(1);
        std::weak_ptr<int> acceptedWeak = acceptedOwner;
        RECORD_REQUIRE(a->retainResource(std::move(acceptedOwner)));
        RECORD_REQUIRE(first.record([&] { return a->end(); }));
        RECORD_CHECK(!tracker.submit({.commandBuffers = &a, .commandBufferCount = 1}, frame) && !frame.hasAcceptedWork());
        render::RecordedBatch batch;
        std::array<render::CommandBuffer*, 2> duplicates{a, a};
        RECORD_CHECK(!batch.seal(frame, duplicates));
        RECORD_REQUIRE(batch.seal(frame, {&a, 1}));
        RECORD_CHECK(batch.valid() && frame.recording() && !frame.hasAcceptedWork());
        RECORD_CHECK(!a->begin(&frame) && !frame.sealRecording());
        RECORD_CHECK(!context.graphicsQueue.submit({.commandBuffers = &a, .commandBufferCount = 1}));
        render::SubmissionReceipt receipt;
        RECORD_CHECK(!tracker.submitBatch(batch, {.commandBuffers = &a, .commandBufferCount = 1}, frame, receipt));
        RECORD_CHECK(!receipt.accepted() && batch.valid());
        render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
        RECORD_REQUIRE(tracker.submitBatch(batch, {.waitSemaphores = &wait, .waitSemaphoreCount = 1}, frame, receipt));
        const auto prefix = receipt.completion();
        RECORD_CHECK(receipt.accepted() && !prefix.isComplete() && frame.recording() && frame.hasAcceptedWork());
        RECORD_CHECK(!frame.completion().isSubmitted() && frame.completion().value() == 0 && !frame.wait(0));
        RECORD_CHECK(!tracker.submitBatch(batch, {}, frame, receipt) && !receipt.accepted());
        auto tailOwner = std::make_shared<int>(2);
        std::weak_ptr<int> tailWeak = tailOwner;
        RECORD_REQUIRE(b->retainResource(std::move(tailOwner)));
        RECORD_REQUIRE(second.record([&] { return b->end(); }));
        // Admission stays open after native acceptance, even on the same pool.
        render::CommandBuffer* c = nullptr;
        RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { c = value; }));
        RECORD_REQUIRE(first.record([&] { return c->end(); }));
        render::RecordedBatch last;
        RECORD_REQUIRE(last.seal(frame, {&c, 1}));
        RECORD_REQUIRE(frame.sealRecording());
        RECORD_CHECK(!frame.recording() && !first.prepare(frame) && !frame.completion().isSubmitted());
        RECORD_REQUIRE(tracker.submitBatch(last, {}, frame, receipt));
        RECORD_REQUIRE(frame.finishSubmission()); // Cancels unaccepted b only.
        RECORD_CHECK(frame.completion().isSubmitted() && !frame.completion().isComplete());
        RECORD_CHECK(!acceptedWeak.expired() && tailWeak.expired() && !first.reset());
        RECORD_REQUIRE(gate->signal(1));
        RECORD_REQUIRE(frame.wait(kTimeout));
        RECORD_REQUIRE(prefix.wait(0));
        RECORD_REQUIRE(first.reset());
        RECORD_REQUIRE(second.reset());
        RECORD_REQUIRE(frame.reset());
        RECORD_CHECK(acceptedWeak.expired());
        {
            // Independent queue completion cannot release a blocked graphics
            // prefix. A failed tail publishes only the signals actually accepted.
            auto* copyQueue = context.device.getQueue(render::QueueType::Copy);
            if (!copyQueue) { copyQueue = &context.graphicsQueue; }
            render::CommandRecordingContext copyContext;
            render::QueueSubmissionTracker copyTracker;
            std::unique_ptr<render::Semaphore> partialGate;
            RECORD_REQUIRE(context.device.createSemaphore().transform([&](auto value) { partialGate = std::move(value); }));
            QueueDrain partialDrain{context.graphicsQueue, *partialGate};
            RECORD_REQUIRE(copyContext.initialize(context.device, *copyQueue));
            RECORD_REQUIRE(copyTracker.initialize(context.device, *copyQueue));
            RECORD_REQUIRE(frame.begin(1, UINT64_MAX, render::FrameSubmissionMode::Pipelined));
            RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { a = value; }));
            RECORD_REQUIRE(second.prepare(frame).transform([&](auto value) { b = value; }));
            auto owner = std::make_shared<int>(4);
            std::weak_ptr<int> weak = owner;
            RECORD_REQUIRE(a->retainResource(std::move(owner)));
            RECORD_REQUIRE(first.record([&] { return a->end(); }));
            render::RecordedBatch prefixBatch;
            RECORD_REQUIRE(prefixBatch.seal(frame, {&a, 1}));
            wait.semaphore = partialGate.get();
            RECORD_REQUIRE(tracker.submitBatch(prefixBatch, {.waitSemaphores = &wait, .waitSemaphoreCount = 1}, frame, receipt));
            render::CommandBuffer* copy = nullptr;
            RECORD_REQUIRE(copyContext.prepare(frame).transform([&](auto value) { copy = value; }));
            RECORD_REQUIRE(copyContext.record([&] { return copy->end(); }));
            render::RecordedBatch copyBatch;
            RECORD_REQUIRE(copyBatch.seal(frame, {&copy, 1}));
            RECORD_REQUIRE(copyTracker.submitBatch(copyBatch, {}, frame, receipt));
            if (!copyQueue->sameQueue(context.graphicsQueue)) { RECORD_REQUIRE(receipt.completion().wait(kTimeout)); }
            auto rejected = std::make_shared<render::SubmissionTransaction>(nullptr, nullptr);
            RECORD_REQUIRE(b->addSubmissionTransaction(rejected));
            RECORD_REQUIRE(second.record([&] { return b->end(); }));
            render::RecordedBatch tailBatch;
            RECORD_REQUIRE(tailBatch.seal(frame, {&b, 1}));
            rejected->cancel();
            RECORD_CHECK(!tracker.submitBatch(tailBatch, {}, frame, receipt) && !receipt.accepted());
            frame.cancel();
            RECORD_CHECK(frame.completion().isSubmitted() && !frame.completion().isComplete() &&
                frame.completion().value() == 0 && !weak.expired());
            RECORD_REQUIRE(partialGate->signal(1));
            RECORD_REQUIRE(frame.wait(kTimeout));
            RECORD_REQUIRE(first.reset());
            RECORD_REQUIRE(second.reset());
            RECORD_REQUIRE(copyContext.reset());
            RECORD_REQUIRE(frame.reset());
            RECORD_CHECK(weak.expired());
        }
        // Stale batches cannot follow a wrapper move/destruction or a new frame.
        RECORD_REQUIRE(frame.begin(1, UINT64_MAX, render::FrameSubmissionMode::Pipelined));
        RECORD_REQUIRE(first.prepare(frame).transform([&](auto value) { a = value; }));
        RECORD_REQUIRE(first.record([&] { return a->end(); }));
        render::RecordedBatch movedBatch;
        RECORD_REQUIRE(movedBatch.seal(frame, {&a, 1}));
        {
            render::CommandBuffer moved(std::move(*a));
            RECORD_CHECK(!movedBatch.valid() && !tracker.submitBatch(movedBatch, {}, frame, receipt));
        }
        RECORD_CHECK(!movedBatch.valid());
        frame.cancel();
        RECORD_REQUIRE(first.reset());
        RECORD_REQUIRE(frame.begin(2));
        RECORD_CHECK(!tracker.submitBatch(last, {}, frame, receipt));
        frame.cancel();
        return RhiTestResult::pass();
    }
};

class PipelinedGraphTest final : public RhiTest {
public:
    PipelinedGraphTest() { type = RhiTestType::Rendering; name = "pipelined_graph_gpu_progress_and_failure"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 3 || !context.device.capabilities().timestampQueries) {
            return RhiTestResult::skip("three workers and timestamps required");
        }
        render::registerRenderGraphPassType("PipelinedProbePass", "Pipeline regression",
            [] { return std::make_unique<RecordingProbePass>(); });
        // Live GPU/CPU overlap, late Result/exception failures, mixed queues,
        // explicit joined reference, unreviewed-pass fallback, and inline waves.
        for (uint32_t mode = 0; mode < 9; ++mode) {
            RecordingProbe probe;
            recordingProbe = &probe;
            std::unique_ptr<render::TimestampQueryPool> progress;
            RECORD_REQUIRE(context.device.createTimestampQueryPool(context.graphicsQueue, {.queryCount = 1})
                .transform([&](auto value) { progress = std::move(value); }));
            render::RenderFrameContext setupFrame;
            render::CommandRecordingContext setup;
            render::QueueSubmissionTracker tracker;
            RECORD_REQUIRE(setup.initialize(context.device, context.graphicsQueue));
            RECORD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
            RECORD_REQUIRE(setupFrame.begin(0));
            render::CommandBuffer* commands = nullptr;
            RECORD_REQUIRE(setup.prepare(setupFrame).transform([&](auto value) { commands = value; }));
            RECORD_REQUIRE(setup.record([&]() -> render::Result<> {
                auto result = commands->resetTimestampQueries(*progress, 0, 1);
                return result ? commands->end() : result;
            }));
            RECORD_REQUIRE(tracker.submit({.commandBuffers = &commands, .commandBufferCount = 1}, setupFrame));
            RECORD_REQUIRE(setupFrame.wait(kTimeout));
            if (mode < 4 || mode >= 6) { probe.progress = progress.get(); }
            probe.rendezvous = mode < 3 ? 3 : 0;
            render::RenderGraph graph;
            constexpr std::array workloads{3, 1, 2, 2, 1};
            for (uint32_t i = 0; i < 5; ++i) {
                const auto name = "Probe" + std::to_string(i);
                graph.addNode("PipelinedProbePass", name, {{"index", i}, {"work", workloads[i]},
                    {"pipeline", mode != 5}, {"fail", mode == 1 && i == 2}, {"throw", (mode == 2 || mode == 8) && i == 2},
                    {"prepareThrow", mode == 7 && i == 4}, {"serial", mode == 8 && i == 2},
                    {"copyQueue", mode == 3 && i % 2 == 1}});
                if (i) { graph.addEdge("Probe" + std::to_string(i - 1) + ".data", name + ".source"); }
            }
            graph.markOutput("Probe4.data");
            render::RenderGraphExecutor executor;
            std::string log;
            RECORD_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
            auto result = executor.execute({.graphicsQueue = &context.graphicsQueue,
                .copyQueue = mode == 3 ? context.device.getQueue(render::QueueType::Copy) : nullptr,
                .recordingWorkerLimit = mode == 6 ? 1u : mode == 7 ? 2u : 4u, .recordingBatchWorkload = 4,
                .submissionMode = mode == 4 ? render::FrameSubmissionMode::Joined : render::FrameSubmissionMode::Pipelined});
            const auto& stats = executor.executionStats();
            RECORD_CHECK(stats.pipelinedSubmission == (mode != 4 && mode != 5));
            RECORD_CHECK(!probe.wrongThread && executor.lastSubmittedCompletion().isSubmitted());
            if (probe.progress) { RECORD_CHECK(probe.gpuProgressObserved && probe.prematureSubmit); }
            if (mode < 3) { RECORD_CHECK(stats.batchesSubmittedWhileRecording >= 1); }
            if (mode == 1 || mode == 2 || mode >= 7) {
                RECORD_CHECK(!result && !executor.compiled());
                const std::vector<int> expected = mode == 7 ? std::vector<int>{1, 2, 3, 4} :
                    mode == 8 ? std::vector<int>{1, 2, -3} : std::vector<int>{1, 2, -5, -3};
                RECORD_CHECK(probe.events == expected);
                RECORD_CHECK((mode == 7 || probe.owners[2].expired()) && probe.owners[4].expired());
                RECORD_CHECK(!probe.owners[0].expired() && !probe.owners[1].expired());
            } else {
                RECORD_REQUIRE(result);
                RECORD_CHECK((probe.events == std::vector<int>{1, 2, 3, 4, 5}));
                if (mode == 4 || mode == 5) { RECORD_CHECK(!probe.prematureSubmit); }
                if (mode == 5) { RECORD_CHECK(stats.submissionBlockingPasses.size() == 5); }
            }
            RECORD_REQUIRE(executor.waitForSubmittedWork(kTimeout));
            if (result) {
                auto* output = executor.outputResource("Probe4.data")->buffer;
                output->invalidate();
                const auto* mapped = output->map();
                RECORD_CHECK(mapped);
                std::array<uint32_t, 4> words{};
                std::memcpy(words.data(), mapped, sizeof(words));
                output->unmap();
                RECORD_CHECK((words == std::array<uint32_t, 4>{17, 23, 42, 99}));
            }
            for (auto& owner : probe.owners) { RECORD_CHECK(owner.expired()); }
        }
        recordingProbe = nullptr;
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(PipelinedBatchTest);
METALLIC_REGISTER_RHI_TEST(PipelinedGraphTest);

METALLIC_REGISTER_RHI_TEST(RecordingContextTest);
METALLIC_REGISTER_RHI_TEST(RecordingGraphTest);
METALLIC_REGISTER_RHI_TEST(RecordingSlotsTest);
METALLIC_REGISTER_RHI_TEST(RecordingBuiltinPixelsTest);

} // namespace
} // namespace metallic::tests
