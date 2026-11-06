#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace metallic::tests {
namespace {

#define JOINED_REQUIRE(expression) do { \
    const render::Result<> checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + toString(checked)); } \
} while (false)
#define JOINED_CHECK(condition) do { \
    if (!(condition)) { return RHITestResult::fail(#condition); } \
} while (false)

constexpr uint64_t kWaitTimeout = 5'000'000'000ull;
constexpr std::array<uint32_t, 4> kWords{17, 23, 42, 99};

struct JoinedProbe {
    render::Semaphore* gate = nullptr;
    render::TimestampQueryPool* progress = nullptr;
};
JoinedProbe* joinedProbe = nullptr;

struct ProbeBinding {
    JoinedProbe* previous = joinedProbe;
    explicit ProbeBinding(JoinedProbe& probe) { joinedProbe = &probe; }
    ~ProbeBinding() { joinedProbe = previous; }
};

// Unblock before destroying the executor or any commands retained by its frame.
struct GateDrain {
    render::Device& device;
    render::Semaphore& gate;
    uint64_t releaseValue = 1;
    ~GateDrain()
    {
        if (gate.currentValue() < releaseValue) { (void)gate.signal(releaseValue); }
        (void)device.waitIdle();
    }
};

render::Result<> copyWords(render::CommandBuffer& commands, render::Buffer& source,
    render::Buffer& destination, uint64_t offset = 0, uint64_t size = sizeof(kWords))
{
    auto input = source.slice({offset, size});
    if (!input) { return std::unexpected(input.error()); }
    auto output = destination.slice({offset, size});
    if (!output) { return std::unexpected(output.error()); }
    return commands.copyBuffer(*input, *output);
}

bool checkWords(render::RenderGraphExecutor& executor, std::string_view outputName, RHITestContext& context)
{
    auto* buffer = executor.outputResource(outputName)->buffer;
    buffer->invalidate();
    auto* mapped = buffer->map();
    if (!mapped) { return false; }
    std::array<uint32_t, 4> actual;
    std::memcpy(actual.data(), mapped, sizeof(actual));
    buffer->unmap();
    bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(actual));
    return actual == kWords;
}

bool observeProgress(render::TimestampQueryPool& pool)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        render::TimestampQueryResult result;
        if (!pool.readResults(0, {&result, 1})) { return false; }
        if (result.available) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

const render::RenderGraphExecutionSegmentSnapshot* passSegment(
    const render::RenderGraphExecutionSnapshot& snapshot, uint32_t passId)
{
    const auto found = std::find_if(snapshot.segments.begin(), snapshot.segments.end(),
        [passId](const auto& segment) { return segment.passId == passId; });
    return found == snapshot.segments.end() ? nullptr : &*found;
}

const render::RenderGraphExecutionBatchSnapshot* segmentBatch(
    const render::RenderGraphExecutionSnapshot& snapshot, uint32_t segmentId)
{
    const auto found = std::find_if(snapshot.batches.begin(), snapshot.batches.end(),
        [segmentId](const auto& batch) {
            return std::find(batch.segmentIds.begin(), batch.segmentIds.end(), segmentId) != batch.segmentIds.end();
        });
    return found == snapshot.batches.end() ? nullptr : &*found;
}

class JoinedSubmissionTransferPass final : public render::RenderGraphPass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::CPURecordingPolicy cpuRecordingPolicy() const override
    {
        return properties().value("serial", false) || properties().value("fork", false)
            ? render::CPURecordingPolicy::Serial : render::CPURecordingPolicy::ParallelJoined;
    }
    render::QueueType queueType() const override
    {
        return properties().value("compute", false) ? render::QueueType::Compute : render::QueueType::Graphics;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (properties().value("copy", false)) { reflection.addBufferInput("source").buffer(sizeof(kWords)).transferRead(); }
        auto& output = reflection.addBufferOutput("data").buffer(sizeof(kWords)).transferWrite();
        output.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        gate_ = properties().value("gated", false) ? joinedProbe->gate : nullptr;
        progress_ = properties().value("progress", false) ? joinedProbe->progress : nullptr;
        if (properties().value("copy", false)) { return {}; }
        auto result = context.device->createBuffer({.size = sizeof(kWords),
            .usage = render::BufferUsageBits::TransferSource, .memoryLocation = render::MemoryLocation::HostUpload,
            .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute})
            .transform([&](auto buffer) { input_ = std::move(buffer); });
        if (!result) { return result; }
        auto* mapped = input_->map();
        if (!mapped) { return render::makeError(render::Error::Failure); }
        std::memcpy(mapped, kWords.data(), sizeof(kWords));
        input_->flush();
        input_->unmap();
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto profile = context.profileScope("Joined transfer");
        auto* output = context.outputBuffer("data").buffer();
        auto* input = properties().value("copy", false) ? context.inputBuffer("source").buffer() : input_.get();
        const auto waitForGate = [&](render::CommandBuffer& commands) -> render::Result<> {
            if (!gate_) { return {}; }
            const render::SemaphoreSubmitDesc wait{.semaphore = gate_, .value = properties().value("gateValue", uint64_t(1))};
            return commands.addDependency({&wait, 1});
        };
        const auto markProgress = [&](render::CommandBuffer& commands) -> render::Result<> {
            return progress_ ? commands.writeTimestamp(*progress_, 0, render::PipelineStageBits::BottomOfPipe)
                : render::Result<>{};
        };
        if (properties().value("fork", false)) {
            return context.parallelCompute([&](render::CommandBuffer& commands) -> render::Result<> {
                auto result = waitForGate(commands);
                return result ? copyWords(commands, *input, *output, 0, 8) : result;
            }, [&](render::CommandBuffer& commands) -> render::Result<> {
                auto result = copyWords(commands, *input, *output, 8, 8);
                return result ? markProgress(commands) : result;
            });
        }
        auto result = waitForGate(context.commandBuffer());
        if (result) { result = copyWords(context.commandBuffer(), *input, *output); }
        return result ? markProgress(context.commandBuffer()) : result;
    }
private:
    std::unique_ptr<render::Buffer> input_;
    render::Semaphore* gate_ = nullptr;
    render::TimestampQueryPool* progress_ = nullptr;
};

void registerJoinedSubmissionPass()
{
    static const bool registered = render::registerRenderGraphPassType("JoinedSubmissionTransferPass",
        "Joined submission batching regression", [] { return std::make_unique<JoinedSubmissionTransferPass>(); });
    (void)registered;
}

render::Result<> prepareProgress(RHITestContext& context, render::TimestampQueryPool& pool)
{
    bench::GPUCommands reset(context.graphicsQueue);
    auto result = reset.initialize(context.device);
    if (result) { result = reset.commands->resetTimestampQueries(pool, 0, 1); }
    return result ? reset.submitAndWait() : result;
}

class JoinedSubmissionChainTest final : public RHITest {
public:
    JoinedSubmissionChainTest() { type = RHITestType::Rendering; name = "joined_submission_serial_chain"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"graph.joined.singleQueue.coalescing", "graph.joined.recordingBatch.independence",
            "graph.pipelined.submissions.preserved"}, bench::Layer::RenderGraph, "core", "sync", {"readback.bin"});
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 2) { return RHITestResult::skip("two recording workers required"); }
        registerJoinedSubmissionPass();
        for (const auto [workers, workload] : {std::pair{1u, 1u}, std::pair{2u, 1u}, std::pair{2u, 3u}}) {
            for (const auto mode : {render::FrameSubmissionMode::Joined, render::FrameSubmissionMode::Pipelined}) {
                render::RenderGraph graph;
                for (uint32_t index = 0; index < 6; ++index) {
                    const auto name = "Transfer" + std::to_string(index);
                    graph.addNode("JoinedSubmissionTransferPass", name, {{"copy", index != 0}, {"serial", index == 3}});
                    if (index) { graph.addEdge("Transfer" + std::to_string(index - 1) + ".data", name + ".source"); }
                }
                graph.markOutput("Transfer5.data");
                render::RenderGraphExecutor executor;
                executor.setExecutionCaptureEnabled(true);
                std::string log;
                JOINED_REQUIRE(executor.compile(context.device, graph, 1, 1, log));
                JOINED_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue,
                    .recordingWorkerLimit = workers, .recordingBatchWorkload = workload,
                    .submissionMode = mode, .schedulingDiagnostics = true}));
                const auto snapshot = executor.executionSnapshot();
                JOINED_CHECK(snapshot && snapshot->success && snapshot->pipelinedSubmission ==
                    (mode == render::FrameSubmissionMode::Pipelined));
                const auto& stats = executor.executionStats();
                JOINED_CHECK(snapshot->batches.size() == stats.submittedBatchCount);
                JOINED_CHECK(stats.scheduling.nativeSubmits == stats.submittedBatchCount);
                if (mode == render::FrameSubmissionMode::Joined) {
                    JOINED_CHECK(stats.submittedBatchCount == 1 && stats.batchesSubmittedWhileRecording == 0);
                    JOINED_CHECK(snapshot->batches[0].segmentIds.size() == snapshot->segments.size());
                    if (workers > 1) {
                        JOINED_CHECK(stats.recordingBatchCount > 0);
                        if (workload == 1) { JOINED_CHECK(stats.recordingTaskCount > 0); }
                    }
                } else {
                    JOINED_CHECK(stats.submittedBatchCount > 1);
                }
                if (context.device.capabilities().timestampQueries) {
                    JOINED_CHECK(snapshot->segments.front().role == render::RenderGraphSegmentRole::Prologue);
                    JOINED_CHECK(snapshot->segments.back().role == render::RenderGraphSegmentRole::Epilogue);
                }
                JOINED_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
                JOINED_CHECK(checkWords(executor, "Transfer5.data", context));
                std::vector<render::RenderGraphExecutionStats> timings;
                JOINED_REQUIRE(executor.collectCompletedGpuExecutionStats().transform([&](auto value) { timings = std::move(value); }));
                if (context.device.capabilities().timestampQueries) {
                    JOINED_CHECK(timings.size() == 1 && timings[0].gpuTimingAvailable && timings[0].nodes.size() == 6);
                }
            }
        }
        return RHITestResult::pass("Joined merges timing and serial/worker recording boundaries; pipelining and GPU data remain correct");
    }
};

class JoinedSubmissionCrossQueueWaitTest final : public RHITest {
public:
    JoinedSubmissionCrossQueueWaitTest() { type = RHITestType::Rendering; name = "joined_submission_cross_queue_wait_boundary"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization,
                .capabilities = {bench::Capability::IndependentCompute, bench::Capability::TimestampQueries},
                .queues = {render::QueueType::Graphics, render::QueueType::Compute}},
            .coverage = {"graph.joined.crossQueue.wait.firstConsumer", "graph.joined.crossQueue.producer.signal"},
            .artifacts = {"readback.bin"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto* compute = context.device.getQueue(render::QueueType::Compute);
        if (!compute || compute->sameQueue(context.graphicsQueue) || !context.device.capabilities().timestampQueries) {
            return RHITestResult::skip("independent compute and graphics timestamps required");
        }
        registerJoinedSubmissionPass();
        std::unique_ptr<render::Semaphore> gate;
        std::unique_ptr<render::TimestampQueryPool> progress;
        JOINED_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        JOINED_REQUIRE(context.device.createTimestampQueryPool(context.graphicsQueue, {.queryCount = 1})
            .transform([&](auto value) { progress = std::move(value); }));
        JOINED_REQUIRE(prepareProgress(context, *progress));
        JoinedProbe probe{gate.get(), progress.get()};
        ProbeBinding binding(probe);
        render::RenderGraph graph;
        graph.addNode("JoinedSubmissionTransferPass", "Seed", {{"compute", true}});
        graph.addNode("JoinedSubmissionTransferPass", "Slow", {{"compute", true}, {"gated", true}});
        graph.addNode("JoinedSubmissionTransferPass", "Independent", {{"copy", true}, {"progress", true}});
        graph.addNode("JoinedSubmissionTransferPass", "Consumer", {{"copy", true}});
        graph.addEdge("Seed.data", "Independent.source");
        graph.addEdge("Slow.data", "Consumer.source");
        graph.markOutput("Independent.data");
        graph.markOutput("Consumer.data");
        render::RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        JOINED_REQUIRE(executor.compile(context.device, graph, 1, 1, log));
        GateDrain drain{context.device, *gate};
        JOINED_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue, .computeQueue = compute,
            .recordingWorkerLimit = 1, .submissionMode = render::FrameSubmissionMode::Joined}));
        const auto snapshot = executor.executionSnapshot();
        JOINED_CHECK(snapshot && snapshot->success);
        const auto* seed = passSegment(*snapshot, graph.findNode("Seed")->id);
        const auto* slow = passSegment(*snapshot, graph.findNode("Slow")->id);
        const auto* independent = passSegment(*snapshot, graph.findNode("Independent")->id);
        const auto* consumer = passSegment(*snapshot, graph.findNode("Consumer")->id);
        JOINED_CHECK(seed && slow && independent && consumer);
        const auto* seedBatch = segmentBatch(*snapshot, seed->id);
        const auto* slowBatch = segmentBatch(*snapshot, slow->id);
        const auto* independentBatch = segmentBatch(*snapshot, independent->id);
        const auto* consumerBatch = segmentBatch(*snapshot, consumer->id);
        JOINED_CHECK(seedBatch && slowBatch && independentBatch && consumerBatch);
        JOINED_CHECK(seedBatch != slowBatch && independentBatch != consumerBatch);
        JOINED_CHECK(std::find(independentBatch->waitPredecessors.begin(), independentBatch->waitPredecessors.end(), seed->id)
            != independentBatch->waitPredecessors.end());
        JOINED_CHECK(std::find(independentBatch->waitPredecessors.begin(), independentBatch->waitPredecessors.end(), slow->id)
            == independentBatch->waitPredecessors.end());
        JOINED_CHECK(std::find(consumerBatch->waitPredecessors.begin(), consumerBatch->waitPredecessors.end(), slow->id)
            != consumerBatch->waitPredecessors.end());
        JOINED_CHECK(observeProgress(*progress) && !executor.lastSubmittedCompletion().isComplete());
        JOINED_REQUIRE(gate->signal(1));
        JOINED_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        JOINED_CHECK(checkWords(executor, "Independent.data", context));
        JOINED_CHECK(checkWords(executor, "Consumer.data", context));
        return RHITestResult::pass("Early producer signal and independent graphics progress survive later compute waits");
    }
};

class JoinedSubmissionImplicitWaitTest final : public RHITest {
public:
    JoinedSubmissionImplicitWaitTest() { type = RHITestType::Rendering; name = "joined_submission_implicit_wait_boundary"; }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"graph.joined.commandDependency.firstConsumer"},
            bench::Layer::RenderGraph, "core", "sync", {"readback.bin"});
        result.requirements.capabilities.push_back(bench::Capability::TimestampQueries);
        return result;
    }
    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) { return RHITestResult::skip("graphics timestamps required"); }
        registerJoinedSubmissionPass();
        std::unique_ptr<render::Semaphore> gate;
        std::unique_ptr<render::TimestampQueryPool> progress;
        JOINED_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        JOINED_REQUIRE(context.device.createTimestampQueryPool(context.graphicsQueue, {.queryCount = 1})
            .transform([&](auto value) { progress = std::move(value); }));
        JOINED_REQUIRE(prepareProgress(context, *progress));
        JoinedProbe probe{gate.get(), progress.get()};
        ProbeBinding binding(probe);
        render::RenderGraph graph;
        graph.addNode("JoinedSubmissionTransferPass", "Independent", {{"progress", true}});
        graph.addNode("JoinedSubmissionTransferPass", "WaitOnce", {{"copy", true}, {"gated", true}, {"gateValue", 1}});
        graph.addNode("JoinedSubmissionTransferPass", "WaitRepeat", {{"copy", true}, {"gated", true}, {"gateValue", 1}});
        graph.addNode("JoinedSubmissionTransferPass", "WaitHigher", {{"copy", true}, {"gated", true}, {"gateValue", 2}});
        graph.addEdge("Independent.data", "WaitOnce.source");
        graph.addEdge("WaitOnce.data", "WaitRepeat.source");
        graph.addEdge("WaitRepeat.data", "WaitHigher.source");
        graph.markOutput("Independent.data");
        graph.markOutput("WaitOnce.data");
        graph.markOutput("WaitRepeat.data");
        graph.markOutput("WaitHigher.data");
        render::RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        JOINED_REQUIRE(executor.compile(context.device, graph, 1, 1, log));
        GateDrain drain{context.device, *gate, 2};
        JOINED_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue,
            .recordingWorkerLimit = 1, .submissionMode = render::FrameSubmissionMode::Joined}));
        const auto snapshot = executor.executionSnapshot();
        JOINED_CHECK(snapshot && snapshot->batches.size() == 3);
        const auto* independent = passSegment(*snapshot, graph.findNode("Independent")->id);
        const auto* once = passSegment(*snapshot, graph.findNode("WaitOnce")->id);
        const auto* repeat = passSegment(*snapshot, graph.findNode("WaitRepeat")->id);
        const auto* higher = passSegment(*snapshot, graph.findNode("WaitHigher")->id);
        JOINED_CHECK(independent && once && repeat && higher);
        const auto* independentBatch = segmentBatch(*snapshot, independent->id);
        const auto* onceBatch = segmentBatch(*snapshot, once->id);
        const auto* repeatBatch = segmentBatch(*snapshot, repeat->id);
        const auto* higherBatch = segmentBatch(*snapshot, higher->id);
        JOINED_CHECK(independentBatch && onceBatch && repeatBatch && higherBatch);
        JOINED_CHECK(independentBatch != onceBatch && onceBatch == repeatBatch && repeatBatch != higherBatch);
        JOINED_CHECK(observeProgress(*progress) && !executor.lastSubmittedCompletion().isComplete());
        JOINED_REQUIRE(gate->signal(2));
        JOINED_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        JOINED_CHECK(checkWords(executor, "Independent.data", context));
        JOINED_CHECK(checkWords(executor, "WaitOnce.data", context));
        JOINED_CHECK(checkWords(executor, "WaitRepeat.data", context));
        JOINED_CHECK(checkWords(executor, "WaitHigher.data", context));
        return RHITestResult::pass("Repeated timeline waits coalesce; new and increased waits preserve independent GPU progress");
    }
};

class JoinedSubmissionForkJoinTest final : public RHITest {
public:
    JoinedSubmissionForkJoinTest() { type = RHITestType::Rendering; name = "joined_submission_fork_join_parallelism"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "sync", .profile = "async", .layer = bench::Layer::RenderGraph,
            .requirements = {.validation = bench::Validation::Synchronization,
                .capabilities = {bench::Capability::IndependentCompute, bench::Capability::TimestampQueries},
                .queues = {render::QueueType::Graphics, render::QueueType::Compute}},
            .coverage = {"graph.joined.forkJoin.independentGraphics.progress", "graph.joined.forkJoin.visibility"},
            .artifacts = {"readback.bin"}};
    }
    RHITestResult run(RHITestContext& context) override
    {
        auto* compute = context.device.getQueue(render::QueueType::Compute);
        if (!compute || compute->sameQueue(context.graphicsQueue) || !context.device.capabilities().timestampQueries) {
            return RHITestResult::skip("independent compute and graphics timestamps required");
        }
        registerJoinedSubmissionPass();
        std::unique_ptr<render::Semaphore> gate;
        std::unique_ptr<render::TimestampQueryPool> progress;
        JOINED_REQUIRE(context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); }));
        JOINED_REQUIRE(context.device.createTimestampQueryPool(context.graphicsQueue, {.queryCount = 1})
            .transform([&](auto value) { progress = std::move(value); }));
        JOINED_REQUIRE(prepareProgress(context, *progress));
        JoinedProbe probe{gate.get(), progress.get()};
        ProbeBinding binding(probe);
        render::RenderGraph graph;
        graph.addNode("JoinedSubmissionTransferPass", "Branches", {{"fork", true}, {"gated", true}, {"progress", true}});
        graph.markOutput("Branches.data");
        render::RenderGraphExecutor executor;
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        JOINED_REQUIRE(executor.compile(context.device, graph, 1, 1, log));
        GateDrain drain{context.device, *gate};
        JOINED_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue, .computeQueue = compute,
            .recordingWorkerLimit = 1, .submissionMode = render::FrameSubmissionMode::Joined}));
        const auto snapshot = executor.executionSnapshot();
        JOINED_CHECK(snapshot && executor.executionStats().asyncComputeBranches == 1);
        const render::RenderGraphExecutionSegmentSnapshot* software = nullptr;
        const render::RenderGraphExecutionSegmentSnapshot* hardware = nullptr;
        const render::RenderGraphExecutionSegmentSnapshot* join = nullptr;
        for (const auto& segment : snapshot->segments) {
            if (segment.role == render::RenderGraphSegmentRole::ComputeBranch) { software = &segment; }
            if (segment.role == render::RenderGraphSegmentRole::GraphicsBranch) { hardware = &segment; }
            if (segment.role == render::RenderGraphSegmentRole::Join) { join = &segment; }
        }
        JOINED_CHECK(software && hardware && join);
        const auto* hardwareBatch = segmentBatch(*snapshot, hardware->id);
        const auto* joinBatch = segmentBatch(*snapshot, join->id);
        JOINED_CHECK(hardwareBatch && joinBatch && hardwareBatch != joinBatch);
        JOINED_CHECK(hardwareBatch->waitPredecessors.empty());
        JOINED_CHECK(std::find(joinBatch->waitPredecessors.begin(), joinBatch->waitPredecessors.end(), software->id)
            != joinBatch->waitPredecessors.end());
        JOINED_CHECK(observeProgress(*progress) && !executor.lastSubmittedCompletion().isComplete());
        JOINED_REQUIRE(gate->signal(1));
        JOINED_REQUIRE(executor.waitForSubmittedWork(kWaitTimeout));
        JOINED_CHECK(checkWords(executor, "Branches.data", context));
        return RHITestResult::pass("Graphics branch progresses independently; join waits for both branch writes");
    }
};

METALLIC_REGISTER_RHI_TEST(JoinedSubmissionChainTest);
METALLIC_REGISTER_RHI_TEST(JoinedSubmissionCrossQueueWaitTest);
METALLIC_REGISTER_RHI_TEST(JoinedSubmissionImplicitWaitTest);
METALLIC_REGISTER_RHI_TEST(JoinedSubmissionForkJoinTest);

} // namespace
} // namespace metallic::tests
