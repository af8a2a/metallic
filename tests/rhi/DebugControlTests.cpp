#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <cstring>

namespace metallic::tests {
namespace {
using namespace render;
using debug::DebugValue;

#define DEBUG_REQUIRE(expression) do { const auto result = (expression); \
    if (!result) { return RhiTestResult::fail(std::string(#expression) + ": " + resultToString(result)); } } while (false)

class DebugCheckpointPass final : public ComputePass {
public:
    QueueType queueType() const override { return QueueType::Copy; }
    bool supportsAsyncQueue() const override { return true; }
    std::vector<std::string> debugCheckpoints() const override { return {"Early", "Late", "AfterPass"}; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferOutput("values").buffer(16, 4).transferWrite();
        return reflection;
    }
    Result compile(const RenderGraphCompileContext& context, std::string&) override
    {
        Result result = context.device->createBuffer({.size = 32, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy}, upload_);
        if (!result) { return result; }
        void* mapped = upload_->map();
        if (!mapped) { return makeError(Error::Failure); }
        const uint32_t words[] = {11, 1, 2, 3, 22, 4, 5, 6};
        std::memcpy(mapped, words, sizeof(words)); upload_->flush(); upload_->unmap();
        return {};
    }
    Result execute(RenderGraphExecutionContext& context) override
    {
        auto* output = context.output("values");
        auto& commands = context.commandBuffer();
        commands.copyBuffer({.source = upload_.get(), .destination = output->buffer, .size = 16});
        const DebugResourceBinding binding{.id = "Ids", .buffer = output->buffer, .state = output->state, .size = 16};
        context.debugCheckpoint("Early", std::span(&binding, 1), {{"sourcePhase", "Early"}});
        BufferBarrierDesc barrier{.buffer = output->buffer, .before = output->state, .after = output->state};
        commands.barrier({.buffers = &barrier, .bufferCount = 1});
        commands.copyBuffer({.source = upload_.get(), .destination = output->buffer, .sourceOffset = 16, .size = 16});
        context.debugCheckpoint("Late", std::span(&binding, 1), {{"sourcePhase", "Late"}});
        return context.properties().value("fail", false) ? makeError(Error::Failure) : Result{};
    }
private:
    std::unique_ptr<Buffer> upload_;
};

DebugValue call(RenderDebugRuntime& runtime, std::string method, DebugValue params = DebugValue::object())
{
    return runtime.core().dispatch({{"id", "gpu-test"}, {"method", method}, {"params", params}});
}

std::string capture(RenderDebugRuntime& runtime, const char* checkpoint)
{
    auto response = call(runtime, "capture.batch", {{"pass", "Probe"}, {"checkpoint", checkpoint},
        {"resources", DebugValue::array({{{"id", "Ids"}, {"count", 4}}})}});
    return response.at("result").at("job");
}

struct FrameCommands {
    RenderFrameContext frame;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    QueueSubmissionTracker tracker;
    ~FrameCommands()
    {
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
    }
    Result initialize(Device& device, Queue& queue)
    {
        auto result = device.createCommandPool(queue, pool);
        if (result) { result = pool->createCommandBuffer(commands); }
        return result ? tracker.initialize(device, queue) : result;
    }
    Result begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    Result submit()
    {
        auto result = commands->end();
        if (!result) { return result; }
        CommandBuffer* raw = commands.get();
        return tracker.submit({.commandBuffers = &raw, .commandBufferCount = 1}, frame);
    }
};

class DebugControlCaptureTest final : public RhiTest {
public:
    DebugControlCaptureTest() { name = "DebugControl checkpoint capture and lifetime"; type = RhiTestType::Command; }
    RhiTestResult run(RhiTestContext& context) override
    {
        registerRenderGraphPassType("DebugCheckpointPass", "Debug capture fixture", [] { return std::make_unique<DebugCheckpointPass>(); });
        RenderGraph graph;
        graph.addNode("DebugCheckpointPass", "Probe"); graph.markOutput("Probe.values");
        RenderDebugRuntime runtime;
        RenderGraphExecutor executor;
        FrameCommands frame;
        executor.setDebugObserver(&runtime);
        std::string log;
        DEBUG_REQUIRE(executor.compile(context.device, graph, 4, 4, log));
        DEBUG_REQUIRE(frame.initialize(context.device, context.graphicsQueue));
        const auto group = call(runtime, "capture.batch", {{"batches", {
            {{"pass", "Probe"}, {"checkpoint", "Early"}, {"resources", {{{"id", "Ids"}, {"count", 4}}}}},
            {{"pass", "Probe"}, {"checkpoint", "Late"}, {"resources", {{{"id", "Ids"}, {"count", 4}}}}}}}});
        if (group["status"] != "ok") { return RhiTestResult::fail(group.dump()); }
        const std::string early = group["result"]["jobs"][0]["job"], late = group["result"]["jobs"][1]["job"];
        DEBUG_REQUIRE(frame.begin(70));
        DEBUG_REQUIRE(executor.execute(*frame.commands));
        runtime.poll();
        if (call(runtime, "jobs.get", {{"job", early}})["result"]["state"] != "Recorded" ||
            call(runtime, "frame.latest")["status"] != "error") {
            return RhiTestResult::fail("Unsubmitted recording was published as complete");
        }
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait(5'000'000'000ull)); runtime.poll();
        for (const auto& [id, expected] : {std::pair{early, 11u}, std::pair{late, 22u}}) {
            const auto result = call(runtime, "eval", {{"job", id}, {"expression", "buffers[\"Ids\"][0]"}});
            if (result.value("status", "") != "ok" || result["result"]["value"] != expected) {
                return RhiTestResult::fail("Checkpoint copy did not preserve original data: " + result.dump());
            }
            if (result["result"]["evidence"]["provenance"]["submissionFrame"] != 70) {
                return RhiTestResult::fail("Lost external submission frame identity");
            }
        }
        // The copy-queue self-submitting route uses the same observer, including
        // aggregate completion and cancellation semantics.
        const std::string next = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue, .copyQueue = context.device.getQueue(QueueType::Copy)}));
        DEBUG_REQUIRE(executor.waitForSubmittedWork(5'000'000'000ull)); runtime.poll();
        if (call(runtime, "eval", {{"job", next}, {"expression", "buffers[\"Ids\"][0]"}})["result"]["value"] != 11 ||
            call(runtime, "eval", {{"job", late}, {"expression", "buffers[\"Ids\"][0]"}})["result"]["value"] != 22) {
            return RhiTestResult::fail("Self submission or retained capture changed content");
        }
        const auto stale = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.compile(context.device, graph, 8, 8, log));
        if (call(runtime, "jobs.get", {{"job", stale}})["result"]["error"]["code"] != "StaleHandle") {
            return RhiTestResult::fail("Resize did not invalidate queued capture");
        }
        const auto reloadStale = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.reloadShaders(log));
        if (call(runtime, "jobs.get", {{"job", reloadStale}})["result"]["error"]["code"] != "StaleHandle") {
            return RhiTestResult::fail("Shader reload did not invalidate queued capture");
        }
        graph.setNodeRuntimeProperty(graph.findNode("Probe")->id, "fail", true);
        executor.syncRuntimeProperties(graph);
        const auto abandoned = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.begin(71));
        if (executor.execute(*frame.commands)) { return RhiTestResult::fail("Expected fixture recording failure"); }
        DEBUG_REQUIRE(frame.pool->reset()); frame.frame.cancel(); runtime.poll();
        if (call(runtime, "jobs.get", {{"job", abandoned}})["result"]["state"] == "Ready") {
            return RhiTestResult::fail("Abandoned recording produced evidence");
        }
        // Submit a recorded prefix after the pass reports failure, then reject
        // a later segment. Already accepted evidence must survive frame.cancel.
        const auto prefixJob = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.begin(72));
        if (executor.execute(*frame.commands)) { return RhiTestResult::fail("Expected prefix fixture failure"); }
        DEBUG_REQUIRE(frame.commands->end());
        CommandBuffer* raw = frame.commands.get();
        GpuCompletionPoint prefix, failed;
        DEBUG_REQUIRE(frame.tracker.submitSegment({.commandBuffers = &raw, .commandBufferCount = 1}, frame.frame, prefix));
        if (frame.tracker.submitSegment({.commandBufferCount = 1}, frame.frame, failed)) { return RhiTestResult::fail("Expected rejected tail submission"); }
        frame.frame.cancel(); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto prefixResult = call(runtime, "eval", {{"job", prefixJob}, {"expression", "buffers[\"Ids\"][0]"}});
        if (prefixResult["status"] != "ok" || prefixResult["result"]["value"] != 11 ||
            prefixResult["result"]["evidence"]["provenance"]["executionComplete"] != false) {
            return RhiTestResult::fail("Submitted prefix capture was released early or attributed to a complete execution");
        }
        const auto legacyJob = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.pool->reset()); DEBUG_REQUIRE(frame.commands->begin());
        (void)executor.execute(*frame.commands); DEBUG_REQUIRE(frame.commands->end());
        const auto legacy = call(runtime, "frame.latest", {{"recorded", true}});
        if (legacy["result"]["source"] != "recorded-untracked" ||
            call(runtime, "jobs.get", {{"job", legacyJob}})["result"]["error"]["code"] != "Unsupported") {
            return RhiTestResult::fail("Legacy metadata was unavailable or an untracked GPU capture was accepted");
        }
        DEBUG_REQUIRE(frame.pool->reset());
        runtime.drain();
        return RhiTestResult::pass("Early/late copies, external/self submission, overwrite isolation, resize and cancellation verified");
    }
};

class DebugControlTextureTest final : public RhiTest {
public:
    DebugControlTextureTest() { name = "DebugControl texture ROI and validation events"; type = RhiTestType::Rendering; }
    RhiTestResult run(RhiTestContext& context) override
    {
        RenderDebugRuntime runtime;
        RenderGraphExecutor executor;
        FrameCommands frame;
        executor.setDebugObserver(&runtime);
        RenderGraph graph = RenderGraph::createDefaultTriangleGraph();
        std::string log;
        DEBUG_REQUIRE(executor.compile(context.device, graph, 16, 16, log));
        const std::string output = graph.firstOutputName();
        const std::string pass = output.substr(0, output.find('.'));
        const auto queued = call(runtime, "capture.batch", {{"pass", pass}, {"resources", DebugValue::array({
            {{"id", output}, {"roi", {{"x", 4}, {"y", 4}, {"width", 3}, {"height", 2}}}}})}});
        if (queued["status"] != "ok") { return RhiTestResult::fail(queued.dump()); }
        DEBUG_REQUIRE(frame.initialize(context.device, context.graphicsQueue));
        DEBUG_REQUIRE(frame.begin(0)); DEBUG_REQUIRE(executor.execute(*frame.commands));
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto job = queued["result"]["job"];
        auto response = call(runtime, "eval", {{"job", job}, {"expression", "count(buffers[\"" + output + "\"])"}});
        if (response["status"] != "ok" || response["result"]["value"] != 6 || response["result"]["coverage"][output]["completeCoverage"] != false) {
            return RhiTestResult::fail("ROI shape or coverage mismatch: " + response.dump());
        }
        const auto sink = runtime.validationSink();
        sink.callback(sink.context, {.severity = 4096, .messageId = 7, .messageIdName = "fixture", .message = "validation evidence"});
        const auto events = runtime.core().events("validation");
        if (events["events"][0]["messageId"] != 7 || !events["events"][0]["execution"].is_null()) {
            return RhiTestResult::fail("Validation event lost severity/identity or invented frame association");
        }
        runtime.drain();
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(DebugControlCaptureTest);
METALLIC_REGISTER_RHI_TEST(DebugControlTextureTest);
} // namespace
} // namespace metallic::tests
