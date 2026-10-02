#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "RHITest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <cstring>
#include <bit>
#include <limits>
#include "Runtime/Render/Core/ComputeKernel.h"
#include "DebugRestoreProbeParameters.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::tests {
namespace {
using namespace render;
using debug::DebugValue;

#define DEBUG_REQUIRE(expression) do { const auto result = (expression); \
    if (!result) { return RHITestResult::fail(std::string(#expression) + ": " + resultToString(result)); } } while (false)

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
    Result<> compile(const RenderGraphCompileContext& context, std::string&) override
    {
        Result<> result = context.device->createBuffer({.size = 32, .usage = BufferUsageBits::TransferSource,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute | QueueAccessBits::Copy}).transform([&](auto rhiValue) { upload_ = std::move(rhiValue); });
        if (!result) { return result; }
        void* mapped = upload_->map();
        if (!mapped) { return makeError(Error::Failure); }
        const uint32_t words[] = {11, 1, 2, 3, 22, 4, 5, 6};
        std::memcpy(mapped, words, sizeof(words)); upload_->flush(); upload_->unmap();
        return {};
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto* output = context.output("values");
        auto& commands = context.commandBuffer();
        {
            auto sourceSlice = upload_.get()->slice({0, 16});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = output->buffer->slice({0, 16});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
        const DebugResourceBinding binding{.id = "Ids", .buffer = output->buffer, .state = output->state, .size = 16};
        context.debugCheckpoint("Early", std::span(&binding, 1), {{"sourcePhase", "Early"}});
        BufferBarrierDesc barrier{
            .buffer = output->buffer,
            .before = metallic::render::resourceSyncScope(output->state, metallic::render::PipelineStageBits::AllCommands),
            .after = metallic::render::resourceSyncScope(output->state, metallic::render::PipelineStageBits::AllCommands),
        };
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
        {
            auto sourceSlice = upload_.get()->slice({16, 16});
            if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
            auto destinationSlice = output->buffer->slice({0, 16});
            if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
            if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
        }
        context.debugCheckpoint("Late", std::span(&binding, 1), {{"sourcePhase", "Late"}});
        return context.properties().value("fail", false) ? makeError(Error::Failure) : Result<>{};
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
    Result<> initialize(Device& device, Queue& queue)
    {
        auto result = device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        if (result) { result = pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }); }
        return result ? tracker.initialize(device, queue) : result;
    }
    Result<> begin(uint64_t index)
    {
        auto result = frame.begin(index);
        if (result) { result = pool->reset(); }
        return result ? commands->begin(&frame) : result;
    }
    Result<> submit()
    {
        auto result = commands->end();
        if (!result) { return result; }
        CommandBuffer* raw = commands.get();
        return tracker.submit({.commandBuffers = {&raw, 1}}, frame);
    }
};

class DebugControlCaptureTest final : public RHITest {
public:
    DebugControlCaptureTest() { name = "DebugControl checkpoint capture and lifetime"; type = RHITestType::Command; }
    RHITestResult run(RHITestContext& context) override
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
        if (group["status"] != "ok") { return RHITestResult::fail(group.dump()); }
        const std::string early = group["result"]["jobs"][0]["job"], late = group["result"]["jobs"][1]["job"];
        DEBUG_REQUIRE(frame.begin(70));
        DEBUG_REQUIRE(executor.execute(*frame.commands));
        runtime.poll();
        if (call(runtime, "jobs.get", {{"job", early}})["result"]["state"] != "Recorded" ||
            call(runtime, "frame.latest")["status"] != "error") {
            return RHITestResult::fail("Unsubmitted recording was published as complete");
        }
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait(5'000'000'000ull)); runtime.poll();
        for (const auto& [id, expected] : {std::pair{early, 11u}, std::pair{late, 22u}}) {
            const auto result = call(runtime, "eval", {{"job", id}, {"expression", "buffers[\"Ids\"][0]"}});
            if (result.value("status", "") != "ok" || result["result"]["value"] != expected) {
                return RHITestResult::fail("Checkpoint copy did not preserve original data: " + result.dump());
            }
            if (result["result"]["evidence"]["provenance"]["submissionFrame"] != 70) {
                return RHITestResult::fail("Lost external submission frame identity");
            }
        }
        // The copy-queue self-submitting route uses the same observer, including
        // aggregate completion and cancellation semantics.
        const std::string next = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.execute({.graphicsQueue = &context.graphicsQueue, .copyQueue = context.device.getQueue(QueueType::Copy)}));
        DEBUG_REQUIRE(executor.waitForSubmittedWork(5'000'000'000ull)); runtime.poll();
        if (call(runtime, "eval", {{"job", next}, {"expression", "buffers[\"Ids\"][0]"}})["result"]["value"] != 11 ||
            call(runtime, "eval", {{"job", late}, {"expression", "buffers[\"Ids\"][0]"}})["result"]["value"] != 22) {
            return RHITestResult::fail("Self submission or retained capture changed content");
        }
        const auto stale = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.compile(context.device, graph, 8, 8, log));
        if (call(runtime, "jobs.get", {{"job", stale}})["result"]["error"]["code"] != "StaleHandle") {
            return RHITestResult::fail("Resize did not invalidate queued capture");
        }
        const auto reloadStale = capture(runtime, "Early");
        DEBUG_REQUIRE(executor.reloadShaders(log));
        if (call(runtime, "jobs.get", {{"job", reloadStale}})["result"]["error"]["code"] != "StaleHandle") {
            return RHITestResult::fail("Shader reload did not invalidate queued capture");
        }
        graph.setNodeRuntimeProperty(graph.findNode("Probe")->id, "fail", true);
        executor.syncRuntimeProperties(graph);
        const auto abandoned = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.begin(71));
        if (!hasError(executor.execute(*frame.commands), Error::Failure)) {
            return RHITestResult::fail("Expected fixture recording failure");
        }
        DEBUG_REQUIRE(frame.pool->reset()); frame.frame.cancel(); runtime.poll();
        if (call(runtime, "jobs.get", {{"job", abandoned}})["result"]["state"] == "Ready") {
            return RHITestResult::fail("Abandoned recording produced evidence");
        }
        // Submit a recorded prefix after the pass reports failure, then reject
        // a later segment. Already accepted evidence must survive frame.cancel.
        // Failed recording advances resource state, so rebuild the graph before
        // retrying. Queue the capture after compile to use the new generation.
        DEBUG_REQUIRE(executor.compile(context.device, graph, 8, 8, log));
        const auto prefixJob = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.begin(72));
        if (!hasError(executor.execute(*frame.commands), Error::Failure)) {
            return RHITestResult::fail("Expected prefix fixture failure");
        }
        DEBUG_REQUIRE(frame.commands->end());
        CommandBuffer* raw = frame.commands.get();
        GPUCompletionPoint prefix, failed;
        DEBUG_REQUIRE(frame.tracker.submitSegment({.commandBuffers = {&raw, 1}}, frame.frame).transform([&](auto value) { prefix = std::move(value); }));
        if (frame.tracker.submitSegment({.commandBuffers = std::array<render::CommandBuffer*, 1>{nullptr}}, frame.frame).transform([&](auto value) { failed = std::move(value); })) { return RHITestResult::fail("Expected rejected tail submission"); }
        frame.frame.cancel(); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto prefixResult = call(runtime, "eval", {{"job", prefixJob}, {"expression", "buffers[\"Ids\"][0]"}});
        if (prefixResult["status"] != "ok" || prefixResult["result"]["value"] != 11 ||
            prefixResult["result"]["evidence"]["provenance"]["executionComplete"] != false) {
            return RHITestResult::fail("Submitted prefix capture was released early or attributed to a complete execution");
        }
        DEBUG_REQUIRE(executor.compile(context.device, graph, 8, 8, log));
        const auto legacyJob = capture(runtime, "Early");
        DEBUG_REQUIRE(frame.pool->reset()); DEBUG_REQUIRE(frame.commands->begin());
        (void)executor.execute(*frame.commands); DEBUG_REQUIRE(frame.commands->end());
        const auto legacy = call(runtime, "frame.latest", {{"recorded", true}});
        if (legacy["result"]["source"] != "recorded-untracked" ||
            call(runtime, "jobs.get", {{"job", legacyJob}})["result"]["error"]["code"] != "Unsupported") {
            return RHITestResult::fail("Legacy metadata was unavailable or an untracked GPU capture was accepted");
        }
        DEBUG_REQUIRE(frame.pool->reset());
        runtime.drain();
        return RHITestResult::pass("Early/late copies, external/self submission, overwrite isolation, resize and cancellation verified");
    }
};

class DebugControlTextureTest final : public RHITest {
public:
    DebugControlTextureTest() { name = "DebugControl texture ROI and validation events"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
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
        if (queued["status"] != "ok") { return RHITestResult::fail(queued.dump()); }
        DEBUG_REQUIRE(frame.initialize(context.device, context.graphicsQueue));
        DEBUG_REQUIRE(frame.begin(0)); DEBUG_REQUIRE(executor.execute(*frame.commands));
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto job = queued["result"]["job"];
        auto response = call(runtime, "eval", {{"job", job}, {"expression", "count(buffers[\"" + output + "\"])"}});
        if (response["status"] != "ok" || response["result"]["value"] != 6 || response["result"]["coverage"][output]["completeCoverage"] != false) {
            return RHITestResult::fail("ROI shape or coverage mismatch: " + response.dump());
        }
        const auto sink = runtime.validationSink();
        sink.callback(sink.context, {.severity = 4096, .messageId = 7, .messageIdName = "fixture", .message = "validation evidence"});
        const auto events = runtime.core().events("validation");
        if (events["events"][0]["messageId"] != 7 || !events["events"][0]["execution"].is_null()) {
            return RHITestResult::fail("Validation event lost severity/identity or invented frame association");
        }
        runtime.drain();
        return RHITestResult::pass();
    }
};


class DebugControlGPUProbeTest final : public RHITest {
public:
    DebugControlGPUProbeTest() { name = "DebugControl fixed GPU probes watch and binding restoration"; type = RHITestType::Command; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<Device> device;
        RenderDebugRuntime runtime;
        auto setup = createDevice({.applicationName = "Debug GPU Probe", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .validationSink = runtime.validationSink()}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!setup && hasError(setup, Error::Unsupported)) { return RHITestResult::skip("Descriptor heap unavailable"); }
        DEBUG_REQUIRE(setup);
        auto& queue = *device->getQueue(QueueType::Graphics);
        FrameCommands frame;
        DEBUG_REQUIRE(frame.initialize(*device, queue));
        runtime.compiled({{"id", "probe-graph"}, {"generation", 1}, {"passes", {{{"name", "Probe"},
            {"active", true}, {"checkpoints", {"Early", "Late", "AfterPass"}}}}}});
        std::unique_ptr<Buffer> upload, ids, floats, signedValues, records, sentinel;
        const auto hostBuffer = [&](const void* data, uint64_t bytes, std::unique_ptr<Buffer>& output) -> Result<> {
            auto result = device->createBuffer({.size = bytes, .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { output = std::move(rhiValue); });
            if (!result) { return result; }
            auto* mapped = output->map(); if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data, bytes); output->flush(); output->unmap(); return {};
        };
        constexpr uint32_t kCount = 65539;
        std::vector<uint32_t> words(kCount * 2);
        for (uint32_t i = 0; i < kCount; ++i) { words[i] = i; words[i + kCount] = i; }
        words[0] = 11; words[kCount - 1] = UINT32_MAX; words[kCount] = 22; words[kCount * 2 - 1] = 77;
        DEBUG_REQUIRE(hostBuffer(words.data(), words.size() * 4, upload));
        const uint32_t floatBits[] = {0xc0200000, 0x40a00000, 0x7fc00000, 0x7f800000, 0xff800000, 0, 0x80000000};
        DEBUG_REQUIRE(hostBuffer(floatBits, sizeof(floatBits), floats));
        const int32_t signedWords[] = {INT32_MIN, -1, 0, INT32_MAX};
        DEBUG_REQUIRE(hostBuffer(signedWords, sizeof(signedWords), signedValues));
        const uint32_t cluster[] = {0, 0, 0, 0x10000003, 0, 0, 0, 0x00000001};
        DEBUG_REQUIRE(hostBuffer(cluster, sizeof(cluster), records));
        DEBUG_REQUIRE(device->createBuffer({.size = kCount * 4, .usage = BufferUsageBits::Storage |
            BufferUsageBits::TransferDestination | BufferUsageBits::TransferSource}).transform([&](auto rhiValue) { ids = std::move(rhiValue); }));
        DEBUG_REQUIRE(device->createBuffer({.size = 4, .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource}).transform([&](auto rhiValue) { sentinel = std::move(rhiValue); }));
        const DebugResourceBinding bindings[] = {
            {.id = "Ids", .buffer = ids.get(), .state = ResourceState::ShaderRead, .size = kCount * 4},
            {.id = "Floats", .buffer = floats.get(), .state = ResourceState::ShaderRead, .layout = "f32"},
            {.id = "Signed", .buffer = signedValues.get(), .state = ResourceState::ShaderRead, .layout = "i32"},
            {.id = "Records", .buffer = records.get(), .state = ResourceState::ShaderRead, .layout = "VisibleClusterRecord"},
            {.id = "Sentinel", .buffer = sentinel.get(), .state = ResourceState::General}};
        DebugValue specs{{"pass", "Probe"}, {"checkpoint", "Early"}, {"probes", {
            {{"id", "Ids"}, {"name", "bounds"}, {"operation", "outOfBounds"}, {"count", kCount}, {"upper", 200}},
            {{"id", "Ids"}, {"name", "count"}, {"operation", "count"}, {"count", kCount}, {"predicate", "ge"}, {"value", 200}},
            {{"id", "Floats"}, {"name", "finite"}, {"operation", "nonFinite"}, {"count", 7}},
            {{"id", "Signed"}, {"name", "signed"}, {"operation", "outOfBounds"}, {"count", 4}, {"lower", -1}, {"upper", 1}},
            {{"id", "Records"}, {"name", "packed"}, {"operation", "count"}, {"field", "source"}, {"count", 2}, {"predicate", "eq"}, {"value", 1}}
        }}};
        auto lateSpec = specs; lateSpec["checkpoint"] = "Late";
        lateSpec["resources"] = {{{"id", "Sentinel"}, {"count", 1}}};
        const auto response = call(runtime, "gpu.probe", {{"batches", {specs, lateSpec}}});
        if (response["status"] != "ok") { return RHITestResult::fail(response.dump()); }
        const auto early = response["result"]["jobs"][0]["job"], late = response["result"]["jobs"][1]["job"];
        const auto watch = call(runtime, "watch.create", {{"probe", lateSpec}, {"everyExecutions", 2},
            {"trigger", {{"probe", "bounds"}, {"value", 0}}}})["result"]["watch"];
        ShaderCompileResult shader;
        DEBUG_REQUIRE(compileSlangShaderToSpirv({.moduleName = "DebugRestoreProbe", .entryPointName = "debugRestoreMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
        ComputeKernel original;
        std::string log;
        DEBUG_REQUIRE(original.initialize(*device, {
            .spirv = shader.spirv,
            .parameters = parameterAbi<DebugRestoreProbeParameters>(kDebugRestoreProbeABI, ParameterTransport::InlinePush),
        }, log));
        DEBUG_REQUIRE(frame.begin(1));
        auto& commands = *frame.commands;
        commands.hostWriteBarrier();
        BufferBarrierDesc sourceBarrier{.buffer = ids.get(), .before = {}, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
        if (auto commandResult = commands.synchronize({.buffers = {&sourceBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        {
            auto sourceSlice = upload.get()->slice({0, kCount * 4});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = ids.get()->slice({0, kCount * 4});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        sourceBarrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite}; sourceBarrier.after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead};
        if (auto commandResult = commands.synchronize({.buffers = {&sourceBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        BufferBarrierDesc outBarrier{.buffer = sentinel.get(), .before = {}, .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
        if (auto commandResult = commands.synchronize({.buffers = {&outBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        auto registry = device->resourceRegistry();
        if (!registry) { return RHITestResult::fail("Resource registry unavailable"); }
        ParameterWriter writer(*device, **registry, &frame.frame);
        const DebugRestoreProbeParameters params{writer.buffer(ids.get()), writer.dataBuffer(sentinel.get(), 4, 4), 0, 0};
        auto encoded = writer.encode(params, kDebugRestoreProbeABI, ParameterTransport::InlinePush);
        if (!encoded) { return RHITestResult::fail("Inline debug restoration parameters failed to encode"); }
        DEBUG_REQUIRE(original.dispatch(commands, *encoded, 1));
        runtime.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 9}, nullptr);
        runtime.boundary(commands, "Early", 0, "Probe", bindings, DebugValue::object());
        std::swap(sourceBarrier.before, sourceBarrier.after); if (auto commandResult = commands.synchronize({.buffers = {&sourceBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        {
            auto sourceSlice = upload.get()->slice({kCount * 4, kCount * 4});
            if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
            auto destinationSlice = ids.get()->slice({0, kCount * 4});
            if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
            if (auto commandResult = commands.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
        }
        std::swap(sourceBarrier.before, sourceBarrier.after); if (auto commandResult = commands.synchronize({.buffers = {&sourceBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        outBarrier.before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}; if (auto commandResult = commands.synchronize({.buffers = {&outBarrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        // No rebind: debug instrumentation must restore heap, pipeline and the
        // complete 32-byte inline block, including the BDA span and typed handle.
        commands.dispatch(1);
        runtime.boundary(commands, "Late", 0, "Probe", bindings, DebugValue::object());
        runtime.endExecution(true); runtime.poll();
        if (call(runtime, "jobs.get", {{"job", early}})["result"]["state"] != "Recorded") {
            return RHITestResult::fail("Probe not recorded or was published before submission: " + call(runtime, "jobs.get", {{"job", early}}).dump());
        }
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        auto first = call(runtime, "jobs.get", {{"job", early}}), second = call(runtime, "jobs.get", {{"job", late}});
        if (first["result"]["state"] != "Ready" || second["result"]["state"] != "Ready") { return RHITestResult::fail(first.dump() + second.dump()); }
        const auto& a = first["result"]["probes"];
        const auto& b = second["result"]["probes"];
        if (a["bounds"]["matchedCount"] != kCount - 200 || a["bounds"]["firstIndex"] != 200 || a["bounds"]["max"] != UINT32_MAX ||
            a["count"]["matchedCount"] != kCount - 200 || b["bounds"]["matchedCount"] != kCount - 201 || b["bounds"]["max"] != kCount - 2 ||
            a["finite"]["finiteCount"] != 4 || a["finite"]["nanCount"] != 1 || a["finite"]["infCount"] != 2 ||
            a["finite"]["min"] != -2.5 || a["finite"]["max"] != 5.0 || a["finite"]["firstIndex"] != 2 ||
            a["signed"]["matchedCount"] != 2 || a["signed"]["min"] != INT32_MIN || a["signed"]["max"] != INT32_MAX ||
            a["packed"]["matchedCount"] != 1) { return RHITestResult::fail("GPU reduction mismatch: " + first.dump() + second.dump()); }
        const auto restored = call(runtime, "eval", {{"job", late}, {"expression", "buffers.Sentinel[0]"}});
        if (restored["result"]["value"] != 22) { return RHITestResult::fail("Probe disturbed original compute binding: " + restored.dump()); }
        const auto triggered = call(runtime, "watch.get", {{"watch", watch}});
        if (triggered["result"]["state"] != "Triggered" || triggered["result"]["result"]["evidence"]["execution"] != 9) {
            return RHITestResult::fail("Watch did not retain exact-boundary evidence: " + triggered.dump());
        }
        if (call(runtime, "eval", {{"job", early}, {"expression", "probes.bounds.matchedCount"}})["result"]["value"] != kCount - 200) {
            return RHITestResult::fail("Typed probe evaluation differs from online result");
        }
        DebugValue invalid = DebugValue::array();
        auto bad = specs; bad["probes"] = DebugValue::array({specs["probes"][0]});
        bad["probes"][0]["field"] = "missing"; invalid.push_back(bad);
        bad["probes"][0].erase("field"); bad["probes"][0]["layoutHash"] = "wrong"; invalid.push_back(bad);
        bad["probes"][0].erase("layoutHash"); bad["probes"][0]["count"] = kCount + 1; invalid.push_back(bad);
        bad["probes"][0]["count"] = kCount; bad["probes"][0]["operation"] = "nonFinite"; invalid.push_back(bad);
        bad["probes"][0]["operation"] = "outOfBounds"; bad["probes"][0]["allocation"] = 2; invalid.push_back(bad);
        const auto rejected = call(runtime, "gpu.probe", {{"batches", invalid}});
        DEBUG_REQUIRE(frame.begin(2));
        runtime.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 10}, nullptr);
        runtime.boundary(commands, "Early", 0, "Probe", bindings, DebugValue::object()); runtime.endExecution(true);
        const char* expectedErrors[] = {"LayoutMismatch", "LayoutMismatch", "OutOfRange", "TypeMismatch", "StaleHandle"};
        for (size_t i = 0; i < 5; ++i) {
            const auto status = call(runtime, "jobs.get", {{"job", rejected["result"]["jobs"][i]["job"]}});
            if (status["result"]["error"]["code"] != expectedErrors[i] || status["result"]["reservedBytes"] != 0) {
                return RHITestResult::fail("Invalid probe was not rejected atomically: " + status.dump());
            }
        }
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto cancelJob = call(runtime, "gpu.probe", specs)["result"]["job"];
        DEBUG_REQUIRE(frame.begin(3));
        runtime.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 11}, nullptr);
        runtime.boundary(commands, "Early", 0, "Probe", bindings, DebugValue::object()); runtime.endExecution(true);
        DEBUG_REQUIRE(frame.submit());
        const auto cancelled = call(runtime, "jobs.cancel", {{"job", cancelJob}});
        if (cancelled["result"]["reservedBytes"] == 0) { return RHITestResult::fail("Cancelled GPU probe released its reservation early"); }
        DEBUG_REQUIRE(frame.frame.wait()); runtime.poll();
        const auto cancelDone = call(runtime, "jobs.get", {{"job", cancelJob}});
        if (cancelDone["result"]["state"] != "Cancelled" || cancelDone["result"]["reservedBytes"] != 0) {
            return RHITestResult::fail("Cancelled GPU probe leaked its reservation");
        }
        debug::DebugLimits tinyLimits; tinyLimits.probeScanBytes = 4;
        RenderDebugRuntime tiny(tinyLimits);
        tiny.compiled({{"id", "probe-graph"}, {"generation", 1}, {"passes", {{{"name", "Probe"}, {"active", true}, {"checkpoints", {"Early"}}}}}});
        const auto budgetJob = call(tiny, "gpu.probe", specs)["result"]["job"];
        DEBUG_REQUIRE(frame.begin(4));
        tiny.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 12}, nullptr);
        tiny.boundary(commands, "Early", 0, "Probe", bindings, DebugValue::object()); tiny.endExecution(true);
        if (call(tiny, "jobs.get", {{"job", budgetJob}})["result"]["error"]["code"] != "BudgetExceeded") {
            return RHITestResult::fail("GPU scan budget was not enforced");
        }
        DEBUG_REQUIRE(frame.submit()); DEBUG_REQUIRE(frame.frame.wait()); tiny.poll(); tiny.drain();
        // The same probe must work on the actual compute queue, with shared
        // source/result allocations and that queue's tracked completion.
        if (auto* computeQueue = device->getQueue(QueueType::Compute)) {
            FrameCommands computeFrame;
            DEBUG_REQUIRE(computeFrame.initialize(*device, *computeQueue));
            auto computeSpec = specs;
            computeSpec["probes"] = DebugValue::array({specs["probes"][2]});
            const auto computeJob = call(runtime, "gpu.probe", computeSpec)["result"]["job"];
            DEBUG_REQUIRE(computeFrame.begin(5)); computeFrame.commands->hostWriteBarrier();
            runtime.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 13}, nullptr);
            runtime.boundary(*computeFrame.commands, "Early", 0, "Probe", bindings, DebugValue::object()); runtime.endExecution(true);
            DEBUG_REQUIRE(computeFrame.submit()); DEBUG_REQUIRE(computeFrame.frame.wait()); runtime.poll();
            const auto completed = call(runtime, "jobs.get", {{"job", computeJob}});
            if (completed["result"]["state"] != "Ready" || completed["result"]["probes"]["finite"]["matchedCount"] != 3) {
                return RHITestResult::fail("Compute queue probe failed: " + completed.dump());
            }
        }
        if (auto* copyQueue = device->getQueue(QueueType::Copy)) {
            FrameCommands copyFrame;
            DEBUG_REQUIRE(copyFrame.initialize(*device, *copyQueue));
            if (!(uint32_t(copyFrame.commands->queueCapabilities()) & uint32_t(QueueAccessBits::Compute))) {
                const auto copyJob = call(runtime, "gpu.probe", specs)["result"]["job"];
                DEBUG_REQUIRE(copyFrame.begin(6));
                runtime.beginExecution(*device, {.graph = "probe-graph", .generation = 1, .execution = 14}, nullptr);
                runtime.boundary(*copyFrame.commands, "Early", 0, "Probe", bindings, DebugValue::object()); runtime.endExecution(true);
                if (call(runtime, "jobs.get", {{"job", copyJob}})["result"]["error"]["code"] != "Unsupported") {
                    return RHITestResult::fail("Transfer-only queue accepted a compute probe");
                }
                DEBUG_REQUIRE(copyFrame.commands->end()); copyFrame.frame.cancel(); runtime.poll();
            }
        }
        const auto events = runtime.core().events("validation");
        for (const auto& event : events["events"]) {
            if (event["severity"].get<uint32_t>() & 4096) { return RHITestResult::fail("Probe validation error: " + event.dump()); }
        }
        runtime.drain();
        return RHITestResult::pass("GPU count, OOB, NaN/Inf, extrema, packed fields, checkpoint isolation and watch verified");
    }
};

METALLIC_REGISTER_RHI_TEST(DebugControlGPUProbeTest);
METALLIC_REGISTER_RHI_TEST(DebugControlCaptureTest);
METALLIC_REGISTER_RHI_TEST(DebugControlTextureTest);
} // namespace
} // namespace metallic::tests
