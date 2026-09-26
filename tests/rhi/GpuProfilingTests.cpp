#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderGraph/RenderGraphGpuLabels.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"

#include <cmath>
#include <stdexcept>

namespace metallic::tests {
namespace {

class NsightGpuTraceStateTest final : public RhiTest {
public:
    NsightGpuTraceStateTest() { type = RhiTestType::Command; name = "nsight_gpu_trace_requires_injection"; }
    RhiTestResult run(RhiTestContext&) override
    {
        std::string error;
        if (render::profiling::endExternalNsightGpuTrace(error) || error.empty()) {
            return RhiTestResult::fail("Stop before start must fail without calling an uninitialized SDK function");
        }
        if (render::profiling::beginExternalNsightGpuTrace(error) || error.empty()) {
            std::string cleanup;
            render::profiling::endExternalNsightGpuTrace(cleanup);
            return RhiTestResult::fail("An ordinary test process must not start GPU Trace without injection");
        }
        if (render::profiling::endExternalNsightGpuTrace(error) || error.empty()) {
            return RhiTestResult::fail("Failed start must not leave an active trace");
        }
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(NsightGpuTraceStateTest);

// Check the native label protocol, including balanced recordings and inherited
// pass names on both queues. GPU fork/join tests separately exercise submission.
struct LabelRecording {
    std::vector<std::string> stack;
    std::vector<std::vector<std::string>> paths;
    bool ended = false;
    void beginDebugLabel(const render::DebugLabelDesc& desc)
    {
        if (ended) { throw std::runtime_error("label emitted after command buffer end"); }
        stack.emplace_back(desc.name);
        paths.push_back(stack);
    }
    void endDebugLabel()
    {
        if (ended || stack.empty()) { throw std::runtime_error("unbalanced native label end"); }
        stack.pop_back();
    }
    void finish()
    {
        if (!stack.empty()) { throw std::runtime_error("scope crossed a command buffer boundary"); }
        ended = true;
    }
};

class GpuProfileLabelsTest final : public RhiTest {
public:
    GpuProfileLabelsTest() { type = RhiTestType::Command; name = "gpu_profiling_label_recordings"; }
    RhiTestResult run(RhiTestContext&) override
    {
        try {
            for (int fail : {0, 1, 2}) {
                LabelRecording producer, compute, graphics, join;
                render::detail::RenderGraphGpuLabels<LabelRecording> labels("VBuffer", {});
                labels.resume(producer);
                labels.begin("Stream traversal", {});
                labels.begin("LOD frontier", {});
                labels.end(); labels.end();
                labels.begin("Visibility raster", {});
                labels.begin("Stream early", {});
                labels.suspend(); producer.finish();
                labels.resume(compute);
                labels.begin("Software raster", {});
                labels.end(); labels.suspend(); compute.finish();
                const std::vector<std::string> softwarePath{"VBuffer", "Visibility raster", "Stream early", "Software raster"};
                if (compute.paths.back() != softwarePath) { throw std::runtime_error("compute lost inherited scope names"); }
                if (fail != 1) {
                    labels.resume(graphics);
                    labels.begin("Hardware raster", {});
                    labels.end(); labels.suspend(); graphics.finish();
                    const std::vector<std::string> hardwarePath{"VBuffer", "Visibility raster", "Stream early", "Hardware raster"};
                    if (graphics.paths.back() != hardwarePath) { throw std::runtime_error("graphics lost inherited scope names"); }
                }
                if (fail == 0) {
                    labels.resume(join);
                    labels.begin("Raster merge", {});
                    labels.end();
                    const std::vector<std::string> mergePath{"VBuffer", "Visibility raster", "Stream early", "Raster merge"};
                    if (join.paths.back() != mergePath) { throw std::runtime_error("join lost inherited scope names"); }
                }
                // Failed branches have no join recording: logical RAII unwind
                // must not issue label commands to the ended producer/branch.
                labels.end(); labels.end();
                if (fail == 0) {
                    labels.begin("Stream End", {}); labels.end();
                    if (join.paths.back() != std::vector<std::string>{"VBuffer", "Stream End"}) {
                        throw std::runtime_error("stream cleanup incorrectly nested under raster");
                    }
                    labels.suspend(); join.finish();
                }
                if (producer.paths[2] != std::vector<std::string>{"VBuffer", "Stream traversal", "LOD frontier"}) {
                    throw std::runtime_error("traversal labels missing before raster");
                }
            }
        } catch (const std::exception& error) {
            return RhiTestResult::fail(error.what());
        }
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(GpuProfileLabelsTest);

class GpuClockCalibrationTest : public RhiTest {
public:
    GpuClockCalibrationTest()
    {
        type = RhiTestType::Command;
        name = "gpu_clock_calibration";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::GpuClockCalibration first;
        auto result = context.graphicsQueue.calibrateTimestamps().transform([&](auto rhiValue) { first = std::move(rhiValue); });
        if (!result && result.error() == render::Error::Unsupported) {
            return RhiTestResult::skip("calibrated device/host timestamps unavailable");
        }
        if (!result) { return RhiTestResult::fail(toString(result)); }
        render::GpuClockCalibration second;
        result = context.graphicsQueue.calibrateTimestamps().transform([&](auto rhiValue) { second = std::move(rhiValue); });
        if (!result || first.cpuNanoseconds == 0 ||
            second.cpuNanoseconds < first.cpuNanoseconds || second.gpuTimestamp < first.gpuTimestamp) {
            return RhiTestResult::fail("calibrated clock samples are invalid or run backwards");
        }
        return RhiTestResult::pass();
    }
};

class RenderGraphGpuProfilingTest : public RhiTest {
public:
    RenderGraphGpuProfilingTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_gpu_profiling";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RhiTestResult::skip("timestamp queries unavailable");
        }
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        auto graph = render::RenderGraph::createDefaultTriangleGraph();
        graph.addNode("CopyColorPass", "Copy");
        graph.addEdge("Triangle.color", "Copy.source");
        graph.unmarkOutput("Triangle.color");
        graph.markOutput("Copy.color");
        // More than two complete wraps of the query ring, then a recompile.
        for (uint32_t frame = 0; frame < 10; ++frame) {
            const uint32_t size = frame < 8 ? 64 : 96;
            result = preview.render(graph, size, size);
            if (!result) { return RhiTestResult::fail(toString(result)); }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = preview.collectCompletedGpuExecutionStats(completed);
            if (!result || completed.size() != 1) {
                return RhiTestResult::fail("completed frame did not resolve exactly one GPU timing sample");
            }
            const auto& stats = completed.front();
            if (!stats.gpuTimingAvailable || !std::isfinite(stats.gpuMilliseconds) ||
                stats.gpuMilliseconds < 0.0 || stats.nodes.empty()) {
                return RhiTestResult::fail("frame GPU timing unavailable or invalid");
            }
            double passMilliseconds = 0.0;
            for (const auto& pass : stats.nodes) {
                if (!pass.gpuTimingAvailable || !std::isfinite(pass.gpuMilliseconds) ||
                    pass.gpuMilliseconds < 0.0) {
                    return RhiTestResult::fail("pass GPU timing unavailable or invalid");
                }
                passMilliseconds += pass.gpuMilliseconds;
            }
            if (passMilliseconds > stats.gpuMilliseconds + 0.001) {
                return RhiTestResult::fail("frame GPU interval does not enclose its passes");
            }
            completed.clear();
            result = preview.collectCompletedGpuExecutionStats(completed);
            if (!result || !completed.empty()) {
                return RhiTestResult::fail("GPU timing sample was published twice");
            }
        }
        return RhiTestResult::pass();
    }
};

class ProfileBudgetPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(16).transferWrite();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto parent = context.profileScope("Parent");
        for (uint32_t i = 0; i < context.properties().value("scopes", 300u); ++i) {
            auto scope = context.profileScope("Child");
        }
        return {};
    }
};

class GpuProfileBudgetTest final : public RhiTest {
public:
    GpuProfileBudgetTest() { type = RhiTestType::Command; name = "gpu_profiling_scope_budget"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) { return RhiTestResult::skip("timestamp queries unavailable"); }
        render::registerRenderGraphPassType("ProfileBudgetPass", "Profiling scope budget test", [] { return std::make_unique<ProfileBudgetPass>(); });
        render::RenderGraph graph;
        const auto first = graph.addNode("ProfileBudgetPass", "First")->id;
        graph.addNode("ProfileBudgetPass", "Tail", {{"scopes", 1}});
        graph.markOutput("First.data"); graph.markOutput("Tail.data");
        render::RenderGraphExecutor executor;
        std::string log;
        auto result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) { return RhiTestResult::fail(log); }
        for (uint32_t f = 0; f < 8; ++f) {
            const bool overflow = f % 2 == 0;
            graph.setNodeRuntimeProperty(first, "scopes", overflow ? 300 : 2);
            executor.syncRuntimeProperties(graph);
            result = executor.execute({.graphicsQueue = &context.graphicsQueue});
            if (result) { result = executor.waitForSubmittedWork(5'000'000'000ull); }
            if (!result) { return RhiTestResult::fail(toString(result)); }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = executor.collectCompletedGpuExecutionStats(completed);
            if (!result || completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow != overflow) {
                return RhiTestResult::fail("scope overflow lost frame timing or contaminated a later ring slot");
            }
            const auto& frame = completed[0];
            // The executor contributes the final Upload flush scope as well.
            if (frame.nodes[0].sections.size() != (overflow ? 256u : 4u)) { return RhiTestResult::fail("scope metadata did not respect budget"); }
            for (const auto& node : frame.nodes) {
                if (!node.gpuTimingAvailable) { return RhiTestResult::fail("scope budget lost pass timing"); }
                for (const auto& section : node.sections) {
                    if (section.parent != UINT32_MAX && section.parent >= node.sections.size()) { return RhiTestResult::fail("invalid parent after scope overflow"); }
                    if (!overflow && !section.gpuTimingAvailable) { return RhiTestResult::fail("normal frame scope timing unavailable after overflow"); }
                }
            }
        }
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(GpuProfileBudgetTest);

METALLIC_REGISTER_RHI_TEST(GpuClockCalibrationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGpuProfilingTest);

class CancelledGpuProfilingTest : public RhiTest {
public:
    CancelledGpuProfilingTest()
    {
        type = RhiTestType::Command;
        name = "cancelled_gpu_profiling";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RhiTestResult::skip("timestamp queries unavailable");
        }
        render::RenderGraphExecutor executor;
        auto graph = render::RenderGraph::createDefaultTriangleGraph();
        std::string log;
        auto result = executor.compile(context.device, graph, 32, 32, log);
        if (!result) { return RhiTestResult::fail(log); }
        std::unique_ptr<render::CommandPool> pool;
        result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        if (!result) { return RhiTestResult::fail(toString(result)); }
        std::unique_ptr<render::CommandBuffer> commands;
        result = pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); });
        if (!result) { return RhiTestResult::fail(toString(result)); }
        render::RenderFrameContext frame;
        render::QueueSubmissionTracker tracker;
        result = tracker.initialize(context.device, context.graphicsQueue);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        for (uint64_t index = 0; index < 6; ++index) {
            result = frame.begin(index);
            if (!result) { return RhiTestResult::fail(toString(result)); }
            result = pool->reset();
            if (!result) { return RhiTestResult::fail(toString(result)); }
            result = commands->begin(&frame);
            if (!result) { return RhiTestResult::fail(toString(result)); }
            result = executor.execute(*commands);
            if (!result) { return RhiTestResult::fail(toString(result)); }
            result = commands->end();
            if (!result) { return RhiTestResult::fail(toString(result)); }
            const bool submit = index == 0 || index == 5;
            if (submit) {
                render::CommandBuffer* buffers[] = {commands.get()};
                result = tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame);
                if (!result) { return RhiTestResult::fail(toString(result)); }
                result = frame.wait(5'000'000'000ull);
                if (!result) { return RhiTestResult::fail(toString(result)); }
            } else {
                frame.cancel();
            }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = executor.collectCompletedGpuExecutionStats(completed);
            if (!result || completed.size() != (submit ? 1u : 0u)) {
                return RhiTestResult::fail("cancelled recording published stale timing or leaked a query slot");
            }
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(CancelledGpuProfilingTest);

class ExternalGpuProfilingCompletionTest final : public RhiTest {
public:
    ExternalGpuProfilingCompletionTest()
    {
        type = RhiTestType::Command;
        name = "gpu_profiling_external_requires_completion";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RhiTestResult::skip("timestamp queries unavailable");
        }
        render::registerRenderGraphPassType("ProfileBudgetPass", "Profiling scope budget test",
            [] { return std::make_unique<ProfileBudgetPass>(); });
        render::RenderGraph graph;
        graph.addNode("ProfileBudgetPass", "Profile", {{"scopes", 2u}});
        graph.markOutput("Profile.data");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(context.device, graph, 1, 1, log)) { return RhiTestResult::fail(log); }
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        std::unique_ptr<render::Fence> fence;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !context.device.createFence(false).transform([&](auto value) { fence = std::move(value); })) {
            return RhiTestResult::fail("external profiling setup failed");
        }
        const auto validSample = [](const render::RenderGraphExecutionStats& stats, uint64_t execution) {
            return stats.executionId == execution && stats.gpuTimingAvailable && std::isfinite(stats.gpuMilliseconds) &&
                stats.gpuMilliseconds >= 0.0 && stats.nodes.size() == 1 && stats.nodes.front().gpuTimingAvailable;
        };
        // Wrap the query ring while alternating accepted frames and discarded
        // raw recordings. Frame-associated cancellation has a separate test.
        for (uint32_t index = 0; index < 6; ++index) {
            if (!executor.execute({.graphicsQueue = &context.graphicsQueue}) ||
                !executor.waitForSubmittedWork(5'000'000'000ull)) {
                return RhiTestResult::fail("self-submitted profiling frame failed");
            }
            std::vector<render::RenderGraphExecutionStats> completed;
            if (!executor.collectCompletedGpuExecutionStats(completed) || completed.size() != 1 ||
                !validSample(completed.front(), executor.executionStats().executionId)) {
                return RhiTestResult::fail("query ring lost the self-submitted sample after an external recording");
            }
            if (!pool->reset() || !commands->begin() || !executor.execute(*commands) || !commands->end()) {
                return RhiTestResult::fail("raw external profiling recording failed");
            }
            const auto& rawStats = executor.executionStats();
            if (rawStats.gpuTimingAvailable || rawStats.nodes.size() != 1 ||
                rawStats.nodes.front().gpuTimingAvailable || rawStats.nodes.front().sections.empty() ||
                !std::isfinite(rawStats.cpuMilliseconds) || rawStats.cpuMilliseconds < 0.0) {
                return RhiTestResult::fail("raw external recording lost CPU scopes or advertised GPU timing without completion");
            }
            completed.clear();
            if (!executor.collectCompletedGpuExecutionStats(completed) || !completed.empty()) {
                return RhiTestResult::fail("unsubmitted timestamps were queried or published");
            }
            if (index % 3u != 0u) {
                if (!pool->reset() || !executor.collectCompletedGpuExecutionStats(completed) || !completed.empty()) {
                    return RhiTestResult::fail("cancelled raw timestamps were queried or published");
                }
                continue;
            }
            std::unique_ptr<render::Semaphore> gate;
            if (!context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); })) {
                return RhiTestResult::fail("external completion gate creation failed");
            }
            const render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
            render::CommandBuffer* submitted[] = {commands.get()};
            if (!fence->reset() || !context.graphicsQueue.submit({.waitSemaphores = &wait,
                    .waitSemaphoreCount = 1, .commandBuffers = submitted,
                    .commandBufferCount = 1, .signalFence = fence.get()})) {
                return RhiTestResult::fail("raw profiling submission failed");
            }
            // Queue acceptance is observable while the unsignaled host gate
            // keeps execution incomplete. Never infer completion from query
            // availability, which can still belong to a prior reset generation.
            const bool incomplete = !fence->isSignaled();
            const auto collected = executor.collectCompletedGpuExecutionStats(completed);
            const auto released = gate->signal(1);
            if (!released || !fence->wait(5'000'000'000ull)) {
                (void)context.graphicsQueue.waitIdle();
                return RhiTestResult::fail("accepted raw profiling completion failed");
            }
            if (!incomplete || !collected || !completed.empty()) {
                return RhiTestResult::fail("accepted but incomplete raw recording published GPU timestamps");
            }
            if (!executor.collectCompletedGpuExecutionStats(completed) || !completed.empty()) {
                return RhiTestResult::fail("raw recording without an exposed completion published GPU timestamps");
            }
        }
        return RhiTestResult::pass("raw recordings retain CPU scopes; framed execution owns GPU timing completion");
    }
};

METALLIC_REGISTER_RHI_TEST(ExternalGpuProfilingCompletionTest);

} // namespace
} // namespace metallic::tests
