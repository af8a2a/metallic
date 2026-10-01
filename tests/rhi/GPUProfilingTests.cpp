#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderGraph/RenderGraphGPULabels.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"

#include <cmath>
#include <stdexcept>

namespace metallic::tests {
namespace {

class NsightGPUTraceStateTest final : public RHITest {
public:
    NsightGPUTraceStateTest() { type = RHITestType::Command; name = "nsight_gpu_trace_requires_injection"; }
    RHITestResult run(RHITestContext&) override
    {
        std::string error;
        if (render::profiling::endExternalNsightGpuTrace(error) || error.empty()) {
            return RHITestResult::fail("Stop before start must fail without calling an uninitialized SDK function");
        }
        if (render::profiling::beginExternalNsightGpuTrace(error) || error.empty()) {
            std::string cleanup;
            render::profiling::endExternalNsightGpuTrace(cleanup);
            return RHITestResult::fail("An ordinary test process must not start GPU Trace without injection");
        }
        if (render::profiling::endExternalNsightGpuTrace(error) || error.empty()) {
            return RHITestResult::fail("Failed start must not leave an active trace");
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(NsightGPUTraceStateTest);

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

class GPUProfileLabelsTest final : public RHITest {
public:
    GPUProfileLabelsTest() { type = RHITestType::Command; name = "gpu_profiling_label_recordings"; }
    RHITestResult run(RHITestContext&) override
    {
        try {
            for (int fail : {0, 1, 2}) {
                LabelRecording producer, compute, graphics, join;
                render::detail::RenderGraphGPULabels<LabelRecording> labels("VBuffer", {});
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
            return RHITestResult::fail(error.what());
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(GPUProfileLabelsTest);

class GPUClockCalibrationTest : public RHITest {
public:
    GPUClockCalibrationTest()
    {
        type = RHITestType::Command;
        name = "gpu_clock_calibration";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::GPUClockCalibration first;
        auto result = context.graphicsQueue.calibrateTimestamps().transform([&](auto rhiValue) { first = std::move(rhiValue); });
        if (!result && result.error() == render::Error::Unsupported) {
            return RHITestResult::skip("calibrated device/host timestamps unavailable");
        }
        if (!result) { return RHITestResult::fail(toString(result)); }
        render::GPUClockCalibration second;
        result = context.graphicsQueue.calibrateTimestamps().transform([&](auto rhiValue) { second = std::move(rhiValue); });
        if (!result || first.cpuNanoseconds == 0 ||
            second.cpuNanoseconds < first.cpuNanoseconds || second.gpuTimestamp < first.gpuTimestamp) {
            return RHITestResult::fail("calibrated clock samples are invalid or run backwards");
        }
        return RHITestResult::pass();
    }
};

class RenderGraphGPUProfilingTest : public RHITest {
public:
    RenderGraphGPUProfilingTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_gpu_profiling";
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RHITestResult::skip("timestamp queries unavailable");
        }
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation);
        if (!result) { return RHITestResult::fail(toString(result)); }
        auto graph = render::RenderGraph::createDefaultTriangleGraph();
        graph.addNode("CopyColorPass", "Copy");
        graph.addEdge("Triangle.color", "Copy.source");
        graph.unmarkOutput("Triangle.color");
        graph.markOutput("Copy.color");
        // More than two complete wraps of the query ring, then a recompile.
        for (uint32_t frame = 0; frame < 10; ++frame) {
            const uint32_t size = frame < 8 ? 64 : 96;
            result = preview.render(graph, size, size);
            if (!result) { return RHITestResult::fail(toString(result)); }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); });
            if (!result || completed.size() != 1) {
                return RHITestResult::fail("completed frame did not resolve exactly one GPU timing sample");
            }
            const auto& stats = completed.front();
            if (!stats.gpuTimingAvailable || !std::isfinite(stats.gpuMilliseconds) ||
                stats.gpuMilliseconds < 0.0 || stats.nodes.empty()) {
                return RHITestResult::fail("frame GPU timing unavailable or invalid");
            }
            double passMilliseconds = 0.0;
            for (const auto& pass : stats.nodes) {
                if (!pass.gpuTimingAvailable || !std::isfinite(pass.gpuMilliseconds) ||
                    pass.gpuMilliseconds < 0.0) {
                    return RHITestResult::fail("pass GPU timing unavailable or invalid");
                }
                passMilliseconds += pass.gpuMilliseconds;
            }
            if (passMilliseconds > stats.gpuMilliseconds + 0.001) {
                return RHITestResult::fail("frame GPU interval does not enclose its passes");
            }
            completed.clear();
            result = preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); });
            if (!result || !completed.empty()) {
                return RHITestResult::fail("GPU timing sample was published twice");
            }
        }
        return RHITestResult::pass();
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

class GPUProfileBudgetTest final : public RHITest {
public:
    GPUProfileBudgetTest() { type = RHITestType::Command; name = "gpu_profiling_scope_budget"; }
    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) { return RHITestResult::skip("timestamp queries unavailable"); }
        render::registerRenderGraphPassType("ProfileBudgetPass", "Profiling scope budget test", [] { return std::make_unique<ProfileBudgetPass>(); });
        render::RenderGraph graph;
        const auto first = graph.addNode("ProfileBudgetPass", "First")->id;
        graph.addNode("ProfileBudgetPass", "Tail", {{"scopes", 1}});
        graph.markOutput("First.data"); graph.markOutput("Tail.data");
        render::RenderGraphExecutor executor;
        std::string log;
        auto result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) { return RHITestResult::fail(log); }
        for (uint32_t f = 0; f < 8; ++f) {
            const bool overflow = f % 2 == 0;
            graph.setNodeRuntimeProperty(first, "scopes", overflow ? 300 : 2);
            executor.syncRuntimeProperties(graph);
            result = executor.execute({.graphicsQueue = &context.graphicsQueue});
            if (result) { result = executor.waitForSubmittedWork(5'000'000'000ull); }
            if (!result) { return RHITestResult::fail(toString(result)); }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); });
            if (!result || completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow != overflow) {
                return RHITestResult::fail("scope overflow lost frame timing or contaminated a later ring slot");
            }
            const auto& frame = completed[0];
            // The executor contributes the final Upload flush scope as well.
            if (frame.nodes[0].sections.size() != (overflow ? 256u : 4u)) { return RHITestResult::fail("scope metadata did not respect budget"); }
            for (const auto& node : frame.nodes) {
                if (!node.gpuTimingAvailable) { return RHITestResult::fail("scope budget lost pass timing"); }
                for (const auto& section : node.sections) {
                    if (section.parent != UINT32_MAX && section.parent >= node.sections.size()) { return RHITestResult::fail("invalid parent after scope overflow"); }
                    if (!overflow && !section.gpuTimingAvailable) { return RHITestResult::fail("normal frame scope timing unavailable after overflow"); }
                }
            }
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(GPUProfileBudgetTest);

METALLIC_REGISTER_RHI_TEST(GPUClockCalibrationTest);
METALLIC_REGISTER_RHI_TEST(RenderGraphGPUProfilingTest);

class CancelledGPUProfilingTest : public RHITest {
public:
    CancelledGPUProfilingTest()
    {
        type = RHITestType::Command;
        name = "cancelled_gpu_profiling";
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RHITestResult::skip("timestamp queries unavailable");
        }
        render::RenderGraphExecutor executor;
        auto graph = render::RenderGraph::createDefaultTriangleGraph();
        std::string log;
        auto result = executor.compile(context.device, graph, 32, 32, log);
        if (!result) { return RHITestResult::fail(log); }
        std::unique_ptr<render::CommandPool> pool;
        result = context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        std::unique_ptr<render::CommandBuffer> commands;
        result = pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        render::RenderFrameContext frame;
        render::QueueSubmissionTracker tracker;
        result = tracker.initialize(context.device, context.graphicsQueue);
        if (!result) { return RHITestResult::fail(toString(result)); }
        for (uint64_t index = 0; index < 6; ++index) {
            result = frame.begin(index);
            if (!result) { return RHITestResult::fail(toString(result)); }
            result = pool->reset();
            if (!result) { return RHITestResult::fail(toString(result)); }
            result = commands->begin(&frame);
            if (!result) { return RHITestResult::fail(toString(result)); }
            result = executor.execute(*commands);
            if (!result) { return RHITestResult::fail(toString(result)); }
            result = commands->end();
            if (!result) { return RHITestResult::fail(toString(result)); }
            const bool submit = index == 0 || index == 5;
            if (submit) {
                render::CommandBuffer* buffers[] = {commands.get()};
                result = tracker.submit({.commandBuffers = {buffers, 1}}, frame);
                if (!result) { return RHITestResult::fail(toString(result)); }
                result = frame.wait(5'000'000'000ull);
                if (!result) { return RHITestResult::fail(toString(result)); }
            } else {
                frame.cancel();
            }
            std::vector<render::RenderGraphExecutionStats> completed;
            result = executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); });
            if (!result || completed.size() != (submit ? 1u : 0u)) {
                return RHITestResult::fail("cancelled recording published stale timing or leaked a query slot");
            }
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(CancelledGPUProfilingTest);

class ExternalGPUProfilingCompletionTest final : public RHITest {
public:
    ExternalGPUProfilingCompletionTest()
    {
        type = RHITestType::Command;
        name = "gpu_profiling_external_requires_completion";
    }

    RHITestResult run(RHITestContext& context) override
    {
        if (!context.device.capabilities().timestampQueries) {
            return RHITestResult::skip("timestamp queries unavailable");
        }
        render::registerRenderGraphPassType("ProfileBudgetPass", "Profiling scope budget test",
            [] { return std::make_unique<ProfileBudgetPass>(); });
        render::RenderGraph graph;
        graph.addNode("ProfileBudgetPass", "Profile", {{"scopes", 2u}});
        graph.markOutput("Profile.data");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(context.device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        std::unique_ptr<render::Fence> fence;
        if (!context.device.createCommandPool(context.graphicsQueue).transform([&](auto value) { pool = std::move(value); }) ||
            !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
            !context.device.createFence(false).transform([&](auto value) { fence = std::move(value); })) {
            return RHITestResult::fail("external profiling setup failed");
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
                return RHITestResult::fail("self-submitted profiling frame failed");
            }
            std::vector<render::RenderGraphExecutionStats> completed;
            if (!executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); }) || completed.size() != 1 ||
                !validSample(completed.front(), executor.executionStats().executionId)) {
                return RHITestResult::fail("query ring lost the self-submitted sample after an external recording");
            }
            if (!pool->reset() || !commands->begin() || !executor.execute(*commands) || !commands->end()) {
                return RHITestResult::fail("raw external profiling recording failed");
            }
            const auto& rawStats = executor.executionStats();
            if (rawStats.gpuTimingAvailable || rawStats.nodes.size() != 1 ||
                rawStats.nodes.front().gpuTimingAvailable || rawStats.nodes.front().sections.empty() ||
                !std::isfinite(rawStats.cpuMilliseconds) || rawStats.cpuMilliseconds < 0.0) {
                return RHITestResult::fail("raw external recording lost CPU scopes or advertised GPU timing without completion");
            }
            completed.clear();
            if (!executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); }) || !completed.empty()) {
                return RHITestResult::fail("unsubmitted timestamps were queried or published");
            }
            if (index % 3u != 0u) {
                if (!pool->reset() || !executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); }) || !completed.empty()) {
                    return RHITestResult::fail("cancelled raw timestamps were queried or published");
                }
                continue;
            }
            std::unique_ptr<render::Semaphore> gate;
            if (!context.device.createSemaphore().transform([&](auto value) { gate = std::move(value); })) {
                return RHITestResult::fail("external completion gate creation failed");
            }
            const render::SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
            render::CommandBuffer* submitted[] = {commands.get()};
            if (!fence->reset() || !context.graphicsQueue.submit({
                .waitSemaphores = {&wait, 1},
                .commandBuffers = {submitted, 1},
                .signalFence = fence.get(),
            })) {
                return RHITestResult::fail("raw profiling submission failed");
            }
            // Queue acceptance is observable while the unsignaled host gate
            // keeps execution incomplete. Never infer completion from query
            // availability, which can still belong to a prior reset generation.
            const bool incomplete = !fence->isSignaled();
            const auto collected = executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); });
            const auto released = gate->signal(1);
            if (!released || !fence->wait(5'000'000'000ull)) {
                (void)context.graphicsQueue.waitIdle();
                return RHITestResult::fail("accepted raw profiling completion failed");
            }
            if (!incomplete || !collected || !completed.empty()) {
                return RHITestResult::fail("accepted but incomplete raw recording published GPU timestamps");
            }
            if (!executor.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); }) || !completed.empty()) {
                return RHITestResult::fail("raw recording without an exposed completion published GPU timestamps");
            }
        }
        return RHITestResult::pass("raw recordings retain CPU scopes; framed execution owns GPU timing completion");
    }
};

METALLIC_REGISTER_RHI_TEST(ExternalGPUProfilingCompletionTest);

} // namespace
} // namespace metallic::tests
