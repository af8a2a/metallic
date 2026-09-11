#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"

#include <cmath>

namespace metallic::tests {
namespace {

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
        auto result = context.graphicsQueue.calibrateTimestamps(first);
        if (!result && result.error() == render::Error::Unsupported) {
            return RhiTestResult::skip("calibrated device/host timestamps unavailable");
        }
        if (!result) { return RhiTestResult::fail(toString(result)); }
        render::GpuClockCalibration second;
        result = context.graphicsQueue.calibrateTimestamps(second);
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
        result = context.device.createCommandPool(context.graphicsQueue, pool);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        std::unique_ptr<render::CommandBuffer> commands;
        result = pool->createCommandBuffer(commands);
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

} // namespace
} // namespace metallic::tests
