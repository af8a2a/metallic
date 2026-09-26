#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Task/TaskSystem.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

Result<> verifyRed(RenderGraphExecutor& executor, RhiTestContext& context, const std::string& output)
{
    RenderFrameContext frame;
    CommandRecordingContext recording;
    QueueSubmissionTracker tracker;
    std::unique_ptr<Buffer> readback;
    auto result = context.device.createBuffer({.size = 32 * 32 * 4, .usage = BufferUsageBits::TransferDestination,
        .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readback = std::move(value); });
    if (result) { result = recording.initialize(context.device, context.graphicsQueue); }
    if (result) { result = tracker.initialize(context.device, context.graphicsQueue); }
    if (result) { result = frame.begin(0); }
    CommandBuffer* commands = nullptr;
    if (result) { result = recording.prepare(frame).transform([&](auto value) { commands = value; }); }
    if (result) { result = executor.transitionOutput(*commands, output, ResourceState::TransferSource); }
    if (result) {
        commands->copyTextureToBuffer({.texture = executor.outputResource(output)->texture, .buffer = readback.get(),
            .width = 32, .height = 32});
        result = commands->end();
    }
    if (result) { result = tracker.submit({.commandBuffers = &commands, .commandBufferCount = 1}, frame); }
    if (result) { result = frame.wait(5'000'000'000ull); }
    if (!result) { return result; }
    readback->invalidate();
    auto* pixels = static_cast<const uint8_t*>(readback->map());
    bool correct = pixels != nullptr;
    if (pixels) {
        for (uint32_t i = 0; i < 32 * 32; ++i) {
            correct &= pixels[i * 4] == 255 && pixels[i * 4 + 1] == 0 && pixels[i * 4 + 2] == 0 && pixels[i * 4 + 3] == 255;
        }
        readback->unmap();
    }
    return correct ? Result<>{} : makeError(Error::Failure);
}

class SchedulingDiagnosticsTest final : public RhiTest {
public:
    SchedulingDiagnosticsTest() { type = RhiTestType::Rendering; name = "scheduling_diagnostics_and_benchmark"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        const bool benchmark = std::getenv("METALLIC_TEST_SCHEDULING_BENCHMARK") != nullptr;
        const uint32_t repeats = benchmark ? 3 : 1;
        const uint32_t warmup = benchmark ? 30 : 0;
        const uint32_t frames = benchmark ? 120 : 2;
        const auto* tasks = task::tryGetTaskSystem();
        if (!tasks || tasks->workerCount() < 4) { return RhiTestResult::skip("four workers required"); }
        struct Mode { uint32_t workers, workload; FrameSubmissionMode submission; bool diagnostics; };
        const std::array modes{
            Mode{1, 4, FrameSubmissionMode::Joined, true}, Mode{4, 4, FrameSubmissionMode::Joined, true},
            Mode{1, 4, FrameSubmissionMode::Pipelined, true}, Mode{4, 4, FrameSubmissionMode::Pipelined, true},
            Mode{4, 16, FrameSubmissionMode::Pipelined, true}, Mode{4, 4, FrameSubmissionMode::Pipelined, false}};
        Json rows = Json::array();
        for (uint32_t count : {16u, 64u}) {
            RenderGraph graph;
            graph.addNode("ClearColorPass", "Pass0", {{"color", {1.f, 0.f, 0.f, 1.f}}});
            for (uint32_t i = 1; i < count; ++i) {
                const auto name = "Pass" + std::to_string(i);
                graph.addNode("CopyColorPass", name);
                graph.addEdge("Pass" + std::to_string(i - 1) + ".color", name + ".source");
            }
            const auto output = "Pass" + std::to_string(count - 1) + ".color";
            graph.markOutput(output);
            RenderGraphExecutor executor;
            std::string log;
            auto result = executor.compile(context.device, graph, 32, 32, log);
            if (!result) { return RhiTestResult::fail(log); }
            for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
                // Rotate/reverse configurations to reduce systematic order bias.
                for (uint32_t step = 0; step < modes.size(); ++step) {
                    const uint32_t index = (repeat + (repeat % 2 ? uint32_t(modes.size()) - 1 - step : step)) % uint32_t(modes.size());
                    const auto& mode = modes[index];
                    struct Sample { RenderGraphExecutionStats stats; double wallMs; };
                    std::vector<Sample> samples;
                    samples.reserve(frames);
                    for (uint32_t frame = 0; frame < warmup + frames; ++frame) {
                        const auto begin = std::chrono::steady_clock::now();
                        result = executor.execute({.graphicsQueue = &context.graphicsQueue,
                            .recordingWorkerLimit = mode.workers, .recordingBatchWorkload = mode.workload,
                            .submissionMode = mode.submission, .schedulingDiagnostics = mode.diagnostics});
                        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                        if (!result) { return RhiTestResult::fail(toString(result)); }
                        const auto& stats = executor.executionStats();
                        const auto& metrics = stats.scheduling;
                        if (metrics.enabled != mode.diagnostics || (mode.diagnostics &&
                            (metrics.nativeSubmits != stats.submittedBatchCount || metrics.invalidScopes || metrics.renderingScopes != 1 ||
                             metrics.drawCalls != 0 || metrics.dispatchCalls != 0 || metrics.nativeSubmitNs > metrics.submitNs ||
                             metrics.firstPassSubmitNs < metrics.firstSubmitNs || metrics.firstPassSubmitNs > metrics.executeNs ||
                             metrics.recordingEndNs > metrics.executeNs || metrics.maxRenderingNs > metrics.renderingNs))) {
                            return RhiTestResult::fail("Scheduling metric domain/count invariant failed");
                        }
                        if (frame >= warmup) { samples.push_back({stats, wall}); }
                    }
                    result = executor.waitForSubmittedWork(5'000'000'000ull);
                    if (result) { result = verifyRed(executor, context, output); }
                    if (!result) { return RhiTestResult::fail(std::string("Benchmark readback: ") + toString(result)); }
                    for (uint32_t i = 0; i < samples.size(); ++i) {
                        const auto& sample = samples[i];
                        rows.push_back({{"passCount", count}, {"repeat", repeat}, {"mode", index}, {"frame", i},
                            {"workers", mode.workers}, {"batchWorkload", mode.workload}, {"wallMs", sample.wallMs},
                            {"pipelined", sample.stats.pipelinedSubmission}, {"scheduling", sample.stats.scheduling},
                            {"batches", sample.stats.submittedBatchCount}, {"workerBatches", sample.stats.recordingTaskCount}});
                    }
                }
            }
        }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream output(context.outputDirectory / "SchedulingBenchmark.json");
        output << Json{{"protocol", "scheduling-cpu-v1"}, {"warmupFrames", warmup}, {"timedFramesPerMode", frames},
            {"validation", context.enableValidation}, {"workerCount", tasks->workerCount()},
            {"scope", "CPU execute wall time, including slot wait; 32x32 red texture copy chain, output verified after each mode. No presentation."},
            {"rows", std::move(rows)}}.dump(2);
        return output.good() ? RhiTestResult::pass() : RhiTestResult::fail("Cannot save scheduling results");
    }
};
METALLIC_REGISTER_RHI_TEST(SchedulingDiagnosticsTest);
} // namespace
} // namespace metallic::tests
