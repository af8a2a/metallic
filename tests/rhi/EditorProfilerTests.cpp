#include "RhiTest.h"
#include "ImGuiTestSnapshot.h"
#include "Editor/EditorProfiler.h"
#include "Runtime/Render/RenderSample.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

void checkProfile(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

bool saveProfilerPanel(EditorProfiler& profiler, const char* tab, const std::filesystem::path& path,
    std::string& message, bool sortGpu);

class EditorProfilerHistoryTest final : public RhiTest {
public:
    EditorProfilerHistoryTest() { type = RhiTestType::Command; name = "editor_profiler_history"; }
    RhiTestResult run(RhiTestContext&) override
    {
        try {
            EditorProfiler profiler;
            RenderGraphExecutionStats stats{.executionId = 10, .graphGeneration = 1, .cpuMilliseconds = 2};
            stats.nodes.push_back({.id = 7, .name = "Visibility", .type = "VisibilityBufferPass", .cpuMilliseconds = 1});
            stats.preparation = {
                {.name = "Preflight", .cpuMilliseconds = 4, .cpuOnly = true},
                {.name = "Poll completions", .parent = 0, .cpuMilliseconds = 3, .cpuOnly = true},
                {.name = "Scene revisions", .parent = 0, .cpuMilliseconds = .1, .cpuOnly = true},
                {.name = "Submission slot wait", .cpuMilliseconds = .2, .cpuOnly = true}};
            stats.nodes[0].sections.push_back({.name = "Software", .queue = QueueType::Compute, .cpuMilliseconds = .1});
            stats.streaming.push_back({.passName = "Visibility", .assetPath = "scene-a", .generation = 1, .frameIndex = 10,
                .geometryUsedBytes = 1024, .geometryBudgetBytes = 4096, .totalPages = 100, .residentPages = 4});
            { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
            stats.executionId = 11; stats.streaming[0].frameIndex = 11;
            { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
            auto completed = stats; completed.executionId = 10; completed.gpuTimingAvailable = true; completed.gpuMilliseconds = 3;
            completed.nodes[0].gpuTimingAvailable = true; completed.nodes[0].gpuMilliseconds = 2;
            completed.nodes[0].sections[0].gpuTimingAvailable = true; // Valid zero duration remains available.
            profiler.updateRenderGraphGpuStats(completed);
            const auto& displayed = profiler.displayFrame();
            const auto findPreparation = [&](const char* name) {
                const auto found = std::find_if(displayed.nodes.begin(), displayed.nodes.end(),
                    [&](const auto& node) { return node.name == name; });
                checkProfile(found != displayed.nodes.end(), std::string("Missing preparation scope: ") + name);
                return static_cast<size_t>(found - displayed.nodes.begin());
            };
            const size_t preflight = findPreparation("Graph preparation / Preflight");
            const size_t poll = findPreparation("Poll completions");
            const size_t revisions = findPreparation("Scene revisions");
            const size_t slot = findPreparation("Graph preparation / Submission slot wait");
            checkProfile(displayed.nodes[poll].parent == preflight && displayed.nodes[revisions].parent == preflight &&
                displayed.nodes[slot].parent == displayed.nodes[preflight].parent &&
                displayed.nodes[poll].cpuOnly && !displayed.nodes[poll].gpuTimingAvailable &&
                displayed.nodes[poll].cpuMilliseconds == 3,
                "Preparation hierarchy/timing was flattened or contaminated by delayed GPU results");
            checkProfile(displayed.index == 0 && displayed.nodes.back().gpuTimingAvailable && displayed.nodes.back().gpuMilliseconds == 0,
                "delayed nested/zero GPU result not backfilled into completed frame");
            checkProfile(!profiler.history().back().nodes.back().gpuTimingAvailable, "GPU result incorrectly attached to current frame");
            for (uint64_t i = 12; i < 530; ++i) {
                stats.executionId = i; stats.streaming[0].frameIndex = i;
                auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats);
            }
            checkProfile(profiler.history().size() == 500 && profiler.streamingHistory()[0].samples.size() == 500, "history is not bounded");
            stats.graphGeneration = 2; stats.executionId = 10;
            stats.streaming[0].generation = 2; stats.streaming[0].frameIndex = 0;
            { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
            profiler.updateRenderGraphGpuStats(completed);
            checkProfile(profiler.history().size() == 1 && !profiler.displayFrame().nodes.back().gpuTimingAvailable,
                "old graph timing contaminated recompiled graph with reused execution ID");
            checkProfile(profiler.streamingHistory().size() == 1 && profiler.streamingHistory()[0].samples.size() == 1,
                "reloaded stream retained old history");
            { auto frame = profiler.beginFrame(); }
            checkProfile(profiler.streamingHistory().empty() && profiler.displayFrame().nodes.size() == 1,
                "leaving streaming scene retained stale data");
            return RhiTestResult::pass("Delayed GPU backfill, nested zero duration, bounded histories, reload and scene removal");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EditorProfilerHistoryTest);

EditorProfiler::Aggregate bruteProfilerAggregate(const EditorProfiler& profiler, size_t scopeId, bool gpu)
{
    EditorProfiler::Aggregate result;
    for (const auto& frame : profiler.history()) {
        for (const auto& node : frame.nodes) {
            if (node.scopeId != scopeId || (gpu && !node.gpuTimingAvailable)) { continue; }
            const double value = gpu ? node.gpuMilliseconds : node.cpuMilliseconds;
            if (!std::isfinite(value)) { continue; }
            result.average += value;
            if (!result.count) { result.minimum = result.maximum = value; }
            else { result.minimum = std::min(result.minimum, value); result.maximum = std::max(result.maximum, value); }
            ++result.count;
        }
    }
    if (result.count) { result.average /= static_cast<double>(result.count); }
    return result;
}

void checkProfilerAggregates(EditorProfiler& profiler, const std::string& phase)
{
    std::set<size_t> scopes;
    for (const auto& frame : profiler.history()) {
        for (const auto& node : frame.nodes) {
            checkProfile(node.scopeId != SIZE_MAX, phase + ": raw history node has no stable scope ID");
            scopes.insert(node.scopeId);
        }
    }
    for (const auto& node : profiler.presentationFrame().nodes) {
        checkProfile(node.scopeId != SIZE_MAX, phase + ": presentation node has no stable scope ID");
        scopes.insert(node.scopeId);
    }
    for (size_t scopeId : scopes) {
        const auto actual = profiler.historyStatistics(scopeId);
        for (bool gpu : {false, true}) {
            const auto expected = bruteProfilerAggregate(profiler, scopeId, gpu);
            const auto& aggregate = gpu ? actual.gpu : actual.cpu;
            const auto label = phase + ": scope " + std::to_string(scopeId) + (gpu ? " GPU" : " CPU");
            checkProfile(aggregate.count == expected.count, label + " sample count differs from raw history");
            if (!expected.count) { continue; }
            const auto close = [](double a, double b) {
                return std::isfinite(a) && std::abs(a - b) <= 1e-10 * std::max(1.0, std::abs(b));
            };
            checkProfile(close(aggregate.average, expected.average) && close(aggregate.minimum, expected.minimum) &&
                close(aggregate.maximum, expected.maximum), label + " aggregate differs from raw history");
        }
    }
    const auto unknown = profiler.historyStatistics(SIZE_MAX);
    checkProfile(unknown.cpu.count == 0 && unknown.gpu.count == 0, phase + ": invalid scope returned samples");
}

size_t profilerChild(const EditorProfiler::Frame& frame, size_t parent, const char* name, size_t occurrence = 0)
{
    checkProfile(parent < frame.nodes.size(), "Invalid fixture parent");
    for (size_t child : frame.nodes[parent].children) {
        if (frame.nodes[child].name == name && occurrence-- == 0) { return child; }
    }
    throw std::runtime_error(std::string("Missing fixture scope: ") + name);
}

class EditorProfilerIncrementalStatisticsTest final : public RhiTest {
public:
    EditorProfilerIncrementalStatisticsTest() { type = RhiTestType::Command; name = "editor_profiler_incremental_statistics"; }
    RhiTestResult run(RhiTestContext&) override
    {
        try {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double infinity = std::numeric_limits<double>::infinity();
            EditorProfiler cpuProfiler;
            const auto emitCpu = [&](uint32_t index) {
                std::vector<RenderGraphProfileSection> sections;
                if (index % 2) { sections.push_back({.name = "Conditional insertion", .cpuMilliseconds = 9}); }
                const auto first = static_cast<uint32_t>(sections.size());
                sections.push_back({.name = "Repeated", .cpuMilliseconds = 20.0 + index % 3});
                sections.push_back({.name = "Leaf", .parent = first, .cpuMilliseconds = 2.0 + index % 3});
                const auto second = static_cast<uint32_t>(sections.size());
                sections.push_back({.name = "Repeated", .cpuMilliseconds = 200.0 + index % 5});
                sections.push_back({.name = "Leaf", .parent = second, .cpuMilliseconds = 40.0 + index % 5});
                sections.push_back({.name = "Rolling extrema", .cpuMilliseconds = index == 0 ? 0.0 : index == 1 ? 1000.0 : 10.0 + index % 41});
                const double edge = index % 5 == 0 ? 0.0 : index % 5 == 1 ? nan : index % 5 == 2 ? infinity : index % 5 == 3 ? -infinity : 4.0;
                sections.push_back({.name = "Nonfinite", .cpuMilliseconds = edge});
                sections.push_back({.name = "Valid zero", .cpuMilliseconds = 0});
                if (index % 3 == 0) { sections.push_back({.name = "Missing sometimes", .cpuMilliseconds = 7}); }
                if (index == 0) { sections.push_back({.name = "Vanished", .cpuMilliseconds = 12}); }
                auto frame = cpuProfiler.beginFrame();
                cpuProfiler.addCpuProfile(sections);
            };
            emitCpu(0);
            const auto& initial = cpuProfiler.history().back();
            const size_t firstRepeated = initial.nodes[profilerChild(initial, 0, "Repeated")].scopeId;
            const size_t secondRepeated = initial.nodes[profilerChild(initial, 0, "Repeated", 1)].scopeId;
            const size_t firstLeaf = initial.nodes[profilerChild(initial, profilerChild(initial, 0, "Repeated"), "Leaf")].scopeId;
            const size_t secondLeaf = initial.nodes[profilerChild(initial, profilerChild(initial, 0, "Repeated", 1), "Leaf")].scopeId;
            const size_t rolling = initial.nodes[profilerChild(initial, 0, "Rolling extrema")].scopeId;
            const size_t vanished = initial.nodes[profilerChild(initial, 0, "Vanished")].scopeId;
            const size_t zero = initial.nodes[profilerChild(initial, 0, "Valid zero")].scopeId;
            checkProfile(firstRepeated != secondRepeated && firstLeaf != secondLeaf,
                "Repeated sibling occurrences or their children share a scope identity");
            checkProfilerAggregates(cpuProfiler, "Initial CPU sample");
            for (uint32_t index = 1; index < 520; ++index) {
                emitCpu(index);
                const auto& current = cpuProfiler.history().back();
                const size_t first = profilerChild(current, 0, "Repeated");
                const size_t second = profilerChild(current, 0, "Repeated", 1);
                checkProfile(current.nodes[first].scopeId == firstRepeated && current.nodes[second].scopeId == secondRepeated &&
                    current.nodes[profilerChild(current, first, "Leaf")].scopeId == firstLeaf &&
                    current.nodes[profilerChild(current, second, "Leaf")].scopeId == secondLeaf,
                    "Conditional sibling insertion changed occurrence/subtree identities");
                if (index < 5 || index >= 498) { checkProfilerAggregates(cpuProfiler, "CPU frame " + std::to_string(index)); }
                if (index == 499) {
                    const auto extrema = cpuProfiler.historyStatistics(rolling).cpu;
                    checkProfile(extrema.count == 500 && extrema.minimum == 0 && extrema.maximum == 1000,
                        "Full CPU window lost seeded extrema");
                }
                if (index == 500) {
                    const auto extrema = cpuProfiler.historyStatistics(rolling).cpu;
                    checkProfile(extrema.count == 500 && extrema.minimum >= 10 && extrema.maximum == 1000 &&
                        cpuProfiler.historyStatistics(vanished).cpu.count == 0,
                        "Evicting first sample retained minimum or a vanished scope sample");
                }
                if (index == 501) {
                    const auto extrema = cpuProfiler.historyStatistics(rolling).cpu;
                    checkProfile(extrema.minimum >= 10 && extrema.maximum <= 50,
                        "Evicting previous maximum retained stale CPU extrema");
                }
            }
            checkProfile(cpuProfiler.history().size() == 500 && cpuProfiler.historyStatistics(zero).cpu.count == 500 &&
                cpuProfiler.historyStatistics(zero).cpu.average == 0 && cpuProfiler.historyStatistics(firstRepeated).cpu.maximum < 30 &&
                cpuProfiler.historyStatistics(secondRepeated).cpu.minimum >= 200,
                "Missing/zero samples or repeated siblings were combined incorrectly");
            const auto lastCpu = cpuProfiler.history().back();
            cpuProfiler.clearHistory();
            const auto clearedCpu = cpuProfiler.presentationFrame();
            checkProfile(cpuProfiler.history().empty() && clearedCpu.nodes.size() == lastCpu.nodes.size(),
                "Clear did not rebuild presentation from only the latest CPU frame");
            for (const auto& node : clearedCpu.nodes) {
                const auto statistics = cpuProfiler.historyStatistics(node.scopeId);
                checkProfile(statistics.cpu.count == 0 && statistics.gpu.count == 0,
                    "Clear presentation repopulated history statistics with latest values");
                checkProfile(node.name != "Vanished", "Clear retained an expired presentation scope");
            }
            checkProfilerAggregates(cpuProfiler, "Cleared CPU history");
            emitCpu(520);
            checkProfile(cpuProfiler.history().size() == 1, "CPU samples did not restart after clear");
            checkProfilerAggregates(cpuProfiler, "Restarted CPU history");

            EditorProfiler gpuProfiler;
            const auto graphStats = [](uint64_t executionId, uint64_t generation, bool available, double value) {
                RenderGraphExecutionStats stats{.executionId = executionId, .graphGeneration = generation,
                    .cpuMilliseconds = 5, .gpuMilliseconds = value, .gpuTimingAvailable = available};
                stats.nodes.push_back({.id = 7, .name = "GPU pass", .type = "Test", .cpuMilliseconds = 2,
                    .gpuMilliseconds = value, .gpuTimingAvailable = available});
                stats.nodes[0].sections = {
                    {.name = "Repeated", .cpuMilliseconds = .5, .gpuMilliseconds = value, .gpuTimingAvailable = available},
                    {.name = "Leaf", .parent = 0, .cpuMilliseconds = .1, .gpuMilliseconds = value * .1, .gpuTimingAvailable = available},
                    {.name = "Repeated", .cpuMilliseconds = 5, .gpuMilliseconds = value * 10, .gpuTimingAvailable = available},
                    {.name = "Leaf", .parent = 2, .cpuMilliseconds = 1, .gpuMilliseconds = value * 2, .gpuTimingAvailable = available}};
                return stats;
            };
            const auto emitGpu = [&](const RenderGraphExecutionStats& stats) {
                auto frame = gpuProfiler.beginFrame();
                gpuProfiler.addRenderGraphStats(stats);
            };
            emitGpu(graphStats(1, 1, false, 999));
            emitGpu(graphStats(2, 1, false, 999));
            emitGpu(graphStats(3, 1, true, 0));
            const auto& gpuInitial = gpuProfiler.history().back();
            const size_t envelopeIndex = profilerChild(gpuInitial, 0, "RenderGraph GPU envelope");
            const size_t passIndex = profilerChild(gpuInitial, envelopeIndex, "GPU pass (Test)");
            const size_t pass = gpuInitial.nodes[passIndex].scopeId;
            const size_t gpuFirst = gpuInitial.nodes[profilerChild(gpuInitial, passIndex, "Repeated")].scopeId;
            const size_t gpuSecond = gpuInitial.nodes[profilerChild(gpuInitial, passIndex, "Repeated", 1)].scopeId;
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 1 && gpuProfiler.historyStatistics(pass).gpu.average == 0,
                "Unavailable GPU value was counted or available zero was dropped");
            checkProfilerAggregates(gpuProfiler, "Initial GPU samples");
            auto completed = graphStats(1, 1, true, 10);
            gpuProfiler.updateRenderGraphGpuStats(completed);
            checkProfilerAggregates(gpuProfiler, "Delayed GPU backfill");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 2 && gpuProfiler.historyStatistics(pass).gpu.average == 5 &&
                !gpuProfiler.history()[1].nodes[passIndex].gpuTimingAvailable && gpuProfiler.history()[2].nodes[passIndex].gpuMilliseconds == 0,
                "Backfill changed the wrong execution or did not add a matching sample");
            gpuProfiler.updateRenderGraphGpuStats(completed);
            checkProfilerAggregates(gpuProfiler, "Repeated GPU backfill");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 2, "Repeated GPU completion duplicated a sample");
            completed = graphStats(1, 1, true, 2);
            gpuProfiler.updateRenderGraphGpuStats(completed);
            checkProfilerAggregates(gpuProfiler, "Corrected GPU backfill");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.average == 1 && gpuProfiler.historyStatistics(pass).gpu.maximum == 2 &&
                gpuProfiler.historyStatistics(gpuFirst).gpu.maximum == 2 && gpuProfiler.historyStatistics(gpuSecond).gpu.maximum == 20,
                "GPU correction retained stale extrema or conflated repeated siblings");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(2, 1, true, 14));
            checkProfilerAggregates(gpuProfiler, "Second delayed GPU result");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 1, false, 999));
            checkProfilerAggregates(gpuProfiler, "Invalidated GPU result");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 2 && gpuProfiler.historyStatistics(pass).gpu.average == 7,
                "Invalidating GPU availability retained the previous sample");
            completed = graphStats(2, 1, true, nan);
            completed.gpuMilliseconds = infinity;
            completed.nodes[0].sections[2].gpuMilliseconds = -infinity;
            gpuProfiler.updateRenderGraphGpuStats(completed);
            checkProfilerAggregates(gpuProfiler, "Nonfinite GPU correction");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 1 && gpuProfiler.historyStatistics(pass).gpu.average == 0,
                "Nonfinite GPU correction was counted");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(2, 1, true, 6));
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 1, true, 0));
            checkProfilerAggregates(gpuProfiler, "Restored GPU samples and valid zero");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 3 && gpuProfiler.historyStatistics(pass).cpu.count == 3 &&
                gpuProfiler.historyStatistics(pass).cpu.average == 2, "GPU correction changed CPU statistics or dropped a valid zero");
            for (uint64_t executionId = 4; executionId <= 505; ++executionId) {
                emitGpu(graphStats(executionId, 1, true, 15.0 + executionId % 7));
                if (executionId >= 503) { checkProfilerAggregates(gpuProfiler, "GPU rollover " + std::to_string(executionId)); }
            }
            const auto rolled = gpuProfiler.historyStatistics(pass).gpu;
            checkProfile(rolled.count == 500 && rolled.minimum >= 15 && rolled.maximum <= 21,
                "GPU rollover retained evicted zero/minimum or correction extrema");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 1, true, 9999));
            checkProfilerAggregates(gpuProfiler, "Completion for evicted execution");
            checkProfile(gpuProfiler.historyStatistics(pass).gpu.count == 500 && gpuProfiler.historyStatistics(pass).gpu.maximum <= 21,
                "Completion outside history reinserted an evicted GPU sample");
            emitGpu(graphStats(1, 2, false, 999));
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 1, true, 9999));
            checkProfilerAggregates(gpuProfiler, "Old generation with reused execution ID");
            const auto& newGeneration = gpuProfiler.history().back();
            const size_t newPassIndex = profilerChild(newGeneration, profilerChild(newGeneration, 0, "RenderGraph GPU envelope"), "GPU pass (Test)");
            const size_t newPass = newGeneration.nodes[newPassIndex].scopeId;
            checkProfile(gpuProfiler.history().size() == 1 && gpuProfiler.historyStatistics(newPass).gpu.count == 0,
                "Old generation contaminated new graph statistics");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 2, true, 11));
            checkProfilerAggregates(gpuProfiler, "Current generation completion");
            checkProfile(gpuProfiler.historyStatistics(newPass).gpu.count == 1 && gpuProfiler.historyStatistics(newPass).gpu.average == 11,
                "Current generation completion was not recorded");
            const auto lastGpu = gpuProfiler.history().back();
            gpuProfiler.clearHistory();
            const auto clearedGpu = gpuProfiler.presentationFrame();
            checkProfile(clearedGpu.nodes.size() == lastGpu.nodes.size(), "Clear lost latest GPU presentation topology");
            gpuProfiler.updateRenderGraphGpuStats(graphStats(1, 2, true, 19));
            checkProfilerAggregates(gpuProfiler, "GPU completion after clear");
            checkProfile(gpuProfiler.history().empty(), "Completion after clear created a history frame");
            for (const auto& node : gpuProfiler.presentationFrame().nodes) {
                const auto statistics = gpuProfiler.historyStatistics(node.scopeId);
                checkProfile(statistics.cpu.count == 0 && statistics.gpu.count == 0,
                    "Latest GPU presentation contaminated cleared history statistics");
            }
            emitGpu(graphStats(2, 2, true, 4));
            checkProfilerAggregates(gpuProfiler, "GPU restart after clear");
            checkProfile(gpuProfiler.history().size() == 1, "GPU history did not restart after clear");

            EditorProfiler reusedExecution;
            for (int index = 0; index < 2; ++index) {
                auto stats = graphStats(42, 1, false, 999);
                stats.nodes[0].cpuMilliseconds = 3 + index;
                auto frame = reusedExecution.beginFrame();
                reusedExecution.addRenderGraphStats(stats);
            }
            const auto& reusedFrame = reusedExecution.history().back();
            const size_t reusedPass = reusedFrame.nodes[profilerChild(reusedFrame,
                profilerChild(reusedFrame, 0, "RenderGraph GPU envelope"), "GPU pass (Test)")].scopeId;
            reusedExecution.updateRenderGraphGpuStats(graphStats(42, 1, true, 8));
            checkProfilerAggregates(reusedExecution, "GPU completion shared by multiple history frames");
            checkProfile(reusedExecution.historyStatistics(reusedPass).gpu.count == 2 &&
                reusedExecution.historyStatistics(reusedPass).gpu.average == 8 &&
                reusedExecution.historyStatistics(reusedPass).cpu.average == 3.5,
                "Shared execution completion did not update every history contribution");
            reusedExecution.updateRenderGraphGpuStats(graphStats(42, 1, true, 2));
            checkProfilerAggregates(reusedExecution, "Correction shared by multiple history frames");
            checkProfile(reusedExecution.historyStatistics(reusedPass).gpu.count == 2 &&
                reusedExecution.historyStatistics(reusedPass).gpu.maximum == 2,
                "Shared execution correction duplicated contributions or retained old extrema");
            reusedExecution.updateRenderGraphGpuStats(graphStats(42, 1, false, 999));
            checkProfilerAggregates(reusedExecution, "Invalidation shared by multiple history frames");
            checkProfile(reusedExecution.historyStatistics(reusedPass).gpu.count == 0,
                "Shared execution invalidation did not remove every history contribution");

            EditorProfiler fullCatalog;
            std::vector<RenderGraphProfileSection> catalogSections;
            for (size_t index = 0; index < 8191; ++index) {
                catalogSections.push_back({.name = "Catalog " + std::to_string(index), .cpuMilliseconds = 5});
            }
            { auto frame = fullCatalog.beginFrame(); fullCatalog.addCpuProfile(catalogSections); }
            const auto fullView = fullCatalog.presentationFrame();
            const size_t known = fullView.nodes[profilerChild(fullView, 0, "Catalog 0")].scopeId;
            checkProfile(fullView.nodes.size() == 8192 && !fullView.profilingOverflow,
                "Scope catalog did not accept its exact capacity");
            for (size_t index = 0; index < 4; ++index) {
                std::vector<RenderGraphProfileSection> overflowSections{{.name = "Catalog 0", .cpuMilliseconds = 7}};
                for (size_t item = 0; item < 16; ++item) {
                    overflowSections.push_back({.name = "Overflow " + std::to_string(index * 16 + item), .cpuMilliseconds = 99});
                }
                { auto frame = fullCatalog.beginFrame(); fullCatalog.addCpuProfile(overflowSections); }
                const auto view = fullCatalog.presentationFrame();
                checkProfile(view.nodes.size() == 8192 && view.profilingOverflow &&
                    view.nodes[profilerChild(view, 0, "Catalog 0")].scopeId == known,
                    "Overflow grew the scope catalog, lost a known identity or omitted its warning");
                const auto& raw = fullCatalog.history().back();
                for (size_t item = 2; item < raw.nodes.size(); ++item) {
                    checkProfile(raw.nodes[item].scopeId == SIZE_MAX,
                        "A new name acquired a scope identity after catalog capacity was exhausted");
                }
                const auto actual = fullCatalog.historyStatistics(known).cpu;
                const auto expected = bruteProfilerAggregate(fullCatalog, known, false);
                checkProfile(actual.count == index + 2 && actual.count == expected.count &&
                    actual.average == expected.average && actual.minimum == 5 && actual.maximum == 7,
                    "Catalog overflow stopped updating a known scope or counted rejected names");
            }
            return RhiTestResult::pass("Raw-history CPU/GPU aggregates, 500-sample extrema eviction, occurrence identities, missing/nonfinite/zero samples, shared-execution GPU corrections, resets and bounded scope catalog");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EditorProfilerIncrementalStatisticsTest);

class EditorProfilerCaptureTest final : public RhiTest {
public:
    EditorProfilerCaptureTest() { type = RhiTestType::Command; name = "editor_profiler_capture_attribution"; }
    RhiTestResult run(RhiTestContext&) override
    {
        EditorProfiler profiler;
        profiler.beginCapture();
        RenderGraphExecutionStats stats{.graphGeneration=1};
        stats.nodes.push_back({.id=7, .name="Pass", .cpuMilliseconds=1});
        stats.streaming.push_back({.passName="Pass", .assetPath="capture", .generation=1});
        for (uint64_t i=0; i<540; ++i) {
            stats.executionId=i; stats.streaming[0].frameIndex=i;
            auto frame=profiler.beginFrame(); profiler.addRenderGraphStats(stats);
        }
        profiler.endCapture();
        checkProfile(profiler.history().size()==500 && profiler.capturedFrames().size()==540,
            "Capture was truncated with UI history");
        checkProfile(profiler.capturedFrames()[0].streaming[0].frameIndex==0,
            "Capture lost original streaming frame");
        auto completed=stats; completed.executionId=0; completed.gpuTimingAvailable=true;
        completed.nodes[0].gpuTimingAvailable=true; completed.nodes[0].gpuMilliseconds=0;
        // Resolve old capture after a graph reload reuses the same execution ID.
        stats.graphGeneration=2; stats.executionId=0;
        { auto frame=profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
        profiler.updateRenderGraphGpuStats(completed);
        checkProfile(profiler.capturedFrames()[0].nodes.back().gpuTimingAvailable &&
            profiler.capturedFrames()[0].nodes.back().gpuMilliseconds==0,
            "Completed capture did not accept a delayed valid zero GPU result");
        checkProfile(!profiler.capturedFrames().back().nodes.back().gpuTimingAvailable &&
            !profiler.history().back().nodes.back().gpuTimingAvailable,
            "GPU result crossed frame/generation boundaries");
        profiler.beginCapture();
        { auto frame=profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
        profiler.endCapture();
        profiler.updateRenderGraphGpuStats(completed);
        checkProfile(profiler.capturedFrames().size()==1 && !profiler.capturedFrames()[0].nodes.back().gpuTimingAvailable,
            "Restarted capture retained stale GPU mappings");
        return RhiTestResult::pass("Unbounded capture retains frame/stream identity beyond UI history and accepts only matching delayed GPU results");
    }
};
METALLIC_REGISTER_RHI_TEST(EditorProfilerCaptureTest);

class EditorProfilerIntermittentScopesTest final : public RhiTest {
public:
    EditorProfilerIntermittentScopesTest() { type = RhiTestType::Command; name = "editor_profiler_intermittent_scopes"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            EditorProfiler profiler;
            profiler.beginCapture();
            auto emit = [&](bool optional) {
                auto frame = profiler.beginFrame();
                auto parent = profiler.scope("Raster");
                if (optional) { auto scope = profiler.scope("Optional HZB"); }
                { auto stable = profiler.scope("Always"); auto child = profiler.scope("Child"); }
            };
            const auto find = [&](const EditorProfiler::Frame& frame, const char* name) {
                for (size_t i = 0; i < frame.nodes.size(); ++i) { if (frame.nodes[i].name == name) { return i; } }
                return SIZE_MAX;
            };
            emit(false);
            const auto first = profiler.presentationFrame();
            const auto stable = find(first, "Always"), child = find(first, "Child");
            emit(true);
            const auto withOptional = profiler.presentationFrame();
            const auto optional = find(withOptional, "Optional HZB");
            checkProfile(optional != SIZE_MAX && find(withOptional, "Always") == stable && find(withOptional, "Child") == child,
                "New conditional scope changed existing UI node identities");
            for (int i = 0; i < 20; ++i) {
                emit(i % 2 == 0);
                const auto view = profiler.presentationFrame();
                checkProfile(view.nodes.size() == withOptional.nodes.size() && find(view, "Always") == stable &&
                    view.nodes[child].parent == stable && std::isfinite(view.nodes[optional].cpuMilliseconds) == (i % 2 == 0),
                    "Intermittent scope changed topology or retained a stale last duration");
            }
            checkProfile(find(profiler.displayFrame(), "Optional HZB") == SIZE_MAX &&
                find(profiler.capturedFrames().back(), "Optional HZB") == SIZE_MAX,
                "Presentation placeholders leaked into raw history/capture");
            std::filesystem::create_directories(context.outputDirectory);
            std::string log;
            checkProfile(saveProfilerPanel(profiler, "Table", context.outputDirectory / "Profiler-Stable-Scopes.png", log, false), log);
            for (int i = 0; i < 501; ++i) { emit(false); }
            checkProfile(find(profiler.presentationFrame(), "Optional HZB") == optional,
                "History rollover removed a known scope");
            profiler.clearHistory();
            checkProfile(find(profiler.presentationFrame(), "Optional HZB") == SIZE_MAX,
                "Clear history retained old UI scope catalog");
            RenderGraphExecutionStats stats{.executionId = 1, .graphGeneration = 1};
            stats.nodes.push_back({.id = 1, .name = "GPU pass", .cpuMilliseconds = 1});
            { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
            profiler.presentationFrame();
            stats.graphGeneration = 2; stats.nodes[0].name = "New pass";
            { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
            checkProfile(find(profiler.presentationFrame(), "GPU pass ()") == SIZE_MAX,
                "Graph generation retained stale UI topology");
            { auto frame = profiler.beginFrame(); }
            checkProfile(profiler.presentationFrame().nodes.size() == 1, "Scene exit retained old graph rows");
            return RhiTestResult::pass("Stable intermittent topology and identities, missing last values, history rollover, raw capture isolation and reset");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EditorProfilerIntermittentScopesTest);

class EditorProfilerSortingTest final : public RhiTest {
public:
    EditorProfilerSortingTest() { type = RhiTestType::Command; name = "editor_profiler_column_sorting"; }
    RhiTestResult run(RhiTestContext& testContext) override
    {
        auto* previous = ImGui::GetCurrentContext();
        auto* context = ImGui::CreateContext();
        struct Cleanup {
            ImGuiContext* context;
            ImGuiContext* previous;
            ~Cleanup() { ImGui::DestroyContext(context); ImGui::SetCurrentContext(previous); }
        } cleanup{context, previous};
        try {
            auto& io = ImGui::GetIO(); io.IniFilename = nullptr; io.LogFilename = nullptr;
            io.DisplaySize = ImVec2(1500, 1000); io.DeltaTime = 1.f / 60;
            unsigned char* pixels = nullptr; int width = 0, height = 0;
            io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height); io.Fonts->SetTexID(ImTextureID(1));
            EditorProfiler profiler;
            RenderGraphExecutionStats stats{.graphGeneration = 1, .gpuTimingAvailable = true};
            const char* names[]{"Alpha", "Bravo", "Charlie", "Missing", "Zero"};
            const QueueType queues[]{QueueType::Graphics, QueueType::Compute, QueueType::Copy, QueueType::Graphics, QueueType::Compute};
            const double gpu[2][5]{{9, 2, 3, 999, 0}, {1, 8, 3, 999, 0}};
            const double cpu[2][5]{{4, 1, 10, 5, 0}, {2, 1, 0, 5, 0}};
            for (uint32_t f = 0; f < 2; ++f) {
                stats.executionId = f; stats.nodes.clear();
                for (uint32_t n = 0; n < 5; ++n) {
                    stats.nodes.push_back({.id = n, .name = names[n], .type = "Test", .cpuMilliseconds = cpu[f][n],
                        .gpuMilliseconds = gpu[f][n], .gpuTimingAvailable = n != 3, .queue = queues[n]});
                }
                stats.nodes[0].sections = {{.name = "Alpha child low", .gpuMilliseconds = 1, .gpuTimingAvailable = true},
                    {.name = "Alpha child high", .gpuMilliseconds = 7, .gpuTimingAvailable = true}};
                auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats);
            }
            const auto draw = [&](bool expand = true) {
                ImGui::NewFrame();
                if (ImGui::FindWindowByName("Profiler")) { ImGui::SetWindowSize("Profiler", io.DisplaySize); }
                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::LogToBuffer(expand ? 16 : 0);
                bool open = true; profiler.drawWindow(&open, {});
                std::string text = context->LogBuffer.c_str();
                ImGui::LogFinish(); ImGui::Render();
                return text;
            };
            const auto table = [&]() {
                // BeginTabItem adds a scope ID; this isolated context owns one table.
                return context->Tables.GetAliveCount() ? context->Tables.GetByIndex(0) : nullptr;
            };
            const auto click = [&](ImVec2 point) {
                io.AddMousePosEvent(point.x, point.y); draw();
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); draw();
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); draw();
                io.AddMousePosEvent(-100, -100);
                return draw();
            };
            const auto header = [&](int column) {
                auto* t = table(); checkProfile(t && column < t->ColumnsCount, "missing profiler table column");
                return click(ImVec2(t->Columns[column].MinX + 15, t->OuterRect.Min.y + ImGui::GetTextLineHeight() * .5f));
            };
            const auto expectOrder = [&](const std::string& text, std::initializer_list<const char*> expected) {
                size_t previousPosition = 0;
                for (auto* name : expected) {
                    const auto position = text.find(std::string(name) + " (Test)");
                    checkProfile(position != std::string::npos && position >= previousPosition, "incorrect order at " + std::string(name) + "\n" + text);
                    previousPosition = position;
                }
            };
            draw(); draw();
            expectOrder(draw(), {"Alpha", "Bravo", "Charlie", "Missing", "Zero"});
            auto text = header(1);
            expectOrder(text, {"Alpha", "Bravo", "Charlie", "Zero", "Missing"});
            checkProfile(text.find("Alpha child high") < text.find("Alpha child low") && text.find("Alpha child low") < text.find("Bravo (Test)"),
                "sorting flattened child scopes or failed to sort siblings");
            expectOrder(header(1), {"Zero", "Charlie", "Alpha", "Bravo", "Missing"});
            expectOrder(header(1), {"Alpha", "Bravo", "Charlie", "Missing", "Zero"});
            expectOrder(header(2), {"Charlie", "Missing", "Alpha", "Bravo", "Zero"});
            expectOrder(header(2), {"Zero", "Bravo", "Alpha", "Charlie", "Missing"});
            expectOrder(header(0), {"Alpha", "Bravo", "Charlie", "Missing", "Zero"});
            expectOrder(header(0), {"Zero", "Missing", "Charlie", "Bravo", "Alpha"});
            expectOrder(header(3), {"Bravo", "Zero", "Charlie", "Alpha", "Missing"});
            // Toggle Detailed through the real checkbox and exercise every added column.
            click(ImVec2(17, 59));
            checkProfile(table()->ColumnsCount == 10, "Detailed checkbox did not expose timing columns");
            expectOrder(header(4), {"Bravo", "Charlie", "Alpha", "Zero", "Missing"});
            expectOrder(header(5), {"Charlie", "Bravo", "Alpha", "Zero", "Missing"});
            expectOrder(header(6), {"Alpha", "Bravo", "Charlie", "Zero", "Missing"});
            expectOrder(header(7), {"Missing", "Alpha", "Bravo", "Charlie", "Zero"});
            expectOrder(header(8), {"Missing", "Alpha", "Bravo", "Charlie", "Zero"});
            expectOrder(header(9), {"Charlie", "Missing", "Alpha", "Bravo", "Zero"});
            // A backfilled result changes ordering without clicking the header again.
            header(1);
            auto backfill = stats; backfill.nodes[3].gpuTimingAvailable = true; backfill.nodes[3].gpuMilliseconds = 20;
            profiler.updateRenderGraphGpuStats(backfill);
            expectOrder(draw(), {"Missing", "Alpha", "Bravo", "Charlie", "Zero"});
            // Close Alpha by its stable ImGui ID, then sort while keeping it closed.
            const auto& frame = profiler.displayFrame();
            size_t alpha = 0;
            for (size_t i = 0; i < frame.nodes.size(); ++i) { if (frame.nodes[i].name == "Alpha (Test)") { alpha = i; break; } }
            std::vector<size_t> chain;
            for (size_t i = alpha;; i = frame.nodes[i].parent) { chain.push_back(i); if (i == 0) { break; } }
            ImGuiID id = table()->ID;
            for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
                const void* pointer = reinterpret_cast<void*>(*it + 1); id = ImHashData(&pointer, sizeof(pointer), id);
            }
            table()->InnerWindow->StateStorage.SetInt(id, 0);
            checkProfile(draw(false).find("Alpha child low") == std::string::npos, "failed to collapse fixture scope");
            auto* t = table();
            const auto old = context->CurrentTable; context->CurrentTable = t;
            ImGui::TableSetColumnSortDirection(1, ImGuiSortDirection_Ascending, false);
            context->CurrentTable = old;
            text = draw(false);
            checkProfile(table()->InnerWindow->StateStorage.GetInt(id, -1) == 0 && text.find("Alpha child low") == std::string::npos,
                "sorting changed collapsed scope identity");
            const auto stableNodeCount = profiler.presentationFrame().nodes.size();
            for (uint32_t i = 0; i < 12; ++i) {
                stats.executionId = 100 + i;
                stats.nodes[0].sections.clear();
                if (i % 2 == 0) {
                    stats.nodes[0].sections = {{.name = "Alpha child low", .gpuTimingAvailable = true},
                        {.name = "Alpha child high", .gpuTimingAvailable = true}};
                }
                { auto frame = profiler.beginFrame(); profiler.addRenderGraphStats(stats); }
                text = draw(false);
                checkProfile(profiler.presentationFrame().nodes.size() == stableNodeCount &&
                    table()->InnerWindow->StateStorage.GetInt(id, -1) == 0 && text.find("Alpha child low") == std::string::npos,
                    "Conditional child changed the live ImGui collapse state");
            }
            std::filesystem::create_directories(testContext.outputDirectory);
            std::ofstream(testContext.outputDirectory / "ProfilerSortedTable.txt") << text;
            return RhiTestResult::pass("Real column clicks: tri-state, all timing/name/queue columns, missing/zero/ties, subtree ordering, GPU backfill and collapsed identity");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(EditorProfilerSortingTest);

// Render real ImGui draw data offscreen for UI QA, without creating an OS window
// or taking over the user's desktop. Only the font atlas is used by this panel.
bool saveProfilerPanel(EditorProfiler& profiler, const char* tab, const std::filesystem::path& path, std::string& message, bool sortGpu = false)
{
    auto* previous = ImGui::GetCurrentContext();
    auto* context = ImGui::CreateContext();
    struct Cleanup {
        ImGuiContext* context;
        ImGuiContext* previous;
        ~Cleanup() { ImGui::DestroyContext(context); ImGui::SetCurrentContext(previous); }
    } cleanup{context, previous};
    auto& io = ImGui::GetIO(); io.IniFilename = nullptr; io.LogFilename = nullptr;
    constexpr int width = 1180, height = 1120;
    io.DisplaySize = ImVec2(width, height); io.DeltaTime = 1.0f / 60;
    io.Fonts->AddFontDefault();
    unsigned char* atlas = nullptr; int tw = 0, th = 0;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &tw, &th); io.Fonts->SetTexID(ImTextureID(1));
    for (int f = 0; f < 4; ++f) {
        ImGui::NewFrame();
        if (sortGpu && context->Tables.GetAliveCount()) {
            auto* table = context->Tables.GetByIndex(0);
            const auto previousTable = context->CurrentTable;
            context->CurrentTable = table;
            ImGui::TableSetColumnSortDirection(1, ImGuiSortDirection_Descending, false);
            context->CurrentTable = previousTable;
        }
        if (auto* window = ImGui::FindWindowByName("Profiler")) {
            ImGui::SetWindowSize("Profiler", ImVec2(width, height), ImGuiCond_Always);
            if (auto* bar = context->TabBars.GetByKey(window->GetID("ProfilerTabs"))) {
                for (auto& item : bar->Tabs) {
                    if (std::string_view(ImGui::TabBarGetTabName(bar, &item)) == tab) { bar->NextSelectedTabId = item.ID; }
                }
            }
        }
        ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(ImVec2(width, height));
        bool open = true;
        const bool expandTable = std::string_view(tab) == "Table";
        if (expandTable) { ImGui::LogToBuffer(16); }
        profiler.drawWindow(&open, {});
        if (expandTable) { ImGui::LogFinish(); }
        ImGui::Render();
    }
    const auto* data = ImGui::GetDrawData();
    return saveImGuiTestDrawDataPng(*data, atlas, tw, th, width, height, path, message);
}

class MiniZorahProfilerTest final : public RhiTest {
public:
    MiniZorahProfilerTest() { type = RhiTestType::Rendering; name = "minizorah_profiler_streaming"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!std::getenv("METALLIC_TEST_MINIZORAH")) { return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1 for full scene profiler validation"); }
        std::filesystem::create_directories(context.outputDirectory);
        const bool stress = std::getenv("METALLIC_TEST_CLAS_ROAM_STRESS") != nullptr;
        const uint32_t frameCount = stress ? 2400u : 360u;
        Json report{{"stress", stress}, {"resolution", {1920, 1080}}, {"lodPixelError", 1.5}, {"frames", Json::array()}};
        const auto save = [&]() { std::ofstream(context.outputDirectory / "MiniZorahProfiler.json") << report.dump(2); };
        try {
            std::string log; RenderSampleLoadResult sample;
            checkProfile(loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log), log);
            auto graph = std::move(sample.graph);
            const bool clasEnabled = std::getenv("METALLIC_TEST_CLAS_OFF") == nullptr;
            graph.findNode("GPUDriven")->properties["enableClas"] = clasEnabled;
            report["clasEnabled"] = clasEnabled;
            graph.findNode("GPUDriven")->properties["maxClasBytes"] = (std::getenv("METALLIC_TEST_CLAS_LEGACY") ? 1536ull : 512ull) << 20;
            graph.findNode("GPUDriven")->properties["compactClas"] = std::getenv("METALLIC_TEST_CLAS_LEGACY") == nullptr;
            graph.findNode("GPUDriven")->properties["coldPageRetentionFrames"] = std::getenv("METALLIC_TEST_CLAS_LEGACY") ? 0 : 120;
            const auto node = graph.findNode("GPUDriven")->id;
            graph.findNode(node)->properties["debugStreamingPages"] = false;
            graph.findNode(node)->properties["asyncLateRaster"] = true;
            if (stress) {
                // Force pending CLAS and geometry cache turnover without
                // exhausting system VRAM or depending on interactive input.
                graph.findNode(node)->properties["maxClasBytes"] = 64ull << 20;
                graph.findNode(node)->properties["maxResidentBytes"] = 128ull << 20;
            }
            RenderView view;
            const Json original = graph.viewProperties().at("camera");
            checkProfile(view.setCameraProperties(original), "invalid camera");
            RenderGraphPreviewRenderer preview;
            checkProfile(bool(preview.initialize(context.enableValidation, true, false)), "preview initialization failed");
            preview.bindRenderView(&view);
            EditorProfiler profiler;
            bool sawCompute = false, sawResident = false; uint64_t bytes = 0; uint32_t peakRequests = 0;
            for (uint32_t f = 0; f < frameCount; ++f) {
                // Exercise both recording layouts regardless of sample defaults.
                const bool asyncRaster = f >= frameCount / 2;
                graph.setNodeRuntimeProperty(node, "asyncSoftwareRaster", asyncRaster);
                Json camera = original;
                // Finish with a stationary view so asynchronous build/move and
                // cold retirement can converge independently of fresh demand.
                const uint32_t cameraFrame = stress ? f : std::min(f, 179u);
                const float angle = f < 60 ? 0.0f : stress
                    ? std::sin(float(f - 60) * .021f) * 2.7f
                    : std::sin(float(cameraFrame - 60) * .04f) * .16f;
                if (stress) {
                    camera["eye"][0] = original["eye"][0].get<float>() + 12.f * std::sin(float(f) * .007f);
                    camera["eye"][2] = original["eye"][2].get<float>() + 5.f * std::sin(float(f) * .013f);
                }
                const float x = original["center"][0].get<float>() - original["eye"][0].get<float>();
                const float z = original["center"][2].get<float>() - original["eye"][2].get<float>();
                camera["center"][0] = camera["eye"][0].get<float>() + std::cos(angle)*x + std::sin(angle)*z;
                camera["center"][2] = camera["eye"][2].get<float>() - std::sin(angle)*x + std::cos(angle)*z;
                checkProfile(view.setCameraProperties(camera), "invalid roaming camera");
                {
                    auto frame = profiler.beginFrame();
                    auto renderScope = profiler.scope("Render");
                    checkProfile(bool(preview.render(graph, 1920, 1080, "GPUDriven.color", false)), preview.lastLog());
                    profiler.addRenderGraphStats(preview.executionStats());
                }
                std::vector<RenderGraphExecutionStats> completed;
                checkProfile(bool(preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) && completed.size() == 1, "missing GPU frame sample");
                const auto& stats = completed.front(); profiler.updateRenderGraphGpuStats(stats);
                checkProfile(stats.gpuTimingAvailable && !stats.profilingOverflow, "GPU envelope unavailable or scope overflow");
                checkProfile(stats.streaming.size() == 1, "streaming sample missing without debug observer");
                const auto& stream = stats.streaming.front();
                report["lastObservedStream"] = {{"pages", stream.totalPages}, {"resident", stream.residentPages},
                    {"used", stream.geometryUsedBytes}, {"budget", stream.geometryBudgetBytes}, {"frame", stream.frameIndex}};
                checkProfile(stream.totalPages > 100000 && stream.residentPages <= stream.totalPages &&
                    stream.geometryBudgetBytes > 0 && stream.geometryUsedBytes <= stream.geometryBudgetBytes,
                    "invalid current-scene residency counters");
                sawResident |= stream.residentPages > 0 && stream.geometryUsedBytes > 0;
                checkProfile(stream.clasEnabled == clasEnabled && stream.clasUsedBytes <= stream.clasCapacityBytes &&
                    (clasEnabled ? stream.clasCapacityBytes > 0 : stream.clasUsedBytes == 0),
                    "stream CLAS pool missing or exceeded capacity");
                checkProfile(stream.clasBuiltClusters <= 8192, "CLAS exceeded per-frame build budget");
                if (f > 2) { checkProfile(stream.feedbackFrame != UINT64_MAX, "feedback age unavailable when debug capture is disabled"); }
                bytes += stream.uploadBytes; peakRequests = std::max(peakRequests, stream.requests);
                Json nodes = Json::array();
                bool sawCandidates = false, sawTraversal = false, sawClassify = false, sawHardware = false, sawRaster = false;
                for (const auto& pass : stats.nodes) {
                    checkProfile(pass.gpuTimingAvailable, "missing pass GPU timing: " + pass.name);
                    Json sections = Json::array();
                    for (const auto& section : pass.sections) {
                        if (section.cpuOnly) {
                            checkProfile(!section.gpuTimingAvailable, "CPU-only work acquired a GPU interval: " + section.name);
                        } else {
                            checkProfile(section.gpuTimingAvailable && std::isfinite(section.gpuMilliseconds) && section.gpuMilliseconds >= 0,
                                "missing inner GPU timing: " + section.name);
                        }
                        sawCandidates |= section.name == "Candidates"; sawTraversal |= section.name == "Stream traversal";
                        sawClassify |= section.name == "Soft/hard classification"; sawHardware |= section.name == "Hardware raster";
                        sawRaster |= section.name == "Visibility raster";
                        if (section.name == "Visibility raster" || section.name == "Stream traversal" || section.name == "Stream End") {
                            checkProfile(section.parent == UINT32_MAX, "streaming and raster must be separate top-level scopes");
                        }
                        if (section.name == "Stream early" || section.name == "Stream late") {
                            // Stage instrumentation wraps the helper's scope of
                            // the same name. Both must remain inside raster work.
                            bool insideRaster = false;
                            for (uint32_t ancestor = section.parent, depth = 0;
                                 ancestor < pass.sections.size() && depth < pass.sections.size(); ++depth) {
                                if (pass.sections[ancestor].name == "Visibility raster") { insideRaster = true; break; }
                                ancestor = pass.sections[ancestor].parent;
                            }
                            checkProfile(insideRaster, "raster phases missing their visibility-only ancestor");
                        }
                        sawCompute |= section.name == "Software raster" && section.queue == QueueType::Compute;
                        if (section.name == "Software raster") {
                            checkProfile(section.queue == (asyncRaster ? QueueType::Compute : QueueType::Graphics),
                                "software raster scope attributed to the wrong queue");
                        }
                        sections.push_back({{"name",section.name},{"parent",section.parent},{"cpuOnly",section.cpuOnly},{"queue",section.queue == QueueType::Compute ? "compute" : "graphics"},
                            {"gpuMs",section.gpuMilliseconds},{"cpuMs",section.cpuMilliseconds}});
                    }
                    nodes.push_back({{"name",pass.name},{"gpuMs",pass.gpuMilliseconds},{"cpuMs",pass.cpuMilliseconds},{"sections",sections}});
                }
                checkProfile(sawCandidates && sawTraversal && sawClassify && sawHardware && sawRaster, "missing GPUDriven stage instrumentation");
                report["frames"].push_back({{"frame",f},{"asyncRaster",asyncRaster},{"gpuMs",stats.gpuMilliseconds},{"cpuMs",stats.cpuMilliseconds},{"nodes",nodes},
                    {"streamFrame",stream.frameIndex},{"residentPages",stream.residentPages},{"pendingPages",stream.pendingPages},
                    {"clasBytes", stream.clasUsedBytes}, {"clasEncodedBytes", stream.clasEncodedBytes},
                    {"clasWorstCaseBytes", stream.clasWorstCaseBytes}, {"clasScratchBytes", stream.clasScratchBytes}, {"clasMovedClusters", stream.clasMovedClusters}, {"clasPages", stream.clasResidentPages}, {"clasClusters", stream.clasResidentClusters},
                    {"clasBuiltPages", stream.clasBuiltPages}, {"clasBuiltClusters", stream.clasBuiltClusters},
                    {"clasPendingPages", stream.clasPendingPages}, {"clasRejectedPages", stream.clasRejectedPages},
                    {"clasTotalBuiltPages", stream.clasTotalBuiltPages},
                    {"geometryBytes",stream.geometryUsedBytes},{"budgetBytes",stream.geometryBudgetBytes},{"requests",stream.requests},
                    {"uploads",stream.uploads},{"evictions",stream.evictions},{"uploadBytes",stream.uploadBytes}});
            }
            if (stress) {
                uint64_t evictions = 0, deferred = 0;
                for (const auto& sample : report["frames"]) {
                    evictions += sample["evictions"].get<uint32_t>();
                    deferred += sample["clasRejectedPages"].get<uint32_t>();
                }
                checkProfile(evictions > 100 && deferred > 100, "Roam stress did not exercise eviction and CLAS pressure");
                report["evictionsObserved"] = evictions;
                report["clasDeferralsObserved"] = deferred;
            } else if (clasEnabled) {
                const auto& finalStream = profiler.streamingHistory().front().samples.back();
                checkProfile(finalStream.clasResidentPages > 1000, "MiniZorah CLAS did not stream into residency");
                checkProfile(finalStream.clasPendingPages == 0 && finalStream.clasRejectedPages == 0 &&
                    finalStream.clasResidentPages == finalStream.residentPages, "MiniZorah CLAS backlog did not converge");
            }
            checkProfile(bytes > 0 && peakRequests > 0 && sawCompute && sawResident, "did not observe streaming traffic or async compute timings");
            for (const char* tab : {"Table", "Streaming", "LineChart", "BarChart"}) {
                checkProfile(saveProfilerPanel(profiler, tab, context.outputDirectory / (std::string("Profiler-")+tab+".png"), log), log);
            }
            checkProfile(saveProfilerPanel(profiler, "Table", context.outputDirectory / "Profiler-Table-Sorted.png", log, true), log);
            report["status"] = "passed"; report["asyncComputeTimed"] = sawCompute;
            report["uploadBytesObserved"] = bytes; report["peakRequests"] = peakRequests;
            save();
            return RhiTestResult::pass(std::to_string(frameCount) + " MiniZorah frames: nested GPU timings, asynchronous software raster, streaming telemetry and offscreen Profiler UI");
        } catch (const std::exception& error) { report["status"] = "failed"; report["error"] = error.what(); save(); return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahProfilerTest);

} // namespace
} // namespace metallic::tests
