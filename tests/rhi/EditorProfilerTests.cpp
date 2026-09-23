#include "RhiTest.h"
#include "Editor/EditorProfiler.h"
#include "Runtime/Render/RenderSample.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

void checkProfile(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

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
    std::vector<uint8_t> pixels(size_t(width) * height * 4, 24);
    for (size_t i = 3; i < pixels.size(); i += 4) { pixels[i] = 255; }
    const auto edge = [](ImVec2 a, ImVec2 b, ImVec2 p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); };
    for (const auto* list : data->CmdLists) {
        for (const auto& command : list->CmdBuffer) {
            if (command.UserCallback || command.GetTexID() != ImTextureID(1)) { continue; }
            for (unsigned int i = 0; i + 2 < command.ElemCount; i += 3) {
                const auto& a = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i] + command.VtxOffset];
                const auto& b = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i + 1] + command.VtxOffset];
                const auto& c = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i + 2] + command.VtxOffset];
                const float area = edge(a.pos, b.pos, c.pos); if (std::abs(area) < 1e-6f) { continue; }
                const int x0 = std::max({0, int(std::floor(std::min({a.pos.x,b.pos.x,c.pos.x}))), int(std::ceil(command.ClipRect.x))});
                const int y0 = std::max({0, int(std::floor(std::min({a.pos.y,b.pos.y,c.pos.y}))), int(std::ceil(command.ClipRect.y))});
                const int x1 = std::min({width, int(std::ceil(std::max({a.pos.x,b.pos.x,c.pos.x}))), int(command.ClipRect.z)});
                const int y1 = std::min({height, int(std::ceil(std::max({a.pos.y,b.pos.y,c.pos.y}))), int(command.ClipRect.w)});
                for (int y = y0; y < y1; ++y) { for (int x = x0; x < x1; ++x) {
                    const ImVec2 p(float(x)+.5f,float(y)+.5f);
                    const float wa = edge(b.pos,c.pos,p)/area, wb = edge(c.pos,a.pos,p)/area, wc = 1-wa-wb;
                    if (wa < 0 || wb < 0 || wc < 0) { continue; }
                    const int tx = std::clamp(int((wa*a.uv.x+wb*b.uv.x+wc*c.uv.x)*tw),0,tw-1);
                    const int ty = std::clamp(int((wa*a.uv.y+wb*b.uv.y+wc*c.uv.y)*th),0,th-1);
                    const auto* texel = atlas + (size_t(ty)*tw+tx)*4;
                    const auto channel = [&](int shift) { return wa*((a.col>>shift)&255)+wb*((b.col>>shift)&255)+wc*((c.col>>shift)&255); };
                    const float alpha = channel(IM_COL32_A_SHIFT)*texel[3]/(255.f*255.f);
                    auto* dest = pixels.data()+(size_t(y)*width+x)*4;
                    constexpr int shifts[]{IM_COL32_R_SHIFT,IM_COL32_G_SHIFT,IM_COL32_B_SHIFT};
                    for (int ch=0;ch<3;++ch) { dest[ch]=uint8_t(std::clamp(channel(shifts[ch])*texel[ch]/255.f*alpha+dest[ch]*(1-alpha),0.f,255.f)); }
                } }
            }
        }
    }
    return saveRgba8Png(path, pixels.data(), width, height, message);
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
                checkProfile(bool(preview.collectCompletedGpuExecutionStats(completed)) && completed.size() == 1, "missing GPU frame sample");
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
                bool sawCandidates = false, sawTraversal = false, sawClassify = false, sawHardware = false;
                for (const auto& pass : stats.nodes) {
                    checkProfile(pass.gpuTimingAvailable, "missing pass GPU timing: " + pass.name);
                    Json sections = Json::array();
                    for (const auto& section : pass.sections) {
                        checkProfile(section.gpuTimingAvailable && std::isfinite(section.gpuMilliseconds) && section.gpuMilliseconds >= 0,
                            "missing inner GPU timing: " + section.name);
                        sawCandidates |= section.name == "Candidates"; sawTraversal |= section.name == "Stream traversal";
                        sawClassify |= section.name == "Soft/hard classification"; sawHardware |= section.name == "Hardware raster";
                        sawCompute |= section.name == "Software raster" && section.queue == QueueType::Compute;
                        sections.push_back({{"name",section.name},{"parent",section.parent},{"queue",section.queue == QueueType::Compute ? "compute" : "graphics"},
                            {"gpuMs",section.gpuMilliseconds},{"cpuMs",section.cpuMilliseconds}});
                    }
                    nodes.push_back({{"name",pass.name},{"gpuMs",pass.gpuMilliseconds},{"cpuMs",pass.cpuMilliseconds},{"sections",sections}});
                }
                checkProfile(sawCandidates && sawTraversal && sawClassify && sawHardware, "missing GPUDriven stage instrumentation");
                report["frames"].push_back({{"frame",f},{"gpuMs",stats.gpuMilliseconds},{"cpuMs",stats.cpuMilliseconds},{"nodes",nodes},
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
