#include "Editor/EditorProfiler.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
#include "Runtime/Render/Profiling/TracyProfiler.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <utility>

namespace metallic {
namespace {

constexpr size_t kProfilerHistorySize = 500;
constexpr size_t kProfilerScopeTreeLimit = 8192;
constexpr float kPi = 3.14159265358979323846f;

ImU32 imguiColor(uint32_t rgba)
{
    const int r = static_cast<int>((rgba >> 24u) & 0xffu);
    const int g = static_cast<int>((rgba >> 16u) & 0xffu);
    const int b = static_cast<int>((rgba >> 8u) & 0xffu);
    const int a = static_cast<int>(rgba & 0xffu);
    return IM_COL32(r, g, b, a);
}

const EditorProfiler::Node* nodeByPath(const EditorProfiler::Frame& frame, const std::vector<std::string>& path)
{
    if (frame.nodes.empty()) { return nullptr; }
    size_t index = 0;
    for (const auto& part : path) {
        const auto& children = frame.nodes[index].children;
        const auto iter = std::find_if(children.begin(), children.end(), [&](size_t child) { return frame.nodes[child].name == part; });
        if (iter == children.end()) { return nullptr; }
        index = *iter;
    }
    return &frame.nodes[index];
}

double sampleValue(const EditorProfiler::Node* node, bool gpu)
{
    return node && (!gpu || node->gpuTimingAvailable)
        ? (gpu ? node->gpuMilliseconds : node->cpuMilliseconds) : std::numeric_limits<double>::quiet_NaN();
}

void drawDuration(double value, bool available = true)
{
    if (available && std::isfinite(value)) { ImGui::Text("%.3f", value); }
    else { ImGui::TextDisabled("--"); }
}

const char* queueName(render::QueueType queue)
{
    return queue == render::QueueType::Compute ? "Compute" : queue == render::QueueType::Copy ? "Copy" : "Graphics";
}

enum class ProfilerColumn : ImGuiID {
    Timer = 1, GPUAverage, CPUAverage, Queue, GPULast, GPUMinimum, GPUMaximum, CPULast, CPUMinimum, CPUMaximum
};

using ProfilerTableRow = EditorProfiler::HistoryStatistics;

const char* tableQueueName(const EditorProfiler::Node& node)
{
    if (node.cpuOnly || node.renderGraphExecutionId == UINT64_MAX) { return "CPU"; }
    return node.renderGraphNodeId == UINT32_MAX ? "Envelope" : queueName(node.queue);
}

double tableSortValue(const EditorProfiler::Node& node, const ProfilerTableRow& row, ProfilerColumn column)
{
    const double missing = std::numeric_limits<double>::quiet_NaN();
    switch (column) {
    case ProfilerColumn::GPUAverage: return row.gpu.count ? row.gpu.average : missing;
    case ProfilerColumn::CPUAverage: return row.cpu.count ? row.cpu.average : missing;
    case ProfilerColumn::GPULast: return node.gpuTimingAvailable ? node.gpuMilliseconds : missing;
    case ProfilerColumn::GPUMinimum: return row.gpu.count ? row.gpu.minimum : missing;
    case ProfilerColumn::GPUMaximum: return row.gpu.count ? row.gpu.maximum : missing;
    case ProfilerColumn::CPULast: return node.cpuMilliseconds;
    case ProfilerColumn::CPUMinimum: return row.cpu.count ? row.cpu.minimum : missing;
    case ProfilerColumn::CPUMaximum: return row.cpu.count ? row.cpu.maximum : missing;
    default: return missing;
    }
}

void drawProfilerTableNode(const EditorProfiler::Frame& frame,
    size_t index, uint32_t depth, bool detailed, const std::vector<ProfilerTableRow>& rows,
    const ImGuiTableColumnSortSpecs* sort)
{
    const auto& node = frame.nodes[index];
    const bool children = !node.children.empty();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth;
    if (!children) { flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen; }
    else if (depth < 4) { flags |= ImGuiTreeNodeFlags_DefaultOpen; }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    const bool present = std::isfinite(node.cpuMilliseconds);
    ImGui::PushStyleColor(ImGuiCol_Text, present ? imguiColor(node.color) : ImGui::GetColorU32(ImGuiCol_TextDisabled));
    const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(index + 1), flags, "%s", node.name.c_str());
    ImGui::PopStyleColor();
    const auto& row = rows[index];
    const auto& gpu = row.gpu;
    const auto& cpu = row.cpu;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s\nCPU samples: %zu | GPU samples: %zu\nAbsent scopes and missing GPU queries are excluded from averages.\nNested / concurrent intervals must not be added together.", present ? "Recorded in displayed frame" : "Not recorded in displayed frame; historical row retained", cpu.count, gpu.count);
    }
    ImGui::TableNextColumn(); drawDuration(gpu.average, gpu.count != 0);
    ImGui::TableNextColumn(); drawDuration(cpu.average, cpu.count != 0);
    ImGui::TableNextColumn();
    if (node.renderGraphExecutionId != UINT64_MAX) { ImGui::TextUnformatted(tableQueueName(node)); }
    else { ImGui::TextDisabled("CPU"); }
    if (detailed) {
        ImGui::TableNextColumn(); drawDuration(node.gpuMilliseconds, node.gpuTimingAvailable);
        ImGui::TableNextColumn(); drawDuration(gpu.minimum, gpu.count != 0);
        ImGui::TableNextColumn(); drawDuration(gpu.maximum, gpu.count != 0);
        ImGui::TableNextColumn(); drawDuration(node.cpuMilliseconds);
        ImGui::TableNextColumn(); drawDuration(cpu.minimum, cpu.count != 0);
        ImGui::TableNextColumn(); drawDuration(cpu.maximum, cpu.count != 0);
    }
    if (open && children) {
        // Sort only a display copy of siblings. Original node indices and their
        // parent ID stack stay stable, so sorting never moves or reopens a scope.
        auto children = node.children;
        if (sort && sort->SortDirection != ImGuiSortDirection_None) {
            const auto column = static_cast<ProfilerColumn>(sort->ColumnUserID);
            std::stable_sort(children.begin(), children.end(), [&](size_t a, size_t b) {
                const auto& left = frame.nodes[a];
                const auto& right = frame.nodes[b];
                int order = 0;
                if (column == ProfilerColumn::Timer) { order = left.name.compare(right.name); }
                else if (column == ProfilerColumn::Queue) { order = std::string_view(tableQueueName(left)).compare(tableQueueName(right)); }
                else {
                    const double x = tableSortValue(left, rows[a], column);
                    const double y = tableSortValue(right, rows[b], column);
                    const bool xValid = std::isfinite(x), yValid = std::isfinite(y);
                    // Missing queries stay last in both directions; zero is valid.
                    if (xValid != yValid) { return xValid; }
                    if (!xValid) { return false; }
                    order = x < y ? -1 : x > y ? 1 : 0;
                }
                return sort->SortDirection == ImGuiSortDirection_Ascending ? order < 0 : order > 0;
            });
        }
        for (const size_t child : children) {
            drawProfilerTableNode(frame, child, depth + 1, detailed, rows, sort);
        }
        ImGui::TreePop();
    }
}

void drawProfilerTable(const EditorProfiler::Frame& frame, const EditorProfiler& profiler, bool detailed)
{
    if (frame.nodes.empty()) { ImGui::TextDisabled("No profiler samples yet."); return; }
    if (!ImGui::BeginTable("ProfilerTable", detailed ? 10 : 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
        ImGuiTableFlags_Sortable | ImGuiTableFlags_SortTristate, ImVec2(0, 0))) { return; }
    const auto setupColumn = [](const char* name, ProfilerColumn column, float width, bool numeric = true) {
        ImGui::TableSetupColumn(name, ImGuiTableColumnFlags_WidthFixed |
            (numeric ? ImGuiTableColumnFlags_PreferSortDescending : ImGuiTableColumnFlags_None), width, static_cast<ImGuiID>(column));
    };
    ImGui::TableSetupColumn("Timer", ImGuiTableColumnFlags_WidthStretch, 300, static_cast<ImGuiID>(ProfilerColumn::Timer));
    setupColumn("GPU avg ms", ProfilerColumn::GPUAverage, 90);
    setupColumn("CPU avg ms", ProfilerColumn::CPUAverage, 90);
    setupColumn("Queue", ProfilerColumn::Queue, 90, false);
    if (detailed) {
        setupColumn("GPU last", ProfilerColumn::GPULast, 85);
        setupColumn("GPU min", ProfilerColumn::GPUMinimum, 85);
        setupColumn("GPU max", ProfilerColumn::GPUMaximum, 85);
        setupColumn("CPU last", ProfilerColumn::CPULast, 85);
        setupColumn("CPU min", ProfilerColumn::CPUMinimum, 85);
        setupColumn("CPU max", ProfilerColumn::CPUMaximum, 85);
    }
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableHeadersRow();
    auto* specs = ImGui::TableGetSortSpecs();
    const auto* sort = specs && specs->SpecsCount > 0 ? &specs->Specs[0] : nullptr;
    // Appends, evictions and delayed GPU results already updated the aggregates.
    // Sorting reads them directly, independently of ImGui's SpecsDirty flag.
    std::vector<ProfilerTableRow> rows(frame.nodes.size());
    for (size_t i = 0; i < frame.nodes.size(); ++i) { rows[i] = profiler.historyStatistics(frame.nodes[i].scopeId); }
    drawProfilerTableNode(frame, 0, 0, detailed, rows, sort);
    if (specs) { specs->SpecsDirty = false; }
    ImGui::EndTable();
}

struct PlotSeries {
    std::string name;
    ImU32 color;
    std::vector<double> values;
};

// NaN means unavailable, producing a gap rather than a misleading zero sample.
void drawHistoryPlot(const char* title, const std::vector<uint64_t>& frames, const std::vector<PlotSeries>& series,
    const char* unit, float height, bool stacked = false, double reference = 0.0)
{
    ImGui::PushID(title);
    ImGui::TextUnformatted(title);
    if (frames.empty()) { ImGui::TextDisabled("No samples."); ImGui::PopID(); return; }
    const float width = std::max(ImGui::GetContentRegionAvail().x, 120.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 lo(pos.x + 55, pos.y + 8), hi(pos.x + width - 10, pos.y + height - 24);
    auto* draw = ImGui::GetWindowDrawList();
    ImGui::InvisibleButton("plot", ImVec2(width, height));
    draw->AddRectFilled(lo, hi, IM_COL32(25, 27, 31, 255));
    double maximum = reference;
    for (size_t i = 0; i < frames.size(); ++i) {
        double sum = 0;
        for (const auto& line : series) {
            if (i >= line.values.size() || !std::isfinite(line.values[i])) { continue; }
            sum = stacked ? sum + line.values[i] : std::max(sum, line.values[i]);
        }
        maximum = std::max(maximum, sum);
    }
    maximum = std::max(maximum * 1.08, 1.0);
    const auto xAt = [&](size_t i) {
        return lo.x + (hi.x - lo.x) * float(frames[i] - frames.front()) / float(std::max<uint64_t>(frames.back() - frames.front(), 1));
    };
    const auto yAt = [&](double value) { return hi.y - float(std::clamp(value / maximum, 0.0, 1.0)) * (hi.y - lo.y); };
    char text[96];
    for (int grid = 0; grid <= 4; ++grid) {
        const double value = maximum * grid / 4;
        const float y = yAt(value);
        draw->AddLine(ImVec2(lo.x, y), ImVec2(hi.x, y), IM_COL32(80, 82, 88, 100));
        std::snprintf(text, sizeof(text), "%.1f", value);
        draw->AddText(ImVec2(pos.x, y - 6), IM_COL32(190, 190, 195, 255), text);
    }
    std::vector<double> base(frames.size(), 0);
    for (const auto& line : series) {
        for (size_t i = 0; i < frames.size(); ++i) {
            if (i >= line.values.size() || !std::isfinite(line.values[i])) { continue; }
            const double value = line.values[i];
            const ImVec2 point(xAt(i), yAt(base[i] + value));
            if (i && std::isfinite(line.values[i - 1])) {
                const ImVec2 previous(xAt(i - 1), yAt(base[i - 1] + line.values[i - 1]));
                if (stacked) {
                    // Adjacent filled spans share an edge; antialiasing each one
                    // produces vertical seams in otherwise constant memory usage.
                    const auto flags = draw->Flags;
                    draw->Flags &= ~ImDrawListFlags_AntiAliasedFill;
                    draw->AddQuadFilled(previous, point, ImVec2(point.x, yAt(base[i])),
                        ImVec2(previous.x, yAt(base[i - 1])), (line.color & ~IM_COL32_A_MASK) | IM_COL32(0, 0, 0, 170));
                    draw->Flags = flags;
                }
                draw->AddLine(previous, point, line.color, 1.5f);
            } else { draw->AddCircleFilled(point, 2, line.color); }
        }
        if (stacked) {
            for (size_t i = 0; i < frames.size(); ++i) { if (std::isfinite(line.values[i])) { base[i] += line.values[i]; } }
        }
    }
    if (reference > 0) {
        const float y = yAt(reference);
        for (float x = lo.x; x < hi.x; x += 9) { draw->AddLine(ImVec2(x, y), ImVec2(std::min(x + 5, hi.x), y), IM_COL32(235, 215, 135, 200)); }
    }
    std::snprintf(text, sizeof(text), "%llu", static_cast<unsigned long long>(frames.front()));
    draw->AddText(ImVec2(lo.x, hi.y + 4), IM_COL32_WHITE, text);
    std::snprintf(text, sizeof(text), "frame %llu", static_cast<unsigned long long>(frames.back()));
    draw->AddText(ImVec2(std::max(lo.x, hi.x - ImGui::CalcTextSize(text).x), hi.y + 4), IM_COL32_WHITE, text);
    if (ImGui::IsItemHovered()) {
        const double fraction = std::clamp(double((ImGui::GetIO().MousePos.x - lo.x) / (hi.x - lo.x)), 0.0, 1.0);
        const auto target = frames.front() + uint64_t(fraction * double(frames.back() - frames.front()));
        auto it = std::lower_bound(frames.begin(), frames.end(), target);
        size_t i = it == frames.end() ? frames.size() - 1 : size_t(it - frames.begin());
        if (i && target - frames[i - 1] < frames[i] - target) { --i; }
        draw->AddLine(ImVec2(xAt(i), lo.y), ImVec2(xAt(i), hi.y), IM_COL32(255, 255, 255, 160));
        ImGui::BeginTooltip();
        ImGui::Text("Frame %llu", static_cast<unsigned long long>(frames[i]));
        for (const auto& line : series) {
            if (std::isfinite(line.values[i])) { ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(line.color), "%s: %.3f %s", line.name.c_str(), line.values[i], unit); }
            else { ImGui::TextDisabled("%s: pending / unavailable", line.name.c_str()); }
        }
        if (reference > 0) { ImGui::Text("Capacity / budget: %.1f %s", reference, unit); }
        ImGui::EndTooltip();
    }
    const float legendRight = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < series.size(); ++i) {
        if (i && ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + ImGui::CalcTextSize(series[i].name.c_str()).x < legendRight) {
            ImGui::SameLine();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, series[i].color);
        ImGui::TextWrapped("%s", series[i].name.c_str());
        ImGui::PopStyleColor();
    }
    ImGui::PopID();
}

std::vector<std::string> nodePath(const EditorProfiler::Frame& frame, size_t index)
{
    std::vector<std::string> path;
    while (index && index < frame.nodes.size()) { path.push_back(frame.nodes[index].name); index = frame.nodes[index].parent; }
    std::reverse(path.begin(), path.end());
    return path;
}

const EditorProfiler::Node* chooseChartScope(const EditorProfiler::Frame& frame, int& metric, std::vector<std::string>& path)
{
    ImGui::SetNextItemWidth(120);
    ImGui::Combo("Metric", &metric, "CPU\0GPU\0");
    const auto* selected = nodeByPath(frame, path);
    if (!selected || (metric == 1 && selected->renderGraphExecutionId == UINT64_MAX)) {
        for (size_t i = 0; i < frame.nodes.size(); ++i) {
            if (frame.nodes[i].renderGraphExecutionId != UINT64_MAX) { path = nodePath(frame, i); selected = &frame.nodes[i]; break; }
        }
    }
    ImGui::SetNextItemWidth(-65);
    if (ImGui::BeginCombo("Scope", selected ? selected->name.c_str() : "Frame")) {
        for (size_t i = 0; i < frame.nodes.size(); ++i) {
            const auto& node = frame.nodes[i];
            if (metric == 1 && node.renderGraphExecutionId == UINT64_MAX) { continue; }
            auto candidate = nodePath(frame, i);
            std::string label = "Frame";
            for (const auto& part : candidate) { label += " / " + part; }
            if (ImGui::Selectable(label.c_str(), candidate == path)) { path = std::move(candidate); selected = &node; }
        }
        ImGui::EndCombo();
    }
    return selected;
}

void drawTimingCharts(const EditorProfiler::Frame& frame, const EditorProfiler& profiler,
    int& metric, std::vector<std::string>& path, bool lines)
{
    const auto& history = profiler.history();
    const auto* selected = chooseChartScope(frame, metric, path);
    if (!selected) { return; }
    const bool gpu = metric == 1;
    ImGui::TextDisabled("Elapsed ms; nested and concurrent GPU intervals overlap.");
    std::vector<std::pair<std::vector<std::string>, const EditorProfiler::Node*>> paths{{path, selected}};
    for (auto child : selected->children) {
        auto childPath = path; childPath.push_back(frame.nodes[child].name);
        paths.push_back({std::move(childPath), &frame.nodes[child]});
    }
    if (lines) {
        std::vector<uint64_t> frames;
        for (const auto& sample : history) { frames.push_back(sample.index); }
        std::vector<PlotSeries> series;
        for (const auto& [p, node] : paths) {
            PlotSeries line{node->name, imguiColor(node->color), {}};
            for (const auto& sample : history) { line.values.push_back(sampleValue(nodeByPath(sample, p), gpu)); }
            series.push_back(std::move(line));
        }
        drawHistoryPlot(gpu ? "GPU history (ms)" : "CPU history (ms)", frames, series, "ms",
            std::max(160.0f, ImGui::GetContentRegionAvail().y - 95));
    } else {
        double max = 0.001;
        for (const auto& [p, node] : paths) {
            const auto stats = profiler.historyStatistics(node->scopeId);
            max = std::max(max, (gpu ? stats.gpu : stats.cpu).average);
        }
        for (const auto& [p, node] : paths) {
            const auto stats = profiler.historyStatistics(node->scopeId);
            const auto& avg = gpu ? stats.gpu : stats.cpu;
            ImGui::TextUnformatted(node->name.c_str());
            char label[64];
            if (avg.count) { std::snprintf(label, sizeof(label), "%.3f ms (%zu samples)", avg.average, avg.count); }
            else { std::snprintf(label, sizeof(label), "pending / unavailable"); }
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, imguiColor(node->color));
            ImGui::ProgressBar(float(avg.average / max), ImVec2(-1, 20), label);
            ImGui::PopStyleColor();
        }
    }
}

void drawStreaming(const std::vector<EditorProfiler::StreamingHistory>& sources, std::string& selected, bool& showBudget)
{
    if (sources.empty()) { ImGui::TextDisabled("The current scene has no meshlet streaming runtime."); return; }
    auto it = std::find_if(sources.begin(), sources.end(), [&](const auto& source) { return source.passName == selected; });
    if (it == sources.end()) { it = sources.begin(); selected = it->passName; }
    if (ImGui::BeginCombo("Source", selected.c_str())) {
        for (const auto& source : sources) { if (ImGui::Selectable(source.passName.c_str(), source.passName == selected)) { selected = source.passName; } }
        ImGui::EndCombo();
    }
    it = std::find_if(sources.begin(), sources.end(), [&](const auto& source) { return source.passName == selected; });
    ImGui::TextWrapped("Asset: %s", it->assetPath.c_str());
    if (it->samples.empty()) { return; }
    const auto& last = it->samples.back();
    constexpr double mib = 1024.0 * 1024.0;
    ImGui::Text("Geometry %.1f / %.1f MiB | Resident %u / %u pages", last.geometryUsedBytes / mib,
        last.geometryBudgetBytes / mib, last.residentPages, last.totalPages);
    if (last.clasEnabled) {
        ImGui::Text("CLAS %.1f / %.1f MiB | Resident %u pages / %u clusters", last.clasUsedBytes / mib,
            last.clasCapacityBytes / mib, last.clasResidentPages, last.clasResidentClusters);
        ImGui::Text("CLAS backing %.1f MiB in %u chunks | Free inside backing %.1f MiB",
            last.clasAllocatedBytes / mib, last.clasStorageChunks,
            (last.clasAllocatedBytes >= last.clasUsedBytes ? last.clasAllocatedBytes - last.clasUsedBytes : 0) / mib);
        ImGui::Text("CLAS start/grow %.1f / %.1f MiB | empty %.1f MiB | growths %llu | returned %.1f MiB",
            last.clasStartBytes / mib, last.clasGrowBytes / mib, last.clasEmptyBytes / mib,
            static_cast<unsigned long long>(last.clasGrowthCount), last.clasReleasedBytes / mib);
        ImGui::Text("CLAS roots used/backing %.1f / %.1f MiB | transient %.1f / %.1f MiB | fragmented free %.1f MiB",
            last.clasPersistentUsedBytes / mib, last.clasPersistentAllocatedBytes / mib,
            last.clasTransientUsedBytes / mib, last.clasTransientAllocatedBytes / mib, last.clasFragmentedFreeBytes / mib);
        ImGui::Text("CLAS root growth %.1f MiB", last.clasPersistentGrowBytes / mib);
        if (last.clasEncodedBytes) {
            ImGui::Text("CLAS encoded %.1f MiB | Fixed slots %.1f MiB | Build/move workspace %.1f MiB",
                last.clasEncodedBytes / mib, last.clasWorstCaseBytes / mib, last.clasScratchBytes / mib);
            ImGui::Text("CLAS relocated %u clusters this frame", last.clasMovedClusters);
        }
        ImGui::Text("CLAS built %u pages / %u clusters | Pending %u | Retiring %u | Budget deferred %u",
            last.clasBuiltPages, last.clasBuiltClusters, last.clasPendingPages, last.clasRetiringPages, last.clasRejectedPages);
    }
    else { ImGui::TextDisabled("CLAS: disabled for this rendering path"); }
    ImGui::Text("Pending pages %u | I/O queued %u, active %u | Upload pipeline %u", last.pendingPages, last.ioQueued, last.ioActive, last.uploadQueued);
    ImGui::Text("Requests %u | Completed uploads %u | Evictions %u | Upload %.3f MiB/frame", last.requests, last.uploads, last.evictions, last.uploadBytes / mib);
    const auto& speed = last.throughput;
    ImGui::Text("Load %.1f MiB/s (%.0f pages/s) | Prepared %.1f MiB/s", speed.loadedStoredMiBPerSecond,
        speed.loadedPagesPerSecond, speed.preparedMiBPerSecond);
    ImGui::Text("Copy payload %.1f MiB/s | Geometry ready %.1f MiB/s (%.0f pages/s)", speed.transferMiBPerSecond,
        speed.geometryReadyMiBPerSecond, speed.geometryReadyPagesPerSecond);
    ImGui::TextDisabled("Rolling %.2f s; load counts mapped file payload, not physical disk I/O. Ready = completed upload receipt.", speed.windowSeconds);
    ImGui::Text("GPU installation pages %llu | Small-batch CPU pages %llu",
        static_cast<unsigned long long>(last.totalGpuDecompressedPages),
        static_cast<unsigned long long>(speed.totals.smallBatchCpuPages));
    if (last.feedbackFrame == UINT64_MAX) { ImGui::TextDisabled("Feedback: pending"); }
    else { ImGui::TextDisabled("Stream frame %llu | Feedback frame %llu (age %llu)",
        static_cast<unsigned long long>(last.frameIndex), static_cast<unsigned long long>(last.feedbackFrame),
        static_cast<unsigned long long>(last.frameIndex >= last.feedbackFrame ? last.frameIndex - last.feedbackFrame : 0)); }
    if (last.requestOverflows || last.allocationFailures || last.loadFailures) {
        ImGui::TextColored(ImVec4(1, .65f, .2f, 1), "Request overflow %u | Allocation failures %u | Total I/O failures %llu",
            last.requestOverflows, last.allocationFailures, static_cast<unsigned long long>(last.loadFailures));
    }
    ImGui::TextDisabled("CPU-visible counters; requests refer to completed GPU feedback. Upload pipeline includes I/O.");
    if (last.lodTransitionTelemetryEnabled && ImGui::CollapsingHeader("LOD Transition Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Demanded %u | Own page blocked %u | Dependency blocked %u", last.lodDemandedGroups,
            last.lodOwnPageBlockedGroups, last.lodDependencyBlockedGroups);
        ImGui::Text("Selected threshold %u groups / %u clusters | Catch-up %u / %u",
            last.lodThresholdSelectedGroups, last.lodThresholdSelectedClusters,
            last.lodCatchupSelectedGroups, last.lodCatchupSelectedClusters);
        ImGui::Text("Unclassified new selections %u / %u | Catch-up active %u",
            last.lodUnclassifiedSelectedGroups, last.lodUnclassifiedSelectedClusters, last.lodCatchupActivatedGroups);
        ImGui::TextDisabled("Instance-groups in emitted geometry cut; excludes capacity fallback. History %.1f MiB.",
            last.lodTransitionHistoryBytes / mib);
        ImGui::TextDisabled("Threshold/catch-up require consecutive automatic visible frames. First/reentry remains unclassified.");
    }
    if (ImGui::CollapsingHeader("Page Prefetch")) {
        ImGui::Text("Retention %s | Demand reserve %.1f MiB | Reclaim target %.1f MiB",
            last.adaptivePageRetentionEnabled ? "bounded headroom" : "legacy", last.geometryDemandReserveBytes / mib,
            last.geometryReclaimReserveBytes / mib);
        ImGui::Text("Cold cache %.1f MiB | Pending frees %.1f MiB | Evicted %.1f MiB (%u unused prefetch)",
            last.coldResidentBytes / mib, last.pendingFreeBytes / mib, last.evictedGeometryBytes / mib,
            last.evictedPrefetchPages);
        ImGui::Text("Prefetch %s | Memory watermark %s | Queue %s", last.prefetchEnabled ? "enabled" : "disabled",
            last.prefetchMemoryWatermarkBlocked ? "blocked" : "available", last.prefetchQueueBlocked ? "blocked" : "available");
        ImGui::Text("Forecast %s | Horizon %.1f ms | Move %.3f | Rotate %.2f deg",
            last.prefetchForecastActive ? "active" : "inactive", last.prefetchHorizonMilliseconds,
            last.prefetchTranslationDistance, last.prefetchRotationDegrees);
        ImGui::Text("Recent demand latency p95 %.1f ms (%llu samples) | GPU requests/dropped %u/%u",
            last.prefetchDemandLatencyP95Milliseconds, static_cast<unsigned long long>(last.prefetchLatencySamples),
            last.prefetchGpuRequests, last.prefetchGpuDropped);
        ImGui::Text("Total admitted/used/deferred %llu/%llu/%llu",
            static_cast<unsigned long long>(last.totalPrefetchAdmitted), static_cast<unsigned long long>(last.totalPrefetchUsed),
            static_cast<unsigned long long>(last.totalPrefetchDeferred));
    }
    if (ImGui::CollapsingHeader("CPU Request / Reclaim Work")) {
        const auto& work = last.cpuWork;
        ImGui::Text("Allocation attempts %u | Budget retries suppressed %u",
            work.allocationAttempts, work.budgetRetrySuppressed);
        ImGui::Text("Demand transitions %u | Epoch updates %u | Stale batches %u",
            work.demandTransitions, work.demandEpochUpdates, work.demandStaleBatches);
        ImGui::Text("Unused membership checks %u | Prefetch checks %u | Cold candidate checks %u",
            work.demandMembershipTests, work.demandPrefetchVisited, work.coldCandidateTests);
        ImGui::Text("Resident visits %u | Newer than feedback %u", work.demandVisited, work.demandNewerThanFeedback);
        ImGui::Text("Unused %u | Refreshed %u | Incomplete feedback protected %u",
            work.demandUnused, work.demandRefreshed, work.demandIncompleteProtected);
        ImGui::Separator();
        ImGui::Text("Cold candidates %u | Scheduling visits %u | CLAS lookups %u",
            work.coldCandidates, work.coldVisited, work.coldClasLookups);
        ImGui::Text("Rejected: state %u | Age %u | Schedule failed %u",
            work.coldStateRejected, work.coldAgeRejected, work.coldScheduleFailed);
        ImGui::Text("Scheduled: pressure %u | Retention %u | Pending free credits %u",
            work.coldPressureScheduled, work.coldRetentionScheduled, work.pendingFreePages);
        ImGui::TextDisabled("Current frame work counts, including repeated visits. Timings: Table > Stream Begin.");
    }
    std::vector<uint64_t> frames;
    std::vector<PlotSeries> memory{{"Geometry", IM_COL32(64, 218, 100, 255), {}}, {"CLAS", IM_COL32(75, 151, 250, 255), {}}};
    std::vector<PlotSeries> pages{{"Requests", IM_COL32(255, 211, 92, 255), {}}, {"Uploads", IM_COL32(92, 217, 161, 255), {}}, {"Evictions", IM_COL32(246, 123, 123, 255), {}}};
    std::vector<PlotSeries> uploads{{"Upload MiB/frame", IM_COL32(104, 178, 248, 255), {}}};
    std::vector<PlotSeries> throughput{{"Loaded payload", IM_COL32(255, 211, 92, 255), {}},
        {"Copy payload", IM_COL32(104, 178, 248, 255), {}}, {"Geometry ready", IM_COL32(92, 217, 161, 255), {}}};
    std::vector<PlotSeries> pending{{"Pending pages", IM_COL32(255, 211, 92, 255), {}}, {"I/O queued", IM_COL32(246, 123, 123, 255), {}}, {"I/O active", IM_COL32(104, 178, 248, 255), {}}};
    std::vector<PlotSeries> clasTraffic{{"Built pages", IM_COL32(75, 151, 250, 255), {}},
        {"Pending pages", IM_COL32(255, 211, 92, 255), {}}, {"Retiring pages", IM_COL32(246, 123, 123, 255), {}}};
    for (const auto& sample : it->samples) {
        clasTraffic[0].values.push_back(sample.clasBuiltPages);
        clasTraffic[1].values.push_back(sample.clasPendingPages);
        clasTraffic[2].values.push_back(sample.clasRetiringPages);
        frames.push_back(sample.frameIndex);
        memory[0].values.push_back(sample.geometryUsedBytes / mib); memory[1].values.push_back(sample.clasUsedBytes / mib);
        pages[0].values.push_back(sample.requests); pages[1].values.push_back(sample.uploads); pages[2].values.push_back(sample.evictions);
        uploads[0].values.push_back(sample.uploadBytes / mib);
        throughput[0].values.push_back(sample.throughput.loadedStoredMiBPerSecond);
        throughput[1].values.push_back(sample.throughput.transferMiBPerSecond);
        throughput[2].values.push_back(sample.throughput.geometryReadyMiBPerSecond);
        pending[0].values.push_back(sample.pendingPages); pending[1].values.push_back(sample.ioQueued); pending[2].values.push_back(sample.ioActive);
    }
    if (!last.clasEnabled) { memory.pop_back(); }
    if (ImGui::CollapsingHeader("Streaming Memory", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Include capacity in memory chart", &showBudget);
        drawHistoryPlot("Streaming memory (MiB)", frames, memory, "MiB", 200, true,
            showBudget ? (last.geometryBudgetBytes + last.clasCapacityBytes) / mib : 0.0);
    }
    if (last.textureStreaming && ImGui::CollapsingHeader("Texture Residency")) {
        ImGui::Text("Resident %.1f / %.1f MiB | Pending reserve %.2f MiB | Retiring %.2f MiB",
            last.textureResidentBytes/mib, last.textureBudgetBytes/mib, last.texturePendingBytes/mib, last.textureRetiredBytes/mib);
        ImGui::Text("Refined %u | Requested %u | In flight %u",last.textureRefinedImages,last.textureRequestedImages,last.texturePendingImages);
        ImGui::Text("Upgrades %llu | Cold downgrades %llu | Budget deferrals %llu",
            (unsigned long long)last.textureUpgrades,(unsigned long long)last.textureDowngrades,(unsigned long long)last.textureBudgetDeferrals);
        ImGui::Text("Feedback frames %llu | Uploaded %.2f MiB | Max request latency %llu frames",
            (unsigned long long)last.textureFeedbackFrames,last.textureUploadBytes/mib,(unsigned long long)last.textureMaxRequestFrames);
        if (last.textureUpload.sequence) {
            const auto& upload = last.textureUpload;
            if (upload.gpuTimingAvailable) { ImGui::Text("Last independent upload GPU %.3f ms (graphics)", upload.gpuMilliseconds); }
            else { ImGui::TextUnformatted("Last independent upload GPU: unavailable"); }
            ImGui::Text("Submit frame %llu -> observed %llu | %.2f ms incl. queue/poll | %.2f MiB",
                (unsigned long long)upload.submitFrame, (unsigned long long)upload.completionFrame,
                upload.completionObservedMilliseconds, upload.bytes/mib);
        }
        std::vector<PlotSeries> textures{{"Resident",IM_COL32(105,194,242,255),{}},
            {"Pending",IM_COL32(255,211,92,255),{}},{"Retiring",IM_COL32(246,123,123,255),{}}};
        for (const auto& sample : it->samples) {
            textures[0].values.push_back(sample.textureResidentBytes/mib);
            textures[1].values.push_back(sample.texturePendingBytes/mib);
            textures[2].values.push_back(sample.textureRetiredBytes/mib);
        }
        drawHistoryPlot("Texture allocation (MiB)",frames,textures,"MiB",180,true,last.textureBudgetBytes/mib);
        ImGui::TextDisabled("Resident/retiring are image allocations; pending is reserved capacity. Alpha/displacement stay pinned.");
    }
    if (ImGui::CollapsingHeader("Page Traffic")) {
        drawHistoryPlot("Page traffic (pages/frame)", frames, pages, "pages", 170);
    }
    if (ImGui::CollapsingHeader("Loading Speed", ImGuiTreeNodeFlags_DefaultOpen)) {
        drawHistoryPlot("Streaming throughput (MiB/s)", frames, throughput, "MiB/s", 170);
    }
    if (ImGui::CollapsingHeader("Upload Traffic")) {
        drawHistoryPlot("Upload traffic (MiB/frame)", frames, uploads, "MiB", 150);
    }
    if (last.clasEnabled && ImGui::CollapsingHeader("CLAS Streaming")) {
        drawHistoryPlot("CLAS build / backlog (pages)", frames, clasTraffic, "pages", 150);
    }
    if (ImGui::CollapsingHeader("Streaming Backlog")) {
        drawHistoryPlot("Streaming backlog (pages)", frames, pending, "pages", 150);
    }
}

void drawPieSlice(
    ImDrawList* drawList,
    const ImVec2& center,
    float radius,
    float startRadians,
    float endRadians,
    ImU32 color)
{
    constexpr int kMaxSegments = 64;
    const float angle = std::max(endRadians - startRadians, 0.0f);
    const int segments = std::clamp(static_cast<int>(angle / (2.0f * kPi) * kMaxSegments), 2, kMaxSegments);
    drawList->PathLineTo(center);
    for (int segment = 0; segment <= segments; ++segment) {
        const float t = static_cast<float>(segment) / static_cast<float>(segments);
        const float radians = startRadians + (endRadians - startRadians) * t;
        drawList->PathLineTo(ImVec2(
            center.x + std::cos(radians) * radius,
            center.y + std::sin(radians) * radius));
    }
    drawList->PathFillConvex(color);
}

void drawPieChart(const EditorProfiler::Frame& frame)
{
    if (frame.nodes.empty() || frame.nodes[0].children.empty()) {
        ImGui::TextDisabled("No profiler samples yet.");
        return;
    }

    const float availableWidth = ImGui::GetContentRegionAvail().x;
    const float radius = std::min(availableWidth * 0.25f, 120.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 center(pos.x + radius + 10.0f, pos.y + radius + 10.0f);
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    const EditorProfiler::Node& root = frame.nodes[0];
    const double total = std::max(root.cpuMilliseconds, 0.001);
    float angle = -0.5f * kPi;
    for (size_t childIndex : root.children) {
        const EditorProfiler::Node& child = frame.nodes[childIndex];
        const float slice = static_cast<float>(child.cpuMilliseconds / total) * 2.0f * kPi;
        drawPieSlice(drawList, center, radius, angle, angle + slice, imguiColor(child.color));
        angle += slice;
    }
    drawList->AddCircle(center, radius, IM_COL32(100, 100, 100, 160), 64, 1.0f);
    ImGui::Dummy(ImVec2(availableWidth, radius * 2.0f + 24.0f));

    if (ImGui::BeginTable("ProfilerPieLegend", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("Section", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("CPU ms", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Share", ImGuiTableColumnFlags_WidthFixed, 72.0f);
        ImGui::TableHeadersRow();
        for (size_t childIndex : root.children) {
            const EditorProfiler::Node& child = frame.nodes[childIndex];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(imguiColor(child.color)), "%s", child.name.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", child.cpuMilliseconds);
            ImGui::TableNextColumn();
            ImGui::Text("%.1f%%", child.cpuMilliseconds * 100.0 / total);
        }
        ImGui::EndTable();
    }
}

template <typename Changed>
void applyRenderGraphGpuStats(
    std::vector<EditorProfiler::Node>& nodes,
    const render::RenderGraphExecutionStats& stats,
    Changed changed)
{
    for (EditorProfiler::Node& node : nodes) {
        if (node.renderGraphExecutionId != stats.executionId) {
            continue;
        }
        const double previous = sampleValue(&node, true);
        if (node.renderGraphNodeId == UINT32_MAX) {
            node.gpuMilliseconds = stats.gpuMilliseconds;
            node.gpuTimingAvailable = stats.gpuTimingAvailable;
            changed(node, previous);
            continue;
        }

        const auto iter = std::find_if(
            stats.nodes.begin(),
            stats.nodes.end(),
            [&](const render::RenderGraphNodeExecutionStat& stat) {
                return stat.id == node.renderGraphNodeId;
            });
        if (iter != stats.nodes.end()) {
            if (node.renderGraphSectionIndex == UINT32_MAX) {
                node.gpuMilliseconds = iter->gpuMilliseconds;
                node.gpuTimingAvailable = iter->gpuTimingAvailable;
            } else if (node.renderGraphSectionIndex < iter->sections.size()) {
                const auto& section = iter->sections[node.renderGraphSectionIndex];
                node.gpuMilliseconds = section.gpuMilliseconds;
                node.gpuTimingAvailable = section.gpuTimingAvailable;
            }
            changed(node, previous);
        }
    }
}

void applyRenderGraphGpuStats(std::vector<EditorProfiler::Node>& nodes,
    const render::RenderGraphExecutionStats& stats)
{
    applyRenderGraphGpuStats(nodes, stats, [](const auto&, double) {});
}

} // namespace

void EditorProfiler::SampleAggregate::add(double value)
{
    if (!std::isfinite(value)) { return; }
    values.insert(value);
    sum += value;
}

void EditorProfiler::SampleAggregate::remove(double value)
{
    if (!std::isfinite(value)) { return; }
    const auto found = values.find(value);
    if (found == values.end()) { return; }
    values.erase(found);
    sum = values.empty() ? 0.0 : sum - value;
}

EditorProfiler::Aggregate EditorProfiler::SampleAggregate::statistics() const
{
    if (values.empty()) { return {}; }
    return {sum / double(values.size()), *values.begin(), *values.rbegin(), values.size()};
}

EditorProfiler::HistoryStatistics EditorProfiler::historyStatistics(size_t scopeId) const
{
    if (scopeId >= scopes_.size()) { return {}; }
    return {scopes_[scopeId].cpu.statistics(), scopes_[scopeId].gpu.statistics()};
}

void EditorProfiler::registerFrameScopes(std::vector<Node>& nodes)
{
    if (nodes.empty()) { return; }
    const uint64_t revision = ++scopeRevision_;
    if (scopeTree_.nodes.empty()) {
        scopeTree_.nodes.emplace_back();
        scopes_.emplace_back();
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        const size_t parent = i && node.parent < i ? nodes[node.parent].scopeId : 0;
        node.scopeId = SIZE_MAX;
        if (parent >= scopes_.size()) { continue; }
        size_t target = 0;
        if (i) {
            if (scopes_[parent].children.find(node.name) == scopes_[parent].children.end() &&
                scopes_.size() >= kProfilerScopeTreeLimit) {
                scopeTree_.profilingOverflow = true;
                continue;
            }
            auto& siblings = scopes_[parent].children[node.name];
            size_t occurrence = 0;
            if (!siblings.empty()) {
                auto& first = scopes_[siblings.front()];
                if (first.occurrenceRevision != revision) {
                    first.occurrenceRevision = revision;
                    first.occurrences = 0;
                }
                occurrence = first.occurrences++;
            }
            if (occurrence >= siblings.size()) {
                if (scopes_.size() >= kProfilerScopeTreeLimit) {
                    scopeTree_.profilingOverflow = true;
                    continue;
                }
                target = scopes_.size();
                const bool first = siblings.empty();
                siblings.push_back(target);
                scopes_.emplace_back();
                scopeTree_.nodes.emplace_back();
                scopeTree_.nodes[parent].children.push_back(target);
                if (first) {
                    scopes_[target].occurrenceRevision = revision;
                    scopes_[target].occurrences = 1;
                }
            } else {
                target = siblings[occurrence];
            }
        }
        node.scopeId = target;
        auto children = std::move(scopeTree_.nodes[target].children);
        scopeTree_.nodes[target] = node;
        scopeTree_.nodes[target].parent = parent;
        scopeTree_.nodes[target].children = std::move(children);
    }
}

void EditorProfiler::addHistoryFrame(const Frame& frame)
{
    std::vector<uint64_t> executions;
    for (const auto& node : frame.nodes) {
        if (node.scopeId < scopes_.size()) {
            auto& scope = scopes_[node.scopeId];
            scope.cpu.add(node.cpuMilliseconds);
            scope.gpu.add(sampleValue(&node, true));
        }
        if (node.renderGraphExecutionId != UINT64_MAX &&
            std::find(executions.begin(), executions.end(), node.renderGraphExecutionId) == executions.end()) {
            executions.push_back(node.renderGraphExecutionId);
        }
    }
    for (uint64_t execution : executions) { historyExecutions_.emplace(execution, frame.index); }
}

void EditorProfiler::removeHistoryFrame(const Frame& frame)
{
    std::vector<uint64_t> executions;
    for (const auto& node : frame.nodes) {
        if (node.scopeId < scopes_.size()) {
            auto& scope = scopes_[node.scopeId];
            scope.cpu.remove(node.cpuMilliseconds);
            scope.gpu.remove(sampleValue(&node, true));
        }
        if (node.renderGraphExecutionId != UINT64_MAX &&
            std::find(executions.begin(), executions.end(), node.renderGraphExecutionId) == executions.end()) {
            executions.push_back(node.renderGraphExecutionId);
        }
    }
    for (uint64_t execution : executions) {
        const auto [first, last] = historyExecutions_.equal_range(execution);
        for (auto it = first; it != last;) {
            if (it->second == frame.index) { it = historyExecutions_.erase(it); }
            else { ++it; }
        }
    }
}

EditorProfiler::FrameScope::FrameScope(EditorProfiler* profiler)
    : profiler_(profiler)
{
}

EditorProfiler::FrameScope::~FrameScope()
{
    if (profiler_ != nullptr) {
        profiler_->endFrame();
    }
}

EditorProfiler::FrameScope::FrameScope(FrameScope&& other) noexcept
    : profiler_(std::exchange(other.profiler_, nullptr))
{
}

EditorProfiler::FrameScope& EditorProfiler::FrameScope::operator=(FrameScope&& other) noexcept
{
    if (this != &other) {
        if (profiler_ != nullptr) {
            profiler_->endFrame();
        }
        profiler_ = std::exchange(other.profiler_, nullptr);
    }
    return *this;
}

EditorProfiler::Scope::Scope(EditorProfiler* profiler, size_t nodeIndex)
    : profiler_(profiler)
    , nodeIndex_(nodeIndex)
{
}

EditorProfiler::Scope::~Scope()
{
    if (profiler_ != nullptr) {
        profiler_->endSection(nodeIndex_);
    }
}

EditorProfiler::Scope::Scope(Scope&& other) noexcept
    : profiler_(std::exchange(other.profiler_, nullptr))
    , nodeIndex_(other.nodeIndex_)
{
}

EditorProfiler::Scope& EditorProfiler::Scope::operator=(Scope&& other) noexcept
{
    if (this != &other) {
        if (profiler_ != nullptr) {
            profiler_->endSection(nodeIndex_);
        }
        profiler_ = std::exchange(other.profiler_, nullptr);
        nodeIndex_ = other.nodeIndex_;
    }
    return *this;
}

void EditorProfiler::beginCapture()
{
    capturedFrames_.clear();
    capturedExecutions_.clear();
    capturing_ = true;
    captureOverflow_ = false;
}

EditorProfiler::FrameScope EditorProfiler::beginFrame()
{
    if (frameActive_) {
        endFrame();
    }

    currentNodes_.clear();
    currentStreaming_.clear();
    currentOverflow_ = false;
    stack_.clear();
    frameActive_ = true;
    beginSection("Frame", 0xffffffffu);
    return FrameScope(this);
}

EditorProfiler::Scope EditorProfiler::scope(std::string_view name, uint32_t color)
{
    if (!frameActive_) {
        return {};
    }
    return Scope(this, beginSection(name, color == 0 ? colorFromName(name) : color));
}

void EditorProfiler::addCpuProfile(const std::vector<render::RenderGraphProfileSection>& sections)
{
    if (!frameActive_) { return; }
    const size_t parent = stack_.empty() ? 0 : stack_.back();
    std::vector<size_t> nodes;
    for (const auto& section : sections) {
        const auto index = addFinishedSection(section.parent < nodes.size() ? nodes[section.parent] : parent,
            section.name, colorFromName(section.name), section.cpuMilliseconds);
        currentNodes_[index].cpuOnly = true;
        nodes.push_back(index);
    }
}

void EditorProfiler::addRenderGraphStats(const render::RenderGraphExecutionStats& stats)
{
    if (!frameActive_ || stats.nodes.empty()) {
        return;
    }

    if (graphGeneration_ != stats.graphGeneration) {
        clearHistory();
        latestFrame_ = {};
        graphGeneration_ = stats.graphGeneration;
    }
    currentStreaming_.insert(currentStreaming_.end(), stats.streaming.begin(), stats.streaming.end());
    currentOverflow_ |= stats.profilingOverflow;
    const size_t parent = stack_.empty() ? 0 : stack_.back();
    std::vector<size_t> preparationNodes;
    preparationNodes.reserve(stats.preparation.size());
    for (const auto& section : stats.preparation) {
        const bool nested = section.parent < preparationNodes.size();
        const size_t sectionParent = nested ? preparationNodes[section.parent] : parent;
        const auto index = addFinishedSection(sectionParent,
            nested ? section.name : "Graph preparation / " + section.name,
            colorFromName(section.name), section.cpuMilliseconds);
        currentNodes_[index].cpuOnly = true;
        preparationNodes.push_back(index);
    }
    const size_t group = addFinishedSection(
        parent,
        "RenderGraph GPU envelope",
        colorFromName("RenderGraph GPU envelope"),
        stats.cpuMilliseconds);
    currentNodes_[group].gpuMilliseconds = stats.gpuMilliseconds;
    currentNodes_[group].gpuTimingAvailable = stats.gpuTimingAvailable;
    currentNodes_[group].renderGraphExecutionId = stats.executionId;
    for (const render::RenderGraphNodeExecutionStat& stat : stats.nodes) {
        const size_t nodeIndex = addFinishedSection(
            group,
            stat.name + " (" + stat.type + ")",
            colorFromName(stat.type),
            stat.cpuMilliseconds);
        currentNodes_[nodeIndex].gpuMilliseconds = stat.gpuMilliseconds;
        currentNodes_[nodeIndex].gpuTimingAvailable = stat.gpuTimingAvailable;
        currentNodes_[nodeIndex].renderGraphExecutionId = stats.executionId;
        currentNodes_[nodeIndex].renderGraphNodeId = stat.id;
        currentNodes_[nodeIndex].queue = stat.queue;
        std::vector<size_t> sectionNodes;
        for (uint32_t i = 0; i < stat.sections.size(); ++i) {
            const auto& section = stat.sections[i];
            const size_t sectionParent = section.parent < sectionNodes.size() ? sectionNodes[section.parent] : nodeIndex;
            const size_t child = addFinishedSection(sectionParent, section.name, colorFromName(section.name), section.cpuMilliseconds);
            auto& node = currentNodes_[child];
            node.gpuMilliseconds = section.gpuMilliseconds;
            node.gpuTimingAvailable = section.gpuTimingAvailable;
            node.renderGraphExecutionId = stats.executionId;
            node.renderGraphNodeId = stat.id;
            node.renderGraphSectionIndex = i;
            node.queue = section.queue;
            node.cpuOnly = section.cpuOnly;
            sectionNodes.push_back(child);
        }
    }
}

void EditorProfiler::updateRenderGraphGpuStats(const render::RenderGraphExecutionStats& stats)
{
    const auto captured = capturedExecutions_.find({stats.graphGeneration, stats.executionId});
    if (captured != capturedExecutions_.end()) {
        applyRenderGraphGpuStats(capturedFrames_[captured->second].nodes, stats);
        capturedExecutions_.erase(captured);
    }
    if (stats.graphGeneration != graphGeneration_) { return; }
    applyRenderGraphGpuStats(currentNodes_, stats);
    applyRenderGraphGpuStats(latestFrame_.nodes, stats);
    const auto [first, last] = historyExecutions_.equal_range(stats.executionId);
    for (auto it = first; it != last; ++it) {
        if (history_.empty() || it->second < history_.front().index) { continue; }
        const uint64_t index = it->second - history_.front().index;
        if (index >= history_.size() || history_[index].index != it->second) { continue; }
        applyRenderGraphGpuStats(history_[index].nodes, stats, [&](const Node& node, double previous) {
            if (node.scopeId >= scopes_.size()) { return; }
            const double value = sampleValue(&node, true);
            if (value == previous || (!std::isfinite(value) && !std::isfinite(previous))) { return; }
            auto& aggregate = scopes_[node.scopeId].gpu;
            aggregate.remove(previous);
            aggregate.add(value);
        });
    }
}

const EditorProfiler::Frame& EditorProfiler::displayFrame() const
{
    if (std::none_of(latestFrame_.nodes.begin(), latestFrame_.nodes.end(), [](const auto& node) {
        return node.renderGraphExecutionId != UINT64_MAX;
    })) { return latestFrame_; }
    for (auto it = history_.rbegin(); it != history_.rend(); ++it) {
        if (std::any_of(it->nodes.begin(), it->nodes.end(), [](const auto& node) { return node.gpuTimingAvailable; })) { return *it; }
    }
    return latestFrame_;
}

void EditorProfiler::clearHistory()
{
    history_.clear();
    historyExecutions_.clear();
    scopeTree_ = {};
    scopes_.clear();
    scopeRevision_ = 0;
    // Clear keeps the latest raw sample for display, but it is not a new
    // history sample. Remap its identities lazily if that view is requested.
    for (auto& node : latestFrame_.nodes) { node.scopeId = SIZE_MAX; }
    for (auto& source : streamingHistory_) { source.samples.clear(); }
}

EditorProfiler::Frame EditorProfiler::presentationFrame()
{
    if (scopeTree_.nodes.empty()) { registerFrameScopes(latestFrame_.nodes); }
    const auto& displayed = displayFrame();
    Frame result = scopeTree_;
    result.index = displayed.index;
    result.profilingOverflow |= displayed.profilingOverflow;
    for (auto& node : result.nodes) {
        node.cpuMilliseconds = std::numeric_limits<double>::quiet_NaN();
        node.gpuMilliseconds = 0;
        node.gpuTimingAvailable = false;
    }
    for (const auto& node : displayed.nodes) {
        if (node.scopeId >= result.nodes.size()) { continue; }
        auto& target = result.nodes[node.scopeId];
        target.cpuMilliseconds = node.cpuMilliseconds;
        target.gpuMilliseconds = node.gpuMilliseconds;
        target.gpuTimingAvailable = node.gpuTimingAvailable;
        target.cpuOnly = node.cpuOnly;
        target.renderGraphExecutionId = node.renderGraphExecutionId;
        target.renderGraphNodeId = node.renderGraphNodeId;
        target.renderGraphSectionIndex = node.renderGraphSectionIndex;
        target.queue = node.queue;
    }
    return result;
}

bool EditorProfiler::drawWindow(bool* open, const GraphicsCaptureControls& graphicsCapture)
{
    if (open != nullptr && !*open) {
        return false;
    }

    bool captureRequested = false;
    ImGui::SetNextWindowSize(ImVec2(820.0f, 560.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Profiler", open)) {
        ImGui::End();
        return false;
    }

    ImGui::BeginDisabled(!graphicsCapture.canCapture);
    if (ImGui::Button(graphicsCapture.gpuTrace ? "Export Current View GPU Trace" : "Export View Capture + GPU Trace")) {
        captureRequested = true;
    }
    ImGui::EndDisabled();

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (!graphicsCapture.sdkCompiled) {
            ImGui::SetTooltip("Nsight Graphics SDK was not available when Metallic was built.");
        } else if (!graphicsCapture.runtimeEnabled) {
            ImGui::SetTooltip("Restart Metallic with --nsight-capture to collect a capture and its replay GPU Trace.");
        } else if (graphicsCapture.capturePending) {
            ImGui::SetTooltip("An Nsight export is already pending.");
        } else if (!graphicsCapture.canCapture) {
            ImGui::SetTooltip(
                "%s",
                graphicsCapture.statusText != nullptr && graphicsCapture.statusText[0] != '\0'
                    ? graphicsCapture.statusText
                    : "The current View is not ready for capture.");
        } else {
            ImGui::SetTooltip(
                "%s", graphicsCapture.gpuTrace
                    ? "Profile the next complete View frame in the live application."
                    : "Save the next complete View frame, then collect GPU Trace from its replay. "
                      "Metrics are automatic. Editor rendering pauses during replay profiling.");
        }
    }

    ImGui::SameLine();
    if (graphicsCapture.capturePending) {
        ImGui::TextDisabled("%s", graphicsCapture.gpuTrace ? "Capturing next full frame..."
            : "Collecting capture and replay GPU Trace; rendering will pause...");
    } else if (graphicsCapture.statusText != nullptr && graphicsCapture.statusText[0] != '\0') {
        ImGui::TextDisabled("%s", graphicsCapture.statusText);
    }

    if (graphicsCapture.replayTracePath != nullptr && graphicsCapture.replayTracePath[0] != '\0') {
        ImGui::TextWrapped("Replay GPU Trace: %s", graphicsCapture.replayTracePath);
        if (ImGui::SmallButton("Copy GPU Trace Path")) { ImGui::SetClipboardText(graphicsCapture.replayTracePath); }
    }
    if (graphicsCapture.replayTraceError != nullptr && graphicsCapture.replayTraceError[0] != '\0') {
        ImGui::TextWrapped("Replay collection failed (capture preserved): %s", graphicsCapture.replayTraceError);
    }

    if (graphicsCapture.capturePath != nullptr && graphicsCapture.capturePath[0] != '\0') {
        ImGui::TextWrapped("Last capture: %s", graphicsCapture.capturePath);
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy Path")) {
            ImGui::SetClipboardText(graphicsCapture.capturePath);
        }
    }
    ImGui::Separator();

    ImGui::Checkbox("Detailed", &detailed_);
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear history")) { clearHistory(); }
    const auto frame = [&] {
        auto profileScope = scope("Presentation Frame");
        return presentationFrame();
    }();
    ImGui::SameLine();
    ImGui::TextDisabled("%zu / %zu frames", history_.size(), kProfilerHistorySize);
    const bool hasGpu = std::any_of(frame.nodes.begin(), frame.nodes.end(), [](const auto& node) { return node.gpuTimingAvailable; });
    if (hasGpu) { ImGui::TextDisabled("GPU completed: editor frame %llu (%llu frames behind)",
        static_cast<unsigned long long>(frame.index), static_cast<unsigned long long>(latestFrame_.index - frame.index)); }
    else { ImGui::TextDisabled("GPU queries pending / unavailable"); }
    ImGui::TextDisabled("GPU envelope covers RenderGraph; editor UI and presentation are outside this interval.");
    if (render::profiling::NsightGraphicsCapture::vulkanInjectionActive()) {
        ImGui::TextColored(ImVec4(1, .65f, .2f, 1),
            "Graphics Capture is active; live timings include capture overhead. Measure a separate baseline without injection.");
    }
    if (frame.profilingOverflow) { ImGui::TextColored(ImVec4(1, .65f, .2f, 1), "Profiler scope/display budget exceeded; some entries may be unavailable."); }
    if (ImGui::BeginTabBar("ProfilerTabs")) {
        if (ImGui::BeginTabItem("Table")) {
            auto profileScope = scope("Profiler Table");
            drawProfilerTable(frame, *this, detailed_);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("BarChart")) {
            drawTimingCharts(frame, *this, chartMetric_, chartPath_, false);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LineChart")) {
            drawTimingCharts(frame, *this, chartMetric_, chartPath_, true);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("PieChart")) {
            ImGui::TextDisabled("CPU frame sections only; GPU intervals can overlap.");
            drawPieChart(displayFrame());
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Streaming")) {
            ImGui::BeginChild("StreamingScroll", ImVec2(0, 0));
            drawStreaming(streamingHistory_, selectedStream_, streamingShowBudget_);
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
    return captureRequested;
}

size_t EditorProfiler::beginSection(std::string_view name, uint32_t color)
{
    const size_t parent = stack_.empty() ? 0 : stack_.back();
    const size_t nodeIndex = currentNodes_.size();
    currentNodes_.push_back(Node{
        .name = std::string(name),
        .color = color == 0 ? colorFromName(name) : color,
        .parent = parent,
        .beginTime = Clock::now(),
    });
    if (nodeIndex != parent && parent < currentNodes_.size()) {
        currentNodes_[parent].children.push_back(nodeIndex);
    }
    stack_.push_back(nodeIndex);
    return nodeIndex;
}

void EditorProfiler::endSection(size_t nodeIndex)
{
    if (!frameActive_ || nodeIndex >= currentNodes_.size()) {
        return;
    }

    const auto now = Clock::now();
    currentNodes_[nodeIndex].cpuMilliseconds =
        std::chrono::duration<double, std::milli>(now - currentNodes_[nodeIndex].beginTime).count();
    if (!stack_.empty() && stack_.back() == nodeIndex) {
        stack_.pop_back();
    }
}

size_t EditorProfiler::addFinishedSection(
    size_t parent,
    std::string name,
    uint32_t color,
    double cpuMilliseconds)
{
    const size_t nodeIndex = currentNodes_.size();
    currentNodes_.push_back(Node{
        .name = std::move(name),
        .color = color,
        .cpuMilliseconds = cpuMilliseconds,
        .parent = parent,
        .beginTime = Clock::now(),
    });
    if (parent < currentNodes_.size()) {
        currentNodes_[parent].children.push_back(nodeIndex);
    }
    return nodeIndex;
}

void EditorProfiler::endFrame()
{
    if (!frameActive_) {
        return;
    }

    while (!stack_.empty()) {
        endSection(stack_.back());
    }

    if (capturing_) {
        if (capturedFrames_.size() >= 60000) { captureOverflow_ = true; capturing_ = false; }
        else {
            const size_t index = capturedFrames_.size();
            capturedFrames_.push_back({currentNodes_, frameIndex_, currentOverflow_, currentStreaming_});
            for (const auto& node : currentNodes_) {
                if (node.renderGraphExecutionId != UINT64_MAX) {
                    capturedExecutions_[{graphGeneration_, node.renderGraphExecutionId}] = index;
                }
            }
        }
    }
    const auto hasGraph = [](const auto& nodes) {
        return std::any_of(nodes.begin(), nodes.end(), [](const auto& node) { return node.renderGraphExecutionId != UINT64_MAX; });
    };
    if (hasGraph(latestFrame_.nodes) && !hasGraph(currentNodes_)) { clearHistory(); }
    latestFrame_.index = frameIndex_++;
    latestFrame_.profilingOverflow = currentOverflow_;
    // Remove sources no longer present, including a switch to a non-streaming scene.
    std::erase_if(streamingHistory_, [&](const auto& source) {
        return std::none_of(currentStreaming_.begin(), currentStreaming_.end(), [&](const auto& sample) {
            return sample.passName == source.passName && sample.assetPath == source.assetPath && sample.generation == source.generation;
        });
    });
    for (auto& sample : currentStreaming_) {
        auto it = std::find_if(streamingHistory_.begin(), streamingHistory_.end(), [&](const auto& source) {
            return source.passName == sample.passName && source.assetPath == sample.assetPath && source.generation == sample.generation;
        });
        if (it == streamingHistory_.end()) {
            streamingHistory_.push_back({sample.passName, sample.assetPath, sample.generation, {}});
            it = std::prev(streamingHistory_.end());
        }
        auto& samples = it->samples;
        if (!samples.empty() && sample.frameIndex < samples.back().frameIndex) { samples.clear(); }
        if (!samples.empty() && sample.frameIndex == samples.back().frameIndex) { samples.back() = std::move(sample); }
        else { samples.push_back(std::move(sample)); }
        if (samples.size() > kProfilerHistorySize) { samples.erase(samples.begin()); }
    }
    latestFrame_.nodes = currentNodes_;
    registerFrameScopes(latestFrame_.nodes);
    addHistoryFrame(latestFrame_);
    history_.push_back(latestFrame_);
    if (history_.size() > kProfilerHistorySize) {
        removeHistoryFrame(history_.front());
        history_.erase(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(history_.size() - kProfilerHistorySize));
    }

    frameActive_ = false;
    METALLIC_TRACY_FRAME_MARK();
    currentNodes_.clear();
    stack_.clear();
}

uint32_t EditorProfiler::colorFromName(std::string_view name)
{
    uint32_t hash = 2166136261u;
    for (char c : name) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 16777619u;
    }

    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    ImGui::ColorConvertHSVtoRGB(static_cast<float>(hash % 360u) / 360.0f, 0.58f, 0.88f, r, g, b);
    return (static_cast<uint32_t>(r * 255.0f) << 24u) |
        (static_cast<uint32_t>(g * 255.0f) << 16u) |
        (static_cast<uint32_t>(b * 255.0f) << 8u) |
        0xffu;
}

} // namespace metallic
