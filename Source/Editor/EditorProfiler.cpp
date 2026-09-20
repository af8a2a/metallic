#include "Editor/EditorProfiler.h"
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

struct Aggregate {
    double average = 0.0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = 0.0;
    size_t count = 0;
};

Aggregate aggregateByPath(const std::vector<EditorProfiler::Frame>& history, const std::vector<std::string>& path, bool gpu)
{
    Aggregate result;
    for (const auto& frame : history) {
        const double value = sampleValue(nodeByPath(frame, path), gpu);
        if (!std::isfinite(value)) { continue; }
        result.average += value;
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
        ++result.count;
    }
    if (result.count) { result.average /= double(result.count); }
    return result;
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
    Timer = 1, GpuAverage, CpuAverage, Queue, GpuLast, GpuMinimum, GpuMaximum, CpuLast, CpuMinimum, CpuMaximum
};

struct ProfilerTableRow {
    Aggregate gpu;
    Aggregate cpu;
    bool ready = false;
};

const ProfilerTableRow& tableRow(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history,
    size_t index, std::vector<ProfilerTableRow>& rows)
{
    auto& row = rows[index];
    if (!row.ready) {
        std::vector<std::string> path;
        for (size_t current = index; current != 0; current = frame.nodes[current].parent) { path.push_back(frame.nodes[current].name); }
        std::reverse(path.begin(), path.end());
        row.gpu = aggregateByPath(history, path, true);
        row.cpu = aggregateByPath(history, path, false);
        row.ready = true;
    }
    return row;
}

const char* tableQueueName(const EditorProfiler::Node& node)
{
    if (node.cpuOnly || node.renderGraphExecutionId == UINT64_MAX) { return "CPU"; }
    return node.renderGraphNodeId == UINT32_MAX ? "Envelope" : queueName(node.queue);
}

double tableSortValue(const EditorProfiler::Node& node, const ProfilerTableRow& row, ProfilerColumn column)
{
    const double missing = std::numeric_limits<double>::quiet_NaN();
    switch (column) {
    case ProfilerColumn::GpuAverage: return row.gpu.count ? row.gpu.average : missing;
    case ProfilerColumn::CpuAverage: return row.cpu.count ? row.cpu.average : missing;
    case ProfilerColumn::GpuLast: return node.gpuTimingAvailable ? node.gpuMilliseconds : missing;
    case ProfilerColumn::GpuMinimum: return row.gpu.count ? row.gpu.minimum : missing;
    case ProfilerColumn::GpuMaximum: return row.gpu.count ? row.gpu.maximum : missing;
    case ProfilerColumn::CpuLast: return node.cpuMilliseconds;
    case ProfilerColumn::CpuMinimum: return row.cpu.count ? row.cpu.minimum : missing;
    case ProfilerColumn::CpuMaximum: return row.cpu.count ? row.cpu.maximum : missing;
    default: return missing;
    }
}

void drawProfilerTableNode(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history,
    size_t index, uint32_t depth, bool detailed, std::vector<ProfilerTableRow>& rows,
    const ImGuiTableColumnSortSpecs* sort)
{
    const auto& node = frame.nodes[index];
    const bool children = !node.children.empty();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth;
    if (!children) { flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen; }
    else if (depth < 4) { flags |= ImGuiTreeNodeFlags_DefaultOpen; }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushStyleColor(ImGuiCol_Text, imguiColor(node.color));
    const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(index + 1), flags, "%s", node.name.c_str());
    ImGui::PopStyleColor();
    const auto& row = tableRow(frame, history, index, rows);
    const auto& gpu = row.gpu;
    const auto& cpu = row.cpu;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("CPU samples: %zu | GPU samples: %zu\nGPU results arrive after completion; missing queries are excluded.\nNested / concurrent intervals must not be added together.", cpu.count, gpu.count);
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
            if (column != ProfilerColumn::Timer && column != ProfilerColumn::Queue) {
                for (size_t child : children) { tableRow(frame, history, child, rows); }
            }
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
            drawProfilerTableNode(frame, history, child, depth + 1, detailed, rows, sort);
        }
        ImGui::TreePop();
    }
}

void drawProfilerTable(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history, bool detailed)
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
    setupColumn("GPU avg ms", ProfilerColumn::GpuAverage, 90);
    setupColumn("CPU avg ms", ProfilerColumn::CpuAverage, 90);
    setupColumn("Queue", ProfilerColumn::Queue, 90, false);
    if (detailed) {
        setupColumn("GPU last", ProfilerColumn::GpuLast, 85);
        setupColumn("GPU min", ProfilerColumn::GpuMinimum, 85);
        setupColumn("GPU max", ProfilerColumn::GpuMaximum, 85);
        setupColumn("CPU last", ProfilerColumn::CpuLast, 85);
        setupColumn("CPU min", ProfilerColumn::CpuMinimum, 85);
        setupColumn("CPU max", ProfilerColumn::CpuMaximum, 85);
    }
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableHeadersRow();
    auto* specs = ImGui::TableGetSortSpecs();
    const auto* sort = specs && specs->SpecsCount > 0 ? &specs->Specs[0] : nullptr;
    // Recompute on every displayed sample, including delayed GPU backfills, not
    // just SpecsDirty. Each row's history aggregate is computed once per draw.
    std::vector<ProfilerTableRow> rows(frame.nodes.size());
    drawProfilerTableNode(frame, history, 0, 0, detailed, rows, sort);
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

void drawTimingCharts(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history,
    int& metric, std::vector<std::string>& path, bool lines)
{
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
        for (const auto& [p, node] : paths) { max = std::max(max, aggregateByPath(history, p, gpu).average); }
        for (const auto& [p, node] : paths) {
            const auto avg = aggregateByPath(history, p, gpu);
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
    if (ImGui::CollapsingHeader("CPU Request / Reclaim Work")) {
        const auto& work = last.cpuWork;
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

void applyRenderGraphGpuStats(
    std::vector<EditorProfiler::Node>& nodes,
    const render::RenderGraphExecutionStats& stats)
{
    for (EditorProfiler::Node& node : nodes) {
        if (node.renderGraphExecutionId != stats.executionId) {
            continue;
        }
        if (node.renderGraphNodeId == UINT32_MAX) {
            node.gpuMilliseconds = stats.gpuMilliseconds;
            node.gpuTimingAvailable = stats.gpuTimingAvailable;
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
        }
    }
}

} // namespace

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

void EditorProfiler::addRenderGraphStats(const render::RenderGraphExecutionStats& stats)
{
    if (!frameActive_ || stats.nodes.empty()) {
        return;
    }

    if (graphGeneration_ != stats.graphGeneration) {
        history_.clear();
        latestFrame_ = {};
        graphGeneration_ = stats.graphGeneration;
    }
    currentStreaming_.insert(currentStreaming_.end(), stats.streaming.begin(), stats.streaming.end());
    currentOverflow_ |= stats.profilingOverflow;
    const size_t parent = stack_.empty() ? 0 : stack_.back();
    for (const auto& section : stats.preparation) {
        const auto index = addFinishedSection(parent, "Graph preparation / " + section.name,
            colorFromName(section.name), section.cpuMilliseconds);
        currentNodes_[index].cpuOnly = true;
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
    for (Frame& frame : history_) {
        applyRenderGraphGpuStats(frame.nodes, stats);
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
    if (ImGui::Button("Export Current View Capture")) {
        captureRequested = true;
    }
    ImGui::EndDisabled();

    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        if (!graphicsCapture.sdkCompiled) {
            ImGui::SetTooltip("Nsight Graphics SDK was not available when Metallic was built.");
        } else if (!graphicsCapture.runtimeEnabled) {
            ImGui::SetTooltip("Restart Metallic with --nsight-capture to enable startup-time injection.");
        } else if (graphicsCapture.capturePending) {
            ImGui::SetTooltip("A Graphics Capture is already being written.");
        } else if (!graphicsCapture.canCapture) {
            ImGui::SetTooltip(
                "%s",
                graphicsCapture.statusText != nullptr && graphicsCapture.statusText[0] != '\0'
                    ? graphicsCapture.statusText
                    : "The current View is not ready for capture.");
        } else {
            ImGui::SetTooltip(
                "Capture the current View during the next complete presented frame. "
                "Optimized Slang source symbols are embedded in captured SPIR-V.");
        }
    }

    ImGui::SameLine();
    if (graphicsCapture.capturePending) {
        ImGui::TextDisabled("Capturing next full frame...");
    } else if (graphicsCapture.statusText != nullptr && graphicsCapture.statusText[0] != '\0') {
        ImGui::TextDisabled("%s", graphicsCapture.statusText);
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
    if (ImGui::SmallButton("Clear history")) { history_.clear(); for (auto& source : streamingHistory_) { source.samples.clear(); } }
    const auto& frame = displayFrame();
    ImGui::SameLine();
    ImGui::TextDisabled("%zu / %zu frames", history_.size(), kProfilerHistorySize);
    const bool hasGpu = std::any_of(frame.nodes.begin(), frame.nodes.end(), [](const auto& node) { return node.gpuTimingAvailable; });
    if (hasGpu) { ImGui::TextDisabled("GPU completed: editor frame %llu (%llu frames behind)",
        static_cast<unsigned long long>(frame.index), static_cast<unsigned long long>(latestFrame_.index - frame.index)); }
    else { ImGui::TextDisabled("GPU queries pending / unavailable"); }
    ImGui::TextDisabled("GPU envelope covers RenderGraph; editor UI and presentation are outside this interval.");
    if (frame.profilingOverflow) { ImGui::TextColored(ImVec4(1, .65f, .2f, 1), "Profiler scope budget exceeded; some GPU intervals unavailable."); }
    if (ImGui::BeginTabBar("ProfilerTabs")) {
        if (ImGui::BeginTabItem("Table")) {
            drawProfilerTable(frame, history_, detailed_);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("BarChart")) {
            drawTimingCharts(frame, history_, chartMetric_, chartPath_, false);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LineChart")) {
            drawTimingCharts(frame, history_, chartMetric_, chartPath_, true);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("PieChart")) {
            ImGui::TextDisabled("CPU frame sections only; GPU intervals can overlap.");
            drawPieChart(frame);
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
    history_.push_back(latestFrame_);
    if (history_.size() > kProfilerHistorySize) {
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
