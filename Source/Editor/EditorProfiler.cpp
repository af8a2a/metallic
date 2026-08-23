#include "Editor/EditorProfiler.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <utility>

namespace metallic {
namespace {

constexpr size_t kProfilerHistorySize = 240;
constexpr float kPi = 3.14159265358979323846f;

ImU32 imguiColor(uint32_t rgba)
{
    const int r = static_cast<int>((rgba >> 24u) & 0xffu);
    const int g = static_cast<int>((rgba >> 16u) & 0xffu);
    const int b = static_cast<int>((rgba >> 8u) & 0xffu);
    const int a = static_cast<int>(rgba & 0xffu);
    return IM_COL32(r, g, b, a);
}

double nodeValueByPath(const EditorProfiler::Frame& frame, const std::vector<std::string>& path)
{
    if (frame.nodes.empty()) {
        return 0.0;
    }
    size_t nodeIndex = 0;
    for (const std::string& part : path) {
        const auto& children = frame.nodes[nodeIndex].children;
        const auto iter = std::find_if(
            children.begin(),
            children.end(),
            [&](size_t childIndex) {
                return frame.nodes[childIndex].name == part;
            });
        if (iter == children.end()) {
            return 0.0;
        }
        nodeIndex = *iter;
    }
    return frame.nodes[nodeIndex].cpuMilliseconds;
}

struct Aggregate {
    double average = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    size_t count = 0;
};

Aggregate aggregateByPath(const std::vector<EditorProfiler::Frame>& history, const std::vector<std::string>& path)
{
    Aggregate aggregate;
    double total = 0.0;
    double minimum = std::numeric_limits<double>::max();
    double maximum = 0.0;
    for (const EditorProfiler::Frame& frame : history) {
        const double value = nodeValueByPath(frame, path);
        if (value <= 0.0) {
            continue;
        }
        total += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        ++aggregate.count;
    }

    if (aggregate.count > 0) {
        aggregate.average = total / static_cast<double>(aggregate.count);
        aggregate.minimum = minimum;
        aggregate.maximum = maximum;
    }
    return aggregate;
}

void drawDuration(double milliseconds)
{
    if (milliseconds <= 0.0) {
        ImGui::TextDisabled("--");
        return;
    }
    ImGui::Text("%.3f", milliseconds);
}

void drawProfilerTableNode(
    const EditorProfiler::Frame& frame,
    const std::vector<EditorProfiler::Frame>& history,
    size_t nodeIndex,
    std::vector<std::string>& path,
    uint32_t depth)
{
    const EditorProfiler::Node& node = frame.nodes[nodeIndex];
    const bool hasChildren = !node.children.empty();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth;
    if (!hasChildren) {
        flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    } else if (depth < 2) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushStyleColor(ImGuiCol_Text, imguiColor(node.color));
    const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(nodeIndex), flags, "%s", node.name.c_str());
    ImGui::PopStyleColor();

    const Aggregate aggregate = aggregateByPath(history, path);
    ImGui::TableNextColumn();
    drawDuration(node.cpuMilliseconds);
    ImGui::TableNextColumn();
    if (node.gpuTimingAvailable) {
        ImGui::Text("%.3f", node.gpuMilliseconds);
    } else {
        ImGui::TextDisabled("--");
    }
    ImGui::TableNextColumn();
    drawDuration(aggregate.average);
    ImGui::TableNextColumn();
    drawDuration(aggregate.minimum);
    ImGui::TableNextColumn();
    drawDuration(aggregate.maximum);

    if (open && hasChildren) {
        for (size_t childIndex : node.children) {
            path.push_back(frame.nodes[childIndex].name);
            drawProfilerTableNode(frame, history, childIndex, path, depth + 1);
            path.pop_back();
        }
        ImGui::TreePop();
    }
}

void drawProfilerTable(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history)
{
    if (frame.nodes.empty()) {
        ImGui::TextDisabled("No profiler samples yet.");
        return;
    }

    if (!ImGui::BeginTable(
            "ProfilerTable",
            6,
            ImGuiTableFlags_Borders |
                ImGuiTableFlags_RowBg |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_ScrollY,
            ImVec2(0.0f, 0.0f))) {
        return;
    }

    ImGui::TableSetupColumn("Timer", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Last CPU ms", ImGuiTableColumnFlags_WidthFixed, 92.0f);
    ImGui::TableSetupColumn("Last GPU ms", ImGuiTableColumnFlags_WidthFixed, 92.0f);
    ImGui::TableSetupColumn("Avg", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableHeadersRow();

    std::vector<std::string> path;
    drawProfilerTableNode(frame, history, 0, path, 0);
    ImGui::EndTable();
}

void drawBarChart(const EditorProfiler::Frame& frame)
{
    if (frame.nodes.empty() || frame.nodes[0].children.empty()) {
        ImGui::TextDisabled("No profiler samples yet.");
        return;
    }

    const EditorProfiler::Node& root = frame.nodes[0];
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 34.0f;
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, ImVec2(pos.x + width, pos.y + height), IM_COL32(18, 18, 18, 255));
    drawList->AddRect(pos, ImVec2(pos.x + width, pos.y + height), IM_COL32(100, 100, 100, 160));

    const double total = std::max(root.cpuMilliseconds, 0.001);
    float cursorX = pos.x;
    for (size_t childIndex : root.children) {
        const EditorProfiler::Node& child = frame.nodes[childIndex];
        const float fraction = static_cast<float>(std::max(child.cpuMilliseconds, 0.0) / total);
        const float segmentWidth = std::max(width * fraction, child.cpuMilliseconds > 0.0 ? 1.0f : 0.0f);
        const ImVec2 min(cursorX, pos.y);
        const ImVec2 max(std::min(cursorX + segmentWidth, pos.x + width), pos.y + height);
        drawList->AddRectFilled(min, max, imguiColor(child.color));
        if (segmentWidth > 58.0f) {
            drawList->AddText(ImVec2(min.x + 5.0f, min.y + 9.0f), IM_COL32_WHITE, child.name.c_str());
        }
        cursorX += segmentWidth;
    }
    ImGui::Dummy(ImVec2(width, height + 8.0f));

    if (ImGui::BeginTable("ProfilerBarLegend", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
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

void drawLineChart(const EditorProfiler::Frame& frame, const std::vector<EditorProfiler::Frame>& history)
{
    if (frame.nodes.empty() || history.empty()) {
        ImGui::TextDisabled("No profiler samples yet.");
        return;
    }

    const float width = ImGui::GetContentRegionAvail().x;
    const float height = std::max(180.0f, ImGui::GetContentRegionAvail().y - 48.0f);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImVec2 size(width, height);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(18, 18, 18, 255));
    drawList->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(100, 100, 100, 160));

    double maxValue = 0.0;
    for (const EditorProfiler::Frame& sample : history) {
        maxValue = std::max(maxValue, sample.nodes.empty() ? 0.0 : sample.nodes[0].cpuMilliseconds);
    }
    maxValue = std::max(maxValue, 0.001);

    auto plotPath = [&](const std::vector<std::string>& path, uint32_t color) {
        if (history.size() < 2) {
            return;
        }
        ImVec2 previous;
        bool hasPrevious = false;
        for (size_t index = 0; index < history.size(); ++index) {
            const double value = path.empty()
                ? (history[index].nodes.empty() ? 0.0 : history[index].nodes[0].cpuMilliseconds)
                : nodeValueByPath(history[index], path);
            const float x = pos.x + (static_cast<float>(index) / static_cast<float>(history.size() - 1)) * size.x;
            const float y = pos.y + size.y - static_cast<float>(std::clamp(value / maxValue, 0.0, 1.0)) * size.y;
            const ImVec2 point(x, y);
            if (hasPrevious) {
                drawList->AddLine(previous, point, imguiColor(color), 2.0f);
            }
            previous = point;
            hasPrevious = true;
        }
    };

    plotPath({}, 0xffffffffu);
    const EditorProfiler::Node& root = frame.nodes[0];
    for (size_t childIndex : root.children) {
        plotPath({frame.nodes[childIndex].name}, frame.nodes[childIndex].color);
    }

    char label[64] = {};
    std::snprintf(label, sizeof(label), "%.2f ms", maxValue);
    drawList->AddText(ImVec2(pos.x + 6.0f, pos.y + 5.0f), IM_COL32(220, 220, 220, 255), label);
    ImGui::Dummy(size);

    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(IM_COL32_WHITE), "Frame");
    for (size_t childIndex : root.children) {
        ImGui::SameLine();
        const EditorProfiler::Node& child = frame.nodes[childIndex];
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(imguiColor(child.color)), "%s", child.name.c_str());
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
            node.gpuMilliseconds = iter->gpuMilliseconds;
            node.gpuTimingAvailable = iter->gpuTimingAvailable;
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

EditorProfiler::FrameScope EditorProfiler::beginFrame()
{
    if (frameActive_) {
        endFrame();
    }

    currentNodes_.clear();
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

    const size_t parent = stack_.empty() ? 0 : stack_.back();
    const size_t group = addFinishedSection(
        parent,
        "RenderGraph Passes",
        colorFromName("RenderGraph Passes"),
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
    }
}

void EditorProfiler::updateRenderGraphGpuStats(const render::RenderGraphExecutionStats& stats)
{
    if (!stats.gpuTimingAvailable) {
        return;
    }

    applyRenderGraphGpuStats(currentNodes_, stats);
    applyRenderGraphGpuStats(latestFrame_.nodes, stats);
    for (Frame& frame : history_) {
        applyRenderGraphGpuStats(frame.nodes, stats);
    }
}

void EditorProfiler::drawWindow(bool* open)
{
    if (open != nullptr && !*open) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(520.0f, 360.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Profiler", open)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("ProfilerTabs")) {
        if (ImGui::BeginTabItem("Table")) {
            drawProfilerTable(latestFrame_, history_);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("BarChart")) {
            drawBarChart(latestFrame_);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("LineChart")) {
            drawLineChart(latestFrame_, history_);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("PieChart")) {
            drawPieChart(latestFrame_);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
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

    latestFrame_.nodes = currentNodes_;
    history_.push_back(latestFrame_);
    if (history_.size() > kProfilerHistorySize) {
        history_.erase(history_.begin(), history_.begin() + static_cast<std::ptrdiff_t>(history_.size() - kProfilerHistorySize));
    }

    frameActive_ = false;
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
