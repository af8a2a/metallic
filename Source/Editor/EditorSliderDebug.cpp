#include "Editor/EditorApplication.h"

#include <imgui.h>
#include <ImGuizmo.h>

#include <algorithm>
#include <cmath>

namespace metallic {
namespace {

render::RenderGraphProperties sliderProperties(const render::RenderGraphNode& node)
{
    auto properties = node.properties;
    properties.merge_patch(node.runtimeProperties);
    return properties;
}

} // namespace

render::RenderGraphNode* EditorApplication::viewportSliderDebugNode()
{
    auto* output = activePreviewRenderGraphNode();
    if (output == nullptr) { return nullptr; }
    std::vector<std::string> pending{output->name};
    render::RenderGraphNode* inactiveNr = nullptr;
    for (size_t i = 0; i < pending.size(); ++i) {
        const std::string name = pending[i];
        auto* node = renderGraph_.findNode(name);
        if (node != nullptr && node->type == "SliderDebugPass") { return node; }
        if (node != nullptr && node->type == "DlssNrPass") {
            if (sliderProperties(*node).value("sliderDebug", false)) { return node; }
            if (inactiveNr == nullptr) { inactiveNr = node; }
        }
        for (const auto& edge : renderGraph_.edges()) {
            if (edge.dstPass == name && std::find(pending.begin(), pending.end(), edge.srcPass) == pending.end()) {
                pending.push_back(edge.srcPass);
            }
        }
    }
    // Keep the NR checkbox accessible while its divider is disabled.
    return inactiveNr;
}

void EditorApplication::setSliderDebugProperty(
    uint32_t nodeId, const char* key, render::RenderGraphProperties value)
{
    if (renderGraph_.setNodeRuntimeProperty(nodeId, key, std::move(value))) {
        // Comparison changes do not invalidate either producer's accumulation.
        if (graphExecutor_ != nullptr && !renderGraph_.dirty()) {
            graphExecutor_->syncRuntimeProperties(renderGraph_);
        }
        viewportPreviewNeedsRender_ = true;
    }
}

void EditorApplication::drawSliderDebugControls()
{
    auto* node = viewportSliderDebugNode();
    if (node == nullptr) { return; }
    const auto properties = sliderProperties(*node);
    float split = properties.value("splitPosition", 0.5f);
    split = std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f;
    bool horizontal = properties.value("orientation", "vertical") == "horizontal";
    bool swap = properties.value("swapSides", false);
    ImGui::PushID("SliderDebugControls");
    if (node->type == "DlssNrPass") {
        bool enabled = properties.value("sliderDebug", false);
        if (ImGui::Checkbox("DLSS-NR Slider Debug", &enabled)) {
            setSliderDebugProperty(node->id, "sliderDebug", enabled);
        }
        if (!enabled) { ImGui::PopID(); return; }
        ImGui::SameLine();
    }
    ImGui::TextUnformatted("Compare");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140.0f * mainScale_);
    if (ImGui::SliderFloat("##Position", &split, 0.0f, 1.0f, "%.2f")) {
        setSliderDebugProperty(node->id, "splitPosition", split);
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Top / bottom", &horizontal)) {
        setSliderDebugProperty(node->id, "orientation", horizontal ? "horizontal" : "vertical");
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Swap A/B", &swap)) {
        setSliderDebugProperty(node->id, "swapSides", swap);
    }
    ImGui::PopID();
}

bool EditorApplication::drawSliderDebugOverlay(const ImVec2& min, const ImVec2& max)
{
    auto* node = viewportSliderDebugNode();
    if (node == nullptr || !viewportInteractionEnabled_) {
        sliderDragNodeId_ = 0;
        return false;
    }
    const auto properties = sliderProperties(*node);
    if (node->type == "DlssNrPass" && !properties.value("sliderDebug", false)) {
        sliderDragNodeId_ = 0;
        return false;
    }
    float split = properties.value("splitPosition", 0.5f);
    split = std::isfinite(split) ? std::clamp(split, 0.0f, 1.0f) : 0.5f;
    const bool horizontal = properties.value("orientation", "vertical") == "horizontal";
    const bool swap = properties.value("swapSides", false);
    const auto& io = ImGui::GetIO();
    const float start = horizontal ? min.y : min.x;
    const float extent = horizontal ? max.y - min.y : max.x - min.x;
    if (extent <= 0.0f) { sliderDragNodeId_ = 0; return false; }
    const float mouse = horizontal ? io.MousePos.y : io.MousePos.x;
    const bool cameraGesture = io.KeyAlt || ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
        ImGui::IsMouseDown(ImGuiMouseButton_Middle) || viewportCameraDragButton_ != -1;
    const bool canGrab = !cameraGesture && !io.KeyCtrl && !ImGuizmo::IsUsingAny() && !gizmoWasUsing_;
    const bool hovered = canGrab && viewportHovered_ &&
        std::abs(mouse - (start + split * extent)) <= 8.0f * mainScale_;
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { sliderDragNodeId_ = node->id; }
    const bool wasDragging = sliderDragNodeId_ == node->id;
    if (!canGrab || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) { sliderDragNodeId_ = 0; }
    const bool dragging = sliderDragNodeId_ == node->id;
    if (dragging) {
        split = std::clamp((mouse - start) / extent, 0.0f, 1.0f);
        setSliderDebugProperty(node->id, "splitPosition", split);
    }
    if (hovered || dragging) {
        ImGui::SetMouseCursor(horizontal ? ImGuiMouseCursor_ResizeNS : ImGuiMouseCursor_ResizeEW);
    }

    std::string labelA = "A", labelB = "B";
    if (node->type == "DlssNrPass") {
        labelA = "A: Before DLSS-NR";
        labelB = "B: After DLSS-NR";
    }
    for (const auto& edge : renderGraph_.edges()) {
        if (edge.dstPass != node->name) { continue; }
        if (edge.dstField == "sourceA") { labelA += ": " + edge.srcPass; }
        if (edge.dstField == "sourceB") { labelB += ": " + edge.srcPass; }
    }
    if (swap) { std::swap(labelA, labelB); }
    const float position = start + split * extent;
    const ImVec2 p0 = horizontal ? ImVec2(min.x, position) : ImVec2(position, min.y);
    const ImVec2 p1 = horizontal ? ImVec2(max.x, position) : ImVec2(position, max.y);
    auto* draw = ImGui::GetWindowDrawList();
    draw->PushClipRect(min, max, true);
    draw->AddLine(p0, p1, IM_COL32(0, 0, 0, 220), 3.0f * mainScale_);
    draw->AddLine(p0, p1, IM_COL32(235, 238, 245, 255), mainScale_);
    const ImVec2 handle = horizontal ? ImVec2((min.x + max.x) * 0.5f, position)
        : ImVec2(position, (min.y + max.y) * 0.5f);
    draw->AddCircleFilled(handle, 7.0f * mainScale_, IM_COL32(30, 35, 44, 255));
    draw->AddCircle(handle, 7.0f * mainScale_, IM_COL32(235, 238, 245, 255));
    const auto drawLabel = [&](const std::string& label, bool second) {
        const ImVec2 size = ImGui::CalcTextSize(label.c_str());
        const ImVec2 origin(
            second && !horizontal ? max.x - size.x - 12.0f * mainScale_ : min.x + 12.0f * mainScale_,
            second && horizontal ? max.y - size.y - 12.0f * mainScale_ : min.y + 12.0f * mainScale_);
        draw->AddRectFilled(ImVec2(origin.x - 4, origin.y - 2),
            ImVec2(origin.x + size.x + 4, origin.y + size.y + 2), IM_COL32(10, 12, 16, 200), 3.0f);
        draw->AddText(origin, IM_COL32(235, 238, 245, 255), label.c_str());
    };
    drawLabel(labelA, false);
    drawLabel(labelB, true);
    draw->PopClipRect();
    return hovered || wasDragging || dragging;
}

} // namespace metallic
