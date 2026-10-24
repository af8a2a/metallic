#pragma once

#include "ImGuiTestSnapshot.h"
#include "Editor/EditorRenderGraphViewer.h"
#include "imgui_internal.h"

#include <cfloat>

namespace metallic::tests {

class ViewerUIContext {
public:
    explicit ViewerUIContext(bool exerciseMatrixSelection = false) : exerciseMatrixSelection_(exerciseMatrixSelection), previous_(ImGui::GetCurrentContext()), context_(ImGui::CreateContext())
    {
        auto& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(kWidth, kHeight);
        io.DeltaTime = 1.0f / 60;
        io.Fonts->AddFontDefault();
        io.Fonts->GetTexDataAsRGBA32(&atlas_, &atlasWidth_, &atlasHeight_);
        io.Fonts->SetTexID(ImTextureID(1));
    }
    ~ViewerUIContext()
    {
        ImGui::DestroyContext(context_);
        ImGui::SetCurrentContext(previous_);
    }
    std::string save(editor::RenderGraphExecutionViewer& viewer, editor::RenderGraphExecutionViewer::Tab tab,
        const std::filesystem::path& path)
    {
        viewer.setTab(tab);
        for (uint32_t frame = 0; frame < 4; ++frame) { draw(viewer); }
        if (exerciseMatrixSelection_ && tab == editor::RenderGraphExecutionViewer::Tab::Resources && viewer.snapshot()) {
            const auto& snapshot = *viewer.snapshot();
            ImGuiWindow* matrix = nullptr;
            for (auto* window : context_->Windows) {
                if (window->Active && std::string_view(window->Name).find("ResourcePassMatrix") != std::string_view::npos) {
                    matrix = window; break;
                }
            }
            if (!matrix || snapshot.resources.empty()) { return "resource matrix canvas is missing"; }
            const uint64_t resource = snapshot.resources.back().id;
            size_t passIndex = 0;
            for (; passIndex < snapshot.passes.size(); ++passIndex) {
                const auto& uses = snapshot.passes[passIndex].uses;
                if (std::any_of(uses.begin(), uses.end(), [&](const auto& use) { return use.resourceId == resource; })) { break; }
            }
            if (passIndex == snapshot.passes.size()) { return "captured fixture resource had no matrix use"; }
            // Queue-colored pass strips locate the transformed matrix, independently
            // of its fit scale and translation. Select the final captured resource.
            ImRect strips(ImVec2(FLT_MAX, FLT_MAX), ImVec2(-FLT_MAX, -FLT_MAX));
            for (const auto& vertex : matrix->DrawList->VtxBuffer) {
                if (vertex.col == IM_COL32(110, 170, 235, 255) || vertex.col == IM_COL32(177, 137, 233, 255) ||
                    vertex.col == IM_COL32(90, 192, 160, 255)) { strips.Add(vertex.pos); }
            }
            if (strips.Min.x >= strips.Max.x || strips.Min.y >= strips.Max.y) { return "matrix pass header geometry is missing"; }
            const float zoom = strips.GetHeight() / 5.0f;
            const ImVec2 point(strips.Min.x + (float(passIndex) * 64 + 32) * zoom,
                strips.Min.y + (12 + float(snapshot.resources.size() - 1) * 25 + 10) * zoom);
            auto& io = ImGui::GetIO();
            io.AddMousePosEvent(point.x, point.y); draw(viewer);
            io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); draw(viewer);
            io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); draw(viewer);
            io.AddMousePosEvent(-100, -100); draw(viewer);
            if (viewer.selectedResourceId() != resource || viewer.selectedPassId() != snapshot.passes[passIndex].id) {
                return "clicking an actual resource/pass cell did not select both inspector identities";
            }
        }
        const auto* data = ImGui::GetDrawData();
        if (!data || data->TotalVtxCount < 100 || data->TotalIdxCount < 150) {
            return "execution viewer emitted no substantial UI geometry";
        }
        for (const auto* list : data->CmdLists) {
            for (const auto& vertex : list->VtxBuffer) {
                if (!std::isfinite(vertex.pos.x) || !std::isfinite(vertex.pos.y) ||
                    !std::isfinite(vertex.uv.x) || !std::isfinite(vertex.uv.y)) {
                    return "execution viewer emitted non-finite UI geometry";
                }
            }
            for (const auto& command : list->CmdBuffer) {
                if (command.UserCallback || command.GetTexID() != ImTextureID(1)) {
                    return "viewer screenshot encountered an unsupported external texture or callback";
                }
                const auto clip = command.ClipRect;
                if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) || !std::isfinite(clip.w) ||
                    clip.x > clip.z || clip.y > clip.w) {
                    return "execution viewer emitted an invalid clipping rectangle";
                }
                if (uint64_t(command.IdxOffset) + command.ElemCount > uint64_t(list->IdxBuffer.Size)) {
                    return "execution viewer emitted invalid draw indices";
                }
            }
        }
        std::string message;
        if (!saveImGuiTestDrawDataPng(*data, atlas_, atlasWidth_, atlasHeight_, kWidth, kHeight, path, message)) {
            return message;
        }
        return {};
    }
private:
    void draw(editor::RenderGraphExecutionViewer& viewer)
    {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(kWidth, kHeight));
        ImGui::Begin("Render Graph Execution", nullptr,
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize);
        viewer.draw();
        ImGui::End();
        ImGui::Render();
    }
    static constexpr int kWidth = 1400, kHeight = 900;
    bool exerciseMatrixSelection_ = false;
    ImGuiContext* previous_ = nullptr;
    ImGuiContext* context_ = nullptr;
    unsigned char* atlas_ = nullptr;
    int atlasWidth_ = 0, atlasHeight_ = 0;
};

} // namespace metallic::tests
