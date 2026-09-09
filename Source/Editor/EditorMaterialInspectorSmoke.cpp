#include "Editor/EditorApplication.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace metallic {

bool EditorApplication::runMaterialInspectorSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) {
            spdlog::error("[Smoke Material Inspector] {}", message);
        }
        return condition;
    };
    std::error_code pathError;
    const bool isolatedScene = std::filesystem::equivalent(
        scene_.sourcePath().parent_path(), std::filesystem::current_path() / "scene", pathError);
    if (!expect(isolatedScene && !pathError, "Use the CTest runner's isolated scene directory for save testing")) {
        return false;
    }
    if (!expect(scene_.valid() && scene_.materials().size() >= 2, "Scene contains both editable fixture materials") ||
        !renderFrame() || !expect(viewportPreviewValid_, "Comparison viewport renders before editing")) {
        return false;
    }

    constexpr int32_t kMaterialIndex = 0;
    const auto resolvesMaterial = [&] {
        const auto indices = selectedMaterialIndices();
        return std::find(indices.begin(), indices.end(), kMaterialIndex) != indices.end();
    };
    sceneSelection_ = SceneSelection{
        .type = SceneSelectionType::Node,
        .object = scene_.objectForNode(0).entity(),
        .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
        .index = 0,
        .nodeIndex = 0,
    };
    if (!expect(resolvesMaterial(), "Selecting the shaderball node exposes its material")) {
        return false;
    }
    sceneSelection_ = SceneSelection{.type = SceneSelectionType::Mesh, .index = 0, .meshIndex = 0};
    if (!expect(resolvesMaterial(), "Mesh selection exposes its material")) {
        return false;
    }
    sceneSelection_ = SceneSelection{.type = SceneSelectionType::RenderPrimitive, .index = 0, .primitiveIndex = 0};
    if (!expect(resolvesMaterial(), "Primitive selection exposes its material")) {
        return false;
    }
    sceneSelection_ = SceneSelection{
        .type = SceneSelectionType::Material,
        .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
        .index = kMaterialIndex,
    };
    if (!expect(resolvesMaterial(), "Direct material selection exposes its controls")) {
        return false;
    }
    resetTransformHistory();
    const auto before = scene_.materials()[kMaterialIndex];
    const uint64_t historyRevision = historyResources_.invalidationRevision();
    const uint64_t geometryRevision = scene_.geometryTransformRevision();
    const uint64_t resourceIdentity = scene_.resourceIdentity();
    const uint64_t sceneLifetime = scene_.sceneGraph().lifetimeRevision();
    ImRect roughnessRect;
    bool locatedRoughness = false;

    // Use ImGui's existing navigation geometry to find the real control. This
    // avoids depending on font size, the user's saved docking layout, or a
    // production-only test hook. Mouse events stay inside this ImGui context.
    const auto inspectorFrame = [&](ImVec2 mouse, bool mouseDown, int32_t locateMaterial = -1,
                                   bool bothMaterials = false, bool ctrl = false, const char* input = nullptr) {
        auto& io = ImGui::GetIO();
        io.AddMousePosEvent(mouse.x, mouse.y);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, mouseDown);
        io.AddKeyEvent(ImGuiMod_Ctrl, ctrl);
        if (input != nullptr) { io.AddInputCharactersUTF8(input); }
        io.DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame();
        const auto* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 20.0f, viewport->WorkPos.y + 20.0f));
        ImGui::SetNextWindowSize(ImVec2(700.0f, 650.0f));
        ImGui::Begin("Material inspector interaction test", nullptr,
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoMove);
        auto* window = ImGui::GetCurrentWindow();
        if (locateMaterial >= 0) {
            ImGui::PushID("MaterialInspector");
            ImGui::PushID(locateMaterial);
            const ImGuiID id = ImGui::GetID("Roughness");
            ImGui::PopID();
            ImGui::PopID();
            ImGui::FocusWindow(window);
            ImGui::SetNavID(id, ImGuiNavLayer_Main, ImGui::GetCurrentFocusScope(), ImRect());
        }
        drawSelectedMaterialInspector();
        if (bothMaterials) { drawMaterialInspector(1); }
        if (locateMaterial >= 0 && GImGui->NavIdIsAlive) {
            roughnessRect = ImGui::WindowRectRelToAbs(window, window->NavRectRel[ImGuiNavLayer_Main]);
            locatedRoughness = roughnessRect.GetWidth() > 0.0f && roughnessRect.GetHeight() > 0.0f;
        }
        ImGui::End();
        ImGui::Render();
        ImGui::UpdatePlatformWindows();
    };

    inspectorFrame(ImVec2(-100.0f, -100.0f), false);
    inspectorFrame(ImVec2(-100.0f, -100.0f), false, kMaterialIndex);
    if (!expect(locatedRoughness, "Material selection exposes the Roughness control")) {
        return false;
    }
    const ImVec2 start = roughnessRect.GetCenter();
    // A pair of drag frames verifies continuous input forms one undo command.
    inspectorFrame(start, false);
    inspectorFrame(start, true);
    inspectorFrame(ImVec2(start.x + 30.0f, start.y), true);
    inspectorFrame(ImVec2(start.x + 60.0f, start.y), true);
    inspectorFrame(ImVec2(start.x + 60.0f, start.y), false);
    const auto edited = scene_.materials()[kMaterialIndex];
    if (!expect(std::abs(edited.roughnessFactor - before.roughnessFactor) > 0.0001f,
            "Dragging Roughness changes the authored material") ||
        !expect(!inspectorPropertyEditing_ && transformCommands_.size() == 1 && transformCommandCursor_ == 1,
            "One completed drag creates one undo command") ||
        !expect(scene_.dirty(), "A material edit marks the scene dirty") ||
        !expect(historyResources_.invalidationRevision() > historyRevision,
            "Material edits invalidate temporal accumulation") ||
        !expect(scene_.resourceIdentity() == resourceIdentity &&
                scene_.geometryTransformRevision() == geometryRevision &&
                scene_.sceneGraph().lifetimeRevision() == sceneLifetime,
            "A shading edit preserves scene and geometry identity")) {
        return false;
    }
    if (!renderFrame() || !expect(viewportPreviewValid_, "Both comparison paths render after editing")) {
        return false;
    }

    undoTransform();
    if (!expect(scene_.materials()[kMaterialIndex].roughnessFactor == before.roughnessFactor &&
                !scene_.dirty() && transformCommandCursor_ == 0,
            "Undo restores the original material and clean state")) {
        return false;
    }
    redoTransform();
    if (!expect(scene_.materials()[kMaterialIndex].roughnessFactor == edited.roughnessFactor &&
                scene_.dirty() && transformCommandCursor_ == 1,
            "Redo restores the edited material and dirty state") || !renderFrame()) {
        return false;
    }

    // The CTest runner supplies a private copy of the scene. Never point this
    // smoke at a working asset when testing persistence.
    saveScene();
    if (!expect(!scene_.dirty() && savedTransformCommandCursor_ == 1 &&
                std::filesystem::is_regular_file(scene_.documentPath()),
            "Save writes the scene sidecar and establishes a clean history point")) {
        return false;
    }
    scene::SceneDocument reloaded;
    if (!expect(reloaded.load(scene_.sourcePath()) && !reloaded.materials().empty() &&
                reloaded.materials()[kMaterialIndex].roughnessFactor == edited.roughnessFactor,
            "Reload restores the saved material parameter")) {
        return false;
    }
    undoTransform();
    if (!expect(scene_.dirty(), "Undo after saving leaves unsaved material changes")) {
        return false;
    }
    redoTransform();
    if (!expect(!scene_.dirty(), "Redo back to the saved material clears dirty state")) {
        return false;
    }

    // Keep the later panel's text input active, then start a drag in the earlier
    // panel. The old input's deactivation is reported after the new drag begins
    // and must not finish that new material's transaction.
    const auto secondBefore = scene_.materials()[1];
    inspectorFrame(ImVec2(-100.0f, -100.0f), false, -1, true);
    locatedRoughness = false;
    inspectorFrame(ImVec2(-100.0f, -100.0f), false, 1, true);
    if (!expect(locatedRoughness, "The second material panel exposes an independent Roughness control")) {
        return false;
    }
    const ImVec2 secondStart = roughnessRect.GetCenter();
    locatedRoughness = false;
    inspectorFrame(ImVec2(-100.0f, -100.0f), false, 0, true);
    if (!expect(locatedRoughness, "The first material remains available with both panels drawn")) {
        return false;
    }
    const ImVec2 firstStart = roughnessRect.GetCenter();
    inspectorFrame(secondStart, false, -1, true, true);
    inspectorFrame(secondStart, true, -1, true, true);
    inspectorFrame(secondStart, false, -1, true);
    inspectorFrame(secondStart, false, -1, true, false, "0.55");
    inspectorFrame(secondStart, false, -1, true);
    const auto* secondActive = std::get_if<MaterialEditValue>(&inspectorPropertyStartValue_);
    if (!expect(inspectorPropertyEditing_ && secondActive != nullptr && secondActive->materialIndex == 1 &&
                GImGui->TempInputId != 0 && GImGui->ActiveId == GImGui->TempInputId,
            "Ctrl-click starts an active text edit in the second material") ||
        !expect(std::abs(scene_.materials()[1].roughnessFactor - secondBefore.roughnessFactor) > 0.0001f,
            "Typing updates the material while the numeric input remains active")) {
        spdlog::error("[Smoke Material Inspector] Handoff input state: transaction={} material={} activeId={} tempInputId={} roughness={} original={}",
            inspectorPropertyEditing_, secondActive != nullptr ? secondActive->materialIndex : -1,
            GImGui->ActiveId, GImGui->TempInputId, scene_.materials()[1].roughnessFactor, secondBefore.roughnessFactor);
        return false;
    }
    inspectorFrame(firstStart, false, -1, true);
    inspectorFrame(firstStart, true, -1, true);
    const auto* firstActive = std::get_if<MaterialEditValue>(&inspectorPropertyStartValue_);
    if (!expect(inspectorPropertyEditing_ && firstActive != nullptr && firstActive->materialIndex == 0 &&
                transformCommands_.size() == 2,
            "An old panel's deactivation preserves the newly activated material transaction")) {
        return false;
    }
    inspectorFrame(ImVec2(firstStart.x - 20.0f, firstStart.y), true, -1, true);
    inspectorFrame(ImVec2(firstStart.x - 40.0f, firstStart.y), true, -1, true);
    inspectorFrame(ImVec2(firstStart.x - 40.0f, firstStart.y), false, -1, true);
    if (!expect(!inspectorPropertyEditing_ && transformCommands_.size() == 3 && transformCommandCursor_ == 3 &&
                std::abs(scene_.materials()[0].roughnessFactor - edited.roughnessFactor) > 0.0001f,
            "Each material handoff produces exactly one complete undo command")) {
        return false;
    }
    undoTransform();
    if (!expect(scene::materialPropertiesEqual(scene_.materials()[0], edited) &&
                !scene::materialPropertiesEqual(scene_.materials()[1], secondBefore),
            "Undo of the first panel leaves the second panel's edit intact")) {
        return false;
    }
    undoTransform();
    if (!expect(scene::materialPropertiesEqual(scene_.materials()[1], secondBefore) && !scene_.dirty(),
            "Undo of both handoff edits restores the saved materials and clean state")) {
        return false;
    }

    spdlog::info("[Smoke Material Inspector] Passed real Roughness drag, grouped undo/redo, material handoff, dirty state, save/reload and comparison rendering ({} -> {})",
        before.roughnessFactor, edited.roughnessFactor);
    return true;
}

} // namespace metallic
