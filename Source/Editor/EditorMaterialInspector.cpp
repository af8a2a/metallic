#include "Editor/EditorApplication.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include <algorithm>
#include <string>

namespace metallic {

std::vector<int32_t> EditorApplication::selectedMaterialIndices() const
{
    std::vector<int32_t> indices;
    const auto add = [&](int32_t index) {
        if (index >= 0 && static_cast<size_t>(index) < scene_.materials().size() &&
            std::find(indices.begin(), indices.end(), index) == indices.end()) {
            indices.push_back(index);
        }
    };
    if (sceneSelection_.type == SceneSelectionType::Material) {
        add(sceneSelection_.index);
    } else if (sceneSelection_.type == SceneSelectionType::RenderPrimitive) {
        if (sceneSelection_.index >= 0 &&
            static_cast<size_t>(sceneSelection_.index) < scene_.renderPrimitives().size()) {
            add(scene_.renderPrimitives()[sceneSelection_.index].materialIndex);
        }
    } else if (sceneSelection_.type == SceneSelectionType::Mesh) {
        for (const auto& primitive : scene_.renderPrimitives()) {
            if (primitive.meshIndex == sceneSelection_.index) { add(primitive.materialIndex); }
        }
    } else if (const auto object = selectedSceneObject()) {
        for (const auto& node : scene_.renderNodes()) {
            if (node.object == object.entity()) { add(node.materialIndex); }
        }
    }
    return indices;
}

void EditorApplication::notifySceneMaterialsChanged(bool invalidateAccelerationStructure)
{
    renderWorld_.notifySceneChanged(
        render::RenderChangeBits::Material | render::RenderChangeBits::InvalidateTemporalHistory);
    historyResources_.invalidateAll();
    historyFrameIndex_ = 0;
    if (graphExecutor_ == nullptr || !graphExecutor_->compiled()) { viewportPreviewValid_ = false; }
    viewportPreviewNeedsRender_ = true;
    if (invalidateAccelerationStructure && sceneAccelerationStructure_ != nullptr &&
        sceneAccelerationStructure_->valid()) {
        sceneAccelerationStructure_->clear();
        sceneAccelerationStructureStatus_ = "Static RTAS cleared after an alpha-mask change.";
    }
}

void EditorApplication::drawSelectedMaterialInspector()
{
    const auto materials = selectedMaterialIndices();
    for (const int32_t index : materials) { drawMaterialInspector(index); }
    if (!materials.empty()) { ImGui::Separator(); }
}

void EditorApplication::drawMaterialInspector(int32_t materialIndex)
{
    if (materialIndex < 0 || static_cast<size_t>(materialIndex) >= scene_.materials().size()) { return; }
    // A frame-local copy keeps the command's before value stable during editing.
    const scene::RenderMaterial properties = scene_.materials()[materialIndex];
    scene::RenderMaterial edited = properties;
    ImGui::PushID("MaterialInspector");
    ImGui::PushID(materialIndex);
    const std::string title = "Material: " + (properties.name.empty()
        ? std::string("Unnamed") : properties.name) + "###MaterialProperties";
    if (!ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto* active = std::get_if<MaterialEditValue>(&inspectorPropertyStartValue_);
        if (inspectorPropertyEditing_ && active != nullptr && active->materialIndex == materialIndex) {
            finishActiveInspectorPropertyTransaction();
        }
        ImGui::PopID();
        ImGui::PopID();
        return;
    }

    const auto references = std::count_if(scene_.renderNodes().begin(), scene_.renderNodes().end(),
        [materialIndex](const auto& node) { return node.materialIndex == materialIndex; });
    ImGui::TextDisabled("Material %d | %zu primitive instances", materialIndex, static_cast<size_t>(references));
    if (references > 1) { ImGui::TextWrapped("Edits apply to every instance sharing this material."); }

    const bool transformEditing = gizmoWasUsing_ || inspectorTransformEditing_ || ImGuizmo::IsUsingAny();
    const bool componentEditing = inspectorPropertyEditing_ &&
        !std::holds_alternative<MaterialEditValue>(inspectorPropertyStartValue_);
    if (transformEditing || componentEditing) {
        ImGui::TextDisabled("Finish the active transform or component edit to edit this material.");
    }
    bool changed = false;
    bool editDeactivated = false;
    const auto beginEdit = [&]() {
        beginInspectorPropertyEdit(scene::kNullSceneEntity, scene_.sceneGraph().lifetimeRevision(),
            MaterialEditValue{materialIndex, properties});
    };
    const auto trackEdit = [&]() {
        if (ImGui::IsItemActivated()) { beginEdit(); }
        editDeactivated = ImGui::IsItemDeactivated() || editDeactivated;
    };
    const auto scalar = [&](const char* label, float& value, float min, float max, float step = 0.01f) {
        changed = ImGui::DragFloat(label, &value, step, min, max, "%.3f", ImGuiSliderFlags_AlwaysClamp) || changed;
        trackEdit();
    };
    const auto color = [&](const char* label, float3& value, bool hdr = false) {
        float values[3]{value.x, value.y, value.z};
        if (ImGui::ColorEdit3(label, values, ImGuiColorEditFlags_Float | (hdr ? ImGuiColorEditFlags_HDR : 0))) {
            value = float3(values[0], values[1], values[2]);
            changed = true;
        }
        trackEdit();
    };

    ImGui::BeginDisabled(transformEditing || componentEditing);
    // Keyboard input previews immediately, just like dragging. Applying values
    // before focus leaves also preserves transaction order between materials.
    ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, true);
    ImGui::PushItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x * 0.55f));
    float base[4]{edited.baseColorFactor.x, edited.baseColorFactor.y, edited.baseColorFactor.z, edited.baseColorFactor.w};
    if (ImGui::ColorEdit4("Base color", base, ImGuiColorEditFlags_Float)) {
        edited.baseColorFactor = float4(base[0], base[1], base[2], base[3]);
        changed = true;
    }
    trackEdit();
    scalar("Metallic", edited.metallicFactor, 0.0f, 1.0f);
    scalar("Roughness", edited.roughnessFactor, 0.0f, 1.0f);
    color("Emission", edited.emissiveFactor, true);

    if (ImGui::CollapsingHeader("Surface details")) {
        scalar("Normal scale", edited.normalTextureScale, -10.0f, 10.0f);
        scalar("Occlusion", edited.occlusionTextureStrength, 0.0f, 1.0f);
        if (ImGui::BeginCombo("Alpha mode", edited.alphaMode.c_str())) {
            for (const char* mode : {"OPAQUE", "MASK", "BLEND"}) {
                if (ImGui::Selectable(mode, edited.alphaMode == mode) && edited.alphaMode != mode) {
                    beginEdit();
                    edited.alphaMode = mode;
                    changed = true;
                    editDeactivated = true;
                }
            }
            ImGui::EndCombo();
        }
        if (edited.alphaMode == "MASK") { scalar("Alpha cutoff", edited.alphaCutoff, 0.0f, 1.0f); }
        if (edited.alphaMode == "BLEND") {
            ImGui::TextWrapped("VBuffer currently skips alpha-blended surfaces.");
        }
        if (ImGui::Checkbox("Double sided", &edited.doubleSided)) {
            beginEdit();
            changed = true;
            editDeactivated = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Controls raster back-face culling. Ray paths currently trace both faces.");
        }
    }
    if (ImGui::CollapsingHeader("Transmission and volume")) {
        scalar("Transmission", edited.transmissionFactor, 0.0f, 1.0f);
        scalar("IOR", edited.ior, 1.0f, 10.0f);
        scalar("Thickness", edited.thicknessFactor, 0.0f, 1000000.0f);
        scalar("Attenuation distance", edited.attenuationDistance, 0.0f, 1000000.0f);
        ImGui::TextDisabled("Distance 0 disables volume attenuation.");
        color("Attenuation color", edited.attenuationColor);
        scalar("Diffuse transmission", edited.diffuseTransmissionFactor, 0.0f, 1.0f);
        color("Diffuse tint", edited.diffuseTransmissionColor);
    }
    ImGui::PopItemWidth();
    ImGui::PopItemFlag();
    ImGui::EndDisabled();

    if (changed && !scene::materialPropertiesEqual(properties, edited)) {
        beginEdit();
        if (applySceneEditValue(scene::kNullSceneEntity, MaterialEditValue{materialIndex, edited})) {
            sceneStatus_ = "Updated material: " + properties.name;
        } else {
            sceneStatus_ = "Material values must be finite and within their supported ranges.";
        }
    }
    const auto* activeEdit = std::get_if<MaterialEditValue>(&inspectorPropertyStartValue_);
    if (editDeactivated && activeEdit != nullptr && activeEdit->materialIndex == materialIndex) {
        finishActiveInspectorPropertyTransaction();
    }

    if (ImGui::CollapsingHeader("Texture inputs")) {
        ImGui::TextWrapped("Factors multiply the assigned textures. Color values use linear RGB.");
        const auto texture = [&](const char* label, const scene::RenderTextureInfo& info) {
            if (info.textureIndex < 0) { ImGui::TextDisabled("%s: none", label); }
            else { ImGui::Text("%s: texture %d, UV %d", label, info.textureIndex, info.texCoord); }
        };
        texture("Base color", properties.baseColorTexture);
        texture("Metallic / roughness", properties.metallicRoughnessTexture);
        texture("Normal", properties.normalTexture);
        texture("Occlusion", properties.occlusionTexture);
        texture("Emission", properties.emissiveTexture);
        texture("Transmission", properties.transmissionTexture);
        texture("Thickness", properties.thicknessTexture);
        texture("Diffuse transmission", properties.diffuseTransmissionTexture);
        texture("Diffuse tint", properties.diffuseTransmissionColorTexture);
    }
    ImGui::TextDisabled("Ctrl+Z / Ctrl+Y: undo / redo | Ctrl+S: save scene");
    ImGui::PopID();
    ImGui::PopID();
}

} // namespace metallic
