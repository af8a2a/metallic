#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "scene_graph_ui.h"

#include "scene_graph.h"
#include "imgui.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr uintptr_t kSceneTreeId = 0x10000000u;
constexpr uintptr_t kMeshTreeIdBase = 0x20000000u;

const ImGuiTreeNodeFlags kTreeNodeFlags =
    ImGuiTreeNodeFlags_SpanAllColumns |
    ImGuiTreeNodeFlags_SpanFullWidth |
    ImGuiTreeNodeFlags_SpanTextWidth |
    ImGuiTreeNodeFlags_OpenOnArrow |
    ImGuiTreeNodeFlags_OpenOnDoubleClick;

const ImGuiTableFlags kBrowserTableFlags =
    ImGuiTableFlags_ScrollY |
    ImGuiTableFlags_RowBg |
    ImGuiTableFlags_BordersOuter |
    ImGuiTableFlags_BordersV |
    ImGuiTableFlags_Resizable;

float3 quaternionToEulerDeg(const float4& q) {
    float sinr = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    float roll = std::atan2(sinr, cosr);

    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    float pitch = 0.0f;
    if (std::fabs(sinp) >= 1.0f)
        pitch = std::copysign(kPi * 0.5f, sinp);
    else
        pitch = std::asin(sinp);

    float siny = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    float yaw = std::atan2(siny, cosy);

    const float toDeg = 180.0f / kPi;
    return float3(roll * toDeg, pitch * toDeg, yaw * toDeg);
}

float4 eulerDegToQuaternion(const float3& eulerDeg) {
    const float toRad = kPi / 180.0f;
    float rx = eulerDeg.x * toRad * 0.5f;
    float ry = eulerDeg.y * toRad * 0.5f;
    float rz = eulerDeg.z * toRad * 0.5f;

    float cx = std::cos(rx), sx = std::sin(rx);
    float cy = std::cos(ry), sy = std::sin(ry);
    float cz = std::cos(rz), sz = std::sin(rz);

    return float4(
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz);
}

bool isValidNode(const SceneGraph& scene, uint32_t nodeIdx) {
    return nodeIdx < scene.nodes.size();
}

std::string nodeDisplayName(const SceneNode& node) {
    if (!node.name.empty())
        return node.name;
    return "Node " + std::to_string(node.id);
}

std::string meshDisplayName(const SceneNode& node) {
    const std::string name = node.name.empty() ? "Mesh" : node.name;
    return "[" + std::to_string(node.meshIndex) + "] " + name;
}

std::string cameraDisplayName(const SceneNode& node) {
    const std::string name = node.name.empty() ? "default" : node.name;
    return "Camera: " + name;
}

std::string lightDisplayName(const SceneNode& node) {
    const std::string name = node.name.empty() ? "Light" : node.name;
    return "Light: " + name;
}

std::vector<uint32_t> collectPrimitiveRows(const SceneGraph& scene, uint32_t nodeIdx) {
    std::vector<uint32_t> rows;
    if (!isValidNode(scene, nodeIdx))
        return rows;

    const SceneNode& node = scene.nodes[nodeIdx];
    if (node.primitiveGroupCount > 0)
        rows.push_back(nodeIdx);

    for (uint32_t childIdx : node.children) {
        if (!isValidNode(scene, childIdx))
            continue;
        const SceneNode& child = scene.nodes[childIdx];
        if (child.generatedPrimitive && child.meshIndex == node.meshIndex)
            rows.push_back(childIdx);
    }

    return rows;
}

bool hasVisibleTreeChildren(const SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return false;

    const SceneNode& node = scene.nodes[nodeIdx];
    if (node.meshIndex >= 0 || node.hasLight || node.cameraIndex >= 0)
        return true;

    for (uint32_t childIdx : node.children) {
        if (isValidNode(scene, childIdx) && !scene.nodes[childIdx].generatedPrimitive)
            return true;
    }

    return false;
}

size_t countPrimitiveRows(const SceneGraph& scene) {
    size_t count = 0;
    for (const SceneNode& node : scene.nodes) {
        if (node.generatedPrimitive || node.primitiveGroupCount > 0)
            ++count;
    }
    return count;
}

size_t countDisplayNodes(const SceneGraph& scene) {
    size_t count = 0;
    for (const SceneNode& node : scene.nodes) {
        if (!node.generatedPrimitive)
            ++count;
    }
    return count;
}

size_t countMeshNodes(const SceneGraph& scene) {
    size_t count = 0;
    for (const SceneNode& node : scene.nodes) {
        if (!node.generatedPrimitive && node.meshIndex >= 0)
            ++count;
    }
    return count;
}

size_t countLights(const SceneGraph& scene) {
    size_t count = 0;
    for (const SceneNode& node : scene.nodes) {
        if (node.hasLight)
            ++count;
    }
    return count;
}

size_t countCameras(const SceneGraph& scene) {
    size_t count = 0;
    for (const SceneNode& node : scene.nodes) {
        if (node.cameraIndex >= 0)
            ++count;
    }
    return count;
}

void statRow(const char* label, size_t value) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", label);
    ImGui::TableNextColumn();
    ImGui::Text("%zu", value);
}

void renderAssetInfoTab(const SceneGraph& scene) {
    if (ImGui::BeginTable("SceneAssetInfoTable", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        statRow("Root Nodes", scene.rootNodes.size());
        statRow("Nodes", countDisplayNodes(scene));
        statRow("Mesh Nodes", countMeshNodes(scene));
        statRow("Primitive Rows", countPrimitiveRows(scene));
        statRow("Lights", countLights(scene));
        statRow("Cameras", countCameras(scene));
        ImGui::EndTable();
    }
}

void renderPrimitiveInHierarchy(SceneGraph& scene, uint32_t primitiveNodeIdx) {
    if (!isValidNode(scene, primitiveNodeIdx))
        return;

    const SceneNode& node = scene.nodes[primitiveNodeIdx];
    const bool selected = scene.selectedNode == static_cast<int32_t>(primitiveNodeIdx);
    const std::string label = "[P] Primitive " + std::to_string(node.primitiveIndexInMesh);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns))
        scene.selectedNode = static_cast<int32_t>(primitiveNodeIdx);

    ImGui::TableNextColumn();
    if (node.materialIndex != UINT32_MAX) {
        ImGui::Text("M%u", node.materialIndex);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Material %u", node.materialIndex);
    }
}

void renderMeshInHierarchy(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    SceneNode& node = scene.nodes[nodeIdx];
    std::vector<uint32_t> primitives = collectPrimitiveRows(scene, nodeIdx);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    ImGuiTreeNodeFlags flags = kTreeNodeFlags | ImGuiTreeNodeFlags_DefaultOpen;
    if (primitives.empty())
        flags |= ImGuiTreeNodeFlags_Leaf;

    const std::string label = "[M] " + meshDisplayName(node);
    const bool open = ImGui::TreeNodeEx(
        reinterpret_cast<void*>(kMeshTreeIdBase + nodeIdx),
        flags,
        "%s",
        label.c_str());

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        scene.selectedNode = static_cast<int32_t>(nodeIdx);

    ImGui::TableNextColumn();

    if (open) {
        for (uint32_t primitiveNodeIdx : primitives)
            renderPrimitiveInHierarchy(scene, primitiveNodeIdx);
        ImGui::TreePop();
    }
}

void renderCameraInHierarchy(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    const SceneNode& node = scene.nodes[nodeIdx];
    const bool selected = scene.selectedNode == static_cast<int32_t>(nodeIdx);
    const std::string label = "[C] " + cameraDisplayName(node);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns))
        scene.selectedNode = static_cast<int32_t>(nodeIdx);
    ImGui::TableNextColumn();
}

void renderLightInHierarchy(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    const SceneNode& node = scene.nodes[nodeIdx];
    const bool selected = scene.selectedNode == static_cast<int32_t>(nodeIdx);
    const std::string label = "[L] " + lightDisplayName(node);

    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns))
        scene.selectedNode = static_cast<int32_t>(nodeIdx);
    ImGui::TableNextColumn();
}

void renderNodeHierarchy(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    SceneNode& node = scene.nodes[nodeIdx];
    if (node.generatedPrimitive)
        return;

    ImGui::TableNextRow();
    ImGui::TableNextColumn();

    ImGuiTreeNodeFlags flags = kTreeNodeFlags;
    if (!hasVisibleTreeChildren(scene, nodeIdx))
        flags |= ImGuiTreeNodeFlags_Leaf;
    if (node.parent < 0)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    if (scene.selectedNode == static_cast<int32_t>(nodeIdx))
        flags |= ImGuiTreeNodeFlags_Selected;

    if (!node.visible)
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));

    const std::string label = "[N] [" + std::to_string(node.id) + "] " + nodeDisplayName(node);
    const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<uintptr_t>(nodeIdx + 1)), flags, "%s", label.c_str());

    if (!node.visible)
        ImGui::PopStyleColor();

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        scene.selectedNode = static_cast<int32_t>(nodeIdx);

    ImGui::TableNextColumn();
    ImGui::PushID(static_cast<int>(nodeIdx));
    if (ImGui::SmallButton(node.visible ? "V" : "H"))
        node.visible = !node.visible;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(node.visible ? "Visible" : "Hidden");
    ImGui::PopID();

    if (open) {
        if (node.cameraIndex >= 0)
            renderCameraInHierarchy(scene, nodeIdx);
        if (node.hasLight)
            renderLightInHierarchy(scene, nodeIdx);
        if (node.meshIndex >= 0)
            renderMeshInHierarchy(scene, nodeIdx);

        for (uint32_t childIdx : node.children) {
            if (isValidNode(scene, childIdx) && !scene.nodes[childIdx].generatedPrimitive)
                renderNodeHierarchy(scene, childIdx);
        }

        ImGui::TreePop();
    }
}

void renderSceneGraphTab(SceneGraph& scene) {
    if (ImGui::BeginTable("SceneGraphTable", 2, kBrowserTableFlags)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("VH").x + 16.0f);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableNextColumn();

        ImGuiTreeNodeFlags sceneFlags =
            ImGuiTreeNodeFlags_SpanAllColumns |
            ImGuiTreeNodeFlags_SpanFullWidth |
            ImGuiTreeNodeFlags_SpanTextWidth |
            ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_DefaultOpen;

        const bool open = ImGui::TreeNodeEx(reinterpret_cast<void*>(kSceneTreeId), sceneFlags, "Scene-0");

        ImGui::TableNextColumn();
        ImGui::TextDisabled("T");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Scene root");

        if (open) {
            for (uint32_t rootIdx : scene.rootNodes) {
                if (isValidNode(scene, rootIdx))
                    renderNodeHierarchy(scene, rootIdx);
            }
            ImGui::TreePop();
        }

        ImGui::EndTable();
    }
}

float listChildHeight(size_t rowCount) {
    const float line = ImGui::GetTextLineHeightWithSpacing();
    return std::clamp(rowCount * line + ImGui::GetStyle().FramePadding.y * 2.0f, 60.0f, 220.0f);
}

void renderNodesGroup(SceneGraph& scene) {
    const size_t nodeCount = countDisplayNodes(scene);
    const std::string header = "[N] Nodes (" + std::to_string(nodeCount) + ")";
    if (!ImGui::CollapsingHeader(header.c_str()))
        return;

    ImGui::BeginChild("NodesScrollRegion", ImVec2(0.0f, listChildHeight(nodeCount)), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const SceneNode& node = scene.nodes[i];
        if (node.generatedPrimitive)
            continue;
        const bool selected = scene.selectedNode == static_cast<int32_t>(i);
        const std::string label = "[" + std::to_string(node.id) + "] " + nodeDisplayName(node);
        if (ImGui::Selectable(label.c_str(), selected))
            scene.selectedNode = static_cast<int32_t>(i);
    }
    ImGui::EndChild();
}

void renderMeshesGroup(SceneGraph& scene) {
    const size_t meshCount = countMeshNodes(scene);
    const std::string header = "[M] Meshes (" + std::to_string(meshCount) + ")";
    if (!ImGui::CollapsingHeader(header.c_str()))
        return;

    ImGui::BeginChild("MeshesScrollRegion", ImVec2(0.0f, listChildHeight(meshCount)), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const SceneNode& node = scene.nodes[i];
        if (node.generatedPrimitive || node.meshIndex < 0)
            continue;
        const bool selected = scene.selectedNode == static_cast<int32_t>(i);
        const std::string label = meshDisplayName(node);
        if (ImGui::Selectable(label.c_str(), selected))
            scene.selectedNode = static_cast<int32_t>(i);
    }
    ImGui::EndChild();
}

void renderPrimitivesGroup(SceneGraph& scene) {
    const size_t primitiveCount = countPrimitiveRows(scene);
    const std::string header = "[P] Primitives (" + std::to_string(primitiveCount) + ")";
    if (!ImGui::CollapsingHeader(header.c_str()))
        return;

    ImGui::BeginChild("PrimitivesScrollRegion", ImVec2(0.0f, listChildHeight(primitiveCount)), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const SceneNode& node = scene.nodes[i];
        if (!node.generatedPrimitive && node.primitiveGroupCount == 0)
            continue;

        const bool selected = scene.selectedNode == static_cast<int32_t>(i);
        std::string label = "[" + std::to_string(node.primitiveGroupStart) + "] Primitive " +
            std::to_string(node.primitiveIndexInMesh);
        if (node.parent >= 0 && node.parent < static_cast<int32_t>(scene.nodes.size()))
            label += "  (" + nodeDisplayName(scene.nodes[node.parent]) + ")";
        else if (!node.name.empty())
            label += "  (" + node.name + ")";

        if (ImGui::Selectable(label.c_str(), selected))
            scene.selectedNode = static_cast<int32_t>(i);
    }
    ImGui::EndChild();
}

void renderCamerasGroup(SceneGraph& scene) {
    const size_t cameraCount = countCameras(scene);
    const std::string header = "[C] Cameras (" + std::to_string(cameraCount) + ")";
    if (!ImGui::CollapsingHeader(header.c_str()))
        return;

    ImGui::BeginChild("CamerasScrollRegion", ImVec2(0.0f, listChildHeight(cameraCount)), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const SceneNode& node = scene.nodes[i];
        if (node.cameraIndex < 0)
            continue;
        const bool selected = scene.selectedNode == static_cast<int32_t>(i);
        const std::string label = "[" + std::to_string(node.cameraIndex) + "] " + cameraDisplayName(node);
        if (ImGui::Selectable(label.c_str(), selected))
            scene.selectedNode = static_cast<int32_t>(i);
    }
    ImGui::EndChild();
}

void renderLightsGroup(SceneGraph& scene) {
    const size_t lightCount = countLights(scene);
    const std::string header = "[L] Lights (" + std::to_string(lightCount) + ")";
    if (!ImGui::CollapsingHeader(header.c_str()))
        return;

    ImGui::BeginChild("LightsScrollRegion", ImVec2(0.0f, listChildHeight(lightCount)), false, ImGuiWindowFlags_HorizontalScrollbar);
    for (size_t i = 0; i < scene.nodes.size(); ++i) {
        const SceneNode& node = scene.nodes[i];
        if (!node.hasLight)
            continue;
        const bool selected = scene.selectedNode == static_cast<int32_t>(i);
        const std::string label = "[" + std::to_string(node.lightIndex) + "] " + lightDisplayName(node);
        if (ImGui::Selectable(label.c_str(), selected))
            scene.selectedNode = static_cast<int32_t>(i);
    }
    ImGui::EndChild();
}

void renderSceneListTab(SceneGraph& scene) {
    renderNodesGroup(scene);
    renderMeshesGroup(scene);
    renderPrimitivesGroup(scene);
    renderCamerasGroup(scene);
    renderLightsGroup(scene);
}

void renderTransformSection(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    SceneNode& node = scene.nodes[nodeIdx];
    TransformComponent& transform = node.transform;

    if (!ImGui::CollapsingHeader("TRANSFORM", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    bool transformChanged = false;

    if (transform.useLocalMatrix)
        ImGui::TextDisabled("Authored matrix; editing TRS will replace it.");

    float t[3] = {transform.translation.x, transform.translation.y, transform.translation.z};
    if (ImGui::DragFloat3("Translation", t, 0.01f)) {
        transform.translation = float3(t[0], t[1], t[2]);
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    const float3 euler = quaternionToEulerDeg(transform.rotation);
    float r[3] = {euler.x, euler.y, euler.z};
    if (ImGui::DragFloat3("Rotation", r, 0.1f)) {
        transform.rotation = eulerDegToQuaternion(float3(r[0], r[1], r[2]));
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    float s[3] = {transform.scale.x, transform.scale.y, transform.scale.z};
    if (ImGui::DragFloat3("Scale", s, 0.01f, 0.001f, 100.0f)) {
        transform.scale = float3(s[0], s[1], s[2]);
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    if (transformChanged)
        scene.markDirty(nodeIdx);
}

void renderCameraSection(const SceneNode& node) {
    if (node.cameraIndex < 0)
        return;

    if (!ImGui::CollapsingHeader("CAMERA", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::Text("Camera Index: %d", node.cameraIndex);
    ImGui::Text("Type: %s", node.camera.type == CameraType::Perspective ? "Perspective" : "Orthographic");
    if (node.camera.type == CameraType::Perspective) {
        const float fovDeg = node.camera.yfov * 180.0f / kPi;
        ImGui::Text("Y FOV: %.3f rad (%.2f deg)", node.camera.yfov, fovDeg);
        ImGui::Text("Aspect Ratio: %.3f", node.camera.aspectRatio);
    }
    ImGui::Text("Z Near: %.4f", node.camera.znear);
    ImGui::Text("Z Far: %.4f", node.camera.zfar);
}

void renderLightSection(SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    SceneNode& node = scene.nodes[nodeIdx];
    if (!node.hasLight)
        return;

    if (!ImGui::CollapsingHeader("LIGHT", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    ImGui::Text("Type: Directional");
    if (node.lightIndex >= 0)
        ImGui::Text("Light Index: %d", node.lightIndex);

    float dir[3] = {
        node.light.directional.direction.x,
        node.light.directional.direction.y,
        node.light.directional.direction.z
    };
    if (ImGui::DragFloat3("Direction", dir, 0.01f, -1.0f, 1.0f)) {
        float3 d(dir[0], dir[1], dir[2]);
        float len = length(d);
        if (len > 1e-6f)
            node.light.directional.direction = d / len;
    }

    float color[3] = {
        node.light.directional.color.x,
        node.light.directional.color.y,
        node.light.directional.color.z
    };
    if (ImGui::ColorEdit3("Color", color))
        node.light.directional.color = float3(color[0], color[1], color[2]);

    ImGui::DragFloat("Intensity", &node.light.directional.intensity, 0.01f, 0.0f, 100.0f);

    if (scene.sunLightNode == static_cast<int32_t>(nodeIdx))
        ImGui::TextDisabled("Scene Sun Source");
}

void renderMeshSection(const SceneGraph& scene, uint32_t nodeIdx) {
    if (!isValidNode(scene, nodeIdx))
        return;

    const SceneNode& node = scene.nodes[nodeIdx];
    if (node.meshIndex < 0)
        return;

    if (!ImGui::CollapsingHeader("MESH", ImGuiTreeNodeFlags_DefaultOpen))
        return;

    const std::vector<uint32_t> primitiveRows = collectPrimitiveRows(scene, nodeIdx);
    ImGui::Text("Mesh Index: %d", node.meshIndex);
    ImGui::Text("Primitive Rows: %zu", primitiveRows.size());

    if (node.primitiveGroupCount > 0) {
        ImGui::Separator();
        ImGui::Text("Primitive Group Start: %u", node.primitiveGroupStart);
        ImGui::Text("Primitive Group Count: %u", node.primitiveGroupCount);
        if (node.materialIndex != UINT32_MAX)
            ImGui::Text("Material Index: %u", node.materialIndex);
        ImGui::Text("Meshlet Start: %u", node.meshletStart);
        ImGui::Text("Meshlet Count: %u", node.meshletCount);
        ImGui::Text("Index Start: %u", node.indexStart);
        ImGui::Text("Index Count: %u", node.indexCount);
    }
}

void renderInspector(SceneGraph& scene) {
    if (scene.selectedNode < 0 || scene.selectedNode >= static_cast<int32_t>(scene.nodes.size())) {
        ImGui::TextDisabled("No selection");
        ImGui::Separator();
        ImGui::TextWrapped("Select an element in the Scene Browser or 3D view to view its properties.");
        return;
    }

    const uint32_t nodeIdx = static_cast<uint32_t>(scene.selectedNode);
    SceneNode& node = scene.nodes[nodeIdx];

    if (node.generatedPrimitive) {
        ImGui::Text("[P] Primitive %u", node.primitiveIndexInMesh);
    } else if (node.cameraIndex >= 0) {
        ImGui::Text("[C] Node: %s", nodeDisplayName(node).c_str());
    } else if (node.hasLight) {
        ImGui::Text("[L] Node: %s", nodeDisplayName(node).c_str());
    } else {
        ImGui::Text("[N] Node: %s", nodeDisplayName(node).c_str());
    }

    ImGui::TextDisabled("ID: %u", node.id);
    if (node.generatedPrimitive && node.parent >= 0)
        ImGui::TextDisabled("Parent Node: %d", node.parent);
    ImGui::Separator();

    ImGui::Checkbox("Visible", &node.visible);

    renderTransformSection(scene, nodeIdx);
    renderCameraSection(node);
    renderLightSection(scene, nodeIdx);
    renderMeshSection(scene, nodeIdx);
}

} // namespace

void drawSceneGraphUI(SceneGraph& scene) {
    ImGui::SetNextWindowSize(ImVec2(500.0f, 520.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Scene Browser")) {
        if (ImGui::CollapsingHeader("Asset Info"))
            renderAssetInfoTab(scene);

        if (ImGui::BeginTabBar("SceneBrowserTabs")) {
            if (ImGui::BeginTabItem("Scene Graph")) {
                renderSceneGraphTab(scene);
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("Scene List")) {
                renderSceneListTab(scene);
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(500.0f, 260.0f), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Inspector")) {
        renderInspector(scene);
    }
    ImGui::End();
}
