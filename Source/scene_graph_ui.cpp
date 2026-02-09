#include "scene_graph_ui.h"
#include "scene_graph.h"
#include "imgui.h"
#include <cmath>

static float3 quaternionToEulerDeg(const float4& q) {
    // Roll (X)
    float sinr = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    float roll = std::atan2(sinr, cosr);

    // Pitch (Y)
    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    float pitch;
    if (std::fabs(sinp) >= 1.0f)
        pitch = std::copysign(3.14159265f / 2.0f, sinp);
    else
        pitch = std::asin(sinp);

    // Yaw (Z)
    float siny = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    float yaw = std::atan2(siny, cosy);

    const float toDeg = 180.0f / 3.14159265f;
    return float3(roll * toDeg, pitch * toDeg, yaw * toDeg);
}

static float4 eulerDegToQuaternion(const float3& eulerDeg) {
    const float toRad = 3.14159265f / 180.0f;
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

static void drawNodeTree(SceneGraph& scene, uint32_t nodeIdx) {
    SceneNode& node = scene.nodes[nodeIdx];

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.children.empty())
        flags |= ImGuiTreeNodeFlags_Leaf;
    if (scene.selectedNode == static_cast<int32_t>(nodeIdx))
        flags |= ImGuiTreeNodeFlags_Selected;

    if (!node.visible)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

    bool open = ImGui::TreeNodeEx((void*)(uintptr_t)nodeIdx, flags, "%s", node.name.c_str());

    if (!node.visible)
        ImGui::PopStyleColor();

    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
        scene.selectedNode = static_cast<int32_t>(nodeIdx);

    if (open) {
        for (uint32_t childIdx : node.children)
            drawNodeTree(scene, childIdx);
        ImGui::TreePop();
    }
}

static void drawPropertyPanel(SceneGraph& scene) {
    if (scene.selectedNode < 0 || scene.selectedNode >= static_cast<int32_t>(scene.nodes.size())) {
        ImGui::TextDisabled("No node selected");
        return;
    }

    SceneNode& node = scene.nodes[scene.selectedNode];
    TransformComponent& transform = node.transform;

    ImGui::Text("Name: %s", node.name.c_str());
    ImGui::Text("ID: %u", node.id);
    ImGui::Separator();

    bool transformChanged = false;

    ImGui::Checkbox("Visible", &node.visible);

    ImGui::Separator();
    ImGui::Text("Transform");

    float t[3] = {transform.translation.x, transform.translation.y, transform.translation.z};
    if (ImGui::DragFloat3("Translation", t, 0.01f)) {
        transform.translation = float3(t[0], t[1], t[2]);
        transform.useLocalMatrix = false;
        transformChanged = true;
    }

    float3 euler = quaternionToEulerDeg(transform.rotation);
    float e[3] = {euler.x, euler.y, euler.z};
    if (ImGui::DragFloat3("Rotation", e, 0.1f)) {
        transform.rotation = eulerDegToQuaternion(float3(e[0], e[1], e[2]));
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
        scene.markDirty(static_cast<uint32_t>(scene.selectedNode));

    ImGui::Separator();
    ImGui::Text("Mesh Info");
    if (node.meshIndex >= 0) {
        ImGui::Text("Mesh Index: %d", node.meshIndex);
        ImGui::Text("Meshlet Start: %u", node.meshletStart);
        ImGui::Text("Meshlet Count: %u", node.meshletCount);
        ImGui::Text("Index Start: %u", node.indexStart);
        ImGui::Text("Index Count: %u", node.indexCount);
    } else {
        ImGui::TextDisabled("No mesh (transform node)");
    }
}

void drawSceneGraphUI(SceneGraph& scene) {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Scene Graph"))  {
        ImGui::End();
        return;
    }

    float totalWidth = ImGui::GetContentRegionAvail().x;
    float treeWidth = totalWidth * 0.4f;

    // Left panel: tree view
    ImGui::BeginChild("TreePanel", ImVec2(treeWidth, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeX);
    for (uint32_t rootIdx : scene.rootNodes)
        drawNodeTree(scene, rootIdx);
    ImGui::EndChild();

    ImGui::SameLine();

    // Right panel: property editor
    ImGui::BeginChild("PropertyPanel", ImVec2(0, 0), ImGuiChildFlags_Borders);
    drawPropertyPanel(scene);
    ImGui::EndChild();

    ImGui::End();
}
