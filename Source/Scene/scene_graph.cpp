#include "scene_graph.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include <cgltf.h>
#include <spdlog/spdlog.h>
#include <algorithm>

static float4x4 computeTRS(const float3& t, const float4& q, const float3& s) {
    float4x4 T, R, S;
    T.SetupByTranslation(t);
    R.SetupByQuaternion(q);
    S.SetupByScale(s);
    return T * R * S;
}

static void addNodeRecursive(const cgltf_data* data,
                             const cgltf_node* gltfNode,
                             int32_t parentIdx,
                             const LoadedMesh& mesh,
                             const std::vector<uint32_t>& meshletGroupPrefix,
                             SceneGraph& sg) {
    uint32_t nodeIdx = static_cast<uint32_t>(sg.nodes.size());
    sg.nodes.emplace_back();
    SceneNode& node = sg.nodes.back();
    node.id = nodeIdx;
    node.parent = parentIdx;
    node.name = gltfNode->name ? gltfNode->name : ("Node_" + std::to_string(nodeIdx));

    // Extract TRS
    if (gltfNode->has_translation) {
        node.transform.translation = float3(gltfNode->translation[0],
                                            gltfNode->translation[1],
                                            gltfNode->translation[2]);
    }
    if (gltfNode->has_rotation) {
        node.transform.rotation = float4(gltfNode->rotation[0],
                                         gltfNode->rotation[1],
                                         gltfNode->rotation[2],
                                         gltfNode->rotation[3]);
    }

    if (gltfNode->has_matrix) {
        const float* m = gltfNode->matrix;
        float4x4 mat;
        // cgltf stores column-major
        mat.Col(0) = float4(m[0], m[1], m[2], m[3]);
        mat.Col(1) = float4(m[4], m[5], m[6], m[7]);
        mat.Col(2) = float4(m[8], m[9], m[10], m[11]);
        mat.Col(3) = float4(m[12], m[13], m[14], m[15]);
        node.transform.localMatrix = mat;
        node.transform.useLocalMatrix = true;
        node.transform.translation = float3(m[12], m[13], m[14]);
        node.transform.scale = float3(1.f, 1.f, 1.f);
        node.transform.rotation = mat.GetQuaternion();
    }

    // Map mesh reference
    if (gltfNode->mesh) {
        int32_t mi = static_cast<int32_t>(gltfNode->mesh - data->meshes);
        node.meshIndex = mi;
        if (mi >= 0 && mi < static_cast<int32_t>(mesh.meshRanges.size())) {
            const auto& range = mesh.meshRanges[mi];
            uint32_t groupCount = static_cast<uint32_t>(mesh.primitiveGroups.size());
            uint32_t firstGroup = std::min(range.firstGroup, groupCount);
            uint32_t lastGroup = std::min(range.firstGroup + range.groupCount, groupCount);

            if (firstGroup < lastGroup) {
                if (lastGroup < meshletGroupPrefix.size()) {
                    node.meshletStart = meshletGroupPrefix[firstGroup];
                    node.meshletCount = meshletGroupPrefix[lastGroup] - node.meshletStart;
                }

                node.indexStart = mesh.primitiveGroups[firstGroup].indexOffset;
                const auto& lastPrim = mesh.primitiveGroups[lastGroup - 1];
                uint32_t indexEnd = lastPrim.indexOffset + lastPrim.indexCount;
                node.indexCount = indexEnd - node.indexStart;
            }
        }
    }

    // Add as child of parent
    if (parentIdx >= 0)
        sg.nodes[parentIdx].children.push_back(nodeIdx);

    // Recurse children
    for (cgltf_size ci = 0; ci < gltfNode->children_count; ci++) {
        addNodeRecursive(data, gltfNode->children[ci], static_cast<int32_t>(nodeIdx),
                         mesh, meshletGroupPrefix, sg);
    }
}

bool SceneGraph::buildFromGLTF(const std::string& gltfPath,
                                const LoadedMesh& mesh,
                                const MeshletData& meshletData) {
    nodes.clear();
    rootNodes.clear();
    selectedNode = -1;
    sunLightNode = -1;

    cgltf_options options = {};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, gltfPath.c_str(), &data);
    if (result != cgltf_result_success) {
        spdlog::error("SceneGraph: Failed to parse glTF: {}", gltfPath);
        return false;
    }

    // Build prefix sums for meshlet offsets per primitive group.
    std::vector<uint32_t> meshletGroupPrefix(mesh.primitiveGroups.size() + 1, 0);
    for (size_t i = 0; i < mesh.primitiveGroups.size(); i++) {
        uint32_t meshletsInGroup = (i < meshletData.meshletsPerGroup.size())
            ? meshletData.meshletsPerGroup[i]
            : 0;
        meshletGroupPrefix[i + 1] = meshletGroupPrefix[i] + meshletsInGroup;
    }

    // Walk scene hierarchy
    if (data->scenes_count == 0) {
        cgltf_free(data);
        return false;
    }

    const cgltf_scene& scene = data->scenes[data->scene ? (data->scene - data->scenes) : 0];
    for (cgltf_size ni = 0; ni < scene.nodes_count; ni++) {
        uint32_t rootIdx = static_cast<uint32_t>(nodes.size());
        addNodeRecursive(data, scene.nodes[ni], -1,
                         mesh, meshletGroupPrefix, *this);
        rootNodes.push_back(rootIdx);
    }

    addDirectionalLightNode("Sun", normalize(float3(0.5f, 1.0f, 0.8f)), true);

    cgltf_free(data);

    spdlog::info("SceneGraph: {} nodes, {} roots", nodes.size(), rootNodes.size());

    return true;
}

void SceneGraph::updateTransforms() {
    for (auto& node : nodes) {
        if (!node.transform.dirty) continue;

        if (!node.transform.useLocalMatrix) {
            node.transform.localMatrix = computeTRS(
                node.transform.translation,
                node.transform.rotation,
                node.transform.scale);
        }

        if (node.parent >= 0)
            node.transform.worldMatrix =
                nodes[node.parent].transform.worldMatrix * node.transform.localMatrix;
        else
            node.transform.worldMatrix = node.transform.localMatrix;

        node.transform.dirty = false;

        // Mark children dirty so they recompute
        for (uint32_t childId : node.children) {
            nodes[childId].transform.dirty = true;
        }
    }
}

void SceneGraph::markDirty(uint32_t nodeId) {
    if (nodeId >= nodes.size()) return;
    nodes[nodeId].transform.dirty = true;
    for (uint32_t childId : nodes[nodeId].children)
        markDirty(childId);
}

bool SceneGraph::isNodeVisible(uint32_t nodeId) const {
    uint32_t id = nodeId;
    while (id < nodes.size()) {
        if (!nodes[id].visible) return false;
        if (nodes[id].parent < 0) break;
        id = static_cast<uint32_t>(nodes[id].parent);
    }
    return true;
}

uint32_t SceneGraph::addDirectionalLightNode(const std::string& name,
                                             const float3& direction,
                                             bool setAsSunSource) {
    uint32_t nodeIdx = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();
    SceneNode& node = nodes.back();
    node.id = nodeIdx;
    node.name = name;
    node.hasLight = true;
    node.light.type = LightType::Directional;

    float dirLen = length(direction);
    node.light.directional.direction =
        (dirLen > 1e-6f) ? (direction / dirLen) : normalize(float3(0.5f, 1.0f, 0.8f));

    rootNodes.push_back(nodeIdx);
    if (setAsSunSource)
        sunLightNode = static_cast<int32_t>(nodeIdx);
    return nodeIdx;
}

float3 SceneGraph::getSunLightDirection() const {
    return getSunDirectionalLight().direction;
}

DirectionalLight SceneGraph::getSunDirectionalLight() const {
    if (sunLightNode >= 0 && sunLightNode < static_cast<int32_t>(nodes.size())) {
        const SceneNode& sunNode = nodes[sunLightNode];
        if (sunNode.hasLight && sunNode.light.type == LightType::Directional) {
            DirectionalLight light = sunNode.light.directional;
            float dirLen = length(light.direction);
            if (dirLen > 1e-6f)
                light.direction /= dirLen;
            else
                light.direction = normalize(float3(0.5f, 1.0f, 0.8f));
            light.intensity = std::max(0.0f, light.intensity);
            return light;
        }
    }

    DirectionalLight fallback;
    fallback.direction = normalize(float3(0.5f, 1.0f, 0.8f));
    fallback.color = float3(1.f, 1.f, 1.f);
    fallback.intensity = 1.0f;
    return fallback;
}
