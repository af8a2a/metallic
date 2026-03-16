#include <spdlog/spdlog.h>

#include "scene_graph.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include <cgltf.h>
#include <algorithm>
#include <cmath>

static float4x4 computeTRS(const float3& t, const float4& q, const float3& s) {
    float4x4 T, R, S;
    T.SetupByTranslation(t);
    R.SetupByQuaternion(q);
    S.SetupByScale(s);
    return T * R * S;
}

static float4 quaternionFromBasis(const float3& c0, const float3& c1, const float3& c2) {
    const float m00 = c0.x;
    const float m01 = c1.x;
    const float m02 = c2.x;
    const float m10 = c0.y;
    const float m11 = c1.y;
    const float m12 = c2.y;
    const float m20 = c0.z;
    const float m21 = c1.z;
    const float m22 = c2.z;

    float4 q;
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / s;
        q.x = 0.25f * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25f * s;
        q.z = (m12 + m21) / s;
    } else {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25f * s;
    }

    const float len = length(float3(q.x, q.y, q.z));
    const float norm = std::sqrt(len * len + q.w * q.w);
    if (norm > 1e-6f) {
        q /= norm;
    } else {
        q = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return q;
}

static void decomposeLocalMatrix(const float* m, TransformComponent& transform) {
    transform.translation = float3(m[12], m[13], m[14]);

    float3 basisX(m[0], m[1], m[2]);
    float3 basisY(m[4], m[5], m[6]);
    float3 basisZ(m[8], m[9], m[10]);

    transform.scale = float3(length(basisX), length(basisY), length(basisZ));

    const float3 safeScale(
        transform.scale.x > 1e-6f ? transform.scale.x : 1.0f,
        transform.scale.y > 1e-6f ? transform.scale.y : 1.0f,
        transform.scale.z > 1e-6f ? transform.scale.z : 1.0f);

    basisX /= safeScale.x;
    basisY /= safeScale.y;
    basisZ /= safeScale.z;
    transform.rotation = quaternionFromBasis(basisX, basisY, basisZ);
}

static std::string makeNodeName(const cgltf_node* gltfNode, uint32_t nodeIndex) {
    if (gltfNode->name && gltfNode->name[0] != '\0') {
        return gltfNode->name;
    }
    if (gltfNode->mesh && gltfNode->mesh->name && gltfNode->mesh->name[0] != '\0') {
        return gltfNode->mesh->name;
    }
    return "Node_" + std::to_string(nodeIndex);
}

static std::string makePrimitiveNodeName(const SceneNode& parentNode,
                                         const cgltf_primitive& primitive,
                                         uint32_t primitiveIndex,
                                         uint32_t materialIndex) {
    if (primitive.material && primitive.material->name && primitive.material->name[0] != '\0') {
        return primitive.material->name;
    }

    std::string name = parentNode.name + "/Primitive_" + std::to_string(primitiveIndex);
    name += " [Mat " + std::to_string(materialIndex) + "]";
    return name;
}

static void assignPrimitiveGroupRange(SceneNode& node,
                                      int32_t meshIndex,
                                      uint32_t firstGroup,
                                      uint32_t groupCount,
                                      const LoadedMesh& mesh,
                                      const std::vector<uint32_t>& meshletGroupPrefix) {
    node.meshIndex = meshIndex;

    const uint32_t totalGroupCount = static_cast<uint32_t>(mesh.primitiveGroups.size());
    const uint32_t clampedFirstGroup = std::min(firstGroup, totalGroupCount);
    const uint32_t clampedLastGroup = std::min(firstGroup + groupCount, totalGroupCount);
    if (clampedFirstGroup >= clampedLastGroup) {
        return;
    }

    if (clampedLastGroup < meshletGroupPrefix.size()) {
        node.meshletStart = meshletGroupPrefix[clampedFirstGroup];
        node.meshletCount = meshletGroupPrefix[clampedLastGroup] - node.meshletStart;
    }

    node.indexStart = mesh.primitiveGroups[clampedFirstGroup].indexOffset;
    const auto& lastPrim = mesh.primitiveGroups[clampedLastGroup - 1];
    const uint32_t indexEnd = lastPrim.indexOffset + lastPrim.indexCount;
    node.indexCount = indexEnd - node.indexStart;
}

static void addNodeRecursive(const cgltf_data* data,
                             const cgltf_node* gltfNode,
                             int32_t parentIdx,
                             const LoadedMesh& mesh,
                             const std::vector<uint32_t>& meshletGroupPrefix,
                             SceneGraph& sg) {
    uint32_t nodeIdx = static_cast<uint32_t>(sg.nodes.size());
    sg.nodes.emplace_back();
    sg.nodes[nodeIdx].id = nodeIdx;
    sg.nodes[nodeIdx].parent = parentIdx;
    sg.nodes[nodeIdx].name = makeNodeName(gltfNode, nodeIdx);

    // Extract TRS
    if (gltfNode->has_translation) {
        sg.nodes[nodeIdx].transform.translation = float3(gltfNode->translation[0],
                                                         gltfNode->translation[1],
                                                         gltfNode->translation[2]);
    }
    if (gltfNode->has_rotation) {
        sg.nodes[nodeIdx].transform.rotation = float4(gltfNode->rotation[0],
                                                      gltfNode->rotation[1],
                                                      gltfNode->rotation[2],
                                                      gltfNode->rotation[3]);
    }
    if (gltfNode->has_scale) {
        sg.nodes[nodeIdx].transform.scale = float3(gltfNode->scale[0],
                                                   gltfNode->scale[1],
                                                   gltfNode->scale[2]);
    }

    if (gltfNode->has_matrix) {
        const float* m = gltfNode->matrix;
        float4x4 mat;
        // cgltf stores column-major
        mat.Col(0) = float4(m[0], m[1], m[2], m[3]);
        mat.Col(1) = float4(m[4], m[5], m[6], m[7]);
        mat.Col(2) = float4(m[8], m[9], m[10], m[11]);
        mat.Col(3) = float4(m[12], m[13], m[14], m[15]);
        sg.nodes[nodeIdx].transform.localMatrix = mat;
        sg.nodes[nodeIdx].transform.useLocalMatrix = true;
        decomposeLocalMatrix(m, sg.nodes[nodeIdx].transform);
    }

    if (parentIdx >= 0) {
        sg.nodes[parentIdx].children.push_back(nodeIdx);
    }

    // Map mesh reference
    if (gltfNode->mesh) {
        int32_t mi = static_cast<int32_t>(gltfNode->mesh - data->meshes);
        if (mi >= 0 && mi < static_cast<int32_t>(mesh.meshRanges.size())) {
            const auto& range = mesh.meshRanges[mi];
            if (range.groupCount > 1) {
                uint32_t trianglePrimitiveIndex = 0;
                for (cgltf_size primitiveIndex = 0;
                     primitiveIndex < gltfNode->mesh->primitives_count;
                     ++primitiveIndex) {
                    const cgltf_primitive& primitive = gltfNode->mesh->primitives[primitiveIndex];
                    if (primitive.type != cgltf_primitive_type_triangles) {
                        continue;
                    }

                    const uint32_t groupIndex = range.firstGroup + trianglePrimitiveIndex;
                    if (groupIndex >= mesh.primitiveGroups.size()) {
                        break;
                    }

                    const uint32_t childIdx = static_cast<uint32_t>(sg.nodes.size());
                    sg.nodes.emplace_back();
                    SceneNode& childNode = sg.nodes.back();
                    childNode.id = childIdx;
                    childNode.parent = static_cast<int32_t>(nodeIdx);
                    childNode.name = makePrimitiveNodeName(sg.nodes[nodeIdx],
                                                           primitive,
                                                           trianglePrimitiveIndex,
                                                           mesh.primitiveGroups[groupIndex].materialIndex);
                    assignPrimitiveGroupRange(childNode,
                                              mi,
                                              groupIndex,
                                              1,
                                              mesh,
                                              meshletGroupPrefix);
                    sg.nodes[nodeIdx].children.push_back(childIdx);
                    ++trianglePrimitiveIndex;
                }
            } else {
                assignPrimitiveGroupRange(sg.nodes[nodeIdx],
                                          mi,
                                          range.firstGroup,
                                          range.groupCount,
                                          mesh,
                                          meshletGroupPrefix);
            }
        }
    }

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

    if (!rootNodes.empty()) {
        selectedNode = static_cast<int32_t>(rootNodes.front());
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
