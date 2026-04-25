#include <spdlog/spdlog.h>

#include "scene_graph.h"
#include "cluster_lod_builder.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include <algorithm>
#include <cfloat>
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

static bool nearlyZero(float value, float epsilon = 1e-4f) {
    return std::fabs(value) <= epsilon;
}

static bool nearlyOne(float value, float epsilon = 1e-4f) {
    return std::fabs(value - 1.0f) <= epsilon;
}

static bool isIdentityRotation(const float4& rotation, float epsilon = 1e-4f) {
    return std::fabs(rotation.x) <= epsilon &&
           std::fabs(rotation.y) <= epsilon &&
           std::fabs(rotation.z) <= epsilon &&
           std::fabs(std::fabs(rotation.w) - 1.0f) <= epsilon;
}

static bool isIdentityTransform(const TransformComponent& transform) {
    if (transform.useLocalMatrix) {
        return false;
    }

    return nearlyZero(transform.translation.x) &&
           nearlyZero(transform.translation.y) &&
           nearlyZero(transform.translation.z) &&
           isIdentityRotation(transform.rotation) &&
           nearlyOne(transform.scale.x) &&
           nearlyOne(transform.scale.y) &&
           nearlyOne(transform.scale.z);
}

static bool descendantsHaveIdentityTransforms(const SceneGraph& scene, uint32_t nodeId) {
    if (nodeId >= scene.nodes.size()) {
        return false;
    }

    for (uint32_t childId : scene.nodes[nodeId].children) {
        if (childId >= scene.nodes.size() ||
            !isIdentityTransform(scene.nodes[childId].transform) ||
            !descendantsHaveIdentityTransforms(scene, childId)) {
            return false;
        }
    }

    return true;
}

bool SceneGraph::applyBakedSingleRootScale(const LoadedMesh& mesh) {
    if (!mesh.hasBakedRootScale || nearlyOne(mesh.bakedRootScale)) {
        return true;
    }

    std::vector<uint32_t> sceneRoots;
    sceneRoots.reserve(rootNodes.size());
    for (uint32_t candidateRootId : rootNodes) {
        if (candidateRootId < nodes.size() && !nodes[candidateRootId].hasLight) {
            sceneRoots.push_back(candidateRootId);
        }
    }

    if (sceneRoots.size() != 1) {
        spdlog::error("SceneGraph: Expected one scene root for baked root scale {}, found {}",
                      mesh.bakedRootScale,
                      sceneRoots.size());
        return false;
    }

    const uint32_t rootId = sceneRoots.front();
    if (rootId >= nodes.size()) {
        spdlog::error("SceneGraph: Invalid root node {} for baked root scale {}", rootId, mesh.bakedRootScale);
        return false;
    }

    SceneNode& rootNode = nodes[rootId];
    TransformComponent& transform = rootNode.transform;
    const bool hasIdentityTranslation =
        nearlyZero(transform.translation.x) &&
        nearlyZero(transform.translation.y) &&
        nearlyZero(transform.translation.z);
    const bool hasIdentityRot = isIdentityRotation(transform.rotation);
    const bool hasUniformPositiveScale =
        transform.scale.x > 1e-6f &&
        std::fabs(transform.scale.x - transform.scale.y) <= 1e-5f &&
        std::fabs(transform.scale.x - transform.scale.z) <= 1e-5f;

    if (transform.useLocalMatrix ||
        !hasIdentityTranslation ||
        !hasIdentityRot ||
        !hasUniformPositiveScale ||
        std::fabs(transform.scale.x - mesh.bakedRootScale) > 1e-5f ||
        !descendantsHaveIdentityTransforms(*this, rootId)) {
        spdlog::error("SceneGraph: Baked root scale {} no longer matches scene root '{}'",
                      mesh.bakedRootScale,
                      rootNode.name);
        return false;
    }

    transform.translation = float3(0.f, 0.f, 0.f);
    transform.rotation = float4(0.f, 0.f, 0.f, 1.f);
    transform.scale = float3(1.f, 1.f, 1.f);
    transform.localMatrix = float4x4::Identity();
    transform.worldMatrix = float4x4::Identity();
    transform.useLocalMatrix = false;

    markDirty(rootId);
    spdlog::info("SceneGraph: Applied baked root scale {} for '{}'",
                 mesh.bakedRootScale,
                 rootNode.name);
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
