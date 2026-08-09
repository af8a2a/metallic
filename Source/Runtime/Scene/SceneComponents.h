#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <entt/entity/entity.hpp>

#include "ml.h"

namespace metallic::scene {

using SceneEntity = entt::entity;
inline constexpr SceneEntity kNullSceneEntity = entt::null;

struct TagComponent {
    std::string name;
};

// Stable import identity used by documents and compatibility projections.
// EnTT entity values are intentionally never serialized.
struct SourceNodeComponent {
    int32_t nodeIndex = -1;
};

struct TransformComponent {
    float4x4 authoredLocalMatrix = float4x4::Identity();
    float4x4 localMatrix = float4x4::Identity();
    float4x4 worldMatrix = float4x4::Identity();
    uint64_t transformRevision = 0;
    bool dirty = true;
};

struct RelationshipComponent {
    SceneEntity parent = kNullSceneEntity;
    std::vector<SceneEntity> children;
};

struct VisibilityComponent {
    bool localVisible = true;
    bool worldVisible = true;
};

struct MeshComponent {
    int32_t meshIndex = -1;
    std::vector<int32_t> renderNodeIndices;
};

struct CameraComponent {
    int32_t cameraIndex = -1;
    int32_t renderCameraIndex = -1;
};

struct LightComponent {
    int32_t lightIndex = -1;
    int32_t renderLightIndex = -1;
};

// Marks a runtime graph root, including generated root objects.
struct RootComponent {};

// Marks every object reachable from the current runtime roots.
struct ActiveSceneComponent {};

// Marks runtime-created objects that have no serialized source node.
struct GeneratedComponent {};

} // namespace metallic::scene
