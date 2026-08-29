#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <entt/entity/entity.hpp>

#include "ml.h"

namespace metallic::scene {

using SceneEntity = entt::entity;
inline constexpr SceneEntity kNullSceneEntity = entt::null;

enum class CameraType : uint8_t {
    Perspective,
    Orthographic,
};

struct CameraProperties {
    CameraType type = CameraType::Perspective;
    double yfov = 0.0;
    double aspectRatio = 0.0;
    double xmag = 0.0;
    double ymag = 0.0;
    double znear = 0.0;
    double zfar = 0.0;
};

struct LightProperties {
    std::string type;
    float3 color{1.0f, 1.0f, 1.0f};
    double intensity = 1.0;
    double range = 0.0;
    double innerConeAngle = 0.0;
    double outerConeAngle = 0.7853981633974483;
};

bool validCameraProperties(const CameraProperties& properties);
bool validLightProperties(const LightProperties& properties);
bool cameraPropertiesNearlyEqual(
    const CameraProperties& lhs,
    const CameraProperties& rhs);
bool lightPropertiesNearlyEqual(
    const LightProperties& lhs,
    const LightProperties& rhs);

struct TagComponent {
    std::string name;
};

// Stable import identity used by documents and compatibility projections.
// EnTT entity values are intentionally never serialized.
struct SourceNodeComponent {
    // nodeIndex addresses the flattened compatibility projection. sourceId and
    // sourceNodeIndex preserve the stable identity inside a composed document.
    int32_t nodeIndex = -1;
    std::string sourceId;
    int32_t sourceNodeIndex = -1;
};

// Identifies the runtime mount object that owns one composed scene source.
struct SceneSourceComponent {
    std::string sourceId;
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
    CameraProperties authoredProperties;
    CameraProperties properties;
};

struct LightComponent {
    int32_t lightIndex = -1;
    int32_t renderLightIndex = -1;
    LightProperties authoredProperties;
    LightProperties properties;
};

// Marks a runtime graph root, including generated root objects.
struct RootComponent {};

// Marks every object reachable from the current runtime roots.
struct ActiveSceneComponent {};

// Marks runtime-created objects that have no serialized source node.
struct GeneratedComponent {};

} // namespace metallic::scene
