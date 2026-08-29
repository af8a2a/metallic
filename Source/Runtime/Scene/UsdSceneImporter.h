#pragma once

#include "Runtime/Scene/scene.h"

#include <filesystem>
#include <string>
#include <vector>

namespace metallic::scene::detail {

struct UsdImportedNode {
    std::string name;
    int32_t parent = kInvalidSceneIndex;
    std::vector<int32_t> children;
    int32_t meshIndex = kInvalidSceneIndex;
    int32_t cameraIndex = kInvalidSceneIndex;
    float4x4 localMatrix = float4x4::Identity();
    bool visible = true;
};

struct UsdImportedCamera {
    std::string name;
    CameraProperties properties;
};

struct UsdImportedScene {
    std::string name;
    SceneAssetInfo assetInfo;
    std::vector<int32_t> rootNodeIndices;
    std::vector<UsdImportedNode> nodes;
    std::vector<SceneMesh> meshes;
    std::vector<std::vector<RenderPrimitive>> meshPrimitives;
    std::vector<RenderImage> images;
    std::vector<RenderTexture> textures;
    std::vector<RenderMaterial> materials;
    std::vector<UsdImportedCamera> cameras;
    std::string warning;
    std::string error;
};

bool isUsdScenePath(const std::filesystem::path& path);

bool importUsdScene(
    const std::filesystem::path& path,
    UsdImportedScene& imported);

} // namespace metallic::scene::detail
