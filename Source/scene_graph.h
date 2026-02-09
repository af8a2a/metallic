#pragma once

#include <ml.h>
#include <string>
#include <vector>
#include <cstdint>

struct LoadedMesh;
struct MeshletData;

struct SceneNode {
    std::string name;
    uint32_t id = 0;
    int32_t parent = -1;
    std::vector<uint32_t> children;

    // TRS (for UI editing)
    float3 translation = float3(0.f, 0.f, 0.f);
    float4 rotation = float4(0.f, 0.f, 0.f, 1.f); // quaternion xyzw
    float3 scale = float3(1.f, 1.f, 1.f);

    // Computed matrices
    float4x4 localMatrix = float4x4::Identity();
    float4x4 worldMatrix = float4x4::Identity();
    bool useLocalMatrix = false; // preserve authored matrix until TRS is edited

    // Mesh reference (-1 = no mesh, pure transform node)
    int32_t meshIndex = -1;
    uint32_t meshletStart = 0;
    uint32_t meshletCount = 0;
    // For vertex pipeline
    uint32_t indexStart = 0;
    uint32_t indexCount = 0;

    bool visible = true;
    bool dirty = true;
};

class SceneGraph {
public:
    std::vector<SceneNode> nodes;
    std::vector<uint32_t> rootNodes;
    int32_t selectedNode = -1;

    bool buildFromGLTF(const std::string& gltfPath,
                       const LoadedMesh& mesh,
                       const MeshletData& meshletData);
    void updateTransforms();
    void markDirty(uint32_t nodeId);
    bool isNodeVisible(uint32_t nodeId) const;
};
