#include "gpu_scene.h"

#include "cluster_lod_builder.h"
#include "mesh_loader.h"
#include "meshlet_builder.h"
#include "rhi_resource_utils.h"
#include "scene_graph.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace {

struct GeometryKey {
    int32_t meshIndex = -1;
    uint32_t primitiveGroupStart = 0;
    uint32_t primitiveGroupCount = 0;
    uint32_t meshletStart = 0;
    uint32_t meshletCount = 0;
    uint32_t indexStart = 0;
    uint32_t indexCount = 0;
    uint32_t lodRootNode = UINT32_MAX;

    bool operator==(const GeometryKey& rhs) const {
        return meshIndex == rhs.meshIndex &&
               primitiveGroupStart == rhs.primitiveGroupStart &&
               primitiveGroupCount == rhs.primitiveGroupCount &&
               meshletStart == rhs.meshletStart &&
               meshletCount == rhs.meshletCount &&
               indexStart == rhs.indexStart &&
               indexCount == rhs.indexCount &&
               lodRootNode == rhs.lodRootNode;
    }
};

struct GeometryKeyHash {
    size_t operator()(const GeometryKey& key) const noexcept {
        size_t hash = 0;
        auto hashCombine = [&hash](uint64_t value) {
            hash ^= static_cast<size_t>(value) + 0x9e3779b9u + (hash << 6u) + (hash >> 2u);
        };

        hashCombine(static_cast<uint64_t>(static_cast<uint32_t>(key.meshIndex)));
        hashCombine(key.primitiveGroupStart);
        hashCombine(key.primitiveGroupCount);
        hashCombine(key.meshletStart);
        hashCombine(key.meshletCount);
        hashCombine(key.indexStart);
        hashCombine(key.indexCount);
        hashCombine(key.lodRootNode);
        return hash;
    }
};

bool isRenderableMeshletNode(const SceneNode& node) {
    return !node.hasLight && node.meshIndex >= 0 && node.meshletCount > 0;
}

uint32_t firstMaterialIndex(const LoadedMesh& mesh, const SceneNode& node) {
    if (node.primitiveGroupCount == 0 ||
        node.primitiveGroupStart >= mesh.primitiveGroups.size()) {
        return UINT32_MAX;
    }

    return mesh.primitiveGroups[node.primitiveGroupStart].materialIndex;
}

void computeGeometryBounds(const MeshletData& meshletData,
                           const SceneNode& node,
                           float boundsCenterRadius[4]) {
    boundsCenterRadius[0] = 0.0f;
    boundsCenterRadius[1] = 0.0f;
    boundsCenterRadius[2] = 0.0f;
    boundsCenterRadius[3] = 0.0f;

    if (node.meshletCount == 0 ||
        meshletData.cpuBounds.empty() ||
        static_cast<size_t>(node.meshletStart) + node.meshletCount > meshletData.cpuBounds.size()) {
        return;
    }

    float boundsMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float boundsMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t meshletOffset = 0; meshletOffset < node.meshletCount; ++meshletOffset) {
        const GPUMeshletBounds& bounds = meshletData.cpuBounds[node.meshletStart + meshletOffset];
        const float radius = bounds.center_radius[3];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            boundsMin[axis] = std::min(boundsMin[axis], bounds.center_radius[axis] - radius);
            boundsMax[axis] = std::max(boundsMax[axis], bounds.center_radius[axis] + radius);
        }
    }

    const float center[3] = {
        0.5f * (boundsMin[0] + boundsMax[0]),
        0.5f * (boundsMin[1] + boundsMax[1]),
        0.5f * (boundsMin[2] + boundsMax[2]),
    };

    float radius = 0.0f;
    for (uint32_t meshletOffset = 0; meshletOffset < node.meshletCount; ++meshletOffset) {
        const GPUMeshletBounds& bounds = meshletData.cpuBounds[node.meshletStart + meshletOffset];
        const float dx = bounds.center_radius[0] - center[0];
        const float dy = bounds.center_radius[1] - center[1];
        const float dz = bounds.center_radius[2] - center[2];
        radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz) + bounds.center_radius[3]);
    }

    boundsCenterRadius[0] = center[0];
    boundsCenterRadius[1] = center[1];
    boundsCenterRadius[2] = center[2];
    boundsCenterRadius[3] = radius;
}

uint32_t computeVisibilityFlags(const SceneGraph& sceneGraph, const SceneNode& node) {
    uint32_t flags = 0;
    if (sceneGraph.isNodeVisible(node.id)) {
        flags |= kGpuSceneInstanceVisible;
    }
    if (node.lodRootNode != UINT32_MAX) {
        flags |= kGpuSceneInstanceHasLod;
    }
    return flags;
}

void storeMatrix(float dst[16], const float4x4& src) {
    std::memcpy(dst, &src, sizeof(float) * 16);
}

void uploadInstanceTable(GpuSceneTables& tables) {
    if (!tables.instanceBuffer.nativeHandle() || tables.instances.empty()) {
        return;
    }

    void* mappedData = rhiBufferContents(tables.instanceBuffer);
    if (!mappedData) {
        return;
    }

    std::memcpy(mappedData,
                tables.instances.data(),
                tables.instances.size() * sizeof(GPUSceneInstance));
}

} // namespace

bool buildGpuSceneTables(const RhiDevice& device,
                         const LoadedMesh& mesh,
                         const MeshletData& meshletData,
                         const ClusterLODData* clusterLodData,
                         const SceneGraph& sceneGraph,
                         GpuSceneTables& out) {
    releaseGpuSceneTables(out);
    out = GpuSceneTables{};
    out.nodeToInstance.assign(sceneGraph.nodes.size(), UINT32_MAX);

    std::unordered_map<GeometryKey, uint32_t, GeometryKeyHash> geometryMap;
    geometryMap.reserve(sceneGraph.nodes.size());

    uint32_t dispatchStart = 0;
    for (const SceneNode& node : sceneGraph.nodes) {
        if (!isRenderableMeshletNode(node)) {
            continue;
        }

        GeometryKey key{};
        key.meshIndex = node.meshIndex;
        key.primitiveGroupStart = node.primitiveGroupStart;
        key.primitiveGroupCount = node.primitiveGroupCount;
        key.meshletStart = node.meshletStart;
        key.meshletCount = node.meshletCount;
        key.indexStart = node.indexStart;
        key.indexCount = node.indexCount;
        key.lodRootNode = node.lodRootNode;

        uint32_t geometryIndex = UINT32_MAX;
        const auto geometryIt = geometryMap.find(key);
        if (geometryIt == geometryMap.end()) {
            geometryIndex = static_cast<uint32_t>(out.geometries.size());
            GPUSceneGeometry geometry{};
            geometry.meshletStart = node.meshletStart;
            geometry.meshletCount = node.meshletCount;
            geometry.primitiveGroupStart = node.primitiveGroupStart;
            geometry.primitiveGroupCount = node.primitiveGroupCount;
            geometry.indexStart = node.indexStart;
            geometry.indexCount = node.indexCount;
            geometry.materialIndex = firstMaterialIndex(mesh, node);
            geometry.lodRootNode = node.lodRootNode;
            computeGeometryBounds(meshletData, node, geometry.boundsCenterRadius);
            out.geometries.push_back(geometry);
            geometryMap.emplace(key, geometryIndex);
        } else {
            geometryIndex = geometryIt->second;
        }

        GPUSceneInstance instance{};
        const float4x4 worldMatrix = transpose(node.transform.worldMatrix);
        storeMatrix(instance.worldMatrix, worldMatrix);
        storeMatrix(instance.prevWorldMatrix, worldMatrix);
        instance.geometryIndex = geometryIndex;
        instance.dispatchStart = dispatchStart;
        instance.sceneNodeIndex = node.id;
        instance.visibilityFlags = computeVisibilityFlags(sceneGraph, node);
        out.instances.push_back(instance);

        if (node.id < out.nodeToInstance.size()) {
            out.nodeToInstance[node.id] = static_cast<uint32_t>(out.instances.size() - 1);
        }

        if ((instance.visibilityFlags & kGpuSceneInstanceVisible) != 0) {
            ++out.visibleInstanceCount;
        }

        dispatchStart += out.geometries[geometryIndex].meshletCount;
    }

    out.geometryCount = static_cast<uint32_t>(out.geometries.size());
    out.instanceCount = static_cast<uint32_t>(out.instances.size());
    out.totalMeshletDispatchCount = dispatchStart;

    // Fill packedClusterStart/Count from ClusterLODData
    if (clusterLodData && !clusterLodData->packedClusters.empty()) {
        // Build a map from primitiveGroupIndex to the LOD 0 level entry.
        // clusterLodData->allMeshlets interleaves all LOD levels for all groups,
        // so geom.meshletStart (which indexes the original LOD-0-only meshlet array)
        // cannot be used directly as packedClusterStart.
        std::unordered_map<uint32_t, const ClusterLODLevel*> lod0Map;
        for (const auto& level : clusterLodData->levels) {
            if (level.depth == 0) {
                lod0Map[level.primitiveGroupIndex] = &level;
            }
        }
        for (auto& geom : out.geometries) {
            auto it = lod0Map.find(geom.primitiveGroupStart);
            if (it != lod0Map.end()) {
                geom.packedClusterStart = it->second->meshletStart;
                geom.packedClusterCount = it->second->meshletCount;
            }
        }

        // Build CPU worklist: all LOD 0 clusters for all instances
        out.clusterVisWorklist.clear();
        out.clusterVisWorklist.reserve(out.totalMeshletDispatchCount);
        for (uint32_t instIdx = 0; instIdx < out.instanceCount; instIdx++) {
            const auto& inst = out.instances[instIdx];
            if (inst.geometryIndex >= out.geometryCount) continue;
            const auto& geom = out.geometries[inst.geometryIndex];
            for (uint32_t c = 0; c < geom.packedClusterCount; c++) {
                out.clusterVisWorklist.push_back(
                    ClusterInfo{instIdx, geom.packedClusterStart + c});
            }
        }
        out.clusterVisWorklistCount = static_cast<uint32_t>(out.clusterVisWorklist.size());
    }

    if (out.instances.empty() || out.geometries.empty()) {
        spdlog::warn("GpuScene: no renderable meshlet instances were collected from the scene graph");
        releaseGpuSceneTables(out);
        return false;
    }

    out.geometryBuffer = rhiCreateSharedBuffer(device,
                                               out.geometries.data(),
                                               out.geometries.size() * sizeof(GPUSceneGeometry),
                                               "GPU Scene Geometries");
    out.instanceBuffer = rhiCreateSharedBuffer(device,
                                               out.instances.data(),
                                               out.instances.size() * sizeof(GPUSceneInstance),
                                               "GPU Scene Instances");
    if (!out.geometryBuffer.nativeHandle() || !out.instanceBuffer.nativeHandle()) {
        spdlog::error("GpuScene: failed to create scene table buffers");
        releaseGpuSceneTables(out);
        return false;
    }

    // Upload cluster vis worklist
    if (!out.clusterVisWorklist.empty()) {
        out.clusterVisWorklistBuffer = rhiCreateSharedBuffer(
            device,
            out.clusterVisWorklist.data(),
            out.clusterVisWorklist.size() * sizeof(ClusterInfo),
            "Cluster Vis Worklist");
    }

    spdlog::info("GpuScene: built {} instances, {} geometries, {} meshlet dispatches",
                 out.instanceCount,
                 out.geometryCount,
                 out.totalMeshletDispatchCount);
    return true;
}

void updateGpuSceneTables(const SceneGraph& sceneGraph, GpuSceneTables& tables) {
    if (tables.instances.empty()) {
        tables.visibleInstanceCount = 0;
        return;
    }

    uint32_t visibleInstanceCount = 0;
    for (GPUSceneInstance& instance : tables.instances) {
        std::memcpy(instance.prevWorldMatrix, instance.worldMatrix, sizeof(instance.worldMatrix));

        if (instance.sceneNodeIndex >= sceneGraph.nodes.size()) {
            instance.visibilityFlags = 0;
            continue;
        }

        const SceneNode& node = sceneGraph.nodes[instance.sceneNodeIndex];
        const float4x4 worldMatrix = transpose(node.transform.worldMatrix);
        storeMatrix(instance.worldMatrix, worldMatrix);
        instance.visibilityFlags = computeVisibilityFlags(sceneGraph, node);
        if ((instance.visibilityFlags & kGpuSceneInstanceVisible) != 0) {
            ++visibleInstanceCount;
        }
    }

    tables.visibleInstanceCount = visibleInstanceCount;
    uploadInstanceTable(tables);
}

void releaseGpuSceneTables(GpuSceneTables& tables) {
    rhiReleaseHandle(tables.geometryBuffer);
    rhiReleaseHandle(tables.instanceBuffer);
    rhiReleaseHandle(tables.clusterVisWorklistBuffer);
    tables.geometries.clear();
    tables.instances.clear();
    tables.nodeToInstance.clear();
    tables.geometryCount = 0;
    tables.instanceCount = 0;
    tables.totalMeshletDispatchCount = 0;
    tables.visibleInstanceCount = 0;
}
