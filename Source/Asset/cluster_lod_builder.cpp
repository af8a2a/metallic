#include "cluster_lod_builder.h"

#include "mesh_loader.h"
#include "rhi_resource_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <imgui.h>
#include <limits>
#include <meshoptimizer.h>
#include <numeric>
#include <spdlog/spdlog.h>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {

static constexpr size_t kLodMaxVertices = 32;
static constexpr size_t kLodMinTriangles = 8;
static constexpr size_t kLodMaxTriangles = 32;
static constexpr size_t kPartitionSize = 8;
static constexpr size_t kHierarchyNodeWidth = 8;
static constexpr float kSimplifyRatio = 0.5f;
static constexpr float kSimplifyThreshold = 0.85f;
static constexpr float kClusterSplit = 2.0f;

constexpr char kClusterLodCacheMagic[8] = {'M', 'L', 'C', 'L', 'O', 'D', '0', '1'};
constexpr uint32_t kClusterLodCacheVersion = 2;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr uint32_t kInvalidIndex = UINT32_MAX;

struct Cluster {
    std::vector<unsigned int> indices;
    uint32_t meshletIndex = kInvalidIndex;
    float center[3] = {};
    float radius = 0.0f;
    float error = 0.0f;
};

struct BoundsResult {
    float center[3] = {};
    float radius = 0.0f;
    float error = 0.0f;
};

struct NodeRange {
    uint32_t offset = 0;
    uint32_t count = 0;
};

struct ClusterLODCacheHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t maxVertices = 0;
    uint32_t minTriangles = 0;
    uint32_t maxTriangles = 0;
    uint32_t partitionSize = 0;
    uint32_t hierarchyNodeWidth = 0;
    uint32_t reserved0 = 0;
    float simplifyRatio = 0.0f;
    float simplifyThreshold = 0.0f;
    float clusterSplit = 0.0f;
    float reserved1 = 0.0f;
    uint64_t meshSignature = 0;
    uint64_t meshletCount = 0;
    uint64_t meshletVertexCount = 0;
    uint64_t packedTriangleCount = 0;
    uint64_t boundsCount = 0;
    uint64_t materialCount = 0;
    uint64_t groupMeshletIndexCount = 0;
    uint64_t groupCount = 0;
    uint64_t nodeCount = 0;
    uint64_t levelCount = 0;
    uint64_t primitiveGroupRootCount = 0;
};

static_assert(std::is_trivially_copyable_v<ClusterLODCacheHeader>);
static_assert(std::is_trivially_copyable_v<GPUClusterGroup>);
static_assert(std::is_trivially_copyable_v<GPULodNode>);
static_assert(std::is_trivially_copyable_v<ClusterLODLevel>);

void releaseClusterLODHandles(ClusterLODData& data) {
    rhiReleaseHandle(data.meshletBuffer);
    rhiReleaseHandle(data.meshletVerticesBuffer);
    rhiReleaseHandle(data.meshletTrianglesBuffer);
    rhiReleaseHandle(data.boundsBuffer);
    rhiReleaseHandle(data.materialIDsBuffer);
    rhiReleaseHandle(data.groupMeshletIndicesBuffer);
    rhiReleaseHandle(data.groupBuffer);
    rhiReleaseHandle(data.nodeBuffer);
    rhiReleaseHandle(data.levelBuffer);
    rhiReleaseHandle(data.packedClusterBuffer);
    rhiReleaseHandle(data.clusterVertexDataBuffer);
    rhiReleaseHandle(data.clusterIndexDataBuffer);
}

void hashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

template <typename T>
void hashValue(uint64_t& hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    hashBytes(hash, &value, sizeof(T));
}

uint64_t computeMeshSignature(const LoadedMesh& mesh) {
    uint64_t hash = kFnvOffsetBasis;
    hashValue(hash, mesh.vertexCount);
    hashValue(hash, mesh.indexCount);
    hashValue(hash, mesh.hasBakedRootScale);
    hashValue(hash, mesh.bakedRootScale);

    if (!mesh.cpuPositions.empty()) {
        hashBytes(hash, mesh.cpuPositions.data(), mesh.cpuPositions.size() * sizeof(float));
    }
    if (!mesh.cpuIndices.empty()) {
        hashBytes(hash, mesh.cpuIndices.data(), mesh.cpuIndices.size() * sizeof(uint32_t));
    }

    const uint32_t primitiveGroupCount = static_cast<uint32_t>(mesh.primitiveGroups.size());
    hashValue(hash, primitiveGroupCount);
    for (const auto& group : mesh.primitiveGroups) {
        hashValue(hash, group.indexOffset);
        hashValue(hash, group.indexCount);
        hashValue(hash, group.vertexOffset);
        hashValue(hash, group.vertexCount);
        hashValue(hash, group.materialIndex);
    }

    return hash;
}

std::string sanitizeCacheStem(std::string stem) {
    if (stem.empty()) {
        return "scene";
    }

    for (char& ch : stem) {
        const bool isAlphaNum =
            (ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9');
        if (!isAlphaNum && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    return stem;
}

std::filesystem::path makeCacheFilePath(const std::string& sourcePath,
                                        const std::string& cacheDirectory,
                                        uint64_t meshSignature) {
    std::ostringstream fileName;
    fileName << sanitizeCacheStem(std::filesystem::path(sourcePath).stem().string())
             << "_"
             << std::hex
             << std::setw(16)
             << std::setfill('0')
             << std::nouppercase
             << meshSignature
             << ".meshletlodcache";
    return std::filesystem::path(cacheDirectory) / fileName.str();
}

template <typename T>
bool writeVector(std::ofstream& file, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (values.empty()) {
        return true;
    }

    const size_t byteSize = values.size() * sizeof(T);
    if (byteSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(byteSize));
    return static_cast<bool>(file);
}

template <typename T>
bool readVector(std::ifstream& file, uint64_t count, std::vector<T>& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count == 0) {
        out.clear();
        return true;
    }
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        return false;
    }

    out.resize(static_cast<size_t>(count));
    const size_t byteSize = out.size() * sizeof(T);
    if (byteSize > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(byteSize));
    return static_cast<bool>(file);
}

BoundsResult computeBounds(const float* positions,
                           size_t vertexCount,
                           size_t stride,
                           const std::vector<unsigned int>& indices,
                           float error) {
    meshopt_Bounds bounds = meshopt_computeClusterBounds(
        indices.data(),
        indices.size(),
        positions,
        vertexCount,
        stride);
    BoundsResult result{};
    result.center[0] = bounds.center[0];
    result.center[1] = bounds.center[1];
    result.center[2] = bounds.center[2];
    result.radius = bounds.radius;
    result.error = error;
    return result;
}

BoundsResult mergeBounds(const std::vector<Cluster>& clusters,
                         const std::vector<int>& group) {
    struct SphereData {
        float cx;
        float cy;
        float cz;
        float radius;
    };

    std::vector<SphereData> spheres(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
        const Cluster& cluster = clusters[group[i]];
        spheres[i] = {cluster.center[0], cluster.center[1], cluster.center[2], cluster.radius};
    }

    meshopt_Bounds merged = meshopt_computeSphereBounds(
        &spheres[0].cx,
        spheres.size(),
        sizeof(SphereData),
        &spheres[0].radius,
        sizeof(SphereData));

    float maxError = 0.0f;
    for (int clusterIndex : group) {
        maxError = std::max(maxError, clusters[clusterIndex].error);
    }

    BoundsResult result{};
    result.center[0] = merged.center[0];
    result.center[1] = merged.center[1];
    result.center[2] = merged.center[2];
    result.radius = merged.radius;
    result.error = maxError;
    return result;
}

std::vector<Cluster> clusterize(const float* positions,
                                size_t vertexCount,
                                size_t stride,
                                const unsigned int* indices,
                                size_t indexCount) {
    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, kLodMaxVertices, kLodMinTriangles);
    std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
    std::vector<unsigned int> meshletVertices(indexCount);
    std::vector<unsigned char> meshletTriangles(indexCount);

    size_t count = meshopt_buildMeshletsFlex(
        meshlets.data(),
        meshletVertices.data(),
        meshletTriangles.data(),
        indices,
        indexCount,
        positions,
        vertexCount,
        stride,
        kLodMaxVertices,
        kLodMinTriangles,
        kLodMaxTriangles,
        0.0f,
        kClusterSplit);
    meshlets.resize(count);

    std::vector<Cluster> result(count);
    for (size_t i = 0; i < count; ++i) {
        const meshopt_Meshlet& meshlet = meshlets[i];
        meshopt_optimizeMeshlet(
            &meshletVertices[meshlet.vertex_offset],
            &meshletTriangles[meshlet.triangle_offset],
            meshlet.triangle_count,
            meshlet.vertex_count);

        Cluster& cluster = result[i];
        cluster.indices.resize(meshlet.triangle_count * 3);
        for (size_t j = 0; j < meshlet.triangle_count * 3; ++j) {
            cluster.indices[j] =
                meshletVertices[meshlet.vertex_offset + meshletTriangles[meshlet.triangle_offset + j]];
        }
    }

    return result;
}

std::vector<std::vector<int>> partitionClusters(const std::vector<Cluster>& clusters,
                                                const std::vector<int>& pending,
                                                const std::vector<unsigned int>& positionRemap,
                                                const float* positions,
                                                size_t vertexCount,
                                                size_t stride) {
    if (pending.size() <= kPartitionSize) {
        return {pending};
    }

    std::vector<unsigned int> clusterIndices;
    std::vector<unsigned int> clusterCounts(pending.size());

    size_t totalIndexCount = 0;
    for (size_t i = 0; i < pending.size(); ++i) {
        totalIndexCount += clusters[pending[i]].indices.size();
    }
    clusterIndices.reserve(totalIndexCount);

    for (size_t i = 0; i < pending.size(); ++i) {
        const Cluster& cluster = clusters[pending[i]];
        clusterCounts[i] = static_cast<unsigned int>(cluster.indices.size());
        for (unsigned int index : cluster.indices) {
            clusterIndices.push_back(positionRemap[index]);
        }
    }

    std::vector<unsigned int> partitionIds(pending.size());
    const size_t partitionCount = meshopt_partitionClusters(
        partitionIds.data(),
        clusterIndices.data(),
        clusterIndices.size(),
        clusterCounts.data(),
        clusterCounts.size(),
        positions,
        vertexCount,
        stride,
        kPartitionSize);

    std::vector<std::vector<int>> partitions(partitionCount);
    for (size_t i = 0; i < partitionCount; ++i) {
        partitions[i].reserve(kPartitionSize + kPartitionSize / 3);
    }

    for (size_t i = 0; i < pending.size(); ++i) {
        partitions[partitionIds[i]].push_back(pending[i]);
    }

    return partitions;
}

void lockBoundary(std::vector<unsigned char>& locks,
                  const std::vector<std::vector<int>>& groups,
                  const std::vector<Cluster>& clusters,
                  const std::vector<unsigned int>& positionRemap) {
    for (size_t i = 0; i < locks.size(); ++i) {
        locks[i] &= static_cast<unsigned char>(~((1 << 0) | (1 << 7)));
    }

    for (const auto& group : groups) {
        for (int clusterIndex : group) {
            for (unsigned int vertexIndex : clusters[clusterIndex].indices) {
                const unsigned int remappedIndex = positionRemap[vertexIndex];
                locks[remappedIndex] |= locks[remappedIndex] >> 7;
            }
        }

        for (int clusterIndex : group) {
            for (unsigned int vertexIndex : clusters[clusterIndex].indices) {
                const unsigned int remappedIndex = positionRemap[vertexIndex];
                locks[remappedIndex] |= 1 << 7;
            }
        }
    }

    for (size_t i = 0; i < locks.size(); ++i) {
        const unsigned int remappedIndex = positionRemap[i];
        locks[i] = locks[remappedIndex] & 1;
    }
}

void orderNodesSpatially(const GPULodNode* nodes,
                         uint32_t nodeCount,
                         std::vector<uint32_t>& order) {
    order.resize(nodeCount);
    std::iota(order.begin(), order.end(), 0u);
    if (nodeCount <= 1) {
        return;
    }

    float boundsMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float boundsMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            boundsMin[axis] = std::min(boundsMin[axis], nodes[nodeIndex].center[axis]);
            boundsMax[axis] = std::max(boundsMax[axis], nodes[nodeIndex].center[axis]);
        }
    }

    uint32_t dominantAxis = 0;
    float dominantExtent = boundsMax[0] - boundsMin[0];
    for (uint32_t axis = 1; axis < 3; ++axis) {
        const float extent = boundsMax[axis] - boundsMin[axis];
        if (extent > dominantExtent) {
            dominantAxis = axis;
            dominantExtent = extent;
        }
    }

    std::sort(order.begin(),
              order.end(),
              [&](uint32_t lhs, uint32_t rhs) {
                  return nodes[lhs].center[dominantAxis] < nodes[rhs].center[dominantAxis];
              });
}

uint32_t appendBaseMeshlet(const MeshletData& meshletData,
                           uint32_t sourceMeshletIndex,
                           ClusterLODData& out) {
    const GPUMeshlet& sourceMeshlet = meshletData.cpuMeshlets[sourceMeshletIndex];

    GPUMeshlet copiedMeshlet{};
    copiedMeshlet.vertex_offset = static_cast<uint32_t>(out.allMeshletVertices.size());
    copiedMeshlet.triangle_offset = static_cast<uint32_t>(out.allPackedTriangles.size());
    copiedMeshlet.vertex_count = sourceMeshlet.vertex_count;
    copiedMeshlet.triangle_count = sourceMeshlet.triangle_count;

    const size_t vertexStart = sourceMeshlet.vertex_offset;
    const size_t vertexEnd = vertexStart + sourceMeshlet.vertex_count;
    out.allMeshletVertices.insert(out.allMeshletVertices.end(),
                                  meshletData.cpuMeshletVertices.begin() + static_cast<std::ptrdiff_t>(vertexStart),
                                  meshletData.cpuMeshletVertices.begin() + static_cast<std::ptrdiff_t>(vertexEnd));

    const size_t rawTriangleStart = static_cast<size_t>(sourceMeshlet.triangle_offset) * 3;
    for (uint32_t triangleIndex = 0; triangleIndex < sourceMeshlet.triangle_count; ++triangleIndex) {
        const size_t rawIndex = rawTriangleStart + static_cast<size_t>(triangleIndex) * 3;
        const uint32_t packedTriangle =
            uint32_t(meshletData.cpuMeshletTriangles[rawIndex + 0]) |
            (uint32_t(meshletData.cpuMeshletTriangles[rawIndex + 1]) << 8) |
            (uint32_t(meshletData.cpuMeshletTriangles[rawIndex + 2]) << 16);
        out.allPackedTriangles.push_back(packedTriangle);
    }

    out.allMeshlets.push_back(copiedMeshlet);
    out.allBounds.push_back(meshletData.cpuBounds[sourceMeshletIndex]);
    out.allMaterialIDs.push_back(meshletData.cpuMaterialIDs[sourceMeshletIndex]);
    return static_cast<uint32_t>(out.allMeshlets.size() - 1);
}

Cluster makeBaseCluster(const MeshletData& meshletData,
                        uint32_t sourceMeshletIndex,
                        uint32_t localMeshletIndex) {
    const GPUMeshlet& sourceMeshlet = meshletData.cpuMeshlets[sourceMeshletIndex];

    Cluster cluster{};
    cluster.meshletIndex = localMeshletIndex;
    cluster.indices.resize(sourceMeshlet.triangle_count * 3);

    const size_t rawTriangleStart = static_cast<size_t>(sourceMeshlet.triangle_offset) * 3;
    for (uint32_t triangleIndex = 0; triangleIndex < sourceMeshlet.triangle_count; ++triangleIndex) {
        for (uint32_t vertexIndex = 0; vertexIndex < 3; ++vertexIndex) {
            const size_t rawIndex = rawTriangleStart + static_cast<size_t>(triangleIndex) * 3 + vertexIndex;
            const unsigned char localVertex = meshletData.cpuMeshletTriangles[rawIndex];
            cluster.indices[triangleIndex * 3 + vertexIndex] =
                meshletData.cpuMeshletVertices[sourceMeshlet.vertex_offset + localVertex];
        }
    }

    const GPUMeshletBounds& bounds = meshletData.cpuBounds[sourceMeshletIndex];
    cluster.center[0] = bounds.center_radius[0];
    cluster.center[1] = bounds.center_radius[1];
    cluster.center[2] = bounds.center_radius[2];
    cluster.radius = bounds.center_radius[3];
    cluster.error = 0.0f;
    return cluster;
}

uint32_t emitSimplifiedMeshlet(const Cluster& cluster,
                               uint32_t materialID,
                               ClusterLODData& out,
                               const float* positions,
                               size_t vertexCount,
                               size_t stride) {
    std::vector<unsigned int> localVertices;
    std::vector<unsigned char> localTriangles;
    localVertices.reserve(cluster.indices.size());
    localTriangles.reserve(cluster.indices.size());

    for (unsigned int globalVertexIndex : cluster.indices) {
        unsigned char localVertexIndex = 0;
        bool found = false;
        for (size_t i = 0; i < localVertices.size(); ++i) {
            if (localVertices[i] == globalVertexIndex) {
                localVertexIndex = static_cast<unsigned char>(i);
                found = true;
                break;
            }
        }

        if (!found) {
            localVertexIndex = static_cast<unsigned char>(localVertices.size());
            localVertices.push_back(globalVertexIndex);
        }

        localTriangles.push_back(localVertexIndex);
    }

    meshopt_Bounds bounds = meshopt_computeClusterBounds(
        cluster.indices.data(),
        cluster.indices.size(),
        positions,
        vertexCount,
        stride);

    GPUMeshletBounds gpuBounds{};
    gpuBounds.center_radius[0] = bounds.center[0];
    gpuBounds.center_radius[1] = bounds.center[1];
    gpuBounds.center_radius[2] = bounds.center[2];
    gpuBounds.center_radius[3] = bounds.radius;
    gpuBounds.cone_apex_pad[0] = bounds.cone_apex[0];
    gpuBounds.cone_apex_pad[1] = bounds.cone_apex[1];
    gpuBounds.cone_apex_pad[2] = bounds.cone_apex[2];
    gpuBounds.cone_apex_pad[3] = 0.0f;
    gpuBounds.cone_axis_cutoff[0] = bounds.cone_axis[0];
    gpuBounds.cone_axis_cutoff[1] = bounds.cone_axis[1];
    gpuBounds.cone_axis_cutoff[2] = bounds.cone_axis[2];
    gpuBounds.cone_axis_cutoff[3] = bounds.cone_cutoff;

    GPUMeshlet gpuMeshlet{};
    gpuMeshlet.vertex_offset = static_cast<uint32_t>(out.allMeshletVertices.size());
    gpuMeshlet.triangle_offset = static_cast<uint32_t>(out.allPackedTriangles.size());
    gpuMeshlet.vertex_count = static_cast<uint32_t>(localVertices.size());
    gpuMeshlet.triangle_count = static_cast<uint32_t>(cluster.indices.size() / 3);

    out.allMeshletVertices.insert(out.allMeshletVertices.end(), localVertices.begin(), localVertices.end());
    for (uint32_t triangleIndex = 0; triangleIndex < gpuMeshlet.triangle_count; ++triangleIndex) {
        const uint32_t v0 = localTriangles[triangleIndex * 3 + 0];
        const uint32_t v1 = localTriangles[triangleIndex * 3 + 1];
        const uint32_t v2 = localTriangles[triangleIndex * 3 + 2];
        out.allPackedTriangles.push_back(v0 | (v1 << 8) | (v2 << 16));
    }

    out.allMeshlets.push_back(gpuMeshlet);
    out.allBounds.push_back(gpuBounds);
    out.allMaterialIDs.push_back(materialID);
    return static_cast<uint32_t>(out.allMeshlets.size() - 1);
}

uint32_t buildLocalHierarchy(ClusterLODData& data) {
    if (data.levels.empty()) {
        return kInvalidIndex;
    }

    const uint32_t lodLevelCount = static_cast<uint32_t>(data.levels.size());
    std::vector<NodeRange> levelNodeRanges(lodLevelCount);

    uint32_t nodeOffset = 1 + lodLevelCount;
    for (uint32_t levelIndex = 0; levelIndex < lodLevelCount; ++levelIndex) {
        const ClusterLODLevel& level = data.levels[levelIndex];
        uint32_t nodeCount = level.groupCount;
        uint32_t iterationCount = nodeCount;
        while (iterationCount > 1) {
            iterationCount = (iterationCount + static_cast<uint32_t>(kHierarchyNodeWidth) - 1) /
                             static_cast<uint32_t>(kHierarchyNodeWidth);
            nodeCount += iterationCount;
        }
        if (nodeCount > 0) {
            nodeCount -= 1;
        }

        levelNodeRanges[levelIndex].offset = nodeOffset;
        levelNodeRanges[levelIndex].count = nodeCount;
        nodeOffset += nodeCount;
    }

    data.nodes.clear();
    data.nodes.resize(nodeOffset);

    for (uint32_t levelIndex = 0; levelIndex < lodLevelCount; ++levelIndex) {
        ClusterLODLevel& level = data.levels[levelIndex];
        const NodeRange& levelNodeRange = levelNodeRanges[levelIndex];

        const uint32_t groupCount = level.groupCount;
        const uint32_t groupOffset = level.groupStart;
        if (groupCount == 0) {
            level.rootNode = kInvalidIndex;
            continue;
        }

        uint32_t currentNodeOffset = levelNodeRange.offset;
        uint32_t lastNodeOffset = currentNodeOffset;

        for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            const uint32_t globalGroupIndex = groupOffset + groupIndex;
            const GPUClusterGroup& group = data.groups[globalGroupIndex];

            GPULodNode& node = (groupCount == 1)
                ? data.nodes[1 + levelIndex]
                : data.nodes[currentNodeOffset++];
            node.center[0] = group.center[0];
            node.center[1] = group.center[1];
            node.center[2] = group.center[2];
            node.radius = group.radius;
            node.maxError = group.parentError;
            node.childOffset = globalGroupIndex;
            node.childCount = 1;
            node.isLeaf = 1;
        }

        if (groupCount == 1) {
            level.rootNode = 1 + levelIndex;
            continue;
        }

        uint32_t iterationCount = groupCount;
        std::vector<uint32_t> partitionedIndices;
        std::vector<GPULodNode> orderedNodes;

        while (iterationCount > 1) {
            const uint32_t lastNodeCount = iterationCount;
            GPULodNode* lastNodes = data.nodes.data() + lastNodeOffset;

            orderNodesSpatially(lastNodes, lastNodeCount, partitionedIndices);

            orderedNodes.assign(lastNodes, lastNodes + lastNodeCount);
            for (uint32_t nodeIndex = 0; nodeIndex < lastNodeCount; ++nodeIndex) {
                lastNodes[nodeIndex] = orderedNodes[partitionedIndices[nodeIndex]];
            }

            iterationCount = (lastNodeCount + static_cast<uint32_t>(kHierarchyNodeWidth) - 1) /
                             static_cast<uint32_t>(kHierarchyNodeWidth);

            GPULodNode* newNodes = (iterationCount == 1)
                ? &data.nodes[1 + levelIndex]
                : data.nodes.data() + currentNodeOffset;

            for (uint32_t nodeIndex = 0; nodeIndex < iterationCount; ++nodeIndex) {
                GPULodNode& node = newNodes[nodeIndex];
                GPULodNode* children = lastNodes + nodeIndex * static_cast<uint32_t>(kHierarchyNodeWidth);
                const uint32_t childCount =
                    std::min((nodeIndex + 1) * static_cast<uint32_t>(kHierarchyNodeWidth), lastNodeCount) -
                    nodeIndex * static_cast<uint32_t>(kHierarchyNodeWidth);

                node = {};
                node.childOffset = lastNodeOffset + nodeIndex * static_cast<uint32_t>(kHierarchyNodeWidth);
                node.childCount = childCount;
                node.isLeaf = 0;
                node.maxError = 0.0f;
                for (uint32_t childIndex = 0; childIndex < childCount; ++childIndex) {
                    node.maxError = std::max(node.maxError, children[childIndex].maxError);
                }

                meshopt_Bounds merged = meshopt_computeSphereBounds(
                    &children[0].center[0],
                    childCount,
                    sizeof(GPULodNode),
                    &children[0].radius,
                    sizeof(GPULodNode));
                node.center[0] = merged.center[0];
                node.center[1] = merged.center[1];
                node.center[2] = merged.center[2];
                node.radius = merged.radius;
            }

            if (iterationCount == 1) {
                lastNodeOffset = 1 + levelIndex;
            } else {
                lastNodeOffset = currentNodeOffset;
                currentNodeOffset += iterationCount;
            }
        }

        level.rootNode = 1 + levelIndex;
    }

    if (lodLevelCount == 1) {
        const GPULodNode& child = data.nodes[1];
        GPULodNode& root = data.nodes[0];
        root = {};
        root.center[0] = child.center[0];
        root.center[1] = child.center[1];
        root.center[2] = child.center[2];
        root.radius = child.radius;
        root.maxError = child.maxError;
        root.childOffset = 1;
        root.childCount = 1;
        root.isLeaf = 0;
        return 0;
    }

    meshopt_Bounds merged = meshopt_computeSphereBounds(
        &data.nodes[1].center[0],
        lodLevelCount,
        sizeof(GPULodNode),
        &data.nodes[1].radius,
        sizeof(GPULodNode));

    GPULodNode& root = data.nodes[0];
    root = {};
    root.center[0] = merged.center[0];
    root.center[1] = merged.center[1];
    root.center[2] = merged.center[2];
    root.radius = merged.radius;
    root.childOffset = 1;
    root.childCount = lodLevelCount;
    root.isLeaf = 0;
    root.maxError = 0.0f;
    for (uint32_t levelIndex = 0; levelIndex < lodLevelCount; ++levelIndex) {
        root.maxError = std::max(root.maxError, data.nodes[1 + levelIndex].maxError);
    }

    return 0;
}

void assignLod0PrimitiveGroupRoots(ClusterLODData& data,
                                   uint32_t primitiveGroupCount) {
    data.primitiveGroupLodRoots.assign(primitiveGroupCount, kInvalidIndex);
    if (primitiveGroupCount == 1u) {
        for (const ClusterLODLevel& level : data.levels) {
            if (level.depth == 0u && level.rootNode != kInvalidIndex) {
                data.primitiveGroupLodRoots[0] = level.rootNode;
                return;
            }
        }
        return;
    }

    for (const ClusterLODLevel& level : data.levels) {
        if (level.depth != 0u ||
            level.primitiveGroupIndex >= primitiveGroupCount ||
            level.rootNode == kInvalidIndex) {
            continue;
        }
        data.primitiveGroupLodRoots[level.primitiveGroupIndex] = level.rootNode;
    }
}

bool buildPrimitiveGroupClusterLOD(const LoadedMesh& mesh,
                                   const MeshletData& meshletData,
                                   uint32_t primitiveGroupIndex,
                                   uint32_t baseMeshletStart,
                                   uint32_t baseMeshletCount,
                                   const std::vector<unsigned int>& positionRemap,
                                   ClusterLODData& out) {
    out = ClusterLODData{};
    if (baseMeshletCount == 0) {
        return true;
    }

    const float* allPositions = mesh.cpuPositions.data();
    constexpr size_t kPositionStride = sizeof(float) * 3;

    std::vector<Cluster> clusters;
    clusters.reserve(baseMeshletCount * 2);
    for (uint32_t meshletIndex = 0; meshletIndex < baseMeshletCount; ++meshletIndex) {
        const uint32_t sourceMeshletIndex = baseMeshletStart + meshletIndex;
        const uint32_t copiedMeshletIndex = appendBaseMeshlet(meshletData, sourceMeshletIndex, out);
        clusters.push_back(makeBaseCluster(meshletData, sourceMeshletIndex, copiedMeshletIndex));
    }

    std::vector<int> pending(baseMeshletCount);
    std::iota(pending.begin(), pending.end(), 0);

    std::vector<unsigned char> locks(mesh.vertexCount, 0);
    uint32_t currentLevelMeshletStart = 0;
    uint32_t currentLevelMeshletCount = baseMeshletCount;
    uint32_t depth = 0;
    const uint32_t materialID =
        (!mesh.primitiveGroups.empty() && primitiveGroupIndex < mesh.primitiveGroups.size())
            ? mesh.primitiveGroups[primitiveGroupIndex].materialIndex
            : 0u;

    while (pending.size() > 1) {
        ClusterLODLevel level{};
        level.primitiveGroupIndex = primitiveGroupIndex;
        level.depth = depth;
        level.meshletStart = currentLevelMeshletStart;
        level.meshletCount = currentLevelMeshletCount;
        level.groupStart = static_cast<uint32_t>(out.groups.size());
        level.rootNode = kInvalidIndex;

        const auto groups = partitionClusters(
            clusters,
            pending,
            positionRemap,
            allPositions,
            mesh.vertexCount,
            kPositionStride);
        lockBoundary(locks, groups, clusters, positionRemap);

        std::vector<int> newPending;
        newPending.reserve(pending.size());
        const uint32_t nextLevelMeshletStart = static_cast<uint32_t>(out.allMeshlets.size());

        for (const auto& group : groups) {
            std::vector<unsigned int> mergedIndices;
            for (int clusterIndex : group) {
                const auto& clusterIndices = clusters[clusterIndex].indices;
                mergedIndices.insert(mergedIndices.end(), clusterIndices.begin(), clusterIndices.end());
            }

            BoundsResult groupBounds = mergeBounds(clusters, group);

            GPUClusterGroup gpuGroup{};
            gpuGroup.center[0] = groupBounds.center[0];
            gpuGroup.center[1] = groupBounds.center[1];
            gpuGroup.center[2] = groupBounds.center[2];
            gpuGroup.radius = groupBounds.radius;
            gpuGroup.error = groupBounds.error;
            gpuGroup.parentError = FLT_MAX;
            gpuGroup.clusterStart = static_cast<uint32_t>(out.groupMeshletIndices.size());
            gpuGroup.clusterCount = static_cast<uint32_t>(group.size());
            for (int clusterIndex : group) {
                out.groupMeshletIndices.push_back(clusters[clusterIndex].meshletIndex);
            }

            const size_t targetIndexCount = size_t((mergedIndices.size() / 3) * kSimplifyRatio) * 3;
            bool isTerminal = mergedIndices.size() < 6 || targetIndexCount < 3;
            std::vector<unsigned int> simplifiedIndices;
            float simplifyError = 0.0f;

            if (!isTerminal) {
                simplifiedIndices.resize(mergedIndices.size());
                const size_t simplifiedSize = meshopt_simplify(
                    simplifiedIndices.data(),
                    mergedIndices.data(),
                    mergedIndices.size(),
                    allPositions,
                    mesh.vertexCount,
                    kPositionStride,
                    targetIndexCount,
                    FLT_MAX,
                    meshopt_SimplifySparse |
                        meshopt_SimplifyLockBorder |
                        meshopt_SimplifyErrorAbsolute,
                    &simplifyError);
                simplifiedIndices.resize(simplifiedSize);

                if (simplifiedIndices.size() > size_t(mergedIndices.size() * kSimplifyThreshold) ||
                    simplifiedIndices.size() < 3) {
                    isTerminal = true;
                }
            }

            if (isTerminal) {
                out.groups.push_back(gpuGroup);
                continue;
            }

            const float nextError = std::max(groupBounds.error, simplifyError);
            gpuGroup.parentError = nextError;
            out.groups.push_back(gpuGroup);

            for (int clusterIndex : group) {
                clusters[clusterIndex].indices.clear();
            }

            auto newClusters = clusterize(
                allPositions,
                mesh.vertexCount,
                kPositionStride,
                simplifiedIndices.data(),
                simplifiedIndices.size());

            for (auto& cluster : newClusters) {
                cluster.center[0] = groupBounds.center[0];
                cluster.center[1] = groupBounds.center[1];
                cluster.center[2] = groupBounds.center[2];
                cluster.radius = groupBounds.radius;
                cluster.error = nextError;
                cluster.meshletIndex = emitSimplifiedMeshlet(
                    cluster,
                    materialID,
                    out,
                    allPositions,
                    mesh.vertexCount,
                    kPositionStride);

                const int newClusterIndex = static_cast<int>(clusters.size());
                clusters.push_back(std::move(cluster));
                newPending.push_back(newClusterIndex);
            }
        }

        level.groupCount = static_cast<uint32_t>(out.groups.size()) - level.groupStart;
        if (level.groupCount > 0) {
            out.levels.push_back(level);
        }

        pending = std::move(newPending);
        currentLevelMeshletStart = nextLevelMeshletStart;
        currentLevelMeshletCount =
            static_cast<uint32_t>(out.allMeshlets.size()) - nextLevelMeshletStart;
        ++depth;

        if (pending.empty()) {
            break;
        }
    }

    if (!pending.empty()) {
        ClusterLODLevel level{};
        level.primitiveGroupIndex = primitiveGroupIndex;
        level.depth = depth;
        level.meshletStart = currentLevelMeshletStart;
        level.meshletCount = currentLevelMeshletCount;
        level.groupStart = static_cast<uint32_t>(out.groups.size());
        level.rootNode = kInvalidIndex;

        for (int clusterIndex : pending) {
            const Cluster& cluster = clusters[clusterIndex];
            GPUClusterGroup gpuGroup{};
            gpuGroup.center[0] = cluster.center[0];
            gpuGroup.center[1] = cluster.center[1];
            gpuGroup.center[2] = cluster.center[2];
            gpuGroup.radius = cluster.radius;
            gpuGroup.error = cluster.error;
            gpuGroup.parentError = FLT_MAX;
            gpuGroup.clusterStart = static_cast<uint32_t>(out.groupMeshletIndices.size());
            gpuGroup.clusterCount = 1;
            out.groupMeshletIndices.push_back(cluster.meshletIndex);
            out.groups.push_back(gpuGroup);
        }

        level.groupCount = static_cast<uint32_t>(out.groups.size()) - level.groupStart;
        if (level.groupCount > 0) {
            out.levels.push_back(level);
        }
    }

    if (out.levels.empty() || out.groups.empty() || out.allMeshlets.empty()) {
        return false;
    }

    buildLocalHierarchy(out);
    assignLod0PrimitiveGroupRoots(out, 1u);
    out.totalMeshletCount = static_cast<uint32_t>(out.allMeshlets.size());
    out.totalGroupCount = static_cast<uint32_t>(out.groups.size());
    out.totalNodeCount = static_cast<uint32_t>(out.nodes.size());
    out.lodLevelCount = static_cast<uint32_t>(out.levels.size());
    return true;
}

void appendClusterLOD(const ClusterLODData& local,
                      uint32_t primitiveGroupIndex,
                      ClusterLODData& out) {
    const uint32_t meshletOffset = static_cast<uint32_t>(out.allMeshlets.size());
    const uint32_t vertexOffset = static_cast<uint32_t>(out.allMeshletVertices.size());
    const uint32_t triangleOffset = static_cast<uint32_t>(out.allPackedTriangles.size());
    const uint32_t groupMeshletIndexOffset = static_cast<uint32_t>(out.groupMeshletIndices.size());
    const uint32_t groupOffset = static_cast<uint32_t>(out.groups.size());
    const uint32_t nodeOffset = static_cast<uint32_t>(out.nodes.size());

    out.allMeshletVertices.insert(
        out.allMeshletVertices.end(),
        local.allMeshletVertices.begin(),
        local.allMeshletVertices.end());
    out.allPackedTriangles.insert(
        out.allPackedTriangles.end(),
        local.allPackedTriangles.begin(),
        local.allPackedTriangles.end());
    out.allBounds.insert(out.allBounds.end(), local.allBounds.begin(), local.allBounds.end());
    out.allMaterialIDs.insert(
        out.allMaterialIDs.end(),
        local.allMaterialIDs.begin(),
        local.allMaterialIDs.end());

    out.allMeshlets.reserve(out.allMeshlets.size() + local.allMeshlets.size());
    for (const GPUMeshlet& localMeshlet : local.allMeshlets) {
        GPUMeshlet meshlet = localMeshlet;
        meshlet.vertex_offset += vertexOffset;
        meshlet.triangle_offset += triangleOffset;
        out.allMeshlets.push_back(meshlet);
    }

    out.groupMeshletIndices.reserve(out.groupMeshletIndices.size() + local.groupMeshletIndices.size());
    for (uint32_t meshletIndex : local.groupMeshletIndices) {
        out.groupMeshletIndices.push_back(meshletIndex + meshletOffset);
    }

    out.groups.reserve(out.groups.size() + local.groups.size());
    for (const GPUClusterGroup& localGroup : local.groups) {
        GPUClusterGroup group = localGroup;
        group.clusterStart += groupMeshletIndexOffset;
        out.groups.push_back(group);
    }

    out.nodes.reserve(out.nodes.size() + local.nodes.size());
    for (const GPULodNode& localNode : local.nodes) {
        GPULodNode node = localNode;
        if (node.isLeaf != 0) {
            node.childOffset += groupOffset;
        } else {
            node.childOffset += nodeOffset;
        }
        out.nodes.push_back(node);
    }

    out.levels.reserve(out.levels.size() + local.levels.size());
    for (const ClusterLODLevel& localLevel : local.levels) {
        ClusterLODLevel level = localLevel;
        level.primitiveGroupIndex = primitiveGroupIndex;
        level.meshletStart += meshletOffset;
        level.groupStart += groupOffset;
        if (level.rootNode != kInvalidIndex) {
            level.rootNode += nodeOffset;
        }
        out.levels.push_back(level);
    }

    if (!local.primitiveGroupLodRoots.empty() &&
        primitiveGroupIndex < out.primitiveGroupLodRoots.size() &&
        local.primitiveGroupLodRoots[0] != kInvalidIndex) {
        out.primitiveGroupLodRoots[primitiveGroupIndex] = local.primitiveGroupLodRoots[0] + nodeOffset;
    }
}

bool validateClusterLodPayload(const ClusterLODData& data,
                               uint32_t expectedPrimitiveGroupCount) {
    if (data.allMeshlets.empty()) {
        spdlog::error("ClusterLOD payload is empty");
        return false;
    }
    if (data.allBounds.size() != data.allMeshlets.size()) {
        spdlog::error("ClusterLOD bounds count {} does not match meshlet count {}",
                      data.allBounds.size(),
                      data.allMeshlets.size());
        return false;
    }
    if (data.allMaterialIDs.size() != data.allMeshlets.size()) {
        spdlog::error("ClusterLOD material count {} does not match meshlet count {}",
                      data.allMaterialIDs.size(),
                      data.allMeshlets.size());
        return false;
    }
    if (data.primitiveGroupLodRoots.size() != expectedPrimitiveGroupCount) {
        spdlog::error("ClusterLOD primitive group root count {} does not match expected {}",
                      data.primitiveGroupLodRoots.size(),
                      expectedPrimitiveGroupCount);
        return false;
    }

    for (const GPUMeshlet& meshlet : data.allMeshlets) {
        if (static_cast<size_t>(meshlet.vertex_offset) + meshlet.vertex_count > data.allMeshletVertices.size()) {
            spdlog::error("ClusterLOD meshlet vertex range [{}, {}) is out of bounds {}",
                          meshlet.vertex_offset,
                          static_cast<size_t>(meshlet.vertex_offset) + meshlet.vertex_count,
                          data.allMeshletVertices.size());
            return false;
        }
        if (static_cast<size_t>(meshlet.triangle_offset) + meshlet.triangle_count > data.allPackedTriangles.size()) {
            spdlog::error("ClusterLOD meshlet triangle range [{}, {}) is out of bounds {}",
                          meshlet.triangle_offset,
                          static_cast<size_t>(meshlet.triangle_offset) + meshlet.triangle_count,
                          data.allPackedTriangles.size());
            return false;
        }
    }

    for (const GPUClusterGroup& group : data.groups) {
        if (static_cast<size_t>(group.clusterStart) + group.clusterCount > data.groupMeshletIndices.size()) {
            spdlog::error("ClusterLOD group child range [{}, {}) is out of bounds {}",
                          group.clusterStart,
                          static_cast<size_t>(group.clusterStart) + group.clusterCount,
                          data.groupMeshletIndices.size());
            return false;
        }
        for (uint32_t childIndex = 0; childIndex < group.clusterCount; ++childIndex) {
            const uint32_t meshletIndex = data.groupMeshletIndices[group.clusterStart + childIndex];
            if (meshletIndex >= data.allMeshlets.size()) {
                spdlog::error("ClusterLOD group references invalid meshlet {}", meshletIndex);
                return false;
            }
        }
    }

    for (const ClusterLODLevel& level : data.levels) {
        if (level.primitiveGroupIndex >= expectedPrimitiveGroupCount) {
            spdlog::error("ClusterLOD level references invalid primitive group {}", level.primitiveGroupIndex);
            return false;
        }
        if (static_cast<size_t>(level.groupStart) + level.groupCount > data.groups.size()) {
            spdlog::error("ClusterLOD level group range [{}, {}) is out of bounds {}",
                          level.groupStart,
                          static_cast<size_t>(level.groupStart) + level.groupCount,
                          data.groups.size());
            return false;
        }
        if (level.meshletCount > 0 &&
            static_cast<size_t>(level.meshletStart) + level.meshletCount > data.allMeshlets.size()) {
            spdlog::error("ClusterLOD level meshlet range [{}, {}) is out of bounds {}",
                          level.meshletStart,
                          static_cast<size_t>(level.meshletStart) + level.meshletCount,
                          data.allMeshlets.size());
            return false;
        }
        if (level.rootNode != kInvalidIndex && level.rootNode >= data.nodes.size()) {
            spdlog::error("ClusterLOD level references invalid root node {}", level.rootNode);
            return false;
        }
    }

    for (const GPULodNode& node : data.nodes) {
        if (node.childCount == 0) {
            spdlog::error("ClusterLOD node has zero children");
            return false;
        }

        if (node.isLeaf != 0) {
            if (static_cast<size_t>(node.childOffset) + node.childCount > data.groups.size()) {
                spdlog::error("ClusterLOD leaf node child range [{}, {}) is out of bounds {}",
                              node.childOffset,
                              static_cast<size_t>(node.childOffset) + node.childCount,
                              data.groups.size());
                return false;
            }
        } else if (static_cast<size_t>(node.childOffset) + node.childCount > data.nodes.size()) {
            spdlog::error("ClusterLOD interior node child range [{}, {}) is out of bounds {}",
                          node.childOffset,
                          static_cast<size_t>(node.childOffset) + node.childCount,
                          data.nodes.size());
            return false;
        }
    }

    for (uint32_t rootNode : data.primitiveGroupLodRoots) {
        if (rootNode != kInvalidIndex && rootNode >= data.nodes.size()) {
            spdlog::error("ClusterLOD primitive group root {} is out of bounds {}", rootNode, data.nodes.size());
            return false;
        }
    }

    return true;
}

void buildPackedClusterData(const LoadedMesh& mesh, ClusterLODData& data) {
    const size_t meshletCount = data.allMeshlets.size();
    data.packedClusters.resize(meshletCount);
    data.clusterVertexData.clear();
    data.clusterIndexData.clear();

    // Reserve approximate space
    data.clusterVertexData.reserve(meshletCount * 32 * 12);
    data.clusterIndexData.reserve(meshletCount * 32 * 3);

    // Build a LOD level lookup: meshlet index → LOD level
    std::vector<uint8_t> meshletLodLevel(meshletCount, 0);
    for (size_t li = 0; li < data.levels.size(); li++) {
        const auto& level = data.levels[li];
        for (uint32_t mi = 0; mi < level.meshletCount; mi++) {
            uint32_t idx = level.meshletStart + mi;
            if (idx < meshletCount)
                meshletLodLevel[idx] = static_cast<uint8_t>(li);
        }
    }

    for (size_t i = 0; i < meshletCount; i++) {
        const GPUMeshlet& m = data.allMeshlets[i];
        PackedCluster& pc = data.packedClusters[i];

        pc.triCountM1 = static_cast<uint8_t>(m.triangle_count > 0 ? m.triangle_count - 1 : 0);
        pc.vtxCountM1 = static_cast<uint8_t>(m.vertex_count > 0 ? m.vertex_count - 1 : 0);
        pc.lodLevel = meshletLodLevel[i];
        pc.groupChildIndex = 0;
        pc.attributeBits = 0;
        pc.localMaterialID = (i < data.allMaterialIDs.size())
            ? static_cast<uint8_t>(data.allMaterialIDs[i] & 0xFF) : 0;
        pc.reserved = 0;

        // Vertex data: copy positions as contiguous float3 array
        pc.vertexByteOffset = static_cast<uint32_t>(data.clusterVertexData.size());
        for (uint32_t v = 0; v < m.vertex_count; v++) {
            uint32_t globalVertexIndex = data.allMeshletVertices[m.vertex_offset + v];
            const float* pos = &mesh.cpuPositions[globalVertexIndex * 3];
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pos);
            data.clusterVertexData.insert(data.clusterVertexData.end(), bytes, bytes + 12);
        }

        // Index data: copy raw uint8 local triangle indices
        pc.indexByteOffset = static_cast<uint32_t>(data.clusterIndexData.size());
        for (uint32_t t = 0; t < m.triangle_count; t++) {
            uint32_t packed = data.allPackedTriangles[m.triangle_offset + t];
            data.clusterIndexData.push_back(static_cast<uint8_t>(packed & 0xFF));
            data.clusterIndexData.push_back(static_cast<uint8_t>((packed >> 8) & 0xFF));
            data.clusterIndexData.push_back(static_cast<uint8_t>((packed >> 16) & 0xFF));
        }
    }

    spdlog::info("Packed {} clusters: vertex data {:.1f} KB, index data {:.1f} KB",
                 meshletCount,
                 data.clusterVertexData.size() / 1024.0,
                 data.clusterIndexData.size() / 1024.0);
}

bool uploadClusterLodBuffers(const RhiDevice& device, ClusterLODData& data) {
    if (data.allMeshlets.empty()) {
        return false;
    }

    const uint32_t expectedPrimitiveGroupCount = data.primitiveGroupLodRoots.empty()
        ? 1u
        : static_cast<uint32_t>(data.primitiveGroupLodRoots.size());
    if (!validateClusterLodPayload(data, expectedPrimitiveGroupCount)) {
        return false;
    }

    data.meshletBuffer = rhiCreateSharedBuffer(
        device,
        data.allMeshlets.data(),
        data.allMeshlets.size() * sizeof(GPUMeshlet),
        "LOD Meshlets");
    data.meshletVerticesBuffer = rhiCreateSharedBuffer(
        device,
        data.allMeshletVertices.data(),
        data.allMeshletVertices.size() * sizeof(unsigned int),
        "LOD Meshlet Vertices");
    data.meshletTrianglesBuffer = rhiCreateSharedBuffer(
        device,
        data.allPackedTriangles.data(),
        data.allPackedTriangles.size() * sizeof(uint32_t),
        "LOD Meshlet Triangles");
    data.boundsBuffer = rhiCreateSharedBuffer(
        device,
        data.allBounds.data(),
        data.allBounds.size() * sizeof(GPUMeshletBounds),
        "LOD Meshlet Bounds");
    data.materialIDsBuffer = rhiCreateSharedBuffer(
        device,
        data.allMaterialIDs.data(),
        data.allMaterialIDs.size() * sizeof(uint32_t),
        "LOD Meshlet Material IDs");

    if (!data.groupMeshletIndices.empty()) {
        data.groupMeshletIndicesBuffer = rhiCreateSharedBuffer(
            device,
            data.groupMeshletIndices.data(),
            data.groupMeshletIndices.size() * sizeof(uint32_t),
            "LOD Group Meshlet Indices");
    }
    if (!data.groups.empty()) {
        data.groupBuffer = rhiCreateSharedBuffer(
            device,
            data.groups.data(),
            data.groups.size() * sizeof(GPUClusterGroup),
            "LOD Groups");
    }
    if (!data.nodes.empty()) {
        data.nodeBuffer = rhiCreateSharedBuffer(
            device,
            data.nodes.data(),
            data.nodes.size() * sizeof(GPULodNode),
            "LOD Nodes");
    }
    if (!data.levels.empty()) {
        data.levelBuffer = rhiCreateSharedBuffer(
            device,
            data.levels.data(),
            data.levels.size() * sizeof(ClusterLODLevel),
            "LOD Levels");
    }

    // Upload packed cluster buffers
    if (!data.packedClusters.empty()) {
        data.packedClusterBuffer = rhiCreateSharedBuffer(
            device,
            data.packedClusters.data(),
            data.packedClusters.size() * sizeof(PackedCluster),
            "Packed Clusters");
    }
    if (!data.clusterVertexData.empty()) {
        data.clusterVertexDataBuffer = rhiCreateSharedBuffer(
            device,
            data.clusterVertexData.data(),
            data.clusterVertexData.size(),
            "Cluster Vertex Data");
    }
    if (!data.clusterIndexData.empty()) {
        data.clusterIndexDataBuffer = rhiCreateSharedBuffer(
            device,
            data.clusterIndexData.data(),
            data.clusterIndexData.size(),
            "Cluster Index Data");
    }

    if (!data.meshletBuffer.nativeHandle() ||
        !data.meshletVerticesBuffer.nativeHandle() ||
        !data.meshletTrianglesBuffer.nativeHandle() ||
        !data.boundsBuffer.nativeHandle() ||
        !data.materialIDsBuffer.nativeHandle() ||
        (!data.groupMeshletIndices.empty() && !data.groupMeshletIndicesBuffer.nativeHandle()) ||
        (!data.groups.empty() && !data.groupBuffer.nativeHandle()) ||
        (!data.nodes.empty() && !data.nodeBuffer.nativeHandle()) ||
        (!data.levels.empty() && !data.levelBuffer.nativeHandle())) {
        releaseClusterLODHandles(data);
        spdlog::error("Failed to create GPU buffers for ClusterLOD payload");
        return false;
    }

    data.totalMeshletCount = static_cast<uint32_t>(data.allMeshlets.size());
    data.totalGroupCount = static_cast<uint32_t>(data.groups.size());
    data.totalNodeCount = static_cast<uint32_t>(data.nodes.size());
    data.lodLevelCount = static_cast<uint32_t>(data.levels.size());
    return true;
}

bool saveClusterLODToCache(const LoadedMesh& mesh,
                           const std::string& sourcePath,
                           const std::string& cacheDirectory,
                           const ClusterLODData& data) {
    const uint32_t expectedPrimitiveGroupCount = mesh.primitiveGroups.empty()
        ? 1u
        : static_cast<uint32_t>(mesh.primitiveGroups.size());
    if (!validateClusterLodPayload(data, expectedPrimitiveGroupCount)) {
        spdlog::warn("Skipping ClusterLOD cache write because payload validation failed");
        return false;
    }

    std::filesystem::path cacheDir(cacheDirectory);
    std::error_code createError;
    std::filesystem::create_directories(cacheDir, createError);
    if (createError) {
        spdlog::warn("Failed to create ClusterLOD cache directory {}: {}",
                     cacheDir.string(),
                     createError.message());
        return false;
    }

    const uint64_t meshSignature = computeMeshSignature(mesh);
    const std::filesystem::path cachePath = makeCacheFilePath(sourcePath, cacheDirectory, meshSignature);

    ClusterLODCacheHeader header{};
    std::memcpy(header.magic, kClusterLodCacheMagic, sizeof(header.magic));
    header.version = kClusterLodCacheVersion;
    header.maxVertices = static_cast<uint32_t>(kLodMaxVertices);
    header.minTriangles = static_cast<uint32_t>(kLodMinTriangles);
    header.maxTriangles = static_cast<uint32_t>(kLodMaxTriangles);
    header.partitionSize = static_cast<uint32_t>(kPartitionSize);
    header.hierarchyNodeWidth = static_cast<uint32_t>(kHierarchyNodeWidth);
    header.simplifyRatio = kSimplifyRatio;
    header.simplifyThreshold = kSimplifyThreshold;
    header.clusterSplit = kClusterSplit;
    header.meshSignature = meshSignature;
    header.meshletCount = data.allMeshlets.size();
    header.meshletVertexCount = data.allMeshletVertices.size();
    header.packedTriangleCount = data.allPackedTriangles.size();
    header.boundsCount = data.allBounds.size();
    header.materialCount = data.allMaterialIDs.size();
    header.groupMeshletIndexCount = data.groupMeshletIndices.size();
    header.groupCount = data.groups.size();
    header.nodeCount = data.nodes.size();
    header.levelCount = data.levels.size();
    header.primitiveGroupRootCount = data.primitiveGroupLodRoots.size();

    std::ofstream file(cachePath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        spdlog::warn("Failed to open ClusterLOD cache file for write: {}", cachePath.string());
        return false;
    }

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    const bool ok = static_cast<bool>(file) &&
                    writeVector(file, data.allMeshlets) &&
                    writeVector(file, data.allMeshletVertices) &&
                    writeVector(file, data.allPackedTriangles) &&
                    writeVector(file, data.allBounds) &&
                    writeVector(file, data.allMaterialIDs) &&
                    writeVector(file, data.groupMeshletIndices) &&
                    writeVector(file, data.groups) &&
                    writeVector(file, data.nodes) &&
                    writeVector(file, data.levels) &&
                    writeVector(file, data.primitiveGroupLodRoots);
    if (!ok) {
        spdlog::warn("Failed to write ClusterLOD cache file {}", cachePath.string());
        return false;
    }

    spdlog::info("Saved ClusterLOD cache with {} meshlets, {} groups, {} nodes to {}",
                 data.totalMeshletCount,
                 data.totalGroupCount,
                 data.totalNodeCount,
                 cachePath.string());
    return true;
}

bool loadClusterLODFromCache(const RhiDevice& device,
                             const LoadedMesh& mesh,
                             const std::string& sourcePath,
                             const std::string& cacheDirectory,
                             ClusterLODData& out) {
    const uint64_t meshSignature = computeMeshSignature(mesh);
    const std::filesystem::path cachePath = makeCacheFilePath(sourcePath, cacheDirectory, meshSignature);
    if (!std::filesystem::exists(cachePath)) {
        return false;
    }

    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        spdlog::warn("Failed to open ClusterLOD cache file {}", cachePath.string());
        return false;
    }

    ClusterLODCacheHeader header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        spdlog::warn("Failed to read ClusterLOD cache header {}", cachePath.string());
        return false;
    }

    const uint32_t expectedPrimitiveGroupCount = mesh.primitiveGroups.empty()
        ? 1u
        : static_cast<uint32_t>(mesh.primitiveGroups.size());

    if (std::memcmp(header.magic, kClusterLodCacheMagic, sizeof(header.magic)) != 0 ||
        header.version != kClusterLodCacheVersion ||
        header.maxVertices != kLodMaxVertices ||
        header.minTriangles != kLodMinTriangles ||
        header.maxTriangles != kLodMaxTriangles ||
        header.partitionSize != kPartitionSize ||
        header.hierarchyNodeWidth != kHierarchyNodeWidth ||
        std::fabs(header.simplifyRatio - kSimplifyRatio) > 1e-6f ||
        std::fabs(header.simplifyThreshold - kSimplifyThreshold) > 1e-6f ||
        std::fabs(header.clusterSplit - kClusterSplit) > 1e-6f ||
        header.meshSignature != meshSignature ||
        header.primitiveGroupRootCount != expectedPrimitiveGroupCount) {
        spdlog::warn("ClusterLOD cache {} is incompatible with the current mesh", cachePath.string());
        return false;
    }

    ClusterLODData cached;
    cached.sourceSceneSignature = meshSignature;
    if (!readVector(file, header.meshletCount, cached.allMeshlets) ||
        !readVector(file, header.meshletVertexCount, cached.allMeshletVertices) ||
        !readVector(file, header.packedTriangleCount, cached.allPackedTriangles) ||
        !readVector(file, header.boundsCount, cached.allBounds) ||
        !readVector(file, header.materialCount, cached.allMaterialIDs) ||
        !readVector(file, header.groupMeshletIndexCount, cached.groupMeshletIndices) ||
        !readVector(file, header.groupCount, cached.groups) ||
        !readVector(file, header.nodeCount, cached.nodes) ||
        !readVector(file, header.levelCount, cached.levels) ||
        !readVector(file, header.primitiveGroupRootCount, cached.primitiveGroupLodRoots)) {
        spdlog::warn("Failed to read ClusterLOD cache payload {}", cachePath.string());
        return false;
    }

    const auto currentOffset = file.tellg();
    file.seekg(0, std::ios::end);
    const auto endOffset = file.tellg();
    if (currentOffset != endOffset) {
        spdlog::warn("ClusterLOD cache {} has unexpected trailing data", cachePath.string());
        return false;
    }

    assignLod0PrimitiveGroupRoots(cached, expectedPrimitiveGroupCount);

    if (!validateClusterLodPayload(cached, expectedPrimitiveGroupCount)) {
        spdlog::warn("ClusterLOD cache {} failed payload validation", cachePath.string());
        return false;
    }

    buildPackedClusterData(mesh, cached);

    if (!uploadClusterLodBuffers(device, cached)) {
        spdlog::warn("Failed to create GPU ClusterLOD buffers from cache {}", cachePath.string());
        return false;
    }

    out = std::move(cached);
    spdlog::info("Loaded ClusterLOD cache with {} meshlets, {} groups, {} nodes from {}",
                 out.totalMeshletCount,
                 out.totalGroupCount,
                 out.totalNodeCount,
                 cachePath.string());
    return true;
}

} // namespace

bool buildClusterLOD(const RhiDevice& device,
                     const LoadedMesh& mesh,
                     const MeshletData& meshletData,
                     ClusterLODData& out) {
    releaseClusterLOD(out);
    out = ClusterLODData{};
    out.sourceSceneSignature = computeMeshSignature(mesh);

    const auto buildStart = std::chrono::steady_clock::now();

    const float* allPositions = mesh.cpuPositions.empty()
        ? static_cast<const float*>(rhiBufferContents(mesh.positionBuffer))
        : mesh.cpuPositions.data();
    if (!allPositions || meshletData.cpuMeshlets.empty()) {
        spdlog::error("ClusterLOD: requires CPU-readable positions and source meshlets");
        return false;
    }

    const uint32_t primitiveGroupCount = mesh.primitiveGroups.empty()
        ? 1u
        : static_cast<uint32_t>(mesh.primitiveGroups.size());
    out.primitiveGroupLodRoots.assign(primitiveGroupCount, kInvalidIndex);

    std::vector<unsigned int> positionRemap(mesh.vertexCount);
    meshopt_generatePositionRemap(positionRemap.data(), allPositions, mesh.vertexCount, sizeof(float) * 3);

    std::vector<uint32_t> meshletPrefix(primitiveGroupCount + 1, 0);
    for (uint32_t groupIndex = 0; groupIndex < primitiveGroupCount; ++groupIndex) {
        const uint32_t meshletCount = (groupIndex < meshletData.meshletsPerGroup.size())
            ? meshletData.meshletsPerGroup[groupIndex]
            : 0u;
        meshletPrefix[groupIndex + 1] = meshletPrefix[groupIndex] + meshletCount;
    }

    for (uint32_t primitiveGroupIndex = 0; primitiveGroupIndex < primitiveGroupCount; ++primitiveGroupIndex) {
        const uint32_t baseMeshletStart = meshletPrefix[primitiveGroupIndex];
        const uint32_t baseMeshletCount = meshletPrefix[primitiveGroupIndex + 1] - baseMeshletStart;
        if (baseMeshletCount == 0) {
            continue;
        }

        ClusterLODData local;
        if (!buildPrimitiveGroupClusterLOD(mesh,
                                           meshletData,
                                           primitiveGroupIndex,
                                           baseMeshletStart,
                                           baseMeshletCount,
                                           positionRemap,
                                           local)) {
            spdlog::warn("ClusterLOD: failed to build primitive group {} hierarchy", primitiveGroupIndex);
            continue;
        }

        appendClusterLOD(local, primitiveGroupIndex, out);
    }

    if (out.allMeshlets.empty() || out.groups.empty() || out.nodes.empty()) {
        spdlog::warn("ClusterLOD: no primitive groups produced a valid hierarchy");
        return false;
    }

    assignLod0PrimitiveGroupRoots(out, primitiveGroupCount);
    buildPackedClusterData(mesh, out);

    if (!uploadClusterLodBuffers(device, out)) {
        return false;
    }

    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("ClusterLOD: built {} primitive-group roots, {} levels, {} meshlets, {} groups, {} nodes in {} ms",
                 std::count_if(out.primitiveGroupLodRoots.begin(),
                               out.primitiveGroupLodRoots.end(),
                               [](uint32_t rootNode) { return rootNode != kInvalidIndex; }),
                 out.lodLevelCount,
                 out.totalMeshletCount,
                 out.totalGroupCount,
                 out.totalNodeCount,
                 elapsedMs);
    return true;
}

bool loadOrBuildClusterLOD(const RhiDevice& device,
                           const LoadedMesh& mesh,
                           const MeshletData& meshletData,
                           const std::string& sourcePath,
                           const std::string& cacheDirectory,
                           ClusterLODData& out) {
    releaseClusterLOD(out);
    out = ClusterLODData{};

    if (loadClusterLODFromCache(device, mesh, sourcePath, cacheDirectory, out)) {
        return true;
    }

    if (!buildClusterLOD(device, mesh, meshletData, out)) {
        return false;
    }

    if (!saveClusterLODToCache(mesh, sourcePath, cacheDirectory, out)) {
        spdlog::warn("Continuing with runtime-generated ClusterLOD for {}", sourcePath);
    }

    return true;
}

void releaseClusterLOD(ClusterLODData& data) {
    releaseClusterLODHandles(data);
    data.allMeshlets.clear();
    data.allMeshletVertices.clear();
    data.allPackedTriangles.clear();
    data.allBounds.clear();
    data.allMaterialIDs.clear();
    data.groupMeshletIndices.clear();
    data.groups.clear();
    data.nodes.clear();
    data.levels.clear();
    data.primitiveGroupLodRoots.clear();
    data.packedClusters.clear();
    data.clusterVertexData.clear();
    data.clusterIndexData.clear();
    data.totalMeshletCount = 0;
    data.totalGroupCount = 0;
    data.totalNodeCount = 0;
    data.lodLevelCount = 0;
    data.sourceSceneSignature = 0u;
}

void drawClusterLODStats(const ClusterLODData& data) {
    if (!ImGui::CollapsingHeader("Cluster LOD")) {
        return;
    }

    ImGui::Text("Primitive Groups with LOD: %u",
                static_cast<uint32_t>(std::count_if(
                    data.primitiveGroupLodRoots.begin(),
                    data.primitiveGroupLodRoots.end(),
                    [](uint32_t rootNode) { return rootNode != kInvalidIndex; })));
    ImGui::Text("LOD Levels: %u", data.lodLevelCount);
    ImGui::Text("Total Meshlets: %u", data.totalMeshletCount);
    ImGui::Text("Total Groups: %u", data.totalGroupCount);
    ImGui::Text("Total Nodes: %u", data.totalNodeCount);
    ImGui::Separator();

    if (ImGui::TreeNode("Per-Level Details")) {
        for (const ClusterLODLevel& level : data.levels) {
            uint32_t triangleCount = 0;
            for (uint32_t meshletIndex = level.meshletStart;
                 meshletIndex < level.meshletStart + level.meshletCount;
                 ++meshletIndex) {
                triangleCount += data.allMeshlets[meshletIndex].triangle_count;
            }

            ImGui::Text("Group %u / LOD %u: %u meshlets, %u tris, %u groups, root %u",
                        level.primitiveGroupIndex,
                        level.depth,
                        level.meshletCount,
                        triangleCount,
                        level.groupCount,
                        level.rootNode);
        }
        ImGui::TreePop();
    }
}
