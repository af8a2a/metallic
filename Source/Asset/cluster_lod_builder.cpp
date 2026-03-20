#include "cluster_lod_builder.h"
#include "mesh_loader.h"
#include "rhi_resource_utils.h"

#include <meshoptimizer.h>
#include <spdlog/spdlog.h>
#include <imgui.h>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cfloat>

// Configuration
static constexpr size_t LOD_MAX_VERTICES    = 64;
static constexpr size_t LOD_MIN_TRIANGLES   = 20;
static constexpr size_t LOD_MAX_TRIANGLES   = 124;
static constexpr size_t PARTITION_SIZE      = 8;
static constexpr float  SIMPLIFY_RATIO      = 0.5f;
static constexpr float  SIMPLIFY_THRESHOLD  = 0.85f;
static constexpr float  CLUSTER_SPLIT       = 2.0f;

namespace {

struct Cluster {
    std::vector<unsigned int> indices;
    size_t vertexCount = 0;
    float  center[3]   = {};
    float  radius      = 0.f;
    float  error       = 0.f;
};

struct BoundsResult {
    float center[3];
    float radius;
    float error;
};

static BoundsResult computeBounds(const float* positions, size_t vertexCount,
                                  size_t stride, const std::vector<unsigned int>& indices,
                                  float error)
{
    meshopt_Bounds b = meshopt_computeClusterBounds(
        indices.data(), indices.size(),
        positions, vertexCount, stride);
    return {{b.center[0], b.center[1], b.center[2]}, b.radius, error};
}

static BoundsResult mergeBounds(const std::vector<Cluster>& clusters,
                                const std::vector<int>& group)
{
    struct SD { float cx, cy, cz, r; };
    std::vector<SD> spheres(group.size());
    for (size_t i = 0; i < group.size(); ++i) {
        const auto& c = clusters[group[i]];
        spheres[i] = {c.center[0], c.center[1], c.center[2], c.radius};
    }
    meshopt_Bounds merged = meshopt_computeSphereBounds(
        &spheres[0].cx, spheres.size(), sizeof(SD),
        &spheres[0].r, sizeof(SD));

    float maxErr = 0.f;
    for (size_t i = 0; i < group.size(); ++i)
        maxErr = std::max(maxErr, clusters[group[i]].error);

    return {{merged.center[0], merged.center[1], merged.center[2]}, merged.radius, maxErr};
}

static std::vector<Cluster> clusterize(const float* positions, size_t vertexCount,
                                       size_t stride,
                                       const unsigned int* indices, size_t indexCount)
{
    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, LOD_MAX_VERTICES, LOD_MIN_TRIANGLES);
    std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
    std::vector<unsigned int> meshletVerts(indexCount);
    std::vector<unsigned char> meshletTris(indexCount);

    size_t count = meshopt_buildMeshletsFlex(
        meshlets.data(), meshletVerts.data(), meshletTris.data(),
        indices, indexCount,
        positions, vertexCount, stride,
        LOD_MAX_VERTICES, LOD_MIN_TRIANGLES, LOD_MAX_TRIANGLES,
        0.f, CLUSTER_SPLIT);
    meshlets.resize(count);

    std::vector<Cluster> result(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& m = meshlets[i];
        meshopt_optimizeMeshlet(
            &meshletVerts[m.vertex_offset],
            &meshletTris[m.triangle_offset],
            m.triangle_count, m.vertex_count);

        result[i].vertexCount = m.vertex_count;
        result[i].indices.resize(m.triangle_count * 3);
        for (size_t j = 0; j < m.triangle_count * 3; ++j)
            result[i].indices[j] = meshletVerts[m.vertex_offset + meshletTris[m.triangle_offset + j]];
    }
    return result;
}

static std::vector<std::vector<int>> partitionClusters(
    const std::vector<Cluster>& clusters,
    const std::vector<int>& pending,
    const std::vector<unsigned int>& posRemap,
    const float* positions, size_t vertexCount, size_t stride)
{
    if (pending.size() <= PARTITION_SIZE)
        return {pending};

    std::vector<unsigned int> clusterIndices;
    std::vector<unsigned int> clusterCounts(pending.size());

    size_t totalIdx = 0;
    for (size_t i = 0; i < pending.size(); ++i)
        totalIdx += clusters[pending[i]].indices.size();
    clusterIndices.reserve(totalIdx);

    for (size_t i = 0; i < pending.size(); ++i) {
        const auto& c = clusters[pending[i]];
        clusterCounts[i] = static_cast<unsigned int>(c.indices.size());
        for (unsigned int idx : c.indices)
            clusterIndices.push_back(posRemap[idx]);
    }

    std::vector<unsigned int> partIds(pending.size());
    size_t partCount = meshopt_partitionClusters(
        partIds.data(), clusterIndices.data(), clusterIndices.size(),
        clusterCounts.data(), clusterCounts.size(),
        positions, vertexCount, stride, PARTITION_SIZE);

    std::vector<std::vector<int>> partitions(partCount);
    for (size_t i = 0; i < partCount; ++i)
        partitions[i].reserve(PARTITION_SIZE + PARTITION_SIZE / 3);

    for (size_t i = 0; i < pending.size(); ++i)
        partitions[partIds[i]].push_back(pending[i]);

    return partitions;
}

static void lockBoundary(std::vector<unsigned char>& locks,
                         const std::vector<std::vector<int>>& groups,
                         const std::vector<Cluster>& clusters,
                         const std::vector<unsigned int>& remap)
{
    for (size_t i = 0; i < locks.size(); ++i)
        locks[i] &= static_cast<unsigned char>(~((1 << 0) | (1 << 7)));

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        for (int ci : groups[gi]) {
            for (unsigned int v : clusters[ci].indices) {
                unsigned int r = remap[v];
                locks[r] |= locks[r] >> 7;
            }
        }
        for (int ci : groups[gi]) {
            for (unsigned int v : clusters[ci].indices) {
                unsigned int r = remap[v];
                locks[r] |= 1 << 7;
            }
        }
    }

    for (size_t i = 0; i < locks.size(); ++i) {
        unsigned int r = remap[i];
        locks[i] = locks[r] & 1;
    }
}

static void emitMeshlet(const Cluster& cluster,
                        std::vector<GPUMeshlet>& outMeshlets,
                        std::vector<unsigned int>& outVertices,
                        std::vector<uint32_t>& outPackedTris,
                        std::vector<GPUMeshletBounds>& outBounds,
                        const float* positions, size_t vertexCount, size_t stride)
{
    std::vector<unsigned int> localVerts;
    std::vector<unsigned char> localTris;
    localVerts.reserve(cluster.vertexCount);
    localTris.reserve(cluster.indices.size());

    for (size_t i = 0; i < cluster.indices.size(); ++i) {
        unsigned int gv = cluster.indices[i];
        unsigned char localIdx = 0;
        bool found = false;
        for (size_t j = 0; j < localVerts.size(); ++j) {
            if (localVerts[j] == gv) {
                localIdx = static_cast<unsigned char>(j);
                found = true;
                break;
            }
        }
        if (!found) {
            localIdx = static_cast<unsigned char>(localVerts.size());
            localVerts.push_back(gv);
        }
        localTris.push_back(localIdx);
    }

    meshopt_Bounds b = meshopt_computeClusterBounds(
        cluster.indices.data(), cluster.indices.size(),
        positions, vertexCount, stride);

    GPUMeshletBounds gb;
    gb.center_radius[0] = b.center[0];
    gb.center_radius[1] = b.center[1];
    gb.center_radius[2] = b.center[2];
    gb.center_radius[3] = b.radius;
    gb.cone_apex_pad[0] = b.cone_apex[0];
    gb.cone_apex_pad[1] = b.cone_apex[1];
    gb.cone_apex_pad[2] = b.cone_apex[2];
    gb.cone_apex_pad[3] = 0.0f;
    gb.cone_axis_cutoff[0] = b.cone_axis[0];
    gb.cone_axis_cutoff[1] = b.cone_axis[1];
    gb.cone_axis_cutoff[2] = b.cone_axis[2];
    gb.cone_axis_cutoff[3] = b.cone_cutoff;

    GPUMeshlet gm;
    gm.vertex_offset = static_cast<uint32_t>(outVertices.size());
    gm.triangle_offset = static_cast<uint32_t>(outPackedTris.size());
    gm.vertex_count = static_cast<uint32_t>(localVerts.size());
    gm.triangle_count = static_cast<uint32_t>(cluster.indices.size() / 3);

    outVertices.insert(outVertices.end(), localVerts.begin(), localVerts.end());

    for (size_t t = 0; t < gm.triangle_count; ++t) {
        uint32_t v0 = localTris[t * 3 + 0];
        uint32_t v1 = localTris[t * 3 + 1];
        uint32_t v2 = localTris[t * 3 + 2];
        outPackedTris.push_back(v0 | (v1 << 8) | (v2 << 16));
    }

    outMeshlets.push_back(gm);
    outBounds.push_back(gb);
}

} // anonymous namespace

// ============================================================================
// buildClusterLOD
// ============================================================================
bool buildClusterLOD(const RhiDevice& device,
                     const LoadedMesh& mesh,
                     const MeshletData& meshletData,
                     ClusterLODData& out)
{
    const auto buildStart = std::chrono::steady_clock::now();

    const float* allPositions = mesh.cpuPositions.empty()
        ? static_cast<const float*>(rhiBufferContents(mesh.positionBuffer))
        : mesh.cpuPositions.data();
    const uint32_t* allIndices = mesh.cpuIndices.empty()
        ? static_cast<const uint32_t*>(rhiBufferContents(mesh.indexBuffer))
        : mesh.cpuIndices.data();
    constexpr size_t kStride = sizeof(float) * 3;

    if (!allPositions || !allIndices) {
        spdlog::error("ClusterLOD: requires CPU-readable position and index data");
        return false;
    }

    // === LOD 0: copy from existing meshlet data ===
    out.allMeshlets       = meshletData.cpuMeshlets;
    out.allMeshletVertices.resize(meshletData.cpuMeshletVertices.size());
    for (size_t i = 0; i < meshletData.cpuMeshletVertices.size(); ++i)
        out.allMeshletVertices[i] = meshletData.cpuMeshletVertices[i];
    out.allBounds         = meshletData.cpuBounds;
    out.allMaterialIDs    = meshletData.cpuMaterialIDs;

    // Pack LOD 0 triangles from raw 3-byte format
    out.allPackedTriangles.clear();
    {
        const auto& raw = meshletData.cpuMeshletTriangles;
        for (size_t i = 0; i + 2 < raw.size(); i += 3)
            out.allPackedTriangles.push_back(
                uint32_t(raw[i]) | (uint32_t(raw[i+1]) << 8) | (uint32_t(raw[i+2]) << 16));
    }

    uint32_t lod0Count = static_cast<uint32_t>(out.allMeshlets.size());

    ClusterLODLevel lod0;
    lod0.meshletStart = 0;
    lod0.meshletCount = lod0Count;
    lod0.groupStart   = 0;
    lod0.groupCount   = 0;
    out.levels.push_back(lod0);

    spdlog::info("ClusterLOD: LOD 0 = {} meshlets", lod0Count);

    // Build position remap for connectivity
    std::vector<unsigned int> posRemap(mesh.vertexCount);
    meshopt_generatePositionRemap(posRemap.data(), allPositions, mesh.vertexCount, kStride);

    // Convert LOD 0 meshlets to internal Cluster representation
    std::vector<Cluster> clusters;
    clusters.reserve(lod0Count * 4);

    for (uint32_t mi = 0; mi < lod0Count; ++mi) {
        const auto& gm = meshletData.cpuMeshlets[mi];
        Cluster c;
        c.indices.resize(gm.triangle_count * 3);
        for (uint32_t t = 0; t < gm.triangle_count; ++t) {
            for (uint32_t v = 0; v < 3; ++v) {
                size_t rawIdx = (gm.triangle_offset + t) * 3 + v;
                unsigned char localVert = meshletData.cpuMeshletTriangles[rawIdx];
                c.indices[t * 3 + v] = meshletData.cpuMeshletVertices[gm.vertex_offset + localVert];
            }
        }
        c.vertexCount = gm.vertex_count;
        BoundsResult br = computeBounds(allPositions, mesh.vertexCount, kStride, c.indices, 0.f);
        c.center[0] = br.center[0]; c.center[1] = br.center[1]; c.center[2] = br.center[2];
        c.radius = br.radius;
        c.error = 0.f;
        clusters.push_back(std::move(c));
    }

    std::vector<int> pending(lod0Count);
    for (uint32_t i = 0; i < lod0Count; ++i)
        pending[i] = static_cast<int>(i);

    std::vector<unsigned char> locks(mesh.vertexCount, 0);
    int depth = 0;

    // === Iterative simplification loop ===
    while (pending.size() > 1) {
        auto groups = partitionClusters(clusters, pending, posRemap,
                                        allPositions, mesh.vertexCount, kStride);
        lockBoundary(locks, groups, clusters, posRemap);

        std::vector<int> newPending;
        newPending.reserve(pending.size());

        uint32_t levelMeshletStart = static_cast<uint32_t>(out.allMeshlets.size());
        uint32_t levelGroupStart   = static_cast<uint32_t>(out.groups.size());
        uint32_t stuckCount = 0;

        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const auto& group = groups[gi];

            // Merge indices
            std::vector<unsigned int> merged;
            for (int ci : group)
                merged.insert(merged.end(), clusters[ci].indices.begin(),
                              clusters[ci].indices.end());

            size_t targetCount = size_t((merged.size() / 3) * SIMPLIFY_RATIO) * 3;
            BoundsResult groupBounds = mergeBounds(clusters, group);

            // Simplify
            std::vector<unsigned int> simplified(merged.size());
            float simplifyError = 0.f;
            size_t simplifiedSize = meshopt_simplify(
                simplified.data(), merged.data(), merged.size(),
                allPositions, mesh.vertexCount, kStride,
                targetCount, FLT_MAX,
                meshopt_SimplifySparse | meshopt_SimplifyLockBorder | meshopt_SimplifyErrorAbsolute,
                &simplifyError);
            simplified.resize(simplifiedSize);

            if (simplified.size() > size_t(merged.size() * SIMPLIFY_THRESHOLD)) {
                // Stuck: emit terminal group
                stuckCount++;
                GPUClusterGroup cg;
                cg.center[0] = groupBounds.center[0];
                cg.center[1] = groupBounds.center[1];
                cg.center[2] = groupBounds.center[2];
                cg.radius    = groupBounds.radius;
                cg.error     = groupBounds.error;
                cg.parentError  = FLT_MAX;
                cg.clusterStart = 0;
                cg.clusterCount = 0;
                out.groups.push_back(cg);
                continue;
            }

            // Enforce error monotonicity
            float newError = std::max(groupBounds.error, simplifyError);

            // Record the group for the children (current clusters)
            GPUClusterGroup childGroup;
            childGroup.center[0] = groupBounds.center[0];
            childGroup.center[1] = groupBounds.center[1];
            childGroup.center[2] = groupBounds.center[2];
            childGroup.radius    = groupBounds.radius;
            childGroup.error     = groupBounds.error;
            childGroup.parentError = newError;
            // clusterStart/Count refer to the child meshlets in the group
            // For LOD 0 children, these point into the LOD 0 range
            childGroup.clusterStart = 0;
            childGroup.clusterCount = static_cast<uint32_t>(group.size());
            out.groups.push_back(childGroup);

            // Free old cluster memory
            for (int ci : group)
                clusters[ci].indices = std::vector<unsigned int>();

            // Re-clusterize simplified result
            auto newClusters = clusterize(allPositions, mesh.vertexCount, kStride,
                                          simplified.data(), simplified.size());

            for (auto& nc : newClusters) {
                BoundsResult nb = computeBounds(allPositions, mesh.vertexCount, kStride,
                                                nc.indices, 0.f);
                // Use group-merged bounds for monotonicity
                nc.center[0] = groupBounds.center[0];
                nc.center[1] = groupBounds.center[1];
                nc.center[2] = groupBounds.center[2];
                nc.radius    = groupBounds.radius;
                nc.error     = newError;

                // Emit as GPU meshlet
                emitMeshlet(nc, out.allMeshlets, out.allMeshletVertices,
                            out.allPackedTriangles, out.allBounds,
                            allPositions, mesh.vertexCount, kStride);
                // Material: inherit from first child (simplified group shares material)
                uint32_t matId = 0;
                if (!group.empty() && group[0] < static_cast<int>(meshletData.cpuMaterialIDs.size()))
                    matId = meshletData.cpuMaterialIDs[group[0]];
                out.allMaterialIDs.push_back(matId);

                int newIdx = static_cast<int>(clusters.size());
                clusters.push_back(std::move(nc));
                newPending.push_back(newIdx);
            }
        }

        uint32_t levelMeshletEnd = static_cast<uint32_t>(out.allMeshlets.size());
        uint32_t levelGroupEnd   = static_cast<uint32_t>(out.groups.size());

        ClusterLODLevel level;
        level.meshletStart = levelMeshletStart;
        level.meshletCount = levelMeshletEnd - levelMeshletStart;
        level.groupStart   = levelGroupStart;
        level.groupCount   = levelGroupEnd - levelGroupStart;
        out.levels.push_back(level);

        depth++;
        spdlog::info("ClusterLOD: LOD {} = {} meshlets, {} groups ({} stuck)",
                     depth, level.meshletCount, level.groupCount, stuckCount);

        pending = std::move(newPending);

        if (pending.empty())
            break;
    }

    // Handle final single cluster
    if (pending.size() == 1) {
        const auto& c = clusters[pending[0]];
        GPUClusterGroup cg;
        cg.center[0] = c.center[0]; cg.center[1] = c.center[1]; cg.center[2] = c.center[2];
        cg.radius    = c.radius;
        cg.error     = c.error;
        cg.parentError = FLT_MAX;
        cg.clusterStart = 0;
        cg.clusterCount = 1;
        out.groups.push_back(cg);
    }

    // === Build node hierarchy (simple bottom-up) ===
    // For now, create a flat root node pointing to all groups
    if (!out.groups.empty()) {
        GPULodNode root;
        // Compute root bounding sphere from all groups
        struct SD { float cx, cy, cz, r; };
        std::vector<SD> allSpheres(out.groups.size());
        for (size_t i = 0; i < out.groups.size(); ++i) {
            allSpheres[i] = {out.groups[i].center[0], out.groups[i].center[1],
                             out.groups[i].center[2], out.groups[i].radius};
        }
        meshopt_Bounds rootBounds = meshopt_computeSphereBounds(
            &allSpheres[0].cx, allSpheres.size(), sizeof(SD),
            &allSpheres[0].r, sizeof(SD));

        root.center[0] = rootBounds.center[0];
        root.center[1] = rootBounds.center[1];
        root.center[2] = rootBounds.center[2];
        root.radius    = rootBounds.radius;
        root.maxError  = FLT_MAX;
        root.childOffset = 0;
        root.childCount  = static_cast<uint32_t>(out.groups.size());
        root.isLeaf      = 1;
        out.nodes.push_back(root);
    }

    // === Fill summary ===
    out.totalMeshletCount = static_cast<uint32_t>(out.allMeshlets.size());
    out.totalGroupCount   = static_cast<uint32_t>(out.groups.size());
    out.totalNodeCount    = static_cast<uint32_t>(out.nodes.size());
    out.lodLevelCount     = static_cast<uint32_t>(out.levels.size());

    // === Upload GPU buffers ===
    if (!out.allMeshlets.empty()) {
        out.meshletBuffer = rhiCreateSharedBuffer(device,
            out.allMeshlets.data(), out.allMeshlets.size() * sizeof(GPUMeshlet),
            "LOD Meshlets");
        out.meshletVerticesBuffer = rhiCreateSharedBuffer(device,
            out.allMeshletVertices.data(), out.allMeshletVertices.size() * sizeof(unsigned int),
            "LOD Meshlet Vertices");
        out.meshletTrianglesBuffer = rhiCreateSharedBuffer(device,
            out.allPackedTriangles.data(), out.allPackedTriangles.size() * sizeof(uint32_t),
            "LOD Meshlet Triangles");
        out.boundsBuffer = rhiCreateSharedBuffer(device,
            out.allBounds.data(), out.allBounds.size() * sizeof(GPUMeshletBounds),
            "LOD Bounds");
        out.materialIDsBuffer = rhiCreateSharedBuffer(device,
            out.allMaterialIDs.data(), out.allMaterialIDs.size() * sizeof(uint32_t),
            "LOD Material IDs");
    }
    if (!out.groups.empty()) {
        out.groupBuffer = rhiCreateSharedBuffer(device,
            out.groups.data(), out.groups.size() * sizeof(GPUClusterGroup),
            "LOD Groups");
    }
    if (!out.nodes.empty()) {
        out.nodeBuffer = rhiCreateSharedBuffer(device,
            out.nodes.data(), out.nodes.size() * sizeof(GPULodNode),
            "LOD Nodes");
    }

    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("ClusterLOD: built {} levels, {} total meshlets, {} groups, {} nodes in {} ms",
                 out.lodLevelCount, out.totalMeshletCount, out.totalGroupCount,
                 out.totalNodeCount, elapsedMs);

    return true;
}

// ============================================================================
// ImGui stats panel
// ============================================================================
void drawClusterLODStats(const ClusterLODData& data)
{
    if (!ImGui::CollapsingHeader("Cluster LOD"))
        return;

    ImGui::Text("LOD Levels: %u", data.lodLevelCount);
    ImGui::Text("Total Meshlets: %u", data.totalMeshletCount);
    ImGui::Text("Total Groups: %u", data.totalGroupCount);
    ImGui::Text("Total Nodes: %u", data.totalNodeCount);
    ImGui::Separator();

    if (ImGui::TreeNode("Per-Level Details")) {
        for (uint32_t i = 0; i < data.lodLevelCount; ++i) {
            const auto& lvl = data.levels[i];
            uint32_t triCount = 0;
            for (uint32_t m = lvl.meshletStart; m < lvl.meshletStart + lvl.meshletCount; ++m)
                triCount += data.allMeshlets[m].triangle_count;

            float ratio = (i > 0 && data.levels[i-1].meshletCount > 0)
                ? float(lvl.meshletCount) / float(data.levels[i-1].meshletCount)
                : 1.0f;

            ImGui::Text("LOD %u: %u meshlets, %u tris, %u groups (ratio %.2f)",
                        i, lvl.meshletCount, triCount, lvl.groupCount, ratio);
        }
        ImGui::TreePop();
    }
}
