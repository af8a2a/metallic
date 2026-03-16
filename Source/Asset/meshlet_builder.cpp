#include "meshlet_builder.h"
#include "mesh_loader.h"
#include "rhi_resource_utils.h"

#include <meshoptimizer.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <vector>
#include <cstring>

static constexpr size_t MAX_VERTICES  = 64;
static constexpr size_t MAX_TRIANGLES = 124;
static constexpr float  CONE_WEIGHT   = 0.5f;

bool buildMeshlets(const RhiDevice& device, const LoadedMesh& mesh, MeshletData& out) {
    out.meshletsPerGroup.clear();

    const auto buildStart = std::chrono::steady_clock::now();

    const auto* allPositions = mesh.cpuPositions.empty()
        ? static_cast<const float*>(rhiBufferContents(mesh.positionBuffer))
        : mesh.cpuPositions.data();
    const auto* allIndices = mesh.cpuIndices.empty()
        ? static_cast<const uint32_t*>(rhiBufferContents(mesh.indexBuffer))
        : mesh.cpuIndices.data();
    constexpr size_t kPositionStride = sizeof(float) * 3;

    if (!allPositions || !allIndices) {
        spdlog::error("Meshlet builder requires CPU-readable position and index data");
        return false;
    }

    // Accumulated output across all primitive groups
    std::vector<GPUMeshlet> allGpuMeshlets;
    std::vector<unsigned int> allMeshletVertices;
    std::vector<uint32_t> allPackedTriangles;
    std::vector<GPUMeshletBounds> allBounds;
    std::vector<uint32_t> allMaterialIDs;

    size_t meshletBound = 0;
    if (mesh.primitiveGroups.empty()) {
        meshletBound = meshopt_buildMeshletsBound(mesh.indexCount, MAX_VERTICES, MAX_TRIANGLES);
    } else {
        for (const auto& group : mesh.primitiveGroups) {
            meshletBound += meshopt_buildMeshletsBound(group.indexCount, MAX_VERTICES, MAX_TRIANGLES);
        }
    }
    allGpuMeshlets.reserve(meshletBound);
    allBounds.reserve(meshletBound);
    allMaterialIDs.reserve(meshletBound);

    std::vector<meshopt_Meshlet> meshlets;
    std::vector<unsigned int> meshletVertices;
    std::vector<unsigned char> meshletTriangles;
    std::vector<uint32_t> localIndices;

    auto buildGroupMeshlets = [&](const LoadedMesh::PrimitiveGroup& group) {
        const uint32_t* groupIndices = allIndices + group.indexOffset;
        size_t groupIndexCount = group.indexCount;
        const uint32_t groupVertexOffset = group.vertexOffset;
        const uint32_t groupVertexCount = group.vertexCount;
        const auto* groupPositions = allPositions + static_cast<size_t>(groupVertexOffset) * 3;

        if (groupIndexCount == 0 || groupVertexCount == 0) {
            out.meshletsPerGroup.push_back(0);
            return true;
        }

        // Compute worst-case buffer sizes for this group
        size_t maxMeshlets = meshopt_buildMeshletsBound(groupIndexCount, MAX_VERTICES, MAX_TRIANGLES);
        meshlets.resize(maxMeshlets);
        meshletVertices.resize(maxMeshlets * MAX_VERTICES);
        meshletTriangles.resize(maxMeshlets * MAX_TRIANGLES * 3);

        const uint32_t groupVertexEnd = groupVertexOffset + groupVertexCount;
        const uint32_t* meshletSourceIndices = groupIndices;
        if (groupVertexOffset != 0 || groupVertexCount != mesh.vertexCount) {
            localIndices.resize(groupIndexCount);
            for (size_t i = 0; i < groupIndexCount; ++i) {
                const uint32_t index = groupIndices[i];
                if (index < groupVertexOffset || index >= groupVertexEnd) {
                    spdlog::error("Primitive group index {} out of range [{}, {})",
                                  index,
                                  groupVertexOffset,
                                  groupVertexEnd);
                    return false;
                }
                localIndices[i] = index - groupVertexOffset;
            }
            meshletSourceIndices = localIndices.data();
        }

        // Build meshlets for this group
        size_t meshletCount = meshopt_buildMeshlets(
            meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
            meshletSourceIndices, groupIndexCount,
            groupPositions, groupVertexCount, kPositionStride,
            MAX_VERTICES, MAX_TRIANGLES, CONE_WEIGHT);

        meshlets.resize(meshletCount);

        // Optimize each meshlet
        for (size_t i = 0; i < meshletCount; i++) {
            meshopt_optimizeMeshlet(
                &meshletVertices[meshlets[i].vertex_offset],
                &meshletTriangles[meshlets[i].triangle_offset],
                meshlets[i].triangle_count,
                meshlets[i].vertex_count);
        }

        // Compute bounds
        for (size_t i = 0; i < meshletCount; i++) {
            meshopt_Bounds bounds = meshopt_computeMeshletBounds(
                &meshletVertices[meshlets[i].vertex_offset],
                &meshletTriangles[meshlets[i].triangle_offset],
                meshlets[i].triangle_count,
                groupPositions, groupVertexCount, kPositionStride);

            GPUMeshletBounds gb;
            gb.center_radius[0] = bounds.center[0];
            gb.center_radius[1] = bounds.center[1];
            gb.center_radius[2] = bounds.center[2];
            gb.center_radius[3] = bounds.radius;
            gb.cone_apex_pad[0] = bounds.cone_apex[0];
            gb.cone_apex_pad[1] = bounds.cone_apex[1];
            gb.cone_apex_pad[2] = bounds.cone_apex[2];
            gb.cone_apex_pad[3] = 0.0f;
            gb.cone_axis_cutoff[0] = bounds.cone_axis[0];
            gb.cone_axis_cutoff[1] = bounds.cone_axis[1];
            gb.cone_axis_cutoff[2] = bounds.cone_axis[2];
            gb.cone_axis_cutoff[3] = bounds.cone_cutoff;
            allBounds.push_back(gb);
        }

        // Merge vertex indices (global offsets into allMeshletVertices)
        size_t vertexBaseOffset = allMeshletVertices.size();
        if (meshletCount > 0) {
            const meshopt_Meshlet& last = meshlets.back();
            size_t usedVertices = last.vertex_offset + last.vertex_count;
            allMeshletVertices.reserve(allMeshletVertices.size() + usedVertices);
            for (size_t i = 0; i < usedVertices; ++i) {
                allMeshletVertices.push_back(meshletVertices[i] + groupVertexOffset);
            }
        }

        // Pack triangles and merge
        size_t packedBaseOffset = allPackedTriangles.size();
        for (size_t i = 0; i < meshletCount; i++) {
            const meshopt_Meshlet& m = meshlets[i];
            for (size_t t = 0; t < m.triangle_count; t++) {
                size_t srcIdx = m.triangle_offset + t * 3;
                uint32_t v0 = meshletTriangles[srcIdx + 0];
                uint32_t v1 = meshletTriangles[srcIdx + 1];
                uint32_t v2 = meshletTriangles[srcIdx + 2];
                allPackedTriangles.push_back(v0 | (v1 << 8) | (v2 << 16));
            }
        }

        // Build GPU meshlet descriptors with global offsets
        size_t triOffset = packedBaseOffset;
        for (size_t i = 0; i < meshletCount; i++) {
            GPUMeshlet gm;
            gm.vertex_offset   = static_cast<uint32_t>(vertexBaseOffset + meshlets[i].vertex_offset);
            gm.triangle_offset = static_cast<uint32_t>(triOffset);
            gm.vertex_count    = meshlets[i].vertex_count;
            gm.triangle_count  = meshlets[i].triangle_count;
            allGpuMeshlets.push_back(gm);
            triOffset += meshlets[i].triangle_count;
        }

        // All meshlets in this group share the same material
        for (size_t i = 0; i < meshletCount; i++) {
            allMaterialIDs.push_back(group.materialIndex);
        }

        out.meshletsPerGroup.push_back(static_cast<uint32_t>(meshletCount));
        return true;
    };

    if (mesh.primitiveGroups.empty()) {
        LoadedMesh::PrimitiveGroup group;
        group.indexOffset = 0;
        group.indexCount = mesh.indexCount;
        group.vertexOffset = 0;
        group.vertexCount = mesh.vertexCount;
        group.materialIndex = 0;
        if (!buildGroupMeshlets(group)) {
            return false;
        }
    } else {
        for (const auto& group : mesh.primitiveGroups) {
            if (!buildGroupMeshlets(group)) {
                return false;
            }
        }
    }

    size_t totalMeshlets = allGpuMeshlets.size();
    if (totalMeshlets == 0) {
        spdlog::error("No meshlets built");
        return false;
    }

    out.meshletBuffer = rhiCreateSharedBuffer(
        device, allGpuMeshlets.data(), allGpuMeshlets.size() * sizeof(GPUMeshlet), "Meshlets");
    out.meshletVertices = rhiCreateSharedBuffer(
        device, allMeshletVertices.data(), allMeshletVertices.size() * sizeof(uint32_t), "Meshlet Vertices");
    out.meshletTriangles = rhiCreateSharedBuffer(
        device, allPackedTriangles.data(), allPackedTriangles.size() * sizeof(uint32_t), "Meshlet Triangles");
    out.boundsBuffer = rhiCreateSharedBuffer(
        device, allBounds.data(), allBounds.size() * sizeof(GPUMeshletBounds), "Meshlet Bounds");
    out.materialIDs = rhiCreateSharedBuffer(
        device, allMaterialIDs.data(), allMaterialIDs.size() * sizeof(uint32_t), "Meshlet Material IDs");
    out.meshletCount = static_cast<uint32_t>(totalMeshlets);

    // Print stats
    size_t totalTris = 0, totalVerts = 0;
    for (const auto& gm : allGpuMeshlets) {
        totalTris += gm.triangle_count;
        totalVerts += gm.vertex_count;
    }
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("Built {} meshlets from {} groups (avg {} verts, {} tris per meshlet)",
                 totalMeshlets,
                 out.meshletsPerGroup.size(),
                 totalVerts / totalMeshlets,
                 totalTris / totalMeshlets);
    spdlog::info("Meshlet build completed in {} ms", elapsedMs);

    return true;
}
