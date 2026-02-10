#include "meshlet_builder.h"
#include "mesh_loader.h"
#include <meshoptimizer.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>

static constexpr size_t MAX_VERTICES  = 64;
static constexpr size_t MAX_TRIANGLES = 124;
static constexpr float  CONE_WEIGHT   = 0.5f;

bool buildMeshlets(MTL::Device* device, const LoadedMesh& mesh, MeshletData& out) {
    out.meshletsPerGroup.clear();

    const auto* allPositions = static_cast<const float*>(mesh.positionBuffer->contents());
    size_t totalVertexCount = mesh.vertexCount;
    size_t vertexStride = sizeof(float) * 3;

    // Accumulated output across all primitive groups
    std::vector<GPUMeshlet> allGpuMeshlets;
    std::vector<unsigned int> allMeshletVertices;
    std::vector<uint32_t> allPackedTriangles;
    std::vector<GPUMeshletBounds> allBounds;
    std::vector<uint32_t> allMaterialIDs;

    // If no primitive groups, fall back to treating entire mesh as one group
    std::vector<LoadedMesh::PrimitiveGroup> groups = mesh.primitiveGroups;
    if (groups.empty()) {
        LoadedMesh::PrimitiveGroup g;
        g.indexOffset = 0;
        g.indexCount = mesh.indexCount;
        g.vertexOffset = 0;
        g.vertexCount = mesh.vertexCount;
        g.materialIndex = 0;
        groups.push_back(g);
    }

    const auto* allIndices = static_cast<const uint32_t*>(mesh.indexBuffer->contents());

    for (const auto& group : groups) {
        const uint32_t* groupIndices = allIndices + group.indexOffset;
        size_t groupIndexCount = group.indexCount;

        if (groupIndexCount == 0) {
            out.meshletsPerGroup.push_back(0);
            continue;
        }

        // Compute worst-case buffer sizes for this group
        size_t maxMeshlets = meshopt_buildMeshletsBound(groupIndexCount, MAX_VERTICES, MAX_TRIANGLES);
        std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
        std::vector<unsigned int> meshletVertices(maxMeshlets * MAX_VERTICES);
        std::vector<unsigned char> meshletTriangles(maxMeshlets * MAX_TRIANGLES * 3);

        // Build meshlets for this group
        // Note: indices in groupIndices are global vertex indices (already offset)
        size_t meshletCount = meshopt_buildMeshlets(
            meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
            groupIndices, groupIndexCount,
            allPositions, totalVertexCount, vertexStride,
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
                allPositions, totalVertexCount, vertexStride);

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
            allMeshletVertices.insert(allMeshletVertices.end(),
                meshletVertices.begin(), meshletVertices.begin() + usedVertices);
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
    }

    size_t totalMeshlets = allGpuMeshlets.size();
    if (totalMeshlets == 0) {
        spdlog::error("No meshlets built");
        return false;
    }

    // Create Metal buffers
    out.meshletBuffer = device->newBuffer(
        allGpuMeshlets.data(), allGpuMeshlets.size() * sizeof(GPUMeshlet),
        MTL::ResourceStorageModeShared);
    out.meshletVertices = device->newBuffer(
        allMeshletVertices.data(), allMeshletVertices.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    out.meshletTriangles = device->newBuffer(
        allPackedTriangles.data(), allPackedTriangles.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    out.boundsBuffer = device->newBuffer(
        allBounds.data(), allBounds.size() * sizeof(GPUMeshletBounds),
        MTL::ResourceStorageModeShared);
    out.materialIDs = device->newBuffer(
        allMaterialIDs.data(), allMaterialIDs.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    out.meshletCount = static_cast<uint32_t>(totalMeshlets);

    // Print stats
    size_t totalTris = 0, totalVerts = 0;
    for (const auto& gm : allGpuMeshlets) {
        totalTris += gm.triangle_count;
        totalVerts += gm.vertex_count;
    }
    spdlog::info("Built {} meshlets from {} groups (avg {} verts, {} tris per meshlet)",
                 totalMeshlets, groups.size(), totalVerts / totalMeshlets, totalTris / totalMeshlets);

    return true;
}
