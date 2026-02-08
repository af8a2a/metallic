#include "meshlet_builder.h"
#include "mesh_loader.h"
#include <meshoptimizer.h>
#include <iostream>
#include <vector>
#include <cstring>

static constexpr size_t MAX_VERTICES  = 64;
static constexpr size_t MAX_TRIANGLES = 124;
static constexpr float  CONE_WEIGHT   = 0.5f;

bool buildMeshlets(MTL::Device* device, const LoadedMesh& mesh, MeshletData& out) {
    // Read back CPU-side data from Metal buffers (StorageModeShared = CPU-accessible)
    const auto* indices = static_cast<const uint32_t*>(mesh.indexBuffer->contents());
    const auto* positions = static_cast<const float*>(mesh.positionBuffer->contents());
    size_t indexCount = mesh.indexCount;
    size_t vertexCount = mesh.vertexCount;
    size_t vertexStride = sizeof(float) * 3;

    // Compute worst-case buffer sizes
    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, MAX_VERTICES, MAX_TRIANGLES);
    std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
    std::vector<unsigned int> meshletVertices(maxMeshlets * MAX_VERTICES);
    std::vector<unsigned char> meshletTriangles(maxMeshlets * MAX_TRIANGLES * 3);

    // Build meshlets
    size_t meshletCount = meshopt_buildMeshlets(
        meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
        indices, indexCount,
        positions, vertexCount, vertexStride,
        MAX_VERTICES, MAX_TRIANGLES, CONE_WEIGHT);

    // Trim to actual size
    meshlets.resize(meshletCount);

    // Optimize each meshlet for rasterizer throughput
    for (size_t i = 0; i < meshletCount; i++) {
        meshopt_optimizeMeshlet(
            &meshletVertices[meshlets[i].vertex_offset],
            &meshletTriangles[meshlets[i].triangle_offset],
            meshlets[i].triangle_count,
            meshlets[i].vertex_count);
    }

    // Compute actual used sizes for vertex and triangle arrays
    const meshopt_Meshlet& last = meshlets.back();
    size_t totalVertices = last.vertex_offset + last.vertex_count;
    size_t totalTriangles = last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3); // aligned

    // Pack uint8_t triangle indices into uint32_t array
    // Each uint32_t holds one triangle: bytes [v0, v1, v2, 0]
    size_t totalTriangleIndices = last.triangle_offset + last.triangle_count * 3;
    size_t packedTriangleCount = 0;
    for (size_t i = 0; i < meshletCount; i++)
        packedTriangleCount += meshlets[i].triangle_count;

    std::vector<uint32_t> packedTriangles(packedTriangleCount);
    size_t packedOffset = 0;
    for (size_t i = 0; i < meshletCount; i++) {
        const meshopt_Meshlet& m = meshlets[i];
        for (size_t t = 0; t < m.triangle_count; t++) {
            size_t srcIdx = m.triangle_offset + t * 3;
            uint32_t v0 = meshletTriangles[srcIdx + 0];
            uint32_t v1 = meshletTriangles[srcIdx + 1];
            uint32_t v2 = meshletTriangles[srcIdx + 2];
            packedTriangles[packedOffset++] = v0 | (v1 << 8) | (v2 << 16);
        }
    }

    // Build GPU meshlet descriptors with updated triangle offsets (now in uint32_t units)
    std::vector<GPUMeshlet> gpuMeshlets(meshletCount);
    size_t triOffset = 0;
    for (size_t i = 0; i < meshletCount; i++) {
        gpuMeshlets[i].vertex_offset   = meshlets[i].vertex_offset;
        gpuMeshlets[i].triangle_offset = static_cast<uint32_t>(triOffset);
        gpuMeshlets[i].vertex_count    = meshlets[i].vertex_count;
        gpuMeshlets[i].triangle_count  = meshlets[i].triangle_count;
        triOffset += meshlets[i].triangle_count;
    }

    // Create Metal buffers
    out.meshletBuffer = device->newBuffer(
        gpuMeshlets.data(), gpuMeshlets.size() * sizeof(GPUMeshlet),
        MTL::ResourceStorageModeShared);
    out.meshletVertices = device->newBuffer(
        meshletVertices.data(), totalVertices * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    out.meshletTriangles = device->newBuffer(
        packedTriangles.data(), packedTriangles.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    out.meshletCount = static_cast<uint32_t>(meshletCount);

    // Print stats
    size_t totalTris = 0, totalVerts = 0;
    for (size_t i = 0; i < meshletCount; i++) {
        totalTris += meshlets[i].triangle_count;
        totalVerts += meshlets[i].vertex_count;
    }
    std::cout << "Built " << meshletCount << " meshlets"
              << " (avg " << (totalVerts / meshletCount) << " verts, "
              << (totalTris / meshletCount) << " tris per meshlet)" << std::endl;

    return true;
}
