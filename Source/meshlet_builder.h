#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

struct LoadedMesh;

struct GPUMeshlet {
    uint32_t vertex_offset;
    uint32_t triangle_offset;
    uint32_t vertex_count;
    uint32_t triangle_count;
};

struct GPUMeshletBounds {
    float center[3];
    float radius;
    float cone_apex[3];
    float pad0;
    float cone_axis[3];
    float cone_cutoff;
};

struct MeshletData {
    MTL::Buffer* meshletBuffer    = nullptr;  // GPUMeshlet[]
    MTL::Buffer* meshletVertices  = nullptr;  // uint32_t[] — indices into original vertex buffer
    MTL::Buffer* meshletTriangles = nullptr;  // uint32_t[] — packed local triangle indices (3 per uint32)
    MTL::Buffer* boundsBuffer     = nullptr;  // GPUMeshletBounds[]
    uint32_t meshletCount = 0;
};

bool buildMeshlets(MTL::Device* device, const LoadedMesh& mesh, MeshletData& out);
