#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>
#include <vector>

struct LoadedMesh;

struct GPUMeshlet {
    uint32_t vertex_offset;
    uint32_t triangle_offset;
    uint32_t vertex_count;
    uint32_t triangle_count;
};

struct GPUMeshletBounds {
    float center_radius[4];    // xyz=center, w=radius
    float cone_apex_pad[4];    // xyz=cone_apex, w=unused
    float cone_axis_cutoff[4]; // xyz=cone_axis, w=cone_cutoff
};

static_assert(sizeof(GPUMeshletBounds) == 48, "GPUMeshletBounds must match shader layout");

struct MeshletData {
    MTL::Buffer* meshletBuffer    = nullptr;  // GPUMeshlet[]
    MTL::Buffer* meshletVertices  = nullptr;  // uint32_t[] — indices into original vertex buffer
    MTL::Buffer* meshletTriangles = nullptr;  // uint32_t[] — packed local triangle indices (3 per uint32)
    MTL::Buffer* boundsBuffer     = nullptr;  // GPUMeshletBounds[]
    MTL::Buffer* materialIDs      = nullptr;  // uint32_t[] — material index per meshlet
    uint32_t meshletCount = 0;
    std::vector<uint32_t> meshletsPerGroup; // meshlet count per primitive group
};

bool buildMeshlets(MTL::Device* device, const LoadedMesh& mesh, MeshletData& out);
