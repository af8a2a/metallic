#pragma once

#include <cstdint>
#include <vector>

#include "rhi_backend.h"

struct LoadedMesh;

struct GPUMeshlet {
    uint32_t vertex_offset;
    uint32_t triangle_offset;
    uint32_t vertex_count;
    uint32_t triangle_count;
};

struct GPUMeshletBounds {
    float center_radius[4];
    float cone_apex_pad[4];
    float cone_axis_cutoff[4];
};

static_assert(sizeof(GPUMeshletBounds) == 48, "GPUMeshletBounds must match shader layout");

struct MeshletData {
    RhiBufferHandle meshletBuffer;
    RhiBufferHandle meshletVertices;
    RhiBufferHandle meshletTriangles;
    RhiBufferHandle boundsBuffer;
    RhiBufferHandle materialIDs;
    uint32_t meshletCount = 0;
    std::vector<uint32_t> meshletsPerGroup;

    // CPU-side data retained for LOD building
    std::vector<GPUMeshlet>       cpuMeshlets;
    std::vector<unsigned int>     cpuMeshletVertices;
    std::vector<unsigned char>    cpuMeshletTriangles;  // raw 3-byte-per-tri format
    std::vector<GPUMeshletBounds> cpuBounds;
    std::vector<uint32_t>         cpuMaterialIDs;
};

bool buildMeshlets(const RhiDevice& device, const LoadedMesh& mesh, MeshletData& out);
