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
    void* meshletBuffer = nullptr;
    RhiBufferHandle meshletBufferRhi;
    void* meshletVertices = nullptr;
    void* meshletTriangles = nullptr;
    void* boundsBuffer = nullptr;
    void* materialIDs = nullptr;
    RhiBufferHandle meshletVerticesRhi;
    RhiBufferHandle meshletTrianglesRhi;
    RhiBufferHandle boundsBufferRhi;
    RhiBufferHandle materialIDsRhi;
    uint32_t meshletCount = 0;
    std::vector<uint32_t> meshletsPerGroup;
};

bool buildMeshlets(void* deviceHandle, const LoadedMesh& mesh, MeshletData& out);
