#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "rhi_backend.h"

struct LoadedMesh {
    RhiBufferHandle positionBuffer;
    RhiBufferHandle normalBuffer;
    RhiBufferHandle uvBuffer;
    RhiBufferHandle indexBuffer;
    uint32_t vertexCount = 0;
    uint32_t indexCount  = 0;
    float bboxMin[3] = {};
    float bboxMax[3] = {};

    struct PrimitiveGroup {
        uint32_t indexOffset;
        uint32_t indexCount;
        uint32_t vertexOffset;
        uint32_t vertexCount;
        uint32_t materialIndex;
    };
    std::vector<PrimitiveGroup> primitiveGroups;

    // Per-glTF-mesh range into primitiveGroups (indexed by glTF mesh index)
    struct MeshPrimitiveRange {
        uint32_t firstGroup;
        uint32_t groupCount;
    };
    std::vector<MeshPrimitiveRange> meshRanges;
};

bool loadGLTFMesh(const RhiDevice& device, const std::string& gltfPath, LoadedMesh& out);
