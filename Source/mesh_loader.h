#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <cstdint>

struct LoadedMesh {
    MTL::Buffer* positionBuffer = nullptr;
    MTL::Buffer* normalBuffer   = nullptr;
    MTL::Buffer* uvBuffer       = nullptr;
    MTL::Buffer* indexBuffer    = nullptr;
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
};

bool loadGLTFMesh(MTL::Device* device, const std::string& gltfPath, LoadedMesh& out);
