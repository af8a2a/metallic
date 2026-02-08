#pragma once

#include <Metal/Metal.hpp>
#include <string>

struct LoadedMesh {
    MTL::Buffer* positionBuffer = nullptr;
    MTL::Buffer* normalBuffer   = nullptr;
    MTL::Buffer* indexBuffer    = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount  = 0;
    float bboxMin[3] = {};
    float bboxMax[3] = {};
};

bool loadGLTFMesh(MTL::Device* device, const std::string& gltfPath, LoadedMesh& out);
