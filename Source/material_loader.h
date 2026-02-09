#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <cstdint>

static constexpr uint32_t INVALID_TEXTURE_INDEX = 0xFFFFFFFF;

struct GPUMaterial {
    uint32_t baseColorTexIndex;
    uint32_t normalTexIndex;
    uint32_t metallicRoughnessTexIndex;
    uint32_t alphaMode; // 0=OPAQUE, 1=MASK

    float baseColorFactor[4];
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float _pad;
};
static_assert(sizeof(GPUMaterial) == 48, "GPUMaterial must be 48 bytes");

struct LoadedMaterials {
    std::vector<MTL::Texture*> textures;
    MTL::Buffer* materialBuffer = nullptr;
    MTL::SamplerState* sampler = nullptr;
    uint32_t materialCount = 0;
};

bool loadGLTFMaterials(MTL::Device* device, MTL::CommandQueue* commandQueue,
                       const std::string& gltfPath, LoadedMaterials& out);
