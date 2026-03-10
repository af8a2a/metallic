#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "rhi_backend.h"

static constexpr uint32_t INVALID_TEXTURE_INDEX = 0xFFFFFFFF;
static constexpr uint32_t MAX_SCENE_TEXTURES = 96;

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
    std::vector<RhiTextureHandle> textures;
    std::vector<const RhiTexture*> textureViews;
    RhiBufferHandle materialBuffer;
    RhiSamplerHandle sampler;
    uint32_t materialCount = 0;
};

bool loadGLTFMaterials(const RhiDevice& device, const RhiCommandQueue& commandQueue,
                       const std::string& gltfPath, LoadedMaterials& out);
