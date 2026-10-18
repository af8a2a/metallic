#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render {

// Scene upload transport shared by PT, VBuffer and streaming consumers.
// Shader models extract their own parameter types; this record is not a
// material interface and must not grow to hold additional material models.
struct SceneMaterialData
{
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4] = {};
    float params[4] = {};
    // z: stable MaterialProgramId, normalized before GPU publication.
    float textureParams[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    float glassParams[4] = {0.0f, 1.5f, 0.0f, 0.0f};
    float attenuationColor[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float diffuseTransmission[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    float rtxcrHairBaseColor[4] = {0.2f, 0.2f, 0.2f, 0.0f};
    float rtxcrHairParams0[4] = {0.3f, 0.3f, 1.55f, 3.0f};
    float rtxcrHairParams1[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float rtxcrHairDiffuseTint[4] = {};
    struct TextureInfo
    {
        uint32_t textureIndex = UINT32_MAX;
        uint32_t texCoord = 0;
        // Immutable source LOD metadata occupies the former alignment padding.
        // Nonnegative: exact power-of-two LOD; negative: source area for GPU log2.
        float sourceLodTerm = 0.0f;
        uint32_t sourceMipCount = 1;
        float transform0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float transform1[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    };
    TextureInfo baseColorTexture;
    TextureInfo metallicRoughnessTexture;
    TextureInfo normalTexture;
    TextureInfo occlusionTexture;
    TextureInfo emissiveTexture;
    TextureInfo transmissionTexture;
    TextureInfo thicknessTexture;
    TextureInfo diffuseTransmissionTexture;
    TextureInfo diffuseTransmissionColorTexture;
    float specular[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    TextureInfo specularTexture;
    TextureInfo specularColorTexture;
};

static_assert(std::is_trivially_copyable_v<SceneMaterialData>);
// Match Slang StructuredBuffer std430 offsets, including required float4 alignment.
static_assert(sizeof(SceneMaterialData::TextureInfo) == 48);
static_assert(offsetof(SceneMaterialData::TextureInfo, sourceLodTerm) == 8);
static_assert(offsetof(SceneMaterialData::TextureInfo, sourceMipCount) == 12);
static_assert(offsetof(SceneMaterialData::TextureInfo, transform0) == 16);
static_assert(offsetof(SceneMaterialData::TextureInfo, transform1) == 32);
static_assert(sizeof(SceneMaterialData) == 720);
static_assert(offsetof(SceneMaterialData, baseColorTexture) == 176);
static_assert(offsetof(SceneMaterialData, specular) == 608);

} // namespace metallic::render
