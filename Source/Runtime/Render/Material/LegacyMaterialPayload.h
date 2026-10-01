#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render {

// Compatibility upload ABI shared by existing PT, VBuffer and stream consumers.
// New models must define their own schema instead of adding fields here.
struct LegacyMaterialPayload
{
    float baseColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float emissive[4] = {};
    float params[4] = {};
    // z: 0 = legacy inference, otherwise the stable MaterialProgramId.
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
        uint32_t ntcTextureSetIndex = UINT32_MAX;
        uint32_t ntcChannelMapping = UINT32_MAX;
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

static_assert(std::is_trivially_copyable_v<LegacyMaterialPayload>);
static_assert(sizeof(LegacyMaterialPayload::TextureInfo) == 48);
static_assert(sizeof(LegacyMaterialPayload) == 720);
static_assert(offsetof(LegacyMaterialPayload, baseColorTexture) == 176);
static_assert(offsetof(LegacyMaterialPayload, specular) == 608);

} // namespace metallic::render
