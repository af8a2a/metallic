#include "Source/RHI/bindless_scene_constants.h"

static const uint kBindlessSceneTextureCount = METALLIC_BINDLESS_MAX_SAMPLED_IMAGES;
static const uint kBindlessSceneSamplerCount = METALLIC_BINDLESS_MAX_SAMPLERS;
static const uint kBindlessSceneSamplerIndex = METALLIC_BINDLESS_SCENE_SAMPLER_INDEX;

[[vk::binding(METALLIC_BINDLESS_SAMPLED_IMAGE_BINDING, METALLIC_BINDLESS_SET)]]
Texture2D bindlessSceneTextures[kBindlessSceneTextureCount];

[[vk::binding(METALLIC_BINDLESS_SAMPLER_BINDING, METALLIC_BINDLESS_SET)]]
SamplerState bindlessSceneSamplers[kBindlessSceneSamplerCount];

float4 sampleBindlessSceneTexture(uint textureIndex, float2 uv) {
    return bindlessSceneTextures[textureIndex].Sample(bindlessSceneSamplers[kBindlessSceneSamplerIndex], uv);
}

float4 sampleBindlessSceneTextureGrad(uint textureIndex,
                                      float2 uv,
                                      float2 duvdx,
                                      float2 duvdy) {
    return bindlessSceneTextures[textureIndex].SampleGrad(bindlessSceneSamplers[kBindlessSceneSamplerIndex],
                                                          uv,
                                                          duvdx,
                                                          duvdy);
}
