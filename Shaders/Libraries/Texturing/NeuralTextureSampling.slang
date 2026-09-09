#ifndef METALLIC_HAS_NTC
#define METALLIC_HAS_NTC 0
#endif
#ifndef METALLIC_NTC_COOPERATIVE_VECTOR
#define METALLIC_NTC_COOPERATIVE_VECTOR 0
#endif

#if METALLIC_HAS_NTC

#if METALLIC_NTC_COOPERATIVE_VECTOR
#include <libntc/shaders/InferenceCoopVec.hlsli>
#else
#include <libntc/shaders/Inference.hlsli>
#endif

static const uint kMaxNeuralTextureSets = 64u;
static const uint kInvalidNeuralTextureSetIndex = 0xffffffffu;

[[vk::binding(29, 0)]]
Texture2DArray<float4> gNtcLatentTextures[kMaxNeuralTextureSets];

[[vk::binding(30, 0)]]
StructuredBuffer<NtcTextureSetConstants> gNtcTextureSetConstants;

[[vk::binding(31, 0)]]
ByteAddressBuffer gNtcWeights;

// x is the byte offset of this texture set's weights in gNtcWeights.
// y is 1 for converted CoopVec FP8 weights and 0 for Generic INT8 weights.
[[vk::binding(32, 0)]]
StructuredBuffer<uint4> gNtcTextureSetInfo;

[[vk::binding(33, 0)]]
SamplerState gNtcLatentSampler;

bool metallicHasNeuralTexture(uint textureSetIndex, uint textureSetCount)
{
    return textureSetIndex != kInvalidNeuralTextureSetIndex &&
        textureSetIndex < textureSetCount &&
        textureSetIndex < kMaxNeuralTextureSets;
}

bool metallicNeuralTextureDimensions(
    uint textureSetIndex,
    uint textureSetCount,
    uint mipLevel,
    out uint2 dimensions,
    out uint mipCount)
{
    dimensions = uint2(1u, 1u);
    mipCount = 1u;
    if (!metallicHasNeuralTexture(textureSetIndex, textureSetCount)) {
        return false;
    }

    NtcTextureSetConstants constants = gNtcTextureSetConstants[textureSetIndex];
    mipCount = uint(max(NtcGetTextureMipLevels(constants), 1));
    uint clampedMip = min(mipLevel, mipCount - 1u);
    dimensions = uint2(max(NtcGetTextureDimensions(constants, int(clampedMip)), int2(1, 1)));
    return true;
}

float4 metallicSampleNeuralTexture(
    uint textureSetIndex,
    uint textureSetCount,
    uint packedChannelMapping,
    int2 texel,
    int mipLevel,
    float4 fallback)
{
    if (!metallicHasNeuralTexture(textureSetIndex, textureSetCount)) {
        return fallback;
    }

    float channels[NTC_MLP_OUTPUT_CHANNELS];
    [unroll]
    for (uint channelIndex = 0u; channelIndex < NTC_MLP_OUTPUT_CHANNELS; ++channelIndex) {
        channels[channelIndex] = 0.0;
    }

    NtcTextureSetConstants constants = gNtcTextureSetConstants[textureSetIndex];
    bool sampled = false;
#if METALLIC_NTC_COOPERATIVE_VECTOR
    if (gNtcTextureSetInfo[textureSetIndex].y != 0u) {
        sampled = NtcSampleTextureSet_CoopVec_FP8(
            constants,
            gNtcLatentTextures[textureSetIndex],
            gNtcLatentSampler,
            gNtcWeights,
            gNtcTextureSetInfo[textureSetIndex].x,
            texel,
            mipLevel,
            false,
            channels);
    } else
#endif
    {
        sampled = NtcSampleTextureSet(
            constants,
            gNtcLatentTextures[textureSetIndex],
            gNtcLatentSampler,
            gNtcWeights,
            gNtcTextureSetInfo[textureSetIndex].x,
            texel,
            mipLevel,
            false,
            channels);
    }
    if (!sampled) {
        return fallback;
    }

    float4 result = fallback;
    [unroll]
    for (uint componentIndex = 0u; componentIndex < 4u; ++componentIndex) {
        uint channel = (packedChannelMapping >> (componentIndex * 8u)) & 0xffu;
        if (channel < NTC_MLP_OUTPUT_CHANNELS) {
            result[componentIndex] = channels[channel];
        }
    }
    return result;
}

#else

static const uint kInvalidNeuralTextureSetIndex = 0xffffffffu;

bool metallicHasNeuralTexture(uint textureSetIndex, uint textureSetCount)
{
    return false;
}

bool metallicNeuralTextureDimensions(
    uint textureSetIndex,
    uint textureSetCount,
    uint mipLevel,
    out uint2 dimensions,
    out uint mipCount)
{
    dimensions = uint2(1u, 1u);
    mipCount = 1u;
    return false;
}

float4 metallicSampleNeuralTexture(
    uint textureSetIndex,
    uint textureSetCount,
    uint packedChannelMapping,
    int2 texel,
    int mipLevel,
    float4 fallback)
{
    return fallback;
}

#endif
