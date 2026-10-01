#pragma once

// One CPU/Slang declaration for the RTXDI denoising and composition parameters.
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using RTXDIUInt = uint32_t;
using RTXDIStorage4 = ShaderStorageImage;
using RTXDIStorage2 = ShaderStorageImage;
using RTXDIStorage1 = ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias RTXDIUInt = uint;
typealias RTXDIStorage4 = DescriptorHandle<RWTexture2D<float4>>;
typealias RTXDIStorage2 = DescriptorHandle<RWTexture2D<float2>>;
typealias RTXDIStorage1 = DescriptorHandle<RWTexture2D<float>>;
#endif

struct RTXDIConfidencePush
{
    RTXDIUInt mode;
    RTXDIUInt width;
    RTXDIUInt height;
    RTXDIUInt gradientWidth;
    RTXDIUInt gradientHeight;
    RTXDIUInt hasHistory;
    RTXDIUInt filterStep;
    RTXDIUInt padding0;
    float darknessBias;
    float sensitivity;
    float blendFactor;
    float padding1;
};

struct RTXDIConfidenceParams {
    RTXDIStorage4 noisyDiffuse;
    RTXDIStorage4 noisySpecular;
    RTXDIStorage4 baseColorMetalness;
    RTXDIStorage4 motionVectors;
    RTXDIStorage2 previousLuminance;
    RTXDIStorage2 currentLuminance;
    RTXDIStorage4 gradientA;
    RTXDIStorage4 gradientB;
    RTXDIStorage1 previousDiffuseConfidence;
    RTXDIStorage1 previousSpecularConfidence;
    RTXDIStorage1 diffuseConfidence;
    RTXDIStorage1 specularConfidence;
    RTXDIStorage1 currentDiffuseConfidence;
    RTXDIStorage1 currentSpecularConfidence;
    RTXDIConfidencePush settings;
};

struct RTXDICompositePush
{
    RTXDIUInt width;
    RTXDIUInt height;
    float exposure;
    RTXDIUInt outputLinear;
};

struct RTXDICompositeParams {
    RTXDIStorage4 denoisedDiffuse;
    RTXDIStorage4 denoisedSpecular;
    RTXDIStorage4 baseColorMetalness;
    RTXDIStorage4 emissive;
    RTXDIStorage4 output;
    RTXDICompositePush settings;
};

#ifdef __cplusplus
inline constexpr uint64_t kRTXDIConfidenceABI = 0x5254434f4e460001ull;
inline constexpr uint64_t kRTXDICompositeABI = 0x5254434f4d500001ull;
static_assert(sizeof(RTXDIConfidencePush) == 48);
static_assert(sizeof(RTXDICompositePush) == 16);
static_assert(sizeof(RTXDIConfidenceParams) == 160 && offsetof(RTXDIConfidenceParams, settings) == 112);
static_assert(sizeof(RTXDICompositeParams) == 56 && offsetof(RTXDICompositeParams, settings) == 40);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
