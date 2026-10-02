#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using SampleUInt = uint32_t;
using SampleOutput = ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias SampleUInt = uint;
typealias SampleOutput = DescriptorHandle<RWTexture2D<float4>>;
#endif
struct RTXCRMaterialSamplePush
{
    SampleUInt width;
    SampleUInt height;
    SampleUInt viewMode;
    SampleUInt padding0;

    float exposure;
    float lightAzimuthDegrees;
    float hairMelanin;
    float hairMelaninRedness;

    float hairLongitudinalRoughness;
    float hairAzimuthalRoughness;
    float hairCuticleAngleDegrees;
    float hairIor;

    float sssScale;
    float sssAnisotropy;
    float sssMaxSampleRadius;
    float padding1;
};
struct RTXCRMaterialSampleParams {
    SampleOutput output;
    RTXCRMaterialSamplePush settings;
};
#ifdef __cplusplus
inline constexpr uint64_t kRTXCRMaterialSampleABI = 0x5254584352530001ull;
static_assert(sizeof(RTXCRMaterialSamplePush) == 64);
static_assert(sizeof(RTXCRMaterialSampleParams) == 72 && offsetof(RTXCRMaterialSampleParams, settings) == 8);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
