#pragma once

// One field declaration for C++ and Slang. Explicit padding keeps nested vector
// data aligned for both physical-storage parameters and inline push data.
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ColorGrading.h"
#include <cstddef>
namespace metallic::render {
using PostUInt = uint32_t;
using PostSampled2D = ShaderSampledImage;
using PostSampled2DScalar = ShaderSampledImage;
using PostSampled2DMotion = ShaderSampledImage;
using PostStorage2DScalar = ShaderStorageImage;
using PostStorage2DMotion = ShaderStorageImage;
using PostSampled3D = ShaderSampledImage;
using PostStorage2D = ShaderStorageImage;
using PostStorage3D = ShaderStorageImage;
using PostSampler = ShaderSampler;
using PostUIntData = ShaderDataSpan;
using PostFloat4Data = ShaderDataSpan;
#else
import ShaderCore;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;
namespace Metallic {
typealias PostUInt = uint;
typealias PostSampled2D = DescriptorHandle<Texture2D<float4>>;
typealias PostSampled2DScalar = DescriptorHandle<Texture2D<float>>;
typealias PostSampled2DMotion = DescriptorHandle<Texture2D<float2>>;
typealias PostStorage2DScalar = DescriptorHandle<RWTexture2D<float>>;
typealias PostStorage2DMotion = DescriptorHandle<RWTexture2D<float2>>;
typealias PostSampled3D = DescriptorHandle<Texture3D<float4>>;
typealias PostStorage2D = DescriptorHandle<RWTexture2D<float4>>;
typealias PostStorage3D = DescriptorHandle<RWTexture3D<float4>>;
typealias PostSampler = DescriptorHandle<SamplerState>;
typealias PostUIntData = DataSpan<uint>;
typealias PostFloat4Data = DataSpan<float4>;
#endif

struct DisplayOutputPush {
    PostUInt hdr, inputEncoding, calibration, sampledSrgb;
    float paperWhiteNits, peakNits, exposure;
    PostUInt toneCurve, hasLut;
};

struct FinalBlitParams {
    PostStorage2D output;
    PostSampled2D source;
    PostSampled3D lut;
    PostSampler lutSampler;
    DisplayOutputPush display;
    PostUInt padding;
};

struct SliderDebugPush {
    float splitPosition;
    PostUInt horizontal, swapSides;
};

struct SliderDebugParams {
    PostSampled2D sourceA, sourceB;
    PostStorage2D output;
    SliderDebugPush display;
    PostUInt padding;
};

struct AutoExposurePush {
    PostUInt width, height, tileCount, resetHistory;
    float minEV100, maxEV100, compensation, manualEV100;
    float lowPercent, highPercent, histogramMin, histogramMax;
    float speedUp, speedDown, transitionDistance, deltaSeconds;
    PostUInt automatic;
    float sourceExposure, artisticExposure;
};

struct AutoExposureParams {
    PostSampled2D source;
    PostStorage2D output;
    PostUIntData histogram;
    PostFloat4Data history, exposure;
    AutoExposurePush display;
    PostUInt padding;
};

struct DLSSSupportParams {
    PostSampled2DScalar depth;
    PostStorage2D color;
};

struct UpscalerGuideResolveParams {
    PostSampled2DScalar depth;
    PostSampled2DMotion motion;
    PostStorage2DScalar outputDepth;
    PostStorage2DMotion outputMotion;
    float jitterX, jitterY;
};

struct GradingPush {
    PostUInt transform, hdr;
    float peak, paperWhite;
    ColorGradingParameters grade;
};

#ifdef __cplusplus
using GradingSettings = uint64_t;
#else
typealias GradingSettings = GradingPush*;
#endif
struct ColorGradingLUTParams {
    PostStorage3D output;
    PostSampled2D custom0, custom1, custom2, custom3;
    PostSampled2DScalar reach, gamut, gammaTable;
    PostSampler sampler;
    GradingSettings display;
};

#ifdef __cplusplus
inline constexpr uint64_t kDLSSSupportABI = 0x444c535353550001ull;
static_assert(sizeof(DLSSSupportParams) == 16 && offsetof(DLSSSupportParams, color) == 8);
inline constexpr uint64_t kUpscalerGuideResolveABI = 0x5550475549440001ull;
static_assert(sizeof(UpscalerGuideResolveParams) == 40 && offsetof(UpscalerGuideResolveParams, jitterX) == 32);
inline constexpr uint64_t kFinalBlitABI = 0x46424c4954000001ull;
inline constexpr uint64_t kSliderDebugABI = 0x534c494445000001ull;
inline constexpr uint64_t kAutoExposureABI = 0x4558504f53000001ull;
inline constexpr uint64_t kColorGradingLUTABI = 0x4752414445000002ull;
static_assert(sizeof(FinalBlitParams) == 72 && offsetof(FinalBlitParams, display) == 32);
static_assert(sizeof(SliderDebugParams) == 40 && offsetof(SliderDebugParams, display) == 24);
static_assert(sizeof(AutoExposureParams) == 144 && offsetof(AutoExposureParams, display) == 64);
static_assert(alignof(ColorGradingLUTParams) == 8);
static_assert(sizeof(ColorGradingLUTParams) == 80 && offsetof(ColorGradingLUTParams, display) == 72);
static_assert(offsetof(ColorGradingLUTParams, sampler) == 64 && offsetof(GradingPush, grade) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
