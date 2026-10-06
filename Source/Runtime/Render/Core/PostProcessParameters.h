#pragma once

// One field declaration for C++ and Slang. Explicit padding keeps nested vector
// data aligned for both physical-storage parameters and inline push data.
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ColorGrading.h"
#include <cstddef>
namespace metallic::render {
using PostUInt = uint32_t;
using PostSampled2D = GPUResourceHandle<ResourceViewKind::SampledImage>;
using PostSampled2DScalar = PostSampled2D;
using PostSampled3D = PostSampled2D;
using PostStorage2D = GPUResourceHandle<ResourceViewKind::StorageImage>;
using PostStorage3D = PostStorage2D;
using PostSampler = GPUSamplerHandle;
using PostUIntData = GPUBufferSpan;
using PostFloat4Data = GPUBufferSpan;
#else
import ShaderCore;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;
namespace Metallic {
typealias PostUInt = uint;
typealias PostSampled2D = ResourceHandle<Texture2D<float4>>;
typealias PostSampled2DScalar = ResourceHandle<Texture2D<float>>;
typealias PostSampled3D = ResourceHandle<Texture3D<float4>>;
typealias PostStorage2D = ResourceHandle<RWTexture2D<float4>>;
typealias PostStorage3D = ResourceHandle<RWTexture3D<float4>>;
typealias PostSampler = SamplerHandle;
typealias PostUIntData = RWBufferSpan<uint>;
typealias PostFloat4Data = RWBufferSpan<float4>;
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
    PostUInt bypass;
};

struct AutoExposureParams {
    PostSampled2D source;
    PostStorage2D output;
    PostUIntData histogram;
    PostFloat4Data history, exposure;
    AutoExposurePush display;
    PostUInt padding;
};

struct GradingPush {
    PostUInt transform, hdr;
    float peak, paperWhite;
    ColorGradingParameters grade;
};

#ifdef __cplusplus
struct alignas(16) ColorGradingLUTParams {
#else
struct ColorGradingLUTParams {
#endif
    PostStorage3D output;
    PostSampled2D custom0, custom1, custom2, custom3;
    PostSampled2DScalar reach, gamut, gammaTable;
    PostSampler sampler;
    PostUInt padding0, padding1, padding2;
    GradingPush display;
};

#ifdef __cplusplus
inline constexpr uint64_t kFinalBlitABI = 0x46424c4954000002ull;
inline constexpr uint64_t kSliderDebugABI = 0x534c494445000002ull;
inline constexpr uint64_t kAutoExposureABI = 0x4558504f53000003ull;
inline constexpr uint64_t kColorGradingLUTABI = 0x4752414445000002ull;
static_assert(sizeof(FinalBlitParams) == 56 && offsetof(FinalBlitParams, display) == 16);
static_assert(sizeof(SliderDebugParams) == 28 && offsetof(SliderDebugParams, display) == 12);
static_assert(sizeof(AutoExposureParams) == 128 && offsetof(AutoExposureParams, display) == 44);
static_assert(alignof(ColorGradingLUTParams) == 16);
static_assert(sizeof(ColorGradingLUTParams) == 192 && offsetof(ColorGradingLUTParams, display) == 48);
static_assert(offsetof(ColorGradingLUTParams, sampler) == 32 && offsetof(GradingPush, grade) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
