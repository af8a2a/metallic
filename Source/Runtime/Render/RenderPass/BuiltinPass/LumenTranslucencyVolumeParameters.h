#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/RenderView.h"
#include "LumenGlobalTraceBindings.h"
#include "LumenSky.h"

namespace metallic::render::builtin_pass {
using VolumeTraceBindings = ShaderLumenGlobalTraceBindings;
using VolumeFloat3 = std::array<float, 3>;
using VolumeUInt4 = std::array<uint32_t, 4>;
using VolumeDepth = ShaderSampledImage;
using VolumeDepthOut = ShaderStorageImage;
using VolumeNoise = ShaderSampledImage;
using VolumeRGB = ShaderSampledImage;
using VolumeRGBOut = ShaderStorageImage;
#else
namespace Metallic.Lumen {
typealias VolumeTraceBindings = LumenGlobalTraceBindings;
typealias VolumeFloat3 = float3;
typealias VolumeUInt4 = uint4;
typealias VolumeDepth = ResourceHandle<Texture2D<float>>;
typealias VolumeDepthOut = ResourceHandle<RWTexture2D<float>>;
typealias VolumeNoise = ResourceHandle<Texture2D<float2>>;
typealias VolumeRGB = ResourceHandle<Texture3D<float4>>;
typealias VolumeRGBOut = ResourceHandle<RWTexture3D<float4>>;
#endif

struct LumenTranslucencyVolumeParameters {
    ViewCameraConstants camera;
    VolumeUInt4 grid; // active XYZ, pixel size
    VolumeUInt4 tracing; // clipmaps, blue-noise frame, allocated slices, spatial filter enabled
    VolumeFloat3 frameJitter;
    float maxRayIntensity;
    LumenSkyParameters sky;
    VolumeTraceBindings scene;
    VolumeNoise blueNoise;
    VolumeDepth depth, maxVisibleDepth;
    VolumeDepthOut maxVisibleDepthOut;
    VolumeRGB trace, filtered;
    VolumeRGBOut traceOut, filteredOut, ambientOut, directionalOut;
};

#ifdef __cplusplus
constexpr uint64_t kLumenTranslucencyVolumeABI = 0x4c54564f00000002ull;
static_assert(offsetof(LumenTranslucencyVolumeParameters, maxRayIntensity) == 124);
static_assert(offsetof(LumenTranslucencyVolumeParameters, scene) == 144);
static_assert(offsetof(LumenTranslucencyVolumeParameters, directionalOut) == 224);
static_assert(sizeof(LumenTranslucencyVolumeParameters) == 240);
#endif
} // namespace metallic::render::builtin_pass (C++) / Metallic.Lumen (Slang)
