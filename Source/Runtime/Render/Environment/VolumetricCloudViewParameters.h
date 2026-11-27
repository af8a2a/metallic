#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>

namespace metallic::render {
using CloudViewFloat2 = std::array<float, 2>;
using CloudViewUInt2 = std::array<uint32_t, 2>;
using CloudViewUInt = uint32_t;
using CloudViewColorIn = ShaderSampledImage;
using CloudViewScalarIn = ShaderSampledImage;
using CloudViewColorOut = ShaderStorageImage;
using CloudViewDepthPairIn = ShaderSampledImage;
using CloudViewDepthPairOut = ShaderStorageImage;
using CloudViewScalarOut = ShaderStorageImage;
using CloudViewAtmosphere = ShaderBuffer;
using CloudViewBuffer = ShaderBuffer;
#else
import ShaderCore;
import AtmosphereTypes;
using Metallic;
namespace Metallic {
typealias CloudViewFloat2 = float2;
typealias CloudViewUInt2 = uint2;
typealias CloudViewUInt = uint;
typealias CloudViewColorIn = ResourceHandle<Texture2D<float4>>;
typealias CloudViewScalarIn = ResourceHandle<Texture2D<float>>;
typealias CloudViewColorOut = ResourceHandle<RWTexture2D<float4>>;
typealias CloudViewDepthPairIn = ResourceHandle<Texture2D<float2>>;
typealias CloudViewDepthPairOut = ResourceHandle<RWTexture2D<float2>>;
typealias CloudViewScalarOut = ResourceHandle<RWTexture2D<float>>;
typealias CloudViewAtmosphere = ResourceHandle<StructuredBuffer<GPUAtmosphereParameters>>;
typealias CloudViewBuffer = ResourceHandle<ByteAddressBuffer>;
#endif

struct VolumetricCloudViewParams {
    ViewCameraConstants camera;
    CloudViewFloat2 jitter;
    CloudViewColorIn sceneColor;
    CloudViewScalarIn depth;
    CloudViewAtmosphere atmosphereParameters;
    CloudViewColorIn transmittanceLut, multiScatteringLut;
    CloudViewBuffer aerialPerspective, worldShadowFaces;
    CloudViewUInt shadowMask;
    CloudViewScalarIn blueNoise;
    CloudViewColorOut color, scattering;
    CloudViewScalarOut transmittance, cloudDepth;
    CloudViewUInt skyVisible;
    ViewCameraConstants previousCamera;
    CloudViewUInt2 tracingPixelOffset;
    CloudViewUInt noiseFrameIndex;
    float invDistanceToSampleCountMax; // Reciprocal metres.
    CloudViewColorIn currentFar, currentNear, currentDepth;
    CloudViewColorIn previousFar, previousNear, previousDepth;
    CloudViewColorOut writeFar, writeNear, writeDepth;
    CloudViewDepthPairIn currentRepresentativeDepth, previousRepresentativeDepth;
    CloudViewDepthPairOut writeRepresentativeDepth;
    CloudViewUInt viewRaySampleMaxCount, shadowViewRaySampleMaxCount;
};

#ifdef __cplusplus
inline constexpr uint64_t kVolumetricCloudViewABI = 0x56434c4400000004ull;
static_assert(offsetof(VolumetricCloudViewParams, sceneColor) == 88);
static_assert(offsetof(VolumetricCloudViewParams, cloudDepth) == 136);
static_assert(offsetof(VolumetricCloudViewParams, previousCamera) == 144);
static_assert(offsetof(VolumetricCloudViewParams, invDistanceToSampleCountMax) == 236);
static_assert(offsetof(VolumetricCloudViewParams, writeFar) == 264);
static_assert(offsetof(VolumetricCloudViewParams, writeRepresentativeDepth) == 284);
static_assert(offsetof(VolumetricCloudViewParams, viewRaySampleMaxCount) == 288);
static_assert(offsetof(VolumetricCloudViewParams, shadowViewRaySampleMaxCount) == 292);
static_assert(sizeof(VolumetricCloudViewParams) == 304);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
