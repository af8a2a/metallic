#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>

namespace metallic::render {
using CloudViewFloat2 = std::array<float, 2>;
using CloudViewUInt = uint32_t;
using CloudViewColorIn = ShaderSampledImage;
using CloudViewScalarIn = ShaderSampledImage;
using CloudViewColorOut = ShaderStorageImage;
using CloudViewScalarOut = ShaderStorageImage;
using CloudViewAtmosphere = ShaderBuffer;
using CloudViewBuffer = ShaderBuffer;
#else
import ShaderCore;
import AtmosphereTypes;
using Metallic;
namespace Metallic {
typealias CloudViewFloat2 = float2;
typealias CloudViewUInt = uint;
typealias CloudViewColorIn = ResourceHandle<Texture2D<float4>>;
typealias CloudViewScalarIn = ResourceHandle<Texture2D<float>>;
typealias CloudViewColorOut = ResourceHandle<RWTexture2D<float4>>;
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
};

#ifdef __cplusplus
inline constexpr uint64_t kVolumetricCloudViewABI = 0x56434c4400000001ull;
static_assert(offsetof(VolumetricCloudViewParams, sceneColor) == 88);
static_assert(offsetof(VolumetricCloudViewParams, cloudDepth) == 136);
static_assert(sizeof(VolumetricCloudViewParams) == 144);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
