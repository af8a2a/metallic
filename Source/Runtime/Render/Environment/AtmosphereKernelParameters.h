#pragma once

// CPU/Slang ABI. All transport coefficients and radiance spectra use samples
// at 680, 550 and 440 nm; conversion to the working color space occurs once.
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>
namespace metallic::render {
using AtmosphereFloat4 = std::array<float, 4>;
using AtmosphereUInt4 = std::array<uint32_t, 4>;
using AtmosphereUInt = uint32_t;
using AtmosphereBuffer = GPUBufferSpan;
using AtmosphereSampledImage = ShaderSampledImage;
using AtmosphereStorageImage = ShaderStorageImage;
#define ATMOSPHERE_PUBLIC
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias AtmosphereFloat4 = float4;
typealias AtmosphereUInt4 = uint4;
typealias AtmosphereUInt = uint;
typealias AtmosphereBuffer = RWBufferSpan<float4>;
typealias AtmosphereSampledImage = ResourceHandle<Texture2D<float4>>;
typealias AtmosphereStorageImage = ResourceHandle<RWTexture2D<float4>>;
#define ATMOSPHERE_PUBLIC public
#endif

#ifdef __cplusplus
struct alignas(16) GPUAtmosphereParameters {
#else
public struct GPUAtmosphereParameters {
#endif
    ATMOSPHERE_PUBLIC AtmosphereFloat4 observerPlanetBottom; // Observer relative to centre, km; bottom radius.
    ATMOSPHERE_PUBLIC AtmosphereFloat4 observerWorldTop;     // World origin metres; top radius.
    ATMOSPHERE_PUBLIC AtmosphereFloat4 rayleighScaleHeight;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 mieScatteringScaleHeight;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 mieExtinctionAnisotropy;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 ozoneCenter;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 groundOzoneWidth;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 sunDirectionRadius;   // Direction toward source; angular radius.
    ATMOSPHERE_PUBLIC AtmosphereFloat4 sunIrradianceEnabled; // TOA W/m^2/nm; enabled.
    ATMOSPHERE_PUBLIC AtmosphereFloat4 moonDirectionRadius;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 moonIrradianceEnabled;
    ATMOSPHERE_PUBLIC AtmosphereFloat4 settings;             // Maximum aerial distance km; physical enabled.
};

#ifdef __cplusplus
struct alignas(16) AtmospherePrecomputeParams {
#else
public struct AtmospherePrecomputeParams {
#endif
    ATMOSPHERE_PUBLIC AtmosphereBuffer parameters;
    ATMOSPHERE_PUBLIC AtmosphereSampledImage transmittance;
    ATMOSPHERE_PUBLIC AtmosphereSampledImage multiScattering;
    ATMOSPHERE_PUBLIC AtmosphereSampledImage skyView;
    ATMOSPHERE_PUBLIC AtmosphereSampledImage sourceMip;
    ATMOSPHERE_PUBLIC AtmosphereStorageImage output;
    ATMOSPHERE_PUBLIC AtmosphereBuffer aerial;
    ATMOSPHERE_PUBLIC AtmosphereUInt reserved;
    ATMOSPHERE_PUBLIC AtmosphereUInt4 dimensions; // x,y,z; mode (capture=0,mip=1).
};

#ifdef __cplusplus
inline constexpr uint64_t kAtmospherePrecomputeABI = 0x41544d4f53500001ull;
static_assert(sizeof(GPUAtmosphereParameters) == 192 && alignof(GPUAtmosphereParameters) == 16);
static_assert(sizeof(AtmospherePrecomputeParams) == 64 && offsetof(AtmospherePrecomputeParams, dimensions) == 48);
#endif
#undef ATMOSPHERE_PUBLIC
} // namespace metallic::render (C++) / Metallic (Slang)
