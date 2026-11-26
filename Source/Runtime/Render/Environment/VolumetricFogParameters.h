#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>

namespace metallic::render {
using FogFloat4 = std::array<float, 4>;
using FogUInt = uint32_t;
using FogUInt4 = std::array<uint32_t, 4>;
using FogSampled2D = ShaderSampledImage;
using FogSampled2DScalar = ShaderSampledImage;
using FogSampled3D = ShaderSampledImage;
using FogSampled3DScalar = ShaderSampledImage;
using FogStorage2D = ShaderStorageImage;
using FogStorage2DScalar = ShaderStorageImage;
using FogStorage3D = ShaderStorageImage;
using FogStorage3DScalar = ShaderStorageImage;
using FogAtmosphere = ShaderBuffer;
using FogLightGrid = ShaderBuffer;
using FogLights = ShaderBuffer;
using FogLightCells = ShaderBuffer;
using FogLightIndices = ShaderBuffer;
using FogSampler = ShaderSampler;
using FogInstanceSpan = GPUBufferSpan;
using FogTileRead = ShaderSampledImage;
using FogTileWrite = ShaderStorageImage;
#else
import ShaderCore;
import AtmosphereTypes;
import Lighting;
using Metallic;
using Metallic.Lighting;
namespace Metallic {
typealias FogFloat4 = float4;
typealias FogUInt = uint;
typealias FogUInt4 = uint4;
typealias FogSampled2D = ResourceHandle<Texture2D<float4>>;
typealias FogSampled2DScalar = ResourceHandle<Texture2D<float>>;
typealias FogSampled3D = ResourceHandle<Texture3D<float4>>;
typealias FogSampled3DScalar = ResourceHandle<Texture3D<float>>;
typealias FogStorage2D = ResourceHandle<RWTexture2D<float4>>;
typealias FogStorage2DScalar = ResourceHandle<RWTexture2D<float>>;
typealias FogStorage3D = ResourceHandle<RWTexture3D<float4>>;
typealias FogStorage3DScalar = ResourceHandle<RWTexture3D<float>>;
typealias FogAtmosphere = ResourceHandle<StructuredBuffer<GPUAtmosphereParameters>>;
typealias FogLightGrid = ResourceHandle<StructuredBuffer<ClusterLightGridParams>>;
typealias FogLights = ResourceHandle<StructuredBuffer<ClusterLightData>>;
typealias FogLightCells = ResourceHandle<StructuredBuffer<ClusterLightGridCell>>;
typealias FogLightIndices = ResourceHandle<StructuredBuffer<uint>>;
typealias FogSampler = SamplerHandle;
typealias FogInstanceSpan = BufferSpan<uint4>;
typealias FogTileRead = ResourceHandle<Texture2DArray<uint>>;
typealias FogTileWrite = ResourceHandle<RWTexture2DArray<uint>>;
#endif

struct LocalFogParameters {
    FogInstanceSpan instances; // Three packed uint4 records per sorted sphere.
    float startDistanceCm;
    FogTileRead tiles;
    FogTileWrite tilesOut;
    FogSampled3D emissive;
    FogStorage3D emissiveOut;
};

struct VolumetricFogParams {
    ViewCameraConstants current, previous;
    FogFloat4 gridZ;               // UE B, O, S in centimeters
    FogFloat4 fogLayer0, fogLayer1; // density, falloff, base height (cm)
    FogFloat4 albedoExtinction;
    FogFloat4 emissive;             // UE scene units (authoring value / 10000)
    FogFloat4 lightDirectionPhase;  // direction toward the light, phase g
    FogFloat4 lightColorIntensity;  // working-space irradiance, scattering intensity
    FogFloat4 distances;            // start (cm), inverse fade (1/cm), history weight, jitter enabled
    FogUInt4 grid;                  // froxel extent, pixels per froxel
    FogUInt4 temporal;              // frame number, temporal enabled, conservative depth enabled, miss samples
    uint64_t scene;
    FogAtmosphere atmosphereParameters;
    FogSampled2D transmittance, cloudShadow;
    FogSampled2DScalar depth, currentDepth, previousDepth;
    FogSampled3D material, history, scattering, integrated;
    FogSampled3DScalar shadow;
    FogSampled2D sceneColor;
    FogStorage3D materialOut, scatteringOut, integratedOut, historyScatteringOut;
    FogStorage3DScalar shadowOut;
    FogStorage2DScalar currentDepthOut, historyDepthOut;
    FogStorage2D colorOut;
    FogLightGrid lightGrid;
    FogLights lights;
    FogLightCells lightCells;
    FogLightIndices lightIndices, lightCandidates;
    FogSampler sampler;
    LocalFogParameters localFog;
};

#ifdef __cplusplus
inline constexpr uint64_t kVolumetricFogABI = 0x56464f4700000003ull;
static_assert(offsetof(VolumetricFogParams, scene) == 320);
static_assert(offsetof(VolumetricFogParams, sampler) == 428);
static_assert(offsetof(VolumetricFogParams, localFog) == 432);
static_assert(sizeof(VolumetricFogParams) == 464);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
