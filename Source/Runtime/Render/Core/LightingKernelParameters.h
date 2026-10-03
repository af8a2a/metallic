#pragma once

// Shared declarations: handles select native views; ordinary data uses bounded descriptor spans.
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>
namespace metallic::render {
using LightUInt = uint32_t;
using LightUInt2 = std::array<uint32_t, 2>;
using LightFloat4 = std::array<float, 4>;
using LightSampledImage = ShaderSampledImage;
using LightSampledScalarImage = ShaderSampledImage;
using LightStorageImage = ShaderStorageImage;
using LightStorageScalarImage = ShaderStorageImage;
using LightGridData = GPUBufferSpan;
using LightGridCells = GPUBufferSpan;
using LightGridLights = GPUBufferSpan;
using LightPunctualData = GPUBufferSpan;
using LightUIntData = GPUBufferSpan;
using LightUInt4Data = GPUBufferSpan;
using LightFloat4Data = GPUBufferSpan;
#else
import ShaderCore;
import Lighting;
using Metallic;
using Metallic.Lighting;
namespace Metallic {
typealias LightUInt = uint;
typealias LightUInt2 = uint2;
typealias LightFloat4 = float4;
typealias LightSampledImage = ResourceHandle<Texture2D<float4>>;
typealias LightSampledScalarImage = ResourceHandle<Texture2D<float>>;
typealias LightStorageImage = ResourceHandle<RWTexture2D<float4>>;
typealias LightStorageScalarImage = ResourceHandle<RWTexture2D<float>>;
typealias LightGridData = RWBufferSpan<ClusterLightGridParams>;
typealias LightGridCells = RWBufferSpan<ClusterLightGridCell>;
typealias LightGridLights = RWBufferSpan<ClusterLightData>;
typealias LightPunctualData = RWBufferSpan<GPUPunctualLight>;
typealias LightUIntData = RWBufferSpan<uint>;
typealias LightUInt4Data = RWBufferSpan<uint4>;
typealias LightFloat4Data = RWBufferSpan<float4>;
#endif

struct ClusterLightGridBuildParams {
    LightGridData grid;
    LightGridLights lights;
    LightUIntData candidates;
    LightGridCells cells;
    LightUIntData indices;
};

struct LightGridDebugPush {
    LightUInt width, height, mode, sliceIndex;
    float viewDepth, heatmapMaxLights;
    LightUInt flags, reserved;
};
struct LightGridDebugParams {
    LightGridData grid;
    LightGridCells cells;
    LightStorageImage output;
    LightGridDebugPush settings;
};

struct PrepareLightsPdfPush {
    LightUInt mode, lightCount, sourceMipLevel, padding0;
    LightUInt2 sourceSize, destinationSize;
};
struct PrepareLightsPdfParams {
    LightSampledImage environment;
    LightStorageScalarImage sourceMip, destinationMip;
    LightPunctualData lights;
    PrepareLightsPdfPush settings;
};

struct BuildReGIRPush {
    LightUInt lightCount, gridSize, lightsPerCell, buildSamples;
    LightUInt frameIndex, padding0, lightSlotCount, padding1;
    LightFloat4 sceneCenterRadius;
    float samplingJitter;
    LightUInt padding2, padding3, padding4;
};
#ifdef __cplusplus
struct alignas(16) BuildReGIRParams {
#else
struct BuildReGIRParams {
#endif
    LightSampledScalarImage localLightPdf;
    LightUInt4Data output;
    LightPunctualData lights;
    LightUInt padding0;
    BuildReGIRPush settings;
};

struct EnvironmentLightingPrecomputePush {
    LightUInt mode, width, height, partialCount;
    LightUInt dispatchWidth, procedural, padding1, padding2;
};
struct EnvironmentLightingPrecomputeParams {
    LightSampledImage radiance;
    LightFloat4Data partials, coefficients, specular;
    EnvironmentLightingPrecomputePush settings;
};

#ifdef __cplusplus
inline constexpr uint64_t kClusterLightGridBuildABI = 0x4347524944000002ull;
inline constexpr uint64_t kLightGridDebugABI = 0x4c47444247000002ull;
inline constexpr uint64_t kPrepareLightsPdfABI = 0x4c50444600000002ull;
inline constexpr uint64_t kBuildReGIRABI = 0x5245474952000002ull;
inline constexpr uint64_t kEnvironmentLightingPrecomputeABI = 0x454e565052000002ull;
static_assert(sizeof(ClusterLightGridBuildParams) == 60);
static_assert(sizeof(LightGridDebugParams) == 60 && offsetof(LightGridDebugParams, settings) == 28);
static_assert(sizeof(PrepareLightsPdfParams) == 56 && offsetof(PrepareLightsPdfParams, settings) == 24);
static_assert(sizeof(BuildReGIRParams) == 96 && offsetof(BuildReGIRParams, settings) == 32);
static_assert(sizeof(EnvironmentLightingPrecomputeParams) == 72 && offsetof(EnvironmentLightingPrecomputeParams, settings) == 40);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
