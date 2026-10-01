#pragma once

// Shared declarations: handles select native views; ordinary data uses bounded BDA spans.
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
using LightGridData = ShaderDataSpan;
using LightGridCells = ShaderDataSpan;
using LightGridLights = ShaderDataSpan;
using LightPunctualData = ShaderDataSpan;
using LightUIntData = ShaderDataSpan;
using LightUInt4Data = ShaderDataSpan;
using LightFloat4Data = ShaderDataSpan;
#else
import ShaderCore;
import Lighting;
using Metallic;
using Metallic.Lighting;
namespace Metallic {
typealias LightUInt = uint;
typealias LightUInt2 = uint2;
typealias LightFloat4 = float4;
typealias LightSampledImage = DescriptorHandle<Texture2D<float4>>;
typealias LightSampledScalarImage = DescriptorHandle<Texture2D<float>>;
typealias LightStorageImage = DescriptorHandle<RWTexture2D<float4>>;
typealias LightStorageScalarImage = DescriptorHandle<RWTexture2D<float>>;
typealias LightGridData = DataSpan<ClusterLightGridParams>;
typealias LightGridCells = DataSpan<ClusterLightGridCell>;
typealias LightGridLights = DataSpan<ClusterLightData>;
typealias LightPunctualData = DataSpan<GPUPunctualLight>;
typealias LightUIntData = DataSpan<uint>;
typealias LightUInt4Data = DataSpan<uint4>;
typealias LightFloat4Data = DataSpan<float4>;
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
    LightUInt padding0, padding1;
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
inline constexpr uint64_t kClusterLightGridBuildABI = 0x4347524944000001ull;
inline constexpr uint64_t kLightGridDebugABI = 0x4c47444247000001ull;
inline constexpr uint64_t kPrepareLightsPdfABI = 0x4c50444600000001ull;
inline constexpr uint64_t kBuildReGIRABI = 0x5245474952000001ull;
inline constexpr uint64_t kEnvironmentLightingPrecomputeABI = 0x454e565052000001ull;
static_assert(sizeof(ClusterLightGridBuildParams) == 80);
static_assert(sizeof(LightGridDebugParams) == 72 && offsetof(LightGridDebugParams, settings) == 40);
static_assert(sizeof(PrepareLightsPdfParams) == 72 && offsetof(PrepareLightsPdfParams, settings) == 40);
static_assert(sizeof(BuildReGIRParams) == 112 && offsetof(BuildReGIRParams, settings) == 48);
static_assert(sizeof(EnvironmentLightingPrecomputeParams) == 88 && offsetof(EnvironmentLightingPrecomputeParams, settings) == 56);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
