#pragma once
#include "PathTraceInlineParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using NRCQueryPathHandle = ShaderBuffer;
using NRCTrainingPathHandle = ShaderBuffer;
using NRCVertexHandle = ShaderBuffer;
using NRCRadianceHandle = ShaderBuffer;
using NRCCounterHandle = ShaderBuffer;
#else
#include "../../../../Shaders/ThirdParty/RadianceCache/Nrc/Nrc.hlsli"
namespace Metallic {
typealias NRCQueryPathHandle = DescriptorHandle<RWStructuredBuffer<NrcPackedQueryPathInfo>>;
typealias NRCTrainingPathHandle = DescriptorHandle<RWStructuredBuffer<NrcPackedTrainingPathInfo>>;
typealias NRCVertexHandle = DescriptorHandle<RWStructuredBuffer<NrcPackedPathVertex>>;
typealias NRCRadianceHandle = DescriptorHandle<RWStructuredBuffer<NrcRadianceParams, ScalarDataLayout>>;
typealias NRCCounterHandle = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct NRCTraceParameters
{
    PathTraceInlineParameters path;
    PathTraceSettings cacheSettings;
    NRCQueryPathHandle queryPath;
    NRCTrainingPathHandle trainingPath;
    NRCVertexHandle vertices;
    NRCRadianceHandle radiance;
    NRCCounterHandle counters;
};
#ifdef __cplusplus
inline constexpr uint64_t kNRCTraceABI = 0x4e52435452430001ull;
static_assert(sizeof(NRCTraceParameters) == 88);
static_assert(offsetof(NRCTraceParameters, cacheSettings) == 40);
static_assert(offsetof(NRCTraceParameters, queryPath) == 48);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
