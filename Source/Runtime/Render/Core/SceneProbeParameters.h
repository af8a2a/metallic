#pragma once
#include "PathTraceParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using SceneProbeResources = uint64_t;
using SceneProbeOutput = ShaderDataSpan;
using SceneProbeHits = ShaderDataSpan;
using SceneProbeVertices = ShaderDataSpan;
using SceneProbeUInt = uint32_t;
#else
namespace Metallic {
typealias SceneProbeResources = PathTraceParameters*;
typealias SceneProbeOutput = DataSpan<float4>;
typealias SceneProbeHits = DataSpan<uint2>;
typealias SceneProbeVertices = DataSpan<SceneShadingVertex>;
typealias SceneProbeUInt = uint;
#endif
struct ScenePositionProbeParameters {
    SceneProbeResources resources;
    SceneProbeOutput output;
    float translationX;
    SceneProbeUInt padding;
};
struct OpacityMicromapProbeParameters {
    SceneProbeResources resources;
    SceneProbeHits output;
    SceneProbeUInt materialTextureCount, padding;
};
struct SceneVertexProbeParameters {
    SceneProbeVertices vertices;
    SceneProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kOpacityMicromapProbeABI = 0x4f4d4d5052420001ull;
static_assert(sizeof(OpacityMicromapProbeParameters) == 32);
inline constexpr uint64_t kScenePositionProbeABI = 0x5343504f53500001ull;
inline constexpr uint64_t kSceneVertexProbeABI = 0x5343565458500001ull;
static_assert(sizeof(ScenePositionProbeParameters) == 32);
static_assert(sizeof(SceneVertexProbeParameters) == 32);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
