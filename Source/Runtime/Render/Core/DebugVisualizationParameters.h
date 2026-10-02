#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using DebugFloat4 = float[4];
using DebugUInt = uint32_t;
using DebugScene = ShaderAccelerationStructure;
using DebugOutput = ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias DebugFloat4 = float4;
typealias DebugUInt = uint;
typealias DebugScene = DescriptorHandle<RaytracingAccelerationStructure>;
typealias DebugOutput = DescriptorHandle<RWTexture2D<float4>>;
#endif
struct SceneRayQueryVisualizationPush {
    DebugFloat4 eye;
    DebugFloat4 center;
    DebugFloat4 upProjection;
    DebugFloat4 viewport;
    DebugFloat4 clipOrtho;
    DebugUInt mode;
    DebugUInt width;
    DebugUInt height;
    DebugUInt padding;
};
struct SceneRayQueryVisualizationParams {
    DebugScene scene;
    DebugOutput output;
    SceneRayQueryVisualizationPush settings;
};
#ifdef __cplusplus
inline constexpr uint64_t kSceneRayQueryVisualizationABI = 0x5251564953000001ull;
static_assert(sizeof(SceneRayQueryVisualizationPush) == 96);
static_assert(sizeof(SceneRayQueryVisualizationParams) == 112 && offsetof(SceneRayQueryVisualizationParams, settings) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
