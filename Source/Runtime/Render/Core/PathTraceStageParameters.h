#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using PathTraceUInt = uint32_t;
using PathTraceFloat4 = float[4];
using PathTraceStorageImage = ShaderStorageImage;
using SharcHashHandle = ShaderBuffer;
using SharcAccumulationHandle = ShaderBuffer;
using SharcResolvedHandle = ShaderBuffer;
#else
import ShaderCore;
#include "../../../../Shaders/ThirdParty/RadianceCache/Sharc/SharcTypes.h"
using Metallic;
namespace Metallic {
typealias PathTraceUInt = uint;
typealias PathTraceFloat4 = float4;
typealias PathTraceStorageImage = DescriptorHandle<RWTexture2D<float4>>;
// The SHaRC SDK requires structured-buffer objects for its atomics and resolve.
typealias SharcHashHandle = DescriptorHandle<RWStructuredBuffer<uint64_t>>;
typealias SharcAccumulationHandle = DescriptorHandle<RWStructuredBuffer<SharcAccumulationData>>;
typealias SharcResolvedHandle = DescriptorHandle<RWStructuredBuffer<SharcPackedData>>;
#endif

#ifdef __cplusplus
struct alignas(16) SceneSharcMaintenancePush {
#else
struct SceneSharcMaintenancePush {
#endif
    PathTraceFloat4 cameraPosition;
    PathTraceFloat4 cameraPositionPrev;
    float sceneScale;
    PathTraceUInt entriesNum;
    PathTraceUInt accumulationFrameNum;
    PathTraceUInt staleFrameNumMax;
    PathTraceUInt frameIndex;
    PathTraceUInt padding0, padding1, padding2;
};

struct SharcMaintenanceParams {
    SharcHashHandle hashEntries;
    SharcAccumulationHandle accumulation;
    SharcResolvedHandle resolved;
    PathTraceUInt padding0, padding1;
    SceneSharcMaintenancePush settings;
};

struct ScenePathTraceTonemapPush {
    PathTraceUInt width, height;
    float exposure;
    PathTraceUInt outputLinear, hasHistory, accumulationFrame;
};

struct PathTraceTonemapParams {
    PathTraceStorageImage source;
    PathTraceStorageImage output;
    PathTraceStorageImage historyPrevious;
    ScenePathTraceTonemapPush settings;
};

#ifdef __cplusplus
inline constexpr uint64_t kSharcMaintenanceABI = 0x5054534841520001ull;
inline constexpr uint64_t kPathTraceTonemapABI = 0x5054544f4e450001ull;
static_assert(sizeof(SceneSharcMaintenancePush) == 64);
static_assert(sizeof(SharcMaintenanceParams) == 96 && offsetof(SharcMaintenanceParams, settings) == 32);
static_assert(sizeof(PathTraceTonemapParams) == 48 && offsetof(PathTraceTonemapParams, settings) == 24);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
