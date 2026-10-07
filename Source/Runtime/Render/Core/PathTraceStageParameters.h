#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using PathTraceUInt = uint32_t;
using PathTraceFloat4 = float[4];
using PathTraceStorageImage = ShaderStorageImage;
using PathTraceAerialHandle = ShaderBuffer;
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
typealias PathTraceStorageImage = ResourceHandle<RWTexture2D<float4>>;
typealias PathTraceAerialHandle = ResourceHandle<RWStructuredBuffer<float4>>;
// The SHaRC SDK requires structured-buffer objects for its atomics and resolve.
typealias SharcHashHandle = ResourceHandle<RWStructuredBuffer<uint64_t>>;
typealias SharcAccumulationHandle = ResourceHandle<RWStructuredBuffer<SharcAccumulationData>>;
typealias SharcResolvedHandle = ResourceHandle<RWStructuredBuffer<SharcPackedData>>;
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
    PathTraceUInt padding0;
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
    PathTraceAerialHandle primaryAerial;
    ScenePathTraceTonemapPush settings;
};

#ifdef __cplusplus
inline constexpr uint64_t kSharcMaintenanceABI = 0x5054534841520002ull;
inline constexpr uint64_t kPathTraceTonemapABI = 0x5054544f4e450003ull;
static_assert(sizeof(SceneSharcMaintenancePush) == 64);
static_assert(sizeof(SharcMaintenanceParams) == 80 && offsetof(SharcMaintenanceParams, settings) == 16);
static_assert(sizeof(PathTraceTonemapParams) == 40 && offsetof(PathTraceTonemapParams, settings) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
