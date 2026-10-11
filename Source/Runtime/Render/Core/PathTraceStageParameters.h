#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using PathTraceUInt = uint32_t;
using PathTraceFloat4 = float[4];
using SharcHashHandle = ShaderBuffer;
using SharcAccumulationHandle = ShaderBuffer;
using SharcResolvedHandle = ShaderBuffer;
using SharcChangesHandle = ShaderBuffer;
#else
import ShaderCore;
import SHARC;
using Metallic.SHARC;
using Metallic;
namespace Metallic {
typealias PathTraceUInt = uint;
typealias PathTraceFloat4 = float4;
// The SHARC module requires structured-buffer objects for its atomics and resolve.
typealias SharcHashHandle = ResourceHandle<RWStructuredBuffer<uint64_t>>;
typealias SharcAccumulationHandle = ResourceHandle<RWStructuredBuffer<SHARCAccumulationData>>;
typealias SharcResolvedHandle = ResourceHandle<RWStructuredBuffer<SHARCPackedData>>;
typealias SharcChangesHandle = ResourceHandle<StructuredBuffer<uint>>;
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
    PathTraceUInt forceClear, padding1, padding2;
};

struct SharcMaintenanceParams {
    SharcHashHandle hashEntries;
    SharcAccumulationHandle accumulation;
    SharcResolvedHandle resolved;
    SharcChangesHandle geometryChanges;
    SceneSharcMaintenancePush settings;
};

#ifdef __cplusplus
inline constexpr uint64_t kSharcMaintenanceABI = 0x5054534841520003ull;
static_assert(sizeof(SceneSharcMaintenancePush) == 64);
static_assert(sizeof(SharcMaintenanceParams) == 80 && offsetof(SharcMaintenanceParams, settings) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
