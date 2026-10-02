#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <array>
#include <cstddef>
namespace metallic::render {
using ResidentLODFloat4 = std::array<float, 4>;
using ResidentLODClusters = ShaderDataSpan;
using ResidentLODRecords = ShaderDataSpan;
using ResidentLODInstances = ShaderDataSpan;
using ResidentLODGroups = ShaderDataSpan;
using ResidentLODSelections = ShaderDataSpan;
using ResidentLODWords = ShaderDataSpan;
#else
import ShaderCore;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
namespace Metallic {
typealias ResidentLODFloat4 = float4;
typealias ResidentLODClusters = DataSpan<GPUDrivenPreviewMeshlet>;
typealias ResidentLODRecords = DataSpan<VisibleClusterRecord>;
typealias ResidentLODInstances = DataSpan<GPUDrivenPreviewInstance>;
typealias ResidentLODGroups = DataSpan<MeshletLODGroupRecord>;
typealias ResidentLODSelections = DataSpan<uint4>;
typealias ResidentLODWords = DataSpan<uint>;
#endif
struct ResidentLODParameters
{
    ResidentLODFloat4 eye;
    ResidentLODFloat4 forward;
    ResidentLODFloat4 projection;
    ResidentLODClusters clusters;
    ResidentLODRecords records;
    ResidentLODInstances instances;
    ResidentLODGroups groups;
    ResidentLODSelections output;
    ResidentLODWords arguments;
    ResidentLODWords scratch;
    uint32_t offset;
    uint32_t count;
    uint32_t capacity;
    uint32_t instanceCount;
    uint32_t groupCount;
    uint32_t manualLevel;
};
#ifdef __cplusplus
inline constexpr uint64_t kResidentLODABI = 0x5245534c4f440001ull;
static_assert(sizeof(ResidentLODParameters) == 184);
static_assert(offsetof(ResidentLODParameters, clusters) == 48);
static_assert(offsetof(ResidentLODParameters, offset) == 160);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
