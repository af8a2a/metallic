#pragma once

#ifdef __cplusplus
#include <array>
#include <cstdint>
namespace metallic::render {
#define VSM_LOD_FLOAT4 std::array<float, 4>
#else
#define uint32_t uint
#define VSM_LOD_FLOAT4 float4
#endif

struct VirtualShadowMapLODGroupGPU
{
    VSM_LOD_FLOAT4 sphere;
    float error;
    uint32_t level, flags, parentOffset;
    uint32_t parentCount, clusterOffset, clusterCount, pageIndex;
};

struct VirtualShadowMapLODNodeGPU
{
    VSM_LOD_FLOAT4 sphere;
    float maxError;
    uint32_t escapeIndex, groupOffset, groupCount;
};

struct VirtualShadowMapLODClusterGPU
{
    uint32_t group, refinedGroup, triangleFirst, triangleCount;
    uint32_t level, instance, flags, reserved;
};

struct VirtualShadowMapLODInstanceGPU
{
    VSM_LOD_FLOAT4 world0, world1, world2, world3;
    VSM_LOD_FLOAT4 boundsMin, boundsMax;
    uint32_t groupOffset, groupCount, nodeOffset, nodeCount;
    uint32_t metricOffset, clusterOffset, clusterCount, stateOffset;
    uint32_t baseClusterCount, flags, metricGroupCount, baseTriangleCount;
};

// One immutable context per recorded face. Selection addresses are view-major;
// the final view is the complete coarse fallback, after all clipmap/mip views.
struct VirtualShadowMapLODContextGPU
{
    uint32_t groups, parents, nodes, metrics;
    uint32_t instances, clusters, triangleClusters, pageTable;
    uint32_t pageRequests, groupStates, selected, previous;
    uint32_t changed, statistics, instanceCount, clusterCount;
    uint32_t groupStateCount, viewCount;
    float maxErrorTexels, maxWorldError;
    uint32_t previousValid, requestsEnabled, reserved0, reserved1;
};

// Instance flags: resident geometry and protected base geometry, respectively.
// Cluster flag 1 identifies authored/base slots, excluded from hierarchy cuts.
#ifdef __cplusplus
inline constexpr uint32_t kVirtualShadowMapLODInstanceResident = 1;
inline constexpr uint32_t kVirtualShadowMapLODInstanceProtected = 2;
inline constexpr uint32_t kVirtualShadowMapLODClusterBase = 1;
inline constexpr uint32_t kVirtualShadowMapLODStatisticsWords = 32;
static_assert(sizeof(VirtualShadowMapLODGroupGPU) == 48);
static_assert(sizeof(VirtualShadowMapLODNodeGPU) == 32);
static_assert(sizeof(VirtualShadowMapLODClusterGPU) == 32);
static_assert(sizeof(VirtualShadowMapLODInstanceGPU) == 144);
static_assert(sizeof(VirtualShadowMapLODContextGPU) == 96);
} // namespace metallic::render
#else
#undef uint32_t
#endif
#undef VSM_LOD_FLOAT4
