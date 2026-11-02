#language slang 2026
module VirtualShadowMapPageHierarchyProbe;

import ShaderCore;
import VirtualShadowMaps;
using Metallic;
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h"

struct PageHierarchyProbePush
{
    uint owners, statistics, queries, output;
    uint axis, levels, physicalPages, queryCount;
    uint lightType, reserved0, reserved1, reserved2;
};
[[vk::push_constant]] ConstantBuffer<PageHierarchyProbePush> hierarchyProbe;

ShadowPush pageHierarchyProbePush()
{
    ShadowPush push = {};
    push.owners = hierarchyProbe.owners;
    push.statistics = hierarchyProbe.statistics;
    push.virtualPagesPerAxis = hierarchyProbe.axis;
    push.clipmapCount = hierarchyProbe.levels;
    push.physicalPageCount = hierarchyProbe.physicalPages;
    push.totalPages = hierarchyProbe.levels * hierarchyProbe.axis * hierarchyProbe.axis + 1u;
    push.lightType = hierarchyProbe.lightType;
    return push;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pageHierarchyResetProbeMain(uint3 id : SV_DispatchThreadID)
{
    ShadowPush push = pageHierarchyProbePush();
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    vsmResetPageHierarchy(id.x, push, owners);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pageHierarchyBuildProbeMain(uint3 id : SV_DispatchThreadID)
{
    vsmBuildPageHierarchy(id.x, pageHierarchyProbePush());
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pageHierarchyQueryProbeMain()
{
    ShadowPush push = pageHierarchyProbePush();
    StructuredBuffer<uint4> queries = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(hierarchyProbe.queries));
    RWStructuredBuffer<uint4> output = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(hierarchyProbe.output));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint nodeCounter = kVirtualShadowMapBuildDiagnosticWordOffset + kVSMBuildDiagnosticClusterHierarchyNodeTests * 2u;
    // One invocation isolates each production query's diagnostic counter delta.
    for (uint index = 0u; index < hierarchyProbe.queryCount; ++index) {
        uint4 rect = queries[index * 2u];
        uint level = queries[index * 2u + 1u].x;
        uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
        ShadowProjection projection = {};
        projection.page = uint4(level * push.virtualPagesPerAxis * push.virtualPagesPerAxis, axis, level, 0u);
        uint before = statistics[nodeCounter];
        bool hit = vsmHierarchyOverlaps(int2(rect.xy), int2(rect.zw), projection, push);
        output[index * 2u] = uint4(hit ? 1u : 0u, statistics[nodeCounter] - before,
            vsmPageHierarchyLevelWords(push), vsmPageHierarchyOffset(push));
        uint2 first = uint2(index & 7u, (index >> 3u) & 7u);
        uint2 last = min(uint2(7u), first + ((rect.zw - rect.xy) & uint2(7u)));
        uint2 rectangleMask = vsmReceiverRectangleMask(first, last);
        // Exercise page boundaries, both mask words, and supported power-of-two
        // page sizes. A nonzero origin also permits an AABB crossing the page.
        uint2 page = uint2(3u, 2u);
        uint pageSize = 8u << (index % 6u);
        uint cellSize = pageSize / 8u;
        uint2 firstTexel = page * pageSize + first * cellSize;
        uint2 lastTexel = page * pageSize + (last + 1u) * cellSize - 1u;
        firstTexel -= uint2(first.x == 0u ? cellSize : 0u, first.y == 0u ? cellSize : 0u);
        lastTexel += uint2(last.x == 7u ? cellSize : 0u, last.y == 7u ? cellSize : 0u);
        uint2 texelMask = vsmReceiverTexelMask(firstTexel, lastTexel, page, pageSize);
        output[index * 2u + 1u] = uint4(rectangleMask, texelMask);
    }
}
