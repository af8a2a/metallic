#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_GPU_LOD
#define VSM_GPU_LOD 0
#endif

static const float kVSMLargestError = 3.402823466e38;

VirtualShadowMapLODContextGPU vsmLODContext(uint handle)
{
    StructuredBuffer<VirtualShadowMapLODContextGPU> contexts =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODContextGPU>>(handle));
    return contexts[0];
}

float vsmLODWorldScale(VirtualShadowMapLODInstanceGPU instance)
{
    float3 x = instance.world0.xyz, y = instance.world1.xyz, z = instance.world2.xyz;
    return sqrt(max(dot(x, x) + abs(dot(x, y)) + abs(dot(x, z)),
        max(dot(y, y) + abs(dot(y, x)) + abs(dot(y, z)),
            dot(z, z) + abs(dot(z, x)) + abs(dot(z, y)))));
}

float vsmLODOperatorNorm(float3 x, float3 y)
{
    float xx = dot(x, x), xy = dot(x, y), yy = dot(y, y);
    return sqrt(0.5 * (xx + yy + length(float2(xx - yy, 2.0 * xy))));
}

float vsmLODProjectedError(float4 sphere, float error, VirtualShadowMapLODInstanceGPU instance,
    float scale, ShadowProjection projection, float resolution, bool hierarchy)
{
    if (error == 0.0) { return 0.0; }
    float3 center = instance.world0.xyz * sphere.x + instance.world1.xyz * sphere.y +
        instance.world2.xyz * sphere.z + instance.world3.xyz;
    float3 magnitude = abs(instance.world0.xyz) * abs(sphere.x) + abs(instance.world1.xyz) * abs(sphere.y) +
        abs(instance.world2.xyz) * abs(sphere.z) + abs(instance.world3.xyz);
    float radius = (sphere.w + error) * scale + max(magnitude.x, max(magnitude.y, magnitude.z)) * (64.0 * 1.192092896e-7);
    if (!all(isfinite(center)) || !isfinite(radius)) { return kVSMLargestError; }
    float3 x = projection.rowX.xyz, y = projection.rowY.xyz, w = projection.rowW.xyz;
    float cx = dot(projection.rowX, float4(center, 1.0));
    float cy = dot(projection.rowY, float4(center, 1.0));
    float cw = dot(projection.rowW, float4(center, 1.0));
    float wLength = length(w), minW = cw - radius * wLength;
    if (wLength != 0.0) {
        if (cw + radius * wLength < projection.originNear.w) { return 0.0; }
        if (minW <= projection.originNear.w) { return kVSMLargestError; }
    }
    if (cw + cx + radius * length(w + x) < 0.0 || cw - cx + radius * length(w - x) < 0.0 ||
        cw + cy + radius * length(w + y) < 0.0 || cw - cy + radius * length(w - y) < 0.0) { return 0.0; }
    float numerator;
    if (hierarchy) {
        float aNorm = vsmLODOperatorNorm(x, y);
        float maxClipXY = length(float2(cx, cy)) + radius * aNorm;
        numerator = aNorm + maxClipXY * wLength / minW;
    } else {
        numerator = vsmLODOperatorNorm(x - cx / cw * w, y - cy / cw * w);
    }
    float projected = error * scale * 0.5 * resolution * numerator / minW * (1.0 + radius * wLength / minW);
    return isfinite(projected) ? projected : kVSMLargestError;
}

// Each GPU task owns one complete (projection, caster) DAG. The immutable
// topology guarantees refined group IDs precede every parent owner.
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmLODSelectMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    if (lane != 0u) { return; }
    VirtualShadowMapLODContextGPU context = vsmLODContext(vsmCacheMetadata(push, 50u));
    uint instanceIndex = push.workBase + group.x, view = group.y;
    if (instanceIndex >= context.instanceCount || view >= context.viewCount) { return; }
    StructuredBuffer<VirtualShadowMapLODInstanceGPU> instances =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODInstanceGPU>>(context.instances));
    StructuredBuffer<VirtualShadowMapLODGroupGPU> groups =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODGroupGPU>>(context.groups));
    StructuredBuffer<VirtualShadowMapLODClusterGPU> clusters =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODClusterGPU>>(context.clusters));
    RWStructuredBuffer<uint> states = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.groupStates));
    RWStructuredBuffer<uint> selected = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.selected));
    RWStructuredBuffer<uint> changed = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.changed));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.statistics));
    VirtualShadowMapLODInstanceGPU instance = instances[instanceIndex];
    uint stateBase = view * context.groupStateCount + instance.stateOffset;
    uint selectedBase = view * context.clusterCount;
    uint stats = view * 32u;
    if (instanceIndex == 0u) { statistics[stats] = 3u; statistics[stats + 1u] = 1u; statistics[stats + 2u] = 1u; }
    for (uint index = 0u; index < instance.groupCount; ++index) { states[stateBase + index] = 0u; }
    bool resident = (instance.flags & 1u) != 0u;
    bool protectedGeometry = (instance.flags & 2u) != 0u;
    bool baseGeometry = protectedGeometry || instance.groupCount == 0u || (resident && context.maxErrorTexels == 0.0);
    uint metricTests = 0u, nodeVisits = 0u, topologyTests = 0u;
    uint missingPages = 0u;
    if (!baseGeometry) {
        StructuredBuffer<uint> parents = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.parents));
        for (uint index = 0u; index < instance.groupCount; ++index) {
            VirtualShadowMapLODGroupGPU current = groups[instance.groupOffset + index];
            if ((current.flags & 1u) != 0u || context.maxErrorTexels == 0.0) { states[stateBase + index] = 1u; }
        }
        if (context.maxErrorTexels != 0.0) {
            StructuredBuffer<VirtualShadowMapLODNodeGPU> nodes =
                resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODNodeGPU>>(context.nodes));
            StructuredBuffer<uint> metrics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.metrics));
            StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
            ShadowProjection projection = projections[view];
            float resolution = float(projection.page.y * push.pageSize);
            float scale = vsmLODWorldScale(instance);
            float pruneThreshold = context.maxErrorTexels - max(context.maxErrorTexels * 1e-4, 1e-5);
            uint index = 0u;
            while (index < instance.nodeCount) {
                VirtualShadowMapLODNodeGPU node = nodes[instance.nodeOffset + index];
                ++nodeVisits;
                float bound = vsmLODProjectedError(node.sphere, node.maxError, instance, scale, projection, resolution, true);
                if (bound < pruneThreshold && (bound == 0.0 || node.maxError * scale <= context.maxWorldError)) {
                    index = node.escapeIndex;
                    continue;
                }
                for (uint local = 0u; local < node.groupCount; ++local) {
                    uint groupIndex = metrics[instance.metricOffset + node.groupOffset + local];
                    VirtualShadowMapLODGroupGPU current = groups[instance.groupOffset + groupIndex];
                    ++metricTests;
                    float error = vsmLODProjectedError(current.sphere, current.error, instance, scale, projection, resolution, false);
                    if (error > context.maxErrorTexels || (error > 0.0 && current.error * scale > context.maxWorldError)) {
                        states[stateBase + groupIndex] = 1u;
                    }
                }
                ++index;
            }
        }
        // A child demanded by this view forces all owners, including owners
        // whose own metric spheres are outside the view.
        for (uint index = 0u; index < instance.groupCount; ++index) {
            if (states[stateBase + index] == 0u) { continue; }
            VirtualShadowMapLODGroupGPU current = groups[instance.groupOffset + index];
            for (uint parent = 0u; parent < current.parentCount; ++parent) {
                states[stateBase + parents[current.parentOffset + parent]] = 1u;
            }
        }
        for (uint remaining = instance.groupCount; remaining != 0u; --remaining) {
            uint index = remaining - 1u;
            if (states[stateBase + index] == 0u) { continue; }
            ++topologyTests;
            VirtualShadowMapLODGroupGPU current = groups[instance.groupOffset + index];
            bool drawable = true;
            if (!resident) {
                StructuredBuffer<uint2> pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(context.pageTable));
                uint state = pageTable[current.pageIndex].x & 7u;
                drawable = state == 2u || state == 3u;
                missingPages += drawable ? 0u : 1u;
                if (context.requestsEnabled != 0u) {
                    RWStructuredBuffer<uint> requests = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.pageRequests));
                    InterlockedOr(requests[current.pageIndex], 1u);
                }
            }
            bool active = drawable;
            for (uint parent = 0u; parent < current.parentCount; ++parent) {
                active = active && (states[stateBase + parents[current.parentOffset + parent]] & 2u) != 0u;
            }
            states[stateBase + index] = active ? 3u : 1u;
        }
    }
    bool finest = resident;
    for (uint local = 0u; local < instance.clusterCount; ++local) {
        uint index = instance.clusterOffset + local;
        VirtualShadowMapLODClusterGPU cluster = clusters[index];
        bool active = local < instance.baseClusterCount;
        if (!baseGeometry) {
            active = false;
            if ((cluster.flags & 1u) == 0u) {
                active = (states[stateBase + cluster.group] & 2u) != 0u;
                if (cluster.refinedGroup != 0xffffffffu) { active = active && (states[stateBase + cluster.refinedGroup] & 2u) == 0u; }
            }
        }
        selected[selectedBase + index] = active ? 1u : 0u;
        if (active && cluster.level != 0u) { finest = false; }
    }
    // The authored prefix preserves original winding/order and tessellation
    // slots when a resident cut reaches the finest level.
    bool useBase = baseGeometry || finest;
    bool differs = context.previousValid == 0u;
    uint selectedCount = 0u, selectedSlots = 0u;
    for (uint local = 0u; local < instance.clusterCount; ++local) {
        uint index = instance.clusterOffset + local;
        VirtualShadowMapLODClusterGPU cluster = clusters[index];
        if (useBase) { selected[selectedBase + index] = local < instance.baseClusterCount ? 1u : 0u; }
        uint active = selected[selectedBase + index];
        if (context.previousValid != 0u) {
            StructuredBuffer<uint> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.previous));
            differs = differs || active != previous[selectedBase + index];
        }
        if (active != 0u) {
            ++selectedCount; selectedSlots += cluster.triangleCount;
            InterlockedAdd(statistics[stats + 16u + min(cluster.level, 15u)], 1u);
            if (!resident && instance.groupCount != 0u && context.requestsEnabled != 0u) {
                VirtualShadowMapLODGroupGPU owner = groups[instance.groupOffset + cluster.group];
                RWStructuredBuffer<uint> requests = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(context.pageRequests));
                InterlockedOr(requests[owner.pageIndex], 1u);
            }
        }
    }
    changed[view * context.instanceCount + instanceIndex] = differs ? 1u : 0u;
    InterlockedAdd(statistics[stats + 3u], topologyTests);
    InterlockedAdd(statistics[stats + 4u], selectedCount);
    InterlockedAdd(statistics[stats + 5u], selectedSlots);
    InterlockedAdd(statistics[stats + (context.requestsEnabled != 0u ? 6u : 7u)], missingPages);
    InterlockedAdd(statistics[stats + 8u], missingPages != 0u || instance.groupCount == 0u ? 1u : 0u);
    InterlockedAdd(statistics[stats + 9u], protectedGeometry ? 1u : 0u);
    InterlockedAdd(statistics[stats + 10u], instance.baseTriangleCount);
    InterlockedAdd(statistics[stats + 11u], metricTests);
    InterlockedAdd(statistics[stats + 12u], nodeVisits);
    InterlockedAdd(statistics[stats + 13u], context.maxErrorTexels != 0.0 && !baseGeometry ? instance.metricGroupCount - metricTests : 0u);
}

bool vsmLODSelected(ShadowPush push, uint triangleIndex, uint view)
{
#if !VSM_GPU_LOD
    return true;
#else
    uint handle = vsmCacheMetadata(push, 50u);
    if (handle == 0xffffffffu) { return true; }
    if (triangleIndex >= push.triangleCount) { return false; }
    VirtualShadowMapLODContextGPU context = vsmLODContext(handle);
    if (view >= context.viewCount) { return false; }
    StructuredBuffer<uint> triangleClusters = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.triangleClusters));
    uint cluster = triangleClusters[triangleIndex];
    if (cluster == 0xffffffffu) { return false; }
    StructuredBuffer<uint> selected = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.selected));
    return selected[view * context.clusterCount + cluster] != 0u;
#endif
}

bool vsmLODSelectedRange(ShadowPush push, uint firstTriangle, uint view)
{
#if !VSM_GPU_LOD
    return true;
#else
    if (firstTriangle >= push.triangleCount) { return false; }
    VirtualShadowMapLODContextGPU context = vsmLODContext(vsmCacheMetadata(push, 50u));
    if (view >= context.viewCount) { return false; }
    StructuredBuffer<uint> triangleClusters = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.triangleClusters));
    StructuredBuffer<VirtualShadowMapLODClusterGPU> clusters =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODClusterGPU>>(context.clusters));
    StructuredBuffer<uint> selected = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.selected));
    // Authored resident triangles and appended meshlets retain their exact
    // ordering; a 64-slot Build group may straddle several LOD cluster ranges.
    uint end = firstTriangle + min(64u, push.triangleCount - firstTriangle);
    uint triangle = firstTriangle;
    while (triangle < end) {
        uint index = triangleClusters[triangle];
        if (index == 0xffffffffu) { ++triangle; continue; }
        if (selected[view * context.clusterCount + index] != 0u) { return true; }
        VirtualShadowMapLODClusterGPU cluster = clusters[index];
        triangle = cluster.triangleFirst + cluster.triangleCount;
    }
    return false;
#endif
}

bool vsmLODPageChanged(ShadowPush push, uint view, uint2 page, bool coarse)
{
#if !VSM_GPU_LOD
    return false;
#else
    uint handle = vsmCacheMetadata(push, 50u);
    if (handle == 0xffffffffu) { return false; }
    VirtualShadowMapLODContextGPU context = vsmLODContext(handle);
    StructuredBuffer<uint> changed = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.changed));
    StructuredBuffer<VirtualShadowMapLODInstanceGPU> instances =
        resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODInstanceGPU>>(context.instances));
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection projection = projections[view];
    for (uint instanceIndex = 0u; instanceIndex < context.instanceCount; ++instanceIndex) {
        if (changed[view * context.instanceCount + instanceIndex] == 0u) { continue; }
        if (coarse && push.lightType == 0u) { return true; }
        VirtualShadowMapLODInstanceGPU instance = instances[instanceIndex];
        uint4 rectangle;
        if (vsmChangedBoundRect(instance.boundsMin.xyz, instance.boundsMax.xyz, projection, rectangle) &&
            all(page >= rectangle.xy) && all(page <= rectangle.zw)) { return true; }
    }
    return false;
#endif
}

void vsmLODPublishChangedCasters(ShadowPush push)
{
#if VSM_GPU_LOD
    uint handle = vsmCacheMetadata(push, 50u);
    if (handle == 0xffffffffu || (push.cacheMode & 1u) == 0u) { return; }
    VirtualShadowMapLODContextGPU context = vsmLODContext(handle);
    StructuredBuffer<uint> changed = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.changed));
    uint count = 0u;
    for (uint instanceIndex = 0u; instanceIndex < context.instanceCount; ++instanceIndex) {
        bool differs = false;
        for (uint view = 0u; view < context.viewCount; ++view) {
            differs = differs || changed[view * context.instanceCount + instanceIndex] != 0u;
        }
        count += differs ? 1u : 0u;
    }
    vsmCacheAdd(push, kVSMCacheLODChangedCasters, count);
#endif
}
