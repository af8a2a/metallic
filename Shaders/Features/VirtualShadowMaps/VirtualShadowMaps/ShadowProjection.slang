#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_OPTIMIZE_PROJECTION_STATISTICS
#define VSM_OPTIMIZE_PROJECTION_STATISTICS 0
#endif
#ifndef VSM_ONE_PASS_PROJECTION
#define VSM_ONE_PASS_PROJECTION 0
#endif
#ifndef VSM_LIGHT_GRID_PROJECTION
#define VSM_LIGHT_GRID_PROJECTION 0
#endif
#ifndef VSM_REFINED_RECEIVER
#define VSM_REFINED_RECEIVER 0
#endif
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif

#if VSM_ONE_PASS_PROJECTION
// Common filter settings stay in push constants. Only face-specific values
// come from this table; its 80-byte layout matches ShadowProjectionJob in C++.
struct ShadowProjectionJob
{
    uint pages, statistics, projections, atlas;
    uint filterMode, shadow, debug, physicalPagesX;
    uint clipmapCount, enabled, lightType, projectionFace;
    uint pointFaces, projectionFlags, samplePhase;
    float lightRange, sourceRadius, angularRadius, lodScreenRayLength;
    uint receiverGeometry;
};

#endif

struct VSMProjectionReceiver
{
    VSMReceiverBasis basis;
    float3 position;
    float linearDepth;
    bool valid;
    float searchLength;
    float3 normal;
    VSMReceiverFootprint footprint;
    bool hasNormal;
#if VSM_ONE_PASS_PROJECTION && VSM_WORKLOAD_DIAGNOSTICS
    uint normalUses;
#endif
};

bool vsmLoadProjectionReceiver(uint2 pixel, ShadowPush push, out VSMProjectionReceiver receiver)
{
    receiver = {};
    ViewConstants view = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view))[0];
    receiver.basis = vsmMakeReceiverBasis(view);
    if (any(pixel >= uint2(receiver.basis.viewportSize))) { return false; }
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    receiver.valid = vsmReceiver(int2(pixel), receiver.basis, depth, receiver.position, receiver.linearDepth);
    receiver.searchLength = vsmSearchLength(receiver.position, receiver.linearDepth, receiver.basis, push);
    return true;
}

bool vsmProjectReceiver(uint3 id, ShadowPush push, inout VSMProjectionReceiver receiver,
    out float visibility, out float4 debugValue, out uint3 filterCounts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    , out uint projectionCounts
#endif
    , bool useGroupCache = false
)
{
    visibility = 1.0;
    debugValue = float4(0.0, 0.0, 0.0, 1.0);
    filterCounts = uint3(0u);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    projectionCounts = 0u;
#endif
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position = receiver.position;
    float linearDepth = receiver.linearDepth;
#if VSM_DIRECTIONAL_PROJECTION
    const bool sharedPointLayer = false;
#else
    bool sharedPointLayer = (push.projectionFlags & 1u) != 0u && push.lightType == 2u;
#endif
    if (!receiver.valid) {
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        if (push.filterMode != 0u) { projectionCounts |= 1u << 0u; }
        else
#endif
        InterlockedAdd(statistics[8], 1u);
        return !sharedPointLayer || push.projectionFace == 0u;
    }
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    if (push.filterMode != 0u) { projectionCounts |= 1u << 1u; }
    else
#endif
    InterlockedAdd(statistics[6], 1u);
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection proj0 = vsmGetCachedProjection(projections, 0u, useGroupCache);
#if !VSM_DIRECTIONAL_PROJECTION
    if (sharedPointLayer && virtualShadowMapPointFace(position - proj0.originNear.xyz) != push.projectionFace) { return false; }
#endif
    if (push.enabled == 0u) { return true; }
    bool nearPlane;
    if (!vsmReceiverDomain(position, proj0, push, nearPlane)) {
        if (nearPlane) {
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
            if (push.filterMode != 0u) { projectionCounts |= 1u << 2u; }
            else
#endif
            InterlockedAdd(statistics[29], 1u);
        }
        return true;
    }
    vsmWorkAdd(push, kVSMWorkProjectionReceivers, 1u);
    if (!receiver.hasNormal) {
        receiver.normal = vsmReceiverNormal(int2(id.xy), receiver.basis, depth, position, linearDepth);
        receiver.footprint = vsmReceiverFootprint(position, receiver.normal, linearDepth, receiver.basis, push);
        receiver.hasNormal = true;
    }
#if VSM_ONE_PASS_PROJECTION && VSM_WORKLOAD_DIAGNOSTICS
    ++receiver.normalUses;
#endif
    float3 normal = receiver.normal;
    if (push.lightType == 0u) {
        float3 toLight = normalize(proj0.rowZ.xyz);
        float sinAlpha = max(sin(push.angularRadius), 0.1);
        if (dot(normal, toLight) < -sinAlpha) {
            visibility = 0.0;
            return true;
        }
    }
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint receiverLevel = vsmReceiverLevel(position, projections, push,
        vsmProjectedReceiverFootprint(position, receiver.footprint, proj0, push), useGroupCache);
    uint selectedLevel = push.clipmapCount;
    uint firstLevel = 0xffffffffu;
    uint physicalPage = 0xffffffffu;
    uint2 virtualTexel = uint2(0u);
    float3 projected = float3(0.0);
    // Any fine-work overflow invalidates the entire partial fine atlas. The
    // independent coarse draw always traverses every caster slot.
    bool allowClipmaps = statistics[5] == 0u;
#if VSM_DIRECTIONAL_PROJECTION
    bool ownReceiverMapping = false;
#endif
    if (allowClipmaps) {
        for (uint level = receiverLevel; level < push.clipmapCount; ++level) {
            ShadowProjection projection = vsmGetCachedProjection(projections, level, useGroupCache);
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            if (firstLevel == 0xffffffffu) { firstLevel = level; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint page = vsmVirtualPage(texel, projection, push.pageSize);
            uint mappedPhysical, mappedLevel;
#if VSM_DIRECTIONAL_PROJECTION
            bool isOwnMapping;
            if (vsmLookupPropagated(position, float2(projection.rowX.w, projection.rowY.w),
                    page, p, texel, projections, pages, push, level,
                    mappedPhysical, mappedLevel, isOwnMapping, useGroupCache)) {
                if (level == receiverLevel && isOwnMapping) { ownReceiverMapping = true; }
#else
            if (vsmLookupPropagated(position, float2(projection.rowX.w, projection.rowY.w),
                    page, p, texel, projections, pages, push, level,
                    mappedPhysical, mappedLevel, useGroupCache)) {
#endif
                if (mappedLevel == push.clipmapCount) { break; }
                physicalPage = mappedPhysical; selectedLevel = mappedLevel;
                projected = p; virtualTexel = texel;
                break;
            }
            vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
            uint4 entry = pages[page];
            if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
            physicalPage = entry.x;
            selectedLevel = level;
            projected = p;
            virtualTexel = texel;
            break;
        }
    }
    if (physicalPage == 0xffffffffu) {
        ShadowProjection coarse = vsmGetCachedProjection(projections, push.clipmapCount, useGroupCache);
        projected = vsmProjectPosition(position, coarse);
        if (!vsmContains(projected, coarse, push.pageSize, false)) {
            visibility = 0.0;
            debugValue = float4(1.0, 0.0, 1.0, 1.0);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
            if (push.filterMode != 0u) { projectionCounts |= 1u << 3u; }
            else
#endif
            InterlockedAdd(statistics[9], 1u);
            return true;
        }
        physicalPage = pages[coarse.page.x].x;
        virtualTexel = vsmVirtualTexel(projected, coarse, push.pageSize);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        if (push.filterMode != 0u) { projectionCounts |= 1u << 4u; }
        else
#endif
        InterlockedAdd(statistics[7], 1u);
        debugValue = float4(0.0, 0.0, 1.0, 1.0);
    } else {
        if (selectedLevel > firstLevel) {
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
            if (push.filterMode != 0u) { projectionCounts |= 1u << 5u; }
            else
#endif
            InterlockedAdd(statistics[13], 1u);
        }
        debugValue = float4(float(selectedLevel + 1u) / float(push.clipmapCount), 0.5, 0.0, 1.0);
    }
    if (push.filterMode != 0u) {
        ShadowPush trace = push;
        trace.rayLength = receiver.searchLength;
#if VSM_DIRECTIONAL_PROJECTION
        visibility = vsmFilterVisibilityDirectional(position, normal, receiverLevel, id.xy,
            allowClipmaps, ownReceiverMapping, virtualTexel, physicalPage,
            projections, trace, filterCounts, useGroupCache);
#else
        visibility = vsmFilterVisibility(position, normal, receiverLevel, id.xy, projections, trace, filterCounts, useGroupCache);
#endif
        return true;
    }
    ShadowProjection projection = vsmGetCachedProjection(projections, selectedLevel, useGroupCache);
    float compareDepth;
#if !VSM_DIRECTIONAL_PROJECTION
    if (projection.metric.w > 0.5) {
        compareDepth = vsmPerspectiveReceiverDepth(position, normal, virtualTexel, projection, push.pageSize);
    } else
#endif
    {
        float3 axisDepth = normalize(projection.rowZ.xyz);
        float denominator = dot(normal, axisDepth);
        float correction = 0.0;
        if (abs(denominator) > 1e-5) {
            float2 sampleNDC = (float2(virtualTexel) + 0.5) *
                (2.0 / float(projection.page.y * push.pageSize)) - 1.0;
            float2 offset = (sampleNDC - projected.xy) * projection.metric.x;
            float3 deltaXYWorld = normalize(projection.rowX.xyz) * offset.x + normalize(projection.rowY.xyz) * offset.y;
            correction = -dot(normal, deltaXYWorld) * length(projection.rowZ.xyz) / denominator;
        }
        compareDepth = projected.z + correction + projection.metric.z;
    }
    uint2 physical = vsmPhysicalPageCoord(physicalPage, push.physicalPagesX);
    uint2 texel = vsmAtlasTexel(physical, vsmPageLocalTexel(virtualTexel, push.pageSize), push.pageSize);
    VSMAtlas atlas = resolveUniform(ResourceHandle<VSMAtlas>(push.atlas));
    float storedDepth = vsmAtlasLoad(atlas, texel, push);
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    if (push.lodScreenRayLength > 0.0 && storedDepth != 0.0 && storedDepth > compareDepth) {
        float depthScale = length(projection.rowZ.xyz);
        float bias = projection.metric.z / depthScale;
        float blockerDistance = (storedDepth - compareDepth) / depthScale + bias;
        ViewConstants view = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view))[0];
        float offset = vsmLODScreenRayOffset(position, normal, linearDepth, normalize(projection.rowZ.xyz),
            view, depth, push.lodScreenRayLength, blockerDistance, bias);
#if VSM_REFINED_RECEIVER
        if (compareDepth + offset * depthScale >= storedDepth &&
            !vsmAllowLODReceiverCorrection(id.xy, position, linearDepth, normalize(projection.rowZ.xyz),
                view, push.receiverGeometry, push.lodScreenRayLength)) { offset = 0.0; }
#endif
        compareDepth += offset * depthScale;
    }
    // The producer's scene-Z/far margin keeps real casters above zero. A receiver
    // plane extrapolated past the scene must not turn empty depth into a blocker.
    visibility = storedDepth == 0.0 || compareDepth >= storedDepth ? 1.0 : 0.0;
    return true;
}

// Ownership and filtering finish before either output is stored.
void vsmProjectAndStore(uint3 id, ShadowPush push, inout VSMProjectionReceiver receiver, out uint3 filterCounts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    , out uint projectionCounts
#endif
    , bool useGroupCache = false
)
{
    float visibility;
    float4 debugValue;
    if (!vsmProjectReceiver(id, push, receiver, visibility, debugValue, filterCounts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
        , useGroupCache
    )) { return; }
    RWTexture2D<float> shadow = resolveUniform(ResourceHandle<RWTexture2D<float>>(push.shadow));
    shadow[id.xy] = visibility;
    if (push.debug != 0xffffffffu) {
        RWTexture2D<float4> debug = resolveUniform(ResourceHandle<RWTexture2D<float4>>(push.debug));
        debug[id.xy] = debugValue;
    }
}

uint2 vsmMortonPixel(uint2 dispatchPixel)
{
    uint lane = (dispatchPixel.x & 7u) + ((dispatchPixel.y & 7u) << 3u);
    uint2 xy = uint2((lane & 1u) | ((lane >> 1u) & 2u) | ((lane >> 2u) & 4u),
        ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u) | ((lane >> 3u) & 4u));
    return (dispatchPixel & ~7u) + xy;
}

void vsmRecordProjectionStatistics(uint3 counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    , uint projectionCounts
#endif
    , ShadowPush push
)
{
    if (push.filterMode == 0u) { return; }
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    counts.z += (projectionCounts >> 3u) & 1u;
#endif
    // Integer subgroup reductions preserve the per-face totals without a
    // workgroup barrier or shared storage.
    uint3 totals = WaveActiveSum(counts);
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    if (WaveIsFirstLane()) {
        if (totals.x != 0u) { InterlockedAdd(statistics[30], totals.x); }
        if (totals.y != 0u) { InterlockedAdd(statistics[31], totals.y); }
        if (totals.z != 0u) { InterlockedAdd(statistics[9], totals.z); }
    }
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    [unroll]
    for (uint event = 0u; event < 5u; ++event) {
        uint bit = event < 3u ? event : event + 1u;
        uint total = WaveActiveSum((projectionCounts >> bit) & 1u);
        if (WaveIsFirstLane() && total != 0u) {
            uint index = event == 0u ? 8u : event == 1u ? 6u : event == 2u ? 29u : event == 3u ? 7u : 13u;
            InterlockedAdd(statistics[index], total);
        }
    }
#endif
}

#if !VSM_ONE_PASS_PROJECTION
[shader("compute")]
[numthreads(8, 8, 1)]
void vsmProjectMain(uint3 id : SV_DispatchThreadID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    id.xy = vsmMortonPixel(id.xy);
    VSMProjectionReceiver receiver;
    if (!vsmLoadProjectionReceiver(id.xy, push, receiver)) { return; }
    uint3 counts = uint3(0u);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    uint projectionCounts = 0u;
#endif
    vsmProjectAndStore(id, push, receiver, counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
    );
    vsmRecordProjectionStatistics(counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
        , push
    );
}
#else
void vsmProjectionGridAdd(uint diagnostics, uint counter, uint laneValue)
{
#if VSM_WORKLOAD_DIAGNOSTICS
    if (diagnostics == 0xffffffffu) { return; }
    uint value = WaveActiveSum(laneValue);
    if (WaveIsFirstLane() && value != 0u) {
        RWStructuredBuffer<uint> counters = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(diagnostics));
        uint previous;
        InterlockedAdd(counters[counter * 2u], value, previous);
        if (previous > 0xffffffffu - value) { InterlockedAdd(counters[counter * 2u + 1u], 1u); }
    }
#endif
}

void vsmProjectOnePassJob(uint3 id, ShadowPush common, ShadowProjectionJob job, inout VSMProjectionReceiver receiver
#if VSM_LIGHT_GRID_PROJECTION
    , bool member
#endif
)
{
    ShadowPush push = common;
    push.pages = job.pages; push.statistics = job.statistics;
    push.projections = job.projections; push.atlas = job.atlas;
    push.filterMode = job.filterMode; push.shadow = job.shadow;
    push.debug = job.debug; push.physicalPagesX = job.physicalPagesX;
    push.clipmapCount = job.clipmapCount; push.enabled = job.enabled;
    push.lightType = job.lightType; push.projectionFace = job.projectionFace;
    push.pointFaces = job.pointFaces; push.projectionFlags = job.projectionFlags;
    push.lodScreenRayLength = job.lodScreenRayLength;
    push.receiverGeometry = job.receiverGeometry;
    push.samplePhase = job.samplePhase; push.lightRange = job.lightRange;
    push.sourceRadius = job.sourceRadius; push.angularRadius = job.angularRadius;
    uint3 counts = uint3(0u);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    uint projectionCounts = 0u;
#endif
#if VSM_LIGHT_GRID_PROJECTION
    if (member) {
#endif
    vsmProjectAndStore(id, push, receiver, counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
    );
#if VSM_LIGHT_GRID_PROJECTION
    }
#endif
    vsmRecordProjectionStatistics(counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
        , push
    );
}

#if VSM_LIGHT_GRID_PROJECTION
uint2 vsmProjectionLocalMask(uint2 pixel, VSMProjectionReceiver receiver, VirtualShadowMapProjectionGrid context)
{
    uint2 mask = uint2(0u);
    uint candidatesCount = 0u, fallback = 0u;
    if (receiver.valid) {
        ClusterLightGridParams grid = resolveUniform(ResourceHandle<StructuredBuffer<ClusterLightGridParams>>(context.parameters))[0];
        StructuredBuffer<ClusterLightGridCell> cells = resolveUniform(ResourceHandle<StructuredBuffer<ClusterLightGridCell>>(context.cells));
        StructuredBuffer<uint> indices = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.lightIndices));
        StructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(context.candidates));
        ClusterLightGridLookup lookup = clusterLightGridLookup(grid, cells, pixel, receiver.linearDepth);
        fallback = lookup.fallback;
        uint globalCount = clusterLightGridGlobalLightCount(grid);
        candidatesCount = lookup.count + globalCount;
        for (uint i = 0u; i < candidatesCount; ++i) {
            uint source = i < lookup.count ? clusterLightGridLocalLightIndex(lookup, indices, candidates, i) :
                clusterLightGridGlobalLightIndex(grid, candidates, i - lookup.count);
            // LightsPass validates sourceSlotCount + two celestial slots <= 64.
            mask[source >> 5u] |= 1u << (source & 31u);
        }
        mask &= uint2(context.admittedLow, context.admittedHigh);
    }
    vsmProjectionGridAdd(context.diagnostics, kVSMGridReceiverLookups, receiver.valid ? 1u : 0u);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridFallbacks, fallback);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridLocalCandidates, candidatesCount);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridLocalMemberships, countbits(mask.x) + countbits(mask.y));
    return mask;
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmProjectOnePassMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush common)
{
    id.xy = vsmMortonPixel(id.xy);
    VSMProjectionReceiver receiver;
    if (!vsmLoadProjectionReceiver(id.xy, common, receiver)) { return; }
    StructuredBuffer<ShadowProjectionJob> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjectionJob>>(common.geometry));
#if VSM_LIGHT_GRID_PROJECTION
    StructuredBuffer<VirtualShadowMapProjectionGrid> contexts = resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjectionGrid>>(common.arguments));
    VirtualShadowMapProjectionGrid context = contexts[0];
    uint2 localMask = vsmProjectionLocalMask(id.xy, receiver, context);
    // Publication slots include Sun/Moon. A single traversal keeps the large
    // filtering body at one call site, rather than cloning it for each domain.
    uint2 laneMask = uint2(localMask.x << 2u, (localMask.y << 2u) | (localMask.x >> 30u));
    laneMask.x |= (contexts[0].jobs[0].count != 0u ? 1u : 0u) |
        (contexts[0].jobs[1].count != 0u ? 2u : 0u);
    uint2 waveMask = uint2(WaveActiveBitOr(laneMask.x), WaveActiveBitOr(laneMask.y));
    // Keep every active lane on the same job: descriptor indices and integer
    // reductions remain uniform even when receivers occupy different Z cells.
    uint waveJobs = 0u, laneJobs = 0u;
    for (uint word = 0u; word < 2u; ++word) {
        uint pending = waveMask[word];
        while (pending != 0u) {
            uint bit = uint(firstbitlow(pending));
            pending &= pending - 1u;
            uint slot = word * 32u + bit;
            bool member = (laneMask[word] & (1u << bit)) != 0u;
            VirtualShadowMapProjectionJobRange range = contexts[0].jobs[slot];
            if (slot >= 2u) {
                waveJobs += range.count;
                laneJobs += member ? range.count : 0u;
            }
            for (uint face = 0u; face < range.count; ++face) {
                vsmProjectOnePassJob(id, common, jobs[range.first + face], receiver, member);
            }
        }
    }
    vsmProjectionGridAdd(context.diagnostics, kVSMGridWaveLocalJobs, WaveIsFirstLane() ? waveJobs : 0u);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridSkippedWaveLocalJobs, WaveIsFirstLane() ? context.localJobCount - waveJobs : 0u);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridProjectedLaneJobs, laneJobs);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridPossibleLaneJobs, context.localJobCount);
#else
    // Without a grid, every in-bounds lane visits the complete job table.
    for (uint jobIndex = 0u; jobIndex < common.triangleCount; ++jobIndex) {
        ShadowProjectionJob job = jobs[jobIndex];
        vsmProjectOnePassJob(id, common, job, receiver);
    }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
    if (common.arguments != 0xffffffffu) {
#if !VSM_LIGHT_GRID_PROJECTION
        VirtualShadowMapProjectionGrid context = resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjectionGrid>>(common.arguments))[0];
        vsmProjectionGridAdd(context.diagnostics, kVSMGridWaveLocalJobs, WaveIsFirstLane() ? context.localJobCount : 0u);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridProjectedLaneJobs, context.localJobCount);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridPossibleLaneJobs, context.localJobCount);
#endif
        vsmProjectionGridAdd(context.diagnostics, kVSMGridReceiverCenterLoads, 1u);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridReceiverNormalBuilds, uint(receiver.hasNormal));
        vsmProjectionGridAdd(context.diagnostics, kVSMGridReceiverNormalUses, receiver.normalUses);
    }
#endif
}
#endif
