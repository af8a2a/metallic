#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_RECEIVER_COVERAGE_DIAGNOSTICS
#define VSM_RECEIVER_COVERAGE_DIAGNOSTICS 0
#endif
#ifndef VSM_LOCAL_MARK
#define VSM_LOCAL_MARK 0
#endif
#ifndef VSM_DIRECTIONAL_MARK
#define VSM_DIRECTIONAL_MARK 0
#endif
#ifndef VSM_MARK_PAGE_SHIFT
#define VSM_MARK_PAGE_SHIFT -1
#endif
#ifndef VSM_MARK_CACHE_LEVELS
#define VSM_MARK_CACHE_LEVELS 12
#endif

#if VSM_RECEIVER_COVERAGE_DIAGNOSTICS
// Diagnostic-only observation of the directional cached Mark domain/address.
// This does not request a page or update the per-receiver cache.
uint vsmReceiverCoveragePage(float3 position, StructuredBuffer<ShadowProjection> projections,
    ShadowPush push, uint level)
{
    float4 samplePosition = float4(position, 1.0);
    float sampleZ = dot(projections[0].rowZ, samplePosition);
    uint pageAxis = push.virtualPagesPerAxis;
    float resolution = float(pageAxis * push.pageSize);
    float margin = 1.0 / resolution;
    float2 projected = float2(dot(projections[level].rowX, samplePosition),
        dot(projections[level].rowY, samplePosition));
    if (!(all(projected >= -1.0 + margin) && all(projected < 1.0 - margin) &&
        sampleZ >= 0.0 && sampleZ <= 1.0)) { return 0xffffffffu; }
    uint2 texel = uint2(floor((projected * 0.5 + 0.5) * resolution));
#if VSM_MARK_PAGE_SHIFT >= 0
    uint2 pageXY = texel >> VSM_MARK_PAGE_SHIFT;
#else
    uint2 pageXY = texel / push.pageSize;
#endif
    return level * (pageAxis * pageAxis) + pageXY.y * pageAxis + pageXY.x;
}
#endif

#if VSM_DIRECTIONAL_MARK
#if VSM_RECEIVER_MASK
void vsmWriteReceiverMask(ShadowPush push, uint page, uint2 mask)
{
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint offset = vsmReceiverMaskOffset(push) + page * 2u;
    if (mask.x != 0u) { InterlockedOr(owners[offset], mask.x); }
    if (mask.y != 0u) { InterlockedOr(owners[offset + 1u], mask.y); }
    vsmWorkAdd(push, kVSMWorkMarkReceiverMaskAtomicOrs, uint(mask.x != 0u) + uint(mask.y != 0u));
}

struct VSMReceiverMaskAtomic
{
    Atomic<uint> x;
    Atomic<uint> y;
};
groupshared VSMReceiverMaskAtomic gVsmReceiverMasks[kVSMMarkHashCapacity];

[inline]
bool vsmAccumulateReceiverMask(uint page, uint2 mask)
{
    uint slot;
    if (!vsmClaimMarkSlot(page, slot)) { return false; }
    if ((mask.x | mask.y) == 0u) { return true; }

    uint curX = gVsmReceiverMasks[slot].x.load(MemoryOrder.Relaxed);
    uint curY = gVsmReceiverMasks[slot].y.load(MemoryOrder.Relaxed);
    if (((curX & mask.x) == mask.x) && ((curY & mask.y) == mask.y)) {
        return true;
    }
    if ((curX & mask.x) != mask.x) {
        gVsmReceiverMasks[slot].x.or(mask.x, MemoryOrder.Relaxed);
    }
    if ((curY & mask.y) != mask.y) {
        gVsmReceiverMasks[slot].y.or(mask.y, MemoryOrder.Relaxed);
    }
    return true;
}
#endif

struct VSMDirectionalFootprintContext
{
    float3 toLight;
    float3 right;
    float3 up;
    float rayLength;
    float coneRadius;
    float texelSize;
    float3 commonPos;
};

float2 vsmDirectionalFootprintRange(float4 row, float rowLength, float3 position, VSMDirectionalFootprintContext ctx)
{
    float center = dot(row, float4(position, 1.0));
    float axial = ctx.rayLength * dot(row.xyz, ctx.toLight);
    float transverse = ctx.coneRadius * rowLength;
    float dither = 0.5 * ctx.texelSize * (abs(dot(row.xyz, ctx.right)) + abs(dot(row.xyz, ctx.up)));
    // Include the original endpoint, world-space lerp and texel dither paths.
    // The axial term also retains finite-precision XY/light-axis nonorthogonality.
    float magnitude = dot(abs(row.xyz), ctx.commonPos) + abs(row.w) + 1.0;
    float error = magnitude * (256.0 * 5.960464477539063e-8);
    float spread = transverse + dither + error;
    return float2(center + min(0.0, axial) - spread, center + max(0.0, axial) + spread);
}

float2 vsmDirectionalFootprintRange(float4 row, float3 position, VSMDirectionalFootprintContext ctx)
{
    return vsmDirectionalFootprintRange(row, length(row.xyz), position, ctx);
}

struct VSMDirectionalAxisBase
{
    float centerBase;
    float axialBase;
    float transverseDitherBase;
    float commonMagnitudeBase;
};

VSMDirectionalAxisBase vsmCreateDirectionalAxisBase(float4 firstRow, float firstRowLength, float3 position, VSMDirectionalFootprintContext ctx)
{
    VSMDirectionalAxisBase b;
    b.centerBase = dot(firstRow.xyz, position);
    b.axialBase = ctx.rayLength * dot(firstRow.xyz, ctx.toLight);
    float transverse = ctx.coneRadius * firstRowLength;
    float dither = 0.5 * ctx.texelSize * (abs(dot(firstRow.xyz, ctx.right)) + abs(dot(firstRow.xyz, ctx.up)));
    b.transverseDitherBase = transverse + dither;
    b.commonMagnitudeBase = dot(abs(firstRow.xyz), ctx.commonPos);
    return b;
}

float2 vsmDirectionalFootprintFastRange(VSMDirectionalAxisBase baseAxis, float rowW, float scale)
{
    float center = baseAxis.centerBase * scale + rowW;
    float axial = baseAxis.axialBase * scale;
    float magnitude = baseAxis.commonMagnitudeBase * scale + abs(rowW) + 1.0;
    float error = magnitude * (256.0 * 5.960464477539063e-8);
    float spread = baseAxis.transverseDitherBase * scale + error;
    return float2(center + min(0.0, axial) - spread, center + max(0.0, axial) + spread);
}

void vsmMarkDirectionalFootprint(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel,
    float3 toLight, float3 right, float3 up, float sinAngularRadius, inout uint3 workCounts)
{
    ++workCounts.x;
    // The receiver-selected texel sets dither for every sampled clipmap.
    ShadowProjection firstProj = projections[firstLevel];
    float texelSize = firstProj.metric.y;
    float coneRadius = push.rayLength * sinAngularRadius;
    VSMDirectionalFootprintContext ctx;
    ctx.toLight = toLight;
    ctx.right = right;
    ctx.up = up;
    ctx.rayLength = push.rayLength;
    ctx.coneRadius = coneRadius;
    ctx.texelSize = texelSize;
    ctx.commonPos = abs(position) + (push.rayLength + texelSize + 1.0);

    float2 depthRange = vsmDirectionalFootprintRange(projections[0].rowZ, position, ctx);
    if (all(isfinite(depthRange)) && (depthRange.y < 0.0 || depthRange.x > 1.0)) { return; }

    uint virtualPages = push.virtualPagesPerAxis;
#if VSM_MARK_PAGE_SHIFT >= 0
    uint axisTexels = virtualPages << VSM_MARK_PAGE_SHIFT;
#else
    uint axisTexels = virtualPages * push.pageSize;
#endif
    float resolution = float(axisTexels);
    uint2 defaultLastPage = uint2(virtualPages - 1u);
    uint2 defaultLastTexel = uint2(axisTexels - 1u);

    float firstInvExtent = length(firstProj.rowX.xyz);
    VSMDirectionalAxisBase baseAxisX = vsmCreateDirectionalAxisBase(firstProj.rowX, firstInvExtent, position, ctx);
    VSMDirectionalAxisBase baseAxisY = vsmCreateDirectionalAxisBase(firstProj.rowY, firstInvExtent, position, ctx);

#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
    uint subShift = uint(VSM_MARK_PAGE_SHIFT - 3);
    uint subMask = push.pageSize - 1u;
#endif
#endif

    for (uint level = firstLevel; level < push.clipmapCount; ++level) {
        ++workCounts.y;
        float rowXw = projections[level].rowX.w;
        float rowYw = projections[level].rowY.w;
        float scale = asfloat(uint(127 - int(level - firstLevel)) << 23u);
        float2 xRange = vsmDirectionalFootprintFastRange(baseAxisX, rowXw, scale);
        float2 yRange = vsmDirectionalFootprintFastRange(baseAxisY, rowYw, scale);
        float2 low = float2(xRange.x, yRange.x), high = float2(xRange.y, yRange.y);
        uint2 firstPage = uint2(0u), lastPage = defaultLastPage;
        uint2 firstTexel = uint2(0u), lastTexel = defaultLastTexel;
        // Unbounded numerical ranges retain the full grid, never an empty request.
        if (all(isfinite(low)) && all(isfinite(high)) && all(low <= high)) {
            if (any(high < -1.0) || any(low > 1.0)) { continue; }
            // Intersect before float-to-uint conversion; one texel covers rounding
            // of the projected-domain-to-texel conversion and its floor boundary.
            float halfRes = 0.5 * resolution;
            float maxRes = resolution - 1.0;
            firstTexel = uint2(clamp(fma(clamp(low, -1.0, 1.0), float2(halfRes), float2(halfRes - 1.0)),
                0.0, maxRes));
            lastTexel = uint2(clamp(fma(clamp(high, -1.0, 1.0), float2(halfRes), float2(halfRes + 1.0)),
                0.0, maxRes));
#if VSM_MARK_PAGE_SHIFT >= 0
            firstPage = firstTexel >> VSM_MARK_PAGE_SHIFT;
            lastPage = lastTexel >> VSM_MARK_PAGE_SHIFT;
#else
            firstPage = firstTexel / push.pageSize;
            lastPage = lastTexel / push.pageSize;
#endif
        }
        uint2 pageBase = projections[level].page.xy;
        if (all(firstPage == lastPage)) {
            uint page = pageBase.x + firstPage.y * pageBase.y + firstPage.x;
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
            uint2 firstSub = (firstTexel & subMask) >> subShift;
            uint2 lastSub = (lastTexel & subMask) >> subShift;
            uint2 mask = vsmReceiverRectangleMask(firstSub, lastSub);
#else
            uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel, firstPage, push.pageSize);
#endif
            if (!vsmAccumulateReceiverMask(page, mask)) {
                vsmWriteReceiverMask(push, page, mask);
                vsmPublishMarkPage(pages, page, workCounts);
            }
#else
            vsmMarkPage(pages, page, workCounts);
#endif
        } else {
            for (uint y = firstPage.y; y <= lastPage.y; ++y) {
                uint rowBasePage = pageBase.x + y * pageBase.y;
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
                uint firstSubY = (y == firstPage.y) ? ((firstTexel.y & subMask) >> subShift) : 0u;
                uint lastSubY = (y == lastPage.y) ? ((lastTexel.y & subMask) >> subShift) : 7u;
                bool fullY = (firstSubY == 0u && lastSubY == 7u);
#endif
#endif
                for (uint x = firstPage.x; x <= lastPage.x; ++x) {
                    uint page = rowBasePage + x;
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
                    uint firstSubX = (x == firstPage.x) ? ((firstTexel.x & subMask) >> subShift) : 0u;
                    uint lastSubX = (x == lastPage.x) ? ((lastTexel.x & subMask) >> subShift) : 7u;
                    uint2 mask = (fullY && firstSubX == 0u && lastSubX == 7u) ?
                        uint2(0xffffffffu, 0xffffffffu) :
                        vsmReceiverRectangleMask(uint2(firstSubX, firstSubY), uint2(lastSubX, lastSubY));
#else
                    // Page deduplication must not discard another receiver's subtiles.
                    uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel, uint2(x, y), push.pageSize);
#endif
                    if (!vsmAccumulateReceiverMask(page, mask)) {
                        vsmWriteReceiverMask(push, page, mask);
                        vsmPublishMarkPage(pages, page, workCounts);
                    }
#else
                    vsmMarkPage(pages, page, workCounts);
#endif
                }
            }
        }
    }
}
#endif

// Groupshared receiver basis and directional uniform structures have been replaced
// with direct register/scalar loads to eliminate intra-workgroup barriers and serialization.

void vsmMarkReceiver(uint3 id, ShadowPush push, out uint3 workCounts, out uint receiverDispatch)
{
    workCounts = uint3(0u);
    receiverDispatch = 0u;
    if (push.enabled == 0u) { return; }
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    VSMReceiverBasis basis = vsmMakeReceiverBasis(views[0]);
#if VSM_RECEIVER_COVERAGE_DIAGNOSTICS
    // CPU enables this permutation only for one directional SMRT recording.
    // Every in-viewport pixel owns one record, including rejected receivers.
    if (any(id.xy >= uint2(basis.viewportSize))) { return; }
    RWStructuredBuffer<uint> coverageData = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint coverageRecord = kVSMReceiverCoverageStatisticsWordOffset +
        (id.y * uint(basis.viewportSize.y) + id.x) * kVSMReceiverCoverageRecordWords;
    coverageData[coverageRecord] = 0u;
#endif
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
    if (!vsmReceiver(int2(id.xy), basis, depth, position, linearDepth)) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    bool nearPlane;
    if (push.lightType == 2u && push.filterMode != 0u) {
        float3 delta = abs(position - projections[0].originNear.xyz);
        if ((push.lightRange > 0.0 && length(delta) >= push.lightRange) ||
            max(delta.x, max(delta.y, delta.z)) <= projections[0].originNear.w) { return; }
    } else if (!vsmReceiverDomain(position, projections[0], push, nearPlane)) { return; }
    receiverDispatch = 1u;
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    // Resolution belongs to the receiver, independently of the filter kernel.
    float3 normal = float3(0.0);
    if (push.receiverFootprint != 0u || push.filterMode == 1u || push.lightType == 0u) {
        normal = vsmReceiverNormal(int2(id.xy), basis, depth, position, linearDepth);
    }
    float3 toLight = float3(0.0);
    float sinAngularRadius = 0.0;
    if (push.lightType == 0u) {
        toLight = normalize(projections[0].rowZ.xyz);
        sinAngularRadius = sin(push.angularRadius);
        float sinAlpha = max(sinAngularRadius, 0.1);
        if (dot(normal, toLight) < -sinAlpha) { return; }
    }
    uint receiverLevel = vsmReceiverLevel(position, projections, push,
        vsmProjectedReceiverFootprint(position, normal, linearDepth, basis, projections[0], push));
    if (push.filterMode == 0u) {
        vsmMarkPosition(position, projections, pages, push, receiverLevel, workCounts);
        return;
    }
    if (push.filterMode == 1u) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                vsmMarkPosition(vsmPCFPosition(position, normal, projections[receiverLevel], push, int2(x, y)),
                    projections, pages, push, receiverLevel, workCounts);
            }
        }
    } else if (push.filterMode == 2u) {
        ShadowPush trace = push;
        trace.rayLength = vsmSearchLength(position, linearDepth, basis, push);
#if VSM_RECEIVER_COVERAGE_DIAGNOSTICS
        coverageData[coverageRecord + 1u] = asuint(position.x);
        coverageData[coverageRecord + 2u] = asuint(position.y);
        coverageData[coverageRecord + 3u] = asuint(position.z);
        coverageData[coverageRecord + 4u] = asuint(trace.rayLength);
        coverageData[coverageRecord + 5u] = receiverLevel;
        coverageData[coverageRecord + 6u] = 0u;
        coverageData[coverageRecord + 7u] = 0u;
        coverageData[coverageRecord + 8u] = vsmReceiverCoveragePage(position, projections, push, receiverLevel);
        [unroll]
        for (uint level = 0u; level < kVSMReceiverCoverageLevelCapacity; ++level) {
            coverageData[coverageRecord + kVSMReceiverCoverageWitnessOffset + level] = 0xffffffffu;
        }
        coverageData[coverageRecord + kVSMReceiverCoverageLinearDepthOffset] = asuint(linearDepth);
        coverageData[coverageRecord] = 1u;
#endif
#if VSM_DIRECTIONAL_MARK
        float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
        float3 up = cross(toLight, right);
        vsmMarkDirectionalFootprint(position, projections, pages, trace, receiverLevel,
            toLight, right, up, sinAngularRadius, workCounts);
#else
        uint lastPages[VSM_MARK_CACHE_LEVELS];
        [unroll]
        for (uint level = 0u; level < VSM_MARK_CACHE_LEVELS; ++level) { lastPages[level] = 0xffffffffu; }
#if VSM_LOCAL_MARK
        ShadowProjection originProjection = projections[0], receiverProjection = projections[receiverLevel];
        float3 toLight = push.lightType == 0u ? normalize(originProjection.rowZ.xyz) :
            normalize(originProjection.originNear.xyz - position);
        float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
        float3 up = cross(toLight, right);
        // Directional normalization must retain the receiver level's rowZ;
        // do not infer bitwise equality from scaled directions.
        float3 ditherToLight = push.lightType == 0u ? normalize(receiverProjection.rowZ.xyz) :
            normalize(receiverProjection.originNear.xyz - position);
        float3 ditherRight = normalize(cross(abs(ditherToLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), ditherToLight));
        float3 ditherUp = cross(ditherToLight, ditherRight);
        float texelSize = vsmWorldTexelSize(position, receiverProjection, push.pageSize, push.lightType);
        float2 scramble = float2(vsmSampleRandom(id.xy, 0u, push.samplePhase), vsmSampleRandom(id.xy, 1u, push.samplePhase));
        float stepRandom = vsmSampleRandom(id.xy, 2u, push.samplePhase);
        float3 forward = float3(0.0);
        float startW = 0.0;
        if (push.lightType != 0u) {
            forward = normalize(originProjection.rowW.xyz);
            if (push.lightType == 2u) {
                float3 faceRight, faceUp;
                virtualShadowMapPointBasis(virtualShadowMapPointFace(position - originProjection.originNear.xyz), faceRight, faceUp, forward);
            }
            startW = dot(position - originProjection.originNear.xyz, forward);
        }
#endif
        float invRayCount = 1.0 / float(push.rayCount);
        float invSamplesPerRay = 1.0 / float(push.samplesPerRay);
        for (uint ray = 0u; ray < push.rayCount; ++ray) {
#if VSM_LOCAL_MARK
            float2 disk = vsmDiskSample(ray, invRayCount, scramble);
            float3 endpoint = vsmRayEndpoint(position, originProjection, trace, toLight, right * disk.x + up * disk.y);
            float2 offset = float2(vsmSampleRandom(id.xy, 3u + 2u * ray, push.samplePhase),
                vsmSampleRandom(id.xy, 4u + 2u * ray, push.samplePhase)) - 0.5;
            float3 dither = (ditherRight * offset.x + ditherUp * offset.y) * texelSize;
            float endW = push.lightType != 0u ? dot(endpoint - originProjection.originNear.xyz, forward) : 0.0;
#else
            float3 endpoint = vsmRayEndpoint(position, projections[0], trace, ray, id.xy);
            float3 dither = vsmRayDither(position, projections[receiverLevel], trace, ray, id.xy);
#endif
            for (uint step = 0u; step <= push.samplesPerRay; ++step) {
#if VSM_LOCAL_MARK
                float t = 1.0 - (float(step + 1u) - stepRandom) * invSamplesPerRay;
                float time = step == push.samplesPerRay ? 0.0 : t * t;
                if (push.lightType != 0u) { time = time * startW / lerp(endW, startW, time); }
                float3 point = lerp(position, endpoint, time);
#else
                float time = vsmRaySampleTime(step, push, id.xy, invSamplesPerRay);
                float3 point = vsmRayPosition(position, endpoint, time, projections[0], push);
#endif
                vsmMarkPosition(point + dither, projections, pages, push, receiverLevel, lastPages, workCounts);
            }
        }
#endif
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmMarkMain(uint3 id : SV_DispatchThreadID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    // All 8x8 lanes initialize and synchronize before receiver/tail rejection.
    for (uint entry = lane; entry < kVSMMarkHashCapacity; entry += 64u) {
        gVsmMarkedPages[entry].store(0xffffffffu, MemoryOrder.Relaxed);
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_MASK
        gVsmReceiverMasks[entry].x.store(0u, MemoryOrder.Relaxed);
        gVsmReceiverMasks[entry].y.store(0u, MemoryOrder.Relaxed);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 counts;
    uint receiverDispatch;
    vsmMarkReceiver(id, push, counts, receiverDispatch);
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_MASK
    // All mask contributions are complete before publishing requests and masks.
    GroupMemoryBarrierWithGroupSync();
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    for (uint entry = lane; entry < kVSMMarkHashCapacity; entry += 64u) {
        uint page = gVsmMarkedPages[entry].load(MemoryOrder.Relaxed);
        if (page != 0xffffffffu) {
            uint2 mask = uint2(gVsmReceiverMasks[entry].x.load(MemoryOrder.Relaxed),
                gVsmReceiverMasks[entry].y.load(MemoryOrder.Relaxed));
            vsmWriteReceiverMask(push, page, mask);
            vsmPublishMarkPage(pages, page, counts);
        }
    }
#endif
    uint4 totals = WaveActiveSum(uint4(counts, receiverDispatch));
    if (WaveIsFirstLane()) {
        if (totals.w != 0u) { vsmWorkAdd(push, kVSMWorkMarkReceiverDispatches, totals.w); }
        if (totals.x != 0u) { vsmWorkAdd(push, kVSMWorkMarkSamplePositions, totals.x); }
        if (totals.y != 0u) { vsmWorkAdd(push, kVSMWorkMarkLevelTests, totals.y); }
        if (totals.z != 0u) { vsmWorkAdd(push, kVSMWorkMarkPageRequests, totals.z); }
    }
}

