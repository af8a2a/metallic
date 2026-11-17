#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif
#ifndef VSM_PROJECTION_PAGE_SHIFT
#define VSM_PROJECTION_PAGE_SHIFT -1
#endif

#if VSM_DIRECTIONAL_PROJECTION
// makeProjections owns this contract: fine XY axes are dyadic, every level
// (including coarse) shares rowZ, and the separate coarse XY encloses the scene.
// Keep both origins: lookup has texel dither; receiver-plane comparison does not.
struct VSMDirectionalRayBase
{
    float3 comparisonOrigin;
    float sourceExtent;
    float2 sourceTranslation;
    uint receiverLevel;
    uint receiverPageOffset;
    uint receiverMappingOffset;
    uint coarseLevel;
    float invDepthScale;
    float halfReceiverResolution;
    float receiverMarginMax;
    bool allowClipmaps;
#if VSM_PROJECTION_PAGE_SHIFT < 0
    uint pageSizeShift;
    uint pageSizeMask;
#endif
    uint physicalPagesXShift;
    uint physicalPagesXMask;
    uint virtualPagesPerAxisShift;
    bool powerOfTwoLayout;
    bool powerOfTwoPhysical;
};

struct VSMDirectionalRay
{
    float3 lookupOrigin;
    float3 step;
};

struct VSMDirectionalRayState
{
    VSMDirectionalRayBase base;
    VSMDirectionalRay ray;
    float depthScale;

    property float3 comparisonOrigin { get { return base.comparisonOrigin; } }
    property float3 lookupOrigin { get { return ray.lookupOrigin; } }
    property float3 step { get { return ray.step; } }
    property float2 sourceTranslation { get { return base.sourceTranslation; } }
    property uint receiverLevel { get { return base.receiverLevel; } }
    property uint receiverPageOffset { get { return base.receiverPageOffset; } }
    property uint coarseLevel { get { return base.coarseLevel; } }
    property float receiverDepth { get { return base.comparisonOrigin.z; } }
    property float invDepthScale { get { return base.invDepthScale; } }
    property float receiverResolution { get { return 2.0 * base.halfReceiverResolution; } }
    property float receiverMargin { get { return 1.0 - base.receiverMarginMax; } }
    property float halfReceiverResolution { get { return base.halfReceiverResolution; } }
    property float receiverMarginMax { get { return base.receiverMarginMax; } }
    property bool allowClipmaps { get { return base.allowClipmaps; } }
    property uint receiverMappingOffset { get { return base.receiverMappingOffset; } }
    property uint virtualPagesPerAxisShift { get { return base.virtualPagesPerAxisShift; } }
    property bool powerOfTwoLayout { get { return base.powerOfTwoLayout; } }
    property bool powerOfTwoPhysical { get { return base.powerOfTwoPhysical; } }
};

VSMDirectionalRayBase vsmCreateDirectionalRayBase(float3 position,
    ShadowProjection first, StructuredBuffer<uint4> pages, ShadowPush push, uint receiverLevel)
{
    VSMDirectionalRayBase rayBase;
    rayBase.comparisonOrigin = vsmProjectPosition(position, first);
    rayBase.sourceExtent = first.metric.x;
    rayBase.sourceTranslation = float2(first.rowX.w, first.rowY.w);
    rayBase.receiverLevel = receiverLevel;
    rayBase.coarseLevel = push.clipmapCount;
    rayBase.receiverPageOffset = first.page.x;
    rayBase.receiverMappingOffset = push.totalPages + first.page.x;
    rayBase.invDepthScale = 1.0 / length(first.rowZ.xyz);
    float receiverResolution = float(push.virtualPagesPerAxis * push.pageSize);
    rayBase.halfReceiverResolution = 0.5 * receiverResolution;
    rayBase.receiverMarginMax = 1.0 - (1.0 / receiverResolution);
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    rayBase.allowClipmaps = statistics[5] == 0u;
    bool isPow2PhysicalX = (push.physicalPagesX & (push.physicalPagesX - 1u)) == 0u && push.physicalPagesX != 0u;
    bool isPow2VirtualPages = (push.virtualPagesPerAxis & (push.virtualPagesPerAxis - 1u)) == 0u && push.virtualPagesPerAxis != 0u;
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    rayBase.powerOfTwoLayout = isPow2VirtualPages;
#else
    bool isPow2PageSize = (push.pageSize & (push.pageSize - 1u)) == 0u && push.pageSize != 0u;
    rayBase.powerOfTwoLayout = isPow2PageSize && isPow2VirtualPages;
    rayBase.pageSizeShift = isPow2PageSize ? uint(firstbitlow(push.pageSize)) : 0u;
    rayBase.pageSizeMask = isPow2PageSize ? (push.pageSize - 1u) : 0u;
#endif
    rayBase.powerOfTwoPhysical = isPow2PhysicalX;
    rayBase.physicalPagesXShift = isPow2PhysicalX ? uint(firstbitlow(push.physicalPagesX)) : 0u;
    rayBase.physicalPagesXMask = isPow2PhysicalX ? (push.physicalPagesX - 1u) : 0u;
    rayBase.virtualPagesPerAxisShift = isPow2VirtualPages ? uint(firstbitlow(push.virtualPagesPerAxis)) : 0u;
    return rayBase;
}

VSMDirectionalRayState vsmCreateDirectionalRayState(float3 position, float3 endpoint, float3 dither,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages, ShadowPush push, uint receiverLevel, bool useGroupCache = false)
{
    ShadowProjection first = vsmGetCachedProjection(projections, receiverLevel, useGroupCache);
    VSMDirectionalRayBase base = vsmCreateDirectionalRayBase(position, first, pages, push, receiverLevel);
    float3 direction = endpoint - position;
    VSMDirectionalRay ray;
    ray.lookupOrigin = vsmProjectPosition(position + dither, first);
    ray.step = float3(dot(first.rowX.xyz, direction), dot(first.rowY.xyz, direction), dot(first.rowZ.xyz, direction));
    VSMDirectionalRayState state;
    state.base = base;
    state.ray = ray;
    state.depthScale = length(first.rowZ.xyz);
    return state;
}

VSMDirectionalRayState vsmCreateDirectionalRayState(float3 position, float3 endpoint, float3 dither,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push, uint receiverLevel, bool useGroupCache = false)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    return vsmCreateDirectionalRayState(position, endpoint, dither, projections, pages, push, receiverLevel, useGroupCache);
}

float3 vsmDirectionalRayPosition(VSMDirectionalRay ray, VSMDirectionalRayBase rayBase, float time, uint level,
    float2 levelTranslation, float targetExtent, bool lookup)
{
    float3 position = (lookup ? ray.lookupOrigin : rayBase.comparisonOrigin) + time * ray.step;
    float scale = (level == rayBase.coarseLevel) ?
        (rayBase.sourceExtent / targetExtent) :
        asfloat(uint(127 - (int(level) - int(rayBase.receiverLevel))) << 23u);
    float2 translation = levelTranslation - rayBase.sourceTranslation * scale;
    position.xy = position.xy * scale + translation;
    return position;
}

float3 vsmDirectionalRayPosition(VSMDirectionalRay ray, VSMDirectionalRayBase rayBase, float time, uint level,
    ShadowProjection projection, bool lookup)
{
    return vsmDirectionalRayPosition(ray, rayBase, time, level, float2(projection.rowX.w, projection.rowY.w), projection.metric.x, lookup);
}

float3 vsmDirectionalRayPosition(VSMDirectionalRayState ray, float time, uint level,
    ShadowProjection projection, bool lookup)
{
    return vsmDirectionalRayPosition(ray.ray, ray.base, time, level, projection, lookup);
}

bool vsmDirectionalRayMapping(float2 sourceTranslation, uint sourcePage, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    ShadowPush push, uint level, out uint2 physical, out uint targetLevel, bool useGroupCache = false)
{
    uint4 mapping = pages[push.totalPages + sourcePage];
    vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
    physical = vsmMappingPhysicalCoord(mapping.y);
    targetLevel = vsmMappingLevel(mapping.y);
    if (mapping.w == 0u) {
        vsmWorkAdd(push, kVSMWorkProjectionMappingSlowPaths, 1u);
        return false;
    }
    if (mapping.w == 1u || targetLevel == push.clipmapCount) { return true; }
    // Directional propagation publishes only own/relative tags. Unsupported
    // integer layouts publish tag zero and retain the explicit level search.
    vsmMapDirectionalAncestor(mapping, sourceTranslation, projected, texel, projections, push, level, useGroupCache);
    return true;
}

struct VSMDirectionalRayCache
{
    uint lastPage;
    uint2 lastPhysicalCoord;
    bool lastPageHit;
};

ShadowLookup vsmSampleDirectionalAtlasPage(
    uint2 physical, uint level, float3 projected, uint2 virtualTexel,
    VSMDirectionalRayBase rayBase, VSMAtlas atlas, ShadowPush push)
{
    ShadowLookup result;
    result.projectionsHandle = push.projections;
    result.level = level;
    result.projected = projected;
    result.virtualTexel = virtualTexel;
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    uint2 atlasTexel = vsmAtlasTexel(physical, vsmPageLocalTexel(virtualTexel, push.pageSize), push.pageSize);
#else
    uint2 atlasTexel;
    if (rayBase.powerOfTwoLayout) {
        atlasTexel = (physical << rayBase.pageSizeShift) | (virtualTexel & rayBase.pageSizeMask);
    } else {
        atlasTexel = vsmAtlasTexelFromVirtual(physical, virtualTexel, push.pageSize);
    }
#endif
    result.storedDepth = vsmAtlasLoad(atlas, atlasTexel, push);
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

ShadowLookup vsmSamplePhysicalPage(
    uint physicalPage, uint level, float3 projected, uint2 virtualTexel,
    VSMDirectionalRayBase rayBase, VSMAtlas atlas, ShadowPush push)
{
    uint2 physical = rayBase.powerOfTwoPhysical ?
        uint2(physicalPage & rayBase.physicalPagesXMask, physicalPage >> rayBase.physicalPagesXShift) :
        vsmPhysicalPageCoord(physicalPage, push.physicalPagesX);
    return vsmSampleDirectionalAtlasPage(physical, level, projected, virtualTexel, rayBase, atlas, push);
}

[noinline]
ShadowLookup vsmDirectionalRayLookupSlow(
    float3 p, uint2 texel, uint page, uint4 mapping, bool inBounds,
    float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache)
{
    ShadowLookup result;
    result.projectionsHandle = push.projections;
    result.valid = false;
    result.storedDepth = 0.0;
    result.level = push.clipmapCount;
    result.projected = p;
    result.virtualTexel = texel;

    if (p.z < 0.0 || p.z > 1.0) {
        cache.lastPage = 0xffffffffu;
        cache.lastPageHit = false;
        return result;
    }

    bool mappedToCoarse = false;
    if (inBounds) {
        if (mapping.w == 0u) {
            vsmWorkAdd(push, kVSMWorkProjectionMappingSlowPaths, 1u);
            vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
            uint4 entry = pages[page];
            if (entry.x != 0xffffffffu && (entry.y & 2u) != 0u && (entry.w & 1u) != 0u) {
                return vsmSamplePhysicalPage(entry.x, rayBase.receiverLevel, p, texel, rayBase, atlas, push);
            }
        } else if (vsmMappingLevel(mapping.y) == push.clipmapCount) {
            mappedToCoarse = true;
        } else {
            vsmMapDirectionalAncestor(mapping, rayBase.sourceTranslation, p, texel, projections, push, rayBase.receiverLevel, useGroupCache);
            return vsmSampleDirectionalAtlasPage(vsmMappingPhysicalCoord(mapping.y), vsmMappingLevel(mapping.y),
                p, texel, rayBase, atlas, push);
        }
    } else {
        cache.lastPage = 0xffffffffu;
        cache.lastPageHit = false;
    }

    if (!mappedToCoarse) {
        float2 basePXY = p.xy;
        bool isPow2 = rayBase.powerOfTwoLayout;
#if VSM_PROJECTION_PAGE_SHIFT >= 0
        uint pageSizeShift = uint(VSM_PROJECTION_PAGE_SHIFT);
#else
        uint pageSizeShift = rayBase.pageSizeShift;
#endif
        uint vpaShift = rayBase.virtualPagesPerAxisShift;
        for (uint level = rayBase.receiverLevel + 1u; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float2 levelTranslation = float2(projection.rowX.w, projection.rowY.w);
            uint levelPageOffset = projection.page.x;
            float scale = asfloat(uint(127 - (int(level) - int(rayBase.receiverLevel))) << 23u);
            float2 translation = levelTranslation - rayBase.sourceTranslation * scale;
            float3 fallbackP = float3(basePXY * scale + translation, p.z);
            float margin = 1.0 / float(push.virtualPagesPerAxis * push.pageSize);
            if (!(all(fallbackP.xy >= -1.0 + margin) && all(fallbackP.xy < 1.0 - margin) &&
                  fallbackP.z >= 0.0 && fallbackP.z <= 1.0)) { continue; }
            uint2 fallbackTexel = uint2(floor((fallbackP.xy * 0.5 + 0.5) * float(push.virtualPagesPerAxis * push.pageSize)));
            uint fallbackPage;
            if (isPow2) {
                uint2 pageXY = fallbackTexel >> pageSizeShift;
                fallbackPage = levelPageOffset + ((pageXY.y << vpaShift) | pageXY.x);
            } else {
                uint2 pageCoord = fallbackTexel / push.pageSize;
                fallbackPage = levelPageOffset + pageCoord.y * push.virtualPagesPerAxis + pageCoord.x;
            }
            uint2 mappedPhysical;
            uint mappedLevel;
            if (vsmDirectionalRayMapping(levelTranslation, fallbackPage, fallbackP, fallbackTexel,
                    projections, pages, push, level, mappedPhysical, mappedLevel, useGroupCache)) {
                if (mappedLevel == push.clipmapCount) { break; }
                return vsmSampleDirectionalAtlasPage(mappedPhysical, mappedLevel, fallbackP, fallbackTexel, rayBase, atlas, push);
            }
            vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
            uint4 entry = pages[fallbackPage];
            if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
            return vsmSamplePhysicalPage(entry.x, level, fallbackP, fallbackTexel, rayBase, atlas, push);
        }
    }

    ShadowProjection coarse = vsmGetCachedProjection(projections, push.clipmapCount, useGroupCache);
    result.projected = vsmDirectionalRayPosition(ray, rayBase, time, push.clipmapCount, coarse, true);
    if (!vsmContains(result.projected, coarse, push.pageSize, false)) { return result; }
    vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
    uint4 coarseEntry = pages[coarse.page.x];
    bool coarseValid = (coarseEntry.y & 2u) != 0u && (coarseEntry.w & 1u) != 0u;
    if (!coarseValid) {
        ++counts.z;
        return result;
    }
    uint2 coarsePhysicalCoord = vsmPhysicalPageCoord(coarseEntry.x, push.physicalPagesX);
    result.virtualTexel = vsmVirtualTexel(result.projected, coarse, push.pageSize);
    ++counts.y;
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    uint2 finalAtlasTexel = vsmAtlasTexel(coarsePhysicalCoord, vsmPageLocalTexel(result.virtualTexel, push.pageSize), push.pageSize);
#else
    uint2 finalAtlasTexel;
    if (rayBase.powerOfTwoLayout) {
        finalAtlasTexel = (coarsePhysicalCoord << rayBase.pageSizeShift) | (result.virtualTexel & rayBase.pageSizeMask);
    } else {
        finalAtlasTexel = vsmAtlasTexelFromVirtual(coarsePhysicalCoord, result.virtualTexel, push.pageSize);
    }
#endif
    result.storedDepth = vsmAtlasLoad(atlas, finalAtlasTexel, push);
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

ShadowLookup vsmDirectionalRayLookup(float time, float3 stepOffset, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
{
    ++counts.x;
    float3 p = ray.lookupOrigin + stepOffset;
    if (p.z < 0.0 || p.z > 1.0) {
        cache.lastPage = 0xffffffffu;
        cache.lastPageHit = false;
        ShadowLookup invalidResult;
        invalidResult.projectionsHandle = push.projections;
        invalidResult.level = push.clipmapCount;
        invalidResult.projected = p;
        invalidResult.virtualTexel = uint2(0u);
        invalidResult.storedDepth = 0.0;
        invalidResult.valid = false;
        return invalidResult;
    }
    if (rayBase.allowClipmaps) {
        if (all(abs(p.xy) < rayBase.receiverMarginMax)) {
            uint2 texel = uint2(fma(p.xy, float2(rayBase.halfReceiverResolution), float2(rayBase.halfReceiverResolution)));
#if VSM_PROJECTION_PAGE_SHIFT >= 0
            // The page size is known even when the physical atlas width is not a power of two.
            uint2 pageCoord = texel >> VSM_PROJECTION_PAGE_SHIFT;
            uint2 localTexel = vsmPageLocalTexel(texel, push.pageSize);
            uint pageOffset = (pageCoord.y << rayBase.virtualPagesPerAxisShift) | pageCoord.x;
#else
            bool isPow2 = rayBase.powerOfTwoLayout;
            uint2 pageCoord;
            uint2 localTexel;
            uint pageOffset;
            uint pageSizeShift = rayBase.pageSizeShift;
            uint pageSizeMask = rayBase.pageSizeMask;
            uint vpaShift = rayBase.virtualPagesPerAxisShift;
            if (isPow2) {
                pageCoord = texel >> pageSizeShift;
                localTexel = texel & pageSizeMask;
                pageOffset = (pageCoord.y << vpaShift) | pageCoord.x;
            } else {
                pageCoord = texel / push.pageSize;
                localTexel = texel % push.pageSize;
                pageOffset = pageCoord.y * push.virtualPagesPerAxis + pageCoord.x;
            }
#endif
            uint page = rayBase.receiverPageOffset + pageOffset;
            vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
            uint2 physical;
            if (page == cache.lastPage && cache.lastPageHit) {
                physical = cache.lastPhysicalCoord;
            } else {
                uint4 mapping = pages[rayBase.receiverMappingOffset + pageOffset];
                if (mapping.w == 1u) {
                    physical = vsmMappingPhysicalCoord(mapping.y);
                    cache.lastPhysicalCoord = physical;
                    cache.lastPage = page;
                    cache.lastPageHit = true;
                } else {
                    cache.lastPage = page;
                    cache.lastPageHit = false;
                    return vsmDirectionalRayLookupSlow(p, texel, page, mapping, true, time, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
                }
            }
#if VSM_PROJECTION_PAGE_SHIFT >= 0
            uint2 atlasTexel = vsmAtlasTexel(physical, localTexel, push.pageSize);
#else
            uint2 atlasTexel;
            if (isPow2) {
                atlasTexel = (physical << pageSizeShift) | localTexel;
            } else {
                atlasTexel = physical * push.pageSize + localTexel;
            }
#endif
            float storedDepth = vsmAtlasLoad(atlas, atlasTexel, push);
            vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
            ShadowLookup result;
            result.projectionsHandle = push.projections;
            result.level = rayBase.receiverLevel;
            result.projected = p;
            result.virtualTexel = texel;
            result.storedDepth = storedDepth;
            result.valid = true;
            return result;
        }
        return vsmDirectionalRayLookupSlow(p, uint2(0u), 0u, uint4(0u), false, time, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
    }
    return vsmDirectionalRayLookupSlow(p, uint2(0u), 0u, uint4(0u), false, time, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, time * ray.step, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
{
    VSMDirectionalRayCache localCache;
    localCache.lastPage = 0xffffffffu;
    localCache.lastPhysicalCoord = uint2(0u);
    localCache.lastPageHit = false;
    return vsmDirectionalRayLookup(time, ray, rayBase, projections, pages, atlas, push, counts, localCache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    VSMAtlas atlas = resolveUniform(ResourceHandle<VSMAtlas>(push.atlas));
    return vsmDirectionalRayLookup(time, ray, rayBase, projections, pages, atlas, push, counts, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRayState ray,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, ray.ray, ray.base, projections, pages, atlas, push, counts, cache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRayState ray,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    VSMAtlas atlas, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, ray.ray, ray.base, projections, pages, atlas, push, counts, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRayState ray,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, ray.ray, ray.base, projections, push, counts, useGroupCache);
}

float vsmDirectionalRayDepth(float depth, VSMDirectionalRayBase rayBase)
{
    return (depth - rayBase.comparisonOrigin.z) * rayBase.invDepthScale;
}

float vsmDirectionalRayDepth(float depth, VSMDirectionalRayState ray)
{
    return vsmDirectionalRayDepth(depth, ray.base);
}

void vsmDirectionalSampleCompareAndReference(
    ShadowLookup sample, float time, float3 stepOffset, VSMDirectionalRay dirRay,
    VSMDirectionalRayBase rayBase, StructuredBuffer<ShadowProjection> projections,
    ShadowPush push, float receiverInvResolution, float coarseInvResolution,
    float2 receiverScaledSlope, float2 normalSlope, float biasZ, bool useGroupCache,
    out float compare, out float reference)
{
    if (sample.level == rayBase.receiverLevel) {
        float2 sampleNDC = fma(float2(sample.virtualTexel), float2(receiverInvResolution), float2(0.5 * receiverInvResolution - 1.0));
        float2 offset = sampleNDC - (rayBase.comparisonOrigin.xy + stepOffset.xy);
        float correction = -dot(receiverScaledSlope, offset);
        float relDepth = stepOffset.z + correction + biasZ;
        compare = rayBase.comparisonOrigin.z + relDepth;
        reference = relDepth * rayBase.invDepthScale;
    } else {
        ShadowProjection projection = projections[sample.level];
        float extent = projection.metric.x;
        float3 projected = vsmDirectionalRayPosition(dirRay, rayBase, time, sample.level, projection, false);
        float invResolution = (sample.level == push.clipmapCount) ? coarseInvResolution : receiverInvResolution;
        compare = vsmDirectionalComparisonDepth(projected, sample.virtualTexel,
            extent, invResolution, normalSlope, biasZ);
        reference = vsmDirectionalRayDepth(compare, rayBase);
    }
}
#endif

