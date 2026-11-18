#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif
#ifndef VSM_PROJECTION_PAGE_SHIFT
#define VSM_PROJECTION_PAGE_SHIFT -1
#endif

struct ShadowLookup
{
    // Point-light lookup may select another face. Carry its table identity,
    // not a complete projection through fine/ancestor/coarse control flow.
    uint projectionsHandle;
    float3 projected;
    uint2 virtualTexel;
    float storedDepth;
    uint level;
    bool valid;
};

ShadowProjection vsmLookupProjection(ShadowLookup sample)
{
#if VSM_DIRECTIONAL_PROJECTION
    StructuredBuffer<ShadowProjection> projections = resolveUniform(
#else
    StructuredBuffer<ShadowProjection> projections = resolveNonUniform(
#endif
        ResourceHandle<StructuredBuffer<ShadowProjection>>(sample.projectionsHandle));
    return projections[sample.level];
}

// The directional mapping producer certifies a dyadic ancestor and shared Z.
// Preserve its integer page ownership while continuing the continuous ray XY.
// Only the derived mapping tail uses this address. The host caps physical
// Each shared atlas layer holds at most 64 by 64 pages. Shared coordinates
// retain logical Y across layers; standalone keeps its existing representation.
uint vsmPackDirectionalMappingAddress(uint level, uint physicalPage, uint physicalPagesX)
{
    uint2 physical = vsmPhysicalPageCoord(physicalPage, physicalPagesX);
#if VSM_SHARED_PHYSICAL_POOL
    return level | (physical.x << 4u) | (physical.y << 10u);
#else
    return level | (physical.x << 16u) | (physical.y << 24u);
#endif
}

uint vsmMappingLevel(uint address)
{
#if VSM_SHARED_PHYSICAL_POOL
    return address & 0xfu;
#else
    return address & 0xffffu;
#endif
}

uint2 vsmMappingPhysicalCoord(uint address)
{
#if VSM_SHARED_PHYSICAL_POOL
    return uint2((address >> 4u) & 0x3fu, (address >> 10u) & 0xfffu);
#else
    return uint2((address >> 16u) & 0xffu, address >> 24u);
#endif
}

void vsmMapDirectionalAncestor(uint4 mapping, float2 sourceTranslation, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, ShadowPush push, uint level,
    out float2 targetTranslation, out float extent, bool useGroupCache = false)
{
    uint targetLevel = vsmMappingLevel(mapping.y);
    float scale = asfloat((127u - (targetLevel - level)) << 23u);
    ShadowProjection targetProj = projections[targetLevel];
    targetTranslation = float2(targetProj.rowX.w, targetProj.rowY.w);
    extent = targetProj.metric.x;
    uint targetPageY = targetProj.page.y;
    float2 translation = targetTranslation - sourceTranslation * scale;
    projected.xy = projected.xy * scale + translation;
    uint2 page = uint2(mapping.z & 0xffffu, mapping.z >> 16u);
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    uint2 firstTexel = page << VSM_PROJECTION_PAGE_SHIFT;
#else
    uint2 firstTexel = page * push.pageSize;
#endif
    float2 address = (projected.xy * 0.5 + 0.5) * float(targetPageY * push.pageSize);
    texel = uint2(clamp(address, float2(firstTexel), float2(firstTexel + push.pageSize - 1u)));
    vsmWorkAdd(push, kVSMWorkProjectionMappedAncestors, 1u);
}

void vsmMapDirectionalAncestor(uint4 mapping, float2 sourceTranslation, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, ShadowPush push, uint level, bool useGroupCache = false)
{
    float2 targetTranslation;
    float extent;
    vsmMapDirectionalAncestor(mapping, sourceTranslation, projected, texel, projections, push, level, targetTranslation, extent, useGroupCache);
}

bool vsmLookupPropagated(float3 position, float2 sourceTranslation, uint sourcePage, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    ShadowPush push, uint level, out uint physicalPage, out uint targetLevel,
    out bool isOwnMapping, bool useGroupCache)
{
    isOwnMapping = false;
    // One-pass jobs carry their own clipmapCount; fixed VPA^2 level strides
    // also apply to local mips, whose unused prefix slots remain unmapped.
    uint tableSize = push.clipmapCount * push.virtualPagesPerAxis * push.virtualPagesPerAxis + 1u;
    uint4 mapping = pages[tableSize + sourcePage];
    vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
    physicalPage = mapping.x;
    targetLevel = vsmMappingLevel(mapping.y);
    if (mapping.w == 0u) {
        vsmWorkAdd(push, kVSMWorkProjectionMappingSlowPaths, 1u);
        return false;
    }
    if (mapping.w == 1u) {
        isOwnMapping = true;
        return true;
    }
    if (targetLevel == push.clipmapCount) { return true; }
    if (mapping.w == 2u) {
        vsmMapDirectionalAncestor(mapping, sourceTranslation, projected, texel, projections, push, level, useGroupCache);
        return true;
    }
    // Only local-light fallback needs a full target projection here.
    ShadowProjection target = vsmGetCachedProjection(projections, targetLevel, useGroupCache);
    float3 targetPosition = vsmProjectPosition(position, target);
    // Local dyadic mips retain their original projection arithmetic.
    if (!vsmContains(targetPosition, target, push.pageSize, true)) {
        vsmWorkAdd(push, kVSMWorkProjectionMappingSlowPaths, 1u);
        return false;
    }
    projected = targetPosition;
    texel = vsmVirtualTexel(targetPosition, target, push.pageSize);
    vsmWorkAdd(push, kVSMWorkProjectionMappedAncestors, 1u);
    return true;
}

bool vsmLookupPropagated(float3 position, float2 sourceTranslation, uint sourcePage, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    ShadowPush push, uint level, out uint physicalPage, out uint targetLevel, bool useGroupCache = false)
{
    bool isOwnMapping;
    return vsmLookupPropagated(position, sourceTranslation, sourcePage, projected, texel,
        projections, pages, push, level, physicalPage, targetLevel, isOwnMapping, useGroupCache);
}

ShadowLookup vsmFilterLookup(float3 position, ShadowPush push, uint firstLevel, inout uint3 filterCounts)
{
    ShadowLookup result;
    result.valid = false;
    result.storedDepth = 0.0;
    uint atlasHandle = push.atlas, pagesHandle = push.pages, projectionsHandle = push.projections;
    uint statisticsHandle = push.statistics, physicalPagesX = push.physicalPagesX;
#if !VSM_DIRECTIONAL_PROJECTION
    if (push.lightType == 2u) {
        StructuredBuffer<ShadowProjection> own = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        StructuredBuffer<ShadowFace> faces = resolveUniform(ResourceHandle<StructuredBuffer<ShadowFace>>(push.pointFaces));
        ShadowFace face = faces[virtualShadowMapPointFace(position - own[0].originNear.xyz)];
        atlasHandle = face.atlas; pagesHandle = face.pages; projectionsHandle = face.projections;
        statisticsHandle = face.statistics; physicalPagesX = face.physicalPagesX;
    }
#endif
#if VSM_DIRECTIONAL_PROJECTION
    // Per-face directional push descriptors stay uniform through fine/ancestor/coarse selection.
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(projectionsHandle));
#else
    StructuredBuffer<ShadowProjection> projections = resolveNonUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(projectionsHandle));
#endif
    result.projectionsHandle = projectionsHandle;
#if VSM_DIRECTIONAL_PROJECTION
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(pagesHandle));
    StructuredBuffer<uint> faceStatistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(statisticsHandle));
#else
    StructuredBuffer<uint4> pages = resolveNonUniform(ResourceHandle<StructuredBuffer<uint4>>(pagesHandle));
    StructuredBuffer<uint> faceStatistics = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(statisticsHandle));
#endif
    uint physicalPage = 0xffffffffu;
    result.level = push.clipmapCount;
    ++filterCounts.x;
    if (faceStatistics[5] == 0u) {
        for (uint level = firstLevel; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint page = vsmVirtualPage(texel, projection, push.pageSize);
            uint mappedPhysical, mappedLevel;
            if (vsmLookupPropagated(position, float2(projection.rowX.w, projection.rowY.w),
                    page, p, texel, projections, pages, push, level,
                    mappedPhysical, mappedLevel)) {
                if (mappedLevel == push.clipmapCount) { break; }
                physicalPage = mappedPhysical; result.level = mappedLevel;
                result.projected = p; result.virtualTexel = texel;
                break;
            }
            vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
            uint4 entry = pages[page];
            if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
            physicalPage = entry.x; result.level = level;
            result.projected = p; result.virtualTexel = texel;
            break;
        }
    }
    if (physicalPage == 0xffffffffu) {
        ShadowProjection coarse = projections[push.clipmapCount];
        result.projected = vsmProjectPosition(position, coarse);
        // The complete coarse projection encloses all represented casters.
        // A trace segment outside that volume has no represented depth layer.
        if (!vsmContains(result.projected, coarse, push.pageSize, false)) { return result; }
        vsmWorkAdd(push, kVSMWorkProjectionPageTableLoads, 1u);
        uint4 entry = pages[coarse.page.x];
        if ((entry.y & 2u) == 0u || (entry.w & 1u) == 0u) {
            ++filterCounts.z;
            return result;
        }
        physicalPage = entry.x;
        result.virtualTexel = vsmVirtualTexel(result.projected, coarse, push.pageSize);
        ++filterCounts.y;
    }
    uint2 physical = vsmPhysicalPageCoord(physicalPage, physicalPagesX);
    uint2 texel = vsmAtlasTexelFromVirtual(physical, result.virtualTexel, push.pageSize);
#if VSM_DIRECTIONAL_PROJECTION
    VSMAtlas atlas = resolveUniform(ResourceHandle<VSMAtlas>(atlasHandle));
#else
    VSMAtlas atlas = resolveNonUniform(ResourceHandle<VSMAtlas>(atlasHandle));
#endif
    result.storedDepth = vsmAtlasLoad(atlas, texel, push);
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

