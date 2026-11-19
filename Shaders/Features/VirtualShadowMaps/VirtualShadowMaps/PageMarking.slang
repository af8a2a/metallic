#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_LOCAL_MARK
#define VSM_LOCAL_MARK 0
#endif
#ifndef VSM_DIRECTIONAL_MARK
#define VSM_DIRECTIONAL_MARK 0
#endif
#ifndef VSM_VIEW_MARK
#define VSM_VIEW_MARK 0
#endif
#ifndef VSM_MARK_CACHE_LEVELS
#define VSM_MARK_CACHE_LEVELS 12
#endif

// Read-before-write filters must remain atomic with other lanes' updates.
#if (VSM_DIRECTIONAL_MARK || VSM_VIEW_MARK) && VSM_RECEIVER_MASK
static const uint kVSMMarkHashBits = 7u;
#else
static const uint kVSMMarkHashBits = 8u;
#endif
static const uint kVSMMarkHashCapacity = 1u << kVSMMarkHashBits;
groupshared Atomic<uint> gVsmMarkedPages[kVSMMarkHashCapacity];

#if (VSM_DIRECTIONAL_MARK || VSM_VIEW_MARK) && VSM_RECEIVER_MASK
[inline]
bool vsmClaimMarkSlot(uint page, out uint slot)
{
    // The page key remains valid until flush. A colliding page publishes directly.
    slot = (page * 2654435761u) >> (32u - kVSMMarkHashBits);
    uint current = gVsmMarkedPages[slot].load(MemoryOrder.Relaxed);
    if (current == page) { return true; }
    if (current != 0xffffffffu) { return false; }
    uint previous = gVsmMarkedPages[slot].compareExchange(0xffffffffu, page, MemoryOrder.Relaxed, MemoryOrder.Relaxed);
    return previous == 0xffffffffu || previous == page;
}
#endif

void vsmPublishMarkPage(RWStructuredBuffer<uint4> pages, uint page, inout uint3 workCounts)
{
    InterlockedOr(pages[page].y, 1u);
    ++workCounts.z;
}

void vsmMarkPage(RWStructuredBuffer<uint4> pages, uint page, inout uint3 workCounts)
{
#if (VSM_DIRECTIONAL_MARK || VSM_VIEW_MARK) && VSM_RECEIVER_MASK
    uint slot;
    if (vsmClaimMarkSlot(page, slot)) { return; }
#else
    // A cache hit reuses an inserting lane's eventual OR. Hash collisions only
    // cause repeated requests; allocation observes this dispatch after all ORs.
    uint slot = (page * 2654435761u) >> (32u - kVSMMarkHashBits);
    if (gVsmMarkedPages[slot].load(MemoryOrder.Relaxed) == page) { return; }
    uint previous = gVsmMarkedPages[slot].exchange(page, MemoryOrder.Relaxed);
    if (previous == page) { return; }
#endif
    vsmPublishMarkPage(pages, page, workCounts);
}

void vsmMarkPosition(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel, inout uint3 workCounts,
    bool pointFaceSelected = false)
{
    ++workCounts.x;
    // Hard and the nine-tap PCF kernel keep the short direct path. Cache setup
    // pays off for the longer SMRT traces handled by the overload below.
    if (!pointFaceSelected && push.lightType == 2u &&
        virtualShadowMapPointFace(position - projections[0].originNear.xyz) != push.projectionFace) { return; }
#if VSM_DIRECTIONAL_MARK
    // makeProjections and cache reuse preserve one rowZ across all clipmaps.
    float4 samplePosition = float4(position, 1.0);
    float sampleZ = dot(projections[0].rowZ, samplePosition);
#elif VSM_LOCAL_MARK
    // makeLocalProjections repeats all four rows across mips; makeProjections
    // repeats rowZ across directional clipmaps. Keep the original dot products
    // (no endpoint interpolation or reassociation at page boundaries).
    ShadowProjection sharedProjection = projections[firstLevel];
    float3 sharedProjected = push.lightType == 0u ?
        float3(0.0, 0.0, dot(sharedProjection.rowZ, float4(position, 1.0))) :
        vsmProjectPosition(position, sharedProjection);
#endif
    for (uint level = firstLevel; level < push.clipmapCount; ++level) {
        ++workCounts.y;
        ShadowProjection projection = projections[level];
#if VSM_DIRECTIONAL_MARK
        float3 projected = float3(dot(projection.rowX, samplePosition),
            dot(projection.rowY, samplePosition), sampleZ);
#elif VSM_LOCAL_MARK
        float3 projected = sharedProjected;
        if (push.lightType == 0u) {
            float4 p = float4(position, 1.0);
            projected.xy = float2(dot(projection.rowX, p), dot(projection.rowY, p));
        }
#else
        float3 projected = vsmProjectPosition(position, projection);
#endif
        if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
        uint page = vsmVirtualPage(vsmVirtualTexel(projected, projection, push.pageSize), projection, push.pageSize);
        vsmMarkPage(pages, page, workCounts);
    }
}

#if !VSM_DIRECTIONAL_MARK
// Directional SMRT uses vsmMarkDirectionalFootprint; only local/generic rays
// retain the per-receiver last-page cache.
void vsmMarkPosition(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel, inout uint lastPages[VSM_MARK_CACHE_LEVELS], inout uint3 workCounts)
{
    ++workCounts.x;
    if (push.lightType == 2u && virtualShadowMapPointFace(position - projections[0].originNear.xyz) != push.projectionFace) { return; }
    // Match the query's first usable level. Requesting finer pages here spends
    // the admission budget on pages the receiver will never sample.
    // The selected Mark permutation bounds clipmapCount by its cache size.
    // Constant indices keep the small per-receiver cache in registers across all SMRT rays.
#if VSM_LOCAL_MARK
    ShadowProjection sharedProjection = projections[firstLevel];
    float3 sharedProjected = push.lightType == 0u ?
        float3(0.0, 0.0, dot(sharedProjection.rowZ, float4(position, 1.0))) :
        vsmProjectPosition(position, sharedProjection);
#endif
    [unroll]
    for (uint level = 0u; level < VSM_MARK_CACHE_LEVELS; ++level) {
        if (level < firstLevel || level >= push.clipmapCount) { continue; }
        ++workCounts.y;
        ShadowProjection projection = projections[level];
#if VSM_LOCAL_MARK
        float3 projected = sharedProjected;
        if (push.lightType == 0u) {
            float4 p = float4(position, 1.0);
            projected.xy = float2(dot(projection.rowX, p), dot(projection.rowY, p));
        }
#else
        float3 projected = vsmProjectPosition(position, projection);
#endif
        if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
        uint page = vsmVirtualPage(vsmVirtualTexel(projected, projection, push.pageSize), projection, push.pageSize);
        if (lastPages[level] == page) { continue; }
        lastPages[level] = page;
        vsmMarkPage(pages, page, workCounts);
    }
}
#endif

