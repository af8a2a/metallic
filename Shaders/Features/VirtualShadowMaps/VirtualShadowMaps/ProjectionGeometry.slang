#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif

#if VSM_SHARED_PHYSICAL_POOL
typealias VSMAtlas = Texture2DArray<float>;
#else
typealias VSMAtlas = Texture2D<float>;
#endif

float vsmAtlasLoad(VSMAtlas atlas, uint2 logicalTexel, ShadowPush push)
{
#if VSM_SHARED_PHYSICAL_POOL
    uint layerHeight = push.physicalPagesY * push.pageSize;
    return atlas.Load(int4(logicalTexel.x, logicalTexel.y % layerHeight, logicalTexel.y / layerHeight, 0));
#else
    return atlas.Load(int3(logicalTexel, 0));
#endif
}

uint vsmAtlasLayer(uint physicalPage, ShadowPush push)
{
#if VSM_SHARED_PHYSICAL_POOL
    return physicalPage / (push.physicalPagesX * push.physicalPagesY);
#else
    return 0u;
#endif
}

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_DIRECTIONAL_BUILD
#define VSM_DIRECTIONAL_BUILD 0
#endif
#ifndef VSM_DIRECTIONAL_MARK
#define VSM_DIRECTIONAL_MARK 0
#endif
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif
#ifndef VSM_MARK_PAGE_SHIFT
#define VSM_MARK_PAGE_SHIFT -1
#endif
#ifndef VSM_PROJECTION_PAGE_SHIFT
#define VSM_PROJECTION_PAGE_SHIFT -1
#endif

float4 vsmProjectClip(float3 position, ShadowProjection projection)
{
    float4 p = float4(position, 1.0);
    return float4(dot(projection.rowX, p), dot(projection.rowY, p), dot(projection.rowZ, p), dot(projection.rowW, p));
}

// Match the ordered scalar arithmetic of projectedBounds used by cache
// compatibility checks. A rounded page boundary changes which cache is reused.
float vsmChangedBoundRow(float4 row, float3 p)
{
    return spirv_asm {
        %x : $$float = OpFMul $(row.x) $(p.x);
        %y : $$float = OpFMul $(row.y) $(p.y);
        %z : $$float = OpFMul $(row.z) $(p.z);
        %xy : $$float = OpFAdd %x %y;
        %xyz : $$float = OpFAdd %xy %z;
        %v : $$float = OpFAdd %xyz $(row.w);
        OpDecorate %x NoContraction;
        OpDecorate %y NoContraction;
        OpDecorate %z NoContraction;
        OpDecorate %xy NoContraction;
        OpDecorate %xyz NoContraction;
        OpDecorate %v NoContraction;
        result : $$float = OpCopyObject %v
    };
}

bool vsmChangedBoundRect(float3 minimum, float3 maximum, ShadowProjection projection, out uint4 rectangle)
{
    float2 lo = float2(3.402823466e38), hi = -lo;
    float loW = lo.x, hiW = -loW;
    const bool perspective = projection.metric.w != 0.0;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 p = float3((corner & 1u) != 0u ? maximum.x : minimum.x,
            (corner & 2u) != 0u ? maximum.y : minimum.y, (corner & 4u) != 0u ? maximum.z : minimum.z);
        precise float2 xy = float2(vsmChangedBoundRow(projection.rowX, p), vsmChangedBoundRow(projection.rowY, p));
        if (perspective) {
            float w = vsmChangedBoundRow(projection.rowW, p);
            loW = min(loW, w); hiW = max(hiW, w);
            xy /= max(w, projection.originNear.w);
        }
        lo = min(lo, xy); hi = max(hi, xy);
    }
    rectangle = uint4(0u);
    // Even across the near plane, x > max(w, near) at all eight corners
    // proves x > w throughout the box (likewise for the other three sides).
    if (any(hi < -1.0) || any(lo > 1.0)) { return false; }
    if (perspective && loW <= projection.originNear.w) {
        if (hiW < projection.originNear.w) { return false; }
        lo = float2(-1.0); hi = float2(1.0);
    }
    float4 edges = clamp(float4(lo, hi), -1.0, 1.0);
    float4 halfUnit = float4(.5), axis = float4(float(projection.page.y));
    float4 guard = float4(-.0001, -.0001, .0001, .0001);
    float4 page = spirv_asm {
        %half : $$float4 = OpFMul $edges $halfUnit;
        %uv : $$float4 = OpFAdd %half $halfUnit;
        %scaled : $$float4 = OpFMul %uv $axis;
        %p : $$float4 = OpFAdd %scaled $guard;
        OpDecorate %half NoContraction;
        OpDecorate %uv NoContraction;
        OpDecorate %scaled NoContraction;
        OpDecorate %p NoContraction;
        result : $$float4 = OpCopyObject %p
    };
    rectangle = uint4(clamp(floor(page), 0.0, float(projection.page.y - 1u)));
    return true;
}

groupshared uint g_vsmChangedRectCount;

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmProjectChangedBoundsMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    uint table = group.y == 0u ? 0u : vsmCacheMetadata(push, group.y == 1u ? 36u : kVSMCacheMergeRemap);
    RWStructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.reuse));
    uint headerIndex = table + group.x;
    uint4 header = changes[headerIndex];
    if (header.z == 0u) { return; } // Cold, force-redraw or fully scrolled level.
    uint4 source = changes[table + max(1u, push.clipmapCount)];
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection projection = projections[group.x];
    if (lane == 0u) { g_vsmChangedRectCount = 0u; }
    GroupMemoryBarrierWithGroupSync();
    for (uint bound = lane; bound < source.y; bound += 64u) {
        uint4 minimum = changes[source.x + bound * 2u];
        uint4 rect;
        if (minimum.w == 0u || !vsmChangedBoundRect(asfloat(minimum.xyz),
            asfloat(changes[source.x + bound * 2u + 1u].xyz), projection, rect)) { continue; }
#if VSM_STATIC_PAGE_HZB
        if (group.y == 0u) { rect = uint4(rect.xy, rect.z | (rect.w << 16u), bound); }
#endif
        uint slot;
        InterlockedAdd(g_vsmChangedRectCount, 1u, slot);
        changes[header.z + slot] = rect;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        changes[headerIndex].w = g_vsmChangedRectCount;
        if (group.y < 2u) {
            vsmCacheAdd(push, group.y == 0u ? kVSMCacheDirtyRectangles : kVSMCacheStaticDirtyRectangles,
                g_vsmChangedRectCount);
        }
    }
}

float3 vsmProjectPosition(float3 position, ShadowProjection projection)
{
    float4 clip = vsmProjectClip(position, projection);
#if VSM_DIRECTIONAL_BUILD || VSM_DIRECTIONAL_MARK || VSM_DIRECTIONAL_PROJECTION
    // Entry-specific directional permutations; makeProjections writes metric.w=0.
    return clip.xyz;
#else
    return projection.metric.w > 0.5 ? clip.xyz / clip.w : clip.xyz;
#endif
}

// The selected face owns both demand and query. Receivers outside a punctual
// light's range/cone contribute no direct light; near-plane truncation is counted
// separately by the query pass and deliberately returns neutral visibility.
bool vsmReceiverDomain(float3 position, ShadowProjection projection, ShadowPush push, out bool nearPlane)
{
    nearPlane = false;
#if VSM_DIRECTIONAL_MARK
    return true;
#else
    if (projection.metric.w < 0.5) { return true; }
    float3 delta = position - projection.originNear.xyz;
    float distance = length(delta);
    if (push.lightRange > 0.0 && distance >= push.lightRange) { return false; }
    float axial = dot(projection.rowW, float4(position, 1.0));
    if (projection.metric.w > 1.5) {
        if (virtualShadowMapPointFace(delta) != push.projectionFace) { return false; }
    } else {
        float outerCosine = rsqrt(1.0 + projection.metric.x * projection.metric.x);
        if (axial < distance * outerCosine) { return false; }
    }
    nearPlane = axial <= projection.originNear.w;
    return !nearPlane;
#endif
}

bool vsmContains(float3 position, ShadowProjection projection, uint pageSize, bool inset)
{
#if !VSM_DIRECTIONAL_MARK
    if (projection.metric.w > 0.5) {
        // A canonical point face includes its boundary. Insetting it would
        // reject receivers exactly on a cube edge without selecting a neighbor.
        return all(position.xy >= -1.0) && all(position.xy <= 1.0) && position.z >= 0.0 && position.z <= 1.0;
    }
#endif
    float margin = inset ? 1.0 / float(projection.page.y * pageSize) : 0.0;
    return all(position.xy >= -1.0 + margin) && all(position.xy < 1.0 - margin) &&
        position.z >= 0.0 && position.z <= 1.0;
}

uint2 vsmVirtualTexel(float3 position, ShadowProjection projection, uint pageSize)
{
    uint resolution = projection.page.y * pageSize;
    uint2 texel = uint2(floor((position.xy * 0.5 + 0.5) * float(resolution)));
#if VSM_DIRECTIONAL_MARK
    return texel;
#else
    return projection.metric.w > 0.5 ? min(texel, uint2(resolution - 1u)) : texel;
#endif
}

uint vsmVirtualPage(uint2 texel, ShadowProjection projection, uint pageSize)
{
#if VSM_MARK_PAGE_SHIFT >= 0
    uint2 page = texel >> VSM_MARK_PAGE_SHIFT;
#elif VSM_PROJECTION_PAGE_SHIFT >= 0
    uint2 page = texel >> VSM_PROJECTION_PAGE_SHIFT;
#else
    uint2 page = texel / pageSize;
#endif
    return projection.page.x + page.y * projection.page.y + page.x;
}

uint2 vsmPageLocalTexel(uint2 texel, uint pageSize)
{
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    return texel & ((1u << VSM_PROJECTION_PAGE_SHIFT) - 1u);
#else
    return texel % pageSize;
#endif
}

uint2 vsmAtlasTexel(uint2 physical, uint2 localTexel, uint pageSize)
{
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    return (physical << VSM_PROJECTION_PAGE_SHIFT) | localTexel;
#else
    return physical * pageSize + localTexel;
#endif
}

uint2 vsmAtlasTexelFromVirtual(uint2 physical, uint2 texel, uint pageSize)
{
#if VSM_PROJECTION_PAGE_SHIFT >= 0
    return (physical << VSM_PROJECTION_PAGE_SHIFT) | (texel & ((1u << VSM_PROJECTION_PAGE_SHIFT) - 1u));
#else
    return physical * pageSize + (texel % pageSize);
#endif
}

float4 vsmPageBounds(uint virtualPage, ShadowProjection projection)
{
    uint local = virtualPage - projection.page.x;
    uint2 page = uint2(local % projection.page.y, local / projection.page.y);
    return float4(float2(page) * (2.0 / float(projection.page.y)) - 1.0,
        float2(page + 1u) * (2.0 / float(projection.page.y)) - 1.0);
}

uint2 vsmPhysicalPageCoord(uint physicalPage, uint physicalPagesX)
{
    if ((physicalPagesX & (physicalPagesX - 1u)) == 0u) {
        uint shift = uint(firstbitlow(physicalPagesX));
        return uint2(physicalPage & (physicalPagesX - 1u), physicalPage >> shift);
    }
    return uint2(physicalPage % physicalPagesX, physicalPage / physicalPagesX);
}

float2 vsmAtlasPosition(float2 pagePosition, uint physicalPage, ShadowPush push)
{
    uint2 page = vsmPhysicalPageCoord(physicalPage, push.physicalPagesX);
#if VSM_SHARED_PHYSICAL_POOL
    page.y %= push.physicalPagesY;
#endif
    return (float2(page) + pagePosition) * (2.0 / float2(push.physicalPagesX, push.physicalPagesY)) - 1.0;
}

// Six homogeneous half-planes turn a triangle into at most nine vertices.
// Attributes interpolate along the original edge; preserving clip.w at raster
// output then supplies perspective-correct interpolation across the clipped fan.
uint vsmClipPlane(inout ShadowClipVertex polygon[9], uint count, uint axis, float edge, float side)
{
    if (count == 0u) { return 0u; }
    ShadowClipVertex clipped[9];
    uint clippedCount = 0u;
    ShadowClipVertex previous = polygon[count - 1u];
    float previousDistance = side * (previous.position[axis] - edge * previous.position.w);
    for (uint index = 0u; index < count; ++index) {
        ShadowClipVertex current = polygon[index];
        float currentDistance = side * (current.position[axis] - edge * current.position.w);
        if ((previousDistance < 0.0 && currentDistance > 0.0) ||
            (previousDistance > 0.0 && currentDistance < 0.0)) {
            float t = previousDistance / (previousDistance - currentDistance);
            ShadowClipVertex intersection;
            intersection.position = lerp(previous.position, current.position, t);
            if (axis == 2u && edge == 1.0) {
                // At reversed-Z near, z=w. For a triangle crossing the light,
                // interpolated w subtracts large values and loses precision;
                // z stays near the small near distance. Preserve z so later
                // page clipping does not amplify that error into self-shadow.
                intersection.position.w = intersection.position.z;
            } else {
                intersection.position[axis] = edge * intersection.position.w;
            }
            intersection.uv = lerp(previous.uv, current.uv, t);
            clipped[clippedCount++] = intersection;
        }
        if (currentDistance >= 0.0) { clipped[clippedCount++] = current; }
        previous = current;
        previousDistance = currentDistance;
    }
    for (uint index = 0u; index < clippedCount; ++index) { polygon[index] = clipped[index]; }
    return clippedCount;
}

uint vsmClipProjection(inout ShadowClipVertex polygon[9], float4 bounds, bool perspective)
{
    uint count = 3u;
    if (perspective) {
        count = vsmClipPlane(polygon, count, 2u, 1.0, -1.0); // Reversed-Z near: w-z >= 0.
        count = vsmClipPlane(polygon, count, 2u, 0.0, 1.0);  // Far: z >= 0.
    }
    count = vsmClipPlane(polygon, count, 0u, bounds.x, 1.0);
    count = vsmClipPlane(polygon, count, 0u, bounds.z, -1.0);
    count = vsmClipPlane(polygon, count, 1u, bounds.y, 1.0);
    return vsmClipPlane(polygon, count, 1u, bounds.w, -1.0);
}

