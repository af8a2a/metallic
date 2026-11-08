#language slang 2026
module VirtualShadowMaps;

#if VSM_STATIC_PAGE_HZB
__include "VirtualShadowMapStaticHZB.slang";
#endif

#if VSM_PROPAGATED_MAPPING_PROBE
__include "../../../tests/rhi/shaders/VirtualShadowMapPropagatedMappingProbe.slang";
#endif

import ShaderCore;
import Material;
import MaterialProgram;
import VirtualShadowMapGeometry;
import VirtualShadowMapReceiver;
import VirtualShadowMapCommon;
import Lighting;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapProjectionGrid.h"

#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_RECEIVER_COVERAGE_DIAGNOSTICS
#define VSM_RECEIVER_COVERAGE_DIAGNOSTICS 0
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
#ifndef VSM_LOCAL_MARK
#define VSM_LOCAL_MARK 0
#endif
#ifndef VSM_DIRECTIONAL_BUILD
#define VSM_DIRECTIONAL_BUILD 0
#endif
#ifndef VSM_DIRECTIONAL_MARK
#define VSM_DIRECTIONAL_MARK 0
#endif
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_COVERAGE_DIAGNOSTICS
#error The full-cone Mark candidate does not support the legacy per-ray receiver coverage records.
#endif
#ifndef VSM_DIRECTIONAL_PROJECTION
#define VSM_DIRECTIONAL_PROJECTION 0
#endif
// Keep the sampling helpers' light-type arithmetic shared with Mark. Folding
// those expressions changes grazing-boundary samples in the Sponza replay.
#ifndef VSM_ADAPTIVE_RAY_COUNT
#define VSM_ADAPTIVE_RAY_COUNT 0
#endif
#ifndef VSM_MARK_PAGE_SHIFT
#define VSM_MARK_PAGE_SHIFT -1
#endif
#ifndef VSM_PROJECTION_PAGE_SHIFT
#define VSM_PROJECTION_PAGE_SHIFT -1
#endif
#ifndef VSM_MARK_CACHE_LEVELS
#define VSM_MARK_CACHE_LEVELS 12
#endif
#ifndef VSM_COMPACT_BUILD
#define VSM_COMPACT_BUILD 0
#endif
#ifndef VSM_PAGE_WINDOW_RASTER
#define VSM_PAGE_WINDOW_RASTER 0
#endif
#ifndef VSM_CLUSTER_PAGE_RASTER
#define VSM_CLUSTER_PAGE_RASTER 0
#endif
#ifndef VSM_INSTANCE_BUILD
#define VSM_INSTANCE_BUILD 0
#endif
#ifndef VSM_DIRTY_PAGE_HIERARCHY
#define VSM_DIRTY_PAGE_HIERARCHY 0
#endif

// Allocator/remap-only probes may supply owners without the dirty-page tail.
// Production always enables this storage; build always consumes the bitset.
#ifndef VSM_CLUSTER_PAGE_BITSET
#define VSM_CLUSTER_PAGE_BITSET 0
#endif

using Metallic;
using Metallic.Material;
using Metallic.Shadow;
using Metallic.Lighting;

public struct ShadowPush
{
    uint geometry, materials, pages, coverageInputs;
    uint workItems, statistics, projections, atlas;
    uint depth, shadow, debug, view;
    uint arguments, triangleCount, clipmapCount, virtualPagesPerAxis;
    uint physicalPagesX, physicalPagesY, physicalPageCount, pageSize;
    uint workCapacity, batchCount, workBase, rasterCoarse;
    uint sampler, enabled, totalPages, lightType;
    uint previousPages, reuse, owners, cacheMode;
    uint coarseBatchCount, cancelledCount, cacheEpoch, dirtyBoundsCount;
    float lightRange;
    uint projectionFace, filterMode, pointFaces;
    float pcfRadius, rayLength, sourceRadius, angularRadius;
    uint rayCount, samplesPerRay, receiverFootprint, projectionFlags;
    float rayLengthScale, maxRayAngle;
    uint samplePhase, sampleReserved;
    float lodScreenRayLength;
    uint receiverGeometry;
};

// sampleReserved names a 32-word diagnostic prefix followed by immutable layer metadata.
uint vsmCacheMetadata(ShadowPush push, uint word)
{
    StructuredBuffer<uint> data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.sampleReserved));
    return data[word];
}

// An incremental static update keeps the old D32 page and draws only casters
// transitioning this frame. Finalize certifies the result like a static redraw.
static const uint kVSMPageMergeStatic = 16u;

uint4 vsmMergeCluster(ShadowPush push, uint cluster)
{
    StructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.reuse));
    return changes[vsmCacheMetadata(push, kVSMCacheMergeClustersBase) + cluster];
}

bool vsmMergeClusterHasLane(uint4 cluster, uint lane)
{
    return (((lane < 32u ? cluster.y : cluster.z) >> (lane & 31u)) & 1u) != 0u;
}

void vsmCacheAdd(ShadowPush push, uint counter, uint value)
{
    if (value == 0u) { return; }
    RWStructuredBuffer<uint> data = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
    InterlockedAdd(data[counter], value);
}

uint vsmClusterPageBitsetOffset(ShadowPush push)
{
    // Matches the tail appended to the owners allocation in VirtualShadowMaps.cpp.
    return push.physicalPageCount * 2u + push.totalPages + (push.totalPages + 255u) / 256u;
}

public uint vsmReceiverMaskOffset(ShadowPush push)
{
    return vsmClusterPageBitsetOffset(push) + (push.totalPages + 31u) / 32u;
}

#if VSM_RECEIVER_MASK
uint vsmValidReceiverMaskByteOffset(ShadowPush push, uint page)
{
    // Persistent tail after residency and propagated mappings. Each
    // page owns two uint words; byte addressing avoids uint4 read/modify/write
    // between adjacent pages. CPU allocation rounds the final size to 16 bytes.
    return push.totalPages * 32u + page * 8u;
}

// Eight rows of eight subtiles, packed as two groups of four rows.
public uint2 vsmReceiverRectangleMask(uint2 first, uint2 last)
{
    if (any(first > last)) { return uint2(0u); }
    uint row = (0xffu << first.x) & (0xffu >> (7u - last.x));
    uint row4 = row * 0x01010101u;
    uint2 mask = uint2(0u);
    if (first.y <= 3u) {
        uint shiftLo = first.y * 8u;
        uint shiftHi = (3u - min(last.y, 3u)) * 8u;
        mask.x = row4 & ((0xffffffffu << shiftLo) & (0xffffffffu >> shiftHi));
    }
    if (last.y >= 4u) {
        uint firstY1 = max(first.y, 4u) - 4u;
        uint lastY1 = last.y - 4u;
        uint shiftLo = firstY1 * 8u;
        uint shiftHi = (3u - lastY1) * 8u;
        mask.y = row4 & ((0xffffffffu << shiftLo) & (0xffffffffu >> shiftHi));
    }
    return mask;
}

public uint2 vsmReceiverTexelMask(uint2 first, uint2 last, uint2 page, uint pageSize)
{
    int2 origin = int2(page * pageSize);
    uint2 localFirst = uint2(clamp(int2(first) - origin, int2(0), int2(pageSize - 1u)));
    uint2 localLast = uint2(clamp(int2(last) - origin, int2(0), int2(pageSize - 1u)));
#if VSM_MARK_PAGE_SHIFT >= 0
    return vsmReceiverRectangleMask((localFirst * 8u) >> VSM_MARK_PAGE_SHIFT,
        (localLast * 8u) >> VSM_MARK_PAGE_SHIFT);
#else
    return vsmReceiverRectangleMask(localFirst * 8u / pageSize, localLast * 8u / pageSize);
#endif
}
#endif

#if VSM_DIRTY_PAGE_HIERARCHY
public uint vsmPageHierarchyAxis(ShadowPush push)
{
    return 1u << uint(firstbithigh(push.virtualPagesPerAxis * 2u - 1u));
}

public uint vsmPageHierarchyLevelWords(ShadowPush push)
{
    uint axis = vsmPageHierarchyAxis(push);
    // Inclusive XY bounds, followed by mip1..root. Mip0 is the dirty bitset.
    return 4u + (axis * axis - 1u) / 3u;
}

public uint vsmPageHierarchyOffset(ShadowPush push)
{
    return vsmReceiverMaskOffset(push) + (VSM_RECEIVER_MASK ? push.totalPages * 2u : 0u);
}

uint vsmPageHierarchyMipOffset(uint paddedAxis, uint mip)
{
    uint previousAxis = paddedAxis >> (mip - 1u);
    return (paddedAxis * paddedAxis - previousAxis * previousAxis) / 3u;
}

public void vsmResetPageHierarchy(uint index, ShadowPush push, RWStructuredBuffer<uint> owners)
{
    uint levelWords = vsmPageHierarchyLevelWords(push);
    if (index < push.clipmapCount * levelWords) {
        owners[vsmPageHierarchyOffset(push) + index] = index % levelWords < 2u ? 0xffffffffu : 0u;
    }
}
#endif

#if VSM_COMPACT_BUILD
uint vsmCompactCandidateOffset(ShadowPush push)
{
    uint offset = push.physicalPageCount * 2u + push.totalPages + (push.totalPages + 255u) / 256u;
    offset += (push.totalPages + 31u) / 32u;
#if VSM_RECEIVER_MASK
    offset += push.totalPages * 2u;
#endif
#if VSM_DIRTY_PAGE_HIERARCHY
    offset += push.clipmapCount * vsmPageHierarchyLevelWords(push);
#endif
    return offset;
}

uint vsmCompactBatchCount(ShadowPush push)
{
#if VSM_INSTANCE_BUILD
    return vsmCacheMetadata(push, kVSMCacheMergeDrawOffset) - push.batchCount - push.coarseBatchCount;
#else
    uint clusters = push.triangleCount / 64u + (push.triangleCount % 64u != 0u ? 1u : 0u);
    return (clusters + 65534u) / 65535u;
#endif
}
#endif

void vsmWorkAdd(ShadowPush push, uint counter, uint value)
{
#if VSM_WORKLOAD_DIAGNOSTICS
    if (value == 0u) { return; }
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint word = 32u + counter * 2u;
    uint previous;
    InterlockedAdd(statistics[word], value, previous);
    if (previous > 0xffffffffu - value) { InterlockedAdd(statistics[word + 1u], 1u); }
#endif
}

#if VSM_WORKLOAD_DIAGNOSTICS
void vsmBuildDiagnosticAdd(ShadowPush push, uint counter, uint value)
{
    // Reuse the workload's low/high carry contract after its counter block.
    vsmWorkAdd(push, kVirtualShadowMapWorkCounterCount + counter, value);
}

void vsmRecordClusterLevel(ShadowPush push, uint scanTests, uint outcome)
{
    // Every completed predicate has one outcome and one scan observation.
    // Count actual bitset loads, not the rectangle's logical page area.
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterLevelTests, 1u);
#if VSM_DIRTY_PAGE_HIERARCHY
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyNodeTests, scanTests);
    vsmBuildDiagnosticAdd(push, outcome, 1u);
#else
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterBitsetLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterBitsetWordTests, scanTests);
    const uint histogram = kVSMBuildDiagnosticClusterBitsetScanHistogram;
    vsmBuildDiagnosticAdd(push, outcome, 1u);
    uint bucket = 0u;
    if (scanTests != 0u) { bucket = min(uint(firstbithigh(scanTests)) + 1u, kVirtualShadowMapBuildDiagnosticHistogramCount - 1u); }
    vsmBuildDiagnosticAdd(push, histogram + bucket, 1u);
#endif
}
#endif

public struct ShadowProjection
{
    float4 rowX, rowY, rowZ, rowW, metric;
    uint4 page; // Virtual-page offset, pages per axis, level, coarse flag.
    float4 originNear;
};

struct ShadowFace
{
    uint atlas, pages, projections, statistics;
    uint physicalPagesX, clipmapCount, pageSize, reserved;
};

struct ShadowMaterial
{
    float4 alpha;
    uint4 resources;
    float4 uvX, uvY;
};

struct ShadowClipVertex
{
    float4 position;
    float2 uv;
};

struct ShadowRasterVertex
{
    float4 position : SV_Position;
    [vk::location(0)] float2 uv : TEXCOORD0;
    [vk::location(1)] nointerpolation uint materialIndex : TEXCOORD1;
};

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
    if (perspective && loW <= projection.originNear.w) {
        if (hiW < projection.originNear.w) { return false; }
        lo = float2(-1.0); hi = float2(1.0);
    }
    if (any(hi < -1.0) || any(lo > 1.0)) { return false; }
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
    uint4 source = changes[table + push.clipmapCount];
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

struct VSMReceiverBasis
{
    float3 eye;
    float3 forward;
    float3 right;
    float3 up;
    float3 rightAspect;
    float tanHalfFov;
    float orthoHalfHeight;
    float2 jitter;
    float2 invViewport;
    float2 ndcScale;
    int2 viewportSize;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float nearTimesFar;
    bool isOrtho;
    bool reversed;
};

VSMReceiverBasis vsmMakeReceiverBasis(ViewConstants view)
{
    VSMReceiverBasis basis;
    basis.eye = view.current.eye.xyz;
    basis.forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    basis.right = normalize(cross(basis.forward, view.current.upProjection.xyz));
    basis.up = cross(basis.right, basis.forward);
    basis.rightAspect = basis.right * view.current.viewport.x;
    basis.tanHalfFov = tan(view.current.viewport.w * 0.5);
    basis.orthoHalfHeight = view.current.clipOrtho.z * 0.5;
    basis.jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    basis.invViewport = 1.0 / view.current.viewport.yz;
    basis.ndcScale = float2(2.0, -2.0) * basis.invViewport;
    basis.viewportSize = int2(view.current.viewport.yz);
    basis.nearPlane = view.current.clipOrtho.x;
    basis.farPlane = view.current.clipOrtho.y;
    basis.farMinusNear = view.current.clipOrtho.y - view.current.clipOrtho.x;
    basis.nearTimesFar = view.current.clipOrtho.x * view.current.clipOrtho.y;
    basis.isOrtho = view.current.upProjection.w > 0.5;
    basis.reversed = view.current.clipOrtho.w > 0.5;
    return basis;
}

// Uses the actual receiver projection, including temporal jitter in pixel units.
// clipOrtho.w selects reversed (1) or standard (0) receiver depth only; shadow
// projections always use reversed depth.
bool vsmReceiver(int2 pixel, VSMReceiverBasis basis, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    position = float3(0.0);
    linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= basis.viewportSize)) { return false; }
    float value = depth.Load(int3(pixel, 0));
    if (value == (basis.reversed ? 0.0 : 1.0)) { return false; }
    if (basis.isOrtho) {
        linearDepth = basis.reversed ? lerp(basis.farPlane, basis.nearPlane, value) : lerp(basis.nearPlane, basis.farPlane, value);
    } else {
        float denominator = basis.reversed ? basis.nearPlane + value * basis.farMinusNear
            : basis.farPlane - value * basis.farMinusNear;
        linearDepth = basis.nearTimesFar / denominator;
    }
    float2 ndc = fma(float2(pixel) + 0.5 + basis.jitter, basis.ndcScale, float2(-1.0, 1.0));
    float halfHeight = basis.isOrtho ? basis.orthoHalfHeight
        : linearDepth * basis.tanHalfFov;
    position = basis.eye + basis.forward * linearDepth +
        halfHeight * (basis.rightAspect * ndc.x + basis.up * ndc.y);
    return true;
}

bool vsmReceiver(int2 pixel, ViewConstants view, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    return vsmReceiver(pixel, vsmMakeReceiverBasis(view), depth, position, linearDepth);
}

float3 vsmReceiverNormal(int2 pixel, VSMReceiverBasis basis, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    float3 left, right, top, bottom;
    float zl, zr, zt, zb;
    bool hasLeft = vsmReceiver(pixel - int2(1, 0), basis, depth, left, zl);
    bool hasRight = vsmReceiver(pixel + int2(1, 0), basis, depth, right, zr);
    bool hasTop = vsmReceiver(pixel - int2(0, 1), basis, depth, top, zt);
    bool hasBottom = vsmReceiver(pixel + int2(0, 1), basis, depth, bottom, zb);
    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));
    float3 dx = useLeft ? position - left : (hasRight ? right - position : float3(0.0));
    float3 dy = useTop ? position - top : (hasBottom ? bottom - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

float3 vsmReceiverNormal(int2 pixel, ViewConstants view, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    return vsmReceiverNormal(pixel, vsmMakeReceiverBasis(view), depth, position, linearDepth);
}

// Screen depth limits the uncertain segment near a simplified shadow receiver.
// As in UE's screen-ray bias, this changes only the comparison depth, never the
// receiver's page/clipmap choice. Point samples are a bounded heuristic, not a
// visibility oracle for hidden or subpixel geometry.
float vsmLODScreenRayOffset(float3 position, float3 normal, float linearDepth, float3 toLight,
    ViewConstants view, Texture2D<float> depth, float screenLength, float blockerDistance, float bias)
{
    if (screenLength == 0.0) { return 0.0; }
    bool orthographic = view.current.upProjection.w > 0.5;
    float tangent = tan(view.current.viewport.w * 0.5);
    float viewHeight = orthographic ? view.current.clipOrtho.z : 2.0 * linearDepth * tangent;
    float maximumLength = screenLength * viewHeight;
    if (blockerDistance - bias > maximumLength) { return 0.0; }
    float normalLight = dot(normal, toLight);
    if (normalLight <= 1e-5) { return 0.0; }
    // Concentrate samples around the depth conflict. A fixed camera-scaled
    // stride can jump completely over a nearby thin blocker before its first
    // sample, even when that blocker is present in the primary depth buffer.
    float rayLength = min(maximumLength, max(0.0, blockerDistance) + bias);
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    for (uint step = 1u; step <= 4u; ++step) {
        float time = float(step) * 0.25;
        float3 delta = position + toLight * (time * rayLength) - view.current.eye.xyz;
        float rayDepth = dot(delta, forward);
        float safeLength = rayLength * max(0.0, time - 0.375);
        if (rayDepth <= view.current.clipOrtho.x || rayDepth >= view.current.clipOrtho.y) { return 0.0; }
        float halfHeight = orthographic ? viewHeight * 0.5 : rayDepth * tangent;
        float2 ndc = float2(dot(delta, right) / view.current.viewport.x, dot(delta, up)) / halfHeight;
        float2 screen = float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * view.current.viewport.yz - jitter;
        if (any(screen < 0.0) || any(screen >= view.current.viewport.yz)) { return 0.0; }
        float3 samplePosition;
        float sampleDepth;
        if (vsmReceiver(int2(screen), view, depth, samplePosition, sampleDepth)) {
            // A thin surface may already be behind the sampled ray point. Its
            // height above the receiver plane still indicates a possible near
            // blocker; keep the original shadow instead of stepping over it.
            float planeDistance = dot(normal, samplePosition - position) / normalLight;
            if (planeDistance > bias && planeDistance <= rayLength + bias) { return 0.0; }
            if (rayDepth > sampleDepth) { return safeLength; }
        }
    }
    return rayLength;
}

#if VSM_REFINED_RECEIVER
float3 vsmPrimaryCameraNormal(VirtualShadowMapReceiverTriangle triangle, ViewConstants view)
{
    float3 toCamera = view.current.upProjection.w > 0.5 ? view.current.eye.xyz - view.current.center.xyz :
        view.current.eye.xyz - triangle.a;
    return dot(triangle.normal, toCamera) >= 0.0 ? triangle.normal : -triangle.normal;
}

void vsmLimitReceiverRay(float planeDistance, float planeRate, inout float distance)
{
    if (planeRate < 0.0) { distance = min(distance, planeDistance / -planeRate); }
}

bool vsmReceiverCandidateBlocks(uint2 pixel, VirtualShadowMapReceiverContext context,
    float3 origin, float3 direction, bool nearReceiver, inout uint4 recent0, inout uint4 recent1,
    inout uint previousVisibility)
{
    if (nearReceiver) {
        return vsmPrimaryClusterBlocksRay(pixel, context, origin, direction, recent0, recent1);
    }
    return vsmPrimaryTriangleBlocksRayAtPixel(pixel, context, origin, direction, recent0, recent1, previousVisibility);
}

bool vsmAllowLODReceiverCorrection(uint2 pixel, float3 position, float linearDepth, float3 toLight,
    ViewConstants view, uint receiverGeometry, float screenLength)
{
    // Missing primary geometry is an optional representation, not permission
    // to infer that a depth-only miss is visible. Keep the original shadow.
    if (receiverGeometry == 0xffffffffu) { return false; }
    VirtualShadowMapReceiverContext context = vsmResolveReceiverContext(receiverGeometry);
    VirtualShadowMapReceiverTriangle receiver;
    if (!vsmPrimaryReceiverTriangle(pixel, context, receiver)) { return false; }
    if (dot(vsmPrimaryCameraNormal(receiver, view), toLight) <= 0.0) { return false; }

    // Missing/discontinuous primary ownership cannot establish that the shadow
    // is a receiver error. Adjacent facing alone does not establish occlusion;
    // the candidate clusters below test their actual triangles instead.
    static const int2 neighbors[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    for (uint i = 0u; i < 4u; ++i) {
        int2 neighbor = int2(pixel) + neighbors[i];
        if (any(neighbor < 0) || any(neighbor >= int2(view.current.viewport.yz))) { return false; }
        VirtualShadowMapReceiverTriangle adjacent;
        if (!vsmPrimaryReceiverTriangle(uint2(neighbor), context, adjacent) ||
            adjacent.instanceIndex != receiver.instanceIndex) { return false; }
    }

    bool orthographic = view.current.upProjection.w > 0.5;
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 cameraRay = orthographic ? forward : normalize(position - view.current.eye.xyz);
    float denominator = dot(receiver.normal, cameraRay);
    if (abs(denominator) <= 1e-5) { return false; }
    // Recover the primary triangle's surface rather than tracing from rounded
    // D32 reconstruction, which can lie just inside its own fine geometry.
    float3 origin = position + cameraRay * (dot(receiver.normal, receiver.a - position) / denominator);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float3 cameraOrigin = origin - view.current.eye.xyz;
    float axial = dot(cameraOrigin, forward), axialRate = dot(toLight, forward);
    float2 transverse = float2(dot(cameraOrigin, right), dot(cameraOrigin, up));
    float2 transverseRate = float2(dot(toLight, right), dot(toLight, up));
    float tangent = tan(view.current.viewport.w * 0.5);
    float2 scale = float2(view.current.viewport.x, 1.0) *
        (orthographic ? view.current.clipOrtho.z * 0.5 : tangent);
    float2 extent = scale * (orthographic ? 1.0 : axial);
    float2 extentRate = scale * (orthographic ? 0.0 : axialRate);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    float2 lower = float2(-1.0, -1.0) + float2(2.0, -2.0) * jitter / view.current.viewport.yz;
    float2 upper = lower + 2.0;
    float2 lowerDistance = transverse - lower * extent;
    float2 upperDistance = upper * extent - transverse;
    if (axial <= view.current.clipOrtho.x || axial >= view.current.clipOrtho.y ||
        any(lowerDistance < 0.0) || any(upperDistance < 0.0)) { return false; }
    // Candidate collection ends at the camera frustum, not at the bias length.
    // A distant fine blocker can produce a short conflict in the LOD atlas.
    float rayLength = 1e30;
    vsmLimitReceiverRay(axial - view.current.clipOrtho.x, axialRate, rayLength);
    vsmLimitReceiverRay(view.current.clipOrtho.y - axial, -axialRate, rayLength);
    [unroll] for (uint axis = 0u; axis < 2u; ++axis) {
        vsmLimitReceiverRay(lowerDistance[axis], transverseRate[axis] - lower[axis] * extentRate[axis], rayLength);
        vsmLimitReceiverRay(upperDistance[axis], upper[axis] * extentRate[axis] - transverseRate[axis], rayLength);
    }
    float endDepth = axial + axialRate * rayLength;
    if (endDepth <= 0.0) { return false; }
    float2 ndc = (transverse + transverseRate * rayLength) / (scale * (orthographic ? 1.0 : endDepth));
    float2 endScreen = float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * view.current.viewport.yz - jitter;
    endScreen = clamp(endScreen, float2(0.0), view.current.viewport.yz);

    // Near the receiver, include cluster microtriangles that never covered a
    // pixel center. Beyond the original offset interval, query only the visible
    // primary triangles. All candidate intersections test the whole forward ray.
    // This finite screen representation does not replace hidden/offscreen scene
    // geometry, change the shadow caster cut, or require an additional AS.
    uint4 recent0 = uint4(0xffffffffu), recent1 = uint4(0xffffffffu);
    for (uint i = 0u; i < 4u; ++i) {
        if (vsmPrimaryClusterBlocksRay(uint2(int2(pixel) + neighbors[i]), context,
            origin, toLight, recent0, recent1)) { return false; }
    }
    float2 startScreen = float2(pixel) + 0.5;
    float2 deltaScreen = endScreen - startScreen;
    float viewHeight = orthographic ? view.current.clipOrtho.z : 2.0 * linearDepth * tangent;
    float nearDistance = min(screenLength * viewHeight, rayLength);
    float nearDepth = axial + axialRate * nearDistance;
    float2 nearNDC = (transverse + transverseRate * nearDistance) / (scale * (orthographic ? 1.0 : nearDepth));
    float2 nearScreen = float2(nearNDC.x + 1.0, 1.0 - nearNDC.y) * 0.5 * view.current.viewport.yz - jitter;
    float screenDistanceSquared = dot(deltaScreen, deltaScreen);
    float nearFraction = screenDistanceSquared > 1e-20 ?
        saturate(dot(nearScreen - startScreen, deltaScreen) / screenDistanceSquared) : 1.0;
    int2 cell = int2(pixel), lastCell = min(int2(endScreen), int2(view.current.viewport.yz) - 1);
    int2 direction = int2(deltaScreen.x >= 0.0 ? 1 : -1, deltaScreen.y >= 0.0 ? 1 : -1);
    float2 stepTime = 1.0 / max(abs(deltaScreen), float2(1e-30));
    float2 nextTime = stepTime * 0.5;
    float cellEntryTime = 0.0;
    uint previousVisibility = 0u;
    // A clipped 2D grid ray crosses at most width + height cell boundaries.
    uint maximumVisits = uint(view.current.viewport.y + view.current.viewport.z) + 1u;
    for (uint visited = 0u; visited < maximumVisits; ++visited) {
        // Accumulated step times can round a boundary tie toward an exterior
        // cell before lastCell. Do not read outside the primary image.
        if (any(cell < 0) || any(cell >= int2(view.current.viewport.yz))) { return false; }
        if (vsmReceiverCandidateBlocks(uint2(cell), context, origin, toLight, cellEntryTime <= nearFraction,
            recent0, recent1, previousVisibility)) { return false; }
        if (all(cell == lastCell)) { return true; }
        if (nextTime.x == nextTime.y) {
            // At a corner, the following iteration visits the Y-side cell.
            // Include the X-side cell as well before reaching the diagonal.
            int2 corner = cell + int2(direction.x, 0);
            if (all(corner >= 0) && all(corner < int2(view.current.viewport.yz)) &&
                vsmReceiverCandidateBlocks(uint2(corner), context, origin, toLight, nextTime.x <= nearFraction,
                    recent0, recent1, previousVisibility)) { return false; }
        }
        if (nextTime.x < nextTime.y) { cellEntryTime = nextTime.x; cell.x += direction.x; nextTime.x += stepTime.x; }
        else { cellEntryTime = nextTime.y; cell.y += direction.y; nextTime.y += stepTime.y; }
    }
    // Keep the original shadow if numerical traversal did not reach its exit.
    return false;
}
#endif

float vsmPerspectiveReceiverDepth(float3 position, float3 normal, uint2 virtualTexel,
    ShadowProjection projection, uint pageSize)
{
    float2 sampleNDC = (float2(virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 ray = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * sampleNDC.x + normalize(projection.rowY.xyz) * sampleNDC.y);
    float3 samplePosition = position;
    float denominator = dot(normal, ray);
    if (abs(denominator) > 1e-5 * length(ray)) {
        float distanceAlongRay = dot(normal, position - projection.originNear.xyz) / denominator;
        float3 intersection = projection.originNear.xyz + ray * distanceAlongRay;
        if (distanceAlongRay > 0.0 && all(isfinite(intersection))) { samplePosition = intersection; }
    }
    float axial = dot(projection.rowW, float4(samplePosition, 1.0));
    // A plane extrapolated to/before near cannot be meaningfully compared to
    // clipped caster depth. It is nearer than every stored sample.
    if (axial <= projection.originNear.w) { return 1.0; }
    float3 delta = samplePosition - projection.originNear.xyz;
    float distance = length(delta);
    float biasScale = max(1.0 - projection.metric.z / distance, projection.originNear.w / axial);
    float3 biasedPosition = projection.originNear.xyz + delta * biasScale;
    float4 clip = vsmProjectClip(biasedPosition, projection);
    return clip.z / clip.w;
}

uint vsmSampleHash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float vsmSampleRandom(uint2 pixel, uint dimension, uint phase)
{
    float spatial = float(vsmSampleHash(pixel.x ^ vsmSampleHash(pixel.y + dimension * 0x9e3779b9u)) >> 8) / 16777216.0;
    // Preserve deterministic raw rendering. With history enabled, rotate both
    // disk and texel/step dimensions with a bounded low-discrepancy sequence.
    return phase == 0u ? spatial : frac(spatial + float(phase) * ((dimension & 1u) != 0u ? 0.569840291 : 0.754877666));
}

float2 vsmDiskSample(uint ray, uint rayCount, float2 scramble)
{
    // Scramble the stratified disk per receiver. A common disk point at one ray
    // shifts a hard shadow instead of estimating area-source visibility.
    float2 sample = frac(float2(float(ray) / float(rayCount), float(ray) * 0.61803398875) +
        scramble);
    float angle = sample.y * 6.28318530718;
    return sqrt(sample.x) * float2(cos(angle), sin(angle));
}

float2 vsmDiskSample(uint ray, uint rayCount, uint2 pixel, uint phase)
{
    return vsmDiskSample(ray, rayCount,
        float2(vsmSampleRandom(pixel, 0u, phase), vsmSampleRandom(pixel, 1u, phase)));
}

float3 vsmRayEndpointDirectional(float3 position, float3 toLight, float3 offset, float tanAlpha, float rayLength)
{
    return position + normalize(toLight + offset * tanAlpha) * rayLength;
}

float3 vsmRayEndpoint(float3 position, ShadowProjection projection, ShadowPush push, float3 toLight, float3 offset)
{
    if (push.lightType == 0u) {
        return position + normalize(toLight + offset * tan(push.angularRadius)) * push.rayLength;
    }
    float distance = length(projection.originNear.xyz - position);
    // Metallic's local source is a planar disk at the light center.
    float3 direction = normalize(toLight * distance + offset * push.sourceRadius);
    float sinTheta = length(cross(direction, toLight));
    float cosTheta = dot(direction, toLight);
    // A single center-light depth layer becomes a poor proxy near the emitter.
    // Limit its angular span, and stay clear of the perspective singularity.
    float distanceScale = 0.75 * saturate(1.5 / (cosTheta + sinTheta / tan(push.maxRayAngle)));
    float3 endpoint = position + direction * min(push.rayLength, distance * distanceScale);
    if (push.lightType == 1u) {
        float startW = dot(projection.rowW, float4(position, 1.0));
        float endW = dot(projection.rowW, float4(endpoint, 1.0));
        // Very wide spots can cross the near plane even with the angular cap.
        // Clip once in world space before interpolating post-projection depth.
        if (endW < projection.originNear.w) {
            endpoint = lerp(position, endpoint, (startW - projection.originNear.w) / (startW - endW));
        }
    }
    return endpoint;
}

float3 vsmRayEndpoint(float3 position, ShadowProjection projection, ShadowPush push, uint ray, uint2 pixel)
{
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float2 disk = vsmDiskSample(ray, push.rayCount, pixel, push.samplePhase);
    return vsmRayEndpoint(position, projection, push, toLight, right * disk.x + up * disk.y);
}

float vsmRaySampleTime(uint step, ShadowPush push, float stepRandom)
{
    if (step == push.samplesPerRay) { return 0.0; }
    float t = 1.0 - (float(step + 1u) - stepRandom) / float(push.samplesPerRay);
    return t * t;
}

float vsmRaySampleTime(uint step, ShadowPush push, uint2 pixel)
{
    if (step == push.samplesPerRay) { return 0.0; }
    return vsmRaySampleTime(step, push, vsmSampleRandom(pixel, 2u, push.samplePhase));
}

float3 vsmRayPosition(float3 position, float3 endpoint, float time, ShadowProjection projection, ShadowPush push)
{
    if (push.lightType != 0u) {
        float3 forward = normalize(projection.rowW.xyz);
        if (push.lightType == 2u) {
            float3 right, up;
            virtualShadowMapPointBasis(virtualShadowMapPointFace(position - projection.originNear.xyz), right, up, forward);
        }
        float startW = dot(position - projection.originNear.xyz, forward);
        float endW = dot(endpoint - projection.originNear.xyz, forward);
        // Linear interpolation after perspective division, reconstructed into
        // world space so a point-light trace can cross canonical cube faces.
        time = time * startW / lerp(endW, startW, time);
    }
    return lerp(position, endpoint, time);
}

float vsmWorldTexelSize(float3 position, ShadowProjection projection, uint pageSize, uint lightType)
{
    if (lightType == 0u) { return projection.metric.y; }
    float3 delta = abs(position - projection.originNear.xyz);
    float axial = lightType == 2u ? max(delta.x, max(delta.y, delta.z)) :
        dot(projection.rowW, float4(position, 1.0));
    return 2.0 * max(axial, projection.originNear.w) * projection.metric.x / float(projection.page.y * pageSize);
}

float3 vsmRayDither(float3 position, ShadowProjection projection, ShadowPush push, uint ray, uint2 pixel)
{
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float2 offset = float2(vsmSampleRandom(pixel, 3u + 2u * ray, push.samplePhase),
        vsmSampleRandom(pixel, 4u + 2u * ray, push.samplePhase)) - 0.5;
    // Dither the shadow-map texel independently of the source sample. This
    // antialiases contact edges narrower than a receiver-selected texel. A
    // world-space offset remains continuous when a ray crosses a cube face.
    return (right * offset.x + up * offset.y) * vsmWorldTexelSize(position, projection, push.pageSize, push.lightType);
}

float vsmPixelFootprint(float linearDepth, ViewConstants view, ShadowPush push)
{
    if (push.receiverFootprint == 0u) { return 0.0; }
    return view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z / view.current.viewport.z :
        2.0 * linearDepth * tan(view.current.viewport.w * 0.5) / view.current.viewport.z;
}

float vsmSearchLength(float3 position, float linearDepth, VSMReceiverBasis basis, ShadowPush push)
{
    if (push.rayLengthScale == 0.0) { return push.rayLength; }
    float distance = basis.isOrtho ? linearDepth : length(position - basis.eye);
    return min(push.rayLength, push.rayLengthScale * distance);
}

float vsmSearchLength(float3 position, float linearDepth, ViewConstants view, ShadowPush push)
{
    if (push.rayLengthScale == 0.0) { return push.rayLength; }
    // Perspective follows view-origin distance, as directional SMRT does in UE.
    // Orthographic receivers use axial depth so panning cannot change the cap.
    float distance = view.current.upProjection.w > 0.5 ? linearDepth : length(position - view.current.eye.xyz);
    return min(push.rayLength, push.rayLengthScale * distance);
}

float vsmProjectedReceiverFootprint(float3 position, float3 normal, float linearDepth,
    VSMReceiverBasis basis, ShadowProjection projection, ShadowPush push)
{
    if (push.receiverFootprint == 0u) { return 0.0; }
    float footprint = basis.isOrtho ? (basis.orthoHalfHeight * 2.0) * basis.invViewport.y :
        (2.0 * linearDepth * basis.tanHalfFov) * basis.invViewport.y;
    float3 viewRay = basis.isOrtho ? basis.forward : position - basis.eye;
    float denominator = dot(normal, viewRay);
    // An undefined or edge-on depth plane has no finite pixel differential.
    if (abs(denominator) <= 1e-5 * length(viewRay)) { return footprint; }
    float3 dx = basis.right - viewRay * (dot(normal, basis.right) / denominator);
    float3 dy = basis.up - viewRay * (dot(normal, basis.up) / denominator);
    float3 lightRight = normalize(projection.rowX.xyz), lightUp = normalize(projection.rowY.xyz);
    if (push.lightType != 0u) {
        float3 fromLight = position - projection.originNear.xyz;
        float3 lightForward = normalize(projection.rowW.xyz);
        if (push.lightType == 2u) {
            // Every face's demand dispatch must choose the same receiver mip.
            virtualShadowMapPointBasis(virtualShadowMapPointFace(fromLight), lightRight, lightUp, lightForward);
        }
        float axial = dot(fromLight, lightForward);
        dx -= fromLight * (dot(dx, lightForward) / axial);
        dy -= fromLight * (dot(dy, lightForward) / axial);
    }
    float2 a = float2(dot(dx, lightRight), dot(dx, lightUp));
    float2 b = float2(dot(dy, lightRight), dot(dy, lightUp));
    float aa = dot(a, a), ab = dot(a, b), bb = dot(b, b);
    float maximumSquared = 0.5 * (aa + bb + length(float2(aa - bb, 2.0 * ab)));
    // The smaller singular value preserves resolution along the compressed
    // axis at grazing incidence. An area ratio misses projection shear.
    return footprint * abs(a.x * b.y - a.y * b.x) * rsqrt(maximumSquared);
}

float vsmProjectedReceiverFootprint(float3 position, float3 normal, float linearDepth,
    ViewConstants view, ShadowProjection projection, ShadowPush push)
{
    return vsmProjectedReceiverFootprint(position, normal, linearDepth, vsmMakeReceiverBasis(view), projection, push);
}

struct VSMReceiverBasisGroup
{
    float3 eye;
    float3 forward;
    float3 right;
    float3 up;
    float3 rightAspect;
    float tanHalfFov;
    float orthoHalfHeight;
    float2 jitter;
    float2 invViewport;
    float2 ndcScale;
    int2 viewportSize;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float nearTimesFar;
    uint isOrtho;
    uint reversed;
};

struct VSMDirectionalUniformData
{
    float3 toLight;
    float3 right;
    float3 up;
    float3 normX;
    float3 normY;
    float biasZ;
    float sinAlpha;
    float tanAngularRadius;
    float coarseInvResolution;
    float receiverInvResolution;
    uint allowClipmaps;
    uint coarseValid;
    uint2 coarsePhysicalCoord;
    uint powerOfTwoLayout;
    uint pageSizeShift;
    uint pageSizeMask;
    uint physicalPagesXShift;
    uint physicalPagesXMask;
    uint virtualPagesPerAxisShift;
};

#if !VSM_ONE_PASS_PROJECTION
static const uint kVSMMaxCachedProjections = 24u;
groupshared ShadowProjection gVsmProjections[kVSMMaxCachedProjections];
groupshared VSMReceiverBasisGroup gVsmReceiverBasis;
groupshared VSMDirectionalUniformData gVsmDirectionalUniform;

VSMReceiverBasis vsmGetGroupReceiverBasis()
{
    VSMReceiverBasis b;
    b.eye = gVsmReceiverBasis.eye;
    b.forward = gVsmReceiverBasis.forward;
    b.right = gVsmReceiverBasis.right;
    b.up = gVsmReceiverBasis.up;
    b.rightAspect = gVsmReceiverBasis.rightAspect;
    b.tanHalfFov = gVsmReceiverBasis.tanHalfFov;
    b.orthoHalfHeight = gVsmReceiverBasis.orthoHalfHeight;
    b.jitter = gVsmReceiverBasis.jitter;
    b.invViewport = gVsmReceiverBasis.invViewport;
    b.ndcScale = gVsmReceiverBasis.ndcScale;
    b.viewportSize = gVsmReceiverBasis.viewportSize;
    b.nearPlane = gVsmReceiverBasis.nearPlane;
    b.farPlane = gVsmReceiverBasis.farPlane;
    b.farMinusNear = gVsmReceiverBasis.farMinusNear;
    b.nearTimesFar = gVsmReceiverBasis.nearTimesFar;
    b.isOrtho = gVsmReceiverBasis.isOrtho != 0u;
    b.reversed = gVsmReceiverBasis.reversed != 0u;
    return b;
}
#endif

ShadowProjection vsmGetCachedProjection(StructuredBuffer<ShadowProjection> projections, uint level, bool useGroupCache = false)
{
#if !VSM_ONE_PASS_PROJECTION
    if (useGroupCache && level < kVSMMaxCachedProjections) {
        return gVsmProjections[level];
    }
#endif
    return projections[level];
}

uint vsmReceiverLevel(float3 position, StructuredBuffer<ShadowProjection> projections,
    ShadowPush push, float footprint, bool useGroupCache = false)
{
    uint level = 0u;
    while (level + 1u < push.clipmapCount &&
        vsmWorldTexelSize(position, vsmGetCachedProjection(projections, level, useGroupCache), push.pageSize, push.lightType) < footprint * 0.5) { ++level; }
    return level;
}

float3 vsmPCFPosition(float3 position, float3 normal, ShadowProjection projection,
    ShadowPush push, int2 tap)
{
    float3 right = normalize(projection.rowX.xyz), up = normalize(projection.rowY.xyz);
    float3 toLight = push.lightType == 0u ? normalize(projection.rowZ.xyz) :
        normalize(projection.originNear.xyz - position);
    if (push.lightType == 2u) {
        float3 forward;
        virtualShadowMapPointBasis(virtualShadowMapPointFace(position - projection.originNear.xyz), right, up, forward);
    }
    float scale = push.pcfRadius * vsmWorldTexelSize(position, projection, push.pageSize, push.lightType);
    float3 offset = (right * float(tap.x) + up * float(tap.y)) * scale;
    float denominator = dot(normal, toLight);
    if (abs(denominator) > 1e-5) { offset -= toLight * (dot(normal, offset) / denominator); }
    return position + offset;
}

#if VSM_DIRECTIONAL_PROJECTION
float vsmDirectionalComparisonDepth(float3 normal, float3 projected, uint2 texel,
    float extent, float invResolution, float3 normX, float3 normY,
    float invDenom, float biasZ)
{
    float2 sampleNDC = (float2(texel) + 0.5) * invResolution - 1.0;
    float2 offset = (sampleNDC - projected.xy) * extent;
    float3 delta = normX * offset.x + normY * offset.y;
    float correction = -dot(normal, delta) * invDenom;
    return projected.z + correction + biasZ;
}

float vsmDirectionalComparisonDepth(float3 projected, uint2 texel,
    float extent, float invResolution, float2 normalSlope, float biasZ)
{
    float2 sampleNDC = fma(float2(texel), float2(invResolution), float2(0.5 * invResolution - 1.0));
    float2 offset = (sampleNDC - projected.xy) * extent;
    float correction = -dot(normalSlope, offset);
    return projected.z + correction + biasZ;
}
#endif

float vsmComparisonDepth(float3 position, float3 normal, float3 projected, uint2 texel,
    ShadowProjection projection, uint pageSize)
{
#if !VSM_DIRECTIONAL_PROJECTION
    if (projection.metric.w > 0.5) { return vsmPerspectiveReceiverDepth(position, normal, texel, projection, pageSize); }
#endif
    float denominator = dot(normal, normalize(projection.rowZ.xyz));
    float correction = 0.0;
    if (abs(denominator) > 1e-5) {
        float2 sampleNDC = (float2(texel) + 0.5) * (2.0 / float(projection.page.y * pageSize)) - 1.0;
        float2 offset = (sampleNDC - projected.xy) * projection.metric.x;
        float3 delta = normalize(projection.rowX.xyz) * offset.x + normalize(projection.rowY.xyz) * offset.y;
        correction = -dot(normal, delta) * length(projection.rowZ.xyz) / denominator;
    }
    return projected.z + correction + projection.metric.z;
}

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
void vsmMapDirectionalAncestor(uint4 mapping, float2 sourceTranslation, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, ShadowPush push, uint level, bool useGroupCache = false)
{
    uint targetLevel = mapping.y;
    float scale = asfloat((127u - (targetLevel - level)) << 23u);
    ShadowProjection targetProj = vsmGetCachedProjection(projections, targetLevel, useGroupCache);
    float2 translation = float2(targetProj.rowX.w, targetProj.rowY.w) -
        sourceTranslation * scale;
    projected.xy = projected.xy * scale + translation;
    uint2 page = uint2(mapping.z & 0xffffu, mapping.z >> 16u);
    uint2 firstTexel = page * push.pageSize;
    float2 address = (projected.xy * 0.5 + 0.5) * float(targetProj.page.y * push.pageSize);
    texel = uint2(clamp(address, float2(firstTexel), float2(firstTexel + push.pageSize - 1u)));
    vsmWorkAdd(push, kVSMWorkProjectionMappedAncestors, 1u);
}

bool vsmLookupPropagated(float3 position, float2 sourceTranslation, uint sourcePage, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    ShadowPush push, uint level, out uint physicalPage, out uint targetLevel, bool useGroupCache = false)
{
    // One-pass jobs carry their own clipmapCount; fixed VPA^2 level strides
    // also apply to local mips, whose unused prefix slots remain unmapped.
    uint tableSize = push.clipmapCount * push.virtualPagesPerAxis * push.virtualPagesPerAxis + 1u;
    uint4 mapping = pages[tableSize + sourcePage];
    vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
    physicalPage = mapping.x;
    targetLevel = mapping.y;
    if (mapping.w == 0u) {
        vsmWorkAdd(push, kVSMWorkProjectionMappingSlowPaths, 1u);
        return false;
    }
    if (mapping.w == 1u) { return true; }
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
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(atlasHandle));
#else
    Texture2D<float> atlas = resolveNonUniform(ResourceHandle<Texture2D<float>>(atlasHandle));
#endif
    result.storedDepth = atlas.Load(int3(texel, 0));
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

#if VSM_DIRECTIONAL_PROJECTION
// makeProjections owns this contract: fine XY axes are dyadic, every level
// (including coarse) shares rowZ, and the separate coarse XY encloses the scene.
// Keep both origins: lookup has texel dither; receiver-plane comparison does not.
struct VSMDirectionalRayBase
{
    float3 comparisonOrigin;
    float2 coarseComparisonOrigin;
    float2 sourceTranslation;
    uint receiverLevel;
    uint receiverPageOffset;
    uint receiverMappingOffset;
    uint coarseLevel;
    float invDepthScale;
    float halfReceiverResolution;
    float receiverMarginMax;
    bool allowClipmaps;
    uint2 coarsePhysicalCoord;
    bool coarseValid;
    uint pageSizeShift;
    uint pageSizeMask;
    uint physicalPagesXShift;
    uint physicalPagesXMask;
    uint virtualPagesPerAxisShift;
    bool powerOfTwoLayout;
};

struct VSMDirectionalRay
{
    float3 lookupOrigin;
    float3 step;
    float2 coarseLookupOrigin;
    float2 coarseStep;
};

struct VSMDirectionalRayState
{
    VSMDirectionalRayBase base;
    VSMDirectionalRay ray;
    float depthScale;

    property float3 comparisonOrigin { get { return base.comparisonOrigin; } }
    property float3 lookupOrigin { get { return ray.lookupOrigin; } }
    property float3 step { get { return ray.step; } }
    property float2 coarseComparisonOrigin { get { return base.coarseComparisonOrigin; } }
    property float2 coarseLookupOrigin { get { return ray.coarseLookupOrigin; } }
    property float2 coarseStep { get { return ray.coarseStep; } }
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
    property uint2 coarsePhysicalCoord { get { return base.coarsePhysicalCoord; } }
    property bool coarseValid { get { return base.coarseValid; } }
    property uint receiverMappingOffset { get { return base.receiverMappingOffset; } }
    property uint virtualPagesPerAxisShift { get { return base.virtualPagesPerAxisShift; } }
};

VSMDirectionalRayBase vsmCreateDirectionalRayBase(float3 position,
    ShadowProjection first, ShadowProjection coarse, StructuredBuffer<uint4> pages, ShadowPush push, uint receiverLevel, bool useGroupCache = false)
{
    VSMDirectionalRayBase rayBase;
    rayBase.comparisonOrigin = vsmProjectPosition(position, first);
    rayBase.coarseComparisonOrigin = vsmProjectPosition(position, coarse).xy;
    rayBase.sourceTranslation = float2(first.rowX.w, first.rowY.w);
    rayBase.receiverLevel = receiverLevel;
    rayBase.coarseLevel = push.clipmapCount;
    rayBase.receiverPageOffset = first.page.x;
    rayBase.receiverMappingOffset = push.totalPages + first.page.x;
    rayBase.invDepthScale = 1.0 / length(first.rowZ.xyz);
    float receiverResolution = float(push.virtualPagesPerAxis * push.pageSize);
    rayBase.halfReceiverResolution = 0.5 * receiverResolution;
    rayBase.receiverMarginMax = 1.0 - (1.0 / receiverResolution);
#if !VSM_ONE_PASS_PROJECTION
    if (useGroupCache && push.lightType == 0u) {
        rayBase.allowClipmaps = gVsmDirectionalUniform.allowClipmaps != 0u;
        rayBase.coarseValid = gVsmDirectionalUniform.coarseValid != 0u;
        rayBase.coarsePhysicalCoord = gVsmDirectionalUniform.coarsePhysicalCoord;
        rayBase.powerOfTwoLayout = gVsmDirectionalUniform.powerOfTwoLayout != 0u;
        rayBase.pageSizeShift = gVsmDirectionalUniform.pageSizeShift;
        rayBase.pageSizeMask = gVsmDirectionalUniform.pageSizeMask;
        rayBase.physicalPagesXShift = gVsmDirectionalUniform.physicalPagesXShift;
        rayBase.physicalPagesXMask = gVsmDirectionalUniform.physicalPagesXMask;
        rayBase.virtualPagesPerAxisShift = gVsmDirectionalUniform.virtualPagesPerAxisShift;
    } else
#endif
    {
        StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
        rayBase.allowClipmaps = statistics[5] == 0u;
        uint4 coarseEntry = pages[coarse.page.x];
        rayBase.coarseValid = (coarseEntry.y & 2u) != 0u && (coarseEntry.w & 1u) != 0u;
        rayBase.coarsePhysicalCoord = vsmPhysicalPageCoord(coarseEntry.x, push.physicalPagesX);
        bool isPow2PageSize = (push.pageSize & (push.pageSize - 1u)) == 0u && push.pageSize != 0u;
        bool isPow2PhysicalX = (push.physicalPagesX & (push.physicalPagesX - 1u)) == 0u && push.physicalPagesX != 0u;
        bool isPow2VirtualPages = (push.virtualPagesPerAxis & (push.virtualPagesPerAxis - 1u)) == 0u && push.virtualPagesPerAxis != 0u;
        rayBase.powerOfTwoLayout = isPow2PageSize && isPow2PhysicalX && isPow2VirtualPages;
        rayBase.pageSizeShift = isPow2PageSize ? uint(firstbitlow(push.pageSize)) : 0u;
        rayBase.pageSizeMask = isPow2PageSize ? (push.pageSize - 1u) : 0u;
        rayBase.physicalPagesXShift = isPow2PhysicalX ? uint(firstbitlow(push.physicalPagesX)) : 0u;
        rayBase.physicalPagesXMask = isPow2PhysicalX ? (push.physicalPagesX - 1u) : 0u;
        rayBase.virtualPagesPerAxisShift = isPow2VirtualPages ? uint(firstbitlow(push.virtualPagesPerAxis)) : 0u;
    }
    return rayBase;
}

VSMDirectionalRayState vsmCreateDirectionalRayState(float3 position, float3 endpoint, float3 dither,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages, ShadowPush push, uint receiverLevel, bool useGroupCache = false)
{
    ShadowProjection first = vsmGetCachedProjection(projections, receiverLevel, useGroupCache);
    ShadowProjection coarse = vsmGetCachedProjection(projections, push.clipmapCount, useGroupCache);
    VSMDirectionalRayBase base = vsmCreateDirectionalRayBase(position, first, coarse, pages, push, receiverLevel, useGroupCache);
    float3 direction = endpoint - position;
    VSMDirectionalRay ray;
    ray.lookupOrigin = vsmProjectPosition(position + dither, first);
    ray.step = float3(dot(first.rowX.xyz, direction), dot(first.rowY.xyz, direction), dot(first.rowZ.xyz, direction));
    ray.coarseLookupOrigin = vsmProjectPosition(position + dither, coarse).xy;
    ray.coarseStep = float2(dot(coarse.rowX.xyz, direction), dot(coarse.rowY.xyz, direction));
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
    ShadowProjection projection, bool lookup)
{
    float3 position = (lookup ? ray.lookupOrigin : rayBase.comparisonOrigin) + time * ray.step;
    if (level == rayBase.coarseLevel) {
        position.xy = (lookup ? ray.coarseLookupOrigin : rayBase.coarseComparisonOrigin) + time * ray.coarseStep;
    } else {
        float scale = asfloat(uint(127 - (int(level) - int(rayBase.receiverLevel))) << 23u);
        float2 translation = float2(projection.rowX.w, projection.rowY.w) - rayBase.sourceTranslation * scale;
        position.xy = position.xy * scale + translation;
    }
    return position;
}

float3 vsmDirectionalRayPosition(VSMDirectionalRayState ray, float time, uint level,
    ShadowProjection projection, bool lookup)
{
    return vsmDirectionalRayPosition(ray.ray, ray.base, time, level, projection, lookup);
}

bool vsmDirectionalRayMapping(float2 sourceTranslation, uint sourcePage, inout float3 projected,
    inout uint2 texel, StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    ShadowPush push, uint level, out uint physicalPage, out uint targetLevel, bool useGroupCache = false)
{
    uint4 mapping = pages[push.totalPages + sourcePage];
    vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
    physicalPage = mapping.x;
    targetLevel = mapping.y;
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

ShadowLookup vsmSamplePhysicalPage(
    uint physicalPage, uint level, float3 projected, uint2 virtualTexel,
    VSMDirectionalRayBase rayBase, Texture2D<float> atlas, ShadowPush push)
{
    ShadowLookup result;
    result.projectionsHandle = push.projections;
    result.level = level;
    result.projected = projected;
    result.virtualTexel = virtualTexel;
    uint2 physical;
    if (rayBase.powerOfTwoLayout) {
        uint physXMask = rayBase.physicalPagesXMask;
        uint physXShift = rayBase.physicalPagesXShift;
        uint pageSizeShift = rayBase.pageSizeShift;
        uint pageSizeMask = rayBase.pageSizeMask;
        physical = uint2(physicalPage & physXMask, physicalPage >> physXShift);
        result.storedDepth = atlas.Load(int3((physical << pageSizeShift) | (virtualTexel & pageSizeMask), 0));
    } else {
        physical = vsmPhysicalPageCoord(physicalPage, push.physicalPagesX);
        result.storedDepth = atlas.Load(int3(vsmAtlasTexelFromVirtual(physical, virtualTexel, push.pageSize), 0));
    }
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

[noinline]
ShadowLookup vsmDirectionalRayLookupSlow(
    float3 p, uint2 texel, uint page, uint4 mapping, bool inBounds,
    float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache)
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
        } else if (mapping.y == push.clipmapCount) {
            mappedToCoarse = true;
        } else {
            vsmMapDirectionalAncestor(mapping, rayBase.sourceTranslation, p, texel, projections, push, rayBase.receiverLevel, useGroupCache);
            return vsmSamplePhysicalPage(mapping.x, mapping.y, p, texel, rayBase, atlas, push);
        }
    } else {
        cache.lastPage = 0xffffffffu;
        cache.lastPageHit = false;
    }

    if (!mappedToCoarse) {
        float2 basePXY = p.xy;
        bool isPow2 = rayBase.powerOfTwoLayout;
        uint pageSizeShift = rayBase.pageSizeShift;
        uint vpaShift = rayBase.virtualPagesPerAxisShift;
        for (uint level = rayBase.receiverLevel + 1u; level < push.clipmapCount; ++level) {
            ShadowProjection projection = vsmGetCachedProjection(projections, level, useGroupCache);
            float scale = asfloat(uint(127 - (int(level) - int(rayBase.receiverLevel))) << 23u);
            float2 translation = float2(projection.rowX.w, projection.rowY.w) - rayBase.sourceTranslation * scale;
            float3 fallbackP = float3(basePXY * scale + translation, p.z);
            if (!vsmContains(fallbackP, projection, push.pageSize, true)) { continue; }
            uint2 fallbackTexel = vsmVirtualTexel(fallbackP, projection, push.pageSize);
            uint fallbackPage;
            if (isPow2) {
                uint2 pageXY = fallbackTexel >> pageSizeShift;
                fallbackPage = projection.page.x + ((pageXY.y << vpaShift) | pageXY.x);
            } else {
                fallbackPage = vsmVirtualPage(fallbackTexel, projection, push.pageSize);
            }
            uint mappedPhysical, mappedLevel;
            if (vsmDirectionalRayMapping(float2(projection.rowX.w, projection.rowY.w), fallbackPage, fallbackP, fallbackTexel,
                    projections, pages, push, level, mappedPhysical, mappedLevel, useGroupCache)) {
                if (mappedLevel == push.clipmapCount) { break; }
                return vsmSamplePhysicalPage(mappedPhysical, mappedLevel, fallbackP, fallbackTexel, rayBase, atlas, push);
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
    if (!rayBase.coarseValid) {
        ++counts.z;
        return result;
    }
    result.virtualTexel = vsmVirtualTexel(result.projected, coarse, push.pageSize);
    ++counts.y;
    uint2 finalAtlasTexel;
    if (rayBase.powerOfTwoLayout) {
        finalAtlasTexel = (rayBase.coarsePhysicalCoord << rayBase.pageSizeShift) | (result.virtualTexel & rayBase.pageSizeMask);
    } else {
        finalAtlasTexel = vsmAtlasTexelFromVirtual(rayBase.coarsePhysicalCoord, result.virtualTexel, push.pageSize);
    }
    result.storedDepth = atlas.Load(int3(finalAtlasTexel, 0));
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    result.valid = true;
    return result;
}

ShadowLookup vsmDirectionalRayLookup(float time, float3 stepOffset, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
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
            uint2 pageCoord;
            uint2 localTexel;
            uint pageOffset;
            bool isPow2 = rayBase.powerOfTwoLayout;
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
            uint page = rayBase.receiverPageOffset + pageOffset;
            vsmWorkAdd(push, kVSMWorkProjectionMappingLoads, 1u);
            uint2 physical;
            if (page == cache.lastPage && cache.lastPageHit) {
                physical = cache.lastPhysicalCoord;
            } else {
                uint4 mapping = pages[rayBase.receiverMappingOffset + pageOffset];
                if (mapping.w == 1u) {
                    if (isPow2) {
                        physical = uint2(mapping.x & rayBase.physicalPagesXMask, mapping.x >> rayBase.physicalPagesXShift);
                    } else {
                        physical = vsmPhysicalPageCoord(mapping.x, push.physicalPagesX);
                    }
                    cache.lastPhysicalCoord = physical;
                    cache.lastPage = page;
                    cache.lastPageHit = true;
                } else {
                    cache.lastPage = page;
                    cache.lastPageHit = false;
                    return vsmDirectionalRayLookupSlow(p, texel, page, mapping, true, time, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
                }
            }
            uint2 atlasTexel;
            if (isPow2) {
                atlasTexel = (physical << pageSizeShift) | localTexel;
            } else {
                atlasTexel = physical * push.pageSize + localTexel;
            }
            float storedDepth = atlas.Load(int3(atlasTexel, 0));
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
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, time * ray.step, ray, rayBase, projections, pages, atlas, push, counts, cache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRay ray, VSMDirectionalRayBase rayBase,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
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
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    return vsmDirectionalRayLookup(time, ray, rayBase, projections, pages, atlas, push, counts, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRayState ray,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, inout VSMDirectionalRayCache cache, bool useGroupCache = false)
{
    return vsmDirectionalRayLookup(time, ray.ray, ray.base, projections, pages, atlas, push, counts, cache, useGroupCache);
}

ShadowLookup vsmDirectionalRayLookup(float time, VSMDirectionalRayState ray,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages,
    Texture2D<float> atlas, ShadowPush push, inout uint3 counts, bool useGroupCache = false)
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
#endif

float vsmRayDepth(float depth, ShadowProjection projection, uint2 virtualTexel, float3 receiver, uint pageSize)
{
#if VSM_DIRECTIONAL_PROJECTION
    return (depth - dot(projection.rowZ, float4(receiver, 1.0))) / length(projection.rowZ.xyz);
#else
    if (projection.metric.w < 0.5) {
        return (depth - dot(projection.rowZ, float4(receiver, 1.0))) / length(projection.rowZ.xyz);
    }
    float2 ndc = (float2(virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 texelRay = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * ndc.x + normalize(projection.rowY.xyz) * ndc.y);
    float scale = dot(projection.rowZ.xyz, forward);
    float offset = dot(projection.rowZ, float4(projection.originNear.xyz, 1.0));
    // Inverse radial depth gives every cube face the same continuous history
    // coordinate. Larger values, as for directional depth, are toward the light.
    return (depth - scale) / (offset * length(texelRay));
#endif
}

float vsmFilterVisibility(float3 position, float3 normal, uint receiverLevel, uint2 pixel,
    StructuredBuffer<ShadowProjection> projections, ShadowPush push, inout uint3 filterCounts, bool useGroupCache = false)
{
    ShadowProjection receiverProj = vsmGetCachedProjection(projections, receiverLevel, useGroupCache);
    if (push.filterMode == 1u) {
        float sum = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 tap = vsmPCFPosition(position, normal, receiverProj, push, int2(x, y));
                ShadowLookup sample = vsmFilterLookup(tap, push, receiverLevel, filterCounts);
                float compare = sample.valid ? vsmComparisonDepth(tap, normal, sample.projected,
                    sample.virtualTexel, vsmLookupProjection(sample), push.pageSize) : 1.0;
                sum += !sample.valid || sample.storedDepth == 0.0 || compare >= sample.storedDepth ? 1.0 : 0.0;
            }
        }
        return sum / 9.0;
    }
    float visibility = 0.0;
    float stepRandom = vsmSampleRandom(pixel, 2u, push.samplePhase);
#if VSM_ADAPTIVE_RAY_COUNT
    uint tracedRays = push.rayCount;
#endif
    // A ray keeps its receiver's resolution. An angular footprint shrinks to
    // zero at contact and would request entire fine clipmaps, starving the
    // receiver pages and forcing nearly all samples onto the coarse map.
    ShadowProjection proj0 = vsmGetCachedProjection(projections, 0u, useGroupCache);
    float3 toLight;
    float3 right;
    float3 up;
    float tanAngularRadius;
#if !VSM_ONE_PASS_PROJECTION
    if (useGroupCache && push.lightType == 0u) {
        toLight = gVsmDirectionalUniform.toLight;
        right = gVsmDirectionalUniform.right;
        up = gVsmDirectionalUniform.up;
        tanAngularRadius = gVsmDirectionalUniform.tanAngularRadius;
    } else
#endif
    {
        toLight = push.lightType == 0u ? normalize(proj0.rowZ.xyz) :
            normalize(proj0.originNear.xyz - position);
        right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
        up = cross(toLight, right);
        tanAngularRadius = tan(push.angularRadius);
    }
    float receiverTexelSize = vsmWorldTexelSize(position, receiverProj, push.pageSize, push.lightType);
#if VSM_DIRECTIONAL_PROJECTION
    float receiverExtent = receiverProj.metric.x;
    float3 normX, normY;
    float biasZ;
    float receiverInvResolution, coarseInvResolution;
#if !VSM_ONE_PASS_PROJECTION
    if (useGroupCache && push.lightType == 0u) {
        normX = gVsmDirectionalUniform.normX;
        normY = gVsmDirectionalUniform.normY;
        biasZ = gVsmDirectionalUniform.biasZ;
        coarseInvResolution = gVsmDirectionalUniform.coarseInvResolution;
        receiverInvResolution = gVsmDirectionalUniform.receiverInvResolution;
    } else
#endif
    {
        normX = normalize(proj0.rowX.xyz);
        normY = normalize(proj0.rowY.xyz);
        biasZ = proj0.metric.z;
        receiverInvResolution = 2.0 / float(push.virtualPagesPerAxis * push.pageSize);
        coarseInvResolution = 2.0 / float(push.pageSize);
    }
    float denominator = dot(normal, toLight);
    float invDenom = abs(denominator) > 1e-5 ? length(proj0.rowZ.xyz) / denominator : 0.0;
    float2 normalSlope = float2(dot(normal, normX), dot(normal, normY)) * invDenom;
    float2 receiverScaledSlope = normalSlope * receiverExtent;
    float invSamplesPerRay = 1.0 / float(push.samplesPerRay);
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    ShadowProjection coarseProj = vsmGetCachedProjection(projections, push.clipmapCount, useGroupCache);
    VSMDirectionalRayBase rayBase = vsmCreateDirectionalRayBase(position, receiverProj, coarseProj, pages, push, receiverLevel, useGroupCache);
    VSMDirectionalRayCache rayCache;
    rayCache.lastPage = 0xffffffffu;
    rayCache.lastPhysicalCoord = uint2(0u);
    rayCache.lastPageHit = false;
#endif
    for (uint ray = 0u; ray < push.rayCount; ++ray) {
        vsmWorkAdd(push, kVSMWorkProjectionRays, 1u);
        float2 disk = vsmDiskSample(ray, push.rayCount, pixel, push.samplePhase);
        float3 rayOffset = right * disk.x + up * disk.y;
        float3 endpoint = (push.lightType == 0u) ?
            vsmRayEndpointDirectional(position, toLight, rayOffset, tanAngularRadius, push.rayLength) :
            vsmRayEndpoint(position, proj0, push, toLight, rayOffset);
        float2 ditherOffset = float2(vsmSampleRandom(pixel, 3u + 2u * ray, push.samplePhase),
            vsmSampleRandom(pixel, 4u + 2u * ray, push.samplePhase)) - 0.5;
        float3 dither = (right * ditherOffset.x + up * ditherOffset.y) * receiverTexelSize;
#if VSM_DIRECTIONAL_PROJECTION
        VSMDirectionalRay dirRay;
        float3 direction = endpoint - position;
        dirRay.lookupOrigin = vsmProjectPosition(position + dither, receiverProj);
        dirRay.step = float3(dot(receiverProj.rowX.xyz, direction), dot(receiverProj.rowY.xyz, direction), dot(receiverProj.rowZ.xyz, direction));
        dirRay.coarseLookupOrigin = vsmProjectPosition(position + dither, coarseProj).xy;
        dirRay.coarseStep = float2(dot(coarseProj.rowX.xyz, direction), dot(coarseProj.rowY.xyz, direction));
#endif
        float maxSlope = push.lightType == 0u ? 0.05 : 0.05 * abs(
            1.0 / length(endpoint - proj0.originNear.xyz) -
            1.0 / length(position - proj0.originNear.xyz));
        float historyDepth = 0.0, historyTime = 0.0, historySlope = 0.0, previousReference = 0.0;
        bool hasHistory = false;
        bool blocked = false;
        // Trace back toward the receiver, including t=0. Depth discontinuities
        // can expose only the front layer: continue the preceding surface with
        // bounded slope through that gap instead of inventing an intersection.
#if VSM_DIRECTIONAL_PROJECTION
        for (uint step = 0u; step <= push.samplesPerRay; ++step) {
            float time = vsmRaySampleTime(step, push, stepRandom);
            float3 stepOffset = time * dirRay.step;
            ShadowLookup sample = vsmDirectionalRayLookup(time, stepOffset, dirRay, rayBase, projections, pages, atlas, push, filterCounts, rayCache, useGroupCache);
            if (!sample.valid) { continue; }
            float compare;
            float reference;
            if (sample.level == rayBase.receiverLevel) {
                float2 sampleNDC = fma(float2(sample.virtualTexel), float2(receiverInvResolution), float2(0.5 * receiverInvResolution - 1.0));
                float2 offset = sampleNDC - (rayBase.comparisonOrigin.xy + stepOffset.xy);
                float correction = -dot(receiverScaledSlope, offset);
                float relDepth = stepOffset.z + correction + biasZ;
                compare = rayBase.comparisonOrigin.z + relDepth;
                reference = relDepth * rayBase.invDepthScale;
            } else {
                ShadowProjection projection = vsmGetCachedProjection(projections, sample.level, useGroupCache);
                float3 projected = vsmDirectionalRayPosition(dirRay, rayBase, time, sample.level, projection, false);
                float extent = projection.metric.x;
                float invResolution = (sample.level == push.clipmapCount) ? coarseInvResolution : receiverInvResolution;
                compare = vsmDirectionalComparisonDepth(projected, sample.virtualTexel,
                    extent, invResolution, normalSlope, biasZ);
                reference = vsmDirectionalRayDepth(compare, rayBase);
            }
            float diff = (sample.storedDepth - compare) * rayBase.invDepthScale;
            if (!hasHistory) {
                blocked = diff > 0.0;
                if (blocked) { break; }
                historyDepth = reference + diff;
                historyTime = time;
                hasHistory = true;
            } else {
                float tolerance = abs(reference - previousReference) * 1.05;
                if (diff <= tolerance) {
                    blocked = diff > 0.0;
                    float depth = reference + diff;
                    if (depth != historyDepth) {
                        float timeDelta = time - historyTime;
                        historySlope = clamp((depth - historyDepth) / timeDelta, -maxSlope, maxSlope);
                        historyDepth = depth;
                        historyTime = time;
                    }
                } else {
                    float timeDelta = time - historyTime;
                    float predictedSeparation = (historyDepth + historySlope * timeDelta) - reference;
                    blocked = predictedSeparation > 0.0 && predictedSeparation <= tolerance;
                }
            }
            if (blocked) { break; }
            previousReference = reference;
        }
#else
        for (uint step = 0u; step <= push.samplesPerRay; ++step) {
            float time = vsmRaySampleTime(step, push, stepRandom);
            float3 point = vsmRayPosition(position, endpoint, time, proj0, push);
            ShadowLookup sample = vsmFilterLookup(point + dither, push, receiverLevel, filterCounts);
            if (!sample.valid) { continue; }
            ShadowProjection projection = vsmLookupProjection(sample);
            float compare = vsmComparisonDepth(point, normal, vsmProjectPosition(point, projection),
                sample.virtualTexel, projection, push.pageSize);
            float reference = vsmRayDepth(compare, projection, sample.virtualTexel, position, push.pageSize);
            float depth = push.lightType != 0u && sample.storedDepth == 0.0 ? 0.0 :
                vsmRayDepth(sample.storedDepth, projection, sample.virtualTexel, position, push.pageSize);
            if (!hasHistory) {
                // This ordinary comparison also retains blockers beyond the
                // bounded search; there is no separate hard-shadow tail.
                blocked = depth > reference;
                historyDepth = depth; historyTime = time;
                hasHistory = true;
            } else {
                float tolerance = abs(reference - previousReference) * 1.05;
                float reconstructed = depth;
                if (depth - reference > tolerance) {
                    reconstructed = historyDepth + historySlope * (time - historyTime);
                } else if (depth != historyDepth) {
                    historySlope = clamp((depth - historyDepth) / (time - historyTime), -maxSlope, maxSlope);
                    historyDepth = depth; historyTime = time;
                }
                float separation = reconstructed - reference;
                blocked = separation > 0.0 && separation <= tolerance;
            }
            if (blocked) { break; }
            previousReference = reference;
        }
#endif
        visibility += blocked ? 0.0 : 1.0;
#if VSM_ADAPTIVE_RAY_COUNT
        if (ray + 1u < push.rayCount) {
            // Vote only among this light's active receivers. This heuristic
            // assumes full light after one miss, or umbra after two hits in
            // every participating lane; it is not the fixed-ray estimator.
            bool keepTracing = ray == 0u ? blocked : (visibility > 0.0);
            if (!WaveActiveAnyTrue(keepTracing)) {
                tracedRays = ray + 1u;
                vsmWorkAdd(push, ray == 0u ? kVSMWorkProjectionAdaptiveLitReceivers :
                    kVSMWorkProjectionAdaptiveUmbraReceivers, 1u);
                vsmWorkAdd(push, kVSMWorkProjectionAdaptiveSavedRays, push.rayCount - tracedRays);
                break;
            }
        }
#endif
    }
#if VSM_ADAPTIVE_RAY_COUNT
    return visibility / float(tracedRays);
#else
    return visibility / float(push.rayCount);
#endif
}

// Read-before-write filters must remain atomic with other lanes' updates.
groupshared Atomic<uint> gVsmMarkedPages[256];

void vsmMarkPage(RWStructuredBuffer<uint4> pages, uint page, inout uint3 workCounts)
{
    // A cache hit reuses an inserting lane's eventual OR. Hash collisions only
    // cause repeated requests; allocation observes this dispatch after all ORs.
    uint slot = (page * 2654435761u) >> 24u;
    if (gVsmMarkedPages[slot].load(MemoryOrder.Relaxed) == page) { return; }
    uint previous = gVsmMarkedPages[slot].exchange(page, MemoryOrder.Relaxed);
    if (previous == page) { return; }
    InterlockedOr(pages[page].y, 1u);
    ++workCounts.z;
}

void vsmMarkPosition(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel, inout uint3 workCounts)
{
    ++workCounts.x;
    // Hard and the nine-tap PCF kernel keep the short direct path. Cache setup
    // pays off for the longer SMRT traces handled by the overload below.
    if (push.lightType == 2u && virtualShadowMapPointFace(position - projections[0].originNear.xyz) != push.projectionFace) { return; }
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

#if VSM_STATIC_PAGE_HZB
// Support points give the extrema of an orthographic row over the entire world AABB.
// Bound dot/clip rounding by operand magnitude even when translation nearly cancels.
float2 vsmHZBProjectedRange(float3 minimum, float3 maximum, float4 row)
{
    float low = dot(row, float4(select(row.xyz >= 0.0, minimum, maximum), 1.0));
    float high = dot(row, float4(select(row.xyz >= 0.0, maximum, minimum), 1.0));
    float error = (dot(abs(row.xyz), max(abs(minimum), abs(maximum))) + abs(row.w) + 1.0) *
        (256.0 * 5.960464477539063e-8);
    return float2(low - error, high + error);
}

bool vsmStaticPageHidesBounds(ShadowPush push, ShadowProjection projection, uint2 page,
    uint physicalPage, float3 minimum, float3 maximum)
{
    // Match fine Raster's canonical page translation, including after clipmap scroll.
    float axis = float(projection.page.y);
    float halfPage = 1.0 / axis;
    if (projection.page.w == 0u) {
        float2 center = projection.originNear.xy + float2(page) + 0.5 - 0.5 * axis;
        projection.rowX.w = -center.x * (2.0 * halfPage);
        projection.rowY.w = -center.y * (2.0 * halfPage);
    }
    float2 x = vsmHZBProjectedRange(minimum, maximum, projection.rowX);
    float2 y = vsmHZBProjectedRange(minimum, maximum, projection.rowY);
    float closestDepth = vsmHZBProjectedRange(minimum, maximum, projection.rowZ).y;
    float scale = float(push.pageSize) / (2.0 * halfPage);
    float2 lo = (float2(x.x, y.x) + halfPage) * scale - 1.0;
    float2 hi = (float2(x.y, y.y) + halfPage) * scale + 1.0;
    if (!all(isfinite(lo)) || !all(isfinite(hi)) || !isfinite(closestDepth)) { return false; }
    float last = float(push.pageSize - 1u);
    return vsmStaticHZBOccludes(push, physicalPage, uint2(clamp(lo, 0.0, last)),
        uint2(clamp(hi, 0.0, last)), closestDepth);
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmResetMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
    if (id.x == 0u) {
        RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
        work[0] = uint4(0u);
    }
#endif
#if VSM_STATIC_PAGE_HZB
    if (id.x == 0u) {
        RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
        jobs[0] = 0u; jobs[1] = 1u; jobs[2] = 1u;
    }
#endif
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    if (id.x < push.totalPages) {
        bool coarse = id.x == push.totalPages - 1u;
#if VSM_RECEIVER_MASK
        uint2 validReceiverMask = uint2(0u);
#endif
        uint4 entry = uint4(coarse ? 0u : 0xffffffffu, coarse && push.enabled != 0u ? 3u : 0u,
            coarse ? push.clipmapCount : id.x / (push.virtualPagesPerAxis * push.virtualPagesPerAxis),
            2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u));
        uint mappingFlags = 0u;
        if ((push.cacheMode & 1u) != 0u) {
            // Fine levels occupy fixed VPA^2 strides; local mips only use the
            // packed axis^2 prefix. The remaining slots must stay unmapped.
            StructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.reuse));
            uint4 change = uint4(0u);
            uint2 page = uint2(0u);
            uint oldIndex = coarse ? id.x : 0xffffffffu;
            if (!coarse) {
                uint stride = push.virtualPagesPerAxis * push.virtualPagesPerAxis;
                uint level = id.x / stride, local = id.x % stride;
                uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
                change = changes[level];
                if (local < axis * axis) {
                    page = uint2(local % axis, local / axis);
                    int2 oldPage = int2(page) + asint(change.xy);
                    if (all(uint2(oldPage) < uint2(axis))) {
                        oldIndex = level * stride + uint(oldPage.y) * axis + uint(oldPage.x);
                    }
                }
            }
            if (oldIndex != 0xffffffffu) {
                StructuredBuffer<uint4> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.previousPages));
                uint4 old = previous[oldIndex];
                bool mappingDirty = (push.cacheMode & 4u) != 0u || (coarse && push.dirtyBoundsCount != 0u);
                // Only resident pages need rectangle tests. Count consumes these
                // flags before replacing the same scratch words with scan ranks.
                if (old.x != 0xffffffffu) {
                    uint staticFlags = 0u;
                    if ((push.cacheMode & 16u) != 0u) {
                        bool staticDirty = (push.cacheMode & 4u) != 0u || (coarse && vsmCacheMetadata(push, 35u) != 0u);
                        if (!coarse) {
                            uint4 staticChange = changes[vsmCacheMetadata(push, 36u) + entry.z];
                            for (uint rectangle = 0u; !staticDirty && rectangle < staticChange.w; ++rectangle) {
                                uint4 bounds = changes[staticChange.z + rectangle];
                                staticDirty = all(page >= bounds.xy) && all(page <= bounds.zw);
                            }
                        }
                        vsmCacheAdd(push, kVSMCacheStaticInvalidatedResidentPages, staticDirty ? 1u : 0u);
                        staticFlags = staticDirty || (old.w & 4u) == 0u ? 8u : 4u;
                    }
#if VSM_STATIC_PAGE_HZB && VSM_WORKLOAD_DIAGNOSTICS
                    bool hadOverlap = false;
#endif
#if VSM_STATIC_PAGE_HZB
                    StructuredBuffer<ShadowProjection> projections = resolveUniform(
                        ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
                    if (coarse && mappingDirty && staticFlags == 4u && (old.w & 1u) != 0u && push.dirtyBoundsCount <= 32u) {
                        // The coarse page has no per-level rect list. Its stable
                        // projection covers the whole scene. Only a proof for
                        // every old/new dirty AABB permits keeping its depth.
                        uint base = vsmCacheMetadata(push, 39u);
                        mappingDirty = false;
                        for (uint b = 0u; b < push.dirtyBoundsCount && !mappingDirty; ++b) {
                            uint4 lo = changes[base + b * 2u];
                            mappingDirty = lo.w == 0u || !vsmStaticPageHidesBounds(push, projections[entry.z], page, old.x,
                                asfloat(lo.xyz), asfloat(changes[base + b * 2u + 1u].xyz));
                        }
#if VSM_WORKLOAD_DIAGNOSTICS
                        hadOverlap = true;
#endif
                    }
#endif
                    for (uint rectangle = 0u; !mappingDirty && rectangle < change.w; ++rectangle) {
                        uint4 bounds = changes[change.z + rectangle];
#if VSM_STATIC_PAGE_HZB
                        mappingDirty = all(page >= bounds.xy) && all(page <= uint2(bounds.z & 0xffffu, bounds.z >> 16u));
                        if (mappingDirty && staticFlags == 4u && (old.w & 1u) != 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                            hadOverlap = true;
#endif
                            uint index = vsmCacheMetadata(push, 39u) + 2u * bounds.w;
                            mappingDirty = !vsmStaticPageHidesBounds(push, projections[entry.z], page, old.x,
                                asfloat(changes[index].xyz), asfloat(changes[index + 1u].xyz));
                        }
#else
                        mappingDirty = all(page >= bounds.xy) && all(page <= bounds.zw);
#endif
                    }
                    if ((push.cacheMode & 32u) != 0u) {
                        bool transitioned = coarse;
                        if (!coarse) {
                            uint4 merge = changes[vsmCacheMetadata(push, kVSMCacheMergeRemap) + entry.z];
                            for (uint rectangle = 0u; !transitioned && rectangle < merge.w; ++rectangle) {
                                uint4 bounds = changes[merge.z + rectangle];
                                transitioned = all(page >= bounds.xy) && all(page <= bounds.zw);
                            }
                        }
                        if (transitioned) {
                            // The rasterized caster has not changed. Its depth was already in
                            // the composite atlas. Independent invalidations still require restore.
                            if (staticFlags == 4u) { staticFlags = 8u | kVSMPageMergeStatic; }
                            else { mappingDirty = true; }
                        }
                    }
                    entry.x = old.x;
                    // A deferred or overflowing static merge must recover from all static
                    // geometry before this slot can be restored into the composite again.
                    bool dirty = mappingDirty || (old.w & 1u) == 0u ||
                        ((push.cacheMode & 16u) != 0u && (old.w & 4u) == 0u);
#if VSM_RECEIVER_MASK
                    if (!dirty && !coarse) {
                        ByteAddressBuffer previousCoverage = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.previousPages));
                        validReceiverMask = previousCoverage.Load2(vsmValidReceiverMaskByteOffset(push, oldIndex));
                        // Full redraw on a merge would publish complete page depth. Do not
                        // upgrade a partially covered composite to full without rasterizing it.
                        if ((staticFlags & kVSMPageMergeStatic) != 0u && any(validReceiverMask != uint2(0xffffffffu))) {
                            dirty = true;
                            mappingDirty = true;
                            validReceiverMask = uint2(0u);
                        }
                    }
#endif
#if VSM_STATIC_PAGE_HZB && VSM_WORKLOAD_DIAGNOSTICS
                    vsmCacheAdd(push, kVSMCacheHZBSavedInvalidations, hadOverlap && !mappingDirty ? 1u : 0u);
#endif
                    mappingFlags = (mappingDirty ? 1u : 0u) | (oldIndex != id.x ? 2u : 0u);
                    entry.w = (dirty ? 2u : 1u) | staticFlags;
                    vsmCacheAdd(push, kVSMCacheCompositeInvalidatedResidentPages, mappingDirty ? 1u : 0u);
                }
            }
        }
        pages[id.x] = entry;
#if VSM_RECEIVER_MASK
        RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
        validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, id.x), validReceiverMask);
#endif
        owners[push.physicalPageCount + id.x] = mappingFlags;
    }
    // Each counter has exactly one writer, including the diagnostic constants.
    if (id.x < 32u) {
        uint value = id.x == 10u ? push.triangleCount : id.x == 11u ? push.clipmapCount :
            id.x == 12u ? push.physicalPageCount : id.x == 14u ? 1u - push.enabled :
            id.x == 22u ? ((push.cacheMode & 8u) != 0u ? 1u : 0u) :
            id.x == 23u ? push.cacheEpoch : id.x == 25u ? push.dirtyBoundsCount :
            id.x == 26u ? ((push.cacheMode & 4u) != 0u ? 1u : 0u) :
            id.x == 27u ? ((push.cacheMode & 2u) != 0u ? 1u : 0u) :
            id.x == 28u ? push.cancelledCount : 0u;
        statistics[id.x] = value;
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (id.x < (kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / 4u) { statistics[32u + id.x] = 0u; }
#endif
    if (id.x < push.physicalPageCount) { owners[id.x] = 0u; }
#if VSM_CLUSTER_PAGE_BITSET
    if (id.x < (push.totalPages + 31u) / 32u) { owners[vsmClusterPageBitsetOffset(push) + id.x] = 0u; }
#endif
#if VSM_RECEIVER_MASK
    if (id.x < push.totalPages * 2u) { owners[vsmReceiverMaskOffset(push) + id.x] = 0u; }
#endif
#if VSM_DIRTY_PAGE_HIERARCHY
    vsmResetPageHierarchy(id.x, push, owners);
#endif
#if VSM_COMPACT_BUILD
    uint argumentBatches = push.batchCount + push.coarseBatchCount;
    if (id.x == 0u) { owners[vsmCompactCandidateOffset(push)] = 0u; }
    argumentBatches += vsmCompactBatchCount(push);
    if (id.x < argumentBatches * 3u) {
#else
    if (id.x < (push.batchCount + push.coarseBatchCount) * 3u) {
#endif
        arguments[id.x] = id.x % 3u == 0u ? 0u : 1u;
    }
}

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

// The final group barrier completes all mask contributions before flush.
groupshared Atomic<uint> gVsmReceiverPages[256];
struct VSMReceiverMaskAtomic
{
    Atomic<uint> x;
    Atomic<uint> y;
};
groupshared VSMReceiverMaskAtomic gVsmReceiverMasks[256];

bool vsmAccumulateReceiverMask(uint page, uint2 mask)
{
    if ((mask.x | mask.y) == 0u) { return true; }
    uint slot = (page * 2654435761u) >> 24u;
    uint current = gVsmReceiverPages[slot].load(MemoryOrder.Relaxed);
    if (current != page) {
        if (current != 0xffffffffu) { return false; }
        uint previous = gVsmReceiverPages[slot].compareExchange(0xffffffffu, page, MemoryOrder.Relaxed, MemoryOrder.Relaxed);
        if (previous != 0xffffffffu && previous != page) { return false; }
    }
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

float2 vsmDirectionalFootprintRange(float4 row, float3 position, VSMDirectionalFootprintContext ctx)
{
    float center = dot(row, float4(position, 1.0));
    float axial = ctx.rayLength * dot(row.xyz, ctx.toLight);
    float transverse = ctx.coneRadius * length(row.xyz);
    float dither = 0.5 * ctx.texelSize * (abs(dot(row.xyz, ctx.right)) + abs(dot(row.xyz, ctx.up)));
    // Include the original endpoint, world-space lerp and texel dither paths.
    // The axial term also retains finite-precision XY/light-axis nonorthogonality.
    float magnitude = dot(abs(row.xyz), ctx.commonPos) + abs(row.w) + 1.0;
    float error = magnitude * (256.0 * 5.960464477539063e-8);
    float spread = transverse + dither + error;
    return float2(center + min(0.0, axial) - spread, center + max(0.0, axial) + spread);
}

void vsmMarkDirectionalFootprint(float3 position, StructuredBuffer<ShadowProjection> projections,
    RWStructuredBuffer<uint4> pages, ShadowPush push, uint firstLevel,
    float3 toLight, float3 right, float3 up, float sinAngularRadius, inout uint3 workCounts)
{
    ++workCounts.x;
    // The receiver-selected texel sets dither for every sampled clipmap.
    float texelSize = projections[firstLevel].metric.y;
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

    for (uint level = firstLevel; level < push.clipmapCount; ++level) {
        ++workCounts.y;
        ShadowProjection projection = projections[level];
        float2 xRange = vsmDirectionalFootprintRange(projection.rowX, position, ctx);
        float2 yRange = vsmDirectionalFootprintRange(projection.rowY, position, ctx);
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
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
        uint subShift = uint(VSM_MARK_PAGE_SHIFT - 3);
        uint subMask = push.pageSize - 1u;
#endif
#endif
        for (uint y = firstPage.y; y <= lastPage.y; ++y) {
            uint rowBasePage = projection.page.x + y * projection.page.y;
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
            uint firstSubY = (y == firstPage.y) ? ((firstTexel.y & subMask) >> subShift) : 0u;
            uint lastSubY = (y == lastPage.y) ? ((lastTexel.y & subMask) >> subShift) : 7u;
#endif
#endif
            for (uint x = firstPage.x; x <= lastPage.x; ++x) {
                uint page = rowBasePage + x;
#if VSM_RECEIVER_MASK
#if VSM_MARK_PAGE_SHIFT >= 3
                uint firstSubX = (x == firstPage.x) ? ((firstTexel.x & subMask) >> subShift) : 0u;
                uint lastSubX = (x == lastPage.x) ? ((lastTexel.x & subMask) >> subShift) : 7u;
                uint2 mask = (firstSubX == 0u && lastSubX == 7u && firstSubY == 0u && lastSubY == 7u) ?
                    uint2(0xffffffffu, 0xffffffffu) :
                    vsmReceiverRectangleMask(uint2(firstSubX, firstSubY), uint2(lastSubX, lastSubY));
#else
                // Page deduplication must not discard another receiver's subtiles.
                uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel, uint2(x, y), push.pageSize);
#endif
                if (!vsmAccumulateReceiverMask(page, mask)) { vsmWriteReceiverMask(push, page, mask); }
#endif
                vsmMarkPage(pages, page, workCounts);
            }
        }
    }
}
#endif

#if VSM_DIRECTIONAL_MARK
struct VSMMarkDirectionalUniformData
{
    float3 toLight;
    float3 right;
    float3 up;
    float sinAngularRadius;
    float sinAlpha;
};
groupshared VSMMarkDirectionalUniformData gVsmMarkDirectionalUniform;
#endif
groupshared VSMReceiverBasisGroup gVsmMarkReceiverBasis;

VSMReceiverBasis vsmGetMarkGroupReceiverBasis()
{
    VSMReceiverBasis b;
    b.eye = gVsmMarkReceiverBasis.eye;
    b.forward = gVsmMarkReceiverBasis.forward;
    b.right = gVsmMarkReceiverBasis.right;
    b.up = gVsmMarkReceiverBasis.up;
    b.rightAspect = gVsmMarkReceiverBasis.rightAspect;
    b.tanHalfFov = gVsmMarkReceiverBasis.tanHalfFov;
    b.orthoHalfHeight = gVsmMarkReceiverBasis.orthoHalfHeight;
    b.jitter = gVsmMarkReceiverBasis.jitter;
    b.invViewport = gVsmMarkReceiverBasis.invViewport;
    b.ndcScale = gVsmMarkReceiverBasis.ndcScale;
    b.viewportSize = gVsmMarkReceiverBasis.viewportSize;
    b.nearPlane = gVsmMarkReceiverBasis.nearPlane;
    b.farPlane = gVsmMarkReceiverBasis.farPlane;
    b.farMinusNear = gVsmMarkReceiverBasis.farMinusNear;
    b.nearTimesFar = gVsmMarkReceiverBasis.nearTimesFar;
    b.isOrtho = gVsmMarkReceiverBasis.isOrtho != 0u;
    b.reversed = gVsmMarkReceiverBasis.reversed != 0u;
    return b;
}

void vsmMarkReceiver(uint3 id, ShadowPush push, out uint3 workCounts)
{
    workCounts = uint3(0u);
    if (push.enabled == 0u) { return; }
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    VSMReceiverBasis basis = vsmGetMarkGroupReceiverBasis();
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
    vsmWorkAdd(push, kVSMWorkMarkReceiverDispatches, 1u);
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    // Resolution belongs to the receiver, independently of the filter kernel.
    float3 normal = float3(0.0);
    if (push.receiverFootprint != 0u || push.filterMode == 1u || push.lightType == 0u) {
        normal = vsmReceiverNormal(int2(id.xy), basis, depth, position, linearDepth);
    }
    float3 toLight = float3(0.0);
    float sinAngularRadius = 0.0;
    if (push.lightType == 0u) {
#if VSM_DIRECTIONAL_MARK
        toLight = gVsmMarkDirectionalUniform.toLight;
        sinAngularRadius = gVsmMarkDirectionalUniform.sinAngularRadius;
        float sinAlpha = gVsmMarkDirectionalUniform.sinAlpha;
#else
        toLight = normalize(projections[0].rowZ.xyz);
        sinAngularRadius = sin(push.angularRadius);
        float sinAlpha = max(sinAngularRadius, 0.1);
#endif
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
        float3 right = gVsmMarkDirectionalUniform.right;
        float3 up = gVsmMarkDirectionalUniform.up;
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
        for (uint ray = 0u; ray < push.rayCount; ++ray) {
#if VSM_LOCAL_MARK
            float2 disk = vsmDiskSample(ray, push.rayCount, scramble);
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
                float t = 1.0 - (float(step + 1u) - stepRandom) / float(push.samplesPerRay);
                float time = step == push.samplesPerRay ? 0.0 : t * t;
                if (push.lightType != 0u) { time = time * startW / lerp(endW, startW, time); }
                float3 point = lerp(position, endpoint, time);
#else
                float time = vsmRaySampleTime(step, push, id.xy);
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
    for (uint entry = lane; entry < 256u; entry += 64u) {
        gVsmMarkedPages[entry].store(0xffffffffu, MemoryOrder.Relaxed);
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_MASK
        gVsmReceiverPages[entry].store(0xffffffffu, MemoryOrder.Relaxed);
        gVsmReceiverMasks[entry].x.store(0u, MemoryOrder.Relaxed);
        gVsmReceiverMasks[entry].y.store(0u, MemoryOrder.Relaxed);
#endif
    }
    if (lane == 0u) {
        StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
        ViewConstants view = views[0];
        VSMReceiverBasis b = vsmMakeReceiverBasis(view);
        gVsmMarkReceiverBasis.eye = b.eye;
        gVsmMarkReceiverBasis.forward = b.forward;
        gVsmMarkReceiverBasis.right = b.right;
        gVsmMarkReceiverBasis.up = b.up;
        gVsmMarkReceiverBasis.rightAspect = b.rightAspect;
        gVsmMarkReceiverBasis.tanHalfFov = b.tanHalfFov;
        gVsmMarkReceiverBasis.orthoHalfHeight = b.orthoHalfHeight;
        gVsmMarkReceiverBasis.jitter = b.jitter;
        gVsmMarkReceiverBasis.invViewport = b.invViewport;
        gVsmMarkReceiverBasis.ndcScale = b.ndcScale;
        gVsmMarkReceiverBasis.viewportSize = b.viewportSize;
        gVsmMarkReceiverBasis.nearPlane = b.nearPlane;
        gVsmMarkReceiverBasis.farPlane = b.farPlane;
        gVsmMarkReceiverBasis.farMinusNear = b.farMinusNear;
        gVsmMarkReceiverBasis.nearTimesFar = b.nearTimesFar;
        gVsmMarkReceiverBasis.isOrtho = b.isOrtho ? 1u : 0u;
        gVsmMarkReceiverBasis.reversed = b.reversed ? 1u : 0u;

#if VSM_DIRECTIONAL_MARK
        if (push.lightType == 0u) {
            StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
            float3 toLight = normalize(projections[0].rowZ.xyz);
            float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
            float3 up = cross(toLight, right);
            float sinAngularRadius = sin(push.angularRadius);
            gVsmMarkDirectionalUniform.toLight = toLight;
            gVsmMarkDirectionalUniform.right = right;
            gVsmMarkDirectionalUniform.up = up;
            gVsmMarkDirectionalUniform.sinAngularRadius = sinAngularRadius;
            gVsmMarkDirectionalUniform.sinAlpha = max(sinAngularRadius, 0.1);
        }
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    uint3 counts;
    vsmMarkReceiver(id, push, counts);
#if VSM_DIRECTIONAL_MARK && VSM_RECEIVER_MASK
    // Receiver rejection and divergent page loops have completed on all lanes.
    GroupMemoryBarrierWithGroupSync();
    for (uint entry = lane; entry < 256u; entry += 64u) {
        uint page = gVsmReceiverPages[entry].load(MemoryOrder.Relaxed);
        if (page != 0xffffffffu) {
            uint2 mask = uint2(gVsmReceiverMasks[entry].x.load(MemoryOrder.Relaxed),
                gVsmReceiverMasks[entry].y.load(MemoryOrder.Relaxed));
            vsmWriteReceiverMask(push, page, mask);
        }
    }
#endif
    vsmWorkAdd(push, kVSMWorkMarkSamplePositions, counts.x);
    vsmWorkAdd(push, kVSMWorkMarkLevelTests, counts.y);
    vsmWorkAdd(push, kVSMWorkMarkPageRequests, counts.z);
}

// Allocation scratch follows the physical owner table in the same buffer:
// local page ranks, block counts/prefixes, then free/evictable physical slots.
// Every scan invocation participates, including rounded dispatch tails.
groupshared uint3 gVsmAllocationScan[256];
uint3 vsmAllocationPrefix(uint3 value, uint lane)
{
    gVsmAllocationScan[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1u; offset < 256u; offset <<= 1u) {
        uint3 previous = lane >= offset ? gVsmAllocationScan[lane - offset] : uint3(0u);
        GroupMemoryBarrierWithGroupSync();
        gVsmAllocationScan[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return gVsmAllocationScan[lane] - value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmCountMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        counts.x = id.x + 1u < push.totalPages && (entry.y & 1u) != 0u ? 1u : 0u;
        if ((push.cacheMode & 1u) != 0u && entry.x != 0xffffffffu) {
            uint mappingFlags = scratch[push.physicalPageCount + id.x];
            counts.y = mappingFlags & 1u;
            counts.z = (mappingFlags >> 1u) & 1u;
        }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { scratch[push.physicalPageCount + id.x] = prefix.x; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        scratch[push.physicalPageCount + push.totalPages + group.x] = total.x;
        InterlockedAdd(statistics[18], total.y);
        InterlockedAdd(statistics[19], total.z);
    }
}

// The established allocation entry now prefixes per-block demand counts;
// its one group replaces the old serial traversal of the full virtual table.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAllocateMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint first = lane * blocksPerLane, end = min(first + blocksPerLane, blockCount);
    uint offset = push.physicalPageCount + push.totalPages;
    uint count = 0u;
    for (uint block = first; block < end; ++block) { count += scratch[offset + block]; }
    uint prefix = vsmAllocationPrefix(uint3(count, 0u, 0u), lane).x;
    for (uint block = first; block < end; ++block) {
        uint value = scratch[offset + block];
        scratch[offset + block] = prefix;
        prefix += value;
    }
    if (lane == 0u) {
        uint requested = gVsmAllocationScan[255].x;
        uint allocated = min(requested, push.physicalPageCount - 1u);
        statistics[0] = requested;
        statistics[1] = allocated;
        statistics[2] = requested - allocated;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAdmitMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
    uint rankOffset = push.physicalPageCount;
    uint blockOffset = rankOffset + push.totalPages;
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        bool fine = id.x + 1u < push.totalPages;
        if (fine && (entry.y & 1u) != 0u) {
            uint rank = owners[blockOffset + group.x] + owners[rankOffset + id.x];
            // Keep requested parent coverage before spending the remaining
            // budget on refinement. A missing fine page can then use a nearby
            // coarser mip instead of the complete low-resolution scene map.
            rank = statistics[0] - 1u - rank;
            if (rank < push.physicalPageCount - 1u) {
                entry.y |= 2u;
#if VSM_RECEIVER_MASK
                // Reset carried coverage through the same oldIndex remap as
                // residency. Mark is complete before Admit: retain a clean page
                // only when its existing depth covers every requested subtile.
                if (entry.x != 0xffffffffu && (entry.w & 3u) == 1u) {
                    uint offset = vsmReceiverMaskOffset(push) + id.x * 2u;
                    uint2 requested = uint2(owners[offset], owners[offset + 1u]);
                    ByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.pages));
                    uint2 valid = validCoverage.Load2(vsmValidReceiverMaskByteOffset(push, id.x));
                    if (any((requested & ~valid) != uint2(0u))) {
                        entry.w = (entry.w & ~1u) | 2u;
                        InterlockedAdd(statistics[18], 1u);
                        vsmCacheAdd(push, kVSMCacheCompositeInvalidatedResidentPages, 1u);
                    }
                }
#endif
            }
            else { entry.x = 0xffffffffu; entry.w = 2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u); }
        }
        pages[id.x] = entry;
        if (entry.x != 0xffffffffu) { owners[entry.x] = id.x + 1u; }
        if (fine && (entry.y & 2u) != 0u) {
            counts.x = entry.x == 0xffffffffu ? 1u : 0u;
            counts.y = entry.x != 0xffffffffu && (entry.w & 2u) != 0u ? 1u : 0u;
            counts.z = entry.x != 0xffffffffu && (entry.w & 3u) == 1u ? 1u : 0u;
#if VSM_CLUSTER_PAGE_BITSET
            // Both raster layers consume the same Build list, including static-only merges.
            if (entry.x != 0xffffffffu && (entry.w & 10u) != 0u) {
                InterlockedOr(owners[vsmClusterPageBitsetOffset(push) + (id.x >> 5u)], 1u << (id.x & 31u));
            }
#endif
        }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { owners[rankOffset + id.x] = counts.x != 0u ? prefix.x : 0xffffffffu; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        owners[blockOffset + group.x] = total.x;
        InterlockedAdd(statistics[16], total.y);
        InterlockedAdd(statistics[17], total.z);
    }
}

uint2 vsmPhysicalHistory(ShadowPush push, uint previousBase, uint index)
{
    if ((push.cacheMode & 3u) == 3u) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousPages));
        return previous[previousBase + index];
    }
    return uint2(index, 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmPhysicalMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint firstBlock = lane * blocksPerLane, endBlock = min(firstBlock + blocksPerLane, blockCount);
    uint blockOffset = push.physicalPageCount + push.totalPages;
    uint candidatesOffset = blockOffset + blockCount;
    uint slotsPerLane = (push.physicalPageCount - 1u + 255u) / 256u;
    uint firstSlot = 1u + lane * slotsPerLane, endSlot = min(firstSlot + slotsPerLane, push.physicalPageCount);
    // Use the actual descriptor range: the pool may return an oversized
    // buffer, and the previous snapshot can have a different mapping layout.
    uint historySize = (push.physicalPageCount + 1u) & ~1u;
    uint previousBase = 0u, nextBase = 0u, frame = 1u;
    if ((push.cacheMode & 3u) == 3u) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousPages));
        uint size, stride;
        previous.GetDimensions(size, stride);
        previousBase = size - historySize;
        frame = previous[previousBase].y + 1u;
    }
    RWStructuredBuffer<uint2> history = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.pages));
    if ((push.cacheMode & 2u) != 0u) {
        uint size, stride;
        history.GetDimensions(size, stride);
        nextBase = size - historySize;
        if (lane == 0u) { history[nextBase] = uint2(0u, frame); }
    }
    uint3 counts = uint3(0u);
    for (uint block = firstBlock; block < endBlock; ++block) { counts.x += owners[blockOffset + block]; }
    for (uint index = firstSlot; index < endSlot; ++index) {
        uint owner = owners[vsmPhysicalHistory(push, previousBase, index).x];
        if (owner == 0u) { ++counts.y; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { ++counts.z; }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    uint3 totals = gVsmAllocationScan[255];
    uint candidateCount = totals.y + totals.z;
    for (uint block = firstBlock; block < endBlock; ++block) {
        uint count = owners[blockOffset + block];
        owners[blockOffset + block] = prefix.x;
        prefix.x += count;
    }
    // Stable partition in the previous frame's LRU order: free, unrequested,
    // pinned. Assign consumes the first missing candidates. Rotate that prefix
    // to the newest end so every surviving unrequested owner keeps its order.
    // Same-frame hits tie deterministically; ordering never compares frame IDs.
    for (uint index = firstSlot; index < endSlot; ++index) {
        uint2 prior = vsmPhysicalHistory(push, previousBase, index);
        uint physical = prior.x;
        uint owner = owners[physical];
        uint rank;
        if (owner == 0u) { rank = prefix.y++; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { rank = totals.y + prefix.z++; }
        else { rank = candidateCount + index - 1u - prefix.y - prefix.z; }
        if (rank < candidateCount) { owners[candidatesOffset + rank] = physical; }
        if ((push.cacheMode & 2u) != 0u) {
            bool assigned = rank < totals.x;
            uint nextRank = assigned ? rank + push.physicalPageCount - 1u - totals.x : rank - totals.x;
            uint lastRequestedFrame = assigned || rank >= candidateCount ? frame : (owner != 0u ? prior.y : 0u);
            history[nextBase + 1u + nextRank] = uint2(physical, lastRequestedFrame);
#if VSM_WORKLOAD_DIAGNOSTICS
            if (assigned && owner != 0u) {
                vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictions, 1u);
                vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictionAgeFrames, frame - prior.y);
            }
#endif
        }
    }
    if (lane == 0u) {
        uint missing = totals.x;
        statistics[20] = missing;
        statistics[16] += missing;
        uint coarseFlags = pages[push.totalPages - 1u].w;
        statistics[21] = push.enabled != 0u && (coarseFlags & 2u) != 0u ? 1u : 0u;
        if ((push.cacheMode & 32u) != 0u) {
            // Slot zero is pinned: the final word of the physical candidate array
            // is unused. Publish a snapshot before Finalize updates coarse validity.
            owners[candidatesOffset + push.physicalPageCount - 1u] = coarseFlags;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAssignMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uniform ShadowPush push)
{
    if (id.x >= push.totalPages) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blockOffset = push.physicalPageCount + push.totalPages;
    uint localRank = owners[push.physicalPageCount + id.x];
    if (localRank == 0xffffffffu) { return; }
    uint rank = owners[blockOffset + group.x] + localRank;
    uint physical = owners[blockOffset + blockCount + rank];
    uint evicted = owners[physical];
    if (evicted != 0u) {
        // Only missing admitted pages run this phase. An evicted unrequested
        // page has no competing invocation reading/writing its page entry.
        uint4 old = pages[evicted - 1u];
        old.x = 0xffffffffu; old.w = 2u;
        pages[evicted - 1u] = old;
#if VSM_RECEIVER_MASK
        RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
        validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, evicted - 1u), uint2(0u));
#endif
    }
    uint4 entry = pages[id.x];
    entry.x = physical; entry.w = 2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u);
    pages[id.x] = entry;
    owners[physical] = id.x + 1u;
#if VSM_CLUSTER_PAGE_BITSET
    // This path handles only admitted missing pages. The writes above make
    // each one resident and dirty; no invocation reads another evicted page.
    InterlockedOr(owners[vsmClusterPageBitsetOffset(push) + (id.x >> 5u)], 1u << (id.x & 31u));
#endif
}

groupshared float3 g_vsmClusterMinimum, g_vsmClusterMaximum;
groupshared uint g_vsmClusterValid, g_vsmClusterLevelMask;

public void vsmBuildPageHierarchy(uint index, ShadowPush push)
{
#if VSM_DIRTY_PAGE_HIERARCHY
    uint bitsetWords = (push.totalPages + 31u) / 32u;
    if (index >= bitsetWords) { return; }
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    if (statistics[16] == 0u && (push.cacheMode & 32u) == 0u) { return; }
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint bits = owners[vsmClusterPageBitsetOffset(push) + index];
    uint pageStride = push.virtualPagesPerAxis * push.virtualPagesPerAxis;
    uint paddedAxis = vsmPageHierarchyAxis(push);
    while (bits != 0u) {
        uint page = index * 32u + uint(firstbitlow(bits));
        bits &= bits - 1u;
        // Admit/Assign publish only resident, admitted, dirty fine-page bits.
        uint level = page / pageStride;
        uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
        uint localPage = page % pageStride;
        uint2 xy = uint2(localPage % axis, localPage / axis);
        uint base = vsmPageHierarchyOffset(push) + level * vsmPageHierarchyLevelWords(push);
        InterlockedMin(owners[base], xy.x);
        InterlockedMin(owners[base + 1u], xy.y);
        InterlockedMax(owners[base + 2u], xy.x);
        InterlockedMax(owners[base + 3u], xy.y);
        uint mipOffset = base + 4u;
        for (uint mipAxis = paddedAxis >> 1u; mipAxis != 0u; mipAxis >>= 1u) {
            xy >>= 1u;
            uint previous;
            InterlockedOr(owners[mipOffset + xy.y * mipAxis + xy.x], 1u, previous);
            // The first writer propagates to the root. The dispatch publication
            // barrier completes that writer before any cluster consumes nodes.
            if (previous != 0u) { break; }
            mipOffset += mipAxis * mipAxis;
        }
    }
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildPageHierarchyMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    vsmBuildPageHierarchy(id.x, push);
}

#if VSM_DIRTY_PAGE_HIERARCHY
public bool vsmHierarchyOverlaps(int2 first, int2 last, ShadowProjection projection, ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint base = vsmPageHierarchyOffset(push) + projection.page.z * vsmPageHierarchyLevelWords(push);
    uint2 low = uint2(owners[base], owners[base + 1u]);
    uint2 high = uint2(owners[base + 2u], owners[base + 3u]);
    bool empty = any(low > high) || any(uint2(last) < low) || any(uint2(first) > high);
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyBoundsRejects, empty ? 1u : 0u);
#endif
    if (empty) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterEmptyLevels, 1u);
#endif
        return false;
    }
    uint2 clippedFirst = max(uint2(first), low), clippedLast = min(uint2(last), high);
    uint2 span = clippedLast - clippedFirst;
    uint maxSpan = max(span.x, span.y);
    // Inclusive extent <= 2^mip intersects at most two aligned cells per axis.
    uint mip = maxSpan == 0u ? 0u : uint(firstbithigh(maxSpan)) + 1u;
    uint nodeTests = 0u;
    uint flags = 0u;
    if (mip == 0u) {
        uint page = projection.page.x + clippedFirst.y * projection.page.y + clippedFirst.x;
        flags = owners[vsmClusterPageBitsetOffset(push) + (page >> 5u)] & (1u << (page & 31u));
        nodeTests = 1u;
    } else {
        uint paddedAxis = vsmPageHierarchyAxis(push);
        uint axis = paddedAxis >> mip;
        uint offset = base + 4u + vsmPageHierarchyMipOffset(paddedAxis, mip);
        uint2 start = clippedFirst >> mip, end = clippedLast >> mip;
        for (uint y = start.y; y <= end.y; ++y) {
            for (uint x = start.x; x <= end.x; ++x) {
                flags |= owners[offset + y * axis + x];
                ++nodeTests;
            }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyNodeTests, nodeTests);
    vsmBuildDiagnosticAdd(push, flags != 0u ? kVSMBuildDiagnosticClusterPageHits : kVSMBuildDiagnosticClusterEmptyLevels, 1u);
#endif
    return flags != 0u;
}
#endif

#if VSM_STATIC_PAGE_HZB
bool vsmClusterVisibleInStaticHZB(int2 first, int2 last, float3 minimum, float3 maximum,
    ShadowProjection projection, StructuredBuffer<uint4> pages, ShadowPush push, bool afterStaticRaster)
{
    // Query only on a broad-phase hit. Bound the extra work on large clusters;
    // keeping a level falls back to the original per-triangle page enumeration.
    if (any(last - first >= 4)) { return true; }
    bool tested = false;
    for (int y = first.y; y <= last.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            uint2 xy = uint2(x, y);
            uint4 entry = pages[projection.page.x + xy.y * projection.page.y + xy.x];
            if ((entry.y & 2u) == 0u || (entry.w & (afterStaticRaster ? 2u : 10u)) == 0u) { continue; }
            // Build can use only retained static depth. After static Raster and
            // its HZB update, newly rendered complete pages also certify occlusion.
            if ((entry.w & (afterStaticRaster ? 4u : 12u)) != 4u) { return true; }
#if VSM_WORKLOAD_DIAGNOSTICS
            if (afterStaticRaster) {
                if (!tested) { vsmWorkAdd(push, kVSMWorkRasterHZBClusterLevelTests, 1u); }
            } else { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticHZBPageTests, 1u); }
#endif
            tested = true;
            if (!vsmStaticPageHidesBounds(push, projection, xy, entry.x, minimum, maximum)) { return true; }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (afterStaticRaster) { vsmWorkAdd(push, kVSMWorkRasterHZBRejectedLevels, uint(tested)); }
    else { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticHZBLevelRejects, uint(tested)); }
#endif
    // A positive HMip node may contain no exact dirty pages in this rectangle.
    // Keep its original decision unless we actually proved occlusion.
    return !tested;
}

#if VSM_CLUSTER_HZB
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmCullDynamicClustersHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    uint cluster = push.workBase + group.x;
    uint firstTriangle = cluster * 64u;
    if (lane == 0u) {
        uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, cluster * 2u);
        if (push.triangleCount - firstTriangle > 32u) {
            dynamicBits |= loadVirtualShadowMapDynamicMaskWord(push.geometry, cluster * 2u + 1u);
        }
        g_vsmClusterLevelMask = 0xffffffffu;
        g_vsmClusterValid = 0u;
        if (dynamicBits != 0u) {
            float3 minimum, maximum;
            g_vsmClusterValid = uint(loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle, minimum, maximum));
            g_vsmClusterMinimum = minimum;
            g_vsmClusterMaximum = maximum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_vsmClusterValid != 0u && lane <= push.clipmapCount) {
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        ShadowProjection projection = projections[lane];
        float3 minimum = g_vsmClusterMinimum, maximum = g_vsmClusterMaximum;
        // Enumerate a superset of the pages Build can emit for any enclosed
        // triangle. The coarse level always covers one physical page.
        float2 x = vsmHZBProjectedRange(minimum, maximum, projection.rowX);
        float2 y = vsmHZBProjectedRange(minimum, maximum, projection.rowY);
        float2 lo = float2(x.x, y.x), hi = float2(x.y, y.y);
        if (all(isfinite(lo)) && all(isfinite(hi)) && all(lo <= 1.0) && all(hi >= -1.0)) {
            int axis = int(projection.page.y);
            int2 first = min(int2(floor(saturate(lo * 0.5 + 0.5) * float(axis))), int2(axis - 1));
            int2 last = min(int2(floor(saturate(hi * 0.5 + 0.5) * float(axis))), int2(axis - 1));
            if (!vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, true)) {
                InterlockedAnd(g_vsmClusterLevelMask, ~(1u << lane));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        RWStructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        visibility[cluster] = g_vsmClusterLevelMask;
    }
}
#endif
#endif

bool vsmClusterNeedsLevel(float3 minimum, float3 maximum, ShadowProjection projection,
    StructuredBuffer<uint4> pages, ShadowPush push)
{
    float2 lo = float2(1e30), hi = -lo;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 position = float3((corner & 1u) != 0u ? maximum.x : minimum.x,
            (corner & 2u) != 0u ? maximum.y : minimum.y,
            (corner & 4u) != 0u ? maximum.z : minimum.z);
        float4 clip = vsmProjectClip(position, projection);
#if !VSM_DIRECTIONAL_BUILD
        // A box crossing the perspective near plane cannot be bounded by
        // dividing its corners. Retain this level for the exact triangle clip.
        if (projection.metric.w > 0.5 && clip.w <= projection.originNear.w) {
#if VSM_WORKLOAD_DIAGNOSTICS
            vsmRecordClusterLevel(push, 0u, kVSMBuildDiagnosticClusterNearPlaneKeeps);
#endif
            return true;
        }
        float2 projected = projection.metric.w > 0.5 ? clip.xy / clip.w : clip.xy;
#else
        float2 projected = clip.xy;
#endif
        lo = min(lo, projected); hi = max(hi, projected);
    }
    int axis = int(projection.page.y);
    // Expand independently of the world-space bound to cover projected float
    // rounding and floor-to-page boundaries. Do not cull on projected depth.
    float guard = 2.0 / float(axis);
    if (any(hi < -1.0 - guard) || any(lo > 1.0 + guard)) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmRecordClusterLevel(push, 0u, kVSMBuildDiagnosticClusterOutsideLevels);
#endif
        return false;
    }
    int2 first = clamp(int2(floor(saturate(lo * 0.5 + 0.5) * float(axis))) - 1, int2(0), int2(axis - 1));
    int2 last = clamp(int2(floor(saturate(hi * 0.5 + 0.5) * float(axis))) + 1, int2(0), int2(axis - 1));
#if VSM_DIRTY_PAGE_HIERARCHY
#if VSM_STATIC_PAGE_HZB
    if (!vsmHierarchyOverlaps(first, last, projection, push)) { return false; }
    return vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, false);
#else
    return vsmHierarchyOverlaps(first, last, projection, push);
#endif
#else
    StructuredBuffer<uint> bitset = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint offset = vsmClusterPageBitsetOffset(push);
    uint cachedWord = 0xffffffffu, rawWord = 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
    uint wordTests = 0u;
#endif
    for (int y = first.y; y <= last.y; ++y) {
        uint firstPage = projection.page.x + uint(y * axis + first.x);
        uint lastPage = projection.page.x + uint(y * axis + last.x);
        uint firstWord = firstPage >> 5u, lastWord = lastPage >> 5u;
        for (uint word = firstWord; word <= lastWord; ++word) {
            // Row ranges advance monotonically. Keep the unmasked word when
            // an unaligned next row shares it (for example a 127-page axis).
            if (word != cachedWord) {
                rawWord = bitset[offset + word];
                cachedWord = word;
#if VSM_WORKLOAD_DIAGNOSTICS
                ++wordTests;
#endif
            }
            uint bits = rawWord;
            if (word == firstWord) { bits &= 0xffffffffu << (firstPage & 31u); }
            if (word == lastWord) { bits &= 0xffffffffu >> (31u - (lastPage & 31u)); }
            if (bits != 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                vsmRecordClusterLevel(push, wordTests, kVSMBuildDiagnosticClusterPageHits);
#endif
#if VSM_STATIC_PAGE_HZB
                return vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, false);
#else
                return true;
#endif
            }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmRecordClusterLevel(push, wordTests, kVSMBuildDiagnosticClusterEmptyLevels);
#endif
    return false;
#endif
}

#if VSM_COMPACT_BUILD
uint vsmCullCluster(uint firstTriangle, uint lane, ShadowPush push,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages)
{
    if (lane == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterGroups, 1u);
#endif
        g_vsmClusterLevelMask = 0u;
        float3 minimum, maximum;
        g_vsmClusterValid = loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle,
            minimum, maximum) ? 1u : 0u;
        g_vsmClusterMinimum = minimum;
        g_vsmClusterMaximum = maximum;
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterBoundsValidGroups, g_vsmClusterValid);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
#if VSM_WORKLOAD_DIAGNOSTICS
    // Keep the predicate's diagnostic side effects inside the lane/valid guard.
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u) {
        if (vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
        }
    }
#else
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u &&
        vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
        InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    uint levelMask = g_vsmClusterLevelMask;
#if VSM_WORKLOAD_DIAGNOSTICS
    if (lane == 0u && levelMask != 0u) { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterSurvivingGroups, 1u); }
#endif
    return levelMask;
}
#endif

#if VSM_COMPACT_BUILD
#if VSM_INSTANCE_BUILD
struct VSMInstanceRange
{
    float3 minimum;
    uint first;
    float3 maximum;
    uint count;
};

bool vsmInstanceNeedsLevel(VSMInstanceRange instance, ShadowProjection projection, ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint base = vsmPageHierarchyOffset(push) + projection.page.z * vsmPageHierarchyLevelWords(push);
    uint2 low = uint2(owners[base], owners[base + 1u]);
    uint2 high = uint2(owners[base + 2u], owners[base + 3u]);
    if (any(low > high)) { return false; }
    if (any(instance.minimum > instance.maximum)) { return true; }
    // The instance path is compiled only for orthographic directional lights.
    // Use the same projection order and one-page rounding guard as cluster culling.
    float2 lo = float2(1e30), hi = -lo;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 position = float3((corner & 1u) ? instance.maximum.x : instance.minimum.x,
            (corner & 2u) ? instance.maximum.y : instance.minimum.y,
            (corner & 4u) ? instance.maximum.z : instance.minimum.z);
        float2 xy = vsmProjectClip(position, projection).xy;
        lo = min(lo, xy); hi = max(hi, xy);
    }
    float axis = float(projection.page.y);
    return all((hi * 0.5 + 0.5) * axis >= float2(low) - 1.0) &&
        all((lo * 0.5 + 0.5) * axis <= float2(high) + 2.0);
}

groupshared uint g_vsmInstanceTicketBase;
#endif
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmCompactMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
#if VSM_INSTANCE_BUILD
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    uint input = push.workBase + group.x;
    uint first, end;
    if (statistics[16] == 0u) {
        if ((push.cacheMode & 32u) == 0u || input >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        first = vsmMergeCluster(push, input).x;
        end = first + min(push.triangleCount - first, 64u);
    } else {
        if (input >= vsmCacheMetadata(push, kVSMCacheInstanceCount)) { return; }
        StructuredBuffer<VSMInstanceRange> instances = resolveUniform(ResourceHandle<StructuredBuffer<VSMInstanceRange>>(
            vsmCacheMetadata(push, kVSMCacheInstanceTable)));
        VSMInstanceRange instance = instances[input];
        if (lane == 0u) { g_vsmClusterLevelMask = 0u; }
        GroupMemoryBarrierWithGroupSync();
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        if (instance.count != 0u && lane < push.clipmapCount && vsmInstanceNeedsLevel(instance, projections[lane], push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
        }
        GroupMemoryBarrierWithGroupSync();
#if VSM_WORKLOAD_DIAGNOSTICS
        if (lane == 0u) {
            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceTests, 1u);
            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceRejects, uint(g_vsmClusterLevelMask == 0u));
        }
#endif
        if (g_vsmClusterLevelMask == 0u) { return; }
        first = instance.first;
        end = first + instance.count;
    }
    uint clusterFirst = first >> 6u;
    uint count = ((end - 1u) >> 6u) - clusterFirst + 1u;
    RWStructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint offset = vsmCompactCandidateOffset(push);
    if (lane == 0u) {
        uint slot;
        InterlockedAdd(candidates[offset], count, slot);
        g_vsmInstanceTicketBase = slot;
        RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        // Contiguous reservations have no holes; publish the high-water mark in each indirect bin.
        uint stop = slot + count;
        for (uint i = slot; i < stop;) {
            uint bin = i / 65535u;
            uint remaining = min(stop - i, 65535u - i % 65535u);
            InterlockedMax(arguments[(push.batchCount + push.coarseBatchCount + bin) * 3u], i % 65535u + remaining);
            i += remaining;
        }
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceClusterTickets, count);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = lane; j < count; j += 64u) {
        uint aligned = (clusterFirst + j) * 64u;
        uint ticket = offset + 1u + (g_vsmInstanceTicketBase + j) * 2u;
        candidates[ticket] = max(aligned, first);
        candidates[ticket + 1u] = aligned + min(end - aligned, 64u);
    }
#else
    // The source ranges are aligned to 64 triangles. Rounded tail lanes still
    // participate in the identical group-cooperative cluster predicate.
    uint firstTriangle = push.workBase + group.x * 64u;
    if (firstTriangle >= push.triangleCount || push.enabled == 0u) { return; }
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    if (statistics[16] == 0u && (push.cacheMode & 32u) == 0u) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    if (statistics[16] == 0u) {
        uint cluster = firstTriangle >> 6u;
        if (cluster >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        firstTriangle = vsmMergeCluster(push, cluster).x;
    }
    uint levelMask = vsmCullCluster(firstTriangle, lane, push, projections, pages);
    if (lane == 0u && levelMask != 0u) {
        RWStructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
        uint offset = vsmCompactCandidateOffset(push);
        uint slot;
        InterlockedAdd(candidates[offset], 1u, slot);
        // At most one candidate per source group: the full source-cluster
        // capacity bounds this append and every indirect bin to 65535 groups.
        candidates[offset + 1u + slot * 2u] = firstTriangle;
        candidates[offset + 2u + slot * 2u] = levelMask;
        RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        uint argument = (push.batchCount + push.coarseBatchCount + slot / 65535u) * 3u;
        InterlockedAdd(arguments[argument], 1u);
    }
#endif
}
#endif

#if VSM_CLUSTER_PAGE_RASTER
void vsmAppendClusterPage(uint page, uint triangleIndex, RWStructuredBuffer<uint4> work,
    RWStructuredBuffer<uint> statistics, uint capacity)
{
    // Every invocation here belongs to the same original aligned cluster,
    // including indirect Build and sparse demotion. Each 16-slot quarter
    // bounds worst-case clipped mesh output to 144 vertices and 112 primitives.
    uint key = (page << 2u) | ((triangleIndex >> 4u) & 3u);
    for (;;) {
        uint firstKey = WaveReadLaneFirst(key);
        if (key == firstKey) {
            uint mask = WaveActiveBitOr(1u << (triangleIndex & 31u));
            if (WaveIsFirstLane()) {
                uint count = countbits(mask);
                uint first;
                InterlockedAdd(statistics[3], count, first);
                uint admitted = min(capacity - min(capacity, first), count);
                if (admitted < count) {
                    InterlockedAdd(statistics[5], count - admitted);
                    for (uint i = admitted; i < count; ++i) { mask &= mask - 1u; }
                }
                if (mask != 0u) {
                    uint ticket;
                    InterlockedAdd(work[0].x, 1u, ticket);
                    bool high = (triangleIndex & 32u) != 0u;
                    work[1u + ticket] = uint4(triangleIndex & ~63u, page, high ? 0u : mask, high ? mask : 0u);
                }
            }
            break;
        }
    }
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID,
    uint lane : SV_GroupIndex, uniform ShadowPush push)
{
#if VSM_COMPACT_BUILD
    StructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint candidate = vsmCompactCandidateOffset(push) + 1u + (push.workBase + group.x) * 2u;
#if VSM_INSTANCE_BUILD
    uint firstSlot = candidates[candidate];
    uint firstTriangle = firstSlot & ~63u;
    uint triangleIndex = firstTriangle + lane;
#else
    uint triangleIndex = candidates[candidate] + lane;
    uint levelMask = candidates[candidate + 1u];
    if (triangleIndex >= push.triangleCount) { return; }
#endif
#else
    uint triangleIndex = push.workBase + id.x;
    // Each CPU dispatch starts at a multiple of 64, including batched draws.
    // Only uniform exits precede the two group barriers; tail lanes participate.
    uint firstTriangle = push.workBase + group.x * 64u;
    if (firstTriangle >= push.triangleCount || push.enabled == 0u) { return; }
#endif
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    bool mergeFrame = (push.cacheMode & 32u) != 0u;
    bool mergeOnly = mergeFrame && statistics[16] == 0u;
#if !VSM_COMPACT_BUILD
    if (!mergeFrame && statistics[16] == 0u) { return; }
    bool selected = true;
    if (mergeOnly) {
        // Any independent static invalidation also dirties the composite. When
        // no fine composite needs work, the bitset contains only static merges.
        uint cluster = firstTriangle >> 6u;
        if (cluster >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        uint4 record = vsmMergeCluster(push, cluster);
        firstTriangle = record.x;
        triangleIndex = firstTriangle + lane;
        selected = vsmMergeClusterHasLane(record, lane);
    }
#endif
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
#if VSM_INSTANCE_BUILD
    uint levelMask = vsmCullCluster(firstTriangle, lane, push, projections, pages);
    if (levelMask == 0u || triangleIndex < firstSlot || triangleIndex >= candidates[candidate + 1u]) { return; }
#endif
#if !VSM_COMPACT_BUILD
    if (lane == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterGroups, 1u);
#endif
        g_vsmClusterLevelMask = 0u;
        float3 minimum, maximum;
        g_vsmClusterValid = loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle,
            minimum, maximum) ? 1u : 0u;
        g_vsmClusterMinimum = minimum;
        g_vsmClusterMaximum = maximum;
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterBoundsValidGroups, g_vsmClusterValid);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
#if VSM_WORKLOAD_DIAGNOSTICS
    // Keep the predicate's diagnostic side effects inside the lane/valid guard.
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u) {
        if (vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
        }
    }
#else
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u &&
        vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
        InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    uint levelMask = g_vsmClusterLevelMask;
#if VSM_WORKLOAD_DIAGNOSTICS
    if (lane == 0u && levelMask != 0u) { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterSurvivingGroups, 1u); }
#endif
    if (!selected || triangleIndex >= push.triangleCount || levelMask == 0u) { return; }
#endif
    bool mergeTriangle = true;
    if (mergeFrame && (VSM_COMPACT_BUILD != 0 || !mergeOnly)) {
        uint word = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
        mergeTriangle = ((word >> (triangleIndex & 31u)) & 1u) != 0u;
        if (mergeOnly && !mergeTriangle) { return; }
    }
    vsmWorkAdd(push, kVSMWorkBuildActiveTriangleSlots, 1u);
    VirtualShadowMapTriangle triangle;
    if (!loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) { return; }
    vsmWorkAdd(push, kVSMWorkBuildDecodedTriangles, 1u);
    vsmWorkAdd(push, kVSMWorkBuildLevelTests, countbits(levelMask));
    StructuredBuffer<uint> bitset = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint bitsetOffset = vsmClusterPageBitsetOffset(push);
#if VSM_WORKLOAD_DIAGNOSTICS
    uint wordTests = 0u, pageHits = 0u;
    bool layeredBuild = (push.cacheMode & 16u) != 0u;
    bool dynamicTriangle = false;
    bool staticFineRequired = false, compositeFineHit = false;
    if (layeredBuild) {
        // The dynamic layer descriptor has the same slot indices as full Build.
        uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, 34u), triangleIndex >> 5u);
        dynamicTriangle = ((mask >> (triangleIndex & 31u)) & 1u) != 0u;
        vsmBuildDiagnosticAdd(push, dynamicTriangle ? kVSMBuildDiagnosticLayeredDynamicDecodedTriangles :
            kVSMBuildDiagnosticLayeredStaticDecodedTriangles, 1u);
    }
#endif
#if VSM_PAGE_WINDOW_RASTER
    uint geometrySlot = 0xffffffffu;
#endif
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
#else
    RWStructuredBuffer<uint2> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
#endif
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        if ((levelMask & (1u << level)) == 0u) { continue; }
        ShadowProjection projection = projections[level];
        float3 lo, hi;
#if !VSM_DIRECTIONAL_BUILD
        if (projection.metric.w > 0.5) {
            ShadowClipVertex polygon[9];
            for (uint vertex = 0u; vertex < 3u; ++vertex) {
                polygon[vertex].position = vsmProjectClip(triangle.vertices[vertex].position.xyz, projection);
                polygon[vertex].uv = float2(0.0);
            }
            // Projecting only the original vertices misses triangles crossing
            // near/w=0. Clip the complete face before any perspective division.
            uint count = vsmClipProjection(polygon, float4(-1.0, -1.0, 1.0, 1.0), true);
            if (count < 3u) { continue; }
            lo = hi = polygon[0].position.xyz / polygon[0].position.w;
            for (uint vertex = 1u; vertex < count; ++vertex) {
                float3 projected = polygon[vertex].position.xyz / polygon[vertex].position.w;
                lo = min(lo, projected); hi = max(hi, projected);
            }
        } else
#endif
        {
            float3 a = vsmProjectPosition(triangle.vertices[0].position.xyz, projection);
            float3 b = vsmProjectPosition(triangle.vertices[1].position.xyz, projection);
            float3 c = vsmProjectPosition(triangle.vertices[2].position.xyz, projection);
            lo = min(a, min(b, c)); hi = max(a, max(b, c));
        }
        if (hi.z < 0.0 || lo.z > 1.0 || any(hi.xy < -1.0) || any(lo.xy > 1.0)) { continue; }
        int axis = int(projection.page.y);
        // Clip the projected bounds before integer conversion; off-clipmap
        // world coordinates need not fit the page-index integer range.
        int2 first = min(int2(floor(saturate(lo.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        int2 last = min(int2(floor(saturate(hi.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
#if VSM_RECEIVER_MASK
        // Build uses clipmap translation; Raster forms canonical page translation.
        // Bound the operand magnitudes of both paths before converting their
        // rounding allowance to texels. The 256-epsilon allowance follows Mark's
        // conservative footprint policy; one extra texel covers integer boundaries.
        float resolution = float(uint(axis) * push.pageSize);
        float3 positionMagnitude = max(abs(triangle.vertices[0].position.xyz),
            max(abs(triangle.vertices[1].position.xyz), abs(triangle.vertices[2].position.xyz)));
        float2 clipMagnitude = float2(dot(abs(projection.rowX.xyz), positionMagnitude),
            dot(abs(projection.rowY.xyz), positionMagnitude)) + abs(float2(projection.rowX.w, projection.rowY.w)) + 1.0;
        float2 pageTranslationMagnitude = (abs(projection.originNear.xy) + float(axis) + 1.0) * (2.0 / float(axis));
        float2 texelPadding = 1.0 + (clipMagnitude + pageTranslationMagnitude) *
            (256.0 * 5.960464477539063e-8) * (0.5 * resolution);
        uint2 firstTexel = uint2(clamp(saturate(lo.xy * 0.5 + 0.5) * resolution - texelPadding, 0.0, resolution - 1.0));
        uint2 lastTexel = uint2(clamp(saturate(hi.xy * 0.5 + 0.5) * resolution + texelPadding, 0.0, resolution - 1.0));
#endif
        uint cachedWord = 0xffffffffu, rawWord = 0u;
        for (int y = first.y; y <= last.y; ++y) {
            uint firstPage = projection.page.x + uint(y * axis + first.x);
            uint lastPage = projection.page.x + uint(y * axis + last.x);
            uint firstWord = firstPage >> 5u, lastWord = lastPage >> 5u;
            for (uint word = firstWord; word <= lastWord; ++word) {
                // Keep the raw word: adjacent non-aligned rows can share it,
                // but their masks must not consume one another's set bits.
                if (word != cachedWord) {
                    rawWord = bitset[bitsetOffset + word];
                    cachedWord = word;
#if VSM_WORKLOAD_DIAGNOSTICS
                    ++wordTests;
#endif
                }
                uint bits = rawWord;
                if (word == firstWord) { bits &= 0xffffffffu << (firstPage & 31u); }
                if (word == lastWord) { bits &= 0xffffffffu >> (31u - (lastPage & 31u)); }
#if VSM_PAGE_WINDOW_RASTER
                // A nibble stays within this row and dirty word, including when the row starts mid-word.
                uint pending = bits;
                while (pending != 0u) {
                    uint shift = uint(firstbitlow(pending)) & ~3u;
                    bits = pending & (15u << shift);
                    pending ^= bits;
                    uint basePage = (word << 5u) + shift;
                    uint accepted = 0u;
#endif
                while (bits != 0u) {
                    uint page = (word << 5u) + uint(firstbitlow(bits));
                    bits &= bits - 1u;
                    // Admit/Assign set exactly resident, admitted, dirty fine
                    // pages. Physical pins admitted owners through this build.
                    // Increasing set bits preserve this triangle's row order.
                    if (!mergeTriangle && (pages[page].w & (2u | kVSMPageMergeStatic)) == kVSMPageMergeStatic) {
                        continue;
                    }
#if VSM_WORKLOAD_DIAGNOSTICS
                    ++pageHits;
#endif
#if VSM_RECEIVER_MASK
                    // Both layers share this work list. Static-dirty pages must
                    // remain complete for reuse by future receiver footprints.
                    if ((pages[page].w & 8u) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageTests, 1u);
#endif
                        uint localPage = page - projection.page.x;
                        uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel,
                            uint2(localPage % uint(axis), localPage / uint(axis)), push.pageSize);
                        uint offset = vsmReceiverMaskOffset(push) + page * 2u;
                        if (((mask.x & bitset[offset]) | (mask.y & bitset[offset + 1u])) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageRejects, 1u);
#endif
                            continue;
                        }
                    }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
                    if (layeredBuild && !dynamicTriangle) {
                        uint flags = pages[page].w;
                        // Incremental static merge consumes only the transition mask.
                        staticFineRequired = staticFineRequired || ((flags & 8u) != 0u &&
                            ((flags & kVSMPageMergeStatic) == 0u || mergeTriangle));
                        compositeFineHit = compositeFineHit || (flags & 2u) != 0u;
                    }
#endif
#if VSM_PAGE_WINDOW_RASTER
                    accepted |= 1u << (page - basePage);
#elif VSM_CLUSTER_PAGE_RASTER
                    vsmAppendClusterPage(page, triangleIndex, work, statistics, push.workCapacity);
#else
                    uint index;
                    InterlockedAdd(statistics[3], 1u, index);
                    if (index < push.workCapacity) { work[index] = uint2(triangleIndex, page); }
                    else { InterlockedAdd(statistics[5], 1u); }
#endif
                }
#if VSM_PAGE_WINDOW_RASTER
                    if (accepted == 0u) { continue; }
                    uint count = countbits(accepted), firstWork;
                    InterlockedAdd(statistics[3], count, firstWork);
                    uint admitted = min(count, push.workCapacity - min(firstWork, push.workCapacity));
                    if (admitted < count) {
                        InterlockedAdd(statistics[5], count - admitted);
                        for (uint i = admitted; i < count; ++i) { accepted &= accepted - 1u; }
                        if (accepted == 0u) { continue; }
                    }
                    uint ticket;
                    InterlockedAdd(work[0].x, 1u, ticket);
                    if (geometrySlot == 0xffffffffu) {
                        geometrySlot = push.triangleCount <= push.workCapacity ? triangleIndex : ticket;
                    }
                    work[1u + ticket] = uint4(triangleIndex, basePage, accepted, geometrySlot);
                }
#endif
            }
        }
    }
#if VSM_PAGE_WINDOW_RASTER
    // Build owns each selected triangle once, including split instance ranges and demotion.
    // The Build -> mesh barrier publishes this payload together with every referring ticket.
    if (geometrySlot != 0xffffffffu) {
        uint base = 1u + push.workCapacity + 4u * geometrySlot;
        for (uint i = 0u; i < 3u; ++i) {
            work[base + i] = asuint(float4(triangle.vertices[i].position.xyz, triangle.vertices[i].uv.x));
        }
        work[base + 3u] = uint4(asuint(float3(triangle.vertices[0].uv.y, triangle.vertices[1].uv.y,
            triangle.vertices[2].uv.y)), triangle.materialIndex);
    }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetWordTests, wordTests);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetPageHits, pageHits);
    if (layeredBuild && !dynamicTriangle) {
        vsmBuildDiagnosticAdd(push, staticFineRequired ? kVSMBuildDiagnosticStaticRequiredFineTriangles :
            (compositeFineHit ? kVSMBuildDiagnosticStaticCompositeOnlyTriangles :
                kVSMBuildDiagnosticStaticNoFinePageTriangles), 1u);
    }
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmFinalizeMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    uint count = min(statistics[3], push.workCapacity);
    if (id.x == 0u) { statistics[4] = count; }
    uint coarsePage = push.totalPages - 1u;
    bool redrawCoarse = statistics[21] != 0u;
    bool mergeFrame = (push.cacheMode & 32u) != 0u;
#if VSM_CLUSTER_PAGE_RASTER
    if (id.x == 0u) {
        uint clearOffset = vsmCacheMetadata(push, kVSMCacheMergeDrawOffset) +
            (mergeFrame ? push.batchCount + push.coarseBatchCount : 0u);
        arguments[clearOffset * 3u] = statistics[16] != 0u || redrawCoarse ? push.physicalPageCount : 0u;
        arguments[clearOffset * 3u + 1u] = 1u;
        arguments[clearOffset * 3u + 2u] = 1u;
    }
#endif
#if VSM_CLUSTER_HZB
    uint clusterCount = push.triangleCount / 64u + uint((push.triangleCount & 63u) != 0u);
    if (id.x < (clusterCount + 65534u) / 65535u && vsmCacheMetadata(push, 40u) != 0xffffffffu) {
        RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
        uint offset = 3u + push.physicalPageCount + id.x * 3u;
        jobs[offset] = ((count != 0u && statistics[16] != 0u) || redrawCoarse) ? min(clusterCount - id.x * 65535u, 65535u) : 0u;
        jobs[offset + 1u] = 1u;
        jobs[offset + 2u] = 1u;
    }
#endif
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        if (push.enabled != 0u && (entry.y & 2u) != 0u && (entry.w & 10u) != 0u) {
            if (id.x == coarsePage || statistics[5] == 0u) { entry.w |= 1u; }
            else { entry.w &= ~1u; }
            if ((entry.w & 2u) != 0u) {
                vsmCacheAdd(push, id.x == coarsePage ? kVSMCacheCompositeDirtyCoarsePages : kVSMCacheCompositeDirtyFinePages, 1u);
            }
            if ((push.cacheMode & 16u) != 0u && (entry.w & 8u) != 0u) {
                if (id.x == coarsePage || statistics[5] == 0u) { entry.w |= 4u; }
                else { entry.w &= ~4u; }
#if VSM_WORKLOAD_DIAGNOSTICS
                if ((entry.w & (4u | kVSMPageMergeStatic)) == (4u | kVSMPageMergeStatic)) {
                    vsmWorkAdd(push, kVSMWorkRasterStaticMergedPages, 1u);
                }
#endif
#if VSM_STATIC_PAGE_HZB
                if ((entry.w & 4u) != 0u) {
                    RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
                    uint job;
                    InterlockedAdd(jobs[0], 1u, job);
                    jobs[3u + job] = entry.x;
                }
#endif
                vsmCacheAdd(push, id.x == coarsePage ? kVSMCacheStaticDirtyCoarsePages : kVSMCacheStaticDirtyFinePages, 1u);
            }
            pages[id.x] = entry;
#if VSM_RECEIVER_MASK
            // The pending cache publishes this tail with the page table only
            // after submission; the same validity bit gates projection while
            // raster completes the declared draw list. Overflow cannot certify
            // partial work. Redraw replaces coverage, never unions stale depth.
            RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
            uint2 valid = uint2(0u);
            if (id.x != coarsePage && (entry.w & 1u) != 0u) {
                if ((entry.w & 8u) != 0u) { valid = uint2(0xffffffffu); }
                else {
                    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
                    uint offset = vsmReceiverMaskOffset(push) + id.x * 2u;
                    valid = uint2(owners[offset], owners[offset + 1u]);
                }
            }
            validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, id.x), valid);
#endif
        }
        if ((push.cacheMode & 16u) != 0u && id.x != coarsePage && entry.x != 0xffffffffu &&
            (entry.y & 2u) != 0u && (entry.w & 12u) == 4u) { vsmCacheAdd(push, kVSMCacheStaticReusedFinePages, 1u); }
        if (id.x != coarsePage && entry.x != 0xffffffffu && (entry.w & 1u) != 0u) {
            InterlockedAdd(statistics[24], 1u);
        }
    }
    if (id.x < push.batchCount + push.coarseBatchCount) {
        bool coarse = id.x >= push.batchCount;
        uint first = (coarse ? id.x - push.batchCount : id.x) * 65535u;
        // Every demotion touches the complete coarse static page, even when its
        // previously composed depth is still exact.
        uint fineDrawCount = count;
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
        StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
        fineDrawCount = work[0].x;
#endif
        uint drawCount = coarse ? ((redrawCoarse || (mergeFrame && push.enabled != 0u)) ? push.triangleCount : 0u) : fineDrawCount;
        uint dynamicCount = (coarse ? redrawCoarse : statistics[16] != 0u) ? drawCount : 0u;
        if (coarse && mergeFrame && drawCount != 0u) {
            StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
            uint coarseFlags = owners[vsmClusterPageBitsetOffset(push) - 1u];
            if ((coarseFlags & kVSMPageMergeStatic) != 0u) {
                uint clusters = vsmCacheMetadata(push, kVSMCacheMergeClusterCount);
                drawCount = clusters <= push.triangleCount / 64u ? clusters * 64u : push.triangleCount;
            }
        }
        arguments[id.x * 3u] = first < drawCount ? min(drawCount - first, 65535u) : 0u;
        arguments[id.x * 3u + 1u] = 1u;
        arguments[id.x * 3u + 2u] = 1u;
        if (mergeFrame) {
            uint offset = (vsmCacheMetadata(push, kVSMCacheMergeDrawOffset) + id.x) * 3u;
            arguments[offset] = first < dynamicCount ? min(dynamicCount - first, 65535u) : 0u;
            arguments[offset + 1u] = 1u;
            arguments[offset + 2u] = 1u;
        }
    }
}

// Projection-only metadata follows the unchanged residency/ownership prefix.
// Finalize must publish all validity bits before this independent dispatch.
void vsmBuildPropagatedMapping(uint3 id, ShadowPush push)
{
    if (id.x >= push.totalPages) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    pages[push.totalPages + id.x] = uint4(0xffffffffu, push.clipmapCount, 0u, 0u);
    if (id.x == push.totalPages - 1u) { return; }
    uint axis = push.virtualPagesPerAxis, stride = axis * axis;
    uint level = id.x / stride, local = id.x % stride;
    uint4 entry = pages[id.x];
    if (entry.x != 0xffffffffu && (entry.y & 2u) != 0u && (entry.w & 1u) != 0u) {
        pages[push.totalPages + id.x] = uint4(entry.x, level, 0u, 1u);
        return;
    }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection source = projections[level];
    if (push.lightType != 0u) {
        uint sourceAxis = source.page.y;
        if ((sourceAxis & (sourceAxis - 1u)) != 0u || local >= sourceAxis * sourceAxis) { return; }
        uint2 sourcePage = uint2(local % sourceAxis, local / sourceAxis);
        uint target = push.clipmapCount, physical = 0xffffffffu;
        for (uint next = level + 1u; next < push.clipmapCount; ++next) {
            uint delta = next - level;
            ShadowProjection parent = projections[next];
            if (parent.page.y != max(1u, sourceAxis >> delta) ||
                any(parent.rowX != source.rowX) || any(parent.rowY != source.rowY) ||
                any(parent.rowZ != source.rowZ) || any(parent.rowW != source.rowW)) { return; }
            // Local projections share identical normalized coordinates and
            // inclusive containment. Scaling their nonnegative address by a
            // power of two commutes with floor, including upper-edge clamping.
            uint2 parentPage = sourcePage >> min(delta, uint(firstbithigh(sourceAxis)));
            uint4 parentEntry = pages[parent.page.x + parentPage.y * parent.page.y + parentPage.x];
            if (parentEntry.x != 0xffffffffu && (parentEntry.y & 2u) != 0u && (parentEntry.w & 1u) != 0u) {
                target = next; physical = parentEntry.x;
                break;
            }
        }
        pages[push.totalPages + id.x] = uint4(physical, target, 0u, 3u);
        return;
    }
    if (axis < 2u || (axis & (axis - 1u)) != 0u) { return; }
    if (any(abs(source.originNear.xy) >= 67108864.0)) { return; }
    int2 basePage = int2(local % axis, local / axis) + int2(source.originNear.xy) - int(axis / 2u);
    uint packedParentPage = 0u;
    uint targetLevel = push.clipmapCount, physicalPage = 0xffffffffu;
    for (uint next = level + 1u; next < push.clipmapCount; ++next) {
        uint delta = next - level;
        float scale = asfloat((127u - delta) << 23u);
        ShadowProjection parent = projections[next];
        if (any(abs(parent.originNear.xy) >= 67108864.0) ||
            any(parent.rowX.xyz != source.rowX.xyz * scale) ||
            any(parent.rowY.xyz != source.rowY.xyz * scale) ||
            any(parent.rowZ != source.rowZ)) { return; }
        int2 parentPage = (basePage >> delta) - int2(parent.originNear.xy) + int(axis / 2u);
        if (any(parentPage < 0) || any(parentPage >= int(axis))) { continue; }
        // The half-texel containment inset cuts through outer page interiors.
        // Preserve that domain with the original search on these pages.
        if (any(parentPage == 0) || any(parentPage == int(axis) - 1)) { return; }
        uint4 parentEntry = pages[parent.page.x + uint(parentPage.y) * axis + uint(parentPage.x)];
        if (parentEntry.x != 0xffffffffu && (parentEntry.y & 2u) != 0u && (parentEntry.w & 1u) != 0u) {
            targetLevel = next; physicalPage = parentEntry.x;
            packedParentPage = uint(parentPage.x) | (uint(parentPage.y) << 16u);
            break;
        }
    }
    pages[push.totalPages + id.x] = uint4(physicalPage, targetLevel, packedParentPage, 2u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmPropagateMappingMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    vsmBuildPropagatedMapping(id, push);
}

struct ShadowClearVertex
{
    float4 position : SV_Position;
};

[shader("fragment")]
float vsmClearFragmentMain(ShadowClearVertex input, uniform ShadowPush push) : SV_Depth
{
    if ((push.cacheMode & 16u) != 0u && (push.rasterCoarse & 2u) == 0u) {
        Texture2D<float> staticAtlas = resolveUniform(ResourceHandle<Texture2D<float>>(vsmCacheMetadata(push, 32u)));
        return staticAtlas.Load(int3(int2(input.position.xy), 0));
    }
    return 0.0;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmClearMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowClearVertex vertices[4], out indices uint3 triangles[2], uniform ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint owner = owners[group.x];
    bool clear = false;
    if (owner != 0u && push.enabled != 0u) {
        uint4 entry = pages[owner - 1u];
        bool staticPass = (push.rasterCoarse & 2u) != 0u;
        clear = (entry.y & 2u) != 0u && (staticPass ? (entry.w & (8u | kVSMPageMergeStatic)) == 8u : (entry.w & 2u) != 0u);
    }
    SetMeshOutputCounts(clear ? 4u : 0u, clear ? 2u : 0u);
    if (!clear) { return; }
    if (lane == 0u) {
        vsmWorkAdd(push, kVSMWorkRasterClearedPages, 1u);
        if ((push.cacheMode & 16u) != 0u && (push.rasterCoarse & 2u) == 0u) { vsmCacheAdd(push, kVSMCacheRestoredPages, 1u); }
    }
    if (lane < 4u) {
        float2 corner = float2(lane & 1u, lane >> 1u);
        vertices[lane].position = float4(vsmAtlasPosition(corner, group.x, push), 0.0, 1.0);
    }
    if (lane < 2u) { triangles[lane] = lane == 0u ? uint3(0u, 2u, 1u) : uint3(1u, 2u, 3u); }
}

#if VSM_PAGE_WINDOW_RASTER
bool vsmRasterPageLayerActive(inout uint triangleIndex, uint4 page, ShadowPush push, out bool mergeStatic)
{
    const bool staticPass = (push.rasterCoarse & 2u) != 0u;
    bool layerPageActive = staticPass ? (page.w & 8u) != 0u :
        ((push.cacheMode & 32u) == 0u || (page.w & 2u) != 0u);
#if VSM_CLUSTER_HZB
    if (layerPageActive && (push.rasterCoarse & 4u) != 0u) {
        StructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        layerPageActive = (visibility[triangleIndex >> 6u] & (1u << page.z)) != 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
        if (!layerPageActive) {
            uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, triangleIndex >> 5u);
            vsmWorkAdd(push, kVSMWorkRasterHZBSkippedTriangles, (dynamicBits >> (triangleIndex & 31u)) & 1u);
        }
#endif
    }
#endif
    mergeStatic = staticPass && (push.cacheMode & 32u) != 0u && (page.w & kVSMPageMergeStatic) != 0u;
    if (mergeStatic) {
        if ((push.rasterCoarse & 1u) != 0u) {
            uint4 record = vsmMergeCluster(push, triangleIndex >> 6u);
            uint sourceLane = triangleIndex & 63u;
            triangleIndex = record.x + sourceLane;
            layerPageActive = vsmMergeClusterHasLane(record, sourceLane);
        } else {
            uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
            layerPageActive = (mask & (1u << (triangleIndex & 31u))) != 0u;
        }
    }
    return layerPageActive;
}

float4 vsmDirectionalPageBounds(uint virtualPage, inout ShadowProjection projection)
{
    uint local = virtualPage - projection.page.x;
    uint2 pageXY = uint2(local % projection.page.y, local / projection.page.y);
    // The snapped world-page translation belongs inside the vertex dot product.
    // Subtracting rolling clipmap bounds afterwards changes cached depths.
    float2 center = projection.originNear.xy + float2(pageXY) + 0.5 - 0.5 * float(projection.page.y);
    float halfPage = 1.0 / float(projection.page.y);
    projection.rowX.w = -center.x * (2.0 * halfPage);
    projection.rowY.w = -center.y * (2.0 * halfPage);
    return float4(-halfPage, -halfPage, halfPage, halfPage);
}
#endif

// Fine packed and single-triangle raster must produce the same fan, UVs and
// canonical page coordinates, including for alpha-masked materials.
uint vsmRasterPolygon(uint triangleIndex, uint virtualPage, ShadowPush push, bool skipInteriorClip,
    out ShadowClipVertex polygon[9], out uint material, out uint physicalPage, out float4 bounds)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    material = physicalPage = 0u;
    bounds = float4(0.0);
    VirtualShadowMapTriangle triangle;
    const bool staticPass = (push.rasterCoarse & 2u) != 0u;
    bool layerPageActive = staticPass ? (pages[virtualPage].w & 8u) != 0u :
        ((push.cacheMode & 32u) == 0u || (pages[virtualPage].w & 2u) != 0u);
#if VSM_CLUSTER_HZB
    if (layerPageActive && (push.rasterCoarse & 4u) != 0u) {
        StructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        layerPageActive = (visibility[triangleIndex >> 6u] & (1u << pages[virtualPage].z)) != 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
        if (!layerPageActive) {
            uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, triangleIndex >> 5u);
            vsmWorkAdd(push, kVSMWorkRasterHZBSkippedTriangles, (dynamicBits >> (triangleIndex & 31u)) & 1u);
        }
#endif
    }
#endif
    bool mergeStatic = staticPass && (push.cacheMode & 32u) != 0u && (pages[virtualPage].w & kVSMPageMergeStatic) != 0u;
    if (mergeStatic) {
        if ((push.rasterCoarse & 1u) != 0u) {
            uint4 record = vsmMergeCluster(push, triangleIndex >> 6u);
            uint sourceLane = triangleIndex & 63u;
            triangleIndex = record.x + sourceLane;
            layerPageActive = vsmMergeClusterHasLane(record, sourceLane);
        } else {
            uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
            layerPageActive = (mask & (1u << (triangleIndex & 31u))) != 0u;
        }
    }
    if (layerPageActive && loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) {
        vsmWorkAdd(push, kVSMWorkRasterDecodedTriangles, 1u);
#if VSM_WORKLOAD_DIAGNOSTICS
        if (mergeStatic) { vsmWorkAdd(push, kVSMWorkRasterStaticMergedTriangles, 1u); }
        if ((push.cacheMode & 16u) != 0u) {
            vsmCacheAdd(push, staticPass ? kVSMCacheStaticDecodedTriangles : kVSMCacheDynamicDecodedTriangles, 1u);
        }
#endif
        uint4 page = pages[virtualPage];
        ShadowProjection projection = projections[page.z];
        bounds = vsmPageBounds(virtualPage, projection);
        if (projection.metric.w < 0.5 && (push.rasterCoarse & 1u) == 0u) {
            uint local = virtualPage - projection.page.x;
            uint2 pageXY = uint2(local % projection.page.y, local / projection.page.y);
            // Directional originNear.xy carries the snapped page-grid
            // origin. Form a canonical world-page translation before the
            // vertex dot product: subtracting rolling clipmap bounds after
            // projection changes clipping/depth rounding in cached pages.
            float2 center = projection.originNear.xy + float2(pageXY) + 0.5 - 0.5 * float(projection.page.y);
            float halfPage = 1.0 / float(projection.page.y);
            projection.rowX.w = -center.x * (2.0 * halfPage);
            projection.rowY.w = -center.y * (2.0 * halfPage);
            bounds = float4(-halfPage, -halfPage, halfPage, halfPage);
        }
        bool inside = skipInteriorClip;
        for (uint i = 0u; i < 3u; ++i) {
            polygon[i].position = vsmProjectClip(triangle.vertices[i].position.xyz, projection);
            polygon[i].uv = triangle.vertices[i].uv.xy;
            if (skipInteriorClip) {
                float4 clip = polygon[i].position;
                inside = inside && all(clip.xy >= bounds.xy * clip.w) && all(clip.xy <= bounds.zw * clip.w) &&
                    (projection.metric.w <= 0.5 || (clip.z >= 0.0 && clip.z <= clip.w));
            }
        }
        material = triangle.materialIndex;
        physicalPage = page.x;
        uint count = inside ? 3u : vsmClipProjection(polygon, bounds, projection.metric.w > 0.5);
        return count >= 3u ? count : 0u;
    }
    return 0u;
}

ShadowRasterVertex vsmRasterVertex(ShadowClipVertex vertex, float4 bounds, uint physicalPage, uint material, ShadowPush push)
{
    float2 pagePosition = saturate((vertex.position.xy / vertex.position.w - bounds.xy) / (bounds.zw - bounds.xy));
    ShadowRasterVertex result;
    result.position = float4(vsmAtlasPosition(pagePosition, physicalPage, push) * vertex.position.w,
        vertex.position.z, vertex.position.w);
    result.uv = vertex.uv;
    result.materialIndex = material;
    return result;
}

void vsmRecordRasterOutput(ShadowPush push, uint triangleCount)
{
    vsmWorkAdd(push, kVSMWorkRasterOutputPrimitives, triangleCount);
#if VSM_WORKLOAD_DIAGNOSTICS
    if ((push.cacheMode & 16u) != 0u) {
        vsmCacheAdd(push, (push.rasterCoarse & 2u) != 0u ? kVSMCacheStaticOutputPrimitives : kVSMCacheDynamicOutputPrimitives, triangleCount);
    }
#endif
}

groupshared ShadowClipVertex g_vsmPolygon[9];
groupshared uint g_vsmVertexCount;
groupshared uint g_vsmMaterial;
groupshared uint g_vsmPhysicalPage;
groupshared float4 g_vsmBounds;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[9], out indices uint3 triangles[7], uniform ShadowPush push)
{
    if (lane == 0u) {
        vsmWorkAdd(push, (push.rasterCoarse & 1u) != 0u ? kVSMWorkRasterCoarseGroups : kVSMWorkRasterFineGroups, 1u);
        uint triangleIndex = push.workBase + group.x;
        uint virtualPage = push.totalPages - 1u;
        if ((push.rasterCoarse & 1u) == 0u) {
            StructuredBuffer<uint2> work = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
            uint2 item = work[push.workBase + group.x];
            triangleIndex = item.x;
            virtualPage = item.y;
        }
        ShadowClipVertex polygon[9];
        uint material, physicalPage;
        float4 bounds;
        uint count = vsmRasterPolygon(triangleIndex, virtualPage, push, false, polygon, material, physicalPage, bounds);
        g_vsmVertexCount = count;
        g_vsmMaterial = material;
        g_vsmPhysicalPage = physicalPage;
        g_vsmBounds = bounds;
        for (uint i = 0u; i < count; ++i) { g_vsmPolygon[i] = polygon[i]; }
    }
    GroupMemoryBarrierWithGroupSync();
    uint vertexCount = g_vsmVertexCount;
    uint triangleCount = vertexCount >= 3u ? vertexCount - 2u : 0u;
    if (lane == 0u) { vsmRecordRasterOutput(push, triangleCount); }
    SetMeshOutputCounts(vertexCount, triangleCount);
    if (lane < vertexCount) {
        vertices[lane] = vsmRasterVertex(g_vsmPolygon[lane], g_vsmBounds, g_vsmPhysicalPage, g_vsmMaterial, push);
    }
    if (lane < triangleCount) { triangles[lane] = uint3(0u, lane + 1u, lane + 2u); }
}

#if VSM_CLUSTER_PAGE_RASTER
// A Build ticket is one nonempty 16-slot quarter of an original 64-triangle
// cluster. Six clip planes can expand each source triangle to nine vertices.
groupshared uint g_vsmPackedOffsets[17];

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterPackedMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[144], out indices uint3 triangles[112], uniform ShadowPush push)
{
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    uint4 ticket = work[1u + push.workBase + group.x];
    uint bits = ticket.z != 0u ? ticket.z : ticket.w;
    uint quarter = (ticket.z != 0u ? 0u : 32u) + ((bits & 0xffffu) != 0u ? 0u : 16u);
    uint slotMask = bits >> (quarter & 31u);
    ShadowClipVertex polygon[9];
    uint vertexCount = 0u, material = 0u, physicalPage = 0u;
    float4 bounds = float4(0.0);
    if (lane < 16u) {
        if ((slotMask & (1u << lane)) != 0u) {
            vertexCount = vsmRasterPolygon(ticket.x + quarter + lane, ticket.y, push, true,
                polygon, material, physicalPage, bounds);
        }
        // A packed sum gives both output offsets; max total 144 cannot carry.
        g_vsmPackedOffsets[lane] = vertexCount | (uint(max(int(vertexCount) - 2, 0)) << 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < 16u; ++i) {
            uint counts = g_vsmPackedOffsets[i];
            g_vsmPackedOffsets[i] = sum;
            sum += counts;
        }
        g_vsmPackedOffsets[16] = sum;
        vsmWorkAdd(push, kVSMWorkRasterFineGroups, 1u);
        vsmRecordRasterOutput(push, sum >> 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint totals = g_vsmPackedOffsets[16];
    SetMeshOutputCounts(totals & 0xffffu, totals >> 16u);
    if (vertexCount != 0u) {
        uint offsets = g_vsmPackedOffsets[lane];
        uint firstVertex = offsets & 0xffffu;
        for (uint i = 0u; i < vertexCount; ++i) {
            vertices[firstVertex + i] = vsmRasterVertex(polygon[i], bounds, physicalPage, material, push);
        }
        for (uint i = 2u; i < vertexCount; ++i) {
            triangles[(offsets >> 16u) + i - 2u] = uint3(firstVertex, firstVertex + i - 1u, firstVertex + i);
        }
    }
}
#endif

#if VSM_PAGE_WINDOW_RASTER
// Each lane clips to one of four consecutive virtual pages. A directional
// triangle clipped by four XY planes produces at most seven vertices.
groupshared uint g_vsmWindowCounts[4];
groupshared ShadowClipVertex g_vsmWindowPolygon[28];
groupshared float4 g_vsmWindowBounds[4];
groupshared uint2 g_vsmWindowPageMaterial[4];

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterWindowMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[28], out indices uint3 triangles[20], uniform ShadowPush push)
{
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    uint ticketIndex = 1u + push.workBase + group.x;
    uint4 ticket = work[ticketIndex];
    ShadowClipVertex polygon[9];
    uint count = 0u, physicalPage = 0u, material = 0u;
    float4 bounds = float4(0.0);
    if (lane < 4u) {
        if ((ticket.z & (1u << lane)) != 0u) {
            StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
            uint virtualPage = ticket.y + lane;
            uint4 page = pages[virtualPage];
            uint triangleIndex = ticket.x;
            bool mergeStatic;
            if (vsmRasterPageLayerActive(triangleIndex, page, push, mergeStatic) &&
                virtualShadowMapRasterLayerContains(push.geometry, triangleIndex)) {
                StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
                ShadowProjection projection = projections[page.z];
                // Preserve the scalar raster's projection and clip control flow.
                // The producer supplies decoded world positions. The final
                // depth dot stays in the mesh stage alongside canonical XY.
                bounds = vsmPageBounds(virtualPage, projection);
                if (projection.metric.w < 0.5 && (push.rasterCoarse & 1u) == 0u) {
                    bounds = vsmDirectionalPageBounds(virtualPage, projection);
                }
                uint payload = 1u + push.workCapacity + 4u * ticket.w;
                uint4 attributes = work[payload + 3u];
                for (uint i = 0u; i < 3u; ++i) {
                    float4 source = asfloat(work[payload + i]);
                    polygon[i].position = vsmProjectClip(source.xyz, projection);
                    polygon[i].uv = float2(source.w, asfloat(attributes[i]));
                }
                material = attributes.w;
                physicalPage = page.x;
                count = vsmClipProjection(polygon, bounds, projection.metric.w > 0.5);
                if (count < 3u) { count = 0u; }
#if VSM_WORKLOAD_DIAGNOSTICS
                if (mergeStatic) { vsmWorkAdd(push, kVSMWorkRasterStaticMergedTriangles, 1u); }
#endif
            }
        }
        g_vsmWindowCounts[lane] = count | (uint(max(int(count) - 2, 0)) << 16u);
        g_vsmWindowBounds[lane] = bounds;
        g_vsmWindowPageMaterial[lane] = uint2(physicalPage, material);
        for (uint i = 0u; i < count; ++i) {
            g_vsmWindowPolygon[lane * 7u + i] = polygon[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint pageLane = lane >> 3u;
    uint vertex = lane & 7u;
    uint total = 0u, offset = 0u;
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        uint counts = g_vsmWindowCounts[i];
        total += counts;
        if (i < pageLane) { offset += counts; }
    }
    SetMeshOutputCounts(total & 0xffffu, total >> 16u);
    if (lane == 0u) {
        vsmWorkAdd(push, kVSMWorkRasterFineGroups, 1u);
        vsmRecordRasterOutput(push, total >> 16u);
    }
    uint first = offset & 0xffffu;
    uint pageCounts = g_vsmWindowCounts[pageLane];
    if (vertex < (pageCounts & 0xffffu)) {
        // Match the scalar path's rounded clipping/bounds handoff. In
        // particular, 1/axis is inexact for non-power-of-two clipmaps.
        uint2 pageMaterial = g_vsmWindowPageMaterial[pageLane];
        vertices[first + vertex] = vsmRasterVertex(g_vsmWindowPolygon[pageLane * 7u + vertex],
            g_vsmWindowBounds[pageLane], pageMaterial.x, pageMaterial.y, push);
    }
    if (vertex < (pageCounts >> 16u)) {
        triangles[(offset >> 16u) + vertex] = uint3(first, first + vertex + 1u, first + vertex + 2u);
    }
}
#endif

[shader("fragment")]
void vsmDepthFragmentMain(ShadowRasterVertex input, bool frontFace : SV_IsFrontFace, uniform ShadowPush push)
{
    StructuredBuffer<ShadowMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<ShadowMaterial>>(push.materials));
    ShadowMaterial material = materials[input.materialIndex];
    if (!frontFace && material.resources.y == 0u) { discard; }
    if (material.alpha.z < 0.5) { return; }
    CoverageContext context;
    context.uv = input.uv;
    context.baseAlpha = material.alpha.x;
    context.cutoff = material.alpha.y;
    context.alphaMode = material.alpha.z;
    CoverageProgram program;
    program.inputs = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.coverageInputs));
    MaterialInstanceRef instance = coverageInstanceRef(input.materialIndex);
    if (program.usesBaseAlpha(instance) && material.resources.x != 0xffffffffu) {
        Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(material.resources.x));
        SamplerState sampler = resolveUniform(SamplerHandle(push.sampler));
        float3 uv = float3(input.uv, 1.0);
        float2 textureUV = float2(dot(material.uvX.xyz, uv), dot(material.uvY.xyz, uv));
        context.baseAlpha *= texture.Sample(sampler, textureUV).a;
    }
    if (!program.evaluateCoverage(context, instance).accepted) { discard; }
}

#if VSM_ONE_PASS_PROJECTION
// Common filter settings stay in push constants. Only face-specific values
// come from this table; its 84-byte layout matches ShadowProjectionJob in C++.
struct ShadowProjectionJob
{
    uint pages, statistics, projections, atlas;
    uint filterMode, shadow, debug, physicalPagesX;
    uint clipmapCount, enabled, lightType, projectionFace;
    uint pointFaces, projectionFlags, samplePhase, view;
    float lightRange, sourceRadius, angularRadius, lodScreenRayLength;
    uint receiverGeometry;
};

#endif

#if VSM_LIGHT_GRID_PROJECTION
struct VSMProjectionReceiver
{
    ViewConstants view;
    float3 position;
    float linearDepth;
    bool valid;
};
#endif

bool vsmProjectReceiver(uint3 id, ShadowPush push, out float visibility, out float4 debugValue, out uint3 filterCounts
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
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
#if !VSM_ONE_PASS_PROJECTION
    VSMReceiverBasis basis = useGroupCache ? vsmGetGroupReceiverBasis() : vsmMakeReceiverBasis(views[0]);
#else
    ViewConstants view = views[0];
    VSMReceiverBasis basis = vsmMakeReceiverBasis(view);
#endif
    if (any(id.xy >= uint2(basis.viewportSize))) { return false; }
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
#if VSM_DIRECTIONAL_PROJECTION
    const bool sharedPointLayer = false;
#else
    bool sharedPointLayer = (push.projectionFlags & 1u) != 0u && push.lightType == 2u;
#endif
    if (!vsmReceiver(int2(id.xy), basis, depth, position, linearDepth)) {
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
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    float3 normal = vsmReceiverNormal(int2(id.xy), basis, depth, position, linearDepth);
    uint receiverLevel = vsmReceiverLevel(position, projections, push,
        vsmProjectedReceiverFootprint(position, normal, linearDepth, basis, proj0, push), useGroupCache);
    uint selectedLevel = push.clipmapCount;
    uint firstLevel = 0xffffffffu;
    uint physicalPage = 0xffffffffu;
    uint2 virtualTexel = uint2(0u);
    float3 projected = float3(0.0);
    // Any fine-work overflow invalidates the entire partial fine atlas. The
    // independent coarse draw always traverses every caster slot.
    bool allowClipmaps;
#if !VSM_ONE_PASS_PROJECTION
    if (useGroupCache && push.lightType == 0u) {
        allowClipmaps = gVsmDirectionalUniform.allowClipmaps != 0u;
    } else
#endif
    {
        allowClipmaps = statistics[5] == 0u;
    }
    if (allowClipmaps) {
        for (uint level = receiverLevel; level < push.clipmapCount; ++level) {
            ShadowProjection projection = vsmGetCachedProjection(projections, level, useGroupCache);
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            if (firstLevel == 0xffffffffu) { firstLevel = level; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint page = vsmVirtualPage(texel, projection, push.pageSize);
            uint mappedPhysical, mappedLevel;
            if (vsmLookupPropagated(position, float2(projection.rowX.w, projection.rowY.w),
                    page, p, texel, projections, pages, push, level,
                    mappedPhysical, mappedLevel, useGroupCache)) {
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
        physicalPage = 0u;
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
    ShadowProjection projection = vsmGetCachedProjection(projections, selectedLevel, useGroupCache);
    if (push.lightType == 0u) {
#if !VSM_ONE_PASS_PROJECTION
        float3 toLight = useGroupCache ? gVsmDirectionalUniform.toLight : normalize(proj0.rowZ.xyz);
        float sinAlpha = useGroupCache ? gVsmDirectionalUniform.sinAlpha : max(sin(push.angularRadius), 0.1);
#else
        float3 toLight = normalize(proj0.rowZ.xyz);
        float sinAlpha = max(sin(push.angularRadius), 0.1);
#endif
        if (dot(normal, toLight) < -sinAlpha) {
            visibility = 0.0;
            return true;
        }
    }
    if (push.filterMode != 0u) {
        ShadowPush trace = push;
        trace.rayLength = vsmSearchLength(position, linearDepth, basis, push);
        visibility = vsmFilterVisibility(position, normal, receiverLevel, id.xy, projections, trace, filterCounts, useGroupCache);
        return true;
    }
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
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    float storedDepth = atlas.Load(int3(texel, 0));
    vsmWorkAdd(push, kVSMWorkProjectionAtlasLoads, 1u);
    if (push.lodScreenRayLength > 0.0 && storedDepth != 0.0 && storedDepth > compareDepth) {
        float depthScale = length(projection.rowZ.xyz);
        float bias = projection.metric.z / depthScale;
        float blockerDistance = (storedDepth - compareDepth) / depthScale + bias;
        ViewConstants view = views[0];
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

// Ownership and filtering finish before either output is stored. Receiver
// reconstruction retains the original per-job arithmetic in both paths.
void vsmProjectAndStore(uint3 id, ShadowPush push, out uint3 filterCounts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    , out uint projectionCounts
#endif
    , bool useGroupCache = false
)
{
    float visibility;
    float4 debugValue;
    if (!vsmProjectReceiver(id, push, visibility, debugValue, filterCounts
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

#if VSM_OPTIMIZE_PROJECTION_STATISTICS
// Slots 0/1/2 retain filter 30/31/9; the other five combine one-bit receiver events.
static const uint kVsmProjectionCounterCount = 8u;
#else
static const uint kVsmProjectionCounterCount = 3u;
#endif
uint2 vsmMortonPixel(uint2 dispatchPixel)
{
    uint lane = (dispatchPixel.x & 7u) + ((dispatchPixel.y & 7u) << 3u);
    uint2 xy = uint2((lane & 1u) | ((lane >> 1u) & 2u) | ((lane >> 2u) & 4u),
        ((lane >> 1u) & 1u) | ((lane >> 2u) & 2u) | ((lane >> 3u) & 4u));
    return (dispatchPixel & ~7u) + xy;
}

#if !VSM_ONE_PASS_PROJECTION
groupshared uint gVsmFilterCounts[kVsmProjectionCounterCount];

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmProjectMain(uint3 id : SV_DispatchThreadID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    id.xy = vsmMortonPixel(id.xy);
    if (push.enabled != 0u && lane <= push.clipmapCount && lane < kVSMMaxCachedProjections) {
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        gVsmProjections[lane] = projections[lane];
    }
    if (push.filterMode != 0u && lane < kVsmProjectionCounterCount) {
        gVsmFilterCounts[lane] = 0u;
    }
    if (lane == 0u) {
        StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
        ViewConstants view = views[0];
        VSMReceiverBasis b = vsmMakeReceiverBasis(view);
        gVsmReceiverBasis.eye = b.eye;
        gVsmReceiverBasis.forward = b.forward;
        gVsmReceiverBasis.right = b.right;
        gVsmReceiverBasis.up = b.up;
        gVsmReceiverBasis.rightAspect = b.rightAspect;
        gVsmReceiverBasis.tanHalfFov = b.tanHalfFov;
        gVsmReceiverBasis.orthoHalfHeight = b.orthoHalfHeight;
        gVsmReceiverBasis.jitter = b.jitter;
        gVsmReceiverBasis.invViewport = b.invViewport;
        gVsmReceiverBasis.ndcScale = b.ndcScale;
        gVsmReceiverBasis.viewportSize = b.viewportSize;
        gVsmReceiverBasis.nearPlane = b.nearPlane;
        gVsmReceiverBasis.farPlane = b.farPlane;
        gVsmReceiverBasis.farMinusNear = b.farMinusNear;
        gVsmReceiverBasis.nearTimesFar = b.nearTimesFar;
        gVsmReceiverBasis.isOrtho = b.isOrtho ? 1u : 0u;
        gVsmReceiverBasis.reversed = b.reversed ? 1u : 0u;

#if VSM_DIRECTIONAL_PROJECTION
        if (push.enabled != 0u && push.lightType == 0u) {
            StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
            ShadowProjection proj0 = projections[0u];
            ShadowProjection coarseProj = projections[push.clipmapCount];
            float3 toLight = normalize(proj0.rowZ.xyz);
            float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
            float3 up = cross(toLight, right);
            gVsmDirectionalUniform.toLight = toLight;
            gVsmDirectionalUniform.right = right;
            gVsmDirectionalUniform.up = up;
            gVsmDirectionalUniform.normX = normalize(proj0.rowX.xyz);
            gVsmDirectionalUniform.normY = normalize(proj0.rowY.xyz);
            gVsmDirectionalUniform.biasZ = proj0.metric.z;
            gVsmDirectionalUniform.sinAlpha = max(sin(push.angularRadius), 0.1);
            gVsmDirectionalUniform.tanAngularRadius = tan(push.angularRadius);
            gVsmDirectionalUniform.coarseInvResolution = 2.0 / float(push.pageSize);
            gVsmDirectionalUniform.receiverInvResolution = 2.0 / float(push.virtualPagesPerAxis * push.pageSize);

            StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
            gVsmDirectionalUniform.allowClipmaps = (statistics[5] == 0u) ? 1u : 0u;

            StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
            uint4 coarseEntry = pages[coarseProj.page.x];
            gVsmDirectionalUniform.coarseValid = ((coarseEntry.y & 2u) != 0u && (coarseEntry.w & 1u) != 0u) ? 1u : 0u;
            gVsmDirectionalUniform.coarsePhysicalCoord = vsmPhysicalPageCoord(coarseEntry.x, push.physicalPagesX);

            bool isPow2PageSize = (push.pageSize & (push.pageSize - 1u)) == 0u && push.pageSize != 0u;
            bool isPow2PhysicalX = (push.physicalPagesX & (push.physicalPagesX - 1u)) == 0u && push.physicalPagesX != 0u;
            bool isPow2VirtualPages = (push.virtualPagesPerAxis & (push.virtualPagesPerAxis - 1u)) == 0u && push.virtualPagesPerAxis != 0u;
            gVsmDirectionalUniform.powerOfTwoLayout = (isPow2PageSize && isPow2PhysicalX && isPow2VirtualPages) ? 1u : 0u;
            gVsmDirectionalUniform.pageSizeShift = isPow2PageSize ? uint(firstbitlow(push.pageSize)) : 0u;
            gVsmDirectionalUniform.pageSizeMask = isPow2PageSize ? (push.pageSize - 1u) : 0u;
            gVsmDirectionalUniform.physicalPagesXShift = isPow2PhysicalX ? uint(firstbitlow(push.physicalPagesX)) : 0u;
            gVsmDirectionalUniform.physicalPagesXMask = isPow2PhysicalX ? (push.physicalPagesX - 1u) : 0u;
            gVsmDirectionalUniform.virtualPagesPerAxisShift = isPow2VirtualPages ? uint(firstbitlow(push.virtualPagesPerAxis)) : 0u;
        }
#endif
    }
    GroupMemoryBarrierWithGroupSync();

    uint3 counts;
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    uint projectionCounts;
#endif
    if (push.filterMode == 0u) {
        vsmProjectAndStore(id, push, counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
            , projectionCounts
#endif
            , true
        );
        return;
    }
    // Early returns, including viewport tails and point-face rejection, stay
    // inside the receiver helper so every lane reaches the group barriers.
    vsmProjectAndStore(id, push, counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
            , true
    );
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    // All events were recorded at the original sites, including counts before
    // compact point-face ownership and disabled-light rejection.
    counts.z += (projectionCounts >> 3u) & 1u;
    if ((projectionCounts & (1u << 0u)) != 0u) { InterlockedAdd(gVsmFilterCounts[3], 1u); }
    if ((projectionCounts & (1u << 1u)) != 0u) { InterlockedAdd(gVsmFilterCounts[4], 1u); }
    if ((projectionCounts & (1u << 2u)) != 0u) { InterlockedAdd(gVsmFilterCounts[5], 1u); }
    if ((projectionCounts & (1u << 4u)) != 0u) { InterlockedAdd(gVsmFilterCounts[6], 1u); }
    if ((projectionCounts & (1u << 5u)) != 0u) { InterlockedAdd(gVsmFilterCounts[7], 1u); }
#endif
    if (counts.x != 0u) { InterlockedAdd(gVsmFilterCounts[0], counts.x); }
    if (counts.y != 0u) { InterlockedAdd(gVsmFilterCounts[1], counts.y); }
    if (counts.z != 0u) { InterlockedAdd(gVsmFilterCounts[2], counts.z); }
    GroupMemoryBarrierWithGroupSync();
    if (lane < kVsmProjectionCounterCount) {
        uint count = gVsmFilterCounts[lane];
        if (count != 0u) {
            // Attribute cross-face point queries to the receiver's dispatch,
            // just like the original per-lookup counters. Preserve all slots.
            RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
            uint index = lane == 0u ? 30u : lane == 1u ? 31u : 9u;
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
            if (lane >= 3u) {
                index = lane == 3u ? 8u : lane == 4u ? 6u : lane == 5u ? 29u : lane == 6u ? 7u : 13u;
            }
#endif
            InterlockedAdd(statistics[index], count);
        }
    }
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

void vsmProjectOnePassJob(uint3 id, ShadowPush common, ShadowProjectionJob job
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
    push.samplePhase = job.samplePhase; push.view = job.view; push.lightRange = job.lightRange;
    push.sourceRadius = job.sourceRadius; push.angularRadius = job.angularRadius;
    uint3 counts = uint3(0u);
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    uint projectionCounts = 0u;
#endif
#if VSM_LIGHT_GRID_PROJECTION
    if (member) {
#endif
    vsmProjectAndStore(id, push, counts
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
        , projectionCounts
#endif
    );
#if VSM_LIGHT_GRID_PROJECTION
    }
#endif
    if (push.filterMode == 0u) { return; }
#if VSM_OPTIMIZE_PROJECTION_STATISTICS
    counts.z += (projectionCounts >> 3u) & 1u;
#endif
    // Integer subgroup reductions preserve the per-face totals without a
    // workgroup barrier between jobs or shared storage for each face.
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
    StructuredBuffer<ShadowProjectionJob> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjectionJob>>(common.geometry));
#if VSM_LIGHT_GRID_PROJECTION
    StructuredBuffer<VirtualShadowMapProjectionGrid> contexts = resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjectionGrid>>(common.arguments));
    VirtualShadowMapProjectionGrid context = contexts[0];
    VSMProjectionReceiver receiver;
    receiver.view = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(common.view))[0];
    if (any(id.xy >= uint2(receiver.view.current.viewport.yz))) { return; }
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(common.depth));
    receiver.valid = vsmReceiver(int2(id.xy), receiver.view, depth, receiver.position, receiver.linearDepth);
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
                vsmProjectOnePassJob(id, common, jobs[range.first + face], member);
            }
        }
    }
    vsmProjectionGridAdd(context.diagnostics, kVSMGridWaveLocalJobs, WaveIsFirstLane() ? waveJobs : 0u);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridSkippedWaveLocalJobs, WaveIsFirstLane() ? context.localJobCount - waveJobs : 0u);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridProjectedLaneJobs, laneJobs);
    vsmProjectionGridAdd(context.diagnostics, kVSMGridPossibleLaneJobs, context.localJobCount);
#else
    // Baseline keeps full-screen, job-uniform traversal for the controlled A/B.
    for (uint jobIndex = 0u; jobIndex < common.triangleCount; ++jobIndex) {
        ShadowProjectionJob job = jobs[jobIndex];
        vsmProjectOnePassJob(id, common, job);
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (common.arguments != 0xffffffffu) {
        VirtualShadowMapProjectionGrid context = resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjectionGrid>>(common.arguments))[0];
        ViewConstants view = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(common.view))[0];
        bool inViewport = all(id.xy < uint2(view.current.viewport.yz));
        bool activeWave = WaveActiveAnyTrue(inViewport);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridWaveLocalJobs, WaveIsFirstLane() && activeWave ? context.localJobCount : 0u);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridProjectedLaneJobs, inViewport ? context.localJobCount : 0u);
        vsmProjectionGridAdd(context.diagnostics, kVSMGridPossibleLaneJobs, inViewport ? context.localJobCount : 0u);
    }
#endif
#endif
}
#endif
