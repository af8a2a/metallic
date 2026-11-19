#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_MARK_PAGE_SHIFT
#define VSM_MARK_PAGE_SHIFT -1
#endif
#ifndef VSM_COMPACT_BUILD
#define VSM_COMPACT_BUILD 0
#endif
#ifndef VSM_INSTANCE_BUILD
#define VSM_INSTANCE_BUILD 0
#endif
#ifndef VSM_DIRTY_PAGE_HIERARCHY
#define VSM_DIRTY_PAGE_HIERARCHY 0
#endif
#ifndef VSM_COMPACT_SHADOW_VIEWS
#define VSM_COMPACT_SHADOW_VIEWS 0
#endif
#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif

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
    uint rayCount, samplesPerRay, receiverLOD, projectionFlags;
    float rayLengthScale, maxRayAngle;
    uint samplePhase, sampleReserved;
    float lodScreenRayLength;
    uint receiverGeometry;
    uint physicalPool, poolFace;
};

// sampleReserved names a 32-word diagnostic prefix followed by per-face metadata and counters.
uint vsmCacheMetadata(ShadowPush push, uint word)
{
    StructuredBuffer<uint> data = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.sampleReserved));
    return data[word];
}

bool vsmRetainsSharedPage(uint slot, ShadowPush push)
{
#if VSM_SHARED_PHYSICAL_POOL
    if ((push.cacheMode & kVSMCacheRestoreInactive) != 0u) {
        StructuredBuffer<uint2> history = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(
            vsmCacheMetadata(push, kVSMCachePreviousSlotAssignments)));
        uint2 assigned = history[push.physicalPageCount + slot];
        uint serialHigh = vsmCacheMetadata(push, kVSMCachePoolSerialHigh);
        return assigned.y < serialHigh ||
            (assigned.y == serialHigh && assigned.x <= vsmCacheMetadata(push, kVSMCachePoolSerialLow));
    }
#endif
    return true;
}

uint vsmFineViewForLane(ShadowPush push, uint lane)
{
#if VSM_COMPACT_SHADOW_VIEWS
    if (lane >= push.clipmapCount) { return push.clipmapCount; }
    uint mask = vsmCacheMetadata(push, kVSMActiveFineViewMask);
    uint prefix = (1u << (lane + 1u)) - 1u;
    if ((mask & prefix) == prefix) { return lane; }
    if (lane >= countbits(mask)) { return push.clipmapCount; }
    // Keep the original projection and page-table ordinal in downstream masks.
    for (uint view = 0u; view < lane; ++view) { mask &= mask - 1u; }
    return uint(firstbitlow(mask));
#else
    return lane;
#endif
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
    uint waveCounter = WaveReadLaneFirst(counter);
    if (WaveActiveAllTrue(counter == waveCounter)) {
        uint total = WaveActiveSum(value);
        if (WaveIsFirstLane() && total != 0u) {
            RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
            uint word = 32u + waveCounter * 2u;
            uint previous;
            InterlockedAdd(statistics[word], total, previous);
            if (previous > 0xffffffffu - total) { InterlockedAdd(statistics[word + 1u], 1u); }
        }
        return;
    }

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

