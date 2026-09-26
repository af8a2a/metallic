// Separate module keeps the histogram/remap shared allocation out of the
// measured cooperative reference and other stream raster entrypoints.
#include "GPUDrivenStreamAsset.slang"
#include "StreamRasterWorkBins.slang"
#ifndef METALLIC_WORK_CONTROL_TRACE
#define METALLIC_WORK_CONTROL_TRACE 0
#endif
#if METALLIC_WORK_CONTROL_TRACE
import ShaderTrace;
bool workControlTraceSelected(uint3 group, uint lane)
{
    return all(group == uint3(TRACE_GROUP_X, TRACE_GROUP_Y, TRACE_GROUP_Z)) && lane == TRACE_LOCAL;
}
ShaderTraceIdentity workControlTraceIdentity(uint3 group, uint lane)
{
    ShaderTraceIdentity id;
    id.session = uint2(TRACE_SESSION_LO, TRACE_SESSION_HI);
    id.run = uint2(TRACE_RUN_LO, TRACE_RUN_HI);
    id.dispatch = uint2(TRACE_DISPATCH_LO, TRACE_DISPATCH_HI);
    id.site = 2; id.group = group; id.localIndex = lane;
    return id;
}
#endif

void rasterStreamClusterWorkBins(uint3 group, uint lane, GPUDrivenStreamAssetPush push, bool reorder
#if METALLIC_WORK_CONTROL_TRACE
    , inout uint3 traceCounts
#endif
)
{
    DescriptorHandle<StructuredBuffer<uint>> bins =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.hybridClusterBuffer, 0));
    uint record;
    if (!hybridBinRecord(bins, kHybridSoftwareBin, hybridFlatGroup(group), record)) { return; }
    if (lane < WaveGetLaneCount()) {
        bool valid = loadStreamSoftwareWave(record, push);
        if (lane == 0u) { g_StreamSoftwareValid = valid ? 1u : 0u; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamSoftwareValid == 0u) { return; }
    if (lane < g_StreamSoftware.vertexCount) {
        HybridScreenVertex vertex = hybridPrepareVertex(streamSoftwareClip(lane, push), float2(bins[6], bins[7]), bins[10]);
        // Reuse the existing raw-word scratch. Do not store snapped signed
        // coordinates as float bits: NaN canonicalization can change them.
        g_StreamRasterWords[lane] = uint4(uint2(vertex.position), asuint(vertex.depth), 0u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint triangle = lane;
    if (reorder) {
        uint bucket = 3u;
        if (lane < g_StreamSoftware.triangleCount) {
            uint3 tri = streamSoftwareTriangle(lane, push);
            bucket = streamRasterWorkBucket(asint(g_StreamRasterWords[tri.x].xy), asint(g_StreamRasterWords[tri.y].xy),
                asint(g_StreamRasterWords[tri.z].xy), (g_StreamSoftware.instanceFlags & 2u) != 0u, bins[6], bins[7], bins[10]);
        }
        uint work = streamRasterReorderWork(bucket, lane);
        if ((work >> 7u) == 3u) { return; }
        triangle = work & 127u;
    }
    if (triangle < g_StreamSoftware.triangleCount) {
        uint3 tri = streamSoftwareTriangle(triangle, push);
        HybridScreenVertex a, b, c;
        a.position = asint(g_StreamRasterWords[tri.x].xy); a.depth = asfloat(g_StreamRasterWords[tri.x].z);
        b.position = asint(g_StreamRasterWords[tri.y].xy); b.depth = asfloat(g_StreamRasterWords[tri.y].z);
        c.position = asint(g_StreamRasterWords[tri.z].xy); c.depth = asfloat(g_StreamRasterWords[tri.z].z);
#if METALLIC_WORK_CONTROL_TRACE
        if (!reorder && workControlTraceSelected(group, lane)) {
            ++traceCounts.x;
            if (!TRACE_PREDICATE_ENABLED || triangle == TRACE_TRIANGLE_ID) {
                ++traceCounts.y;
                if (traceCounts.z < TRACE_MAX_RECORDS - 2) {
                    ShaderTraceIdentity id = workControlTraceIdentity(group, lane);
                    printf("MTS1 %u %u %u %u %u %u 2 1 %u %u %u %u %u 13 %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
                        id.session.x, id.session.y, id.run.x, id.run.y, id.dispatch.x, id.dispatch.y,
                        group.x, group.y, group.z, lane, traceCounts.z + 1,
                        g_StreamSoftware.recordIndex, triangle, g_StreamSoftware.instanceFlags, g_StreamSoftware.triangleCount,
                        asuint(a.position.x), asuint(a.position.y), asuint(a.depth),
                        asuint(b.position.x), asuint(b.position.y), asuint(b.depth),
                        asuint(c.position.x), asuint(c.position.y), asuint(c.depth));
                    ++traceCounts.z;
                }
            }
        }
#endif
        hybridRasterPreparedTriangle(a, b, c,
            packVisibilityId(g_StreamSoftware.recordIndex, triangle), (g_StreamSoftware.instanceFlags & 2u) != 0u,
            bins[6], bins[7], bins[9], bins[10], bins[11], false);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterWorkBinsMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    if (WaveGetLaneCount() < 32u) { rasterStreamCluster(group, lane, push, false, false); return; }
#if METALLIC_WORK_CONTROL_TRACE
    uint3 traceCounts = 0;
    rasterStreamClusterWorkBins(group, lane, push, true, traceCounts);
#else
    rasterStreamClusterWorkBins(group, lane, push, true);
#endif
}

// Measurement control: same vertex setup/raster arithmetic without reordering.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterWorkControlMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
#if METALLIC_WORK_CONTROL_TRACE
    uint3 traceCounts = 0;
    ShaderTraceIdentity id = workControlTraceIdentity(group, lane);
    bool selected = workControlTraceSelected(group, lane);
    if (selected) { shaderTraceBegin(id); }
    if (WaveGetLaneCount() < 32u) { rasterStreamCluster(group, lane, push, false, false); }
    else { rasterStreamClusterWorkBins(group, lane, push, false, traceCounts); }
    if (selected) {
        shaderTraceWords5(id, 2, traceCounts.z + 1,
            uint4(traceCounts, WaveGetLaneCount() < 32u ? 1u : 0u), traceCounts.y > traceCounts.z ? 1u : 0u);
    }
#else
    if (WaveGetLaneCount() < 32u) { rasterStreamCluster(group, lane, push, false, false); return; }
    rasterStreamClusterWorkBins(group, lane, push, false);
#endif
}
