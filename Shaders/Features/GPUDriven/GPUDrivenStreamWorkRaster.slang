// Separate module keeps the histogram/remap shared allocation out of the
// measured cooperative reference and other stream raster entrypoints.
#include "GPUDrivenStreamAsset.slang"
#include "StreamRasterWorkBins.slang"

void rasterStreamClusterWorkBins(uint3 group, uint lane, GPUDrivenStreamAssetPush push, bool reorder)
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
    rasterStreamClusterWorkBins(group, lane, push, true);
}

// Measurement control: same vertex setup/raster arithmetic without reordering.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterWorkControlMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    if (WaveGetLaneCount() < 32u) { rasterStreamCluster(group, lane, push, false, false); return; }
    rasterStreamClusterWorkBins(group, lane, push, false);
}
