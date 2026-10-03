import ShaderCore;
using Metallic;

// Production stream raster for fixed wave32 devices; 64/128 are A/B references.
#include "GPUDrivenStreamWorkRaster.slang"

void rasterStreamClusterGroup(uint3 group, uint lane, GPUDrivenStreamAssetPush push, uint groupSize)
{
    StructuredBuffer<uint> bins =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.hybridClusterBuffer));
    uint record;
    if (!hybridBinRecord(bins, kHybridSoftwareBin, hybridFlatGroup(group), record)) { return; }
    if (lane < WaveGetLaneCount()) {
        bool valid = loadStreamSoftwareWave(record, push);
        if (lane == 0u) { g_StreamSoftwareValid = valid ? 1u : 0u; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamSoftwareValid == 0u) { return; }
    for (uint vertexIndex = lane; vertexIndex < g_StreamSoftware.vertexCount; vertexIndex += groupSize) {
        HybridScreenVertex vertex = hybridPrepareVertex(streamSoftwareClip(vertexIndex, push), float2(bins[6], bins[7]), bins[10]);
        g_StreamRasterWords[vertexIndex] = uint4(uint2(vertex.position), asuint(vertex.depth), 0u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint triangle = lane; triangle < g_StreamSoftware.triangleCount; triangle += groupSize) {
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

// The host gates these entries on fixed subgroup size 32.
[shader("compute")]
[numthreads(32, 1, 1)]
void streamClusterRasterGroup32Main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    rasterStreamClusterGroup(group, lane, push, 32u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void streamClusterRasterGroup64Main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    rasterStreamClusterGroup(group, lane, push, 64u);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterGroup128Main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    rasterStreamClusterGroup(group, lane, push, 128u);
}
