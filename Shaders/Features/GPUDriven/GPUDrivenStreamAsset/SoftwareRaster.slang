#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Software raster scratch, triangle rasterization and raster entry points.
public groupshared StreamRasterCluster g_StreamRasterCluster;
public groupshared float4 g_StreamRasterClip[128];
// Raw words avoid interpreting negative snapped coordinates as float NaNs.
public groupshared uint4 g_StreamRasterWords[128];
public groupshared uint g_StreamBinVisible;
public groupshared uint g_StreamBinHardware;

public void rasterStreamCluster(uint3 group, uint lane, GPUDrivenStreamAssetPush push, bool prepared, bool incrementalDepth)
{
    StructuredBuffer<uint> bins =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.hybridClusterBuffer));
    uint record;
    if (!hybridBinRecord(bins, kHybridSoftwareBin, hybridFlatGroup(group), record)) { return; }
    if (lane == 0u) {
        bool retryLate;
        g_StreamBinVisible = loadStreamRasterCluster(record, false, false, push, g_StreamRasterCluster, retryLate) ? 1u : 0u;
        if (prepared && g_StreamBinVisible != 0u) { g_StreamRasterProjection = prepareStreamRasterProjection(g_StreamRasterCluster.params); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamBinVisible == 0u) { return; }
    if (lane < g_StreamRasterCluster.vertexCount) {
        float4 clip = prepared ? streamClusterClipCached(g_StreamRasterCluster, lane, push, g_StreamRasterProjection) :
            streamClusterClip(g_StreamRasterCluster, lane, push);
        if (prepared) {
            HybridScreenVertex vertex = hybridPrepareVertex(clip, float2(bins[6],bins[7]), bins[10]);
            // Reuse the clip scratch: one shared allocation in both A/B paths.
            g_StreamRasterWords[lane] = uint4(uint2(vertex.position),asuint(vertex.depth),0u);
        }
        else { g_StreamRasterWords[lane] = asuint(clip); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamRasterCluster.triangleCount) {
        uint3 tri = streamClusterTriangle(g_StreamRasterCluster, lane, push);
        uint id = packVisibilityId(g_StreamRasterCluster.recordIndex, lane);
        bool doubleSided = (g_StreamRasterCluster.instanceFlags & 2u) != 0u;
        if (prepared) {
            HybridScreenVertex a, b, c;
            a.position = asint(g_StreamRasterWords[tri.x].xy); a.depth = asfloat(g_StreamRasterWords[tri.x].z);
            b.position = asint(g_StreamRasterWords[tri.y].xy); b.depth = asfloat(g_StreamRasterWords[tri.y].z);
            c.position = asint(g_StreamRasterWords[tri.z].xy); c.depth = asfloat(g_StreamRasterWords[tri.z].z);
            hybridRasterPreparedTriangle(a, b, c,
                id, doubleSided, bins[6], bins[7], bins[10], bins[11], incrementalDepth);
        } else {
            hybridRasterTriangle(asfloat(g_StreamRasterWords[tri.x]), asfloat(g_StreamRasterWords[tri.y]), asfloat(g_StreamRasterWords[tri.z]),
                id, doubleSided, bins[6], bins[7], bins[10], bins[11]);
        }
    }
}

[shader("compute")]
[numthreads(128,1,1)]
void streamClusterRasterMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{ rasterStreamCluster(group,lane,push,true,false); }

[shader("compute")]
[numthreads(128,1,1)]
void streamClusterRasterLegacyMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{ rasterStreamCluster(group,lane,push,false,false); }

[shader("compute")]
[numthreads(128,1,1)]
void streamClusterRasterPlaneMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{ rasterStreamCluster(group,lane,push,true,true); }

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterCooperativeMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    // The cooperative decoder uses lanes 0..20. Keep the reference on narrow subgroups.
    if (WaveGetLaneCount() < 32u) { rasterStreamCluster(group, lane, push, false, false); return; }
    rasterStreamClusterCooperative(group, lane, push);
}
