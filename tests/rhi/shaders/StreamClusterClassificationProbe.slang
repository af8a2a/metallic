// Reference schedule retained only for GPU equivalence tests.
#include "../../../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang"

[shader("compute")]
[numthreads(128, 1, 1)]
void referenceStreamClusterBinMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> handle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    RWStructuredBuffer<uint> bins = handle;
    uint candidate = hybridFlatGroup(group);
    if (candidate >= bins[12]) { return; }
    uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
    if (lane == 0u) {
        hybridClassifyCluster(bins, 0xffffffffu, record, candidate);
        bool retryLate;
        g_StreamBinVisible = loadStreamRasterCluster(record, true, true, push, g_StreamRasterCluster, retryLate) ? 1u : 0u;
        if (push.traversalPhase == 0u && retryLate) {
            InterlockedOr(bins[hybridRetryBase(bins[5]) + record / bins[13]], 1u << (record % bins[13]));
        }
        g_StreamBinHardware = 0u;
        g_StreamRasterProjection = prepareStreamRasterProjection(g_StreamRasterCluster.params);
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamBinVisible == 0u || g_StreamRasterCluster.triangleCount == 0u) { return; }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> raster =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
    // The retained reference predates Full coverage materials. Their HW-only
    // contract is independent of the candidate scheduling being compared.
    if (asfloat(bins[8]) == 0.0 || (g_StreamRasterCluster.instanceFlags & 12u) != 0u ||
        raster[0].tessellationBuffer != 0xffffffffu) {
        if (lane == 0u) { hybridClassifyCluster(bins, 0u, record, candidate); }
        return;
    }
    if (lane < g_StreamRasterCluster.vertexCount) {
        g_StreamRasterClip[lane] = streamClusterClip(g_StreamRasterCluster, lane, push);
        float4 cached = streamClusterClipCached(g_StreamRasterCluster,lane,push,g_StreamRasterProjection);
        // Header overflow is compared verbatim after binning, including HW
        // triangles, so a changed projection cannot hide in a bin decision.
        if (any(asuint(cached) != asuint(g_StreamRasterClip[lane]))) { InterlockedOr(bins[14],0x80000000u); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamRasterCluster.triangleCount) {
        uint3 tri = streamClusterTriangle(g_StreamRasterCluster, lane, push);
        if (!hybridTriangleFits(g_StreamRasterClip[tri.x], g_StreamRasterClip[tri.y], g_StreamRasterClip[tri.z],
            float2(bins[6], bins[7]), asfloat(bins[8]))) { InterlockedOr(g_StreamBinHardware, 1u); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) { hybridClassifyCluster(bins, g_StreamBinHardware != 0u ? 0u : kHybridSoftwareBin, record, candidate); }
}

