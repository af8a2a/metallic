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


groupshared uint g_ReferenceLoadValid;
[shader("compute")]
[numthreads(128, 1, 1)]
void verifyStreamSoftwareLoadMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    uint candidate = hybridFlatGroup(group);
    if (candidate >= bins[12] || WaveGetLaneCount() < 32u) { return; }
    uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
    if (lane < WaveGetLaneCount()) {
        bool valid = loadStreamSoftwareWave(record, push);
        if (lane == 0u) { g_StreamSoftwareValid = valid ? 1u : 0u; }
    }
    if (lane == 0u) {
        bool retry;
        g_ReferenceLoadValid = loadStreamRasterCluster(record, false, false, push, g_StreamRasterCluster, retry) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamSoftwareValid != g_ReferenceLoadValid) { InterlockedOr(bins[14], 0x40000000u); return; }
    if (g_ReferenceLoadValid == 0u) { return; }
    bool same = g_StreamSoftware.vertexCount == g_StreamRasterCluster.vertexCount &&
        g_StreamSoftware.triangleCount == g_StreamRasterCluster.triangleCount &&
        g_StreamSoftware.recordIndex == g_StreamRasterCluster.recordIndex &&
        g_StreamSoftware.instanceFlags == g_StreamRasterCluster.instanceFlags;
    if (lane < g_StreamRasterCluster.vertexCount) {
        same = same && all(asuint(streamSoftwareClip(lane, push)) == asuint(streamClusterClip(g_StreamRasterCluster, lane, push)));
    }
    if (lane < g_StreamRasterCluster.triangleCount) {
        same = same && all(streamSoftwareTriangle(lane, push) == streamClusterTriangle(g_StreamRasterCluster, lane, push));
    }
    if (!same) { InterlockedOr(bins[14], 0x20000000u); }
}

// Frozen pre-P0 classifier for same-input correctness and GPU timing.
[shader("compute")]
[numthreads(128, 1, 1)]
void legacyStreamClusterBinMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> handle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    RWStructuredBuffer<uint> bins = handle;
    uint work = hybridFlatGroup(group);
    if (work >= bins[0]) { return; }
    uint item = hybridStreamClassifyBase(work);
    uint candidate = bins[item];
    uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> raster = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
    bool needsCoverage = false;
    if (raster[0].gpuSceneInstanceBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> groups = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instances = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(push.bufferShaderIndexBase + raster[0].gpuSceneInstanceBuffer, 0));
        needsCoverage = (instances[groups[record / bins[13]].gpuSceneInstanceIndex].identity.w & 12u) != 0u;
    }
    if (raster[0].tessellationBuffer != 0xffffffffu || needsCoverage) {
        if (lane == 0u) { hybridClassifyCluster(bins, 0u, record, candidate); }
        return;
    }
    if (lane == 0u) {
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> params =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> groups =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
        g_StreamRasterCluster = (StreamRasterCluster)0;
        g_StreamRasterCluster.params = params[0];
        g_StreamRasterCluster.group = groups[record / bins[13]];
        g_StreamRasterCluster.pageWord = bins[item + 1u];
        g_StreamRasterCluster.vertexCount = bins[item + 2u] & 0xffffu;
        g_StreamRasterCluster.triangleCount = (bins[item + 2u] >> 16u) & 0x7fffu;
        g_StreamRasterCluster.positionStrideWords = (bins[item + 2u] & 0x80000000u) != 0u ? 3u : 4u;
        g_StreamRasterCluster.triangleBaseByte = bins[item + 3u];
        g_StreamBinHardware = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamRasterCluster.vertexCount) { g_StreamRasterClip[lane] = streamClusterClip(g_StreamRasterCluster, lane, push); }
    GroupMemoryBarrierWithGroupSync();
    bool hardware = false;
    if (lane < g_StreamRasterCluster.triangleCount) {
        uint3 tri = streamClusterTriangle(g_StreamRasterCluster, lane, push);
        hardware = !hybridTriangleFits(g_StreamRasterClip[tri.x], g_StreamRasterClip[tri.y], g_StreamRasterClip[tri.z],
            float2(bins[6], bins[7]), asfloat(bins[8]));
    }
    bool waveHardware = WaveActiveAnyTrue(hardware);
    if (WaveIsFirstLane() && waveHardware) { InterlockedOr(g_StreamBinHardware, 1u); }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) { hybridClassifyCluster(bins, g_StreamBinHardware != 0u ? 0u : kHybridSoftwareBin, record, candidate); }
}
