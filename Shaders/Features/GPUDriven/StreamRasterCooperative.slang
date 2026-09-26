// SW-only descriptor. No traversal/culling params, bounds, material or request state.
struct StreamSoftwareRasterDesc
{
    float4 eye, center, upProjection, viewport, clipOrtho;
    float4 world0, world1, world2, world3;
    uint positionWord, positionStride, vertexCount, triangleCount;
    uint triangleByte, recordIndex, instanceFlags, renderCamera;
};
groupshared StreamSoftwareRasterDesc g_StreamSoftware;
groupshared uint g_StreamSoftwareValid;
groupshared float4 g_StreamSoftwareClip[128];

// Called by the complete first wave only. Every return is wave-uniform;
// all waves rejoin at the caller's group barrier, including rejected pages.
bool loadStreamSoftwareWave(uint record, GPUDrivenStreamAssetPush push)
{
    uint lane = WaveGetLaneIndex();
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> params =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>> header =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.activeHeaderBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindings =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.rasterBindingsBuffer, 0));
    uint field = 0u;
    switch (lane) {
    case 0: field = params[0].drawTaskCount; break;
    case 1: field = params[0].pageBufferBytes; break;
    case 2: field = params[0].scenePageCount; break;
    case 3: field = header[0].activeGroupCount; break;
    case 4: field = header[0].activeGroupCapacity; break;
    case 5: field = header[0].maxActiveGroupClusters; break;
    case 6: field = bindings[0].visibleRecordBase; break;
    case 7: field = bindings[0].visibleRecordCapacity; break;
    case 8: field = bindings[0].gpuSceneInstanceBuffer; break;
    case 9: field = params[0].renderViewport.y > 0.0 ? 1u : 0u; break;
    }
    uint tasks = WaveReadLaneAt(field, 0u) / kMeshletTriangleChunkCount;
    uint recordCapacity = WaveReadLaneAt(field, 7u);
    uint recordIndex;
    if (record >= tasks || !resolveVisibilityRecordIndex(record, WaveReadLaneAt(field, 6u),
        recordCapacity != 0u ? recordCapacity : tasks, recordIndex)) { return false; }
    uint maxClusters = max(WaveReadLaneAt(field, 5u), 1u);
    uint groupIndex = record / maxClusters;
    uint clusterIndex = record - groupIndex * maxClusters;
    if (groupIndex >= min(WaveReadLaneAt(field, 3u), WaveReadLaneAt(field, 4u))) { return false; }
    uint pageBufferBytes = WaveReadLaneAt(field, 1u);
    uint pageCount = WaveReadLaneAt(field, 2u);
    uint instanceBuffer = WaveReadLaneAt(field, 8u);
    bool renderCamera = WaveReadLaneAt(field, 9u) != 0u;
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> groups =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.activeGroupBuffer, 0));
    field = 0u;
    switch (lane) {
    case 0: field = groups[groupIndex].pageIndex; break;
    case 1: field = groups[groupIndex].clusterCount; break;
    case 2: field = groups[groupIndex].clusterSelectionMask; break;
    case 3: field = groups[groupIndex].flags; break;
    case 4: field = groups[groupIndex].gpuSceneInstanceIndex; break;
    }
    uint pageIndex = WaveReadLaneAt(field, 0u);
    uint instanceIndex = WaveReadLaneAt(field, 4u);
    if (pageIndex >= pageCount || instanceIndex == kInvalidClusterIndex ||
        clusterIndex >= WaveReadLaneAt(field, 1u) || clusterIndex >= 32u ||
        (WaveReadLaneAt(field, 3u) & (kActiveGroupLoadRequest | kActiveGroupUnloadRequest)) != 0u ||
        ((WaveReadLaneAt(field, 2u) >> clusterIndex) & 1u) == 0u) { return false; }
    uint instanceFlags = 0u;
    if (instanceBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instances =
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(instanceBuffer, 0));
        uint count = 0u, stride;
        if (lane == 0u) { StructuredBuffer<GPUDrivenPreviewInstance> buffer = instances; buffer.GetDimensions(count, stride); }
        if (instanceIndex >= WaveReadLaneFirst(count)) { return false; }
        if (lane == 0u) { instanceFlags = instances[instanceIndex].identity.w; }
        instanceFlags = WaveReadLaneFirst(instanceFlags);
    }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> table =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.pageTableBuffer, 0));
    field = lane == 0u ? table[pageIndex].deviceOffsetAndState : 0u;
    field = WaveReadLaneFirst(field);
    uint state = field & kStreamPageStateMask;
    uint pageOffset = field & kStreamPageDeviceOffsetMask;
    if ((state != kStreamStateResident && state != kStreamStateLockedFallback) ||
        pageBufferBytes < kPayloadHeaderBytes || pageOffset > pageBufferBytes - kPayloadHeaderBytes) { return false; }
    DescriptorHandle<StructuredBuffer<uint>> pages =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.pageBuffer, 0));
    uint pageWord = pageOffset / 4u;
    // One word per lane, coalesced header fetch. Only consume validated ranges.
    field = lane < kPayloadHeaderBytes / 4u ? pages[pageWord + lane] : 0u;
    uint bytes = WaveReadLaneAt(field, 12u);
    uint clusterCount = WaveReadLaneAt(field, 2u);
    uint vertexCount = WaveReadLaneAt(field, 3u);
    uint indexCount = WaveReadLaneAt(field, 4u);
    uint format = WaveReadLaneAt(field, 20u);
    if (bytes < kPayloadHeaderBytes || bytes > pageBufferBytes || pageOffset > pageBufferBytes - bytes ||
        clusterIndex >= clusterCount || (WaveReadLaneAt(field, 14u) & kPayloadAttributePosition) == 0u ||
        (format != kPayloadFormatFloat32x4 && format != kPayloadFormatFloat32x3)) { return false; }
    uint clusterOffset = WaveReadLaneAt(field, 9u);
    uint positionOffset = WaveReadLaneAt(field, 10u);
    uint triangleOffset = WaveReadLaneAt(field, 11u);
    uint positionStride = format == kPayloadFormatFloat32x3 ? 3u : 4u;
    if ((clusterOffset & 3u) != 0u || (positionOffset & 3u) != 0u ||
        !streamPayloadArrayRangeValid(clusterOffset, clusterCount, kPayloadClusterBytes, bytes) ||
        !streamPayloadArrayRangeValid(positionOffset, vertexCount, positionStride * 4u, bytes) ||
        !streamPayloadRangeValid(triangleOffset, indexCount, bytes)) { return false; }
    field = lane < 4u ? pages[pageWord + clusterOffset / 4u + clusterIndex * kPayloadClusterWords + lane] : 0u;
    uint vertexOffset = WaveReadLaneAt(field, 0u);
    uint vertices = WaveReadLaneAt(field, 1u);
    uint triangleStart = WaveReadLaneAt(field, 2u);
    uint triangles = WaveReadLaneAt(field, 3u);
    if (vertices == 0u || vertices > kMaxMeshletVertices || vertexOffset > vertexCount || vertices > vertexCount - vertexOffset ||
        triangles > kMaxMeshletTriangles || triangleStart > indexCount || triangles > (indexCount - triangleStart) / 3u) { return false; }
    // Disjoint writes need no wave/group shared reads before the outer barrier.
    switch (lane) {
    case 0: g_StreamSoftware.eye = renderCamera ? params[0].renderEye : params[0].eye; break;
    case 1: g_StreamSoftware.center = renderCamera ? params[0].renderCenter : params[0].center; break;
    case 2: g_StreamSoftware.upProjection = renderCamera ? params[0].renderUpProjection : params[0].upProjection; break;
    case 3: g_StreamSoftware.viewport = renderCamera ? params[0].renderViewport : params[0].viewport; break;
    case 4: g_StreamSoftware.clipOrtho = renderCamera ? params[0].renderClipOrtho : params[0].clipOrtho; break;
    case 5: g_StreamSoftware.world0 = groups[groupIndex].world0; break;
    case 6: g_StreamSoftware.world1 = groups[groupIndex].world1; break;
    case 7: g_StreamSoftware.world2 = groups[groupIndex].world2; break;
    case 8: g_StreamSoftware.world3 = groups[groupIndex].world3; break;
    case 9: g_StreamSoftware.positionWord = pageWord + positionOffset / 4u + vertexOffset * positionStride; break;
    case 10: g_StreamSoftware.positionStride = positionStride; break;
    case 11: g_StreamSoftware.vertexCount = vertices; break;
    case 12: g_StreamSoftware.triangleCount = triangles; break;
    case 13: g_StreamSoftware.triangleByte = pageOffset + triangleOffset + triangleStart; break;
    case 14: g_StreamSoftware.recordIndex = recordIndex; break;
    case 15: g_StreamSoftware.instanceFlags = instanceFlags; break;
    case 16: g_StreamSoftware.renderCamera = renderCamera ? 1u : 0u; break;
    }
    return true;
}

float4 streamSoftwareClip(uint vertex, GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pages =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.pageBuffer, 0));
    uint word = g_StreamSoftware.positionWord + vertex * g_StreamSoftware.positionStride;
    float3 local = float3(asfloat(pages[word]), asfloat(pages[word + 1u]), asfloat(pages[word + 2u]));
    float3 worldPosition = g_StreamSoftware.world0.xyz * local.x + g_StreamSoftware.world1.xyz * local.y +
        g_StreamSoftware.world2.xyz * local.z + g_StreamSoftware.world3.xyz;
    // Keep the reference transform/projection expression order and jitter signs.
    float3 forward = safeNormalize(g_StreamSoftware.center.xyz - g_StreamSoftware.eye.xyz, float3(0, 0, -1));
    float3 cameraUp = safeNormalize(g_StreamSoftware.upProjection.xyz, float3(0, 1, 0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1, 0, 0));
    float3 up = safeNormalize(cross(right, forward), float3(0, 1, 0));
    float3 delta = worldPosition - g_StreamSoftware.eye.xyz;
    float3 view = float3(dot(delta, right), dot(delta, up), dot(delta, forward));
    float nearZ = max(g_StreamSoftware.clipOrtho.x, 0.0001);
    float farZ = max(g_StreamSoftware.clipOrtho.y, nearZ + 0.0001);
    float depth = conservativeProjectDepth(view.z, nearZ, farZ, g_StreamSoftware.upProjection.w > 0.5, g_StreamSoftware.clipOrtho.w > 0.5);
    float aspect = max(g_StreamSoftware.viewport.x, 0.001);
    float4 clip;
    if (g_StreamSoftware.upProjection.w > 0.5) {
        float h = max(g_StreamSoftware.clipOrtho.z * 0.5, 0.0001);
        clip = float4(view.x / (h * aspect), view.y / h, depth, 1.0);
    } else {
        float yScale = 1.0 / tan(clamp(g_StreamSoftware.viewport.w, 0.017453292, 3.12413936) * 0.5);
        clip = float4(view.x * yScale / aspect, view.y * yScale, depth * view.z, view.z);
    }
    if (g_StreamSoftware.renderCamera != 0u) {
        clip.xy -= 2.0 * float2(g_StreamSoftware.eye.w, -g_StreamSoftware.center.w) / g_StreamSoftware.viewport.yz * clip.w;
    }
    clip.y = -clip.y;
    return clip;
}

uint3 streamSoftwareTriangle(uint triangle, GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pages =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.pageBuffer, 0));
    uint address = g_StreamSoftware.triangleByte + triangle * 3u;
    uint3 indices = uint3(readPageByte(pages, address), readPageByte(pages, address + 1u), readPageByte(pages, address + 2u));
    indices = uint3(indices.x < g_StreamSoftware.vertexCount ? indices.x : 0u,
        indices.y < g_StreamSoftware.vertexCount ? indices.y : 0u, indices.z < g_StreamSoftware.vertexCount ? indices.z : 0u);
    bool reflected = dot(cross(g_StreamSoftware.world0.xyz, g_StreamSoftware.world1.xyz), g_StreamSoftware.world2.xyz) < 0.0;
    return reflected ? indices.xzy : indices;
}

void rasterStreamClusterCooperative(uint3 group, uint lane, GPUDrivenStreamAssetPush push)
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
    if (lane < g_StreamSoftware.vertexCount) { g_StreamSoftwareClip[lane] = streamSoftwareClip(lane, push); }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamSoftware.triangleCount) {
        uint3 tri = streamSoftwareTriangle(lane, push);
        hybridRasterTriangle(g_StreamSoftwareClip[tri.x], g_StreamSoftwareClip[tri.y], g_StreamSoftwareClip[tri.z],
            packVisibilityId(g_StreamSoftware.recordIndex, lane), (g_StreamSoftware.instanceFlags & 2u) != 0u,
            bins[6], bins[7], bins[9], bins[10], bins[11]);
    }
}
