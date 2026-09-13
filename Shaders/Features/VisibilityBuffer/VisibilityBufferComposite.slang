#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"

// Debug consumer ABI; the raster push layout remains unchanged.
struct VisibilityBufferCompositePush
{
    uint imageShaderIndexBase, bufferShaderIndexBase;
    uint paramsBuffer, visibilityImage, depthImage;
    uint residentRecords, meshletBuffer, residentRecordCapacity;
    uint streamRecords, streamGroups;
};

// Matches MeshletStreamGpuActiveGroup. Logical pageIndex survives compaction
// and physical page eviction/reload; dataIndex and pageDeviceOffset do not.
struct VisibilityDebugActiveGroup
{
    uint pageDeviceOffset, pageIndex, clusterCount, primitiveIndex;
    uint lodLevel, materialIndex, mask, flags;
    uint instanceIndex, gpuSceneInstanceIndex, padding0, padding1;
    float4 world0, world1, world2, world3;
};

struct FullscreenVertexOutput
{
    float4 position : SV_Position;
};

static const float2 kFullscreenTrianglePositions[3] =
{
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};

[shader("vertex")]
FullscreenVertexOutput visibilityBufferCompositeVertexMain(uint vertexIndex : SV_VertexID)
{
    FullscreenVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    return output;
}

// Debug display only: no triangle attribute reconstruction, textures or BSDF.
// Both resident and stream producers share the same packed visibility namespace.
uint visibilityHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float3 visibilityHashColor(uint value)
{
    value = visibilityHash(value);
    return 0.25 + 0.75 * float3(
        value & 0xffu, (value >> 8u) & 0xffu, (value >> 16u) & 0xffu) / 255.0;
}

// Resolve the temporary visibility slot to geometry identity before coloring.
// Instances of the same geometry intentionally share meshlet colors.
bool visibilityDebugIdentity(uint index, VisibilityBufferCompositePush push,
    out uint clusterSeed, out uint lodLevel)
{
    clusterSeed = lodLevel = 0u;
    uint count, stride;
    if (index < push.residentRecordCapacity) {
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> recordsHandle =
            DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.bufferShaderIndexBase + push.residentRecords, 0));
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> clustersHandle =
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
        StructuredBuffer<VisibleClusterRecord> records = recordsHandle;
        StructuredBuffer<GPUDrivenPreviewMeshlet> clusters = clustersHandle;
        records.GetDimensions(count, stride);
        if (index >= count) { return false; }
        VisibleClusterRecord record = records[index];
        clusters.GetDimensions(count, stride);
        if (visibleClusterSource(record) != kVisibleClusterSourceResident || record.clusterIndex >= count) { return false; }
        clusterSeed = visibilityHash(record.clusterIndex + 1u);
        lodLevel = clusters[record.clusterIndex].lod.x;
        return true;
    }
    if (push.streamRecords == 0xffffffffu || push.streamGroups == 0xffffffffu) { return false; }
    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> recordsHandle =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.bufferShaderIndexBase + push.streamRecords, 0));
    DescriptorHandle<StructuredBuffer<VisibilityDebugActiveGroup>> groupsHandle =
        DescriptorHandle<StructuredBuffer<VisibilityDebugActiveGroup>>(uint2(push.bufferShaderIndexBase + push.streamGroups, 0));
    StructuredBuffer<VisibleClusterRecord> records = recordsHandle;
    StructuredBuffer<VisibilityDebugActiveGroup> groups = groupsHandle;
    index -= push.residentRecordCapacity;
    records.GetDimensions(count, stride);
    if (index >= count) { return false; }
    VisibleClusterRecord record = records[index];
    groups.GetDimensions(count, stride);
    if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage || record.dataIndex >= count) { return false; }
    VisibilityDebugActiveGroup group = groups[record.dataIndex];
    if (record.clusterIndex >= group.clusterCount || record.instanceIndex != group.gpuSceneInstanceIndex) { return false; }
    clusterSeed = visibilityHash(group.pageIndex ^ 0x9e3779b9u) ^ visibilityHash(record.clusterIndex + 1u);
    lodLevel = group.lodLevel;
    return true;
}

[shader("fragment")]
float4 visibilityBufferCompositeFragmentMain(
    FullscreenVertexOutput input,
    uniform VisibilityBufferCompositePush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<Texture2D<uint>> visibilityHandle =
        DescriptorHandle<Texture2D<uint>>(uint2(
            push.imageShaderIndexBase + push.visibilityImage, 0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    uint2 pixel = min(uint2(input.position.xy), uint2(params.width - 1u, params.height - 1u));
    Texture2D<uint> visibility = visibilityHandle;
    uint packedVisibility = visibility.Load(int3(pixel, 0));
    uint recordIndex;
    uint triangleIndex;
    if (params.mode == 6u || !unpackVisibilityId(packedVisibility, recordIndex, triangleIndex)) {
        return params.clearColor;
    }
    if (params.mode == 4u) {
        DescriptorHandle<Texture2D<float>> depthHandle =
            DescriptorHandle<Texture2D<float>>(uint2(
                push.imageShaderIndexBase + push.depthImage, 0));
        Texture2D<float> depth = depthHandle;
        float z = saturate(depth.Load(int3(pixel, 0)));
        return float4(z, z, z, 1.0);
    }
    if (params.mode == 5u) {
        return float4(1.0);
    }
    uint clusterSeed, lodLevel;
    if (!visibilityDebugIdentity(recordIndex, push, clusterSeed, lodLevel)) { return params.clearColor; }
    uint seed = clusterSeed;
    if (params.mode == 2u) { seed = lodLevel + 1u; }
    if (params.mode == 3u) { seed = visibilityHash(clusterSeed) ^ visibilityHash(triangleIndex + 1u); }
    return float4(visibilityHashColor(seed), 1.0);
}
