import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

// Debug consumer ABI; the raster push layout remains unchanged.
struct VisibilityBufferCompositePush
{

    uint paramsBuffer, visibilityImage, depthImage;
    uint residentRecords, meshletBuffer, residentRecordCapacity;
    uint streamRecords, streamGroups;
    uint shadedColors;
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

// Debug display only: no material textures or BSDF.
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
    // Match the reference's restrained ID-albedo range, leaving room for form shading.
    return 0.3 + 0.5 * float3(
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
            DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.residentRecords, 0));
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> clustersHandle =
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.meshletBuffer, 0));
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
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.streamRecords, 0));
    DescriptorHandle<StructuredBuffer<VisibilityDebugActiveGroup>> groupsHandle =
        DescriptorHandle<StructuredBuffer<VisibilityDebugActiveGroup>>(uint2(push.streamGroups, 0));
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

float3 debugViewPosition(int2 pixel, float depth, GPUDrivenPreviewParams params)
{
    float nearZ = max(params.renderClipOrtho.x, 0.0001);
    float farZ = max(params.renderClipOrtho.y, nearZ + 0.0001);
    bool ortho = params.renderUpProjection.w > 0.5;
    bool reversed = params.renderClipOrtho.w > 0.5;
    float z = ortho ? (reversed ? farZ - depth * (farZ - nearZ) : nearZ + depth * (farZ - nearZ))
        : nearZ * farZ / max(reversed ? nearZ + depth * (farZ - nearZ) : farZ - depth * (farZ - nearZ), 1e-8);
    float2 uv = (float2(pixel) + 0.5 + float2(params.renderEye.w, params.renderCenter.w)) /
        float2(params.width, params.height);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float halfHeight = ortho ? max(params.renderClipOrtho.z * 0.5, 0.0001)
        : z * tan(clamp(params.renderViewport.w, 0.017453292, 3.12413936) * 0.5);
    return float3(ndc * float2(max(params.renderViewport.x, 0.001), 1.0) * halfHeight, z);
}

bool debugNeighbor(int2 pixel, Texture2D<uint> visibility, Texture2D<float> depth,
    GPUDrivenPreviewParams params, out float3 position)
{
    position = float3(0);
    if (any(pixel < 0) || any(pixel >= int2(params.width, params.height)) ||
        visibility.Load(int3(pixel, 0)) == 0u) { return false; }
    position = debugViewPosition(pixel, depth.Load(int3(pixel, 0)), params);
    return all(isfinite(position));
}

float debugGeometryLighting(uint2 pixel, Texture2D<uint> visibility, Texture2D<float> depth,
    GPUDrivenPreviewParams params)
{
    if (params.renderViewport.y <= 0.0 || params.width < 2u || params.height < 2u) { return 1.0; }
    float3 p = debugViewPosition(int2(pixel), depth.Load(int3(pixel, 0)), params);
    float3 l, r, t, b;
    bool vl = debugNeighbor(int2(pixel) + int2(-1, 0), visibility, depth, params, l);
    bool vr = debugNeighbor(int2(pixel) + int2(1, 0), visibility, depth, params, r);
    bool vt = debugNeighbor(int2(pixel) + int2(0, -1), visibility, depth, params, t);
    bool vb = debugNeighbor(int2(pixel) + int2(0, 1), visibility, depth, params, b);
    if ((!vl && !vr) || (!vt && !vb)) { return 0.65; }
    // Select the closer side at depth discontinuities instead of differentiating
    // across a foreground/background edge. This also avoids meshlet-boundary lines.
    float3 dx = vl && (!vr || abs(l.z - p.z) < abs(r.z - p.z)) ? p - l : r - p;
    float3 dy = vt && (!vb || abs(t.z - p.z) < abs(b.z - p.z)) ? p - t : b - p;
    float3 n = cross(dx, dy);
    float scale = max(abs(n.x), max(abs(n.y), abs(n.z)));
    if (!isfinite(scale) || scale < 1e-20) { return 0.65; }
    n = normalize(n / scale);
    if (n.z > 0.0) { n = -n; }
    float3 forward = safeNormalize(params.renderCenter.xyz - params.renderEye.xyz, float3(0, 0, -1));
    float3 right = safeNormalize(cross(forward, params.renderUpProjection.xyz), float3(1, 0, 0));
    float3 up = cross(right, forward);
    float3 worldNormal = right * n.x + up * n.y + forward * n.z;
    // A soft view fill keeps back-lit surfaces readable while the world-space
    // key and sky retain orientation cues (as in the reference's flashlight).
    float lighting = 0.38 + 0.12 * saturate(worldNormal.y) + 0.30 * saturate(-n.z) +
        0.20 * saturate(dot(worldNormal, normalize(float3(-0.45, 0.8, 0.4))));
    // Small screen-space contact occlusion, range-limited to avoid dark halos
    // from unrelated foreground objects. Not a replacement for ray-traced shadows.
    float radius = max(max(length(dx), length(dy)) * 12.0, 1e-5);
    float occlusion = 0.0;
    static const int2 offsets[8] = {int2(3, 0), int2(-3, 0), int2(0, 3), int2(0, -3),
        int2(6, 6), int2(-6, 6), int2(6, -6), int2(-6, -6)};
    [unroll] for (uint i = 0; i < 8u; ++i) {
        float3 q;
        if (!debugNeighbor(int2(pixel) + offsets[i], visibility, depth, params, q)) { continue; }
        float3 delta = q - p;
        float distance = length(delta);
        if (distance > 1e-6) {
            occlusion += saturate((dot(n, delta) / distance - 0.08) * 2.0) *
                saturate(1.0 - distance / radius);
        }
    }
    return lighting * (1.0 - 0.45 * occlusion / 8.0);
}

[shader("fragment")]
float4 visibilityBufferCompositeFragmentMain(
    FullscreenVertexOutput input,
    uniform VisibilityBufferCompositePush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(
            push.paramsBuffer, 0));
    DescriptorHandle<Texture2D<uint>> visibilityHandle =
        DescriptorHandle<Texture2D<uint>>(uint2(
            push.visibilityImage, 0));
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
                push.depthImage, 0));
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
    // With tessellation disabled, generated triangles are the source triangles.
    if (params.mode == 3u || params.mode == 7u) { seed = visibilityHash(clusterSeed) ^ visibilityHash(triangleIndex + 1u); }
    float3 color = visibilityHashColor(seed);
    if (push.shadedColors != 0u) {
        DescriptorHandle<Texture2D<float>> depthHandle = DescriptorHandle<Texture2D<float>>(
            uint2(push.depthImage, 0));
        Texture2D<float> depth = depthHandle;
        color *= debugGeometryLighting(pixel, visibility, depth, params);
    }
    return float4(color, 1.0);
}
