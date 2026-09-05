#include "GPUDrivenCullingCommon.slang"

#ifndef GPU_DRIVEN_AMPLIFICATION_WAVE_OPS
#define GPU_DRIVEN_AMPLIFICATION_WAVE_OPS 1
#endif

static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kAmplificationGroupSize = 32;

#ifndef VISIBILITY_BUFFER_ALPHA_MASKED
#define VISIBILITY_BUFFER_ALPHA_MASKED 0
#endif

// Opaque rasterization exports only position and a per-primitive visibility ID.
// Masked materials additionally export the attributes required for alpha testing.
struct VisibilityBufferVertexOutput
{
    float4 position : SV_Position;
#if VISIBILITY_BUFFER_ALPHA_MASKED
    [vk::location(0)]
    float2 texcoord : TEXCOORD1;
    [vk::location(1)]
    nointerpolation uint materialIndex : TEXCOORD2;
#endif
};

struct VisibilityBufferPrimitiveOutput
{
    uint visibilityId : SV_PrimitiveID;
};

struct VisibilityBufferFragmentInput
{
    float4 position : SV_Position;
    [vk::location(0)]
    float2 texcoord : TEXCOORD1;
    [vk::location(1)]
    nointerpolation uint materialIndex : TEXCOORD2;
    uint visibilityId : SV_PrimitiveID;
};

struct VisibilityBufferAmplificationPayload
{
    uint meshletDrawIndices[kAmplificationGroupSize];
};

groupshared VisibilityBufferAmplificationPayload g_AmplificationPayload;
groupshared uint g_AmplificationVisibleCount;

float2 transformMaterialUv(GPUDrivenPreviewTextureInfo textureInfo, float2 uv)
{
    return float2(
        dot(textureInfo.transform0.xy, uv) + textureInfo.transform0.z,
        dot(textureInfo.transform1.xy, uv) + textureInfo.transform1.z);
}

float4 sampleImageBilinear(Texture2D<float4> texture, float2 uv)
{
    uint width = 1u;
    uint height = 1u;
    texture.GetDimensions(width, height);
    float2 texel = frac(uv) * float2(width, height) - 0.5;
    int2 baseTexel = int2(floor(texel));
    float2 fraction = frac(texel);
    int x0 = (baseTexel.x % int(width) + int(width)) % int(width);
    int y0 = (baseTexel.y % int(height) + int(height)) % int(height);
    int x1 = (x0 + 1) % int(width);
    int y1 = (y0 + 1) % int(height);
    float4 v00 = texture.Load(int3(x0, y0, 0));
    float4 v10 = texture.Load(int3(x1, y0, 0));
    float4 v01 = texture.Load(int3(x0, y1, 0));
    float4 v11 = texture.Load(int3(x1, y1, 0));
    return lerp(lerp(v00, v10, fraction.x), lerp(v01, v11, fraction.x), fraction.y);
}

float materialBaseColorAlpha(
    GPUDrivenPreviewMaterial material,
    float2 uv,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push)
{
    GPUDrivenPreviewTextureInfo textureInfo = material.baseColorTexture;
    if (textureInfo.textureIndex == kInvalidTextureIndex ||
        textureInfo.textureIndex >= params.materialTextureCount) {
        return material.baseColor.a;
    }
    DescriptorHandle<StructuredBuffer<uint>> remapHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.materialTextureRemapBuffer,
            0));
    StructuredBuffer<uint> remap = remapHandle;
    DescriptorHandle<Texture2D<float4>> textureHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(
            push.imageShaderIndexBase + remap[textureInfo.textureIndex],
            0));
    Texture2D<float4> texture = textureHandle;
    return material.baseColor.a *
        sampleImageBilinear(texture, transformMaterialUv(textureInfo, uv)).a;
}

[shader("amplification")]
[numthreads(kAmplificationGroupSize, 1, 1)]
void visibilityBufferAmplificationMain(
    uint3 groupID : SV_GroupID,
    uint groupThreadIndex : SV_GroupIndex,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(
            push.bufferShaderIndexBase + push.meshletBuffer,
            0));
    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> meshletDrawHandle =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(
            push.bufferShaderIndexBase + push.meshletDrawBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(
            push.bufferShaderIndexBase + push.instanceBuffer,
            0));
    DescriptorHandle<StructuredBuffer<uint>> visibilityHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.instanceVisibilityBuffer,
            0));

    GPUDrivenPreviewParams params = paramsHandle[0];
    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<VisibleClusterRecord> meshletDraws = meshletDrawHandle;
    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    StructuredBuffer<uint> visibility = visibilityHandle;

    uint localMeshletIndex =
        groupID.x * kAmplificationGroupSize + groupThreadIndex;
    uint meshletDrawIndex = params.meshletOffset + localMeshletIndex;
    bool visible = localMeshletIndex < params.meshletCount;
    if (visible) {
        VisibleClusterRecord meshletDraw = meshletDraws[meshletDrawIndex];
        uint expectedVisibility = push.passIndex == 0u ? 1u : 3u;
        visible = meshletDraw.instanceIndex < params.instanceCount &&
            visibleClusterSource(meshletDraw) == kVisibleClusterSourceResident &&
            visibleClusterDrawBucket(meshletDraw) == push.mipLevel &&
            push.mipLevel < kDrawBucketCount &&
            visibility[meshletDraw.instanceIndex] == expectedVisibility;
        // Stream culling reuses the shared visibility state after resident
        // culling, so producer ownership must be checked independently here.
        if (visible && push.streamOwnerMaskBuffer != kInvalidTextureIndex) {
            DescriptorHandle<StructuredBuffer<uint>> streamOwnerMaskHandle =
                DescriptorHandle<StructuredBuffer<uint>>(uint2(
                    push.bufferShaderIndexBase + push.streamOwnerMaskBuffer,
                    0));
            StructuredBuffer<uint> streamOwnerMask = streamOwnerMaskHandle;
            visible = streamOwnerMask[meshletDraw.instanceIndex] == 0u;
        }
        if (visible) {
            GPUDrivenPreviewInstance instance = instances[meshletDraw.instanceIndex];
            GPUDrivenPreviewMeshlet meshlet = meshlets[meshletDraw.clusterIndex];
            visible = meshletVisible(
                meshlet,
                instance.identity.w,
                instanceTransform(instance),
                params);
        }
    }

    uint visibleCount = 0u;
#if GPU_DRIVEN_AMPLIFICATION_WAVE_OPS
    uint wavePrefix = WavePrefixCountBits(visible);
    uint waveVisibleCount = WaveActiveCountBits(visible);
    visibleCount = waveVisibleCount;

    // The Vulkan pipeline requests a full Wave32 when supported. On NVIDIA,
    // this 32-thread task group becomes one native wave and matches the D3D12
    // sample's zero-atomic payload compaction path.
    if (WaveGetLaneCount() == kAmplificationGroupSize) {
        if (visible) {
            g_AmplificationPayload.meshletDrawIndices[wavePrefix] =
                meshletDrawIndex;
        }
    } else {
        // Keep the wave path valid when required subgroup size 32 is
        // unavailable. Each subgroup reserves one contiguous payload range.
        if (groupThreadIndex == 0u) {
            g_AmplificationVisibleCount = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        uint waveBase = 0u;
        if (WaveIsFirstLane()) {
            InterlockedAdd(
                g_AmplificationVisibleCount,
                waveVisibleCount,
                waveBase);
        }
        waveBase = WaveReadLaneFirst(waveBase);
        if (visible) {
            g_AmplificationPayload.meshletDrawIndices[waveBase + wavePrefix] =
                meshletDrawIndex;
        }
        GroupMemoryBarrierWithGroupSync();
        visibleCount = g_AmplificationVisibleCount;
    }
#else
    // Some Vulkan task-shader implementations do not expose subgroup ballot.
    // Retain a portable groupshared compaction path for those devices.
    if (groupThreadIndex == 0u) {
        g_AmplificationVisibleCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (visible) {
        uint payloadIndex = 0u;
        InterlockedAdd(g_AmplificationVisibleCount, 1u, payloadIndex);
        g_AmplificationPayload.meshletDrawIndices[payloadIndex] =
            meshletDrawIndex;
    }
    GroupMemoryBarrierWithGroupSync();
    visibleCount = g_AmplificationVisibleCount;
#endif

    DispatchMesh(visibleCount, 1u, 1u, g_AmplificationPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void visibilityBufferMeshMain(
    uint3 groupID : SV_GroupID,
    uint groupThreadIndex : SV_GroupIndex,
    in payload VisibilityBufferAmplificationPayload payload,
    out vertices VisibilityBufferVertexOutput vertices[kMaxMeshletVertices],
    out indices uint3 triangles[kMaxMeshletTriangles],
    out primitives VisibilityBufferPrimitiveOutput primitives[kMaxMeshletTriangles],
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> positionHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> meshletHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.meshletBuffer, 0));
    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> meshletDrawHandle =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(
            push.bufferShaderIndexBase + push.meshletDrawBuffer,
            0));
    DescriptorHandle<StructuredBuffer<uint>> meshletVertexHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletVertexBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> meshletTriangleHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletTriangleBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>> geometryHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>>(uint2(
            push.bufferShaderIndexBase + push.transformBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(
            push.bufferShaderIndexBase + push.instanceBuffer,
            0));
    StructuredBuffer<GPUDrivenPreviewVertex> verticesBuffer = positionHandle;
    StructuredBuffer<GPUDrivenPreviewMeshlet> meshlets = meshletHandle;
    StructuredBuffer<VisibleClusterRecord> meshletDraws = meshletDrawHandle;
    StructuredBuffer<uint> meshletVertices = meshletVertexHandle;
    StructuredBuffer<uint> meshletTriangles = meshletTriangleHandle;
    StructuredBuffer<GPUDrivenPreviewParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenPreviewGeometry> geometries = geometryHandle;
    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    GPUDrivenPreviewParams params = paramsBuffer[0];

    uint meshletDrawIndex = payload.meshletDrawIndices[groupID.x];
    VisibleClusterRecord meshletDraw = meshletDraws[meshletDrawIndex];
    GPUDrivenPreviewMeshlet meshlet = meshlets[meshletDraw.clusterIndex];
    GPUDrivenPreviewGeometry geometry = geometries[meshletDraw.dataIndex];
    GPUDrivenPreviewInstance instance = instances[meshletDraw.instanceIndex];
    SceneTransform transform = instanceTransform(instance);
    uint vertexCount = meshlet.ranges.y;
    uint triangleCount = meshlet.ranges.w;
    if (vertexCount > kMaxMeshletVertices ||
        triangleCount > kMaxMeshletTriangles) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    SetMeshOutputCounts(vertexCount, triangleCount);

    CameraData projectionCamera = push.projectWithCullingCamera != 0u ?
        cullingCamera(params) : renderCamera(params);
    if (groupThreadIndex < vertexCount) {
        uint positionIndex = geometry.payload.x +
            meshletVertices[meshlet.ranges.x + groupThreadIndex];
        GPUDrivenPreviewVertex vertex = verticesBuffer[positionIndex];
        float3 worldPosition = transformPoint(
            transform,
            vertex.position.xyz);
        float4 position = projectPosition(worldPosition, projectionCamera);
        position.y = -position.y;
        vertices[groupThreadIndex].position = position;
#if VISIBILITY_BUFFER_ALPHA_MASKED
        vertices[groupThreadIndex].texcoord = vertex.texcoord.xy;
        vertices[groupThreadIndex].materialIndex = instance.identity.y;
#endif
    }

    if (groupThreadIndex < triangleCount) {
        triangles[groupThreadIndex] = uint3(
            loadPackedMeshletTriangleIndex(
                meshletTriangles,
                meshlet.ranges.z,
                groupThreadIndex,
                0u),
            loadPackedMeshletTriangleIndex(
                meshletTriangles,
                meshlet.ranges.z,
                groupThreadIndex,
                1u),
            loadPackedMeshletTriangleIndex(
                meshletTriangles,
                meshlet.ranges.z,
                groupThreadIndex,
                2u));
        primitives[groupThreadIndex].visibilityId =
            packVisibilityId(meshletDrawIndex, groupThreadIndex);
    }
}

[shader("fragment")]
[earlydepthstencil]
uint visibilityBufferFragmentMain(uint visibilityId : SV_PrimitiveID) : SV_Target0
{
    return visibilityId;
}

[shader("fragment")]
uint visibilityBufferMaskedFragmentMain(
    VisibilityBufferFragmentInput input,
    uniform GPUDrivenPreviewPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>> materialHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(uint2(
            push.bufferShaderIndexBase + push.materialBuffer,
            0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    uint materialIndex = min(input.materialIndex, max(params.materialCount, 1u) - 1u);
    GPUDrivenPreviewMaterial material = materialHandle[materialIndex];
    if (materialBaseColorAlpha(material, input.texcoord, params, push) < material.params.z) {
        discard;
    }
    return input.visibilityId;
}
