#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Deferred debug shading and full-screen composition.
float edgeFunction(float2 a, float2 b, float2 point)
{
    float2 edge = b - a;
    float2 offset = point - a;
    return edge.x * offset.y - edge.y * offset.x;
}

uint packStreamColor(float3 color)
{
    uint3 encoded = uint3(round(saturate(color) * 255.0));
    return encoded.x | (encoded.y << 8u) | (encoded.z << 16u) | 0xff000000u;
}

float4 unpackStreamColor(uint packed)
{
    return float4(
        float(packed & 0xffu),
        float((packed >> 8u) & 0xffu),
        float((packed >> 16u) & 0xffu),
        float((packed >> 24u) & 0xffu)) / 255.0;
}

float3 streamPerspectiveBarycentrics(
    float2 pixelCenter,
    float4 clip0,
    float4 clip1,
    float4 clip2,
    uint width,
    uint height)
{
    float2 viewport = float2(max(width, 1u), max(height, 1u));
    float2 screen0 = (clip0.xy / max(abs(clip0.w), 0.000001) * 0.5 + 0.5) * viewport;
    float2 screen1 = (clip1.xy / max(abs(clip1.w), 0.000001) * 0.5 + 0.5) * viewport;
    float2 screen2 = (clip2.xy / max(abs(clip2.w), 0.000001) * 0.5 + 0.5) * viewport;
    float area = edgeFunction(screen0, screen1, screen2);
    if (abs(area) < 0.000001) {
        return float3(1.0, 0.0, 0.0);
    }
    float3 affine = float3(
        edgeFunction(screen1, screen2, pixelCenter),
        edgeFunction(screen2, screen0, pixelCenter),
        edgeFunction(screen0, screen1, pixelCenter)) / area;
    float3 weighted = affine / float3(
        max(abs(clip0.w), 0.000001),
        max(abs(clip1.w), 0.000001),
        max(abs(clip2.w), 0.000001));
    float weight = weighted.x + weighted.y + weighted.z;
    return abs(weight) > 0.000001 ? weighted / weight : affine;
}

float3 loadPayloadPosition(StructuredBuffer<uint> words, uint pageWord, uint offsetBytes, uint index, uint format)
{
    uint base = pageWord + offsetBytes / 4u + index * (format == kPayloadFormatFloat32x3 ? 3u : 4u);
    return asfloat(uint3(words[base], words[base + 1u], words[base + 2u]));
}

float4 loadPayloadFloat4(
    StructuredBuffer<uint> pageWords,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 4u;
    return float4(
        asfloat(pageWords[word + 0u]),
        asfloat(pageWords[word + 1u]),
        asfloat(pageWords[word + 2u]),
        asfloat(pageWords[word + 3u]));
}

float2 loadPayloadFloat2(
    StructuredBuffer<uint> pageWords,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 2u;
    return float2(
        asfloat(pageWords[word + 0u]),
        asfloat(pageWords[word + 1u]));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenStreamAssetDeferredMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    uint width = max(rasterBindings.width, 1u);
    uint height = max(rasterBindings.height, 1u);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height) {
        return;
    }

    Texture2D<uint> visibilityHandle =
        resolveUniform(ResourceHandle<Texture2D<uint>>(rasterBindings.visibilityImage));
    RWStructuredBuffer<uint> outputHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(rasterBindings.deferredColorBuffer));
    Texture2D<uint> visibility = visibilityHandle;
    RWStructuredBuffer<uint> output = outputHandle;
    uint outputIndex = dispatchThreadID.y * width + dispatchThreadID.x;
    uint packedVisibility = visibility.Load(int3(dispatchThreadID.xy, 0));
    uint recordIndex = 0u;
    uint triangleIndex = 0u;
    if (!unpackVisibilityId(packedVisibility, recordIndex, triangleIndex)) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint visibleRecordCapacity = rasterBindings.visibleRecordCapacity != 0u
        ? rasterBindings.visibleRecordCapacity
        : params.drawTaskCount / kMeshletTriangleChunkCount;
    if (!visibilityRecordRangeValid(
            rasterBindings.visibleRecordBase,
            visibleRecordCapacity) ||
        recordIndex < rasterBindings.visibleRecordBase ||
        recordIndex - rasterBindings.visibleRecordBase >= visibleRecordCapacity) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    StructuredBuffer<CompactStreamVisibleRecord> visibleClusterHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<CompactStreamVisibleRecord>>(rasterBindings.visibleClusterBuffer));
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroupHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
    StructuredBuffer<uint> pageHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    StructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    StructuredBuffer<CompactStreamVisibleRecord> visibleClusters = visibleClusterHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;

    uint localRecordIndex = recordIndex - rasterBindings.visibleRecordBase;
    VisibleClusterRecord record = unpackStreamVisibleRecord(visibleClusters[localRecordIndex]);
    if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetActiveHeader activeHeader;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaderHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaders = activeHeaderHandle;
    activeHeader = activeHeaders[0];
    uint activeGroupCount = min(activeHeader.activeGroupCount, activeHeader.activeGroupCapacity);
    if (record.dataIndex >= activeGroupCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetActiveGroup group = activeGroups[record.dataIndex];
    record.instanceIndex = group.gpuSceneInstanceIndex;
    if (record.instanceIndex == kInvalidClusterIndex ||
        record.instanceIndex != group.gpuSceneInstanceIndex ||
        group.pageIndex >= params.scenePageCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (!streamPageDrawable(streamPageState(pageEntry)) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset ||
        params.pageBufferBytes < kPayloadHeaderBytes ||
        pageDeviceOffset > params.pageBufferBytes - kPayloadHeaderBytes) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    if (payloadBytes < kPayloadHeaderBytes ||
        payloadBytes > params.pageBufferBytes ||
        pageDeviceOffset > params.pageBufferBytes - payloadBytes ||
        record.clusterIndex >= clusterCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    uint attributeFlags = pageWords[pageWord + 14u];
    uint normalOffsetBytes = pageWords[pageWord + 16u];
    uint texcoordOffsetBytes = pageWords[pageWord + 17u];
    uint positionFormat = pageWords[pageWord + 20u];
    uint normalFormat = pageWords[pageWord + 21u];
    uint texcoordFormat = pageWords[pageWord + 22u];
    uint tangentOffsetBytes = pageWords[pageWord + 24u];
    uint tangentFormat = pageWords[pageWord + 25u];
    bool hasNormals = (attributeFlags & kPayloadAttributeNormal) != 0u;
    bool hasTexcoords = (attributeFlags & kPayloadAttributeTexcoord0) != 0u;
    bool hasTangents = (attributeFlags & kPayloadAttributeTangent) != 0u;
    if ((attributeFlags & kPayloadAttributePosition) == 0u ||
        (positionFormat != kPayloadFormatFloat32x4 && positionFormat != kPayloadFormatFloat32x3) ||
        (clusterOffsetBytes & 3u) != 0u ||
        (positionOffsetBytes & 3u) != 0u ||
        !streamPayloadArrayRangeValid(
            clusterOffsetBytes,
            clusterCount,
            kPayloadClusterBytes,
            payloadBytes) ||
        !streamPayloadArrayRangeValid(
            positionOffsetBytes,
            payloadVertexCount,
            positionFormat == kPayloadFormatFloat32x3 ? 12u : 16u,
            payloadBytes) ||
        !streamPayloadRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes) ||
        (hasNormals &&
            (streamNormalStride(normalFormat) == 0u ||
             (normalOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 normalOffsetBytes,
                 payloadVertexCount,
                 streamNormalStride(normalFormat),
                 payloadBytes))) ||
        (hasTexcoords &&
            (texcoordFormat != kPayloadFormatFloat32x2 ||
             (texcoordOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 texcoordOffsetBytes,
                 payloadVertexCount,
                 8u,
                 payloadBytes))) ||
        (hasTangents &&
            (streamTangentStride(tangentFormat) == 0u ||
             (tangentOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 tangentOffsetBytes,
                 payloadVertexCount,
                 streamTangentStride(tangentFormat),
                 payloadBytes)))) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint clusterWord = pageWord + clusterOffsetBytes / 4u +
        record.clusterIndex * kPayloadClusterWords;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = pageWords[clusterWord + 1u];
    uint localTriangleOffset = pageWords[clusterWord + 2u];
    uint triangleCount = pageWords[clusterWord + 3u];
    if (triangleIndex >= triangleCount ||
        triangleCount > kMaxMeshletTriangles ||
        vertexCount == 0u ||
        vertexCount > kMaxMeshletVertices ||
        vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset ||
        localTriangleOffset > triangleIndexCount ||
        triangleCount > (triangleIndexCount - localTriangleOffset) / 3u) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint triangleByte = pageDeviceOffset + triangleOffsetBytes +
        localTriangleOffset + triangleIndex * 3u;
    uint3 localIndices = uint3(
        readPageByte(pageWords, triangleByte + 0u),
        readPageByte(pageWords, triangleByte + 1u),
        readPageByte(pageWords, triangleByte + 2u));
    if (any(localIndices >= vertexCount)) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint3 vertexIndices = vertexOffset + localIndices;
    float3 local0 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.x, positionFormat);
    float3 local1 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.y, positionFormat);
    float3 local2 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.z, positionFormat);
    float3 world0 = group.world0.xyz * local0.x + group.world1.xyz * local0.y +
        group.world2.xyz * local0.z + group.world3.xyz;
    float3 world1 = group.world0.xyz * local1.x + group.world1.xyz * local1.y +
        group.world2.xyz * local1.z + group.world3.xyz;
    float3 world2 = group.world0.xyz * local2.x + group.world1.xyz * local2.y +
        group.world2.xyz * local2.z + group.world3.xyz;
    float4 clip0 = projectPosition(world0, params);
    float4 clip1 = projectPosition(world1, params);
    float4 clip2 = projectPosition(world2, params);
    clip0.y = -clip0.y;
    clip1.y = -clip1.y;
    clip2.y = -clip2.y;
    float3 barycentrics = streamPerspectiveBarycentrics(
        float2(dispatchThreadID.xy) + 0.5,
        clip0,
        clip1,
        clip2,
        width,
        height);

    float3 geometryNormal = safeNormalize(
        cross(world1 - world0, world2 - world0),
        float3(0.0, 1.0, 0.0));
    float3 authoredNormal = geometryNormal;
    if (hasNormals) {
        float3 normal0 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.x,normalFormat);
        float3 normal1 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.y,normalFormat);
        float3 normal2 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.z,normalFormat);
        float3 localNormal = normal0 * barycentrics.x + normal1 * barycentrics.y +
            normal2 * barycentrics.z;
        authoredNormal = transformStreamNormal(
            group,
            localNormal,
            geometryNormal);
    }
    if (dot(authoredNormal, geometryNormal) < 0.0) {
        authoredNormal = -authoredNormal;
    }

    float3 tangent = safeNormalize(
        cross(abs(authoredNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), authoredNormal),
        float3(1.0, 0.0, 0.0));
    float handedness = 1.0;
    if (hasTangents) {
        float4 tangent0 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.x,tangentFormat);
        float4 tangent1 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.y,tangentFormat);
        float4 tangent2 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.z,tangentFormat);
        float4 localTangent = tangent0 * barycentrics.x + tangent1 * barycentrics.y +
            tangent2 * barycentrics.z;
        float3 worldTangent = transformStreamDirection(group, localTangent.xyz);
        tangent = safeNormalize(
            worldTangent - authoredNormal * dot(authoredNormal, worldTangent),
            tangent);
        handedness = localTangent.w < 0.0 ? -1.0 : 1.0;
    }
    float3 bitangent = safeNormalize(
        cross(authoredNormal, tangent) * handedness,
        cross(authoredNormal, tangent));
    float3 shadingNormal = authoredNormal;
    float2 uv = float2(0.0, 0.0);
    if (hasTexcoords) {
        float2 uv0 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.x);
        float2 uv1 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.y);
        float2 uv2 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.z);
        uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
    }

    uint colorSeed = group.pageIndex;
    if (params.debugColorMode == 1u) {
        colorSeed = group.lodLevel;
    } else if (params.debugColorMode == 2u) {
        colorSeed = group.primitiveIndex;
    } else if (params.debugColorMode == 3u) {
        colorSeed = group.instanceIndex;
    } else if (params.debugColorMode == 4u) {
        colorSeed = ((group.pageIndex + 1u) * 0x9e3779b9u) ^ record.clusterIndex;
    }
    float3 color = hashColor(colorSeed + 1u);
    if (params.debugColorMode == 5u) {
        float3 lightDirection = safeNormalize(float3(0.35, 0.85, 0.4), float3(0.0, 1.0, 0.0));
        float diffuse = 0.18 + 0.82 * max(dot(shadingNormal, lightDirection), 0.0);
        float tangentPattern = 0.92 + 0.08 * sin((dot(uv, float2(1.0, 0.0)) + dot(bitangent, tangent)) * 40.0);
        color = float3(0.18, 0.42, 0.12) * diffuse * tangentPattern;
    }
    output[outputIndex] = packStreamColor(color);
}

[shader("vertex")]
GPUDrivenStreamAssetFullscreenOutput gpuDrivenStreamAssetCompositeVertexMain(
    uint vertexID : SV_VertexID)
{
    GPUDrivenStreamAssetFullscreenOutput output;
    float2 position = vertexID == 0u
        ? float2(-1.0, -1.0)
        : (vertexID == 1u ? float2(3.0, -1.0) : float2(-1.0, 3.0));
    output.position = float4(position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 gpuDrivenStreamAssetCompositeFragmentMain(
    GPUDrivenStreamAssetFullscreenOutput input,
    uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    uint width = max(rasterBindings.width, 1u);
    uint height = max(rasterBindings.height, 1u);
    uint2 pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    StructuredBuffer<uint> colorHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(rasterBindings.deferredColorBuffer));
    StructuredBuffer<uint> colors = colorHandle;
    return unpackStreamColor(colors[pixel.y * width + pixel.x]);
}

