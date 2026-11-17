implementing VisibilityBufferShading;

import GPUDriven;
using Metallic.GPUDriven;
import Core;
import MaterialTextureDecode;
using Metallic;
import Material;
using Metallic.Material;

import OpenPBR;
import OpenPBRClosure;
import OpenPBRTextureLUT;
import MaterialProgram;
using Metallic.OpenPBR;

uint streamDeferredPageState(GPUDrivenStreamDeferredPageTableEntry entry)
{
    return entry.deviceOffsetAndState & kStreamPageStateMask;
}

uint streamDeferredPageOffset(GPUDrivenStreamDeferredPageTableEntry entry)
{
    return streamDeferredPageState(entry) == 0u
        ? kStreamInvalidDeviceOffset
        : entry.deviceOffsetAndState & kStreamPageDeviceOffsetMask;
}

bool streamDeferredPageDrawable(uint state)
{
    return state == kStreamStateResident || state == kStreamStateLockedFallback;
}

bool streamDeferredRangeValid(uint byteOffset, uint byteCount, uint payloadBytes)
{
    return byteOffset <= payloadBytes && byteCount <= payloadBytes - byteOffset;
}

bool streamDeferredArrayRangeValid(
    uint byteOffset,
    uint elementCount,
    uint elementStride,
    uint payloadBytes)
{
    return elementStride != 0u &&
        byteOffset <= payloadBytes &&
        elementCount <= (payloadBytes - byteOffset) / elementStride;
}

uint streamDeferredReadByte(StructuredBuffer<uint> words, uint byteOffset)
{
    uint word = words[byteOffset >> 2u];
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

float3 streamDeferredLoadPosition(StructuredBuffer<uint> words, uint pageWord, uint offsetBytes, uint index, uint format)
{
    uint base = pageWord + offsetBytes / 4u + index * (format == kStreamPayloadFormatFloat32x3 ? 3u : 4u);
    return asfloat(uint3(words[base], words[base + 1u], words[base + 2u]));
}

float4 streamDeferredLoadFloat4(
    StructuredBuffer<uint> words,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 4u;
    return float4(
        asfloat(words[word + 0u]),
        asfloat(words[word + 1u]),
        asfloat(words[word + 2u]),
        asfloat(words[word + 3u]));
}

float2 streamDeferredLoadFloat2(
    StructuredBuffer<uint> words,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 2u;
    return float2(asfloat(words[word + 0u]), asfloat(words[word + 1u]));
}

SceneTransform streamDeferredTransform(GPUDrivenStreamDeferredActiveGroup group)
{
    SceneTransform transform;
    transform.world0 = group.world0;
    transform.world1 = group.world1;
    transform.world2 = group.world2;
    transform.world3 = group.world3;
    return transform;
}

float4 projectStreamDeferredPosition(
    float3 worldPosition,
    GPUDrivenStreamDeferredParams params)
{
    if (params.renderViewport.y > 0.0) {
        params.eye = params.renderEye;
        params.center = params.renderCenter;
        params.upProjection = params.renderUpProjection;
        params.viewport = params.renderViewport;
        params.clipOrtho = params.renderClipOrtho;
    }
    float3 forward = safeNormalize(
        params.center.xyz - params.eye.xyz,
        float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(params.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - params.eye.xyz;
    float3 viewPosition = float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));
    float zNear = max(params.clipOrtho.x, 0.0001);
    float zFar = max(params.clipOrtho.y, zNear + 0.0001);
    float ndcZ = conservativeProjectDepth(viewPosition.z, zNear, zFar, params.upProjection.w > 0.5);
    float aspect = max(params.viewport.x, 0.001);
    float4 clip;
    if (params.upProjection.w > 0.5) {
        float halfHeight = max(params.clipOrtho.z * 0.5, 0.0001);
        clip = float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    } else {
        float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
        float yScale = 1.0 / tan(fovRadians * 0.5);
        float w = viewPosition.z;
        clip = float4(
            viewPosition.x * yScale / aspect,
            viewPosition.y * yScale,
            ndcZ * w,
            w);
    }
    if (params.renderViewport.y > 0.0) {
        // The caller flips clip Y for Vulkan after this projection. Negate the
        // Y jitter here to match the post-flip raster pixel offset.
        clip.xy -= 2.0 * float2(params.renderEye.w, -params.renderCenter.w) /
            max(params.renderViewport.yz, float2(1.0)) * clip.w;
    }
    return clip;
}

bool decodeStreamDeferredSurface(
    VisibleClusterRecord record,
    uint triangleIndex,
    float2 pixelCenter,
    GPUDrivenPreviewParams previewParams,
    GPUDrivenPreviewPush push,
    GPUDrivenStreamDeferredBindings bindings,
    out GPUDrivenDeferredSurface surface)
{
    surface.worldPosition = float3(0.0);
    surface.geometryNormal = float3(0.0, 0.0, 1.0);
    surface.normal = surface.geometryNormal;
    surface.tangent = float3(1.0, 0.0, 0.0);
    surface.texcoord = float2(0.0);
    surface.tangentSign = 1.0;
    surface.materialIndex = 0u;
    surface.primitiveIndex = 0u;
    surface.lodLevel = 0u;

    StructuredBuffer<GPUDrivenStreamDeferredParams> streamParamsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamDeferredParams>>(bindings.paramsBuffer));
    StructuredBuffer<GPUDrivenStreamDeferredActiveHeader> activeHeaderHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamDeferredActiveHeader>>(bindings.activeHeaderBuffer));
    StructuredBuffer<GPUDrivenStreamDeferredActiveGroup> activeGroupHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamDeferredActiveGroup>>(bindings.activeGroupBuffer));
    StructuredBuffer<GPUDrivenStreamDeferredPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamDeferredPageTableEntry>>(bindings.pageTableBuffer));
    StructuredBuffer<uint> pageHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(bindings.pageBuffer));
    StructuredBuffer<GPUDrivenStreamDeferredParams> streamParamsBuffer = streamParamsHandle;
    StructuredBuffer<GPUDrivenStreamDeferredActiveHeader> activeHeaders = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamDeferredActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<GPUDrivenStreamDeferredPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<uint> pageWords = pageHandle;
    GPUDrivenStreamDeferredParams streamParams = streamParamsBuffer[0];
    GPUDrivenStreamDeferredActiveHeader activeHeader = activeHeaders[0];
    uint activeGroupCount = min(activeHeader.activeGroupCount, activeHeader.activeGroupCapacity);
    if (record.dataIndex >= activeGroupCount) {
        return false;
    }
    GPUDrivenStreamDeferredActiveGroup group = activeGroups[record.dataIndex];
    record.instanceIndex = group.gpuSceneInstanceIndex;
    if (record.instanceIndex >= previewParams.instanceCount ||
        group.pageIndex >= streamParams.scenePageCount ||
        record.clusterIndex >= group.clusterCount) {
        return false;
    }
    GPUDrivenStreamDeferredPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageDeviceOffset = streamDeferredPageOffset(pageEntry);
    if (!streamDeferredPageDrawable(streamDeferredPageState(pageEntry)) ||
        pageDeviceOffset == kStreamInvalidDeviceOffset ||
        streamParams.pageBufferBytes < kStreamPayloadHeaderBytes ||
        pageDeviceOffset > streamParams.pageBufferBytes - kStreamPayloadHeaderBytes) {
        return false;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    if (payloadBytes < kStreamPayloadHeaderBytes ||
        payloadBytes > streamParams.pageBufferBytes ||
        pageDeviceOffset > streamParams.pageBufferBytes - payloadBytes ||
        record.clusterIndex >= clusterCount) {
        return false;
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
    bool hasNormals = (attributeFlags & kStreamPayloadAttributeNormal) != 0u;
    bool hasTexcoords = (attributeFlags & kStreamPayloadAttributeTexcoord0) != 0u;
    bool hasTangents = (attributeFlags & kStreamPayloadAttributeTangent) != 0u;
    if ((attributeFlags & kStreamPayloadAttributePosition) == 0u ||
        (positionFormat != kStreamPayloadFormatFloat32x4 && positionFormat != kStreamPayloadFormatFloat32x3) ||
        (clusterOffsetBytes & 3u) != 0u ||
        (positionOffsetBytes & 3u) != 0u ||
        !streamDeferredArrayRangeValid(
            clusterOffsetBytes,
            clusterCount,
            kStreamPayloadClusterBytes,
            payloadBytes) ||
        !streamDeferredArrayRangeValid(
            positionOffsetBytes,
            payloadVertexCount,
            positionFormat == kStreamPayloadFormatFloat32x3 ? 12u : 16u,
            payloadBytes) ||
        !streamDeferredRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes) ||
        (hasNormals &&
            (streamNormalStride(normalFormat) == 0u ||
             (normalOffsetBytes & 3u) != 0u ||
             !streamDeferredArrayRangeValid(
                 normalOffsetBytes,
                 payloadVertexCount,
                 streamNormalStride(normalFormat),
                 payloadBytes))) ||
        (hasTexcoords &&
            (texcoordFormat != kStreamPayloadFormatFloat32x2 ||
             (texcoordOffsetBytes & 3u) != 0u ||
             !streamDeferredArrayRangeValid(
                 texcoordOffsetBytes,
                 payloadVertexCount,
                 8u,
                 payloadBytes))) ||
        (hasTangents &&
            (streamTangentStride(tangentFormat) == 0u ||
             (tangentOffsetBytes & 3u) != 0u ||
             !streamDeferredArrayRangeValid(
                 tangentOffsetBytes,
                 payloadVertexCount,
                 streamTangentStride(tangentFormat),
                 payloadBytes)))) {
        return false;
    }

    uint clusterWord = pageWord + clusterOffsetBytes / 4u +
        record.clusterIndex * kStreamPayloadClusterWords;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = pageWords[clusterWord + 1u];
    uint localTriangleOffset = pageWords[clusterWord + 2u];
    uint triangleCount = pageWords[clusterWord + 3u];
    if (triangleIndex >= triangleCount ||
        triangleCount > kStreamMaxMeshletTriangles ||
        vertexCount == 0u ||
        vertexCount > kStreamMaxMeshletVertices ||
        vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset ||
        localTriangleOffset > triangleIndexCount ||
        triangleCount > (triangleIndexCount - localTriangleOffset) / 3u) {
        return false;
    }
    uint triangleByte = pageDeviceOffset + triangleOffsetBytes +
        localTriangleOffset + triangleIndex * 3u;
    uint3 localIndices = uint3(
        streamDeferredReadByte(pageWords, triangleByte + 0u),
        streamDeferredReadByte(pageWords, triangleByte + 1u),
        streamDeferredReadByte(pageWords, triangleByte + 2u));
    if (any(localIndices >= vertexCount)) {
        return false;
    }

    uint3 vertexIndices = vertexOffset + localIndices;
    float3 local0 = streamDeferredLoadPosition(
        pageWords,
        pageWord,
        positionOffsetBytes,
        vertexIndices.x, positionFormat);
    float3 local1 = streamDeferredLoadPosition(
        pageWords,
        pageWord,
        positionOffsetBytes,
        vertexIndices.y, positionFormat);
    float3 local2 = streamDeferredLoadPosition(
        pageWords,
        pageWord,
        positionOffsetBytes,
        vertexIndices.z, positionFormat);
    SceneTransform transform = streamDeferredTransform(group);
    float3 world0 = transformPoint(transform, local0);
    float3 world1 = transformPoint(transform, local1);
    float3 world2 = transformPoint(transform, local2);
    float4 clip0 = projectStreamDeferredPosition(world0, streamParams);
    float4 clip1 = projectStreamDeferredPosition(world1, streamParams);
    float4 clip2 = projectStreamDeferredPosition(world2, streamParams);
    clip0.y = -clip0.y;
    clip1.y = -clip1.y;
    clip2.y = -clip2.y;
    float3 barycentrics = perspectiveBarycentrics(
        pixelCenter,
        clip0,
        clip1,
        clip2,
        previewParams);
    surface.worldPosition = world0 * barycentrics.x +
        world1 * barycentrics.y + world2 * barycentrics.z;
    surface.geometryNormal = safeNormalize(
        cross(world1 - world0, world2 - world0),
        float3(0.0, 0.0, 1.0));
    surface.normal = surface.geometryNormal;
    if (hasNormals) {
        float3 normal0 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.x,normalFormat);
        float3 normal1 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.y,normalFormat);
        float3 normal2 = streamLoadNormal(pageWords,pageWord,normalOffsetBytes,vertexIndices.z,normalFormat);
        float3 localNormal = normal0 * barycentrics.x +
            normal1 * barycentrics.y + normal2 * barycentrics.z;
        surface.normal = transformNormal(transform, localNormal);
    }
    if (dot(surface.normal, surface.geometryNormal) < 0.0) {
        surface.normal = -surface.normal;
    }
    surface.tangent = safeNormalize(
        cross(
            abs(surface.normal.y) < 0.999
                ? float3(0.0, 1.0, 0.0)
                : float3(1.0, 0.0, 0.0),
            surface.normal),
        float3(1.0, 0.0, 0.0));
    if (hasTangents) {
        float4 tangent0 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.x,tangentFormat);
        float4 tangent1 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.y,tangentFormat);
        float4 tangent2 = streamLoadTangent(pageWords,pageWord,tangentOffsetBytes,vertexIndices.z,tangentFormat);
        float4 localTangent = tangent0 * barycentrics.x +
            tangent1 * barycentrics.y + tangent2 * barycentrics.z;
        float3 worldTangent = transformDirection(transform, localTangent.xyz);
        surface.tangent = safeNormalize(
            worldTangent - surface.normal * dot(surface.normal, worldTangent),
            surface.tangent);
        surface.tangentSign = localTangent.w < 0.0 ? -1.0 : 1.0;
    }
    if (hasTexcoords) {
        float2 uv0 = streamDeferredLoadFloat2(
            pageWords,
            pageWord,
            texcoordOffsetBytes,
            vertexIndices.x);
        float2 uv1 = streamDeferredLoadFloat2(
            pageWords,
            pageWord,
            texcoordOffsetBytes,
            vertexIndices.y);
        float2 uv2 = streamDeferredLoadFloat2(
            pageWords,
            pageWord,
            texcoordOffsetBytes,
            vertexIndices.z);
        surface.texcoord = uv0 * barycentrics.x + uv1 * barycentrics.y +
            uv2 * barycentrics.z;
    }
    StructuredBuffer<GPUDrivenPreviewInstance> instanceHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenPreviewInstance> instances = instanceHandle;
    GPUDrivenPreviewInstance instance = instances[record.instanceIndex];
    surface.materialIndex = instance.identity.y;
    surface.primitiveIndex = group.primitiveIndex;
    surface.lodLevel = group.lodLevel;
    return true;
}
bool decodeStreamVisibilitySurface(
    uint recordIndex,
    uint triangleIndex,
    float2 pixelCenter,
    GPUDrivenPreviewParams params,
    GPUDrivenPreviewPush push,
    out GPUDrivenDeferredSurface surface)
{
    surface = (GPUDrivenDeferredSurface)0;
    if (push.streamDeferredBindingsBuffer == kInvalidTextureIndex) {
        return false;
    }
    StructuredBuffer<GPUDrivenStreamDeferredBindings> bindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamDeferredBindings>>(push.streamDeferredBindingsBuffer));
    StructuredBuffer<GPUDrivenStreamDeferredBindings> bindingsBuffer = bindingsHandle;
    GPUDrivenStreamDeferredBindings bindings = bindingsBuffer[0];
    if (!visibilityRecordRangeValid(
            bindings.visibleRecordBase,
            bindings.visibleRecordCapacity) ||
        recordIndex < bindings.visibleRecordBase ||
        recordIndex - bindings.visibleRecordBase >= bindings.visibleRecordCapacity) {
        return false;
    }

    uint localRecordIndex = recordIndex - bindings.visibleRecordBase;
    StructuredBuffer<CompactStreamVisibleRecord> streamRecordHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<CompactStreamVisibleRecord>>(bindings.visibleClusterBuffer));
    StructuredBuffer<CompactStreamVisibleRecord> streamRecords = streamRecordHandle;
    VisibleClusterRecord record = unpackStreamVisibleRecord(streamRecords[localRecordIndex]);
    if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) {
        return false;
    }

    if (!decodeStreamDeferredSurface(
            record,
            triangleIndex,
            pixelCenter,
            params,
            push,
            bindings,
            surface)) {
        return false;
    }
    surface.lodGroupIndex = record.clusterIndex;
    return true;
}
