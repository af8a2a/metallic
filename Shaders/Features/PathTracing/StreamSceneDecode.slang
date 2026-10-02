#ifndef METALLIC_STREAM_SCENE_DECODE
#define METALLIC_STREAM_SCENE_DECODE
#include "../../../Source/Runtime/Render/Core/StreamSceneParameters.h"
import GPUDriven;
using Metallic.GPUDriven;
#include "../VisibilityBuffer/VisibilityStreamDecode.slang"

bool loadStreamRayInstance(StreamSceneParameters scene, uint index, out StreamRayInstance instance)
{
    instance = (StreamRayInstance)0;
    if (scene.instances.data == nullptr || !scene.instances.contains(index)) { return false; }
    instance = scene.instances.data[index];
    return true;
}

// CLAS IDs are pageIndex * maxPageClusters + localCluster. IDs survive arena moves.
// Validate the complete position chain before callers decode optional attributes.
bool decodeStreamRayTriangle(StreamSceneParameters scene, uint clusterId, uint triangle,
    out uint pageIndex, out uint cluster, out float3 a, out float3 b, out float3 c)
{
    pageIndex = cluster = 0;
    a = b = c = float3(0);
    if (scene.header.data == nullptr || !scene.header.contains(2u) ||
        scene.pages.data == nullptr || scene.pageTable.data == nullptr) { return false; }
    uint clusterStride = scene.header.data[2];
    if (clusterStride == 0u) { return false; }
    pageIndex = clusterId / clusterStride;
    cluster = clusterId % clusterStride;
    return decodeStreamTriangleAtPage(pageIndex, cluster, triangle,
        StreamDataBuffer<uint>(scene.pages), StreamDataBuffer<uint2>(scene.pageTable),
        scene.pages.length(), a, b, c);
}
// Keep optional attribute reads behind validation of the same triangle.
bool decodeStreamRaySurface(StreamSceneParameters scene, uint clusterId, uint triangle,
    out float3 a, out float3 b, out float3 c,
    out float3 normals[3], out float2 uvs[3], out float4 tangents[3], out uint attributes)
{
    attributes = 0;
    for (uint i = 0; i < 3; ++i) { normals[i] = float3(0); uvs[i] = float2(0); tangents[i] = float4(0); }
    uint pageIndex, cluster;
    if (!decodeStreamRayTriangle(scene, clusterId, triangle, pageIndex, cluster, a, b, c)) { return false; }
    attributes = decodeStreamAttributesAtPage(pageIndex, cluster, triangle,
        StreamDataBuffer<uint>(scene.pages), StreamDataBuffer<uint2>(scene.pageTable), normals, uvs, tangents);
    return true;
}
#endif
