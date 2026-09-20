#ifndef METALLIC_VISIBILITY_STREAM_DECODE
#define METALLIC_VISIBILITY_STREAM_DECODE
#include "StreamAttributeDecode.slang"
// Large descriptor-heap buffers can report zero from GetDimensions(). Use the
// producer's bounded pool capacity, just as the raster decoder does. Offset is
// asserted against MeshletStreamGpuParams::pageBufferBytes on the CPU.
uint streamPageWordCount(ByteAddressBuffer params)
{
    return params.Load(100u) / 4u;
}
// Shared validation and position decoding for the cooked StreamAsset format.
// Callers supply their descriptors; no pass-specific bindings live here.
struct StreamActiveGroup {
    uint pageOffset, pageIndex, clusterCount, primitiveIndex;
    uint lodLevel, materialIndex, mask, flags;
    uint instanceIndex, gpuSceneInstanceIndex, padding0, padding1;
    float4 world0, world1, world2, world3;
};
uint streamPageByte(StructuredBuffer<uint> pages, uint address)
{
    return (pages[address / 4] >> ((address & 3u) * 8u)) & 255u;
}
float3 streamPagePosition(StructuredBuffer<uint> pages, uint word)
{
    return asfloat(uint3(pages[word], pages[word + 1], pages[word + 2]));
}

bool decodeStreamTriangleAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages, StructuredBuffer<uint2> pageTable, uint pageWordCount,
    out float3 a, out float3 b, out float3 c)
{
    a = b = c = float3(0);
    uint count, stride;
    pageTable.GetDimensions(count, stride);
    if (pageIndex >= count) { return false; }
    uint entry = pageTable[pageIndex].x;
    if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
    uint page = (entry & 0xfffffff8u) / 4u;
    count = pageWordCount;
    if (page > count || count - page < 28u) { return false; }
    uint bytes = pages[page + 12];
    uint positionFormat = pages[page + 20];
    uint positionStrideWords = positionFormat == 4u ? 3u : 4u;
    uint positionStrideBytes = positionStrideWords * 4u;
    if (bytes < 112u || bytes / 4u > count - page || (positionFormat != 2u && positionFormat != 4u) ||
        (pages[page + 14] & 1u) == 0u || clusterIndex >= pages[page + 2]) { return false; }
    uint clusters = pages[page + 9], positions = pages[page + 10], triangles = pages[page + 11];
    if (((clusters | positions) & 3u) != 0u || clusters > bytes || clusterIndex >= (bytes - clusters) / 96u) { return false; }
    uint cluster = page + clusters / 4u + clusterIndex * 24u;
    uint vertexOffset = pages[cluster], vertexCount = pages[cluster + 1];
    uint triangleOffset = pages[cluster + 2], triangleCount = pages[cluster + 3];
    if (triangle >= triangleCount || triangles > bytes || triangleOffset > bytes - triangles ||
        triangle >= (bytes - triangles - triangleOffset) / 3u || positions > bytes ||
        vertexOffset > (bytes - positions) / positionStrideBytes ||
        vertexCount > (bytes - positions) / positionStrideBytes - vertexOffset) { return false; }
    uint address = page * 4u + triangles + triangleOffset + triangle * 3u;
    uint3 local = uint3(streamPageByte(pages, address), streamPageByte(pages, address + 1), streamPageByte(pages, address + 2));
    if (any(local >= vertexCount)) { return false; }
    uint base = page + positions / 4u + vertexOffset * positionStrideWords;
    a = streamPagePosition(pages, base + local.x * positionStrideWords);
    b = streamPagePosition(pages, base + local.y * positionStrideWords);
    c = streamPagePosition(pages, base + local.z * positionStrideWords);
    return true;
}

bool decodeStreamTriangle(uint index, uint triangle, StructuredBuffer<VisibleClusterRecord> records,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, uint pageWordCount, out VisibleClusterRecord record,
    out float3 a, out float3 b, out float3 c)
{
    record = (VisibleClusterRecord)0; a = b = c = float3(0);
    uint recordCount, recordStride;
    records.GetDimensions(recordCount, recordStride);
    if (index >= recordCount) { return false; }
    record = records[index];
    if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) { return false; }
    uint count, stride;
    groups.GetDimensions(count, stride);
    if (record.dataIndex >= count) { return false; }
    StreamActiveGroup group = groups[record.dataIndex];
    pageTable.GetDimensions(count, stride);
    if (group.pageIndex >= count || record.clusterIndex >= group.clusterCount ||
        record.instanceIndex != group.gpuSceneInstanceIndex) { return false; }
    return decodeStreamTriangleAtPage(group.pageIndex, record.clusterIndex, triangle, pages, pageTable, pageWordCount, a, b, c);
}

// Optional shading streams. Position-only pages continue to use geometric
// normals and zero UV; every optional range is validated before dereferencing.
uint decodeStreamAttributesAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3], out float4 tangents[3])
{
    for (uint i = 0; i < 3; ++i) { normals[i] = float3(0); uvs[i] = float2(0); tangents[i] = float4(0); }
    uint page = (pageTable[pageIndex].x & 0xfffffff8u) / 4u;
    uint bytes = pages[page + 12u], flags = pages[page + 14u];
    uint cluster = page + pages[page + 9u] / 4u + clusterIndex * 24u;
    uint address = page * 4u + pages[page + 11u] + pages[cluster + 2u] + triangle * 3u;
    uint normalOffset = pages[page + 16u], uvOffset = pages[page + 17u];
    uint required = pages[cluster] + pages[cluster + 1u];
    uint normalFormat = pages[page + 21u], normalStride = streamNormalStride(normalFormat);
    bool hasNormal = (flags & 2u) != 0u && normalStride != 0u && normalOffset % 4u == 0u &&
        normalOffset <= bytes && required <= (bytes - normalOffset) / max(normalStride, 1u);
    bool hasUv = (flags & 4u) != 0u && pages[page + 22u] == 1u && uvOffset % 4u == 0u &&
        uvOffset <= bytes && required <= (bytes - uvOffset) / 8u;
    uint tangentOffset = pages[page + 24u];
    uint tangentFormat = pages[page + 25u], tangentStride = streamTangentStride(tangentFormat);
    bool hasTangent = (flags & 16u) != 0u && tangentStride != 0u && tangentOffset % 4u == 0u &&
        tangentOffset <= bytes && required <= (bytes - tangentOffset) / max(tangentStride, 1u);
    for (uint i = 0; i < 3; ++i) {
        uint v = pages[cluster] + streamPageByte(pages, address + i);
        if (hasNormal) {
            normals[i] = streamLoadNormal(pages,page,normalOffset,v,normalFormat);
        }
        if (hasUv) {
            uint uv = page + uvOffset / 4u + v * 2u;
            uvs[i] = asfloat(uint2(pages[uv], pages[uv + 1u]));
        }
        if (hasTangent) {
            tangents[i] = streamLoadTangent(pages,page,tangentOffset,v,tangentFormat);
        }
    }
    return (hasNormal ? 2u : 0u) | (hasUv ? 4u : 0u) | (hasTangent ? 16u : 0u);
}

uint decodeStreamAttributes(VisibleClusterRecord record, uint triangle,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3], out float4 tangents[3])
{
    return decodeStreamAttributesAtPage(groups[record.dataIndex].pageIndex, record.clusterIndex, triangle,
        pages, pageTable, normals, uvs, tangents);
}
#endif
