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

bool decodeStreamTriangle(uint index, uint triangle, StructuredBuffer<VisibleClusterRecord> records,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out VisibleClusterRecord record,
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
    uint entry = pageTable[group.pageIndex].x;
    if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
    uint page = (entry & 0xfffffff8u) / 4u;
    pages.GetDimensions(count, stride);
    if (page > count || count - page < 28u) { return false; }
    uint bytes = pages[page + 12];
    uint positionFormat = pages[page + 20];
    uint positionStrideWords = positionFormat == 4u ? 3u : 4u;
    uint positionStrideBytes = positionStrideWords * 4u;
    if (bytes < 112u || bytes / 4u > count - page || (positionFormat != 2u && positionFormat != 4u) ||
        (pages[page + 14] & 1u) == 0u || record.clusterIndex >= pages[page + 2]) { return false; }
    uint clusters = pages[page + 9], positions = pages[page + 10], triangles = pages[page + 11];
    if (((clusters | positions) & 3u) != 0u || clusters > bytes || record.clusterIndex >= (bytes - clusters) / 96u) { return false; }
    uint cluster = page + clusters / 4u + record.clusterIndex * 24u;
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

// Optional shading streams. Position-only pages continue to use geometric
// normals and zero UV; every optional range is validated before dereferencing.
bool decodeStreamAttributes(VisibleClusterRecord record, uint triangle,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3])
{
    for (uint i = 0; i < 3; ++i) { normals[i] = float3(0); uvs[i] = float2(0); }
    uint page = (pageTable[groups[record.dataIndex].pageIndex].x & 0xfffffff8u) / 4u;
    uint bytes = pages[page + 12u], flags = pages[page + 14u];
    uint cluster = page + pages[page + 9u] / 4u + record.clusterIndex * 24u;
    uint address = page * 4u + pages[page + 11u] + pages[cluster + 2u] + triangle * 3u;
    uint normalOffset = pages[page + 16u], uvOffset = pages[page + 17u];
    uint required = pages[cluster] + pages[cluster + 1u];
    bool hasNormal = (flags & 2u) != 0u && pages[page + 21u] == 2u && normalOffset % 4u == 0u &&
        normalOffset <= bytes && required <= (bytes - normalOffset) / 16u;
    bool hasUv = (flags & 4u) != 0u && pages[page + 22u] == 1u && uvOffset % 4u == 0u &&
        uvOffset <= bytes && required <= (bytes - uvOffset) / 8u;
    for (uint i = 0; i < 3; ++i) {
        uint v = pages[cluster] + streamPageByte(pages, address + i);
        if (hasNormal) {
            uint n = page + normalOffset / 4u + v * 4u;
            normals[i] = asfloat(uint3(pages[n], pages[n + 1u], pages[n + 2u]));
        }
        if (hasUv) {
            uint uv = page + uvOffset / 4u + v * 2u;
            uvs[i] = asfloat(uint2(pages[uv], pages[uv + 1u]));
        }
    }
    return hasNormal && hasUv;
}
