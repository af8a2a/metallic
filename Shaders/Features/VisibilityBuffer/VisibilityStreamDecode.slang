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
    if (bytes < 112u || bytes / 4u > count - page || pages[page + 20] != 2u ||
        (pages[page + 14] & 1u) == 0u || record.clusterIndex >= pages[page + 2]) { return false; }
    uint clusters = pages[page + 9], positions = pages[page + 10], triangles = pages[page + 11];
    if (((clusters | positions) & 3u) != 0u || clusters > bytes || record.clusterIndex >= (bytes - clusters) / 96u) { return false; }
    uint cluster = page + clusters / 4u + record.clusterIndex * 24u;
    uint vertexOffset = pages[cluster], vertexCount = pages[cluster + 1];
    uint triangleOffset = pages[cluster + 2], triangleCount = pages[cluster + 3];
    if (triangle >= triangleCount || triangles > bytes || triangleOffset > bytes - triangles ||
        triangle >= (bytes - triangles - triangleOffset) / 3u || positions > bytes ||
        vertexOffset > (bytes - positions) / 16u || vertexCount > (bytes - positions) / 16u - vertexOffset) { return false; }
    uint address = page * 4u + triangles + triangleOffset + triangle * 3u;
    uint3 local = uint3(streamPageByte(pages, address), streamPageByte(pages, address + 1), streamPageByte(pages, address + 2));
    if (any(local >= vertexCount)) { return false; }
    uint base = page + positions / 4u + vertexOffset * 4u;
    a = streamPagePosition(pages, base + local.x * 4u); b = streamPagePosition(pages, base + local.y * 4u); c = streamPagePosition(pages, base + local.z * 4u);
    return true;
}
