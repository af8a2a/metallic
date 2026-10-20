#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
// Large descriptor-heap buffers can report zero from GetDimensions(). Use the
// producer's bounded pool capacity, just as the raster decoder does. Offset is
// asserted against MeshletStreamGPUParams::pageBufferBytes on the CPU.
public uint streamPageWordCount(ByteAddressBuffer params)
{
    return params.Load(100u) / 4u;
}
// Shared validation and position decoding for the cooked StreamAsset format.
// Callers supply their descriptors; no pass-specific bindings live here.
public struct StreamActiveGroup {
    public uint pageOffset, pageIndex, clusterCount, primitiveIndex;
    public uint lodLevel, materialIndex, mask, flags;
    public uint instanceIndex, gpuSceneInstanceIndex, padding0, padding1;
    public float4 world0, world1, world2, world3;
};
public uint streamPageByte(StructuredBuffer<uint> pages, uint address)
{
    return (pages[address / 4] >> ((address & 3u) * 8u)) & 255u;
}
public float3 streamPagePosition(StructuredBuffer<uint> pages, uint word)
{
    return asfloat(uint3(pages[word], pages[word + 1], pages[word + 2]));
}

// Validated triangle metadata reused immediately by optional attribute decoding.
// Vertices are absolute within the page's vertex streams, not cluster-local.
public struct StreamTriangleDecodeContext
{
    public uint page, bytes, flags, requiredVertices;
    public uint3 vertices;
};

public bool decodeStreamTriangleAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages, StructuredBuffer<uint2> pageTable, uint pageWordCount,
    out float3 a, out float3 b, out float3 c, out StreamTriangleDecodeContext context)
{
    a = b = c = float3(0);
    context = (StreamTriangleDecodeContext)0;
    uint count, stride;
    pageTable.GetDimensions(count, stride);
    if (pageIndex >= count) { return false; }
    uint entry = pageTable[pageIndex].x;
    if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
    uint page = (entry & 0xfffffff8u) / 4u;
    count = pageWordCount;
    if (page > count || count - page < 28u) { return false; }
    uint bytes = pages[page + 12];
    uint flags = pages[page + 14];
    uint positionFormat = pages[page + 20];
    uint positionStrideWords = positionFormat == 4u ? 3u : 4u;
    if (bytes < 112u || bytes / 4u > count - page || (positionFormat != 2u && positionFormat != 4u) ||
        (flags & 1u) == 0u || clusterIndex >= pages[page + 2]) { return false; }
    uint clusters = pages[page + 9], positions = pages[page + 10], triangles = pages[page + 11];
    if (((clusters | positions) & 3u) != 0u || clusters > bytes || clusterIndex >= (bytes - clusters) / 96u) { return false; }
    uint cluster = page + clusters / 4u + clusterIndex * 24u;
    uint vertexOffset = pages[cluster], vertexCount = pages[cluster + 1];
    uint triangleOffset = pages[cluster + 2], triangleCount = pages[cluster + 3];
    if (triangle >= triangleCount || triangles > bytes || triangleOffset > bytes - triangles ||
        triangle >= (bytes - triangles - triangleOffset) / 3u || positions > bytes) { return false; }
    uint availablePositionBytes = bytes - positions;
    uint maxVertices = positionFormat == 4u ? availablePositionBytes / 12u : availablePositionBytes >> 4u;
    if (vertexOffset > maxVertices || vertexCount > maxVertices - vertexOffset) { return false; }
    uint address = page * 4u + triangles + triangleOffset + triangle * 3u;
    // Three validated bytes occupy one word at offsets 0/1 and two at 2/3.
    uint byteOffset = address & 3u;
    uint word = address >> 2u;
    uint packedIndices = pages[word] >> (byteOffset * 8u);
    if (byteOffset > 1u) {
        packedIndices |= pages[word + 1u] << ((4u - byteOffset) * 8u);
    }
    uint3 local = uint3(packedIndices, packedIndices >> 8u, packedIndices >> 16u) & 255u;
    if (any(local >= vertexCount)) { return false; }
    uint base = page + positions / 4u + vertexOffset * positionStrideWords;
    a = streamPagePosition(pages, base + local.x * positionStrideWords);
    b = streamPagePosition(pages, base + local.y * positionStrideWords);
    c = streamPagePosition(pages, base + local.z * positionStrideWords);
    context.page = page;
    context.bytes = bytes;
    context.flags = flags;
    context.requiredVertices = vertexOffset + vertexCount;
    context.vertices = vertexOffset + local;
    return true;
}

public bool decodeStreamTriangleAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages, StructuredBuffer<uint2> pageTable, uint pageWordCount,
    out float3 a, out float3 b, out float3 c)
{
    StreamTriangleDecodeContext context;
    return decodeStreamTriangleAtPage(pageIndex, clusterIndex, triangle, pages, pageTable,
        pageWordCount, a, b, c, context);
}

// Committed CLAS hits reference validated page/cluster/triangle ranges. The
// streaming frame retains their geometry until GPU completion. Keep local-index
// and position-capacity checks, and let optional attributes validate themselves.
public bool decodeStreamCommittedTriangleAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages, StructuredBuffer<uint2> pageTable,
    out float3 a, out float3 b, out float3 c, out StreamTriangleDecodeContext context)
{
    a = b = c = float3(0);
    context = (StreamTriangleDecodeContext)0;
    uint entry = pageTable[pageIndex].x;
    if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
    uint page = (entry & 0xfffffff8u) / 4u;
    uint bytes = pages[page + 12u], flags = pages[page + 14u];
    uint positionFormat = pages[page + 20u];
    uint positionStrideWords = positionFormat == 4u ? 3u : 4u;
    uint clusters = pages[page + 9u], positions = pages[page + 10u], triangles = pages[page + 11u];
    uint cluster = page + clusters / 4u + clusterIndex * 24u;
    uint vertexOffset = pages[cluster], vertexCount = pages[cluster + 1u];
    uint triangleOffset = pages[cluster + 2u];
    // Published CLAS topology guarantees the index range independently of the
    // position stream. Start this fetch before checking vertex capacity.
    uint address = page * 4u + triangles + triangleOffset + triangle * 3u;
    uint byteOffset = address & 3u;
    uint word = address >> 2u;
    uint packedIndices = pages[word] >> (byteOffset * 8u);
    uint availablePositionBytes = bytes - positions;
    uint maxVertices = select(positionFormat == 4u, availablePositionBytes / 12u, availablePositionBytes >> 4u);
    bool validPositions = (positions <= bytes) & (vertexOffset <= maxVertices) &
        (vertexCount <= maxVertices - vertexOffset);
    if (byteOffset > 1u) {
        packedIndices |= pages[word + 1u] << ((4u - byteOffset) * 8u);
    }
    uint3 local = uint3(packedIndices, packedIndices >> 8u, packedIndices >> 16u) & 255u;
    if (!(validPositions & all(local < vertexCount))) { return false; }
    uint base = page + positions / 4u + vertexOffset * positionStrideWords;
    a = streamPagePosition(pages, base + local.x * positionStrideWords);
    b = streamPagePosition(pages, base + local.y * positionStrideWords);
    c = streamPagePosition(pages, base + local.z * positionStrideWords);
    context.page = page;
    context.bytes = bytes;
    context.flags = flags;
    context.requiredVertices = vertexOffset + vertexCount;
    context.vertices = vertexOffset + local;
    return true;
}

public bool decodeStreamTriangle(uint index, uint triangle, StructuredBuffer<CompactStreamVisibleRecord> records,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, uint pageWordCount, out VisibleClusterRecord record,
    out float3 a, out float3 b, out float3 c)
{
    record = (VisibleClusterRecord)0; a = b = c = float3(0);
    uint recordCount, recordStride;
    records.GetDimensions(recordCount, recordStride);
    if (index >= recordCount) { return false; }
    record = unpackStreamVisibleRecord(records[index]);
    uint count, stride;
    groups.GetDimensions(count, stride);
    // Cleared packed records decode to dataIndex == 0xffffffffu.
    if (record.dataIndex >= count) { return false; }
    StreamActiveGroup group = groups[record.dataIndex];
    record.instanceIndex = group.gpuSceneInstanceIndex;
    if (record.clusterIndex >= group.clusterCount ||
        record.instanceIndex == 0xffffffffu) { return false; }
    return decodeStreamTriangleAtPage(group.pageIndex, record.clusterIndex, triangle, pages, pageTable, pageWordCount, a, b, c);
}

bool streamHasTexcoords(StructuredBuffer<uint> pages, uint page, uint cluster, out uint uvOffset)
{
    uint bytes = pages[page + 12u];
    uvOffset = pages[page + 17u];
    uint required = pages[cluster] + pages[cluster + 1u];
    return (pages[page + 14u] & 4u) != 0u && pages[page + 22u] == 1u && uvOffset % 4u == 0u &&
        uvOffset <= bytes && required <= (bytes - uvOffset) / 8u;
}

bool streamHasTexcoords(StructuredBuffer<uint> pages, uint page, uint bytes, uint flags,
    uint required, out uint uvOffset)
{
    uvOffset = pages[page + 17u];
    return (flags & 4u) != 0u && pages[page + 22u] == 1u && uvOffset % 4u == 0u &&
        uvOffset <= bytes && required <= (bytes - uvOffset) / 8u;
}

float2 streamLoadTexcoord(StructuredBuffer<uint> pages, uint page, uint uvOffset, uint vertex)
{
    uint uv = page + uvOffset / 4u + vertex * 2u;
    return asfloat(uint2(pages[uv], pages[uv + 1u]));
}

// The triangle comes from a published CLAS hit (validated topology and geometry
// retained through GPU completion). Optional UV ranges still need their own check.
public void decodeStreamTexcoordsAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages, StructuredBuffer<uint2> pageTable, out float2 uvs[3])
{
    for (uint i = 0; i < 3; ++i) { uvs[i] = float2(0); }
    uint page = (pageTable[pageIndex].x & 0xfffffff8u) / 4u;
    uint cluster = page + pages[page + 9u] / 4u + clusterIndex * 24u;
    uint uvOffset;
    if (!streamHasTexcoords(pages, page, cluster, uvOffset)) { return; }
    uint address = page * 4u + pages[page + 11u] + pages[cluster + 2u] + triangle * 3u;
    for (uint i = 0; i < 3; ++i) {
        uint vertex = pages[cluster] + streamPageByte(pages, address + i);
        uvs[i] = streamLoadTexcoord(pages, page, uvOffset, vertex);
    }
}

// Optional shading streams. Position-only pages continue to use geometric
// normals and zero UV; every optional range is validated before dereferencing.
public uint decodeStreamAttributes(StructuredBuffer<uint> pages, StreamTriangleDecodeContext context,
    out float3 normals[3], out float2 uvs[3], out float4 tangents[3], bool decodeShading = true)
{
    normals[0] = normals[1] = normals[2] = float3(0);
    uvs[0] = uvs[1] = uvs[2] = float2(0);
    tangents[0] = tangents[1] = tangents[2] = float4(0);
    uint page = context.page, bytes = context.bytes, flags = context.flags;
    uint required = context.requiredVertices;
    uint uvOffset;
    bool hasUv = streamHasTexcoords(pages, page, bytes, flags, required, uvOffset);
    if (hasUv) {
        uvs[0] = streamLoadTexcoord(pages, page, uvOffset, context.vertices.x);
        uvs[1] = streamLoadTexcoord(pages, page, uvOffset, context.vertices.y);
        uvs[2] = streamLoadTexcoord(pages, page, uvOffset, context.vertices.z);
    }
    uint attributes = hasUv ? 4u : 0u;
    // Shadow transport needs only UVs. Keep shading-stream metadata and loads
    // inside this specialization so they disappear from the shadow shader.
    if (decodeShading) {
        uint normalOffset = pages[page + 16u];
        uint normalFormat = pages[page + 21u], normalStride = streamNormalStride(normalFormat);
        uint normalShift = normalFormat == 2u ? 4u : 2u;
        bool hasNormal = (flags & 2u) != 0u && normalStride != 0u && normalOffset % 4u == 0u &&
            normalOffset <= bytes && required <= ((bytes - normalOffset) >> normalShift);
        if (hasNormal) {
            uint base = page + normalOffset / 4u;
            if (normalFormat == 5u) {
                // Issue all three independent loads before consuming their encodings.
                uint p0 = pages[base + context.vertices.x];
                uint p1 = pages[base + context.vertices.y];
                uint p2 = pages[base + context.vertices.z];
                normals[0] = decodeSceneNormal(p0);
                normals[1] = decodeSceneNormal(p1);
                normals[2] = decodeSceneNormal(p2);
            } else {
                normals[0] = streamPagePosition(pages, base + context.vertices.x * 4u);
                normals[1] = streamPagePosition(pages, base + context.vertices.y * 4u);
                normals[2] = streamPagePosition(pages, base + context.vertices.z * 4u);
            }
            attributes |= 2u;
        }
        uint tangentOffset = pages[page + 24u];
        uint tangentFormat = pages[page + 25u], tangentStride = streamTangentStride(tangentFormat);
        uint tangentShift = tangentFormat == 2u ? 4u : 2u;
        bool hasTangent = (flags & 16u) != 0u && tangentStride != 0u && tangentOffset % 4u == 0u &&
            tangentOffset <= bytes && required <= ((bytes - tangentOffset) >> tangentShift);
        if (hasTangent) {
            uint base = page + tangentOffset / 4u;
            if (tangentFormat == 6u) {
                uint p0 = pages[base + context.vertices.x];
                uint p1 = pages[base + context.vertices.y];
                uint p2 = pages[base + context.vertices.z];
                tangents[0] = decodeSceneTangent(p0);
                tangents[1] = decodeSceneTangent(p1);
                tangents[2] = decodeSceneTangent(p2);
            } else {
                tangents[0] = streamLoadTangent(pages, page, tangentOffset, context.vertices.x, 2u);
                tangents[1] = streamLoadTangent(pages, page, tangentOffset, context.vertices.y, 2u);
                tangents[2] = streamLoadTangent(pages, page, tangentOffset, context.vertices.z, 2u);
            }
            attributes |= 16u;
        }
    }
    return attributes;
}

public uint decodeStreamAttributesAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3], out float4 tangents[3])
{
    StreamTriangleDecodeContext context;
    context.page = (pageTable[pageIndex].x & 0xfffffff8u) / 4u;
    context.bytes = pages[context.page + 12u];
    context.flags = pages[context.page + 14u];
    uint cluster = context.page + pages[context.page + 9u] / 4u + clusterIndex * 24u;
    uint address = context.page * 4u + pages[context.page + 11u] + pages[cluster + 2u] + triangle * 3u;
    uint vertexOffset = pages[cluster];
    context.requiredVertices = vertexOffset + pages[cluster + 1u];
    context.vertices = vertexOffset + uint3(streamPageByte(pages, address),
        streamPageByte(pages, address + 1u), streamPageByte(pages, address + 2u));
    return decodeStreamAttributes(pages, context, normals, uvs, tangents);
}

public uint decodeStreamAttributes(VisibleClusterRecord record, uint triangle,
    StructuredBuffer<StreamActiveGroup> groups, StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3], out float4 tangents[3])
{
    return decodeStreamAttributesAtPage(groups[record.dataIndex].pageIndex, record.clusterIndex, triangle,
        pages, pageTable, normals, uvs, tangents);
}
} // namespace Metallic.GPUDriven
