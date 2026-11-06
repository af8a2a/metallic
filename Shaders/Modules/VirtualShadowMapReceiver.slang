#language slang 2026
module VirtualShadowMapReceiver;

import ShaderCore;
import GPUDriven;

using Metallic;
using Metallic.GPUDriven;

namespace Metallic.Shadow
{

// Published with the same view/frame/scene identity as the primary visibility.
// Tessellated primary surfaces use a different representation and do not bind
// this table. The shadow caster LOD cut is independent of these resources.
struct ReceiverGeometry
{
    uint visibility, records, groups, pages;
    uint pageTable, params, recordCapacity, flags;
};

public struct VirtualShadowMapReceiverTriangle
{
    public float3 a, b, c;
    public float3 normal;
    public uint instanceIndex;
};

// Resolved once per receiver correction. Resource dimensions and the retained
// stream pool capacity remain fixed throughout its near/far candidate queries.
public struct VirtualShadowMapReceiverContext
{
    public Texture2D<uint> visibility;
    public StructuredBuffer<CompactStreamVisibleRecord> records;
    public StructuredBuffer<StreamActiveGroup> groups;
    public StructuredBuffer<uint> pages;
    public StructuredBuffer<uint2> pageTable;
    public uint recordCount, groupCount, pageTableCount, pageWordCount;
};

public VirtualShadowMapReceiverContext vsmResolveReceiverContext(uint geometry)
{
    StructuredBuffer<ReceiverGeometry> tables = resolveUniform(ResourceHandle<StructuredBuffer<ReceiverGeometry>>(geometry));
    ReceiverGeometry table = tables[0];
    VirtualShadowMapReceiverContext context;
    context.visibility = resolveUniform(ResourceHandle<Texture2D<uint>>(table.visibility));
    context.records = resolveUniform(ResourceHandle<StructuredBuffer<CompactStreamVisibleRecord>>(table.records));
    context.groups = resolveUniform(ResourceHandle<StructuredBuffer<StreamActiveGroup>>(table.groups));
    context.pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(table.pages));
    context.pageTable = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(table.pageTable));
    ByteAddressBuffer parameters = resolveUniform(ResourceHandle<ByteAddressBuffer>(table.params));
    uint stride;
    context.records.GetDimensions(context.recordCount, stride);
    context.recordCount = min(context.recordCount, table.recordCapacity);
    context.groups.GetDimensions(context.groupCount, stride);
    context.pageTable.GetDimensions(context.pageTableCount, stride);
    context.pageWordCount = streamPageWordCount(parameters);
    return context;
}

struct ReceiverCluster
{
    StreamActiveGroup group;
    uint positionBase, positionStrideWords, firstIndexByte, vertexCount, triangleCount;
    float normalSign;
};

bool vsmDecodeReceiverCluster(CompactStreamVisibleRecord packed, uint visibleTriangle,
    VirtualShadowMapReceiverContext context, out ReceiverCluster decoded)
{
    decoded = (ReceiverCluster)0;
    VisibleClusterRecord record = unpackStreamVisibleRecord(packed);
    if (record.dataIndex >= context.groupCount) { return false; }
    StreamActiveGroup group = context.groups[record.dataIndex];
    if (record.clusterIndex >= group.clusterCount || group.gpuSceneInstanceIndex == 0xffffffffu) { return false; }

    if (group.pageIndex >= context.pageTableCount) { return false; }
    uint entry = context.pageTable[group.pageIndex].x;
    if ((entry & 7u) != 2u && (entry & 7u) != 3u) { return false; }
    uint page = (entry & 0xfffffff8u) / 4u;
    if (page > context.pageWordCount || context.pageWordCount - page < 28u) { return false; }
    StructuredBuffer<uint> pages = context.pages;
    uint bytes = pages[page + 12u], flags = pages[page + 14u], positionFormat = pages[page + 20u];
    uint clusterCount = pages[page + 2u], payloadVertexCount = pages[page + 3u], indexBytes = pages[page + 4u];
    if (bytes < 112u || bytes / 4u > context.pageWordCount - page || (flags & 1u) == 0u ||
        (positionFormat != 2u && positionFormat != 4u) || record.clusterIndex >= clusterCount) { return false; }
    uint clusters = pages[page + 9u], positions = pages[page + 10u], triangles = pages[page + 11u];
    uint positionStrideWords = positionFormat == 4u ? 3u : 4u;
    if (((clusters | positions) & 3u) != 0u || clusters > bytes ||
        clusterCount > (bytes - clusters) / 96u || positions > bytes ||
        payloadVertexCount > (bytes - positions) / (positionStrideWords * 4u) ||
        triangles > bytes || indexBytes > bytes - triangles) { return false; }
    uint cluster = page + clusters / 4u + record.clusterIndex * 24u;
    uint vertexOffset = pages[cluster], vertexCount = pages[cluster + 1u];
    uint triangleOffset = pages[cluster + 2u], triangleCount = pages[cluster + 3u];
    if (vertexCount == 0u || vertexCount > 128u || vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset || triangleCount == 0u || triangleCount > 128u ||
        visibleTriangle >= triangleCount || triangleOffset > indexBytes ||
        triangleCount > (indexBytes - triangleOffset) / 3u) { return false; }

    decoded.group = group;
    decoded.positionBase = page + positions / 4u + vertexOffset * positionStrideWords;
    decoded.positionStrideWords = positionStrideWords;
    decoded.firstIndexByte = page * 4u + triangles + triangleOffset;
    decoded.vertexCount = vertexCount;
    decoded.triangleCount = triangleCount;
    decoded.normalSign = dot(group.world0.xyz, cross(group.world1.xyz, group.world2.xyz)) < 0.0 ? -1.0 : 1.0;
    return true;
}

bool vsmDecodeReceiverTriangle(VirtualShadowMapReceiverContext context, ReceiverCluster cluster,
    uint triangleIndex, out VirtualShadowMapReceiverTriangle triangle)
{
    triangle = (VirtualShadowMapReceiverTriangle)0;
    uint address = cluster.firstIndexByte + triangleIndex * 3u;
    uint byteOffset = address & 3u, word = address >> 2u;
    uint packedIndices = context.pages[word] >> (byteOffset * 8u);
    if (byteOffset > 1u) { packedIndices |= context.pages[word + 1u] << ((4u - byteOffset) * 8u); }
    uint3 local = uint3(packedIndices, packedIndices >> 8u, packedIndices >> 16u) & 255u;
    if (any(local >= cluster.vertexCount)) { return false; }
    float3 a = streamPagePosition(context.pages, cluster.positionBase + local.x * cluster.positionStrideWords);
    float3 b = streamPagePosition(context.pages, cluster.positionBase + local.y * cluster.positionStrideWords);
    float3 c = streamPagePosition(context.pages, cluster.positionBase + local.z * cluster.positionStrideWords);
    StreamActiveGroup group = cluster.group;
    float3 e1 = b - a, e2 = c - a;
    e1 = group.world0.xyz * e1.x + group.world1.xyz * e1.y + group.world2.xyz * e1.z;
    e2 = group.world0.xyz * e2.x + group.world1.xyz * e2.y + group.world2.xyz * e2.z;
    float3 normal = cross(e1, e2);
    float lengthSquared = dot(normal, normal);
    // A structurally valid degenerate triangle has no ray intersection. The
    // zero normal also prevents it from being used as a receiver plane.
    if (lengthSquared <= 1e-20) { return true; }
    triangle.a = group.world0.xyz * a.x + group.world1.xyz * a.y + group.world2.xyz * a.z + group.world3.xyz;
    triangle.b = triangle.a + e1;
    triangle.c = triangle.a + e2;
    triangle.normal = normal * (cluster.normalSign * rsqrt(lengthSquared));
    triangle.instanceIndex = group.gpuSceneInstanceIndex;
    return true;
}

public bool vsmPrimaryReceiverTriangle(uint2 pixel, VirtualShadowMapReceiverContext context,
    out VirtualShadowMapReceiverTriangle triangle)
{
    triangle = (VirtualShadowMapReceiverTriangle)0;
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(context.visibility.Load(int3(pixel, 0)), recordIndex, triangleIndex) ||
        recordIndex >= context.recordCount) { return false; }
    ReceiverCluster cluster;
    if (!vsmDecodeReceiverCluster(context.records[recordIndex], triangleIndex, context, cluster) ||
        !vsmDecodeReceiverTriangle(context, cluster, triangleIndex, triangle)) { return false; }
    return dot(triangle.normal, triangle.normal) > 0.0;
}

bool vsmPrimaryTriangleBlocksRay(VirtualShadowMapReceiverTriangle triangle,
    float3 origin, float3 direction)
{
    float denominator = dot(triangle.normal, direction);
    if (abs(denominator) <= 1e-6) { return false; }
    float distance = dot(triangle.normal, triangle.a - origin) / denominator;
    // Reject only the numerical intersection with the receiver itself. This
    // epsilon is far smaller than the shadow comparison's world-space bias.
    if (distance <= 1e-5) { return false; }
    float3 e1 = triangle.b - triangle.a, e2 = triangle.c - triangle.a;
    float3 area = cross(e1, e2);
    float3 offset = origin + direction * distance - triangle.a;
    float2 uv = float2(dot(cross(offset, e2), area), dot(cross(e1, offset), area)) / dot(area, area);
    return all(uv >= -1e-5) && uv.x + uv.y <= 1.00001;
}

public bool vsmPrimaryClusterBlocksRay(uint2 pixel, VirtualShadowMapReceiverContext context,
    float3 origin, float3 direction, inout uint4 recent0, inout uint4 recent1)
{
    uint recordIndex, visibleTriangle;
    if (!unpackVisibilityId(context.visibility.Load(int3(pixel, 0)), recordIndex, visibleTriangle)) { return false; }
    // Background contributes no candidate. A visible ID whose geometry cannot
    // be decoded must preserve the original shadow instead of clearing it.
    if (recordIndex >= context.recordCount) { return true; }
    CompactStreamVisibleRecord packed = context.records[recordIndex];
    uint key = packed.packed & 0x3fffffffu;
    if (any(recent0 == key) || any(recent1 == key)) { return false; }
    ReceiverCluster cluster;
    if (!vsmDecodeReceiverCluster(packed, visibleTriangle, context, cluster)) { return true; }

    // Eight recent group/cluster keys include the instance through the active
    // group index. Eviction only causes a repeated query, never a lost candidate.
    recent1 = uint4(recent0.w, recent1.xyz);
    recent0 = uint4(key, recent0.xyz);
    for (uint triangleIndex = 0u; triangleIndex < cluster.triangleCount; ++triangleIndex) {
        VirtualShadowMapReceiverTriangle triangle;
        if (!vsmDecodeReceiverTriangle(context, cluster, triangleIndex, triangle)) { return true; }
        // Test the complete positive light ray. The receiver offset limit does
        // not bound the distance of a real occluder in this candidate cluster.
        if (vsmPrimaryTriangleBlocksRay(triangle, origin, direction)) { return true; }
    }
    return false;
}

public bool vsmPrimaryTriangleBlocksRayAtPixel(uint2 pixel, VirtualShadowMapReceiverContext context,
    float3 origin, float3 direction, uint4 recent0, uint4 recent1, inout uint previousVisibility)
{
    uint visibility = context.visibility.Load(int3(pixel, 0));
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(visibility, recordIndex, triangleIndex)) {
        previousVisibility = 0u;
        return false;
    }
    // Initialize this cache to the reserved background ID, not UINT_MAX, which
    // is a valid encoded record/triangle. Only completed misses enter the cache.
    if (visibility == previousVisibility) { return false; }
    if (recordIndex >= context.recordCount) { return true; }
    CompactStreamVisibleRecord packed = context.records[recordIndex];
    uint key = packed.packed & 0x3fffffffu;
    // The near queries already tested every triangle of these clusters against
    // the same ray. Reaching the far traversal means all of them missed.
    if (any(recent0 == key) || any(recent1 == key)) {
        previousVisibility = visibility;
        return false;
    }
    ReceiverCluster cluster;
    VirtualShadowMapReceiverTriangle triangle;
    if (!vsmDecodeReceiverCluster(packed, triangleIndex, context, cluster) ||
        !vsmDecodeReceiverTriangle(context, cluster, triangleIndex, triangle)) { return true; }
    if (vsmPrimaryTriangleBlocksRay(triangle, origin, direction)) { return true; }
    previousVisibility = visibility;
    return false;
}

} // namespace Metallic.Shadow
