#language slang 2026
module StreamTriangleDecodeProbe;
import Core;
import GPUDriven;
import StreamRayGeometry;
using Metallic;
using Metallic.GPUDriven;
using Metallic.StreamGeometry;

// Frozen pre-context implementation; never call the compatibility wrappers here.
uint referenceStreamPageByte(StructuredBuffer<uint> pages, uint address)
{
    return (pages[address / 4] >> ((address & 3u) * 8u)) & 255u;
}

float3 referenceStreamPagePosition(StructuredBuffer<uint> pages, uint word)
{
    return asfloat(uint3(pages[word], pages[word + 1], pages[word + 2]));
}

bool referenceDecodeStreamTriangleAtPage(uint pageIndex, uint clusterIndex, uint triangle,
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
    uint3 local = uint3(referenceStreamPageByte(pages, address), referenceStreamPageByte(pages, address + 1), referenceStreamPageByte(pages, address + 2));
    if (any(local >= vertexCount)) { return false; }
    uint base = page + positions / 4u + vertexOffset * positionStrideWords;
    a = referenceStreamPagePosition(pages, base + local.x * positionStrideWords);
    b = referenceStreamPagePosition(pages, base + local.y * positionStrideWords);
    c = referenceStreamPagePosition(pages, base + local.z * positionStrideWords);
    return true;
}

bool referenceStreamHasTexcoords(StructuredBuffer<uint> pages, uint page, uint cluster, out uint uvOffset)
{
    uint bytes = pages[page + 12u];
    uvOffset = pages[page + 17u];
    uint required = pages[cluster] + pages[cluster + 1u];
    return (pages[page + 14u] & 4u) != 0u && pages[page + 22u] == 1u && uvOffset % 4u == 0u &&
        uvOffset <= bytes && required <= (bytes - uvOffset) / 8u;
}

float2 referenceStreamLoadTexcoord(StructuredBuffer<uint> pages, uint page, uint uvOffset, uint vertex)
{
    uint uv = page + uvOffset / 4u + vertex * 2u;
    return asfloat(uint2(pages[uv], pages[uv + 1u]));
}

uint referenceDecodeStreamAttributesAtPage(uint pageIndex, uint clusterIndex, uint triangle,
    StructuredBuffer<uint> pages,
    StructuredBuffer<uint2> pageTable, out float3 normals[3], out float2 uvs[3], out float4 tangents[3])
{
    for (uint i = 0; i < 3; ++i) { normals[i] = float3(0); uvs[i] = float2(0); tangents[i] = float4(0); }
    uint page = (pageTable[pageIndex].x & 0xfffffff8u) / 4u;
    uint bytes = pages[page + 12u], flags = pages[page + 14u];
    uint cluster = page + pages[page + 9u] / 4u + clusterIndex * 24u;
    uint address = page * 4u + pages[page + 11u] + pages[cluster + 2u] + triangle * 3u;
    uint normalOffset = pages[page + 16u], uvOffset;
    uint required = pages[cluster] + pages[cluster + 1u];
    uint normalFormat = pages[page + 21u], normalStride = streamNormalStride(normalFormat);
    bool hasNormal = (flags & 2u) != 0u && normalStride != 0u && normalOffset % 4u == 0u &&
        normalOffset <= bytes && required <= (bytes - normalOffset) / max(normalStride, 1u);
    bool hasUv = referenceStreamHasTexcoords(pages, page, cluster, uvOffset);
    uint tangentOffset = pages[page + 24u];
    uint tangentFormat = pages[page + 25u], tangentStride = streamTangentStride(tangentFormat);
    bool hasTangent = (flags & 16u) != 0u && tangentStride != 0u && tangentOffset % 4u == 0u &&
        tangentOffset <= bytes && required <= (bytes - tangentOffset) / max(tangentStride, 1u);
    for (uint i = 0; i < 3; ++i) {
        uint v = pages[cluster] + referenceStreamPageByte(pages, address + i);
        if (hasNormal) {
            normals[i] = streamLoadNormal(pages,page,normalOffset,v,normalFormat);
        }
        if (hasUv) {
            uvs[i] = referenceStreamLoadTexcoord(pages, page, uvOffset, v);
        }
        if (hasTangent) {
            tangents[i] = streamLoadTangent(pages,page,tangentOffset,v,tangentFormat);
        }
    }
    return (hasNormal ? 2u : 0u) | (hasUv ? 4u : 0u) | (hasTangent ? 16u : 0u);
}

struct ProbePush { uint count; uint pageWords; };

[shader("compute")]
[numthreads(64, 1, 1)]
void streamTriangleDecodeProbeMain(uint3 id : SV_DispatchThreadID)
{
    ProbePush push = getConstants<ProbePush>();
    if (id.x >= push.count) { return; }
    SceneResourceParameters resources = getResourceParameters<SceneResourceParameters>();
    StructuredBuffer<uint> pages = resolveBuffer<StructuredBuffer<uint>>(resources.rayStreamPages);
    StructuredBuffer<uint2> table = resolveBuffer<StructuredBuffer<uint2>>(resources.rayStreamPageTable);
    StructuredBuffer<uint4> cases = resolveBuffer<StructuredBuffer<uint4>>(resources.rayStreamHeader);
    RWStructuredBuffer<uint4> output = resolveBuffer<RWStructuredBuffer<uint4>>(resources.probeOutput);
    uint4 test = cases[id.x];
    if (test.w == 0xffffffffu) {
        uint2 genericAddress = streamRayClusterAddress(test.x, test.y);
        uint2 defaultAddress = streamRayClusterAddress(test.x);
        uint2 reference = uint2(test.x / test.y, test.x % test.y);
        uint2 referenceFixed = uint2(test.x / 32u, test.x % 32u);
        uint failures = any(genericAddress != reference) ? 1u : 0u;
        failures |= any(defaultAddress != referenceFixed) ? 2u : 0u;
        failures |= genericAddress.x != test.z ? 4u : 0u;
        output[id.x] = uint4(failures, genericAddress, defaultAddress.x);
        return;
    }
    float3 ra, rb, rc, a, b, c;
    bool referenceValid = referenceDecodeStreamTriangleAtPage(test.x, test.y, test.z, pages, table, push.pageWords, ra, rb, rc);
    StreamTriangleDecodeContext context;
    bool valid = decodeStreamTriangleAtPage(test.x, test.y, test.z, pages, table, push.pageWords, a, b, c, context);
    uint failures = valid != referenceValid ? 1u : 0u;
    failures |= valid != ((test.w & 1u) != 0u) ? 2u : 0u;
    failures |= !all(asuint(a) == asuint(ra)) || !all(asuint(b) == asuint(rb)) || !all(asuint(c) == asuint(rc)) ? 4u : 0u;
    float3 wa, wb, wc;
    bool wrapperValid = decodeStreamTriangleAtPage(test.x, test.y, test.z, pages, table, push.pageWords, wa, wb, wc);
    failures |= wrapperValid != referenceValid || !all(asuint(wa) == asuint(ra)) ||
        !all(asuint(wb) == asuint(rb)) || !all(asuint(wc) == asuint(rc)) ? 1024u : 0u;
    uint flags = 0u;
    if (valid && referenceValid) {
        float3 rn[3], n[3]; float2 ruv[3], uv[3]; float4 rt[3], t[3];
        uint referenceFlags = referenceDecodeStreamAttributesAtPage(test.x, test.y, test.z, pages, table, rn, ruv, rt);
        flags = decodeStreamAttributes(pages, context, n, uv, t);
        float3 wn[3]; float2 wuv[3]; float4 wt[3];
        uint wrapperFlags = decodeStreamAttributesAtPage(test.x, test.y, test.z, pages, table, wn, wuv, wt);
        failures |= wrapperFlags != referenceFlags ? 2048u : 0u;
        failures |= flags != referenceFlags || flags != (test.w >> 1u) ? 8u : 0u;
        for (uint i = 0; i < 3; ++i) {
            failures |= !all(asuint(wn[i]) == asuint(rn[i])) || !all(asuint(wuv[i]) == asuint(ruv[i])) ||
                !all(asuint(wt[i]) == asuint(rt[i])) ? 2048u : 0u;
            failures |= !all(asuint(n[i]) == asuint(rn[i])) ? 16u : 0u;
            failures |= !all(asuint(uv[i]) == asuint(ruv[i])) ? 32u : 0u;
            failures |= !all(asuint(t[i]) == asuint(rt[i])) ? 64u : 0u;
        }
        // Covers packed three-byte extraction independently of position values.
        uint page = (table[test.x].x & 0xfffffff8u) / 4u;
        uint cluster = page + pages[page + 9u] / 4u + test.y * 24u;
        uint address = page * 4u + pages[page + 11u] + pages[cluster + 2u] + test.z * 3u;
        uint3 referenceIndices = pages[cluster] + uint3(referenceStreamPageByte(pages, address),
            referenceStreamPageByte(pages, address + 1u), referenceStreamPageByte(pages, address + 2u));
        failures |= !all(context.vertices == referenceIndices) ? 128u : 0u;
        failures |= context.page != page || context.bytes != pages[page + 12u] ||
            context.flags != pages[page + 14u] || context.requiredVertices != pages[cluster] + pages[cluster + 1u] ? 256u : 0u;
    } else if (!valid) {
        failures |= context.page != 0u || context.bytes != 0u || context.flags != 0u ||
            context.requiredVertices != 0u || any(context.vertices != uint3(0)) ? 512u : 0u;
    }
    output[id.x] = uint4(failures, valid ? 1u : 0u, flags, test.w);
}
