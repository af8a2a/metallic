#language slang 2026
module VirtualShadowMapPrototype;

import ShaderCore;
import Material;
import MaterialProgram;

using Metallic;
using Metallic.Material;

struct VSMPrototypePush
{
    uint triangles;
    uint materials;
    uint pages;
    uint coverageInputs;
    uint workItems;
    uint arguments;
    uint projection;
    uint atlas;
    uint receivers;
    uint queryResults;
    uint triangleCount;
    uint receiverCount;
    uint virtualPagesX;
    uint virtualPagesY;
    uint physicalPagesX;
    uint physicalPagesY;
    uint pageSize;
    uint workCapacity;
    uint sampler;
    uint reserved; // forceAllPagesDirty when the complete atlas is cleared.
};

struct VSMPrototypeVertex
{
    float4 position;
    float4 uv;
};

struct VSMPrototypeTriangle
{
    VSMPrototypeVertex vertices[3];
    uint materialIndex;
    // Scalar padding preserves the CPU 112-byte stride under std430.
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct VSMPrototypeMaterial
{
    float4 alpha;
    uint4 resources;
};

struct VSMPrototypePage
{
    uint physicalPage;
    uint dirty;
    uint2 reserved;
};

struct VSMPrototypeQuery
{
    float visibility;
    float depth;
    uint virtualPage;
    uint status;
    uint physicalPage;
    // A uint3 here would align to byte 32 and make the stride 48 instead of 32.
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct VSMPrototypeClipVertex
{
    float3 position;
    float2 uv;
};

struct VSMPrototypeRasterVertex
{
    float4 position : SV_Position;
    [vk::location(0)] float2 uv : TEXCOORD0;
    [vk::location(1)] nointerpolation uint materialIndex : TEXCOORD1;
};

float3 vsmPrototypeProject(float4 position, VSMPrototypePush push)
{
    StructuredBuffer<float4> projection = resolveUniform(ResourceHandle<StructuredBuffer<float4>>(push.projection));
    // The validated P0 projection is orthographic with homogeneous w = 1.
    return float3(dot(projection[0], position), dot(projection[1], position), dot(projection[2], position));
}

float4 vsmPrototypePageBounds(uint virtualPage, VSMPrototypePush push)
{
    uint2 page = uint2(virtualPage % push.virtualPagesX, virtualPage / push.virtualPagesX);
    float2 dimensions = float2(push.virtualPagesX, push.virtualPagesY);
    return float4(float2(page) * (2.0 / dimensions) - 1.0,
        float2(page + 1u) * (2.0 / dimensions) - 1.0);
}

float2 vsmPrototypeAtlasPosition(float2 pagePosition, uint physicalPage, VSMPrototypePush push)
{
    uint2 physical = uint2(physicalPage % push.physicalPagesX, physicalPage / push.physicalPagesX);
    return (float2(physical) + pagePosition) * (2.0 / float2(push.physicalPagesX, push.physicalPagesY)) - 1.0;
}

[shader("compute")]
[numthreads(8, 1, 1)]
void vsmPrototypeResetMain(uint3 id : SV_DispatchThreadID, uniform VSMPrototypePush push)
{
    if (id.x >= 8u) { return; }
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    arguments[id.x] = id.x == 1u || id.x == 2u || id.x == 4u || id.x == 5u ? 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmPrototypeBuildMain(uint3 id : SV_DispatchThreadID, uniform VSMPrototypePush push)
{
    uint pageCount = push.virtualPagesX * push.virtualPagesY;
    StructuredBuffer<VSMPrototypePage> pages = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypePage>>(push.pages));
    RWStructuredBuffer<uint2> workItems = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));

    if (id.x < pageCount) {
        VSMPrototypePage page = pages[id.x];
        if (page.physicalPage != 0xffffffffu && (page.dirty != 0u || push.reserved != 0u)) {
            uint clearIndex;
            InterlockedAdd(arguments[3], 1u, clearIndex);
            workItems[push.workCapacity + clearIndex] = uint2(0u, id.x);
        }
    }

    if (id.x >= push.triangleCount) { return; }
    StructuredBuffer<VSMPrototypeTriangle> triangles = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypeTriangle>>(push.triangles));
    VSMPrototypeTriangle triangle = triangles[id.x];
    float3 a = vsmPrototypeProject(triangle.vertices[0].position, push);
    float3 b = vsmPrototypeProject(triangle.vertices[1].position, push);
    float3 c = vsmPrototypeProject(triangle.vertices[2].position, push);
    float3 boundsMin = min(a, min(b, c));
    float3 boundsMax = max(a, max(b, c));
    if (boundsMax.z < 0.0 || boundsMin.z > 1.0) { return; }

    for (uint pageIndex = 0u; pageIndex < pageCount; ++pageIndex) {
        VSMPrototypePage page = pages[pageIndex];
        if (page.physicalPage == 0xffffffffu || (page.dirty == 0u && push.reserved == 0u)) { continue; }
        float4 bounds = vsmPrototypePageBounds(pageIndex, push);
        if (any(boundsMax.xy < bounds.xy) || any(boundsMin.xy > bounds.zw)) { continue; }
        uint workIndex;
        InterlockedAdd(arguments[6], 1u, workIndex);
        if (workIndex < push.workCapacity) {
            workItems[workIndex] = uint2(id.x, pageIndex);
        } else {
            InterlockedAdd(arguments[7], 1u);
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void vsmPrototypeFinalizeMain(uniform VSMPrototypePush push)
{
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    arguments[0] = min(arguments[6], push.workCapacity);
}

// A triangle clipped against four half-planes has at most seven vertices.
uint vsmPrototypeClipPlane(inout VSMPrototypeClipVertex polygon[7], uint count, uint axis, float edge, float side)
{
    if (count == 0u) { return 0u; }
    VSMPrototypeClipVertex clipped[7];
    uint clippedCount = 0u;
    VSMPrototypeClipVertex previous = polygon[count - 1u];
    float previousDistance = side * (previous.position[axis] - edge);
    for (uint index = 0u; index < count; ++index) {
        VSMPrototypeClipVertex current = polygon[index];
        float currentDistance = side * (current.position[axis] - edge);
        // Strict crossings avoid duplicating vertices exactly on a page edge.
        if ((previousDistance < 0.0 && currentDistance > 0.0) ||
            (previousDistance > 0.0 && currentDistance < 0.0)) {
            float t = previousDistance / (previousDistance - currentDistance);
            VSMPrototypeClipVertex intersection;
            intersection.position = lerp(previous.position, current.position, t);
            intersection.position[axis] = edge;
            intersection.uv = lerp(previous.uv, current.uv, t);
            clipped[clippedCount++] = intersection;
        }
        if (currentDistance >= 0.0) { clipped[clippedCount++] = current; }
        previous = current;
        previousDistance = currentDistance;
    }
    for (uint index = 0u; index < clippedCount; ++index) { polygon[index] = clipped[index]; }
    return clippedCount;
}

groupshared VSMPrototypeClipVertex g_vsmPrototypePolygon[7];
groupshared uint g_vsmPrototypeVertexCount;
groupshared uint g_vsmPrototypeMaterialIndex;
groupshared uint g_vsmPrototypePhysicalPage;
groupshared float4 g_vsmPrototypePageBounds;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmPrototypeClearMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices VSMPrototypeRasterVertex vertices[4], out indices uint3 triangles[2],
    uniform VSMPrototypePush push)
{
    StructuredBuffer<uint2> workItems = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
    StructuredBuffer<VSMPrototypePage> pages = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypePage>>(push.pages));
    uint virtualPage = workItems[push.workCapacity + group.x].y;
    uint physicalPage = pages[virtualPage].physicalPage;
    SetMeshOutputCounts(4u, 2u);
    if (lane < 4u) {
        float2 corner = float2(lane & 1u, lane >> 1u);
        vertices[lane].position = float4(vsmPrototypeAtlasPosition(corner, physicalPage, push), 0.0, 1.0);
        vertices[lane].uv = float2(0.0);
        vertices[lane].materialIndex = 0xffffffffu;
    }
    if (lane < 2u) { triangles[lane] = lane == 0u ? uint3(0u, 1u, 2u) : uint3(2u, 1u, 3u); }
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmPrototypeRasterMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices VSMPrototypeRasterVertex vertices[7], out indices uint3 triangles[5],
    uniform VSMPrototypePush push)
{
    if (lane == 0u) {
        StructuredBuffer<uint2> workItems = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
        StructuredBuffer<VSMPrototypeTriangle> source = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypeTriangle>>(push.triangles));
        StructuredBuffer<VSMPrototypePage> pages = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypePage>>(push.pages));
        uint2 work = workItems[group.x];
        VSMPrototypeTriangle triangle = source[work.x];
        float4 bounds = vsmPrototypePageBounds(work.y, push);
        VSMPrototypeClipVertex polygon[7];
        for (uint index = 0u; index < 3u; ++index) {
            polygon[index].position = vsmPrototypeProject(triangle.vertices[index].position, push);
            polygon[index].uv = triangle.vertices[index].uv.xy;
        }
        uint count = vsmPrototypeClipPlane(polygon, 3u, 0u, bounds.x, 1.0);
        count = vsmPrototypeClipPlane(polygon, count, 0u, bounds.z, -1.0);
        count = vsmPrototypeClipPlane(polygon, count, 1u, bounds.y, 1.0);
        count = vsmPrototypeClipPlane(polygon, count, 1u, bounds.w, -1.0);
        g_vsmPrototypeVertexCount = count >= 3u ? count : 0u;
        g_vsmPrototypeMaterialIndex = triangle.materialIndex;
        g_vsmPrototypePhysicalPage = pages[work.y].physicalPage;
        g_vsmPrototypePageBounds = bounds;
        for (uint index = 0u; index < count; ++index) { g_vsmPrototypePolygon[index] = polygon[index]; }
    }
    GroupMemoryBarrierWithGroupSync();
    uint vertexCount = g_vsmPrototypeVertexCount;
    uint triangleCount = vertexCount >= 3u ? vertexCount - 2u : 0u;
    SetMeshOutputCounts(vertexCount, triangleCount);
    if (lane < vertexCount) {
        VSMPrototypeClipVertex vertex = g_vsmPrototypePolygon[lane];
        float4 bounds = g_vsmPrototypePageBounds;
        // Clamp only the rounding error of the already clipped page coordinates.
        float2 pagePosition = saturate((vertex.position.xy - bounds.xy) / (bounds.zw - bounds.xy));
        float2 atlasPosition = vsmPrototypeAtlasPosition(pagePosition, g_vsmPrototypePhysicalPage, push);
        vertices[lane].position = float4(atlasPosition, vertex.position.z, 1.0);
        vertices[lane].uv = vertex.uv;
        vertices[lane].materialIndex = g_vsmPrototypeMaterialIndex;
    }
    if (lane < triangleCount) { triangles[lane] = uint3(0u, lane + 1u, lane + 2u); }
}

[shader("fragment")]
void vsmPrototypeDepthFragmentMain(VSMPrototypeRasterVertex input, bool frontFace : SV_IsFrontFace,
    uniform VSMPrototypePush push)
{
    // Dirty-page clearing uses the same fragment entry with an always-pass depth state.
    if (input.materialIndex == 0xffffffffu) { return; }
    StructuredBuffer<VSMPrototypeMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypeMaterial>>(push.materials));
    VSMPrototypeMaterial material = materials[input.materialIndex];
    if (!frontFace && material.resources.y == 0u) { discard; }
    if (material.alpha.z < 0.5) { return; }

    CoverageContext context;
    context.uv = input.uv;
    context.baseAlpha = material.alpha.x;
    context.cutoff = material.alpha.y;
    context.alphaMode = material.alpha.z;
    CoverageProgram program;
    program.inputs = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.coverageInputs));
    MaterialInstanceRef instance = coverageInstanceRef(input.materialIndex);
    if (program.usesBaseAlpha(instance) && material.resources.x != 0xffffffffu) {
        Texture2D<float4> alphaTexture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(material.resources.x));
        SamplerState alphaSampler = resolveUniform(SamplerHandle(push.sampler));
        context.baseAlpha *= alphaTexture.Sample(alphaSampler, input.uv).a;
    }
    if (!program.evaluateCoverage(context, instance).accepted) { discard; }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmPrototypeQueryMain(uint3 id : SV_DispatchThreadID, uniform VSMPrototypePush push)
{
    if (id.x >= push.receiverCount) { return; }
    StructuredBuffer<float4> receivers = resolveUniform(ResourceHandle<StructuredBuffer<float4>>(push.receivers));
    StructuredBuffer<VSMPrototypePage> pages = resolveUniform(ResourceHandle<StructuredBuffer<VSMPrototypePage>>(push.pages));
    StructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<VSMPrototypeQuery> output = resolveUniform(ResourceHandle<RWStructuredBuffer<VSMPrototypeQuery>>(push.queryResults));
    float4 receiver = receivers[id.x];
    float3 position = vsmPrototypeProject(float4(receiver.xyz, 1.0), push);
    float2 uv = position.xy * 0.5 + 0.5;
    VSMPrototypeQuery result;
    result.visibility = -1.0;
    result.depth = 0.0;
    result.virtualPage = 0xffffffffu;
    result.status = 2u;
    result.physicalPage = 0xffffffffu;
    result.reserved0 = 0u;
    result.reserved1 = 0u;
    result.reserved2 = 0u;
    if (all(uv >= 0.0) && all(uv < 1.0) && position.z >= 0.0 && position.z <= 1.0) {
        uint2 virtualTexel = uint2(floor(uv * float2(push.virtualPagesX, push.virtualPagesY) * float(push.pageSize)));
        uint2 virtualPage = virtualTexel / push.pageSize;
        uint2 pageTexel = virtualTexel % push.pageSize;
        result.virtualPage = virtualPage.y * push.virtualPagesX + virtualPage.x;
        result.physicalPage = pages[result.virtualPage].physicalPage;
        result.status = arguments[7] != 0u ? 3u : 1u;
        if (arguments[7] == 0u && result.physicalPage != 0xffffffffu) {
            uint2 physicalPage = uint2(result.physicalPage % push.physicalPagesX, result.physicalPage / push.physicalPagesX);
            uint2 physicalTexel = physicalPage * push.pageSize + pageTexel;
            Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
            result.depth = atlas.Load(int3(physicalTexel, 0));
            result.visibility = position.z + receiver.w >= result.depth ? 1.0 : 0.0;
            result.status = 0u;
        }
    }
    output[id.x] = result;
}
