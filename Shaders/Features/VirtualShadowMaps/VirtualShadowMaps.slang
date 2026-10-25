#language slang 2026
module VirtualShadowMaps;

import ShaderCore;
import Material;
import MaterialProgram;
import VirtualShadowMapGeometry;

using Metallic;
using Metallic.Material;
using Metallic.Shadow;

struct ShadowPush
{
    uint geometry, materials, pages, coverageInputs;
    uint workItems, statistics, projections, atlas;
    uint depth, shadow, debug, view;
    uint arguments, triangleCount, clipmapCount, virtualPagesPerAxis;
    uint physicalPagesX, physicalPagesY, physicalPageCount, pageSize;
    uint workCapacity, batchCount, workBase, rasterCoarse;
    uint sampler, enabled, totalPages, reserved;
};

struct ShadowProjection
{
    float4 rowX, rowY, rowZ, metric;
    uint4 page; // Virtual-page offset, pages per axis, level, coarse flag.
};

struct ShadowMaterial
{
    float4 alpha;
    uint4 resources;
    float4 uvX, uvY;
};

struct ShadowClipVertex
{
    float3 position;
    float2 uv;
};

struct ShadowRasterVertex
{
    float4 position : SV_Position;
    [vk::location(0)] float2 uv : TEXCOORD0;
    [vk::location(1)] nointerpolation uint materialIndex : TEXCOORD1;
};

float3 vsmProjectPosition(float3 position, ShadowProjection projection)
{
    float4 p = float4(position, 1.0);
    return float3(dot(projection.rowX, p), dot(projection.rowY, p), dot(projection.rowZ, p));
}

bool vsmContains(float3 position, ShadowProjection projection, uint pageSize, bool inset)
{
    float margin = inset ? 1.0 / float(projection.page.y * pageSize) : 0.0;
    return all(position.xy >= -1.0 + margin) && all(position.xy < 1.0 - margin) &&
        position.z >= 0.0 && position.z <= 1.0;
}

uint2 vsmVirtualTexel(float3 position, ShadowProjection projection, uint pageSize)
{
    return uint2(floor((position.xy * 0.5 + 0.5) * float(projection.page.y * pageSize)));
}

uint vsmVirtualPage(uint2 texel, ShadowProjection projection, uint pageSize)
{
    uint2 page = texel / pageSize;
    return projection.page.x + page.y * projection.page.y + page.x;
}

float4 vsmPageBounds(uint virtualPage, ShadowProjection projection)
{
    uint local = virtualPage - projection.page.x;
    uint2 page = uint2(local % projection.page.y, local / projection.page.y);
    return float4(float2(page) * (2.0 / float(projection.page.y)) - 1.0,
        float2(page + 1u) * (2.0 / float(projection.page.y)) - 1.0);
}

float2 vsmAtlasPosition(float2 pagePosition, uint physicalPage, ShadowPush push)
{
    uint2 page = uint2(physicalPage % push.physicalPagesX, physicalPage / push.physicalPagesX);
    return (float2(page) + pagePosition) * (2.0 / float2(push.physicalPagesX, push.physicalPagesY)) - 1.0;
}

// Uses the actual receiver projection, including temporal jitter in pixel units.
// clipOrtho.w selects reversed (1) or standard (0) receiver depth only; shadow
// projections always use reversed depth.
bool vsmReceiver(int2 pixel, ViewConstants view, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    position = float3(0.0);
    linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(view.current.viewport.yz))) { return false; }
    float value = depth.Load(int3(pixel, 0));
    bool reversed = view.current.clipOrtho.w > 0.5;
    if (value == (reversed ? 0.0 : 1.0)) { return false; }
    float nearPlane = view.current.clipOrtho.x;
    float farPlane = view.current.clipOrtho.y;
    if (view.current.upProjection.w > 0.5) {
        linearDepth = reversed ? lerp(farPlane, nearPlane, value) : lerp(nearPlane, farPlane, value);
    } else {
        float denominator = reversed ? nearPlane + value * (farPlane - nearPlane)
            : farPlane - value * (farPlane - nearPlane);
        linearDepth = nearPlane * farPlane / denominator;
    }
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    float2 uv = (float2(pixel) + 0.5 + jitter) / view.current.viewport.yz;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float halfHeight = view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z * 0.5
        : linearDepth * tan(view.current.viewport.w * 0.5);
    position = view.current.eye.xyz + forward * linearDepth +
        halfHeight * (right * ndc.x * view.current.viewport.x + up * ndc.y);
    return true;
}

float3 vsmReceiverNormal(int2 pixel, ViewConstants view, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    float3 left, right, top, bottom;
    float zl, zr, zt, zb;
    bool hasLeft = vsmReceiver(pixel - int2(1, 0), view, depth, left, zl);
    bool hasRight = vsmReceiver(pixel + int2(1, 0), view, depth, right, zr);
    bool hasTop = vsmReceiver(pixel - int2(0, 1), view, depth, top, zt);
    bool hasBottom = vsmReceiver(pixel + int2(0, 1), view, depth, bottom, zb);
    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));
    float3 dx = useLeft ? position - left : (hasRight ? right - position : float3(0.0));
    float3 dy = useTop ? position - top : (hasBottom ? bottom - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmResetMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    if (id.x < push.totalPages) {
        bool coarse = id.x == push.totalPages - 1u;
        pages[id.x] = uint4(coarse ? 0u : 0xffffffffu, coarse ? 1u : 0u,
            coarse ? push.clipmapCount : id.x / (push.virtualPagesPerAxis * push.virtualPagesPerAxis), 0u);
    }
    // Each counter has exactly one writer, including the diagnostic constants.
    if (id.x < 16u) {
        uint value = id.x == 10u ? push.triangleCount : id.x == 11u ? push.clipmapCount :
            id.x == 12u ? push.physicalPageCount : id.x == 14u ? 1u - push.enabled : 0u;
        statistics[id.x] = value;
    }
    if (id.x < push.batchCount * 3u) { arguments[id.x] = id.x % 3u == 0u ? 0u : 1u; }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmMarkMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    if (push.enabled == 0u) { return; }
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    ViewConstants view = views[0];
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
    if (!vsmReceiver(int2(id.xy), view, depth, position, linearDepth)) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    // All covering ancestors are requested, so an unallocated fine page can
    // use a coarser projection with its own origin and depth mapping.
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        ShadowProjection projection = projections[level];
        float3 projected = vsmProjectPosition(position, projection);
        if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
        uint page = vsmVirtualPage(vsmVirtualTexel(projected, projection, push.pageSize), projection, push.pageSize);
        InterlockedOr(pages[page].y, 1u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void vsmAllocateMain(uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint requested = 0u, allocated = 0u;
    // Virtual pages are laid out fine-to-coarse. Physical zero always belongs
    // to the complete scene projection and never competes for this budget.
    for (uint page = 0u; page + 1u < push.totalPages; ++page) {
        uint4 entry = pages[page];
        if (entry.y == 0u) { continue; }
        ++requested;
        if (allocated + 1u < push.physicalPageCount) {
            entry.x = ++allocated;
            pages[page] = entry;
        }
    }
    statistics[0] = requested;
    statistics[1] = allocated;
    statistics[2] = requested - allocated;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    uint triangleIndex = push.workBase + id.x;
    if (triangleIndex >= push.triangleCount || push.enabled == 0u) { return; }
    VirtualShadowMapTriangle triangle;
    if (!loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint2> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        ShadowProjection projection = projections[level];
        float3 a = vsmProjectPosition(triangle.vertices[0].position.xyz, projection);
        float3 b = vsmProjectPosition(triangle.vertices[1].position.xyz, projection);
        float3 c = vsmProjectPosition(triangle.vertices[2].position.xyz, projection);
        float3 lo = min(a, min(b, c)), hi = max(a, max(b, c));
        if (hi.z < 0.0 || lo.z > 1.0 || any(hi.xy < -1.0) || any(lo.xy > 1.0)) { continue; }
        int axis = int(projection.page.y);
        // Clip the projected bounds before integer conversion; off-clipmap
        // world coordinates need not fit the page-index integer range.
        int2 first = min(int2(floor(saturate(lo.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        int2 last = min(int2(floor(saturate(hi.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        for (int y = first.y; y <= last.y; ++y) {
            for (int x = first.x; x <= last.x; ++x) {
                uint page = projection.page.x + uint(y * axis + x);
                if (pages[page].x == 0xffffffffu) { continue; }
                uint index;
                InterlockedAdd(statistics[3], 1u, index);
                if (index < push.workCapacity) { work[index] = uint2(triangleIndex, page); }
                else { InterlockedAdd(statistics[5], 1u); }
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmFinalizeMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    uint count = min(statistics[3], push.workCapacity);
    if (id.x == 0u) { statistics[4] = count; }
    if (id.x >= push.batchCount) { return; }
    uint first = id.x * 65535u;
    arguments[id.x * 3u] = first < count ? min(count - first, 65535u) : 0u;
    arguments[id.x * 3u + 1u] = 1u;
    arguments[id.x * 3u + 2u] = 1u;
}

// Four XY half-planes turn a triangle into at most seven vertices. Clipping
// occurs before atlas remapping, so neighboring physical tiles stay isolated.
uint vsmClipPlane(inout ShadowClipVertex polygon[7], uint count, uint axis, float edge, float side)
{
    if (count == 0u) { return 0u; }
    ShadowClipVertex clipped[7];
    uint clippedCount = 0u;
    ShadowClipVertex previous = polygon[count - 1u];
    float previousDistance = side * (previous.position[axis] - edge);
    for (uint index = 0u; index < count; ++index) {
        ShadowClipVertex current = polygon[index];
        float currentDistance = side * (current.position[axis] - edge);
        if ((previousDistance < 0.0 && currentDistance > 0.0) ||
            (previousDistance > 0.0 && currentDistance < 0.0)) {
            float t = previousDistance / (previousDistance - currentDistance);
            ShadowClipVertex intersection;
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

groupshared ShadowClipVertex g_vsmPolygon[7];
groupshared uint g_vsmVertexCount;
groupshared uint g_vsmMaterial;
groupshared uint g_vsmPhysicalPage;
groupshared float4 g_vsmBounds;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[7], out indices uint3 triangles[5], uniform ShadowPush push)
{
    if (lane == 0u) {
        g_vsmVertexCount = 0u;
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        uint triangleIndex = push.workBase + group.x;
        uint virtualPage = push.totalPages - 1u;
        if (push.rasterCoarse == 0u) {
            StructuredBuffer<uint2> work = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
            uint2 item = work[push.workBase + group.x];
            triangleIndex = item.x;
            virtualPage = item.y;
        }
        VirtualShadowMapTriangle triangle;
        if (loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) {
            uint4 page = pages[virtualPage];
            ShadowProjection projection = projections[page.z];
            float4 bounds = vsmPageBounds(virtualPage, projection);
            ShadowClipVertex polygon[7];
            for (uint i = 0u; i < 3u; ++i) {
                polygon[i].position = vsmProjectPosition(triangle.vertices[i].position.xyz, projection);
                polygon[i].uv = triangle.vertices[i].uv.xy;
            }
            uint count = vsmClipPlane(polygon, 3u, 0u, bounds.x, 1.0);
            count = vsmClipPlane(polygon, count, 0u, bounds.z, -1.0);
            count = vsmClipPlane(polygon, count, 1u, bounds.y, 1.0);
            count = vsmClipPlane(polygon, count, 1u, bounds.w, -1.0);
            g_vsmVertexCount = count >= 3u ? count : 0u;
            g_vsmMaterial = triangle.materialIndex;
            g_vsmPhysicalPage = page.x;
            g_vsmBounds = bounds;
            for (uint i = 0u; i < count; ++i) { g_vsmPolygon[i] = polygon[i]; }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint vertexCount = g_vsmVertexCount;
    uint triangleCount = vertexCount >= 3u ? vertexCount - 2u : 0u;
    SetMeshOutputCounts(vertexCount, triangleCount);
    if (lane < vertexCount) {
        ShadowClipVertex vertex = g_vsmPolygon[lane];
        float2 pagePosition = saturate((vertex.position.xy - g_vsmBounds.xy) / (g_vsmBounds.zw - g_vsmBounds.xy));
        vertices[lane].position = float4(vsmAtlasPosition(pagePosition, g_vsmPhysicalPage, push), vertex.position.z, 1.0);
        vertices[lane].uv = vertex.uv;
        vertices[lane].materialIndex = g_vsmMaterial;
    }
    if (lane < triangleCount) { triangles[lane] = uint3(0u, lane + 1u, lane + 2u); }
}

[shader("fragment")]
void vsmDepthFragmentMain(ShadowRasterVertex input, bool frontFace : SV_IsFrontFace, uniform ShadowPush push)
{
    StructuredBuffer<ShadowMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<ShadowMaterial>>(push.materials));
    ShadowMaterial material = materials[input.materialIndex];
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
        Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(material.resources.x));
        SamplerState sampler = resolveUniform(SamplerHandle(push.sampler));
        float3 uv = float3(input.uv, 1.0);
        float2 textureUV = float2(dot(material.uvX.xyz, uv), dot(material.uvY.xyz, uv));
        context.baseAlpha *= texture.Sample(sampler, textureUV).a;
    }
    if (!program.evaluateCoverage(context, instance).accepted) { discard; }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmProjectMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    StructuredBuffer<ViewConstants> views = resolveUniform(ResourceHandle<StructuredBuffer<ViewConstants>>(push.view));
    ViewConstants view = views[0];
    if (any(id.xy >= uint2(view.current.viewport.yz))) { return; }
    RWTexture2D<float> shadow = resolveUniform(ResourceHandle<RWTexture2D<float>>(push.shadow));
    RWTexture2D<float4> debug = resolveUniform(ResourceHandle<RWTexture2D<float4>>(push.debug));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    Texture2D<float> depth = resolveUniform(ResourceHandle<Texture2D<float>>(push.depth));
    float3 position;
    float linearDepth;
    shadow[id.xy] = 1.0;
    debug[id.xy] = float4(0.0, 0.0, 0.0, 1.0);
    if (!vsmReceiver(int2(id.xy), view, depth, position, linearDepth)) {
        InterlockedAdd(statistics[8], 1u);
        return;
    }
    InterlockedAdd(statistics[6], 1u);
    if (push.enabled == 0u) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint selectedLevel = push.clipmapCount;
    uint firstLevel = 0xffffffffu;
    uint physicalPage = 0xffffffffu;
    uint2 virtualTexel = uint2(0u);
    float3 projected = float3(0.0);
    // Any fine-work overflow invalidates the entire partial fine atlas. The
    // independent coarse draw always traverses every caster slot.
    if (statistics[5] == 0u) {
        for (uint level = 0u; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float3 p = vsmProjectPosition(position, projection);
            if (!vsmContains(p, projection, push.pageSize, true)) { continue; }
            if (firstLevel == 0xffffffffu) { firstLevel = level; }
            uint2 texel = vsmVirtualTexel(p, projection, push.pageSize);
            uint page = vsmVirtualPage(texel, projection, push.pageSize);
            if (pages[page].x == 0xffffffffu) { continue; }
            physicalPage = pages[page].x;
            selectedLevel = level;
            projected = p;
            virtualTexel = texel;
            break;
        }
    }
    if (physicalPage == 0xffffffffu) {
        ShadowProjection coarse = projections[push.clipmapCount];
        projected = vsmProjectPosition(position, coarse);
        if (!vsmContains(projected, coarse, push.pageSize, false)) {
            shadow[id.xy] = 0.0;
            debug[id.xy] = float4(1.0, 0.0, 1.0, 1.0);
            InterlockedAdd(statistics[9], 1u);
            return;
        }
        physicalPage = 0u;
        virtualTexel = vsmVirtualTexel(projected, coarse, push.pageSize);
        InterlockedAdd(statistics[7], 1u);
        debug[id.xy] = float4(0.0, 0.0, 1.0, 1.0);
    } else {
        if (selectedLevel > firstLevel) { InterlockedAdd(statistics[13], 1u); }
        debug[id.xy] = float4(float(selectedLevel + 1u) / float(push.clipmapCount), 0.5, 0.0, 1.0);
    }
    ShadowProjection projection = projections[selectedLevel];
    float3 normal = vsmReceiverNormal(int2(id.xy), view, depth, position, linearDepth);
    float3 axisDepth = normalize(projection.rowZ.xyz);
    float denominator = dot(normal, axisDepth);
    float correction = 0.0;
    if (abs(denominator) > 1e-5) {
        float2 sampleNDC = (float2(virtualTexel) + 0.5) *
            (2.0 / float(projection.page.y * push.pageSize)) - 1.0;
        float2 offset = (sampleNDC - projected.xy) * projection.metric.x;
        float3 deltaXYWorld = normalize(projection.rowX.xyz) * offset.x + normalize(projection.rowY.xyz) * offset.y;
        correction = -dot(normal, deltaXYWorld) * length(projection.rowZ.xyz) / denominator;
    }
    uint2 physical = uint2(physicalPage % push.physicalPagesX, physicalPage / push.physicalPagesX);
    uint2 texel = physical * push.pageSize + virtualTexel % push.pageSize;
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    float storedDepth = atlas.Load(int3(texel, 0));
    // The producer's scene-Z margin keeps real casters above zero. A receiver
    // plane extrapolated past the scene must not turn empty depth into a blocker.
    shadow[id.xy] = storedDepth == 0.0 || projected.z + correction + projection.metric.z >= storedDepth ? 1.0 : 0.0;
}
