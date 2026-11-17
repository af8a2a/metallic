#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_PAGE_WINDOW_RASTER
#define VSM_PAGE_WINDOW_RASTER 0
#endif
#ifndef VSM_CLUSTER_PAGE_RASTER
#define VSM_CLUSTER_PAGE_RASTER 0
#endif

struct ShadowClearVertex
{
    float4 position : SV_Position;
};

[shader("fragment")]
float vsmClearFragmentMain(ShadowClearVertex input, uniform ShadowPush push) : SV_Depth
{
    if ((push.cacheMode & 16u) != 0u && (push.rasterCoarse & 2u) == 0u) {
        Texture2D<float> staticAtlas = resolveUniform(ResourceHandle<Texture2D<float>>(vsmCacheMetadata(push, 32u)));
        return staticAtlas.Load(int3(int2(input.position.xy), 0));
    }
    return 0.0;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmClearMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowClearVertex vertices[4], out indices uint3 triangles[2], uniform ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint owner = owners[group.x];
    bool clear = false;
    if (owner != 0u && push.enabled != 0u) {
        uint4 entry = pages[owner - 1u];
        bool staticPass = (push.rasterCoarse & 2u) != 0u;
        clear = (entry.y & 2u) != 0u && (staticPass ? (entry.w & (8u | kVSMPageMergeStatic)) == 8u : (entry.w & 2u) != 0u);
    }
    SetMeshOutputCounts(clear ? 4u : 0u, clear ? 2u : 0u);
    if (!clear) { return; }
    if (lane == 0u) {
        vsmWorkAdd(push, kVSMWorkRasterClearedPages, 1u);
        if ((push.cacheMode & 16u) != 0u && (push.rasterCoarse & 2u) == 0u) { vsmCacheAdd(push, kVSMCacheRestoredPages, 1u); }
    }
    if (lane < 4u) {
        float2 corner = float2(lane & 1u, lane >> 1u);
        vertices[lane].position = float4(vsmAtlasPosition(corner, group.x, push), 0.0, 1.0);
    }
    if (lane < 2u) { triangles[lane] = lane == 0u ? uint3(0u, 2u, 1u) : uint3(1u, 2u, 3u); }
}

#if VSM_PAGE_WINDOW_RASTER
bool vsmRasterPageLayerActive(inout uint triangleIndex, uint4 page, ShadowPush push, out bool mergeStatic)
{
    const bool staticPass = (push.rasterCoarse & 2u) != 0u;
    bool layerPageActive = staticPass ? (page.w & 8u) != 0u :
        ((push.cacheMode & 32u) == 0u || (page.w & 2u) != 0u);
#if VSM_CLUSTER_HZB
    if (layerPageActive && (push.rasterCoarse & 4u) != 0u) {
        StructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        layerPageActive = (visibility[triangleIndex >> 6u] & (1u << page.z)) != 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
        if (!layerPageActive) {
            uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, triangleIndex >> 5u);
            vsmWorkAdd(push, kVSMWorkRasterHZBSkippedTriangles, (dynamicBits >> (triangleIndex & 31u)) & 1u);
        }
#endif
    }
#endif
    mergeStatic = staticPass && (push.cacheMode & 32u) != 0u && (page.w & kVSMPageMergeStatic) != 0u;
    if (mergeStatic) {
        if ((push.rasterCoarse & 1u) != 0u) {
            uint4 record = vsmMergeCluster(push, triangleIndex >> 6u);
            uint sourceLane = triangleIndex & 63u;
            triangleIndex = record.x + sourceLane;
            layerPageActive = vsmMergeClusterHasLane(record, sourceLane);
        } else {
            uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
            layerPageActive = (mask & (1u << (triangleIndex & 31u))) != 0u;
        }
    }
    return layerPageActive;
}

float4 vsmDirectionalPageBounds(uint virtualPage, inout ShadowProjection projection)
{
    uint local = virtualPage - projection.page.x;
    uint2 pageXY = uint2(local % projection.page.y, local / projection.page.y);
    // The snapped world-page translation belongs inside the vertex dot product.
    // Subtracting rolling clipmap bounds afterwards changes cached depths.
    float2 center = projection.originNear.xy + float2(pageXY) + 0.5 - 0.5 * float(projection.page.y);
    float halfPage = 1.0 / float(projection.page.y);
    projection.rowX.w = -center.x * (2.0 * halfPage);
    projection.rowY.w = -center.y * (2.0 * halfPage);
    return float4(-halfPage, -halfPage, halfPage, halfPage);
}
#endif

// Fine packed and single-triangle raster must produce the same fan, UVs and
// canonical page coordinates, including for alpha-masked materials.
uint vsmRasterPolygon(uint triangleIndex, uint virtualPage, ShadowPush push, bool skipInteriorClip,
    out ShadowClipVertex polygon[9], out uint material, out uint physicalPage, out float4 bounds)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    material = physicalPage = 0u;
    bounds = float4(0.0);
    VirtualShadowMapTriangle triangle;
    const bool staticPass = (push.rasterCoarse & 2u) != 0u;
    bool layerPageActive = staticPass ? (pages[virtualPage].w & 8u) != 0u :
        ((push.cacheMode & 32u) == 0u || (pages[virtualPage].w & 2u) != 0u);
#if VSM_CLUSTER_HZB
    if (layerPageActive && (push.rasterCoarse & 4u) != 0u) {
        StructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        layerPageActive = (visibility[triangleIndex >> 6u] & (1u << pages[virtualPage].z)) != 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
        if (!layerPageActive) {
            uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, triangleIndex >> 5u);
            vsmWorkAdd(push, kVSMWorkRasterHZBSkippedTriangles, (dynamicBits >> (triangleIndex & 31u)) & 1u);
        }
#endif
    }
#endif
    bool mergeStatic = staticPass && (push.cacheMode & 32u) != 0u && (pages[virtualPage].w & kVSMPageMergeStatic) != 0u;
    if (mergeStatic) {
        if ((push.rasterCoarse & 1u) != 0u) {
            uint4 record = vsmMergeCluster(push, triangleIndex >> 6u);
            uint sourceLane = triangleIndex & 63u;
            triangleIndex = record.x + sourceLane;
            layerPageActive = vsmMergeClusterHasLane(record, sourceLane);
        } else {
            uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
            layerPageActive = (mask & (1u << (triangleIndex & 31u))) != 0u;
        }
    }
    if (layerPageActive && loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) {
        vsmWorkAdd(push, kVSMWorkRasterDecodedTriangles, 1u);
#if VSM_WORKLOAD_DIAGNOSTICS
        if (mergeStatic) { vsmWorkAdd(push, kVSMWorkRasterStaticMergedTriangles, 1u); }
        if ((push.cacheMode & 16u) != 0u) {
            vsmCacheAdd(push, staticPass ? kVSMCacheStaticDecodedTriangles : kVSMCacheDynamicDecodedTriangles, 1u);
        }
#endif
        uint4 page = pages[virtualPage];
        ShadowProjection projection = projections[page.z];
        bounds = vsmPageBounds(virtualPage, projection);
        if (projection.metric.w < 0.5 && (push.rasterCoarse & 1u) == 0u) {
            uint local = virtualPage - projection.page.x;
            uint2 pageXY = uint2(local % projection.page.y, local / projection.page.y);
            // Directional originNear.xy carries the snapped page-grid
            // origin. Form a canonical world-page translation before the
            // vertex dot product: subtracting rolling clipmap bounds after
            // projection changes clipping/depth rounding in cached pages.
            float2 center = projection.originNear.xy + float2(pageXY) + 0.5 - 0.5 * float(projection.page.y);
            float halfPage = 1.0 / float(projection.page.y);
            projection.rowX.w = -center.x * (2.0 * halfPage);
            projection.rowY.w = -center.y * (2.0 * halfPage);
            bounds = float4(-halfPage, -halfPage, halfPage, halfPage);
        }
        bool inside = skipInteriorClip;
        for (uint i = 0u; i < 3u; ++i) {
            polygon[i].position = vsmProjectClip(triangle.vertices[i].position.xyz, projection);
            polygon[i].uv = triangle.vertices[i].uv.xy;
            if (skipInteriorClip) {
                float4 clip = polygon[i].position;
                inside = inside && all(clip.xy >= bounds.xy * clip.w) && all(clip.xy <= bounds.zw * clip.w) &&
                    (projection.metric.w <= 0.5 || (clip.z >= 0.0 && clip.z <= clip.w));
            }
        }
        material = triangle.materialIndex;
        physicalPage = page.x;
        uint count = inside ? 3u : vsmClipProjection(polygon, bounds, projection.metric.w > 0.5);
        return count >= 3u ? count : 0u;
    }
    return 0u;
}

ShadowRasterVertex vsmRasterVertex(ShadowClipVertex vertex, float4 bounds, uint physicalPage, uint material, ShadowPush push)
{
    float2 pagePosition = saturate((vertex.position.xy / vertex.position.w - bounds.xy) / (bounds.zw - bounds.xy));
    ShadowRasterVertex result;
    result.position = float4(vsmAtlasPosition(pagePosition, physicalPage, push) * vertex.position.w,
        vertex.position.z, vertex.position.w);
    result.uv = vertex.uv;
    result.materialIndex = material;
    return result;
}

void vsmRecordRasterOutput(ShadowPush push, uint triangleCount)
{
    vsmWorkAdd(push, kVSMWorkRasterOutputPrimitives, triangleCount);
#if VSM_WORKLOAD_DIAGNOSTICS
    if ((push.cacheMode & 16u) != 0u) {
        vsmCacheAdd(push, (push.rasterCoarse & 2u) != 0u ? kVSMCacheStaticOutputPrimitives : kVSMCacheDynamicOutputPrimitives, triangleCount);
    }
#endif
}

groupshared ShadowClipVertex g_vsmPolygon[9];
groupshared uint g_vsmVertexCount;
groupshared uint g_vsmMaterial;
groupshared uint g_vsmPhysicalPage;
groupshared float4 g_vsmBounds;

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[9], out indices uint3 triangles[7], uniform ShadowPush push)
{
    if (lane == 0u) {
        vsmWorkAdd(push, (push.rasterCoarse & 1u) != 0u ? kVSMWorkRasterCoarseGroups : kVSMWorkRasterFineGroups, 1u);
        uint triangleIndex = push.workBase + group.x;
        uint virtualPage = push.totalPages - 1u;
        if ((push.rasterCoarse & 1u) == 0u) {
            StructuredBuffer<uint2> work = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.workItems));
            uint2 item = work[push.workBase + group.x];
            triangleIndex = item.x;
            virtualPage = item.y;
        }
        ShadowClipVertex polygon[9];
        uint material, physicalPage;
        float4 bounds;
        uint count = vsmRasterPolygon(triangleIndex, virtualPage, push, false, polygon, material, physicalPage, bounds);
        g_vsmVertexCount = count;
        g_vsmMaterial = material;
        g_vsmPhysicalPage = physicalPage;
        g_vsmBounds = bounds;
        for (uint i = 0u; i < count; ++i) { g_vsmPolygon[i] = polygon[i]; }
    }
    GroupMemoryBarrierWithGroupSync();
    uint vertexCount = g_vsmVertexCount;
    uint triangleCount = vertexCount >= 3u ? vertexCount - 2u : 0u;
    if (lane == 0u) { vsmRecordRasterOutput(push, triangleCount); }
    SetMeshOutputCounts(vertexCount, triangleCount);
    if (lane < vertexCount) {
        vertices[lane] = vsmRasterVertex(g_vsmPolygon[lane], g_vsmBounds, g_vsmPhysicalPage, g_vsmMaterial, push);
    }
    if (lane < triangleCount) { triangles[lane] = uint3(0u, lane + 1u, lane + 2u); }
}

#if VSM_CLUSTER_PAGE_RASTER
// A Build ticket is one nonempty 16-slot quarter of an original 64-triangle
// cluster. Six clip planes can expand each source triangle to nine vertices.
groupshared uint g_vsmPackedOffsets[17];

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterPackedMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[144], out indices uint3 triangles[112], uniform ShadowPush push)
{
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    uint4 ticket = work[1u + push.workBase + group.x];
    uint bits = ticket.z != 0u ? ticket.z : ticket.w;
    uint quarter = (ticket.z != 0u ? 0u : 32u) + ((bits & 0xffffu) != 0u ? 0u : 16u);
    uint slotMask = bits >> (quarter & 31u);
    ShadowClipVertex polygon[9];
    uint vertexCount = 0u, material = 0u, physicalPage = 0u;
    float4 bounds = float4(0.0);
    if (lane < 16u) {
        if ((slotMask & (1u << lane)) != 0u) {
            vertexCount = vsmRasterPolygon(ticket.x + quarter + lane, ticket.y, push, true,
                polygon, material, physicalPage, bounds);
        }
        // A packed sum gives both output offsets; max total 144 cannot carry.
        g_vsmPackedOffsets[lane] = vertexCount | (uint(max(int(vertexCount) - 2, 0)) << 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        uint sum = 0u;
        for (uint i = 0u; i < 16u; ++i) {
            uint counts = g_vsmPackedOffsets[i];
            g_vsmPackedOffsets[i] = sum;
            sum += counts;
        }
        g_vsmPackedOffsets[16] = sum;
        vsmWorkAdd(push, kVSMWorkRasterFineGroups, 1u);
        vsmRecordRasterOutput(push, sum >> 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint totals = g_vsmPackedOffsets[16];
    SetMeshOutputCounts(totals & 0xffffu, totals >> 16u);
    if (vertexCount != 0u) {
        uint offsets = g_vsmPackedOffsets[lane];
        uint firstVertex = offsets & 0xffffu;
        for (uint i = 0u; i < vertexCount; ++i) {
            vertices[firstVertex + i] = vsmRasterVertex(polygon[i], bounds, physicalPage, material, push);
        }
        for (uint i = 2u; i < vertexCount; ++i) {
            triangles[(offsets >> 16u) + i - 2u] = uint3(firstVertex, firstVertex + i - 1u, firstVertex + i);
        }
    }
}
#endif

#if VSM_PAGE_WINDOW_RASTER
// Each lane clips to one of four consecutive virtual pages. A directional
// triangle clipped by four XY planes produces at most seven vertices.
groupshared uint g_vsmWindowCounts[4];
groupshared ShadowClipVertex g_vsmWindowPolygon[28];
groupshared float4 g_vsmWindowBounds[4];
groupshared uint2 g_vsmWindowPageMaterial[4];

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void vsmRasterWindowMeshMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ShadowRasterVertex vertices[28], out indices uint3 triangles[20], uniform ShadowPush push)
{
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    uint ticketIndex = 1u + push.workBase + group.x;
    uint4 ticket = work[ticketIndex];
    ShadowClipVertex polygon[9];
    uint count = 0u, physicalPage = 0u, material = 0u;
    float4 bounds = float4(0.0);
    if (lane < 4u) {
        if ((ticket.z & (1u << lane)) != 0u) {
            StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
            uint virtualPage = ticket.y + lane;
            uint4 page = pages[virtualPage];
            uint triangleIndex = ticket.x;
            bool mergeStatic;
            if (vsmRasterPageLayerActive(triangleIndex, page, push, mergeStatic) &&
                virtualShadowMapRasterLayerContains(push.geometry, triangleIndex)) {
                StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
                ShadowProjection projection = projections[page.z];
                // Preserve the scalar raster's projection and clip control flow.
                // The producer supplies decoded world positions. The final
                // depth dot stays in the mesh stage alongside canonical XY.
                bounds = vsmPageBounds(virtualPage, projection);
                if (projection.metric.w < 0.5 && (push.rasterCoarse & 1u) == 0u) {
                    bounds = vsmDirectionalPageBounds(virtualPage, projection);
                }
                uint payload = 1u + push.workCapacity + 4u * ticket.w;
                uint4 attributes = work[payload + 3u];
                for (uint i = 0u; i < 3u; ++i) {
                    float4 source = asfloat(work[payload + i]);
                    polygon[i].position = vsmProjectClip(source.xyz, projection);
                    polygon[i].uv = float2(source.w, asfloat(attributes[i]));
                }
                material = attributes.w;
                physicalPage = page.x;
                count = vsmClipProjection(polygon, bounds, projection.metric.w > 0.5);
                if (count < 3u) { count = 0u; }
#if VSM_WORKLOAD_DIAGNOSTICS
                if (mergeStatic) { vsmWorkAdd(push, kVSMWorkRasterStaticMergedTriangles, 1u); }
#endif
            }
        }
        g_vsmWindowCounts[lane] = count | (uint(max(int(count) - 2, 0)) << 16u);
        g_vsmWindowBounds[lane] = bounds;
        g_vsmWindowPageMaterial[lane] = uint2(physicalPage, material);
        for (uint i = 0u; i < count; ++i) {
            g_vsmWindowPolygon[lane * 7u + i] = polygon[i];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint pageLane = lane >> 3u;
    uint vertex = lane & 7u;
    uint total = 0u, offset = 0u;
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        uint counts = g_vsmWindowCounts[i];
        total += counts;
        if (i < pageLane) { offset += counts; }
    }
    SetMeshOutputCounts(total & 0xffffu, total >> 16u);
    if (lane == 0u) {
        vsmWorkAdd(push, kVSMWorkRasterFineGroups, 1u);
        vsmRecordRasterOutput(push, total >> 16u);
    }
    uint first = offset & 0xffffu;
    uint pageCounts = g_vsmWindowCounts[pageLane];
    if (vertex < (pageCounts & 0xffffu)) {
        // Match the scalar path's rounded clipping/bounds handoff. In
        // particular, 1/axis is inexact for non-power-of-two clipmaps.
        uint2 pageMaterial = g_vsmWindowPageMaterial[pageLane];
        vertices[first + vertex] = vsmRasterVertex(g_vsmWindowPolygon[pageLane * 7u + vertex],
            g_vsmWindowBounds[pageLane], pageMaterial.x, pageMaterial.y, push);
    }
    if (vertex < (pageCounts >> 16u)) {
        triangles[(offset >> 16u) + vertex] = uint3(first, first + vertex + 1u, first + vertex + 2u);
    }
}
#endif

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

