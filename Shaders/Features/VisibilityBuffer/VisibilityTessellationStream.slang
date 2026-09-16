// Task/mesh patch path for the same active cut used by ordinary stream raster.
struct TessStreamPayload { uint record; uint first; uint count; uint material; uint tessellated; };
groupshared TessStreamPayload g_TessStream;
groupshared float3 g_StreamTessSource[3];
groupshared float3 g_StreamTessNormal[3];
groupshared float2 g_StreamTessUv[3];
groupshared uint3 g_StreamTessRates;
groupshared uint4 g_StreamTessPattern;

[shader("amplification")]
[numthreads(32, 1, 1)]
void streamTessellationTask(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    if (lane == 0u) {
        uint draw = hybridFlatGroup(group);
        bool prebinned = push.hybridClusterBuffer != 0xffffffffu;
        uint record = prebinned ? draw : draw / 2u;
        uint chunk = prebinned ? 0u : draw % 2u;
        bool valid = true;
        if (prebinned) {
            DescriptorHandle<StructuredBuffer<uint>> bins = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
            valid = hybridBinRecord(bins, 0u, draw, record);
        }
        StreamRasterCluster c; bool retry;
        if (valid) { valid = loadStreamRasterCluster(record, !prebinned, !prebinned && chunk == 0u, push, c, retry); }
        g_TessStream = (TessStreamPayload)0;
        if (valid) {
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> rh = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
            GPUDrivenStreamAssetRasterBindings bindings = rh[0];
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instances = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(push.bufferShaderIndexBase + bindings.gpuSceneInstanceBuffer, 0));
            DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + bindings.tessellationBuffer, 0));
            DescriptorHandle<StructuredBuffer<uint>> pages = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
            uint material = instances[c.group.gpuSceneInstanceIndex].identity.y;
            // Do not invent UVs for position-only assets. Such clusters retain
            // their original geometry until recooked with authored attributes.
            bool attributes = (pages[c.pageWord + 14u] & 6u) == 6u && pages[c.pageWord + 21u] == 2u && pages[c.pageWord + 22u] == 1u;
            g_TessStream.record = record;
            g_TessStream.first = chunk * 64u;
            g_TessStream.count = prebinned ? c.triangleCount : min(c.triangleCount - min(c.triangleCount, chunk * 64u), 64u);
            g_TessStream.material = material;
            g_TessStream.tessellated = attributes && tessMaterial(data, material).enabled != 0u ? 1u : 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(g_TessStream.count == 0u ? 0u : (g_TessStream.tessellated != 0u ? g_TessStream.count : 1u), 1, 1, g_TessStream);
}

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void streamTessellationMesh(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    in payload TessStreamPayload payload, out vertices TessVertex vertices[128],
    out indices uint3 triangles[128], out primitives TessPrimitive primitives[128], uniform GPUDrivenStreamAssetPush push)
{
    if (lane == 0u) {
        bool retry;
        g_StreamMeshValid = loadStreamRasterCluster(payload.record, false, false, push, g_StreamMeshCluster, retry) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamMeshValid == 0u) { SetMeshOutputCounts(0, 0); return; }
    StreamRasterCluster c = g_StreamMeshCluster;
    if (payload.tessellated == 0u) {
        SetMeshOutputCounts(c.vertexCount, payload.count);
        for (uint v = lane; v < c.vertexCount; v += 64u) {
            vertices[v].position = streamClusterClip(c, v, push);
            vertices[v].uv = float2(0); vertices[v].bary = float2(-1);
            vertices[v].world = float3(0); vertices[v].doubleSided = (c.instanceFlags & 2u) != 0u;
        }
        for (uint t = lane; t < payload.count; t += 64u) {
            triangles[t] = streamClusterTriangle(c, payload.first + t, push);
            primitives[t].visibilityId = packVisibilityId(c.recordIndex, payload.first + t);
        }
        return;
    }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> rh = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + rh[0].tessellationBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> pages = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
    TessMaterial material = tessMaterial(data, payload.material);
    uint originalTriangle = payload.first + group.x;
    // Decode in authored order; only the output index winding is reflected.
    uint address = c.triangleBaseByte + originalTriangle * 3u;
    uint3 original = uint3(readPageByte(pages, address), readPageByte(pages, address + 1u), readPageByte(pages, address + 2u));
    uint payloadBytes = pages[c.pageWord + 12u], normalOffset = pages[c.pageWord + 16u], uvOffset = pages[c.pageWord + 17u];
    uint required = c.vertexOffset + c.vertexCount;
    if (any(original >= c.vertexCount) || (normalOffset | uvOffset) % 4u != 0u ||
        normalOffset > payloadBytes || required > (payloadBytes - normalOffset) / 16u ||
        uvOffset > payloadBytes || required > (payloadBytes - uvOffset) / 8u) { SetMeshOutputCounts(0, 0); return; }
    if (lane < 3u) {
        uint v = c.vertexOffset + original[lane];
        uint p = c.pageWord + c.positionOffsetBytes / 4u + v * c.positionStrideWords;
        float3 local = asfloat(uint3(pages[p], pages[p + 1u], pages[p + 2u]));
        g_StreamTessSource[lane] = c.group.world0.xyz * local.x + c.group.world1.xyz * local.y + c.group.world2.xyz * local.z + c.group.world3.xyz;
        uint n = c.pageWord + normalOffset / 4u + v * 4u;
        g_StreamTessNormal[lane] = transformStreamNormal(c.group, asfloat(uint3(pages[n], pages[n + 1u], pages[n + 2u])), float3(0, 1, 0)) *
            (dot(cross(c.group.world0.xyz, c.group.world1.xyz), c.group.world2.xyz) < 0.0 ? -1.0 : 1.0);
        uint uv = c.pageWord + uvOffset / 4u + v * 2u;
        g_StreamTessUv[lane] = asfloat(uint2(pages[uv], pages[uv + 1u]));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        float3 forward = normalize(c.params.renderCenter.xyz - c.params.renderEye.xyz);
        bool ortho = c.params.renderUpProjection.w > 0.5;
        float scale = c.params.renderViewport.z / (ortho ? c.params.renderClipOrtho.z : 2.0 * tan(c.params.renderViewport.w * 0.5));
        for (uint e = 0; e < 3u; ++e) {
            g_StreamTessRates[e] = tessEdgeRate(g_StreamTessSource[e], g_StreamTessSource[(e + 1u) % 3u], c.params.renderEye.xyz,
                forward, scale, c.params.renderClipOrtho.x, ortho, asfloat(data[2]), data[3]);
        }
        g_StreamTessPattern = tessPattern(data, g_StreamTessRates);
    }
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(g_StreamTessPattern.y, g_StreamTessPattern.w);
    if (lane < g_StreamTessPattern.y) {
        float3 bary = tessBary(data, g_StreamTessPattern, lane);
        float3 p, n; float2 uv;
        tessDomain(bary, g_StreamTessRates, g_StreamTessSource, g_StreamTessNormal, g_StreamTessUv, p, n, uv);
        p += n * tessHeight(material, uv, push.imageShaderIndexBase);
        float4 clip = projectPosition(p, c.params); clip.y = -clip.y;
        vertices[lane].position = clip; vertices[lane].uv = uv;
        vertices[lane].bary = bary.yz;
        vertices[lane].world = p; vertices[lane].doubleSided = (c.instanceFlags & 2u) != 0u;
    }
    bool reflected = dot(cross(c.group.world0.xyz, c.group.world1.xyz), c.group.world2.xyz) < 0.0;
    for (uint t = lane; t < g_StreamTessPattern.w; t += 64u) {
        uint3 tri = tessTriangle(data, g_StreamTessPattern, t);
        triangles[t] = reflected ? tri.xzy : tri;
        primitives[t].visibilityId = packVisibilityId(c.recordIndex, originalTriangle);
    }
}

[shader("fragment")]
TessFragment streamTessellationFragment(TessVertex input, uint visibilityId : SV_PrimitiveID, bool front : SV_IsFrontFace)
{
    if (!front && input.doubleSided == 0u) { discard; }
    TessFragment output;
    output.visibility = visibilityId;
    // Derivatives of the displaced plane avoid Slang's unsupported user
    // per-primitive fragment varyings (slang#7019). Preserve authored winding.
    output.domain = tessSurfaceDomain(input.bary, input.world, input.uv, front);
    return output;
}
