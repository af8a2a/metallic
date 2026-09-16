// Included after VisibilityBuffer.slang's ordinary cluster accessors.
struct TessResidentPayload
{
    uint records[32];
    uint ends[32];
};
groupshared TessResidentPayload g_TessResident;
groupshared float3 g_TessSourcePosition[3];
groupshared float3 g_TessSourceNormal[3];
groupshared float2 g_TessSourceUv[3];
groupshared uint3 g_TessRates;
groupshared uint4 g_TessPattern;

[shader("amplification")]
[numthreads(32, 1, 1)]
void visibilityTessellationTask(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenPreviewPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> ph =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = ph[0];
    DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.tessellationBuffer, 0));
    uint candidate = hybridFlatGroup(group) * 32u + lane, record = 0xffffffffu;
    bool visible;
    if (push.environmentSHBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<uint>> bins = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.environmentSHBuffer, 0));
        visible = hybridBinRecord(bins, push.mipLevel, candidate, record);
    } else {
        record = residentLodRecordIndex(candidate, params, push);
        visible = residentClusterVisible(record, push);
        if (visible) { visible = visibleClusterDrawBucket(residentRasterCluster(record, push).record) == push.mipLevel; }
    }
    uint count = 0;
    if (visible) {
        ResidentRasterCluster c = residentRasterCluster(record, push);
        count = tessMaterial(data, c.instance.identity.y).enabled != 0u ? c.meshlet.ranges.w : 1u;
    }
    g_TessResident.records[lane] = record;
    bool singleWave = false;
#if GPU_DRIVEN_AMPLIFICATION_WAVE_OPS
    singleWave = WaveGetLaneCount() >= 32u;
    g_TessResident.ends[lane] = singleWave ? WavePrefixSum(count) + count : count;
#else
    g_TessResident.ends[lane] = count;
#endif
    GroupMemoryBarrierWithGroupSync();
    // Wave32/64 uses a register prefix; retain a tiny shared scan only for
    // devices with smaller subgroups or no task-shader subgroup operations.
    if (!singleWave && lane == 0u) {
        uint sum = 0;
        for (uint i = 0; i < 32u; ++i) { sum += g_TessResident.ends[i]; g_TessResident.ends[i] = sum; }
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(g_TessResident.ends[31], 1, 1, g_TessResident);
}

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void visibilityTessellationMesh(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    in payload TessResidentPayload payload, out vertices TessVertex vertices[128],
    out indices uint3 triangles[128], out primitives TessPrimitive primitives[128],
    uniform GPUDrivenPreviewPush push)
{
    uint source = 0;
    for (uint stride = 16; stride > 0; stride >>= 1) {
        uint probe = source + stride - 1;
        if (payload.ends[probe] <= group.x) { source += stride; }
    }
    uint record = payload.records[source];
    uint originalTriangle = group.x - (source == 0u ? 0u : payload.ends[source - 1u]);
    ResidentRasterCluster c = residentRasterCluster(record, push);
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> ph =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = ph[0];
    DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.tessellationBuffer, 0));
    TessMaterial material = tessMaterial(data, c.instance.identity.y);
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> vb = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.bufferShaderIndexBase + push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> vi = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.meshletVertexBuffer, 0));
    CameraData camera = push.projectWithCullingCamera != 0u ? cullingCamera(params) : renderCamera(params);
    SceneTransform transform = instanceTransform(c.instance);
    if (material.enabled == 0u) {
        SetMeshOutputCounts(c.meshlet.ranges.y, c.meshlet.ranges.w);
        for (uint v = lane; v < c.meshlet.ranges.y; v += 64u) {
            GPUDrivenPreviewVertex input = vb[c.geometry.payload.x + vi[c.meshlet.ranges.x + v]];
            vertices[v].position = residentClusterClip(c, v, params, push);
            vertices[v].uv = input.texcoord;
            vertices[v].bary = float2(-1);
#if VISIBILITY_BUFFER_ALPHA_MASKED
            vertices[v].material = c.instance.identity.y;
#endif
            vertices[v].world = float3(0);
        }
        for (uint t = lane; t < c.meshlet.ranges.w; t += 64u) {
            triangles[t] = residentClusterTriangle(c, t, push);
            primitives[t].visibilityId = packVisibilityId(record, t);
        }
        return;
    }
    uint3 original = residentClusterTriangle(c, originalTriangle, push);
    if (lane < 3u) {
        GPUDrivenPreviewVertex input = vb[c.geometry.payload.x + vi[c.meshlet.ranges.x + original[lane]]];
        g_TessSourcePosition[lane] = transformPoint(transform, input.position.xyz);
        float3 co0 = cross(transform.world1.xyz, transform.world2.xyz);
        float3 co1 = cross(transform.world2.xyz, transform.world0.xyz);
        float3 co2 = cross(transform.world0.xyz, transform.world1.xyz);
        g_TessSourceNormal[lane] = tessNormal((co0 * input.normal.x + co1 * input.normal.y + co2 * input.normal.z) *
            (dot(transform.world0.xyz, co0) < 0.0 ? -1.0 : 1.0), float3(0, 1, 0));
        g_TessSourceUv[lane] = input.texcoord;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        float3 forward = normalize(params.renderCenter.xyz - params.renderEye.xyz);
        bool ortho = params.renderUpProjection.w > 0.5;
        float scale = float(params.height) / (ortho ? params.renderClipOrtho.z : 2.0 * tan(params.renderViewport.w * 0.5));
        for (uint e = 0; e < 3u; ++e) {
            g_TessRates[e] = tessEdgeRate(g_TessSourcePosition[e], g_TessSourcePosition[(e + 1u) % 3u],
                params.renderEye.xyz, forward, scale, params.renderClipOrtho.x, ortho, asfloat(data[2]), data[3]);
        }
        g_TessPattern = tessPattern(data, g_TessRates);
    }
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(g_TessPattern.y, g_TessPattern.w);
    if (lane < g_TessPattern.y) {
        float3 bary = tessBary(data, g_TessPattern, lane);
        float3 p, n; float2 uv;
        tessDomain(bary, g_TessRates, g_TessSourcePosition, g_TessSourceNormal, g_TessSourceUv, p, n, uv);
        p += n * tessHeight(material, uv, push.imageShaderIndexBase);
        float4 clip = projectPosition(p, camera); clip.y = -clip.y;
        if (push.projectWithCullingCamera == 0u) { clip.xy -= 2.0 * float2(params.renderEye.w, params.renderCenter.w) / float2(params.width, params.height) * clip.w; }
        vertices[lane].position = clip;
        vertices[lane].uv = uv;
        vertices[lane].bary = bary.yz;
#if VISIBILITY_BUFFER_ALPHA_MASKED
        vertices[lane].material = c.instance.identity.y;
#endif
        vertices[lane].world = p;
    }
    bool reflected = dot(cross(transform.world0.xyz, transform.world1.xyz), transform.world2.xyz) < 0.0;
    for (uint t = lane; t < g_TessPattern.w; t += 64u) {
        uint3 tri = tessTriangle(data, g_TessPattern, t);
        triangles[t] = reflected ? tri.xzy : tri;
        primitives[t].visibilityId = packVisibilityId(record, originalTriangle);
    }
}

[shader("fragment")]
TessFragment visibilityTessellationFragment(TessVertex input, uint visibilityId : SV_PrimitiveID, bool front : SV_IsFrontFace,
    uniform GPUDrivenPreviewPush push)
{
#if VISIBILITY_BUFFER_ALPHA_MASKED
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> parameters = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>> materials = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(uint2(push.bufferShaderIndexBase + push.materialBuffer, 0));
    GPUDrivenPreviewMaterial material = materials[input.material];
    if (materialBaseColorAlpha(material, input.uv, parameters[0], push) < material.params.z) { discard; }
#endif
    TessFragment output;
    output.visibility = visibilityId;
    // Derivatives of the displaced plane avoid Slang's unsupported user
    // per-primitive fragment varyings (slang#7019). Preserve authored winding.
    output.domain = tessSurfaceDomain(input.bary, input.world, input.uv, front);
    return output;
}
