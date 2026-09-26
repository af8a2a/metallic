// Included after VisibilityBuffer.slang's ordinary cluster accessors.
#include "VisibilityTessellationSplit.slang"
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
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = ph[0];
    DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(params.tessellationBuffer, 0));
    uint draw = hybridFlatGroup(group);
    uint candidate = draw / kTessTasksPerCluster;
    uint first = (draw % kTessTasksPerCluster) * kTessRootsPerTask;
    uint record = 0xffffffffu;
    if (lane == 0u) {
        bool visible;
        if (push.environmentSHBuffer != 0xffffffffu) {
            DescriptorHandle<StructuredBuffer<uint>> bins = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.environmentSHBuffer, 0));
            visible = hybridBinRecord(bins, push.mipLevel, candidate, record);
        } else {
            record = residentLodRecordIndex(candidate, params, push);
            visible = residentClusterVisible(record, push);
            if (visible) { visible = visibleClusterDrawBucket(residentRasterCluster(record, push).record) == push.mipLevel; }
        }
        g_PrebinVisible = visible ? 1u : 0u;
        if (visible) { g_PrebinCluster = residentRasterCluster(record, push); }
        g_TessPatches.record = record; g_TessPatches.first = first;
    }
    GroupMemoryBarrierWithGroupSync();
    uint count = 0u;
    if (g_PrebinVisible != 0u) {
        ResidentRasterCluster c = g_PrebinCluster;
        bool enabled = tessMaterial(data, c.instance.identity.y).enabled != 0u;
        if (lane == 0u) {
            g_TessPatches.enabled = enabled ? 1u : 0u;
        }
        if (!enabled) { count = first == 0u && lane == 0u ? 1u : 0u; }
        else if (lane < kTessRootsPerTask && first + lane < c.meshlet.ranges.w) {
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> vb = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.positionBuffer, 0));
            DescriptorHandle<StructuredBuffer<uint>> vi = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.meshletVertexBuffer, 0));
            uint3 tri = residentClusterTriangle(c, first + lane, push);
            float3 source[3];
            for (uint i = 0u; i < 3u; ++i) {
                source[i] = transformPoint(instanceTransform(c.instance), vb[c.geometry.payload.x + vi[c.meshlet.ranges.x + tri[i]]].position.xyz);
            }
            TessView view;
            view.eye = params.renderEye.xyz;
            view.forward = normalize(params.renderCenter.xyz - params.renderEye.xyz);
            view.orthographic = params.renderUpProjection.w > 0.5;
            view.pixelScale = float(params.height) / (view.orthographic ? params.renderClipOrtho.z : 2.0 * tan(params.renderViewport.w * 0.5));
            view.nearPlane = params.renderClipOrtho.x;
            count = tessBuildLeaves(lane, source, view, push.tessellationEdgePixels, push.tessellationMaxFactor, push.tessellationMaxSplitDepth);
        }
    }
    tessDispatchLeaves(lane, count);
}

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void visibilityTessellationMesh(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    in payload TessPatchPayload payload, out vertices TessVertex vertices[128],
    out indices uint3 triangles[128], out primitives TessPrimitive primitives[128],
    uniform GPUDrivenPreviewPush push)
{
    uint source = tessPayloadRoot(payload, group.x);
    uint record = payload.record;
    uint originalTriangle = payload.first + source;
    uint leaf = group.x - (source == 0u ? 0u : payload.ends[source - 1u]);
    uint patchBase = (source * kTessLeavesPerRoot + leaf) * 3u;
    uint3 patch = uint3(payload.patches[patchBase], payload.patches[patchBase + 1u], payload.patches[patchBase + 2u]);
    ResidentRasterCluster c = residentRasterCluster(record, push);
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> ph =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.paramsBuffer, 0));
    GPUDrivenPreviewParams params = ph[0];
    DescriptorHandle<StructuredBuffer<uint>> data = DescriptorHandle<StructuredBuffer<uint>>(uint2(params.tessellationBuffer, 0));
    TessMaterial material = tessMaterial(data, c.instance.identity.y);
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>> vb = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>(uint2(push.positionBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> vi = DescriptorHandle<StructuredBuffer<uint>>(uint2(push.meshletVertexBuffer, 0));
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
            vertices[v].originalVisibilityId = 0u;
            vertices[v].debugPatchSeed = tessDebugHash(c.record.clusterIndex + 1u);
        }
        for (uint t = lane; t < c.meshlet.ranges.w; t += 64u) {
            triangles[t] = residentRasterWinding(residentClusterTriangle(c, t, push), instanceTransform(c.instance));
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
        TessView view;
        view.eye = params.renderEye.xyz;
        view.forward = normalize(params.renderCenter.xyz - params.renderEye.xyz);
        view.orthographic = params.renderUpProjection.w > 0.5;
        view.pixelScale = float(params.height) / (view.orthographic ? params.renderClipOrtho.z : 2.0 * tan(params.renderViewport.w * 0.5));
        view.nearPlane = params.renderClipOrtho.x;
        g_TessRates = tessPatchRates(patch, g_TessSourcePosition, view, push.tessellationEdgePixels, push.tessellationMaxFactor);
        g_TessPattern = tessPattern(data, g_TessRates);
    }
    GroupMemoryBarrierWithGroupSync();
    SetMeshOutputCounts(g_TessPattern.y, g_TessPattern.w);
    if (lane < g_TessPattern.y) {
        float3 bary = tessPatchDomain(tessBary(data, g_TessPattern, lane), g_TessRates, patch, g_TessSourcePosition);
        float3 p, n; float2 uv;
        tessSourceDomain(bary, g_TessSourcePosition, g_TessSourceNormal, g_TessSourceUv, p, n, uv);
        p += n * tessHeight(material, uv);
        float4 clip = projectPosition(p, camera); clip.y = -clip.y;
        if (push.projectWithCullingCamera == 0u) { clip.xy -= 2.0 * float2(params.renderEye.w, params.renderCenter.w) / float2(params.width, params.height) * clip.w; }
        vertices[lane].position = clip;
        vertices[lane].uv = uv;
        vertices[lane].bary = bary.yz;
#if VISIBILITY_BUFFER_ALPHA_MASKED
        vertices[lane].material = c.instance.identity.y;
#endif
        vertices[lane].world = p;
        vertices[lane].originalVisibilityId = packVisibilityId(record, originalTriangle);
        vertices[lane].debugPatchSeed = tessDebugPatchSeed(tessDebugHash(c.record.clusterIndex + 1u), originalTriangle, patch);
    }
    bool reflected = dot(cross(transform.world0.xyz, transform.world1.xyz), transform.world2.xyz) < 0.0;
    for (uint t = lane; t < g_TessPattern.w; t += 64u) {
        uint3 tri = tessTriangle(data, g_TessPattern, t);
        triangles[t] = reflected ? tri.xzy : tri;
        primitives[t].visibilityId = t;
    }
}

uint tessResidentVisibility(TessVertex input, uint visibilityId, GPUDrivenPreviewPush push)
{
#if VISIBILITY_BUFFER_ALPHA_MASKED
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> parameters = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(push.paramsBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>> materials = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(uint2(push.materialBuffer, 0));
    GPUDrivenPreviewMaterial material = materials[input.material];
    if (materialBaseColorAlpha(material, input.uv, parameters[0], push) < material.params.z) { discard; }
#endif
    return input.originalVisibilityId != 0u ? input.originalVisibilityId : visibilityId;
}

[shader("fragment")]
TessFragment visibilityTessellationFragment(TessVertex input, uint visibilityId : SV_PrimitiveID, bool front : SV_IsFrontFace,
    uniform GPUDrivenPreviewPush push)
{
    TessFragment output;
    output.visibility = tessResidentVisibility(input, visibilityId, push);
    output.debugColor = tessDebugColor(input, visibilityId);
    // Derivatives of the displaced plane avoid Slang's unsupported user
    // per-primitive fragment varyings (slang#7019). Preserve authored winding.
    output.domain = tessSurfaceDomain(input.bary, input.world, input.uv, front);
    return output;
}

// Frozen-camera raster only contributes to its HZB. Retain the exact same
// masked coverage and source ID, without exporting viewport-only MRTs.
[shader("fragment")]
uint visibilityTessellationCullingFragment(TessVertex input, uint visibilityId : SV_PrimitiveID,
    bool front : SV_IsFrontFace, uniform GPUDrivenPreviewPush push) : SV_Target0
{
    return tessResidentVisibility(input, visibilityId, push);
}
