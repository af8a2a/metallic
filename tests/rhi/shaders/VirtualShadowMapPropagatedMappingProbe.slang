#language slang 2026
implementing VirtualShadowMaps;

struct PropagatedMappingProbePush
{
    uint pages, projections, statistics, atlas;
    uint positions, output, count, axis;
    uint levels, pageSize, physicalPagesX, reserved;
};
[[vk::push_constant]] ConstantBuffer<PropagatedMappingProbePush> mappingProbe;

ShadowPush propagatedMappingProbePush()
{
    ShadowPush push = {};
    push.pages = mappingProbe.pages; push.projections = mappingProbe.projections;
    push.statistics = mappingProbe.statistics; push.atlas = mappingProbe.atlas;
    push.virtualPagesPerAxis = mappingProbe.axis; push.clipmapCount = mappingProbe.levels;
    push.totalPages = mappingProbe.levels * mappingProbe.axis * mappingProbe.axis + 1u;
    push.pageSize = mappingProbe.pageSize; push.physicalPagesX = mappingProbe.physicalPagesX;
    push.lightType = mappingProbe.reserved;
    push.enabled = 1u;
    return push;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void propagatedMappingBuildProbeMain(uint3 id : SV_DispatchThreadID)
{
    vsmBuildPropagatedMapping(id, propagatedMappingProbePush());
}

[shader("compute")]
[numthreads(64, 1, 1)]
void propagatedMappingLookupProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= mappingProbe.count) { return; }
    StructuredBuffer<float4> positions = resolveUniform(ResourceHandle<StructuredBuffer<float4>>(mappingProbe.positions));
    RWStructuredBuffer<uint4> output = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(mappingProbe.output));
    ShadowPush push = propagatedMappingProbePush();
    float3 position = positions[id.x].xyz;
    uint3 counts = 0u;
    ShadowLookup sample = vsmFilterLookup(position, push, 0u, counts);
    output[id.x * 8u] = uint4(sample.valid ? 1u : 0u, sample.level, asuint(sample.storedDepth), counts.x);
    output[id.x * 8u + 1u] = uint4(sample.valid ? sample.virtualTexel : uint2(0u), counts.yz);
    output[id.x * 8u + 2u] = uint4(sample.valid ? asuint(sample.projected) : uint3(0u), 0u);
    uint4 mapping = 0u;
    uint4 sourcePosition = 0u, sourceAddress = 0u, targetPosition = 0u, targetAddress = 0u;
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(mappingProbe.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(mappingProbe.pages));
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(mappingProbe.statistics));
    ShadowProjection source = projections[0];
    float3 projected = vsmProjectPosition(position, source);
    if (statistics[5] == 0u && vsmContains(projected, source, push.pageSize, true)) {
        uint2 texel = vsmVirtualTexel(projected, source, push.pageSize);
        uint page = vsmVirtualPage(texel, source, push.pageSize);
        uint4 metadata = pages[push.totalPages + page];
        sourcePosition = uint4(asuint(projected), 0u);
        sourceAddress = uint4(texel, page, metadata.z);
        uint physical, target;
        bool direct = vsmLookupPropagated(position, float2(source.rowX.w, source.rowY.w), page,
            projected, texel, projections, pages,
            push, 0u, physical, target);
        mapping = uint4(direct ? 1u : 0u, metadata.x, metadata.y, metadata.w);
        targetPosition = uint4(asuint(projected), target);
        targetAddress = uint4(texel, physical, 0u);
    }
    output[id.x * 8u + 3u] = mapping;
    output[id.x * 8u + 4u] = sourcePosition;
    output[id.x * 8u + 5u] = sourceAddress;
    output[id.x * 8u + 6u] = targetPosition;
    output[id.x * 8u + 7u] = targetAddress;
}

#if VSM_PROJECTED_RAY_PROBE
struct ProjectedRayProbeInput
{
    float4 position;
    float4 endpoint; // XYZ ray endpoint, W independent linear-depth query.
    float4 dither;
    uint4 control; // receiver level, ray, phase, generate a real SMRT ray
};

[shader("compute")]
[numthreads(64, 1, 1)]
void projectedRayStateProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= mappingProbe.count) { return; }
    StructuredBuffer<ProjectedRayProbeInput> inputs = resolveUniform(ResourceHandle<StructuredBuffer<ProjectedRayProbeInput>>(mappingProbe.positions));
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(mappingProbe.projections));
    RWStructuredBuffer<uint4> output = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(mappingProbe.output));
    ProjectedRayProbeInput input = inputs[id.x];
    ShadowPush push = propagatedMappingProbePush();
    push.rayLength = 3.0;
    push.angularRadius = 0.08;
    push.rayCount = 4u;
    push.samplesPerRay = 8u;
    push.samplePhase = input.control.z;
    uint2 pixel = uint2(id.x * 37u + 3u, input.control.z * 11u + 5u);
    float3 position = input.position.xyz;
    float3 endpoint = input.endpoint.xyz, dither = input.dither.xyz;
    if (input.control.w != 0u) {
        endpoint = vsmRayEndpoint(position, projections[0], push, input.control.y, pixel);
        dither = vsmRayDither(position, projections[input.control.x], push, input.control.y, pixel);
    }
    VSMDirectionalRayState ray = vsmCreateDirectionalRayState(position, endpoint, dither,
        projections, push, input.control.x);
    float3 toLight = normalize(projections[0].rowZ.xyz);
    float3 right = normalize(cross(abs(toLight.y) < 0.95 ? float3(0, 1, 0) : float3(1, 0, 0), toLight));
    float3 up = cross(toLight, right);
    float texelSize = projections[input.control.x].metric.y;
    float coneRadius = push.rayLength * sin(push.angularRadius);
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    Texture2D<float> atlas = resolveUniform(ResourceHandle<Texture2D<float>>(push.atlas));
    for (uint sampleIndex = 0u; sampleIndex < 11u; ++sampleIndex) {
        float time = sampleIndex == 0u ? 0.0 : sampleIndex == 1u ? 1.0 :
            vsmRaySampleTime(sampleIndex - 2u, push, pixel);
        uint3 counts = 0u;
        ShadowLookup sample = vsmDirectionalRayLookup(time, ray, projections, pages, atlas, push, counts);
        float3 baseLookup = vsmDirectionalRayPosition(ray, time, input.control.x, projections[input.control.x], true);
        float3 baseCompare = vsmDirectionalRayPosition(ray, time, input.control.x, projections[input.control.x], false);
        float3 comparison = vsmDirectionalRayPosition(ray, time, sample.level, projections[sample.level], false);
        uint firstLevel = push.clipmapCount, sourcePage = 0xffffffffu;
        float3 firstPosition = 0.0;
        uint checked = 0u, missing = 0u;
        for (uint level = input.control.x; level < push.clipmapCount; ++level) {
            ShadowProjection projection = projections[level];
            float3 projected = vsmDirectionalRayPosition(ray, time, level, projection, true);
            if (!vsmContains(projected, projection, push.pageSize, true)) { continue; }
            uint2 texel = vsmVirtualTexel(projected, projection, push.pageSize);
            if (firstLevel == push.clipmapCount) {
                firstLevel = level; firstPosition = projected;
                sourcePage = vsmVirtualPage(texel, projection, push.pageSize);
            }
            if (input.control.w == 0u) { continue; }
            ++checked;
            float2 x = vsmDirectionalFootprintRange(projection.rowX, position, toLight, right, up,
                push.rayLength, coneRadius, texelSize);
            float2 y = vsmDirectionalFootprintRange(projection.rowY, position, toLight, right, up,
                push.rayLength, coneRadius, texelSize);
            float resolution = float(projection.page.y * push.pageSize);
            uint2 lower = uint2(clamp((clamp(float2(x.x, y.x), -1.0, 1.0) * 0.5 + 0.5) * resolution - 1.0,
                0.0, resolution - 1.0));
            uint2 upper = uint2(clamp((clamp(float2(x.y, y.y), -1.0, 1.0) * 0.5 + 0.5) * resolution + 1.0,
                0.0, resolution - 1.0));
            float2 depth = vsmDirectionalFootprintRange(projection.rowZ, position, toLight, right, up,
                push.rayLength, coneRadius, texelSize);
            missing += (any(texel < lower) || any(texel > upper) || depth.y < 0.0 || depth.x > 1.0) ? 1u : 0u;
        }
        uint offset = (id.x * 11u + sampleIndex) * 10u;
        output[offset] = uint4(sample.valid ? 1u : 0u, sample.level, sample.valid ? sample.virtualTexel : uint2(0u));
        output[offset + 1u] = uint4(asuint(sample.projected), asuint(sample.storedDepth));
        output[offset + 2u] = uint4(asuint(comparison), asuint(vsmDirectionalRayDepth(input.endpoint.w, ray)));
        output[offset + 3u] = uint4(asuint(endpoint), asuint(time));
        output[offset + 4u] = uint4(asuint(dither), asuint(ray.receiverDepth));
        output[offset + 5u] = uint4(asuint(baseLookup), asuint(ray.depthScale));
        output[offset + 6u] = uint4(asuint(baseCompare), asuint(input.endpoint.w));
        output[offset + 7u] = uint4(checked, missing, counts.xy);
        output[offset + 8u] = uint4(counts.z, sourcePage, input.control.x, 0u);
        output[offset + 9u] = uint4(asuint(firstPosition), firstLevel);
    }
}
#endif
