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
#if VSM_PROPAGATED_MAPPING
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
#endif
    output[id.x * 8u + 3u] = mapping;
    output[id.x * 8u + 4u] = sourcePosition;
    output[id.x * 8u + 5u] = sourceAddress;
    output[id.x * 8u + 6u] = targetPosition;
    output[id.x * 8u + 7u] = targetAddress;
}
