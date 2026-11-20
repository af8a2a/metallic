module GlobalDistanceFieldContinuationProbe;

import Core;
import GlobalDistanceField;
import GlobalDistanceFieldPages;

using Metallic;
using Metallic.DistanceField;

struct ContinuationProbeResult
{
    float distance;
    uint classification;
    uint hitClipmap;
    uint steps;
};

struct ContinuationProbeParams
{
    ResourceHandle<Texture3D<uint>> pageTable;
    ResourceHandle<Texture3D<float>> coarseMip;
    ResourceHandle<SamplerState> sampler;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domains;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> stepLimitDomains;
    ResourceHandle<RWStructuredBuffer<ContinuationProbeResult>> output;
};

struct ContinuationProbeRay
{
    float y;
    float minDistance;
    float maxDistance;
    bool usesStepLimitDomain;
};

static const ContinuationProbeRay kRays[] =
{
    {-6.0f, 0.0f, 14.0f, false}, // A missing fine page followed by a hit.
    {-2.0f, 0.0f, 14.0f, false}, // A missing fine page followed by a miss.
    {0.25f, 0.0f, 14.0f, false}, // A covered miss in both levels.
    {0.25f, 2.0f, 2.0f, false}, // An empty query inside the fine domain.
    {5.0f, 0.0f, 14.0f, true}, // Fine step exhaustion followed by a hit.
    {5.0f, 0.0f, 7.0f, true}, // Fine step exhaustion without a later interval.
};

ContinuationProbeResult traceProbe(ContinuationProbeParams params, ContinuationProbeRay ray,
    StructuredBuffer<GlobalDistanceFieldDomain> domains)
{
    QuantizedGlobalDistanceFieldSampler field;
    field.pageTable = resolveUniform(params.pageTable);
    field.coarseMip = resolveUniform(params.coarseMip);
    field.sampler = resolveUniform(params.sampler);
    // Constant/overflow page entries never read a physical page.
    field.pageAtlas = field.coarseMip;
    field.domain = domains[0];
    field.clipmapIndex = 0u;
    uint hitClipmap;
    // Keep each slab direction nonzero while staying in its test row.
    const float3 direction = normalize(float3(1.0f, 0.0001f, 0.0001f));
    GlobalDistanceFieldTrace trace = traceGlobalDistanceFieldClipmaps(field, domains, 2u,
        float3(0.0f, ray.y, 0.0f), direction, ray.minDistance, ray.maxDistance, hitClipmap);
    return {trace.hitDistance, trace.classification, hitClipmap, trace.stepCount};
}

[shader("compute")]
[numthreads(6, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    ContinuationProbeParams params = getResourceParameters<ContinuationProbeParams>();
    ContinuationProbeRay ray = kRays[id.x];
    RWStructuredBuffer<ContinuationProbeResult> output = resolveUniform(params.output);
    if (ray.usesStepLimitDomain) {
        output[id.x] = traceProbe(params, ray, resolveUniform(params.stepLimitDomains));
    } else {
        output[id.x] = traceProbe(params, ray, resolveUniform(params.domains));
    }
}
