#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

#include "../../../Shaders/Libraries/Lighting/ClusterLightGridCommon.slang"

#define gParams METALLIC_RESOURCE(StructuredBuffer<ClusterLightGridParams>, 0)
#define gLights METALLIC_RESOURCE(StructuredBuffer<ClusterLightData>, 1)
#define gCandidates METALLIC_RESOURCE(StructuredBuffer<uint>, 2)
#define gCells METALLIC_RESOURCE(StructuredBuffer<ClusterLightGridCell>, 3)
#define gLightIndices METALLIC_RESOURCE(StructuredBuffer<uint>, 4)
#define gOutput METALLIC_RESOURCE(RWStructuredBuffer<uint4>, 5)

[shader("compute")]
[numthreads(1, 1, 1)]
void clusterLightGridLookupProbeMain()
{
    ClusterLightGridParams params = gParams[0];
    for (uint query = 0u; query < 8u; ++query) {
        uint2 pixel = uint2(0u);
        float depth = params.eyeNear.w;
        if (query == 1u) { depth = params.rightFar.w; }
        if (query == 2u) {
            depth = params.viewport.w != 0u
                ? (params.eyeNear.w + params.rightFar.w) * 0.5
                : sqrt(params.eyeNear.w * params.rightFar.w);
        }
        if (query == 3u) { depth = params.eyeNear.w * 0.5; }
        if (query == 4u) { depth = params.rightFar.w * 2.0; }
        if (query == 5u) { pixel.x = params.viewport.x; }
        if (query == 6u) { pixel.y = params.viewport.y; }
        if (query == 7u) { depth = asfloat(0x7fc00000u); }
        ClusterLightGridLookup lookup = clusterLightGridLookup(params, gCells, pixel, depth);
        uint mask = 0u;
        for (uint light = 0u; light < lookup.count; ++light) {
            uint source = clusterLightGridLocalLightIndex(lookup, gLightIndices, gCandidates, light);
            mask |= source < 32u ? (1u << source) : 0u;
        }
        gOutput[query] = uint4(lookup.fallback, lookup.count, mask,
            clusterLightGridLocalLightIndex(lookup, gLightIndices, gCandidates, lookup.count));
    }
    uint globalCount = clusterLightGridGlobalLightCount(params);
    uint globalMask = 0u;
    float globalIntensity = 0.0;
    for (uint light = 0u; light < globalCount; ++light) {
        uint source = clusterLightGridGlobalLightIndex(params, gCandidates, light);
        globalMask |= source < 32u ? (1u << source) : 0u;
        globalIntensity += gLights[source].colorIntensity.w;
    }
    gOutput[8] = uint4(globalCount, globalMask,
        clusterLightGridGlobalLightIndex(params, gCandidates, globalCount), uint(round(globalIntensity)));
}
