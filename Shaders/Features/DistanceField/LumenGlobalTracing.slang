module LumenGlobalTracing;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
import GlobalDistanceFieldObjectGrid;
import LumenSurfaceCache;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

struct LumenGlobalTraceRay
{
    float4 originAndMinDistance;
    float4 directionAndMaxDistance;
};

struct LumenGlobalTraceResult
{
    float4 metrics; // Hit distance, surface expansion, card weight sum, radiance factor.
    float4 normal; // World normal and dominant card weight.
    float4 radiance;
    uint4 state; // Geometry classification, clipmap, steps, cache classification.
    uint4 candidates;
    uint4 card; // Dominant card, its scene object and bit-cast atlas UV.
};

static const uint kCacheSuppressed = 1u;
static const uint kCacheUncovered = 2u;
static const uint kCacheSampled = 3u;
static const uint kCacheSceneMismatch = 4u;
typealias TraceRadianceTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

struct LumenGlobalTraceParams
{
    ResourceHandle<Texture3D<uint>> pageTable;
    ResourceHandle<Texture3D<float>> pageAtlas;
    ResourceHandle<Texture3D<float>> coarseMip;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domains;
    ResourceHandle<StructuredBuffer<uint4>> objectGrid;
    ResourceHandle<StructuredBuffer<uint>> objectGridScene;
    ResourceHandle<StructuredBuffer<SurfaceCacheCard>> cards;
    ResourceHandle<StructuredBuffer<uint>> cardScene;
    ResourceHandle<Texture2D<float>> cardDepth;
    ResourceHandle<Texture2D<float4>> cardLighting;
    ResourceHandle<StructuredBuffer<LumenGlobalTraceRay>> rays;
    ResourceHandle<RWStructuredBuffer<LumenGlobalTraceResult>> results;
    ResourceHandle<TraceRadianceTexture> radiance;
    SamplerHandle sampler;
    uint rayCount;
    uint clipmapCount;
};

// UE 5.7.4 GlobalDistanceFieldShared.ush:277. Differentiate the encoded fine atlas,
// independently of the coarse mip used for ray marching. The Metallic lattice has
// fixed bounds rather than a toroidally scrolled page table.
float sampleGlobalDistanceFieldNormalPage(QuantizedGlobalDistanceFieldSampler field, float3 worldPosition)
{
    float3 volumePosition = worldPosition - field.domain.minAndVoxelSize.xyz;
    float3 pagePosition;
    uint pageIndex = loadGlobalDistanceFieldPage(field.pageTable,
        volumePosition / field.domain.minAndVoxelSize.w, pagePosition, field.clipmapIndex);
    return sampleGlobalDistanceFieldEncodedPage(field.pageAtlas, field.sampler, pageIndex, pagePosition);
}

float3 computeLumenGlobalDistanceFieldNormal(QuantizedGlobalDistanceFieldSampler field,
    float3 worldPosition, float3 fallbackNormal)
{
    float texelOffset = 0.5f * field.domain.minAndVoxelSize.w;
    float3 x = float3(texelOffset, 0.0f, 0.0f);
    float3 y = float3(0.0f, texelOffset, 0.0f);
    float3 z = float3(0.0f, 0.0f, texelOffset);
    float right = sampleGlobalDistanceFieldNormalPage(field, worldPosition + x);
    float left = sampleGlobalDistanceFieldNormalPage(field, worldPosition - x);
    float front = sampleGlobalDistanceFieldNormalPage(field, worldPosition + y);
    float back = sampleGlobalDistanceFieldNormalPage(field, worldPosition - y);
    float up = sampleGlobalDistanceFieldNormalPage(field, worldPosition + z);
    float down = sampleGlobalDistanceFieldNormalPage(field, worldPosition - z);
    float3 gradient = float3(right - left, front - back, up - down);
    float gradientLength = length(gradient);
    return gradientLength > 0.001f ? gradient / gradientLength : fallbackNormal;
}

LumenGlobalTraceResult traceLumenGlobalRay(LumenGlobalTraceParams params, LumenGlobalTraceRay ray)
{
    LumenGlobalTraceResult result = (LumenGlobalTraceResult)0;
    result.metrics.x = -1.0f;
    result.state.y = ~0u;
    result.candidates = uint4(~0u);
    result.card.xy = uint2(~0u);

    StructuredBuffer<uint> cardScene = resolveUniform(params.cardScene);
    StructuredBuffer<uint> objectGridScene = resolveUniform(params.objectGridScene);
    if (cardScene[2] != objectGridScene[0] || cardScene[3] != objectGridScene[1]) {
        result.state.w = kCacheSceneMismatch;
        return result;
    }

    StructuredBuffer<GlobalDistanceFieldDomain> domains = resolveUniform(params.domains);
    QuantizedGlobalDistanceFieldSampler field = {resolveUniform(params.pageTable), resolveUniform(params.pageAtlas),
        resolveUniform(params.coarseMip), resolveUniform(params.sampler), domains[0], 0u};
    float3 origin = ray.originAndMinDistance.xyz;
    float3 rayDirection = ray.directionAndMaxDistance.xyz;
    uint hitClipmapIndex;
    // UE 5.7.4 FConeTraceInput: unit SDF step factors, full coverage and expansion
    // from ray time. The existing quantized tracer uses this same specialization.
    GlobalDistanceFieldTrace hit = traceGlobalDistanceFieldClipmaps(field, domains, params.clipmapCount,
        origin, rayDirection, ray.originAndMinDistance.w, ray.directionAndMaxDistance.w, hitClipmapIndex);
    result.metrics.xy = float2(hit.hitDistance, hit.surfaceExpansion);
    result.state.xyz = uint3(hit.classification, hitClipmapIndex, hit.stepCount);
    if (hit.classification != kGlobalDistanceFieldHit) {
        return result;
    }

    // UE 5.7.4 LumenSoftwareRayTracing.ush:649, opaque diffuse path.
    field.domain = domains[hitClipmapIndex];
    field.clipmapIndex = hitClipmapIndex;
    const float3 sampleWorldPosition = origin + rayDirection * hit.hitDistance;
    const float3 sampleWorldNormal = computeLumenGlobalDistanceFieldNormal(field, sampleWorldPosition, -rayDirection);
    const float clipmapVoxelExtent = field.domain.minAndVoxelSize.w * 0.5f;
    const float3 gridSampleWorldPosition = sampleWorldPosition + sampleWorldNormal * clipmapVoxelExtent;
    const float radianceFactor = smoothstep(1.5f * clipmapVoxelExtent, 2.0f * clipmapVoxelExtent, hit.hitDistance);
    result.normal.xyz = sampleWorldNormal;
    result.metrics.w = radianceFactor;
    if (radianceFactor <= 0.0f) {
        result.state.w = kCacheSuppressed;
        return result;
    }

    LumenSurfaceCache cache;
    cache.cards = resolveUniform(params.cards);
    cache.depth = resolveUniform(params.cardDepth);
    cache.lighting = resolveUniform(params.cardLighting);
    cache.sampler = field.sampler;
    cache.cardCount = cardScene[0];
    cache.tileResolution = cardScene[1];
    result.candidates = loadGlobalDistanceFieldObjectGridCell(field.pageTable, resolveUniform(params.objectGrid),
        field.domain, hitClipmapIndex, gridSampleWorldPosition);
    LumenCardSampleAccumulator accumulator = initLumenCardSampleAccumulator();
    const float surfaceCacheBias = kGlobalDistanceFieldObjectGridCardRangeInVoxels * clipmapVoxelExtent;
    for (uint candidate = 0; candidate < 4u; ++candidate) {
        uint objectId = result.candidates[candidate];
        if (objectId == kGlobalDistanceFieldInvalidObjectGridEntry) {
            break;
        }
        sampleLumenMeshCards(cache, objectId, sampleWorldPosition, sampleWorldNormal,
            surfaceCacheBias, accumulator);
        if (accumulator.sampleWeightSum >= 0.9f) {
            break;
        }
    }
    result.metrics.z = accumulator.sampleWeightSum;
    result.radiance.xyz = radianceFactor * evaluateLumenCardRadiance(accumulator);
    result.state.w = kCacheUncovered;
    if (accumulator.cardIndex != ~0u) {
        result.state.w = kCacheSampled;
        result.normal.w = accumulator.maxSampleWeight;
        result.card.xy = uint2(accumulator.cardIndex, cache.cards[accumulator.cardIndex].metadata.x);
        result.card.zw = asuint(accumulator.physicalAtlasUV);
    }
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenGlobalTraceMain(uint3 thread : SV_DispatchThreadID)
{
    LumenGlobalTraceParams params = getParameters<LumenGlobalTraceParams>();
    if (thread.x >= params.rayCount) {
        return;
    }
    StructuredBuffer<LumenGlobalTraceRay> rays = resolveUniform(params.rays);
    LumenGlobalTraceResult result = traceLumenGlobalRay(params, rays[thread.x]);
    RWStructuredBuffer<LumenGlobalTraceResult> results = resolveUniform(params.results);
    results[thread.x] = result;
    TraceRadianceTexture radiance = resolveUniform(params.radiance);
    radiance[thread.xy] = float4(result.radiance.xyz, 1.0f);
}
