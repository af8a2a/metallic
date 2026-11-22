module LumenGlobalRay;

import ShaderCore;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
import GlobalDistanceFieldObjectGrid;
import LumenSurfaceCache;

using Metallic.DistanceField;

namespace Metallic.Lumen
{
public struct LumenGlobalTraceRay
{
    public float4 originAndMinDistance;
    public float4 directionAndMaxDistance;
};

public struct LumenGlobalTraceResult
{
    public float4 metrics; // Hit distance, surface expansion, card weight sum, radiance factor.
    public float4 normal; // World normal and dominant card weight.
    public float4 radiance;
    public uint4 state; // Geometry classification, clipmap, steps, cache classification.
    public uint4 candidates;
    public uint4 card; // Dominant card, its scene object and bit-cast atlas UV.
};

static const uint kCacheSuppressed = 1u;
static const uint kCacheUncovered = 2u;
static const uint kCacheSampled = 3u;
static const uint kCacheSceneMismatch = 4u;
public struct LumenGlobalTraceBindings
{
    public ResourceHandle<Texture3D<uint>> pageTable;
    public ResourceHandle<Texture3D<float>> pageAtlas;
    public ResourceHandle<Texture3D<float>> coarseMip;
    public ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domains;
    public ResourceHandle<StructuredBuffer<uint4>> objectGrid;
    public ResourceHandle<StructuredBuffer<uint>> objectGridScene;
    public ResourceHandle<StructuredBuffer<SurfaceCacheCard>> cards;
    public ResourceHandle<StructuredBuffer<uint>> cardScene;
    public ResourceHandle<Texture2D<float>> cardDepth;
    public ResourceHandle<Texture2D<float4>> cardLighting;
    public SamplerHandle sampler;
};

public struct LumenGlobalTraceScene
{
    public QuantizedGlobalDistanceFieldSampler field;
    public StructuredBuffer<GlobalDistanceFieldDomain> domains;
    public StructuredBuffer<uint4> objectGrid;
    public LumenSurfaceCache cache;
    public uint clipmapCount;
    public bool identityMatches;
};

public LumenGlobalTraceScene resolveLumenGlobalTraceScene(LumenGlobalTraceBindings bindings, uint clipmapCount)
{
    LumenGlobalTraceScene scene;
    StructuredBuffer<uint> cardScene = resolveUniform(bindings.cardScene);
    StructuredBuffer<uint> objectGridScene = resolveUniform(bindings.objectGridScene);
    scene.identityMatches = cardScene[2] == objectGridScene[0] && cardScene[3] == objectGridScene[1];
    scene.domains = resolveUniform(bindings.domains);
    scene.field = {resolveUniform(bindings.pageTable), resolveUniform(bindings.pageAtlas),
        resolveUniform(bindings.coarseMip), resolveUniform(bindings.sampler), scene.domains[0], 0u};
    scene.objectGrid = resolveUniform(bindings.objectGrid);
    scene.cache = {resolveUniform(bindings.cards), resolveUniform(bindings.cardDepth),
        resolveUniform(bindings.cardLighting), scene.field.sampler, cardScene[0], cardScene[1]};
    scene.clipmapCount = clipmapCount;
    return scene;
}

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

public LumenGlobalTraceResult traceLumenGlobalRay(LumenGlobalTraceScene scene, LumenGlobalTraceRay ray,
    bool rejectRecedingExpansion, out bool recedingExpansionRejected)
{
    LumenGlobalTraceResult result = (LumenGlobalTraceResult)0;
    result.metrics.x = -1.0f;
    result.state.y = ~0u;
    result.candidates = uint4(~0u);
    result.card.xy = uint2(~0u);
    recedingExpansionRejected = false;

    if (!scene.identityMatches) {
        result.state.w = kCacheSceneMismatch;
        return result;
    }

    QuantizedGlobalDistanceFieldSampler field = scene.field;
    float3 origin = ray.originAndMinDistance.xyz;
    float3 rayDirection = ray.directionAndMaxDistance.xyz;
    uint hitClipmapIndex;
    // UE 5.7.4 FConeTraceInput: unit SDF step factors, full coverage and expansion
    // from ray time. The existing quantized tracer uses this same specialization.
    GlobalDistanceFieldTrace hit = traceGlobalDistanceFieldClipmaps(field, scene.domains, scene.clipmapCount,
        origin, rayDirection, ray.originAndMinDistance.w, ray.directionAndMaxDistance.w, hitClipmapIndex,
        0.0f, 0.0f, rejectRecedingExpansion);
    recedingExpansionRejected = hit.recedingExpansionRejected;
    result.metrics.xy = float2(hit.hitDistance, hit.surfaceExpansion);
    result.state.xyz = uint3(hit.classification, hitClipmapIndex, hit.stepCount);
    if (hit.classification != kGlobalDistanceFieldHit) {
        return result;
    }

    // UE 5.7.4 LumenSoftwareRayTracing.ush:649, opaque diffuse path.
    field.domain = scene.domains[hitClipmapIndex];
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

    result.candidates = loadGlobalDistanceFieldObjectGridCell(field.pageTable, scene.objectGrid,
        field.domain, hitClipmapIndex, gridSampleWorldPosition);
    LumenCardSampleAccumulator accumulator = initLumenCardSampleAccumulator();
    const float surfaceCacheBias = kGlobalDistanceFieldObjectGridCardRangeInVoxels * clipmapVoxelExtent;
    for (uint candidate = 0; candidate < 4u; ++candidate) {
        uint objectId = result.candidates[candidate];
        if (objectId == kGlobalDistanceFieldInvalidObjectGridEntry) {
            break;
        }
        sampleLumenMeshCards(scene.cache, objectId, sampleWorldPosition, sampleWorldNormal,
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
        result.card.xy = uint2(accumulator.cardIndex, scene.cache.cards[accumulator.cardIndex].metadata.x);
        result.card.zw = asuint(accumulator.physicalAtlasUV);
    }
    return result;
}

public LumenGlobalTraceResult traceLumenGlobalRay(LumenGlobalTraceScene scene, LumenGlobalTraceRay ray)
{
    bool recedingExpansionRejected;
    return traceLumenGlobalRay(scene, ray, false, recedingExpansionRejected);
}

} // namespace Metallic.Lumen
