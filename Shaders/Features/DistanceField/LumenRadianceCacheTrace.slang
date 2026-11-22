module LumenRadianceCacheTrace;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import LumenGlobalRay;
import LumenProbeMath;
import LumenRadianceCache;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

typealias RadianceCacheImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

struct LumenRadianceCacheTraceParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<StructuredBuffer<LumenRadianceCacheClipmap>> clipmaps;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<StructuredBuffer<float4>> traceData;
    ResourceHandle<RWStructuredBuffer<uint>> updateState;
    ResourceHandle<RWStructuredBuffer<uint2>> traceTiles;
    ResourceHandle<RWStructuredBuffer<uint>> indirectArgs;
    ResourceHandle<RWStructuredBuffer<uint2>> rawAtlas;
    ResourceHandle<RWStructuredBuffer<uint2>> filteredAtlas;
    ResourceHandle<RadianceCacheImage> radianceAtlas;
    uint gdfClipmapCount;
    float3 cameraPosition;
    float maxTraceDistance;
};

struct RadianceCacheTraceData
{
    float3 worldCenter;
    uint clipmapIndex;
    uint probeIndex;
    bool forceDownsample;
};

// UE 5.7.4 LumenRadianceCacheTracingCommon.ush, GetProbeTraceData.
RadianceCacheTraceData unpackProbeTraceData(float4 packed)
{
    uint word = asuint(packed.w);
    return {packed.xyz, (word >> 24u) & 0x7fu, word & 0xffffffu, (word >> 31u) != 0u};
}

static const uint kTraceTileSize = 8u;
static const uint kTraceTileGroupStride = 128u;
static const uint kTraceTileCountWord = 29u;
static const uint kTraceDispatchOffset = 9u;
static const float kMinTraceDistance = 0.05f;
static const float kDownsampleDistanceSquared = 40.0f * 40.0f;
static const float kSpatialFilterMaxRadianceHitAngle = 0.2f;
static const float kMaxHalfFloat = 65504.0f;

groupshared uint globalTraceTileOffset;

// UE 5.7.4 LumenRadianceCache.usf, GenerateProbeTraceTilesCS (FORCE_UNIFORM_TRACES).
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadianceCacheGenerateTilesMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    LumenRadianceCacheTraceParams params = getParameters<LumenRadianceCacheTraceParams>();
    RadianceCacheTraceData trace = unpackProbeTraceData(resolveUniform(params.traceData)[group.z]);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[trace.clipmapIndex];
    float3 toCamera = params.cameraPosition - trace.worldCenter;
    float distanceFromCameraSq = dot(toCamera, toCamera);
    uint uniformLevel = 1u;
    if (distanceFromCameraSq >= kDownsampleDistanceSquared) {
        uniformLevel = 0u;
    } else if (distanceFromCameraSq < clipmap.traceSettings.w) {
        uniformLevel = 2u;
    }
    if (trace.forceDownsample) {
        uniformLevel = 0u;
    }

    uint tileResolution = (clipmap.layout.z / kTraceTileSize / 2u) << uniformLevel;
    if (tileResolution == 0u) {
        tileResolution = 1u;
        uniformLevel = 1u;
    }
    uint numTraceTiles = tileResolution * tileResolution;
    if (lane == 0u) {
        RWStructuredBuffer<uint> updateState = resolveUniform(params.updateState);
        InterlockedAdd(updateState[kTraceTileCountWord], numTraceTiles, globalTraceTileOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    RWStructuredBuffer<uint2> traceTiles = resolveUniform(params.traceTiles);
    for (uint tileIndex = lane; tileIndex < numTraceTiles; tileIndex += kTraceTileSize * kTraceTileSize) {
        uint2 tile = uint2(tileIndex % tileResolution, tileIndex / tileResolution);
        uint packedTile = (tile.x & 0xffu) | ((tile.y & 0xffu) << 8u) | ((uniformLevel & 0xffu) << 16u);
        traceTiles[globalTraceTileOffset + tileIndex] = uint2(packedTile, group.z);
    }
}

// UE 5.7.4 LumenRadianceCache.usf, SetupTraceFromProbesCS (software dispatch).
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenRadianceCacheTraceArgsMain()
{
    LumenRadianceCacheTraceParams params = getParameters<LumenRadianceCacheTraceParams>();
    uint tileCount = resolveUniform(params.updateState)[kTraceTileCountWord];
    RWStructuredBuffer<uint> indirectArgs = resolveUniform(params.indirectArgs);
    indirectArgs[kTraceDispatchOffset + 0u] = kTraceTileGroupStride;
    indirectArgs[kTraceDispatchOffset + 1u] = (tileCount + kTraceTileGroupStride - 1u) / kTraceTileGroupStride;
    indirectArgs[kTraceDispatchOffset + 2u] = 1u;
}

uint rawProbeAddress(uint probeIndex, uint2 texel, uint probeResolution)
{
    return (probeIndex * probeResolution + texel.y) * probeResolution + texel.x;
}

uint2 packRadianceDepth(float3 radiance, float depth)
{
    return uint2(packHalf2x16(radiance.xy), packHalf2x16(float2(radiance.z, depth)));
}

float4 unpackRadianceDepth(uint2 packed)
{
    return float4(unpackHalf2x16ToFloat(packed.x), unpackHalf2x16ToFloat(packed.y));
}

groupshared float3 sharedTraceRadiance[8][8];
groupshared float sharedTraceHitDistance[8][8];

// UE 5.7.4 LumenRadianceCache.usf, TraceForProbeTexel and TraceFromProbesCS.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadianceCacheTraceMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenRadianceCacheTraceParams params = getParameters<LumenRadianceCacheTraceParams>();
    uint tileIndex = group.y * kTraceTileGroupStride + group.x;
    if (tileIndex >= resolveUniform(params.updateState)[kTraceTileCountWord]) {
        return;
    }

    uint2 packedTile = resolveUniform(params.traceTiles)[tileIndex];
    uint2 tile = uint2(packedTile.x & 0xffu, (packedTile.x >> 8u) & 0xffu);
    uint tileLevel = (packedTile.x >> 16u) & 0xffu;
    RadianceCacheTraceData trace = unpackProbeTraceData(resolveUniform(params.traceData)[packedTile.y]);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[trace.clipmapIndex];
    uint probeResolution = clipmap.layout.z;
    uint traceResolution = (probeResolution / 2u) << tileLevel;
    uint2 traceTexel = tile * kTraceTileSize + lane.xy;

    // The uniform tile allocator issues only complete 8x8 tiles. It uses level
    // one if a half-resolution probe would produce zero trace tiles.
    float2 probeUV = (float2(traceTexel) + 0.5f) / float(traceResolution);
    float3 direction = lumenEquiAreaSphericalMapping(probeUV);
    float minTraceDistance = max(kMinTraceDistance, clipmap.traceSettings.x);
    LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.gdfClipmapCount);
    LumenGlobalTraceRay ray = {float4(trace.worldCenter, minTraceDistance), float4(direction, params.maxTraceDistance)};
    LumenGlobalTraceResult hit = traceLumenGlobalRay(scene, ray);
    const bool opaqueHit = hit.state.x == kGlobalDistanceFieldHit;
    // LumenSoftwareRayTracing.ush, the cache's bZeroRadianceIfRayStartsInsideGeometry branch.
    sharedTraceRadiance[lane.y][lane.x] = opaqueHit && hit.metrics.x > minTraceDistance ? hit.radiance.xyz : 0.0f;
    sharedTraceHitDistance[lane.y][lane.x] = opaqueHit ? hit.metrics.x : params.maxTraceDistance;
    GroupMemoryBarrierWithGroupSync();

    RWStructuredBuffer<uint2> rawAtlas = resolveUniform(params.rawAtlas);
    if (traceResolution < probeResolution) {
        uint upsampleFactor = probeResolution / traceResolution;
        uint2 probeTexelBase = traceTexel * upsampleFactor;
        float3 radiance = sharedTraceRadiance[lane.y][lane.x];
        float hitDistance = min(sharedTraceHitDistance[lane.y][lane.x], kMaxHalfFloat);
        uint2 packedSample = packRadianceDepth(radiance, hitDistance);
        for (uint y = 0u; y < upsampleFactor; ++y) {
            for (uint x = 0u; x < upsampleFactor; ++x) {
                rawAtlas[rawProbeAddress(trace.probeIndex, probeTexelBase + uint2(x, y), probeResolution)] = packedSample;
            }
        }
    } else {
        uint downsampleFactor = traceResolution / probeResolution;
        uint writeTileSize = kTraceTileSize / downsampleFactor;
        if (all(lane.xy < writeTileSize)) {
            float3 radiance = 0.0f;
            for (uint y = 0u; y < downsampleFactor; ++y) {
                for (uint x = 0u; x < downsampleFactor; ++x) {
                    radiance += sharedTraceRadiance[lane.y * downsampleFactor + y][lane.x * downsampleFactor + x];
                }
            }
            radiance /= float(downsampleFactor * downsampleFactor);
            float hitDistance = kMaxHalfFloat;
            for (uint y = 0u; y < downsampleFactor; ++y) {
                for (uint x = 0u; x < downsampleFactor; ++x) {
                    hitDistance = min(hitDistance, sharedTraceHitDistance[lane.y * downsampleFactor + y][lane.x * downsampleFactor + x]);
                }
            }
            uint2 probeTexel = writeTileSize * tile + lane.xy;
            rawAtlas[rawProbeAddress(trace.probeIndex, probeTexel, probeResolution)] = packRadianceDepth(radiance, hitDistance);
        }
    }
}

uint2 directionTexel(float3 direction, uint probeResolution)
{
    // The signed axes can map exactly to a face endpoint. Keep point sampling
    // within this physical probe rather than the following packed slot.
    return min(uint2(lumenInverseEquiAreaSphericalMapping(direction) * float(probeResolution)), probeResolution - 1u);
}

// UE 5.7.4 FastMathThirdParty.ush, acosFast.
float radianceCacheAcosFast(float inputValue)
{
    float x = abs(inputValue);
    float result = -0.156583f * x + (0.5f * kLumenProbePi);
    result *= sqrt(1.0f - x);
    return inputValue >= 0.0f ? result : kLumenProbePi - result;
}

bool radianceCacheProbeSeesPoint(RWStructuredBuffer<uint2> atlas, uint probeIndex, uint probeResolution,
    float3 rayVector)
{
    float distanceSq = dot(rayVector, rayVector);
    if (distanceSq == 0.0f) {
        return true;
    }
    uint2 occlusionTexel = directionTexel(rayVector, probeResolution);
    float depth = unpackRadianceDepth(atlas[rawProbeAddress(probeIndex, occlusionTexel, probeResolution)]).w;
    return depth * depth >= distanceSq;
}

// UE 5.7.4 LumenRadianceCache.usf, FilterProbeRadianceWithGatherCS.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadianceCacheFilterMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheTraceParams params = getParameters<LumenRadianceCacheTraceParams>();
    RadianceCacheTraceData trace = unpackProbeTraceData(resolveUniform(params.traceData)[thread.z]);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[trace.clipmapIndex];
    uint probeResolution = clipmap.layout.z;
    uint2 texel = thread.xy;
    if (any(texel >= probeResolution)) {
        return;
    }

    RWStructuredBuffer<uint2> rawAtlas = resolveUniform(params.rawAtlas);
    uint atlasAddress = rawProbeAddress(trace.probeIndex, texel, probeResolution);
    float4 traceSample = unpackRadianceDepth(rawAtlas[atlasAddress]);
    float3 radiance = traceSample.xyz;
    float hitDistance = traceSample.w;
    float totalWeight = 1.0f;
    float2 probeUV = (float2(texel) + 0.5f) / float(probeResolution);
    float3 direction = lumenEquiAreaSphericalMapping(probeUV);
    int3 probeCoord = int3(floor(lumenRadianceProbeCoordFloat(trace.worldCenter, clipmap)));

    const int3 offsets[6] = {int3(-1, 0, 0), int3(1, 0, 0), int3(0, -1, 0),
        int3(0, 1, 0), int3(0, 0, -1), int3(0, 0, 1)};
    StructuredBuffer<uint> indirection = resolveUniform(params.indirection);
    for (uint offsetIndex = 0u; offsetIndex < 6u; ++offsetIndex) {
        int3 neighbor = probeCoord + offsets[offsetIndex];
        if (all(neighbor >= 0) && all(neighbor < int(clipmap.layout.x))) {
            uint neighborAddress = lumenRadianceIndirectionAddress(uint3(neighbor), trace.clipmapIndex,
                clipmap.layout.x, clipmap.layout.y);
            uint neighborIndex = indirection[neighborAddress];
            if (neighborIndex != kLumenRadianceInvalidProbe) {
                float4 neighborSample = unpackRadianceDepth(rawAtlas[rawProbeAddress(neighborIndex, texel, probeResolution)]);
                float neighborHitDistance = neighborSample.w;
                // The screen-probe radiance permutation has CalculateIrradiance disabled, so UE's offsets are zero.
                float3 neighborWorldPosition = lumenRadianceProbeWorldPosition(uint3(neighbor), clipmap);

                const float occlusionTestOffset = 2.0f * clipmap.traceSettings.x;
                float3 neighborOcclusionTestPosition = neighborWorldPosition + occlusionTestOffset * direction;
                float3 toNeighborOcclusionPosition = neighborOcclusionTestPosition - trace.worldCenter;
                float3 occlusionTestPosition = trace.worldCenter + occlusionTestOffset * direction;
                float3 toOcclusionPosition = occlusionTestPosition - neighborWorldPosition;
                float occlusionWeight = radianceCacheProbeSeesPoint(rawAtlas, trace.probeIndex, probeResolution,
                    toNeighborOcclusionPosition) && radianceCacheProbeSeesPoint(rawAtlas, neighborIndex, probeResolution,
                    toOcclusionPosition) ? 1.0f : 0.0f;

                if (hitDistance >= 0.0f) {
                    neighborHitDistance = min(neighborHitDistance, hitDistance);
                }
                float3 neighborHitPosition = neighborWorldPosition + direction * neighborHitDistance;
                float3 toNeighborHit = neighborHitPosition - trace.worldCenter;
                float toNeighborHitLength = length(toNeighborHit);
                if (toNeighborHitLength > 0.0f) {
                    float neighborAngle = radianceCacheAcosFast(clamp(dot(toNeighborHit, direction) / toNeighborHitLength, -1.0f, 1.0f));
                    float angleWeight = 1.0f - saturate(neighborAngle / kSpatialFilterMaxRadianceHitAngle);
                    float weight = angleWeight * occlusionWeight;
                    radiance += neighborSample.xyz * weight;
                    totalWeight += weight;
                }
            }
        }
    }

    resolveUniform(params.filteredAtlas)[atlasAddress] = packRadianceDepth(radiance / totalWeight, hitDistance);
}

// UE 5.7.4 LumenRadianceCache.usf, FixupBordersAndGenerateMipsCS at the default single mip.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenRadianceCacheBordersMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheTraceParams params = getParameters<LumenRadianceCacheTraceParams>();
    RadianceCacheTraceData trace = unpackProbeTraceData(resolveUniform(params.traceData)[thread.z]);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[trace.clipmapIndex];
    uint probeResolution = clipmap.layout.z;
    const uint border = 1u;
    uint finalProbeResolution = probeResolution + 2u * border;
    uint2 finalProbeTexel = thread.xy;
    if (any(finalProbeTexel >= finalProbeResolution)) {
        return;
    }

    uint2 texel = wrapOctahedralBorder(finalProbeTexel, finalProbeResolution, border);
    uint2 probeAtlasCoord = uint2(trace.probeIndex % clipmap.layout.w, trace.probeIndex / clipmap.layout.w);
    uint2 finalAtlasCoord = finalProbeTexel + finalProbeResolution * probeAtlasCoord;
    uint2 packedRadiance = resolveUniform(params.filteredAtlas)[rawProbeAddress(trace.probeIndex, texel, probeResolution)];
    resolveUniform(params.radianceAtlas)[finalAtlasCoord] = float4(unpackRadianceDepth(packedRadiance).xyz, 0.0f);
}
