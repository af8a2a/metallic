module LumenScreenProbeGather;

import ShaderCore;
import ParameterRoot;
import LumenGlobalRay;
import LumenProbeMath;
import GlobalDistanceField;
import LumenAdaptiveProbes;
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
import LumenReflectionSampling;
import ScenePathMath;
#endif
#if defined(METALLIC_LUMEN_STRUCTURED_IMPORTANCE_SAMPLING)
import LumenProbeImportance;
#endif
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
import LumenScreenProbeWork;
#endif
#if defined(METALLIC_LUMEN_SCREEN_TRACES) || defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_FULL_RESOLUTION_JITTER)
import LumenScreenTrace;
#endif
#if defined(METALLIC_LUMEN_DETAIL_TRACING)
import LumenMeshSDFGrid;
#endif
#if defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION)
import LumenMeshSDFTrace;
import MeshDistanceField;
#endif
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
import LumenRadianceCache;
import LumenRadianceCacheInterpolation;
#endif

using Metallic;
using Metallic.Lumen;
using Metallic.DistanceField;

#if defined(METALLIC_LUMEN_PROBE_TRACE_FALLBACK_ONLY) || defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
#define METALLIC_LUMEN_PROBE_TRACE_COMPACTED 1
#endif

static const uint kProbeDownsampleFactor = 16u;
static const uint kProbeRayResolution = kLumenProbeTraceResolution;
static const uint kRaysPerProbe = kProbeRayResolution * kProbeRayResolution;
static const float kSurfaceBias = 0.05f;
static const float kMinTraceDistance = 0.0001f;
static const float kInterpolationDepthWeight = -200.0f;
static const float kMinInterpolationFallbackWeight = 0.0001f;

typealias LumenSceneColorTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;
typealias LumenCompositeTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;

#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
// Matches LumenProbeTraceReadbackRecord and the RenderDebug layout.
struct LumenProbeTraceReadback
{
    float screenEnd;
    float detailEnd;
    float gdfMin;
    float gdfMax;
    float3 wrcRadiance;
    float gdfHitDistance;
    uint gdfClassification;
    uint gdfHitClipmap;
    uint gdfSteps;
    uint flags;
};

static const uint kReadbackReceiverValid = 1u << 0u;
static const uint kReadbackScreenInvoked = 1u << 1u;
static const uint kReadbackScreenAccepted = 1u << 2u;
static const uint kReadbackDetailInvoked = 1u << 3u;
static const uint kReadbackDetailAccepted = 1u << 4u;
static const uint kReadbackGDFInvoked = 1u << 5u;
static const uint kReadbackWRCCovered = 1u << 6u;
static const uint kReadbackWRCSampled = 1u << 7u;
static const uint kReadbackGDFPrefixInvoked = 1u << 8u;
static const uint kReadbackGDFPrefixAccepted = 1u << 9u;
static const uint kReadbackGDFPrefixClassificationShift = 10u;
static const uint kReadbackGDFPrefixRecedingRejected = 1u << 14u;
static const uint kReadbackGDFPrefixMeshSDFInvoked = 1u << 15u;
static const uint kReadbackGDFPrefixMeshSDFConfirmed = 1u << 16u;
static const uint kReadbackGDFPrefixMeshSDFInconclusive = 1u << 17u;
static const uint kInvalidProbeTraceReadbackIndex = ~0u;

void writeProbeTraceReadback(ResourceHandle<RWStructuredBuffer<LumenProbeTraceReadback>> buffer, uint index,
    LumenProbeTraceReadback record)
{
    if (index != kInvalidProbeTraceReadbackIndex) {
        resolveUniform(buffer)[index] = record;
    }
}
#endif

#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
struct LumenProbeInterpolationReadback
{
    uint4 indices;
    float4 weights;
};
#endif

#if defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION)
struct LumenMeshSDFGatherBindings
{
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> atlas;
    SamplerHandle atlasSampler;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
#if defined(METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION)
    ResourceHandle<StructuredBuffer<LumenMeshSDFValidationBounds>> validationBounds;
#else
    ResourceHandle<StructuredBuffer<uint2>> gridCells;
#endif
    ResourceHandle<StructuredBuffer<uint>> gridIndices;
    uint4 grid;
    float3 cameraPosition;
    float maxMeshSDFTraceDistance;
    float cardTraceEndDistance;
    uint bricksPerAxis;
};
#endif

struct LumenScreenProbeGatherParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<Texture2D<float4>> diffuse;
    ResourceHandle<Texture2D<float4>> directColor;
    ResourceHandle<RWStructuredBuffer<float4>> traceOutput;
    ResourceHandle<StructuredBuffer<float4>> traceInput;
    ResourceHandle<LumenSceneColorTexture> indirectDiffuse;
    ResourceHandle<LumenCompositeTexture> color;
    uint width;
    uint height;
    uint probeWidth;
    uint probeHeight;
    uint clipmapCount;
    float maxTraceDistance;
    ResourceHandle<StructuredBuffer<uint2>> probePositions;
    ResourceHandle<RWStructuredBuffer<float>> traceDistance;
    ResourceHandle<StructuredBuffer<float>> closestHZB;
    ResourceHandle<Texture2D<float>> previousDepth;
    ResourceHandle<Texture2D<float4>> previousColor;
#if defined(METALLIC_LUMEN_SCREEN_TRACES)
    ResourceHandle<StructuredBuffer<LumenScreenTraceView>> traceView;
#else
    uint traceView;
#endif
    uint frameIndex;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
    ResourceHandle<StructuredBuffer<uint>> cacheIndirection;
    ResourceHandle<StructuredBuffer<LumenRadianceCacheClipmap>> cacheClipmaps;
    ResourceHandle<Texture2D<float4>> cacheRadiance;
#else
    uint3 unusedRadianceCache;
#endif
#if defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION)
    LumenMeshSDFGatherBindings detail;
#else
    uint unusedDetail[17];
#endif
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    ResourceHandle<RWStructuredBuffer<LumenProbeTraceReadback>> traceReadback;
    uint4 traceReadbackTiles;
#else
    uint unusedProbeTraceReadback[5];
#endif
#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
    ResourceHandle<RWStructuredBuffer<LumenProbeInterpolationReadback>> interpolationReadback;
    uint4 interpolationReadbackPixels;
#else
    uint unusedProbeInterpolationReadback[5];
#endif
    ResourceHandle<Texture2D<float2>> blueNoiseVec2;
    ResourceHandle<Texture2D<float4>> probeIrradiance;
    SamplerHandle irradianceSampler;
    float screenTraceRelativeDepthThickness;
    ResourceHandle<Texture2D<float4>> shDirectional;
    uint rayDirectionFrameIndex;
    ResourceHandle<Texture2D<float>> jitterDepth;
    ResourceHandle<StructuredBuffer<ViewConstants>> jitterView;
    ResourceHandle<RWStructuredBuffer<float4>> jitterOffsets;
    float fullResolutionJitterWidth;
    ResourceHandle<LumenSceneColorTexture> indirectIrradiance;
    uint writeIndirectIrradiance;
    ResourceHandle<Texture2D<float>> blueNoiseScalar;
    ResourceHandle<StructuredBuffer<float4>> previousProbeExposure;
    float probeExposureFallback;
#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
    ResourceHandle<StructuredBuffer<uint>> probeWorkList;
    ResourceHandle<StructuredBuffer<uint>> probeWorkState;
#endif
#if defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
    ResourceHandle<StructuredBuffer<uint>> probeRayWorkList;
    ResourceHandle<StructuredBuffer<uint>> probeRayWorkCount;
#endif
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
    ResourceHandle<StructuredBuffer<uint>> probeOrigins;
    ResourceHandle<StructuredBuffer<uint>> probeTileAdaptiveCounts;
    ResourceHandle<StructuredBuffer<uint>> probeTileAdaptiveIndices;
#endif
#if defined(METALLIC_LUMEN_STRUCTURED_IMPORTANCE_SAMPLING)
    ResourceHandle<StructuredBuffer<uint>> structuredRayInfo;
#endif
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
    ResourceHandle<Texture2D<float4>> previousPosition;
    ResourceHandle<RWStructuredBuffer<uint>> probeTraceMoving;
    ResourceHandle<StructuredBuffer<float>> probeMoving;
    #if defined(METALLIC_LUMEN_DETAIL_TRACING)
    ResourceHandle<StructuredBuffer<LumenMeshSDFInstanceMotion>> detailMotion;
    #else
    uint unusedDetailMotion;
    #endif
#endif
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    ResourceHandle<Texture2D<float4>> roughSpecularMaterial;
    ResourceHandle<LumenSceneColorTexture> roughSpecular;
    uint writeRoughSpecular;
#endif
};

#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
// UE 5.7.4 IsTraceMoving, with EncodeScreenProbeSpeed/GetScreenProbeSpeed.
// Quantize the receiver speed in centimeters, the reference world-unit scale.
bool isLumenProbeTraceMoving(float4 receiver, float4 previousReceiver, float3 hitWorldVelocity)
{
    if (previousReceiver.w <= 0.0f) { return false; }
    const float probeWorldSpeed = f16tof32(f32tof16(length((receiver.xyz - previousReceiver.xyz) * 100.0f)) & 0x7ffeu);
    return abs(probeWorldSpeed - length(hitWorldVelocity * 100.0f)) / max(receiver.w * 100.0f, 100.0f) > 0.005f;
}

float weightedLumenProbeMoving(StructuredBuffer<float> moving, uint probe, float weight)
{
    if (weight == 0.0f) { return 0.0f; }
    return moving[probe] * weight;
}
#endif

#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS) && defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
// The debug ROI keeps tile/slot addresses even when no atlas entry owns that slot.
[shader("compute")]
[numthreads(kProbeRayResolution, kProbeRayResolution, 1)]
void lumenClearCompactProbeTraceReadbackMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    let params = getParameters<LumenScreenProbeGatherParams>();
    const uint address = lumenProbeTraceAddress(lumenProbeIndex(group.xy, group.z, params.traceReadbackTiles.z), lane.xy);
    const LumenProbeTraceReadback invalid = {-1.0f, -1.0f, -1.0f, -1.0f, float3(0.0f), -1.0f, ~0u, ~0u, 0u, 0u};
    resolveUniform(params.traceReadback)[address] = invalid;
}
#endif

#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
uint3 lumenTraceProbeGroupForIndex(LumenScreenProbeGatherParams params, uint probeIndex)
{
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
    probeIndex = resolveUniform(params.probeOrigins)[probeIndex];
#endif
    return uint3(lumenProbeTileForIndex(probeIndex, params.probeWidth), probeIndex % kLumenProbeSlotsPerTile);
}
#endif

#if defined(METALLIC_LUMEN_ACTIVE_PROBES)
typealias LumenProbeIrradianceStorage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 5>;

struct LumenScreenProbeClearParams
{
    ResourceHandle<StructuredBuffer<uint>> workList;
    ResourceHandle<StructuredBuffer<uint>> workState;
    ResourceHandle<RWStructuredBuffer<float4>> traceRadiance;
    ResourceHandle<RWStructuredBuffer<float>> traceDistance;
    ResourceHandle<RWStructuredBuffer<float4>> filteredRadiance;
    ResourceHandle<RWStructuredBuffer<float4>> scratch;
    ResourceHandle<RWStructuredBuffer<float>> compositedDistance;
    ResourceHandle<LumenProbeIrradianceStorage> irradiance;
    ResourceHandle<LumenSceneColorTexture> shDirectional;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    ResourceHandle<RWStructuredBuffer<LumenProbeTraceReadback>> traceReadback;
#else
    uint unusedTraceReadback;
#endif
    uint4 traceReadbackTiles;
    uint probeWidth;
    uint probeHeight;
};

[shader("compute")]
[numthreads(kProbeRayResolution, kProbeRayResolution, 1)]
void lumenClearInactiveScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    let params = getParameters<LumenScreenProbeClearParams>();
    const uint index = group.y * kLumenProbeDispatchWidth + group.x;
    if (index >= resolveUniform(params.workState)[1]) { return; }
    const uint capacity = params.probeWidth * params.probeHeight * kLumenProbeSlotsPerTile;
    const uint probeIndex = resolveUniform(params.workList)[capacity + index];
    const uint traceAddress = lumenProbeTraceAddress(probeIndex, lane.xy);
    resolveUniform(params.traceRadiance)[traceAddress] = 0.0f;
    resolveUniform(params.traceDistance)[traceAddress] = -1.0f;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    const uint2 tile = lumenProbeTileForIndex(probeIndex, params.probeWidth) - params.traceReadbackTiles.xy;
    if (all(tile < params.traceReadbackTiles.zw)) {
        const uint slot = probeIndex % kLumenProbeSlotsPerTile;
        const uint address = lumenProbeTraceAddress(lumenProbeIndex(tile, slot, params.traceReadbackTiles.z), lane.xy);
        const LumenProbeTraceReadback readback = {-1.0f, -1.0f, -1.0f, -1.0f, float3(0.0f), -1.0f, ~0u, ~0u, 0u, 0u};
        writeProbeTraceReadback(params.traceReadback, address, readback);
    }
#endif
    if (any(lane.xy >= kLumenProbeGatherResolution)) { return; }
    const uint gatherAddress = lumenProbeGatherAddress(probeIndex, lane.xy);
    resolveUniform(params.filteredRadiance)[gatherAddress] = 0.0f;
    resolveUniform(params.scratch)[gatherAddress] = 0.0f;
#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
    resolveUniform(params.compositedDistance)[gatherAddress] = -1.0f;
    const uint2 probeCoord = lumenProbeAtlasCoord(probeIndex, params.probeWidth);
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
    if (all(lane.xy == 0u)) { resolveUniform(params.irradiance)[probeCoord] = 0.0f; }
    if (lane.y == 0u && lane.x < 6u) {
        const uint atlasWidth = params.probeWidth * kLumenProbeSlotsPerTileAxis;
        resolveUniform(params.shDirectional)[probeCoord + uint2(lane.x * atlasWidth, 0u)] = 0.0f;
    }
#else
    resolveUniform(params.irradiance)[probeCoord * kLumenIrradianceProbeWithBorderResolution + lane.xy] = 0.0f;
#endif
#endif
}
#endif

// UE 5.7.4 GetUniformScreenProbeScreenPosition at Hammersley phase zero.
uint2 uniformProbePosition(uint2 tile, uint2 viewSize)
{
    return min(tile * kProbeDownsampleFactor, viewSize - 1u);
}

float3 probeRayDirection(uint2 ray, float2 probeTexelCenter)
{
    return lumenEquiAreaSphericalMapping((float2(ray) + probeTexelCenter) / float(kProbeRayResolution));
}

float3 storeTraceLighting(float3 lighting, LumenScreenProbeGatherParams params)
{
#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
    // The composite pass clamps and quantizes each raw trace before scattering.
    return lighting;
#else
    float preExposure = lumenProbePreExposure(params.previousProbeExposure, params.probeExposureFallback);
    return lumenDecodeProbeLighting(lumenQuantizeProbeLighting(lighting * preExposure, kRaysPerProbe),
        kRaysPerProbe) / preExposure;
#endif
}

float lumenProbeSoftwareTraceStart(float savedEnd)
{
    return max(kMinTraceDistance, savedEnd - kSurfaceBias * 2.0f);
}

#if defined(METALLIC_LUMEN_DETAIL_TRACING) && !defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
bool shouldTraceLumenProbeDetail(LumenMeshSDFGatherBindings detail, bool sceneIdentityMatches, bool reachedRadianceCache,
    float receiverDepth, float screenEnd)
{
    return detail.grid.w != 0u && sceneIdentityMatches && !reachedRadianceCache &&
        receiverDepth < detail.cardTraceEndDistance && screenEnd < detail.maxMeshSDFTraceDistance;
}
#endif

[shader("compute")]
#if defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
[numthreads(kLumenProbeFallbackGroupSize, 1, 1)]
void lumenTraceScreenProbeDetailMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
#elif defined(METALLIC_LUMEN_PROBE_TRACE_FALLBACK_ONLY)
[numthreads(kLumenProbeFallbackGroupSize, 1, 1)]
void lumenTraceScreenProbeFallbackMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
#else
[numthreads(kProbeRayResolution, kProbeRayResolution, 1)]
void lumenTraceScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
#endif
{
    LumenScreenProbeGatherParams params = getParameters<LumenScreenProbeGatherParams>();
#if defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
    const uint compactedIndex = (group.y * kLumenProbeDispatchWidth + group.x) * kLumenProbeFallbackGroupSize + lane.x;
#if defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
    const uint countIndex = 1u;
#else
    const uint countIndex = 0u;
#endif
    if (compactedIndex >= resolveUniform(params.probeRayWorkCount)[countIndex]) { return; }
    const uint denseRay = resolveUniform(params.probeRayWorkList)[compactedIndex];
    const uint probeIndex = denseRay / kRaysPerProbe;
    const uint rayIndex = denseRay % kRaysPerProbe;
    lane = uint3(rayIndex % kProbeRayResolution, rayIndex / kProbeRayResolution, 0u);
    group = lumenTraceProbeGroupForIndex(params, probeIndex);
#elif defined(METALLIC_LUMEN_ACTIVE_PROBES)
    uint probeIndex;
    if (!resolveLumenActiveProbe(group, resolveUniform(params.probeWorkList), resolveUniform(params.probeWorkState),
            probeIndex)) { return; }
    group = lumenTraceProbeGroupForIndex(params, probeIndex);
#elif defined(METALLIC_LUMEN_FINAL_GATHER)
    const uint probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
#else
    const uint probeIndex = group.y * params.probeWidth + group.x;
#endif
    Texture2D<float4> position = resolveUniform(params.position);
    RWStructuredBuffer<float4> traceOutput = resolveUniform(params.traceOutput);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    const uint2 readbackTile = group.xy - params.traceReadbackTiles.xy;
    const uint readbackIndex = all(readbackTile < params.traceReadbackTiles.zw) ?
        lumenProbeTraceAddress(lumenProbeIndex(readbackTile, group.z, params.traceReadbackTiles.z), lane.xy) :
        kInvalidProbeTraceReadbackIndex;
    LumenProbeTraceReadback readback = {-1.0f, -1.0f, -1.0f, -1.0f, float3(0.0f), -1.0f, ~0u, ~0u, 0u, 0u};
#if defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
    if (readbackIndex != kInvalidProbeTraceReadbackIndex) {
        readback = resolveUniform(params.traceReadback)[readbackIndex];
    }
#endif
#endif
    const uint address = lumenProbeTraceAddress(probeIndex, lane.xy);
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE) && !defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
    resolveUniform(params.probeTraceMoving)[address] = 0u;
#endif
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
    RWStructuredBuffer<float> distances = resolveUniform(params.traceDistance);
    uint2 probePixel = probes[probeIndex];
#if !defined(METALLIC_LUMEN_ACTIVE_PROBES)
    if (probePixel.x == kLumenInvalidProbe) {
        traceOutput[address] = 0.0f;
        distances[address] = -1.0f;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
        return;
    }
#endif
#else
    uint2 probePixel = uniformProbePosition(group.xy, uint2(params.width, params.height));
#endif
    float4 receiver = position.Load(int3(probePixel, 0));
    float4 trace = 0.0f;
    if (receiver.w > 0.0f) {
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        readback.flags |= kReadbackReceiverValid;
#endif
        Texture2D<float4> normal = resolveUniform(params.normal);
#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
        // Every lane and adaptive slot in a screen tile uses the same UE GetProbeTexelCenter.
        uint2 screenTileCoord = lumenScreenTileCoord(probePixel, params.frameIndex);
        float2 probeTexelCenter = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoiseVec2), screenTileCoord,
            params.rayDirectionFrameIndex);
#else
        float2 probeTexelCenter = 0.5f;
#endif
#if defined(METALLIC_LUMEN_STRUCTURED_IMPORTANCE_SAMPLING)
        // UE 5.7.4 GetProbeTracingUV. The destination remains the original trace lane.
        const uint rayInfo = resolveUniform(params.structuredRayInfo)[address];
        uint2 rayTexelCoord;
        uint rayLevel;
        lumenUnpackProbeRayInfo(rayInfo, rayTexelCoord, rayLevel);
        const uint mipSize = lumenProbeRayMipSize(rayLevel);
        const float numSupersamples = 1.0f;
        const float invSupersampledMipSize = 1.0f / (mipSize * numSupersamples);
        const float2 probeUV = (rayTexelCoord * numSupersamples + probeTexelCenter) * invSupersampledMipSize;
        const float3 rayDirection = lumenEquiAreaSphericalMapping(probeUV);
#else
        float3 rayDirection = probeRayDirection(lane.xy, probeTexelCenter);
#endif
        float3 receiverNormal = normalize(normal.Load(int3(probePixel, 0)).xyz);
        // UE 5.7.4 TraceVoxels, opaque software tracing.
        float3 samplePosition = receiver.xyz + kSurfaceBias * rayDirection;
        samplePosition += kSurfaceBias * receiverNormal;
        float traceDistance = params.maxTraceDistance;
        float voxelStartDistance = kMinTraceDistance;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE) && !defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
        LumenRadianceCacheInputs cacheInputs = {resolveUniform(params.cacheIndirection), resolveUniform(params.cacheClipmaps),
            resolveUniform(params.cacheRadiance), resolveUniform(params.scene.sampler)};
        LumenRadianceCacheCoverage coverage = getLumenRadianceCacheCoverage(cacheInputs, receiver.xyz);
        if (coverage.valid) {
            traceDistance = min(traceDistance, coverage.minTraceDistance);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackWRCCovered;
#endif
        }
#endif
#if defined(METALLIC_LUMEN_SCREEN_TRACES) || defined(METALLIC_LUMEN_DETAIL_TRACING)
        uint2 traceBufferCoord = uint2(group.x * kLumenProbeSlotsPerTile + group.z, group.y) * kProbeRayResolution + lane.xy;
        float screenDistance = 0.0f;
        float screenContinuationDistance = 0.0f;
#endif
#if defined(METALLIC_LUMEN_SCREEN_TRACES)
#if defined(METALLIC_LUMEN_PROBE_TRACE_COMPACTED)
        screenContinuationDistance = distances[address];
#else
        LumenScreenTraceInputs screenInputs = {resolveUniform(params.closestHZB), resolveUniform(params.previousDepth),
            resolveUniform(params.previousColor), resolveUniform(params.traceView)};
        float3 screenRadiance;
        float4 hitWorldVelocity;
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
        let previousPosition = resolveUniform(params.previousPosition);
        const bool motionHistory = true;
#else
        let previousPosition = position;
        const bool motionHistory = false;
#endif
        const bool screenHit = traceLumenScreenRay(screenInputs, probePixel, receiver.xyz, receiverNormal,
            rayDirection, traceDistance, params.maxTraceDistance, params.screenTraceRelativeDepthThickness,
            traceBufferCoord, motionHistory, position, previousPosition, screenRadiance,
            screenDistance, screenContinuationDistance, hitWorldVelocity);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        readback.flags |= kReadbackScreenInvoked;
        readback.screenEnd = screenContinuationDistance;
#endif
        if (screenHit) {
            traceOutput[address] = float4(storeTraceLighting(screenRadiance, params), 5.0f);
            distances[address] = screenDistance;
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
            if (hitWorldVelocity.w > 0.0f) {
                resolveUniform(params.probeTraceMoving)[address] = uint(isLumenProbeTraceMoving(receiver,
                    previousPosition.Load(int3(probePixel, 0)), hitWorldVelocity.xyz));
            }
#endif
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackScreenAccepted;
            writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
            return;
        }
#endif
#if defined(METALLIC_LUMEN_PROBE_TRACE_SCREEN_ONLY)
        // Use the final atlas for temporary continuation state. Detail eligibility follows
        // UE ScreenProbeCompactTracesCS and stays separate from final GDF eligibility.
        float pendingClass = kLumenProbeRayPending;
#if defined(METALLIC_LUMEN_SPLIT_DETAIL_TRACING)
        bool reachedRadianceCache = false;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        reachedRadianceCache = coverage.valid && screenDistance >= traceDistance;
#endif
        let cardScene = resolveUniform(params.scene.cardScene);
        let objectGridScene = resolveUniform(params.scene.objectGridScene);
        const bool identityMatches = cardScene[2] == objectGridScene[0] && cardScene[3] == objectGridScene[1];
        if (shouldTraceLumenProbeDetail(params.detail, identityMatches, reachedRadianceCache, receiver.w,
                screenContinuationDistance)) {
            pendingClass = kLumenProbeRayPendingDetail;
        }
#endif
        traceOutput[address] = float4(0.0f, 0.0f, 0.0f, pendingClass);
        distances[address] = screenContinuationDistance;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
        return;
#else
#if defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_SCREEN_GDF_CONTINUATION)
        voxelStartDistance = lumenProbeSoftwareTraceStart(screenContinuationDistance);
#elif defined(METALLIC_LUMEN_RADIANCE_CACHE)
        if (coverage.valid) {
            // TraceVoxels also handles unresolved screen rays that reached the cache interval.
            voxelStartDistance = lumenProbeSoftwareTraceStart(screenContinuationDistance);
        }
#endif
#endif
#endif
#if !defined(METALLIC_LUMEN_PROBE_TRACE_SCREEN_ONLY)
        LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
#if defined(METALLIC_LUMEN_DETAIL_TRACING) && !defined(METALLIC_LUMEN_PROBE_TRACE_FALLBACK_ONLY)
        const LumenMeshSDFGatherBindings detail = params.detail;
#if defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
        {
#else
        bool reachedRadianceCache = false;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        reachedRadianceCache = coverage.valid && screenDistance >= traceDistance;
#endif
        // UE ScreenProbeCompactTracesCS tests the screen endpoint and view depth before TraceMeshSDFs.
        if (shouldTraceLumenProbeDetail(detail, scene.identityMatches, reachedRadianceCache, receiver.w,
                screenContinuationDistance)) {
#endif
            uint cellIndex = lumenMeshSDFGridCellIndex(probePixel, receiver.w, detail.grid.xyz);
            uint2 cell = resolveUniform(detail.gridCells)[cellIndex];
            MeshDistanceFieldResources meshes = {resolveUniform(detail.mips), resolveUniform(detail.indirection),
                resolveUniform(detail.atlas), resolveUniform(detail.atlasSampler), detail.bricksPerAxis};
            LumenMeshSDFTraceScene meshScene = {meshes, resolveUniform(detail.instances),
                resolveUniform(detail.gridIndices), scene.cache};
            LumenMeshSDFTraceInput meshRay = {samplePosition, rayDirection, voxelStartDistance,
                params.maxTraceDistance, min(detail.maxMeshSDFTraceDistance, params.maxTraceDistance), cell.x, cell.y, true};
            LumenMeshSDFTraceResult detailHit = traceLumenMeshSDFs(meshScene, meshRay);
            float distanceFromViewpoint = length(detail.cameraPosition - receiver.xyz);
            float distanceFade = saturate(6.0f * distanceFromViewpoint / detail.cardTraceEndDistance - 5.0f);
            float transparency = lerp(detailHit.transparency, 1.0f, distanceFade);
            float detailDistance = min(detailHit.hitDistance + length(receiver.xyz - samplePosition), params.maxTraceDistance);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackDetailInvoked;
            readback.detailEnd = detailDistance;
#endif
            if (transparency < screenTraceNoise(float2(traceBufferCoord) + 0.5f, 0.0f)) {
                traceOutput[address] = float4(storeTraceLighting(detailHit.radiance * (1.0f - distanceFade), params), 9.0f);
                distances[address] = detailDistance;
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
                const uint hitObject = detailHit.instanceIndex;
                const float3 hitWorldPosition = samplePosition + rayDirection * detailHit.hitDistance;
                const float3 hitVolumePosition = meshDistanceFieldWorldToVolume(meshScene.instances[hitObject], hitWorldPosition);
                const float3 oldHitWorldPosition =
                    lumenMeshSDFPreviousPosition(resolveUniform(params.detailMotion)[hitObject], hitVolumePosition);
                const float3 hitWorldVelocity = hitWorldPosition - oldHitWorldPosition;
                resolveUniform(params.probeTraceMoving)[address] = uint(isLumenProbeTraceMoving(receiver,
                    resolveUniform(params.previousPosition).Load(int3(probePixel, 0)), hitWorldVelocity));
#endif
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
                readback.flags |= kReadbackDetailAccepted;
                writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
                return;
            }
            const float detailContinuationDistance = decodeLumenTraceContinuationDistance(detailDistance * 100.0f);
#if defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
            traceOutput[address] = float4(0.0f, 0.0f, 0.0f, kLumenProbeRayPending);
            distances[address] = detailContinuationDistance;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
            return;
#else
            voxelStartDistance = lumenProbeSoftwareTraceStart(detailContinuationDistance);
#endif
        }
#endif
#if !defined(METALLIC_LUMEN_PROBE_TRACE_DETAIL_ONLY)
        LumenGlobalTraceRay ray = {float4(samplePosition, voxelStartDistance), float4(rayDirection, traceDistance)};
        LumenGlobalTraceResult hit;
#if defined(METALLIC_LUMEN_GDF_PREFIX_VALIDATION)
        // Screen tracing uses a different biased origin. Verify the part it would skip on this ray.
        const float prefixEnd = min(voxelStartDistance, traceDistance);
        bool prefixAccepted = false;
        if (prefixEnd > kMinTraceDistance) {
            LumenGlobalTraceRay prefix = {float4(samplePosition, kMinTraceDistance), float4(rayDirection, prefixEnd)};
#if defined(METALLIC_LUMEN_GDF_PREFIX_REJECT_RECEDING)
            bool recedingExpansionRejected;
            hit = traceLumenGlobalRay(scene, prefix, true, recedingExpansionRejected);
#else
            hit = traceLumenGlobalRay(scene, prefix);
#endif
            prefixAccepted = hit.state.x == kGlobalDistanceFieldHit;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackGDFPrefixInvoked |
                ((hit.state.x & 0xfu) << kReadbackGDFPrefixClassificationShift);
#if defined(METALLIC_LUMEN_GDF_PREFIX_REJECT_RECEDING)
            if (recedingExpansionRejected) { readback.flags |= kReadbackGDFPrefixRecedingRejected; }
#endif
#endif
#if defined(METALLIC_LUMEN_GDF_PREFIX_MESH_VALIDATION)
            if (prefixAccepted) {
                const LumenMeshSDFGatherBindings validation = params.detail;
                MeshDistanceFieldResources meshes = {resolveUniform(validation.mips),
                    resolveUniform(validation.indirection), resolveUniform(validation.atlas),
                    resolveUniform(validation.atlasSampler), validation.bricksPerAxis};
                float meshDistance;
                float3 meshRadiance;
                bool inconclusive;
                const bool confirmed = validateLumenMeshSDFPrefix(meshes, resolveUniform(validation.instances),
                    resolveUniform(validation.validationBounds), validation.grid.w, scene.cache,
                    samplePosition, rayDirection, kMinTraceDistance, prefixEnd,
                    meshDistance, meshRadiance, inconclusive);
                prefixAccepted = confirmed || inconclusive;
                if (confirmed) {
                    hit.metrics.x = meshDistance;
                    hit.radiance.xyz = meshRadiance;
                }
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
                readback.flags |= kReadbackGDFPrefixMeshSDFInvoked;
                if (confirmed) { readback.flags |= kReadbackGDFPrefixMeshSDFConfirmed; }
                if (inconclusive) { readback.flags |= kReadbackGDFPrefixMeshSDFInconclusive; }
#endif
            }
#endif
            if (prefixAccepted) {
                ray = prefix;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
                readback.flags |= kReadbackGDFPrefixAccepted;
#endif
            }
        }
        if (!prefixAccepted)
#endif
        {
            hit = traceLumenGlobalRay(scene, ray);
        }
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        readback.flags |= kReadbackGDFInvoked;
        readback.gdfMin = ray.originAndMinDistance.w;
        readback.gdfMax = ray.directionAndMaxDistance.w;
        readback.gdfClassification = hit.state.x;
        readback.gdfHitClipmap = hit.state.y;
        readback.gdfSteps = hit.state.z;
        readback.gdfHitDistance = hit.metrics.x;
#endif
        float3 lighting = hit.radiance.xyz;
        float traceClass = float(hit.state.x);
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        const float transparency = hit.state.x == kGlobalDistanceFieldHit ? 0.0f : 1.0f;
        if (coverage.valid) {
            if (transparency > 0.5f) {
                traceClass = 6.0f;
            }
            // UE 5.7.4 TraceVoxels calls SampleRadianceCacheAndApply for valid coverage.
            const float3 cacheRadiance = sampleLumenRadianceCache(cacheInputs, coverage, receiver.xyz, rayDirection);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackWRCSampled;
            readback.wrcRadiance = cacheRadiance;
#endif
            lighting += cacheRadiance * transparency;
        }
#endif
        trace = float4(storeTraceLighting(lighting, params), traceClass);
#if defined(METALLIC_LUMEN_FINAL_GATHER)
        bool unresolved = hit.state.x == kGlobalDistanceFieldUnavailable || hit.state.x == kGlobalDistanceFieldStepLimit;
        distances[address] = hit.state.x == kGlobalDistanceFieldHit ? hit.metrics.x :
            (unresolved ? -1.0f : params.maxTraceDistance);
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        if (coverage.valid) {
            // TraceVoxels with USE_VOXEL_TRACE_HIT_DISTANCE and SCREEN_PROBE_EXTRA_AO disabled.
            distances[address] = transparency <= 0.5f ? traceDistance : params.maxTraceDistance;
        }
#endif
#endif
#endif
#endif
    } else {
#if defined(METALLIC_LUMEN_FINAL_GATHER)
        distances[address] = -1.0f;
#endif
    }
    traceOutput[address] = trace;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
}

struct ScreenProbeSample
{
    uint2 atlasCoords[4];
    float4 weights;
};

// UE 5.7.4 CalculateUniformUpsampleInterpolationWeights, opaque plane weighting.
ScreenProbeSample calculateProbeInterpolation(uint2 pixel, uint2 viewSize, uint2 probeSize, float4 receiver,
    float3 normal, Texture2D<float4> position)
{
    uint2 screenTileCoord00 = min(pixel / kProbeDownsampleFactor, max(probeSize, 2u) - 2u);
    const uint bilinearExpand = 1u;
    float2 bilinear = float2(pixel - screenTileCoord00 * kProbeDownsampleFactor + bilinearExpand) /
        float(kProbeDownsampleFactor + 2u * bilinearExpand);
    ScreenProbeSample sample;
    sample.weights = float4((1.0f - bilinear.y) * (1.0f - bilinear.x), (1.0f - bilinear.y) * bilinear.x,
        bilinear.y * (1.0f - bilinear.x), bilinear.y * bilinear.x);

    float4 scenePlane = float4(normal, dot(receiver.xyz, normal));
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
        uint2 tile = min(screenTileCoord00 + uint2(corner % 2u, corner / 2u), probeSize - 1u);
        sample.atlasCoords[corner] = tile;
        float4 probe = position.Load(int3(uniformProbePosition(tile, viewSize), 0));
        float planeDistance = abs(dot(float4(probe.xyz, -1.0f), scenePlane));
        float relativeDepthDifference = abs(planeDistance / receiver.w);
        sample.weights[corner] *= probe.w > 0.0f ? exp2(kInterpolationDepthWeight * relativeDepthDifference) : 0.0f;
    }

    float weightSum = dot(sample.weights, 1.0f);
    if (weightSum >= kMinInterpolationFallbackWeight) {
        sample.weights /= max(weightSum, kMinInterpolationFallbackWeight);
    } else {
        sample.weights = 0.0f;
    }
    return sample;
}

#if defined(METALLIC_LUMEN_FINAL_GATHER) && defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
// UE 5.7.4 GetScreenProbeSH and EvaluateSHIrradiance, opaque diffuse with AO = 1.
float3 interpolateScreenProbeSH(uint4 probeIndices, float4 weights, float3 normal,
    uint probeWidth, Texture2D<float4> ambient, Texture2D<float4> directional)
{
    const ThreeBandSHVector transfer = calcDiffuseTransferSH3(normal, 1.0f);
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
    uint atlasWidth = probeWidth;
#else
    uint atlasWidth = probeWidth * kLumenProbeSlotsPerTileAxis;
#endif
    ThreeBandSHVector lighting[3];
    [unroll]
    for (uint c = 0u; c < 3u; ++c) {
        lighting[c] = (ThreeBandSHVector)0;
    }
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
    const uint numCorners = 1u;
#else
    const uint numCorners = 4u;
#endif
    [unroll]
    for (uint corner = 0u; corner < numCorners; ++corner) {
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
        // Skip unwritten slots while preserving signed weights at view edges.
        if (weights[corner] == 0.0f) { continue; }
        uint2 coord = lumenPackedProbeAtlasCoord(probeIndices[corner], probeWidth);
#else
        uint2 coord = lumenProbeAtlasCoord(probeIndices[corner], probeWidth);
#endif
        float3 ambientRGB = ambient.Load(int3(coord, 0)).xyz;
        [unroll]
        for (uint c = 0u; c < 3u; ++c) {
            float4 sh0 = directional.Load(int3(coord.x + (c * 2u) * atlasWidth, coord.y, 0));
            float4 sh1 = directional.Load(int3(coord.x + (c * 2u + 1u) * atlasWidth, coord.y, 0));
            lighting[c].v0 += float4(ambientRGB[c], sh0.xyz) * weights[corner];
            lighting[c].v1 += float4(sh0.w, sh1.xyz) * weights[corner];
            lighting[c].v2 += sh1.w * weights[corner];
        }
    }
    float3 diffuse;
    [unroll]
    for (uint c = 0u; c < 3u; ++c) {
        float coefficient = dot(lighting[c].v0, transfer.v0);
        coefficient += dot(lighting[c].v1, transfer.v1);
        coefficient += lighting[c].v2 * transfer.v2;
        diffuse[c] = max(0.0f, coefficient);
    }
    return 4.0f * kLumenProbePi * diffuse;
}
#else
// UE 5.7.4 GetScreenProbeIrradiance and InterpolateScreenProbeIrradiance.
float3 interpolateScreenProbeIrradiance(uint4 probeIndices, float4 weights, float3 normal,
    uint2 probeSize, Texture2D<float4> irradiance, SamplerState bilinearSampler)
{
    float2 probeUV = lumenInverseEquiAreaSphericalMapping(normal);
    float2 probeUVCoord = probeUV * kLumenIrradianceProbeResolution + 1.0f;
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
    uint atlasWidth, atlasHeight;
    irradiance.GetDimensions(atlasWidth, atlasHeight);
    float2 atlasExtent = float2(atlasWidth, atlasHeight);
#else
    float2 atlasExtent = float2(probeSize * kLumenIrradianceProbeTileStride);
#endif
    float3 result = 0.0f;
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
    const uint numCorners = 1u;
#else
    const uint numCorners = 4u;
#endif
    [unroll]
    for (uint corner = 0u; corner < numCorners; ++corner) {
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
        if (weights[corner] == 0.0f) { continue; }
        uint2 coord = lumenPackedProbeAtlasCoord(probeIndices[corner], probeSize.x);
#else
        uint2 coord = lumenProbeAtlasCoord(probeIndices[corner], probeSize.x);
#endif
        float2 atlasUV = (coord * kLumenIrradianceProbeWithBorderResolution + probeUVCoord) / atlasExtent;
        result += irradiance.SampleLevel(bilinearSampler, atlasUV, 0.0f).xyz * weights[corner];
    }
    return result;
}
#endif
#else
// UE 5.7.4 DIFFUSE_INTEGRATION_NUMERICAL_INTEGRAL, opaque receivers.
float3 integrateProbeDiffuse(uint4 probeIndices, float4 weights, float3 normal,
    StructuredBuffer<float4> radiance)
{
    int numValidSamples = 0;
    float3 diffuseLighting = 0.0f;
    for (uint y = 0u; y < kProbeRayResolution; ++y) {
        for (uint x = 0u; x < kProbeRayResolution; ++x) {
            uint2 ray = uint2(x, y);
            float nDotL = dot(probeRayDirection(ray, 0.5f), normal);
            if (nDotL > 0.0f) {
                float sampleWeight = saturate(nDotL);
                float3 interpolatedRadiance = 0.0f;
                [unroll]
                for (uint corner = 0u; corner < 4u; ++corner) {
                    if (weights[corner] > 0.0f) {
                        uint address = lumenProbeTraceAddress(probeIndices[corner], ray);
                        interpolatedRadiance += radiance[address].xyz * weights[corner];
                    }
                }
                diffuseLighting += interpolatedRadiance * sampleWeight;
                ++numValidSamples;
            }
        }
    }
    if (numValidSamples > 0) {
        float normalizeTerm = 2.0f * kLumenProbePi / float(numValidSamples);
        diffuseLighting *= normalizeTerm;
    }
    return diffuseLighting * (1.0f / kLumenProbePi);
}
#endif

#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
float3 sampleRoughSpecularProbe(StructuredBuffer<float4> radiance, uint probeIndex, float weight,
    uint4 texelOffsets, float2 texelFraction)
{
    // Compact placement initializes radiance only for active probes.
    if (weight == 0.0f) { return 0.0f; }

    const uint4 addresses = lumenProbeGatherAddress(probeIndex, 0u) + texelOffsets;
    const float3 row0 = lerp(radiance[addresses.x].xyz, radiance[addresses.y].xyz, texelFraction.x);
    const float3 row1 = lerp(radiance[addresses.z].xyz, radiance[addresses.w].xyz, texelFraction.x);
    return lerp(row0, row1, texelFraction.y) * weight;
}

// UE 5.7.4 InterpolateFromScreenProbes and ScreenProbeFixupBordersCS, default single radiance mip.
// Address the one-texel border directly in Metallic's linear filtered radiance buffer.
float3 interpolateRoughSpecularRadiance(StructuredBuffer<float4> radiance, uint4 probeIndices, float4 weights,
    float3 coneDirection)
{
    const uint borderedResolution = kLumenProbeGatherResolution + 2u;
    const float2 texel = lumenInverseEquiAreaSphericalMapping(coneDirection) * kLumenProbeGatherResolution + 0.5f;
    const uint2 base = uint2(floor(texel));
    const float2 texelFraction = frac(texel);
    const uint4 offsets = uint4(
        lumenProbeGatherAddress(0u, wrapOctahedralBorder(base, borderedResolution, 1u)),
        lumenProbeGatherAddress(0u, wrapOctahedralBorder(base + uint2(1u, 0u), borderedResolution, 1u)),
        lumenProbeGatherAddress(0u, wrapOctahedralBorder(base + uint2(0u, 1u), borderedResolution, 1u)),
        lumenProbeGatherAddress(0u, wrapOctahedralBorder(base + uint2(1u, 1u), borderedResolution, 1u)));
    float3 result = sampleRoughSpecularProbe(radiance, probeIndices.x, weights.x, offsets, texelFraction);
#if !defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
    result += sampleRoughSpecularProbe(radiance, probeIndices.y, weights.y, offsets, texelFraction);
    result += sampleRoughSpecularProbe(radiance, probeIndices.z, weights.z, offsets, texelFraction);
    result += sampleRoughSpecularProbe(radiance, probeIndices.w, weights.w, offsets, texelFraction);
#endif
    return result;
}

// UE 5.7.4 rough specular integration for an opaque isotropic material, with short-range AO off.
float3 evaluateRoughSpecular(LumenScreenProbeGatherParams params, uint2 pixel, float3 receiverPosition, float3 normal,
    uint4 probeIndices, float4 weights, float3 irradianceOverPi)
{
    const float roughness = resolveUniform(params.roughSpecularMaterial).Load(int3(pixel, 0)).w;
    const float fadeLength = 0.2f;
    const float maxRoughnessToEvaluateRoughSpecular = 0.8f;
    // Keep Gather independent of the dedicated reflection pass's configurable tracing threshold.
    const float diffuseLerp = saturate((roughness - maxRoughnessToEvaluateRoughSpecular + fadeLength) / fadeLength);
    float3 specularLighting = irradianceOverPi;
    if (diffuseLerp < 1.0f) {
        const ViewConstants view = resolveUniform(params.jitterView)[0];
        const float3 v = view.current.upProjection.w > 0.5f
            ? -ScenePath.cameraForwardFrom(view.current.eye, view.current.center)
            : -normalize(receiverPosition - view.current.eye.xyz);
        const float3x3 tangentBasis = getReflectionTangentBasis(normal);
        const float3 tangentV = mul(tangentBasis, v);
        const float sampleRoughness = max(max(roughness, 0.01f), 0.2f);
        const float2 alpha = (sampleRoughness * sampleRoughness).xx;
        const uint2 seed = lumenRand3DPCG16(int3(pixel, view.frame.x % 8u)).xy;
        const uint numSpecularSamples = 4u;
        const StructuredBuffer<float4> radiance = resolveUniform(params.traceInput);
        float3 roughSpecularLighting = 0.0f;
        for (uint sampleIndex = 0u; sampleIndex < numSpecularSamples; ++sampleIndex) {
            float2 e = lumenHammersley16(sampleIndex, numSpecularSamples, seed);
            const float bias = 1.0f - 0.1f;
            e.y = (e.y - 0.5f) * bias + 0.5f;
            const float4 tangentH = importanceSampleVisibleReflectionGGX(e, alpha, tangentV);
            const float3 h = mul(tangentH.xyz, tangentBasis);
            const float3 l = 2.0f * dot(v, h) * h - v;
            // SampleDefaultLitBxDF uses Weight=1 for this term. EnvBRDF is applied during composite.
            const float3 lighting = interpolateRoughSpecularRadiance(radiance, probeIndices, weights, l);
            roughSpecularLighting += lighting / (1.0f + WorkingColor.luminance(lighting));
        }
        roughSpecularLighting /= float(numSpecularSamples);
        roughSpecularLighting /= 1.0f - WorkingColor.luminance(roughSpecularLighting);
        specularLighting = lerp(roughSpecularLighting, specularLighting, diffuseLerp);
    }
    return specularLighting;
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenIntegrateScreenProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenScreenProbeGatherParams params = getParameters<LumenScreenProbeGatherParams>();
    uint2 viewSize = uint2(params.width, params.height);
    if (any(thread.xy >= viewSize)) {
        return;
    }
    Texture2D<float4> position = resolveUniform(params.position);
    Texture2D<float4> directColor = resolveUniform(params.directColor);
    LumenSceneColorTexture indirectDiffuse = resolveUniform(params.indirectDiffuse);
    LumenCompositeTexture color = resolveUniform(params.color);
    float4 receiver = position.Load(int3(thread.xy, 0));
    float3 indirect = 0.0f;
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    float3 indirectRoughSpecular = 0.0f;
#endif
#if defined(METALLIC_LUMEN_EXPORT_INDIRECT_IRRADIANCE)
    float3 rawIndirectIrradiance = 0.0f;
#endif
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    float2 noiseOffset = 0.0f;
#endif
#if defined(METALLIC_LUMEN_FULL_RESOLUTION_JITTER)
    float2 jitterNoiseOffset = 0.0f;
#endif
#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
    LumenProbeInterpolationReadback readback = {uint4(~0u), float4(0.0f)};
#endif
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    float encodedAlpha = 0.0f;
#else
    const float encodedAlpha = 1.0f;
#endif
    if (receiver.w > 0.0f) {
        Texture2D<float4> normalTexture = resolveUniform(params.normal);
        float3 normal = normalize(normalTexture.Load(int3(thread.xy, 0)).xyz);
        Texture2D<float4> diffuse = resolveUniform(params.diffuse);
#if defined(METALLIC_LUMEN_FINAL_GATHER)
#if defined(METALLIC_LUMEN_FULL_RESOLUTION_JITTER)
        // UE 5.7.4 InterpolateAndIntegrate. Scene depth is in meters here.
        if (params.fullResolutionJitterWidth > 0.0f) {
            float effectiveJitterWidth = params.fullResolutionJitterWidth *
                lerp(1.0f, .5f, saturate((receiver.w - 5.0f) / 5.0f));
            ViewConstants view = resolveUniform(params.jitterView)[0];
            float2 screenTileJitterE = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoiseVec2), thread.xy, view.frame.x);
            jitterNoiseOffset = (screenTileJitterE * 2 - 1) * kLumenProbeDownsampleFactor * effectiveJitterWidth;
            float2 jitteredScreenUV = clamp(float2(thread.xy) + jitterNoiseOffset, 0.0f, float2(viewSize) - 1.0f) *
                (1.0f / float2(viewSize));
            float depthWeight;
            {
                float4 scenePlane = float4(normal, dot(receiver.xyz, normal));
                float3 jitteredPosition = lumenScenePositionAtUV(view.current, view.jitter.xy, viewSize,
                    resolveUniform(params.jitterDepth), jitteredScreenUV);
                float planeDistance = abs(dot(float4(jitteredPosition, -1), scenePlane));
                float relativeDepthDifference = planeDistance / receiver.w;
                depthWeight = exp2(-1000000.0f * (relativeDepthDifference * relativeDepthDifference));
            }
            if (depthWeight > .01f) {
                noiseOffset = jitterNoiseOffset;
            }
        }
#endif
        StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
        LumenProbeSample sample = calculateLumenPackedAdaptiveInterpolation(thread.xy, noiseOffset, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position, probes,
            resolveUniform(params.probeTileAdaptiveCounts), resolveUniform(params.probeTileAdaptiveIndices), params.frameIndex);
#else
        LumenProbeSample sample = calculateLumenAdaptiveInterpolation(thread.xy, noiseOffset, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position, probes, params.frameIndex);
#endif
        uint4 probeIndices = sample.indices;
        // UE 5.7.4 InterpolateAndIntegrate, static lighting with zero moving fraction.
        encodedAlpha = sample.lightingValid ? 0.004f : 0.0f;
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
        // UE 5.7.4 STOCHASTIC_PROBE_INTERPOLATION and SCREEN_PROBE_JITTER_INDEX.
        bool sampleProbes = any(sample.weights > 0.0f);
        float randomValue = min(lumenProbeBlueNoiseScalar(resolveUniform(params.blueNoiseScalar),
            thread.xy, params.frameIndex), 0.99f);
        float weightSum = dot(sample.weights, 1.0f);
        randomValue *= weightSum;
        uint pickedProbe;
        if (randomValue >= sample.weights[0] + sample.weights[1] + sample.weights[2]) {
            pickedProbe = sample.indices[3];
        } else if (randomValue >= sample.weights[0] + sample.weights[1]) {
            pickedProbe = sample.indices[2];
        } else if (randomValue >= sample.weights[0]) {
            pickedProbe = sample.indices[1];
        } else {
            pickedProbe = sample.indices[0];
        }
        probeIndices = uint4(pickedProbe);
        sample.weights = float4(sampleProbes ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f);
#endif
#if defined(METALLIC_LUMEN_FAST_LIGHTING_UPDATE)
        if (sample.lightingValid) {
            let moving = resolveUniform(params.probeMoving);
            float lightingIsMoving = weightedLumenProbeMoving(moving, probeIndices.x, sample.weights.x);
    #if !defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
            lightingIsMoving += weightedLumenProbeMoving(moving, probeIndices.y, sample.weights.y);
            lightingIsMoving += weightedLumenProbeMoving(moving, probeIndices.z, sample.weights.z);
            lightingIsMoving += weightedLumenProbeMoving(moving, probeIndices.w, sample.weights.w);
    #endif
            encodedAlpha = max(lightingIsMoving, 0.004f);
        }
#endif
#else
        ScreenProbeSample sample = calculateProbeInterpolation(thread.xy, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position);
        uint4 probeIndices;
        [unroll]
        for (uint corner = 0u; corner < 4u; ++corner) {
            probeIndices[corner] = sample.atlasCoords[corner].y * params.probeWidth + sample.atlasCoords[corner].x;
        }
#endif
#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
#if defined(METALLIC_LUMEN_COMPACT_PROBE_ATLAS)
        let origins = resolveUniform(params.probeOrigins);
        readback.indices = uint4(origins[probeIndices.x], origins[probeIndices.y], origins[probeIndices.z], origins[probeIndices.w]);
#else
        readback.indices = probeIndices;
#endif
        readback.weights = sample.weights;
#endif
        if (any(sample.weights > 0.0f)) {
#if defined(METALLIC_LUMEN_FINAL_GATHER) && defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
#if defined(METALLIC_LUMEN_IRRADIANCE_SH3)
            float3 irradiance = interpolateScreenProbeSH(probeIndices, sample.weights, normal, params.probeWidth,
                resolveUniform(params.probeIrradiance), resolveUniform(params.shDirectional));
#else
            float3 irradiance = interpolateScreenProbeIrradiance(probeIndices, sample.weights, normal,
                uint2(params.probeWidth, params.probeHeight), resolveUniform(params.probeIrradiance),
                resolveUniform(params.irradianceSampler));
#endif
            float3 irradianceOverPi = irradiance * (1.0f / kLumenProbePi);
#else
            StructuredBuffer<float4> radiance = resolveUniform(params.traceInput);
            float3 irradianceOverPi = integrateProbeDiffuse(probeIndices, sample.weights, normal, radiance);
#endif
#if defined(METALLIC_LUMEN_EXPORT_INDIRECT_IRRADIANCE)
            rawIndirectIrradiance = irradianceOverPi;
#endif
            indirect = irradianceOverPi * diffuse.Load(int3(thread.xy, 0)).rgb;
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
            if (params.writeRoughSpecular != 0u) {
                indirectRoughSpecular = evaluateRoughSpecular(params, thread.xy, receiver.xyz, normal,
                    probeIndices, sample.weights, irradianceOverPi);
            }
#endif
        }
    }
#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
    uint2 readbackPixel = thread.xy - params.interpolationReadbackPixels.xy;
    if (all(readbackPixel < params.interpolationReadbackPixels.zw)) {
        uint index = readbackPixel.y * params.interpolationReadbackPixels.z + readbackPixel.x;
        resolveUniform(params.interpolationReadback)[index] = readback;
#if defined(METALLIC_LUMEN_PROBE_JITTER_READBACK)
        if (params.fullResolutionJitterWidth > 0.0f) {
            resolveUniform(params.jitterOffsets)[index] = float4(jitterNoiseOffset, noiseOffset);
        }
#endif
    }
#endif
    indirectDiffuse[thread.xy] = float4(indirect, encodedAlpha);
#if defined(METALLIC_LUMEN_ROUGH_SPECULAR)
    if (params.writeRoughSpecular != 0u) {
        resolveUniform(params.roughSpecular)[thread.xy] = float4(indirectRoughSpecular, 0.0f);
    }
#endif
#if defined(METALLIC_LUMEN_EXPORT_INDIRECT_IRRADIANCE)
    if (params.writeIndirectIrradiance != 0u) {
        resolveUniform(params.indirectIrradiance)[thread.xy] = float4(rawIndirectIrradiance, encodedAlpha);
    }
#endif
    color[thread.xy] = float4(directColor.Load(int3(thread.xy, 0)).rgb + indirect, 1.0f);
}
