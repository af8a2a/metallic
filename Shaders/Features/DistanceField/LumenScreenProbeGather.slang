module LumenScreenProbeGather;

import ShaderCore;
import ParameterRoot;
import LumenGlobalRay;
import LumenProbeMath;
import GlobalDistanceField;
#if defined(METALLIC_LUMEN_FINAL_GATHER)
import LumenAdaptiveProbes;
#endif
#if defined(METALLIC_LUMEN_SCREEN_TRACES) || defined(METALLIC_LUMEN_DETAIL_TRACING)
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
};

// UE 5.7.4 GetUniformScreenProbeScreenPosition at Hammersley phase zero.
uint2 uniformProbePosition(uint2 tile, uint2 viewSize)
{
    return min(tile * kProbeDownsampleFactor, viewSize - 1u);
}

float3 probeRayDirection(uint2 ray, float2 probeTexelCenter)
{
    return lumenEquiAreaSphericalMapping((float2(ray) + probeTexelCenter) / float(kProbeRayResolution));
}

float3 storeTraceLighting(float3 lighting)
{
#if defined(METALLIC_LUMEN_JITTER_PROBE_DIRECTIONS)
    // The composite pass clamps and quantizes each raw trace before scattering.
    return lighting;
#else
    return lumenDecodeProbeLighting(lumenQuantizeProbeLighting(lighting, kRaysPerProbe), kRaysPerProbe);
#endif
}

[shader("compute")]
[numthreads(kProbeRayResolution, kProbeRayResolution, 1)]
void lumenTraceScreenProbesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    LumenScreenProbeGatherParams params = getParameters<LumenScreenProbeGatherParams>();
    Texture2D<float4> position = resolveUniform(params.position);
    RWStructuredBuffer<float4> traceOutput = resolveUniform(params.traceOutput);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
    const uint2 readbackTile = group.xy - params.traceReadbackTiles.xy;
    const uint readbackIndex = all(readbackTile < params.traceReadbackTiles.zw) ?
        lumenProbeTraceAddress(lumenProbeIndex(readbackTile, group.z, params.traceReadbackTiles.z), lane.xy) :
        kInvalidProbeTraceReadbackIndex;
    LumenProbeTraceReadback readback = {-1.0f, -1.0f, -1.0f, -1.0f, float3(0.0f), -1.0f, ~0u, ~0u, 0u, 0u};
#endif
    uint probeIndex = group.y * params.probeWidth + group.x;
#if defined(METALLIC_LUMEN_FINAL_GATHER)
    probeIndex = lumenProbeIndex(group.xy, group.z, params.probeWidth);
    StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
    RWStructuredBuffer<float> distances = resolveUniform(params.traceDistance);
    uint2 probePixel = probes[probeIndex];
    uint address = lumenProbeTraceAddress(probeIndex, lane.xy);
    if (probePixel.x == kLumenInvalidProbe) {
        traceOutput[address] = 0.0f;
        distances[address] = -1.0f;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
        return;
    }
#else
    uint2 probePixel = uniformProbePosition(group.xy, uint2(params.width, params.height));
    uint address = lumenProbeTraceAddress(probeIndex, lane.xy);
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
        float3 rayDirection = probeRayDirection(lane.xy, probeTexelCenter);
        float3 receiverNormal = normalize(normal.Load(int3(probePixel, 0)).xyz);
        // UE 5.7.4 TraceVoxels, opaque software tracing.
        float3 samplePosition = receiver.xyz + kSurfaceBias * rayDirection;
        samplePosition += kSurfaceBias * receiverNormal;
        float traceDistance = params.maxTraceDistance;
        float voxelStartDistance = kMinTraceDistance;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
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
        LumenScreenTraceInputs screenInputs = {resolveUniform(params.closestHZB), resolveUniform(params.previousDepth),
            resolveUniform(params.previousColor), resolveUniform(params.traceView)};
        float3 screenRadiance;
        const bool screenHit = traceLumenScreenRay(screenInputs, probePixel, receiver.xyz, receiverNormal,
            rayDirection, traceDistance, params.maxTraceDistance, params.screenTraceRelativeDepthThickness,
            traceBufferCoord, screenRadiance,
            screenDistance, screenContinuationDistance);
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
        readback.flags |= kReadbackScreenInvoked;
        readback.screenEnd = screenContinuationDistance;
#endif
        if (screenHit) {
            traceOutput[address] = float4(storeTraceLighting(screenRadiance), 5.0f);
            distances[address] = screenDistance;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
            readback.flags |= kReadbackScreenAccepted;
            writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
            return;
        }
#if defined(METALLIC_LUMEN_DETAIL_TRACING) || defined(METALLIC_LUMEN_SCREEN_GDF_CONTINUATION)
        voxelStartDistance = max(kMinTraceDistance, screenContinuationDistance - kSurfaceBias * 2.0f);
#elif defined(METALLIC_LUMEN_RADIANCE_CACHE)
        if (coverage.valid) {
            // TraceVoxels also handles unresolved screen rays that reached the cache interval.
            voxelStartDistance = max(kMinTraceDistance, screenContinuationDistance - kSurfaceBias * 2.0f);
        }
#endif
#endif
        LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
#if defined(METALLIC_LUMEN_DETAIL_TRACING)
        const LumenMeshSDFGatherBindings detail = params.detail;
        bool reachedRadianceCache = false;
#if defined(METALLIC_LUMEN_RADIANCE_CACHE)
        reachedRadianceCache = coverage.valid && screenDistance >= traceDistance;
#endif
        // UE ScreenProbeCompactTracesCS tests the screen endpoint and view depth before TraceMeshSDFs.
        if (detail.grid.w != 0u && scene.identityMatches && !reachedRadianceCache &&
            receiver.w < detail.cardTraceEndDistance && screenContinuationDistance < detail.maxMeshSDFTraceDistance) {
            uint cellIndex = lumenMeshSDFGridCellIndex(probePixel, receiver.w, detail.grid.xyz);
            uint2 cell = resolveUniform(detail.gridCells)[cellIndex];
            MeshDistanceFieldResources meshes = {resolveUniform(detail.mips), resolveUniform(detail.indirection),
                resolveUniform(detail.atlas), resolveUniform(detail.atlasSampler), detail.bricksPerAxis};
            LumenMeshSDFTraceScene meshScene = {meshes, resolveUniform(detail.instances),
                resolveUniform(detail.gridIndices), scene.cache};
            LumenMeshSDFTraceInput meshRay = {samplePosition, rayDirection, voxelStartDistance,
                params.maxTraceDistance, min(detail.maxMeshSDFTraceDistance, params.maxTraceDistance), cell.x, cell.y};
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
                traceOutput[address] = float4(storeTraceLighting(detailHit.radiance * (1.0f - distanceFade)), 9.0f);
                distances[address] = detailDistance;
#if defined(METALLIC_LUMEN_PROBE_TRACE_READBACK)
                readback.flags |= kReadbackDetailAccepted;
                writeProbeTraceReadback(params.traceReadback, readbackIndex, readback);
#endif
                return;
            }
            voxelStartDistance = max(kMinTraceDistance, detailDistance - kSurfaceBias * 2.0f);
        }
#endif
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
        trace = float4(storeTraceLighting(lighting), traceClass);
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
    uint atlasWidth = probeWidth * kLumenProbeSlotsPerTileAxis;
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
        uint2 coord = lumenProbeAtlasCoord(probeIndices[corner], probeWidth);
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
    float2 atlasExtent = float2(probeSize * kLumenIrradianceProbeTileStride);
    float3 result = 0.0f;
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
    const uint numCorners = 1u;
#else
    const uint numCorners = 4u;
#endif
    [unroll]
    for (uint corner = 0u; corner < numCorners; ++corner) {
        float2 atlasUV = (lumenProbeAtlasCoord(probeIndices[corner], probeSize.x) * kLumenIrradianceProbeWithBorderResolution + probeUVCoord) / atlasExtent;
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
        StructuredBuffer<uint2> probes = resolveUniform(params.probePositions);
        LumenProbeSample sample = calculateLumenAdaptiveInterpolation(thread.xy, viewSize,
            uint2(params.probeWidth, params.probeHeight), receiver, normal, position, probes, params.frameIndex);
        uint4 probeIndices = sample.indices;
        // UE 5.7.4 InterpolateAndIntegrate, static lighting with zero moving fraction.
        encodedAlpha = sample.lightingValid ? 0.004f : 0.0f;
#if defined(METALLIC_LUMEN_STOCHASTIC_INTERPOLATION)
        // UE 5.7.4 STOCHASTIC_PROBE_INTERPOLATION and the rendering-commandlet BlackDummy.
        bool sampleProbes = any(sample.weights > 0.0f);
        float randomValue = min(0.0f, 0.99f);
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
        readback.indices = probeIndices;
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
            indirect = irradianceOverPi * diffuse.Load(int3(thread.xy, 0)).rgb;
        }
    }
#if defined(METALLIC_LUMEN_PROBE_INTERPOLATION_READBACK)
    uint2 readbackPixel = thread.xy - params.interpolationReadbackPixels.xy;
    if (all(readbackPixel < params.interpolationReadbackPixels.zw)) {
        uint index = readbackPixel.y * params.interpolationReadbackPixels.z + readbackPixel.x;
        resolveUniform(params.interpolationReadback)[index] = readback;
    }
#endif
    indirectDiffuse[thread.xy] = float4(indirect, encodedAlpha);
    color[thread.xy] = float4(directColor.Load(int3(thread.xy, 0)).rgb + indirect, 1.0f);
}
