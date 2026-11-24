module LumenReflection;

import ShaderCore;
import ParameterRoot;
import ScenePathMath;
import GlobalDistanceField;
import GlobalDistanceFieldPages;
import LumenGlobalRay;
import LumenMeshSDFGrid;
import LumenMeshSDFTrace;
import LumenProbeMath;
import LumenScreenTrace;
import MeshDistanceField;

__include "LumenReflectionTemporal.slang";
__include "LumenReflectionBilateral.slang";

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

struct LumenReflectionScreenBindings
{
    ResourceHandle<StructuredBuffer<float>> closestHZB;
    ResourceHandle<Texture2D<float>> previousDepth;
    ResourceHandle<Texture2D<float4>> previousColor;
    ResourceHandle<StructuredBuffer<LumenScreenTraceView>> traceView;
};

struct LumenReflectionMeshSDFBindings
{
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> atlas;
    SamplerHandle atlasSampler;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
    ResourceHandle<StructuredBuffer<uint2>> gridCells;
    ResourceHandle<StructuredBuffer<uint>> gridIndices;
    uint4 grid;
    float3 cameraPosition;
    float maxMeshSDFTraceDistance;
    float cardTraceEndDistance;
    uint bricksPerAxis;
};

struct LumenReflectionParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<Texture2D<float4>> position, normal, specular, sceneColor;
    ResourceHandle<Texture2D<float2>> blueNoise;
    ResourceHandle<RWTexture2D<float4>> outputReflections, trace, color;
    ResourceHandle<Texture2D<float4>> reflections;
    ViewCameraConstants camera;
    uint width, height, clipmapCount, frameIndex;
    float maxTraceDistance, maxRoughness;
    ResourceHandle<RWTexture2D<float4>> outputRayData;
    ResourceHandle<RWTexture2D<float>> outputRayHitDistance;
    ResourceHandle<RWTexture2D<float4>> outputResolvedReflections;
    ResourceHandle<RWTexture2D<float>> outputResolvedHitDistance;
    ResourceHandle<Texture2D<float4>> rayData;
    ResourceHandle<Texture2D<float>> rayHitDistance;
    ResourceHandle<Texture2D<float4>> bilateralReflections;
    LumenReflectionScreenBindings screen;
    LumenReflectionMeshSDFBindings detail;
#if defined(METALLIC_LUMEN_REFLECTION_TILED_TRACING)
    uint2 basePadding;
    ResourceHandle<StructuredBuffer<uint>> activeTiles, clearTiles, workCounts, rayList;
    uint maxDispatchX;
#endif
};

static const float kGGXSamplingBias = 0.1f;
static const float kMinReflectionConeAngle = 0.00001f;
static const float kSurfaceBias = 0.05f; // UE SurfaceBias defaults to 5 cm.
static const float kMaxRayIntensity = 40.0f;
static const float kInvRoughnessFadeLength = 1.0f / 0.1f;
static const uint kReflectionScreenHit = 5u;
static const uint kReflectionMeshSDFHit = 9u;
static const float4 kEmptyReflectionTrace = float4(-1.0f, float(kGlobalDistanceFieldEmpty), -1.0f, 0.0f);
static const float3 kInvalidReflectionLighting = float3(65024.0f, 65024.0f, 64512.0f);

// UE 5.7.4 MonteCarlo.ush, isotropic material tangent basis.
float3x3 getReflectionTangentBasis(float3 tangentZ)
{
    const float signZ = tangentZ.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -rcp(signZ + tangentZ.z);
    const float b = tangentZ.x * tangentZ.y * a;
    float3 tangentX = float3(1.0f + signZ * a * (tangentZ.x * tangentZ.x), signZ * b, -signZ * tangentZ.x);
    float3 tangentY = float3(b, signZ + a * (tangentZ.y * tangentZ.y), -tangentZ.y);
    return float3x3(tangentX, tangentY, tangentZ);
}

// UE 5.7.4 MonteCarlo.ush, VisibleGGXPDF_aniso overload with
// GGX_BOUNDED_VNDF_SAMPLING and bLimitVDNFToReflection enabled.
float visibleReflectionGGXPDF(float3 v, float3 h, float2 alpha)
{
    float noV = v.z;
    float noH = h.z;
    float voH = dot(v, h);
    float a2 = alpha.x * alpha.y;
    float3 hs = float3(alpha.y * h.x, alpha.x * h.y, a2 * noH);
    float s = dot(hs, hs);
    float d = (1.0f / kLumenProbePi) * a2 * ((a2 / s) * (a2 / s));
    float lenV = length(float3(v.x * alpha.x, v.y * alpha.y, noV));
    float a = saturate(min(alpha.x, alpha.y));
    float bound = 1.0f + length(v.xy);
    float ka2 = a * a, s2 = bound * bound;
    float k = (s2 - ka2 * s2) / (s2 + ka2 * v.z * v.z);
    return (2.0f * d * voH) / (k * noV + lenV);
}

// UE 5.7.4 MonteCarlo.ush, ImportanceSampleVisibleGGX with the bounded reflection sampling range.
float4 importanceSampleVisibleReflectionGGX(float2 e, float2 alpha, float3 v)
{
    float3 vh = normalize(float3(alpha * v.xy, v.z));
    float phi = (2.0f * kLumenProbePi) * e.x;
    float a = saturate(min(alpha.x, alpha.y));
    float s = 1.0f + length(v.xy);
    float a2 = a * a, s2 = s * s;
    float k = (s2 - a2 * s2) / (s2 + a2 * v.z * v.z);
    float z = lerp(1.0f, -k * vh.z, e.y);
    float sinTheta = sqrt(saturate(1.0f - z * z));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 h = float3(x, y, z) + vh;
    h = normalize(float3(alpha * h.xy, max(0.0f, h.z)));
    return float4(h, visibleReflectionGGXPDF(v, h, alpha));
}

// UE 5.7.4 LumenReflections.usf, full-resolution, opaque non-Substrate,
// isotropic single-lobe ray generation. Distance quantization is in centimeters.
float4 generateReflectionRay(LumenReflectionParams params, uint2 pixel, float3 worldPosition, float3 normal,
    float roughness)
{
    float3 cameraVector = params.camera.upProjection.w > 0.5f
        ? ScenePath.cameraForwardFrom(params.camera.eye, params.camera.center)
        : normalize(worldPosition - params.camera.eye.xyz);

    float3 rayDirection;
    float coneAngle = 0.0f;
    if (roughness < 0.001f) {
        rayDirection = reflect(cameraVector, normal);
    } else {
        float2 e = lumenProbeBlueNoiseVec2(resolveUniform(params.blueNoise), pixel, params.frameIndex % 1024u);
        e.y *= 1.0f - kGGXSamplingBias;
        float3x3 tangentBasis = getReflectionTangentBasis(normal);
        float3 tangentV = mul(tangentBasis, -cameraVector);
        float2 alpha = (roughness * roughness).xx;
        float4 ggxSample = importanceSampleVisibleReflectionGGX(e, alpha, tangentV);
        float3 worldH = mul(ggxSample.xyz, tangentBasis);
        rayDirection = reflect(cameraVector, worldH);
        coneAngle = 1.0f / max(ggxSample.w, 0.0001f);
    }
    coneAngle = max(coneAngle, kMinReflectionConeAngle);

    // Reading the reference PF_FloatRGBA ray buffer rounds the direction and cone.
    return f16tof32(f32tof16(float4(rayDirection, coneAngle)));
}

float quantizeReflectionDistanceCm(float distanceCm)
{
    return f16tof32(f32tof16(distanceCm));
}

float reflectionTraceDistanceThroughR16(float distance)
{
    return quantizeReflectionDistanceCm(distance * 100.0f) * 0.01f;
}

float reflectionTraceDistanceCm(LumenReflectionParams params)
{
    return min(params.maxTraceDistance * 100.0f, 65504.0f);
}

float reflectionQuantizedMaxDistance(float maxTraceDistanceCm)
{
    return f16tof32(f32tof16(maxTraceDistanceCm) & 0x7fffu) * 0.01f;
}

#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES)
bool traceReflectionScreen(LumenReflectionParams params, uint2 pixel, float3 position, float3 normal,
    float3 direction, float maxTraceDistance, out float3 lighting, out float hitDistance)
{
    const LumenScreenTraceInputs screen = {resolveUniform(params.screen.closestHZB),
        resolveUniform(params.screen.previousDepth), resolveUniform(params.screen.previousColor),
        resolveUniform(params.screen.traceView)};
    return traceLumenReflectionScreenRay(screen, pixel, position, normal, direction, maxTraceDistance,
        params.frameIndex, lighting, hitDistance);
}
#endif

#if defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
bool traceReflectionDetail(LumenReflectionParams params, LumenGlobalTraceScene scene, uint2 pixel, float4 position,
    float3 samplePosition, float3 rayDirection, float maxTraceDistance, float traceStart,
    out float3 lighting, out float opaqueHitDistance, out uint cardState)
{
    const LumenReflectionMeshSDFBindings detail = params.detail;
    const float maxMeshSDFTraceDistance = detail.maxMeshSDFTraceDistance;
    const float cardTraceEndDistance = detail.cardTraceEndDistance;
    const uint cellIndex = lumenMeshSDFGridCellIndex(pixel, position.w, detail.grid.xyz);
    const uint2 cell = resolveUniform(detail.gridCells)[cellIndex];
    const MeshDistanceFieldResources meshes = {resolveUniform(detail.mips), resolveUniform(detail.indirection),
        resolveUniform(detail.atlas), resolveUniform(detail.atlasSampler), detail.bricksPerAxis};
    const LumenMeshSDFTraceScene meshScene = {meshes, resolveUniform(detail.instances),
        resolveUniform(detail.gridIndices), scene.cache};
    const LumenMeshSDFTraceInput meshRay = {samplePosition, rayDirection,
        max(traceStart - kSurfaceBias, 0.0f), maxTraceDistance, maxMeshSDFTraceDistance,
        cell.x, cell.y, false};
    const LumenMeshSDFTraceResult meshHit = traceLumenMeshSDFs(meshScene, meshRay);
    opaqueHitDistance = meshHit.hitDistance;
    const float distanceFromViewpoint = length(detail.cameraPosition - position.xyz);
    float distanceFade = saturate(4.0f * distanceFromViewpoint / cardTraceEndDistance - 3.0f);
    distanceFade = max(distanceFade, saturate((opaqueHitDistance - maxMeshSDFTraceDistance * .7f) /
        (maxMeshSDFTraceDistance * .3f)));
    const float transparency = lerp(meshHit.transparency, 1.0f, distanceFade);
    lighting = meshHit.radiance;
    cardState = meshHit.cardCovered ? kCacheSampled : kCacheUncovered;
    return transparency < screenTraceNoise(float2(pixel) + 0.5f, 0.0f);
}
#endif

float3 traceReflectionVoxels(LumenReflectionParams params, LumenGlobalTraceScene scene, uint2 pixel, float3 position,
    float3 samplePosition, float4 rayData, float traceHitDistance, float maxTraceDistance, float maxTraceDistanceCm,
    out float4 debugTrace, out float rayHitDistance)
{
    float startDistance = 0.0f;
#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES) || defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
    // UE TraceVoxels pulls back four half-voxel extents from the previous method's endpoint.
    const uint clipmapForSurfaceExpand = computeGlobalDistanceFieldClipmapIndex(scene.domains, scene.clipmapCount,
        position + traceHitDistance * rayData.xyz);
    const float pullbackForSurfaceExpand = scene.domains[clipmapForSurfaceExpand].minAndVoxelSize.w * 2.0f;
    startDistance = max(traceHitDistance - pullbackForSurfaceExpand, 0.0f);
#endif
    const float stepFactorNoise = lerp(0.95f, 1.0f / 0.95f, screenTraceNoise(float2(pixel),
        float(params.frameIndex % 8u)));
    const float stepFactor = lerp(stepFactorNoise, 1.0f, saturate(rayData.w / (kLumenProbePi / 256.0f)));
    const LumenGlobalTraceRay ray = {float4(samplePosition, startDistance), float4(rayData.xyz, maxTraceDistance)};
    const LumenGlobalTraceResult hit = traceLumenReflectionGlobalRay(scene, ray, stepFactor);
    rayHitDistance = hit.state.x == kGlobalDistanceFieldHit ? -max(hit.metrics.x * 100.0f, 0.0f) : maxTraceDistanceCm;
    debugTrace = float4(hit.metrics.x, float(hit.state.x),
        hit.state.y == ~0u ? -1.0f : float(hit.state.y), float(hit.state.w));
    return hit.radiance.xyz;
}

float3 traceSoftwareReflection(LumenReflectionParams params, uint2 pixel, float4 position, float3 normal,
    float4 rayData, out float4 debugTrace, out float rayHitDistance)
{
    const float maxTraceDistanceCm = reflectionTraceDistanceCm(params);
    const float maxTraceDistance = reflectionQuantizedMaxDistance(maxTraceDistanceCm);
    float traceHitDistance = 0.0f;
#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES)
    float3 screenLighting;
    if (traceReflectionScreen(params, pixel, position.xyz, normal, rayData.xyz, maxTraceDistance,
            screenLighting, traceHitDistance)) {
        debugTrace = float4(traceHitDistance, float(kReflectionScreenHit), -1.0f, 0.0f);
        rayHitDistance = -max(traceHitDistance * 100.0f, 0.0f);
        return screenLighting;
    }
    traceHitDistance = reflectionTraceDistanceThroughR16(traceHitDistance);
#endif

    const float3 samplePosition = position.xyz + kSurfaceBias * rayData.xyz;
    const LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
#if defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
    // UE 5.7.4 ReflectionCompactTracesCS, then TraceMeshSDFs for the unresolved near rays.
    const LumenReflectionMeshSDFBindings detail = params.detail;
    if (scene.identityMatches && detail.grid.w != 0u && position.w < detail.cardTraceEndDistance &&
        traceHitDistance <= detail.maxMeshSDFTraceDistance) {
        float3 lighting;
        float opaqueHitDistance;
        uint cardState;
        if (traceReflectionDetail(params, scene, pixel, position, samplePosition, rayData.xyz, maxTraceDistance,
                traceHitDistance, lighting, opaqueHitDistance, cardState)) {
            debugTrace = float4(opaqueHitDistance, float(kReflectionMeshSDFHit), -1.0f, float(cardState));
            rayHitDistance = -max(opaqueHitDistance * 100.0f, 0.0f);
            return lighting;
        }
        traceHitDistance = reflectionTraceDistanceThroughR16(opaqueHitDistance);
    }
#endif
    return traceReflectionVoxels(params, scene, pixel, position.xyz, samplePosition, rayData, traceHitDistance,
        maxTraceDistance, maxTraceDistanceCm, debugTrace, rayHitDistance);
}

float reflectionRoughnessFade(LumenReflectionParams params, float roughness)
{
    // UE 5.7.4 LumenReflectionsCombine.ush, opaque single-lobe fade.
    return saturate((params.maxRoughness - roughness) * kInvRoughnessFadeLength);
}

float4 clampReflectionLighting(float3 lighting, float fadeAlpha)
{
    float maxLighting = max(lighting.x, max(lighting.y, lighting.z));
    if (maxLighting > kMaxRayIntensity) {
        lighting *= kMaxRayIntensity / maxLighting;
    }
    return float4(lighting, fadeAlpha);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionTraceMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    const float4 specular = resolveUniform(params.specular).Load(int3(pixel, 0));
    const float fadeAlpha = reflectionRoughnessFade(params, specular.w);
    float4 reflections = 0.0f;
    float4 debugTrace = kEmptyReflectionTrace;
    float4 rayData = 0.0f;
    float rayHitDistance = 0.0f;
    if (position.w > 0.0f && any(specular.xyz > 0.0f) && fadeAlpha > 0.0f) {
        const float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
        rayData = generateReflectionRay(params, pixel, position.xyz, normal, specular.w);

        float3 lighting = traceSoftwareReflection(params, pixel, position, normal, rayData, debugTrace, rayHitDistance);
        reflections = clampReflectionLighting(lighting, fadeAlpha);
    }
    resolveUniform(params.outputReflections)[pixel] = reflections;
    resolveUniform(params.trace)[pixel] = debugTrace;
    resolveUniform(params.outputRayData)[pixel] = rayData;
    resolveUniform(params.outputRayHitDistance)[pixel] = rayHitDistance;
}

#if defined(METALLIC_LUMEN_REFLECTION_TILED_TRACING)
bool getReflectionTilePixel(LumenReflectionParams params, uint2 group, uint2 lane, bool clear, out uint2 pixel)
{
    const uint tileCount = resolveUniform(params.workCounts)[clear ? 1u : 0u];
    const uint tileIndex = group.y * min(tileCount, params.maxDispatchX) + group.x;
    pixel = 0;
    if (tileIndex >= tileCount) { return false; }
    const uint tile = clear ? resolveUniform(params.clearTiles)[tileIndex] : resolveUniform(params.activeTiles)[tileIndex];
    pixel = uint2(tile & 0xfffu, tile >> 12u) * 8u + lane;
    return all(pixel < uint2(params.width, params.height));
}

bool getReflectionCompactedPixel(LumenReflectionParams params, uint2 group, uint lane, uint countIndex, out uint2 pixel)
{
    const uint rayCount = resolveUniform(params.workCounts)[countIndex];
    const uint groupCount = (rayCount + 63u) / 64u;
    const uint rayIndex = (group.y * min(groupCount, params.maxDispatchX) + group.x) * 64u + lane;
    pixel = 0;
    if (rayIndex >= rayCount) { return false; }
    const uint ray = resolveUniform(params.rayList)[rayIndex];
    pixel = uint2(ray & 0x3fffu, ray >> 14u);
    return true;
}

void initializeTiledReflectionTrace(LumenReflectionParams params, uint2 pixel, float4 rayData, float fadeAlpha)
{
    resolveUniform(params.outputReflections)[pixel] = float4(0.0f, 0.0f, 0.0f, fadeAlpha);
    resolveUniform(params.trace)[pixel] = kEmptyReflectionTrace;
    resolveUniform(params.outputRayData)[pixel] = rayData;
    resolveUniform(params.outputRayHitDistance)[pixel] = 0.0f;
}

void writeTiledReflectionTrace(LumenReflectionParams params, uint2 pixel, float3 lighting, float4 debugTrace)
{
    const float roughness = resolveUniform(params.specular).Load(int3(pixel, 0)).w;
    resolveUniform(params.outputReflections)[pixel] = clampReflectionLighting(lighting, reflectionRoughnessFade(params, roughness));
    resolveUniform(params.trace)[pixel] = debugTrace;
}

// UE 5.7.4 ReflectionGenerateRaysCS, reading the full-resolution tracing tiles.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionGenerateTiledRaysMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionTilePixel(params, group.xy, lane.xy, false, pixel)) { return; }

    const float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    const float4 specular = resolveUniform(params.specular).Load(int3(pixel, 0));
    float fadeAlpha = reflectionRoughnessFade(params, specular.w);
    float4 rayData = 0.0f;
    if (position.w > 0.0f && any(specular.xyz > 0.0f) && fadeAlpha > 0.0f) {
        const float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
        rayData = generateReflectionRay(params, pixel, position.xyz, normal, specular.w);
    } else {
        fadeAlpha = 0.0f;
    }
    initializeTiledReflectionTrace(params, pixel, rayData, fadeAlpha);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionClearTilesMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionTilePixel(params, group.xy, lane.xy, true, pixel)) { return; }

    initializeTiledReflectionTrace(params, pixel, 0.0f, 0.0f);
    resolveUniform(params.outputResolvedReflections)[pixel] = float4(kInvalidReflectionLighting, 0.0f);
    resolveUniform(params.outputResolvedHitDistance)[pixel] = asfloat(0x7f7fffffu);
}

#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES)
// UE 5.7.4 ReflectionTraceScreenTexturesCS, preserving the miss endpoint in centimeters.
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionTraceScreenTiledMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionTilePixel(params, group.xy, lane.xy, false, pixel)) { return; }
    const float4 rayData = resolveUniform(params.rayData).Load(int3(pixel, 0));
    if (rayData.w <= 0.0f) { return; }

    const float3 position = resolveUniform(params.position).Load(int3(pixel, 0)).xyz;
    const float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
    const float maxTraceDistance = reflectionQuantizedMaxDistance(reflectionTraceDistanceCm(params));
    float3 lighting;
    float hitDistance;
    const bool hit = traceReflectionScreen(params, pixel, position, normal, rayData.xyz, maxTraceDistance, lighting, hitDistance);
    if (hit) {
        writeTiledReflectionTrace(params, pixel, lighting, float4(hitDistance, float(kReflectionScreenHit), -1.0f, 0.0f));
    }
    const float hitDistanceCm = hitDistance * 100.0f;
    // Use the explicit half conversion of the fused handoff before the R16F store.
    resolveUniform(params.outputRayHitDistance)[pixel] = hit ? -max(hitDistanceCm, 0.0f) : quantizeReflectionDistanceCm(hitDistanceCm);
}
#endif

#if defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
// UE 5.7.4 ReflectionTraceMeshSDFsCS, using the first compacted ray list.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenReflectionTraceDetailTiledMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionCompactedPixel(params, group.xy, lane, 2u, pixel)) { return; }
    const LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
    if (!scene.identityMatches || params.detail.grid.w == 0u) { return; }

    const float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    const float3 direction = resolveUniform(params.rayData).Load(int3(pixel, 0)).xyz;
    const float3 samplePosition = position.xyz + kSurfaceBias * direction;
    const float maxTraceDistance = reflectionQuantizedMaxDistance(reflectionTraceDistanceCm(params));
    let rayHitDistance = resolveUniform(params.outputRayHitDistance);
    float3 lighting;
    float hitDistance;
    uint cardState;
    const bool hit = traceReflectionDetail(params, scene, pixel, position, samplePosition, direction, maxTraceDistance,
        rayHitDistance[pixel] * 0.01f, lighting, hitDistance, cardState);
    if (hit) {
        writeTiledReflectionTrace(params, pixel, lighting, float4(hitDistance, float(kReflectionMeshSDFHit), -1.0f, float(cardState)));
    }
    const float hitDistanceCm = hitDistance * 100.0f;
    rayHitDistance[pixel] = hit ? -max(hitDistanceCm, 0.0f) : quantizeReflectionDistanceCm(hitDistanceCm);
}
#endif

// UE 5.7.4 ReflectionTraceVoxelsCS, using the remaining compacted rays.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenReflectionTraceGDFTiledMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionCompactedPixel(params, group.xy, lane, 3u, pixel)) { return; }
    const float3 position = resolveUniform(params.position).Load(int3(pixel, 0)).xyz;
    const float4 rayData = resolveUniform(params.rayData).Load(int3(pixel, 0));
    const float3 samplePosition = position + kSurfaceBias * rayData.xyz;
    const LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
    const float maxTraceDistanceCm = reflectionTraceDistanceCm(params);
    const float maxTraceDistance = reflectionQuantizedMaxDistance(maxTraceDistanceCm);
    let rayHitDistance = resolveUniform(params.outputRayHitDistance);
    float4 debugTrace;
    float hitDistanceCm;
    const float3 lighting = traceReflectionVoxels(params, scene, pixel, position, samplePosition, rayData,
        rayHitDistance[pixel] * 0.01f, maxTraceDistance, maxTraceDistanceCm, debugTrace, hitDistanceCm);
    writeTiledReflectionTrace(params, pixel, lighting, debugTrace);
    rayHitDistance[pixel] = hitDistanceCm;
}
#endif

static const uint kSpatialReconstructionNumSamples = 5u;
static const float kSpatialReconstructionKernelRadius = 8.0f;
static const float kDenoiserOneOverTonemapRange = 1.0f / 10.0f;

// UE 5.7.4 MonteCarlo.ush, UniformSampleDiskConcentric.
float2 uniformSampleDiskConcentric(float2 e)
{
    float2 p = 2.0f * e - 0.99999994f;
    float2 a = abs(p);
    float lo = min(a.x, a.y);
    float hi = max(a.x, a.y);
    float epsilon = 5.42101086243e-20f;
    float phi = (kLumenProbePi / 4.0f) * (lo / (hi + epsilon) + 2.0f * float(a.y >= a.x));
    const uint signMask = 0x80000000u;
    float2 disk = asfloat((asuint(float2(cos(phi), sin(phi))) & ~signMask) | (asuint(p) & signMask));
    return disk * hi;
}

// UE 5.7.4 LumenReflectionResolve.usf, GetSampleWeight and BRDF.ush, D_GGX.
float getReflectionSampleWeight(float3 rayDirection, float rayPDF, float3x3 tangentBasis, float3 v, float a2)
{
    float3 h = mul(tangentBasis, normalize(v + rayDirection));
    float d = (h.z * a2 - h.z) * h.z + 1.0f;
    float sampleNDF = a2 / (kLumenProbePi * d * d);
    return sampleNDF / rayPDF;
}

// UE 5.7.4 LumenReflectionDenoiserCommon.ush. Callers supply valid lighting in the working color space.
float3 reflectionRGBToDenoiserSpace(float3 lighting)
{
    float weight = 1.0f / (1.0f + WorkingColor.luminance(lighting) * kDenoiserOneOverTonemapRange);
    return lighting * weight;
}

float3 reflectionDenoiserSpaceToRGB(float3 lighting)
{
    float invWeight = 1.0f / (1.0f - WorkingColor.luminance(lighting) * kDenoiserOneOverTonemapRange);
    return lighting * invWeight;
}

// UE 5.7.4 LumenReflectionResolve.usf, 1x1, opaque single-lobe spatial reconstruction.
void resolveReflectionPixel(LumenReflectionParams params, uint2 pixel)
{
    uint2 extent = uint2(params.width, params.height);

    let reflections = resolveUniform(params.reflections);
    let rayData = resolveUniform(params.rayData);
    float4 centerRay = rayData.Load(int3(pixel, 0));
    float4 centerReflection = reflections.Load(int3(pixel, 0));
    float3 centerLighting = kInvalidReflectionLighting;
    float minRayHitT = asfloat(0x7f7fffffu);
    // Trace writes a positive cone for every emitted ray, including valid black misses.
    if (centerRay.w > 0.0f) {
        let rayHitDistance = resolveUniform(params.rayHitDistance);
        centerLighting = reflectionRGBToDenoiserSpace(centerReflection.rgb);
        float centerWeightSum = 1.0f;
        minRayHitT = abs(rayHitDistance.Load(int3(pixel, 0)));

        float roughness = resolveUniform(params.specular).Load(int3(pixel, 0)).w;
        float kernelRadiusInPixels = 0.0f;
        if (roughness > 0.05f) {
            kernelRadiusInPixels = kSpatialReconstructionKernelRadius * saturate(roughness * 8.0f);
        }

        if (kernelRadiusInPixels > 1.0f) {
            uint2 randomSeed = lumenRand3DPCG16(int3(pixel, params.frameIndex % 8u)).xy;
            let position = resolveUniform(params.position);
            float3 translatedWorldPosition = position.Load(int3(pixel, 0)).xyz - params.camera.eye.xyz;
            float3 v = -normalize(translatedWorldPosition);
            float3x3 tangentBasis = getReflectionTangentBasis(
                normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz));
            float effectiveRoughness = clamp(roughness, 0.0f, 1.0f);
            float alpha = effectiveRoughness * effectiveRoughness;
            float a2 = alpha * alpha;
            float2 kernelMajorAxis = float2(kernelRadiusInPixels, 0.0f);
            float2 kernelMinorAxis = float2(0.0f, kernelRadiusInPixels);

            float centerRayHitT = minRayHitT;
            float centerSampleWeight = getReflectionSampleWeight(centerRay.xyz, 1.0f / abs(centerRay.w),
                tangentBasis, v, a2);
            centerSampleWeight = max(centerSampleWeight, 0.001f);
            centerLighting *= centerSampleWeight;
            centerWeightSum = centerSampleWeight;

            float3 neighborLighting = 0.0f;
            float neighborWeightSum = 0.0f;
            for (uint neighborIndex = 0u; neighborIndex < kSpatialReconstructionNumSamples; ++neighborIndex) {
                float2 neighborOffsetInRect = lumenHammersley16(neighborIndex, kSpatialReconstructionNumSamples, randomSeed);
                float2 neighborOffset = uniformSampleDiskConcentric(neighborOffsetInRect);
                neighborOffset = neighborOffset.x * kernelMajorAxis + neighborOffset.y * kernelMinorAxis;
                int2 neighborPixel = int2(float2(pixel) + neighborOffset + 0.5f);
                if (all(neighborPixel >= 0) && all(neighborPixel < int2(extent))) {
                    float4 neighborRay = rayData.Load(int3(neighborPixel, 0));
                    if (neighborRay.w > 0.0f) {
                        float3 neighborTranslatedWorldPosition = position.Load(int3(neighborPixel, 0)).xyz - params.camera.eye.xyz;
                        float traceHitDistance = abs(rayHitDistance.Load(int3(neighborPixel, 0)));
                        traceHitDistance = min(traceHitDistance, centerRayHitT);

                        float3 neighborHitPosition = neighborTranslatedWorldPosition + neighborRay.xyz * (traceHitDistance * 0.01f);
                        float distanceToNeighborHit = length(neighborHitPosition - translatedWorldPosition);
                        float3 directionToNeighborHit = neighborRay.xyz;
                        if (distanceToNeighborHit > 0.0f) {
                            directionToNeighborHit = (neighborHitPosition - translatedWorldPosition) / distanceToNeighborHit;
                        }

                        float sampleWeight = getReflectionSampleWeight(directionToNeighborHit, 1.0f / abs(neighborRay.w),
                            tangentBasis, v, a2);
                        if (sampleWeight > 1e-6f) {
                            float3 sampleRadiance = reflectionRGBToDenoiserSpace(reflections.Load(int3(neighborPixel, 0)).rgb);
                            neighborLighting += sampleRadiance * sampleWeight;
                            neighborWeightSum += sampleWeight;
                            minRayHitT = min(minRayHitT, traceHitDistance);
                        }
                    }
                }
            }

            // The UE 5.7.4 default SpatialReconstruction.MinWeight is zero.
            centerLighting += neighborLighting;
            centerWeightSum += neighborWeightSum;
        }

        if (centerWeightSum > 1e-6f) {
            centerLighting = centerLighting / centerWeightSum;
        }
        centerLighting = reflectionDenoiserSpaceToRGB(centerLighting);
    }

    resolveUniform(params.outputResolvedReflections)[pixel] = float4(centerLighting, centerReflection.a);
    resolveUniform(params.outputResolvedHitDistance)[pixel] = minRayHitT;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionResolveMain(uint3 thread : SV_DispatchThreadID)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    if (any(thread.xy >= uint2(params.width, params.height))) { return; }
    resolveReflectionPixel(params, thread.xy);
}

#if defined(METALLIC_LUMEN_REFLECTION_TILED_TRACING)
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionResolveTiledMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    const LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel;
    if (!getReflectionTilePixel(params, group.xy, lane.xy, false, pixel)) { return; }
    resolveReflectionPixel(params, pixel);
}
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionCompositeMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const float4 sceneColor = resolveUniform(params.sceneColor).Load(int3(pixel, 0));
    const float4 reflection = resolveUniform(params.bilateralReflections).Load(int3(pixel, 0));
    float3 lighting = 0.0f;
    // UE 5.7.4 ReflectionEnvironmentPixelShader.usf, standalone Lumen
    // reflection composite. The supported permutation has no sky or probe fallback.
    if (reflection.a > 0.0f) {
        const float3 specular = resolveUniform(params.specular).Load(int3(pixel, 0)).xyz;
        lighting = reflection.rgb * reflection.a;
        lighting *= specular;
    }
    resolveUniform(params.color)[pixel] = float4(sceneColor.rgb - min(-lighting, 0.0f), sceneColor.a);
}
