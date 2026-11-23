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
};

static const float kGGXSamplingBias = 0.1f;
static const float kMinReflectionConeAngle = 0.00001f;
static const float kSurfaceBias = 0.05f; // UE SurfaceBias defaults to 5 cm.
static const float kMaxRayIntensity = 40.0f;
static const float kInvRoughnessFadeLength = 1.0f / 0.1f;
static const uint kReflectionScreenHit = 5u;
static const uint kReflectionMeshSDFHit = 9u;

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

float reflectionTraceDistanceThroughR16(float distance)
{
    return f16tof32(f32tof16(distance * 100.0f)) * 0.01f;
}

float3 traceSoftwareReflection(LumenReflectionParams params, uint2 pixel, float4 position, float3 normal,
    float4 rayData, out float4 debugTrace, out float rayHitDistance)
{
    const float maxTraceDistanceCm = min(params.maxTraceDistance * 100.0f, 65504.0f);
    const float maxTraceDistance = f16tof32(f32tof16(maxTraceDistanceCm) & 0x7fffu) * 0.01f;
    float traceHitDistance = 0.0f;
#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES)
    const LumenScreenTraceInputs screen = {resolveUniform(params.screen.closestHZB),
        resolveUniform(params.screen.previousDepth), resolveUniform(params.screen.previousColor),
        resolveUniform(params.screen.traceView)};
    float3 screenLighting;
    if (traceLumenReflectionScreenRay(screen, pixel, position.xyz, normal, rayData.xyz, maxTraceDistance,
            params.frameIndex, screenLighting, traceHitDistance)) {
        debugTrace = float4(traceHitDistance, float(kReflectionScreenHit), -1.0f, 0.0f);
        rayHitDistance = -max(traceHitDistance * 100.0f, 0.0f);
        return screenLighting;
    }
    traceHitDistance = reflectionTraceDistanceThroughR16(traceHitDistance);
#endif

    const float3 samplePosition = position.xyz + kSurfaceBias * rayData.xyz;
    const LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
#if defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
    const LumenReflectionMeshSDFBindings detail = params.detail;
    const float maxMeshSDFTraceDistance = detail.maxMeshSDFTraceDistance;
    const float cardTraceEndDistance = detail.cardTraceEndDistance;
    // UE 5.7.4 ReflectionCompactTracesCS, then TraceMeshSDFs for the unresolved near rays.
    if (scene.identityMatches && detail.grid.w != 0u && position.w < cardTraceEndDistance &&
        traceHitDistance <= maxMeshSDFTraceDistance) {
        const uint cellIndex = lumenMeshSDFGridCellIndex(pixel, position.w, detail.grid.xyz);
        const uint2 cell = resolveUniform(detail.gridCells)[cellIndex];
        const MeshDistanceFieldResources meshes = {resolveUniform(detail.mips), resolveUniform(detail.indirection),
            resolveUniform(detail.atlas), resolveUniform(detail.atlasSampler), detail.bricksPerAxis};
        const LumenMeshSDFTraceScene meshScene = {meshes, resolveUniform(detail.instances),
            resolveUniform(detail.gridIndices), scene.cache};
        const LumenMeshSDFTraceInput meshRay = {samplePosition, rayData.xyz,
            max(traceHitDistance - kSurfaceBias, 0.0f), maxTraceDistance, maxMeshSDFTraceDistance,
            cell.x, cell.y, false};
        const LumenMeshSDFTraceResult meshHit = traceLumenMeshSDFs(meshScene, meshRay);
        const float opaqueHitDistance = meshHit.hitDistance;
        const float distanceFromViewpoint = length(detail.cameraPosition - position.xyz);
        float distanceFade = saturate(4.0f * distanceFromViewpoint / cardTraceEndDistance - 3.0f);
        distanceFade = max(distanceFade, saturate((opaqueHitDistance - maxMeshSDFTraceDistance * .7f) /
            (maxMeshSDFTraceDistance * .3f)));
        const float transparency = lerp(meshHit.transparency, 1.0f, distanceFade);
        if (transparency < screenTraceNoise(float2(pixel) + 0.5f, 0.0f)) {
            const uint cardState = meshHit.cardCovered ? kCacheSampled : kCacheUncovered;
            debugTrace = float4(opaqueHitDistance, float(kReflectionMeshSDFHit), -1.0f, float(cardState));
            rayHitDistance = -max(opaqueHitDistance * 100.0f, 0.0f);
            return meshHit.radiance;
        }
        traceHitDistance = reflectionTraceDistanceThroughR16(opaqueHitDistance);
    }
#endif

    float startDistance = 0.0f;
#if defined(METALLIC_LUMEN_REFLECTION_SCREEN_TRACES) || defined(METALLIC_LUMEN_REFLECTION_DETAIL_TRACING)
    // UE TraceVoxels pulls back four half-voxel extents from the previous method's endpoint.
    const uint clipmapForSurfaceExpand = computeGlobalDistanceFieldClipmapIndex(scene.domains, scene.clipmapCount,
        position.xyz + traceHitDistance * rayData.xyz);
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

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionTraceMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    if (any(pixel >= uint2(params.width, params.height))) { return; }

    const float4 position = resolveUniform(params.position).Load(int3(pixel, 0));
    const float4 specular = resolveUniform(params.specular).Load(int3(pixel, 0));
    // UE 5.7.4 LumenReflectionsCombine.ush, opaque single-lobe fade.
    const float fadeAlpha = saturate((params.maxRoughness - specular.w) * kInvRoughnessFadeLength);
    float4 reflections = 0.0f;
    float4 debugTrace = float4(-1.0f, float(kGlobalDistanceFieldEmpty), -1.0f, 0.0f);
    float4 rayData = 0.0f;
    float rayHitDistance = 0.0f;
    if (position.w > 0.0f && any(specular.xyz > 0.0f) && fadeAlpha > 0.0f) {
        const float3 normal = normalize(resolveUniform(params.normal).Load(int3(pixel, 0)).xyz);
        rayData = generateReflectionRay(params, pixel, position.xyz, normal, specular.w);

        float3 lighting = traceSoftwareReflection(params, pixel, position, normal, rayData, debugTrace, rayHitDistance);
        float maxLighting = max(lighting.x, max(lighting.y, lighting.z));
        if (maxLighting > kMaxRayIntensity) {
            lighting *= kMaxRayIntensity / maxLighting;
        }
        reflections = float4(lighting, fadeAlpha);
    }
    resolveUniform(params.outputReflections)[pixel] = reflections;
    resolveUniform(params.trace)[pixel] = debugTrace;
    resolveUniform(params.outputRayData)[pixel] = rayData;
    resolveUniform(params.outputRayHitDistance)[pixel] = rayHitDistance;
}

static const uint kSpatialReconstructionNumSamples = 5u;
static const float kSpatialReconstructionKernelRadius = 8.0f;
static const float kDenoiserOneOverTonemapRange = 1.0f / 10.0f;
static const float3 kInvalidReflectionLighting = float3(65024.0f, 65024.0f, 64512.0f);

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
[shader("compute")]
[numthreads(8, 8, 1)]
void lumenReflectionResolveMain(uint3 thread : SV_DispatchThreadID)
{
    LumenReflectionParams params = getParameters<LumenReflectionParams>();
    uint2 pixel = thread.xy;
    uint2 extent = uint2(params.width, params.height);
    if (any(pixel >= extent)) { return; }

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
