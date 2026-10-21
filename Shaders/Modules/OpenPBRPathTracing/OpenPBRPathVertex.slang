implementing OpenPBRPathTracing;

import Core;
import MaterialTextureDecode;
import RTXCRMaterialAdapter;
import RayMaterialExecution;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import SceneSurfaceMaterial;
import SurfaceLighting;
import ScenePathTracing;
import SHARC;
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.ScenePathTracing;
using Metallic.SHARC;

namespace Metallic.OpenPBRPathTracing
{
#if defined(METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION) && METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION != 0 && METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION != 1
#error METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION must be 0 or 1.
#endif
#if METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION
#if !defined(METALLIC_PT_SINGLE_SAMPLE) || METALLIC_PT_SINGLE_SAMPLE != 1 || !defined(SHARC_QUERY) || SHARC_QUERY != 1 || (defined(SHARC_UPDATE) && SHARC_UPDATE != 0)
#error Single cache-query specialization requires globally configured SINGLE1 QUERY1 UPDATE0.
#endif
#if METALLIC_PT_CACHE_QUERY_SPECIALIZATION || METALLIC_STANDARD_PT_CACHE_QUERY_SPECIALIZATION
#error Single cache-query specialization is independent of the General and standard cache markers.
#endif
#if defined(MATERIAL_CLASS) || defined(MATERIAL_PROGRAM_ID) || defined(METALLIC_DEFERRED_FP16) || defined(METALLIC_DEFERRED_RUNTIME_PRECISION) || defined(METALLIC_DEFERRED_CLASS_SPECIALIZATION)
#error Single cache-query specialization excludes manual material and Deferred shapes.
#endif
#if !defined(METALLIC_CUSTOM_MATERIALS) || METALLIC_CUSTOM_MATERIALS != 0 || !defined(METALLIC_GLOBAL_VIEW) || METALLIC_GLOBAL_VIEW != 0 || !defined(METALLIC_HAS_RTXCR) || METALLIC_HAS_RTXCR != 1
#error Single cache-query specialization requires the builtin local-view RTXCR family.
#endif
#if !defined(METALLIC_DEFERRED_LIGHT_GRID) || METALLIC_DEFERRED_LIGHT_GRID != 0 || !defined(METALLIC_REALTIME_DEFERRED) || METALLIC_REALTIME_DEFERRED != 0 || !defined(METALLIC_DEFERRED_UPSCALER_GUIDES) || METALLIC_DEFERRED_UPSCALER_GUIDES != 0
#error Single cache-query specialization excludes Deferred source families.
#endif
#if !defined(METALLIC_WORKING_SPACE_ACESCG) || METALLIC_WORKING_SPACE_ACESCG != 1
#error Single cache-query specialization is restricted to ACEScg.
#endif
// Cache policy only. SINGLE remains the original static global1 on both arms.
[SpecializationConstant]
public const bool kOpenPBRSingleCacheQueryEnabled = false;
#endif
// Shared vertex transition. Callers own traversal and the compact transport state.
// Explicit control results preserve termination, alpha retry and debug-return behavior.
public enum OpenPBRPathControl { Stop, Next, Return };

public OpenPBRPathControl shadeOpenPBRSurfacePathVertex(
    HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    inout OpenPBRInteriorStack interiorStack, inout bool captureSpecularHitDistance
#if (SHARC_UPDATE || SHARC_QUERY)
    , inout OpenPBRCacheContext cache
#endif
)
{
    float3 shadingNormal = hit.normal;
    RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth, false);
    var program = makeOpenPBRMaterialProgram(hit, push, rayCone, ray.direction, seed, false, true);
    // Always dispatch from this hit's instance. Raster specialization and
    // primary PreparedClosure are never inherited by a secondary hit.
    let evaluated = evaluateRaySurface(program, work);
    let materialEval = evaluated.closure;
    // Value inputs (including textured volume absorption) belong to this hit.
    OpenPBRMaterialParameters material = program.program.material;
    hit = program.program.hit;
    seed = program.program.seed;
    float4 normalSample = program.program.normalSample;
    if (depth == 0u &&
        push.debugView != kOpenPBRDebugViewFinal &&
        push.debugView != kOpenPBRDebugViewShadowTransmittance) {
        radiance = openPbrPrimaryDebugColor(
            push,
            hit,
            shadingNormal,
            normalSample,
            program.program.surfaceSummary(),
            ray.direction);
        return OpenPBRPathControl.Return;
    }
    float3 viewDirection = Metallic.ScenePath::safeNormalize(-ray.direction, hit.normal);
    uint replacedMask = 0u;
#if (SHARC_UPDATE || SHARC_QUERY)
#if METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION
    OpenPBRCacheAddress cacheAddress = (OpenPBRCacheAddress)0;
#if METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION
    if (kOpenPBRSingleCacheQueryEnabled) {
#endif
#if SHARC_UPDATE
    // Training registers hits even when depth, delta or footprint rejects reuse.
    bool queryAllowed = false;
#else
    bool queryAllowed = push.debugView == kOpenPBRDebugViewFinal &&
        openPbrCacheCanQuery(cache, hit, rayCone, depth, previousScatterDelta);
    if (queryAllowed)
#endif
    {
        cacheAddress = openPbrCacheAddress(cache, hit, materialEval.inputs, material, interiorStack, viewDirection);
    }
#if SHARC_UPDATE
    queryAllowed = push.debugView == kOpenPBRDebugViewFinal && cacheAddress.eligibleMask != 0u &&
        openPbrCacheCanQuery(cache, hit, rayCone, depth, previousScatterDelta);
#endif
    float3 cachedRadiance;
    if (queryAllowed) {
        uint queryMask = openPbrCacheQuery(cache, cacheAddress, cachedRadiance);
#if SHARC_UPDATE
        // Bootstrap only a complete suffix, after at least one exact bounce.
        if (cache.count > 0u && depth >= 2u && (queryMask | cacheAddress.absentMask) == 3u) {
            openPbrCacheAddLighting(cache, cachedRadiance);
            radiance += throughput * cachedRadiance;
            return OpenPBRPathControl.Stop;
        }
#else
        replacedMask = queryMask;
        radiance += throughput * cachedRadiance;
        if ((replacedMask | cacheAddress.absentMask) == 3u) {
            // The real first specular endpoint was recorded before lookup.
            return OpenPBRPathControl.Stop;
        }
#endif
    }
#if METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION
    }
#endif
#else
    OpenPBRCacheAddress cacheAddress = (OpenPBRCacheAddress)0;
#if METALLIC_PT_CACHE_QUERY_SPECIALIZATION
    if (kOpenPBRCacheQueryEnabled) {
#endif
#if SHARC_UPDATE
    // Training registers hits even when depth, delta or footprint rejects reuse.
    bool queryAllowed = false;
#else
    bool queryAllowed = push.debugView == kOpenPBRDebugViewFinal &&
        openPbrCacheCanQuery(cache, hit, rayCone, depth, previousScatterDelta);
    if (queryAllowed)
#endif
    {
        cacheAddress = openPbrCacheAddress(cache, hit, materialEval.inputs, material, interiorStack, viewDirection);
    }
#if SHARC_UPDATE
    queryAllowed = push.debugView == kOpenPBRDebugViewFinal && cacheAddress.eligibleMask != 0u &&
        openPbrCacheCanQuery(cache, hit, rayCone, depth, previousScatterDelta);
#endif
    float3 cachedRadiance;
    if (queryAllowed) {
        uint queryMask = openPbrCacheQuery(cache, cacheAddress, cachedRadiance);
#if SHARC_UPDATE
        // Bootstrap only a complete suffix, after at least one exact bounce.
        if (cache.count > 0u && depth >= 2u && (queryMask | cacheAddress.absentMask) == 3u) {
            openPbrCacheAddLighting(cache, cachedRadiance);
            radiance += throughput * cachedRadiance;
            return OpenPBRPathControl.Stop;
        }
#else
        replacedMask = queryMask;
        radiance += throughput * cachedRadiance;
        if ((replacedMask | cacheAddress.absentMask) == 3u) {
            // The real first specular endpoint was recorded before lookup.
            return OpenPBRPathControl.Stop;
        }
#endif
    }
#if METALLIC_PT_CACHE_QUERY_SPECIALIZATION
    }
#endif
#endif
#endif
    PathSurfaceLights lights;
    lights.push = push;
    lights.hit = hit;
    lights.surfaceIor = openPbrMaterialIor(material);
    lights.transmittingSurface = true;
    lights.interiorStack = interiorStack;
    lights.viewDirection = viewDirection;
    lights.seed = seed;
    OpenPBRDirectLighting directLighting;
    float3 emission;
    EnvironmentLightSample lightSample;
    bool sampledEnvironment;
    float lightBsdfPdf = 0.0;
    OpenPBRDiffuseSpecular lightComponents = openPBRMakeZeroDiffuseSpecular();
    OpenPBRDiffuseSpecular scatterComponents = openPBRMakeZeroDiffuseSpecular();
    BSDFWeightSample scatter = (BSDFWeightSample)0;
    {
        SurfaceSamplingContext sampling;
        sampling.throughput = throughput;
        sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
        sampling.exteriorIor = openPbrExteriorIorForHit(interiorStack, hit);
        let prepared = prepareRaySurface(materialEval, work, sampling);

        emission = max(prepared.emission(), float3(0.0));
        if (material.emissive.w != 0.0) {
            radiance += throughput * emission;
#if (SHARC_UPDATE || SHARC_QUERY)
            openPbrCacheAddLighting(cache, emission);
#endif
            return OpenPBRPathControl.Stop;
        }

        // The environment proposal and ordinary secondary BSDF miss exclude
        // celestial disks, so their separate finite NEE estimator has weight one.
        // Local delta sources retain their source-selection PDF.
        directLighting = openPbrPrepareDirectComponents(prepared, lights, replacedMask);
        seed = lights.seed;
        bool directLightingEnabled = !openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting);
        sampledEnvironment = directLightingEnabled && sampleEnvironmentLight(push, seed, lightSample);
        if (sampledEnvironment) {
            lightBsdfPdf = prepared.pdf(lightSample.direction);
            lightComponents = openPbrSelectComponents(
                prepared.evalProjectedComponents(lightSample.direction), replacedMask);
        }
        // Visibility consumes no path RNG. Finish the continuation after all
        // light proposals, before any shadow traversal, for every cache policy.
        // The primary diagnostic returns without continuation random numbers.
        if (depth != 0u || push.debugView != kOpenPBRDebugViewShadowTransmittance) {
            scatter = prepared.sampleWeightedComponents(float3(rand(seed), rand(seed), rand(seed)), scatterComponents);
        }
    }

    // Only sampled light data and continuation results cross these traversals.
    // Resolve shadows and accumulate direct sources in their original order.
    OpenPBRDiffuseSpecular localComponents = openPbrResolveDirectComponents(directLighting, lights);
    if (sampledEnvironment) {
        float3 lightWeight = openPBRGetSumOfDiffuseSpecular(lightComponents);
        float3 shadowTransmittance = float3(0.0, 0.0, 0.0);
        if (lightBsdfPdf > 0.0 && luminanceMax(lightWeight) > 0.0) {
            OpenPBRInteriorStack shadowInteriorStack = interiorStack;
            bool lightCrossesSurface =
                dot(viewDirection, hit.geometryNormal) *
                dot(lightSample.direction, hit.geometryNormal) < 0.0;
            if (lightCrossesSurface) {
                openPbrUpdateInteriorStackAfterTransmission(
                    shadowInteriorStack,
                    hit,
                    material);
            }
            float3 shadowOrigin = offsetRay(
                hit.position,
                openPbrRayOffsetNormal(lightSample.direction, hit.geometryNormal));
            if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) {
                shadowTransmittance = float3(1.0, 1.0, 1.0);
            } else if (openPbrDebugFlag(push, kOpenPBRDebugUseOpaqueShadows)) {
                shadowTransmittance = traceShadow(
                    shadowOrigin,
                    lightSample.direction,
                    10000.0,
                    push)
                    ? float3(0.0, 0.0, 0.0)
                    : float3(1.0, 1.0, 1.0);
            } else {
                shadowTransmittance = openPbrShadowTransmittance(
                    shadowOrigin,
                    lightSample.direction,
                    10000.0,
                    shadowInteriorStack,
                    push);
            }
        }
        if (depth == 0u && push.debugView == kOpenPBRDebugViewShadowTransmittance) {
            radiance = shadowTransmittance;
            return OpenPBRPathControl.Return;
        }
        if (luminanceMax(shadowTransmittance) > 0.0) {
            float lightMisWeight = environmentLightMisWeight(
                push,
                lightSample.pdfSolidAngle,
                lightBsdfPdf);
            float3 incident = materialEval.occlusion * shadowTransmittance *
                (lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001));
            localComponents = openPBRAddDiffuseSpecular(localComponents, openPBRScaleDiffuseSpecular(lightComponents, incident));
        }
    }
    if (depth == 0u && push.debugView == kOpenPBRDebugViewShadowTransmittance) {
        radiance = sampledEnvironment
            ? float3(0.0, 0.0, 0.0)
            : float3(1.0, 0.0, 1.0);
        return OpenPBRPathControl.Return;
    }

    float3 localRadiance = emission + openPBRGetSumOfDiffuseSpecular(localComponents);
    radiance += throughput * localRadiance;
#if (SHARC_UPDATE || SHARC_QUERY)
    bool cacheRegistered = openPbrCacheRecordHit(cache, cacheAddress, localComponents, emission);
#endif

    scatterComponents = openPbrSelectComponents(scatterComponents, replacedMask);
    scatter.weight = openPBRGetSumOfDiffuseSpecular(scatterComponents);
    if (scatter.pdf <= 0.0 || scatter.flags == 0u) { return OpenPBRPathControl.Stop; }
    float3 sampledDirection = scatter.wiWS;

    if (depth == 0u) {
        captureSpecularHitDistance =
            bool(scatter.flags & kBSDFReflection) &&
            bool(scatter.flags & (kBSDFGlossy | kBSDFDelta));
    }

    sampledDirection = Metallic.ScenePath::safeNormalize(sampledDirection, hit.normal);
    throughput *= max(scatter.weight, float3(0.0, 0.0, 0.0));
    previousBsdfPdf = scatter.pdf;
    previousScatterDelta = bool(scatter.flags & kBSDFDelta);
    if (!previousScatterDelta) {
        rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
    }
    if (bool(scatter.flags & kBSDFTransmission)) {
        openPbrUpdateInteriorStackAfterTransmission(interiorStack, hit, material);
    }
    ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(sampledDirection, hit.geometryNormal));
    ray.direction = sampledDirection;

#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
    openPbrCacheScatter(cache, scatterComponents, cacheRegistered);
#endif
    if (depth >= 2u) {
        float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
        if (rand(seed) > continueProbability) {
            return OpenPBRPathControl.Stop;
        }
        throughput /= max(continueProbability, 0.001);
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
        openPbrCacheSetThroughput(cache, 1.0 / max(continueProbability, 0.001));
#endif
    }
    return OpenPBRPathControl.Next;
}

#if METALLIC_CUSTOM_MATERIALS
public OpenPBRPathControl shadeSlabSurfacePathVertex<TProgram : ISceneSurfaceMaterialProgram>(
    inout Material<TProgram> material, HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    OpenPBRInteriorStack interiorStack, inout bool captureSpecularHitDistance
#if (SHARC_UPDATE || SHARC_QUERY)
    , inout OpenPBRCacheContext cache
#endif
)
{
    float3 authoredNormal = hit.normal;
    RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth, false);
    let evaluated = evaluateRaySurface(material, work);
    hit = material.program.evaluatedHit();
    seed = material.program.evaluatedSeed();
    let summary = material.program.surfaceSummary();
    if (depth == 0u && push.debugView != kOpenPBRDebugViewFinal &&
        push.debugView != kOpenPBRDebugViewShadowTransmittance) {
        radiance = openPbrPrimaryDebugColor(push, hit, authoredNormal,
            material.program.sampledNormal(), summary, ray.direction);
        return OpenPBRPathControl.Return;
    }
    SurfaceSamplingContext sampling;
    sampling.throughput = throughput;
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = openPbrExteriorIorForHit(interiorStack, hit);
    let prepared = prepareRaySurface(evaluated.closure, work, sampling);
    float3 localRadiance = max(prepared.surfaceEmission(evaluated.emission), float3(0));
    PathSurfaceLights lights;
    lights.push = push;
    lights.hit = hit;
    lights.surfaceIor = 1.0;
    lights.transmittingSurface = false;
    lights.interiorStack = interiorStack;
    lights.viewDirection = Metallic.ScenePath::safeNormalize(-ray.direction, hit.normal);
    lights.seed = seed;
    localRadiance = integrateDirectLighting(prepared, lights, localRadiance);
    seed = lights.seed;
    EnvironmentLightSample lightSample;
    bool sampledEnvironment = !openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting) &&
        sampleEnvironmentLight(push, seed, lightSample);
    float3 shadowTransmittance = float3(0);
    if (sampledEnvironment) {
        float density = prepared.pdf(lightSample.direction);
        float3 weight = prepared.evalProjected(lightSample.direction);
        if (density > 0.0 && luminanceMax(weight) > 0.0) {
            float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(lightSample.direction, hit.geometryNormal));
            if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) {
                shadowTransmittance = float3(1);
            } else if (openPbrDebugFlag(push, kOpenPBRDebugUseOpaqueShadows)) {
                shadowTransmittance = traceShadow(origin, lightSample.direction, 10000.0, push) ? float3(0) : float3(1);
            } else {
                shadowTransmittance = openPbrShadowTransmittance(origin, lightSample.direction,
                    10000.0, interiorStack, push);
            }
        }
        float misWeight = environmentLightMisWeight(push, lightSample.pdfSolidAngle, density);
        localRadiance += weight * shadowTransmittance * lightSample.radiance *
            (misWeight / max(lightSample.pdfSolidAngle, 0.000001));
    }
    if (depth == 0u && push.debugView == kOpenPBRDebugViewShadowTransmittance) {
        radiance = sampledEnvironment ? shadowTransmittance : float3(1, 0, 1);
        return OpenPBRPathControl.Return;
    }
    radiance += throughput * localRadiance;
#if (SHARC_UPDATE || SHARC_QUERY)
    // Slab never enters the OpenPBR component cache, but exact suffixes still
    // contribute to any earlier registered OpenPBR vertex.
    openPbrCacheAddLighting(cache, localRadiance);
#endif
    BSDFWeightSample scatter = sampleSurfaceWeight(prepared, float3(rand(seed), rand(seed), rand(seed)));
    if (scatter.pdf <= 0.0 || scatter.flags == 0u) { return OpenPBRPathControl.Stop; }
    if (depth == 0u) {
        captureSpecularHitDistance = bool(scatter.flags & kBSDFReflection) &&
            bool(scatter.flags & (kBSDFGlossy | kBSDFDelta));
    }
    throughput *= max(scatter.weight, float3(0));
    previousBsdfPdf = scatter.pdf;
    previousScatterDelta = bool(scatter.flags & kBSDFDelta);
    if (!previousScatterDelta) { rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf); }
    ray.direction = Metallic.ScenePath::safeNormalize(scatter.wiWS, hit.normal);
    ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(ray.direction, hit.geometryNormal));
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
    openPbrCacheSetThroughput(cache, max(scatter.weight, float3(0)));
#endif
    if (depth >= 2u) {
        float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
        if (rand(seed) > continueProbability) { return OpenPBRPathControl.Stop; }
        throughput /= max(continueProbability, 0.001);
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
        openPbrCacheSetThroughput(cache, 1.0 / max(continueProbability, 0.001));
#endif
    }
    return OpenPBRPathControl.Next;
}
#endif

public OpenPBRPathControl shadeOpenPBRPathVertex(
    bool found, HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, inout uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    inout OpenPBRInteriorStack interiorStack,
    inout float specularHitDistance, inout bool captureSpecularHitDistance,
    inout uint alphaLayers, inout float3 primaryEndpoint,
    float3 primaryOrigin, inout bool primarySurface
#if (SHARC_UPDATE || SHARC_QUERY)
    , inout OpenPBRCacheContext cache
#endif
)
{
    if (!found) {
        // Diagnostic colors carry display/data encoding, not scene radiance.
        if (push.debugView != kOpenPBRDebugViewFinal) {
            radiance = float3(0.0);
            return OpenPBRPathControl.Return;
        }
        if (captureSpecularHitDistance) {
            specularHitDistance = 65504.0;
            captureSpecularHitDistance = false;
        }
        float misWeight = depth == 0u
            ? 1.0
            : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
        float3 missRadiance = environmentRadiance(ray.direction, push, depth == 0u, primaryOrigin) * misWeight;
        if (depth > 0u && previousScatterDelta && push.environmentMode != kEnvironmentModeDisabled) {
            missRadiance += physicalDeltaMissDisks(getSceneEnvironmentResources(),
                ray.direction, ray.origin) * max(push.environmentIntensity, 0.0);
        }
        radiance += throughput * missRadiance;
#if (SHARC_UPDATE || SHARC_QUERY)
        openPbrCacheAddLighting(cache, missRadiance);
#endif
        return OpenPBRPathControl.Stop;
    }
    if (captureSpecularHitDistance) {
        specularHitDistance = hit.t;
        captureSpecularHitDistance = false;
    }
    if (depth == 0u) { primaryEndpoint = hit.position; primarySurface = true; }

    rayCone = propagateRayCone(rayCone, hit.t);
    if (!openPbrDebugFlag(push, kOpenPBRDebugDisableVolumeAttenuation)) {
        float3 volumeTransmittance = openPbrVolumeTransmittance(interiorStack, hit.t);
        throughput *= volumeTransmittance;
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
        openPbrCacheSetThroughput(cache, volumeTransmittance);
#endif
    }
    if (luminanceMax(throughput) <= 0.000001) {
        return OpenPBRPathControl.Stop;
    }

    SceneMaterialData transport = loadSceneMaterial(hit.materialIndex);
    if (materialDomain(transport) == kMaterialDomainFiber) {
        const float3 stableNormal = hit.normal;
        if (dot(hit.normal, ray.direction) > 0) { hit.normal = -hit.normal; }
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
        float3 beforeRadiance = radiance;
        float3 beforeThroughput = throughput;
#endif
        bool continued = shadeSceneFiber(sceneFiberParameters(transport), hit, stableNormal, push, depth,
            ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta);
#if (SHARC_UPDATE || SHARC_QUERY) && SHARC_UPDATE
        openPbrCacheAddLighting(cache, max(radiance - beforeRadiance, 0.0) / max(beforeThroughput, 1e-20));
        openPbrCacheSetThroughput(cache, throughput / max(beforeThroughput, 1e-20));
#endif
        if (!continued) { return OpenPBRPathControl.Stop; }
        return OpenPBRPathControl.Next;
    }
    if (transport.textureParams.w > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {
        float alpha = evaluateSceneCoverage(hit.materialIndex, sceneMaterialCoverageParameters(transport), hit.texcoord, push).opacity;
        if (rand(seed) >= alpha) {
            if (++alphaLayers >= 16u) { return OpenPBRPathControl.Stop; }
            ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(ray.direction, hit.geometryNormal));
            if (depth == 0u) { primarySurface = false; }
            --depth;
            return OpenPBRPathControl.Next;
        }
    }
#if METALLIC_CUSTOM_MATERIALS
    uint family = sceneSurfaceMaterialFamily(hit.materialIndex);
    if (family == 1u) {
        var material = makeSceneSlabMaterialProgram<SceneSingleSlabModel>(hit, push, rayCone, ray.direction, seed, false, true);
        return shadeSlabSurfacePathVertex(material, hit, push, ray, rayCone, seed, depth,
            radiance, throughput, previousBsdfPdf, previousScatterDelta,
            interiorStack, captureSpecularHitDistance
#if (SHARC_UPDATE || SHARC_QUERY)
            , cache
#endif
        );
    }
    if (family == 2u) {
        var material = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, rayCone, ray.direction, seed, false, true);
        return shadeSlabSurfacePathVertex(material, hit, push, ray, rayCone, seed, depth,
            radiance, throughput, previousBsdfPdf, previousScatterDelta,
            interiorStack, captureSpecularHitDistance
#if (SHARC_UPDATE || SHARC_QUERY)
            , cache
#endif
        );
    }
#endif
    return shadeOpenPBRSurfacePathVertex(hit, push,
        ray, rayCone, seed, depth, radiance, throughput, previousBsdfPdf,
        previousScatterDelta, interiorStack, captureSpecularHitDistance
#if (SHARC_UPDATE || SHARC_QUERY)
        , cache
#endif
    );
}
} // namespace Metallic.OpenPBRPathTracing
