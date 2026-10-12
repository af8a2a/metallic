// Shared vertex transition. Callers own traversal and the compact transport state.
// Explicit control results preserve termination, alpha retry and debug-return behavior.
enum OpenPBRPathControl { Stop, Next, Return };

OpenPBRPathControl shadeOpenPBRSurfacePathVertex(
    HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    inout OpenPBRInteriorStack interiorStack, inout bool captureSpecularHitDistance
#if METALLIC_PT_SHARC
    , inout OpenPBRCacheContext cache
#endif
)
{
    float3 shadingNormal = hit.normal;
    RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth);
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
    float3 viewDirection = safeNormalize(-ray.direction, hit.normal);
    uint replacedMask = 0u;
#if METALLIC_PT_SHARC
    OpenPBRCacheAddress cacheAddress = (OpenPBRCacheAddress)0;
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
#endif
    SurfaceSamplingContext sampling;
    sampling.throughput = throughput;
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = openPbrExteriorIorForHit(interiorStack, hit);
    let prepared = prepareRaySurface(materialEval, work, sampling);

    float3 emission = max(prepared.emission(), float3(0.0));
    float3 localRadiance = emission;
    if (material.emissive.w != 0.0) {
        radiance += throughput * localRadiance;
#if METALLIC_PT_SHARC
        openPbrCacheAddLighting(cache, localRadiance);
#endif
        return OpenPBRPathControl.Stop;
    }

    PathSurfaceLights lights;
    lights.push = push;
    lights.hit = hit;
    lights.surfaceIor = openPbrMaterialIor(material);
    lights.transmittingSurface = true;
    lights.interiorStack = interiorStack;
    lights.viewDirection = viewDirection;
    lights.seed = seed;
    // The environment proposal and ordinary secondary BSDF miss exclude
    // celestial disks, so their separate finite NEE estimator has weight one.
    // Local delta sources retain their source-selection PDF.
    OpenPBRDiffuseSpecular localComponents = openPbrDirectComponents(prepared, lights, replacedMask);
    seed = lights.seed;

    EnvironmentLightSample lightSample;
    bool directLightingEnabled = !openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting);
    bool sampledEnvironment = directLightingEnabled && sampleEnvironmentLight(push, seed, lightSample);
#if SHARC_QUERY && METALLIC_PT_SINGLE_SAMPLE
    // Visibility does not consume the path RNG. Finish all remaining BSDF work
    // after the environment proposal, so the prepared closure need not survive
    // its nested transparent-shadow traversal. Keep lighting/termination order.
    float lightBsdfPdf = 0.0;
    OpenPBRDiffuseSpecular lightComponents = openPBRMakeZeroDiffuseSpecular();
    if (sampledEnvironment) {
        lightBsdfPdf = prepared.pdf(lightSample.direction);
        lightComponents = openPbrSelectComponents(
            prepared.evalProjectedComponents(lightSample.direction), replacedMask);
    }
    OpenPBRDiffuseSpecular scatterComponents;
    BSDFWeightSample scatter = (BSDFWeightSample)0;
    // This diagnostic returns before sampling in the general path as well.
    if (depth != 0u || push.debugView != kOpenPBRDebugViewShadowTransmittance) {
        scatter = prepared.sampleWeightedComponents(float3(rand(seed), rand(seed), rand(seed)), scatterComponents);
    }
#endif
    if (sampledEnvironment) {
#if !(SHARC_QUERY && METALLIC_PT_SINGLE_SAMPLE)
        float lightBsdfPdf = prepared.pdf(lightSample.direction);
        OpenPBRDiffuseSpecular lightComponents = openPbrSelectComponents(
            prepared.evalProjectedComponents(lightSample.direction), replacedMask);
#endif
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

    localRadiance = emission + openPBRGetSumOfDiffuseSpecular(localComponents);
    radiance += throughput * localRadiance;
#if METALLIC_PT_SHARC
    bool cacheRegistered = openPbrCacheRecordHit(cache, cacheAddress, localComponents, emission);
#endif

#if !(SHARC_QUERY && METALLIC_PT_SINGLE_SAMPLE)
    OpenPBRDiffuseSpecular scatterComponents;
    BSDFWeightSample scatter = prepared.sampleWeightedComponents(float3(rand(seed), rand(seed), rand(seed)), scatterComponents);
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

    sampledDirection = safeNormalize(sampledDirection, hit.normal);
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

#if METALLIC_PT_SHARC && SHARC_UPDATE
    openPbrCacheScatter(cache, scatterComponents, cacheRegistered);
#endif
    if (depth >= 2u) {
        float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
        if (rand(seed) > continueProbability) {
            return OpenPBRPathControl.Stop;
        }
        throughput /= max(continueProbability, 0.001);
#if METALLIC_PT_SHARC && SHARC_UPDATE
        openPbrCacheSetThroughput(cache, 1.0 / max(continueProbability, 0.001));
#endif
    }
    return OpenPBRPathControl.Next;
}

#if METALLIC_CUSTOM_MATERIALS
OpenPBRPathControl shadeSlabSurfacePathVertex<TProgram : ISceneSurfaceMaterialProgram>(
    inout Material<TProgram> material, HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    OpenPBRInteriorStack interiorStack, inout bool captureSpecularHitDistance
#if METALLIC_PT_SHARC
    , inout OpenPBRCacheContext cache
#endif
)
{
    float3 authoredNormal = hit.normal;
    RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth);
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
    lights.viewDirection = safeNormalize(-ray.direction, hit.normal);
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
#if METALLIC_PT_SHARC
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
    ray.direction = safeNormalize(scatter.wiWS, hit.normal);
    ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(ray.direction, hit.geometryNormal));
#if METALLIC_PT_SHARC && SHARC_UPDATE
    openPbrCacheSetThroughput(cache, max(scatter.weight, float3(0)));
#endif
    if (depth >= 2u) {
        float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
        if (rand(seed) > continueProbability) { return OpenPBRPathControl.Stop; }
        throughput /= max(continueProbability, 0.001);
#if METALLIC_PT_SHARC && SHARC_UPDATE
        openPbrCacheSetThroughput(cache, 1.0 / max(continueProbability, 0.001));
#endif
    }
    return OpenPBRPathControl.Next;
}
#endif

OpenPBRPathControl shadeOpenPBRPathVertex(
    bool found, HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, inout uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    inout OpenPBRInteriorStack interiorStack,
    inout float specularHitDistance, inout bool captureSpecularHitDistance,
    inout uint alphaLayers, inout float3 primaryEndpoint,
    float3 primaryOrigin, inout bool primarySurface
#if METALLIC_PT_SHARC
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
#if METALLIC_PT_SHARC
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
#if METALLIC_PT_SHARC && SHARC_UPDATE
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
#if METALLIC_PT_SHARC && SHARC_UPDATE
        float3 beforeRadiance = radiance;
        float3 beforeThroughput = throughput;
#endif
        bool continued = shadeSceneFiber(sceneFiberParameters(transport), hit, stableNormal, push, depth,
            ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta);
#if METALLIC_PT_SHARC && SHARC_UPDATE
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
#if METALLIC_PT_SHARC
            , cache
#endif
        );
    }
    if (family == 2u) {
        var material = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, rayCone, ray.direction, seed, false, true);
        return shadeSlabSurfacePathVertex(material, hit, push, ray, rayCone, seed, depth,
            radiance, throughput, previousBsdfPdf, previousScatterDelta,
            interiorStack, captureSpecularHitDistance
#if METALLIC_PT_SHARC
            , cache
#endif
        );
    }
#endif
    return shadeOpenPBRSurfacePathVertex(hit, push,
        ray, rayCone, seed, depth, radiance, throughput, previousBsdfPdf,
        previousScatterDelta, interiorStack, captureSpecularHitDistance
#if METALLIC_PT_SHARC
        , cache
#endif
    );
}
