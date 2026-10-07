#include "OpenPBRSurface.slang"
#include "ScenePathTrace.slang"

static const uint kOpenPBRInteriorStackCapacity = 8u;

struct OpenPBRInteriorStackEntry
{
    uint materialIndex;
    float ior;
};

struct OpenPBRInteriorStack
{
    OpenPBRInteriorStackEntry entries[kOpenPBRInteriorStackCapacity];
    uint count;
};

OpenPBRInteriorStack makeOpenPBRInteriorStack()
{
    OpenPBRInteriorStack stack;
    stack.count = 0u;
    return stack;
}

float openPbrMaterialIor(PathTraceMaterial material)
{
    return clamp(material.glassParams.y, 1.0, 3.0);
}

float openPbrInteriorStackTopIor(OpenPBRInteriorStack stack)
{
    return stack.count > 0u ? stack.entries[stack.count - 1u].ior : OpenPBRVacuumIor;
}

float openPbrInteriorStackNextIor(OpenPBRInteriorStack stack)
{
    return stack.count > 1u ? stack.entries[stack.count - 2u].ior : OpenPBRVacuumIor;
}

float openPbrExteriorIorForHit(OpenPBRInteriorStack stack, HitInfo hit)
{
    if (hit.frontFace) {
        return openPbrInteriorStackTopIor(stack);
    }
    if (stack.count > 0u && stack.entries[stack.count - 1u].materialIndex == hit.materialIndex) {
        return openPbrInteriorStackNextIor(stack);
    }
    return openPbrInteriorStackTopIor(stack);
}

float3 openPbrVolumeTransmittance(OpenPBRInteriorStack stack, float distance)
{
    if (stack.count == 0u || distance <= 0.0) {
        return float3(1.0, 1.0, 1.0);
    }

    PathTraceMaterial mediumMaterial = loadSceneMaterial(stack.entries[stack.count - 1u].materialIndex);
    float attenuationDistance = max(mediumMaterial.glassParams.w, 0.0);
    if (attenuationDistance <= 0.0001) {
        return float3(1.0, 1.0, 1.0);
    }

    float3 attenuationColor = clamp(
        mediumMaterial.attenuationColor.rgb,
        float3(0.0000001, 0.0000001, 0.0000001),
        float3(1.0, 1.0, 1.0));
    float exponent = max(distance, 0.0) / attenuationDistance;
    return float3(
        pow(attenuationColor.x, exponent),
        pow(attenuationColor.y, exponent),
        pow(attenuationColor.z, exponent));
}

void openPbrInteriorStackPush(inout OpenPBRInteriorStack stack, uint materialIndex, float ior)
{
    if (stack.count > 0u && stack.entries[stack.count - 1u].materialIndex == materialIndex) {
        stack.entries[stack.count - 1u].ior = ior;
        return;
    }
    if (stack.count >= kOpenPBRInteriorStackCapacity) {
        stack.entries[kOpenPBRInteriorStackCapacity - 1u].materialIndex = materialIndex;
        stack.entries[kOpenPBRInteriorStackCapacity - 1u].ior = ior;
        return;
    }

    stack.entries[stack.count].materialIndex = materialIndex;
    stack.entries[stack.count].ior = ior;
    stack.count += 1u;
}

void openPbrInteriorStackPopMaterial(inout OpenPBRInteriorStack stack, uint materialIndex)
{
    if (stack.count == 0u) {
        return;
    }
    if (stack.entries[stack.count - 1u].materialIndex == materialIndex) {
        stack.count -= 1u;
        return;
    }

    for (uint i = stack.count; i > 0u; --i) {
        uint entryIndex = i - 1u;
        if (stack.entries[entryIndex].materialIndex == materialIndex) {
            for (uint j = entryIndex; j + 1u < stack.count; ++j) {
                stack.entries[j] = stack.entries[j + 1u];
            }
            stack.count -= 1u;
            return;
        }
    }
}

void openPbrUpdateInteriorStackAfterTransmission(
    inout OpenPBRInteriorStack stack,
    HitInfo hit,
    PathTraceMaterial material)
{
    if (hit.frontFace) {
        openPbrInteriorStackPush(stack, hit.materialIndex, openPbrMaterialIor(material));
    } else {
        openPbrInteriorStackPopMaterial(stack, hit.materialIndex);
    }
}

float3 openPbrRayOffsetNormal(float3 direction, float3 geometryNormal)
{
    return dot(direction, geometryNormal) >= 0.0 ? geometryNormal : -geometryNormal;
}

static const uint kOpenPBRShadowTransmissionMaxHits = 16u;

float3 openPbrShadowTransmittance(
    float3 origin,
    float3 direction,
    float maxDistance,
    OpenPBRInteriorStack interiorStack,
    ScenePathTracePush push)
{
    Ray ray;
    ray.origin = origin;
    ray.direction = safeNormalize(direction, float3(0.0, 1.0, 0.0));
    float remainingDistance = max(maxDistance, 0.001);
    float3 transmittance = float3(1.0, 1.0, 1.0);

    for (uint hitIndex = 0u; hitIndex < kOpenPBRShadowTransmissionMaxHits; ++hitIndex) {
        ScenePathTracePush shadowPush = push;
        shadowPush.clipOrtho.x = 0.001;
        shadowPush.clipOrtho.y = remainingDistance;
        HitInfo shadowHit;
        if (!traceClosest(ray, shadowPush, shadowHit) || shadowHit.t >= remainingDistance) {
            return transmittance;
        }

        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableVolumeAttenuation)) {
            transmittance *= openPbrVolumeTransmittance(interiorStack, shadowHit.t);
        }
        PathTraceMaterial shadowMaterial = loadSceneMaterial(shadowHit.materialIndex);
        if (shadowMaterial.textureParams.w > 1.5) {
            float alpha = evaluateSceneCoverage(shadowHit.materialIndex, shadowMaterial, shadowHit.texcoord, push).opacity;
            transmittance *= 1.0 - alpha;
            remainingDistance -= shadowHit.t;
            ray.origin = offsetRay(shadowHit.position, openPbrRayOffsetNormal(ray.direction, shadowHit.geometryNormal));
            if (remainingDistance <= 0.001 || luminanceMax(transmittance) <= 0.0001) { return transmittance; }
            continue;
        }
        float4 transmissionSample = loadMaterialTextureMip0(
            shadowMaterial.transmissionTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        float4 metallicRoughnessSample = loadMaterialTextureMip0(
            shadowMaterial.metallicRoughnessTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures)) {
            transmissionSample = float4(1.0, 1.0, 1.0, 1.0);
            metallicRoughnessSample = float4(1.0, 1.0, 1.0, 1.0);
        }
        float metallic = saturate(shadowMaterial.params.x * metallicRoughnessSample.b);
        float surfaceTransmission =
            saturate(shadowMaterial.glassParams.x * transmissionSample.r) *
            (1.0 - metallic);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableTransmission)) {
            surfaceTransmission = 0.0;
        }
        if (surfaceTransmission <= 0.01) {
            return float3(0.0, 0.0, 0.0);
        }

        float4 baseColorSample = loadMaterialTextureMip0(
            shadowMaterial.baseColorTexture,
            shadowHit.texcoord,
            float4(1.0, 1.0, 1.0, 1.0),
            push);
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures)) {
            baseColorSample = float4(1.0, 1.0, 1.0, 1.0);
        }
        float3 baseColor = saturate(
            materialTextureModulate(shadowMaterial.baseColor.rgb, baseColorSample.rgb, shadowMaterial.baseColorTexture.transform0.w));
        transmittance *= surfaceTransmission * powColor(baseColor, 0.5);
        if (luminanceMax(transmittance) <= 0.0001) {
            return float3(0.0, 0.0, 0.0);
        }

        // This NEE visibility term uses straight-through transmission; fully
        // refracted connections remain the responsibility of BSDF-sampled paths.
        // An exit interface must not collapse visibility to an opaque black band.
        openPbrUpdateInteriorStackAfterTransmission(
            interiorStack,
            shadowHit,
            shadowMaterial);
        remainingDistance -= shadowHit.t;
        ray.origin = offsetRay(
            shadowHit.position,
            openPbrRayOffsetNormal(ray.direction, shadowHit.geometryNormal));
    }

    return float3(0.0, 0.0, 0.0);
}

struct PathSurfaceLights : ISurfaceLightProvider
{
    ScenePathTracePush push;
    HitInfo hit;
    PathTraceMaterial material;
    OpenPBRInteriorStack interiorStack;
    float3 viewDirection;
    uint seed;
    uint count() { return openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting) ? 0u : kCelestialLightCount + 1u; }
    [mutating]
    SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result = (SurfaceLightSample)0;
        if (index < kCelestialLightCount) {
            PunctualLightSample light = sampleCelestialLightAtPosition(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights), index, hit.position);
            result.direction = light.direction;
            result.illuminance = light.illuminance;
            result.distance = light.distance;
            result.inverseSourcePdf = any(light.illuminance > 0.0) ? 1.0 : 0.0;
            result.source = celestialSourceIdentity(index);
            return result;
        }
        uint source;
        float inversePdf;
        if (selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), hit.position, seed, source, inversePdf)) {
            PunctualLightSample light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), source, hit.position);
            result.direction = light.direction;
            result.illuminance = light.illuminance;
            result.distance = light.distance;
            result.inverseSourcePdf = inversePdf;
            result.source = source;
        }
        return result;
    }
    float3 visibility(SurfaceLightSample light)
    {
        if ((light.source & kCelestialSourceBit) != 0u &&
            !celestialLightCastsShadow(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights),
                light.source & ~kCelestialSourceBit)) { return float3(1.0); }
        float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
        OpenPBRInteriorStack shadowStack = interiorStack;
        if (dot(viewDirection, hit.geometryNormal) * dot(light.direction, hit.geometryNormal) < 0.0) {
            openPbrUpdateInteriorStackAfterTransmission(shadowStack, hit, material);
        }
        float distance = max(light.distance - length(origin - hit.position), 0.0);
        return openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) ? float3(1)
            : openPbrShadowTransmittance(origin, light.direction, distance, shadowStack, push);
    }
};

float3 pathTraceOpenPBRWithPrimaryHit(
    Ray ray,
    RayCone rayCone,
    ScenePathTracePush push,
    inout uint seed,
    out float specularHitDistance,
    bool hasPrimaryHit,
    HitInfo primaryHit)
{
    float3 radiance = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);
    float previousBsdfPdf = 0.0;
    bool previousScatterDelta = true;
    uint maxDepth = clamp(push.maxDepth, 1u, 32u);
    OpenPBRInteriorStack interiorStack = makeOpenPBRInteriorStack();
    specularHitDistance = 0.0;
    bool captureSpecularHitDistance = false;
    uint alphaLayers = 0u;
    float3 primaryEndpoint = float3(0.0);
    float3 primaryOrigin = ray.origin;
    bool primarySurface = false;

    for (uint depth = 0u; depth < maxDepth; ++depth) {
        HitInfo hit;
        bool found;
        if (depth == 0u && hasPrimaryHit) {
            hit = primaryHit;
            found = true;
        } else {
            found = traceClosest(ray, push, hit);
        }
        if (!found) {
            // Diagnostic colors carry display/data encoding, not scene radiance.
            if (push.debugView != kOpenPBRDebugViewFinal) { return float3(0.0); }
            if (captureSpecularHitDistance) {
                specularHitDistance = 65504.0;
                captureSpecularHitDistance = false;
            }
            float misWeight = depth == 0u
                ? 1.0
                : environmentBsdfMisWeight(push, previousBsdfPdf, ray.direction, previousScatterDelta);
            radiance += throughput * environmentRadiance(ray.direction, push, depth == 0u, primaryOrigin) * misWeight;
            break;
        }
        if (captureSpecularHitDistance) {
            specularHitDistance = hit.t;
            captureSpecularHitDistance = false;
        }
        if (depth == 0u) { primaryEndpoint = hit.position; primarySurface = true; }

        rayCone = propagateRayCone(rayCone, hit.t);
        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableVolumeAttenuation)) {
            throughput *= openPbrVolumeTransmittance(interiorStack, hit.t);
        }
        if (luminanceMax(throughput) <= 0.000001) {
            break;
        }

        PathTraceMaterial material = loadSceneMaterial(hit.materialIndex);
        if (materialDomain(material) == kMaterialDomainFiber) {
            const float3 stableNormal = hit.normal;
            if (dot(hit.normal, ray.direction) > 0) { hit.normal = -hit.normal; }
            if (!shadeSceneFiber(material, hit, stableNormal, push, depth,
                    ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta)) { break; }
            continue;
        }
        if (material.textureParams.w > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {
            float alpha = evaluateSceneCoverage(hit.materialIndex, material, hit.texcoord, push).opacity;
            if (rand(seed) >= alpha) {
                if (++alphaLayers >= 16u) { break; }
                ray.origin = offsetRay(hit.position, openPbrRayOffsetNormal(ray.direction, hit.geometryNormal));
                hasPrimaryHit = false;
                if (depth == 0u) { primarySurface = false; }
                --depth;
                continue;
            }
        }
        float3 shadingNormal = hit.normal;
        RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth);
        SceneOpenPBRMaterialProgram program = makeOpenPBRMaterialProgram(hit, push, rayCone, ray.direction, seed, false, true);
        // Always dispatch from this hit's instance. Raster specialization and
        // primary PreparedClosure are never inherited by a secondary hit.
        let evaluated = evaluateRaySurface(program, work);
        let materialEval = evaluated.closure;
        // Value inputs (including textured volume absorption) belong to this hit.
        material = program.material;
        hit = program.hit;
        seed = program.seed;
        float4 normalSample = program.normalSample;
        if (depth == 0u &&
            push.debugView != kOpenPBRDebugViewFinal &&
            push.debugView != kOpenPBRDebugViewShadowTransmittance) {
            return openPbrPrimaryDebugColor(
                push,
                hit,
                shadingNormal,
                normalSample,
                materialEval,
                ray.direction);
        }
        float3 viewDirection = safeNormalize(-ray.direction, hit.normal);
        SurfaceSamplingContext sampling;
        sampling.throughput = throughput;
        sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
        sampling.exteriorIor = openPbrExteriorIorForHit(interiorStack, hit);
        let prepared = prepareRaySurface(materialEval, work, sampling);

        radiance += throughput * max(prepared.emission(), float3(0.0, 0.0, 0.0));
        if (material.emissive.w != 0.0) { break; }

        PathSurfaceLights lights;
        lights.push = push;
        lights.hit = hit;
        lights.material = material;
        lights.interiorStack = interiorStack;
        lights.viewDirection = viewDirection;
        lights.seed = seed;
        // Delta light sources use MIS weight one. The shared kernel retains
        // the original throughput-first multiplication and selection PDF.
        radiance = integrateDirectLighting(prepared, lights, radiance, throughput);
        seed = lights.seed;

        EnvironmentLightSample lightSample;
        bool directLightingEnabled = !openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting);
        bool sampledEnvironment = directLightingEnabled && sampleEnvironmentLight(push, seed, lightSample);
        if (sampledEnvironment) {
            float lightBsdfPdf = prepared.pdf(lightSample.direction);
            float3 lightWeight = evaluateSurfaceProjected(prepared, lightSample.direction);
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
                return shadowTransmittance;
            }
            if (luminanceMax(shadowTransmittance) > 0.0) {
                float lightMisWeight = environmentLightMisWeight(
                    push,
                    lightSample.pdfSolidAngle,
                    lightBsdfPdf);
                radiance += throughput *
                    lightWeight *
                    materialEval.occlusion *
                    shadowTransmittance *
                    (lightSample.radiance * lightMisWeight / max(lightSample.pdfSolidAngle, 0.000001));
            }
        }
        if (depth == 0u && push.debugView == kOpenPBRDebugViewShadowTransmittance) {
            return sampledEnvironment
                ? float3(0.0, 0.0, 0.0)
                : float3(1.0, 0.0, 1.0);
        }

        BSDFWeightSample scatter = sampleSurfaceWeight(prepared, float3(rand(seed), rand(seed), rand(seed)));
        if (scatter.pdf <= 0.0 || scatter.flags == 0u) { break; }
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

        if (depth >= 2u) {
            float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
            if (rand(seed) > continueProbability) {
                break;
            }
            throughput /= max(continueProbability, 0.001);
        }
    }

    return primarySurface && push.debugView == kOpenPBRDebugViewFinal
        ? applyPrimaryAerialPerspective(radiance, primaryEndpoint, primaryOrigin) : radiance;
}

// Preserve the ray-primary reference API; the raster path supplies an authored
// first hit while sharing BSDF sampling, nested IORs and volume attenuation.
float3 pathTraceOpenPBR(Ray ray, RayCone rayCone, ScenePathTracePush push,
    inout uint seed, out float specularHitDistance)
{
    HitInfo unusedHit = {};
    return pathTraceOpenPBRWithPrimaryHit(ray, rayCone, push, seed,
        specularHitDistance, false, unusedHit);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    bool debugViewEnabled = push.debugView != kOpenPBRDebugViewFinal;
    uint samples = debugViewEnabled ? 1u : clamp(push.samples, 1u, 16u);
    // Keep diagnostic environment/shadow samples stable between frames.
    uint sampleFrame = debugViewEnabled ? 0u : push.sampleFrame;
    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (sampleFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = push.temporalJitter != 0u
            ? float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY)
            : (samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed)));
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        float ignoredSpecularHitDistance;
        color += pathTraceOpenPBR(ray, rayCone, push, seed, ignoredSpecularHitDistance);
    }
    color /= float(samples);

    if (debugViewEnabled) {
        color = openPbrDisplayDebugColor(push, color);
    } else if (push.outputLinear == 0u) {
        color = Metallic.WorkingColor::toLinearRec709(color);
        color *= lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
        color = color / (color + float3(1.0, 1.0, 1.0));
        color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
        color = saturate(color);
    }

    if (push.enableAccumulation != 0u && !debugViewEnabled) {
        if (push.hasHistory != 0u) {
            float accumulationWeight = 1.0 / float(push.accumulationFrame + 1u);
            float3 previousColor = gHistoryPrevious[pixel].rgb;
            color = lerp(previousColor, color, accumulationWeight);
        }
        gHistoryCurrent[pixel] = float4(color, 1.0);
    }

    gOutput[pixel] = float4(color, 1.0);
}
