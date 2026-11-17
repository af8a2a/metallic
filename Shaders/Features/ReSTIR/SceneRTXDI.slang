#language slang 2026
module SceneRTXDI;

// ReSTIR DI implementation for Metallic. Keep the fused pass entry here;
// camera, surfaces, light sampling/evaluation and reservoir reuse live below.
__include "SceneRTXDI/Common.slang";
__include "SceneRTXDI/Camera.slang";
__include "SceneRTXDI/LightSampling.slang";
__include "SceneRTXDI/Surface.slang";
__include "SceneRTXDI/LightEvaluation.slang";
__include "SceneRTXDI/Reservoir.slang";

import Core;
import Lighting;
import ScenePathTypes;
import NRDEncoding;
using Metallic;
using Metallic.Lighting;
using Metallic.Interop.NRDEncoding;

#define gOutput resolveUniform(getResourceParameters<SceneResourceParameters>().output)
#define gReservoirCurrent resolveUniform(getResourceParameters<SceneResourceParameters>().reservoirCurrent)
#define gReservoirPrevious resolveUniform(getResourceParameters<SceneResourceParameters>().reservoirPrevious)
#define gPositionCurrent resolveUniform(getResourceParameters<SceneResourceParameters>().positionCurrent)
#define gNormalCurrent resolveUniform(getResourceParameters<SceneResourceParameters>().normalCurrent)
#define gNoisyDiffuse resolveUniform(getResourceParameters<SceneResourceParameters>().noisyDiffuse)
#define gNoisySpecular resolveUniform(getResourceParameters<SceneResourceParameters>().noisySpecular)
#define gNrdNormalRoughness resolveUniform(getResourceParameters<SceneResourceParameters>().nrdNormalRoughness)
#define gNrdMotionVectors resolveUniform(getResourceParameters<SceneResourceParameters>().nrdMotion)
#define gNrdViewZ resolveUniform(getResourceParameters<SceneResourceParameters>().nrdViewZ)
#define gBaseColorMetalness resolveUniform(getResourceParameters<SceneResourceParameters>().baseColorMetalness)
#define gEmissive resolveUniform(getResourceParameters<SceneResourceParameters>().emissive)
#define gEnvironmentMap resolveUniform(getResourceParameters<SceneResourceParameters>().environmentMap)

float3 toneMap(float3 color, float exposure)
{
    color = 1.0 - exp(-max(Metallic.WorkingColor::toLinearRec709(color), 0.0) * exposure);
    return pow(saturate(color), 1.0 / 2.2);
}

float3 skyColor(float3 direction)
{
    float sky = saturate(direction.y * 0.5 + 0.5);
    return Metallic.WorkingColor::fromLinearRec709(lerp(float3(0.025, 0.03, 0.04), float3(0.12, 0.18, 0.28), sky));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRtxdiMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    SceneRTXDIPush push = getConstants<SceneRTXDIPush>();
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    let environment = getResourceParameters<SceneResourceParameters>().environment;
    float physicalExposure = push.outputLinear != 0u ? 1.0 : lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
    float nrdInversePreExposure = physicalNrdInversePreExposure(environment);
    if (physicalEnvironmentEnabled(environment)) {
        nrdInversePreExposure *= max(physicalExposure, 1.0);
    }

    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.frameIndex + 1u) * 0x9e3779b9u);
    float2 samplePixel = float2(pixel) + float2(0.5, 0.5);
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        // Background visibility is independent of whether an HDR map has
        // finished loading. Hidden environments also hide the preview fallback.
        float3 sky = behaviorEnabled(push, kBehaviorEnvironmentVisible)
            ? (behaviorEnabled(push, kBehaviorEnvironmentEnabled)
                ? environmentRadiance(ray.direction, push)
                : skyColor(ray.direction))
            : float3(0.0, 0.0, 0.0);
        if (behaviorEnabled(push, kBehaviorEnvironmentVisible) &&
            behaviorEnabled(push, kBehaviorEnvironmentEnabled) && physicalEnvironmentEnabled(environment)) {
            sky = physicalVisibleEnvironment(environment, ray.direction, ray.origin) * max(push.environmentIntensity, 0.0);
        }
        gReservoirCurrent[pixel] = packReservoir(emptyReservoir());
        gPositionCurrent[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNormalCurrent[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        gNoisyDiffuse[pixel] = packNrdRadianceHitDistance(float3(0.0, 0.0, 0.0), 0.0, true);
        gNoisySpecular[pixel] = packNrdRadianceHitDistance(float3(0.0, 0.0, 0.0), 0.0, true);
        gNrdNormalRoughness[pixel] = packNrdNormalRoughness(
            float3(0.0, 1.0, 0.0),
            1.0,
            0.0);
        gNrdMotionVectors[pixel] = float4(
            motionVectorToPrevious(
                ray.origin + ray.direction * max(push.clipOrtho.y, 1.0),
                samplePixel,
                push),
            0.0,
            0.0);
        gNrdViewZ[pixel] = max(push.clipOrtho.y, 1.0);
        gBaseColorMetalness[pixel] = float4(0.0, 0.0, 0.0, 0.0);
        sky *= physicalExposure;
        // Alpha carries the frame-wide inverse pre-exposure for composition;
        // RGB remains the physical HDR sky, including visible source disks.
        gEmissive[pixel] = float4(sky, nrdInversePreExposure);
        gOutput[pixel] = float4(visualizationMode(push) != kVisualizationShaded ? float3(0.0)
            : (push.outputLinear != 0u ? sky : toneMap(sky, push.exposure)), 1.0);
        return;
    }

    Surface surface = makeSurface(hit, ray, push);
    gPositionCurrent[pixel] = float4(surface.position, 1.0);
    gNormalCurrent[pixel] = float4(surface.normal, 1.0);
    gNrdNormalRoughness[pixel] = packNrdNormalRoughness(
        surface.normal,
        surface.roughness,
        0.0);
    gNrdMotionVectors[pixel] = float4(
        motionVectorToPrevious(surface.position, samplePixel, push),
        0.0,
        0.0);
    gNrdViewZ[pixel] = cameraLinearDepth(surface.position, push);
    gBaseColorMetalness[pixel] = float4(surface.baseColor, surface.metalness);
    gEmissive[pixel] = float4(surface.emissive * physicalExposure, nrdInversePreExposure);

    DIReservoir reservoir = emptyReservoir();
    uint localSampleCount = push.lightCount > 0u ? max(push.initialSampleCount, 1u) : 0u;
    uint environmentSampleCount = behaviorEnabled(push, kBehaviorEnvironmentEnabled)
        ? push.environmentSampleCount
        : 0u;
    uint initialSampleCount = localSampleCount + environmentSampleCount;
    float localTechniqueProbability = float(localSampleCount) / float(max(initialSampleCount, 1u));
    float environmentTechniqueProbability = float(environmentSampleCount) / float(max(initialSampleCount, 1u));
    for (uint sampleIndex = 0u; sampleIndex < localSampleCount; ++sampleIndex) {
        uint lightIndex;
        float inverseSourcePdf;
        bool sampled = sampleLocalLight(push, surface.position, seed, lightIndex, inverseSourcePdf);
        LightEvaluation evaluation = sampled
            ? evaluateLight(lightIndex, surface, push)
            : emptyLightEvaluation();
        if (behaviorEnabled(push, kBehaviorInitialVisibility) && evaluation.target > 0.0) {
            float3 offsetNormal = dot(evaluation.direction, surface.geometryNormal) >= 0.0
                ? surface.geometryNormal
                : -surface.geometryNormal;
            if (traceShadow(
                    surface.position + offsetNormal * 0.001,
                    evaluation.direction,
                    evaluation.distance,
                    push)) {
                evaluation.target = 0.0;
            }
        }
        updateReservoir(
            reservoir,
            lightIndex,
            evaluation.target * inverseSourcePdf / max(localTechniqueProbability, 0.000001),
            evaluation.target,
            1u,
            randomFloat(seed));
    }

    for (uint sampleIndex = 0u; sampleIndex < environmentSampleCount; ++sampleIndex) {
        uint2 texel;
        float texelPdf;
        bool sampled = sampleEnvironmentTexel(push, seed, texel, texelPdf);
        uint environmentWidth = 1u;
        uint environmentHeight = 1u;
        gEnvironmentMap.GetDimensions(environmentWidth, environmentHeight);
        uint texelIndex = texel.y * environmentWidth + texel.x;
        uint lightIndex = encodeEnvironmentSample(texelIndex);
        LightEvaluation evaluation = sampled
            ? evaluateEnvironmentSample(lightIndex, surface, push)
            : emptyLightEvaluation();
        if (behaviorEnabled(push, kBehaviorInitialVisibility) && evaluation.target > 0.0) {
            float3 offsetNormal = dot(evaluation.direction, surface.geometryNormal) >= 0.0
                ? surface.geometryNormal
                : -surface.geometryNormal;
            if (traceShadow(
                    surface.position + offsetNormal * 0.001,
                    evaluation.direction,
                    evaluation.distance,
                    push)) {
                evaluation.target = 0.0;
            }
        }
        float sourcePdfSolidAngle = texelPdf /
            environmentTexelSolidAngle(texel.y, environmentWidth, environmentHeight);
        updateReservoir(
            reservoir,
            lightIndex,
            evaluation.target /
                max(sourcePdfSolidAngle * environmentTechniqueProbability, 0.000001),
            evaluation.target,
            1u,
            randomFloat(seed));
    }

    uint maximumM = max(initialSampleCount * max(push.maxHistoryLength, 1u), initialSampleCount);
    float positionThreshold = max(push.sceneCenterRadius.w * push.depthThreshold, 0.0001);
    if (push.hasHistory != 0u && behaviorEnabled(push, kBehaviorTemporalReuse)) {
        float2 previousUv;
        if (projectWorldToPreviousUv(surface.position, push, previousUv)) {
            int2 previousPixel = int2(previousUv * float2(push.width, push.height));
            if (previousSurfaceMatches(previousPixel, surface, positionThreshold, push)) {
                DIReservoir previousReservoir = unpackReservoir(gReservoirPrevious[previousPixel]);
                if (reservoirValid(previousReservoir, push)) {
                    float currentTarget = evaluateLightSample(previousReservoir.lightIndex, surface, push).target;
                    combineReservoir(
                        reservoir,
                        previousReservoir,
                        currentTarget,
                        maximumM,
                        randomFloat(seed),
                        push);
                }
            }
        }
    }

    if (push.hasHistory != 0u && behaviorEnabled(push, kBehaviorSpatialReuse)) {
        uint spatialSampleCount = min(push.spatialSampleCount, 16u);
        for (uint sampleIndex = 0u; sampleIndex < spatialSampleCount; ++sampleIndex) {
            float angle = randomFloat(seed) * 2.0 * kPi;
            float sampleRadius = sqrt(randomFloat(seed)) * 24.0 + 1.0;
            int2 offset = int2(round(float2(cos(angle), sin(angle)) * sampleRadius));
            int2 neighborPixel = int2(pixel) + offset;
            float spatialPositionThreshold = positionThreshold * (1.0 + sampleRadius / 24.0);
            if (!previousSurfaceMatches(neighborPixel, surface, spatialPositionThreshold, push)) {
                continue;
            }
            DIReservoir neighborReservoir = unpackReservoir(gReservoirPrevious[neighborPixel]);
            if (!reservoirValid(neighborReservoir, push)) {
                continue;
            }
            float currentTarget = evaluateLightSample(neighborReservoir.lightIndex, surface, push).target;
            combineReservoir(
                reservoir,
                neighborReservoir,
                currentTarget,
                maximumM,
                randomFloat(seed),
                push);
        }
    }

    clampReservoirHistory(reservoir, maximumM);
    gReservoirCurrent[pixel] = packReservoir(reservoir);

    float3 noisyDiffuse = float3(0.0, 0.0, 0.0);
    float3 noisySpecular = float3(0.0, 0.0, 0.0);
    float lightDistance = 0.0;
    float3 color = surface.emissive;
    // Celestial sources are summed independently of the local/environment DI
    // reservoir, so an empty local domain still illuminates every surface.
    for (uint celestialSlot = 0u; celestialSlot < kCelestialLightCount; ++celestialSlot) {
        let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
            getResourceParameters<SceneResourceParameters>().environment.celestialLights);
        LightEvaluation selected = evaluateDeltaLight(sampleCelestialLightAtPosition(celestial, celestialSlot, surface.position), surface);
        if (selected.target <= 0.0) { continue; }
        float3 offsetNormal = dot(selected.direction, surface.geometryNormal) >= 0.0
            ? surface.geometryNormal : -surface.geometryNormal;
        if (!celestialLightCastsShadow(celestial, celestialSlot) ||
            !traceShadow(surface.position + offsetNormal * 0.001, selected.direction, selected.distance, push)) {
            noisyDiffuse += selected.diffuseRadiance;
            noisySpecular += selected.specularRadiance;
            color += selected.contribution;
            lightDistance = min(max(push.clipOrtho.y, push.sceneCenterRadius.w * 16.0), 65504.0);
        }
    }
    if (reservoirValid(reservoir, push)) {
        LightEvaluation selected = evaluateLightSample(reservoir.lightIndex, surface, push);
        float reservoirWeight = reservoir.weightSum /
            max(float(reservoir.M) * reservoir.selectedTarget, 1e-30);
        float3 offsetNormal = dot(selected.direction, surface.geometryNormal) >= 0.0
            ? surface.geometryNormal
            : -surface.geometryNormal;
        bool occluded = traceShadow(
            surface.position + offsetNormal * 0.001,
            selected.direction,
            selected.distance,
            push);
        if (!occluded) {
            noisyDiffuse += selected.diffuseRadiance * reservoirWeight;
            noisySpecular += selected.specularRadiance * reservoirWeight;
            // Environment visibility can use an unbounded shadow ray; NRD stores
            // hit distance in a finite, half-float-sized channel.
            lightDistance = min(selected.distance,
                min(max(push.clipOrtho.y, push.sceneCenterRadius.w * 16.0), 65504.0));
            color += selected.contribution * reservoirWeight;
        }
    }
    let aerial = physicalAerialSegment(environment, surface.position, ray.origin);
    noisyDiffuse *= aerial.transmittance;
    noisySpecular *= aerial.transmittance;
    color = aerial.transmittance * color + aerial.radiance;
    // NRD denoises the attenuated lighting signals. Its composite adds this
    // emissive guide, which carries in-scattering exactly once.
    gEmissive[pixel] = float4((aerial.transmittance * surface.emissive + aerial.radiance) * physicalExposure, nrdInversePreExposure);
    // RELAX accepts half-float Rec.709 radiance. Apply the same pre-exposure
    // to both signals before frontend sanitization; hit distance is unchanged.
    noisyDiffuse *= physicalExposure / nrdInversePreExposure;
    noisySpecular *= physicalExposure / nrdInversePreExposure;
    color *= physicalExposure;
    gNoisyDiffuse[pixel] = packNrdRadianceHitDistance(
        noisyDiffuse,
        lightDistance,
        true);
    gNoisySpecular[pixel] = packNrdRadianceHitDistance(
        noisySpecular,
        lightDistance,
        true);

    if (visualizationMode(push) == kVisualizationLightId) {
        float3 selectedColor = reservoirValid(reservoir, push)
            ? hashColor(reservoir.lightIndex)
            : float3(0.0, 0.0, 0.0);
        gOutput[pixel] = float4(selectedColor, 1.0);
        return;
    }
    if (visualizationMode(push) == kVisualizationHistory) {
        float history = saturate(float(reservoir.M) / float(max(maximumM, 1u)));
        float3 heat = lerp(float3(0.02, 0.05, 0.2), float3(1.0, 0.25, 0.02), history);
        gOutput[pixel] = float4(heat, 1.0);
        return;
    }
    if (visualizationMode(push) == kVisualizationReGIRCells) {
        gOutput[pixel] = float4(visualizeReGIRCell(surface.position), 1.0);
        return;
    }

    gOutput[pixel] = float4(push.outputLinear != 0u ? color : toneMap(color, push.exposure), 1.0);
}
