// Program-independent fiber path integration. Geometry, visibility, RNG and
// continuation stay in the renderer; projected scattering belongs to Prepared.
import FiberMaterial;
import FiberLighting;
using Metallic.Material;

bool integrateSceneFiber<TPrepared : IPreparedFiberClosure>(TPrepared prepared,
    float misRoughness, HitInfo hit, ScenePathTracePush push, uint depth,
    inout Ray ray, inout RayCone rayCone, inout uint seed,
    inout float3 throughput, inout float3 radiance,
    inout float previousBsdfPdf, inout bool previousScatterDelta)
{
    // Compatibility policy: RTXCR's previous environment MIS proxy is retained
    // explicitly. It is not an exact standalone fiber PDF; no NdotL multiplies
    // projected scattering. A future PDF capability can replace this policy.
    for (uint celestialSlot = 0u; celestialSlot < kCelestialLightCount; ++celestialSlot) {
        let celestial = resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
            getResourceParameters<SceneResourceParameters>().environment.celestialLights);
        PunctualLightSample light = sampleCelestialLightAtPosition(celestial, celestialSlot, hit.position);
        if (!any(light.illuminance > 0.0)) { continue; }
        float3 scattering = max(prepared.evalProjected(light.direction), 0.0);
        if (luminanceMax(scattering * light.illuminance) > 0.0) {
            float3 origin = offsetRay(hit.position,
                dot(light.direction, hit.geometryNormal) >= 0.0 ? hit.geometryNormal : -hit.geometryNormal);
            if (!celestialLightCastsShadow(celestial, celestialSlot) ||
                !traceShadow(origin, light.direction, light.distance, push)) {
                radiance += throughput * scattering * light.illuminance;
            }
        }
    }
    uint fiberLightIndex;
    float fiberInverseSourcePdf;
    if (selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().regirGrid), resolveUniform(getResourceParameters<SceneResourceParameters>().lightsPdf)), hit.position, seed, fiberLightIndex, fiberInverseSourcePdf)) {
        PunctualLightSample light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), fiberLightIndex, hit.position);
        float3 fiberScattering = max(prepared.evalProjected(light.direction), 0.0);
        if (luminanceMax(fiberScattering * light.illuminance) > 0.0) {
            float3 shadowOrigin = offsetRay(hit.position,
                dot(light.direction, hit.geometryNormal) >= 0.0
                    ? hit.geometryNormal : -hit.geometryNormal);
            if (!traceShadow(shadowOrigin, light.direction,
                    max(light.distance - length(shadowOrigin - hit.position), 0.0), push)) {
                // Punctual lights are delta distributions: the BSDF and
                // environment samplers cannot hit them, so MIS is one.
                radiance += throughput * fiberScattering * light.illuminance * fiberInverseSourcePdf;
            }
        }
    }

    EnvironmentLightSample fiberLightSample;
    if (sampleEnvironmentLight(push, seed, fiberLightSample)) {
        float approximateFiberPdf = fallbackBsdfPdf(
            ray.direction,
            hit.normal,
            fiberLightSample.direction,
            0.0,
            misRoughness,
            0.5);
        float lightMisWeight = environmentLightMisWeight(
            push,
            fiberLightSample.pdfSolidAngle,
            approximateFiberPdf);
        float3 lightRadianceOverPdf =
            fiberLightSample.radiance * lightMisWeight /
            max(fiberLightSample.pdfSolidAngle, 0.000001);
        float3 shadowOffsetNormal = dot(fiberLightSample.direction, hit.geometryNormal) >= 0.0
            ? hit.geometryNormal
            : -hit.geometryNormal;
        bool occluded = traceShadow(
            offsetRay(hit.position, shadowOffsetNormal),
            fiberLightSample.direction,
            10000.0,
            push);
        if (!occluded) {
            radiance += throughput * Metallic.Lighting.evaluateFiberDirect(
                prepared, fiberLightSample.direction, lightRadianceOverPdf);
        }
    }

    float2 fiberRandom[2] = {
        float2(rand(seed), rand(seed)),
        float2(rand(seed), rand(seed)),
    };
    FiberSample scatter = prepared.sampleWeighted(float4(fiberRandom[0], fiberRandom[1]));
    if (scatter.pdf <= 0.0) { return false; }
    ray.direction = safeNormalize(scatter.wiWS, reflect(ray.direction, hit.normal));
    throughput *= scatter.weight;
    previousBsdfPdf = max(scatter.pdf, 0.000001);
    previousScatterDelta = false;
    rayCone = expandRayConeAfterScatter(rayCone, previousBsdfPdf);
    float3 offsetNormal = dot(ray.direction, hit.geometryNormal) >= 0.0
        ? hit.geometryNormal
        : -hit.geometryNormal;
    ray.origin = offsetRay(hit.position, offsetNormal);

    if (depth >= 2u) {
        float continueProbability = min(luminanceMax(throughput) + 0.05, 0.92);
        if (rand(seed) > continueProbability) {
            return false;
        }
        throughput /= max(continueProbability, 0.001);
    }
    return true;
}

bool shadeSceneFiber(PathTraceMaterial material, HitInfo hit, float3 stableNormal,
    ScenePathTracePush push, uint depth, inout Ray ray, inout RayCone rayCone,
    inout uint seed, inout float3 throughput, inout float3 radiance,
    inout float previousBsdfPdf, inout bool previousScatterDelta)
{
    RayMaterialWork work = sceneRayMaterialWork(hit, rayCone, ray.direction, seed, depth);
    let frame = makeRTXCRHairSurface(stableNormal, hit.tangent.xyz, stableNormal);
    work.surface.shadingNormalWS = frame.shadingNormal;
    work.surface.tangentWS = frame.tangent;
    work.surface.bitangentWS = cross(frame.shadingNormal, frame.tangent);
    RTXCRChiangMaterialProgram<SceneRTXCRHairParameters> program;
    program.parameters.value = material;
    // Adapter holds the already selected instance, so offsets are relative to it.
    work.instance = (MaterialInstanceRef)0;
    let materialResult = evaluateRayFiber(program, work);
    let prepared = prepareRayFiber(materialResult.closure, work);
    return integrateSceneFiber(prepared, sceneFiberMISRoughness(material), hit, push, depth,
        ray, rayCone, seed, throughput, radiance, previousBsdfPdf, previousScatterDelta);
}
