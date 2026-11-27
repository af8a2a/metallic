module VolumetricCloudView;

import ShaderCore;
import ParameterRoot;
import Atmosphere;
import AtmosphereVirtualShadowMap;
import Clouds;
import CloudLightEnvironment;
import VolumetricCloudLighting;
import VolumetricCloudViewTrace;
import LumenProbeMath;
import LumenScreenTrace;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Clouds;
using Metallic.Lumen;

#include "../../../Source/Runtime/Render/Environment/VolumetricCloudViewParameters.h"
__include "VolumetricCloudViewTemporal.slang";

// Reference: UE 5.7.4 NvRTX VolumetricCloud.usf, DistanceFromCloudLayerEntryPoint.
static const float kStopTracingTransmittance = 0.005f;

float3 cloudDirectTransmittance(GPUAtmosphereParameters a, float3 positionKm, float3 lightDir,
    float3 illuminance, DirectionalAtmosphereShadow shadow, float tKm)
{
    if (all(illuminance <= 0.0f)) { return 0.0f; }
    float opaqueShadow = shadow.visibility(tKm);
    if (opaqueShadow == 0.0f) { return 0.0f; }
    return opaqueShadow * cloudShadowTransmittanceOrders(a, positionKm, lightDir, kCloudShadowSampleCount);
}

CloudViewIntegral traceViewCloud(GPUAtmosphereParameters a, float3 originKm,
    float3 direction, float maximumDistanceKm, float invDistanceToSampleCountMax, float stepNoise, DirectionalAtmosphereShadow shadow,
    float closestDepthMeters, out CloudViewIntegral closest, out float noCloudDepthKm)
{
    CloudViewIntegral result = CloudViewIntegral::empty();
    closest = result;
    noCloudDepthKm = maximumDistanceKm;
    float groundKm = groundDistance(a, originKm, direction);
    if (groundKm >= 0.0f) { maximumDistanceKm = min(maximumDistanceKm, groundKm); }
    // UE's default tracing-distance mode marches the first contiguous spherical layer intersection.
    float2 range = Clouds::intervals(a, originKm, direction, maximumDistanceKm).xy;
    if (range.x < 0.0f || range.y <= range.x || range.x > 350.0f) { return result; }

    float tMin = range.x * 1000.0f;
    float tMax = tMin + min(50000.0f, (range.y - range.x) * 1000.0f);
    noCloudDepthKm = tMax * 0.001f;
    uint stepCount = uint(max(2.0f, 96.0f * saturate((tMax - tMin) * invDistanceToSampleCountMax)));
    float stepMeters = (tMax - tMin) / float(stepCount);
    float t = tMin + stepNoise * stepMeters;
    float3 sunIlluminance = cloudAtmosphereLightIlluminanceOnGround(a, 0u);
    float3 moonIlluminance = cloudAtmosphereLightIlluminanceOnGround(a, 1u);
    float3 sunPhase = cloudPhaseOrders(dot(a.sunDirectionRadius.xyz, direction), a.cloudOptics.z);
    float3 moonPhase = cloudPhaseOrders(dot(a.moonDirectionRadius.xyz, direction), a.cloudOptics.z);
    float3 distantSky = cloudDistantSkyLuminance(a) * kCloudMaterialAmbientOcclusion;
    var sunShadow = shadow;
    var moonShadow = shadow;
    sunShadow.selectLight(0u, direction, 0.0f);
    moonShadow.selectLight(1u, direction, 0.0f);
    [loop]
    for (uint step = 0; step < stepCount; ++step) {
        float tKm = t * 0.001f;
        float3 positionKm = originKm + direction * tKm;
        float extinction = density(a, positionKm) * a.cloudOptics.x * 0.001f;
        if (extinction > 0.0f) {
            float3 sunT = cloudDirectTransmittance(a, positionKm, a.sunDirectionRadius.xyz, sunIlluminance, sunShadow, tKm);
            float3 moonT = cloudDirectTransmittance(a, positionKm, a.moonDirectionRadius.xyz, moonIlluminance, moonShadow, tKm);
            float3 scatteringOrders = cloudScatteringOrders(a.cloudOptics.y, extinction);
            float3 extinctionOrders = cloudExtinctionOrders(extinction);
            [unroll]
            for (int ms = 2; ms >= 0; --ms) {
                float3 incident = sunT[ms] * sunIlluminance * sunPhase[ms];
                incident += moonT[ms] * moonIlluminance * moonPhase[ms];
                if (ms == 0) { incident += distantSky; }
                result.integrateOrder(t, stepMeters, extinctionOrders[ms], incident * scatteringOrders[ms], uint(ms));
            }
            // The near integral is the prefix of the same march, including its sample jitter.
            if (t <= closestDepthMeters) {
                float previousT = closest.transmittance;
                closest = result;
                if (t == closestDepthMeters) { closest.transmittance = previousT; }
            }
            if (result.transmittance < kStopTracingTransmittance) { break; }
        }
        t += stepMeters;
    }
    return result;
}

AtmosphereSegment cloudAerialPerspective(VolumetricCloudViewParams p, GPUAtmosphereParameters a,
    float3 originMeters, float3 originKm, float3 direction, float distanceMeters, DirectionalAtmosphereShadow shadow)
{
    float distanceKm = distanceMeters * 0.001f;
    if (distanceKm <= a.settings.z) {
        AtmosphereSegment result;
        result.radiance = 0.0f;
        result.transmittance = 1.0f;
        return result;
    }
    if (p.aerialPerspective.index != 0xffffffffu && all(originMeters == a.observerWorldTop.xyz) &&
        distanceKm <= a.settings.x) {
        BufferSpan<float4> aerial = {p.aerialPerspective, 0u, 32u * 32u * 32u * 2u};
        return aerialPerspective(a, aerial, originMeters + direction * distanceMeters);
    }
    return integrateAerialPerspective(a, resolveUniform(p.transmittanceLut), resolveUniform(p.multiScatteringLut),
        originKm, direction, distanceKm - a.settings.z, shadow, false);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void clearCloudViewMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricCloudViewParams>();
    if (any(id.xy >= uint2(p.camera.viewport.yz))) { return; }
    resolveUniform(p.scattering)[id.xy] = 0.0f;
    resolveUniform(p.transmittance)[id.xy] = 1.0f;
    resolveUniform(p.cloudDepth)[id.xy] = 0.0f;
    resolveUniform(p.color)[id.xy] = resolveUniform(p.sceneColor).Load(int3(id.xy, 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void volumetricCloudViewMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricCloudViewParams>();
    uint2 size = uint2(p.camera.viewport.yz);
    if (any(id.xy >= size)) { return; }
    uint2 pixel = id.xy;
    float4 background = resolveUniform(p.sceneColor).Load(int3(pixel, 0));
    float deviceDepth = resolveUniform(p.depth).Load(int3(pixel, 0));
    float3 radiance = 0.0f;
    float transmittance = 1.0f;
    float cloudDepthMeters = 0.0f;
    if (deviceDepth > 0.0f || p.skyVisible != 0u) {
        GPUAtmosphereParameters a = resolveUniform(p.atmosphereParameters)[0];
        ScreenTraceProjection camera = makeScreenTraceProjection(p.camera, p.jitter);
        float2 screen = screenTraceScreenPosition((float2(pixel) + 0.5f + p.jitter) / float2(size));
        float2 offset = screen * camera.halfExtent;
        float3 rayOffset = camera.right * offset.x + camera.up * offset.y;
        float3 direction = camera.orthographic ? camera.forward : normalize(camera.forward + rayOffset);
        float3 originMeters = camera.orthographic ? camera.eye + rayOffset : camera.eye;
        float3 originKm = worldToAtmosphereKm(a, originMeters);
        float distanceKm = deviceDepth > 0.0f ?
            screenTraceConvertFromDeviceZ(camera, deviceDepth) * 0.001f / dot(direction, camera.forward) : 1e9f;
        let shadow = DirectionalAtmosphereShadow::forRay(p.worldShadowFaces.index, p.shadowMask, originMeters);
        // Freeze UE's blue-noise phase in this full-resolution view without history.
        float stepNoise = lumenProbeBlueNoiseScalar(resolveUniform(p.blueNoise), pixel, 0u);
        CloudViewIntegral closest;
        float noCloudDepthKm;
        CloudViewIntegral cloud = traceViewCloud(a, originKm, direction, distanceKm, p.invDistanceToSampleCountMax, stepNoise, shadow,
            -1.0f, closest, noCloudDepthKm);
        if (cloud.depthWeightsSum > 0.0f) {
            cloudDepthMeters = cloud.representativeDepthMeters();
            AtmosphereSegment aerial = cloudAerialPerspective(p, a, originMeters, originKm, direction, cloudDepthMeters, shadow);
            // UE VolumetricCloud.usf applies AP over cloud coverage at the representative depth.
            radiance = aerial.radiance * (1.0f - cloud.transmittance) +
                aerial.transmittance * spectralToWorking(cloud.luminance);
            transmittance = cloud.transmittance < kStopTracingTransmittance ? 0.0f : cloud.transmittance;
        }
    }
    resolveUniform(p.scattering)[pixel] = float4(radiance, 1.0f - transmittance);
    resolveUniform(p.transmittance)[pixel] = transmittance;
    resolveUniform(p.cloudDepth)[pixel] = cloudDepthMeters;
    resolveUniform(p.color)[pixel] = float4(radiance + transmittance * background.rgb, background.a);
}
