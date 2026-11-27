module VolumetricCloudView;

import ShaderCore;
import ParameterRoot;
import Atmosphere;
import AtmosphereVirtualShadowMap;
import Clouds;
import VolumetricCloudViewTrace;
import LumenProbeMath;
import LumenScreenTrace;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Clouds;
using Metallic.Lumen;

#include "../../../Source/Runtime/Render/Environment/VolumetricCloudViewParameters.h"

// Reference: UE 5.7.4 NvRTX VolumetricCloud.usf, DistanceFromCloudLayerEntryPoint.
// Reuse the existing spherical weather field and its spectral Sun/Moon lighting.
static const float kStopTracingTransmittance = 0.005f;

float3 cloudLight(GPUAtmosphereParameters a, Texture2D<float4> transmittanceLut, float3 positionKm,
    float3 viewDir, float3 lightDir, float4 irradiance, DirectionalAtmosphereShadow shadow, float tKm)
{
    if (irradiance.w <= 0.0f || all(irradiance.xyz <= 0.0f)) { return 0.0f; }
    float lightT = Clouds::transmittance(a, positionKm, lightDir, 1e9f, 8u);
    float phase = scatteringPhase(dot(viewDir, lightDir), a.cloudOptics.z, a.cloudOptics.y, lightT);
    return transmittanceToSource(a, transmittanceLut, positionKm, lightDir) *
        irradiance.xyz * irradiance.w * phase * shadow.visibility(tKm);
}

CloudViewIntegral traceViewCloud(VolumetricCloudViewParams p, GPUAtmosphereParameters a, float3 originKm,
    float3 direction, float maximumDistanceKm, float stepNoise, DirectionalAtmosphereShadow shadow)
{
    CloudViewIntegral result = CloudViewIntegral::empty();
    float groundKm = groundDistance(a, originKm, direction);
    if (groundKm >= 0.0f) { maximumDistanceKm = min(maximumDistanceKm, groundKm); }
    // UE's default tracing-distance mode marches the first contiguous spherical layer intersection.
    float2 range = Clouds::intervals(a, originKm, direction, maximumDistanceKm).xy;
    if (range.x < 0.0f || range.y <= range.x || range.x > 350.0f) { return result; }

    float tMin = range.x * 1000.0f;
    float tMax = tMin + min(50000.0f, (range.y - range.x) * 1000.0f);
    uint stepCount = uint(max(2.0f, 96.0f * saturate((tMax - tMin) * (1.0f / 15000.0f))));
    float stepMeters = (tMax - tMin) / float(stepCount);
    float t = tMin + stepNoise * stepMeters;
    let lightTransmittance = resolveUniform(p.transmittanceLut);
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
            float3 incident = cloudLight(a, lightTransmittance, positionKm, direction,
                a.sunDirectionRadius.xyz, a.sunIrradianceEnabled, sunShadow, tKm) +
                cloudLight(a, lightTransmittance, positionKm, direction,
                    a.moonDirectionRadius.xyz, a.moonIrradianceEnabled, moonShadow, tKm);
            result.integrateSample(t, stepMeters, extinction, (extinction * a.cloudOptics.y) * incident);
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
    if (p.atmosphereParameters.index != 0xffffffffu && (deviceDepth > 0.0f || p.skyVisible != 0u)) {
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
        CloudViewIntegral cloud = traceViewCloud(p, a, originKm, direction, distanceKm, stepNoise, shadow);
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
