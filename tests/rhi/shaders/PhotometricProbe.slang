module PhotometricProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Lighting;
using Metallic.Lighting;

#define gOutput resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<PhotometricProbeResources>().output)
#define gSH resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<PhotometricProbeResources>().irradiance)

float3 iesProbePosition(float verticalDegrees, float horizontalDegrees)
{
    const float radians = 0.0174532925199433f;
    float vertical = verticalDegrees * radians;
    float horizontal = horizontalDegrees * radians;
    return float3(-sin(vertical) * cos(horizontal), sin(vertical) * sin(horizontal), cos(vertical));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void photometricProbeMain()
{
    gOutput[0] = float4(samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights), 0u, float3(0, 0, 1)).illuminance, lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights)));
    gOutput[1] = float4(samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights), 0u, float3(0, 0, 2)).illuminance, 0);
    gOutput[2] = float4(sampleCelestialLight(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(getResourceParameters<PhotometricProbeResources>().celestialLights), 0u).illuminance, 0);
    gOutput[3] = float4(sampleCelestialLight(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(getResourceParameters<PhotometricProbeResources>().celestialLights), 0u).illuminance, 0);
    gOutput[4] = float4(samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights), 1u, float3(0, 0, 1)).illuminance, 0);
    gOutput[5] = float4(samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights), 1u, float3(1, 0, 0)).illuminance, 0);
    float3 normals[6] = {float3(1,0,0), float3(-1,0,0), float3(0,1,0),
        float3(0,-1,0), float3(0,0,1), float3(0,0,-1)};
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gSH);
    for (uint n = 0; n < 6; ++n) {
        gOutput[6 + n] = float4(environment.evaluate(normals[n]), 0);
    }
    let lights = resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<PhotometricProbeResources>().lights);
    gOutput[12] = float4(samplePunctualLight(lights, 2u, iesProbePosition(30.0f, 0.0f)).illuminance, 0);
    gOutput[13] = float4(samplePunctualLight(lights, 2u, iesProbePosition(30.0f, 90.0f)).illuminance, 0);
    gOutput[14] = float4(samplePunctualLight(lights, 2u, iesProbePosition(50.0f, 45.0f)).illuminance, 0);
    gOutput[15] = float4(samplePunctualLight(lights, 2u, iesProbePosition(110.0f, 0.0f)).illuminance, 0);
    gOutput[16] = float4(samplePunctualLight(lights, 2u, iesProbePosition(30.0f, 315.0f)).illuminance, 0);
    gOutput[17] = float4(samplePunctualLight(lights, 3u, iesProbePosition(30.0f, 0.0f)).illuminance, 0);
    gOutput[18] = float4(samplePunctualLight(lights, 3u, iesProbePosition(50.0f, 0.0f)).illuminance, 0);
}
