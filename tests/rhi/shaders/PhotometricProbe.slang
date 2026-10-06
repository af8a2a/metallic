import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Lighting;
using Metallic.Lighting;

#define gOutput resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<PhotometricProbeResources>().output)
#define gSH resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<PhotometricProbeResources>().irradiance)

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
}
