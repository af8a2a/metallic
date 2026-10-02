#include "../PhotometricProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<PhotometricProbeParameters> gProbe;
#define gOutput gProbe.output.data
#define gSH resolveDescriptor(gProbe.sphericalHarmonics)
#define gLights resolveDescriptor(gProbe.lights)

[shader("compute")]
[numthreads(1, 1, 1)]
void photometricProbeMain()
{
    if (!gProbe.output.contains(11u)) { return; }
    gOutput[0] = float4(samplePunctualLight(gLights, 0u, float3(0, 0, 1)).illuminance, lightingExposure(gLights));
    gOutput[1] = float4(samplePunctualLight(gLights, 0u, float3(0, 0, 2)).illuminance, 0);
    gOutput[2] = float4(samplePunctualLight(gLights, 1u, float3(0, 0, 1)).illuminance, 0);
    gOutput[3] = float4(samplePunctualLight(gLights, 1u, float3(0, 0, 200)).illuminance, 0);
    gOutput[4] = float4(samplePunctualLight(gLights, 2u, float3(0, 0, 1)).illuminance, 0);
    gOutput[5] = float4(samplePunctualLight(gLights, 2u, float3(1, 0, 0)).illuminance, 0);
    float3 normals[6] = {float3(1,0,0), float3(-1,0,0), float3(0,1,0),
        float3(0,-1,0), float3(0,0,1), float3(0,0,-1)};
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gSH);
    for (uint n = 0; n < 6; ++n) {
        gOutput[6 + n] = float4(environment.evaluate(normals[n]), 0);
    }
}
