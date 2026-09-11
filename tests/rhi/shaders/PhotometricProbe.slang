#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

#include "../../../Shaders/Libraries/Lighting/PhotometricLighting.slang"
#include "../../../Shaders/Libraries/Math/SphericalHarmonics.slang"
#define gOutput METALLIC_RESOURCE(RWStructuredBuffer<float4>, 0)
#define gSH METALLIC_RESOURCE(StructuredBuffer<float4>, 1)

[shader("compute")]
[numthreads(1, 1, 1)]
void photometricProbeMain()
{
    gOutput[0] = float4(samplePunctualLight(0u, float3(0, 0, 1)).illuminance, lightingExposure());
    gOutput[1] = float4(samplePunctualLight(0u, float3(0, 0, 2)).illuminance, 0);
    gOutput[2] = float4(samplePunctualLight(1u, float3(0, 0, 1)).illuminance, 0);
    gOutput[3] = float4(samplePunctualLight(1u, float3(0, 0, 200)).illuminance, 0);
    gOutput[4] = float4(samplePunctualLight(2u, float3(0, 0, 1)).illuminance, 0);
    gOutput[5] = float4(samplePunctualLight(2u, float3(1, 0, 0)).illuminance, 0);
    float3 normals[6] = {float3(1,0,0), float3(-1,0,0), float3(0,1,0),
        float3(0,-1,0), float3(0,0,1), float3(0,0,-1)};
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gSH);
    for (uint n = 0; n < 6; ++n) {
        gOutput[6 + n] = float4(environment.evaluate(normals[n]), 0);
    }
}
