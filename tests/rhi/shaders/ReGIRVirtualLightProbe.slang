import ShaderCore;
using Metallic;
import Lighting;
using Metallic.Lighting;

#include "../ReGIRProbeParameters.h"
[[vk::push_constant]] ConstantBuffer<ReGIRProbeParameters> gProbe;
#define gProbeOutput resolveDescriptor(gProbe.output)
#define gProbeLights resolveDescriptor(gProbe.lights)
#define gProbeGrid resolveDescriptor(gProbe.grid)
#define gProbePdf resolveDescriptor(gProbe.pdf)

uint2 probeMortonPosition(uint index)
{
    uint2 position = 0;
    for (uint bit = 0; bit < 16; ++bit) {
        position.x |= ((index >> (2u * bit)) & 1u) << bit;
        position.y |= ((index >> (2u * bit + 1u)) & 1u) << bit;
    }
    return position;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void reGIRVirtualLightProbeMain(uint3 id : SV_DispatchThreadID)
{
    ReGIRProbeParameters push = gProbe;
    if (id.x >= push.sampleCount) { return; }
    uint seed = (id.x + 1u) * 747796405u + push.seedOffset * 2891336453u;
    uint lightIndex = 0xffffffffu;
    float inverseSourcePdf = 0.0;
    float3 estimate = 0.0;
    if (selectPunctualLight(PunctualSamplingResources(gProbeLights, gProbeGrid, gProbePdf), push.position, seed, lightIndex, inverseSourcePdf)) {
        estimate = samplePunctualLight(gProbeLights, lightIndex, push.position).illuminance * inverseSourcePdf;
    } else {
        lightIndex = 0xffffffffu;
    }
    gProbeOutput[id.x] = float4(estimate, asfloat(lightIndex));
    if (id.x == 0u) {
        float3 exact = 0.0;
        for (uint i = 0; i < punctualLightCount(gProbeLights); ++i) {
            exact += samplePunctualLight(gProbeLights, i, push.position).illuminance;
        }
        gProbeOutput[push.sampleCount] = float4(exact, 0.0);
        uint width, height, mipCount;
        gProbePdf.GetDimensions(0u, width, height, mipCount);
        gProbeOutput[push.sampleCount + 1u] = float4(
            gProbePdf.Load(int3(0, 0, int(mipCount) - 1)), 0, 0, 0);
        for (uint i = 0; i < 16; ++i) {
            float power = i < punctualLightCount(gProbeLights)
                ? punctualLightPower(gProbeLights[i + 1u]) : 0.0;
            uint2 texel = probeMortonPosition(i);
            float weight = texel.x < width && texel.y < height
                ? gProbePdf.Load(int3(texel, 0)) : 0.0;
            gProbeOutput[push.sampleCount + 2u + i] = float4(power, weight, 0, 0);
        }
    }
}
