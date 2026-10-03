import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Lighting;
using Metallic.Lighting;

#define gProbeOutput resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<ReGIRVirtualLightProbeResources>().output)

struct ReGIRProbePush
{
    float3 position;
    uint sampleCount;
    uint seedOffset;
    uint padding0;
    uint padding1;
    uint padding2;
};

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
    ReGIRProbePush push = getConstants<ReGIRProbePush>();
    if (id.x >= push.sampleCount) { return; }
    uint seed = (id.x + 1u) * 747796405u + push.seedOffset * 2891336453u;
    uint lightIndex = 0xffffffffu;
    float inverseSourcePdf = 0.0;
    float3 estimate = 0.0;
    if (selectPunctualLight(PunctualSamplingResources(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights), resolveBuffer<StructuredBuffer<uint4>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lightAlias), resolveUniform(getResourceParameters<ReGIRVirtualLightProbeResources>().lightsPdf)), push.position, seed, lightIndex, inverseSourcePdf)) {
        estimate = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights), lightIndex, push.position).illuminance * inverseSourcePdf;
    } else {
        lightIndex = 0xffffffffu;
    }
    gProbeOutput[id.x] = float4(estimate, asfloat(lightIndex));
    if (id.x == 0u) {
        float3 exact = 0.0;
        for (uint i = 0; i < punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights)); ++i) {
            exact += samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights), i, push.position).illuminance;
        }
        gProbeOutput[push.sampleCount] = float4(exact, 0.0);
        uint width, height, mipCount;
        resolveUniform(getResourceParameters<ReGIRVirtualLightProbeResources>().lightsPdf).GetDimensions(0u, width, height, mipCount);
        gProbeOutput[push.sampleCount + 1u] = float4(
            resolveUniform(getResourceParameters<ReGIRVirtualLightProbeResources>().lightsPdf).Load(int3(0, 0, int(mipCount) - 1)), 0, 0, 0);
        for (uint i = 0; i < 16; ++i) {
            float power = i < punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights))
                ? punctualLightPower(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<ReGIRVirtualLightProbeResources>().lights)[i + 1u]) : 0.0;
            uint2 texel = probeMortonPosition(i);
            float weight = texel.x < width && texel.y < height
                ? resolveUniform(getResourceParameters<ReGIRVirtualLightProbeResources>().lightsPdf).Load(int3(texel, 0)) : 0.0;
            gProbeOutput[push.sampleCount + 2u + i] = float4(power, weight, 0, 0);
        }
    }
}
