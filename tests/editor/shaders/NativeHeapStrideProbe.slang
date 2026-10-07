import ShaderCore;
using Metallic;

struct ProbePush
{
    uint inputBuffer;
    uint outputBuffer;
    uint sampledImage;
    uint storageImage;
    uint sampler;
    uint index;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uniform ProbePush push)
{
    let input = resolveUniform(ResourceHandle<StructuredBuffer<float4>>(push.inputBuffer));
    let output = resolveUniform(ResourceHandle<RWStructuredBuffer<float4>>(push.outputBuffer));
    let image = resolveUniform(ResourceHandle<Texture2D<float4>>(push.sampledImage));
    let storage = resolveUniform(ResourceHandle<RWTexture2D<float4>>(push.storageImage));
    let sampler = resolveUniform(SamplerHandle(push.sampler));
    float4 value = input[push.index] + image.SampleLevel(sampler, float2(0.5), 0.0);
    output[push.index] = value;
    storage[uint2(push.index, 0)] = value;
}
