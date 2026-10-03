import Core;
import ParameterRoot;
using Metallic;

struct Params {
    BufferSpan<ResourceHandle<Texture2D<float4>>> images;
    SamplerHandle samplers[2];
    ResourceHandle<RWStructuredBuffer<uint4>> output;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint lane : SV_DispatchThreadID)
{
    Params p = getParameters<Params>();
    Texture2D<float4> image = resolveNonUniform(p.images[lane & 1]);
    SamplerState sampler = resolveNonUniform(p.samplers[(lane >> 1) & 1]);
    resolveUniform(p.output)[lane] = uint4(round(image.SampleLevel(sampler, float2(0.5), 0) * 255));
}
