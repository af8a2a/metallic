import Core;
import ParameterRoot;
using Metallic;

struct Params {
    DescriptorHandle<Texture2D<float4>>* images;
    DescriptorHandle<SamplerState> samplers[2];
    DescriptorHandle<RWStructuredBuffer<uint4>> output;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint lane : SV_DispatchThreadID)
{
    Params p = getParameters<Params>();
    Texture2D<float4> image = resolveDescriptor(nonuniform(p.images[lane & 1]));
    SamplerState sampler = resolveDescriptor(nonuniform(p.samplers[(lane >> 1) & 1]));
    resolveDescriptor(p.output)[lane] = uint4(round(image.SampleLevel(sampler, float2(0.5), 0) * 255));
}
