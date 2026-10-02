#include "../TextureBindingProbeParameters.h"
[[vk::push_constant]] ConstantBuffer<NonuniformTextureParams> gTextureProbe;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint lane : SV_DispatchThreadID)
{
    NonuniformTextureParams p = gTextureProbe;
    Texture2D<float4> image = resolveDescriptor(nonuniform(p.images[lane & 1]));
    SamplerState sampler = resolveDescriptor(nonuniform(p.samplers[(lane >> 1) & 1]));
    resolveDescriptor(p.output)[lane] = uint4(round(image.SampleLevel(sampler, float2(0.5), 0) * 255));
}
