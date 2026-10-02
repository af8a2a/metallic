#include "../../../Source/Runtime/Render/Core/TextureProbeParameters.h"
using Metallic;
#include "../../Interop/MaterialTextureDecode.hlsli"
[[vk::push_constant]] ConstantBuffer<TextureResourceProbeParameters> push;
[numthreads(1,1,1)]
void main()
{
    if (!push.output.contains(2u)) { return; }
    Texture2D<float4> texture = resolveDescriptor(push.texture);
    float4* output = push.output.data;
    float4 sample = texture.Load(int3(0,0,push.mip));
    output[0] = sample;
    output[1] = decodeMaterialTextureSample(sample,float(push.flags));
    output[2] = float4(materialTextureColor(sample.rgb,float(push.flags)),sample.a);
}
