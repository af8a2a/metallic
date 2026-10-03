import Core;
using Metallic;
#include "../../Interop/MaterialTextureDecode.hlsli"
struct TextureProbePush { uint textureIndex; uint mip; uint flags; uint unused; };
[numthreads(1,1,1)]
void main()
{
    TextureProbePush push = getConstants<TextureProbePush>();
    Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<TextureProbeResourceParameters>().textures.load(push.textureIndex)));
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<TextureProbeResourceParameters>().output);
    float4 sample = texture.Load(int3(0,0,push.mip));
    output[0] = sample;
    output[1] = decodeMaterialTextureSample(sample,float(push.flags));
    output[2] = float4(materialTextureColor(sample.rgb,float(push.flags)),sample.a);
}
