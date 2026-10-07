import ShaderCore;
using Metallic;

struct Push
{
    uint constantBuffer;
    uint storageBuffer;
    uint bufferView;
    uint sampledImage;
    uint storageImage;
    uint sampler;
    uint outputBuffer;
    uint item;
};

struct ConstantData
{
    uint4 value;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void writeImageMain(uniform Push push)
{
    RWTexture2D<float4> image = resolveUniform(ResourceHandle<RWTexture2D<float4>>(push.storageImage));
    image[uint2(0)] = float4(float(push.item + 1) / 255.0, 0, 0, 1);
    image[uint2(1, 0)] = float4(float(200 - push.item) / 255.0, 0, 0, 1);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void readMain(uniform Push push)
{
    ConstantBuffer<ConstantData> constant = resolveUniform(ResourceHandle<ConstantBuffer<ConstantData>>(push.constantBuffer));
    ByteAddressBuffer storage = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.storageBuffer));
    ByteAddressBuffer view = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.bufferView));
    Texture2D<float4> image = resolveUniform(ResourceHandle<Texture2D<float4>>(push.sampledImage));
    SamplerState sampler = resolveUniform(SamplerHandle(push.sampler));
    RWByteAddressBuffer output = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.outputBuffer));
    output.Store4(push.item * 16, uint4(constant.value.x, storage.Load(0), view.Load(0),
        uint(round(image.SampleLevel(sampler, float2(1.25, 0.5), 0).r * 255.0))));
}
