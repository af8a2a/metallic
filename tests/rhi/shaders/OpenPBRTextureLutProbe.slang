import ShaderCore;
using Metallic;
import OpenPBRTextureLUT;
using Metallic.OpenPBR;

struct ProbeParams
{
    RWBufferSpan<float4> output;
    uint textures[8];
    SamplerHandle linearClamp;
    uint mode;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gProbe;

float4 texel(uint id, int3 p)
{
    if (id == 0 || id == 3) {
        return resolveNonUniform(ResourceHandle<Texture3D<float4>>(gProbe.textures[id])).Load(int4(p, 0));
    }
    return resolveNonUniform(ResourceHandle<Texture2D<float4>>(gProbe.textures[id])).Load(int3(p.xy, 0));
}

uint hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRTextureLutMain(uint3 tid : SV_DispatchThreadID)
{
    uint id = tid.x % 8, index = tid.x / 8;
    int3 size = int3(32, id == 6 ? 1 : 32, id == 0 || id == 3 ? 32 : 1);
    float3 uv = (float3(index % 32, (index / 32) % size.y, (index / 1024) % size.z) + 0.5) / float3(size);
    if (gProbe.mode != 0) {
        uv = float3(hash(index * 3), hash(index * 3 + 1), hash(index * 3 + 2)) / 4294967296.0;
        // Include outside-domain clamp and exact [0,1] boundaries on every axis.
        if (index < 64) { uv = float3(index % 4, (index / 4) % 4, index / 16) * 0.5 - 0.25; }
        if (index == 64) { uv = 0; }
        if (index == 65) { uv = 1; }
    }
#if METALLIC_OPENPBR_FILTERED
    float4 value;
    if (id == 0 || id == 3) {
        value = sampleOpenPBRTexture3D(resolveNonUniform(ResourceHandle<Texture3D<float4>>(gProbe.textures[id])),
            resolveUniform(gProbe.linearClamp), uv);
    } else {
        value = sampleOpenPBRTexture2D(resolveNonUniform(ResourceHandle<Texture2D<float4>>(gProbe.textures[id])),
            resolveUniform(gProbe.linearClamp), uv.xy);
    }
#else
    float3 p = clamp(uv * float3(size) - 0.5, float3(0), float3(size - 1));
    int3 lo = int3(floor(p)), hi = min(lo + 1, size - 1);
    float3 f = p - float3(lo);
    float4 a = lerp(lerp(texel(id, lo), texel(id, int3(hi.x, lo.yz)), f.x),
        lerp(texel(id, int3(lo.x, hi.y, lo.z)), texel(id, int3(hi.xy, lo.z)), f.x), f.y);
    float4 b = lerp(lerp(texel(id, int3(lo.xy, hi.z)), texel(id, int3(hi.x, lo.y, hi.z)), f.x),
        lerp(texel(id, int3(lo.x, hi.yz)), texel(id, hi), f.x), f.y);
    float4 value = lerp(a, b, f.z);
#endif
    gProbe.output.store(tid.x, value);
}
