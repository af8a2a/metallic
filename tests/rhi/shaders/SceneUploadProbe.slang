import Core;
using Metallic;

struct Params { uint outputOffset; };

[shader("compute")]
[numthreads(8, 1, 1)]
void sceneUploadProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 7u) { return; }
    Params params = getConstants<Params>();
    uint width, height, levels;
    getResource<Texture2D<float4>>(0).GetDimensions(id.x, width, height, levels);
    float4 first = getResource<Texture2D<float4>>(0).Load(int3(0, 0, id.x));
    float4 last = getResource<Texture2D<float4>>(0).Load(int3(width - 1, height - 1, id.x));
    getResource<RWStructuredBuffer<uint4>>(1)[params.outputOffset + id.x * 2u] = uint4(round(first * 255.0));
    getResource<RWStructuredBuffer<uint4>>(1)[params.outputOffset + id.x * 2u + 1u] = uint4(round(last * 255.0));
}
