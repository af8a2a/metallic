#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

struct Params { uint outputOffset; };

[shader("compute")]
[numthreads(8, 1, 1)]
void sceneUploadProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 7u) { return; }
    Params params = METALLIC_CONSTANTS(Params);
    uint width, height, levels;
    METALLIC_RESOURCE(Texture2D<float4>, 0).GetDimensions(id.x, width, height, levels);
    float4 first = METALLIC_RESOURCE(Texture2D<float4>, 0).Load(int3(0, 0, id.x));
    float4 last = METALLIC_RESOURCE(Texture2D<float4>, 0).Load(int3(width - 1, height - 1, id.x));
    METALLIC_RESOURCE(RWStructuredBuffer<uint4>, 1)[params.outputOffset + id.x * 2u] = uint4(round(first * 255.0));
    METALLIC_RESOURCE(RWStructuredBuffer<uint4>, 1)[params.outputOffset + id.x * 2u + 1u] = uint4(round(last * 255.0));
}
