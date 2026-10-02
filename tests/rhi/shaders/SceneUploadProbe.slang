#include "../SceneUploadProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<SceneUploadProbeParameters> gProbe;

[shader("compute")]
[numthreads(8, 1, 1)]
void sceneUploadProbeMain(uint3 id : SV_DispatchThreadID)
{
    SceneUploadProbeParameters params = gProbe;
    if (id.x >= params.mipCount || params.outputOffset > params.output.length() ||
        id.x >= (params.output.length() - params.outputOffset) / 2u) { return; }
    uint width, height, levels;
    resolveDescriptor(params.image).GetDimensions(0, width, height, levels);
    if (id.x >= levels) { return; }
    resolveDescriptor(params.image).GetDimensions(id.x, width, height, levels);
    float4 first = resolveDescriptor(params.image).Load(int3(0, 0, id.x));
    float4 last = resolveDescriptor(params.image).Load(int3(width - 1, height - 1, id.x));
    params.output.data[params.outputOffset + id.x * 2u] = uint4(round(first * 255.0));
    params.output.data[params.outputOffset + id.x * 2u + 1u] = uint4(round(last * 255.0));
}
