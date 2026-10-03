import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct Params { uint outputOffset; };

[shader("compute")]
[numthreads(8, 1, 1)]
void sceneUploadProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 7u) { return; }
    Params params = getConstants<Params>();
    uint width, height, levels;
    resolveUniform(getResourceParameters<SceneUploadProbeResources>().texture).GetDimensions(id.x, width, height, levels);
    float4 first = resolveUniform(getResourceParameters<SceneUploadProbeResources>().texture).Load(int3(0, 0, id.x));
    float4 last = resolveUniform(getResourceParameters<SceneUploadProbeResources>().texture).Load(int3(width - 1, height - 1, id.x));
    resolveBuffer<RWStructuredBuffer<uint4>>(getResourceParameters<SceneUploadProbeResources>().output)[params.outputOffset + id.x * 2u] = uint4(round(first * 255.0));
    resolveBuffer<RWStructuredBuffer<uint4>>(getResourceParameters<SceneUploadProbeResources>().output)[params.outputOffset + id.x * 2u + 1u] = uint4(round(last * 255.0));
}
