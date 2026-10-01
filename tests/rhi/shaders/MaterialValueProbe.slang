import Core;
using Metallic;
import Material;
using Metallic.Material;
#include "MaterialValueDispatch.hlsli"

[shader("compute")]
[numthreads(4, 1, 1)]
void materialValueProbeMain(uint3 id : SV_DispatchThreadID)
{
    MaterialValueInstance instance = getResource<StructuredBuffer<MaterialValueInstance>>(97)[id.x];
    RWStructuredBuffer<float> output = getResource<RWStructuredBuffer<float>>(0);
    uint cursor = id.x * 25;
    output[cursor++] = float(instance.programId);
    for (uint p = 0; p < 4; ++p) {
        for (uint c = 0; c < 4; ++c) { output[cursor++] = instance.parameters[p][c]; }
    }
    PathTraceMaterial input = {};
    input.baseColor = float4(0.2, 0.3, 0.4, 0.75);
    input.params = float4(0.1, 0.2, 0.3, 0.4);
    PathTraceMaterial result = evaluateMaterialValue(id.x, float3(0.25, 0.5, 0.75), float3(0, 1, 0), float2(0.5, 0.25), input);
    for (uint c = 0; c < 4; ++c) { output[cursor++] = result.baseColor[c]; }
    for (uint c = 0; c < 4; ++c) { output[cursor++] = result.params[c]; }
}
