#include "../../../Source/Runtime/Render/Core/MaterialErrorParameters.h"
[[vk::push_constant]] ConstantBuffer<MaterialErrorParams> gParams;
using Metallic;
[shader("compute")]
[numthreads(8, 8, 1)]
void materialErrorMain(uint3 id : SV_DispatchThreadID)
{
    let output = resolveDescriptor(gParams.output);
    uint width, height;
    output.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    bool checker = ((id.x / 8 + id.y / 8) & 1) != 0;
    output[id.xy] = gParams.color != 0
        ? float4(checker ? float3(1, 0, 1) : float3(0.08, 0, 0.08), 1)
        : float4(0, 0, 0, 0);
}
