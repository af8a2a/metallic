#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<UpscalerGuideResolveParams> p;
[shader("compute")]
[numthreads(8,8,1)]
void guideFillMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 2 || id.y >= 2) { return; }
    resolveDescriptor(p.outputDepth)[id.xy] = (id.x == 1 && id.y == 1) ? 0.25 : 0.75;
    resolveDescriptor(p.outputMotion)[id.xy] = float2(id.xy) * 0.125;
}
