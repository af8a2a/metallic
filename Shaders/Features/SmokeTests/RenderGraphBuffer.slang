#include "../../../Source/Runtime/Render/Core/RenderGraphBufferParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<RenderGraphBufferParams> gParams;

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferWriteMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (gParams.output.length() < 4) { return; }
    const uint4 pattern = uint4(0x11223344u, 0xAABBCCDDu, 0xDEADBEEFu, 0xCAFEBABEu);
    for (uint i = 0; i < 4; ++i) { gParams.output.data[i] = pattern[i]; }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void renderGraphBufferCopyMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (gParams.source.length() < 4 || gParams.output.length() < 4) { return; }
    for (uint i = 0; i < 4; ++i) { gParams.output.data[i] = gParams.source.data[i]; }
}
