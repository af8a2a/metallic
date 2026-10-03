#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"
import ShaderCore;
using Metallic;

// Force storage decorations for the same shared type used by raw parameter loads.
[[vk::push_constant]] ConstantBuffer<ColorGradingLUTParams> params;
[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    resolveUniform(params.output)[uint3(0)] = float4(params.display.peak, 0, 0, 1);
}
