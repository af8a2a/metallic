import ShaderCore;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;

#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"
import ParameterRoot;

[shader("compute")]
[numthreads(4, 4, 4)]
void composeColorGradingLUT(uint3 id : SV_DispatchThreadID)
{
    ColorGradingLUTParams params = getParameters<ColorGradingLUTParams>();
    RWTexture3D<float4> lut = resolveDescriptor(params.output);
    GradingPush p = params.display;
    GradingResources resources = {
        resolveDescriptor(params.custom0), resolveDescriptor(params.custom1),
        resolveDescriptor(params.custom2), resolveDescriptor(params.custom3),
        resolveDescriptor(params.reach), resolveDescriptor(params.gamut),
        resolveDescriptor(params.gammaTable), resolveDescriptor(params.sampler)
    };
    float3 linear = gradingLutDecode(float3(id) / 63.0);
    lut[id] = float4(evaluateColorGrading(linear, p.grade, resources, p.transform,
        p.hdr != 0u, p.peak, p.paperWhite), 1.0);
}
