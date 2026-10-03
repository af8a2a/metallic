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
    RWTexture3D<float4> lut = resolveUniform(params.output);
    GradingPush p = params.display;
    GradingResources resources = {
        resolveUniform(params.custom0), resolveUniform(params.custom1),
        resolveUniform(params.custom2), resolveUniform(params.custom3),
        resolveUniform(params.reach), resolveUniform(params.gamut),
        resolveUniform(params.gammaTable), resolveUniform(params.sampler)
    };
    float3 linear = gradingLutDecode(float3(id) / 63.0);
    lut[id] = float4(evaluateColorGrading(linear, p.grade, resources, p.transform,
        p.hdr != 0u, p.peak, p.paperWhite), 1.0);
}
