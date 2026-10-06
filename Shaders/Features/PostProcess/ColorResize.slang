import ShaderCore;
using Metallic;
#include "../../../Source/Runtime/Render/Core/ColorResizeParameters.h"

[[vk::push_constant]] ConstantBuffer<ColorResizeParams> gParams;
#define gSource resolveUniform(gParams.source)
#define gOutput resolveUniform(gParams.output)

// Preserve source encoding. This is a spatial resize, with no exposure, grading
// or color-space conversion, used to bypass temporal HDR SDKs for debug images.
[shader("compute")]
[numthreads(8, 8, 1)]
void colorResizeMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height, sourceWidth, sourceHeight;
    gOutput.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    gSource.GetDimensions(sourceWidth, sourceHeight);
    float2 pixel = (float2(id.xy) + 0.5) * float2(sourceWidth, sourceHeight) / float2(width, height) - 0.5;
    int2 origin = int2(floor(pixel));
    int2 maximum = int2(sourceWidth, sourceHeight) - 1;
    float2 weight = frac(pixel);
    float4 c00 = gSource.Load(int3(clamp(origin, int2(0), maximum), 0));
    float4 c10 = gSource.Load(int3(clamp(origin + int2(1, 0), int2(0), maximum), 0));
    float4 c01 = gSource.Load(int3(clamp(origin + int2(0, 1), int2(0), maximum), 0));
    float4 c11 = gSource.Load(int3(clamp(origin + int2(1, 1), int2(0), maximum), 0));
    gOutput[id.xy] = lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y);
}
