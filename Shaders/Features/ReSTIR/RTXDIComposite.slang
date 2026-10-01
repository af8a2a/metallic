#include "../../../Source/Runtime/Render/Core/RTXDIPostProcessParameters.h"

[[vk::push_constant]] ConstantBuffer<RTXDICompositeParams> gParams;

import ShaderCore;
using Metallic;

#define gDenoisedDiffuse resolveDescriptor(gParams.denoisedDiffuse)

#define gDenoisedSpecular resolveDescriptor(gParams.denoisedSpecular)

#define gBaseColorMetalness resolveDescriptor(gParams.baseColorMetalness)

#define gEmissive resolveDescriptor(gParams.emissive)

#define gOutput resolveDescriptor(gParams.output)

float3 toneMap(float3 color, float exposure)
{
    color = 1.0 - exp(-max(color, 0.0) * exposure);
    return pow(saturate(color), 1.0 / 2.2);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rtxdiCompositeMain(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    RTXDICompositePush push = gParams.settings;
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float4 baseColorMetalness = gBaseColorMetalness[pixel];
    float3 baseColor = saturate(baseColorMetalness.rgb);
    float metalness = saturate(baseColorMetalness.a);
    float3 diffuseAlbedo = baseColor * (1.0 - metalness);
    float3 specularF0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metalness);
    float3 color = gDenoisedDiffuse[pixel].rgb * diffuseAlbedo;
    color += gDenoisedSpecular[pixel].rgb * specularF0;
    color += gEmissive[pixel].rgb;
    gOutput[pixel] = float4(push.outputLinear != 0u ? color : toneMap(color, push.exposure), 1.0);
}
