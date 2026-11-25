module TextureFootprintProbe;

import Core;
import Material;
using Metallic;
#include "../TestResourceParameters.h"
using Metallic.Material;

struct TextureFootprintProbePush
{
    uint width;
    uint height;
    uint index;
    uint orthographic;
    float aspect;
    float verticalFovRadians;
    float orthographicHeight;
    float reserved;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void textureFootprintProbeMain()
{
    TextureFootprintProbePush push = getConstants<TextureFootprintProbePush>();
    float2 cone = primaryTextureRayCone(
        uint2(push.width, push.height), push.aspect, push.verticalFovRadians,
        push.orthographicHeight, push.orthographic != 0u);
    float2 footprint = float2(cone.x + cone.y, (cone.x + cone.y) * 0.5);
    float sourceLod = projectedTextureLod(footprint, uint2(8192, 1));
    float residentLod = projectedTextureLod(footprint, uint2(32, 1));
    resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<TextureFootprintProbeResources>().output)[push.index] =
        float4(cone, sourceLod, residentLod);
}
