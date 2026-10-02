#include "../../../Source/Runtime/Render/Core/BindlessSmokeParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<BindlessSmokeParameters> gBindlessSmoke;

struct BindlessSmokeVertexOutput
{
    float4 position : SV_Position;
};

static const float2 kFullscreenTrianglePositions[3] =
{
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};

[shader("vertex")]
BindlessSmokeVertexOutput bindlessSmokeVertexMain(uint vertexIndex : SV_VertexID)
{
    BindlessSmokeVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 bindlessSmokeFragmentMain() : SV_Target0
{
    Texture2D<float4> source = resolveDescriptor(gBindlessSmoke.sourceImage);
    return source.Load(int3(0, 0, 0));
}
