#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<DLSSSupportParams> gSupport;

struct StreamlineDLSSVertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 kFullscreenTrianglePositions[3] =
{
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};

static const float2 kFullscreenTriangleUvs[3] =
{
    float2(0.0, 0.0),
    float2(2.0, 0.0),
    float2(0.0, 2.0),
};

[shader("vertex")]
StreamlineDLSSVertexOutput streamlineDlssDepthVertexMain(uint vertexIndex : SV_VertexID)
{
    StreamlineDLSSVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    output.uv = kFullscreenTriangleUvs[vertexIndex];
    return output;
}

[shader("fragment")]
float streamlineDlssDepthFragmentMain(
    StreamlineDLSSVertexOutput input) : SV_Depth
{
    Texture2D<float> depthGuide = resolveDescriptor(gSupport.depth);

    uint width = 1;
    uint height = 1;
    depthGuide.GetDimensions(width, height);
    float2 uv = saturate(input.uv);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1, height - 1));
    return saturate(depthGuide.Load(int3(texel, 0)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void streamlineDlssAlphaMain(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    RWTexture2D<float4> outputImage = resolveDescriptor(gSupport.color);

    uint width = 1;
    uint height = 1;
    outputImage.GetDimensions(width, height);
    if (dispatchThreadId.x >= width || dispatchThreadId.y >= height) {
        return;
    }

    float4 color = outputImage[dispatchThreadId.xy];
    color.a = 1.0;
    outputImage[dispatchThreadId.xy] = color;
}
