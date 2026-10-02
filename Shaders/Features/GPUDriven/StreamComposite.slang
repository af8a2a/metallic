#include "../../../Source/Runtime/Render/Core/StreamCompositeParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<StreamCompositeParameters> gParams;

struct GPUDrivenStreamAssetFullscreenOutput
{
    float4 position : SV_Position;
};

float4 unpackStreamColor(uint packed)
{
    return float4(
        float(packed & 0xffu),
        float((packed >> 8u) & 0xffu),
        float((packed >> 16u) & 0xffu),
        float((packed >> 24u) & 0xffu)) / 255.0;
}

[shader("vertex")]
GPUDrivenStreamAssetFullscreenOutput gpuDrivenStreamAssetCompositeVertexMain(
    uint vertexID : SV_VertexID)
{
    GPUDrivenStreamAssetFullscreenOutput output;
    float2 position = vertexID == 0u
        ? float2(-1.0, -1.0)
        : (vertexID == 1u ? float2(3.0, -1.0) : float2(-1.0, 3.0));
    output.position = float4(position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 gpuDrivenStreamAssetCompositeFragmentMain(
    GPUDrivenStreamAssetFullscreenOutput input) : SV_Target0
{
    if (gParams.width == 0 || gParams.height == 0) { return float4(0); }
    uint2 pixel = min(uint2(input.position.xy), uint2(gParams.width - 1u, gParams.height - 1u));
    // Validate multiplication before forming the flattened offset.
    uint count = gParams.colors.length();
    if (pixel.y > count / gParams.width) { return float4(0); }
    uint row = pixel.y * gParams.width;
    if (row >= count || pixel.x >= count - row) { return float4(0); }
    return unpackStreamColor(gParams.colors.data[row + pixel.x]);
}
