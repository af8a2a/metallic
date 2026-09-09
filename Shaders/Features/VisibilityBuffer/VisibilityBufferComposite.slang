#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"

struct FullscreenVertexOutput
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
FullscreenVertexOutput visibilityBufferCompositeVertexMain(uint vertexIndex : SV_VertexID)
{
    FullscreenVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    return output;
}

// Debug display only: no triangle attribute reconstruction, textures or BSDF.
// Both resident and stream producers share the same packed visibility namespace.
float3 visibilityHashColor(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return 0.25 + 0.75 * float3(
        value & 0xffu, (value >> 8u) & 0xffu, (value >> 16u) & 0xffu) / 255.0;
}

[shader("fragment")]
float4 visibilityBufferCompositeFragmentMain(
    FullscreenVertexOutput input,
    uniform GPUDrivenPreviewPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<Texture2D<uint>> visibilityHandle =
        DescriptorHandle<Texture2D<uint>>(uint2(
            push.imageShaderIndexBase + push.visibilityImage, 0));
    GPUDrivenPreviewParams params = paramsHandle[0];
    uint2 pixel = min(uint2(input.position.xy), uint2(params.width - 1u, params.height - 1u));
    Texture2D<uint> visibility = visibilityHandle;
    uint packedVisibility = visibility.Load(int3(pixel, 0));
    uint recordIndex;
    uint triangleIndex;
    if (params.mode == 6u || !unpackVisibilityId(packedVisibility, recordIndex, triangleIndex)) {
        return params.clearColor;
    }
    if (params.mode == 4u) {
        DescriptorHandle<Texture2D<float>> depthHandle =
            DescriptorHandle<Texture2D<float>>(uint2(
                push.imageShaderIndexBase + push.depthImage, 0));
        Texture2D<float> depth = depthHandle;
        float z = saturate(depth.Load(int3(pixel, 0)));
        return float4(z, z, z, 1.0);
    }
    if (params.mode == 5u) {
        return float4(1.0);
    }
    uint seed = params.mode == 3u ? packedVisibility : recordIndex + 1u;
    return float4(visibilityHashColor(seed), 1.0);
}
