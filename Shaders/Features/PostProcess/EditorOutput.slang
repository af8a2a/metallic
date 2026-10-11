import Core;
using Metallic;

struct OutputPush
{
    ResourceHandle<Texture2D<float4>> source;
};
[[vk::push_constant]] ConstantBuffer<OutputPush> output;

[shader("vertex")]
float4 editorOutputVertex(uint vertex : SV_VertexID) : SV_Position
{
    float2 uv = float2((vertex << 1u) & 2u, vertex & 2u);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 editorOutputPQFragment(float4 position : SV_Position) : SV_Target0
{
    // UI alpha blending was completed in the FP16 linear composition target.
    float3 color = resolveUniform(output.source).Load(int3(int2(position.xy), 0)).rgb;
    return float4(displayScRgbToHDR10(color), 1.0);
}
