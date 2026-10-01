import Core;
using Metallic;

#ifndef DISPLAY_HDR
#define DISPLAY_HDR 0
#endif
#ifndef DISPLAY_SRGB_ATTACHMENT
#define DISPLAY_SRGB_ATTACHMENT 0
#endif
#ifndef DISPLAY_SCRGB_IMAGE
#define DISPLAY_SCRGB_IMAGE 0
#endif
#ifndef DISPLAY_WHITE_NITS
#define DISPLAY_WHITE_NITS 203.0
#endif

struct EditorPush { float2 scale; float2 translate; };
[[vk::push_constant]] ConstantBuffer<EditorPush> push;
[[vk::binding(0, 0)]] Texture2D<float4> sourceTexture;
[[vk::binding(0, 1)]] SamplerState sourceSampler;

struct VertexInput {
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR0;
};
struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

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
    return float4(displayScRgbToHDR10(sourceTexture.Load(int3(int2(position.xy), 0)).rgb), 1.0);
}

[shader("vertex")]
VertexOutput editorDisplayVertex(VertexInput input)
{
    VertexOutput output;
    output.position = float4(input.position * push.scale + push.translate, 0.0, 1.0);
    output.color = input.color;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 editorDisplayFragment(VertexOutput input) : SV_Target0
{
    float4 sampled = sourceTexture.Sample(sourceSampler, input.uv);
    float3 color;
#if DISPLAY_SCRGB_IMAGE
    color = sampled.rgb * displaySrgbToLinear(input.color.rgb);
#if !DISPLAY_HDR
    // Detached ImGui windows use SDR swapchains. Reference white stays stable;
    // highlights clip in that preview, while the main window retains HDR.
    color = saturate(color * (80.0 / DISPLAY_WHITE_NITS));
#endif
#else
    color = displaySrgbToLinear(sampled.rgb) * displaySrgbToLinear(input.color.rgb);
#if DISPLAY_HDR
    color *= DISPLAY_WHITE_NITS / 80.0;
#endif
#endif
#if !DISPLAY_HDR && !DISPLAY_SRGB_ATTACHMENT
    color = displayLinearToSrgb(color);
#endif
    return float4(color, sampled.a * input.color.a);
}
