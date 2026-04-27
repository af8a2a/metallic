// Passthrough: samples input texture and writes it directly (no processing).
// Useful as a debug output pass to bypass tonemapping.

struct OutputUniforms {
    uint compareMode; // 0=single source, 1=left/right split compare
    uint3 _pad0;
};

[[vk::push_constant]] ConstantBuffer<OutputUniforms> outputUniforms;

Texture2D<float4> inputTexture;
Texture2D<float4> compareTexture;
SamplerState linearSampler;

struct VSOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOut vertexMain(uint vertexID : SV_VertexID) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 uvs[3] = { float2(0.0, 0.0), float2(2.0, 0.0), float2(0.0, 2.0) };

    VSOut output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = uvs[vertexID];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOut input) : SV_Target {
    float2 uv = input.uv;
    float3 color;
    if (outputUniforms.compareMode != 0u) {
        bool rightPane = uv.x >= 0.5;
        float2 paneUv = float2(rightPane ? (uv.x - 0.5) * 2.0 : uv.x * 2.0, uv.y);
        color = rightPane
            ? compareTexture.Sample(linearSampler, paneUv).rgb
            : inputTexture.Sample(linearSampler, paneUv).rgb;

        float3 paneTint = rightPane ? float3(1.0, 0.55, 0.15) : float3(0.15, 0.55, 1.0);
        if (abs(uv.x - 0.5) < 0.0025) {
            color = float3(1.0, 1.0, 1.0);
        } else if (uv.y < 0.025 || uv.x < 0.004 || uv.x > 0.996) {
            color = lerp(color, paneTint, 0.7);
        }
    } else {
        color = inputTexture.Sample(linearSampler, uv).rgb;
    }

    return float4(saturate(color), 1.0);
}
