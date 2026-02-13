// Passthrough: samples input texture and writes it directly (no processing).
// Useful as a debug output pass to bypass tonemapping.

Texture2D<float4> inputTexture;
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
    float3 color = inputTexture.Sample(linearSampler, input.uv).rgb;
    return float4(saturate(color), 1.0);
}
