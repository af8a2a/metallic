struct TriangleVertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kTrianglePositions[3] =
{
    float2(0.0, -0.62),
    float2(0.62, 0.58),
    float2(-0.62, 0.58),
};

static const float3 kTriangleColors[3] =
{
    float3(0.18, 0.48, 1.0),
    float3(0.18, 0.95, 0.55),
    float3(1.0, 0.32, 0.32),
};

[shader("vertex")]
TriangleVertexOutput triangleVertexMain(uint vertexIndex : SV_VertexID)
{
    TriangleVertexOutput output;
    output.position = float4(kTrianglePositions[vertexIndex], 0.0, 1.0);
    output.color = kTriangleColors[vertexIndex];
    return output;
}

[shader("fragment")]
float4 triangleFragmentMain(TriangleVertexOutput input) : SV_Target0
{
    return float4(input.color, 1.0);
}
