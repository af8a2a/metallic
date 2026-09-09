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

static const float2 kFullscreenTrianglePositions[3] =
{
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
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

[shader("vertex")]
TriangleVertexOutput reversedDepthNearVertexMain(uint vertexIndex : SV_VertexID)
{
    TriangleVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.75, 1.0);
    output.color = float3(0.0, 1.0, 0.0);
    return output;
}

[shader("vertex")]
TriangleVertexOutput reversedDepthFarVertexMain(uint vertexIndex : SV_VertexID)
{
    TriangleVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.25, 1.0);
    output.color = float3(1.0, 0.0, 0.0);
    return output;
}

[shader("fragment")]
float4 solidGreenFragmentMain(TriangleVertexOutput input) : SV_Target0
{
    return float4(0.0, 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 solidRedFragmentMain(TriangleVertexOutput input) : SV_Target0
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
