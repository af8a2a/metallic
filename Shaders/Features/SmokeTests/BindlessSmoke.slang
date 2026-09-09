struct BindlessSmokeVertexOutput
{
    float4 position : SV_Position;
};

struct BindlessHeapPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
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
float4 bindlessSmokeFragmentMain(uniform BindlessHeapPush bindless) : SV_Target0
{
    DescriptorHandle<Texture2D<float4>> sourceHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(bindless.imageShaderIndexBase, 0));
    Texture2D<float4> source = sourceHandle;
    return source.Load(int3(0, 0, 0));
}
