struct ImageSampleVertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
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

static const float2 kFullscreenTriangleUvs[3] =
{
    float2(0.0, 0.0),
    float2(2.0, 0.0),
    float2(0.0, 2.0),
};

[shader("vertex")]
ImageSampleVertexOutput imageSampleVertexMain(uint vertexIndex : SV_VertexID)
{
    ImageSampleVertexOutput output;
    output.position = float4(kFullscreenTrianglePositions[vertexIndex], 0.0, 1.0);
    output.uv = kFullscreenTriangleUvs[vertexIndex];
    return output;
}

[shader("fragment")]
float4 imageSampleFragmentMain(ImageSampleVertexOutput input, uniform BindlessHeapPush bindless) : SV_Target0
{
    DescriptorHandle<Texture2D<float4>> imageHandle =
        DescriptorHandle<Texture2D<float4>>(uint2(bindless.imageShaderIndexBase, 0));
    Texture2D<float4> image = imageHandle;

    uint width = 1;
    uint height = 1;
    image.GetDimensions(width, height);

    float2 uv = saturate(input.uv);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1, height - 1));
    return image.Load(int3(texel, 0));
}
