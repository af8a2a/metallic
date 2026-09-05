[[vk::binding(0, 0)]]
RWTexture2D<float4> gOutput;

[[vk::binding(1, 0)]]
Texture2D<float4> gSource;

[shader("compute")]
[numthreads(8, 8, 1)]
void finalBlitUvMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    gOutput[pixel] = float4(uv, 0.0, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void finalBlitMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    uint sourceWidth, sourceHeight;
    gSource.GetDimensions(sourceWidth, sourceHeight);
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float2 sourcePixel = uv * float2(sourceWidth, sourceHeight) - 0.5;
    int2 origin = int2(floor(sourcePixel));
    int2 maximum = int2(sourceWidth, sourceHeight) - 1;
    float2 weight = frac(sourcePixel);
    // Explicit bilinear filtering also supports float RT formats without a
    // hardware linear-filter feature requirement.
    float3 c00 = gSource.Load(int3(clamp(origin, int2(0), maximum), 0)).rgb;
    float3 c10 = gSource.Load(int3(clamp(origin + int2(1, 0), int2(0), maximum), 0)).rgb;
    float3 c01 = gSource.Load(int3(clamp(origin + int2(0, 1), int2(0), maximum), 0)).rgb;
    float3 c11 = gSource.Load(int3(clamp(origin + int2(1, 1), int2(0), maximum), 0)).rgb;
    gOutput[pixel] = float4(lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y), 1.0);
}
