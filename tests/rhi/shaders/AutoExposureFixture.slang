[[vk::binding(0, 0)]] RWTexture2D<float4> gOutput;
struct FixturePush { uint width; uint height; float luminance; uint outliers; };
[shader("compute")]
[numthreads(8, 8, 1)]
void autoExposureFixtureMain(uint3 pixel : SV_DispatchThreadID, uniform FixturePush push)
{
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float value = push.luminance;
    if (push.outliers == 1u && pixel.x < max(push.width / 20u, 1u)) { value = 1000000.0; }
    if (push.outliers == 2u) { value = asfloat(0x7fc00000u); }
    if (push.outliers == 3u && pixel.x < (push.width * 4u) / 5u) { value = 0.0; }
    gOutput[pixel.xy] = float4(value, value, value, 1.0);
}
