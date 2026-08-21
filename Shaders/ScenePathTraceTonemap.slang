// Tonemap pass used with the NRC query mode. The path tracer writes linear
// HDR radiance into the history texture, the NRC resolve adds the predicted
// query radiance on top, and this pass applies the same Reinhard + gamma
// curve the inline path uses when no external resolve is needed.

struct ScenePathTraceTonemapPush
{
    uint width;
    uint height;
    uint padding0;
    uint padding1;
};

[[vk::binding(0, 0)]]
RWTexture2D<float4> gSource;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceTonemapMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                               uniform ScenePathTraceTonemapPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 color = max(gSource[pixel].rgb, float3(0.0, 0.0, 0.0));
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
    color = saturate(color);

    gOutput[pixel] = float4(color, 1.0);
}
