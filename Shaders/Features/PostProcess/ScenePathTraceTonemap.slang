// Tonemap pass used with the NRC query mode. The path tracer writes linear
// HDR radiance into the history texture, the NRC resolve adds the predicted
// query radiance on top, and this pass applies the same Reinhard + gamma
// curve the inline path uses when no external resolve is needed.

struct ScenePathTraceTonemapPush
{
    uint width;
    uint height;
    float exposure;
    uint outputLinear;
    uint hasHistory;
    uint accumulationFrame;
};

[[vk::binding(0, 0)]]
RWTexture2D<float4> gSource;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gHistoryPrevious;

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceTonemapMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                               uniform ScenePathTraceTonemapPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 color = gSource[pixel].rgb;
    if (push.hasHistory != 0u) {
        color = lerp(gHistoryPrevious[pixel].rgb, color,
            1.0 / float(push.accumulationFrame + 1u));
    }
    gSource[pixel] = float4(color, 1.0);
    if (push.outputLinear != 0u) {
        gOutput[pixel] = float4(color, 1.0);
        return;
    }
    color = max(color * push.exposure, float3(0.0, 0.0, 0.0));
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
    color = saturate(color);

    gOutput[pixel] = float4(color, 1.0);
}
