#include "../../Libraries/Resources/ComputeResources.slang"

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

#define gSource METALLIC_RESOURCE(RWTexture2D<float4>, 0)

#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 1)

#define gHistoryPrevious METALLIC_RESOURCE(RWTexture2D<float4>, 2)

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceTonemapMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTraceTonemapPush push = METALLIC_CONSTANTS(ScenePathTraceTonemapPush);
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
