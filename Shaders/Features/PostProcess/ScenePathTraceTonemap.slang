#include "../../../Source/Runtime/Render/Core/PathTraceStageParameters.h"
[[vk::push_constant]] ConstantBuffer<PathTraceTonemapParams> gStage;

import ShaderCore;
using Metallic;

// Tonemap pass used with the NRC query mode. The path tracer writes linear
// HDR radiance into the history texture, the NRC resolve adds the predicted
// query radiance on top, and this pass applies the same Reinhard + gamma
// curve the inline path uses when no external resolve is needed.

#define gSource resolveUniform(gStage.source)

#define gOutput resolveUniform(gStage.output)

#define gHistoryPrevious resolveUniform(gStage.historyPrevious)

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceTonemapMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTraceTonemapPush push = gStage.settings;
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    float3 color = gSource[pixel].rgb;
    // Native NRC resolve adds a neural prediction to the traced prefix. On a
    // cache reset this prediction can undershoot the nonnegative radiance
    // learned by this path (including a fully dark scene). Project finite
    // resolved radiance to that domain before camera transport and history.
    // Keep nonfinite values observable so an invalid resolve is still detected.
    if (all(isfinite(color))) {
        color = max(color, float3(0.0));
    }
    // NRC resolves complete surface radiance before this stage. Keep the
    // camera segment outside the trained/query radiance and apply it once.
    if (gStage.primaryAerial.index != 0xffffffffu) {
        let aerial = resolveUniform(gStage.primaryAerial);
        uint count, stride;
        aerial.GetDimensions(count, stride);
        uint index = (pixel.y * push.width + pixel.x) * 2u;
        if (index + 1u < count) {
            color = aerial[index].xyz * color + aerial[index + 1u].xyz;
        }
    }
    if (push.hasHistory != 0u) {
        color = lerp(gHistoryPrevious[pixel].rgb, color,
            1.0 / float(push.accumulationFrame + 1u));
    }
    gSource[pixel] = float4(color, 1.0);
    if (push.outputLinear != 0u) {
        gOutput[pixel] = float4(color, 1.0);
        return;
    }
    color = max(Metallic.WorkingColor::toLinearRec709(color) * push.exposure, float3(0.0, 0.0, 0.0));
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(max(color, float3(0.0, 0.0, 0.0)), 1.0 / 2.2);
    color = saturate(color);

    gOutput[pixel] = float4(color, 1.0);
}
