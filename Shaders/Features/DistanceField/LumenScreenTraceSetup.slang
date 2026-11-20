module LumenScreenTraceSetup;

import ShaderCore;
import ParameterRoot;
import LumenScreenTrace;

using Metallic;
using Metallic.Lumen;

typealias ScreenTraceDepthImage = _Texture<float, __Shape2D, 0, 0, 0, 1, 0, 0, 6>;
typealias ScreenTraceColorImage = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 1>;

struct LumenScreenTraceSetupParams
{
    LumenScreenTraceView traceView;
    ResourceHandle<Texture2D<float>> depth;
    ResourceHandle<Texture2D<float4>> color;
    ResourceHandle<Texture2D<float>> historyDepth;
    ResourceHandle<Texture2D<float4>> historyColor;
    ResourceHandle<ScreenTraceDepthImage> previousDepth;
    ResourceHandle<ScreenTraceColorImage> previousColor;
    ResourceHandle<ScreenTraceDepthImage> currentDepth;
    ResourceHandle<ScreenTraceColorImage> currentColor;
    ResourceHandle<RWStructuredBuffer<float>> closestHZB;
    ResourceHandle<RWStructuredBuffer<LumenScreenTraceView>> view;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenScreenTraceSetupMain(uint3 dispatch : SV_DispatchThreadID)
{
    LumenScreenTraceSetupParams params = getParameters<LumenScreenTraceSetupParams>();
    uint2 extent = uint2(params.traceView.cameras.current.viewport.yz);
    if (any(dispatch.xy >= extent)) {
        return;
    }

    int3 pixel = int3(dispatch.xy, 0);
    float depth = resolveUniform(params.depth).Load(pixel);
    float4 color = resolveUniform(params.color).Load(pixel);
    float previousDepth = 0.0f;
    float4 previousColor = 0.0f;
    if (params.traceView.state.x != 0u) {
        previousDepth = resolveUniform(params.historyDepth).Load(pixel);
        previousColor = resolveUniform(params.historyColor).Load(pixel);
    }

    resolveUniform(params.previousDepth)[dispatch.xy] = previousDepth;
    resolveUniform(params.previousColor)[dispatch.xy] = previousColor;
    resolveUniform(params.currentDepth)[dispatch.xy] = depth;
    resolveUniform(params.currentColor)[dispatch.xy] = color;
    resolveUniform(params.closestHZB)[dispatch.y * extent.x + dispatch.x] = depth;
    if (all(dispatch.xy == 0u)) {
        resolveUniform(params.view)[0] = params.traceView;
    }
}

struct LumenClosestHZBReduceParams
{
    ResourceHandle<RWStructuredBuffer<float>> closestHZB;
    uint2 sourceExtent;
    uint2 destinationExtent;
    uint sourceOffset;
    uint destinationOffset;
    uint roundSource;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenClosestHZBReduceMain(uint3 dispatch : SV_DispatchThreadID)
{
    LumenClosestHZBReduceParams params = getParameters<LumenClosestHZBReduceParams>();
    if (any(dispatch.xy >= params.destinationExtent)) {
        return;
    }

    // UE 5.7.4 HZB.usf Gather4 clamps to the last in-viewport pair.
    uint2 r0 = min(dispatch.xy * 2u, max(params.sourceExtent, 2u) - 2u);
    uint2 r1 = min(r0 + 1u, params.sourceExtent - 1u);
    RWStructuredBuffer<float> closestHZB = resolveUniform(params.closestHZB);
    uint row0 = params.sourceOffset + r0.y * params.sourceExtent.x;
    uint row1 = params.sourceOffset + r1.y * params.sourceExtent.x;
    float4 deviceZ = float4(closestHZB[row0 + r0.x], closestHZB[row0 + r1.x],
        closestHZB[row1 + r0.x], closestHZB[row1 + r1.x]);
    float maxDeviceZ = max(max(max(deviceZ.x, deviceZ.y), deviceZ.z), deviceZ.w);
    // UE shares full precision within each 4-mip build batch and reads rounded R16F between batches.
    if (params.roundSource != 0u) {
        maxDeviceZ = roundUpScreenTraceDepth(maxDeviceZ);
    }
    closestHZB[params.destinationOffset + dispatch.y * params.destinationExtent.x + dispatch.x] = maxDeviceZ;
}
