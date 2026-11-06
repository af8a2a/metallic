module VirtualShadowMapDepthFixture;

import ShaderCore;
using Metallic;

struct DepthFixturePush
{
    uint sourceImage;
    uint standardDepth;
};

[shader("vertex")]
float4 vsmDepthFixtureVertexMain(uint vertex : SV_VertexID) : SV_Position
{
    const float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return float4(positions[vertex], 0, 1);
}

// Convert actual visibility-raster depth on the GPU; no synthetic receivers.
[shader("fragment")]
float vsmDepthFixtureFragmentMain(float4 position : SV_Position,
    uniform DepthFixturePush push) : SV_Depth
{
    Texture2D<float> source = resolveUniform(ResourceHandle<Texture2D<float>>(push.sourceImage));
    float depth = source.Load(int3(int2(position.xy), 0));
    return push.standardDepth != 0 ? 1.0 - depth : depth;
}
