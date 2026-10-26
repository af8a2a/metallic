#language slang 2026
module VirtualShadowMapContent;

import ShaderCore;
import VirtualShadowMapGeometry;
using Metallic;
using Metallic.Shadow;

struct ContentPush { uint geometry, output, count, reserved; };
[[vk::push_constant]] ConstantBuffer<ContentPush> push;

[shader("compute")]
[numthreads(64, 1, 1)]
void contentMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= push.count) { return; }
    VirtualShadowMapTriangle triangle;
    bool valid = loadVirtualShadowMapTriangle(push.geometry, id.x, triangle);
    triangle.reserved1 = valid ? 1u : 0u;
    RWStructuredBuffer<VirtualShadowMapTriangle> output =
        resolveUniform(ResourceHandle<RWStructuredBuffer<VirtualShadowMapTriangle>>(push.output));
    output[id.x] = triangle;
}
