import GPUDriven;
using Metallic.GPUDriven;
#include "../../../Shaders/Features/VisibilityBuffer/VisibilityTessellation.slang"
#include "../../../Shaders/Features/VisibilityBuffer/VisibilityTessellationSplit.slang"

struct ProbeCase { float4 position[3]; float4 eyeNear; float4 forwardScale; float4 options; };
struct Push { uint images, buffers, input, output; };

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Push push)
{
    if (lane >= kTessRootsPerTask) { return; }
    uint index = group.x * kTessRootsPerTask + lane;
    DescriptorHandle<StructuredBuffer<ProbeCase>> cases = DescriptorHandle<StructuredBuffer<ProbeCase>>(uint2(push.buffers + push.input, 0));
    DescriptorHandle<RWStructuredBuffer<uint4>> output = DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.buffers + push.output, 0));
    ProbeCase test = cases[index];
    float3 source[3];
    for (uint i = 0u; i < 3u; ++i) { source[i] = test.position[i].xyz; }
    TessView view;
    view.eye = test.eyeNear.xyz; view.nearPlane = test.eyeNear.w;
    view.forward = test.forwardScale.xyz; view.pixelScale = test.forwardScale.w;
    view.orthographic = test.options.w != 0.0;
    uint count = tessBuildLeaves(lane, source, view, test.options.x, uint(test.options.y), uint(test.options.z));
    uint3 rootRates = tessPatchRates(uint3(0u,4096u,4096u << 16u), source, view, test.options.x, uint(test.options.y) + 1u);
    uint mask = (rootRates.x > test.options.y ? 1u : 0u) | (rootRates.y > test.options.y ? 2u : 0u) | (rootRates.z > test.options.y ? 4u : 0u);
    output[index * 65u] = uint4(count, mask, 0u, 0u);
    for (uint i = 0u; i < count; ++i) {
        uint base = (lane * kTessLeavesPerRoot + i) * 3u;
        uint3 patch = uint3(g_TessPatches.patches[base],g_TessPatches.patches[base+1u],g_TessPatches.patches[base+2u]);
        uint3 rates = tessPatchRates(patch, source, view, test.options.x, uint(test.options.y));
        output[index * 65u + 1u + i] = uint4(patch, rates.x | (rates.y << 8u) | (rates.z << 16u));
    }
}
