#include "../../../Source/Runtime/Render/Core/StreamSceneParameters.h"
#include "../StreamProbeParameters.h"
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
#include "../../../Shaders/Features/PathTracing/StreamSceneDecode.slang"
[[vk::push_constant]] ConstantBuffer<StreamProbeParameters> gStreamProbe;
[shader("compute")]
[numthreads(16, 1, 1)]
void streamDataDecodeMain(uint3 id : SV_DispatchThreadID)
{
    StreamProbeParameters p = gStreamProbe;
    if (id.x >= 24 || !p.output.contains(id.x)) { return; }
    if (id.x == 9) { p.output.data[id.x] = p.stream.instances.length() == 1 ? p.stream.instances.data[0].materialIndex : 0; return; }
    if (id.x == 10) { p.output.data[id.x] = p.stream.header.contains(2) ? p.stream.header.data[2] : 0; return; }
    if (id.x >= 11) {
        // Exercise the same named-resource decoder used by CLAS surface/alpha queries.
        uint instanceIndex = id.x == 12 ? 1u : 0u;
        if (id.x == 13) { p.stream.instances.stride = 4; }
        if (id.x == 14) { p.stream.instances.data = nullptr; }
        if (id.x == 15) { p.stream.header.count = 2; }
        if (id.x == 16) { p.stream.header.stride = 8; }
        if (id.x == 17) { p.stream.header.data = nullptr; }
        if (id.x == 18) { p.stream.pages.data = nullptr; }
        if (id.x == 19) { p.stream.pageTable.data = nullptr; }
        if (id.x == 20) { p.stream.pages.stride = 8; }
        if (id.x == 21) { p.stream.pageTable.stride = 4; }
        if (id.x == 22) { p.stream.header.data = &p.stream.header.data[1]; } // Padded fourth word is zero cluster stride.
        StreamRayInstance instance;
        float3 normals[3]; float2 uvs[3]; float4 tangents[3];
        uint attributes = 0;
        float3 a, b, c;
        bool valid = loadStreamRayInstance(p.stream, instanceIndex, instance);
        if (valid) { valid = decodeStreamRaySurface(p.stream, id.x == 23 ? 0xffffffffu : 0u, 0u,
            a, b, c, normals, uvs, tangents, attributes); }
        p.output.data[id.x] = valid && instance.materialIndex == 17u && attributes == 0u && all(normals[0] == float3(0)) && all(uvs[0] == float2(0)) &&
            all(tangents[0] == float4(0)) &&
            all(a == float3(0)) && all(b == float3(1,0,0)) && all(c == float3(0,1,0)) ? 1u : 0u;
        return;
    }
    uint page = id.x == 1 ? 1 : 0, cluster = id.x == 2 ? 1 : 0, triangle = id.x == 3 ? 1 : 0;
    if (id.x == 4) { p.stream.pages.count = 27; }
    if (id.x == 5) { p.stream.pages.count = 61; }
    if (id.x == 6) { p.stream.pageTable.count = 0; p.stream.pageTable.data = nullptr; }
    if (id.x == 7) { p.stream.pages.stride = 8; p.stream.pages.data = nullptr; }
    if (id.x == 8) { p.stream.pages.count = 62; } // Final partial word must fit too.
    float3 a, b, c;
    bool valid = decodeStreamTriangleAtPage(page, cluster, triangle,
        StreamDataBuffer<uint>(p.stream.pages), StreamDataBuffer<uint2>(p.stream.pageTable), p.stream.pages.length(), a, b, c);
    p.output.data[id.x] = valid ? (all(a == float3(0)) && all(b == float3(1,0,0)) && all(c == float3(0,1,0)) ? 1u : 2u) : 0u;
}
