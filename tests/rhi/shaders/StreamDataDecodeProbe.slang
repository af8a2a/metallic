#include "../../../Source/Runtime/Render/Core/StreamSceneParameters.h"
import ParameterRoot;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
#include "../../../Shaders/Features/VisibilityBuffer/VisibilityStreamDecode.slang"
struct StreamDecodeParams { StreamSceneParameters stream; DataSpan<uint> output; };
[shader("compute")]
[numthreads(16, 1, 1)]
void streamDataDecodeMain(uint3 id : SV_DispatchThreadID)
{
    StreamDecodeParams p = getParameters<StreamDecodeParams>();
    if (id.x >= 11 || !p.output.contains(id.x)) { return; }
    if (id.x == 9) { p.output.data[id.x] = p.stream.instances.length() == 1 ? p.stream.instances.data[0].materialIndex : 0; return; }
    if (id.x == 10) { p.output.data[id.x] = p.stream.header.contains(2) ? p.stream.header.data[2] : 0; return; }
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
