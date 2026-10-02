import ParameterRoot;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
#include "../../../Shaders/Features/VisibilityBuffer/VisibilityStreamDecode.slang"
struct StreamDecodeParams { DataSpan<uint> pages; DataSpan<uint2> table; DataSpan<uint> output; };
[shader("compute")]
[numthreads(16, 1, 1)]
void streamDataDecodeMain(uint3 id : SV_DispatchThreadID)
{
    StreamDecodeParams p = getParameters<StreamDecodeParams>();
    if (id.x >= 9 || !p.output.contains(id.x)) { return; }
    uint page = id.x == 1 ? 1 : 0, cluster = id.x == 2 ? 1 : 0, triangle = id.x == 3 ? 1 : 0;
    if (id.x == 4) { p.pages.count = 27; }
    if (id.x == 5) { p.pages.count = 61; }
    if (id.x == 6) { p.table.count = 0; p.table.data = nullptr; }
    if (id.x == 7) { p.pages.stride = 8; p.pages.data = nullptr; }
    if (id.x == 8) { p.pages.count = 62; } // Final partial word must fit too.
    float3 a, b, c;
    bool valid = decodeStreamTriangleAtPage(page, cluster, triangle,
        StreamDataBuffer<uint>(p.pages), StreamDataBuffer<uint2>(p.table), p.pages.length(), a, b, c);
    p.output.data[id.x] = valid ? (all(a == float3(0)) && all(b == float3(1,0,0)) && all(c == float3(0,1,0)) ? 1u : 2u) : 0u;
}
