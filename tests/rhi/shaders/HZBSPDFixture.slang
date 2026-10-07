import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct FixturePush { uint width; uint height; uint seed; };
[shader("compute")]
[numthreads(8, 8, 1)]
void hzbSpdFixtureMain(uint3 id : SV_DispatchThreadID)
{
    FixturePush push = getConstants<FixturePush>();
    if (id.x >= push.width || id.y >= push.height) { return; }
    uint hash = (id.x * 73u + id.y * 137u + push.seed * 31u) % 1024u;
    // Each 64x64 tile has a different range so the last-group tail must
    // actually combine spatially distinct tile results.
    uint tile = ((id.x / 64u) * 7u + (id.y / 64u) * 13u + push.seed * 17u) % 64u;
    float depth = float(tile * 1024u + hash + 1u) / 65537.0;
    if (id.x + 1u == push.width || id.y + 1u == push.height) {
        depth = 0.0;
    }
    if (id.x == 0u && id.y == 0u) { depth = asfloat(0x7fc00000u); }
    resolveUniform(getResourceParameters<HZBSPDFixtureResources>().depth)[id.xy] = depth;
    if (id.x == 0u && id.y == 0u) {
        resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<HZBSPDFixtureResources>().counter)[0] = 0u;
        resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<HZBSPDFixtureResources>().counter)[1] = push.seed;
    }
}
