#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

struct FixturePush { uint width; uint height; uint seed; uint reversedZ; };
[shader("compute")]
[numthreads(8, 8, 1)]
void hzbSpdFixtureMain(uint3 id : SV_DispatchThreadID)
{
    FixturePush push = METALLIC_CONSTANTS(FixturePush);
    if (id.x >= push.width || id.y >= push.height) { return; }
    uint hash = (id.x * 73u + id.y * 137u + push.seed * 31u) % 1024u;
    // Each 64x64 tile has a different range so the last-group tail must
    // actually combine spatially distinct tile results.
    uint tile = ((id.x / 64u) * 7u + (id.y / 64u) * 13u + push.seed * 17u) % 64u;
    float depth = float(tile * 1024u + hash + 1u) / 65537.0;
    if (id.x + 1u == push.width || id.y + 1u == push.height) {
        depth = push.reversedZ != 0u ? 0.0 : 1.0;
    }
    if (id.x == 0u && id.y == 0u) { depth = asfloat(0x7fc00000u); }
    METALLIC_RESOURCE(RWTexture2D<float>, 0)[id.xy] = depth;
    if (id.x == 0u && id.y == 0u) {
        METALLIC_RESOURCE(RWStructuredBuffer<uint>, 1)[0] = 0u;
        METALLIC_RESOURCE(RWStructuredBuffer<uint>, 1)[1] = push.seed;
    }
}
