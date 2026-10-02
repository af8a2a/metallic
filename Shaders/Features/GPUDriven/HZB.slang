#include "../../../Source/Runtime/Render/Core/HZBParameters.h"
[[vk::push_constant]] ConstantBuffer<HZBParameters> gHZB;

[shader("compute")]
[numthreads(8, 8, 1)]
void hzbMain(uint3 id : SV_DispatchThreadID)
{
    HZBParameters p = gHZB;
    uint count = p.hzb.length();
    if (p.hzb.data == nullptr || p.width == 0u || p.height == 0u ||
        id.x >= p.width || id.y >= p.height || p.destinationOffset > count ||
        p.height > (count - p.destinationOffset) / p.width) { return; }
    uint destination = p.destinationOffset + id.y * p.width + id.x;
    if (p.mipLevel == 0u) {
        Texture2D<float> depth = resolveDescriptor(p.depth);
        p.hzb.data[destination] = depth.Load(int3(id.xy, 0));
        return;
    }
    if (p.sourceWidth == 0u || p.sourceHeight == 0u || p.sourceOffset > count ||
        p.sourceHeight > (count - p.sourceOffset) / p.sourceWidth) { return; }
    uint2 base = id.xy * 2u;
    uint2 limit = uint2(p.sourceWidth - 1u, p.sourceHeight - 1u);
    uint2 s0 = min(base, limit), s1 = min(base + uint2(1, 0), limit);
    uint2 s2 = min(base + uint2(0, 1), limit), s3 = min(base + uint2(1, 1), limit);
    float d0 = p.hzb.data[p.sourceOffset + s0.y * p.sourceWidth + s0.x];
    float d1 = p.hzb.data[p.sourceOffset + s1.y * p.sourceWidth + s1.x];
    float d2 = p.hzb.data[p.sourceOffset + s2.y * p.sourceWidth + s2.x];
    float d3 = p.hzb.data[p.sourceOffset + s3.y * p.sourceWidth + s3.x];
    p.hzb.data[destination] = p.reversedZ != 0u
        ? min(min(d0, d1), min(d2, d3)) : max(max(d0, d1), max(d2, d3));
}
