#include "../NativeNestedProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<NativeNestedProbeParameters> gProbe;

[shader("compute")]
[numthreads(32, 1, 1)]
void nestedBufferMain(uint3 id : SV_DispatchThreadID)
{
    uint base = id.x * 1000u;
    NestedRecord record;
    record.prefix = base + uint4(1, 2, 3, 4);
    record.children[0].first = base + uint4(5, 6, 7, 8);
    record.children[0].second = base + uint4(9, 10, 11, 12);
    record.children[1].first = base + uint4(13, 14, 15, 16);
    record.children[1].second = base + uint4(17, 18, 19, 20);
    record.suffix = base + uint4(21, 22, 23, 24);
    resolveDescriptor(gProbe.records)[id.x] = record;
    DeviceMemoryBarrierWithGroupSync();

    uint source = (id.x + 1u) % 32u;
    uint dst = id.x * 200u;
    // Raw reads independently check aggregate stores; a matching bug in typed
    // reads and writes cannot cancel out and pass this half of the comparison.
    for (uint word = 0; word < 24; ++word) {
        uint value = resolveDescriptor(gProbe.raw).Load(source * 96u + word * 4u);
        resolveDescriptor(gProbe.output).Store(dst + word * 4u, value);
    }
    NestedRecord read = resolveDescriptor(gProbe.readRecords)[source];
    resolveDescriptor(gProbe.output).Store4(dst + 96u, read.prefix);
    resolveDescriptor(gProbe.output).Store4(dst + 112u, read.children[0].first);
    resolveDescriptor(gProbe.output).Store4(dst + 128u, read.children[0].second);
    resolveDescriptor(gProbe.output).Store4(dst + 144u, read.children[1].first);
    resolveDescriptor(gProbe.output).Store4(dst + 160u, read.children[1].second);
    resolveDescriptor(gProbe.output).Store4(dst + 176u, read.suffix);
    uint count, stride;
    resolveDescriptor(gProbe.readRecords).GetDimensions(count, stride);
    resolveDescriptor(gProbe.output).Store2(dst + 192u, uint2(count, stride));
}

