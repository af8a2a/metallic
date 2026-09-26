import Core;
using Metallic;

struct NestedChild { uint4 first; uint4 second; };
struct NestedRecord { uint4 prefix; NestedChild children[2]; uint4 suffix; };

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
    getResource<RWStructuredBuffer<NestedRecord>>(0)[id.x] = record;
    DeviceMemoryBarrierWithGroupSync();

    uint source = (id.x + 1u) % 32u;
    uint dst = id.x * 200u;
    // Raw reads independently check aggregate stores; a matching bug in typed
    // reads and writes cannot cancel out and pass this half of the comparison.
    for (uint word = 0; word < 24; ++word) {
        uint value = getResource<ByteAddressBuffer>(0).Load(source * 96u + word * 4u);
        getResource<RWByteAddressBuffer>(1).Store(dst + word * 4u, value);
    }
    NestedRecord read = getResource<StructuredBuffer<NestedRecord>>(0)[source];
    getResource<RWByteAddressBuffer>(1).Store4(dst + 96u, read.prefix);
    getResource<RWByteAddressBuffer>(1).Store4(dst + 112u, read.children[0].first);
    getResource<RWByteAddressBuffer>(1).Store4(dst + 128u, read.children[0].second);
    getResource<RWByteAddressBuffer>(1).Store4(dst + 144u, read.children[1].first);
    getResource<RWByteAddressBuffer>(1).Store4(dst + 160u, read.children[1].second);
    getResource<RWByteAddressBuffer>(1).Store4(dst + 176u, read.suffix);
    uint count, stride;
    getResource<StructuredBuffer<NestedRecord>>(0).GetDimensions(count, stride);
    getResource<RWByteAddressBuffer>(1).Store2(dst + 192u, uint2(count, stride));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void unsafeNativeAsMain()
{
    RayDesc ray = {float3(0, 0, 1), 0.001, float3(0, 0, -1), 100.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    // Deliberately bypass resolveDescriptor: compilation must reject this ABI.
    query.TraceRayInline(*DescriptorHandle<RaytracingAccelerationStructure>(gComputeResources.resources[0]), 0, 255, ray);
    while (query.Proceed()) {}
    getResource<RWByteAddressBuffer>(1).Store(0, query.CommittedStatus());
}
