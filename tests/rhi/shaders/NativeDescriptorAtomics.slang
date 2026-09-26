struct Child { uint4 first; uint4 second; };
struct Record { uint4 prefix; Child children[2]; uint4 suffix; };
struct Push { uint atomics; uint counters; uint records; uint count; uint base; };

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Push push)
{
    if (id.x >= push.count) { return; }
    RWStructuredBuffer<uint64_t> atomics = DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(push.atomics, 0));
    RWStructuredBuffer<uint> counters = DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.counters, 0));
    StructuredBuffer<Record> records = DescriptorHandle<StructuredBuffer<Record>>(uint2(push.records, 0));
    ByteAddressBuffer raw = DescriptorHandle<ByteAddressBuffer>(uint2(push.records, 0));
    uint ticket;
    InterlockedAdd(counters[0], 1u, ticket);
    uint64_t old;
    InterlockedAdd(atomics[push.base], uint64_t(0x100000001), old);
    uint64_t packed = (uint64_t(id.x + 1) << 32) | uint64_t(0xf00d);
    InterlockedMax(atomics[push.base + 1], packed);
    InterlockedMin(atomics[push.base + 2], packed);
    uint64_t previous;
    InterlockedCompareExchange(atomics[push.base + 3], uint64_t(0), uint64_t(0xfeed00000000beef), previous);
    atomics[push.base + 8 + id.x] = old;
    atomics[push.base + 8 + push.count + id.x] = ticket;
    atomics[push.base + 8 + 2 * push.count + id.x] = previous;
    uint source = (id.x * 7 + 3) % push.count;
    Record record = records[source];
    uint count, stride;
    atomics.GetDimensions(count, stride);
    uint error = count != 1024 || stride != 8;
    uint words[24];
    for (uint i = 0; i < 4; ++i) {
        words[i] = record.prefix[i];
        words[4 + i] = record.children[0].first[i];
        words[8 + i] = record.children[0].second[i];
        words[12 + i] = record.children[1].first[i];
        words[16 + i] = record.children[1].second[i];
        words[20 + i] = record.suffix[i];
    }
    for (uint i = 0; i < 24; ++i) {
        uint expected = source * 1000 + i + 1;
        error |= words[i] != expected || raw.Load(source * 96 + i * 4) != expected;
    }
    if (error != 0) { InterlockedAdd(counters[1], 1u); }
}
