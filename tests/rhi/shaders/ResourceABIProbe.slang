import ShaderCore;
using Metallic;

struct Record
{
    uint first;
    uint values[3];
};

struct ResourceABIProbeParams
{
    BufferSpan<uint> sources[2];
    BufferSpan<Record> records;
    RWBufferSpan<uint> output;
};

[[vk::push_constant]] ConstantBuffer<ResourceABIProbeParams> gParams;

[shader("compute")]
[numthreads(32, 1, 1)]
void resourceABIProbeMain(uint lane : SV_GroupIndex)
{
    if (lane < 8) {
        // Different descriptors in the same wave, with independent byte offsets.
        gParams.output.store(lane, gParams.sources[lane & 1].loadNonUniform(lane / 2));
    }
    if (lane == 8) {
        gParams.output.store(8, gParams.sources[0].load(gParams.sources[0].count));
        Record record = gParams.records.load(0);
        gParams.output.store(9, record.first + record.values[2]);
        gParams.output.store(gParams.output.count, 0xffffffffu);
    }
}
