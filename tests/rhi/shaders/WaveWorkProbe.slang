#include "../../../Shaders/Features/GPUDriven/WaveWorkDistribution.slang"

struct ProbePush
{
    uint inputBuffer;
    uint outputBuffer;
};

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    DescriptorHandle<StructuredBuffer<uint>> input =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.inputBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint4>> output =
        DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.outputBuffer, 0));
    uint count = input[id.x];
    WaveWorkDistribution work = waveWorkBegin(count);
    output[id.x * 65u] = uint4(WaveGetLaneIndex(), WaveReadLaneFirst(id.x), count, work.total);
    for (uint batch = 0u; batch < work.total; batch += WaveGetLaneCount()) {
        uint localItem;
        uint source = waveWorkNext(work, batch, localItem);
        uint owner = WaveReadLaneAt(id.x, source);
        uint item = batch + WaveGetLaneIndex();
        if (item < work.total && localItem < 64u) {
            output[owner * 65u + 1u + localItem] = uint4(item, owner, localItem, source);
        }
    }
}
