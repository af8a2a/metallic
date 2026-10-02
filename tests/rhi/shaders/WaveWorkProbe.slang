#include "../WaveWorkProbeParameters.h"
#include "../../../Shaders/Features/GPUDriven/WaveWorkDistribution.slang"

[[vk::push_constant]] ConstantBuffer<WaveWorkProbeParameters> gWaveWork;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    WaveWorkProbeParameters p = gWaveWork;
    bool valid = id.x < min(p.input.length(), p.output.length() / 65u);
    // Invalid/tail lanes still participate in every wave operation.
    uint count = valid ? p.input.data[id.x] : 0u;
    WaveWorkDistribution work = waveWorkBegin(count);
    if (valid) p.output.data[id.x * 65u] = uint4(WaveGetLaneIndex(), WaveReadLaneFirst(id.x), count, work.total);
    for (uint batch = 0u; batch < work.total; batch += WaveGetLaneCount()) {
        uint localItem;
        uint source = waveWorkNext(work, batch, localItem);
        uint owner = WaveReadLaneAt(id.x, source);
        uint item = batch + WaveGetLaneIndex();
        if (item < work.total && localItem < 64u) {
            p.output.data[owner * 65u + 1u + localItem] = uint4(item, owner, localItem, source);
        }
    }
}
