#language slang 2026
module TextureFeedbackWaveProbe;

import ShaderCore;
import StreamTextureSampling;
using Metallic;

struct TextureFeedbackWaveParams
{
    BufferSpan<uint4> inputs;
    RWBufferSpan<uint> demandMip;
    RWBufferSpan<uint> demandHits;
    uint laneCount;
    uint iterations;
};
[[vk::push_constant]] ConstantBuffer<TextureFeedbackWaveParams> gProbe;

void runFeedbackProbe(uint index, bool aggregate)
{
    if (index >= gProbe.laneCount) { return; }
    uint4 input = gProbe.inputs.load(index);
    if (input.z == 0u) { return; }
    RWStructuredBuffer<uint> demandMip = resolveUniform(
        ResourceHandle<RWStructuredBuffer<uint>>(gProbe.demandMip.resource.index));
    RWStructuredBuffer<uint> demandHits = resolveUniform(
        ResourceHandle<RWStructuredBuffer<uint>>(gProbe.demandHits.resource.index));
    for (uint iteration = 0u; iteration < gProbe.iterations; ++iteration) {
        uint wantedMip = (input.y + iteration) & 15u;
        if (aggregate) {
            recordTextureDemand(demandMip, demandHits, input.x, wantedMip);
        } else {
            // Frozen pre-aggregation behavior, independent of the production helper.
            uint ignored;
            InterlockedMin(demandMip[input.x], wantedMip, ignored);
            InterlockedAdd(demandHits[input.x], 1u, ignored);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void textureFeedbackWaveReference(uint3 id : SV_DispatchThreadID)
{
    runFeedbackProbe(id.x, false);
}

[shader("compute")]
[numthreads(64, 1, 1)]
#if METALLIC_TEXTURE_FEEDBACK_WAVE
[MaximallyReconverges]
#endif
void textureFeedbackWaveProduction(uint3 id : SV_DispatchThreadID)
{
    runFeedbackProbe(id.x, true);
}
