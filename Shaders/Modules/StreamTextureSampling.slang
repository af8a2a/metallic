#language slang 2026
module StreamTextureSampling;

namespace Metallic
{

// The caller validates the record and all participating lanes use the same buffer.
// The wave permutation is restricted to compute entries with maximal reconvergence.
public void recordTextureDemand(RWStructuredBuffer<uint> feedback, uint textureIndex, uint wantedMip)
{
#if METALLIC_TEXTURE_FEEDBACK_WAVE
    bool pending = true;
    for (;;) {
        uint key = WaveActiveMin(pending ? textureIndex : 0xffffffffu);
        if (key == 0xffffffffu) { break; }
        bool selected = pending && textureIndex == key;
        uint mip = WaveActiveMin(selected ? wantedMip : 0xffffffffu);
        uint hits = WaveActiveSum(selected ? 1u : 0u);
        if (WaveIsFirstLane()) {
            uint ignored;
            InterlockedMin(feedback[key * 8u + 4u], mip, ignored);
            InterlockedAdd(feedback[key * 8u + 5u], hits, ignored);
        }
        pending = pending && !selected;
    }
#else
    uint ignored;
    InterlockedMin(feedback[textureIndex * 8u + 4u], wantedMip, ignored);
    InterlockedAdd(feedback[textureIndex * 8u + 5u], 1u, ignored);
#endif
}

public float streamTextureSamplingFloor(RWStructuredBuffer<uint> metadata, uint textureIndex)
{
    uint count, stride;
    metadata.GetDimensions(count, stride);
    uint offset = textureIndex * 8u;
    return offset + 6u < count ? max(asfloat(metadata[offset + 6u]), 0.0) : 0.0;
}

// Diagnostic and deterministic coverage paths request the finest source mip;
// publication and memory budgets still control which mip is available this frame.
public void requestStreamTextureMip0(RWStructuredBuffer<uint> feedback, uint textureIndex)
{
    uint count, stride;
    feedback.GetDimensions(count, stride);
    uint offset = textureIndex * 8u;
    if (offset + 7u < count && feedback[offset] != 0u) {
        recordTextureDemand(feedback, textureIndex, 0u);
    }
}
} // namespace Metallic
