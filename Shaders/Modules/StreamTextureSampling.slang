#language slang 2026
module StreamTextureSampling;

namespace Metallic
{

// The host publishes one immutable entry per bound texture slot for this frame.
// Width/height describe the actual resident texture, including NPOT mip tails.
public struct TextureSamplingMetadata
{
    public float samplingFloor;
    // Exact POT dimension LOD, or negative resident pixel area for GPU log2.
    public float residentLodTerm;
    public uint residentWidth;
    public uint residentHeight;
    public uint residentMipCount;
    public uint residentFirstMipAndFlags;
};

public bool streamTextureDemandEnabled(TextureSamplingMetadata metadata)
{
    return (metadata.residentFirstMipAndFlags & 0x80000000u) != 0u;
}

// The caller validates the record and all participating lanes use the same buffer.
// The wave permutation is restricted to compute entries with maximal reconvergence.
public void recordTextureDemand(RWStructuredBuffer<uint> demandMip, RWStructuredBuffer<uint> demandHits,
    uint textureIndex, uint wantedMip)
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
            InterlockedMin(demandMip[key], mip, ignored);
            InterlockedAdd(demandHits[key], hits, ignored);
        }
        pending = pending && !selected;
    }
#else
    uint ignored;
    InterlockedMin(demandMip[textureIndex], wantedMip, ignored);
    InterlockedAdd(demandHits[textureIndex], 1u, ignored);
#endif
}

// Trusted scene entry: callers retain the material texture-index check, and
// host validation ties metadata/demand counts to the same texture snapshot.
public float streamTextureSamplingFloor(StructuredBuffer<TextureSamplingMetadata> metadata, uint textureIndex)
{
    return max(metadata[textureIndex].samplingFloor, 0.0);
}

// Diagnostic and deterministic coverage paths request the finest source mip;
// publication and memory budgets still control which mip is available this frame.
public void requestStreamTextureMip0(StructuredBuffer<TextureSamplingMetadata> metadata,
    RWStructuredBuffer<uint> demandMip, RWStructuredBuffer<uint> demandHits, uint textureIndex)
{
    if (streamTextureDemandEnabled(metadata[textureIndex])) {
        recordTextureDemand(demandMip, demandHits, textureIndex, 0u);
    }
}
} // namespace Metallic
