#language slang 2026
module StreamTextureSampling;

namespace Metallic
{

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
        uint ignored;
        InterlockedMin(feedback[offset + 4u], 0u, ignored);
        InterlockedAdd(feedback[offset + 5u], 1u, ignored);
    }
}
} // namespace Metallic
