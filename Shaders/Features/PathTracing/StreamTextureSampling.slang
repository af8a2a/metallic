#ifndef METALLIC_STREAM_TEXTURE_SAMPLING
#define METALLIC_STREAM_TEXTURE_SAMPLING

float streamTextureSamplingFloor(uint textureIndex)
{
#if SHARC_UPDATE
    RWStructuredBuffer<uint> metadata = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().textureFeedback);
#else
    RWStructuredBuffer<uint> metadata = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureFeedbackOffset));
#endif
    uint count, stride;
    metadata.GetDimensions(count, stride);
    uint offset = textureIndex * 8u;
    return offset + 6u < count ? max(asfloat(metadata[offset + 6u]), 0.0) : 0.0;
}

// Diagnostic and deterministic coverage paths request the finest source mip;
// publication and memory budgets still control which mip is available this frame.
void requestStreamTextureMip0(uint textureIndex)
{
#if SHARC_UPDATE
    RWStructuredBuffer<uint> feedback = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().textureFeedback);
#else
    RWStructuredBuffer<uint> feedback = resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneTextureFeedbackOffset));
#endif
    uint count, stride;
    feedback.GetDimensions(count, stride);
    uint offset = textureIndex * 8u;
    if (offset + 7u < count && feedback[offset] != 0u) {
        uint ignored;
        InterlockedMin(feedback[offset + 4u], 0u, ignored);
        InterlockedAdd(feedback[offset + 5u], 1u, ignored);
    }
}

#endif
