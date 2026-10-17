#language slang 2026
implementing Core;

#ifndef METALLIC_TRUSTED_SCENE_RESOURCE_BYTES
#define METALLIC_TRUSTED_SCENE_RESOURCE_BYTES 0
#endif
#ifndef METALLIC_INLINE_COMPUTE_ROOT
#define METALLIC_INLINE_COMPUTE_ROOT 0
#endif

#if METALLIC_TRUSTED_SCENE_RESOURCE_BYTES && !METALLIC_INLINE_COMPUTE_ROOT
#error Trusted scene resources require the inline compute root ABI
#endif

namespace Metallic
{
// Only the host-validated SceneResourceParameters packet can use this contract.
// Other programs retain the checked getter, including its short-packet behavior.
[ForceInline]
public SceneResourceParameters getSceneResourceParameters()
{
#if METALLIC_TRUSTED_SCENE_RESOURCE_BYTES
    static_assert(sizeof(SceneResourceParameters) == METALLIC_TRUSTED_SCENE_RESOURCE_BYTES,
        "Trusted scene packet size must match the host ABI");
    let span = gInlineComputeRoot.resources;
    return resolveUniform(span.resource).Load<SceneResourceParameters>(span.byteOffset);
#else
    return getResourceParameters<SceneResourceParameters>();
#endif
}

// Keep field legality independent of the host's complete-packet guarantee.
public TField getSceneResourceParameterField<TField>(uint byteOffset)
{
#if METALLIC_TRUSTED_SCENE_RESOURCE_BYTES
    static_assert(sizeof(SceneResourceParameters) == METALLIC_TRUSTED_SCENE_RESOURCE_BYTES,
        "Trusted scene packet size must match the host ABI");
    TField zero = {};
    if (sizeof(TField) > sizeof(SceneResourceParameters) ||
        byteOffset > sizeof(SceneResourceParameters) - sizeof(TField) ||
        (byteOffset & 3u) != 0u) { return zero; }
    let span = gInlineComputeRoot.resources;
    return resolveUniform(span.resource).Load<TField>(span.byteOffset + byteOffset);
#else
    return getResourceParameterField<SceneResourceParameters, TField>(byteOffset);
#endif
}
} // namespace Metallic
