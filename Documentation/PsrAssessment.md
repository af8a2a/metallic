# PSR implementation assessment

Date: 2026-09-10. Scope: Metallic's current DLSS-RR/OpenPBR path and the authored
ABeautifulGame scene. Reference: `E:/vk_denoise_dlssrr`, commit
`f8113a84fa952278bb1286e7899f81b7afe8f41e`.

## Recommendation

Defer PSR as a response to the current chess-scene motion noise. A narrowly scoped
mirror-only prototype is worthwhile when near-perfect metallic mirrors become a
target use case. Broad support for rough reflections and glass is a substantially
larger project and should not be justified by the simple reference implementation.

## Evidence from the current scene

The reference continues through a surface only when metallic is 1 and its squared
roughness is at most `MICROFACET_MIN_ROUGHNESS² + 0.001`. The dependency defines
`MICROFACET_MIN_ROUGHNESS = 0.0014142`, giving a perceptual roughness cutoff of
approximately **0.03165**. The reference caps its traversal loop at five hits.

All 15 scene materials use metallic/roughness textures. I decoded their source
ORM JPEGs, read G as roughness and B as metalness, and applied the glTF factors.
No sRGB conversion was applied to these data channels, matching the shader.

| Measurement | Result |
| --- | --- |
| Source texels satisfying the reference's mirror condition | 0 in every material |
| Chessboard roughness: minimum / median / maximum | 0.0549 / 0.2706 / 0.4784 |
| Chessboard maximum metallic value | 0.9647 |
| Relaxed condition: roughness <= 0.05, metallic >= 0.98 | 0 in 13 materials; approximately 0.000119% in each bishop material |
| Materials with authored transmission weight 1 | Pawn_Top_White and Pawn_Top_Black |

This is a source-texture audit, not a GPU screen-coverage measurement. It includes
unused UV texels and does not capture unsaved runtime material overrides, texture
filtering, or runtime texture compression. It nevertheless provides strong
evidence that a faithful port would rarely, if ever, activate on the current
authored scene. In particular, the board's roughness is above the mirror cutoff
even at its minimum.

The reproducible local audit and full per-material results are in
`.cache/psr-assessment/audit_materials.py` and
`.cache/psr-assessment/material-texture-stats.json` (local, untracked output).

## Expected benefit by surface type

| Target | Expected benefit of reference-style PSR | Reason |
| --- | --- | --- |
| Near-perfect metallic mirrors, including repeated mirror reflections | High potential | The denoiser can follow reflected geometry, its detail and virtual motion rather than the mirror plane |
| Current rough chessboard reflections | Low | They fail the mirror gate; relaxing the gate creates a mismatch between a single virtual surface and a spread of reflected directions |
| Glass pawn tops | Not covered by this implementation | Reflection and transmission coexist; the reference requires a fully metallic surface |
| Ordinary diffuse or indirect-light noise | Little direct benefit | PSR changes which surface guides reconstruction; it does not generally reduce Monte Carlo lighting variance |

NVIDIA's [PSR explanation](https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/)
describes the limitations of nonzero roughness and the additional reflection/
refraction treatment for glass. The [reference README](https://github.com/nvpro-samples/vk_denoise_dlssrr/blob/main/README.md#mirror-like-surfaces)
documents using the reflected surface as the denoising guide.

## Cost in Metallic

The change belongs in path tracing and guide generation, rather than in a DLSS
option. `OpenPBRRayQueryPathTraceGuides.slang` currently traces radiance and then
evaluates guides with a separate first-hit query. A correct implementation must
keep the radiance path and the replacement guide surface consistent:

- Resolve a mirror prefix and retain throughput, emission, path length and the
  composed mirror transform.
- Generate virtual normal, depth, motion, albedo and specular hit distance for the
  replacement surface together. Changing only depth or motion is insufficient.
- Reuse the same prefix for the first radiance sample and its guides; independently
  sampling the two paths can make their hit surfaces disagree.
- Preserve authored geometry and shading normals through TBN/normal mapping;
  keep mirror transforms and face-forward shading normals separate.
- Define fallback at material boundaries, maximum path length, sky exits and
  camera cuts; test mirror-to-mirror paths and threshold stability under motion.
- Keep actual surface depth available to consumers that expect camera-visible
  geometry, especially if guides are shared with DLSS-SR or debug views.

A prototype restricted to opaque, near-perfect metallic mirrors is a moderate
shader integration task. General glass/refraction, curved-mirror reprojection,
rough lobes and moving geometry need substantially more validation and design.

Runtime cost is not necessarily one entirely new reflection path: the current
path tracer already follows reflected rays. Reusing that work could keep the
increment small. Adding an independent mirror chain to the existing guide query
would add intersections and material evaluations, with divergence and register
pressure. No GPU timing or image-quality improvement is claimed without a PSR
implementation and A/B measurement.

## Better next measurement

First measure the corrected F-preset path on a repeatable chess-camera trajectory
and inspect reflected hit distance, reflected geometry motion, and sky guides.
Metallic already supplies specular hit distance plus camera matrices, so PSR is
an additional quality technique, not a missing mandatory RR input. Optional
specular motion vectors are another targeted avenue for reflection motion; their
correctness and cost also require investigation.

If mirror content is added, build a switchable prototype limited initially to
one or two opaque mirror interactions. Compare identical camera paths, resolution,
samples, exposure and RR preset. Measure reflected-edge stability, temporal error
against a converged reference, disocclusion recovery and GPU time. Keep the
feature only if it improves those mirror regions without introducing energy,
boundary or material-appearance errors.
