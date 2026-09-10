# Ray-query shading vertices

`ScenePathTraceResources` supplies the same compact vertex ABI to the path tracer
(including OpenPBR, guides and secondary rays in visibility-buffer shading),
material visualization and RTXDI. Raster geometry keeps its existing ABI.

| Data | Encoding | Bytes per vertex |
| --- | --- | ---: |
| Shading normal | Octahedral snorm16 × 2 | 4 |
| Tangent | Octahedral snorm15 × 2, handedness and validity bits | 4 |
| UV0 | float32 × 2 | 8 |
| Positions on fallback devices | Separate three-scalar record | 12 |

The former interleaved record was 64 bytes. The shading stream is now 16 bytes
(75% smaller). Devices without position fetch additionally retain a 12-byte
position stream: 28 bytes total (56.25% smaller). These figures describe vertex
buffers, not total VRAM, cache traffic or frame-time improvements. Encoding adds
shader arithmetic; performance still needs scene-specific measurement.

CPU packing is in `Source/Runtime/Render/SceneShadingVertex.h`; the matching shader
ABI and decoders are in `Shaders/Libraries/Scene/SceneShadingVertex.slang`.
Zero/missing normals preserve geometric-normal fallback. Tangent sign remains
explicit, and UVs retain float32 precision for large, tiled and negative values.
Directions are normalized by the octahedral representation.

On position-fetch devices, no shading position buffer is generated or uploaded.
Triangle positions come from a BLAS built with `AllowDataAccess`, including after
compaction. Build-only BLAS vertex/index buffers are released after GPU completion;
instance/scratch buffers remain for TLAS refits. On fallback devices, binding 54
contains positions. Three scalar fields enforce a 12-byte stride even under
std430; a `StructuredBuffer<float3>` would use a 16-byte stride there.

Committed-hit world position uses `ray.origin + ray.direction * CommittedRayT()`.
Triangle positions still determine geometric normals and UV-derived tangent
frames. Authored tangent interpolation and fallback tangent reconstruction run
only in their respective branches. Normal orientation stays unchanged until
normal-map evaluation and the existing final shading-normal face-forward step.

Regression coverage includes CPU-to-GPU packing over 519 directions, poles and
zero vectors, both handedness signs and exact float32 UV readback; indexed
triangles under nonuniform scale, front/back hits, misses, BLAS compaction and
TLAS refits with position fetch both enabled and disabled. Shader compile tests
cover both feature variants for Standard/OpenPBR guides, material visualization
and RTXDI.
