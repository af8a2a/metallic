# DLSS Ray Reconstruction temporal inputs

The comparison used the local NVIDIA sample at `E:/vk_denoise_dlssrr`, commit
`f8113a84fa952278bb1286e7899f81b7afe8f41e`.

## Camera movement

The sample keeps the previous view/projection matrices and evaluates RR with
`reset = (m_frame == 0)`. Ordinary camera movement does not restart the frame
counter (`dlss_rr/src/dlssrr_sample.cpp`).

Metallic's viewport camera update previously incremented the companion DLSS
pass's `resetSerial` on every movement. That made the RR pass discard both its
previous camera and reconstruction history each moving frame. Camera updates now
only synchronize the DLSS camera. The Reset control, mode changes, output/render
size changes, and graph preparation still reset DLSS. The editor still invalidates
ordinary pixel accumulation, which does not reproject between camera positions.

## Motion vectors and jitter

Both implementations shoot primary rays through `pixel + 0.5 + jitter` and pass
`-jitter` to NGX. The NVIDIA sample subtracts this same jittered sample position
from the previous-frame projection (`dlss_rr/shaders/primary_rgen.slang`).

Metallic previously subtracted `pixel + 0.5`, introducing a false
`jitter / renderExtent` motion even for a stationary camera. The shared motion
helper now includes the current jitter when computing the sample position. This
matches `motionVectorsJittered = false`: sampling is jittered, but the returned
motion describes camera/object displacement without a jitter displacement.

The sample supplies motion in pixels directly to NGX. Metallic supplies
`previousUv - currentUv` to Streamline with `mvecScale = {1, 1}`; Streamline
multiplies by the render extent before calling NGX. These conventions are
equivalent; copying the sample's pixel values without changing Streamline's
scale would multiply motion by the render extent twice.

The reference sky motion uses a direction at infinity (`w = 0`). Metallic's sky
was projected as a point at the camera far plane, causing false sky parallax
under translation. Sky projection now excludes camera translation and retains
perspective camera rotation. This change applies to both Standard and OpenPBR
guide shaders.

## Remaining differences

The reference enables Primary Surface Replacement (PSR) and path regularization
by default. PSR follows near-perfect metallic mirrors and supplies guide data for
the reflected virtual surface. Regularization propagates maximum roughness along
the path to reduce indirect specular variance. Metallic currently supplies the
first surface's guides and does not implement this PSR path. These are separate
rendering changes, especially for mirror reflections; this temporal-input fix
does not add them or change the BSDF/normal-map basis.

The reference also provides a tone-mapped environment color as the sky diffuse
albedo guide and zero sky normals. Metallic currently uses its default sky guide
values. This remains a possible follow-up for environment detail preservation.

Metallic uses Streamline 2.14.1 with RR Preset F. The inspected reference exposes
Default/D/E presets, so its preset menu should not be copied as a version update.

## Regression checks

- `RhiRendering.dlss_motion_vector_reprojection` runs the actual shader helpers on
  the GPU: 16 jitter offsets, perspective/orthographic projection, stationary and
  translated cameras, perspective rotation, sky motion, and invalid history.
- `MetallicEditorDlssCameraSmoke` renders 16 moving RR frames through the editor's
  viewport camera update path and checks camera synchronization, preserved reset
  counters, explicit Reset, and continued invalidation of ordinary accumulation.
  It requires a GPU/driver supporting DLSS-RR.
- `RhiRendering.render_graph_pathtracing_guides_shader_compile` compiles both
  guide shaders.

These checks validate integration and reprojection inputs; they do not measure
the visual noise reduction of the original animated chess scene.
