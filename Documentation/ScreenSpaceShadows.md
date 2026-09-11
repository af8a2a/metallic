# Full ray-traced shadows and NRD SIGMA

The realtime sample and standalone GPUDriven default share
`Pipelines/Samples/realtime_lighting.metallic_graph.json`:

`VBuffer.depth/rasterInfo → Shadows (RayTracedShadowPass: TLAS rays + SIGMA) → Deferred → DLSS-SR`

Every shadow sample queries the scene acceleration structure. There is no depth
march, screen-space occlusion test, viewport clipping, or geometry fallback branch.
Hardware depth supplies the receiver position, depth-derived normal and SIGMA
view/motion guides; primary visibility remains the existing deferred VBuffer path.
Occluders outside the viewport or hidden behind the camera can cast shadows.

## Reference and tracing

The implementation follows the sun-shadow section of the user-provided
`E:/NRD-Sample/Shaders/TraceOpaque.cs.hlsl` (lines 858–896): sample the light disk,
trace scene geometry, and pass the occluder distance and tangent of the angular
radius to `SIGMA_FrontEnd_PackPenumbra`. `NRDSample.cpp` sets SIGMA's light direction
and computes `tan(radians(sunAngularDiameter * 0.5))`. Metallic's control is already
an angular **radius**, so it is converted to radians without another factor of 0.5.

One random light-disk sample is generated per valid receiver per frame. Metallic
uses a pixel/frame hash rather than the sample's scrambling/ranking texture set;
there is no longer an eight-frame repeating sequence. Directional rays sample the
angular disk. Point and spot rays end at a sampled emitter position, accounting
for both source radius and origin offset, and cannot intersect past that endpoint.

The ray query resolves the closest committed triangle distance for SIGMA, rather
than accepting the first traversal hit. Alpha-mask and conventional/NTC texture
sampling reuse the scene shadow helpers. A miss uses NRD's FP16-max sentinel.
The current integration uses `SIGMA_SHADOW`; the sample's optional translucent
shadow variant is not enabled. Colored transmission is outside this shadow signal.

The pass reuses Deferred's prepared `SceneResourceManager` snapshot, including
RTAS, geometry, materials and textures. It requires acceleration-structure and
RayQuery device capabilities; unsupported devices receive an explicit error.
Authored hit normals and normal-map tangent frames are not modified.

## Pipeline and controls

`Shadows.shadow` is sqrt-encoded R8 visibility. `Shadows.parameters` identifies the
selected stable light slot. Deferred squares visibility once and applies it to
that light's direct contribution. It does not clamp the result with another hard
shadow. Both material-binned and ordinary deferred dispatches consume this signal.
Other lights, environment illumination and emission keep their existing evaluation.

Automatic light selection prefers the first active directional light, then the
first active local light. Disabled slots remain in the source table, matching
GPUScene/LightGrid. Invalid or disabled explicit selections fall back to Auto.

| Property | Default | Meaning |
| --- | --- | --- |
| `rayTracedShadows` | `true` | Apply ray-traced/SIGMA visibility for the selected light |
| `sigmaDenoise` | `true` | Filter shadow samples with SIGMA |
| `shadowDebug` | `false` | Show unpacked shadow visibility |
| `shadowLightIndex` | `-1` | Auto or a stable scene light slot |
| `shadowRayLength` | `100000` | Maximum TLAS ray length in metres; local rays also stop at the emitter |
| `shadowBias` | `0.01` | Receiver normal offset in metres |
| `shadowAngularRadius` | `0.266` | Directional source angular radius in degrees |
| `shadowLightRadius` | `0.05` | Point/spot emitter radius in metres |
| `sigmaHistoryLength` | `5` | Stabilization history, 0–7; zero still uses spatial filtering |

Zero source radius produces hard shadows. Nonzero radius widens the penumbra with
blocker distance. Without SIGMA (including `METALLIC_ENABLE_NRD=OFF`), the raw
stochastic shadow sample is displayed. `Deferred.debugDisableShadows` bypasses all
direct shadows. Disabling `rayTracedShadows` restores Deferred's existing direct
visibility evaluation for the selected light.

## Saved graphs

`ScreenSpaceShadowPass` remains a registration alias for `RayTracedShadowPass`.
The original C++/shader/document file names are retained to avoid disrupting
existing callers and worktree changes. They contain the full ray-traced path.
The old `screenSpaceShadows` enable key is read only when `rayTracedShadows` is
absent. Old `shadowSteps`, `shadowDistance`, `shadowThickness` and
`preserveGeometryShadows` values are ignored and have no UI controls.

Existing realtime Deferred graphs without connected shadow inputs use the same
full ray-traced implementation internally. Connect both `shadow` and
`shadowParameters` together; connecting only one is an execution error.

## SIGMA data and lifetime

SIGMA uses the vendored NRD 4.16 kernels and Metallic's bindless scheduler:
classification, tile smoothing, history copy, blur, post-blur and stabilization.
Inputs are R16F penumbra, packed world normal, R32F linear depth, and previous-minus-
current UV motion. The trace preserves the R8 output until SIGMA copies history.

The pass serializes frames and publishes a graph-owned snapshot while retaining
private SIGMA history. Submitted frames retain geometry and shadow resources.
Resize, camera cuts, scene/transform/light edits, settings changes and discarded
recordings invalidate history. Camera motion reprojects; moving-object transforms
restart history because object motion is not reconstructed from depth.

## Validation

```powershell
cmake --build build --target Metallic MetallicGPUDrivenSample MetallicNrdTests MetallicRhiTests --parallel 6
ctest --test-dir build -R '^MetallicNrdTests$' --output-on-failure
build/tests/MetallicRhiTests.exe --filter realtime_ray_traced_sigma_shadows --rhi-validation --output-dir .tmp/full-ray-traced-shadows
build/tests/MetallicRhiTests.exe --filter render_graph_sample_load
```

The NRD geometry test explicitly enables RayQuery. A TLAS quad behind the camera
casts onto a depth-only receiver plane. A fabricated depth-buffer occluder must
not create a shadow. Tests also exercise alpha-cutout geometry, local lights,
ray-length clipping, perspective/orthographic and reversed/conventional depth,
noise reduction, history changes, disabled/no-light behavior, resize and discard.

The realtime test checks Auto/explicit/invalid light selection, ineffective legacy
screen-space settings, effective ray length, both deferred resolve paths and
0/1/8 degree penumbra growth. It verifies brighter pixels inside the previous
hard-shadow edge in final lighting. Captures include `ShadowAngle0.png`,
`ShadowAngle1.png`, `ShadowAngle8.png` and `ShadowAngle8Lighting.png`.
