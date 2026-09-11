# Metallic NRD integration

This is the NRD **4.16** shader snapshot from NVIDIA NRD commit
`36183520b006b65860bebc96d3dd7185e5edeb45`. The NVIDIA license is preserved in
[LICENSE.txt](LICENSE.txt), together with the original copyright notices.

Metallic supports REBLUR diffuse + specular radiance, RELAX diffuse + specular
radiance, SIGMA shadows, and two independent REFERENCE accumulators. The frontend packing stays
at normal encoding 2 and linear roughness encoding 1, matching Metallic's existing
RTXDI shaders. The shared math routines come from `External/MathLib`.

## Ownership and dispatch

- `Source/Runtime/Render/Denoising/NrdReblur.cpp`, `NrdRelax.cpp`, and
  `NrdReference.cpp` and `NrdSigma.cpp` contain the adapted, readable pass recipes and shader
  constant calculations. Their upstream CPU implementations are reference
  material; no NRD SDK API, library, instance, or embedded shader is loaded.
- `NrdPlan.cpp` owns internal texture descriptions, transient reuse, history
  ping-pong, camera transforms, pass selection, and dispatch dimensions.
- `Source/Runtime/Render/RenderGraph/NrdRuntime.cpp` allocates textures through
  the RHI, transitions them between passes, uploads constants with `Streamer`,
  and calls `CommandBuffer::dispatch` directly.
- Each distinct texture view/access receives a Metallic descriptor handle once
  per frame. A dispatch uploads an index table containing the actual
  `BindlessHandle::shaderIndex` values. Descriptors are shared between passes;
  changing REFERENCE's input signal never overwrites an earlier dispatch's
  descriptors.
- `*.bindings.hlsli` dereference Slang `DescriptorHandle<Texture2D<...>>`,
  `DescriptorHandle<RWTexture2D<...>>`, and sampler handles directly. The numeric
  slots in `*.resources.hlsli` describe algorithm inputs, not HLSL registers.
  There are no per-pass `ShaderBindingMappingDesc` records. Slang's runtime heap
  arrays use the same fixed heap mappings as Metallic's other native bindless
  shaders (samplers at binding 0, resources at binding 2).
- The RHI prepends its two-word heap header to push data. User data contains two
  device addresses: constants and the resource index table. Shader constants
  explicitly use column-major matrices to match their CPU representation.
  Shader-side addresses are pointers, so this path does not require shader Int64.

`NrdDenoisePass` serializes its temporal instance across frames. Graph mode changes,
reset, resize, and discarded recordings invalidate history. REFERENCE diffuse and
specular have separate frame counters. `timeDeltaSeconds` is converted to the
milliseconds expected by the denoising equations. REBLUR additionally requires the
RHI's `shaderImageGatherExtended` capability.

## Building and validating

`METALLIC_ENABLE_NRD=ON` enables this integration. CMake does not configure the NRD
submodule, download DXC, build ShaderMake, or compile every shader permutation.
Only the stages first used at runtime are compiled through Metallic's normal
Slang disk cache. Consequently a cold first use still incurs shader compilation;
subsequent launches reuse the cache. Building the engine does not require an
initialized `External/nrd` checkout.

With tests enabled:

```powershell
cmake --build build-full --target MetallicNrdTests --parallel 6
ctest --test-dir build-full -R '^MetallicNrdTests$' --output-on-failure
```

The focused suite compiles every supported permutation, checks that descriptor
bindings are runtime heap arrays, checks CPU schedules, and runs all four modes
with Vulkan validation. GPU cases cover independent signals, temporal averaging,
reset, odd image dimensions, resize, confidence toggles, and discarded recording.

## Updating the snapshot

Initialize `External/nrd` only when updating the snapshot, then run:

```powershell
python scripts/VendorNrd.py
```

Review the CPU recipes and settings against the same upstream revision, update
the revision in this document and the script, then run the focused suite. The
script preserves the algorithm bodies and applies the following Slang adapters:
native descriptor access, constant-field aliases with a separate prefix to avoid
macro rescanning, explicit compute entry points, parenthesized comparisons that
Slang otherwise parses as generic arguments, and REFERENCE bounds checks.

SIGMA is integrated into the realtime deferred resolve through `RayTracedShadowPass`.
The legacy `ScreenSpaceShadows` source files now trace every shadow ray against the
scene TLAS, following the NRD-Sample sun-disk/occluder-distance packing workflow.
It uses a dedicated SIGMA-only plan, preserving its encoded output across frames
for the Copy/TemporalStabilization stages. Its owner serializes frame overlap;
resize, camera cuts, scene/transform/light changes, settings changes and cancelled
recordings invalidate its history. See `Documentation/ScreenSpaceShadows.md` for
trace inputs, controls, limitations and validation commands.
