# Experimental DLSS Neural Rendering

`DlssNrPass` integrates the direct NGX feature-18 contract from
`Unity-DLSS-RR/src/DLSSNRRuntime.cpp` with Metallic's Vulkan backend. It is
separate from the Streamline DLSS-SR and DLSS-RR passes. The NR DLL is not
publicly released, so builds without a manually installed runtime disable NR.
The recovered API is experimental: Vulkan exports alone do
not establish that a particular DLL, GPU and driver can execute this feature.

## Build

Manually place your own runtime at **`External/nvngx_dlssnr.dll`** before
configuring. Metallic does not download it, copy it from the Unity reference
checkout, or commit it. The repository's `*.dll` ignore rule excludes it from
source control. This is the only accepted runtime location; the former
`METALLIC_DLSS_NR_RUNTIME` cache override is no longer used.

```powershell
cmake -S . -B build-dlss-nr -DMETALLIC_BUILD_TESTS=ON -DMETALLIC_ENABLE_DLSS_NR=ON
cmake --build build-dlss-nr --target Metallic MetallicRhiTests --config Release
```

Windows x64/MSVC, a working Streamline SDK, and its bundled NGX SDK are
required. Override `METALLIC_DLSS_NR_NGX_ROOT` if that SDK lives elsewhere.
When available, the manually supplied DLL is deployed beside the executable.
If the DLL, platform or SDK requirements are missing, configuration succeeds
with **`METALLIC_HAS_DLSS_NR=0`** and prints the reason. This also applies when
`METALLIC_ENABLE_DLSS_NR=ON`; that option cannot bypass the missing DLL check.

`METALLIC_HAS_DLSS_NR` is derived by CMake; do not define it manually. The
`METALLIC_ENABLE_DLSS_NR` option defaults to `ON`, allowing detection of the
manually supplied DLL. Set it to `OFF` to disable NR even when the DLL exists.
Options are cached **per build directory**; older caches may retain `OFF`.
Configure the profile you actually build and run, for example:

```powershell
cmake --preset metallic-release -DMETALLIC_ENABLE_DLSS_NR=ON `
    -DMETALLIC_DLSS_NR_NGX_ROOT=External/streamline/_sdk/external/ngx-sdk
cmake --build --preset metallic-release
```

An older cache can retain the unpackaged NGX header path, so update
`METALLIC_DLSS_NR_NGX_ROOT` as above when switching to the packaged SDK.
After configuring externally, reload the IDE's CMake project to refresh its
code model. For a custom IDE profile, also save these `-D` arguments in that
profile's CMake options so resetting its cache retains the opt-in.
With the local DLL and SDK present, configuration emits `Experimental DLSS-NR
enabled` and the runtime target's compiler command contains
`METALLIC_HAS_DLSS_NR=1`. Adding or removing the DLL triggers reconfiguration
on the next build. Rebuilding a disabled executable also removes any previously
deployed NR DLL from its output directory.

## Editor and render graph

Select **PathTracingSample / DLSS-NR (Experimental)**. Its pipeline is:

```text
PathTrace -> DLSS-RR (DLAA) -> AutoExposure -> DLSS-NR -> FinalBlit
     motionVectors + depth --------------------^
```

RR uses DLAA so the resolved color, motion vectors and depth all have the same
extent. NR operates on tone-mapped, sRGB-encoded RGBA8 UNORM display color.
AutoExposure performs the existing exposure, tone curve and sRGB encoding
before NR; the result then goes directly to FinalBlit. Linear HDR must first
be converted to display color. There is no extra gamma conversion after NR.
Do not feed noisy Monte Carlo samples directly to NR as a replacement for RR.

The native-resolution pass has these ports:

| Port | Format | Contract |
| --- | --- | --- |
| `inputColor` | RGBA8 UNORM | Tone-mapped sRGB display color, values in [0, 1] |
| `motionVectors` | RG16F | Current-to-previous UV displacement, without jitter |
| `depth` | R32F | Normalized hardware depth, not `linearDepth` |
| `color` | RGBA8 UNORM | Distinct writable output at the same resolution |

All four textures must be distinct, single-mip, single-layer 2D storage
textures. All three inputs also require **Sampled** usage, even though their
layout is General. Omitting it can produce near-black output while NGX still
returns success. The pass converts Metallic's UV motion to input pixels by multiplying
by width/height. Unity's negative scaling does not apply. `depthInverted`
defaults to true to match Metallic's reversed-Z guides.

Runtime controls expose enable, preset 0–3, style 0–2, intensity, local tone,
local structure, skin structure, automatic masking, UI correction, reversed Z,
and history reset. Artistic controls are written before feature creation;
changing them recreates the feature because this runtime captures tuning at
creation. Each pass owns a separate feature and parameter map. History
resets after graph compilation, configuration/extent changes, history
invalidation, scene changes, frame discontinuities, and `resetSerial` changes.
Signal camera cuts through graph history invalidation or `resetSerial`.

### Slider debug comparison

Enable **DLSS-NR Slider Debug** in the viewport toolbar, or **Slider Debug
(Before / After)** in the `DlssNr` node's runtime settings. It defaults off.
The left side shows `inputColor` before NR and the right side shows the NR
result from the same frame. Both use the same exposure and display color space.

Drag the viewport divider, or use **Split Position**, **Top / bottom**, and
**Swap A/B**, just as with `SliderDebugPass`. The properties are `sliderDebug`,
`splitPosition` (default 0.5), `orientation` (`vertical` or `horizontal`), and
`swapSides`. Position 0 shows all after-NR, position 1 all before-NR; swapping
reverses them. Disabling comparison restores the complete NR result.

NR still evaluates the whole image. The comparison shares SliderDebug's
pixel-center selection shader and only reveals original pixels over the NR
output, without scaling, blending or another full-size texture. Divider and
labels are editor overlays and are excluded from image captures. Comparison
controls do not rebuild the graph or reset NR history. When NR is disabled or
falls back, both sides show the original image. The comparison shader uses the
editor's bindless descriptor heap and is initialized on first use.

`fallbackToInput=true` logs initialization/evaluation failures and copies the
input color. An evaluation failure latches bypass until graph recompilation;
it does not retry or spam the log every frame. Set `fallbackToInput=false`
for strict validation: the graph then reports the NGX error instead of treating
the fallback image as evidence of successful NR execution. Turning NR off
always performs an exact copy and works without the SDK or an NVIDIA GPU.
Zero intensity also copies the input, since the snippet may otherwise leave
its output unwritten.

The lower-level `vulkan::DlssNrContext` also validates the reference's recovered
2x upscaling contract and accepts matching RGBA8 or RGBA16F display-color
textures. The graph pass exposes only native resolution. Actual GPU validation
covers native mode; 2x is only contract-validated in this integration.

## Runtime ownership and compatibility

The runtime uses `NVSDK_NGX_VULKAN_Init_Ext2`, API `0x15`, and
`NVSDK_NGX_VULKAN_CreateFeature1` with feature `18`. As in the reference, direct
snippet initialization needs a caller-module compatibility shim. Metallic
temporarily redirects only the snippet's imported `GetModuleFileNameW` to the
already loaded NGX core during each snippet API call, then restores it. The
runtime checks the caller on create/evaluate/shutdown as well as init. It never
modifies the DLL on disk. This behavior is confined to the opt-in backend.

The device must have been created with `enableStreamline=true`; this initializes
the NGX core and enables Metallic's NVIDIA Vulkan extension set. Contexts share
one snippet initialization per device; each stores its feature handle and
parameter map. The
parameter allocator is resolved from Streamline's already loaded NGX core;
Metallic does not link a second NGX loader or initialize/shut down that core.
Destroy contexts before their device. Feature replacement/destruction waits
for GPU use; this pass does not opt into frame overlap or async queues.
Before evaluation, Metallic reuses Streamline's empty legacy descriptor set
workaround for `VK_EXT_descriptor_heap`. After NGX dispatch, it invalidates its
cached descriptor binding state so subsequent passes rebind correctly.

## Validation

```powershell
build-dlss-nr/tests/MetallicRhiTests.exe --filter dlss_nr
build-dlss-nr/tests/MetallicRhiTests.exe --rhi-streamline --filter dlss_nr_runtime `
    --output-dir build-dlss-nr/tests/dlss-nr-output
$env:METALLIC_SMOKE_TEST_SAMPLE = 'pathtracing-sample-dlss-nr'
build-dlss-nr/Source/Metallic.exe --smoke-test
Remove-Item Env:METALLIC_SMOKE_TEST_SAMPLE
```

The focused tests cover resource/settings validation, graph/sample wiring,
exact bypass pixels across resize, missing-runtime fallback, strict failure,
and optional real Vulkan initialization/three-frame evaluation, tuning-driven
feature recreation, exact zero-intensity bypass, plus an eight-frame scene test
that saves images before and after NR. `--rhi-streamline`
creates a device with the editor's bindless heap and ray-query features.
Validation is on by default. Multi-config builds add the configuration directory
to executable paths. Runtime tests skip when the build or runtime reports
unsupported, and never enable
fallback. A passing bypass test or editor smoke test by itself does not prove
that neural rendering ran.

`MetallicDlssNrConfiguration` checks missing/present local DLLs, explicit
disable, ignored legacy runtime overrides, unsupported dependencies and NGX
LFS pointers. It also builds a tiny fixture through DLL removal and restoration
to verify automatic reconfiguration, compiler definitions and deployment.
All fixture DLLs stay under the test build directory; they are not NR binaries.

`dlss_nr_runtime_slider` compares every RGBA byte against the original input
and an uninterrupted reference sequence across both axes, swapped sides,
endpoints, odd dimensions, and enable/disable transitions. The reference is
captured in a separate feature lifetime: concurrently evaluated identical
features did not produce identical histories with the tested snippet. The scene
test also saves `dlss_nr_scene_slider.png`. Run the existing editor interaction
smoke test on the NR sample with both `METALLIC_SMOKE_TEST_SLIDER=1` and
`METALLIC_SMOKE_TEST_SAMPLE=pathtracing-sample-dlss-nr` set for
`Metallic.exe --smoke-test`.

The upstream SR/RR pass also declares Sampled usage on all NGX inputs; otherwise
its sampled reads can yield black color and invalid shader-read transitions.

Local validation on 2026-09-10 used an RTX 5060, driver 610.47, and the reference
checkout's DLL. The parameter/bypass and DLSS regression tests exit successfully.
Native RGBA8 and RGBA16F evaluations, eight scene frames, and editor frame
submission work. The scene run with Vulkan validation has no validation errors.
The disabled backend also compiles without the NGX headers. This machine hangs
during Streamline's NGX shutdown. A no-NR control reproduced the same wait; thread
stacks locate it in `_nvngx.dll -> NvTelemetryAPI64.dll::UninitializeTelemetry`,
with a telemetry bridge waiting for a named pipe. Thus a complete process-exit
pass cannot be claimed here. No driver services or global settings were changed.

The Unity reference supplies the feature IDs, entry points, controls and caller
shim. Display-color handling and creation-time tuning were also checked against
[OptiScaler's native Vulkan NR implementation](https://github.com/Dagherbou/OptiScaler_DLSSNR/tree/973761621353b99bee3dc7d4bb27b117fef2644f/OptiScaler/dlssnr).
