# Experimental DLSS Neural Rendering

`DlssNrPass` integrates the direct NGX feature-18 contract from
`Unity-DLSS-RR/src/DLSSNRRuntime.cpp` with Metallic's Vulkan backend. It is
separate from the Streamline DLSS-SR and DLSS-RR passes and is disabled at build
time by default. The recovered API is experimental: Vulkan exports alone do
not establish that a particular DLL, GPU and driver can execute this feature.

## Build

Supply your own `nvngx_dlssnr.dll`; the project does not download or commit it.
For the local reference checkout:

```powershell
cmake -S . -B build-dlss-nr -DMETALLIC_BUILD_TESTS=ON `
    -DMETALLIC_ENABLE_DLSS_NR=ON `
    -DMETALLIC_DLSS_NR_RUNTIME=E:/Unity-DLSS-RR/External/NVIDIA-DLSS/lib/nvngx_dlssnr.dll
cmake --build build-dlss-nr --target Metallic MetallicRhiTests --config Release
```

Windows x64/MSVC, a working Streamline SDK, and its bundled NGX SDK are
required. Override `METALLIC_DLSS_NR_NGX_ROOT` if that SDK lives elsewhere.
The DLL is copied next to the executable. Missing build dependencies produce
a configure error only when the experimental option is explicitly enabled.

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
one snippet initialization per device and maintain independent histories. The
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
