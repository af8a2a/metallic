# Render Graph resource inspection

Open **Render Graph Editor > Resources**, or right-click an output pin and choose
**Inspect Resource**. The list includes Texture and Buffer bindings from executed
passes, plus resources published by their registered debug checkpoints. Filter by
resource or pass name. Input bindings are identified by the consuming pass name.

Select a resource to capture it on the next execution. **Refresh** captures once;
**Live** refreshes at most twice per second, with one pending request. Disabling
Live freezes the last completed snapshot. The displayed execution and generation
identify the data being shown; graph recompilation discards stale snapshots.

- Textures: RGB or individual channels, exposure, min/max display range, fit/zoom,
  scroll, ROI, and exact raw pixel fields/bytes on hover. Display values are
  mapped as `(value * exp2(exposure) - min) / (max - min)` and clamped; non-finite
  results are magenta. This is a diagnostic display, without scene tonemapping.
- Buffers: first element/count, virtualized rows, hexadecimal values, and
  uint32/int32/float32 interpretation of raw 32-bit words. Registered structured
  layouts use named fields. A buffer's stride alone never implies field types.

The texture path currently captures mip 0 and layer 0 of uncompressed 2D images:
RGBA/BGRA8 UNORM/sRGB, R/RG/RGBA16F, R/RG/RGBA32F, R/RG/RGBA32 uint/sint, and D32F.
sRGB images expose their stored UNORM values. Compressed/packed formats and 3D
images show an explicit unsupported reason. Oversized captures require a smaller
ROI; the default runtime budget is 16 MiB including evidence metadata. Raw buffer
ranges are measured in 32-bit words, with at most 65,536 requested elements.

Inspection uses the existing RenderDebugRuntime pass-boundary copies and restores
source resource synchronization state. CPU access waits for GPU completion through
polling; opening or closing the local inspector changes graph allocation policy
and causes recompilation. While attached, debug resources are pinned, transfer
source usages are enabled, and debug execution restricts parallel recording.
Therefore timings taken in this mode are not normal renderer performance timings.
Closing Resources detaches the local observer; an explicit `--debug-control`
session keeps its observer active. Preview uploads create separate retained images
so in-flight frames never sample a replaced graph resource or a rewritten preview.

## Native diagnostics without desktop automation

Build `Metallic`, then run from the repository root:

```powershell
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
.\build-release\Source\Metallic.exe --skip-shader-warmup `
    --resource-inspector-smoke .cache\InspectorEvidence
Remove-Item Env:METALLIC_SMOKE_TEST_HIDDEN
```

This uses the real editor and GPU. It verifies known texture/buffer values, preview
GPU upload and display conversion, channel/exposure controls, live/frozen snapshots,
ROI/ranges, invalid ranges, recompile, and close/reopen. Exit code 0 indicates success.
Evidence includes:

- `report.json`: check results, artifact identities and capture provenance.
- `texture/` and `buffer/`: standard `manifest.json` and `0.bin` captures, consumable
  by `metallicctl --capture <directory>` offline.
- `texture-ui.ppm` and `buffer-ui.ppm`: native ImGui rendering of the editor window,
  cropped and read back from an offscreen target. No screenshots, clicks, focus
  changes, or desktop automation are used. HDR UI output is converted to SDR using
  the configured paper white for these images.

For example, verify the known fixture's 18th word offline (expected value 292):

```powershell
.\build-release\Source\metallicctl.exe --capture .cache\InspectorEvidence\buffer `
    eval 'buffers["Data.values"][17]' --json
```

In a tests-enabled build, build `Metallic` first and run
`ctest --test-dir <build-dir> -R '^MetallicResourceInspectorSmoke$' --output-on-failure`.
The smoke uses an isolated generated graph, does not save scene/graph changes, and
does not overwrite the user's ImGui layout. Artifacts belong in local output folders.

For a running editor started with `--debug-control`, `metallicctl --pid <pid> hello
--json` exposes `result.engine.resourceInspector`, including the selected resource,
pending job, status, live flag, preview readiness, and captured execution/generation.
Existing `capture.batch`, `jobs.get`, and `artifact.read` commands remain available
for direct capture automation without opening the Resources page.
