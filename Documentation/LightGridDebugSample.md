# LightGrid coverage heatmap and test bench

Select **LightGrid / Coverage Heatmap** in the **Lighting** sample category. The sample uses no model, HDR image or other external asset. It generates a deterministic mixture of point and spot lights, runs the GPU clustered LightGrid, and presents `LightGridDebug.color` through `FinalBlit`. It does not change or persist virtual lights in the scene.

Select the **LightGridDebug** graph node to edit its **Runtime Settings** in the Inspector, including the camera eye, center, projection and clipping planes. The standalone pass is not connected to the editor's viewport mouse/WASD camera controls. Defaults are 512 lights, 25% spot lights, 3 metre ranges, seed 1, 32 pixel tiles, 32 depth slices and capacity 64. Animation is off so repeated runs have the same fixture.

## Reading the image

The heatmap shows **conservative LightGrid light-list coverage**, not rendered illumination, brightness, visibility or shadow coverage. The GPU tests light volumes against clustered cells; it does not test a receiver at every screen pixel. Coarse cells can therefore contain false-positive light candidates. No CPU cell-light calculation or readback generates the image.

- **Black** means zero lights inside a valid grid layer; nonzero counts run from **blue through cyan, green and yellow to red**. In `depth` mode, distances outside the grid also display black because no cell exists there, not because the lights have no influence. `heatmapMaxLights` sets the red end of the scale; it does not change culling.
- **Magenta** means the cell's bounded-local count exceeded `maxLightsPerCell`. This reports true `totalCount`, not the clamped stored count. The lighting query interface falls back to all bounded-local candidates for overflowed cells.
- `showGrid` displays screen-tile boundaries. `showLegend` displays the color scale, and `showCounts` enables numeric tile counts. The legend is hidden below 96 pixels wide or 48 pixels high; tile counts are hidden when the digits do not fit, including narrow partial edge tiles.
- `includeGlobalLights` adds directional and range-zero local lights to the displayed counts. Global lights are not stored in per-cell local lists and do not cause local-list overflow.

`visualization` selects which part of the 3D grid is shown:

| Mode | Meaning |
| --- | --- |
| `peak` | Maximum `totalCount` across all depth slices at each XY tile. This is not the sum or number of unique lights along the viewing ray. |
| `slice` | One Z layer, selected by zero-based `sliceIndex`. Values beyond the configured layer count select the last layer. Perspective layers are logarithmic; orthographic layers are linear. |
| `depth` | Layer selected by a uniform positive view-space distance `viewDepth`, in metres. Distances outside the near/far domain display black and do not include global-light counts. This is a diagnostic plane, not a scene-depth-buffer visualization. |

## Fixture and grid controls

`source="bench"` builds the isolated procedural fixture. `lightCount`, `lightRange`, `seed` and `spotFraction` control its density and composition. `layout="volume"` distributes lights through a volume; `layout="overlap"` concentrates them for overflow testing. `animate` enables motion for live-update testing. Changing these settings does not edit the world's authored lights.

`source="world"` instead synchronizes the current RenderWorld scene's imported lights and virtual world lights into a private, light-only GPUScene, then runs the production coarse collection and clustered-grid builder. It builds its own grid, rather than reusing a raster pass's existing grid snapshot. Imported lights loaded privately by another pass or supplied through a GPUSceneSubsystem source override are not automatically part of this RenderWorld source. Add `LightGridDebugPass` to a scene graph, supply a matching `camera`, and connect its `color` output to a preview or `FinalBlit`. It does not substitute a screen-culled grid for path-tracing light transport. Selecting the standalone sample intentionally starts with an empty scene, so world mode initially has no lights there.

`tileSize`, `depthSliceCount` and `maxLightsPerCell` trade grid resolution, storage and intersection work. Camera settings define the grid's view volume; the default eye is at the origin looking down -Z, near/far are 0.1/60 metres, and perspective vertical FOV is 60 degrees. Orthographic mode uses `camera.orthoHeight`. Follow the allocation and camera limits in [Physical lighting](PhysicalLighting.md#gpu-clustered-lightgrid); this debug pass does not bypass them.

## Repeatable checks

Start from the sample defaults and keep `animate=false` when comparing changes:

| Check | Settings | Expected observation |
| --- | --- | --- |
| Empty source | `lightCount=0` | Zero heatmap coverage apart from enabled overlays; no stale cells. |
| Determinism | Same seed, count, layout and camera | Same coverage; changing the seed changes the volume fixture. |
| Point / spot filtering | Compare `spotFraction=0` and `spotFraction=1` | Different coverage as cone filtering affects the spot-light cells. |
| Overflow | `layout="overlap"`, `lightCount=512`, `maxLightsPerCell=8` | Magenta overlapping cells; increasing capacity reduces overflow when enough entries fit. |
| Slice mapping | `visualization="slice"`, sweep `sliceIndex` | Different depth-layer coverage; switch to `depth` to inspect a known view distance. |
| Resolution | Compare tile sizes 16, 32 and 64 | Finer/coarser screen-space cells, including partial tiles at viewport edges. |
| Live updates | Toggle `animate`, change count or resize viewport | Grid and heatmap update without retaining removed-light coverage. |
| Projection | Switch perspective / orthographic | Correctly rebuilt grid with logarithmic / linear depth layers. |

Use the existing GPU profiler to compare the **combined LightGridDebug node cost** under the same viewport, camera and fixture settings. Its node timing includes grid construction and heatmap rendering; it is not an independent culling time. `peak` scans every Z cell for each tile, so visualization settings also affect the result. Separating grid-build and heatmap dispatch costs requires a GPU capture or additional timing instrumentation. Capacity controls list storage, not a light-count limit, so an overflow test is not equivalent to reducing the number of lights.

For an editor smoke run from a PowerShell terminal in the repository root:

```powershell
$env:METALLIC_SMOKE_TEST_SAMPLE = 'light-grid-debug'
& .\build\Source\Metallic.exe --smoke-test
Remove-Item Env:METALLIC_SMOKE_TEST_SAMPLE
```

Use the executable path produced by your generator/configuration if it differs. The smoke check exercises startup, graph compilation and a rendered frame; it is not a sustained GPU benchmark.
