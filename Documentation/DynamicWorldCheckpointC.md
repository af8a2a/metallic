# Checkpoint C — Dynamic World

Implementation notes for M6–M8 and M8.1 of the user-supplied
`D:/Metallic_Physical_Environment_Roadmap.md`, recorded on 2026-10-07.
The roadmap is a design reference; this document describes the implemented
contracts and their current validation status.

Checkpoint C extends [Checkpoint B](PhysicalEnvironmentCheckpointB.md) with
authored clock/location state, evaluated Sun/Moon motion and lunar phase,
weather-dependent atmospheric composition, volumetric clouds, cloud shadows
and dynamic environment lighting. M6–M8/M8.1 are implemented and the native
CPU/GPU validation below passed. Broader temporal quality, performance and the
prior SDK acceptance items remain explicitly outside this evidence.

## Sample and editor workflow

Open **Samples / Environment / Dynamic World / OpenPBR LookDev**
(`dynamic-world-lookdev`). Its
[scene document](../Asset/LookDev/DynamicWorld/DynamicWorld.metallic_scene.json)
references the existing OpenPBR glTF geometry and uses the atmosphere LookDev
comparison graph, with actual PT and Deferred branches. The authored start is
UTC JD `2460483.0`, latitude 35° north, longitude 0°, astronomical mode,
240 simulated seconds per real second, both celestial sources enabled and
moving clouds. Auto exposure is enabled; manual EV100 14 is the fallback.

In the editor's **Environment** controls:

1. Select **Physical atmosphere**, then open **Astronomy and clock**. Choose
   Manual, Simplified day cycle or Astronomical, and edit the start UTC Julian
   date, latitude/longitude, signed playback scale and pause state.
2. Read the runtime JD and Moon illuminated fraction, phase angle, distance
   and integrated phase. **Use runtime UTC as start** explicitly copies the
   current runtime JD into the authored start; this is a document edit.
3. In automatic modes, Sun/Moon directions and angular radii are evaluated
   outputs. The Sun's authored TOA spectrum and the lunar albedo control the
   evaluated lunar irradiance. Manual mode retains separately authored disk
   directions, radii and spectra. Source enable flags remain independent.
4. Open **Weather and clouds** to edit coverage, density, layer base/top,
   extinction, aerosol density, humidity, precipitation state, wind speed,
   horizontal wind direction and deterministic noise seed. Pause freezes
   astronomy; set wind speed to zero to freeze cloud drift.
5. Save or revert the scene document through the existing document workflow.
   Runtime playback alone does not mark the document dirty. HDRI intensity
   and rotation remain HDRI controls; physical lighting retains absolute TOA
   calibration, while background visibility is common to both sources.

## Authored state and frame evaluation

[WorldEnvironment](../Source/Runtime/Environment/WorldEnvironment.h) owns
`EnvironmentTimeState`, `AstronomyState` and `WeatherState`. Scene documents
persist them under `world.environment.time`, `.astronomy` and `.weather`.
Mode names are `manual`, `simplifiedDayCycle` and `astronomical`. Missing fields
in older documents preserve Manual mode, the paused J2000 clock and dry,
cloud-disabled weather. Loading validates the candidate transactionally;
invalid values do not replace the loaded document.

UTC Julian date and location use doubles. The supported date interval is
1900-01-01 through 2100-01-01, JD `2415020.5` through `2488069.5`. Playback
accepts signed simulated seconds per real second, including reverse and zero
speed. Latitude is north-positive in [-90°, 90°], longitude is east-positive
in [-180°, 180°]. World directions use local ENU: **X east, Y up, Z north**.
Stored celestial directions point along emitted light; shading points toward
the source with their negation.

[RenderWorld](../Source/Runtime/Render/Subsystem/RenderWorld.cpp) advances the
clock once per host frame, before change consumption, using a steady-clock
delta. Consumers read its cached detached snapshot throughout that frame.
The snapshot carries the authored celestial lights and atmosphere separately
from evaluated celestial lights and weather-adjusted atmosphere, along with
the authored time/astronomy/weather and evaluated astronomy outputs.

Changing pause or playback scale retains the accumulated runtime JD. Editing
the authored start JD explicitly resets it. Runtime JD clamps to the supported
date endpoints. Loading, reverting or clearing a document restarts its authored
clock and wind even when the SceneDocument address is unchanged. Ordinary
property and texture edits preserve playback; graph lifetime distinguishes
document replacement from GPU resource identity updates. Cloud advection uses
a separate real elapsed time; clock pause and simulated playback speed do not
change wind speed. The explicit
`setEnvironmentElapsedSeconds()` API provides a deterministic wind seek without
changing the astronomical clock. Scene overrides evaluate at their authored
JD rather than advancing independently inside a render pass.

Celestial, atmosphere, astronomy and weather revisions identify the affected
domains. Humidity/aerosol edits can change both weather and effective medium;
moving active clouds change weather. Runtime evaluation updates lighting and
temporal invalidation as needed, without geometry changes or writing evaluated
directions, JD or wind displacement back into the document.

## Astronomy and reflected Moon light

[Astronomy.cpp](../Source/Runtime/Environment/Astronomy.cpp) evaluates the Sun
and Moon, distances, apparent angular radii, Moon phase angle, illuminated
fraction and Moon-to-Sun direction. Astronomical mode uses low-precision orbital
elements with lunar longitude, latitude and distance perturbations, sidereal
rotation and ellipsoidal observer subtraction for topocentric directions.
UTC is a visual ephemeris input: leap seconds, TT-UTC, nutation, aberration and
atmospheric refraction are outside its precision contract. The observer model
uses the authored geographic location, rather than inferring a geographic
position from arbitrary scene coordinates.

Simplified day cycle is an intentional approximation: uniform local mean-solar
time, zero solar declination, no seasonal equation of time, and a coplanar
Moon with a uniform 29.530588853-day synodic period. Use Astronomical mode for
seasonal motion and lunar orbit variation. Manual mode remains useful for
LookDev and deterministic rendering cases. Automatic modes preserve both
authored enable flags; they do not dim the authored Sun because of weather.

The Moon uses reflected solar light with a spectrally neutral Lambert albedo.
For phase angle α (zero at full Moon, π at new Moon),

```text
illuminated fraction = (1 + cos(α)) / 2
Φ(α) = [sin(α) + (π - α) cos(α)] / π
Moon TOA = Sun TOA × lunarAlbedo × (2/3) × (MoonRadius / MoonDistance)² × Φ(α)
```

The evaluated Moon spectrum keeps B's 680/550/440 nm W/m²/nm contract. Its
photometric irradiance follows the same spectral/CIE-Y integration used by the
physical atmosphere. The automatic primary Moon disk resolves a Lambert
sphere's terminator and limb shading, normalized to the already phase-scaled
integrated flux. Physical path tracing samples the finite disk in solid angle,
including each sampled ray's atmosphere, planet and cloud transmittance.
Realtime surface lighting keeps the directional approximation. Secondary ideal
reflection/refraction misses include the visible physical disks with MIS weight
one; ordinary environment capture and non-delta proposals stay disk-free.
The permanent GPU probe checks projected flux, partial planet-limb visibility
and phase energy, and emits a four-phase Moon atlas using the actual shader.
Manual disks retain uniform radiance. This model does not
include a lunar texture, opposition surge, eclipses or a measured lunar BRDF.
The Sun's TOA spectrum remains an authored absolute input; orbital distance
is reported, while automatic evaluation changes its direction and disk radius.

## Weather and volumetric cloud transport

[WeatherState](../Source/Runtime/Environment/Weather.h) affects propagation.
Aerosol density and humidity scale the authored Mie scattering and extinction
together by `aerosolDensity × (1 + 3 × humidity²)`. The bounded coefficient
ceiling preserves their spectral ratio and single-scattering albedo. Default
dry weather preserves B's medium exactly. This hygroscopic growth rule is a
phenomenological rendering approximation, not a forecast or aerosol chemistry
model. Precipitation is a normalized weather state that increases cloud density
by `1 + 0.5 × precipitation`; rain particles, rainfall accumulation and wet
surface materials are outside this implementation.

[Clouds.slang](../Shaders/Modules/Clouds.slang) maps the authored spherical
layer to the default UE 5.7.4 NvRTX `m_SimpleVolumetricCloud_Inst` shape.
Its [material evaluation](../Shaders/Modules/SimpleVolumetricCloud.slang)
samples the RGB stratocumulus, altostratus and cirrostratus layout on a
horizontal tile of `cloudPatternWorldSizeKm` kilometres (UE default 256).
`cloudVerticalNoiseWorldSizeKm` controls the vertical scale of the 3D noise
independently and defaults to the reference value of 256 km. Matching it to
the horizontal size gives the noise equal world-space scales on all three
axes. `cloudPatternOffsetKm` translates the layout and its 3D noise together
along world X/Z in kilometres. Its default is (0, 0). The main view and the
ground shadow sample the same translated density field. Each channel
selects a separate height profile. The red channel
of the Perlin–Worley volume distorts the coordinates of the green channel,
which breaks up the cloud body. The material uses the maximum layout weight
as its conservative occupancy bound. The active layer spans 2% to 98% of
the authored thickness. Extinction fades in over 500 m from the camera.

The reference instance has zero storm and global mask contributions and
disables its third noise sample. Seed zero preserves the reference placement.
Other seeds translate the texture coordinates deterministically. Metallic
coverage 0.5 maps to the reference global coverage of -0.2. The authored
extinction of 8/km at unit density maps to 0.008/m. The two noise scales
advect with factors 7 and -6 using the scene wind direction and displacement.
The CPU wraps displacement at eight horizontal tile periods (2048 km by
default) before conversion to GPU floats.
Camera, light and environment capture rays all sample this material. The
[texture pack](../Asset/Atmosphere/CloudShape/README.md) contains the three
Windows platform mip-zero images and their import instructions. Each device
uploads the 9 MiB pack on first use by active clouds.

The [cloud lighting](../Shaders/Modules/VolumetricCloudLighting.slang) follows
the same UE instance and the default `VolumetricCloud.usf` lighting permutation.
It mixes two Henyey–Greenstein lobes (g = 0.8 and 0.166667) with weight 0.575
on the second lobe. The grey medium has albedo 0.98. Two additional scattering
orders use the material's scattering, extinction and phase controls of 0.666667,
0.25 and 0.18. Each source traces its own 15 km cloud shadow ray at ten
quadratically spaced segment midpoints. Each order integrates with its own
extinction. Only order zero advances the view transmittance.

The [cloud lighting LUT](../Shaders/Features/Environment/CloudLightingLut.slang)
stores the distant sky luminance and the ground-level illuminance of the Sun
and Moon in three RGBA32F texels. The sky calculation integrates the gas at
6 km altitude over 64 stratified directions with ten samples per ray. Its
disk-free luminance enters order zero once with the default material AO of 0.5.
Each celestial source uses its top-of-atmosphere irradiance attenuated along
a 15-step ray starting at 500 m, matching the UE component default with
per-sample atmospheric light transmittance disabled.
This path covers the default physical sky, Sun and Moon. UE's optional
local-light and ground-contribution cloud permutations are outside its scope.
The GPU generates the reference sphere sample sequence with the same seed
and stratum order. The parallel reduction synchronizes the final four lanes
explicitly. During ground-level integration, a zero extinction coefficient
keeps a vacuum band transparent even when a below-horizon ray crosses dense
subsurface samples.

The environment provider clips disjoint spherical layer intervals, supports
below-cloud, inside-cloud and above-cloud rays, and integrates gas and cloud
extinction together. It uses the same lighting inputs and ten-step light rays
as the primary view, with 24 view steps per clipped cloud interval and 32 per
clear interval. The finite capture integration and shared sky LUT feed the
existing Metallic IBL and path-tracing paths.

The same prescribed field attenuates surface Sun/Moon lighting, visible disks
and ground illumination. It contributes to sky view and aerial perspective,
then to disk-free environment capture, diffuse SH, prefiltered specular and
the environment PDF. Deferred and PT consume this shared lighting environment.
Primary ray-origin handling and HDR/exposure contracts continue from B.

Active physical clouds must lie inside the atmospheric shell. Cross-field
validation rejects an incompatible layer transactionally, including scene JSON
loads, and the editor constrains layer controls to the shell. Dormant cloud
settings may remain authored while clouds or physical atmosphere are inactive.

Physical sky view and lighting capture use RGBA32F to retain bright forward
cloud scattering above 65504. Transmittance and multi-scattering use RGBA16F.
The cloud shadow columns and aerial data use FP32.

### Deferred primary-view clouds

The [physical LookDev graph](../Pipelines/Samples/lookdev_atmosphere.metallic_graph.json)
enables `independentViewClouds` on Deferred. The
[Sponza GI atmosphere graph](../Pipelines/Samples/sponza_gi_atmosphere.metallic_graph.json)
composites the cloud over Lumen reflections before DLSS-SR. Both graphs use
the scene's existing Weather and clouds controls.
Deferred builds camera sky and aerial perspective with gas extinction and
scattering. Clouds still shadow the illumination of that gas, the ground and
opaque surfaces. The shared environment capture and PT keep the combined
gas and cloud transport described above.

`VolumetricCloudViewPass` reads scene-linear HDR, opaque reversed-Z depth,
the matching raster camera and its jitter. The optional `cameraAerial` input
reuses the Deferred camera's gas integration. A direct gas integration covers
orthographic rays and distances outside the LUT. The optional
`worldShadowFaces` input supplies directional VSM visibility. Place this pass
before display mapping or temporal upscaling. It publishes the following
full-resolution outputs.

| Output | Meaning |
| --- | --- |
| `scattering` | Premultiplied, atmosphere-adjusted cloud radiance in scene-linear RGB. Alpha is one minus cloud transmittance. |
| `transmittance` | Scalar view transmittance in [0, 1]. An empty ray returns 1. |
| `cloudDepth` | Transmittance-weighted representative distance along the ray, in metres. An empty ray returns 0. |
| `color` | Scattering RGB plus transmittance times the incoming scene RGB, preserving scene alpha. |

The trace follows the first contiguous cloud-shell intersection, truncated
by opaque depth and the planet. It uses the UE 5.7.4 NvRTX
`VolumetricCloud.usf` default distance-from-layer-entry convention, ray-step
integration, representative depth and aerial perspective at that depth.
Entry is limited to 350 km and the trace to 50 km. The step count scales
from 2 to 96, reaching 96 at a 15 km interval.

Both directional slots can receive opaque VSM shadows in the view pass.
The UE reference applies its opaque shadow query to the first slot.
Metallic uses its three atmospheric wavelengths, gas LUTs and existing
spectral-to-working-color conversion. The sky directions and ground
illuminance are evaluated on the GPU. The adapters use Y-up world metres, kilometres for the
spherical atmosphere, reversed Z and a zero empty-depth marker.

`distanceToSampleMaxCountKm` controls the cloud ray length at which the
view march reaches its maximum sample count, matching UE's
`r.VolumetricCloud.DistanceToSampleMaxCount`. Both view paths default to
15 km. The Sponza showcase uses 0.6 km for its 80 m thick cloud layer.
`viewRaySampleMaxCount` caps the main view march (default 96, range 2–768).
`shadowViewRaySampleMaxCount` sets the sample budget for self-shadowing
inside the visible clouds (default 10, range 2–80). Both controls apply
to the full-resolution and temporal paths and refresh cloud history on edit.
For perspective views, `temporalReconstruction` defaults to true.
The [UE Mode 0 reconstruction](../Shaders/Features/Environment/VolumetricCloudViewTemporal.slang)
traces one ray per 4×4 block of render pixels, recording both the near- and
far-depth cloud integrals. Each frame refreshes one of four interleaved
positions in the half-resolution image. The other positions reproject
history with the cloud-front depth and the previous camera. Near/far scene
depth changes can reject that history. A bilateral resolve uses the full
opaque depth to select the cloud contribution before compositing.
The first frame, a camera cut, a resize or a change to authored weather
starts with a spatial reconstruction. Ordinary camera motion and advancing
wind or time of day can reuse history.

Metallic fuses the 4×4 depth reduction into the trace. It uses scene-linear
FP32 cloud color and transmittance throughout. An additional RG32F image
carries opacity-weighted near/far representative distances through the same
sample choices as the cloud colors. The resolve recovers `cloudDepth` in metres.
The trace uses frame-indexed blue noise and the unjittered projection.
Orthographic views and graphs with `temporalReconstruction` disabled use the
full-resolution trace with fixed blue noise. The temporal path covers opaque
surfaces and sky; UE's water, MSAA and holdout permutations are outside this path.

The GPU regression `volumetric_cloud_view_depth_and_composite` checks the
outputs against an opaque ceiling before and within the cloud layer, an open
orthographic view, an in-cloud perspective view and disabled clouds. It checks
four consecutive reconstruction phases, camera translation, camera cuts,
re-enabling clouds and resizing the render targets.
`cloud_transport_weather_shadow_domains` checks forward, side and back
scattering, per-order self-shadow and radiance, the Beer–Lambert solution for
a homogeneous slab, and the lighting inputs under the Sun and Moon. It also compares
world-space density samples against an offline evaluation of the reference
material and verifies half and zero extinction at camera distances of 250 m
and 0 m.

## Celestial shadow scheduling

[buildCelestialShadowPlan](../Source/Runtime/Render/SceneLightResources.cpp)
evaluates Sun/Moon importance as unattenuated photometric irradiance times
positive elevation relative to the observer's spherical local up. Physical
mode derives that irradiance from TOA; HDRI uses authored lux. The dominant
source receives 48 cloud-shadow integration samples. A second source at least
`0.0001` of dominant importance receives 12. Sources at or below the local
horizon receive no map budget; ties retain the stable Sun slot.

The Sun and Moon each have an orthographic light-space projection with a
half-width set by `cloudShadowExtentKm` (32 km by default). The ground point
under the observer snaps to UE's 20 km world grid at the default extent.
Narrower projections shrink the snap interval in proportion to the extent.
A per-light adjustment aligns world zero to its tracing texels. The
[producer](../Shaders/Features/Environment/CloudShadow.slang) traces the
first illuminated spherical cloud interval using the UE full-resolution
shadow branch. It fuses the default 2×2 spatial filter into each output
texel. Two 256² tiles share a 512×256 atlas. Metallic uses FP32 through the
filter and in the atlas to accommodate its authored layer depths. UE writes
separate R11G11B10 textures before and after filtering.

Each column stores front depth in km, mean positive extinction per metre and
integrated optical depth. The [receiver](../Shaders/Modules/CloudShadowMap.slang)
interpolates within its tile and reconstructs optical depth from its distance
behind the cloud front, capped by the column integral. Alpha flags columns
whose light-side cloud interval crosses the near plane. The filter preserves
that flag over its whole footprint. A tall layer or grazing Sun can therefore
use the full analytic ray for the affected receivers. Surfaces, fog and
single-scattered atmospheric light sample the atlas at their own world
positions. Out-of-projection and unscheduled receivers also use analytic
cloud transmittance. The cloud material retains its ray-marched self-shadow.
The shadow atlas is rebuilt with the immutable atmosphere publication. Its
integration uses the existing Metallic 48/12 light budgets and the UE
single-frame spatial filter.

Fog injection and Lumen Surface Cache select their physical-sun shaders
when the environment publishes atmosphere parameters. The authored-sun
shaders use the scene irradiance directly. The GPU cases
`volumetric_fog_graph_toggle` and `lumen_surface_cache_capture` compare
clear and cloud-shadowed sunlight in the fog and on a surface card,
respectively.

The existing single-source screen-space shadow path automatically selects the
largest positive elevation-weighted celestial importance, then falls back to
the first enabled local light. Enabled explicit requests retain their stable
slots; invalid or disabled requests use automatic selection. Sun/Moon remain
slots 0/1 and disabled local slots remain present. Cloud shadow transmittance
and geometry visibility are separate factors in the resulting illumination.

## Publication reuse and M9 boundary

The physical environment provider retains immutable publications keyed by
evaluated GPU contents, with frame retention and cancellation handling. A new
publication with identical static medium shares the transmittance and
multi-scattering image owners. Time, Sun/Moon direction, phase or cloud drift
therefore rebuild dynamic resources while reusing those medium LUTs. Changing
the effective Mie medium, Rayleigh/ozone coefficients, radii or ground albedo
requires the corresponding medium computation again.

Dynamic sky view, cloud lighting, cloud shadows, lighting capture/mips, aerial perspective,
SH projection, specular prefilter and environment PDF are recorded together
with explicit resource dependencies. C uses synchronous publication of the
current evaluated state. It does not claim async overlap, temporally amortized
IBL, a throttled update cadence or a measured frame-time improvement. Those,
quality scaling, stronger spectral references and broader performance work
belong to M9. Per-frame resource regeneration also needs measured full-scene
cost and memory evidence before any performance acceptance claim.

## Validation

Evidence was collected on 2026-10-07 in the existing Windows/MSVC Release
`build-scheduling-release` tree, with an RTX 5070 Ti. That configuration has
NRC/Streamline enabled and NRD disabled. The rendering cases use native output
and cache mode off; they do not execute NRC, NRD/SIGMA or DLSS reconstruction.
Generated logs, XML, HDR files and images remain local build output.

| Area | Permanent coverage / collected evidence | Result |
| --- | --- | --- |
| Build | `MetallicSceneTests`, `MetallicRHITests`, `Metallic`, `LookDev`, `MetallicShaderRequestsTests` | Pass |
| Astronomy and persistence | `AstronomyTests.cpp`, `Weather.*`, `WorldEnvironment.*`: directions, phases, distances, locations, revision domains, round trip, cross-field rejection, legacy defaults and dynamic sample load | 17 focused tests pass; full scene suite 153 pass / 9 skip |
| Runtime owner | `dynamic_world_runtime_state`, `dynamic_world_scene_reload_clock`: cached advance, pause/reverse/zero scale, same-address reload, JD reset, wind seek, clean document and overrides | Pass |
| Celestial shadows | `dynamic_world_celestial_shadow_budget`, `dynamic_world_screen_space_shadow_selection`: importance, 48/12 budgets, horizon/threshold, preserved irradiance and stable selection | Pass |
| Shader/resource contract | All 320 parameter bytes and cloud-shadow binding 101 read back on GPU; warmup request coverage; mapped/native Slang compilation of PT variants | Pass; 7 ShaderRequests tests pass |
| Cloud transport and lighting | `cloud_transport_weather_shadow_domains`, `cloud_provider_capture_sh_specular_pdf`: wind, humidity/precipitation, source energy, ground/elevated/orbital rays, capture, SH, sharp/rough specular and PDF | Pass |
| Finite disks and Moon | `celestial_finite_disk_energy_phase_bounds`: cone/projected flux, limb clipping, full/quarter/crescent/new phase, non-axis tiny disk, zero radius and delta miss | Pass; four-phase GPU atlas inspected |
| B compatibility and publication | Atmosphere transport/planet-scale probes, provider cancellation/content eviction, celestial override reuse and immutable shared medium LUT ownership | Pass |
| Production rendering | `dynamic_world_raster_pt_cases`: six states, each rendered with actual isolated PT and Deferred graph branches; 128², 16 frames per path/state | Pass; raw HDR and 12 display images inspected |
| Dynamic history | `dynamic_world_playback_history`: paused astronomy with moving wind, then 12 simulated hours, fresh history/resources and clean authored document | Pass on both paths |
| Editor/sample smoke | `LookDev --sample dynamic-world-lookdev --render-path deferred --smoke-test --skip-shader-warmup` | Exit 0; actual editor frame submitted and presented |

The final core RHI run passed **18/18**, including four CPU runtime/shadow
contract tests and actual GPU resource/transport probes. The final rendering
run passed **3/3**, including B's source/history regression. Neither run skipped
a selected test. The full scene suite's nine skips concern optional USD support
or missing large Zorah/SuperSponza assets; they are not runtime evidence for
those paths.

The six rendering states are ClearNoon, BrokenNoon, OvercastNoon,
HumidRainMedium, Twilight and FullMoonNight. Within the tested sky region,
PT/Deferred mean RGB relative error is at most **0.004601%**, supporting shared
environment publication. Full-image relative RMSE is **19.8–49.6%** at 16
frames, so this is not full-image convergence or equivalence evidence.
The branches differ in surface-lighting approximation and the short PT run
retains sampling noise. Fixed EV100 values of 14, 8 and -3 make the noon,
twilight and lunar-night display captures reviewable.

The cloud probe retained a forward-scattering RGB value of approximately
`[100362, 91519, 80831]`, and sky/capture probes also retained values above
65504. This exercises the FP32 publication change rather than accepting an
FP16 brightness clamp. Cloud-shadow checks include a 300 m terrain receiver
and source-ray ground projection; the finite-disk probe includes a non-axis
Sun with a radius of `1e-8` radians and preserved integrated flux.

Final evidence files:

- [Build log](../build-scheduling-release/checkpoint-c-build-verified.log)
- [Full scene log](../build-scheduling-release/checkpoint-c-all-scene-verified.log)
  and [XML](../build-scheduling-release/checkpoint-c-all-scene-verified.xml)
- [Shader request log](../build-scheduling-release/checkpoint-c-shader-requests-final.log)
- [Core RHI HTML report](../build-scheduling-release/checkpoint-c-core-final/reports/17913387413549688/report.html)
  and [XML](../build-scheduling-release/checkpoint-c-core-final.xml)
- [Rendering HTML report](../build-scheduling-release/checkpoint-c-rendering-final/reports/17913388197184830/report.html),
  [metrics](../build-scheduling-release/checkpoint-c-rendering-final/DynamicWorldRendering.json),
  [contact sheet](../build-scheduling-release/checkpoint-c-rendering-final/DynamicWorldContactSheet.png)
  and [XML](../build-scheduling-release/checkpoint-c-rendering-final.xml)
- [GPU Moon phase atlas](../build-scheduling-release/checkpoint-c-core-final/MoonPhaseAtlas.png)
- [Editor smoke log](../build-scheduling-release/checkpoint-c-editor-smoke.log)

Reproduction from an x64 Visual Studio developer shell at the repository root:

```powershell
cmake --build build-scheduling-release --target MetallicSceneTests MetallicRHITests Metallic LookDev MetallicShaderRequestsTests
.\build-scheduling-release\tests\MetallicSceneTests.exe
.\build-scheduling-release\tests\MetallicShaderRequestsTests.exe --gtest_filter=ShaderRequests.*
# Replay the exact selected core test names from the recorded XML.
[xml]$coreEvidence = Get-Content build-scheduling-release/checkpoint-c-core-final.xml
$coreFilter = ($coreEvidence.testsuites.testsuite.testcase | ForEach-Object { "$($_.classname).$($_.name)" }) -join ':'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=$coreFilter --output-dir build-scheduling-release/checkpoint-c-core-recheck
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.physical_environment_source_and_history:RHIRendering.dynamic_world_raster_pt_cases:RHIRendering.dynamic_world_playback_history --output-dir build-scheduling-release/checkpoint-c-rendering-recheck
.\build-scheduling-release\Source\LookDev.exe --sample dynamic-world-lookdev --render-path deferred --smoke-test --skip-shader-warmup
cmake -DSOURCE_DIRECTORY=E:/metallic -P Tools/CheckShaderRegistry.cmake
git diff --check
```

Shader registry validation and `git diff --check` passed. Working-color
validation still reports the existing Rec.709 luminance weights in the vendor
snapshot `Shaders/Modules/RTXCRHair/BSDFUtils.slang`; this checkpoint does not
modify that snapshot.

The editor smoke does not establish interactive control usability or a long
temporal run. Native GPU captures establish these selected states and update
contracts, not full-scene memory behavior or measured performance. The prior
B report records NRC and DLSS numerical rendering separately from unresolved
SDK teardown/device-reinitialization issues. C did not rerun those SDK paths,
NRD/SIGMA, or claim to fix their shutdown/reinitialization gaps. They remain
open acceptance items alongside M9 performance and quality work.

## Follow-up: Moon longitude seam

The Moon-only Manual state exposed a sky seam when the emitted direction was
`(0.840, -1.000, 0.000)`: the source lies on the sky LUT's periodic longitude
boundary. The forward producer's first/last columns and direct transport
integrals agreed. The old software bilinear sampler's negative-index expression
returned column **63** for texel **-1** in the 192-wide sky LUT on the tested GPU,
instead of the intended column **191**. The resulting lookup mixed unrelated
sky directions across the seam.

`Atmosphere::sampleLinear` now wraps longitude into a positive period before
the half-texel offset and uses unsigned remainder on nonnegative indices. The
32³ aerial sampler follows the same indexing convention. Source radiometry,
Moon phase and transport integration are unchanged by this correction.

Two permanent regressions were added:

- `atmosphere_moon_longitude_seam`: four Moon/Sun/cloud states, eight elevation
  bands and 2,064 directions per state; checks periodic sky/capture/production
  consumer and 10 km aerial radiance/transmittance, angular roundtrip, and
  cloud-free source-plane/producer symmetry. TSV records also retain the old
  wrap expression and an independent branch-based interpolation reference.
- `physical_moon_longitude_seam_rendering`: actual isolated native PT and
  Deferred branches, clear and frozen-cloud Moon states, 512×256 HDR/display
  captures, fixed EV100 -3 and four accumulated frames. Its camera is outside
  the LookDev geometry at `(-10, 2, 0)`, facing away from the mesh toward the
  Moon. The numerical sky region excludes the Moon disk and requires the
  clear-sky central adjacent-pixel difference to remain below 0.5%.

Before the fix, the actual clear-sky PT and Deferred adjacent-pixel differences
were **16.216%** and **16.239%**. The paired reproduction uses the same camera,
source, exposure, viewport and accumulation settings. After the fix these
differences are **0.025901%** and **0.000002476%**, respectively. The frozen-cloud
images also have a continuous central sky; the cloud case is inspected visually
rather than being required to have a physically symmetric field.

The combined follow-up run passed **23/23**, with no selected skips: the prior
18 core contracts/probes, three production rendering/history tests, and the two
new seam regressions. Of these, four are CPU runtime/shadow contracts; the rest
exercise GPU resources, transport or rendering. Across all four numerical
seam states, the maximum sky/capture/consumer/aerial relative continuity error
is **0.002313%**, with angular roundtrip error below **4.04e-7** in vector length.
The selected native PT/Deferred captures were inspected, including both clear
sky regions without geometry occlusion and frozen-cloud detail. This evidence
does not rerun NRC or DLSS reconstruction.

- [Combined follow-up HTML report](../build-scheduling-release/moon-seam-final/reports/17913408661872346/report.html)
  and [XML](../build-scheduling-release/moon-seam-final.xml)
- [Moon sampling/aerial metrics](../build-scheduling-release/moon-seam-final/AtmosphereSeamMetrics.json)
- [Actual rendering metrics](../build-scheduling-release/moon-seam-final/MoonLongitudeSeamRendering.json)
- [PT before/after comparison](../build-scheduling-release/moon-seam-final/MoonSeamBeforeAfter.png)
- [Final clear Moon PT](../build-scheduling-release/moon-seam-final/ClearMoon_PathTrace.png)
  and [Deferred](../build-scheduling-release/moon-seam-final/ClearMoon_Deferred.png)
- [Final cloud Moon PT](../build-scheduling-release/moon-seam-final/FrozenCloudMoon_PathTrace.png)
  and [Deferred](../build-scheduling-release/moon-seam-final/FrozenCloudMoon_Deferred.png)

The editor, LookDev and RHI test targets build successfully in the existing
Release tree. Shader registry audit and whitespace checks pass. The two new
tests can be replayed independently with:

```powershell
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.atmosphere_moon_longitude_seam:RHIRendering.physical_moon_longitude_seam_rendering --output-dir build-scheduling-release/moon-seam-recheck
```
