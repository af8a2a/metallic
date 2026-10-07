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
references the existing OpenPBR glTF geometry and uses the existing LookDev
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

[Clouds.slang](../Shaders/Modules/Clouds.slang) defines a prescribed spherical
cloud layer from base/top altitudes. Seeded periodic value noise supplies
8 km and 2 km shape scales and 0.5 km erosion; coverage thresholds that field
and a vertical profile fades the layer boundaries. It is an advected density
field, without fluid dynamics or cloud microphysics. Double-precision wind
displacement is reduced to the common 2048 km period before conversion to GPU
floats, avoiding an unbounded displacement coordinate during long playback.

Cloud extinction is grey. The fixed approximation uses scattering albedo 0.98,
Henyey–Greenstein anisotropy 0.75, a transmitted direct term and three broadened
higher scattering orders. This finite-order closure is not a full volumetric
path-traced multiple-scattering solution. Integration clips disjoint spherical
layer intervals, supports below-cloud, inside-cloud and above-cloud rays, and
adds cloud extinction/scattering to the existing atmosphere integrals. Each
clipped cloud region uses 24 integration steps and each clear region uses 32
gas integration steps.

The same prescribed field attenuates surface Sun/Moon lighting, visible disks
and ground illumination. It contributes to sky view and aerial perspective,
then to disk-free environment capture, diffuse SH, prefiltered specular and
the environment PDF. Production Deferred and PT therefore consume the same
published cloudy environment rather than independent authored sky colors.
Primary ray-origin handling and HDR/exposure contracts continue from B.

Active physical clouds must lie inside the atmospheric shell. Cross-field
validation rejects an incompatible layer transactionally, including scene JSON
loads, and the editor constrains layer controls to the shell. Dormant cloud
settings may remain authored while clouds or physical atmosphere are inactive.

Physical sky view and lighting capture use RGBA32F to retain bright forward
cloud scattering above 65504. Transmittance, multi-scattering and cloud shadow
maps retain RGBA16F; aerial data remains FP32.

## Celestial shadow scheduling

[buildCelestialShadowPlan](../Source/Runtime/Render/SceneLightResources.cpp)
evaluates Sun/Moon importance as unattenuated photometric irradiance times
positive elevation relative to the observer's spherical local up. Physical
mode derives that irradiance from TOA; HDRI uses authored lux. The dominant
source receives 48 cloud-shadow integration samples. A second source at least
`0.0001` of dominant importance receives 12. Sources at or below the local
horizon receive no map budget; ties retain the stable Sun slot.

The two-source cloud shadow map is 256² RGBA16F over a ±32 km observer-centered
ground footprint. R/G store Sun/Moon transmittance; B/A indicate valid mapped
source channels. Receivers below 500 m, below the cloud base and inside the
footprint use a source-ray projection onto the spherical ground before
sampling, preserving height-dependent cloud-shadow parallax.
Out-of-footprint, elevated or unbudgeted receivers use analytic cloud
transmittance. The budget controls map work and does not rewrite or discard a
source's direct irradiance.

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

Dynamic sky view, cloud shadows, lighting capture/mips, aerial perspective,
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
