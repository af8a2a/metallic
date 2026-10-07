# Checkpoint B — Physical Environment

Implementation and evidence for M3–M5 of the user-supplied
`D:/Metallic_Physical_Environment_Roadmap.md`, recorded on 2026-10-06.

Physical atmosphere is an environment provider consumed by the production
Deferred, PT and ReSTIR paths. Select **Environment / Physical atmosphere** in
the editor or the **Environment / Physical Atmosphere** built-in sample
(`physical-atmosphere-lookdev`). HDRI remains the default for existing documents.
See [physical lighting](PhysicalLighting.md) for authoring and exposure.

The core M3–M5 implementation and the listed rendering/history checks are
complete. Clean SDK shutdown and device reinitialization remain open acceptance
items for NRC and DLSS; their pixel results are recorded separately below.

## State, units and transport

[AtmosphereState](../Source/Runtime/Environment/Atmosphere.h) stores the planet
center in double-precision world metres, radii and integration distances in
kilometres, and scattering/absorption coefficients in km⁻¹. The GPU boundary
subtracts the double observer and planet positions before narrowing to floats.
Shading positions are reconstructed relative to that observer. Planet geometry
is spherical; the ray integrals clip to the atmosphere shell and ground sphere,
including observers outside the atmosphere and rays through vacuum.

`world.environment.source` selects `hdri` or `physicalAtmosphere`.
`world.environment.atmosphere` persists the medium; Sun/Moon persist
`topOfAtmosphereIrradiance` at 680/550/440 nm in W/m²/nm. Scene loading validates
these values transactionally. Detached snapshots separate celestial,
atmosphere and lighting revisions. Weather remains reserved for Checkpoint C.

The three propagation samples are spectral coefficients, not RGB channels.
Piecewise-linear spectral reconstruction and a 1 nm integration of analytic
CIE 1931 matching functions convert radiance/irradiance through XYZ to ACEScg.
The 683 lm/W factor provides photometric transport units. The default solar
TOA spectrum integrates to approximately 130,938 lux; manual exposure is applied
after transport. The disabled Moon preset has a full-Moon-scale TOA irradiance;
phase, ephemerides and lunar surface modeling remain later work.

The implementation uses the transport integrals and isotropic multiple-scattering
closure described by [Hillaire, EGSR 2020](https://sebh.github.io/publications/egsr2020.pdf).
The matching-function approximation follows
[Wyman, Sloan and Shirley, JCGT 2013](https://jcgt.org/published/0002/02/01/paper.pdf),
equation 4/table 1. These are original shader implementations; no reference
implementation or vendor shader snapshot was copied.

## GPU publication and consumers

| Resource | Resolution / format | Purpose |
| --- | --- | --- |
| Transmittance | 256 × 64, RGBA16F | Rayleigh/Mie/ozone extinction to a celestial source |
| Multi-scattering | 32 × 32, RGBA16F | Isotropic geometric-series closure, including ground response |
| Sky view | 192 × 108, RGBA16F | Observer-dependent sky radiance |
| Lighting capture | 512 × 256, RGBA16F, complete mip chain | Disk-free lat-long radiance for existing IBL/PDF/PT consumers |
| Aerial perspective | 32³, two RGBA32F values per cell in a storage buffer | In-scattering and transmittance over camera segments |
| Atmosphere parameters | 192 bytes | Shared CPU/Slang medium, observer and celestial state |

[AtmosphereResourcesGPU](../Source/Runtime/Render/Environment/AtmosphereResources.cpp)
records the five precomputation kernels. The environment subsystem publishes
immutable, frame-retained resources and feeds the existing GPU SH projection,
cosine convolution, specular prefilter and importance-PDF builders. The eight-entry
content cache retains stable content revisions across allocation eviction.
Canceled, unsubmitted publications cannot be reused. Shader reload retires the
cached publications.

Sky-view and aerial direction coordinates use the observer's local radial
basis. Their vertical coordinate splits at the spherical planet limb, with
zenith angle `acos(-sqrt(1 - (bottomRadius / observerRadius)²))`. Each interval
uses `u² / (u² + (1-u)²)` to concentrate samples at its endpoints, preserving
zenith detail and the ground/orbital horizon. The inverse mapping is shared by
LUT generation and lookup. The lighting capture converts this sky-view mapping
to the existing lat-long environment interface.

Aerial distance slices use `d(s) = 0.01 * ((1 + maxDistanceKm / 0.01)^s - 1)`
with `s = slice / 31`. The first slice is zero distance; subsequent slices
cover nearby haze and long segments logarithmically. Lookup interpolates
between adjacent slices in travelled distance, so in-scattering approaches
zero linearly near the camera. Segments beyond the LUT's authored range use
the clipped ray integral, including an orbital ray crossing vacuum before
entering the atmosphere.

Primary transport carries the actual world-space ray origin through the named
environment consumer. Perspective rays originating at the published observer
use the sky/aerial LUTs. Orthographic pixels have distinct parallel-ray origins;
their camera segments and background sky are integrated from those origins,
including source extinction and planet-limb visibility. The same actual-origin
path also handles segments beyond the aerial LUT's distance range.

The named environment parameter block is 28 bytes; scene parameters are 472
bytes. The fixed two-record `GPUCelestialLight` contract remains 48 bytes per
record. Explicit optional compute bindings preserve invalid sentinels for HDRI;
missing required resources still fail dispatch validation.

Sun/Moon direct lighting evaluates TOA irradiance multiplied by atmospheric
transmittance at the actual shading position. Planet occlusion suppresses direct
light below the local horizon. Primary backgrounds add finite disks with
per-ray atmospheric extinction and planet-limb visibility. Lighting captures,
SH, prefiltered specular and the environment PDF exclude those disks, avoiding
a second solar contribution. Secondary PT misses use the captured radiance,
which is the initial M4 environment interface.

Primary surface color composites `T * surface + in-scattering` before exposure.
Perspective rays beginning at the observer use the aerial/sky LUTs. Orthographic
pixels retain their actual primary origins: offset segments and backgrounds use
the spherical ray integrals, with solar-disk extinction and limb visibility
evaluated at those origins.
PT applies it once to the completed primary camera segment; radiance caches
retain world radiance. Native NRC resolve receives per-pixel camera-segment
metadata and applies the composite after cache resolution and before history
averaging. ReSTIR separates attenuated diffuse/specular signals from the
emissive/in-scattering guide so the denoiser composite adds scattering once.

For physical scenes, NRC's default expected-average-radiance normalization
uses unattenuated TOA photometric Y divided by pi, with a floor of one. The
authored Sun/Moon spectra contribute even while disabled, keeping the
normalizer stable during source enable/disable tests; an explicit
positive `nrc.maxExpectedRadiance` property overrides it; zero selects automatic
normalization (HDRI defaults to one). Scene/environment changes request a cache reset, stored as a
pulse so clearing the request on the next frame does not configure/reset
the cache a second time. Native NRC rendering/history passed; clean SDK
shutdown and device reinitialization remain outstanding below.

The NRD adapter scales only the 16F noisy diffuse/specular RGB signals. Its
inverse scale is constant across the frame and is stored in the 32F emissive
guide's alpha; emissive RGB, sky and aerial in-scattering retain their absolute
HDR values. The composite restores denoised RGB before applying albedo/F0 and
adding emissive RGB. HDRI uses inverse scale one. The bound uses enabled
Sun/Moon TOA spectra, absolute spectral-to-Rec709 transform coefficients, and
the current RTXDI power-lobe maximum `130 / (2*pi)`, targeting a noisy RGB
maximum of 64. This leaves headroom for RELAX's FP16 second moments, whose
square must remain representable; see the bundled
[NRD HDR input guidance](../External/NRD/README.md).

RTXDI confidence reads the same optional emissive alpha and restores absolute
luminance before comparison. Its luminance history is RG32F and its filtered
gradients are RGBA32F, without the former 65,504 clamp. Darkness bias and the
denominator epsilon therefore retain their original units across scale changes.
Graphs without this optional input use inverse scale one.

Physical color and accumulation, including guide-export paths, remain RGBA32F
to preserve approximately 10⁹ cd/m² solar-disk radiance. DLSS SR/RR color inputs
and outputs also use RGBA32F. NVIDIA's
[RR color-resource contract](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md)
permits standard color formats; guide formats remain unchanged.

## Validation

These results were captured on an NVIDIA GeForce RTX 5070 Ti, driver 616.92,
with Vulkan core validation enabled. Generated captures and raw HDR readbacks
stay under the ignored `build/` directory. They validate correctness in the
listed workloads; the durations in test logs are case wall time.

| Scope | Result | Evidence |
| --- | --- | --- |
| Scene persistence, revision domains and input validation | 141 passed, 9 skipped; CTest passed | `build/checkpoint-b-scene.log`; `build-scheduling-release/Testing/Temporary/LastTest.log` |
| Final atmosphere transport/LUTs, actual primary ray origins, named ABI, provider cancellation and eviction | 10 core GPU tests passed | `build/checkpoint-b-final-gpu.log`; `build/physical-environment-checkpoint-b/final/` |
| Final isolated Raster/PT cases and source/history | 2 passed in the same final run: 12/12 GPU tests passed overall | `build/checkpoint-b-final-gpu.log`; `final/PhysicalEnvironmentRendering.json`; `final/PhysicalEnvironmentHistory.json` |
| Confidence, virtual celestial lights and LookDev execution | Earlier production run: 10 passed; initial NRC fixture failed because native NRC was not executed | `build/checkpoint-b-production.log`; `build/physical-environment-checkpoint-b/production/` |
| NRD chromaticity, physical HDR/aerial integration and ray-traced shadow history | Earlier 3 passed; final target-64 physical HDR/aerial test passed again after actual-origin changes | `build/checkpoint-b-nrd.log`; `build/checkpoint-b-nrd-final.log`; `build/physical-environment-nrd/PhysicalNRDIntegration.json` |
| LookDev physical atmosphere startup/presentation smoke | Passed, exit 0; physical sample loaded and a Vulkan frame presented | `build/checkpoint-b-lookdev-smoke.log` |
| Native NRC source/history and camera-segment resolve | Latest Auto/actual-origin run: 1/1 numerical/history test passed, exit 0; 54 leaked Vulkan objects remain at teardown, so clean SDK lifecycle remains outstanding | `build/checkpoint-b-nrc-acceptance.log`; `build/physical-environment-checkpoint-b/nrc-acceptance/PhysicalEnvironmentNRCHistory.json` |
| DLSS SR/RR 32F contract, bypass and RR history | 5 pixel/contract tests passed; process exited 1 after SDK teardown logged an exception/minidump and a leaked semaphore; final acceptance pending | `build/checkpoint-b-dlss.log` |
| ShaderRegistry usage audit | Passed | `build/checkpoint-b-shader-audit.log` |
| Working-color audit | Failed on unchanged vendor RTXCRHair luminance weights; that source matches HEAD | `build/checkpoint-b-color-audit.log`; `Shaders/Modules/RTXCRHair/BSDFUtils.slang` |

The final source diff passed `git diff --check`.

The nine Scene skips cover unavailable large Zorah/SuperSponza fixtures and
optional USD import support. They do not count as runtime coverage of those
assets or import paths.

The core filter was
`*atmosphere_*:*physical_environment_provider_*:*named_optional_resource_sentinel_and_required_contract`.
The final run included this filter and the two real Raster/PT case/history
tests, passing all twelve tests after the actual-primary-origin fix.
It exercises zero/vacuum limits, atmospheric extinction, Rayleigh/Mie/ozone
and ground response, finite disks crossing the planet limb, actual aerial
lookup, orbital angular mapping, optional resource sentinels, retained owners,
ten eviction publications and canceled-command recovery with and without a
pool reset. Rebuilding an evicted allocation preserved the content revision
and retained SH values; HDRI publications contained no physical resources.

The actual aerial LUT was compared with ray integration at three distances:

| Distance | Maximum relative RGB in-scattering error | Maximum absolute channel transmittance error |
| --- | --- | --- |
| 10 m | 0.0002922 | 4.172 × 10⁻⁷ |
| 1 km | 0.0081990 | 9.739 × 10⁻⁵ |
| 50 km | 0.0003276 | 1.119 × 10⁻⁴ |

These measurements come from `verified/aerial-noon.tsv`. The allowed bounds
are 15% in-scattering and 0.015 absolute transmittance error. Vacuum
in-scattering was zero, transmittance differed from one by at most
2.98 × 10⁻⁷, and the zero-distance sample was exactly `(L=0, T=1)`.
The primary disk reached approximately 1.75 × 10⁹ in working RGB while the
disk-free sky and lighting capture remained near 4,300–7,400. The separate
planet-limb probe preserved visible upper disk rays while the hidden centre
and lower disk rays returned zero.

At an observer altitude of 150 km, five directions across the spherical limb
had a maximum relative photometric-Y sky lookup error of 0.0186835 against
ray integration, within the test's 20% bound. Direction mapping round-trip
error was at most 2.03 × 10⁻⁷ and direction unit-length error at most
5.96 × 10⁻⁸.
`verified/orbital-limb.tsv` records the probe values;
`verified/OrbitalSkyLookup.png` and `verified/OrbitalSkyReference.png` show
the paired 640 × 360 GPU outputs, with matching `.rgba32f` files.

`atmosphere_actual_primary_ray_origin` binds the production named-resource
consumer and compares two parallel 50 km camera segments starting at nominal
altitudes of 20 and 40 km with ray-integration references. The GPU outputs
match within the test's 10⁻⁵ relative radiance and 10⁻⁶ absolute transmittance
bounds, respond to their different starting altitudes, and differ from an
eye-origin lookup. Offset-origin sky and Sun extinction match their independent
references; the upper disk remains visible at the offset planet limb while the
lower ray and the eye-origin planet mask are occluded. Zero-length and vacuum
segments preserve `(L=0, T=1)` within tolerance, and the shared-origin
perspective path still matches the actual aerial LUT. The final run's
`parallel-ray-origins.tsv`, `parallel-ray-vacuum.tsv` and
`offset-origin-limb.tsv` preserve all production/reference probe values.

`physical_environment_raster_pt_cases` renders 32 frames at 128 × 128 per
case and per isolated path, with caches off and manual EV100 14. Its twelve
cases cover noon, sunset at 2°, twilight at −6°, observer altitudes of
2/10/100/200 km, Rayleigh-only, tenfold Mie, ozone off, and black/white ground.
The high-altitude cases shift the double planet centre while retaining the
local camera and mesh. Execution records show actual PT or VBuffer/Deferred
passes followed by the same exposure, grading and FinalBlit chain.
Every HDR readback was finite and nonnegative within tolerance. Background
corner relative error was at most 2.67433 × 10⁻⁵; full-image relative RMSE
ranged from 0.1395 to 0.5961, retaining the expected foreground differences
between SH-based raster lighting and multi-bounce PT. Noon, sunset and twilight
were distinct; twilight was darker and ground albedo changed the output.
`final/PhysicalEnvironmentRendering.json` records final-run means and errors, with
RGBA32F and display PNG pairs for every case.

`physical_environment_source_and_history` switched each real path through
physical → disabled sources → HDRI → disabled sources → physical and then
halved TOA irradiance. Dark-background samples returned zero after both
switches; the constant HDRI returned `(32,16,8)`. Half-TOA background error was
zero for Deferred and 0.00020298 for PT. Measurements are in
`final/PhysicalEnvironmentHistory.json`. The Checkpoint A HDRI regression
was byte-identical for Deferred; the 32-frame, 256 × 256 PT comparison had
relative RMSE 3.5823231 × 10⁻⁵ and maximum absolute RGB error 0.0034214
(`production/HDRIRegression.json`).

The permanent NRD physical test is
`NRDWorkingColor.PhysicalAtmosphereHDRAndAerial`: four 32-frame, 96 × 96
cases execute RTXDI → RELAX → Composite and read the 32F composite/emissive,
16F noisy signals and depth. The final target-64 run passed and observed
physical Sun-disk RGB of 1.720073728 × 10⁹, finite noisy signals, a frame-wide
inverse scale of 51,188.8711 and HDRI inverse scale one. Physical and tenfold-Mie
composite maxima were 70,706.2578 and 70,033.7422, both above the FP16 limit.
On 2,141 non-emissive surface pixels, summed aerial guide energy changed
from 2,811.686 to 5,779.625 with tenfold Mie. Final values are in
`build/physical-environment-nrd/PhysicalNRDIntegration.json`; the focused
`MetallicNRDTests` run after actual-primary-origin changes is recorded in
`build/checkpoint-b-nrd-final.log` (one test passed).
`rtxdi_confidence_pre_exposure_domain` separately passed actual
diffuse/specular output checks for equivalent scale-one/scale-1024 dark
signals, scale changes with unchanged absolute lighting, and absolute HDR
history above 65,504. The legacy optional-input fallback and exact composite
passed `rtxdi_typed_post_process_pixels_history`.

The SDK-enabled `physical_environment_nrc_source_and_history` test passed
again with zero selecting Auto normalization and the actual-primary-origin
changes applied: one test passed and the process exited zero. It executes
actual native NRC stages and camera-segment composition. Dark background after
both source switches was zero; the constant
HDRI returned `(32,16,8)`; restored physical sky matched the original, and
half-TOA relative background error was zero. Values are in
`nrc-acceptance/PhysicalEnvironmentNRCHistory.json`, with the final run recorded
in `build/checkpoint-b-nrc-acceptance.log`.

NRC lifecycle acceptance is still incomplete. The latest acceptance process
reported 54 leaked Vulkan objects at preview/device destruction. The earlier
traced run in `build/checkpoint-b-nrc-final.log` reported the same leak count
even though `Context::Destroy` returned status zero and SDK shutdown completed.
Creating an NRC context on the subsequent device in that traced process did
not return normally and raised
SEH `0xc0000005` while accessing a stale buffer. The existing HDRI NRC
stage/history/cancellation test, run in a fresh process, also reproduced 112
leaked objects (`build/checkpoint-b-nrc-baseline.log`). This demonstrates a
broader SDK lifecycle gap in the current tree; it does not establish that
the gap predates Checkpoint B. The temporary tracing wrapper was removed and
the wrapper source matches HEAD. No speculative lifecycle fix is included.

DLSS's earlier focused filter ran
`dlss_float32_hdr_contract`, `dlss_float32_hdr_off_copy`,
`working_color_dlss_rr_history`, `working_color_dlss_sr_debug_bypass` and
`working_color_dlss_rr_debug_bypass`. The off-copy fixture preserved
`(1e9, 2e8, 0.125, 1)` byte-for-byte through both SR/RR mode-off paths. All
five assertions passed, but the teardown diagnostics remain an unresolved
stability issue and are not counted as clean SDK acceptance.

## Approximation boundaries

- This is a three-sample spectral atmosphere with an isotropic multiple-scattering
  closure, not a full spectral renderer. Aerial channel attenuation uses an
  equal-energy spectral normalization; arbitrary material spectra are approximated.
- The lighting capture is observer-centered. Secondary PT rays use this captured
  sky rather than integrating participating media independently at every bounce.
- FP16 sky/capture LUTs saturate at 65,504; primary celestial disks are evaluated
  separately in FP32. Extremely bright authored spectra can clip the diffuse sky
  LUT. Finite-disk direct-light BSDF integration remains a future replacement for
  the existing delta-light estimator.
- Raster diffuse IBL remains unoccluded nine-coefficient SH. It does not match
  multi-bounce PT on the foreground; the permanent A/B test quantifies the error
  and separately checks the common sky background.
- The NRD signal bound follows the current RTXDI celestial delta-light power
  lobe. Arbitrarily intense local lights or a different BSDF require a matching
  bound; the 32F emissive/sky path preserves its HDR range independently.
- Planet/observer subtraction is double precision at the atmosphere boundary.
  Scene geometry and existing camera storage remain float-based; this work does
  not introduce general scene origin rebasing, terrain or a planet mesh.
- Astronomy, weather, clouds and cloud shadows belong to Checkpoint C. Full-scene
  memory/performance measurements are outside this correctness checkpoint.
