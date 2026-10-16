# Shader cache warmup

For isolated shader compilation and compute pipeline creation measurements, use
the manual [ShaderPipelineBench harness](ShaderPipelineBench.md). It does not run
startup warmup or render frames and reports compilation and pipeline timings separately.

After configuring the project, run this target manually:

~~~powershell
cmake --build build-release --target MetallicShaderWarmup --config Release
~~~

Use your own build directory/configuration in place of the example. The target
builds the small MetallicShaderCompiler executable and warms the project's
.cache/shaders/spirv directory using the same SlangCompiler.cpp as the runtime.
It does not start Metallic, create a GPU device, load scenes, or build the editor.
Mapped-mode compilation requires no GPU. Native-mode compilation queries Vulkan
physical devices for descriptor strides, without creating a logical device or
submitting work. Supplying both strides explicitly also permits offline native
compilation.

The editor, LookDev, all rendering samples, and MetallicShaderPrintfProbe warm
the complete SPIR-V catalog synchronously before initializing rendering. After
that succeeds, creation of the actual rendering device automatically precreates
the supported compute Native PSOs before returning the device to its caller.
Warmup failure aborts startup with a nonzero exit code. To explicitly skip both
startup phases:

~~~powershell
.\build-release\Source\Metallic.exe --skip-shader-warmup
.\build-release\Source\LookDev.exe --skip-shader-warmup
.\build-release\Source\MetallicPathTracingSample.exe --skip-shader-warmup
~~~

The flag also works with smoke-test launches and the GPU printf probe. Help,
sample listing, offline cooking, CLI control tools, auxiliary launchers and test
executables do not automatically run this startup warmup.

The shared implementation runs inside each rendering process, after selecting
its shader debug mode and before GPU initialization. It uses the existing
parallel workers and progress display, then restores application logging.
When requests use native descriptor heaps, warmup queries a supported adapter's
resource/sampler stride pair before starting workers. Every request compiles once;
the pair is passed only to native requests. Listing requests and explicitly
skipping warmup do not query Vulkan.
There is no external compiler process to locate or deploy. Startup always uses
the complete catalog; METALLIC_SHADER_WARMUP_ARGS only configures the manual
target and cannot bypass or filter startup warmup.

The standalone compiler and manual warmup target remain outside default builds.
Rendering targets link the shared implementation; building them does not execute
warmup. Runtime compile-on-cache-miss remains available for unlisted variants,
source changes, and launches using --skip-shader-warmup.

Repeated invocations validate source bytes, search namespaces and compiler
identity before reusing current entries. Compilation, source changes during the
batch, or persistence/read-back failures reject the complete warmup and produce
a nonzero exit code. The standalone tool remains an offline SPIR-V compiler;
application startup adds the device phase described below.

## Native pipeline startup phase

This follows the separation used by
[Unreal PSO precaching](https://dev.epicgames.com/documentation/unreal-engine/pso-precaching-for-unreal-engine):
shader code and a complete pipeline description are different cache entries.
Compute permutations can be enumerated at startup; graphics PSOs require their
own render-state collectors. The current implementation waits synchronously
for the compute phase before making the rendering device available. It moves
driver work into startup; it does not claim to reduce total compilation work.

The device phase uses the real device, its enabled features and exact native
descriptor strides. It resolves validated shader-cache entries through
`ShaderRegistry`, constructs complete compute descriptions, and acquires each
PSO through the same registry path used by rendering. Layout size, indirect
binding policy and specialization tuples match runtime. In particular, source
deduplication does not discard the separate OpenPBR cache/guide, Standard path,
and Deferred precision/material-class PSOs.

The device retains execution owners from successful warmup until teardown.
This prevents the normal 128-object LRU from evicting a warmup object before
its first runtime acquisition, at the cost of keeping those native objects in
memory. The normal persistent pipeline-cache writer also receives the newly
created PSOs. A later exact request can reuse the live native object without
calling `vkCreateComputePipelines`; a binary-cache metadata hit alone would not
prove that. Driver/optimization overrides still apply through the registry.
Live-object reuse assumes the normal pipeline backend and automatic cache
selection. Diagnostic shader-object/pipeline-binary modes with an explicit
native-binary cache directory deliberately bypass the registry object cache;
prewarming those modes does not provide the same zero-creation guarantee.

Graphics shader stages are reported separately and are not given guessed PSO
state. Render target/depth formats, stage combinations and external ImGui ABIs
remain owned by their rendering paths. Unsupported compute device features and
unowned request layouts are also reported separately, rather than counted as
precreated PSOs. New/unlisted shaders and future runtime policy changes retain
the ordinary on-demand path.

`--skip-shader-warmup` bypasses both startup phases. Standalone compiler/list/help
and ordinary test launches do not enable automatic native warmup. Explicit
native warmup tests can supply a small request set without running the full
startup catalog. Changing the device, shaders or complete pipeline state never
reuses an incompatible native object.

Focused validation lives in `NativeShaderWarmupRequestsTests.cpp` (catalog ABI
and closed specialization coverage), `ShaderWarmupStartupTests.cpp` (startup
gating), and `NativeShaderWarmupTests.cpp` (actual Vulkan creation calls).
The RHI tests disable ordinary LRU retention before warmup, then require repeated
warmup and exact runtime acquisition to make zero additional native creation
calls. Changed push layouts and specialization values must remain distinct;
source/native failures must propagate and permit a later retry.

On 2026-10-16, a Release `pathtracing-sample` editor smoke run on RTX 5060 /
driver 616.92 completed the source catalog with 132 cache hits. The device
phase created and retained 109 compute PSOs, reused one duplicate request, and
skipped 35 graphics stages; unsupported and unknown-ABI counts were both zero.
It completed before `createDevice` returned. The subsequent graph compilation
and presented first frame issued zero compute pipeline creation calls.
A subsequent fresh-process control used the same sample/settings with
`--skip-shader-warmup`, retained the persistent caches, and presented its first
frame successfully; it issued 23 compute pipeline creation calls. Both runs
exited successfully without logged errors. The focused request suite passed
56 tests and the Vulkan validation run passed four native-cache/warmup tests.
This was a correctness run with existing persistent caches, not a cold-cache
benchmark: maximum native creation was 67.261 s, summed native creation was
697.797 s, and the complete native phase took 727.279 s. CPU shader tests
overlapped its early portion. These timings describe work moved into startup,
not an end-to-end speedup or a controlled before/after timing comparison.

## Parallel compilation and progress

Warmup uses a worker pool. The default is the available CPU thread count capped
at 12 to limit simultaneous Slang compiler memory use. The environment variable
`METALLIC_SHADER_WARMUP_JOBS` overrides this default for both application startup
and the standalone compiler. `--jobs N` takes precedence over the environment
variable; `--jobs 1` restores serial compilation. Values must be positive integers;
an unset or empty environment variable uses the default. Invalid values fail
warmup before compilation. Listing, help, and explicitly skipped startup warmup
do not require a valid worker setting. The actual worker count is capped by the
number of selected requests.

More workers increase simultaneous compiler memory use. Compare cold-cache
elapsed time and peak memory before increasing beyond the default; cache-hit
runs do not measure compilation throughput.

~~~powershell
.\build-release\Source\MetallicShaderCompiler.exe --jobs 8
# Also applies when starting a sample from this shell (or its IDE environment).
$env:METALLIC_SHADER_WARMUP_JOBS = '12'
.\build-release\Source\MetallicPathTracingSample.exe
Remove-Item Env:METALLIC_SHADER_WARMUP_JOBS
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS=--jobs;8"
cmake --build build-release --target MetallicShaderWarmup
~~~

CMake's --parallel option controls the build, not shader compiler workers.
Each worker reuses its thread-local Slang global session (the standard library).
Warmup captures immutable source files and search namespaces, then shares a
compilation session only between requests with exactly equal effective session
settings. Heavy programs retain separate root groups for parallelism. Session
reuse ends with the batch; ordinary runtime cache misses get fresh sessions.
SDK elapsed-time counters are sampled before and after each request, rather
than reporting accumulated time from earlier compilations. Groups run in a
static estimated-work order, independent of saved local timings.

After source and compiler validation, warmup writes V3 cache entries and decodes
the actual persisted files before registering the complete result. V3 identities
include the shared compiler archive's generated build fingerprint and actual
Slang DLL bytes, allowing the standalone tool and renderer to share entries.
Unqualified toolchain/link settings retain a host-specific identity. Older V2
entries become misses. Platforms without qualified snapshot monitoring compile
with fresh sessions without persistent cache reuse.

Progress reports completed requests/total, percentage, cache hits, failures,
elapsed time, and shader name in catalog order after the batch finishes. A
worker failure rejects the batch and can stop remaining work.
The progress percentage measures request count, not estimated remaining time.
Each completion also reports per-request total/setup/module/code-generation
seconds. The summary sums these overlapping worker intervals; those sums are
not complete warmup wall time.

### Complete cold-cache measurements

The production source-batch port was built in the existing Release tree and
measured in three independent processes with eight workers and separate empty
SPIR-V caches. O1, disabled shader debug, actual native stride queries (32/32
bytes), and all 174 requests were retained. Complete process walls were
**57.022 / 55.395 / 55.866 seconds**, with a **55.866-second median**. Every
run had zero existing hits and zero failures; all 522 final payloads were
byte-identical to the original catalog oracle. The source/build input seals
remained unchanged. This does not meet the ten-second target or establish a
production speedup over earlier measurements taken in different runs.

The batch used 77 session groups. Snapshot capture and boundary validation
reported 0.4 and 0.2 seconds; actual cache publication/readback reported
2.7–2.8 seconds. These rounded intervals are included in process wall time.
Summed code-generation work was 325–338 seconds across overlapping workers,
not end-to-end wall time. Peak working set was 6.20–6.21 GiB and peak commit
7.03–7.08 GiB. Evidence is in
`.cache/SlangCompileGoal/ProductionSourceWarmupValidation/Run-01/Result.json`.

The port passed 56 behavior checks and 14 startup checks, with one unavailable
optimizer skip. One large OpenPBR CaptureSymbols case was stopped after its
known long compilation and excluded from the remaining suite; its disabled-O1
compile and cache-hit portion completed, but the full debug case is unqualified.
The actual Streamline-enabled renderer submitted and presented one smoke-test
frame with startup warmup explicitly skipped. A separate, quiet three-request
FinalBlit check verified real tool-to-runtime V3 cache hits against distinct
host executables and byte-identical cache-disabled recompilation.

Before the ordinary-cache follow-up, runtime hits exposed substantial source-capture/validation cost:
the final three-request check took 0.794 / 0.757 / 0.558 seconds per `getShader`,
while cache lookup itself took about 1 millisecond and no compiler phase ran.
These are the earlier port's timings, not the current ordinary-cache path.

An ignored ordinary-cache experiment reused bounded immutable source epochs
while retaining fresh sessions on misses and complete source/producer validation.
All three arms passed the same twelve source/cache/registration checks (36
checks total). Three independent same-host, three-request hit comparisons took
2.100 / 1.965 / 3.747 seconds for the original path and 1.508 / 0.938 / 0.946
seconds for epoch reuse. Each cache hit performed no compilation; persisted
cache files and final payloads remained byte-identical, and input seals were
unchanged. Original-path variance was large. These are ordinary-hit process
measurements, not complete startup/cold-cache gains or a production adoption.
The one-pair single-end-only arm did not show a gain. Phase measurements still
showed substantial full SDK and source-byte validation cost. Evidence is in
`.cache/SlangCompileGoal/OrdinarySnapshotReuseCandidate/Run-01/`, `Run-03/` and
`Run-04/`.

A successor retained the initial FNV producer identity but saved sealed SDK/host
bytes and compared complete freshly read bytes at subsequent full validations.
Handle identity, size, timestamp, loaded-module paths and complete source
validation were retained. The same three-arm behavior suite passed 36 checks.
After discarding measurements that overlapped a preparation-script error,
three quiet, order-reversed comparisons took 1.854 / 2.042 / 1.859 seconds for
the original three-hit process and 0.811 / 0.804 / 0.778 seconds for epoch reuse
plus full byte comparison. All actual cache-hit, byte-identity and input-seal
gates passed. These are isolated ordinary-hit comparisons; they do not measure
complete cold warmup. Evidence is
in `.cache/SlangCompileGoal/OrdinaryBulkProducerCandidate/QuietRun-01/`,
`QuietRun-02/` and `QuietRun-03/`.

The follow-up is now integrated: ordinary requests reuse at most eight immutable
source epochs, keep fresh sessions on misses, and retain complete source and
producer-byte checks. Repeated hits use one complete return-boundary validation.
The actual Release Core/Identity/tool/renderer build passed; twelve focused
source/cache checks and fourteen startup checks passed. A distinct runtime host
hit three tool-generated FinalBlit entries in 544 / 224 / 218 milliseconds. The
last two calls reused their epoch with zero capture time; all outputs matched
fresh cache-disabled compilation and passed Vulkan scalar-layout validation.
The Streamline-enabled renderer also submitted and presented one smoke frame
with warmup skipped. Evidence is under
`.cache/SlangCompileGoal/RealShaderCompilerIdentityValidation/ActualBuild-05/`,
`RuntimeRegistryHandoff-07/`, `StartupSmoke-02/`, and
`.cache/SlangCompileGoal/ProductionWarmupSourceValidation/Checks-03/`.

A fresh complete cold-cache retest of this production build took
**59.785 / 57.621 / 64.751 seconds**, with a **59.785-second median**. The same
174 requests, eight workers, O1, disabled debug and actual native 32/32-byte
strides were retained. All 522 outputs matched the original oracle, with zero
hits/failures and unchanged 907-file input seals. The batch still used 77 groups;
capture was 0.4 seconds, boundary validation 0.1 seconds, and publication/readback
2.8 seconds. Summed code-generation work was 343–392 seconds across workers.
Peak working set was 6.25–6.29 GiB and peak commit 7.08–7.12 GiB. This retest
does not meet ten seconds or support a cold-compilation improvement. The earlier
55.866-second result remains historical; these separate runs do not establish
a causal regression. Evidence is in
`.cache/SlangCompileGoal/ProductionSourceWarmupValidation/Run-02/Result.json`.

A later five-file Deferred experiment introduced an evaluation-only prepared
closure and used a static trait to omit aggregate sampling-weight preparation.
After fixing an interface declaration, actual O1 compilation, finalization,
scalar-layout Vulkan validation and source/SDK identity checks passed for
requests 140 and 145. The first pair's code-generation times increased from
3.971 to 5.495 seconds and from 3.494 to 5.904 seconds; both final candidates
added 519 instructions. It was rejected without additional pairs, GPU runs or
production adoption. This is a rejected N1 direction, not a stable speedup
comparison. Evidence is in
`.cache/SlangCompileGoal/SharcFieldCurrentSourceRefresh/ProjectedReview-01/`.

A subsequent pure-forward diagnostic captured the actual pinned SDK optimizer
input for request 140. Original and candidate raw instruction counts were
25,771 and 25,726, with 441/439 functions and 1,196/1,194 calls. Raw composite
construct/extract/insert/copy counts were unchanged. Final output increased
from 175,493 to 176,012 instructions; final loads/stores increased by 141/214,
while final composite counts remained unchanged. Eight named lobe-estimate
call sites remained in each raw module; this is a static module count, not a
runtime execution count or proof that all calls are live.

The proxy forwarded the complete original ABI/arguments/raw bytes to the real
SDK and forwarded its sole output callback. Both outputs retained the earlier
qualified hashes and passed finalization, scalar-layout validation, source/SDK
and audit gates. SDK-reported downstream totals were 2.868 and 2.914 seconds.
Capture and profiling I/O were included, so this diagnostic does not reproduce
or supersede the N1 performance comparison. It supplies no per-pass timing and
does not justify deleting a default O1 pass. Evidence is in
`.cache/SlangCompileGoal/DeferredProjectedRawForwardExperiment/Run-01/`.

A one-file Deferred experiment narrowed the internal light provider's stored
push structure to its debug flags, retaining the original bit tests and public
push/resource ABI. Fresh current-source qualification passed for requests 140
and 145, including original-oracle checks, four external scalar-layout
validations and four complete module audits. Code-generation time increased
from 3.540 to 6.788 seconds and from 3.683 to 6.752 seconds. Each final candidate
removed only one instruction and sixteen bytes. This N1 candidate was rejected
without further timing pairs, GPU runs or live shader changes. Evidence is in
`.cache/SlangCompileGoal/DeferredDebugFlagsNarrowExperiment/N1-01/`.

An isolated backend built from the exact Slang 2026.18.2 wrapper and dependency
commits now measures the actual default O1 pass manager. Only pass execution
and object cleanup received buffered `chrono` diagnostics. Separate original
and instrumented DLL processes replayed the captured original request140 raw
input with its complete original ABI. Both outputs matched the original SDK
output byte-for-byte and passed external Vulkan scalar-layout validation; loaded
DLL/export identities, immutable input, sole output callback and artifact seals
passed. This admits a same-source diagnostic, not a production speedup.

The instrumented backend took 2.768 seconds: CCP execution took 0.471 seconds,
the two aggressive DCE passes 0.415 and 0.409 seconds, simplification 0.321
seconds, and dead-function elimination 0.294 seconds. Exhaustive inline took
0.098 seconds, about 3.5% of this backend interval. All fourteen actual pass
names and their order matched the pinned default branch. No conditional
CompactIds manager ran. These timings cover one raw Deferred input, not the
full frontend, longest SHARC request or complete cold warmup. Evidence is in
`.cache/SlangCompileGoal/PinnedBackendPassProfileExperiment/Replay/Run-01/`.

The longest production request, OpenPBR SHARC update index126, was subsequently
captured against the current production compiler configuration. Its 808-file
source graph and SDK/build identities remained unchanged. The actual original
ABI and raw input were forwarded without replacement or extra optimizer flags;
raw, SDK output and final output passed external scalar-layout validation. SDK
and final output retained the original `f604746b...` payload hash. The raw input
contained 41,033 instructions, 582 functions and 1,800 calls; output contained
415,205 instructions, one function and zero calls. The original SDK backend
interval was 26.546 seconds, with capture/profiling I/O included in the enclosing
34.862-second process. This is a qualified capture for diagnosis, not a speedup
or complete warmup measurement. Evidence is in
`.cache/SlangCompileGoal/SharcUpdateRawForwardExperiment/Run-01/`.

The captured SHARC input also passed independent original/instrumented backend
replay admission: both retained the exact `f604746b...` output, complete ABI and
immutable input, four external validation processes passed, and all artifact
seals remained unchanged. The same-source diagnostic backend took 22.480
seconds. CCP execution took 5.891 seconds, scalar replacement 4.256 seconds and
simplification 4.164 seconds, together about 64% of that backend interval.
Aggressive DCE took 2.255 and 1.701 seconds; dead-function elimination took
1.213 seconds. Exhaustive inline took only 0.336 seconds (1.49%). Actual default
fourteen-pass order was verified, with no conditional CompactIds invocation.
These diagnostic intervals do not establish an end-to-end speedup or justify
removing a default optimization pass. Evidence is in
`.cache/SlangCompileGoal/PinnedBackendPassProfileExperiment/Replay126/Run-01/`.

An isolated algorithm experiment skipped dead-function non-semantic dependency
traversal only when the module contained no `NonSemantic.*` import. The copied
optimizer archive replaced exactly one object; public symbol/link-map provenance
and ABI exports were checked, and baseline source/build/SDK artifacts stayed
unchanged. Both raw140 and raw126 candidate outputs matched their original
oracles and passed validation. One backend comparison took 2.903 to 2.795
seconds for140 but 21.853 to 22.861 seconds for126. This does not support a gain
for the longest request. The candidate is rejected without production adoption
or a stable-speedup claim. These are raw-backend N1 comparisons, not complete
cold warmup. Evidence is in
`.cache/SlangCompileGoal/DeadFunctionNonSemanticGuardExperiment/RunN1-01/`.
A separate valid upstream `NonSemantic.Test` dependent-tree fixture passed both
full default-O1 backends with identical output, retaining the original
non-semantic traversal path. This supplemental behavior check does not change
the rejected N1 timing decision; its evidence is in `NonSemanticFixture-01/`.

A separate combined constant-folding experiment checked algebraic rules before
constructing an unused operand-constant vector, and reserved constant-vector
capacity before collecting eligible operands. Constant-manager initialization,
folding rules, operand order and ID allocation were retained. The copied archive
replaced only `fold.cpp.obj`, with symbol/link-map/export and input-seal checks.
Raw126 and140 outputs matched their original oracles and passed validation.
The N1 backend comparisons were 22.896 to 23.068 seconds for126 and 2.830 to
2.774 seconds for140. There is no gain for the longest request, so this combined
candidate is rejected without further pairs or production adoption. Evidence is
in `.cache/SlangCompileGoal/SimplifyEmptyRulesGuardExperiment/RunN1-01/`.

An isolated CCP propagator experiment replaced redundant status-map lookups
with `try_emplace` and used the existing instruction-pointer def-use overload.
Copied-object provenance, ABI/default14 order and output bytes passed for126
and140. N1 backend intervals were 22.487 to 22.319 seconds for126 and 2.938 to
5.268 seconds for140. Several unmodified passes also varied, so these separate
processes do not attribute the timing difference to the patch. There is no
consistent backend gain; the candidate was rejected without production adoption.
Evidence is in `.cache/SlangCompileGoal/CCPPropagatorLookupExperiment/RunN1-01/`.

Subsequent environment inspection identified an Intel Core Ultra 5 245KF with
14 logical processors. Before relying on additional short-request comparisons,
record actual Windows CPU-set efficiency classes and child-process affinity,
and compare both arms on the same selected core. Earlier uncontrolled N1
observations remain historical; this measurement concern does not establish
that core scheduling caused their differences.

Actual CPU-set calibration then replayed the same unchanged baseline140 DLL,
raw input and complete ABI on two selected cores. Windows reported class1
logical0 for the selected P core and class0 logical2 for the selected E core;
actual child affinity masks were verified as `0x1` and `0x4`. Backend intervals
were 2.508 and 11.344 seconds (4.52x E/P). Outputs remained byte-identical,
external validation and artifact seals passed, and the parent's original
affinity was restored. This demonstrates sensitivity to selected-core conditions,
not a compiler optimization or proof of causality for historical N1 differences.
Future paired raw comparisons must record and retain the same core affinity.
Complete cold warmup still requires its own all174-request measurement. Evidence
is in `.cache/SlangCompileGoal/HybridCpuBaselineCalibration/Run-01/`.

Restricting the actual complete production warmup to all six returned P cores
did not improve startup. One independent cold process retained eight workers,
all174 requests, O1, disabled debug and actual32/32 native strides. Its verified
child mask was `0x3c03`; complete process wall was 71.728 seconds, with zero hits
or failures,174 byte-identical original outputs/readbacks and unchanged907-file
input seals. The parent mask was restored. This distinct affinity condition is
not a paired comparison against the earlier59.785-second default median. It is
rejected without further repetitions or production affinity changes. Short raw
algorithm comparisons still need controlled core affinity; full warmup must
retain its own end-to-end evidence. See
`.cache/SlangCompileGoal/ProductionSourceWarmupValidation/Run-POnly01/Result.json`
and `.cache/SlangCompileGoal/HybridCpuBaselineCalibration/FullWarmupPOnly-01/`.

An isolated same-source backend build enabled MSVC C++ LTO while retaining the
original shader O1 flow, ABI, floating-point options and allocator configuration.
The complete390-command multiset matched the baseline after removing only LTO
and physical input/output-path differences; actual `/GL`, final `/LTCG`, export
sets and source/build/production seals passed. The initial output-path parser
failure was retained and qualified separately without restarting the build.

Serial raw126/140 comparisons then verified the same P-core child mask `0x1`,
complete ABI, original fourteen-pass order, external validation and byte-identical
outputs. The N1 backend intervals were 29.509 to23.592 seconds for126 and 5.469
to5.923 seconds for140. This mixed result does not establish a stable or complete
warmup gain. Production integration remains unqualified. Evidence is in
`.cache/SlangCompileGoal/PinnedBackendLTOExperiment/Qualification-02/` and
`.cache/SlangCompileGoal/PinnedBackendLTORawQualification/RunN1-01/`.

The subsequent complete source pair retained all174 requests, eight workers,
the default all-core mask, O1, disabled debug and independent empty SPIR-V caches.
Original and LTO runtime copies took62.850 and70.298 seconds respectively, through
teardown. Each returned174 byte-identical original payloads with verified cache
readbacks, zero hits/failures and unchanged907-file input seals. Actual backend
module paths, direct export owners and distinct cache producer identities were
observed. Backend consumption is inferred from these observations and the
mandatory O1 path; this run did not trace backend calls. This N1 pair showed no
complete-warmup benefit, so LTO is not adopted. See
`.cache/SlangCompileGoal/PinnedBackendLTOFullSourceExperiment/RunN1-01/`.

An isolated bounded ID-to-definition lookup cache retains the authoritative
unordered map and its public iteration order. It mirrors definitions in a
geometrically grown pointer array, with map fallback for IDs at or above
4,194,304. The header layout change was built through a fresh complete optimizer
backend. Direct enabled/disabled semantic comparisons passed empty/growth/hole
cases, duplicate definitions, conditional clearing, high-ID fallback, const and
non-const lookups, and all public APIs on a valid module. Input seals remained
unchanged. These tests do not exercise the Slang frontend or establish a shader
compile, memory, GPU or complete-warmup improvement. Evidence is in
`.cache/SlangCompileGoal/DefUseDenseCacheExperiment/Build-02/Manifest.json` and
`.cache/SlangCompileGoal/DefUseDenseCacheExperiment/Checks-01/Summary.json`.

The cache then passed actual same-P-core raw126/140 comparisons with the original
ABI, fourteen passes, byte-identical outputs, external validation and unchanged
seals. N1 backend times were24.726 to23.428 seconds for126 and2.560 to2.539
seconds for140. For126, peak working set increased from1,071,501,312 to
1,109,798,912 bytes; maximum sampled private usage increased from1,290,608,640 to
1,361,408,000 bytes. These modest single-run differences do not establish stable
or complete-startup gains. The candidate remains isolated and is not adopted.
See `.cache/SlangCompileGoal/DefUseDenseCacheExperiment/RawQualification/N1-01/`.

An independent use-list candidate replaces the global ordered edge set with
per-definition buckets, retaining ordered sets of users and stable empty buckets
during callbacks. A fresh full backend build and six original-manager semantic
comparison groups passed, including callback insertion/deletion, rehash,
definition-address reuse and logical-edge equality with empty buckets. Same-P
raw126/140 N1 retained the original ABI, fourteen passes, byte-identical outputs,
external validation and all input seals. Times were33.598 to25.162 seconds for126
and2.650 to6.931 seconds for140. For126, peak working set increased by159.637 MiB
and maximum sampled private usage by195.749 MiB. These mixed results do not
justify adoption or establish a complete-startup gain. See
`.cache/SlangCompileGoal/UseBucketsExperiment/RawQualification/Run-01/`.

Three subsequent AB/BA/AB pairs per input retained the same P core and passed
all twelve SDK and sixteen validator processes, original bytes, ABI, fourteen
passes, pre/post input seals and affinity restoration. For126, the paired median
backend-wall ratio was0.985921 and whole-process CPU ratio0.978671; for140 they
were1.636097 and1.365269, with all three140 CPU ratios above1. Process CPU time
comes from `GetProcessTimes` over the entire child lifetime, not a backend/pass
interval. Wall includes time not accounted as process CPU; these observations
do not identify a particular scheduling, waiting or clock cause. The original
buckets are rejected without production adoption. See
`.cache/SlangCompileGoal/UseBucketsExperiment/PairedCpuQualification/Run-01/`.

A separate singleton successor stores zero/one user inline and promotes to a
stable ordered set only for a second distinct user, never demoting that set.
Its fresh full backend and eight original-manager semantic groups passed,
including promotion during callbacks, ordering/early-stop and sticky reuse.
Same-P raw126/140 N1 retained the original ABI, fourteen passes, original bytes,
external validation and all source/build/SDK seals. Backend-wall intervals were
45.494 to27.877 seconds and8.215 to7.216 seconds; whole-child CPU times were
20.656 to16.922 seconds and3.938 to3.250 seconds. These N1 CPU reductions are not
stable or complete-startup gains. For126, peak working set still increased by
97.367 MiB and sampled peak private usage by131.649 MiB. The candidate remains
isolated, with no production adoption. See
`.cache/SlangCompileGoal/UseBucketsSingletonExperiment/RawQualification/Run-01/`.

The singleton's subsequent complete cold source pair retained all174 requests,
eight workers, default all-core affinity, O1, disabled debug, actual32/32 native
strides and independent empty caches. Each arm passed174 original-byte payloads
and actual readbacks, zero hits/failures, complete907-file/source/SDK/tool/runtime
post-seals and loaded-module/export-owner/producer checks. Full process wall
was79.633 to96.321 seconds; whole-process CPU was368.625 to373.453 seconds.
Peak private usage increased496.63 MiB. This busy-environment N1 does not identify
a causal source of wall variation, but establishes no full-startup benefit; the
singleton is not adopted. The first preparation variable-name failure is retained,
and the independent second preparation passed. See
`.cache/SlangCompileGoal/UseBucketsSingletonFullSourceExperiment/RunN1-01/`.

Read-only system observations also record measurement conditions independently.
One5-second native snapshot during a pinned raw replay found every logical core
97.7--100% busy. A later5.006-second snapshot during a fresh backend build found
62.531 busy CPU-seconds (88.48%); aggregate counters matched bracketing
`GetSystemTimes` observations. Readable process deltas included Rider.Backend,
Unity and multiple compiler processes, but permission/field failures and exited
processes prevented full attribution. These timestamped snapshots neither
explain historical N1/N3 intervals nor change the complete-warmup wall criterion.
No process, affinity or system setting was changed. See
`.cache/SlangCompileGoal/LogicalCPUObservation/Run-01/` and
`.cache/SlangCompileGoal/LogicalCPUObservation/Process-Run-01/`.

The guarded scalar-projection candidate retains existing member loads and
memory operands, but rewires valid static extracts directly to those members
before recursive scalar replacement. A fresh backend build passed original
compile-option/multiplicity, export and source/production-seal checks. Six valid
straight-line fixtures passed restricted interpretation and external validation;
two invalid index/type fixtures were rejected before SDK invocation. Whole-value,
unknown, named and volatile cases retained the original path. These fixtures use
explicit scalar replacement at opt0, so they do not qualify the production O1
fourteen-pass flow or establish performance. The first interpreter-opcode and
debug-name placement failures are preserved separately. See
`.cache/SlangCompileGoal/ScalarProjectionExperiment/Checks-03/`.

Actual same-P raw126/140 then failed the unchanged original-byte gate. Both
supplementary comparisons, using the same pinned `spirv-opt --compact-ids` and
external validators, produced equal full compacted bytes and proved ID-only
differences for those outputs. Original failures remain recorded. Both had zero
multi-index nested projections (3739/680 eligible whole loads respectively),
with unchanged final instruction counts. Thus this candidate did not expose the
intended nested pruning on these real requests and is not adopted. Unqualified
backend intervals were26.975 to26.228 seconds and3.038 to8.605 seconds; they are
not a qualified gain. Full input post-seals and child-process closure passed.
See `.cache/SlangCompileGoal/ScalarProjectionExperiment/Closure-01/`.

The independent copy-chain successor lets component analysis look through pure,
same-type `OpCopyObject` chains without changing those instructions. Unknown,
whole-value, debug, malformed, empty and cyclic cases retain all components.
Its fresh backend passed original compile flags/multiplicity, exports and
source/production seals. Fourteen valid and three pre-SDK invalid fixtures then
passed, including branching and256-deep chains, whole Private storage and full
observable output. Input copy instruction words were checked separately. Three
earlier fixture failures are preserved: an eliminated local store did not cover
whole use, a decorated Function structure was invalid, and a vector did not
trigger recursive scalar replacement. These opt0 scalar-only fixtures do not
qualify production O1 or performance. See
`.cache/SlangCompileGoal/ScalarProjectionCopyChainExperiment/Checks-04/`.

Real same-P raw126/140 found1951/164 accepted copy-chain loads and5451/603
terminal extracts. The unchanged original-byte gate failed for both; equal full
compacted bytes again proved ID-only differences. Final instruction counts
remained415205/175493. Whole-process CPU increased16.516 to17.344 seconds and
2.672 to2.953 seconds, with no quiet-environment or stable-gain claim. Complete
input post-seals and process closure passed; this candidate is not adopted.
For126, the first-rejected-use histogram was dominated by53524 composite
constructs and6694 stores. This is an order-biased first-use count, not all uses
or pass-time attribution, and motivates a separately guarded construct analysis.
See `.cache/SlangCompileGoal/ScalarProjectionCopyChainExperiment/Closure-01/`.

A separately guarded constructor-tracing Revision02 followed Struct/Array
embedding paths through validated constructors, copies and extracts, committing
only complete safe component sets. Whole/debug/Phi/unknown uses and queue/path
budgets retain all components. Its fresh 282-step pinned backend build preserved
compile-command multiplicity, flags, exports and the 907 production inputs;
all 31 direct shapes passed (26 valid, five rejected before the SDK).
Same-P-core default-O1 raw replays for 126 and 140 both failed the original
byte oracle, with the replay driver's explicit mismatch diagnostic rather than
a crash. Fixed-tool compact-ID comparisons and external validation subsequently
proved full byte equality for both inputs, so only ID numbering differed.
One pair's complete child CPU fell from 18.453 to 14.422 seconds for 126 but
rose from 2.469 to 2.875 seconds for 140. These are CPU times, not startup walls;
the mixed single-pair result does not establish a stable gain. The 40,900
trimmed-load and 50,405 elided-member counters for 126 describe intermediate
usage sets, not final instruction reductions. All input seals remained unchanged,
all children exited and parent affinity was restored. No production adoption or
complete-startup qualification is claimed. Evidence is under
`.cache/SlangCompileGoal/CtorProjectionTraceExperiment/Closure-01/`.

The same revision subsequently completed three fresh AB/BA/AB raw-replay pairs
for each input, with the same actual ABI, default fourteen backend passes and
one-core affinity. Complete child CPU ratios for 126 were 0.787731, 0.776181
and 0.757157 (median 22.38% lower). For 140 they were 0.915033, 1.005917 and
0.931429 (median 6.86% lower, with one regression). Wall-ratio medians were
0.748152 and 1.027786 respectively; CPU and elapsed wall are separate scopes.
All six candidate executions retained the original hard byte-oracle failure.
The independent checks required actual SDK success with the precise mismatch
diagnostic, external validation and complete fixed-tool compacted byte equality.
They passed for every pair, establishing only ID-numbering differences.
All recorded children exited, input seals stayed unchanged and parent affinity
was restored. This raw-replay evidence does not qualify current-source shader
compilation, GPU behavior, production adoption or complete startup. Evidence is
`.cache/SlangCompileGoal/CtorProjectionTraceExperiment/PairedCpu-01/Summary.json`.

Current-source qualification then compiled the actual 126 and 140 requests with
the unchanged frontend, Core/provider, 808-file reference graph and real native
catalog query. Both original outputs reproduced their production oracle exactly.
Both candidate outputs passed mandatory SDK finalization/validation, standalone
validation and whole-byte equality after the same fixed ID compaction. Captured
raw inputs were identical across arms; complete nullable ABI values and input
span contents matched, comparing cross-process addresses only by nullness.
Each source output also matched the already admitted same-DLL/raw replay.
Source per-pass timing markers were unavailable through this capture transport;
default-pass evidence is the sealed build/policy, actual O1 ABI and bound replay,
not an observed source pass trace. The original tools' reference/index-only
rejections and the missing-marker observation failure remain preserved. The
successful independent admission reuses the completed 126 outputs and fresh
140 compilations without rewriting those failed attempts. It still establishes
no complete-startup gain or production adoption. Evidence is
`.cache/SlangCompileGoal/CtorProjectionSourceQualificationExperiment/Qualification-04/`.

The direct staged production CLI then completed its first complete 174-request
cold pair with the original and constructor-tracing backends. Walls were
64.478 and 64.602 seconds (+0.19%); whole-process CPU was 382.953 and 373.750
seconds (-2.40%). Both arms retained eight workers, disabled debug/O1, actual
native 32/32 strides, empty output caches, zero hits/failures and no IR input
cache. Original outputs matched all 174 oracle payloads. Candidate raw bytes
matched 32; all 174 candidate/reference payloads matched as complete bytes after
the same fixed ID compaction. All 348 raw validators, 348 compact operations
and 348 compact validators succeeded outside the timed CLI intervals. Source,
build, tool and runtime seals passed and all children exited. This qualifies
the complete output scope but shows no first-pair startup gain. Evidence is
`.cache/SlangCompileGoal/CtorProjectionFullSourceExperiment/Pair-01/`.

Subsequent BA/AB performance pairs use the same admitted runtimes and settings,
new empty caches and actual complete CLI processes. Every new cache payload
must match its same-arm N1 raw output in full bytes before reusing that exact
payload's prior external validation and compaction evidence. These repeats
report zero fresh external validator/compaction executions and explicitly
identify the reused checks. No validation is reused for changed bytes. Evidence
is under `.cache/SlangCompileGoal/CtorProjectionFullSourceRepeatExperiment/`.
Both repeats completed successfully: the BA pair took 70.488 versus 70.110
seconds, and the final AB pair 81.658 versus 65.099 seconds. Every new raw
payload matched its same-arm validated N1 bytes, with source/build/runtime
seals unchanged and all children exited. Across all three pairs, the median
candidate/reference wall ratio was 0.994640 (0.54% lower), ranging from
0.797218 to 1.001925; the direction was mixed. Whole-process CPU ratios were
0.975968, 0.961641 and 0.928834 (median 3.84% lower). All samples were retained.
The unpaired wall medians were 70.488 and 65.099 seconds; they are not the
paired speedup estimate. These results do not qualify a stable startup gain,
and the backend remains unadopted. Every candidate startup exceeded ten seconds.
The independent summary is
`.cache/SlangCompileGoal/CtorProjectionFullSourcePairAudit/Pairs-01.json`.

A separate successor stopped constructor tracing once every valid origin member
had already been observed, conservatively retaining all members. Its fresh
282-step backend build preserved the default O1 passes, flags, exports and
production inputs. All 31 direct fixtures passed, including the six partial
trimming cases, five invalid inputs, copies and budget fallbacks. Only the
full-member fixture took the new diagnostic path. These fixtures use explicit
O0 plus scalar replacement; they do not qualify full source O1 compilation.

Actual default-O1 raw replays for 126 and 140 passed external validation and
complete fixed-tool compacted equality with the original. A fresh same-core
parent arm also produced exactly the same complete raw output bytes as this
successor. Its single-order incremental CPU observations were nevertheless
5.20% and 7.28% higher than the parent. These whole-child CPU observations do
not establish a causal pass regression or stable performance result; they give
no reason to expand this successor to full startup. No source/GPU/full174
adoption is claimed. Evidence is under
`.cache/SlangCompileGoal/CtorProjectionSaturationGuardExperiment/RawN1-01/` and
`ParentArm-01/`.
The recorded trace states fell only from 105,785 to 105,782 for 126 and remained
4,293 for 140; trimming and elided-member counts were unchanged. The successor
was closed without a full-startup run. Its terminal seals are in
`.cache/SlangCompileGoal/CtorProjectionSaturationGuardExperiment/Closure-01/`.

An independent stock-backend experiment reused operand constants within one
default-rule fold attempt, after an unsuccessful constant fold and before its
algebraic fold. Literal operands, custom rule sets, nonidentity public calls and
early failures retain the original lookup path. The two-file/header change used
a fresh complete backend build with unchanged O1 passes, compile flags and SDK
exports. Both actual production raw inputs reproduced the original payloads
byte-for-byte, including IDs, and passed actual ABI/pass tracing, external
validation and final input seals. One pair's whole-child CPU ratios were
1.021891 for 126 and 0.952128 for 140; walls were 1.144508 and 1.012242. This
mixed N1 result does not qualify a stable gain or justify full-startup expansion.
The initial private `const bool` also removed the class's original implicit move
assignment, making this candidate inadmissible despite the SDK build passing.
The sealed candidate and measurement remain unchanged. A separate ordinary-bool
repair built successfully and passed 27 actual cases in 17 groups against the
stock backend. Complete module words, numeric bits, custom-rule/mapping behavior,
overflow handling and the public move contract matched. The probe evaluator was
corrected to recognize the stock backend's `OpConstantNull`; the earlier failed
attempts were retained. These are direct semantics checks, without a new raw
timing, full-startup or GPU qualification for the repair. Evidence is under
`.cache/SlangCompileGoal/FoldOperandReuseExperiment/RawN1-01/` and
`.cache/SlangCompileGoal/FoldOperandReuseRevision02/DirectProbe-03Qualification/`.

An independent wrapper experiment inserted simplification and interface-preserving
ADCE before exhaustive inline, retaining all fourteen original O1 passes in
their original order. A fresh complete backend build matched the stock 390
compile recipes and exports. Actual raw requests 126 and 140 executed the
expected sixteen passes with unchanged input ABI, and their outputs passed
external SPIR-V validation. One AB pair increased whole-child CPU from 15.891
to 19.969 seconds for 126 (+25.66%) and from 2.516 to 2.922 for 140 (+16.15%);
wall ratios were 1.781185 and 1.548394. Request 126 lost one final instruction;
140 retained the same instruction count. Both complete outputs differed even
after ID compaction, so semantic equivalence is not claimed. The original byte
gate failures remain recorded. All twenty children exited, input seals matched
and the runner restored its affinity. This candidate was rejected without
full-startup or GPU expansion. Evidence is under
`.cache/SlangCompileGoal/PreInlineCleanupExperiment/RawScreen-01/`.

A separate wrapper candidate inserted local single-block load/store elimination
followed by simplification before inline, retaining the original fourteen O1
passes without extra early ADCE. The fresh backend again matched the 390 stock
compile recipes and exports. Both selected raw inputs had zero load/store
MemoryAccess masks and no Volatile decorations; this restricted admission does
not qualify arbitrary memory-access semantics. One actual AB pair reduced
request 126 whole-child CPU from 16.422 to 14.703 seconds (-10.47%), but increased
140 from 2.469 to 2.859 (+15.82%); wall ratios were 0.699983 and 1.079180.
Request 126 lost 31 final instructions, while 140 retained its count. Exact
fourteen/sixteen-pass traces and fresh output validation passed, but both
compacted outputs differed from stock. All sixteen children exited and final
seals matched. This mixed screen does not establish a stable gain, semantic
equivalence, GPU performance or complete startup qualification; it was not
adopted. Evidence is under
`.cache/SlangCompileGoal/PreInlineLocalMemoryExperiment/RawScreenDecision-01/`.

A third independent wrapper inserted SSA rewrite then simplification before
inline, retaining all fourteen stock O1 passes. Its fresh 390-recipe build and
restricted raw-input admission passed. The first actual AB pair reduced
whole-child CPU from 17.031 to 14.297 seconds for request 126 (-16.06%) and
from 3.094 to 2.531 for 140 (-18.18%). Final instruction counts changed from
415,205 to 390,570 and from 175,493 to 160,883. Exact pass traces and external
output/compacted-output validation passed; both compacted outputs still differed
from stock. All sixteen children exited with unchanged seals and restored
affinity. These are promising single-pair backend measurements, without
semantic, GPU or complete-startup qualification. Two additional BA/AB pairs
failed to confirm that initial gain: request 126's three CPU ratios were
0.839450, 1.077003 and 0.982841 (median -1.72%); request 140's were 0.818182,
1.040541 and 1.124183 (median +4.05%). The respective wall-ratio medians were
0.993845 and 0.970019. All eight repeat payloads exactly matched their own
arm's previously validated complete output; their validation was reused with
zero fresh validator processes. All children exited and final input seals
matched. This inconsistent result was not expanded to full startup or adopted.
A previous installed-library SSA-only prefix on
General Guides had a three-pair API-wall median ratio of 1.060103; that negative
history remains recorded. It used a separate parse/serialize prefix and the
complete original SDK suffix, whereas this candidate uses pinned SSA plus
simplification within one backend manager. Evidence is under
`.cache/SlangCompileGoal/PreInlineSSARevision02/RawScreenDecision-01/`,
`.cache/SlangCompileGoal/PreInlineSSARevision02/RawRepeat-01/` and
`.cache/SlangCompileGoal/PreInlineSSAExperiment/GuidesPaired/`.

A separate backend experiment retained Slang's O1 setting and omitted only the
SPIRV-Tools exhaustive-inline pass. This explicitly changes the backend policy
to thirteen passes. One wrapper translation unit was rebuilt with the original
compile/link recipe against sealed stock archives; the optimizer libraries were
reused. Raw request 126 whole-child CPU fell from 14.984 to 0.563 seconds, and
140 from 2.938 to 0.203. Final modules retained 582/441 functions and 1,800/1,196
calls instead of one function and no calls. Full-module validation passed;
different optimized outputs are not semantic-equivalence evidence.

The retained PrivateToLocal pass is unsafe to generalize to Private globals in
multiply called helpers. Before extending the experiment, a fresh original
capture covered the current complete 174-request workload and all 907 production
inputs. Every raw module was Private-free, Logical/GLSL450 and had zero load/store
MemoryAccess masks. All 174 actual callbacks preserved the original nullable
O1 ABI, and all published payloads matched the original oracle. Two duplicate
output groups retained their mapping ambiguity. This is workload admission,
without a general-purpose policy or production adoption.

An independent direct-CLI AB pair then measured **69.186 seconds reference and
19.795 seconds candidate**, including planning, native32/32 discovery, all 174
cold compilations, V3 publication/readback and teardown. Both used eight workers,
unchanged production CLI/frontend bytes, zero hits/failures and fresh caches.
All 348 external final-module validations passed after the measured processes
exited. Ten candidate payloads matched stock; 164 differed. A separate
fourteen-worker candidate run took **18.660 seconds**, with all 174 payloads
exactly matching the validated candidate bytes (zero fresh validators, 174
reused validations). It has no same-setting reference comparison. Neither run
met ten seconds. The remaining eight-worker VisibilityBufferDeferred requests
accounted for 46.475 summed module-load seconds and 35.575 summed codegen
seconds; these overlap across workers and are not process-wall savings.

Two small external-input Vulkan probes passed complete bit-exact CPU-oracle
and cross-arm output checks. Their single-round 64-dispatch GPU medians were
0.165808 versus 0.166400 ms and 10.482544 versus 10.033216 ms. Pipeline creation
was measured separately, with uncontrolled driver disk-cache state. These
microprobes do not qualify production scene correctness or GPU performance;
later whole-scene checks rejected global adoption as recorded below. Evidence is under
`.cache/SlangCompileGoal/BackendDeferredInlineExperiment/RawScreenDecision-01/`,
`BackendDeferredInlineCurrentSourceCapture01/Run-01/`,
`BackendDeferredInlineFullSource01/Pair-01/`, `Jobs14-01/`, `TimingAnalysis-01/`,
and `BackendDeferredInlineGPUProbe01/Pair-02Review/` beneath the same goal directory.

The subsequent whole-runtime check used the unchanged native RHI executable,
fresh shader and pipeline caches, O1, normal driver optimization and all three
default material scenes. Two stock controls produced byte-identical HDR images
for OpenPBRPathTrace, OpenPBRDeferred and RTXCRChiang. With the thirteen-pass
backend, both OpenPBR images remained byte-identical, but RTXCRChiang failed the
unchanged absolute-plus-relative `2e-5` bound: maximum absolute error was
0.07189524, RMSE 0.000277436, and 1,644 pixels exceeded the bound. All processes
exited, original caches were restored and input seals matched. This rejects
global adoption; validation-on GPU timestamps are diagnostic and do not qualify
runtime performance. Evidence is in
`.cache/SlangCompileGoal/BackendDeferredInlineWholeRuntimeGPU01/`, including
`CandidateComparison.json` and `DiagnosticDetails01.json`.

A separate, unadopted source-template experiment compiled eight fresh generic
material families in-process and froze their class constants into 32 ordinary
SPIR-V outputs before cache publication. It retained the original 174 public
requests and macros, performed 150 actual frontend compilations, used no
preexisting template cache, and passed 174 actual V3 readbacks plus 174 fresh
external module validations. The 142 outputs outside the template scope were
byte-identical to their respective backend's original outputs. Under overlapping
GPU correctness work, the thirteen-pass backend took 14.293 seconds, while the
stock fourteen-pass backend took 68.939 seconds. These diagnostic runs do not
qualify GPU correctness, runtime performance or a stable startup improvement;
the stock-backend template route was not expanded. Evidence is under
`.cache/SlangCompileGoal/FrozenMaterialWarmupSource01/Full174Functional-01/` and
`Full174Original14Functional-01/`.

The class-4 GPU template probe did not establish correctness. Its two original
controls were byte-identical, while the candidate comparison failed. The flat
control was affected by an unqualified asynchronous HDR environment switch;
the binned capture also contained unexplained zero-alpha pixels. The fixed
tolerance was retained, the route was not adopted, and GPU timing qualification
was not attempted. These fixture and candidate uncertainties are preserved in
`DeferredMaterialSpecConstantGPU14001/Run-03/` under the same goal directory.

A cache-only candidate separately retained the production source compiler and
all fourteen original O1 backend passes. It qualified the common batch proof
once while keeping three complete source checks, per-file notification checks,
atomic writes and exact persisted readbacks. All 174 cold outputs were
byte-identical to the stock backend, and 174 fresh external validations passed.
The observed whole CLI took 47.243 seconds during concurrent GPU correctness
work; it is a functional diagnostic, not a controlled speedup or a ten-second
result. Evidence is in
`.cache/SlangCompileGoal/WarmupPreparedProofProductionPort01/Full174Functional-01/`.

The cache-only change was then enabled in production. The compatible Release
build of both the CLI and renderer passed, and four focused warmup tests passed
without skips, including real cold/all-hit/mixed compilation and failed final
proof qualification after two actual writes. A new 908-input production epoch
was sealed rather than inheriting the earlier 907-input admission. The actual
production CLI completed all 174 cold requests in **39.942 seconds** with fourteen
workers and then all 174 real cache hits in **4.006 seconds**. Both runs read
back all 174 V3 files and matched the same already validated stock14 payloads;
the 174 exact-byte validation results were reused. Original frontend/backend
bytes and shared compiler identity matched before and after. This single run
does not isolate a stable end-to-end speedup. The rebuilt renderer's actual
startup smoke also passed: its default eight-worker cold warmup logged
**57.460 seconds**, and the complete smoke process took 72.041 seconds. These
scopes are distinct, and none meets the ten-second cold-warmup goal. Production
evidence is in `Production-01/` and `ProductionSmoke-01/` under the same directory.

A private per-definition user-bucket backend retained all fourteen passes and
passed six focused behavior tests. Its current 908-input, complete174 cold run
read back every V3 file and matched every production SPIR-V payload byte for
byte, reusing 174 exact-payload validator results. It took 42.251 seconds and
324.078 CPU seconds; the separate production observation was 39.942 seconds and
344.984 CPU seconds. These are unpaired observations, so no speedup is
established and the SDK change has not been adopted. Evidence is in
`.cache/SlangCompileGoal/PerDefUseBucketExperiment01/Full174Functional-01/`.

A bounded sampler then collected 1,770 instruction-pointer snapshots of the
heaviest raw backend request using only its own child process. The mapped
fourteen-pass backend retained identical machine code and validated output.
Global user-tree equal-range lookup, ID lookup and tree insertion accounted
for 11.13%, 8.53% and 6.50% of the snapshots respectively. These are diagnostic
sample shares, not exact function CPU times or a startup measurement. All child
processes exited. Evidence is in
`.cache/SlangCompileGoal/OwnedRaw126RipSampling01/Result-01.md`.

An API probe confirmed that distinct supplied-source root variants can share
one imported module in a session when each root has its own virtual path.
Using one canonical physical path failed on the second variant. Both virtual
roots emitted validated SPIR-V with the expected integer results. An isolated
implementation retained the original request identity and recorded explicit
generated-source derivations. Its capacity-four grouping measured **18.232
seconds** for all174 requests, jobs14, O1/debug-off and a fresh process/cache,
including native32/32, publication/readback and teardown. All174 final payloads
were byte-identical to the earlier backend13 candidate and passed174 fresh
validators. This single observation remains above ten seconds and does not
establish a stable scheduling gain or production adoption. No generated source
is claimed to equal its physical input. Evidence is under
`.cache/SlangCompileGoal/RootLocalPermutationSource02/Full174-01/`.
An independent in-memory cache-codec experiment also retained exact bytes for
all 174 existing envelopes while preparing their common proof once. This is
neither source-proof qualification nor a measured full-warmup improvement.

The complete catalog also has21 exact root/configuration groups with multiple
distinct entrypoints. Slang supports compiling these together through
[`getTargetCode`](https://shader-slang.org/docs/compilation-api/), but the measured
baseline codegen costs put only3.411 worker-seconds in the smaller members of
those groups, against379.830 worker-seconds overall. This optimistic estimate
assumes those smaller costs disappear entirely; it is not a measured saving or
wall-time bound. It does not justify prioritizing multi-entry output changes
over the heavy single-entry requests. No multi-entry compilation was performed.
See `.cache/SlangCompileGoal/MultiEntryEligibility-01.json`.

Use the evidence runner to measure the complete compiler process, including
catalog preparation, native stride queries, cache read-back and teardown:

~~~powershell
python -B Tools/Perf/MeasureShaderWarmup.py `
  --executable build-release/Source/MetallicShaderCompiler.exe `
  --output-dir .cache/shader-warmup-measurement-new --jobs 8 --repeats 3
~~~

The output directory must be new. Every repetition starts an independent process
with a new, empty SPIR-V cache; no shader IR disk cache is introduced. The runner
preserves compiler settings and records the inherited environment, full catalog,
source/EXE/DLL SHA256 identities, raw logs, phases and Windows process memory
peaks. It rejects cache hits, incomplete catalogs, compilation/read-back failures
and changing inputs. OS filesystem caches are not flushed. Listing prepares
generated material source before freezing inputs; the timed processes still
perform their own complete catalog preparation.

The default acceptance threshold is strictly below 10 seconds for **every**
repetition. Valid measurements above the threshold have `status: passed` and
`goal_reached: false`; the runner returns 1 until the goal is reached. A failed
measurement also returns 1, with `status: failed` and a diagnostic in the report.
Shader/pipeline creation and GPU dispatch performance remain separate scopes.

The first 2026-10-13 Release/O1, debug-disabled, eight-worker iteration measured
107.9 seconds of internal warmup time for 202 requests. Standard-library reuse,
vertex-request sharing and early heavy-program scheduling measured 93.7 seconds
for 187 requests. Deferred field reads and shared OpenPBR sparse-training
requests then measured 81.2 seconds internally (81.478 seconds for the complete
process), for 186 requests. All three used empty SPIR-V caches and reported zero
existing hits/failures; these are single exploratory runs, not repeated timing
distributions or evidence that the 10-second goal has been achieved. Local logs
are under `.cache/SlangCompileGoal/` and remain outside source control.

After refining the scheduling priorities, three fresh-process measurements of
the same 186-request catalog took **82.388, 65.226 and 65.990 seconds** (median
65.990). Every run had zero existing hits/failures and unchanged input SHA256
identities. Peak working sets were 6.24-6.37 GiB, with 7.08-7.27 GiB peak private
commit. OS filesystem caches were allowed to remain warm; the range is retained
rather than dropping the slower run. The initial 107.9-second result is a single
exploratory internal-timer baseline, not a matched repeated comparison against
this complete-process distribution. The evidence runner correctly reports
`status: passed`, `goal_reached: false`; the 10-second target remains open.
The sealed input identities and raw logs are in
`.cache/SlangCompileGoal/round3-jobs8/summary.json`.

Sharing inactive display settings and background-only Deferred precision then
reduced the complete catalog to 174 requests. Three independent cold-cache
processes took **55.115, 54.657 and 52.884 seconds** (median 54.657), with zero
existing hits/failures and unchanged measured input identities. Peak working
sets were 6.35-6.37 GiB and peak commit was 7.25-7.31 GiB. Settings
remained Release/O1, debug disabled and eight workers. These later measurements
are not an interleaved comparison isolating the new request-sharing change;
the 10-second goal remains open. Evidence is in
`.cache/SlangCompileGoal/round4-jobs8/summary.json`.

The display factory preserves all 16 original combinations using eight requests.
It shares inactive attachment/white settings, including the exact `80.000000`
HDR identity case. Deferred background bins share FP16/FP32 requests because
material class zero shades only the environment. Duplicate class definitions,
other class spellings and explicit precision overrides keep their original
identity. All 16 original display and eight background combinations were
compiled independently and matched the canonical final SPIR-V byte for byte.
The 19 focused factory/cache checks and 14 rebuilt startup checks passed,
including actual ShaderRegistry warmup/runtime disk-cache hits.
The surviving 174 catalog identities also mapped to the previous 186-request
catalog, with all 174 complete final SPIR-V payloads byte-identical. The local
comparison is `.cache/SlangCompileGoal/round4-comparison/Comparison.json`.

An ignored backend experiment temporarily prevented SDK inlining at three
numeric OpenPBR function boundaries, ran the pinned SDK's unchanged O1 passes,
then cleared only the temporary control bits before returning SPIR-V. Three
additional alternating pairs reduced the single mapped General Guides shader's
compile time by a paired median 45.06%; this was not a complete warmup result.
The actual production request, candidate bytes and descriptor ABI were verified
in Opaque1, Mixed1 and Sky. HDR, eight consecutive frames and seven guide outputs
matched the original numerically exactly. However, three independent AB/BA/AB
pairs with Vulkan validation explicitly disabled increased GPU dispatch-scope
medians by 3.43%, 4.27% and 10.55% respectively. These are GPU timestamp intervals,
excluding compilation, native creation and CPU frame time. The policy was
rejected and did not change the production compiler or the full-warmup results
above. Local evidence is under
`.cache/SlangCompileGoal/TransientInliningBarrierExperiment/GuidesPaired/` and
`DispatchTiming/`. Two earlier launcher/configuration failures remain sealed;
only the successful final runner's six validation-off processes supply timings.

The guarded implementation subsequently matched the three measured instances by
full name and ID-independent numeric signature, retaining all other overloads.
One instrumented full-catalog pair took 68.628 versus 58.457 seconds for 174
requests with eight workers and zero hits/failures. The candidate transformed
46 requests; the other 128 had no target. Actual SDK/proxy paths, all accepted
callbacks, unchanged source identities, four-stage SPIR-V validation and all
174 reference payloads against production were verified. Both measured walls
include common capture I/O and SDK validation overhead, so this single pair's
14.82% reduction cannot be applied directly to the production measurements
above. It supplies CPU exploration, not GPU qualification or goal completion;
the original three-boundary policy remains rejected. The seal is in
`.cache/SlangCompileGoal/FullCatalogBackendExperiment/Round2/Full174N1/`.

Adding an explicit driver Inline hint to the three surviving functions also
passed exact-output checks, but its first independent validation-off GPU pair
increased dispatch intervals by 3.85%, 5.32% and 10.65% in Opaque1, Mixed1 and
Sky. The predeclared gate stopped this experiment after one pair; this is not a
replicated timing result. The hint variant was rejected, and no new CPU gain
was measured or inherited from the earlier variant. Evidence is in
`.cache/SlangCompileGoal/InlineDriverHintRenderingExperiment/GPU/`.

An additional ignored variant ran the same initial O1 barrier experiment, then
used the original SDK at optimization level zero with explicit exhaustive entry
inlining and dead-function elimination. The final shader has one function and
no calls or function-control hints. Three fresh-source AB/BA/AB pairs reduced
complete process wall by a paired median 13.26% (range 6.81-14.75%), including
both optimizer calls, capture/validation overhead, finalization and teardown.
All thirty stage validations passed; the three reference payloads matched the
production oracle and the three candidate payloads were identical to each other.
This measures the single mapped General Guides shader. Correctness runs in
Opaque1, Mixed1 and Sky matched all HDR, temporal and guide readbacks exactly,
but the first validation-off GPU pair increased dispatch intervals by 1.78%
and 9.84% in Opaque1 and Sky; Mixed1 stayed within 0.09%. The predeclared gate
stopped this variant after one pair, so these are not replicated GPU timings.
The variant was rejected; no compiler policy or full-startup result changed.
Evidence is in `.cache/SlangCompileGoal/LateInlineExperiment/FullPairs/` and
`.cache/SlangCompileGoal/LateInlineRenderingExperiment/GPU/`.

Another ignored variant added constant propagation, simplification and dead-code
cleanup after exhaustive inlining. An owned, validated input token removed a
duplicate input validation while retaining mandatory final validation and the
original SDK/options binding. Three fresh-source AB/BA/AB pairs reduced the
single mapped General Guides complete-process wall by a paired median 16.94%
(range 11.49-18.31%); all thirty stage validations passed. The three final
payloads were identical, with 199,172 instructions, one function and no calls.
Fresh GPU correctness runs nevertheless exceeded the unchanged numerical
tolerance in four of 24 temporal readbacks and the Mixed1 final HDR output.
The largest temporal absolute error was 0.0022181; all 21 guide readbacks were
bit-identical, and no readback contained NaN or infinity. The variant was
rejected before validation-off GPU timing. No full-startup gain or production
change is claimed. Evidence is in
`.cache/SlangCompileGoal/TailVerifiedInputExperiment/FullPairs/` and
`.cache/SlangCompileGoal/TailVerifiedInputRenderingExperiment/Run-01/`.

Source-only exploration also tested a resolver-specific thin-film feature
guard. Single pairs for SHARC update and ordinary Guides showed no useful
code-generation gain (20.513 to 20.397 seconds and 11.830 to 11.827 seconds).
Both final shaders gained 1,521 instructions. This candidate was not adopted
or advanced to GPU qualification.

A separate five-file source candidate removed the existing SHARC_UPDATE
whole-packet-read exceptions, using the already declared field views without
changing bindings, offsets or the SDK O1 policy. Three additional AB/BA/AB pairs
for the largest mapped update request took 30.001, 28.234 and 28.310 seconds with
the original source, versus 17.988, 19.879 and 18.362 seconds with narrow reads.
The paired complete-process ratio median was 0.648588 (35.14% lower, range
29.59-40.04%). These single-request walls include full catalog/native planning,
source capture/boundary checks, an extra SDK final validator, output readback
and teardown. Code-generation intervals in those three pairs fell from 21.359,
20.892 and 20.873 seconds to 10.911, 10.269 and 11.038 seconds, a paired median
48.92% reduction. An earlier single pair's extra final validation stayed near
3.62 seconds in both arms.
All reference outputs matched the production oracle, and all candidate outputs
were identical. Complete module audits, actual consumed SDK/import identity,
source/tool seals and both SDK and external SPIR-V validation passed. This is
single-request CPU qualification; complete startup remains unqualified. A fresh
reference/reference/candidate GPU run passed all four existing SHARC tests and
the fixed cross-process Off, NoQuery and seven-guide comparisons. The unchanged
Query tolerance was exceeded by 116 captures between original repeats and by
101 captures between original and candidate, so the raw Query comparison does
not establish equivalence. A supplemental single-active-training fixture is
being prepared; no source change has been adopted. CPU evidence is in
`.cache/SlangCompileGoal/SourceFeatureQualificationExperiment/SharcField-126-Pairs-01/`.
GPU evidence is in
`.cache/SlangCompileGoal/SourceSharcFieldRenderingExperiment/Run-01/`.

The subsequent admission rebuilt the current RHI test graph and a supplementary
GPU executable in a separate compatible Release validation tree, preserving the
production build settings and SDK. A fresh actual 174-request catalog/native
capture sealed 808 runtime inputs. The original and candidate request 126 both
passed the mandatory SDK finalizer, external SPIR-V validation, complete module
audits and source/tool seals, reproducing the expected `f604746b` and `b570e222`
payloads. This refresh establishes current-source identity, not an additional
stable performance gain. The supplemental GPU run uses 1,024 static training
frames and roughness 1.0; it requires actual specular and combined Query coverage
and two original runs within the existing absolute/relative `2e-5` raw-pixel
limits before running the candidate. Both original repeats and the candidate
passed all three actual diffuse/mixed/metal tests without skips. Each arm
consumed the target shader eight times per scene without rejection. The 15,900
color-frame and 105 guide comparisons were all byte-identical for both the
original repeat control and original/candidate pair, with maximum absolute
error zero. This qualifies the supplemental single-training-invocation scope;
the separate current-source standard four-scene capture has built and both
original arms passed all four actual tests. Their strict Off/NoQuery/guide
comparison had no required failures; 112 Query captures exceeded the fixed
`2e-5` limits between original repeats, so this standard scope does not establish
raw Query determinism. The candidate also passed all four actual tests and
had zero required comparison failures; 83 Query captures exceeded the same
limits versus the original, so standard raw Query equivalence remains
unqualified. The controlled supplemental Query comparison remains byte-exact.
The complete cold startup measurement is recorded below. Evidence is under
`.cache/SlangCompileGoal/SharcFieldCurrent808Admission/` and
`.cache/SlangCompileGoal/SourceSharcCoverageCurrentRevision03/`.

The five-file transform affects the two production `SHARC_UPDATE=1` requests:
125 (`ScenePathTrace.scenePathTraceMain`) and 126
(`OpenPBRRayQueryPathTrace.openPbrRayQueryPathTraceMain`). Request 125 now passed
an independent current-source pair with the original SDK finalizer and external
validation. Its original `cec5110f` payload changed to `4a311e0d`; final
instructions rose from 123,465 to 124,245, while bytes fell from 1,908,896 to
1,906,816 and ID bound fell from 648,219 to 212,770. This one identity pair took
19.408 versus 6.938 complete-process seconds; it is not a stable performance
claim or complete-startup result. Request 125 received separate GPU admission;
the OpenPBR results do not admit its changed payload. The first controlled original GPU run passed diffuse
and mixed fixtures but failed its inherited OpenPBR-only metal
`DiffuseOnly error == 0` assertion. Standard `ScenePathTrace` never reads the
lobe-mask parameter, so Query, DiffuseOnly and SpecularOnly controls were
identical; their metal relative RMSE was 0.1108323. This is a test-contract
failure in the original arm, not candidate behavior. No repeat or candidate arm
started. The run actually used material metallic factors 0/0/1 and roughness 1;
a separate revision uses true 0/0.5/1 material coverage and accurate
ignored-mask semantics, preserving complete strict raw comparisons and the
existing RadianceCache checks. Its first original run passed all three actual
tests with target consumption eight times per material and no rejection.
Query relative RMSE was 0.0989915, 0.0501396 and 0.1108323 for diffuse, mixed
and metal respectively; no isolated-lobe coverage is claimed. Its second
original run also passed, and all 15,900 five-mode color captures were
byte-identical between original repeats, with maximum error zero. The
candidate then passed all three actual tests, consumed the qualified `4a311e0d`
shader eight times per material without rejection, and matched all 15,900 raw
captures byte-for-byte. This qualifies the controlled single-trainer material
scope. All three original/repeat/candidate runs of the existing RadianceCache
standard test also passed, consuming the actual qualified shader without
rejection. Its additional 256 fresh linear preview frames were byte-identical
for both comparisons, including Query, with maximum error zero. These preview
frames are separate from the original test's tonemapped PNG captures; the
original accuracy/invalidation/cache-miss gates also passed. Its actual
production descriptor mode is mapped with
zero native strides. The full catalog's two native requests are editor entries
37 and 38, which continue to query actual 32/32-byte strides.

The exact five-file transform was then measured with the real production tool
over three complete cold-cache pairs, ordered AB/BA/AB. Reference walls were
57.906, 50.273 and 54.318 seconds; candidate walls were 96.426, 55.531 and
45.478 seconds. All six processes compiled all 174 requests at O1 with eight
workers, zero hits/failures, fresh SPIR-V output caches and no IR input cache.
Every reference payload matched the original oracle; candidates matched the
original oracle for 172 requests and the independently admitted payloads for
125 and 126. All 1,044 actual cache outputs/readbacks and external validation
results passed. The process wall includes planning, the actual native query,
source capture, compilation, publication/readback and teardown. The extra
external validators ran outside that timed process; OS file caches were not
flushed.

The reference median was 54.318 seconds and candidate median 55.531 seconds.
The median paired candidate/reference ratio was 1.104589 (10.46% slower),
with ratios ranging from 0.837253 to 1.665199. All samples are retained;
the mixed direction does not qualify a stable startup gain. The transform
was rejected and all five live shader files restored to their exact reference
bytes. The subsequent 333-file module audit passed. The complete cold startup
goal remains above ten seconds.

The measured comparison script initially rejected the fixed candidate-only
admission metadata. A separate audit reproduced that failure and accepted only
the exact admission path and SHA, retaining the other source/build/SDK/adapter,
payload, cache and validation gates. It changed no sealed measurement or live
source. Evidence is under
`.cache/SlangCompileGoal/SharcFieldFullCatalogPortProof/Pair01-A/` through
`Pair03-B/`, with the independent summary at
`.cache/SlangCompileGoal/SharcFieldFullCatalogPairAudit/Pairs-01.json`.

A separate ignored frontend experiment shared one session/root module for the
25 default mapped GPUDrivenStreamAsset entries, while independently composing,
linking and generating every entry. Three independent fresh-session processes
took 24.852, 25.102 and 24.367 seconds; the shared-module processes took 12.090,
12.113 and 12.108 seconds. All six produced 25/25 final payloads byte-identical
to the sealed production catalog, with no SPIR-V or disk IR input cache. Six
source edit, edit-and-restore and higher-priority shadow-module fixtures also
matched separate fresh-process oracles after discarding old pending outputs.
Full-process medians were 24.852 versus 12.108 seconds. Module-load medians fell
from 13.504 to 0.537 seconds; code generation stayed near 2.0 seconds. Conservative
full-namespace/source validation still cost about 9.0 seconds and is included in
those walls. This isolated 25-entry result is not a complete 174-request startup
measurement or a production session-reuse change. Local results and source seals
are in `.cache/SlangCompileGoal/WarmupModuleBatchExperiment/`.

The notification-based sibling kept immutable actual-read bytes and complete
source/namespace comparisons before and after publication, replacing per-entry
full scans with fail-closed overlapped directory notifications. The same source
and SDK identities passed all six change fixtures and 200/200 final payload
comparisons across eight measured processes. Its additional three pairs took
16.523, 16.958 and 16.551 seconds with fresh sessions, versus 3.360, 3.929 and
3.489 seconds with module sharing. The shared median includes 0.233 seconds of
capture, 0.319 seconds of complete end validation, 0.538 seconds of module load
and 2.008 seconds of code generation, plus comparison/output/teardown. This
validates an immutable snapshot; it does not establish atomic freshness of the
live filesystem at return, warm v2 namespace/shadow correctness, or full-catalog
startup performance. Production code remains unchanged. Source and result seals
are under `.cache/SlangCompileGoal/WarmupNotificationEpochExperiment/`.

The subsequent full-catalog prototype rebuilt the actual 174-request factory
catalog and queried the native 32/32-byte device strides in each process. Its
first eight-worker pair took 56.655 seconds with fresh sessions and 57.254
seconds with shared imports, keeping expensive root programs separate. Both
produced all 174 oracle-identical payloads, performed real experimental-cache
readback, and passed complete source/SDK identity checks. Sessions fell from
174 to 77, root loads from 174 to 114, and summed module-load intervals from
107.224 to 76.215 seconds; summed code generation increased from 312.712 to
333.337 seconds. These overlapping intervals are not startup timings. This
single pair establishes full-catalog correctness, but supplies no adopted
end-to-end speedup. Production session handling remains unchanged. Evidence is
in `.cache/SlangCompileGoal/FullWarmupSessionBatchExperiment/Run-01/`.

A subsequent variant scheduled these same groups by estimated cost from an
independent validated fresh-process report, retaining each group's entry order.
Three additional AB/BA/AB pairs took 55.849, 56.006 and 57.488 seconds with fresh
sessions, versus 52.041, 52.187 and 51.249 seconds with grouped historical-cost
scheduling. Complete-process medians were 56.006 and 52.041 seconds; the paired
ratio median was 0.931810, a 6.82% reduction. All 1044 payload comparisons,
experimental-cache readbacks, native queries and source/SDK/cost-input checks
passed. Actual group/entry timestamps were recorded independently of the cost
estimates. This remains an unintegrated prototype, and every measured run still
exceeded ten seconds. Evidence is in
`.cache/SlangCompileGoal/FullWarmupSessionBatchExperiment/Run-03/`.

A separate production-oriented frontend candidate extracted owned session
configuration and ordinary registry registration without changing production
source. Isolated checks matched original final payloads for six small-fixture
policies: mapped O0/O1/O2/O3, native explicit 32/32 strides and capture O1.
Real source edits with preserved size/mtime changed fresh compilation output;
restoration recovered the original bytes. Separate V3 cache/snapshot fixtures
passed real-file readback, namespace-shadow rejection, edit-and-restore
notification rejection and legacy V2 rejection. Those cache fixtures use a
known minimal SPIR-V payload; they do not establish the complete snapshot-to-
Slang-to-V3 production path. Full-catalog integration and static scheduling
remain unqualified. Evidence is in
`.cache/SlangCompileGoal/ProductionWarmupFrontendCandidate/Checks-01/`.

A subsequent isolated V3 batch candidate passed actual Slang cold cross-root
compilation, all-hit and mixed-hit batches, ordinary API reuse of a warmup cache,
and output changes after private-header edits with preserved size/mtime or a
higher-precedence module shadow. Accepted cache reads remain in the batch's
dependency proof, including all-hit batches. SDK dependency tokens are resolved
only through the current snapshot's issued identities and delivered files.
Unavailable ordinary cache persistence and source/cache overlap retain valid
compiled output without writing into the source namespace. All four focused
fixtures passed. One subsequent complete cold-cache pair took 60.377 seconds
with the original compiler and 59.010 seconds with the V3 batch candidate
(2.26% lower). Both independent processes completed all 174 requests with eight
workers, O1, disabled debug, actual native stride discovery and zero hits/failures.
All 174 final payloads matched each other and the previous production oracle;
all 348 persisted files passed independent checksum/entrypoint readback.
The candidate used 77 session groups, with 0.2 seconds of snapshot capture,
0.2 seconds of boundary validation and 5.4 seconds of actual publication/readback
included in its process wall. This is one pair, not a replicated speedup. Its
producer key currently includes the whole executable, preventing warmup-tool
cache reuse by a different renderer executable; production adoption remains
pending. No ten-second result is claimed.
Evidence is in
`.cache/SlangCompileGoal/ProductionWarmupV3BatchCandidate/Checks-02/` and
`.cache/SlangCompileGoal/ProductionWarmupV3BatchValidation/Run-01/`.

A successor cached hashes of immutable file images, read persisted files in
contiguous blocks, and separated the shared compiler build fingerprint from
the guarded host executable. Actual different-executable warmup-to-runtime
handoff hit without recompilation; changed build fingerprints and isolated SDK
byte changes missed. The original cache invalidation and corrupt/short/trailing
file checks also passed. Its complete 174-request cold pair took 57.712 versus
57.215 seconds (0.86% lower, one pair), with all final payloads identical.
Actual publication/readback was 4.2 seconds: proof encode/decode 1.7, atomic
writes 0.5, file read/decode 1.5 and full source validation 0.3 seconds, rounded.
These results do not demonstrate a stable startup gain. Production build-system
fingerprint generation and broader toolchain inputs remain unintegrated.
Evidence is in
`.cache/SlangCompileGoal/ProductionWarmupV3SharedCompilerCandidate/Checks-01/` and
`.cache/SlangCompileGoal/ProductionWarmupV3BatchValidation/Run-02/`.

A further successor retained one full decoder acceptance on every actual
persisted file, removing its redundant in-memory decoder pass and encoding
SPIR-V into one reserved buffer. All prior cache checks, a source edit during
the real cache writer, and cross-executable handoff checks passed. Its first
complete cold pair took 63.528 versus 55.834 seconds, with all 174 payloads
identical and zero hits/failures. Actual publication/readback took 3.0 seconds,
including 0.3 seconds of proof encoding, 0.5 of writes, 1.7 of file read/decode
and 0.3 of source validation, rounded. The slower reference in this single pair
was followed by two more independent BA/AB pairs. The three reference walls
were 63.528, 58.001 and 61.925 seconds; candidate walls were 55.834, 55.759 and
55.904 seconds. The paired ratio median was 0.902774 (9.72% lower, range
3.87-12.11%). All 522 per-arm final payload comparisons and 1,044 actual cache
file readbacks passed, with unchanged source/SDK/build inputs. This qualifies
the isolated frontend candidate for production integration, which is still
pending; every complete wall remains above ten seconds.
Evidence is in
`.cache/SlangCompileGoal/ProductionWarmupV3SingleReadbackCandidate/Checks-01/` and
`.cache/SlangCompileGoal/ProductionWarmupV3BatchValidation/Run-03/`.
The three-pair summary is
`.cache/SlangCompileGoal/ProductionWarmupV3BatchValidation/SingleReadback3PairSummary.json`.

One source-only Deferred experiment removed an unused material member from a
large helper result while preserving prepare/emission/direct-lighting order.
Four complete 333-file ownership audits and both original/candidate compilations
for actual requests 140 and 145 passed. Code generation nevertheless changed
from 3.298 to 3.357 seconds and 4.496 to 4.785 seconds (+1.77% and +6.43%).
The final shaders each lost two instructions, but aggregate-copy counts stayed
unchanged. The candidate was rejected without further GPU or full-catalog runs.
Evidence is in
`.cache/SlangCompileGoal/DeferredResultQualificationExperiment/N1-01/Decision.json`.

On the same machine (14 logical threads, 63.60 GiB RAM), an explicit
`--jobs 14` measured **48.913, 47.446 and 46.279 seconds** (median 47.446) for
the unchanged 174-request catalog. All three independent cold-cache processes
had zero hits/failures and unchanged input identities. Peak working sets were
9.06-9.19 GiB and peak commit was 10.20-10.39 GiB; available physical memory
before the experiment was 17.48 GiB. This separately measured worker setting
also remains above 10 seconds. The default cap stays at eight workers.
Evidence is `.cache/SlangCompileGoal/round4-jobs14/summary.json`.

The request sharing preserves runtime coverage: editor vertex code does not
consume fragment display macros, and OpenPBR `SHARC_UPDATE=1` Guides dispatches
the ordinary sparse-training function without writing guides. Only an
unambiguous single update definition is shared. The original Slang entrypoints
remain registered. The two former training binaries have identical executable
instructions after ID normalization, with only an `OpName` difference.

Release validation passed 49 shader/request tests (one unavailable-build-only
test skipped), 14 startup tests, 11 evidence-runner tests and the module audit.
The full shader suite was stopped at its large CaptureSymbols/O1 compilation
after exceeding the usual 300-second suite timeout; that path is not qualified
by this release-only performance measurement. Five Vulkan-validation GPU checks
passed: shadow history, async scene handoff, stream transmission, field views/
bounds and optional-resource sentinels. The raster/ray shading-normal and
ABeautifulGame FP16 checks failed with both the original and candidate shader
sources. Their three saved PNGs were byte-identical across the source A/B;
these pre-existing test failures were not reported as passes. The outputs were
also visually inspected. Full-scene temporal/memory behavior and an ordinary
paired GPU performance comparison remain outside this validation scope.

## Selection and shader debug modes

~~~powershell
# Build only the tool (does not warm any shaders).
cmake --build build-release --target MetallicShaderCompiler --config Release

# Inspect the request list without compiling.
.\build-release\Source\MetallicShaderCompiler.exe --list

# Warm only matching module/entry names.
.\build-release\Source\MetallicShaderCompiler.exe --filter FinalBlit

# Match the runtime shader debug policy when capturing/debugging shaders.
.\build-release\Source\MetallicShaderCompiler.exe --debug-mode capture
.\build-release\Source\MetallicShaderCompiler.exe --debug-mode debug
~~~

For multi-configuration generators the executable is under Source/<Config>/.
The default shader debug mode is disabled, independently of C++ Debug/Release.
capture corresponds to CaptureSymbols; debug corresponds to ShaderDebug.
Disabled and capture modes use Slang O1 with default backend optimization passes.
`METALLIC_SLANG_OPTIMIZATION=O0|O1|O2|O3` overrides that default for standalone
warmup and runtime requests. Unset it, or use `default`, to retain the existing
debug-mode policy. Set the same value before launching warmup and the renderer;
each effective optimization level has a separate shader-cache identity. Explicit
`ShaderRequest::optimizationLevel` values take precedence over the environment.
Without an optimization override, the explicit debug mode retains O0 for variable inspection. Runtime and warmup
share this policy and its versioned cache identity.
The tool also honors the runtime METALLIC_SLANG_DESCRIPTOR_MODE environment
variable. Cache entries for different modes are separate.
Native cache identity also includes the resource and sampler byte strides. The
compiler emits literal descriptor `ArrayStride` values instead of relying on
device-size specialization of the emitted SPIR-V. At runtime, device creation
publishes the selected adapter's actual strides before shader compilation.
This is the temporary `TO-REMOVE(VVL payload-size)` compiler policy; see
[the verified cause and removal conditions](NativeDescriptorHeapStrideWorkaround.md).

For offline native warmup, specify the two strides together. The resource stride
must match the runtime's common buffer/image descriptor slot stride; the sampler
stride must match its sampler descriptor stride. For example, if the target
adapter uses 64-byte resource slots and 16-byte sampler slots:

~~~powershell
$env:METALLIC_SLANG_DESCRIPTOR_MODE = "native"
.\build-release\Source\MetallicShaderCompiler.exe --resource-heap-stride 64 --sampler-heap-stride 16
~~~

Without these arguments, native warmup discovers a supported adapter automatically.
Failure to find a valid native stride pair aborts compilation. Explicit strides
produce cache entries for that ABI; another adapter can require different entries.

The stride query obtains Vulkan functions through its own local dispatch and
uses volk only for header declarations. Keep `MetallicShaderWarmupCore` free of
a volk implementation link dependency. Adding that dependency can move
`volk.lib` after `sl.interposer.lib` in application links on Windows: Streamline's
Vulkan function imports then replace volk's same-named global function pointers,
and `volkInitializeCustom` crashes while writing an import thunk in the code
section. This startup failure also affects mapped mode, even when every warmup
request is a cache hit and the native query never runs. Validate changes to this
dependency with a Streamline-enabled application startup; tests built with
Streamline disabled cannot cover the collision.

To pass arguments through the CMake target, configure with a semicolon-separated
list, for example:

~~~powershell
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS=--filter;FinalBlit"
cmake --build build-release --target MetallicShaderWarmup
# Clear the filter to restore the complete request list.
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS="
~~~

--cache-dir <path> redirects output for isolated validation. Such a directory
will not be used automatically by the application.

## Coverage and maintenance

Tools/ShaderWarmupRequests.h lists explicit runtime requests: editor display,
postprocessing, environment lighting, basic samples, GPU-driven/mesh/tessellation
rendering, ReSTIR, conventional-texture path tracing and guides, SHARC path
tracing variants, and the default global-view reference/realtime deferred paths.
The realtime deferred variants include upscaler guides and opaque/transmission
StreamAsset paths. RTXCR sample requests are added only when its SDK is configured.

This is a bounded warmup set, not every combination of runtime properties.
Optional NRD SDK permutations, local-view deferred variants, and other
unlisted combinations continue to compile on demand. No cache is required to run.

When adding or changing requests, copy the owning runtime pass's module, entry,
capabilities, additional search paths, and macro definitions **in the same order**.
The runtime cache key includes all of them, even explicit zero-valued defines.
For native requests it also includes both descriptor strides.
Compile include-file entry points through the same root module used by the pass.
The tool deliberately reuses runtime hashing, dependency validation, SPIR-V
processing, and cache serialization instead of duplicating that logic.
