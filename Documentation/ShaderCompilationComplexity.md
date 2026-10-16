# Shader compilation complexity (2026-10-16)

The latest worktree follow-up synchronizes Deferred precision selection as a
native Bool specialization, together with its request factory and CPU tests.
The three files are byte-identical to the isolated source used by the successful
six-target native build and 122 focused CPU cases. The actual complete list
retains 132 compiled requests and all 178 logical policies. One complete
fresh-cache CLI observation takes 12.964 seconds at O1 with default12 workers,
zero cache hits or failures, and 132 passing fresh whole-module validators.
This exceeds the ten-second goal. GPU output/performance acceptance remains
pending; the thirteen-case original-baseline/candidate/baseline comparison is
in progress. The 12.363-second complete observation below belongs to the
preceding `8d9e1fa4` source, not this new precision change.

The current source investigation removes repeated whole-scene packet expansion
from hot geometry and texture accessors, reusing their existing field reads.
Earlier permutation candidates share OpenPBR Off/Query and guide-policy
compilation with separate native specialization values, and unify Deferred's
covered builtin programs. Shared atmosphere resolve is held on a failed
rendering comparison. Preserve O1, the complete
workload, image behavior and GPU performance. At the user's request, the coherent
refactor and the two UPDATE environment field reads are now synchronized to the
repository worktree for review. This includes their C++/Vulkan contracts and tests.
The canonical Release editor and five affected targets build successfully, and
all 122 focused CPU tests pass in the selected ACEScg/Rec709 factory, startup,
native-cache and specialization cases. Module ownership passes for 334 files.
The actual complete list retains 132 compiled requests and all 178 logical
policies. One complete fresh shader-cache CLI observation takes 12.363 seconds
(12.045 seconds inside warmup), at O1, disabled debug and default12 workers,
with zero cache hits or failures. All 132 fresh whole-module validators pass;
130 full payloads equal the earlier fused132 outputs, and the two UPDATE
outputs equal their respective closed environment-field source candidates.
This observation exceeds the time goal. Background work remains running and
the OS file cache is not flushed. The narrow counterfactual GPU comparisons
below do not qualify the complete worktree stack or stable under-ten-second
performance; acceptance remains held on the
regressions below. The preceding synchronized refactor is recorded in commit
`8d9e1fa4` (`refactor shader`); the precision follow-up is a working-tree change.
Failed experiments, private GPU fixture overlays and
captures are excluded.
The latest private combination completes its fixed three fresh-cache CLI
observations in 9.342, 9.436 and 9.062 seconds for all 132 requests and 178
logical policies. The cold-time gate passes, but the completed twelve-case
direct-source A/C/A comparison shows GPU regressions, so the combination is
held. Whole-node GPU median differences are:

| Scope | C0 versus A0 | C0 versus A1 |
| --- | ---: | ---: |
| Deferred binned local-half | +13.20% | +7.49% |
| Deferred binned global-float | +15.38% | +15.23% |
| Deferred flat local-half | +4.08% | +4.16% |
| Standard Query | +2.98% | +1.30% |
| Standard guides Off | +3.22% | +2.98% |
| OpenPBR General guides Off | +1.20% | +1.33% |

Each case contributes 512 correlated frame samples in each of three processes,
with background work retained. The two local-half Deferred controls differ
5.31%; global-float controls differ 0.13%. Its shading section rises about
17.2-17.4%, while classification stays close to the controls. These are
observations from one candidate process, not a causal decomposition. Shader
policy sharing and runtime flags need separate counterfactuals. Adoption is
held and the overall goal stays active.
The isolated Deferred class counterfactual restores four static material-class
requests while keeping the current runtime view, guide and precision policies.
Its first complete fresh-cache CLI takes 11.609 seconds for 135 requests and
all 178 logical policies, at O1 and default12. There are zero cache hits or
failures; all 135 fresh whole-module validations pass. The 131 retained complete
descriptors and SPIR-V payloads agree byte for byte with the 132-request parent;
the four new static class modules have empty native specialization lists.
This single observation exceeds the time goal. Its separate four-case A/C/A
comparison is now closed. Relative to the two fused132 controls, static classes
change local binned node medians by +0.011/-0.153 percent and global binned node
medians by -0.170/-0.414 percent. The unchanged flat shader's controls themselves
differ by -3.340 percent. All fourteen full HDR/guide/visibility captures match
byte for byte in each of the three pairs, and the contact sheet was inspected.
This observation does not support static class recovery as a fix for the main
GPU regression. The class-static135 counterfactual is excluded from the worktree
sync; the shared class policies remain part of the unaccepted fused132 stack.
The separate precision counterfactual restores static FP0/FP1 while retaining
material-class sharing. Its private editor and five affected targets build, and
the same 122 focused CPU cases pass. The actual complete list contains 134
requests and retains all 178 logical policies. One fresh shader-cache CLI
observation takes 15.119 seconds end to end (14.688 seconds inside warmup),
with 87.594 seconds of aggregate process CPU, O1, disabled debug and default12
workers. There are zero cache hits or failures. All 134 fresh whole-module
validators pass; all 130 retained complete descriptors and SPIR-V payloads
match the current canonical parent, and the four new static-precision modules
are validated separately. This observation exceeds the time goal and is not a
paired cold-time comparison. Background work remains running and the OS file
cache is not flushed.
Its direct-source four-case A/C/A GPU comparison is now closed. Each process
retains all 512 timed frames per case. The Deferred node medians are:

| Precision counterfactual scope | A0 ms | C0 ms | A1 ms | C0 versus A0 | C0 versus A1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binned local-half | 0.901856 | 0.813376 | 0.899584 | -9.811% | -9.583% |
| Binned global-float | 0.865280 | 0.817104 | 0.870992 | -5.568% | -6.187% |
| Flat local-half | 0.393504 | 0.379616 | 0.394592 | -3.529% | -3.795% |

The corresponding local/global shading sections fall by 10.70-10.96 and
6.21-6.92 percent; material classification stays within 0.08 percent of both
controls. Flat-half is affected by this change and is not an unchanged control.
The fourth case checks visibility of all four material IDs and is not a shading
performance measurement. All fourteen complete HDR/guide/visibility captures
match byte for byte in every A/A and A/C pair; the contact sheet was inspected.
All 160 owned processes, including 154 fresh whole-module validators, close,
and each phase restores its original shader-cache state. Same-name section
records are summed within each frame; inclusive sections can overlap and are
not added to node times. These correlated frames from one candidate process
support runtime precision as one contributor to the regression. They do not
establish recovery of the complete earlier stack or full-scene GPU acceptance.
The 15.119-second static-precision counterfactual remains private and is excluded
from the synchronized source.
The next source-complexity investigation follows the mutable OpenPBR material
through the generic evaluation wrappers. After evaluation, path transport reads
only three floats from material fields totaling 656 logical scalar bytes. A
private four-module prototype carries only those three floats (12 logical scalar
bytes) and preserves the parameter-load, texture, random-number and IOR-clamp
order. Its actual builtin UPDATE125 O1 A/C/C/A source screen is now closed:
process CPU is 5.047/5.000/5.141/4.891 seconds, with means 4.969/5.070 seconds
(candidate/reference 1.02044). Instructions change from 170656 to 170544;
this does not show a useful compilation-cost reduction. Both complete controls
reproduce the actual current payload, and both candidates agree byte for byte.
Four fresh whole-module validations and the staged module audit pass, preserving
complete interfaces, capabilities and empty specialization lists. All nine
directly owned process records close; inputs and the four-owner source boundary
remain exact. The candidate is held without further requests, native/GPU or
full-warmup expansion, and is not included in the synchronized refactor.
One further source candidate changes only UPDATE `sceneMaterials()` to the
existing typed field getter for its four-byte handle. It preserves the complete
packet guards, CPU-checked offset, resolver and 720-byte material-record load.
Its UPDATE125 O1 A/C/C/A CPU observations are 6.656/5.766/5.578/6.172 seconds;
means are 6.414/5.672 seconds, an 11.57 percent reduction with 7.28 percent
baseline drift. A reverse C/A/A/C repeat observes 6.250/5.875/5.750/5.375 seconds:
both arm means are 5.8125 seconds. The initial benefit is not reproduced.
All four complete controls match the actual current payload, and all four
candidates share the same complete payload. Instructions change from 170656 to
170518, with one function and no calls. Eight fresh whole-module validations
and two staged ownership audits pass, retaining complete interfaces,
capabilities and empty specialization lists; all eighteen directly owned
process records close. This single-return candidate is also held, with no
native/GPU/full-warmup expansion or adoption. Neither source screen changes the
12.363-second complete-warmup observation above.
An independent precision-policy probe retains the current float/pack-unpack
math and replaces only its push-bit selector with a default-false native Bool
specialization constant. The actual current flat126 and binned128 descriptors
each compile at O1 with the original SDK; both fresh whole-module validators
and the staged ownership audit pass. Complete parent resource interfaces,
literal local sizes and capabilities remain exact, with no Float16, Int16 or
variable-pointer capability or native 16-bit scalar type. Actual declarations
show precision Bool ID0 for flat; binned retains fixed-metalness Int32 ID0 and
opaque-primary Bool ID1 and adds precision Bool ID2. Both emitted modules retain
the two pack/unpack pairs. These are observed IDs, not inferred ordering.
Independent IdRef, local store/load and entry-CFG review confirms their roles:
the false precision edge cannot reach either pack island, while the true edge
can reach both; opaque-primary controls its separate transmission-zeroing path.
This offline reachability check does not establish native driver folding.
All five directly owned source-probe process records close. The synchronized
request factory supplies every eligible precision value explicitly, sharing
the original eligibility predicate with native policy selection. It owns two
flat values and eight complete binned three-value tuples in static storage;
static/custom/background fallbacks retain their existing contracts. The 46
existing request-test names are retained with updated native policy assertions.
The editor, RHI tests, request tests, startup tests, native-cache tests and
shader compiler build successfully from an isolated current-source snapshot.
All 122 focused CPU cases pass: 46 request cases in each of ACEScg and Rec709,
14 startup, 13 native-cache and three specialization cases, with no skips.
The three RHI cases initialize devices but perform no scene dispatch. All nine
owned native-build/test Jobs close and the source/SDK/dependency closure checks
pass before synchronization. The shader, factory and test bytes copied into
the worktree exactly match that tested snapshot. These checks do not qualify
actual Deferred pipeline delivery, driver folding or GPU output/performance.
The subsequent actual complete list retains 132 compiled requests and all 178
logical policies. The complete fresh shader-cache CLI takes 12.963756 seconds
end to end (12.610 seconds inside warmup), with 79.250 seconds of aggregate
process CPU, O1, disabled debug and default12 workers. There are zero cache
hits or failures; all 132 fresh whole-module validators pass. The 130 unchanged
complete SPIR-V payloads match the parent, and the two precision payloads match
their closed source probes. All 133 owned process records close and the
compute lock is released. Background work remains running and the OS file
cache is not flushed. This single observation exceeds the time goal and is
not a paired compilation-speed comparison. GPU validation must cover both precision
values for flat and binned Deferred, including all eight class/precision tuples,
and compare against fresh controls from the original pre-fusion baseline.
A separate private Fiber prototype targets repeated direct-light BSDF expansion:
it places celestial, punctual and environment evaluation in a stage loop with
one authored evaluation site, preserving the existing sampling and shadow order.
Its UPDATE125 A/C/C/A source screen is prepared but has not run. Retained
evaluator sharing, compilation savings and Fiber rendering/GPU behavior remain
unverified; this prototype is excluded from the synchronized production source.
An isolated two-return source screen now changes only UPDATE environment-map
and environment-PDF access to their existing typed field getters. Standard
UPDATE A/C/C/A compiler CPU times are 3.438/3.078/3.078/3.328 seconds: mean CPU
falls from 3.383 to 3.078 seconds (9.0%). Both controls reproduce the complete
current payload and both candidates agree byte for byte. All four fresh
whole-module validations pass with the same complete interface, capabilities
and empty specialization lists. Instructions fall from 120423 to 119678.
This is a single-request source observation; it does not establish incremental
full-warmup or GPU effects. A separate OpenPBR UPDATE A/C/C/A screen observes CPU times
5.641/5.016/5.063/5.438 seconds: mean CPU falls from 5.539 to 5.039 seconds
(9.03 percent). Both complete controls reproduce their own current payload;
both candidates agree, and four fresh whole validators preserve full interface,
capabilities and empty specialization lists. This same two-return refactor is
included in the worktree sync. The complete canonical cold observation is
recorded above; its incremental full-warmup effect and scene/GPU behavior
remain unqualified.
In the earlier twelve-case comparison, each A/A or A/C pair compares 57 complete
captures: 55 agree byte for byte,
including all 45 auxiliary captures and ten final images. The two OpenPBR
Query final images differ in the controls as well; their raw errors are
retained rather than declared exact. Two fixed-exposure contact sheets were
inspected. The four actual Deferred native class tuples are observed; General
plain Off/Query tuples remain pending the deferred textured comparison.
The earlier three fixed complete fresh-cache CLI runs with only the two
resource-field source changes take 9.509, 9.344 and 10.133 seconds across all
131 requests. The previous repaired baseline takes 11.686 seconds. All three
complete compilations validate, but the third exceeds ten seconds: stable
under-ten-second performance was not established at that earlier stage.
The newer private combination keeps Standard PlainOff static and applies the
STF public sample-position composition. Its five-target native build and
119 CPU contracts pass. The actual complete list contains 132 requests and
retains all 178 logical policies. Its first fresh-cache complete CLI takes
9.342 seconds (9.008 inside warmup), with 70.234 seconds whole-process CPU,
zero cache hits/failures and all 132 fresh whole validations passing. Of the
131 retained descriptors, 123 produce identical full payloads and eight change;
the new static payload is e48d. All own interfaces, capabilities, local modes
and specialization contracts pass. Both Query outputs also reproduce the
earlier STF source-screen payloads. Parent metadata readsets all contain
SceneSurface, so that broad readership does not identify an unaffected set;
the 123 actual equal payloads are the byte-equivalence evidence. The fixed
second and third observations take 9.436 and 9.062 seconds; every one of their
132 payloads is byte-identical to the first run. All 396 fresh whole-module
validations and 399 owned children close successfully. The maximum CLI time
is 9.436 seconds and the median is 9.342 seconds. Whole-process CPU across the
three runs is 70.234/65.078/67.047 seconds, which does not demonstrate an
aggregate CPU reduction. These observations pass the fixed cold-time gate;
affected scene/GPU checks are incomplete and adoption is held on the
regressions above. Unity and existing
background work remain running, and the OS file cache is not flushed.
The earlier two-field UPDATE change now has five closed OpenPBR textured
artifact observations. All fifteen final HDR images agree byte for byte,
and all 85 owned processes and 75 fresh whole-module validators close. The
new consecutive A1/C0/C1/A2 comparison has whole-node candidate median
differences of +0.185/+0.314/+0.794 percent for Off/NoQuery/Query; the two
Query controls themselves span 0.752 percent. UPDATE section differences
are +0.919/+0.009 percent for NoQuery/Query. Historical A0 is retained but
excluded from that adjacent control comparison. The 64 frame values within
each mode are correlated. These limited observations preserve numerical
evidence and describe timing variation; they do not qualify the complete
fused source stack. Its separate twelve-case direct-source A/C/A comparison
has closed both controls and the candidate.
The positive textured direct-source tests are deferred while the observed
GPU regressions are investigated.
Standard's false/false HDR is exact
after the repair, while GPU performance and the other policies remain unqualified.
A new single-owner source screen removes whole-scene packet expansion from
the SHARC UPDATE geometry accessor by reusing its existing five field getters.
On the current repaired source graph, Update124 A,C,C,A CPU observations are
6.922/5.922/6.313/6.891 seconds: mean CPU time falls from 6.906 to 6.117 seconds
(11.4 percent). Both controls reproduce the complete current payload, both
candidates agree exactly, and all four complete modules pass fresh validation
with unchanged interfaces, capabilities, one function and no calls. This is a
positive standalone source screen. The second UPDATE request then observes
4.844/4.141/4.297/4.797 CPU seconds in A,C,C,A order, reducing mean CPU time
from 4.820 to 4.219 seconds (12.5 percent). Its controls match the complete
current payload, both candidates agree, and all four fresh whole validations
retain the complete interface and capabilities. A separate Standard121 A/C
pair reproduces the complete current payload byte for byte in both arms.
Runtime behavior and GPU performance remain to be checked. No source change
has been adopted. The 11.686-second result is the repaired baseline for the
subsequent field-projection cold run below.
An incremental screen over that geometry candidate changes only the UPDATE
returns of `sceneMaterialTexture` and `sceneTextureFeedback` to their existing
field expressions. It retains the full twelve-byte texture-index span, dynamic
nonuniform texture resolution and four-byte feedback handle, leaving sampling,
feedback atomics and caller order unchanged. Update124 A,C,C,A CPU observations
are 6.031/5.250/5.000/6.031 seconds: mean CPU time falls from 6.031 to 5.125
seconds (15.0 percent). Both controls reproduce the geometry candidate completely,
both new candidates match, and all four modules pass fresh validation with
unchanged interfaces and capabilities. Instructions decrease from 174069 to
172495, with one function and no calls. This is an incremental source screen;
the combined second UPDATE request then observes 4.078/3.375/3.406/4.359 CPU
seconds in A,C,C,A order, reducing mean CPU time from 4.219 to 3.391 seconds
(19.6 percent) over its geometry-only control. Its controls and candidates each
match their respective complete payloads; all four whole validations preserve
the interface and capabilities. Both separately compiled Standard121 arms
remain byte-exact to the current reference. Runtime/GPU checks remain pending.
These standalone reductions are not end-to-end startup savings.
The resulting two-owner private native build and all 119 CPU contracts pass.
Its first complete fresh-cache CLI takes 9.509 seconds, with 9.027 seconds
inside warmup, no cache hits and no worker flag or jobs-environment override.
All 131 fresh whole-module validations pass: the other 129 payloads exactly
match the repaired I12 baseline, and both UPDATE payloads exactly match their
closed source controls. Standard remains complete-byte-exact. O1, g0, postprocess
none and all 178 logical policy correspondences are retained. Whole-process CPU
is 66.625 seconds versus the baseline's 64.875; this run does not demonstrate an
aggregate CPU reduction or a stable end-to-end speedup. The two fixed additional
fresh-cache runs take 9.344 and 10.133 seconds, with whole-process CPU times of
66.922 and 63.656 seconds. All 131 validations and all 129 unchanged/two changed
payload checks pass in both repeats. Unity and existing background work remain
running; the shader caches are fresh, and the OS file cache is not flushed.
The median whole CLI time is 9.509 seconds and the maximum is 10.133 seconds.
These three retained observations do not establish stable under-ten-second
performance. Affected GPU/runtime checks remain outstanding; adoption and the
goal remain unqualified.
A fixed adjacent Standard artifact sequence A/A/C/C/A retains all five runs.
The three canonical PathTrace node medians are 12.503/12.528/12.490 ms;
the two repaired 9ed4 medians are 12.775/12.762 ms. All five complete HDR images
are byte-identical, all eleven non-target modules and cooked scene semantics
match, and every phase closes its source/cache/whole-validation gates. The
candidate process-median observation is 2.12 percent slower, and both candidate
medians exceed every control. This default-false artifact route remains held.
The old host supplies no native specialization values; production supplies
explicit false/false. Its actual production-path GPU comparison remains
separate, and this diagnostic does not establish the cause of the residual.
One actual production-path run now compiles the complete 9ed4 source through
fresh V3 and records recipe v2 with explicit Bool32 false/false. Its complete
HDR matches all three controls, as do the eleven non-target modules and cooked
scene semantics. The 512-frame node median is 12.736 ms, above all three
canonical controls; its shading median is 12.313 ms. GPU no-regression remains
unqualified. This comparison uses two native binaries, so it establishes an
actual production-path observation without isolating the cause.
The newly longest Query122/129 requests have been captured from the original
backend without changing their inputs: source and diagnostic replay both
reproduce the complete current cold payloads. Their new raw modules are
716512 and 710796 bytes. An instrumented replay shows distributed CCP,
aggressive dead-code elimination and simplification costs rather than a single
SROA-dominated pass. These rebuilt-instrumentation timings are diagnostic;
they are not production-release pass timings or startup savings. Source
complexity analysis now uses these exact Query raw modules instead of
extrapolating the UPDATE packet-expansion counts.
Both Query raw modules contain four typed 80-byte cache-parameter loads,
not the UPDATE path's repeated per-word whole-scene expansion. This is a weak
optimization lead and remains held. STF contributes roughly ten percent of
the body expansion proxy; its sampler-state copies and fixed magnification-NONE
configuration motivate a smaller public-API composition. A single-owner source
candidate retains the vendor sample-position calculation, addressing and load,
while bypassing generic magnification branches for the existing NONE policy.
Current Query122 A,C,C,A CPU observations are 5.516/5.391/5.266/5.328 seconds;
mean CPU decreases 1.73 percent, within the control drift. Query129 observes
5.438/5.250/5.188/5.453 seconds, a 4.16 percent mean decrease with both candidates
below both controls. All eight complete modules pass fresh validation with
unchanged interfaces, capabilities and their respective specialization shapes;
both controls reproduce current cold payloads and both candidates agree.
Each candidate removes 1726 final instructions and retains one function with
no calls. This is preliminary standalone source evidence, not stable CPU or
startup qualification. A production sampling probe uses two distinct textures,
four mips, UV boundaries, residency floors and RNG states across 67 cases, with
divergent texture indices inside 32-thread groups. Its original-SDK A/C source
pair now passes full validation with equal interfaces/capabilities and no
specialization; a faster passive DLL observer preserves both original output
blobs and resolves an earlier missing-observation failure without changing
shader code. Its separate RHI test host builds successfully, and the numerical
GPU test passes: all 536 output words match exactly across 67 cases, both
textures cover all four mips, seed advancement and early fallbacks agree,
and both validation counters remain zero. Two actual optimized compute
pipeline recipes contain the complete validated A/C modules with empty native
specialization. This bounded numerical check does not qualify scene timings
or complete cold startup. No STF change is adopted.
The two-field UPDATE GPU comparison completes its fixed Standard A/A/C/C/A
sequence. All three mode images match byte for byte across all five processes,
and texture feedback/refinement, numerical and input gates pass. NoQuery UPDATE
process medians are 0.081264/0.080832/0.080848/0.080832/0.080848 ms; Query UPDATE
medians are 0.080864/0.080688/0.080816/0.080752/0.080848 ms. The candidate median
of process medians differs by -0.01/-0.08 percent respectively. Whole-node
differences are within 0.24 percent, with control ranges of 0.41-0.65 percent.
This bounded textured Standard fixture shows no observed regression; it does
not qualify other scenes/policies, the STF change or the earlier Standard
source-union residual. The first control reuses its original capture through
explicit offline metadata recovery; its original failed controller receipts
remain unchanged. Each process contributes one timing observation, with
64 measured frames per mode; those frames are not independent repetitions.
A separate current-source Standard PlainOff diagnostic uses the canonical six
ordered base macros without specialization. Its one complete module passes
fresh validation with the canonical interface/capabilities, literal 8x8x1 and
no specialization records. It emits 2116896 bytes and 136622 instructions,
with one function and no calls, at 2.922 seconds whole-process CPU. The complete
26e0 payload differs from canonical e1d3. One subsequent same-host raw-artifact
GPU observation retains complete canonical HDR, all eleven non-target modules
and cooked scene semantics. Its 512-frame node median is 12.426 ms and shading
median 11.977 ms, versus historical canonical node controls of 12.490-12.528 ms.
Those first controls are not adjacent. A subsequent fixed adjacent A/C/A
sequence observes node medians of 12.533664/12.417152/12.533648 ms and shading
medians of 12.105344/11.990416/12.099632 ms. Complete HDR remains byte-identical
in all three phases, with the same eleven non-target modules and cooked scene
semantics. The candidate is 0.93 percent below the control process median;
this bounded fixture shows no observed regression. All 42 owned processes and
36 fresh whole-module validations close successfully. Unity remains running.
These GPU timings exclude compilation and do not measure the additional
request's cold-start cost. The native factory and STF combination remains a
private candidate awaiting numerical execution, native build and complete
132-request/178-logical-policy cold measurement. No startup gain is inferred.
The original 131-request source-union candidate fails the completed current-canonical
Standard PlainOff GPU comparison. Across 512 timed frames, the PathTrace node
median is 107.581 ms, versus 12.437/12.562 ms in the two canonical controls.
The canonical controls have byte-identical HDR output; the candidate exceeds
the fixed absolute/relative tolerance in 209 RGB components at 99 pixels.
The remaining eleven runtime modules and cooked geometry match, and all
processes, observers and private caches are closed/restored. This blocks
adoption despite the sub-ten-second cold compilation results. The proposed
default-worker change is held from adoption while shader function boundaries
and aggregate lifetimes are investigated; register pressure and spills are not
yet established.
An isolated four-variant Standard source screen reproduces the failed module
byte for byte, then changes only the alpha helper's inline annotation and/or
push argument. All four complete modules pass validation with the same resource
interface and capabilities. Original, wide-inline, narrow-noinline and
narrow-inline source CPU observations are 3.219/3.328/3.016/3.609 seconds.
Wide-inline emits one function and no calls (152482 instructions); the narrow
non-inline helper replaces the full push pointer with two scalar arguments
(129917 instructions, eight calls). These single source observations do not
establish whole-warmup savings or GPU correctness.
The subsequent same-host artifact comparison reproduces original A HDR exactly
at 12.501 ms, and the failed module reproduces original C HDR exactly at
107.636 ms using its false specialization defaults on the canonical host.
Removing only the alpha helper's noinline annotation restores byte-identical
original A HDR and reduces the node median to 12.804 ms. All eleven other
runtime modules remain byte-identical, and these phases close/restore fully.
This isolates that function boundary as a trigger for the large regression;
it does not identify a register/spill mechanism or qualify the remaining
timing difference, other policies, or full cold warmup after the
repair. Narrow-noinline retains the failed HDR exactly and takes 91.615 ms;
reducing the push argument alone does not repair the failure. A repeated
wide-inline run retains exact canonical HDR at 12.785 ms, with a later canonical
control at 12.287 ms. The observed residual difference is 2.4--4.2 percent and
remains unqualified while background work stays running.
Separate executable-statistics diagnostics report 255 registers and 12288
shared-memory bytes for both canonical and wide-inline modules; their driver
binaries are 1356160 and 1344896 bytes. The failed non-inline module has 168
registers, 7936 shared-memory bytes and a larger 2516224-byte binary. These
capture-flag diagnostics do not explain ordinary-frame timing. The unusually
large raw local-memory counter is retained without interpreting it as actual
allocation or spills; no internal representation was exported.
The independent two-owner candidate combines the alpha annotation repair with
a default cap of twelve workers. Its five-target native build and all 119 CPU
contracts pass. The complete fresh-cache CLI takes 11.686 seconds (11.370
inside warmup; 64.875 process CPU seconds), with no worker flag or environment
override. All 131 fresh whole-module validations, complete interfaces and
capabilities pass; 124 payloads equal the old 131-request candidate and seven
change. Standard matches the complete wide-inline source artifact. This
observation exceeds ten seconds. Its CPU cost is similar to the old explicit
twelve-worker result, so the wall-time difference alone does not establish
extra CPU work caused by the annotation repair. The earlier sub-ten-second
results below used the failed non-inline module and cannot qualify this repair.
A one-owner numeric-cache-context source experiment retains only numeric
leaves and three single-index engine handles across the Standard path loop,
constructing typed SHARC views inside the existing true query branch. The
fresh wide-inline control is byte-exact; both full modules validate with the
same interface, capabilities and two false Bool specialization constants. One
standalone source observation is 3.594/3.391 CPU seconds and
3.682/3.521 wall seconds, with 152482/152432 instructions. This is source-only
screening. Its subsequent same-host false/false artifact phase retains exact
canonical HDR at a 12.637-ms PathTrace node median; true-query cost and GPU
performance remain unqualified. A separate two-owner alpha coverage loader
reads the original five fields directly from the same typed material buffer,
avoiding a full 720-byte material value at the helper boundary. The complete
wide-inline control remains exact; both modules validate. One source pair
observes 3.672/3.438 CPU seconds and 3.694/3.481 wall seconds, while instructions
increase slightly from 152482 to 152517. Its same-host false/false phase also
retains exact canonical HDR, at 12.882 ms. Neither isolated source pair proves
a stable gain, and neither GPU timing qualifies nonregression. A subsequent
fresh canonical control retains exact HDR at 12.444 ms. The combined
numeric-context/coverage source comparison in A,C,C,A order observes CPU
3.594/3.375/3.359/3.469 seconds; both A payloads equal the full wide-inline
artifact and both C payloads agree exactly. C has 152467 instructions. Its
subsequent same-host GPU phase keeps exact HDR at 12.819 ms, so it remains
held on GPU performance and has no new full-cold acceptance.
Do not remove the coverage interpreter under `METALLIC_CUSTOM_MATERIALS=0`:
the CPU registry supports coverage-only programs with zero surface programs
(`MaterialCoverageTests.cpp`), while the request's custom flag follows surface
program presence. Such MASK materials still set the uploaded coverage-program
flag and execute the interpreter. A possible narrower investigation is sharing
only the interpreter call after that runtime check; its resource boundary,
positive coverage-program correctness and GPU cost would require fresh tests.
That narrower one-owner experiment has now completed: the original scalar-only
CoverageContext/MaterialInstanceRef cross a free VM helper boundary, with the
resource resolved inside it and the original texture/feedback order retained.
Both full modules validate with identical interfaces and capabilities, without
4441. Instructions decrease from 152482 to 149041, with two functions and eight
calls; one source CPU pair is 3.578/3.516 seconds. The ordinary opaque-scene GPU
comparison nevertheless fails the same 209-component HDR gate, at 98.707 ms
for PathTrace and 95.022 seconds of driver pipeline compilation. All processes
and caches close/restore. The executed positive-VM path was never qualified.
Further noinline candidates are held; neither this observation nor the earlier
alpha failure proves the compiler/driver mechanism. A current original-SDK
detailed profile reproduces the complete repaired Update124 and Standard121
payloads exactly, with fresh whole-module validation. Their instrumented source
children use 6.938/3.438 CPU seconds and 7.098/3.525 wall seconds. Inclusive
`generateOutput` counters cover 5.991/2.701 seconds; `linkAndOptimizeIR` covers
0.670/0.418 seconds. The nested counters are not additive, and the report does
not identify an `emitSPIRV` counter or establish the remaining output work as a
specific optimizer pass. One additional Update diagnostic adds only the
downstream timing flag and changes the output path. It again reproduces the
complete repaired payload and passes fresh validation. The compiler reports
5.05 seconds of downstream compilation; the child uses 6.844 CPU seconds and
7.022 wall seconds. This locates the dominant cost in the downstream aggregate,
without assigning it to an individual pass. A readonly inventory finds 1905 of
1922 Update switches have a constant selector and only a default edge; 414
nested wrapper pairs match the whole-resource getter's root/packet guards.
The final invalid branches store scalar/vector values, so this does not prove
that complete 384-byte packets are still copied. Single-exit getter control
flow and the interpreter's inner dispatch are separate candidates that preserve
the existing inline behavior. The interpreter experiment changes only its inner
operation 4--17 if-chain to a switch, retaining every expression, load order,
default and common clamp/store. Its Update A,C,C,A CPU observations are
6.953/6.656/6.766/6.484 seconds (mean C/A 0.999). Both candidate modules match
each other completely and pass fresh validation with unchanged interfaces and
capabilities; instructions decrease from 177679 to 176829. There is no CPU gain
signal, so this route is held without Standard, GPU or native expansion.
The separate whole-resource getter experiment retains its ForceInline attribute
and every guard/load, changing only its two-return body to an if/else assignment
and one return. Update A,C,C,A CPU observations are 6.922/7.031/6.984/6.844
seconds (mean C/A 1.018). Both candidate modules match completely and pass all
four fresh validations with identical interfaces, capabilities and empty
specialization records. Instructions fall from 177679 to 174092 and exactly 414
constant/default-only switches disappear. Function variables, loads and stores
each decrease by only one; all 49 local arrays retain their shapes and counts.
The control-flow mechanism works, but it provides no CPU gain signal. This
route is also held without broader getter edits, Standard, GPU or native runs.
A two-owner shadow-stack experiment then consumes only the two already-isolated
mutable shadow snapshots by `__ref`, retaining the public by-value wrapper for
other callers and the original body/math. Update A,C,C,A CPU observations are
6.891/7.063/6.906/6.891 seconds (mean C/A 1.014). All complete modules validate
with unchanged interfaces, capabilities, one function and zero calls; neither
4441 nor 4442 appears. The two redundant snapshot-to-callee copies disappear:
interior-stack arrays fall from nine to seven, whole-array loads from eight to
six and stores from nine to seven. Instructions fall by 192. This real data-copy
reduction also has no CPU gain signal, so the route is held at source screening.

Current pre-backend Update input is now captured through an existing
nontransforming forwarding proxy. Original and same-version diagnostic backend
source runs receive the same complete 1489824-byte raw module and both emit the
complete repaired payload, with fresh validation. The diagnostic CLI attempt
remains failed because its successful downstream callback's profiling text is
not emitted in CLI logs; it supplies no pass decomposition. Its raw capture is
retained for direct callback diagnostics and a current static call-graph review.
An independent existing ctypes worker now replays that same current raw input
directly through the admitted diagnostic DLL and retains the callback text. It
reproduces the complete repaired output, passes fresh validation, and confirms
the original fourteen-pass order with no conditional compact-ids run. In this
diagnostic build's 5.141-second SDK call, CCP uses 0.853 seconds plus 0.111 seconds
cleanup; the two aggressive DCE passes use 0.821/0.594 seconds, instruction
simplification 0.772 seconds, dead-function elimination 0.680 seconds, scalar
replacement 0.349 seconds and exhaustive inlining 0.144 seconds. These are
current-input diagnostic pass costs, not measured release-DLL pass costs or
startup savings. They replace the old raw module's SROA-heavy profile as the
working hypothesis: prioritize the volume of expanded code entering cleanup,
with source/whole-module and eventual GPU gates retained.
The newly screened alpha texcoord-only helper preserves the original geometry
guards, full triangle/attribute decoders and exact UV expression; all seven
committed full-hit paths stay unchanged. Its Update A,C,C,A CPU observations are
7.141/7.031/7.078/6.891 seconds (mean C/A 1.006), with instructions decreasing
from 177679 to 176875. All four complete modules validate with unchanged
interfaces/capabilities and empty specialization records. This supplies the
previously missing original-SDK source-CPU evidence, and holds UV-only on no
gain signal rather than relying solely on final DCE.

The current raw call graph instead identifies full packet expansion as a larger
candidate: streamRaySurface has eight whole Scene packet loads, each decoding
all 96 words. Packet address/load operations and dynamic constructions account
for 3792 of its 4352 direct-body instructions (87.1 percent), excluding root
checks and compute-packet calls. Only instances, header, pages, page table and
page count are extracted. These are pre-optimization static counts, not GPU
loads or compiler timing. The earlier five-owner field-read experiment measured
mixed complete-CLI wall scope without child CPU, before the current getter
ForceInline/alpha-inline source epoch. A single-owner, current-epoch field-read
screen is therefore being prepared; the older result remains recorded.
The latest default-eight complete private cold-cache observation is 10.648
seconds for 131 requests after the Standard three-policy source union (61.000
process CPU seconds; 10.347 seconds inside warmup). All 119 CPU contracts and
131 fresh whole-module validations pass. All 130 other complete payloads equal
the closed 132-request parent, and the new Standard payload equals its source
screen. The current canonical 178 logical requests map completely to these 131
sources through the recorded policy unions. CPU cost falls from 65.203 seconds,
but whole-process wall time is unchanged in this observation. The completed
GPU comparison fails as described above; the target has not been reached.

The best default-eight complete private cold-cache observation remains
10.633 seconds for 132 requests after adding the SINGLE Off/Query source union
(65.203 process CPU seconds; 10.396 seconds inside warmup). The five-target build
and all 119 CPU contracts pass. Every actual V3 key, new-source namespace and
source read passes; 131 complete payloads equal the closed 133-request parent,
and the SINGLE payload equals its independently validated Early02 source module.
All logical production policies remain covered. This is one observation and
still exceeds ten seconds; GPU behavior/performance and adoption remain pending.
The preceding complete private cold-cache candidate observation is 12.513 seconds
for 133 requests after combining OpenPBR's four policies and standard PathTrace
Off/Query (74.063 process CPU seconds; 12.224 seconds inside warmup). All 133
actual V3 keys, new namespaces, source reads and complete payloads pass: 131
payloads equal the closed 135-request parent, and two equal their independently
validated source candidates. Its five-target build and 111 CPU contracts pass.
The lower CPU cost has not reduced whole-process wall time in this observation.
The preceding best complete candidate observation was 12.164 seconds for 135
requests after the OpenPBR Off/Query source union (78.438 process CPU
seconds; 11.867 seconds reported inside warmup). All 135 actual V3 payloads pass
typed-key, new-source-namespace and source-read checks: 133 exactly match their
closed 137-request parent payloads, and two exactly match their independently
validated shared-source payloads. This is one candidate observation, without a
new paired control or GPU qualification. The preceding class-specialization
candidate measured 14.121 seconds for 137 requests, versus 14.865
seconds for 140 requests in the reverse C-first pair. The preceding same-binary
A-first pair is 19.460/13.354 seconds. Both directions favor the candidate but
the effect size varies greatly; no stable gain is established. The earlier candidate-only137-request
diagnostic is16.680 seconds. The earlier138-request
OpenPBR runtime guide-policy diagnostic is15.516 seconds. Neither meets the
target or demonstrates a startup improvement. Earlier reverse-order140-request
observations are14.497 seconds for the getter-only reference and14.598 seconds
with the punctual fallback tail merge. That earlier 14.497-second result is a
reference observation, not an attributed optimization gain. The initial larger
CDF difference does not repeat.
The target remains below 10 seconds. Smaller programs and individual source-child timings
are screening evidence, not startup acceptance.

An explicit 12-worker diagnostic on the same 133-request executable, source,
SDK and settings measures 10.594 seconds for the complete CLI (75.516 process
CPU seconds; 10.235 seconds inside warmup). All 133 typed V3 payloads equal
the closed default-eight run byte for byte, with the same actual producer,
new empty cache, unchanged namespace and source reads. Peak private usage is
7.103 GB. This separates a scheduling limit from compilation work: it does not
change the default worker policy, meet ten seconds or qualify GPU behavior.
Repeating this diagnostic on the 132-request SINGLE candidate gives 10.377
seconds for the full CLI (10.095 seconds inside warmup). All 132 complete
payloads also equal its default-eight result, with the same actual producer.
The remaining gap persists at twelve workers; Standard's separate heavy
ordinary/guide source programs remain a source-complexity candidate.
After the Standard union, the same explicit 12-worker diagnostic measures 9.748
seconds for the complete 131-request CLI (9.441 seconds inside warmup; 65.391
process CPU seconds). All 131 fresh whole-module validations pass and every
complete payload and actual producer equals the default-eight result. Peak
private usage is 6.625 GB. This is the first sub-ten-second full-cold observation;
the default worker policy is still eight. Two separately released fresh-cache
repetitions measure 9.704 and 9.646 seconds; all 393 whole-module validations
across the three 12-worker runs pass, with complete payloads and producer equal
to the default-eight run. Their median is 9.704 seconds. GPU nonregression
fails in the completed Standard PlainOff comparison; the proposed
default-worker change and adoption are held.

The Standard three-policy source screen now passes both complete parent controls
and a shared ordinary/guide module: 130022 instructions, 22 resource variables,
unchanged capabilities and fresh whole-module validation. Serial parent source
CPU is 2.906+2.938 seconds; the union takes 3.031 seconds. These are source-child
costs, not startup savings. Actual Bool IDs are cache0 and guide1; the additional
SpecConstantOp is LogicalNot of guide1. Native integration and the complete
131-request cold run now pass the compilation gates above; the completed
Standard PlainOff rendering/performance comparison fails.

The 132-request candidate's reference GPU sanity attempt reaches its 600-second
limit during native pipeline creation. Two successful first creations took
about 297 and 263 seconds with optimization enabled. The Job and observer close,
and the private source-cache and directory namespace are restored. It supplies
no completed A/A/C image or GPU-performance qualification. Its successful
private driver-cache bytes are retained for a separately admitted recovery.

A fresh snapshot of the current checkout now builds the Release RHI executable
and compiler CLI through their normal targets with the original SDK and matching
candidate settings. The actual canonical CLI lists 178 requests. Its source,
generated provider and executables are separately recorded; this establishes a
current baseline for further comparisons, not rendering or GPU qualification.

The newer Off/Query private native candidate builds all five selected targets
and passes 103 CPU contracts. Its actual CLI lists 135 source requests, covering
all 137 parent requests through exactly two Off/Query aliases. The complete
cold135 run passes the functional gates above. A separate five-owner source candidate also compiles
all four ordinary/Guides and Off/Query policies into one SPIR-V module. Its
187712 instructions are 1093 above its Query128 parent, with the full resource
interface equal to both actual Query parents. Independent final-module dataflow
and branch-pruned CFG checks identify cache-query as Bool SpecId0 and guide
export as Bool SpecId1 for this exact payload. The private 133-request native
integration passes the build, CPU and complete cold-payload checks above; GPU
behavior remains unqualified. The source-only SINGLE Off/Query union also passes
compilation and the strict Query interface gate, but increases instructions
from 188975 to 214558 and source CPU from 5.016 to 5.438 seconds versus Query
alone. Its longer individual job is a risk despite avoiding another compile.
An independent revision removes the duplicate authored BSDF call sites by
using the existing Query early-sampling schedule for both SINGLE cache policies.
It passes a new A138/C source screen with 189059 instructions (only 84 above
Query138), versus 214558 in the first union. Fresh source CPU is 4.828/5.125
seconds for A/C; this does not establish a timing gain. Off's changed schedule
still needs numerical, temporal and GPU-performance validation. This revision
is separate from the next combined ordinary/Guides and standard-PT candidate.
Unity and the existing background workload remain running.

Current source investigations capture immutable resource handles at evaluation
boundaries and examine an OpenPBR plain/guides program union. Capture reduces
the final program but has not demonstrated a stable startup gain. The union is
a shader/CPU integration candidate. Its Off and Query source screens, private
native build,64 CPU contracts and complete cold138 functional checks pass.
Broader GPU qualification is held while there is no demonstrated startup gain.
Environment mip-quad query reuse is also held after mixed forward/reverse
source timings. Reflection geometry reuse has a smaller positive reverse
source observation with more final instructions and remains held. Remaining
feature-call constants have a weak first source result and remain held. The standalone Off/Query
prototype is rejected by the pinned SDK's context-initialization check.
The UPDATE prepared-state schedule screen is also held on weak source results.
The next larger lead is Deferred's four binned material-class programs: actual
compiled pairs differ only in one stored policy literal. A specialization-constant
design is being checked to share Slang code while retaining four native PSOs.
The first real source screen reproduces the complete class-4 reference and
passes both whole-module validators, with the intended signed-int and boolean
specialization IDs/defaults. It is rejected at the exact interface gate: all27
descriptor-set decorations move from set0 to set1, while the30 resource types,
entry/layout and capabilities otherwise match. A separate declaration/layout
draft must preserve the original interface before native integration. No
resource remap, full-startup saving or GPU qualification is inferred.

This closed screen is retained in
`ShaderComplexityGovernance01/DeferredMaterialClassSpecialization01/SourceScreen13501`.
The reference/candidate source-child CPU observations are3.703/3.734 seconds,
with158273/158278 instructions and complete payload hashes31deb66e/94ba32ae.
Both compilers, both validators and the ownership audit close;875 input pins
and both813-file graphs recheck exactly. These are individual source observations,
not a four-program or complete-startup comparison. The original SDK, O1,
background workload and canonical shader sources remain unchanged.

The separate auto-ID revision02 now passes that exact interface/capability gate
against all four class parents. Its actual signed-int/bool IDs are0/1, defaults
(-1,false), with literal LocalSize32x1x1. The whole candidate is0c0f5bce,
2529244 bytes and158278 instructions. A again reproduces31deb66e; source-child
CPU observations are3.547/3.625 seconds. Both whole validators, the ownership
audit,877 input pins and both813-file graphs pass, with all five children closed.
Only the two declaration attributes changed from the rejected explicit-ID
prototype. The next step is private native integration with full value-aware
pipeline/artifact/recipe/binary identities. The expected140-to137 source-catalog
reduction, four native pipelines, actual runtime link path, rendering, GPU
performance and complete-startup saving remain unqualified.

The generic compute-specialization and three-owner factory drafts also pass
root/independent source review: owned canonical scalar32 values participate in
material artifacts, full native-key equality, recipe/replay and binary
compatibility identity, with borrowed native data kept alive for synchronous
creation. Empty-value paths preserve their old identities. A private coherent
native build and tests are being prepared; source review is not C++ typechecking
or backend execution. No canonical runtime/shader adoption has occurred.

The first private native attempt builds all four selected targets and passes34
factory tests in each independent ACEScg/Rec709 process. It then correctly fails
the positive-test gate: the selected RHI executable contains none of the
NativePipelineCache tests, which belong to a separate executable. No RHI CPU
test ran and Startup tests were not reached. All seven owned jobs close; root
rechecks18307 staged files and1176 read-only inputs unchanged. A separate test-only
registration/target repair is being prepared, preserving this failed record;
native specialization dispatch, rendering and startup performance are pending.

The test-only repair now passes:13 standalone native-cache tests,3 specialization
CPU tests and14 startup tests, with zero skips. The earlier34+34 factory results
are inherited from their successful independent processes; the original overall
failed build record is preserved. All six recovery jobs close naturally, including
the build's resident MSVC child. No process termination or explicit GPU dispatch
was used. The repaired private stage has18308 files and exactly three test-file
changes; its20 production deltas are unchanged.

An independent offline comparison also substitutes each actual specialization
tuple into the shared module and reproduces all four original instruction
streams after scalar/ID normalization:157891 retained instructions and472862
operands per policy. Wrong-policy and descriptor-set negative controls fail.
This establishes the stated structural projection, not driver folding or GPU
performance.

The actual new native catalogue admits exactly137 typed requests. The complete
cold137 diagnostic then passes all functional gates:137 compiled, zero hits and
failures,136 complete own-parent payloads unchanged, and the single shared
payload exactly0c0f5bce with a fresh validator and all four parent interfaces/caps
preserved. Whole CLI lifetime is16.680 seconds with103.578 process CPU seconds,
6.680 seconds above target. O1/g0/no postprocess/default eight workers and Unity
background work are retained; OS caches are not flushed. This is a candidate-only
observation with different root/provider/workload from the historical140 run,
so it does not establish a stable improvement or regression. Both owned children
close and the lock releases. The candidate is held; native/GPU qualification and
adoption are not advanced without demonstrated startup benefit.

A new private counterfactual tool now compares140 static-class requests with137
shared-class requests using the same executable, regenerated provider, physical
source root and SDK. Both actual ordered lists pass. In the first complete A140
then C137 pair, wall time is19.460/13.354 seconds and process CPU116.250/87.703
seconds. All140 and137 complete blobs reproduce their own closed oracles, with
full typed V3 keys, exact same-source namespace/read proofs and one equal actual
producer. Both children close and the matching lock releases. O1/g0/no postprocess,
default eight workers, independent empty SPIR-V caches and background work remain.
The candidate is still3.354 seconds above target. A reverse pair is needed to
assess execution-order and background variation before attributing this difference;
no stable gain, GPU performance or adoption is qualified.

The reverse C137 then A140 pair also passes every full-catalog/payload/source/SDK
gate. C uses14.121 wall seconds and92.891 CPU seconds; A uses14.865 wall seconds
and96.828 CPU seconds. The observed difference is0.744 wall seconds and3.938 CPU
seconds, much smaller than the forward6.106/28.547-second differences. Both
directions favor C, but the magnitude is unstable under the retained background
and OS-cache conditions. All four complete run receipts are preserved, with
zero cache hits/failures, and both reverse children close with the lock released.
The latest candidate remains4.121 seconds above target; the best candidate
observation is13.354 seconds. Further timing-only repeats are held while scoped
Off/Query sharing and actual rendering/performance qualification proceed.

A separate three-owner OpenPBR Off/Query auto-specialization prototype also
remains rejected. Its fresh Query128 reference reproduces1859f817 exactly;
the candidate compiles and both whole validators pass, with one actual bool
SpecId0/default false and literal8x8x1. The complete resource interface is exactly
equal, but the candidate adds VariablePointersStorageBuffer capability4441 and
grows from186619 to290030 instructions (2889692 to4527536 bytes). The strict
capability gate rejects this expansion. Both source children, both validators
and ownership audit close;880 input pins and both813-file graphs recheck exactly.
Source-child CPU observations5.578/4.547 seconds cannot override that failure or
establish a startup gain. The next investigation is the opaque-resource value
flow and selection of scalar handles before typed resolution; no new Vulkan
feature requirement, native integration or GPU qualification is accepted.

The separate index-selection revision passes that same strict source gate. It
selects six numeric leaves and three32-bit engine handles before resolving the
typed resources after the branch; the original query island is unchanged. The
new whole module253a0177 has2891360 bytes and186740 instructions, only121 more
than the original Query128, and exactly its interface/capabilities. The added
VariablePointersStorageBuffer capability is gone. A again reproduces1859f817;
actual bool SpecId0/defaultfalse and literal8x8x1 are preserved. Both compilers,
both whole validators and ownership audit close, with1707 input pins and both813
graphs unchanged. Source-child CPU4.688/4.641 seconds is not a demonstrated
speedup. No native union, GPU result or source-catalog saving is qualified.

The original Off/Query integration also needs an explicit device contract. Float16/Int16 are
optional enabled Vulkan features, and the current factory receives no trusted
enabled-feature profile. A false specialization still retains Query's capability
requirements. A global catalog must preserve static Off fallbacks; a device-specific
catalog must share the same selected adapter/profile as runtime and cover every
production request for that profile. Existing startup warmup precedes device
creation, and the heap-stride query does not prove final adapter/feature selection.
The four enabled16 profiles, unsupported Query behavior, SINGLE/custom/manual
fallbacks and matching CLI list/cold scope must be resolved before integration.

A separate one-owner SHARC Query decoder now removes that specific type
dependency in an actual source compile. The original scalar half conversions
become four GLSL UnpackHalf2x16 operations, with no16-bit scalar types and exactly
the two Float16/Int16 capabilities removed. The complete Query resource interface
is unchanged. A reproduces the complete1859f817 parent; C is32915b3f,
2889040 bytes and186579 instructions,40 fewer than the parent. Both validators
and the ownership audit pass, all five children close, and888 pins plus both813
source graphs recheck. Single source-child CPU5.688/5.422 seconds does not prove
a speedup. This source screen supplies no numerical, native or rendering
qualification. No device
filter or smaller production workload is used.

The disjoint four-owner combination also passes an actual source screen. Its
whole modulef09bb988 retains bool SpecId0/defaultfalse, literal8x8x1 and the
complete Query interface, with exactly Float16/Int16 removed and no variable
pointer capability. It has2890708 bytes and186700 instructions,81 more than
the original Query. A again reproduces1859f817. All five children close;
1733 pins and both813 graphs recheck. CPU5.422/4.984 seconds is one source
observation. No native union, Guides variant, GPU result or catalog saving is
inherited from these source checks.

The separate scoped Guides129 screen now passes too. Its own reference reproduces
the complete96d3a71c parent; C retains the complete Query129 interface and removes
exactly Float16/Int16. The actual bool SpecId0/defaultfalse, four unpack operations
and literal8x8x1 remain, with no16-bit scalar types or variable-pointer capability.
It has2891188 bytes and186729 instructions,81 more than Query129. All five children
close;1756 pins and both813-file graphs recheck. Source-child CPU5.016/5.094 seconds
supplies no speedup claim. The earlier path-string preflight failure is preserved;
only four recorded path spellings changed in a separate revision. The scoped
decoder guard includes the internal union marker, keeping legacy Query/UPDATE
projections outside this prototype unchanged. Native rendering, GPU performance
and full-catalog startup savings are still pending.

The scoped OpenPBR Off/Query native prototype now builds in a fresh short-root
source/build pair. Exactly six existing files differ from Source137: four shader
owners, the request factory header and its CPU tests. All five native targets
build;38 factory tests in each independent ACEScg/Rec709 process,14 startup tests
and13 standalone native-cache tests pass,103 total with no skips. All eight owned
jobs close. The module audit, original SDK,272 configured settings and read-only
Source137/Build137 pins are retained. A missing experiment-loader helper fails
before staging; a separate thin repair preserves that record and stages18308
files. The new provider is generated from the actual new archive/executable
inputs. This qualifies C++ integration and CPU contracts; the135-request count
remains projected until the new complete actual list passes. Rendering, GPU
performance and startup benefit remain pending.

The subsequent actual native list now admits exactly135 requests, with complete
ordered visible requests/defines matching the typed137-to135 projection. Only
Plain123/128 and Guides124/129 alias; all137 old logical policies and their140
historical mappings remain covered, with native false/true values distinct.
The sole list child closes and the lock releases. This is actual list evidence;
V3 request keys, source namespaces, actual producer and full cold performance
are pending the complete135 run, with no source filter or worker override.

A separate standard-material prototype covers exactly the actual121 Plain Off
and127 Plain Query pair. Its two owners preserve the path loop/RNG/lighting
islands, guard cache setup/lookup/roughness tracking with one actual Bool SpecId0,
and scope packed decoding to its own marker. A reproduces complete25143e39;
C retains the complete Query127 interface and removes exactly Float16/Int16,
with no variable-pointer capability or16-bit scalar types. It has1856748 bytes
and118311 instructions,96 more than Query127. All five children close and both
813-file graphs recheck. CPU2.688/2.703 seconds shows no per-program speedup;
the intended saving is one shared source instead of two. A future135-to134
count remains a projection. No StandardGuides Query request is invented, and
this prototype is not inserted into the current separate native135 epoch.

The subsequent exhaustive GPU decoder probe passes on RTX5060/driver616.92.
All65536 half patterns occur in both packed roles, with131072 complete float-bit
comparisons and no NaN exemption. Every pair is identical; no poison or validation
VUID remains, including teardown. Root independently compares the raw1MiB and
checks all126980 non-NaN values against CPU half decoding. Fresh whole validation
and an actual-entry output dependency gate prove that both legacy conversions
and the distinct unpack operation feed the four output lanes from the same
packed word. The original include-path build failure is preserved; a separate
one-line include repair builds and all five build/GPU/validator jobs close.
This qualifies the decoder operation on that device and SDK, with O1 and
background work retained. Complete SHARC/union rendering, GPU performance and
startup savings remain unqualified.

The existing generic scalar backend GPU tests also execute all three paths
positively after shortening only their output directory. All20 tuple/import
readbacks are exact. The original long-directory run failed native binary file
writes; its245/240-character target paths also receive temporary suffixes.
Both records are preserved. The strict short-run controller still fails because
it expected per-backend disk SPIR-V files which were not published. Actual native
test success is retained separately; full observed-payload admission is pending
a trace-only test dump. This is no PT/Deferred rendering or performance result.

The trace-only test object now closes that actual-bytecode evidence gap. Pipeline,
ShaderObject and PipelineBinary again pass all20 uint4 readbacks, including binary
imports. All three actual returned SPIR-V streams are the same2956 bytes
(7f47d793), with actual signed Int32 SpecId0 and Bool32 SpecId1. One fresh whole
validator passes; the other two reuse its exact full bytes. All five owned jobs
close across build/GPU/validation. The earlier failed cache-publication gate and
the GPU phase's original pending status remain preserved; the independent later
validation receipt supplies the positive closure. This qualifies generic scalar
backend correctness on the observed device, with O1 and background work retained.
PT/Deferred rendering, GPU performance and startup benefit remain separate.

Further authored-complexity review finds no strong new loop cut: dispersion is
already disabled, heavy lobe handlers are distinct single bodies, and path/cache
loops retain ordered state. An all-mode SHARC union is blocked by UPDATE's actual
64-bit atomic compare/exchange and optional Int64Atomics capability. The SINGLE
Guide pair has equal interfaces but different complete walkers and primary-hit
lifetimes. A sample-count shortcut would not preserve those contracts; no such
rewrite is implemented.

A separate static-class4 whole-target screen retains both Deferred entry names,
their8x8x1/32x1x1 workgroups, exact per-entry interfaces and the exact global
resource/capability union. Both single-entry references reproduce their complete
old payloads. The two serial source children use6.766 CPU seconds together;
one whole-target child uses5.922 seconds, a0.844-second single observation.
The combined module has314658 instructions and two fully inlined functions.
All seven children close. This modest source-cost opportunity is held: the
whole job is longer than either original job, and neither startup benefit nor
GPU equivalence is established. No native/RHI integration is advanced.

## Latest getter plus punctual fallback 140-request observations

The A1/C1 pair and subsequent reverse C2/A2 pair execute the same140 requests.
Times cover whole CLI lifetime, including initialization, source capture,
compilation, publication/readback and teardown; external validation is excluded.

| Execution order | Program | Whole CLI seconds | Process CPU seconds |
| --- | --- | ---: | ---: |
| A1 | Getter reference |20.644|128.266|
| C1 | Getter plus punctual tail merge |15.690|101.875|
| C2 | Same candidate, reverse pair |14.598|95.016|
| A2 | Same reference, reverse pair |14.497|95.688|

The initial4.954-second wall difference is not stable: the reverse candidate is
0.102 seconds slower than its reference, with0.672 less process CPU seconds.
Even128 unchanged payloads became faster in A1/C1; the cause of variation is
not established. No stable CDF startup benefit is demonstrated. All runs retain
the same binary/provider/physical source root, actual140 V3 keys, original SDK,
O1/g0/no postprocess/default eight workers and independent initially empty
SPIR-V caches:140 compiled, zero hits/failures. OS file cache is not flushed;
Unity and background work remain running. The best observed reference remains
4.497 seconds above target; GPU correctness/performance and adoption remain
unqualified.

A reproduces all140 closed getter-parent complete payloads. C changes12 complete
payloads, each passing a fresh whole-module validator and exact own-parent full
resource-interface/capability comparison;128 others are byte-identical. C126
exactly equals the combined source-screen18f23aed. Both private owners restore
their original bytes/timestamps and Stage; all14 combined children close and
both locks are released. Root independently rechecks37,729 unique input pins and
both output manifests. C2 and A2 each reproduce all140 complete payloads of
their own closed candidate/reference oracle, with no fresh validator required.
Both reverse children close, both owners restore bytes/timestamps and Stage,
and locks are released; root independently rechecks38,655 unique input pins
and both complete reverse output manifests. Canonical shader source is unchanged. The source edit
shares four mutually exclusive global-CDF fallback calls through one identical
tail; it does not inherit GPU qualification from source control-flow review.

An isolated getter-only source screen captures the complete Scene packet once
within the SHARC-update streamRaySurface read-only boundary, replacing eight
static getter sites while resolving/reading child resources at their original
use sites. One A/C source pair costs5.594/5.250 process CPU seconds, with
160946/155724 instructions; the candidate is333f49d9. Both fresh whole
validators and exact full interfaces/capabilities pass, all five children
close, and input/source bytes remain exact. This is a source-child observation,
not a complete-startup or stable gain. On the combined getter-plus-CDF stack,
a separate source pair reproduces18f23aed before the edit and costs4.906/4.641
process CPU seconds, with156765/151543 instructions. Its candidate00887834
passes fresh whole validation and exact full interfaces/capabilities; all five
children close and both813-file source graphs remain exact. This remains a
single source pair with no complete-startup or GPU qualification.
A separate private material-input bridge has compiled with the original SDK/O1.
It reuses only the20-byte texture/feedback/sampler handle bundle after normal
resolution, preserving existing resource reads and feedback at their use sites.
Its source A/C CPU costs are6.594/5.578 seconds and
instructions156765/150988; C is944dfd7e,2,344,788bytes, two functions/seven
calls. Both complete modules pass fresh validation and exact full interface/
capability comparison, all five children close, and root rechecks all input
pins plus both813-file physical graphs. The internal source conforms in this
actual compile; its initial Summary planning flag still says pending and is
not rewritten after closure. This is one source pair, with no stable or full
startup claim. The four-owner combination with stream capture then reproduces
the same A18f and produces476c2a2d:145766 instructions,2,265,704bytes, two
functions/seven calls. A/C CPU is6.375/6.094 seconds, but wall time is6.811/6.979.
Both full validators/interfaces/capabilities, all five children, input pins
and both813-file graphs pass. The smaller final program is not a demonstrated
stable compile-time or startup gain; no complete140 run follows from this N1.
No new candidate is GPU qualified. Evidence:
`.cache/SlangCompileGoal/ActualGetterPunctualCold140Pair20261015.json` and
`.cache/SlangCompileGoal/ActualGetterPunctualBalanced20261015.json`, plus
`ShaderComplexityGovernance01/StreamRaySurfacePacketCaptureGetterAlphaSourceScreen01/Screen-01`
and `ShaderComplexityGovernance01/StreamPacketCaptureGetterPunctualAlphaSourceScreen01/Screen-01`,
plus `ShaderComplexityGovernance01/SceneTextureHandleCaptureGetterPunctualAlphaSourceScreen01/Screen-01`.

The one-expression primary-guide sentinel screen is HOLD. Adding the redundant
unsigned `guidePixel.x != UINT_MAX` condition preserves the original truth
value for all uint32 width/pixel values, but does not expose the intended
pruning: actual123 goes184838 to184848 instructions and4.563 to4.578 process
CPU seconds. Its A exactly reproduces own C1123 payload9a69aa9e; C is13e31895.
All five children, fresh full validation, exact full interface/capabilities,
input pins and both813-file graphs pass. Do not expand this hypothesis. It is
distinct from the earlier slower NoGuideWriter generic-interface replacement,
which also remains closed. See
`ShaderComplexityGovernance01/PrimaryGuideSentinelGuard01/Screen-01`.

The new OpenPBR runtime guide-policy union instead targets repeated complete
program compilation. Actual123/124 and128/129 each differ by29 final
instructions with equal full interfaces/capabilities, but their jitter,
debug-frame/sample, primary-query lifetime and output policies differ. They
cannot be aliased by deleting requests or adding only a guide-write flag.
The three-owner draft preserves those policies around one authored trace-core
call and uses the audited free bit29. Root and an independent reviewer verify
all836 code/input pins and the full source mapping. The source screens below
check actual O1 core retention; disabled-handle dominance, numeric behavior and
GPU performance remain unqualified.
The matching strict factory/flag/optional-seven-bindings/key design would map
only qualified Off/Query pairs, projecting140 to138; single137/138 and UPDATE
stay separate. The four-owner CPU/test draft is independently reviewed and
sealed, then built in the private snapshot; it is not applied to canonical source. The actual Off source
screen reproduces each own complete123/124 parent and compiles one shared
program with original SDK/O1: A123/A124 use4.797/4.281 CPU seconds, C4.359.
The two A children sum9.078 CPU seconds; this is source-child accounting,
not a complete-warmup projection. Final instructions are184838/184867/185850,
each with two functions/seven alpha calls. C110d49ea passes a fresh whole
validator and exact full interfaces/capabilities against both parents; all
seven children close and root rechecks1730 code/input pins plus both813-file
physical graphs. One authored core and a near-single-parent final program
support the absence of wholesale core duplication; actual guide-off resource
dominance, numeric output and GPU performance remain unqualified. A bounded
offline CFG inspection confirms the actual bit29 load at push byte248 and
excludes the final guide-distance store when its branches are false; it cannot
prove the sentinel/phi correlations excluding the six primary-guide accesses.
That is an unqualified proof obligation, not a demonstrated shader defect.

The separate Query screen also reproduces both own complete128/129 parents.
A128/A129/C use4.203/4.578/4.703 CPU seconds and186619/186648/187631 instructions;
each retains two functions/seven alpha calls. C2903c8c1 passes fresh whole
validation and strict interfaces/capabilities against both parents; all seven
children close, with1748 code/input pins and both813-file graphs rechecked.
The two A children sum8.781 CPU seconds; no end-to-end saving is inferred.
New short private source/build roots now pass the coherent nine-owner native
build, preserving the original compiler/SDK/settings and all closed artifacts.
The exact four targets build; independent ACEScg/Rec.709 processes each pass25
factory tests and all14 Startup tests pass, with no skips. All seven owned
jobs close. Root independently rehashes379 code/input pins, all18305 staged
files and the complete build evidence/artifacts/adjacent SDK. The separate
actual catalog lists138 ordered requests, retaining all140 own parents via
exactly two aliases. The subsequent complete candidate-only cold diagnostic
costs15.515898 whole CLI seconds and96.890625 process CPU seconds:138 compiled,
zero hits and zero failures, default8 workers/original SDK/O1/g0/none/ACEScg,
new explicit empty SPIR-V cache, OS cache not flushed and background work kept.
The136 other complete payloads exactly match their own C1 parents. Both shared
programs exactly match source-screen110d49ea/2903c8c1 and pass fresh whole
validation plus strict full interfaces/capabilities against both own parents.
All actual138 typed V3 keys/source-byte proofs reconcile; all three children
close and the lock releases. Root independently rehashes37,393 inputs and all
428 output files. The generated64-hex provider and observed16-hex V3 producer
are separate retained identities. Different root/provider/request count makes
the historical140 result an unpaired comparison. No startup improvement,
GPU correctness/performance or adoption is established; do not automatically
repeat or expand GPU work on the strength of fewer requests alone. See
`ShaderComplexityGovernance01/OpenPBRRuntimeGuidePolicyUnion01`.

A separate one-owner environment mip-quad source screen is HOLD. It resolves
the same PDF texture and same-mip integer dimensions once for four sibling
weights, retaining each bounds/Load/max/zero fallback and the complete FP/RNG
tail. First A/C source CPU is6.828/6.203s and wall7.436/6.330s; final instructions
156765/155647, bytes2428016/2412488, two functions/seven calls. A exactly
reproduces18f23aed; C is65907cb5. Both fresh whole validators, strict complete
interfaces/capabilities, ownership audit, all five children and both813-file
graphs pass. The reverse C/A confirmation then costs7.453/6.953 CPU seconds
and8.422/7.291 wall seconds, reproducing each own complete blob and explicitly
reusing its closed validation. Root rechecks all2,546 reverse input pins and
the complete output manifest; both children close and the lock releases.
The timing direction reverses despite the small final program reduction. No
stable source benefit, third pair, full warmup, GPU test or adoption follows.
See `ShaderComplexityGovernance01/EnvironmentImportanceQuadQueryReuseSourceScreen12601`.

A separate current-source reflection geometry screen replays only the repaired
two-owner value/PDF geometry reuse, retaining the current feature constants.
It shares normal dots, half-vector and GGX D for the eligible unselected
reflection case, preserving the value/accumulation/PDF order and fallbacks.
The adjacent A/C source pair uses6.719/5.953 CPU seconds and7.172/6.127 wall
seconds; instructions rise156765 to158696 (+1.23%), bytes2428016 to2458252,
with two functions/seven calls. A exactly reproduces18f23aed; C is0de867b7.
Both fresh whole validators, strict complete interfaces/capabilities, module
audit, all five children and both813-file graphs pass. Root independently
rechecks863 package/input pins and the physical graphs before closing evidence.
The reverse C/A costs6.594/6.703 CPU and6.683/6.963 wall seconds, with both
complete payloads exactly equal their first observations and validation
explicitly reused. All2,529 reverse input pins and output files recheck; both
children close and the lock releases. The CPU direction repeats, but the
advantage falls from11.4% to1.6%, with no stable startup or GPU result. HOLD
automatic third pairs, full-warmup/GPU expansion and adoption. See
`ShaderComplexityGovernance01/ReflectionGeometryCurrent140SourceScreen12601`.

The remaining feature-call investigation finds18 direct Translucency and10
Metallic calls in the qualified retained optimized IR, representing42/25
model copies; both constant-true callees still feed branches at that stage.
This is intermediate-model evidence, not final SPV call or runtime cost.
Adding ForceInline to these two methods already has a closed weak historical
screen, so that route is not repeated. A separate associated-constant draft
keeps the original methods and all five current concrete truth values, while
changing44 consumer sites across nine owned shader files. The historical
link-time experiment tool and its test have a separately grouped compatibility
draft preserving all five feature-mask variants, including false values; those
files are not production shader ABI or source-graph deltas. The actual original
SDK/O1 general126 source screen then passes: A/C CPU4.90625/4.828125 seconds,
wall5.372386/4.957693, instructions156765/155802 (-0.61%), bytes2428016/2417184,
two functions/seven calls. A reproduces full18f23aed before C930b1d13. Both fresh
whole validators, exact full interfaces/capabilities, ownership audit, all five
children and both813-file graphs pass. Root rechecks921 code/input pins and
physical graphs. The general production source conforms in this actual compile;
other contexts and extern/export link compositions remain unqualified. The
1.59% CPU observation is weak N1 evidence, so HOLD automatic repeats, full
warmup/GPU expansion and adoption. Its closed Admission retains four unused
reflection descriptive/hash fields from the adapter parent; the root record
explicitly identifies them. Execution instead binds the new frozen917860
manifest/Transform0eec, nine complete owner pins/inverses and exact graph
deltas. Closed files are preserved. See
`ShaderComplexityGovernance01/RemainingFeatureConstantsCurrent140SourceScreen12601`.

The UPDATE lifetime experiment changes only three existing preprocessor guards
to use the already-authored QUERY/single pre-shadow BSDF schedule for UPDATE.
All statement bodies, random expressions and radiance/cache-write positions
remain unchanged. The A/C source126 screen uses6.594/6.516 CPU seconds and
7.167/6.550 wall seconds, with156765/156809 instructions,2428016/2428564 bytes
and two functions/seven calls. A exactly reproduces18f23aed; C is2a7a9ab1.
Both fresh whole validators, strict full interfaces/capabilities, ownership
audit, all five children and both813-file graphs pass. Root rechecks881 code/
input pins and the physical graphs. The1.18% CPU observation with44 extra
instructions is weak N1 evidence; HOLD automatic repeats, full warmup/GPU
expansion and adoption. The old185-slot prepared model proves neither actual
register savings nor removal of the shadow subtree. See
`ShaderComplexityGovernance01/UpdatePreparedBeforeShadowLifetimeSourceScreen12601`.

The standalone Off/Query policy draft has a pre-execution admission correction:
the complete old OFF123 and QUERY128 interfaces differ, and QUERY additionally
requires Float16 and Int16 capabilities. One candidate cannot exactly equal
both inventories. Its first bounded legality gate therefore requires exact
QUERY128 interface/capabilities and explicitly retains the OFF mismatch.
The actual single-C compile then rejects `OpenPBRCacheContext cache = {};`
with E39999: got0 arguments, expected1, identifying the constructor requiring
SHARCParameters. It exits0xffffffff before downstream emission; no SPV or
whole validator is produced. The ownership audit passes, both actual children
close and the lock releases. Root rechecks880 code/input pins, the complete813
graph, copied runtime and output evidence. The original frontend is observed;
the glslang backend is not reached. The rejected initializer is closed without
retry, dummy bindings, uninitialized state or an extra owner. Off compatibility,
disabled-resource safety, compile performance and renderer integration remain
unqualified. See `ShaderComplexityGovernance01/CacheQueryRuntimePolicySourceTypecheck12801`.

The guide-union GPU design remains unimplemented. The current baseline
`captureOutputs` renders another frame for each AOV, so it cannot establish
same-frame color/seven-AOV equality. A future bounded test needs one observer
copying all outputs after the same pass plus matching actual history/push/
scene and pre-query cache contents. A procedural room already exists, but
there is no ready-made immutable positive-QUERY replay fixture. Matching
frames/assets alone does not close the known parent nondeterminism, and an
incremental union test would not qualify its unadopted parent. See its
`GPUQualificationDesign01/Plan.md`; no GPU work was run for this diagnostic.

## Earlier whole-getter 140-request pair

The fresh alpha reference and alpha-plus-getter candidate complete the same140
requests in **20.883 /19.699 seconds** of whole CLI lifetime, with
**127.594 /122.766 process CPU seconds**. Both retain the same binary, provider,
physical source root, actual V3 keys, original SDK, O1/g0, no postprocess and
default eight workers. Each uses an independent initially empty SPIR-V cache:
140 compiled, zero hits and zero failures. OS file cache is not flushed; Unity
and background work remain running. The1.183s wall/4.828s CPU differences are
one paired observation. The target remains9.699s away, with no stable speedup,
GPU correctness/performance qualification or adoption.

The candidate temporarily adds only the historical ForceInline annotation to
the whole-packet getter in its private source tree. All140 reference payloads
equal their closed alpha parents. The candidate has26 changed payloads passing
fresh whole-module validation and exact full interface/capability comparisons;
114 others retain byte-identical validated payloads. Request126 exactly
reproduces1cdfdcd4 from its source screen. All28 combined children close,
source bytes/timestamps and the original Stage are restored, and both locks
are released. A separate read-only closure rechecks37,733 unique input pins and
both complete output manifests. Canonical shader source is unchanged.

Continue source-level structural investigations, including a bounded
SHARC-off OpenPBR surface-function boundary to partition the large optimizer region.
Evidence: `.cache/SlangCompileGoal/ActualGetterAlphaCold140Pair20261015.json`
and `.cache/SlangCompileGoal/GetterAlphaCold140Guard01/{A140,C140}/Summary.json`.

The two subsequent source126 pairs close with original SDK/O1 and full parent
payload gates, fresh whole validators, equal full interfaces/capabilities and
five closed children each. Direct whole-packet access through the field helper
retains the zero-size old path but costs10.484/13.203CPU seconds and
155420/156229 instructions in A/C; hold it without larger runs. Adding the
punctual tail merge to the getter costs7.484/7.000CPU seconds and
160946/156765 instructions. Its A exactly matches actual cold1401261cdfdcd4;
C is18f23aed,2,428,016bytes. These are single source-child observations and do
not change the latest complete-warmup result or qualify GPU behavior/performance.
See `ShaderComplexityGovernance01/WholeGetterRootSpanProjectionAlphaSourceScreen01/Screen-01`
and `ShaderComplexityGovernance01/GetterPunctualAlphaSourceScreen01/Screen-01`.

Two further bounded screens do not supply a new retained program cut. A
SHARC-off surface-vertex noinline hint produces exactly the same complete
request123 payloadfb216447 (189019instructions/two functions/seven calls);
its6.969/6.125CPU-second pair cannot establish the proposed boundary or stable
gain. A single ForceInline on getConstants fails with Slang process exit
0xC0000005 and no candidate payload after its A126 reproduces1cdfdcd4 and passes
validation. All five surface-screen children and all four constant-screen
children are independently confirmed closed; source/input bytes remain exact
and the lock is released. Preserve the compiler failure and do not expand that
candidate. These screens do not qualify a complete-warmup improvement. Evidence is under
`ShaderComplexityGovernance01/GetterVertexBoundary123SourceScreen01/Screen-01`
and `ShaderComplexityGovernance01/ConstantsGetterForceInlineGetterAlphaSourceScreen01/Screen-01`.

## Earlier alpha-boundary 140-request diagnostic

The one-owner alpha candidate completes all140 requests in **20.942 seconds**
of CLI lifetime, with127.266 process CPU seconds. It preserves original SDK,
O1/g0, no postprocess, default eight workers and an independent empty SPIR-V
cache:140 compiled, zero hits and zero failures. OS file cache is not flushed;
Unity and background work remain running. The previous Precision140 observation
was20.382s/127.516CPU; this is an unpaired comparison and shows no demonstrated
complete-startup benefit. Alpha is **HOLD**, with no shader adoption or GPU
performance qualification. Its prepared positive GPU tests have not run.

All140 actual V3 keys retain a one-to-one complete-descriptor mapping to their
own Precision140 parents, including the historical145 policy metadata. Thirteen
changed complete payloads pass fresh whole-module validation and exact own-parent
full resource-interface/capability comparisons;127 others are byte-identical.
Request126 exactly reproduces source-screen3f834fe2:155420instructions, two
functions/seven calls. All14 children close and source/input seals remain
unchanged. Peak working set is4.935GB. The separate native build passes20ACEScg,
20Rec709 and14startup CPU tests plus the333-file ownership audit. Two failed
deep-output build attempts remain preserved; the successful build uses a shorter
private output root with the same source/compiler/SDK/settings.

Request125/126 overlapping elapsed intervals fall10.642/14.689 to8.658/12.703s,
while aggregate request and process costs remain approximately unchanged. These
intervals are not exclusive CPU time or an observed critical path. The source
screen's smaller program and single-child gain therefore do not establish a
warmup improvement. Continue bounded source screens for the historical whole
packet getter and a new four-to-one punctual-CDF fallback tail merge; neither
is adopted or inherits alpha GPU qualification. Evidence:
`ShaderComplexityGovernance01/StreamRayAlphaBoundaryCurrent140SourceScreen01/NativeQualificationPlan01/Full140DiagnosticRevision02/Cold-01/Summary.json`.

Two subsequent isolated source pairs reproduce complete alpha126 before their
single-owner edits. Both retain O1/original SDK, pass two fresh whole validators
and exact complete interface/capabilities, and close all five children. These
are **single source-child observations**, excluding preparation and validation:

| Candidate on alpha stack | A / C process CPU seconds | A / C final instructions | Scope |
| --- | ---: | ---: | --- |
| Historical whole getter ForceInline |10.250 /6.984|155420 /160946|Compile signal repeats; larger final program; subsequent complete-warmup pair above, no GPU qualification |
| New punctual global-fallback tail merge |10.172 /9.203|155420 /151239|Four source fallback calls share one tail call; no GPU or complete-warmup qualification |

The punctual edit changes no sampling expression or distribution. Its first two
fallbacks still precede jitter RNG and the other two still follow three draws;
invalid RIS remains a false return rather than a second global proposal. These
are source-control-flow facts, not a GPU seed-output proof. The actual final
SPIR-V shrinks2,410,724 to2,353,084bytes and retains two functions/seven alpha
calls. Existing ReGIR tests exercise several relevant branches but do not export
seed values and omit short-header/bad-slot-range/bad-CDF cases. Both followups
remain unadopted. Their source-screen evidence is independently sealed in
`ShaderComplexityGovernance01/WholeGetterForceInlineAlphaSourceScreen01/Screen-01`
and `ShaderComplexityGovernance01/PunctualFallbackAlphaSourceScreen01/Screen-01`.

## Latest precision-unified 140-request diagnostic

The earlier private candidate completes all140 requests in **20.382 seconds** of retained
CLI lifetime, with127.516 process CPU seconds. This includes initialization,
native descriptor query, source capture, compilation, publication/readback and
teardown; the external validators are excluded. It retains the original SDK,
O1/g0, no postprocess and default eight workers. The independent SPIR-V cache
starts empty:140 compiled, zero hits and zero failures; OS file cache is not
flushed. Unity and background work remain running.

All140 actual V3 keys reconcile with complete ordered descriptors. Every old145
recipe maps once: five FP32/FP16 pairs become five unions, class0 keeps its old
static FP32 program and134 other requests stay one-to-one. All135 unchanged
outputs equal their own previous complete payloads. The five unions each pass
fresh whole-module validation and exact full interface/capability comparison
against their own FP32 parent. Class1's complete payload also equals the earlier
source-screen program and therefore retains its recorded bit28 control proof.
All six children close, source/input seals remain unchanged and the lock is
released. Peak working set is5.150GB;43 session groups are reported.

The target remains **10.382 seconds away**. The previous24.510-second145-request
run is an unpaired observation with a different private source/provider/count;
no stable speedup percentage is claimed. Production GPU correctness, temporal
behavior and preserved GPU performance remain unqualified, and no shader change
is adopted. The existing fixed OpenPBRDefault scene covers only class1/background
and trivial1/1 precision rounding; extending its precision switch alone cannot
qualify all changed classes or nontrivial rounding. See
`DeferredPrecisionRuntimeExperiment01/NativeQualificationPlan01/Full140Diagnostic01/Cold-01/Summary.json`
and `DeferredPrecisionRuntimeExperiment01/GPUFeasibilityReview01/Review.md`.

Request126 OpenPBR SHARC_UPDATE remains largest at14.689s overlapping request
elapsed, including12.770s codegen;125 Scene SHARC_UPDATE is10.642/9.858s.
These intervals are neither exclusive CPU costs nor an observed critical path.
An actual final-SPIR-V review rules out UV-only alpha geometry as a remaining
cut: unused position/normal/tangent reconstruction is already eliminated.
Seven alpha callback copies remain, making one shared function boundary a new
bounded source-screen hypothesis, with numerical and GPU behavior still unknown.

The subsequent one-owner A126/C126 source screen reproduces actual cold140126
in every baseline byte. Source-child CPU is11.219s versus9.172s in one pair;
instructions fall173,934 to155,420, with two functions/seven calls instead of
one/zero. Both fresh whole validators, strict complete interface/capabilities,
private ownership, all five children and source seals pass. Independent offline
analysis identifies the single3201-instruction Bool/DontInline alpha helper:
each of seven disjoint Proceed-true regions calls it once, and false return
cuts off the same-query Confirm. Each callback region shrinks525blocks/3045
instructions to6/27; coverage GLSL and atomic count/scope/value shapes remain.
This does not prove pointer targets, floating results, dynamic feedback or GPU
performance. The earlier18,270 callback-copy model is not a whole-program upper
bound; observed total reduction18,514 can include other DCE/glue/type changes.
One source pair does not establish a stable or complete-startup speedup.
The next validation uses a fresh coherent native root and existing MASK/BLEND/UV
coverage tests; the default opaque scene is insufficient. See
`ShaderComplexityGovernance01/StreamRayAlphaBoundaryCurrent140SourceScreen01/Screen-01/Summary.json`
and `ShaderComplexityGovernance01/StreamRayAlphaMechanism01/Gate.json`.

## Previous guide/view-unified 145-request diagnostic

The private native candidate completes all145 requests with the original SDK,
O1, debug disabled, no postprocess and the unchanged default eight workers.
The retained CLI lifetime is **24.510 seconds**, including initialization,
compilation, publication/readback and teardown (148.859 process CPU seconds).
Its independent SPIR-V cache starts empty:145 compiled, zero hits and zero
failures. The OS file cache was not flushed. All145 actual V3 keys reconcile;
the11 changed Deferred programs pass fresh whole-module validation and strict
resource-interface/capability checks against their own global1 union. The134
other programs match the closed guide-unified parent's complete bytes and retain
that validation provenance. All12 children close, input/source seals remain
unchanged and the user cache is untouched. Peak working set is5.187GB.

This observation remains **14.510 seconds above the10-second goal**. The earlier
31.810-second156-request result below is an unpaired historical diagnostic with
a different private source/provider and request count; no stable speedup
percentage is claimed. Actual off/on rendering, same-candidate consumed-payload
intersection and GPU performance remain unqualified. Unity and background work
remain running at the user's request. No shader optimization is adopted. See
`DeferredGlobalViewRuntimeExperiment01/NativeQualificationPlan01/Full145Diagnostic01/Cold-01/Summary.json`.

The actual145 log still puts OpenPBR SHARC_UPDATE request126 first at13.458s
request elapsed, including12.267s codegen; Scene SHARC_UPDATE125 is10.054s,
including9.438s codegen. Deferred flat/binned sum37.423s codegen and OpenPBR
path/guides sum46.750s. These are overlapping request intervals, not family
CPU/wall or an observed critical path. The fresh review finds no strong new
ready source cut beyond the current precision investigation. Preparing reusable
MMS average-energy denominators is a separate low-priority lead, but extra
prepared state, eager lookups and generic context call semantics remain costs;
no candidate is prepared. See `Current145ComplexityReview01/Review.md`.

Deferred precision aliases only affect the environment coefficient leaf. In
the closed ACEScg guide-parent modules, all ten FP16/FP32 pairs differ by84B
and Float16 capability; the FP16 form has two half conversion results and no
half arithmetic. A union retaining those half operations still needs a legal
FP32-only fallback, so the unchanged cross-device catalog cannot simply remove
five programs based on the current GPU's capability. A separate device-qualified
profile would change the measured workload and is not the current goal.

A tiny original-SDK/O1/g0 direct-SPIR-V screen now confirms that the float2
`packHalf2x16`/`unpackHalf2x16ToFloat` route emits GLSL450 pack/unpack without
Float16 or Int16. Its native control retains two FConvert instructions; a third
entry retains both paths. All three whole-module validators and canonical/probe
ownership audits pass; all eight children close and input seals remain unchanged.
Only the frontend was observed in the short child sampling windows; this is not
an observed backend-call trace. This establishes availability/lowering, without
numerical, coefficient-sequence, complete-shader or GPU-performance equivalence.
The specification does not promise identical conversion/pack rounding and
denormal behavior. Non-ACEScg can have real half arithmetic and is outside this
two-roundtrip argument. See
`ShaderComplexityGovernance01/DeferredFP16LeafUnificationReview01/PackRoundTripProbe01/Compile-01/Summary.json`.

The subsequent scalar GPU screen passes355,628 finite inputs in[0,1], including
signed zero, all half values/midpoint neighborhoods, subnormal boundaries and
seeded samples. On RTX5060/driver616.92, native, pack and combined entries each
complete three poisoned submissions; all complete readbacks and cross-entry
results are bit-identical. The pack-only entry runs with shaderFloat16 disabled.
The four build/GPU children exit normally without timeout, and an independent
process snapshot finds no active owned PID or compiler image. This does not
qualify descendant cleanup for a failed/timed-out cmd build; future such builds
must use process-Job ownership. Full coefficient sequences, other devices,
non-ACEScg, complete shaders and GPU performance remain unqualified. See
`ShaderComplexityGovernance01/DeferredFP16LeafUnificationReview01/PackRoundTripProbe01/GPUNumeric01/Run-01/Summary.json`.

The complete coefficient-sequence microcheck also passes104,529 generated bases,
each with all nine diffuse/specular/transmission RGB components (940,761 scalar
outputs). A single runtime-selected unified program matches the separately
compiled original FP16 and FP32 coefficient blocks in all complete bytes across
three poisoned submissions per arm. All three original-SDK/O1 source compiles,
three fresh validators, ownership audit and four GPU runs close successfully;
input seals remain unchanged. The native FP16 program retains both roundtrips
(four FConvert instructions); the unified program has two pack/unpack sites and
only Shader capability. The already compiled scalar runner is reused without
a new build shell. This still does not qualify production shader output,
other devices, non-ACEScg or GPU performance. The next gate is the real Deferred
source cost against actual cold145 FP32/FP16 controls, without changing the
cross-device workload or adopting the shader. See
`ShaderComplexityGovernance01/DeferredFP16LeafUnificationReview01/CoefficientSequenceProbe01/Run-01/Summary.json`.

One real Deferred class1 source screen now reproduces the actual cold145
FP32/FP16 controls132/137 in every byte. Their separate process CPU costs are
4.172/4.063s (sum8.234s), versus3.594s for one runtime-precision program. This is
one standalone source observation; it does not predict startup savings or
establish a stable percentage. The unified program has158,266 instructions,
33 more than the original FP16 program, and its complete resource interface and
capabilities match the original FP32 program exactly. The two-owner shader patch
uses a distinct push bit28 and retains the old static/non-ACEScg expressions as
the default fallback. Three fresh whole-module validators, private253-file
ownership, the811-file native/813-file source join, all seven children and
input/source seals pass. No host integration, real precision dispatch, GPU
performance, new complete warmup or adoption has run. The separately recorded
seal lookup repair changes only its report path; the actual screen remains
unchanged. See
`DeferredPrecisionRuntimeExperiment01/SourceScreen13213701/Screen-01/Summary.json`.

An offline review of that actual unified SPIR-V verifies two real bit28 branches:
each true arm contains one pack/unpack pair, and each false arm returns the
unrounded input. Removing both true edges excludes all pack/unpack operations.
Guide26, view27 and precision28 remain independent; HDR remains reachable with
all three disabled. The actual Vulkan root remains12B, the Scene constants wire
extent252B and the flag offset248. This is logical control/dataflow evidence,
not a driver execution or performance result. See
`DeferredPrecisionRuntimeExperiment01/SPVRuntimePrecisionReview01/Review.md`.

The reviewed host prototype restricts runtime precision to ACEScg builtins with
no custom materials: flat Deferred with no extras, or binned Deferred with one
literal MATERIAL_CLASS1/2/3/4. Other extras, expressions, duplicates, explicit
static precision and custom materials retain the original ordered static path;
class0 retains its original FP32 descriptor. Rec709 retains the original paths.
The new runtime macro is factory-owned. This conservatively projects five pair
merges,145 to140 requests, without removing the original behavior coverage.
These counts await the actual new native catalog and complete cold diagnostic.
Half-precision lifecycle/rebuild and quality-profile behavior remain unchanged.
See `DeferredPrecisionRuntimeExperiment01/HostIntegrationRevision02/Plan.md`.

The fresh private native integration now builds all four selected targets and
passes20 ACEScg factory tests,20 Rec709 factory tests in separate processes and
14 startup mock tests, without failures or skips. The333-file ownership audit,
all272 selected build settings (only the derived dependency-output root changes),
source/SDK/metadata seals and all seven owned children pass. The physical stage
contains18,305 source files and exactly five production plus one test delta.
The actual precision catalog and complete cold result above subsequently pass;
production GPU qualification remains pending. See
`DeferredPrecisionRuntimeExperiment01/NativeQualificationPlan01/Build-01/Summary.json`.

## Historical guide-unified 156-request diagnostic

The private native GUIDES candidate completes all156 requests with the original
SDK, O1, debug disabled, no postprocess and the unchanged default eight workers.
The retained CLI lifetime is **31.810 seconds**, including initialization,
compilation, publication/readback and teardown (194.453 process CPU seconds).
The SPIR-V cache begins independently empty:156 compiled, zero hits and zero
failures. The OS file cache was not flushed. All156 actual V3 keys match the
reviewed complete-descriptor projection;22 changed programs pass fresh whole
validation and strict resource-interface/capability checks against their own
guide1 union, and134 programs match their validated static-feature parent in
complete bytes. All23 children close, source/input seals remain unchanged and
the user cache is untouched. Peak working set is5.233GB.

This observation remains **21.810 seconds above the10-second goal**. The earlier
45.051-second static-feature178 result is a historical diagnostic with a
different private source/provider and request count, so these are not a fresh
paired comparison or a stable speedup percentage. Actual guide off/on rendering,
consumed-payload intersection and GPU performance remain unqualified. Background
Unity work is retained at the user's request; GPU timings under that load are
diagnostic only. No shader optimization is adopted. See
`DeferredUpscalerGuidesRuntimeExperiment01/NativeQualificationPlan01/Full156Diagnostic01/Cold-01/Summary.json`.

The actual156 census puts Deferred22 at78.465s (45.24%), OpenPBR path/query/
guides7 at56.358s (32.50%) and Scene PT/guides4 at22.078s (12.73%) of summed
codegen elapsed. These overlapping sums prioritize source work; they are not
family CPU or wall time. All156 reopened final programs contain5,752,376 SPIR-V
instructions in90,659,632 bytes. There are154 unique complete payloads; the only
two duplicate pairs are historical capability variants with just0.408s combined
codegen elapsed. No new large duplicate-program opportunity was found. The
remaining per-program workload is close to the corresponding historical156
subset; the preceding guide-axis change removes repeated compilation rather
than substantially shrinking each surviving shader.

The next minimal source screen uses actual GUIDES requests137/148, which differ
only in GLOBAL_VIEW. A uniform bit27 retains the existing view read and camera
history assignments; the projected Deferred count is22 to11. Both controls must
reproduce their actual cold156 complete payloads, and the unified program must
retain the global-view resource union. One actual three-source screen now
passes those gates, three fresh whole-module validators and the253-file ownership
audit. Local/global source CPU is3.969/4.453s (sum8.422s), versus4.203s for one
unified program. The unified program has158,233 instructions,255 (0.16%) more
than global1. All seven children close and input/source seals remain unchanged.
An actual final-SPIR-V CFG/dataflow check confirms that bit27 loads Scene
push-buffer byte248 and excludes the view344 handle, descriptor and all typed
ViewConstants reads on its false edge. Bit26 independently gates both guides;
HDR remains reachable with both flags off. This is logical-program evidence,
with no runtime invalid-handle or driver-speculation claim.

A new coherent native tree now builds all four targets with the original
settings and SDK;15 request/factory tests and14 startup tests pass without skips,
and the333-file ownership audit passes. Its actual CLI list confirms145
requests:11 Deferred plus134 others in parent order;144 visible pairs retain
the existing omitted-field distinction. The complete cold145 diagnostic above
now admits all actual typed/V3 keys. There is no stable speedup or runtime
qualification. A separate current-contract seven-owner BSDF eval/PDF
fusion source126 pair now compiles and passes fresh whole-module validation,
strict interface/capability equality,253-file ownership and all five closures.
Its reference reproduces the actual cold156 payload. Instructions fall173,934
to167,829 (3.51%), but process CPU rises12.094 to13.313s in this one pair.
This provides no useful compilation-cost signal, so the candidate is closed
without repeats, GPU tests or warmup expansion; no stable regression is claimed.
See `OpenPBREvalPdfFusionCurrentStatic8GuidesRevision02/Decision.md`.
Disabling zero-weight lobes through the existing
translucency switch would alter PDF semantics. See
`Full156Diagnostic01/OfflineProfileReview01` under the native GUIDES plan and
`ShaderComplexityGovernance01/Guides156RemainingComplexityReview01`.

## Latest static-feature 178-request diagnostic

The static-constants candidate now completes one full178-request cold-cache
diagnostic with the original SDK, O1, disabled debug information and the CLI's
unchanged default eight workers. The retained CLI lifetime is **45.051 seconds**
(276.422 process CPU seconds); its internal warmup interval is44.696 seconds.
All178 requests compile with zero cache hits or failures. The48 changed modules
pass fresh whole-module validation and strict interfaces/capabilities; the130
byte-identical modules retain their linked original validation. All64 stages
in the actual-production/catalog intersection match the same candidate's
consumed payloads. All50 children close, exact eight-source/timestamp restoration
passes, and the user cache remains untouched. The OS file cache was not flushed.
This remains **35.051 seconds above the10-second goal**. The same-epoch historical
original58.915-second run is a diagnostic reference, not a new paired or stable
speedup measurement. Positive8, Stateful and Native numerical checks pass in
their documented scopes below. GPU runtime performance remains unqualified,
and the candidate is not adopted. See
`Complete178CandidateDiagnostic01/Candidate-01/Summary.json`.

The closed request logs put Deferred's44 variants at157.425 seconds, or61.89%,
of summed codegen elapsed, followed by the seven OpenPBR ray-query/guide variants
at58.399 seconds, or22.96%. These are overlapping phase sums, not family CPU or
startup wall time. Deferred classes1–4 have similar costs; the new payload hashes
show no duplicate Deferred modules. This prioritizes their common source work
over further class-specific pruning. Request126 remains the longest individual
observation at18.010 seconds, including16.424 seconds codegen under concurrent
load. Group start/end times are absent, so the actual startup critical chain is
unknown. See `Complete178CandidateDiagnostic01/ProfilePriorityAnalysis01`.

The next permutation-governance prototype unifies Deferred's upscaler-guides
axis. This macro guards only two depth/motion resource aliases and their output
block. A six-owner private patch uses an explicit uniform bit26 in the existing
252-byte Scene push-buffer span and optional bindings in the existing384-byte packet, retaining
the original output math. Runtime and warmup factories change together. This
projects44 Deferred programs to22 and the catalog178 to156 while preserving
precision, view, material-class and guide functionality. These are descriptor
counts, not measured startup savings. One original-SDK/O1 source screen now
reproduces both actual parent payloads exactly: guide0 CPU4.719 seconds and
guide1 CPU4.781 seconds, versus4.672 seconds for one unified program. The latter
has158,015 instructions, only12 more than guide1, and its resource interface
and capabilities equal guide1's union. Three fresh whole-module validations,
the253-file ownership audit, all seven children and input seals pass. This is
one three-source observation, not stable or complete-startup performance.
C++ integration now builds all four native targets in a separate tree with the
original compiler, SDK, Release settings and runtime feature availability.
Twelve request/factory tests and fourteen startup tests pass without skips.
The actual new CLI list confirms156 entries:22 Deferred entries and the other134
parent entries in unchanged order. Its visible request/define text has155 unique
pairs because one existing pair differs in fields omitted from the list. The
complete-descriptor factory tests pass; the subsequent complete cold156 run
also reconciles all actual V3 request hashes. Two initial builds failed only in
the newly added test assertion (missing type qualification/direct header); both
records are retained, and the same configured tree passed after test-only fixes.
Off/on guide/HDR outputs and GPU performance remain pending. GLOBAL_VIEW is
under a separate source-screen investigation. See
`DeferredUpscalerGuidesRuntimeExperiment01/Plan.md` and
`ShaderComplexityGovernance01/DeferredRuntimeAxesReview01`.

An offline control/dataflow review of the actual unified SPIR-V confirms that
bit26 is loaded from Scene push-buffer byte248 and controls the optional guide
path. Both resource-field loads (depth336/motion340), their descriptor/image
loads and the two guide writes are unreachable along the false edge. The HDR
write remains reachable. This is final-program evidence, not an off/on GPU
execution. The Scene span is separate from the actual12-byte Vulkan root push
constants. See `SourceScreen13715901/SPVRuntimeGuardReview01` under the GUIDES
experiment.

A separate source-complexity screen exposes the Scene material resolver's
existing zero thin-film weight as a context constant. It adds a resolver-specific
context, retaining positive/manual Scene, Preview, Stateful and Native behavior,
and uses that constant directly at the original thin-film consumers. On request126
with the current static-feature parent, final instructions fall from173,934 to
159,163 (8.49%) and bytes from2,678,916 to2,446,540. Process CPU is15.047 versus
14.625 seconds (2.80% lower) in one normal source pair. Both whole modules pass
fresh validation and exact interfaces/capabilities, and the reference reproduces
the closed178 payload. This is a structural reduction with a weak single-pair
timing signal; it is held pending further evidence, with no numerical/GPU,
startup or adoption claim. See
`ShaderComplexityGovernance01/SceneThinFilmContractReassessment01/Candidate01/SourceScreen12601/Screen-01`.
One subsequent diagnostic reproduces the complete candidate payload and confirms
that all three thin-film calls and all three presence calls disappear from the
entry-reachable optimized Slang IR. Reachable functions fall from493 to469.
This confirms early pruning; dump timings are excluded from performance results.

## Earlier three-source 178-request diagnostic (2026-10-15)

The two missing HZB SPD variants are now included. The expanded workload ran
once with the original sources and once with the same three-source combination
described below: original SDK, O1, debug disabled, default eight workers and
separate empty SPIR-V caches. Recorded whole CLI wall times were **58.915 and
43.164 seconds**, with **358.016 and 259.875 CPU seconds**. Both arms compiled
all 178 requests with zero hits or compile failures. All 356 decoded payloads
exactly matched their previously whole-validated same-arm outputs, including
the two HZB stages consumed by actual rendering; all 69 changed candidate
interfaces and capabilities matched the new reference.

The candidate arm nevertheless **failed its final source timestamp gate**.
The guard had verified original bytes and the complete source identity and
reported successful timestamp restoration. The later stat check found changed
access times; all three modification times were unchanged. The cause is unknown.
An independent cleanup restored the recorded timestamps with only `utime` and
stat, after all owned processes exited. The original failed record is preserved;
there is no passing pair aggregate or accepted performance result from this run.
See `ShaderComplexityCombinedFullWarmup178Acceptance01`, including
`CleanupRepair-01`. The ten-second target and GPU performance qualification
remain unmet, and the shader combination remains unadopted.

An independent request137 screen then used the original four shader modules
and only reused the identical source-transmittance query inside atmosphere
integration. Final instructions fell from 175,492 to 166,882 (4.91%), but
process CPU changed from 5.234 to 5.219 seconds (0.30%). It failed the predeclared
3% CPU-improvement rule for repeat testing and was closed after one pair,
without GPU tests or adoption. Both whole modules passed validation and matched
interfaces and capabilities; the reference fully matched the original payload.
See `AtmosphereSourceTransmittanceCSEExperiment01/OriginalParent13701`.

An independent integer-only corner-hash boundary then completed three
request137 source pairs in AB/BA/AB order. It keeps the original 24 scalar hash
expressions, with all eight float conversions and seven lerps in the caller.
Paired process CPU ratios were **0.91867, 0.87977 and 0.89831**, a median
**10.17% reduction**; median wall reduction was 10.45%. Absolute reference CPU
varied from 3.688 to 5.328 seconds, and OS caches were not flushed. These are
single-request source-child observations, not complete startup results.

Final instructions fell from **175,492 to 142,516** (18.79%). The candidate
actually retained two functions and 132 call sites. A word-stream/type/call-graph
audit verified the reachable helper's DontInline flag and integer-only inputs,
return and operations; this is not inferred from its source attribute. Both arms
were complete-byte stable across all three pairs, the reference matched its
original oracle, and six fresh whole-module validations plus interface/capability
comparisons passed. All 16 recorded children exited and the private graphs left
canonical source untouched. See
`CloudNoiseIntegerCornerOutlineOriginal01/SourceScreen13701/ThreePairs-01`.
Fresh positive-cloud original A/A controls subsequently matched all 2,076
float32 components and nine actual stage payloads exactly. The candidate then
failed the unchanged numerical gate in 43 components: 26 elevated-terrain and
17 orbital-downward components. Maximum failed relative error was about 0.499%;
the other ten probe labels passed. All 27 fresh whole-module validations,
resource interfaces/capabilities and seven actual integer-helper audits passed,
which does not override the output failure. Sources, exact timestamps and user
caches were restored; all recorded children exited. The candidate is rejected
without GPU timing or full178 expansion. Cause is unknown; the integer-only
function audit does not establish equivalent surrounding floating computation.
See `CloudNoiseIntegerCornerOutlineGPUQualification02/Qualification-01`.

The earlier GPU01 control remains a harness failure: its renderer tests passed,
but a missing ROOT import prevented the final extra timestamp check. GPU02
fixed that import, performed actual import checks and ran fresh controls; it
does not inherit or rewrite GPU01. A private fourteen-hash DAG inside the same
outlined helper passed integer/float-leaf reference checks, but remains held
because its required function boundary has no numerical qualification.

The literal default resolved-input factory screen is also closed. With only one
early-inline attribute and the original parent, request126 process CPU rose from
21.047 to 23.984 seconds in one pair (13.96%); final instructions changed from
415,205 to 415,203, and bytes from 6,433,464 to 6,433,420. The reference matched
its complete original payload; both fresh whole-module validations, resource
interfaces, capabilities and input seals passed. No useful CPU or structural
signal justified repeats, further IR capture, GPU tests or full178 expansion.
See `DefaultResolvedInputsEarlyInline01/SourceScreen12601/Decision-01`.

The current investigation returns to authored shader complexity: repeated
OpenPBR evaluation geometry, actual Deferred/atmosphere call chains, and module
dependencies that may force unnecessary frontend work. These are source-review
questions, with no new performance claim or adopted shader change. Lower source
line or import counts alone do not establish less compiler work.

Re-reading the closed **original178 reference** gives the current priorities:

| Family | Requests | Summed codegen elapsed | Share of summed codegen |
| --- | ---: | ---: | ---: |
| VisibilityBufferDeferred | 44 | 191.134 s | 53.97% |
| OpenPBRRayQueryPathTrace and Guides | 7 | 115.229 s | 32.54% |
| ScenePathTrace | 3 | 18.613 s | 5.26% |

These request intervals overlap under eight workers; they are not CPU or
end-to-end durations. Codegen is 83.62% of summed elapsed request intervals,
versus 15.09% for module loading. The longest request, OpenPBR SHARC update126,
took 28.251 seconds elapsed (26.019 codegen) in this one reference. Reducing
Deferred duplication alone therefore does not establish a route to ten seconds;
the large OpenPBR kernels remain an independent priority. Import cleanup alone
has no demonstrated benefit of that scale. See `CriticalPathComplexityReview01`.

One small source screen makes Deferred's existing non-primary environment
contract explicit. Its mirror-level environment call passes literal `false`,
but captured parent IR retains the general Boolean overload and its physical
primary-background branch. The private two-file patch adds a non-primary leaf
with the identical disabled/map/procedural/intensity expressions and changes
that one call. Original primary/aerial/cloud-shadow paths remain intact. The
final module fell from 175,492 to 121,613 instructions (30.70%) and from
2,793,184 to 1,919,520 bytes. Three original-parent source pairs in AB/BA/AB
order produced CPU ratios **1.06780, 0.78187 and 0.81232** (paired median
18.77% lower). The first pair regressed; the mixed directions do not establish
stable compilation benefit. All three reference payloads fully matched the
original oracle and all three candidate payloads matched each other. Six fresh
whole-module validations, resource/layout/capability checks and input seals
passed; all fifteen owned children exited. This remains a potential source
candidate, without GPU-performance qualification, full178 expansion
or adoption. See `DeferredNonPrimaryEnvironmentBoundaryExperiment01`, including
`OriginalParent13701/ThreePairs-01`.

A fresh original-source four-case GPU control subsequently passed: four finite
HDR outputs, all 64 actual consumed stages with fresh whole-module validation,
and source/cache/child closure. It is **not** candidate qualification or A/A.
Coverage review found its sole default material has untextured roughness0.3;
with eight filter layers, it takes level2 rather than the changed level0 path.
The physical-cloud case shares this material, and the two path-tracing cases
do not use this Deferred callsite. New private roughness0 map/procedural cases
and same-camera base-color witnesses are now sealed, retaining the original
four cases. Their gate requires actual Ready/map availability, exact main/witness
visibility and a positive base-color foreground intersection. Fresh original
A/A controls and then the independent-empty-cache candidate now pass: all eight
HDR and four typed visibility outputs match each other and both candidate-to-A
comparisons in complete original bytes (all24 comparisons, zero error). Each
map/procedural pair has137,058 valid positive foreground pixels. Four actually
consumed Deferred payloads changed; the other60 stages retained complete bytes.
All192 fresh whole-module validations passed and204 children closed, with exact
source/timestamp and user-cache restoration. This qualifies this bounded
reflection-positive output contract, not transmission, GPU performance or full
startup. The old control supplies only byte-validated cache seeds, not inherited
images. See `NonPrimaryEnvironmentPositiveFixtures01`,
`DeferredNonPrimaryEnvironmentPositiveGPUQualification01`, `FixtureCoverageReview01` and
`DeferredNonPrimaryEnvironmentGPUQualification01/R0Closure-01`.

Two broader boundaries remain research findings. OpenPBR's resolver already
inlines naturally into `evaluateRaySurface`; adding an attribute to the resolver
alone has no identified new effect. Its explicit depth/subsurface/thin-wall
constants cross a large returned aggregate and seven observed producer/prepare
functions in four owners before reaching dynamic loads and branches. Safely
exposing these facts needs an explicit design that preserves SHARC's early exit
before preparation. Separately, GPUDriven could separate five geometry/layout
implementation units from seven unrelated culling/LOD/raster units. That owner
cut has no measured cost benefit and cannot remove the heavy entries' real
OpenPBR or atmosphere work. See the bounded call-chain and module-boundary
reviews under `ShaderComplexityGovernance01`. The private seven-attribute,
four-owner early-propagation diagnostic is now closed: original126 CPU rose
20.875 to 24.078 seconds (15.34%) in one pair, while final instructions fell
only 415,205 to 415,018 (0.045%). The complete original control, two fresh
whole-module validations, interfaces/capabilities and input seals passed; all
five children exited. This gives no useful expansion signal and does not prove
field/feature elimination. It was explicitly a shared-generic-API experiment,
not a Scene-only contract; no shader change was adopted. SHARC call positions
were intact. See `EarlyResolvedInputsPropagationChain01/SourceScreen12601`.

A separate Scene-resolved feature-contract review identifies an existing type
route through source/context, closure and preparation. A preparation-only cut
still reaches at least seven owners; full thin-wall consumption reaches about
nine and overlaps the earlier held proposal. Manual Scene and positive test
inputs must keep the general path. Simply switching `VolumesEnabled` off is
not equivalent: the current false branch changes tint initialization when
transmission weight is zero. Static Boolean methods also do not establish
early compiler folding; the mechanism needs explicit verification before any
benefit claim. See `ShaderComplexityGovernance01/SceneResolvedFeatureContractReview01`.

The independent seven-owner Scene-resolved volume-preparation candidate now has
one original126 source pair. It adds three static input bounds only on the
resolver-owned route; manual and positive test contexts retain the general path.
Process CPU changed24.531→23.969 seconds (2.29%), while final instructions fell
only415,205→415,105 (0.024%) and bytes6,433,464→6,432,256. The original payload
matched fullf604, interfaces/capabilities matched, two fresh whole-module
validators and the252-file private ownership audit passed, and all five children
closed. This weak single-pair signal does not justify repeat/GPU/full-catalog
expansion. No early-IR folding or runtime benefit is established. See
`SceneResolvedVolumePreparationPrototypeRevision03/SourceScreen12601/Pair-01`.

The larger Deferred design now has a concrete data contract: reuse FP32 color
for unclamped base HDR, store the exact evaluated world endpoint plus a mode in
one RGBA32 image, then finish view atmosphere in one common dispatch after all
material bins. Deferred currently has no PT accumulation; the design preserves
that rule, branch/guide/clamp order and reconstruction failures. It needs at
least nine production files, a prefix-preserving local resource packet and
explicit storage dependencies. It adds **64 logical bytes per pixel** and
**16 bytes per pixel of residency**, plus a dispatch; these are format-level
access counts, not measured hardware bandwidth or a predicted GPU benefit.
`ShaderComplexityGovernance01/DeferredAtmosphereResolveDesign01` records the
design and its qualification requirements. A private request137 compile-budget
counterfactual is now closed: removing only physical primary-view and aerial
contributions reduced source-child CPU from **4.594 to 2.406 seconds** in one
pair, instructions from **175,492 to 61,396**, and bytes from **2,793,184 to
943,416**. The original control matched its complete payload and both modules
passed fresh validation; all five children exited. This deliberately changes
the image and excludes the real resolve shader's compilation, pipeline, dispatch,
barrier and memory costs. It cannot be adopted, rendered as an equivalent result,
or counted toward ten-second acceptance. See
`ViewAtmosphereCompileBudget01/SourceScreen13701/Decision-01`.

The complete nine-file private common-resolve prototype is now sealed under
`DeferredAtmosphereSharedResolvePrototype01`. It preserves numerical expressions,
FP32 endpoint transport, CPU resource/barrier handling and two added warmup
requests (180 in the later actual native listing below). Its 37 pure transformation
checks and the private graph's252-file module-ownership audit passed; this is
not a canonical333-file audit or typecheck. Independent text review found no
specific blocker and documented conditional exactly-once pixel coverage, actual
ABI and runtime obligations.

The first real four-source screen now passes under the original SDK/O1, with
fresh current178 source admission and the complete original137 payload gate:

| Kernel | Source-child CPU | Final instructions |
| --- | ---: | ---: |
| Original137 | 5.047 s | 175,492 |
| Prototype137 material stage | 3.078 s | 60,590 |
| New resolve, local view | 1.453 s | 45,503 |
| New resolve, global view | 1.422 s | 45,524 |

All four outputs passed fresh whole-module validation; nine owned children
exited and all input/SDK seals held. The three candidate kernels sum to5.953
CPU seconds, including both new startup descriptors. Comparing that sum with
one original material variant is an unequal subset; neither the material's
39.01% CPU reduction nor the summed cost establishes complete warmup benefit.
The common kernels are intended to be shared across the Deferred catalog,
which still needs whole-catalog testing. Identical observed137 interfaces did
not establish the new392-byte ABI; the later native build and dedicated ABI
test below address those bounded contracts. Main-render pixel coverage,
numerical equivalence and GPU performance remain unqualified.
See `DeferredAtmosphereSharedResolvePrototype01/SourceScreen137Resolve01/Decision-01`.

An offline audit of those three actual candidate modules confirms the resource
getter's98-word minimum,384-byte endpoint displacement and dynamic handle flow
into the storage-image read/write operations. The original Scene prefix still
uses96 words. The bounds check protects the packet field load; its fallback
handle0 can also reach an image operation, so it is not an image-safety guard.
A short-packet native test must report that handle through a separate safe
output without dereferencing it. CPU sizeof/offset/prefix assertions now pass;
the later native test below checks encoder upload/root count and image selection.
Viewport coverage remains pending. See `DeferredAtmosphereResolveSPVABIAudit01`.

The isolated physical source tree passes the full336-file ownership audit and
builds the four selected native targets. Eight shader-request/layout tests and
fourteen startup tests pass without skips. A new layout test initially failed
C++ preprocessing because template commas expanded inside a GoogleTest macro;
parenthesizing its two arguments fixes that single private test file. The failed
source/logs and interrupted attempt remain preserved separately. The incremental
continuation retains the same compiler, SDK,272 selected settings and actual
feature mask, with no reconfigure or production-source revision. All owned
processes closed and the copied sources/dependencies/SDK/build metadata passed
final checks. This qualifies the native build and CPU contracts, without running
the main resolve rendering or full warmup. See
`DeferredAtmosphereSharedResolvePrototype01/NativeExecutionRecovery09/Execute-01`.

The new native GPU ABI test now passes with validation: both392/384-byte variants
match their exact16-word expected arrays, including98/96-word spans, the384-byte
endpoint displacement, distinct nonzero image handles and exact dyadic pixel
values. The short variant reports the zero endpoint without dereferencing it.
Three actual V3 publications pass source-byte proofs and fresh whole-module
validation; live original frontend/backend modules were observed. All four own
processes and the observer closed, and only the private snapshot cache was
isolated/restored. The ordinary GTest context does not export the32-word buffer;
the executed internal assertions are the numeric evidence. This is a packet ABI
check, without viewport/render-equivalence or performance qualification. See
`DeferredAtmosphereSharedResolvePrototype01/NativeABIRun01/Run-01`.

The newly built compiler's actual `--list` then reports180 requests, with the two
local/global atmosphere resolve descriptors appended as179/180. This is a list
check: the first178 module/entry names and ordered printed defines match the
old actual catalog exactly. It does not establish full typed descriptor or
cache identity, shader compilation, cache-hit or cold-warmup measurements.

The shared-resolve private host subsequently passes the original eight-case
GTest, but its independent fixed numerical admission **fails RTXCRChiang**:
30,808 components in10,440 pixels exceed the original tolerance; maximum
absolute error is0.066075. Both original controls are byte-exact. All other
seven HDR outputs, including all six Deferred cases, and all four uint visibility
captures are byte-exact against both controls. The recorded PT shader payload
multiset is unchanged. A concrete input confound remains: original logs load
the canonical Claire meshlet cache, while the private host cooks a new cache;
the two current cooked blobs have equal size but different SHA. The original
historical cache SHA was not pinned, so neither identity nor causation can be
backdated. The failed gate remains unchanged; no renderer retry or adoption
follows. Validation-on, nonpaired complete Deferred node medians are about
5.8–8.0% higher for several cases, without GPU performance qualification. A
separate closure preserves17 generated asset files and restores the private
cache parent after the original wrapper's cleanup warning. See
`SharedResolvePositive8GPURevision03/FailureReview-01` and
`SharedResolvePositive8GPURevision03/FailureClosureRevision02`.

The independent OpenPBR critical-path track now has an eight-file private
prototype in `OpenPBRContextStaticConstantsPrototype01`: two interface Boolean
constants retain the five existing contexts' exact feature values, and19
consumers read them directly. The original20 methods and generated-provider
anchor remain intact. This differs from the older link-time constant wrapped
in a method. It
preserves positive/stateful and custom1 features but extends external provider
conformance requirements. There are no new runtime fields or permutations.

Its first original126 A/C source screen passes under the original SDK/O1:
whole source-child CPU **24.219→10.766 seconds**, wall27.550→11.315 seconds,
final instructions **415,205→173,934**, and bytes6,433,464→2,678,916.
The reference exactly matches originalf604; both fresh whole-module validators,
interfaces/capabilities, input seals and all five child closures passed. The
candidate also matches the older two-false-method candidate in full bytes for
this one request, without inheriting its GPU or API qualification. This is one
pair with a substantial structural signal, not a stable or full-startup result;
actual early-IR elimination and the other contexts remain unverified. Preview,
custom1, stateful and positive-dispersion checks precede any adoption. See
`OpenPBRContextStaticConstantsPrototype01/SourceScreen12601/Decision-01`.

The first supplementary context screen stopped at Preview: original A compiled
and passed fresh validation, while C reported E30601 because the new explicit
public constants exceeded the original internal context's visibility. No later
case ran; all four children closed. Removing only the constants' public
modifiers then failed E38105: the interface requires public constants. That
second attempt also stopped with four closed children. A third revision restores
public constants and exposes only the two context types; fields, feature values
and arithmetic stay unchanged. Both Preview modules now compile and pass fresh
validation, but the original strict interface gate fails:18 declarations become16.
The removed resources are the runtime sampler array at set0/binding0 and the
3D texture array at set0/binding2. The original module contains actual accesses
to both; they cannot be called unused merely from this comparison. Surviving
resource declarations, the112-byte push layout, entry point and capabilities
match. This is a preserved failed gate, with five closed children, without
Preview GPU/numerical qualification. A later bounded CFG audit refines the
reachability finding: all three original access chains are dominated by a
branch on an actual `OpConstantFalse`. Selecting only that literal false edge,
while retaining every other CFG edge, makes their two blocks unreachable.
This proves these accesses cannot execute in this exact A artifact; it does
not prove complete A/C numerical equivalence or turn the original strict gate
into a pass. See the separately sealed `CFGFollowup-01`.

The remaining three contexts now pass separate original/candidate source pairs
under the original strict resource/capability gates. Six source children, six
fresh whole-module validators and one ownership audit all exited successfully;
input/source/SDK seals remained equal. Their single-pair process CPU observations
are19.078→15.188 seconds for custom1,55.047→15.125 for the Stateful probe, and
7.313→5.031 for positive Native. Instructions respectively change764,912→637,356,
1,730,054→531,628 and294,263→200,510. These are context conformance and structural
signals, without numerical/GPU qualification, stable speedup or full-startup
claims. Preview's earlier strict failure is retained separately. See
`OpenPBRContextStaticConstantsPrototype01/Revision03/RemainingThreeContexts01/Screen-01`.

The first original-source GPU run passes its native test and64 fresh consumed
stage validations, but its wrapper fails a coverage assertion: it required the
legacy Preview owner even though the eight ordinary cases consume only the
other five production owners. All eight source files, exact timestamps, user
caches and68 children close correctly. That failed run remains unchanged.
An independent offline admission pins its raw outputs and failure closure,
reconstructs the64 full-byte V3/stage/validator joins, and passes the unchanged
eight-HDR/four-uint, positive witness, Ready/map,320-byte packet and24×512 sample
checks. It invokes no new compiler, validator or renderer. This admits captured
original data under the five reachable-owner contract. A fresh original run
then passes the complete A/A gate: all twelve outputs and the64-stage payload
multiset match in full bytes, with all64 fresh validators and68 child closures
passing and exact eight-source/timestamp/cache restoration. The static-constant
candidate then passes both original comparisons: all24 output comparisons are
byte-exact with zero error; five consumed stages change (four Deferred and one
PT), while all64 strict interfaces/capabilities and fresh validators pass.
All68 owned children close and exact source/timestamp/cache restoration passes.
The root separately retains the existing controller handle after its subagent
turn fails; that controller exits0 without restarting or terminating the run.
An identical display mapping of the closed A/C images shows no differences.
This first qualifies only these eight cases; legacy Preview GPU behavior,
GPU performance and complete startup remain unqualified. See
`OpenPBRContextStaticConstantsGPUQualificationRevision02/CapturedR0Admission-01`.

The separate Stateful test now passes fresh original A/A and candidate runs.
Its1/8-light maximum error remains1.90274e-7 against the unchanged3e-4 limit;
parameter/texture/prepare/evaluation counts and all event counts match exactly.
Original A/A transcripts and the actual shader payload are byte-exact; candidate
transcripts match both originals, with strict shader interfaces/capabilities
and fresh whole validation passing. All15 children close across the three runs,
and each restores exact eight-source timestamps and user caches. The actual
module shrinks27,453,980→8,511,704 bytes. One same-workload diagnostic changes
source compilation59.444→16.831 seconds and driver PSO creation123.970→32.544
seconds; these are separate probe costs, not startup or GPU execution timings.
The unchanged GTest's numerical assertions are authoritative: its null evidence
context exports no Stateful float bins, so no external cross-arm float-byte
proof is claimed. See
`OpenPBRContextStaticConstantsGPUQualificationRevision02/StatefulC0-01`.

The positive Native/Adobe probe also passes fresh original A/A and candidate
runs. Both modes retain32,768 lanes and16 material families, original2e-4
scaled tolerance and exact event fields. Four actual4MiB outputs match in full
bytes for A/A and all eight candidate-to-original comparisons. The Vendor
shader stays2,019,492 bytes; the Native shader changes4,710,992→3,229,700 bytes.
All six fresh whole-module validations, strict interfaces/capabilities and18
child closures pass, with exact source/timestamp/cache restoration. These
probes use disk-cache-disabled compilation: frozen sources, actual live SDK
and selected GTest establish their bounded route, without a V3 parser-read
claim. GPU performance and complete warmup remain unqualified. See
`OpenPBRContextStaticConstantsGPUQualificationRevision02/NativeQualification-01`.

A separate one-owner screen tests only the two SHARC LUT reads on that exact
static-constants parent. They use the existing28-byte typed field view instead
of returning the full384-byte Scene packet; there are no new offsets, fields or
attributes. The original arm fully matches the actual static-parent payload.
One source pair changes CPU14.547→13.609 seconds (6.45%), but instructions grow
173,934→174,089 and bytes2,678,916→2,679,536. Both fresh whole-module validators,
strict interfaces/capabilities, input seals and all five children pass. This
isolated N1 does not show a complexity reduction or stable benefit, so it is
held without repeat/GPU/full-catalog expansion. It inherits no result from the
older rejected five-file projection. See
`OpenPBRSurfaceLUTProjectionStaticConstants01/Pair-01`.

A bounded SHARC consumption review finds no further large computation that can
simply move behind its cache decision. A successful update bootstrap already
returns before BSDF/LUT preparation; surviving training paths need the complete
estimator and must retain texture/RNG, cache-record and transport ordering.
The unlit path consumes prepared emission with back-face/thin-wall semantics,
so raw evaluated emission is not a safe substitute. This closes that source
review without a new candidate. See
`ShaderComplexityGovernance01/Static8SharcConsumptionReview01`.

The smaller explicit non-primary environment leaf completed all six warm
processes in the fixed A1/C1/C2/A2/A3/C3 order. All72 same-arm output comparisons
and36 candidate-to-original comparisons were byte-exact;384 consumed-stage
checks matched their same-arm anchors, with zero source compiles. All24 timing
streams retain512 frames per run. Eleven paired medians exceeded1 and19 streams
had mixed pair directions. The strict all24 necessary performance flag is false;
observed background activity and repeat variation leave causal attribution
inconclusive. All24 owned children closed and sources/timestamps/caches restored.
There is no GPU-performance qualification, adoption or cold-start improvement
claim. See `DeferredNonPrimaryEnvironmentWarmTimingRevision02/ThreePairsClosure-01`.

## Frozen 176-request comparison (2026-10-15)

Later actual rendering consumption exposed two missing ordinary VisibilityBuffer
HZB SPD requests (`HZB_SPD_WAVE_OPS=0` and `1`). Both use the same compiler
producer as the catalog; neither entry exists in the 176 descriptors. Therefore
this catalog does not cover complete startup. The comparison below retains its
frozen 176-request scope. The two real requests are now appended as 177/178;
22 descriptor, real-cache and startup tests passed without skips in a separate
Release build, and the new CLI listed all 178 while preserving the old 176
identities and order. The expanded cold workload still needs measurement before
complete-startup acceptance. The new GPU controls consumed and freshly validated
both extra stages independently.

The latest frozen catalog comparison uses the original Slang SDK, O1, debug
disabled, no SPIR-V postprocessing, default eight workers and independent empty
caches. The candidate combines the two false Scene feature-method attributes,
the full resource-getter attribute and the cloud integer-prefix sharing below.
Whole CLI wall time was **55.873 to 41.523 seconds** (25.68% lower), process CPU
**354.891 to 255.375 seconds** (28.04% lower), and peak working set
**7.558 to 5.044 GB** (decimal). Both arms compiled and read back all 176 entries,
with zero hits and failures. Original 174 payloads matched their frozen original
controls; the two added variants received fresh whole-module validation. All 69
changed candidate payloads received fresh validation and matched baseline
resource interfaces and capabilities; 107 exact payloads reused their validated
baseline evidence. Source bytes and timestamps were restored after both arms.

This is one paired observation, not a stable speedup. Whole CLI time includes
planning, source capture, compilation, cache publication/readback and teardown;
it excludes external validation, editor startup and native pipeline creation.
The ten-second target remains unmet. The combined version subsequently passed
new four-case GPU correctness controls: PT, Deferred, RTXCR and an inside-cloud
orthographic Deferred case. Both original controls and the candidate had
byte-identical final HDR outputs (6,111,232 float32 components), all 64 consumed
pipeline stages matched resource interfaces and capabilities, and the actual
320-byte atmosphere packet matched. Eleven consumed stage payloads changed;
53 were exact. Known 176 payloads reused exact validated evidence; the two extra
HZB stages received fresh whole-module validation. The 512-frame timing streams
and source/cache cleanup passed, but validation-on correctness timings are not
GPU performance results. Six subsequent warm validation-off runs retained all
12 streams and 512 samples per stream, with zero source compilations. The last
candidate run failed the same-arm RTXCR HDR repeatability check against C0
(maxAbs 0.06607, RMSE 0.000265); the other three outputs remained exact. The
failed run and all samples are retained. This series does not qualify GPU
performance, and background GPU activity was also observed. The cause of the
RTXCR difference remains under investigation; it is not attributed to the
source changes or background activity. The source combination remains unadopted.
Evidence is in `ShaderComplexityCombinedFullWarmup176Acceptance01/Pair-01`.
Combined rendering evidence is in
`ShaderComplexityCombinedGPUQualification02/Qualification-01`.

Scanning the same 176 final payloads in each arm gives **11,584,910 to
8,332,596 SPIR-V instructions** (28.07% lower), with largest functions
**455,603 to 205,961 instructions**. Both catalogs still contain one function
per module and no calls. This measures final program size, not optimizer work
or GPU execution cost. The remaining 44 Deferred variants account for 64.20%
of overlapping code-generation intervals and 70.69% of candidate instructions;
the 36 binned variants alone account for 51.33% of those code-generation
intervals. The current combination has no separately captured optimized IR.

Nonzero material bins do not establish successful surface reconstruction: the
binner and reconstruction have different failure checks, and shading preserves
the background fallback. Successful surfaces also need aerial transport and
source-dependent cloud shadows. Removing these paths by material class would
change behavior. There is no existing universal post-shading fullscreen resolve
to receive them; a separate atmosphere pass would need explicit validity and
endpoint data, additional bandwidth/dispatch and rendering/performance checks.
The retained investigation identifies small integer-packing and identical-source
transmittance reuse screens, without predicting enough savings to reach ten
seconds. See `ShaderComplexityGovernance01/Combined176RemainingHotspots01`.

The integer-packing screen subsequently produced a smaller request137 module
(144,052 to 135,084 instructions) but increased standalone process CPU from
2.984 to 3.719 seconds in one source pair. Both modules passed whole validation
and interface checks; the baseline fully matched the frozen combined payload.
It failed the predeclared 3% CPU-improvement threshold for repeat testing and was
closed without GPU tests or adoption. This single observation does not establish
a stable regression; it does reject final instruction count as sufficient proof
of a compile improvement. See `CloudNoisePackedHashExperiment01`.

## Frozen 174-request investigation (2026-10-14)

The current target is the complete cold-cache startup warmup below ten seconds,
with O1 and GPU execution performance retained. An earlier source-only candidate
completed one **174-request, default-eight-worker** cold-cache pair: whole CLI
wall time **50.949 to 41.269 seconds**, process CPU **329.813 to 270.922 seconds**,
and peak working set **6.805 to 4.782 GB** (decimal). Both arms used the original
SDK, O1, fresh processes, independent empty caches, zero hits and zero failures.
This is a 19.00% wall-time observation, not a repeated end-to-end qualification.
It does not include editor startup or native pipeline creation. The ten-second
target remains unmet, and the candidate is not yet adopted.

A subsequent full-resource-getter inline candidate, on top of that two-method
parent, completed one further cold174/default-eight pair: whole CLI wall
**44.615 to 41.450 seconds** (7.10%), CPU **269.078 to 256.188 seconds** (4.79%).
All 174 V3 entries were decoded in each arm. Reference payloads fully matched
the two-method parent; candidate 115 were unchanged and 59 changed, with 58 fresh
whole-module validations and one exact previously validated payload. All 59
changed interfaces matched. Peak working set was 4.773 to 4.674 GB (decimal).
This is a single paired observation, not a stable speedup or GPU performance
qualification. Compare within each pair; these runs do not establish a new
original-to-compound percentage. A prior reference run failed a final timestamp
cleanup check and remains excluded; the repaired runner checks all source bytes
before restoring and checking the original timestamps.

The original source variant of the current 174-request catalog uses the original
Slang frontend/backend, disabled debug and no SPIR-V postprocessing. Its modules contain
**10,672,422 instructions**, 174 functions and no calls: each module has one
function, with a largest function of 450,038 instructions. Forty-four Deferred
requests account for 224.308 seconds of summed code-generation intervals and
nine core path-tracing requests for 136.858 seconds, together 93.85% of the
catalog sum. Those overlapping intervals identify priorities; they are neither
process CPU time nor end-to-end savings.

Source experiments now target the expansion entering optimization:

| Candidate | Measured source result | Qualification |
| --- | --- | --- |
| `[ForceInline]` on the two existing false Scene context feature methods | Request 126: three AB/BA/AB pairs reduced process CPU by a paired median 49.24%; final instructions 415,205 to 173,934. Complete cold 174-request pair above reduced CPU by 17.86%. | Ordinary PT, Deferred and RTXCR final HDR outputs were byte-identical to original controls. Five SHARC static mean HDR outputs were also byte-identical to both repeated original controls; the existing within-process guide/history tests passed. Cross-arm guide and moving float buffers were not saved. GPU execution performance remains unqualified. All 174 outputs were read back; 46 changed payloads passed whole-module validation. Source changes were restored after each guarded experiment. |
| One `[noinline]` on the thin-film pure-math helper | Request 126: three AB/BA/AB pairs reduced process CPU by a paired median 18.9%; final instructions 415,205 to 295,463 | Ordinary path-tracing HDR failed the fixed numerical comparison, while two original controls were byte-identical. Not adopted; no GPU performance or complete-warmup qualification. |
| Static no-op guide writer for plain PT | Request 123: instructions 449,791 to 431,191, but one pair increased process CPU from 11.359 to 12.578 seconds | Source compilation and whole-module validation passed; no performance qualification. Not adopted. |
| Explicit dielectric context for Deferred class 1 | Request 137: instructions 175,492 to 175,498; one pair increased CPU from 3.969 to 4.578 seconds | Source compilation, whole-module validation and resource-interface comparison passed. No measured gain; not adopted or expanded to GPU tests. |
| Resolved Scene thin-film capability with early inlining | Against the two-method parent, request 137: instructions 157,246 to 154,941 (1.47%); CPU 4.703 to 4.391 seconds in one pair | Parent full-byte identity, candidate whole-module validation and resource-interface comparison passed. Small structural reduction; held without GPU or further-request qualification. The eight-file source API extension is not adopted. |
| Coarse boundary on atmosphere environment integration | Against the two-method parent, request 137: three AB/BA/AB pairs reduced process CPU by a paired median 18.25%; instructions 157,246 to 93,320 (40.65%) | Rejected by positive-cloud GPU numerical comparison. Both parent controls were byte-identical across 2,076 saved float components and all nine consumed shader stages. Candidate elevated-terrain and orbital probes failed 43 components against both controls at the fixed gate. Existing within-process assertions passed, showing why cross-arm comparison is required. No GPU performance or complete-warmup qualification; not adopted. |
| Share integer hash prefixes across cloud noise corners | Against the two-method parent, request 137: three AB/BA/AB pairs reduced process CPU by a paired median 6.00% and wall by 8.31%; final instructions 157,246 to 144,046 (8.39%), bytes 2,513,000 to 2,249,000 (10.51%) | The eight lattice corners share two z and four yz hash prefixes, reducing source hash evaluations from 24 to 14. Original wrapping, float conversions, scale and lerp order remain. Each arm's full payload was stable across all three runs; three fresh candidate whole-module validations and interface/capability comparisons passed. 191,680 integer corner and float32 leaf-bit checks matched. Existing positive-cloud and actual-origin GPU probes also matched all 2,076 saved float32 components byte-for-byte; all nine consumed shader stages passed interface/capability checks, with seven fresh changed-module validations and two exact-payload reuses. These probes do not establish full-scene temporal or GPU performance equivalence. The combined complete176 pair is recorded above; GPU execution performance remains pending. Not adopted. |
| Earlier inlining of the full resource getter | Against the two-method parent, request 126: three AB/BA/AB pairs reduced process CPU by a paired median 29.79%; final instructions increased 173,934 to 181,860 (4.56%). One complete cold174 pair reduced CPU4.79% and wall7.10%, as detailed above. | One `[ForceInline]` on `getResourceParameters<T>()`, preserving its body and guards. Each arm's complete payload was stable across three runs; six fresh whole-module validations and resource-interface comparisons passed. Ordinary PT, Deferred and RTXCR HDR outputs were byte-identical to the parent; all 58 consumed payloads passed fresh whole-module validation and interface/capability comparison. Five SHARC final HDR outputs were also byte-identical to the two-method parent, and all 16 consumed payloads passed fresh whole-module validation and interface/capability comparison. Cross-arm guide and moving float buffers were not exported. GPU execution performance remains unqualified; not adopted. |

These experiments used fresh source compilations with the original SDK
and O1. Their single-request CPU measurements exclude graph preparation and
external validation; they are not startup measurements. Smaller SPIR-V alone
does not establish equivalent rendering or faster GPU execution.

The two-method candidate preserves shader formulas, resource interfaces and
feature values. Its hypothesis is that exposing constants before expansion lets
dead feature branches disappear earlier. After this candidate, the 44 Deferred
requests account for **160.065 seconds (67.53%)** of summed code-generation
intervals, the nine core PT requests for 61.083 seconds (25.77%), and the other
121 for 15.876 seconds. These are overlapping elapsed intervals in the paired
eight-worker run, not CPU time or additive startup durations. They prioritize
further source complexity work. Adding early inlining to the two true feature
methods produced only a 0.09% instruction reduction in one request and has not
been expanded or adopted.

The two-method candidate also completed three validation-off GPU pairs in
AB/BA/AB order, using the same three ordinary fixtures and each arm's verified
warm shader/native PSO caches. All final HDR outputs remained byte-identical,
all 58 consumed shader stages matched their validated same-arm payloads, and
all nine primary timestamp metrics retained 64 frames per arm per pair.
Paired-median shading ratios were PT 0.99217, Deferred 0.99705 and RTXCR 0.98677.
Deferred whole-graph ratio was 1.00135, so the strict all-primary-nonregression
flag remains false. Background GPU activity was observed in every run; neither
uncontended performance nor SHARC execution timing is qualified. These GPU
measurements are separate from cold-cache compilation and do not establish
additional startup savings.

Removing two unused SceneSurface Hair/Adapter imports preserved the complete
request137 payload, but its single-pair CPU change was only 4.547 to 4.500 seconds;
no frontend or stable performance gain is established. Reflection geometry
reuse had a repaired request126 observation of CPU 13.953 to 11.719 seconds with
instructions increasing 1.11%; its request137 instruction count was unchanged.
Both remain unadopted. Current-parent IR is now captured with the original SDK's
`-dump-ir-after checkUnsupportedInst`, with full final-payload equality as a
gate. This is diagnostic Slang IR before SPIR-V legalization, not raw SPIR-V or
a compile-time benchmark; use it to test whether historical expansion/copy
hotspots remain after the two-method change.

The current request137 diagnostic has 384 reachable printed functions. A static
declared-call expansion model assigns about 29.0% of its printed statement
units to cloud hash/lattice helpers. The superseded first model counted calls
again when the dump reprinted them inside assembly operands; the revised model
counts declared calls only and verifies eight lattice calls per noise function.
This is neither an exact IR instruction count nor a GPU cost estimate. It
ignores branch pruning, loop iterations and later optimization. Five modeled
copies of `Atmosphere.integrateEnvironment` enter through visible-environment
and aerial-segment paths. A one-attribute coarse boundary experiment preserves
the Sun/Moon calculation, cloud and clear sample counts, and inner constant
arguments. Its repeatable compile gain did not survive positive-cloud numerical
qualification, so this boundary is rejected. Evidence uses
`ShaderComplexityGovernance01/CurrentParentOptimizedIR13701/Analysis-02`.

The corresponding current-parent request126 diagnostic was separately captured
with identical final bytes. Its hotspot distribution differs: 79 declared call
sites return the full 384-byte `SceneResourceParameters`, then use only fields
(77 sites use one field; two LUT sites use a span and sampler). The getter's
printed body constructs the whole packet, with 399 own statement units. This
supports testing earlier source inlining and field elimination, without changing
the packet, validity checks or resource bindings. It does not predict final
instruction or timing savings: the measured source candidate above compiled
faster while producing a larger final module. Its separate diagnostic confirms
that the getter disappeared by expanding the full packet construction into all
79 call sites; field elimination has not occurred at that captured boundary.
The compile gain therefore does not establish reduced parameter reads, and its
precise downstream cause has not been isolated. Broad SHARC field-read replacement was already
rejected after three complete-catalog pairs with mixed results; this investigation
must not present that old patch as new. The earlier ineffective inline experiment
targeted `getResourceParameterField`, not the full `getResourceParameters` getter.

For continuing source work, retain per-request final payload identity,
instruction/function/call counts and the largest function alongside compile
costs. Reuse the structural analyzer described below; source line counts and
import counts do not measure generated complexity. Express proven material
capabilities through explicit source/context types, preserve texture and RNG
order, and validate changed payloads before expanding a candidate. In
particular, opaque primary surfaces cannot simply disable `enableTranslucency`:
that feature also controls dielectric multiple-scattering reflection.

The frozen 174-request inventory lacks the runtime OpenPBR guides single-sample
combination (`METALLIC_PT_SINGLE_SAMPLE`). Its first compilation must not be
omitted when claiming complete runtime coverage. The catalog source now adds
the base and SHARC-query single-sample requests (174 to 176), retaining existing
request identities and relative order. Twenty request/cache/startup tests passed
without skips in a separate Release tree; both added variants compiled at O1
and then hit the cache through identical runtime descriptors. A rebuilt
176-request warmup pair is recorded at the start of this document; the later HZB
coverage finding there supersedes the assumption that these two additions
completed runtime coverage. The older
measurements above retain their frozen 174-request scope. Keep request-matrix changes explicit instead of
reducing the workload to reach a timing target.

Local evidence is retained under
`.cache/SlangCompileGoal/ShaderComplexityGovernance01/Investigation.md`,
`SourceColdThinFilmNoInlineExperiment01/Decision-01`,
`SourceColdThinFilmGPUQualification01`, `NoGuideWriterSourceScreen01`, and
`DeferredClass1CapabilityExperiment01/Pair-01`.
The two-method source evidence is in
`ContextFeatureGetterForceInlineExperiment01/ThreePairs-01`, ordinary GPU
correctness in `ContextFeatureGetterForceInlineGPUQualification01/Correctness-01`,
SHARC comparisons in `ContextFeatureGetterForceInlineSHARCQualification01/CandidateVsR0-01.json`
and `CandidateVsR1-01.json`,
and the complete pair in `ContextFeatureGetterFullWarmupAcceptance01/Comparison-01`.
Generated payloads, captures and plots remain local output.

## Earlier single-program investigation

The sections below retain the earlier investigation and its proposed experiments
as historical context. The active source-complexity work and current decisions
are recorded above; the old SDK/pass proposals are not the current work plan.

The subsequent [resource narrow-read implementation](PathTracingResourceNarrowReads.md)
has now been measured: paired Slang CLI wall time decreased 41.56%, while native
Pipeline creation showed no stable improvement. The original static evidence
below remains the baseline for that experiment.

This document distinguishes measured compiler evidence from candidate shader
experiments. Production shader source and external snapshots were not modified
by this research. The fixed baseline is the measured Guides request with
`METALLIC_CUSTOM_MATERIALS=0`,
`METALLIC_HAS_RTXCR=1`, SHARC off and the general multi-sample path. Its O1
SPIR-V contains one large function. Attribute experiments should retain O1,
floating-point mode, descriptor ABI, source capabilities and rendering settings.

## Existing cost evidence and identity

The `compute-cold-optimization-n3` O1 prefill report records front-end/cache wall
19.816 s, code-generation API 18.775 s, and SDK-reported downstream time 17.999 s.
The downstream duration is contained in code generation, not an extra additive
phase. Module loading took 0.783 s and component `link_ms` only 0.035 ms; that
small component-link call does not include all IR linking/optimization.

The independent `compute-frontend-cli-O1` diagnostic produced byte-for-byte
identical SPIR-V to that Benchmark module and entry point. Its raw SDK report
has `compileInner=18.762 s`, downstream compile 17.40 s, specialization 196.83 ms,
resource-usage specialization 66.77 ms, and force-inlining passes 7.20 ms.
Nested timers overlap. These observations locate the high front-end cost in the
downstream SPIR-V processing as a whole; they do not assign 17.40 s to one pass
or reveal the driver's internal compiler stages.

The relevant program is module
`Features/PathTracing/OpenPBRRayQueryPathTraceGuides`, Slang entry
`openPbrRayQueryPathTraceGuidesMain`, finalized SPIR-V entry `main`, mapped ABI,
Slang 2026.18.2 / O1 / debug disabled. The identical diagnostic bytecode is
6,940,612 bytes, FNV-1a64 `ef751a428c7a9c48`, SHA-256
`414c4001b496f6e44cdd2e86faa57339d891acc4473e77f34da0b9d196f63b38`.
Retain both raw compiler and final device-rewritten identities; a shared source
filename or entry label is insufficient for an ISA/cold-creation comparison.
Compiler binary, dependent modules/includes, macro set, device/driver and PSO
state identities belong with each result.

The prior O0 arm exceeded the declared HDR tolerance at two pixels. This analysis
does not change the production O1 default or treat O0 as an accepted equivalent.
See [Compute experiments](ComputePipelineExperiments.md) for complete paired
creation/execution conditions and the existing qualification failure.

## Static SPIR-V findings from exact real PSO recipes

`Tools/Perf/AnalyzeSPIRVComplexity.py` now reads raw SPIR-V without changing it,
checks recipe byte count/FNV binding, and exports instruction categories,
per-function CFG/SCCs, static call graph, callsite expansion contributors and
storage classes. The SDK Khronos opcode grammar and GLSL.std.450 grammar are
hashed in its report. This structural reader complements `spirv-val`; it is not
a complete validator. Thirteen CPU tests cover diamonds, loops, 64-bit switches,
recursion, deep graphs, source attribution, memory storage classes and malformed
inputs/recipe identity. The completed 36-module analysis is in
[`spirv-complexity-20261012-final`](../build-release/spirv-complexity-20261012-final/Complexity.json).

| Guides | O0 | O1 | O2 |
| --- | ---: | ---: | ---: |
| SPIR-V bytes | 709,664 | 6,940,612 | 1,992,720 |
| Module instructions | 42,673 | 447,484 | 115,049 |
| Functions | 574 | 1 | 1 |
| Largest function instructions | 4,192 | 446,028 | 113,727 |
| Basic blocks | 4,594 | 58,220 | 17,277 |
| CFG edges | 5,020 | 74,427 | 22,503 |
| Loop merges | 55 | 495 | 242 |
| Phi nodes | 0 | 0 | 7,369 |
| Ordinary cold native median, 3 processes, seconds | 56.160 | 63.109 | 50.303 |

Cold medians come from the prior controlled-cache, capture-disabled experiment;
the static analysis reuses its corresponding finalized bytecode. The O1 Guides
instruction count is 27.65 times the next largest shader in this particular
12-Compute opaque-scene cohort (`gpuDrivenStreamAssetCooperativeLodMain`,
16,181 instructions). It is already the dominant startup PSO in the prior scene
measurement. This cohort does not contain a SceneRTXDI recipe, so these findings
do not establish a SceneRTXDI ranking or identify the driver's internal passes.

The O0 entry reaches 574 distinct functions and its full static-callsite
expansion proxy is **1,029,336**, with maximum static call depth **20**. Thus
smaller serialized O0 does not establish a smaller program after inlining.
The proxy visits all static callsites, including both branch arms, counts a
callsite inside a loop once, repeats function headers/parameters, and performs
no constant folding/dead-code removal. It is neither an actual optimized code
size nor dynamic work nor a fraction of Driver compile time. An independent
topological recomputation agrees with the analyzer's SCC-based calculation.

| O0 static expansion contributor | Occurrences × own instructions | Contribution to proxy |
| --- | ---: | ---: |
| `Metallic.getResourceParameters` | 603 × 497 = 299,691 | 29.115% |
| `openPBRComplexDivide` | 2,223 × 30 = 66,690 | 6.479% |
| `openPBRSquare` | 5,252 × 6 = 31,512 | 3.061% |
| `Metallic.getParameters` | 604 × 52 = 31,408 | 3.051% |
| `openPBRFresnel` | 795 × 36 = 28,620 | 2.780% |

Function IDs disambiguate identically named generic specializations; retained
`OpName` does not encode their complete signatures. All three Guides modules
lack `OpLine`/`OpNoLine`, so the tool does not invent source-line attribution.
The O0 names support matching source functions and selecting experiments;
they do not prove which source line consumed compile time.

O1 has **26,347 Function-storage variables, 40,669 Function-storage loads and
49,761 Function-storage stores**; O2 has **66 / 319 / 147** respectively. These
are IR temporaries, not evidence of local GPU memory or spilling. O1/O2 have
279/36 surviving `LoopControl Unroll` requests; counts neither specify trip
counts nor establish which requests the Driver honored. A large CFG SCC can
contain many nested loops and should not be described as a single loop.

Reproduce the structural analysis without compiling or creating a Pipeline:

```powershell
& 'C:/Users/jiangwentao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B `
  Tools/Perf/AnalyzeSPIRVComplexity.py `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O0-pipeline-normal/recipes `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O1-pipeline-normal/recipes `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O2-pipeline-normal/recipes `
  --output-dir build-release/spirv-complexity-new
```

Metallic shader disk-cache `.spv` files contain a container header/dependencies;
use raw recipe modules or validate and extract the container first.

## Highest-priority source experiment: narrow resource reads

The whole-packet getter is a stronger measured structural candidate than a
blanket optimization-level change. `Shaders/Modules/Core/NamedResources.slang:8`
loads the 24-byte two-span compute root, then the full 384-byte
`SceneResourceParameters` packet for each leaf access. The existing
`getResourceParameterField<TPacket,TField>` at line 18 reads only the resources
span and the requested field while preserving full-root/full-packet size,
alignment and bounds checks. Keep the existing wire ABI and use this helper at
the leaf adapters; introduce separately declared views with `offsetof` checks
rather than rearranging the packet.

Candidate views, in bytes, derived from current declarations:

| Field/view | Offset | Bytes | Contract to preserve |
| --- | ---: | ---: | --- |
| `scene` / acceleration-structure address | 0 | 8 | `resolveAccelerationStructure` |
| `lights` + `lightsPdf` | 88 | 8 | Span/count and PDF behavior |
| material sampler | 108 | 4 | Sampler identity |
| material textures span | 112 | 12 | Bounds and `resolveNonUniform` |
| materials handle | 140 | 4 | Material identity |
| OpenPBR 2D/3D LUT spans + sampler | 172 | 28 | Both bounds and nonuniform resolution |
| ray stream header/instances/page table/pages | 244 | 16 | Resource identities |
| ReGIR grid | 260 | 4 | Separate from nonadjacent lights fields |
| environment view | 356 | 28 | Distribution/LUT and sampling semantics |

These offsets are research candidates, not a new API or an implementation.
Confirm CPU/shader offsets by static assertions before use. The existing
`named_resource_field_views_and_bounds` RHI test covers truncated packet,
out-of-bounds and unaligned accesses. Extend coverage for new views and compare
the same final GPU outputs. Guides reads camera constants once at its entry;
most of the 604 root-getter occurrences above accompany the 603 resource-getter
occurrences. Begin with resource leaf getters, then inspect O1 bytecode before
attributing a benefit to this change. If O1/Driver already eliminates the
additional fields, the experiment may show no native or GPU improvement.

## Driver executable evidence and ISA availability

Added opt-in Compute Pipeline capture using
`METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY=<fresh directory>`. It enables
`VK_KHR_pipeline_executable_properties` when supported and creates pipelines
with both capture-statistics and capture-internal-representations flags. It
retains the driver's statistic names/descriptions/formats and exact representation
bytes in a per-capture manifest with device/driver/cache UUID, PSO and both
input/final SPIR-V fingerprints. Properties/statistics/representation enumeration
is capped at 256 entries, representations at 256 MiB total, with finite
`VK_INCOMPLETE` retries and exclusive capture-directory creation across
processes. Unsupported queries and empty representations are recorded. Files
use numeric names, not driver-provided path text. The opt-in does not change
ordinary pipelines. Shader Object has no `VkPipeline` executable to query.

Queries/export happen outside the native creation timer; they remain inside
the caller's pipeline wall time. Capture flags may change compilation. Use
fresh explicit application caches and independent processes, and keep these
diagnostic timings separate from the ordinary creation/GPU experiments.
[Vulkan describes internal representations as implementation-dependent](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutableInternalRepresentationsKHR.html):
they may be intermediate IR, binary or assembly. Their label/content must be
examined before calling them ISA.

Three independent captured recipe replays completed on the same RTX 5060 /
driver 616.92, with debug/validation disabled, controlled internal cache disabled
and empty application caches. All input/final SPIR-V byte fingerprints match
the static table. All three query the `CS` executable successfully, subgroup 32,
and return **zero internal representations** (`VK_SUCCESS`). The driver
executable statistics are:

| Captured Guides | O0 | O1 | O2 |
| --- | ---: | ---: | ---: |
| Register Count | 255 | 255 | 255 |
| Binary Size (bytes) | 1,637,248 | 1,776,640 | 1,777,024 |
| Stack Size (bytes) | 0 | 0 | 0 |
| Shared Memory Size (bytes) | 12,288 | 12,288 | 12,288 |
| Local Memory Size, raw UINT64 | 68,719,479,456 | 68,719,489,520 | 68,719,489,600 |

The local-memory statistic is implausibly large for per-thread allocation and
is retained verbatim, unqualified. Do not truncate it to 32 bits, reinterpret
its upper bits, or claim measured spill traffic/occupancy from this value.
The equal register count alone does not establish occupancy or local-memory
behavior. Captured O1→O2 cuts serialized instructions by 74.3%, but executable
byte size changes by only **384 bytes (+0.022%)** and register count is unchanged.
The Driver therefore largely converges these two front-end forms to similar
code size; easier incoming IR may still reduce intermediate compilation work.
Neither the code-size statistic nor the existing ~20% lower ordinary O2 native
median identifies a particular Driver pass. O2's much higher front-end cost also
prevents treating that native reduction as an overall source-cold speedup.

The [capture summary](../build-release/driver-complexity-20261012-v2/summary.json)
links the exact recipes, commands, Bench executable hash and raw manifests.
It records n=1 diagnostic durations, not a new performance qualification.
To reproduce a single baseline capture in a fresh output directory:

```powershell
$env:METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY = "$PWD/build-release/driver-capture-new/representations"
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
try {
  & ./build-release/Source/MetallicShaderPipelineBench.exe `
    --recipe-file build-release/compute-gpu-optimization-512/block-00-O1-pipeline-normal/recipes/compute-5aa0734d26f3cc25.recipe.json `
    --iterations 1 --pipeline-cache empty --shader-cache reuse `
    --descriptor-mode mapped --debug-mode disabled `
    --cache-dir build-release/driver-capture-new/cache `
    --output-dir build-release/driver-capture-new/bench
} finally {
  Remove-Item Env:METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY
  Remove-Item Env:METALLIC_VK_INTERNAL_PIPELINE_CACHE
}
```

### Native binary and Nsight attempts

The existing O1 Shader Object and Pipeline Binary containers were parsed and
verified against their magic/version/lengths/FNV checksum and successful
independent-process import records. Original caches were not modified.
Shader Object payload is 4,430,994 bytes with no ELF. Pipeline Binary payload
is 5,385,894 bytes; its standard Zstd frame at byte 4 decompresses exactly to
12,090,716 bytes beginning `NVDANVVMNvuc`, still with no ELF. Installed CUDA
13.3.73 `nvdisasm` rejects all these payloads as non-ELF. Header text alone
does not prove an NVVM IR schema. Raw disassembly with guessed instruction
offsets/architecture was not used. These cache payload sizes do not measure
machine instructions or executable code bytes. See
[NativeBinaryInspection.json](../build-release/shader-complexity-research/native-binaries/NativeBinaryInspection.json)
and its sibling `Inspection.md` for exact commands, source/import bindings,
tool identities, hashes, decompression checks and rejection logs.

The [Nsight Graphics skill](C:/Users/jiangwentao/.codex/skills/cli-anything-nsight-graphics/SKILL.md)
and its CLI doctor confirm installed Nsight Graphics 2026.3.1 capture/replay.
That supported CLI exposes trace/capture metadata, screenshots and timing,
but no arbitrary shader/SASS export. Native desktop UI control is unavailable
in this session. [NVIDIA's Shader Profiler documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)
says full SASS disassembly requires Pro builds; this installation's Pro
eligibility was not established. A next diagnostic session can use Real-Time
Shader Profiling to examine actual instruction mix/stalls and, where eligible,
full SASS. Source correlation requires a separately identified debug-bearing
module: current release SPIR-V has no `OpLine`. Retain ordinary no-capture
timings/output checks as the performance reference.

No trustworthy Driver ISA was obtained by these three routes. The executable
statistics and static SPIR-V findings above are the available evidence, and
they support experiments without fabricating assembly-level attribution.

Validation completed for this change:

- Release `MetallicShaderPipelineBench` and `MetallicRHITests` build succeeded.
- Focused perf CTest passed 5/5 targets, including the 13-case complexity suite.
- Bench CLI tests passed six cases; four opt-in GPU cases were skipped.
- `typed_named_kernel_parameters_and_lifetime` passed with diagnostics absent,
  and twice in separate processes with capture and Vulkan validation enabled;
  no validation errors were emitted. Each process created two distinct capture
  directories for the same two PSO hashes, with earlier manifests retained.
- All three real Guides diagnostic replays succeeded and preserve their exact
  raw/final bytecode identities. The real shadow-only candidate passed SPIR-V
  validation. No candidate shader was run in the renderer or promoted, and
  production shader source/optimization defaults remain unchanged.

## Source boundaries and feature state

- `OpenPBRSurface.slang:33-46` specializes sheen/coat off when custom materials
  are off; dispersion is always off; translucency and metallic are on. The
  fixed request's growth cannot be assigned to disabled coat/dispersion or Slab
  variants. This conclusion must be revisited for requests with other macros.
- `OpenPBRPathVertex.slang:59-77` retains the fiber branch at runtime. The
  `METALLIC_HAS_RTXCR=1` request contains Chiang hair preparation/evaluation even
  when the measured room contains only opaque surface materials.
- `SceneFiberPath.slang:8-106` evaluates the prepared fiber closure in celestial,
  punctual and environment direct-light paths, then samples it. The same hair
  evaluator has multiple static call sites before exhaustive inlining.
- `OpenPBRPathVertex.slang:170-240` keeps prepared BSDF state alive over direct
  lighting, PDF/evaluation and nested visibility; `:275` later samples it. The
  SHARC-query single-sample special case already moves remaining BSDF work before
  visibility, but the general SHARC-off multi-sample baseline does not use that
  branch. Preserve RNG and lighting/termination order in any lifetime experiment.
- `OpenPBRRadianceCache.slang:10-24` directly evaluates the full mixture in a
  light loop; `OpenPBRPathVertex.slang:209-211` separately calls PDF and projected
  evaluation for the environment. `OpenPBR/AggregateLobe.slang:145-213` traverses
  the six base lobes independently for value and PDF. `:217-307` samples a
  selected lobe, then evaluates/PDFs the other matched lobes to produce the full
  mixture weight. Dropping those mixture terms changes the estimator.
- `OpenPBRClosure.slang:11-128` stores and passes a nested prepared BSDF value;
  `OpenPBR/BSDF.slang:43-52` nests fuzz → coating → aggregate and its six lobes.
  Whole-struct constants/loads/inserts/extracts may grow intermediate IR, but
  source by-value syntax alone does not prove actual memory copies or spills.
- `OpenPBRRayQueryPathTraceGuides.slang:65-72` evaluates its own material. Its
  guide stream, footprint and face-forward policy deliberately differ from the
  path stream (`:112-123`). Reusing the path closure for guides would violate
  that existing contract. Keep authored normal/TBN data stable.

## Lowest-risk experiment: selective noinline at O1

[Slang's documented noinline attribute](https://docs.shader-slang.org/en/latest/external/core-module-reference/attributes/noinline.html)
requests that a downstream compiler retain a function boundary. SPIR-V
`DontInline` is a performance hint, not a guarantee that the driver preserves a
machine-code call. [Function control specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_function_control)
and [SPIRV-Tools inline pass](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/inline_pass.cpp)
show the representation and the optimizer's check.

A local integer probe tested the installed Slang 2026.18.2 compiler. O1 and O2
without `[noinline]` each produced one function, zero calls and two loop bodies.
With `[noinline]` each produced two functions, two calls and one loop body;
the helper retained `DontInline`. O1 byte size changed 1,676 → 1,428; O2
1,272 → 1,168. All four binaries passed `spirv-val --target-env vulkan1.3`.
This proves the pinned SDK can preserve this boundary in simple SPIR-V; it does
not measure Guides creation, numerical equivalence or GPU performance.
The [probe summary](../build-release/shader-complexity-research/InlineProbeResults.json)
and [source](../build-release/shader-complexity-research/InlineProbe.slang) are
local diagnostics outside source control. They record hashes and compiler/tool
identities. Reproduce a probe arm with:

```powershell
$root = 'build-release/shader-complexity-research'
& External/slang/bin/slangc.exe "$root/InlineProbe.slang" -entry main `
  -stage compute -target spirv -profile spirv_1_6 -O1 `
  -D TEST_NOINLINE=1 -o "$root/O1-1.spv"
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe' --target-env vulkan1.3 `
  "$root/O1-1.spv"
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-dis.exe' "$root/O1-1.spv" `
  -o "$root/O1-1.spvasm"
```

A real Guides **shadow-only** isolated O1 compile was also completed. All 339
shader files were copied into an ignored overlay along with the required
relative CPU interop/includes; the only shader edit adds `[noinline]` to
`openPbrShadowTransmittance`. The depfile confirms that shader dependencies came
from the overlay, and original source hashes remained unchanged. It uses the
same fixed request's CLI flags/macros/capabilities as the byte-identical O1
reference, with depfile/include diagnostics added. `spirv-val --target-env
vulkan1.4` passed. The full binding, compiler/source hashes, command and results
are in [CandidateShadowOnly.json](../build-release/shader-complexity-research/CandidateShadowOnly.json).

| O1 Guides | Baseline | Shadow-only noinline |
| --- | ---: | ---: |
| Bytes | 6,940,612 | 6,946,528 (+0.085%) |
| Instructions | 447,484 | 447,751 (+0.060%) |
| Functions / calls | 1 / 0 | 3 / 2 |
| Largest function instructions | 446,028 | 427,449 (-4.17%) |
| Basic blocks / CFG edges | 58,220 / 74,427 | 58,222 / 74,427 |

The two helpers each contain 9,364 instructions and 1,453 basic blocks, retain
`DontInline`, and each have one static caller. Their common source name does not
mean they were deduplicated. This proves preservation of the boundary in the
large real shader but shows **no total-size reduction** for this candidate.
Compile timing overlapped the C++ build and is not a performance measurement.
Driver creation, dispatch and output qualification were not run for that
historical candidate. The 2026-10-13 material-record narrow-read follow-up has
now passed fixed-scene correctness, without establishing stable compiler or
native-creation gains. The current-source next experiment therefore revisits
the shadow function boundary, with paired uncached source compilation,
identical two-pass cleanup and separate native creation. It must reproduce the
current original API bytes rather than reuse this older candidate. No current
cold-creation improvement is established by the historical result above.

### Current-source shadow boundary follow-up (2026-10-13)

The new pure
[overlay planner](../Tools/Perf/PathTraceFunctionBoundaryOverlay.py) changes
only the annotation before `openPbrShadowTransmittance` in its current owned
implementation. The
[paired compiler harness](../Tools/Perf/ValidatePathTraceFunctionBoundaries.py)
retains all General features, resource layouts, macros and RNG/texture order.
It uses a fresh explicit-postprocess-`none` API reference at request 146,
Slang 2026.18.2, mapped O1, debug/validation off, SHARC off and
CUSTOM_MATERIALS=0. There is no module/SPIR-V cache reuse in the six independent
source compilation processes, ordered AB/BA/AB.

All three Original outputs match the fresh 6,947,064-byte API reference
byte-for-byte; each arm is deterministic across its three processes. Every raw
output passes Vulkan 1.4 scalar-layout validation. The
[compiler result](../build-release/path-function-boundaries-20261013/compile/summary.json)
and preparation are sealed with source/include/SDK/helper/grammar identities,
actual process intervals and before/after input checks.

| Raw O1 structure | Original General | Shadow noinline General |
| --- | ---: | ---: |
| Bytes | 6,947,064 | 6,953,072 |
| Instructions | 449,813 | 450,080 |
| Basic blocks / CFG edges | 60,501 / 77,877 | 60,503 / 77,877 |
| Functions / calls | 1 / 0 | 3 / 2 |
| Largest function instructions | 448,375 | 429,406 |
| Reachable call depth, edges | 0 | 1 |

The two retained helpers each have 9,559 instructions, 1,663 blocks and one
static caller, with `DontInline`. The largest function shrinks by about 4.23%,
but total code grows and the helpers are not deduplicated. Function counts here
include `OpFunction`, parameters and end instructions; they must not be mixed
with another parser's body-only counts. Static structure does not prove that
the Driver retains native calls.

Original CLI wall times are `10.160760 / 9.677695 / 9.974684` seconds;
candidate times are `10.081328 / 9.848359 / 9.794524` seconds. Paired ratios
are `0.992182 / 1.017635 / 0.981938`, with median **-0.78%** and a direction
reversal. This does not establish a stable compiler gain. SDK phase counters
overlap and remain separate from process wall time. Static analysis and
validation are outside those compile windows.

The compiler evidence remains unchanged while the analyzer is improved: its
CFG membership check now constructs one set of block IDs instead of repeatedly
converting the whole graph for every block. This changes tool overhead, not
shader compilation. The following cleanup/native/GPU stages archive their
actual tool versions independently; historical verification retains the older
compiler-stage analyzer. Production shaders and compiler defaults are unchanged.

The [uniform cleanup result](../build-release/path-function-boundaries-20261013/analysis/summary.json)
applies only `--eliminate-dead-branches` and `--simplify-instructions`, with
Vulkan 1.4 scalar validation. It retains the candidate's original reachable
function IDs, `DontInline` controls and actual calls.

| Two-pass-cleaned structure | Original General | Shadow noinline General |
| --- | ---: | ---: |
| Bytes | 3,314,908 | 3,320,916 |
| Instructions | 215,311 | 215,578 |
| Basic blocks / CFG edges | 28,217 / 36,088 | 28,219 / 36,088 |
| Functions / calls | 1 / 0 | 3 / 2 |
| Largest function instructions | 213,863 | 194,910 |
| Largest function blocks | 28,217 | 24,895 |
| Function variables / OpLoad | 24,775 / 21,545 | 24,778 / 21,573 |

The largest function now decreases by about **8.86%**, while total bytes
increase by about **0.18%**. There is no reduction in total CFG edges. The
additional Function variables/loads are static SPIR-V counts, not evidence of
native parameter copies, register pressure or spills. Original cleaned SHA-256
is `ca442308a2b31c7b8d06534ad530c8fab95955ecd37de211d7d64ce382f6c4ba`;
candidate is `0a5bcb5e5bc8b46a553579773aeaa6ed7fae33229195bce8f7610431c3760580`.
These exact artifacts are the inputs to the separate Driver comparison.

The [native comparison](../build-release/path-function-boundary-native-20261013/summary.json)
was scheduled as six independent AB/BA/AB processes. It replays the full
captured v6 Compute Recipe with 12-byte user push, normal Driver optimization,
mapped ABI, empty application cache, controllable Driver cache disabled,
debug/validation off and no executable-capture flags. The original template
hash is verified before substituting either cleaned artifact. Frontend work,
initialization, cache export and GPU execution are outside the native window.

Only the first Original call completed: **53,318.697 ms native** and
**53,318.642 ms Driver**, with valid miss feedback and no object/binary reuse.
The first Shadow call entered `vkCreateComputePipeline` but did not return
within the predeclared **600-second process timeout**. Its PID was 42320;
the retained process record has `timed_out=true`, `native_crash=false`,
09:05:32.593744Z start and 09:15:35.679055Z end. CPU time continued increasing
while the call was pending. The process was terminated at timeout, and the
failed batch's complete flat seal was independently verified.

The 603.064-second process wall time includes setup and timeout cleanup. It is
**not** a returned native/Driver duration. There is no candidate End record,
successful target report or final creation metric. The other four scheduled
calls did not execute, so no paired median, P95 or six-process performance
conclusion is admitted. A timeout does not distinguish an optimizer bug from a
very long finite compile. Controllable cache disabling also does not prove all
compiler state is cold.

The fixed-scene GPU stage was not started because the candidate failed its
bounded creation experiment. It has no rendering or dispatch qualification.
The largest SPIR-V function reduction therefore does not justify adopting this
Shadow noinline change; production remains unchanged. Keep the failed evidence
and investigate an independent memory-to-SSA cleanup of the original O1
General artifact next, without combining it with this boundary change.

The pure overlay/compiler tool passes **13 CPU evidence tests**. The analyzer
passes **15 tests**, including complete CFG golden values and missing targets
in unreachable blocks. The repository's 333 authored Slang files and both
generated shader overlays pass module ownership audits. The new analyzer also
reproduces all six real raw outputs' previously recorded structural metrics in
the analysis verification. No C++ rebuild is needed for these tool-only changes;
the existing Release Bench executes the ordinary native replay above.

Reproduce the compiler stages using fresh directories and a fresh explicit
`none`, mapped O1, debug-disabled compile-only request-146 reference:

```powershell
python -B Tools/Perf/ValidatePathTraceFunctionBoundaries.py prepare `
  --report build-release/<fresh-reference>/report.json --request-index 146 `
  --slang-root External/slang --output-dir build-release/<fresh-boundary>/preparation
python -B Tools/Perf/ValidatePathTraceFunctionBoundaries.py run `
  --preparation build-release/<fresh-boundary>/preparation `
  --spirv-val C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe `
  --output-dir build-release/<fresh-boundary>/compile
python -B Tools/Perf/ValidatePathTraceFunctionBoundaries.py analyze `
  --input-dir build-release/<fresh-boundary>/compile `
  --spirv-tools C:/VulkanSDK/1.4.363.0/Bin `
  --output-dir build-release/<fresh-boundary>/analysis
```

The ignored Native `Measure.py`/`Support.py`, archived helpers, actual commands,
full state/bytecode bindings, before/after source checks and optional NVIDIA
telemetry are retained with the failed batch. The first fresh API report used
the unspecified/default postprocess setting; it was not admitted. The separate
`function-boundary-reference2-20261013` report explicitly selects `none` and
is the actual compiler/Native reference. Neither that rejected configuration
nor the failed native batch was rewritten or restarted in place.

Run isolated variants in this order; inspect final device SPIR-V before timing:

| Variant | Boundary | Reason / risk |
| --- | --- | --- |
| Shadow-only | `Modules/OpenPBRPathTracing/OpenPBRPathTrace.slang`, `openPbrShadowTransmittance` | Shared transparent traversal under direct and environment lighting. Keeps deeply nested ray-query/texture/control flow outside the large path caller. Pass-by-value interior stack and pass parameters may create an ABI/temporary cost. |
| Fiber-only | `Modules/ScenePathTracing/SceneFiberPath.slang`, `shadeSceneFiber` | Isolates the entire uncommon material domain while keeping runtime semantics. Check actual ray-query legality/driver call behavior and fiber scenes. |
| Fiber evaluator | `RTXCRFiber.slang:23`, `RTXCRChiangPreparedClosure.evalProjected` | Narrows repeated Chiang evaluator expansion across three light paths; a smaller alternative to the whole integrator boundary. |
| BSDF eval/PDF | `OpenPBRClosure.slang:30-35`, `evalProjectedComponents` and `pdf` | Reuses costly pure BSDF subgraphs at the integrator facade. Large prepared-state arguments could trade inlining growth for stack/register overhead; capture ISA/local-memory evidence. |
| BSDF sample | `OpenPBRClosure.slang:60`, `sampleWeightedComponents` | Separates full mixture sampling after visibility. Keep failure-output handling and full mixture PDF unchanged. |

First run each boundary independently, then combine the winners. Do not begin
with every function noinline: resource/legalization functions may require
inlining and many small calls can cost more than they save. Inspect `OpFunction`
controls, maximum function size, branch/Phi counts and maximum call depth. If
Slang/legalization already removed a boundary, classify it as an ineffective
hint before claiming the driver was tested with a smaller caller.

## Custom SPIR-V passes: explicit second experiment

The installed SDK documentation (`External/slang/share/doc/slang/user-guide/08-compiling.md:378-384`)
says `-Xspirv-opt` appends passes to the selected optimization preset; it does
not replace O1's preset. The public API exposes
`CompilerOptionName::DownstreamArgs` with tool name plus one argument per line
(`slang.h:1109`). There is no proven public `UseDefaultPasses=false` switch in
this SDK. Adding `--ssa-rewrite` at O1 cannot undo exhaustive inlining.
[The current Slang downstream implementation](https://github.com/shader-slang/slang/blob/master/source/slang-glslang/slang-glslang.cpp)
corroborates preset-plus-extra-pass behavior and includes exhaustive inline in
the default sequence; it is an upstream reference, not a claim that current
master equals this installed binary.

Following the current Shadow timeout, the next independent experiment should
start from the **current O1 two-pass-cleaned Original General** artifact above,
without changing source optimization or function boundaries. Compare an exact
copy `C` with `S = C + --ssa-rewrite`. A later separate pair can compare
`L = C + --eliminate-local-single-block + --eliminate-local-single-store` with
`LS = L + --ssa-rewrite`; comparing only C with LS cannot isolate SSA's effect.
Do not initially add scalar replacement, an optimization preset or candidate-only
DCE. Optimization time, analysis time, native creation and GPU dispatch are
separate scopes.

The official [SSA pass implementation](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/ssa_rewrite_pass.cpp)
constructs reaching definitions and Phi nodes for eligible Function locals.
Its [variable eligibility logic](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/mem_pass.cpp)
conservatively excludes unsupported accesses and escaping pointers. A variable
count alone does not establish how many loads/stores can be promoted; SSA can
also leave original variables/stores or introduce Undef on paths without a
definition. These are upstream implementation references; the exact installed
`ef96ed76` source was not fetched, so pinned-binary outputs remain authoritative.

First validate each stage with the same Vulkan 1.4/scalar configuration and
check exact entry/resource/push/specialization ABI. Retain one function and zero
calls, inspect Function-specific loads/stores rather than total OpLoad alone,
and record Phi, Undef and precision decorations. Qualify the fixed General
HDR/temporal/seven-guide outputs at the existing tolerances, especially loop
state, BLEND/transmission and volume branches, before an expensive native pair.
Smaller memory-form SPIR-V or more Phi nodes do not predict Driver creation or
register-allocation gains. This C/S experiment has not yet been executed.

The reproducible alternate experiment starts with the already preserved O0
SPIR-V function graph and runs an explicit external optimizer sequence. Do not
promote the result because it is smaller: the previous O0 rendering experiment
failed the existing tolerance for two pixels. It must be newly qualified against
the current O1 reference. Keep an unmodified O0 arm to distinguish SSA cleanup
from function preservation, plus a pure `--compact-ids` control to distinguish
ID density from actual program transformations.

Installed tools are in `C:/VulkanSDK/1.4.363.0/Bin`, reported
`SPIRV-Tools v2026.4 v2026.4.rc2-0-gef96ed76`. One conservative candidate is:

```powershell
$opt = 'C:/VulkanSDK/1.4.363.0/Bin/spirv-opt.exe'
& $opt --target-env=vulkan1.3 --preserve-bindings --preserve-interface `
  --eliminate-dead-branches --eliminate-dead-functions `
  --eliminate-local-single-block --eliminate-local-single-store `
  --ssa-rewrite --simplify-instructions --eliminate-dead-code-aggressive `
  --cfg-cleanup --compact-ids input-O0.spv -o output-ssa-no-inline.spv
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe' --target-env vulkan1.3 `
  --scalar-block-layout output-ssa-no-inline.spv
```

This is a prepared candidate, not a completed Guides A/B. Avoid `-O`, `-Os`,
`--inline-entry-points-exhaustive` and unrestricted `--scalar-replacement=0` in
the first experiment. Add bounded scalar replacement later only if aggregate
opcodes remain a measured problem. Run validation after each pass during
diagnostics and keep that overhead separate from timed compilation. Descriptor
heap extension requests need tools/environment validation matching their exact
capabilities rather than assuming a mapped-descriptor command applies.

## Medium-risk changes if boundary hints are effective

1. Fuse BSDF projected value and PDF for the same direction at the adapter API,
   sharing GGX half-vector/Fresnel/distribution intermediates. Keep the same
   six-lobe mixture, normalization, event flags and numerical operation order
   where possible. Compare environment eval/PDF and all sample weights; avoid
   changing vendor snapshots for a first prototype.
2. Shorten prepared-closure live ranges around visibility by computing the
   values needed after traversal earlier. The query/single-sample branch in
   `OpenPBRPathVertex.slang:189-206` provides an existing narrow example. Extending
   it requires proving identical random-number consumption and debug return
   order. Measure actual registers, local-memory spills and ISA bytes; smaller
   SPIR-V alone is insufficient.
3. Derive a small validated scene capability mask (fiber, nested dielectric,
   custom families) and compile a general fallback plus only the needed feature
   variant. The guarantee must include secondary rays, stream arrivals, hot
   updates and shadows. A material weight equal to zero in one fixture is not
   sufficient reason to remove the feature from the general renderer.
4. If selective boundaries still leave a large machine-code body, separate
   primary guides/material evaluation and the secondary continuation kernel.
   Preserve authored primary hits and independent guide RNG, carry compact
   transport state, and budget queue/bandwidth/barrier costs. Existing
   `OpenPBRPathContinuation.slang` and `RayMaterialQueue.slang` provide relevant
   boundaries, but do not establish that full wavefront continuation is wired.
   [pbrt's primary implementation discussion](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
   explains the tradeoff: smaller kernels can allocate registers separately and
   improve coherence, but queued state adds memory traffic and launches. This
   supports a bounded experiment, not an assumed speedup on Metallic.

Each candidate should record final bytecode hash, unchanged workload/settings,
front-end/downstream/native creation times, pipeline executable code/register/
local-memory statistics when available, and pass timestamps from ordinary runs.
Qualify final HDR and all seven guides at current tolerances on opaque, mixed
MASK/BLEND/nested-dielectric, sky/miss, debug-normal and fiber workloads; include
secondary hits and scene changes for capability variants. Do not label O1→O0
as equivalent merely because almost every pixel agrees. A verified improvement
must reduce cold creation and retain acceptable GPU time and correctness.
