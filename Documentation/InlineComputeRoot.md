# Direct compute resource root

`ScenePathTracePass.inlineComputeRoot` is a default-enabled feature for the ordinary,
unsplit compute path tracer, selecting a separate 24-byte push-constant ABI. An explicit
`false` restores the buffered root for A/B comparisons. Realtime lighting, visibility
deferred, ray pipelines and primary-guide splitting default to `false` and reject an
explicit `true`. Preparation, compilation and execution use the same selection helper.
Existing pipeline assets inherit this policy without adding an explicit property.
The old `experimentalInlineComputeRoot` name remains a compatibility alias for authored
graphs and archived experiments. An explicit boolean `inlineComputeRoot` takes
precedence; new configurations and the A/B runner use the non-experimental name.
The inline path retains its existing fail-closed compilation policy: an initial
material compilation failure rejects graph compilation instead of displaying the
legacy error material. Shader reload failures likewise reject the replacement graph.

The root contains `resources` at offset 0 and `constants` at offset 12. Both are 12-byte
descriptor-relative spans. Their buffers, descriptor resolution and optional-resource
sentinels remain unchanged. `ComputeResourceParameters.h` owns the shared wire layout
and asserts the CPU size, alignment and offsets.

The generic shader ABI still defaults to `ParameterRoot`: a 12-byte push span points
to the uploaded 24-byte root. The supported path tracer appends the global definition
`METALLIC_INLINE_COMPUTE_ROOT=1` and imports `InlineComputeRoot`: the same 24-byte root
is pushed directly. A compilation importing the legacy root in this mode is rejected.
The request factory selects the same entry point and specialization constants before
the transport definition is appended. `Core` keeps its macro default at `0`; other
shaders and the legacy CPU API retain the buffered-root default.

`getResourceParameterField` retains packet-size, field-size, offset-range and alignment
checks in both modes. The legacy short-root zero-return contract is preserved. The
direct mode has no external root span to truncate; its CPU encoder must supply the
complete root. It still returns zero for a short resource packet or invalid field offset.
Direct transport does not add forced inlining or load the whole scene resource packet.

## Host-guaranteed scene packet

For the supported inline compute path tracer, `trustedSceneParameters` defaults to
`true`. Explicit `false` selects the checked scene accessors without changing the
24-byte root ABI. Enabling it with buffered roots, primary-guide splitting, realtime
lighting, visibility deferred or ray pipelines is rejected.

Path-tracing code uses the fixed-type `getSceneResourceParameters()` and
`getSceneResourceParameterField<TField>(byteOffset)` entry points. Their default
compilation delegates to the generic checked APIs. Only the path tracer's material
request supplies `METALLIC_TRUSTED_SCENE_RESOURCE_BYTES=sizeof(SceneResourceParameters)`.
The material compiler validates that declaration against the shared CPU structure,
the encoder's packet size and `InlinePush` transport before compiling or looking up
an executable. Slang asserts the same shared structure size. The request definition
participates in shader/material cache identity and pass recompile checks.

The encoder uploads the complete packet and sets its word count; failed allocation,
encoding or ABI validation prevents dispatch. Trusted scene loads therefore omit the
packet-length comparison. Field size, offset range and alignment checks remain, as do
optional-resource sentinels and resource-specific bounds. Generic
`getResourceParameters<T>` and `getResourceParameterField<TPacket,TField>` remain
unchanged, including short-packet zero returns inside trusted compilations.

Trust is a source-compiled, fixed-Scene contract, not a property inferred from an
arbitrary SPIR-V blob. A handled shader artifact override is rejected for a trusted
material request; override experiments must select `trustedSceneParameters=false`.
Low-level raw kernel/root construction remains the caller's responsibility and is
not an alternative production path for bypassing the encoder contract.

`trusted_scene_resource_contract_and_bounds` tests host declaration failures, the
checked short-packet contract under both compilation modes, valid trusted fields,
dynamic offsets and sentinels on mapped/native descriptors. The existing root/field
tests continue to cover legacy short roots and other generic packet types.

For a same-root comparison, use `Tools/Perf/MeasurePathTraceInlineRoot.py run
--experiment trusted-scene-parameters` with the usual executable, cases and new output
directory arguments. Both arms explicitly use inline roots; only scene packet trust
changes. The runner checks actual shader identities, unchanged non-target pipelines,
HDR/temporal/guide equivalence, and independent-process AB/BA/AB timing pairs.

The CPU encoder validates member layout and bindings, encodes the complete resource
packet, retains its allocations, and uses the existing `ParameterWriter` inline
transport. Encoder ABI interning, material executable caching and prepared dispatch
compatibility include the transport. A graph recompile is required when it changes;
execution rejects a property/compiled-ABI mismatch. Vulkan checks push capacity.

## Validation

Build `MetallicRHITests` before running these focused tests with validation enabled:

- `compute_resource_root_transport_encoding_and_abi`: encoding, ABI separation,
  indirect constant offsets, upload bytes and resource retention.
- `compute_resource_root_transport_material_cache`: same-code/different-transport
  cache separation and incompatible direct/indirect dispatch rejection.
- `inline_compute_root_shader_fields_and_bounds`: mapped/native descriptors, both
  transports, complete/short packets, dynamic offsets, oversized fields, sentinels,
  nonzero span offsets and constants, compared against CPU expectations.
- `named_resource_field_views_and_bounds`: existing legacy malformed-root contract.
- `compute_kernel_prepared_standalone_batch`: existing prepared dispatch retention.

Run `python -B Tools/CheckShaderModules.py` after shader structure changes.

`Tools/Perf/MeasurePathTraceInlineRoot.py` runs a validation-on A/B correctness pair
followed by three independent-process timing pairs in AB/BA/AB order. It uses production
OpenPBR path-tracing cases, preserves scene and lighting identities, verifies exported
pipeline recipes and compares HDR and guide readbacks. Timings cover graph, pass and
the path-trace shading interval; they do not measure end-to-end application performance.
Use a new output directory and the runner's `verify` command to recheck evidence.

Static SPIR-V counts establish whether the indirection was removed; they do not
establish GPU speedup. The default policy does not extend the measurement scope
documented below; use explicit `false`/`true` pairs to qualify additional workloads.

### Single-frame Nsight diagnostic

The `material_phase0_baseline` case file accepts an optional top-level
`nsightTraceFrame` (zero-based). Use one ordinary case, validation disabled, and
`0 < nsightTraceFrame < warmupFrames`; neither that frame nor the preceding frame
may perform a readback. Artifact overrides and asset round-trip cases are rejected.
Launch the test under the official Nsight GPU Trace CLI with
`--start-with-ngfx-sdk --stop-with-ngfx-sdk --real-time-shader-profiler --auto-export`.
Create the output directory before launching. The test does not inject Nsight itself.

The test selects optimized capture symbols and requires O1. It starts the SDK range
after the preceding frame has drained and stops it after the chosen render drains.
The report is marked `diagnostic: true` and `performanceEligible: false`;
`sdkTrace.complete` confirms SDK calls, while the caller must separately verify the
fresh trace and exports. A successful SDK stop does not prove export or application
shutdown success. Capture symbols are restored after the test.

The 2026-10-17 default-on check omitted the property, confirmed the actual 24-byte
pipeline recipe, and matched all 16 Mixed1 readbacks against the prior explicit-on
run (maximum absolute error 0). One optimized-symbol GPU Trace was exported with
Nsight 2026.3.1; its readbacks also matched exactly. Local evidence resides in
`build/inline-root-nsight-20261017/`, with the accepted capture in `capture-v4`.
The exported marker table attributes only 1 ns to `Path trace shading` and the
long interval to `Stream End`; those marker durations are not accepted as pass
timings. This diagnostic is not a matched trace A/B or proof that all memory waits
have disappeared. Source/SASS attribution must be inspected in Shader Profiler;
automatic XLS export does not provide per-source-line stall tables.

## Initial measurement, 2026-10-17

This measurement was made while the property defaulted to `false`; the runner set
both transport values explicitly. On an RTX 5070 Ti, the three fixed 512×512 OpenPBR
fixtures used 2 samples,
maxDepth 12, SHARC off, 16 warmup frames and 32 timed frames per process. One
validation-on correctness pair preceded three validation-off pairs (AB/BA/AB).
Each percentage below is the median of the three candidate/baseline process-median
ratios, expressed as a time change; the range is observed variation, not a confidence
interval.

| Fixture | Path trace shading time change | Observed paired range |
| --- | ---: | ---: |
| Opaque1 | -15.71% | -18.58% to -14.02% |
| Mixed1 | -17.81% | -18.56% to -17.20% |
| Sky | -2.72% | -3.91% to -1.04% |

All eight processes passed their rendering tests. Final HDR, eight temporal captures
and seven guides per case were numerically identical across all runs (maximum absolute
error 0). Per-case recipes confirmed the 12/24-byte root layouts, unchanged entry
points and specialization values, and identical non-target pipelines. The current
verifier also passed with input hashes rechecked. The five focused RHI tests and all
132 default shader warmup requests passed; direct-root standard, SHARC Update/Query
and single-sample variants additionally compiled and passed SPIR-V validation.

This is exploratory evidence without a separate confirmation stage. At the time,
the runner's decision was `inconclusive` for production adoption and the property
remained off by default. It was subsequently enabled by user request for the supported
compute path; this policy change does not alter that measurement decision.
SHARC runtime rendering, full production scenes and end-to-end frame rate were not
measured. Local sealed evidence is in `build/inline-compute-root-ab-20261017-v2`;
the reviewed plots and HDR previews are in
`build/inline-compute-root-20261017/report/Report.html`. The first attempt is retained
separately because its report parser incorrectly expected an artifact-only status field;
the renderer passed, and that attempt is excluded from these statistics.

## Trusted scene packet measurement, 2026-10-17

This comparison fixes `inlineComputeRoot=true` in both arms and changes only
`trustedSceneParameters`. It uses the same RTX 5070 Ti, three 512×512 fixtures,
2 samples, maxDepth 12, SHARC off, 16 warmup frames and 32 timed frames. A
validation-on correctness pair precedes three validation-off process pairs in
AB/BA/AB order. Percentages are paired process-median time changes; ranges are
the three observed changes, not confidence intervals.

| Fixture | Path trace shading time change | Observed paired range |
| --- | ---: | ---: |
| Opaque1 | -19.93% | -24.35% to -19.33% |
| Mixed1 | -18.55% | -18.78% to -17.49% |
| Sky | +10.37% | +10.04% to +10.80% |

Sky regresses in all three pairs (about 0.270 ms to 0.298 ms for shading);
removing the checks is not a universal speedup. Its graph-time change is +3.00%,
while Opaque1 and Mixed1 graph times change by -19.50% and -18.42%. These are
fixture GPU intervals, not end-to-end application frame rates. The runner's
decision remains `inconclusive`: three exploratory pairs, no separate confirmation
stage or predeclared acceptance threshold. The supported path tracer enables the
new contract by default; explicit `trustedSceneParameters=false` retains a
same-root checked comparison and permits artifact override experiments.

All eight rendering processes passed. Every final HDR, eight temporal captures
and seven guides per case matched numerically with maximum absolute error 0.
Recipes confirmed identical root size, entry points, specialization and non-target
pipelines, with different target SPIR-V. Current-input verification passed. Six
focused validation-enabled RHI tests also passed, including short generic packets
inside trusted compilations and rejection of artifact overrides. Twenty static
shader compilations and SPIR-V validations passed; SHARC variants were compiled
but not rendered in this experiment. Full production scenes remain unmeasured.
The actual correctness-run target SPIR-V recorded by native PSO recipes has
477 `resources.count` loads in checked mode and zero in trusted mode; static
StorageBuffer load counts remain 2177 in both. Both recorded binaries pass
`spirv-val`. These instruction counts prove elimination of the count reads,
not elimination of all runtime memory stalls.

Sealed evidence: `build/trusted-scene-parameters-ab-20261017/`. Reviewed plots,
readback previews and reproduction script:
`build/trusted-scene-parameters-20261017/report/Report.html`.
