# Direct compute resource root experiment

`ScenePathTracePass.experimentalInlineComputeRoot` defaults to `false`. Setting it to
`true` selects a separate 24-byte push-constant ABI for the unsplit compute path tracer.
Realtime lighting, visibility deferred, ray pipelines and primary-guide splitting reject
this experimental property. Production pipeline assets keep their existing defaults.

The root contains `resources` at offset 0 and `constants` at offset 12. Both are 12-byte
descriptor-relative spans. Their buffers, descriptor resolution and optional-resource
sentinels remain unchanged. `ComputeResourceParameters.h` owns the shared wire layout
and asserts the CPU size, alignment and offsets.

The default shader imports `ParameterRoot`: a 12-byte push span points to the uploaded
24-byte root. The candidate uses the global compiler definition
`METALLIC_INLINE_COMPUTE_ROOT=1` and imports `InlineComputeRoot`: the same 24-byte root
is pushed directly. A compilation importing the legacy root in this mode is rejected.
The request factory selects the same entry point and specialization constants before
the transport definition is appended.

`getResourceParameterField` retains packet-size, field-size, offset-range and alignment
checks in both modes. The legacy short-root zero-return contract is preserved. The
candidate has no external root span to truncate; its CPU encoder must supply the
complete root. It still returns zero for a short resource packet or invalid field offset.
This experiment does not add forced inlining or load the whole scene resource packet.

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
establish GPU speedup. Keep this experiment disabled unless the intended production
workload supplies supporting correctness and performance evidence.

## Initial measurement, 2026-10-17

On an RTX 5070 Ti, the three fixed 512×512 OpenPBR fixtures used 2 samples,
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

This is exploratory evidence without a separate confirmation stage. The runner's
decision remains `inconclusive` for production adoption and the property stays off.
SHARC runtime rendering, full production scenes and end-to-end frame rate were not
measured. Local sealed evidence is in `build/inline-compute-root-ab-20261017-v2`;
the reviewed plots and HDR previews are in
`build/inline-compute-root-20261017/report/Report.html`. The first attempt is retained
separately because its report parser incorrectly expected an artifact-only status field;
the renderer passed, and that attempt is excluded from these statistics.
