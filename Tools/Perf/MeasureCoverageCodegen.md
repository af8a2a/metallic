# Coverage native code generation A/B

`MeasureCoverageCodegen.py` runs the opt-in `material_coverage_codegen_benchmark`
RHI fixture in independent processes. Both arms use the same binary, authored
Coverage IR, instance parameters, UV distribution, dispatch and output layout.
`legacy` compiles the frozen interpreter in `tests/rhi/shaders/LegacyCoverageOracle.slang`;
`native` calls the production `CoverageProgram` and generated Slang module.
Production shaders have no legacy backend switch.

This is a **Coverage evaluator throughput microbenchmark**. It excludes material
texture sampling, rasterization, ray traversal, lighting and presentation, and
does not establish an end-to-end scene improvement. The `control` workload has no
custom Coverage; `simple` has one small expression; `complex` has five nested
operator groups within the normal depth-24 authoring limit; `mixed` interleaves eight programs; `edge` exercises all 18 operations,
large UV inputs and normalization fallback. `edge` is a correctness stress case,
not representative material content.

Build `MetallicRHITests` in the compatible configured Release tree first. Do not
change its compiler, generator or SDK configuration for this experiment. Run GPU
experiments serially after other rendering/profiling work has completed:

```powershell
python -B Tools/Perf/MeasureCoverageCodegen.py run `
  --exe build-pass-stages-nrd/tests/MetallicRHITests.exe `
  --output build/coverage-codegen-ab --blocks 2
```

The default is two ABBA blocks: eight independent processes, four pairs. Each
process executes five cases at 512 x 512 with four evaluations per pixel, 16
warmup dispatches and 32 measured dispatches. One final capture is outside the
timing window. No measured frames are discarded. The statistic for one process
is the median dispatch GPU time; pairwise reduction is `100 * (1-native/legacy)`.
The report shows individual process results, frame trajectories, paired gains
and a two-sided Student t confidence interval across process pairs. Frame count
is never treated as the independent sample count.

Each run preserves logs, status, workload JSON, raw RGBA32F results, the exact
compiled SPIR-V, generated source and timing frames. The manifest seals raw
evidence; the report is outside that seal. Repeated same-arm images and SPIR-V
must be identical. Custom workloads must have different legacy/native SPIR-V
hashes; the no-program control may match. Native/legacy opacity sums must agree within absolute 2e-5;
the per-repeat acceptance bitmask and alpha-dependency flag must agree exactly.
Nonfinite pixels, missing/zero GPU durations, failed/skipped tests, changed input
hashes and incomplete process groups invalidate evidence. A failed correctness
gate remains visible in the report and cannot be called a performance win.

Shared shader/driver caches are retained. Compilation and pipeline setup are
outside the GPU timestamp scope, and each case warms independently. Validation
is disabled for timing; run the production correctness regression separately:

```powershell
build-pass-stages-nrd/tests/MetallicRHITests.exe `
  --gtest_filter=*material_coverage_compiler*:*material_coverage_winner_shadow_ray*:*material_coverage_slab_winner_shadow_ray* `
  --rhi-validation --output-dir build/coverage-codegen-correctness
```

Those tests check 11 live editing states, including adding Coverage-only code,
updating parameters, changing nonzero Coverage-only executable code while the
Surface key stays fixed, clearing it, shared texture alpha and BLEND handling.
They compare actual visibility-buffer raster winners with production RayQuery
and shadow-candidate evaluators. The microbenchmark does not replace these tests.

Re-verify evidence or generate a new report without launching GPU work:

```powershell
python -B Tools/Perf/MeasureCoverageCodegen.py verify --input build/coverage-codegen-ab
python -B Tools/Perf/MeasureCoverageCodegen.py report `
  --input build/coverage-codegen-ab --output build/coverage-codegen-report-review
```

Keep clocks unmodified and record competing workloads. The cooperative lock
only coordinates participating runners; it does not prove system-wide GPU
exclusivity. CI overlap and small process counts can produce an inconclusive
result. The report requires NumPy and matplotlib and records their versions.
