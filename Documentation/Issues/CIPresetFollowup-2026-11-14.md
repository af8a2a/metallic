# #63 CPU CI preset selection and build closure — 2026-11-14

## Bounded delivery

This is a local CPU preset correction, not hosted CI deployment or full #63 acceptance.

- The previous test preset already selected `MetallicCtlArgumentsTests` through
  `debug`, but its build preset omitted `MetallicCtl`.
- Adding `tools` also selects the CPU RHI CLI matrix and shader-pipeline benchmark
  regression. Their executables require explicit `MetallicRHITests` and
  `MetallicShaderPipelineBench` build targets. The latter is `EXCLUDE_FROM_ALL`.
- The preset now selects `scene|task|debug|build|tools|skip-policy`, preserves the
  existing required-Python setting, and builds the six required executable targets.
- `MetallicGPUPageTests` has both `scene` and `rhi` labels and contains real Vulkan
  decompression tests. It is explicitly excluded by name from this CPU test
  preset. The whole mixed bundle, including its CPU subtests, remains outside
  this lane. Its registration and separate GPU invocation remain unchanged.
- The tools suites exercise Python fixtures, source/report contracts and native
  CLI/compile-only behavior. The shader benchmark's four `--gpu` opt-in methods
  remain skipped by default; this is not GPU performance evidence.

See [preset definitions](../../CMakePresets.json),
[CTest registration](../../tests/CMakeLists.txt),
[perf registration](../../tests/perf/CMakeLists.txt), and the
[build guide](../Build.md#daily-development).

## Regression contract and red evidence

[The new CPU contract](../../tests/perf/TestCIPreset.py) checks required labels,
exact target names, required Python, GPU-label expansion, and the mixed page
bundle exclusion. Negative mutations cover each required label/target, optional
Python, broad GPU selection, missing generated commands, and direct/CMake/Python
executable references (including POSIX paths without `.exe`).

The integration branch enumerates the configured tree with
`ctest --show-only=json-v1`; it does not execute nested tests. It applies the
preset's include-label and exclude-name filters, cross-checks the selected name
set against official CTest preset enumeration (with `--test-dir` overriding the
preset build directory), then checks repository native executable references in
direct commands, CMake `-D...EXECUTABLE` arguments and Python `--executable`
arguments against build-preset targets.

Observed red steps before the final correction:

1. Baseline: 8 contract methods, 2 failures. The real generated model identified
   `MetallicCtlArgumentsTests: metallicctl`; the policy also rejected missing
   `tools` and missing Ctl/RHI targets.
2. Adding only `tools`: 2 failures remained; the real model identified all three
   missing executables: Ctl, RHI tests, and shader-pipeline benchmark.
3. Manual enumeration exposed an additional CTest behavior: an unbuilt direct
   executable can have **no `command` field** in JSON. The checker must fail this
   case, not silently omit it. This led to an explicit missing-command negative
   fixture and the mixed GPU-page selection regression, for 10 methods total.

## Local validation

Evidence links under `build/` below are local ignored artifacts, not uploaded or
hosted CI archives. Preserve each invocation separately.

### Initial compatible Release attempt — not green

The scoped build/selection contract and tools regressions passed; the original
complete CPU invocation **did not pass** in the default local output environment.

- All six executable targets built successfully, exit 0. See the
  [build log](../../build/d2-round6-20261114/build.log).
- The configured contract ran **10/10 methods successfully, no skips**, including
  the official preset selected-name comparison and generated-command closure.
  See the [contract log](../../build/d2-round6-20261114/preset-contract.log).
- Official preset enumeration selected **58 jobs** in the compatible tree;
  `MetallicGPUPageTests` was not selected. Enumeration alone is not execution.
- The actual full preset run returned exit 8: **56 passed, 2 failed**, 72.40 s.
  See its original [log](../../build/d2-round6-20261114/cpu-lane.log) and
  [JUnit](../../build/d2-round6-20261114/cpu-lane.xml).
  - `MetallicSceneTests`: 172 inner tests, **134 passed, 29 failed, 9 skipped**.
    Failures report denied writes/copies/directory creation under existing
    source-tree build outputs and the default system temporary directory.
    Skips cover unavailable USD/large-scene inputs. Scene was already selected
    by the old preset; this diff does not modify its implementation or exclude it.
  - `MetallicShaderPipelineBenchTests`: 10 inner methods, **3 passed, 3 failed,
    4 GPU methods skipped**. All three failures report denied directory creation
    under the default system temporary directory, before compile-only checks.
- Read-only integrity-label inspection found Low labels on both native
  executables, a Medium shell token, and no Low label on the failing `build` and
  system-temp roots. This supports an output-permission explanation, but is not
  proof that every Scene failure has exactly the same cause. No ACL was changed.
- A separate diagnostic regression set process-local `TEMP`/`TMP` to a new CPU
  temporary directory inside the existing build tree: **37/37 tools CTest jobs
  passed**, exit 0, 55.19 s. See its separate
  [log](../../build/d2-round6-20261114/tools-isolated-temp.log) and
  [JUnit](../../build/d2-round6-20261114/tools-isolated-temp.xml).
  This includes the Ctl/RHI CPU CLI wrappers and configured closure contract;
  the benchmark passed **6 CPU methods with 4 explicit GPU skips**. This run did
  not rerun Scene and is not a default-environment or full-58 pass. Results from
  different invocations must not be combined into a fictitious all-green run.
- Before the native build, the independent Python-only tools subset passed
  **33/33 CTest jobs**, exit 0, 57.24 s. It excluded the Ctl/RHI/benchmark wrappers
  and configured closure contract and includes three pre-existing skip-policy
  jobs. The fixture-only contract separately passed 9 with 1 configured-integration
  skip; this historical run is superseded by the configured 10/10 result above.
- `cmake --list-presets=all` and `git diff --check` passed. Independent read-only
  review approved the source changes; all execution counts here are root runs.

Key execution commands, after initializing the matching VS developer shell:

```powershell
cmake --build build-release --target MetallicSceneTests MetallicTaskTests MetallicDebugTests MetallicCtl MetallicRHITests MetallicShaderPipelineBench --parallel 2
python -B tests/perf/TestCIPreset.py --build-dir build-release -v
ctest --preset metallic-ci --test-dir build-release --output-on-failure `
  --output-junit E:/metallic/build/d2-round6-20261114/cpu-lane.xml

# Separate diagnostic invocation, process-local environment only; no ACL edits.
$tempRoot = 'E:\metallic\build-release\d2-round6-20261114-temp'
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
ctest --test-dir build-release -L '^tools$' --output-on-failure `
  --output-junit E:/metallic/build/d2-round6-20261114/tools-isolated-temp.xml
```

### Clean Debug follow-up — full selected regression green

The new `build/d2-round6-20261114/clean-ci` directory configured successfully
with MSVC 14.52/Ninja, Debug, tests and required Python enabled,
NRD/OpenUSD/NTC/Streamline disabled, and sccache off; see the
[configure log](../../build/d2-round6-20261114/clean-ci-configure.log).
The same six targets built successfully (exit 0); see the
[clean Debug build log](../../build/d2-round6-20261114/clean-ci-build.log).
NVTX Events and Aftermath were enabled; Capture detected SDK 0.9.2 from the
installed Nsight Graphics 2026.4.0. This is not an all-optional-integrations-off
configuration or Nsight integration execution.

- The complete native preset invocation passed **58/58 CTest jobs, exit 0,
  99.95 s**, with default process `TEMP`/`TMP` and no GTEST environment overrides.
  See the [full log](../../build/d2-round6-20261114/clean-ci-cpu-lane.log) and
  [JUnit](../../build/d2-round6-20261114/clean-ci-cpu-lane.xml). This is a single
  complete invocation, not a union of the earlier red Release/tools subsets.
- Outer JUnit reports zero failed/skipped CTest jobs; this does **not** mean
  every inner case ran. The configured preset contract passed **10/10**, and
  the benchmark passed **6 CPU methods with 4 explicit opt-in GPU skips**.
- Following that complete run, separate native GoogleTest XML invocations in
  the clean tree's `tests` working directory also returned exit 0:
  [Scene XML](../../build/d2-round6-20261114/clean-ci-scene-gtest.xml) records
  **172 total, 163 passed, 0 failed, 9 skipped**;
  [Task XML](../../build/d2-round6-20261114/clean-ci-task-gtest.xml) **34/34**;
  [Debug XML](../../build/d2-round6-20261114/clean-ci-debug-gtest.xml) **27/27**.
  Scene skips remain USD-disabled and explicit large-scene/large-image opt-ins,
  not new passes. These XML files preserve case-level details that the default
  CTest successful-output truncation can omit; they are separate reruns, not
  additional jobs in the 58-job count.
- Read-only `icacls` on the fresh Scene executable displayed no explicit Low
  mandatory label. No ACL, owner or integrity label was changed. This observation
  does not establish the sole cause of all old Release failures.

Commands from the repository root after initializing the VS developer shell:

```powershell
cmake --preset metallic-ci -B build/d2-round6-20261114/clean-ci
cmake --build build/d2-round6-20261114/clean-ci --target MetallicSceneTests MetallicTaskTests MetallicDebugTests MetallicCtl MetallicRHITests MetallicShaderPipelineBench --parallel 2
ctest --preset metallic-ci --test-dir build/d2-round6-20261114/clean-ci --output-on-failure `
  --output-junit E:/metallic/build/d2-round6-20261114/clean-ci-cpu-lane.xml
```

**Commit gate:** the full selected CPU regression requirement is met, and
independent read-only final review approved this bounded source/evidence scope
before committing and pushing. This does not waive whole-issue acceptance.

The original compatible `build-release` attempt retained its original
Release/MSVC 14.52/Ninja/Windows SDK 10.0.26100 settings and feature choices
(including NRD/OpenUSD off). It was not a clean Debug preset run and has not been
repaired or rerun to green by the new Debug result. Neither its configuration,
file permissions nor existing GPU artifacts were modified.

## Source-only ownership follow-up — 2026-11-14

The 58-job result above did not include the source ownership audit: its original
`shader;architecture` labels do not match the CPU preset. This follow-up adds
`tools;cpu` to both `MetallicShaderModuleOwnershipAudit` and its five-method
fixture regression `MetallicShaderModuleOwnershipTests`, retaining their original
labels. These Python checks read repository Slang sources or temporary fixtures;
they do not invoke Slang, create a Vulkan device, or need another build target.
The six-target executable closure and GPU-page exclusion are unchanged.

The configured preset contract now requires the ownership audit, its fixture
regression, and `MetallicShaderPipelineBenchTests` to be registered **and selected**.
Synthetic negative mutations remove each registration, remove its selectable
labels (leaving `shader;architecture;cpu`), or exclude it by name. The pre-fix
[configured red run](../../build/d2-round6-20261114/cpu-audit-contract-red.log)
ran 11 methods with one failure, identifying exactly the two missing ownership
jobs. After the registration update and reconfigure, the
[configured contract](../../build/d2-round6-20261114/cpu-audit-contract-green.log)
passed **11/11, no skips**; this alone is not the full lane regression.
Official preset enumeration selects 60 jobs, including all three required checks
and excluding `MetallicGPUPageTests`. The separate
[direct source audit](../../build/d2-round6-20261114/cpu-audit-source-check.log)
returned exit 0: **392 Slang files, 278 modules, 114 implementation units**.
This checks ownership syntax, not shader compilation or GPU behavior.

The same compatible Debug tree was reconfigured and all six selected targets
built or remained up to date, exit 0; see the
[configure/build log](../../build/d2-round6-20261114/cpu-audit-configure-build.log).
The developer shell still selected MSVC 14.52.37025 and Windows SDK 10.0.26100.0;
compiler, generator and optional SDK choices were not changed.
The subsequent **complete selected CPU run passed 60/60, 102.13 s, exit 0**, with
default TEMP/TMP and no GTEST/CTEST environment overrides. This is a new invocation,
not a reinterpretation of the historical 58-job results. See its
[full log](../../build/d2-round6-20261114/cpu-audit-cpu-lane.log) and
[JUnit](../../build/d2-round6-20261114/cpu-audit-cpu-lane.xml).

The JUnit records the ownership fixture **5/5**, the real source audit's
**392/278/114** totals, the configured contract **11/11**, and benchmark
**6 CPU methods passed / 4 explicit opt-in GPU skips**. Its outer `skipped=0`
counts CTest jobs, not native cases. Successful native Scene/Task/Debug output is
still truncated at 1024 bytes; the earlier separate native XML evidence above
is not a case-level rerun of this new 60-job invocation.

Independent read-only final review approved this four-file follow-up after
checking the full logs/JUnit, exact two-job selection delta, historical evidence
boundaries and 89 local documentation links. This approval does not cover the
separate hosted-workflow candidate or whole-issue acceptance.

## Acceptance balance

#63 remains open. The companion [editor/Nsight source matrix](EditorValidationMatrix-2026-11-14.md)
inventories all 18 editor `GTEST_SKIP` sites and selected registration/regex-skip
boundaries; it adds no runtime passes, cross-platform implementation, or skip
allowances. It is not a complete repository-wide platform/skip baseline.
This CPU slice does not deploy a workflow or hosted runner, archive hosted JUnit,
prove the Nsight integration-on collection timeout/runtime path, or select every
shader/perf suite. The ownership audit and its fixture regression are explicitly
included by the follow-up above; other `shader;perf;cpu` suites without a matching
CPU preset label remain outside this lane. The benchmark CPU regression is not
a replacement for GPU collection.
A future workflow must use the matching test preset/filters rather than assume
an unfiltered invocation of every registered test has this build closure.

No renderer/shader implementation, permission, SDK/compiler selection, external
submodule content, or existing GPU-artifact location is changed by this slice.
