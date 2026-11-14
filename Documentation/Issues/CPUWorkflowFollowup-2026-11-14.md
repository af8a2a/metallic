# Windows hosted CPU workflow follow-up — 2026-11-14

## Scope and baseline

This is a separate D2 slice after `a9d4f8795`, whose locally executed CPU preset
passed 60/60. That historical result used the compatible Debug tree's existing
SDK options; it does **not** validate the new hosted SDK-off configuration.
See the [prior ownership follow-up](CIPresetFollowup-2026-11-14.md#source-only-ownership-follow-up--2026-11-14).
No renderer, shader, vendor, compiler-selection or skip-policy changes are made.

The [workflow](../../.github/workflows/CPU.yml) defines push, pull-request and
manual triggers on `windows-2025`, read-only contents permission, official
checkout/setup-Python/upload-artifact actions and a 120-minute job bound.
Checkout does not retain credentials or fetch every optional/licensed submodule.
Manual dispatch requires the workflow on the repository's default branch;
initial feature-branch validation relies on the push trigger.

The [lane script](../../scripts/RunCPULane.ps1) imports an x64 VsDevCmd environment
without printing the full environment, records actual tool/SDK versions, and
executes the existing `metallic-ci` configure/build/test presets. The hosted
variant additionally sets `METALLIC_ENABLE_NVIDIA_SDKS=OFF`,
`METALLIC_ENABLE_SPIRV_TOOLS=OFF` and `METALLIC_ENABLE_ISPC=OFF`. Build parallelism
is 2. It keeps native failures intact, including those piped to Tee-Object, and
rejects an empty CTest selection. Diagnostics and JUnit under `build-ci` are
uploaded with `if: always()`; an early failure may leave some files absent.

The existing [dependency bootstrap](../../CMakeLists.txt#L73-L149) initializes
only required submodules. [Slang setup](../../cmake/SetupSlang.cmake) defaults to
`External/slang`, matching the script's version probe. Its
[installer](../../scripts/InstallSlang.ps1) pins Slang 2026.18.2 and its archive
SHA256, verifies the staged compiler, and refuses to overwrite an existing
incomplete/mismatched installation. GDeflate/zstd use hash-pinned FetchContent
archives; GitHub/codeload network access is required. A separate Vulkan SDK,
optional NVIDIA SDKs, SPIRV-Tools and ISPC are not requested by this candidate.

## Source contracts and preflight

[Workflow source tests](../../tests/perf/TestCPUWorkflow.py) recognize the authored
YAML/PowerShell forms, **not** arbitrary YAML/PowerShell semantics. Their negative
mutations cover native status removal/tampering, presets, SDK switches, JUnit,
logs, tool facts, artifact retention and masked failures. The workflow runs them
before configuration; CTest also registers `MetallicCPUWorkflowTests` with
`build;tools;cpu`, timeout 30. The [preset contract](../../tests/perf/TestCIPreset.py)
now requires that job alongside the three existing ownership/benchmark entries;
each required name has unregistered, unlabelled and explicitly excluded variants.

Actual pre-registration red: 11 methods, one expected failure, exactly
`MetallicCPUWorkflowTests` missing from the configured model; see the
[red log](../../build/d2-round8-20261114/workflow-contract-red.log). This is not a
simulated assertion against an invented registration list.

Authenticated repository preflight found Actions enabled, allowed actions `all`,
no SHA-pinning requirement and the existing push credential's `workflow` scope.
The repository is public; its default branch is `modernization`. No repository
permissions or credentials were changed. Slang release API metadata lists the
63,200,096-byte Windows x64 archive and the same SHA256 as the installer; see
[metadata](../../build/d2-round8-20261114/slang-release-metadata.json). Metadata
availability is not proof that a fresh runner downloaded/extracted the archive.

## Execution and acceptance status

Actual source and registration checks:

- [Source-contract log](../../build/d2-round8-20261114/workflow-source-checks.log):
  **16/16 methods passed**, no skips. Independent initial review also parsed the
  YAML with already-installed PyYAML 6.0.3 `BaseLoader` and parsed the PowerShell
  AST without syntax errors; no parser dependency is added to the stdlib tests.
- After registration and compatible reconfiguration, the
  [configured contract](../../build/d2-round8-20261114/workflow-contract-green.log)
  passed **11/11**. Actual CTest
  [enumeration](../../build/d2-round8-20261114/workflow-preset-list.json) contains
  **61 unique jobs**, exactly one addition (`MetallicCPUWorkflowTests`) over the
  historical 60; no jobs were dropped and GPUPage remains excluded. Enumeration
  alone is not execution.
- [Isolated runtime fixtures](../../build/d2-round8-20261114/runtime-fixtures/REPORT.md)
  extract the real function/environment AST fragments rather than substitute
  implementations. The [final run](../../build/d2-round8-20261114/runtime-fixtures/results.log)
  passed all five child cases: native success continues; native stderr is logged
  and **expected exit 37** stops continuation; a spaced-path mock VsDevCmd imports
  a value containing `=`; **expected exit 41** is not hidden by `&& set`; real
  vswhere/VsDevCmd imports x64 host/target. Source SHA256 was unchanged.
- The discovery-only real child selected local MSVC tools **14.38.33130**,
  cl ProductVersion **14.38.33145.0**, SDK **10.0.26100.0**. It performed **no
  compilation** and does not establish a renderer compiler minimum. It is not
  the compatible build's explicitly selected MSVC 14.52.37025 environment.
- The initial mock harness incorrectly assumed CMD positional arguments retain
  unquoted `=`; its unexpected exit 92/verifier 1 was diagnosed and fixed in the
  fixture only. That initial failure is retained in the fixture report, not
  relabelled as an expected producer failure or a passing run.

Compatible local build and full selected execution (not the hosted variant):

```powershell
# In the existing x64 MSVC 14.52.37025 environment; retain all cache options.
cmake -S . -B build/d2-round6-20261114/clean-ci
cmake --build build/d2-round6-20261114/clean-ci --target MetallicSceneTests MetallicTaskTests MetallicDebugTests MetallicCtl MetallicRHITests MetallicShaderPipelineBench --parallel 2
ctest --preset metallic-ci --test-dir build/d2-round6-20261114/clean-ci --output-on-failure --no-tests=error --output-junit E:/metallic/build/d2-round8-20261114/workflow-cpu-lane.xml
```

- Compatible Ninja Debug reconfigure and all six targets: **exit 0**, retained
  Windows SDK **10.0.26100.0**, NRD/OpenUSD/NTC/Streamline/sccache OFF and existing
  NVIDIA/NVTX/Nsight Capture/Aftermath settings. No generator, compiler or SDK
  switches were changed for this local tree. See the
  [configure/build log](../../build/d2-round8-20261114/workflow-configure-build.log).
- Actual complete CPU invocation: **61/61, 90.06 seconds, exit 0**; see the
  [full log](../../build/d2-round8-20261114/workflow-cpu-lane.log) and
  [JUnit](../../build/d2-round8-20261114/workflow-cpu-lane.xml). The non-exclusive
  `tools` label contains **40 jobs**. This is a fresh complete invocation, not a
  union of partial reruns or the old 60-job result. No GTEST_/CTEST_ overrides
  were set; TEMP/TMP were the ordinary user temp directory, not an isolated retry.
- JUnit includes configured contract **11/11**, workflow contract **16/16**,
  ownership fixtures **5/5** and audit **392 files / 278 modules / 114
  implementation units**. Benchmark has **6 CPU passes / 4 GPU opt-in skips**,
  not ten GPU/runtime passes. Outer JUnit `skipped=0` does not erase these skips.
  Native Scene/Task/Debug stdout is truncated at 1024 bytes; no complete native
  case totals or no-inner-skip claims are inferred from that truncated output.

Independent final review approved this seven-file source slice and the new local
results; this is not hosted/whole-issue approval. All 24 local documentation
links/ranges and staged diff checks passed.

The initial workflow must be pushed before its hosted push run can exist. Until
an actual run is linked, Windows 2025 compiler/STL compatibility, first Slang
bootstrap, the SDK-off six-target build, full hosted tests and uploaded archives
remain **unverified**. A local SDK-enabled run cannot substitute for those gates.
Outer CTest JUnit jobs and inner native skips must be counted separately.

This slice does not establish GPU performance/correctness, editor validation,
non-Windows persistent-cache semantics or actual Nsight integration-on capture,
external-process and timeout behavior. Issue #63 remains open unless all of its
acceptance criteria, not just workflow source presence, are satisfied.
