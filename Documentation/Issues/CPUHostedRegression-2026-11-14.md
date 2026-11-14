# Windows CPU hosted failure and bounded repair — 2026-11-14

## Actual first hosted result

The first [Windows CPU run](https://github.com/af8a2a/metallic/actions/runs/43554479526)
ran commit `026b2a782ba983031fff9003859a92223e00f172` on the feature branch.
It **failed: 46 PASS / 15 FAIL / 61 outer CTest jobs**, 140.42s CTest time;
its approximately 29m04s job time is not a test or renderer performance result.
The downloaded JUnit records 61 testcases, failures=15, skipped=0. The latter
counts outer jobs only, not inner GoogleTest skips.

Checkout, setup-python and the source contract passed. Actual configure,
verified Slang installation and all six CPU build targets succeeded. Slang
2026.18.2 archive SHA256 was
`747602aec6b3623658d55fea87492d71828e15d16802d7941204fde418ceee8e`.
Diagnostic preservation also succeeded; this is not a successful complete lane.

Observed runner/tool facts, not assumptions from the runner label:

- Image `win25-vs2026 / 20261103.298.1`.
- VS18 Enterprise VC tools `14.51.36231`; cl product version `14.51.36260.0`.
- Windows SDK `10.0.26100.0`; Python **3.12.10**.
- Debug, Ninja, NVIDIA SDK umbrella/SPIRV-Tools/ISPC OFF, build parallelism 2.

The [diagnostic artifact](https://github.com/af8a2a/metallic/actions/runs/43554479526/artifacts/14098018587)
is `cpu-windows-diagnostics`, 76,976 B, digest
`sha256:1500356c729c9a71982c802aedef859eae8eeab59445d3ed8bf954f62424b066`.
It contains selected tool-version and configure/build/test logs, JUnit and CTest
last-test/failed-test logs. Root downloaded and inspected these, including all
15 failing outer names; this analysis is not based only on the Actions annotation.

## Three observed failure families

### Python dependencies: twelve outer jobs

The hosted interpreter lacked numpy, matplotlib and psutil. Checking all 38
selected Python entrypoints also found the initially masked SciPy dependency:
the performance verifier actually exercises `scipy.stats.t`. The selected
`tools` label comprises those 38 commands plus two CMake/native CLI contracts.
A conservative local AST import closure covered 73 Python files.

The dedicated [CPU manifest](../../requirements-cpu.txt) pins only numpy 2.2.6,
matplotlib 3.10.3, psutil 7.0.0 and scipy 1.15.3. All four were checked against
non-yanked Python3.12-compatible Windows x64 wheels; psutil uses abi3.
Pillow is a mandatory matplotlib dependency. Terrain/OpenEXR and unrelated
optional GPU tooling were not installed. **These are direct pins, not a full
transitive/hash lock.** The actual resolved set is recorded, not inferred.

[The runner script](../../scripts/RunCPULane.ps1) resolves one executable after
VS environment import, checks Python3.12, uses that same executable for pip
install/check/list, and explicitly supplies it as CMake `Python3_EXECUTABLE`.
All new pip logs remain in the workflow's already-archived diagnostics directory.
The [source contract](../../tests/perf/TestCPUWorkflow.py) grows from 16 to 23
methods with negative mutations for omitted installation, wrong interpreter,
changed pins, omitted logs and install/configure order. It is still an authored
source recognizer, not a general YAML/PowerShell parser or a hosted execution.

### Windows short/long aliases: two outer jobs

The runner's temporary root used `RUNNER~1`, while resolved descendants used
`runneradmin`. [The ownership checker](../../Tools/CheckShaderModules.py) now
normalizes its public `check(root)` input once, before collecting resolved files;
the CLI no longer repeats that normalization. Ownership rejection is unchanged.

[The workload fixture](../../tests/perf/TestWorkloadCase.py) now resolves the
repository root at its producer, before patching the tool's global ROOT. The
production [asset closure](../../Tools/Perf/WorkloadAssets.py) already establishes
that contract and is **unchanged**. Added cases still reject graph, glTF and
stream dependencies escaping the repository. The ownership fixtures also prove
relative-root acceptance and mismatched-owner rejection.

A real `GetShortPathNameW` C-volume alias reproduced the original six exceptions:
old HEAD's 21 fixture methods produced five ownership errors plus one closure
error; the candidate's 23 methods all passed under the same short-name input.
This was not a mock or a TEMP/TMP environment override. E-volume new directories
had no short alias; that unavailable probe was recorded separately and was not
called a defect reproduction. The uniquely owned C temporary directory was
path-verified before deletion and its absence checked afterward; no 8dot3 policy
or permissions were changed.

### Native CSV provenance: one outer job, two assertions

The two fixture blobs and original working files have the expected LF bytes.
Converting them to CRLF yields **exactly both hashes observed in hosted failures**:

| Fixture | Original SHA256 | Hosted/CRLF SHA256 |
|---|---|---|
| [Real](../../tests/perf/fixtures/NsightSourceReal.csv) | `80cd2b66b282bc3dcbfe13a5a8c0740ca4f1be7c70823cfedc7e957121dbc979` | `6c439b9d32d1f5e03c37edeecb5423500de460a998a84ec2f51d685ee0d74846` |
| [SelectedRange](../../tests/perf/fixtures/NsightSourceSelectedRange.csv) | `a66664e91366738c8dd1bff1de7758a713fd60528747ae896fb0b77e22d2f5bd` | `6d8853c6651c550b4f2d4d31b9b7b2c56705305b450f24296863ca8bf20019c6` |

[Git attributes](../../.gitattributes) disable text conversion for **only these
two files**. No CSV contents, provenance JSON, raw SHA algorithm or original
assertions were changed. [The regression](../../tests/perf/TestNativeNsightSource.py)
uses actual isolated Git indexes/checkouts with core.autocrlf=false and true,
then compares the exported bytes against the original provenance. Without the
rules it reproduces both hosted hashes and fails twice; with the rules all
13 methods pass. It requires Git, already required by the build bootstrap.

## Collected local validation, not hosted replacement

- Reused the compatible already-built Debug CPU tree, MSVC14.52.37025 and
  SDK10.0.26100.0, without changing its compiler/generator/SDK options. No C++ or
  Slang source changed, so no renderer rebuild was needed for this repair.
- Actual full selected CTest: **61/61 PASS**, exit0, **88.61s**. Configured preset
  contract remains 11/11; workflow contract 23/23; ownership fixtures 6/6;
  workload fixtures 17/17; native Nsight-source fixtures 13/13. Ownership audit
  remains 392 files / 278 modules / 114 implementation units.
- A new Python **3.12.4** venv had only pip before installation, user site disabled
  and `include-system-site-packages=false`. Wheel-only installation of the four
  pins succeeded; pip check passed; the complete resolved package list was saved.
- All **38/38 selected Python commands passed again** with only their executable
  replaced by that venv interpreter. Original arguments, working directories and
  configured per-command timeout budgets were retained; PYTHONPATH/PYTHONHOME
  were removed. This is an additional isolated Python check, **not** a second
  61-job CTest run, a new SDK-off build or Python3.12.10 hosted equivalence.
- The benchmark still has four GPU opt-in skips. Native Scene/Task/Debug output
  remains truncated by the existing CTest policy. No zero-inner-skip claim follows
  from outer JUnit skipped=0; timings are regression durations, not speedups.

The native helper's direct `pwsh -File` fixture preserves exit37/41/8. An actual
two-layer GitHub-shaped `pwsh -Command` wrapper instead reports outer exit1 for
all three. Logs survive and failure prevents subsequent steps. Five fixture
categories, direct and wrapped (ten executions), passed, including a real pip
missing-manifest failure. The initial wrong wrapper-code expectation was recorded
and investigated. **Do not claim hosted exact numeric propagation of CTest8.**
The original hosted outer exit1 is consistent with this observed shell boundary;
there was no reason to weaken failure handling or alter the helper for that.

Local generated evidence is ignored, not published as Actions artifacts:

- [Full 61-job log](<E:/metallic/build/d2-round9-20261114/candidate-local.log>) and
  [JUnit](<E:/metallic/build/d2-round9-20261114/candidate-local.xml>).
- [Clean Python replay, all commands and codes](<E:/metallic/build/d2-round9-20261114/python312-validation/results.json>),
  [pip install](<E:/metallic/build/d2-round9-20261114/python312-validation/pip-install.log>) and
  [resolved packages](<E:/metallic/build/d2-round9-20261114/python312-validation/packages.log>).
- [Byte/hash proof and downloaded hosted JUnit summary](<E:/metallic/build/d2-round9-20261114/fixture-bytes/initial-provenance-and-hosted-junit.json>),
  [two real checkout failures](<E:/metallic/build/d2-round9-20261114/fixture-bytes/checkout-red-both.log>) and
  [green](<E:/metallic/build/d2-round9-20261114/fixture-bytes/checkout-green.log>).
- [Real short-name red/green and cleanup](<E:/metallic/build/d2-round9-20261114/path-alias/REPORT.md>).
- [Python import/pin/wrapper validation](<E:/metallic/build/d2-round9-20261114/python-deps/ValidationSummary.md>).

## Acceptance gates still open

Independent review and the collected regressions are required before committing
and pushing this repair. A **new actual hosted run and its artifacts** must then
be checked; local green does not erase the first hosted failure.

Separately, frozen-026b actual Nsight integration-ON validation completed five
independent default-budget, -j1 case invocations: **4 PASS / 1 FAIL / 0 SKIP**.
It was not a single complete matrix invocation. Startup failed after a real
handshake with app1 and a source-epoch error; it was not a script-timeout failure.
One separately preserved diagnostic startup rerun still failed the unchanged
Streamline-error gate, despite app0, eight frames and a passing initialized
Streamline smoke. No `sl.dlss_g.dll` existed in the six-file deployed runtime;
no DLL, cache or associated generated file was deleted to force a pass. These
observations do not establish SDK caller/provenance or a causal fix.
See the [complete local Nsight report](<E:/metallic/build/d2-round8-20261114/nsight-integration/REPORT.md>).

That runtime run retained naturally written caches, did not establish cold-cache
behavior, did not inject forced timeout cleanup, and did not exercise independent
visual/temporal acceptance. Shared editor/#59 and real non-Windows cache-policy
validation also remain separate gates. **Issue #63 remains OPEN** until its full
acceptance criteria, not just this CPU slice, are satisfied.
