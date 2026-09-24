# Performance evidence baseline (M0)

`Baseline.py` discovers the installed Nsight CLI, pins tool binaries by SHA-256,
records the current GPU/driver and source snapshot, indexes existing captures,
and archives historical reports/tables. It does not capture, replay, launch an
editor, change GPU settings, or operate the desktop. Python's standard library
is sufficient; the existing Nsight wrapper, `rg`, Git and `nvidia-smi` are used
for discovery.

```powershell
python -B Tools/Perf/Baseline.py --output build/perf-baseline-<unique-run> --source-location remote-only
python -B Tools/Perf/Baseline.py --verify build/perf-baseline-<unique-run>/Baseline.json
cmake -S tests/perf -B build/perf-tests
ctest --test-dir build/perf-tests -C Debug --output-on-failure
```

The output directory must not already exist. `raw/` retains exact command output
and hashes; `archive/` preserves reports, source snapshots and exported tables.
Large capture/trace files and installed binaries are referenced by path and hash
without copying. The integrity check therefore requires those referenced files
to remain available. An unchanged archive can still be verified if its original
source file has moved.

`verified` means only the explicitly named scope was exercised successfully.
`supported-unverified` records advertised support without a current validation.
`unsupported` is reserved for a known capability absence. `blocked` records an
environmental or evidence dependency. Historical success never proves current
runtime capture success. Missing, empty, unreadable and hash-mismatched files
are separate states; none count as valid evidence.

Use `--child <observed-cua-child.exe>` for its read-only status query. A separately
authorized `cua-child.exe mcp --timeout 25` initialization probe can be recorded
with `--child-probe-dir <dir>` containing `cua-child-initialize.stdout.txt` and
`cua-child-initialize.stderr.txt`. The collector never starts or reconnects a
desktop session. A ready worker still does not prove Nsight UI automation.

The original Shader Profiler CSV is currently remote-only, as confirmed by the
user. Once synchronized, pass `--shader-csv <path>`; the historical SHA-256 in
`Documentation/StreamClusterBinProfile20260924.json` must match. An independently
generated replacement can be indexed with `--new-export`, but remains an open
provenance gap until its capture, shader identity, selection and environment are
recorded. Do not synthesize a source CSV from the historical derived JSON.

The original CSV and its historical temporary parser are searched by name under
the repository. Extra known evidence directories can be provided with repeated
`--search-root`; search failures are retained in raw stderr. Search is bounded
to these roots, not a claim that the file is absent from every machine.

`case.json` is an explicitly historical minimal case for the archived September
20 legacy SW investigation. Its achieved scope includes neighboring HW/SW work,
its queue/dispatch and captured source hash remain unknown, and it is ineligible
for timing acceptance. It is not the current production WorkControl case.

The M0 result stays `partial` while the original source fixture or required
environment evidence is unavailable. Integrity passing means the evidence has
not changed; it does not mean M0, shader correlation or profiling is complete.

The checked-in samples under `tests/perf/fixtures/` are real historical GPU Trace
TSV bytes, with source hashes and physical line provenance. They deliberately
retain duplicate metric column names. They are not a substitute for the missing
Shader Profiler source/IL CSV.
