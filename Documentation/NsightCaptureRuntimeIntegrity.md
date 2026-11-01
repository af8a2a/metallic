# Nsight capture runtime integrity on Windows

A capture launcher running as the expected user does not establish the target's
write permissions. Windows creates a process at the lower of the user's and the
executable file's integrity levels. A Low target cannot write Medium cache or
report objects even when its user has discretionary Modify access. See
[Microsoft's Mandatory Integrity Control documentation](https://learn.microsoft.com/en-us/windows/win32/secauthz/mandatory-integrity-control).

## Confirm the failure

In the VSM capture repair on 2026-11-01, the original Release executables had an
inherited `Low Mandatory Level:(NW)` label. `ngfx.exe` was Medium, but the actual
`MetallicRHITests.exe` child was Low. It failed both HTML report creation and
`StreamerSubsystem`'s scene-streaming source-identity write. The scene-streaming
cache is rooted at `PROJECT_SOURCE_DIR/.cache/scene-streams`, so changing the
working directory or TEMP does not address this failure.

Inspect the exact target file with `icacls` and record the actual child process's
`TokenIntegrityLevel`; inspecting only the shell or launcher misses this case.
A Python write probe also missed it because that interpreter's executable did
not have the Low label. An actual native zero-test GTest invocation reproduced
report-write failure without running a GPU test.

## Capture-only repair

Use an approved Medium execution context to create a fresh capture runtime
directory. Write new files containing the exact bytes of the target EXE and its
sibling DLLs, retaining the original working directory and shader/asset paths.
Verify every source/staged SHA-256 and inspect the new EXE's integrity label.
The new files must use the fresh directory's security metadata; copying the old
file's security descriptor would reproduce the problem.

Keep the original build artifacts and directory permissions unchanged. Verify
that the actual Nsight-launched child is Medium, that it writes its reports and
scene cache, and that the capture contains the requested rendered workload.
An exported UI-only trace following scene compilation failure is invalid.
Recheck staged hashes against the current build before every subsequent capture.

The bounded repair scripts and receipts are local outputs under
`build/vsm-nsight-repair-20261101/`; the byte-identical runtime and valid A/B traces
are under `Captures/NsightGraphics/VSMRelativeMappingMedium-20261101/`.
`StageRuntime.py` records all copied file hashes, `NativeWriteProof.py` compares
native zero-test report creation, and `CaptureAB.py` records host/target tokens
and rejects stale source/binary identities. They are evidence for this workload,
not permission to execute an archived runner against a different checkout.

The fix required no renderer source changes, ACL broadening, driver changes or
ordinary timing reruns. Both optimized SDK frame-48 captures completed with
1920-by-1080 Projection dispatches and matching existing workload settings.
