"""Bounded shader candidate experiments: guarded patch, ABBA, exact outputs and decisions.

No automatic promotion. All processes run serially; only the declared shader is
temporarily changed. An interrupted transaction can be recovered from its journal.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import csv
import datetime
import difflib
import hashlib
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import WorkloadCase as w

ROOT = w.ROOT
TARGET = "Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang"
PROTOCOL = "metallic-shader-experiment-v1"
POLICY = {"minimumGain": .03, "maximumGraphRegression": .02,
          "confidence": .95, "primaryMetric": "softwareTotalMs", "promotion": "manual"}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE, read only
        if not handle:
            return ctypes.get_last_error() != 87  # Only invalid PID proves absence.
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gpu_identity():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10, check=True).stdout.strip()


def validate_build(build_dir, executable):
    cache = (build_dir / 'CMakeCache.txt').read_text(encoding='utf-8')
    entries = dict(line.split('=',1) for line in cache.splitlines() if '=' in line and not line.startswith(('#','//')))
    w.require(entries.get('CMAKE_BUILD_TYPE:STRING') == 'Release', 'M3 requires a single-config Release build')
    w.require(Path(entries.get('CMAKE_HOME_DIRECTORY:INTERNAL', '')).resolve() == ROOT.resolve(), 'Build belongs to another source tree')
    w.require(executable.resolve() == (build_dir/'Source/MetallicGPUDrivenSample.exe').resolve(), 'Executable does not match the built target')
    return {'cacheSha256': w.digest(build_dir/'CMakeCache.txt'), 'configuration':'Release', 'sourceRoot':str(ROOT)}


def candidate_bytes(spec, original):
    w.require(set(spec) == {"protocol", "id", "hypothesis", "path", "baseSha256", "replacements"},
              "Unknown or missing candidate fields")
    w.require(spec["protocol"] == PROTOCOL and spec["path"] == TARGET,
              "First milestone supports only the WorkRaster shader file")
    w.require(all(isinstance(spec[k], str) and spec[k].strip() for k in ("id", "hypothesis")), "Missing hypothesis/id")
    w.require(sha(original) == spec["baseSha256"], "Candidate baseline hash mismatch")
    changes = spec["replacements"]
    w.require(isinstance(changes, list) and 1 <= len(changes) <= 16, "Invalid replacement count")
    result = original.decode("utf-8")
    for change in changes:
        w.require(set(change) == {"before", "after", "count"} and
                  isinstance(change["before"], str) and isinstance(change["after"], str) and
                  change["before"] and type(change["count"]) is int and 1 <= change["count"] <= 64,
                  "Invalid replacement")
        w.require(result.count(change["before"]) == change["count"], "Candidate context count mismatch")
        result = result.replace(change["before"], change["after"])
    encoded = result.encode("utf-8")
    w.require(encoded != original, "Candidate is a textual no-op")
    return encoded


def interval(values):
    w.require(len(values) in T95 and all(math.isfinite(v) for v in values), "Invalid independent pair sample")
    mean = statistics.mean(values)
    margin = T95[len(values)] * statistics.stdev(values) / math.sqrt(len(values))
    return {"pairs": len(values), "values": values, "mean": mean, "lower95": mean-margin, "upper95": mean+margin}


def inputs(result):
    identity = copy.deepcopy(result["identity"])
    for resource in w.OUTPUTS:
        del identity["snapshot"][resource]
    for dispatch in identity["snapshot"]["productionDispatches"]:
        # This is the one intentional binding difference in a shader candidate.
        del dispatch["spirvFnv1a64"]
    return identity


def verdict(status, reason, **evidence):
    return {"decision": status, "reason": reason, "candidateAccepted": status == "accept", **evidence}


def independent_processes(rows):
    previous_end = None
    for row in rows:
        process = row['process']
        start, end = process.get('startedUnix'), process.get('endedUnix')
        if (type(process.get('pid')) is not int or process['pid'] <= 0 or
            not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)) or
            start >= end or (previous_end is not None and start < previous_end)):
            return False
        previous_end = end
    return True


def assess(rows, case, policy, expected_runs):
    """Recomputed from raw runs by verify; never trust a supplied pass boolean."""
    if not rows:
        return verdict("inconclusive", "no_completed_runs")
    results = [r["result"] for r in rows]
    if any(not r["valid"] or not r["normalTiming"] for r in results):
        return verdict("inconclusive", "invalid_or_instrumented_run")
    if any(r["process"]["exitCode"] != 0 or not r["competition"]["covered"] for r in rows):
        return verdict("inconclusive", "process_or_environment_evidence_missing")
    a = [r for row, r in zip(rows, results) if row["arm"] == "A"]
    b = [r for row, r in zip(rows, results) if row["arm"] == "B"]
    if not a or not b:
        return verdict("inconclusive", "missing_comparison_arm")
    # Compare output bytes via hashes of validated readbacks before timing.
    output = lambda r: {key: r["identity"]["snapshot"][key] for key in w.OUTPUTS}
    if any(output(r) != output(a[0]) for r in results):
        return verdict("reject", "exact_depth_visibility_mismatch")
    if any(inputs(r) != inputs(a[0]) for r in results):
        return verdict("inconclusive", "workload_input_or_binding_drift")
    if any(r["identity"] != arm[0]["identity"] for arm in (a, b) for r in arm):
        return verdict("inconclusive", "within_arm_identity_drift")
    fingerprints = lambda r: [d["spirvFnv1a64"] for d in r["identity"]["snapshot"]["productionDispatches"]]
    if fingerprints(a[0]) == fingerprints(b[0]):
        return verdict("inconclusive", "candidate_compiled_to_same_shader")
    if len(rows) != expected_runs:
        return verdict("inconclusive", "incomplete_schedule")
    if [r["arm"] for r in rows] != list("ABBA") * (expected_runs // 4):
        return verdict("inconclusive", "invalid_interleaving")
    if not independent_processes(rows):
        return verdict("inconclusive", "independent_process_evidence_invalid")
    qualification = {label: w.compare_runs(arm, case) for label, arm in (("A", a), ("B", b))}
    if any(q["status"] != "stable" for q in qualification.values()):
        return verdict("inconclusive", "within_arm_timing_noise", qualification=qualification)
    gains = {}
    for metric in ("softwareTotalMs", "graphGpuMs"):
        samples = []
        for offset in range(0, len(rows), 2):
            pair = {r["arm"]: r["result"]["timings"][metric]["median"] for r in rows[offset:offset+2]}
            w.require(all(v > 0 and math.isfinite(v) for v in pair.values()), "Invalid pair timing")
            samples.append(1 - pair["B"] / pair["A"])
        gains[metric] = interval(samples)
    evidence = {"gains": gains, "qualification": qualification,
                "correctness": "exact depth/visibility match across all checkpoints and arms",
                "statisticalUnit": "adjacent independent-process pair; frame medians are not independent trials"}
    primary, graph = gains["softwareTotalMs"], gains["graphGpuMs"]
    if primary["upper95"] < policy["minimumGain"]:
        return verdict("reject", "gain_below_threshold", **evidence)
    if graph["upper95"] < -policy["maximumGraphRegression"]:
        return verdict("reject", "graph_regression", **evidence)
    if primary["lower95"] >= policy["minimumGain"] and graph["lower95"] >= -policy["maximumGraphRegression"]:
        return verdict("accept", "qualified_gain", **evidence)
    return verdict("inconclusive", "confidence_interval_crosses_gate", **evidence)


def stop_owned(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def monitor_coverage(directory, pid):
    result = w.competition_summary(directory, pid)
    capture = w.load(directory / 'Capture.json')
    windows = [(c['measurementBeginUnixMs'], c['measurementEndUnixMs']) for c in capture['cases']]
    covered = [False] * len(windows)
    with (directory / 'GpuProcesses.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if int(row['pid']) != pid:
                continue
            timestamp = datetime.datetime.fromisoformat(row['timestamp']).timestamp()*1000
            for index, (start, end) in enumerate(windows):
                covered[index] |= timestamp >= start and timestamp-1000 <= end
    result.update(windowCoverage=covered, covered=bool(covered) and all(covered))
    return result


def collect(case, executable, directory, timeout):
    directory.mkdir()
    w.save(directory / "Config.json", w.engine_config(case))
    env = os.environ.copy()
    # Pin all Metallic overrides instead of inheriting an unrelated capture session.
    for key in list(env):
        if key.startswith("METALLIC_"):
            del env[key]
    env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"),
               METALLIC_FULL_ROAM_OUTPUT=str(directory), METALLIC_FULL_ROAM_HIDDEN="1",
               METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = monitor = telemetry = None
    started = time.time()
    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err, \
         (directory / "GpuProcesses.csv").open("wb") as counters, \
         (directory / "GpuProcesses.stderr.log").open("wb") as counter_errors, \
         (directory / "Gpu.csv").open("wb") as gpu:
        try:
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw",
                                          "--format=csv", "-l", "1"], stdout=gpu, stderr=subprocess.STDOUT, creationflags=flags)
            process = subprocess.Popen([str(executable), "--sample", case["sampleId"]], cwd=ROOT, env=env,
                                       stdout=out, stderr=err, creationflags=flags)
            monitor = subprocess.Popen(["powershell.exe", "-NoProfile", "-File", str(Path(__file__).with_name("MeasureExperimentGpu.ps1")),
                                        "-TargetProcessId", str(process.pid)],
                                       stdout=counters, stderr=counter_errors, creationflags=flags)
            code = process.wait(timeout=timeout)
            w.require(code == 0, f"Workload exit code {code}")
        finally:
            stop_owned(process)
            stop_owned(monitor)
            stop_owned(telemetry)
            w.save(directory / "Process.json", {"pid": process.pid if process else None,
                   "exitCode": process.returncode if process else None, "startedUnix": started,
                   "endedUnix": time.time(), "environment": {k:v for k,v in env.items() if k.startswith("METALLIC_")}})
    logs = "\n".join((directory / n).read_text(encoding="utf-8", errors="replace") for n in ("stdout.log", "stderr.log"))
    w.require(not any(s in logs for s in ("Validation Error", "VUID-", "DeviceLost", "VK_ERROR_DEVICE_LOST")), "GPU error in workload log")
    result = w.analyze_run(directory, case)
    w.require(result["normalTiming"], "Instrumented run")
    competition = monitor_coverage(directory, process.pid)
    w.save(directory / "Validation.json", result)
    w.save(directory / "Competition.json", competition)
    return {"result": result, "competition": competition, "process": w.load(directory / "Process.json")}


class ShaderTransaction:
    def __init__(self, output, original, candidate):
        self.output, self.original, self.candidate = output, original, candidate
        self.path = ROOT / TARGET
        self.current = original

    def install(self, data):
        w.require(self.path.read_bytes() == self.current, "Concurrent shader edit: refusing to overwrite")
        # Journal is durable before mutation; recovery also accepts either known state.
        w.save(self.output / "Transaction.json", {"path": TARGET, "baselineSha256": sha(self.original),
               "candidateSha256": sha(self.candidate), "state": "active"})
        if data != self.current:
            self.path.write_bytes(data)
        self.current = data

    def restore(self):
        self.install(self.original)
        value = w.load(self.output / "Transaction.json")
        value["state"] = "restored"
        w.save(self.output / "Transaction.json", value)


def recover(directory):
    lock = ROOT / "build" / "shader-experiment.lock"
    if lock.exists():
        owner = w.load(lock)
        w.require(Path(owner["output"]).resolve() == directory.resolve(), "Lock belongs to another experiment")
        w.require(not process_alive(owner["pid"]), "Experiment process is still alive; recovery refused")
    journal = w.load(directory / "Transaction.json")
    w.require(journal["path"] == TARGET, "Invalid recovery target")
    original = (directory / "Baseline.slang").read_bytes()
    w.require(sha(original) == journal["baselineSha256"], "Recovery baseline damaged")
    path = ROOT / TARGET
    current = w.digest(path)
    w.require(current in (journal["baselineSha256"], journal["candidateSha256"]),
              "Concurrent shader edit: manual recovery required; baseline preserved")
    if current != journal["baselineSha256"]:
        path.write_bytes(original)
    journal["state"] = "restored"
    w.save(directory / "Transaction.json", journal)
    if lock.exists():
        lock.unlink()
    # Keep the original manifest immutable. Recovery changes transaction evidence;
    # an interrupted experiment never becomes eligible for acceptance.
    return {"state": "restored", "path": TARGET}


def artifact_hashes(directory):
    return {p.relative_to(directory).as_posix(): w.digest(p) for p in directory.rglob("*")
            if p.is_file() and p != directory / "Manifest.json"}


def execute(args):
    case = w.validate_case(w.load(args.case))
    w.require(case["variant"] == "swWorkControl" and case["sampleId"] == "gpu-driven-sample" and
              "primeCameraOffset" in case and not case.get("nsightTraceFrames") and not case.get("profileHoldSeconds"),
              "M3 requires the qualified MiniZorah nonzero-late normal timing case")
    w.require(2 <= args.blocks <= 5 and 2 <= args.confirmation_blocks <= 5 and 1 <= args.timeout <= 900,
              "Invalid bounded run budget")
    candidate = w.load(args.candidate)
    original = (ROOT / TARGET).read_bytes()
    changed = candidate_bytes(candidate, original)
    assets = w.load(args.assets)
    w.require(assets["protocol"] == "metallic-declared-assets-v1" and assets["stream"] == w.ASSETS[case["sampleId"]], "Wrong assets")
    w.asset_metadata_matches(assets)
    validate_build(args.build_dir.resolve(), args.exe.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    policy = dict(POLICY)
    manifest = {"protocol": PROTOCOL, "status": "running", "runs": [], "policy": policy,
                "blocks": args.blocks, "confirmationBlocks": args.confirmation_blocks,
                "toolHashes": {n: w.digest(Path(__file__).with_name(n)) for n in ("ExperimentRunner.py", "WorkloadCase.py", "MeasureExperimentGpu.ps1")}}
    w.save(output / "Manifest.json", manifest)
    for name in manifest["toolHashes"]:
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    for name, value in (("Candidate.json", candidate), ("Case.json", case), ("Assets.json", assets)):
        w.save(output / name, value)
    (output / "Baseline.slang").write_bytes(original)
    (output / "Candidate.slang").write_bytes(changed)
    (output / "Candidate.patch").write_text("".join(difflib.unified_diff(original.decode().splitlines(True),
         changed.decode().splitlines(True), fromfile="a/"+TARGET, tofile="b/"+TARGET)), encoding="utf-8")
    transaction = ShaderTransaction(output, original, changed)
    final = verdict("inconclusive", "experiment_did_not_finish")
    lock = ROOT / "build" / "shader-experiment.lock"
    lock.parent.mkdir(exist_ok=True)
    # Lock is a cooperative runner mutex. A stale lock requires explicit recovery.
    lock_handle = None
    try:
        lock_handle = lock.open("x", encoding="utf-8")
        lock_handle.write(__import__('json').dumps({"output": str(output), "pid": os.getpid()})); lock_handle.flush()
        sources = w.source_inventory()
        manifest["baselineSources"] = sources
        for relative in sources:
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / relative).read_bytes())
            w.require(w.digest(dest) == sources[relative], "Source changed during archive")
        # Host build is explicit; shader candidate compilation occurs on process
        # startup through the production Slang compiler before timed warmup.
        build_command = ["cmake", "--build", str(args.build_dir.resolve()), "--target", "MetallicGPUDrivenSample", "--config", "Release"]
        with (output / "Build.log").open("wb") as log:
            build = subprocess.run(build_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        w.save(output / "Build.json", {"argv": build_command, "exitCode": build.returncode,
               "shaderBuild": "production Slang at process startup, bound SPIR-V checked per arm"})
        w.require(build.returncode == 0, "Host build failed")
        manifest['buildIdentity'] = validate_build(args.build_dir.resolve(), args.exe.resolve())
        (output/'CMakeCache.txt').write_bytes((args.build_dir/'CMakeCache.txt').read_bytes())
        exe = args.exe.resolve()
        runtime = {str(p): w.digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]}
        manifest["runtime"] = runtime
        gpu = gpu_identity()
        manifest["gpu"] = gpu
        for stage, blocks in (("discovery", args.blocks), ("confirmation", args.confirmation_blocks)):
            stage_rows = []
            for index, arm in enumerate(list("ABBA") * blocks):
                transaction.install(original if arm == "A" else changed)
                expected = {**sources, TARGET: sha(transaction.current)}
                w.require(w.source_inventory() == expected, "Source drift outside declared candidate")
                w.require(all(w.digest(Path(p)) == h for p,h in runtime.items()), "Runtime changed")
                w.asset_metadata_matches(assets)
                w.require(gpu_identity() == gpu, "GPU/driver identity changed")
                name = f"{stage}-{index+1:02d}-{arm}"
                print(f"{name}: start", flush=True)
                row = collect(case, exe, output / name, args.timeout)
                row.update(arm=arm, stage=stage, directory=name)
                stage_rows.append(row)
                manifest["runs"].append({k:row[k] for k in ("arm", "stage", "directory")})
                w.require(w.source_inventory() == expected, "Sources changed during measurement")
                w.require(all(w.digest(Path(p)) == h for p,h in runtime.items()), "Runtime changed during measurement")
                w.asset_metadata_matches(assets)
                check = assess(stage_rows, case, policy, blocks*4)
                w.save(output / "Manifest.json", manifest)
                print(f"{name}: verified; {check['reason']}", flush=True)
                if check["reason"] in ("exact_depth_visibility_mismatch", "workload_input_or_binding_drift",
                                       "within_arm_identity_drift", "candidate_compiled_to_same_shader",
                                       "process_or_environment_evidence_missing", "invalid_or_instrumented_run"):
                    break
            final = assess(stage_rows, case, policy, blocks*4)
            w.save(output / (stage + "-Decision.json"), final)
            if final["decision"] != "accept":
                break
        # Discovery acceptance only schedules an independent confirmation stage.
        if final["decision"] == "accept":
            w.require(stage == "confirmation", "Missing independent confirmation")
    except BaseException as error:
        final = verdict("inconclusive", "execution_failure", error=str(error))
    finally:
        if lock_handle is not None:
            try:
                transaction.restore()
                manifest["restored"] = True
            except Exception as error:
                manifest["restored"] = False
                final = verdict("inconclusive", "restore_conflict", error=str(error))
            lock_handle.close()
            if manifest.get("restored"):
                lock.unlink()
        manifest["status"] = "complete"
        w.save(output / "Decision.json", final)
        manifest["artifacts"] = artifact_hashes(output)
        w.save(output / "Manifest.json", manifest)
    return final


def verify(directory):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL and manifest["status"] == "complete", "Incomplete experiment")
    w.require(manifest["policy"] == POLICY, "Decision policy changed")
    w.require(2 <= manifest['blocks'] <= 5 and 2 <= manifest['confirmationBlocks'] <= 5, "Invalid schedule budget")
    for relative, expected in manifest["artifacts"].items():
        w.require(w.digest(w.file_in(directory, relative)) == expected, f"Artifact changed: {relative}")
    w.require(artifact_hashes(directory) == manifest["artifacts"], "Artifact inventory changed")
    w.require({'ExperimentRunner.py', 'WorkloadCase.py'} <= set(manifest['toolHashes']) <=
              {'ExperimentRunner.py', 'WorkloadCase.py', 'MeasureExperimentGpu.ps1'}, 'Missing/unknown producer tools')
    for name, expected in manifest["toolHashes"].items():
        w.require(w.digest(w.file_in(directory, name)) == expected, "Archived producer tool changed")
    case = w.validate_case(w.load(directory / "Case.json"))
    candidate = w.load(directory / "Candidate.json")
    w.require(candidate_bytes(candidate, (directory / "Baseline.slang").read_bytes()) ==
              (directory / "Candidate.slang").read_bytes(), "Patch does not match candidate")
    final = w.load(directory / "Decision.json")
    w.require(final["decision"] in ("accept", "reject", "inconclusive") and
              final["candidateAccepted"] == (final["decision"] == "accept"), "Invalid decision")
    stages = {}
    seen = set()
    for entry in manifest["runs"]:
        w.require(entry['stage'] in ('discovery', 'confirmation') and entry['arm'] in ('A','B'), "Unknown schedule entry")
        w.require(entry["directory"] not in seen, "Duplicate process evidence")
        seen.add(entry["directory"])
        run = w.file_in(directory, entry["directory"] + "/Capture.json").parent
        row = {**entry, "result": w.analyze_run(run, case), "process": w.load(run / "Process.json")}
        process = row['process']
        for measured in w.load(run / 'Capture.json')['cases']:
            start, end = measured['measurementBeginUnixMs'], measured['measurementEndUnixMs']
            w.require(process['startedUnix']*1000 <= start < end <= process['endedUnix']*1000,
                      'Measurement window is outside its process lifetime')
        row["competition"] = monitor_coverage(run, row["process"]["pid"])
        stages.setdefault(entry["stage"], []).append(row)
    for stage, rows in stages.items():
        count = manifest["blocks" if stage == "discovery" else "confirmationBlocks"] * 4
        calculated = assess(rows, case, manifest["policy"], count)
        stage_file = directory / (stage + "-Decision.json")
        if stage_file.exists():
            w.require(calculated == w.load(stage_file), "Stage decision differs from raw evidence")
    w.require(independent_processes([r for rows in stages.values() for r in rows]),
              'Reused/overlapping independent-process evidence')
    if final["decision"] == "accept":
        w.require(manifest.get("restored") and set(stages) == {"discovery", "confirmation"}, "Acceptance missing confirmation/restore")
        w.require(all(w.load(directory / (s + "-Decision.json"))["decision"] == "accept" for s in stages), "Acceptance gate failed")
        w.require(final == w.load(directory / "confirmation-Decision.json"), "Final decision mismatch")
        w.require(w.load(directory / 'Build.json')['exitCode'] == 0 and
                  manifest['buildIdentity']['configuration'] == 'Release' and
                  w.digest(directory/'CMakeCache.txt') == manifest['buildIdentity']['cacheSha256'] and
                  w.load(directory / 'Transaction.json')['state'] == 'restored', "Build or restore failed")
    elif final["reason"] not in ("execution_failure", "restore_conflict"):
        last = "confirmation" if "confirmation" in stages else "discovery"
        w.require(final == w.load(directory / (last + "-Decision.json")), "Final decision mismatch")
    return {"integrity": "passed", "producerToolsVerified": True,
            "verifierSha256": w.digest(Path(__file__)), **final}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("case", "candidate", "assets", "exe", "build-dir", "output"):
        run.add_argument("--"+name, type=Path, required=True)
    run.add_argument("--blocks", type=int, default=3)
    run.add_argument("--confirmation-blocks", type=int, default=2)
    run.add_argument("--timeout", type=int, default=300)
    for name in ("verify", "recover"):
        commands.add_parser(name).add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = execute(args) if args.command == "run" else (verify if args.command == "verify" else recover)(args.directory.resolve())
        print(__import__('json').dumps(result, indent=2))
        return 2 if result.get("decision") == "inconclusive" else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
