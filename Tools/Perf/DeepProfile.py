"""Separate compiler resources / Nsight range counters from optimization timing.

Uses the production in-frame case. No isolated replay, counter attribution to a
single dispatch, occupancy inference, or candidate promotion is performed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import ExperimentRunner as e
import WorkloadCase as w

PROTOCOL = "metallic-deep-profile-v1"
PREFIX = "RenderGraphPass: VBuffer (VisibilityBufferPass)/Visibility raster/Stream "
RANGES = {p: PREFIX + p for p in ("early", "late")}
# Retain read-only compatibility with initial native-export probes. These
# generic raster markers do NOT identify the WorkControl production dispatch.
LEGACY_RANGES = {p: PREFIX + p + "/Raster merge/Hybrid raster: software triangles" for p in ("early", "late")}
# Exact export names; duplicates are independent columns, never silently folded.
METRICS = {
    "GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed": "percent of peak sustained elapsed",
    "GPUTrace.lts__throughput.avg.pct_of_peak_sustained_elapsed": "percent of peak sustained elapsed",
    "GPUTrace.l1tex__throughput.avg.pct_of_peak_sustained_elapsed": "percent of peak sustained elapsed",
    "dram__sectors.avg.pct_of_peak_sustained_elapsed": "percent of peak sustained elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "raw exported count",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "raw exported count",
}
STAT = re.compile(r"\[PipelineStatistics\] entry=(\S+) spirv=([0-9a-f]{16}) executable=(.*?) subgroup=(\d+) (.*?)=(.*?) \((.*)\)$")
BINDING = re.compile(r"\[PipelineStatisticsBinding\] cacheKey=([0-9a-f]{16}) inputSpirvFnv1a64=(\d+) deviceSpirvFnv1a64=(\d+) shader=(\S+) entry=(\S+)$")


def resources(log, bindings):
    """Join the queried actual compute pipeline to recorded bound SPIR-V."""
    hashes = {int(d["spirvFnv1a64"]) for d in bindings}
    w.require(len(hashes) == 1, "Ambiguous bound shader")
    fingerprint = next(iter(hashes))
    w.require("[PipelineStatistics] enabled=true" in log, "Driver statistics unavailable")
    w.require(not re.search(r"\[PipelineStatistics\].*unavailable", log), "Incomplete pipeline statistics query")
    mappings = {}
    for line in log.splitlines():
        match = BINDING.search(line)
        if match and int(match[2]) == fingerprint:
            key, source, device, name, entry = match.groups()
            w.require(name == bindings[0]["module"] + "." + bindings[0]["entryPoint"], "Wrong source shader mapping")
            mapping = {"inputSpirvFnv1a64": source, "deviceSpirvFnv1a64": device, "shader": name, "entry": entry}
            w.require(key not in mappings or mappings[key] == mapping, "Conflicting shader mapping")
            mappings[key] = mapping
    w.require(len(mappings) == 1, "Missing/ambiguous production shader mapping")
    records = []
    for line_no, line in enumerate(log.splitlines(), 1):
        match = STAT.search(line)
        if not match or match[2] not in mappings:
            continue
        entry, spirv, executable, subgroup, name, value, description = match.groups()
        w.require(entry == mappings[spirv]["entry"], "Statistics entry point differs from binding")
        w.require(value != "unavailable", "Unknown statistic format")
        # Keep driver text and descriptions. Local Memory Size has previously
        # contained a sentinel-like value; do not relabel it as spills.
        records.append({"line": line_no, "entry": entry, "cacheKey": spirv, **mappings[spirv],
                        "executable": executable, "subgroup": int(subgroup), "name": name,
                        "value": value, "description": description})
    w.require(records, "No statistics for the actual bound production shader")
    for name in ("Register Count", "Shared Memory Size", "Binary Size"):
        w.require(any(r["name"] == name for r in records), f"Missing driver statistic: {name}")
    # Multiple pipeline creations may repeat the same executable statistics.
    unique = {}
    for r in records:
        key = (r["executable"], r["name"])
        value = {k: v for k, v in r.items() if k != "line"}
        w.require(key not in unique or unique[key] == value, "Conflicting pipeline statistics")
        unique[key] = value
    return {"kind": "compiler-resources", "boundSpirvFnv1a64": str(fingerprint),
            "statistics": sorted(unique.values(), key=lambda v: (v["executable"], v["name"])),
            "rawRecords": records, "hardwareCounters": False}


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    w.require(len(rows) > 1 and len(rows[0]) > 1, f"Empty export: {path}")
    w.require(all(len(r) == len(rows[0]) for r in rows), f"Ragged export: {path}")
    return rows


def number(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"Unavailable counter: {text}") from exc
    w.require(math.isfinite(value) and value >= 0, "Nonfinite/negative counter")
    return value


def triage(directory, ranges=None):
    ranges = RANGES if ranges is None else ranges
    w.require(ranges in (RANGES, LEGACY_RANGES), "Unregistered marker selection")
    path = directory / "GPUTRACE_REGIMES.xls"
    rows = table(path)
    events = table(directory / "D3DPERF_EVENTS.xls")
    w.require(rows[0][0] == "flattened_event_name" and events[0] == ["event_text", "time_ms", "time_ms"],
              "Unknown Nsight export schema")
    event_names = [r[0] for r in events[1:]]
    if any(name.startswith(" ") for name in event_names):
        # Native D3DPERF_EVENTS uses eight spaces per hierarchy level. Names
        # may themselves contain '/', so reconstruct from indentation only.
        stack, flattened = [], []
        for name in event_names:
            spaces = len(name) - len(name.lstrip(" "))
            depth = spaces // 8
            w.require(spaces % 8 == 0 and depth <= len(stack), "Invalid event indentation")
            stack = stack[:depth] + [name[spaces:]]
            flattened.append("/".join(stack))
        event_names = flattened
    w.require([r[0] for r in rows[1:]] == event_names, "Event/metric rows do not align")
    selections = {}
    for phase, marker in ranges.items():
        matches = [(i, row) for i, row in enumerate(rows[1:], 1) if row[0] == marker]
        w.require(len(matches) == 1, f"Missing/ambiguous exact range: {phase}")
        i, row = matches[0]
        metrics = []
        for name, unit in METRICS.items():
            columns = [j for j, header in enumerate(rows[0]) if header == name]
            w.require(len(columns) == 2, f"Expected two export columns for {name}")
            for j in columns:
                metrics.append({"name": name, "column": j + 1, "unit": unit, "value": number(row[j])})
        duration = [number(x) for x in events[i][1:]]
        w.require(all(x > 0 for x in duration), "Empty marker interval")
        selections[phase] = {"marker": marker, "row": i + 1, "durationMsColumns": duration, "metrics": metrics}
    return {"kind": "range-hardware-counters", "metricSet": "Top-Level Triage",
            "scope": "device activity over exported marker interval; not dispatch-exclusive",
            "duplicateColumnMeaning": "unlabeled by native export; kept separately without choosing or summing",
            "selections": selections}


def run_process(command, env, directory, timeout):
    # psutil is only needed for live process ownership, not offline verification.
    import psutil
    owned = {}
    status = {"command": command, "startedUnix": time.time()}
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        process = psutil.Popen(command, cwd=w.ROOT, env=env, stdout=stdout, stderr=stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        status["pid"] = process.pid
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                try:
                    for child in process.children(recursive=True):
                        owned[child.pid] = child
                except psutil.NoSuchProcess:
                    # The renderer may exit between poll() and children().
                    # Reap its actual exit status below instead of losing a capture.
                    process.wait(timeout=10)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("Diagnostic process timed out")
                time.sleep(.25)
            status["exitCode"] = process.returncode
        finally:
            # psutil guards PID reuse; never kill an unowned Nsight/renderer.
            for child in reversed(list(owned.values())):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            psutil.wait_procs(list(owned.values()), timeout=10)
            status["endedUnix"] = time.time()
            w.save(directory / "Process.json", status)
    w.require(status.get("exitCode") == 0, "Diagnostic process failed")


def inspect_run(directory, case, mode):
    w.require(mode in ("resources", "triage"), "Unknown diagnostic mode")
    process = w.load(directory / "Process.json")
    w.require(process.get("exitCode") == 0, "Failed diagnostic process")
    result = w.analyze_run(directory / "app", case)
    w.require(not result["normalTiming"], "Diagnostic run mislabeled as normal timing")
    capture = w.load(directory / "app/Capture.json")
    w.require(not capture.get("graphicsCaptureInjected") and not capture.get("renderDocInjected"),
              "Competing capture instrumentation")
    log = (directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
    if mode == "resources":
        w.require(capture.get("pipelineStatisticsRequested") is True and not capture.get("gpuTraceInjected"),
                  "Unexpected resource diagnostic instrumentation")
        profile = resources(log, result["identity"]["snapshot"]["productionDispatches"])
    else:
        w.require(capture.get("gpuTraceInjected") is True and not capture.get("pipelineStatisticsRequested"),
                  "Unexpected triage instrumentation")
        sdk = capture.get("sdkTrace", {})
        w.require(sdk.get("complete") is True and sdk.get("frames") == 1 and sdk.get("workloadCase") == capture["workloadCase"],
                  "Incomplete/wrong SDK capture")
        w.require(w.snapshot_identity(directory / "app", sdk["snapshot"], case) == result["identity"]["snapshot"],
                  "SDK workload differs from validated frames")
        w.require("Succeeded to export data:" in log, "Nsight export did not complete")
        combined = log + (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
        w.require(not re.search(r"buffer (?:overflow|exhaust)|counters? (?:unavailable|not available)|ERR_NVGPU", combined, re.I),
                  "Counter collection warning requires review")
        traces = list((directory / "trace").glob("*.ngfx-gputrace"))
        w.require(len(traces) == 1 and traces[0].stat().st_size > 0, "Missing/ambiguous fresh trace")
        invocation = w.load(directory / "Invocation.json")
        tables = list((directory / "trace").glob("*/GPUTRACE_REGIMES.xls"))
        w.require(len(tables) == 1, "Missing/ambiguous Nsight export collection")
        repro = table(tables[0].parent / "REPRO_INFO.xls")
        clocks = [r[1] for r in repro if r[0] == "GPU Clocks"]
        policy = invocation.get("clockPolicy", "unaltered")
        w.require(policy in ("unaltered", "base") and
                  clocks == [{"unaltered": "Unaltered", "base": "Locked to Base"}[policy]],
                  "Nsight did not report the requested clock policy")
        profile = triage(tables[0].parent, invocation.get("rangeSelection", LEGACY_RANGES))
    return {"validation": result, "profile": profile, "process": process}


def assess(rows, mode):
    w.require(len(rows) >= 3 and e.independent_processes(rows), "Need independent serial diagnostic processes")
    reference = rows[0]["validation"]
    for row in rows:
        w.require(row["validation"]["valid"] and not row["validation"]["normalTiming"], "Invalid/non-diagnostic run")
        w.require(e.inputs(row["validation"]) == e.inputs(reference), "Diagnostic workload inputs changed")
        w.require(all(row["validation"]["identity"]["snapshot"][r] == reference["identity"]["snapshot"][r]
                      for r in w.OUTPUTS), "Diagnostic depth/visibility differs")
    summary = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        w.require(len(selected) == 3, "Exactly three repeats per arm required")
        w.require(all(r["validation"]["identity"] == selected[0]["validation"]["identity"] for r in selected),
                  "Within-arm shader/input drift")
        if mode == "resources":
            stats = selected[0]["profile"]["statistics"]
            w.require(all(r["profile"]["statistics"] == stats for r in selected), "Unstable compiler resources")
            summary[arm] = {"repeatable": True, "statistics": stats}
        else:
            metrics = []
            for phase in RANGES:
                sets = [r["profile"]["selections"][phase]["metrics"] for r in selected]
                for index, metric in enumerate(sets[0]):
                    key = {k: v for k, v in metric.items() if k != "value"}
                    w.require(all({k: v for k, v in s[index].items() if k != "value"} == key for s in sets),
                              "Counter schema drift")
                    values = [s[index]["value"] for s in sets]
                    median = statistics.median(values)
                    spread = (max(values) - min(values)) / median if median else (0 if not any(values) else None)
                    metrics.append({**key, "phase": phase, "values": values, "median": median,
                                    "relativeSpread": spread, "stable": spread is not None and spread <= .1,
                                    "nonzero": median > 0})
            summary[arm] = {"metrics": metrics, "repeatable": all(m["stable"] for m in metrics),
                            "stableNonzeroMetrics": sum(m["stable"] and m["nonzero"] for m in metrics)}
    return {"protocol": PROTOCOL, "mode": mode, "arms": summary, "normalTiming": False,
            "relativeSpreadLimit": .1, "optimizationDecision": "not-evaluated"}


def execute(args):
    import psutil
    case = w.validate_case(w.load(args.case))
    w.require(case["sampleId"] == "gpu-driven-sample" and case["variant"] == "swWorkControl" and
              "primeCameraOffset" in case, "Only qualified MiniZorah history workload is supported")
    case.update(rounds=1, sampleFrames=8, nsightTraceFrames=1 if args.mode == "triage" else 0)
    case.pop("profileHoldSeconds", None)
    w.validate_case(case)
    w.require(1 <= args.timeout <= 900, "Invalid timeout")
    exe = args.exe.resolve()
    e.validate_build(args.build_dir.resolve(), exe)
    if args.mode == "triage":
        w.require(args.ngfx and args.ngfx.is_file() and args.architecture, "Nsight binary and architecture required")
    assets = w.load(args.assets)
    w.require(assets["protocol"] == "metallic-declared-assets-v1" and assets["stream"] == w.ASSETS[case["sampleId"]], "Wrong assets")
    w.asset_metadata_matches(assets)
    original = (w.ROOT / e.TARGET).read_bytes()
    changed = e.candidate_bytes(w.load(args.candidate), original) if args.candidate else original
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "Baseline.slang").write_bytes(original)
    (output / "Candidate.slang").write_bytes(changed)
    for name, value in (("Case.json", case), ("Assets.json", assets)):
        w.save(output / name, value)
    if args.candidate:
        w.save(output / "Candidate.json", w.load(args.candidate))
    transaction = e.ShaderTransaction(output, original, changed)
    lock = w.ROOT / "build/shader-experiment.lock"
    manifest = {"protocol": PROTOCOL, "status": "running", "mode": args.mode, "runs": []}
    rows = []
    acquired = False
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"output": str(output), "pid": os.getpid()}, handle)
        acquired = True
        sources = w.source_inventory()
        for relative in sources:
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((w.ROOT / relative).read_bytes())
            w.require(w.digest(dest) == sources[relative], "Source changed during archive")
        for name in ("DeepProfile.py", "ExperimentRunner.py", "WorkloadCase.py"):
            (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        runtime = {str(p): w.digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]}
        if args.mode == "triage":
            runtime[str(args.ngfx.resolve())] = w.digest(args.ngfx)
        manifest.update(runtime=runtime, gpu=e.gpu_identity(), architecture=args.architecture,
                        build=e.validate_build(args.build_dir.resolve(), exe))
        for index, arm in enumerate("ABABAB" if args.candidate else "AAA", 1):
            w.require(not any(p.info["name"] and p.info["name"].lower() in
                              {exe.name.lower(), "ngfx.exe", "ngfx-replay.exe"}
                              for p in psutil.process_iter(["name"])),
                      "Another renderer/CLI profiling process is present")
            transaction.install(original if arm == "A" else changed)
            expected = {**sources, e.TARGET: e.sha(transaction.current)}
            w.require(w.source_inventory() == expected, "Source drift")
            w.require(all(w.digest(Path(p)) == h for p, h in runtime.items()), "Runtime changed")
            w.require(e.gpu_identity() == manifest["gpu"], "GPU/driver changed")
            w.asset_metadata_matches(assets)
            directory = output / f"{index:02}-{arm}"
            (directory / "app").mkdir(parents=True)
            w.save(directory / "Config.json", w.engine_config(case))
            env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
            env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"),
                       METALLIC_FULL_ROAM_OUTPUT=str(directory / "app"), METALLIC_FULL_ROAM_HIDDEN="1",
                       METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
                       METALLIC_VK_PIPELINE_STATISTICS="1" if args.mode == "resources" else "0")
            command = [str(exe), "--sample", case["sampleId"]]
            if args.mode == "triage":
                (directory / "trace").mkdir()
                command = [str(args.ngfx.resolve()), "--activity", "GPU Trace Profiler", "--verbose",
                           "--exe", str(exe), "--dir", str(w.ROOT), "--args", "--sample " + case["sampleId"],
                           "--output-dir", str(directory / "trace"), "--start-with-ngfx-sdk", "--stop-with-ngfx-sdk",
                           "--max-duration-ms", "100", "--architecture", args.architecture,
                           "--metric-set-name", "Top-Level Triage", "--disable-collect-shader-pipelines",
                           "--set-gpu-clocks", args.clocks, "--auto-export", "--trace-timeout", "120"]
            w.save(directory / "Invocation.json", {"command": command,
                   "rangeSelection": RANGES, "clockPolicy": args.clocks if args.mode == "triage" else "unaltered",
                   "environment": {k: v for k, v in env.items() if k.startswith("METALLIC_")}})
            run_process(command, env, directory, args.timeout)
            row = {"arm": arm, **inspect_run(directory, case, args.mode)}
            w.require(w.source_inventory() == expected, "Source changed during capture")
            w.require(all(w.digest(Path(p)) == h for p, h in runtime.items()), "Runtime changed during capture")
            w.asset_metadata_matches(assets)
            w.save(directory / "Analysis.json", row)
            rows.append(row)
            manifest["runs"].append({"arm": arm, "directory": directory.name})
            print(json.dumps({"completed": index, "arm": arm, "mode": args.mode}), flush=True)
        w.save(output / "Summary.json", assess(rows, args.mode))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if acquired:
            try:
                transaction.restore()
                lock.unlink()
            except Exception as exc:
                manifest.update(status="restore-required", restoreError=str(exc))
        manifest["files"] = e.artifact_hashes(output)
        w.save(output / "Manifest.json", manifest)
    w.require(manifest["status"] == "complete", "Diagnostic transaction requires recovery")
    return {"status": manifest["status"], "directory": str(output)}


def verify(directory):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL and manifest["status"] == "complete", "Incomplete diagnostics")
    for name, digest in manifest["files"].items():
        w.require(w.digest(w.file_in(directory, name)) == digest, f"Changed evidence: {name}")
    case = w.load(directory / "Case.json")
    rows = []
    for row in manifest["runs"]:
        path = w.file_in(directory, row["directory"] + "/Analysis.json").parent
        current = {"arm": row["arm"], **inspect_run(path, case, manifest["mode"])}
        w.require(current == w.load(path / "Analysis.json"), "Analysis mismatch")
        rows.append(current)
    summary = assess(rows, manifest["mode"])
    w.require(summary == w.load(directory / "Summary.json"), "Summary mismatch")
    w.require(w.load(directory / "Transaction.json")["state"] == "restored", "Unrestored transaction")
    return {"verified": True, "runs": len(rows), "mode": manifest["mode"],
            "verifierSha256": w.digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("resources", "triage"), required=True)
    for key in ("case", "assets", "exe", "build-dir", "output"):
        run.add_argument("--" + key, type=Path, required=True)
    run.add_argument("--candidate", type=Path)
    run.add_argument("--ngfx", type=Path)
    run.add_argument("--architecture")
    run.add_argument("--clocks", choices=("unaltered", "base"), default="unaltered")
    run.add_argument("--timeout", type=int, default=240)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args) if args.command == "run" else verify(args.directory.resolve()), indent=2))
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
