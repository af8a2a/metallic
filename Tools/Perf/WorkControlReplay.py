"""Isolated WorkControl correctness evidence. Archived code is never executed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path
import struct

import DeepProfile as d
import ExperimentRunner as e
import WorkloadCase as w
import NvPerf as n

PROTOCOL = "metallic-work-control-replay-experiment-v1"
BOUND = {"pages", "groups", "header", "pageTable", "params", "rasterBindings", "bins", "pixels", "instances"}
GUARDS = {"requests", "visibleRecords", "lodState", "hzb0", "hzb1", "instanceVisibility", "visibleInstanceIds", "visibleInstanceCounter"}
NAMES = BOUND | GUARDS | {"arguments"}
FAULTS = ("cancel-before-submit", "cancel-after-submit", "submit-failure", "device-error", "timeout", "restore-failure", "state-leak")


def inspect_failure(directory, fault):
    report = w.load(directory / "app/replay/Replay.json")
    process = w.load(directory / "Process.json")
    expected = {"restore-failure": "replay_restore_failed", "state-leak": "replay_production_state_leak"}.get(fault, "injected_" + fault)
    version = report.get("faultInjectionVersion", 1)
    w.require(version in (1, 2), "Unknown fault injection version")
    if version == 2:
        expected = {"submit-failure": "replay_submit_failed", "device-error": "replay_submit_device_lost", "timeout": "replay_wait_timeout"}.get(fault, expected)
    w.require(fault in FAULTS and report.get("faultInjection") == fault and report.get("status") == "failed" and
              report.get("error") == expected and report.get("counterEligible") is False and process.get("exitCode") not in (None, 0),
              "Expected injected failure was not observed")
    w.require(not (directory / "app/replay/nvperf").exists(), "Profiler started before failed correctness gate")
    if fault == "restore-failure":
        index = 1 if version == 2 else 0
        w.require(not equal_files(directory / "app/replay/pixels-input.bin", directory / f"app/replay/{index}-pixels-restored.bin"), "Missing restore fault bytes")
    if version == 2 and fault in ("device-error", "timeout"):
        w.require(report.get("processPoisoned") is True and process["exitCode"] == 3, "Poisoned process resumed")
    if fault == "state-leak":
        w.require(not equal_files(directory / "app/replay/pageTable-production-before.bin", directory / "app/replay/pageTable-production-after.bin"), "Missing state fault bytes")
    return {"status": "expected-failure-verified", "fault": fault, "injected": True, "counterEligible": False}


def equal_files(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


def inspect_replay(directory):
    report = w.load(directory / "Replay.json")
    w.require(report.get("protocol") == "metallic-work-control-replay-v1" and report.get("status") == "complete", "Incomplete replay")
    w.require("faultInjection" not in report, "Injected failure cannot become successful evidence")
    counter_passes = report.get("counterPasses", 0)
    w.require(type(counter_passes) is int and 0 <= counter_passes <= 16, "Invalid counter pass count")
    w.require((counter_passes == 0 and report.get("scope") == "isolated-correctness-only" and report.get("counterEligible") is False) or
              (counter_passes > 0 and report.get("scope") == "isolated-dispatch-RHI-exclusive" and report.get("counterEligible") is True),
              "Correctness-only evidence cannot grant counter attribution")
    if counter_passes:
        w.require(report.get("submissionIsolation") == "all-RHI-queues-owner-lease", "Missing submission exclusion")
        profile = w.load(w.file_in(directory, "nvperf/NvPerf.json"))
        w.require(profile.get("protocol") == "metallic-nvperf-isolated-v1" and profile.get("status") == "complete" and
                  profile.get("scope") == report["scope"] and profile.get("measurementKind") == "diagnostic" and
                  profile.get("clockPolicy") == "unaltered", "Wrong isolated profiler scope/policy")
        w.require(profile.get("requiredPasses") == profile.get("passesCollected") == counter_passes and
                  profile.get("numRangesDropped") == 0 and profile.get("numTraceBytesDropped") == 0, "Incomplete/dropped passes")
        ranges = profile.get("ranges", [])
        w.require(len(ranges) == 1 and ranges[0]["name"] == "WorkControl/isolated/" + report["phase"] and ranges[0]["index"] == 0,
                  "Wrong isolated range")
        w.require([m["name"] for m in ranges[0]["metrics"]] == profile["metricNames"] and profile["metricNames"], "Wrong metric set")
        for metric in ranges[0]["metrics"]:
            value = metric["value"]
            w.require(type(value) in (int, float) and math.isfinite(value) and value >= 0 and isinstance(metric.get("dimUnits"), list),
                      "Invalid metric value/units")
            if metric["name"] == "gpu__time_duration.sum":
                w.require(value > 0, "Empty range")
        for name in n.RAW_FILES:
            w.require(w.file_in(directory, "nvperf/" + name).stat().st_size > 0, "Empty raw counter evidence")
    w.require(report.get("measurementKind") == "diagnostic" and report.get("productionStatePublished") is False,
              "Replay publication or timing claim")
    w.require(report.get("sameRetainedExecution") is True and report.get("bindingPolicy") == "same-typed-slots-private-allocations",
              "Missing retained executable/binding identity")
    w.require(report.get("correctnessPasses") == 2 and report.get("phase") in ("early", "late"), "Wrong replay passes/phase")
    rows = report["bindings"]
    w.require(len(rows) == len(NAMES) and {r["name"] for r in rows} == NAMES, "Incomplete resource closure")
    bindings = {r["name"]: r for r in rows}
    if "bindingGenerationEvidence" in report:
        w.require(report["bindingGenerationEvidence"] == "allocation-id-and-frozen-graph-generation", "Unknown generation policy")
        allocation_ids = [row[key] for row in rows for key in ("sourceAllocation", "scratchAllocation")]
        w.require(all(type(i) is int and i > 0 for i in allocation_ids) and len(set(allocation_ids)) == len(allocation_ids),
                  "Missing or aliased allocation generation")
        w.require(type(report["heapAbi"]["nativeDescriptorHeap"]) is bool and report["heapAbi"]["maxBuffers"] >= len(BOUND),
                  "Invalid heap ABI")
    indices = [r["shaderIndex"] for r in rows if r["name"] in BOUND]
    w.require(len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < 0xffffffff for i in indices), "Aliased/invalid descriptor slot")
    sources = {r["sourceAddress"] for r in rows}
    scratch = {r["scratchAddress"] for r in rows}
    w.require(len(sources) == len(rows) and len(scratch) == len(rows) and not (sources & scratch) and "0" not in sources | scratch,
              "Scratch/production allocation alias")
    for name in GUARDS | {"arguments"}:
        w.require(bindings[name]["shaderIndex"] == 0xffffffff, "Production guard exposed as shader descriptor")
    for name in ("Input.spv", "Device.spv"):
        code = w.file_in(directory, name).read_bytes()
        w.require(len(code) >= 20 and len(code) % 4 == 0 and code[:4] == b"\x03\x02\x23\x07", "Missing actual SPIR-V")
    w.require(w.fnv((directory / "Input.spv").read_bytes()) == report["productionShader"]["spirvFnv1a64"], "SPIR-V differs from production binding")
    push = (directory / "Push.bin").read_bytes()
    w.require(len(push) == 136, "Unexpected push ABI")
    words = struct.unpack("<34I", push)
    for name, index in {"pages": 0, "groups": 1, "pageTable": 2, "params": 3, "header": 7, "rasterBindings": 27, "bins": 29}.items():
        w.require(words[index] == bindings[name]["shaderIndex"], "Push descriptor binding mismatch")
    bins = (directory / "bins-input.bin").read_bytes()[:64]
    raster = (directory / "rasterBindings-input.bin").read_bytes()[:68]
    w.require(len(bins) == 64 and len(raster) == 68, "Truncated nested binding")
    w.require(struct.unpack_from("<I", bins, 44)[0] == bindings["pixels"]["shaderIndex"], "Nested pixel binding mismatch")
    w.require(struct.unpack_from("<I", raster, 64)[0] == bindings["instances"]["shaderIndex"], "Nested instance binding mismatch")
    args = (directory / "arguments-input.bin").read_bytes()
    w.require(len(args) >= 60 and report["indirect"]["offset"] == 48, "Indirect range mismatch")
    dims = list(struct.unpack_from("<3I", args, 48))
    w.require(dims == report["indirect"]["dimensions"] and all(dims), "Wrong or empty dispatch")
    expected_stages = ["archive-inputs", "production-before"] + ["restore-inputs", "isolated-dispatch", "compare-output"] * 2
    if counter_passes:
        expected_stages += ["production-gate"] + ["restore-inputs", "isolated-dispatch", "compare-output"] * counter_passes
    expected_stages += ["production-after"]
    ledger = report["submissions"]
    w.require([s["stage"] for s in ledger] == expected_stages and
              all(s.get("accepted") is True and s.get("completed") is True for s in ledger), "Incomplete or unrestored submission")
    for name, row in bindings.items():
        w.require(type(row["bytes"]) is int and row["bytes"] > 0, "Invalid resource size")
        initial = w.file_in(directory, name + "-input.bin")
        w.require(initial.stat().st_size == row["bytes"], "Truncated resource")
        for index in range(2 + counter_passes):
            restored = w.file_in(directory, f"{index}-{name}-restored.bin")
            actual = w.file_in(directory, f"{index}-{name}-output.bin")
            expected = w.file_in(directory, "Control.bin") if name == "pixels" else initial
            w.require(equal_files(initial, restored), "Input restoration mismatch")
            w.require(equal_files(expected, actual), "Output or read-only resource mismatch")
        before = w.file_in(directory, name + "-production-before.bin")
        after = w.file_in(directory, name + "-production-after.bin")
        w.require(before.stat().st_size == row["bytes"] and equal_files(before, after), "Production state leak")
        if counter_passes:
            w.require(equal_files(before, w.file_in(directory, name + "-production-gate.bin")), "Pre-counter production leak")
    return report


def inspect_run(directory, case):
    process = w.load(directory / "Process.json")
    w.require(process.get("exitCode") == 0, "Failed replay process")
    capture = w.load(directory / "app/Capture.json")
    w.require(capture.get("status") == "capture_complete" and capture.get("measurementKind") == "diagnostic" and
              capture.get("workControlReplayRequested") is True, "Invalid capture lifecycle")
    w.require(len(capture["cases"]) == 1, "Ambiguous target frame")
    row = capture["cases"][0]
    identities = [w.snapshot_identity(directory / "app", row[key], case) for key in ("before", "control", "after")]
    w.require(identities[0] == identities[1] == identities[2], "Frozen workload/downstream changed")
    report = inspect_replay(directory / "app/replay")
    frozen = report["frozenIdentity"]
    w.require(frozen["snapshot"] == row["control"] and frozen["workloadCase"] == capture["workloadCase"] and
              frozen["camera"] == capture["camera"] and frozen["renderExtent"] == capture["renderExtent"], "Wrong frozen frame")
    selected = next(x for x in row["control"]["productionDispatches"] if x["phase"] == report["phase"])
    w.require(all(selected[k] == v for k, v in report["productionShader"].items()), "Production pipeline metadata mismatch")
    phase = "AfterStreamEarlyBins" if report["phase"] == "early" else "AfterStreamLateBins"
    w.require(report["indirect"]["dimensions"] == row["control"][phase]["softwareDispatch"], "Wrong same-frame indirect args")
    return {"phase": report["phase"], "identity": identities[0], "process": process,
            "inputSpirvSha256": w.digest(directory / "app/replay/Input.spv"),
            "deviceSpirvSha256": w.digest(directory / "app/replay/Device.spv"),
            "scope": report["scope"], "counterEligible": report["counterEligible"]}


def verify(output):
    manifest = w.load(output / "Manifest.json")
    w.require(manifest.get("protocol") == PROTOCOL and manifest.get("status") in ("complete", "expected-failure"), "Incomplete experiment")
    w.require(e.artifact_hashes(output) == manifest["files"], "Artifact inventory/hash mismatch")
    if manifest["status"] == "expected-failure":
        w.require(manifest["runs"] == ["01"], "Ambiguous failure run")
        return inspect_failure(output / "01", manifest["fault"])
    case = w.load(output / "Case.json")
    w.validate_case(case)
    rows = []
    for name in manifest["runs"]:
        directory = (output / name).resolve()
        w.require(directory.parent == output.resolve() and directory.is_dir(), "Invalid run directory")
        rows.append(inspect_run(directory, case))
    w.require(rows and e.independent_processes(rows), "Non-independent replay processes")
    w.require(all(row["identity"] == rows[0]["identity"] and row["scope"] == rows[0]["scope"] and
                  row["phase"] == rows[0]["phase"] and row["inputSpirvSha256"] == rows[0]["inputSpirvSha256"] and
                  row["deviceSpirvSha256"] == rows[0]["deviceSpirvSha256"] for row in rows), "Cross-process identity/scope drift")
    return {"status": "verified", "runs": len(rows), "scope": rows[0]["scope"],
            "counterEligible": rows[0]["counterEligible"], "counterImageReevaluated": False, "candidateAccepted": False}


def run(args):
    import psutil
    case = {**w.load(args.case), "rounds": 1, "sampleFrames": 8, "profileHoldSeconds": 0, "nsightTraceFrames": 0}
    w.validate_case(case)
    w.require(case["variant"] == "swWorkControl" and "primeCameraOffset" in case, "Requires primed WorkControl")
    assets = w.load(args.assets)
    w.require(assets.get("protocol") == "metallic-declared-assets-v1" and assets.get("stream") == w.ASSETS[case["sampleId"]], "Wrong assets")
    w.asset_metadata_matches(assets)
    output, exe = args.output.resolve(), args.exe.resolve()
    w.require(math.isfinite(args.timeout) and 0 < args.timeout <= 3600, "Timeout must be finite and within 1..3600 seconds")
    w.require(not args.fault or (args.runs == 1 and not args.counters), "Fault tests require one unprofiled process")
    w.require(not args.split_metric_passes or args.counters, "Split pass groups need counters")
    output.mkdir(parents=True, exist_ok=False)
    w.save(output / "Case.json", case)
    w.save(output / "Assets.json", assets)
    metrics = w.load(args.metrics) if args.metrics else n.DEFAULT_METRICS
    w.require(isinstance(metrics, list) and 1 <= len(metrics) <= 16 and all(isinstance(m, str) and m for m in metrics) and
              len(metrics) == len(set(metrics)), "Invalid metric request")
    w.save(output / "Metrics.json", metrics)
    manifest = {"protocol": PROTOCOL, "status": "running", "runs": []}
    if args.fault:
        manifest["fault"] = args.fault
    lock = w.ROOT / "build/shader-experiment.lock"
    acquired = False
    try:
        with lock.open("x") as stream:
            json.dump({"pid": os.getpid(), "output": str(output)}, stream)
        acquired = True
        sources = w.source_inventory()
        runtime = {str(p): w.digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]}
        build = exe.parent.parent
        manifest["build"] = e.validate_build(build, exe)
        (output / "CMakeCache.txt").write_bytes((build / "CMakeCache.txt").read_bytes())
        for name in ("WorkControlReplay.py", "NvPerf.py", "WorkloadCase.py", "DeepProfile.py", "ExperimentRunner.py"):
            (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        if args.counters:
            cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines()
                         if "=" in line and not line.startswith(("#", "//")))
            w.require(cache.get("METALLIC_ENABLE_NVPERF:BOOL") == "ON", "NvPerf disabled in build")
            sdk = Path(cache["METALLIC_NVPERF_SDK_ROOT:PATH"])
            runtime.update({str(p): w.digest(p) for p in (sdk / "NvPerf/bin/x64").glob("*.dll")})
            manifest["sdkHeaders"] = {str(p.relative_to(sdk)): w.digest(p) for folder in (sdk / "NvPerf/include", sdk / "redist/NvPerfUtility/include") for p in folder.rglob("*.h")}
        manifest.update(sources=sources, runtime=runtime, gpu=e.gpu_identity())
        for relative, digest in sources.items():
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((w.ROOT / relative).read_bytes())
            w.require(w.digest(dest) == digest, "Source archive drift")
        for index in range(args.runs):
            w.require(not any(p.info["name"] and p.info["name"].lower() in {exe.name.lower(), "ngfx.exe", "ngfx-replay.exe"}
                              for p in psutil.process_iter(["name"])), "Competing renderer/profiler")
            directory = output / f"{index + 1:02}"
            (directory / "app").mkdir(parents=True)
            w.save(directory / "Config.json", w.engine_config(case))
            env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
            env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"), METALLIC_FULL_ROAM_OUTPUT=str(directory / "app"),
                       METALLIC_FULL_ROAM_HIDDEN="1", METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
                       METALLIC_VK_PIPELINE_STATISTICS="0", METALLIC_NVPERF="1" if args.counters else "0", METALLIC_WORK_CONTROL_REPLAY="1",
                       METALLIC_NVPERF_METRICS=str(output / "Metrics.json"),
                       METALLIC_WORK_CONTROL_REPLAY_PHASE=args.phase)
            if args.split_metric_passes:
                env["METALLIC_NVPERF_SPLIT_METRIC_PASSES"] = "1"
            if args.fault:
                env["METALLIC_WORK_CONTROL_REPLAY_FAULT"] = args.fault
            command = [str(exe), "--sample", case["sampleId"]]
            w.save(directory / "Invocation.json", {"command": command, "environment": {k: v for k, v in env.items() if k.startswith("METALLIC_")}})
            try:
                d.run_process(command, env, directory, args.timeout)
            except ValueError:
                if not args.fault:
                    raise
            w.save(directory / "Analysis.json", inspect_failure(directory, args.fault) if args.fault else inspect_run(directory, case))
            manifest["runs"].append(directory.name)
            w.require(w.source_inventory() == sources and all(w.digest(Path(p)) == h for p, h in runtime.items()), "Source/runtime drift")
            w.asset_metadata_matches(assets)
        manifest["status"] = "expected-failure" if args.fault else "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if acquired:
            lock.unlink()
        manifest["files"] = e.artifact_hashes(output)
        w.save(output / "Manifest.json", manifest)
    return verify(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("run")
    live.add_argument("--case", type=Path, default=Path(__file__).with_name("WorkloadCase.MiniZorahHistory.json"))
    live.add_argument("--assets", type=Path, required=True)
    live.add_argument("--exe", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--phase", choices=("early", "late"), required=True)
    live.add_argument("--runs", type=int, choices=(1, 3), default=3)
    live.add_argument("--timeout", type=float, default=240)
    live.add_argument("--counters", action="store_true")
    live.add_argument("--metrics", type=Path)
    live.add_argument("--split-metric-passes", action="store_true", help="Use SDK pass groups to exercise multi-pass replay with the same metrics")
    live.add_argument("--fault", choices=FAULTS)
    offline = sub.add_parser("verify")
    offline.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args) if args.command == "run" else verify(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
