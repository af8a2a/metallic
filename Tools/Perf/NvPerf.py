"""Headless in-app NvPerf capture; single-pass diagnostic evidence, never timing acceptance."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

import DeepProfile as d
import ExperimentRunner as e
import WorkloadCase as w

PROTOCOL = "metallic-nvperf-experiment-v1"
DEFAULT_METRICS = ["gpu__time_duration.sum", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"]
RAW_FILES = ("CounterAvailability.bin", "ConfigImage.bin", "CounterDataPrefix.bin", "CounterDataImage.bin")


def validate_profile(profile, metrics):
    w.require(profile.get("protocol") == "metallic-nvperf-v1" and profile.get("status") == "complete",
              "Incomplete NvPerf collection")
    w.require(profile.get("backend") == "nvperf-vulkan-range" and
              profile.get("measurementKind") == "diagnostic" and profile.get("clockPolicy") == "unaltered",
              "Unexpected NvPerf backend/policy")
    w.require(profile.get("requiredPasses") == 1 and profile.get("passesCollected") == 1,
              "Incomplete or unsupported multipass collection")
    w.require(profile.get("numRangesDropped") == 0 and profile.get("numTraceBytesDropped") == 0,
              "Missing overflow evidence or dropped counters")
    w.require(profile.get("metricNames") == metrics and metrics and len(set(metrics)) == len(metrics), "Wrong metric request")
    w.require(type(profile.get("vulkanVersionOfficiallySupported")) is bool and
              type(profile.get("vulkanApiVersion")) is int, "Missing SDK compatibility evidence")
    ranges = profile.get("ranges", [])
    w.require(len(ranges) == 2 and {r["name"] for r in ranges} == {"WorkControl/early", "WorkControl/late"},
              "Missing/duplicate production ranges")
    w.require({r["index"] for r in ranges} == {0, 1}, "Wrong range indices")
    for row in ranges:
        w.require([m["name"] for m in row["metrics"]] == metrics, "Missing/reordered metrics")
        for metric in row["metrics"]:
            value = metric["value"]
            w.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid metric value")
            w.require(isinstance(metric.get("dimUnits"), list), "Missing dimensional units")
            if metric["name"] == "gpu__time_duration.sum":
                w.require(value > 0, "Empty GPU range")
    return profile


def inspect_run(directory, case, metrics):
    process = w.load(directory / "Process.json")
    w.require(process.get("exitCode") == 0, "Failed capture process")
    validation = w.analyze_run(directory / "app", case)
    capture = w.load(directory / "app/Capture.json")
    w.require(not validation["normalTiming"] and capture.get("nvPerfRequested") is True and
              capture.get("measurementKind") == "diagnostic", "NvPerf incorrectly labeled normal timing")
    w.require(not any(capture.get(k) for k in ("graphicsCaptureInjected", "gpuTraceInjected",
                  "renderDocInjected", "pipelineStatisticsRequested", "validationRequested")), "Competing instrumentation")
    collection = capture.get("nvPerf", {})
    w.require(collection.get("complete") is True and collection.get("frames") == 1 and
              collection.get("workloadCase") == capture["workloadCase"], "Wrong/incomplete captured workload")
    w.require(w.snapshot_identity(directory / "app", collection["snapshot"], case) == validation["identity"]["snapshot"],
              "NvPerf workload binding differs from validated frames")
    profile = validate_profile(w.load(directory / "app/nvperf/NvPerf.json"), metrics)
    for name in RAW_FILES:
        w.require(w.file_in(directory, "app/nvperf/" + name).stat().st_size > 0, "Empty raw counter evidence")
    return {"process": process, "validation": validation, "profile": profile}


def assess(rows):
    w.require(len(rows) == 3 and e.independent_processes(rows), "Need three independent serial captures")
    reference = rows[0]
    for row in rows:
        w.require(row["validation"]["identity"] == reference["validation"]["identity"], "Workload/readbacks changed")
        for key in ("chip", "queueFamily", "metricNames", "requiredPasses", "libraryDirectory",
                    "vulkanApiVersion", "vulkanVersionOfficiallySupported"):
            w.require(row["profile"][key] == reference["profile"][key], "Collection configuration changed")
    metrics = []
    for phase in ("WorkControl/early", "WorkControl/late"):
        for name in reference["profile"]["metricNames"]:
            measurements = [next(m for r in row["profile"]["ranges"] if r["name"] == phase
                                 for m in r["metrics"] if m["name"] == name) for row in rows]
            w.require(all(m["dimUnits"] == measurements[0]["dimUnits"] for m in measurements), "Metric units changed")
            values = [m["value"] for m in measurements]
            median = statistics.median(values)
            spread = (max(values) - min(values)) / median if median else (0 if max(values) == 0 else None)
            metrics.append({"range": phase, "metric": name, "values": values, "median": median,
                            "dimUnits": measurements[0]["dimUnits"], "relativeSpread": spread,
                            "repeatableWithin10Percent": spread is not None and spread <= .1})
    return {"runs": 3, "workloadAndReadbacksIdentical": True, "metrics": metrics,
            "vulkanVersionOfficiallySupported": reference["profile"]["vulkanVersionOfficiallySupported"],
            "candidateAccepted": False, "measurementKind": "diagnostic", "clockPolicy": "unaltered",
            "scope": "NvPerf command ranges around production dispatches; not source-line or PC attribution"}


def run(args):
    import psutil
    case = {**w.load(args.case), "rounds": 1, "sampleFrames": 8, "profileHoldSeconds": 0, "nsightTraceFrames": 0}
    w.validate_case(case)
    w.require(case["variant"] == "swWorkControl" and "primeCameraOffset" in case, "Need primed WorkControl case")
    metrics = w.load(args.metrics) if args.metrics else DEFAULT_METRICS
    w.require(isinstance(metrics, list) and 1 <= len(metrics) <= 16 and
              all(isinstance(m, str) and m for m in metrics) and len(set(metrics)) == len(metrics), "Invalid metric request")
    exe, output, build = args.exe.resolve(), args.output.resolve(), args.build_dir.resolve()
    build_info = e.validate_build(build, exe)
    cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines()
                 if "=" in line and not line.startswith(("#", "//")))
    w.require(cache.get("METALLIC_ENABLE_NVPERF:BOOL") == "ON", "NvPerf disabled in build")
    sdk = Path(cache["METALLIC_NVPERF_SDK_ROOT:PATH"])
    assets = w.load(args.assets)
    w.require(assets.get("protocol") == "metallic-declared-assets-v1" and
              assets.get("stream") == w.ASSETS[case["sampleId"]], "Wrong assets for workload")
    w.asset_metadata_matches(assets)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("Case.json", case), ("Metrics.json", metrics), ("Assets.json", assets)):
        w.save(output / name, value)
    manifest = {"protocol": PROTOCOL, "status": "running", "runs": [], "build": build_info}
    lock = w.ROOT / "build/shader-experiment.lock"
    acquired = False
    try:
        with lock.open("x") as stream:
            json.dump({"pid": os.getpid(), "output": str(output)}, stream)
        acquired = True
        sources = w.source_inventory()
        for relative, digest in sources.items():
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((w.ROOT / relative).read_bytes())
            w.require(w.digest(dest) == digest, "Source archive drift")
        for name in ("NvPerf.py", "DeepProfile.py", "ExperimentRunner.py", "WorkloadCase.py"):
            (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
        (output / "CMakeCache.txt").write_bytes((build / "CMakeCache.txt").read_bytes())
        runtime = {str(p): w.digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll")), *sorted((sdk / "NvPerf/bin/x64").glob("*.dll"))]}
        manifest.update(runtime=runtime, sources=sources, gpu=e.gpu_identity(), sdkRoot=str(sdk),
                        sdkHeaders={str(p.relative_to(sdk)): w.digest(p) for base in (sdk / "NvPerf/include", sdk / "redist/NvPerfUtility/include") for p in base.rglob("*.h")})
        rows = []
        for index in range(1, 4):
            w.require(not any(p.info["name"] and p.info["name"].lower() in
                {exe.name.lower(), "ngfx.exe", "ngfx-replay.exe"} for p in psutil.process_iter(["name"])), "Competing profiler/renderer")
            w.require(w.source_inventory() == sources and all(w.digest(Path(p)) == h for p, h in runtime.items()), "Source/runtime drift")
            w.require(e.gpu_identity() == manifest["gpu"], "GPU identity changed")
            w.asset_metadata_matches(assets)
            directory = output / f"{index:02}"
            (directory / "app").mkdir(parents=True)
            w.save(directory / "Config.json", w.engine_config(case))
            env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
            env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"), METALLIC_FULL_ROAM_OUTPUT=str(directory / "app"),
                       METALLIC_FULL_ROAM_HIDDEN="1", METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
                       METALLIC_VK_PIPELINE_STATISTICS="0", METALLIC_NVPERF="1", METALLIC_NVPERF_METRICS=str(output / "Metrics.json"))
            command = [str(exe), "--sample", case["sampleId"]]
            w.save(directory / "Invocation.json", {"command": command, "environment": {k: v for k, v in env.items() if k.startswith("METALLIC_")}})
            d.run_process(command, env, directory, args.timeout)
            row = inspect_run(directory, case, metrics)
            w.require(w.source_inventory() == sources and all(w.digest(Path(p)) == h for p, h in runtime.items()), "Source/runtime changed during capture")
            w.asset_metadata_matches(assets)
            w.save(directory / "Analysis.json", row)
            rows.append(row)
            manifest["runs"].append(directory.name)
            print(json.dumps({"completed": index, "backend": "nvperf"}), flush=True)
        w.save(output / "Summary.json", assess(rows))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if acquired:
            lock.unlink()
        manifest["files"] = e.artifact_hashes(output)
        w.save(output / "Manifest.json", manifest)
    return {"status": "complete", "directory": str(output)}


def verify(directory):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest.get("protocol") == PROTOCOL and manifest.get("status") == "complete", "Incomplete experiment")
    for name, digest in manifest["files"].items():
        w.require(w.digest(w.file_in(directory, name)) == digest, f"Changed evidence: {name}")
    w.require(e.artifact_hashes(directory) == manifest["files"], "Evidence inventory differs from manifest")
    case, metrics = w.load(directory / "Case.json"), w.load(directory / "Metrics.json")
    w.require(len(manifest["runs"]) == 3 and len(set(manifest["runs"])) == 3, "Wrong run inventory")
    rows = []
    for name in manifest["runs"]:
        path = w.file_in(directory, name + "/Analysis.json").parent
        row = inspect_run(path, case, metrics)
        w.require(row == w.load(path / "Analysis.json"), "Analysis mismatch")
        rows.append(row)
    w.require(assess(rows) == w.load(directory / "Summary.json"), "Summary mismatch")
    return {"verified": True, "runs": 3, "rawHashesVerified": True, "counterImageReevaluated": False,
            "verifierSha256": w.digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("run")
    for name in ("case", "assets", "exe", "build-dir", "output"):
        live.add_argument("--" + name, type=Path, required=True)
    live.add_argument("--metrics", type=Path)
    live.add_argument("--timeout", type=int, default=240)
    offline = commands.add_parser("verify")
    offline.add_argument("directory", type=Path)
    args = parser.parse_args()
    w.require(args.command != "run" or 1 <= args.timeout <= 900, "Invalid timeout")
    print(json.dumps(run(args) if args.command == "run" else verify(args.directory.resolve()), indent=2))


if __name__ == "__main__":
    main()
