"""Frozen MiniZorah single-pass NvPerf diagnostics, independent of timing acceptance."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np

import MeasureSpatialDemand as m
import SpatialDemandEvidence as evidence

ROOT = m.ROOT
LEGACY_PROTOCOL = "metallic-spatial-demand-counters-v1"
PROTOCOL = "metallic-spatial-demand-counters-v2"
PROFILE_PROTOCOL = "metallic-nvperf-streaming-v1"
RANGES = ["Streaming/Seed", "Streaming/Demand", "Streaming/Frontier"]
RAW_FILES = ("CounterAvailability.bin", "ConfigImage.bin", "CounterDataPrefix.bin", "CounterDataImage.bin")
DEFAULT_GROUPS = {
    "packing": ["gpu__time_duration.sum", "sm__ctas_launched.sum", "sm__warps_launched.sum",
                "smsp__inst_executed.sum", "smsp__thread_inst_executed.sum", "smsp__thread_inst_executed_pred_on.sum"],
    "memory": ["gpu__time_duration.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_realtime.sum",
               "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_realtime.sum", "lts__t_sectors.sum",
               "lts__t_sector_hit_rate.pct", "dram__bytes_op_read.sum", "dram__bytes_op_write.sum"],
    "issue": ["gpu__time_duration.sum", "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
              "sm__warps_active.avg.per_cycle_active", "smsp__warps_eligible.avg.per_cycle_active",
              "smsp__issue_active.avg.pct_of_peak_sustained_active",
              "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
              "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
              "smsp__warp_issue_stalled_membar_per_warp_active.pct"],
}
require, read, save = m.require, m.read, m.save


def validate_groups(groups):
    require(isinstance(groups, dict) and groups, "Expected metric-group mapping")
    for group, metrics in groups.items():
        require(isinstance(group, str) and group and all(c.isascii() and (c.isalnum() or c in "-_") for c in group), "Unsafe metric group name")
        require(isinstance(metrics, list) and 1 <= len(metrics) <= 16 and
                all(isinstance(v, str) and v and v.strip() == v for v in metrics) and
                len(set(metrics)) == len(metrics), "Each metric group requires 1..16 distinct exact names")
    return groups


def validate_profile(profile, metrics, row):
    require(profile["protocol"] == PROFILE_PROTOCOL and profile["status"] == "complete", "Incomplete streaming counter profile")
    require(profile["backend"] == "nvperf-vulkan-range" and profile["measurementKind"] == "diagnostic" and
            profile["clockPolicy"] == "unaltered" and profile["scope"] == "three-production-streaming-dispatches-on-graphics-queue",
            "Wrong counter backend/scope/policy")
    require(profile["requiredPasses"] == profile["passesCollected"] == 1, "Only complete single-pass collection is supported")
    require(profile["numRangesDropped"] == profile["numTraceBytesDropped"] == 0, "Counter data was dropped")
    require(profile["metricNames"] == metrics and profile["expectedRanges"] == RANGES, "Wrong metric/range plan")
    require(type(profile["vulkanVersionOfficiallySupported"]) is bool and type(profile["vulkanApiVersion"]) is int,
            "Missing SDK/Vulkan compatibility evidence")
    ranges, dispatches = profile["ranges"], profile["dispatches"]
    require(len(ranges) == 3 and {r["name"] for r in ranges} == set(RANGES) and
            {r["index"] for r in ranges} == {0, 1, 2}, "Missing/duplicate exact command ranges")
    require(len(dispatches) == 3 and [r["range"] for r in dispatches] == RANGES, "Missing/reordered actual dispatch identities")
    bindings = {"Streaming/Seed": (10, row["maskShaderIdentity"]),
                "Streaming/Demand": (13, row["demandShaderIdentity"]),
                "Streaming/Frontier": (5, row["frontierShaderIdentity"])}
    for dispatch in dispatches:
        name, identity = dispatch["range"], dispatch["identity"]
        phase, shader = bindings[name]
        require(identity["phase"] == phase and identity["queue"] == "graphics" and
                identity["scope"] == "production-dispatch-only-excludes-barriers", "Dispatch phase/queue/scope mismatch")
        for key in ("module", "entryPoint", "workgroupSize", "subgroupSize"):
            require(identity[key] == shader[key], f"Bound dispatch differs from snapshot shader: {key}")
        require(str(identity["spirvHash"]) == str(shader["spirvHash"]) and 0 < int(identity["spirvHash"]) < 2**64,
                "Bound dispatch SPIR-V does not match semantic snapshot")
        threads = identity["threadCount"]
        require(type(threads) is int and threads > 0, "Empty/invalid dispatch")
        groups = (threads + 63) // 64 if phase == 13 else threads
        require(identity["dispatchXYZ"] == [min(groups, 65535), (groups + 65534) // 65535, 1], "Actual dispatch dimensions disagree with phase contract")
    require(dispatches[0]["identity"]["threadCount"] == dispatches[2]["identity"]["threadCount"], "Seed/frontier instance counts differ")
    for r in ranges:
        require([v["name"] for v in r["metrics"]] == metrics, "Missing/reordered counter metrics")
        for metric in r["metrics"]:
            value = metric["value"]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0 and isinstance(metric["dimUnits"], list),
                    "Invalid metric value or missing SDK units")
            if metric["name"] == "gpu__time_duration.sum":
                require(value > 0, "Empty measured command range")
    return profile


def image_identity(directory):
    required = {"VBuffer.depth", "VBuffer.visibility"}
    result = {}
    for item in read(directory / "Images.json"):
        name = item["resource"]
        if name not in required:
            continue
        require(name not in result, "Duplicate correctness image")
        kind, channels = item["format"].rsplit("x", 1)
        dtype = {"float16": "<f2", "float32": "<f4", "unorm8": "u1", "srgb8": "u1", "uint32": "<u4"}[kind]
        raw = evidence.safe_file(directory, item["file"]).read_bytes()
        values = np.frombuffer(raw, dtype)
        require(values.size == item["width"] * item["height"] * int(channels) and np.isfinite(values).all(),
                "Invalid depth/visibility layout or nonfinite pixels")
        result[name] = {"sha256": evidence.digest(raw), "format": item["format"], "width": item["width"], "height": item["height"]}
    require(result.keys() == required, "Missing existing-graph depth/visibility snapshots")
    return result


def consistent(rows, spatial_protocol=m.PROTOCOL):
    m.consistent(rows, spatial_protocol)
    for row in rows:
        require(row["images"] == rows[0]["images"], "Cross-run depth/visibility bytes differ")


def load_run(directory, arm, metrics, spatial_protocol=m.PROTOCOL):
    base = directory / "run1"
    row = m.load_run(directory, arm, True, spatial_protocol)
    capture = read(base / "Capture.json")
    require(capture.get("nvPerfRequested") is True and capture["frames"] == 0 and not capture["captureQuality"] and
            not capture["streamingRegression"]["performanceComparable"], "Counter run entered ordinary timing or changed quality graph")
    require(not any(capture.get(k) for k in ("graphicsCaptureInjected", "gpuTraceInjected", "renderDocInjected",
                "pipelineStatisticsRequested", "validationRequested")), "Competing instrumentation")
    nv = capture["nvPerf"]
    require(nv["protocol"] == PROFILE_PROTOCOL and nv["complete"] is True and nv["frames"] == 1, "Missing one-frame counter collection")
    snapshots = capture["streamingRegression"]["sample"]["captures"]
    require(len(snapshots) == 2 and [s["directory"] for s in snapshots] == ["steady/before/seed0", "steady/after/seed0"] and
            all(s["role"].endswith("-counter-stability-snapshot-not-target-frame") for s in snapshots), "Missing before/after semantic stability snapshots")
    warmup = capture["streamingRegression"]["sample"]["counterWarmup"]
    require(len(warmup) == 16, "Missing fixed post-readback counter warmup")
    for state in warmup:
        require(state["diagnosticSourceIdentity"] == capture["streamingRegression"]["sourceIdentity"] and
                state["nodeDemandC1"] is True and state["nodeDemandC2"] is True and state["rtDemandMode"] == 1 and
                state["rtHistoryPacked"] is False and state["rtHistorySparseWrites"] is False and
                state["terminalOnlyMask"] is True and state["terminalMaskDiagnostics"] is False,
                "Post-readback warmup changed frozen source or production path")
        require(m.check_topology(state, arm, spatial_protocol) == row["decoded"][0]["topology"], "Post-readback topology drift")
    require(all(v["demand"][key] == row["decoded"][0]["demand"][key] for v in row["decoded"]
                for key in ("processedTasks", "visitedNodes", "testedGroups")), "Before/after normalization-count instability")
    images = [image_identity(Path(item["directory"])) for item in row["decoded"]]
    require(all(value == images[0] for value in images), "Before/after depth/visibility bytes differ")
    row["images"] = images[0]
    directory_nv = evidence.safe_file(base, nv["directory"])
    profile = validate_profile(read(directory_nv / "NvPerf.json"), metrics, row)
    for name in RAW_FILES:
        require((directory_nv / name).stat().st_size > 0, f"Missing raw counter evidence: {name}")
    row["profile"] = profile
    return row


def summarize_values(values):
    median = statistics.median(values)
    spread = (max(values) - min(values)) / median if median else (0 if max(values) == 0 else None)
    return {"values": values, "n": len(values), "median": median, "relativeSpread": spread,
            "repeatableWithin10Percent": len(values) > 1 and spread is not None and spread <= .1}


def summarize(rows, pilot, spatial_protocol=m.PROTOCOL):
    consistent(rows, spatial_protocol)
    require([r["arm"] for r in rows] == list("AB" if pilot else "ABBAAB"), "Wrong fixed independent-process schedule")
    first = rows[0]["profile"]
    reference_units = {r["name"]: [(v["name"], v["dimUnits"]) for v in r["metrics"]] for r in first["ranges"]}
    for row in rows:
        require({r["name"]: [(v["name"], v["dimUnits"]) for v in r["metrics"]] for r in row["profile"]["ranges"]} == reference_units,
                "Metric units changed across arms or repetitions")
        for key in ("chip", "queueFamily", "metricNames", "libraryDirectory", "vulkanApiVersion", "vulkanVersionOfficiallySupported", "dispatches"):
            require(row["profile"][key] == first[key], f"Counter/dispatch identity drift: {key}")
    result = {"measurementKind": "diagnostic", "candidateAccepted": False, "pilotOnly": pilot, "arms": {},
              "normalizationScope": "Raw sums divided by deferred semantic-snapshot work counts, not target-frame per-task attribution",
              "vulkanVersionOfficiallySupported": first["vulkanVersionOfficiallySupported"]}
    for arm in "AB":
        selected = [r for r in rows if r["arm"] == arm]
        arm_result = {}
        for name in RANGES:
            metrics = []
            for metric in first["metricNames"]:
                cells = [next(v for r in row["profile"]["ranges"] if r["name"] == name for v in r["metrics"] if v["name"] == metric) for row in selected]
                require(all(c["dimUnits"] == cells[0]["dimUnits"] for c in cells), "Metric units changed")
                value = dict(metric=metric, dimUnits=cells[0]["dimUnits"], **summarize_values([c["value"] for c in cells]))
                if name == "Streaming/Demand" and metric.endswith(".sum"):
                    value["perSnapshotWorkCount"] = {}
                    for counter in ("processedTasks", "visitedNodes", "testedGroups"):
                        denominators = [r["decoded"][0]["demand"][counter] for r in selected]
                        value["perSnapshotWorkCount"][counter] = summarize_values([c["value"] / n for c, n in zip(cells, denominators)]) if all(n > 0 for n in denominators) else None
                metrics.append(value)
            arm_result[name] = metrics
        # Same-range instruction ratios estimate average participating lanes;
        # pred-on additionally excludes lanes disabled by instruction predication.
        ratios = {}
        for name in RANGES:
            ratios[name] = {}
            for label, numerator, denominator in (
                ("meanActiveLanesPerWarpInstruction", "smsp__thread_inst_executed.sum", "smsp__inst_executed.sum"),
                ("meanPredOnLanesPerWarpInstruction", "smsp__thread_inst_executed_pred_on.sum", "smsp__inst_executed.sum"),
                ("meanPredOnLanesPerRealtimeWarpInstruction", "sm__thread_inst_executed_pred_on_realtime.sum", "sm__inst_executed_realtime.sum"),
                ("globalLoadL1SectorHitFraction", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_realtime.sum", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_realtime.sum")):
                values = []
                for row in selected:
                    data = {v["name"]: v["value"] for r in row["profile"]["ranges"] if r["name"] == name for v in r["metrics"]}
                    if numerator not in data or denominator not in data or data[denominator] == 0:
                        values = []
                        break
                    values.append(data[numerator] / data[denominator])
                if values:
                    ratios[name][label] = summarize_values(values)
        result["arms"][arm] = {"ranges": arm_result, "derivedRatios": ratios}
    return result


def build_identity(build, exe):
    cache_path = build / "CMakeCache.txt"
    cache = dict(line.split("=", 1) for line in cache_path.read_text().splitlines() if "=" in line and not line.startswith(("#", "//")))
    require(cache["CMAKE_BUILD_TYPE:STRING"] == "Release" and cache["METALLIC_ENABLE_NVPERF:BOOL"] in ("ON", "TRUE", "1"), "Need an existing NvPerf-enabled Release build")
    require(Path(cache["CMAKE_HOME_DIRECTORY:INTERNAL"]).resolve() == ROOT and exe.is_relative_to(build), "Build/executable/source mismatch")
    sdk = Path(cache["METALLIC_NVPERF_SDK_ROOT:PATH"]).resolve()
    parts = [sdk / p for p in ("NvPerf/include/nvperf_vulkan_target.h", "NvPerf/include/windows-desktop-x64/nvperf_host_impl.h",
             "redist/NvPerfUtility/include/NvPerfVulkan.h", "NvPerf/bin/x64/nvperf_grfx_host.dll")]
    require(all(p.is_file() for p in parts), "SDK is incomplete")
    files = {cache_path, exe, *exe.parent.glob("*.dll"), *(sdk / "NvPerf/bin/x64").glob("*.dll")}
    for folder in (sdk / "NvPerf/include", sdk / "redist/NvPerfUtility/include"):
        files.update(folder.rglob("*.h"))
    return {"directory": str(build), "sdkRoot": str(sdk), "files": {str(p): m.file_hash(p) for p in sorted(files)}}


def identity(exe, build):
    return {"source": {**m.source_identity(exe), str(Path(__file__).resolve()): m.file_hash(__file__)}, "build": build_identity(build, exe)}


def file_inventory(output):
    return {p.relative_to(output).as_posix(): m.file_hash(p) for p in sorted(output.rglob("*")) if p.is_file() and p != output / "Manifest.json"}


def collect(args):
    raise RuntimeError("Spatial candidates were withdrawn; this tool is verification-only. Historical evidence remains verifiable.")
    output, exe, build = args.output.resolve(), args.exe.resolve(), args.build_dir.resolve()
    require(not output.exists() and exe.is_file() and 1 <= args.timeout <= 900, "Fresh output, existing executable and bounded timeout required")
    groups = validate_groups(read(args.metrics) if args.metrics else DEFAULT_GROUPS)
    config = read(args.config)
    require(config.get("sample") == "gpu-driven-sample" and config.get("dlssMode") == "Off" and config.get("temporalJitter") is False and
            config.get("stochasticTextureFiltering") is False and config.get("shadowOverrides", {}).get("sigmaDenoise") is False, "Require the deterministic frozen MiniZorah configuration")
    for key in ("detailDemandComparison", "blasQualityComparison", "rasterComparison", "freezeStreaming", "enableLodTransitionTelemetry", "captureQuality", "captureReflections"):
        require(not config.get(key), f"Incompatible diagnostic config: {key}")
    assets = read(args.assets)
    m.check_assets(assets)
    frozen = identity(exe, build)
    pilot = args.runs == 1
    order = "AB" if pilot else "ABBAAB"
    config.update(captureQuality=False, captureReflections=False, workloadCounters=False, workloadEvery=0,
                  streamingRegression={"settleFrames": 16, "sampleFrames": 64, "seeds": [0], "nvPerf": True},
                  durationSeconds=1, warmupSeconds=10)
    m.LOCK.parent.mkdir(exist_ok=True)
    with m.LOCK.open("x") as lock:
        json.dump({"pid": os.getpid(), "output": str(output)}, lock)
    manifest = {"protocol": PROTOCOL, "spatialProtocol": m.PROTOCOL, "status": "running", "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "identity": frozen, "groups": groups, "order": order, "pilotOnly": pilot, "config": config, "runs": [],
                "gpuExclusivityEstablished": False, "clockPolicy": "unaltered", "ordinaryTimingAcceptance": False}
    try:
        output.mkdir(parents=True)
        shutil.copyfile(args.assets, output / "Assets.json")
        shutil.copyfile(build / "CMakeCache.txt", output / "CMakeCache.txt")
        for name in ("MeasureSpatialDemandCounters.py", "MeasureSpatialDemand.py", "SpatialDemandEvidence.py", "NvPerf.py"):
            shutil.copyfile(Path(__file__).with_name(name), output / name)
        save(output / "Config.json", config)
        save(output / "Manifest.json", manifest)
        all_rows = []
        for group, metrics in groups.items():
            metric_path = output / f"Metrics-{group}.json"
            save(metric_path, metrics)
            rows = []
            for index, arm in enumerate(order, 1):
                require(identity(exe, build) == frozen, "Source/build/SDK/runtime changed")
                m.check_assets(assets)
                directory = output / f"{group}-{index:02}-{arm}"
                env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_") and not k.startswith("VK_INSTANCE_LAYERS")}
                env.update(METALLIC_SPATIAL_DEMAND=str(int(arm == "B")), METALLIC_NVPERF="1", METALLIC_NVPERF_METRICS=str(metric_path))
                command = [args.powershell, "-NoProfile", "-File", str(ROOT / "Tools/RunZorahFullRoam.ps1"), "-OutputRoot", str(directory),
                           "-Executable", str(exe), "-RouteConfig", str(output / "Config.json"), "-Runs", "1", "-DurationSeconds", "1",
                           "-WarmupSeconds", "10", "-NoVSync", "-TimeoutSeconds", str(args.timeout)]
                started = time.time()
                code = None
                with (output / f"{group}-{index:02}-launcher.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait(timeout=args.timeout + 60)
                    except subprocess.TimeoutExpired:
                        # Own launcher tree only; never terminate unrelated renderers.
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=log, stderr=subprocess.STDOUT, check=False)
                        process.wait(timeout=15)
                record = {"group": group, "arm": arm, "directory": directory.name, "startedUnix": started, "endedUnix": time.time(),
                          "launcherPid": process.pid, "exitCode": code, "command": command,
                          "environment": {k: v for k, v in env.items() if k.startswith("METALLIC_")}}
                manifest["runs"].append(record)
                save(output / "Manifest.json", manifest)
                require(code == 0, "Counter process failed/timed out; preserve evidence, no automatic retry")
                require(identity(exe, build) == frozen, "Source/build/SDK/runtime changed during capture")
                m.check_assets(assets)
                row = load_run(directory, arm, metrics)
                rows.append(row)
                all_rows.append(row)
                consistent(all_rows)
                print(f"{group} {index}/{len(order)} {arm} complete", flush=True)
            save(output / f"Summary-{group}.json", summarize(rows, pilot))
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        m.LOCK.unlink()
        if output.is_dir():
            manifest["files"] = file_inventory(output)
            save(output / "Manifest.json", manifest)


def spatial_protocol_for(manifest):
    if manifest["protocol"] == LEGACY_PROTOCOL:
        return m.V3_PROTOCOL
    require(manifest["protocol"] == PROTOCOL and manifest["spatialProtocol"] == m.PROTOCOL,
            "Unknown counter spatial protocol")
    return manifest["spatialProtocol"]


def verify(output):
    output = output.resolve()
    manifest = read(output / "Manifest.json")
    require(manifest["protocol"] in (LEGACY_PROTOCOL, PROTOCOL) and manifest["status"] == "complete", "Incomplete counter experiment")
    spatial_protocol = spatial_protocol_for(manifest)
    groups = validate_groups(manifest["groups"])
    order = "AB" if manifest["pilotOnly"] else "ABBAAB"
    require(manifest["order"] == order and [(r["group"], r["arm"]) for r in manifest["runs"]] ==
            [(group, arm) for group in groups for arm in order], "Changed independent-process budget/order")
    require(file_inventory(output) == manifest["files"], "Counter evidence inventory/hash changed")
    rows, last_end = {group: [] for group in groups}, 0
    all_rows = []
    for record in manifest["runs"]:
        require(record["exitCode"] == 0 and record["startedUnix"] >= last_end and record["endedUnix"] > record["startedUnix"], "Failed/overlapping processes")
        last_end = record["endedUnix"]
        directory = evidence.safe_file(output, record["directory"])
        row = load_run(directory, record["arm"], groups[record["group"]], spatial_protocol)
        rows[record["group"]].append(row)
        all_rows.append(row)
    consistent(all_rows, spatial_protocol)
    summaries = {group: summarize(values, manifest["pilotOnly"], spatial_protocol) for group, values in rows.items()}
    for group, summary in summaries.items():
        require(summary == read(output / f"Summary-{group}.json"), "Recomputed diagnostic summary differs")
    return {"verified": True, "protocol": manifest["protocol"], "spatialProtocol": spatial_protocol, "counterImageReevaluated": False, "candidateAccepted": False,
            "pilotOnly": manifest["pilotOnly"], "summaries": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Disabled: spatial candidates withdrawn; verification-only")
    for name in ("exe", "build-dir", "config", "assets", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--metrics", type=Path, help="JSON mapping of group names to 1..16 exact metrics")
    run.add_argument("--runs", type=int, choices=(1, 3), default=3, help="Per-arm count; one is pilot only")
    run.add_argument("--timeout", type=int, default=240)
    run.add_argument("--powershell", default="pwsh")
    offline = commands.add_parser("verify")
    offline.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        collect(args)
    else:
        print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
