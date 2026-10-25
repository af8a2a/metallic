"""Bounded frozen-source MiniZorah spatial demand A/B; no build or source edits."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import SpatialDemandEvidence as evidence
from ExperimentRunner import block_gain
from WorkloadAssets import dependencies

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "build/shader-experiment.lock"
LEGACY_PROTOCOL = "metallic-spatial-demand-ab-v1"
V2_PROTOCOL = "metallic-spatial-demand-ab-v2"
V3_PROTOCOL = "metallic-spatial-demand-ab-v3"
PROTOCOL = "metallic-spatial-demand-ab-v4"
MODERN_PROTOCOLS = (V3_PROTOCOL, PROTOCOL)
PROTOCOLS = (LEGACY_PROTOCOL, V2_PROTOCOL, V3_PROTOCOL, PROTOCOL)
METADATA_CAP_BYTES = 64 * 1024 * 1024
METRICS = {"demandMs": "Detail demand", "streamTraversalMs": "Stream traversal", "graphGpuMs": "RenderGraph GPU envelope",
           "demandSeedMs": "LOD clear / demand seed", "frontierMs": "LOD frontier"}
read, require = evidence.read, evidence.require


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(exe):
    paths = {exe, Path(__file__), Path(evidence.__file__), ROOT / "Tools/RunZorahFullRoam.ps1",
             ROOT / "Tools/MeasureGPUCompetition.ps1", ROOT / "Tools/Perf/ExperimentRunner.py",
             ROOT / "Tools/Perf/WorkloadCase.py", ROOT / "Tools/Perf/WorkloadAssets.py"}
    for folder, patterns in (("Source", ("*.cpp", "*.h")), ("Shaders", ("*.slang", "*.hlsli", "*.h")),
                             ("External", ("*.slang", "*.hlsli", "*.hlsl", "*.h")), ("Pipelines", ("*.json",))):
        for pattern in patterns:
            paths.update((ROOT / folder).rglob(pattern))
    paths.update(exe.parent.glob("*.dll"))
    return {str(p): file_hash(p) for p in sorted(paths)}


def check_assets(manifest):
    require(manifest["protocol"] == "metallic-declared-assets-v1" and
            manifest["stream"] == "Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin", "Wrong asset manifest")
    require(manifest["graph"] == "Pipelines/Samples/gpu_driven_realtime.metallic_graph.json" and
            manifest["scene"] == "Asset/MiniZorah/zorah_main_public.v2.gltf", "Wrong graph/scene asset closure")
    required = {p.relative_to(ROOT).as_posix() for p in dependencies(*(ROOT / manifest[k] for k in ("graph", "scene", "stream")))}
    require(required <= manifest["files"].keys(), "Missing asset dependency records")
    for relative, record in manifest["files"].items():
        checksum = record["sha256"]
        require(isinstance(checksum, str) and len(checksum) == 64 and all(c in "0123456789abcdefABCDEF" for c in checksum), "Missing asset content hash")
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT), "Asset path escapes repository")
        stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtimeNs"]), f"Asset metadata drift: {relative}")


def check_topology(state, arm, protocol):
    require(protocol in PROTOCOLS, "Unknown spatial demand protocol")
    result = {key: state.get(key) for key in ("spatialDemandHierarchy", "spatialDemandNodes", "spatialDemandBytes", "demandTaskCount")}
    result["lodTopologyBytes"] = state["immutableMetadata"]["topologyBytes"]
    if protocol == LEGACY_PROTOCOL:
        return result
    for key in ("spatialDemandLayout", "spatialDemandLeafGroups", "spatialDemandTaskNodes"):
        result[key] = state[key]
    for key in ("spatialDemandNodes", "spatialDemandBytes", "demandTaskCount", "lodTopologyBytes"):
        require(type(result[key]) is int and result[key] >= 0, f"Invalid topology counter: {key}")
    require(result["lodTopologyBytes"] > 0, "Missing immutable topology allocation")
    require(result["spatialDemandHierarchy"] == (arm == "B"), "Topology mode mismatch")
    if arm == "A":
        require(result["spatialDemandNodes"] == result["spatialDemandBytes"] == 0, "Baseline allocated candidate metadata")
        require((result["spatialDemandLayout"], result["spatialDemandLeafGroups"], result["spatialDemandTaskNodes"]) ==
                ("ordered-tiles", 64, 8), "Baseline topology contract changed")
    else:
        require(result["spatialDemandNodes"] > 0 and 0 < result["spatialDemandBytes"] <= METADATA_CAP_BYTES,
                "Candidate metadata exceeds declared 64 MiB gate or is empty")
        if protocol == V2_PROTOCOL:
            require(result["spatialDemandBytes"] <= result["lodTopologyBytes"], "Candidate metadata exceeds total topology")
        expected = (("ordered-task-spatial-v4", 8, 128) if protocol == PROTOCOL else
                    ("separate-group-refs-v3", 8, 128) if protocol == V3_PROTOCOL else ("group-refs-v2", 8, 32))
        require((result["spatialDemandLayout"], result["spatialDemandLeafGroups"], result["spatialDemandTaskNodes"]) ==
                expected, "Candidate topology contract changed")
    if protocol in MODERN_PROTOCOLS:
        for key in ("originalTopologyHash", "primitiveMetadataHash"):
            value = state[key]
            require(isinstance(value, str) and value.isascii() and value.isdecimal() and
                    len(value) <= 20 and str(int(value)) == value and int(value) < 2**64,
                    f"Invalid canonical decimal uint64 identity: {key}")
            result[key] = value
    return result


def load_run(directory, arm, diagnostic, protocol=PROTOCOL):
    base = directory / "run1"
    capture = read(base / "Capture.json")
    launch = read(directory / "Manifest.json")
    require(capture["protocol"] == "minizorah-streaming-regression-v1" and capture["status"] == "capture_complete", "Incomplete capture")
    require(capture["diagnosticRun"] == diagnostic and not capture["validationRequested"], "Incorrect timing/diagnostic mode")
    require(capture["demandStabilization"]["completed"] and capture["canonicalOwnerReset"]["completed"], "Unsettled source")
    require(capture["config"] == read(directory / "Config.json"), "Capture/config mismatch")
    sample = capture["streamingRegression"]["sample"]
    decoded = []
    for item in sample["captures"]:
        state = item["state"]
        require(state["spatialDemandHierarchy"] == (arm == "B"), "Requested spatial mode did not execute")
        directory_path = evidence.safe_file(base, item["directory"])
        value = evidence.decode(directory_path, state)
        value.update(seed=item["seed"], directory=str(directory_path))
        value["topology"] = check_topology(state, arm, protocol)
        decoded.append(value)
    require(decoded, "Missing semantic diagnostics")
    require(all(v["semantics"] == decoded[0]["semantics"] for v in decoded), "Within-run semantic instability")
    result = {"directory": str(directory), "arm": arm, "diagnostic": diagnostic, "decoded": decoded,
              "camera": capture["streamingRegression"]["camera"], "outputExtent": capture["outputExtent"],
              "renderExtent": capture["renderExtent"], "frontierShaderIdentity": sample["frontierShaderIdentity"],
              "maskShaderIdentity": sample["maskShaderIdentity"], "demandShaderIdentity": decoded[0]["demandShaderIdentity"],
              "gpu": launch["gpu"], "timings": {}}
    if not diagnostic:
        frames = [json.loads(line) for line in (base / sample["timings"]).read_text().splitlines()]
        require(len(frames) == 64 and len({f["frame"] for f in frames}) == 64, "Expected 64 unique timing frames")
        for metric, name in METRICS.items():
            values = []
            for frame in frames:
                require(not frame["diagnostic"], "Instrumented timing frame")
                found = [s["gpuMs"] for s in frame["scopes"] if (s["path"] == name or s["path"].endswith("/" + name)) and
                         (name == "RenderGraph GPU envelope" or "VBuffer (" in s["path"])]
                require(len(found) == 1 and isinstance(found[0], (int, float)) and math.isfinite(found[0]) and found[0] > 0,
                        f"Missing/duplicate/invalid metric {metric}")
                values.append(found[0])
            result["timings"][metric] = {"median": statistics.median(values), "values": values}
    for name in ("Gpu.csv", "GpuProcesses.csv"):
        path = base / name
        require(path.is_file() and len(path.read_text(encoding="utf-8-sig", errors="replace").splitlines()) > 1, f"Missing competition telemetry: {name}")
    return result


def consistent(rows, protocol=PROTOCOL):
    first = rows[0]
    for row in rows:
        for key in ("camera", "outputExtent", "renderExtent", "frontierShaderIdentity", "maskShaderIdentity", "demandShaderIdentity", "gpu"):
            require(row[key] == first[key], f"Workload identity drift: {key}")
        require(row["decoded"][0]["semantics"] == first["decoded"][0]["semantics"], "Cross-process semantic source/cut/RT mismatch")
        if protocol in MODERN_PROTOCOLS:
            for item in row["decoded"]:
                for key in ("lodTopologyBytes", "originalTopologyHash", "primitiveMetadataHash"):
                    require(item["topology"][key] == first["decoded"][0]["topology"][key],
                            f"Original topology/primitive metadata drift: {key}")


    if protocol == PROTOCOL:
        reference = first["decoded"][0]
        for row in rows:
            for item in row["decoded"]:
                require(item["topology"]["demandTaskCount"] == reference["topology"]["demandTaskCount"],
                        "v4 ordered task template count differs across arms/snapshots")
                for key in ("seededTasks", "processedTasks"):
                    require(item["demand"][key] == reference["demand"][key],
                            f"v4 actual ordered task count differs across arms/snapshots: {key}")


def quality_gate(rows, protocol=PROTOCOL):
    consistent(rows, protocol)
    reference = {item["seed"]: evidence.images(item["directory"]) for item in rows[0]["decoded"]}
    comparisons = []
    for row in rows:
        require({item["seed"] for item in row["decoded"]} == reference.keys(), "Quality seeds changed")
        for item in row["decoded"]:
            errors = evidence.compare_images(reference[item["seed"]], evidence.images(item["directory"]))
            comparisons.append({"directory": row["directory"], "seed": item["seed"], "images": errors})
    return {"passed": all(v["exact"] for c in comparisons for v in c["images"].values()), "comparisons": comparisons}


def schedule(protocol):
    require(protocol in PROTOCOLS, "Unknown protocol")
    return {"quality": "ABBA", "discovery": "ABBA" * 4,
            "confirmation": "ABBA" * (6 if protocol in MODERN_PROTOCOLS else 4)}


def analyze_stage(rows, protocol=PROTOCOL, stage="discovery"):
    require(stage in ("discovery", "confirmation"), "Unknown timing stage")
    order = schedule(protocol)[stage]
    require(len(rows) == len(order) and [r["arm"] for r in rows] == list(order),
            f"Incomplete {len(order) // 4}-block ABBA {stage} stage")
    consistent(rows, protocol)
    gains = {metric: block_gain([{"result": row} for row in rows], metric) for metric in METRICS}
    primary = "streamTraversalMs" if protocol in MODERN_PROTOCOLS else "demandMs"
    guard_metrics = ("graphGpuMs", "demandSeedMs", "frontierMs") if protocol in MODERN_PROTOCOLS else ("graphGpuMs",)
    p = gains[primary]
    decision = "confirmed_gain" if p["lower95"] > 0 and all(gains[m]["lower95"] > -.02 for m in guard_metrics) else "inconclusive"
    if p["upper95"] < 0 or any(gains[m]["upper95"] < -.02 for m in guard_metrics):
        decision = "confirmed_regression"
    spread = {arm: {metric: (max(v) - min(v)) / statistics.median(v)
                   for metric in METRICS for v in [[r["timings"][metric]["median"] for r in rows if r["arm"] == arm]]}
              for arm in "AB"}
    return {"effectAssessment": decision, "primaryMetric": primary, "guardMetrics": list(guard_metrics),
            "gains": gains, "relativeRange": spread,
            "noiseWarning": any(x > .1 for values in spread.values() for x in values.values()),
            "statisticalUnit": f"{len(order) // 4} independent ABBA blocks; log time-ratio Student t 95% CI, no frame pooling or outlier removal"}


def collect(args):
    raise RuntimeError("Spatial candidates were withdrawn; this tool is verification-only. Historical evidence remains verifiable.")
    output, exe = args.output.resolve(), args.exe.resolve()
    require(not output.exists() and exe.is_file(), "Use a fresh output and existing executable")
    config = read(args.config)
    require(config.get("sample") == "gpu-driven-sample" and config.get("dlssMode") == "Off" and
            config.get("temporalJitter") is False and config.get("stochasticTextureFiltering") is False and
            config.get("shadowOverrides", {}).get("sigmaDenoise") is False, "Require deterministic MiniZorah config")
    for key in ("detailDemandComparison", "blasQualityComparison", "rasterComparison", "freezeStreaming", "enableLodTransitionTelemetry"):
        require(not config.get(key), f"Incompatible configuration: {key}")
    assets = read(args.assets)
    check_assets(assets)
    identities = source_identity(exe)
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open("x") as stream:
        json.dump({"pid": os.getpid(), "output": str(output)}, stream)
    try:
        output.mkdir(parents=True)
        shutil.copyfile(args.assets, output / "Assets.json")
        config.update(streamingRegression={"settleFrames": 16, "sampleFrames": 64, "seeds": [0, 1]},
                      workloadCounters=False, workloadEvery=0, durationSeconds=1, warmupSeconds=10)
        manifest = {"protocol": PROTOCOL, "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "sourceIdentity": identities, "assetVerification": "Existing content manifest; size/mtime checked before and after every process",
                    "configuration": config, "schedule": schedule(PROTOCOL),
                    "policy": {"primaryMetric": "streamTraversalMs", "primaryLower95GreaterThan": 0,
                               "guardLower95GreaterThan": {m: -.02 for m in ("graphGpuMs", "demandSeedMs", "frontierMs")},
                               "candidateMetadataCapBytes": METADATA_CAP_BYTES,
                               "metadataCapScope": "Candidate spatialDemandBytes only; acceptance gate, not runtime/residency budget",
                               "baselineTopology": {"layout": "ordered-tiles", "leafGroups": 64, "taskNodes": 8},
                               "candidateTopology": {"layout": "ordered-task-spatial-v4", "leafGroups": 8, "taskNodes": 128},
                               "earlyStop": "Correctness failure or completed-stage significant regression only"},
                    "gpuExclusivityEstablished": False, "runs": []}
        save(output / "Manifest.json", manifest)
        all_results = []
        for stage, order in manifest["schedule"].items():
            stage_results = []
            for i, arm in enumerate(order):
                check_assets(assets)
                require(source_identity(exe) == identities, "Source/binary identity drift")
                directory = output / f"{stage}-{i + 1:02d}-{arm}"
                current = dict(config, captureQuality=stage == "quality", captureReflections=stage == "quality")
                config_path = output / f"{stage}-{i + 1:02d}-Config.json"
                save(config_path, current)
                env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_") and not k.startswith("VK_INSTANCE_LAYERS")}
                env["METALLIC_SPATIAL_DEMAND"] = str(int(arm == "B"))
                command = [args.powershell, "-NoProfile", "-File", str(ROOT / "Tools/RunZorahFullRoam.ps1"),
                           "-OutputRoot", str(directory), "-Executable", str(exe), "-RouteConfig", str(config_path),
                           "-Runs", "1", "-DurationSeconds", "1", "-WarmupSeconds", "10", "-NoVSync", "-TimeoutSeconds", str(args.timeout)]
                started = time.time()
                with (output / f"{stage}-{i + 1:02d}-launcher.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    code = process.wait()  # Child launcher owns the bounded GPU timeout and cleanup.
                record = {"stage": stage, "arm": arm, "directory": str(directory), "launcherPid": process.pid,
                          "startedUnix": started, "endedUnix": time.time(), "exitCode": code, "command": command}
                manifest["runs"].append(record)
                save(output / "Manifest.json", manifest)
                require(code == 0, "Capture launcher failed; preserve logs, no automatic retry")
                check_assets(assets)
                require(source_identity(exe) == identities, "Source/binary changed during process")
                result = load_run(directory, arm, stage == "quality")
                stage_results.append(result)
                consistent(stage_results)
                all_results.append(result)
                print(f"{stage} {i + 1}/{len(order)} {arm} complete", flush=True)
            if stage == "quality":
                gate = quality_gate(stage_results)
                save(output / "Quality.json", gate)
                require(gate["passed"], "Correctness failure: exact image gate; stop before timing")
            else:
                consistent(all_results)
                decision = analyze_stage(stage_results, PROTOCOL, stage)
                save(output / f"{stage}.json", decision)
                if decision["effectAssessment"] == "confirmed_regression":
                    manifest["stoppedAfter"] = stage
                    save(output / "Manifest.json", manifest)
                    break
        save(output / "Results.json", all_results)
        manifest["evidenceFiles"] = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
                                     if p.is_file() and p.name != "Manifest.json"}
        save(output / "Manifest.json", manifest)
    finally:
        LOCK.unlink()


def verify(args):
    output = args.output.resolve()
    manifest = read(output / "Manifest.json")
    protocol = manifest["protocol"]
    require(protocol in PROTOCOLS, "Unknown protocol")
    require(manifest["schedule"] == schedule(protocol), "Schedule differs from fixed protocol budget")
    if protocol != LEGACY_PROTOCOL:
        candidate = {"layout": "ordered-task-spatial-v4" if protocol == PROTOCOL else "separate-group-refs-v3", "leafGroups": 8, "taskNodes": 128} if protocol in MODERN_PROTOCOLS else {
            "layout": "group-refs-v2", "leafGroups": 8, "taskNodes": 32}
        require(manifest["policy"]["candidateMetadataCapBytes"] == METADATA_CAP_BYTES and
                manifest["policy"]["baselineTopology"] == {"layout": "ordered-tiles", "leafGroups": 64, "taskNodes": 8} and
                manifest["policy"]["candidateTopology"] == candidate, "Unknown topology acceptance policy")
    if protocol in MODERN_PROTOCOLS:
        require(manifest["policy"]["primaryMetric"] == "streamTraversalMs" and
                manifest["policy"]["primaryLower95GreaterThan"] == 0 and
                manifest["policy"]["guardLower95GreaterThan"] == {m: -.02 for m in ("graphGpuMs", "demandSeedMs", "frontierMs")},
                "Unknown traversal timing acceptance policy")
    for name, checksum in manifest["evidenceFiles"].items():
        require(file_hash(evidence.safe_file(output, name)) == checksum, f"Evidence changed: {name}")
    rows, last_end = {}, 0
    for record in manifest["runs"]:
        require(record["exitCode"] == 0 and record["startedUnix"] >= last_end and record["endedUnix"] > record["startedUnix"], "Failed/overlapping process")
        last_end = record["endedUnix"]
        require(record["stage"] in schedule(protocol), "Unexpected collection stage")
        directory = output / Path(record["directory"]).name
        rows.setdefault(record["stage"], []).append(load_run(directory, record["arm"], record["stage"] == "quality", protocol))
    require([r["arm"] for r in rows["quality"]] == list("ABBA"), "Incomplete quality ABBA")
    quality = quality_gate(rows["quality"], protocol)
    require(quality["passed"], "Image mismatch")
    require(list(rows) in (["quality", "discovery"], ["quality", "discovery", "confirmation"]), "Missing or reordered stages")
    expected_runs = [(stage, arm) for stage in rows for arm in schedule(protocol)[stage]]
    require([(r["stage"], r["arm"]) for r in manifest["runs"]] == expected_runs, "Runs differ from fixed stage order/budget")
    consistent([r for group in rows.values() for r in group], protocol)
    stages = {stage: analyze_stage(values, protocol, stage) for stage, values in rows.items() if stage != "quality"}
    require("confirmation" in stages or stages.get("discovery", {}).get("effectAssessment") == "confirmed_regression", "Missing independent confirmation")
    if protocol in MODERN_PROTOCOLS:
        require("confirmation" not in stages or stages["discovery"]["effectAssessment"] != "confirmed_regression",
                "Confirmation followed a mandatory completed-stage regression stop")
        if "confirmation" not in stages:
            require(manifest.get("stoppedAfter") == "discovery", "Missing declared discovery stop")
    result = {"protocol": protocol, "evidenceValidity": "valid", "quality": quality, "stages": stages}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Disabled: spatial candidates withdrawn; verification-only")
    run.add_argument("--exe", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--assets", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--powershell", default="pwsh")
    run.add_argument("--timeout", type=int, default=240)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args) if args.command == "run" else verify(args)


if __name__ == "__main__":
    main()
