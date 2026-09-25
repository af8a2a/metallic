"""Bounded in-frame workload evidence and independent-run A/A qualification.

No Nsight UI or third-party Python dependencies. A/A qualifies repeatability,
not a candidate speedup, reference-renderer correctness or isolated replay.
"""
from __future__ import annotations

import argparse
import csv
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

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "swLegacy": (1, "streamClusterRasterLegacyMain"),
    "swPrepared": (0, "streamClusterRasterMain"),
    "swPlane": (2, "streamClusterRasterPlaneMain"),
    "swCooperative": (3, "streamClusterRasterCooperativeMain"),
    "swWorkBins": (4, "streamClusterRasterWorkBinsMain"),
    "swWorkControl": (5, "streamClusterRasterWorkControlMain"),
}
PHASES = ("AfterStreamEarlyBins", "AfterStreamLateBins")
OUTPUTS = ("VBuffer.visibility", "VBuffer.depth")
ASSETS = {"gpu-driven-zorah-full": "Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin",
          "gpu-driven-sample": "Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_in(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), "Evidence path escapes run directory")
    require(path.is_file(), f"Missing evidence: {relative}")
    return path


def validate_case(case):
    allowed = {"protocol", "id", "sampleId", "variant", "scope", "width", "height", "renderWidth", "renderHeight", "sampleFrames", "settleFrames",
               "warmupSeconds", "rounds", "aaRelativeSpreadLimit", "profileHoldSeconds", "nsightTraceFrames", "primeCameraOffset"}
    require(isinstance(case, dict) and not set(case) - allowed, "Unknown workload case fields")
    require(case.get("protocol") == "metallic-workload-case-v1", "Unsupported workload case protocol")
    require(isinstance(case.get("id"), str) and 0 < len(case["id"]) <= 128, "Missing case id")
    require(case.get("variant") in VARIANTS, "Unregistered shader variant")
    require(case.get("sampleId") in ASSETS, "Unregistered workload sample")
    require(case.get("scope") == "in-frame-early-late", "Only in-frame early/late is supported")
    for name, low, high in (("width", 64, 8192), ("height", 64, 8192), ("renderWidth", 32, 8192), ("renderHeight", 32, 8192), ("sampleFrames", 8, 512),
                            ("settleFrames", 2, 120), ("rounds", 1, 3)):
        require(type(case.get(name)) is int and low <= case[name] <= high, f"Invalid {name}")
    for name, low, high in (("warmupSeconds", 0, 120), ("aaRelativeSpreadLimit", 0.001, 0.5)):
        value = case.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, f"Invalid {name}")
    hold, trace = case.get("profileHoldSeconds", 0), case.get("nsightTraceFrames", 0)
    require(type(hold) in (int, float) and math.isfinite(hold) and 0 <= hold <= 300, "Invalid hold")
    require(type(trace) is int and 0 <= trace <= 3, "Invalid SDK trace frame count")
    require(not trace or (not hold and case["rounds"] == 1), "SDK trace needs one round and no hold")
    if "primeCameraOffset" in case:
        offset = case["primeCameraOffset"]
        require(isinstance(offset, list) and len(offset) == 3 and
                all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 100 for v in offset) and any(offset),
                "Invalid history camera offset")
        require(not hold and trace <= 1, "History priming needs no hold and at most one trace frame")
    return case


def engine_config(case):
    validate_case(case)
    return {**{k: case[k] for k in ("width", "height", "sampleFrames", "settleFrames", "rounds", "warmupSeconds")},
            **({"primeCameraOffset": case["primeCameraOffset"]} if "primeCameraOffset" in case else {}),
            "rasterComparison": True, "workloadCounters": False, "sampleId": case["sampleId"],
            "profileHoldSeconds": case.get("profileHoldSeconds", 0),
            "nsightTraceFrames": case.get("nsightTraceFrames", 0),
            "workloadCase": {k: case[k] for k in ("id", "variant", "scope")}}


def fnv(data):
    result = 14695981039346656037
    for byte in data:
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(result)


def snapshot_identity(directory, snapshot, case):
    require(snapshot["activeGroups"] > 0, "Empty active cut")
    identity = {k: snapshot[k] for k in ("activeGroups", "cutHash", "pageMappingsHash")}
    require(sum(snapshot[p]["softwareClusters"] for p in PHASES) > 0, "Zero software workload")
    if "primeCameraOffset" in case:
        require(snapshot[PHASES[1]]["softwareClusters"] > 0, "History case has zero late software workload")
    for phase in PHASES:
        bins = snapshot[phase]
        require(0 <= bins["softwareClusters"] <= bins["capacity"], "Software bin overflow")
        require(bins.get("softwareListHash"), "Missing stable software-list identity")
        args = bins.get("softwareDispatch")
        require(isinstance(args, list) and len(args) == 3 and all(type(x) is int and x >= 0 for x in args),
                "Invalid indirect dispatch")
        require(not bins["softwareClusters"] or math.prod(args) > 0, "Nonempty bin has empty dispatch")
        identity[phase] = bins
        counters = snapshot[phase.replace("Bins", "ClusterCull")]
        require(counters["candidateOverflow"] == 0, "Candidate overflow")
        identity[phase.replace("Bins", "ClusterCull")] = counters
    dispatches = snapshot["productionDispatches"]
    require(len(dispatches) == 2 and {d["phase"] for d in dispatches} == {"early", "late"}, "Missing/duplicate phase binding")
    expected_mode, expected_entry = VARIANTS[case["variant"]]
    for item in dispatches:
        require(item["mode"] == expected_mode and item["entryPoint"] == expected_entry, "Wrong production shader")
        module = "Features/GPUDriven/GPUDrivenStreamWorkRaster" if expected_mode >= 4 else "Features/GPUDriven/GPUDrivenStreamAsset"
        require(item["module"] == module, "Wrong shader module")
        require(item["spirvFnv1a64"] and item["queue"] == "graphics", "Missing SPIR-V or unexpected queue")
        require(item["scope"] == "production-dispatch" and item["dispatch"] == "indirect" and
                item["argumentOffsetBytes"] == 48, "Wrong dispatch binding")
        require(not item["forceHardware"] and not item["asyncRequested"] and not item["workloadEnabled"] and
                item["snapshotFrozen"] and item["paramsHzbValid"], "Unexpected production state/history")
    identity["productionDispatches"] = sorted(dispatches, key=lambda d: d["phase"])
    for resource in OUTPUTS:
        entry = snapshot[resource]
        path = file_in(directory, entry["file"])
        data = path.read_bytes()
        require(len(data) == entry["pixels"] * 4 and len(data) > 0, "Invalid output size")
        require(fnv(data) == entry["hash"], "Output readback hash mismatch")
        identity[resource] = {"sha256": hashlib.sha256(data).hexdigest(), "pixels": entry["pixels"]}
    return identity


def analyze_run(directory, case):
    directory = Path(directory).resolve()
    validate_case(case)
    capture = load(directory / "Capture.json")
    require(capture.get("status") == "capture_complete", f"Capture failed: {capture.get('error')}")
    require(capture.get("protocol") == case["protocol"], "Wrong capture protocol")
    require(capture["config"] == engine_config(case), "Engine did not execute the requested case")
    require(capture["workloadCase"] == engine_config(case)["workloadCase"], "Wrong workload selection")
    require(capture["outputExtent"] == [case["width"], case["height"]], "Output extent changed")
    require(capture["renderExtent"] == [case["renderWidth"], case["renderHeight"]], "Unexpected dynamic render extent")
    require(capture["hidden"] and not capture["validationRequested"], "Wrong timing environment")
    normal = (capture["measurementKind"] == "normal-timing" and not capture["graphicsCaptureInjected"]
              and not capture.get("gpuTraceInjected", False) and not capture.get("renderDocInjected", False)
              and not capture.get("pipelineStatisticsRequested", False)
              and not capture.get("nvPerfRequested", False))
    require(len(capture["cases"]) == case["rounds"], "Missing workload rounds")
    reference, residency = None, None
    timings = {"graphGpuMs": [], "softwareEarlyMs": [], "softwareLateMs": []}
    for index, row in enumerate(capture["cases"]):
        require(row["round"] == index + 1 and row["variant"] == case["variant"] and not row["fullHardware"], "Wrong round/variant")
        for side in ("before", "after"):
            current = snapshot_identity(directory, row[side], case)
            require(all(current[r]["pixels"] == case["renderWidth"] * case["renderHeight"] for r in OUTPUTS),
                    "Readback extent differs from declared render extent")
            if reference is None:
                reference = current
            require(current == reference, "Frozen input, binding or output changed within run")
        frames = load(file_in(directory, row["framesFile"]))
        require(len(frames) == row["frames"] == case["sampleFrames"], "Frame count mismatch")
        require(len({f["frame"] for f in frames}) == len(frames), "Duplicate measured frame")
        for frame in frames:
            require(len(frame["streaming"]) == 1, "Ambiguous streaming instance")
            streaming = frame["streaming"][0]
            shader = streaming["softwareRaster"]
            require(shader == {k: v for k, v in reference["productionDispatches"][0].items()
                                if k not in ("phase", "queue", "dispatch", "argumentOffsetBytes", "scope")},
                    "Measured frame shader/state differs from diagnostic binding")
            resident = {k: v for k, v in streaming.items() if k != "softwareRaster"}
            if residency is None:
                residency = resident
            require(resident == residency, "Residency changed during measured frames")
            # These are lifetime counters, incremented at texture publication.
            # Frozen means unchanged during the run, not zero since startup.
            for metric, suffix in (("graphGpuMs", "/RenderGraph GPU envelope"),
                                   ("softwareEarlyMs", "/Stream early/Software raster"),
                                   ("softwareLateMs", "/Stream late/Software raster")):
                values = [s["gpuMs"] for s in frame["scopes"] if s["path"].endswith(suffix)]
                require(len(values) == 1 and type(values[0]) in (int, float) and math.isfinite(values[0]) and values[0] >= 0,
                        f"Missing/ambiguous/nonfinite timing: {metric}")
                timings[metric].append(values[0])
            require(not any("diagnostic" in s["path"].lower() for s in frame["scopes"]), "Diagnostic work in timing frames")
    require(all(v > 0 for v in timings["graphGpuMs"]), "Zero GPU timing")
    timing_summary = {k: {"median": statistics.median(v), "samples": len(v)} for k, v in timings.items()}
    timing_summary["softwareTotalMs"] = {"median": statistics.median([a+b for a, b in zip(timings["softwareEarlyMs"], timings["softwareLateMs"])]),
                                        "samples": len(timings["graphGpuMs"])}
    require(timing_summary["softwareTotalMs"]["median"] > 0, "Zero software GPU timing")
    return {"protocol": "metallic-workload-validation-v1", "valid": True, "normalTiming": normal,
            "identity": {"case": case, "snapshot": reference, "camera": capture["camera"],
                         "renderExtent": capture["renderExtent"], "residency": {k: v for k, v in residency.items() if k not in ("textureUpgrades", "textureDowngrades")},
                         "historyInvalidationPolicy": capture["historyInvalidationPolicy"], "graph": capture["graph"]},
            "texturePublicationCounters": {k: residency[k] for k in ("textureUpgrades", "textureDowngrades")},
            "timings": timing_summary, "correctnessScope": "repeatability of visibility/depth; not candidate equivalence"}


def compare_runs(results, case):
    require(len(results) >= 3, "A/A needs at least three independent process runs")
    require(all(r["valid"] and r["normalTiming"] for r in results), "Invalid or instrumented run cannot qualify A/A")
    require(all(r["identity"] == results[0]["identity"] for r in results), "Cross-run workload/input/output mismatch")
    metrics = {}
    stable = True
    for metric in ("graphGpuMs", "softwareTotalMs"):
        medians = [r["timings"][metric]["median"] for r in results]
        spread = (max(medians) - min(medians)) / statistics.median(medians)
        metrics[metric] = {"runMedians": medians, "relativeSpread": spread}
        stable &= spread <= case["aaRelativeSpreadLimit"]
    return {"status": "stable" if stable else "inconclusive", "independentRuns": len(results),
            "statisticalUnit": "independent-process run median", "limit": case["aaRelativeSpreadLimit"],
            "metrics": metrics, "candidateAccepted": False}


def source_files():
    extensions = {".cpp", ".h", ".hpp", ".slang", ".json", ".cmake", ".txt"}
    return sorted(p for folder in ("Source", "Shaders", "Pipelines", "cmake")
                  for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in extensions) + [ROOT / "CMakeLists.txt"]


def source_inventory():
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}


def asset_metadata_matches(manifest):
    for relative, record in manifest["files"].items():
        path = file_in(ROOT, relative)
        stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtimeNs"]), f"Asset changed after hashing: {relative}")


def competition_summary(directory, pid):
    capture = load(directory / "Capture.json")
    windows = [(c["measurementBeginUnixMs"], c["measurementEndUnixMs"]) for c in capture["cases"]]
    peaks = {}
    own_samples = 0
    with (directory / "GpuProcesses.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            timestamp = datetime.datetime.fromisoformat(row["timestamp"]).timestamp() * 1000
            # PDH values cover the preceding one-second interval, not an instant.
            if not any(timestamp >= start and timestamp - 1000 <= end for start, end in windows):
                continue
            if int(row["pid"]) == pid:
                own_samples += 1
            else:
                key = (row["pid"], row["process"], row["engine"])
                peaks[key] = max(peaks.get(key, 0), float(row["utilization"]))
    return {"targetSamples": own_samples, "covered": own_samples > 0,
            "otherProcesses": [{"pid": p, "process": n, "engine": e, "peakPercent": v} for (p, n, e), v in peaks.items()],
            "semantics": "all PDH GPU engines, intervals overlapping measured host windows; background activity is retained"}


def execute(case, executable, output, runs, timeout, assets_path=None):
    validate_case(case)
    require(3 <= runs <= 10 and 1 <= timeout <= 900, "Invalid run budget")
    require(not case.get("nsightTraceFrames") and not case.get("profileHoldSeconds"), "A/A runner only accepts normal timing cases")
    require(executable.is_file(), "Build the workload executable first")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "Runner.py")
    save(output / "Case.json", case)
    save(output / "Config.json", engine_config(case))
    asset_manifest = load(assets_path) if assets_path else None
    if asset_manifest:
        require(asset_manifest["protocol"] == "metallic-declared-assets-v1" and asset_manifest["stream"] == ASSETS[case["sampleId"]], "Wrong asset closure")
        asset_metadata_matches(asset_manifest)
        save(output / "Assets.json", asset_manifest)
    sources = source_inventory()
    for relative in sources:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        require(digest(target) == sources[relative], "Source changed during snapshot")
    asset = ROOT / ASSETS[case["sampleId"]]
    stat = asset.stat()
    # Without --assets this stream has metadata identity only. The
    # limitation remains visible even when local fixed-state A/A is stable.
    assets = {"path": str(asset), "bytes": stat.st_size, "mtimeNs": stat.st_mtime_ns,
              "verification": "metadata-only", "contentSha256": None}
    if asset_manifest:
        assets.update(verification="sha256-manifest; metadata rechecked around runs",
                      contentSha256=asset_manifest["files"][ASSETS[case["sampleId"]]]["sha256"])
    runtime = {str(p.resolve()): digest(p) for p in [executable, *sorted(executable.parent.glob("*.dll"))]}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=15, creationflags=flags)
    require(gpu.returncode == 0 and gpu.stdout.strip(), "Cannot identify measured GPU")
    manifest = {"protocol": "metallic-workload-evidence-v1", "caseHash": canonical(case), "sources": sources,
                "runtime": runtime, "asset": assets, "gpu": gpu.stdout.strip(), "runs": [],
                "portableInputSnapshotComplete": False, "externalProfilerAbsenceVerified": False,
                "declaredAssetClosureHashed": asset_manifest is not None,
                "status": "running"}
    save(output / "Manifest.json", manifest)
    results = []
    try:
        for index in range(runs):
            if asset_manifest:
                asset_metadata_matches(asset_manifest)
            directory = output / f"run{index+1}"
            directory.mkdir()
            env = os.environ.copy()
            for key in ("METALLIC_DEBUG_CONTROL", "METALLIC_DEBUG_VALIDATION"):
                env.pop(key, None)
            env.update(METALLIC_FULL_ROAM_CONFIG=str(output / "Config.json"), METALLIC_FULL_ROAM_OUTPUT=str(directory),
                       METALLIC_FULL_ROAM_HIDDEN="1", METALLIC_NSIGHT_GRAPHICS_CAPTURE="0")
            monitor = competition = None
            with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr, \
                 (directory / "Gpu.csv").open("wb") as telemetry, (directory / "GpuProcesses.csv").open("wb") as process_gpu, \
                 (directory / "GpuProcesses.stderr.log").open("wb") as process_gpu_errors:
                try:
                    competition = subprocess.Popen(["powershell.exe", "-NoProfile", "-File", str(ROOT / "Tools/MeasureGpuCompetition.ps1")],
                                                   stdout=process_gpu, stderr=process_gpu_errors, creationflags=flags)
                    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,clocks.gr,temperature.gpu,power.draw",
                                                "--format=csv", "-l", "1"], stdout=telemetry, stderr=subprocess.STDOUT, creationflags=flags)
                    process = subprocess.Popen([str(executable), "--sample", case["sampleId"]], cwd=ROOT,
                                               env=env, stdout=stdout, stderr=stderr, creationflags=flags)
                    try:
                        deadline = time.monotonic() + timeout
                        complete_at = None
                        reclaimed = False
                        while process.poll() is None:
                            if (directory / "Capture.json").exists():
                                try:
                                    completed = load(directory / "Capture.json")
                                except (ValueError, OSError):
                                    completed = {}
                                if completed.get("status") == "failed":
                                    process.kill()
                                    process.wait()
                                    raise ValueError(f"Workload failed: {completed.get('error')}; owned process stopped")
                                if completed.get("status") == "capture_complete":
                                    complete_at = complete_at or time.monotonic()
                                    if time.monotonic() - complete_at >= 20:
                                        process.kill()
                                        process.wait()
                                        reclaimed = True
                                        break
                            if time.monotonic() >= deadline:
                                raise subprocess.TimeoutExpired(process.args, timeout)
                            time.sleep(0.25)
                        code = process.returncode
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise ValueError(f"Workload run {index+1} timed out; owned process stopped")
                    except BaseException:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        raise
                finally:
                    if competition is not None:
                        competition.terminate()
                        competition.wait(timeout=10)
                    if monitor is not None:
                        monitor.terminate()
                        monitor.wait(timeout=10)
            manifest["runs"].append({"directory": directory.name, "pid": process.pid, "exitCode": code,
                                     "teardownReclaimed": reclaimed})
            require(code == 0 or reclaimed, f"Workload process failed: {code}")
            logs = "\n".join((directory / n).read_text(encoding="utf-8", errors="replace") for n in ("stdout.log", "stderr.log"))
            require(not any(s in logs for s in ("Validation Error", "VUID-", "DeviceLost", "VK_ERROR_DEVICE_LOST")), "GPU/validation error in log")
            result = analyze_run(directory, case)
            environment = competition_summary(directory, process.pid)
            save(directory / "Competition.json", environment)
            save(directory / "Validation.json", result)
            results.append(result)
            print(f"run {index+1}/{runs}: workload identity/output verified", flush=True)
        require(source_inventory() == sources, "Sources changed during A/A")
        require(all(digest(Path(p)) == h for p, h in runtime.items()), "Runtime binary changed during A/A")
        require((asset.stat().st_size, asset.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns), "Asset metadata changed during A/A")
        if asset_manifest:
            asset_metadata_matches(asset_manifest)
        aa = compare_runs(results, case)
        save(output / "AA.json", aa)
        manifest["status"] = aa["status"]
        manifest["limitations"] = ["Portable asset copies are not archived; declared dependencies refer to content-hashed originals" if asset_manifest else "Asset content/dependency closure is not pinned", "Background GPU activity is recorded, not forcibly excluded",
                                   "Only synchronous graphics-queue in-frame early/late is covered", "No Nsight shader/range correlation asserted"]
        if any(r["teardownReclaimed"] for r in manifest["runs"]):
            manifest["limitations"].append("Completed capture process required reclamation after teardown timeout")
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["artifacts"] = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*")
                                  if p.is_file() and p.name != "Manifest.json"}
        save(output / "Manifest.json", manifest)
    return aa


def verify(directory):
    manifest = load(directory / "Manifest.json")
    require(manifest["protocol"] == "metallic-workload-evidence-v1", "Unknown evidence manifest")
    require(manifest["status"] in ("stable", "inconclusive") and "error" not in manifest, "Evidence run did not finish validation")
    for relative, expected in manifest["artifacts"].items():
        require(digest(file_in(directory, relative)) == expected, f"Artifact hash mismatch: {relative}")
    case = validate_case(load(file_in(directory, "Case.json")))
    require(canonical(case) == manifest["caseHash"], "Case hash mismatch")
    require(len(manifest["runs"]) >= 3 and len({r["directory"] for r in manifest["runs"]}) == len(manifest["runs"]), "Missing/duplicate runs")
    for row in manifest["runs"]:
        require(row["exitCode"] == 0 or row.get("teardownReclaimed") is True, "Failed process in evidence")
    results = [analyze_run(file_in(directory, r["directory"] + "/Capture.json").parent, case) for r in manifest["runs"]]
    result = compare_runs(results, case)
    require(result["status"] == manifest["status"], "Manifest verdict differs from recomputed A/A")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--exe", type=Path, default=ROOT / "build-release/Source/MetallicGPUDrivenSample.exe")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--timeout", type=int, default=600)
    run.add_argument("--assets", type=Path, help="Content-hashed dependency manifest from WorkloadAssets.py")
    config = commands.add_parser("config")
    config.add_argument("case", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "config":
            result = engine_config(load(args.case))
        elif args.command == "verify":
            result = verify(args.directory.resolve())
        else:
            result = execute(load(args.case), args.exe.resolve(), args.output.resolve(), args.runs, args.timeout, args.assets)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("status", "stable") == "stable" else 2
    except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
