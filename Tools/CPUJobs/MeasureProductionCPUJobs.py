"""Run fixed-scene editor A/B blocks and retain OS, frame and GPU-memory evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

from ProcessResources import monitor_command


DOMAIN_NAMES = ["Other", "Geometry", "CLAS", "CLASScratch", "RayTracing", "MaterialTextures", "FrameResources", "Upload"]
SUCCESS = {"passed", "capture_complete", "complete"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values) -> dict | None:
    values = sorted(values)
    if not values:
        return None
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid nonfinite/negative measurement")
    def quantile(q):
        return values[max(0, math.ceil(q * len(values)) - 1)]
    return {"samples": len(values), "mean": statistics.mean(values), "p50": quantile(.5),
            "p95": quantile(.95), "p99": quantile(.99), "min": values[0], "max": values[-1]}


def summarize_device_memory(frames: list[dict], capture: dict) -> dict:
    snapshots = [row["deviceMemory"] for row in frames if isinstance(row.get("deviceMemory"), dict)]
    for key in ("deviceMemoryAfterLoad", "deviceMemoryAfterWarmup", "deviceMemoryAfterCapture"):
        if isinstance(capture.get(key), dict):
            snapshots.append(capture[key])
    maxima = defaultdict(list)
    domain_peaks = defaultdict(list)
    tracked = defaultdict(list)
    missing = []
    for snapshot in snapshots:
        for key in ("trackedAllocationBytes", "trackedDeviceLocalBytes", "peakTrackedAllocationBytes", "peakTrackedDeviceLocalBytes"):
            if isinstance(snapshot.get(key), int):
                tracked[key].append(snapshot[key])
        heaps = snapshot.get("heaps", [])
        local = [heap for heap in heaps if heap.get("deviceLocal") is True]
        for source, destination in (
            ("allocationBytes", "deviceLocalAllocatedSampledPeakBytes"),
            ("blockBytes", "deviceLocalBlockSampledPeakBytes"),
            ("usageBytes", "deviceLocalUsageSampledPeakBytes"),
        ):
            if local and all(isinstance(heap.get(source), (int, float)) for heap in local):
                maxima[destination].append(sum(heap[source] for heap in local))
        if isinstance(snapshot.get("availableBytes"), (int, float)):
            maxima["availableBytes"].append(snapshot["availableBytes"])
        domains = snapshot.get("domains", [])
        if isinstance(domains, list):
            domain_items = [(DOMAIN_NAMES[index] if index < len(DOMAIN_NAMES) else str(index), value)
                            for index, value in enumerate(domains)]
        elif isinstance(domains, dict):
            domain_items = domains.items()
        else:
            domain_items = []
        for name, domain in domain_items:
            if isinstance(domain, dict) and isinstance(domain.get("peakAllocationBytes"), (int, float)):
                domain_peaks[str(name)].append(domain["peakAllocationBytes"])
    result = {}
    for key in ("deviceLocalAllocatedSampledPeakBytes", "deviceLocalBlockSampledPeakBytes",
                "deviceLocalUsageSampledPeakBytes"):
        result[key] = max(maxima[key]) if maxima[key] else None
        if result[key] is None:
            missing.append(f"No RHI device-local heap {key} samples")
    result["minimumAvailableBytes"] = min(maxima["availableBytes"]) if maxima["availableBytes"] else None
    result["domainAllocationPeakBytes"] = {name: max(values) for name, values in domain_peaks.items()}
    for key in ("trackedAllocationBytes", "trackedDeviceLocalBytes", "peakTrackedAllocationBytes", "peakTrackedDeviceLocalBytes"):
        result[key] = max(tracked[key]) if tracked[key] else None
        if result[key] is None:
            missing.append(f"No RHI global {key} counter")
    if not domain_peaks:
        missing.append("No RHI per-domain lifetime allocation high-water counters")
    result["snapshotCount"] = len(snapshots)
    result["missing"] = missing
    result["definitions"] = {
        "deviceLocalAllocatedSampledPeakBytes": "Maximum observed sum of allocationBytes over device-local Vulkan heaps; application/VMA allocations, not process private memory.",
        "deviceLocalBlockSampledPeakBytes": "Maximum observed sum of VMA blockBytes over device-local heaps; do not add to allocationBytes or heap usage.",
        "deviceLocalUsageSampledPeakBytes": "Maximum observed sum of RHI usageBytes over device-local heaps; may include driver-accounted usage. Not an exact instantaneous lifetime VRAM peak.",
        "domainAllocationPeakBytes": "Each domain's own allocation high-water (host and device allocations according to RHI); domain peaks occur at different times and must not be summed into a simultaneous total.",
        "peakTrackedDeviceLocalBytes": "Simultaneous lifetime high-water at published/tracked RHI allocation/free events under its budget mutex; excludes temporary native allocations failing before tracking, external SDK allocations and driver memory.",
        "peakTrackedAllocationBytes": "Simultaneous lifetime high-water of all RHI-tracked allocations, host plus device. Not a sum of domain peaks.",
        "trackedAllocationBytes": "Maximum observed current RHI total (host plus device), distinct from its lifetime event high-water.",
        "trackedDeviceLocalBytes": "Maximum observed current RHI device-local total, distinct from its lifetime event high-water.",
        "sampling": "Measured frames plus explicitly exported after-load/warmup/capture snapshots. Peaks between snapshots are not established by heap samples.",
    }
    return result


def summarize_frames(frames: list[dict], capture: dict) -> dict:
    if not frames:
        raise ValueError("No measured frames")
    scopes = {entry["id"]: entry for entry in capture.get("scopes", [])}
    scope_values = defaultdict(list)
    scheduling = defaultdict(list)
    exclusive = defaultdict(list)
    frame_path = defaultdict(list)
    gpu_frames = []
    missing = []
    identities = [row.get("frame", row.get("frameIndex")) for row in frames]
    frame_range = None
    if all(isinstance(identity, int) for identity in identities):
        if any(b <= a for a, b in zip(identities, identities[1:])):
            raise ValueError("Measured frame identities must be unique and increasing")
        frame_range = {"first": identities[0], "last": identities[-1]}
    else:
        missing.append("Measured frame IDs not exported; frame range unavailable")
    for row in frames:
        if row.get("phase") in {"loading", "warmup", "prewarm"}:
            raise ValueError("Loading/warmup frame appeared in measured frame evidence")
        if not isinstance(row.get("frameMs"), (int, float)) or row["frameMs"] <= 0:
            raise ValueError("Frame time must be finite and positive")
        for path, value in row.get("cpuScopesMs", {}).items():
            if value is not None:
                scope_values[path].append(value)
        frame_gpu = row.get("gpuFrameMs")
        for scope in row.get("scopes", []):
            definition = scopes.get(scope["id"])
            if definition is None:
                raise ValueError(f"Scope ID missing from Capture.json: {scope['id']}")
            path = definition["path"]
            if scope.get("cpuMs") is not None:
                scope_values[path].append(scope["cpuMs"])
            if path.split("/")[-1] == "RenderGraph GPU envelope" and scope.get("gpuMs") is not None:
                if frame_gpu is not None:
                    raise ValueError("Ambiguous multiple GPU frame envelopes")
                frame_gpu = scope["gpuMs"]
        if frame_gpu is not None:
            gpu_frames.append(frame_gpu)
        metric = row.get("scheduling") or {}
        if metric.get("enabled"):
            for key, value in metric.items():
                if key.endswith("Ns") and isinstance(value, (int, float)):
                    scheduling[key].append(value)
        for key, value in row.get("exclusiveCpuSegmentsMs", {}).items():
            if value is not None:
                exclusive[key].append(value)
        segments = row.get("cpuFramePathSegmentsMs")
        if segments is not None:
            if row.get("cpuFramePathContract") != "nonoverlapping-coordinator-wall-segments-v1":
                raise ValueError("Unknown coordinator frame-path ownership contract")
            last = 0.
            totals = defaultdict(float)
            for segment in segments:
                start, end = segment["startMs"], segment["endMs"]
                if not all(math.isfinite(value) for value in (start, end)) or start < last or end < start or end > row["frameMs"]:
                    raise ValueError("Coordinator intervals overlap or escape the frame envelope")
                totals["Unattributed frame envelope"] += start - last
                totals[segment["name"]] += end - start
                last = end
            totals["Unattributed frame envelope"] += row["frameMs"] - last
            for key, value in totals.items():
                frame_path[key].append(value)
    if not gpu_frames:
        missing.append("No application GPU frame envelope timings")
    elif len(gpu_frames) != len(frames):
        missing.append(f"GPU envelope missing for {len(frames) - len(gpu_frames)} frames")
    if not scheduling:
        missing.append("Scheduling diagnostics disabled or not exported")
    if not scope_values:
        missing.append("No application CPU scope timings")
    result = {
        "frameCount": len(frames), "frameMs": distribution([row["frameMs"] for row in frames]),
        "measuredFrameRange": frame_range,
        "gpuFrameMs": distribution(gpu_frames),
        "cpuScopesMs": {path: distribution(values) for path, values in sorted(scope_values.items())},
        "schedulingNs": {key: distribution(values) for key, values in sorted(scheduling.items())},
        "exclusiveCpuSegmentsMs": {key: distribution(values) for key, values in sorted(exclusive.items())},
        "cpuFramePathSegmentsMs": {key: distribution(values) for key, values in sorted(frame_path.items())},
        "deviceMemory": summarize_device_memory(frames, capture),
        "missing": missing,
        "definitions": {
            "frameMs": capture.get("frameTimeDefinition", "Application-exported frame wall interval"),
            "gpuFrameMs": "Application GPU envelope; excludes unrelated queues/present unless the application explicitly defines otherwise.",
            "cpuScopesMs": "Individual scope wall occurrences, including nested scopes. Do not add parent and children or parallel workers. This is not an inferred DAG critical path.",
            "schedulingNs": "Original non-exclusive/nested SchedulingMetrics durations. Worker totals are parallel-work sums, not wall-clock critical paths.",
            "exclusiveCpuSegmentsMs": capture.get("exclusiveCpuSegmentsDefinition") if exclusive else None,
            "cpuFramePathSegmentsMs": "Verified non-overlapping coordinator intervals within each frame, plus explicit unattributed remainder. This is the coordinator wall path, not a TaskGraph longest path.",
        },
    }
    if exclusive and not capture.get("exclusiveCpuSegmentsDefinition"):
        raise ValueError("Exclusive CPU segments require an application ownership/overlap definition")
    return result


def parse_workers(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Workers must be comma-separated integers") from error
    if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Worker counts must be distinct and positive")
    return values


def effective_identity(capture: dict, workers: int, purpose="performance", diagnostics=False, strict_conditions=False) -> dict:
    if capture.get("workerCount") != workers:
        raise ValueError("Application did not use the requested TaskSystem worker count")
    if capture.get("validationRequested") is not (purpose == "validation"):
        raise ValueError("Capture validation state differs from the declared purpose")
    identity = {key: capture.get(key) for key in
                ("scene", "camera", "outputExtent", "renderExtent", "recordingWorkerLimit", "validationRequested")}
    identity["graph"] = capture.get("effectiveGraph", capture.get("graph"))
    for key in ("scene", "camera", "graph", "outputExtent", "renderExtent", "recordingWorkerLimit"):
        if identity[key] is None:
            raise ValueError(f"Application omitted effective workload identity: {key}")
    if strict_conditions:
        sync_requested = capture.get("synchronizationValidationRequested")
        if not isinstance(sync_requested, bool) or sync_requested is not capture.get("config", {}).get("synchronizationValidation", False):
            raise ValueError("Synchronization-validation request differs from application config")
        if purpose == "performance" and sync_requested:
            raise ValueError("Performance capture must disable synchronization validation")
        if capture.get("synchronizationValidationEnabled") is not sync_requested:
            raise ValueError("Native synchronization-validation state differs from the requested config")
        for key in ("validationEnabled", "validationMessengerActive"):
            if capture.get(key) is not (purpose == "validation"):
                raise ValueError(f"Native Vulkan {key} differs from the declared purpose")
        if capture.get("diagnostics") is not diagnostics:
            raise ValueError("Application diagnostics state differs from runner configuration")
        if capture.get("graphicsCaptureInjected") is not False or capture.get("shaderDebugMode") != "disabled":
            raise ValueError("Production comparison requires capture injection/debug symbols disabled")
        if capture.get("vsyncRequested") is not False:
            raise ValueError("Production comparison requires VSync disabled")
        reflex = capture.get("reflex") or {}
        if reflex.get("requestedMode") != 0 or reflex.get("frameLimitUs") != 0:
            raise ValueError("Production comparison requires Reflex off and frame limit zero")
        identity.update({key: capture.get(key) for key in
                         ("graphicsCaptureInjected", "shaderDebugMode", "measurementKind", "vsyncRequested",
                          "framesInFlight", "swapchainImageCount", "presentModeFromConsole", "synchronizationValidationRequested",
                          "validationEnabled", "synchronizationValidationEnabled", "validationMessengerActive")})
        identity["reflex"] = {key: reflex.get(key) for key in ("available", "requestedMode", "frameLimitUs")}
    return identity


def benchmark_environment(diagnostics: bool, validation: bool) -> tuple[dict, dict]:
    environment = os.environ.copy()
    prefixes = ("METALLIC_FULL_ROAM_", "METALLIC_NSIGHT_", "METALLIC_SHADER_TRACE", "METALLIC_SMOKE_TEST_",
                "METALLIC_DEBUG_", "METALLIC_CPU_JOB_BENCH_")
    names = {"METALLIC_SHADER_CAPTURE_SYMBOLS", "METALLIC_RENDER_SCHEDULING_DIAGNOSTICS", "METALLIC_REFLEX_MODE"}
    removed = [key for key in environment if key.startswith(prefixes) or key in names]
    for key in removed:
        environment.pop(key)
    overrides = {"METALLIC_FULL_ROAM_HIDDEN": "1", "METALLIC_FULL_ROAM_NO_VSYNC": "1",
                 "METALLIC_REFLEX_MODE": "off", "METALLIC_SHADER_CAPTURE_SYMBOLS": "0",
                 "METALLIC_NSIGHT_GRAPHICS_CAPTURE": "0", "METALLIC_NSIGHT_SHADER_DEBUG": "0",
                 "METALLIC_RENDER_SCHEDULING_DIAGNOSTICS": "1" if diagnostics else "0",
                 "METALLIC_DEBUG_CONTROL": "1" if validation else "0",
                 "METALLIC_DEBUG_VALIDATION": "1" if validation else "0", "METALLIC_SHADER_TRACE": "0"}
    environment.update(overrides)
    return environment, {"removedNames": sorted(removed), "overrides": overrides}


def validation_messages(console: str) -> dict:
    patterns = ("Vulkan validation:", "SYNC-HAZARD", "VUID-")
    lines = console.splitlines()
    matches = [line for line in lines if any(pattern in line for pattern in patterns)]
    return {"scanned": True, "matchedLineCount": len(matches),
            "patternLineCounts": {pattern: sum(pattern in line for line in lines) for pattern in patterns},
            "firstMessage": matches[0] if matches else None}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default=[1, 2, 4, 8])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--prewarm", type=int, default=1, help="Excluded fresh processes per candidate/worker before measured blocks")
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--diagnostics", choices=["off", "on"], default="off")
    parser.add_argument("--purpose", choices=["performance", "validation"], default="performance")
    parser.add_argument("--output-env", default="METALLIC_CPU_JOB_BENCH_OUTPUT")
    parser.add_argument("--config-env", default="METALLIC_CPU_JOB_BENCH_CONFIG")
    parser.add_argument("--asset", action="append", type=Path, default=[], help="Repeat for scene/graph/content input identity")
    parser.add_argument("--app-arg", action="append", default=[], help="Additional app argument; use --app-arg=--flag")
    args = parser.parse_args()
    if args.blocks < 1 or args.prewarm < 0 or args.sample_ms < 10 or args.timeout <= 0:
        parser.error("Blocks/timeout must be positive, prewarm nonnegative and sample-ms at least 10")
    for name in ("shared", "legacy", "config"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.asset = [path.resolve(strict=True) for path in args.asset]
    args.output = args.output.resolve()
    return args


def block_comparisons(processes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for process in processes:
        if process["phase"] == "measured":
            groups[(process["workers"], process["block"])].append(process)
    comparisons = []
    for (workers, block), rows in sorted(groups.items()):
        arms = {candidate: [row for row in rows if row["candidate"] == candidate]
                for candidate in ("legacy", "shared")}
        if any(len(values) != 2 for values in arms.values()):
            raise ValueError(f"Incomplete ABBA block workers={workers} block={block}")
        item = {"workers": workers, "block": block,
                "legacyProcessIds": [row["runId"] for row in arms["legacy"]],
                "sharedProcessIds": [row["runId"] for row in arms["shared"]]}
        values = {candidate: statistics.mean(row["frameMs"]["p50"] for row in arm)
                  for candidate, arm in arms.items()}
        item["frameMs"] = {"legacyMeanP50": values["legacy"], "sharedMeanP50": values["shared"],
                           "legacyOverSharedRatio": values["legacy"] / values["shared"]}
        for metric in ("mean", "p95", "p99"):
            if all(metric in row["frameMs"] for row in rows):
                by_arm = {candidate: statistics.mean(row["frameMs"][metric] for row in arm)
                          for candidate, arm in arms.items()}
                item["frameMs"][metric] = {
                    "legacyMean": by_arm["legacy"], "sharedMean": by_arm["shared"],
                    "legacyOverSharedRatio": by_arm["legacy"] / by_arm["shared"],
                }
        comparisons.append(item)
    return comparisons


def run_process(args: argparse.Namespace, repository: Path, base_config: dict, candidate: str,
                workers: int, phase: str, block: int, ordinal: int, run_id: int) -> dict:
    directory = args.output / f"run-{run_id:03}-{candidate}-w{workers}-{phase}"
    directory.mkdir()
    config = dict(base_config, workerCount=workers, diagnostics=args.diagnostics == "on",
                  validation=args.purpose == "validation", expectedVsync=False,
                  expectedReflexMode=0, expectedReflexFrameLimitUs=0)
    config_path = directory / "Config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [str(args.shared if candidate == "shared" else args.legacy), "--skip-shader-warmup", *args.app_arg]
    environment, environment_record = benchmark_environment(args.diagnostics == "on", args.purpose == "validation")
    environment[args.output_env] = str(directory)
    environment[args.config_env] = str(config_path)
    invocation = {"runId": run_id, "candidate": candidate, "workers": workers, "phase": phase,
                  "block": block, "ordinal": ordinal, "command": command, "configSha256": sha256(config_path),
                  "environment": environment_record,
                  "environmentOverrides": {args.output_env: str(directory), args.config_env: str(config_path),
                                           **environment_record["overrides"]}}
    (directory / "Invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    print(f"run={run_id} phase={phase} block={block} ordinal={ordinal} workers={workers} candidate={candidate}", flush=True)
    exit_code, process_resources = monitor_command(command, cwd=repository, environment=environment,
        console_path=directory / "Console.log", resources_path=directory / "OSResources.jsonl",
        timeout_seconds=args.timeout, sample_ms=args.sample_ms)
    console_text = (directory / "Console.log").read_text(encoding="utf-8", errors="replace")
    validation_log = validation_messages(console_text) if args.purpose == "validation" else {"scanned": False}
    if exit_code or (validation_log.get("matchedLineCount") or 0):
        failure = {"schema": 1, "status": "failed", **{key: invocation[key] for key in
                   ("runId", "candidate", "workers", "phase", "block", "ordinal")},
                   "purpose": args.purpose, "exitCode": exit_code, "processResources": process_resources,
                   "validationMessages": validation_log}
        (directory / "ProcessSummary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        if validation_log.get("matchedLineCount"):
            raise ValueError(f"Validation warning/error requires investigation; see {directory / 'ProcessSummary.json'}")
    if exit_code:
        raise RuntimeError(f"Production benchmark exited {exit_code}; see {directory / 'Console.log'}")
    capture_path = directory / "Capture.json"
    frames_path = directory / "Frames.jsonl"
    capture = json.loads(capture_path.read_text(encoding="utf-8-sig"))
    if capture.get("status") not in SUCCESS:
        raise ValueError(f"Application capture failed: {capture.get('status')}: {capture.get('error')}")
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    for frame in frames:
        reflex = frame.get("streamlineBegin") or {}
        for key in ("effectiveMode", "frameLimitUs"):
            value = reflex.get(key)
            if value is not None and value != 0:
                raise ValueError(f"Per-frame Reflex {key} was not zero/off")
            if value is None and reflex.get("active"):
                raise ValueError(f"Active per-frame Reflex omitted {key}")
    modes = re.findall(r"\[Swapchain\].*?presentMode=(\d+)", console_text)
    capture["presentModeFromConsole"] = int(modes[-1]) if modes else None
    summary = summarize_frames(frames, capture)
    identity = effective_identity(capture, workers, args.purpose, args.diagnostics == "on", strict_conditions=True)
    if capture.get("frames") is not None and len(frames) != capture["frames"]:
        raise ValueError("Application frame count differs from raw frame records")
    record = {"schema": 1, "status": "passed", **{key: invocation[key] for key in
              ("runId", "candidate", "workers", "phase", "block", "ordinal")}, **summary,
              "purpose": args.purpose, "diagnostics": args.diagnostics == "on",
              "validationMessages": validation_log,
              "processResources": process_resources,
              "captureIdentity": {key: capture.get(key) for key in
                  ("protocol", "config", "scene", "camera", "graph", "outputExtent", "renderExtent",
                   "workerCount", "recordingWorkerLimit", "validationRequested", "loadingSeconds")},
              "effectiveIdentity": identity,
              "artifacts": {"capture": str(capture_path.relative_to(args.output)),
                            "frames": str(frames_path.relative_to(args.output)),
                            "osResources": str((directory / "OSResources.jsonl").relative_to(args.output)),
                            "console": str((directory / "Console.log").relative_to(args.output))}}
    for key, name in (("taskGraphs", "TaskGraphs.jsonl"), ("image", "FinalColor.bin")):
        if (directory / name).is_file():
            record["artifacts"][key] = str((directory / name).relative_to(args.output))
    if capture.get("image") is not None:
        record["image"] = capture["image"]
    record["artifactSha256"] = {key: sha256(args.output / path) for key, path in record["artifacts"].items()}
    (directory / "ProcessSummary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def run(args: argparse.Namespace) -> dict:
    repository = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to mix/overwrite production evidence: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a JSON object")
    sources = ["Tools/CPUJobs/MeasureProductionCPUJobs.py", "Tools/CPUJobs/ProcessResources.py",
               "Tools/CPUJobs/LegacyMeshletSceneAdapter.cpp", "Source/Runtime/Scene/MeshletBuildParallel.cpp",
               "Source/Runtime/Task/TaskSystem.cpp", "Source/Runtime/Task/ParallelFor.cpp",
               "Source/Editor/EditorCPUJobBenchmark.cpp", "Source/Editor/EditorApplication.cpp",
               "Source/Editor/EditorApplication.h", "Source/CMakeLists.txt", "Tools/CPUJobs/CMakeLists.txt",
               "Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp",
               "Source/Runtime/Render/Profiling/SchedulingDiagnostics.h", "Source/Runtime/Render/GAPI/RHI.h",
               "Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp"]
    metadata = {
        "schema": 1, "protocol": "metallic-production-cpu-jobs-abba-v1", "status": "running",
        "startedUtc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "logicalProcessors": os.cpu_count(), "repositoryRevision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "executables": {candidate: {"path": str(path), "sha256": sha256(path)} for candidate, path in
                        (("legacy", args.legacy), ("shared", args.shared))},
        "sourceSha256": {name: sha256(repository / name) for name in sources if (repository / name).exists()},
        "assetSha256": {str(path): sha256(path) for path in args.asset},
        "config": config, "configSha256": sha256(args.config),
        "workers": args.workers, "blocks": args.blocks, "prewarmProcessesPerCandidateWorker": args.prewarm,
        "diagnostics": args.diagnostics, "purpose": args.purpose, "sampleMilliseconds": args.sample_ms,
        "conditions": {
            "order": "Fresh-process ABBA blocks, reversed to BAAB on alternate blocks; worker order rotates between blocks. No outcome-dependent extra sampling.",
            "cache": "Shared application/OS/driver caches are retained. Excluded prewarm processes run both candidates before measured blocks. No claim of cold-cache startup.",
            "frameWarmup": "Application config controls discarded warmup frames; measured Frames.jsonl must exclude them.",
            "resources": "OS samples cover launch to exit, including scene load/warmup/render/export/teardown. Memory/process CPU are not GPU metrics.",
            "scopes": "Nested or parallel CPU/GPU scopes are reported independently; no inferred summed frame critical path.",
            "gpuMemory": "Only app RHI heap/domain telemetry; missing stays null. Heap allocation/block/usage are distinct quantities and are not stacked.",
            "statistics": "Nearest-rank within-process frame quantiles; independent process/block is the A/B unit, not individual frames.",
            "externalActivity": "Caller must reserve the machine/GPU; script does not detect other agent builds or desktop load.",
        },
    }
    metadata_path = args.output / "Metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    processes = []
    identities = {}
    try:
        with (args.output / "Raw.jsonl").open("w", encoding="utf-8", newline="\n") as raw:
            def invoke(candidate, workers, phase, block, ordinal):
                record = run_process(args, repository, config, candidate, workers, phase, block, ordinal, len(processes))
                previous = identities.setdefault(workers, record["effectiveIdentity"])
                if previous != record["effectiveIdentity"]:
                    raise ValueError(f"Effective scene/camera/graph/quality identity changed for workers={workers}")
                processes.append(record)
                raw.write(json.dumps(record, separators=(",", ":")) + "\n")
                raw.flush()
            for workers in args.workers:
                for repeat in range(args.prewarm):
                    for ordinal, candidate in enumerate(("legacy", "shared")):
                        invoke(candidate, workers, "prewarm", -1 - repeat, ordinal)
            for block in range(args.blocks):
                worker_order = args.workers[block % len(args.workers):] + args.workers[:block % len(args.workers)]
                for workers in worker_order:
                    order = ["legacy", "shared", "shared", "legacy"]
                    if block % 2:
                        order = ["shared", "legacy", "legacy", "shared"]
                    for ordinal, candidate in enumerate(order):
                        invoke(candidate, workers, "measured", block, ordinal)
        summary = {"schema": 1, "status": "complete", "purpose": args.purpose, "processes": processes,
                   "comparisons": block_comparisons(processes) if args.purpose == "performance" else [],
                   "definitions": metadata["conditions"]}
        (args.output / "Summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        metadata["status"] = "complete"
        metadata["completedUtc"] = datetime.now(timezone.utc).isoformat()
        metadata["processCount"] = len(processes)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        result = run(arguments())
        print(json.dumps({"processes": len(result["processes"]), "blocks": len(result["comparisons"])}))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Production CPU-job measurement failed: {error}", file=sys.stderr)
        sys.exit(1)
