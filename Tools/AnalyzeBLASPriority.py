"""Analyze archived owner-priority A/B captures; current runtime has one production policy.

Historical manifest fields remain required. This tool does not enable retired runtime paths.
Quality captures are separate evidence.
"""
import argparse
import csv
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from AnalyzeBLASAllocator import FIELDS as ALLOCATOR_FIELDS, file_identity, read_json
from AnalyzeZorahFullRoam import diagnostic_run, distribution


PRIORITY_COUNTS = (
    "blasRequestedVisibleConsumers", "blasRequestedOffscreenConsumers",
    "blasRejectedVisibleConsumers", "blasRejectedOffscreenConsumers",
    "blasAdmittedOffscreenConsumers", "blasPriorityEvictedOwners",
)
PRIORITY_ERRORS = ("blasRejectedMaxErrorPixels", "blasRejectedVisibleMaxErrorPixels")
ALLOCATION_DIAGNOSTICS = (
    "blasCompactionTargetBytes", "blasWaitingBuildCount", "blasPostBuildBudgetRejected",
    "blasPhysicalAllocationFailed", "blasWaitingClaimBytes", "blasMaxWaitingBytes", "blasUnknownBuildCount",
    "blasProjectedRetryDeferredCount", "blasProjectedDirtyClaimBytes",
    "blasPrioritySortReused", "blasPriorityAdmissionReused",
)
PRIORITY_ONLY_DIAGNOSTICS = {
    "blasPostBuildBudgetRejected", "blasPhysicalAllocationFailed", "blasUnknownBuildCount",
    "blasProjectedRetryDeferredCount", "blasProjectedDirtyClaimBytes",
}
PRIORITY_POLICY_DIAGNOSTICS = {"blasWaitingBuildCount", "blasWaitingClaimBytes", "blasMaxWaitingBytes"}
FIELDS = (*ALLOCATOR_FIELDS, *PRIORITY_COUNTS, *PRIORITY_ERRORS)
MIB = 1024 * 1024
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
LEGACY_STORAGE_SEMANTICS = "legacy-including-projected-deferrals"
CURRENT_STORAGE_SEMANTICS = "physical-and-known-storage-only-v2"
STORAGE_SEMANTICS_NOTE = (
    "blasStorageRejected semantics are part of capture identity: a missing version field means legacy, "
    "including projected retry deferrals when that policy is active; physical-and-known-storage-only-v2 "
    "excludes those deferrals. Raw counters are preserved and different semantics are never combined."
)
PROXY_DEFINITION = (
    "Priority counters describe final desired-owner selection, not actual TLAS fallback: "
    "a rejected desired owner may be rescued by a merged representation. Visible rejection/error "
    "is a foreground-degradation proxy, not measured shadow, reflection or image error."
)


def validate_conditions(manifest, capture, quality=False, comparison="priority"):
    storage_semantics = capture.get("blasStorageRejectedSemantics", LEGACY_STORAGE_SEMANTICS)
    if storage_semantics not in (LEGACY_STORAGE_SEMANTICS, CURRENT_STORAGE_SEMANTICS):
        raise ValueError("Unknown blasStorageRejectedSemantics: " + str(storage_semantics))
    if manifest.get("blasAllocator") != "gpu" or manifest.get("blasRepresentations") != "1":
        raise ValueError("Priority A/B requires explicit GPU allocator and representations=1 in both variants")
    if manifest.get("blasPriority") not in ("0", "1"):
        raise ValueError("Archived A/B manifest must record blasPriority=0 or 1")
    if comparison == "cache" and (manifest["blasPriority"] != "1" or manifest.get("blasPriorityCache") not in ("0", "1")):
        raise ValueError("Archived cache comparison requires blasPriority=1 and blasPriorityCache=0/1")
    if comparison == "merged-sharing" and (manifest["blasPriority"] != "1" or manifest.get("blasPriorityCache") != "1" or
                                          manifest.get("blasMergedSharing") not in ("0", "1")):
        raise ValueError("Archived merged-sharing comparison requires priority/cache=1 and blasMergedSharing=0/1")
    if comparison == "intermediate-fallback" and (manifest["blasPriority"] != "1" or
            manifest.get("blasPriorityCache") != "1" or manifest.get("blasMergedSharing") != "1" or
            manifest.get("blasFallbackPolicy") not in ("0", "1") or
            manifest.get("blasMixedTileFastPath") != manifest.get("blasFallbackPolicy")):
        raise ValueError("Intermediate fallback comparison requires priority/cache/merged=1 and explicit fallback policy=0/1 with matching mixed-tile fast path")
    if not manifest.get("blasStorageBytes") or int(manifest["blasStorageBytes"]) <= 0:
        raise ValueError("Priority A/B requires an explicit shared storage budget")
    if manifest.get("blasRelocateBytes") is None or int(manifest["blasRelocateBytes"]) < 0:
        raise ValueError("Priority A/B requires an explicit shared relocation budget")
    if capture.get("status") != "capture_complete" or not capture.get("frames") or capture.get("missingGpuFrames") != 0:
        raise ValueError("Incomplete capture or GPU timing coverage")
    if quality and not capture.get("captureQuality"):
        raise ValueError("Quality comparison requires dedicated captureQuality runs")
    if not quality and (diagnostic_run(capture) or capture.get("validationRequested") or capture.get("captureQuality")):
        raise ValueError("Validation, workload and quality captures are separate from performance acceptance")
    if capture.get("trajectoryClock") != "deterministic frame index" or capture.get("streamingFreeze", {}).get("enabled"):
        raise ValueError("Priority pressure replay requires a live deterministic route")
    conditions = {
        "manifest": {key: manifest.get(key) for key in (
            "executableSha256", "shaderSha256", "asset", "gpu", "config", "validation", "vsync", "hidden",
            "blasAllocator", "blasRepresentations", "blasInstanceReuse", "clasAllocator", "blasStorageBytes",
            "blasRelocateBytes", "blasPriorityCache", "blasMergedSharing", "blasFallbackPolicy", "blasMixedTileFastPath", "reflexMode", "streamMaintenanceBeforePacing")},
        "capture": {**{key: capture.get(key) for key in (
            "outputExtent", "renderExtent", "config", "absoluteKeyframes", "graph", "hidden", "vsync",
            "shaderDebugMode", "graphicsCaptureInjected", "frameTimeDefinition", "gpuTimeDefinition",
            "captureQuality", "captureReflections", "reflectionDiagnostic")},
            "blasStorageRejectedSemantics": storage_semantics},
    }
    if comparison == "cache":
        del conditions["manifest"]["blasPriorityCache"]
        conditions["manifest"]["blasPriority"] = manifest["blasPriority"]
    if comparison == "merged-sharing":
        del conditions["manifest"]["blasMergedSharing"]
        conditions["manifest"]["blasPriority"] = manifest["blasPriority"]
    if comparison == "intermediate-fallback":
        del conditions["manifest"]["blasFallbackPolicy"]
        del conditions["manifest"]["blasMixedTileFastPath"]
        conditions["manifest"]["blasPriority"] = manifest["blasPriority"]
    return conditions


def finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid nonnegative measurement: {name}")
    return value


def device_peak(path):
    if not path.exists():
        return None
    values = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = next((key for key in row if key and key.strip().startswith("memory.used")), None)
            if key:
                try:
                    values.append(float(row[key].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
    return max(values) if values else None


def competition_evidence(directory):
    csv_path, error_path = directory / "GpuProcesses.csv", directory / "GpuProcesses.stderr.txt"
    error_bytes = error_path.read_bytes() if error_path.exists() else b""
    error_encoding = "utf-8-sig"
    try:
        error_text = error_bytes.decode(error_encoding).strip()
    except UnicodeDecodeError:
        error_encoding = "gb18030"
        error_text = error_bytes.decode(error_encoding, errors="replace").strip()
    records = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="", errors="replace") as source:
            records = [row for row in csv.DictReader(source) if row.get("timestamp")]
    status = "activity-with-errors" if error_text and records else "error-reported" if error_text else "missing" if not csv_path.exists() else "activity-recorded" if records else "no-samples"
    processes = {}
    for row in records:
        key = (row.get("process", ""), row.get("pid", ""))
        entry = processes.setdefault(key, {"process": key[0], "pid": key[1], "activityRows": 0,
            "timestamps": set(), "engineTypes": set(), "maxReportedEngineUtilizationPercent": 0.0})
        entry["activityRows"] += 1
        entry["timestamps"].add(row["timestamp"])
        entry["engineTypes"].add(row.get("engine", "").rsplit("engtype_", 1)[-1])
        entry["maxReportedEngineUtilizationPercent"] = max(entry["maxReportedEngineUtilizationPercent"], float(row["utilization"]))
    summaries = []
    for entry in processes.values():
        timestamps = sorted(entry.pop("timestamps"))
        summaries.append({**entry, "engineTypes": sorted(entry["engineTypes"]), "activeTimestamps": len(timestamps),
            "firstActiveTimestamp": timestamps[0], "lastActiveTimestamp": timestamps[-1]})
    summaries.sort(key=lambda entry: (-entry["activityRows"], entry["process"], entry["pid"]))
    return {"status": status, "sampleRows": len(records), "stderrBytes": len(error_bytes),
        "stderrEncoding": error_encoding,
        "stderrExcerpt": error_text[-4096:] if error_text else None,
        "processNames": sorted({row["process"] for row in records if row.get("process")}),
        "processActivity": summaries,
        "establishesNoCompetition": False,
        "scope": "Get-Counter logs valid per-counter process activity >=0.1%, without heartbeat coverage, across launch/loading/warmup/route. Peaks are individual engine-counter samples, not summed GPU occupancy. Stderr can coexist with valid rows when other counter samples fail. Failure, missing files or an empty/header-only CSV do not demonstrate absence of competition. Whole-device Gpu.csv cannot identify competing processes.",
        "inputs": [file_identity(path) for path in (csv_path, error_path) if path.exists()]}


def analyze_run(directory, discard_first, comparison="priority"):
    paths = [directory.parent / "Manifest.json", directory / "Capture.json", directory / "Frames.jsonl"]
    manifest, capture = map(read_json, paths[:2])
    conditions = validate_conditions(manifest, capture, comparison=comparison)
    all_rows = [json.loads(line) for line in paths[2].read_text(encoding="utf-8-sig").splitlines() if line]
    if len(all_rows) != capture["frames"] or len(all_rows) != capture["config"].get("routeFrames"):
        raise ValueError("Capture does not contain the full deterministic route")
    if len({row["frame"] for row in all_rows}) != len(all_rows):
        raise ValueError("Repeated editor frame identity")
    if [row["routeSample"] for row in all_rows] != list(range(len(all_rows))):
        raise ValueError("Missing, repeated or reordered deterministic route sample")
    if discard_first >= len(all_rows):
        raise ValueError("Discard interval removes the entire route")
    rows = all_rows[discard_first:]
    chain = [entry for entry in capture["scopes"] if not entry.get("cpuOnly") and
             entry["path"].rsplit("/", 1)[-1].startswith(("BLAS ", "RT representation "))]
    names = {entry["id"]: entry["path"].rsplit("/", 1)[-1] for entry in chain}
    if not {"RT representation classify", "RT representation cut", "BLAS build"}.issubset(names.values()):
        raise ValueError("Full RT preparation chain lacks classification, cut or BLAS build")
    if len({entry["path"].rsplit("/", 1)[0] for entry in chain}) != 1 or len({entry["queue"] for entry in chain}) != 1:
        raise ValueError("RT preparation stages must be siblings on one queue")
    chain_ids = set(names)
    envelope_ids = {entry["id"] for entry in capture["scopes"] if entry["path"].endswith("/RenderGraph GPU envelope")}
    reflex_ids = {entry["id"] for entry in capture["scopes"] if entry["path"] == "Frame/Idle / Reflex Frame Pacing"}
    timings, stream_rows, route = [], [], {}
    expected_priority = manifest["blasPriority"] == "1"
    expected_cache = manifest.get("blasPriorityCache") == "1"
    expected_merged_sharing = manifest.get("blasMergedSharing") == "1"
    for row in rows:
        if row.get("diagnostic"):
            raise ValueError("Diagnostic frame in performance route")
        frame_ms = finite(row["frameMs"], "frameMs")
        if frame_ms <= 0:
            raise ValueError("Frame time must be positive")
        stages = [scope for scope in row["scopes"] if scope["id"] in chain_ids]
        if len(stages) != len(chain_ids) or {scope["id"] for scope in stages} != chain_ids:
            raise ValueError("Missing or repeated RT preparation sibling stage")
        executions = {scope.get("executionId") for scope in stages}
        if None in executions or len(executions) != 1:
            raise ValueError("RT stages do not share one GPU execution")
        stage_gpu = {names[scope["id"]]: finite(scope.get("gpuMs"), names[scope["id"]]) for scope in stages}
        stage_cpu = [finite(scope.get("cpuMs"), "RT stage CPU") for scope in stages]
        envelope = [scope for scope in row["scopes"] if scope["id"] in envelope_ids]
        if len(envelope) != 1 or envelope[0].get("executionId") not in executions:
            raise ValueError("Missing, duplicate or mismatched RenderGraph GPU envelope")
        reflex = [scope for scope in row["scopes"] if scope["id"] in reflex_ids]
        if len(reflex) > 1:
            raise ValueError("Repeated Reflex pacing scope")
        timings.append({"frame": row["frame"], "routeSample": row["routeSample"], "stage": row["stage"],
            "executionId": next(iter(executions)), "frameMs": frame_ms,
            "rtPreparationGpuMs": sum(stage_gpu.values()), "rtPreparationCpuMs": sum(stage_cpu),
            "blasBuildGpuMs": stage_gpu["BLAS build"],
            "graphGpuEnvelopeMs": finite(envelope[0].get("gpuMs"), "GPU envelope"),
            "reflexPacingCpuMs": finite(reflex[0].get("cpuMs"), "Reflex pacing") if reflex else None,
            "stageGpuMs": stage_gpu})
        streams = [sample for sample in row["streaming"] if sample.get("pass") == "VBuffer"]
        if len(streams) != 1:
            raise ValueError("Expected one VBuffer streaming sample per frame")
        stream = streams[0]
        if stream.get("blasGPUAllocatorEnabled") is not True or stream.get("blasRepresentationsEnabled") is not True or \
                stream.get("blasPriorityEnabled") is not expected_priority:
            raise ValueError("Requested priority/allocator/representations differ from effective runtime settings")
        if comparison in ("cache", "merged-sharing", "intermediate-fallback") and stream.get("blasPriorityCacheEnabled") is not expected_cache:
            raise ValueError("Requested priority cache differs from effective runtime settings")
        if comparison in ("merged-sharing", "intermediate-fallback") and stream.get("blasMergedSharingEnabled") is not expected_merged_sharing:
            raise ValueError("Requested merged sharing differs from effective runtime settings")
        if comparison == "intermediate-fallback" and stream.get("blasFallbackPolicy") != int(manifest["blasFallbackPolicy"]):
            raise ValueError("Requested intermediate fallback differs from effective runtime settings")
        if comparison == "intermediate-fallback" and stream.get("blasMixedTileFastPathEnabled") is not (manifest["blasMixedTileFastPath"] == "1"):
            raise ValueError("Requested mixed-tile fast path differs from effective runtime settings")
        if stream["frame"] in route:
            raise ValueError("Repeated runtime frame identity")
        route[stream["frame"]] = row
        stream_rows.append(stream)
    observed, duplicates, outside = {}, 0, 0
    for stream in stream_rows:
        if not stream.get("blasFeedbackAvailable"):
            continue
        frame = stream["blasFeedbackFrame"]
        if frame not in route:
            outside += 1
            continue
        if frame in observed:
            duplicates += 1
            continue
        values = {key: finite(stream[key], key) for key in FIELDS}
        for key in ALLOCATION_DIAGNOSTICS:
            if key in stream:
                values[key] = finite(stream[key], key)
                if not isinstance(values[key], int):
                    raise ValueError(f"GPU integer diagnostic is not an integer: {key}")
        if any(not isinstance(values[key], int) for key in (*ALLOCATOR_FIELDS, *PRIORITY_COUNTS)):
            raise ValueError("GPU integer counter is not an integer")
        if values["blasAllocatedBytes"] < values["blasEncodedBytes"] or values["blasLargestFreeBytes"] > values["blasFreeBytes"] or \
                values["blasAllocatedBytes"] + values["blasFreeBytes"] != values["blasPersistentBytes"]:
            raise ValueError("Invalid GPU allocator storage accounting")
        if values["blasArenaRepack"] or values["blasRelocatedBytes"] > int(manifest["blasRelocateBytes"]):
            raise ValueError("GPU allocator repacked or exceeded its relocation budget")
        if values["blasRejectedVisibleConsumers"] > values["blasRequestedVisibleConsumers"] or \
                values["blasRejectedOffscreenConsumers"] + values["blasAdmittedOffscreenConsumers"] != values["blasRequestedOffscreenConsumers"]:
            raise ValueError("Priority consumer counts violate requested/admitted/rejected accounting")
        values.update({"frame": frame, "routeSample": route[frame]["routeSample"], "stage": route[frame]["stage"],
            "feedbackLag": finite(stream["frame"] - frame, "feedback lag"),
            "blasBufferBytes": sum(values[key] for key in ("blasPersistentBytes", "blasStagingBytes", "blasRelocationStagingBytes", "blasMoveScratchBytes")),
            "internalWasteBytes": values["blasAllocatedBytes"] - values["blasEncodedBytes"],
            "fragmentedFreeBytes": values["blasFreeBytes"] - values["blasLargestFreeBytes"],
            "cleanMoveTrafficBytes": 2 * values["blasRelocatedBytes"]})
        observed[frame] = values
    samples = sorted(observed.values(), key=lambda value: value["routeSample"])
    if len(samples) < .9 * len(rows):
        raise ValueError("Less than 90% unique route-aligned GPU feedback; repeat the capture")
    metric_keys = (*FIELDS, "blasBufferBytes", "internalWasteBytes", "fragmentedFreeBytes", "cleanMoveTrafficBytes", "feedbackLag")
    timing_keys = ("frameMs", "rtPreparationGpuMs", "rtPreparationCpuMs", "blasBuildGpuMs", "graphGpuEnvelopeMs", "reflexPacingCpuMs")
    settled = [sample for sample in samples if sample["stage"] == "final-static"]
    result = {"run": directory.parent.name + "/" + directory.name, "priority": expected_priority,
        "comparison": comparison, "comparisonEnabled": manifest.get("blasFallbackPolicy") == "1" if comparison == "intermediate-fallback" else expected_merged_sharing if comparison == "merged-sharing" else
            expected_cache if comparison == "cache" else expected_priority,
        "started": manifest.get("started"), "recordedFrames": len(all_rows), "measuredFrames": len(rows),
        "discardedFirstFrames": discard_first, "feedbackSamples": len(samples), "missingRouteFeedback": len(rows) - len(samples),
        "duplicateFeedback": duplicates, "outsideRouteFeedback": outside,
        "metrics": {key: distribution([sample[key] for sample in samples]) for key in metric_keys},
        "allocationDiagnostics": {key: {
            "status": "missing" if not any(key in sample for sample in samples) else
                      "not instrumented" if not expected_priority and key in PRIORITY_ONLY_DIAGNOSTICS else "reported",
            "availableSamples": sum(key in sample for sample in samples) if expected_priority or key not in PRIORITY_ONLY_DIAGNOSTICS else 0,
            "notInstrumentedSamples": sum(key in sample for sample in samples) if not expected_priority and key in PRIORITY_ONLY_DIAGNOSTICS else 0,
            "missingSamples": sum(key not in sample for sample in samples),
            "statistics": distribution([sample[key] for sample in samples if key in sample]) if expected_priority or key not in PRIORITY_ONLY_DIAGNOSTICS else None,
            "scope": "Only the priority-on allocator branch is instrumented. Baseline raw zeros are retained but are not evidence of zero failures and are excluded from statistics." if key in PRIORITY_ONLY_DIAGNOSTICS else
                     "Priority-policy waiting state; priority-off zeros describe an inactive policy, not equivalent baseline failure instrumentation." if key in PRIORITY_POLICY_DIAGNOSTICS else
                     "Exact ordering/admission cache reuse, reported for the completed BLAS producer; cache-off is zero." if key in ("blasPrioritySortReused", "blasPriorityAdmissionReused") else
                     "Compaction target is instrumented on all GPU allocator paths.",
        } for key in ALLOCATION_DIAGNOSTICS},
        "screenErrorBounds": {key: {
            "unboundedOrSaturatedSamples": sum(sample[key] >= FLOAT32_MAX for sample in samples),
            "sampleCount": len(samples),
            "finiteBoundsPixels": distribution([sample[key] for sample in samples if sample[key] < FLOAT32_MAX]),
            "scope": "FLT_MAX is an unbounded or saturated conservative screen-error bound, not a measured pixel error. The producer can reach it for near-plane or inside-sphere projection. Raw metrics are retained; finite summaries exclude FLT_MAX."
        } for key in PRIORITY_ERRORS},
        "timings": {key: distribution([sample[key] for sample in timings if sample[key] is not None]) for key in timing_keys},
        "stageGpuMs": {name: distribution([sample["stageGpuMs"][name] for sample in timings]) for name in names.values()},
        "routeStageTimings": {stage: {key: distribution([sample[key] for sample in timings if sample["stage"] == stage and sample[key] is not None])
            for key in timing_keys} for stage in dict.fromkeys(sample["stage"] for sample in timings)},
        "chainScopeDefinitions": chain, "frameTimings": timings,
        "deviceSampledPeakMiB": device_peak(directory / "Gpu.csv"),
        "gpuCompetition": competition_evidence(directory),
        "finalStatic": {"samples": len(settled), "last32Builds": [sample["blasBuildCount"] for sample in settled[-32:]],
            "last32Evictions": [sample["blasPriorityEvictedOwners"] for sample in settled[-32:]],
            "last32SortReused": [sample.get("blasPrioritySortReused") for sample in settled[-32:]],
            "last32AdmissionReused": [sample.get("blasPriorityAdmissionReused") for sample in settled[-32:]]},
        "inputs": [file_identity(path) for path in paths] + ([file_identity(directory / "Gpu.csv")] if (directory / "Gpu.csv").exists() else [])}
    return conditions, result, samples


def plot(output, results, observations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for run_index, result in enumerate(results):
        label = f"{result['run']} {result.get('comparison', 'priority')}={'on' if result.get('comparisonEnabled', result['priority']) else 'off'}"
        rows = result["frameTimings"]
        for axis, key, title in ((axes[0, 0], "rtPreparationGpuMs", "Full RT preparation GPU ms"),
                                 (axes[0, 1], "graphGpuEnvelopeMs", "RenderGraph GPU envelope ms")):
            axis.plot([row["routeSample"] for row in rows], [row[key] for row in rows], label=label, linewidth=.7)
            axis.set_title(title)
        for axis, key, title in ((axes[1, 0], "blasRejectedVisibleConsumers", "Rejected visible consumers (proxy)"),
                                 (axes[1, 1], "blasRejectedVisibleMaxErrorPixels", "Finite rejected visible error bounds px (proxy)"),
                                 (axes[2, 0], "blasAdmittedOffscreenConsumers", "Admitted offscreen consumers"),
                                 (axes[2, 1], "blasPriorityEvictedOwners", "Priority-evicted owners")):
            samples = observations[result["run"]]
            values = [row[key] if key not in PRIORITY_ERRORS or row[key] < FLOAT32_MAX else None for row in samples]
            axis.plot([row["routeSample"] for row in samples], values, label=label, linewidth=.7)
            if key in PRIORITY_ERRORS:
                bound = result["screenErrorBounds"][key]
                axis.text(.02, .96 - run_index * .06,
                          f"{result['run']}: {bound['unboundedOrSaturatedSamples']}/{bound['sampleCount']} unbounded/saturated (omitted)",
                          transform=axis.transAxes, va="top", fontsize=7)
            axis.set_title(title)
    for axis in axes.flat:
        axis.set_xlabel("Deterministic route sample")
        axis.set_xlim(min(row["routeSample"] for result in results for row in result["frameTimings"]),
                      max(row["routeSample"] for result in results for row in result["frameTimings"]))
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Owner-priority A/B: full scheduling cost and desired-owner selection proxies")
    figure.savefig(output / "Priority.png", dpi=140)
    plt.close(figure)


def compare_quality(args, directories):
    import numpy as np
    from AnalyzeRTRepresentation import image_arrays
    if len(directories) != 2:
        raise ValueError("Separate quality comparison requires exactly one priority-off and one priority-on run")
    records = [(directory, read_json(directory.parent / "Manifest.json"), read_json(directory / "Capture.json"))
               for directory in directories]
    records.sort(key=lambda item: item[1].get("blasPriority", ""))
    if [manifest.get("blasPriority") for _, manifest, _ in records] != ["0", "1"]:
        raise ValueError("Quality comparison requires explicit off/on variants")
    conditions = [validate_conditions(manifest, capture, quality=True) for _, manifest, capture in records]
    if conditions[0] != conditions[1]:
        raise ValueError("Quality variants have different workloads, budgets, binary, shader or image settings")
    reference_conditions = None
    if args.quality_reference:
        reference_root = args.quality_reference.resolve()
        reference_runs = [reference_root] if (reference_root / "Capture.json").exists() else \
            [path.parent for path in reference_root.glob("run*/Capture.json")]
        if len(reference_runs) != 1:
            raise ValueError("--quality-reference must identify one run directory or one capture root containing run1")
        reference_run = reference_runs[0]
        reference_manifest = read_json(reference_run.parent / "Manifest.json")
        reference_capture = read_json(reference_run / "Capture.json")
        reference_conditions = validate_conditions(reference_manifest, reference_capture, quality=True)
        if reference_manifest["blasPriority"] != "0":
            raise ValueError("The high-budget quality reference must use priority off")
        reference_budget = int(reference_manifest["blasStorageBytes"])
        if reference_budget < int(records[0][1]["blasStorageBytes"]):
            raise ValueError("Reference storage budget is smaller than the compared pair")
        matched_reference = {"manifest": dict(reference_conditions["manifest"]), "capture": reference_conditions["capture"]}
        matched_reference["manifest"]["blasStorageBytes"] = conditions[0]["manifest"]["blasStorageBytes"]
        if matched_reference != conditions[0]:
            raise ValueError("Only storage budget may differ between the quality pair and its reference")
        records.append((reference_run, reference_manifest, reference_capture))
    points, inputs = [], []
    for directory, manifest, capture in records:
        frames_path = directory / "Frames.jsonl"
        rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8-sig").splitlines() if line]
        if len(rows) != capture["frames"] or len(rows) != capture["config"].get("routeFrames"):
            raise ValueError("Quality route coverage is incomplete")
        for row in rows:
            streams = [stream for stream in row["streaming"] if stream.get("pass") == "VBuffer"]
            if len(streams) != 1 or streams[0].get("blasPriorityEnabled") is not (manifest["blasPriority"] == "1") or \
                    streams[0].get("blasGPUAllocatorEnabled") is not True or streams[0].get("blasRepresentationsEnabled") is not True:
                raise ValueError("Quality requested policy differs from runtime")
        samples = {f"sample{sample['routeSample']}": sample for sample in capture.get("qualitySamples", [])}
        if len(samples) != len(capture.get("qualitySamples", [])):
            raise ValueError("Duplicate quality route sample")
        if {sample["routeSample"] for sample in samples.values()} != set(capture["config"].get("qualitySamples", [])):
            raise ValueError("Requested quality samples are missing")
        samples["final"] = {"camera": capture["qualityCamera"], "directory": "Quality"}
        points.append(samples)
        inputs.extend(file_identity(path) for path in (directory.parent / "Manifest.json", directory / "Capture.json", frames_path))
    if any(point.keys() != points[0].keys() for point in points[1:]):
        raise ValueError("Quality sample identities differ")
    output = (args.output or args.root / "PriorityQualityAnalysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    images, reference_errors, plots = {}, {}, []
    for sample_name in points[0]:
        if any(point[sample_name]["camera"] != points[0][sample_name]["camera"] for point in points[1:]):
            raise ValueError("Quality cameras differ at " + sample_name)
        arrays = []
        for (directory, _, _), samples in zip(records, points):
            path = (directory / samples[sample_name]["directory"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Quality directory escapes its run")
            arrays.append(image_arrays(path))
            inputs.extend(file_identity(file) for file in sorted(path.iterdir()) if file.is_file())
        if any(array.keys() != arrays[0].keys() for array in arrays[1:]):
            raise ValueError("Quality output resources differ at " + sample_name)
        required = {"Shadows.shadow", "Deferred.color", "FinalBlit.color"}
        if records[0][2].get("captureReflections"):
            required.add("ReflectionQuality.color")
        if not required.issubset(arrays[0]):
            raise ValueError("Required quality output is missing at " + sample_name)
        images[sample_name] = {}
        if reference_conditions is not None:
            reference_errors[sample_name] = {}
        for name, before in arrays[0].items():
            after = arrays[1][name]
            if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError("Invalid quality image at " + sample_name + "/" + name)
            channels = min(3, before.shape[-1])
            before, after = before[..., :channels].astype(np.float64), after[..., :channels].astype(np.float64)
            delta = np.abs(before - after)
            def display(value):
                if channels == 1:
                    return np.clip(value[..., 0], 0, 1)
                linear = np.maximum(value, 0)
                return (linear / (1 + linear)) ** (1 / 2.2)
            shown = np.abs(display(before) - display(after))
            pixel_delta = shown if channels == 1 else shown.max(axis=-1)
            images[sample_name][name] = {"rawMAE": float(delta.mean()), "rawP99": float(np.quantile(delta, .99)),
                "rawMax": float(delta.max()), "exactPixels": float(np.mean(np.all(delta == 0, axis=-1))),
                "displayMAE": float(shown.mean()), "displayP99": float(np.quantile(shown, .99)),
                "displayPixelsOverOnePercent": float(np.mean(pixel_delta > .01))}
            if reference_conditions is not None:
                reference = arrays[2][name]
                if reference.shape != arrays[0][name].shape or not np.isfinite(reference).all():
                    raise ValueError("Invalid reference image at " + sample_name + "/" + name)
                reference = reference[..., :channels].astype(np.float64)
                height, width = reference.shape[:2]
                if not height or not width:
                    raise ValueError("Empty quality reference image")
                rectangles = {"fullFrame": (0, 0, width, height),
                    "central50Percent": (width // 4, height // 4, max(width // 4 + 1, 3 * width // 4), max(height // 4 + 1, 3 * height // 4))}
                comparisons = {}
                for region, (x0, y0, x1, y1) in rectangles.items():
                    oracle = reference[y0:y1, x0:x1]
                    def errors(value):
                        difference = np.abs(value[y0:y1, x0:x1] - oracle)
                        fixed = np.abs(display(value[y0:y1, x0:x1]) - display(oracle))
                        pixel_difference = difference.max(axis=-1)
                        fixed_pixel_difference = fixed if channels == 1 else fixed.max(axis=-1)
                        return {"rawMAE": float(difference.mean()), "rawRMSE": float(np.sqrt(np.mean(difference ** 2))),
                            "rawP95": float(np.quantile(difference, .95)), "rawP99": float(np.quantile(difference, .99)),
                            "rawP999": float(np.quantile(difference, .999)), "rawMax": float(difference.max()),
                            "rawPixelsOver001": float(np.mean(pixel_difference > .01)),
                            "displayMAE": float(fixed.mean()), "displayRMSE": float(np.sqrt(np.mean(fixed ** 2))),
                            "displayP95": float(np.quantile(fixed, .95)), "displayP99": float(np.quantile(fixed, .99)),
                            "displayP999": float(np.quantile(fixed, .999)), "displayMax": float(fixed.max()),
                            "displayPixelsOverOnePercent": float(np.mean(fixed_pixel_difference > .01))}
                    baseline_error, candidate_error = errors(before), errors(after)
                    comparisons[region] = {"rectangleXYXY": [x0, y0, x1, y1], "pixels": (x1 - x0) * (y1 - y0),
                        "baselineToReference": baseline_error, "candidateToReference": candidate_error,
                        "candidateMinusBaseline": {key: candidate_error[key] - baseline_error[key] for key in baseline_error}}
                reference_errors[sample_name][name] = comparisons
                if not args.no_plot:
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
                    axes[0].imshow(display(reference), cmap="gray", vmin=0, vmax=1)
                    axes[0].set_title("Higher-budget priority-off reference")
                    raw_errors = [np.abs(before - reference).mean(axis=-1), np.abs(after - reference).mean(axis=-1)]
                    maximum = max(float(error.max()) for error in raw_errors)
                    for axis, error, label in zip(axes[1:], raw_errors, ("Off vs reference", "On vs reference")):
                        heat = axis.imshow(error, cmap="magma", vmin=0, vmax=maximum if maximum else 1)
                        axis.set_title(label)
                    figure.colorbar(heat, ax=list(axes[1:]))
                    x0, y0, x1, y1 = rectangles["central50Percent"]
                    for axis in axes:
                        axis.add_patch(Rectangle((x0 - .5, y0 - .5), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=1))
                        axis.axis("off")
                    figure.suptitle(sample_name + " / " + name + " | central rectangle is only a foreground proxy")
                    filename = sample_name + "-" + name + "-reference.png"
                    figure.savefig(output / filename, dpi=140)
                    plt.close(figure)
                    plots.append(filename)
            if not args.no_plot:
                import matplotlib.pyplot as plt
                figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
                for axis, value, title in zip(axes[:2], (before, after), ("Priority off", "Priority on")):
                    axis.imshow(display(value), cmap="gray", vmin=0, vmax=1)
                    axis.set_title(title)
                heat = axes[2].imshow(delta.mean(axis=-1), cmap="magma")
                axes[2].set_title("Raw absolute difference")
                figure.colorbar(heat, ax=axes[2])
                for axis in axes:
                    axis.axis("off")
                figure.suptitle(sample_name + " / " + name + " (same fixed display transform)")
                filename = sample_name + "-" + name + ".png"
                figure.savefig(output / filename, dpi=140)
                plt.close(figure)
                plots.append(filename)
    command = [sys.executable, "-B", str(output / Path(__file__).name), str(args.root.resolve()), "--quality", "--output", str(output)]
    if args.no_plot:
        command.append("--no-plot")
    if args.quality_reference:
        command.extend(["--quality-reference", str(args.quality_reference.resolve())])
    scripts = []
    for name in (Path(__file__).name, "AnalyzeBLASAllocator.py", "AnalyzeZorahFullRoam.py", "AnalyzeRTRepresentation.py"):
        source, destination = Path(__file__).resolve().with_name(name), output / name
        if source != destination:
            shutil.copy2(source, destination)
        scripts.append(file_identity(destination))
    scope = ("Matched diagnostic route executions plus one extra final frame. Readback/drain timings are excluded. "
        "RGB display uses fixed Reinhard and gamma 1/2.2; shadow display clamps to [0,1]. No per-image normalization. "
        "Full material path tracing includes reflections, not isolated specular radiance. Sparse snapshots are not a temporal sequence. "
        "Pairwise differences alone do not establish accuracy. Optional reference errors measure distance to a matched higher-budget priority-off render, not ground truth. "
        "The central 50% width/height rectangle is an explicit foreground proxy, without subject detection. Negative candidate-minus-baseline error means closer to this reference. "
        "Raw error fractions use an absolute 0.01 threshold in native image units; fixed-display fractions use 0.01 in [0,1]. "
        "High-budget streaming may still fall back, and without A/A controls stochastic/residency variation cannot be attributed solely to policy. "
        + STORAGE_SEMANTICS_NOTE)
    report = {"conditions": conditions[0], "referenceConditions": reference_conditions, "scope": scope,
        "images": images, "referenceErrors": reference_errors, "inputs": inputs,
        "gpuCompetition": [{"run": str(directory), **competition_evidence(directory)} for directory, _, _ in records],
        "reproducibility": {"python": sys.version, "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib")},
            "scripts": scripts, "commandArguments": command, "windowsCommand": "& " + subprocess.list2cmdline(command)}}
    (output / "Quality.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# BLAS priority quality diagnostics", "", scope, "",
        "| Sample / output | Raw MAE | Raw P99 | Raw max | Exact pixels | Display pixels differing >1% |", "|---|---:|---:|---:|---:|---:|"]
    for sample, resources in images.items():
        for name, value in resources.items():
            lines.append(f"| {sample}/{name} | {value['rawMAE']:.7g} | {value['rawP99']:.7g} | {value['rawMax']:.7g} | " +
                         f"{value['exactPixels']:.4%} | {value['displayPixelsOverOnePercent']:.4%} |")
    if reference_conditions is not None:
        lines += ["", "## Error against the higher-budget reference", "",
            "Negative deltas mean the candidate is closer to this reference. Full native/display metrics, P95/P99/P99.9 and threshold fractions are retained in Quality.json.", "",
            "| Sample / output / region | Raw MAE off/on | Raw RMSE off/on | Display MAE off/on | Delta display MAE | Delta display >1% fraction |",
            "|---|---:|---:|---:|---:|---:|"]
        for sample, resources in reference_errors.items():
            for name, regions in resources.items():
                for region, values in regions.items():
                    baseline, candidate, changes = values["baselineToReference"], values["candidateToReference"], values["candidateMinusBaseline"]
                    columns = [f"{baseline[key]:.7g}/{candidate[key]:.7g}" for key in ("rawMAE", "rawRMSE", "displayMAE")]
                    lines.append(f"| {sample}/{name}/{region} | " + " | ".join(columns) +
                        f" | {changes['displayMAE']:+.7g} | {changes['displayPixelsOverOnePercent']:+.4%} |")
    for filename in plots:
        lines += ["", f"![{filename}]({filename})"]
    lines += ["", "## Reproduce", "", "```powershell", report["reproducibility"]["windowsCommand"], "```", ""]
    (output / "Quality.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "qualitySamplesIncludingFinal": len(images)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing A1/B1/B2/A2 run captures")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discard-first", type=int, default=0)
    parser.add_argument("--exploratory", action="store_true", help="Permit incomplete ABBA blocks; report descriptive data only")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--compare-cache", action="store_true", help="Compare admission caching off/on with owner priority enabled on both sides")
    parser.add_argument("--compare-merged-sharing", action="store_true", help="Compare merged sharing off/on with owner priority and admission caching enabled on both sides")
    parser.add_argument("--compare-intermediate-fallback", action="store_true", help="Compare fallback policy 0/1 with priority/cache/merged enabled")
    parser.add_argument("--quality", action="store_true", help="Compare a separate off/on diagnostic pair and matched image samples; no performance analysis")
    parser.add_argument("--quality-reference", type=Path, help="Separate higher-budget priority-off quality run or capture root; only storage budget may differ")
    args = parser.parse_args()
    if args.discard_first < 0:
        parser.error("--discard-first must be nonnegative")
    if args.quality_reference and not args.quality:
        parser.error("--quality-reference requires --quality")
    if sum((args.compare_cache, args.compare_merged_sharing, args.compare_intermediate_fallback)) > 1:
        parser.error("Choose one performance comparison mode")
    if (args.compare_cache or args.compare_merged_sharing or args.compare_intermediate_fallback) and args.quality:
        parser.error("Policy/cache comparisons require separate matched quality diagnostics")
    comparison = "intermediate-fallback" if args.compare_intermediate_fallback else "merged-sharing" if args.compare_merged_sharing else "cache" if args.compare_cache else "priority"
    directories = sorted(path.parent for path in args.root.glob("*/run*/Capture.json"))
    if not directories:
        parser.error("No captures matching */run*/Capture.json")
    if args.quality:
        if args.discard_first or args.exploratory:
            parser.error("Quality mode does not use performance frame exclusions or exploratory ABBA options")
        compare_quality(args, directories)
        return
    results, observations, expected = [], {}, None
    for directory in directories:
        conditions, result, samples = analyze_run(directory, args.discard_first, comparison)
        if expected is not None and conditions != expected:
            raise ValueError(f"Workload, budget, binary, shader or warmup mismatch: {directory}")
        expected = conditions
        results.append(result)
        observations[result["run"]] = samples
    results.sort(key=lambda value: value["started"] or "")
    if {result["comparisonEnabled"] for result in results} != {False, True}:
        raise ValueError("Comparison requires explicit selected mode off and on captures")
    modes = [result["comparisonEnabled"] for result in results]
    abba = len(modes) % 4 == 0 and all(modes[index:index + 4] == [False, True, True, False] for index in range(0, len(modes), 4))
    if not abba and not args.exploratory:
        raise ValueError("Capture start order must form off/on/on/off ABBA blocks; use --exploratory for incomplete evidence")
    output = (args.output or args.root / "PriorityAnalysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = []
    for name in (Path(__file__).name, "AnalyzeBLASAllocator.py", "AnalyzeZorahFullRoam.py"):
        source = Path(__file__).resolve().with_name(name)
        destination = output / name
        if source != destination:
            shutil.copy2(source, destination)
        scripts.append(file_identity(destination))
    versions = {}
    for package in ("matplotlib", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    command = [sys.executable, "-B", str(output / Path(__file__).name), str(args.root.resolve()), "--output", str(output),
               "--discard-first", str(args.discard_first)]
    if args.exploratory:
        command.append("--exploratory")
    if args.no_plot:
        command.append("--no-plot")
    if args.compare_cache:
        command.append("--compare-cache")
    if args.compare_merged_sharing:
        command.append("--compare-merged-sharing")
    if args.compare_intermediate_fallback:
        command.append("--compare-intermediate-fallback")
    caveats = [PROXY_DEFINITION, STORAGE_SEMANTICS_NOTE,
        "Full RT preparation is the per-execution sum of same-parent/same-queue RT representation and BLAS siblings, including priority sort inside classify and group admission inside cut. It excludes Fallback BLAS, traversal parents and TLAS; stage P99 values are never added.",
        "Frame time includes Reflex pacing and presentation. The GPU envelope excludes editor composite/present and independent texture submissions; neither metric is a pure BLAS kernel duration.",
        "Counters are delayed, deduplicated and aligned by blasFeedbackFrame to runtime frame identity. Missing feedback is reported, never filled with zero.",
        "Allocation diagnostics added by later capture schemas retain per-field available/missing sample counts. Missing fields in older captures remain absent from raw samples and are never treated as zero; statistics use available producer-aligned feedback only.",
        "Post-build budget rejection, physical allocation failure, unknown-build, projected retry deferral and projected dirty-claim diagnostics are instrumented only with priority on. Baseline raw zeros are retained but classified not instrumented/NA and excluded from statistics; the baseline's total blasFailedAllocations remains meaningful. Waiting claims/counts describe priority-policy state, so baseline zeros indicate an inactive policy. Compaction target is valid on all GPU allocator paths.",
        "FLT_MAX screen-error bounds are unbounded or saturated conservative bounds, not measured pixel errors. Raw metrics remain in Report.json/Samples.json; screenErrorBounds counts them separately and finite summaries/plots exclude them. Their cause cannot be determined from the scalar counter alone.",
        "Equal budgets and routes do not guarantee identical dynamic residency. This is a warmed streaming trajectory, not a cold-start measurement.",
        "Selected BLAS bytes include persistent, dirty staging, relocation staging and Move scratch only. They exclude build scratch, metadata and references. Device memory peaks are device-wide 1 Hz observations over loading/warmup/route, not per-process peaks.",
        "GPU competition evidence is reported separately per run. A Get-Counter failure or missing/empty GpuProcesses.csv does not establish absence of competition; raw stderr identity and an excerpt are retained. Whole-device utilization cannot substitute for per-process coverage.",
        "Run ranges and ABBA block ratios are descriptive, not confidence intervals. One ABBA block provides only one paired estimate; do not infer repeatability from a single block.",
        "Quality/validation runs are rejected here. Compare matched route-sample shadow, lighting and reflection outputs separately; selection proxies cannot establish image quality or temporal stability."]
    blocks = []
    if abba:
        for index in range(0, len(results), 4):
            block = results[index:index + 4]
            gains = {}
            for metric in ("frameMs", "rtPreparationGpuMs", "graphGpuEnvelopeMs"):
                values = [result["timings"][metric]["mean"] for result in block]
                gains[metric] = math.exp((math.log(values[0]) + math.log(values[3]) - math.log(values[1]) - math.log(values[2])) / 2) if min(values) > 0 else None
            blocks.append({"runs": [result["run"] for result in block], "baselineOverCandidateGeometricRatio": gains})
    report = {"conditions": expected, "comparison": comparison, "runs": results, "completeABBABlocks": len(blocks), "blocks": blocks,
        "exploratory": args.exploratory, "proxyDefinition": PROXY_DEFINITION, "caveats": caveats,
        "reproducibility": {"python": sys.version, "platform": platform.platform(), "dependencies": versions,
            "scripts": scripts, "commandArguments": command, "windowsCommand": "& " + subprocess.list2cmdline(command)}}
    (output / "Report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "Samples.json").write_text(json.dumps(observations, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# BLAS owner-priority comparison", "", PROXY_DEFINITION, "",
        f"| Run | {'Merged sharing (priority/cache on)' if args.compare_merged_sharing else 'Admission cache (priority on)' if args.compare_cache else 'Priority'} | Frame mean/P99 ms | Full RT prep GPU mean/P99 ms | Graph GPU mean/P99 ms | Selected BLAS peak MiB | Device sampled peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for result in results:
        values = [f"{result['timings'][key]['mean']:.3f}/{result['timings'][key]['p99']:.3f}" for key in ("frameMs", "rtPreparationGpuMs", "graphGpuEnvelopeMs")]
        lines.append(f"| {result['run']} | {'on' if result['comparisonEnabled'] else 'off'} | " + " | ".join(values) +
                     f" | {result['metrics']['blasBufferBytes']['max']/MIB:.3f} | {result['deviceSampledPeakMiB']} |")
    lines += ["", "| Run | Rejected visible consumers mean/P99 | Finite visible error bound px mean/P99 | Admitted offscreen consumers mean | Evicted owners total | Reflex CPU wait mean/P99 ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for result in results:
        metrics, pacing = result["metrics"], result["timings"]["reflexPacingCpuMs"]
        rejected = metrics["blasRejectedVisibleConsumers"]
        bound = result["screenErrorBounds"]["blasRejectedVisibleMaxErrorPixels"]["finiteBoundsPixels"]
        values = [f"{rejected['mean']:.3f}/{rejected['p99']:.3f}",
                  f"{bound['mean']:.3f}/{bound['p99']:.3f}" if bound else "none (all unbounded/saturated)"]
        pacing_text = f"{pacing['mean']:.3f}/{pacing['p99']:.3f}" if pacing else "not recorded"
        lines.append(f"| {result['run']} | " + " | ".join(values) + f" | {metrics['blasAdmittedOffscreenConsumers']['mean']:.3f} | " +
                     f"{sum(sample['blasPriorityEvictedOwners'] for sample in observations[result['run']])} | {pacing_text} |")
    lines += ["", "| Run | All rejected bound: unbounded/saturated samples | Visible rejected bound: unbounded/saturated samples |", "|---|---:|---:|"]
    for result in results:
        values = [f"{result['screenErrorBounds'][key]['unboundedOrSaturatedSamples']}/{result['screenErrorBounds'][key]['sampleCount']}" for key in PRIORITY_ERRORS]
        lines.append(f"| {result['run']} | " + " | ".join(values) + " |")
    lines += ["", "| Run | Allocation diagnostic | Status | Available/missing/not instrumented samples | Mean | P99 | Max |",
              "|---|---|---|---:|---:|---:|---:|"]
    for result in results:
        for key, diagnostic in result["allocationDiagnostics"].items():
            statistics = diagnostic["statistics"]
            values = [f"{statistics[metric]:.3f}" for metric in ("mean", "p99", "max")] if statistics else ["NA (not instrumented)" if diagnostic["status"] == "not instrumented" else "missing"] * 3
            lines.append(f"| {result['run']} | {key} | {diagnostic['status']} | {diagnostic['availableSamples']}/{diagnostic['missingSamples']}/{diagnostic['notInstrumentedSamples']} | " + " | ".join(values) + " |")
    lines += ["", "| Run | GPU competition logger | Activity rows | Stderr bytes |", "|---|---|---:|---:|"]
    for result in results:
        evidence = result["gpuCompetition"]
        lines.append(f"| {result['run']} | {evidence['status']} | {evidence['sampleRows']} | {evidence['stderrBytes']} |")
    lines += ["", "| Run | Process | PID | Active timestamps | Activity rows | Peak reported engine utilization % | Engine types |",
              "|---|---|---|---:|---:|---:|---|"]
    for result in results:
        for process in result["gpuCompetition"]["processActivity"]:
            lines.append(f"| {result['run']} | {process['process']} | {process['pid']} | {process['activeTimestamps']} | {process['activityRows']} | {process['maxReportedEngineUtilizationPercent']:.3f} | {', '.join(process['engineTypes'])} |")
    lines += ["", "Process rows report valid active engine samples over launch/loading/warmup/route, not route-only continuous coverage. Peaks are not summed GPU utilization. Activity-with-errors retains valid rows while acknowledging failed counter samples."]
    lines += ["", "Counter errors, missing data and header-only CSV files are evidence gaps, not proof of an idle GPU. Report.json retains per-run stderr excerpts and raw-file SHA-256 identities."]
    lines += ["", f"Complete ABBA blocks: {len(blocks)}. Ratios and all per-frame/stage timings are in Report.json; aligned feedback is in Samples.json.", ""]
    lines += ["- " + caveat for caveat in caveats]
    if not args.no_plot:
        plot(output, results, observations)
        lines += ["", "![Priority cost and selection proxies](Priority.png)"]
    lines += ["", "## Reproduce", "", f"Python {platform.python_version()}; dependencies {json.dumps(versions)}. Input and copied script SHA-256 identities are in Report.json.",
              "", "```powershell", report["reproducibility"]["windowsCommand"], "```", ""]
    (output / "Report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results), "completeABBABlocks": len(blocks)}))


if __name__ == "__main__":
    main()
