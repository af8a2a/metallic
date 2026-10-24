"""Analyze archived BLAS allocator A/B captures; current runtime has one production allocator.

Historical manifest fields remain required. This tool does not enable retired runtime paths.
Never infer visual correctness from timings.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from AnalyzeZorahFullRoam import diagnostic_run, distribution


FIELDS = (
    "blasEncodedBytes", "blasAllocatedBytes", "blasFreeBytes", "blasLargestFreeBytes",
    "blasRelocatedCount", "blasRelocatedBytes", "blasSuppressedBuilds", "blasFailedAllocations",
    "blasPersistentBytes", "blasStagingBytes", "blasRelocationStagingBytes", "blasMoveScratchBytes", "blasArenaUsedBytes",
    "blasBuildCount", "blasArenaRepack", "blasStorageRejected", "blasSemanticChangeCount",
    "blasAdmittedInstances", "blasRequestedInstances", "blasLiveClusterReferences",
)
MIB = 1024 * 1024


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_conditions(manifest, capture):
    if manifest.get("blasAllocator") not in ("legacy", "gpu"):
        raise ValueError("Archived A/B manifest must record blasAllocator=legacy or gpu")
    if not manifest.get("blasStorageBytes"):
        raise ValueError("Capacity-pressure comparison requires an explicit shared BLAS storage budget")
    if capture.get("status") != "capture_complete" or not capture.get("frames"):
        raise ValueError("Incomplete capture")
    if capture.get("missingGpuFrames") != 0:
        raise ValueError("Capture has missing or unreported GPU frame coverage")
    if diagnostic_run(capture) or capture.get("validationRequested"):
        raise ValueError("Use separate validation/quality runs; diagnostic timings are not acceptance samples")
    if capture.get("trajectoryClock") != "deterministic frame index":
        raise ValueError("Pressure comparison requires deterministic routeFrames")
    if capture.get("streamingFreeze", {}).get("enabled"):
        raise ValueError("Frozen streaming does not exercise allocator pressure")
    return {
        "manifest": {key: manifest.get(key) for key in (
            "executableSha256", "shaderSha256", "asset", "gpu", "config", "validation", "vsync",
            "hidden", "blasRepresentations", "blasInstanceReuse", "clasAllocator", "blasStorageBytes",
            "blasRelocateBytes", "reflexMode", "streamMaintenanceBeforePacing")},
        "capture": {key: capture.get(key) for key in (
            "outputExtent", "renderExtent", "config", "absoluteKeyframes", "graph", "hidden", "vsync",
            "shaderDebugMode", "graphicsCaptureInjected", "frameTimeDefinition", "gpuTimeDefinition")},
    }


def analyze_run(directory, discard_first):
    manifest_path = directory.parent / "Manifest.json"
    capture_path = directory / "Capture.json"
    frames_path = directory / "Frames.jsonl"
    manifest, capture = read_json(manifest_path), read_json(capture_path)
    conditions = validate_conditions(manifest, capture)
    all_rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8-sig").splitlines() if line]
    if len(all_rows) != capture["frames"] or len({row["frame"] for row in all_rows}) != len(all_rows):
        raise ValueError(f"Frame count/identity mismatch: {directory}")
    if len(all_rows) != capture.get("config", {}).get("routeFrames"):
        raise ValueError("Completed capture does not contain the full configured deterministic route")
    if discard_first >= len(all_rows):
        raise ValueError("Discard interval removes every measured frame")
    rows = all_rows[discard_first:]
    streams_by_row = []
    route_frames = {}
    for row in rows:
        streams = [sample for sample in row["streaming"] if sample.get("pass") == "VBuffer"]
        if len(streams) != 1:
            raise ValueError("Expected one VBuffer streaming sample per frame")
        stream = streams[0]
        # Runtime stream frame IDs have their own origin; they are not editor
        # profiler frame IDs. Match feedback against the producing runtime ID.
        if stream["frame"] in route_frames:
            raise ValueError("Repeated runtime frame identity in measured route")
        route_frames[stream["frame"]] = row
        streams_by_row.append((row, stream))
    timings = {}
    definitions = {entry["id"]: entry for entry in capture["scopes"]}
    chain_definitions = [entry for entry in capture["scopes"]
                         if entry["path"].rsplit("/", 1)[-1].startswith("BLAS ") and not entry.get("cpuOnly")]
    if not chain_definitions or len({entry["path"].rsplit("/", 1)[0] for entry in chain_definitions}) != 1:
        raise ValueError("BLAS timing sum requires sibling scopes with exactly one parent")
    if len({entry["queue"] for entry in chain_definitions}) != 1:
        raise ValueError("Cannot sum BLAS timings across concurrent queues")
    chain_ids = {entry["id"] for entry in chain_definitions}
    chain_names = {entry["id"]: entry["path"].rsplit("/", 1)[-1] for entry in chain_definitions}
    envelope_ids = {entry["id"] for entry in capture["scopes"] if entry["path"].endswith("/RenderGraph GPU envelope")}
    reflex_path = "Frame/Idle / Reflex Frame Pacing"
    reflex_ids = {entry["id"] for entry in capture["scopes"] if entry["path"] == reflex_path}
    frame_timings = []
    for row in rows:
        if not math.isfinite(row["frameMs"]) or row["frameMs"] <= 0:
            raise ValueError("Invalid frame timing")
        for scope in row["scopes"]:
            path = definitions[scope["id"]]["path"]
            if "BLAS" not in path and not path.endswith("/RenderGraph GPU envelope") and path != reflex_path:
                continue
            record = timings.setdefault(path, {"cpuMs": [], "gpuMs": []})
            for kind in ("cpuMs", "gpuMs"):
                value = scope.get(kind)
                if value is not None:
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid {kind}: {path}")
                    record[kind].append(value)
        chain = [scope for scope in row["scopes"] if scope["id"] in chain_ids]
        if len(chain) != len(chain_ids) or {scope["id"] for scope in chain} != chain_ids:
            raise ValueError("Missing or repeated sibling BLAS stage in a measured frame")
        if any(scope.get("gpuMs") is None for scope in chain):
            raise ValueError("Missing sibling BLAS GPU timing; cannot manufacture a chain sum")
        execution_ids = {scope.get("executionId") for scope in chain}
        if None in execution_ids or len(execution_ids) != 1:
            raise ValueError("Sibling BLAS stages refer to different GPU executions")
        envelope = [scope for scope in row["scopes"] if scope["id"] in envelope_ids]
        reflex = [scope for scope in row["scopes"] if scope["id"] in reflex_ids]
        if len(reflex) > 1:
            raise ValueError("Repeated Reflex pacing scope in a measured frame")
        frame_timings.append({"frame": row["frame"], "routeSample": row["routeSample"], "stage": row["stage"],
            "frameMs": row["frameMs"], "executionId": next(iter(execution_ids)),
            "reflexPacingCpuMs": reflex[0].get("cpuMs") if reflex else None,
            "blasChainGpuMs": sum(scope["gpuMs"] for scope in chain),
            "blasChainCpuMs": sum(scope["cpuMs"] for scope in chain),
            "graphGpuEnvelopeMs": envelope[0]["gpuMs"] if len(envelope) == 1 else None,
            "blasStageGpuMs": {chain_names[scope["id"]]: scope["gpuMs"] for scope in chain}})
    build_scopes = [values for path, values in timings.items() if path.endswith("/BLAS build")]
    if len(build_scopes) != 1 or len(build_scopes[0]["gpuMs"]) != len(rows):
        raise ValueError("Missing per-frame BLAS build GPU timing coverage")

    observed, duplicates, outside_route = {}, 0, 0
    gpu_mode = manifest["blasAllocator"] == "gpu"
    for row, stream in streams_by_row:
        if stream.get("blasGPUAllocatorEnabled") is not gpu_mode:
            raise ValueError("Requested allocator differs from effective runtime allocator")
        if not stream.get("blasFeedbackAvailable"):
            continue
        frame_id = stream["blasFeedbackFrame"]
        if frame_id not in route_frames:
            outside_route += 1
            continue
        values = {key: stream[key] for key in FIELDS}
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("Invalid allocator counter")
        if frame_id in observed:
            duplicates += 1
            # CPU physical-buffer counters can advance independently of the
            # delayed GPU feedback; retain the first observed copy per GPU frame.
            continue
        if values["blasAllocatedBytes"] < values["blasEncodedBytes"]:
            raise ValueError("Encoded BLAS exceeds allocated live storage")
        if values["blasLargestFreeBytes"] > values["blasFreeBytes"]:
            raise ValueError("Largest free range exceeds all free bytes")
        if gpu_mode and values["blasAllocatedBytes"] + values["blasFreeBytes"] != values["blasPersistentBytes"]:
            raise ValueError("GPU allocator live/free accounting does not cover its persistent arena")
        values.update({
            "frame": frame_id, "routeSample": route_frames[frame_id]["routeSample"],
            "stage": route_frames[frame_id]["stage"], "feedbackLag": stream["frame"] - frame_id,
            "internalWasteBytes": values["blasAllocatedBytes"] - values["blasEncodedBytes"],
            "fragmentedFreeBytes": values["blasFreeBytes"] - values["blasLargestFreeBytes"],
            "legacyUnreclaimedBytes": max(0, values["blasArenaUsedBytes"] - values["blasAllocatedBytes"]) if not gpu_mode else 0,
            "reusableHoleBytesBelowHighWater": max(0, values["blasArenaUsedBytes"] - values["blasAllocatedBytes"]) if gpu_mode else 0,
            "cleanMoveTrafficBytes": 2 * values["blasRelocatedBytes"],
            "blasBufferBytes": values["blasPersistentBytes"] + values["blasStagingBytes"] +
                values["blasRelocationStagingBytes"] + values["blasMoveScratchBytes"],
        })
        observed[frame_id] = values
    samples = sorted(observed.values(), key=lambda item: item["routeSample"])
    if not samples:
        raise ValueError("No route-aligned BLAS feedback")
    if len(samples) < .9 * len(rows):
        raise ValueError("Less than 90% unique route-aligned BLAS feedback; repeat the measurement")
    if gpu_mode and any(sample["blasArenaRepack"] for sample in samples):
        raise ValueError("GPU allocator performed a full repack")
    if gpu_mode and manifest.get("blasRelocateBytes") is not None:
        if any(sample["blasRelocatedBytes"] > int(manifest["blasRelocateBytes"]) for sample in samples):
            raise ValueError("GPU allocator exceeded the declared relocation byte budget")
    settled = [sample for sample in samples if sample["stage"] == "final-static"]
    gpu_peak = None
    gpu_csv = directory / "Gpu.csv"
    if gpu_csv.exists():
        with gpu_csv.open(encoding="utf-8-sig", newline="") as source:
            records = csv.DictReader(source)
            memory = []
            for record in records:
                key = next((key for key in record if key and key.strip().startswith("memory.used")), None)
                if key:
                    try:
                        memory.append(float(record[key].strip().split()[0]))
                    except (ValueError, IndexError):
                        pass
            gpu_peak = max(memory) if memory else None
    metrics = {key: distribution([sample[key] for sample in samples]) for key in (*FIELDS,
        "internalWasteBytes", "fragmentedFreeBytes", "legacyUnreclaimedBytes", "reusableHoleBytesBelowHighWater",
        "cleanMoveTrafficBytes", "blasBufferBytes", "feedbackLag")}
    result = {
        "run": directory.parent.name + "/" + directory.name,
        "allocator": manifest["blasAllocator"], "started": manifest.get("started"),
        "recordedFrames": len(all_rows), "measuredFrames": len(rows), "discardedFirstFrames": discard_first,
        "feedbackSamples": len(samples), "duplicateFeedback": duplicates, "outsideRouteFeedback": outside_route,
        "missingRouteFeedback": len(rows) - len(samples),
        "frameMs": distribution([row["frameMs"] for row in rows]), "metrics": metrics,
        "blasChainGpuMs": distribution([row["blasChainGpuMs"] for row in frame_timings]),
        "blasChainCpuMs": distribution([row["blasChainCpuMs"] for row in frame_timings]),
        "reflexPacingCpuMs": distribution([row["reflexPacingCpuMs"] for row in frame_timings
                                           if row["reflexPacingCpuMs"] is not None]),
        "reflexPacingSamples": sum(row["reflexPacingCpuMs"] is not None for row in frame_timings),
        "chainScopeDefinitions": chain_definitions,
        "chainDefinition": "Per-execution sum of same-parent, same-queue BLAS-prefixed sibling scopes; excludes Fallback BLAS, traversal parents, RT representation classification/cuts and TLAS. Sum before calculating percentiles.",
        "frameTimings": frame_timings,
        "routeStageTimings": {stage: {
            "frameMs": distribution([row["frameMs"] for row in frame_timings if row["stage"] == stage]),
            "blasChainGpuMs": distribution([row["blasChainGpuMs"] for row in frame_timings if row["stage"] == stage]),
        } for stage in dict.fromkeys(row["stage"] for row in frame_timings)},
        "scopes": {path: {kind: distribution(values) for kind, values in counts.items()} for path, counts in timings.items()},
        "repackFrames": sum(sample["blasArenaRepack"] != 0 for sample in samples),
        "failedAllocationFrames": sum(sample["blasFailedAllocations"] != 0 for sample in samples),
        "rejectedFrames": sum(sample["blasStorageRejected"] != 0 for sample in samples),
        "finalStatic": {"samples": len(settled), "builds": sum(sample["blasBuildCount"] for sample in settled),
            "relocations": sum(sample["blasRelocatedCount"] for sample in settled),
            "last32Builds": [sample["blasBuildCount"] for sample in settled[-32:]],
            "last32Relocations": [sample["blasRelocatedCount"] for sample in settled[-32:]]},
        "deviceSampledPeakMiB": gpu_peak,
        "inputs": [file_identity(path) for path in (manifest_path, capture_path, frames_path)] +
            ([file_identity(gpu_csv)] if gpu_csv.exists() else []),
    }
    return conditions, result, samples


def plot(output, results, samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    fields = (("blasBuildCount", "BLAS builds", 1), ("cleanMoveTrafficBytes", "Clean Move traffic MiB (scratch round trip)", MIB),
              ("internalWasteBytes", "Live internal waste MiB", MIB), ("fragmentedFreeBytes", "Fragmented free MiB", MIB),
              ("blasBufferBytes", "Selected BLAS storage components MiB", MIB), ("blasStorageRejected", "Storage-rejected owners", 1))
    for axis, (field, title, scale) in zip(axes.flat, fields):
        for result, run_samples in zip(results, samples):
            axis.plot([sample["routeSample"] for sample in run_samples],
                      [sample[field] / scale for sample in run_samples], label=result["run"], linewidth=.8, alpha=.8)
        axis.set(title=title, xlabel="Deterministic route sample")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("BLAS allocator capacity-pressure replay (delayed feedback aligned by GPU frame)")
    figure.savefig(output / "Comparison.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for result in results:
        timing = result["frameTimings"]
        label = f"{result['run']} {result['allocator']} (n={len(timing)})"
        for row, (field, title) in enumerate((("frameMs", "Editor frame"), ("blasChainGpuMs", "BLAS sibling-stage GPU sum"))):
            values = [sample[field] for sample in timing]
            axes[row, 0].plot([sample["routeSample"] for sample in timing], values, label=label, linewidth=.7, alpha=.8)
            ordered = sorted(values)
            axes[row, 1].step(ordered, [(index + 1) / len(ordered) for index in range(len(ordered))],
                              where="post", label=label, linewidth=1)
            axes[row, 0].set(title=title + " timeline", xlabel="Deterministic route sample", ylabel="Milliseconds")
            axes[row, 1].set(title=title + " empirical distribution", xlabel="Milliseconds", ylabel="Cumulative fraction of frames")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Per-run frame timings; full measured route after configured warmup")
    figure.savefig(output / "Timings.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    for result in results:
        timing = [sample for sample in result["frameTimings"] if sample["reflexPacingCpuMs"] is not None]
        if not timing:
            continue
        label = f"{result['run']} {result['allocator']} (n={len(timing)})"
        values = [sample["reflexPacingCpuMs"] for sample in timing]
        axes[0].plot([sample["routeSample"] for sample in timing], values, label=label, linewidth=.7, alpha=.8)
        ordered = sorted(values)
        axes[1].step(ordered, [(index + 1) / len(ordered) for index in range(len(ordered))],
                     where="post", label=label, linewidth=1)
    axes[0].set(title="Reflex pacing CPU wait timeline", xlabel="Deterministic route sample", ylabel="Milliseconds")
    axes[1].set(title="Reflex pacing CPU wait empirical distribution", xlabel="Milliseconds",
                ylabel="Cumulative fraction of recorded samples")
    for axis in axes:
        axis.grid(alpha=.2)
    if axes[0].lines:
        axes[0].legend(fontsize=8)
    figure.suptitle("Frame time includes Reflex pacing; full-frame changes cannot be attributed to BLAS alone")
    figure.savefig(output / "ReflexPacing.png", dpi=140)
    plt.close(figure)

    names = list(dict.fromkeys(entry["path"].rsplit("/", 1)[-1]
                              for result in results for entry in result["chainScopeDefinitions"]))
    figure, axes = plt.subplots(1, 2, figsize=(16, max(6, .65 * len(names))), constrained_layout=True)
    width = .8 / len(results)
    for run_index, result in enumerate(results):
        for axis, statistic in zip(axes, ("mean", "p99")):
            values = []
            for name in names:
                path = next((entry["path"] for entry in result["chainScopeDefinitions"]
                             if entry["path"].rsplit("/", 1)[-1] == name), None)
                values.append(result["scopes"][path]["gpuMs"][statistic] if path else math.nan)
            axis.barh([index - .4 + width * (run_index + .5) for index in range(len(names))],
                      values, height=width, label=result["run"] + " " + result["allocator"])
            axis.set(yticks=range(len(names)), yticklabels=names, xlabel="GPU milliseconds", title=statistic.upper() + " per stage")
            axis.grid(axis="x", alpha=.2)
    for axis in axes:
        axis.invert_yaxis()
    axes[0].legend(fontsize=8)
    figure.suptitle("Sibling BLAS GPU stages; absent variant-specific stages are missing, not zero")
    figure.savefig(output / "Stages.png", dpi=140)
    plt.close(figure)


def run_range(values):
    """Descriptive range of recorded run summaries; deliberately no CI."""
    return {"runs": len(values), "meanOfRuns": sum(values) / len(values), "min": min(values), "max": max(values)}


def summarize_modes(results):
    return {mode: {
        "frameMeanMs": run_range([result["frameMs"]["mean"] for result in results if result["allocator"] == mode]),
        "frameP99Ms": run_range([result["frameMs"]["p99"] for result in results if result["allocator"] == mode]),
        "blasChainGpuMeanMs": run_range([result["blasChainGpuMs"]["mean"] for result in results if result["allocator"] == mode]),
        "blasChainGpuP99Ms": run_range([result["blasChainGpuMs"]["p99"] for result in results if result["allocator"] == mode]),
        "blasChainGpuMaxMs": run_range([result["blasChainGpuMs"]["max"] for result in results if result["allocator"] == mode]),
    } for mode in ("legacy", "gpu")}


def reproducibility(output, args):
    scripts = []
    for name in (Path(__file__).name, "AnalyzeZorahFullRoam.py"):
        source = Path(__file__).resolve().with_name(name)
        destination = output.resolve() / name
        if source != destination:
            shutil.copy2(source, destination)
        scripts.append(file_identity(destination))
    command = [sys.executable, "-B", str(output.resolve() / Path(__file__).name),
               str(args.root.resolve()), "--output", str(output.resolve()), "--discard-first", str(args.discard_first)]
    if args.no_plot:
        command.append("--no-plot")
    versions = {}
    for package in ("matplotlib", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "dependencies": versions,
            "commandArguments": command, "windowsCommand": "& " + subprocess.list2cmdline(command), "scripts": scripts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing allocator captures, e.g. A1/B1 or A1/B1/B2/A2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--discard-first", type=int, default=0, help="Explicit additional measured-route frame exclusion")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.discard_first < 0:
        parser.error("--discard-first cannot be negative")
    directories = sorted(path.parent for path in args.root.glob("*/run*/Capture.json"))
    if not directories:
        parser.error("No captures matching */run*/Capture.json")
    results, samples, expected = [], [], None
    for directory in directories:
        conditions, result, observations = analyze_run(directory, args.discard_first)
        if expected is not None and conditions != expected:
            raise ValueError(f"A/B workload, budget, binary, shader or warmup conditions differ: {directory}")
        expected = conditions
        results.append(result); samples.append(observations)
    variants = {result["allocator"] for result in results}
    if variants != {"legacy", "gpu"}:
        raise ValueError("Comparison needs both legacy and gpu captures")
    order = sorted(range(len(results)), key=lambda index: results[index]["started"] or "")
    results, samples = [results[index] for index in order], [samples[index] for index in order]
    output = args.output or args.root / "AllocatorAnalysis"
    output.mkdir(parents=True, exist_ok=True)
    caveats = [
        "Recorded frames begin after readiness and configured wall-clock warmup; this is not an I/O/cache cold-start comparison.",
        "Allocator counters are delayed GPU observations, deduplicated and aligned using blasFeedbackFrame. Missing tail feedback is reported, never assumed zero.",
        "Scope timings are inclusive recorded occurrences; nested parent/child scopes must not be summed. Conditional stages are not frame-amortized.",
        "The comparable BLAS chain sums only same-parent/same-queue BLAS-prefixed sibling scopes from each GPU execution before calculating mean/P99. Fallback BLAS, traversal parents, RT representation classification/cuts and TLAS are excluded.",
        "Mode summaries are means and observed ranges of the recorded runs, with each mode's run count reported explicitly. A single run per mode cannot measure repeatability; its identical range endpoints do not establish zero variation. These summaries are descriptive, not confidence intervals or a statistical speedup claim.",
        "Editor frame timing includes CPU Reflex Frame Pacing waits, reported separately. Differences in pacing prevent attribution of full-frame changes to the BLAS allocator; the pacing scope is never added to the GPU chain.",
        "Legacy free bytes include abandoned holes, while largest-free reports only its immediately allocatable append tail. legacyUnreclaimedBytes is high-water minus live allocation; GPU holes are reusable. These overlap fragmentedFreeBytes and must not be added twice.",
        "Live internal waste is allocated live bytes minus encoded live bytes; it excludes abandoned legacy holes. Reported peaks are separate maxima and must not be added as if they occurred in the same frame.",
        "Relocation budget/counters count logical BLAS bytes. Scratch-assisted compaction makes two native Move copies, so cleanMoveTrafficBytes is twice blasRelocatedBytes.",
        "blasBufferBytes sums selected storage components: persistent, dirty staging, relocation staging and Move scratch. It excludes common BLAS build scratch, reference and metadata buffers and is not total BLAS or process memory. Device CSV peaks are 1 Hz device-wide samples including loading/warmup and other processes.",
        "Counters/timings do not establish shadow or reflection quality. Inspect separate matched image/ray-query evidence, especially when rejected owner counts differ.",
    ]
    reproduction = reproducibility(output, args)
    mode_summary = summarize_modes(results)
    report = {"conditions": expected, "runs": results, "modeSummary": mode_summary,
              "reproducibility": reproduction, "caveats": caveats,
              "baselinePressureReproduced": any(result["repackFrames"] for result in results if result["allocator"] == "legacy")}
    (output / "Report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "Samples.json").write_text(json.dumps(dict(zip((result["run"] for result in results), samples)), indent=2), encoding="utf-8")
    lines = ["# BLAS allocator capacity-pressure comparison", "",
             "| Run | Frame mean / P99 ms | BLAS chain GPU mean / P99 / max ms | Build P99 / max | Repack frames | Live internal waste peak MiB | Selected BLAS storage peak MiB | Device sampled peak MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        metric = result["metrics"]
        lines.append(f"| {result['run']} ({result['allocator']}) | {result['frameMs']['mean']:.3f} / {result['frameMs']['p99']:.3f} | "
                     f"{result['blasChainGpuMs']['mean']:.3f} / {result['blasChainGpuMs']['p99']:.3f} / {result['blasChainGpuMs']['max']:.3f} | "
                     f"{metric['blasBuildCount']['p99']} / {metric['blasBuildCount']['max']} | {result['repackFrames']} | "
                     f"{metric['internalWasteBytes']['max']/MIB:.3f} | {metric['blasBufferBytes']['max']/MIB:.3f} | {result['deviceSampledPeakMiB']} |")
    lines += ["", f"Baseline repack pressure reproduced: {report['baselinePressureReproduced']}.", ""]
    lines += ["| Mode | Runs | Frame mean ms (run range) | Frame P99 ms (run range) | BLAS chain GPU mean ms (run range) | BLAS chain GPU P99 ms (run range) |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode, summary in mode_summary.items():
        formatted = [f"{summary[key]['meanOfRuns']:.3f} ({summary[key]['min']:.3f}–{summary[key]['max']:.3f})"
                     for key in ("frameMeanMs", "frameP99Ms", "blasChainGpuMeanMs", "blasChainGpuP99Ms")]
        lines.append(f"| {mode} | {summary['frameMeanMs']['runs']} | " + " | ".join(formatted) + " |")
    lines += ["", "Ranges are descriptive minima/maxima of the recorded run summaries; a single-run range cannot measure repeatability. No confidence intervals are computed.", "",
              "| Run | Reflex pacing CPU samples | Mean ms | P99 ms | Max ms |",
              "|---|---:|---:|---:|---:|"]
    for result in results:
        pacing = result["reflexPacingCpuMs"]
        values = [f"{pacing[key]:.3f}" if pacing and pacing[key] is not None else "not recorded"
                  for key in ("mean", "p99", "max")]
        lines.append(f"| {result['run']} ({result['allocator']}) | {result['reflexPacingSamples']} | " + " | ".join(values) + " |")
    pacing_means = [result["reflexPacingCpuMs"]["mean"] for result in results if result["reflexPacingSamples"]]
    if pacing_means:
        lines += ["", f"Observed per-run Reflex pacing means range from {min(pacing_means):.3f} to {max(pacing_means):.3f} ms. "
                  "This wait is part of editor frame time; full-frame differences cannot be attributed to BLAS alone."]
    lines += ["",
              "| GPU sibling stage (mean / P99 ms) | " + " | ".join(result["run"] for result in results) + " |",
              "|---|" + "---:|" * len(results)]
    stage_names = list(dict.fromkeys(entry["path"].rsplit("/", 1)[-1]
                                    for result in results for entry in result["chainScopeDefinitions"]))
    for name in stage_names:
        columns = []
        for result in results:
            path = next((entry["path"] for entry in result["chainScopeDefinitions"]
                         if entry["path"].rsplit("/", 1)[-1] == name), None)
            timing = result["scopes"][path]["gpuMs"] if path else None
            columns.append(f"{timing['mean']:.4f} / {timing['p99']:.4f}" if timing else "not recorded")
        lines.append("| " + name + " | " + " | ".join(columns) + " |")
    lines += ["", "Stage P99 values are not summed; the chain distribution comes from the per-frame sum.", ""]
    lines.extend("- " + text for text in caveats)
    if not args.no_plot:
        lines += ["", "![Allocator counters](Comparison.png)", "", "![Frame timing distributions](Timings.png)",
                  "", "![Reflex pacing waits](ReflexPacing.png)", "", "![Sibling BLAS stage timings](Stages.png)"]
    lines += ["", "## Reproduce", "", f"Python: `{platform.python_version()}`; dependencies: `{json.dumps(reproduction['dependencies'])}`.",
              "", "Scripts and input SHA-256 identities are recorded in Report.json. Samples.json retains the aligned allocator observations; Report.json retains per-frame timings.",
              "", "```powershell", reproduction["windowsCommand"], "```", ""]
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_plot:
        plot(output, results, samples)
    print(json.dumps({"output": str(output.resolve()), "runs": len(results),
                      "baselinePressureReproduced": report["baselinePressureReproduced"]}))


if __name__ == "__main__":
    main()
