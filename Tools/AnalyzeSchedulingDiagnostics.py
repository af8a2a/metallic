"""Summarize measured CPU scheduling domains; never infer GPU idle from CPU data."""
import argparse
import json
from pathlib import Path
from statistics import mean


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "samples": len(ordered), "mean": mean(ordered),
        "p50": ordered[int(.50 * (len(ordered) - 1))],
        "p95": ordered[int(.95 * (len(ordered) - 1))],
        "p99": ordered[int(.99 * (len(ordered) - 1))], "max": ordered[-1],
    }


def metrics_summary(rows):
    metrics = [row["scheduling"] for row in rows if row["scheduling"]["enabled"]]
    if not metrics:
        return {}
    for metric in metrics:
        require(metric["invalidScopes"] == 0, "Unbalanced or unobserved rendering scope")
        require(metric["nativeSubmitNs"] <= metric["submitNs"] <= metric["executeNs"], "Nested submission domain mismatch")
        require(0 < metric["firstSubmitNs"] <= metric["firstPassSubmitNs"] <= metric["executeNs"], "Missing first pass submission")
        require(metric["recordingEndNs"] <= metric["executeNs"], "Recording endpoint outside execute")
    result = {}
    for key in metrics[0]:
        if key == "enabled":
            continue
        scale = 1e-6 if key.endswith("Ns") else 1
        label = key[:-2] + "Ms" if key.endswith("Ns") else key
        result[label] = distribution([metric[key] * scale for metric in metrics])
    result["executeWithoutFrameWaitMs"] = distribution([
        (metric["executeNs"] - metric["frameWaitNs"]) * 1e-6 for metric in metrics])
    result["postRecordingToReturnMs"] = distribution([
        (metric["executeNs"] - metric["recordingEndNs"]) * 1e-6 for metric in metrics])
    return result


def benchmark(path):
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    groups = {}
    for row in document["rows"]:
        key = f"passes={row['passCount']},mode={row['mode']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, rows in groups.items():
        example = rows[0]
        summary[key] = {
            "workers": example["workers"], "batchWorkload": example["batchWorkload"],
            "pipelined": example["pipelined"], "diagnostics": example["scheduling"]["enabled"],
            "wallMs": distribution([row["wallMs"] for row in rows]),
            "byRepeat": {repeat: distribution([row["wallMs"] for row in rows if row["repeat"] == repeat])
                         for repeat in sorted({row["repeat"] for row in rows})},
            **metrics_summary(rows),
        }
    return {"source": str(path.resolve()), "protocol": document["protocol"],
            "warmupFrames": document["warmupFrames"], "validation": document["validation"], "groups": summary}


def minizorah(path, warmup):
    baseline = json.loads((path.parent / "Baseline.json").read_text(encoding="utf-8-sig"))
    require(baseline["status"] == "passed", "Scene correctness checks did not pass")
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    require(len(all_rows) == baseline["frameCount"], "Incomplete scene capture")
    require([row["frame"] for row in all_rows] == list(range(len(all_rows))), "Missing/reordered frames")
    rows = all_rows[warmup:]
    require(rows, "No measured frames after warmup")
    phases = {}
    sections = {}
    for row in rows:
        for name, duration in row["cpuPhases"].items():
            phases.setdefault(name, []).append(duration)
        for node in row["nodes"]:
            paths = []
            durations = {}
            for index, section in enumerate(node["sections"]):
                parent = section["parent"]
                name = (paths[parent] if parent < index else node["name"]) + "/" + section["name"]
                paths.append(name)
                durations[name] = durations.get(name, 0) + section["cpuMs"]
            for name, duration in durations.items():
                sections.setdefault(name, []).append(duration)
    return {"source": str(path.resolve()), "warmupFramesExcluded": warmup, "measuredFrames": len(rows),
            "validation": baseline["validation"], "clasEnabled": baseline["clasEnabled"],
            "recordingWorkerLimit": baseline.get("recordingWorkerLimit", 0),
            "finalHoldFrames": baseline.get("finalHoldFrames", 0), "rasterQueues": baseline["rasterQueues"],
            "cpuExecuteMs": distribution([row["cpuExecuteMs"] for row in rows]),
            "hostFrameMs": distribution([row["hostFrameMs"] for row in rows]),
            "gpuMs": distribution([row["gpuMs"] for row in rows]),
            "pipelinedFrames": sum(row["pipelinedSubmission"] for row in rows),
            "blockingPasses": sorted({name for row in rows for name in row["submissionBlockingPasses"]}),
            "overlapBlockingPasses": sorted({name for row in rows for name in row.get("overlapBlockingPasses", [])}),
            "drainReasonMasks": sorted({row["drainReasonMask"] for row in rows if "drainReasonMask" in row}),
            "recordingTasks": distribution([row["recordingTasks"] for row in rows]),
            "preparationTasks": distribution([row["preparationTasks"] for row in rows]),
            "cpuPhasesMs": {name: distribution(values) for name, values in phases.items()},
            "cpuSectionsMs": {name: distribution(values) for name, values in sections.items()},
            "routePhases": {phase: {"frames": sum(row["phase"] == phase for row in rows),
                                   **metrics_summary([row for row in rows if row["phase"] == phase])}
                            for phase in sorted({row["phase"] for row in rows})},
            **metrics_summary(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, nargs="*", default=[])
    parser.add_argument("--minizorah", type=Path, nargs="*", default=[])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or not (args.benchmark or args.minizorah):
        parser.error("Supply results and a nonnegative warmup")
    summary = {"scope": "CPU wall-clock attribution. Nested and parallel domains are not additive; concurrent GPU workloads confound GPU/frame comparisons.",
               "benchmarks": [benchmark(path) for path in args.benchmark],
               "minizorah": [minizorah(path, args.warmup) for path in args.minizorah]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(summary['benchmarks'])} benchmark and {len(summary['minizorah'])} scene summaries to {args.output}")


if __name__ == "__main__":
    main()
