"""Summarize workload snapshots separately from uninstrumented timing samples."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summarize(root):
    capture = load(root / "Capture.json")
    if capture["status"] != "capture_complete":
        raise ValueError("Incomplete capture")
    snapshots = [("roam", capture.get("workloads", []))]
    for case in capture.get("cases", []):
        for point in ("before", "after"):
            snapshots.append((case["name"] + "-" + point, case[point].get("workloads", [])))
    result = {"diagnosticCountsOnly": True, "samples": [], "waitReasons": {}, "waitCpuMs": {}}
    for label, samples in snapshots:
        bins = {(w["frame"], w["phase"]): w["bins"] for w in samples if "bins" in w}
        for sample in samples:
            if "counts" not in sample:
                continue
            c = sample["counts"]
            key = (sample["frame"], "AfterStreamEarlyBins" if sample["phase"] == "StreamEarlyWorkload" else "AfterStreamLateBins")
            if key not in bins or c["clusters"] + c["invalidClusters"] != bins[key]["softwareClusters"]:
                raise ValueError(f"Dispatch/bin count mismatch: {label}, {key}")
            if c["coveredSamples"] != c["atomicAttempts"] or c["coveredSamples"] > c["bboxVisits"]:
                raise ValueError("Invalid coverage/atomic counts")
            if c["nonemptyTriangles"] + c["emptyTriangles"] != c["triangles"]:
                raise ValueError("Triangle accounting mismatch")
            result["samples"].append(dict(label=label, **sample, bins=bins[key],
                emptyTriangleFraction=c["emptyTriangles"] / c["triangles"] if c["triangles"] else None,
                bboxCoverageFraction=c["coveredSamples"] / c["bboxVisits"] if c["bboxVisits"] else None,
                triangleSlotFill=c["triangles"] / c["launchedTriangleLanes"] if c["launchedTriangleLanes"] else None))
    frames_file = root / "Frames.jsonl"
    if frames_file.exists():
        frames = [json.loads(line) for line in frames_file.read_text(encoding="utf-8-sig").splitlines()]
        definitions = {s["id"]: s["path"] for s in capture["scopes"]}
        values = defaultdict(list)
        reasons = Counter()
        for frame in frames:
            reason = frame.get("graphPreparation", {})
            reasons[str(reason.get("drainReasonMask", "missing"))] += 1
            for scope in frame["scopes"]:
                name = definitions[scope["id"]]
                if "Graph preparation / " in name:
                    values[name.split("Graph preparation / ")[-1]].append(scope["cpuMs"])
        result["waitReasons"] = dict(reasons)
        result["waitCpuMs"] = {name: dict(count=len(v), mean=sum(v)/len(v), max=max(v)) for name, v in values.items()}
    (root / "WorkloadSummary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(result["samples"]), "waitReasons": result["waitReasons"]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="run directory containing Capture.json")
    summarize(parser.parse_args().directory)
