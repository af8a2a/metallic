"""Compare CPU Stream Begin recordings from two builds on the same camera replay."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def summarize(values):
    ordered = sorted(values)
    return {"samples": len(values), "meanMs": statistics.mean(values),
            "p95Ms": ordered[int((len(values) - 1) * .95)], "maxMs": max(values)}


def capture(root, route, replay_hash):
    manifest = read(root / "Manifest.json")
    if manifest["replaySha256"].lower() != replay_hash:
        raise ValueError(f"Replay hash mismatch: {root}")
    result = {"directory": str(root.resolve()), "manifest": manifest, "cases": {}}
    for case in ("m1", "m2"):
        directory = root / case
        baseline = read(directory / "Baseline.json")
        if baseline["status"] != "passed" or baseline["validation"] or baseline["qualityRun"]:
            raise ValueError(f"Not a passed performance run: {directory}")
        if read(directory / "Cameras.json") != [f["camera"] for f in route["frames"]]:
            raise ValueError(f"Executed camera mismatch: {directory}")
        frames = [json.loads(line) for line in (directory / "Frames.jsonl").read_text().splitlines()]
        if len(frames) != len(route["frames"]) or baseline["frameCount"] != len(frames):
            raise ValueError(f"Incomplete frames: {directory}")
        groups = defaultdict(list)
        for index, frame in enumerate(frames):
            if frame["frame"] != index or frame["phase"] != route["frames"][index]["phase"]:
                raise ValueError(f"Frame/phase mismatch: {directory}")
            scopes = {}
            for node in frame["nodes"]:
                paths = []
                for section in node["sections"]:
                    parent = node["name"] if section["parent"] == 0xffffffff else paths[section["parent"]]
                    path = parent + "/" + section["name"]
                    paths.append(path)
                    if path == "GPUDriven/Stream Begin" or path.startswith("GPUDriven/Stream Begin/"):
                        scopes[path.removeprefix("GPUDriven/")] = section["cpuMs"]
            if "Stream Begin" not in scopes:
                raise ValueError(f"Missing CPU scopes: {directory}")
            groups[frame["phase"]].append(scopes)
            if frame["phase"] != "warmup":
                groups["allMeasured"].append(scopes)
        phases = {}
        for phase, rows in groups.items():
            names = set().union(*(row.keys() for row in rows))
            phases[phase] = {name: summarize([row.get(name, 0.) for row in rows]) for name in sorted(names)}
        result["cases"][case] = {"scopes": phases, "finalCounters": baseline["finalStream"]["stats"],
                                  "finalFrameStream": frames[-1]["stream"],
                                  "finalCut": baseline["quality"][-1]["cut"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    route = read(args.replay)
    replay_hash = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    before, after = (capture(root, route, replay_hash) for root in (args.before, args.after))
    if before["manifest"]["sourceSha256"] != after["manifest"]["sourceSha256"]:
        raise ValueError("Replay test source changed between builds")
    comparison = {}
    for phase, scopes in before["cases"]["m1"]["scopes"].items():
        comparison[phase] = {}
        for name in scopes:
            means = [[data["cases"][case]["scopes"][phase].get(name, {}).get("meanMs", 0.)
                      for case in ("m1", "m2")] for data in (before, after)]
            old, new = (statistics.mean(values) for values in means)
            comparison[phase][name] = {"beforeMs": means[0], "afterMs": means[1],
                                       "reductionPercent": (1 - new / old) * 100 if old else None}
    result = {"replaySha256": replay_hash, "before": before, "after": after, "comparison": comparison,
              "limitations": ["CPU steady-clock elapsed time includes scheduling/preemption.",
                  "Two serial runs per build; ranges are observations, not confidence intervals.",
                  "Fixed frame camera steps are identical; asynchronous I/O timing can change page events.",
                  "Conditional CPU scopes are amortized across every frame in the phase."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "Evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase in ("start_hold", "forward_2", "return_hold", "allMeasured"):
        if phase in comparison:
            print(phase, json.dumps(comparison[phase]["Stream Begin"]))
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["Stream Begin", "Stream Begin/CLAS completion / expiry/Expire retired CLAS",
                 "Stream Begin/Joint cold page reclaim/Credit pending frees",
                 "Stream Begin/Joint cold page reclaim/Prepare cold candidates/Scan resident candidates",
                 "Stream Begin/Joint cold page reclaim/Prepare cold candidates/Sort cold candidates",
                 "Stream Begin/GPU request feedback/Consume requests/Update resident demand"]
        labels = ["Stream Begin total", "Expire retired CLAS", "Credit pending frees",
                  "Scan cold candidates", "Sort cold candidates", "Update resident demand"]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for offset, field, color, label in [(-.19, "beforeMs", "#8495a7", "Before"),
                                           (.19, "afterMs", "#21a3a3", "After")]:
            entries = [comparison["forward_2"][name][field] for name in names]
            ax.barh([i + offset for i in range(len(names))],
                    [statistics.mean(v) for v in entries], height=.36, color=color, label=label)
            for i, values in enumerate(entries):
                ax.plot(values, [i + offset] * 2, "|", color="#253545", markersize=8)
                mean = statistics.mean(values)
                ax.annotate(f"{mean:.3f}" if mean >= .001 else "<0.001", (max(values), i + offset),
                            xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
        ax.set_yticks(range(len(names)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("CPU elapsed milliseconds / frame (300-frame phase; mean of two runs)")
        ax.set_title("MiniZorah: Stream Begin optimization — forward_2")
        ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
        ax.legend()
        ax.text(0, -.2, "Bars: two-run mean; ticks: run means. Nested scopes overlap; do not sum rows.",
                transform=ax.transAxes, fontsize=9)
        fig.savefig(args.output / "CpuComparison.png", dpi=160)


if __name__ == "__main__":
    main()
