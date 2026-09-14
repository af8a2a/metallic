"""Validate and summarize RunMiniZorahBaseline.ps1 output (stdlib; plots optional)."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


PHASES = ("cold_start", "static_warm", "roam_first", "roam_repeat", "settle", "static_return")
PERFORMANCE = ("a1", "b1", "b2", "a2")
QUALITY = ("quality-a", "quality-b")
MIB = 1024 ** 2


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"samples": len(values), "mean": statistics.mean(values),
            **{name: values[int(fraction * (len(values) - 1))]
               for name, fraction in (("p50", .50), ("p95", .95), ("p99", .99))},
            "min": values[0], "max": values[-1]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_case(directory):
    report = read_json(directory / "Baseline.json")
    require(report["status"] == "passed" and report["frameCount"] == 8400, f"{directory.name}: incomplete run")
    require(report["resolution"] == [1920, 1080] and report["lodPixelError"] == 1.5, "View settings differ")
    require(report["qualityRun"] == directory.name.startswith("quality"), "Wrong sampling mode")
    require(report["validation"] == report["qualityRun"], "Performance must exclude validation")
    require(report["clasEnabled"] == (directory.name in ("b1", "b2", "quality-b")), "Wrong CLAS mode")
    cameras = read_json(directory / "Cameras.json")
    require(len(cameras) == 8400 and cameras[600:4200] == cameras[4200:7800], "Route replay mismatch")
    camera_hash = hashlib.sha256((directory / "Cameras.json").read_bytes()).hexdigest()
    graph = json.loads(json.dumps(report["graph"]))
    for node in graph["nodes"]:
        if node["name"] == "GPUDriven":
            require(node["properties"].pop("enableClas") == report["clasEnabled"], "Graph/CLAS mismatch")
    phase_rows = collections.defaultdict(list)
    scopes = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    rows = []
    seen_ids = set()
    raw_hash = hashlib.sha256()
    with (directory / "Frames.jsonl").open("rb") as source:
        for line in source:
            raw_hash.update(line)
            row = json.loads(line)
            require(row["frame"] == len(rows) and row["executionId"] not in seen_ids, "Missing/duplicate frame")
            seen_ids.add(row["executionId"])
            require(all(math.isfinite(row[key]) and row[key] >= 0
                        for key in ("gpuMs", "cpuRecordMs", "cpuExecuteMs", "hostFrameMs")), "Invalid timing")
            expected_phase = PHASES[0 if row["frame"] < 300 else 1 if row["frame"] < 600 else
                                    2 if row["frame"] < 4200 else 3 if row["frame"] < 7800 else
                                    4 if row["frame"] < 8100 else 5]
            require(row["phase"] == expected_phase, "Phase mismatch")
            for node in row.pop("nodes"):
                paths = []
                scopes[row["phase"]][node["name"]]["gpuMs"].append(node["gpuMs"])
                scopes[row["phase"]][node["name"]]["cpuMs"].append(node["cpuMs"])
                for index, scope in enumerate(node["sections"]):
                    parent = scope["parent"]
                    require(parent == 0xffffffff or parent < index, "Invalid scope parent")
                    path = (node["name"] if parent == 0xffffffff else paths[parent]) + "/" + scope["name"]
                    paths.append(path)
                    path += " [" + scope["queue"] + "]"
                    for key in ("gpuMs", "cpuMs"):
                        scopes[row["phase"]][path][key].append(scope[key])
            stream = row["stream"]
            require(stream["geometryBytes"] <= stream["geometryBudgetBytes"] and
                    stream["clasBytes"] <= stream["clasCapacityBytes"] and not stream["allocationFailures"],
                    "Budget or allocation invariant failed")
            require(not rows or stream["totalUploadBytes"] >= rows[-1]["stream"]["totalUploadBytes"],
                    "Upload counter regressed")
            rows.append(row)
            phase_rows[row["phase"]].append(row)
    require(len(rows) == 8400 and len(seen_ids) == 8400, "Timing frame count mismatch")
    summary = {"clasEnabled": report["clasEnabled"], "qualityRun": report["qualityRun"],
               "cameraSha256": camera_hash, "framesSha256": raw_hash.hexdigest(),
               "compileMs": report["compileMs"], "deviceCreateMs": report["deviceCreateMs"],
               "runWallSeconds": report["runWallSeconds"], "overlappingFrames": report["overlappingFrames"],
               "memoryAfterReplay": report["memoryAfterReplay"], "phases": {}, "scopes": {},
               "quality": [{"frame": q["frame"], "phase": q["phase"], "cut": q.get("cut"),
                            "terminalReady": q["stream"]["terminalReady"]} for q in report["quality"]],
               "latency": report["finalStream"]["latency"], "finalStreamStats": report["finalStream"]["stats"]}
    for phase, group in phase_rows.items():
        summary["phases"][phase] = {key: distribution([row[key] for row in group])
                                    for key in ("gpuMs", "cpuRecordMs", "cpuExecuteMs", "hostFrameMs")}
        summary["phases"][phase]["stream"] = {
            key: distribution([row["stream"][key] for row in group]) for key in rows[0]["stream"]}
        summary["phases"][phase]["evictions"] = sum(row["stream"]["evictions"] for row in group)
        summary["phases"][phase]["uploadedMiB"] = sum(row["stream"]["uploadBytes"] for row in group) / MIB
        summary["scopes"][phase] = {path: {key: distribution(values) for key, values in timing.items()}
                                    for path, timing in scopes[phase].items()}
    memory_keys = ("geometryBytes", "clasBytes", "clasEncodedBytes", "clasCapacityBytes", "clasScratchBytes")
    summary["memoryMiB"] = {key: {"peak": max(row["stream"][key] for row in rows) / MIB,
                                       "final": rows[-1]["stream"][key] / MIB} for key in memory_keys}
    summary["streamTotals"] = {"uploadedMiB": rows[-1]["stream"]["totalUploadBytes"] / MIB,
        **{key: sum(row["stream"][key] for row in rows) for key in ("evictions", "clasBuilt", "clasMoved")},
        "maxClasPending": max(row["stream"]["clasPending"] for row in rows),
        "framesWithClasDeferred": sum(row["stream"]["clasDeferred"] > 0 for row in rows),
        "final": rows[-1]["stream"]}
    # Includes device/graph setup and final diagnostics; it is machine telemetry,
    # not a timestamp-aligned attribution of GPU utilization to this renderer.
    with (directory / "GpuDuring.csv").open(encoding="utf-8-sig", newline="") as source:
        telemetry = [{key.strip(): value.strip() for key, value in row.items()}
                     for row in csv.DictReader(source)]
    require(bool(telemetry), "GPU telemetry missing")
    summary["gpuTelemetry"] = {"gpu": telemetry[0]["name"], "driver": telemetry[0]["driver_version"],
        "scope": "Whole test process; device-wide counters include other applications",
        "distributions": {key: distribution([float(row[key].split()[0]) for row in telemetry])
            for key in telemetry[0] if key not in ("timestamp", "name", "driver_version")}}
    stdout = (directory / "stdout.log").read_text(encoding="utf-8-sig")
    stderr = (directory / "stderr.log").read_text(encoding="utf-8-sig")
    require("[  PASSED  ] 1 test." in stdout, "Google Test result missing")
    require(not any(token in stdout + stderr for token in ("VUID-", "DeviceLost", "Validation Error", "[error]")),
            "Review error/validation diagnostics before accepting baseline")
    summary["diagnosticsChecked"] = True
    summary["longestHostFrames"] = sorted(rows, key=lambda row: row["hostFrameMs"], reverse=True)[:10]
    require(summary["quality"][-1]["cut"]["visibleOverTargetRefinements"] == 0, "Final quality not converged")
    return summary, rows, graph


def make_plots(root, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"a1": "#159b88", "a2": "#86c9b6", "b1": "#366bd6", "b2": "#99b5f0"}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for case in PERFORMANCE:
        rows = cases[case]
        for axis, key, label in ((axes[0], "gpuMs", "GPU frame envelope (ms)"),
                                 (axes[1], "geometryBytes", "Geometry live allocation (MiB)"),
                                 (axes[2], "clasBytes", "CLAS live allocation (MiB)")):
            axis.set_ylabel(label)
            # Plot 60-frame medians; exact unaggregated tails remain in Frames.jsonl.
            points = []
            for start in range(0, len(rows), 60):
                values = [(r[key] if key == "gpuMs" else r["stream"][key] / MIB)
                          for r in rows[start:start + 60]]
                points.append(statistics.median(values))
            axis.plot(range(30, len(rows), 60), points, label=case.upper(), color=colors[case], linewidth=1.4)
    for axis in axes:
        for frame in (600, 4200, 7800):
            axis.axvline(frame, color="#888888", linewidth=.7, linestyle="--")
        axis.grid(alpha=.2)
    axes[0].legend(ncol=4)
    axes[0].set_title("MiniZorah fixed replay | 1920 x 1080, 1.5 px | A: VBuffer / B: VBuffer + CLAS\n60-frame medians; unpaced replay, performance runs only")
    axes[-1].set_xlabel("Replay frame: hold 0-599 / route 600-4199 / repeated route 4200-7799 / return 7800-8399")
    fig.savefig(root / "BaselineTimeline.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = {"protocol": "minizorah-fixed-v1", "manifest": read_json(args.root / "Manifest.json"), "cases": {}}
    cases = {}
    reference_graph = reference_hash = None
    for case in (*PERFORMANCE, *QUALITY):
        summary, rows, graph = summarize_case(args.root / case)
        if reference_graph is None:
            reference_graph, reference_hash = graph, summary["cameraSha256"]
        require(graph == reference_graph and summary["cameraSha256"] == reference_hash,
                f"{case}: A/B conditions differ beyond enableClas")
        result["cases"][case] = summary
        if case in PERFORMANCE:
            cases[case] = rows
        print(f"{case}: 8400 frames verified, GPU roam p50={summary['phases']['roam_first']['gpuMs']['p50']:.3f} ms")
    result["comparison"] = {}
    for phase in PHASES:
        result["comparison"][phase] = {}
        for metric in ("gpuMs", "cpuExecuteMs", "hostFrameMs"):
            a = [row[metric] for case in ("a1", "a2") for row in cases[case] if row["phase"] == phase]
            b = [row[metric] for case in ("b1", "b2") for row in cases[case] if row["phase"] == phase]
            result["comparison"][phase][metric] = {"a": distribution(a), "b": distribution(b),
                "medianDeltaMs": statistics.median(b) - statistics.median(a)}
    (args.root / "Summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.plots:
        make_plots(args.root, cases)
    print(f"Wrote {args.root / 'Summary.json'}; all camera hashes and normalized graph configurations match.")


if __name__ == "__main__":
    main()
