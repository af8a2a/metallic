"""Compare complete repetitions of the repository MiniZorah camera replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def gpu_competition(directory):
    path = directory / "GpuProcesses.csv"
    if not path.exists():
        return {"available": False}
    samples = {}
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["process"] == "MetallicRhiTests" or not row["engine"].endswith("engtype_3d"):
                continue
            samples.setdefault(row["process"], []).append(float(row["utilization"]))
    return {"available": True, "other3dEngineProcesses": {
        process: distribution(values) for process, values in samples.items()},
        "note": "Per-engine activity samples, not whole-card ownership; absent/idle intervals are not emitted."}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"samples": 0}
    return {"samples": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values), "p95": values[int((len(values) - 1) * .95)],
            "p99": values[int((len(values) - 1) * .99)], "max": values[-1]}


def read_runs(root, cases):
    manifest = json.loads((root / "Manifest.json").read_text(encoding="utf-8-sig"))
    runs = {}
    for case in cases:
        directory = root / case
        report = json.loads((directory / "Baseline.json").read_text(encoding="utf-8-sig"))
        if report["status"] != "passed":
            raise ValueError(f"{directory}: failed report: {report.get('error')}")
        frames = [json.loads(line) for line in (directory / "Frames.jsonl").read_text().splitlines()]
        if len(frames) != report["frameCount"] or report.get("qualityRun"):
            raise ValueError(f"{directory}: timing repetitions must be complete, without diagnostic readbacks")
        phases = {}
        for phase in dict.fromkeys(frame["phase"] for frame in frames):
            samples = [frame for frame in frames if frame["phase"] == phase]
            phases[phase] = {
                key: distribution([frame[key] for frame in samples])
                for key in ("hostFrameMs", "gpuMs", "cpuRecordMs")}
            phases[phase]["priorFrameDrainFrames"] = sum("graph.priorFrameDrain" in f.get("cpuPhases", {}) for f in samples)
            phases[phase]["slotWaitMs"] = distribution([f.get("cpuPhases", {}).get("graph.slotWait", 0) for f in samples])
            phases[phase]["submitWaitMs"] = distribution([f.get("cpuPhases", {}).get("graph.submitWait", 0) for f in samples])
            phases[phase]["recordMs"] = distribution([f.get("cpuPhases", {}).get("graph.record", 0) for f in samples])
            phases[phase]["nodesGpuMs"] = {
                node: distribution([n["gpuMs"] for f in samples for n in f["nodes"] if n["name"] == node])
                for node in dict.fromkeys(n["name"] for f in samples for n in f["nodes"])}
        stable = [f for f in frames if f["phase"] != "warmup"]
        runs[case] = {"phases": phases, "steady": {key: distribution([f[key] for f in stable])
            for key in ("hostFrameMs", "gpuMs", "cpuRecordMs")},
            "streamEnd": frames[-1]["stream"], "finalQuality": report["quality"][-1],
            "runWallSeconds": report["runWallSeconds"], "process": json.loads((directory / "Process.json").read_text(encoding="utf-8-sig")),
            "gpuCompetition": gpu_competition(directory),
            "gpuFramesInFlight": report.get("gpuFramesInFlight", 2),
            "configuration": {key: report.get(key) for key in
                ("protocol", "realtime", "frameCount", "resolution", "lodPixelError", "lighting", "asset", "validation")}}
    return {"root": str(root.resolve()), "manifest": manifest, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["m1", "m2"])
    parser.add_argument("--quality", type=Path, help="Separate, untimed quality run directory")
    args = parser.parse_args()
    before, after = (read_runs(root, args.cases) for root in (args.before, args.after))
    if before["manifest"]["replaySha256"] != after["manifest"]["replaySha256"]:
        raise ValueError("Camera replay hashes differ")
    for case in args.cases:
        if before["runs"][case]["configuration"] != after["runs"][case]["configuration"]:
            raise ValueError(f"{case}: render configuration differs")
    comparison = {}
    for phase in before["runs"][args.cases[0]]["phases"]:
        comparison[phase] = {}
        for metric in ("hostFrameMs", "gpuMs"):
            b = [before["runs"][case]["phases"][phase][metric]["p50"] for case in args.cases]
            a = [after["runs"][case]["phases"][phase][metric]["p50"] for case in args.cases]
            comparison[phase][metric] = {"beforeP50": b, "afterP50": a,
                "meanOfMediansChangePercent": (statistics.mean(a) / statistics.mean(b) - 1) * 100}
    result = {"protocol": "metallic-cfg-roam-comparison-v1", "before": before, "after": after,
              "comparison": comparison, "notes": [
                  "Warmup is retained as a separate phase; steady aggregates exclude only warmup.",
                  "All frames from every requested repetition are retained; no outlier removal.",
                  "Frame-step replay is unpaced. Host time measures throughput, not an interactive FPS cap.",
                  "GPU execution remains ordered across frames; CPU recording may overlap GPU execution.",
                  "Timing deltas do not establish an isolated speedup when other GPU engines are active or competition was not measured."]}
    competitors = [run["gpuCompetition"] for batch in (before, after) for run in batch["runs"].values()]
    result["gpuCompetitionCoverageComplete"] = all(item["available"] for item in competitors)
    result["gpuCompetitionObserved"] = any(
        values["max"] > 10 for item in competitors for values in item.get("other3dEngineProcesses", {}).values())
    result["timingInterpretation"] = (
        "Concurrent GPU activity detected; deltas are observations, not isolated implementation speedups or regressions."
        if result["gpuCompetitionObserved"] else
        "GPU competition was not measured for every repetition; an isolated performance conclusion needs controlled reruns."
        if not result["gpuCompetitionCoverageComplete"] else
        "No other 3D engine process exceeded 10% in the recorded samples; inspect clocks, budgets and render configuration before attributing deltas.")
    if args.quality:
        quality = json.loads((args.quality / "Baseline.json").read_text(encoding="utf-8-sig"))
        if quality["status"] != "passed" or not quality.get("qualityRun"):
            raise ValueError("Expected a completed separate quality run")
        quality_manifest = json.loads((args.quality.parent / "Manifest.json").read_text(encoding="utf-8-sig"))
        if quality_manifest["replaySha256"] != after["manifest"]["replaySha256"]:
            raise ValueError("Quality camera replay differs from timing runs")
        configuration = after["runs"][args.cases[0]]["configuration"]
        if any(quality.get(key) != configuration[key] for key in configuration if key != "validation"):
            raise ValueError("Quality render configuration differs from timing runs")
        result["qualityRun"] = {"root": str(args.quality.resolve()), "status": quality["status"],
            "frameCount": quality["frameCount"], "validation": quality["validation"],
            "manifest": quality_manifest,
            "process": json.loads((args.quality / "Process.json").read_text(encoding="utf-8-sig")),
            "checkpoints": quality["quality"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["timingInterpretation"])
    for phase, metrics in comparison.items():
        host = metrics["hostFrameMs"]
        print(f"{phase}: host P50 {host['beforeP50']} -> {host['afterP50']} ({host['meanOfMediansChangePercent']:+.1f}%)")


if __name__ == "__main__":
    main()
