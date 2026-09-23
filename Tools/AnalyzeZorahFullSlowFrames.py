"""Attribute Full replay slow frames using existing CPU/GPU scopes, without extra GPU work.

Scope costs are inclusive. GPU timings are
associated with their original execution; they are not added to CPU wall time.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    percentile = lambda q: ordered[max(0, math.ceil(len(ordered) * q) - 1)]
    return dict(count=len(values), mean=sum(values)/len(values), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), max=ordered[-1])


def analyze(directory):
    capture = load(directory / "Capture.json")
    assert capture["status"] == "capture_complete" and not capture.get("diagnosticRun", False)
    rows = [json.loads(line) for line in (directory / "Frames.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    assert len(rows) == capture["frames"] and capture["missingGpuFrames"] == 0
    paths = {s["id"]: s["path"] for s in capture["scopes"]}
    children = defaultdict(list)
    for ident, path in paths.items():
        parents = [i for i, p in paths.items() if path.startswith(p + "/")]
        if parents:
            children[max(parents, key=lambda i: len(paths[i]))].append(ident)
    samples = defaultdict(lambda: defaultdict(list))
    stages = defaultdict(list)
    slowest = []
    for row in rows:
        slow = row["frameMs"] > 1000/30
        stages[row["stage"]].append(row["frameMs"])
        cpu, gpu = defaultdict(float), defaultdict(float)
        for scope in row["scopes"]:
            cpu[scope["id"]] += scope["cpuMs"]
            if scope["gpuMs"] is not None:
                gpu[scope["id"]] += scope["gpuMs"]
        for ident in paths:
            for group in ("all", "slow" if slow else "withinBudget"):
                samples[ident][group + "Cpu"].append(cpu.get(ident, 0))
                if ident in gpu:
                    samples[ident][group + "Gpu"].append(gpu[ident])
        slowest.append(dict(frame=row["frame"], seconds=row["seconds"], stage=row["stage"], frameMs=row["frameMs"],
                            availableBytes=row["availableBytes"], streaming=row["streaming"],
                            cpu={paths[i]: t for i, t in cpu.items() if t >= 1},
                            gpu={paths[i]: t for i, t in gpu.items() if t >= 1}))
    scopes = [dict(path=paths[i], leaf=not children[i], **{k:stats(v) for k,v in data.items()}) for i,data in samples.items()]
    return dict(run=directory.name, frames=stats([r["frameMs"] for r in rows]),
                slowFrames=sum(r["frameMs"] > 1000/30 for r in rows),
                stages={k:dict(frameMs=stats(v), slowFrames=sum(t > 1000/30 for t in v)) for k,v in stages.items()},
                scopes=scopes,
                drainReasons=dict(Counter(str(r["graphPreparation"]["drainReasonMask"]) for r in rows)),
                externalCompletions=dict(Counter(str(r["graphPreparation"]["externalCompletionCount"]) for r in rows)),
                hzbValidFrames=sum(all(s["softwareRaster"]["paramsHzbValid"] for s in r["streaming"]) for r in rows),
                maxCounters={k:max(s[k] for r in rows for s in r["streaming"]) for k in
                           ("allocationFailures", "loadFailures", "requestOverflows", "blasOverflowCount")},
                streaming={k:stats([s[k] for r in rows for s in r["streaming"]]) for k in
                           ("requests", "allocationFailures", "uploads", "evictions", "geometryBytes", "geometryCapacity",
                            "clasBytes", "clasCapacity", "blasOverflowCount")},
                slowest=sorted(slowest,key=lambda r:r["frameMs"],reverse=True)[:12])


def main(root):
    directories = [root] if (root/"Capture.json").exists() else sorted(p for p in root.glob("run*") if (p/"Capture.json").exists())
    if not directories:
        raise ValueError("No complete captures")
    condition_keys = ("config", "absoluteKeyframes", "outputExtent", "renderExtent", "graph", "hidden", "vsync", "validationRequested")
    conditions = [{k:load(p/"Capture.json")[k] for k in condition_keys} for p in directories]
    assert all(c == conditions[0] for c in conditions), "Replay conditions differ"
    results = [analyze(p) for p in directories]
    (root/"SlowFrames.json").write_text(json.dumps(dict(conditions=conditions[0], runs=results),indent=2)+"\n",encoding="utf-8")
    lines = ["# Full slow frame attribution", "", "Inclusive GPU intervals are not summed with CPU wall time. Missing CPU scopes contribute zero per frame; GPU means cover recorded samples only.", ""]
    for r in results:
        lines += [f"## {r['run']}", "", f"Frames {r['frames']['count']}; mean {r['frames']['mean']:.3f} ms; P95 {r['frames']['p95']:.3f} ms; over 33.33 ms: {r['slowFrames']}.", "",
                  "| CPU leaf scope (inclusive) | All mean | Within budget | Slow mean | P95 |", "|---|---:|---:|---:|---:|"]
        for s in sorted((s for s in r['scopes'] if s['leaf']),key=lambda s:s['slowCpu']['mean'] if s['slowCpu'] else 0,reverse=True)[:16]:
            fmt=lambda k: f"{s[k]['mean']:.3f}" if s.get(k) else "--"
            lines.append(f"| {s['path']} | {fmt('allCpu')} | {fmt('withinBudgetCpu')} | {fmt('slowCpu')} | {s['allCpu']['p95']:.3f} |")
        lines += ["", "| GPU scope (inclusive) | All mean | Within budget | Slow mean | P95 |", "|---|---:|---:|---:|---:|"]
        for s in sorted((s for s in r['scopes'] if s.get('allGpu')),key=lambda s:s['allGpu']['mean'],reverse=True)[:12]:
            fmt=lambda k: f"{s[k]['mean']:.3f}" if s.get(k) else "--"
            lines.append(f"| {s['path']} | {fmt('allGpu')} | {fmt('withinBudgetGpu')} | {fmt('slowGpu')} | {s['allGpu']['p95']:.3f} |")
        lines.append("")
    (root/"SlowFrames.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps([{k:r[k] for k in ('run','frames','slowFrames','drainReasons','hzbValidFrames','maxCounters')} for r in results]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    main(parser.parse_args().directory)
