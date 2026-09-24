"""Compare deterministic MiniZorah P0/P1 roam profiles, excluding diagnostic runs."""
import argparse
import copy
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def distribution(values):
    values = sorted(values)
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Missing or invalid GPU timings")
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1]}


def conditions(capture):
    result = copy.deepcopy({key: capture[key] for key in (
        "sample", "config", "absoluteKeyframes", "outputExtent", "renderExtent", "graph", "hidden")})
    result["config"].pop("cullHardwareClassification")
    for node in result["graph"]:
        node["properties"].pop("cullHardwareClassification", None)
    return result


def analyze(root):
    manifest = read(root / "Manifest.json")
    if [r["variant"] for r in manifest["runs"] if not r["diagnostic"]] != manifest["sequence"]:
        raise ValueError("The planned A/B sequence is incomplete")
    results, diagnostics = [], []
    baseline_conditions = None
    baseline_trajectory = None
    for item in manifest["runs"]:
        folder = root / item["directory"]
        capture = read(folder / "Capture.json")
        if capture["status"] != "capture_complete" or capture["missingGpuFrames"]:
            raise ValueError(f"Incomplete capture: {folder}")
        if capture["sample"] != "gpu-driven-sample" or capture["trajectoryClock"] != "deterministic frame index":
            raise ValueError("Expected deterministic MiniZorah sample")
        if item["diagnostic"]:
            culls = [r for r in capture.get("workloads", []) if r.get("cullHeader")]
            bins = [r for r in capture.get("workloads", []) if r.get("binHeader")]
            if not culls or not bins or any(r["counts"]["candidateOverflow"] for r in culls):
                raise ValueError("Missing diagnostic cull/bin counters or overflow")
            diagnostics.append({"directory": item["directory"], "variant": item["variant"],
                                "culls": culls, "bins": bins})
            continue
        if capture.get("diagnosticRun"):
            raise ValueError("Diagnostic frames must not enter the performance sample")
        current = conditions(capture)
        if baseline_conditions is None:
            baseline_conditions = current
        elif current != baseline_conditions:
            raise ValueError("A/B capture conditions differ beyond the classifier selection")
        rows = [json.loads(line) for line in (folder / "Frames.jsonl").read_text().splitlines()]
        if len(rows) != capture["frames"] or len(rows) != capture["config"]["routeFrames"]:
            raise ValueError("Incomplete deterministic route")
        trajectory = [(r["routeSample"], r["seconds"], r["stage"]) for r in rows]
        if baseline_trajectory is None:
            baseline_trajectory = trajectory
        elif trajectory != baseline_trajectory:
            raise ValueError("A/B camera sample indices or times differ")
        definitions = {s["id"]: s["path"] for s in capture["scopes"]}
        metrics = {key: [] for key in ("classify", "cull", "stable_bins", "cull_classify_bins", "vbuffer", "graph", "editor_loop")}
        per_stage, residency = {}, []
        for row in rows:
            gpu = {definitions[s["id"]]: s["gpuMs"] for s in row["scopes"] if s["gpuMs"] is not None}
            def required_sum(predicate):
                values = [v for path, v in gpu.items() if predicate(path)]
                if not values:
                    raise ValueError(f"Missing requested GPU scope in {folder}: {list(gpu)}")
                return sum(values)
            def stream_section(name):
                return required_sum(lambda p: ("/Stream early/" in p or "/Stream late/" in p) and p.endswith("/" + name))
            frame = {"classify": stream_section("Soft/hard classification"), "cull": stream_section("Cluster cull"),
                     "stable_bins": stream_section("Stable bins"),
                     "vbuffer": required_sum(lambda p: p.endswith("/VBuffer (VisibilityBufferPass)")),
                     "graph": required_sum(lambda p: p.endswith("/RenderGraph GPU envelope")),
                     "editor_loop": row["frameMs"]}
            # These are sibling stages; do not add their parent VBuffer duration.
            frame["cull_classify_bins"] = frame["classify"] + frame["cull"] + frame["stable_bins"]
            stage = per_stage.setdefault(row["stage"], {key: [] for key in metrics})
            for key, value in frame.items():
                metrics[key].append(value)
                stage[key].append(value)
            for stream in row["streaming"]:
                identity = stream.get("softwareRaster")
                if identity and identity["cullHardwareClassification"] != (item["variant"] == "p1"):
                    raise ValueError("Recorded shader selection differs from requested A/B variant")
                residency.append({key: stream[key] for key in ("geometryBytes", "residentPages", "pendingPages", "requests", "uploads")})
        results.append({"directory": item["directory"], "variant": item["variant"], "frames": len(rows),
                        "metrics_ms": {key: distribution(values) for key, values in metrics.items()},
                        "stages_ms": {stage: {key: distribution(values) for key, values in m.items()} for stage, m in per_stage.items()},
                        "residency": {key: distribution([r[key] for r in residency]) for key in residency[0]} if residency else {}})
    summary = {}
    for key in results[0]["metrics_ms"]:
        a = [r["metrics_ms"][key]["mean"] for r in results if r["variant"] == "p0"]
        b = [r["metrics_ms"][key]["mean"] for r in results if r["variant"] == "p1"]
        old, new = statistics.mean(a), statistics.mean(b)
        summary[key] = {"p0_mean_ms": old, "p1_mean_ms": new, "saved_ms": old - new,
                        "reduction_percent": 100 * (1 - new / old), "p0_run_means": a, "p1_run_means": b,
                        "run_mean_ranges_overlap": max(min(a), min(b)) <= min(max(a), max(b))}
    counter_comparison = None
    if len(diagnostics) == 2:
        a, b = sorted(diagnostics, key=lambda item: item["variant"])
        counter_comparison = {}
        for kind, payload in (("culls", "counts"), ("bins", "bins")):
            left, right = a[kind], b[kind]
            if len(left) != len(right) or any(x["camera"] != y["camera"] or x["phase"] != y["phase"] for x, y in zip(left, right)):
                raise ValueError("Diagnostic camera samples differ")
            counter_comparison[kind] = {
                "samples": len(left), "exactly_matching_samples": sum(x[payload] == y[payload] for x, y in zip(left, right)),
                "p0_totals": {key: sum(r[payload][key] for r in left) for key in left[0][payload]},
                "p1_totals": {key: sum(r[payload][key] for r in right) for key in right[0][payload]}}
    result = {"protocol": manifest["protocol"], "conditions": baseline_conditions,
              "summary": summary, "runs": results, "diagnostics": diagnostics,
              "diagnostic_comparison": counter_comparison,
              "limits": "Live streaming; equal camera trajectory does not imply identical cuts/residency. Compare run spread and diagnostic workload counts; editor loop is affected by present/Reflex."}
    (root / "Summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    analyze(parser.parse_args().capture)
