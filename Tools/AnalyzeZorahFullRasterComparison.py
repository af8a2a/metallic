"""Analyze frozen-state raster A/B; diagnostics are outside measured frame files."""
import array
import json
import math
import statistics
import sys
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def distribution(values):
    values = sorted(values)
    def percentile(q):
        x = (len(values) - 1) * q
        lo = int(x)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99), "max": max(values)}


def words(path, kind="I"):
    result = array.array(kind)
    result.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        result.byteswap()
    return result


def image_difference(directory, ref, other):
    ids0 = words(directory / ref["VBuffer.visibility"]["file"])
    ids1 = words(directory / other["VBuffer.visibility"]["file"])
    z0 = words(directory / ref["VBuffer.depth"]["file"], "f")
    z1 = words(directory / other["VBuffer.depth"]["file"], "f")
    u0 = words(directory / ref["VBuffer.depth"]["file"])
    u1 = words(directory / other["VBuffer.depth"]["file"])
    assert len(ids0) == len(ids1) == len(z0) == len(z1)
    covered = sum(x != 0 for x in ids0)
    coverage = sum(bool(x) != bool(y) for x, y in zip(ids0, ids1))
    differing = sum(x != y for x, y in zip(ids0, ids1))
    errors = [abs(a - b) for a, b, x, y in zip(z0, z1, ids0, ids1) if x and y]
    assert all(math.isfinite(v) for v in errors)
    ulps = [abs(a - b) for a, b, x, y in zip(u0, u1, ids0, ids1) if x and y]
    return {"pixels": len(ids0), "referenceCovered": covered, "coverageDifferentPixels": coverage,
            "visibilityDifferentPixels": differing, "visibilityDifferentPercent": differing / len(ids0) * 100,
            "coveredDepthMaxAbs": max(errors, default=0), "coveredDepthMeanAbs": statistics.mean(errors) if errors else 0,
            "coveredDepthMaxUlp": max(ulps, default=0), "coveredDepthOver8UlpPixels": sum(x > 8 for x in ulps)}


def analyze(directory):
    capture = load(directory / "Capture.json")
    assert capture["status"] == "capture_complete", capture.get("error")
    assert capture["protocol"] == "zorah-full-raster-comparison-v1"
    base = capture["cases"][0]["before"]
    reference = next(c["after"] for c in capture["cases"] if c["fullHardware"])
    results = []
    pooled = {}
    residency = None
    mode_snapshots = {}
    for case in capture["cases"]:
        for stamp in (case["before"], case["after"]):
            assert stamp["cutHash"] == base["cutHash"]
            assert stamp["pageMappingsHash"] == base["pageMappingsHash"]
        for resource in ("VBuffer.visibility", "VBuffer.depth"):
            assert case["before"][resource]["hash"] == case["after"][resource]["hash"], "Unstable image within fixed mode"
            key = (case.get("variant",str(case["maxPixels"])), resource)
            if key in mode_snapshots:
                assert mode_snapshots[key] == case["after"][resource]["hash"], "Image changed across rounds"
            mode_snapshots[key] = case["after"][resource]["hash"]
        rows = load(directory / case["framesFile"])
        assert len(rows) == case["frames"]
        values = {k: [] for k in ["rasterTotal", "classification", "software", "hardware", "cull", "candidates", "bins", "merge", "graphGpu", "editorLoop"]}
        leaves = {"classification": "Soft/hard classification", "software": "Software raster", "hardware": "Hardware raster",
                  "cull": "Cluster cull", "candidates": "Candidates", "bins": "Stable bins", "merge": "Raster merge"}
        for row in rows:
            scopes = row["scopes"]
            totals = [s for s in scopes if s["path"].endswith(("/Stream early", "/Stream late"))]
            assert len(totals) == 2 and all(s["gpuMs"] is not None for s in totals)
            values["rasterTotal"].append(sum(s["gpuMs"] for s in totals))
            for key, leaf in leaves.items():
                selected = [s for s in scopes if ("/Stream early/" in s["path"] or "/Stream late/" in s["path"]) and s["path"].endswith("/" + leaf)]
                assert all(s["gpuMs"] is not None for s in selected)
                values[key].append(sum(s["gpuMs"] for s in selected))
                if case["fullHardware"] and key in ("classification", "software", "merge"):
                    assert not selected, "Full HW is still executing a hybrid stage"
            graph = [s["gpuMs"] for s in scopes if s["path"].endswith("/RenderGraph GPU envelope")]
            assert len(graph) == 1 and graph[0] is not None
            values["graphGpu"].append(graph[0])
            values["editorLoop"].append(row["editorLoopMs"])
            assert len(row["streaming"]) == 1
            if residency is None:
                residency = row["streaming"]
            assert row["streaming"] == residency, "Geometry/CLAS/texture residency changed"
        key = case.get("variant",str(case["maxPixels"]))
        if key not in pooled:
            pooled[key] = {k: [] for k in values}
        for k, v in values.items():
            pooled[key][k].extend(v)
        results.append({"name": case["name"], "maxPixels": case["maxPixels"], "variant": key,
                        "classificationCounts": {k: v for k,v in case["after"].items() if k.endswith("ClusterCull")},
                        "timingsMs": {k: distribution(v) for k, v in values.items()},
                        "earlyBins": case["after"]["AfterStreamEarlyBins"], "lateBins": case["after"]["AfterStreamLateBins"],
                        "vsFullHardware": image_difference(directory, reference, case["after"]),
                        "beforeAfter": image_difference(directory, case["before"], case["after"])})
    if capture["config"].get("metadataComparison"):
        for resource in ("VBuffer.visibility", "VBuffer.depth"):
            assert mode_snapshots[("exact8",resource)] == mode_snapshots[("fast8",resource)], "Metadata path changed raster output"
        exact = next(r for r in results if r["variant"] == "exact8")
        for r in results:
            if r["variant"] == "fast8":
                assert r["earlyBins"] == exact["earlyBins"] and r["lateBins"] == exact["lateBins"], "Metadata classification changed bins"
            if r["variant"] in ("exact8", "fast8"):
                for phase, bin_key in (("AfterStreamEarlyClusterCull", "earlyBins"), ("AfterStreamLateClusterCull", "lateBins")):
                    counts = r["classificationCounts"][phase]
                    bins = r[bin_key]
                    assert counts["candidateOverflow"] == 0
                    assert sum(counts[k] for k in ("exactClusters", "fastSoftware", "fastHardware")) == bins["hardwareClusters"] + bins["softwareClusters"]
                    if r["variant"] == "exact8":
                        assert counts["fastSoftware"] == counts["fastHardware"] == 0
    if capture["config"].get("swComparison"):
        legacy = next(c["after"] for c in capture["cases"] if c["variant"] == "swLegacy")
        for r,c in zip(results,capture["cases"]):
            r["vsLegacySoftware"] = image_difference(directory,legacy,c["after"])
            if r["variant"] in ("swPrepared", "swCooperative"):
                for resource in ("VBuffer.visibility", "VBuffer.depth"):
                    assert c["after"][resource]["hash"] == legacy[resource]["hash"], "Exact SW changed reference image"
                for phase in ("AfterStreamEarlyBins", "AfterStreamLateBins"):
                    assert c["after"][phase] == legacy[phase], "Exact SW changed bins"
    summary = {"protocol": capture["protocol"], "outputExtent": capture["outputExtent"], "renderExtent": capture["renderExtent"],
               "cutHash": base["cutHash"], "pageMappingsHash": base["pageMappingsHash"], "activeGroups": base["activeGroups"],
               "residentState": residency, "sameCutAndResidency": True, "sameModeImagesStable": True, "cases": results,
               "pooled": {mode: {k: distribution(v) for k, v in values.items()} for mode, values in pooled.items()}}
    (directory / "Summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen Full raster comparison", "", f"Output {capture['outputExtent']}, render {capture['renderExtent']}; fixed camera, jitter disabled, serialized HW/SW.",
             "Diagnostic readbacks and recovery frames excluded. Frozen traversal/streaming/TLAS: graph/editor time is not live-roaming FPS.",
             f"Cut `{base['cutHash']}`, page mappings `{base['pageMappingsHash']}`, {base['activeGroups']} active groups. Residency identical in measured samples.", "",
             "| Mode | Samples | Raster mean ms | Raster p50 | Raster p95 | Classify mean | SW mean | HW mean | Graph mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, stats in sorted(summary["pooled"].items(), key=lambda x: {"exact8": 8,"fast8": 9}.get(x[0],int(x[0]) if x[0].isdigit() else 10)):
        label = {"0":"Full HW","exact8":"8 px exact","fast8":"8 px metadata","swLegacy":"8 px legacy SW","swPrepared":"8 px prepared SW","swPlane":"8 px depth plane","swCooperative":"8 px cooperative load"}.get(mode,mode + " px")
        t = stats["rasterTotal"]
        lines.append(f"| {label} | {t['count']} | {t['mean']:.3f} | {t['p50']:.3f} | {t['p95']:.3f} | {stats['classification']['mean']:.3f} | {stats['software']['mean']:.3f} | {stats['hardware']['mean']:.3f} | {stats['graphGpu']['mean']:.3f} |")
    lines += ["", "| Case | Early HW / SW | Coverage different vs HW | Visibility different % | Max depth abs | Depth >8 ULP |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        b, d = r["earlyBins"], r["vsFullHardware"]
        lines.append(f"| {r['name']} | {b['hardwareClusters']} / {b['softwareClusters']} | {d['coverageDifferentPixels']} | {d['visibilityDifferentPercent']:.5f} | {d['coveredDepthMaxAbs']:.7g} | {d['coveredDepthOver8UlpPixels']} |")
    if capture["config"].get("swComparison"):
        lines += ["", "Prepared SW is required to match legacy depth/visibility and bins exactly.", "",
                  "| Case | Coverage different vs legacy SW | Visibility different | Max depth ULP |",
                  "|---|---:|---:|---:|"]
        for r in results:
            if r["variant"] == "0":
                continue
            d = r["vsLegacySoftware"]
            lines.append(f"| {r['name']} | {d['coverageDifferentPixels']} | {d['visibilityDifferentPixels']} | {d['coveredDepthMaxUlp']} |")
    (directory / "Summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    return summary


if __name__ == "__main__":
    target = Path(sys.argv[1])
    if (target / "Capture.json").exists():
        analyze(target)
    else:
        runs = sorted(target.glob("run*/Summary.json"))
        assert runs, "No completed raster comparisons"
        # Independent processes may have different settled cuts. Never pool them.
        (target / "Comparison.json").write_text(json.dumps({"runs": [str(p) for p in runs],
            "note": "Each run compares variants within its own frozen cut; independent runs are not pooled."}, indent=2) + "\n", encoding="utf-8")
