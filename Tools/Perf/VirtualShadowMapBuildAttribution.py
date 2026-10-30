"""Verify and plot split-build attribution; never classify optimization acceptance.

python -B Tools/Perf/VirtualShadowMapBuildAttribution.py SOURCE NEW_REPORT
SOURCE holds a declared two-block ABBA Plan.json and sealed Acceptance packages.
"""
from __future__ import annotations

import argparse
import copy
import datetime
from pathlib import Path
import platform
import re
import statistics
import sys

import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapBitset as bitset
import VirtualShadowMapGPUProfile as profile
import VirtualShadowMapOptimization as optimization


PROTOCOL = "virtual-shadow-map-build-attribution-v1"
METRICS = (*profile.BUILD_SUBSTAGES.values(), "buildUnattributedMs")


def schedule():
    return [{"index": index, "block": index // 4, "arm": arm, "package": f"process-{index + 1:02d}-{arm}"}
            for index, arm in enumerate(("A", "B", "B", "A") * 2)]


def normalized_workload(spec, graph, arm):
    if spec.get("mode") != "timing":
        raise ValueError("Attribution requires timing kernels without workload instrumentation")
    case = spec["cases"][0]
    if case.get("frames", spec.get("frames")) != 96:
        raise ValueError("Attribution budget requires exactly 96 frames")
    graph = copy.deepcopy(graph)
    shadows = [node for node in graph["nodes"] if node.get("name") == "Shadows"]
    if len(shadows) != 1 or shadows[0]["properties"].pop("benchmarkBuildSubstages", None) is not True:
        raise ValueError("Attribution requires explicit benchmarkBuildSubstages=true")
    # This is an attribution-only path, not a bypass of the ordinary acceptance gate.
    return bitset.normalized_workload(spec, graph, arm, compact=True)


def verify(source):
    source = Path(source).resolve()
    plan, receipts = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    expected = schedule()
    if (plan.get("protocol") != PROTOCOL or plan.get("blocks") != 2 or plan.get("frames") != 96 or
            plan.get("warmupFrames") != 32 or plan.get("comparisonStartFrame") != 16 or
            plan.get("processes") != expected or len(receipts) != len(expected)):
        raise ValueError("Unknown attribution plan or incomplete fixed two-block ABBA budget")
    rows, frames, faces, input_rows = [], [], [], []
    identity = workload = canonical = previous_end = None
    for item, receipt in zip(expected, receipts):
        if any(receipt.get(key) != value for key, value in item.items()) or receipt.get("status") != "passed":
            raise ValueError("Failed, reordered or replaced attribution process")
        start, end = (datetime.datetime.fromisoformat(receipt[key]) for key in ("started", "ended"))
        if (start.tzinfo is None or end.tzinfo is None or end < start or
                previous_end is not None and start < previous_end):
            raise ValueError("Invalid or overlapping process timestamps")
        previous_end = end
        package = source / item["package"]
        if acceptance.digest(package / "Manifest.json") != receipt["manifestSHA256"]:
            raise ValueError("Attribution child package changed after receipt")
        spec, captures = acceptance.verify_package(package)
        integrity, children = acceptance.read(package / "Integrity.json"), acceptance.read(package / "Processes.json")
        if (len(captures) != 1 or integrity["requestedRuns"] != 1 or integrity["completedRuns"] != 1 or
                len(children) != 1 or children[0].get("timedOut")):
            raise ValueError("Attribution expects one complete independent process per package")
        path, data = captures[0]
        case = data["cases"][0]
        if data["mode"] != "timing" or case.get("buildSubstageDiagnostics") is not True:
            raise ValueError("Missing runtime build-substage attribution declaration")
        recorded = acceptance.read(package / "Identity.json")
        planned = plan["inputs"][item["arm"]]
        if any(recorded["inputs"].get(planned[key]) != planned[key + "SHA256"] for key in ("cases", "graph")):
            raise ValueError("Attribution package inputs differ from declared plan")
        current = {key: recorded[key] for key in ("sources", "binaries")}
        if (not all(current.values()) or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                for values in current.values() for value in values.values()) or
                identity is not None and current != identity):
            raise ValueError("Empty or changed source/binary identity")
        identity = current
        current = normalized_workload(spec, case["resolvedGraph"], item["arm"])
        if workload is not None and current != workload:
            raise ValueError("Attribution workloads differ beyond compactBuild")
        workload = current
        streams = [optimization.stream_signature(frame, "timing", frame["frame"] >= 16) for frame in case["frames"]]
        current = [row for row in streams if row["frame"] >= 16]
        if canonical is not None and current != canonical:
            raise ValueError("Canonical streaming inputs drift between attribution processes")
        canonical = current
        input_rows.extend({"process": item["index"], "arm": item["arm"], **row} for row in streams)
        timings = []
        for frame in case["frames"]:
            # Reuse the strict optional-sidecar validators: enabled work is invalid here.
            profile.work_frame(frame, "timing")
            profile.build_frame(frame, "timing")
            row = profile.build_substage_frame(frame, required=True)
            branch = "compact" if item["arm"] == "B" else "direct"
            if any(face["branch"] != branch for face in row["faces"]):
                raise ValueError("Measured build branch differs from the declared arm")
            faces.extend({"process": item["index"], "arm": item["arm"], "frame": row["frame"],
                          "warmup": row["warmup"], **face} for face in row.pop("faces"))
            timings.append({**profile.profile_frame(frame), **row})
        if case.get("gpuWorkDiagnostics", data.get("gpuWorkDiagnostics", False)):
            raise ValueError("Attribution cannot contain workload counter instrumentation")
        valid = [row for row in timings if not row["warmup"]]
        names = [name for name in ("buildMs", "graphMs", "shadowsMs", *METRICS) if name in valid[0]]
        if any(any(name not in row for name in names) for row in valid):
            raise ValueError("Partially missing build-substage attribution")
        row = {**item, "capture": str(path), "manifestSHA256": receipt["manifestSHA256"],
               "postWarmupFrames": len(valid), "metrics": {
                   name: statistics.median(frame[name] for frame in valid) for name in names},
               "shaders": bitset.shader_identity((path.parent / "Process.log").read_text(encoding="utf-8", errors="replace"),
                                                 item["arm"], "timing", compact=True)}
        rows.append(row)
        frames.extend({"process": item["index"], "arm": item["arm"], "block": item["block"], **row} for row in timings)
    bitset.check_shaders(rows, compact=True)
    return {"protocol": PROTOCOL, "source": str(source), "evidenceValidity": "valid",
            "performanceAcceptance": "unavailable: split timestamps are attribution instrumentation",
            "planSHA256": acceptance.digest(source / "Plan.json"),
            "receiptsSHA256": acceptance.digest(source / "Processes.json"), "identity": identity,
            "workload": workload, "canonicalInputSignatures": canonical, "inputFrames": input_rows,
            "processes": rows, "frames": frames, "faces": faces}


def report(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Report must be outside the sealed source")
    result = verify(source)
    output.mkdir(parents=True, exist_ok=False)
    for name, key in (("Frames", "frames"), ("Faces", "faces"), ("Inputs", "inputFrames"), ("Processes", "processes")):
        acceptance.save(output / f"{name}.json", result[key])
        if name != "Processes":
            profile.write_csv(output / f"{name}.csv", result[key])
    medians = [{"process": row["index"], "arm": row["arm"], "block": row["block"], **row["metrics"]}
               for row in result["processes"]]
    profile.write_csv(output / "ProcessMedians.csv", medians)
    acceptance.save(output / "Summary.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for position, row in enumerate(medians):
        bottom = 0.
        for index, metric in enumerate(METRICS):
            if metric in row:
                axes[0].bar(position + 1, row[metric], bottom=bottom, color=f"C{index}",
                            label=metric if position == (1 if metric in ("compactMs", "compactSyncMs", "indirectMs") else 0) else None)
                bottom += row[metric]
        axes[1].scatter(position + 1, row["buildMs"], color="C0" if row["arm"] == "A" else "C1")
    axes[0].set(ylabel="Sum of per-process substage medians (ms)", xlabel="Independent process in fixed ABBA order",
                title="Split-build attribution (medians need not add to parent median)")
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="Inclusive build median (ms)", xlabel="Independent process in fixed ABBA order",
                title="Instrumented parent build; A=direct, B=compact")
    for axis in axes:
        axis.set_xticks(range(1, 9), [f"{index + 1}\n{row['arm']}" for index, row in enumerate(medians)])
    fig.suptitle("VSM build attribution: 2 ABBA blocks, 96 frames/process, first 32 excluded")
    fig.savefig(output / "BuildAttribution.png", dpi=150)
    plt.close(fig)
    lines = ["# VSM build GPU attribution", "",
             "A: direct build. B: compact candidates and indirect build. Fixed two ABBA blocks, 8 independent processes; 96 frames each, first 32 excluded.", "",
             "These additional timestamps are diagnostic attribution, not ordinary optimization acceptance. Work counters, validation and image readback are off. No stages are added twice to the inclusive build time. Missing/nonapplicable branch stages remain unavailable, never measured zero.", "",
             "Each table value is the median of four independent process medians. Medians do not add exactly; per-frame parent/child bounds are checked before aggregation. No confidence interval is inferred from frames.", "",
             "| Stage (ms) | A direct | B compact |", "| --- | ---: | ---: |"]
    for name in ("buildMs", *METRICS, "shadowsMs", "graphMs"):
        values = []
        for arm in ("A", "B"):
            samples = [row["metrics"][name] for row in result["processes"] if row["arm"] == arm and name in row["metrics"]]
            values.append(f"{statistics.median(samples):.6f}" if samples else "unavailable")
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines.extend(["", "`compactSyncMs` covers synchronization between compact writes and indirect reads; it is not an isolated hardware barrier latency. `finalizeSyncMs` covers the build-output dependency before finalize. `finalizeMs` includes finalize dispatch and subsequent barriers before raster. Child GPU spans include timestamp boundary effects. `buildUnattributedMs` is the measured parent-minus-children remainder, not a kernel.", "",
                  "Actual shader hashes, sealed receipts, source/binary identities and canonical streaming inputs were verified within this batch. Timing texture semantic hashes and images are unavailable here; use separate diagnostic equivalence captures. Separate before/after attribution batches cannot be pooled across changed source/binary identity.", "",
                  "![Build attribution](BuildAttribution.png)", "",
                  "Reproduce using `python -B Reproduce.py NEW_REPORT_DIRECTORY`; source packages are never modified."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tools = {str(Path(module.__file__).resolve()): acceptance.digest(module.__file__)
             for module in (acceptance, bitset, profile, optimization)}
    tools[str(Path(__file__).resolve())] = acceptance.digest(__file__)
    acceptance.save(output / "ReportTools.json", {"hashes": tools, "python": platform.python_version(),
                                                 "matplotlib": matplotlib.__version__, "unit": "milliseconds"})
    script = ("import hashlib, pathlib, subprocess, sys\n"
              f"hashes = {tools!r}\n"
              "for path, expected in hashes.items():\n"
              "    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() != expected: raise SystemExit('Report tool changed: '+path)\n"
              f"subprocess.run([sys.executable, '-B', {str(Path(__file__).resolve())!r}, {str(source)!r}, sys.argv[1]], check=True)\n")
    (output / "Reproduce.py").write_text(script, encoding="utf-8")
    return output / "report.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(report(args.source, args.output))
