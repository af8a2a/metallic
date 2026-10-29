"""Report VSM GPU stages from verified acceptance-v2 packages.

Usage: python -B Tools/Perf/VirtualShadowMapGPUProfile.py SOURCE OUTPUT
       [--diagnostic DIAGNOSTIC_PACKAGE]

The five stages are mark, allocator, build, raster and projection; reset is
reported separately. Per-face stage spans are summed once. Independent process
medians, rather than pooled frames, are the summary unit.
"""
import argparse
import collections
import csv
import math
from pathlib import Path
import statistics
import sys

import VirtualShadowMapAcceptance as acceptance


STAGES = ("reset", "mark", "allocator", "build", "raster", "projection")
SECTION_NAMES = {f"VSM {stage}": stage for stage in STAGES}
ROOT_PARENT = 0xffffffff
WORK_COUNTERS = (
    "markReceiverDispatches", "markSamplePositions", "markLevelTests", "markPageRequests",
    "buildActiveTriangleSlots", "buildDecodedTriangles", "buildLevelTests", "buildPageTests",
    "rasterCoarseGroups", "rasterFineGroups", "rasterDecodedTriangles", "rasterOutputPrimitives",
    "rasterClearedPages", "projectionReceivers", "projectionRays", "projectionAtlasLoads",
)
COUNTER_DEFINITIONS = {
    "markReceiverDispatches": "Threads with a depth receiver in the light/face domain; point-light face dispatches may repeat receivers.",
    "markSamplePositions": "Calls to markPosition, including samples subsequently rejected by the point-face check.",
    "markLevelTests": "Actual level-projection tests after face matching.",
    "markPageRequests": "Executed page-request InterlockedOr operations; not unique requested pages.",
    "buildActiveTriangleSlots": "Slots reaching triangle decode after range, enabled, dirty-count and optional cluster coarse-cull exits; not slot capacity.",
    "buildDecodedTriangles": "Successful triangle decodes in build.",
    "buildLevelTests": "Actual triangle level iterations after optional cluster level-mask rejection; cluster prechecks are excluded.",
    "buildPageTests": "Candidate pages in exact triangle projected bounds before dirty/admission checks; cluster precheck page probes are excluded.",
    "rasterCoarseGroups": "Mesh shader groups entering the coarse raster path.",
    "rasterFineGroups": "Mesh shader groups entering the fine raster path.",
    "rasterDecodedTriangles": "Raster mesh groups with a successful triangle decode.",
    "rasterOutputPrimitives": "Triangles emitted after clipping; not fragments or covered samples.",
    "rasterClearedPages": "Pages cleared by the hot-cache clear shader; cold attachment clears are excluded.",
    "projectionReceivers": "Receivers passing canonical-face, enabled and domain checks.",
    "projectionRays": "Actual SMRT ray-loop iterations; hard/PCF paths do not execute these rays.",
    "projectionAtlasLoads": "Executed hard/filtered atlas.Load calls, including coarse/cross-face loads; failed lookups excluded.",
}
PAGE_STATISTICS = {0: "requestedFinePages", 1: "admittedFinePages", 2: "unadmittedFinePages",
                   16: "admittedDirtyFinePages", 17: "admittedCleanFinePages", 18: "invalidatedMappedPages",
                   19: "remappedPages", 20: "newlyAllocatedFinePages", 24: "validMappedFinePages"}


def duration(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing or invalid GPU timestamp: {label}")
    return value


def describe(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No post-warmup samples; missing data cannot be reported as zero")

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"count": len(ordered), "mean": statistics.mean(ordered), "median": statistics.median(ordered),
            "p95": percentile(.95), "p99": percentile(.99), "minimum": ordered[0], "maximum": ordered[-1]}


def profile_frame(frame):
    nodes = [node for node in frame["nodes"] if node["name"] == "Shadows"]
    if len(nodes) != 1 or nodes[0]["type"] != "VirtualShadowMapLightsPass":
        raise ValueError("Expected exactly one production Shadows VSM Lights pass")
    node = nodes[0]
    sections = node["sections"]
    spans = {stage: [] for stage in STAGES}
    selected = set()
    for index, section in enumerate(sections):
        parent = section["parent"]
        if parent != ROOT_PARENT and (isinstance(parent, bool) or not isinstance(parent, int) or parent < 0 or parent >= index):
            raise ValueError("Invalid GPU section parent ordering")
        stage = SECTION_NAMES.get(section["name"])
        if stage is None:
            continue
        if section.get("cpuOnly", False):
            raise ValueError(f"CPU-only section cannot supply GPU timing: {section['name']}")
        ancestor = parent
        while ancestor != ROOT_PARENT:
            if ancestor in selected:
                raise ValueError("Nested VSM stage spans would double count GPU work")
            ancestor = sections[ancestor]["parent"]
        selected.add(index)
        spans[stage].append(duration(section["gpuMs"], section["name"]))
    counts = {stage: len(values) for stage, values in spans.items()}
    if not counts["reset"] or len(set(counts.values())) != 1:
        raise ValueError(f"Missing or inconsistent per-face VSM stage spans: {counts}")
    totals = {stage: sum(values) for stage, values in spans.items()}
    shadow_ms = duration(node["gpuMs"], "Shadows")
    measured = sum(totals.values())
    # Timestamp quantization permits a tiny discrepancy, not overlapping scopes.
    if measured > shadow_ms + max(1e-5, shadow_ms * 1e-5):
        raise ValueError("Summed VSM stages exceed the owning Shadows GPU span")
    return {"frame": frame["frame"], "warmup": frame["warmup"], "graphMs": frame["graphMs"],
            "shadowsMs": shadow_ms, "stageSumMs": measured, "otherShadowsMs": max(0., shadow_ms - measured),
            "faceCount": counts["reset"], **{f"{stage}Ms": value for stage, value in totals.items()}}


def load_package(source):
    source = Path(source).resolve()
    _, captures = acceptance.verify_package(source)
    # Older report verification did not reject an empty collection or compare
    # completedRuns. Stage statistics need actual completed process samples.
    integrity = acceptance.read(source / "Integrity.json")
    processes = acceptance.read(source / "Processes.json")
    if not captures or len(captures) != integrity["completedRuns"] or any(p.get("timedOut") for p in processes):
        raise ValueError("Incomplete evidence cannot be reported as accepted measurements")
    return source, [(index, path, data) for index, (path, data) in enumerate(captures, 1)]


def integer_counter(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffffffffffff:
        raise ValueError(f"Invalid unsigned GPU counter: {label}")
    return value


def work_frame(frame, mode):
    shadow = frame.get("shadow", {})
    work = shadow.get("gpuWork")
    result = {"frame": frame["frame"], "warmup": frame["warmup"], "gpuWorkAvailable": False}
    if work is not None:
        if work.get("version") != 1 or not isinstance(work.get("enabled"), bool):
            raise ValueError("Unknown or malformed GPU work schema")
        if work["enabled"]:
            if mode != "diagnostic":
                raise ValueError("GPU work instrumentation cannot enter ordinary timing evidence")
            faces, totals = work["faces"], work["totals"]
            if not faces or set(totals) != set(WORK_COUNTERS):
                raise ValueError("Missing GPU work faces or counters")
            face_ids = set()
            summed = dict.fromkeys(WORK_COUNTERS, 0)
            for face in faces:
                identity = (integer_counter(face["lightIndex"], "lightIndex"), integer_counter(face["face"], "face"))
                if identity in face_ids or identity[1] >= 6 or set(face["counters"]) != set(WORK_COUNTERS):
                    raise ValueError("Duplicated or malformed GPU work face")
                face_ids.add(identity)
                for name in WORK_COUNTERS:
                    summed[name] += integer_counter(face["counters"][name], name)
            for name in WORK_COUNTERS:
                if summed[name] != integer_counter(totals[name], name):
                    raise ValueError(f"GPU work face totals disagree: {name}")
            result.update(gpuWorkAvailable=True, **totals)
        elif "faces" in work or "totals" in work:
            raise ValueError("Disabled GPU work must not publish stale counters")
    if mode == "diagnostic" and "totals" in shadow:
        totals = shadow["totals"]
        if len(totals) != 32:
            raise ValueError("Invalid legacy page-statistics ABI")
        result.update({name: integer_counter(totals[index], name) for index, name in PAGE_STATISTICS.items()})
    return result


def summarize_package(source):
    source, captures = load_package(source)
    rows, work_rows, work_faces, processes = [], [], [], []
    for index, path, data in captures:
        if data["mode"] == "timing" and data.get("gpuWorkDiagnostics", False):
            raise ValueError("GPU work instrumentation cannot enter ordinary timing evidence")
        for case in data["cases"]:
            if case.get("gpuWorkDiagnostics", False) and data["mode"] != "diagnostic":
                raise ValueError("GPU work instrumentation cannot enter ordinary timing evidence")
            frames = [profile_frame(frame) for frame in case["frames"]]
            work = [work_frame(frame, data["mode"]) for frame in case["frames"]]
            if case.get("gpuWorkDiagnostics", False) and not all(row["gpuWorkAvailable"] for row in work):
                raise ValueError("Declared GPU work instrumentation is missing frame readback")
            for raw, row in zip(case["frames"], work):
                if row["gpuWorkAvailable"]:
                    work_faces.extend({"case": case["id"], "process": index, "frame": raw["frame"],
                                       "warmup": raw["warmup"], "lightIndex": face["lightIndex"],
                                       "face": face["face"], **face["counters"]}
                                      for face in raw["shadow"]["gpuWork"]["faces"])
            valid = [frame for frame in frames if not frame["warmup"]]
            columns = ["graphMs", "shadowsMs", "stageSumMs", "otherShadowsMs", *[f"{stage}Ms" for stage in STAGES]]
            metrics = {column: describe([frame[column] for frame in valid]) for column in columns}
            counter_names = list(WORK_COUNTERS) + list(PAGE_STATISTICS.values())
            counters = {name: describe([row[name] for row in work if not row["warmup"] and name in row])
                        for name in counter_names if any(not row["warmup"] and name in row for row in work)}
            processes.append({"case": case["id"], "process": index, "mode": data["mode"],
                              "source": str(path), "metrics": metrics, "counters": counters,
                              "workCounterFrames": sum(not row["warmup"] and row["gpuWorkAvailable"] for row in work)})
            rows.extend({"case": case["id"], "process": index, "mode": data["mode"], **frame} for frame in frames)
            work_rows.extend({"case": case["id"], "process": index, "mode": data["mode"], **frame} for frame in work)
    grouped = collections.defaultdict(list)
    for process in processes:
        grouped[(process["case"], process["mode"])].append(process)
    cases = []
    for (case, mode), entries in grouped.items():
        counter_names = sorted(set().union(*(entry["counters"] for entry in entries)))
        cases.append({"case": case, "mode": mode, "independentProcesses": len(entries),
                      "processMedianSummary": {column: describe([entry["metrics"][column]["median"] for entry in entries])
                                               for column in entries[0]["metrics"]},
                      "counterProcessMedianSummary": {name: describe([entry["counters"][name]["median"] for entry in entries
                                                                       if name in entry["counters"]]) for name in counter_names}})
    return {"source": str(source), "manifestSHA256": acceptance.digest(source / "Manifest.json"),
            "cases": cases, "processes": processes, "frames": rows, "workFrames": work_rows, "workFaces": work_faces}


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_package(package, output, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = []
    for ordinal, case in enumerate(package["cases"], 1):
        frames = [row for row in package["frames"] if row["case"] == case["case"]]
        processes = [row for row in package["processes"] if row["case"] == case["case"]]
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        positions = list(range(len(processes)))
        for offset, stage in enumerate(STAGES):
            values = [process["metrics"][f"{stage}Ms"]["median"] for process in processes]
            axes[0].bar([position + offset * .12 for position in positions], values, width=.12, label=stage)
        axes[0].set_xticks([position + .3 for position in positions], [f"process {p['process']}" for p in processes])
        axes[0].set(ylabel="Median per-frame GPU span (ms)",
                    title=f"{case['case']}: {case['mode']} — {len(processes)} independent processes")
        axes[0].legend(ncol=3)
        axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.22)
        for process in processes:
            subset = [row for row in frames if row["process"] == process["process"]]
            axes[1].plot([row["frame"] for row in subset], [row["shadowsMs"] for row in subset],
                         linewidth=.8, label=f"Shadows, process {process['process']}")
            warmup = [row["frame"] for row in subset if row["warmup"]]
            if warmup:
                axes[1].axvspan(min(warmup), max(warmup) + .5, color="grey", alpha=.08)
        axes[1].set(xlabel="Frame (shading marks warmup)", ylabel="Shadows GPU span (ms)")
        axes[1].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=.25)
        filename = f"{prefix}-{ordinal}.png"
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)
        images.append(filename)
    return images, matplotlib.__version__


def plot_work(case, output, filename):
    import matplotlib.pyplot as plt

    values = case["counterProcessMedianSummary"]
    names = [name for name in WORK_COUNTERS if name in values]
    if not names:
        return None
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.barh(names, [values[name]["median"] for name in names], color="#386cb0")
    axis.invert_yaxis()
    axis.set_xscale("symlog", linthresh=1)
    axis.set(xlabel="Count per frame (median of diagnostic process medians; symlog scale)",
             title=f"{case['case']}: instrumented GPU work, separate from ordinary timings")
    axis.grid(axis="x", alpha=.25)
    figure.savefig(output / filename, dpi=150)
    plt.close(figure)
    return filename


def report(args):
    package = summarize_package(args.source)
    diagnostic = summarize_package(args.diagnostic) if args.diagnostic else None
    if diagnostic and any(case["mode"] != "diagnostic" for case in diagnostic["cases"]):
        raise ValueError("--diagnostic must reference a diagnostic acceptance package")
    output = args.output.resolve()
    for source in (package, diagnostic):
        if source and (Path(source["source"]) == output or Path(source["source"]) in output.parents):
            raise ValueError("Report output must be outside the sealed source package")
    output.mkdir(parents=True, exist_ok=False)
    packages = [("Primary", package)] + ([("Diagnostic", diagnostic)] if diagnostic else [])
    text = ["# VSM GPU stage profile", "",
            "Five stages: mark, allocator, build, raster and projection. Reset is reported separately.",
            "Each frame sums matching spans across faces once; inclusive parent sections are excluded.",
            "Summary cells are medians of independent process medians. Frames are not independent experiments.",
            "Other Shadows includes work outside these spans (publication, temporal stabilization and intervening commands); it is not CPU time.",
            "Diagnostic timestamps include validation/readback/instrumentation overhead and are not ordinary performance measurements.",
            "Counter values come only from diagnostic readback. Missing counters remain unavailable, never zero.",
            "Page statistics describe allocation/cache state; capacities and page counts are not executed triangle/ray counts.",
            "Separate packages are shown separately; this report does not establish that their workloads or source/binary identities match.",
            "No RT/VSM speedup or end-to-end FPS is inferred. Input assets may only be identified by path in the source package.", ""]
    for label, item in packages:
        images, version = plot_package(item, output, label)
        acceptance.save(output / f"{label}.json", item)
        write_csv(output / f"{label}Frames.csv", item["frames"])
        write_csv(output / f"{label}Work.csv", item["workFrames"])
        if item["workFaces"]:
            write_csv(output / f"{label}WorkFaces.csv", item["workFaces"])
        text.extend([f"## {label} package", "", f"Source: `{item['source']}`", "",
                     f"Manifest SHA256: `{item['manifestSHA256']}`", ""])
        for ordinal, (case, filename) in enumerate(zip(item["cases"], images), 1):
            text.extend([f"### {case['case']} ({case['mode']})", "",
                         "| Scope | Median of process medians (ms) | Process median range (ms) |",
                         "| --- | ---: | ---: |"])
            for column, values in case["processMedianSummary"].items():
                text.append(f"| {column} | {values['median']:.4f} | {values['minimum']:.4f}–{values['maximum']:.4f} |")
            text.extend(["", f"Independent processes: {case['independentProcesses']}. Reset is excluded from the five-stage definition.",
                         "", f"![VSM GPU stages]({filename})", ""])
            counters = case["counterProcessMedianSummary"]
            if counters:
                text.extend(["| Diagnostic counter | Median of process medians | Processes with evidence |",
                             "| --- | ---: | ---: |"])
                for name, values in counters.items():
                    text.append(f"| {name} | {values['median']:.1f} | {values['count']} |")
                text.extend(["", "Per-process sample counts and ranges are preserved in JSON; individual frames are in the Work CSV.", ""])
            if not any(name in counters for name in WORK_COUNTERS):
                text.extend(["GPU execution counters: **unavailable** (instrumentation disabled or no diagnostic readback).", ""])
            work_image = plot_work(case, output, f"{label}-{ordinal}-work.png")
            if work_image:
                text.extend([f"![Diagnostic GPU work]({work_image})", ""])
    text.extend(["## GPU counter definitions", "", "| Counter | Measured quantity |", "| --- | --- |"])
    text.extend(f"| {name} | {definition} |" for name, definition in COUNTER_DEFINITIONS.items())
    text.extend(["", "Legacy requested/admitted/dirty/clean/remapped/allocated/valid page statistics are separate state counters.",
                 "`validMappedFinePages` includes cached pages not requested this frame; `invalidatedMappedPages` may include coarse and unrequested pages.",
                 "No triangle capacity, dispatch upper bound, or allocation size is substituted for executed work.", ""])
    acceptance.save(output / "ReportIdentity.json", {"python": sys.version, "matplotlib": version,
                    "toolSHA256": acceptance.digest(Path(__file__)),
                    "sources": {label: item["manifestSHA256"] for label, item in packages}})
    diagnostic_args = ["--diagnostic", str(args.diagnostic.resolve())] if args.diagnostic else []
    (output / "Reproduce.py").write_text(
        "\"\"\"Recreate this report: python -B Reproduce.py NEW_OUTPUT_DIRECTORY\"\"\"\n"
        "import subprocess\nimport sys\n\n"
        f"command = [sys.executable, '-B', {str(Path(__file__).resolve())!r}, {str(args.source.resolve())!r}, sys.argv[1]]\n"
        f"command.extend({diagnostic_args!r})\nsubprocess.run(command, check=True)\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(output / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified acceptance-v2 package")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--diagnostic", type=Path, help="Separate diagnostic package; never pooled with timing")
    args = parser.parse_args()
    report(args)


if __name__ == "__main__":
    main()
