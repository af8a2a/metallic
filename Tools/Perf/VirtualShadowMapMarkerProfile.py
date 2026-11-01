"""Verify and report nested VSM GPU markers from an acceptance-v2 package.

Usage: python -B Tools/Perf/VirtualShadowMapMarkerProfile.py SOURCE NEW_REPORT
This tool only reads completed captures; it never launches a GPU workload.
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
import platform
import statistics

import VirtualShadowMapGPUProfile as profile


ROOT_PARENT = 0xffffffff
# Allow endpoint quantization, not overlapping ranges or a percentage-sized gap.
TIMESTAMP_TOLERANCE_MS = 0.0001
REQUIRED_CHILDREN = {
    "VSM reset": ("VSM reset prepare", "VSM reset dispatch", "VSM reset publish"),
    "VSM mark": ("VSM mark dispatch", "VSM mark publish"),
    "VSM allocator": ("VSM allocator count", "VSM allocator allocate", "VSM allocator admit",
                      "VSM allocator physical", "VSM allocator assign"),
    "VSM projection": ("VSM projection prepare", "VSM projection dispatch", "VSM projection publish"),
    "VSM one-pass projection": ("VSM projection prepare", "VSM projection dispatch", "VSM projection publish"),
    "VSM static raster": ("VSM raster prepare", "VSM raster publish"),
    "VSM dynamic raster and restore": ("VSM raster prepare", "VSM raster publish"),
}
STAGE_LABELS = {f"{name}Ms": name.capitalize() for name in profile.STAGES}
STAGE_LABELS["otherShadowsMs"] = "Other Shadows"


def duration(value, label):
    return profile.duration(value, label)


def marker_frame(frame):
    """Validate individual instances before aggregating repeated face paths."""
    stage = profile.profile_frame(frame)
    node = next(n for n in frame["nodes"] if n["name"] == "Shadows")
    sections = node["sections"]
    paths, children, times = [], collections.defaultdict(list), {}
    for index, section in enumerate(sections):
        parent = section["parent"]
        if (isinstance(parent, bool) or not isinstance(parent, int) or
                (parent != ROOT_PARENT and not 0 <= parent < index)):
            raise ValueError(f"Invalid marker parent at frame {frame['frame']}, section {index}")
        name = section["name"]
        if not isinstance(name, str) or not name:
            raise ValueError("Marker names must be nonempty strings")
        path = (() if parent == ROOT_PARENT else paths[parent]) + (name,)
        paths.append(path)
        children[parent].append(index)
        if section.get("cpuOnly", False):
            if section.get("gpuMs") is not None:
                raise ValueError(f"CPU-only marker has GPU timing: {path}")
        else:
            times[index] = duration(section.get("gpuMs"), "/".join(path))

    for parent, child_indices in children.items():
        if parent != ROOT_PARENT and parent not in times:
            continue
        parent_ms = duration(node["gpuMs"], "Shadows") if parent == ROOT_PARENT else times[parent]
        gpu_children = [index for index in child_indices if index in times]
        child_ms = sum(times[index] for index in gpu_children)
        tolerance = TIMESTAMP_TOLERANCE_MS * max(1, len(gpu_children))
        if child_ms > parent_ms + tolerance:
            label = "Shadows" if parent == ROOT_PARENT else "/".join(paths[parent])
            raise ValueError(f"Direct GPU children exceed parent at frame {frame['frame']}: "
                             f"{label}: {child_ms:.9f} > {parent_ms:.9f} ms (+{tolerance:.9f})")

    for index, section in enumerate(sections):
        if index not in times:
            continue
        direct = {sections[c]["name"] for c in children[index] if c in times}
        required = REQUIRED_CHILDREN.get(section["name"], ())
        if section["name"] == "VSM raster":
            required = ("VSM raster diagnostics publish",)
            layered = {"VSM static raster", "VSM dynamic raster and restore"} & direct
            if layered and len(layered) != 2:
                raise ValueError("Incomplete layered raster markers")
            if not layered:
                required += ("VSM raster prepare", "VSM raster publish")
        if set(required) - direct:
            raise ValueError(f"Missing detailed children of {'/'.join(paths[index])}: "
                             f"{sorted(set(required) - direct)}")

    aggregate = {}
    for index, value in times.items():
        path = paths[index]
        row = aggregate.setdefault(path, {"path": list(path), "milliseconds": 0.0,
                                          "instances": 0, "directChildMilliseconds": 0.0})
        row["milliseconds"] += value
        row["instances"] += 1
        row["directChildMilliseconds"] += sum(times[c] for c in children[index] if c in times)
    return stage, [{"frame": frame["frame"], "warmup": frame["warmup"], **row}
                   for row in aggregate.values()]


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "path": " / ".join(row["path"]) if "path" in row else ""})


def report(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Report must be outside the sealed capture package")
    source, captures = profile.load_package(source)
    frame_rows, process_rows, stage_rows, modes = [], [], [], set()
    for process, capture_path, data in captures:
        modes.add(data["mode"])
        for case in data["cases"]:
            if not case.get("gpuProfileDiagnostics", data.get("gpuProfileDiagnostics", False)):
                raise ValueError("Detailed GPU profiling was not enabled")
            stage_frames, marker_frames = [], []
            for frame in case["frames"]:
                stage, markers = marker_frame(frame)
                stage_frames.append(stage)
                marker_frames.extend(markers)
                frame_rows.extend({"process": process, "case": case["id"], **row} for row in markers)
            valid = [row for row in stage_frames if not row["warmup"]]
            if not valid:
                raise ValueError("No post-warmup frames")
            stage_rows.append({"process": process, "case": case["id"], "postWarmupFrames": len(valid),
                               "mode": data["mode"], "source": str(capture_path),
                               **{key: statistics.median(row[key] for row in valid)
                                  for key in (*STAGE_LABELS, "shadowsMs", "graphMs")}})
            grouped = collections.defaultdict(list)
            for row in marker_frames:
                if not row["warmup"]:
                    grouped[tuple(row["path"])].append(row)
            for path, rows in grouped.items():
                process_rows.append({"process": process, "case": case["id"], "path": list(path),
                                     "presentFrames": len(rows), "postWarmupFrames": len(valid),
                                     "missingFrames": len(valid) - len(rows),
                                     "minimumInstances": min(row["instances"] for row in rows),
                                     "maximumInstances": max(row["instances"] for row in rows),
                                     "medianMs": statistics.median(row["milliseconds"] for row in rows),
                                     "medianDirectChildrenMs": statistics.median(row["directChildMilliseconds"] for row in rows)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    metadata = {"source": str(source), "manifestSHA256": profile.acceptance.digest(source / "Manifest.json"),
                "python": platform.python_version(), "matplotlib": matplotlib.__version__,
                "plotScript": str(Path(__file__).resolve()),
                "plotScriptSHA256": profile.acceptance.digest(Path(__file__)),
                "modes": sorted(modes), "independentProcesses": len(captures),
                "aggregation": "Sum same full parent path within each frame, then median per process over post-warmup frames where present. Missing conditional markers are not replaced with zero.",
                "scope": "GPU inclusive marker milliseconds; nested levels are separate and must not be summed together. No speedup inference or confidence interval.",
                "timestampToleranceMsPerDirectChild": TIMESTAMP_TOLERANCE_MS,
                "processStages": stage_rows, "processMarkers": process_rows, "frames": frame_rows}
    profile.acceptance.save(output / "Data.json", metadata)
    write_csv(output / "Markers.csv", process_rows,
              ["process", "case", "path", "presentFrames", "postWarmupFrames", "missingFrames",
               "minimumInstances", "maximumInstances", "medianMs", "medianDirectChildrenMs"])
    write_csv(output / "Frames.csv", frame_rows,
              ["process", "case", "frame", "warmup", "path", "instances", "milliseconds", "directChildMilliseconds"])

    figure, axes = plt.subplots(len(stage_rows), 1, figsize=(10, max(3, 3 * len(stage_rows))), squeeze=False)
    for axis, row in zip(axes.flat, stage_rows):
        values = [row[key] for key in STAGE_LABELS]
        axis.barh(list(STAGE_LABELS.values()), values, color="#4878a8")
        axis.invert_yaxis()
        axis.set(xlabel="GPU ms (per-process median)",
                 title=f"{row['case']} / process {row['process']} / {row['postWarmupFrames']} post-warmup frames")
        axis.set_xlim(0, max(values, default=1) * 1.2 or 1)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:.4f}", va="center", fontsize=9)
    figure.suptitle(f"VSM main stages — {', '.join(sorted(modes))}; n={len(captures)} processes")
    figure.tight_layout(rect=[0, 0, 1, .96])
    figure.savefig(output / "Stages.png", dpi=160)
    plt.close(figure)

    detail_groups = collections.defaultdict(list)
    for row in process_rows:
        if len(row["path"]) > 1:
            detail_groups[(row["process"], row["case"], tuple(row["path"][:-1]))].append(row)
    figures = []
    # One panel per immediate parent; no chart sums different hierarchy levels.
    groups = list(detail_groups.items())
    for page in range(0, len(groups), 8):
        selected = groups[page:page + 8]
        fig, axes = plt.subplots(len(selected), 1, figsize=(12, max(3, sum(max(2.1, len(rows) * .31 + .9) for _, rows in selected))), squeeze=False)
        for axis, ((process, case, parent), rows) in zip(axes.flat, selected):
            rows.sort(key=lambda row: row["medianMs"], reverse=True)
            values = [row["medianMs"] for row in rows]
            axis.barh([row["path"][-1] for row in rows], values, color="#d6953f")
            axis.invert_yaxis()
            axis.set(xlabel="GPU ms (per-process median; children only)",
                     title=f"{case} / process {process}: {' / '.join(parent)}")
            axis.set_xlim(0, max(values, default=1) * 1.55 or 1)
            for index, row in enumerate(rows):
                axis.text(row["medianMs"], index, f" {row['medianMs']:.5f} ({row['presentFrames']}/{row['postWarmupFrames']} frames)", va="center", fontsize=8)
        fig.suptitle("VSM detailed markers — inclusive children, separated by immediate parent", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .985])
        name = f"Details-{page // 8 + 1}.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        figures.append(name)

    lines = ["# VSM nested GPU marker report", "",
             f"Verified package: `{source}`. Mode: {', '.join(sorted(modes))}. Independent processes: **{len(captures)}**.", "",
             "These are attribution measurements. Detailed timestamp boundaries can perturb scheduling; diagnostic validation/readbacks add overhead. They are not ordinary performance acceptance or an A/B speedup estimate. Frames are not independent experiments; no confidence interval is computed.", "",
             "Each frame first sums repeated face instances with exactly the same complete parent path. Process summaries use post-warmup medians. Conditional absent markers remain missing; their present-frame count is reported. Inclusive parent and child times must not be added, and medians of children need not sum to the median of the parent.", "",
             "GPU timestamps delimit command ranges, not isolated shader instructions. Prepare/publish spans include the dependency work between timestamp endpoints; dispatch timing does not split receiver setup, SMRT traversal, or individual shader samples. Allocator children and build finalize include their publication dependencies. Raster prepare includes attachment setup/cold clear; raster publish includes end-rendering/publication. Low or zero spans may reflect timestamp resolution or empty GPU work.", "",
             f"Validation: package integrity and runtime errors checked by the current acceptance verifier; every marker parent precedes its child; GPU durations are finite/nonnegative; each direct-child sum is bounded by its parent allowing {TIMESTAMP_TOLERANCE_MS:g} ms per child for endpoint quantization. Required details are checked only for the executed parent path; layered and one-pass paths are not mandatory.", "",
             "![Main stages](Stages.png)", "", "## Per-process main stages", "",
             "| Case / process | Frames | Projection ms | Build ms | Mark ms | Shadows ms | Graph ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stage_rows:
        lines.append(f"| {row['case']} / {row['process']} | {row['postWarmupFrames']} | {row['projectionMs']:.6f} | {row['buildMs']:.6f} | {row['markMs']:.6f} | {row['shadowsMs']:.6f} | {row['graphMs']:.6f} |")
    lines += ["", "## Immediate-child detail panels", ""]
    for name in figures:
        lines += [f"![Nested markers]({name})", ""]
    lines += ["## Exact data and reproduction", "",
              "`Data.json` preserves full per-frame paths, aggregate instance counts, per-process medians, missing-frame counts and input manifest hash. `Markers.csv` and `Frames.csv` contain the corresponding tables. Timing units are milliseconds; no unit conversion or extra outlier exclusion is applied.", "",
              "```powershell", f'python -B Tools/Perf/VirtualShadowMapMarkerProfile.py "{source}" "<new-report-directory>"', "```", "",
              f"Python {platform.python_version()}, Matplotlib {matplotlib.__version__}. The script path and SHA-256 are in Data.json. The report never modifies the sealed input package."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.source, args.output)
