"""Render CPU-job evidence without pooling frames or adding overlapping scopes.

Reads archived M0/M2/meshlet data and/or MeasureProductionCPUJobs.py output.
Writes a NEW report directory; input evidence and its manifests are read-only.
Requires matplotlib. No renderer, benchmark process or compiler is launched.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import textwrap

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLORS = ["#2563eb", "#e47716", "#159477", "#a855b5", "#64748b", "#c43e50"]
LABELS = {
    "task_graph_directed": "TaskGraph directed",
    "stdexec_default": "raw stdexec ordinary",
    "stdexec_directed": "raw stdexec directed",
    "legacy_atomic": "legacy atomic pool",
    "oneTBB": "oneTBB",
    "static": "Static",
    "dynamic": "Dynamic",
    "adaptive_half_steal": "Adaptive half-steal",
    "half_steal": "Adaptive half-steal",
    "scalar-interval": "Scalar interval",
    "cpp-auto-vectorized": "C++ auto-vectorized",
    "ispc-avx2": "ISPC AVX2",
    "ispc-sse4.2": "ISPC SSE4.2",
    "aos-to-soa": "AoS -> SoA packing",
    "legacy": "legacy adapter",
    "shared": "shared TaskSystem",
}


def number(value):
    """Missing is None, never zero. Reject malformed/nonfinite numeric evidence."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Boolean in numeric evidence: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite numeric evidence: {value!r}")
    return result


def quantile(values, probability, historical=False):
    values = sorted(values)
    if not values:
        return None
    index = math.floor(probability * (len(values) - 1)) if historical else max(0, math.ceil(probability * len(values)) - 1)
    return values[index]


def distribution(record, field, statistic="p50"):
    value = record.get(field)
    return number(value.get(statistic)) if isinstance(value, dict) else None


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def json_lines(path):
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def frame_path_segments(frame):
    """Only explicit non-overlapping coordinator intervals can be stacked.

    This is a coordinator wall path, not an inferred TaskGraph longest path.
    Inclusive scopes and scalar duration fields cannot establish these intervals.
    """
    contract = frame.get("cpuFramePathContract")
    segments = frame.get("cpuFramePathSegmentsMs")
    if contract != "nonoverlapping-coordinator-wall-segments-v1" or segments is None:
        return None
    wall = number(frame.get("frameMs"))
    if wall is None:
        return None
    ordered = []
    for segment in segments:
        start, end = number(segment.get("startMs")), number(segment.get("endMs"))
        if start is None or end is None or start < 0 or end < start or end > wall:
            raise ValueError("CPU frame-path interval is incomplete, negative or outside its frame")
        ordered.append((start, end, str(segment["name"])))
    ordered.sort()
    last, unattributed = 0.0, 0.0
    totals = {}
    for start, end, name in ordered:
        if start < last:
            raise ValueError("Overlapping CPU frame-path intervals cannot form a stacked breakdown")
        unattributed += start - last
        last = end
        totals[name] = totals.get(name, 0.0) + end - start
    totals["Unattributed frame envelope"] = unattributed + wall - last
    return totals


class Report:
    def __init__(self, output, title, budgets):
        self.output = output
        self.title = title
        self.budgets = budgets
        self.inputs = {}
        self.figures = []
        self.notes = []
        self.tables = []
        self.plot_context = ""
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
            "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
            "axes.spines.top": False, "axes.spines.right": False,
            "grid.color": "#dce3ed", "grid.alpha": .65, "savefig.facecolor": "white",
        })

    def read_json(self, path):
        self.inputs[str(path.resolve())] = digest(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def read_lines(self, path):
        self.inputs[str(path.resolve())] = digest(path)
        return json_lines(path)

    def read_csv(self, path):
        self.inputs[str(path.resolve())] = digest(path)
        with path.open(encoding="utf-8-sig", newline="") as source:
            return list(csv.DictReader(source))

    def save(self, figure, name, title, caption):
        if self.plot_context:
            figure.suptitle(self.plot_context, fontsize=10, color="#9a3412")
            caption = self.plot_context + ". " + caption
        figure.text(.01, -.025, textwrap.fill(caption, 155), ha="left", va="top", fontsize=8, color="#526176")
        figure.savefig(self.output / name, dpi=160, bbox_inches="tight")
        plt.close(figure)
        self.figures.append((name, title, caption))

    def note(self, text):
        self.notes.append(text)

    def table(self, title, headings, rows):
        self.tables.append((title, headings, rows))

    def finish(self):
        for filename, expected in self.inputs.items():
            if digest(Path(filename)) != expected:
                raise RuntimeError(f"Evidence changed while plotting: {filename}; create a fresh report after sampling completes")
        manifest = {
            "schema": 1, "generatedUtc": datetime.now(timezone.utc).isoformat(),
            "plottingPlatform": platform.platform(), "python": sys.version, "matplotlib": matplotlib.__version__,
            "plotterSHA256": digest(Path(__file__).resolve()), "inputsSHA256": self.inputs,
            "budgets": self.budgets, "notes": self.notes,
            "figures": [{"file": name, "title": title, "caption": caption} for name, title, caption in self.figures],
            "figureSHA256": {name: digest(self.output / name) for name, _, _ in self.figures},
        }
        (self.output / "PlotManifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        lines = [f"# {self.title}", "", "Generated from immutable evidence; no renderer or benchmark was executed.", ""]
        html_parts = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
            f"<title>{html.escape(self.title)}</title>",
            "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1320px;margin:32px auto;padding:0 24px;color:#17243a;background:#fff}"
            "h1,h2{line-height:1.2}section{margin:36px 0}img{width:100%;height:auto;border:1px solid #e2e8f0;border-radius:8px}"
            "figcaption,.note{color:#526176}table{border-collapse:collapse;font-size:14px;width:100%;margin:16px 0}"
            "td,th{text-align:left;border-bottom:1px solid #dce3ed;padding:8px;vertical-align:top;overflow-wrap:anywhere}code{overflow-wrap:anywhere}</style>",
            f"<h1>{html.escape(self.title)}</h1><p class='note'>Generated from immutable evidence; no renderer or benchmark was executed.</p>"]
        for note in self.notes:
            lines += [f"- {note}"]
            html_parts.append(f"<p class='note'>{html.escape(note)}</p>")
        lines.append("")
        for name, title, caption in self.figures:
            lines += [f"## {title}", "", f"![{title}]({name})", "", caption, ""]
            html_parts.append(f"<section><h2>{html.escape(title)}</h2><figure><img src='{html.escape(name)}' alt='{html.escape(title)}'>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure></section>")
        for title, headings, rows in self.tables:
            lines += [f"## {title}", "", "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
            html_parts.append(f"<section><h2>{html.escape(title)}</h2><table><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headings) + "</tr>")
            for row in rows:
                lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
                html_parts.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
            lines.append("")
            html_parts.append("</table></section>")
        lines += ["[Input and plot provenance](PlotManifest.json)", ""]
        html_parts.append("<p><a href='PlotManifest.json'>Input and plot provenance</a></p></html>")
        (self.output / "Report.md").write_text("\n".join(lines), encoding="utf-8")
        (self.output / "Report.html").write_text("\n".join(html_parts), encoding="utf-8")


def grouped_lines(axis, rows, key, label, historical=False):
    groups = {}
    for row in rows:
        groups.setdefault((row[key], int(row["workers"])), []).append(number(row["wallNs"]) / 1e6)
    names = sorted({name for name, _ in groups})
    for index, name in enumerate(names):
        workers = sorted(worker for group, worker in groups if group == name)
        medians = [quantile(groups[name, worker], .5, historical) for worker in workers]
        tails = [quantile(groups[name, worker], .95, historical) for worker in workers]
        axis.plot(workers, medians, "o-", color=COLORS[index % len(COLORS)], label=label(name))
        axis.fill_between(workers, medians, tails, color=COLORS[index % len(COLORS)], alpha=.10)
    axis.set_xticks([1, 2, 4, 8])
    axis.set_xlabel("Requested workers")
    axis.set_ylabel("Wall time (ms)")
    axis.grid(axis="y")


def historical_plots(report, root):
    identities = []

    def dataset(name):
        path = root / name / "Raw.jsonl"
        if not path.exists():
            report.note(f"Historical {name} evidence unavailable: {path}")
            return []
        metadata = root / name / "Metadata.json"
        if metadata.exists():
            info = report.read_json(metadata)
            if info.get("executables"):
                identities.extend([[name, candidate, value.get("sha256", "unavailable")]
                    for candidate, value in info["executables"].items()])
            else:
                identities.append([name, "benchmark executable", info.get("executableSha256", "unavailable")])
        return report.read_lines(path)

    baseline, small, ranges = dataset("Baseline"), dataset("SmallTasks"), dataset("Range")
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), constrained_layout=True)
    for axis, rows, title in zip(axes, [baseline, small], ["256-leaf fan-out/join", "1,024 tiny DAG leaves"]):
        selected = [row for row in rows if row.get("domain") == "fanout_join" and row.get("workload") == "uniform"
            and row.get("cache") == "warm" and row.get("diagnostics") is False
            and row.get("backend") in {"task_graph_directed", "stdexec_default", "stdexec_directed"}]
        grouped_lines(axis, selected, "backend", lambda value: LABELS.get(value, value), historical=True)
        axis.set_title(title)
        axis.legend(fontsize=8)
    report.save(figure, "HistoricalScheduler.png", "Archived scheduler baseline",
        "P50 lines and P50-P95 bands; 20 warm invocations per point, diagnostics off, grain 64. "
        "Raw stdexec and TaskGraph have different graph bookkeeping; these are not deque-only costs. "
        "Archived floor-index quantiles are retained. Invocations are not independent fresh processes.")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for axis, workload in zip(axes, ["uniform", "long_tail", "bandwidth"]):
        selected = [row for row in ranges if row.get("domain") == "range" and row.get("workload") == workload
            and row.get("cache") == "warm" and row.get("diagnostics") is False]
        grouped_lines(axis, selected, "backend", lambda value: LABELS.get(value, value), historical=True)
        axis.set_title(workload.replace("_", " ").title())
        axis.legend(fontsize=8)
    report.save(figure, "HistoricalRange.png", "Archived range strategies",
        "P50 lines and P50-P95 bands; grain 64, 20 warm invocations per point, diagnostics off. "
        "Uniform/long-tail contain 65,536 elements; bandwidth contains 16,777,216 uint32 values. "
        "Identical kernels within each workload; useful work, scheduling and join are timed together. No FPS inference.")

    kernel_path = root / "Kernels/Raw.csv"
    if kernel_path.exists():
        info = report.read_json(root / "Kernels/Metadata.json")
        identities.extend([["Kernels", candidate, value] for candidate, value in info.get("executable_sha256", {}).items()])
        kernels = report.read_csv(kernel_path)
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        for axis, count in zip(axes, [65536, 1048576]):
            for backend_index, backend_kind in enumerate(["scalar-interval", "cpp-auto-vectorized", "ispc"]):
                for config_index, config in enumerate(["multi", "sse4", "avx2"]):
                    selected = [number(row["wall_ns"]) / 1e6 for row in kernels if row["phase"] == "warm"
                        and int(row["count"]) == count and row["configuration"] == config
                        and (row["backend"].startswith("ispc-") if backend_kind == "ispc" else row["backend"] == backend_kind)]
                    if not selected:
                        continue
                    median, tail = statistics.median(selected), quantile(selected, .95)
                    x = config_index + (backend_index - 1) * .23
                    axis.bar(x, median, width=.21, color=COLORS[backend_index], label=["Scalar", "C++ auto", "ISPC"][backend_index] if config_index == 0 else None)
                    axis.errorbar(x, median, yerr=[[0], [max(0, tail - median)]], fmt="none", color="#344358", capsize=3)
            axis.set_xticks([0, 1, 2], ["SSE4.2/AVX2\ndispatch", "SSE4.2 only", "AVX2 only"])
            axis.set_title(f"{count:,} AABBs; 1 caller thread")
            axis.set_ylabel("Kernel wall time (ms)")
            axis.grid(axis="y")
            axis.legend()
        report.save(figure, "HistoricalSIMD.png", "Archived SIMD kernel isolation",
            "Bars show warm P50, whiskers P95; each bar has 93 warm samples from 3 processes x 31 invocations. "
            "Same interval algorithm and SoA input; allocation/repacking/scheduling excluded. Multi-target selects AVX2 on this host. "
            "The 31-element timer-quantized case is retained in raw evidence, omitted from this scale comparison.")

        figure, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
        for index, backend in enumerate(["cpp-auto-vectorized", "ispc-avx2", "aos-to-soa"]):
            values, tails = [], []
            for count in [65536, 1048576]:
                samples = [number(row["wall_ns"]) / 1e6 for row in kernels if row["phase"] == "warm"
                    and int(row["count"]) == count and row["configuration"] == "multi" and row["backend"] == backend]
                values.append(statistics.median(samples) if samples else math.nan)
                tails.append(quantile(samples, .95) if samples else math.nan)
            x = [point + (index - 1) * .14 for point in range(2)]
            axis.plot(x, values, "o", color=COLORS[index], label=LABELS[backend])
            axis.vlines(x, values, tails, color=COLORS[index], alpha=.5)
        axis.set_xticks([0, 1], ["65,536 AABBs", "1,048,576 AABBs"])
        axis.set_ylabel("Independently timed wall time (ms; log scale)")
        axis.set_yscale("log")
        axis.grid(axis="y")
        axis.legend()
        report.save(figure, "HistoricalPacking.png", "Archived layout-conversion cost",
            "Runtime-dispatch configuration, 93 warm samples per point. Dots show P50, whiskers P95 on a log axis. "
            "Simple AoS-to-SoA packing is timed separately from kernels; values must not be added into an asserted end-to-end scene speedup. "
            "Optimized packing and production SoA lifetime were not measured.")

    meshlets = dataset("Meshlet")
    samples = [row for row in meshlets if row.get("type") == "sample" and row.get("phase") == "steady"]
    if samples:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for axis, configuration in zip(axes, ["plain", "attributes", "seams"]):
            for candidate_index, candidate in enumerate(["legacy", "shared"]):
                for cells_index, cells in enumerate([64, 128]):
                    selected = [row for row in samples if row["configuration"] == configuration and row["candidate"] == candidate and row["cells"] == cells]
                    workers = sorted({int(row["requestedWorkers"]) for row in selected})
                    grouped = [[number(row["wallNs"]) / 1e6 for row in selected if int(row["requestedWorkers"]) == worker] for worker in workers]
                    medians = [quantile(values, .5) for values in grouped]
                    tails = [quantile(values, .95) for values in grouped]
                    axis.plot(workers, medians, marker="o", linestyle=["--", "-"][cells_index], color=COLORS[candidate_index], label=f"{candidate}; {cells}x{cells} cells")
                    axis.fill_between(workers, medians, tails, color=COLORS[candidate_index], alpha=.08)
            axis.set_title(configuration.title())
            axis.set_xticks([1, 2, 4, 8])
            axis.set_xlabel("Requested workers")
            axis.set_ylabel("Complete meshlet/LOD build (ms)")
            axis.grid(axis="y")
            axis.legend(fontsize=8)
        report.save(figure, "HistoricalMeshlets.png", "Archived generated geometry cooker",
            "P50 lines and P50-P95 bands; 6 process samples per point, complete logical geometry hashes checked by the original runner. "
            "Generated grids with cold process/warmup protocol documented in the source evidence; this is not SuperSponza or a renderer frame.")
    report.note(f"Historical plots use archived 2026-11-06 evidence from {root}; they are separate workloads and binaries from production A/B.")
    report.table("Archived comparison executable identities", ["Evidence set", "Candidate", "Executable SHA256"], identities)


def process_labels(records):
    return [f"B{row['block']}/W{row.get('workers', '?')}\nP{row['runId']} (n={row.get('frameCount', '?')})" for row in records]


def in_measured_window(frame, record):
    window = record.get("measuredFrameRange")
    index = number(frame.get("frame", frame.get("frameIndex")))
    if not isinstance(window, dict) or index is None:
        return None
    first, last = number(window.get("first")), number(window.get("last"))
    if first is None or last is None:
        return None
    if first > last:
        raise ValueError("Reversed measured frame range")
    return first <= index <= last


def metric_series(axis, records, extractor, label, color, marker="o", scale=1.0):
    points = [(index, extractor(row)) for index, row in enumerate(records)]
    points = [(index, value * scale) for index, value in points if value is not None]
    if points:
        axis.plot([point[0] for point in points], [point[1] for point in points], marker=marker,
            linestyle="none", color=color, label=f"{label} (n={len(points)})")
    return len(points)


def artifact_path(root, filename):
    # Raw records are produced on Windows. Relative artifact paths remain usable
    # after restoring that directory layout on another plotting host.
    path = Path(str(filename).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def verify_recorded_artifact(report, record, key, path):
    expected = (record.get("artifactSha256") or {}).get(key)
    if expected is not None and report.inputs[str(path.resolve())] != expected:
        raise ValueError(f"Artifact hash differs from Raw.jsonl for P{record['runId']} / {key}: {path}")


def load_frames(report, root, record):
    filename = record.get("artifacts", {}).get("frames")
    if not filename:
        return []
    path = artifact_path(root, filename)
    if not path.exists():
        report.note(f"Per-frame artifact missing for {record['runId']}: {path}")
        return []
    frames = report.read_lines(path)
    verify_recorded_artifact(report, record, "frames", path)
    return frames


def observed_graph_path(graph):
    """Reconstruct a latest-predecessor completion chain from an actual DAG.

    Node wall durations on this dependency chain cannot overlap. Gaps are
    observed ready-to-start intervals, not a claim of worker sleep or pure
    scheduler cost. Separately compute the longest dependency-node-wall path.
    """
    if graph.get("status") != "Succeeded":
        return None
    nodes = {int(node["nodeIndex"]): node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Diagnostic DAG has duplicate IDs")
    if not nodes:
        return None
    submitted, completed = int(graph["submittedNs"]), int(graph["completedNs"])
    predecessors = {key: [] for key in nodes}
    successors = {key: [] for key in nodes}
    for prerequisite, dependent in graph["edges"]:
        if prerequisite not in nodes or dependent not in nodes:
            raise ValueError("Diagnostic DAG edge references a missing node")
        predecessors[dependent].append(prerequisite)
        successors[prerequisite].append(dependent)
        if int(nodes[prerequisite]["finishNs"]) > int(nodes[dependent]["startNs"]):
            raise ValueError("Dependency-linked node intervals overlap; critical-chain stack would be invalid")
    for node in nodes.values():
        if node["state"] != "Succeeded" or not (submitted <= int(node["startNs"]) <= int(node["finishNs"]) <= completed):
            raise ValueError("Successful DAG has incomplete/out-of-envelope node intervals")
    indegree = {key: len(values) for key, values in predecessors.items()}
    ready = [key for key, count in indegree.items() if count == 0]
    longest = {}
    visited = 0
    while ready:
        key = ready.pop()
        longest[key] = int(nodes[key]["finishNs"]) - int(nodes[key]["startNs"]) + max((longest[parent] for parent in predecessors[key]), default=0)
        visited += 1
        for child in successors[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise ValueError("Diagnostic task graph is cyclic")
    terminal = max((key for key in nodes if not successors[key]), key=lambda key: int(nodes[key]["finishNs"]))
    chain = [terminal]
    while predecessors[chain[-1]]:
        chain.append(max(predecessors[chain[-1]], key=lambda key: int(nodes[key]["finishNs"])))
    chain.reverse()
    node_wall, gaps, previous = 0, 0, submitted
    for key in chain:
        node = nodes[key]
        gaps += int(node["startNs"]) - previous
        node_wall += int(node["finishNs"]) - int(node["startNs"])
        previous = int(node["finishNs"])
    return {"chain": chain, "nodes": nodes, "nodeWallNs": node_wall,
        "readyGapNs": gaps, "completionNs": completed - previous,
        "wallNs": completed - submitted, "longestDependencyNodeWallNs": max(longest.values()),
        "submittedNs": submitted}


def task_graph_plots(report, root, records, frames_by_run):
    representatives = []
    for record in records:
        filename = (record.get("artifacts") or {}).get("taskGraphs")
        if filename:
            path = artifact_path(root, filename)
        else:
            capture_name = (record.get("artifacts") or {}).get("capture")
            if not capture_name:
                continue
            capture_path = artifact_path(root, capture_name)
            if not capture_path.exists():
                continue
            capture = report.read_json(capture_path)
            verify_recorded_artifact(report, record, "capture", capture_path)
            if not capture.get("taskGraphDiagnostics"):
                continue
            path = capture_path.parent / capture["taskGraphDiagnostics"]
        if not path.exists():
            report.note(f"TaskGraph diagnostic artifact unavailable for process {record['runId']}: {path}")
            continue
        frames = [frame for frame in frames_by_run[str(record["runId"])] if in_measured_window(frame, record) is True]
        if not frames or any(frame.get("originSteadyNs") is None for frame in frames):
            report.note(f"No steady-clock measured-frame envelope for TaskGraph association in P{record['runId']}.")
            continue
        begin = min(int(frame["originSteadyNs"]) for frame in frames)
        end = max(int(frame["originSteadyNs"]) + round(number(frame["frameMs"]) * 1e6) for frame in frames)
        graphs = report.read_lines(path)
        verify_recorded_artifact(report, record, "taskGraphs", path)
        measured = [graph for graph in graphs if int(graph["submittedNs"]) >= begin and int(graph["completedNs"]) <= end]
        paths = [(graph, observed_graph_path(graph)) for graph in measured]
        paths = [(graph, result) for graph, result in paths if result is not None]
        if paths:
            graph, result = max(paths, key=lambda item: item[1]["wallNs"])
            representatives.append((record, graph, result, len(paths)))
    if not representatives:
        report.note("No dependency-linked TaskGraph intervals inside a measured frame window: DAG completion-chain plots unavailable. "
            "Diagnostic sink runs are separate from performance-off captures.")
        return
    figure, axis = plt.subplots(figsize=(max(11, len(representatives) * .7), 5.2), constrained_layout=True)
    bottom = [0.0] * len(representatives)
    for key, label, color in [("nodeWallNs", "Node wall on observed completion chain", COLORS[0]),
            ("readyGapNs", "Observed ready-to-start gaps", COLORS[1]),
            ("completionNs", "Completion epilogue", COLORS[4])]:
        values = [result[key] / 1e6 for _, _, result, _ in representatives]
        axis.bar(range(len(values)), values, bottom=bottom, label=label, color=color)
        bottom = [lower + value for lower, value in zip(bottom, values)]
    axis.scatter(range(len(representatives)), [result["longestDependencyNodeWallNs"] / 1e6 for _, _, result, _ in representatives],
        color="#17243a", marker="x", label="Longest dependency chain: node wall only", zorder=5)
    axis.set_xticks(range(len(representatives)), [f"P{row['runId']}/W{row.get('workers', '?')}\nE{graph['executionId']}" for row, graph, _, _ in representatives],
        rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("TaskGraph wall time (ms)")
    axis.set_title("Largest observed measured-window graph in each diagnostic process")
    axis.grid(axis="y")
    axis.legend(fontsize=8)
    report.save(figure, "ProductionTaskGraphChains.png", "Diagnostic DAG completion-chain breakdown",
        "One largest-wall graph per process, selected explicitly; not a median or performance-off sample. Actual edges validate non-overlap along each chain. "
        "Stacks partition graph wall into node intervals, observed ready-to-start gaps and completion epilogue; they are not added to inclusive frame scopes. "
        "The black marker is a separately computed longest dependency-node-wall path. Diagnostics may perturb scheduling; "
        "the graph-completed timestamp precedes sink serialization, so this is not a measurement of all observer cost.")
    report.table("Diagnostic graph selection", ["Run", "Workers", "Candidate", "Eligible graphs", "Graph execution", "Name", "Nodes", "Chain nodes", "Graph wall ms"],
        [[row["runId"], row.get("workers", "unavailable"), row["candidate"], count, graph["executionId"], graph["name"],
            len(result["nodes"]), len(result["chain"]), f"{result['wallNs'] / 1e6:.5f}"] for row, graph, result, count in representatives])

    row, graph, result, _ = max(representatives, key=lambda item: item[2]["wallNs"])
    workers = sorted({int(node["workerThreadId"]) for node in result["nodes"].values()})
    positions = {worker: index for index, worker in enumerate(workers)}
    figure, axis = plt.subplots(figsize=(13, max(3.5, len(workers) * .65)), constrained_layout=True)
    for key, node in result["nodes"].items():
        start = (int(node["startNs"]) - result["submittedNs"]) / 1e6
        duration = (int(node["finishNs"]) - int(node["startNs"])) / 1e6
        axis.broken_barh([(start, duration)], (positions[int(node["workerThreadId"])] - .3, .6),
            facecolors=COLORS[1] if key in result["chain"] else COLORS[0], alpha=.85)
        if len(result["nodes"]) <= 24:
            axis.text(start, positions[int(node["workerThreadId"])], f" {key}: {node['name']}", fontsize=7, va="center")
    axis.set_yticks(range(len(workers)), [str(worker) for worker in workers], fontsize=8)
    axis.set_xlabel("Milliseconds since graph submission")
    axis.set_ylabel("Recorded worker-thread identity")
    axis.set_title(f"P{row['runId']} / E{graph['executionId']} / {graph['name']} — orange: completion chain")
    axis.grid(axis="x")
    report.save(figure, "ProductionTaskGraphTimeline.png", "Diagnostic worker intervals and actual DAG chain",
        "Actual start/finish intervals on separate recorded worker identities (TaskSystem hashes std::thread::id; these are not Windows TIDs). "
        "Parallel intervals are shown, not summed. Orange nodes form the latest-predecessor completion chain. "
        "Task gaps include readiness bookkeeping, scheduling and preemption; neither gaps nor node wall are pure CPU execution or measured worker sleep.")


def inspect_render_image(report, capture_path, index, transfer):
    import numpy as np

    capture_path = capture_path.resolve()
    capture = report.read_json(capture_path)
    image = capture.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"Capture contains no exported image: {capture_path}")
    width, height = int(image["width"]), int(image["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid render-image extent")
    image_format = image["format"]
    dtype = {"RGBA8Unorm": np.dtype("u1"), "RGBA16Sfloat": np.dtype("<f2")}.get(image_format)
    if dtype is None:
        raise ValueError(f"Unsupported raw render format: {image_format}")
    path = capture_path.parent / image["file"]
    report.inputs[str(path)] = digest(path)
    pixels = np.fromfile(path, dtype=dtype)
    if pixels.size != width * height * 4 or path.stat().st_size != int(image["bytes"]):
        raise ValueError("Raw image byte count differs from capture metadata")
    pixels = pixels.reshape(height, width, 4).astype(np.float32)
    if image_format == "RGBA8Unorm":
        pixels /= 255.0
    finite = np.isfinite(pixels)
    rgb = pixels[:, :, :3]
    stats = {"capture": str(capture_path), "rawImageSHA256": report.inputs[str(path)],
        "format": image_format, "colorEncoding": image.get("colorEncoding"), "width": width, "height": height,
        "finiteRGBA": bool(finite.all()), "nonfiniteRGBAValues": int((~finite).sum()),
        "nonfiniteRGBValues": int((~np.isfinite(rgb)).sum())}
    if finite.all():
        stats.update({"rgbMinimum": float(rgb.min()), "rgbMaximum": float(rgb.max()), "rgbMean": float(rgb.mean()),
            "alphaMinimum": float(pixels[:, :, 3].min()), "alphaMaximum": float(pixels[:, :, 3].max()),
            "rgbValuesBelowZero": int((rgb < 0).sum()), "rgbValuesAboveOne": int((rgb > 1).sum())})
        selected_transfer = transfer
        if transfer == "auto":
            selected_transfer = "linear-rec709" if image.get("colorEncoding") in {2, 4} else "raw"
        preview = np.clip(rgb, 0, 1)
        if selected_transfer == "linear-rec709":
            preview = np.where(preview <= .0031308, 12.92 * preview, 1.055 * np.power(preview, 1 / 2.4) - .055)
        name = f"RenderPreview{index}.png"
        plt.imsave(report.output / name, preview)
        caption = (f"Native {width}x{height} RGB preview from {image_format}; encoding enum={image.get('colorEncoding', 'unavailable')}; "
            f"preview transfer={selected_transfer}, fixed clipping to raw [0,1], alpha ignored for display but checked for finiteness. "
            "No adaptive exposure or tone mapping. SceneLinear/ExposedLinear basis is not guessed; HDR values outside the clipping interval remain in raw statistics.")
        report.figures.append((name, f"Render output preview {index}", caption))
    else:
        report.note(f"Render image {index} has {stats['nonfiniteRGBAValues']} nonfinite RGBA values; no zero substitution or preview was produced.")
    (report.output / f"RenderImage{index}.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return capture, image, pixels, stats


def compare_render_images(report, images, context):
    import numpy as np

    (capture_a, descriptor_a, pixels_a, stats_a), (capture_b, descriptor_b, pixels_b, stats_b) = images
    if not stats_a["finiteRGBA"] or not stats_b["finiteRGBA"]:
        report.note("Image comparison unavailable because a source image has nonfinite values.")
        return
    for field in ["width", "height", "format", "colorEncoding"]:
        if descriptor_a.get(field) != descriptor_b.get(field):
            raise ValueError(f"Image comparison has different {field}")
    for field in ["scene", "camera", "graph", "outputExtent", "renderExtent"]:
        if field not in capture_a or field not in capture_b or capture_a[field] != capture_b[field]:
            raise ValueError(f"Image comparison lacks an identical effective {field}")
    delta = np.abs(pixels_b.astype(np.float64) - pixels_a.astype(np.float64))
    rgb_delta = delta[:, :, :3]
    result = {"contextSuppliedByCaller": context, "sourceSHA256": [stats_a["rawImageSHA256"], stats_b["rawImageSHA256"]],
        "rawBytesEqual": stats_a["rawImageSHA256"] == stats_b["rawImageSHA256"],
        "rgbMeanAbsoluteDifference": float(rgb_delta.mean()), "rgbMaximumAbsoluteDifference": float(rgb_delta.max()),
        "rgbRootMeanSquareDifference": float(np.sqrt(np.square(rgb_delta).mean())),
        "fractionOfPixelsWithAnyRGBDifference": float(np.any(rgb_delta != 0, axis=2).mean()),
        "maximumAlphaDifference": float(delta[:, :, 3].max()),
        "definition": "Differences in the raw image's coordinate encoding; RGBA8 is normalized to [0,1]. No automatic pass/fail threshold or temporal-equivalence claim."}
    (report.output / "RenderImageComparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    heatmap = rgb_delta.max(axis=2)
    figure, axis = plt.subplots(figsize=(12, 6.8), constrained_layout=True)
    if heatmap.max() > 0:
        image = axis.imshow(heatmap, cmap="magma", vmin=0, vmax=float(heatmap.max()))
        figure.colorbar(image, ax=axis, label="Maximum absolute RGB-channel difference (raw coordinate units)", shrink=.8)
    else:
        axis.imshow(np.zeros_like(heatmap), cmap="gray", vmin=0, vmax=1)
        axis.set_title("Identical raw RGB values: absolute difference = 0")
    axis.set_axis_off()
    report.save(figure, "RenderImageDifference.png", "Render output numerical comparison",
        f"Caller context: {context}. Raw RGB MAE={result['rgbMeanAbsoluteDifference']:.7g}; "
        f"maximum={result['rgbMaximumAbsoluteDifference']:.7g}; RMSE={result['rgbRootMeanSquareDifference']:.7g}. "
        "Heatmap normalization uses the observed maximum and is not an acceptance threshold. Temporal algorithms, frame seeds, exposure and priming require caller control; "
        "different pixels are not automatically classified as a regression.")


def production_plots(report, root):
    metadata = report.read_json(root / "Metadata.json")
    if metadata.get("status") != "complete":
        report.note(f"Production collection status is {metadata.get('status', 'unavailable')}; this is a partial diagnostic view, not a completed A/B acceptance result.")
    raw = report.read_lines(root / "Raw.jsonl")
    purpose = metadata.get("purpose", "unavailable")
    instrumented = metadata.get("diagnostics") == "on" or (metadata.get("config") or {}).get("diagnostics") is True
    if purpose == "validation":
        report.plot_context = "VALIDATION telemetry — not a performance A/B or FPS result"
    elif purpose == "performance" and instrumented:
        report.plot_context = "INSTRUMENTED diagnostic run — observer overhead included; separate from performance-off data"
    elif purpose == "performance":
        report.plot_context = "PERFORMANCE run — per-process/block units; warmup/prewarm excluded"
    else:
        report.plot_context = "CAPTURE PURPOSE UNAVAILABLE — diagnostic view only"
    report.note(f"Production input purpose={purpose}; scheduling diagnostics={metadata.get('diagnostics', 'unavailable')}; "
        f"app diagnostic sink={(metadata.get('config') or {}).get('diagnostics', 'unavailable')}. Validation and performance populations are not mixed.")
    records = [row for row in raw if row.get("phase") == "measured" and row.get("status") == "passed"
        and row.get("purpose", purpose) == purpose]
    excluded = [row for row in raw if row not in records]
    report.note(f"Production: {len(records)} passed measured processes; {len(excluded)} prewarm/failed/incomplete records excluded from latency comparisons. "
        "Frames are correlated within a process; per-process quantiles are plotted individually, never pooled as independent experiments.")
    if not records:
        report.note("No passed measured production processes: production plots unavailable.")
        return
    # Raw lines are chronological. Ordinal resets in each ABBA/worker block,
    # so sorting by ordinal would fabricate a different invocation sequence.
    candidates = sorted({str(row["candidate"]) for row in records})
    colors = {candidate: COLORS[index % len(COLORS)] for index, candidate in enumerate(candidates)}
    labels = process_labels(records)
    nlabels = ", ".join(f"{candidate}: {sum(row['candidate'] == candidate for row in records)} processes" for candidate in candidates)
    worker_limits = sorted({int(row["workers"]) for row in records if row.get("workers") is not None})
    report.note(f"Production worker limits: {worker_limits or 'unavailable'}; W labels identify each limit. "
        "No results are averaged across different worker limits, blocks or executable candidates.")
    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(records) * .36), 8), sharex=True, constrained_layout=True)
    for axis, field, title in zip(axes, ["frameMs", "gpuFrameMs"], ["Frame wall time", "Reported GPU frame envelope"]):
        count = 0
        for candidate in candidates:
            for statistic, marker in [("p50", "o"), ("p95", "^"), ("p99", "x")]:
                count += metric_series(axis, records,
                    lambda row: distribution(row, field, statistic) if row["candidate"] == candidate else None,
                    f"{candidate} {statistic.upper()}", colors[candidate], marker)
        if not count:
            axis.text(.5, .5, "Unavailable: metric not recorded", transform=axis.transAxes, ha="center", va="center")
        if field == "frameMs" and report.budgets["frameMs"] is not None:
            axis.axhline(report.budgets["frameMs"], color="#bd3340", linestyle="--", label=f"Supplied budget {report.budgets['frameMs']:g} ms")
        axis.set_title(title)
        axis.set_ylabel("Milliseconds per frame")
        axis.grid(axis="y")
        if count:
            axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("Block and fresh process identity, in actual invocation order")
    report.save(figure, "ProductionFrameBlocks.png", "Production frame distributions by process/block",
        f"{nlabels}. Every marker is one process quantile of its measured frames; P95/P99 are not confidence bounds. "
        "Missing GPU data is unavailable, not zero. " + ("No frame budget was supplied." if report.budgets["frameMs"] is None else "Dashed line is the explicitly supplied frame budget."))

    frames_by_run = {str(row["runId"]): load_frames(report, root, row) for row in records}
    with_frames = [row for row in records if frames_by_run[str(row["runId"])]]
    if with_frames:
        all_frame_values = [number(frame.get("frameMs")) for row in with_frames for frame in frames_by_run[str(row["runId"])]]
        maximum = max((value for value in all_frame_values if value is not None), default=None)
        for page, start in enumerate(range(0, len(with_frames), 8), 1):
            page_rows = with_frames[start:start + 8]
            figure, axes = plt.subplots(len(page_rows), 1, figsize=(13, max(3.2, len(page_rows) * 1.9)), squeeze=False, constrained_layout=True)
            for axis, row in zip(axes.flat, page_rows):
                frames = frames_by_run[str(row["runId"])]
                points = [(number(frame.get("frame", frame.get("frameIndex"))), number(frame.get("frameMs"))) for frame in frames]
                points = [(index, value) for index, value in points if index is not None and value is not None]
                if points:
                    axis.plot([item[0] for item in points], [item[1] for item in points], linewidth=.8, color=colors[row["candidate"]])
                window = row.get("measuredFrameRange")
                if isinstance(window, dict) and number(window.get("first")) is not None and number(window.get("last")) is not None:
                    axis.axvspan(number(window["first"]), number(window["last"]), color=colors[row["candidate"]], alpha=.08, label="Runner measured frame range")
                    axis.legend(fontsize=7, loc="upper right")
                axis.set_title(f"{row['candidate']} / W{row.get('workers', '?')} / block {row['block']} / P{row['runId']} / n={len(points)} recorded frames", loc="left", fontsize=10)
                axis.set_ylabel("Frame wall (ms)")
                if maximum is not None and maximum > 0:
                    axis.set_ylim(0, maximum * 1.05)
                axis.grid(axis="y")
                if report.budgets["frameMs"] is not None:
                    axis.axhline(report.budgets["frameMs"], color="#bd3340", linestyle="--", linewidth=.8)
            axes[-1, 0].set_xlabel("Recorded frame index (resets with each process)")
            report.save(figure, f"ProductionFrameSeries{page:02}.png", f"Production recorded frame time series, page {page}",
                "Every recorded frame is shown in its own process, with common y limits across all pages; no smoothing, interpolation, rebasing or cross-process concatenation. "
                "Shading uses the runner's explicit measuredFrameRange when available; absent windows are not guessed. "
                "The per-process Raw.frameMs distribution remains the measured-window authority.")
    else:
        report.note("Per-frame artifacts unavailable: frame time-series plot omitted rather than synthesized from quantiles.")

    scopes = sorted({path for row in records for path in (row.get("cpuScopesMs") or {})})
    def scope_rank(path):
        values = [distribution({"scope": (row.get("cpuScopesMs") or {}).get(path)}, "scope") for row in records]
        return max((value for value in values if value is not None), default=-math.inf)

    ranking = sorted(scopes, key=scope_rank, reverse=True)[:16]
    if ranking:
        figure, axis = plt.subplots(figsize=(12.5, max(5, len(ranking) * .42)), constrained_layout=True)
        cohorts = sorted({(str(row["candidate"]), row.get("workers")) for row in records})
        markers = ["o", "s", "^", "x", "d", "+"]
        for cohort_index, (candidate, workers) in enumerate(cohorts):
            candidate_labeled = False
            for scope_index, scope in enumerate(ranking):
                values = [distribution({"scope": (row.get("cpuScopesMs") or {}).get(scope)}, "scope") for row in records
                    if row["candidate"] == candidate and row.get("workers") == workers]
                values = [value for value in values if value is not None]
                if values:
                    y = scope_index + (cohort_index - (len(cohorts) - 1) / 2) * min(.22, .75 / len(cohorts))
                    axis.scatter(values, [y] * len(values), color=colors[candidate], s=24, alpha=.65,
                        marker=markers[(worker_limits.index(workers) if workers in worker_limits else 0) % len(markers)],
                        label=f"{candidate} / W{workers}" if not candidate_labeled else None)
                    candidate_labeled = True
        axis.set_yticks(range(len(ranking)), ranking, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Per-process scope P50 wall duration (ms); overlapping scopes are NOT additive")
        axis.set_title("CPU scope durations: inclusive/overlapping, not a DAG critical path")
        axis.grid(axis="x")
        axis.legend(fontsize=8, ncol=2)
        report.save(figure, "ProductionCPUScopes.png", "Production CPU scope wall durations",
            f"{nlabels}; each dot is one process's scope-occurrence median, which may have a different sample count than frameCount. "
            f"Top {len(ranking)} paths ranked by their largest process median. "
            "Nested, parallel and idle scopes can overlap; no scope durations are summed, subtracted into CPU work, or relabeled as a critical path.")
        populations = []
        for scope in ranking:
            for candidate, workers in cohorts:
                entries = [(row["runId"], (row.get("cpuScopesMs") or {}).get(scope)) for row in records
                    if row["candidate"] == candidate and row.get("workers") == workers]
                entries = [(run_id, entry) for run_id, entry in entries if isinstance(entry, dict)]
                if entries:
                    populations.append([scope, candidate, workers, "; ".join(f"P{run_id}: n={entry.get('samples', 'unavailable')}" for run_id, entry in entries)])
        report.table("CPU scope-occurrence populations", ["Scope path", "Candidate", "Workers", "Available occurrences per process"], populations)
    report.note("A TaskGraph/DAG critical path is unavailable unless dependency-linked task intervals are separately recorded. "
        "Inclusive profiler scope durations alone do not establish that path.")

    path_frames = []
    for row in records:
        usable = []
        for frame in frames_by_run[str(row["runId"])]:
            if in_measured_window(frame, row) is not True:
                continue
            values = frame_path_segments(frame)
            if values is not None:
                usable.append(values)
        if usable:
            inventory = set(usable[0])
            if any(set(frame) != inventory for frame in usable):
                report.note(f"Coordinator path inventory varies in {row['runId']}; stacked chart omitted for this process (missing phases not filled with zero).")
            else:
                path_frames.append((row, {name: statistics.mean(frame[name] for frame in usable) for name in sorted(inventory)}, len(usable)))
    if path_frames:
        figure, axis = plt.subplots(figsize=(max(10, len(path_frames) * .65), 5), constrained_layout=True)
        inventory = sorted({name for _, values, _ in path_frames for name in values})
        if all(set(values) == set(inventory) for _, values, _ in path_frames):
            bottom = [0.0] * len(path_frames)
            for index, name in enumerate(inventory):
                values = [values[name] for _, values, _ in path_frames]
                axis.bar(range(len(path_frames)), values, bottom=bottom, label=name, color=COLORS[index % len(COLORS)])
                bottom = [lower + value for lower, value in zip(bottom, values)]
            axis.set_xticks(range(len(path_frames)), [f"P{row['runId']}/W{row.get('workers', '?')}\nn={count}" for row, _, count in path_frames], rotation=45, ha="right", fontsize=8)
            axis.set_ylabel("Mean coordinator frame wall time (ms)")
            axis.legend(fontsize=8)
            report.save(figure, "ProductionCoordinatorPath.png", "Production non-overlapping coordinator frame path",
                "Only explicit producer-contracted intervals are used, with non-overlap/envelope validation. Stacks use means from the same complete frames, "
                "not sums of P50s. Unattributed frame envelope is a measured duration difference, not worker idle or useful CPU work. This is not a reconstructed TaskGraph longest path.")
        else:
            plt.close(figure)
            report.note("Coordinator phase inventory differs across processes: no missing phases were converted to zero; stacked comparison omitted.")
    else:
        report.note("No contracted, non-overlapping coordinator intervals: coordinator frame-path breakdown unavailable.")

    task_graph_plots(report, root, records, frames_by_run)

    figure, axes = plt.subplots(3, 1, figsize=(max(12, len(records) * .36), 10), sharex=True, constrained_layout=True)
    memory_fields = [
        ("lifetimePeakPrivateBytes", "Observed OS commit high-water", "o"),
        ("sampledPeakPrivateBytes", "Sampled peak private", "^"),
        ("lifetimePeakWorkingSetBytes", "Observed OS working-set high-water", "s"),
        ("sampledPeakWorkingSetBytes", "Sampled peak working set", "x"),
    ]
    for index, (field, label, marker) in enumerate(memory_fields):
        metric_series(axes[0], records, lambda row, field=field: number((row.get("processResources") or {}).get(field)), label, COLORS[index], marker, 1 / (1 << 20))
    for index, (field, label) in enumerate([
            ("deviceLocalAllocatedSampledPeakBytes", "Device-local allocated (sample peak)"),
            ("deviceLocalBlockSampledPeakBytes", "Device-local blocks (sample peak)"),
            ("deviceLocalUsageSampledPeakBytes", "Reported device-local usage (sample peak)")]):
        metric_series(axes[1], records, lambda row, field=field: number((row.get("deviceMemory") or {}).get(field)), label, COLORS[index], ["o", "^", "x"][index], 1 / (1 << 20))
    metric_series(axes[2], records, lambda row: number((row.get("processResources") or {}).get("sampledPeakThreads")), "Process sampled peak threads", COLORS[0])
    for axis, title, unit in zip(axes, ["Process memory peaks", "Device-local memory: separate accounting scopes", "Process thread-count peak"], ["MiB", "MiB", "Threads"]):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8, ncol=2)
        else:
            axis.text(.5, .5, "Unavailable: no recorded metric", transform=axis.transAxes, ha="center")
    for axis, budget_key in [(axes[0], "cpuMemoryMiB"), (axes[1], "vramMiB")]:
        if report.budgets[budget_key] is not None:
            axis.axhline(report.budgets[budget_key], color="#bd3340", linestyle="--", label=f"Supplied budget {report.budgets[budget_key]:g} MiB")
            axis.legend(fontsize=8, ncol=2)
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
    report.save(figure, "ProductionResourcePeaks.png", "Production memory and process-thread peaks",
        f"{nlabels}; one marker per available process metric. OS high-waters include startup only through the last successful sample; "
        "current-value sampled peaks depend on poll cadence/window. "
        "Device-local allocated/block/usage measures are not additive or automatically process-private. Missing values remain absent. "
        "No RAM/VRAM budget is assumed when none is supplied; whole-device capacity is not an application budget.")

    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(records) * .36), 7), sharex=True, constrained_layout=True)
    tracked_count = 0
    for axis, fields, title in [
            (axes[0], [("peakTrackedDeviceLocalBytes", "Event high-water: device-local", "o"),
                ("trackedDeviceLocalBytes", "Sampled current maximum: device-local", "x")], "RHI-tracked device-local allocations"),
            (axes[1], [("peakTrackedAllocationBytes", "Event high-water: host + device", "o"),
                ("trackedAllocationBytes", "Sampled current maximum: host + device", "x")], "All RHI-tracked allocations (not VRAM-only)")]:
        for index, (field, label, marker) in enumerate(fields):
            tracked_count += metric_series(axis, records, lambda row, field=field: number((row.get("deviceMemory") or {}).get(field)),
                label, COLORS[index], marker, 1 / (1 << 20))
        axis.set_title(title)
        axis.set_ylabel("MiB")
        axis.grid(axis="y")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "Unavailable: tracked high-water counters not exported", transform=axis.transAxes, ha="center")
    if tracked_count:
        axes[-1].set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
        report.save(figure, "ProductionRHIHighWaters.png", "Production RHI allocation high-waters",
            "Event high-waters are simultaneous totals maintained at RHI allocation/free events, through the last queried snapshot. "
            "They cover RHI-tracked allocations, excluding untracked external SDK/driver memory. Current-value sampled maxima are separate. "
            "Host+device totals are not VRAM-only; domains and these totals are not added together. Missing counters remain unavailable.")
    else:
        plt.close(figure)
        report.note("Global RHI event high-water counters unavailable in this capture revision; heap sample peaks are not relabeled as exact lifetime peaks.")

    figure, axis = plt.subplots(figsize=(max(12, len(records) * .36), 3.8), constrained_layout=True)
    count = 0
    for candidate in candidates:
        count += metric_series(axis, records,
            lambda row: number((row.get("processResources") or {}).get("averageCpuCores")) if row["candidate"] == candidate else None,
            candidate, colors[candidate])
    if count:
        axis.set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylabel("Process CPU time / process wall time (cores)")
        axis.set_title("Process-lifetime average CPU occupancy; not a sampled peak")
        axis.grid(axis="y")
        axis.legend()
        report.save(figure, "ProductionCPUOccupancy.png", "Production CPU occupancy",
            "Each available marker is process CPU time divided by lifetime wall time, across all process threads. "
            "It includes loading/warmup when those fall in the producer's process window. It is neither worker utilization nor a CPU usage peak. "
            "No host-wide CPU utilization or peak CPU usage is synthesized.")
    else:
        plt.close(figure)
        report.note("Process CPU occupancy unavailable: no CPU-time/wall-time ratio recorded. CPU usage peak is not inferred from missing samples.")

    domains = sorted({domain for row in records for domain in ((row.get("deviceMemory") or {}).get("domainAllocationPeakBytes") or {})})
    if domains:
        figure, axis = plt.subplots(figsize=(12, max(4.5, len(domains) * .45)), constrained_layout=True)
        domain_cohorts = sorted({(str(row["candidate"]), row.get("workers")) for row in records})
        for cohort_index, (candidate, workers) in enumerate(domain_cohorts):
            labeled = False
            for domain_index, domain in enumerate(domains):
                values = [number(((row.get("deviceMemory") or {}).get("domainAllocationPeakBytes") or {}).get(domain))
                    for row in records if row["candidate"] == candidate and row.get("workers") == workers]
                values = [value / (1 << 20) for value in values if value is not None]
                if values:
                    axis.scatter(values, [domain_index + (cohort_index - (len(domain_cohorts) - 1) / 2) * min(.22, .75 / len(domain_cohorts))] * len(values),
                        color=colors[candidate], alpha=.65,
                        marker=["o", "s", "^", "x", "d", "+"][(worker_limits.index(workers) if workers in worker_limits else 0) % 6],
                        label=f"{candidate} / W{workers}" if not labeled else None)
                    labeled = True
        axis.set_yticks(range(len(domains)), domains, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel("Per-process domain allocation peak (MiB)")
        axis.grid(axis="x")
        axis.legend(fontsize=8, ncol=2)
        report.save(figure, "ProductionMemoryDomains.png", "Production domain allocation peaks",
            "Each point is one process's reported peak for that allocation domain. Peaks can occur on different frames. "
            "RHI domains may include host and device allocations; this is not a VRAM-only partition. "
            "Domain definitions and ownership come from the producer; they are not summed into a simultaneous total or a memory budget.")

    identity_rows = []
    executables = metadata.get("executables") or {}
    for candidate in candidates:
        identity = executables.get(candidate, {})
        identity_rows.append([candidate, identity.get("path", "unavailable") if isinstance(identity, dict) else identity,
            identity.get("sha256", "unavailable") if isinstance(identity, dict) else "unavailable"])
    report.table("Production comparison identity", ["Candidate", "Executable", "SHA256"], identity_rows)
    workload_rows = []
    for workers in worker_limits:
        identity = next(row.get("effectiveIdentity", {}) for row in records if row.get("workers") == workers)
        workload_rows.append([workers, identity.get("scene", "unavailable"), identity.get("graph", "unavailable"),
            identity.get("camera", "unavailable"), identity.get("outputExtent", "unavailable"),
            identity.get("renderExtent", "unavailable"), identity.get("recordingWorkerLimit", "unavailable"),
            identity.get("validationRequested", "unavailable")])
    report.table("Production effective workload", ["Workers", "Scene", "Graph", "Camera", "Output extent", "Render extent", "Recording limit", "Validation"], workload_rows)
    report.table("Production process populations", ["Run", "Block", "Workers", "Candidate", "Measured frames", "Frame P50 ms", "Frame P95 ms", "Frame P99 ms"],
        [[row["runId"], row["block"], row.get("workers", "unavailable"), row["candidate"], row.get("frameCount", "unavailable")]
            + [f"{value:.4f}" if value is not None else "unavailable" for value in [distribution(row, "frameMs", stat) for stat in ["p50", "p95", "p99"]]] for row in records])


def production_cook_plots(report, root):
    metadata = report.read_json(root / "Metadata.json")
    records = [row for row in report.read_lines(root / "Raw.jsonl") if row.get("status") == "passed" and row.get("phase") == "measured"]
    report.plot_context = "CPU-ONLY PRODUCTION COOK — new output each process; OS/source caches retained; not frames/FPS"
    if metadata.get("status") != "complete":
        report.note(f"Production cook collection status={metadata.get('status', 'unavailable')}; partial diagnostic evidence only.")
    if not records:
        report.note("No passed production cook processes; cold-cook figures unavailable.")
        return
    labels = [f"B{row['block']}/W{row['workers']}\nP{row['runId']}" for row in records]
    candidates = sorted({row["candidate"] for row in records})
    colors = {candidate: COLORS[index % len(COLORS)] for index, candidate in enumerate(candidates)}
    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(records) * .36), 7.5), sharex=True, constrained_layout=True)
    for candidate in candidates:
        metric_series(axes[0], records, lambda row: number((row.get("processResources") or {}).get("processWallNs")) if row["candidate"] == candidate else None,
            f"{candidate}: process lifetime wall", colors[candidate], "o", 1e-9)
        metric_series(axes[0], records, lambda row: number(row.get("cookElapsedSeconds")) if row["candidate"] == candidate else None,
            f"{candidate}: cooker reported elapsed", colors[candidate], "x")
        metric_series(axes[1], records, lambda row: number((row.get("processResources") or {}).get("averageCpuCores")) if row["candidate"] == candidate else None,
            candidate, colors[candidate])
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Complete cook process; validation subprocess excluded")
    axes[1].set_ylabel("Process CPU / process wall (cores)")
    axes[1].set_title("Process-average CPU occupancy; not worker utilization or CPU peak")
    for axis in axes:
        axis.grid(axis="y")
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
    report.save(figure, "ProductionCookProcesses.png", "Production cooker process samples",
        f"{len(records)} passed cook processes, shown separately in source order. Worker counts are not pooled. "
        "Cook process wall and application elapsed have different boundaries; all-page/source-attribute validation is a separate process and excluded. "
        "Fresh output does not imply a flushed OS file cache.")

    figure, axes = plt.subplots(2, 1, figsize=(max(12, len(records) * .36), 7), sharex=True, constrained_layout=True)
    for index, (field, label, marker) in enumerate([
            ("lifetimePeakPrivateBytes", "Observed OS commit high-water", "o"),
            ("sampledPeakPrivateBytes", "Sampled private maximum", "^"),
            ("lifetimePeakWorkingSetBytes", "Observed OS working-set high-water", "s")]):
        metric_series(axes[0], records, lambda row, field=field: number((row.get("processResources") or {}).get(field)), label, COLORS[index], marker, 1 / (1 << 20))
    metric_series(axes[1], records, lambda row: number((row.get("processResources") or {}).get("sampledPeakThreads")),
        "Cook sampled peak process threads", COLORS[0])
    axes[0].set_ylabel("MiB")
    axes[1].set_ylabel("Threads")
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for axis in axes:
        axis.grid(axis="y")
        axis.legend(fontsize=8)
    axes[-1].set_xticks(range(len(records)), labels, rotation=45, ha="right", fontsize=8)
    report.save(figure, "ProductionCookResources.png", "Production cooker process resources",
        "Only cook-process OS counters are plotted, through the last successful sample. Separate validation-process peaks are not merged. "
        "The cooker is CPU-only; aligned device-payload byte counts are storage/accounting estimates, not measured VRAM.")
    report.table("Production cook populations and correctness", ["Run", "Block", "Workers", "Candidate", "Fresh output", "Payload bytes", "Payload SHA256", "Validation"],
        [[row["runId"], row["block"], row["workers"], row["candidate"], row.get("freshOutput", "unavailable"),
            row.get("payloadBytes", "unavailable"), row.get("payloadSha256", "unavailable"), row.get("payloadValidation", "unavailable")] for row in records])
    report.table("Production cook executable identity", ["Candidate", "Executable", "SHA256"],
        [[candidate, entry.get("path", "unavailable"), entry.get("sha256", "unavailable")] for candidate, entry in (metadata.get("executables") or {}).items()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, help="Archived CPUJobs20261106 evidence root (read only)")
    parser.add_argument("--production", type=Path, help="Production Metadata.json/Raw.jsonl evidence directory (read only)")
    parser.add_argument("--production-cook", type=Path, help="MeasureProductionMeshlets.py evidence directory (CPU-only; read only)")
    parser.add_argument("--output", type=Path, required=True, help="New, empty report directory")
    parser.add_argument("--title", default="CPU job system: measured evidence")
    parser.add_argument("--frame-budget-ms", type=float)
    parser.add_argument("--cpu-memory-budget-mib", type=float)
    parser.add_argument("--vram-budget-mib", type=float)
    parser.add_argument("--image-capture", type=Path, action="append", default=[], help="Optional Capture.json with FinalColor.bin to inspect/preview")
    parser.add_argument("--preview-transfer", choices=["auto", "raw", "linear-rec709"], default="auto")
    parser.add_argument("--compare-images", action="store_true", help="Explicit numerical comparison of two image captures after caller-controlled priming")
    parser.add_argument("--comparison-context", help="Required with --compare-images: fixed camera/priming/temporal controls, recorded verbatim")
    args = parser.parse_args()
    if not args.historical and not args.production and not args.production_cook and not args.image_capture:
        parser.error("Supply --historical, --production, --production-cook and/or --image-capture evidence")
    if args.compare_images and (len(args.image_capture) != 2 or not args.comparison_context):
        parser.error("Image comparison needs exactly two --image-capture files and an explicit --comparison-context")
    budgets = {"frameMs": args.frame_budget_ms, "cpuMemoryMiB": args.cpu_memory_budget_mib, "vramMiB": args.vram_budget_mib}
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in budgets.values()):
        parser.error("Supplied budgets must be finite and positive; omit unavailable budgets")
    args.output = args.output.resolve()
    if args.historical and (args.output == args.historical.resolve() or args.output.is_relative_to(args.historical.resolve())):
        parser.error("Write to a new report directory outside the archived evidence tree")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; existing reports/evidence are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    report = Report(args.output, args.title, budgets)
    report.note("P50/P95/P99 summarize the recorded sample population, not confidence intervals. Missing data is never replaced with zero.")
    report.note("Budgets: " + "; ".join(f"{name}={value:g}" if value is not None else f"{name}=not supplied" for name, value in budgets.items()))
    if args.historical:
        historical_plots(report, args.historical.resolve())
    if args.production_cook:
        production_cook_plots(report, args.production_cook.resolve())
        report.plot_context = ""
    if args.production:
        production_plots(report, args.production.resolve())
        report.plot_context = ""
    images = [inspect_render_image(report, path, index, args.preview_transfer) for index, path in enumerate(args.image_capture, 1)]
    if args.compare_images:
        compare_render_images(report, images, args.comparison_context)
    report.finish()
    if any(not image[3]["finiteRGBA"] for image in images):
        raise ValueError("A render image contains nonfinite values; see the generated image statistics/report")
    print(f"Wrote {len(report.figures)} figures and HTML/Markdown to {args.output}")


if __name__ == "__main__":
    main()
