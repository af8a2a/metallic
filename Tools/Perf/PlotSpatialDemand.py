"""Plot verified spatial-demand evidence into a fresh directory outside the evidence package."""
import argparse
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import MeasureSpatialDemand as measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require = measurement.require
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.evidence.resolve()), "Choose a new report directory outside evidence")
    result = measurement.verify(argparse.Namespace(output=args.evidence))
    manifest = measurement.read(args.evidence / "Manifest.json")
    rows = []
    for record in manifest["runs"]:
        if record["stage"] != "quality":
            row = measurement.load_run(args.evidence / Path(record["directory"]).name, record["arm"], False, manifest["protocol"])
            row["stage"] = record["stage"]
            rows.append(row)
    args.output.mkdir(parents=True)
    measurement.save(args.output / "Analysis.json", result)
    measurement.save(args.output / "PlotData.json", rows)
    shutil.copyfile(__file__, args.output / "plot_report.py")
    fig, axes = plt.subplots(len(measurement.METRICS), 2, figsize=(13, 3.3 * len(measurement.METRICS)), constrained_layout=True)
    for index, metric in enumerate(measurement.METRICS):
        ax, interval_ax = axes[index]
        for arm, color in (("A", "#3973ac"), ("B", "#d27e2b")):
            points = [(i + 1, row["timings"][metric]["median"]) for i, row in enumerate(rows) if row["arm"] == arm]
            ax.scatter(*zip(*points), label=f"{arm}: {'baseline' if arm == 'A' else 'spatial'} (n={len(points)} processes)", color=color)
        ax.set(xlabel="Independent process (ABBA order)", ylabel=f"{metric} (ms)", title=f"{metric}: median of 64 frozen-source frames")
        if "confirmation" in result["stages"]:
            first_confirmation = next(i for i, row in enumerate(rows) if row["stage"] == "confirmation")
            ax.axvline(first_confirmation + .5, color="gray", linestyle="--", label="Independent confirmation")
        ax.legend(fontsize=8)
        for stage_index, (stage, summary) in enumerate(result["stages"].items()):
            g = summary["gains"][metric]
            blocks = len(g["values"])
            interval_ax.scatter((np.arange(blocks) - (blocks - 1) / 2) * .07 + stage_index,
                                np.array(g["values"]) * 100, color="#888888", s=18)
            interval_ax.errorbar(stage_index, g["estimate"] * 100,
                                 yerr=[[100 * (g["estimate"] - g["lower95"])], [100 * (g["upper95"] - g["estimate"])]],
                                 fmt="o", capsize=5, label=f"{stage}: n={blocks} blocks")
        interval_ax.axhline(0, color="gray", linestyle=":")
        interval_ax.set(xticks=range(len(result["stages"])), xticklabels=list(result["stages"]),
                        ylabel="Reduction B vs A (%)", title="Block log-ratio estimate and 95% Student t CI")
        interval_ax.legend(fontsize=8)
    fig.suptitle("MiniZorah frozen residency; diagnostics excluded; all samples retained", fontsize=13)
    fig.savefig(args.output / "timings.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key in zip(axes, ("visitedNodes", "testedGroups", "allGroupTests")):
        for arm, color in (("A", "#3973ac"), ("B", "#d27e2b")):
            points = [(i + 1, row["decoded"][0]["demand"][key]) for i, row in enumerate(rows) if row["arm"] == arm]
            ax.scatter(*zip(*points), color=color, label=arm)
        ax.set(title=key, xlabel="Independent process", ylabel="Count (deferred diagnostic)")
        ax.legend()
    fig.suptitle("Traversal counts: deferred readback, separate from timing")
    fig.savefig(args.output / "traversal.png", dpi=150)
    plt.close(fig)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "manifestSha256": measurement.file_hash(args.evidence / "Manifest.json")}
    measurement.save(args.output / "Versions.json", versions)
    stage_counts = ", ".join(f"{stage}: n={summary['gains']['streamTraversalMs']['blocks']} ABBA blocks"
                             for stage, summary in result["stages"].items())
    primary = list(result["stages"].values())[-1]["primaryMetric"]
    (args.output / "report.md").write_text(
        f"# MiniZorah spatial demand comparison\n\nEvidence: `{args.evidence.resolve()}`.\n\n"
        f"Captured stages: {', '.join(result['stages'])}. "
        f"Final assessment: {list(result['stages'].values())[-1]['effectAssessment']}. "
        "A declared regression stop does not run an acceptance confirmation stage.\n\n"
        "Ordinary timing: 64 samples per process after 16 settle frames, converged frozen geometry/texture source, "
        f"canonical owner reset. Independent statistical units: {stage_counts}. Primary metric: {primary}. "
        "All frames retained; no outlier removal. "
        "Error bars are 95% Student t intervals of block log time ratios. Positive reductions favor B. "
        "Cooperative lock and recorded GPU telemetry do not prove system-wide GPU exclusivity.\n\n"
        "![Timings](timings.png)\n\n![Traversal counts](traversal.png)\n\n"
        "See Analysis.json for recomputed exact image/semantic gates and stage outcomes; PlotData.json retains field mappings and frame values. "
        "This fixed view does not establish roaming or full-scene performance.\n\n"
        f"Reproduce from repository: `python -B Tools/Perf/PlotSpatialDemand.py {args.evidence.resolve()} --output <new-directory>`\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
