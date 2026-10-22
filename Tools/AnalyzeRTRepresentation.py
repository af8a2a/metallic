"""Review production RT representation A/B captures and untimed image readbacks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCOPES = ["Detail demand", "LOD frontier", "LOD mask", "LOD emit",
          "RT representation classify", "RT representation cut",
          "BLAS cut compare", "BLAS build", "Async TLAS build", "Stream traversal",
          "Shadows (RayTracedShadowPass)", "RenderGraph GPU envelope"]
COUNTERS = ["blasBuildCount", "blasLiveClusterReferences", "blasArenaUsedBytes",
            "blasArenaRepack", "blasOverflowCount", "blasStorageRejected",
            "blasRepresentationCount", "blasSharingInstances", "blasMergedInstances",
            "blasCachedInstances", "blasRepresentationGroupOverflow",
            "blasCacheDependencyInvalidations", "blasSemanticChangeCount", "blasBuildBudgetRejected",
            "geometryBytes", "clasAllocatedBytes", "textureResidentBytes", "pendingPages", "uploadBytes"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stats(values):
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("Nonfinite measurement")
    return dict(n=len(data), mean=float(np.mean(data)), median=float(np.median(data)),
                p95=float(np.quantile(data, .95)), minimum=float(np.min(data)), maximum=float(np.max(data)))


def load_run(root):
    capture = read(root / "run1/Capture.json")
    manifest = read(root / "Manifest.json")
    if capture["status"] != "capture_complete" or capture["missingGpuFrames"]:
        raise ValueError(f"Incomplete capture: {root}")
    if capture.get("diagnosticRun") or capture.get("validationRequested"):
        raise ValueError(f"Instrumented performance capture: {root}")
    rows = [json.loads(line) for line in (root / "run1/Frames.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    if not rows or len(rows) != capture["frames"]:
        raise ValueError(f"Frame count mismatch: {root}")
    scopes = {s["id"]: s["path"].split("/")[-1] for s in capture["scopes"]}
    values = {name: [] for name in SCOPES + COUNTERS + ["Frame CPU"]}
    for row in rows:
        if row.get("diagnostic"):
            raise ValueError("Diagnostic frame mixed into performance capture")
        values["Frame CPU"].append(row["frameMs"])
        seen = set()
        for scope in row["scopes"]:
            name = scopes[scope["id"]]
            if name in SCOPES and scope["gpuMs"] is not None:
                if name in seen:
                    raise ValueError(f"Ambiguous scope {name}")
                values[name].append(scope["gpuMs"])
                seen.add(name)
        samples = [s for s in row["streaming"] if s["pass"] == "VBuffer"]
        if len(samples) != 1 or not samples[0]["blasFeedbackAvailable"]:
            raise ValueError("Missing BLAS feedback")
        for name in COUNTERS:
            if name in samples[0]:
                values[name].append(samples[0][name])
    for name in ["BLAS build", "LOD frontier", "Detail demand"]:
        if len(values[name]) != len(rows):
            raise ValueError(f"Missing required timing {name}")
    source = root / "run1/Frames.jsonl"
    return dict(path=str(root.resolve()), frames=len(rows), manifest=manifest,
                sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                measurements={k: stats(v) for k, v in values.items()}), values


def image_arrays(directory):
    result = {}
    for item in read(directory / "Images.json"):
        kind, channels = item["format"].rsplit("x", 1)
        dtype = {"float16": "<f2", "float32": "<f4", "unorm8": "u1", "srgb8": "u1"}[kind]
        data = np.fromfile(directory / item["file"], dtype=dtype).reshape(item["height"], item["width"], int(channels)).astype(np.float32)
        if kind in ("unorm8", "srgb8"):
            data /= 255
        result[item["resource"]] = data
    return result


def compare_images(a, b, output):
    left, right = image_arrays(a), image_arrays(b)
    if left.keys() != right.keys():
        raise ValueError("Quality output identities differ")
    report = {}
    for name, x in left.items():
        y = right[name]
        if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError(f"Invalid image: {name}")
        channels = min(x.shape[-1], 3)
        x, y = x[..., :channels], y[..., :channels]
        delta = np.abs(x - y)
        report[name] = dict(shape=list(x.shape), mae=float(delta.mean()), p99=float(np.quantile(delta, .99)),
                            maximum=float(delta.max()), exactPixels=float(np.mean(np.all(delta == 0, axis=-1))))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        def preview(v):
            if channels == 1:
                return np.clip(v[..., 0], 0, 1)
            # Shared fixed display transform; numerical comparison uses raw values.
            return np.clip(np.maximum(v, 0) / (1 + np.maximum(v, 0)), 0, 1) ** (1 / 2.2)
        axes[0].imshow(preview(x), cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(preview(y), cmap="gray", vmin=0, vmax=1)
        heat = axes[2].imshow(delta.mean(axis=-1), cmap="magma")
        fig.colorbar(heat, ax=axes[2], label="Mean absolute channel difference (raw units)")
        for ax, title in zip(axes, ("A: per-instance", "B: representations", "Difference")):
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(name + " — one untimed frame per variant; temporal/residency differences included")
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=140)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Contains A1, B1, B2 and A2 runner directories")
    parser.add_argument("--quality-a", type=Path)
    parser.add_argument("--quality-b", type=Path)
    args = parser.parse_args()
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    runs, series = {}, {}
    for name in ("A1", "B1", "B2", "A2"):
        runs[name], series[name] = load_run(args.root / name)
    first = runs["A1"]["manifest"]
    for name, run in runs.items():
        manifest = run["manifest"]
        for key in ("executableSha256", "shaderSha256", "config", "asset", "gpu", "vsync"):
            if manifest[key] != first[key]:
                raise ValueError(f"A/B identity mismatch: {name} {key}")
        if str(manifest["blasRepresentations"]) != ("0" if name.startswith("A") else "1"):
            raise ValueError("Representation toggle mismatch")
    names = [name for name in SCOPES if any(run["measurements"][name] for run in runs.values())]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    x = np.arange(len(names))
    for index, (run_name, run) in enumerate(runs.items()):
        means = [run["measurements"][name]["mean"] if run["measurements"][name] else np.nan for name in names]
        axes[0].bar(x + (index - 1.5) * .19, means, width=.19, label=run_name)
        axes[1].plot(series[run_name]["blasBuildCount"], label=run_name, alpha=.75)
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_ylabel("GPU milliseconds, per-run mean")
    axes[0].set_title("Identical scene/settings; separate scopes are not an end-to-end sum. Missing scopes remain missing.")
    axes[1].set_xlabel("Measured frame index after warmup")
    axes[1].set_ylabel("Physical BLAS builds / feedback frame")
    axes[1].set_ylim(0, max(1, max(max(s["blasBuildCount"]) for s in series.values())) * 1.05)
    for ax in axes:
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "RTRepresentationABBA.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, divisor, label in zip(axes,
            ("blasLiveClusterReferences", "blasArenaUsedBytes"), (1e6, 1024 ** 2),
            ("Live cluster references (millions)", "BLAS arena used (MiB)")):
        ax.bar(list(runs), [r["measurements"][metric]["mean"] / divisor for r in runs.values()],
               color=["tab:blue", "tab:orange", "tab:orange", "tab:blue"])
        ax.set_ylabel(label)
        ax.set_title("Per-run frame mean; allocation usage, not total VRAM")
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "RTRepresentationMemory.png", dpi=150)
    plt.close(fig)
    with (output / "Measurements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "metric", "n", "mean", "median", "p95", "minimum", "maximum"])
        for label, run in runs.items():
            for metric, value in run["measurements"].items():
                if value:
                    writer.writerow([label, metric, *value.values()])
    result = dict(runs=runs, versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
    if args.quality_a or args.quality_b:
        if not (args.quality_a and args.quality_b):
            raise ValueError("Specify both quality directories")
        result["quality"] = compare_images(args.quality_a, args.quality_b, output)
    (output / "Analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# RT representation A/B", "", "![A/B measurements](RTRepresentationABBA.png)", "",
             "Four serial runs in ABBA order; per-run frame distributions are descriptive, not independent trials. No confidence interval is inferred from frames.", "",
             "| GPU scope | A1 mean ms | B1 mean ms | B2 mean ms | A2 mean ms |", "|---|---:|---:|---:|---:|"]
    for metric in names:
        vals = [f"{runs[n]['measurements'][metric]['mean']:.4f}" if runs[n]['measurements'][metric] else "unavailable" for n in runs]
        lines.append("| " + metric + " | " + " | ".join(vals) + " |")
    lines += ["", "![Representation references and storage](RTRepresentationMemory.png)", "",
              "| Counter (per-frame mean) | A1 | B1 | B2 | A2 |", "|---|---:|---:|---:|---:|"]
    for metric in COUNTERS:
        vals = [f"{runs[n]['measurements'][metric]['mean']:.3f}" if runs[n]['measurements'][metric] else "unavailable" for n in runs]
        lines.append("| " + metric + " | " + " | ".join(vals) + " |")
    lines += ["", "Counts and allocation pressure: [CSV](Measurements.csv). Raw paths, hashes, configuration and tool versions: [JSON](Analysis.json).",
              "", "Reproduce: `python -B Tools/AnalyzeRTRepresentation.py " + str(args.root) +
              (f" --quality-a {args.quality_a} --quality-b {args.quality_b}" if args.quality_a else "") + "`"]
    if "quality" in result:
        lines += ["", "Quality captures contain single final-route frames; pixel differences include stochastic sampling and residency differences. They do not establish reference-image accuracy or secondary reflection quality."]
        for name in result["quality"]:
            lines += ["", f"![{name}]({name}.png)"]
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "Report.md")


if __name__ == "__main__":
    main()
