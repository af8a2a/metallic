"""Plot verified multi-material production GPU timings in a separate report directory."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from MaterialBaseline import digest, read, save, verify


def plot(evidence, output):
    verification = verify(evidence)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "plot_report.py")
    shutil.copy2(Path(__file__).with_name("MaterialBaseline.py"), output / "MaterialBaseline.py")
    save(output / "Verification.json", verification)
    reports = [read(p) for p in sorted(evidence.glob("run-*/MaterialBaseline.json"))]
    config = read(evidence / "fixtures/Cases.json")
    save(output / "PlotData.json", reports)
    rows, summary = [], []
    invalid = verification["invalidHDR"]
    for i, case in enumerate(reports[0]["cases"]):
        if case["id"] in invalid:
            continue
        node_name = case["requiredTiming"][0]
        graph_medians, pass_medians = [], []
        for run, report in enumerate(reports):
            current = report["cases"][i]
            assert current["id"] == case["id"]
            for phase in ("warmup", "frames"):
                for f in current.get(phase, []):
                    node = next(n for n in f["nodes"] if n["name"] == node_name)
                    rows.append(dict(case=case["id"], process=run + 1, phase=phase, frame=f["frame"],
                                     graph_ms=f["graphMs"], material_pass_ms=node["gpuMs"]))
            graph_medians.append(float(np.median([f["graphMs"] for f in current["frames"]])))
            pass_medians.append(float(np.median([next(n["gpuMs"] for n in f["nodes"] if n["name"] == node_name) for f in current["frames"]])))
        span = float(np.ptp(graph_medians) / np.median(graph_medians))
        pass_span = float(np.ptp(pass_medians) / np.median(pass_medians))
        summary.append(dict(case=case["id"], group=case["group"], backend=case["backend"],
            graph_medians_ms=graph_medians, pass_medians_ms=pass_medians,
            graph_ms=float(np.median(graph_medians)), pass_ms=float(np.median(pass_medians)),
            aa_relative_range=span, pass_aa_relative_range=pass_span,
            stable=max(span, pass_span) <= config["aaRelativeRangeLimit"],
            image_aa_max_error=verification["imageAA"][case["id"]]["maxAbsoluteDifference"]))
    with (output / "Frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(output / "Summary.json", summary)
    with (output / "Summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    colors = ["#286da8", "#d77923", "#368444"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 13), constrained_layout=True)
    for ax, backend in zip(axes, ("Deferred", "PT")):
        selected = sorted([s for s in summary if s["group"] == "Surface" and s["backend"] == backend], key=lambda s: s["pass_ms"])
        y = np.arange(len(selected))
        for run in range(3):
            ax.scatter([s["pass_medians_ms"][run] for s in selected], y + (run - 1) * .14,
                       s=22, color=colors[run], label=f"Process {run + 1}")
        ax.set_yticks(y, [s["case"].removesuffix("-" + backend) + (" *" if not s["stable"] else "") for s in selected], fontsize=8)
        ax.set(xlabel="Material pass GPU time (ms)", title=backend + (" / IBL 64 samples" if backend == "Deferred" else " / 4 spp, depth 12"))
        ax.grid(axis="x", alpha=.25); ax.legend()
    fig.suptitle("512 x 512 material baseline | 3 independent processes x 64 warmed frames\nPoints: process medians; * graph or pass A/A relative range > 10%; no frame filtering", fontsize=12)
    fig.savefig(output / "MaterialPass.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for ax, s in zip(axes, [s for s in summary if s["group"] != "Surface"]):
        for run in range(3):
            data = [r for r in rows if r["case"] == s["case"] and r["process"] == run+1]
            ax.plot([r["frame"] for r in data], [r["graph_ms"] for r in data], label=f"Process {run+1}", lw=1)
        ax.axvspan(0, config["warmupFrames"]-1, alpha=.08, color="black")
        ax.set(title=s["case"] + " (separate geometry and lighting workload)", xlabel="Frame index", ylabel="Graph GPU ms")
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / "Fiber.png", dpi=150); plt.close(fig)
    worst = sorted(summary, key=lambda s: s["aa_relative_range"], reverse=True)[:6]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for ax, s in zip(axes.flat, worst):
        for run in range(3):
            data = [r for r in rows if r["case"] == s["case"] and r["process"] == run+1 and r["phase"] == "frames"]
            ax.plot([r["frame"] for r in data], [r["graph_ms"] for r in data], lw=1, label=f"Process {run+1}")
        ax.set(title=f"{s['case']}\nA/A range {s['aa_relative_range']:.1%}", xlabel="Measured frame", ylabel="Graph GPU ms")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.savefig(output / "Drift.png", dpi=150); plt.close(fig)
    # Raw ACEScg/AP1 -> linear Rec.709 preview, simple Reinhard then sRGB; not an image oracle.
    matrix = np.array([[1.7050515, -.6217907, -.0832584], [-.1302572, 1.1408027, -.0105485], [-.0240033, -.1289688, 1.1529717]])
    for backend in ("Deferred", "PT", "Other"):
        cases = [c for c in reports[0]["cases"] if (c["group"] != "Surface" if backend == "Other" else c["group"] == "Surface" and c["backend"] == backend)]
        fig, axes = plt.subplots((len(cases)+4)//5, 5, figsize=(15, 3*((len(cases)+4)//5)), squeeze=False, constrained_layout=True)
        for ax in axes.flat: ax.axis("off")
        for ax, c in zip(axes.flat, cases):
            if c["id"] in invalid:
                ax.text(.5, .5, "INVALID HDR\nExcluded from timing ranking", ha="center", va="center", color="red")
                ax.set_title(c["id"], fontsize=7)
                continue
            pixels = np.fromfile(evidence / "run-0" / c["image"], dtype="<f2" if c["format"] == "RGBA16F" else "<f4").reshape(c["height"], c["width"], 4)[..., :3].astype(float)
            rgb = np.maximum(pixels @ matrix.T, 0); rgb = rgb/(1+rgb)
            rgb = np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)
            ax.imshow(np.clip(rgb,0,1)); ax.set_title(c["id"], fontsize=7)
        fig.savefig(output / f"Preview-{backend}.png", dpi=110); plt.close(fig)
    lines = ["# Material GPU baseline", "", "GPU identity (name, UUID, driver, initial P-state, temperature): `" + (evidence / "GPU.csv").read_text().splitlines()[1] + "`.",
        "Capture HEAD: `" + read(evidence / "Identity.json")["commit"] + "`; exact dirty sources and executable hashes are recorded in Identity.json.",
        "512 x 512. Three independent processes, 32 warmup + 64 measured frames per case; capture at frame 96.",
        "Material pass and graph timestamps are GPU milliseconds, not editor frame time/FPS. No measured samples filtered.",
        "The summary is the median of three process medians. A/A relative range = (max-min)/median, threshold 10% applied conservatively to BOTH graph and material pass; this is qualification, not a confidence interval.",
        "Graph timestamps are a graphics-queue envelope. Deferred VBuffer forks two compute/graphics raster branches and joins before shading; the graph envelope includes those waits. All enclosing node intervals use graphics timestamps. Concurrent/nested scopes are not summed. The verifier checks the observed queue contract against the registered backend.",
        "Production rendering passes; compilation, scene loading, CPU waits and final HDR readback excluded from GPU timing. Disk caches retained; GPU clocks unaltered.",
        "Surface: White Studio HDRI, authored fixed cameras, manual exposure, Deferred program binning + FP16 weights + advanced IBL 64 samples; PT 4 spp/frame, max depth 12, accumulated 388 spp at capture.",
        "Uniform/textured and sweep cases can differ in mesh coverage and material complexity; timings are scene baselines, not isolated BSDF instruction costs.",
        "Slab Single/Mix/Layer replace all materials on the M01 uniform scene. Fiber uses Claire groom at 4 spp/depth 4 with its own studio HDRI. Native Strands uses the small 576-segment, K=8 prototype and its own lighting. These backends are not directly comparable.",
        "Validation disabled for timings. Separate validation pilot is not included. Known METALLIC profiler flags sanitized; arbitrary external profiler absence and GPU exclusivity are not proven.",
        "Background GPU applications retained. GPUProcesses.txt and per-process Telemetry.csv preserve observed whole-device load, memory, temperature and clocks; no per-process competition attribution.",
        "", "![Material pass](MaterialPass.png)", "", "| Case | Graph ms | Material pass ms | Graph process medians ms | Graph A/A | Pass A/A | Status |", "|---|---:|---:|---|---:|---:|---|"]
    for s in summary:
        lines.append(f"| {s['case']} | {s['graph_ms']:.4f} | {s['pass_ms']:.4f} | " + ", ".join(f"{v:.4f}" for v in s["graph_medians_ms"]) + f" | {s['aa_relative_range']:.1%} | {s['pass_aa_relative_range']:.1%} | {'stable' if s['stable'] else 'inconclusive'} |")
    lines += ["", "![Fiber](Fiber.png)", "", "![Largest A/A drift](Drift.png)", "", "## Output review", "",
        "HDR finiteness checked for every case/run. Any case with a nonfinite component in any run is excluded from performance tables/charts. Raw invalid images and timestamps remain in the evidence and PlotData.json. A/A errors are observations, not new quality tolerances. Previews use AP1 to Rec.709 + Reinhard + sRGB; raw HDR files are authoritative.",
        "", "Invalid HDR cases (nonfinite float component counts by process):", "", "```json", json.dumps(invalid, indent=2), "```",
        "", "![Deferred](Preview-Deferred.png)", "", "![PT](Preview-PT.png)", "", "![Fiber](Preview-Other.png)", "", "## Reproduce", "",
        f"Evidence: `{evidence.resolve()}`", f"Manifest SHA256: `{digest(evidence / 'Manifest.json')}`",
        f"Python {sys.version.split()[0]}, NumPy {np.__version__}, Matplotlib {matplotlib.__version__}.",
        "Field mapping: cases[].frames[].graphMs; nodes[name=requiredTiming[0]].gpuMs. No unit conversion. Warmup frames retained separately. Nested GPU sections must not be summed.",
        f"`python -B Tools/Perf/PlotMaterialCatalog.py {evidence} <new-report-dir>`", "",
        "To repeat capture use the original configuration path recorded in Environment.json with the same assets and build configuration; verify hashes before comparing. No optimization or speedup acceptance is implied.",
        "The measured PT uses the current fused production ray loop. Optional M8 ray-queue classification is not included in these production pass timings."]
    quality = {}
    for name in invalid:
        quality[name] = {}
        for run, report in enumerate(reports):
            c = next(c for c in report["cases"] if c["id"] == name)
            pixels = np.fromfile(evidence / f"run-{run}" / c["image"], dtype="<f2" if c["format"] == "RGBA16F" else "<f4").reshape(c["height"], c["width"], 4)
            quality[name][f"run-{run}"] = dict(nan=int(np.isnan(pixels).sum()), infinity=int(np.isinf(pixels).sum()),
                first_nonfinite_y_x_channel=np.argwhere(~np.isfinite(pixels))[:32].tolist())
    save(output / "InvalidHDR.json", quality)
    lines += ["", "Nonfinite coordinates and NaN/Inf counts: [InvalidHDR.json](InvalidHDR.json)."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args(); plot(args.evidence.resolve(), args.output.resolve())
