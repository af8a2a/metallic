"""Render verified material baseline evidence into a separate review directory."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from MaterialBaseline import digest, images, read, save, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resources-log", type=Path)
    args = parser.parse_args()
    verification = verify(args.evidence)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "plot_report.py")
    shutil.copy2(Path(__file__).with_name("MaterialBaseline.py"), args.output / "MaterialBaseline.py")
    reports = [read(p) for p in sorted(args.evidence.glob("run-*/MaterialBaseline.json"))]
    save(args.output / "PlotData.json", reports)
    save(args.output / "Verification.json", verification)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    lines = ["# Material system Phase 0 GPU baseline", "",
             "Three independent processes; each line is 64 consecutive warmed frames (indices 32-95).",
             "GPU graph timestamps in milliseconds; graphics queue, fixed fixtures, no validation/capture injection.",
             "Disk caches retained. Frame samples are correlated; no confidence interval or speedup claim.", "",
             "| Case | Process medians (ms) | A/A HDR max error |", "| --- | --- | --- |"]
    for case_index, axis in enumerate(axes):
        name = reports[0]["cases"][case_index]["id"]
        medians = []
        for index, report in enumerate(reports):
            case = report["cases"][case_index]
            frames = case["frames"]
            values = [f["graphMs"] for f in frames]
            medians.append(float(np.median(values)))
            axis.plot([f["frame"] for f in frames], values, label=f"Process {index + 1} (median {medians[-1]:.3f})", linewidth=1)
        axis.set(title=name, xlabel="Frame index (history begins at 0)", ylabel="Graph GPU time (ms)")
        axis.legend()
        axis.grid(alpha=0.2)
        lines.append(f"| {name} | " + ", ".join(f"{x:.4f}" for x in medians) +
                     f" | {verification['imageAA'][name]['maxAbsoluteDifference']:.8g} |")
    fig.savefig(args.output / "GPUTime.png", dpi=150)
    plt.close(fig)
    lines += ["", "![GPU time](GPUTime.png)", "", "## Linear HDR reference previews", "",
              "Exposure multiplier 1, sRGB transfer, no tone mapping; display highlights clip at 1.",
              "The raw RGBA16F/32F captures remain authoritative."]
    for name, runs in images(args.evidence).items():
        rgb = np.maximum(runs[0][:, :, :3], 0)
        srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
        Image.fromarray(np.uint8(np.round(np.clip(srgb, 0, 1) * 255))).save(args.output / f"{name}.png")
        lines += ["", f"### {name}", "", f"![{name}]({name}.png)"]
    lines += ["", "## Repeatability", "", "```json", json.dumps(verification["imageAA"], indent=2), "```",
              "OpenPBR PT is not bitwise repeatable in this capture. Its measured A/A error is an observation, not an automatic regression tolerance.",
              "The timing plot includes spikes and process drift; no samples were removed to tighten the distribution."]
    if args.resources_log:
        text = args.resources_log.read_text(encoding="utf-8-sig")
        if "[PipelineStatistics]" not in text or "[  PASSED  ] 1 test." not in text:
            raise ValueError("Incomplete pipeline resource capture")
        bindings, resources, graph = {}, [], None
        for line_number, line in enumerate(text.splitlines(), 1):
            start = re.search(r"Begin compile graph '([^']+)'", line)
            if start:
                graph = start[1]
            binding = re.search(r"\[PipelineStatisticsBinding\] cacheKey=(\w+) inputSpirvFnv1a64=(\d+) deviceSpirvFnv1a64=(\d+) shader=(\S+) entry=(\S+)", line)
            if binding:
                bindings[binding[1]] = {"graph": graph, "cacheKey": binding[1], "inputSpirvFnv1a64": binding[2],
                                       "deviceSpirvFnv1a64": binding[3], "shader": binding[4], "entry": binding[5]}
            stat = re.search(r"\[PipelineStatistics\] entry=(\S+) spirv=(\w+) executable=(\S+) subgroup=(\d+) (.*?)=(.*?) \((.*?)\)", line)
            if stat and stat[2] in bindings:
                record = bindings[stat[2]]
                if record["shader"] == "ScenePathTracePass.base" or record["shader"].startswith("materialBinning"):
                    resources.append({**record, "executable": stat[3], "subgroup": int(stat[4]), "name": stat[5],
                                      "value": stat[6], "description": stat[7], "line": line_number})
        registers = [r for r in resources if r["name"] == "Register Count"]
        if not registers:
            raise ValueError("No material registers in diagnostic log")
        save(args.output / "PipelineResources.json", resources)
        shutil.copy2(args.resources_log, args.output / "PipelineResources.log")
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        labels = [r["graph"] + " / " + r["shader"] + "\n" + r["cacheKey"] for r in registers]
        axis.barh(range(len(registers)), [int(r["value"]) for r in registers])
        axis.set_yticks(range(len(registers)), labels=labels, fontsize=8)
        axis.set(xlabel="Driver Register Count (temporary registers per shader stage)",
                 title="Material pipeline compiler resources (diagnostic n=1, not runtime occupancy)")
        fig.savefig(args.output / "Registers.png", dpi=150)
        plt.close(fig)
        lines += ["", "## Separate pipeline resource diagnostics", "", "![Registers](Registers.png)",
                  "Source identity uses graph compilation context, shader debug name, cache key and both input/device SPIR-V fingerprints.",
                  "Five Deferred programs retain distinct keys; labels do not assume a feature class from creation order.",
                  "Raw driver statistic names, descriptions and log line numbers are in PipelineResources.json. Diagnostic n=1.",
                  "Large Local Memory Size sentinel values are preserved; they are not interpreted as allocated bytes or spills.",
                  f"Resource log SHA256: `{digest(args.resources_log)}`."]
    lines += ["", "## Measurement coverage", "",
              "Material classification has its own GPU scope (reset + classify + indirect argument build and barriers).",
              "Node/section timings and their parent indices remain in PlotData.json; do not sum nested scopes.",
              "VBuffer resolve, material evaluation, OpenPBR prepare, direct and environment lighting are fused in Deferred shading; independent timings are unavailable.",
              "Occupancy, texture latency, instruction count, bin occupancy and mixed-material tile ratio were not collected by this harness.", "",
              "## Reproduction", "", f"Evidence: `{args.evidence.resolve()}`",
              f"Manifest SHA256: `{digest(args.evidence / 'Manifest.json')}`",
              f"Python: `{sys.version.split()[0]}`; NumPy `{np.__version__}`; Matplotlib `{matplotlib.__version__}`.",
              "Field mapping: cases[].frames[].graphMs, no unit conversion; median within each independent process.",
              "No measured frames were excluded beyond the declared warmup window.", "",
              f"`python -B tools/Perf/PlotMaterialBaseline.py {args.evidence} <new-report-directory>" +
              (f" --resources-log {args.resources_log}" if args.resources_log else "") + "`"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
