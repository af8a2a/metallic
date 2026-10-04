"""Phase 11 prototype A/B. Reuses the performance tools' process/evidence helpers.

This is not the WorkControl optimizer and never promotes a production candidate.
Timing, validation and compiler-resource diagnostics are separate process batches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

import DeepProfile as d
import ExperimentRunner as e
import WorkloadCase as w

PROTOCOL = "metallic-closure-scheduling-v1"


def inspect(directory, mode):
    process = w.load(directory / "Process.json")
    w.require(process.get("exitCode") == 0, "Failed benchmark process")
    cases = ET.parse(directory / "Tests.xml").findall(".//testcase")
    w.require({c.get("name") for c in cases} == {"material_closure_family_classification", "material_closure_fused_split_ab"}
              and len(cases) == 2 and all(c.get("status") == "run" and c.get("result") == "completed" and
              c.find("failure") is None and c.find("skipped") is None for c in cases),
              "Missing, failed or skipped correctness test")
    report = w.load(directory / "ClosureScheduling.json")
    w.require(report["validation"] == (mode == "validation") and report["pipelineStatistics"] == (mode == "resources"),
              "Diagnostic instrumentation mixed with ordinary timing")
    w.require(report["validationErrors"] == 0, "Vulkan validation errors")
    w.require(report["warmupPairs"] == 2 and report["measuredPairs"] == 6, "Changed sampling policy")
    rows = report["samples"]
    w.require(len(rows) == 112 and len(report["switchSamples"]) == 32, "Incomplete sample set")
    seen = set()
    fixtures = ((1, 1, 8, False), (17, 9, 8, True), (257, 129, 8, True),
                (1024, 512, 2, False), (1024, 512, 2, True), (1024, 512, 8, False), (1024, 512, 8, True))
    for row in rows:
        key = (row["width"], row["height"], row["programCount"], row["mixed"], row["pair"], row["split"])
        w.require(key not in seen and 0 <= row["pair"] < 8, "Duplicate or invalid frame")
        seen.add(key)
        w.require(row["warmup"] == (row["pair"] < 2) and row["closureFamilyCount"] == 1, "Wrong warmup/family contract")
        w.require(0 <= row["maxOutputDifference"] <= 2e-5, "Image mismatch")
        for field in ("programClassificationMs", "materialMs", "familyClassificationMs", "lightingMs", "totalMs"):
            w.require(math.isfinite(row[field]) and row[field] >= 0, "Invalid GPU duration")
    w.require(seen == {(*fixture, pair, split) for fixture in fixtures for pair in range(8) for split in (False, True)},
              "Missing paired workload")
    switches = set()
    for row in report["switchSamples"]:
        key = (row["programCount"], row["sample"], row["distinct"])
        w.require(key not in switches and row["warmup"] == (row["sample"] < 2), "Duplicate switch sample or wrong warmup")
        switches.add(key)
        w.require(math.isfinite(row["gpuMs"]) and row["gpuMs"] >= 0 and row["dispatches"] == row["programCount"] * 64,
                  "Invalid switch probe")
    w.require(switches == {(count, sample, distinct) for count in (2, 8) for sample in range(8) for distinct in (False, True)},
              "Missing switching control")
    w.require(len(report["shaders"]) == 29 and len({r["inputSpirvFnv1a64"] for r in report["shaders"] if '.fusedMain.' in r["label"]}) == 8,
              "Missing distinct compiled programs")
    resources = []
    if mode == "resources":
        text = (directory / "stdout.log").read_text(encoding="utf-8", errors="replace")
        w.require("[PipelineStatistics] enabled=true" in text, "Pipeline statistics unsupported")
        bindings = {}
        for line in text.splitlines():
            binding = d.BINDING.search(line)
            if binding:
                cache, source, device, label, entry = binding.groups()
                bindings[cache] = (int(source), int(device), label)
            stat = d.STAT.search(line)
            if stat and stat[2] in bindings:
                source, device, label = bindings[stat[2]]
                if not label.startswith("ClosureSchedulingProbe."):
                    continue
                w.require(any(s["label"] == label and s["inputSpirvFnv1a64"] == source for s in report["shaders"]),
                          "Pipeline resource identity does not match submitted SPIR-V")
                resources.append({"label": label, "inputSpirvFnv1a64": source, "deviceSpirvFnv1a64": device,
                                  "name": stat[5], "value": stat[6], "description": stat[7], "subgroup": int(stat[4])})
        w.require(len({r["label"] for r in resources}) == 29, "Missing executable resource statistics")
    return report, resources


def verify(output):
    manifest = w.load(output / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL and manifest["status"] == "complete", "Incomplete evidence package")
    for name, digest in manifest["files"].items():
        w.require(w.digest(w.file_in(output, name)) == digest, f"Evidence changed: {name}")
    reports, resources = [], []
    previous = 0
    for name in manifest["runs"]:
        directory = output / name
        process = w.load(directory / "Process.json")
        w.require(process["startedUnix"] >= previous, "GPU processes overlapped")
        previous = process["endedUnix"]
        report, stats = inspect(directory, manifest["mode"])
        reports.append(report)
        resources.append(stats)
    w.require(len(reports) == manifest["runCount"] and len(reports) >= 3, "Requires three independent processes")
    w.require(all(r["shaders"] == reports[0]["shaders"] for r in reports), "Shader identity changed between processes")
    if manifest["mode"] == "resources":
        w.require(all(r == resources[0] for r in resources), "Compiler resource statistics were not repeatable")
    return manifest, reports, resources


def run(args):
    output = args.output.resolve()
    exe = args.exe.resolve()
    w.require(exe.is_file() and 3 <= args.runs <= 8, "Missing EXE or invalid run count")
    build = exe.parent.parent
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    w.require("CMAKE_BUILD_TYPE:STRING=Release" in cache and
              f"CMAKE_HOME_DIRECTORY:INTERNAL={w.ROOT.as_posix()}" in cache, "Requires this repository's Release build")
    output.mkdir(parents=True, exist_ok=False)
    lock = w.ROOT / "build/shader-experiment.lock"
    manifest = {"protocol": PROTOCOL, "status": "running", "mode": args.mode, "runCount": args.runs, "runs": [],
                "python": sys.version,
                "gpu": e.gpu_identity(), "clocks": "unaltered", "systemGpuExclusive": False,
                "runtime": {str(p): w.digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]},
                "cmakeCacheSha256": w.digest(build / "CMakeCache.txt")}
    acquired = False
    try:
        with lock.open("x", encoding="utf-8") as handle:
            json.dump({"output": str(output), "pid": os.getpid()}, handle)
        acquired = True
        sources = [Path(__file__), Path(d.__file__), Path(e.__file__), Path(w.__file__),
                   w.ROOT / "tests/rhi/MaterialClosureSchedulingTests.cpp",
                   w.ROOT / "tests/rhi/shaders/ClosureSchedulingProbe.slang",
                   *sorted((w.ROOT / "Source/Runtime/Material").glob("MaterialClosure*.*")),
                   *sorted((w.ROOT / "Shaders/Modules").rglob("*.slang"))]
        manifest["sources"] = {str(p.relative_to(w.ROOT)): w.digest(p) for p in sources}
        for relative, digest in manifest["sources"].items():
            archive = output / "source" / relative
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes((w.ROOT / relative).read_bytes())
            w.require(w.digest(archive) == digest, "Source changed while archiving")
        env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
        env.update(METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0")
        if args.mode == "resources":
            env["METALLIC_VK_PIPELINE_STATISTICS"] = "1"
        for index in range(args.runs):
            directory = output / f"run-{index}"
            directory.mkdir()
            command = [str(exe), "--gtest_filter=*material_closure_family_classification*:*material_closure_fused_split_ab*",
                       "--rhi-validation" if args.mode == "validation" else "--rhi-no-validation",
                       "--output-dir", str(directory), f"--gtest_output=xml:{directory / 'Tests.xml'}"]
            d.run_process(command, env, directory, args.timeout)
            inspect(directory, args.mode)
            manifest["runs"].append(directory.name)
            w.require(all(w.digest(w.ROOT / p) == digest for p, digest in manifest["sources"].items()), "Source changed during experiment")
            w.require(all(w.digest(Path(p)) == digest for p, digest in manifest["runtime"].items()), "Runtime changed during experiment")
            print(f"Completed {args.mode} process {index + 1}/{args.runs}", flush=True)
        manifest["status"] = "complete"
    finally:
        manifest["files"] = {str(p.relative_to(output)).replace('\\', '/'): w.digest(p)
                             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "Manifest.json"}
        w.save(output / "Manifest.json", manifest)
        if acquired:
            lock.unlink()
    verify(output)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest, reports, _ = verify(args.output.resolve())
    w.require(manifest["mode"] == "timing", "Plot ordinary timing separately from diagnostics")
    destination = args.report.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, report in enumerate(reports):
        for count in (2, 8):
            for mixed in (False, True):
                for split in (False, True):
                    values = [r["totalMs"] for r in report["samples"] if r["width"] == 1024 and
                              r["programCount"] == count and r["mixed"] == mixed and r["split"] == split and not r["warmup"]]
                    rows.append({"process": index, "programs": count, "families": 1, "mixed": mixed, "split": split,
                                 "medianMs": statistics.median(values), "rawMs": values})
    w.save(destination / "ProcessMedians.json", rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    for case, (count, mixed) in enumerate(((2, False), (2, True), (8, False), (8, True))):
        labels.append(f"{count}:1 {'mixed' if mixed else 'coherent'}")
        for split, color, offset in ((False, "#3c6faa", -.15), (True, "#e29437", .15)):
            values = [r["medianMs"] for r in rows if r["programs"] == count and r["mixed"] == mixed and r["split"] == split]
            ax.scatter([case + offset] * len(values), values, c=color, label=("Split" if split else "Fused") if case == 0 else None)
            ax.hlines(statistics.median(values), case + offset - .10, case + offset + .10, colors=color, linewidth=3)
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Material + classification + lighting GPU time (ms)",
           title=f"1024x512 Slab prototype, 16 lights; n={len(reports)} independent processes\nPoints: process medians of 6 pairs; bars: median across processes")
    ax.set_ylim(bottom=0); ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(destination / "FusedSplit.png", dpi=160); plt.close(fig)
    switches = []
    for index, report in enumerate(reports):
        for count in (2, 8):
            values = {distinct: statistics.median([r["gpuMs"] for r in report["switchSamples"] if
                       r["programCount"] == count and r["distinct"] == distinct and not r["warmup"]]) for distinct in (False, True)}
            switches.append({"process": index, "programs": count, "sameMs": values[False], "distinctMs": values[True],
                             "deltaNsPerDispatch": (values[True] - values[False]) * 1e6 / (count * 64)})
    w.save(destination / "Switching.json", switches)
    fig, ax = plt.subplots(figsize=(7, 4))
    for count in (2, 8):
        values = [r["deltaNsPerDispatch"] for r in switches if r["programs"] == count]
        ax.scatter([count] * len(values), values, label=f"{count} programs, {count * 64} dispatches/batch")
    ax.axhline(0, color="gray"); ax.set(xticks=[2, 8], ylabel="Distinct - same pipeline (ns/dispatch)",
        title="Tiny dispatch control; process medians, not production switching cost")
    ax.legend(); fig.tight_layout(); fig.savefig(destination / "Switching.png", dpi=160); plt.close(fig)
    resources = None
    if args.resources:
        resource_manifest, _, resources = verify(args.resources.resolve())
        w.require(resource_manifest["mode"] == "resources", "Expected a separate resources batch")
        w.save(destination / "PipelineResources.json", resources)
        registers = {r["label"]: int(r["value"]) for r in resources[0] if r["name"] == "Register Count"}
        needed = [f"ClosureSchedulingProbe.{entry}.P{p}" for entry in ("fusedMain", "evaluateMain") for p in range(8)]
        needed.append("ClosureSchedulingProbe.lightFamilyMain.P0")
        w.require(all(label in registers for label in needed), "Driver Register Count metric unavailable")
        fig, ax = plt.subplots(figsize=(9, 4))
        for entry, label, color in (("fusedMain", "Fused", "#3c6faa"), ("evaluateMain", "Material evaluation", "#e29437")):
            ax.plot(range(8), [registers[f"ClosureSchedulingProbe.{entry}.P{p}"] for p in range(8)], 'o-', color=color, label=label)
        ax.axhline(registers["ClosureSchedulingProbe.lightFamilyMain.P0"], color="#419772", label="Shared family lighting")
        ax.set(xticks=range(8), xlabel="Compiled Material Program", ylabel="Register Count (driver temporary registers)",
               title="Separate pipeline diagnostic; 3 processes agree exactly")
        ax.set_ylim(bottom=0); ax.legend(); fig.tight_layout(); fig.savefig(destination / "Registers.png", dpi=160); plt.close(fig)
    (destination / "ClosureScheduling.py").write_bytes(Path(__file__).read_bytes())
    (destination / "report.md").write_text(
        f"# Closure scheduling prototype\n\nGPU: {manifest['gpu']}\n\n"
        f"Evidence: {args.output.resolve()}\n\nPython {sys.version}; matplotlib {matplotlib.__version__}.\n\n"
        "![Fused / Split](FusedSplit.png)\n\n![Switching control](Switching.png)\n\n"
        + (f"![Compiler resources](Registers.png)\n\nResource evidence: {args.resources.resolve()}\n\n" if resources else "") +
        "Scope: material evaluation, optional 96-byte/pixel closure writes, family classification, 16-light shading. "
        "Common program classification, resets, CPU recording, shader compilation and readback are excluded. "
        "All buffers used for closure/output/atomics are device-local. Both paths include exactly-once instrumentation. "
        "Two warmup pairs are excluded; all six measured pairs are retained. Dots are independent-process medians, "
        "not independent frames or confidence intervals. Clocks are unaltered; system GPU exclusivity is not proven. "
        "This synthetic fixture does not establish an OpenPBR scene speedup or justify a production closure buffer.\n\n"
        f"Reproduce: `python Tools/Perf/ClosureScheduling.py plot {args.output.resolve()} --report NEW_DIRECTORY`\n\n"
        "Script snapshot: [ClosureScheduling.py](ClosureScheduling.py); rerun the current repository script using the command above. "
        "Raw timings and field mapping: "
        "ProcessMedians.json (`totalMs` -> `medianMs`), Switching.json (`gpuMs`, ms -> ns/dispatch). "
        "PipelineResources.json, when present, contains separate diagnostic observations; driver names and values are retained. "
        "The Local Memory Size value 68719476736 is an anomalous driver value, not evidence of spilling. "
        "Register counts do not directly establish runtime occupancy.\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--exe", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--mode", choices=("timing", "resources", "validation"), default="timing")
    execute.add_argument("--runs", type=int, default=3)
    execute.add_argument("--timeout", type=int, default=240)
    verify_command = commands.add_parser("verify"); verify_command.add_argument("output", type=Path)
    plotting = commands.add_parser("plot"); plotting.add_argument("output", type=Path)
    plotting.add_argument("--report", type=Path, required=True); plotting.add_argument("--resources", type=Path)
    args = parser.parse_args()
    if args.command == "run": run(args)
    elif args.command == "plot": plot(args)
    else:
        manifest, _, _ = verify(args.output.resolve())
        print(f"Verified {manifest['mode']}: {manifest['runCount']} serial processes, hashes, identities and correctness")


if __name__ == "__main__":
    main()
