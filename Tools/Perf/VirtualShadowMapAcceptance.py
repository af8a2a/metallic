"""Run and plot the production VSM acceptance test without changing renderer settings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime_errors(directory):
    # SDK errors can be logged while the RHI test still returns success, e.g.
    # Streamline's missing presentCommon bookkeeping in a headless preview.
    return [line for line in (directory / "Process.log").read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"\[(?:error|critical)\]", line, re.IGNORECASE)]


def validate_capture(directory, spec):
    # The RHI HTML reporter also archives copies under reports/. Only the
    # primary output belongs to this process; archives are not extra samples.
    path = directory / "VirtualShadowMapAcceptance.json"
    data = read(path)
    expected = {case["id"]: case for case in spec["cases"]}
    if len(expected) != len(spec["cases"]) or not expected:
        raise ValueError("Distinct nonempty case IDs required")
    if (data.get("protocol") != "virtual-shadow-map-acceptance-v2" or data.get("status") != "passed"
            or data.get("mode") != spec["mode"] or len(data["cases"]) != len(expected)
            or {case["id"] for case in data["cases"]} != set(expected)):
        raise ValueError("Missing, skipped, partial or failed acceptance capture")
    for case in data["cases"]:
        settings = expected[case["id"]]
        count = settings.get("frames", spec.get("frames", 512))
        warmup = settings.get("warmupFrames", spec.get("warmupFrames", 16))
        strict_capabilities = settings.get("requireNoRayCapabilities", not settings.get("enableStreamline", False))
        comparisons = case.get("comparisons", [])
        if [c["frame"] for c in comparisons] != sorted(set(settings.get("comparisonFrames", []))):
            raise ValueError("Missing or duplicated declared cache comparison frame")
        for comparison in comparisons:
            for label in ("cachedHDR", "redrawnHDR"):
                hdr = comparison[label]
                if (hdr["format"] not in ("RGBA16Sfloat", "RGBA32Sfloat") or hdr["width"] <= 0
                        or hdr["height"] <= 0 or hdr["components"] != hdr["width"] * hdr["height"] * 4
                        or hdr["nonfiniteComponents"] != 0):
                    raise ValueError("Invalid or nonfinite HDR comparison receipt")
        if (case.get("status") != "passed" or not case.get("noRayTracingAllocations")
                or (strict_capabilities and not case.get("noRayQueryOrRTAS"))
                or [f["frame"] for f in case["frames"]] != list(range(count))):
            raise ValueError("Incomplete no-shadow-TLAS frame sequence")
        for frame in case["frames"]:
            value = frame.get("graphMs")
            if (value is None or not math.isfinite(value) or value <= 0
                    or frame["warmup"] != (frame["frame"] < warmup)
                    or frame["memory"]["rayTracingAllocationBytes"] != 0
                    or frame["memory"]["clasAllocationBytes"] != 0
                    or frame["memory"]["domains"]["RayTracing"]["peakAllocationBytes"] != 0
                    or frame["memory"]["domains"]["CLAS"]["peakAllocationBytes"] != 0):
                raise ValueError("Invalid GPU timing, warmup or no-RT evidence")
    if spec["mode"] == "timing" and (data["validation"] or not data["performanceEligible"]):
        raise ValueError("Diagnostic timestamps cannot enter ordinary timing evidence")
    return path, data


def run(args):
    cases = args.cases.resolve()
    spec = read(cases)
    if spec["mode"] not in ("diagnostic", "timing") or not 1 <= args.runs <= 10:
        raise ValueError("Expected diagnostic/timing mode and 1..10 independent processes")
    if len(spec["cases"]) != 1:
        raise ValueError("Run each named *Cases.json separately: acceptance uses one scene/device per process")
    validation = spec.get("validation", spec["mode"] == "diagnostic")
    if spec["mode"] == "timing" and validation:
        raise ValueError("Ordinary timing cannot enable validation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    inputs = {str(cases): digest(cases)}
    for case in spec["cases"]:
        graph = Path(case["graph"])
        if not graph.is_absolute():
            graph = cases.parent / graph
        inputs[str(graph.resolve())] = digest(graph)
    sources = sorted({*ROOT.glob("Source/**/*.cpp"), *ROOT.glob("Source/**/*.h"),
                      *ROOT.glob("Shaders/**/*.slang"), *ROOT.glob("Shaders/**/*.h"),
                      ROOT / "tests/rhi/VirtualShadowMapAcceptanceTests.cpp", Path(__file__)})
    identity = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "status": subprocess.check_output(["git", "status", "--short", "--ignore-submodules=all"],
                                                   cwd=ROOT, text=True).strip(),
                "platform": platform.platform(), "python": sys.version,
                "binaries": {str(p): digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]},
                "inputs": inputs, "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "assetIdentity": "Referenced asset paths only; no full cook/texture content manifest",
                "cache": "Existing disk caches retained; fresh preview per case/process; clocks unchanged",
                "measurement": "GPU graph/pass spans; CPU recording separately; serialized preview, no editor presentation",
                "gpuExclusivity": "Not enforced; user applications are not stopped"}
    save(output / "Identity.json", identity)
    save(output / "Cases.json", spec)
    env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
    env["METALLIC_VSM_ACCEPTANCE_CASES"] = str(cases)
    results = []
    for index in range(args.runs):
        directory = output / f"run-{index + 1}"
        directory.mkdir()
        command = [str(exe), "--filter", "virtual_shadow_map_acceptance", "--rhi-bindless",
                   "--rhi-validation" if validation else "--rhi-no-validation",
                   "--output-dir", str(directory)]
        with (directory / "Process.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=args.timeout, check=False)
                result = {"command": command, "exitCode": process.returncode, "timedOut": False}
            except subprocess.TimeoutExpired:
                result = {"command": command, "exitCode": None, "timedOut": True}
        results.append(result)
        if result["exitCode"] == 0:
            try:
                validate_capture(directory, spec)
                errors = runtime_errors(directory)
                if errors:
                    result["runtimeErrors"] = errors
                    raise ValueError("Runtime/SDK errors were logged despite a passing RHI result")
                result["captureValid"] = True
            except (ValueError, OSError, KeyError) as error:
                result["captureValid"] = False
                result["captureError"] = str(error)
        save(output / "Processes.json", results)
        if result["exitCode"] != 0 or not result.get("captureValid"):
            break
    unchanged = all(Path(path).is_file() and digest(path) == value for path, value in inputs.items())
    unchanged &= all(digest(ROOT / path) == value for path, value in identity["sources"].items())
    binaries_unchanged = all(digest(path) == value for path, value in identity["binaries"].items())
    save(output / "Integrity.json", {"sourceAndInputsUnchanged": unchanged,
                                    "binariesUnchanged": binaries_unchanged, "requestedRuns": args.runs,
                                    "completedRuns": len(results)})
    save(output / "Manifest.json", {str(p.relative_to(output)): digest(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()})
    if (not unchanged or not binaries_unchanged or len(results) != args.runs
            or any(r["exitCode"] != 0 or not r.get("captureValid") for r in results)):
        raise RuntimeError(f"Capture incomplete or failed; evidence preserved at {output}")
    print(output)


def verify_package(source):
    """Apply the report's package and per-capture checks without generating output."""
    source = Path(source).resolve()
    manifest = read(source / "Manifest.json")
    actual = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob("*"))
              if p.is_file() and p.name != "Manifest.json"}
    if manifest != actual:
        raise ValueError("Evidence hash/file set mismatch")
    integrity, processes, spec = read(source / "Integrity.json"), read(source / "Processes.json"), read(source / "Cases.json")
    if (not integrity["sourceAndInputsUnchanged"] or not integrity["binariesUnchanged"]
            or len(processes) != integrity["requestedRuns"]
            or any(p["exitCode"] != 0 or not p.get("captureValid") for p in processes)):
        raise ValueError("Incomplete evidence cannot be reported as accepted measurements")
    captures = []
    for index in range(len(processes)):
        if runtime_errors(source / f"run-{index + 1}"):
            raise ValueError("Runtime/SDK errors invalidate this package; inspect the preserved process log")
        path, data = validate_capture(source / f"run-{index + 1}", spec)
        captures.append((path, data))
    return spec, captures


def report(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, captures = verify_package(source)
    grouped = {}
    for path, data in captures:
        for case in data["cases"]:
            grouped.setdefault(case["id"], []).append((path, data, case))
    summary = {"source": str(source), "manifestSHA256": digest(source / "Manifest.json"),
               "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
               "scope": "Descriptive per-frame quantiles; independent process medians; no speedup inference",
               "cases": []}
    text = ["# VSM acceptance evidence", "", summary["scope"], "",
            "GPU timings from diagnostic runs include validation/readback overhead and are not performance acceptance.",
            "Memory is tracked live RHI/VMA allocation bytes, not process VRAM or total physical residency.",
            "Assets are referenced by path; this package does not fully hash large cooked geometry or textures.", ""]
    for identifier, entries in grouped.items():
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        item = {"id": identifier, "processes": []}
        rows = []
        for index, (path, data, case) in enumerate(entries):
            frames = case.get("frames", [])
            valid = [f for f in frames if not f["warmup"]]
            times = np.array([f["graphMs"] for f in valid], dtype=float)
            if not len(times) or not np.isfinite(times).all():
                raise ValueError(f"No valid finite timings for {identifier}; do not fill missing samples")
            metrics = {"source": str(path), "frames": len(frames), "timedFrames": len(times),
                       "mode": data.get("mode", read(source / "Cases.json")["mode"]),
                       "graphMeanMs": float(times.mean()), "graphMedianMs": float(np.median(times)),
                       "graphP95Ms": float(np.percentile(times, 95)), "graphP99Ms": float(np.percentile(times, 99))}
            memory = [f["memory"]["liveAllocationBytes"] for f in frames]
            metrics.update(peakLiveBytes=max(memory), finalLiveBytes=memory[-1])
            item["processes"].append(metrics)
            axes[0].plot([f["frame"] for f in frames], [f.get("graphMs", np.nan) for f in frames],
                         linewidth=.8, label=f"process {index + 1}")
            axes[1].plot([f["frame"] for f in frames], np.array(memory) / 2**20,
                         linewidth=.8, label=f"process {index + 1}")
            for frame in frames:
                rows.append({"process": index + 1, **frame})
        axes[0].set(title=f"{identifier}: {item['processes'][0]['mode']}, n={len(entries)} processes", ylabel="Graph GPU (ms)")
        axes[1].set(xlabel="Frame (includes warmup and declared events)", ylabel="Live RHI allocation (MiB)")
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend()
        figure.savefig(output / f"{identifier}.png", dpi=160)
        plt.close(figure)
        save(output / f"{identifier}.json", rows)
        summary["cases"].append(item)
        text.extend([f"## {identifier}", "", f"![Frame timings and memory]({identifier}.png)", "",
                     "| Process | Timed frames | Mean / median ms | P95 / P99 ms | Peak live MiB |",
                     "| --- | ---: | ---: | ---: | ---: |"])
        for index, m in enumerate(item["processes"]):
            text.append(f"| {index + 1} | {m['timedFrames']} | {m['graphMeanMs']:.3f} / {m['graphMedianMs']:.3f} | "
                        f"{m['graphP95Ms']:.3f} / {m['graphP99Ms']:.3f} | {m['peakLiveBytes'] / 2**20:.1f} |")
        text.extend(["", "No frames removed except the declared warmup window from quantiles. "
                     "P95/P99 describe this replay; frames are not independent experiments.", ""])
    save(output / "Summary.json", summary)
    text.extend(["## Reproduce", "", "```powershell",
                 f'python -B Tools/Perf/VirtualShadowMapAcceptance.py report "{source}" "<new-report-directory>"', "```"])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (output / "plot_report.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--cases", type=Path, required=True)
    capture.add_argument("--exe", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--runs", type=int, default=3)
    capture.add_argument("--timeout", type=float, default=1800)
    capture.set_defaults(action=run)
    plot = commands.add_parser("report")
    plot.add_argument("source", type=Path)
    plot.add_argument("output", type=Path)
    plot.set_defaults(action=report)
    arguments = parser.parse_args()
    arguments.action(arguments)
