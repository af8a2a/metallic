"""Paired process A/B for the test-only Coverage VM and production native evaluator.

This measures compute evaluator dispatches, never raster/ray-query frame speedups.
The old interpreter is compiled exclusively from tests/rhi/shaders.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CASES = ["control", "simple", "complex", "mixed", "edge"]
SCOPE = "Coverage evaluator compute dispatch; excludes rasterization, traversal, lighting and presentation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inputs(executable):
    files = [executable, Path(__file__), *executable.parent.glob("*.dll")]
    files += list((ROOT / "Shaders").rglob("*.slang"))
    files += list((ROOT / "Shaders").rglob("*.hlsli"))
    files += list((ROOT / "Source/Runtime/Render/Material").glob("*.cpp"))
    files += list((ROOT / "Source/Runtime/Material").glob("*.cpp"))
    files += [ROOT / "tests/rhi/MaterialCoverageBenchmarkTests.cpp",
              ROOT / "tests/rhi/shaders/MaterialCoverageBenchmark.slang",
              ROOT / "tests/rhi/shaders/LegacyCoverageOracle.slang"]
    return {str(path.resolve()): digest(path) for path in sorted(set(files))}


def run(args):
    executable = args.exe.resolve()
    require(executable.is_file(), "Missing RHI executable")
    require(args.blocks >= 1, "At least one ABBA block is required")
    require(not args.output.exists(), "Use a fresh output directory")
    args.output.mkdir(parents=True)
    lock = ROOT / "build/shader-experiment.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "runner": str(Path(__file__).resolve())}, handle)
    try:
        sources = inputs(executable)
        try:
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True, timeout=20).strip()
        except (OSError, subprocess.SubprocessError):
            gpu = "unavailable"
        identity = {"version": 1, "scope": SCOPE, "blocks": args.blocks, "gpu": gpu,
                    "platform": platform.platform(), "python": sys.version, "inputs": sources,
                    "cachePolicy": "Shared ordinary shader/driver caches; compile and PSO setup excluded; 16 warmup dispatches per case/process",
                    "clockPolicy": "unaltered", "exclusivity": "cooperative lock only; system-wide GPU exclusivity not proven",
                    "runs": []}
        save(args.output / "Identity.json", identity)
        env = dict(os.environ)
        # Match the ordinary experiment runner: do not inherit an unrelated
        # profiler, shader artifact, compiler override or scene experiment.
        for name in list(env):
            if name.startswith("METALLIC_"):
                del env[name]
        env.update(METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0")
        identity["environment"] = {k: v for k, v in env.items() if k.startswith(("METALLIC_", "VK_"))}
        for index, arm in enumerate(["legacy", "native", "native", "legacy"] * args.blocks):
            require(inputs(executable) == sources, "Sources/executable changed during experiment")
            directory = args.output / f"run-{index:02d}-{arm}"
            directory.mkdir()
            command = [str(executable), "--gtest_filter=*material_coverage_codegen_benchmark*",
                       "--rhi-no-validation", "--output-dir", str(directory.resolve()),
                       "--gtest_output=xml:" + str((directory / "gtest.xml").resolve())]
            record = {"index": index, "arm": arm, "directory": directory.name, "command": command,
                      "started": dt.datetime.now(dt.timezone.utc).isoformat()}
            env["METALLIC_COVERAGE_BENCHMARK_ARM"] = arm
            with (directory / "Process.log").open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    record["exitCode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    record["timeout"] = True
                    record["exitCode"] = None
            record["ended"] = dt.datetime.now(dt.timezone.utc).isoformat()
            identity["runs"].append(record)
            save(args.output / "Identity.json", identity)
            require(record["exitCode"] == 0, f"Benchmark failed: {directory / 'Process.log'}")
            print(f"{index + 1}/{args.blocks * 4}: {arm} complete", flush=True)
        require(inputs(executable) == sources, "Sources/executable changed during experiment")
        save(args.output / "Manifest.json", {str(p.relative_to(args.output)): digest(p)
             for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "Manifest.json"})
    finally:
        lock.unlink()
    report(args.output, args.output / "Report")


def verify(directory):
    import numpy as np
    import xml.etree.ElementTree as ET
    manifest = read(directory / "Manifest.json")
    for path, checksum in manifest.items():
        resolved = (directory / path).resolve()
        require(resolved.is_relative_to(directory.resolve()), "Artifact escapes evidence directory")
        require(digest(resolved) == checksum, f"Artifact changed: {path}")
    identity = read(directory / "Identity.json")
    records = identity["runs"]
    require(len(records) == identity["blocks"] * 4, "Incomplete ABBA blocks")
    medians, frames, images, workload, spirv = [], [], {}, {}, {}
    for index, record in enumerate(records):
        require(record["index"] == index and record["arm"] == ["legacy", "native", "native", "legacy"][index % 4]
                and record["exitCode"] == 0, "Invalid independent process order/status")
        root = directory / record["directory"]
        xml = ET.parse(root / "gtest.xml").getroot()
        tests = list(xml.iter("testcase"))
        require(len(tests) == 1 and not any(t.find("failure") is not None or t.find("skipped") is not None for t in tests), "Skipped/failed test")
        raw = read(root / "CoverageBenchmark.json")
        require(raw["arm"] == record["arm"] and not raw["validation"] and raw["scope"] == SCOPE and
                raw["warmupFrames"] == 16 and raw["sampleFrames"] == 32 and
                [c["id"] for c in raw["cases"]] == CASES, "Wrong benchmark configuration")
        process_medians, process_frames = {}, {}
        for case in raw["cases"]:
            name = case["id"]
            require(case["size"] == 512 and case["repeats"] == 4 and len(case["warmup"]) == 16 and len(case["frames"]) == 32,
                    "Workload/timing window changed")
            signature = {k: case[k] for k in ("size", "repeats", "materials")}
            require(workload.setdefault(name, signature) == signature, "A/B workload changed")
            all_frames = case["warmup"] + case["frames"]
            require([f["frame"] for f in all_frames] == list(range(48)), "Missing or duplicate frames")
            require(all(isinstance(f["gpuMs"], (float, int)) and math.isfinite(f["gpuMs"]) and f["gpuMs"] > 0 and
                        math.isfinite(f["graphMs"]) and f["graphMs"] >= f["gpuMs"] for f in all_frames), "Invalid GPU timestamp")
            process_medians[name] = statistics.median(f["gpuMs"] for f in case["frames"])
            process_frames[name] = [f["gpuMs"] for f in all_frames]
            data = np.fromfile(root / case["image"], dtype="<f4")
            require(data.size == 512 * 512 * 4 and np.isfinite(data).all(), "Invalid/nonfinite readback")
            data = data.reshape((-1, 4))
            key = (name, record["arm"])
            if key in images:
                require(np.array_equal(images[key], data), "Same-arm readback not deterministic")
            else:
                images[key] = data
            spirv_hash = digest(root / f"{name}.spv")
            require(spirv.setdefault(key, spirv_hash) == spirv_hash, "Same-arm shader identity changed")
        medians.append(process_medians)
        frames.append(process_frames)
    correctness, summary = {}, {}
    for name in CASES:
        distinct_executable = spirv[(name, "legacy")] != spirv[(name, "native")]
        require(name == "control" or distinct_executable, f"Custom case used identical legacy/native SPIR-V: {name}")
        a, b = images[(name, "legacy")], images[(name, "native")]
        difference = np.abs(a[:, 0] - b[:, 0])
        exact_acceptance = np.array_equal(a[:, 1:], b[:, 1:])
        maximum = float(difference.max())
        correctness[name] = {"maxOpacitySumAbsoluteError": maximum, "opacitySumAbsoluteTolerance": 2e-5,
                             "acceptedAndFlagsExact": exact_acceptance, "pixels": len(a),
                             "legacySpirvSha256": spirv[(name, "legacy")], "nativeSpirvSha256": spirv[(name, "native")],
                             "distinctExecutable": distinct_executable,
                             "passed": exact_acceptance and maximum <= 2e-5}
        gains, pairs = [], []
        for i in range(0, len(records), 2):
            baseline = i if records[i]["arm"] == "legacy" else i + 1
            candidate = i + 1 if baseline == i else i
            av, bv = medians[baseline][name], medians[candidate][name]
            gains.append(100 * (1 - bv / av))
            pairs.append({"legacyProcess": baseline, "nativeProcess": candidate, "legacyMs": av, "nativeMs": bv, "gainPercent": gains[-1]})
        n = len(gains)
        # Two-sided Student t interval over independent paired-process gains.
        critical = {1: 12.7062047364, 3: 3.1824463053, 5: 2.5705818356, 7: 2.3646242510,
                    9: 2.2621571629, 11: 2.2009851601, 13: 2.1603686565, 15: 2.1314495456}
        require(n - 1 in critical, "Supported ABBA block count is 1..8")
        mean = statistics.mean(gains)
        half = critical[n - 1] * statistics.stdev(gains) / math.sqrt(n)
        summary[name] = {"pairs": pairs, "nPairs": n, "gainPercentMean": mean,
                         "gainPercent95CI": [mean - half, mean + half],
                         "decision": "correctness-failed" if not correctness[name]["passed"] else
                                     "improved" if mean - half > 0 else "regressed" if mean + half < 0 else "inconclusive"}
    return {"scope": SCOPE, "identity": identity, "processMediansMs": medians,
            "framesMs": frames, "correctness": correctness, "summary": summary}


def report(directory, destination):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = verify(directory)
    require(not destination.exists(), "Use a fresh report directory")
    destination.mkdir(parents=True)
    save(destination / "PlotData.json", data)
    shutil.copyfile(__file__, destination / Path(__file__).name)
    records = data["identity"]["runs"]
    fig, axes = plt.subplots(1, len(CASES), figsize=(16, 4.8))
    for ax, name in zip(axes, CASES):
        for arm, color in (("legacy", "#b34b40"), ("native", "#25749f")):
            x = [i for i, r in enumerate(records) if r["arm"] == arm]
            y = [data["processMediansMs"][i][name] for i in x]
            ax.plot(x, y, "o-", color=color, label=arm)
        ax.set_title(name)
        ax.set_xlabel("Independent process index")
        ax.set_ylabel("Dispatch GPU ms (32-frame median)")
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Coverage evaluator only: 512 x 512 threads, 4 evaluations/thread; 16 warmup frames")
    fig.tight_layout(); fig.savefig(destination / "ProcessMedians.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, len(CASES), figsize=(16, 4.8))
    for ax, name in zip(axes, CASES):
        for i, r in enumerate(records):
            ax.plot(range(48), data["framesMs"][i][name], color="#b34b40" if r["arm"] == "legacy" else "#25749f", alpha=.5)
        ax.axvline(15.5, color="black", linestyle="--")
        ax.set_title(name); ax.set_xlabel("Frame (first 16 excluded)"); ax.set_ylabel("Dispatch GPU ms"); ax.grid(alpha=.25)
    fig.suptitle("Within-process warmup and timing trajectories (frames are not independent experiment samples)")
    fig.tight_layout(); fig.savefig(destination / "FrameTimelines.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(CASES):
        item = data["summary"][name]
        if item["decision"] == "correctness-failed":
            ax.text(i, 0, "correctness failed", rotation=90, ha="center")
            continue
        mean = item["gainPercentMean"]
        ax.scatter([i] * item["nPairs"], [p["gainPercent"] for p in item["pairs"]], color="#777777", alpha=.7)
        ax.errorbar(i, mean, yerr=mean - item["gainPercent95CI"][0], fmt="s", color="#25749f", capsize=5)
    ax.axhline(0, color="black", linewidth=1); ax.set_xticks(range(len(CASES)), CASES)
    ax.set_ylabel("Dispatch reduction (%) — positive is faster")
    ax.set_title(f"Paired process gains: n={len(records)//2}; mean and 95% Student t CI; points are pairs")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(destination / "PairedGains.png", dpi=160); plt.close(fig)
    rows = []
    for name in CASES:
        item, gate = data["summary"][name], data["correctness"][name]
        lo, hi = item["gainPercent95CI"]
        rows.append(f"| {name} | {item['gainPercentMean']:.2f}% | [{lo:.2f}, {hi:.2f}]% | {gate['maxOpacitySumAbsoluteError']:.8g} | {gate['acceptedAndFlagsExact']} | {item['decision']} |")
    text = f"""# Coverage native codegen A/B

{SCOPE}. No full-scene or editor speedup is established.

GPU: {data['identity']['gpu']}. {len(records)} independent processes in ABBA blocks; each process has 5 fixed workloads,
16 excluded warmup dispatches and 32 measured dispatches per workload. Statistical unit is the paired process median.
All frames retained; no outlier removal. The control measures no custom coverage. `edge` stresses semantics and is not representative content.
Compiler/PSO setup, output readback and presentation are outside the node timestamp. Shared warm shader/driver caches;
unaltered clocks. The cooperative lock does not prove system-wide GPU exclusivity. Validation is disabled for timings.

| Workload | Mean dispatch reduction | 95% paired t CI | Max opacity sum error | Acceptance/flags exact | Result |
|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

Opacity sums use absolute tolerance 2e-5; acceptance, alpha-dependency flags and alpha channel must match exactly.
Same-arm outputs and SPIR-V identities must remain identical across independent runs.
Custom cases must have distinct legacy/native SPIR-V hashes (control may match); hashes are retained in PlotData.json.
Failed correctness cases are not performance wins.
Small process counts yield uncertain intervals; frame counts are not substituted for process counts.

![Process medians](ProcessMedians.png)
![Paired gains](PairedGains.png)
![Frame trajectories](FrameTimelines.png)

Source: `{directory.resolve()}`. Raw manifest SHA-256: `{digest(directory / 'Manifest.json')}`.
Field mapping: `CoverageBenchmark.json cases[].frames[].gpuMs` -> per-process median (ms); paired gain = 100*(1-native/legacy).
Python {platform.python_version()}, NumPy {np.__version__}, matplotlib {matplotlib.__version__}.

Reproduce without GPU work:
`python -B Tools/Perf/MeasureCoverageCodegen.py report --input "{directory.resolve()}" --output "<new-report-directory>"`
"""
    (destination / "Report.md").write_text(text, encoding="utf-8")
    require(all(v["passed"] for v in data["correctness"].values()), f"Correctness failed; evidence retained in {destination}")
    print(destination / "Report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--exe", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--blocks", type=int, choices=range(1, 9), default=2)
    capture.add_argument("--timeout", type=int, default=1200)
    for name in ("report", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--input", type=Path, required=True)
        if name == "report":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "report":
        report(args.input, args.output)
    else:
        data = verify(args.input)
        print(json.dumps({"correctness": data["correctness"], "summary": data["summary"]}, indent=2))
        require(all(v["passed"] for v in data["correctness"].values()), "Coverage correctness failed")


if __name__ == "__main__":
    main()
