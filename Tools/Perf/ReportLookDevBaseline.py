"""Offline qualification of the fixed LookDev 1 SPP accumulation baseline."""
import argparse
import csv
import math
from pathlib import Path
import platform
import statistics

import MaterialBaseline as mb


STABILITY_LIMIT = 0.10
SCOPES = ("PathTrace", "Shading", "Graph")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spread(values):
    median = statistics.median(values)
    return {"medianMs": median, "minMs": min(values), "maxMs": max(values),
            "rangeOverMedian": (max(values) - min(values)) / median}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(source, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, output = source.resolve(), output.resolve()
    require(not output.exists(), "Use a new report directory")
    require(not output.is_relative_to(source), "Report must be outside sealed evidence")
    verification = mb.verify(source)
    require(verification["runCount"] == 3, "Exactly three independent processes required")
    require(not verification["invalidHDR"], "Finite HDR images required")
    config = mb.read(source / "fixtures/Cases.json")
    for key, expected in {"frames": 768, "warmupFrames": 256, "timingFrames": 256}.items():
        require(config.get(key) == expected, f"Expected {key}={expected}")
    require(len(config["cases"]) == 1, "Single fixed LookDev case required")
    spec = config["cases"][0]
    expected = {"width": 1920, "height": 1080, "samplesPerFrame": 1,
                "accumulatedSpp": 768, "maxDepth": 12, "backend": "PathTrace",
                "scene": "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf",
                "requiredTiming": ["PathTrace", "Path trace shading"],
                "output": "PathTrace.color", "useRenderView": True,
                "renderViewCameraNode": "PathTrace"}
    for key, value in expected.items():
        require(spec.get(key) == value, f"Unexpected fixed-workload field {key}")
    require(spec.get("frames", config["frames"]) == 768, "Case frame override differs")
    rows, processes, image_paths, graph_checks, environment_transitions = [], [], [], [], []
    for path in sorted(source.glob("run-*/MaterialBaseline.json")):
        case = mb.read(path)["cases"][0]
        nodes = [n for n in case["resolvedGraph"]["nodes"] if n["name"] == "PathTrace"]
        require(len(nodes) == 1 and nodes[0]["type"] == "ScenePathTracePass",
                "Production ScenePathTracePass required")
        props = nodes[0]["properties"]
        require(props.get("samples") == 1 and props.get("accumulate") is True,
                "Resolved graph must accumulate exactly one sample per frame")
        require(props.get("maxDepth") == 12 and props.get("bsdf") == "openpbr",
                "Resolved graph must use default OpenPBR depth")
        require(props.get("cacheMode") == "off", "Expected cacheMode=off")
        graph_checks.append({"process": path.parent.name, "type": nodes[0]["type"],
                             "properties": props})
        environment_transitions.append({"process": path.parent.name,
                                        "transitions": case.get("environmentTransitions", [])})
        image_path = path.parent / case["image"]
        image_paths.append(image_path)
        values = {scope: [] for scope in SCOPES}
        for frame in case["frames"]:
            node = next(n for n in frame["nodes"] if n["name"] == "PathTrace")
            shading = next(s for s in node["sections"] if s["name"] == "Path trace shading")
            times = (node["gpuMs"], shading["gpuMs"], frame["graphMs"])
            require(all(isinstance(t, (int, float)) and math.isfinite(t) and t > 0 for t in times),
                    "All measured GPU scopes must have positive finite timings")
            row = {"process": path.parent.name, "frame": frame["frame"]}
            for scope, value in zip(SCOPES, times):
                values[scope].append(value)
                row[scope + "Ms"] = value
            rows.append(row)
        processes.append({"process": path.parent.name,
                          **{scope: spread(values[scope]) for scope in SCOPES}})
    raw = image_paths[0].read_bytes()
    exact = all(path.read_bytes() == raw for path in image_paths[1:])
    case = mb.read(sorted(source.glob("run-*/MaterialBaseline.json"))[0])["cases"][0]
    pixels = np.frombuffer(raw, dtype="<f2" if case["format"] == "RGBA16F" else "<f4")
    pixels = pixels.reshape(1080, 1920, 4).astype(np.float32)
    require(np.isfinite(pixels).all(), "Nonfinite preview image")
    aggregate = {scope: spread([p[scope]["medianMs"] for p in processes]) for scope in SCOPES}
    stable = all(aggregate[scope]["rangeOverMedian"] <= STABILITY_LIMIT for scope in ("PathTrace", "Graph"))
    summary = {"version": 1, "input": str(source), "verification": verification,
               "contract": {**expected, "frames": 768, "warmupFrames": 256, "timingFrames": 256},
               "stabilityLimit": STABILITY_LIMIT,
               "stabilityRule": "PathTrace and Graph three process medians: (max-min)/median <= 0.10; Shading diagnostic only",
               "processes": processes, "aggregateProcessMedians": aggregate,
               "rawImagesBitwiseEqual": exact, "finiteImages": True,
               "stable": stable, "qualified": stable and exact,
               "meaning": "1 SPP baseline calibration only; no comparison or speedup against 4 SPP.",
               "resolvedGraphChecks": graph_checks,
               "environmentTransitions": environment_transitions,
               "accumulationMeaning": "768 SPP is the nominal frame budget, not a readback of final accumulationFrame; environment or streaming changes can reset accumulation",
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "matplotlib": matplotlib.__version__},
               "inputHashes": {str(p): mb.digest(p) for p in
                               [source / "Manifest.json", source / "Identity.json",
                                source / "Process.json", source / "fixtures/Cases.json",
                                Path(__file__).resolve(), Path(mb.__file__).resolve(), *image_paths]}}
    output.mkdir(parents=True)
    mb.save(output / "Summary.json", summary)
    write_csv(output / "Frames.csv", rows)
    write_csv(output / "Processes.csv", [{"process": p["process"],
              **{scope + "MedianMs": p[scope]["medianMs"] for scope in SCOPES},
              **{scope + "FrameRangeOverMedian": p[scope]["rangeOverMedian"] for scope in SCOPES}}
              for p in processes])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for column, scope in enumerate(SCOPES):
        medians = [p[scope]["medianMs"] for p in processes]
        axes[0, column].plot(range(1, 4), medians, "o-")
        axes[0, column].set(xticks=[1, 2, 3], xlabel="Independent process", ylabel="GPU ms",
                            title=f"{scope}: median {statistics.median(medians):.3f} ms\n"
                                  f"range/median {aggregate[scope]['rangeOverMedian']:.1%}")
        for process in processes:
            samples = [r for r in rows if r["process"] == process["process"]]
            axes[1, column].plot([r["frame"] for r in samples], [r[scope + "Ms"] for r in samples],
                                 label=process["process"], linewidth=0.8, alpha=0.8)
        axes[1, column].set(xlabel="Timed frame", ylabel="GPU ms")
        axes[1, column].legend()
    fig.suptitle("LookDev 1920 x 1080 | 1 SPP accumulation | baseline calibration")
    fig.savefig(output / "Timings.png", dpi=150)
    plt.close(fig)
    # A display-only transform; raw float bytes above remain the equality authority.
    rgb = np.maximum(pixels[:, :, :3], 0)
    rgb = rgb / (1 + rgb)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    plt.imsave(output / "Preview.png", rgb)
    table = "\n".join(f"| {scope} | {aggregate[scope]['medianMs']:.4f} | "
                      f"{aggregate[scope]['rangeOverMedian']:.2%} |" for scope in SCOPES)
    status = "QUALIFIED" if summary["qualified"] else "NOT QUALIFIED"
    text = f"""# LookDev 1 SPP baseline: {status}

Default OpenPBRDefault scene, 1920 x 1080, depth 12, one sample per frame with accumulation.
Each of three independent processes renders 768 frames (nominal 768 SPP budget); frames 256-511 are timed.
Resolved production ScenePathTracePass confirms samples=1 and accumulate=true.
The harness does not read back the final accumulationFrame. Environment or streaming changes can reset
accumulation, so 768 is not a verified final accumulated sample count. Environment transitions are retained in Summary.json.

| Scope | Median of process medians (ms) | Process range / median |
| --- | ---: | ---: |
{table}

Predeclared stability threshold: PathTrace and Graph process range / median <= 10%; Shading is diagnostic only.
Stable: **{stable}**. Finite HDR: **True**. Raw images bitwise equal: **{exact}**.
Per-frame measurements are correlated samples, not independent experiment replicates.
Graph is the captured GPU graph scope; PathTrace and shading are pass scopes, not end-to-end application time.
This is a new workload baseline: no 4 SPP speedup, sample-normalized comparison, or previous optimization acceptance is inferred.
Existing disk caches are retained; warmup is excluded. This report does not establish exclusive GPU use or locked clocks.

![Timings](Timings.png)

![Preview](Preview.png)

Preview uses Reinhard then sRGB for display only. Equality uses original floating-point RGBA bytes.

[Raw evidence](<{source.as_posix()}>) · [Manifest](<{(source / 'Manifest.json').as_posix()}>) ·
[Summary](Summary.json) · [Frame CSV](Frames.csv) · [Process CSV](Processes.csv) ·
[Report script](<{Path(__file__).resolve().as_posix()}>)
"""
    (output / "Report.md").write_text(text, encoding="utf-8")
    print(f"{status}: {output / 'Report.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Sealed MaterialBaseline.py run output")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside input")
    args = parser.parse_args()
    report(args.input, args.output)
