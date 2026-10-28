"""Verify VSM CPU diagnostic packages and report exclusive CPU attribution.

Run from the repository root. Durations are host elapsed milliseconds, not
thread execution time. Frames describe a replay, not independent experiments.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("vsm_acceptance", ROOT / "Tools/Perf/VirtualShadowMapAcceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=min(values), maximum=max(values))


GROUPS = ["State replacement", "Buffer creation", "Buffer upload", "Buffer descriptors",
          "Page remapping", "Demand and raster commands", "Projection", "Material programs",
          "Other VSM work", "Outside VSM scope"]


def category(name):
    return {
        "Replace VSM recording state": GROUPS[0],
        "Create VSM buffers": GROUPS[1],
        "Upload VSM buffers": GROUPS[2],
        "Acquire VSM buffer descriptors": GROUPS[3],
        "Build VSM page remapping": GROUPS[4],
        "Record VSM demand and build commands": GROUPS[5],
        "Record VSM raster commands": GROUPS[5],
        "Prepare VSM point face table": GROUPS[6],
        "Bind VSM projection resources": GROUPS[6],
        "Record VSM projection barriers and dispatch": GROUPS[6],
        "VSM face project": GROUPS[6],
        "Project all faces": GROUPS[6],
        "Prepare VSM material programs": GROUPS[7],
    }.get(name, GROUPS[8])


def profile(frame):
    node = next(n for n in frame["nodes"] if n["name"] == "Shadows")
    sections = node["sections"]
    roots = [i for i, s in enumerate(sections) if s["name"] in ("VSM lights CPU", "VSM single light CPU")]
    if not roots:
        return node, None
    assert len(roots) == 1
    root = roots[0]
    selected = {root}
    children = collections.defaultdict(list)
    for i, s in enumerate(sections):
        parent = s["parent"]
        if parent in selected:
            assert parent < i
            children[parent].append(i)
            selected.add(i)
    phases = collections.defaultdict(float)
    groups = dict.fromkeys(GROUPS, 0.0)
    faces = []
    counters = collections.Counter()
    for i in sorted(selected):
        s = sections[i]
        assert s["cpuOnly"] and s["gpuMs"] is None
        value = s["cpuMs"]
        assert math.isfinite(value) and value >= 0
        exclusive = value - sum(sections[j]["cpuMs"] for j in children[i])
        assert exclusive >= -1e-8, (s["name"], value, exclusive)
        phases[s["name"]] += exclusive
        groups[category(s["name"])] += exclusive
        work = {c["name"]: c["value"] for c in s["cpuWork"]}
        if s["name"] == "Prepare VSM buffers":
            counters.update(work)
        elif s["name"] == "Prepare VSM pipeline cache":
            counters["pipelineCacheMisses"] += work["cacheMiss"]
        elif s["name"] == "Prepare VSM projections and dirty bounds":
            counters["dirtyBounds"] += work["dirtyBounds"]
        elif s["name"] == "VSM face record":
            counters["recordedFaces"] += 1
            counters["reusedFaces"] += work["reuse"]
            faces.append({"lightType": work["lightType"], "face": work["face"],
                          "recordMs": value, "work": work})
    groups[GROUPS[-1]] = node["cpuMs"] - sections[root]["cpuMs"]
    assert groups[GROUPS[-1]] >= -1e-8
    error = abs(sum(groups.values()) - node["cpuMs"])
    assert error < 1e-8
    return node, dict(groups=groups, exclusivePhases=dict(phases), faces=faces,
                      counters=dict(counters), closureErrorMs=error,
                      scopeMs=sections[root]["cpuMs"], sectionCount=len(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Collection containing Plan.json, Runs.json and acceptance packages")
    parser.add_argument("output", type=Path, help="New report directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    plan, runs, complete = (read(source / name) for name in ("Plan.json", "Runs.json", "Complete.json"))
    assert complete == {"runs": len(runs), "status": "completed"}
    assert len({r["package"] for r in runs}) == len(runs), "Duplicated process package"
    assert [(r["scene"], r["position"], r["variant"]) for r in runs] == [
        (scene, i, mode) for scene in plan["scenes"] for i, mode in enumerate(plan["order"], 1)]
    output.mkdir(parents=True, exist_ok=False)
    processes, flat, package_hashes = [], [], {}
    identity_reference = None
    scene_inputs = {}
    for run in runs:
        package = source / run["package"]
        manifest = read(package / "Manifest.json")
        assert manifest == {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*"))
                            if p.is_file() and p.name != "Manifest.json"}
        integrity = read(package / "Integrity.json")
        assert integrity["sourceAndInputsUnchanged"] and integrity["binariesUnchanged"]
        identity = read(package / "Identity.json")
        build_identity = {key: identity[key] for key in ("commit", "sources", "binaries")}
        if identity_reference is None:
            identity_reference = build_identity
        assert build_identity == identity_reference, "Cross-process source/binary identity mismatch"
        for path, expected in identity["inputs"].items():
            archived = source / "Inputs" / Path(path).name
            assert digest(archived) == expected, f"Archived input mismatch: {path}"
        case_spec = read(package / "Cases.json")
        assert len(case_spec["cases"]) == 1
        graph = read(source / "Inputs" / case_spec["cases"][0]["graph"])
        shadow = next(n for n in graph["nodes"] if n["name"] == "Shadows")
        assert shadow["properties"].pop("benchmarkCPUProfile") == (run["variant"] == "on")
        assert not shadow["properties"].get("benchmarkCPUThreadDiagnostics", False)
        case_spec["cases"][0]["graph"] = "<profile-toggle-graph>"
        normalized = {"graph": graph, "cases": case_spec}
        assert normalized == scene_inputs.setdefault(run["scene"], normalized), "Workload differs beyond profile toggle"
        process = read(package / "Processes.json")
        assert len(process) == 1 and process[0]["exitCode"] == 0 and process[0]["captureValid"]
        assert not acceptance.runtime_errors(package / "run-1")
        _, data = acceptance.validate_capture(package / "run-1", read(package / "Cases.json"))
        assert data["mode"] == "diagnostic" and not data["performanceEligible"]
        assert data["cpuProfileDiagnostics"] == (run["variant"] == "on")
        assert len(data["cases"]) == 1 and data["cases"][0]["id"] == run["scene"]
        case = data["cases"][0]
        package_hashes[run["package"]] = digest(package / "Manifest.json")
        values, phase_values, face_values, prep_values = [], collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        rows = []
        for frame in case["frames"]:
            assert (frame["width"], frame["height"], frame["actualRenderWidth"], frame["actualRenderHeight"]) == (768, 512, 512, 341)
            node, details = profile(frame)
            assert bool(details) == (run["variant"] == "on")
            row = dict(scene=run["scene"], variant=run["variant"], process=run["position"],
                       frame=frame["frame"], warmup=frame["warmup"], nodeMs=node["cpuMs"],
                       graphCpuMs=frame["cpuMs"], graphGpuMs=frame["graphMs"], wallMs=frame["wallMs"])
            if details:
                row.update(details["groups"])
                row.update(closureErrorMs=details["closureErrorMs"], sectionCount=details["sectionCount"])
                if not frame["warmup"]:
                    for name, value in details["exclusivePhases"].items():
                        phase_values[name].append(value)
                    frame_faces = collections.defaultdict(float)
                    for face in details["faces"]:
                        frame_faces[f'{face["lightType"]}/{face["face"]}'] += face["recordMs"]
                    for key, value in frame_faces.items():
                        face_values[key].append(value)
            rows.append(row)
            if not frame["warmup"]:
                values.append(row)
                for s in frame["preparation"]:
                    prep_values[s["name"]].append(s["cpuMs"])
        assert len(values) == plan["frames"] - plan["warmup"]
        stats = {key: describe([v[key] for v in values]) for key in ["nodeMs", "graphCpuMs", "graphGpuMs", "wallMs"]}
        if run["variant"] == "on":
            stats["groups"] = {key: describe([v[key] for v in values]) for key in GROUPS}
            stats["exclusivePhases"] = {key: describe(v) for key, v in phase_values.items()}
            stats["faceRecordMs"] = {key: describe(v) for key, v in face_values.items()}
            valid_details = [profile(f)[1] for f in case["frames"] if not f["warmup"]]
            counter_names = set().union(*(d["counters"] for d in valid_details))
            stats["counterRanges"] = {k: [min(d["counters"].get(k, 0) for d in valid_details),
                                           max(d["counters"].get(k, 0) for d in valid_details)] for k in sorted(counter_names)}
        stats["preparationMs"] = {key: describe(v) for key, v in prep_values.items()}
        processes.append(dict(**run, metrics=stats, captures=case["captures"]))
        flat.extend(rows)

    summary = dict(plan=plan, source=str(source), packageManifestSHA256=package_hashes,
                   crossProcessBuildIdentityEqual=True, workloadDiffersOnlyByCPUProfileToggle=True,
                   sourceIdentitySHA256=hashlib.sha256(json.dumps(identity_reference, sort_keys=True).encode()).hexdigest(),
                   python=sys.version, platform=platform.platform(), numpy=np.__version__, matplotlib=matplotlib.__version__, pillow=PIL.__version__,
                   scope="Diagnostic CPU elapsed time. Per-frame exclusive accounting; process means equally weighted. No optimization/FPS/GPU benefit claim.", scenes={})
    for scene in plan["scenes"]:
        on = [p for p in processes if p["scene"] == scene and p["variant"] == "on"]
        off = [p for p in processes if p["scene"] == scene and p["variant"] == "off"]
        mean_total = statistics.mean(p["metrics"]["nodeMs"]["mean"] for p in on)
        groups = {g: statistics.mean(p["metrics"]["groups"][g]["mean"] for p in on) for g in GROUPS}
        assert abs(sum(groups.values()) - mean_total) < 1e-8
        hashes = {c["rawSha256"] for p in on + off for c in p["captures"]}
        assert all(len(p["captures"]) == 1 and p["captures"][0]["frame"] == plan["frames"] - 1 for p in on + off)
        images = {}
        for p in off + on:
            capture = p["captures"][0]
            pixels = np.array(Image.open(source / p["package"] / "run-1" / capture["file"]).convert("RGBA"))
            assert pixels.shape == (512, 768, 4)
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == capture["rawSha256"]
            images[p["package"]] = pixels
        reference = images[off[0]["package"]].astype(np.int16)
        differences = {}
        for name, pixels in images.items():
            delta = np.abs(pixels.astype(np.int16) - reference)
            differences[name] = dict(reference=off[0]["package"], changedPixels=int(np.any(delta, axis=2).sum()),
                changedComponents=int((delta != 0).sum()), oneByteComponents=int((delta == 1).sum()),
                maximumByteDifference=int(delta.max()), rmseBytes=float(np.sqrt(np.mean(delta.astype(float) ** 2))))
        summary["scenes"][scene] = dict(meanNodeMs=mean_total, groupMeanMs=groups,
            groupShares={g: v / mean_total for g, v in groups.items()}, outputIdentical=len(hashes) == 1,
            outputDifferences=differences,
            outputHashes={h: [p["package"] for p in on + off if p["captures"][0]["rawSha256"] == h] for h in sorted(hashes)},
            onProcessMedians=[p["metrics"]["nodeMs"]["median"] for p in on],
            offProcessMedians=[p["metrics"]["nodeMs"]["median"] for p in off])
    save(output / "Summary.json", summary)
    save(output / "Processes.json", processes)
    save(output / "Frames.json", flat)
    columns = list(dict.fromkeys(k for row in flat for k in row))
    with (output / "Frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(flat)

    colors = plt.get_cmap("tab10")(np.arange(len(GROUPS)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    for ax, scene in zip(axes, plan["scenes"]):
        ps = [p for p in processes if p["scene"] == scene and p["variant"] == "on"]
        bottom = np.zeros(len(ps))
        for g, color in zip(GROUPS, colors):
            vals = np.array([p["metrics"]["groups"][g]["mean"] for p in ps])
            ax.bar(np.arange(len(ps)), vals, bottom=bottom, color=color, label=g)
            bottom += vals
        note = " | image control mismatch retained" if not summary["scenes"][scene]["outputIdentical"] else ""
        ax.set(title=f"{scene}: CPU profile ON, n={len(ps)}{note}", ylabel="Shadows CPU mean (ms/frame)",
               xticks=range(len(ps)), xticklabels=[f'Process {p["position"]}' for p in ps])
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle(f'Exclusive CPU attribution | {plan["frames"]} frames/process; first {plan["warmup"]} warmup excluded')
    fig.savefig(output / "CPUAttribution.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained")
    for ax, scene in zip(axes, plan["scenes"]):
        rows = [r for r in flat if r["scene"] == scene and r["variant"] == "on" and not r["warmup"]]
        for key, label, color in [("nodeMs", "Shadows total", "#222222"), (GROUPS[0], GROUPS[0], colors[0]), (GROUPS[1], GROUPS[1], colors[1])]:
            ax.plot(range(len(rows)), [r[key] for r in rows], label=label, color=color, linewidth=.8)
        for boundary in range(plan["frames"] - plan["warmup"], len(rows), plan["frames"] - plan["warmup"]):
            ax.axvline(boundary - .5, color="grey", linestyle="--", alpha=.5)
        ax.set(title=scene, ylabel="CPU elapsed (ms/frame)", xlabel="Retained frame index; dashed lines separate processes")
        ax.legend(ncol=3); ax.grid(alpha=.2)
    fig.savefig(output / "CPUFrames.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, scene in zip(axes, plan["scenes"]):
        for x, variant in enumerate(("off", "on")):
            ps = [p for p in processes if p["scene"] == scene and p["variant"] == variant]
            vals = [p["metrics"]["nodeMs"]["median"] for p in ps]
            ax.scatter(np.full(len(vals), x), vals, s=45)
            for p, v in zip(ps, vals): ax.annotate(str(p["position"]), (x, v), xytext=(6, 0), textcoords="offset points", fontsize=8)
        ax.set(title=scene, ylabel="Shadows CPU process median (ms)", xticks=[0, 1], xticklabels=["Profile OFF", "Profile ON"], xlim=(-.4, 1.4))
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Instrumentation control | same binary, diagnostic recipe, 3 processes per mode")
    fig.savefig(output / "InstrumentationControl.png", dpi=170); plt.close(fig)

    text = ["# VSM CPU attribution", "", "Diagnostic elapsed CPU time; no optimization acceptance or GPU/FPS inference.", "",
            "Fixed cameras, temporal jitter, 768 x 512 output / 512 x 341 DLSS Quality internal resolution. Resident caster cache active.",
            "Three independent processes per scene/mode in OFF, ON, ON, OFF, OFF, ON order. 128 frames/process; only first 16 excluded.",
            "Validation and CPU thread diagnostics off. Existing diagnostic metadata readbacks retained; color captured at frame 127.",
            "Durations include clock/counter overhead. Frame distributions are descriptive, not independent-sample confidence intervals.", "",
            "![CPU attribution](CPUAttribution.png)", "", "| Exclusive category | LookDev ms / share | Sponza ms / share |", "| --- | ---: | ---: |"]
    for g in GROUPS:
        text.append("| " + g + " | " + " | ".join(f'{summary["scenes"][s]["groupMeanMs"][g]:.4f} / {100 * summary["scenes"][s]["groupShares"][g]:.2f}%' for s in plan["scenes"]) + " |")
    text += ["", "Shares use mean exclusive durations / mean total from the same retained frames; they sum to 100%. Parent-inclusive timings are not added to their children.",
             "Other VSM work includes explicitly retained parent residuals, allocation/counter instrumentation, admission, cache preparation, publication and local teardown. Outside VSM scope is Shadows node minus the complete executeProfiled scope.",
             "State replacement can release old resources on this CPU thread; it is not the complete frame retirement cost. Shared preparation is exported separately and cannot all be attributed to VSM.", "",
             "![CPU frame series](CPUFrames.png)", "", "![Instrumentation control](InstrumentationControl.png)", "",
             "| Scene | OFF process medians ms | ON process medians ms | Final output |", "| --- | --- | --- | --- |"]
    for scene, s in summary["scenes"].items():
        status = "All 6 raw hashes identical" if s["outputIdentical"] else "MISMATCH: retained, output equivalence not established"
        text.append(f'| {scene} | ' + ", ".join(f"{v:.4f}" for v in s["offProcessMedians"]) + " | " + ", ".join(f"{v:.4f}" for v in s["onProcessMedians"]) + f" | {status} |")
    text += ["", "Control is descriptive with 3 processes per mode. It does not prove zero instrumentation overhead, and diagnostic times are not substituted for earlier ordinary timings.", ""]
    if any(not s["outputIdentical"] for s in summary["scenes"].values()):
        text += ["**Output-control failure retained:** CPU attribution remains descriptive and all frames/processes stay included. This report does not certify full-image equivalence. See Summary.json for every capture hash and owning process, and the repository CPU profiling note for the bounded followup.", ""]
    for scene in plan["scenes"]:
        ps = [p for p in processes if p["scene"] == scene and p["variant"] == "on"]
        phases = ps[0]["metrics"]["exclusivePhases"].keys()
        text += [f"## {scene} detailed phases", "", "| Phase (exclusive) | Mean ms/frame |", "| --- | ---: |"]
        aggregate = {k: statistics.mean(p["metrics"]["exclusivePhases"][k]["mean"] for p in ps) for k in phases}
        for k, v in sorted(aggregate.items(), key=lambda item: -item[1]): text.append(f"| {k} | {v:.6f} |")
        text += ["", "| Light type / face | Mean inclusive record ms/frame |", "| --- | ---: |"]
        for k in ps[0]["metrics"]["faceRecordMs"]:
            text.append(f'| {k} | {statistics.mean(p["metrics"]["faceRecordMs"][k]["mean"] for p in ps):.6f} |')
        text += ["", "Light types: 0 directional, 1 spot, 2 point face. Multiple lights of the same type/face would be grouped.", "",
                 "| Buffer/work counter | Retained-frame min .. max |", "| --- | ---: |"]
        for k in ps[0]["metrics"]["counterRanges"]:
            ranges = [p["metrics"]["counterRanges"][k] for p in ps]
            text.append(f"| {k} | {min(r[0] for r in ranges)} .. {max(r[1] for r in ranges)} |")
        text += ["", "| Shared graph preparation (inclusive, outside graph CPU) | Mean ms/frame |", "| --- | ---: |"]
        for k in ps[0]["metrics"]["preparationMs"]:
            text.append(f'| {k} | {statistics.mean(p["metrics"]["preparationMs"][k]["mean"] for p in ps):.6f} |')
        text.append("")
    text += ["## Reproduce", "", f"Python {platform.python_version()}, NumPy {np.__version__}, Matplotlib {matplotlib.__version__}, Pillow {PIL.__version__}.", "",
             "Raw package manifests, runtime errors, input/source integrity and capture eligibility are rechecked before plotting. Large cooked assets remain path-identified, not fully content-hashed.", "",
             "```powershell", f'python -B Tools/Perf/VirtualShadowMapCPUProfile.py "{source}" "<new-output-directory>"', "```", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    (output / "plot_report.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    main()
