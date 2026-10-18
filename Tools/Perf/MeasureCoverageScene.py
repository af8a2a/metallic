"""Serial whole-render-graph Coverage A/B with transactional shader restoration."""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

import MaterialBaseline as baseline
from MeasureCoverageCodegen import require, digest, read, save

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "Shaders/Modules/Material/Coverage.slang"
LOCK = ROOT / "build/shader-experiment.lock"
SCOPE = "Whole render graph GPU graphics-queue envelope, including graph waits; excludes CPU, presentation, loading, compilation and readback"


def snapshot(exes, cases, legacy):
    paths = set(baseline.asset_paths(cases))
    paths.update([Path(__file__), ROOT / "Tools/Perf/MaterialBaseline.py", ROOT / "Tools/Perf/MeasureCoverageCodegen.py", ROOT / "Tools/Perf/PrepareCoverageSceneBinaries.py", legacy])
    for directory, patterns in [("Source", ["*.cpp", "*.h"]), ("Shaders", ["*.slang", "*.hlsli"]), ("tests/rhi", ["*.cpp", "*.slang"])]:
        for pattern in patterns:
            paths.update((ROOT / directory).rglob(pattern))
    for exe in exes.values():
        paths.add(exe)
        paths.update(exe.parent.glob("*.dll"))
    paths.discard(TARGET)
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def restore(output):
    journal = read(output / "Transaction.json")
    require(Path(journal["target"]).resolve() == TARGET.resolve(), "Unexpected transaction target")
    current = digest(TARGET)
    require(current in (journal["nativeHash"], journal["installedHash"], journal.get("previousHash")), "Coverage.slang changed externally; preserving it")
    backup = output / "NativeCoverage.slang"
    require(digest(backup) == journal["nativeHash"], "Native backup changed")
    TARGET.write_bytes(backup.read_bytes())
    journal.update(installedHash=journal["nativeHash"], restored=True)
    save(output / "Transaction.json", journal)


def run(args):
    output, cases = args.output.resolve(), args.cases.resolve()
    exes = {"legacy": args.legacy_exe.resolve(), "native": args.native_exe.resolve()}
    legacy = args.legacy_shader.resolve()
    require(all(p.is_file() for p in [*exes.values(), legacy, cases]), "Missing input")
    require(not output.exists(), "Use a fresh evidence directory")
    require(not (ROOT / "build/material-phase0.lock").exists(), "Material baseline experiment already running")
    LOCK.parent.mkdir(exist_ok=True)
    with LOCK.open("x") as stream:
        json.dump({"pid": os.getpid(), "output": str(output)}, stream)
    installed = digest(TARGET)
    try:
        output.mkdir(parents=True)
        shutil.copyfile(TARGET, output / "NativeCoverage.slang")
        shutil.copyfile(legacy, output / "LegacyCoverage.slang")
        build_evidence = args.build_evidence.resolve()
        require((build_evidence / "BuildIdentity.json").is_file(), "Missing paired build evidence")
        builds = read(build_evidence / "BuildIdentity.json")
        require(builds["sourceRestored"], "Build source restoration not confirmed")
        for arm, exe in exes.items():
            require(builds["builds"][arm]["exitCode"] == 0, "Paired build failed")
            for filename, checksum in builds["builds"][arm]["binaries"].items():
                require(digest(exe.parent / filename) == checksum, "Executable/DLL differs from paired build identity")
        legacy_dlls = {k:v for k,v in builds["builds"]["legacy"]["binaries"].items() if k.lower().endswith(".dll")}
        native_dlls = {k:v for k,v in builds["builds"]["native"]["binaries"].items() if k.lower().endswith(".dll")}
        require(legacy_dlls == native_dlls, "Paired runtime DLLs differ")
        (output / "build-evidence").mkdir()
        for evidence in build_evidence.iterdir():
            if evidence.is_file():
                shutil.copyfile(evidence, output / "build-evidence" / evidence.name)
        shutil.copytree(cases.parent, output / "fixtures")
        (output / "graphs").mkdir()
        for spec in read(cases)["cases"]:
            shutil.copyfile(cases.parent / spec["graph"], output / "graphs" / (spec["id"] + ".json"))
        require(digest(legacy) != installed, "Native and legacy shader sources are identical")
        journal = {"target": str(TARGET), "nativeHash": installed, "installedHash": installed, "restored": False}
        save(output / "Transaction.json", journal)
        sources = snapshot(exes, cases, legacy)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()
        identity = {"scope": SCOPE, "blocks": args.blocks, "gpu": gpu, "sources": sources, "cases": read(cases), "runs": [],
                    "platform": platform.platform(), "exes": {k: str(v) for k,v in exes.items()},
                    "cachePolicy": "Retain shared shader/driver disk caches; fixed per-case warmup; compilation excluded",
                    "clockPolicy": "Unaltered", "exclusivity": "Cooperative lock only; system-wide GPU exclusivity not proven"}
        save(output / "Identity.json", identity)
        env = {k:v for k,v in os.environ.items() if not k.startswith("METALLIC_")}
        require(not env.get("VK_INSTANCE_LAYERS"), "Injected Vulkan layers invalidate normal timing")
        env.update(METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0", METALLIC_MATERIAL_BASELINE_CASES=str(cases))
        save(output / "Environment.json", {k:v for k,v in env.items() if k.startswith(("VK_", "METALLIC_"))})
        plan = [(a, False) for a in ["legacy", "native", "native", "legacy"] * args.blocks]
        if args.receipts:
            plan += [("legacy", True), ("native", True)]
        for index, (arm, diagnostic) in enumerate(plan):
            require(snapshot(exes, cases, legacy) == sources, "Inputs changed during experiment")
            require(digest(TARGET) == installed, "Coverage shader changed externally")
            payload = (output / ("LegacyCoverage.slang" if arm == "legacy" else "NativeCoverage.slang")).read_bytes()
            import hashlib
            previous = installed
            installed = hashlib.sha256(payload).hexdigest()
            journal.update(installedHash=installed, previousHash=previous, restored=False)
            save(output / "Transaction.json", journal)  # recoverable even if interrupted before write
            TARGET.write_bytes(payload)
            directory = output / f"run-{index:02d}-{arm}{'-diagnostic' if diagnostic else ''}"
            directory.mkdir()
            process_env = dict(env)
            if diagnostic:
                process_env["METALLIC_PIPELINE_RECIPE_DIRECTORY"] = str(directory / "recipes")
            command = [str(exes[arm]), "--gtest_filter=*material_phase0_baseline*", "--rhi-no-validation", "--output-dir", str(directory), "--gtest_output=xml:" + str(directory / "Tests.xml")]
            record = {"index": index, "arm": arm, "diagnostic": diagnostic, "directory": directory.name, "command": command, "shaderHash": installed,
                      "started": dt.datetime.now(dt.timezone.utc).isoformat()}
            print(f"Starting {index+1}/{len(plan)} {arm}, diagnostic={diagnostic}", flush=True)
            with (directory / "Process.log").open("w", encoding="utf-8") as log:
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=process_env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        record["exitCode"] = process.wait(timeout=args.timeout)
                    except BaseException:
                        # Stop only this runner-owned child before restoring its shader source.
                        process.kill()
                        process.wait(timeout=30)
                        raise
                except subprocess.TimeoutExpired:
                    record.update(exitCode=None, timeout=True)
            record["ended"] = dt.datetime.now(dt.timezone.utc).isoformat()
            identity["runs"].append(record)
            save(output / "Identity.json", identity)
            require(record["exitCode"] == 0, f"Process failed; see {directory / 'Process.log'}")
        require(snapshot(exes, cases, legacy) == sources, "Inputs changed during experiment")
    finally:
        try:
            if (output / "Transaction.json").exists():
                restore(output)
        finally:
            LOCK.unlink()
    baseline.seal(output)
    print(f"Evidence sealed: {output}")


def verify(directory):
    import numpy as np
    directory = directory.resolve()
    manifest = read(directory / "Manifest.json")
    actual = {str(p.relative_to(directory)).replace("\\", "/"): digest(p) for p in directory.rglob("*") if p.is_file() and p.name != "Manifest.json"}
    require(actual == manifest, "Evidence file set/hash changed")
    identity = read(directory / "Identity.json")
    require(read(directory / "Transaction.json")["restored"], "Native source restoration not recorded")
    config = identity["cases"]
    variant_hashes = {"native": digest(directory / "NativeCoverage.slang"), "legacy": digest(directory / "LegacyCoverage.slang")}
    require(variant_hashes["native"] != variant_hashes["legacy"], "Shader variants are identical")
    for record in identity["runs"]:
        require(record["shaderHash"] == variant_hashes[record["arm"]], "Run shader hash does not match archived arm")
    records = [r for r in identity["runs"] if not r["diagnostic"]]
    require(len(records) == identity["blocks"] * 4, "Incomplete ABBA")
    specs = {c["id"]:c for c in config["cases"]}
    require(len(specs) == len(config["cases"]) and specs, "Duplicate/empty cases")
    medians, frames, image_groups, signatures = [], [], {}, {}
    for i, record in enumerate(records):
        require(record["index"] == i and record["arm"] == ["legacy","native","native","legacy"][i%4] and record["exitCode"] == 0, "Invalid order/status")
        root = directory / record["directory"]
        xml = ET.parse(root / "Tests.xml").getroot()
        require(len(list(xml.iter("testcase"))) == 1 and not list(xml.iter("failure")) and not list(xml.iter("skipped")), "Failed/skipped test")
        raw = read(root / "MaterialBaseline.json")
        require(not raw["validation"] and {c["id"] for c in raw["cases"]} == set(specs), "Invalid case set/diagnostic timing")
        process, timeline = {}, {}
        for case in raw["cases"]:
            name = case["id"]
            require(all(case.get(k) == v for k,v in specs[name].items()), "Fixture metadata mismatch")
            signature = {k:case.get(k) for k in ("resolvedGraph", "resolvedMaterials", "width", "height", "format")}
            require(signatures.setdefault(name, signature) == signature, "Resolved workload changed")
            require(not specs[name].get("requireEnvironment",True) or case["environmentTransitions"], "Environment never became ready")
            warmup, count = config["warmupFrames"], config["timingFrames"]
            if specs[name].get("requireEnvironment", True):
                transitions = case["environmentTransitions"]
                require(transitions and transitions[0]["frame"] < warmup and transitions[0]["status"] == 2 and transitions[0]["mapAvailable"], "Environment was not ready before timing")
                require(all(t["frame"] < warmup and t["status"] == 2 and t["mapAvailable"] for t in transitions), "Environment transitioned during/after timing")
            require([f["frame"] for f in case["warmup"]] == list(range(warmup)), "Warmup frames missing/reordered")
            all_frames = case["warmup"] + case["frames"]
            require(len({f["executionId"] for f in all_frames}) == len(all_frames), "Duplicate warmup/timing execution IDs")
            require(all(math.isfinite(f["graphMs"]) and f["graphMs"] > 0 and all(n["queue"] == 0 for n in f["nodes"]) for f in all_frames), "Invalid warmup/timing GPU envelope")
            timing = case["frames"]
            require([f["frame"] for f in timing] == list(range(warmup,warmup+count)), "Timing frames missing/reordered")
            require(len({f["executionId"] for f in timing}) == count, "Repeated execution IDs")
            require(all(math.isfinite(f["graphMs"]) and f["graphMs"] > 0 for f in timing), "Invalid graph GPU timestamps")
            required = specs[name]["requiredTiming"]
            for frame in timing:
                require(frame["asyncComputeBranches"] == specs[name]["expectedAsyncComputeBranches"], "Async graph contract changed")
                node = next((n for n in frame["nodes"] if n["name"] == required[0]), None)
                require(node and node["gpuMs"] and math.isfinite(node["gpuMs"]), "Required material scope absent")
                require(not required[1] or any(s["name"] == required[1] and s["gpuMs"] is not None for s in node["sections"]), "Required GPU section absent")
            process[name] = statistics.median(f["graphMs"] for f in timing)
            timeline[name] = [f["graphMs"] for f in case["warmup"] + timing]
            require(case["format"] in ("RGBA16F", "RGBA32F") and Path(case["image"]).name == case["image"], "Invalid HDR metadata")
            pixels = np.fromfile(root / case["image"], dtype="<f2" if case["format"] == "RGBA16F" else "<f4").astype(np.float64)
            require(pixels.size == case["width"] * case["height"] * 4 and np.isfinite(pixels).all() and case["validHDR"] and case["nonfiniteComponents"] == 0, "Invalid HDR pixels")
            image_groups.setdefault((name,record["arm"]), []).append(pixels)
        medians.append(process); frames.append(timeline)
    receipt_evidence = {}
    diagnostics = [r for r in identity["runs"] if r["diagnostic"]]
    require(len(diagnostics) == 2, "Separate bound shader receipt processes are required")
    devices = {}
    pipeline_options = {}
    if diagnostics:
        require(len(diagnostics) == 2 and {r["arm"] for r in diagnostics} == {"native", "legacy"}, "Incomplete diagnostic receipts")
        for record in diagnostics:
            require(record["exitCode"] == 0, "Diagnostic process failed")
            diagnostic_root = directory / record["directory"]
            xml = ET.parse(diagnostic_root / "Tests.xml").getroot()
            require(len(list(xml.iter("testcase"))) == 1 and not list(xml.iter("failure")) and not list(xml.iter("skipped")), "Failed/skipped diagnostic test")
            raw = read(diagnostic_root / "MaterialBaseline.json")
            require(not raw["validation"] and len(raw["cases"]) == len(specs) and {c["id"] for c in raw["cases"]} == set(specs), "Diagnostic configuration mismatch")
            for case in raw["cases"]:
                name = case["id"]
                require(all(case.get(k) == v for k,v in specs[name].items()), "Diagnostic fixture differs from ordinary capture")
                require({k:case.get(k) for k in signatures[name]} == signatures[name], "Diagnostic resolved workload differs")
                require([f["frame"] for f in case["warmup"] + case["frames"]] == list(range(config["warmupFrames"]+config["timingFrames"])), "Diagnostic frame sequence differs")
                require(Path(case["image"]).name == case["image"] and case["validHDR"] and case["nonfiniteComponents"] == 0, "Invalid diagnostic HDR metadata")
                pixels = np.fromfile(diagnostic_root / case["image"], dtype="<f2" if case["format"] == "RGBA16F" else "<f4").astype(np.float64)
                require(np.array_equal(pixels, image_groups[(name,record["arm"])][0]), "Diagnostic HDR differs from ordinary same-arm capture")
            for name, spec in specs.items():
                candidates = []
                for recipe_path in (directory / record["directory"] / "recipes" / name).rglob("*.recipe.json"):
                    recipe = read(recipe_path)
                    for stage, shader in recipe["shaders"].items():
                        entry = shader.get("source_entry_point", "")
                        expected = (entry in ("scenePathTraceMain", "openPbrRayQueryPathTraceMain")) if spec["path"] == "pathtrace" else entry in ("gpuDrivenStreamAssetFragmentMain", "streamTessellationFragment")
                        if not expected:
                            continue
                        expected_modules = {"scenePathTraceMain":"Features/PathTracing/ScenePathTrace", "openPbrRayQueryPathTraceMain":"Features/PathTracing/OpenPBRRayQueryPathTrace", "gpuDrivenStreamAssetFragmentMain":"Features/GPUDriven/GPUDrivenStreamAsset", "streamTessellationFragment":"Features/GPUDriven/GPUDrivenStreamAsset"}
                        require(shader["module_name"] == expected_modules[entry], "Unexpected coverage module/entry receipt")
                        expected_stage, expected_type = ("compute", "compute") if spec["path"] == "pathtrace" else ("fragment", "graphics")
                        require(stage == expected_stage and recipe["pipeline_type"] == expected_type, "Unexpected coverage pipeline stage/type")
                        device = recipe["source_device"]
                        require(not device["validation_enabled"] and not device["synchronization_validation_enabled"], "Recipe device enables validation")
                        require(devices.setdefault("gpu", device) == device, "Recipe GPU/driver/features differ")
                        options = {k:shader.get(k) for k in ("debug_mode", "descriptor_heap_mode", "slang_optimization", "spirv_postprocess", "spirv_postprocess_tool_version")}
                        options["state"] = recipe["state"]
                        require(not recipe["state"]["disable_optimization"] and pipeline_options.setdefault(name, options) == options, "Coverage pipeline options differ")
                        require(not shader.get("source_identity_ambiguous", True), "Ambiguous bound shader identity")
                        spirv = (recipe_path.parent / shader["spirv_file"]).resolve()
                        require(spirv.is_relative_to(directory) and spirv.is_file() and spirv.stat().st_size == shader["spirv_bytes"], "Invalid receipt SPIR-V path/size")
                        candidates.append({"entry": entry, "module": shader["module_name"], "stage": stage, "sha256": digest(spirv), "pipelineType": recipe["pipeline_type"], "inputSpirvHash": shader["input_spirv_hash"]})
                require(candidates, f"No bound production shading shader receipts for {name}")
                receipt_evidence.setdefault(name, {})[record["arm"]] = sorted(candidates, key=lambda c:(c["module"],c["entry"],c["sha256"]))
        for name, receipt in receipt_evidence.items():
            require(specs[name]["workload"] == "control" or {s["sha256"] for s in receipt["native"]} != {s["sha256"] for s in receipt["legacy"]}, "Custom A/B bound SPIR-V is identical")
    summary, correctness = {}, {}
    for name in specs:
        a,b = image_groups[(name,"legacy")],image_groups[(name,"native")]
        aa = {arm: all(np.array_equal(xs[0],x) for x in xs[1:]) for arm,xs in (("legacy",a),("native",b))}
        delta = a[0]-b[0]
        gate = {"sameArmExact": aa, "crossArmExact": bool(np.array_equal(a[0],b[0])), "maxAbsoluteError":float(np.abs(delta).max()), "RMSE":float(np.sqrt(np.mean(delta*delta)))}
        gate["passed"] = all(aa.values()) and gate["crossArmExact"]
        correctness[name] = gate
        pairs=[]
        for i in range(0,len(records),2):
            l = i if records[i]["arm"] == "legacy" else i+1
            n = i+1 if l == i else i
            av,bv=medians[l][name],medians[n][name]
            pairs.append({"legacyProcess":l,"nativeProcess":n,"legacyMs":av,"nativeMs":bv,"gainPercent":100*(1-bv/av)})
        gains=[p["gainPercent"] for p in pairs]
        critical={1:12.7062047364,3:3.1824463053,5:2.5705818356,7:2.3646242510,9:2.2621571629,11:2.2009851601,13:2.1603686565,15:2.1314495456}
        mean=statistics.mean(gains); half=critical[len(gains)-1]*statistics.stdev(gains)/math.sqrt(len(gains))
        summary[name]={"pairs":pairs,"gainPercentMean":mean,"gainPercent95CI":[mean-half,mean+half],"decision":"correctness-failed" if not gate["passed"] else "improved" if mean-half>0 else "regressed" if mean+half<0 else "inconclusive"}
    return {"scope":SCOPE,"identity":identity,"processMediansMs":medians,"framesMs":frames,"correctness":correctness,"summary":summary,"diagnosticReceipts":receipt_evidence}


def report(directory,destination):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data=verify(directory)
    require(not destination.exists() and not destination.resolve().is_relative_to(directory.resolve()), "Use a fresh report directory outside sealed evidence")
    destination.mkdir(parents=True)
    save(destination / "PlotData.json",data)
    shutil.copyfile(__file__,destination / Path(__file__).name)
    names=list(data["summary"]); records=[r for r in data["identity"]["runs"] if not r["diagnostic"]]
    colors={"legacy":"#b34b40","native":"#25749f"}
    for timeline,filename in [(False,"ProcessMedians.png"),(True,"FrameTimelines.png")]:
        fig,axes=plt.subplots(2,math.ceil(len(names)/2),figsize=(16,8),squeeze=False)
        for ax,name in zip(axes.flat,names):
            for arm,color in colors.items():
                indices=[i for i,r in enumerate(records) if r["arm"]==arm]
                if timeline:
                    for j,i in enumerate(indices):
                        ax.plot(data["framesMs"][i][name],color=color,alpha=.55,label=arm if j==0 else None)
                else:
                    ax.plot(indices,[data["processMediansMs"][i][name] for i in indices],"o-",color=color,label=arm)
            if timeline: ax.axvline(data["identity"]["cases"]["warmupFrames"]-.5,color="black",linestyle="--",label="Timing begins")
            ax.set_title(name); ax.set_ylabel("Whole graph GPU ms"); ax.set_xlabel("Frame" if timeline else "Independent process index"); ax.grid(alpha=.25); ax.legend()
        for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
        fig.suptitle("Whole graph GPU envelope; fixed scene/camera/quality; frames are not independent samples")
        fig.tight_layout();fig.savefig(destination/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5))
    for i,name in enumerate(names):
        s=data["summary"][name]; mean=s["gainPercentMean"]
        ax.scatter([i]*len(s["pairs"]),[p["gainPercent"] for p in s["pairs"]],color="gray",alpha=.7)
        ax.errorbar(i,mean,yerr=mean-s["gainPercent95CI"][0],fmt="s",color=colors["native"],capsize=5)
    ax.set_xticks(range(len(names)),names,rotation=15);ax.axhline(0,color="black");ax.set_ylabel("Graph GPU reduction (%) — positive is faster")
    ax.set_title(f"Independent paired process gains, n={len(records)//2}; 95% Student t CI")
    ax.legend(handles=[Line2D([],[],marker="o",linestyle="",color="gray",label="Individual pair"),Line2D([],[],marker="s",color=colors["native"],label="Mean and 95% CI")]);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(destination/"PairedGains.png",dpi=160);plt.close(fig)
    image_links = []
    for name in names:
        previews = {}
        for arm in ("legacy", "native"):
            record = next(r for r in records if r["arm"] == arm)
            root = directory / record["directory"]
            raw = read(root / "MaterialBaseline.json")
            case = next(c for c in raw["cases"] if c["id"] == name)
            pixels = np.fromfile(root / case["image"], dtype="<f2" if case["format"] == "RGBA16F" else "<f4").astype(np.float64)
            previews[arm] = pixels.reshape(case["height"], case["width"], 4)[:, :, :3]
        error = np.max(np.abs(previews["native"] - previews["legacy"]), axis=2)
        maximum = float(error.max())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, arm in zip(axes[:2], ("legacy", "native")):
            positive = np.maximum(previews[arm], 0)
            display = np.power(positive / (1 + positive), 1 / 2.2)
            ax.imshow(display)
            ax.set_title(f"{arm}: shared Reinhard + gamma 2.2")
            ax.axis("off")
        # Error remains in linear working-RGB units; zero error uses a labelled
        # fixed [0, 1] scale so an all-zero image is never autoscaled misleadingly.
        plot = axes[2].imshow(error, cmap="inferno", vmin=0, vmax=maximum if maximum > 0 else 1)
        axes[2].set_title(f"Max-channel absolute HDR error\nmax={maximum:.8g}; linear color scale")
        axes[2].axis("off")
        fig.colorbar(plot, ax=axes[2], fraction=.046, pad=.04, label="Absolute error (linear working RGB)")
        fig.suptitle(f"{name}: first independent process per arm; working-RGB preview only", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .91))
        filename = f"ImageComparison-{name}.png"
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)
        image_links.append(f"![{name} HDR comparison]({filename})")
    rows=[]
    for name,s in data["summary"].items():
        lo,hi=s["gainPercent95CI"];c=data["correctness"][name]
        rows.append(f"| {name} | {statistics.mean(p['legacyMs'] for p in s['pairs']):.6f} | {statistics.mean(p['nativeMs'] for p in s['pairs']):.6f} | {s['gainPercentMean']:.2f}% [{lo:.2f}, {hi:.2f}] | {c['passed']} | {s['decision']} |")
    text=f"""# Coverage scene A/B

{SCOPE}.

GPU: {data['identity']['gpu']}. {len(records)} independent processes in ABBA blocks. Per-process median of {data['identity']['cases']['timingFrames']} frames after {data['identity']['cases']['warmupFrames']} excluded warmup frames. No outlier removal; 95% Student t intervals use independent paired process gains, not frames. Ordinary timing disables validation and recipe capture. Two separate recipe processes are diagnostic only and excluded. Distinct legacy/native executables differ in archived CPU Coverage transport; paired build identity/exact sources are archived. Shared disk shader/driver caches retained; clocks unaltered. Cooperative lock does not prove exclusive system GPU use. This fixed lookdev workload does not establish gains for other scenes.

| Case | Legacy mean of process medians (ms) | Native (ms) | Paired reduction [95% CI] | Exact HDR/AA | Result |
|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

The offline verifier additionally links diagnostic configuration, resolved graph/materials and exact HDR to ordinary captures; validates archived per-arm shader hashes, warmup sequence, graphics queue timing, matched pipeline/device options; and checks distinct actual Coverage raster-fragment/pathtrace SPIR-V. Deferred lighting itself may remain identical because raster Coverage executes in the fragment stage. This strengthens the capture-time verifier without modifying sealed evidence.

Image gate requires finite linear HDR, exact same-arm and cross-arm pixels. Any discrepancy blocks a performance acceptance; maximum absolute error and RMSE are retained in PlotData.json. No quality threshold was relaxed after observing results.

![Process medians](ProcessMedians.png)
![Paired gains](PairedGains.png)
![All frame trajectories](FrameTimelines.png)

Image previews use the same `max(rgb,0)/(1+max(rgb,0))` Reinhard mapping and gamma 1/2.2 on both arms. These are working-RGB previews, not color-managed reference displays. They do not change the raw linear HDR correctness gate. Error panels show maximum absolute RGB component error per pixel with a linear color scale and an explicit maximum; alpha is still checked by the raw HDR gate.

{chr(10).join(image_links)}

Evidence: `{directory.resolve()}`. Manifest SHA256: `{digest(directory/'Manifest.json')}`.
Field mapping: MaterialBaseline.json cases[].frames[].graphMs -> process median; gain=100*(1-native/legacy).
Python {platform.python_version()}, NumPy {np.__version__}, matplotlib {matplotlib.__version__}.
Reproduce: `python -B Tools/Perf/MeasureCoverageScene.py report --input "{directory.resolve()}" --output <fresh-outside-directory>`.
"""
    (destination/"Report.md").write_text(text,encoding="utf-8")
    print(destination/"Report.md")


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run")
    for arg in ("legacy-exe","native-exe","legacy-shader","cases","output","build-evidence"):r.add_argument("--"+arg,type=Path,required=True)
    r.add_argument("--blocks",type=int,choices=range(1,9),default=2);r.add_argument("--timeout",type=int,default=7200);r.add_argument("--receipts",action="store_true",required=True)
    for command in ("verify","report","recover"):
        c=sub.add_parser(command);c.add_argument("--input",type=Path,required=True)
        if command=="report":c.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.command=="run":run(a)
    elif a.command=="recover":restore(a.input)
    elif a.command=="report":report(a.input,a.output)
    else:
        data=verify(a.input);print(json.dumps({"summary":data["summary"],"correctness":data["correctness"]},indent=2));require(all(c["passed"] for c in data["correctness"].values()),"HDR correctness failed")


if __name__=="__main__":main()
