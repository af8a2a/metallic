"""Production CLAS backend A/B. Reuses capture and ABBA statistics, not M3 source swapping."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import ExperimentRunner as experiment
import MaterialBaseline as baseline

ROOT = baseline.ROOT
PROTOCOL = "metallic-clas-tessellation-ab-v1"
MODES = {"A": "triangles", "B": "templates"}
METRICS = ("clasMs", "tessellationMs", "graphMs", "shadowsMs")
POLICY = {"statistic": "abba-log-ratio", "confidence": .95, "minimumGain": .03,
          "maximumGraphRegression": .02, "primaryMetric": "clasMs", "imagePolicy": "exact-bytes"}
read, save, digest = baseline.read, baseline.save, baseline.digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_graph(graph):
    value = copy.deepcopy(graph)
    node = next(n for n in value["nodes"] if n["name"] == "VBuffer")
    node["properties"].pop("rtTessellationBuildMode", None)
    return value


def prepare(args):
    output = args.output.resolve()
    require(not output.exists(), "Use a fresh fixture directory")
    require(args.warmup >= 4 and args.frames >= 1 and 1 <= args.width <= 16384 and 1 <= args.height <= 16384,
            "Invalid dimensions or fixed warmup/timing window")
    source = args.sample.resolve()
    graph = read(source)
    # The measured native-resolution branch still renders visibility, actual
    # CLAS shadows and deferred HDR. Presentation/upscalers are outside this case.
    keep = {"VBuffer", "Shadows", "Deferred"}
    graph["nodes"] = [n for n in graph["nodes"] if n["name"] in keep]
    require({n["name"] for n in graph["nodes"]} == keep, "Expected production sample branch is missing")
    graph["edges"] = [e for e in graph["edges"] if e["src"].split(".")[0] in keep and e["dst"].split(".")[0] in keep]
    graph["outputs"] = ["Deferred.color"]
    camera = copy.deepcopy(graph["view"]["camera"])
    graph["view"]["temporalJitter"] = False
    graph["view"]["renderResolution"] = {"width": args.width, "height": args.height, "adaptive": False}
    for node in graph["nodes"]:
        p = node["properties"]
        if node["name"] == "VBuffer":
            p.update(camera=camera, temporalJitter=False, AsyncComputePreferred=False,
                     asyncSoftwareRaster=False, rtTessellation=True)
        if node["name"] == "Shadows":
            p.update(sigmaDenoise=False, diagnosticSampleFrame=0)
    scene = ROOT / "Asset/Tessellation/DisplacedPlane.gltf"
    asset_files = {source, scene, ROOT / "Asset/ABeautifulGame/environment.hdr"}
    for item in read(scene).get("buffers", []) + read(scene).get("images", []):
        uri = item.get("uri", "")
        if uri and not uri.startswith("data:"):
            asset_files.add((scene.parent / uri).resolve())
    for node in graph["nodes"]:
        stream = node["properties"].get("streamAssetPath")
        if stream:
            asset_files.add((ROOT / stream).resolve())
    require(all(p.is_file() for p in asset_files), "Missing authored sample asset or cooked stream; do not substitute")
    output.mkdir(parents=True)
    common = {"version": 2, "frames": args.warmup + args.frames + 1,
              "warmupFrames": args.warmup, "timingFrames": args.frames,
              "assetFiles": [str(p) for p in sorted(asset_files)],
              "cases": [{"id": "CLASTessellation", "graph": "Graph.json", "scene": None,
                         "output": "Deferred.color", "width": args.width, "height": args.height,
                         "requiredTiming": ["VBuffer", "Tessellation CLAS build"],
                         "expectedAsyncComputeBranches": 0,
                         "expectedNodeQueues": {"VBuffer": 0, "Shadows": 0, "Deferred": 0},
                         "freezeResidencyFrame": args.warmup // 2, "freezeStreamingNode": "VBuffer",
                         "captureOutputs": ["Shadows.shadow", "VBuffer.depth"],
                         "tessellationReceipt": True,
                         "manualExposure": True, "exposureEV100": 0,
                         "lighting": {"sun": {"direction": [.3, -.8, -.5], "illuminance": 12000,
                                                  "angularRadius": 0.0}},
                         "environment": {"path": "Asset/ABeautifulGame/environment.hdr",
                                         "intensity": 1.0, "rotationDegrees": 0.0}}]}
    for mode in MODES.values():
        directory = output / mode
        directory.mkdir()
        arm_graph = copy.deepcopy(graph)
        next(n for n in arm_graph["nodes"] if n["name"] == "VBuffer")["properties"]["rtTessellationBuildMode"] = mode
        save(directory / "Graph.json", arm_graph)
        save(directory / "Cases.json", common)
    save(output / "Fixture.json", {"protocol": PROTOCOL, "sourceSample": str(source),
                                    "sourceSampleSha256": digest(source), "scope": "native VBuffer + Shadows + Deferred HDR"})


def fixture_configs(directory):
    require(read(directory / "Fixture.json")["protocol"] == PROTOCOL, "Unknown fixture protocol")
    configs = {a: read(directory / m / "Cases.json") for a, m in MODES.items()}
    require(configs["A"] == configs["B"], "Arms must use the same case, camera, frames and assets")
    config = configs["A"]
    require(len(config["cases"]) == 1, "One fixed workload per independent process is required")
    require(config["warmupFrames"] >= 4 and config["timingFrames"] >= 1 and
            config["frames"] > config["warmupFrames"] + config["timingFrames"], "Invalid fixed frame windows")
    graphs = {a: read(directory / m / config["cases"][0]["graph"]) for a, m in MODES.items()}
    require(normalized_graph(graphs["A"]) == normalized_graph(graphs["B"]), "Graph differs beyond the CLAS build mode")
    for a, graph in graphs.items():
        require(graph["view"]["renderResolution"] == {"width": config["cases"][0]["width"],
                "height": config["cases"][0]["height"], "adaptive": False}, "Graph must render at the declared native resolution")
        properties = next(n for n in graph["nodes"] if n["name"] == "VBuffer")["properties"]
        require(properties.get("rtTessellationBuildMode") == MODES[a] and properties.get("rtTessellation"), "Wrong backend or disabled tessellation")
    return config, graphs


def identity(exe, fixtures):
    result = baseline.identity(exe, fixtures / "triangles/Cases.json")
    extras = [*fixtures.rglob("*.json"), *ROOT.joinpath("Shaders").rglob("*.h"),
              Path(__file__), Path(experiment.__file__), ROOT / "tests/rhi/MaterialBaselineTests.cpp",
              exe.parent.parent / "CMakeCache.txt"]
    result["additional"] = {str(p.resolve()): digest(p) for p in sorted(set(extras))}
    result["fixtureRoot"] = str(fixtures.resolve())
    result["fixtureFiles"] = {p.relative_to(fixtures).as_posix(): digest(p) for p in fixtures.rglob("*.json")}
    # Working tree status is informative; content hashes own the mutation gate.
    result.pop("status", None)
    return result


def finite_time(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Missing/invalid GPU interval: " + label)
    return float(value)


def stream_content_identity(data, runtime_cook):
    # MeshletStreamAsset.cpp owns this v9/216-byte format. Runtime .source files
    # are session markers; offsets 32/40 are source mtime and metadata fingerprint.
    require(len(data) >= 216, "Truncated runtime stream header")
    magic, version, endian, size, source_size, mtime, fingerprint, dependencies, path_bytes = struct.unpack_from("<8sIIQQqQII", data)
    require(magic == b"MTLMSTRM" and version == 9 and endian == 0x01020304 and size == len(data), "Unsupported stream content identity schema")
    normalized = bytearray(data)
    if runtime_cook:
        require(dependencies == 0 and path_bytes == 0, "Runtime normalization requires a marker with no external dependency paths")
        normalized[32:48] = bytes(16)
    return {"schema": "MTLMSTRM-v9-runtime-marker-normalized" if runtime_cook else "MTLMSTRM-v9-exact",
            "bytes": len(data), "sha256": hashlib.sha256(normalized).hexdigest()}, {
            "rawSha256": hashlib.sha256(data).hexdigest(), "sourceFileSize": source_size,
            "sourceWriteTime": mtime, "sourceDependencyFingerprint": fingerprint,
            "normalizedFields": ["sourceWriteTime@32:8", "sourceDependencyFingerprint@40:8"] if runtime_cook else []}


def capture_stream_asset(directory):
    case = read(directory / "MaterialBaseline.json")["cases"][0]
    paths = {p["assetPath"] for f in case["frames"] for p in f["streaming"] if p["passName"] == "VBuffer"}
    require(len(paths) == 1, "Runtime stream source changed before capture")
    source = Path(next(iter(paths))).resolve()
    shutil.copyfile(source, directory / "StreamAsset.bin")
    if source.parent == (ROOT / ".cache/scene-streams").resolve():
        shutil.copyfile(source.with_suffix(".source"), directory / "StreamSource.txt")


def analyze_run(directory, config, expected_graph, arm):
    import numpy as np
    xml = ET.parse(directory / "Tests.xml").getroot()
    require(len(list(xml.iter("testcase"))) == 1 and not list(xml.iter("failure")) and not list(xml.iter("skipped")), "Failed or skipped production test")
    report = read(directory / "MaterialBaseline.json")
    require(report.get("validation") is False and report.get("synchronizationValidation") is False and
            len(report["cases"]) == 1, "Diagnostic validation or missing case")
    case, spec = report["cases"][0], config["cases"][0]
    require(all(case.get(k) == v for k, v in spec.items()), "Captured fixture differs from the registered case")
    require(case["resolvedGraph"] == expected_graph, "Resolved production graph differs from registered graph")
    require(case["streamingFreeze"] == {"node": spec["freezeStreamingNode"], "appliedFrame": spec["freezeResidencyFrame"],
                                        "property": "benchmarkFreezeResidency"}, "Residency freeze was not applied")
    require(case["environmentTransitions"], "Missing resolved HDR environment evidence")
    warmup, count = config["warmupFrames"], config["timingFrames"]
    require([f["frame"] for f in case["frames"]] == list(range(warmup, warmup + count)), "Incomplete or reordered timing window")
    require([f["frame"] for f in case["warmup"]] == list(range(warmup)), "Incomplete warmup evidence")
    require(len({f["executionId"] for f in case["frames"]}) == count, "Duplicate execution ID")
    series = {m: [] for m in METRICS}
    source_identity, backend_identity, input_identity, asset_path = None, None, None, None
    for frame in case["frames"]:
        require(frame["asyncComputeBranches"] == spec["expectedAsyncComputeBranches"], "Queue overlap contract changed")
        baseline.validate_node_queues(frame["nodes"], spec["expectedNodeQueues"])
        node = next(n for n in frame["nodes"] if n["name"] == "VBuffer")
        def section(name):
            values = [s["gpuMs"] for s in node["sections"] if s["name"] == name]
            require(len(values) == 1, "Required GPU section missing or duplicated: " + name)
            return finite_time(values[0], name)
        series["clasMs"].append(section("Tessellation CLAS build"))
        series["tessellationMs"].append(section("CLAS tessellation") + section("Tessellation publish"))
        series["graphMs"].append(finite_time(frame["graphMs"], "whole graph"))
        series["shadowsMs"].append(finite_time(next(n for n in frame["nodes"] if n["name"] == "Shadows")["gpuMs"], "Shadows"))
        profiles = [p for p in frame["streaming"] if p["passName"] == "VBuffer"]
        require(len(profiles) == 1, "Missing unique production tessellation identity")
        profile = profiles[0]
        current = profile["tessellationIdentity"]
        if isinstance(current, str):
            current = json.loads(current)
        properties = next(n for n in expected_graph["nodes"] if n["name"] == "VBuffer")["properties"]
        require(current["mode"] == MODES[arm] and current["enabled"] and
                current["maxPatches"] == properties["rtTessellationMaxPatches"], "Backend receipt does not match requested work")
        require(current["module"] == "Features/GPUDriven/StreamCLASTessellation" and current["entry"] == "tessellationMain" and
                len(current["spirvSha256"]) == 64, "Missing production shader identity")
        require(current["clasBytes"] > 0 and current["scratchBytes"] > 0 and current["buildInfoStride"] > 0,
                "Missing actual backend allocation identity")
        require((current["templatesReady"] and current["patternCount"] > 0 and current["templateBytes"] > 0 and
                 current["templateAuxiliaryBytes"] > 0) if arm == "B" else
                (not current["templatesReady"] and current["patternCount"] == 0 and current["templateBytes"] == 0 and
                 current["templateAuxiliaryBytes"] == 0), "Actual template pool does not match the requested backend")
        if backend_identity is None:
            backend_identity = current
            source_identity = profile["diagnosticSourceIdentity"]
            asset_path = profile["assetPath"]
            input_identity = {k: profile[k] for k in ("originalTopologyHash", "primitiveMetadataHash",
                                                     "residentPages", "textureResidentBytes")}
        require(current == backend_identity and profile["diagnosticSourceIdentity"] == source_identity,
                "Production backend or frozen source identity changed during timing")
        require(source_identity and profile["assetPath"] == asset_path and
                all(profile[k] == v for k, v in input_identity.items()), "Frozen input publication changed")
    runtime_cook = Path(asset_path).resolve().parent == (ROOT / ".cache/scene-streams").resolve()
    content, provenance = stream_content_identity((directory / "StreamAsset.bin").read_bytes(), runtime_cook)
    if runtime_cook:
        marker = (directory / "StreamSource.txt").read_bytes()
        lines = marker.decode("utf-8").splitlines()
        require(len(marker) == provenance["sourceFileSize"] and len(lines) == 2 and
                lines[1] == Path(asset_path).stem and Path(lines[0]).resolve() in
                {Path(p).resolve() for p in config["assetFiles"]}, "Runtime source marker does not match declared authored input")
        provenance["sourceMarkerSha256"] = hashlib.sha256(marker).hexdigest()
    input_identity["streamContent"] = content
    provenance["assetPath"] = asset_path
    counts = case["tessellationCounts"]
    require(len(counts) == 4 and counts[0] > 0 and counts[1] > 0 and counts[2:] == [0, 0], "No successful production tessellation output")
    input_identity["tessellationCounts"] = counts
    require(case["format"] in ("RGBA16F", "RGBA32F"), "HDR readback format required")
    image = directory / case["image"]
    require(image.parent == directory and image.is_file(), "Invalid HDR readback path")
    pixels = np.fromfile(image, dtype="<f2" if case["format"] == "RGBA16F" else "<f4")
    require(pixels.size == spec["width"] * spec["height"] * 4 and np.isfinite(pixels).all() and
            case["validHDR"] and case["nonfiniteComponents"] == 0, "Invalid HDR image")
    outputs = {"HDR": {"sha256": digest(image), "bytes": image.stat().st_size, "format": case["format"]}}
    captures = case["captures"]
    require([c["output"] for c in captures] == spec["captureOutputs"], "Missing depth/shadow correctness outputs")
    for capture in captures:
        path = directory / capture["file"]
        require(path.parent == directory and path.is_file() and path.stat().st_size == capture["bytes"] > 0, "Invalid correctness capture")
        expected_bytes = spec["width"] * spec["height"] * (4 if capture["output"] == "VBuffer.depth" else 1)
        require(capture["bytes"] == expected_bytes, "Truncated native depth/shadow readback")
        if capture["output"] == "VBuffer.depth":
            require(np.isfinite(np.fromfile(path, dtype="<f4")).all(), "Nonfinite depth readback")
        outputs[capture["output"]] = {"sha256": digest(path), "bytes": capture["bytes"], "format": capture["format"]}
    return {"timings": {m: {"median": statistics.median(v), "frames": v} for m, v in series.items()},
            "identity": backend_identity, "sourceIdentity": source_identity, "inputs": input_identity, "outputs": outputs,
            "streamProvenance": provenance,
            "warmupGraphMs": [f["graphMs"] for f in case["warmup"]]}


def decision(rows, stage, reference=()):
    require(stage in ("discovery", "confirmation"), "Unknown stage")
    require(len(rows) >= 8 and len(rows) % 4 == 0 and [r["arm"] for r in rows] == list("ABBA") * (len(rows)//4), "Incomplete ABBA blocks")
    require(experiment.independent_processes(rows), "Overlapping or invalid independent processes")
    identities = {a: next(r["result"]["identity"] for r in (reference or rows) if r["arm"] == a) for a in MODES}
    expected_inputs = (reference or rows)[0]["result"].get("inputs", {})
    require(all(r["result"].get("inputs", {}) == expected_inputs for r in rows), "Cross-arm or cross-stage input drift")
    expected_outputs = (reference or rows)[0]["result"]["outputs"]
    require(all(r["result"]["identity"] == identities[r["arm"]] for r in rows), "Within-arm or cross-stage shader/backend identity drift")
    if any(r["result"]["outputs"] != expected_outputs for r in rows):
        return {"decision": "reject", "stage": stage, "evidenceValidity": "valid", "effectAssessment": "correctness_failure", "nextAction": "reject_candidate"}
    gains = {m: experiment.block_gain(rows, m) for m in METRICS}
    spread = {a: {m: (max(v)-min(v))/statistics.median(v) for m in METRICS
                  for v in [[r["result"]["timings"][m]["median"] for r in rows if r["arm"] == a]]} for a in MODES}
    primary, graph = gains["clasMs"], gains["graphMs"]
    status, effect, action = "inconclusive", "uncertain", "retain_candidate"
    if primary["upper95"] < 0 or graph["upper95"] < -POLICY["maximumGraphRegression"]:
        status, effect, action = "reject", "confirmed_regression", "reject_candidate"
    elif primary["upper95"] < POLICY["minimumGain"]:
        status, effect, action = "reject", "below_target", "reject_candidate"
    elif stage == "discovery" and primary["estimate"] >= POLICY["minimumGain"]:
        effect, action = "promising", "run_confirmation"
    elif stage == "confirmation" and primary["lower95"] >= POLICY["minimumGain"] and graph["lower95"] >= -POLICY["maximumGraphRegression"]:
        status, effect, action = "accept", "confirmed_gain", "accept_candidate"
    return {"decision": status, "stage": stage, "evidenceValidity": "valid", "effectAssessment": effect,
            "nextAction": action, "gains": gains, "withinArmRelativeRange": spread,
            "noiseWarning": any(v > .1 for arm in spread.values() for v in arm.values()),
            "statisticalUnit": "complete ABBA block of four independent process medians; log-ratio Student-t 95% CI"}


def receipts(directory):
    result = []
    for path in sorted((directory / "recipes").rglob("*.recipe.json")):
        recipe = read(path)
        require(not recipe["source_device"]["validation_enabled"] and not recipe["source_device"]["synchronization_validation_enabled"], "Instrumented diagnostic recipe device")
        for stage, shader in recipe["shaders"].items():
            module = shader.get("module_name", "")
            if module != "Features/GPUDriven/StreamCLASTessellation":
                continue
            spirv = (path.parent / shader["spirv_file"]).resolve()
            require(spirv.is_relative_to(directory.resolve()) and spirv.is_file() and
                    spirv.stat().st_size == shader["spirv_bytes"] and not shader.get("source_identity_ambiguous", True), "Invalid bound production SPIR-V receipt")
            result.append({"module": module, "entry": shader["source_entry_point"], "stage": stage,
                           "sha256": digest(spirv), "device": recipe["source_device"]})
    require(result, "No bound production CLAS tessellation shader receipt")
    return result


def validate_run_order(runs, manifest):
    require(set(manifest["budgets"]) == {"discovery", "confirmation"}, "Unknown stage budget")
    expected = [("diagnostic", arm) for arm in "AB"]
    if not manifest["diagnosticsOnly"]:
        expected += [("discovery", arm) for arm in "ABBA" * manifest["budgets"]["discovery"]]
        if "confirmation" in manifest:
            expected += [("confirmation", arm) for arm in "ABBA" * manifest["budgets"]["confirmation"]]
    require([(r["stage"], r["arm"]) for r in runs] == expected, "Unexpected experiment stage or process order")
    require(len({r["directory"] for r in runs}) == len(runs), "Repeated process evidence directory")


def collect(exe, fixtures, output, arm, stage, timeout):
    directory = output / f"{stage}-{len(list(output.glob('*/Process.json'))):02d}-{arm}"
    directory.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
    require(not env.get("VK_INSTANCE_LAYERS"), "Injected Vulkan layers invalidate normal timing")
    env.update(METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
               METALLIC_MATERIAL_BASELINE_CASES=str(fixtures / MODES[arm] / "Cases.json"))
    if stage == "diagnostic":
        env["METALLIC_PIPELINE_RECIPE_DIRECTORY"] = str(directory / "recipes")
    command = [str(exe), "--gtest_filter=*material_phase0_baseline*", "--rhi-no-validation",
               "--output-dir", str(directory), "--gtest_output=xml:" + str(directory / "Tests.xml")]
    row = {"arm": arm, "stage": stage, "directory": directory.name,
           "process": {"command": command, "startedUnix": time.time()},
           "environment": {k: v for k, v in env.items() if k.startswith(("METALLIC_", "VK_"))}}
    with (directory / "Process.log").open("wb") as log, (directory / "Telemetry.csv").open("wb") as telemetry:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw", "--format=csv", "-l", "1"], stdout=telemetry, stderr=subprocess.DEVNULL)
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row["process"]["pid"] = process.pid
            try:
                row["process"]["exitCode"] = process.wait(timeout=timeout)
            except BaseException:
                process.kill()
                process.wait(timeout=30)
                row["process"]["exitCode"] = process.returncode
                raise
            finally:
                row["process"]["endedUnix"] = time.time()
                save(directory / "Process.json", row)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    require(row["process"]["exitCode"] == 0, "Production process failed: " + str(directory))
    capture_stream_asset(directory)
    return row


def run(args):
    exe, fixtures, output = args.exe.resolve(), args.fixtures.resolve(), args.output.resolve()
    config, graphs = fixture_configs(fixtures)
    require(exe.is_file() and not output.exists(), "Use an existing executable and fresh output directory")
    require("CMAKE_BUILD_TYPE:STRING=Release" in (exe.parent.parent / "CMakeCache.txt").read_text(), "Reuse an existing Release build")
    locks = [ROOT / "build/shader-experiment.lock", ROOT / "build/material-phase0.lock"]
    require(not any(p.exists() for p in locks), "Another cooperative GPU experiment owns the lock")
    locks[0].parent.mkdir(exist_ok=True)
    with locks[0].open("x") as lock:
        json.dump({"pid": os.getpid(), "output": str(output), "protocol": PROTOCOL}, lock)
    output.mkdir(parents=True)
    manifest = {"protocol": PROTOCOL, "policy": POLICY, "status": "running", "runs": [],
                "budgets": {"discovery": args.blocks, "confirmation": args.confirmation_blocks},
                "diagnosticsOnly": args.diagnostics_only, "timeoutSeconds": args.timeout,
                "scope": read(fixtures / "Fixture.json")["scope"],
                "cachePolicy": "Shared disk caches retained; fixed warmup; initialization/template pool construction excluded",
                "gpuExclusivity": "Cooperative lock and whole-device telemetry; not proof of system GPU exclusivity"}
    try:
        shutil.copytree(fixtures, output / "fixtures")
        initial = identity(exe, fixtures)
        save(output / "Identity.json", initial)
        save(output / "Experiment.json", manifest)
        (output / "GPU.txt").write_text(experiment.gpu_identity(), encoding="utf-8")
        for path in [Path(__file__), Path(experiment.__file__), Path(baseline.__file__)]:
            shutil.copy2(path, output / path.name)
        diagnostics = []
        identity_hash = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        for arm in MODES:
            require(identity(exe, fixtures) == initial, "Source, asset or executable changed")
            row = collect(exe, fixtures, output, arm, "diagnostic", args.timeout)
            row["result"] = analyze_run(output / row["directory"], config, graphs[arm], arm)
            row["recipes"] = receipts(output / row["directory"])
            row["inputIdentitySha256"] = identity_hash
            require(any(r["module"] == row["result"]["identity"]["module"] and
                        r["entry"] == row["result"]["identity"]["entry"] and
                        r["sha256"] == row["result"]["identity"]["spirvSha256"] for r in row["recipes"]),
                    "Bound diagnostic SPIR-V does not match the production backend receipt")
            diagnostics.append(row)
            manifest["runs"].append(row)
            save(output / "Experiment.json", manifest)
        require(diagnostics[0]["result"]["outputs"] == diagnostics[1]["result"]["outputs"],
                "Diagnostic exact HDR/depth/shadow mismatch; stop before timing (do not widen tolerance after observing timings)")
        require(diagnostics[0]["result"]["inputs"] == diagnostics[1]["result"]["inputs"], "Diagnostic published input/count mismatch")
        require(diagnostics[0]["recipes"][0]["device"] == diagnostics[1]["recipes"][0]["device"], "Different diagnostic GPUs/configurations")
        discovery = []
        if not args.diagnostics_only:
            for stage, blocks in manifest["budgets"].items():
                rows = []
                for arm in "ABBA" * blocks:
                    require(identity(exe, fixtures) == initial, "Source, asset or executable changed")
                    row = collect(exe, fixtures, output, arm, stage, args.timeout)
                    row["inputIdentitySha256"] = identity_hash
                    row["result"] = analyze_run(output / row["directory"], config, graphs[arm], arm)
                    require(row["result"]["outputs"] == diagnostics[0]["result"]["outputs"], "Ordinary outputs differ from diagnostic correctness snapshot")
                    require(row["result"]["identity"] == next(r["result"]["identity"] for r in diagnostics if r["arm"] == arm), "Ordinary shader/backend differs from diagnostic receipt")
                    rows.append(row)
                    manifest["runs"].append(row)
                    save(output / "Experiment.json", manifest)
                manifest[stage] = decision(rows, stage, discovery)
                save(output / "Decision.json", manifest[stage])
                if stage == "discovery":
                    discovery = rows
                    if manifest[stage]["nextAction"] != "run_confirmation":
                        break
        require(identity(exe, fixtures) == initial, "Inputs changed during experiment")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save(output / "Experiment.json", manifest)
        baseline.seal(output)
        locks[0].unlink()
    print(json.dumps(verify(output), indent=2))


def verify(directory):
    directory = directory.resolve()
    require(read(directory / "Manifest.json") == {str(p.relative_to(directory)).replace("\\", "/"): digest(p)
            for p in directory.rglob("*") if p.is_file() and p.name != "Manifest.json"}, "Evidence file set/hash changed")
    manifest = read(directory / "Experiment.json")
    require(manifest["protocol"] == PROTOCOL and manifest["policy"] == POLICY and manifest["status"] == "complete", "Unsupported/incomplete experiment")
    require(all(type(v) is int and 2 <= v <= 10 for v in manifest["budgets"].values()), "Invalid declared stage budget")
    config, graphs = fixture_configs(directory / "fixtures")
    validate_run_order(manifest["runs"], manifest)
    calculated = []
    captured_identity = read(directory / "Identity.json")
    require(captured_identity["fixtureFiles"] == {p.relative_to(directory / "fixtures").as_posix(): digest(p)
            for p in (directory / "fixtures").rglob("*.json")}, "Archived workload differs from the captured input identity")
    identity_hash = hashlib.sha256(json.dumps(captured_identity, sort_keys=True).encode()).hexdigest()
    for row in manifest["runs"]:
        root = directory / row["directory"]
        process_receipt = read(root / "Process.json")
        require(root.parent == directory and all(process_receipt[k] == row[k] for k in
                ("process", "arm", "stage", "directory", "environment")) and row["process"]["exitCode"] == 0, "Invalid process receipt")
        require(row["environment"].get("METALLIC_NSIGHT_GRAPHICS_CAPTURE") == "0" and
                row["environment"].get("METALLIC_SHADER_CAPTURE_SYMBOLS") == "0" and
                not row["environment"].get("VK_INSTANCE_LAYERS"), "Instrumented process environment")
        require(("METALLIC_PIPELINE_RECIPE_DIRECTORY" in row["environment"]) == (row["stage"] == "diagnostic"),
                "Diagnostic collection leaked into ordinary timing")
        require(row["stage"] == "diagnostic" or not (root / "recipes").exists(), "Ordinary run contains diagnostic recipes")
        require(row["inputIdentitySha256"] == identity_hash, "Source/asset/executable identity differs")
        require(row["environment"].get("METALLIC_MATERIAL_BASELINE_CASES") ==
                str(Path(captured_identity["fixtureRoot"]) / MODES[row["arm"]] / "Cases.json") and
                row["process"]["command"] == [captured_identity["exe"], "--gtest_filter=*material_phase0_baseline*",
                    "--rhi-no-validation", "--output-dir", str(root), "--gtest_output=xml:" + str(root / "Tests.xml")],
                "Process command does not identify the declared executable and workload")
        result = analyze_run(root, config, graphs[row["arm"]], row["arm"])
        require(result == row["result"], "Saved run analysis differs from raw evidence")
        if row["stage"] == "diagnostic":
            require(receipts(root) == row["recipes"], "Bound shader receipt changed")
            require(any(r["module"] == result["identity"]["module"] and r["entry"] == result["identity"]["entry"] and
                        r["sha256"] == result["identity"]["spirvSha256"] for r in row["recipes"]), "Production SPIR-V receipt mismatch")
        calculated.append({**row, "result": result})
    require(experiment.independent_processes(calculated), "Processes overlap or lifetime evidence invalid")
    diagnostics = [r for r in calculated if r["stage"] == "diagnostic"]
    require(len(diagnostics) == 2 and [r["arm"] for r in diagnostics] == list("AB"), "Two diagnostic arms required")
    require(all(r["result"]["outputs"] == diagnostics[0]["result"]["outputs"] for r in calculated), "Correctness snapshot differs")
    require(all(r["result"]["inputs"] == diagnostics[0]["result"]["inputs"] for r in calculated), "Published input identity differs")
    require(diagnostics[0]["recipes"][0]["device"] == diagnostics[1]["recipes"][0]["device"], "Diagnostic GPU changed")
    for row in calculated:
        require(row["result"]["identity"] == next(r["result"]["identity"] for r in diagnostics if r["arm"] == row["arm"]), "Backend/shader identity drift")
    decisions, discovery = {}, []
    for stage in ("discovery", "confirmation"):
        rows = [r for r in calculated if r["stage"] == stage]
        if not rows:
            continue
        require(len(rows) == manifest["budgets"][stage] * 4, "Stage budget not completed")
        require(stage != "confirmation" or decisions["discovery"]["nextAction"] == "run_confirmation", "Unauthorized confirmation")
        decisions[stage] = decision(rows, stage, discovery)
        require(decisions[stage] == manifest[stage], "Saved decision differs from recomputed v2 block statistics")
        if stage == "discovery":
            discovery = rows
    require(manifest["diagnosticsOnly"] or "discovery" in decisions, "Missing discovery")
    require("discovery" not in decisions or decisions["discovery"]["nextAction"] != "run_confirmation" or "confirmation" in decisions, "Promising discovery lacks declared confirmation")
    return {"verified": True, "protocol": PROTOCOL, "diagnosticsOnly": manifest["diagnosticsOnly"], "decisions": decisions}


def report(directory, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    require(not output.exists() and not output.resolve().is_relative_to(directory.resolve()), "Report must be new and outside sealed evidence")
    verified = verify(directory)
    manifest = read(directory / "Experiment.json")
    runs = [r for r in manifest["runs"] if r["stage"] != "diagnostic"]
    require(runs, "No ordinary timings; chart report is not applicable to diagnostic-only evidence")
    output.mkdir(parents=True)
    rows = [{"stage": r["stage"], "arm": r["arm"], "process": i,
             **{m: r["result"]["timings"][m]["median"] for m in METRICS}} for i, r in enumerate(runs)]
    save(output / "PlotData.json", rows)
    stage_colors = {"discovery": "#355c9a", "confirmation": "#8053a4"}
    boundaries = [i - .5 for i in range(1, len(rows)) if rows[i]["stage"] != rows[i - 1]["stage"]]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, axis in zip(METRICS, axes.flat):
        for arm, color in (("A", "#355c9a"), ("B", "#c26432")):
            selected = [r for r in rows if r["arm"] == arm]
            axis.scatter([r["process"] for r in selected], [r[metric] for r in selected], color=color, label=MODES[arm])
        axis.set(title=metric, xlabel="Independent process in ABBA order", ylabel="GPU milliseconds (process median)")
        for boundary in boundaries:
            axis.axvline(boundary, color="#666666", linestyle="--", linewidth=1)
            axis.text(boundary, .98, "Discovery | Confirmation", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("Production native-resolution CLAS A/B — no frame exclusions")
    figure.tight_layout()
    figure.savefig(output / "ProcessMedians.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    for i, (stage, value) in enumerate(verified["decisions"].items()):
        for j, metric in enumerate(METRICS):
            gain = value["gains"][metric]
            x = j + (i - (len(verified["decisions"]) - 1) * .5) * .15
            axis.errorbar(x, 100*gain["estimate"], yerr=[[100*(gain["estimate"]-gain["lower95"])],
                [100*(gain["upper95"]-gain["estimate"])]], fmt="o", capsize=5, color=stage_colors[stage],
                ecolor=stage_colors[stage], label=stage if j == 0 else None)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xticks=range(len(METRICS)), xticklabels=METRICS, ylabel="Gain (%) with 95% Student-t interval",
             title="Complete ABBA blocks; discovery and confirmation are separate")
    axis.legend(); axis.grid(alpha=.2); figure.tight_layout()
    figure.savefig(output / "BlockGains.png", dpi=150); plt.close(figure)
    save(output / "Verification.json", verified)
    backends = {MODES[r["arm"]]: r["result"]["identity"] for r in manifest["runs"] if r["stage"] == "diagnostic"}
    save(output / "BackendAllocations.json", backends)
    save(output / "Versions.json", {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                                   "platform": platform.platform(), "evidence": str(directory.resolve()),
                                   "evidenceManifestSha256": digest(directory / "Manifest.json"),
                                   "plotSource": "Experiment.json runs[].result.timings[metric].median; raw MaterialBaseline.json frames",
                                   "runnerSha256": digest(Path(__file__))})
    memory_rows = "\n".join("| " + mode + " | " + " | ".join(str(value[key]) for key in
        ("templateBytes", "templateAuxiliaryBytes", "clasBytes", "scratchBytes", "maxPatches")) + " |" for mode, value in backends.items())
    confirmation_text = ""
    if "confirmation" in verified["decisions"]:
        confirmation = verified["decisions"]["confirmation"]
        confirmed_rows = [r for r in rows if r["stage"] == "confirmation"]
        summary = []
        labels = {"clasMs": "CLAS build / instantiate", "tessellationMs": "Tessellation record + publish",
                  "graphMs": "Whole measured graph", "shadowsMs": "Ray-traced Shadows"}
        for metric in METRICS:
            medians = {arm: statistics.median(r[metric] for r in confirmed_rows if r["arm"] == arm) for arm in MODES}
            gain = confirmation["gains"][metric]
            summary.append({"metric": metric, "trianglesMs": medians["A"], "templatesMs": medians["B"],
                            "blocks": gain["blocks"], "gain": gain["estimate"], "lower95": gain["lower95"], "upper95": gain["upper95"]})
        save(output / "ConfirmationSummary.json", summary)
        summary_rows = "\n".join(f"| {labels[r['metric']]} | {r['trianglesMs']:.6f} | {r['templatesMs']:.6f} | "
            f"{100*r['gain']:+.2f}% | [{100*r['lower95']:+.2f}%, {100*r['upper95']:+.2f}%] |" for r in summary)
        blocks = confirmation["gains"]["clasMs"]["blocks"]
        confirmation_text = (f"Confirmation: **n = {blocks} complete ABBA blocks** ({len(confirmed_rows)} independent processes, "
            f"{len(confirmed_rows)//2} per arm). The A/B milliseconds below are medians of process medians; gains and intervals "
            "come from the paired block log-ratio estimator, not their simple ratio. Discovery data is excluded. Positive gain means faster; negative gain means regression.\n\n"
            "| GPU scope | Triangles (ms) | Templates (ms) | Block gain | 95% confidence interval |\n"
            "|---|---:|---:|---:|---:|\n" + summary_rows + "\n\n")
        shadows = confirmation["gains"]["shadowsMs"]
        if shadows["upper95"] < 0:
            confirmation_text += (f"**Ray-traced Shadows regressed by {-100*shadows['estimate']:.1f}%** "
                f"(95% interval: {-100*shadows['upper95']:.2f}%–{-100*shadows['lower95']:.2f}% slower). "
                "This consumer cost remains part of the measured graph result; the CLAS-stage gain does not apply to every pass.\n\n")
    shutil.copy2(__file__, output / "MeasureCLASTessellation.py")
    (output / "report.md").write_text("# CLAS template A/B\n\n" +
        "Scope: fixed native-resolution VBuffer, real ray-traced Shadows and Deferred HDR; startup, compilation, template-pool initialization, CPU waits and readbacks excluded. Graph is an enclosing GPU span, not the sum of nested sections.\n\n" +
        "Tessellation is record + publish; final TLAS construction is included only in graph time. No outliers removed. Exact HDR, depth and shadow bytes gate acceptance. No general renderer speedup is inferred.\n\n" +
        confirmation_text +
        "Allocated backend capacity (bytes), retained during steady state; these are allocations, not live output sizes or whole-renderer memory. Template auxiliary storage includes canonical vertices, indices, build records, addresses and template initialization scratch. Initialization time is excluded by the fixed warmup; its persistent memory is retained here.\n\n" +
        "| Backend | Template storage | Template auxiliary | CLAS storage | Runtime scratch | Max patches |\n|---|---:|---:|---:|---:|---:|\n" + memory_rows + "\n\n" +
        "![Process medians](ProcessMedians.png)\n\n![ABBA block intervals](BlockGains.png)\n\n```json\n" + json.dumps(verified, indent=2) +
        "\n```\n\nReproduce with the current checkout's `python -B Tools/Perf/MeasureCLASTessellation.py report --input " + str(directory) + " --output NEW_DIRECTORY`. Archived scripts are evidence, not executed by verify.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--sample", type=Path, default=ROOT / "Pipelines/Samples/gpu_driven_tessellation.metallic_graph.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1444); p.add_argument("--height", type=int, default=694)
    p.add_argument("--warmup", type=int, default=32); p.add_argument("--frames", type=int, default=64)
    p = commands.add_parser("run")
    p.add_argument("--exe", type=Path, required=True); p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, choices=range(2, 11), default=3)
    p.add_argument("--confirmation-blocks", type=int, choices=range(2, 11), default=5)
    p.add_argument("--timeout", type=int, default=300); p.add_argument("--diagnostics-only", action="store_true")
    for name in ("verify", "report"):
        p = commands.add_parser(name); p.add_argument("--input", type=Path, required=True)
        if name == "report": p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run(args)
    elif args.command == "verify": print(json.dumps(verify(args.input), indent=2))
    else: report(args.input, args.output)


if __name__ == "__main__":
    main()
