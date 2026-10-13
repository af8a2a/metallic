"""Qualify and measure exact material-backend artifacts in real render graphs.

One validation-on pair precedes three validation-off AB/BA/AB process pairs.
General can be the linked backend or the original request, with the same two
SPIR-V cleanup passes applied to both General and the linked NoFiber candidate.
Only graph/pass/dispatch GPU timestamps are measured; shader compilation and
native creation are excluded. Historical verification uses archived helpers.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver
import ValidatePathTraceBackendRendering as rendering

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("backend-link-general", "backend-link-no-fiber")
GENERAL_ARMS = ("backend-link-general", "original-general")
SCOPES = ("graph_ms", "pass_ms", "dispatch_scope_ms")
STATISTICAL_UNIT = "One independent-process median per arm/block; frames are not independent repetitions."
TOLERANCE = dict(absolute=2e-5, relative=2e-5)
MEASUREMENT_SCOPE = ("Graph/pass GPU intervals and the Path trace shading dispatch interval; shader/native compilation, "
                     "readbacks and presentation are outside these measurements. Graph/pass may include GPU upload commands.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value):
    return type(value) is int and value > 0


def positive_finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def selected_arms(general_arm=ARMS[0]):
    require(general_arm in GENERAL_ARMS, "Unsupported General control arm")
    return general_arm, ARMS[1]


def arm_roles(general_arm=ARMS[0]):
    control, candidate = selected_arms(general_arm)
    return dict(control=control, candidate=candidate)


def schedule(pairs=3, general_arm=ARMS[0]):
    require(positive_integer(pairs), "Invalid process pair count")
    arms = selected_arms(general_arm)
    rows = [dict(phase="correctness", block=0, arm=arm, validation=True, label=f"correctness-{arm}") for arm in arms]
    for block in range(pairs):
        for arm in (arms if block % 2 == 0 else tuple(reversed(arms))):
            rows.append(dict(phase="timing", block=block, arm=arm, validation=False, label=f"block-{block:02}-{arm}"))
    return rows


def prepare_dispatch_workload(cases_path, output, width=512, warmup_frames=16, timing_frames=32):
    require(positive_integer(width) and width <= 16384 and positive_integer(warmup_frames) and positive_integer(timing_frames),
            "Invalid resolution or frame window")
    path, configuration, inputs, absent = rendering.prepare_workload(cases_path, output)
    first_capture = warmup_frames + timing_frames
    frame_count = first_capture + 8
    configuration.update(frames=frame_count, warmupFrames=warmup_frames, timingFrames=timing_frames)
    for case in configuration["cases"]:
        require(case["requiredTiming"] == ["PathTrace", "Path trace shading"], "Unexpected GPU timestamp scope")
        require(case.get("useRenderView") is False and len(case.get("cameraFrames", [])) == 1, "Need one PassCamera movement")
        case.update(width=width, height=width, frameCount=frame_count, captureFrames=list(range(first_capture, frame_count)))
        case["cameraFrames"][0]["frame"] = first_capture + 3
        require(len(case["captureOutputs"]) == 7 and case.get("guideCameraStep"), "Need seven moving-camera guides")
    experiment.save(path, configuration)
    validate_workload(configuration)
    return path, configuration, inputs, absent


def validate_workload(configuration):
    require([case["id"] for case in configuration["cases"]] == ["Opaque1", "Mixed1", "Sky"], "Missing fixed dispatch cases")
    first, count = configuration["warmupFrames"], configuration["timingFrames"]
    require(positive_integer(first) and positive_integer(count), "Invalid timing window")
    frame_count = first + count + 8
    require(configuration["frames"] == frame_count, "Changed main-frame/capture window")
    sizes = set()
    for case in configuration["cases"]:
        require(case["frameCount"] == frame_count and case["captureFrames"] == list(range(first + count, frame_count)),
                "Capture overlaps timing or misses temporal frames")
        require(case["requiredTiming"] == ["PathTrace", "Path trace shading"] and case.get("expectedAsyncComputeBranches", 0) == 0,
                "Unexpected dispatch scope or queue topology")
        require(positive_integer(case["width"]) and case["height"] == case["width"], "Invalid dispatch resolution")
        sizes.add(case["width"])
        require(case.get("useRenderView") is False and case.get("requireEnvironment") is True and
                len(case["cameraFrames"]) == 1 and case["cameraFrames"][0]["frame"] == first + count + 3,
                "PassCamera movement must follow the timing window")
        require(len(case["captureOutputs"]) == 7 and len(set(case["captureOutputs"])) == 7 and case.get("guideCameraStep"),
                "Missing guide capture sequence")
    require(len(sizes) == 1, "Resolution differs across cases")


def device_identity(evidence):
    require(len(evidence["devices"]) == 1, "Need one observed device")
    identity = copy.deepcopy(evidence["devices"][0])
    require(type(identity.get("validation_enabled")) is bool and identity.get("synchronization_validation_enabled") is False,
            "Unconfirmed validation configuration")
    # Only this explicit switch changes between qualification and timing.
    identity.pop("validation_enabled")
    return identity


def validate_process(process, execution_id):
    require(isinstance(execution_id, str) and len(execution_id) == 32 and all(c in "0123456789abcdef" for c in execution_id),
            "Missing unique process execution identity")
    require(process["execution_id"] == execution_id and positive_integer(process["pid"]) and
            positive_finite(process["started"]) and positive_finite(process["ended"]) and process["ended"] >= process["started"] and
            type(process["exit_code"]) is int and process["exit_code"] == 0 and process["timed_out"] is False and
            process.get("native_crash", False) is False, "Invalid or failed independent process")


def summarize_pairs(runs, configuration, pairs=3, general_arm=ARMS[0]):
    require(positive_integer(pairs), "Invalid process pair count")
    arms = selected_arms(general_arm)
    timing = [run for run in runs if run["phase"] == "timing"]
    expected = [(row["block"], row["arm"]) for row in schedule(pairs, general_arm) if row["phase"] == "timing"]
    require([(run["block"], run["arm"]) for run in timing] == expected, "Missing, duplicate or reordered process pair")
    seen = set()
    case_ids = [case["id"] for case in configuration["cases"]]
    require(case_ids and len(case_ids) == len(set(case_ids)), "Invalid timing case list")
    for run in timing:
        require(run.get("validation") is False, "GPU timing requires validation off")
        validate_process(run["process"], run["execution_id"])
        require(run["execution_id"] not in seen, "Duplicate process execution identity")
        seen.add(run["execution_id"])
        require(list(run["timings"]) == case_ids, "Missing process case timings")
        for case in case_ids:
            require(set(run["timings"][case]) == set(SCOPES), "Missing measured timestamp scope")
            for scope in SCOPES:
                metric = run["timings"][case][scope]
                require(metric["count"] == configuration["timingFrames"] and positive_integer(metric["count"]) and
                        all(positive_finite(metric[name]) for name in ("min", "median", "p95", "max")) and
                        metric["min"] <= metric["median"] <= metric["p95"] <= metric["max"], "Invalid process timestamp statistics")
    per_arm = {arm: {case: {scope: driver.timings([run["timings"][case][scope]["median"] for run in timing if run["arm"] == arm])
        for scope in SCOPES} for case in case_ids} for arm in arms}
    paired = {}
    for case in case_ids:
        paired[case] = {}
        for scope in SCOPES:
            blocks = []
            for block in range(pairs):
                left, right = (next(run for run in timing if run["block"] == block and run["arm"] == arm) for arm in arms)
                a, b = (run["timings"][case][scope]["median"] for run in (left, right))
                blocks.append(dict(block=block, general_ms=a, no_fiber_ms=b, ratio=b / a,
                                   general_execution_id=left["execution_id"], no_fiber_execution_id=right["execution_id"]))
            ratios = [row["ratio"] for row in blocks]
            statistics = driver.timings(ratios)
            paired[case][scope] = dict(blocks=blocks, ratios=ratios, ratio_statistics=statistics,
                percent_change=100 * (statistics["median"] - 1), percent_change_denominator_arm=arms[0],
                percent_change_definition="100 * (median(candidate_process_median_ms / control_process_median_ms) - 1)")
    frame_p95 = {arm: {case: {scope: driver.timings([run["timings"][case][scope]["p95"] for run in timing if run["arm"] == arm])
        for scope in SCOPES} for case in case_ids} for arm in arms}
    return dict(per_arm=per_arm, paired=paired, per_process_frame_p95=frame_p95,
        frame_p95_scope="Distribution of three independent-process frame P95s; diagnostic only.",
        statistical_unit=STATISTICAL_UNIT, arm_pair=list(arms), arm_roles=arm_roles(general_arm))


def verify_timing_rows(report, configuration):
    timings = experiment.gpu_timings(report, configuration)
    for case in report["cases"]:
        require(all(positive_integer(row["executionId"]) for row in case["frames"]), "Invalid GPU execution identity")
        for row in case["frames"]:
            node = next(node for node in row["nodes"] if node["name"] == "PathTrace")
            scope = next(scope for scope in node["sections"] if scope["name"] == "Path trace shading")
            require(row["graphMs"] + 1e-5 >= node["gpuMs"] and node["gpuMs"] + 1e-5 >= scope["gpuMs"],
                    "GPU scope duration exceeds its enclosing timestamp")
    return timings


def non_target_shaders(directory, manifest, configuration, *, expected_validation):
    """Confirm ordinary runtime bytecode rather than trusting a shared disk cache."""
    result = {}
    tag = manifest["artifact"]["source_tag"]
    for case in configuration["cases"]:
        identities = []
        for path in sorted((directory / "recipes" / case["id"]).glob("*.recipe.json")):
            recipe = experiment.read(path)
            require(recipe["source_device"]["validation_enabled"] is expected_validation, "Non-target recipe validation mismatch")
            for stage_name, stage in recipe["shaders"].items():
                if not stage.get("spirv_file") or stage.get("module_name") == tag:
                    continue
                source = (path.parent / stage["spirv_file"]).resolve(strict=True)
                require(source.is_relative_to(path.parent.resolve()), "Non-target SPIR-V escaped its recipe directory")
                blob = rendering.blob_identity(source)
                require(blob["bytes"] == stage["spirv_bytes"] and blob["fnv1a64"] == stage["input_spirv_hash"],
                        "Non-target device/input SPIR-V differs")
                require(stage["slang_optimization"] == "O1" and stage["debug_mode"] == "release" and
                        stage["descriptor_heap_mode"] == "mapped" and stage["spirv_postprocess"] == "none" and
                        stage["spirv_postprocess_tool_version"] in ("", None) and type(stage["source_identity_ambiguous"]) is bool,
                        "Non-target compiler/descriptor/postprocess policy changed")
                identities.append(dict(pipeline_type=recipe["pipeline_type"], stage=stage_name,
                    module=stage["module_name"], source_entry=stage["source_entry_point"], device_entry=stage["entry_point"],
                    **blob, input_fnv=stage["input_spirv_hash"], descriptor=stage["descriptor_heap_mode"],
                    optimization=stage["slang_optimization"], debug=stage["debug_mode"], postprocess=stage["spirv_postprocess"],
                    postprocess_tool_version=stage["spirv_postprocess_tool_version"], source_identity_ambiguous=stage["source_identity_ambiguous"],
                    device=device_identity(dict(devices=[recipe["source_device"]]))))
        require(identities, "Missing per-case non-target device shader evidence")
        result[case["id"]] = sorted(identities, key=lambda row: json.dumps(row, sort_keys=True))
    return result


def freeze_helpers(output):
    destination = output / "Verifier/Tools/Perf"
    destination.mkdir(parents=True)
    files = []
    for source in sorted(Path(__file__).parent.glob("*.py")):
        target = destination / source.name
        shutil.copyfile(source, target)
        require(experiment.digest(source) == experiment.digest(target), "Helper source changed while archiving the actual verifier")
        files.append(dict(source=str(source.resolve()), file=str(target.relative_to(output)).replace("\\", "/"), sha256=experiment.digest(target)))
    experiment.save(output / "FrozenHelpers.json", files)
    return [Path(row["source"]) for row in files]


def snapshot_check(inputs, absent, inventory, label, checks):
    experiment.check_snapshots(inputs)
    rendering.check_absences(absent)
    observed = experiment.gpu_inventory()
    require(observed is not None and observed == inventory, "Observed GPU inventory/driver changed or is unavailable")
    checks.append(dict(label=label, inputs=experiment.snapshots(inputs), gpu_inventory=observed,
                       absent_inputs=[path for path in absent if not Path(path).exists()]))


def validate_selected_artifact(manifest, arm):
    """Bind the declared arm to its actual frozen compiler and cleanup bytes."""
    require(arm in (*GENERAL_ARMS, ARMS[1]) and manifest["version"] == 1 and manifest["artifact_id"] == arm and
            manifest["requires_no_fiber"] is (arm == ARMS[1]), "Wrong selected artifact or scene eligibility arm")
    provenance = manifest["provenance"]["manifests"]
    require(len(provenance) == 3, "Missing selected artifact preparation/compiler/cleanup provenance")
    paths = [Path(row["file"]).resolve(strict=True) for row in provenance]
    require(len(set(paths)) == 3 and all(path.name == "Manifest.json" for path in paths) and
            all(experiment.digest(path) == row["sha256"] for path, row in zip(paths, provenance)),
            "Selected artifact provenance changed")
    preparation, compilation, analysis = (path.parent for path in paths)
    sample = rendering.reference_sample(experiment.read(preparation / "ReferenceReport.json"), 146)
    require(manifest["reference_request"] == sample["request"] and
            manifest["expected_target"] == {key: sample["request"][key] for key in ("module", "entry")} and
            manifest["reference_settings"] == dict(optimization="O1", descriptor="mapped", debug="disabled",
                working_space="acescg", postprocess="none"), "Selected artifact request/compiler policy changed")
    result = experiment.read(analysis / "summary.json")
    inputs = experiment.read(analysis / "Inputs.json")
    compiled = experiment.read(compilation / "summary.json")
    require(result["status"] == "passed" and result["cleanup_passes"] ==
            ["--eliminate-dead-branches", "--simplify-instructions"] and compiled["status"] == "passed" and
            compiled["variants"] == list(rendering.ARMS) and compiled["preparation_manifest_sha256"] == provenance[0]["sha256"],
            "Selected artifact compiler/cleanup policy changed")
    for key, path, row in zip(("preparation", "compile"), paths[:2], provenance[:2]):
        linked = inputs["manifests"][key]
        require(Path(linked["path"]).resolve() == path and linked["sha256"] == row["sha256"],
                "Selected artifact cleanup references a different compiler/preparation")
    artifact = manifest["artifact"]
    source = Path(artifact["file"]).resolve(strict=True)
    require(source == (analysis / arm / "Cleaned.spv").resolve(strict=True), "Selected arm uses a different cleanup artifact")
    identity = rendering.blob_identity(source)
    cleaned = result["outputs"][arm]["cleaned"]
    require(all(artifact[key] == identity[key] for key in identity) and
            identity["bytes"] == cleaned["bytes"] and identity["sha256"] == cleaned["sha256"] and
            artifact["entry_main"] == "main" and artifact["source_tag"] == f"Artifacts/PathBackend/{arm}/TwoPassCleaned" and
            artifact["spirv_postprocess"] == "dead-branches", "Selected arm cleanup bytes/identity differ")
    version_text = "\n".join((analysis / f"opt-version.{stream}.log").read_text(encoding="utf-8") for stream in ("stdout", "stderr"))
    version = next((line.strip() for line in version_text.splitlines() if line.startswith("SPIRV-Tools")), "")
    require(version.startswith("SPIRV-Tools") and artifact["tool_version"] == version, "Selected arm cleanup tool differs")
    validation = experiment.read(analysis / arm / "cleaned-validation.process.json")
    command = validation["command"]
    require(validation["status"] == "passed" and validation["exit_code"] == 0 and len(command) == 5 and
            Path(command[0]).name == "spirv-val.exe" and command[1:4] ==
            ["--target-env", "vulkan1.4", "--scalar-block-layout"] and Path(command[4]).resolve() == source,
            "Selected cleanup artifact validation differs")
    original_input = inputs["inputs"][arm]
    raw_path = Path(original_input["path"]).resolve(strict=True)
    records = [record for record in compiled["runs"] if record["label"] == arm and record["repetition"] == 0]
    require(raw_path.is_relative_to(compilation) and experiment.digest(raw_path) == original_input["sha256"] ==
            result["outputs"][arm]["raw"]["sha256"] and len(records) == 1 and records[0] == original_input["record"] and
            records[0]["spirv_sha256"] == original_input["sha256"], "Selected arm raw compiler/cleanup chain differs")
    if arm == "original-general":
        reference = rendering.blob_identity(preparation / "Reference.spv")
        require(reference["bytes"] == sample["spirv_bytes"] and reference["fnv1a64"] == sample["spirv_fnv1a64"] and
                reference["sha256"] == original_input["sha256"], "Original General control differs from the actual original request")


def execute(command, environment, destination, timeout):
    execution_id = uuid.uuid4().hex
    started = time.time()
    with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, (destination / "stderr.log").open("w", encoding="utf-8") as stderr:
        with driver.suppress_windows_error_dialogs():
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            timed_out = True
    metadata = dict(execution_id=execution_id, pid=process.pid, command=command, started=started, ended=time.time(),
                    exit_code=code, timed_out=timed_out, native_crash=driver.native_process_crashed(code),
                    environment={key: value for key, value in environment.items() if key.startswith("METALLIC_")})
    experiment.save(destination / "process.json", metadata)
    require(code == 0 and not timed_out, f"Artifact render process failed ({code}); preserved {destination}")
    return metadata


def run(args, output, summary):
    require(args.pairs == 3 and args.resolution == 512 and args.warmup_frames == 16 and args.timing_frames == 32,
            "This predeclared experiment requires 512x512, 16 warmup, 32 timing frames and three pairs")
    general_arm = getattr(args, "general_arm", ARMS[0])
    arms = selected_arms(general_arm)
    manifests, artifact_inputs = rendering.prepare_artifacts(args, output)
    for arm in arms:
        validate_selected_artifact(experiment.read(manifests[arm]), arm)
    path, configuration, workload_inputs, absent = prepare_dispatch_workload(args.cases.resolve(strict=True), output,
        args.resolution, args.warmup_frames, args.timing_frames)
    executable = args.executable.resolve(strict=True)
    helpers = freeze_helpers(output)
    inputs = experiment.input_identity(executable, artifact_inputs + workload_inputs + helpers + [
        ROOT / "Source/Runtime/Render/Core/ShaderRegistry.h", ROOT / "tests/rhi/ShaderArtifactFixture.h"], production=True)
    experiment.save(output / "Inputs.json", inputs)
    experiment.save(output / "AbsentInputs.json", absent)
    inventory = experiment.gpu_inventory()
    require(inventory, "Need an observed NVIDIA GPU inventory before timing")
    rows = schedule(args.pairs, general_arm)
    summary.update(command="dispatch", pairs=args.pairs, workload=configuration, tolerance=TOLERANCE,
        schedule=rows, artifacts={arm: str(manifests[arm]) for arm in arms}, executable=str(executable),
        general_arm=general_arm, arm_pair=list(arms), arm_roles=arm_roles(general_arm),
        gpu_inventory=inventory, statistical_unit=STATISTICAL_UNIT,
        scope=MEASUREMENT_SCOPE,
        caveats=["Fixed SHARC-off, CUSTOM0, no-Fiber scenes only; no full-scene or frame-rate claim.",
                 "Validation-on qualification precedes validation-off measurements; both compare actual HDR, temporal and guide outputs.",
                 "Cooperative locks do not establish global GPU exclusivity; GPU clocks are unaltered.",
                 "Three independent-process pairs are exploratory; frame samples are not independent repetitions."])
    checks = []
    device = None
    baseline = None
    ordinary_shaders = None
    for row in rows:
        label = row["label"]
        snapshot_check(inputs, absent, inventory, "before-" + label, checks)
        experiment.save(output / "SnapshotChecks.json", checks)
        destination = output / label
        destination.mkdir()
        environment = experiment.environment_for(dict(optimization="O1", postprocess="none", backend="pipeline", disable_driver_optimization=False),
            experiment.clean_environment(), internal_cache="enabled")
        environment.update(METALLIC_SHADER_ARTIFACT_MANIFEST=str(manifests[row["arm"]]), METALLIC_MATERIAL_BASELINE_CASES=str(path),
            METALLIC_PSO_CACHE_DIRECTORY=str(destination / "pso-cache"), METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / "recipes"),
            METALLIC_WORKING_COLOR_SPACE="acescg")
        command = [str(executable), "--gtest_filter=RHIRendering.material_phase0_baseline",
                   "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)]
        print(f"{label}: exact artifact, validation {'on' if row['validation'] else 'off'}", flush=True)
        metadata = execute(command, environment, destination, args.timeout)
        manifest = experiment.read(manifests[row["arm"]])
        evidence = rendering.verify_runtime(destination, manifest, configuration, expected_validation=row["validation"])
        ordinary = non_target_shaders(destination, manifest, configuration, expected_validation=row["validation"])
        if ordinary_shaders is None:
            ordinary_shaders = ordinary
        require(ordinary_shaders == ordinary, "Actual non-target device shaders changed across processes")
        observed = device_identity(evidence)
        if device is None:
            device = observed
        require(device == observed, "Runtime device/driver/debug configuration changed")
        require(all(stage["device"] == observed for stages in ordinary.values() for stage in stages), "Non-target pipeline used a different device")
        current = Path(evidence["report"])
        if baseline is None:
            baseline = current
        comparison = experiment.compare_readbacks(baseline, current, configuration, **TOLERANCE)
        require(all(case["qualified"] for case in comparison), "Artifact exceeded the predeclared readback tolerance")
        item = dict(**row, execution_id=metadata["execution_id"], process=metadata, evidence=evidence,
                    non_target_shaders=ordinary, correctness=comparison)
        if row["phase"] == "timing":
            item["timings"] = verify_timing_rows(experiment.read(current), configuration)
        summary["runs"].append(item)
        snapshot_check(inputs, absent, inventory, "after-" + label, checks)
        experiment.save(output / "SnapshotChecks.json", checks)
        experiment.save(output / "summary.json", summary)
    summary.update(devices=device, non_target_shaders=ordinary_shaders, correctness_qualified=True,
                   timing_summary=summarize_pairs(summary["runs"], configuration, args.pairs, general_arm))


def validate_seal(output):
    require(experiment.read(output / "Manifest.json") == {str(path.relative_to(output)).replace("\\", "/"): experiment.digest(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path != output / "Manifest.json"}, "Sealed dispatch evidence changed")


def verify_records(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    validate_seal(output)
    summary = experiment.read(output / "summary.json")
    require(summary["schema_version"] == 1 and summary["status"] == "passed" and summary["command"] == "dispatch" and
            summary["pairs"] == 3 and summary["tolerance"] == TOLERANCE and summary["correctness_qualified"] is True,
            "Incomplete, failed or changed dispatch experiment")
    general_arm = summary["general_arm"]
    arms = selected_arms(general_arm)
    require(summary["arm_pair"] == list(arms) and summary["arm_roles"] == arm_roles(general_arm) and
            list(summary["artifacts"]) == list(arms), "Changed selected control/candidate arm roles")
    configuration = summary["workload"]
    validate_workload(configuration)
    require(configuration["warmupFrames"] == 16 and configuration["timingFrames"] == 32 and
            all(case["width"] == 512 for case in configuration["cases"]), "Changed predeclared workload")
    require(experiment.read(output / "Cases.json") == configuration, "Runtime workload differs from recorded configuration")
    for case in configuration["cases"]:
        graph_path = Path(case["graph"]).resolve(strict=True)
        require(graph_path == (output / f"{case['id']}.json").resolve(), "Runtime graph escaped the frozen workload")
        nodes = [node for node in experiment.read(graph_path)["nodes"] if node["type"] == "ScenePathTracePass"]
        require(len(nodes) == 1, "Missing fixed path-trace pass")
        properties = nodes[0]["properties"]
        require(properties["samples"] == 2 and properties["maxDepth"] == 12 and properties["cacheMode"] == "off" and
                properties["bsdf"] == "openpbr" and properties["exportDenoiserGuides"] is True and properties["path"] == case["scene"],
                "Frozen graph shader/resource configuration changed")
    expected = schedule(3, general_arm)
    require(summary["schedule"] == expected and len(summary["runs"]) == len(expected), "Missing or changed dispatch schedule")
    helpers = experiment.read(output / "FrozenHelpers.json")
    require(any(Path(row["file"]).name == Path(__file__).name for row in helpers), "Actual verifier source was not frozen")
    for row in helpers:
        source = (output / row["file"]).resolve(strict=True)
        require(source.is_relative_to((output / "Verifier").resolve()) and experiment.digest(source) == row["sha256"], "Frozen helper changed")
    inputs, absent = experiment.read(output / "Inputs.json"), experiment.read(output / "AbsentInputs.json")
    require(all(inputs.get(row["source"]) == row["sha256"] for row in helpers), "Archived helpers differ from the actual measured input snapshot")
    checks = experiment.read(output / "SnapshotChecks.json")
    require([row["label"] for row in checks] == [prefix + row["label"] for row in expected for prefix in ("before-", "after-")],
            "Missing before/after input checks")
    require(all(row["inputs"] == inputs and row["gpu_inventory"] == summary["gpu_inventory"] and row["absent_inputs"] == absent for row in checks),
            "Inputs, GPU inventory or scene sidecars changed during execution")
    if current_inputs:
        experiment.check_snapshots(inputs)
        rendering.check_absences(absent)
        require(experiment.gpu_inventory() == summary["gpu_inventory"], "Current GPU inventory differs from recorded execution")
    baseline = None
    execution_ids = set()
    previous_end = None
    checked_provenance = set()
    checked_artifacts = set()
    for item, row in zip(summary["runs"], expected):
        require({key: item[key] for key in row} == row, "Reordered or changed process arm")
        destination = output / row["label"]
        process = experiment.read(destination / "process.json")
        require(item["process"] == process, "Process metadata differs from its actual sidecar")
        validate_process(process, item["execution_id"])
        require(item["execution_id"] not in execution_ids and (previous_end is None or process["started"] >= previous_end),
                "Duplicate or overlapping independent process")
        execution_ids.add(item["execution_id"])
        previous_end = process["ended"]
        require(process["command"] == [summary["executable"], "--gtest_filter=RHIRendering.material_phase0_baseline",
            "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)], "Changed runtime command")
        environment = process["environment"]
        require(environment == dict(METALLIC_SHADER_DEBUG_MODE="disabled", METALLIC_SLANG_OPTIMIZATION="O1", METALLIC_SPIRV_POSTPROCESS="none",
            METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION="0", METALLIC_VK_COMPUTE_BACKEND="pipeline", METALLIC_VK_INTERNAL_PIPELINE_CACHE="enabled",
            METALLIC_SHADER_ARTIFACT_MANIFEST=summary["artifacts"][row["arm"]], METALLIC_MATERIAL_BASELINE_CASES=str(output / "Cases.json"),
            METALLIC_PSO_CACHE_DIRECTORY=str(destination / "pso-cache"), METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / "recipes"),
            METALLIC_WORKING_COLOR_SPACE="acescg"), "Changed runtime compiler/artifact/cache environment")
        manifest_path = Path(summary["artifacts"][row["arm"]]).resolve()
        require(manifest_path == (output / f"{row['arm']}.artifact.json").resolve(), "Artifact manifest escaped the evidence directory")
        manifest = experiment.read(manifest_path)
        require(manifest["artifact_id"] == row["arm"] and manifest["requires_no_fiber"] is row["arm"].endswith("no-fiber"), "Wrong scene eligibility arm")
        for provenance in manifest["provenance"]["manifests"]:
            provenance_path = Path(provenance["file"]).resolve(strict=True)
            require(experiment.digest(provenance_path) == provenance["sha256"], "Earlier immutable compiler/cleanup provenance changed")
            if provenance_path not in checked_provenance:
                rendering.validate_manifest(provenance_path)
                checked_provenance.add(provenance_path)
        if row["arm"] not in checked_artifacts:
            validate_selected_artifact(manifest, row["arm"])
            checked_artifacts.add(row["arm"])
        evidence = rendering.verify_runtime(destination, manifest, configuration, expected_validation=row["validation"])
        require(evidence == item["evidence"] and device_identity(evidence) == summary["devices"], "Runtime artifact/device evidence changed")
        require(non_target_shaders(destination, manifest, configuration, expected_validation=row["validation"]) ==
                item["non_target_shaders"] == summary["non_target_shaders"], "Non-target device shader evidence changed")
        path = Path(evidence["report"])
        baseline = baseline or path
        comparison = experiment.compare_readbacks(baseline, path, configuration, **TOLERANCE)
        require(comparison == item["correctness"] and all(case["qualified"] for case in comparison), "Readback qualification changed")
        if row["phase"] == "timing":
            require(verify_timing_rows(experiment.read(path), configuration) == item["timings"], "GPU timing evidence changed")
        else:
            require("timings" not in item, "Validation-on qualification was included in GPU performance statistics")
    require(summarize_pairs(summary["runs"], configuration, 3, general_arm) == summary["timing_summary"] and summary["statistical_unit"] == STATISTICAL_UNIT,
            "Paired process timing summary changed")
    return summary


def verify(output, *, current_inputs=False):
    """Replay the sealed, actual verifier; future working-tree edits are irrelevant."""
    output = Path(output).resolve(strict=True)
    validate_seal(output)
    source = output / "Verifier/Tools/Perf/MeasurePathTraceBackendDispatch.py"
    require(source.is_file(), "Missing archived actual verifier")
    command = [sys.executable, "-B", str(source), "_verify-frozen", str(output)]
    if current_inputs:
        command.append("--current-inputs")
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, "Archived dispatch verification failed: " + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    for option in ("preparation", "compile", "analysis", "reference", "cases", "executable", "output-dir"):
        launch.add_argument("--" + option, type=Path, required=True)
    launch.add_argument("--timeout", type=float, default=1200)
    launch.add_argument("--pairs", type=int, default=3)
    launch.add_argument("--resolution", type=int, default=512)
    launch.add_argument("--warmup-frames", type=int, default=16)
    launch.add_argument("--timing-frames", type=int, default=32)
    launch.add_argument("--general-arm", choices=GENERAL_ARMS, default=ARMS[0],
                        help="General control artifact; both use the same two cleanup passes.")
    for name in ("verify", "check", "_verify-frozen"):
        check = sub.add_parser(name)
        check.add_argument("output", type=Path)
        check.add_argument("--current-inputs", action="store_true")
    args = parser.parse_args()
    if args.command != "run":
        summary = verify_records(args.output, current_inputs=args.current_inputs) if args.command == "_verify-frozen" else verify(
            args.output, current_inputs=args.current_inputs or args.command == "check")
        report = summary if args.command == "_verify-frozen" else dict(status=summary["status"], runs=len(summary["runs"]),
            scope=MEASUREMENT_SCOPE, timing_summary=summary["timing_summary"])
        if args.command != "_verify-frozen" and summary["scope"] != MEASUREMENT_SCOPE:
            report["recorded_scope"] = summary["scope"]
            report["scope_note"] = "The archived exclusion of uploads was too broad; the sealed measurements are unchanged."
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    require(positive_finite(args.timeout), "Invalid process timeout")
    for name in experiment.LOCK_NAMES:
        require(not (ROOT / "build" / name).exists(), "Existing performance experiment lock; inspect its live owner")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(schema_version=1, status="running", runs=[])
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), output=str(output), command=args.command), stream)
    try:
        run(args, output, summary)
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "summary.json", summary)
        experiment.seal(output)
        lock.unlink()
    verify(output)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
