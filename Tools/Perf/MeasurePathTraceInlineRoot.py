"""Measure the production 12-byte descriptor root against a 24-byte inline root.

Two validation-on correctness processes precede three validation-off AB/BA/AB
process pairs. All GPU work is serialized and bounded. HDR/temporal/guide
readbacks follow the timing window. No shader files or source fixtures change.
Verification uses the current trusted helpers, never archived executable code.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
import MeasurePathTraceBackendDispatch as dispatch
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver
import ValidatePathTraceBackendRendering as rendering

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("descriptor-root12", "inline-root24")
PROPERTY = "experimentalInlineComputeRoot"
CASES = ("Opaque1", "Mixed1", "Sky")
TARGET_MODULES = ("Features/PathTracing/OpenPBRRayQueryPathTrace",
                  "Features/PathTracing/OpenPBRRayQueryPathTraceGuides")
TOLERANCE = dict(absolute=2e-5, relative=2e-5)
POLICY = dict(optimization="O1", postprocess="none", backend="pipeline", disable_driver_optimization=False)
require = dispatch.require


def schedule():
    rows = [dict(phase="correctness", block=0, arm=arm, validation=True,
                 label=f"correctness-{arm}") for arm in ARMS]
    for block in range(3):
        for arm in (ARMS if block % 2 == 0 else ARMS[::-1]):
            rows.append(dict(phase="timing", block=block, arm=arm, validation=False,
                             label=f"block-{block:02}-{arm}"))
    return rows


def path_trace_node(graph):
    nodes = [node for node in graph["nodes"] if node["type"] == "ScenePathTracePass"]
    require(len(nodes) == 1 and len(graph["nodes"]) == 1 and not graph.get("edges"),
            "Require a single production ScenePathTracePass without postprocessing")
    require(nodes[0]["name"] == "PathTrace", "Unexpected path-trace timing node")
    return nodes[0]


def prepare_workload(cases_path, output):
    original = experiment.read(cases_path)
    configuration = copy.deepcopy(original)
    configuration.update(frames=56, warmupFrames=16, timingFrames=32, cases=[])
    configuration.pop("assetFiles", None)  # Snapshot only the selected cases' actual inputs below.
    inputs, absent = [cases_path], []
    for source in original["cases"]:
        if source["id"] not in CASES:
            continue
        require(not any(key in source for key in ("materialAsset", "materialRoot")),
                "This experiment does not qualify external material libraries")
        case = copy.deepcopy(source)
        graph_path = (cases_path.parent / case["graph"]).resolve(strict=True)
        graph = experiment.read(graph_path)
        properties = path_trace_node(graph)["properties"]
        require(properties.get("bsdf") == "openpbr" and properties.get("exportDenoiserGuides") is True,
                "Require production OpenPBR with denoiser guides")
        properties.update(samples=2, maxDepth=12, cacheMode="off", accumulate=False,
                          outputLinear=True, experimentalInlineComputeRoot=False)
        require(not properties.get("splitPrimaryGuides", False) and
                properties.get("traversalBackend", "rayquery") in ("rayquery", "ray-query", "rayQuery"),
                "Inline root is restricted to the ordinary compute path")
        scene = (ROOT / properties["path"]).resolve(strict=True)
        require(scene.suffix.lower() == ".gltf", "Require an inspectable glTF fixture")
        properties["path"] = str(scene)
        meshlets = Path(str(scene) + ".meshlets.bin")
        require(meshlets.is_file(), "Prepopulate the scene's meshlet cache before this experiment")
        inputs += [graph_path, scene, meshlets]
        for item in experiment.read(scene).get("buffers", []) + experiment.read(scene).get("images", []):
            uri = item.get("uri", "")
            if uri and not uri.startswith("data:"):
                inputs.append((scene.parent / uri).resolve(strict=True))
        sidecar = scene.with_suffix(".metallic_scene.json")
        if sidecar.exists():
            data = experiment.read(sidecar)
            require(not data.get("world", {}).get("lighting", {}).get("autoExposure", {}).get("enabled", False),
                    "Fixture sidecar must keep manual exposure")
            # This bounded fixture has no external material library or graph assets.
            require("asset://" not in sidecar.read_text(encoding="utf-8"), "External material sidecar is outside this experiment")
            inputs.append(sidecar)
            environment_path = data.get("world", {}).get("environment", {}).get("path")
            if environment_path:
                inputs.append((ROOT / environment_path).resolve(strict=True))
        else:
            absent.append(str(sidecar))
        environment = (ROOT / case["environment"]["path"]).resolve(strict=True)
        case["environment"]["path"] = str(environment)
        inputs.append(environment)
        case.update(scene=str(scene), useRenderView=False, requireEnvironment=True,
                    width=512, height=512, frameCount=56, captureFrames=list(range(48, 56)))
        require(case.get("requiredTiming") == ["PathTrace", "Path trace shading"] and
                len(case.get("cameraFrames", [])) == 1 and len(case.get("captureOutputs", [])) == 7 and
                case.get("guideCameraStep"), "Require the fixed temporal and seven-guide workload")
        case["cameraFrames"][0]["frame"] = 51
        graph_target = output / f"{case['id']}.json"
        experiment.save(graph_target, graph)
        case["graph"] = str(graph_target)
        configuration["cases"].append(case)
        inputs.append(graph_target)
    dispatch.validate_workload(configuration)
    cases_target = output / "Cases.json"
    experiment.save(cases_target, configuration)
    inputs.append(cases_target)
    arms = {}
    for arm in ARMS:
        directory = output / "workloads" / arm
        directory.mkdir(parents=True)
        arm_config = copy.deepcopy(configuration)
        for case in arm_config["cases"]:
            graph = experiment.read(Path(case["graph"]))
            path_trace_node(graph)["properties"][PROPERTY] = arm == ARMS[1]
            path = directory / f"{case['id']}.json"
            experiment.save(path, graph)
            case["graph"] = str(path)
            inputs.append(path)
        path = directory / "Cases.json"
        experiment.save(path, arm_config)
        inputs.append(path)
        arms[arm] = dict(file=str(path), configuration=arm_config)
    return configuration, arms, inputs, sorted(set(absent))


def comparable_report(report, configuration, arm_configuration, arm):
    # Ordinary MaterialBaseline v1 reports have no status field. Process success,
    # complete cases and valid finite readbacks are their qualification contract.
    require(report.get("version") == 1 and not report.get("error") and
            ("status" not in report or report["status"] == "passed") and
            type(report.get("validation")) is bool and report.get("synchronizationValidation") is False,
            "Invalid or failed production render report")
    require([case["id"] for case in report["cases"]] == list(CASES), "Incomplete or reordered cases")
    require(all(case.get("validHDR") is True and case.get("nonfiniteComponents") == 0 for case in report["cases"]),
            "Invalid or nonfinite HDR case")
    result = copy.deepcopy(report)
    for case, specification, canonical in zip(result["cases"], arm_configuration["cases"], configuration["cases"]):
        experiment.validate_requested_case(case, specification, arm_configuration)
        properties = path_trace_node(case["resolvedGraph"])["properties"]
        require(properties.get(PROPERTY) is (arm == ARMS[1]), "Runtime graph used the wrong root arm")
        require(properties.get("samples") == 2 and properties.get("maxDepth") == 12 and
                properties.get("cacheMode") == "off" and properties.get("bsdf") == "openpbr" and
                properties.get("exportDenoiserGuides") is True, "Runtime shading settings changed")
        # Normalize only the declared treatment and its graph filename. Raw reports remain intact.
        properties[PROPERTY] = False
        case["graph"] = canonical["graph"]
    return result


def runtime_identity(directory, configuration, arm, validation):
    text = "\n".join((directory / f"{stream}.log").read_text(encoding="utf-8", errors="replace")
                     for stream in ("stdout", "stderr")).lower()
    require(not any(marker in text for marker in ("initial material compilation failed", "displaying the error material",
                "validation error", "vuid-", "[shaderartifactoverride]")), "Runtime fallback, override or validation error")
    devices, targets, ordinary, files = [], {}, {}, []
    for case in configuration["cases"]:
        target_rows, other_rows = [], []
        manifests = sorted((directory / "recipes" / case["id"]).glob("*.recipe.json"))
        require(manifests, "Missing per-case recipes; use the MaterialBaseline per-case recipe capture implementation")
        for path in manifests:
            recipe = experiment.read(path)
            device = recipe["source_device"]
            require(device.get("validation_enabled") is validation, "Recipe validation policy changed")
            normalized_device = dispatch.device_identity(dict(devices=[device]))
            if normalized_device not in devices:
                devices.append(normalized_device)
            stages = []
            for stage_name, stage in recipe["shaders"].items():
                if not stage.get("spirv_file"):
                    continue
                blob_path = (path.parent / stage["spirv_file"]).resolve(strict=True)
                require(blob_path.parent == path.parent.resolve(), "SPIR-V escaped its recipe directory")
                blob = rendering.blob_identity(blob_path)
                require(blob["bytes"] == stage["spirv_bytes"] and blob["fnv1a64"] == stage["input_spirv_hash"],
                        "Recipe's actual device SPIR-V hash/size differs")
                require(stage.get("source_identity_ambiguous") is False and stage.get("debug_mode") == "release" and
                        stage.get("slang_optimization") == "O1" and stage.get("descriptor_heap_mode") == "mapped" and
                        stage.get("spirv_postprocess") == "none", "Compiler, descriptor or provenance policy changed")
                stages.append(dict(stage=stage_name, module=stage["module_name"], entry=stage["source_entry_point"],
                    device_entry=stage["entry_point"], specialization=stage.get("specialization_constants", []), **blob))
            row = dict(pipeline_type=recipe["pipeline_type"], state=recipe["state"], shaders=stages,
                       pso_hash=recipe["expected_pso_hash"])
            is_target = any(stage["module"] in TARGET_MODULES for stage in stages)
            if is_target:
                state = recipe["state"]
                require(recipe["pipeline_type"] == "compute" and len(stages) == 1 and stages[0]["stage"] == "compute" and
                        state["bindless_user_push_data_size"] == (24 if arm == ARMS[1] else 12) and
                        state["uses_bindless_heap"] and not state["indirect_bindable"] and
                        state["compute_backend"] == 0 and not state["disable_optimization"], "Wrong actual root layout/backend")
                if row not in target_rows:
                    target_rows.append(row)
            elif row not in other_rows:
                other_rows.append(row)
            files.append(dict(file=str(path.relative_to(directory)), sha256=experiment.digest(path)))
        require(len(target_rows) == 1, "Need exactly one unambiguous production target PSO per case")
        require(other_rows, "Missing non-target device shader evidence")
        targets[case["id"]] = target_rows[0]
        ordinary[case["id"]] = sorted(other_rows, key=lambda row: json.dumps(row, sort_keys=True))
    require(len(devices) == 1, "Device/driver changed within one process")
    return dict(device=devices[0], targets=targets, ordinary=ordinary, recipes=files,
        coverage="Per-case production graph and actual native PSO recipes; no per-dispatch command-buffer binding receipt.")


def summarize(runs):
    timed = [run for run in runs if run["phase"] == "timing"]
    require([(run["block"], run["arm"]) for run in timed] ==
            [(row["block"], row["arm"]) for row in schedule() if row["phase"] == "timing"], "Incomplete AB/BA/AB schedule")
    paired = {}
    for case in CASES:
        paired[case] = {}
        for scope in dispatch.SCOPES:
            blocks = []
            for block in range(3):
                a, b = (next(run for run in timed if run["block"] == block and run["arm"] == arm) for arm in ARMS)
                left, right = (run["timings"][case][scope]["median"] for run in (a, b))
                blocks.append(dict(block=block, baseline_ms=left, candidate_ms=right, ratio=right / left,
                    baseline_execution_id=a["execution_id"], candidate_execution_id=b["execution_id"]))
            ratios = driver.timings([row["ratio"] for row in blocks])
            paired[case][scope] = dict(blocks=blocks, ratio_statistics=ratios, percent_change=100 * (ratios["median"] - 1))
    return dict(paired=paired, statistical_unit=dispatch.STATISTICAL_UNIT,
        percent_change_definition="100 * (median(candidate_process_median_ms / baseline_process_median_ms) - 1)",
        decision="inconclusive", decision_reason="Three exploratory process pairs; no predeclared performance acceptance threshold or confirmation stage.")


def validate_target_pair(targets):
    if not all(arm in targets for arm in ARMS):
        return
    for case in CASES:
        shapes = []
        for arm in ARMS:
            shape = copy.deepcopy(targets[arm][case])
            shape.pop("pso_hash")
            shape["state"].pop("bindless_user_push_data_size")
            for stage in shape["shaders"]:
                for key in ("bytes", "sha256", "fnv1a64"):
                    stage.pop(key)
            shapes.append(shape)
        require(shapes[0] == shapes[1], "Target module/entry/specialization or non-root pipeline state differs between arms")
        require(targets[ARMS[0]][case]["shaders"][0]["sha256"] != targets[ARMS[1]][case]["shaders"][0]["sha256"],
                "Both arms used identical target SPIR-V; the root treatment was not established")


def runtime_environment(arm_file, destination, *, live=True):
    base = experiment.clean_environment() if live else {"METALLIC_SHADER_DEBUG_MODE": "disabled"}
    environment = experiment.environment_for(POLICY, base, internal_cache="enabled")
    environment.update(METALLIC_MATERIAL_BASELINE_CASES=arm_file,
        METALLIC_PSO_CACHE_DIRECTORY=str(destination / "pso-cache"),
        METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / "recipes"), METALLIC_WORKING_COLOR_SPACE="acescg")
    return environment


def execute(command, environment, destination, timeout):
    """Retain process evidence and stop only this child on timeout or interruption."""
    metadata = dict(execution_id=uuid.uuid4().hex, pid=None, command=command, started=time.time(),
        ended=None, exit_code=None, timed_out=False, interrupted=False,
        environment={key: value for key, value in environment.items() if key.startswith("METALLIC_")})
    process = None
    with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, \
         (destination / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            with driver.suppress_windows_error_dialogs():
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            metadata["pid"] = process.pid
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                metadata["timed_out"] = True
                raise
            except KeyboardInterrupt:
                metadata["interrupted"] = True
                raise
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                metadata["exit_code"] = process.returncode
                metadata["native_crash"] = driver.native_process_crashed(process.returncode)
            metadata["ended"] = time.time()
            experiment.save(destination / "process.json", metadata)
    require(metadata["exit_code"] == 0, f"Render process failed ({metadata['exit_code']}); retained {destination}")
    return metadata


def analyze(output, summary):
    configuration = summary["workload"]
    dispatch.validate_workload(configuration)
    require(experiment.read(output / "Cases.json") == configuration, "Canonical workload changed")
    expected = schedule()
    require(summary["schedule"] == expected and len(summary["runs"]) == len(expected), "Incomplete process evidence")
    baseline, previous_end, execution_ids = None, None, set()
    target_identities, ordinary, device = {}, None, None
    for item, row in zip(summary["runs"], expected):
        require(all(item[key] == value for key, value in row.items()), "Process order/arm changed")
        destination = output / row["label"]
        process = experiment.read(destination / "process.json")
        require(process == item["process"], "Process sidecar changed")
        dispatch.validate_process(process, item["execution_id"])
        expected_command = [summary["executable"], "--gtest_filter=RHIRendering.material_phase0_baseline",
            "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)]
        expected_environment = runtime_environment(summary["workloads"][row["arm"]]["file"], destination, live=False)
        require(process["command"] == expected_command and process["environment"] ==
                {key: value for key, value in expected_environment.items() if key.startswith("METALLIC_")},
                "Process command/compiler/cache environment differs from the declared experiment")
        require(process["execution_id"] not in execution_ids and
                (previous_end is None or process["started"] >= previous_end), "Processes overlap or reuse execution IDs")
        previous_end = process["ended"]
        execution_ids.add(process["execution_id"])
        raw = experiment.read(destination / "MaterialBaseline.json")
        require(raw.get("validation") is row["validation"], "Report validation policy changed")
        arm_configuration = summary["workloads"][row["arm"]]["configuration"]
        require(experiment.read(summary["workloads"][row["arm"]]["file"]) == arm_configuration, "Arm workload changed")
        normalized = comparable_report(raw, configuration, arm_configuration, row["arm"])
        comparison_path = destination / "ComparableReport.json"
        require(experiment.read(comparison_path) == normalized, "Comparable report changed")
        identity = runtime_identity(destination, arm_configuration, row["arm"], row["validation"])
        require(identity == item["identity"], "Runtime identity evidence changed")
        device = identity["device"] if device is None else device
        ordinary = identity["ordinary"] if ordinary is None else ordinary
        require(identity["device"] == device and identity["ordinary"] == ordinary, "Device or non-target pipelines changed")
        require(target_identities.setdefault(row["arm"], identity["targets"]) == identity["targets"], "Target SPIR-V/layout changed within an arm")
        validate_target_pair(target_identities)
        baseline = baseline or comparison_path
        comparison = experiment.compare_readbacks(baseline, comparison_path, configuration, **TOLERANCE)
        require(comparison == item["correctness"] and all(case["qualified"] for case in comparison), "HDR/guide qualification changed")
        if row["phase"] == "timing":
            require(dispatch.verify_timing_rows(raw, arm_configuration) == item["timings"], "Timing rows changed")
        else:
            require("timings" not in item, "Validation-on run entered timing statistics")
    return summarize(summary["runs"])


def run(args, output, summary):
    configuration, arms, workload_inputs, absent = prepare_workload(args.cases.resolve(strict=True), output)
    executable = args.executable.resolve(strict=True)
    helper_names = (Path(__file__).name, "MeasurePathTraceBackendDispatch.py", "RunComputePipelineExperiments.py",
                    "RunShaderPipelineDriver.py", "ValidatePathTraceBackendRendering.py")
    archived = output / "Sources"
    archived.mkdir()
    for name in helper_names:
        shutil.copyfile(Path(__file__).with_name(name), archived / name)
    cpp = [path for path in (ROOT / "Source/Runtime/Render").rglob("*") if path.suffix in (".cpp", ".h")]
    inputs = experiment.input_identity(executable, workload_inputs + cpp +
        [Path(__file__).with_name(name) for name in helper_names], production=True)
    experiment.save(output / "Inputs.json", inputs)
    experiment.save(output / "AbsentInputs.json", absent)
    inventory = experiment.gpu_inventory()
    require(inventory, "Need an observed NVIDIA GPU/driver inventory")
    summary.update(workload=configuration, workloads=arms, executable=str(executable), schedule=schedule(),
        tolerance=TOLERANCE, gpu_inventory=inventory, scope=dispatch.MEASUREMENT_SCOPE,
        cache_policy="Per-process empty application PSO directory; shared runtime SPIR-V and uncontrolled driver cache; 16 warmup frames.",
        caveats=["Cooperative locks do not establish system-wide GPU exclusivity; clocks are unaltered.",
                 "512x512 bounded fixtures do not establish full-scene or end-to-end frame-rate performance.",
                 "Recipe capture proves per-case native PSO creation, not an independent per-dispatch binding receipt."])
    checks, baseline, ordinary, device, targets = [], None, None, None, {}
    for row in schedule():
        dispatch.snapshot_check(inputs, absent, inventory, "before-" + row["label"], checks)
        experiment.save(output / "SnapshotChecks.json", checks)
        destination = output / row["label"]
        destination.mkdir()
        environment = runtime_environment(arms[row["arm"]]["file"], destination)
        command = [str(executable), "--gtest_filter=RHIRendering.material_phase0_baseline",
            "--rhi-validation" if row["validation"] else "--rhi-no-validation", "--output-dir", str(destination)]
        experiment.save(destination / "Invocation.json", dict(command=command, timeout_seconds=args.timeout,
            environment={key: value for key, value in environment.items() if key.startswith(("METALLIC_", "VK_"))}))
        print(f"{row['label']}: validation={'on' if row['validation'] else 'off'}", flush=True)
        process = execute(command, environment, destination, args.timeout)
        # Save process success before interpreting any potentially incomplete evidence.
        item = dict(**row, execution_id=process["execution_id"], process=process)
        summary["runs"].append(item)
        experiment.save(output / "summary.json", summary)
        raw = experiment.read(destination / "MaterialBaseline.json")
        require(raw.get("validation") is row["validation"], "Unexpected runtime validation policy")
        arm_configuration = arms[row["arm"]]["configuration"]
        experiment.save(destination / "ComparableReport.json", comparable_report(raw, configuration, arm_configuration, row["arm"]))
        identity = runtime_identity(destination, arm_configuration, row["arm"], row["validation"])
        ordinary = identity["ordinary"] if ordinary is None else ordinary
        device = identity["device"] if device is None else device
        require(identity["ordinary"] == ordinary and identity["device"] == device, "Non-target pipeline or device changed")
        require(targets.setdefault(row["arm"], identity["targets"]) == identity["targets"], "Actual target shader changed within an arm")
        validate_target_pair(targets)
        baseline = baseline or destination / "ComparableReport.json"
        comparison = experiment.compare_readbacks(baseline, destination / "ComparableReport.json", configuration, **TOLERANCE)
        item.update(identity=identity, correctness=comparison)
        experiment.save(output / "summary.json", summary)
        require(all(case["qualified"] for case in comparison), "HDR/temporal/guide outputs exceeded the declared tolerance")
        if row["phase"] == "timing":
            item["timings"] = dispatch.verify_timing_rows(raw, arm_configuration)
        dispatch.snapshot_check(inputs, absent, inventory, "after-" + row["label"], checks)
        experiment.save(output / "SnapshotChecks.json", checks)
        experiment.save(output / "summary.json", summary)
    summary.update(correctness_qualified=True, timing_summary=analyze(output, summary), status="completed", decision="inconclusive")


def verify(output, current_inputs=False):
    output = output.resolve(strict=True)
    dispatch.validate_seal(output)
    summary = experiment.read(output / "summary.json")
    require(summary["schema_version"] == 1 and summary["experiment"] == "pathtrace-inline-root", "Wrong evidence schema")
    if summary["status"] == "completed":
        require(analyze(output, summary) == summary["timing_summary"], "Process-level analysis changed")
        inputs, absent = experiment.read(output / "Inputs.json"), experiment.read(output / "AbsentInputs.json")
        checks = experiment.read(output / "SnapshotChecks.json")
        require([item["label"] for item in checks] == [prefix + row["label"] for row in schedule() for prefix in ("before-", "after-")],
                "Missing input immutability checks")
        require(all(item["inputs"] == inputs and item["absent_inputs"] == absent and
                    item["gpu_inventory"] == summary["gpu_inventory"] for item in checks), "Input or device changed during execution")
        if current_inputs:
            experiment.check_snapshots(inputs)
            rendering.check_absences(absent)
            require(experiment.gpu_inventory() == summary["gpu_inventory"], "Current GPU inventory changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("run")
    for option in ("executable", "cases", "output-dir"):
        launch.add_argument("--" + option, type=Path, required=True)
    launch.add_argument("--timeout", type=float, default=1200, help="Maximum seconds per independent process")
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    check.add_argument("--current-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        summary = verify(args.output, args.current_inputs)
        print(json.dumps(dict(status=summary["status"], decision=summary.get("decision"),
            timing_summary=summary.get("timing_summary"), error=summary.get("error")), indent=2, allow_nan=False))
        return
    require(dispatch.positive_finite(args.timeout), "Timeout must be finite and positive")
    for name in experiment.LOCK_NAMES:
        require(not (ROOT / "build" / name).exists(), "Existing experiment lock; inspect its live owner")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(schema_version=1, experiment="pathtrace-inline-root", status="running", runs=[], decision="inconclusive")
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), output=str(output), experiment=summary["experiment"]), stream)
    failed = False
    try:
        experiment.save(output / "summary.json", summary)
        run(args, output, summary)
    except (Exception, KeyboardInterrupt) as error:
        failed = True
        summary.update(status="failed", error=str(error), failure_type=type(error).__name__)
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        experiment.save(output / "summary.json", summary)
        experiment.seal(output)
        lock.unlink()
    if not failed:
        verify(output)
    print(output / "summary.json")
    if failed:
        parser.exit(1, f"Experiment incomplete; all available evidence retained: {output}\n")


if __name__ == "__main__":
    main()
