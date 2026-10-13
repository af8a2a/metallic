"""Measure original General versus exact specialized compute artifacts.

Three independent-process AB/BA/AB pairs replay two-pass-cleaned SPIR-V. The
controllable driver cache is disabled, each application cache is empty, and only
the native creation/driver feedback windows are measured. Historical verification
replays the actual archived verifier and never substitutes current helper code.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import uuid

sys.dont_write_bytecode = True
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver
import ProfileComputeShaderFrontend as frontend
import ValidatePathTraceBackendRendering as rendering

ROOT = Path(__file__).resolve().parents[2]
GENERAL_ARMS = ("original-general", "backend-link-general")
NO_FIBER_ARM = "backend-link-no-fiber"
PASSES = ["--eliminate-dead-branches", "--simplify-instructions"]
METRICS = ("native_create_ms", "driver_ms", "cache_lock_wait_ms")
SCOPE = "One vkCreateComputePipelines native call per fresh process; native_create_ms and driver_ms only."
STATISTICAL_UNIT = "One independent process per arm/pair; three paired ratios, not within-process repetitions."
CAVEATS = ["Driver cache disabled is a requested controllable override, not proof of all compiler/OS state cold.",
           "native_cache_hit is Vulkan application-cache feedback, not internal driver-cache feedback.",
           "Device setup, recipe load, shader module creation, cache save, process wall and shader compilation are excluded.",
           "No dispatch or frame speedup or scene-wide NoFiber eligibility is established by this experiment."]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid numeric {name}")
    return value


def counter(value, expected):
    return type(value) is int and value == expected


def fnv(raw):
    return rendering.fnv(raw)


def blob_identity(path):
    identity = rendering.blob_identity(path)
    raw = Path(path).read_bytes()
    identity["content_hash"] = fnv(struct.pack("<Q", len(raw)) + raw)
    return identity


def pso_hash(recipe, raw):
    require(recipe["pipeline_type"] == "compute", "Compute recipe required")
    state = recipe["state"]
    require(not state.get("custom_descriptor_mappings"), "Unexpected custom mappings")
    entry = recipe["shaders"]["compute"]["entry_point"].encode("utf-8")
    content = fnv(struct.pack("<Q", len(raw)) + raw)
    data = struct.pack("<IIQ", 6, 0x4350534f, content)
    data += struct.pack("<Q", len(entry)) + entry
    data += struct.pack("<BBII", state["uses_bindless_heap"], state["indirect_bindable"],
                        state["bindless_user_push_data_size"], 0)
    if state["disable_optimization"] or state["compute_backend"]:
        data += struct.pack("<IBB", 0x45585031, state["disable_optimization"], state["compute_backend"])
    return f"{fnv(data):016x}"


def schedule(general_arm="original-general"):
    require(general_arm in GENERAL_ARMS, "Unsupported General control arm")
    arms = (general_arm, NO_FIBER_ARM)
    return [arms, tuple(reversed(arms)), arms]


def validate_seal(output):
    output = Path(output).resolve(strict=True)
    expected = experiment.read(output / "Manifest.json")
    require(isinstance(expected, dict), "Missing completed evidence seal")
    actual = {p.relative_to(output).as_posix(): experiment.digest(p) for p in output.rglob("*")
              if p.is_file() and p != output / "Manifest.json"}
    require(actual == expected and actual, "Sealed evidence inventory changed")


def preflight(args):
    """Read all authoritative inputs; scratch manifests never become evidence."""
    for name in ("preparation", "compile", "analysis", "reference", "template_recipe", "executable"):
        setattr(args, name, Path(getattr(args, name)).resolve(strict=True))
    schedule(args.general_arm)
    with tempfile.TemporaryDirectory(prefix="metallic-native-preflight-") as temporary:
        manifests, sources = rendering.prepare_artifacts(args, Path(temporary))
        selected_manifests = {arm: experiment.read(manifests[arm]) for arm in (args.general_arm, NO_FIBER_ARM)}
        # Generated scratch manifests have no runtime authority; stage emits its
        # own immutable recipes. The frozen source chain contains every earlier
        # sealed preparation/compiler/cleanup artifact instead.
        sources = [path for path in sources if not Path(path).is_relative_to(Path(temporary))]
    analysis = experiment.read(args.analysis / "summary.json")
    require(analysis["target_environment"] == "vulkan1.4" and analysis["scalar_block_layout"] is True,
            "Unexpected cleanup target/layout")
    analysis_inputs = experiment.read(args.analysis / "Inputs.json")
    experiment.check_snapshots(analysis_inputs["frozen_tools_helpers"])
    sources += [Path(path) for path in analysis_inputs["frozen_tools_helpers"]]
    template = experiment.read(args.template_recipe)
    require(template["schema"] == "metallic_pipeline_recipe" and counter(template["version"], 1), "Invalid template recipe")
    original = (args.template_recipe.parent / template["shaders"]["compute"]["spirv_file"]).resolve(strict=True)
    require(original.is_relative_to(args.template_recipe.parent), "Template bytecode escaped its capture directory")
    original_identity = blob_identity(original)
    require(pso_hash(template, original.read_bytes()) == template["expected_pso_hash"],
            "PSO ABI hash does not reproduce the original captured template")
    shader = template["shaders"]["compute"]
    require(original_identity["fnv1a64"] == shader["input_spirv_hash"] and original_identity["bytes"] == shader["spirv_bytes"],
            "Template input identity mismatch")
    reference = experiment.read(args.reference)["samples"][0]
    require(original_identity["fnv1a64"] == reference["spirv_fnv1a64"] and original_identity["bytes"] == reference["spirv_bytes"] and
            shader["module_name"] == reference["request"]["module"] and shader["source_entry_point"] == reference["request"]["entry"],
            "Captured template is not the frozen original workload")
    require(shader["entry_point"] == "main" and shader["debug_mode"] == "release" and
            shader["descriptor_heap_mode"] == "mapped" and shader["slang_optimization"] == "O1" and
            shader["spirv_postprocess"] == "none" and shader["source_identity_ambiguous"] is False,
            "Unexpected template source ABI/policy")
    state = template["state"]
    require(state["uses_bindless_heap"] is True and state["indirect_bindable"] is False and
            counter(state["bindless_user_push_data_size"], 12) and counter(state["compute_backend"], 0) and
            state["disable_optimization"] is False, "Unexpected fixed PSO ABI/state")
    preparation = experiment.read(args.preparation / "Preparation.json")
    require(preparation["status"] == "prepared", "Missing prepared source overlay")
    plans = {row["label"]: row for row in preparation["plans"]}
    selected = {}
    for arm, manifest in selected_manifests.items():
        artifact = Path(manifest["artifact"]["file"])
        process = experiment.read(artifact.parent / "cleanup.process.json")
        command = process["command"]
        require(process["status"] == "passed" and counter(process["exit_code"], 0) and
                len(command) == 8 and Path(command[0]).name == "spirv-opt.exe" and
                command[1:5] == ["--target-env=vulkan1.4", "--scalar-block-layout", *PASSES] and command[6] == "-o" and
                Path(command[5]).resolve() == Path(analysis_inputs["inputs"][arm]["path"]).resolve() and
                Path(command[7]).resolve() == artifact, "Artifact was not the exact two-pass output")
        identity = blob_identity(artifact)
        provider = plans[arm].get("provider")
        selected[arm] = dict(source=str(artifact), identity=identity, artifact_manifest=manifest, plan=plans[arm],
                             provider=provider, provider_sha256=experiment.digest(provider) if provider else None,
                             analysis=analysis["outputs"][arm])
    sources += [args.template_recipe, original, Path(__file__)]
    frozen = experiment.input_identity(args.executable, sources)
    return dict(schema_version=1, general_arm=args.general_arm, schedule=[list(row) for row in schedule(args.general_arm)],
                template=str(args.template_recipe), template_sha256=experiment.digest(args.template_recipe),
                template_original_file=str(original), template_original=original_identity, template_document=template,
                template_pso_hash_validated=True, reference=str(args.reference),
                preparation=str(args.preparation), compile=str(args.compile), analysis=str(args.analysis),
                source_manifests=selected_manifests[args.general_arm]["provenance"]["manifests"],
                postprocess="dead-branches", postprocess_tool_version=selected_manifests[args.general_arm]["artifact"]["tool_version"],
                inputs=selected, frozen_files=frozen, executable=str(args.executable))


def freeze_helpers(output):
    destination = output / "Verifier/Tools/Perf"
    destination.mkdir(parents=True)
    rows = []
    for source in sorted(Path(__file__).parent.glob("*.py")):
        target = destination / source.name
        shutil.copyfile(source, target)
        digest = experiment.digest(source)
        require(experiment.digest(target) == digest, "Actual verifier changed while archiving")
        rows.append(dict(source=str(source.resolve()), file=target.relative_to(output).as_posix(), sha256=digest))
    experiment.save(output / "FrozenHelpers.json", rows)
    return [Path(row["source"]) for row in rows]


def substituted_recipe(inputs, arm, blob_name, raw):
    recipe = copy.deepcopy(inputs["template_document"])
    item = inputs["inputs"][arm]
    identity = item["identity"]
    recipe["shaders"]["compute"].update(spirv_file=blob_name, spirv_bytes=identity["bytes"],
        input_spirv_hash=identity["fnv1a64"], module_name=item["artifact_manifest"]["artifact"]["source_tag"],
        source_identity_ambiguous=False, spirv_postprocess=inputs["postprocess"],
        spirv_postprocess_tool_version=inputs["postprocess_tool_version"])
    recipe["expected_pso_hash"] = pso_hash(recipe, raw)
    return recipe


def stage(output, inputs):
    recipes, provenance = output / "recipes", output / "provenance"
    recipes.mkdir()
    provenance.mkdir()
    archives = {}
    # Every frozen source/evidence artifact is copied. Live runtime/SDK binaries
    # remain SHA-pinned before and after execution; they are not needed to audit
    # reports using the archived source and SPIR-V.
    for index, (path, digest) in enumerate(inputs["frozen_files"].items()):
        source = Path(path)
        if source.suffix.lower() not in (".exe", ".dll", ".lib", ".pdb"):
            target = provenance / f"{index:05d}-{source.name}"
            shutil.copyfile(source, target)
            require(experiment.digest(target) == digest, "Provenance copy mismatch")
            archives[path] = dict(path=target.relative_to(output).as_posix(), sha256=digest)
    inputs["archived_frozen_files"] = archives
    for arm in (inputs["general_arm"], NO_FIBER_ARM):
        item = inputs["inputs"][arm]
        blob = recipes / f"{item['identity']['fnv1a64']}.spv"
        shutil.copyfile(item["source"], blob)
        recipe = substituted_recipe(inputs, arm, blob.name, blob.read_bytes())
        path = recipes / f"{arm}.recipe.json"
        experiment.save(path, recipe)
        item.update(recipe=path.relative_to(output).as_posix(), recipe_sha256=experiment.digest(path),
                    staged_spirv=blob.relative_to(output).as_posix(), expected_pso_hash=recipe["expected_pso_hash"])
    experiment.save(output / "Inputs.json", inputs)


def native_command(executable, recipe, destination):
    return [str(executable), "--recipe-file", str(recipe), "--iterations", "1", "--mode", "serial", "--workers", "1",
            "--pipeline-cache", "empty", "--shader-cache", "reuse", "--descriptor-mode", "mapped", "--debug-mode", "disabled",
            "--cache-dir", str(destination / "cache"), "--output-dir", str(destination)]


def native_environment():
    environment = experiment.clean_environment()
    environment.update(METALLIC_VK_INTERNAL_PIPELINE_CACHE="disabled", METALLIC_VK_COMPUTE_BACKEND="pipeline",
                       METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION="0")
    return environment


def execute(command, environment, destination, timeout, execution_id):
    """Keep actual child telemetry and controls even when it fails natively."""
    def preserve(process):
        process.update(execution_id=execution_id, ended_utc=datetime.now(timezone.utc).isoformat(),
                       environment={key: value for key, value in environment.items() if key.startswith("METALLIC_")})
        experiment.save(destination / "process.json", process)
    try:
        process, report = driver.run_process(command, environment, destination, timeout)
    except Exception:
        sidecar = destination / "process.json"
        if sidecar.is_file():
            preserve(experiment.read(sidecar))
        raise
    preserve(process)
    return process, report


def validate_statistics(statistics):
    require(isinstance(statistics, dict) and statistics, "Missing SPIR-V statistics")
    for name, value in statistics.items():
        if isinstance(value, list):
            require(value and all(type(count) is int and count >= 0 for count in value), f"Invalid SPIR-V statistic {name}")
        else:
            require(type(value) is int and value >= 0, f"Invalid SPIR-V statistic {name}")


def validate_report(report, recipe, identity):
    sample = driver.single_sample(report, native=True)
    config = report["configuration"]
    expected = dict(compile_only=False, compute_backend="pipeline", debug_mode="disabled", descriptor_mode="mapped",
                    disable_driver_optimization=False, iterations=1, mode="serial", pipeline_cache="empty", pipeline_replicas=1,
                    workers=1, shader_cache="reuse", validation=False, native_binary_cache_directory="", application_pipeline_cache_used=True)
    require(all(type(config.get(key)) is type(value) and config[key] == value for key, value in expected.items()),
            "Measured settings do not match fixed replay")
    require(report["environment"].get(driver.CACHE_ENVIRONMENT) == "disabled" and
            report["environment"].get("METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION") == "0", "Driver controls mismatch")
    # Bench does not currently enumerate this variable in its environment report.
    # Actual backend is confirmed in configuration and both per-call records;
    # the launcher also freezes its full effective Metallic environment.
    require("METALLIC_VK_COMPUTE_BACKEND" not in report["environment"] or
            report["environment"]["METALLIC_VK_COMPUTE_BACKEND"] == "pipeline", "Unexpected backend override")
    for key in ("METALLIC_VK_PIPELINE_BINARY", "METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY", "METALLIC_VK_PIPELINE_STATISTICS",
                "METALLIC_SPIRV_POSTPROCESS", "METALLIC_SLANG_OPTIMIZATION"):
        require(not report["environment"].get(key), f"Unexpected diagnostic/import/compiler override {key}")
    require(report.get("recipe_source_device_matches") is True and report["device"]["validation_active"] is False,
            "Captured hardware mismatch or active validation")
    for key in ("vendor_id", "device_id", "api_version", "driver_version", "driver_name", "pipeline_cache_uuid"):
        require(type(report["device"][key]) is type(recipe["source_device"][key]) and
                report["device"][key] == recipe["source_device"][key], f"Captured device {key} mismatch")
    require(report["device"]["name"] == recipe["source_device"]["device_name"], "Device name mismatch")
    require(report["build_configuration"] == "Release", "Native timing requires the fixed Release build")
    require(sample.get("shader_source") == "recipe_spirv" and sample.get("pipeline_type") == "compute" and
            sample.get("effective_slang_optimization") == "O1" and sample.get("effective_spirv_postprocess") == "dead-branches",
            "Source replay/policy mismatch")
    require(not sample.get("compilation_phases", {}).get("compiler_executed") and sample.get("compile_wall_ms", 0) == 0 and
            sample.get("shader_cache_hit", False) is False, "Compiler/cache path unexpectedly ran")
    stages = sample["shader_stages"]
    require(len(stages) == 1, "Expected exactly one compute stage")
    shader, stage = recipe["shaders"]["compute"], stages[0]
    expected_stage = dict(stage="compute", module=shader["module_name"], entry="main", source_entry=shader["source_entry_point"],
        debug_mode="release", descriptor_mode="mapped", slang_optimization="O1", spirv_postprocess="dead-branches",
        spirv_postprocess_tool_version=shader["spirv_postprocess_tool_version"], source_identity_ambiguous=False,
        spirv_bytes=identity["bytes"], spirv_fnv1a64=identity["fnv1a64"], device_spirv_bytes=identity["bytes"],
        device_spirv_fnv1a64=identity["fnv1a64"], device_shader_content_hash=identity["content_hash"])
    require(all(type(stage.get(key)) is type(value) and stage[key] == value for key, value in expected_stage.items()),
            "Input/final-device SPIR-V or ABI provenance mismatch")
    validate_statistics(stage["spirv_statistics"])
    validate_statistics(stage["device_spirv_statistics"])
    validate_statistics(sample["spirv_statistics"])
    validate_statistics(sample["device_spirv_statistics"])
    require(stage["device_spirv_statistics"] == stage["spirv_statistics"] == sample["spirv_statistics"] ==
            sample["device_spirv_statistics"], "Device SPIR-V statistics changed")
    for key in ("spirv_bytes", "spirv_fnv1a64", "device_spirv_bytes", "device_spirv_fnv1a64", "device_shader_content_hash"):
        require(type(sample.get(key)) is type(expected_stage[key]) and sample[key] == expected_stage[key], f"Top-level identity mismatch {key}")
    pipe = sample["pipeline"]
    expected_pipe = dict(feedback_valid=True, native_cache_hit=False, metadata_hit=False, native_object_reused=False,
        recipe_hash_matches=True, recipe_state_overridden=False, pso_hash=int(recipe["expected_pso_hash"], 16),
        expected_pso_hash=int(recipe["expected_pso_hash"], 16), compute_backend="pipeline", native_binary_cache_hit=False,
        native_binary_driver_rejected=False, native_binary_persisted=False, native_binary_data_bytes=0)
    require(all(type(pipe.get(key)) is type(value) and pipe[key] == value for key, value in expected_pipe.items()),
            "Native-miss recipe creation qualification failed")
    for key in ("native_binary_import_ms", "native_binary_export_ms"):
        require(number(pipe.get(key), key) == 0, "Unexpected native binary timing")
    values = {key: number(pipe.get(key), key, positive=key != "cache_lock_wait_ms") for key in METRICS}
    require(values["driver_ms"] <= values["native_create_ms"] + .001, "Driver time exceeds native window")
    for key in (*METRICS, "compute_backend", "feedback_valid", "native_cache_hit", "native_object_reused",
                "native_binary_cache_hit", "native_binary_data_bytes", "native_binary_driver_rejected",
                "native_binary_import_ms", "native_binary_export_ms", "native_binary_persisted"):
        require(type(sample["pipeline_attempt"].get(key)) is type(pipe[key]) and sample["pipeline_attempt"][key] == pipe[key],
                "Returned per-call metrics disagree with pipeline metrics")
    for key, expected_value in (("pipelines_created", 1), ("pipelines_skipped", 0), ("failures", 0), ("validation_errors", 0)):
        require(counter(report.get(key), expected_value), f"Invalid replay counter {key}")
    require(len(report["pipeline_batches"]) == 1, "Expected one native creation batch")
    batch = report["pipeline_batches"][0]
    for key, expected_value in (("pipeline_count", 1), ("successful_pipelines", 1), ("actual_workers", 1), ("requested_workers", 1),
        ("native_hits", 0), ("native_misses", 1), ("metadata_hits", 0), ("metadata_misses", 1), ("unknown_feedback", 0),
        ("valid_driver_feedback_count", 1), ("validation_errors", 0), ("cache_count", 1)):
        require(counter(batch.get(key), expected_value), f"Invalid batch counter {key}")
    require(batch["mode"] == "serial" and batch["sample_indices"] == [0], "Unexpected batch topology")
    for key, metric in (("attempted_native_create_sum_ms", "native_create_ms"), ("successful_native_create_sum_ms", "native_create_ms"),
        ("attempted_cache_lock_wait_sum_ms", "cache_lock_wait_ms"), ("successful_cache_lock_wait_sum_ms", "cache_lock_wait_ms"),
        ("driver_duration_sum_ms", "driver_ms")):
        require(number(batch.get(key), key) == values[metric], "Batch metrics differ from returned per-call metrics")
    identity_report = dict(device=report["device"], build_configuration=report["build_configuration"], slang_build=report["slang_build"],
                           shader_target=report["shader_target"], shader_stage=stage)
    return values, identity_report


def aggregate(records, general_arm="original-general"):
    orders = schedule(general_arm)
    require(len(records) == 6 and [(row["pair"], row["label"]) for row in records] ==
            [(pair, arm) for pair, order in enumerate(orders) for arm in order], "Missing, duplicate or reordered native pair")
    for row in records:
        require(type(row["pair"]) is int, "Invalid pair counter")
        for metric in METRICS:
            number(row[metric], metric, positive=metric != "cache_lock_wait_ms")
    arms = {arm: {metric: driver.timings([row[metric] for row in records if row["label"] == arm]) for metric in METRICS}
            for arm in (general_arm, NO_FIBER_ARM)}
    paired = []
    for pair, order in enumerate(orders):
        values = {row["label"]: row for row in records if row["pair"] == pair}
        paired.append(dict(pair=pair, order=list(order), native_no_fiber_over_general=values[NO_FIBER_ARM]["native_create_ms"] /
            values[general_arm]["native_create_ms"], driver_no_fiber_over_general=values[NO_FIBER_ARM]["driver_ms"] / values[general_arm]["driver_ms"]))
    return dict(general_arm=general_arm, arms=arms, paired_ratios=paired,
                median_native_no_fiber_over_general=statistics.median(row["native_no_fiber_over_general"] for row in paired),
                median_driver_no_fiber_over_general=statistics.median(row["driver_no_fiber_over_general"] for row in paired))


def validate_process(process, execution_id):
    require(isinstance(execution_id, str) and len(execution_id) == 32 and all(c in "0123456789abcdef" for c in execution_id) and
            process["execution_id"] == execution_id and type(process["pid"]) is int and process["pid"] > 0,
            "Missing independent process identity")
    require(counter(process["exit_code"], 0) and process["timed_out"] is False and process["native_crash"] is False,
            "Invalid or failed process")
    started = datetime.fromisoformat(process["started_utc"])
    ended = datetime.fromisoformat(process["ended_utc"])
    require(started.tzinfo is not None and ended.tzinfo is not None and ended >= started, "Invalid process timestamp interval")
    number(process["process_wall_ms"], "process_wall_ms", positive=True)
    return started.timestamp(), ended.timestamp()


def snapshot_check(frozen, label, checks):
    experiment.check_snapshots(frozen)
    checks.append(dict(label=label, inputs=experiment.snapshots(frozen)))


def run(args):
    output = Path(args.output_dir).resolve()
    require(not output.exists(), "Refusing to overwrite existing evidence")
    number(args.timeout, "timeout", positive=True)
    with frontend.experiment_lock():
        inputs = preflight(args)
        output.mkdir(parents=True)
        summary = dict(schema_version=1, status="running", pairs=3, scope=SCOPE, caveats=CAVEATS,
                       statistical_unit=STATISTICAL_UNIT, general_arm=args.general_arm, runs=[])
        checks = []
        try:
            helpers = freeze_helpers(output)
            inputs["frozen_files"].update(experiment.snapshots(helpers))
            stage(output, inputs)
            frozen = dict(inputs["frozen_files"])
            frozen.update(experiment.snapshots([output / "Inputs.json", output / "FrozenHelpers.json",
                *[p for p in (output / "recipes").iterdir()], *[p for p in (output / "Verifier").rglob("*") if p.is_file()]]))
            experiment.save(output / "MeasuredInputs.json", frozen)
            identities, common = {}, None
            environment = native_environment()
            index = 0
            for pair, order in enumerate(schedule(args.general_arm)):
                for arm in order:
                    label = f"pair-{pair:02d}-{index % 2:02d}-{arm}"
                    snapshot_check(frozen, "before-" + label, checks)
                    item = inputs["inputs"][arm]
                    recipe_path, directory = output / item["recipe"], output / label
                    print(f"[{index + 1}/6] {arm} pair={pair}; exact cleaned recipe", flush=True)
                    execution_id = uuid.uuid4().hex
                    process, report = execute(native_command(args.executable, recipe_path, directory), environment, directory, args.timeout, execution_id)
                    validate_process(process, execution_id)
                    snapshot_check(frozen, "after-" + label, checks)
                    values, identity = validate_report(report, experiment.read(recipe_path), item["identity"])
                    invariant = {key: value for key, value in identity.items() if key != "shader_stage"}
                    common = common or invariant
                    require(common == invariant, "Device/build changed between arms")
                    require(arm not in identities or identities[arm] == identity, "Repeated final shader identity changed")
                    identities[arm] = identity
                    summary["runs"].append(dict(pair=pair, label=arm, directory=label, execution_id=execution_id, **values))
                    experiment.save(output / "summary.json", summary)
                    print(f"native={values['native_create_ms']:.3f}ms driver={values['driver_ms']:.3f}ms", flush=True)
                    index += 1
            summary.update(status="passed", identities=identities, aggregation=aggregate(summary["runs"], args.general_arm),
                           frozen_inputs_checked_after_all_processes=True)
        except Exception as error:
            summary.update(status="failed", error=str(error))
            raise
        finally:
            experiment.save(output / "SnapshotChecks.json", checks)
            experiment.save(output / "summary.json", summary)
            experiment.seal(output)
    return verify(output)


def archived_file(output, inputs, source):
    record = inputs["archived_frozen_files"][str(source)]
    path = (output / record["path"]).resolve(strict=True)
    require(path.is_relative_to((output / "provenance").resolve()) and experiment.digest(path) == record["sha256"] ==
            inputs["frozen_files"][str(source)], "Archived provenance mismatch")
    return path


def verify_source_manifest(output, inputs, source):
    manifest = experiment.read(archived_file(output, inputs, source))
    require(manifest["schema_version"] == 1 and isinstance(manifest["artifacts"], list) and manifest["artifacts"],
            "Invalid archived source manifest")
    seen = set()
    for row in manifest["artifacts"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in seen, "Invalid archived manifest path")
        seen.add(str(relative))
        original = Path(source).parent / relative
        require(experiment.digest(archived_file(output, inputs, original)) == row["sha256"], "Archived source inventory mismatch")


def verify_records(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    validate_seal(output)
    inputs, summary = experiment.read(output / "Inputs.json"), experiment.read(output / "summary.json")
    general = inputs["general_arm"]
    orders = schedule(general)
    require(summary["status"] == "passed" and counter(summary["pairs"], 3) and summary["scope"] == SCOPE and
            summary["caveats"] == CAVEATS and summary["statistical_unit"] == STATISTICAL_UNIT and summary["general_arm"] == general and
            inputs["schedule"] == [list(row) for row in orders], "Invalid completed native experiment")
    for original in inputs["archived_frozen_files"]:
        archived_file(output, inputs, original)
    helpers = experiment.read(output / "FrozenHelpers.json")
    require(any(Path(row["file"]).name == Path(__file__).name for row in helpers), "Missing actual archived verifier")
    for row in helpers:
        source = (output / row["file"]).resolve(strict=True)
        require(source.is_relative_to((output / "Verifier").resolve()) and experiment.digest(source) == row["sha256"] ==
                inputs["frozen_files"][row["source"]], "Frozen helper mismatch")
    for row in inputs["source_manifests"]:
        require(inputs["frozen_files"][row["file"]] == row["sha256"], "Frozen source manifest differs")
        verify_source_manifest(output, inputs, row["file"])
    template = experiment.read(archived_file(output, inputs, inputs["template"]))
    require(template == inputs["template_document"] and experiment.digest(archived_file(output, inputs, inputs["template"])) ==
            inputs["template_sha256"] and inputs["template_pso_hash_validated"] is True, "Template provenance mismatch")
    original = archived_file(output, inputs, inputs["template_original_file"])
    require(blob_identity(original) == inputs["template_original"] and pso_hash(template, original.read_bytes()) == template["expected_pso_hash"],
            "Original template hash validation failed")
    frozen = experiment.read(output / "MeasuredInputs.json")
    checks = experiment.read(output / "SnapshotChecks.json")
    expected_labels = [f"pair-{pair:02d}-{position:02d}-{arm}" for pair, order in enumerate(orders) for position, arm in enumerate(order)]
    require([row["label"] for row in checks] == [prefix + label for label in expected_labels for prefix in ("before-", "after-")] and
            all(row["inputs"] == frozen for row in checks), "Missing or changed before/after frozen-input checks")
    require(all(frozen.get(path) == digest for path, digest in inputs["frozen_files"].items()), "Measured input snapshot differs from preflight")
    if current_inputs:
        experiment.check_snapshots(frozen)
    require(len(summary["runs"]) == 6, "Incomplete native sample count")
    identities, common, records, executions, previous_end = {}, None, [], set(), None
    for index, (pair, arm) in enumerate((pair, arm) for pair, order in enumerate(orders) for arm in order):
        row = summary["runs"][index]
        directory = output / expected_labels[index]
        require(counter(row["pair"], pair) and row["label"] == arm and row["directory"] == directory.name, "Pair order/directory mismatch")
        item = inputs["inputs"][arm]
        require(item["artifact_manifest"]["artifact_id"] == arm and item["artifact_manifest"]["artifact"]["source_tag"] ==
                f"Artifacts/PathBackend/{arm}/TwoPassCleaned", "Artifact source identity is not the selected source arm")
        recipe_path, blob = output / item["recipe"], output / item["staged_spirv"]
        require(recipe_path.resolve().is_relative_to((output / "recipes").resolve()) and blob.resolve().is_relative_to((output / "recipes").resolve()),
                "Staged replay escaped its evidence directory")
        recipe = experiment.read(recipe_path)
        require(blob_identity(blob) == item["identity"] and experiment.digest(recipe_path) == item["recipe_sha256"] and
                recipe == substituted_recipe(inputs, arm, blob.name, blob.read_bytes()) and recipe["expected_pso_hash"] == item["expected_pso_hash"],
                "Recipe is not exactly the captured template with documented artifact substitution")
        require(blob_identity(archived_file(output, inputs, item["source"])) == item["identity"] and
                experiment.read(archived_file(output, inputs, str(Path(item["source"]).parent / "Analysis.json"))) == item["analysis"],
                "Analysis-to-cleaned-artifact provenance mismatch")
        if item["provider"]:
            require(experiment.digest(archived_file(output, inputs, item["provider"])) == item["provider_sha256"], "Provider provenance mismatch")
        process = experiment.read(directory / "process.json")
        start, end = validate_process(process, row["execution_id"])
        require(row["execution_id"] not in executions and (previous_end is None or start >= previous_end),
                "Duplicate or overlapping native processes")
        executions.add(row["execution_id"])
        previous_end = end
        require(process["command"] == native_command(inputs["executable"], recipe_path, directory) and process["environment"] ==
            dict(METALLIC_SHADER_DEBUG_MODE="disabled", METALLIC_VK_INTERNAL_PIPELINE_CACHE="disabled", METALLIC_VK_COMPUTE_BACKEND="pipeline",
                 METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION="0"), "Process command/control mismatch")
        report = experiment.read(directory / "report.json")
        require(Path(report["configuration"]["recipe_file"]).resolve() == recipe_path.resolve() and
                Path(report["configuration"]["cache_directory"]).resolve() == (directory / "cache").resolve(), "Shared cache or wrong recipe")
        values, identity = validate_report(report, recipe, item["identity"])
        invariant = {key: value for key, value in identity.items() if key != "shader_stage"}
        common = common or invariant
        require(common == invariant and (arm not in identities or identities[arm] == identity), "Device/build/final shader identity changed")
        identities[arm] = identity
        record = dict(pair=pair, label=arm, directory=directory.name, execution_id=row["execution_id"], **values)
        require(row == record, "Summary sample differs from raw native report")
        records.append(record)
    require(summary["aggregation"] == aggregate(records, general) and summary["identities"] == identities and
            summary["frozen_inputs_checked_after_all_processes"] is True, "Summary aggregation or qualification changed")
    return summary


def verify(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    validate_seal(output)
    source = output / "Verifier/Tools/Perf/MeasurePathTraceBackendNative.py"
    require(source.is_file(), "Missing archived actual native verifier")
    command = [sys.executable, "-B", str(source), "_verify-frozen", str(output)]
    if current_inputs:
        command.append("--current-inputs")
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, "Archived native verification failed: " + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check"):
        child = commands.add_parser(name, help="Measure native creation" if name == "run" else "Read-only preflight; launches nothing")
        for option in ("preparation", "compile", "analysis", "reference", "template-recipe", "executable"):
            child.add_argument("--" + option, type=Path, required=True)
        child.add_argument("--general-arm", choices=GENERAL_ARMS, default="original-general")
        if name == "run":
            child.add_argument("--output-dir", type=Path, required=True)
            child.add_argument("--timeout", type=float, default=600)
    for name in ("verify", "_verify-frozen"):
        child = commands.add_parser(name, help="Read-only archived raw-report verification; launches no benchmark")
        child.add_argument("output", type=Path)
        child.add_argument("--current-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        summary = run(args)
    elif args.command == "check":
        inputs = preflight(args)
        print(json.dumps(dict(status="ready", general_arm=args.general_arm, schedule=inputs["schedule"],
            template_pso_hash_validated=inputs["template_pso_hash_validated"], inputs={arm: row["identity"] for arm, row in inputs["inputs"].items()}), indent=2, allow_nan=False))
        return
    elif args.command == "_verify-frozen":
        print(json.dumps(verify_records(args.output, current_inputs=args.current_inputs), indent=2, allow_nan=False))
        return
    else:
        summary = verify(args.output, current_inputs=args.current_inputs)
    print(json.dumps(dict(status=summary["status"], processes=len(summary["runs"]), pairs=summary["pairs"],
        scope=summary["scope"], aggregation=summary["aggregation"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
