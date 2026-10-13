"""Measure compute compilation separately from production RenderGraph GPU work.

compile: one fixed catalog request, prefilled SPIR-V, independent processes with
the controllable driver cache disabled; optionally compare .pso/native imports.
dispatch: fixed MaterialBaseline scene/resources/settings, GPU timestamps and
HDR/guide readbacks outside the timing window. No empty-resource dispatches.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import RunShaderPipelineDriver as driver


ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATIONS = ("default", "O0", "O1", "O2", "O3")
BACKENDS = ("pipeline", "shader-object", "pipeline-binary")
LOCK_NAMES = ("shader-experiment.lock", "openpbr-benchmark.lock", "material-phase0.lock",
              "compute-pipeline-experiment.lock")


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def snapshots(paths):
    return {str(Path(p).resolve()): digest(p) for p in sorted(set(map(Path, paths)))}


def check_snapshots(identity):
    for path, expected in identity.items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError(f"Experiment input changed: {path}")


def input_identity(executable, extra=(), *, production=False):
    # Record the actual installed source and executable; never edit shader files.
    sources = [ROOT / "Source/Runtime/Render/Core" / name for name in
               ("SlangCompiler.h", "SlangCompiler.cpp", "ShaderRegistryCompiler.cpp", "ShaderRegistry.cpp")]
    sources += [ROOT / "Source/Runtime/Render/GAPI/RHI.h",
                ROOT / "Source/Runtime/Render/GAPI/PipelineStateHash.cpp",
                ROOT / "Source/Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp",
                ROOT / "Source/Tools/ShaderPipelineBench.cpp"]
    if production:
        sources += [p for p in (ROOT / "Shaders").rglob("*")
                    if p.is_file() and p.suffix in (".slang", ".hlsli", ".hlsl")]
        sources += [ROOT / "tests/rhi/MaterialBaselineTests.cpp"]
    return snapshots([executable, *executable.parent.glob("*.dll"), Path(__file__),
                      ROOT / "Tools/Perf/RunShaderPipelineDriver.py", *sources, *extra])


def clean_environment():
    environment = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
    if environment.get("VK_INSTANCE_LAYERS"):
        raise ValueError("Remove injected VK_INSTANCE_LAYERS before ordinary timing")
    environment["METALLIC_SHADER_DEBUG_MODE"] = "disabled"
    return environment


def arms(optimizations, backends, optimization_policy):
    flags = (False, True) if optimization_policy == "both" else (optimization_policy == "disabled",)
    result = []
    for optimization in optimizations:
        for backend in backends:
            for disabled in flags:
                if backend == "shader-object" and disabled:
                    continue  # Shader objects have no disable-pipeline-optimization flag.
                result.append(dict(optimization=optimization, backend=backend,
                                   disable_driver_optimization=disabled,
                                   name=f"{optimization}-{backend}-{'disabled' if disabled else 'normal'}"))
    if not result:
        raise ValueError("No valid conditions; shader-object cannot disable pipeline optimization")
    if len({arm["name"] for arm in result}) != len(result):
        raise ValueError("Duplicate experiment conditions")
    return result


def balanced_order(conditions, block):
    # Rotate starting arms and reverse alternate blocks. Preserve the schedule in
    # evidence; small repeat counts do not eliminate all clock/process-order drift.
    shift = block % len(conditions)
    ordered = conditions[shift:] + conditions[:shift]
    return list(reversed(ordered)) if block % 2 else ordered


def environment_for(arm, base, binary_directory=None, *, internal_cache="disabled"):
    environment = dict(base)
    environment.update(METALLIC_SLANG_OPTIMIZATION=arm["optimization"],
                       METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION="1" if arm["disable_driver_optimization"] else "0",
                       METALLIC_VK_COMPUTE_BACKEND=arm["backend"],
                       METALLIC_VK_INTERNAL_PIPELINE_CACHE=internal_cache)
    if arm["backend"] == "pipeline-binary":
        environment["METALLIC_VK_PIPELINE_BINARY"] = "1"
    if binary_directory is not None:
        environment["METALLIC_VK_NATIVE_BINARY_CACHE_DIR"] = str(binary_directory)
    return environment


def shader_identity(report, sample):
    identity = driver.shader_identity(report, sample)
    # Native creation controls intentionally differ across arms; bytecode and
    # front-end policy remain the shared identity for each optimization level.
    identity.pop("compute_backend", None)
    identity.pop("disable_driver_optimization", None)
    identity["slang_optimization"] = report["configuration"]["slang_optimization"]
    identity["effective_slang_optimization"] = sample["effective_slang_optimization"]
    identity["spirv_statistics"] = sample["spirv_statistics"]
    return copy.deepcopy(identity)


def device_shader_identity(sample):
    return {field: copy.deepcopy(sample[field]) for field in
            ("device_spirv_bytes", "device_spirv_fnv1a64", "device_spirv_statistics", "device_shader_content_hash")}


def verify_native(report, expected_shader, arm, *, imported=False, application_cache=False, device=None):
    sample = driver.single_sample(report, native=True)
    if shader_identity(report, sample) != expected_shader:
        raise ValueError("SPIR-V or compiler configuration changed after prefill")
    if not sample.get("shader_cache_hit"):
        raise ValueError("Measured process did not load prefilled SPIR-V")
    configuration = report["configuration"]
    expected_cache = "reuse" if application_cache else "empty"
    if configuration["pipeline_cache"] != expected_cache or configuration["compile_only"]:
        raise ValueError("Native experiment application cache condition or native creation differs")
    if configuration["slang_optimization"] != arm["optimization"]:
        raise ValueError("Wrong requested Slang optimization")
    if configuration["compute_backend"] != arm["backend"] or \
       configuration["disable_driver_optimization"] != arm["disable_driver_optimization"]:
        raise ValueError("Wrong native creation condition")
    if report["environment"].get(driver.CACHE_ENVIRONMENT) != "disabled":
        raise ValueError("Native source/import experiments must disable the controllable driver cache")
    if device is not None and report.get("device") != device:
        raise ValueError("Device or driver configuration changed")
    pipeline = sample["pipeline"]
    wall = sample.get("pipeline_wall_ms")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Invalid total pipeline creation wall time")
    if pipeline["compute_backend"] != arm["backend"]:
        raise ValueError("Unexpected actual compute backend")
    for metric in ("native_create_ms", "cache_lock_wait_ms"):
        value = pipeline.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {metric}")
    if pipeline.get("driver_ms") is not None and (not math.isfinite(pipeline["driver_ms"]) or pipeline["driver_ms"] < 0):
        raise ValueError("Invalid driver_ms")
    if pipeline.get("native_object_reused", False):
        raise ValueError("Native object reuse bypassed driver creation")
    if pipeline.get("native_binary_cache_hit", False) != (imported and not application_cache):
        raise ValueError("Native binary import did not match the requested experiment arm")
    if pipeline.get("native_binary_driver_rejected", False):
        raise ValueError("Driver rejected a cached native binary; fallback is not an import measurement")
    if application_cache:
        if arm["backend"] != "pipeline":
            raise ValueError("Application VkPipelineCache requires the ordinary pipeline backend")
        if imported:
            if report.get("pipeline_cache_load_status") != 1 or not pipeline.get("metadata_hit") or \
               not pipeline.get("feedback_valid") or pipeline.get("native_cache_hit") is not True:
                raise ValueError("Application pipeline cache import requires Loaded metadata and a valid native cache hit")
        elif report.get("pipeline_cache_load_status") != 0:
            raise ValueError("Application pipeline cache prime must start with a missing .pso file")
    return sample


def creation_costs(report, sample):
    """Include file/cache preparation once, where this single request performed it."""
    preparation = sample.get("pipeline_cache_preparation_wall_ms")
    total = sample.get("pipeline_load_and_create_wall_ms")
    values = (preparation, total, report.get("pipeline_cache_preparation_wall_ms"))
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Missing or invalid pipeline cache preparation/load timing; rebuild the benchmark")
    if not math.isclose(total, preparation + sample["pipeline_wall_ms"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Pipeline load-and-create timing does not include cache preparation exactly once")
    if not math.isclose(preparation, report["pipeline_cache_preparation_wall_ms"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Single-request pipeline cache preparation was not attributed to its request")
    return dict(pipeline_cache_preparation_wall_ms=preparation, pipeline_load_and_create_wall_ms=total)


def import_modes(arm, native_binary_cache, application_pipeline_cache):
    enabled = application_pipeline_cache if arm["backend"] == "pipeline" else native_binary_cache
    return (False, True) if enabled else (False,)


def artifact_snapshot(path):
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("Application pipeline cache prime produced no .pso artifact")
    return snapshots([path])


def dispatch_binary_directory(output, arm, enabled):
    # An ordinary Pipeline reference uses VkPipelineCache; only experimental
    # backends accept the separate native binary cache directory.
    return output / "native-binaries" / arm["name"] if enabled and arm["backend"] != "pipeline" else None


def validate_dispatch_binary_backends(backends, enabled):
    if enabled and not any(backend != "pipeline" for backend in backends):
        raise ValueError("Native binary cache requires at least one shader-object or pipeline-binary backend")


def binary_snapshot(directory):
    identity = snapshots(p for p in directory.rglob("*") if p.is_file())
    if not identity:
        raise ValueError("Native binary extraction produced no file")
    return identity


def compile_experiment(args, output, summary):
    executable = args.executable.resolve(strict=True)
    conditions = arms(args.optimizations, args.backends, args.driver_optimization)
    application_pipeline_cache = getattr(args, "application_pipeline_cache", False)
    if args.native_binary_cache and "pipeline" in args.backends and not application_pipeline_cache:
        raise ValueError("Native binary import requires shader-object or pipeline-binary; select --backends explicitly")
    if application_pipeline_cache and "pipeline" not in args.backends:
        raise ValueError("--application-pipeline-cache requires a pipeline backend arm")
    inputs = input_identity(executable)
    save(output / "Inputs.json", inputs)
    base = clean_environment()
    cache = output / "shader-cache"
    common = [str(executable), "--request-index", str(args.request_index), "--iterations", "1",
              "--shader-cache", "reuse", "--pipeline-cache", "empty", "--cache-dir", str(cache),
              "--descriptor-mode", args.descriptor_mode, "--debug-mode", "disabled"]
    summary.update(conditions=conditions, pairs=args.pairs, validation=args.validation,
                   native_binary_cache=args.native_binary_cache,
                   application_pipeline_cache=application_pipeline_cache,
                   scope="One native compute creation per independent process. pipeline_load_and_create_wall_ms includes application cache file/preparation plus pipeline_wall_ms; native binary file preparation is already inside pipeline_wall_ms. Device, shader loading/module preparation, cache export/save, uploads and dispatch excluded.",
                   caveats=["Disabled driver cache does not guarantee all compiler/OS state is cold.",
                            "native_cache_hit is application VkPipelineCache feedback, not an internal driver-cache hit.",
                            "Creation speed alone does not establish GPU execution speed or numerical correctness."])
    if args.validation:
        common.append("--validation")
    if getattr(args, "slang_perf_diagnostics", False):
        common.append("--slang-perf-diagnostics")
    shaders = {}
    for optimization in args.optimizations:
        check_snapshots(inputs)
        destination = output / f"prefill-{optimization}"
        command = common + ["--slang-optimization", optimization, "--compile-only", "--output-dir", str(destination)]
        # Prefill uses the same descriptor ABI/debug policy and no native creation.
        prefill_arm = dict(conditions[0], optimization=optimization)
        process, report = driver.run_process(command, environment_for(prefill_arm, base), destination, args.timeout)
        sample = driver.single_sample(report, native=False)
        if sample.get("execution_model") != 5:
            raise ValueError("Select a compute request from the current --list catalog")
        shaders[optimization] = shader_identity(report, sample)
        check_snapshots(inputs)
        inputs.update(snapshots(sample["dependencies"]))
        save(output / "Inputs.json", inputs)
        summary["excluded_runs"].append(dict(condition="SPIR-V prefill", optimization=optimization,
                                               process=process, report=str(destination / "report.json"),
                                               compile_wall_ms=sample.get("compile_wall_ms"),
                                               compilation_phases=sample.get("compilation_phases")))
        save(output / "summary.json", summary)
    summary["shaders"] = shaders
    device_shaders = {}
    summary["device_shaders"] = device_shaders
    binary_identities = {}
    application_identities = {}
    device = None
    for arm in conditions:
        if len(import_modes(arm, args.native_binary_cache, application_pipeline_cache)) > 1:
            check_snapshots(inputs)
            application = arm["backend"] == "pipeline"
            directory = None if application else output / "native-binaries" / arm["name"]
            cache_file = output / "application-pipeline-caches" / f"{arm['name']}.pso" if application else None
            if application:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
            destination = output / f"{'application-cache' if application else 'binary'}-prime-{arm['name']}"
            command = native_command(common, arm, destination, directory, cache_file)
            process, report = driver.run_process(command, environment_for(arm, base, directory), destination, args.timeout)
            sample = verify_native(report, shaders[arm["optimization"]], arm, application_cache=application, device=device)
            device = report["device"]
            costs = creation_costs(report, sample)
            identity = device_shader_identity(sample)
            previous = device_shaders.setdefault(arm["optimization"], identity)
            if previous != identity:
                raise ValueError("Final device SPIR-V changed between native backend/optimization-flag arms")
            if not application and not sample["pipeline"].get("native_binary_persisted"):
                raise ValueError("Native binary prime did not persist a reusable artifact")
            if application:
                application_identities[arm["name"]] = artifact_snapshot(cache_file)
            else:
                binary_identities[arm["name"]] = binary_snapshot(directory)
            summary["excluded_runs"].append(dict(condition="Application pipeline cache prime" if application else "Native binary extraction", arm=arm,
                                                   process=process, pipeline=sample["pipeline"], pipeline_wall_ms=sample["pipeline_wall_ms"],
                                                   report=str(destination / "report.json"), **costs))
            save(output / "summary.json", summary)
    summary["native_binary_identities"] = binary_identities
    summary["application_pipeline_cache_identities"] = application_identities
    for block in range(args.pairs):
        for arm in balanced_order(conditions, block):
            # When importing, retain both source and import arms for a direct
            # same-bytecode comparison; extraction/priming is never a sample.
            modes = import_modes(arm, args.native_binary_cache, application_pipeline_cache)
            if block % 2:
                modes = tuple(reversed(modes))
            for imported in modes:
                check_snapshots(inputs)
                application = imported and arm["backend"] == "pipeline"
                directory = output / "native-binaries" / arm["name"] if imported and not application else None
                cache_file = output / "application-pipeline-caches" / f"{arm['name']}.pso" if application else None
                if imported:
                    identity = artifact_snapshot(cache_file) if application else binary_snapshot(directory)
                    expected = application_identities[arm["name"]] if application else binary_identities[arm["name"]]
                    if identity != expected:
                        raise ValueError("Cache files changed after priming/extraction")
                label = f"block-{block:02}-{arm['name']}-{'import' if imported else 'source'}"
                destination = output / label
                print(label, flush=True)
                process, report = driver.run_process(native_command(common, arm, destination, directory, cache_file),
                    environment_for(arm, base, directory), destination, args.timeout)
                sample = verify_native(report, shaders[arm["optimization"]], arm, imported=imported, application_cache=application, device=device)
                costs = creation_costs(report, sample)
                identity = device_shader_identity(sample)
                previous = device_shaders.setdefault(arm["optimization"], identity)
                if previous != identity:
                    raise ValueError("Final device SPIR-V changed between native backend/optimization-flag arms")
                device = report["device"]
                if imported:
                    identity = artifact_snapshot(cache_file) if application else binary_snapshot(directory)
                    if identity != expected:
                        raise ValueError("Cache files changed during import")
                summary["runs"].append(dict(block=block, arm=arm, imported=imported, process=process,
                                            cache_kind="application-pipeline-cache" if application else "native-binary" if imported else "none",
                                            pipeline=sample["pipeline"], pipeline_wall_ms=sample["pipeline_wall_ms"],
                                            compile_wall_ms=sample.get("compile_wall_ms"), shader_module_wall_ms=sample.get("shader_module_wall_ms"),
                                            compilation_phases=sample.get("compilation_phases"), report=str(destination / "report.json"), **costs))
                save(output / "summary.json", summary)
    summary["device"] = device
    summary["timings"] = {f"{arm['name']}-{'import' if imported else 'source'}": {
        metric: driver.timings([(run[metric] if metric in run else run["pipeline"][metric]) for run in summary["runs"]
            if run["arm"] == arm and run["imported"] == imported and
               (run.get(metric) if metric in run else run["pipeline"].get(metric)) is not None])
        for metric in ("pipeline_wall_ms", "pipeline_cache_preparation_wall_ms", "pipeline_load_and_create_wall_ms",
                       "compile_wall_ms", "shader_module_wall_ms", "native_create_ms", "driver_ms", "cache_lock_wait_ms", "native_binary_import_ms", "native_binary_export_ms")}
        for arm in conditions for imported in import_modes(arm, args.native_binary_cache, application_pipeline_cache)}
    check_snapshots(inputs)


def native_command(common, arm, destination, binary_directory=None, application_pipeline_cache=None):
    command = common + ["--slang-optimization", arm["optimization"], "--compute-backend", arm["backend"],
                        "--output-dir", str(destination)]
    if arm["disable_driver_optimization"]:
        command.append("--disable-driver-optimization")
    if binary_directory is not None:
        command += ["--native-binary-cache-dir", str(binary_directory)]
    if application_pipeline_cache is not None:
        command[command.index("--pipeline-cache") + 1] = "reuse"
        command += ["--pipeline-cache-file", str(application_pipeline_cache)]
    return command


def workload_configuration(args, output):
    original = args.cases.resolve(strict=True)
    configuration = copy.deepcopy(read(original))
    if args.case_id:
        selected = [case for case in configuration["cases"] if case["id"] in args.case_id]
        if {case["id"] for case in selected} != set(args.case_id):
            raise ValueError("Unknown --case-id")
        configuration["cases"] = selected
    if not configuration["cases"] or len({case["id"] for case in configuration["cases"]}) != len(configuration["cases"]):
        raise ValueError("Workload must contain distinct production cases")
    if any(option is not None for option in (args.resolution, args.warmup_frames, args.timing_frames)) and any(
            case.get("captureFrames") or case.get("cameraFrames") or case.get("frameCount")
            for case in configuration["cases"]):
        raise ValueError("Cannot resize the frame window of a temporal/camera correctness workload")
    configuration["warmupFrames"] = args.warmup_frames if args.warmup_frames is not None else configuration["warmupFrames"]
    configuration["timingFrames"] = args.timing_frames if args.timing_frames is not None else configuration["timingFrames"]
    if args.warmup_frames is not None or args.timing_frames is not None:
        configuration["frames"] = configuration["warmupFrames"] + configuration["timingFrames"] + 1
    if configuration["timingFrames"] < 1 or configuration["frames"] <= configuration["warmupFrames"] + configuration["timingFrames"]:
        raise ValueError("HDR capture must follow a positive timing window")
    paths = [original, *map(Path, configuration.get("assetFiles", []))]
    for case in configuration["cases"]:
        graph = (original.parent / case["graph"]).resolve(strict=True)
        paths.append(graph)
        case["graph"] = str(graph)
        if args.resolution is not None:
            case["width"] = case["height"] = args.resolution
        if not case.get("requiredTiming"):
            raise ValueError(f"Missing production GPU timing scope: {case['id']}")
        # Guides exported by the actual graph are read back after timing. Avoid
        # asking a pass for outputs absent from the declared workload.
        if args.capture_graph_outputs:
            graph_outputs = read(graph).get("outputs", [])
            case["captureOutputs"] = [name for name in graph_outputs if name != case["output"]]
    path = output / "Cases.json"
    save(path, configuration)
    paths.append(path)
    return path, configuration, paths


def readback(path, case, filename=None, expected_bytes=None, capture_format=None):
    import numpy as np
    if capture_format is None:
        layout = {"RGBA32F": ("<f4", 4), "RGBA16F": ("<f2", 4)}.get(case["format"])
    else:
        # The report stores metallic::render::Format. Only floating point HDR,
        # motion and depth guides are accepted; reject unrecognized layouts.
        layout = {20: ("<f2", 1), 25: ("<f2", 2), 30: ("<f2", 4),
                  33: ("<f4", 1), 36: ("<f4", 2), 39: ("<f4", 3),
                  42: ("<f4", 4), 47: ("<f4", 1)}.get(capture_format)
    if layout is None:
        raise ValueError("Unrecognized floating point readback format")
    dtype, channels = layout
    artifact = path.parent / (filename or case["image"])
    raw = artifact.read_bytes()
    size = case["width"] * case["height"] * channels * np.dtype(dtype).itemsize
    if expected_bytes is None:
        expected_bytes = size
    elif size != expected_bytes:
        raise ValueError("Guide byte count does not match its recorded format/resolution")
    if len(raw) != expected_bytes or len(raw) % np.dtype(dtype).itemsize:
        raise ValueError(f"Truncated readback: {artifact}")
    values = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite readback: {artifact}")
    return values


def validate_requested_case(case, specification, configuration):
    # MaterialBaseline copies these controls from the actual Cases.json it read.
    # Checking only agreement between arms would also accept every arm loading
    # the same unintended case file, resolution or camera sequence.
    for field in ("graph", "width", "height", "scene", "environment", "output",
                  "frameCount", "cameraFrames", "captureFrames", "captureOutputs",
                  "guideCameraStep", "useRenderView", "materialAsset", "materialRoot",
                  "requireEnvironment"):
        if field in specification and (field not in case or case[field] != specification[field]):
            raise ValueError(f"Requested production workload changed: {specification['id']} {field}")
    # Differential captures occur during the main loop, in increasing frame
    # order. Guide captures follow that loop, one additional frame per output.
    # This mirrors MaterialBaseline's capture contract, not its readback values.
    frame_count = specification.get("frameCount", configuration.get("frames"))
    if frame_count is not None:
        expected = [(specification["output"], frame)
                    for frame in sorted(set(specification.get("captureFrames", [])))]
        expected += [(output, frame_count + index)
                     for index, output in enumerate(specification.get("captureOutputs", []))]
        captures = case.get("captures", [])
        actual = [(capture.get("output"), capture.get("frame")) for capture in captures]
        if actual != expected:
            raise ValueError(f"Missing, duplicated or reordered requested readback: {specification['id']}")
    elif specification.get("captureFrames") or specification.get("captureOutputs"):
        raise ValueError("Requested readbacks require a recorded production frame count")


def compare_readbacks(baseline_path, candidate_path, configuration, absolute, relative):
    import numpy as np
    baseline, candidate = read(baseline_path), read(candidate_path)
    expected = [case["id"] for case in configuration["cases"]]
    if [case["id"] for case in baseline["cases"]] != expected or [case["id"] for case in candidate["cases"]] != expected:
        raise ValueError("Incomplete or reordered production workload")
    result = []
    for left, right, specification in zip(baseline["cases"], candidate["cases"], configuration["cases"]):
        validate_requested_case(left, specification, configuration)
        validate_requested_case(right, specification, configuration)
        for field in ("resolvedGraph", "width", "height", "format", "environment", "scene", "output"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Production workload changed: {left['id']} {field}")
        if not left.get("validHDR") or not right.get("validHDR"):
            raise ValueError("Invalid HDR cannot qualify a GPU execution comparison")
        arrays = [("final HDR", readback(baseline_path, left), readback(candidate_path, right))]
        lc, rc = left.get("captures", []), right.get("captures", [])
        if len(lc) != len(rc):
            raise ValueError("Missing guide/temporal readback")
        for a, b in zip(lc, rc):
            if a != b:
                raise ValueError("Guide/temporal capture contract changed")
            av = readback(baseline_path, left, a["file"], a["bytes"], a["format"])
            bv = readback(candidate_path, right, b["file"], b["bytes"], b["format"])
            arrays.append((a["output"], av, bv))
        comparisons = []
        for name, a, b in arrays:
            if a.shape != b.shape:
                raise ValueError("Differential readback shape changed")
            error = np.abs(a - b)
            bound = absolute + relative * np.abs(a)
            comparisons.append(dict(output=name, numerically_exact=bool(np.array_equal(a, b)),
                                    max_absolute_error=float(error.max(initial=0)),
                                    within_tolerance=bool(np.all(error <= bound))))
        result.append(dict(case=left["id"], qualified=all(c["within_tolerance"] for c in comparisons), readbacks=comparisons))
    return result


def gpu_timings(report, configuration):
    expected = [case["id"] for case in configuration["cases"]]
    if [case["id"] for case in report["cases"]] != expected:
        raise ValueError("Missing or reordered production cases")
    result = {}
    for case, specification in zip(report["cases"], configuration["cases"]):
        validate_requested_case(case, specification, configuration)
        if not case.get("validHDR"):
            raise ValueError("Invalid HDR cannot qualify GPU performance")
        rows = case["frames"]
        first, count = configuration["warmupFrames"], configuration["timingFrames"]
        if [row["frame"] for row in rows] != list(range(first, first + count)) or \
           len({row["executionId"] for row in rows}) != count or any(row["warmup"] for row in rows):
            raise ValueError("Incomplete or duplicated production timing window")
        node_name, section_name = specification["requiredTiming"]
        values = dict(graph_ms=[], pass_ms=[], dispatch_scope_ms=[])
        for row in rows:
            if row["asyncComputeBranches"] != specification.get("expectedAsyncComputeBranches", 0):
                raise ValueError("Asynchronous workload changed")
            node = next((node for node in row["nodes"] if node["name"] == node_name), None)
            if node is None or node["queue"] != 0:
                raise ValueError("Required production timing node missing or moved to another queue")
            section = next((scope for scope in node["sections"] if scope["name"] == section_name), None)
            if section_name and section is None:
                raise ValueError("Required dispatch timing scope missing")
            for label, value in (("graph_ms", row["graphMs"]), ("pass_ms", node["gpuMs"]),
                                 ("dispatch_scope_ms", section["gpuMs"] if section_name else node["gpuMs"])):
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid GPU timestamp")
                values[label].append(value)
        result[case["id"]] = {name: driver.timings(times) for name, times in values.items()}
    return result


def compiler_log_evidence(path, arm):
    log = path.read_text(encoding="utf-8", errors="replace")
    effective = "O1" if arm["optimization"] == "default" else arm["optimization"]
    levels = re.findall(r"(?:\[Slang\].*Begin compile|\[ShaderCache\].*Loaded).*optimization=(O[0-3])", log)
    if not levels or set(levels) != {effective}:
        raise ValueError("Runtime log did not confirm the requested effective Slang optimization")
    if re.search(r"(?:VUID-|Validation Error)", log):
        raise ValueError("Runtime log contains Vulkan validation errors")
    policies = re.findall(r"\[ComputeCreatePolicy\] backend=([012]) disableOptimization=(true|false) pso=([0-9a-fA-F]{16})", log)
    expected_backend = str(BACKENDS.index(arm["backend"]))
    expected_disabled = "true" if arm["disable_driver_optimization"] else "false"
    confirmed = [pso for backend, disabled, pso in policies if (backend, disabled) == (expected_backend, expected_disabled)]
    if not confirmed:
        raise ValueError("Runtime log did not confirm the requested native compute creation policy")
    return dict(effective_slang_optimization=effective, confirmed_shader_count=len(levels),
                native_compute_pso_hashes=confirmed,
                other_compute_creation_policies=[dict(backend=backend, disable_optimization=disabled, pso_hash=pso)
                    for backend, disabled, pso in policies if (backend, disabled) != (expected_backend, expected_disabled)])


def gpu_inventory():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def recipe_evidence(directory, arm):
    devices, shaders = [], {}
    effective = "O1" if arm["optimization"] == "default" else arm["optimization"]
    expected_backend = BACKENDS.index(arm["backend"])
    selected_compute = 0
    for manifest in sorted(directory.glob("*.recipe.json")):
        recipe = read(manifest)
        if recipe.get("schema") != "metallic_pipeline_recipe":
            raise ValueError("Unexpected runtime pipeline recipe")
        device = recipe["source_device"]
        if device not in devices:
            devices.append(device)
        if recipe["pipeline_type"] == "compute" and recipe["state"]["compute_backend"] == expected_backend and \
           recipe["state"]["disable_optimization"] == arm["disable_driver_optimization"]:
            selected_compute += 1
        for stage in recipe["shaders"].values():
            filename = stage["spirv_file"]
            if filename is None:
                continue
            file = directory / filename
            raw = file.read_bytes()
            if len(raw) != stage["spirv_bytes"] or raw[:4] != b"\x03\x02\x23\x07":
                raise ValueError("Invalid runtime recipe SPIR-V")
            if stage.get("slang_optimization") != effective or stage["debug_mode"] != "release":
                raise ValueError("Runtime recipe front-end policy changed")
            identity = dict(module=stage["module_name"], entry=stage["source_entry_point"],
                            spirv_bytes=len(raw), spirv_sha256=hashlib.sha256(raw).hexdigest(),
                            input_spirv_fnv1a64=stage["input_spirv_hash"],
                            descriptor_mode=stage["descriptor_heap_mode"], optimization=effective)
            shaders[(identity["module"], identity["entry"], identity["spirv_sha256"])] = identity
    if not selected_compute or not devices or not shaders:
        raise ValueError("Runtime did not export real compute PSO/input evidence for the requested backend")
    return dict(devices=devices, shaders=sorted(shaders.values(), key=lambda s: (s["module"], s["entry"], s["spirv_sha256"])),
                selected_compute_recipe_count=selected_compute)


def dispatch_experiment(args, output, summary):
    executable = args.executable.resolve(strict=True)
    conditions = arms(args.optimizations, args.backends, args.driver_optimization)
    validate_dispatch_binary_backends(args.backends, args.native_binary_cache)
    cases_path, configuration, extra = workload_configuration(args, output)
    inputs = input_identity(executable, extra, production=True)
    save(output / "Inputs.json", inputs)
    summary.update(conditions=conditions, pairs=args.pairs, validation=args.validation,
                   native_binary_cache=args.native_binary_cache,
                   workload=configuration, tolerance=dict(absolute=args.absolute_tolerance, relative=args.relative_tolerance),
                   scope="Production graph/pass/dispatch GPU timestamp scopes; excludes compilation, upload, readback, denoiser and presentation.",
                   statistical_unit="One independent-process median per arm/block; no frame-rate claim.",
                   caveats=["Readback tolerance is a correctness gate for this fixed workload, not all scenes.",
                            "Cooperative locks do not establish global GPU exclusivity; clocks are unaltered.",
                            "Small workloads and repeat counts are exploratory, not a production performance recommendation."])
    summary["gpu_inventory"] = gpu_inventory()
    base = clean_environment()
    base["METALLIC_MATERIAL_BASELINE_CASES"] = str(cases_path)
    base["METALLIC_PSO_CACHE_DIRECTORY"] = str(output / "runtime-pso-cache")
    baseline_path = None
    qualified = True
    shader_identities = {}
    device_identities = None
    for block in range(args.pairs):
        for arm in balanced_order(conditions, block):
            check_snapshots(inputs)
            if gpu_inventory() != summary["gpu_inventory"]:
                raise ValueError("GPU inventory or driver changed during production execution")
            label = f"block-{block:02}-{arm['name']}"
            destination = output / label
            destination.mkdir(parents=True, exist_ok=False)
            command = [str(executable), "--gtest_filter=RHIRendering.material_phase0_baseline",
                       "--rhi-validation" if args.validation else "--rhi-no-validation", "--output-dir", str(destination)]
            directory = dispatch_binary_directory(output, arm, args.native_binary_cache)
            environment = environment_for(arm, base, directory, internal_cache="enabled")
            environment["METALLIC_PIPELINE_RECIPE_DIRECTORY"] = str(destination / "recipes")
            print(label, flush=True)
            started = time.time()
            with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, \
                 (destination / "stderr.log").open("w", encoding="utf-8") as stderr:
                with driver.suppress_windows_error_dialogs():
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                timed_out = False
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    process.kill(); code = process.wait(); timed_out = True
            metadata = dict(pid=process.pid, command=command, started=started, ended=time.time(),
                            exit_code=code, timed_out=timed_out, native_crash=driver.native_process_crashed(code),
                            environment={k: v for k, v in environment.items() if k.startswith(("METALLIC_", "VK_"))})
            save(destination / "process.json", metadata)
            if code or timed_out:
                raise RuntimeError(f"Production dispatch process failed ({code}); preserve {destination}")
            report_path = destination / "MaterialBaseline.json"
            report = read(report_path)
            if report["validation"] != args.validation:
                raise ValueError("Runtime validation configuration changed")
            timings = gpu_timings(report, configuration)
            evidence = compiler_log_evidence(destination / "stdout.log", arm)
            recipes = recipe_evidence(destination / "recipes", arm)
            if device_identities is None:
                device_identities = recipes["devices"]
            elif recipes["devices"] != device_identities:
                raise ValueError("Runtime device/driver/debug/validation configuration changed")
            previous = shader_identities.setdefault(arm["optimization"], recipes["shaders"])
            if previous != recipes["shaders"]:
                raise ValueError("Runtime input SPIR-V changed between backend/driver-optimization arms")
            if baseline_path is None:
                baseline_path = report_path
            comparison = compare_readbacks(baseline_path, report_path, configuration,
                                           args.absolute_tolerance, args.relative_tolerance)
            passed = all(case["qualified"] for case in comparison)
            qualified &= passed
            summary["runs"].append(dict(block=block, arm=arm, process=metadata, report=str(report_path),
                                        compiler=evidence, recipes=recipes, timings=timings, correctness=comparison, qualified=passed))
            save(output / "summary.json", summary)
            check_snapshots(inputs)
    summary["correctness_qualified"] = qualified
    summary["devices"] = device_identities
    summary["shaders"] = shader_identities
    summary["timings"] = {arm["name"]: {case["id"]: {
        name: driver.timings([run["timings"][case["id"]][name]["median"] for run in summary["runs"] if run["arm"] == arm])
        for name in ("graph_ms", "pass_ms", "dispatch_scope_ms")}
        for case in configuration["cases"]} for arm in conditions}
    if not qualified:
        raise ValueError("Production readbacks exceeded the predeclared tolerance; timings do not qualify as a correctness-preserving optimization")


def seal(output):
    manifest = {str(p.relative_to(output)).replace("\\", "/"): digest(p)
                for p in sorted(output.rglob("*")) if p.is_file() and p != output / "Manifest.json"}
    save(output / "Manifest.json", manifest)


def verify(output):
    output = Path(output).resolve(strict=True)
    expected = read(output / "Manifest.json")
    actual = {str(p.relative_to(output)).replace("\\", "/"): digest(p)
              for p in sorted(output.rglob("*")) if p.is_file() and p != output / "Manifest.json"}
    if actual != expected:
        raise ValueError("Experiment evidence changed after sealing")
    summary = read(output / "summary.json")
    if summary["status"] != "passed":
        raise ValueError("Incomplete or failed experiment")
    runs = summary["runs"]
    if not runs or any(run["process"]["exit_code"] or run["process"]["timed_out"] for run in runs):
        raise ValueError("Missing or failed measured process")
    expected_sequence = []
    for block in range(summary["pairs"]):
        for arm in balanced_order(summary["conditions"], block):
            modes = import_modes(arm, summary["native_binary_cache"], summary.get("application_pipeline_cache", False)) \
                if summary["command"] == "compile" else (False,)
            if block % 2:
                modes = tuple(reversed(modes))
            expected_sequence.extend((block, arm, imported) for imported in modes)
    if [(run["block"], run["arm"], run.get("imported", False)) for run in runs] != expected_sequence:
        raise ValueError("Incomplete, duplicated or reordered experiment schedule")
    if summary["command"] == "dispatch":
        baseline = Path(runs[0]["report"])
        for run in runs:
            path = Path(run["report"])
            if gpu_timings(read(path), summary["workload"]) != run["timings"]:
                raise ValueError("GPU timing summary changed")
            tolerance = summary["tolerance"]
            if compare_readbacks(baseline, path, summary["workload"], tolerance["absolute"], tolerance["relative"]) != run["correctness"]:
                raise ValueError("Correctness summary changed")
            if "recipes" in run and recipe_evidence(path.parent / "recipes", run["arm"]) != run["recipes"]:
                raise ValueError("Runtime PSO/SPIR-V evidence summary changed")
    else:
        for run in runs:
            report = read(run["report"])
            if summary.get("schema_version", 1) >= 2:
                expected_kind = "none" if not run["imported"] else \
                    "application-pipeline-cache" if run["arm"]["backend"] == "pipeline" else "native-binary"
                if run.get("cache_kind") != expected_kind:
                    raise ValueError("Import artifact kind differs from its backend")
            sample = verify_native(report, summary["shaders"][run["arm"]["optimization"]], run["arm"],
                                   imported=run["imported"], application_cache=run.get("cache_kind") == "application-pipeline-cache",
                                   device=summary["device"])
            if sample["pipeline"] != run["pipeline"]:
                raise ValueError("Native timing summary changed")
            if sample["pipeline_wall_ms"] != run["pipeline_wall_ms"]:
                raise ValueError("Total pipeline creation wall time changed")
            if summary.get("schema_version", 1) >= 2:
                for field, value in creation_costs(report, sample).items():
                    if run[field] != value:
                        raise ValueError("Pipeline load/preparation timing summary changed")
                for field in ("compile_wall_ms", "shader_module_wall_ms", "compilation_phases"):
                    if run.get(field) != sample.get(field):
                        raise ValueError("Shader preparation timing summary changed")
            if device_shader_identity(sample) != summary["device_shaders"][run["arm"]["optimization"]]:
                raise ValueError("Device SPIR-V identity summary changed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compile", "dispatch"):
        run = sub.add_parser(name)
        run.add_argument("--executable", type=Path, required=True)
        run.add_argument("--output-dir", type=Path, required=True)
        run.add_argument("--optimizations", choices=OPTIMIZATIONS, nargs="+", default=["O0", "O1", "O2"])
        run.add_argument("--backends", choices=BACKENDS, nargs="+", default=["pipeline"])
        run.add_argument("--driver-optimization", choices=("normal", "disabled", "both"), default="both")
        run.add_argument("--pairs", type=int, default=3)
        run.add_argument("--timeout", type=float, default=600)
        run.add_argument("--native-binary-cache", action="store_true")
        run.add_argument("--validation", action="store_true")
    native = sub.choices["compile"]
    native.add_argument("--request-index", type=int, required=True)
    native.add_argument("--descriptor-mode", choices=("request", "mapped", "native"), default="request")
    native.add_argument("--application-pipeline-cache", action="store_true",
                        help="Prime a per-arm .pso (excluded), then compare empty-cache source with independent-process cache import")
    native.add_argument("--slang-perf-diagnostics", action="store_true",
                        help="Request detailed Slang performance reports during excluded SPIR-V prefill; SDK counters overlap wall phases")
    dispatch = sub.choices["dispatch"]
    dispatch.add_argument("--cases", type=Path, required=True)
    dispatch.add_argument("--case-id", nargs="+")
    dispatch.add_argument("--resolution", type=int)
    dispatch.add_argument("--warmup-frames", type=int)
    dispatch.add_argument("--timing-frames", type=int)
    dispatch.add_argument("--capture-graph-outputs", action="store_true")
    dispatch.add_argument("--absolute-tolerance", type=float, default=2e-5)
    dispatch.add_argument("--relative-tolerance", type=float, default=2e-5)
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.output), indent=2))
        return
    if args.pairs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or \
       (args.command == "compile" and args.request_index < 0):
        parser.error("pairs/timeout must be positive, timeout finite and request index nonnegative")
    if args.command == "dispatch" and (any(value is not None and value < 1 for value in
            (args.resolution, args.timing_frames)) or (args.warmup_frames is not None and args.warmup_frames < 0) or
            any(not math.isfinite(value) or value < 0 for value in (args.absolute_tolerance, args.relative_tolerance))):
        parser.error("resolution/timing count must be positive; warmup/tolerances must be nonnegative and finite")
    for name in LOCK_NAMES:
        existing = ROOT / "build" / name
        if existing.exists():
            raise RuntimeError(f"Existing GPU experiment lock: {existing}; inspect its owner before running")
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=2, command=args.command, status="running", runs=[], excluded_runs=[])
    save(output / "summary.json", summary)
    with lock.open("x", encoding="utf-8") as file:
        json.dump(dict(pid=os.getpid(), output=str(output), command=args.command), file)
    try:
        if args.command == "compile":
            compile_experiment(args, output, summary)
        else:
            dispatch_experiment(args, output, summary)
        summary["status"] = "passed"
        save(output / "summary.json", summary)
        seal(output)
        verify(output)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        save(output / "summary.json", summary)
        parser.exit(1, f"{error}\nEvidence: {output / 'summary.json'}\n")
    finally:
        lock.unlink()
    print(output / "summary.json")


if __name__ == "__main__":
    main()
