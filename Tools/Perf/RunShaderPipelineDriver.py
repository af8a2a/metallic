"""Compare controllable Vulkan driver-cache states in independent processes.

Every measured process loads fixed SPIR-V from a prefilled cache or recipe blobs
and creates one PSO with an empty application VkPipelineCache. "Disabled" describes the requested
driver-cache control, not a guarantee that compiler/OS state is fully cold.
"Primed" describes a prior successful process, not proof of an internal hit.
"""

import argparse
import copy
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


CACHE_ENVIRONMENT = "METALLIC_VK_INTERNAL_PIPELINE_CACHE"
CONDITIONS = ("driver-cache-disabled", "driver-cache-enabled-primed")


@contextmanager
def suppress_windows_error_dialogs():
    """Children inherit this process error mode; restore the caller afterward."""
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetErrorMode.restype = ctypes.c_uint
    kernel.SetErrorMode.argtypes = (ctypes.c_uint,)
    kernel.SetErrorMode.restype = ctypes.c_uint
    previous = kernel.GetErrorMode()
    kernel.SetErrorMode(previous | 0x0001 | 0x0002)  # FAILCRITICALERRORS | NOGPFAULTERRORBOX
    try:
        yield
    finally:
        kernel.SetErrorMode(previous)


def native_process_crashed(exit_code):
    return exit_code < 0 or (exit_code & 0xffffffff) >= 0xc0000000


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def single_sample(report, *, native):
    if report.get("status") != "passed" or report.get("validation_errors", 0):
        raise ValueError("benchmark failed or reported Vulkan validation errors")
    samples = report.get("samples", [])
    if len(samples) != 1 or samples[0].get("status") != "passed":
        raise ValueError("expected exactly one successful fixed-request sample")
    sample = samples[0]
    if native and sample.get("pipeline_status") != "created":
        raise ValueError("selection did not create a pipeline; select a compute request or captured PSO recipe")
    return sample


def shader_identity(report, sample):
    identity = dict(request=sample["request"], spirv_bytes=sample["spirv_bytes"],
                    build_configuration=report["build_configuration"], slang_build=report["slang_build"],
                    shader_target=report["shader_target"], debug_mode=report["configuration"]["debug_mode"],
                    validation=report["configuration"]["validation"])
    if sample.get("shader_source") == "recipe_spirv":
        identity.update(shader_source="recipe_spirv", shader_stages=sample["shader_stages"],
                        pipeline_type=sample["pipeline_type"])
    else:
        identity.update(spirv_fnv1a64=sample["spirv_fnv1a64"], descriptor_mode=sample["effective_descriptor_mode"],
                        resource_heap_stride=sample["resource_heap_stride"], sampler_heap_stride=sample["sampler_heap_stride"])
    return copy.deepcopy(identity)


def verify_measurement(report, expected_shader, expected_device=None, disabled=False):
    sample = single_sample(report, native=True)
    if shader_identity(report, sample) != expected_shader:
        raise ValueError("SPIR-V, request or compiler configuration changed between processes")
    if expected_shader.get("shader_source") == "recipe_spirv":
        if sample.get("shader_source") != "recipe_spirv" or not sample["pipeline"].get("recipe_hash_matches"):
            raise ValueError("measured process did not reconstruct the captured PSO from recipe SPIR-V")
    elif not sample.get("shader_cache_hit"):
        raise ValueError("measured process did not reuse prefilled SPIR-V")
    if report["configuration"]["pipeline_cache"] != "empty":
        raise ValueError("driver comparison requires an empty application pipeline cache")
    if report["configuration"]["compile_only"]:
        raise ValueError("measured process was compile-only")
    control = report["environment"].get(CACHE_ENVIRONMENT)
    if control != ("disabled" if disabled else "enabled"):
        raise ValueError("driver-cache environment does not match the experiment condition")
    if expected_device is not None and report.get("device") != expected_device:
        raise ValueError("device, driver or active validation configuration changed")
    pipeline = sample["pipeline"]
    for field in ("native_create_ms", "cache_lock_wait_ms"):
        if not isinstance(pipeline.get(field), (int, float)) or not math.isfinite(pipeline[field]) or pipeline[field] < 0:
            raise ValueError(f"invalid {field}")
    if pipeline.get("driver_ms") is not None and (not math.isfinite(pipeline["driver_ms"]) or pipeline["driver_ms"] < 0):
        raise ValueError("invalid driver_ms")
    return sample


def timings(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0, min=None, median=None, p95=None, max=None)
    return dict(count=len(ordered), min=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(.95 * len(ordered)) - 1], max=ordered[-1])


def run_process(command, environment, destination, timeout):
    destination.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, \
         (destination / "stderr.log").open("w", encoding="utf-8") as stderr:
        with suppress_windows_error_dialogs():
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait()
            timed_out = True
    metadata = dict(pid=process.pid, started_utc=started, exit_code=exit_code, timed_out=timed_out,
                    native_crash=not timed_out and native_process_crashed(exit_code),
                    process_wall_ms=(time.perf_counter() - begin) * 1000, command=command)
    save(destination / "process.json", metadata)
    if timed_out:
        raise RuntimeError(f"benchmark timed out; logs: {destination}")
    if exit_code:
        if native_process_crashed(exit_code):
            raise RuntimeError(f"benchmark native crash 0x{exit_code & 0xffffffff:08X}; logs: {destination}")
        raise RuntimeError(f"benchmark exited {exit_code}; logs: {destination}")
    return metadata, json.loads((destination / "report.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--request-index", type=int)
    selection.add_argument("--recipe-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--descriptor-mode", choices=("request", "mapped", "native"), default="request")
    parser.add_argument("--debug-mode", choices=("disabled", "capture", "debug"), default="disabled")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1 or (args.request_index is not None and args.request_index < 0) or args.timeout <= 0:
        parser.error("pairs/timeout must be positive and request index nonnegative")
    executable = args.executable.resolve(strict=True)
    recipe = args.recipe_file.resolve(strict=True) if args.recipe_file else None
    selected = ["--recipe-file", str(recipe)] if recipe else ["--request-index", str(args.request_index)]
    recipe_hash = hashlib.sha256(recipe.read_bytes()).hexdigest() if recipe else None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    common = [str(executable), *selected, "--iterations", "1",
              "--shader-cache", "reuse", "--pipeline-cache", "empty", "--cache-dir", str(cache),
              "--descriptor-mode", args.descriptor_mode, "--debug-mode", args.debug_mode]
    if args.validation:
        common.append("--validation")
    # Both controls opt into the same pipeline-binary extension/feature chain;
    # compare only disableInternalCache, not the default device creation path.
    default_environment = dict(os.environ, **{CACHE_ENVIRONMENT: "enabled"})
    disabled_environment = dict(default_environment, **{CACHE_ENVIRONMENT: "disabled"})
    summary = dict(schema_version=1, status="running", runs=[], excluded_runs=[],
                   conditions=list(CONDITIONS), pairs=args.pairs,
                   scope="One native pipeline per independent process; no dispatch or frame timing.",
                   caveats=["Empty VkPipelineCache does not clear driver/OS/compiler state.",
                            "Disabled condition requests Metallic's controllable driver-cache override.",
                            "Both conditions explicitly select the same diagnostic device feature chain.",
                            "Priming does not prove an internal driver-cache hit.",
                            "native_cache_hit is Vulkan application-pipeline-cache feedback, not an internal-cache hit."])
    summary_path = output / "summary.json"
    if recipe:
        summary["recipe_file"] = str(recipe)
        summary["recipe_manifest_sha256"] = recipe_hash
    save(summary_path, summary)
    try:
        expected_shader = None
        if not recipe:
            destination = output / "spirv-prefill"
            process, report = run_process(common + ["--compile-only", "--output-dir", str(destination)],
                                          default_environment, destination, args.timeout)
            sample = single_sample(report, native=False)
            if sample.get("execution_model") != 5:
                raise ValueError("select a compute request from --list")
            expected_shader = shader_identity(report, sample)
            summary["excluded_runs"].append(dict(condition="spirv-prefill", report=str(destination / "report.json"), process=process))
        destination = output / "driver-prime"
        process, report = run_process(common + ["--output-dir", str(destination)],
                                      default_environment, destination, args.timeout)
        if recipe:
            sample = single_sample(report, native=True)
            expected_shader = shader_identity(report, sample)
            if expected_shader.get("shader_source") != "recipe_spirv" or hashlib.sha256(recipe.read_bytes()).hexdigest() != recipe_hash:
                raise ValueError("recipe source changed or was not imported")
        verify_measurement(report, expected_shader)
        summary["shader_identity"] = expected_shader
        device = report["device"]
        summary["device"] = device
        summary["excluded_runs"].append(dict(condition="driver-prime", report=str(destination / "report.json"), process=process))
        save(summary_path, summary)
        for pair in range(args.pairs):
            # Alternate the order to expose process-order drift. Each condition
            # uses the same empty application cache and warm SPIR-V bytes.
            for condition in CONDITIONS if pair % 2 == 0 else reversed(CONDITIONS):
                disabled = condition == CONDITIONS[0]
                destination = output / f"pair-{pair:02d}-{condition}"
                print(f"pair={pair} {condition}", flush=True)
                if recipe and hashlib.sha256(recipe.read_bytes()).hexdigest() != recipe_hash:
                    raise ValueError("recipe manifest changed between processes")
                process, report = run_process(common + ["--output-dir", str(destination)],
                    disabled_environment if disabled else default_environment, destination, args.timeout)
                sample = verify_measurement(report, expected_shader, device, disabled)
                if recipe and hashlib.sha256(recipe.read_bytes()).hexdigest() != recipe_hash:
                    raise ValueError("recipe manifest changed while creating the pipeline")
                summary["runs"].append(dict(pair=pair, condition=condition, report=str(destination / "report.json"),
                                            process=process, pipeline=sample["pipeline"]))
                save(summary_path, summary)
        summary["timings"] = {condition: {metric: timings([r["pipeline"][metric] for r in summary["runs"]
            if r["condition"] == condition and r["pipeline"].get(metric) is not None])
            for metric in ("native_create_ms", "driver_ms", "cache_lock_wait_ms")} for condition in CONDITIONS}
        summary["status"] = "passed"
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        save(summary_path, summary)
        parser.exit(1, f"{error}\nEvidence: {summary_path}\n")
    save(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
