"""Run matched 1/2/4/8-worker PSO creation and cache-lock-scope comparisons.

Each cell uses a new process, warm SPIR-V, empty application caches and the same
selected requests/replica count. Batch totals exclude compilation, module and
cache initialization, shader dispatch and frames; shard merges are separate.
"""

import argparse
import json
import os
from pathlib import Path

from RunShaderPipelineDriver import CACHE_ENVIRONMENT, run_process, save


def matrix():
    yield "serial", 1, "metadata"
    for workers in (1, 2, 4, 8):
        for scope in ("creation", "metadata"):
            yield "parallel-shared", workers, scope
        yield "parallel-sharded", workers, "metadata"


def identities(report):
    result = {}
    for sample in report.get("samples", []):
        if sample.get("status") != "passed":
            raise ValueError("benchmark contains a failed sample")
        key = str(sample["request"]["index"])
        identity = {field: sample[field] for field in ("request", "spirv_fnv1a64", "spirv_bytes",
            "effective_descriptor_mode", "resource_heap_stride", "sampler_heap_stride")}
        if key in result and result[key] != identity:
            raise ValueError("SPIR-V changed within a run")
        result[key] = identity
    if not result:
        raise ValueError("benchmark contains no samples")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--filter")
    selection.add_argument("--request-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--pipeline-replicas", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--driver-cache", choices=("disabled", "default"), default="disabled")
    parser.add_argument("--descriptor-mode", choices=("request", "mapped", "native"), default="request")
    parser.add_argument("--debug-mode", choices=("disabled", "capture", "debug"), default="disabled")
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.repeats) < 1 or args.pipeline_replicas < 8 or args.timeout <= 0:
        parser.error("iterations/repeats/timeout must be positive and --pipeline-replicas at least 8")
    if args.request_index is not None and args.request_index < 0:
        parser.error("--request-index must be nonnegative")
    executable = args.executable.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = args.cache_dir.resolve() if args.cache_dir else output / "cache"
    selected = ["--filter", args.filter] if args.filter else ["--request-index", str(args.request_index)]
    common = [str(executable), *selected, "--shader-cache", "reuse", "--pipeline-cache", "empty",
              "--cache-dir", str(cache), "--descriptor-mode", args.descriptor_mode, "--debug-mode", args.debug_mode]
    if args.validation:
        common.append("--validation")
    environment = os.environ.copy()
    environment.pop(CACHE_ENVIRONMENT, None)
    if args.driver_cache == "disabled":
        environment[CACHE_ENVIRONMENT] = "disabled"
    summary = dict(schema_version=1, status="running", driver_cache=args.driver_cache, runs=[],
                   scope=__doc__.strip(), warnings=["Internal/OS/compiler state may survive independent processes.",
                   "Replicas intentionally repeat selected PSOs; keep identical job counts across cells."])
    summary_path = output / "summary.json"
    save(summary_path, summary)
    try:
        destination = output / "spirv-prefill"
        process, report = run_process(common + ["--compile-only", "--iterations", "1", "--output-dir", str(destination)],
                                      environment, destination, args.timeout)
        if report.get("status") != "passed":
            raise ValueError("SPIR-V prefill failed")
        expected = identities(report)
        build_configuration = report["build_configuration"]
        compiler_identity = dict(slang_build=report["slang_build"], shader_target=report["shader_target"])
        if any(sample.get("execution_model") != 5 for sample in report["samples"]):
            raise ValueError("matrix selection must contain only compute requests; graphics PSOs require --recipe-file replay")
        summary["shader_identities"] = expected
        summary["build_configuration"] = build_configuration
        summary["compiler_identity"] = compiler_identity
        summary["excluded_prefill"] = dict(report=str(destination / "report.json"), process=process)
        device = None
        count = None
        for repeat in range(args.repeats):
            cells = list(matrix())
            if repeat % 2:
                cells.reverse()
            for mode, workers, scope in cells:
                destination = output / f"repeat-{repeat:02d}-{mode}-{workers}-{scope}"
                command = common + ["--mode", mode, "--workers", str(workers), "--cache-lock-scope", scope,
                    "--pipeline-replicas", str(args.pipeline_replicas), "--iterations", str(args.iterations),
                    "--output-dir", str(destination)]
                if args.merge_shards and mode == "parallel-sharded":
                    command.append("--merge-shards")
                print(f"repeat={repeat} {mode} workers={workers} lock={scope}", flush=True)
                process, report = run_process(command, environment, destination, args.timeout)
                if report.get("status") != "passed" or report.get("validation_errors", 0):
                    raise ValueError("matrix process failed or reported validation errors")
                if identities(report) != expected:
                    raise ValueError("shader requests/SPIR-V drifted across matrix cells")
                if report["build_configuration"] != build_configuration or dict(slang_build=report["slang_build"], shader_target=report["shader_target"]) != compiler_identity:
                    raise ValueError("build/compiler configuration changed across matrix cells")
                created = [sample for sample in report["samples"] if sample.get("pipeline_status") == "created"]
                if len(created) != len(expected) * args.pipeline_replicas * args.iterations or not all(sample.get("shader_cache_hit") for sample in created):
                    raise ValueError("matrix must create native PSOs with prefilled SPIR-V")
                if device is not None and report["device"] != device:
                    raise ValueError("device/driver/validation configuration changed")
                if count is not None and len(created) != count:
                    raise ValueError("pipeline job count differs between matrix cells")
                device, count = report["device"], len(created)
                summary["device"] = device
                summary["native_jobs_per_cell"] = count
                summary["runs"].append(dict(repeat=repeat, mode=mode, workers=workers, cache_lock_scope=scope,
                    report=str(destination / "report.json"), process=process, batches=report["pipeline_batches"],
                    pipeline_cache_merge_wall_ms=report.get("pipeline_cache_merge_wall_ms")))
                save(summary_path, summary)
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
