"""Compile-only pinned Slang CLI diagnostics; never infer costs for unequal SPIR-V.

The component API does not guarantee a detailed per-pass diagnostic report.
This separate CLI diagnostic run must match the Bench's exact finalized bytecode
before its phase report is associated with the measured native workload.
"""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOCK_NAMES = ("shader-experiment.lock", "openpbr-benchmark.lock", "material-phase0.lock",
              "compute-pipeline-experiment.lock")


@contextmanager
def experiment_lock():
    # Large frontend jobs also compete for CPU time with driver compilation.
    # Use the existing experiment lock to keep diagnostic work out of timings.
    for name in LOCK_NAMES:
        existing = ROOT / "build" / name
        if existing.exists():
            raise ValueError(f"Existing performance experiment lock: {existing}")
    path = ROOT / "build/compute-pipeline-experiment.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), command="frontend-profile"), stream)
    try:
        yield
    finally:
        path.unlink()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def decode_shader_cache(data):
    """Read only the current bounded v2 cache format, including its checksum."""
    if not 48 <= len(data) <= 512 * 1024 * 1024:
        raise ValueError("Invalid shader cache file size")
    magic, version, header_size, request_hash, count, reserved, size, checksum = struct.unpack_from("<8sIIQIIQQ", data)
    if magic != b"MTLSPV01" or version != 2 or header_size != 48 or reserved or not 0 < count <= 4096:
        raise ValueError("Invalid shader cache header")
    if size < 20 or size % 4 or size > len(data) - 48:
        raise ValueError("Invalid shader cache SPIR-V size")
    payload = data[48:]
    if fnv(payload) != f"{checksum:016x}":
        raise ValueError("Invalid shader cache checksum")
    offset = 0
    for _ in range(count):
        if offset + 16 > len(payload):
            raise ValueError("Truncated shader cache dependency")
        length, dependency_reserved, dependency_hash = struct.unpack_from("<IIQ", payload, offset)
        offset += 16
        if dependency_reserved or not 0 < length <= 32768 or offset + length > len(payload):
            raise ValueError("Invalid shader cache dependency")
        if b"\0" in payload[offset:offset + length]:
            raise ValueError("Null byte in shader cache dependency path")
        offset += length
    code = payload[offset:]
    if len(code) != size or code[:4] != b"\x03\x02\x23\x07":
        raise ValueError("Invalid shader cache payload")
    return code


def bench_bytes(report, sample):
    # Current Metallic shader cache format v2; reject all unknown layouts.
    expected_size = sample["spirv_bytes"]
    expected_fnv = sample["spirv_fnv1a64"]
    for path in sorted((Path(report["configuration"]["cache_directory"]) / "spirv").glob("*.spv")):
        if not 48 <= path.stat().st_size <= 512 * 1024 * 1024:
            continue
        data = path.read_bytes()
        if struct.unpack_from("<Q", data, 32)[0] != expected_size:
            continue
        try:
            code = decode_shader_cache(data)
        except ValueError:
            continue
        if fnv(code) == expected_fnv:
            return path.resolve(), code
    raise ValueError(f"No validated Bench cache payload matches {expected_fnv}/{expected_size}")


def diagnostic_command(compiler, report, sample, destination):
    request = sample["request"]
    if sample["effective_descriptor_mode"] != "mapped" or report["configuration"]["debug_mode"] != "disabled":
        raise ValueError("This diagnostic helper requires mapped descriptors and disabled shader debug information")
    root = Path(request["search_path"]).resolve()
    paths = [root, *(Path(path).resolve() for path in request["additional_search_paths"])]
    for program_root in (root, ROOT / "Shaders"):
        for subdirectory in ("Modules", "Interop"):
            path = program_root / subdirectory
            if path not in paths:
                paths.append(path)
    command = [str(compiler), str(root / (request["module"] + ".slang")),
               "-module-name", request["module"], "-entry", request["entry"],
               "-target", "spirv", "-profile", request["profile"], "-emit-spirv-directly",
               "-matrix-layout-row-major", "-" + sample["effective_slang_optimization"],
               "-report-detailed-perf-benchmark", "-report-downstream-time", "-o", str(destination)]
    for path in paths:
        command += ["-I", str(path)]
    for capability in request["capabilities"]:
        command += ["-capability", capability]
    for name, value in request["defines"]:
        command += ["-D", f"{name}={value}"]
    working_space = report["environment"].get("METALLIC_WORKING_COLOR_SPACE") or "acescg"
    if working_space not in ("acescg", "rec709"):
        raise ValueError("Unknown measured working color space")
    command += ["-D", "METALLIC_WORKING_SPACE_ACESCG=" + ("1" if working_space == "acescg" else "0")]
    return command


def compare_bytecode(reference, emitted, record):
    """Update evidence and reject unequal input before accepting phase costs."""
    record.update(emitted_fnv1a64=fnv(emitted), emitted_bytes=len(emitted),
                  spirv_sha256=hashlib.sha256(emitted).hexdigest(),
                  matches_bench_byte_for_byte=emitted == reference)
    if emitted != reference:
        record["status"] = "bytecode_mismatch"
        raise ValueError("CLI/component SPIR-V differ; diagnostic timings must not be attributed to the Bench workload")


def seal(output):
    artifacts = [{"path": str(path.relative_to(output)), "sha256": sha(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path.name != "Manifest.json"]
    (output / "Manifest.json").write_text(json.dumps(dict(schema_version=1, artifacts=artifacts), indent=2) + "\n", encoding="utf-8")


def verify(output):
    output = output.resolve()
    manifest = json.loads((output / "Manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = (output / item["path"]).resolve()
        if output not in path.parents or sha(path) != item["sha256"]:
            raise ValueError("Diagnostic artifact checksum mismatch")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "passed" or not summary["runs"]:
        raise ValueError("Incomplete diagnostic experiment")
    expected = [(block, optimization) for block in range(summary["pairs"])
                for optimization in balanced_order(summary["optimizations"], block)]
    if [(record["block"], record["optimization"]) for record in summary["runs"]] != expected:
        raise ValueError("Incomplete or reordered diagnostic schedule")
    for record in summary["runs"]:
        if record["status"] != "passed" or not record["matches_bench_byte_for_byte"]:
            raise ValueError("Unqualified diagnostic bytecode")
        reference = (output / record["reference_path"]).read_bytes()
        emitted = (output / record["spirv_path"]).read_bytes()
        if reference != emitted or fnv(reference) != record["expected_fnv1a64"] or len(reference) != record["expected_bytes"]:
            raise ValueError("Diagnostic bytecode identity mismatch")
    return dict(status="verified", process_count=len(summary["runs"]))


def balanced_order(optimizations, block):
    shift = block % len(optimizations)
    order = optimizations[shift:] + optimizations[:shift]
    return list(reversed(order)) if block % 2 else order


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, help="A fixed-request Bench experiment directory containing prefill-<optimization>")
    parser.add_argument("--output-dir", type=Path, help="A fresh separate directory; never use a sealed experiment directory")
    parser.add_argument("--slang-root", type=Path, default=ROOT / "External/slang", help="The SDK used by the Bench (defaults to the pinned SDK)")
    parser.add_argument("--verify", type=Path, help="Verify completed local diagnostic evidence without compiling")
    parser.add_argument("--optimizations", nargs="+", default=["O0", "O1", "O2"], choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--pairs", type=int, default=1, help="Independent CLI processes for each optimization level")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return 0
    if not args.input_root or not args.output_dir:
        parser.error("input-root and output-dir are required when profiling")
    if args.pairs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("pairs and timeout must be positive and finite")
    if len(set(args.optimizations)) != len(args.optimizations):
        parser.error("optimization conditions must be unique")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    repository = ROOT
    sdk = args.slang_root.resolve()
    compiler = sdk / "bin" / ("slangc.exe" if os.name == "nt" else "slangc")
    version_header = sdk / "include/slang-tag-version.h"
    sdk_files = [compiler, version_header, *sorted((sdk / "bin").glob("*.dll")),
                 *sorted((sdk / "lib").glob("libslang*.so*")), *sorted((sdk / "lib").glob("libslang*.dylib"))]
    summary = {"schema_version": 1, "status": "running", "runs": [],
               "scope": "Independent compile-only CLI diagnostics; per-pass costs are eligible only for exact byte-for-byte Bench matches.",
               "note": "Component API detailed profiling is not guaranteed to emit a report; this diagnostic path excludes all device and driver pipeline creation.",
               "sdk_files": [{"path": str(path), "sha256": sha(path)} for path in sdk_files],
               "optimizations": args.optimizations, "pairs": args.pairs,
               "helper_sha256": sha(__file__)}
    summary_path = output / "summary.json"
    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save()
    old_error_mode = None
    if os.name == "nt":
        old_error_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    try:
        inputs = {}
        version_match = re.search(r'#define\s+SLANG_TAG_VERSION\s+"([^"]+)"', version_header.read_text(encoding="utf-8"))
        if not version_match:
            raise ValueError("Pinned SDK version header is missing its build tag")
        summary["slang_build"] = version_match.group(1)
        common_identity = None
        for optimization in args.optimizations:
            report_path = (args.input_root / f"prefill-{optimization}" / "report.json").resolve()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["status"] != "passed" or report["failures"]:
                raise ValueError(f"Bench prefill failed: {report_path}")
            if not report["configuration"].get("compile_only"):
                raise ValueError("The reference must be a compile-only Bench prefill")
            samples = report["samples"]
            if len(samples) != 1 or samples[0]["effective_slang_optimization"] != optimization:
                raise ValueError("Each input report must have exactly one matching optimization sample")
            sample = samples[0]
            if sample["status"] != "passed" or sample.get("pipeline_type") != "compute":
                raise ValueError("The reference must be a successful compute shader compile")
            if report["slang_build"] != summary["slang_build"]:
                raise ValueError("CLI SDK build tag differs from the Bench compiler")
            identity = dict(request=sample["request"], dependencies=sample["dependencies"],
                            working_color_space=report["environment"].get("METALLIC_WORKING_COLOR_SPACE"))
            if common_identity is not None and identity != common_identity:
                raise ValueError("Optimization arms do not represent the same fixed shader request")
            common_identity = identity
            payload_path, code = bench_bytes(report, sample)
            (output / f"reference-{optimization}.spv").write_bytes(code)
            (output / f"reference-{optimization}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            tracked_paths = [report_path, payload_path, *(Path(path) for path in sample["dependencies"])]
            source_hashes = {str(path): sha(path) for path in tracked_paths}
            inputs[optimization] = (report, sample, code, source_hashes)
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith("METALLIC_"):
                del environment[key]
        # The Bench setup did not enable Slang's optional separate SPIR-V validation.
        environment.pop("SLANG_RUN_SPIRV_VALIDATION", None)
        summary["inherited_slang_environment"] = {key: value for key, value in environment.items() if key.startswith("SLANG_")}
        summary["input_files"] = {optimization: value[3] for optimization, value in inputs.items()}
        save()
        for block in range(args.pairs):
            for optimization in balanced_order(args.optimizations, block):
                report, sample, reference, source_hashes = inputs[optimization]
                directory = output / f"block-{block:02d}-{optimization}"
                directory.mkdir()
                destination = directory / "shader.spv"
                command = diagnostic_command(compiler, report, sample, destination)
                record = {"block": block, "optimization": optimization, "command": command,
                          "status": "running", "expected_fnv1a64": sample["spirv_fnv1a64"],
                          "expected_bytes": len(reference), "reference_path": f"reference-{optimization}.spv",
                          "spirv_path": str(destination.relative_to(output))}
                summary["runs"].append(record)
                save()
                print(f"[ProfileComputeShaderFrontend] block={block} optimization={optimization}", flush=True)
                start = time.perf_counter()
                try:
                    completed = subprocess.run(command, cwd=repository, env=environment, capture_output=True,
                                               timeout=args.timeout, creationflags=0x08000000 if os.name == "nt" else 0)
                except subprocess.TimeoutExpired as error:
                    (directory / "stdout.log").write_bytes(error.stdout or b"")
                    (directory / "stderr.log").write_bytes(error.stderr or b"")
                    record["process_wall_ms"] = (time.perf_counter() - start) * 1000
                    record["status"] = "timeout"
                    record["stdout_log"] = str(directory / "stdout.log")
                    record["stderr_log"] = str(directory / "stderr.log")
                    save()
                    raise
                record["process_wall_ms"] = (time.perf_counter() - start) * 1000
                record["exit_code"] = completed.returncode
                (directory / "stdout.log").write_bytes(completed.stdout)
                (directory / "stderr.log").write_bytes(completed.stderr)
                record["stdout_log"] = str(directory / "stdout.log")
                record["stderr_log"] = str(directory / "stderr.log")
                if completed.returncode != 0 or not destination.is_file():
                    raise ValueError(f"CLI compile failed: {directory}")
                emitted = destination.read_bytes()
                if any(sha(path) != expected for path, expected in source_hashes.items()):
                    raise ValueError("Bench reference or shader dependencies changed during diagnostics")
                compare_bytecode(reference, emitted, record)
                record["status"] = "passed"
                save()
        if any(sha(item["path"]) != item["sha256"] for item in summary["sdk_files"]):
            raise ValueError("Pinned SDK changed during diagnostics")
        summary["status"] = "passed"
        save()
        seal(output)
        verify(output)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        if summary["runs"] and summary["runs"][-1]["status"] == "running":
            summary["runs"][-1]["status"] = "failed"
        save()
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if old_error_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(old_error_mode)
    return 0


if __name__ == "__main__":
    try:
        # Verification is read-only and does not need the timing exclusion lock.
        if any(flag in sys.argv for flag in ("--verify", "--help", "-h")):
            sys.exit(main())
        with experiment_lock():
            sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
