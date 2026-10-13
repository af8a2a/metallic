"""Measure the complete startup shader warmup with a fresh SPIR-V cache per process.

Runs are serial independent processes. Process wall time includes catalog creation,
native descriptor-stride queries, compilation, cache read-back and process teardown.
OS file caches are not flushed. Compiler policy and the inherited environment are
left unchanged; summed request phases overlap across workers and are not wall time.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".slang", ".h", ".hpp", ".hlsli", ".hlsl", ".cpp", ".json"}
NUMBER = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
SUMMARY = re.compile(r"^Shader warmup: (\d+) requests, (\d+) existing cache hits, (\d+) failures(.*)$", re.M)
TIMING = re.compile(rf"^\s*Request (\d+) timing: total=({NUMBER})s, setup=({NUMBER})s, module=({NUMBER})s, codegen=({NUMBER})s\s*$", re.M)
ELAPSED = re.compile(rf"^Shader warmup elapsed: ({NUMBER})s; summed request time: ({NUMBER})s \(module=({NUMBER})s, codegen=({NUMBER})s\)\s*$", re.M)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_files(root, exclude=None):
    roots = [root / name for name in ("Shaders", "Source", "Pipelines", ".cache/materials",
             "External/RTXCR-Material/shaders", "External/RTXCR-Geometry/shaders", "External/slang/include",
             "build/MaterialValidation/PainterLookDev", "build/MaterialValidation/WhiteStudio02")]
    files = {path.resolve() for directory in roots if directory.is_dir()
             for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES}
    files.update(path.resolve() for path in (root / "Tools").glob("*ShaderWarmup*.*") if path.is_file())
    files.update(path.resolve() for path in (root / "cmake/ShaderWarmup.cmake", Path(__file__)) if path.is_file())
    if exclude is not None:
        files = {path for path in files if not path.is_relative_to(exclude.resolve())}
    return files


def input_identity(executable, root=ROOT, exclude=None):
    binaries = {executable.resolve()}
    binaries.update(path.resolve() for path in executable.parent.iterdir() if path.is_file() and
                    (path.suffix.lower() in {".dll", ".dylib"} or ".so" in path.name))
    sources = source_files(root, exclude)
    files = sources | binaries
    return dict(files={str(path): digest(path) for path in sorted(files)},
                source_paths=[str(path) for path in sorted(sources)],
                binary_paths=[str(path) for path in sorted(binaries)],
                excluded_output_directory=str(exclude.resolve()) if exclude is not None else None,
                slang_binaries=[str(path) for path in sorted(binaries) if "slang" in path.name.lower()])


def check_identity(identity, executable, root=ROOT, exclude=None):
    current = input_identity(executable, root, exclude)
    require(current == identity, "Executable, DLL or source inventory/content changed; refusing mixed evidence")


def compiler_environment(environment):
    # Known compiler/loader/native-query settings only; unrelated application
    # variables may contain credentials and do not belong in evidence logs.
    keys = ("PATH", "SLANG_ROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                 "METALLIC_SLANG_OPTIMIZATION", "METALLIC_SLANG_DESCRIPTOR_MODE",
                 "METALLIC_SHADER_DEBUG_MODE", "METALLIC_SHADER_WARMUP_JOBS",
                 "METALLIC_WORKING_COLOR_SPACE", "METALLIC_SPIRV_POSTPROCESS",
                 "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH")
    return {key: environment.get(key) for key in sorted(keys)}


def fresh_cache(path):
    require(not path.exists(), f"Cold cache directory must be new: {path}")
    path.mkdir()
    return path


def ensure_empty_cache(path):
    require(path.is_dir() and not any(path.iterdir()), f"Cold cache directory is not empty: {path}")


class ProcessMemory:
    """Windows OS high-water WS/commit plus sampled PrivateUsage; optional elsewhere."""
    def __init__(self, process):
        self.process = process
        self.result = dict(provider="unavailable", peak_working_set_bytes=None,
                           peak_private_usage_bytes=None, peak_commit_bytes=None, samples=0)
        self.query = None
        if os.name == "nt":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]

            try:
                self.counters_type = Counters
                self.query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
                self.query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
                self.query.restype = wintypes.BOOL
                self.result["provider"] = "Windows GetProcessMemoryInfo"
                self.result["private_usage_scope"] = "maximum sampled PrivateUsage; peak_commit is OS PeakPagefileUsage"
            except (OSError, AttributeError) as error:
                self.result["unavailable_reason"] = str(error)
        elif Path(f"/proc/{process.pid}/status").is_file():
            self.result["provider"] = "Linux /proc/status VmHWM (private usage unavailable)"

    def update(self):
        try:
            if self.query is not None:
                counters = self.counters_type()
                counters.cb = ctypes.sizeof(counters)
                if not self.query(int(self.process._handle), ctypes.byref(counters), counters.cb):
                    return
                values = dict(peak_working_set_bytes=counters.PeakWorkingSetSize,
                              peak_private_usage_bytes=counters.PrivateUsage,
                              peak_commit_bytes=counters.PeakPagefileUsage)
            elif self.result["provider"].startswith("Linux"):
                text = Path(f"/proc/{self.process.pid}/status").read_text()
                match = re.search(r"^VmHWM:\s*(\d+) kB", text, re.M)
                if not match:
                    return
                values = dict(peak_working_set_bytes=int(match[1]) * 1024)
            else:
                return
            for key, value in values.items():
                self.result[key] = max(self.result[key] or 0, value)
            self.result["samples"] += 1
        except (OSError, AttributeError, ValueError):
            pass  # Memory diagnostics never turn an otherwise valid measurement into a failure.


def run_process(command, environment, destination, timeout):
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    timed_out = False
    with (destination / "stdout.log").open("wb") as stdout, (destination / "stderr.log").open("wb") as stderr:
        begin = time.perf_counter()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        memory = ProcessMemory(process)
        deadline = begin + timeout
        while True:
            memory.update()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if process.poll() is None:
                    process.kill()
                    timed_out = True
                exit_code = process.wait()
                break
            try:
                exit_code = process.wait(timeout=min(0.02, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        wall_seconds = time.perf_counter() - begin
        memory.update()  # On Windows the retained process handle exposes OS peak counters after exit.
    result = dict(command=command, cwd=str(ROOT), pid=process.pid, started_utc=started,
                  exit_code=exit_code, timed_out=timed_out, process_wall_seconds=wall_seconds,
                  memory=memory.result, stdout=str(destination / "stdout.log"), stderr=str(destination / "stderr.log"))
    save(destination / "process.json", result)
    return result


def successful_process(process):
    require(not process["timed_out"] and process["exit_code"] == 0, "Warmup process failed or timed out; see retained logs")
    value = process["process_wall_seconds"]
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid complete-process wall time")


def parse_catalog(text):
    rows = []
    for match in re.finditer(r"^\[(\d+)/(\d+)\] (\S+)(?: (.*))?$", text, re.M):
        rows.append(dict(index=int(match[1]), total=int(match[2]), request=match[3], define_text=match[4] or ""))
    summaries = SUMMARY.findall(text)
    require(len(summaries) == 1, "Catalog must contain exactly one complete list summary")
    count, hits, failures, suffix = summaries[0]
    require(int(count) > 0 and int(hits) == 0 and int(failures) == 0 and suffix == " (list only)", "Invalid full catalog summary")
    require([row["index"] for row in rows] == list(range(1, int(count) + 1)) and
            all(row["total"] == int(count) for row in rows), "Catalog requests are missing, duplicated or reordered")
    return rows


def parse_warmup(text, catalog, jobs):
    summaries = SUMMARY.findall(text)
    require(len(summaries) == 1, "Missing or duplicated warmup summary")
    count, hits, failures, suffix = summaries[0]
    require(not suffix and int(count) == len(catalog), "Warmup did not execute the complete catalog")
    require(int(hits) == 0, "Warmup used existing cache hits; this is not a cold-cache measurement")
    require(int(failures) == 0, "Warmup reported compilation/read-back failures")
    workers = re.findall(r"^Shader warmup workers: (\d+)\s*$", text, re.M)
    require(len(workers) == 1 and int(workers[0]) == min(jobs, len(catalog)), "Unexpected warmup worker count")
    processing = re.findall(r"^\s*Processing (\d+)/(\d+): (\S+)(?: \[resource-stride=\d+, sampler-stride=\d+\])?\s*$", text, re.M)
    require(sorted(int(row[0]) for row in processing) == list(range(1, len(catalog) + 1)) and
            all(int(total) == len(catalog) and request == catalog[int(index) - 1]["request"]
                for index, total, request in processing), "Measured requests differ from the full catalog")
    rows = [dict(index=int(match[0]), **dict(zip(("total", "setup", "module", "codegen"), map(float, match[1:]))))
            for match in TIMING.findall(text)]
    require(sorted(row["index"] for row in rows) == list(range(1, len(catalog) + 1)), "Missing or duplicated request phase timings")
    for row in rows:
        require(all(math.isfinite(row[key]) and row[key] >= 0 for key in ("total", "setup", "module", "codegen")), "Invalid request phase timing")
        require(row["setup"] + row["module"] + row["codegen"] <= row["total"] + 0.151, "Request phase times exceed total")
    elapsed = ELAPSED.findall(text)
    require(len(elapsed) == 1, "Missing or duplicated summed warmup phase timing")
    reported = dict(zip(("warmup_internal", "total", "module", "codegen"), map(float, elapsed[0])))
    require(all(math.isfinite(value) and value >= 0 for value in reported.values()), "Invalid summed phase timing")
    sums = {key: math.fsum(row[key] for row in rows) for key in ("total", "setup", "module", "codegen")}
    for key in ("total", "module", "codegen"):
        require(abs(sums[key] - reported[key]) <= 0.051 * (len(catalog) + 1), "Summed phase timing differs from request timings")
    return dict(requests=int(count), existing_cache_hits=0, failures=0, workers=int(workers[0]),
                request_phases_seconds=sorted(rows, key=lambda row: row["index"]),
                summed_phases_seconds=sums, reported_summary_seconds=reported,
                phase_scope="Rounded console seconds summed across overlapping workers; not end-to-end wall time")


def aggregate(runs, threshold):
    values = [run["process"]["process_wall_seconds"] for run in runs]
    return dict(count=len(values), min_seconds=min(values), median_seconds=statistics.median(values),
                max_seconds=max(values), goal_reached=all(value < threshold for value in values),
                acceptance="Every independent complete-process wall time must be strictly below the threshold")


def run(args):
    executable = args.executable.resolve(strict=True)
    require(executable.is_file(), "Executable must be a file")
    output = args.output_dir.resolve()
    require(not output.exists(), "Output directory must be new, including empty directories")
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / Path(__file__).name)
    environment = dict(os.environ)
    require(environment.get("METALLIC_SLANG_OPTIMIZATION") in (None, "", "default", "O1", "o1"),
            "Complete warmup goal requires inherited O1; remove the non-O1 override without changing compiler policy")
    executable_sha256 = digest(executable)
    summary = dict(schema_version=1, status="running", goal_reached=False, executable=str(executable),
                   jobs=args.jobs, repeats=args.repeats, threshold_seconds=args.threshold_seconds, timeout_seconds=args.timeout,
                   environment=compiler_environment(environment), runs=[],
                   host=dict(platform=platform.platform(), python=sys.version),
                   preparation_scope="The excluded --list process prepares generated material sources. Only each timed process's SPIR-V cache is cold.",
                   measurement_scope=__doc__.strip())
    save(output / "summary.json", summary)
    try:
        # Listing prepares custom-material modules. Freeze the source inventory only
        # after preparation, so generated material includes are ordinary measured inputs.
        listing = run_process([str(executable), "--list"], environment, output / "catalog", args.timeout)
        summary["catalog_process"] = listing
        successful_process(listing)
        require(digest(executable) == executable_sha256, "Executable changed during catalog preparation")
        catalog = parse_catalog(Path(listing["stdout"]).read_text(encoding="utf-8", errors="replace"))
        save(output / "catalog.json", catalog)
        summary["catalog"] = catalog
        identity = input_identity(executable, exclude=output)
        summary["inputs"] = identity
        save(output / "summary.json", summary)
        for repeat in range(args.repeats):
            check_identity(identity, executable, exclude=output)
            cache = fresh_cache(output / f"cache-{repeat:02}")
            ensure_empty_cache(cache)
            command = [str(executable), "--cache-dir", str(cache), "--jobs", str(args.jobs)]
            process = run_process(command, environment, output / f"run-{repeat:02}", args.timeout)
            record = dict(repeat=repeat, cache_directory=str(cache), process=process)
            summary["runs"].append(record)
            save(output / "summary.json", summary)
            successful_process(process)
            check_identity(identity, executable, exclude=output)
            record.update(parse_warmup(Path(process["stdout"]).read_text(encoding="utf-8", errors="replace"), catalog, args.jobs))
            require(record["reported_summary_seconds"]["warmup_internal"] <= process["process_wall_seconds"] + 0.051,
                    "Internal warmup interval exceeds complete-process wall time")
            save(output / "summary.json", summary)
        summary["aggregate"] = aggregate(summary["runs"], args.threshold_seconds)
        summary.update(status="passed", goal_reached=summary["aggregate"]["goal_reached"])
    except (OSError, ValueError, RuntimeError) as error:
        summary.update(status="failed", goal_reached=False, error=str(error))
    save(output / "summary.json", summary)
    return summary, output


def positive_finite(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=positive_integer, required=True)
    parser.add_argument("--repeats", type=positive_integer, default=3)
    parser.add_argument("--threshold-seconds", type=positive_finite, default=10)
    parser.add_argument("--timeout", type=positive_finite, default=600)
    args = parser.parse_args(argv)
    try:
        summary, output = run(args)
        report = dict(status=summary["status"], goal_reached=summary["goal_reached"], summary=str(output / "summary.json"))
        if "aggregate" in summary:
            report.update(summary["aggregate"])
        if "error" in summary:
            report["error"] = summary["error"]
        print(json.dumps(report, allow_nan=False))
        return 0 if summary["status"] == "passed" and summary["goal_reached"] else 1
    except (OSError, ValueError) as error:
        print(json.dumps(dict(status="failed", goal_reached=False, error=str(error))))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
