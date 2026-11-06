#!/usr/bin/env python3
"""Capture isolated single-thread kernel and repacking CSV samples."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_identity():
    identifier = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unavailable"
    source = "PROCESSOR_IDENTIFIER/platform.processor"
    if os.name == "nt":
        try:
            import winreg
            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                identifier = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                source = "HKLM/" + path.replace("\\", "/") + "/ProcessorNameString"
        except OSError:
            pass
    return identifier, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--counts", nargs="+", type=int, default=[31, 65536, 1048576])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--compiler", default="unspecified")
    parser.add_argument("--ispc-version", default="unspecified")
    args = parser.parse_args()
    if min(args.counts + [args.runs, args.samples]) <= 0 or max(args.counts + [args.samples]) > 0xFFFFFFFF:
        parser.error("Counts and samples must fit positive uint32; runs must be positive")
    executables = []
    for specification in args.executable:
        label, separator, path = specification.partition("=")
        if not label or not separator or not Path(path).is_file():
            parser.error("Each --executable must be LABEL=an-existing-file")
        executables.append((label, str(Path(path).resolve())))
    args.output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    source_files = [repository / "cmake/CPUKernels.cmake"]
    for directory in [repository / "Source/Runtime/CPUKernels", repository / "tests/cpu"]:
        source_files += [path for path in directory.iterdir()
            if path.suffix in {".cpp", ".h", ".ispc", ".py"} or path.name == "CMakeLists.txt"]
    source_hashes = {str(path.relative_to(repository)).replace("\\", "/"): file_hash(path)
        for path in sorted(source_files)}
    executable_hashes = {label: file_hash(path) for label, path in executables}
    started = time.time()
    raw = []
    for run in range(args.runs):
        order = executables[run % len(executables):] + executables[:run % len(executables)]
        for count in args.counts:
            for label, executable in order:
                result = subprocess.run([executable, "--count", str(count), "--samples", str(args.samples)],
                    text=True, capture_output=True, check=True)
                for sample in csv.DictReader(io.StringIO(result.stdout)):
                    raw.append({"configuration": label, "run": run, **sample})
    for path, expected in source_hashes.items():
        if file_hash(repository / path) != expected:
            raise RuntimeError(f"Source changed during measurement: {path}; rerun the benchmark")
    for label, path in executables:
        if file_hash(path) != executable_hashes[label]:
            raise RuntimeError(f"Executable changed during measurement: {label}; rerun the benchmark")
    raw_fields = ["configuration", "run", "phase", "round", "backend", "count", "wall_ns", "checksum"]
    with (args.output / "Raw.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw)
    groups = {}
    for row in raw:
        key = tuple(row[field] for field in ["configuration", "phase", "backend", "count"])
        groups.setdefault(key, []).append(int(row["wall_ns"]))
    summary = []
    for (configuration, phase, backend, count), values in sorted(groups.items()):
        median = statistics.median(values)
        summary.append(dict(configuration=configuration, phase=phase, backend=backend, count=count,
            samples=len(values), mean_ns=statistics.mean(values), p50_ns=median,
            p95_ns=percentile(values, .95), p99_ns=percentile(values, .99),
            elements_per_second=int(count) * 1e9 / median))
    with (args.output / "Summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    cpu, cpu_source = cpu_identity()
    metadata = {
        "platform": platform.platform(), "machine": platform.machine() or "unavailable",
        "cpu": cpu, "cpu_identity_source": cpu_source,
        "logical_cpu_count": os.cpu_count(), "compiler": args.compiler, "ispc": args.ispc_version,
        "executables": dict(executables), "counts": args.counts, "runs": args.runs,
        "source_sha256": source_hashes, "executable_sha256": executable_hashes,
        "samples_per_process": args.samples, "untimed_warmup_rounds": 2,
        "start_unix_seconds": started, "elapsed_seconds": time.time() - started,
        "conditions": "One caller thread; rotated in-process backend and per-run executable order; "
            "deterministic seed 0x73b01; allocation/setup excluded. first_call is not a cache flush. "
            "Warm samples include no packing; aos-to-soa separately times packing. "
            "Caller must ensure other builds and CPU benchmarks are idle.",
        "unavailable_metrics": ["Hardware SIMD utilization", "CPU performance counters", "Scene frame time"],
    }
    (args.output / "Metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(raw)} raw samples to {args.output}")


if __name__ == "__main__":
    main()
