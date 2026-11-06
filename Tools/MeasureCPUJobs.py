"""Run interleaved CPU-only scheduling comparisons and retain every sample."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import statistics
import subprocess


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "p50": values[int(.50 * (len(values) - 1))],
        "p95": values[int(.95 * (len(values) - 1))],
        "p99": values[int(.99 * (len(values) - 1))],
        "min": values[0], "max": values[-1],
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = "/".join(str(row[key]) for key in
                       ("domain", "workload", "backend", "workers", "grain", "diagnostics", "cache"))
        groups[key].append(row)
    result = {}
    for key, samples in sorted(groups.items()):
        first = samples[0]
        fields = {field: first[field] for field in
                  ("domain", "backend", "workload", "workers", "grain", "diagnostics", "cache", "elements")}
        result[key] = {
            **fields,
            "wallMs": distribution([row["wallNs"] * 1.e-6 for row in samples]),
            "throughputElementsPerSecond": distribution([
                row["elements"] * 1.e9 / row["wallNs"] for row in samples]),
            "processCpuCores": distribution([row["processCpuNs"] / row["wallNs"] for row in samples]),
            "workerActiveMs": distribution([row["activeNs"] * 1.e-6 for row in samples
                                             if row["workerActiveNs"]]),
            "queueP50Us": distribution([row["queueP50Ns"] * .001 for row in samples
                                         if row["queueP50Ns"] is not None]),
            "queueP95Us": distribution([row["queueP95Ns"] * .001 for row in samples
                                         if row["queueP95Ns"] is not None]),
            "queueP99Us": distribution([row["queueP99Ns"] * .001 for row in samples
                                         if row["queueP99Ns"] is not None]),
            "claimAttempts": distribution([row["claimAttempts"] for row in samples
                                             if row.get("claimAttempts") is not None]),
            "stealAttempts": distribution([row["stealAttempts"] for row in samples
                                             if row.get("stealAttempts") is not None]),
            "stealSuccesses": distribution([row["stealSuccesses"] for row in samples
                                              if row.get("stealSuccesses") is not None]),
        }
    return result


def revision(root, path=None):
    command = ["git", "-C", str(root)]
    command += ["rev-parse", "HEAD"] if path is None else ["ls-tree", "HEAD", path]
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--grain", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--mode", choices=["baseline", "range", "all"], default="all")
    parser.add_argument("--workloads", choices=["all", "uniform", "random", "long_tail", "bandwidth", "locality"],
                        nargs="+", default=["all"])
    parser.add_argument("--diagnostics", choices=["off", "on", "both"], default="both")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if any(value <= 0 for value in [*args.workers, args.repeats, args.count, args.grain, args.tasks]):
        parser.error("counts must be positive")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": 1,
        "capturedUtc": datetime.now(timezone.utc).isoformat(),
        "machine": platform.node(), "os": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logicalProcessors": os.cpu_count(),
        "repositoryRevision": revision(root),
        "stdexecLockedRevision": revision(root, "External/stdexec"),
        "oneTBBLockedRevision": revision(root, "External/oneTBB"),
        "executable": str(args.executable.resolve()),
        "executableSha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "workingTreeDiffSha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"])).hexdigest(),
        "status": "running",
        "protocol": {
            "workerOrder": args.workers, "repeats": args.repeats,
            "computeElements": args.count, "bandwidthMinElements": 16 * 1024 * 1024,
            "grain": args.grain, "dagLeaves": args.tasks,
            "mode": args.mode, "diagnostics": args.diagnostics,
            "workloads": args.workloads,
            "firstCall": "First invocation per case after worker construction; input was initialized and serial-validated. Not a flushed-cache measurement.",
            "runtimeConstruction": "Raw records retain TaskSystem, raw stdexec and legacy worker-pool construction time once per benchmark process, separately from all task/range timings.",
            "warm": "Repeated same buffers; deterministic per-round rotation across candidates; no explicit cache flush.",
            "timedScope": "Scheduling+kernel+join, including per-call state allocation. TaskGraph graph construction is excluded; submit validation and completion snapshot are included.",
            "legacy": "Issue-start MeshletBuildParallel snapshot at 8e0d94587; one atomic claim per batch, caller participates (workers-1 background threads).",
            "runtime": "Unified range/TaskGraph use workers background stdexec threads and external caller waits. Other comparison pools remain idle.",
            "validation": "No Vulkan initialization or validation. Each sample checksum compared with fixed-seed serial reference.",
            "processCpu": "GetProcessTimes on Windows (100ns units, OS accounting quantization). Cores=process CPU time / wall, not system-wide utilization.",
            "instrumentation": "off disables optional benchmark/range timestamps; TaskGraph always retains its existing node snapshot clocks and state transitions. Graph active time includes callback dispatch/finish; range/legacy active time is kernel callbacks. Do not subtract these as equivalent domains.",
            "quantiles": "Sorted samples, index floor(q*(n-1)); 20 repeats do not robustly estimate production P99.",
            "unavailable": ["upstream stdexec/oneTBB steal attempts/successes", "kernel wakeup delay distinct from queue delay", "hardware SIMD utilization", "atomic operation time separate from useful work", "system-wide CPU utilization", "real-scene FPS/critical path"],
        },
        "commands": [],
    }
    rows = []
    diagnostic_modes = ["off", "on"] if args.diagnostics == "both" else [args.diagnostics]
    raw_path = args.output / "Raw.jsonl"
    summary_path = args.output / "Summary.json"
    summary_path.unlink(missing_ok=True)
    metadata_path = args.output / "Metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw:
        for diagnostics in diagnostic_modes:
            for workers in args.workers:
                for workload in args.workloads:
                    command = [str(args.executable.resolve()), "--workers", str(workers),
                               "--repeats", str(args.repeats), "--count", str(args.count),
                               "--grain", str(args.grain), "--tasks", str(args.tasks),
                               "--mode", args.mode, "--diagnostics", diagnostics, "--workload", workload]
                    metadata["commands"].append(command)
                    print(f"Measuring workers={workers} diagnostics={diagnostics} workload={workload}", flush=True)
                    completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
                    if completed.returncode:
                        raise RuntimeError(f"Benchmark failed ({completed.returncode}): {completed.stderr}")
                    parsed = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
                    if not parsed:
                        raise RuntimeError("Benchmark produced no samples")
                    for row in parsed:
                        if row["wallNs"] <= 0 or row["elements"] <= 0:
                            raise ValueError("Invalid timing/sample")
                        raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                    raw.flush()
                    rows.extend(parsed)
    metadata["status"] = "complete"
    metadata["sampleCount"] = len(rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summarize(rows), indent=2) + "\n", encoding="utf-8")
    print(f"Retained {len(rows)} samples in {args.output}", flush=True)


if __name__ == "__main__":
    main()
