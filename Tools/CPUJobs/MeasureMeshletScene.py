"""Interleave legacy/shared-runtime full meshlet builds; reject unequal geometry.

This is a CPU-only generated geometry workload. It does not measure scene load,
render critical paths, GPU work or FPS. Each process retains its own warmup and
first-build records; only steady samples enter the summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone


CONFIGURATIONS = {
    "plain": (0, 0),
    "attributes": (1, 0),
    "seams": (1, 1),
}
BACKENDS = {
    "legacy": "legacy-independent-pool",
    "shared": "shared-task-runtime",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(samples: list[float], probability: float) -> float:
    """Nearest-rank quantile, including P50."""
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summary(samples: list[float]) -> dict:
    return {
        "samples": len(samples),
        "mean": statistics.mean(samples),
        "p50": quantile(samples, 0.50),
        "p95": quantile(samples, 0.95),
        "p99": quantile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
        "standardDeviation": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def parse_list(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or len(values) != len(set(values)) or min(values) < 1:
        raise argparse.ArgumentTypeError("Expected distinct positive integers separated by commas")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True, help="Shared-runtime scene benchmark executable")
    parser.add_argument("--legacy", type=Path, required=True, help="Legacy-adapter scene benchmark executable")
    parser.add_argument("--output", type=Path, required=True, help="Result directory; existing Raw.jsonl is not overwritten")
    parser.add_argument("--rounds", type=int, default=6, help="Interleaved process pairs per configuration")
    parser.add_argument("--cells", type=parse_list, default=[64, 128])
    parser.add_argument("--workers", type=parse_list, default=[1, 2, 4, 8])
    parser.add_argument("--configurations", default="plain,attributes,seams")
    parser.add_argument("--timeout", type=float, default=180, help="Seconds allowed per process")
    args = parser.parse_args()
    args.new = args.new.resolve(strict=True)
    args.legacy = args.legacy.resolve(strict=True)
    args.output = args.output.resolve()
    args.configurations = args.configurations.split(",")
    if args.rounds < 1 or args.timeout <= 0:
        parser.error("Rounds and timeout must be positive")
    if not args.configurations or any(name not in CONFIGURATIONS for name in args.configurations):
        parser.error("Configurations must be selected from plain,attributes,seams")
    if max(args.cells) > 512 or max(args.workers) > 64:
        parser.error("Cells must not exceed 512 and worker limits must not exceed 64")
    return args


def metadata(args: argparse.Namespace, repository: Path) -> dict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    sources = [
        "Tools/CPUJobs/MeshletSceneBench.cpp",
        "Tools/CPUJobs/LegacyMeshletSceneAdapter.cpp",
        "Source/Runtime/Scene/scene.cpp",
        "Source/Runtime/Scene/MeshletBuildParallel.cpp",
        "Source/Runtime/Task/TaskSystem.cpp",
        "Source/Runtime/Task/ParallelFor.cpp",
    ]
    return {
        "schema": 1,
        "startedUtc": datetime.now(timezone.utc).isoformat(),
        "repositoryRevision": revision,
        "sourceSHA256": {name: sha256(repository / name) for name in sources},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logicalProcessors": os.cpu_count(),
        "python": sys.version,
        "executables": {
            "shared": {"path": str(args.new), "sha256": sha256(args.new)},
            "legacy": {"path": str(args.legacy), "sha256": sha256(args.legacy)},
        },
        "rounds": args.rounds,
        "cells": args.cells,
        "workers": args.workers,
        "configurations": args.configurations,
        "protocol": {
            "candidateOrder": "AB/BA alternates with outer round and configuration",
            "workerOrder": "rotated/reversed between processes and within each executable",
            "processWarmupRounds": 1,
            "processSteadyRounds": 1,
            "firstBuild": "Serial reference runs first in every process; retained separately",
            "timing": "Full buildStreamMeshletsForPrimitive; input copy, hashing, runtime initialization and source generation excluded",
            "cache": "Each process starts a new runtime and adapter; allocator and OS caches warm during serial references/warmup",
            "correctness": "Full logical geometry/LOD hash must agree across every worker count, candidate, phase and repetition",
            "scope": "Generated meshlet/LOD workload, not production scene rendering or FPS",
            "quantiles": "Nearest rank; small sample P95/P99 commonly equal maximum, not a rare-tail estimate",
            "externalLoad": "Not controlled by this script; record other activity separately",
        },
    }


def run(args: argparse.Namespace) -> dict:
    repository = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "Raw.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw evidence: {raw_path}")
    meta = metadata(args, repository)
    (args.output / "Metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    hashes: dict[tuple, tuple] = {}
    steady: list[dict] = []
    run_ordinal = 0
    with raw_path.open("w", encoding="utf-8", newline="\n") as raw:
        for outer_round in range(args.rounds):
            configuration_order = list(args.configurations)
            offset = outer_round % len(configuration_order)
            configuration_order = configuration_order[offset:] + configuration_order[:offset]
            for configuration_index, configuration in enumerate(configuration_order):
                attributes, seams = CONFIGURATIONS[configuration]
                order = ["legacy", "shared"]
                if (outer_round + configuration_index) % 2:
                    order.reverse()
                worker_order = list(args.workers)
                offset = outer_round % len(worker_order)
                worker_order = worker_order[offset:] + worker_order[:offset]
                if outer_round % 2:
                    worker_order.reverse()
                for candidate in order:
                    process_dir = args.output / f"run-{run_ordinal:03}-{candidate}-{configuration}"
                    process_dir.mkdir()
                    samples_file = process_dir / "Samples.jsonl"
                    command = [
                        str(args.new if candidate == "shared" else args.legacy),
                        "--cells", ",".join(map(str, args.cells)),
                        "--workers", ",".join(map(str, worker_order)),
                        "--rounds", "1", "--warmup", "1",
                        "--attributes", str(attributes), "--seams", str(seams),
                        "--output", str(samples_file),
                    ]
                    record = {
                        "type": "invocation", "runOrdinal": run_ordinal,
                        "outerRound": outer_round, "candidate": candidate,
                        "configuration": configuration, "command": command,
                    }
                    raw.write(json.dumps(record) + "\n")
                    raw.flush()
                    print(f"[{run_ordinal + 1}/{args.rounds * len(args.configurations) * 2}] "
                          f"{candidate} {configuration} round={outer_round}", flush=True)
                    with (process_dir / "Console.log").open("w", encoding="utf-8") as log:
                        subprocess.run(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=args.timeout)
                    rows = [json.loads(line) for line in samples_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if not rows or rows[0].get("type") != "metadata" or rows[0].get("backend") != BACKENDS[candidate]:
                        raise ValueError(f"Incorrect executable/backend metadata in {samples_file}")
                    process_steady = 0
                    for row in rows:
                        row.update({key: record[key] for key in ("runOrdinal", "outerRound", "candidate", "configuration")})
                        if row["type"] == "sample":
                            if not row.get("correct") or row["geometryHash"] != row["serialReferenceHash"]:
                                raise ValueError(f"Failed serial/parallel geometry check in {samples_file}")
                            if row["wallNs"] <= 0:
                                raise ValueError(f"Nonpositive timing in {samples_file}")
                            key = (row["cells"], row["attributes"], row["seams"])
                            value = (row["geometryHash"], row["checksum"], row["lodLevels"],
                                     row["lodGroups"], row["lodClusters"], row["lodVertices"], row["lodTriangleBytes"])
                            if key in hashes and hashes[key] != value:
                                raise ValueError(f"Legacy/shared or repetition geometry mismatch for {key}")
                            hashes[key] = value
                            if row["phase"] == "steady":
                                process_steady += 1
                                steady.append(row)
                        raw.write(json.dumps(row) + "\n")
                    if process_steady != len(args.cells) * len(args.workers):
                        raise ValueError(f"Missing steady samples in {samples_file}")
                    raw.flush()
                    run_ordinal += 1
    groups: dict[tuple, list[dict]] = {}
    pairs: dict[tuple, dict] = {}
    for row in steady:
        key = (row["configuration"], row["cells"], row["requestedWorkers"], row["candidate"])
        groups.setdefault(key, []).append(row)
        pair = (row["configuration"], row["cells"], row["requestedWorkers"], row["outerRound"])
        if row["candidate"] in pairs.setdefault(pair, {}):
            raise ValueError(f"Duplicate steady sample for {pair}")
        pairs[pair][row["candidate"]] = row["wallNs"]
    results = []
    for (configuration, cells, workers, candidate), rows in sorted(groups.items()):
        timings = summary([row["wallNs"] for row in rows])
        results.append({
            "configuration": configuration, "cells": cells, "requestedWorkers": workers,
            "effectiveWorkerLimit": rows[0]["effectiveWorkerLimit"], "candidate": candidate,
            "wallNs": timings,
            "sourceTrianglesPerSecond": 2 * cells * cells * 1e9 / timings["mean"],
            "geometryHash": rows[0]["geometryHash"],
        })
    ratios: dict[tuple, list[float]] = {}
    for (configuration, cells, workers, _), values in sorted(pairs.items()):
        if set(values) != {"legacy", "shared"}:
            raise ValueError(f"Unpaired candidate samples for {(configuration, cells, workers)}")
        ratios.setdefault((configuration, cells, workers), []).append(values["legacy"] / values["shared"])
    result = {
        "schema": 1,
        "scope": meta["protocol"]["scope"],
        "quantiles": meta["protocol"]["quantiles"],
        "correctness": "All full logical geometry hashes agree across both executables and all worker counts/repetitions",
        "steadySampleCount": len(steady),
        "processCount": run_ordinal,
        "results": results,
        "pairedLegacyOverSharedWallRatio": [
            {"configuration": key[0], "cells": key[1], "requestedWorkers": key[2],
             "ratio": summary(values), "interpretation": ">1 means shared runtime used less build wall time"}
            for key, values in sorted(ratios.items())
        ],
    }
    (args.output / "Summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    try:
        outcome = run(arguments())
        print(json.dumps({"processCount": outcome["processCount"], "steadySampleCount": outcome["steadySampleCount"]}))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Meshlet scene measurement failed: {error}", file=sys.stderr)
        sys.exit(1)
