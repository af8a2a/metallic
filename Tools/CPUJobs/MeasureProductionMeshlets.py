"""Fresh-output production meshlet cooking A/B, with separate payload validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from urllib.parse import unquote, urlparse

from MeasureProductionCPUJobs import benchmark_environment, parse_workers, sha256
from ProcessResources import monitor_command


GEOMETRY_FIELDS = (
    "formatVersion", "cookRevision", "fileBytes", "primitives", "instances", "groups", "pages",
    "clustersAllLods", "trianglesAllLods", "levels", "storedPayloadBytes", "allPagesDeviceBytesAligned256",
    "maxPagePayloadBytes", "directoryBytes", "runtimeFrontierStateBytes", "terminalPages",
    "terminalPageCount", "terminalPageBytesAligned256", "minimumResidentBytesIncludingOneStreamPage",
    "terminalInstanceGroups", "terminalInstanceClusters", "rootCutAudit", "geometries",
)


def geometry_identity(manifest: dict) -> dict:
    if manifest.get("status") != "complete":
        raise ValueError("Cook/inspect manifest does not report complete")
    identity = {field: manifest[field] for field in GEOMETRY_FIELDS if field in manifest}
    for field in ("fileBytes", "primitives", "instances", "groups", "pages", "storedPayloadBytes"):
        if field not in identity:
            raise ValueError(f"Cook manifest omitted geometry identity field {field}")
    return identity


def source_dependencies(source: Path) -> list[Path]:
    paths = [source]
    if source.suffix.lower() == ".gltf":
        document = json.loads(source.read_text(encoding="utf-8-sig"))
        for buffer in document.get("buffers", []):
            uri = buffer.get("uri", "")
            parsed = urlparse(uri)
            if parsed.scheme == "data":
                continue
            if parsed.scheme or parsed.netloc:
                raise ValueError("Production geometry evidence requires local buffer dependencies")
            if uri:
                paths.append((source.parent / unquote(parsed.path)).resolve(strict=True))
    return list(dict.fromkeys(paths))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=parse_workers, default=[1, 2, 4, 8])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--sample-ms", type=float, default=100)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--compression", choices=["reference", "none", "byte-rle"], default="reference")
    parser.add_argument("--memory-mib", type=int, default=0)
    parser.add_argument("--validate-attributes", action="store_true")
    parser.add_argument("--keep-payloads", action="store_true", help="Retain every generated stream binary after full validation/hash capture")
    args = parser.parse_args()
    for key in ("shared", "legacy", "source"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.blocks < 1 or args.sample_ms < 10 or args.timeout <= 0 or args.memory_mib < 0:
        parser.error("Invalid blocks, polling interval, timeout or memory budget")
    return args


def one_process(args, repository, candidate, workers, block, ordinal, run_id):
    directory = args.output / f"run-{run_id:03}-{candidate}-w{workers}"
    directory.mkdir()
    payload = directory / "Cooked.meshstream.bin"
    report_path = directory / "Cook.json"
    executable = args.shared if candidate == "shared" else args.legacy
    command = [str(executable), "--source", str(args.source), "--output", str(payload),
               "--report", str(report_path), "--workers", str(workers), "--compression", args.compression,
               "--memory-mib", str(args.memory_mib)]
    environment, environment_record = benchmark_environment(False, False)
    invocation = {"runId": run_id, "candidate": candidate, "workers": workers, "block": block,
                  "ordinal": ordinal, "command": command, "freshOutput": not payload.exists(),
                  "environment": environment_record}
    if not invocation["freshOutput"]:
        raise FileExistsError("Cook output must be new; cached/resumed output is not a cold cook")
    (directory / "Invocation.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
    print(f"cook run={run_id} block={block} ordinal={ordinal} workers={workers} candidate={candidate}", flush=True)
    exit_code, resources = monitor_command(command, cwd=repository, environment=environment,
        console_path=directory / "CookConsole.log", resources_path=directory / "CookOSResources.jsonl",
        timeout_seconds=args.timeout, sample_ms=args.sample_ms)
    if exit_code:
        raise RuntimeError(f"Cook exited {exit_code}; partial/paused cooks are not accepted: {directory}")
    manifest = json.loads(report_path.read_text(encoding="utf-8-sig"))
    identity = geometry_identity(manifest)
    if manifest.get("maxWorkers") != workers:
        raise ValueError("Cooker worker limit differs from requested configuration")
    payload_hash = sha256(payload)
    validation_path = directory / "Validation.json"
    validation_command = [str(executable), "--inspect", "--validate-payloads", "--source", str(args.source),
                          "--output", str(payload), "--report", str(validation_path), "--workers", str(workers)]
    if args.validate_attributes:
        validation_command.append("--validate-attributes")
    (directory / "ValidationInvocation.json").write_text(json.dumps({"command": validation_command}, indent=2) + "\n", encoding="utf-8")
    validation_exit, validation_resources = monitor_command(validation_command, cwd=repository, environment=environment,
        console_path=directory / "ValidationConsole.log", resources_path=directory / "ValidationOSResources.jsonl",
        timeout_seconds=args.timeout, sample_ms=args.sample_ms)
    if validation_exit:
        raise RuntimeError(f"Cooked payload validation exited {validation_exit}: {directory}")
    validated = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if validated.get("payloadValidation") != "all-pages" or geometry_identity(validated) != identity:
        raise ValueError("All-page payload/manifest validation failed or changed geometry")
    if args.validate_attributes and validated.get("attributeValidation", {}).get("status") not in {
            "reference-quantized-source-match", "exact-source-match"}:
        raise ValueError("Source attribute equivalence was not established")
    if sha256(payload) != payload_hash:
        raise ValueError("Read-only validation changed the cooked stream")
    artifacts = {"manifest": report_path, "events": Path(str(report_path) + ".events.jsonl"),
                 "console": directory / "CookConsole.log", "osResources": directory / "CookOSResources.jsonl",
                 "validation": validation_path, "validationConsole": directory / "ValidationConsole.log",
                 "validationOSResources": directory / "ValidationOSResources.jsonl"}
    record = {"schema": 1, "status": "passed", "phase": "measured", "runId": run_id,
              "candidate": candidate, "workers": workers, "block": block, "ordinal": ordinal,
              "freshOutput": True, "cookElapsedSeconds": manifest.get("elapsedSecondsThisInvocation"),
              "processResources": resources, "validationProcessResources": validation_resources,
              "geometryIdentity": identity, "payloadSha256": payload_hash, "payloadBytes": payload.stat().st_size,
              "payloadValidation": validated["payloadValidation"],
              "attributeValidation": validated.get("attributeValidation"),
              "applicationPeakProcessCommitBytes": manifest.get("peakProcessCommitBytes"),
              "applicationPeakWorkingSetBytes": manifest.get("peakWorkingSetBytes"),
              "artifacts": {key: str(path.relative_to(args.output)) for key, path in artifacts.items()},
              "artifactSha256": {key: sha256(path) for key, path in artifacts.items()},
              "payloadRetained": args.keep_payloads or run_id == 0,
              "payloadPath": str(payload.relative_to(args.output))}
    if record["payloadRetained"]:
        record["artifacts"]["payload"] = str(payload.relative_to(args.output))
        record["artifactSha256"]["payload"] = payload_hash
    (directory / "ProcessSummary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def run(args):
    repository = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Refusing to mix/overwrite production cook evidence")
    dependencies = source_dependencies(args.source)
    source_hashes = {str(path): sha256(path) for path in dependencies}
    sources = ["Tools/CPUJobs/MeasureProductionMeshlets.py", "Tools/CPUJobs/ProcessResources.py",
               "Tools/CPUJobs/LegacyMeshletSceneAdapter.cpp", "Source/Tools/MeshletCookMain.cpp",
               "Source/Runtime/Scene/Scene.cpp", "Source/Runtime/Scene/MeshletBuildParallel.cpp",
               "Source/Runtime/Task/TaskSystem.cpp", "Source/Runtime/Task/ParallelFor.cpp"]
    metadata = {"schema": 1, "protocol": "metallic-production-cold-cook-abba-v1", "status": "running",
                "startedUtc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "workers": args.workers, "blocks": args.blocks, "compression": args.compression,
                "memoryMiB": args.memory_mib, "sampleMilliseconds": args.sample_ms,
                "sourceAssetSha256": source_hashes,
                "repositoryRevision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
                "sourceSha256": {name: sha256(repository / name) for name in sources},
                "executables": {name: {"path": str(path), "sha256": sha256(path)} for name, path in
                                (("legacy", args.legacy), ("shared", args.shared))},
                "conditions": {
                    "cold": "New unique output path for every process; no resume/cache-hit output. Source/OS caches are retained, not flushed cold.",
                    "order": "Independent ABBA/BAAB/ABBA blocks with rotating worker order; fixed sample budget.",
                    "timing": "Cook process wall/CPU and cooker elapsed include source decode, build, encode, checkpoint, runtime topology inspection and report work. Separate inspect/all-payload/source-attribute validation is excluded from cook timings and resources.",
                    "correctness": "Every output is fully payload-validated, checked against source dependency identity by the cooker, and whole-file SHA256 plus geometry manifest must match across all candidates/workers/repetitions.",
                    "storage": "By default retain the first validated reference stream, remove later matching run-owned payloads after cross-candidate validation/hash capture, and preserve every manifest/event/log/resource sample. Failed/mismatching outputs remain. --keep-payloads retains all binaries.",
                    "gpu": "CPU-only cooker; no GPU work or measured VRAM. Aligned device payload sizes in manifests are byte accounting, not VRAM measurements.",
                }}
    metadata_path = args.output / "Metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records = []
    reference = None
    try:
        with (args.output / "Raw.jsonl").open("w", encoding="utf-8", newline="\n") as raw:
            for block in range(args.blocks):
                order = ["legacy", "shared", "shared", "legacy"] if block % 2 == 0 else ["shared", "legacy", "legacy", "shared"]
                worker_order = args.workers[block % len(args.workers):] + args.workers[:block % len(args.workers)]
                for workers in worker_order:
                    for ordinal, candidate in enumerate(order):
                        record = one_process(args, repository, candidate, workers, block, ordinal, len(records))
                        identity = (record["payloadSha256"], record["geometryIdentity"])
                        if reference is None:
                            reference = identity
                        elif reference != identity:
                            raise ValueError("Production cooked output differs across candidates/workers/repetitions")
                        if not record["payloadRetained"]:
                            (args.output / record["payloadPath"]).unlink()
                        records.append(record)
                        raw.write(json.dumps(record, separators=(",", ":")) + "\n")
                        raw.flush()
        if source_hashes != {str(path): sha256(path) for path in dependencies}:
            raise ValueError("Production source dependencies changed during the experiment")
        comparisons = []
        for workers in args.workers:
            for block in range(args.blocks):
                rows = [row for row in records if row["workers"] == workers and row["block"] == block]
                arms = {name: [row["processResources"]["processWallNs"] for row in rows if row["candidate"] == name]
                        for name in ("legacy", "shared")}
                means = {name: statistics.mean(values) for name, values in arms.items()}
                comparisons.append({"workers": workers, "block": block, "legacyMeanWallNs": means["legacy"],
                                    "sharedMeanWallNs": means["shared"], "legacyOverSharedWallRatio": means["legacy"] / means["shared"]})
        summary = {"schema": 1, "status": "complete", "processes": records, "comparisons": comparisons,
                   "correctness": "All complete stream file hashes and geometry manifests match; every payload decoded successfully",
                   "conditions": metadata["conditions"]}
        (args.output / "Summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        metadata["status"] = "complete"
        metadata["processCount"] = len(records)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        result = run(arguments())
        print(json.dumps({"processes": len(result["processes"]), "blocks": len(result["comparisons"])}))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Production meshlet measurement failed: {error}", file=sys.stderr)
        sys.exit(1)
