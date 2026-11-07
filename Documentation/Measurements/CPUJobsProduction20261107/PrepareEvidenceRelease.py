"""Preserve original production evidence and prepare a versioned release asset.

This is offline packaging only. It does not run the renderer, tests or benchmarks.
The source Data tree is read-only; use --compact-output for a separate light tree.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def fresh_directory(path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.mkdir(parents=True)


def compact_record(record, raw_line):
    scalar_fields = ("runId", "candidate", "workers", "phase", "block", "ordinal", "status",
        "purpose", "diagnostics", "frameCount", "measuredFrameRange", "frameMs", "gpuFrameMs",
        "cookElapsedSeconds", "freshOutput", "payloadSha256", "payloadBytes", "payloadValidation",
        "applicationPeakProcessCommitBytes", "applicationPeakWorkingSetBytes", "missing")
    result = {key: record[key] for key in scalar_fields if key in record}
    for key in ("processResources", "validationProcessResources"):
        if key in record:
            result[key] = {name: value for name, value in record[key].items() if name != "definitions"}
    if "deviceMemory" in record:
        result["deviceMemory"] = {key: value for key, value in record["deviceMemory"].items() if key != "definitions"}
    if "validationMessages" in record:
        result["validationMessages"] = record["validationMessages"]
    if "attributeValidation" in record:
        result["attributeValidationStatus"] = (record.get("attributeValidation") or {}).get("status")
    if "geometryIdentity" in record:
        identity = json.dumps(record["geometryIdentity"], sort_keys=True, separators=(",", ":")).encode()
        result["geometryIdentitySHA256"] = hashlib.sha256(identity).hexdigest()
    result["artifacts"] = record.get("artifacts", {})
    result["artifactSha256"] = record.get("artifactSha256", {})
    result["originalRawLineSHA256"] = hashlib.sha256(raw_line).hexdigest()
    return result


def compact_bundle(source, destination):
    metadata_path = source / "Metadata.json"
    if metadata_path.exists():
        shutil.copyfile(metadata_path, destination / "Metadata.json")
    records = []
    raw = source / "Raw.jsonl"
    if raw.exists():
        with raw.open("rb") as stream:
            for line in stream:
                if line.strip():
                    records.append(compact_record(json.loads(line), line.rstrip(b"\r\n")))
    original_summary = source / "Summary.json"
    summary = json.loads(original_summary.read_text(encoding="utf-8-sig")) if original_summary.exists() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    result = {"schema": "metallic-production-compact-summary-v1", "bundle": source.name,
        "sourceStatus": metadata.get("status", summary.get("status")),
        "definition": "Per-run scalar/aggregate subset of original records; not the original Raw/Summary and not sufficient for frame-level replotting. Restore the release asset for full data.",
        "originalRawLineHashDefinition": "SHA256 of each original UTF-8 JSONL record excluding its line terminator",
        "originalFiles": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (metadata_path, raw, original_summary) if path.exists()},
        "population": {"completeRecords": len(records), "measuredRecords": sum(row.get("phase") == "measured" for row in records),
            "prewarmRecords": sum(row.get("phase") == "prewarm" for row in records)},
        "processes": records, "comparisons": summary.get("comparisons", [])}
    for key in ("userBudgetRevision", "conditions", "definitions", "correctness"):
        if key in summary:
            result[key] = summary[key]
    write_json(destination / "CompactSummary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-output", required=True, type=Path)
    args = parser.parse_args()
    source, output, compact = (path.resolve() for path in (args.source, args.output, args.compact_output))
    if not source.is_dir() or source == output or source in output.parents or source == compact or source in compact.parents:
        raise ValueError("Outputs must be separate from the source evidence tree")
    fresh_directory(output)
    fresh_directory(compact)
    backup = output / "OriginalData"
    shutil.copytree(source, backup)
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            sha = digest(path)
            if digest(backup / relative) != sha:
                raise ValueError(f"Backup hash mismatch: {relative}")
            files.append({"path": "Data/" + relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha})
    original = json.loads((backup / "Evidence.json").read_text(encoding="utf-8-sig"))
    for bundle in original["bundles"]:
        directory = backup / bundle["name"]
        if digest(directory / "Runs.zip") != bundle["archiveSHA256"]:
            raise ValueError(f"Original run archive hash mismatch: {bundle['name']}")
        for record in bundle["records"]:
            if record["storage"] == "copied" and digest(directory / record["path"]) != record["sha256"]:
                raise ValueError(f"Original record hash mismatch: {bundle['name']}/{record['path']}")
    asset_name = "metallic-cpu-jobs-production-20261107.zip"
    asset = output / asset_name
    with zipfile.ZipFile(asset, "x", compression=zipfile.ZIP_STORED) as archive:
        for record in files:
            archive.write(backup / Path(record["path"]).relative_to("Data"), record["path"])
    with zipfile.ZipFile(asset) as archive:
        if archive.testzip() is not None:
            raise ValueError("Release asset CRC check failed")
        for record in files:
            with archive.open(record["path"]) as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha != record["sha256"]:
                raise ValueError(f"Release asset record hash mismatch: {record['path']}")
    tag = "pr-74-cpu-jobs-evidence-20261107"
    release = {"repository": "af8a2a/metallic", "tag": tag, "asset": asset_name,
        "url": f"https://github.com/af8a2a/metallic/releases/download/{tag}/{asset_name}",
        "bytes": asset.stat().st_size, "sha256": digest(asset)}
    manifest = {"schema": "metallic-production-release-evidence-v1", "createdUtc": datetime.now(timezone.utc).isoformat(),
        "release": release, "archiveLayout": "Data/<bundle>/...; inner Runs.zip archives and original manifests are unchanged",
        "originalEvidenceManifestSHA256": digest(backup / "Evidence.json"), "files": files,
        "verification": "Complete source-to-backup hashes, original copied-record/archive hashes, outer ZIP CRC and all outer member SHA256s matched",
        "retention": "Versioned evidence prerelease assets have no automatic expiry. Retain while this PR/report is referenced; deletion requires a maintainer decision. The ignored local OriginalData backup is an additional copy, not the published retention mechanism."}
    write_json(output / "ReleaseManifest.json", manifest)
    (output / "SHA256SUMS").write_text(release["sha256"] + "  " + asset_name + "\n", encoding="ascii", newline="\n")
    shutil.copyfile(backup / "Evidence.json", compact / "Evidence.json")
    write_json(compact / "Storage.json", manifest)
    for directory in sorted(backup.iterdir()):
        if directory.is_dir():
            target = compact / directory.name
            target.mkdir()
            compact_bundle(directory, target)
    print(json.dumps({"originalBytes": sum(row["bytes"] for row in files), "files": len(files),
        "releaseAsset": str(asset), "assetBytes": release["bytes"], "assetSHA256": release["sha256"],
        "compactBytes": sum(path.stat().st_size for path in compact.rglob("*") if path.is_file()),
        "backup": str(backup)}, indent=2))


if __name__ == "__main__":
    main()
