"""Package original acceptance records in local/release staging, without rewriting inputs.

Use an ignored build-* output. The repository Data tree stores compact catalogs;
PrepareEvidenceRelease.py preserves full packages for versioned release storage.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New local/release staging directory, normally under ignored build-*")
    parser.add_argument("--input", action="append", required=True, help="BundleName=original-directory")
    args = parser.parse_args()
    output = args.output.resolve()
    source_catalog = Path(__file__).resolve().parent / "Data"
    if output == source_catalog or source_catalog in output.parents:
        raise ValueError("Data is a lightweight source catalog; package raw records into an ignored build-* staging directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    inputs = []
    for item in args.input:
        name, source = item.split("=", 1)
        if not name or any(not (character.isalnum() or character in "-_") for character in name):
            raise ValueError(f"Invalid bundle name: {name}")
        source = Path(source).resolve(strict=True)
        if not source.is_dir() or output == source or source in output.parents:
            raise ValueError("Evidence input must be a separate, existing directory")
        inputs.append((name, source))
    if len({name for name, _ in inputs}) != len(inputs):
        raise ValueError("Duplicate bundle name")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"createdUtc": datetime.now(timezone.utc).isoformat(), "bundles": [],
                "definition": "Original bytes; summaries copied, other records zipped. Generated meshstream payload binaries stay in original build output with recorded hashes; source assets are not copied."}
    for name, source in inputs:
        destination = output / name
        destination.mkdir()
        archive = destination / "Runs.zip"
        records = []
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_LZMA) as zipped:
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                record = {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                if path.name.endswith(".meshstream.bin"):
                    record["storage"] = "original build output only; generated payload hash retained"
                elif len(relative.parts) == 1 and path.name in {"Metadata.json", "Raw.jsonl", "Summary.json"}:
                    shutil.copyfile(path, destination / path.name)
                    record["storage"] = "copied"
                else:
                    zipped.write(path, relative.as_posix())
                    record["storage"] = "Runs.zip"
                records.append(record)
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError(f"Archive CRC failure: {archive}")
        manifest["bundles"].append({"name": name, "originalDirectory": str(source), "records": records,
                                    "archiveCompression": "ZIP_LZMA; extract with Python zipfile",
                                    "archiveBytes": archive.stat().st_size, "archiveSHA256": sha256(archive)})
        print(f"Packaged {name}: {len(records)} files, {archive.stat().st_size} archive bytes", flush=True)
    (output / "Evidence.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
