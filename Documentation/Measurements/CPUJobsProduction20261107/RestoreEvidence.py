"""Restore the immutable release evidence for offline replotting; runs no workload."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Downloaded versioned release ZIP")
    parser.add_argument("--output", required=True, type=Path, help="New empty restoration directory")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "Data/Storage.json")
    parser.add_argument("--bundle", action="append", help="Also expand a named bundle's original Runs.zip; repeat as needed")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    archive, output = args.archive.resolve(strict=True), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite restored records: {output}")
    if archive.stat().st_size != manifest["release"]["bytes"] or digest(archive) != manifest["release"]["sha256"]:
        raise ValueError("Release asset size/SHA256 differs from the repository catalog")
    expected = {record["path"]: record for record in manifest["files"]}
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(expected):
            raise ValueError("Release asset member list differs from the verified catalog")
        output.mkdir(parents=True, exist_ok=True)
        source.extractall(output)
    for name, record in expected.items():
        path = output / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Restored file does not match original bytes: {name}")
    original = json.loads((output / "Data/Evidence.json").read_text(encoding="utf-8-sig"))
    bundles = {bundle["name"]: bundle for bundle in original["bundles"]}
    for name in args.bundle or []:
        if name not in bundles:
            raise ValueError(f"Unknown bundle: {name}")
        directory = output / "Data" / name
        bundle = bundles[name]
        if digest(directory / "Runs.zip") != bundle["archiveSHA256"]:
            raise ValueError(f"Original run archive hash mismatch: {name}")
        with zipfile.ZipFile(directory / "Runs.zip") as source:
            source.extractall(directory)
        for record in bundle["records"]:
            if record["storage"] in {"copied", "Runs.zip"}:
                path = directory / record["path"]
                if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"Original record mismatch: {name}/{record['path']}")
        print(f"Restored and verified {name}")
    print(f"Original evidence restored under {output / 'Data'}; no renderer or benchmark launched")


if __name__ == "__main__":
    main()
