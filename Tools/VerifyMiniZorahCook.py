"""Cross-check a completed MiniZorah cook manifest against its source glTF.

The cooker validates binary directories, topology and every payload. This check
independently verifies source coverage and instancing without decoding the 10 GB
source binary. It also works on PrepareMiniZorahCook.py's flat probe scenes.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(source: Path, manifest: Path, events: Path | None) -> dict:
    source_bytes = source.read_bytes()
    source_data = json.loads(source_bytes)
    manifest_bytes = manifest.read_bytes()
    report = json.loads(manifest_bytes)
    require(report["status"] == "complete", "Manifest does not describe a completed cook")
    require(report["payloadValidation"] == "all-pages", "Every payload must be validated by the cooker")
    require(Path(report["source"]).resolve() == source, "Manifest source path mismatch")
    asset = Path(report["asset"])
    require(asset.is_file() and asset.stat().st_size == report["fileBytes"], "Cache size mismatch")
    require(not Path(str(asset) + ".partial").exists(), "A partial checkpoint still exists")

    roots = source_data["scenes"][source_data.get("scene", 0)]["nodes"]
    nodes = source_data["nodes"]
    require(len(roots) == len(nodes) and set(roots) == set(range(len(nodes)))
            and not any(node.get("children") for node in nodes),
            "This verifier requires MiniZorah's flat root-node scene")
    mesh_instances = Counter(node["mesh"] for node in nodes if "mesh" in node)
    expected = []
    for mesh_index, mesh in enumerate(source_data["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            require(primitive.get("mode", 4) == 4 and "indices" in primitive,
                    "Expected an indexed triangle primitive")
            index_count = source_data["accessors"][primitive["indices"]]["count"]
            require(index_count % 3 == 0, "Source triangle index count is invalid")
            expected.append({"mesh": mesh_index, "primitive": primitive_index,
                             "triangles": index_count // 3,
                             "vertices": source_data["accessors"][primitive["attributes"]["POSITION"]]["count"],
                             "instances": mesh_instances[mesh_index]})
    geometries = report["geometries"]
    require(report["primitives"] == len(geometries) == len(expected), "Primitive coverage mismatch")
    require({g["primitive"] for g in geometries} == set(range(len(expected))), "Duplicate cooked primitive")
    require({g["sourcePrimitive"] for g in geometries} == set(range(len(expected))),
            "Source primitive mapping is incomplete or duplicated")
    for geometry in geometries:
        index = geometry["sourcePrimitive"]
        require(geometry["instances"] == expected[index]["instances"],
                f"Source primitive {index}: instance count mismatch")
        require(geometry["terminalGroups"] > 0, f"Source primitive {index}: no terminal groups")
    source_triangles = sum(p["triangles"] for p in expected)
    instance_count = sum(p["instances"] for p in expected)
    require(report["instances"] == instance_count, "Total instance count mismatch")
    require(report["levels"]["0"]["triangles"] == source_triangles, "Leaf triangle total differs from source")
    require(len(set(report["terminalPages"])) == report["terminalPageCount"], "Terminal page set is invalid")
    require(sum(g["terminalGroups"] for g in geometries) == report["terminalPageCount"],
            "Terminal group/page count mismatch")

    if events is not None:
        completed = {}
        for line in events.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["phase"] == "complete":
                completed[event["sourcePrimitive"]] = event
        require(set(completed) == set(range(len(expected))), "Cook event coverage mismatch")
        for index, primitive in enumerate(expected):
            for key in ("mesh", "primitive", "triangles", "vertices"):
                require(completed[index][key] == primitive[key],
                        f"Source primitive {index}: event {key} mismatch")

    summary = {key: value for key, value in report.items() if key not in ("geometries", "terminalPages")}
    summary.update({"source": str(source), "manifest": str(manifest),
                    "sourceGltfSha256": hashlib.sha256(source_bytes).hexdigest(),
                    "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "sourceTriangles": source_triangles,
                    "instancedSourceTriangles": sum(p["triangles"] * p["instances"] for p in expected),
                    "sourceCoverageValidation": "all-primitives-and-instance-counts",
                    "cookEventValidation": "all-source-primitives" if events else "not-requested",
                    "meshoptTemporaryCacheRemoved": not Path(str(asset) + ".meshopt-cache").exists()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source.resolve(strict=True), args.manifest.resolve(strict=True), args.events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {result['primitives']:,} primitives, {result['instances']:,} instances, "
          f"{result['sourceTriangles']:,} source triangles; report={args.output}")


if __name__ == "__main__":
    main()
