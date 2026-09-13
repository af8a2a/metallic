"""Create reproducible thin glTF cook probes without copying MiniZorah's binary.

The distributed MiniZorah scene is a flat set of root nodes. This helper checks
that contract instead of silently dropping parent transforms on other scenes.
"""

import argparse
import copy
import json
from pathlib import Path


def write_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Asset/MiniZorah/zorah_main_public.v2.gltf"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    scene = json.loads(source.read_text(encoding="utf-8"))
    root_nodes = scene["scenes"][scene.get("scene", 0)]["nodes"]
    if set(root_nodes) != set(range(len(scene["nodes"]))) or any(n.get("children") for n in scene["nodes"]):
        raise ValueError("This probe helper requires MiniZorah's flat root-node scene")
    args.directory.mkdir(parents=True, exist_ok=True)
    probes = []
    for name, mesh_index in (("Small", 0), ("Largest", 2018), ("RepeatedFloor", 1949)):
        if mesh_index >= len(scene["meshes"]):
            raise ValueError(f"Expected MiniZorah mesh {mesh_index} is missing")
        probe = copy.deepcopy(scene)
        probe["meshes"] = [scene["meshes"][mesh_index]]
        probe["nodes"] = [dict(node, mesh=0) for node in scene["nodes"] if node.get("mesh") == mesh_index]
        probe["scenes"] = [{"nodes": list(range(len(probe["nodes"])))}]
        probe["scene"] = 0
        for buffer in probe["buffers"]:
            if "uri" in buffer:
                buffer["uri"] = (source.parent / buffer["uri"]).resolve(strict=True).as_posix()
        destination = args.directory / f"{name}.gltf"
        write_if_changed(destination, json.dumps(probe, separators=(",", ":")))
        primitives = probe["meshes"][0]["primitives"]
        triangles = sum(scene["accessors"][primitive["indices"]]["count"] // 3 for primitive in primitives)
        probes.append({"name": name, "meshIndex": mesh_index, "sourceTriangles": triangles,
                       "primitives": len(primitives), "primitiveInstances": len(primitives) * len(probe["nodes"]),
                       "path": str(destination.resolve())})
    write_if_changed(args.directory / "probes.json", json.dumps({"source": str(source), "probes": probes}, indent=2) + "\n")
    print(json.dumps(probes, indent=2))


if __name__ == "__main__":
    main()
