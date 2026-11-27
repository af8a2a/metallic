#!/usr/bin/env python3
"""Add opaque mesh props to the cooked terrain sample's source glTF."""

import argparse
import json
import math
import os
import struct
from pathlib import Path
from urllib.parse import quote, unquote

from BuildTerrainMesh import TerrainBuildError, write_atomic


def box_geometry():
    positions = []
    normals = []
    indices = []
    for axis in range(3):
        for sign in (-1, 1):
            first = len(positions)
            normal = [0.0, 0.0, 0.0]
            normal[axis] = float(sign)
            for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                position = [0.0, 0.0, 0.0]
                position[axis] = 0.5 * sign
                position[(axis + 1) % 3] = u * sign
                position[(axis + 2) % 3] = v
                positions.append(position)
                normals.append(normal)
            indices.extend(first + corner for corner in (0, 1, 2, 0, 2, 3))
    return positions, normals, indices


def relative_path(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()


def append_camera(document: dict, camera: dict) -> None:
    import numpy as np

    eye = np.asarray(camera["eye"], dtype=np.float64)
    forward = np.asarray(camera["center"], dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    document["cameras"] = [{
        "name": "Terrain Survey Camera",
        "type": "perspective",
        "perspective": {
            "yfov": math.radians(camera["fovDegrees"]), "znear": camera["znear"], "zfar": camera["zfar"],
        },
    }]
    # The editor selects the source scene camera after loading the StreamAsset metadata.
    document["scenes"][0]["nodes"].append(len(document["nodes"]))
    document["nodes"].append({
        "name": "Terrain Survey Camera", "camera": 0,
        "matrix": [*right.tolist(), 0, *up.tolist(), 0, *(-forward).tolist(), 0, *eye.tolist(), 1],
    })


def build_sample(terrain_path: Path, scene_path: Path, graph_path: Path, output_path: Path,
                 weight_map: Path | None = None) -> None:
    document = json.loads(terrain_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for buffer in document["buffers"]:
        source = terrain_path.parent / unquote(buffer["uri"])
        buffer["uri"] = quote(relative_path(source, output_path.parent), safe="/")

    box_path = output_path.with_suffix(".bin")
    buffer_index = len(document["buffers"])
    payload = bytearray()

    def append_accessor(data, count, shape, component_type, target, **bounds):
        view = len(document["bufferViews"])
        document["bufferViews"].append({
            "buffer": buffer_index, "byteOffset": len(payload), "byteLength": len(data), "target": target,
        })
        payload.extend(data)
        accessor = len(document["accessors"])
        document["accessors"].append({
            "bufferView": view, "componentType": component_type, "count": count, "type": shape, **bounds,
        })
        return accessor

    positions, normals, indices = box_geometry()
    position = append_accessor(
        b"".join(struct.pack("<3f", *value) for value in positions), len(positions), "VEC3", 5126, 34962,
        min=[-0.5] * 3, max=[0.5] * 3,
    )
    normal = append_accessor(
        b"".join(struct.pack("<3f", *value) for value in normals), len(normals), "VEC3", 5126, 34962,
    )
    index = append_accessor(struct.pack("<36H", *indices), len(indices), "SCALAR", 5123, 34963)
    document["buffers"].append({"uri": quote(box_path.name), "byteLength": len(payload)})

    def append_box_mesh(name, color, metallic, roughness):
        material = len(document["materials"])
        document["materials"].append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*color, 1.0], "metallicFactor": metallic, "roughnessFactor": roughness,
            },
        })
        mesh = len(document["meshes"])
        document["meshes"].append({
            "name": name,
            "primitives": [{"attributes": {"POSITION": position, "NORMAL": normal}, "indices": index, "material": material}],
        })
        return mesh

    if len(document["nodes"]) == 1:
        document["nodes"][0]["name"] = "Houdini Heightfield"
    terrain_material = document["materials"][0]
    terrain_material["name"] = "Terrain Grass"
    if weight_map is not None:
        from BuildTerrainLayerTextures import build_layer_material

        extras = document["extras"]
        terrain_metadata = extras.get("metallicTerrainTiles") or extras["metallicTerrainP0"]
        binding, metadata = build_layer_material(output_path, weight_map, terrain_metadata)
        terrain_material["name"] = binding["sourceName"]
        extras["metallicTerrainLayers"] = metadata
        scene.setdefault("materials", []).append(binding)
    # Grazing light can reach only the reverse face of an open heightfield.
    terrain_material["doubleSided"] = True
    tower = append_box_mesh("Painted Survey Tower", (0.84, 0.24, 0.06), 0.1, 0.65)
    bridge = append_box_mesh("Blue Bridge Steel", (0.30, 0.62, 0.82), 0.25, 0.48)
    for name, mesh, translation, scale in (
        ("Survey Tower", tower, [34, 12, -27], [10, 34, 10]),
        ("Bridge Deck", bridge, [-51.5, 3.5, 6], [17, 3, 60]),
        ("Bridge South Pier", bridge, [-51.5, -12, 27], [6, 29, 6]),
        ("Bridge North Pier", bridge, [-51.5, -12, -13], [6, 29, 6]),
    ):
        document["scenes"][0]["nodes"].append(len(document["nodes"]))
        document["nodes"].append({"name": name, "mesh": mesh, "translation": translation, "scale": scale})
    document["scenes"][0]["name"] = "GPUDriven Terrain VSM"
    document["asset"]["generator"] = "Metallic Tools/BuildTerrainVSMSample.py"
    append_camera(document, json.loads(graph_path.read_text(encoding="utf-8"))["view"]["camera"])

    scene["source"] = output_path.name
    environment = scene["world"]["environment"]
    environment["path"] = relative_path(scene_path.parent / environment["path"], output_path.parent)
    write_atomic(box_path, bytes(payload))
    write_atomic(output_path.with_suffix(".metallic_scene.json"),
                 (json.dumps(scene, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    write_atomic(output_path, (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terrain", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weight-map", type=Path, help="Linear R OpenEXR rock weights aligned to the heightfield")
    args = parser.parse_args()
    try:
        build_sample(args.terrain, args.scene, args.graph, args.output, args.weight_map)
    except TerrainBuildError as error:
        parser.error(str(error))
