#!/usr/bin/env python3
"""Split tiled BuildTerrainMesh output into adjacent west and east glTF regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import unquote

from BuildTerrainMesh import TerrainBuildError, write_atomic
from BuildTerrainVSMSample import append_camera, relative_path


GENERATOR = "Metallic Tools/BuildTerrainRegions.py"


def write_json(path, document):
    write_atomic(path, (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def make_region(source, source_payload, terrain, side, tile_bounds):
    name = f"terrain_region_{side}"
    axis = terrain["tilesPerAxis"]
    x_begin, z_begin, x_end, z_end = tile_bounds
    payload = bytearray()
    views = []
    accessors = []
    meshes = []
    nodes = []
    tiles = []
    position_accessors = []

    def copy_accessor(index):
        accessor = source["accessors"][index]
        view = source["bufferViews"][accessor["bufferView"]]
        begin = view["byteOffset"]
        end = begin + view["byteLength"]
        if view["buffer"] != 0 or not 0 <= begin < end <= len(source_payload):
            raise TerrainBuildError(f"terrain accessor {index} references an invalid buffer range")
        payload.extend(b"\0" * (-len(payload) % 4))
        views.append({**view, "byteOffset": len(payload)})
        payload.extend(source_payload[begin:end])
        accessors.append({**accessor, "bufferView": len(views) - 1})
        return len(accessors) - 1

    for index in (z * axis + x for z in range(z_begin, z_end) for x in range(x_begin, x_end)):
        tile = terrain["tiles"][index]
        mesh = source["meshes"][index]
        (primitive,) = mesh["primitives"]
        if mesh["name"] != tile["name"] or not primitive["extras"]["METALLIC_lockMeshletLODBoundary"]:
            raise TerrainBuildError(f"expected a boundary-locked mesh for terrain tile {tile['tile']}")
        attributes = {key: copy_accessor(value) for key, value in primitive["attributes"].items()}
        position_accessors.append(accessors[attributes["POSITION"]])
        meshes.append({**mesh, "primitives": [{
            **primitive, "attributes": attributes, "indices": copy_accessor(primitive["indices"]),
        }]})
        nodes.append({"name": mesh["name"], "mesh": len(meshes) - 1})
        tiles.append(tile)

    minimum = [min(accessor["min"][component] for accessor in position_accessors) for component in range(3)]
    maximum = [max(accessor["max"][component] for accessor in position_accessors) for component in range(3)]
    metadata = {key: terrain[key] for key in (
        "source", "sourceSha256", "sourceChannel", "sourceDataWindow", "sourceResolution", "horizontalSize",
        "heightScale", "heightOffset", "skirtDepth", "flipZ", "tilesPerAxis",
    )}
    metadata.update({
        "region": side,
        "tileBounds": list(tile_bounds),
        "worldBounds": {"min": minimum, "max": maximum},
        "tiles": tiles,
        "exportedVertexCount": sum(tile["vertexCount"] for tile in tiles),
        "triangleCount": sum(tile["triangleCount"] for tile in tiles),
    })
    (material,) = source["materials"]
    gltf = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"name": name, "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        # Grazing VSM rays can enter the reverse face of an open heightfield.
        "materials": [{**material, "name": f"Terrain Region {side.title()}", "doubleSided": True}],
        "buffers": [{"uri": f"{name}.bin", "byteLength": len(payload)}],
        "bufferViews": views,
        "accessors": accessors,
        "extras": {"metallicTerrainTiles": metadata},
    }
    return gltf, payload


def combine_regions(sources, *, shared_x_split=False):
    gltf = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "scene": 0,
        "scenes": [{"name": "Terrain Regions", "nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
    }
    regions = []
    for source in sources:
        name = source["scenes"][0]["name"]
        terrain = source["extras"]["metallicTerrainTiles"]
        buffer_offset = len(gltf["buffers"])
        view_offset = len(gltf["bufferViews"])
        accessor_offset = len(gltf["accessors"])
        mesh_offset = len(gltf["meshes"])
        material_offset = len(gltf["materials"])

        # Each region has one primitive and one identity instance per tile.
        regions.append({
            "id": terrain["region"],
            "gltf": f"{name}.gltf",
            "streamAsset": f"{name}.gltf.meshstream.bin",
            "sourcePrimitiveOffset": mesh_offset,
            "sourceInstanceOffset": len(gltf["nodes"]),
            "materialOffset": material_offset,
            "primitiveCount": len(source["meshes"]),
            "instanceCount": len(source["nodes"]),
            "materialCount": len(source["materials"]),
            "worldBounds": terrain["worldBounds"],
        })
        gltf["buffers"].extend(source["buffers"])
        gltf["bufferViews"].extend({**view, "buffer": view["buffer"] + buffer_offset}
                                   for view in source["bufferViews"])
        gltf["accessors"].extend({**accessor, "bufferView": accessor["bufferView"] + view_offset}
                                 for accessor in source["accessors"])
        gltf["materials"].extend(source["materials"])
        for mesh in source["meshes"]:
            (primitive,) = mesh["primitives"]
            gltf["meshes"].append({**mesh, "primitives": [{
                **primitive,
                "attributes": {key: value + accessor_offset for key, value in primitive["attributes"].items()},
                "indices": primitive["indices"] + accessor_offset,
                "material": primitive["material"] + material_offset,
            }]})
        gltf["nodes"].extend({**node, "mesh": node["mesh"] + mesh_offset} for node in source["nodes"])

    gltf["scenes"][0]["nodes"] = list(range(len(gltf["nodes"])))
    terrain = {"regions": regions}
    if shared_x_split:
        split_x = regions[0]["worldBounds"]["max"][0]
        if split_x != regions[1]["worldBounds"]["min"][0]:
            raise TerrainBuildError("the west and east regions do not share a world-X boundary")
        terrain = {"splitX": split_x, **terrain}
    gltf["extras"] = {"metallicTerrainRegions": terrain}
    return gltf


def make_sample_scene(aggregate, aggregate_path, lighting, height_field):
    if lighting is None:
        scene = {"version": 3, "sceneIndex": 0, "nodes": []}
    else:
        scene = json.loads(lighting.read_text(encoding="utf-8"))
        environment = scene.get("world", {}).get("environment", {})
        if environment.get("path"):
            environment["path"] = relative_path(lighting.parent / environment["path"], aggregate_path.parent)
    scene["source"] = aggregate_path.name
    if height_field is not None:
        scene["terrain"] = {
            "heightField": relative_path(height_field, aggregate_path.parent),
            "cameraClearance": 2.0,
        }

    terrain = aggregate["extras"]["metallicTerrainRegions"]
    split_x = terrain["splitX"]
    camera_y = max(region["worldBounds"]["max"][1] for region in terrain["regions"]) + 24.0
    append_camera(aggregate, {
        "eye": [split_x - 76.0, camera_y, 37.0],
        "center": [split_x - 56.0, 1.0, 8.0],
        "up": [0.0, 1.0, 0.0],
        "fovDegrees": 38.0,
        "znear": 0.5,
        "zfar": 1000.0,
    })
    return scene


def build_regions(path: Path, directory: Path, lighting: Path | None = None,
                  height_field: Path | None = None) -> None:
    source = json.loads(path.read_text(encoding="utf-8"))
    if source["asset"].get("generator") != "Metallic Tools/BuildTerrainMesh.py":
        raise TerrainBuildError("expected the tiled height glTF from BuildTerrainMesh.py")
    terrain = source["extras"]["metallicTerrainTiles"]
    axis = terrain["tilesPerAxis"]
    tiles = terrain["tiles"]
    if axis < 2 or axis % 2 or len(source["meshes"]) != axis * axis or [tile["tile"] for tile in tiles] != [
        [x, z] for z in range(axis) for x in range(axis)
    ]:
        raise TerrainBuildError("expected a complete, evenly splittable row-major tile grid")

    (buffer,) = source["buffers"]
    binary = path.parent / unquote(buffer["uri"])
    payload = binary.read_bytes()
    if len(payload) != buffer["byteLength"]:
        raise TerrainBuildError(f"terrain buffer byte count mismatch: '{binary}'")

    middle = axis // 2
    west, west_payload = make_region(source, payload, terrain, "west", (0, 0, middle, axis))
    east, east_payload = make_region(source, payload, terrain, "east", (middle, 0, axis, axis))
    aggregate = combine_regions((west, east), shared_x_split=True)
    regions = ((west, west_payload), (east, east_payload))
    aggregate_path = directory / "terrain_regions.gltf"
    sidecar = aggregate_path.with_suffix(".metallic_scene.json")
    outputs = [aggregate_path]
    for region, _ in regions:
        name = region["scenes"][0]["name"]
        outputs.extend((directory / f"{name}.gltf", directory / region["buffers"][0]["uri"]))
    inputs = {path.resolve(), binary.resolve()}
    has_sidecar = lighting is not None or height_field is not None
    if has_sidecar:
        outputs.append(sidecar)
    if lighting is not None:
        inputs.add(lighting.resolve())
    if height_field is not None:
        if not height_field.is_file():
            raise TerrainBuildError(f"missing source terrain heightfield: '{height_field}'")
        inputs.add(height_field.resolve())
    if any(output.resolve() in inputs for output in outputs):
        raise TerrainBuildError("a region output would overwrite the source terrain")
    scene = make_sample_scene(aggregate, aggregate_path, lighting, height_field) if has_sidecar else None

    directory.mkdir(parents=True, exist_ok=True)
    for region, region_payload in regions:
        name = region["scenes"][0]["name"]
        write_atomic(directory / f"{name}.bin", bytes(region_payload))
        region_terrain = region["extras"]["metallicTerrainTiles"]
        gltf_path = directory / f"{name}.gltf"
        write_json(gltf_path, region)
        print(f"Built terrain region '{gltf_path}': tiles={len(region_terrain['tiles'])} "
              f"triangles={region_terrain['triangleCount']}")
    if scene is not None:
        write_json(sidecar, scene)
    write_json(aggregate_path, aggregate)
    split_x = aggregate["extras"]["metallicTerrainRegions"]["splitX"]
    print(f"Built terrain region scene '{aggregate_path}' at split X={split_x:.8g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gltf", type=Path, help="tiled glTF from BuildTerrainMesh.py")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--lighting", type=Path, help="Metallic scene template for the aggregate editor sample")
    parser.add_argument("--height-field", type=Path, help="World-space source height grid for camera clearance")
    args = parser.parse_args()
    try:
        build_regions(args.gltf, args.output_directory, args.lighting, args.height_field)
    except (OSError, TerrainBuildError, ValueError, KeyError, IndexError, TypeError) as error:
        print(f"Terrain region build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
