#!/usr/bin/env python3
"""Build Ancient Rift Valley regions from their shared height grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from BuildTerrainMesh import (
    TerrainBuildError,
    build_terrain_mesh,
    load_height_field,
    write_atomic,
    write_height_field,
    write_terrain_gltf,
)
from BuildTerrainRegions import combine_regions, make_region, write_json
from BuildTerrainVSMMaterial import TERRAIN_MATERIAL_NAMES, terrain_material_binding
from BuildTerrainVSMSample import append_camera, relative_path


WORLD_SIZE = 1000.0
SOURCE_SAMPLES = 2049
SOURCE_REGION_CELLS = 512
SOURCE_REGIONS_PER_AXIS = 4
RENDER_TILES_PER_REGION = 4
SAMPLE_CAMERA = {
    "eye": [66.0, 118.0, -218.0],
    "center": [42.0, 86.0, 125.0],
    "up": [0.0, 1.0, 0.0],
    "fovDegrees": 48.0,
    "znear": 0.5,
    "zfar": 1400.0,
}
WORLD_CAMERA = {
    "eye": [1040.0, 1475.0, -1240.0],
    "center": [0.0, 200.0, 0.0],
    "up": [0.0, 1.0, 0.0],
    "fovDegrees": 42.0,
    "znear": 0.5,
    "zfar": 3500.0,
}


def build_sample(heightmap: Path, lighting: Path, output: Path, regions_per_axis: int = 2) -> None:
    heights, channel, data_window = load_height_field(heightmap, "Y")
    if heights.shape != (SOURCE_SAMPLES, SOURCE_SAMPLES) or data_window != (0, 0, 2048, 2048):
        raise TerrainBuildError("Ancient Rift Valley requires its 2049² metre height grid at EXR origin (0,0)")

    first_region = (SOURCE_REGIONS_PER_AXIS - regions_per_axis) // 2
    begin = first_region * SOURCE_REGION_CELLS
    end = (first_region + regions_per_axis) * SOURCE_REGION_CELLS
    step = WORLD_SIZE / (SOURCE_SAMPLES - 1)
    width = (end - begin) * step
    sample = heights[begin:end + 1, begin:end + 1]
    positions, normals, texcoords, indices, metadata = build_terrain_mesh(
        sample, width, height_scale=1.0, height_offset=0.0, skirt_depth=0.0, flip_z=False)
    if first_region:
        # Preserve the full grid's normal stencil and UV mapping through the central crop.
        halo = heights[begin - 1:end + 2, begin - 1:end + 2]
        _, expanded_normals, _, _, _ = build_terrain_mesh(
            halo, width + 2.0 * step, height_scale=1.0, height_offset=0.0, skirt_depth=0.0, flip_z=False)
        normals = expanded_normals.reshape(*halo.shape, 3)[1:-1, 1:-1].reshape(-1, 3)
        texcoords *= width / WORLD_SIZE
        texcoords += begin / (SOURCE_SAMPLES - 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    height_field = output.with_suffix(".heightfield.bin")
    write_height_field(height_field, positions, metadata, width)
    with tempfile.TemporaryDirectory(prefix="valley-grid-", dir=output.parent) as directory:
        tile_source = Path(directory) / "Ancient Rift Valley.gltf"
        tile_payload = tile_source.with_suffix(".bin")
        write_terrain_gltf(tile_source, tile_payload, heightmap, channel, data_window, positions,
                           normals, texcoords, indices, metadata, WORLD_SIZE, 1.0, 0.0, 0.0,
                           False, RENDER_TILES_PER_REGION * regions_per_axis)
        source = json.loads(tile_source.read_text(encoding="utf-8"))
        payload = tile_payload.read_bytes()

    terrain = source["extras"]["metallicTerrainTiles"]
    terrain["sourceResolution"] = [SOURCE_SAMPLES, SOURCE_SAMPLES]
    for tile in terrain["tiles"]:
        tile["cellBounds"] = [cell + begin for cell in tile["cellBounds"]]

    regions = []
    count = RENDER_TILES_PER_REGION
    for z in range(regions_per_axis):
        for x in range(regions_per_axis):
            x0, z0 = x * count, z * count
            region_id = f"x{x + first_region}_z{z + first_region}"
            document, raw = make_region(source, payload, terrain, region_id,
                                        (x0, z0, x0 + count, z0 + count))
            if not regions:
                document["materials"][0]["name"] = TERRAIN_MATERIAL_NAMES["multilayer"]
            basename = document["scenes"][0]["name"]
            write_atomic(output.parent / document["buffers"][0]["uri"], bytes(raw))
            write_json(output.with_name(f"{basename}.gltf"), document)
            regions.append(document)

    aggregate = combine_regions(regions)
    aggregate["scenes"][0]["name"] = f"Ancient Rift Valley {regions_per_axis}x{regions_per_axis}"
    aggregate["extras"]["metallicTerrainTiles"] = {
        **terrain,
        "sampleWindowInclusive": [begin, begin, end, end],
    }
    camera = SAMPLE_CAMERA if first_region else WORLD_CAMERA
    aggregate["extras"]["metallicTerrainRegions"]["sampleCamera"] = camera
    append_camera(aggregate, camera)

    scene = json.loads(lighting.read_text(encoding="utf-8"))
    scene["source"] = output.name
    scene["terrain"] = {"heightField": height_field.name, "cameraClearance": 2.0}
    environment = scene["world"]["environment"]
    environment["path"] = relative_path(lighting.parent / environment["path"], output.parent)
    binding = terrain_material_binding(output, "multilayer")
    scene["materials"] = [
        {**binding, "materialIndex": index, "sourceName": material["name"]}
        for index, material in enumerate(aggregate["materials"])
    ]
    write_json(output.with_suffix(".metallic_scene.json"), scene)
    write_json(output, aggregate)
    print(f"Built {output}: {len(regions)} regions, {len(aggregate['meshes'])} boundary-locked tiles, "
          f"{len(indices) // 3} triangles, {width:g} m square")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heightmap", type=Path)
    parser.add_argument("--lighting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions-per-axis", type=int, choices=(2, SOURCE_REGIONS_PER_AXIS), default=2)
    args = parser.parse_args()
    try:
        build_sample(args.heightmap, args.lighting, args.output, args.regions_per_axis)
    except (OSError, TerrainBuildError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
