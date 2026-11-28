#!/usr/bin/env python3
"""Attach matching two- and four-layer materials to the shared terrain region geometry."""

import argparse
import json
from pathlib import Path

from BuildTerrainMesh import TerrainBuildError, write_atomic
from BuildTerrainVSMMaterial import terrain_material_binding, write_sample_material


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_changed_json(path, document):
    encoded = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if not path.is_file() or path.read_bytes() != encoded:
        write_atomic(path, encoded)


def build_region_materials(sample_path: Path, layer_asset: Path) -> None:
    geometry = load_json(sample_path)
    regions = geometry["extras"]["metallicTerrainRegions"]["regions"]
    material_names = [material["name"] for material in geometry["materials"]]
    source_scene = load_json(sample_path.with_suffix(".metallic_scene.json"))
    grid_path = sample_path.with_name(regions[0]["gltf"])
    two = sample_path.with_name(f"{sample_path.stem}_two.gltf")
    four = sample_path.with_name(f"{sample_path.stem}_four.gltf")

    write_sample_material(grid_path, four, weight_map=None, layer_asset=layer_asset)
    instance = load_json(four.with_suffix(".material"))
    metadata = load_json(four.with_suffix(".terrain-layers.json"))
    bindings = instance["terrainLayers"]
    channels = {layer["channel"]: layer["name"] for layer in bindings["layers"]}
    if set(channels) != {"R", "G", "B", "A"} or channels["G"] != "grass" or bindings["defaultLayer"] != "grass":
        raise TerrainBuildError("the region comparison requires RGBA layers and grass as the G-channel default")
    # Keep the exact same weights, definition, R/G maps, and world transforms in both variants.
    bindings["layers"] = [layer for layer in bindings["layers"] if layer["channel"] in ("R", "G")]
    metadata["layers"] = [layer for layer in metadata["layers"] if layer["channel"] in ("R", "G")]
    write_changed_json(two.with_suffix(".material"), instance)
    write_changed_json(two.with_suffix(".terrain-layers.json"), metadata)

    for variant in (two, four):
        binding = terrain_material_binding(variant, "multilayer")
        sidecar = {**source_scene, "source": sample_path.name, "materials": [
            {**binding, "materialIndex": index, "sourceName": name} for index, name in enumerate(material_names)
        ]}
        write_changed_json(variant.with_suffix(".metallic_scene.json"), sidecar)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Baked aggregate terrain regions glTF")
    parser.add_argument("--layer-asset", type=Path, required=True, help="Houdini-authored Metallic.TerrainLayers JSON")
    args = parser.parse_args()
    try:
        build_region_materials(args.sample, args.layer_asset)
    except (IndexError, KeyError, OSError, ValueError, TerrainBuildError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
