#!/usr/bin/env python3
"""Reimport the material bound to an assembled terrain sample."""

import argparse
import json
from pathlib import Path

from BuildTerrainMesh import TerrainBuildError, write_atomic


TERRAIN_MATERIAL_NAMES = {"legacy": "Terrain Rock and Grass", "multilayer": "Terrain Multilayer"}


def terrain_material_binding(sample_path: Path, kind: str) -> dict:
    instance = sample_path.with_suffix(".material")
    return {"sourceId": "main", "materialIndex": 0, "sourceName": TERRAIN_MATERIAL_NAMES[kind],
            "materialAsset": {"uri": "asset://" + instance.name, "root": "."}}


def material_kind(weight_map: Path | None, layer_asset: Path | None) -> str | None:
    if weight_map is not None and layer_asset is not None:
        raise TerrainBuildError("select either a legacy weight map or a terrain layer asset")
    if layer_asset is not None:
        return "multilayer"
    return "legacy" if weight_map is not None else None


def write_sample_material(grid_path: Path, sample_path: Path, *, weight_map: Path | None,
                          layer_asset: Path | None) -> None:
    if layer_asset is not None:
        from ImportTerrainLayers import build_terrain_layer_material

        _, metadata = build_terrain_layer_material(grid_path, layer_asset, sample_path)
    else:
        from BuildTerrainLayerTextures import build_layer_material
        from ImportTerrainLayers import load_terrain_grid

        _, metadata = build_layer_material(sample_path, weight_map, load_terrain_grid(grid_path))
    output = sample_path.with_suffix(".terrain-layers.json")
    encoded = (json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if not output.is_file() or output.read_bytes() != encoded:
        write_atomic(output, encoded)


def build_sample_material(sample_path: Path, *, weight_map: Path | None = None,
                          layer_asset: Path | None = None) -> None:
    kind = material_kind(weight_map, layer_asset)
    if kind is None:
        raise TerrainBuildError("material reimport requires either --weight-map or --layer-asset")
    document = json.loads(sample_path.read_text(encoding="utf-8"))
    if document["materials"][0].get("name") != TERRAIN_MATERIAL_NAMES[kind]:
        raise TerrainBuildError(f"'{sample_path}' does not have the expected {kind} terrain material binding")
    write_sample_material(sample_path, sample_path, weight_map=weight_map, layer_asset=layer_asset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Assembled terrain sample glTF")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--layer-asset", type=Path, help="Metallic.TerrainLayers or Ancient Rift Valley DCC source JSON")
    inputs.add_argument("--weight-map", type=Path, help="Legacy linear R OpenEXR rock weights")
    args = parser.parse_args()
    try:
        build_sample_material(args.sample, weight_map=args.weight_map, layer_asset=args.layer_asset)
    except (IndexError, KeyError, OSError, ValueError, TerrainBuildError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
