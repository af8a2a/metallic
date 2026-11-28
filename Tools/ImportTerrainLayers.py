#!/usr/bin/env python3
"""Import Houdini terrain layers against the source grid of a baked terrain glTF."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from BuildTerrainLayerTextures import mirror_weight_map, weight_uv_transform
from BuildTerrainMesh import TerrainBuildError
from TerrainLayerAsset import load_terrain_layer_asset, read_layer_weights


def load_terrain_grid(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    extras = document.get("extras") if isinstance(document, dict) else None
    grid = (extras.get("metallicTerrainTiles") or extras.get("metallicTerrainP0")) if isinstance(extras, dict) else None
    if not isinstance(grid, dict):
        raise TerrainBuildError(f"'{path}' has no baked Metallic terrain source grid")
    horizontal_size = grid.get("horizontalSize")
    if type(horizontal_size) not in (int, float) or not 0 < horizontal_size <= sys.float_info.max:
        raise TerrainBuildError(f"'{path}' must declare a positive terrain horizontalSize")
    return grid


def relative_asset_path(path: Path, directory: Path) -> str:
    try:
        return Path(os.path.relpath(path, directory)).as_posix()
    except ValueError:
        return path.as_posix()


def encode_weight_png(weights: np.ndarray) -> bytes:
    # Quantize cumulative weights so the four byte channels still sum to 255.
    cumulative = np.rint(np.cumsum(weights, axis=-1, dtype=np.float64) * 255).astype(np.int16)
    rgba = np.diff(cumulative, axis=-1, prepend=0).astype(np.uint8)
    with io.BytesIO() as stream:
        Image.fromarray(mirror_weight_map(rgba)).save(stream, format="PNG")
        return stream.getvalue()


def import_terrain_layers(terrain_path: Path, asset_path: Path, output: Path) -> None:
    terrain_path = terrain_path.resolve()
    output = output.resolve()
    terrain = load_terrain_grid(terrain_path)
    asset = load_terrain_layer_asset(asset_path)
    sources = {terrain_path, asset.path, asset.weight_map}
    sources.update(texture for layer in asset.layers for texture in (layer.base_color, layer.normal, layer.roughness))
    if output in sources:
        raise TerrainBuildError("the imported asset must have a separate output path")

    weights = read_layer_weights(asset, terrain)
    png = encode_weight_png(weights)
    # Name the payload by content so an interrupted reimport leaves the previous asset usable.
    weight_file = f"{output.stem}.weights-{hashlib.sha256(png).hexdigest()[:16]}.png"
    weight_output = output.with_name(weight_file)
    if weight_output in sources:
        raise TerrainBuildError("the imported weight texture must have a separate output path")

    rows, columns = weights.shape[:2]
    scale_x, scale_z, offset_x, offset_z = weight_uv_transform(columns, rows)
    horizontal_size = terrain["horizontalSize"]
    layers = []
    for layer in asset.layers:
        repeat_x, repeat_z = (horizontal_size / period for period in layer.world_tile_size)
        if any(not 0 < repeat < math.inf for repeat in (repeat_x, repeat_z)):
            raise TerrainBuildError(f"layer '{layer.name}' worldTileSize produces an unrepresentable UV scale")
        layers.append({
            "name": layer.name,
            "channel": layer.channel,
            "worldTileSize": layer.world_tile_size,
            "uvTransform": [repeat_x, 0, -0.5 * repeat_x, 0, repeat_z, -0.5 * repeat_z],
            "baseColor": {"path": relative_asset_path(layer.base_color, output.parent), "colorSpace": "srgb"},
            "normal": {"path": relative_asset_path(layer.normal, output.parent), "colorSpace": "linear"},
            "roughness": {"path": relative_asset_path(layer.roughness, output.parent), "colorSpace": "linear"},
        })
    document = {
        "type": "Metallic.ImportedTerrainLayers",
        "version": 1,
        "terrain": {
            "path": relative_asset_path(terrain_path, output.parent),
            "sourceDataWindow": terrain["sourceDataWindow"],
            "sourceResolution": [columns, rows],
            "horizontalSize": horizontal_size,
            "flipZ": terrain["flipZ"],
        },
        "defaultLayer": asset.default_layer,
        "weightMap": {
            "path": weight_file,
            "format": "RGBA8",
            "colorSpace": "linear",
            "uvTransform": [scale_x, 0, offset_x, 0, scale_z, offset_z],
        },
        "layers": layers,
    }
    encoded = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        pending_weights = Path(staging) / weight_file
        pending_manifest = Path(staging) / output.name
        pending_weights.write_bytes(png)
        pending_manifest.write_bytes(encoded)
        pending_weights.replace(weight_output)
        pending_manifest.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terrain", type=Path, help="Terrain glTF baked from the Houdini height field")
    parser.add_argument("layers", type=Path, help="Metallic.TerrainLayers JSON authored alongside the weight EXR")
    parser.add_argument("--output", type=Path, required=True, help="Imported terrain layer manifest JSON")
    args = parser.parse_args()
    try:
        import_terrain_layers(args.terrain, args.layers, args.output)
    except (OSError, ValueError, TerrainBuildError) as error:
        parser.exit(1, f"Terrain layer import failed: {error}\n")


if __name__ == "__main__":
    main()
