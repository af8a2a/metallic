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
from TerrainLayerAsset import CHANNELS, load_terrain_layer_asset, read_layer_weights


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


def encode_weight_pngs(weights: np.ndarray) -> tuple[bytes, ...]:
    # Quantize the full cumulative sum before splitting so both packs share one 255-unit total.
    cumulative = np.cumsum(weights, axis=-1, dtype=np.float64)
    cumulative *= 255
    np.rint(cumulative, out=cumulative)
    packed = np.diff(cumulative.astype(np.int16), axis=-1, prepend=np.int16(0)).astype(np.uint8)
    if packed.shape[-1] == len(CHANNELS):
        packed = mirror_weight_map(packed)
    pngs = []
    for start in range(0, packed.shape[-1], len(CHANNELS)):
        with io.BytesIO() as stream:
            rgba = packed[..., start:start + len(CHANNELS)]
            Image.fromarray(rgba).save(stream, format="PNG")
            pngs.append(stream.getvalue())
    return tuple(pngs)


def prepare_terrain_layer_import(terrain_path: Path, asset_path: Path, output: Path) -> tuple[dict, dict[str, bytes]]:
    terrain_path = terrain_path.resolve()
    output = output.resolve()
    terrain = load_terrain_grid(terrain_path)
    asset = load_terrain_layer_asset(asset_path)
    sources = {terrain_path, asset.path, *asset.weight_maps}
    sources.update(texture for layer in asset.layers for texture in (layer.base_color, layer.normal, layer.roughness))
    if output in sources:
        raise TerrainBuildError("the imported asset must have a separate output path")

    weights = read_layer_weights(asset, terrain)
    multi_map = len(asset.weight_maps) > 1
    rows, columns = weights.shape[:2]
    if multi_map:
        # V2 uses native dimensions and a terrain weight ClampToEdge sampler.
        scale_x, scale_z = (columns - 1) / columns, (rows - 1) / rows
        offset_x, offset_z = 0.5 / columns, 0.5 / rows
    else:
        scale_x, scale_z, offset_x, offset_z = weight_uv_transform(columns, rows)
    weight_transform = [scale_x, 0, offset_x, 0, scale_z, offset_z]
    weight_maps = []
    payloads = {}
    for png in encode_weight_pngs(weights):
        # Publish new payloads under distinct names while the previous material can still be in use.
        weight_file = f"{output.stem}.weights-{hashlib.sha256(png).hexdigest()[:16]}.png"
        if output.with_name(weight_file) in sources:
            raise TerrainBuildError("the imported weight texture must have a separate output path")
        payloads[weight_file] = png
        weight_maps.append({"path": weight_file, "format": "RGBA8", "colorSpace": "linear",
                            "uvTransform": weight_transform})
    horizontal_size = terrain["horizontalSize"]
    layers = []
    for layer in asset.layers:
        repeat_x, repeat_z = (horizontal_size / period for period in layer.world_tile_size)
        if any(not 0 < repeat < math.inf for repeat in (repeat_x, repeat_z)):
            raise TerrainBuildError(f"layer '{layer.name}' worldTileSize produces an unrepresentable UV scale")
        layers.append({
            "name": layer.name,
            "channel": layer.channel,
            **({"weightMap": layer.weight_map} if multi_map else {}),
            "worldTileSize": layer.world_tile_size,
            "uvTransform": [repeat_x, 0, -0.5 * repeat_x, 0, repeat_z, -0.5 * repeat_z],
            "baseColor": {"path": relative_asset_path(layer.base_color, output.parent), "colorSpace": "srgb"},
            "normal": {"path": relative_asset_path(layer.normal, output.parent), "colorSpace": "linear"},
            "roughness": {"path": relative_asset_path(layer.roughness, output.parent), "colorSpace": "linear"},
        })
    document = {
        "type": "Metallic.ImportedTerrainLayers",
        "version": 2 if multi_map else 1,
        "terrain": {
            "path": relative_asset_path(terrain_path, output.parent),
            "sourceDataWindow": terrain["sourceDataWindow"],
            "sourceResolution": [columns, rows],
            "horizontalSize": horizontal_size,
            "flipZ": terrain["flipZ"],
        },
        "defaultLayer": asset.default_layer,
        **({"weightMaps": weight_maps} if multi_map else {"weightMap": weight_maps[0]}),
        "layers": layers,
    }
    return document, payloads


def import_terrain_layers(terrain_path: Path, asset_path: Path, output: Path) -> None:
    output = output.resolve()
    document, payloads = prepare_terrain_layer_import(terrain_path, asset_path, output)
    encoded = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        pending_manifest = Path(staging) / output.name
        pending_manifest.write_bytes(encoded)
        for weight_file, png in payloads.items():
            pending_weights = Path(staging) / weight_file
            pending_weights.write_bytes(png)
            pending_weights.replace(output.with_name(weight_file))
        pending_manifest.replace(output)


def build_terrain_layer_material(terrain_path: Path, asset_path: Path, output: Path) -> tuple[dict, dict]:
    output = output.resolve()
    definition = output.with_suffix(".materialdef")
    instance = output.with_suffix(".material")
    if definition == asset_path.resolve():
        raise TerrainBuildError("the material definition must have a separate output path")
    imported, payloads = prepare_terrain_layer_import(terrain_path, asset_path, instance)
    multi_map = imported["version"] == 2
    weight_maps = imported["weightMaps"] if multi_map else [imported["weightMap"]]
    weight_bindings = [{"uri": "asset://" + weights["path"], "transform": weights["uvTransform"]}
                       for weights in weight_maps]

    def resource(source: dict, transform: list, role: str) -> dict:
        path = (output.parent / source["path"]).resolve()
        # Keep the last publication readable and give MaterialAsset reload a
        # new texture identity only when the source bytes change.
        content = path.read_bytes()
        local_name = f"{output.stem}_layer_{hashlib.sha256(content).hexdigest()[:16]}{path.suffix.lower()}"
        payloads[local_name] = content
        binding = {"uri": "asset://" + local_name, "transform": transform}
        if role == "baseColor":
            binding["colorSpace"] = source["colorSpace"]
        return binding

    layer_keys = ("name", "channel", "weightMap") if multi_map else ("name", "channel")
    roles = ("baseColor", "normal", "roughness")
    layers = []
    for layer in imported["layers"]:
        bindings = {role: resource(layer[role], layer["uvTransform"], role) for role in roles}
        layers.append({**{key: layer[key] for key in layer_keys}, **bindings})
    material_def = {"type": "Metallic.MaterialDefinition", "version": 1, "definitionVersion": 1,
                    "implementation": "OpenPBR.TerrainLayers", "defaults": {}}
    material_instance = {
        "type": "Metallic.MaterialInstance", "version": 1, "definitionVersion": 1,
        "definition": "asset://" + definition.name,
        "parameters": {"metalness": 0}, "features": {"doubleSided": True},
        "terrainLayers": {
            **({"weightMaps": weight_bindings} if multi_map else {"weights": weight_bindings[0]}),
            "defaultLayer": imported["defaultLayer"], "layers": layers,
        },
    }
    for path, document in ((definition, material_def), (instance, material_instance)):
        payloads[path.name] = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        # Publish the instance last, after every referenced file exists.
        for name, content in payloads.items():
            destination = output.with_name(name)
            if destination.is_file() and destination.read_bytes() == content:
                continue
            pending = Path(staging) / name
            pending.write_bytes(content)
            pending.replace(destination)

    scene_binding = {"sourceId": "main", "materialIndex": 0, "sourceName": "Terrain Multilayer",
                     "materialAsset": {"uri": "asset://" + instance.name, "root": "."}}
    metadata = {
        "defaultLayer": imported["defaultLayer"], "weightResolution": imported["terrain"]["sourceResolution"],
        "layers": [{key: layer[key] for key in (*layer_keys, "worldTileSize")} for layer in imported["layers"]],
    }
    if multi_map:
        metadata.update(version=2, weightMaps=weight_maps)
    return scene_binding, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("terrain", type=Path, help="Terrain glTF baked from the Houdini height field")
    parser.add_argument("layers", type=Path, help="Metallic.TerrainLayers or Ancient Rift Valley DCC source JSON")
    parser.add_argument("--output", type=Path, required=True, help="Imported terrain layer manifest JSON")
    args = parser.parse_args()
    try:
        import_terrain_layers(args.terrain, args.layers, args.output)
    except (OSError, ValueError, TerrainBuildError) as error:
        parser.exit(1, f"Terrain layer import failed: {error}\n")


if __name__ == "__main__":
    main()
