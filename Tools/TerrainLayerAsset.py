"""Read a terrain layer asset and its linear Houdini RGBA weights."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
import sys

import numpy as np

from BuildTerrainMesh import TerrainBuildError


CHANNELS = "RGBA"


@dataclass(frozen=True)
class TerrainLayer:
    name: str
    channel: str
    world_tile_size: tuple[float, float]
    base_color: Path
    normal: Path
    roughness: Path


@dataclass(frozen=True)
class TerrainLayerAsset:
    path: Path
    weight_map: Path
    default_layer: str
    layers: tuple[TerrainLayer, ...]


def _relative_file(value: object, directory: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise TerrainBuildError(f"{label} must be a relative file path")
    path = Path(value)
    windows_path = PureWindowsPath(value)
    if path.is_absolute() or windows_path.drive or windows_path.root:
        raise TerrainBuildError(f"{label} must be relative to the terrain layer asset")
    resolved = (directory / path).resolve()
    if not resolved.is_file():
        raise TerrainBuildError(f"{label} file does not exist: {resolved}")
    return resolved


def _world_tile_size(value: object, layer_name: str) -> tuple[float, float]:
    dimensions = value if isinstance(value, list) else [value, value]
    if len(dimensions) != 2 or any(
        type(dimension) not in (int, float) or not 0 < dimension <= sys.float_info.max for dimension in dimensions
    ):
        raise TerrainBuildError(
            f"layer '{layer_name}'.worldTileSize must be a positive finite number or a pair [x, z]"
        )
    return float(dimensions[0]), float(dimensions[1])


def load_terrain_layer_asset(path: Path) -> TerrainLayerAsset:
    path = path.resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TerrainBuildError(f"cannot read terrain layer asset '{path}': {error}") from error

    if (not isinstance(document, dict) or document.get("type") != "Metallic.TerrainLayers"
            or type(document.get("version")) is not int or document["version"] != 1):
        raise TerrainBuildError("terrain layer asset must have type Metallic.TerrainLayers and version 1")
    weight_map = _relative_file(document.get("weightMap"), path.parent, "weightMap")
    if weight_map.suffix.lower() != ".exr":
        raise TerrainBuildError("weightMap must be a linear OpenEXR image")

    layers = document.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= len(CHANNELS):
        raise TerrainBuildError("terrain layer asset requires one to four layers")
    imported_layers = []
    names = set()
    channels = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise TerrainBuildError(f"layers[{index}] must be an object")
        name = layer.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise TerrainBuildError(f"layers[{index}].name must be a nonempty layer name without surrounding whitespace")
        channel = layer.get("channel")
        if not isinstance(channel, str) or len(channel) != 1 or channel not in CHANNELS:
            raise TerrainBuildError(f"layer '{name}' must declare one of R, G, B or A")
        if name.casefold() in names or channel in channels:
            raise TerrainBuildError(f"layer '{name}' repeats a layer name or RGBA channel")
        names.add(name.casefold())
        channels.add(channel)
        imported_layers.append(TerrainLayer(
            name=name,
            channel=channel,
            world_tile_size=_world_tile_size(layer.get("worldTileSize"), name),
            base_color=_relative_file(layer.get("baseColor"), path.parent, f"layer '{name}'.baseColor"),
            normal=_relative_file(layer.get("normal"), path.parent, f"layer '{name}'.normal"),
            roughness=_relative_file(layer.get("roughness"), path.parent, f"layer '{name}'.roughness"),
        ))
    default_layer = document.get("defaultLayer")
    if default_layer not in [layer.name for layer in imported_layers]:
        raise TerrainBuildError("defaultLayer must exactly name a declared layer")
    return TerrainLayerAsset(path, weight_map, default_layer, tuple(imported_layers))


def read_layer_weights(asset: TerrainLayerAsset, terrain: dict) -> np.ndarray:
    """Return normalized world-aligned RGBA weights with unassigned channels zero."""
    try:
        import Imath
        import OpenEXR
    except ImportError as error:
        raise TerrainBuildError("terrain layer import requires Tools/requirements-terrain.txt") from error

    data_window = terrain.get("sourceDataWindow")
    flip_z = terrain.get("flipZ")
    if (not isinstance(data_window, (list, tuple)) or len(data_window) != 4
            or any(type(value) is not int for value in data_window) or type(flip_z) is not bool):
        raise TerrainBuildError("terrain source metadata must contain sourceDataWindow and flipZ")
    try:
        image = OpenEXR.InputFile(str(asset.weight_map))
    except (OSError, RuntimeError) as error:
        raise TerrainBuildError(f"cannot open terrain weight EXR '{asset.weight_map}': {error}") from error

    try:
        header = image.header()
        window = header["dataWindow"]
        actual_window = (window.min.x, window.min.y, window.max.x, window.max.y)
        if actual_window != tuple(data_window):
            raise TerrainBuildError(
                f"terrain weight and height EXR data windows must match: {actual_window} != {tuple(data_window)}"
            )
        columns = window.max.x - window.min.x + 1
        rows = window.max.y - window.min.y + 1
        if columns < 2 or rows < 2:
            raise TerrainBuildError("terrain weight EXR must contain at least 2x2 samples")
        channels = header["channels"]
        weights = np.zeros((rows, columns, len(CHANNELS)), dtype=np.float32)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        for layer in asset.layers:
            channel = channels.get(layer.channel)
            if channel is None:
                raise TerrainBuildError(f"terrain weight EXR is missing channel '{layer.channel}' for layer '{layer.name}'")
            if channel.xSampling != 1 or channel.ySampling != 1:
                raise TerrainBuildError(f"terrain weight EXR channel '{layer.channel}' must be full resolution")
            if channel.type.v not in (Imath.PixelType.HALF, Imath.PixelType.FLOAT):
                raise TerrainBuildError(f"terrain weight EXR channel '{layer.channel}' must be HALF or FLOAT")
            data = np.frombuffer(image.channel(layer.channel, pixel_type), dtype=np.float32)
            if data.size != rows * columns:
                raise TerrainBuildError(f"terrain weight EXR channel '{layer.channel}' has an invalid sample count")
            if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
                raise TerrainBuildError(f"terrain weight EXR channel '{layer.channel}' must be finite and in [0, 1]")
            weights[..., CHANNELS.index(layer.channel)] = data.reshape(rows, columns)
    except TerrainBuildError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise TerrainBuildError(f"cannot read terrain weight EXR '{asset.weight_map}': {error}") from error
    finally:
        image.close()

    total = weights.sum(axis=-1, keepdims=True)
    np.divide(weights, total, out=weights, where=total > 0)
    default = next(layer for layer in asset.layers if layer.name == asset.default_layer)
    weights[..., CHANNELS.index(default.channel)][total[..., 0] == 0] = 1
    return weights[::-1].copy() if flip_z else weights
