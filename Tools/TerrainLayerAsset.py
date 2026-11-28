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
class TerrainTriplanar:
    normal_strength: float
    base_color_texture_mix: float
    base_color_mean_linear: tuple[float, float, float]


@dataclass(frozen=True)
class TerrainLayer:
    name: str
    weight_map: int
    channel: str
    world_tile_size: tuple[float, float]
    base_color: Path
    normal: Path
    roughness: Path
    triplanar: TerrainTriplanar | None = None


@dataclass(frozen=True)
class TerrainLayerAsset:
    path: Path
    weight_maps: tuple[Path, ...]
    default_layer: str
    layers: tuple[TerrainLayer, ...]
    source_grid: dict | None

    @property
    def weight_map(self) -> Path:
        return self.weight_maps[0]


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


def _triplanar(value: object, layer_name: str) -> TerrainTriplanar:
    if not isinstance(value, dict) or set(value) != {"normalStrength", "baseColorTextureMix", "baseColorMeanLinear"}:
        raise TerrainBuildError(f"layer '{layer_name}'.triplanar requires normalStrength, baseColorTextureMix and baseColorMeanLinear")
    color = value["baseColorMeanLinear"]
    if not isinstance(color, list) or len(color) != 3 or any(
        type(component) not in (int, float) or not 0 <= component <= 1
        for component in (value["normalStrength"], value["baseColorTextureMix"], *color)
    ):
        raise TerrainBuildError(f"layer '{layer_name}'.triplanar requires unit normal strength, texture mix and linear Rec.709 RGB")
    return TerrainTriplanar(float(value["normalStrength"]), float(value["baseColorTextureMix"]), tuple(map(float, color)))


def _dcc_terrain_layers(document: dict) -> tuple[dict, dict]:
    if document.get("normalConvention") != "OpenGL tangent-space +Y":
        raise TerrainBuildError("DCC terrain asset requires OpenGL tangent-space +Y normals")
    coordinates = document.get("coordinateSystem")
    sampling = document.get("sampling")
    if not isinstance(coordinates, dict) or not isinstance(sampling, dict):
        raise TerrainBuildError("DCC terrain asset requires coordinateSystem and sampling")
    bounds = coordinates.get("worldBoundsXZ")
    if (not isinstance(bounds, list) or len(bounds) != 4
            or any(type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max
                   for value in bounds)
            or bounds[0] != bounds[1] or bounds[2] != bounds[3]
            or bounds[0] != -bounds[2] or not 0 < bounds[2] <= sys.float_info.max / 2
            or coordinates.get("unit") != "meter" or coordinates.get("up") != "+Y"
            or coordinates.get("heightfieldColumnAxis") != "+X"
            or coordinates.get("heightfieldRowAxis") != "+Z"
            or type(coordinates.get("flipZ")) is not bool):
        raise TerrainBuildError("DCC terrain asset requires a centered square Y-up meter grid and explicit flipZ")
    window = sampling.get("dataWindowInclusive")
    if not isinstance(window, list) or len(window) != 4 or any(type(value) is not int for value in window):
        raise TerrainBuildError("DCC terrain asset requires an integer sample data window")
    source_grid = {"sourceDataWindow": tuple(window),
                   "horizontalSize": bounds[2] * 2, "flipZ": coordinates["flipZ"]}

    weight_maps = document.get("weightMaps")
    if not isinstance(weight_maps, list) or any(
        not isinstance(entry, dict) or entry.get("channels") != CHANNELS
        or entry.get("colorSpace") != "Linear" for entry in weight_maps
    ):
        raise TerrainBuildError("DCC terrain asset weightMaps must declare linear RGBA images")
    layers = document.get("layers")
    if not isinstance(layers, list):
        raise TerrainBuildError("DCC terrain asset requires a layers array")
    adapted = []
    for layer in layers:
        if not isinstance(layer, dict) or not isinstance(layer.get("textures"), dict):
            raise TerrainBuildError("DCC terrain layer must have named texture bindings")
        texture_paths = {}
        for role, color_space in (("baseColor", "sRGB"), ("normal", "Linear"), ("roughness", "Linear")):
            texture = layer["textures"].get(role)
            if not isinstance(texture, dict) or texture.get("colorSpace") != color_space:
                raise TerrainBuildError(f"DCC terrain layer '{layer.get('name')}' requires {color_space} {role}")
            texture_paths[role] = texture.get("path")
        adapted.append({**layer, **texture_paths, "worldTileSize": layer.get("worldTileSizeMeters")})
    return {"type": "Metallic.TerrainLayers", "version": 2, "layers": adapted,
            "weightMaps": [entry.get("path") for entry in weight_maps],
            "defaultLayer": document.get("defaultLayer")}, source_grid


def load_terrain_layer_asset(path: Path) -> TerrainLayerAsset:
    path = path.resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TerrainBuildError(f"cannot read terrain layer asset '{path}': {error}") from error

    source_grid = None
    if (isinstance(document, dict) and document.get("type") == "AncientRiftValley.DCCSource"
            and type(document.get("version")) is int and document["version"] == 1):
        document, source_grid = _dcc_terrain_layers(document)
    if (not isinstance(document, dict) or document.get("type") != "Metallic.TerrainLayers"
            or type(document.get("version")) is not int or document["version"] not in (1, 2)):
        raise TerrainBuildError("terrain layer asset must have type Metallic.TerrainLayers and version 1 or 2")
    if document["version"] == 1:
        if "weightMaps" in document:
            raise TerrainBuildError("version 1 terrain layers use weightMap")
        weight_paths = [document.get("weightMap")]
    else:
        weight_paths = document.get("weightMaps")
        if "weightMap" in document or not isinstance(weight_paths, list) or len(weight_paths) != 2:
            raise TerrainBuildError("version 2 terrain layers require exactly two weightMaps")
    weight_maps = tuple(_relative_file(value, path.parent, f"weightMaps[{index}]")
                        for index, value in enumerate(weight_paths))
    if any(weight.suffix.lower() != ".exr" for weight in weight_maps):
        raise TerrainBuildError("weight maps must be linear OpenEXR images")

    layers = document.get("layers")
    max_layers = len(weight_maps) * len(CHANNELS)
    if not isinstance(layers, list) or not 1 <= len(layers) <= max_layers:
        raise TerrainBuildError(f"terrain layer asset requires one to {max_layers} layers")
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
        weight_map = layer.get("weightMap", 0) if len(weight_maps) == 1 else layer.get("weightMap")
        if type(weight_map) is not int or not 0 <= weight_map < len(weight_maps):
            raise TerrainBuildError(f"layer '{name}' must declare a valid weightMap index")
        slot = (weight_map, channel)
        if name.casefold() in names or slot in channels:
            raise TerrainBuildError(f"layer '{name}' repeats a layer name or RGBA channel")
        names.add(name.casefold())
        channels.add(slot)
        imported_layers.append(TerrainLayer(
            name=name,
            weight_map=weight_map,
            channel=channel,
            world_tile_size=_world_tile_size(layer.get("worldTileSize"), name),
            base_color=_relative_file(layer.get("baseColor"), path.parent, f"layer '{name}'.baseColor"),
            normal=_relative_file(layer.get("normal"), path.parent, f"layer '{name}'.normal"),
            roughness=_relative_file(layer.get("roughness"), path.parent, f"layer '{name}'.roughness"),
            triplanar=_triplanar(layer["triplanar"], name) if "triplanar" in layer else None,
        ))
    default_layer = document.get("defaultLayer")
    if default_layer not in [layer.name for layer in imported_layers]:
        raise TerrainBuildError("defaultLayer must exactly name a declared layer")
    return TerrainLayerAsset(path, weight_maps, default_layer, tuple(imported_layers), source_grid)


def _read_weight_map(path: Path, layers: list[TerrainLayer], data_window: tuple[int, ...]) -> np.ndarray:
    try:
        import Imath
        import OpenEXR
    except ImportError as error:
        raise TerrainBuildError("terrain layer import requires Tools/requirements-terrain.txt") from error

    try:
        image = OpenEXR.InputFile(str(path))
    except (OSError, RuntimeError) as error:
        raise TerrainBuildError(f"cannot open terrain weight EXR '{path}': {error}") from error

    try:
        header = image.header()
        window = header["dataWindow"]
        actual_window = (window.min.x, window.min.y, window.max.x, window.max.y)
        if actual_window != data_window:
            raise TerrainBuildError(
                f"terrain weight and height EXR data windows must match: {actual_window} != {data_window}"
            )
        columns = window.max.x - window.min.x + 1
        rows = window.max.y - window.min.y + 1
        if columns < 2 or rows < 2:
            raise TerrainBuildError("terrain weight EXR must contain at least 2x2 samples")
        weights = np.zeros((rows, columns, len(CHANNELS)), dtype=np.float32)
        channels = header["channels"]
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        for layer in layers:
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
        return weights
    except TerrainBuildError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise TerrainBuildError(f"cannot read terrain weight EXR '{path}': {error}") from error
    finally:
        image.close()


def read_layer_weights(asset: TerrainLayerAsset, terrain: dict) -> np.ndarray:
    """Return normalized world-aligned weight slots with unassigned channels zero."""
    data_window = terrain.get("sourceDataWindow")
    flip_z = terrain.get("flipZ")
    if (not isinstance(data_window, (list, tuple)) or len(data_window) != 4
            or any(type(value) is not int for value in data_window) or type(flip_z) is not bool):
        raise TerrainBuildError("terrain source metadata must contain sourceDataWindow and flipZ")
    data_window = tuple(data_window)
    if asset.source_grid is not None and asset.source_grid != {
        "sourceDataWindow": data_window, "horizontalSize": terrain.get("horizontalSize"), "flipZ": flip_z
    }:
        raise TerrainBuildError("DCC terrain asset and baked global grid must agree on data window, size and flipZ")
    packs = []
    for index, path in enumerate(asset.weight_maps):
        layers = [layer for layer in asset.layers if layer.weight_map == index]
        packs.append(_read_weight_map(path, layers, data_window))
    weights = packs[0] if len(packs) == 1 else np.concatenate(packs, axis=-1)

    total = weights.sum(axis=-1, keepdims=True)
    np.divide(weights, total, out=weights, where=total > 0)
    default = next(layer for layer in asset.layers if layer.name == asset.default_layer)
    default_slot = default.weight_map * len(CHANNELS) + CHANNELS.index(default.channel)
    weights[..., default_slot][total[..., 0] == 0] = 1
    return weights[::-1].copy() if flip_z else weights
