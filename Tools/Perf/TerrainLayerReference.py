"""Sample the four-layer terrain fixture in world space for GPU readback checks."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


RGBA = "RGBA"
TEXTURE_ROLES = {"baseColor": "baseColor", "normalTexture": "normal", "roughness": "roughness"}


def repeat_linear(image, uv):
    height, width = image.shape[:2]
    sample = uv * [width, height] - 0.5
    lower = np.floor(sample).astype(np.int64)
    x, y = lower[..., 0], lower[..., 1]
    u, v = np.moveaxis((sample - lower)[..., None], -2, 0)
    left = image[y % height, x % width] * (1.0 - v) + image[(y + 1) % height, x % width] * v
    right = image[y % height, (x + 1) % width] * (1.0 - v) + image[(y + 1) % height, (x + 1) % width] * v
    return left * (1.0 - u) + right * u


def srgb_to_linear(color):
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def _normalize_normals(normals):
    length_squared = np.sum(normals * normals, axis=-1, keepdims=True)
    result = np.empty_like(normals)
    result[...] = (0.0, 0.0, 1.0)
    valid = length_squared > 1e-8
    return np.divide(normals, np.sqrt(length_squared), out=result, where=valid)


def ray_cone_uv_footprint(world, sites, camera, world_size):
    # The terrain's global UV is XZ / world_size. Its projected area supplies
    # the UV-to-surface ratio in the ray-cone calculation. Reconstruct the
    # geometric normal from adjacent positions in the captured depth plane.
    dx, dy = np.gradient(world, axis=(1, 0))
    normal = _normalize_normals(np.cross(dx[sites], dy[sites]))
    rays = world[sites] - camera["eye"]
    distance = np.linalg.norm(rays, axis=-1)
    direction = rays / distance[:, None]
    cosine = np.maximum(np.abs(np.sum(direction * normal, axis=-1)), 1e-6)
    pixel_angle = 2 * np.tan(np.radians(camera["fovDegrees"]) / 2) / world.shape[0]
    return distance * pixel_angle / world_size * np.sqrt(np.abs(normal[:, 1]) / cosine)


def _mip_chain(pixels):
    levels = [pixels.astype(np.float64) / 255]
    pixels = pixels.astype(np.uint16)
    while max(pixels.shape[:2]) > 1:
        height, width = pixels.shape[:2]
        rows = np.arange(max(height // 2, 1)) * 2
        cols = np.arange(max(width // 2, 1)) * 2
        next_rows = np.minimum(rows + 1, height - 1)
        next_cols = np.minimum(cols + 1, width - 1)
        pixels = (pixels[rows[:, None], cols] + pixels[rows[:, None], next_cols]
                  + pixels[next_rows[:, None], cols] + pixels[next_rows[:, None], next_cols] + 2) // 4
        levels.append(pixels.astype(np.float64) / 255)
    return tuple(levels)


@dataclass(frozen=True)
class _Texture:
    mips: tuple[np.ndarray, ...]
    transform: np.ndarray
    srgb: bool

    def sample(self, uv, footprint):
        uv = uv @ self.transform[:, :2].T + self.transform[:, 2]
        if footprint is None:
            sample = repeat_linear(self.mips[0], uv)
        else:
            height, width, components = self.mips[0].shape
            texels_per_uv = np.sqrt(height * width * abs(np.linalg.det(self.transform[:, :2])))
            lod = np.clip(np.log2(np.maximum(1e-8, footprint * texels_per_uv)), 0, len(self.mips) - 1)
            sample = np.zeros((len(uv), components), dtype=np.float64)
            for level, mip in enumerate(self.mips):
                fraction = 1 - np.abs(lod - level)
                used = fraction > 0
                sample[used] += fraction[used, None] * repeat_linear(mip, uv[used])
        return srgb_to_linear(sample) if self.srgb else sample


@dataclass(frozen=True)
class _Layer:
    channel: int
    textures: dict[str, _Texture]


@dataclass(frozen=True)
class TerrainLayerReference:
    world_size: float
    weight_texture: _Texture
    default_channel: int
    layers: tuple[_Layer, ...]

    def _uv(self, world):
        return world[..., [0, 2]] / self.world_size + 0.5

    def _normalized_weights(self, weights):
        total = weights.sum(axis=-1, keepdims=True)
        result = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
        result[..., self.default_channel] += total[..., 0] == 0
        return result

    def weights(self, world, footprint=None):
        """Return the normalized layer weights in RGBA channel order."""
        return self._normalized_weights(self.weight_texture.sample(self._uv(world), footprint))

    def evaluate(self, mode, world, weights=None, *, footprint=None, omitted_channel=None):
        """Return the filtered reference in the same encoding as the Deferred AOV."""
        uv = self._uv(world)
        if weights is None:
            weights = self.weights(world, footprint)
        if omitted_channel is not None:
            weights = weights.copy()
            weights[..., RGBA.index(omitted_channel)] = 0.0
            weights = self._normalized_weights(weights)

        role = TEXTURE_ROLES[mode]
        components = 1 if mode == "roughness" else 3
        value = np.zeros(uv.shape[:-1] + (components,), dtype=np.float64)
        for layer in self.layers:
            sample = layer.textures[role].sample(uv, footprint)
            if mode == "normalTexture":
                sample = _normalize_normals(sample * 2.0 - 1.0)
            value += weights[..., layer.channel:layer.channel + 1] * sample
        if mode == "normalTexture":
            return _normalize_normals(value) * 0.5 + 0.5
        if mode == "roughness":
            return np.clip(value, 0.001, 1.0)
        return value


def _asset_path(root, uri):
    if not uri.startswith("asset://"):
        raise ValueError(f"Expected an asset URI, got {uri}")
    path = (root / uri.removeprefix("asset://")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset escapes its resource root: {uri}")
    return path


def _texture(root, binding, transform, role):
    path = _asset_path(root, binding["uri"])
    if binding.get("texCoord", 0) != 0 or not np.allclose(binding["transform"], transform, atol=1e-9, rtol=0):
        raise ValueError(f"{path.name} UV transform does not match the authored terrain mapping")
    color_space = binding.get("colorSpace", "srgb" if role == "baseColor" else "linear")
    spaces = ("srgb", "linear", "lin_rec709") if role == "baseColor" else ("linear", "lin_rec709")
    if color_space not in spaces:
        raise ValueError(f"Unsupported {role} color space {color_space} on {path.name}")
    with Image.open(path) as image:
        if role == "weights" and image.mode != "RGBA":
            raise ValueError(f"Expected all four RGBA weight channels in {path.name}")
        texels = np.asarray(image.convert("RGBA" if role == "weights" else "RGB"), dtype=np.uint8)
    if role == "roughness":
        texels = texels[..., :1]
    return _Texture(_mip_chain(texels), np.asarray(transform).reshape(2, 3), color_space == "srgb")


def load_terrain_layer_reference(scene: Path):
    """Load bound texture files and check their UV transforms against authored world scales."""
    extras = json.loads(scene.read_text(encoding="utf-8"))["extras"]
    world_size = float(extras["metallicTerrainTiles"]["horizontalSize"])
    authored = extras["metallicTerrainLayers"]
    reference_layers = {layer["name"]: layer for layer in authored["layers"]}
    entries = json.loads(scene.with_suffix(".metallic_scene.json").read_text(encoding="utf-8"))["materials"]
    binding, = (entry["materialAsset"] for entry in entries
                if entry["sourceId"] == "main" and entry["materialIndex"] == 0)
    root = (scene.parent / binding["root"]).resolve()
    material = json.loads(_asset_path(root, binding["uri"]).read_text(encoding="utf-8"))["terrainLayers"]
    if (len(material["layers"]) != 4 or {entry["channel"] for entry in material["layers"]} != set(RGBA)
            or {entry["name"] for entry in material["layers"]} != reference_layers.keys()
            or material["defaultLayer"] != authored["defaultLayer"]):
        raise ValueError("Four-layer material names, channels or default layer differ from the authored tile metadata")

    width, height = authored["weightResolution"]
    transform = [(width - 1) / (2 * width), 0, 0.25 / width, 0, (height - 1) / (2 * height), 0.25 / height]
    weight_texture = _texture(root, material["weights"], transform, "weights")
    if weight_texture.mips[0].shape[:2] != (2 * height, 2 * width):
        raise ValueError("Mirrored weights must cover twice the heightfield resolution on each axis")

    layers = []
    for entry in material["layers"]:
        layer = reference_layers[entry["name"]]
        if entry["channel"] != layer["channel"]:
            raise ValueError(f"{layer['name']} channel differs from the authored terrain channel")
        repeat_x, repeat_z = world_size / np.asarray(layer["worldTileSize"])
        transform = [repeat_x, 0, -0.5 * repeat_x, 0, repeat_z, -0.5 * repeat_z]
        textures = {role: _texture(root, entry[role], transform, role) for role in TEXTURE_ROLES.values()}
        layers.append(_Layer(RGBA.index(entry["channel"]), textures))
    default_channel = RGBA.index(reference_layers[authored["defaultLayer"]]["channel"])
    return TerrainLayerReference(world_size, weight_texture, default_channel, tuple(layers))
