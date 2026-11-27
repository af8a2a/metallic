#!/usr/bin/env python3
"""Build an OpenPBR rock/grass material from a global Houdini EXR weight map."""

import io
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from BuildTerrainMesh import TerrainBuildError, load_height_field, sha256_file, write_atomic


LAYER_PERIODS = {"grass": 2.0, "rock": 6.0}
TEXTURE_SIZE = 256


def write_png(path: Path, values: np.ndarray) -> None:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    with io.BytesIO() as stream:
        Image.fromarray(encoded).save(stream, format="PNG")
        write_atomic(path, stream.getvalue())


def layer_pixels(name: str) -> tuple[np.ndarray, np.ndarray]:
    # Integer frequencies make the generated detail maps repeat at every mip.
    axis = (np.arange(TEXTURE_SIZE, dtype=np.float32) + 0.5) * (2 * np.pi / TEXTURE_SIZE)
    u, v = np.meshgrid(axis, axis)
    if name == "rock":
        relief = (0.52 * np.sin(3 * u + 5 * v + 0.7 * np.sin(2 * u - 3 * v))
                  + 0.30 * np.cos(5 * u - 9 * v) + 0.18 * np.sin(13 * u + 17 * v))
        color = [0.46, 0.43, 0.38]
        contrast, bump, roughness, variation = 0.085, 7.0, 0.50, 0.06
    else:
        relief = (0.46 * np.sin(7 * u + 2 * v + np.sin(4 * u - 3 * v))
                  + 0.34 * np.cos(15 * u - 11 * v) + 0.20 * np.sin(23 * u + 19 * v))
        color = [0.23, 0.39, 0.15]
        contrast, bump, roughness, variation = 0.07, 3.5, 0.82, 0.04
    albedo = np.asarray(color, dtype=np.float32) + contrast * relief[..., None]
    dx = np.roll(relief, -1, axis=1) - np.roll(relief, 1, axis=1)
    dy = np.roll(relief, -1, axis=0) - np.roll(relief, 1, axis=0)
    normal = np.stack((-bump * dx, -bump * dy, np.ones_like(relief)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    packed = np.concatenate((normal * 0.5 + 0.5, (roughness + variation * relief)[..., None]), axis=-1)
    return albedo, packed


def surface_program() -> dict:
    def ref(name):
        return {"ref": name}

    def expr(op, *args, **properties):
        return {"op": op, "args": list(args), **properties}

    def sample(slot):
        return expr("textureSampleLinear", ref("uv"), 0, texture=slot, footprint="RayCone")

    def blend(grass, rock):
        return expr("mix", grass, rock, ref("rockWeight"))

    return {"version": 2, "nodes": {
        "uv": {"op": "uv"},
        "grassColor": sample("baseColor"), "rockColor": sample("emissive"),
        "grassNormalRoughness": sample("normal"), "rockNormalRoughness": sample("metallicRoughness"),
        "rockWeight": expr("swizzle", sample("occlusion"), components="xxxx"),
    }, "outputs": {
        "surfaceBaseColor": blend(ref("grassColor"), ref("rockColor")),
        "surfaceMetallic": 0,
        "surfaceRoughness": blend(expr("swizzle", ref("grassNormalRoughness"), components="wwww"),
                                  expr("swizzle", ref("rockNormalRoughness"), components="wwww")),
        "normalTS": expr("normalize", blend(expr("normalMap", ref("grassNormalRoughness")),
                                            expr("normalMap", ref("rockNormalRoughness")))),
    }}


def build_layer_material(output: Path, weight_path: Path, terrain: dict) -> tuple[dict, dict]:
    weight, channel, window = load_height_field(weight_path, "R")
    if list(window) != terrain["sourceDataWindow"]:
        raise TerrainBuildError("weight and height EXR data windows must match")
    if weight.min() < 0 or weight.max() > 1:
        raise TerrainBuildError("rock weights must be in [0, 1]")
    if terrain["flipZ"]:
        weight = weight[::-1]
    rows, columns = weight.shape
    size = terrain["horizontalSize"]
    stem = output.stem

    def resource(filename, scale_x, scale_y, offset_x=0, offset_y=0):
        return {"uri": "asset://" + quote(filename),
                "transform": [scale_x, 0, offset_x, 0, scale_y, offset_y]}

    resources = {}
    # The Value IR's slots are fixed. Emission and AO remain neutral and the
    # program supplies the final OpenPBR metallic, roughness and normal inputs.
    for layer, color_slot, normal_slot in (("grass", "baseColorTexture", "normalTexture"),
                                          ("rock", "emissiveTexture", "metallicRoughnessTexture")):
        albedo, packed = layer_pixels(layer)
        repeat = size / LAYER_PERIODS[layer]
        color_path = f"{stem}_{layer}_color.png"
        normal_path = f"{stem}_{layer}_normal.png"
        write_png(output.with_name(color_path), albedo)
        write_png(output.with_name(normal_path), packed)
        offset = -0.5 * repeat
        resources[color_slot] = resource(color_path, repeat, repeat, offset, offset) | {"colorSpace": "srgb"}
        resources[normal_slot] = resource(normal_path, repeat, repeat, offset, offset)

    # Mirroring the global map keeps filtering at the outer heightfield edges
    # local with the existing repeating sampler. Terrain UV 0 and 1 address the
    # first and last original texel centers in the mirrored texture.
    mirrored = np.concatenate((weight, weight[:, ::-1]), axis=1)
    mirrored = np.concatenate((mirrored, mirrored[::-1]), axis=0)
    weight_name = f"{stem}_weight.png"
    write_png(output.with_name(weight_name), mirrored)
    resources["occlusionTexture"] = resource(weight_name, (columns - 1) / (2 * columns), (rows - 1) / (2 * rows),
                                              0.25 / columns, 0.25 / rows)

    definition = output.with_suffix(".materialdef")
    instance = output.with_suffix(".material")
    material_def = {"type": "Metallic.MaterialDefinition", "version": 1, "definitionVersion": 1,
                    "implementation": "OpenPBR.Value", "defaults": {}, "surfaceProgram": surface_program()}
    material_instance = {"type": "Metallic.MaterialInstance", "version": 1, "definitionVersion": 1,
                         "definition": "asset://" + quote(definition.name),
                         "parameters": {"metalness": 0, "occlusionStrength": 0, "emission": [0, 0, 0]},
                         "resources": resources, "features": {"doubleSided": True}}
    for path, doc in ((definition, material_def), (instance, material_instance)):
        write_atomic(path, (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    binding = {"sourceId": "main", "materialIndex": 0, "sourceName": "Terrain Rock and Grass",
               "materialAsset": {"uri": "asset://" + quote(instance.name), "root": "."}}
    metadata = {"weightSource": weight_path.name, "weightSourceSha256": sha256_file(weight_path),
                "weightChannel": channel, "weightResolution": [columns, rows],
                "rockWeight": "R", "grassWeight": "1-R", "layerPeriod": LAYER_PERIODS}
    return binding, metadata
