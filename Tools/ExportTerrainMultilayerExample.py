#!/usr/bin/env python3
"""Generate four terrain layer textures and an authoring asset from the sample rock mask."""

import argparse
import json
from pathlib import Path
import tempfile

import Imath
import numpy as np
import OpenEXR

from BuildTerrainLayerTextures import layer_pixels, write_png
from BuildTerrainMesh import TerrainBuildError, load_height_field, write_atomic


def export_example(rock_weight: Path, output: Path) -> None:
    rock, _, (min_x, min_z, max_x, max_z) = load_height_field(rock_weight, "R")
    if rock.min() < 0 or rock.max() > 1:
        raise TerrainBuildError("rock weights must be in [0, 1]")
    weight_path = output.parent / "weights.exr"
    if rock_weight.resolve() in (output.resolve(), weight_path.resolve()):
        raise TerrainBuildError("the sample export requires a separate output directory from its input mask")
    rows, columns = rock.shape
    x = np.linspace(-128, 128, columns, dtype=np.float32)
    z = np.linspace(-128, 128, rows, dtype=np.float32)[:, None]
    # The original winding rock band meets the other two layers around the sample's shared edge.
    blend = np.clip((z + 82.0 - 8.0 * np.sin(x * 0.025)) / 36.0, 0.0, 1.0)
    blend = blend * blend * (3 - 2 * blend)
    channels = dict(zip("RGBA", (rock * (1 - blend), (1 - rock) * (1 - blend), rock * blend, (1 - rock) * blend)))
    output.parent.mkdir(parents=True, exist_ok=True)
    header = OpenEXR.Header(columns, rows)
    header["dataWindow"] = Imath.Box2i(Imath.V2i(min_x, min_z), Imath.V2i(max_x, max_z))
    header["displayWindow"] = header["dataWindow"]
    header["channels"] = {channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for channel in channels}
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        temporary = Path(staging) / weight_path.name
        encoded = OpenEXR.OutputFile(str(temporary), header)
        try:
            encoded.writePixels({channel: weights.tobytes() for channel, weights in channels.items()})
        finally:
            encoded.close()
        temporary.replace(weight_path)

    layers = []
    for name, channel, period, texture in (
        ("rock", "R", 6, "rock"), ("grass", "G", 2, "grass"),
        ("gravel", "B", [3, 5], "rock"), ("snow", "A", 8, "grass"),
    ):
        color, packed = layer_pixels(texture)
        normal = packed[..., :3]
        roughness = packed[..., 3]
        if name == "gravel":
            color = color * [0.85, 0.75, 0.63] + [0.17, 0.12, 0.07]
            roughness = roughness + 0.12
        if name == "snow":
            color = color * 0.45 + [0.70, 0.70, 0.78]
            roughness = roughness - 0.21
        layer = {"name": name, "channel": channel, "worldTileSize": period}
        for role, pixels in (("baseColor", color), ("normal", normal), ("roughness", roughness)):
            filename = f"{name}_{role}.png"
            write_png(output.parent / filename, pixels)
            layer[role] = filename
        layers.append(layer)
    asset = {"type": "Metallic.TerrainLayers", "version": 1, "weightMap": weight_path.name,
             "defaultLayer": "grass", "layers": layers}
    write_atomic(output, (json.dumps(asset, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rock_weight", type=Path, help="Source linear R OpenEXR rock mask")
    parser.add_argument("--output", required=True, type=Path, help="Metallic.TerrainLayers authoring JSON")
    args = parser.parse_args()
    try:
        export_example(args.rock_weight, args.output)
    except (OSError, TerrainBuildError) as error:
        parser.exit(1, f"Terrain layer example export failed: {error}\n")


if __name__ == "__main__":
    main()
