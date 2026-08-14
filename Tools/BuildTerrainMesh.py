#!/usr/bin/env python3
"""Convert a single-channel OpenEXR height field to a static terrain glTF.

This is the P0 bridge into Metallic's existing glTF -> cluster LOD ->
MeshletStreamAsset pipeline. The generated mesh is Y-up and keeps the source
height values; world scaling is explicit on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote


class TerrainBuildError(RuntimeError):
    pass


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return value


def finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("expected a finite number")
    return value


def non_negative_float(text: str) -> float:
    value = finite_float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return value


def load_height_field(path: Path, requested_channel: str | None) -> tuple[Any, str, tuple[int, int, int, int]]:
    try:
        import Imath
        import numpy as np
        import OpenEXR
    except ImportError as error:
        raise TerrainBuildError(
            "OpenEXR height import requires 'OpenEXR' and 'numpy'; install "
            "Tools/requirements-terrain.txt"
        ) from error

    try:
        image = OpenEXR.InputFile(str(path))
    except Exception as error:
        raise TerrainBuildError(f"cannot open OpenEXR height field '{path}': {error}") from error

    try:
        header = image.header()
        data_window = header["dataWindow"]
        min_x = int(data_window.min.x)
        min_y = int(data_window.min.y)
        max_x = int(data_window.max.x)
        max_y = int(data_window.max.y)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if width < 2 or height < 2:
            raise TerrainBuildError(
                f"height field must be at least 2x2 pixels, got {width}x{height}"
            )

        channel_names = list(header.get("channels", {}).keys())
        if not channel_names:
            raise TerrainBuildError("OpenEXR height field has no channels")

        if requested_channel is not None:
            channel = next(
                (name for name in channel_names if name.casefold() == requested_channel.casefold()),
                None,
            )
            if channel is None:
                available = ", ".join(channel_names)
                raise TerrainBuildError(
                    f"OpenEXR channel '{requested_channel}' was not found; available: {available}"
                )
        else:
            channel = None
            for preferred in ("Y", "R", "A", "A.A", "Z"):
                channel = next(
                    (name for name in channel_names if name.casefold() == preferred.casefold()),
                    None,
                )
                if channel is not None:
                    break
            if channel is None:
                channel = channel_names[0]

        channel_info = header["channels"][channel]
        if int(channel_info.xSampling) != 1 or int(channel_info.ySampling) != 1:
            raise TerrainBuildError(
                f"subsampled OpenEXR channels are unsupported ({channel}: "
                f"{channel_info.xSampling}x{channel_info.ySampling})"
            )

        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = image.channel(channel, pixel_type)
        expected_bytes = width * height * 4
        if len(raw) != expected_bytes:
            raise TerrainBuildError(
                f"OpenEXR channel byte count is invalid: expected {expected_bytes}, got {len(raw)}"
            )
        values = np.frombuffer(raw, dtype=np.float32).reshape(height, width).copy()
    finally:
        image.close()

    finite_mask = np.isfinite(values)
    if not bool(finite_mask.all()):
        invalid_count = int(values.size - np.count_nonzero(finite_mask))
        raise TerrainBuildError(f"height field contains {invalid_count} non-finite samples")

    return values, channel, (min_x, min_y, max_x, max_y)


def build_terrain_mesh(
    source_heights: Any,
    horizontal_size: float,
    height_scale: float,
    height_offset: float,
    skirt_depth: float,
    flip_z: bool,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import numpy as np

    heights = np.flip(source_heights, axis=0) if flip_z else source_heights
    heights = heights.astype(np.float32, copy=False)
    row_count, column_count = heights.shape
    spacing_x = horizontal_size / float(column_count - 1)
    spacing_z = horizontal_size / float(row_count - 1)

    x_coordinates = np.linspace(
        -horizontal_size * 0.5,
        horizontal_size * 0.5,
        column_count,
        dtype=np.float32,
    )
    z_coordinates = np.linspace(
        -horizontal_size * 0.5,
        horizontal_size * 0.5,
        row_count,
        dtype=np.float32,
    )
    world_x, world_z = np.meshgrid(x_coordinates, z_coordinates)
    world_y = heights * np.float32(height_scale) + np.float32(height_offset)
    positions = np.stack((world_x, world_y, world_z), axis=-1).reshape(-1, 3)

    gradient_edge_order = 2 if min(row_count, column_count) >= 3 else 1
    gradient_z, gradient_x = np.gradient(
        world_y,
        spacing_z,
        spacing_x,
        edge_order=gradient_edge_order,
    )
    normals = np.stack((-gradient_x, np.ones_like(world_y), -gradient_z), axis=-1)
    normal_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = (normals / np.maximum(normal_lengths, np.float32(1.0e-20))).reshape(-1, 3)

    u_coordinates = np.linspace(0.0, 1.0, column_count, dtype=np.float32)
    v_coordinates = np.linspace(0.0, 1.0, row_count, dtype=np.float32)
    texcoord_u, texcoord_v = np.meshgrid(u_coordinates, v_coordinates)
    texcoords = np.stack((texcoord_u, texcoord_v), axis=-1).reshape(-1, 2)

    grid_rows = np.arange(row_count - 1, dtype=np.uint32)[:, None]
    grid_columns = np.arange(column_count - 1, dtype=np.uint32)[None, :]
    top_left = grid_rows * np.uint32(column_count) + grid_columns
    top_right = top_left + np.uint32(1)
    bottom_left = top_left + np.uint32(column_count)
    bottom_right = bottom_left + np.uint32(1)
    indices = np.stack(
        (
            top_left,
            bottom_left,
            top_right,
            top_right,
            bottom_left,
            bottom_right,
        ),
        axis=-1,
    ).reshape(-1)

    surface_vertex_count = int(positions.shape[0])
    surface_triangle_count = int(indices.size // 3)
    skirt_triangle_count = 0
    if skirt_depth > 0.0:
        north = np.arange(0, column_count, dtype=np.uint32)
        east = np.arange(column_count - 1, row_count * column_count, column_count, dtype=np.uint32)
        south = np.arange(
            row_count * column_count - 1,
            (row_count - 1) * column_count - 1,
            -1,
            dtype=np.int64,
        ).astype(np.uint32)
        west = np.arange(
            (row_count - 1) * column_count,
            -1,
            -column_count,
            dtype=np.int64,
        ).astype(np.uint32)
        edges = (
            (north, (0.0, 0.0, -1.0)),
            (east, (1.0, 0.0, 0.0)),
            (south, (0.0, 0.0, 1.0)),
            (west, (-1.0, 0.0, 0.0)),
        )

        position_parts = [positions]
        normal_parts = [normals]
        texcoord_parts = [texcoords]
        index_parts = [indices]
        next_vertex = surface_vertex_count
        for edge_indices, outward_normal in edges:
            edge_positions = positions[edge_indices]
            lower_positions = edge_positions.copy()
            lower_positions[:, 1] -= np.float32(skirt_depth)
            edge_vertex_count = int(edge_indices.size)
            position_parts.extend((edge_positions, lower_positions))
            edge_normals = np.tile(
                np.asarray(outward_normal, dtype=np.float32),
                (edge_vertex_count * 2, 1),
            )
            normal_parts.append(edge_normals)
            edge_texcoords = texcoords[edge_indices]
            texcoord_parts.extend((edge_texcoords, edge_texcoords))

            segment = np.arange(edge_vertex_count - 1, dtype=np.uint32)
            top_start = np.uint32(next_vertex)
            bottom_start = np.uint32(next_vertex + edge_vertex_count)
            top_0 = top_start + segment
            top_1 = top_0 + np.uint32(1)
            bottom_0 = bottom_start + segment
            bottom_1 = bottom_0 + np.uint32(1)
            edge_triangles = np.stack(
                (top_0, top_1, bottom_0, top_1, bottom_1, bottom_0),
                axis=-1,
            ).reshape(-1)
            index_parts.append(edge_triangles)
            skirt_triangle_count += int(edge_triangles.size // 3)
            next_vertex += edge_vertex_count * 2

        positions = np.concatenate(position_parts, axis=0)
        normals = np.concatenate(normal_parts, axis=0)
        texcoords = np.concatenate(texcoord_parts, axis=0)
        indices = np.concatenate(index_parts, axis=0)

    metadata = {
        "sourceResolution": [int(column_count), int(row_count)],
        "surfaceVertexCount": surface_vertex_count,
        "surfaceTriangleCount": surface_triangle_count,
        "skirtTriangleCount": skirt_triangle_count,
        "vertexCount": int(positions.shape[0]),
        "triangleCount": int(indices.size // 3),
        "sourceHeightMin": float(np.min(source_heights)),
        "sourceHeightMax": float(np.max(source_heights)),
        "worldHeightMin": float(np.min(positions[:, 1])),
        "worldHeightMax": float(np.max(positions[:, 1])),
    }
    return positions, normals, texcoords, indices, metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_bytes(data)
    os.replace(temporary_path, path)


def write_terrain_gltf(
    output_path: Path,
    binary_path: Path,
    source_path: Path,
    source_channel: str,
    data_window: tuple[int, int, int, int],
    positions: Any,
    normals: Any,
    texcoords: Any,
    indices: Any,
    metadata: dict[str, Any],
    horizontal_size: float,
    height_scale: float,
    height_offset: float,
    skirt_depth: float,
    flip_z: bool,
) -> None:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = bytearray()
    buffer_views: list[dict[str, Any]] = []

    def append_buffer_view(array: Any, target: int) -> int:
        padding = (-len(payload)) % 4
        if padding:
            payload.extend(b"\0" * padding)
        offset = len(payload)
        data = array.tobytes(order="C")
        payload.extend(data)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data),
                "target": target,
            }
        )
        return len(buffer_views) - 1

    positions = np.ascontiguousarray(positions, dtype="<f4")
    normals = np.ascontiguousarray(normals, dtype="<f4")
    texcoords = np.ascontiguousarray(texcoords, dtype="<f4")
    indices = np.ascontiguousarray(indices, dtype="<u4")
    position_view = append_buffer_view(positions, 34962)
    normal_view = append_buffer_view(normals, 34962)
    texcoord_view = append_buffer_view(texcoords, 34962)
    index_view = append_buffer_view(indices, 34963)

    position_min = [float(value) for value in np.min(positions, axis=0)]
    position_max = [float(value) for value in np.max(positions, axis=0)]
    relative_binary = os.path.relpath(binary_path, output_path.parent).replace(os.sep, "/")
    terrain_name = output_path.stem
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Metallic Tools/BuildTerrainMesh.py",
        },
        "scene": 0,
        "scenes": [{"name": terrain_name, "nodes": [0]}],
        "nodes": [{"name": terrain_name, "mesh": 0}],
        "meshes": [
            {
                "name": terrain_name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "TerrainP0",
                "doubleSided": False,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.18, 0.42, 0.12, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
            }
        ],
        "buffers": [{"uri": quote(relative_binary, safe="/"), "byteLength": len(payload)}],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": position_min,
                "max": position_max,
            },
            {
                "bufferView": normal_view,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {
                "bufferView": texcoord_view,
                "componentType": 5126,
                "count": int(texcoords.shape[0]),
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": int(indices.size),
                "type": "SCALAR",
                "min": [int(np.min(indices))],
                "max": [int(np.max(indices))],
            },
        ],
        "extras": {
            "metallicTerrainP0": {
                "source": source_path.name,
                "sourceSha256": sha256_file(source_path),
                "sourceChannel": source_channel,
                "sourceDataWindow": list(data_window),
                "horizontalSize": horizontal_size,
                "heightScale": height_scale,
                "heightOffset": height_offset,
                "skirtDepth": skirt_depth,
                "flipZ": flip_z,
                **metadata,
            }
        },
    }

    write_atomic(binary_path, bytes(payload))
    json_bytes = (json.dumps(gltf, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_atomic(output_path, json_bytes)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a single-channel OpenEXR height field into a Y-up glTF terrain mesh "
            "for Metallic's existing MeshletStreamAsset builder."
        )
    )
    parser.add_argument("heightmap", type=Path, help="source OpenEXR height field")
    parser.add_argument("--output", type=Path, help="output .gltf path")
    parser.add_argument("--binary-output", type=Path, help="output glTF .bin path")
    parser.add_argument("--channel", help="OpenEXR channel name (auto-selected by default)")
    parser.add_argument(
        "--horizontal-size",
        type=positive_float,
        default=256.0,
        help="terrain width and depth in world units (default: 256)",
    )
    parser.add_argument(
        "--height-scale",
        type=finite_float,
        default=16.0,
        help="world units per source height unit (default: 16)",
    )
    parser.add_argument(
        "--height-offset",
        type=finite_float,
        default=0.0,
        help="world-space Y offset applied after scaling (default: 0)",
    )
    parser.add_argument(
        "--skirt-depth",
        type=non_negative_float,
        default=0.0,
        help="optional outer boundary skirt depth in world units (default: 0)",
    )
    parser.add_argument(
        "--flip-z",
        action="store_true",
        help="reverse source scanline order before mapping rows to +Z",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_path = arguments.heightmap
    if not source_path.is_file():
        print(f"Terrain build failed: source height field does not exist: {source_path}", file=sys.stderr)
        return 1

    output_path = arguments.output
    if output_path is None:
        output_path = source_path.with_name(source_path.stem + "_terrain.gltf")
    if output_path.suffix.casefold() != ".gltf":
        print("Terrain build failed: --output must use the .gltf extension", file=sys.stderr)
        return 1
    binary_path = arguments.binary_output or output_path.with_suffix(".bin")

    try:
        heights, channel, data_window = load_height_field(source_path, arguments.channel)
        positions, normals, texcoords, indices, metadata = build_terrain_mesh(
            heights,
            arguments.horizontal_size,
            arguments.height_scale,
            arguments.height_offset,
            arguments.skirt_depth,
            arguments.flip_z,
        )
        write_terrain_gltf(
            output_path,
            binary_path,
            source_path,
            channel,
            data_window,
            positions,
            normals,
            texcoords,
            indices,
            metadata,
            arguments.horizontal_size,
            arguments.height_scale,
            arguments.height_offset,
            arguments.skirt_depth,
            arguments.flip_z,
        )
    except (OSError, TerrainBuildError, ValueError) as error:
        print(f"Terrain build failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Built terrain glTF '{output_path}': "
        f"source={metadata['sourceResolution'][0]}x{metadata['sourceResolution'][1]} "
        f"channel={channel} vertices={metadata['vertexCount']} "
        f"triangles={metadata['triangleCount']} "
        f"sourceHeight=[{metadata['sourceHeightMin']:.6g}, {metadata['sourceHeightMax']:.6g}] "
        f"worldHeight=[{metadata['worldHeightMin']:.6g}, {metadata['worldHeightMax']:.6g}]"
    )
    print(f"Wrote terrain vertex/index payload '{binary_path}' ({binary_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
