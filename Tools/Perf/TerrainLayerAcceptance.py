#!/usr/bin/env python3
"""Capture and verify the Houdini material band across a 16x16 terrain edge."""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, expanded_flight_camera, write_json
from RunTerrainStreamingRegression import capture
from VerifyTerrainStreamingRegression import (
    REPORT_NAME, camera_timeline, capture_geometry, decode_lods, lod_pixels, raw_pixels, read, require,
)

sys.path.insert(0, str(ROOT / "Tools"))
from BuildTerrainMesh import load_height_field


SCENE = ROOT / "Asset/MeshletCache/TerrainLayers16x16/terrain_layers_16x16.gltf"
WEIGHT = ROOT / "Asset/Terrain/TerrainRockWeight.exr"
OUTPUT = ROOT / "build/terrain-layer-acceptance"
TILES = ((11, 3), (11, 4))
SIZE = (640, 360)
LIMIT = 272
VIEWS = {84: "mixed", 94: "near", 104: "far"}
MODES = ("baseColor", "mappedNormal", "shadingNormal", "roughness", "final")


def prepare(output):
    require(SCENE.is_file() and Path(str(SCENE) + ".meshstream.bin").is_file(),
            "Bake MetallicTerrainLayers16x16VSMSample first")
    graph = read(GRAPH)
    template = graph["view"]["camera"]
    flight = [expanded_flight_camera(template, frame) for frame in range(85)]
    aim = [56., 14.71, -64.64214324951172]
    near = {**template, "eye": [56., 40., -62.64], "center": aim, "fovDegrees": 48.}
    far = {**template, "eye": [58., 120., -58.], "center": aim, "fovDegrees": 49.}
    events = [{"frame": frame, "camera": camera} for frame, camera in enumerate(flight) if frame > 0]
    events += [{"frame": 85, "camera": near, "cameraCut": True},
               {"frame": 95, "camera": far, "cameraCut": True}]

    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    nodes["VBuffer"].update(path=SCENE.as_posix(), streamAssetPath=SCENE.as_posix() + ".meshstream.bin",
                            maxResidentPages=LIMIT, visualization="lod", shadedDebugColors=False)
    nodes["Deferred"]["path"] = SCENE.as_posix()
    for mode in MODES:
        directory = output / mode
        nodes["Deferred"]["debugView"] = mode
        graph_filename = "TerrainLayers.metallic_graph.json"
        write_json(directory / graph_filename, graph)
        case = dict(id=mode, graph=graph_filename, scene=SCENE.as_posix(), width=SIZE[0], height=SIZE[1],
                    frames=max(VIEWS) + 1, warmupFrames=0, camera=flight[0], events=events,
                    captureFrames=list(VIEWS), streamingEvidenceFrames=list(range(85)) + [94, 104],
                    output="FinalBlit.color" if mode == "final" else "VBuffer.color",
                    comparisonOutput="Deferred.color", requireShadow=True, temporalJitter=False,
                    metadataOnly=True, allowResidentBudgetDeferral=True)
        write_json(directory / "Cases.json", dict(version=1, mode="diagnostic", cases=[case]))


def tile_bounds():
    gltf = read(SCENE)
    tiles = {tuple(tile["tile"]): tile["name"] for tile in gltf["extras"]["metallicTerrainTiles"]["tiles"]}
    meshes = {mesh["name"]: mesh for mesh in gltf["meshes"]}
    bounds = []
    for tile in TILES:
        primitive, = meshes[tiles[tile]]["primitives"]
        position = gltf["accessors"][primitive["attributes"]["POSITION"]]
        bounds.append(np.asarray([position["min"], position["max"]]))
    require(bounds[0][1, 2] == bounds[1][0, 2], "Selected terrain tiles no longer share their authored edge")
    return np.asarray(bounds)


def owners_on_edge(world, covered, bounds):
    owners = np.full(covered.shape, -1, dtype=np.int8)
    xz = world[..., [0, 2]]
    for index, (low, high) in enumerate(bounds[..., [0, 2]]):
        interior = np.all((low + 0.002 < xz) & (xz < high - 0.002), axis=-1)
        owners[covered & interior] = index
    return owners


def repeat_linear(image, uv):
    height, width = image.shape[:2]
    sample = uv * [width, height] - 0.5
    lower = np.floor(sample).astype(np.int64)
    x, y = lower[..., 0], lower[..., 1]
    u, v = np.moveaxis((sample - lower)[..., None], -2, 0)
    left = image[y % height, x % width] * (1.0 - v) + image[(y + 1) % height, x % width] * v
    right = image[y % height, (x + 1) % width] * (1.0 - v) + image[(y + 1) % height, (x + 1) % width] * v
    return left * (1.0 - u) + right * u


def terrain_texture(name):
    path = SCENE.with_name(f"{SCENE.stem}_{name}.png")
    with Image.open(path) as pixels:
        return np.asarray(pixels.convert("RGBA"), dtype=np.float64) / 255.0


def layer_reference(world, weight, grass, rock):
    xz = world[..., [0, 2]]
    grass_value = repeat_linear(grass, xz / 2.)
    rock_value = repeat_linear(rock, xz / 6.)
    return grass_value * (1.0 - weight[..., None]) + rock_value * weight[..., None]


def srgb_to_linear(color):
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def save_aov(directory, name, rgb, mode):
    rgb = np.clip(rgb, 0.0, 1.0)
    if mode == "baseColor":
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
    filename = f"{name}-aov.png"
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8)).save(directory / filename)
    return filename


def band_reference(world, weight_field):
    # The Houdini export samples both world boundaries. Match its pixel centers.
    height, width = weight_field.shape
    grid_uv = np.clip((world[..., [0, 2]] + 128.) / 256., 0., 1.)
    filtered_uv = (grid_uv * [width - 1, height - 1] + 0.5) / [width, height]
    return repeat_linear(weight_field[..., None], filtered_uv)[..., 0]


def edge_pairs(world, owners, values, levels, weight, plane):
    samples, jumps, local, mixed = [], [], [], []
    for far, a, b, beyond in ((np.s_[:, :-3], np.s_[:, 1:-2], np.s_[:, 2:-1], np.s_[:, 3:]),
                              (np.s_[:-3, :], np.s_[1:-2, :], np.s_[2:-1, :], np.s_[3:, :])):
        opposite = (owners[a] >= 0) & (owners[b] >= 0) & (owners[a] != owners[b])
        opposite &= (np.abs(world[a][..., 2] - plane) < 1.) & (np.abs(world[b][..., 2] - plane) < 1.)
        opposite &= np.linalg.norm(world[a] - world[b], axis=-1) < 1.5
        pair_weight = (weight[a] + weight[b]) * 0.5
        samples.append(pair_weight[opposite])
        witnesses = opposite & (pair_weight > 0.1) & (levels[a] != levels[b])
        left = np.where(owners[a] == 0, levels[a], levels[b])[witnesses]
        right = np.where(owners[b] == 1, levels[b], levels[a])[witnesses]
        mixed.append(np.column_stack((left, right)))
        supported = opposite & (owners[far] == owners[a]) & (owners[beyond] == owners[b])
        jump = values[b] - values[a]
        before = values[a] - values[far]
        after = values[beyond] - values[b]
        jumps.append(np.linalg.norm((jump - (before + after) * 0.5)[supported], axis=-1))
        local.append((np.linalg.norm(before[supported], axis=-1) + np.linalg.norm(after[supported], axis=-1)) * 0.5)
    return np.concatenate(samples), np.concatenate(mixed), np.concatenate(jumps), np.concatenate(local)


def inspect(directory, case, bounds, weight_field, layer_textures):
    mode = case["id"]
    spec = case["spec"]
    cameras, _ = camera_timeline(spec)
    nodes = {node["name"]: node["properties"] for node in case["resolvedGraph"]["nodes"]}
    require(case["status"] == "passed" and Path(spec["scene"]).resolve() == SCENE.resolve()
            and not spec["temporalJitter"] and nodes["Deferred"]["debugView"] == mode
            and nodes["VBuffer"]["maxResidentPages"] == LIMIT and nodes["VBuffer"]["autoLod"],
            f"{mode}: unexpected acceptance case or graph settings")
    require([shot["frame"] for shot in case["captures"]] == list(VIEWS), f"{mode}: incomplete captures")
    require(len(case["frames"]) == spec["frames"], f"{mode}: incomplete frame sequence")
    for row in case["frames"]:
        stream, = [item for item in row["streaming"] if item["passName"] == "VBuffer"]
        require(stream["loadFailures"] == stream["requestOverflows"] == 0
                and stream["allocationFailures"] == stream["allocationBudgetDeferrals"]
                and stream["maxResidentPages"] == LIMIT and stream["activePages"] <= LIMIT
                and row["readiness"]["requiredPages"] == row["readiness"]["completedPages"] == 257,
                f"{mode}: failed streaming or fallback at frame {row['frame']}")

    frames = {}
    reference = {}
    for shot in case["captures"]:
        frame = shot["frame"]
        label = f"{mode} {VIEWS[frame]}"
        world, covered = capture_geometry(directory, shot, cameras[frame], SIZE)
        owners = owners_on_edge(world, covered, bounds)
        weight = band_reference(world, weight_field)
        rgb = raw_pixels(directory, shot, "Deferred.color", SIZE, "<f4", "RGBA32Sfloat", 1, 4)[0, ..., :3]
        require(np.isfinite(rgb).all(), f"{label}: nonfinite surface output")
        if mode == "final":
            require(np.percentile(rgb[owners >= 0], 95) > 0, f"{label}: unlit or empty terrain")
            frames[VIEWS[frame]] = {"image": shot["file"], "terrainPixels": int(np.count_nonzero(owners >= 0))}
            continue

        levels = decode_lods(lod_pixels(directory, shot, SIZE))
        require(np.all(levels[owners >= 0] >= 0), f"{label}: mismatched LOD palette")
        weights, mixed, jumps, local = edge_pairs(world, owners, rgb, levels, weight, bounds[0][1, 2])
        require(np.count_nonzero(weights > 0.9) >= 10 and np.count_nonzero(weights < 0.1) >= 10
                and np.count_nonzero((weights > 0.1) & (weights < 0.9)) >= 5,
                f"{label}: missing rock, grass, or blend across the authored tile edge")
        stream, = [item for item in case["frames"][frame]["streaming"] if item["passName"] == "VBuffer"]
        require(frame != 84 or (len(mixed) >= 10 and stream["activePages"] == LIMIT),
                f"{label}: the band needs actual mixed-LOD adjacent pixels at the saturated 272-page limit")
        require(jumps.size >= 25, f"{label}: too few locally supported cross-edge comparisons")
        excess = float(np.percentile(jumps, 95))
        ordinary = float(np.percentile(local, 95))
        require(excess <= max(0.06, 2.0 * ordinary), f"{label}: seam discontinuity {excess:.3g} vs {ordinary:.3g}")
        mixes, counts = np.unique(mixed, axis=0, return_counts=True)
        lod_pairs = [{"lodsByTile": pair.tolist(), "pairs": int(count)} for pair, count in zip(mixes, counts)]
        result = dict(rockPairs=int(np.count_nonzero(weights > 0.9)), grassPairs=int(np.count_nonzero(weights < 0.1)),
                      blendPairs=int(np.count_nonzero((weights > 0.1) & (weights < 0.9))), mixedBandPairs=len(mixed),
                      mixedBandLodPairs=lod_pairs, activePages=stream["activePages"],
                      seamExcessP95=excess, nearbyGradientP95=ordinary, lodImage=shot["file"])
        near_edge = (owners >= 0) & (np.abs(world[..., 2] - bounds[0][1, 2]) < 2.)
        result["aovImage"] = save_aov(directory, f"{mode}-{VIEWS[frame]}", rgb, mode)
        for owner in range(len(bounds)):
            site = near_edge & (owners == owner)
            rock = rgb[site & (weight > 0.98)]
            grass = rgb[site & (weight < 0.02)]
            require(len(rock) >= 20 and len(grass) >= 20, f"{label}: incomplete layer samples on terrain tile {owner}")
            separation = float(np.linalg.norm(np.median(rock, axis=0) - np.median(grass, axis=0)))
            result[f"tile{owner}EndpointSeparation"] = separation
            if mode == "baseColor":
                require(separation >= 0.07, f"{label}: rock and grass base colors are indistinguishable")
        if mode == "baseColor":
            expect = layer_reference(world[near_edge], weight[near_edge], *layer_textures["baseColor"])
            mean_absolute_error = float(np.mean(np.abs(rgb[near_edge] - expect)))
            require(mean_absolute_error <= 0.045, f"{label}: base color differs from linear-sRGB layer sampling")
            result["worldScaleBaseColorMeanAbsoluteError"] = mean_absolute_error
            if VIEWS[frame] == "near":
                # Correlation keeps the repeating world-space phase check meaningful
                # when filtering attenuates contrast relative to the mip-zero reference.
                for material, pure in (("grass", weight[near_edge] < 0.02), ("rock", weight[near_edge] > 0.98)):
                    actual = rgb[near_edge][pure, 0]
                    expected = expect[pure, 0]
                    correlation = float(np.corrcoef(actual, expected)[0, 1])
                    require(correlation >= 0.25, f"{label}: {material} texture lost its world-space phase")
                    result[f"{material}WorldTextureCorrelation"] = correlation
        if mode == "roughness":
            value = rgb[..., 0]
            expect = layer_reference(world[near_edge], weight[near_edge], *layer_textures["roughness"])[..., 0]
            mean_absolute_error = float(np.mean(np.abs(value[near_edge] - expect)))
            require(mean_absolute_error <= 0.043, f"{label}: roughness differs from world-scale layer sampling")
            require(np.ptp(rgb[near_edge], axis=-1).max() < 1e-6, f"{label}: roughness debug output is not scalar")
            result["worldScaleRoughnessMeanAbsoluteError"] = mean_absolute_error
        if mode in ("mappedNormal", "shadingNormal"):
            reference[frame] = (world, owners, weight, rgb, levels)
        frames[VIEWS[frame]] = result
    return frames, reference


def load_case(directory, mode):
    document = read(directory / REPORT_NAME)
    require(document["status"] == "passed" and document["mode"] == "diagnostic" and document["validation"]
            and document["validationMessages"]["count"] == 0 and not document["validationMessages"]["captureFailed"],
            f"{mode}: expected a passing diagnostic receipt with zero Vulkan validation messages")
    case, = document["cases"]
    require(case["id"] == mode, f"{mode}: unexpected terrain layer AOV in receipt")
    return case


def verify(output):
    bounds = tile_bounds()
    weight_field, _, _ = load_height_field(WEIGHT, "R")
    textures = {
        "baseColor": tuple(srgb_to_linear(terrain_texture(f"{layer}_color")[..., :3]) for layer in ("grass", "rock")),
        "roughness": tuple(terrain_texture(f"{layer}_normal")[..., 3:4] for layer in ("grass", "rock")),
    }
    report = {"passed": True, "tiles": TILES, "planeZ": float(bounds[0][1, 2]), "modes": {}}
    normals = {}
    for mode in MODES:
        directory = output / mode / "capture"
        case = load_case(directory, mode)
        result, normals[mode] = inspect(directory, case, bounds, weight_field, textures)
        for view in result.values():
            for key in ("aovImage", "lodImage", "image"):
                if key in view:
                    view[key] = (directory.relative_to(output) / view[key]).as_posix()
        report["modes"][mode] = result
    for frame, label in VIEWS.items():
        world, owners, weight, mapped, levels = normals["mappedNormal"][frame]
        other_world, other_owners, _, authored, other_levels = normals["shadingNormal"][frame]
        match = (owners >= 0) & (owners == other_owners) & (levels == other_levels)
        match &= np.linalg.norm(world - other_world, axis=-1) < 0.002
        for material, mask in (("grass", weight < 0.02), ("rock", weight > 0.98)):
            require(np.count_nonzero(match & mask) >= 20, f"{label}: missing matching {material} geometry for normals")
            p90 = float(np.percentile(np.linalg.norm((mapped - authored)[match & mask], axis=-1), 90))
            require(p90 >= 0.003, f"{label}: {material} did not affect the OpenPBR mapped normal")
            report["modes"]["mappedNormal"][label][f"{material}NormalDifferenceP90"] = p90
    report_path = output / "TerrainLayerAcceptance.json"
    write_json(report_path, report)
    print(f"Terrain layer acceptance passed. {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, help="Run directory; defaults to a fresh directory for run")
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    try:
        output = args.output or OUTPUT
        if args.action == "run" and args.output is None:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="run-", dir=OUTPUT))
        output = output.resolve()
        print(f"Terrain layer acceptance artifacts: {output}", flush=True)
        if args.action != "verify":
            prepare(output)
        if args.action == "run":
            require(args.executable is not None and args.executable.is_file(), "Build and provide MetallicRHITests")
            for mode in MODES:
                capture(args.executable.resolve(), output / mode)
        if args.action != "prepare":
            verify(output)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Terrain layer acceptance failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
