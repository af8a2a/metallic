#!/usr/bin/env python3
"""Capture and verify the Houdini material band across a 16x16 terrain edge."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, expanded_flight_camera, write_json
from RunTerrainStreamingRegression import capture
from TerrainLayerReference import load_terrain_layer_reference, ray_cone_uv_footprint, repeat_linear, srgb_to_linear
from VerifyTerrainStreamingRegression import (
    REPORT_NAME, camera_timeline, capture_geometry, decode_lods, lod_pixels, raw_pixels, read, require,
)

sys.path.insert(0, str(ROOT / "Tools"))
from BuildTerrainMesh import load_height_field


WEIGHT = ROOT / "Asset/Terrain/TerrainRockWeight.exr"
OUTPUT = ROOT / "build/terrain-layer-acceptance"
TILES = ((11, 3), (11, 4))
SIZE = (640, 360)
LIMIT = 272
VIEWS = {84: "mixed", 94: "near", 104: "far"}
MODES = ("baseColor", "mappedNormal", "shadingNormal", "roughness", "final")
REFERENCE_MODES = ("baseColor", "normalTexture", "roughness")
REFERENCE_TOLERANCE = 0.015
REFERENCE_SEAM_TOLERANCE = 0.02
FOUR_LAYER_WEIGHT = 0.02


@dataclass(frozen=True)
class Sample:
    name: str
    scene: Path
    bake_target: str
    modes: tuple[str, ...]


SAMPLES = {sample.name: sample for sample in (
    Sample("legacy", ROOT / "Asset/MeshletCache/TerrainLayers16x16/terrain_layers_16x16.gltf",
           "MetallicTerrainLayers16x16VSMSample", MODES),
    Sample("multilayer", ROOT / "Asset/MeshletCache/TerrainMultilayer16x16/terrain_multilayer_16x16.gltf",
           "MetallicTerrainMultilayer16x16VSMSample", ("normalTexture", *MODES)),
)}


def camera_route(template):
    flight = [expanded_flight_camera(template, frame) for frame in range(85)]
    aim = [56., 14.71, -64.64214324951172]
    near = {**template, "eye": [56., 40., -62.64], "center": aim, "fovDegrees": 48.}
    far = {**template, "eye": [58., 120., -58.], "center": aim, "fovDegrees": 49.}
    events = [{"frame": frame, "camera": camera} for frame, camera in enumerate(flight) if frame > 0]
    events += [{"frame": 85, "camera": near, "cameraCut": True},
               {"frame": 95, "camera": far, "cameraCut": True}]
    return flight[0], events


def prepare(output, sample):
    scene = sample.scene.as_posix()
    stream_asset = scene + ".meshstream.bin"
    require(sample.scene.is_file() and Path(stream_asset).is_file(), f"Bake {sample.bake_target} first")
    graph = read(GRAPH)
    first_camera, events = camera_route(graph["view"]["camera"])
    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    nodes["VBuffer"].update(path=scene, streamAssetPath=stream_asset,
                            maxResidentPages=LIMIT, visualization="lod", shadedDebugColors=False)
    nodes["Deferred"]["path"] = scene
    for mode in sample.modes:
        directory = output / mode
        nodes["Deferred"]["debugView"] = mode
        graph_filename = "TerrainLayers.metallic_graph.json"
        write_json(directory / graph_filename, graph)
        case = dict(id=mode, graph=graph_filename, scene=scene, width=SIZE[0], height=SIZE[1],
                    frames=max(VIEWS) + 1, warmupFrames=0, camera=first_camera, events=events,
                    captureFrames=list(VIEWS), streamingEvidenceFrames=list(range(85)) + [94, 104],
                    output="FinalBlit.color" if mode == "final" else "VBuffer.color",
                    comparisonOutput="Deferred.color", requireShadow=True, temporalJitter=False,
                    metadataOnly=True, allowResidentBudgetDeferral=True)
        write_json(directory / "Cases.json", dict(version=1, mode="diagnostic", cases=[case]))


def tile_bounds(scene):
    gltf = read(scene)
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


def terrain_texture(scene, name):
    path = scene.with_name(f"{scene.stem}_{name}.png")
    with Image.open(path) as pixels:
        return np.asarray(pixels.convert("RGBA"), dtype=np.float64) / 255.0


def layer_reference(world, weight, grass, rock):
    xz = world[..., [0, 2]]
    grass_value = repeat_linear(grass, xz / 2.)
    rock_value = repeat_linear(rock, xz / 6.)
    return grass_value * (1.0 - weight[..., None]) + rock_value * weight[..., None]


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


def edge_pairs(world, owners, values, levels, weight, plane, site=None):
    samples, jumps, local, mixed = [], [], [], []
    for far, a, b, beyond in ((np.s_[:, :-3], np.s_[:, 1:-2], np.s_[:, 2:-1], np.s_[:, 3:]),
                              (np.s_[:-3, :], np.s_[1:-2, :], np.s_[2:-1, :], np.s_[3:, :])):
        opposite = (owners[a] >= 0) & (owners[b] >= 0) & (owners[a] != owners[b])
        opposite &= (np.abs(world[a][..., 2] - plane) < 1.) & (np.abs(world[b][..., 2] - plane) < 1.)
        opposite &= np.linalg.norm(world[a] - world[b], axis=-1) < 1.5
        if site is not None:
            opposite &= site[a] & site[b]
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


def continuity(jumps, local, label, minimum):
    require(jumps.size >= minimum, f"{label}: too few locally supported cross-edge comparisons")
    excess = float(np.percentile(jumps, 95))
    ordinary = float(np.percentile(local, 95))
    require(excess <= max(0.06, 2.0 * ordinary), f"{label}: seam discontinuity {excess:.3g} vs {ordinary:.3g}")
    return excess, ordinary


def lod_pair_counts(mixed):
    mixes, counts = np.unique(mixed, axis=0, return_counts=True)
    return [{"lodsByTile": pair.tolist(), "pairs": int(count)} for pair, count in zip(mixes, counts)]


def edge_coverage(world, covered, owners, region):
    samples = holes = 0
    for first, middle, last in ((np.s_[:, :-2], np.s_[:, 1:-1], np.s_[:, 2:]),
                                (np.s_[:-2, :], np.s_[1:-1, :], np.s_[2:, :])):
        across = region[first] & region[last] & (owners[first] != owners[last])
        across &= np.linalg.norm(world[first] - world[last], axis=-1) < 1.5
        samples += np.count_nonzero(across)
        holes += np.count_nonzero(across & ~covered[middle])
    return int(samples), int(holes)


def inspect_four_layers(mode, frame, label, world, covered, owners, rgb, levels, band_weight, rgba, plane,
                        near_edge, footprint, material, result):
    four_way = near_edge & np.all(rgba > FOUR_LAYER_WEIGHT, axis=-1)
    layer_pixels = [int(np.count_nonzero(four_way & (owners == owner))) for owner in (0, 1)]
    require(min(layer_pixels) >= 30, f"{label}: too few four-layer pixels on each tile: {layer_pixels}")
    pairs, mixed, jumps, local = edge_pairs(world, owners, rgb, levels, band_weight, plane, site=four_way)
    require(frame != 84 or len(mixed) >= 10, f"{label}: expected four-layer pairs across different LODs")
    seam, ordinary = continuity(jumps, local, f"{label} four-layer overlap", 10)
    coverage_samples, holes = edge_coverage(world, covered, owners, four_way & (np.abs(world[..., 2] - plane) < 1.5))
    require(coverage_samples >= 10 and holes == 0, f"{label}: {holes} holes in {coverage_samples} cross-edge coverage samples")
    result["fourLayerOverlap"] = dict(pixelsByTile=layer_pixels, edgePairs=len(pairs), mixedLODPairs=len(mixed),
        mixedLODPairCounts=lod_pair_counts(mixed), seamExcessP95=seam, nearbyGradientP95=ordinary,
        coverageSamples=coverage_samples, uncoveredSamples=holes)

    if mode in REFERENCE_MODES:
        actual = rgb[near_edge]
        if mode == "normalTexture":
            length_error = float(np.max(np.abs(np.linalg.norm(actual * 2.0 - 1.0, axis=-1) - 1.0)))
            require(length_error <= 0.002, f"{label}: blended tangent normals must have unit length")
            result["tangentNormalMaxUnitLengthError"] = length_error
        if mode == "roughness":
            require(np.ptp(actual, axis=-1).max() < 1e-6, f"{label}: roughness debug output is not scalar")
            actual = actual[:, :1]
        weights = rgba[near_edge]
        sites = world[near_edge]
        expected = material.evaluate(mode, sites, weights, footprint=footprint)
        difference = actual - expected
        error = float(np.mean(np.abs(difference)))
        require(error <= REFERENCE_TOLERANCE, f"{label}: {error:.4f} mean error from filtered world-space reference")
        result["worldSpaceReferenceMeanAbsoluteError"] = error
        # Test the excess seam on the residual too, so high-frequency detail
        # cannot hide a tile-local shift in an otherwise continuous material.
        residual = np.zeros((*owners.shape, expected.shape[-1]), dtype=difference.dtype)
        residual[near_edge] = difference
        _, _, seam_residual, _ = edge_pairs(world, owners, residual, levels, band_weight, plane, site=four_way)
        residual_p95 = float(np.percentile(seam_residual, 95))
        require(residual_p95 <= REFERENCE_SEAM_TOLERANCE,
                f"{label}: filtered reference seam residual {residual_p95:.4f} exceeds {REFERENCE_SEAM_TOLERANCE}")
        result["fourLayerOverlap"]["referenceSeamResidualP95"] = residual_p95
        if VIEWS[frame] == "near":
            result["nearOmissions"] = {}
            for index, channel in enumerate("RGBA"):
                conclusive = weights[:, index] > 0.1
                require(np.count_nonzero(conclusive) >= 20, f"{label}: channel {channel} lacks near witnesses")
                missing = material.evaluate(mode, sites[conclusive], weights[conclusive],
                                            footprint=footprint[conclusive], omitted_channel=channel)
                actual_error = float(np.mean(np.abs(actual[conclusive] - expected[conclusive])))
                omission_error = float(np.mean(np.abs(actual[conclusive] - missing)))
                result["nearOmissions"][channel] = dict(referenceError=actual_error, omissionError=omission_error)


def inspect(directory, case, sample, bounds, weight_field, layer_textures, material=None):
    mode = case["id"]
    spec = case["spec"]
    cameras, _ = camera_timeline(spec)
    nodes = {node["name"]: node["properties"] for node in case["resolvedGraph"]["nodes"]}
    first, events = camera_route(read(GRAPH)["view"]["camera"])
    require(case["status"] == "passed" and Path(spec["scene"]).resolve() == sample.scene.resolve()
            and not spec["temporalJitter"] and nodes["Deferred"]["debugView"] == mode
            and nodes["VBuffer"]["maxResidentPages"] == LIMIT and nodes["VBuffer"]["autoLod"],
            f"{mode}: unexpected acceptance case or graph settings")
    require(spec["camera"] == first and spec["events"] == events and spec["frames"] == max(VIEWS) + 1
            and (spec["width"], spec["height"]) == SIZE and spec["warmupFrames"] == 0,
            f"{mode}: unexpected terrain acceptance route")
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
    plane = bounds[0][1, 2]
    for shot in case["captures"]:
        frame = shot["frame"]
        label = f"{mode} {VIEWS[frame]}"
        world, covered = capture_geometry(directory, shot, cameras[frame], SIZE)
        owners = owners_on_edge(world, covered, bounds)
        near_edge = (owners >= 0) & (np.abs(world[..., 2] - plane) < 2.)
        if material is None:
            weight = band_reference(world, weight_field)
        else:
            footprint = ray_cone_uv_footprint(world, near_edge, cameras[frame], material.world_size)
            rgba = np.zeros((*owners.shape, 4), dtype=np.float64)
            rgba[near_edge] = material.weights(world[near_edge], footprint)
            # The gravel layer follows the same rock-band mask as rock.
            weight = rgba[..., 0] + rgba[..., 2]
        rgb = raw_pixels(directory, shot, "Deferred.color", SIZE, "<f4", "RGBA32Sfloat", 1, 4)[0, ..., :3]
        require(np.isfinite(rgb).all(), f"{label}: nonfinite surface output")
        if mode == "final":
            require(np.percentile(rgb[owners >= 0], 95) > 0, f"{label}: unlit or empty terrain")
            frames[VIEWS[frame]] = {"image": shot["file"], "terrainPixels": int(np.count_nonzero(owners >= 0))}
            continue

        levels = decode_lods(lod_pixels(directory, shot, SIZE))
        require(np.all(levels[owners >= 0] >= 0), f"{label}: mismatched LOD palette")
        weights, mixed, jumps, local = edge_pairs(world, owners, rgb, levels, weight, plane)
        require(np.count_nonzero(weights > 0.9) >= 10 and np.count_nonzero(weights < 0.1) >= 10
                and np.count_nonzero((weights > 0.1) & (weights < 0.9)) >= 5,
                f"{label}: missing rock, grass, or blend across the authored tile edge")
        stream, = [item for item in case["frames"][frame]["streaming"] if item["passName"] == "VBuffer"]
        require(frame != 84 or (len(mixed) >= 10 and stream["activePages"] == LIMIT),
                f"{label}: the band needs actual mixed-LOD adjacent pixels at the saturated 272-page limit")
        excess, ordinary = continuity(jumps, local, label, 25)
        result = dict(rockPairs=int(np.count_nonzero(weights > 0.9)), grassPairs=int(np.count_nonzero(weights < 0.1)),
                      blendPairs=int(np.count_nonzero((weights > 0.1) & (weights < 0.9))), mixedBandPairs=len(mixed),
                      mixedBandLodPairs=lod_pair_counts(mixed), activePages=stream["activePages"],
                      seamExcessP95=excess, nearbyGradientP95=ordinary, lodImage=shot["file"])
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
        if material is not None:
            inspect_four_layers(mode, frame, label, world, covered, owners, rgb, levels, weight, rgba, plane,
                                near_edge, footprint, material, result)
        elif mode == "baseColor":
            expect = layer_reference(world[near_edge], weight[near_edge], *layer_textures["baseColor"])
            mean_absolute_error = float(np.mean(np.abs(rgb[near_edge] - expect)))
            require(mean_absolute_error <= 0.045, f"{label}: base color differs from linear-sRGB layer sampling")
            result["worldScaleBaseColorMeanAbsoluteError"] = mean_absolute_error
            if VIEWS[frame] == "near":
                # Correlation keeps the repeating world-space phase check meaningful
                # when filtering attenuates contrast relative to the mip-zero reference.
                for layer_name, pure in (("grass", weight[near_edge] < 0.02), ("rock", weight[near_edge] > 0.98)):
                    actual = rgb[near_edge][pure, 0]
                    expected = expect[pure, 0]
                    correlation = float(np.corrcoef(actual, expected)[0, 1])
                    require(correlation >= 0.25, f"{label}: {layer_name} texture lost its world-space phase")
                    result[f"{layer_name}WorldTextureCorrelation"] = correlation
        if material is None and mode == "roughness":
            value = rgb[..., 0]
            expect = layer_reference(world[near_edge], weight[near_edge], *layer_textures["roughness"])[..., 0]
            mean_absolute_error = float(np.mean(np.abs(value[near_edge] - expect)))
            require(mean_absolute_error <= 0.043, f"{label}: roughness differs from world-scale layer sampling")
            require(np.ptp(rgb[near_edge], axis=-1).max() < 1e-6, f"{label}: roughness debug output is not scalar")
            result["worldScaleRoughnessMeanAbsoluteError"] = mean_absolute_error
        if mode in ("mappedNormal", "shadingNormal"):
            reference[frame] = (world, owners, weight, rgb, levels, rgba if material is not None else None)
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


def verify(output, sample):
    bounds = tile_bounds(sample.scene)
    material = load_terrain_layer_reference(sample.scene) if sample.name == "multilayer" else None
    weight_field = textures = None
    if material is None:
        weight_field, _, _ = load_height_field(WEIGHT, "R")
        textures = {
            "baseColor": tuple(srgb_to_linear(terrain_texture(sample.scene, f"{layer}_color")[..., :3])
                               for layer in ("grass", "rock")),
            "roughness": tuple(terrain_texture(sample.scene, f"{layer}_normal")[..., 3:4] for layer in ("grass", "rock")),
        }
    report = {"passed": True, "variant": sample.name, "tiles": TILES, "planeZ": float(bounds[0][1, 2]), "modes": {}}
    normals = {}
    for mode in sample.modes:
        directory = output / mode / "capture"
        case = load_case(directory, mode)
        result, normals[mode] = inspect(directory, case, sample, bounds, weight_field, textures, material)
        for view in result.values():
            for key in ("aovImage", "lodImage", "image"):
                if key in view:
                    view[key] = (directory.relative_to(output) / view[key]).as_posix()
        report["modes"][mode] = result
    for frame, label in VIEWS.items():
        world, owners, weight, mapped, levels, rgba = normals["mappedNormal"][frame]
        other_world, other_owners, _, authored, other_levels, _ = normals["shadingNormal"][frame]
        match = (owners >= 0) & (owners == other_owners) & (levels == other_levels)
        match &= np.linalg.norm(world - other_world, axis=-1) < 0.002
        masks = (("grass", weight < 0.02), ("rock", weight > 0.98)) if rgba is None else (
            ("fourLayer", np.all(rgba > FOUR_LAYER_WEIGHT, axis=-1)),)
        for name, mask in masks:
            require(np.count_nonzero(match & mask) >= 20, f"{label}: missing matching {name} geometry for normals")
            p90 = float(np.percentile(np.linalg.norm((mapped - authored)[match & mask], axis=-1), 90))
            require(p90 >= 0.003, f"{label}: {name} did not affect the OpenPBR mapped normal")
            report["modes"]["mappedNormal"][label][f"{name}NormalDifferenceP90"] = p90
    if material is not None:
        # The near comparison must distinguish each layer from a reference that omits it.
        for channel in "RGBA":
            controls = [report["modes"][mode]["near"]["nearOmissions"][channel] for mode in REFERENCE_MODES]
            require(any(value["omissionError"] - value["referenceError"] >= 0.002 for value in controls),
                    f"near: no material output distinguishes the contribution from channel {channel}")
    report_path = output / "TerrainLayerAcceptance.json"
    write_json(report_path, report)
    print(f"Terrain layer acceptance passed. {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--variant", choices=SAMPLES, default="legacy")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="Specific run directory")
    outputs.add_argument("--output-root", type=Path, help="Parent for a fresh directory on each run")
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    try:
        sample = SAMPLES[args.variant]
        require(args.output_root is None or args.action == "run", "--output-root is only available for run")
        default_output = OUTPUT if sample.name == "legacy" else ROOT / "build/terrain-multilayer-regression"
        output = args.output or default_output
        if args.action == "run" and args.output is None:
            output_root = args.output_root or default_output
            output_root.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
        output = output.resolve()
        print(f"Terrain layer acceptance artifacts: {output}", flush=True)
        if args.action != "verify":
            prepare(output, sample)
        if args.action == "run":
            require(args.executable is not None and args.executable.is_file(), "Build and provide MetallicRHITests")
            for mode in sample.modes:
                capture(args.executable.resolve(), output / mode)
        if args.action != "prepare":
            verify(output, sample)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Terrain layer acceptance failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
