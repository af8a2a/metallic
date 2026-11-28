#!/usr/bin/env python3
"""Verify the cooked Ancient Rift Valley terrain and its eight-layer OpenPBR readbacks."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, read_json, write_json
from RunTerrainStreamingRegression import capture
from TerrainLayerAcceptance import REFERENCE_TOLERANCE, save_aov
from TerrainLayerReference import load_terrain_layer_reference, ray_cone_uv_footprint
from VerifyTerrainStreamingRegression import (
    REPORT_NAME, capture_geometry, decode_lods, lod_pixels, raw_pixels, require,
)


SCENE = ROOT / "Asset/MeshletCache/AncientRiftValley2x2/ancient_rift_valley_2x2.gltf"
TARGET = "MetallicAncientRiftValley2x2Sample"
RESOLUTION = (640, 360)
MODES = ("baseColor", "normalTexture", "roughness")
CORNER_TARGET = [0.0, 71.62, 0.0]
APRON = [-24.0, 41.14, -195.0]
HOLD_FRAMES = 16


def camera(eye, target, fov):
    return dict(projection="perspective", eye=eye, center=target, up=[0.0, 0.0, -1.0],
                fovDegrees=fov, znear=0.5, zfar=1400.0)


def prepare(directory, component):
    stream = Path(f"{SCENE}.meshstream.bin")
    require(SCENE.is_file() and stream.is_file(), f"Bake {TARGET} first")
    sample = read_json(SCENE)["extras"]["metallicTerrainRegions"]
    regions = sample["regions"]
    require([region["id"] for region in regions] == ["x1_z1", "x2_z1", "x1_z2", "x2_z2"],
            "Expected the four central regions in directory order")
    graph = read_json(GRAPH)
    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    nodes["VBuffer"].update(path=SCENE.as_posix(), streamAssetPath=stream.as_posix(), streamRegions=regions,
                            initialStreamRegions=list(range(len(regions))), geometryRegionLoadDistance=2500.0,
                            geometryRegionUnloadDistance=2500.0, maxResidentPages=4096, maxActiveGroups=32768,
                            maxResidentBytes=256 * 1024 * 1024, maxLockedFallbackPages=4096,
                            visualization="lod", shadedDebugColors=False, initialLoad=True, compactShadingAttributes=False)
    nodes["Shadows"].update(shadowLOD=False, physicalPageBudget=1024, shadowBias=0.008)
    nodes["Deferred"]["path"] = SCENE.as_posix()

    modes = []
    if component != "materials":
        nodes["VBuffer"].update(autoLod=True, lodPixelError=1.5)
        views = [camera(eye, CORNER_TARGET, fov) for eye, fov in (
            ([0.0, 198.0, 0.0], 44.0), ([0.0, 114.0, 0.0], 46.0),
            ([-22.0, 141.0, -20.0], 47.0), ([23.0, 143.0, 30.0], 47.0),
            ([25.0, 126.0, -26.0], 46.0), ([-9.0, 118.0, 23.0], 46.0),
        )]
        prepare_case(directory / "coverage", graph, "coverage", "VBuffer.color", views)
        modes.append("coverage")
    if component != "coverage":
        nodes["VBuffer"].update(autoLod=True, lodPixelError=0.15)
        views = [camera([-24.0, 63.0, -195.0], APRON, 43.0),
                 camera([-24.0, 102.0, -195.0], APRON, 43.0)]
        for mode in MODES:
            nodes["Deferred"]["debugView"] = mode
            prepare_case(directory / mode, graph, mode, "VBuffer.color", views)
        nodes["Deferred"]["debugView"] = "final"
        prepare_case(directory / "final", graph, "final", "FinalBlit.color", [sample["sampleCamera"]])
        modes.extend((*MODES, "final"))
    return modes


def prepare_case(path, graph, name, output, views):
    graph_path = path / "Valley.metallic_graph.json"
    frames = [(index + 1) * HOLD_FRAMES - 1 for index in range(len(views))]
    write_json(graph_path, graph)
    specification = dict(id=name, graph=graph_path.name, scene=SCENE.as_posix(), width=RESOLUTION[0],
                         height=RESOLUTION[1], frames=frames[-1] + 1, warmupFrames=0, camera=views[0],
                         events=[dict(frame=index * HOLD_FRAMES, camera=view, cameraCut=True)
                                 for index, view in enumerate(views[1:], 1)],
                         captureFrames=frames, streamingEvidenceFrames=frames, output=output,
                         comparisonOutput="Deferred.color", requireShadow=True, temporalJitter=False,
                         metadataOnly=True, allowResidentBudgetDeferral=True, requireResidentBudgetDeferral=False)
    write_json(path / "Cases.json", dict(version=1, mode="diagnostic", cases=[specification]))


def validated_captures(path):
    directory = path / "capture"
    report = read_json(directory / REPORT_NAME)
    validation = report["validationMessages"]
    require(report["protocol"] == "virtual-shadow-map-acceptance-v2" and report["status"] == "passed"
            and report["mode"] == "diagnostic" and report["validation"]
            and not validation["captureFailed"] and validation["count"] == 0,
            f"{path.name}: expected a passing Vulkan diagnostic")
    case, = report["cases"]
    spec, = read_json(path / "Cases.json")["cases"]
    require(case["spec"] == spec and case["status"] == "passed", f"{path.name}: unexpected capture case")
    require([shot["frame"] for shot in case["captures"]] == spec["captureFrames"], "Incomplete capture sequence")
    require([row["frame"] for row in case["frames"]] == list(range(spec["frames"])), "Incomplete frame sequence")
    for row in case["frames"]:
        stream, = (entry for entry in row["streaming"] if entry["passName"] == "VBuffer")
        ready = row["readiness"]
        published = stream["regionSelection"]["publishedRegions"]
        require(Path(stream["assetPath"]).resolve() == Path(f"{SCENE}.meshstream.bin").resolve()
                and (row["actualRenderWidth"], row["actualRenderHeight"]) == RESOLUTION
                and published == [0, 1, 2, 3] and ready["ready"]
                and ready["requiredPages"] == ready["completedPages"] > 0,
                f"{path.name} frame {row['frame']}: incomplete four-region terminal cut")
    views = [spec["camera"], *(event["camera"] for event in spec["events"])]
    return case, zip(case["captures"], views, strict=True)


def seam_pairs(world, covered, levels):
    x, z = world[..., 0], world[..., 2]
    sites = covered & (levels >= 0) & (np.abs(x) < 28.0) & (np.abs(z) < 28.0)
    quadrant = (x > 0).astype(np.uint8) + 2 * (z > 0).astype(np.uint8)
    counts = np.bincount(quadrant[sites], minlength=4).tolist()
    edges = {(0, 1): [0, 0, 0], (2, 3): [0, 0, 0], (0, 2): [0, 0, 0], (1, 3): [0, 0, 0]}
    for a, b, c, d, e in ((np.s_[:, :-1], np.s_[:, 1:], np.s_[:, :-2], np.s_[:, 1:-1], np.s_[:, 2:]),
                           (np.s_[:-1, :], np.s_[1:, :], np.s_[:-2, :], np.s_[1:-1, :], np.s_[2:, :])):
        near = np.linalg.norm(world[a] - world[b], axis=-1) <= 3.0
        across_gap = np.linalg.norm(world[c] - world[e], axis=-1) <= 4.0
        for edge, counts_on_edge in edges.items():
            i, j = edge
            matching = ((quadrant[a] == i) & (quadrant[b] == j)) | ((quadrant[a] == j) & (quadrant[b] == i))
            pairs = matching & sites[a] & sites[b] & near
            counts_on_edge[0] += int(pairs.sum())
            counts_on_edge[1] += int(np.count_nonzero(pairs & (levels[a] != levels[b])))
            gap = ((quadrant[c] == i) & (quadrant[e] == j)) | ((quadrant[c] == j) & (quadrant[e] == i))
            counts_on_edge[2] += int(np.count_nonzero(gap & sites[c] & sites[e] & across_gap & ~covered[d]))
    return counts, {f"{i}-{j}": dict(pairs=c[0], mixed=c[1], holes=c[2]) for (i, j), c in edges.items()}


def verify_coverage(path):
    case, captures = validated_captures(path)
    folder = path / "capture"
    frames = []
    for shot, view in captures:
        world, covered = capture_geometry(folder, shot, view, RESOLUTION)
        levels = decode_lods(lod_pixels(folder, shot, RESOLUTION))
        counts, edges = seam_pairs(world, covered, levels)
        require(not any(edge["holes"] for edge in edges.values()), f"Frame {shot['frame']}: hole along a junction edge")
        frames.append(dict(frame=shot["frame"], quadrantPixels=counts, edges=edges,
                           image=(folder / shot["file"]).as_posix()))
    witnesses = [row for row in frames if min(row["quadrantPixels"]) >= 20
                 and all(edge["pairs"] >= 10 and edge["mixed"] > 0 for edge in row["edges"].values())]
    require(witnesses, f"The same frame must show mixed LOD on all four junction edges: {frames}")
    first, = (s for s in case["frames"][0]["streaming"] if s["passName"] == "VBuffer")
    return dict(mixedJunctionFrames=[row["frame"] for row in witnesses], frames=frames,
                publishedRegions=first["regionSelection"]["publishedRegions"],
                lockedTerminalPages=case["frames"][0]["readiness"]["requiredPages"])


def verify_material(path, material, mode):
    _, captures = validated_captures(path)
    folder = path / "capture"
    frames = []
    for shot, view in captures:
        world, covered = capture_geometry(folder, shot, view, RESOLUTION)
        sites = covered & (np.abs(world[..., 0] - APRON[0]) < 11.0) & (np.abs(world[..., 2] - APRON[2]) < 9.0)
        footprint = ray_cone_uv_footprint(world, sites, view, material.world_size)
        weights = material.weights(world[sites], footprint)
        overlap = np.all(weights > 0.02, axis=-1)
        label = f"{mode} frame {shot['frame']}"
        require(int(overlap.sum()) >= 50, f"{label}: the camera missed the eight-layer apron")
        rgb = raw_pixels(folder, shot, "Deferred.color", RESOLUTION, "<f4", "RGBA32Sfloat", 1, 4)[0, ..., :3]
        actual = rgb[sites][overlap]
        if mode == "roughness":
            require(np.ptp(actual, axis=-1).max() < 1e-6, f"{label}: roughness is not scalar")
            actual = actual[:, :1]
        if mode == "normalTexture":
            require(np.max(np.abs(np.linalg.norm(actual * 2 - 1, axis=-1) - 1)) < 0.002,
                    f"{label}: mixed tangent normals are not normalized")
        positions = world[sites][overlap]
        weight = weights[overlap]
        pixel_footprint = footprint[overlap]
        expected = material.evaluate(mode, positions, weight, footprint=pixel_footprint)
        error = float(np.abs(actual - expected).mean())
        require(np.isfinite(error) and error <= REFERENCE_TOLERANCE, f"{label}: reference error {error:.6f}")
        no_second = weight.copy()
        no_second[:, 4:] = 0.0
        no_second /= no_second.sum(axis=-1, keepdims=True)
        incorrect = material.evaluate(mode, positions, no_second, footprint=pixel_footprint)
        without_second_error = float(np.abs(actual - incorrect).mean())
        frames.append(dict(frame=shot["frame"], eightLayerPixels=int(overlap.sum()), referenceError=error,
                           errorIgnoringSecondWeightMap=without_second_error,
                           weightMipMedian=float(np.median(material.weight_texture.lod(pixel_footprint))),
                           depthSha256=shot["rawImages"]["VBuffer.depth"]["rawSha256"],
                           aov=(folder / save_aov(folder, f"{mode}-{shot['frame']}", rgb, mode)).as_posix()))
    return frames


def verify(directory, component):
    result = {}
    if component != "materials":
        result["coverage"] = verify_coverage(directory / "coverage")
    if component != "coverage":
        reference = load_terrain_layer_reference(SCENE)
        require(reference.world_size == 1000 and len(reference.layers) == 8 and reference.second_weight_texture is not None,
                "Expected eight layers, both weight maps and global metre-space UVs")
        result["material"] = {mode: verify_material(directory / mode, reference, mode) for mode in MODES}
        hashes = [[frame["depthSha256"] for frame in frames] for frames in result["material"].values()]
        require(all(item == hashes[0] for item in hashes), "Material modes do not use the same captured geometry")
        require(all(frame["errorIgnoringSecondWeightMap"] > max(frame["referenceError"] * 4, 1 / 255)
                    for frame in result["material"]["baseColor"]), "The GPU base color did not distinguish the second pack")
        _, captures = validated_captures(directory / "final")
        final, _ = next(captures)
        result["final"] = (directory / "final" / "capture" / final["file"]).as_posix()
    write_json(directory / "AncientRiftValleyAcceptance.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--component", choices=("all", "coverage", "materials"), default="all")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        directory = args.output.resolve()
        if args.action == "run":
            require(args.executable and args.executable.is_file(), "Build and provide MetallicRHITests")
            directory.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
        print(f"Ancient Rift Valley evidence: {directory}", flush=True)
        if args.action != "verify":
            modes = prepare(directory, args.component)
        if args.action == "run":
            for mode in modes:
                capture(args.executable.resolve(), directory / mode, timeout=480)
        if args.action != "prepare":
            verify(directory, args.component)
    except (KeyError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Ancient Rift Valley validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
