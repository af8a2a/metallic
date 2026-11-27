#!/usr/bin/env python3
"""Verify all 8x8 terrain seams at the 80-page streaming limit."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from VerifyTerrainStreamingRegression import (
    CLIP_CAMERA, FIXED_VIEWS, REPORT_NAME, ROOT, camera_timeline, capture_geometry,
    decode_lods, lod_pixels, raw_pixels, read, require, save,
)


SCENE = ROOT / "Asset/MeshletCache/TerrainBlocks8x8/terrain_blocks_8x8.gltf"
TILES_PER_AXIS = 8
PAGE_CAP = 80
LOCKED_PAGES = 65
CROSSING_FRAME = 19
CROSSING_TILES = ((1, 5), (2, 5))
CROSSING_EYES = {
    18: [-67.0, 72.92478373654546, 47.50479582926186],
    19: [-64.5, 72.96649247867539, 44.62045073559008],
    20: [-62.0, 72.99161139154354, 41.749789885817165],
}
MOTION_SIZE = (640, 360)
FIXED_SIZE = (960, 540)
TILE_INSET = 0.01


def tile_id(tile):
    x, z = tile
    return z * TILES_PER_AXIS + x


def fixture():
    scene = read(SCENE)
    terrain = scene["extras"]["metallicTerrainTiles"]
    require(terrain["tilesPerAxis"] == TILES_PER_AXIS, "Expected the 8x8 terrain fixture")
    meshes = {mesh["name"]: mesh for mesh in scene["meshes"]}

    def mesh_bounds(mesh):
        primitives = mesh["primitives"]
        accessors = [scene["accessors"][p["attributes"]["POSITION"]] for p in primitives]
        return np.asarray([np.min([a["min"] for a in accessors], axis=0),
                           np.max([a["max"] for a in accessors], axis=0)])

    by_tile = {tuple(tile["tile"]): mesh_bounds(meshes[tile["name"]]) for tile in terrain["tiles"]}
    tiles = [(x, z) for z in range(TILES_PER_AXIS) for x in range(TILES_PER_AXIS)]
    require(set(by_tile) == set(tiles) and len(terrain["tiles"]) == len(tiles), "Missing or repeated terrain tiles")
    bounds = np.asarray([by_tile[tile] for tile in tiles])
    cuts = [
        np.array([by_tile[tile][0, axis] for tile in boundary] + [bounds[:, 1, axis].max()])
        for axis, boundary in ((0, [(x, 0) for x in range(TILES_PER_AXIS)]),
                               (2, [(0, z) for z in range(TILES_PER_AXIS)]))
    ]
    for x, z in tiles:
        require(np.array_equal(by_tile[x, z][:, 0], cuts[0][x:x+2])
                and np.array_equal(by_tile[x, z][:, 2], cuts[1][z:z+2]),
                f"Tile {(x, z)} does not match the authored grid")

    terrain_names = {tile["name"] for tile in terrain["tiles"]}
    props = []
    for node in scene["nodes"]:
        if "mesh" not in node:
            continue
        mesh = scene["meshes"][node["mesh"]]
        if mesh["name"] in terrain_names:
            require(not any(key in node for key in ("rotation", "matrix", "scale", "translation")),
                    "Expected terrain vertices in world coordinates")
            continue
        require("rotation" not in node and "matrix" not in node, "Expected axis-aligned props")
        transformed = mesh_bounds(mesh) * node.get("scale", [1., 1., 1.]) + node.get("translation", [0., 0., 0.])
        props.append((transformed.min(axis=0), transformed.max(axis=0)))

    edges = []
    lookup = np.full((len(tiles) + 1, len(tiles) + 1), -1, dtype=np.int16)
    for x, z in tiles:
        first = tile_id((x, z))
        for axis, coord, neighbor in ((0, x, (x + 1, z)), (2, z, (x, z + 1))):
            if coord + 1 == TILES_PER_AXIS:
                continue
            second = tile_id(neighbor)
            lookup[first, second] = lookup[second, first] = len(edges)
            edges.append((first, second, axis, float(cuts[axis // 2][coord + 1])))
    return tiles, bounds, cuts, props, edges, lookup


def capture_run(directory, case_id, size):
    document = read(directory / REPORT_NAME)
    require(document["protocol"] == "virtual-shadow-map-acceptance-v2" and document["mode"] == "diagnostic"
            and document["status"] == "passed", f"{case_id}: missing passing diagnostic receipt")
    validation = document["validationMessages"]
    require(document["validation"] and not validation["captureFailed"] and validation["count"] == 0,
            f"{case_id}: Vulkan validation capture missing or contains messages")
    case, = document["cases"]
    spec = case["spec"]
    require(case["id"] == spec["id"] == case_id and case["status"] == "passed",
            f"{case_id}: wrong or incomplete case")
    require(Path(spec["scene"]).resolve() == SCENE.resolve()
            and (spec["width"], spec["height"]) == size and spec["output"] == "VBuffer.color"
            and spec["requireShadow"] and not spec["temporalJitter"],
            f"{case_id}: unexpected scene, extent or capture settings")
    graph, = [node["properties"] for node in case["resolvedGraph"]["nodes"] if node["name"] == "VBuffer"]
    require(graph["maxResidentPages"] == PAGE_CAP and graph["autoLod"]
            and graph["visualization"] == "lod" and not graph["shadedDebugColors"],
            f"{case_id}: expected the 80-page limit and unshaded automatic LOD visualization")

    rows = case["frames"]
    require(rows and [row["frame"] for row in rows] == list(range(spec["frames"])),
            f"{case_id}: incomplete frame sequence")
    asset = Path(str(SCENE) + ".meshstream.bin").resolve()
    streams, identities = [], set()
    for row in rows:
        at = f"{case_id} frame {row['frame']}"
        ready = row["readiness"]
        stream, = [entry for entry in row["streaming"] if entry["passName"] == "VBuffer"]
        require(Path(stream["assetPath"]).resolve() == asset and stream["totalPages"] > PAGE_CAP,
                f"{at}: expected the cooked 8x8 stream asset")
        require(ready["ready"] and ready["completedPages"] == ready["requiredPages"] == LOCKED_PAGES,
                f"{at}: terminal fallback is incomplete")
        require(stream["maxResidentPages"] == PAGE_CAP
                and LOCKED_PAGES <= stream["residentPages"] <= stream["activePages"] <= PAGE_CAP,
                f"{at}: expected the 80-page limit")
        require(stream["loadFailures"] == stream["requestOverflows"] == 0
                and stream["allocationFailures"] == stream["allocationBudgetDeferrals"],
                f"{at}: unexpected stream failure")
        require((row["actualRenderWidth"], row["actualRenderHeight"]) == size
                and row["shadow"]["header"]["admittedCount"] > 0,
                f"{at}: requested render extent or VSM missing")
        identities.add((stream["totalPages"], stream["originalTopologyHash"], stream["primitiveMetadataHash"]))
        streams.append(stream)
    require(len(identities) == 1, f"{case_id}: cooked stream identity changed during playback")

    captures = case["captures"]
    frames = spec["captureFrames"]
    require(frames and frames == sorted(set(frames)) and all(0 <= frame < len(rows) for frame in frames)
            and [capture["frame"] for capture in captures] == frames,
            f"{case_id}: missing, unexpected, or duplicate capture")
    return spec, {capture["frame"]: capture for capture in captures}, streams, identities.pop()


def terrain_owners(world, covered, cuts, props):
    tiles = []
    inside = covered.copy()
    for axis, borders in zip((0, 2), cuts):
        coordinates = world[..., axis]
        raw_tile = np.searchsorted(borders, coordinates, side="right") - 1
        tile = np.clip(raw_tile, 0, TILES_PER_AXIS - 1)
        inside &= (coordinates > borders[tile] + TILE_INSET) & (coordinates < borders[tile + 1] - TILE_INSET)
        tiles.append(tile)
    for low, high in props:
        inside &= ~np.all((low - TILE_INSET <= world) & (world <= high + TILE_INSET), axis=-1)
    return np.where(inside, tiles[1] * TILES_PER_AXIS + tiles[0], -1)


def opposing_lods(world, owners, colors, edges, lookup):
    first_owner = np.asarray([a for a, _, _, _ in edges])
    axes = np.asarray([axis for _, _, axis, _ in edges])
    planes = np.asarray([plane for _, _, _, plane in edges])
    histogram = Counter()
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        neighborhood = lookup[owners[first], owners[second]]
        y, x = np.nonzero(neighborhood >= 0)
        edge = neighborhood[y, x]
        axis, plane = axes[edge], planes[edge]
        a, b = world[first][y, x], world[second][y, x]
        # Reject projected overlaps between points that are far apart in world space.
        index = np.arange(len(edge))
        close = (np.abs(a[index, axis] - plane) <= 3.) & (np.abs(b[index, axis] - plane) <= 3.)
        close &= np.linalg.norm(a - b, axis=1) <= 4.
        y, x, edge = y[close], x[close], edge[close]
        if not len(edge):
            continue
        pairs = np.stack((colors[first][y, x], colors[second][y, x]), axis=1)
        lods = decode_lods(pairs)
        require(np.all(lods >= 0), "Opposing terrain pixels do not carry valid LOD colors")
        ordered = owners[first][y, x] == first_owner[edge]
        labels = np.column_stack((edge, np.where(ordered, lods[:, 0], lods[:, 1]),
                                  np.where(ordered, lods[:, 1], lods[:, 0])))
        unique, counts = np.unique(labels, axis=0, return_counts=True)
        histogram.update({tuple(map(int, label)): int(count) for label, count in zip(unique, counts)})
    result = {}
    for (edge, a, b), count in sorted(histogram.items()):
        result.setdefault(edge, []).append({"lodsByTile": [a, b], "pairs": count})
    return result


def pair_counts(pairs):
    return sum(pair["pairs"] for pair in pairs), sum(pair["pairs"] for pair in pairs
                                                     if pair["lodsByTile"][0] != pair["lodsByTile"][1])


def verify_crossing(cameras, cuts, captures, bounds, report):
    require(not cuts and set(CROSSING_EYES).issubset(captures), "Missing continuous frames 18, 19 and 20")
    west, east = (bounds[tile_id(tile)] for tile in CROSSING_TILES)
    for frame, expected in CROSSING_EYES.items():
        camera = cameras[frame]
        require(np.allclose(camera["eye"], expected, rtol=0, atol=1e-7)
                and all(camera[key] == value for key, value in CLIP_CAMERA.items()),
                f"Frame {frame}: original crossing eye or projection changed")
        extent = west if frame < CROSSING_FRAME else east
        eye = np.asarray(camera["eye"])[[0, 2]]
        require(np.all((extent[0, [0, 2]] < eye) & (eye < extent[1, [0, 2]])),
                f"Frame {frame}: eye does not straddle the original tile edge")
    before, after = (np.asarray(cameras[frame]["eye"]) for frame in (CROSSING_FRAME - 1, CROSSING_FRAME))
    plane = east[0, 0]
    hit = before + (after - before) * ((plane - before[0]) / (after[0] - before[0]))
    report["crossing"] = {"tiles": CROSSING_TILES, "frame": CROSSING_FRAME, "planeX": float(plane),
                          "position": hit.tolist(), "frames": []}


def fixed_cameras(cameras, cuts, captures, report):
    expected_frames = [frame for _, first, last, *_ in FIXED_VIEWS for frame in (first, last)]
    require(len(cameras) == 60 and cuts == [20, 40] and list(captures) == expected_frames,
            "Expected the original three fixed camera segments and six captures")
    labels = {}
    report["fixedViews"] = []
    for name, first, last, eye, center, fov in FIXED_VIEWS:
        pinned = {**CLIP_CAMERA, "eye": eye, "center": center, "fovDegrees": fov}
        require(all(camera == pinned for camera in cameras[first:last + 1]), f"Fixed {name}: original camera changed")
        view = {"name": name, "frames": []}
        report["fixedViews"].append(view)
        labels[first] = labels[last] = view
    return labels


def verify_fixed_capture(directory, capture, owners, view):
    frame = capture["frame"]
    terrain = owners >= 0
    count = int(np.count_nonzero(terrain))
    require(10 * count >= FIXED_SIZE[0] * FIXED_SIZE[1],
            f"Fixed {view['name']} frame {frame}: terrain covers less than 10% of the original viewport")
    hdr = raw_pixels(directory, capture, "Deferred.color", FIXED_SIZE, "<f4", "RGBA32Sfloat", 1, 4)
    require(np.isfinite(hdr).all(), f"Fixed {view['name']} frame {frame}: nonfinite Deferred color")
    shadow = raw_pixels(directory, capture, "Shadows.shadows", FIXED_SIZE, "u1", "R8Unorm", 8)[0, :, :, 0]
    lit = int(np.count_nonzero(shadow[terrain] >= 128))
    require(0 < lit < count, f"Fixed {view['name']} frame {frame}: missing lit or shadowed VSM samples")
    view["frames"].append({"frame": frame, "terrainPixels": count, "litPixels": lit, "shadowedPixels": count - lit})


def scan(directory, case_id, size, data, report):
    _, bounds, borders, props, edges, lookup = data
    spec, captures, streams, identity = capture_run(directory, case_id, size)
    cameras, cuts = camera_timeline(spec)
    directed = case_id == "motion-directed-8x8-tight"
    fixed = case_id == "fixed-tight"
    if directed:
        verify_crossing(cameras, cuts, captures, bounds, report)
    views = fixed_cameras(cameras, cuts, captures, report) if fixed else {}
    crossing_edge = lookup[tuple(map(tile_id, CROSSING_TILES))]
    witnesses = {}
    for frame, capture in captures.items():
        camera = cameras[frame]
        require(camera["projection"] == "perspective", f"{case_id} frame {frame}: expected perspective view")
        stream = streams[frame]
        active = stream["activePages"]
        world, covered = capture_geometry(directory, capture, camera, size)
        owners = terrain_owners(world, covered, borders, props)
        colors = lod_pixels(directory, capture, size)
        if fixed:
            verify_fixed_capture(directory, capture, owners, views[frame])
        at_crossing = directed and frame in CROSSING_EYES
        if active != PAGE_CAP and not at_crossing:
            continue
        adjacency = opposing_lods(world, owners, colors, edges, lookup)
        if at_crossing:
            pairs = adjacency.get(crossing_edge, [])
            count, mixed_count = pair_counts(pairs)
            report["crossing"]["frames"].append(
                {"frame": frame, "eye": camera["eye"], "activePages": active,
                 "residentPages": stream["residentPages"], "opposingPairs": count,
                 "opposingMixedLODPairs": mixed_count, "lodPairs": pairs})
            require(count > 0, f"Frame {frame}: original crossing seam is not visible")
            if frame == CROSSING_FRAME:
                require(active == PAGE_CAP and mixed_count > 0,
                        "Frame 19: expected an 80-page saturated mixed-LOD crossing")
        if active == PAGE_CAP:
            for edge, pairs in adjacency.items():
                _, mixed_count = pair_counts(pairs)
                if mixed_count > witnesses.get(edge, {}).get("opposingMixedLODPairs", 0):
                    witnesses[edge] = {"case": case_id, "frame": frame, "activePages": active,
                                       "opposingMixedLODPairs": mixed_count, "lodPairs": pairs}

    report["cases"].append({"id": case_id, "frames": len(streams), "captures": len(captures),
                            "peakActivePages": max(row["activePages"] for row in streams),
                            "budgetDeferrals": sum(row["allocationBudgetDeferrals"] for row in streams),
                            "mixedSeamsAtCap": len(witnesses)})
    return witnesses, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="Original motion-tight capture directory")
    parser.add_argument("--directed", type=Path, required=True, help="Directed 8x8 motion capture directory")
    parser.add_argument("--fixed", type=Path, required=True, help="Original three fixed-view capture directory")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"protocol": "terrain-8x8-streaming-regression-v1", "passed": False,
              "pageLimit": PAGE_CAP, "lockedFallbackPages": LOCKED_PAGES, "cases": []}
    try:
        data = fixture()
        tiles, _, _, _, edges, _ = data
        report["internalSeams"] = len(edges)
        witnesses, identities = {}, set()
        for directory, name, size in ((args.directed, "motion-directed-8x8-tight", MOTION_SIZE),
                                      (args.motion, "motion-tight", MOTION_SIZE), (args.fixed, "fixed-tight", FIXED_SIZE)):
            observed, identity = scan(directory, name, size, data, report)
            identities.add(identity)
            for edge, witness in observed.items():
                if witness["opposingMixedLODPairs"] > witnesses.get(edge, {}).get("opposingMixedLODPairs", 0):
                    witnesses[edge] = witness
        require(len(identities) == 1, "Captures use different cooked stream identities")
        report["mixedSeamsAtCap"] = len(witnesses)
        report["missingEdges"] = [[tiles[a], tiles[b]] for index, (a, b, _, _) in enumerate(edges) if index not in witnesses]
        report["witnesses"] = [{"tiles": [tiles[edges[index][0]], tiles[edges[index][1]]], **witness}
                               for index, witness in sorted(witnesses.items())]
        require(not report["missingEdges"], f"Missing saturated opposing mixed-LOD witnesses for {report['missingEdges']}")
        report["passed"] = True
    except (KeyError, OSError, TypeError, ValueError) as error:
        report["error"] = str(error)
        print(f"8x8 terrain streaming regression failed: {error}", file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save(args.report, report)
    if report["passed"]:
        print(f"8x8 terrain streaming regression passed: {report['mixedSeamsAtCap']}/{report['internalSeams']} seams. {args.report}")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
