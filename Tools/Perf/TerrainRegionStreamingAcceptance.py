#!/usr/bin/env python3
"""Fly across the cooked terrain regions and verify their GPU root handoff."""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, read_json, write_json
from RunTerrainStreamingRegression import capture
from VerifyTerrainStreamingRegression import (
    REPORT_NAME, camera_timeline, capture_geometry, require,
)


SCENE = ROOT / "Asset/MeshletCache/TerrainRegions/terrain_regions.gltf"
PACK_RECEIPT = SCENE.with_name("terrain_regions.meshstream.regions.json")
STREAM = Path(str(SCENE) + ".meshstream.bin")
CASE = "terrain-region-flight"
RESOLUTION = (640, 360)
PAGE_LIMIT = 80
BYTE_LIMIT = 1024 * 1024
LOAD_DISTANCE = 44.0
UNLOAD_DISTANCE = 56.0


def fixture():
    require(STREAM.is_file(), f"Bake the terrain region sample first: {STREAM}")
    require(PACK_RECEIPT.is_file(), f"Missing the terrain region packing receipt: {PACK_RECEIPT}")
    terrain = read_json(SCENE)["extras"]["metallicTerrainRegions"]
    split = terrain["splitX"]
    west, east = terrain["regions"]
    require([west["id"], east["id"]] == ["west", "east"] and
            west["worldBounds"]["max"][0] == east["worldBounds"]["min"][0] == split,
            "Expected adjacent west and east regions sharing the authored plane")
    return terrain


def flight(template, terrain, round_trip):
    split = terrain["splitX"]
    west, east = [region["worldBounds"] for region in terrain["regions"]]
    y = max(west["max"][1], east["max"][1]) + 24.0
    z = 32.0
    # A narrow, downward view fits within a single region outside the handoff.
    # Near the seam it contains ground on both sides without visibility cuts.
    camera = {**template, "projection": "perspective", "up": [0.0, 0.0, -1.0], "fovDegrees": 19.0}
    start, end = split - 86.0, split + 86.0
    x = [start] * 16
    stops = [len(x) - 1]

    def leg(direction, stop):
        for target, steps, hold in ((split - direction * 8.0, 26, 14),
                                    (split + direction * 8.0, 8, 0), (stop, 26, 20)):
            begin = x[-1]
            x.extend(begin + (target - begin) * step / steps for step in range(1, steps + 1))
            x.extend([target] * hold)
        stops.append(len(x) - 1)

    leg(1.0, end)
    if round_trip:
        leg(-1.0, start)
    crossings = [frame for frame, (before, after) in enumerate(zip(x, x[1:]), 1)
                 if (before >= split) != (after >= split)]
    require(len(crossings) == (2 if round_trip else 1), "Unexpected camera crossing count")
    cameras = [{**camera, "eye": [position, y, z], "center": [position, 0.0, z]} for position in x]
    return cameras, crossings, stops


def prepare(directory, round_trip):
    terrain = fixture()
    graph = read_json(GRAPH)
    cameras, crossings, stops = flight(graph["view"]["camera"], terrain, round_trip)
    nodes = {node["name"]: node for node in graph["nodes"]}
    # Manual LOD 31 and the VSM terminal caster path isolate root commitment
    # from raster and shadow detail requests.
    nodes["VBuffer"]["properties"].update(
        path=SCENE.as_posix(), streamAssetPath=STREAM.as_posix(), streamRegions=terrain["regions"],
        initialStreamRegions=[0], geometryRegionLoadDistance=LOAD_DISTANCE,
        geometryRegionUnloadDistance=UNLOAD_DISTANCE, maxResidentPages=PAGE_LIMIT, maxResidentBytes=BYTE_LIMIT,
        autoLod=False, lodLevel=31, initialLoad=True,
    )
    nodes["Shadows"]["properties"]["shadowLOD"] = False
    nodes["Deferred"]["properties"]["path"] = SCENE.as_posix()
    graph_path = directory / "TerrainRegions.metallic_graph.json"
    write_json(graph_path, graph)
    captures = sorted({0, *stops, *(frame + delta for frame in crossings for delta in (-1, 0, 1))})
    spec = {
        "id": CASE, "graph": graph_path.name, "scene": SCENE.as_posix(),
        "width": RESOLUTION[0], "height": RESOLUTION[1], "frames": len(cameras), "camera": cameras[0],
        "events": [{"frame": i, "camera": cameras[i]} for i in range(1, len(cameras))
                   if cameras[i] != cameras[i - 1]],
        "captureFrames": captures, "streamingEvidenceFrames": list(range(len(cameras))),
        "warmupFrames": 0, "temporalJitter": False, "requireShadow": True, "metadataOnly": True,
        "allowResidentBudgetDeferral": True, "requireResidentBudgetDeferral": False,
    }
    write_json(directory / "Cases.json", {"version": 1, "mode": "diagnostic", "cases": [spec]})
    write_json(directory / "Route.json", {"splitX": terrain["splitX"], "crossings": crossings, "stops": stops})


def seam_coverage(directory, image, camera, split):
    world, covered = capture_geometry(directory, image, camera, RESOLUTION)
    x = world[..., 0] - split
    west = covered & (x < -0.01) & (x >= -6.0)
    east = covered & (x > 0.01) & (x <= 6.0)
    pairs = (west[:, :-1] & east[:, 1:]) | (east[:, :-1] & west[:, 1:])
    first, second = world[:, :-1][pairs], world[:, 1:][pairs]
    adjacent = int(np.count_nonzero(np.linalg.norm(first - second, axis=-1) <= 4.0))
    return {"westPixels": int(west.sum()), "eastPixels": int(east.sum()), "adjacentAcrossSeam": adjacent,
            "coveredPixels": int(covered.sum())}


def wanted_regions(camera, bounds, previous):
    eye = np.asarray(camera["eye"], dtype=np.float32)[[0, 2]]
    result = []
    for index, (lower, upper) in enumerate(bounds):
        delta = np.maximum(np.maximum(lower - eye, eye - upper), 0.0)
        threshold = UNLOAD_DISTANCE if index in previous else LOAD_DISTANCE
        if np.hypot(delta[0], delta[1]) <= threshold:
            result.append(index)
    return result


def verify(directory, report):
    terrain = fixture()
    packed = read_json(PACK_RECEIPT)["regions"]
    region_roots = [set(region["terminalPages"]) for region in packed]
    region_bytes = [region["terminalBytes"] for region in packed]
    require([region["id"] for region in packed] == [region["id"] for region in terrain["regions"]]
            and all(region_roots) and region_roots[0].isdisjoint(region_roots[1]),
            "Expected two independent packed terminal page lists")
    expected_peak = sum(region_bytes)
    all_roots = region_roots[0] | region_roots[1]
    for region, roots in zip(packed, region_roots):
        require(all(region["firstPage"] <= root < region["firstPage"] + region["pageCount"] for root in roots),
                f"{region['id']}: terminal pages outside their independent source range")
    route = read_json(directory / "Route.json")
    require(route["splitX"] == terrain["splitX"], "Route does not cross the authored region boundary")
    evidence_dir = directory / "capture"
    receipt = read_json(evidence_dir / REPORT_NAME)
    validation = receipt["validationMessages"]
    require(receipt["protocol"] == "virtual-shadow-map-acceptance-v2" and receipt["mode"] == "diagnostic" and
            receipt["status"] == "passed" and receipt["validation"] and
            not validation["captureFailed"] and validation["count"] == 0,
            "Missing passing production VSM diagnostic without Vulkan validation errors")
    case, = receipt["cases"]
    spec, = read_json(directory / "Cases.json")["cases"]
    require(case["id"] == CASE and case["spec"] == spec and case["status"] == "passed",
            "Capture does not match the recorded flight")
    cameras, cuts = camera_timeline(spec)
    require(not cuts, "Region handoff must run in a continuous view")
    images = {image["frame"]: image for image in case["captures"]}
    require(list(images) == spec["captureFrames"], "Missing or unexpected flight capture")
    frames = case["frames"]
    require([row["frame"] for row in frames] == list(range(spec["frames"])), "Incomplete flight timeline")

    bounds = [np.asarray([item["worldBounds"]["min"], item["worldBounds"]["max"]], dtype=np.float32)[:, [0, 2]]
              for item in terrain["regions"]]
    phases, wanted_phases = [], []
    previous_wanted = [0]
    report["frames"] = []
    for row in frames:
        frame = row["frame"]
        stream, = [item for item in row["streaming"] if item["passName"] == "VBuffer"]
        require(Path(stream["assetPath"]).resolve() == STREAM.resolve() and
                stream["maxResidentPages"] == PAGE_LIMIT and stream["activePages"] <= PAGE_LIMIT and
                stream["geometryUsedBytes"] <= stream["geometryBudgetBytes"] == BYTE_LIMIT,
                f"Frame {frame}: unexpected stream identity or overcommitted page pool")
        require(stream["loadFailures"] == stream["requestOverflows"] == 0 and
                stream["allocationFailures"] == stream["allocationBudgetDeferrals"],
                f"Frame {frame}: unexpected stream failure")
        require((row["actualRenderWidth"], row["actualRenderHeight"]) == RESOLUTION and
                row["shadow"]["header"]["admittedCount"] > 0,
                f"Frame {frame}: expected the fixed extent and an admitted VSM source")
        gpu = row["streamingEvidence"]
        region = stream["regionSelection"]
        published, wanted = region["publishedRegions"], region["wantedRegions"]
        require(published and published == sorted(set(published)) and wanted == sorted(set(wanted)) and
                {0, 1}.issuperset(published) and {0, 1}.issuperset(wanted),
                f"Frame {frame}: invalid wanted or published region set")
        expected_wanted = wanted_regions(cameras[frame], bounds, previous_wanted)
        require(wanted == expected_wanted,
                f"Frame {frame}: expected wanted {expected_wanted} from the current camera and hysteresis, found {wanted}")
        if not wanted_phases or wanted != wanted_phases[-1]["regions"]:
            wanted_phases.append({"frame": frame, "regions": wanted})
        previous_wanted = expected_wanted
        revision = region["publicationRevision"]
        if not phases or published != phases[-1]["regions"]:
            require(not phases or revision > phases[-1]["revision"],
                    f"Frame {frame}: regions changed without a new publication")
            phases.append({"frame": frame, "regions": published, "revision": revision})
        locked = set(gpu["lockedFallbackPages"])
        active_roots = set().union(*(region_roots[index] for index in published))
        pinned = set(region["pinnedTerminalPages"])
        committed = set(region["committedTerminalPages"])
        require(active_roots <= locked <= all_roots,
                f"Frame {frame}: published region without drawable terminal roots")
        ready = row["readiness"]
        require(ready["ready"] and ready["requiredPages"] == ready["completedPages"] == len(active_roots),
                f"Frame {frame}: published region readiness does not match its terminal cut")
        require(sum(region_bytes[index] for index in published) <= region["pinnedTerminalBytes"] <= expected_peak and
                active_roots <= pinned <= all_roots and committed <= all_roots and
                0 <= region["committedTerminalBytes"] <= min(expected_peak, stream["geometryUsedBytes"]),
                f"Frame {frame}: inconsistent physically pinned terrain roots")
        report["frames"].append({"frame": frame, "eyeX": cameras[frame]["eye"][0], **region,
                                  "gpuLockedFallbackPages": len(locked), "activePages": stream["activePages"],
                                  "residentPages": stream["residentPages"], "geometryUsedBytes": stream["geometryUsedBytes"],
                                  "geometryBudgetBytes": stream["geometryBudgetBytes"],
                                  "geometryBudgetDeferrals": stream["allocationBudgetDeferrals"]})
        if frame in images:
            visible = gpu["visiblePages"]
            require(visible and all(page["page"] in locked for page in visible),
                    f"Frame {frame}: coverage proof must use drawable terminal pages")

    expected = [[0], [0, 1], [1]]
    if len(route["crossings"]) == 2:
        expected += [[0, 1], [0]]
    report["publicationPhases"] = phases
    report["wantedPhases"] = wanted_phases
    require([phase["regions"] for phase in phases] == expected, "Unexpected region handoff sequence")
    require([phase["regions"] for phase in wanted_phases] == expected and
            all(request["frame"] <= publication["frame"] for request, publication in zip(wanted_phases, phases)),
            "Region wants and publications did not follow the same ordered handoff")
    require(max(row["pinnedTerminalBytes"] for row in report["frames"]) == expected_peak,
            "Never committed both regions' terminal bytes")
    require(max(row["committedTerminalBytes"] for row in report["frames"]) == expected_peak,
            "Both regions' terminal allocations were never committed together")

    overlaps = []
    for phase, next_phase in zip(phases, phases[1:]):
        if phase["regions"] != [0, 1]:
            continue
        first, end = phase["frame"], next_phase["frame"]
        peak_frame = max(report["frames"][first:end], key=lambda row: row["geometryUsedBytes"])
        overlaps.append({"firstFrame": first, "lastFrame": end - 1, "peakFrame": peak_frame["frame"],
                         "activePagesAtPeak": peak_frame["activePages"],
                         "peakGeometryUsedBytes": peak_frame["geometryUsedBytes"]})
    report["overlaps"] = overlaps
    report["settled"] = []
    require(len(route["stops"]) == len(route["crossings"]) + 1, "Unexpected number of settled region stations")
    for stop, (frame, index) in enumerate(zip(route["stops"], (0, 1, 0))):
        info = report["frames"][frame]
        gpu = frames[frame]["streamingEvidence"]
        require(info["publishedRegions"] == info["wantedRegions"] == [index] and
                set(info["pinnedTerminalPages"]) == set(info["committedTerminalPages"]) == region_roots[index] and
                info["pinnedTerminalBytes"] == info["committedTerminalBytes"] == region_bytes[index] and
                info["geometryUsedBytes"] >= region_bytes[index] and
                set(gpu["lockedFallbackPages"]) == region_roots[index],
                f"Frame {frame}: inactive region roots remain pinned after settling")
        active = packed[index]
        allocated = gpu["residentPages"] + gpu["pendingUploadPages"] + gpu["pendingUnloadPages"]
        require(all(active["firstPage"] <= page < active["firstPage"] + active["pageCount"] for page in allocated),
                f"Frame {frame}: inactive regional detail still occupies the GPU page table")
        overlap = overlaps[max(0, stop - 1)]
        released = overlap["peakGeometryUsedBytes"] - info["geometryUsedBytes"]
        report["settled"].append({"region": active["id"], "frame": frame,
                                  "terminalPages": info["gpuLockedFallbackPages"],
                                  "terminalBytes": info["committedTerminalBytes"],
                                  "activePages": info["activePages"], "residentPages": info["residentPages"],
                                  "geometryUsedBytes": info["geometryUsedBytes"],
                                  "overlapPeakFrame": overlap["peakFrame"],
                                  "geometryUsedBytesBelowOverlap": released})
        require(released > 0, f"Frame {frame}: settled geometry used bytes did not fall below the overlap peak")

    report["crossings"], report["handoffs"] = [], []
    split = terrain["splitX"]
    for leg, frame in enumerate(route["crossings"]):
        before, after = [cameras[index]["eye"][0] for index in (frame - 1, frame)]
        require((before >= split) != (after >= split), f"Frame {frame}: eye did not cross the region plane")
        overlap_phase = 2 * leg + 1
        report["handoffs"].append({"crossingFrame": frame, "fromRegion": terrain["regions"][int(before >= split)]["id"],
                                   "toRegion": terrain["regions"][int(after >= split)]["id"],
                                   "wantedOverlapFrame": wanted_phases[overlap_phase]["frame"],
                                   "publishedOverlapFrame": phases[overlap_phase]["frame"],
                                   "wantedDestinationOnlyFrame": wanted_phases[overlap_phase + 1]["frame"],
                                   "publishedDestinationOnlyFrame": phases[overlap_phase + 1]["frame"]})
        for index in (frame - 1, frame, frame + 1):
            info = report["frames"][index]
            coverage = seam_coverage(evidence_dir, images[index], cameras[index], split)
            visible_pages = {item["page"] for item in frames[index]["streamingEvidence"]["visiblePages"]}
            require(info["publishedRegions"] == [0, 1] and info["committedTerminalBytes"] == expected_peak and
                    all(visible_pages & roots for roots in region_roots) and
                    coverage["adjacentAcrossSeam"] > 0 and coverage["coveredPixels"] == np.prod(RESOLUTION),
                    f"Frame {index}: no drawable fallback continuity across the region seam")
            report["crossings"].append({"frame": index, **coverage})
    report["regionRootBytes"] = region_bytes
    report["peakPinnedTerminalBytes"] = expected_peak
    report["peakCommittedTerminalBytes"] = max(row["committedTerminalBytes"] for row in report["frames"])
    report["peakGeometryUsedBytes"] = max(row["geometryUsedBytes"] for row in report["frames"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--round-trip", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    if args.mode == "run":
        if args.executable is None or not args.executable.is_file():
            parser.error("Build MetallicRHITests and pass its path with --executable")
        directory.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
    print(f"Terrain region artifacts: {directory}", flush=True)

    report = {"protocol": "terrain-region-streaming-acceptance-v1", "passed": False}
    try:
        if args.mode != "verify":
            prepare(directory, args.round_trip)
        if args.mode == "prepare":
            return 0
        if args.mode == "run":
            capture(args.executable.resolve(), directory)
        verify(directory, report)
        report["passed"] = True
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        report["error"] = str(error)
        print(f"Terrain region acceptance failed: {error}", file=sys.stderr)
    write_json(directory / "TerrainRegionStreamingAcceptance.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
