#!/usr/bin/env python3
"""Prepare fixed-camera and continuous-flight VSM acceptance cases for the 4x4 terrain sample."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "Asset/MeshletCache/TerrainBlocks4x4/terrain_blocks_4x4.gltf"
GRAPH = ROOT / "Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json"
MOVING_STEPS = 84
SETTLING_FRAMES = 20
TOTAL_FRAMES = MOVING_STEPS + 1 + SETTLING_FRAMES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def fixed_views(overview):
    shots = (
        ("overview", overview),
        ("southwest", {**overview, "eye": [-182.0, 54.0, 196.0], "center": [-3.0, 8.0, -1.0], "fovDegrees": 44.0}),
        ("northeast", {**overview, "eye": [178.0, 52.0, -192.0], "center": [-4.0, 9.0, 0.0], "fovDegrees": 44.0}),
    )
    return [dict(name=name, camera=camera, earlyFrame=index * 20, frame=(index + 1) * 20 - 1)
            for index, (name, camera) in enumerate(shots)]


def flight_camera(template, frame):
    t = min(frame, MOVING_STEPS) / MOVING_STEPS
    eye = [-112.0 + 210.0 * t,
           70.0 + 3.0 * math.sin(2.0 * math.pi * t),
           101.0 - 206.0 * t - 15.0 * math.sin(math.pi * t)]
    offset = [36.0 - 20.0 * t, -49.0, -28.0 + 36.0 * t]
    return {**template, "eye": eye, "center": [a + b for a, b in zip(eye, offset)], "fovDegrees": 60.0}


def flight_crossings(cameras, terrain):
    size = terrain["horizontalSize"]
    tiles = terrain["tilesPerAxis"]
    cuts = []
    for axis, resolution in zip((0, 2), terrain["sourceResolution"]):
        cells = resolution - 1
        cuts.extend((axis, -0.5 * size + size * (cells * tile // tiles) / cells)
                    for tile in range(1, tiles))

    crossings = []
    for frame, (before, after) in enumerate(zip(cameras, cameras[1:]), 1):
        for axis, plane in cuts:
            low, high = sorted((before["eye"][axis], after["eye"][axis]))
            if low < plane <= high:
                crossings.append(dict(frame=frame, axis="x" if axis == 0 else "z", plane=plane,
                                      before=before["eye"], after=after["eye"]))
    return crossings


def write_case(output, name, cap, camera, events, width, height, frames, captures, *,
               warmup=0, pressure=False, trace_streaming=False):
    directory = output / name
    graph = read_json(GRAPH)
    nodes = {node["name"]: node for node in graph["nodes"]}
    nodes["VBuffer"]["properties"].update(path=SCENE.as_posix(), streamAssetPath=SCENE.as_posix() + ".meshstream.bin",
                                           maxResidentPages=cap, visualization="lod", shadedDebugColors=False)
    nodes["Deferred"]["properties"]["path"] = SCENE.as_posix()
    graph_path = directory / "Terrain.metallic_graph.json"
    write_json(graph_path, graph)

    case = dict(id=name, graph=graph_path.name, scene=SCENE.as_posix(), width=width, height=height,
                frames=frames, camera=camera, warmupFrames=warmup, captureFrames=captures, events=events,
                temporalJitter=False, requireShadow=True, metadataOnly=True, output="VBuffer.color")
    if pressure:
        case["allowResidentBudgetDeferral"] = True
    if trace_streaming:
        case["streamingEvidenceFrames"] = list(range(frames))
    write_json(directory / "Cases.json", dict(version=1, mode="diagnostic", cases=[case]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/terrain-vsm/scale")
    args = parser.parse_args()
    if not SCENE.is_file() or not Path(SCENE.as_posix() + ".meshstream.bin").is_file():
        parser.error("Bake the scene with the MetallicTerrainBlocks4x4VSMSample CMake target first")

    graph = read_json(GRAPH)
    overview = graph["view"]["camera"]
    views = fixed_views(overview)
    cameras = [flight_camera(overview, frame) for frame in range(TOTAL_FRAMES)]
    terrain = read_json(SCENE)["extras"]["metallicTerrainTiles"]
    crossings = flight_crossings(cameras[:MOVING_STEPS + 1], terrain)
    if len({crossing["frame"] for crossing in crossings}) != 6 or len(crossings) != 6:
        parser.error("The 4x4 flight should cross all six tile planes at distinct frames")
    if min(camera["eye"][1] for camera in cameras) <= terrain["worldHeightMax"]:
        parser.error("The flight should remain above the terrain's maximum height")

    flight_frames = set(range(0, MOVING_STEPS + 1, 4)) | {MOVING_STEPS + 10, TOTAL_FRAMES - 1}
    flight_frames.update(crossing["frame"] + offset for crossing in crossings for offset in (-1, 0, 1))
    flight_events = [dict(frame=frame, camera=cameras[frame]) for frame in range(1, MOVING_STEPS + 1)]
    fixed_frames = sorted(view[key] for view in views for key in ("frame", "earlyFrame"))
    fixed_events = [dict(frame=view["earlyFrame"], camera=view["camera"], cameraCut=True) for view in views[1:]]
    tight_cap = 32
    ample_cap = next(node for node in graph["nodes"] if node["name"] == "VBuffer")["properties"]["maxResidentPages"]
    for suffix, cap in (("tight", tight_cap), ("ample", ample_cap)):
        write_case(args.output, f"fixed-{suffix}", cap, overview, fixed_events, 960, 540, 60, fixed_frames,
                   warmup=16, pressure=suffix == "tight")
        write_case(args.output, f"motion-{suffix}", cap, cameras[0], flight_events, 640, 360, TOTAL_FRAMES,
                   sorted(flight_frames), pressure=suffix == "tight", trace_streaming=True)
    for view in views:
        write_case(args.output, f"cold-{view['name']}", tight_cap, view["camera"], [], 960, 540, 1, [0])
    write_json(args.output / "Route.json", dict(scene=SCENE.as_posix(), stream=SCENE.as_posix() + ".meshstream.bin",
                                                fixedViews=views, cameras=cameras, crossings=crossings,
                                                captureFrames=sorted(flight_frames), movingFrames=MOVING_STEPS + 1,
                                                totalFrames=TOTAL_FRAMES, tightBudget=tight_cap, ampleBudget=ample_cap))
    print(f"Wrote terrain acceptance cases to {args.output.resolve()}")


if __name__ == "__main__":
    main()
