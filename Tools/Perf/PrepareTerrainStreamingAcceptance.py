#!/usr/bin/env python3
"""Prepare fixed-camera and continuous-flight VSM acceptance cases for tiled terrain."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "Asset/MeshletCache/TerrainBlocks4x4/terrain_blocks_4x4.gltf"
SCENE_8X8 = ROOT / "Asset/MeshletCache/TerrainBlocks8x8/terrain_blocks_8x8.gltf"
SCENE_16X16 = ROOT / "Asset/MeshletCache/TerrainBlocks16x16/terrain_blocks_16x16.gltf"
GRAPH = ROOT / "Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json"
MOVING_STEPS = 84
EXPANDED_MOVING_STEPS = 112
SETTLING_FRAMES = 20

# Aim at the authored seam surface from both sides while preserving the streaming state.
DIRECTED_TARGETS = (
    (((0, 2), (0, 3)), (-103.0, 20.58487038176328, 63.785953521728516),
     ((-17, 42, 15), (11, 32, 17), (18, 41, -14))),
    (((1, 2), (1, 3)), (-36.0, 25.895427040027965, 63.785953521728516),
     ((-19, 46, -17), (13, 36, -12), (-9, 39, 18))),
    (((1, 3), (2, 3)), (-0.42809364199638367, 11.01667852658886, 93.0),
     ((-18, 47, -16), (-12, 36, 13), (16, 38, 14))),
    (((0, 0), (1, 0)), (-64.64214324951172, 6.660304326949838, -102.0),
     ((-21, 44, 11), (-13, 31, -14), (20, 39, -14))),
)


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


def expanded_flight_camera(template, frame):
    t = min(frame, EXPANDED_MOVING_STEPS) / EXPANDED_MOVING_STEPS
    eye = [-123.0 + 246.0 * t,
           70.0 + 3.0 * math.sin(2.0 * math.pi * t),
           121.0 - 240.0 * t - 8.0 * math.sin(math.pi * t)]
    # Keep the seam beneath the eye in view around every grid crossing.
    offset = [6.0, -64.0, -3.0]
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
               warmup=0, pressure=False, trace_streaming=False, scene=SCENE):
    directory = output / name
    graph = read_json(GRAPH)
    nodes = {node["name"]: node for node in graph["nodes"]}
    nodes["VBuffer"]["properties"].update(path=scene.as_posix(), streamAssetPath=scene.as_posix() + ".meshstream.bin",
                                           maxResidentPages=cap, visualization="lod", shadedDebugColors=False)
    nodes["Deferred"]["properties"]["path"] = scene.as_posix()
    graph_path = directory / "Terrain.metallic_graph.json"
    write_json(graph_path, graph)

    case = dict(id=name, graph=graph_path.name, scene=scene.as_posix(), width=width, height=height,
                frames=frames, camera=camera, warmupFrames=warmup, captureFrames=captures, events=events,
                temporalJitter=False, requireShadow=True, metadataOnly=True, output="VBuffer.color")
    if pressure:
        case["allowResidentBudgetDeferral"] = True
    if trace_streaming:
        case["streamingEvidenceFrames"] = list(range(frames))
    write_json(directory / "Cases.json", dict(version=1, mode="diagnostic", cases=[case]))


def write_directed_case(output, cap, flight, first_crossing):
    name = "motion-directed-tight"
    crossing = first_crossing["frame"]
    opening = crossing + 5
    crossing_frames = [crossing - 1, crossing, crossing + 1]
    # Let the south tile refine first, then bring both sides into view before crossing.
    south_aim = [-101.0, 19.0, 104.0]
    seam_aim = [-94.0, 13.35969996728095, DIRECTED_TARGETS[0][1][2]]
    approach_frames = 3
    approach_start = crossing_frames[0] - approach_frames
    cameras = []
    for frame, camera in enumerate(flight[:opening]):
        t = min(1.0, max(0.0, (frame - approach_start) / approach_frames))
        blend = t * t * (3 - 2 * t)
        center = [(1 - blend) * near + blend * seam for near, seam in zip(south_aim, seam_aim)]
        fov = (1 - blend) * 24.0 + blend * camera["fovDegrees"]
        cameras.append({**camera, "center": center, "fovDegrees": fov})
    captures = {0, 6, 10, *crossing_frames, opening - 1}
    observations = []
    for tiles, aim, offsets in DIRECTED_TARGETS:
        for view, offset in enumerate(offsets):
            end_pose = dict(eye=[coordinate + delta for coordinate, delta in zip(aim, offset)], center=aim)
            start = cameras[-1]
            travel = 12 if view == 0 and observations else 8
            for step in range(1, travel + 1):
                t = step / travel
                blend = t * t * (3 - 2 * t)
                pose = {
                    key: [(1 - blend) * a + blend * b for a, b in zip(start[key], end)]
                    for key, end in end_pose.items()
                }
                cameras.append({**start, **pose})
            arrival = len(cameras) - 1
            cameras.extend([cameras[-1]] * (17 if view == 0 else 11))
            final_frame = len(cameras) - 1
            captures.update((arrival - 2, final_frame))
            observations.append(dict(tiles=tiles, arrivalFrame=arrival, finalFrame=final_frame))

    # The northwest seam remains in view briefly on departure from the second southwest pose.
    departure = observations[1]["finalFrame"]
    captures.update((departure + 1, departure + 2))
    events = [dict(frame=frame, camera=camera) for frame, camera in enumerate(cameras)
              if frame and camera != cameras[frame - 1]]
    write_case(output, name, cap, cameras[0], events, 640, 360, len(cameras), sorted(captures),
               pressure=True, trace_streaming=True)
    crossing_target = {**first_crossing, "tiles": DIRECTED_TARGETS[0][0], "captureFrames": crossing_frames,
                       "mixedLODFrame": crossing}
    write_json(output / name / "Targets.json", dict(firstCrossing=crossing_target, observations=observations))


def append_camera_transition(cameras, end, frames):
    start = cameras[-1]
    for step in range(1, frames + 1):
        t = step / frames
        blend = t * t * (3 - 2 * t)
        pose = {key: [(1 - blend) * a + blend * b for a, b in zip(start[key], end[key])]
                for key in ("eye", "center")}
        fov = (1 - blend) * start["fovDegrees"] + blend * end["fovDegrees"]
        cameras.append({**start, **pose, "fovDegrees": fov})


def append_high_transit(cameras, end):
    altitude = max(73.0, cameras[-1]["eye"][1], end["eye"][1])

    def at_altitude(camera):
        x, _, z = camera["eye"]
        return {**camera, "eye": [x, altitude, z]}

    append_camera_transition(cameras, at_altitude(cameras[-1]), 3)
    append_camera_transition(cameras, at_altitude(end), 8)
    append_camera_transition(cameras, end, 3)


def write_directed_8x8_case(output, cap, flight, crossing):
    name = "motion-directed-8x8-tight"
    frame = crossing["frame"]
    crossing_frames = [frame - 1, frame, frame + 1]
    # Keep the original moving eyes. Prime the eastern tile before exposing the seam.
    prime_aim = [-52.0, 7.0, 47.0]
    seam_aim = [-64.64214324951172, -0.7328170010365, 52.0]
    cameras = []
    for index, camera in enumerate(flight[:frame + 3]):
        t = min(1.0, max(0.0, (index - (frame - 4)) / 3))
        blend = t * t * (3 - 2 * t)
        center = [(1 - blend) * a + blend * b for a, b in zip(prime_aim, seam_aim)]
        cameras.append({**camera, "center": center, "fovDegrees": (1 - blend) * 18.0 + blend * 60.0})

    captures = set(crossing_frames)
    stations = read_json(Path(__file__).with_name("Terrain8x8DirectedRoute.json"))["stations"]
    observations = []
    template = cameras[0]
    for station in stations:
        # A narrow view gives the selected tile the detail slots before revealing its neighbors.
        append_high_transit(cameras, {**template, **station["prime"]})
        entered = len(cameras) - 1
        cameras.extend([cameras[-1]] * 22)
        append_camera_transition(cameras, {**template, **station["reveal"]}, 1)
        reveal = len(cameras) - 1
        cameras.extend([cameras[-1]] * 2)
        samples = [reveal, reveal + 2]
        for view in station.get("additionalViews", []):
            append_camera_transition(cameras, {**template, **view}, 5)
            samples.append(len(cameras) - 1)
        captures.update(samples)
        observations.append(dict(tile=station["tile"], edges=station["edges"], primeFrame=entered,
                                 captureFrames=samples))

    events = [dict(frame=index, camera=camera) for index, camera in enumerate(cameras)
              if index and camera != cameras[index - 1]]
    write_case(output, name, cap, cameras[0], events, 640, 360, len(cameras), sorted(captures),
               pressure=True, trace_streaming=True, scene=SCENE_8X8)
    crossing_target = {**crossing, "tiles": [[1, 5], [2, 5]], "captureFrames": crossing_frames,
                       "mixedLODFrame": frame}
    write_json(output / name / "Targets.json", dict(crossing=crossing_target, observations=observations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-per-axis", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tiles = args.tiles_per_axis
    scene, folder = {4: (SCENE, "scale"), 8: (SCENE_8X8, "scale-8x8"),
                     16: (SCENE_16X16, "scale-16x16")}[tiles]
    output = args.output or ROOT / "build/terrain-vsm" / folder
    stream = Path(str(scene) + ".meshstream.bin")
    if not scene.is_file() or not stream.is_file():
        parser.error(f"Bake the scene with the MetallicTerrainBlocks{tiles}x{tiles}VSMSample CMake target first")

    graph = read_json(GRAPH)
    overview = graph["view"]["camera"]
    views = fixed_views(overview)
    expanded = tiles == 16
    moving_steps = EXPANDED_MOVING_STEPS if expanded else MOVING_STEPS
    total_frames = moving_steps + 1 + SETTLING_FRAMES
    sample_camera = expanded_flight_camera if expanded else flight_camera
    cameras = [sample_camera(overview, frame) for frame in range(total_frames)]
    terrain = read_json(scene)["extras"]["metallicTerrainTiles"]
    if terrain["tilesPerAxis"] != tiles:
        parser.error(f"Expected {tiles}x{tiles} terrain tiles")
    crossings = flight_crossings(cameras[:moving_steps + 1], terrain)
    expected_crossings = 2 * (tiles - 1)
    if len({crossing["frame"] for crossing in crossings}) != expected_crossings or len(crossings) != expected_crossings:
        parser.error(f"The {tiles}x{tiles} flight should cross all {expected_crossings} tile planes at distinct frames")
    if min(camera["eye"][1] for camera in cameras) <= terrain["worldHeightMax"]:
        parser.error("The flight should remain above the terrain's maximum height")

    flight_frames = {0, moving_steps, moving_steps + 10, total_frames - 1}
    if not expanded:
        flight_frames.update(range(0, moving_steps + 1, 4))
    flight_frames.update(crossing["frame"] + offset for crossing in crossings for offset in (-1, 0, 1))
    flight_events = [dict(frame=frame, camera=cameras[frame]) for frame in range(1, moving_steps + 1)]
    fixed_frames = sorted(view[key] for view in views for key in ("frame", "earlyFrame"))
    fixed_events = [dict(frame=view["earlyFrame"], camera=view["camera"], cameraCut=True) for view in views[1:]]
    default_cap = next(node for node in graph["nodes"] if node["name"] == "VBuffer")["properties"]["maxResidentPages"]
    # Tight cases leave 15 detail slots above their terminal cuts.
    tight_cap, ample_cap = {4: (32, default_cap), 8: (80, 512), 16: (272, 2048)}[tiles]
    for suffix, cap in (("tight", tight_cap), ("ample", ample_cap)):
        write_case(output, f"fixed-{suffix}", cap, overview, fixed_events, 960, 540, 60, fixed_frames,
                   warmup=16, pressure=suffix == "tight", scene=scene)
        write_case(output, f"motion-{suffix}", cap, cameras[0], flight_events, 640, 360, total_frames,
                   sorted(flight_frames), pressure=suffix == "tight", trace_streaming=True, scene=scene)
    for view in views:
        write_case(output, f"cold-{view['name']}", tight_cap, view["camera"], [], 960, 540, 1, [0], scene=scene)
    if tiles == 4:
        write_directed_case(output, tight_cap, cameras, crossings[0])
    elif tiles == 8:
        write_directed_8x8_case(output, tight_cap, cameras, next(crossing for crossing in crossings if crossing["frame"] == 19))
    write_json(output / "Route.json", dict(scene=scene.as_posix(), stream=stream.as_posix(),
                                                fixedViews=views, cameras=cameras, crossings=crossings,
                                                captureFrames=sorted(flight_frames), movingFrames=moving_steps + 1,
                                                totalFrames=total_frames, tightBudget=tight_cap, ampleBudget=ample_cap))
    print(f"Wrote terrain acceptance cases to {output.resolve()}")


if __name__ == "__main__":
    main()
