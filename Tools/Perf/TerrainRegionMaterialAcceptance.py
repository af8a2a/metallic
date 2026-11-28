#!/usr/bin/env python3
"""Compare terrain materials using the same geometry, cameras and regional handoff."""

import argparse
import math
from pathlib import Path
from statistics import median
import tempfile

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, read_json, write_json
from RunTerrainStreamingRegression import capture
from TerrainRegionStreamingAcceptance import (
    BYTE_LIMIT, LOAD_DISTANCE, PAGE_LIMIT, RESOLUTION, SCENE, STREAM, UNLOAD_DISTANCE, fixture,
)
from VerifyTerrainStreamingRegression import REPORT_NAME, require
from VirtualShadowMapAcceptance import digest, run as run_vsm


VARIANTS = ("two", "four")
VIEWS = ("baseColor", "normalTexture", "roughness")
TIMEOUT = 180
WARMUP = 16


def route(template, terrain, far_height=125.0):
    split = terrain["splitX"]
    cruise = max(region["worldBounds"]["max"][1] for region in terrain["regions"]) + 24.0
    west, east = (split - 86.0, cruise, 19.0), (split + 86.0, cruise, 19.0)
    far, near = (split - 0.75, far_height, 34.0), (split - 0.75, 12.0, 38.0)
    poses = [west] * WARMUP
    stops = [len(poses) - 1]
    windows = {}
    captures = {}

    def move(target, steps):
        start = poses[-1]
        poses.extend(tuple(a + (b - a) * step / steps for a, b in zip(start, target))
                     for step in range(1, steps + 1))

    for label, pose, steps in (("far", far, 26), ("near", near, 10)):
        move(pose, steps)
        first = len(poses) + 8
        poses.extend([pose] * 24)
        windows[label] = [first, len(poses) - 1]
        captures[str(len(poses) - 1)] = label

    move((split + 1.25, *near[1:]), 4)
    crossing, = [i for i in range(1, len(poses)) if poses[i - 1][0] < split <= poses[i][0]]
    for i, label in enumerate(("handoffBefore", "handoff", "handoffAfter"), crossing - 1):
        captures[str(i)] = label
    windows["handoff"] = [crossing - 1, crossing + 1]
    move(east, 28)
    poses.extend([east] * 20)
    stops.append(len(poses) - 1)

    # Look straight down at the same world-space material band, including during the zoom.
    cameras = [{**template, "projection": "perspective", "up": [0.0, 0.0, -1.0],
                "fovDegrees": fov, "eye": [x, y, 32.0], "center": [x, y - 1.0, 32.0]}
               for x, y, fov in poses]
    return cameras, {"splitX": split, "crossings": [crossing], "stops": stops,
                     "captureFrames": captures, "timingWindows": windows}


def prepare(directory, *, weight_mip_only=False):
    terrain = fixture()
    graph = read_json(GRAPH)
    # The shared weight texture spans the whole world and needs a wider shot than the layer textures.
    cameras, timeline = route(graph["view"]["camera"], terrain, far_height=660.0 if weight_mip_only else 125.0)
    if weight_mip_only:
        _, far_frame = timeline["timingWindows"]["far"]
        cameras = cameras[:far_frame + 1]
        timeline["captureFrames"] = {str(far_frame): "weightMipFar"}
    else:
        write_json(directory / "Route.json", timeline)
    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    # Keep raster and VSM on terminal geometry so only the material evaluation varies.
    nodes["VBuffer"].update(path=SCENE.as_posix(), streamAssetPath=STREAM.as_posix(),
                            streamRegions=terrain["regions"], initialStreamRegions=[0],
                            geometryRegionLoadDistance=LOAD_DISTANCE, geometryRegionUnloadDistance=UNLOAD_DISTANCE,
                            maxResidentPages=PAGE_LIMIT, maxResidentBytes=BYTE_LIMIT, autoLod=False, lodLevel=31)
    nodes["Shadows"]["shadowLOD"] = False
    nodes["Deferred"]["path"] = SCENE.as_posix()

    views = ("baseColor",) if weight_mip_only else (*VIEWS, "final")
    for variant in VARIANTS:
        document = SCENE.with_name(f"terrain_regions_{variant}.metallic_scene.json")
        require(document.is_file(), f"Bake MetallicTerrainRegionMaterialsSample first: {document}")
        for view in views:
            diagnostic = view != "final"
            mode = "diagnostic" if diagnostic else "timing"
            case_dir = directory / ("weight-mip" if weight_mip_only else mode) / variant
            if diagnostic and not weight_mip_only:
                case_dir /= view
            nodes["Deferred"]["debugView"] = view
            graph_path = case_dir / "TerrainRegions.metallic_graph.json"
            write_json(graph_path, graph)
            spec = dict(id=f"terrain-regions-{variant}-{view}", graph=graph_path.name, scene=document.as_posix(),
                        width=RESOLUTION[0], height=RESOLUTION[1], frames=len(cameras), warmupFrames=WARMUP,
                        camera=cameras[0], temporalJitter=False, metadataOnly=True,
                        events=[{"frame": i, "camera": camera} for i, camera in enumerate(cameras) if i > 0
                                and camera != cameras[i - 1]])
            if diagnostic:
                spec.update(output="Deferred.color", comparisonOutput="Deferred.color",
                            captureFrames=sorted(map(int, timeline["captureFrames"])))
            write_json(case_dir / "Cases.json", {"version": 1, "mode": mode, "cases": [spec]})


def frame_measurements(report, timeline):
    case, = report["cases"]
    require(report["status"] == case["status"] == "passed", "Incomplete regional flight receipt")
    rows = []
    phases = []
    for frame in case["frames"]:
        number = frame["frame"]
        stream, = (item for item in frame["streaming"] if item["passName"] == "VBuffer")
        selection = stream["regionSelection"]
        published = selection["publishedRegions"]
        if not phases or phases[-1]["regions"] != published:
            phases.append({"frame": number, "regions": published})
        require((frame["actualRenderWidth"], frame["actualRenderHeight"]) == RESOLUTION and
                Path(stream["assetPath"]).resolve() == STREAM.resolve() and
                stream["activePages"] <= stream["maxResidentPages"] == PAGE_LIMIT and
                stream["geometryUsedBytes"] <= stream["geometryBudgetBytes"] == BYTE_LIMIT and
                stream["allocationFailures"] == stream["loadFailures"] == stream["requestOverflows"] == 0,
                f"Frame {number}: extent, geometry asset or stream failure")
        ready = frame["readiness"]
        require(ready["ready"] and ready["requiredPages"] == ready["completedPages"] > 0,
                f"Frame {number}: published fallback is not drawable")
        deferred, = (node for node in frame["nodes"] if node["name"] == "Deferred")
        gpu_ms = deferred["gpuMs"]
        require(math.isfinite(gpu_ms) and gpu_ms > 0.0, f"Frame {number}: invalid Deferred timestamp")
        rows.append({"frame": number, "warmup": frame["warmup"], "regions": published,
                     "deferredGpuMs": gpu_ms, "textureResidentBytes": stream["textureResidentBytes"],
                     "textureIdentity": stream["textureSemanticIdentitySha256"],
                     "geometryUsedBytes": stream["geometryUsedBytes"],
                     "pinnedTerminalBytes": selection["pinnedTerminalBytes"],
                     "pinnedTerminalPages": selection["pinnedTerminalPages"],
                     "committedTerminalPages": selection["committedTerminalPages"]})
    require([phase["regions"] for phase in phases] == [[0], [0, 1], [1]], "Flight did not publish and release both regions")
    for first, last in timeline["timingWindows"].values():
        require(all(row["regions"] == [0, 1] and not row["warmup"] for row in rows[first:last + 1]),
                f"Frames {first}–{last}: expected both regions ready after warmup")
    stops = [rows[frame] for frame in timeline["stops"]]
    for row, region in zip(stops, ([0], [1])):
        require(row["regions"] == region and row["pinnedTerminalPages"] == row["committedTerminalPages"],
                f"Frame {row['frame']}: the settled region does not own its committed roots")
    require(set(stops[0]["pinnedTerminalPages"]).isdisjoint(stops[1]["pinnedTerminalPages"]),
            "Inactive region roots remain pinned after settling")
    return rows, phases


def compare(runs, window):
    a, b = [[run["windows"][window]["deferredMedianMs"] for run in runs if run["variant"] == variant]
            for variant in VARIANTS]
    spread = max(max(a) - min(a), max(b) - min(b))
    separation = max(min(b) - max(a), min(a) - max(b))
    delta = median(b) - median(a)
    relative = delta / median(a)
    if abs(relative) < 0.01:
        verdict = "below-one-percent"
    elif separation > spread:
        verdict = "four-slower" if delta > 0.0 else "four-faster"
    else:
        verdict = "not-separated"
    return dict(twoMedianMs=median(a), fourMedianMs=median(b), deltaMs=delta, relativeDelta=relative,
                spreadMs=spread, separationMs=separation, verdict=verdict)


def measure(directory, executable):
    timeline = read_json(directory / "Route.json")
    runs = []
    baseline = None
    for index, variant in enumerate(VARIANTS * 4, 1):
        case_dir = directory / "timing" / variant
        output = directory / "runs" / f"{index:02d}-{variant}"
        print(f"Deferred timing process {index}: {variant}", flush=True)
        run_vsm(argparse.Namespace(cases=case_dir / "Cases.json", exe=executable, output=output,
                                   runs=1, timeout=TIMEOUT))
        receipt = output / "run-1" / REPORT_NAME
        report = read_json(receipt)
        case, = report["cases"]
        spec = case["spec"]
        workload = ({key: value for key, value in spec.items() if key not in ("id", "scene")}, case["resolvedGraph"])
        require(baseline is None or baseline == workload, f"{variant}: timing route or production graph differs")
        baseline = workload
        rows, phases = frame_measurements(report, timeline)
        windows = {}
        for label, (first, last) in timeline["timingWindows"].items():
            subset = rows[first:last + 1]
            windows[label] = dict(deferredMedianMs=median(row["deferredGpuMs"] for row in subset),
                                  textureResidentBytes=sorted({row["textureResidentBytes"] for row in subset}))
        stops = [rows[frame] for frame in timeline["stops"]]
        runs.append(dict(process=index, variant=variant, windows=windows, phases=phases, stops=stops,
                         receipt=receipt.relative_to(directory).as_posix(), sha256=digest(receipt), frames=rows))
        if index not in (4, 8):
            continue
        results = {label: compare(runs, label) for label in ("near", "far")}
        write_json(directory / "Timing.json", {"runs": runs, "staticWindows": results, "route": timeline})
        if all(result["verdict"] != "not-separated" for result in results.values()):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/terrain-region-material-validation")
    parser.add_argument("--mode", choices=("all", "diagnostic", "timing"), default="all")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    print(directory, flush=True)
    prepare(directory)
    if args.mode != "timing":
        timeline = read_json(directory / "Route.json")
        publication = {}
        for variant in VARIANTS:
            for view in VIEWS:
                case = directory / "diagnostic" / variant / view
                output = capture(args.executable.resolve(), case, timeout=TIMEOUT)
                _, phases = frame_measurements(read_json(output / REPORT_NAME), timeline)
                publication[f"{variant}/{view}"] = phases
        from TerrainRegionMaterialVerification import verify

        write_json(directory / "Continuity.json", {"materials": verify(directory), "publicationPhases": publication,
                                                    "weightMipFar": capture_weight_mip(directory, args.executable.resolve())})
    if args.mode != "diagnostic":
        measure(directory, args.executable.resolve())


def capture_weight_mip(directory, executable):
    from TerrainRegionMaterialVerification import verify_weight_mip

    prepare(directory, weight_mip_only=True)
    for variant in VARIANTS:
        capture(executable, directory / "weight-mip" / variant, timeout=TIMEOUT)
    return verify_weight_mip(directory)


if __name__ == "__main__":
    main()
