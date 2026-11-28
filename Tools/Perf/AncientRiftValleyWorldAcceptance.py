#!/usr/bin/env python3
"""Compare the 1 km terrain's selected regions against the all-region terminal cut."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, read_json, write_json
from RunTerrainStreamingRegression import capture
from VerifyTerrainStreamingRegression import REPORT_NAME, camera_timeline, capture_geometry, raw_pixels, require

sys.path.insert(0, str(ROOT / "Tools"))
from BuildTerrainMesh import load_height_field


SCENE = ROOT / "Asset/MeshletCache/AncientRiftValley4x4/ancient_rift_valley_4x4.gltf"
HEIGHT = ROOT / "Asset/Terrain/AncientRiftValley/Maps/Height.exr"
TARGET = "MetallicAncientRiftValley4x4Sample"
RESOLUTION = (640, 360)
PROBE_XZ = (325.1953125, 257.8125)
SUN = (0.952424, -0.304776, 0.0)
REMOTE_CASTER = 13  # x1_z3
NEAR_LOAD = 44.0
NEAR_UNLOAD = 56.0
MODES = ("all-regions", "selected", "eye-distance")


def nadir(x, y, z, fov):
    return dict(projection="perspective", eye=[x, y, z], center=[x, y - 1.0, z], up=[0.0, 0.0, -1.0],
                fovDegrees=fov, znear=0.5, zfar=3500.0)


def route():
    x, z = PROBE_XZ
    poses = [nadir(x, 148.5654, z, 26.0)] * 20
    for step in range(1, 13):
        poses.append(nadir(x + (308.0 - x) * step / 12, 147.0, z, 24.0))
    for step in range(1, 33):
        poses.append(nadir(308.0 - 161.0 * step / 32, 147.0, z, 24.0))
    crossing, = [frame for frame in range(32, len(poses))
                 if poses[frame - 1]["eye"][0] >= 250.0 > poses[frame]["eye"][0]]
    poses.extend([poses[-1]] * 16)
    # Jump from a settled valley cut to a view of the whole world and allow the new terminal cut to publish.
    cut = len(poses)
    poses.extend([nadir(0.0, 1450.0, 0.0, 56.0)] * 28)
    checkpoints = {19: "probe", crossing - 1: "crossingBefore", crossing: "crossing", crossing + 1: "crossingAfter",
                   cut - 1: "westSettled", cut: "overviewCut", len(poses) - 1: "overviewSettled"}
    return poses, dict(crossing=crossing, cut=cut, checkpoints=checkpoints)


def fixture():
    stream = Path(f"{SCENE}.meshstream.bin")
    require(SCENE.is_file() and stream.is_file(), f"Build {TARGET} first")
    regions = read_json(SCENE)["extras"]["metallicTerrainRegions"]["regions"]
    receipt = read_json(SCENE.with_suffix(".meshstream.regions.json"))["regions"]
    ids = [f"x{x}_z{z}" for z in range(4) for x in range(4)]
    require([r["id"] for r in regions] == [r["id"] for r in receipt] == ids, "Expected the full 4×4 regional world")
    require(all(r["terminalPages"] for r in receipt), "Each region must supply a terminal cut")
    return stream, regions, receipt


def prepare(directory):
    stream, regions, _ = fixture()
    graph = read_json(GRAPH)
    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    nodes["VBuffer"].update(path=SCENE.as_posix(), streamAssetPath=stream.as_posix(), streamRegions=regions,
                            initialStreamRegions=list(range(len(regions))), initialLoad=True,
                            maxResidentPages=8192, maxResidentBytes=512 * 1024 * 1024,
                            maxLockedFallbackPages=4096, maxActiveGroups=65536,
                            autoLod=False, lodLevel=31, compactShadingAttributes=False)
    nodes["Shadows"].update(clipmapCount=9, physicalPageBudget=1024, shadowLOD=False, shadowBias=0.008)
    nodes["Deferred"]["path"] = SCENE.as_posix()
    camera_path, timeline = route()
    write_json(directory / "Route.json", timeline)
    for mode in MODES:
        all_regions = mode == "all-regions"
        negative = mode == "eye-distance"
        properties = nodes["VBuffer"]
        properties.update(geometryRegionLoadDistance=2500.0 if all_regions else NEAR_LOAD,
                          geometryRegionUnloadDistance=2500.0 if all_regions else NEAR_UNLOAD)
        if mode == "selected":
            properties["geometryRegionSelectionMode"] = "viewAndDirectionalCasters"
        else:
            properties.pop("geometryRegionSelectionMode", None)
        cameras = camera_path[:20] if negative else camera_path
        capture_frames = [19] if negative else sorted(timeline["checkpoints"])
        comparisons = [] if negative else [19, timeline["cut"] - 1, len(cameras) - 1]
        spec = dict(id=mode, graph="ValleyWorld.metallic_graph.json", scene=SCENE.as_posix(),
                    width=RESOLUTION[0], height=RESOLUTION[1], frames=len(cameras), warmupFrames=0,
                    camera=cameras[0], temporalJitter=False, requireShadow=True, metadataOnly=True,
                    events=[dict(frame=i, camera=view, cameraCut=(i == timeline["cut"]))
                            for i, view in enumerate(cameras) if i > 0 and view != cameras[i - 1]],
                    captureFrames=capture_frames, comparisonFrames=comparisons)
        write_json(directory / mode / spec["graph"], graph)
        write_json(directory / mode / "Cases.json", {"version": 1, "mode": "diagnostic", "cases": [spec]})


def read_case(directory, mode, stream_path, packed, cameras, cut):
    case_dir = directory / mode
    output = case_dir / "capture"
    report = read_json(output / REPORT_NAME)
    messages = report["validationMessages"]
    require(report["protocol"] == "virtual-shadow-map-acceptance-v2" and report["mode"] == "diagnostic"
            and report["validation"] and report["status"] == "passed"
            and messages["count"] == 0 and not messages["captureFailed"], f"{mode}: failed Vulkan diagnostic")
    case, = report["cases"]
    spec, = read_json(case_dir / "Cases.json")["cases"]
    require(case["spec"] == spec and case["id"] == mode and case["status"] == "passed", f"{mode}: invalid manifest")
    frames, cuts = camera_timeline(spec)
    require((frames, cuts) == ((cameras[:20], []) if mode == "eye-distance" else (cameras, [cut])),
            f"{mode}: capture used a different route")
    require([row["frame"] for row in case["frames"]] == list(range(spec["frames"])), f"{mode}: incomplete frame data")
    require([row["frame"] for row in case["captures"]] == spec["captureFrames"], f"{mode}: incomplete captures")
    comparisons = case["comparisons"]
    require([row["frame"] for row in comparisons] == spec["comparisonFrames"] and
            all(row["geometryIdentityStable"] and row["byteEqual"] and row["changedShadowBytes"] == 0
                for row in comparisons), f"{mode}: cache/forced redraw comparison failed")
    for row in comparisons:
        cached, redrawn = [next(entry["regionSelection"] for entry in row[key] if entry["passName"] == "VBuffer")
                            for key in ("cachedStreaming", "redrawnStreaming")]
        require(all(cached[key] == redrawn[key] for key in ("publishedRegions", "publicationRevision")),
                f"{mode} frame {row['frame']}: redraw changed the published geometry cut")
    rows = []
    for frame in case["frames"]:
        vbuffer, = (value for value in frame["streaming"] if value["passName"] == "VBuffer")
        info = vbuffer["regionSelection"]
        published = info["publishedRegions"]
        ready = frame["readiness"]
        roots = set().union(*(set(packed[index]["terminalPages"]) for index in published))
        require((frame["actualRenderWidth"], frame["actualRenderHeight"]) == RESOLUTION
                and Path(vbuffer["assetPath"]).resolve() == stream_path.resolve()
                and vbuffer["loadFailures"] == vbuffer["requestOverflows"] == vbuffer["allocationFailures"] == 0
                and ready["ready"] and ready["requiredPages"] == ready["completedPages"] == len(roots)
                and roots and roots <= set(info["pinnedTerminalPages"]),
                f"{mode} frame {frame['frame']}: incomplete or failing terminal cut")
        rows.append(dict(frame=frame["frame"], published=published, wanted=info["wantedRegions"],
                         roots=len(roots), pinned=len(info["pinnedTerminalPages"]),
                         committed=len(info["committedTerminalPages"]), geometryBytes=vbuffer["geometryUsedBytes"],
                         revision=info["publicationRevision"], shadowPages=frame["shadow"]["totals"][12]))
    return output, rows, {shot["frame"]: shot for shot in case["captures"]}, spec["comparisonFrames"]


def read_shadow(directory, shot):
    layers = shot["rawImages"]["Shadows.shadows"]["layerCount"]
    return raw_pixels(directory, shot, "Shadows.shadows", RESOLUTION, "u1", "R8Unorm", layers)[0, ..., 0]


def probe_fixture():
    # Cross-check that source geometry still gives the probe a remote up-sun occluder.
    height, _, _ = load_height_field(HEIGHT, "Y")
    x, z = PROBE_XZ
    spacing = 1000.0 / (height.shape[1] - 1)
    ix, iz = (round((point + 500.0) / spacing) for point in (x, z))
    source_y = float(height[iz, ix])
    world_x = np.arange(height.shape[1]) * spacing - 500.0
    candidate = (world_x >= -250) & (world_x < 0)
    light_y = source_y + 0.15 + (x - world_x[candidate]) * (-SUN[1] / SUN[0])
    margins = height[iz, candidate] - light_y
    require(float(margins.max()) > 20, "The source ridge no longer shadows the valley probe")
    return dict(receiver=[x, source_y, z], casterRegion="x1_z3", sunDirection=SUN,
                ridgeHeightOverRay=float(margins.max()), firstOccluderX=float(world_x[candidate][margins > 0][-1]))


def probe_visibility(directory, shot, camera):
    world, covered = capture_geometry(directory, shot, camera, RESOLUTION)
    xz = world[..., [0, 2]]
    patch = covered & np.all(np.abs(xz - PROBE_XZ) < 1.1, axis=-1)
    require(int(patch.sum()) >= 100, "Expected terrain coverage around the remote shadow probe")
    mask = read_shadow(directory, shot)[patch] / 255.0
    return dict(pixels=int(patch.sum()), mean=float(mask.mean()), median=float(np.median(mask)))


def compare_view(all_path, all_shot, selected_path, selected_shot, camera):
    all_world, all_covered = capture_geometry(all_path, all_shot, camera, RESOLUTION)
    selected_world, selected_covered = capture_geometry(selected_path, selected_shot, camera, RESOLUTION)
    both = all_covered & selected_covered
    all_shadow = read_shadow(all_path, all_shot)
    selected_shadow = read_shadow(selected_path, selected_shot)
    region_xz = np.clip(((all_world[both][:, [0, 2]] + 500) // 250).astype(int), 0, 3)
    indices = region_xz[:, 0] + 4 * region_xz[:, 1]
    return dict(referencePixels=int(all_covered.sum()), selectedPixels=int(selected_covered.sum()),
                missingPixels=int(np.count_nonzero(all_covered & ~selected_covered)),
                extraPixels=int(np.count_nonzero(selected_covered & ~all_covered)),
                changedShadowPixels=int(np.count_nonzero((all_shadow != selected_shadow) & both)),
                visibleRegions=np.unique(indices).tolist(), regionPixels=np.bincount(indices, minlength=16).tolist(),
                commonWorldPositionMaxError=float(np.linalg.norm(all_world[both] - selected_world[both], axis=-1).max(initial=0)))


def verify(directory):
    stream, _, packed = fixture()
    cameras, timeline = route()
    recorded = read_json(directory / "Route.json")
    require(recorded["crossing"] == timeline["crossing"] and recorded["cut"] == timeline["cut"], "Route changed")
    cut = timeline["cut"]
    reference, reference_rows, reference_shots, reference_redraws = read_case(directory, "all-regions", stream, packed, cameras, cut)
    selected, selected_rows, selected_shots, selected_redraws = read_case(directory, "selected", stream, packed, cameras, cut)
    negative, negative_rows, negative_shots, _ = read_case(directory, "eye-distance", stream, packed, cameras, cut)
    all_ids = list(range(len(packed)))
    require(all(row["published"] == all_ids for row in reference_rows), "Reference must keep the full world published")
    require(REMOTE_CASTER in selected_rows[19]["published"] and REMOTE_CASTER not in negative_rows[19]["published"],
            "The selected path must retain the remote caster that eye-distance selection releases")
    probes = {mode: probe_visibility(path, shot, cameras[19]) for mode, path, shot in (
        ("allRegions", reference, reference_shots[19]), ("selected", selected, selected_shots[19]),
        ("eyeDistance", negative, negative_shots[19]))}
    require(max(probes["allRegions"]["mean"], probes["selected"]["mean"]) <= 0.25 and
            probes["eyeDistance"]["mean"] >= 0.65, f"Remote ridge shadow probe failed: {probes}")

    views = {}
    for frame, label in timeline["checkpoints"].items():
        evidence = compare_view(reference, reference_shots[frame], selected, selected_shots[frame], cameras[frame])
        if label != "overviewCut":
            require(evidence["referencePixels"] > 1000 and evidence["missingPixels"] == evidence["extraPixels"] == 0
                    and evidence["commonWorldPositionMaxError"] <= 0.002,
                    f"{label}: selected terrain lost or changed the full-cut depth/coverage: {evidence}")
            require(evidence["changedShadowPixels"] == 0,
                    f"{label}: selected terrain changed {evidence['changedShadowPixels']} visible shadow pixels")
        evidence.update(frame=frame, selectedImage=(selected / selected_shots[frame]["file"]).as_posix(),
                        referenceImage=(reference / reference_shots[frame]["file"]).as_posix())
        views[label] = evidence
    overview = views["overviewSettled"]
    require(min(overview["regionPixels"]) >= 100 and selected_rows[-1]["published"] == all_ids,
            "The settled overview must draw every region")

    for label in ("crossingBefore", "crossing", "crossingAfter"):
        require({14, 15} <= set(views[label]["visibleRegions"]), f"{label}: both shared-edge regions must be visible")
    crossing = timeline["crossing"]
    start, end = selected_rows[19], selected_rows[timeline["cut"] - 1]
    require({14, 15} <= set(selected_rows[crossing - 1]["published"]) and 15 not in end["published"] and 14 in end["published"]
            and start["pinned"] == start["committed"] and end["pinned"] == end["committed"]
            and end["roots"] < reference_rows[0]["roots"] and end["geometryBytes"] < reference_rows[0]["geometryBytes"],
            f"The selected cut did not retain the crossing destination and release the old region: {start}, {end}")
    overview_ready = next(row["frame"] for row in selected_rows[cut:] if row["published"] == all_ids)
    write_json(directory / "AncientRiftValleyWorldAcceptance.json", dict(
        passed=True, terminalPagesAllRegions=reference_rows[0]["roots"],
        crossingFrame=crossing, overviewCutFrame=cut, overviewReadyFrame=overview_ready,
        views=views, remoteProbe={**probe_fixture(), "masks": probes},
        cachedRedrawEqual={"allRegions": reference_redraws, "selected": selected_redraws},
        regionResidency={"allRegions": reference_rows, "selected": selected_rows, "eyeDistance": negative_rows}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        directory = args.output.resolve()
        if args.action == "run":
            require(args.executable and args.executable.is_file(), "Build and provide MetallicRHITests")
            directory.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
        print(f"Ancient Rift Valley world evidence: {directory}", flush=True)
        if args.action != "verify":
            prepare(directory)
        if args.action == "run":
            for mode in MODES:
                capture(args.executable.resolve(), directory / mode, timeout=480)
        if args.action != "prepare":
            verify(directory)
    except (KeyError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Ancient Rift Valley world validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
