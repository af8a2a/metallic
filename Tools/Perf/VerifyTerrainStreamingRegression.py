#!/usr/bin/env python3
"""Verify the 4x4 terrain sample's cap-32 crossing and original fixed captures."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from VirtualShadowMapAcceptance import ROOT, read, save


SCENE = ROOT / "Asset/MeshletCache/TerrainBlocks4x4/terrain_blocks_4x4.gltf"
REPORT_NAME = "VirtualShadowMapAcceptance.json"
PAGE_CAP = 32
LOCKED_PAGES = 17
CROSSING_FRAME = 13
CROSSING_EYES = {
    12: [-82.0, 72.34549444740409, 65.0631724846652],
    13: [-79.5, 72.47871632294799, 62.11001819495169],
    14: [-77.0, 72.59807621135332, 59.16666666666667],
}
CROSSING_TILES = ((0, 2), (0, 3))
FIXED_VIEWS = (
    ("overview", 0, 19, [38., 91., 113.], [-27., 4., 4.], 60.),
    ("southwest", 20, 39, [-182., 54., 196.], [-3., 8., -1.], 44.),
    ("northeast", 40, 59, [178., 52., -192.], [-4., 9., 0.], 44.),
)
CLIP_CAMERA = dict(projection="perspective", up=[0., 1., 0.], znear=0.5, zfar=1000.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def camera_timeline(spec):
    events = {event["frame"]: event for event in spec.get("events", [])}
    require(len(events) == len(spec.get("events", [])) and all(0 < i < spec["frames"] for i in events),
            "Camera events must have unique in-range frame numbers")
    require("cameraMotion" not in spec, "Expected explicit camera poses")
    camera = spec["camera"]
    cameras = []
    for frame in range(spec["frames"]):
        camera = events.get(frame, {}).get("camera", camera)
        cameras.append(camera)
    return cameras, [frame for frame, event in events.items() if event.get("cameraCut", False)]


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
            and (spec["width"], spec["height"]) == size
            and spec["output"] == "VBuffer.color" and spec["requireShadow"] and not spec["temporalJitter"],
            f"{case_id}: unexpected scene, extent or capture settings")
    vbuffer, = [node["properties"] for node in case["resolvedGraph"]["nodes"] if node["name"] == "VBuffer"]
    require(vbuffer["maxResidentPages"] == PAGE_CAP
            and vbuffer["autoLod"] and vbuffer["visualization"] == "lod" and not vbuffer["shadedDebugColors"],
            f"{case_id}: expected the 32-page limit and unshaded automatic LOD visualization")

    rows = case["frames"]
    require(rows and [row["frame"] for row in rows] == list(range(spec["frames"])),
            f"{case_id}: incomplete frame sequence")
    expected_stream = Path(str(SCENE) + ".meshstream.bin").resolve()
    streams = []
    for row in rows:
        frame = row["frame"]
        ready = row["readiness"]
        stream, = [entry for entry in row["streaming"] if entry["passName"] == "VBuffer"]
        require(Path(stream["assetPath"]).resolve() == expected_stream and stream["totalPages"] > PAGE_CAP,
                f"{case_id} frame {frame}: expected the cooked 4x4 stream asset")
        require(ready["ready"] and ready["completedPages"] == ready["requiredPages"] == LOCKED_PAGES,
                f"{case_id} frame {frame}: terminal fallback is incomplete")
        require(stream["maxResidentPages"] == PAGE_CAP
                and LOCKED_PAGES <= stream["residentPages"] <= stream["activePages"] <= PAGE_CAP,
                f"{case_id} frame {frame}: expected the 32-page limit")
        require(stream["loadFailures"] == stream["requestOverflows"] == 0
                and stream["allocationFailures"] == stream["allocationBudgetDeferrals"],
                f"{case_id} frame {frame}: unexpected stream failure")
        require((row["actualRenderWidth"], row["actualRenderHeight"]) == size
                and row["shadow"]["header"]["admittedCount"] > 0,
                f"{case_id} frame {frame}: requested render extent or VSM missing")
        streams.append(stream)

    captures = case["captures"]
    expected_frames = spec["captureFrames"]
    require(expected_frames == sorted(set(expected_frames))
            and all(0 <= frame < len(rows) for frame in expected_frames)
            and [capture["frame"] for capture in captures] == expected_frames,
            f"{case_id}: missing, unexpected, or duplicate capture")
    return spec, {capture["frame"]: capture for capture in captures}, streams


def capture_file(directory, name):
    require(Path(name).name == name, f"Expected a capture filename, found {name}")
    return directory / name


def raw_pixels(directory, capture, resource, size, dtype, format_name, layers, components=1):
    metadata = capture["rawImages"][resource]
    width, height = size
    bytes_per_pixel = np.dtype(dtype).itemsize * components
    expected = {
        "width": width, "height": height, "format": format_name, "layerCount": layers,
        "byteOrder": "little", "texelOrder": "layer,y,x,component", "bytesPerPixel": bytes_per_pixel,
        "rowStrideBytes": width * bytes_per_pixel, "layerStrideBytes": width * height * bytes_per_pixel,
        "byteLength": width * height * layers * bytes_per_pixel,
    }
    require(all(metadata[key] == value for key, value in expected.items()),
            f"Frame {capture['frame']}: incompatible {resource} capture")
    data = capture_file(directory, metadata["file"]).read_bytes()
    require(len(data) == expected["byteLength"] and sha256(data).hexdigest() == metadata["rawSha256"],
            f"Frame {capture['frame']}: incomplete or mismatched {resource} capture")
    return np.frombuffer(data, dtype=dtype).reshape(layers, height, width, components)


def lod_pixels(directory, capture, size):
    with Image.open(capture_file(directory, capture["file"])) as image:
        require(image.format == "PNG" and image.size == size,
                f"Frame {capture['frame']}: expected an exact-size PNG")
        rgba = np.asarray(image.convert("RGBA"))
    require(sha256(rgba.tobytes()).hexdigest() == capture["rawSha256"],
            f"Frame {capture['frame']}: LOD image does not match the GPU capture")
    return rgba[:, :, :3]


def decode_lods(rgb):
    # VisibilityBufferComposite's unshaded mode hashes lodLevel + 1 and
    # writes RGB8. Allow either rounding direction for exact half-byte values.
    palette = []
    for seed in range(1, 33):
        color = ((seed ^ (seed >> 16)) * 0x7feb352d) & 0xffffffff
        color = ((color ^ (color >> 15)) * 0x846ca68b) & 0xffffffff
        color ^= color >> 16
        palette.append([76.5 + 0.5 * ((color >> shift) & 255) for shift in (0, 8, 16)])
    palette = np.asarray(palette)
    lower, upper = np.ceil(palette - 0.5001), np.floor(palette + 0.5001)
    colors, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    match = ((colors[:, None, :] >= lower) & (colors[:, None, :] <= upper)).all(axis=-1)
    require(np.all(match.sum(axis=1) <= 1), "LOD palette has ambiguous RGB8 values")
    levels = np.where(match.any(axis=1), np.argmax(match, axis=1), -1)
    return levels[inverse].reshape(rgb.shape[:2])


def normalized(direction):
    length = np.linalg.norm(direction)
    require(length > 0, "Degenerate camera orientation")
    return direction / length


def capture_geometry(directory, capture, camera, size):
    depth = raw_pixels(directory, capture, "VBuffer.depth", size, "<f4", "D32Sfloat", 1)[0, :, :, 0]
    visibility = raw_pixels(directory, capture, "VBuffer.visibility", size, "<u4", "R32Uint", 1)[0, :, :, 0]
    require(np.isfinite(depth).all() and np.all((0 <= depth) & (depth <= 1)), "Invalid reversed-Z depth")
    covered = (depth > 0) & (visibility != 0)
    width, height = size
    eye = np.asarray(camera["eye"])
    forward = normalized(np.asarray(camera["center"]) - eye)
    right = normalized(np.cross(forward, camera["up"]))
    up = np.cross(right, forward)
    rows, cols = np.ogrid[:height, :width]
    x = (2 * (cols + 0.5) - width) / height
    y = 1 - 2 * (rows + 0.5) / height
    tangent = np.tan(np.radians(camera["fovDegrees"] / 2))
    rays = forward + tangent * (x[..., None] * right + y[..., None] * up)
    near, far = camera["znear"], camera["zfar"]
    distance = near * far / (near + depth.astype(np.float64) * (far - near))
    return eye + rays * distance[..., None], covered


def fixture_bounds():
    scene = read(SCENE)
    terrain = scene["extras"]["metallicTerrainTiles"]
    require(terrain["tilesPerAxis"] == 4, "Expected the 4x4 terrain fixture")
    accessors = scene["accessors"]
    mesh_by_name = {mesh["name"]: mesh for mesh in scene["meshes"]}
    result = {}
    for tile in terrain["tiles"]:
        primitive, = mesh_by_name[tile["name"]]["primitives"]
        accessor = accessors[primitive["attributes"]["POSITION"]]
        result[tuple(tile["tile"])] = np.asarray([accessor["min"], accessor["max"]], dtype=np.float64)
    require(set(result) == {(x, z) for z in range(4) for x in range(4)}, "Missing or repeated terrain tiles")
    terrain_names = {tile["name"] for tile in terrain["tiles"]}
    props = []
    for node in scene["nodes"]:
        if "mesh" not in node:
            continue
        mesh = scene["meshes"][node["mesh"]]
        if mesh["name"] in terrain_names:
            continue
        require("rotation" not in node and "matrix" not in node, "Expected axis-aligned sample props")
        primitive, = mesh["primitives"]
        accessor = accessors[primitive["attributes"]["POSITION"]]
        corners = np.asarray([accessor["min"], accessor["max"]]) * node["scale"] + node["translation"]
        props.append((corners.min(axis=0), corners.max(axis=0)))
    return result, props


def opposing_pairs(world, covered, levels, upper_tile, lower_tile):
    plane = lower_tile[0, 2]
    require(upper_tile[1, 2] == plane and np.all(upper_tile[:, 0] == lower_tile[:, 0]),
            "Tiles (0,2) and (0,3) do not share their authored Z boundary")
    x, z = world[..., 0], world[..., 2]
    along = (x > upper_tile[0, 0] + 0.01) & (x < upper_tile[1, 0] - 0.01)
    band = along & covered & (np.abs(z - plane) <= 3.)
    below = band & (z < plane - 0.01)
    above = band & (z > plane + 0.01)
    combinations = {}
    for first, second in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        across = (below[first] & above[second]) | (above[first] & below[second])
        across &= np.linalg.norm(world[first] - world[second], axis=-1) <= 4.
        from_below = below[first][across]
        a, b = levels[first][across], levels[second][across]
        labels = np.column_stack((np.where(from_below, a, b), np.where(from_below, b, a)))
        values, counts = np.unique(labels, axis=0, return_counts=True)
        for (a, b), count in zip(values, counts):
            pair = (int(a), int(b))
            combinations[pair] = combinations.get(pair, 0) + int(count)
    return [{"lodsByTile": list(pair), "pairs": count} for pair, count in sorted(combinations.items())]


def verify_crossing(directory, bounds, report):
    size = (640, 360)
    spec, captures, streams = capture_run(directory, "motion-directed-tight", size)
    cameras, cuts = camera_timeline(spec)
    require(not cuts and set(CROSSING_EYES).issubset(captures), "Missing continuous crossing captures")
    first, second = (bounds[tile] for tile in CROSSING_TILES)
    axis = [0, 2]
    for frame, camera in enumerate(cameras[:15]):
        tile = second if frame < CROSSING_FRAME else first
        eye = np.asarray(camera["eye"])[axis]
        require(np.all((tile[0, axis] < eye) & (eye < tile[1, axis])),
                f"Frame {frame}: unexpected tile before or after the first crossing")
    for frame, eye in CROSSING_EYES.items():
        require(np.allclose(cameras[frame]["eye"], eye, rtol=0, atol=1e-7)
                and all(cameras[frame][key] == value for key, value in CLIP_CAMERA.items()),
                f"Frame {frame}: original crossing eye or projection changed")

    at_crossing = streams[CROSSING_FRAME]
    plane = float(second[0, 2])
    before, after = (np.asarray(cameras[frame]["eye"]) for frame in (CROSSING_FRAME - 1, CROSSING_FRAME))
    crossing_point = before + (after - before) * ((plane - before[2]) / (after[2] - before[2]))
    result = {"tiles": CROSSING_TILES, "seamPlaneZ": plane, "crossingFrame": CROSSING_FRAME,
              "crossingPosition": crossing_point.tolist(),
              "checkedFrames": len(streams), "pageLimit": PAGE_CAP, "lockedFallbackPages": LOCKED_PAGES,
              "activePagesAtCrossing": at_crossing["activePages"], "residentPagesAtCrossing": at_crossing["residentPages"],
              "frames": []}
    report["crossing"] = result
    require(at_crossing["activePages"] == PAGE_CAP, "Frame 13: 32-page stream budget was not saturated")
    for frame in CROSSING_EYES:
        capture = captures[frame]
        world, covered = capture_geometry(directory, capture, cameras[frame], size)
        levels = decode_lods(lod_pixels(directory, capture, size))
        require(np.all(levels[covered] >= 0), f"Frame {frame}: covered pixels do not carry valid LOD colors")
        pairs = opposing_pairs(world, covered, levels, first, second)
        result["frames"].append({"frame": frame, "eye": cameras[frame]["eye"], "lodPairs": pairs,
                                 "opposingPairs": sum(item["pairs"] for item in pairs),
                                 "opposingMixedLODPairs": sum(item["pairs"] for item in pairs
                                                                if len(set(item["lodsByTile"])) == 2)})
    require(all(row["opposingPairs"] > 0 for row in result["frames"]), "The seam must be visible in frames 12, 13 and 14")
    require(result["frames"][1]["opposingMixedLODPairs"] > 0,
            "Frame 13: no adjacent rendered pixels across the tile seam have different LODs")


def verify_fixed(directory, bounds, prop_bounds, report):
    size = (960, 540)
    spec, captures, streams = capture_run(directory, "fixed-tight", size)
    cameras, cuts = camera_timeline(spec)
    expected_frames = [frame for _, first, last, *_ in FIXED_VIEWS for frame in (first, last)]
    require(len(streams) == 60 and cuts == [20, 40] and sorted(captures) == expected_frames,
            "Expected six fixed captures and the original three camera segments")
    terrain = np.asarray(list(bounds.values()))
    low, high = terrain[:, 0, :].min(axis=0), terrain[:, 1, :].max(axis=0)
    report["fixedViews"] = []
    for name, first, last, eye, center, fov in FIXED_VIEWS:
        pinned = {**CLIP_CAMERA, "eye": eye, "center": center, "fovDegrees": fov}
        require(all(camera == pinned for camera in cameras[first:last + 1]), f"Fixed {name}: original camera changed")
        view = {"name": name, "frames": []}
        report["fixedViews"].append(view)
        for frame in (first, last):
            capture = captures[frame]
            world, covered = capture_geometry(directory, capture, cameras[frame], size)
            terrain_visible = covered & np.all((low <= world) & (world <= high), axis=-1)
            for prop_low, prop_high in prop_bounds:
                terrain_visible &= ~np.all((prop_low - 0.01 <= world) & (world <= prop_high + 0.01), axis=-1)
            terrain_pixels = int(np.count_nonzero(terrain_visible))
            require(10 * terrain_pixels >= size[0] * size[1],
                    f"Fixed {name} frame {frame}: terrain covers less than 10% of the original viewport")
            lod_pixels(directory, capture, size)
            hdr = raw_pixels(directory, capture, "Deferred.color", size, "<f4", "RGBA32Sfloat", 1, 4)
            require(np.isfinite(hdr).all(), f"Fixed {name} frame {frame}: nonfinite Deferred color")
            shadow = raw_pixels(directory, capture, "Shadows.shadows", size, "u1", "R8Unorm", 8)[0, :, :, 0]
            values = shadow[terrain_visible]
            lit, shadowed = int(np.count_nonzero(values > 127)), int(np.count_nonzero(values < 128))
            require(lit > 0 and shadowed > 0, f"Fixed {name} frame {frame}: missing lit or shadowed VSM samples")
            view["frames"].append({"frame": frame, "terrainPixels": terrain_pixels,
                                   "litPixels": lit, "shadowedPixels": shadowed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directed", type=Path, required=True, help="Directed RHI capture directory")
    parser.add_argument("--fixed", type=Path, required=True, help="Fixed-camera RHI capture directory")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = {"protocol": "terrain-streaming-regression-v1", "passed": False}
    try:
        bounds, props = fixture_bounds()
        verify_crossing(args.directed, bounds, report)
        verify_fixed(args.fixed, bounds, props, report)
        report["passed"] = True
    except (OSError, KeyError, TypeError, ValueError) as error:
        report["error"] = str(error)
        print(f"Terrain streaming regression failed: {error}", file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save(args.report, report)
    if report["passed"]:
        print(f"Terrain streaming regression passed. Report: {args.report}")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
