"""Compare fixed Cornell Box probe rays against the scene's actual world triangles.

The manifests are CornellBoxGIBaseline captures with ray_capture readbacks.
Their probe positions select exact capture pixels. The oracle reconstructs the
shader's equal-area directions and biased world origins, then reports first
opaque triangle hits and the observed Screen, Detail, and GDF transitions.
Pass --c for a Screen-to-GDF control with Mesh SDF Detail disabled.
Pass the witness fixture or a previous report to --pin to reuse its pixels,
lanes, and slots.
Pass --floor with --a and --c to compare the fixed floor witnesses. Add
--c-fixed when a capture of the revised Screen-to-GDF path is available.
Pass --d with --c-fixed to compare the GDF prefix-check variant.
Add --e to compare the receding prefix sample check against D.
Add --f for the convex-corner and near-surface prefix witnesses.
"""

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys


PROBE_TILE_EDGE = 16
PROBE_SLOTS = 9
RAY_EDGE = 8
RAY_COUNT = RAY_EDGE * RAY_EDGE
INVALID_PROBE = 0xFFFFFFFF
SURFACE_BIAS = 0.05
MIN_TRACE_DISTANCE = 0.0001
HANDOFF_TOLERANCE = 1.0e-6
DETAIL_TRIANGLE_TOLERANCE = 0.005
GDF_MISS = 1
GDF_HIT = 2
EMITTER_RGB_TOLERANCE = 0.01
GDF_PREFIX_INVOKED = 1 << 8
GDF_PREFIX_ACCEPTED = 1 << 9
GDF_PREFIX_CLASS_SHIFT = 10
GDF_PREFIX_CLASS_MASK = 0xF
GDF_PREFIX_RECEDING_REJECTED = 1 << 14
GDF_PREFIX_MESH_INVOKED = 1 << 15
GDF_PREFIX_MESH_CONFIRMED = 1 << 16
GDF_PREFIX_MESH_INCONCLUSIVE = 1 << 17
FLOOR_WITNESSES = {
    "emissive_edge": ((222, 366), (13, 22, 0), (2, 7)),
    "short_block": ((190, 318), (11, 19, 0), (3, 4)),
    "tall_block": ((190, 318), (11, 19, 0), (7, 6)),
    "open_front": ((78, 334), (4, 20, 0), (4, 4)),
}
CORNER_WITNESSES = {
    "short_corner_upper": ((206, 302), (12, 18, 0), (3, 4)),
    "short_corner_lower": ((206, 318), (12, 19, 0), (3, 4)),
    "near_floor": ((94, 350), (5, 21, 0), (5, 3)),
    "near_left_wall": ((78, 318), (4, 19, 0), (2, 7)),
    "later_ceiling": ((286, 318), (17, 19, 0), (3, 6)),
}
TRACE_RECORD = struct.Struct("<8f4I")
PROBE_POSITION = struct.Struct("<2I")
FLOAT4 = struct.Struct("<4f")
HALF4 = struct.Struct("<4e")
FLOAT = struct.Struct("<f")
RECEIVER_VALID, SCREEN_INVOKED, SCREEN_ACCEPTED, DETAIL_INVOKED, DETAIL_ACCEPTED, GDF_INVOKED = (
    1 << bit for bit in range(6)
)
ROOT = Path(__file__).resolve().parents[1]


def capture_gdf_voxel_size(manifest):
    # Historical Cornell captures used one 3.6 m clipmap with 112 voxels per axis.
    size = manifest.get("gdf_voxel_size_m", 3.6 / 112)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError("Capture GDF voxel size must be positive and finite")
    return size


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def multiply(a, scale):
    return tuple(x * scale for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    return multiply(v, 1.0 / length(v))


def sign(value):
    return (value > 0) - (value < 0)


def probe_ray_direction(lane):
    # probeRayDirection and lumenEquiAreaSphericalMapping. Texel centers are
    # used by both the trace and the fixed-direction numerical integrator.
    x, y = tuple(2.0 * (coordinate + 0.5) / RAY_EDGE - 1.0 for coordinate in lane)
    d = 1.0 - (abs(x) + abs(y))
    r = 1.0 - abs(d)
    phi = 0.0 if r == 0.0 else (math.pi / 4.0) * ((abs(y) - abs(x)) / r + 1.0)
    f = r * math.sqrt(2.0 - r * r)
    return (f * sign(x) * abs(math.cos(phi)), f * sign(y) * abs(math.sin(phi)), sign(d) * (1.0 - r * r))


def transform_position(position, node):
    scale = node.get("scale", (1, 1, 1))
    xyz_w = node.get("rotation", (0, 0, 0, 1))
    translation = node.get("translation", (0, 0, 0))
    v = tuple(component * factor for component, factor in zip(position, scale))
    q = xyz_w[:3]
    uv = multiply(cross(q, v), 2.0)
    return add(add(add(v, multiply(uv, xyz_w[3])), cross(q, uv)), translation)


@dataclass(frozen=True)
class Triangle:
    origin: tuple
    ab: tuple
    ac: tuple
    normal: tuple
    altitudes: tuple
    node_index: int
    mesh_index: int
    primitive_index: int
    triangle_index: int
    object_name: str
    material: str
    emissive: bool


def read_triangles(path):
    asset = json.loads(path.read_text(encoding="utf-8"))
    buffers = []
    for buffer in asset["buffers"]:
        _, data = buffer["uri"].split(",", 1)
        buffers.append(base64.b64decode(data, validate=True))

    def read_accessor(index, kind, layouts):
        accessor = asset["accessors"][index]
        if accessor["type"] != kind or "sparse" in accessor or accessor.get("normalized", False):
            raise ValueError(f"Unsupported Cornell Box accessor {index}")
        layout = struct.Struct(layouts[accessor["componentType"]])
        view = asset["bufferViews"][accessor["bufferView"]]
        data = buffers[view["buffer"]]
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", layout.size)
        return [layout.unpack_from(data, offset + i * stride) for i in range(accessor["count"])]

    triangles = []
    for node_index in asset["scenes"][asset["scene"]]["nodes"]:
        node = asset["nodes"][node_index]
        if node.get("children") or "matrix" in node:
            raise ValueError("Cornell Box oracle requires the scene's flat TRS nodes")
        if "mesh" not in node:
            continue
        mesh_index = node["mesh"]
        for primitive_index, primitive in enumerate(asset["meshes"][mesh_index]["primitives"]):
            material = asset["materials"][primitive["material"]]
            if primitive.get("mode", 4) != 4 or material.get("alphaMode", "OPAQUE") != "OPAQUE":
                raise ValueError("Cornell Box oracle requires opaque indexed triangles")
            points = read_accessor(primitive["attributes"]["POSITION"], "VEC3", {5126: "<3f"})
            points = [transform_position(point, node) for point in points]
            indices = read_accessor(primitive["indices"], "SCALAR", {5121: "<B", 5123: "<H", 5125: "<I"})
            for offset in range(0, len(indices), 3):
                a, b, c = (points[index] for (index,) in indices[offset:offset + 3])
                ab, ac = subtract(b, a), subtract(c, a)
                area_vector = cross(ab, ac)
                double_area = length(area_vector)
                altitudes = tuple(double_area / length(edge) for edge in (subtract(c, b), ac, ab))
                emissive = any(material.get("emissiveFactor", (0, 0, 0))) and material.get(
                    "extensions", {}).get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1) > 0
                triangles.append(Triangle(a, ab, ac, multiply(area_vector, 1.0 / double_area), altitudes,
                    node_index, mesh_index, primitive_index, offset // 3, node["name"], material["name"], emissive))
    return triangles


def first_hit(triangles, origin, direction, near, far):
    closest = far
    hit = None
    for triangle in triangles:
        p = cross(direction, triangle.ac)
        determinant = dot(triangle.ab, p)
        if abs(determinant) < 1.0e-12:
            continue
        tx = subtract(origin, triangle.origin)
        u = dot(tx, p) / determinant
        if u < 0.0 or u > 1.0:
            continue
        q = cross(tx, triangle.ab)
        v = dot(direction, q) / determinant
        if v < 0.0 or u + v > 1.0:
            continue
        distance = dot(triangle.ac, q) / determinant
        if not near <= distance < closest:
            continue

        closest = distance
        weights = (1.0 - u - v, u, v)
        hit = {
            "object": triangle.object_name, "node_index": triangle.node_index, "mesh_index": triangle.mesh_index,
            "primitive_index": triangle.primitive_index, "triangle_index": triangle.triangle_index,
            "material": triangle.material, "emissive": triangle.emissive, "distance": distance,
            "position": add(origin, multiply(direction, distance)), "normal": triangle.normal,
            "barycentrics": weights,
            "edge_clearance": min(weight * height for weight, height in zip(weights, triangle.altitudes)),
            "front_face": determinant > 0.0, "incidence": abs(dot(triangle.normal, direction)),
        }
    return hit


def camera_axes(camera):
    forward = normalize(subtract(camera["center"], camera["eye"]))
    right = normalize(cross(forward, camera["up"]))
    up = cross(right, forward)
    return forward, right, up


def closest_segment_point(point, start, end):
    edge = subtract(end, start)
    t = max(0.0, min(1.0, dot(subtract(point, start), edge) / dot(edge, edge)))
    return add(start, multiply(edge, t))


def closest_triangle_point(point, triangle):
    a, ab, ac, normal = triangle.origin, triangle.ab, triangle.ac, triangle.normal
    b, c = add(a, ab), add(a, ac)
    ap = subtract(point, a)
    projected = subtract(point, multiply(normal, dot(ap, normal)))
    if all(dot(cross(subtract(end, start), subtract(projected, start)), normal) >= 0.0
            for start, end in ((a, b), (b, c), (c, a))):
        return projected
    return min((closest_segment_point(point, start, end) for start, end in ((a, b), (b, c), (c, a))),
        key=lambda position: length(subtract(point, position)))


def nearest_surfaces(point, direction, triangles):
    surfaces = {}
    for triangle in triangles:
        position = closest_triangle_point(point, triangle)
        delta = subtract(point, position)
        distance = length(delta)
        name = triangle.object_name
        if name not in surfaces or distance < surfaces[name]["distance"]:
            surfaces[name] = {"distance": distance, "position": position, "normal": triangle.normal,
                "signed_distance_along_normal": dot(delta, triangle.normal),
                "normal_dot_direction": dot(triangle.normal, direction), "emissive": triangle.emissive,
                "node_index": triangle.node_index, "primitive_index": triangle.primitive_index,
                "triangle_index": triangle.triangle_index}
    return dict(sorted(surfaces.items(), key=lambda entry: entry[1]["distance"]))


def short_block_corner(witness, triangles):
    short = [triangle for triangle in triangles if triangle.object_name == "Short Block"]
    # The two outward +X/+Z faces share the front-right vertical edge.
    right = max(short, key=lambda triangle: triangle.normal[0])
    front = max(short, key=lambda triangle: triangle.normal[2])
    origin, direction = witness["gdf_origin"], witness["direction"]
    offsets = [dot(subtract(origin, face.origin), face.normal) for face in (right, front)]
    slopes = [dot(direction, face.normal) for face in (right, front)]
    closest_time = -dot(offsets, slopes) / dot(slopes, slopes)
    point = add(origin, multiply(direction, closest_time))
    enter_right = -offsets[0] / slopes[0]
    exit_front = -offsets[1] / slopes[1]
    return {
        "front_plane_exit_m": exit_front, "right_plane_enter_m": enter_right,
        "outside_right_at_front_exit_m": offsets[0] + slopes[0] * exit_front,
        "outside_front_at_right_enter_m": offsets[1] + slopes[1] * enter_right,
        "closest_approach_m": closest_time, "closest_ray_point": point,
        "closest_short_block_surface": nearest_surfaces(point, direction, short)["Short Block"],
    }


def clip_box_planes(origin, direction, interval, planes):
    near, far = interval
    for normal, plane_offset in planes:
        distance = plane_offset - dot(normal, origin)
        slope = dot(normal, direction)
        if abs(slope) <= 1.0e-12:
            if distance < 0.0:
                return None
            continue
        crossing = distance / slope
        if slope > 0.0:
            far = min(far, crossing)
        else:
            near = max(near, crossing)
        if near > far:
            return None
    return near, far


def primary_hit(triangles, pixel, resolution, camera):
    forward, right, up = camera_axes(camera)
    width, height = resolution
    scale = math.tan(math.radians(camera["fovDegrees"]) / 2.0)
    horizontal = (2.0 * (pixel[0] + 0.5) / width - 1.0) * scale * width / height
    vertical = (1.0 - 2.0 * (pixel[1] + 0.5) / height) * scale
    direction = add(add(forward, multiply(right, horizontal)), multiply(up, vertical))
    return first_hit(triangles, camera["eye"], direction, camera["znear"], camera["zfar"])


def read_buffer(directory, file, layout, count):
    contents = (directory / file).read_bytes()
    if len(contents) != layout.size * count:
        raise ValueError(f"Unexpected {file} size. Expected {layout.size * count}, got {len(contents)}")
    return contents


def read_capture(path):
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    info = manifest["ray_capture"]
    width, height = manifest["resolution"]
    probe_width = (width + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE
    probe_height = (height + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE
    probe_count = probe_width * probe_height * PROBE_SLOTS
    tx, ty, tw, th = info["tiles"]
    rx, ry, rw, rh = info["pixel_rect"]
    if not (0 <= tx < tx + tw <= probe_width and 0 <= ty < ty + th <= probe_height and
            0 <= rx < rx + rw <= width and 0 <= ry < ry + rh <= height):
        raise ValueError("Probe tile or guide readback rectangle is outside the image")
    directory = path.parent
    positions = read_buffer(directory, info["probe_positions"], PROBE_POSITION, probe_count)
    trace = read_buffer(directory, info["trace"], TRACE_RECORD, tw * th * PROBE_SLOTS * RAY_COUNT)
    radiance = read_buffer(directory, info["radiance"], FLOAT4, probe_count * RAY_COUNT)
    distance = read_buffer(directory, info["distance"], FLOAT, probe_count * RAY_COUNT)
    guide_position = read_buffer(directory, info["position"], FLOAT4, rw * rh)
    guide_normal = read_buffer(directory, info["normal"], HALF4, rw * rh)

    probes = defaultdict(dict)
    counts = Counter()
    for tile_y in range(ty, ty + th):
        for tile_x in range(tx, tx + tw):
            for slot in range(PROBE_SLOTS):
                index = (tile_y * probe_width + tile_x) * PROBE_SLOTS + slot
                local_index = ((tile_y - ty) * tw + tile_x - tx) * PROBE_SLOTS + slot
                px, py = PROBE_POSITION.unpack_from(positions, index * PROBE_POSITION.size)
                if px == INVALID_PROBE:
                    counts["invalid_probe_slots"] += 1
                    continue
                if not (rx <= px < rx + rw and ry <= py < ry + rh):
                    counts["probe_slots_outside_guide_crop"] += 1
                    continue
                offset = (py - ry) * rw + px - rx
                receiver = FLOAT4.unpack_from(guide_position, offset * FLOAT4.size)
                normal = HALF4.unpack_from(guide_normal, offset * HALF4.size)[:3]
                if not all(map(math.isfinite, (*receiver, *normal))) or receiver[3] <= 0.0 or length(normal) == 0.0:
                    counts["invalid_receiver_probe_slots"] += 1
                    continue
                counts["valid_probe_slots"] += 1
                probes[(px, py)][(tile_x, tile_y, slot)] = (
                    receiver, normalize(normal), index * RAY_COUNT, local_index * RAY_COUNT)
    return manifest, probes, counts, (trace, radiance, distance)


def captured_ray(buffers, trace_index, ray_index):
    trace, radiance, distance = buffers
    values = TRACE_RECORD.unpack_from(trace, trace_index * TRACE_RECORD.size)
    return {
        "screen_end": values[0], "detail_end": values[1], "gdf_min": values[2], "gdf_max": values[3],
        "gdf_hit_distance": values[7], "gdf_classification": values[8], "gdf_hit_clipmap": values[9],
        "gdf_steps": values[10], "flags": values[11],
        "radiance_and_class": FLOAT4.unpack_from(radiance, ray_index * FLOAT4.size),
        "trace_distance": FLOAT.unpack_from(distance, ray_index * FLOAT.size)[0],
    }


def classify(ray):
    both_screen_rejected = all(
        capture["flags"] & SCREEN_INVOKED and not capture["flags"] & SCREEN_ACCEPTED
        for capture in (ray["a"], ray["b"]))
    b_flags = ray["b"]["flags"]
    if not (both_screen_rejected and ray["a"]["flags"] & GDF_INVOKED and b_flags & DETAIL_INVOKED):
        return None
    if b_flags & DETAIL_ACCEPTED and not b_flags & GDF_INVOKED:
        return "detail_accepted"
    if b_flags & GDF_INVOKED and not b_flags & DETAIL_ACCEPTED:
        return "gdf_continuation"
    return None


def count_trace_transitions(counts, a, b):
    a_flags, b_flags = a["flags"], b["flags"]
    counts["a_screen_accepted"] += bool(a_flags & SCREEN_ACCEPTED)
    counts["b_screen_accepted"] += bool(b_flags & SCREEN_ACCEPTED)
    counts["b_detail_invoked"] += bool(b_flags & DETAIL_INVOKED)
    counts["b_detail_accepted"] += bool(b_flags & DETAIL_ACCEPTED)
    if a_flags & b_flags & SCREEN_ACCEPTED:
        counts["both_screen_accepted"] += 1
        counts["accepted_screen_endpoint_matches"] += (a["screen_end"] == b["screen_end"] and
            a["trace_distance"] == b["trace_distance"])
        counts["accepted_screen_radiance_changed"] += a["radiance_and_class"][:3] != b["radiance_and_class"][:3]
    if a_flags & b_flags & GDF_INVOKED:
        used_detail = bool(b_flags & DETAIL_INVOKED)
        stage = "detail" if used_detail else "screen"
        expected = max(MIN_TRACE_DISTANCE, b[f"{stage}_end"] - 2.0 * SURFACE_BIAS)
        counts[f"gdf_after_{stage}"] += 1
        counts[f"gdf_after_{stage}_endpoint_matches"] += abs(b["gdf_min"] - expected) <= HANDOFF_TOLERANCE
        counts[f"gdf_after_{stage}_start_changed"] += a["gdf_min"] != b["gdf_min"]


def count_screen_control(trace):
    counts = Counter()
    for record in TRACE_RECORD.iter_unpack(trace):
        flags = record[11]
        counts["detail_invoked"] += bool(flags & DETAIL_INVOKED)
        counts["detail_accepted"] += bool(flags & DETAIL_ACCEPTED)
        if not flags & RECEIVER_VALID:
            continue
        counts["valid_rays"] += 1
        counts["screen_invoked"] += bool(flags & SCREEN_INVOKED)
        if flags & SCREEN_ACCEPTED:
            counts["screen_accepted"] += 1
            continue
        counts["screen_rejected"] += 1
        counts["gdf_after_screen"] += bool(flags & GDF_INVOKED)
        expected_start = max(MIN_TRACE_DISTANCE, record[0] - 2.0 * SURFACE_BIAS)
        counts["gdf_after_screen_endpoint_matches"] += (bool(flags & GDF_INVOKED) and
            abs(record[2] - expected_start) <= HANDOFF_TOLERANCE)
    return counts


def count_control_difference(counts, lhs, rhs):
    counts["compared_rays"] += 1
    counts["screen_acceptance_changed"] += bool((lhs["flags"] ^ rhs["flags"]) & SCREEN_ACCEPTED)
    counts["screen_end_changed"] += lhs["screen_end"] != rhs["screen_end"]
    counts["trace_class_changed"] += lhs["radiance_and_class"][3] != rhs["radiance_and_class"][3]
    counts["trace_rgb_changed"] += lhs["radiance_and_class"][:3] != rhs["radiance_and_class"][:3]
    counts["trace_distance_changed"] += lhs["trace_distance"] != rhs["trace_distance"]
    if lhs["flags"] & rhs["flags"] & GDF_INVOKED:
        counts["paired_gdf_rays"] += 1
        counts["gdf_start_changed"] += lhs["gdf_min"] != rhs["gdf_min"]
        counts["gdf_start_matches_within_tolerance"] += abs(lhs["gdf_min"] - rhs["gdf_min"]) <= HANDOFF_TOLERANCE
        counts["gdf_classification_changed"] += lhs["gdf_classification"] != rhs["gdf_classification"]
        counts["gdf_hit_distance_changed"] += lhs["gdf_hit_distance"] != rhs["gdf_hit_distance"]
        counts["gdf_hit_clipmap_changed"] += lhs["gdf_hit_clipmap"] != rhs["gdf_hit_clipmap"]
        counts["gdf_steps_changed"] += lhs["gdf_steps"] != rhs["gdf_steps"]


def skipped_detail_and_used_gdf(a, b):
    return a["flags"] & b["flags"] & GDF_INVOKED and not b["flags"] & DETAIL_INVOKED


def count_control_ray(groups, a, b, c, case):
    b_flags = b["flags"]
    group = "other"
    if b_flags & SCREEN_ACCEPTED:
        group = "b_screen_accepted"
    elif case:
        group = f"b_{case}"
    elif skipped_detail_and_used_gdf(a, b):
        group = "b_skipped_detail_and_used_gdf"
    for name in ("all", group):
        count_control_difference(groups[name]["a_to_c"], a, c)
        count_control_difference(groups[name]["c_to_b"], c, b)


def rank_positive_screen_handoff(ray):
    hit, c = ray["first_hit"], ray["c"]
    triangulated_gdf_hit = (hit["distance"] >= c["gdf_min"] and c["gdf_classification"] == 2 and
        abs(c["gdf_hit_distance"] - hit["distance"]) < 0.04)
    return (triangulated_gdf_hit, ray["receiver_object"] == "Tall Block", hit["edge_clearance"], hit["incidence"])


def key(ray):
    return (*ray["pixel"], *ray["tile"], ray["slot"], *ray["lane"])


def check_capture_alignment(a, b, name):
    if not math.isclose(capture_gdf_voxel_size(a), capture_gdf_voxel_size(b)):
        raise ValueError(f"A and {name} must use the same GDF voxel size")
    if a["resolution"] != b["resolution"] or a["view"] != b["view"] or any(
            a["ray_capture"][field] != b["ray_capture"][field] for field in ("frame", "tiles", "pixel_rect")):
        raise ValueError(f"A and {name} must use the same camera, resolution, execution frame and capture rectangles")


def floor_screen_origin(receiver, normal, manifest):
    camera = manifest["view"]["camera"]
    forward, right, up = camera_axes(camera)
    depth = dot(subtract(receiver, camera["eye"]), forward)
    # A half-pixel step down and right at fixed perspective view depth.
    corner_offset = multiply(subtract(right, up),
        math.tan(math.radians(camera["fovDegrees"]) / 2.0) * depth / manifest["resolution"][1])
    bias = abs(dot(corner_offset, normal)) * 2.0
    return add(receiver, multiply(normal, bias))


def floor_screen_pixel(position, manifest):
    camera = manifest["view"]["camera"]
    forward, right, up = camera_axes(camera)
    relative = subtract(position, camera["eye"])
    depth = dot(relative, forward)
    width, height = manifest["resolution"]
    scale = math.tan(math.radians(camera["fovDegrees"]) / 2.0) * depth
    return ((width + height * dot(relative, right) / scale) * 0.5 - 0.5,
        height * (1.0 - dot(relative, up) / scale) * 0.5 - 0.5)


def floor_ray_result(capture, gdf_origin, direction, hit):
    gdf_hit = capture["gdf_classification"] == GDF_HIT
    entry = capture["gdf_min"] - hit["distance"] if hit and capture["flags"] & GDF_INVOKED else None
    capture["gdf_start_minus_triangle"] = entry
    capture["gdf_start_past_triangle"] = entry is not None and entry > HANDOFF_TOLERANCE
    capture["gdf_hit_position"] = add(gdf_origin, multiply(direction, capture["gdf_hit_distance"])) if gdf_hit else None
    capture["trace_luminance"] = dot(capture["radiance_and_class"][:3], (0.2126, 0.7152, 0.0722))
    return capture


def prefix_state(flags):
    invoked = bool(flags & GDF_PREFIX_INVOKED)
    return {
        "invoked": invoked,
        "accepted": bool(flags & GDF_PREFIX_ACCEPTED),
        "classification": (flags >> GDF_PREFIX_CLASS_SHIFT) & GDF_PREFIX_CLASS_MASK if invoked else None,
    }


def mesh_validation_state(flags):
    return {"invoked": bool(flags & GDF_PREFIX_MESH_INVOKED),
        "confirmed": bool(flags & GDF_PREFIX_MESH_CONFIRMED),
        "inconclusive": bool(flags & GDF_PREFIX_MESH_INCONCLUSIVE)}


def readback_address(record_index, tiles, probe_pixels):
    probe, ray = divmod(record_index, RAY_COUNT)
    tile_index, slot = divmod(probe, PROBE_SLOTS)
    local_y, local_x = divmod(tile_index, tiles[2])
    tile = (tiles[0] + local_x, tiles[1] + local_y)
    return {"pixel": probe_pixels.get((*tile, slot)), "tile": tile, "slot": slot,
        "lane": (ray % RAY_EDGE, ray // RAY_EDGE)}


def count_prefix_control(c_trace, d_trace, tiles, probe_pixels, voxel_size, mesh_validation=False):
    counts = Counter({name: 0 for name in (
        "prefix_invoked", "prefix_accepted", "prefix_rejected", "expected_prefix_invocation_mismatches",
        "selected_prefix_interval_mismatches", "selected_tail_mismatches", "prefix_outcome_mismatches",
        "upstream_trace_mismatches",
        "accepted_with_c_tail_hit", "accepted_with_c_tail_hit_before_cutoff",
        "accepted_with_c_tail_hit_within_one_voxel", "accepted_without_c_tail_hit",
    )})
    classifications = Counter()
    tail_distance_differences = []
    pairs = zip(TRACE_RECORD.iter_unpack(c_trace), TRACE_RECORD.iter_unpack(d_trace), strict=True)
    for index, (c, d) in enumerate(pairs):
        c_flags, d_flags = c[11], d[11]
        counts["readback_rays"] += 1
        counts["valid_receivers"] += bool(c_flags & RECEIVER_VALID)
        prefix_max = min(c[2], c[3])
        expected_invocation = bool(c_flags & GDF_INVOKED) and prefix_max > MIN_TRACE_DISTANCE
        prefix = prefix_state(d_flags)
        counts["expected_prefix_invocation_mismatches"] += prefix["invoked"] != expected_invocation
        counts["upstream_trace_mismatches"] += (c_flags != (d_flags & 0xFF) or
            abs(c[0] - d[0]) > HANDOFF_TOLERANCE)
        expected_acceptance = prefix["classification"] == GDF_HIT
        if mesh_validation:
            expected_acceptance &= bool(d_flags & (GDF_PREFIX_MESH_CONFIRMED | GDF_PREFIX_MESH_INCONCLUSIVE))
        counts["prefix_outcome_mismatches"] += prefix["accepted"] != expected_acceptance
        if prefix["invoked"]:
            counts["prefix_invoked"] += 1
            classifications[prefix["classification"]] += 1
        if prefix["accepted"]:
            counts["prefix_accepted"] += 1
            counts["selected_prefix_interval_mismatches"] += (abs(d[2] - MIN_TRACE_DISTANCE) > HANDOFF_TOLERANCE or
                abs(d[3] - prefix_max) > HANDOFF_TOLERANCE or d[8] != GDF_HIT)
            if c[8] == GDF_HIT:
                counts["accepted_with_c_tail_hit"] += 1
                counts["accepted_with_c_tail_hit_before_cutoff"] += c[7] < prefix_max
                counts["accepted_with_c_tail_hit_within_one_voxel"] += abs(c[7] - d[7]) <= voxel_size
            else:
                counts["accepted_without_c_tail_hit"] += 1
        else:
            counts["prefix_rejected"] += prefix["invoked"]
            counts["selected_tail_mismatches"] += (d[8:11] != c[8:11] or
                any(abs(d[field] - c[field]) > HANDOFF_TOLERANCE for field in (2, 3)))
            distance_delta = d[7] - c[7]
            if abs(distance_delta) > HANDOFF_TOLERANCE:
                tail_distance_differences.append({**readback_address(index, tiles, probe_pixels),
                    "c_fixed_hit_m": c[7], "d_hit_m": d[7], "delta_m": distance_delta})
    return {"counts": counts, "prefix_classifications": classifications,
        "tail_hit_distance_differences_above_handoff_tolerance": tail_distance_differences}


def check_prefix_witnesses(witnesses, checks, voxel_size):
    for name, ray in witnesses.items():
        if ray:
            c, d = ray["c_fixed"], ray["d"]
            d["prefix"] = prefix_state(d["flags"])
            d["prefix_cutoff"] = min(c["gdf_min"], c["gdf_max"])
            d["luminance_delta_vs_c_fixed"] = d["trace_luminance"] - c["trace_luminance"]
            checks[f"{name}_valid_d_receiver"] = bool(d["flags"] & RECEIVER_VALID)
    emitter = witnesses["emissive_edge"]
    if emitter:
        hit, c, d = emitter["gdf_triangle_first_hit"], emitter["c_fixed"], emitter["d"]
        checks["emitter_c_fixed_starts_past_emissive_triangle"] = bool(hit and hit["emissive"] and
            c["gdf_start_past_triangle"])
        checks["emitter_d_accepts_prefix_near_triangle"] = bool(hit and d["prefix"]["accepted"] and
            d["gdf_classification"] == GDF_HIT and
            abs(d["gdf_hit_distance"] - hit["distance"]) <= voxel_size and
            d["gdf_min"] <= d["gdf_hit_distance"] < d["gdf_max"])
        checks["emitter_d_writes_emissive_hit"] = (d["radiance_and_class"][3] == GDF_HIT and
            d["trace_distance"] == d["gdf_hit_distance"] and
            all(abs(component - 8.0) <= EMITTER_RGB_TOLERANCE for component in d["radiance_and_class"][:3]))
    for name in ("short_block", "tall_block"):
        if witnesses[name]:
            c, d = (witnesses[name][arm] for arm in ("c_fixed", "d"))
            checks[f"{name}_d_prefix_misses_and_keeps_tail"] = (d["prefix"]["classification"] == GDF_MISS and
                not d["prefix"]["accepted"] and all(d[field] == c[field] for field in (
                    "gdf_min", "gdf_max", "gdf_hit_distance", "gdf_steps", "radiance_and_class", "trace_distance")))
    open_front = witnesses["open_front"]
    if open_front:
        a, c, d = (open_front[arm] for arm in ("a", "c_fixed", "d"))
        checks["open_front_d_reintroduces_a_hit_without_triangle"] = (
            open_front["gdf_triangle_first_hit"] is None and c["gdf_classification"] != GDF_HIT and
            d["prefix"]["accepted"] and d["gdf_classification"] == GDF_HIT and
            d["gdf_hit_distance"] == a["gdf_hit_distance"] and d["radiance_and_class"] == a["radiance_and_class"])


def count_receding_rejections(d_trace, e_trace, tiles, probe_pixels):
    counts = Counter({name: 0 for name in (
        "receding_rejected", "receding_rejected_outside_prefix", "receding_rejected_then_prefix_hit",
        "receding_rejected_then_tail", "d_accepted_e_uses_tail", "d_used_tail_e_accepts_prefix",
    )})
    changes = []
    records = zip(TRACE_RECORD.iter_unpack(d_trace), TRACE_RECORD.iter_unpack(e_trace), strict=True)
    for index, (d, e) in enumerate(records):
        d_state, e_state = prefix_state(d[11]), prefix_state(e[11])
        receding_rejected = bool(e[11] & GDF_PREFIX_RECEDING_REJECTED)
        if receding_rejected:
            counts["receding_rejected"] += 1
            counts["receding_rejected_outside_prefix"] += not e_state["invoked"]
            counts["receding_rejected_then_prefix_hit"] += e_state["accepted"]
            counts["receding_rejected_then_tail"] += not e_state["accepted"]
        counts["d_accepted_e_uses_tail"] += d_state["accepted"] and not e_state["accepted"]
        counts["d_used_tail_e_accepts_prefix"] += not d_state["accepted"] and e_state["accepted"]
        if receding_rejected or d_state["accepted"] != e_state["accepted"]:
            changes.append({**readback_address(index, tiles, probe_pixels), "d_prefix": d_state, "e_prefix": e_state,
                "receding_sample_rejected": receding_rejected, "d_gdf_hit_m": d[7], "e_gdf_hit_m": e[7]})
    return {"counts": counts, "rays": changes}


def same_selected_trace(a, b):
    return (all(a[field] == b[field] for field in ("gdf_classification", "gdf_hit_clipmap", "gdf_steps")) and
        all(abs(a[field] - b[field]) <= HANDOFF_TOLERANCE for field in (
            "gdf_min", "gdf_max", "gdf_hit_distance", "trace_distance")) and
        all(abs(lhs - rhs) <= HANDOFF_TOLERANCE for lhs, rhs in zip(a["radiance_and_class"], b["radiance_and_class"])))


def compare_prefix_triangles(probes, d_buffers, e_buffers, triangles, trace_limit, voxel_size):
    counts = Counter()
    rows = []
    for pixel, slots in sorted(probes.items()):
        for slot, (receiver, normal, ray_base, trace_base) in sorted(slots.items()):
            for index in range(RAY_COUNT):
                d = captured_ray(d_buffers, trace_base + index, ray_base + index)
                if not d["flags"] & GDF_PREFIX_ACCEPTED:
                    continue
                counts["d_accepted"] += 1
                lane = (index % RAY_EDGE, index // RAY_EDGE)
                direction = probe_ray_direction(lane)
                origin = add(add(receiver[:3], multiply(direction, SURFACE_BIAS)), multiply(normal, SURFACE_BIAS))
                hit = first_hit(triangles, origin, direction, MIN_TRACE_DISTANCE, trace_limit)
                if hit is None:
                    counts["d_accepted_without_cpu_triangle"] += 1
                    continue
                if not d["gdf_min"] <= hit["distance"] < d["gdf_max"]:
                    counts["d_accepted_with_cpu_triangle_outside_prefix"] += 1
                    continue

                e = captured_ray(e_buffers, trace_base + index, ray_base + index)
                error = abs(d["gdf_hit_distance"] - hit["distance"])
                e_keeps_prefix = bool(e["flags"] & GDF_PREFIX_ACCEPTED)
                counts["d_accepted_with_cpu_triangle_in_prefix"] += 1
                counts["d_hit_within_one_voxel_of_triangle"] += error <= voxel_size
                counts["e_keeps_prefix_with_cpu_triangle_in_prefix"] += e_keeps_prefix
                rows.append({"pixel": pixel, "tile": slot[:2], "slot": slot[2], "lane": lane,
                    "triangle": hit, "d_hit_m": d["gdf_hit_distance"], "e_hit_m": e["gdf_hit_distance"],
                    "d_triangle_distance_error_m": error, "e_keeps_prefix": e_keeps_prefix,
                    "e_keeps_exact_distance_and_rgb": e["gdf_hit_distance"] == d["gdf_hit_distance"] and
                    e["radiance_and_class"] == d["radiance_and_class"]})
    return {"counts": counts, "rays_with_cpu_triangle_in_prefix": rows}


def check_receding_witnesses(witnesses, checks, triangles):
    for name, ray in witnesses.items():
        if ray:
            d, e = ray["d"], ray["e"]
            e["prefix"] = prefix_state(e["flags"])
            e["prefix"]["receding_sample_rejected"] = bool(e["flags"] & GDF_PREFIX_RECEDING_REJECTED)
            e["luminance_delta_vs_d"] = e["trace_luminance"] - d["trace_luminance"]
            checks[f"{name}_valid_e_receiver"] = bool(e["flags"] & RECEIVER_VALID)
            if name in ("emissive_edge", "open_front") and d["gdf_hit_position"] is not None:
                ray["surfaces_at_gdf_origin"] = nearest_surfaces(ray["gdf_origin"], ray["direction"], triangles)
                ray["surfaces_at_d_prefix_hit"] = nearest_surfaces(
                    d["gdf_hit_position"], ray["direction"], triangles)
    emitter = witnesses["emissive_edge"]
    if emitter:
        d, e = emitter["d"], emitter["e"]
        checks["emitter_e_keeps_d_prefix_hit"] = e["prefix"]["accepted"] and same_selected_trace(d, e)
    for name in ("short_block", "tall_block"):
        if witnesses[name]:
            c, e = (witnesses[name][arm] for arm in ("c_fixed", "e"))
            checks[f"{name}_e_keeps_c_fixed_tail"] = (
                e["prefix"]["classification"] == GDF_MISS and same_selected_trace(c, e))
    open_front = witnesses["open_front"]
    if open_front:
        c, e = (open_front[arm] for arm in ("c_fixed", "e"))
        checks["open_front_e_rejects_receding_sample_and_keeps_c_fixed_tail"] = (
            e["prefix"]["receding_sample_rejected"] and not e["prefix"]["accepted"] and same_selected_trace(c, e))


def count_mesh_validation(e_trace, f_trace, tiles, probe_pixels):
    counts = Counter({name: 0 for name in (
        "mesh_invoked", "mesh_confirmed", "mesh_inconclusive", "mesh_disproved", "mesh_flag_mismatches",
        "confirmed_distance_outside_prefix", "e_accepted_f_uses_tail", "e_used_tail_f_accepts_prefix",
    )})
    rays = []
    records = zip(TRACE_RECORD.iter_unpack(e_trace), TRACE_RECORD.iter_unpack(f_trace), strict=True)
    for index, (e, f) in enumerate(records):
        e_state, f_state = prefix_state(e[11]), prefix_state(f[11])
        mesh = mesh_validation_state(f[11])
        counts["mesh_flag_mismatches"] += (mesh["invoked"] != (f_state["classification"] == GDF_HIT) or
            (not mesh["invoked"] and (mesh["confirmed"] or mesh["inconclusive"])) or
            (mesh["confirmed"] and mesh["inconclusive"]))
        counts["e_accepted_f_uses_tail"] += e_state["accepted"] and not f_state["accepted"]
        counts["e_used_tail_f_accepts_prefix"] += not e_state["accepted"] and f_state["accepted"]
        if mesh["invoked"]:
            counts["mesh_invoked"] += 1
            counts["mesh_confirmed"] += mesh["confirmed"]
            counts["mesh_inconclusive"] += mesh["inconclusive"]
            counts["mesh_disproved"] += not (mesh["confirmed"] or mesh["inconclusive"])
            counts["confirmed_distance_outside_prefix"] += mesh["confirmed"] and not (
                f[2] - HANDOFF_TOLERANCE <= f[7] <= f[3] + HANDOFF_TOLERANCE)
            rays.append({**readback_address(index, tiles, probe_pixels), "e_prefix": e_state, "f_prefix": f_state,
                "mesh_validation": mesh, "e_gdf_hit_m": e[7], "f_gdf_hit_m": f[7]})
    return {"counts": counts, "rays": rays}


def add_mesh_validation_geometry(mesh, e_probes, e_buffers, triangles):
    planes_by_box = defaultdict(set)
    for triangle in triangles:
        planes_by_box[triangle.object_name].add((triangle.normal, dot(triangle.normal, triangle.origin)))
    counts = Counter()
    for ray in mesh["rays"]:
        probe = e_probes.get(ray["pixel"], {}).get((*ray["tile"], ray["slot"]))
        if probe is None or not ray["e_prefix"]["accepted"]:
            continue
        receiver, normal, ray_base, trace_base = probe
        lane = ray["lane"]
        index = lane[1] * RAY_EDGE + lane[0]
        e = captured_ray(e_buffers, trace_base + index, ray_base + index)
        direction = probe_ray_direction(lane)
        origin = add(add(receiver[:3], multiply(direction, SURFACE_BIAS)), multiply(normal, SURFACE_BIAS))
        interval = (e["gdf_min"], e["gdf_max"])
        overlaps = {box: overlap for box, planes in planes_by_box.items()
            if (overlap := clip_box_planes(origin, direction, interval, planes)) is not None}
        ray["cpu_box_intersections_in_e_prefix"] = overlaps
        counts["evaluated_e_prefixes"] += 1
        counts["e_prefixes_with_cpu_box_intersection"] += bool(overlaps)
        counts["e_prefixes_without_cpu_box_intersection"] += not overlaps
    mesh["cpu_box_counts"] = counts


def check_mesh_witnesses(witnesses, checks, triangles, voxel_size):
    for name, ray in witnesses.items():
        if not ray:
            continue
        f = ray["f"]
        f["prefix"] = prefix_state(f["flags"])
        f["prefix"]["receding_sample_rejected"] = bool(f["flags"] & GDF_PREFIX_RECEDING_REJECTED)
        f["prefix"]["mesh_validation"] = mesh_validation_state(f["flags"])
        f["luminance_delta_vs_e"] = f["trace_luminance"] - ray["e"]["trace_luminance"]
        checks[f"{name}_valid_f_receiver"] = bool(f["flags"] & RECEIVER_VALID)
        if name in CORNER_WITNESSES and f["gdf_hit_position"] is not None:
            ray["surfaces_at_f_hit"] = nearest_surfaces(f["gdf_hit_position"], ray["direction"], triangles)

    for name in ("short_corner_upper", "short_corner_lower"):
        ray = witnesses[name]
        if ray:
            f = ray["f"]
            mesh = f["prefix"]["mesh_validation"]
            checks[f"{name}_f_rejects_hit_without_cpu_triangle"] = (
                ray["gdf_triangle_first_hit"] is None and ray["e"]["prefix"]["accepted"] and
                mesh["invoked"] and not mesh["confirmed"] and not mesh["inconclusive"] and
                not f["prefix"]["accepted"] and same_selected_trace(ray["c_fixed"], f))

    for name, surface in (("near_floor", "Floor"), ("near_left_wall", "Left Wall")):
        ray = witnesses[name]
        if ray:
            hit = ray["gdf_triangle_first_hit"]
            surfaces = ray.get("surfaces_at_f_hit", {})
            checks[f"{name}_f_keeps_same_near_surface"] = bool(hit and hit["object"] == surface and
                hit["front_face"] and hit["distance"] < 0.5 and ray["e"]["prefix"]["accepted"] and
                next(iter(surfaces), None) == surface and surfaces[surface]["distance"] <= voxel_size)

    emitter = witnesses["emissive_edge"]
    if emitter:
        f, hit = emitter["f"], emitter["gdf_triangle_first_hit"]
        mesh = f["prefix"]["mesh_validation"]
        checks["emitter_f_keeps_emissive_hit"] = bool(hit and hit["emissive"] and f["prefix"]["accepted"] and
            (mesh["confirmed"] or (mesh["inconclusive"] and same_selected_trace(emitter["e"], f))) and
            abs(f["gdf_hit_distance"] - hit["distance"]) <= voxel_size and
            all(abs(channel - 8.0) <= EMITTER_RGB_TOLERANCE for channel in f["radiance_and_class"][:3]))
    open_front = witnesses["open_front"]
    if open_front:
        f = open_front["f"]
        checks["open_front_f_keeps_e_rejection"] = (f["prefix"]["receding_sample_rejected"] and
            not f["prefix"]["mesh_validation"]["invoked"] and same_selected_trace(open_front["e"], f))


def measure_floor(a_path, c_path, c_fixed_path, scene_path, trace_limit, d_path=None, e_path=None, f_path=None):
    paths = {"a": a_path, "c": c_path}
    if c_fixed_path:
        paths["c_fixed"] = c_fixed_path
    if d_path:
        paths["d"] = d_path
    if e_path:
        paths["e"] = e_path
    if f_path:
        paths["f"] = f_path
    captures = {name: read_capture(path) for name, path in paths.items()}
    manifest = captures["a"][0]
    voxel_size = capture_gdf_voxel_size(manifest)
    view = manifest["view"]
    if view["temporalJitter"] or view["camera"]["projection"] != "perspective":
        raise ValueError("Floor oracle requires the fixed perspective view without jitter")
    for name, (other_manifest, _, _, _) in captures.items():
        check_capture_alignment(manifest, other_manifest, name)

    triangles = read_triangles(scene_path)
    witnesses = {}
    checks = {}
    witness_specs = {**FLOOR_WITNESSES, **CORNER_WITNESSES} if f_path else FLOOR_WITNESSES
    for name, (pixel, slot_key, lane) in witness_specs.items():
        probes = {arm: data[1].get(pixel, {}).get(slot_key) for arm, data in captures.items()}
        if any(probe is None for probe in probes.values()):
            witnesses[name] = None
            continue

        receiver4, normal = probes["a"][:2]
        receiver = receiver4[:3]
        direction = probe_ray_direction(lane)
        cosine = dot(normal, direction)
        primary = primary_hit(triangles, pixel, manifest["resolution"], view["camera"])
        gdf_origin = add(add(receiver, multiply(direction, SURFACE_BIAS)), multiply(normal, SURFACE_BIAS))
        screen_origin = floor_screen_origin(receiver, normal, manifest)
        gdf_hit = first_hit(triangles, gdf_origin, direction, MIN_TRACE_DISTANCE, trace_limit)
        screen_hit = first_hit(triangles, screen_origin, direction, MIN_TRACE_DISTANCE, trace_limit)
        if screen_hit:
            screen_hit["projected_pixel"] = floor_screen_pixel(screen_hit["position"], manifest)
        if gdf_hit:
            gdf_hit["projected_pixel"] = floor_screen_pixel(gdf_hit["position"], manifest)

        ray_index = lane[1] * RAY_EDGE + lane[0]
        results = {}
        for arm, (_, _, ray_base, trace_base) in probes.items():
            capture = captured_ray(captures[arm][3], trace_base + ray_index, ray_base + ray_index)
            results[arm] = floor_ray_result(capture, gdf_origin, direction, gdf_hit)
        old_gdf_flags = RECEIVER_VALID | SCREEN_INVOKED | GDF_INVOKED
        expected_receiver = {
            "short_corner_upper": "Tall Block", "short_corner_lower": "Tall Block", "near_floor": "Short Block",
        }.get(name, "Floor")
        receiver_label = "floor_receiver" if name in FLOOR_WITNESSES else "expected_receiver"
        checks[f"{name}_matches_{receiver_label}"] = bool(primary and primary["object"] == expected_receiver and
            length(subtract(receiver, primary["position"])) <= 0.002 and cosine > 0.0 and
            all(probe[:2] == probes["a"][:2] for probe in probes.values()))
        checks[f"{name}_used_gdf_in_a_and_c"] = all((results[arm]["flags"] & old_gdf_flags) == old_gdf_flags and
            not results[arm]["flags"] & (SCREEN_ACCEPTED | DETAIL_INVOKED) for arm in ("a", "c"))
        if c_fixed_path:
            checks[f"{name}_valid_fixed_receiver"] = bool(results["c_fixed"]["flags"] & RECEIVER_VALID)
        witnesses[name] = {
            "pixel": pixel, "tile": slot_key[:2], "slot": slot_key[2], "lane": lane,
            "receiver": receiver, "receiver_normal": normal, "receiver_cosine": cosine, "direction": direction,
            "gdf_origin": gdf_origin, "screen_origin": screen_origin,
            "gdf_triangle_first_hit": gdf_hit, "screen_triangle_first_hit": screen_hit,
            **results,
        }
        if name in CORNER_WITNESSES:
            ray = witnesses[name]
            if results["e"]["gdf_hit_position"] is not None:
                ray["surfaces_at_e_prefix_hit"] = nearest_surfaces(
                    results["e"]["gdf_hit_position"], direction, triangles)
            if name.startswith("short_corner_"):
                ray["short_block_corner"] = short_block_corner(ray, triangles)

    report = {
        "scene": str(scene_path), "captures": {name: str(path) for name, path in paths.items()},
        "gdf_voxel_size_m": voxel_size,
        "resolution": manifest["resolution"], "frame": manifest["ray_capture"]["frame"],
        "probe_tiles": manifest["ray_capture"]["tiles"], "guide_crop": manifest["ray_capture"]["pixel_rect"],
        "screen_end_scope": {
            name: ("UE ProbeRayDistance float bits masked in centimeters and decoded to meters"
                if name in ("c_fixed", "d", "e", "f") else "unencoded Screen distance in meters")
            for name in paths
        },
        "witnesses": witnesses, "checks": checks,
    }
    if d_path:
        c, d = captures["c_fixed"], captures["d"]
        probe_pixels = {slot: pixel for pixel, probes in c[1].items() for slot in probes}
        prefix = count_prefix_control(c[3][0], d[3][0], manifest["ray_capture"]["tiles"], probe_pixels, voxel_size)
        counts = prefix["counts"]
        checks["d_uses_prefix_validation"] = d[0]["gdf_prefix_validation"] is True
        checks["d_matches_c_fixed_probe_placement"] = c[1] == d[1]
        checks["d_preserves_upstream_trace"] = counts["upstream_trace_mismatches"] == 0
        checks["d_invokes_only_nonempty_prefixes"] = counts["prefix_invoked"] > 0 and (
            counts["expected_prefix_invocation_mismatches"] == 0)
        checks["d_accepts_only_gdf_hits"] = counts["prefix_outcome_mismatches"] == 0
        checks["d_selects_correct_prefix_or_tail"] = (
            counts["selected_prefix_interval_mismatches"] == counts["selected_tail_mismatches"] == 0)
        check_prefix_witnesses(witnesses, checks, voxel_size)
        report["gdf_prefix_control"] = prefix
        report["d_tolerances"] = {"handoff_m": HANDOFF_TOLERANCE, "triangle_distance_m": voxel_size,
            "emitter_rgb": EMITTER_RGB_TOLERANCE}
    if e_path:
        e = captures["e"]
        tiles = manifest["ray_capture"]["tiles"]
        prefix_e = count_prefix_control(c[3][0], e[3][0], tiles, probe_pixels, voxel_size)
        receding = count_receding_rejections(d[3][0], e[3][0], tiles, probe_pixels)
        receding_counts = receding["counts"]
        counts_e = prefix_e["counts"]
        checks["e_uses_receding_sample_check"] = e[0]["gdf_prefix_reject_receding"] is True
        checks["e_matches_c_fixed_probe_placement"] = c[1] == e[1]
        checks["e_preserves_upstream_trace"] = counts_e["upstream_trace_mismatches"] == 0
        checks["e_selects_valid_prefix_or_tail"] = all(counts_e[field] == 0 for field in (
            "expected_prefix_invocation_mismatches", "prefix_outcome_mismatches", "selected_prefix_interval_mismatches",
            "selected_tail_mismatches"))
        checks["e_rejections_occur_inside_prefix"] = (receding_counts["receding_rejected"] > 0 and
            receding_counts["receding_rejected_outside_prefix"] == 0)
        check_receding_witnesses(witnesses, checks, triangles)
        prefix_geometry = compare_prefix_triangles(d[1], d[3], e[3], triangles, trace_limit, voxel_size)
        geometry_counts = prefix_geometry["counts"]
        checks["e_triangle_oracle_covers_all_d_prefix_hits"] = geometry_counts["d_accepted"] == counts["prefix_accepted"]
        checks["e_retains_d_prefix_hits_with_cpu_triangle_in_prefix"] = (
            0 < geometry_counts["d_accepted_with_cpu_triangle_in_prefix"] ==
            geometry_counts["e_keeps_prefix_with_cpu_triangle_in_prefix"])
        report["e_gdf_prefix_control"] = prefix_e
        report["gdf_receding_control"] = receding
        report["gdf_prefix_triangle_retention"] = prefix_geometry
    if f_path:
        f = captures["f"]
        prefix_f = count_prefix_control(c[3][0], f[3][0], tiles, probe_pixels, voxel_size, mesh_validation=True)
        mesh = count_mesh_validation(e[3][0], f[3][0], tiles, probe_pixels)
        add_mesh_validation_geometry(mesh, e[1], e[3], triangles)
        counts_f = prefix_f["counts"]
        mesh_counts = mesh["counts"]
        checks["f_uses_prefix_mesh_validation"] = f[0]["gdf_prefix_mesh_validation"] is True
        checks["f_matches_c_fixed_probe_placement"] = c[1] == f[1]
        checks["f_preserves_upstream_trace"] = counts_f["upstream_trace_mismatches"] == 0
        checks["f_mesh_validates_gdf_hits_in_prefix"] = (mesh_counts["mesh_invoked"] > 0 and
            mesh_counts["mesh_flag_mismatches"] == mesh_counts["confirmed_distance_outside_prefix"] == 0)
        checks["f_selects_valid_prefix_or_tail"] = all(counts_f[field] == 0 for field in (
            "expected_prefix_invocation_mismatches", "prefix_outcome_mismatches", "selected_prefix_interval_mismatches",
            "selected_tail_mismatches"))
        check_mesh_witnesses(witnesses, checks, triangles, voxel_size)
        report["f_gdf_prefix_control"] = prefix_f
        report["gdf_prefix_mesh_validation"] = mesh
    return report


def measure(a_path, b_path, scene_path, detail_limit, trace_limit, pin_path, c_path=None):
    a, a_probes, a_counts, a_buffers = read_capture(a_path)
    b, b_probes, b_counts, b_buffers = read_capture(b_path)
    check_capture_alignment(a, b, "B")
    a_info = a["ray_capture"]
    if c_path:
        c, c_probes, c_counts, c_buffers = read_capture(c_path)
        check_capture_alignment(a, c, "C")
        control_counts = count_screen_control(c_buffers[0])
        control_groups = defaultdict(lambda: {"a_to_c": Counter(), "c_to_b": Counter()})

    triangles = read_triangles(scene_path)
    stats = {group: Counter() for group in ("all", "tall_receivers", "rays_first_hitting_tall_block")}
    matched_pixels = set(a_probes.keys() & b_probes.keys())
    tall_first_hits = []
    candidates = defaultdict(list)
    for pixel in sorted(matched_pixels):
        primary = primary_hit(triangles, pixel, a["resolution"], a["view"]["camera"])
        for slot_key in sorted(a_probes[pixel].keys() & b_probes[pixel].keys()):
            receiver, normal, b_ray_base, b_trace_base = b_probes[pixel][slot_key]
            a_receiver, a_normal, a_ray_base, a_trace_base = a_probes[pixel][slot_key]
            if receiver != a_receiver or normal != a_normal:
                stats["all"]["changed_receiver_probe_slots"] += 1
                continue
            if primary is None or length(subtract(receiver[:3], primary["position"])) > 0.002:
                stats["all"]["receiver_geometry_mismatch_slots"] += 1
                continue

            c_probe = c_probes.get(pixel, {}).get(slot_key) if c_path else None
            c_receiver_matches = c_probe is not None and c_probe[:2] == (receiver, normal)
            tall_receiver = primary["object"] == "Tall Block"
            for ray_index in range(RAY_COUNT):
                ca = captured_ray(a_buffers, a_trace_base + ray_index, a_ray_base + ray_index)
                cb = captured_ray(b_buffers, b_trace_base + ray_index, b_ray_base + ray_index)
                if not (ca["flags"] & cb["flags"] & RECEIVER_VALID):
                    stats["all"]["invalid_paired_rays"] += 1
                    continue

                lane = (ray_index % RAY_EDGE, ray_index // RAY_EDGE)
                direction = probe_ray_direction(lane)
                origin = add(add(receiver[:3], multiply(direction, SURFACE_BIAS)), multiply(normal, SURFACE_BIAS))
                detail_min = max(MIN_TRACE_DISTANCE, cb["screen_end"] - 2.0 * SURFACE_BIAS)
                hit = first_hit(triangles, origin, direction, MIN_TRACE_DISTANCE, trace_limit)
                tall_hit = hit is not None and hit["object"] == "Tall Block"
                ray = {
                    "pixel": pixel, "tile": slot_key[:2], "slot": slot_key[2], "lane": lane,
                    "receiver_object": primary["object"], "receiver_position": receiver[:3], "receiver_normal": normal,
                    "receiver_cosine": dot(normal, direction), "origin": origin, "direction": direction,
                    "detail_min": detail_min, "first_hit": hit, "a": ca, "b": cb,
                }
                case = classify(ray)
                if c_receiver_matches:
                    cc = captured_ray(c_buffers, c_probe[3] + ray_index, c_probe[2] + ray_index)
                    if cc["flags"] & RECEIVER_VALID:
                        count_control_ray(control_groups, ca, cb, cc, case)
                        ray["c"] = cc
                        if (skipped_detail_and_used_gdf(ca, cb) and cc["flags"] & GDF_INVOKED and
                                cc["gdf_min"] - ca["gdf_min"] > HANDOFF_TOLERANCE and
                                abs(cc["gdf_min"] - cb["gdf_min"]) <= HANDOFF_TOLERANCE and
                                hit and hit["front_face"] and hit["edge_clearance"] > 0.015 and
                                ray["receiver_cosine"] > 0.0):
                            candidates["positive_screen_handoff"].append(ray)
                groups = [stats["all"]]
                if tall_receiver:
                    groups.append(stats["tall_receivers"])
                if tall_hit:
                    groups.append(stats["rays_first_hitting_tall_block"])
                    tall_first_hits.append(ray)
                for counts in groups:
                    counts["paired_rays"] += 1
                    counts["cpu_first_hits"] += int(hit is not None)
                    counts["receiver_hemisphere_rays"] += int(ray["receiver_cosine"] > 0.0)
                    count_trace_transitions(counts, ca, cb)
                    if case:
                        counts[case] += 1

                if (case and hit and hit["distance"] >= detail_min and hit["front_face"] and
                        hit["edge_clearance"] > 0.015 and ray["receiver_cosine"] > 0.0):
                    candidates[case].append(ray)

    def rank(case, ray):
        hit = ray["first_hit"]
        if case == "detail_accepted":
            observed = ray["b"]["detail_end"] - length(subtract(ray["receiver_position"], ray["origin"]))
            return (hit["distance"] < detail_limit, abs(observed - hit["distance"]) < 0.05,
                hit["object"] == "Tall Block", hit["edge_clearance"], hit["incidence"])
        return (hit["distance"] > detail_limit, hit["object"] == "Tall Block",
            ray["b"]["gdf_classification"] == 2, hit["edge_clearance"], hit["incidence"])

    cases = ("detail_accepted", "gdf_continuation")
    if pin_path:
        pinned = json.loads(pin_path.read_text(encoding="utf-8"))["witnesses"]
        witnesses = {case: next((ray for ray in candidates[case] if key(ray) == key(pinned[case])), None)
                     for case in cases}
    else:
        witnesses = {case: max(candidates[case], key=lambda ray: rank(case, ray), default=None) for case in cases}
    checks = {}
    for case, ray in witnesses.items():
        if not ray:
            continue
        hit = ray["first_hit"]
        b_capture = ray["b"]
        ray["first_hit_beyond_detail_interval"] = hit["distance"] > detail_limit
        detail_distance = b_capture["detail_end"] - length(
            subtract(ray["receiver_position"], ray["origin"]))
        ray["b_detail_distance_from_biased_origin"] = detail_distance
        if case == "detail_accepted":
            checks["detail_hit_is_in_range"] = hit["distance"] < detail_limit
            checks["detail_hit_matches_triangle"] = abs(detail_distance - hit["distance"]) <= DETAIL_TRIANGLE_TOLERANCE
            checks["detail_hit_is_written"] = (b_capture["radiance_and_class"][3] == 9.0 and
                b_capture["trace_distance"] == b_capture["detail_end"])
        else:
            checks["continuation_triangle_is_beyond_detail"] = hit["distance"] > detail_limit
            checks["detail_reached_its_limit"] = abs(detail_distance - detail_limit) <= HANDOFF_TOLERANCE
            expected_start = max(MIN_TRACE_DISTANCE, b_capture["detail_end"] - 2.0 * SURFACE_BIAS)
            checks["gdf_start_uses_detail_endpoint"] = abs(b_capture["gdf_min"] - expected_start) <= HANDOFF_TOLERANCE

    result = {
        "scene": str(scene_path), "a": str(a_path), "b": str(b_path), "resolution": a["resolution"],
        "gdf_voxel_size_m": capture_gdf_voxel_size(a),
        "frame": a_info["frame"], "probe_tiles": a_info["tiles"], "guide_crop": a_info["pixel_rect"],
        "detail_distance_limit": detail_limit, "trace_distance_limit": trace_limit,
        "a_probe_counts": a_counts, "b_probe_counts": b_counts, "matching_probe_pixels": len(matched_pixels),
        "counts": stats, "witnesses": witnesses, "checks": checks, "tall_first_hits": tall_first_hits,
        "tolerances_m": {"handoff": HANDOFF_TOLERANCE, "detail_vs_triangle": DETAIL_TRIANGLE_TOLERANCE},
    }
    if c_path:
        c_to_b = control_groups["b_skipped_detail_and_used_gdf"]["c_to_b"]
        skipped = stats["all"]["gdf_after_screen"]
        control_witness = max(candidates["positive_screen_handoff"], key=rank_positive_screen_handoff, default=None)
        checks["c_detail_disabled"] = control_counts["detail_invoked"] == control_counts["detail_accepted"] == 0
        checks["c_screen_invoked_for_valid_rays"] = (
            control_counts["valid_rays"] > 0 and control_counts["valid_rays"] == control_counts["screen_invoked"])
        checks["c_gdf_start_uses_screen_endpoint"] = (
            control_counts["screen_rejected"] > 0 and
            control_counts["screen_rejected"] == control_counts["gdf_after_screen_endpoint_matches"])
        checks["c_covers_paired_rays"] = (
            control_groups["all"]["c_to_b"]["compared_rays"] == stats["all"]["paired_rays"])
        checks["c_matches_b_gdf_start_when_b_skips_detail"] = (
            skipped > 0 and skipped == c_to_b["compared_rays"] == c_to_b["gdf_start_matches_within_tolerance"])
        checks["c_captures_b_witnesses"] = all(ray and "c" in ray for ray in witnesses.values())
        checks["c_positive_screen_handoff_witness"] = control_witness is not None
        result["c"] = str(c_path)
        result["screen_control"] = {
            "probe_counts": c_counts, "all_readback_tile_rays": control_counts, "comparisons": control_groups,
            "positive_screen_handoff_witness": control_witness,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--c", type=Path)
    parser.add_argument("--floor", action="store_true")
    parser.add_argument("--c-fixed", type=Path)
    parser.add_argument("--d", type=Path)
    parser.add_argument("--e", type=Path)
    parser.add_argument("--f", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=ROOT / "Asset/Lumen/CornellBox.gltf")
    parser.add_argument("--detail-limit", type=float, default=1.8)
    parser.add_argument("--trace-limit", type=float, default=5.0)
    parser.add_argument("--pin", type=Path)
    args = parser.parse_args()
    if args.floor and not args.c:
        parser.error("--floor requires --c")
    if not args.floor and not args.b:
        parser.error("--b is required for the Detail comparison")
    if args.c_fixed and not args.floor:
        parser.error("--c-fixed requires --floor")
    if args.d and not (args.floor and args.c_fixed):
        parser.error("--d requires --floor and --c-fixed")
    if args.e and not args.d:
        parser.error("--e requires --d")
    if args.f and not args.e:
        parser.error("--f requires --e")
    try:
        if not (MIN_TRACE_DISTANCE < args.detail_limit <= args.trace_limit < math.inf):
            raise ValueError("Expected a positive finite Detail limit within the trace limit")
        if args.floor:
            report = measure_floor(args.a.resolve(), args.c.resolve(), args.c_fixed.resolve() if args.c_fixed else None,
                args.scene.resolve(), args.trace_limit, args.d.resolve() if args.d else None,
                args.e.resolve() if args.e else None, args.f.resolve() if args.f else None)
        else:
            report = measure(args.a.resolve(), args.b.resolve(), args.scene.resolve(), args.detail_limit,
                args.trace_limit, args.pin, args.c.resolve() if args.c else None)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (KeyError, OSError, OverflowError, ValueError, TypeError, struct.error) as error:
        print(f"Cornell Box probe oracle error: {error}", file=sys.stderr)
        return 2
    missing = [name for name, ray in report["witnesses"].items() if ray is None]
    if missing:
        print(f"No usable witness for {', '.join(missing)}. See {args.out}", file=sys.stderr)
        return 1
    failed = [name for name, passed in report["checks"].items() if not passed]
    if failed:
        print(f"Witness checks failed: {', '.join(failed)}. See {args.out}", file=sys.stderr)
        return 1
    print(f"Wrote Cornell Box probe witnesses and triangle hits to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
