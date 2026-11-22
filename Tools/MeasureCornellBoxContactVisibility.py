"""Replay the pinned Cornell Box contact visibility witness from one F capture.

Use GPU interpolation taps and preserve the three spatial filters' original
weights. Replay the pinned Screen emitter sample and other verified emitter
samples that reach the five ROIs. Remove a source only at receivers whose
segment to its triangle hit first intersects the tall block.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys

from CheckCornellBoxDetailRays import (
    capture_gdf_voxel_size, EMITTER_RGB_TOLERANCE, FLOAT, FLOAT4, GDF_HIT, HALF4,
    INVALID_PROBE, MIN_TRACE_DISTANCE, PROBE_POSITION, PROBE_SLOTS, PROBE_TILE_EDGE,
    RAY_COUNT, RAY_EDGE, ROOT, SCREEN_ACCEPTED, SURFACE_BIAS,
    add, captured_ray, dot, first_hit, floor_screen_origin, length, multiply,
    normalize, probe_ray_direction, read_buffer, read_capture, read_triangles, subtract,
)
from MeasureCornellBoxGI import (
    REGION_NAMES, compare_region, luminance, read_pixels, relative_to_floor,
    summarize_frame, summarize_region,
)


RECEIVER_PIXEL = (286, 322)
SOURCE_PIXEL = (286, 334)
SOURCE_SLOT = (17, 20, 0)
SOURCE_LANE = (3, 7)
SOURCE_LANE_INDEX = SOURCE_LANE[1] * RAY_EDGE + SOURCE_LANE[0]
FILTER_PASS_COUNT = 3
FILTER_ANGLE = math.radians(10.0)
FILTER_POSITION_SCALE = 1000.0
FILTER_REPLAY_TOLERANCE = 5.0e-4
# Gather indirect is stored in FP16, unlike the Gather color and probe radiance.
GATHER_REPLAY_TOLERANCE = 5.0e-4
SCREEN_HIT_DISTANCE_TOLERANCE = 0.02
SCREEN_HIT_CLASS = 5.0
ORACLE_BIAS = 0.0001
INTERPOLATION = struct.Struct("<4I4f")
CONTACT_LAYOUTS = {
    "position": ("rgba32f", FLOAT4),
    "normal": ("rgba16f", HALF4),
    "diffuse": ("rgba16f", HALF4),
    "interpolation": ("u32x4_f32x4", INTERPOLATION),
    "gather_indirect": ("rgba16f", HALF4),
    "gather_color": ("rgba32f", FLOAT4),
    "history_weight": ("f32", FLOAT),
    "raw_radiance": ("rgba32f", FLOAT4),
    "filtered_radiance": ("rgba32f", FLOAT4),
}


def contact_buffer(directory, capture, name, count):
    contact = capture["contact_visibility"]
    source = contact[name]
    expected_format, layout = CONTACT_LAYOUTS[name]
    if source["frame"] != capture["frame"] or contact["frame"] != capture["frame"]:
        raise ValueError(f"{name} must use the contact capture execution")
    if source["format"] != expected_format:
        raise ValueError(f"{name} must use {expected_format}")
    return read_buffer(directory, source["file"], layout, count)


def read_pixel(layout, data, index):
    return layout.unpack_from(data, index * layout.size)


def channel_max_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def filter_angle(cosine):
    # probeAcosFast in LumenScreenProbeFilter.slang.
    cosine = max(-1.0, min(1.0, cosine))
    x = abs(cosine)
    result = (-0.156583 * x + 0.5 * math.pi) * math.sqrt(1.0 - x)
    return result if cosine >= 0.0 else math.pi - result


def spatial_filter_row(index, positions, distances, probe_size, direction):
    width, height = probe_size
    receiver = positions[index]
    if receiver is None:
        return []
    weights = []
    hit_distance = distances[index]
    if hit_distance >= 0.0:
        weights.append((index, 1.0))
    tile = index // PROBE_SLOTS
    x, y = tile % width, tile // width
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        neighbor_index = (ny * width + nx) * PROBE_SLOTS
        neighbor = positions[neighbor_index]
        neighbor_distance = distances[neighbor_index]
        if neighbor is None or neighbor_distance < 0.0:
            continue
        relative_depth = abs(neighbor[3] - receiver[3]) / receiver[3]
        position_weight = 2.0 ** (-FILTER_POSITION_SCALE * relative_depth * relative_depth)
        if hit_distance >= 0.0:
            neighbor_distance = min(neighbor_distance, hit_distance)
        to_hit = subtract(add(neighbor[:3], multiply(direction, neighbor_distance)), receiver[:3])
        hit_length = length(to_hit)
        if hit_length == 0.0:
            continue
        angle = filter_angle(dot(to_hit, direction) / hit_length)
        weight = position_weight * (1.0 - min(1.0, angle / FILTER_ANGLE))
        if weight > 0.0:
            weights.append((neighbor_index, weight))
    total = sum(weight for _, weight in weights)
    return [(source, weight / total) for source, weight in weights]


def spatial_filter_rows(positions, distances, probe_size, direction):
    return [spatial_filter_row(index, positions, distances, probe_size, direction) for index in range(len(positions))]


def emitter_filter_support(taps_by_pixel, probe_size):
    width, height = probe_size
    tiles = {probe // PROBE_SLOTS for taps in taps_by_pixel.values() for probe, _ in taps}
    support = set()
    for tile in tiles:
        x, y = tile % width, tile // width
        for dy in range(-FILTER_PASS_COUNT, FILTER_PASS_COUNT + 1):
            reach = FILTER_PASS_COUNT - abs(dy)
            for dx in range(-reach, reach + 1):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    support.add((y + dy) * width + x + dx)
    return support


def find_emitter_samples(manifest, layers, triangles, probe_pixels, positions, distances, directions, support, reference):
    width = manifest["resolution"][0]
    voxel_size = capture_gdf_voxel_size(manifest)
    sources = defaultdict(dict)
    unverified = Counter()
    for probe, receiver in enumerate(positions):
        if receiver is None or probe // PROBE_SLOTS not in support:
            continue
        x, y = probe_pixels[probe]
        normal = normalize(read_pixel(HALF4, layers["normal"], y * width + x)[:3])
        screen_origin = floor_screen_origin(receiver[:3], normal, manifest)
        normal_bias = multiply(normal, SURFACE_BIAS)
        for lane, direction in enumerate(directions):
            address = probe * RAY_COUNT + lane
            raw = read_pixel(FLOAT4, layers["raw_radiance"], address)
            rgb, trace_class = raw[:3], raw[3]
            if trace_class not in (GDF_HIT, SCREEN_HIT_CLASS) or channel_max_error(rgb, reference) > EMITTER_RGB_TOLERANCE:
                continue
            if trace_class == SCREEN_HIT_CLASS:
                origin, tolerance = screen_origin, SCREEN_HIT_DISTANCE_TOLERANCE
            else:
                origin = add(add(receiver[:3], multiply(direction, SURFACE_BIAS)), normal_bias)
                tolerance = voxel_size
            hit = first_hit(triangles, origin, direction, MIN_TRACE_DISTANCE, manifest["view"]["camera"]["zfar"])
            distance, = read_pixel(FLOAT, distances, address)
            if hit and hit["emissive"] and abs(hit["distance"] - distance) <= tolerance:
                sources[lane][probe] = (rgb, hit["position"], trace_class)
            else:
                unverified[hit["object"] if hit else "miss"] += 1
    return sources, unverified


def replay_emitter_paths(layers, positions, distances, probe_size, directions, sources):
    paths = [dict() for _ in positions]
    maximum_error = 0.0
    for lane, emitters in sources.items():
        lane_distances = [read_pixel(FLOAT, distances, p * RAY_COUNT + lane)[0] for p in range(len(positions))]
        rows = spatial_filter_rows(positions, lane_distances, probe_size, directions[lane])
        rgb = [read_pixel(FLOAT4, layers["raw_radiance"], p * RAY_COUNT + lane)[:3] for p in range(len(positions))]
        # Keep each source coefficient beside the RGB replay. The filter's
        # distances, weights and normalization remain those of the capture.
        terms = [{p: 1.0} if p in emitters else {} for p in range(len(positions))]
        for _ in range(FILTER_PASS_COUNT):
            next_rgb = []
            next_terms = []
            for row in rows:
                next_rgb.append(tuple(sum(rgb[p][channel] * weight for p, weight in row) for channel in range(3)))
                coefficients = defaultdict(float)
                for p, weight in row:
                    for source, coefficient in terms[p].items():
                        coefficients[source] += coefficient * weight
                next_terms.append(coefficients)
            rgb, terms = next_rgb, next_terms
        for p, coefficients in enumerate(terms):
            filtered = read_pixel(FLOAT4, layers["filtered_radiance"], p * RAY_COUNT + lane)
            maximum_error = max(maximum_error, channel_max_error(rgb[p], filtered[:3]))
            paths[p].update({(lane, source): coefficient for source, coefficient in coefficients.items()})
    return paths, maximum_error


def blocked_emitter_terms(triangles, receiver, normal, diffuse, taps, cosines, paths, sources, pinned):
    weights = defaultdict(float)
    for probe, weight in taps:
        for source, coefficient in paths[probe].items():
            weights[source] += weight * coefficient
    sample_count = sum(cosine > 0.0 for cosine in cosines)
    terms = {}
    pinned_hit = None
    for (lane, probe), coefficient in weights.items():
        cosine = cosines[lane]
        if cosine <= 0.0:
            continue
        rgb, endpoint, _ = sources[lane][probe]
        blocker = visibility_hit(triangles, receiver, normal, endpoint)
        if (lane, probe) == pinned:
            pinned_hit = blocker
        if blocker and blocker["object"] == "Tall Block":
            weight = coefficient * min(1.0, cosine) * 2.0 / sample_count
            terms[lane, probe] = tuple(s * d * weight for s, d in zip(rgb, diffuse))
    return terms, pinned_hit


def integrate(taps, cosines, radiance, diffuse):
    active = [(lane, min(1.0, cosine)) for lane, cosine in enumerate(cosines) if cosine > 0.0]
    rgb = [0.0, 0.0, 0.0]
    for lane, cosine in active:
        for probe, weight in taps:
            trace = read_pixel(FLOAT4, radiance, probe * RAY_COUNT + lane)
            for channel in range(3):
                rgb[channel] += trace[channel] * cosine * weight
    if active:
        return tuple(y * (2.0 / len(active)) * m for y, m in zip(rgb, diffuse))
    return tuple(rgb)


def visibility_hit(triangles, receiver, normal, endpoint):
    origin = add(receiver[:3], multiply(normal, ORACLE_BIAS))
    delta = subtract(endpoint, origin)
    distance = length(delta)
    return first_hit(triangles, origin, normalize(delta), ORACLE_BIAS, distance - ORACLE_BIAS)


def probe_positions(root, ray_capture, positions, width, count):
    pixels = list(PROBE_POSITION.iter_unpack(read_buffer(
        root, ray_capture["probe_positions"], PROBE_POSITION, count)))
    points = [read_pixel(FLOAT4, positions, y * width + x) if x != INVALID_PROBE else None for x, y in pixels]
    return pixels, points


def region_coordinates(region):
    x, y, width, height = region
    return [(px, py) for py in range(y, y + height) for px in range(x, x + width)]


@dataclass
class VisibilityReplay:
    layers: dict
    region_pixels: dict
    taps_by_pixel: dict
    probe_pixels: list
    directions: list
    sources: dict
    paths: list
    emitter_scan: dict
    filter_error: float

    def pixel_inputs(self, index, pixel):
        receiver = read_pixel(FLOAT4, self.layers["position"], index)
        normal = normalize(read_pixel(HALF4, self.layers["normal"], index)[:3])
        diffuse = read_pixel(HALF4, self.layers["diffuse"], index)[:3]
        taps = self.taps_by_pixel[pixel]
        cosines = [dot(normal, direction) for direction in self.directions]
        return receiver, normal, diffuse, taps, cosines


def build_visibility_replay(manifest, root, capture, ray_capture, triangles, reference_rgb):
    if ray_capture["frame"] != capture["frame"] or manifest["view"]["temporalJitter"]:
        raise ValueError("The fixed-camera ray and visibility readbacks must share the lit execution")
    width, height = manifest["resolution"]
    pixel_count = width * height
    probe_size = ((width + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE, (height + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE)
    probe_count = math.prod(probe_size) * PROBE_SLOTS
    ray_count = probe_count * RAY_COUNT
    layers = {key: contact_buffer(root, capture, key, ray_count if key.endswith("radiance") else pixel_count)
              for key in CONTACT_LAYOUTS}
    raw_file = capture["contact_visibility"]["raw_radiance"]["file"]
    if raw_file != ray_capture["radiance"] and layers["raw_radiance"] != read_buffer(
            root, ray_capture["radiance"], FLOAT4, ray_count):
        raise ValueError("The ray and visibility raw radiance must agree")
    probe_pixels, positions = probe_positions(root, ray_capture, layers["position"], width, probe_count)
    distances = read_buffer(root, ray_capture["distance"], FLOAT, ray_count)
    directions = [probe_ray_direction((lane % RAY_EDGE, lane // RAY_EDGE)) for lane in range(RAY_COUNT)]
    region_pixels = {name: region_coordinates(manifest["regions"][name]) for name in REGION_NAMES}
    taps_by_pixel = {}
    for coords in region_pixels.values():
        for x, y in coords:
            interpolation = read_pixel(INTERPOLATION, layers["interpolation"], y * width + x)
            taps_by_pixel[x, y] = [(probe, weight) for probe, weight in zip(interpolation[:4], interpolation[4:]) if weight > 0.0]
    support = emitter_filter_support(taps_by_pixel, probe_size)
    sources, unverified = find_emitter_samples(manifest, layers, triangles, probe_pixels, positions, distances,
                                               directions, support, reference_rgb)
    paths, filter_error = replay_emitter_paths(layers, positions, distances, probe_size, directions, sources)
    emitter_scan = {"verified_samples_in_filter_support": sum(len(samples) for samples in sources.values()),
                    "unverified_samples_by_triangle_hit": unverified}
    return VisibilityReplay(layers, region_pixels, taps_by_pixel, probe_pixels, directions, sources, paths,
                            emitter_scan, filter_error)


def scene_emitter_rgb(scene_path):
    materials = json.loads(scene_path.read_text(encoding="utf-8"))["materials"]
    emitter = next((material for material in materials if material["name"] == "Ceiling Emitter"), None)
    if emitter is None or "emissiveTexture" in emitter:
        raise ValueError("The Cornell Box replay requires the constant Ceiling Emitter material")
    factor = emitter["emissiveFactor"]
    strength = emitter.get("extensions", {}).get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
    if len(factor) != 3 or factor[0] <= 0.0 or any(channel != factor[0] for channel in factor) or strength <= 0.0:
        raise ValueError("The Cornell Box replay requires neutral positive emission in the lit fixture")
    return multiply(factor, strength)


def measure_phase_visibility(manifest, root, capture, ray_capture, scene_path):
    """Return per-pixel removal from one probe phase using the G visibility replay."""
    triangles = read_triangles(scene_path)
    replay = build_visibility_replay(manifest, root, capture, ray_capture, triangles, scene_emitter_rgb(scene_path))
    width = manifest["resolution"][0]
    removed_rgb = {}
    roi_deltas = {}
    gather_error = 0.0
    for name, coords in replay.region_pixels.items():
        blocked_pixels = 0
        blocked_emitters = set()
        for x, y in coords:
            index = y * width + x
            receiver, normal, diffuse, taps, cosines = replay.pixel_inputs(index, (x, y))
            filtered_rgb = integrate(taps, cosines, replay.layers["filtered_radiance"], diffuse)
            gather_rgb = read_pixel(HALF4, replay.layers["gather_indirect"], index)[:3]
            gather_error = max(gather_error, channel_max_error(filtered_rgb, gather_rgb))
            terms, _ = blocked_emitter_terms(triangles, receiver, normal, diffuse, taps, cosines,
                                             replay.paths, replay.sources, None)
            removed_rgb[x, y] = tuple(math.fsum(rgb[channel] for rgb in terms.values()) for channel in range(3))
            blocked_pixels += bool(terms)
            blocked_emitters.update(terms)
        roi_deltas[name] = {"blocked_pixels": blocked_pixels, "blocked_emitters": len(blocked_emitters),
                            "mean_removed_y": math.fsum(luminance(removed_rgb[pixel]) for pixel in coords) / len(coords)}
    return {"removed_rgb": removed_rgb, "emitter_scan": replay.emitter_scan, "roi_deltas": roi_deltas,
            "filter_replay_max_abs_rgb_error": replay.filter_error, "gather_replay_max_abs_rgb_error": gather_error}


def dark_summary(root, capture, pixel_count, ray_count):
    summaries = {key: summarize_frame(*read_pixels(root, capture[key], capture["frame"], pixel_count))
                 for key in ("path_trace", "realtime", "indirect")}
    for key in ("gather_indirect", "gather_color", "raw_radiance", "filtered_radiance"):
        layout = CONTACT_LAYOUTS[key][1]
        count = ray_count if key.endswith("radiance") else pixel_count
        summaries[key] = summarize_frame(layout, contact_buffer(root, capture, key, count))
    return {key: {"nonfinite_rgb": value["nonfinite_rgb_pixels"], "max_abs_rgb": value["max_abs_rgb"]}
            for key, value in summaries.items()}


def compare_image_readbacks(layout, baseline, instrumented):
    same_bytes = baseline == instrumented
    changed_rgb = 0
    max_abs_rgb = 0.0
    sum_y_delta = 0.0
    if not same_bytes:
        for a, b in zip(layout.iter_unpack(baseline), layout.iter_unpack(instrumented)):
            delta = channel_max_error(a[:3], b[:3])
            changed_rgb += delta > 0.0
            max_abs_rgb = max(max_abs_rgb, delta)
            if delta > 0.0:
                sum_y_delta += luminance(subtract(b[:3], a[:3]))
    return {"same_bytes": same_bytes, "changed_rgb_pixels": changed_rgb,
            "max_abs_rgb_delta": max_abs_rgb, "mean_y_delta": sum_y_delta / (len(baseline) // layout.size)}


def cropped_image(layout, image, width, region):
    x, y, crop_width, crop_height = region
    return b"".join(image[(row * width + x) * layout.size:(row * width + x + crop_width) * layout.size]
                    for row in range(y, y + crop_height))


def ray_variation(baseline, instrumented, roi_probes):
    rgb = {}
    labels = {}
    distance = {}
    for i, (a, b) in enumerate(zip(FLOAT4.iter_unpack(baseline["radiance"]), FLOAT4.iter_unpack(instrumented["radiance"]))):
        if a[:3] != b[:3]:
            rgb[i] = channel_max_error(a[:3], b[:3])
        if a[3] != b[3]:
            labels[i] = (a[3], b[3])
    for i, (a, b) in enumerate(zip(FLOAT.iter_unpack(baseline["distance"]), FLOAT.iter_unpack(instrumented["distance"]))):
        if a != b:
            distance[i] = abs(a[0] - b[0])
    scopes = {"whole_buffer": None, **roi_probes}
    result = {}
    for name, probes in scopes.items():
        rgb_deltas = [delta for i, delta in rgb.items() if probes is None or i // RAY_COUNT in probes]
        distance_deltas = [delta for i, delta in distance.items() if probes is None or i // RAY_COUNT in probes]
        result[name] = {
            "radiance_changed_rays": len(rgb_deltas), "radiance_max_abs_rgb_delta": max(rgb_deltas, default=0.0),
            "classification_changed_rays": sum(probes is None or i // RAY_COUNT in probes for i in labels),
            "distance_changed_rays": len(distance_deltas), "distance_max_abs_delta_m": max(distance_deltas, default=0.0),
        }
    return result


def compare_baseline(baseline_path, manifest_path, manifest, pixel_count, source_index, roi_probes):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    if not math.isclose(capture_gdf_voxel_size(baseline), capture_gdf_voxel_size(manifest)):
        raise ValueError("F and the instrumented capture differ in GDF voxel size")
    for key in ("sample", "resolution", "view", "regions", "color_space", "screen_traces", "detail_tracing",
                "screen_gdf_continuation", "gdf_prefix_validation", "gdf_prefix_reject_receding",
                "gdf_prefix_mesh_validation"):
        if baseline[key] != manifest[key]:
            raise ValueError(f"F and the instrumented capture differ in {key}")
    baseline_ray = baseline["ray_capture"]
    instrumented_ray = manifest["ray_capture"]
    for key in ("frame", "tiles", "pixel_rect"):
        if baseline_ray[key] != instrumented_ray[key]:
            raise ValueError(f"F and the instrumented ray readbacks differ in {key}")
    ray_keys = ("position", "normal", "probe_positions", "trace", "radiance", "distance", "closest_hzb")
    baseline_rays = {key: (baseline_path.parent / baseline_ray[key]).read_bytes() for key in ray_keys}
    instrumented_rays = {key: (manifest_path.parent / instrumented_ray[key]).read_bytes() for key in ray_keys}
    rays = {key: baseline_rays[key] == instrumented_rays[key] for key in ray_keys}
    variation = ray_variation(baseline_rays, instrumented_rays, roi_probes)
    source_address = source_index * RAY_COUNT + SOURCE_LANE_INDEX
    witness = {key: read_pixel(layout, baseline_rays[key], source_address) ==
                    read_pixel(layout, instrumented_rays[key], source_address)
               for key, layout in (("radiance", FLOAT4), ("distance", FLOAT))}
    captures = {}
    roi_images = {}
    for name, capture in manifest["captures"].items():
        control = baseline["captures"][name]
        for key in ("frame", "path_samples", "accumulated_frames"):
            if capture[key] != control[key]:
                raise ValueError(f"F and the instrumented {name} capture differ in {key}")
        images = {}
        for key in ("path_trace", "realtime", "indirect"):
            layout, instrumented = read_pixels(manifest_path.parent, capture[key], capture["frame"], pixel_count)
            control_layout, control_image = read_pixels(baseline_path.parent, control[key], control["frame"], pixel_count)
            if layout != control_layout:
                raise ValueError(f"F and the instrumented {name} {key} formats differ")
            images[key] = compare_image_readbacks(layout, control_image, instrumented)
            if name == "on":
                width = manifest["resolution"][0]
                roi_images[key] = {region: compare_image_readbacks(layout,
                    cropped_image(layout, control_image, width, rect), cropped_image(layout, instrumented, width, rect))
                    for region, rect in manifest["regions"].items()}
        captures[name] = images
    return {"manifest": str(baseline_path), "ray_readbacks_same_bytes": rays, "pinned_ray_match": witness,
            "raw_ray_variation": variation, "captures": captures, "lit_roi_images": roi_images}


def measure(manifest_path, scene_path, baseline_path=None):
    manifest, probes, _, ray_buffers = read_capture(manifest_path)
    captures = manifest["captures"]
    capture = captures["on"]
    frame = capture["frame"]
    root = manifest_path.parent
    width, height = manifest["resolution"]
    pixel_count = width * height
    images = {key: read_pixels(root, capture[key], frame, pixel_count)
              for key in ("path_trace", "realtime", "indirect")}
    source_receiver, source_normal, source_base, source_trace = probes[SOURCE_PIXEL][SOURCE_SLOT]
    source_index = source_base // RAY_COUNT
    source_ray = captured_ray(ray_buffers, source_trace + SOURCE_LANE_INDEX, source_base + SOURCE_LANE_INDEX)
    source_rgb = source_ray["radiance_and_class"][:3]
    triangles = read_triangles(scene_path)
    direction = probe_ray_direction(SOURCE_LANE)
    source_origin = floor_screen_origin(source_receiver[:3], source_normal, manifest)
    emitter = first_hit(triangles, source_origin, direction, MIN_TRACE_DISTANCE, manifest["view"]["camera"]["zfar"])
    if (not source_ray["flags"] & SCREEN_ACCEPTED or max(source_rgb) <= 0.0 or emitter is None or
            not emitter["emissive"] or abs(emitter["distance"] - source_ray["trace_distance"]) > SCREEN_HIT_DISTANCE_TOLERANCE):
        raise ValueError("The pinned Screen sample must reproduce the emissive triangle first hit")
    regions = {name: manifest["regions"][name] for name in REGION_NAMES}
    replay = build_visibility_replay(manifest, root, capture, manifest["ray_capture"], triangles, source_rgb)
    layers = replay.layers
    pixels = replay.probe_pixels
    ray_count = len(pixels) * RAY_COUNT
    if source_index not in replay.sources[SOURCE_LANE_INDEX]:
        raise ValueError("The pinned emitter must be in the ROI filter support")
    pinned = (SOURCE_LANE_INDEX, source_index)
    source_coefficients = [terms.get(pinned, 0.0) for terms in replay.paths]
    statistics = {}
    region_deltas = {}
    roi_probes = {}
    gather_replay_error = 0.0
    witness = None
    tall_contact_pixels = []
    for name, coords in replay.region_pixels.items():
        rgb_samples = {}
        raw_blocked_count = 0
        filtered_blocked_count = 0
        source_taps = set()
        roi_probes[name] = set()
        raw_removed_y = []
        removed_y = []
        verified_removed_y = []
        blocked_sources = Counter()
        verified_blocked_pixels = 0
        history_weights = []
        for x, y in coords:
            pixel_index = y * width + x
            receiver, normal, diffuse, taps, cosines = replay.pixel_inputs(pixel_index, (x, y))
            roi_probes[name].update(probe for probe, _ in taps)
            raw_rgb = integrate(taps, cosines, layers["raw_radiance"], diffuse)
            filtered_rgb = integrate(taps, cosines, layers["filtered_radiance"], diffuse)
            gather_indirect = read_pixel(HALF4, layers["gather_indirect"], pixel_index)[:3]
            gather_rgb = read_pixel(FLOAT4, layers["gather_color"], pixel_index)[:3]
            temporal_rgb = read_pixel(*images["realtime"], pixel_index)[:3]
            temporal_indirect = read_pixel(*images["indirect"], pixel_index)[:3]
            path_rgb = read_pixel(*images["path_trace"], pixel_index)[:3]
            history_weight, = read_pixel(FLOAT, layers["history_weight"], pixel_index)
            history_weights.append(history_weight)
            gather_replay_error = max(gather_replay_error, channel_max_error(filtered_rgb, gather_indirect))

            filter_coefficient = sum(weight * source_coefficients[probe] for probe, weight in taps)
            raw_coefficient = sum(weight for probe, weight in taps if probe == source_index)
            cosine = cosines[SOURCE_LANE_INDEX]
            tagged_unfiltered = (0.0, 0.0, 0.0)
            removed_terms, blocker = blocked_emitter_terms(triangles, receiver, normal, diffuse, taps, cosines,
                                                           replay.paths, replay.sources, pinned)
            tagged_filtered = removed_terms.get(pinned, (0.0, 0.0, 0.0))
            if pinned in removed_terms:
                sample_weight = min(1.0, cosine) * 2.0 / sum(c > 0.0 for c in cosines)
                material_weighted = tuple(s * d * sample_weight for s, d in zip(source_rgb, diffuse))
                tagged_unfiltered = multiply(material_weighted, raw_coefficient)
                raw_blocked_count += raw_coefficient > 0.0
                filtered_blocked_count += 1
                source_taps.update(probe for probe, _ in taps if source_coefficients[probe] > 0.0)
            verified_removed_rgb = tuple(math.fsum(rgb[c] for rgb in removed_terms.values()) for c in range(3))
            for source, rgb in removed_terms.items():
                blocked_sources[source] += luminance(rgb)
            verified_blocked_pixels += bool(removed_terms)
            raw_removed_y.append(luminance(tagged_unfiltered))
            removed_y.append(luminance(tagged_filtered))
            verified_removed_y.append(luminance(verified_removed_rgb))
            stages = {
                "path_trace": path_rgb, "raw_integrated_indirect": raw_rgb,
                "filtered_integrated_indirect": filtered_rgb, "gather_indirect": gather_indirect,
                "gather_color": gather_rgb, "g_pinned_only_gather_color": subtract(gather_rgb, tagged_filtered),
                "g_counterfactual_gather_color": subtract(gather_rgb, verified_removed_rgb),
                "temporal_indirect": temporal_indirect, "temporal_color": temporal_rgb,
            }
            for stage, rgb in stages.items():
                rgb_samples.setdefault(stage, []).append(rgb)
            if name == "tall_contact":
                tall_contact_pixels.append({"pixel": [x, y], "first_hit": blocker["object"] if blocker else None,
                    "y": {stage: luminance(rgb) for stage, rgb in stages.items()}, "filtered_removed_y": removed_y[-1],
                    "all_verified_removed_y": verified_removed_y[-1]})
            if (x, y) == RECEIVER_PIXEL:
                witness = {
                    "receiver_pixel": RECEIVER_PIXEL, "receiver_position": receiver[:3],
                    "receiver_to_emitter_first_hit": blocker, "history_weight": history_weight,
                    "raw_source_coefficient": raw_coefficient, "filtered_source_coefficient": filter_coefficient,
                    "raw_removed_rgb": tagged_unfiltered, "filtered_removed_rgb": tagged_filtered,
                    "all_verified_removed_rgb": verified_removed_rgb,
                    "rgb": stages, "y": {stage: luminance(rgb) for stage, rgb in stages.items()},
                    "taps": [{"probe_index": probe, "probe_pixel": pixels[probe], "weight": weight,
                        "raw_lane_rgb": read_pixel(FLOAT4, layers["raw_radiance"], probe * RAY_COUNT + SOURCE_LANE_INDEX)[:3],
                        "filtered_lane_rgb": read_pixel(FLOAT4, layers["filtered_radiance"], probe * RAY_COUNT + SOURCE_LANE_INDEX)[:3],
                        "filtered_source_coefficient": source_coefficients[probe]}
                        for probe, weight in taps],
                }
        statistics[name] = {stage: summarize_region(data) for stage, data in rgb_samples.items()}
        region_deltas[name] = {
            "pixel_count": len(removed_y), "raw_blocked_pixels": raw_blocked_count,
            "filtered_blocked_pixels": filtered_blocked_count, "filtered_blocked_tap_count": len(source_taps),
            "mean_history_weight": math.fsum(history_weights) / len(history_weights),
            "raw_removed_y": math.fsum(raw_removed_y) / len(raw_removed_y),
            "filtered_removed_y": math.fsum(removed_y) / len(removed_y),
            "all_verified_blocked_pixels": verified_blocked_pixels,
            "all_verified_removed_y": math.fsum(verified_removed_y) / len(verified_removed_y),
            "blocked_emitters": [{"probe_pixel": pixels[probe], "lane": (lane % RAY_EDGE, lane // RAY_EDGE),
                "trace_class": replay.sources[lane][probe][2], "mean_removed_y": total / len(coords)}
                for (lane, probe), total in blocked_sources.most_common()],
            "gather_vs_path_trace": compare_region(rgb_samples["path_trace"], rgb_samples["gather_color"]),
            "g_counterfactual_gather_vs_path_trace": compare_region(
                rgb_samples["path_trace"], rgb_samples["g_counterfactual_gather_color"]),
        }

    dark = dark_summary(root, captures["off"], pixel_count, ray_count)
    checks = {
        "gpu_spatial_filter_replayed": replay.filter_error <= FILTER_REPLAY_TOLERANCE,
        "gpu_gather_integration_replayed": gather_replay_error <= GATHER_REPLAY_TOLERANCE,
        "pinned_receiver_first_hit_tall_block": bool(witness and
            witness["receiver_to_emitter_first_hit"] and witness["receiver_to_emitter_first_hit"]["object"] == "Tall Block"),
        "pinned_receiver_removed_emitter_energy": bool(witness and max(witness["filtered_removed_rgb"]) > 0.0),
        "dark_exact_zero_rgb": all(value["max_abs_rgb"] == 0.0 and value["nonfinite_rgb"] == 0 for value in dark.values()),
    }
    pairing = None
    if baseline_path is not None:
        pairing = compare_baseline(baseline_path, manifest_path, manifest, pixel_count, source_index, roi_probes)
        checks["f_guides_and_trace_match"] = all(same for name, same in pairing["ray_readbacks_same_bytes"].items()
                                                if name not in ("radiance", "distance"))
        checks["f_pinned_ray_matches"] = all(pairing["pinned_ray_match"].values())
        checks["f_trace_classifications_match"] = pairing["raw_ray_variation"]["whole_buffer"]["classification_changed_rays"] == 0
        checks["f_path_trace_matches"] = all(images["path_trace"]["same_bytes"] for images in pairing["captures"].values())
    result = {
        "capture": str(manifest_path), "scene": str(scene_path), "frame": frame,
        "gdf_voxel_size_m": capture_gdf_voxel_size(manifest),
        "path_samples": capture["path_samples"], "roi_extents": regions,
        "source": {"probe_pixel": SOURCE_PIXEL, "tile_and_slot": SOURCE_SLOT, "lane": SOURCE_LANE,
            "screen_origin": source_origin, "triangle_first_hit": emitter, "readback": source_ray},
        "emitter_scan": replay.emitter_scan,
        "receiver_witness": witness,
        "replay": {"spatial_lane_max_abs_rgb_error": replay.filter_error,
            "spatial_lane_tolerance": FILTER_REPLAY_TOLERANCE, "gather_roi_max_abs_rgb_error": gather_replay_error,
            "gather_tolerance": GATHER_REPLAY_TOLERANCE},
        "roi_statistics": statistics, "roi_deltas": region_deltas,
        "relative_to_floor": {stage: relative_to_floor({name: stats[stage] for name, stats in statistics.items()})
                              for stage in statistics["tall_contact"]},
        "tall_contact_pixels": tall_contact_pixels, "dark": dark,
        "checks": checks, "passed": all(checks.values()),
    }
    if pairing is not None:
        result["f_pairing"] = pairing
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="F control capture to check instrumentation parity")
    parser.add_argument("--scene", type=Path, default=ROOT / "Asset/Lumen/CornellBox.gltf")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.manifest.resolve(), args.scene.resolve(), args.baseline.resolve() if args.baseline else None)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"Cornell Box contact visibility measurement error: {error}", file=sys.stderr)
        return 2
    if not result["passed"]:
        failed = [name for name, passed in result["checks"].items() if not passed]
        print(f"Cornell Box contact visibility checks failed: {', '.join(failed)}. See {args.out}", file=sys.stderr)
        return 1
    print(f"Cornell Box contact visibility checks passed. See {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
