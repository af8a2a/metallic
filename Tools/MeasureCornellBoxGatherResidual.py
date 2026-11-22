"""Attribute the eight-phase Cornell Box G residual with fixed F trace inputs."""

import argparse
from collections import Counter, defaultdict
from functools import cache
import json
import math
from pathlib import Path
import struct
import sys

from CheckCornellBoxDetailRays import (
    capture_gdf_voxel_size, FLOAT, FLOAT4, GDF_HIT, HALF4, MIN_TRACE_DISTANCE, PROBE_SLOTS, PROBE_TILE_EDGE,
    RAY_COUNT, RAY_EDGE, ROOT, SURFACE_BIAS, add, first_hit, floor_screen_origin, multiply,
    length, normalize, primary_hit, read_buffer, read_triangles, subtract,
)
from CornellBoxEmitterCandidates import infer_screen_sample, review_unverified_emitters
from MeasureCornellBoxContactTemporal import validate_capture_order
from MeasureCornellBoxContactVisibility import (
    FILTER_PASS_COUNT, FILTER_REPLAY_TOLERANCE, GATHER_REPLAY_TOLERANCE, SCREEN_HIT_CLASS, SCREEN_HIT_DISTANCE_TOLERANCE,
    blocked_emitter_terms, build_visibility_replay, channel_max_error, dark_summary, integrate,
    probe_positions, read_pixel, scene_emitter_rgb, spatial_filter_row, visibility_hit,
)
from MeasureCornellBoxGI import luminance, read_pixels, relative_to_floor, summarize_region


CONFIRMED_SCREEN_LABEL = "screen_emitter_texel_without_emitter_first_hit"
G_STAGE_NAMES = ("g_raw_with_f_taps", "g_filter_1_with_f_taps", "g_filter_2_with_f_taps", "g_filter_3_with_f_taps")
TOP_SOURCE_COUNT = 16


def sum_rgb(values):
    return tuple(map(math.fsum, zip(*values))) if values else (0.0, 0.0, 0.0)


def sample_address(probe, lane):
    return probe * RAY_COUNT + lane


def make_spatial_replay(raw, positions, distances, probe_size, directions):
    @cache
    def lane_distances(lane):
        return [read_pixel(FLOAT, distances, sample_address(probe, lane))[0] for probe in range(len(positions))]

    @cache
    def row(probe, lane):
        return spatial_filter_row(probe, positions, lane_distances(lane), probe_size, directions[lane])

    @cache
    def trace(probe, lane):
        return read_pixel(FLOAT4, raw, sample_address(probe, lane))

    @cache
    def filtered(level, probe, lane):
        if level == 0:
            rgb = trace(probe, lane)[:3]
            return rgb, {probe: 1.0} if any(rgb) else {}
        # Each spatial pass is linear at the captured hit distances. Preserve
        # the raw probe identity while following the same normalized rows.
        rgb_terms = []
        paths = defaultdict(float)
        for neighbor, weight in row(probe, lane):
            rgb, previous = filtered(level - 1, neighbor, lane)
            rgb_terms.append(multiply(rgb, weight))
            for source, coefficient in previous.items():
                paths[source] += coefficient * weight
        return sum_rgb(rgb_terms), paths

    return filtered, trace


def integrate_stages(filtered, trace, pixel_inputs, triangles, replay, candidates, source_y):
    receiver, normal, diffuse, taps, cosines = pixel_inputs
    hemisphere = [(lane, min(cosine, 1.0)) for lane, cosine in enumerate(cosines) if cosine > 0.0]
    dominant = max(range(len(taps)), key=lambda index: taps[index][1])
    stages = [[] for _ in G_STAGE_NAMES]
    dominant_terms = []
    candidate_rgb = defaultdict(list)
    filter_error = 0.0
    blocked = {}
    for lane, cosine in hemisphere:
        response = multiply(diffuse, cosine * 2.0 / len(hemisphere))
        emitters = replay.sources.get(lane, {})
        for tap, (probe, tap_weight) in enumerate(taps):
            scale = multiply(response, tap_weight)
            for level in range(FILTER_PASS_COUNT + 1):
                base, paths = filtered(level, probe, lane)
                visible = []
                for source, coefficient in paths.items():
                    key = lane, source
                    if source in emitters:
                        if key not in blocked:
                            endpoint = emitters[source][1]
                            hit = visibility_hit(triangles, receiver, normal, endpoint)
                            blocked[key] = bool(hit and hit["object"] == "Tall Block")
                        if blocked[key]:
                            continue
                    value = multiply(trace(source, lane)[:3], coefficient)
                    visible.append(value)
                    if level == FILTER_PASS_COUNT and (key in candidates or source_y is not None):
                        term = tuple(value[channel] * scale[channel] for channel in range(3))
                        if source_y is not None:
                            source_y[key] += luminance(term)
                        if key in candidates:
                            candidate_rgb[key].append(term)
                radiance = sum_rgb(visible)
                stages[level].append(tuple(radiance[channel] * scale[channel] for channel in range(3)))
                if level == FILTER_PASS_COUNT:
                    actual = read_pixel(FLOAT4, replay.layers["filtered_radiance"], sample_address(probe, lane))[:3]
                    filter_error = max(filter_error, channel_max_error(base, actual))
                    if tap == dominant:
                        dominant_terms.append(tuple(radiance[channel] * response[channel] for channel in range(3)))
    return [sum_rgb(rgb) for rgb in stages], sum_rgb(dominant_terms), {key: sum_rgb(rgb) for key, rgb in candidate_rgb.items()}, filter_error


def summarize_tall_source(key, receivers, count, trace, distances, positions, replay, manifest, triangles, candidates):
    lane, probe = key
    pixel = replay.probe_pixels[probe]
    x, y = pixel
    width = manifest["resolution"][0]
    position = positions[probe][:3]
    normal = normalize(read_pixel(HALF4, replay.layers["normal"], y * width + x)[:3])
    direction = replay.directions[lane]
    trace_rgb = trace(probe, lane)
    distance = read_pixel(FLOAT, distances, sample_address(probe, lane))[0]
    hit = None
    distance_delta = None
    matched = False
    if trace_rgb[3] in (SCREEN_HIT_CLASS, GDF_HIT):
        is_screen = trace_rgb[3] == SCREEN_HIT_CLASS
        origin = (floor_screen_origin(position, normal, manifest) if is_screen else
                  add(add(position, multiply(direction, SURFACE_BIAS)), multiply(normal, SURFACE_BIAS)))
        hit = first_hit(triangles, origin, direction, MIN_TRACE_DISTANCE, manifest["view"]["camera"]["zfar"])
        if hit:
            expected = length(subtract(hit["position"], position)) if is_screen else hit["distance"]
            distance_delta = distance - expected
            matched = abs(distance_delta) <= (SCREEN_HIT_DISTANCE_TOLERANCE if is_screen else capture_gdf_voxel_size(manifest))
    groups = Counter()
    blocked_pixels = 0
    for receiver, receiver_normal, value in receivers:
        if key in candidates:
            group = "reviewed_emitter_candidate"
        elif not matched:
            group = "unresolved_first_hit_or_distance"
        else:
            blocker = visibility_hit(triangles, receiver, receiver_normal, hit["position"])
            tall_blocks = bool(blocker and blocker["object"] == "Tall Block")
            blocked_pixels += tall_blocks
            if hit["emissive"]:
                group = "matched_emitter_blocked" if tall_blocks else "matched_emitter_without_tall_block"
            elif tall_blocks:
                group = "nonemitter_blocked_by_tall_block"
            elif blocker:
                group = "nonemitter_blocked_by_other_surface"
            else:
                group = "nonemitter_visible"
        groups[group] += value / count
    return {"probe_pixel": pixel, "lane": [lane % RAY_EDGE, lane // RAY_EDGE], "raw_radiance_and_class": trace_rgb,
            "trace_distance_m": distance, "triangle_first_hit": hit, "distance_minus_triangle_m": distance_delta,
            "distance_matches_triangle": matched, "mean_y": math.fsum(groups.values()), "receiver_pixels": len(receivers),
            "receiver_to_triangle_endpoint_tall_block_pixels": blocked_pixels, "mean_y_by_geometry": groups}


def right_wall_history_witness(source, manifest, triangles, previous_image, reference, hzb, positions, replay):
    (lane, probe), witness = source
    width, height = manifest["resolution"]
    x, y = replay.probe_pixels[probe]
    receiver = positions[probe][:3]
    normal = normalize(read_pixel(HALF4, replay.layers["normal"], y * width + x)[:3])
    sample = infer_screen_sample(manifest, triangles, receiver, floor_screen_origin(receiver, normal, manifest),
                                 replay.directions[lane], witness["trace_distance_m"], witness["raw_radiance_and_class"][:3],
                                 previous_image, hzb)
    if not sample or not sample["texel_camera_triangle_first_hit"] or sample["texel_camera_triangle_first_hit"]["object"] != "Right Wall":
        return None
    x, y = sample["point_texel"]
    patch = []
    for py in range(max(y - 2, 0), min(y + 3, height)):
        for px in range(max(x - 2, 0), min(x + 3, width)):
            hit = primary_hit(triangles, (px, py), (width, height), manifest["view"]["camera"])
            if hit and hit["object"] == "Right Wall":
                patch.append(read_pixel(*reference, py * width + px)[:3])
    return {"probe_pixel": witness["probe_pixel"], "lane": witness["lane"], "source_mean_y_in_tall_roi": witness["mean_y"],
            "previous_sample": sample, "previous_y": luminance(sample["previous_rgb"]),
            "previous_rgb_max_abs_delta_to_ray": channel_max_error(sample["previous_rgb"], witness["raw_radiance_and_class"][:3]),
            "path_trace_patch_mean_y": summarize_region(patch)["mean_y"], "patch_pixels": len(patch)}


def measure_phase(manifest, root, record, previous, triangles, emitter_rgb):
    capture, rays = record["capture"], record["ray_capture"]
    frame = capture["frame"]
    width, height = manifest["resolution"]
    pixel_count = width * height
    probe_size = ((width + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE, (height + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE)
    probe_count = math.prod(probe_size) * PROBE_SLOTS
    replay = build_visibility_replay(manifest, root, capture, rays, triangles, emitter_rgb)
    _, positions = probe_positions(root, rays, replay.layers["position"], width, probe_count)
    distances = read_buffer(root, rays["distance"], FLOAT, probe_count * RAY_COUNT)
    filtered, trace = make_spatial_replay(replay.layers["raw_radiance"], positions, distances, probe_size, replay.directions)
    candidates = review_unverified_emitters(manifest, root, record, previous, replay, triangles, emitter_rgb)
    direct = read_pixels(root, capture["contact_visibility"]["direct_color"], frame, pixel_count)
    reference = read_pixels(root, capture["path_trace"], frame, pixel_count)
    result = {}
    filter_error = 0.0
    gather_error = 0.0
    total_error = 0.0
    tall_receivers = defaultdict(list)
    for name, coords in replay.region_pixels.items():
        stages = defaultdict(list)
        examined_y = Counter()
        for x, y in coords:
            pixel = x, y
            index = y * width + x
            receiver, normal, diffuse, taps, cosines = inputs = replay.pixel_inputs(index, pixel)
            source_y = Counter() if name == "tall_contact" else None
            g_stages, dominant, unknown, pixel_error = integrate_stages(filtered, trace, inputs, triangles, replay, candidates, source_y)
            filter_error = max(filter_error, pixel_error)
            if source_y is not None:
                for key, value in source_y.items():
                    tall_receivers[key].append((receiver, normal, value))
            scene = read_pixel(*direct, index)[:3]
            current = integrate(taps, cosines, replay.layers["filtered_radiance"], diffuse)
            raw = integrate(taps, cosines, replay.layers["raw_radiance"], diffuse)
            gather_error = max(gather_error, channel_max_error(current, read_pixel(HALF4, replay.layers["gather_indirect"], index)[:3]))
            removed, _ = blocked_emitter_terms(triangles, receiver, normal, diffuse, taps, cosines, replay.paths, replay.sources, None)
            g_color = subtract(read_pixel(FLOAT4, replay.layers["gather_color"], index)[:3], sum_rgb(list(removed.values())))
            total_error = max(total_error, channel_max_error(add(g_stages[-1], scene), g_color))
            for stage, rgb in zip(G_STAGE_NAMES, g_stages):
                stages[stage].append(add(rgb, scene))
            stages["direct"].append(scene)
            for key, rgb in unknown.items():
                examined_y[key] += luminance(rgb)
            confirmed = sum_rgb([rgb for key, rgb in unknown.items() if candidates[key]["classification"] == CONFIRMED_SCREEN_LABEL])
            stages["path_trace"].append(read_pixel(*reference, index)[:3])
            stages["f_raw_with_f_taps"].append(add(raw, scene))
            stages["f_gather_capture"].append(read_pixel(FLOAT4, replay.layers["gather_color"], index)[:3])
            stages["g_gather"].append(g_color)
            stages["g_filtered_dominant_tap"].append(add(dominant, scene))
            stages["g_without_confirmed_screen_candidates"].append(subtract(g_color, confirmed))
            stages["g_without_any_unverified_candidate"].append(subtract(g_color, sum_rgb(list(unknown.values()))))
        for key, y in examined_y.items():
            candidates[key].setdefault("filtered_mean_y_by_roi", {})[name] = y / len(coords)
        result[name] = {stage: summarize_region(values) for stage, values in stages.items()}
    tall_count = len(replay.region_pixels["tall_contact"])
    tall_sources = [(key, summarize_tall_source(key, receivers, tall_count, trace, distances, positions, replay,
                                                manifest, triangles, candidates)) for key, receivers in tall_receivers.items()]
    ordered = sorted(tall_sources, key=lambda pair: pair[1]["mean_y"], reverse=True)
    geometry_y = Counter()
    for _, source in ordered:
        geometry_y.update(source["mean_y_by_geometry"])
    top_sources = [source for _, source in ordered[:TOP_SOURCE_COUNT]]
    blocked_group = "nonemitter_blocked_by_tall_block"
    blocked_sources = sorted((source for _, source in ordered if source["mean_y_by_geometry"][blocked_group] > 0.0),
                             key=lambda source: source["mean_y_by_geometry"][blocked_group], reverse=True)[:TOP_SOURCE_COUNT]
    strongest_wall = next((pair for pair in ordered if pair[1]["raw_radiance_and_class"][3] == SCREEN_HIT_CLASS
                           and pair[1]["distance_matches_triangle"] and pair[1]["triangle_first_hit"]["object"] == "Right Wall"
                           and pair[1]["mean_y_by_geometry"]["nonemitter_visible"] > 0.0), None)
    wall = None
    if strongest_wall:
        previous_image = read_pixels(root, previous["realtime"], frame - 1, pixel_count)
        previous_reference = read_pixels(root, previous["path_trace"], frame - 1, pixel_count)
        hzb = (root / rays["closest_hzb"]).read_bytes()
        wall = right_wall_history_witness(strongest_wall, manifest, triangles, previous_image, previous_reference, hzb, positions, replay)
    # Drop the sparse caches before opening another phase's readbacks.
    filtered.cache_clear()
    trace.cache_clear()
    return {"frame": frame, "phase": frame % 8, "roi": result, "unverified_emitters": list(candidates.values()),
            "unverified_counts": replay.emitter_scan["unverified_samples_by_triangle_hit"], "tall_sources": top_sources,
            "tall_source_geometry_mean_y": geometry_y, "tall_blocked_nonemitter_sources": blocked_sources,
            "right_wall_history_witness": wall,
            "filter_rgb_replay_error": filter_error, "gather_fp16_replay_error": gather_error,
            "corrected_gather_rgb_replay_error": total_error}


def measure(manifest_path, scene_path, steady_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    seed, phases = validate_capture_order(manifest)
    root = manifest_path.parent
    steady = json.loads(steady_path.read_text(encoding="utf-8-sig"))
    if Path(steady["capture"]) != manifest_path or not steady["checks"]["steady_fp16_period_and_seed_decay"]:
        raise ValueError("Expected a converged G report from this exact eight-phase capture")
    triangles = read_triangles(scene_path)
    emitter_rgb = scene_emitter_rgb(scene_path)
    measured = []
    previous = seed
    for record, known in zip(phases, steady["frames"]):
        phase = measure_phase(manifest, root, record, previous, triangles, emitter_rgb)
        if phase["frame"] != known["frame"]:
            raise ValueError("G steady report and capture have different phase executions")
        phase["g_gather_vs_previous_replay_max_abs_roi_y"] = max(
            abs(stages["g_gather"]["mean_y"] - known["roi"]["g_gather"][name]["mean_y"])
            for name, stages in phase["roi"].items())
        measured.append(phase)
        previous = record["capture"]
    averages = {name: {stage: summarize_region([phase["roi"][name][stage]["mean_rgb"] for phase in measured])
                       for stage in stages} for name, stages in measured[0]["roi"].items()}
    deltas = {}
    for name, stages in averages.items():
        y = {stage: stats["mean_y"] for stage, stats in stages.items()}
        deltas[name] = {"raw_gather_minus_path_trace_y": y[G_STAGE_NAMES[0]] - y["path_trace"],
                        "gather_minus_path_trace_y": y["g_gather"] - y["path_trace"],
                        "spatial_filter_net_y": y["g_gather"] - y[G_STAGE_NAMES[0]],
                        "multi_tap_minus_dominant_y": y["g_gather"] - y["g_filtered_dominant_tap"],
                        "confirmed_screen_candidate_y": y["g_gather"] - y["g_without_confirmed_screen_candidates"],
                        "any_unverified_candidate_y": y["g_gather"] - y["g_without_any_unverified_candidate"]}
    width, height = manifest["resolution"]
    ray_count = math.ceil(width / PROBE_TILE_EDGE) * math.ceil(height / PROBE_TILE_EDGE) * PROBE_SLOTS * RAY_COUNT
    dark = {name: dark_summary(root, manifest["captures"][name], width * height, ray_count) for name in ("first_off", "off")}
    geometry_y = Counter()
    for phase in measured:
        geometry_y.update(phase["tall_source_geometry_mean_y"])
    geometry_y = {key: value / len(measured) for key, value in geometry_y.items()}
    checks = {
        "spatial_filter_replays_gpu": all(p["filter_rgb_replay_error"] <= FILTER_REPLAY_TOLERANCE for p in measured),
        "gather_replays_gpu": all(p["gather_fp16_replay_error"] <= GATHER_REPLAY_TOLERANCE for p in measured),
        "gather_source_attribution_matches_captured_integration": all(p["corrected_gather_rgb_replay_error"] <= GATHER_REPLAY_TOLERANCE
                                                                      for p in measured),
        "g_visibility_matches_steady_replay": all(p["g_gather_vs_previous_replay_max_abs_roi_y"] <= 1.0e-8 for p in measured),
        "tall_source_geometry_sums_to_replayed_indirect": all(
            abs(math.fsum(p["tall_source_geometry_mean_y"].values()) + p["roi"]["tall_contact"]["direct"]["mean_y"] -
                p["roi"]["tall_contact"][G_STAGE_NAMES[-1]]["mean_y"]) <= 1.0e-8 for p in measured),
        "dark_rgb_exact_zero": all(v["max_abs_rgb"] == 0.0 and v["nonfinite_rgb"] == 0 for frame in dark.values() for v in frame.values()),
        "steady_history_dark_reset": steady["steady"]["first_dark_after_steady_max_abs_roi_rgb"] == 0.0,
    }
    return {"capture": str(manifest_path), "steady_report": str(steady_path), "gdf_voxel_size_m": capture_gdf_voxel_size(manifest),
            "phases": measured, "mean_roi": averages,
            "mean_relative_to_floor": {stage: relative_to_floor({name: values[stage] for name, values in averages.items()})
                                       for stage in next(iter(averages.values()))}, "roi_deltas": deltas,
            "tall_source_geometry_mean_y": geometry_y, "steady_temporal_mean_roi": steady["steady"]["mean_roi"],
            "dark": dark, "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--steady", required=True, type=Path, help="converged Temporal report from the same capture")
    parser.add_argument("--scene", type=Path, default=ROOT / "Asset/Lumen/CornellBox.gltf")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.manifest.resolve(), args.scene.resolve(), args.steady.resolve())
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"Cornell Box Gather attribution error: {error}", file=sys.stderr)
        return 2
    print(f"Cornell Box Gather attribution {'passed' if result['passed'] else 'failed'}. See {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
