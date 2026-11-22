"""Replay eight Cornell Box probe phases with the captured F Temporal state.

Verify the fixed-camera history footprint and the unmodified F recurrence.
Start the G replay from the same measured seed and remove the triangle-verified
emitter contributions from each Gather input. Keep F's traces and weights fixed.
With --steady, repeat the corrected phases to a periodic FP16 history state.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys

from CheckCornellBoxDetailRays import (
    FLOAT, FLOAT4, HALF4, PROBE_POSITION, PROBE_SLOTS, PROBE_TILE_EDGE, RAY_COUNT, ROOT, floor_screen_pixel,
)
from MeasureCornellBoxContactVisibility import (
    FILTER_REPLAY_TOLERANCE, GATHER_REPLAY_TOLERANCE, contact_buffer, dark_summary,
    measure_phase_visibility, read_pixel, region_coordinates,
)
from MeasureCornellBoxGI import REGION_NAMES, luminance, read_pixels, relative_to_floor, summarize_region


PHASE_COUNT = 8
WITNESS = (286, 322)
HISTORY_REPLAY_TOLERANCE = 5.0e-6
STEADY_SEED_WEIGHT_LIMIT = 1.0e-6
MAX_STEADY_CYCLES = 256
HALF = struct.Struct("<e")
HALF_BITS = struct.Struct("<H")
STAGES = ("path_trace", "f_gather", "f_temporal", "g_gather", "g_temporal")
EXCESS_TERMS = ("current_input_y", "previous_history_y")


def quantize_half(value, toward_zero):
    encoded = HALF.pack(value)
    nearest, = HALF.unpack(encoded)
    if toward_zero and abs(nearest) > abs(value):
        bits, = HALF_BITS.unpack(encoded)
        return HALF.unpack(HALF_BITS.pack(bits - 1))[0]
    return nearest


def float32(value):
    return FLOAT.unpack(FLOAT.pack(value))[0]


def blend(current, history, weight):
    # Temporal blends in FP32 before writing the FP16 history image.
    return float32(history + (current - history) * (1.0 - weight))


def apply_temporal(gather, previous, weight, toward_zero):
    indirect = tuple(blend(current, history, weight) for current, history in zip(gather, previous))
    return indirect, tuple(quantize_half(value, toward_zero) for value in indirect)


def check_same_pixel_history(point, pixel, manifest):
    projected = floor_screen_pixel(point, manifest)
    # The fixed, jitter-free camera must decode to one exact previous texel.
    # The five ROIs are interior, so the footprint's edge guard has no effect.
    for expected, coord, extent in zip(pixel, projected, manifest["resolution"]):
        subpixel_grid = 1 << ((65535 // extent).bit_length() - 1)
        if math.floor(coord * subpixel_grid + 0.5) != expected * subpixel_grid:
            raise ValueError(f"Temporal reprojects pixel {pixel} to a different footprint at {projected}")


def validate_capture_order(manifest):
    records = manifest["temporal_phases"]
    seed = manifest["temporal_seed"]
    start = seed["frame"] + 1
    frames = [record["capture"]["frame"] for record in records]
    if len(records) != PHASE_COUNT or frames != list(range(start, start + PHASE_COUNT)):
        raise ValueError("Expected eight consecutive Temporal phases after the seed")
    if manifest["view"]["temporalJitter"] or manifest["view"]["camera"]["projection"] != "perspective":
        raise ValueError("Temporal phase replay requires the fixed jitter-free perspective camera")
    if manifest["captures"]["on"]["frame"] != frames[-1]:
        raise ValueError("The regular lit capture must be the final phase")
    return seed, records


def phase_data(root, record, manifest, regions, pixel_count):
    capture = record["capture"]
    frame = capture["frame"]
    visibility = capture["contact_visibility"]
    rays = record["ray_capture"]
    stamps = capture["evidence"] + visibility["evidence"] + rays["evidence"]
    if any(stamp["execution"] != frame for stamp in stamps):
        raise ValueError(f"Execution {frame} contains readbacks from another graph execution")
    render_frame = stamps[0]["provenance"]["renderFrame"]
    phase = render_frame % PHASE_COUNT

    width = manifest["resolution"][0]
    grid_width = (width + PROBE_TILE_EDGE - 1) // PROBE_TILE_EDGE
    # This uniform probe is over the central floor in all eight phases.
    probe_index = (20 * grid_width + 12) * PROBE_SLOTS
    probe_bytes = (root / rays["probe_positions"]).read_bytes()
    probe_pixel = read_pixel(PROBE_POSITION, probe_bytes, probe_index)
    jitter_y = (0, 8, 4, 12, 2, 10, 6, 14)
    expected_probe = (12 * PROBE_TILE_EDGE + 2 * phase, 20 * PROBE_TILE_EDGE + jitter_y[phase])
    if probe_pixel != expected_probe:
        raise ValueError(f"Probe layout for phase {phase} does not match the recorded render frame")

    gather = contact_buffer(root, capture, "gather_indirect", pixel_count)
    gather_color = contact_buffer(root, capture, "gather_color", pixel_count)
    weight = contact_buffer(root, capture, "history_weight", pixel_count)
    position = contact_buffer(root, capture, "position", pixel_count)
    direct_color = visibility["direct_color"]
    direct_layout, direct = read_pixels(root, direct_color, frame, pixel_count)
    temporal_layout, temporal = read_pixels(root, capture["indirect"], frame, pixel_count)
    temporal_color_layout, temporal_color = read_pixels(root, capture["realtime"], frame, pixel_count)
    reference_layout, reference = read_pixels(root, capture["path_trace"], frame, pixel_count)
    by_pixel = {}
    for pixel in dict.fromkeys(pixel for coords in regions.values() for pixel in coords):
        x, y = pixel
        index = y * width + x
        history_weight, = read_pixel(FLOAT, weight, index)
        if not 0.0 <= history_weight <= 1.0:
            raise ValueError(f"Invalid Temporal history weight at pixel {pixel}")
        if history_weight > 0.0:
            check_same_pixel_history(read_pixel(FLOAT4, position, index)[:3], pixel, manifest)
        by_pixel[pixel] = {
            "gather_indirect": read_pixel(HALF4, gather, index)[:3],
            "gather_color": read_pixel(FLOAT4, gather_color, index)[:3],
            "direct": read_pixel(direct_layout, direct, index)[:3],
            "temporal_indirect": read_pixel(temporal_layout, temporal, index)[:3],
            "temporal_color": read_pixel(temporal_color_layout, temporal_color, index)[:3],
            "path_trace": read_pixel(reference_layout, reference, index)[:3],
            "weight": history_weight,
        }
    return {"frame": frame, "phase": phase, "uniform_probe_pixel": probe_pixel,
            "path_samples": capture["path_samples"], "pixels": by_pixel}


def observed_half_rounding(phases):
    matches = {"nearest_even": 0, "toward_zero": 0}
    for phase in phases:
        for sample in phase["pixels"].values():
            # Gather and Temporal store indirect RGB to FP16 and add the same
            # unquantized indirect RGB to direct color in their FP32 outputs.
            for layer in ("gather", "temporal"):
                full = sample[f"{layer}_color"]
                half = sample[f"{layer}_indirect"]
                for full_value, direct, stored in zip(full, sample["direct"], half):
                    rgb = full_value - direct
                    matches["nearest_even"] += quantize_half(rgb, False) == stored
                    matches["toward_zero"] += quantize_half(rgb, True) == stored
    return max(matches, key=matches.get), matches


def summarize_samples(samples, regions):
    stats = {stage: {name: summarize_region([samples[pixel][stage] for pixel in coords])
                    for name, coords in regions.items()} for stage in STAGES}
    ratios = {stage: relative_to_floor(stats[stage]) for stage in STAGES}
    return stats, ratios


def history_components(stats, regions):
    result = {}
    for name in regions:
        y = {stage: stats[stage][name]["mean_y"] for stage in STAGES}
        result[name] = {"f_temporal_minus_gather_y": y["f_temporal"] - y["f_gather"],
                        "g_temporal_minus_gather_y": y["g_temporal"] - y["g_gather"],
                        "g_gather_minus_path_trace_y": y["g_gather"] - y["path_trace"],
                        "g_temporal_minus_path_trace_y": y["g_temporal"] - y["path_trace"],
                        "temporal_visibility_reduction_y": y["f_temporal"] - y["g_temporal"]}
    return result


def corrected_gather_inputs(phase, removed, toward_zero):
    inputs = {}
    for pixel, sample in phase["pixels"].items():
        # Preserve measured FP16 inputs where the replay has no correction.
        inputs[pixel] = tuple(quantize_half(float32(full - scene - delta), toward_zero) if delta != 0.0 else original
                              for full, scene, delta, original in zip(sample["gather_color"], sample["direct"],
                                                                   removed[pixel], sample["gather_indirect"]))
    return inputs


def replay_phase(phase, removed, corrected, previous_f, previous_g, seed_weight, toward_zero):
    current_f = {}
    current_g = {}
    samples = {}
    replay_error = 0.0
    store_error = 0.0
    for pixel, sample in phase["pixels"].items():
        direct = sample["direct"]
        gather = sample["gather_indirect"]
        weight = sample["weight"]
        seed_weight[pixel] *= weight
        correction = removed[pixel]
        g_gather = corrected[pixel]
        f_temporal, f_history = apply_temporal(gather, previous_f[pixel], weight, toward_zero)
        g_temporal, current_g[pixel] = apply_temporal(g_gather, previous_g[pixel], weight, toward_zero)
        reference = sample["path_trace"]
        current_excess = (1.0 - weight) * luminance(value + scene - ref for value, scene, ref in zip(g_gather, direct, reference))
        history_excess = weight * luminance(value + scene - ref for value, scene, ref in zip(previous_g[pixel], direct, reference))
        current_f[pixel] = sample["temporal_indirect"]
        replay_error = max(replay_error, *(abs(predicted + scene - color)
                                          for predicted, scene, color in zip(f_temporal, direct, sample["temporal_color"])))
        store_error = max(store_error, *(abs(predicted - stored) for predicted, stored in zip(f_history, current_f[pixel])))
        samples[pixel] = {"path_trace": reference, "f_gather": sample["gather_color"],
                          "f_temporal": sample["temporal_color"],
                          "g_gather": tuple(full - delta for full, delta in zip(sample["gather_color"], correction)),
                          "g_temporal": tuple(value + scene for value, scene in zip(g_temporal, direct)),
                          "current_input_y": current_excess, "previous_history_y": history_excess}
    return samples, current_f, current_g, replay_error, store_error


def settle_period(phases, corrected_inputs, history, period_seed_weight, toward_zero):
    first_closed_cycle = None
    for cycle in range(1, MAX_STEADY_CYCLES + 1):
        start = history
        colors = []
        for phase, corrected in zip(phases, corrected_inputs):
            next_history = {}
            output = {}
            for pixel, sample in phase["pixels"].items():
                indirect, next_history[pixel] = apply_temporal(corrected[pixel], history[pixel], sample["weight"], toward_zero)
                output[pixel] = tuple(value + direct for value, direct in zip(indirect, sample["direct"]))
            history = next_history
            colors.append(output)
        seed_weight = period_seed_weight ** cycle
        if start == history:
            if first_closed_cycle is None:
                first_closed_cycle = cycle
            if seed_weight <= STEADY_SEED_WEIGHT_LIMIT:
                break
    return {"cycles": cycle, "first_closed_fp16_cycle": first_closed_cycle, "max_linear_seed_weight": seed_weight,
            "converged": start == history and seed_weight <= STEADY_SEED_WEIGHT_LIMIT}, colors, history


def summarize_steady(phases, corrected_inputs, initial_history, seed_weight, frames, mean_stats, regions, toward_zero):
    # A zero seed isolates any initialization-dependent FP16 rounding plateau.
    period_seed_weight = max(seed_weight.values())
    from_f, f_colors, final_history = settle_period(phases, corrected_inputs, initial_history, period_seed_weight, toward_zero)
    from_zero, zero_colors, _ = settle_period(phases, corrected_inputs, dict.fromkeys(initial_history, (0.0, 0.0, 0.0)),
                                              period_seed_weight, toward_zero)
    stats = []
    for frame, f_color, zero_color in zip(frames, f_colors, zero_colors):
        roi = {name: summarize_region([f_color[pixel] for pixel in pixels]) for name, pixels in regions.items()}
        zero_roi = {name: summarize_region([zero_color[pixel] for pixel in pixels]) for name, pixels in regions.items()}
        stats.append({"phase": frame["phase"], "frame": frame["frame"], "roi": roi, "zero_seed_roi": zero_roi,
                      "relative_to_floor": relative_to_floor(roi), "witness_y": luminance(f_color[WITNESS])})
    settled = {name: summarize_region([phase["roi"][name]["mean_rgb"] for phase in stats]) for name in regions}
    zero_settled = {name: summarize_region([phase["zero_seed_roi"][name]["mean_rgb"] for phase in stats]) for name in regions}
    deltas = {}
    for name, pixels in regions.items():
        y = settled[name]["mean_y"]
        reference = mean_stats["path_trace"][name]["mean_y"]
        gather = mean_stats["g_gather"][name]["mean_y"]
        first_cycle = mean_stats["g_temporal"][name]["mean_y"]
        deltas[name] = {"first_cycle_minus_steady_y": first_cycle - y,
                        "gather_minus_path_trace_y": gather - reference,
                        "steady_minus_gather_y": y - gather, "steady_minus_path_trace_y": y - reference,
                        "steady_minus_f_temporal_y": y - mean_stats["f_temporal"][name]["mean_y"],
                        "f_seed_minus_zero_seed_y": y - zero_settled[name]["mean_y"],
                        "max_abs_seed_rgb_delta": max(abs(a - b) for f_color, zero_color in zip(f_colors, zero_colors)
                                                      for pixel in pixels for a, b in zip(f_color[pixel], zero_color[pixel]))}
    return {"convergence": {"f_seed": from_f, "zero_seed": from_zero, "linear_seed_weight_limit": STEADY_SEED_WEIGHT_LIMIT},
            "phases": stats, "mean_roi": settled, "zero_seed_mean_roi": zero_settled,
            "mean_relative_to_floor": relative_to_floor(settled), "roi_deltas": deltas}, final_history


def replay_first_dark(root, capture, width, pixel_count, history, toward_zero):
    gather = contact_buffer(root, capture, "gather_indirect", pixel_count)
    weight = contact_buffer(root, capture, "history_weight", pixel_count)
    direct_layout, direct = read_pixels(root, capture["contact_visibility"]["direct_color"], capture["frame"], pixel_count)
    maximum = 0.0
    for (x, y), old_history in history.items():
        index = y * width + x
        current = read_pixel(HALF4, gather, index)[:3]
        indirect, _ = apply_temporal(current, old_history, read_pixel(FLOAT, weight, index)[0], toward_zero)
        maximum = max(maximum, *(abs(value + scene) for value, scene in zip(indirect, read_pixel(direct_layout, direct, index)[:3])))
    return maximum


def measure(manifest_path, scene_path, steady=False):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    seed, phase_records = validate_capture_order(manifest)
    root = manifest_path.parent
    width, height = manifest["resolution"]
    pixel_count = width * height
    ray_count = math.ceil(width / PROBE_TILE_EDGE) * math.ceil(height / PROBE_TILE_EDGE) * PROBE_SLOTS * RAY_COUNT
    regions = {name: region_coordinates(manifest["regions"][name]) for name in REGION_NAMES}
    phases = [phase_data(root, record, manifest, regions, pixel_count) for record in phase_records]
    if [phase["phase"] for phase in phases] != list(range(PHASE_COUNT)):
        raise ValueError("Expected probe phases 0 through 7 in execution order")
    rounding, rounding_matches = observed_half_rounding(phases)
    toward_zero = rounding == "toward_zero"
    seed_layout, seed_data = read_pixels(root, seed["indirect"], seed["frame"], pixel_count)
    previous_f = {pixel: read_pixel(seed_layout, seed_data, pixel[1] * width + pixel[0])[:3]
                  for pixel in phases[0]["pixels"]}
    initial_history = previous_f
    previous_g = initial_history
    seed_weight = dict.fromkeys(previous_f, 1.0)
    frames = []
    corrected_inputs = []
    for phase, record in zip(phases, phase_records):
        visibility = measure_phase_visibility(manifest, root, record["capture"], record["ray_capture"], scene_path)
        corrected = corrected_gather_inputs(phase, visibility["removed_rgb"], toward_zero)
        corrected_inputs.append(corrected)
        samples, previous_f, previous_g, error, store_error = replay_phase(
            phase, visibility["removed_rgb"], corrected, previous_f, previous_g, seed_weight, toward_zero)
        stats, ratios = summarize_samples(samples, regions)
        weights = {name: [phase["pixels"][pixel]["weight"] for pixel in coords] for name, coords in regions.items()}
        frames.append({"frame": phase["frame"], "phase": phase["phase"], "path_samples": phase["path_samples"],
                       "uniform_probe_pixel": phase["uniform_probe_pixel"],
                       "roi": stats, "relative_to_floor": ratios, "history_components": history_components(stats, regions),
                       "history_weights": {name: {"min": min(values), "max": max(values), "mean": math.fsum(values) / len(values)}
                                           for name, values in weights.items()},
                       "seed_history_weight": {name: math.fsum(seed_weight[pixel] for pixel in coords) / len(coords)
                                               for name, coords in regions.items()},
                       "g_temporal_residual_terms": {name: {term: math.fsum(samples[pixel][term] for pixel in coords) / len(coords)
                                                           for term in EXCESS_TERMS} for name, coords in regions.items()},
                       "witness_y": {stage: luminance(samples[WITNESS][stage]) for stage in STAGES},
                       "f_temporal_rgb_replay_error": error, "f_temporal_fp16_replay_error": store_error,
                       "visibility": {key: value for key, value in visibility.items() if key != "removed_rgb"}})
    mean_stats = {stage: {name: summarize_region([frame["roi"][stage][name]["mean_rgb"] for frame in frames])
                         for name in regions} for stage in STAGES}
    mean_terms = {name: {term: math.fsum(frame["g_temporal_residual_terms"][name][term] for frame in frames) / len(frames)
                         for term in EXCESS_TERMS} for name in regions}
    dark = {}
    for name in ("first_off", "off"):
        capture = manifest["captures"][name]
        summary = dark_summary(root, capture, pixel_count, ray_count)
        weight = contact_buffer(root, capture, "history_weight", pixel_count)
        dark[name] = {"frame": capture["frame"], "images": summary,
                      "max_history_weight": max(value for value, in FLOAT.iter_unpack(weight))}
    checks = {
        "f_temporal_rgb_replay": all(frame["f_temporal_rgb_replay_error"] <= HISTORY_REPLAY_TOLERANCE for frame in frames),
        "f_temporal_fp16_history_replay": all(frame["f_temporal_fp16_replay_error"] == 0.0 for frame in frames),
        "g_spatial_filter_replay": all(frame["visibility"]["filter_replay_max_abs_rgb_error"] <= FILTER_REPLAY_TOLERANCE
                                       for frame in frames),
        "g_gather_replay": all(frame["visibility"]["gather_replay_max_abs_rgb_error"] <= GATHER_REPLAY_TOLERANCE
                              for frame in frames),
        "dark_exact_zero_rgb": all(layer["max_abs_rgb"] == 0.0 and layer["nonfinite_rgb"] == 0
                                  for capture in dark.values() for layer in capture["images"].values()),
        "emitter_edit_resets_history": dark["first_off"]["max_history_weight"] == 0.0,
    }
    result = {"capture": str(manifest_path), "seed_frame": seed["frame"], "roi_extents": manifest["regions"],
              "fp16_rounding": rounding, "fp16_rounding_matches": rounding_matches, "frames": frames,
              "mean_roi": mean_stats, "mean_relative_to_floor": {stage: relative_to_floor(mean_stats[stage]) for stage in STAGES},
              "mean_history_components": history_components(mean_stats, regions), "mean_g_temporal_residual_terms": mean_terms,
              "dark": dark, "checks": checks}
    if steady:
        settled, history = summarize_steady(phases, corrected_inputs, initial_history, seed_weight, frames, mean_stats,
                                            regions, toward_zero)
        dark_max = replay_first_dark(root, manifest["captures"]["first_off"], width, pixel_count, history, toward_zero)
        settled["first_dark_after_steady_max_abs_roi_rgb"] = dark_max
        result["steady"] = settled
        checks["steady_fp16_period_and_seed_decay"] = all(settled["convergence"][key]["converged"]
                                                        for key in ("f_seed", "zero_seed"))
        checks["steady_history_resets_in_first_dark_frame"] = dark_max == 0.0
    result["passed"] = all(checks.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scene", type=Path, default=ROOT / "Asset/Lumen/CornellBox.gltf")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steady", action="store_true", help="repeat the corrected phases to steady Temporal history")
    args = parser.parse_args()
    try:
        result = measure(args.manifest.resolve(), args.scene.resolve(), args.steady)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"Cornell Box Temporal replay error: {error}", file=sys.stderr)
        return 2
    print(f"Cornell Box Temporal replay {'passed' if result['passed'] else 'failed'}. See {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
