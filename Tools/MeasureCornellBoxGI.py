"""Measure the fixed-camera Cornell Box GI baseline from scene-linear readbacks.

The manifest defines a resolution [width, height] and five top-left pixel ROIs
under regions. Each ROI is [x, y, width, height]. captures.on and captures.off
each contain a frame number and path_trace, realtime, and indirect objects with
file, format, and frame fields. File paths resolve relative to the manifest.
Inputs are tightly packed, little-endian RGBA32F or RGBA16F in scene-linear
ACEScg. Alpha is ignored.

Optional prior_on and first_off captures use the same input format. prior_on
measures convergence at a matching screen-probe phase. first_off records
whole-frame residual light immediately after extinguishing the emitter.

Compare two measurements with --baseline, --candidate and --screen-traces.
Each invocation describes one pair with the same screen-trace configuration.
"""

import argparse
import json
import math
from pathlib import Path
import struct
import sys


IMAGE_NAMES = ("path_trace", "realtime", "indirect")
REGION_NAMES = ("floor", "short_contact", "tall_contact", "red_spill", "green_spill")
PIXEL_FORMATS = {"rgba16f": struct.Struct("<4e"), "rgba32f": struct.Struct("<4f")}
# Y row of kAP1ToXYZ in Source/Runtime/Render/Core/ColorSpaceMatrices.h.
AP1_TO_Y = (0.2722287168, 0.6740817658, 0.0536895174)


def luminance(rgb):
    return sum(channel * weight for channel, weight in zip(rgb, AP1_TO_Y))


def read_pixels(root, capture, frame, pixel_count):
    if type(capture["frame"]) is not int or capture["frame"] != frame:
        raise ValueError(f"{capture['file']} did not use capture frame {frame}")
    path = root / capture["file"]
    layout = PIXEL_FORMATS[capture["format"].lower()]
    expected_size = pixel_count * layout.size
    if path.stat().st_size != expected_size:
        raise ValueError(f"{path} must contain {expected_size} tightly packed bytes")
    return layout, memoryview(path.read_bytes())


def summarize_frame(layout, raw):
    rgb_total = [0.0, 0.0, 0.0]
    finite_pixels = 0
    negative_pixels = 0
    nonzero_pixels = 0
    max_abs_rgb = 0.0
    for red, green, blue, _ in layout.iter_unpack(raw):
        if not (math.isfinite(red) and math.isfinite(green) and math.isfinite(blue)):
            continue
        finite_pixels += 1
        negative_pixels += min(red, green, blue) < 0.0
        nonzero_pixels += red != 0.0 or green != 0.0 or blue != 0.0
        max_abs_rgb = max(max_abs_rgb, abs(red), abs(green), abs(blue))
        rgb_total[0] += red
        rgb_total[1] += green
        rgb_total[2] += blue
    mean_rgb = [total / finite_pixels for total in rgb_total] if finite_pixels else None
    return {
        "nonfinite_rgb_pixels": len(raw) // layout.size - finite_pixels,
        "negative_rgb_pixels": negative_pixels,
        "nonzero_rgb_pixels": nonzero_pixels,
        "max_abs_rgb": max_abs_rgb if finite_pixels else None,
        "mean_rgb": mean_rgb,
        "mean_y": luminance(mean_rgb) if mean_rgb is not None else None,
    }


def region_pixels(layout, raw, width, region):
    x, y, roi_width, roi_height = region
    for row in range(y, y + roi_height):
        start = (row * width + x) * layout.size
        stop = start + roi_width * layout.size
        for pixel in layout.iter_unpack(raw[start:stop]):
            yield pixel[:3]


def summarize_region(pixels):
    mean_rgb = [math.fsum(channel) / len(pixels) for channel in zip(*pixels)]
    rgb_sum = math.fsum(mean_rgb)
    return {
        "mean_rgb": mean_rgb,
        "mean_y": luminance(mean_rgb),
        "rgb_fraction": [channel / rgb_sum for channel in mean_rgb] if rgb_sum > 0.0 else None,
    }


def compare_region(path_trace, realtime):
    reference_y = [luminance(rgb) for rgb in path_trace]
    y_delta = [luminance(rgb) - reference for rgb, reference in zip(realtime, reference_y)]
    mean_reference_y = math.fsum(reference_y) / len(reference_y)
    mean_y_delta = math.fsum(y_delta) / len(y_delta)
    return {
        "mean_y_delta": mean_y_delta,
        "mean_y_bias_percent": 100.0 * mean_y_delta / mean_reference_y if mean_reference_y > 0.0 else None,
        "mean_absolute_y_error": math.fsum(abs(value) for value in y_delta) / len(y_delta),
        "y_rmse": math.sqrt(math.fsum(value * value for value in y_delta) / len(y_delta)),
    }


def relative_to_floor(region_stats):
    floor = region_stats["floor"]
    summary = {}
    for region in ("short_contact", "tall_contact"):
        summary[f"{region}_y_over_floor"] = (
            region_stats[region]["mean_y"] / floor["mean_y"] if floor["mean_y"] > 0.0 else None
        )
    for region, channel in (("red_spill", 0), ("green_spill", 1)):
        spill_fraction = region_stats[region]["rgb_fraction"]
        floor_fraction = floor["rgb_fraction"]
        summary[f"{region}_channel_fraction_delta"] = (
            spill_fraction[channel] - floor_fraction[channel]
            if spill_fraction is not None and floor_fraction is not None else None
        )
    return summary


def measure_capture(root, capture, width, height, regions):
    frame = capture["frame"]
    if type(frame) is not int or frame < 0:
        raise ValueError("Capture frame must be a nonnegative integer")
    images = {}
    region_data = {}
    for name in IMAGE_NAMES:
        pixels = read_pixels(root, capture[name], frame, width * height)
        image = {**capture[name], **summarize_frame(*pixels)}
        images[name] = image
        if regions and image["nonfinite_rgb_pixels"] == 0:
            data = {region: list(region_pixels(*pixels, width, bounds)) for region, bounds in regions.items()}
            image["regions"] = {region: summarize_region(values) for region, values in data.items()}
            image["relative_to_floor"] = relative_to_floor(image["regions"])
            if name != "indirect":
                region_data[name] = data
    comparison = {}
    if "path_trace" in region_data and "realtime" in region_data:
        comparison = {
            region: compare_region(region_data["path_trace"][region], region_data["realtime"][region])
            for region in regions
        }
    metadata = {
        key: capture[key] for key in ("frame", "accumulated_frames", "path_samples", "evidence") if key in capture
    }
    return {**metadata, "images": images, "realtime_vs_path_trace": comparison}


def compare_convergence(prior, final, regions):
    phases = []
    for capture in (prior, final):
        stamp = next(iter(capture.get("evidence", [])), {})
        render_frame = stamp.get("provenance", {}).get("renderFrame", capture["frame"])
        # LumenAdaptiveProbes.slang repeats probe placement every eight render frames.
        phases.append(render_frame % 8)
    result = {
        "prior_probe_phase": phases[0],
        "on_probe_phase": phases[1],
        "same_lumen_probe_phase": phases[0] == phases[1],
    }
    if not result["same_lumen_probe_phase"]:
        return result
    images = {}
    for name in ("path_trace", "realtime"):
        before = prior["images"][name].get("regions")
        after = final["images"][name].get("regions")
        if before is None or after is None:
            continue
        images[name] = {}
        for region in regions:
            prior_y = before[region]["mean_y"]
            final_y = after[region]["mean_y"]
            delta = final_y - prior_y
            images[name][region] = {
                "prior_mean_y": prior_y,
                "on_mean_y": final_y,
                "mean_y_delta": delta,
                "mean_y_change_percent": 100.0 * delta / prior_y if prior_y > 0.0 else None,
            }
    result["images"] = images
    return result


def measure(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    width, height = manifest["resolution"]
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Resolution must contain positive integer width and height")
    if manifest.get("color_space", "acescg").lower() != "acescg":
        raise ValueError("Readbacks must use scene-linear ACEScg")
    regions = {name: manifest["regions"][name] for name in REGION_NAMES}
    for name, region in regions.items():
        if len(region) != 4 or any(type(value) is not int for value in region):
            raise ValueError(f"{name} must contain four integers [x, y, width, height]")
        x, y, roi_width, roi_height = region
        if min(x, y) < 0 or min(roi_width, roi_height) <= 0 or x + roi_width > width or y + roi_height > height:
            raise ValueError(f"{name} extends outside the {width}x{height} readback")

    capture_inputs = manifest["captures"]
    cases = {
        name: measure_capture(manifest_path.parent, capture_inputs[name], width, height, regions)
        for name in ("on", "off")
    }
    convergence = None
    if "prior_on" in capture_inputs:
        cases["prior_on"] = measure_capture(manifest_path.parent, capture_inputs["prior_on"], width, height, regions)
        convergence = compare_convergence(cases["prior_on"], cases["on"], regions)
    if "first_off" in capture_inputs:
        cases["first_off"] = measure_capture(manifest_path.parent, capture_inputs["first_off"], width, height, {})
    on_images = cases["on"]["images"]
    off_images = cases["off"]["images"]
    all_images = [*on_images.values(), *off_images.values()]
    checks = {
        "all_rgb_finite": all(image["nonfinite_rgb_pixels"] == 0 for image in all_images),
        "all_rgb_nonnegative": all(image["negative_rgb_pixels"] == 0 for image in all_images),
        "emitter_off_all_rgb_zero": all(
            image["nonfinite_rgb_pixels"] == 0 and image["max_abs_rgb"] == 0.0 for image in off_images.values()
        ),
        "emitter_on_visible_on_receivers": all(
            any(region["mean_y"] > 0.0 for region in image.get("regions", {}).values())
            for image in on_images.values()
        ),
    }
    result = {
        "source_manifest": str(manifest_path),
        "resolution": [width, height],
        "color_space": "acescg",
        "regions": regions,
        "captures": cases,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if convergence is not None:
        result["convergence"] = convergence
    return result


def compare_values(reference, baseline, candidate):
    values = {"path_trace": reference, "baseline": baseline, "candidate": candidate}
    if None not in (reference, baseline, candidate):
        baseline_error = baseline - reference
        candidate_error = candidate - reference
        values.update({
            "baseline_signed_error": baseline_error,
            "candidate_signed_error": candidate_error,
            "candidate_minus_baseline": candidate - baseline,
            "change_in_absolute_error": abs(candidate_error) - abs(baseline_error),
        })
    return values


def compare_measurements(baseline_path, candidate_path, screen_traces):
    baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    if not baseline["passed"] or not candidate["passed"]:
        raise ValueError("Both A/B inputs must pass the Cornell Box capture checks")
    for key in ("resolution", "color_space", "regions"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"A/B capture {key} differs")
    baseline_on = baseline["captures"]["on"]
    candidate_on = candidate["captures"]["on"]
    for key in ("frame", "accumulated_frames", "path_samples"):
        if baseline_on[key] != candidate_on[key]:
            raise ValueError(f"A/B capture {key} differs")

    path_trace = baseline_on["images"]["path_trace"]
    candidate_path_trace = candidate_on["images"]["path_trace"]
    realtime_a = baseline_on["images"]["realtime"]
    realtime_b = candidate_on["images"]["realtime"]
    roi_luminance = {}
    for name in REGION_NAMES:
        values = compare_values(path_trace["regions"][name]["mean_y"],
                                realtime_a["regions"][name]["mean_y"], realtime_b["regions"][name]["mean_y"])
        if values["path_trace"] > 0.0:
            values["baseline_bias_percent"] = 100.0 * values["baseline_signed_error"] / values["path_trace"]
            values["candidate_bias_percent"] = 100.0 * values["candidate_signed_error"] / values["path_trace"]
        roi_luminance[name] = values

    reference_relative = path_trace["relative_to_floor"]
    baseline_relative = realtime_a["relative_to_floor"]
    candidate_relative = realtime_b["relative_to_floor"]
    relative = {
        name: compare_values(value, baseline_relative[name], candidate_relative[name])
        for name, value in reference_relative.items()
    }
    return {
        "baseline_metrics": str(baseline_path),
        "candidate_metrics": str(candidate_path),
        "screen_traces": screen_traces,
        "resolution": baseline["resolution"],
        "color_space": baseline["color_space"],
        "regions": baseline["regions"],
        "frame": baseline_on["frame"],
        "accumulated_frames": baseline_on["accumulated_frames"],
        "path_samples": baseline_on["path_samples"],
        "path_trace_roi_summaries_match": path_trace["regions"] == candidate_path_trace["regions"],
        "candidate_path_trace_mean_y_minus_baseline": {
            name: candidate_path_trace["regions"][name]["mean_y"] - path_trace["regions"][name]["mean_y"]
            for name in REGION_NAMES
        },
        "roi_mean_y": roi_luminance,
        "contact_y_over_floor": {name: relative[f"{name}_y_over_floor"] for name in ("short_contact", "tall_contact")},
        "color_spill_channel_fraction_above_floor": {
            name: relative[f"{name}_channel_fraction_delta"] for name in ("red_spill", "green_spill")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--screen-traces", choices=("on", "off"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline and (not args.candidate or not args.screen_traces):
        parser.error("--baseline requires --candidate and --screen-traces")
    if args.manifest and (args.candidate or args.screen_traces):
        parser.error("--candidate and --screen-traces require --baseline")
    try:
        if args.baseline:
            result = compare_measurements(args.baseline.resolve(), args.candidate.resolve(), args.screen_traces)
        else:
            result = measure(args.manifest.resolve())
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"Cornell Box GI measurement error: {error}", file=sys.stderr)
        return 2
    if args.baseline:
        print(f"Cornell Box GI comparison written to {args.out}")
        return 0
    failed = [check for check, passed in result["checks"].items() if not passed]
    if failed:
        print(f"Cornell Box GI checks failed: {', '.join(failed)}. See {args.out}", file=sys.stderr)
        return 1
    print(f"Cornell Box GI checks passed. See {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
