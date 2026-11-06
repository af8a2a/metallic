"""Offline directional Mark coverage model. No renderer or GPU execution.

Input is an acceptance JSON containing receiverCoverageCaptures, or one capture
descriptor. All coverage results are potential, not floating-point certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

VERSION = "virtual-shadow-map-receiver-coverage-analysis-v1"
INVALID = np.uint32(0xffffffff)
TILES = (1, 2, 4, 8)
MARGINS = (1.0, 2.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def captures(value):
    if isinstance(value, dict):
        if "receiverCoverageCaptures" in value and "frames" in value:
            for original in value["receiverCoverageCaptures"]:
                capture = dict(original)
                frame = [f for f in value["frames"] if f.get("frame") == capture.get("frame")]
                if len(frame) == 1:
                    work = frame[0].get("shadow", {}).get("gpuWork", {})
                    faces = [f for f in work.get("faces", []) if f.get("lightIndex") == capture.get("lightIndex") and f.get("face") == 0]
                    if work.get("enabled") and len(faces) == 1:
                        for name in ("markLevelTests", "markSamplePositions", "markReceiverDispatches"):
                            capture[name] = faces[0]["counters"][name]
                yield capture
        elif "recordWords" in value and "projections" in value and "pages" in value:
            yield value
        else:
            for item in value.values():
                yield from captures(item)
    elif isinstance(value, list):
        for item in value:
            yield from captures(item)


def checked_file(root, info):
    path = Path(info["file"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.stat().st_size != info["byteLength"]:
        raise ValueError(f"Byte length mismatch: {path}")
    actual = digest(path)
    if actual.lower() != info["rawSha256"].lower():
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def tile_values(values, width, height, tile, fill):
    """Rows are tiles, columns are in-tile lanes; supports rounded tails."""
    padded_h = ((height + tile - 1) // tile) * tile
    padded_w = ((width + tile - 1) // tile) * tile
    grid = np.full((padded_h, padded_w), fill, dtype=values.dtype)
    grid[:height, :width] = values.reshape(height, width)
    return grid.reshape(padded_h // tile, tile, padded_w // tile, tile).transpose(0, 2, 1, 3).reshape(-1, tile * tile)


def untile(values, width, height, tile):
    rows = (height + tile - 1) // tile
    cols = (width + tile - 1) // tile
    return np.repeat(np.repeat(values.reshape(rows, cols), tile, axis=0), tile, axis=1)[:height, :width].reshape(-1)


def covered_rectangles(rects, witnesses, level, axis):
    """Exact integer set containment for Q rectangles and real GPU W pages.

    This function is exact; the upstream floating-point Q bound is not proven.
    Invalid or other-level witnesses never satisfy a query.
    """
    low_x, low_y, high_x, high_y = (rects[:, i] for i in range(4))
    area = np.maximum(high_x - low_x + 1, 0).astype(np.int64) * np.maximum(high_y - low_y + 1, 0)
    ordered = np.sort(witnesses, axis=1)
    unique = np.ones(ordered.shape, dtype=bool)
    unique[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    base = level * axis * axis
    valid = unique & (ordered >= base) & (ordered < base + axis * axis)
    local = ordered.astype(np.int64) - base
    x, y = local % axis, local // axis
    inside = valid & (x >= low_x[:, None]) & (x <= high_x[:, None]) & (y >= low_y[:, None]) & (y <= high_y[:, None])
    count = inside.sum(axis=1)
    unique_count = valid.sum(axis=1)
    return count == area, area, unique_count


def descending_admission(requested, physical_count):
    """Exact requested-fine-page rank policy; excludes reserved coarse page."""
    ids = np.flatnonzero(requested)
    admitted = np.zeros_like(requested, dtype=bool)
    admitted[ids[-max(physical_count - 1, 0):] if physical_count > 1 else ids[:0]] = True
    return admitted


def set_metrics(candidate, baseline, baseline_admitted, budget):
    admitted = descending_admission(candidate, budget)
    return {
        "requested": int(candidate.sum()),
        "extraVsBaseline": int((candidate & ~baseline).sum()),
        "missingVsBaseline": int((baseline & ~candidate).sum()),
        "admitted": int(admitted.sum()),
        "admittedExtraVsBaseline": int((admitted & ~baseline_admitted).sum()),
        "baselineAdmittedDisplaced": int((baseline_admitted & ~admitted).sum()),
        "baselineRequestedStillAdmitted": int((admitted & baseline).sum()),
    }


def halo_pages(primary, axis, levels):
    ids = np.unique(primary[primary != INVALID]).astype(np.int64)
    result = np.zeros(levels * axis * axis, dtype=bool)
    if not len(ids):
        return result
    level = ids // (axis * axis)
    local = ids % (axis * axis)
    x, y = local % axis, local // axis
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xx, yy = x + dx, y + dy
            valid = (xx >= 0) & (yy >= 0) & (xx < axis) & (yy < axis)
            result[level[valid] * axis * axis + yy[valid] * axis + xx[valid]] = True
    return result


def add_rectangles(result, rects, level, axis, halo=0):
    """Union rectangle requests with a 2D difference grid, no per-pixel loops."""
    valid = (rects[:, 2] >= rects[:, 0]) & (rects[:, 3] >= rects[:, 1])
    r = rects[valid].astype(np.int64)
    if not len(r):
        return
    x0, y0 = np.maximum(r[:, 0] - halo, 0), np.maximum(r[:, 1] - halo, 0)
    x1, y1 = np.minimum(r[:, 2] + halo, axis - 1), np.minimum(r[:, 3] + halo, axis - 1)
    diff = np.zeros((axis + 1, axis + 1), dtype=np.int64)
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y0, x1 + 1), -1)
    np.add.at(diff, (y1 + 1, x0), -1)
    np.add.at(diff, (y1 + 1, x1 + 1), 1)
    mask = diff.cumsum(0).cumsum(1)[:axis, :axis] > 0
    result[level * axis * axis:(level + 1) * axis * axis] |= mask.reshape(-1)


def support_bounds(position, ray_length, first_level, projections, level, axis, page_size, angle):
    """Double directional cone + dither support, BEFORE empirical texel padding.

    No claim that double bounds strictly enclose all GPU floating-point paths.
    All Z is retained conservatively; only XY map intersection is considered.
    """
    forward = projections[0, 8:11].astype(np.float64)
    forward /= np.linalg.norm(forward)
    ref = np.array([0., 1., 0.]) if abs(forward[1]) < .95 else np.array([1., 0., 0.])
    right = np.cross(ref, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    # ShadowProjection: rowX/Y/Z/W, metric, uint4 page, originNear (112 bytes).
    receiver_texel = projections[first_level, 17].astype(np.float64)
    resolution = axis * page_size
    lows, highs = [], []
    for offset in (0, 4):
        row = projections[level, offset:offset + 4].astype(np.float64)
        center = (position @ row[:3] + row[3]) * (.5 * resolution) + .5 * resolution
        axial = float(row[:3] @ forward)
        perpendicular = np.linalg.norm(row[:3] - axial * forward)
        # Direction differs from cone axis by <= angle in exact arithmetic.
        delta = abs(axial) * (1. - math.cos(angle)) + perpendicular * math.sin(angle)
        lo_step = min(0., axial - delta) * (.5 * resolution)
        hi_step = max(0., axial + delta) * (.5 * resolution)
        dither = receiver_texel * .5 * (abs(row[:3] @ right) + abs(row[:3] @ up)) * (.5 * resolution)
        lows.append(center + ray_length * lo_step - dither)
        highs.append(center + ray_length * hi_step + dither)
    return np.column_stack((lows[0], lows[1], highs[0], highs[1]))


def query_rectangles(bounds, active, axis, page_size, margin):
    resolution = axis * page_size
    low = bounds[:, :2] - margin
    high = bounds[:, 2:] + margin
    # Original inset accepts UV texels [0.5, resolution-0.5). Inclusive high
    # enumeration deliberately retains edge pages; no depth rejection is used.
    overlap = active & (high[:, 0] >= .5) & (high[:, 1] >= .5) & (low[:, 0] < resolution - .5) & (low[:, 1] < resolution - .5)
    low = np.floor(np.clip(low, 0., resolution - 1.) / page_size).astype(np.int32)
    high = np.floor(np.clip(high, 0., resolution - 1.) / page_size).astype(np.int32)
    rect = np.column_stack((low, high))
    rect[~overlap] = (axis, axis, -1, -1)
    return rect


def tile_rectangles(rects, width, height, tile, axis):
    return np.column_stack([
        tile_values(rects[:, 0], width, height, tile, axis).min(1),
        tile_values(rects[:, 1], width, height, tile, axis).min(1),
        tile_values(rects[:, 2], width, height, tile, -1).max(1),
        tile_values(rects[:, 3], width, height, tile, -1).max(1),
    ])


def histogram(values, edges):
    return {f"({low},{high}]": int(((values > low) & (values <= high)).sum())
            for low, high in zip(edges, edges[1:])}


def tile_diagnostics(valid, first, linear_depth, width, height):
    result = []
    for tile in TILES:
        count = tile_values(valid.astype(np.int32), width, height, tile, 0).sum(1)
        active = count > 0
        lo = tile_values(np.where(valid, linear_depth, np.inf), width, height, tile, np.inf).min(1)
        hi = tile_values(np.where(valid, linear_depth, -np.inf), width, height, tile, -np.inf).max(1)
        absolute_span = hi[active] - lo[active]
        relative_span = absolute_span / np.maximum(lo[active], 1e-12)
        sorted_level = np.sort(tile_values(np.where(valid, first, -1), width, height, tile, -1), axis=1)
        unique = np.ones(sorted_level.shape, dtype=bool)
        unique[:, 1:] = sorted_level[:, 1:] != sorted_level[:, :-1]
        distinct = (unique & (sorted_level >= 0)).sum(1)
        result.append({"tile": tile, "tiles": int(len(count)), "activeTiles": int(active.sum()),
            "validReceiverCountHistogram": {str(int(k)): int(v) for k, v in zip(*np.unique(count, return_counts=True))},
            "distinctReceiverLevelHistogram": {str(int(k)): int(v) for k, v in zip(*np.unique(distinct, return_counts=True))},
            "mixedLevelActiveTileFraction": float((distinct > 1).sum() / active.sum()) if active.any() else 0.,
            "absoluteLinearDepthSpanHistogramWorldUnits": histogram(absolute_span, (-1., 0., .01, .1, 1., 10., math.inf)),
            "relativeLinearDepthSpanHistogram": histogram(relative_span, (-1., 0., .01, .1, .5, 1., math.inf)),
            "relativeDepthDefinition": "(max linearDepth - min linearDepth) / min linearDepth over valid receivers"})
    return result


def analyze(capture, root):
    if capture.get("version") != 1 or capture.get("recordWords") != 22 or not capture.get("recordsEnabled"):
        raise ValueError("Expected enabled version-1, 22-word receiver capture")
    width, height = int(capture["width"]), int(capture["height"])
    axis, page_size = int(capture["virtualPagesPerAxis"]), int(capture["pageSize"])
    levels, budget = int(capture["clipmapCount"]), int(capture["physicalPageCount"])
    rays, samples = int(capture["rayCount"]), int(capture["samplesPerRay"])
    angle = float(capture["angularRadius"])
    if not (width > 0 and height > 0 and 1 <= levels <= 12 and axis > 0 and page_size > 0 and rays == 2 and samples > 0 and 0 <= angle < math.pi / 2):
        raise ValueError("Unsupported capture dimensions/settings (requires two directional rays)")
    paths = {name: checked_file(root, capture[name]) for name in ("records", "projections", "pages")}
    records = np.memmap(paths["records"], dtype="<u4", mode="r").reshape(height * width, 22)
    projections = np.fromfile(paths["projections"], dtype="<f4").reshape(levels + 1, 28)
    pages = np.fromfile(paths["pages"], dtype="<u4").reshape(levels * axis * axis + 1, 4)
    projection_pages = projections.view("<u4")[:, 20:24]
    if (not np.isfinite(projections[:, :20]).all() or
            not np.array_equal(projections[:levels, 8:12], np.broadcast_to(projections[0, 8:12], (levels, 4))) or
            np.any(projections[:levels, 19] != 0.) or np.any(projections[:levels, 17] <= 0.) or
            np.any(projection_pages[:levels, 0] != np.arange(levels) * axis * axis) or
            np.any(projection_pages[:levels, 1] != axis)):
        raise ValueError("Projection data does not satisfy the directional shared-Z/fine-grid contract")
    valid = records[:, 0] != 0
    first = records[:, 5].astype(np.int32)
    if np.any(valid & ((first < 0) | (first >= levels))):
        raise ValueError("Invalid receiver level")
    first = np.where(valid, first, 0)
    position = records[:, 1:4].copy().view("<f4").astype(np.float64)
    length = records[:, 4].copy().view("<f4").astype(np.float64)
    if not np.isfinite(position[valid]).all() or not np.isfinite(length[valid]).all() or np.any(length[valid] < 0):
        raise ValueError("Invalid receiver position/ray length")
    position[~valid], length[~valid] = 0., 0.
    masks = records[:, 6:8].copy()
    masks[~valid] = 0
    if np.any(masks >= (1 << levels)) or np.any(masks & ((1 << first[:, None]) - 1)):
        raise ValueError("Sweep mask outside receiver level domain")
    witnesses = records[:, 9:21].copy()
    witnesses[~valid] = INVALID
    primary = records[:, 8].copy()
    primary[~valid] = INVALID
    total_fine = levels * axis * axis
    if np.any((primary != INVALID) & (primary >= total_fine)):
        raise ValueError("Invalid primary page")
    baseline = (pages[:-1, 1] & 1) != 0
    actual_admitted = (pages[:-1, 1] & 2) != 0
    replay_admitted = descending_admission(baseline, budget)
    if not np.array_equal(actual_admitted & baseline, replay_admitted):
        raise ValueError("Actual requested/admitted pages disagree with reverse-rank allocator replay")
    all_witness = np.unique(witnesses[witnesses != INVALID])
    if np.any(all_witness >= total_fine) or not baseline[all_witness].all():
        raise ValueError("GPU witnesses are not a subset of actual baseline requests")
    for level in range(12):
        w = witnesses[:, level]
        if np.any((w != INVALID) & ((level >= levels) | (w // (axis * axis) != level) | (first > level))):
            raise ValueError("Cross-level or inactive GPU witness")
    residual_by_receiver = np.zeros(width * height, dtype=np.int32)
    for level in range(levels):
        residual_by_receiver += (((masks >> level) & 1).sum(1) * (samples + 1)).astype(np.int32)
    residual_total = int(residual_by_receiver.sum())
    producer_levels = int((levels - first[valid]).sum()) * rays
    reconstructed_level_tests = residual_total + producer_levels
    gpu_level_tests = capture.get("markLevelTests")
    if gpu_level_tests is not None and int(gpu_level_tests) != reconstructed_level_tests:
        raise ValueError(f"GPU Mark level tests disagree: {gpu_level_tests} != {reconstructed_level_tests}")
    reconstructed_samples = int(valid.sum()) * rays + int((masks != 0).sum()) * (samples + 1)
    for name, reconstructed in (("markReceiverDispatches", int(valid.sum())), ("markSamplePositions", reconstructed_samples)):
        if name in capture and int(capture[name]) != reconstructed:
            raise ValueError(f"GPU {name} disagrees: {capture[name]} != {reconstructed}")
    rows, covered_masks, witness_covered_masks, empty_domain_masks = {}, {}, {}, {}
    for margin in MARGINS:
        for tile in TILES:
            key = (margin, tile)
            tile_count = ((height + tile - 1) // tile) * ((width + tile - 1) // tile)
            covered_masks[key] = np.zeros(tile_count, dtype=np.uint16)
            witness_covered_masks[key] = np.zeros(tile_count, dtype=np.uint16)
            empty_domain_masks[key] = np.zeros(tile_count, dtype=np.uint16)
            rows[key] = dict(tile=tile, marginTexels=margin, potentialSavedSampleLevelTests=0,
                potentialSavedEmptyDomainSampleLevelTests=0, potentialSavedWitnessCoveredSampleLevelTests=0,
                residualSampleLevelTests=residual_total, activeTileLevels=0, coveredTileLevels=0,
                nonemptyCoveredTileLevels=0, supportRejectedTileLevels=0,
                witnessCountBoundRejectedTileLevels=0, totalQueryPages=0,
                membershipPagesAfterCountGate=0, witnessSlotsVisited=0,
                supportRowDotProducts=0, witnessPreparationLevelTests=0,
                scalarBoundsReductionCombines=0, rectangleAreaHistogram={}, perLevel=[])
    policies = {}
    primary_set = np.zeros(total_fine, dtype=bool)
    primary_set[primary[primary != INVALID]] = True
    parent_set = np.zeros(total_fine, dtype=bool)
    halo_set = halo_pages(primary, axis, levels)
    tile_policy = {tile: np.zeros(total_fine, dtype=bool) for tile in (2, 4, 8)}
    for level in range(levels):
        active = valid & (first <= level)
        bounds = support_bounds(position, length, first, projections, level, axis, page_size, angle)
        swept = ((masks >> level) & 1).sum(1).astype(np.int32)
        needs_query = active & (swept != 0)
        work = swept * (samples + 1)
        w = witnesses[:, level]
        parent = w[(valid & (first < level)) & (w != INVALID)]
        parent_set[parent] = True
        # Exact GPU centerpage grouping; do not derive parents with pageXY >> 1.
        local = primary.astype(np.int64) - level * axis * axis
        primary_here = valid & (first == level) & (primary != INVALID)
        r = np.column_stack((local % axis, local // axis, local % axis, local // axis)).astype(np.int32)
        r[~primary_here] = (axis, axis, -1, -1)
        for tile in (2, 4, 8):
            add_rectangles(tile_policy[tile], tile_rectangles(r, width, height, tile, axis), level, axis, halo=1)
        for margin in MARGINS:
            rects = query_rectangles(bounds, needs_query, axis, page_size, margin)
            for tile in TILES:
                key = margin, tile
                row = rows[key]
                tile_active_count = tile_values(needs_query.astype(np.int32), width, height, tile, 0).sum(1)
                tile_active = tile_active_count > 0
                tile_work = tile_values(work, width, height, tile, 0).sum(1)
                wr = tile_values(w, width, height, tile, INVALID)
                qr = tile_rectangles(rects, width, height, tile, axis)
                covered, area, unique_count = covered_rectangles(qr, wr, level, axis)
                covered &= tile_active
                empty_covered = covered & (area == 0)
                witness_covered = covered & (area > 0)
                covered_masks[key][covered] |= np.uint16(1 << level)
                empty_domain_masks[key][empty_covered] |= np.uint16(1 << level)
                witness_covered_masks[key][witness_covered] |= np.uint16(1 << level)
                row["potentialSavedSampleLevelTests"] += int(tile_work[covered].sum())
                row["potentialSavedEmptyDomainSampleLevelTests"] += int(tile_work[empty_covered].sum())
                row["potentialSavedWitnessCoveredSampleLevelTests"] += int(tile_work[witness_covered].sum())
                row["activeTileLevels"] += int(tile_active.sum())
                row["coveredTileLevels"] += int(covered.sum())
                row["nonemptyCoveredTileLevels"] += int((covered & (area > 0)).sum())
                row["supportRejectedTileLevels"] += int((tile_active & ~covered).sum())
                row["witnessCountBoundRejectedTileLevels"] += int((tile_active & (area > unique_count)).sum())
                row["totalQueryPages"] += int(area[tile_active].sum())
                row["membershipPagesAfterCountGate"] += int(area[tile_active & (area <= unique_count)].sum())
                witness_slots = tile_values(active.astype(np.int32), width, height, tile, 0).sum(1)
                row["witnessSlotsVisited"] += int(witness_slots[tile_active].sum())
                row["supportRowDotProducts"] += int(needs_query.sum()) * 2
                # Charge a separate actual ray0 t=0 producer only where its
                # original single-page certificate failed; successes reuse it.
                prep = tile_values(((masks[:, 0] >> level) & 1).astype(np.int32), width, height, tile, 0).sum(1)
                row["witnessPreparationLevelTests"] += int(prep[tile_active].sum())
                row["scalarBoundsReductionCombines"] += int(np.maximum(tile_active_count - 1, 0).sum()) * 4
                row["perLevel"].append({"level": level, "residualSampleLevelTests": int(work.sum()),
                    "potentialSavedSampleLevelTests": int(tile_work[covered].sum()), "queryTiles": int(tile_active.sum()),
                    "potentialSavedEmptyDomainSampleLevelTests": int(tile_work[empty_covered].sum()),
                    "potentialSavedWitnessCoveredSampleLevelTests": int(tile_work[witness_covered].sum()),
                    "coveredQueryTiles": int(covered.sum()), "queryPageArea": int(area[tile_active].sum())})
                histogram = row["rectangleAreaHistogram"]
                # Fixed bins: zero, one, two, 3..4, 5..8, 9..16, 17..64, >64.
                edges = (-1, 0, 1, 2, 4, 8, 16, 64, math.inf)
                for lo, hi in zip(edges, edges[1:]):
                    label = f"{lo + 1}..{hi}"
                    histogram[label] = histogram.get(label, 0) + int((tile_active & (area > lo) & (area <= hi)).sum())
    for key, row in rows.items():
        mask = untile(covered_masks[key], width, height, key[1])
        fully_removed_rays = ((masks != 0) & ((masks & ~mask[:, None].astype(np.uint32)) == 0)).sum()
        row["potentialFullyRemovedResidualRaySweeps"] = int(fully_removed_rays)
        row["potentialSavedSamplePositions"] = int(fully_removed_rays) * (samples + 1)
        for masks_by_key, name in ((witness_covered_masks, "WitnessOnly"), (empty_domain_masks, "EmptyDomainOnly")):
            kind_mask = untile(masks_by_key[key], width, height, key[1]).astype(np.uint32)
            removed = int(((masks != 0) & ((masks & ~kind_mask[:, None]) == 0)).sum())
            row[f"potentialFullyRemovedResidualRaySweeps{name}"] = removed
            row[f"potentialSavedSamplePositions{name}"] = removed * (samples + 1)
        row["potentialFullyRemovedResidualRaySweepsMixedReasons"] = int(fully_removed_rays) - row["potentialFullyRemovedResidualRaySweepsWitnessOnly"] - row["potentialFullyRemovedResidualRaySweepsEmptyDomainOnly"]
        row["remainingSampleLevelTests"] = residual_total - row["potentialSavedSampleLevelTests"]
        row["savedResidualFraction"] = row["potentialSavedSampleLevelTests"] / residual_total if residual_total else 0.
        row["provenSavedSampleLevelTests"] = 0
    policy_sets = {
        "primaryOnly": primary_set,
        "primaryPlusRay0T0Parents": primary_set | parent_set,
        "primaryPlusOnePageHaloPlusRay0T0Parents": halo_set | parent_set,
        **{f"tile{tile}PrimaryRectPlusOnePageHaloPlusRay0T0Parents": value | parent_set for tile, value in tile_policy.items()},
    }
    for name, candidate in policy_sets.items():
        policies[name] = set_metrics(candidate, baseline, replay_admitted, budget)
    return {
        "schema": VERSION,
        "claim": "Potential analytic coverage only; no GPU time prediction or strict floating-point certificate",
        "capture": {k: capture[k] for k in ("version", "frame", "width", "height", "pageSize", "virtualPagesPerAxis", "clipmapCount", "physicalPageCount", "rayCount", "samplesPerRay", "angularRadius", "lightIndex", "recordWords") if k in capture},
        "inputs": {name: {"file": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "validation": {"witnessesSubsetOfBaseline": True, "uniqueWitnessPages": int(len(all_witness)),
            "gpuMarkLevelTests": int(gpu_level_tests) if gpu_level_tests is not None else None,
            "reconstructedMarkLevelTests": reconstructed_level_tests,
            "markLevelTestsMatch": bool(int(gpu_level_tests) == reconstructed_level_tests) if gpu_level_tests is not None else None,
            "allocatorReplayMatchesRequestedAdmittedFlags": bool(np.array_equal(actual_admitted & baseline, replay_admitted)),
            "admittedFlagOutsideRequestedCount": int((actual_admitted & ~baseline).sum())},
        "baseline": {"validReceivers": int(valid.sum()), "residualSampleLevelTests": residual_total,
            "producerLevelTests": producer_levels, "reconstructedMarkLevelTests": reconstructed_level_tests,
            "reconstructedMarkSamplePositions": reconstructed_samples,
            "residualRaySweeps": int((masks != 0).sum()), "requestedFinePages": int(baseline.sum()),
            "admittedFinePages": int(replay_admitted.sum()), "budgetIncludingReservedCoarse": budget},
        "coverage": list(rows.values()), "pagePolicies": policies,
        "tileDiagnostics": tile_diagnostics(valid, first, records[:, 21].copy().view("<f4").astype(np.float64), width, height),
        "limitations": [
            "W is real GPU ray0 t=0 witness pages at each level, including failed ray single-page certificates.",
            "Q uses CPU double cone/dither support plus 1 or 2 texel padding, without a strict GPU FP/SFU error proof; proven coverage is zero.",
            "Tile Q bounds only receivers with residual sweep work at that level, using the all-ray cone; W includes all original active receiver witnesses in the tile. Gaps/depth discontinuities can overestimate demand.",
            "Witness preparation counts a separate ray0 t=0 level query where its original single-page certificate failed; successful certificates reuse the original producer. Actual hoisting/storage cost is unmeasured.",
            "The model conditions Q on already-known residual masks and reuses successful witnesses. Online implementation must wait for original ray certificate results, potentially retaining ray state, adding a producer/barrier phase or regenerating rays. Those state/storage/scheduling/recomputation costs are not measured; this is not a no-prescan algorithm.",
            "Empty-domain savings (Q has zero pages) are separated from nonempty witness coverage. Empty-domain elimination needs no W membership but still requires a proven numerical bound and its own online placement/cost design.",
            "Z rejection is deliberately not used; it can only make this model looser in exact arithmetic.",
            "Logical setup/reduction/query counts have different costs; never subtract them as GPU cycles or milliseconds.",
            "Baseline residual work excludes existing successful ray single-page certificates and lastPages/shared-hash atomic reductions are not sample-level savings.",
            "Policy parents are actual GPU ray0 dithered t=0 witnesses on levels strictly above each receiver's selected level, not guessed child-page shifts.",
            "Admission replay models reverse virtual-page rank with one reserved coarse page; it does not predict cache assignments, redraws, quality, temporal history or multi-light budget feedback.",
        ],
    }


def plot_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for margin in MARGINS:
        rows = [r for r in report["coverage"] if r["marginTexels"] == margin]
        for field, label, color in (("potentialSavedEmptyDomainSampleLevelTests", "Empty XY domain", "C0"),
                                   ("potentialSavedWitnessCoveredSampleLevelTests", "Nonempty W coverage", "C1")):
            axes[0].plot([r["tile"] for r in rows], [100 * r[field] / max(r["residualSampleLevelTests"], 1) for r in rows],
                         marker="o", color=color, linestyle="-" if margin == 1. else "--", label=f"{label}, pad {margin:g}")
    axes[0].set(title="Potential residual sample-level removal", xlabel="Tile side (1 = receiver)", ylabel="Original residual work removed (%)", xticks=TILES)
    axes[0].legend(fontsize=8)
    rows = [r for r in report["coverage"] if r["marginTexels"] == 1.]
    x = np.arange(len(rows))
    costs = (("potentialSavedSampleLevelTests", "Saved sample-level tests"), ("supportRowDotProducts", "Extra support row dots"),
             ("witnessPreparationLevelTests", "Extra witness level tests"), ("scalarBoundsReductionCombines", "Bounds reduction combines"),
             ("membershipPagesAfterCountGate", "Page membership queries"))
    for index, (name, label) in enumerate(costs):
        axes[1].bar(x + (index - 2) * .16, [r[name] / 1e6 for r in rows], width=.15, label=label)
    axes[1].set(title="Logical counts: units are not GPU cost", ylabel="Millions of logical operations", xticks=x, xticklabels=[str(r["tile"]) for r in rows], xlabel="Tile side, 1 texel padding")
    axes[1].legend(fontsize=8)
    names = list(report["pagePolicies"])
    labels = ["Primary", "+ parents", "+ halo", "Tile2 + halo", "Tile4 + halo", "Tile8 + halo"]
    x = np.arange(len(names))
    axes[2].bar(x - .2, [report["pagePolicies"][n]["extraVsBaseline"] for n in names], .4, label="Extra requested")
    axes[2].bar(x + .2, [report["pagePolicies"][n]["baselineAdmittedDisplaced"] for n in names], .4, label="Baseline admitted displaced")
    axes[2].set(title="Hypothetical fixed-budget page policies", ylabel="Pages", xticks=x, xticklabels=labels)
    axes[2].tick_params(axis="x", rotation=40)
    axes[2].legend(fontsize=8)
    fig.suptitle("Offline model only — potential coverage, no measured GPU improvement")
    fig.savefig(output / "Coverage.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=32)
    parser.add_argument("--root", type=Path, help="Base for relative binary paths (default input parent)")
    parser.add_argument("--gpu-mark-level-tests", type=int, help="Same-frame GPU work counter, if absent from receipt")
    args = parser.parse_args()
    source = args.input.resolve()
    found = [c for c in captures(json.loads(source.read_text(encoding="utf-8-sig"))) if c.get("recordsEnabled") and c.get("frame") == args.frame]
    if len(found) != 1:
        parser.error(f"Expected one enabled frame-{args.frame} capture; found {len(found)}")
    selected = dict(found[0])
    if args.gpu_mark_level_tests is not None:
        if "markLevelTests" in selected and int(selected["markLevelTests"]) != args.gpu_mark_level_tests:
            parser.error("Receipt and CLI GPU Mark level counters disagree")
        selected["markLevelTests"] = args.gpu_mark_level_tests
    report = analyze(selected, args.root or source.parent)
    report["manifest"] = {"file": str(source), "sha256": digest(source)}
    report["analyzer"] = {"file": str(Path(__file__).resolve()), "sha256": digest(__file__), "numpy": np.__version__}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "Coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_report(report, args.output)
    print(json.dumps({"report": str((args.output / "Coverage.json").resolve()), "plot": str((args.output / "Coverage.png").resolve()), "baseline": report["baseline"]}))


if __name__ == "__main__":
    main()
