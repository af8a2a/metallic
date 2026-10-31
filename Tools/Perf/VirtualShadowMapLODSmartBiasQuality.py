"""Native visibility diagnostics for the fixed orthographic Bunny fixtures.

Full geometry is the image reference. This is not an independent authored-ray
oracle; the separate C++ receiver oracle remains a required regression check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CRITERIA = dict(interiorErrorFractionExclusive=.01, minimumInteriorPixels=100,
                minimumEdgeSamples=20, edgeP95PixelsInclusive=2., edgeSearchPixels=16,
                erodePixels=1, minimumBunnyPixels=10000, minimumPlanePixels=100000)


def erode(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return np.logical_and.reduce([padded[y:y + mask.shape[0], x:x + mask.shape[1]]
                                  for y in range(3) for x in range(3)])


def read_samples(records):
    samples = {}
    for row in records:
        diagnostic = row["diagnostic"]
        for capture in diagnostic["images"]:
            values = {}
            for name, resource, dtype in (("shadow", "Shadows.shadows", "u1"),
                                           ("depth", "VBuffer.depth", "<f4"),
                                           ("visibility", "VBuffer.visibility", "<u4")):
                receipt = capture["images"][resource]
                # Package verification has already checked file hashes and layout.
                values[name] = np.fromfile(Path(diagnostic["directory"]) / receipt["file"], dtype=dtype).reshape(
                    receipt["layerCount"], receipt["height"], receipt["width"])[0]
            if not set(np.unique(values["shadow"])) <= {0, 255}:
                raise ValueError("This protocol requires native binary hard shadows")
            samples[row["workload"], row["arm"], capture["frame"]] = values
    return samples


def changes(reference, candidate, mask):
    count = int(mask.sum())
    dark = int(np.count_nonzero(mask & (reference == 255) & (candidate == 0)))
    lost = int(np.count_nonzero(mask & (reference == 0) & (candidate == 255)))
    return dict(pixels=count, extraDark=dark, lostOcclusion=lost,
                extraDarkFraction=dark / count if count else None,
                lostOcclusionFraction=lost / count if count else None,
                changed=int(np.count_nonzero(mask & (reference != candidate))),
                visibilityMAE=float(np.abs(candidate.astype(float) - reference)[mask].mean() / 255) if count else None)


def edge_displacement(reference, candidate, mask):
    """Match like-oriented shadow edges on rows/columns; missing edges cost 16 px.

    A positive signed displacement shrinks the shadow (potential detachment),
    while a negative displacement enlarges it. Both directions are reported.
    """
    radius = CRITERIA["edgeSearchPixels"]
    signed, missing = [], 0
    for transpose in (False, True):
        a, b, valid = (v.T if transpose else v for v in (reference, candidate, mask))
        reference_edges = a[:, 1:].astype(np.int16) - a[:, :-1].astype(np.int16)
        candidate_edges = b[:, 1:].astype(np.int16) - b[:, :-1].astype(np.int16)
        ys, xs = np.where((reference_edges != 0) & valid[:, 1:] & valid[:, :-1])
        for y, x in zip(ys, xs):
            left, right = max(0, int(x) - radius), min(candidate_edges.shape[1], int(x) + radius + 1)
            positions = np.flatnonzero((candidate_edges[y, left:right] == reference_edges[y, x])
                & valid[y, left:right] & valid[y, left + 1:right + 1]) + left
            if not len(positions):
                signed.append(float(radius))
                missing += 1
            else:
                delta = int(positions[np.argmin(np.abs(positions - x))]) - int(x)
                signed.append(float(delta if reference_edges[y, x] < 0 else -delta))
    return dict(samples=len(signed), missing=missing,
                p95AbsolutePixels=float(np.percentile(np.abs(signed), 95)) if signed else None,
                meanSignedInwardPixels=float(np.mean(signed)) if signed else None,
                p95InwardPixels=float(np.percentile(np.maximum(signed, 0), 95)) if signed else None,
                p95OutwardPixels=float(np.percentile(np.maximum(-np.array(signed), 0), 95)) if signed else None)


def masks_for(reference, camera):
    # The declared scenes use an orthographic camera along -Z and a plane at Z=0.
    plane_depth = (camera["zfar"] - camera["eye"][2]) / (camera["zfar"] - camera["znear"])
    depth, shadow = reference["depth"], reference["shadow"]
    valid = depth > 0
    bunny = depth > plane_depth + 1e-4
    plane = valid & ~bunny
    if int(bunny.sum()) < CRITERIA["minimumBunnyPixels"] or int(plane.sum()) < CRITERIA["minimumPlanePixels"]:
        raise ValueError("Fixture lacks required Bunny/plane coverage")
    return dict(allReceivers=valid, bunny=bunny, plane=plane,
                bunnyLitInterior=erode(bunny & (shadow == 255)),
                bunnyUmbraInterior=erode(bunny & (shadow == 0)),
                planeLitInterior=erode(plane & (shadow == 255)),
                planeUmbraInterior=erode(plane & (shadow == 0)))


def analyze(records):
    samples = read_samples(records)
    rows, checks = [], []
    for workload in dict.fromkeys(r["workload"] for r in records):
        camera = next(r for r in records if r["workload"] == workload)["normalized"]["cases"][0]["camera"]
        for frame in (32, 47):
            reference = samples[workload, "Off", frame]
            masks = masks_for(reference, camera)
            for arm in ("Legacy", "Smart"):
                current = samples[workload, arm, frame]
                if any(not np.array_equal(reference[k], current[k]) for k in ("depth", "visibility")):
                    raise ValueError("Primary geometry differs between quality arms")
                metrics = {name: changes(reference["shadow"], current["shadow"], mask) for name, mask in masks.items()}
                edge = edge_displacement(reference["shadow"], current["shadow"], erode(masks["plane"]))
                rows.append(dict(workload=workload, frame=frame, arm=arm, metrics=metrics, planeShadowEdges=edge))
                if arm == "Smart":
                    gates = {}
                    for region, metric in (("bunnyLitInterior", "extraDarkFraction"), ("planeLitInterior", "extraDarkFraction"),
                                           ("bunnyUmbraInterior", "lostOcclusionFraction"), ("planeUmbraInterior", "lostOcclusionFraction")):
                        value = metrics[region]
                        gates[region] = value["pixels"] >= CRITERIA["minimumInteriorPixels"] and value[metric] < CRITERIA["interiorErrorFractionExclusive"]
                    gates["planeShadowEdges"] = edge["samples"] >= CRITERIA["minimumEdgeSamples"] and edge["p95AbsolutePixels"] <= CRITERIA["edgeP95PixelsInclusive"]
                    checks.append(dict(workload=workload, frame=frame, gates=gates, passed=all(gates.values())))
    return dict(criteria=CRITERIA, numpy=np.__version__, reference="Off native hard-shadow image",
                limitation=__doc__, rows=rows, checks=checks, passed=all(c["passed"] for c in checks))


def plot(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = read_samples(records)
    for workload in dict.fromkeys(r["workload"] for r in records):
        reference = samples[workload, "Off", 47]
        camera = next(r for r in records if r["workload"] == workload)["normalized"]["cases"][0]["camera"]
        masks = masks_for(reference, camera)
        ys, xs = np.where(masks["bunny"])
        crops = ((slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9)),
                 (slice(None), slice(None)))
        figure, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
        for y, crop in enumerate(crops):
            for x, arm in enumerate(("Off", "Legacy", "Smart")):
                axes[y, x].imshow(samples[workload, arm, 47]["shadow"][crop], cmap="gray", vmin=0, vmax=255)
                axes[y, x].set_title(arm)
            for x, arm in enumerate(("Legacy", "Smart"), 3):
                candidate = samples[workload, arm, 47]["shadow"]
                error = np.zeros((*candidate.shape, 3), dtype=np.uint8)
                error[masks["allReceivers"] & (reference["shadow"] == 255) & (candidate == 0)] = (255, 60, 60)
                error[masks["allReceivers"] & (reference["shadow"] == 0) & (candidate == 255)] = (0, 230, 255)
                axes[y, x].imshow(error[crop])
                axes[y, x].set_title(arm + " vs Off")
        for axis in axes.flat:
            axis.set_axis_off()
        figure.suptitle(workload + ": red=extra dark; cyan=lost occlusion; all changed receiver pixels shown")
        figure.savefig(output / f"{workload}-ShadowQuality.png", dpi=140)
        plt.close(figure)
