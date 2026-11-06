"""CPU-only independent pixel-set gates for the fixed shadow-LOD refinement.

The original authored oracle supplies the frozen Baseline truth. Other declared
views trace the same authored triangles with the existing double-precision BVH.
No Off shadow image is used as geometric ground truth.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapLODImpact as impact
import VirtualShadowMapLODSmartBias as original
import VirtualShadowMapLODSmartBiasOracle as oracle
import VirtualShadowMapLODSmartBiasQuality as native


PROTOCOL = "virtual-shadow-map-lod-smart-bias-refine-oracle-v2"
CRITERIA = dict(primaryEpsilonMetres=1e-5, epsilonMetres=list(oracle.EPSILONS),
                depthTolerance=2e-5, minimumAcneFixRetention=.9, maximumNewTrueLeaks=0,
                baselineQualifiedPixels=23917, baselineUnresolvedPixels=30,
                baselineNewTrueLeaks=63, baselineRedrawAcneFixes=1359,
                baselineWarmAcneFixes=1360, requireOriginalPixelsRestored=True,
                requireTotalAcneBudget=True, erodePixelsForIndependentGates=0,
                baselineDiagnosticManifestSHA256="d2cc7506f48ff3596fabb9941292f69064d4974d8b7fd412ce1fc75d266cd805",
                baselineOracleManifestSHA256="35cc6e65bcf9fd8b41a23c20246fc7fbeab9b48d76f383f21b29f7f2505fc53e")
ARMS = ("Off", "Legacy", "Smart", "Refined")


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def pixel_sets(truth, qualified, legacy, smart, refined):
    lost = qualified & (truth == 0) & (legacy == 0) & (smart == 255)
    fixed = qualified & (truth == 255) & (legacy == 0) & (smart == 255)
    remaining = lost & (refined == 255)
    new = qualified & (truth == 0) & (smart == 0) & (refined == 255)
    extra_vs_legacy = qualified & (truth == 0) & (legacy == 0) & (refined == 255)
    retained = fixed & (refined == 255)
    minimum = math.ceil(int(fixed.sum()) * CRITERIA["minimumAcneFixRetention"])
    legacy_acne = int(np.count_nonzero(qualified & (truth == 255) & (legacy == 0)))
    refined_acne = int(np.count_nonzero(qualified & (truth == 255) & (refined == 0)))
    gates = dict(allOriginalLeaksRestored=not bool(remaining.any()),
                 noNewTrueLeaks=not bool(new.any()), noAddedLeaksVsLegacy=not bool(extra_vs_legacy.any()),
                 acneFixRetention=int(retained.sum()) >= minimum,
                 totalAcneBudget=refined_acne <= legacy_acne - minimum)
    return dict(originalLeakPixels=int(lost.sum()), originalAcneFixPixels=int(fixed.sum()),
                originalLeakMaskSHA256=mask_digest(lost), originalAcneFixMaskSHA256=mask_digest(fixed),
                remainingOriginalLeaks=int(remaining.sum()), newTrueLeaks=int(new.sum()),
                addedTrueLeaksVsLegacy=int(extra_vs_legacy.sum()), retainedAcneFixes=int(retained.sum()),
                minimumRetainedAcneFixes=minimum, legacyFalseShadows=legacy_acne,
                refinedFalseShadows=refined_acne, maximumRefinedFalseShadows=legacy_acne - minimum,
                remainingLeakCoordinates=np.argwhere(remaining)[:, ::-1].tolist(),
                newLeakCoordinates=np.argwhere(new)[:, ::-1].tolist(), gates=gates, passed=all(gates.values()))


def frozen_reference(diagnostic, authored):
    if (acceptance.digest(diagnostic / "Manifest.json") != CRITERIA["baselineDiagnosticManifestSHA256"]
            or acceptance.digest(authored / "Manifest.json") != CRITERIA["baselineOracleManifestSHA256"]):
        raise ValueError("The predeclared original Smart diagnostic or authored oracle was replaced")
    impact.check_seal(authored)
    verified = original.verify(diagnostic)
    receipt = acceptance.read(authored / "Oracle.json")
    if receipt["diagnosticManifestSHA256"] != acceptance.digest(diagnostic / "Manifest.json"):
        raise ValueError("Frozen authored oracle refers to another diagnostic")
    if (receipt["contract"]["originOffsetsMetres"] != list(oracle.EPSILONS)
            or receipt["contract"]["depthTolerance"] != CRITERIA["depthTolerance"]
            or receipt["contract"]["triangleMinimumDistanceMetres"] != oracle.TRIANGLE_EPSILON):
        raise ValueError("Frozen authored oracle numerical contract changed")
    if verified["scenario"] != "Baseline" or verified["phase"] != "diagnostic":
        raise ValueError("The frozen pixel-set reference must be the original Baseline diagnostic")
    truth = np.load(authored / "Redraw-32-Oracle.npz", allow_pickle=False)
    validation = np.load(authored / "Redraw-32-CameraValidation.npz", allow_pickle=False)
    if (not np.array_equal(truth["originOffsets"], np.asarray(oracle.EPSILONS))
            or any(not np.array_equal(truth[key], validation[key]) for key in ("ys", "xs"))):
        raise ValueError("Frozen oracle receiver coordinates or epsilon order changed")
    qualified = np.zeros_like(truth["mask"], dtype=bool)
    qualified[truth["ys"], truth["xs"]] = validation["residual"] <= CRITERIA["depthTolerance"]
    if int(qualified.sum()) != CRITERIA["baselineQualifiedPixels"]:
        raise ValueError("Frozen receiver qualification changed")
    return verified, native.read_samples(verified["records"]), truth, qualified


def authored_visibility(case, reference, bvh, owners):
    camera = case["camera"]
    bunny = native.masks_for(reference, camera)["bunny"]
    ys, xs = np.where(bunny)
    height, width = bunny.shape
    origins, directions = oracle.camera_rays(camera, xs, ys, width, height)
    distances, hit_ids = oracle.camera_hits(origins, directions, camera["znear"], camera["zfar"], *bvh)
    # Authored pixel-center coverage can disagree with the quantized primary
    # raster at silhouettes. Such pixels have no qualified Bunny receiver;
    # retain their evidence without tracing from a different object's surface.
    has_hit = hit_ids >= 0
    hit_owners = np.full(len(hit_ids), -1, dtype=np.int32)
    hit_owners[has_hit] = owners[hit_ids[has_hit]]
    is_bunny = hit_owners == 0
    bunny_indices = np.flatnonzero(is_bunny)
    positions = origins[is_bunny] + directions[is_bunny] * distances[is_bunny, None]
    depth = np.full(len(hit_ids), np.nan)
    depth[has_hit] = (camera["zfar"] - distances[has_hit]) / (camera["zfar"] - camera["znear"])
    residual = np.abs(depth - reference["depth"][ys, xs])
    qualified_receivers = is_bunny & (residual <= CRITERIA["depthTolerance"])
    qualified = np.zeros_like(bunny)
    qualified[ys, xs] = qualified_receivers
    direction = -np.asarray(case["lighting"]["sun"]["direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    subset = np.flatnonzero((xs % 31 == 0) & (ys % 29 == 0))
    if len(subset) < 8:
        raise ValueError("Insufficient independent brute-force validation samples")
    unresolved = np.flatnonzero(~qualified_receivers)
    camera_subset = np.union1d(subset, unresolved)
    for index in camera_subset:
        distance, triangle = oracle.brute(origins[index], directions[index], camera["znear"], camera["zfar"], *bvh[:3])
        if ((triangle >= 0) != bool(has_hit[index]) or
                (triangle >= 0 and (abs(distance - distances[index]) > 1e-9 or owners[triangle] != hit_owners[index]))):
            raise ValueError("Authored BVH camera result differs from brute force")
    shadow_subset = np.flatnonzero((xs[is_bunny] % 31 == 0) & (ys[is_bunny] % 29 == 0))
    if len(shadow_subset) < 8:
        raise ValueError("Insufficient authored Bunny shadow cross-check samples")
    images = []
    for epsilon in oracle.EPSILONS:
        values = oracle.shadow_rays(positions, direction, epsilon, *bvh)
        for index in shadow_subset:
            _, triangle = oracle.brute(positions[index] + direction * epsilon, direction,
                                       oracle.TRIANGLE_EPSILON, 1e30, *bvh[:3])
            if (triangle < 0) != (values[index] == 255):
                raise ValueError("Authored BVH shadow result differs from brute force")
        image = np.full(bunny.shape, 127, dtype=np.uint8)
        image[ys[bunny_indices], xs[bunny_indices]] = values
        images.append(image)
    unresolved_evidence = [dict(x=int(xs[index]), y=int(ys[index]),
        reason="authored-miss" if not has_hit[index] else "authored-other-owner" if not is_bunny[index] else "authored-depth-mismatch",
        authoredTriangle=int(hit_ids[index]), authoredOwner=int(hit_owners[index]),
        gpuDepth=float(reference["depth"][ys[index], xs[index]]),
        authoredDepth=float(depth[index]) if has_hit[index] else None,
        depthResidual=float(residual[index]) if has_hit[index] else None,
        worldDepthResidualMetres=float(residual[index] * (camera["zfar"] - camera["znear"])) if has_hit[index] else None,
        bruteForceCameraMatched=True) for index in unresolved]
    return np.stack(images), qualified, dict(cameraSamples=len(xs), depthErrors=int(np.count_nonzero(~qualified & bunny)),
        authoredMissPixels=int(np.count_nonzero(~has_hit)), authoredOtherOwnerPixels=int(np.count_nonzero(has_hit & ~is_bunny)),
        authoredBunnyDepthMismatchPixels=int(np.count_nonzero(is_bunny & ~qualified_receivers)),
        unknownOraclePixels=int(np.count_nonzero(~is_bunny)), unresolvedReceivers=unresolved_evidence,
        maximumDepthError=float(residual[has_hit].max()) if has_hit.any() else None,
        bruteForceCameraSamples=len(camera_subset), bruteForceUnresolvedSamples=len(unresolved),
        bruteForceShadowSamples=len(shadow_subset) * len(oracle.EPSILONS), validationPassed=True)


def evaluate(verified, source, output, reference_diagnostic, reference_oracle):
    if verified["phase"] != "diagnostic":
        raise ValueError("Independent gates require a complete diagnostic package")
    impact.validate_assets(verified["assets"])
    old, old_samples, frozen, frozen_qualified = frozen_reference(reference_diagnostic, reference_oracle)
    if verified["assets"] != old["assets"]:
        raise ValueError("Authored assets differ from the frozen experiment")
    samples = native.read_samples(verified["records"])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(protocol=PROTOCOL, criteria=CRITERIA, source=str(source),
        diagnosticManifestSHA256=acceptance.digest(source / "Manifest.json"), scenario=verified["scenario"],
        referenceDiagnostic=dict(path=str(reference_diagnostic), sha256=acceptance.digest(reference_diagnostic / "Manifest.json")),
        referenceOracle=dict(path=str(reference_oracle), sha256=acceptance.digest(reference_oracle / "Manifest.json")),
        toolsSHA256={Path(m.__file__).name: acceptance.digest(Path(m.__file__)) for m in (oracle, native)},
        helperSHA256=acceptance.digest(Path(__file__)), rows=[], passed=False,
        scope="Qualified authored Bunny receivers; unresolved pixels and epsilon variants remain explicit. Plane and contact gates are separate.")
    bvh, owners, cache = None, None, {}
    try:
        for workload in verified["workloads"]:
            case = next(r for r in verified["records"] if r["workload"] == workload)["normalized"]["cases"][0]
            for frame in (32, 47):
                current = samples[workload, "Off", frame]
                bunny = native.masks_for(current, case["camera"])["bunny"]
                for arm in ARMS:
                    if any(not np.array_equal(samples[workload, arm, frame][key], current[key]) for key in ("depth", "visibility")):
                        raise ValueError("Refinement changed primary receiver inputs")
                if verified["scenario"] == "Baseline":
                    for arm in ("Off", "Legacy", "Smart"):
                        if any(not np.array_equal(samples[workload, arm, frame][key], old_samples[workload, arm, frame][key])
                               for key in ("depth", "visibility", "shadow")):
                            raise ValueError("Original arm no longer reproduces its frozen v1 output")
                    if not np.array_equal(bunny, frozen["mask"]) or not np.array_equal(
                            current["depth"][frozen["ys"], frozen["xs"]], frozen["gpuDepth"]):
                        raise ValueError("Frozen authored oracle no longer describes these receivers")
                    images, qualified = frozen["visibility"], frozen_qualified
                    geometry = dict(cameraSamples=int(bunny.sum()), depthErrors=int(np.count_nonzero(bunny & ~qualified)),
                                    frozenAuthoredReference=True, validationPassed=True)
                else:
                    key = (str(case["camera"]), str(case["lighting"]), mask_digest(current["depth"]))
                    if key not in cache:
                        if bvh is None:
                            triangles, owners, _ = oracle.authored_triangles(Path(case["scene"]))
                            bvh, _ = oracle.build_bvh(triangles)
                        cache[key] = authored_visibility(case, current, bvh, owners)
                    images, qualified, geometry = cache[key]
                shadows = {arm: samples[workload, arm, frame]["shadow"] for arm in ARMS}
                primary = list(oracle.EPSILONS).index(CRITERIA["primaryEpsilonMetres"])
                results = []
                for epsilon, truth in zip(oracle.EPSILONS, images, strict=True):
                    metrics = pixel_sets(truth, qualified, shadows["Legacy"], shadows["Smart"], shadows["Refined"])
                    metrics.update(epsilonMetres=epsilon,
                        arms={arm: native.changes(truth, shadow, qualified) for arm, shadow in shadows.items()})
                    results.append(metrics)
                if verified["scenario"] == "Baseline":
                    expected_fixes = CRITERIA["baselineRedrawAcneFixes" if workload == "Redraw" else "baselineWarmAcneFixes"]
                    if (results[primary]["originalLeakPixels"] != CRITERIA["baselineNewTrueLeaks"]
                            or results[primary]["originalAcneFixPixels"] != expected_fixes
                            or geometry["depthErrors"] != CRITERIA["baselineUnresolvedPixels"]):
                        raise ValueError("Predeclared original 63-leak/acne-fix pixel sets changed")
                payload = f"{workload}-{frame}-GateSamples.npz"
                np.savez_compressed(output / payload, truth=images, qualified=qualified, bunny=bunny, **shadows)
                row = dict(workload=workload, frame=frame, geometry=geometry, qualifiedPixels=int(qualified.sum()),
                    qualifiedMaskSHA256=mask_digest(qualified), unresolvedPixels=int(np.count_nonzero(bunny & ~qualified)),
                    unresolvedValues={arm: dict(dark=int(np.count_nonzero(bunny & ~qualified & (image == 0))),
                                               lit=int(np.count_nonzero(bunny & ~qualified & (image == 255)))) for arm, image in shadows.items()},
                    samples=payload, epsilonResults=results, passed=results[primary]["passed"])
                report["rows"].append(row)
                acceptance.save(output / "OracleGate.json", report)
        report["passed"] = bool(report["rows"]) and all(r["passed"] for r in report["rows"])
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        acceptance.save(output / "OracleGate.json", report)
        impact.seal(output)
    print(f"{output / 'OracleGate.json'}\nIndependent pixel-set gates passed: {report['passed']}")


def verify_gate(directory, diagnostic):
    impact.check_seal(directory)
    report = acceptance.read(directory / "OracleGate.json")
    if (report["protocol"] != PROTOCOL or report["criteria"] != CRITERIA or not report["passed"]
            or report["diagnosticManifestSHA256"] != acceptance.digest(diagnostic / "Manifest.json")):
        raise ValueError("Independent pixel-set prerequisite is missing, rejected or belongs to another diagnostic")
    plan = acceptance.read(diagnostic / "Plan.json")
    if [(r["workload"], r["frame"]) for r in report["rows"]] != [(w, f) for w in plan["workloads"] for f in (32, 47)]:
        raise ValueError("Independent gate does not cover every diagnostic capture")
    primary = list(oracle.EPSILONS).index(CRITERIA["primaryEpsilonMetres"])
    for row in report["rows"]:
        values = np.load(directory / row["samples"], allow_pickle=False)
        replay = pixel_sets(values["truth"][primary], values["qualified"], values["Legacy"], values["Smart"], values["Refined"])
        if not replay["passed"] or any(row["epsilonResults"][primary][key] != value for key, value in replay.items()):
            raise ValueError("Independent gate cannot be reproduced from its sealed pixel sets")
    return report


def plot_native(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = native.read_samples(records)
    for workload in dict.fromkeys(r["workload"] for r in records):
        reference = samples[workload, "Off", 47]
        camera = next(r for r in records if r["workload"] == workload)["normalized"]["cases"][0]["camera"]
        masks = native.masks_for(reference, camera)
        ys, xs = np.where(masks["bunny"])
        crops = ((slice(max(0, ys.min() - 8), ys.max() + 9), slice(max(0, xs.min() - 8), xs.max() + 9)),
                 (slice(None), slice(None)))
        figure, axes = plt.subplots(2, 6, figsize=(20, 7), constrained_layout=True)
        for y, crop in enumerate(crops):
            for x, arm in enumerate(ARMS):
                axes[y, x].imshow(samples[workload, arm, 47]["shadow"][crop], cmap="gray", vmin=0, vmax=255)
                axes[y, x].set_title(arm)
            for x, arm in enumerate(("Smart", "Refined"), 4):
                candidate = samples[workload, arm, 47]["shadow"]
                error = np.zeros((*candidate.shape, 3), dtype=np.uint8)
                error[masks["allReceivers"] & (reference["shadow"] == 255) & (candidate == 0)] = (255, 60, 60)
                error[masks["allReceivers"] & (reference["shadow"] == 0) & (candidate == 255)] = (0, 230, 255)
                axes[y, x].imshow(error[crop])
                axes[y, x].set_title(arm + " vs Off")
        for axis in axes.flat:
            axis.set_axis_off()
        figure.suptitle(workload + ": native Off image comparison; independent authored gate is separate")
        figure.savefig(output / f"{workload}-ShadowQuality.png", dpi=140)
        plt.close(figure)
