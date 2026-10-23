#!/usr/bin/env python3
"""Audit bounded, same-process MiniZorah BLAS quality comparisons (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROTOCOL = "minizorah-blas-frozen-quality-v1"
MERGED_SHARING_PROTOCOL = "minizorah-blas-frozen-quality-v2"
NAMES = ("A1", "A2", "B0", "B1", "R")
RESOURCES = ("Shadows.shadow", "Deferred.color", "ReflectionQuality.color", "FinalBlit.color")
AUXILIARY = ("VBuffer.depth", "VBuffer.visibility")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def image_arrays(directory, sources):
    metadata = read(directory / "Images.json")
    sources.append(identity(directory / "Images.json"))
    arrays, hashes = {}, {}
    for item in metadata:
        path = (directory / item["file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Image file escapes capture directory")
        kind, channels = item["format"].rsplit("x", 1)
        dtype = {"float16": "<f2", "float32": "<f4", "unorm8": "u1", "srgb8": "u1", "uint32": "<u4"}[kind]
        data = np.fromfile(path, dtype=dtype).reshape(item["height"], item["width"], int(channels))
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite captured image: " + str(path))
        record = identity(path)
        sources.append(record)
        name = item["resource"]
        if name in arrays:
            raise ValueError("Duplicate image resource")
        hashes[name] = record["sha256"]
        arrays[name] = data.astype(np.float64)[..., :min(3, int(channels))]
        if kind in ("unorm8", "srgb8"):
            arrays[name] /= 255
    if set(arrays) not in (set(RESOURCES), set(RESOURCES + AUXILIARY)):
        raise ValueError("Missing or unexpected controlled quality resources")
    return arrays, hashes


def display(value):
    if value.shape[-1] == 1:
        return np.clip(value, 0, 1)
    linear = np.maximum(value, 0)
    return (linear / (1 + linear)) ** (1 / 2.2)


def errors(value, reference):
    delta = np.abs(value - reference)
    shown = np.abs(display(value) - display(reference))
    return dict(rawMAE=float(delta.mean()), rawRMSE=float(np.sqrt(np.mean(delta ** 2))),
                rawP99=float(np.quantile(delta, .99)), rawP999=float(np.quantile(delta, .999)),
                rawMax=float(delta.max()), displayMAE=float(shown.mean()),
                displayPixelsOverOnePercent=float(np.mean(shown.max(axis=-1) > .01)),
                exactPixels=float(np.mean(np.all(delta == 0, axis=-1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run directory with Capture.json, or its single-run parent")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "Capture.json").exists():
        runs = list(root.glob("run*/Capture.json"))
        if len(runs) != 1:
            raise ValueError("Expected exactly one controlled run")
        root = runs[0].parent
    capture_path = root / "Capture.json"
    manifest_path = root.parent / "Manifest.json"
    frames_path = root / "ControlledFrames.jsonl"
    capture, manifest = read(capture_path), read(manifest_path)
    protocol = capture.get("protocol")
    if protocol not in (PROTOCOL, MERGED_SHARING_PROTOCOL) or manifest.get("protocol") != protocol or capture.get("status") != "capture_complete":
        raise ValueError("Incomplete capture or unexpected protocol")
    if capture["config"] != manifest["config"] or capture["config"].get("sample") != "gpu-driven-sample":
        raise ValueError("Controlled config identity or MiniZorah sample mismatch")
    if "minizorah" not in manifest["asset"]["path"].lower():
        raise ValueError("Manifest is not the MiniZorah asset")
    for key in ("executableSha256", "shaderSha256"):
        if len(manifest.get(key, "")) != 64:
            raise ValueError("Missing captured binary/shader identity")
    if capture.get("blasStorageRejectedSemantics") != "physical-and-known-storage-only-v2":
        raise ValueError("Unexpected storage-rejection semantics")
    config = capture["config"]["blasQualityComparison"]
    merged_sharing = config.get("mergedSharingComparison", False)
    if not isinstance(merged_sharing, bool) or merged_sharing != (protocol == MERGED_SHARING_PROTOCOL):
        raise ValueError("Merged-sharing comparison configuration/protocol mismatch")
    poses = config.get("poses", [{"forward": 0, "yaw": 0}])
    seeds = config.get("seeds", [0])
    if not 1 <= len(poses) <= 2 or not 1 <= len(seeds) <= 2 or len(set(seeds)) != len(seeds) or \
            not 4 <= config.get("settleFrames", 16) <= 16 or \
            config["storageBytes"] <= 0 or config["referenceBytes"] < config["storageBytes"]:
        raise ValueError("Unexpected bounded comparison configuration")
    if capture["config"].get("dlssMode") != "Off" or capture["config"].get("temporalJitter") is not False or \
            capture["config"].get("shadowOverrides", {}).get("sigmaDenoise") is not False or \
            "fixedExposureEV100" not in capture["controlledBlas"]:
        raise ValueError("Missing deterministic sampling/exposure settings")
    if not 1 <= capture["config"].get("reflectionSamples", 8) <= 16:
        raise ValueError("Shader currently clamps samples to16; larger requested SPP is not an actual larger-Spp capture")
    rows = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != capture["frames"]:
        raise ValueError("Controlled frame coverage mismatch")
    variants = capture["controlledBlas"]["variants"]
    if [(v["pose"], v["name"]) for v in variants] != [(p, n) for p in range(len(poses)) for n in NAMES]:
        raise ValueError("Expected A1/A2/B0/B1/R cold-reset order for each pose")
    if len({v["resetEpoch"] for v in variants}) != len(variants):
        raise ValueError("Each variant must have a distinct cold-reset epoch")
    output = (args.output or root.parent / "FrozenQualityAnalysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [identity(path) for path in (capture_path, manifest_path, frames_path)]
    failures, audits, comparisons, plots, raster_audits = [], [], [], [], []
    frozen_keys = ("geometryBytes", "residentPages", "clasBytes", "clasTotalBuiltPages", "clasTotalBuiltClusters",
                   "textureResidentBytes", "textureUpgrades", "textureDowngrades")
    reference_keys = ("blasStorageRejected", "blasFailedAllocations", "blasReferenceBudgetRejected",
                      "blasBuildBudgetRejected", "blasMissingClasInstances", "blasInvalidGroups",
                      "blasRejectedVisibleConsumers", "blasRejectedOffscreenConsumers")
    for pose in range(len(poses)):
        group = {v["name"]: v for v in variants if v["pose"] == pose}
        identities = {v["sourceIdentity"] for v in group.values()}
        if len(identities) != 1 or not next(iter(identities)):
            raise ValueError("Frozen source identity mismatch")
        if any(v["camera"] != group["A1"]["camera"] for v in group.values()):
            raise ValueError("Camera changed across controlled policies")
        pose_rows = [row for row in rows if row["pose"] == pose]
        for key in frozen_keys:
            values = {row["state"].get(key) for row in pose_rows}
            if None in values or len(values) != 1:
                failures.append(f"pose{pose}: frozen {key} changed or missing: {sorted(values, key=str)}")
        for name, variant in group.items():
            budget = config["referenceBytes"] if name == "R" else config["storageBytes"]
            expected = ((name != "R", name in ("A1", "A2", "B1")) if merged_sharing
                        else (name.startswith("B"), name == "B1"))
            if variant["storageBytes"] != budget or (variant["priority"], variant["cache"]) != expected:
                raise ValueError("Unexpected effective policy or storage budget")
            if merged_sharing and variant.get("mergedSharing") != name.startswith("B"):
                raise ValueError("Unexpected merged-sharing policy")
            selected = [row["state"] for row in pose_rows if row["variant"] == name]
            if len(variant["settle"]) != config.get("settleFrames", 16) or \
                    len(selected) != len(variant["settle"]) + len(seeds):
                raise ValueError("Variant frame coverage mismatch")
            for state in selected:
                if state["diagnosticSourceIdentity"] != variant["sourceIdentity"] or \
                        (state["blasPriorityEnabled"], state["blasPriorityCacheEnabled"]) != expected or \
                        not state["blasGPUAllocatorEnabled"] or not state["blasRepresentationsEnabled"]:
                    raise ValueError("Runtime source or mode identity mismatch")
                if merged_sharing and state.get("blasMergedSharingEnabled") != name.startswith("B"):
                    raise ValueError("Runtime merged-sharing policy mismatch")
            # CPU feedback is delayed. Judge only distinct completed producers
            # belonging to this epoch, never reinterpret it as image-frame truth.
            start_frame = selected[0]["frame"]
            feedback = {s["blasFeedbackFrame"]: s for s in selected if s["blasFeedbackAvailable"] and
                        start_frame <= s["blasFeedbackFrame"] < s["frame"]}
            tail = [feedback[f] for f in sorted(feedback)[-4:]]
            settled = len(tail) == 4 and all(s["blasBuildCount"] == 0 and s["blasPriorityEvictedOwners"] == 0 and
                                             s["blasArenaRepack"] == 0 for s in tail)
            if not settled:
                failures.append(f"pose{pose}/{name}: fewer than four fresh idle BLAS producers")
            audit = dict(pose=pose, variant=name, settled=settled, producerFrames=sorted(feedback), tail=tail)
            if name == "R":
                audit["referenceClean"] = bool(tail) and all(s[k] == 0 for s in tail for k in reference_keys)
                if not audit["referenceClean"]:
                    failures.append(f"pose{pose}/R: high-budget reference still has rejection/failure/missing CLAS")
            audits.append(audit)
        for seed in seeds:
            images, hashes = {}, {}
            for name, variant in group.items():
                matches = [item for item in variant["captures"] if item["seed"] == seed]
                if len(matches) != 1 or len(variant["captures"]) != len(seeds):
                    raise ValueError("Missing or duplicate fixed-seed capture")
                directory = (root / matches[0]["directory"]).resolve()
                if not directory.is_relative_to(root):
                    raise ValueError("Capture directory escapes run")
                images[name], hashes[name] = image_arrays(directory, sources)
            if any(images[name].keys() != images["A1"].keys() for name in NAMES):
                raise ValueError("Auxiliary raster capture coverage differs")
            if capture["controlledBlas"].get("captureRasterInputs") and not all(key in images["A1"] for key in AUXILIARY):
                raise ValueError("Requested raster diagnostic inputs are absent")
            if all(key in images["A1"] for key in AUXILIARY):
                for left, right in (("A1", "A2"), ("B0", "B1")):
                    masks = {key: np.any(images[left][key] != images[right][key], axis=-1)
                             for key in (*AUXILIARY, "Shadows.shadow", "Deferred.color")}
                    item = dict(pose=pose, seed=seed, pair=[left,right], resources={})
                    for key in AUXILIARY:
                        mask = masks[key]
                        ys, xs = np.where(mask)
                        neighborhood = mask.copy()
                        # shadowDepthNormal uses four direct neighboring depths.
                        neighborhood[1:] |= mask[:-1]
                        neighborhood[:-1] |= mask[1:]
                        neighborhood[:,1:] |= mask[:,:-1]
                        neighborhood[:,:-1] |= mask[:,1:]
                        item["resources"][key] = dict(exact=hashes[left][key] == hashes[right][key],
                            mismatchingPixels=int(mask.sum()), totalPixels=int(mask.size),
                            coordinatesFirst256=[[int(x),int(y)] for x,y in zip(xs[:256],ys[:256])],
                            coordinatesTruncated=len(xs)>256,
                            shadowDifferencePixels=int(masks["Shadows.shadow"].sum()),
                            shadowDirectOverlap=int(np.sum(mask & masks["Shadows.shadow"])),
                            shadowNeighborOverlap=int(np.sum(neighborhood & masks["Shadows.shadow"])),
                            deferredDifferencePixels=int(masks["Deferred.color"].sum()),
                            deferredDirectOverlap=int(np.sum(mask & masks["Deferred.color"])),
                            valuesFirst256=[dict(x=int(x),y=int(y),left=images[left][key][y,x].tolist(),
                                right=images[right][key][y,x].tolist()) for x,y in zip(xs[:256],ys[:256])])
                        if not item["resources"][key]["exact"]:
                            failures.append(f"pose{pose}/seed{seed}/{key}: {left}/{right} are not byte-identical")
                    raster_audits.append(item)
            for resource in RESOURCES:
                data = {name: images[name][resource] for name in NAMES}
                if any(value.shape != data["R"].shape for value in data.values()):
                    raise ValueError("Image extent/channel mismatch")
                aa = hashes["A1"][resource] == hashes["A2"][resource]
                cache_equal = hashes["B0"][resource] == hashes["B1"][resource]
                if not aa:
                    failures.append(f"pose{pose}/seed{seed}/{resource}: A1/A2 are not byte-identical")
                if not cache_equal:
                    failures.append(f"pose{pose}/seed{seed}/{resource}: B0/B1 are not byte-identical")
                reference = data["R"]
                h, w = reference.shape[:2]
                regions = {"fullFrame": (0, 0, w, h), "central50PercentProxy": (w//4, h//4, 3*w//4, 3*h//4)}
                entry = dict(pose=pose, seed=seed, resource=resource, repeatExact=aa, cacheExact=cache_equal,
                             imageHashes={name: hashes[name][resource] for name in NAMES}, regions={})
                for region, (x0, y0, x1, y1) in regions.items():
                    ref = reference[y0:y1, x0:x1]
                    before, repeat, after = (errors(data[name][y0:y1, x0:x1], ref) for name in ("A1", "A2", "B1"))
                    entry["regions"][region] = dict(rectangle=[x0,y0,x1,y1], baseline=before,
                        baselineRepeat=repeat, candidate=after,
                        baselineRepeatRange={key: [min(before[key],repeat[key]),max(before[key],repeat[key])] for key in before},
                        candidateMinusBaseline={key: after[key]-before[key] for key in before},
                        candidateMinusBaselineRepeat={key: after[key]-repeat[key] for key in before})
                squared = (data["B1"] - reference) ** 2
                excess = squared.sum(axis=-1) - ((data["A1"] - reference) ** 2).sum(axis=-1)
                indices = np.argsort(excess.reshape(-1))[-10:][::-1]
                total = float(squared.sum())
                entry["largestSquaredErrorRegressions"] = [dict(x=int(i%w), y=int(i//w),
                    baseline=data["A1"][i//w,i%w].tolist(), candidate=data["B1"][i//w,i%w].tolist(),
                    reference=reference[i//w,i%w].tolist(), squaredErrorExcess=float(excess[i//w,i%w]),
                    candidateSquaredErrorFraction=float(squared[i//w,i%w].sum()/total) if total else 0)
                    for i in indices]
                entry["referenceRange"] = [float(reference.min()), float(reference.max())]
                entry["referenceContent"] = {}
                for region, (x0, y0, x1, y1) in regions.items():
                    selected = reference[y0:y1,x0:x1]
                    content = dict(min=float(selected.min()), max=float(selected.max()),
                        mean=float(selected.mean()), standardDeviation=float(selected.std()),
                        positivePixelFraction=float(np.mean(selected.max(axis=-1) > 0)),
                        spatiallyNonconstant=bool(np.any(selected != selected[0,0])))
                    if resource == "Shadows.shadow":
                        content.update(shadowedPixelFraction=float(np.mean(selected[...,0] < .99)),
                                       litPixelFraction=float(np.mean(selected[...,0] >= .99)))
                    entry["referenceContent"][region] = content
                if float(reference.max()) == float(reference.min()):
                    failures.append(f"pose{pose}/seed{seed}/{resource}: constant reference image")
                comparisons.append(entry)
                fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
                for axis, name in zip(axes[0], ("A1", "B1", "R")):
                    shown = display(data[name])
                    axis.imshow(shown[...,0] if shown.shape[-1] == 1 else shown, cmap="gray", vmin=0, vmax=1)
                    axis.set_title(name)
                for axis, name in zip(axes[1,:2], ("A1", "B1")):
                    heat = np.abs(display(data[name])-display(reference)).max(axis=-1)
                    axis.imshow(heat, cmap="magma", vmin=0, vmax=.1)
                    axis.set_title(name + " vs R, fixed display |error| [0,.1]")
                log_excess = np.log10(1+np.maximum(excess,0))
                raw = axes[1,2].imshow(log_excess, cmap="magma", vmin=0, vmax=max(1e-12, float(log_excess.max())))
                fig.colorbar(raw, ax=axes[1,2], label="log10(1 + positive raw squared-error excess)")
                axes[1,2].set_title("Candidate regression localization")
                for axis in axes.flat:
                    axis.axis("off")
                fig.suptitle(f"MiniZorah pose{pose} seed{seed}: {resource}; A/A={aa}, cache equality={cache_equal}")
                filename = f"pose{pose}-seed{seed}-{resource}.png"
                fig.savefig(output / filename, dpi=130)
                plt.close(fig)
                plots.append(filename)
                if resource == "ReflectionQuality.color" and float(excess.max()) > 0:
                    peak = int(indices[0])
                    x, y = peak % w, peak // w
                    x0, y0, x1, y1 = max(0,x-48), max(0,y-48), min(w,x+49), min(h,y+49)
                    entry["peakCropRectangle"] = [x0,y0,x1,y1]
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                    for axis, name in zip(axes, ("A1", "B1", "R")):
                        axis.imshow(display(data[name][y0:y1,x0:x1]))
                        axis.plot(x-x0, y-y0, "+", color="cyan", markersize=12)
                        axis.set_title(name + " RGB=" + np.array2string(data[name][y,x], precision=3))
                        axis.axis("off")
                    fig.suptitle(f"Fixed identical crop {x0,y0,x1,y1}; peak raw regression pixel {x,y}; not a material/hit identification")
                    filename = f"pose{pose}-seed{seed}-ReflectionQuality-PeakCrop.png"
                    fig.savefig(output / filename, dpi=140)
                    plt.close(fig)
                    plots.append(filename)
    script = Path(__file__).resolve()
    sources.append(identity(script))
    shutil.copyfile(script, output / "AnalyzeFrozenBLASQuality.py")
    report = dict(protocol=protocol, status="valid" if not failures else "not_accepted", failures=failures,
        producerAlignment="Completed producer IDs retained; last four distinct post-reset producers, not image-frame attribution.",
        sourceScope=capture["controlledBlas"], manifest=manifest, rawSources=sources,
        versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__),
        audits=audits, comparisons=comparisons, plots=plots, rasterInputAudits=raster_audits,
        priorityOnlyInstrumentation=dict(fields=["blasPostBuildBudgetRejected", "blasPhysicalAllocationFailed",
            "blasUnknownBuildCount", "blasProjectedRetryDeferredCount", "blasProjectedDirtyClaimBytes"],
            baselineInterpretation="Raw baseline zeros are not instrumented/NA, not evidence of absence. Reference failure audit uses the shared total blasFailedAllocations instead."),
        limits=["Central 50% is a geometric rectangle, not a semantic foreground mask.",
                "OpenPBR path tracing includes direct visibility and reflections; it is not isolated specular radiance.",
                "Owner rejection counts are selection proxies, not image quality.",
                "One process frozen source and cold RT resets; no streaming or performance speedup claim.",
                "HDR outlier concentration alone does not establish geometry, material, normal or PDF cause."])
    (output / "Analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Frozen MiniZorah BLAS quality", "", f"Status: **{report['status']}**. No performance claim.", ""]
    if merged_sharing:
        lines += ["A1/A2: priority and cache on, merged sharing off. B0/B1: priority on, merged sharing on, cache off/on. "
                  "R: ample budget, priority/cache/merged sharing off. All variants share frozen source residency.", ""]
    if failures:
        lines += ["Acceptance gaps:", ""] + ["- " + reason for reason in failures] + [""]
    lines += ["| Pose/seed | Resource | A/A exact | B0/B1 exact | Raw MAE A1 / A2 → B | Raw RMSE A1 / A2 → B | Display >1% A1 / A2 → B |",
              "|---|---|---|---|---|---|---|"]
    for item in comparisons:
        a, repeat, b = (item["regions"]["fullFrame"][name] for name in ("baseline", "baselineRepeat", "candidate"))
        lines.append(f"| {item['pose']}/{item['seed']} | {item['resource']} | {item['repeatExact']} | {item['cacheExact']} | "
                     f"{a['rawMAE']:.6g} / {repeat['rawMAE']:.6g} → {b['rawMAE']:.6g} | "
                     f"{a['rawRMSE']:.6g} / {repeat['rawRMSE']:.6g} → {b['rawRMSE']:.6g} | "
                     f"{100*a['displayPixelsOverOnePercent']:.4g}% / {100*repeat['displayPixelsOverOnePercent']:.4g}% → {100*b['displayPixelsOverOnePercent']:.4g}% |")
    if raster_audits:
        lines += ["", "Raster diagnostics (full coordinates and depth/visibility values in Analysis.json):", ""]
        for item in raster_audits:
            for key, value in item["resources"].items():
                lines.append(f"- Pose{item['pose']} seed{item['seed']} {'/'.join(item['pair'])} {key}: "
                    f"{value['mismatchingPixels']} changed pixels; overlaps shadow differences "
                    f"{value['shadowDirectOverlap']} directly / {value['shadowNeighborOverlap']} within depth-normal neighbors "
                    f"of {value['shadowDifferencePixels']}; Deferred direct overlap "
                    f"{value['deferredDirectOverlap']} of {value['deferredDifferencePixels']}.")
    lines += [""] + report["limits"] + ["", "Raw image hashes, source identities, producer feedback and HDR pixel values are in Analysis.json."]
    (output / "Review.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), status=report["status"], failures=failures), indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
