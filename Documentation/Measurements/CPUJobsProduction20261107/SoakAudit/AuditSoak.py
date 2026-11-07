"""Offline soak correctness/residency gate; no renderer or timing process launched.

python -B AuditSoak.py --input <restored soak evidence root> --output <new output dir>
Reads full frame JSONL streams, preserves unavailable data, checks hashes from Raw.
Requires numpy and Pillow only for native RGBA16F image inspection/one preview.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def summarize(points, tail_first):
    if not points:
        return None
    values = [value for _, value in points]
    tail = [(frame, value) for frame, value in points if frame >= tail_first]
    tail_values = [value for _, value in tail]
    return {"samples": len(points), "firstFrame": points[0][0], "lastFrame": points[-1][0],
        "first": values[0], "last": values[-1], "min": min(values), "max": max(values),
        "tail": {"firstFrame": tail[0][0], "lastFrame": tail[-1][0], "samples": len(tail),
            "first": tail_values[0], "last": tail_values[-1], "min": min(tail_values), "max": max(tail_values),
            "constant": min(tail_values) == max(tail_values)} if tail else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if (output / "Audit.json").exists():
        parser.error("Audit.json already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / Path(__file__).name
    if snapshot.resolve() != Path(__file__).resolve():
        snapshot.write_bytes(Path(__file__).read_bytes())
    metadata = json.loads((root / "Metadata.json").read_text(encoding="utf-8-sig"))
    raw = [json.loads(line) for line in (root / "Raw.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    errors, records, images = [], [], []
    input_hashes = {str((root / name).resolve()): sha(root / name) for name in ["Metadata.json", "Raw.jsonl"]}
    identities = []
    if metadata.get("status") != "complete":
        errors.append("Collection status is not complete")
    if len(raw) != 4:
        errors.append(f"Expected four ABBA processes, observed {len(raw)}")
    for row in raw:
        run = row["runId"]
        if row.get("status") != "passed" or row.get("phase") != "measured":
            errors.append(f"P{run}: not a passed measured process")
        paths = {key: root / value.replace("\\", "/") for key, value in row["artifacts"].items()}
        capture = json.loads(paths["capture"].read_text(encoding="utf-8-sig"))
        for key in ["capture", "image"]:
            actual = sha(paths[key])
            input_hashes[str(paths[key].resolve())] = actual
            if actual != row["artifactSha256"][key]:
                errors.append(f"P{run}: {key} SHA mismatch")
        identity = {key: capture.get(key) for key in ["scene", "camera", "graph", "outputExtent", "renderExtent", "workerCount", "recordingWorkerLimit"]}
        identities.append(identity)
        window = row["measuredFrameRange"]
        expected_frames = list(range(window["first"], window["last"] + 1))
        tail_first = window["last"] - 599
        envelope_ids = {scope["id"] for scope in capture["scopes"] if scope["path"].endswith("/RenderGraph GPU envelope")}
        frames, gpu_count, streaming_count, memory_count = [], 0, 0, 0
        generations, assets = set(), set()
        series = defaultdict(list)
        counters = defaultdict(list)
        digest = hashlib.sha256()
        with paths["frames"].open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                frame = json.loads(line)
                index = frame["frame"]
                frames.append(index)
                generations.add(frame.get("graphGeneration"))
                gpu_values = [scope.get("gpuMs") for scope in frame["scopes"] if scope["id"] in envelope_ids]
                if len(gpu_values) == 1 and gpu_values[0] is not None and np.isfinite(gpu_values[0]):
                    gpu_count += 1
                for item in frame.get("streaming") or []:
                    streaming_count += 1
                    assets.add((item.get("pass"), item.get("asset")))
                    for key in ["loadFailures", "allocationFailures", "pendingPages", "texturePendingBytes", "texturePendingImages"]:
                        value = item.get(key)
                        counters[key].append((index, value))
                        if value is None or value != 0:
                            errors.append(f"P{run} frame {index}: {key}={value!r}")
                    for key in ["residentPages", "geometryBytes", "clasAllocatedBytes", "textureResidentBytes", "textureRefinedImages"]:
                        if item.get(key) is not None:
                            series["streaming." + key].append((index, item[key]))
                memory = frame.get("deviceMemory")
                if memory is not None:
                    memory_count += 1
                    value = memory.get("deniedAllocations")
                    counters["deniedAllocations"].append((index, value))
                    if value is None or value != 0:
                        errors.append(f"P{run} frame {index}: deniedAllocations={value!r}")
                    for key in ["trackedAllocationBytes", "trackedDeviceLocalBytes", "peakTrackedAllocationBytes", "peakTrackedDeviceLocalBytes"]:
                        if memory.get(key) is not None:
                            series["rhi." + key].append((index, memory[key]))
        actual = digest.hexdigest()
        input_hashes[str(paths["frames"].resolve())] = actual
        if actual != row["artifactSha256"]["frames"]:
            errors.append(f"P{run}: Frames SHA mismatch")
        if frames != expected_frames or len(frames) != 6000 or capture.get("frames") != 6000:
            errors.append(f"P{run}: frame population is not exactly 6000 ordered consecutive measured frames")
        if gpu_count != len(frames) or capture.get("missingGpuFrames") != 0:
            errors.append(f"P{run}: incomplete GPU frame envelopes")
        if streaming_count != len(frames) or len(assets) != 1 or len(generations) != 1:
            errors.append(f"P{run}: missing/variable streaming identity or graph generation")
        if not memory_count:
            errors.append(f"P{run}: no device-memory observations")
        snapshots = {}
        for key in ["deviceMemoryAfterLoad", "deviceMemoryAfterWarmup", "deviceMemoryAfterCapture"]:
            item = capture.get(key)
            snapshots[key] = {name: item.get(name) for name in ["deniedAllocations", "trackedAllocationBytes", "trackedDeviceLocalBytes", "peakTrackedAllocationBytes", "peakTrackedDeviceLocalBytes"]} if item else None
            if item and item.get("deniedAllocations") != 0:
                errors.append(f"P{run}: {key} deniedAllocations is not zero")
        descriptor = capture["image"]
        if descriptor["format"] != "RGBA16Sfloat" or descriptor.get("colorEncoding") != 2:
            errors.append(f"P{run}: expected native scRGB RGBA16F output")
        native = np.fromfile(paths["image"], dtype="<f2").astype(np.float32)
        expected = descriptor["width"] * descriptor["height"] * 4
        if native.size != expected:
            raise ValueError(f"P{run}: raw image extent/size mismatch")
        native = native.reshape(descriptor["height"], descriptor["width"], 4)
        finite = bool(np.isfinite(native).all())
        alpha_one = bool(np.all(native[:, :, 3] == 1))
        if not finite or not alpha_one:
            errors.append(f"P{run}: image finite={finite}, alphaOne={alpha_one}")
        image_stats = {"finiteRGBA": finite, "alphaOne": alpha_one, "alphaMin": float(native[:, :, 3].min()),
            "alphaMax": float(native[:, :, 3].max()), "rgbMin": float(native[:, :, :3].min()),
            "rgbMax": float(native[:, :, :3].max()), "rgbMean": float(native[:, :, :3].mean(dtype=np.float64)),
            "descriptor": descriptor, "sha256": row["artifactSha256"]["image"]}
        images.append((row, native))
        records.append({"runId": run, "candidate": row["candidate"], "workers": row["workers"],
            "frameCount": len(frames), "frameRange": window, "gpuEnvelopeFrames": gpu_count,
            "streamingObservations": streaming_count, "deviceMemorySnapshots": memory_count,
            "graphGeneration": sorted(generations), "streamAssetIdentities": sorted(assets),
            "zeroCounterChecks": {key: {"observations": len(values), "allZero": all(value == 0 for _, value in values)} for key, values in counters.items()},
            "measuredMetrics": {key: summarize(values, tail_first) for key, values in series.items()},
            "captureMemorySnapshots": snapshots, "image": image_stats})
    if any(identity != identities[0] for identity in identities[1:]):
        errors.append("Scene/camera/graph/extents/worker identities differ across the four Captures")
    comparisons = []
    for left, right, kind in [(0, 1, "cross backend"), (2, 3, "cross backend"), (0, 3, "legacy repeat"), (1, 2, "shared repeat")]:
        a, b = images[left][1], images[right][1]
        delta = np.abs(a[:, :, :3] - b[:, :, :3])
        comparisons.append({"leftRun": images[left][0]["runId"], "rightRun": images[right][0]["runId"], "kind": kind,
            "rgbMAE": float(delta.mean(dtype=np.float64)), "rgbRMSE": float(np.sqrt(np.square(delta, dtype=np.float64).mean())),
            "rgbMaxAbsoluteDifference": float(delta.max()), "changedPixelFraction": float(np.any(delta != 0, axis=2).mean()),
            "alphaMaxAbsoluteDifference": float(np.abs(a[:, :, 3] - b[:, :, 3]).max()),
            "rawBytesEqual": records[left]["image"]["sha256"] == records[right]["image"]["sha256"]})
    if images and records[0]["image"]["finiteRGBA"]:
        rgb = np.clip(images[0][1][:, :, :3], 0, 1)
        display = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - .055)
        Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(output / "FinalLegacy0Preview.png")
    audit = {"schema": 1, "generatedUtc": datetime.now(timezone.utc).isoformat(), "status": "passed" if not errors else "failed",
        "inputRoot": str(root), "scriptSnapshot": snapshot.name, "scriptSHA256": sha(snapshot), "inputsSHA256": input_hashes,
        "population": "Four fresh-process ABBA instrumented residency/correctness soak runs, 180 warmup plus 6000 captured frames; not a normal FPS performance A/B.",
        "tailDefinition": "Last 600 captured frames; constant means observed min equals max, not an unobserved lifetime guarantee.",
        "identity": identities[0] if identities else None,
        "identityBoundary": "Frames export graphGeneration and streaming asset identity, not per-frame scene/camera/extents. Capture/config identities match across processes; per-frame scene/camera/extent values are unavailable.",
        "memoryBoundary": "RHI tracked current/high-water covers published RHI allocations, excludes untracked native/SDK/driver. Device-memory observations are sampled every 10 frames; no zero fill. Texture-resident bytes are not total VRAM.",
        "previewDefinition": "P0 native scRGB RGB clipped to [0,1], linear Rec709-to-sRGB transfer, no exposure/tonemap; alpha inspected separately.",
        "comparisonContext": "Same fixed scene/camera/graph/extents, all export after frame 6180. Raw scRGB comparisons are diagnostic against same-backend repeats, not a bitwise or statistical equivalence threshold.",
        "processes": records, "imageComparisons": comparisons, "errors": errors}
    (output / "Audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "processes": len(records), "errors": errors,
        "comparisons": comparisons, "output": str(output)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
