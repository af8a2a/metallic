#!/usr/bin/env python3
"""Decode controlled MiniZorah path/owner evidence without executing the renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

PROTOCOL = "minizorah-blas-path-diagnostics-v1"
WIDTH, SAMPLES, VERTEX_BASE, VERTEX_STRIDE, VERTICES = 104, 16, 8, 32, 3
INVALID = 0xffffffff
OWNER_FIELDS = ("clusterReferenceOffset", "clusterReferenceCapacity", "selectedClusterCount",
                "insertedClusterCount", "blasBuildIndex", "flags", "firstGroup", "groupCount",
                "previousFirstGroup", "previousGroupCount", "storageOffset", "storageCapacity",
                "allocationPrefix", "packedPrefix", "buildPrefix", "cachedValid")
TRACE = "ReflectionQuality.pathDiagnostics"
PREFIX = "streaming.VBuffer."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def child(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), "Captured file escapes directory")
    return path


def numeric(words):
    """Only call for declared float fields; integer NaN bit patterns stay uints."""
    values = np.asarray(words, dtype="<u4").view("<f4")
    require(np.isfinite(values).all(), "Nonfinite value in a declared numeric trace field")
    return values.astype(float).tolist()


def decode_vertex(row, depth):
    w = row[VERTEX_BASE + depth * VERTEX_STRIDE:VERTEX_BASE + (depth + 1) * VERTEX_STRIDE]
    require(int(w[0, 0]) in (0, 1), "Unexpected vertex valid marker")
    if w[0, 0] == 0:
        return None
    found = int(w[0, 1])
    require(found in (0, 1), "Unexpected hit marker")
    vertex = dict(depth=depth, found=bool(found), seedBefore=int(w[0, 2]), family=int(w[0, 3]),
                  incomingOrigin=numeric(w[6, :3]), coneWidth=numeric(w[6, 3:])[0],
                  incomingDirection=numeric(w[7, :3]), coneSpread=numeric(w[7, 3:])[0],
                  throughputBefore=numeric(w[8, :3]), previousPdf=numeric(w[8, 3:])[0],
                  radianceBefore=numeric(w[9, :3]), previousDelta=numeric(w[9, 3:])[0],
                  emission=numeric(w[12, :3]), directLighting=numeric(w[13, :3]),
                  environmentNEE=numeric(w[14, :3]), environmentTransmittance=numeric(w[15, :3]),
                  sampledEnvironment=numeric(w[15, 3:])[0], environmentDirection=numeric(w[16, :3]),
                  environmentPdf=numeric(w[16, 3:])[0], scatterDirection=numeric(w[17, :3]),
                  scatterPdf=numeric(w[17, 3:])[0], scatterWeight=numeric(w[18, :3]),
                  scatterFlags=int(w[18, 3]), throughputAfter=numeric(w[19, :3]), seedAfter=int(w[19, 3]),
                  radianceAfter=numeric(w[20, :3]), control=int(w[20, 3]),
                  outgoingOrigin=numeric(w[21, :3]), depthAfter=numeric(w[21, 3:])[0],
                  outgoingDirection=numeric(w[22, :3]))
    if found:
        cluster = int(w[2, 1])
        vertex["hit"] = dict(material=int(w[1, 0]), gpuSceneInstance=int(w[1, 1]),
                             primitive=int(w[1, 2]), triangle=int(w[1, 3]),
                             rawStreamInstance=int(w[2, 0]), cluster=cluster,
                             page=cluster >> 5, localCluster=cluster & 31,
                             barycentrics=numeric(w[2, 2:]), position=numeric(w[3, :3]),
                             t=numeric(w[3, 3:])[0], geometryNormal=numeric(w[4, :3]),
                             frontFace=numeric(w[4, 3:])[0], authoredNormal=numeric(w[5, :3]),
                             rayConeLOD=numeric(w[5, 3:])[0], evaluatedNormal=numeric(w[10, :3]),
                             roughness=numeric(w[10, 3:])[0], uv=numeric(w[11, :2]),
                             metalness=numeric(w[11, 2:3])[0], ior=numeric(w[11, 3:])[0])
    vertex["lights"] = []
    for index in range(3):
        base = 23 + 3 * index
        vertex["lights"].append(dict(slot=index, direction=numeric(w[base, :3]),
            distance=numeric(w[base, 3:])[0], visibility=numeric(w[base + 1, :3]),
            source=int(w[base + 1, 3]), contribution=numeric(w[base + 2, :3]),
            inverseSourcePdf=numeric(w[base + 2, 3:])[0]))
    return vertex


def decode_sample(row, target, sample, pixels, samples):
    require(row.shape == (WIDTH, 4), "Trace row shape mismatch")
    require(row[0, :3].tolist() == [*pixels[target], sample], "Trace target/sample identity mismatch")
    require(int(row[4, 2]) == samples and 1 <= samples <= SAMPLES, "Trace samples mismatch")
    require(1 <= int(row[4, 3]) <= VERTICES, "Trace maxDepth exceeds captured vertex slots")
    result = dict(target=target, pixel=pixels[target], sample=sample, seedBeforeJitter=int(row[0, 3]),
                  seedBeforeTrace=int(row[1, 3]), sampleFrame=int(row[2, 3]), seedAfterTrace=int(row[3, 3]),
                  primaryOrigin=numeric(row[1, :3]), primaryDirection=numeric(row[2, :3]),
                  jitter=numeric(row[4, :2]), finalRadiance=numeric(row[3, :3]),
                  samples=samples, maxDepth=int(row[4, 3]), visits=int(row[5, 0]),
                  alphaLayersBeforeLastVertex=int(row[5, 1]), vertices=[], rawUint32=row.tolist())
    for depth in range(VERTICES):
        vertex = decode_vertex(row, depth)
        if vertex:
            result["vertices"].append(vertex)
    require(result["vertices"] and result["vertices"][0]["depth"] == 0, "Missing primary vertex")
    require(result["visits"] >= len(result["vertices"]), "Visit count is smaller than captured vertices")
    result["completeVertexHistory"] = result["visits"] == len(result["vertices"]) and result["alphaLayersBeforeLastVertex"] == 0
    return result


class Owners:
    def __init__(self, representations, state, tlas):
        require(representations.size % 16 == 0 and tlas.size % 16 == 0, "Expected 64-byte owner/TLAS records")
        self.representations = representations.reshape(-1, 16)
        self.tlas = tlas.reshape(-1, 16)
        self.state = state
        self.count, self.instances = len(self.representations), len(self.tlas)
        require(self.count > 0 and self.count % 3 == 0, "Expected three stable owners per primitive")
        self.instance_base = len(state) - 12 * self.count - 2 - 4 * self.instances
        require(self.instance_base >= 0, "RT state is too short for current layout")
        self.importance_base = self.instance_base + 4 * self.instances
        self.order_base = self.importance_base + 6 * self.count

    def owner(self, owner):
        require(0 <= owner < self.count, "Resolved owner outside captured array")
        result = dict(zip(OWNER_FIELDS, map(int, self.representations[owner])))
        result["owner"] = owner
        result["mode"] = ("merged", "shared", "discrete")[owner % 3]
        result["flagsDecoded"] = [label for bit, label in ((1, "fallback"), (2, "dynamic"), (4, "overflow"),
            (8, "dirty"), (16, "semanticCutDirty")) if result["flags"] & bit]
        result["dynamicReady"] = bool(result["flags"] & 2 and not result["flags"] & 5 and
                                      result["cachedValid"] == 1 and result["storageCapacity"] != 0)
        importance = self.state[self.importance_base + 6 * owner:self.importance_base + 6 * owner + 6]
        result["importanceRaw"] = importance.tolist()
        result["rankRaw"] = int(self.state[self.order_base + 2 * self.count + owner])
        return result

    def join(self, hit):
        instance, primitive = hit["rawStreamInstance"], hit["primitive"]
        require(instance < self.instances, "Raw stream instance outside TLAS; do not use GPUScene ID here")
        require(primitive * 3 + 2 < self.count, "Hit primitive outside owner array")
        selected = int(self.state[self.instance_base + 4 * instance + 2])
        tlas = self.tlas[instance]
        require(int(tlas[12]) & 0xffffff == instance, "TLAS custom ID does not match raw stream instance")
        evidence = dict(resolvedOwner=None if selected == INVALID else selected,
                        terminalFallback=selected == INVALID,
                        requestedOwner="not recorded; phase24 overwrote the desired owner",
                        tlasAddressHex=f"0x{int(tlas[15]):08x}{int(tlas[14]):08x}",
                        primitiveOwners=[self.owner(primitive * 3 + mode) for mode in range(3)])
        if selected != INVALID:
            require(selected // 3 == primitive, "Resolved owner belongs to another primitive")
            evidence["selected"] = self.owner(selected)
        return evidence


class AssetPages:
    """Optional v9 directory lookup; no payload decode or whole-asset loading."""
    def __init__(self, path, captured):
        self.path = path.resolve()
        require(self.path.stat().st_size == captured["bytes"], "Asset size differs from captured manifest")
        with self.path.open("rb") as stream:
            header = stream.read(216)
        magic, version, endian, size = struct.unpack_from("<8sIIQ", header)
        require((magic, version, endian, size) == (b"MTLMSTRM", 9, 0x01020304, self.path.stat().st_size),
                "Only the verified v9 MeshletStreamFileHeader layout is supported")
        self.count = struct.unpack_from("<I", header, 88)[0]
        self.offset = struct.unpack_from("<Q", header, 176)[0]
        require(struct.unpack_from("<I", header, 112)[0] == 32 and self.offset + self.count * 104 <= size,
                "Page directory range/cluster stride mismatch")
        self.records = {}
        self.provenance = dict(path=str(self.path), fileBytes=size, headerSha256=hashlib.sha256(header).hexdigest(),
            pageRecordBytes=104, pageDirectoryOffset=self.offset,
            scope="Analysis-time v9 asset directory; file size matched capture. Selected record hashes below are not a whole-asset identity.")

    def page(self, page):
        require(0 <= page < self.count, "Hit page outside asset directory")
        if page not in self.records:
            with self.path.open("rb") as stream:
                stream.seek(self.offset + page * 104)
                raw = stream.read(104)
            primitive, lod, group, clusters = struct.unpack_from("<4I", raw, 24)
            self.records[page] = dict(page=page, primitive=primitive, lodLevel=lod, group=group,
                clusterCount=clusters, material=struct.unpack_from("<I", raw, 48)[0],
                recordOffset=self.offset + page * 104, recordSha256=hashlib.sha256(raw).hexdigest())
        return self.records[page]


def load_capture(directory, pixels, samples, sources, asset=None):
    manifest = directory / "Diagnostics.json"
    sources.append(identity(manifest))
    metadata = read_json(manifest)
    resources = {}
    for item in metadata:
        name = item["resource"]
        require(name not in resources, "Duplicate diagnostic resource")
        path = child(directory, item["file"])
        sources.append(identity(path))
        words = np.fromfile(path, dtype="<u4")
        require(path.stat().st_size % 4 == 0, "Unaligned diagnostic bytes")
        if name == TRACE:
            require(item["format"] == "float32x4" and item["width"] == WIDTH and
                    item["height"] == SAMPLES * len(pixels), "Unknown trace texture layout")
            words = words.reshape(item["height"], WIDTH, 4)
        else:
            require(item["bytes"] == path.stat().st_size, "Diagnostic buffer length mismatch")
        resources[name] = words
    require(set(resources) == {TRACE, PREFIX + "rtRepresentations", PREFIX + "rtRepresentationState",
                               PREFIX + "tlasInstances"}, "Missing or unexpected diagnostic resources")
    owners = Owners(resources[PREFIX + "rtRepresentations"], resources[PREFIX + "rtRepresentationState"],
                    resources[PREFIX + "tlasInstances"])
    image_manifest = directory / "Images.json"
    sources.append(identity(image_manifest))
    images = [v for v in read_json(image_manifest) if v["resource"] == "ReflectionQuality.color"]
    require(len(images) == 1 and images[0]["format"] == "float32x4", "Missing float32 HDR output")
    item = images[0]
    image_path = child(directory, item["file"])
    sources.append(identity(image_path))
    hdr = np.fromfile(image_path, dtype="<f4").reshape(item["height"], item["width"], 4)
    result = dict(directory=str(directory), ownerCount=owners.count, instanceCount=owners.instances,
                  instanceStateBase=owners.instance_base, targets=[])
    for target, pixel in enumerate(pixels):
        require(not np.any(resources[TRACE][target * SAMPLES + samples:(target + 1) * SAMPLES]),
                "Unused trace rows were not cleared")
        decoded = [decode_sample(resources[TRACE][target * SAMPLES + sample], target, sample, pixels, samples)
                   for sample in range(samples)]
        for sample in decoded:
            for vertex in sample["vertices"]:
                if vertex["found"]:
                    vertex["ownerEvidence"] = owners.join(vertex["hit"])
                    if asset:
                        vertex["hit"]["assetPage"] = asset.page(vertex["hit"]["page"])
                        require(vertex["hit"]["assetPage"]["primitive"] == vertex["hit"]["primitive"],
                                "Trace primitive does not match captured cluster's asset page")
        x, y = pixel
        require(0 <= x < item["width"] and 0 <= y < item["height"], "Target outside HDR image")
        rgb = hdr[y, x, :3].astype(float)
        require(np.isfinite(rgb).all(), "Nonfinite HDR target")
        mean = np.mean([s["finalRadiance"] for s in decoded], axis=0)
        result["targets"].append(dict(pixel=pixel, samples=decoded, hdr=rgb.tolist(),
            sampleMean=mean.tolist(), reconstructionMaxError=float(np.max(np.abs(rgb - mean))),
            reconstructed=bool(np.allclose(rgb, mean, rtol=2e-5, atol=2e-5))))
    return result


def first_difference(a, b, path=""):
    if type(a) is not type(b):
        return dict(field=path, left=a, right=b)
    if isinstance(a, dict):
        for key in dict.fromkeys([*a, *b]):
            if key in ("rawUint32", "ownerEvidence"):
                continue
            if key not in a or key not in b:
                return dict(field=path + "." + key, left=a.get(key), right=b.get(key))
            result = first_difference(a[key], b[key], path + "." + key)
            if result:
                return result
    elif isinstance(a, list):
        for index, (left, right) in enumerate(zip(a, b)):
            result = first_difference(left, right, f"{path}[{index}]")
            if result:
                return result
        if len(a) != len(b):
            return dict(field=path + ".length", left=len(a), right=len(b))
    elif a != b:
        return dict(field=path, left=a, right=b)
    return None


def compare(left, right):
    comparisons = []
    for a, b in zip(left["targets"], right["targets"]):
        require(a["pixel"] == b["pixel"], "Comparison target mismatch")
        row = dict(pixel=a["pixel"], hdrLeft=a["hdr"], hdrRight=b["hdr"], samples=[])
        for sa, sb in zip(a["samples"], b["samples"]):
            # Compare in execution order rather than dictionary/header order:
            # final radiance/seed are outputs and cannot be the first cause.
            inputs = ("seedBeforeJitter", "jitter", "seedBeforeTrace", "primaryOrigin", "primaryDirection")
            first = first_difference({k: sa[k] for k in inputs}, {k: sb[k] for k in inputs}, "primaryInput")
            vertices = []
            for va, vb in zip(sa["vertices"], sb["vertices"]):
                ordered = ("depth", "incomingOrigin", "incomingDirection", "seedBefore", "found", "hit",
                           "family", "throughputBefore", "emission", "lights", "directLighting", "environmentNEE",
                           "environmentTransmittance", "scatterDirection", "scatterPdf", "scatterWeight",
                           "scatterFlags", "radianceAfter", "throughputAfter", "seedAfter", "control")
                difference = first_difference({k: va.get(k) for k in ordered},
                                              {k: vb.get(k) for k in ordered}, f"vertex{va['depth']}")
                vertices.append(dict(depth=va["depth"], firstDivergence=difference,
                                     ownerLeft=va.get("ownerEvidence"), ownerRight=vb.get("ownerEvidence")))
                if first is None:
                    first = difference
            if first is None and len(sa["vertices"]) != len(sb["vertices"]):
                first = dict(field="vertexCount", left=len(sa["vertices"]), right=len(sb["vertices"]))
            row["samples"].append(dict(sample=sa["sample"], firstDivergence=first, vertices=vertices,
                                        completeVertexHistories=sa["completeVertexHistory"] and sb["completeVertexHistory"],
                                        finalLeft=sa["finalRadiance"], finalRight=sb["finalRadiance"]))
        comparisons.append(row)
    return comparisons


def contribution_components(target):
    values = np.zeros(6, dtype=float)
    for sample in target["samples"]:
        if not sample["completeVertexHistory"]:
            values[5] += np.mean(sample["finalRadiance"])
            continue
        before_aerial = np.zeros(3)
        for vertex in sample["vertices"]:
            delta = np.asarray(vertex["radianceAfter"]) - vertex["radianceBefore"]
            before_aerial += delta
            if vertex["depth"] != 0:
                values[4] += delta.mean()
            elif not vertex["found"]:
                values[0] += delta.mean()
            elif vertex["family"] != 0:
                values[5] += delta.mean()
            else:
                throughput = np.asarray(vertex["throughputBefore"])
                known = np.zeros(3)
                for index, key in enumerate(("emission", "directLighting", "environmentNEE"), 1):
                    term = throughput * vertex[key]
                    values[index] += term.mean()
                    known += term
                values[5] += (delta - known).mean()
        values[5] += (np.asarray(sample["finalRadiance"]) - before_aerial).mean()
    return values / len(target["samples"])


def plot_contributions(output, captures, pose, seed, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [name for name in ("A1", "A2", "B0", "B1", "R") if name in captures]
    labels = ("primary miss", "primary emission", "primary direct sources", "primary environment NEE",
              "secondary vertices (all lobes)", "aerial / unclassified residual")
    values = np.asarray([contribution_components(captures[name]["targets"][target]) for name in names])
    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    bottom = np.zeros(len(names))
    for index, label in enumerate(labels):
        axes[0].bar(names, values[:, index], bottom=bottom, label=label)
        bottom += values[:, index]
    axes[0].set_ylabel("Sample-mean radiance, RGB arithmetic mean")
    axes[0].legend(fontsize=7)
    for name in names:
        samples = captures[name]["targets"][target]["samples"]
        axes[1].plot([v["sample"] for v in samples], [np.mean(v["finalRadiance"]) for v in samples], marker=".", label=name)
    axes[1].set_xlabel("Sample index (later jitter can diverge)")
    axes[1].set_ylabel("Final sample radiance, RGB mean")
    axes[1].legend()
    pixel = captures[names[0]]["targets"][target]["pixel"]
    figure.suptitle(f"Pose {pose}, seed {seed}, pixel {pixel}; secondary is not isolated reflection")
    filename = f"pose{pose}-seed{seed}-pixel{pixel[0]}-{pixel[1]}-Contributions.png"
    figure.savefig(output / filename, dpi=140)
    plt.close(figure)
    return filename


def self_test():
    row = np.zeros((WIDTH, 4), dtype="<u4")
    row[0] = [11, 7, 0, INVALID]
    row[1, 3] = INVALID
    row[3, 3] = INVALID
    row[4, 2:] = [1, 3]
    row[5, 0] = 1
    row[8] = [1, 1, INVALID, 2]
    row[9] = [3, 99, 0, 1]
    row[10, :2] = [0, 33]
    sample = decode_sample(row, 0, 0, [[11, 7]], 1)
    assert sample["seedBeforeJitter"] == INVALID and sample["vertices"][0]["hit"]["page"] == 1
    assert sample["vertices"][0]["hit"]["gpuSceneInstance"] == 99
    owners = np.zeros(3 * 16, dtype="<u4")
    state = np.zeros(4 + 12 * 3 + 2, dtype="<u4")
    state[2] = INVALID
    tlas = np.zeros(16, dtype="<u4")
    evidence = Owners(owners, state, tlas).join(sample["vertices"][0]["hit"])
    assert evidence["terminalFallback"] and len(evidence["primitiveOwners"]) == 3
    row[11, 0] = INVALID  # A NaN in an actual float is malformed; NaN-like IDs above are valid.
    try:
        decode_sample(row, 0, 0, [[11, 7]], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Numeric NaN was not rejected")
    print("Path decoder self-test passed (mixed bits, raw/GPUScene IDs, owner tail, malformed numeric value).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--stream-asset", type=Path, help="Optional v9 asset for exact hit page/group/LOD lookup")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.root is None:
            return
    require(args.root is not None, "A controlled run directory is required")
    root = args.root.resolve()
    if not (root / "Capture.json").exists():
        matches = list(root.glob("run*/Capture.json"))
        require(len(matches) == 1, "Expected exactly one controlled run")
        root = matches[0].parent
    capture_path = root / "Capture.json"
    capture = read_json(capture_path)
    require(capture.get("status") == "capture_complete" and
            capture.get("protocol") in ("minizorah-blas-frozen-quality-v1", "minizorah-blas-frozen-quality-v2"),
            "Unexpected or incomplete capture")
    config = capture["config"]
    require(config.get("sample") == "gpu-driven-sample", "Expected controlled MiniZorah")
    pixels = config["diagnosticPixels"]
    require(isinstance(pixels, list) and pixels and all(len(pixel) == 2 for pixel in pixels), "Expected pixel pairs")
    samples = config.get("reflectionSamples", 8)
    output = (args.output or root.parent / "PathDiagnosticsAnalysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [identity(capture_path), identity(Path(__file__))]
    manifest_path = root.parent / "Manifest.json"
    manifest = read_json(manifest_path)
    sources.append(identity(manifest_path))
    require(manifest.get("config") == config, "Manifest/capture config mismatch")
    asset = AssetPages(args.stream_asset, manifest["asset"]) if args.stream_asset else None
    repository = Path(__file__).resolve().parent.parent
    source_paths = ["Shaders/Modules/OpenPBRPathTracing/OpenPBRPathTrace.slang",
                    "Shaders/Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang",
                    "Shaders/Modules/OpenPBRPathTracing/OpenPBRRadianceCache.slang",
                    "Shaders/Modules/PathDiagnostics.slang",
                    "Shaders/Modules/ScenePathTracing/StreamSceneRayQuery.slang",
                    "Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang",
                    "Source/Runtime/Render/Streamer/MeshletStreamRuntime.h",
                    "Source/Editor/EditorRayQualityCapture.h"]
    current_source = [identity(repository / path) for path in source_paths]
    decoded, comparisons, plots, failures, warnings = [], [], [], [], []
    variants = capture["controlledBlas"]["variants"]
    for pose in sorted({v["pose"] for v in variants}):
        group = {v["name"]: v for v in variants if v["pose"] == pose}
        require(set(group) == {"A1", "A2", "B0", "B1", "R"}, "Incomplete cold policy variants")
        require(len({v["sourceIdentity"] for v in group.values()}) == 1, "Frozen source identity mismatch")
        require(all(v["camera"] == group["R"]["camera"] for v in group.values()), "Pose camera mismatch")
        for seed in config["blasQualityComparison"].get("seeds", [0]):
            captures = {}
            for name, variant in group.items():
                matches = [c for c in variant["captures"] if c["seed"] == seed]
                require(len(matches) == 1, "Missing/duplicate fixed-seed capture")
                captures[name] = load_capture(child(root, matches[0]["directory"]), pixels, samples, sources, asset)
                for target in captures[name]["targets"]:
                    require(all(sample["sampleFrame"] == seed for sample in target["samples"]),
                            "Recorded sampleFrame does not match capture seed")
                    for sample in target["samples"]:
                        if not sample["completeVertexHistory"]:
                            warnings.append(f"pose{pose}/{name}/seed{seed}/{target['pixel']}/sample{sample['sample']}: repeated-depth/alpha history overwritten; radiance decomposition unavailable")
                    if not target["reconstructed"]:
                        failures.append(f"pose{pose}/{name}/seed{seed}/{target['pixel']}: sample radiance does not reconstruct HDR")
            decoded.append(dict(pose=pose, seed=seed, variants=captures))
            for left, right in (("A1", "A2"), ("B0", "B1"), ("A1", "R"), ("A2", "R"), ("B1", "R")):
                comparisons.append(dict(pose=pose, seed=seed, left=left, right=right,
                                        targets=compare(captures[left], captures[right])))
            for target in range(len(pixels)):
                plots.append(plot_contributions(output, captures, pose, seed, target))
    report = dict(protocol=PROTOCOL, layout=dict(width=WIDTH, rowsPerTarget=SAMPLES, vertexBase=VERTEX_BASE,
                  vertexStride=VERTEX_STRIDE, vertexCount=VERTICES, integers="raw little-endian uint32 bitcasts"),
                  capturedExecutableSha256=manifest.get("executableSha256"), capturedShaderSha256=manifest.get("shaderSha256"),
                  currentSourceHashes=current_source, sources=sources, failures=failures, warnings=warnings, captures=decoded)
    if asset:
        report["assetDirectory"] = dict(**asset.provenance, selectedPages=list(asset.records.values()))
    (output / "Decoded.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "Comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# MiniZorah recorded path attribution", "",
        "Decoded original production samples; no replay, no inferred occluder, no performance claim.", "",
        "Integer fields retain raw uint32 bits, including NaN-like IDs/seeds. Current source hashes are analysis-time versions; captured binary/shader identities remain separate.", "",
        "Requested owners are unavailable after phase24. Resolved owner and all three primitive slots are evidence; flags alone do not identify which budget originally rejected an owner.", "",
        "Same initial seed does not pair later primary rays: path branches advance the shared RNG. First divergence compares primary inputs before vertex events.", "",
        "Contribution plots show RGB means, not luminance; secondary vertices include every transport lobe, not isolated reflection. Residual includes aerial perspective, unsupported material-family internals and unclassified contributions. Samples with overwritten depth slots are entirely residual; their recorded divergence may follow an earlier overwritten event.", "",
        "## Validation", "", *(failures or ["All captured sample means reconstruct their HDR target values within tolerance."]), *warnings, "",
        "## First recorded divergence", "", "| Pose/seed | Pair | Pixel | Sample | First field |", "|---|---|---|---|---|"]
    for comparison in comparisons:
        for target in comparison["targets"]:
            first = next((s for s in target["samples"] if s["firstDivergence"]), None)
            field = first["firstDivergence"]["field"] if first else "identical recorded events"
            lines.append(f"| {comparison['pose']}/{comparison['seed']} | {comparison['left']}/{comparison['right']} | {target['pixel']} | {first['sample'] if first else '—'} | {field} |")
    lines += ["", "## Primary-hit and published-owner records", "",
              "`slot` shows owner ID / flags / cachedValid / selected cluster count. It is the captured final state, not a reconstructed admission reason. `fallback` means the instance's resolved slot is UINT_MAX.", ""]
    for group in decoded:
        for target_index, pixel in enumerate(pixels):
            lines += [f"### Pose {group['pose']}, seed {group['seed']}, pixel {pixel}", "",
                      "| Variant/sample | Jitter | Primary hit: stream / primitive / page / LOD | Published owner | Primitive slots | Final RGB mean |",
                      "|---|---|---|---|---|---|"]
            for name, variant in group["variants"].items():
                for sample in variant["targets"][target_index]["samples"]:
                    primary = sample["vertices"][0]
                    hit_label, owner_label, slots = "miss", "—", "—"
                    if primary["found"]:
                        hit = primary["hit"]
                        hit_label = f"{hit['rawStreamInstance']} / {hit['primitive']} / {hit['page']} / {hit.get('assetPage', {}).get('lodLevel', 'unrecorded')}"
                        evidence = primary["ownerEvidence"]
                        owner_label = "fallback" if evidence["terminalFallback"] else str(evidence["resolvedOwner"])
                        slots = "; ".join(f"{slot['owner']}/{slot['flags']}/{slot['cachedValid']}/{slot['selectedClusterCount']}"
                                          for slot in evidence["primitiveOwners"])
                    if not sample["completeVertexHistory"]:
                        hit_label += " (last depth-0 record; earlier events overwritten)"
                    jitter = ",".join(f"{value:.6g}" for value in sample["jitter"])
                    lines.append(f"| {name}/{sample['sample']} | {jitter} | {hit_label} | {owner_label} | {slots} | {np.mean(sample['finalRadiance']):.7g} |")
            lines.append("")
    lines += ["", "Raw fields, per-sample/per-vertex comparisons and owner evidence: [Decoded.json](Decoded.json), [Comparisons.json](Comparisons.json).", ""]
    for plot in plots:
        lines += [f"![Recorded radiance contributions]({plot})", ""]
    (output / "Report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(output=str(output), captures=len(decoded), comparisons=len(comparisons), failures=failures)))


if __name__ == "__main__":
    main()
