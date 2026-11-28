"""Verify world-space terrain material captures during a region handoff."""

from pathlib import Path

import numpy as np

from TerrainLayerAcceptance import REFERENCE_SEAM_TOLERANCE, REFERENCE_TOLERANCE, save_aov
from TerrainLayerReference import RGBA, TEXTURE_ROLES, load_terrain_layer_reference, ray_cone_uv_footprint
from TerrainRegionStreamingAcceptance import SCENE, STREAM
from VerifyTerrainStreamingRegression import REPORT_NAME, camera_timeline, capture_geometry, raw_pixels, read, require


VARIANTS = {"two": "RG", "four": RGBA}
VIEWS = {"near", "far", "handoffBefore", "handoff", "handoffAfter"}
MIN_PIXELS_PER_SIDE = 20
MIN_EDGE_PAIRS = 10


def _load_case(directory, mode, scene_document, capture_frames):
    receipt = read(directory / REPORT_NAME)
    validation = receipt["validationMessages"]
    require(receipt["protocol"] == "virtual-shadow-map-acceptance-v2" and receipt["mode"] == "diagnostic"
            and receipt["status"] == "passed" and receipt["validation"] and not validation["captureFailed"]
            and validation["count"] == 0, f"{directory}: expected a passing Vulkan diagnostic")
    case, = receipt["cases"]
    spec = case["spec"]
    nodes = {node["name"]: node["properties"] for node in case["resolvedGraph"]["nodes"]}
    vbuffer = nodes["VBuffer"]
    require(case["status"] == "passed" and Path(spec["scene"]).resolve() == scene_document.resolve()
            and not spec["temporalJitter"]
            and nodes["Deferred"]["debugView"] == mode and spec["comparisonOutput"] == "Deferred.color"
            and Path(vbuffer["streamAssetPath"]).resolve() == STREAM.resolve()
            and not vbuffer["autoLod"] and vbuffer["lodLevel"] == 31,
            f"{directory}: unexpected scene, AOV or geometry settings")
    require([shot["frame"] for shot in case["captures"]] == sorted(capture_frames),
            f"{directory}: missing or unexpected material capture")
    cameras, cuts = camera_timeline(spec)
    require(not cuts and [row["frame"] for row in case["frames"]] == list(range(spec["frames"]))
            and all(not case["frames"][frame]["warmup"] for frame in capture_frames),
            f"{directory}: incomplete continuous flight or capture during warmup")
    return case, cameras


def _scene_variants():
    documents = {name: SCENE.with_name(f"{SCENE.stem}_{name}.metallic_scene.json") for name in VARIANTS}
    common = []
    for document in documents.values():
        source = read(document)
        require((document.parent / source["source"]).resolve() == SCENE.resolve(),
                f"{document}: expected the canonical terrain region geometry")
        common.append({key: value for key, value in source.items() if key != "materials"})
    require(common[0] == common[1], "Terrain region scene documents must agree outside material bindings")
    references = {name: load_terrain_layer_reference(SCENE, scene_document=documents[name], channels=channels)
                  for name, channels in VARIANTS.items()}
    two, four = references["two"], references["four"]
    require(two.weight_texture.path == four.weight_texture.path and two.default_channel == four.default_channel,
            "The variants must share the RGBA weights and default layer")
    four_layers = {layer.channel: layer for layer in four.layers}
    for layer in two.layers:
        for role, texture in layer.textures.items():
            other = four_layers[layer.channel].textures[role]
            require(texture.path == other.path and texture.srgb == other.srgb
                    and np.array_equal(texture.transform, other.transform),
                    "The variants must share R/G layer textures and world transforms")
    return documents, references


def _crossing_samples(world, covered, split, four_way):
    west = world[..., 0] < split
    indices = np.arange(covered.size).reshape(covered.shape)
    first, second = [], []
    holes = 0
    for left, right, start, middle, end in (
        (np.s_[:, :-1], np.s_[:, 1:], np.s_[:, :-2], np.s_[:, 1:-1], np.s_[:, 2:]),
        (np.s_[:-1, :], np.s_[1:, :], np.s_[:-2, :], np.s_[1:-1, :], np.s_[2:, :]),
    ):
        pairs = four_way[left] & four_way[right] & (west[left] != west[right])
        pairs &= np.linalg.norm(world[left] - world[right], axis=-1) < 4.0
        first.append(indices[left][pairs])
        second.append(indices[right][pairs])
        across_gap = four_way[start] & four_way[end] & (west[start] != west[end])
        across_gap &= np.linalg.norm(world[start] - world[end], axis=-1) < 4.0
        holes += np.count_nonzero(across_gap & ~covered[middle])
    return np.concatenate(first), np.concatenate(second), int(holes)


def _geometry(directory, shot, camera, size, split, material, label, *, half_band=2.0, min_pairs=MIN_EDGE_PAIRS):
    world, covered = capture_geometry(directory, shot, camera, size)
    west = world[..., 0] < split
    sites = covered & (np.abs(world[..., 0] - split) < half_band)
    footprint = ray_cone_uv_footprint(world, sites, camera, material.world_size)
    require(np.isfinite(footprint).all() and np.all(footprint > 0), f"{label}: invalid texture footprint")
    weights = material.weights(world[sites], footprint)
    four_way = np.zeros(sites.shape, dtype=bool)
    four_way[sites] = np.all(weights > 0.02, axis=-1)
    counts = [int(np.count_nonzero(four_way & side)) for side in (west, ~west)]
    require(min(counts) >= MIN_PIXELS_PER_SIDE, f"{label}: too few four-layer pixels on each side: {counts}")
    first, second, holes = _crossing_samples(world, covered, split, four_way)
    require(len(first) >= min_pairs and holes == 0,
            f"{label}: expected at least {min_pairs} adjacent four-layer pairs and no holes; got {len(first)} / {holes}")
    fallback = np.all(weights[:, :2] == 0, axis=-1)
    return dict(world=world[sites], covered=covered, west=west[sites], sites=sites, footprint=footprint,
                fourWay=four_way[sites], pixelsByRegion=counts, first=first, second=second,
                fallback=fallback, depthHash=shot["rawImages"]["VBuffer.depth"]["rawSha256"])


def _verify_aov(directory, shot, size, variant, mode, label, region, material, expected_other):
    rgb = raw_pixels(directory, shot, "Deferred.color", size, "<f4", "RGBA32Sfloat", 1, 4)[0, ..., :3]
    require(np.isfinite(rgb).all(), f"{label}: nonfinite {mode} readback")
    sites = region["sites"]
    actual = rgb[sites]
    result = {"aovImage": save_aov(directory, f"{variant}-{mode}-{label}", rgb, mode)}
    if mode == "roughness":
        require(np.ptp(actual, axis=-1).max() < 1e-6, f"{label}: roughness must be scalar")
        actual = actual[:, :1]
    if mode == "normalTexture":
        unit_error = float(np.max(np.abs(np.linalg.norm(actual * 2.0 - 1.0, axis=-1) - 1.0)))
        require(unit_error <= 0.002, f"{label}: blended tangent normals must have unit length")
        result["tangentNormalMaxUnitLengthError"] = unit_error

    footprint = region["footprint"]
    weights = material.weights(region["world"], footprint)
    expected = material.evaluate(mode, region["world"], weights, footprint=footprint)
    difference = actual - expected
    errors = [float(np.mean(np.abs(difference[side]))) for side in (region["west"], ~region["west"])]
    require(max(errors) <= REFERENCE_TOLERANCE, f"{label}: {mode} mean errors for west/east: {errors}")
    image_shape = (*sites.shape, expected.shape[-1])
    residual = np.zeros(image_shape, dtype=difference.dtype)
    residual[sites] = difference
    first, second = region["first"], region["second"]
    residual = residual.reshape(-1, expected.shape[-1])
    seam = float(np.percentile(np.linalg.norm(residual[first] - residual[second], axis=-1), 95))
    require(seam <= REFERENCE_SEAM_TOLERANCE, f"{label}: {mode} seam residual P95 {seam:.4f}")

    if variant == "two":
        fallback = region["fallback"]
        require(np.all(weights[fallback, material.default_channel] == 1), f"{label}: fallback weighting differs")
        if np.any(fallback):
            error = float(np.mean(np.abs(difference[fallback])))
            require(error <= REFERENCE_TOLERANCE, f"{label}: {mode} default layer mismatch ({error:.4f})")
            result["defaultLayerMeanAbsoluteError"] = error

    witness = region["fourWay"]
    error = float(np.mean(np.abs(difference[witness])))
    other = float(np.mean(np.abs(actual[witness] - expected_other[witness])))
    if mode == "baseColor":
        require(other > max(0.01, error * 2), f"{label}: {variant} base color indistinguishable from the alternate layer count")
    if label == "weightMipFar":
        world = region["world"]
        mip_zero = material.evaluate(mode, world, material.weights(world), footprint=footprint)
        mip_zero_error = float(np.mean(np.abs(actual[witness] - mip_zero[witness])))
        require(mip_zero_error > max(error * 2.0, error + 0.00002),
                f"{variant}: GPU base color did not distinguish filtered weights from weight mip 0")
        result["weightMipZeroMeanAbsoluteError"] = mip_zero_error
    if mode == "normalTexture" and label in ("near", "far"):
        filtered_error = float(np.mean(np.abs(difference)))
        controls = {"mipZero": None, "oneMipFiner": footprint * 0.5, "oneMipCoarser": footprint * 2.0}
        control_errors = {name: float(np.mean(np.abs(actual - material.evaluate(mode, region["world"], footprint=f))))
                          for name, f in controls.items()}
        require(min(control_errors.values()) > max(0.005, filtered_error * 2),
                f"{label}: tangent normals do not distinguish the reconstructed sampling mip: {control_errors}")
        result["mipControlMeanAbsoluteErrors"] = control_errors

    layer_mips = {}
    active_pixels = {}
    for layer in material.layers:
        active = weights[:, layer.channel] > 0.1
        active_count = int(np.count_nonzero(active))
        channel = RGBA[layer.channel]
        require(active_count >= MIN_PIXELS_PER_SIDE, f"{label}: missing active {channel} layer witnesses")
        active_pixels[channel] = active_count
        layer_mips[channel] = float(np.median(layer.textures[TEXTURE_ROLES[mode]].lod(footprint[active])))
    result.update(meanAbsoluteErrorByRegion=errors, seamReferenceResidualP95=seam, fourLayerOverlapMAE=error,
                  alternateLayerCountMAE=other, medianTextureMipByChannel=layer_mips,
                  activeLayerPixelsByChannel=active_pixels)
    return result


def verify(experiment_root: Path):
    """Return verifiable evidence for both layer counts and the five route captures."""
    route = read(experiment_root / "Route.json")
    captures = {int(frame): label for frame, label in route["captureFrames"].items()}
    split = read(SCENE)["extras"]["metallicTerrainRegions"]["splitX"]
    crossing, = route["crossings"]
    require(route["splitX"] == split and len(captures) == 5 and set(captures.values()) == VIEWS
            and [captures.get(crossing + offset) for offset in (-1, 0, 1)] ==
            ["handoffBefore", "handoff", "handoffAfter"], "Unexpected cross-region material capture route")
    documents, references = _scene_variants()
    geometry, camera_sequence, size = {}, None, None
    results = {"variants": {}, "views": {}, "splitX": split, "crossingFrame": crossing}
    for variant, material in references.items():
        results["variants"][variant] = {}
        alternate = references["two" if variant == "four" else "four"]
        for mode in TEXTURE_ROLES:
            directory = experiment_root / "diagnostic" / variant / mode / "capture"
            case, cameras = _load_case(directory, mode, documents[variant], captures)
            extent = (case["spec"]["width"], case["spec"]["height"])
            if camera_sequence is None:
                camera_sequence, size = cameras, extent
                was_east = cameras[crossing - 1]["eye"][0] >= split
                is_east = cameras[crossing]["eye"][0] >= split
                require(was_east != is_east, "The reported handoff frame does not cross the region boundary")
            require(cameras == camera_sequence and extent == size, f"{directory}: camera trajectory or resolution differs")
            views = {}
            for shot in case["captures"]:
                frame = shot["frame"]
                label = captures[frame]
                streaming, = (entry for entry in case["frames"][frame]["streaming"] if entry["passName"] == "VBuffer")
                require(streaming["regionSelection"]["publishedRegions"] == [0, 1],
                        f"{directory} {label}: both regions must be published")
                if frame not in geometry:
                    region = _geometry(directory, shot, cameras[frame], size, split, references["four"], label)
                    geometry[frame] = region
                    results["views"][label] = dict(frame=frame, depthSha256=region["depthHash"],
                        fourLayerPixelsByRegion=region["pixelsByRegion"], adjacentPairs=len(region["first"]),
                        twoLayerDefaultPixels=int(np.count_nonzero(region["fallback"])),
                        medianWorldUVFootprint=float(np.median(region["footprint"])),
                        medianWeightTextureMip=float(np.median(material.weight_texture.lod(region["footprint"]))))
                else:
                    region = geometry[frame]
                    _, covered = capture_geometry(directory, shot, cameras[frame], size)
                    require(shot["rawImages"]["VBuffer.depth"]["rawSha256"] == region["depthHash"]
                            and np.array_equal(covered, region["covered"]), f"{directory} {label}: geometry changed")
                expected_other = alternate.evaluate(mode, region["world"], footprint=region["footprint"])
                view = _verify_aov(directory, shot, size, variant, mode, label, region, material, expected_other)
                view["aovImage"] = (directory.relative_to(experiment_root) / view["aovImage"]).as_posix()
                views[label] = view
            near, far = (views[label]["medianTextureMipByChannel"] for label in ("near", "far"))
            require(all(far[channel] >= near[channel] + 0.5 for channel in near),
                    f"{directory}: far view did not exercise visibly coarser layer mips")
            results["variants"][variant][mode] = views
    require(sum(view["twoLayerDefaultPixels"] for view in results["views"].values()) >= MIN_PIXELS_PER_SIDE,
            "The route did not see the isolated default-layer patch")
    return results


def verify_weight_mip(experiment_root):
    documents, references = _scene_variants()
    split = read(SCENE)["extras"]["metallicTerrainRegions"]["splitX"]
    results = {}
    matched_geometry = None
    baseline_camera_sequence = None
    for variant, material in references.items():
        directory = experiment_root / "weight-mip" / variant / "capture"
        spec, = read(directory.parent / "Cases.json")["cases"]
        frame, = spec["captureFrames"]
        case, cameras = _load_case(directory, "baseColor", documents[variant], [frame])
        require(case["spec"] == spec and (baseline_camera_sequence is None or cameras == baseline_camera_sequence),
                "The weight mip captures must use the same camera route")
        baseline_camera_sequence = cameras
        shot, = case["captures"]
        stream, = (row for row in case["frames"][frame]["streaming"] if row["passName"] == "VBuffer")
        require(stream["regionSelection"]["publishedRegions"] == [0, 1], "Both regions must publish for weight mip proof")
        size = (spec["width"], spec["height"])
        region = _geometry(directory, shot, cameras[frame], size, split, references["four"], "weightMipFar",
                           half_band=5.0, min_pairs=6)
        require(matched_geometry is None or region["depthHash"] == matched_geometry,
                "Weight mip captures must agree on the visible geometry")
        matched_geometry = region["depthHash"]
        witnesses = region["fourWay"]
        footprint = region["footprint"]
        mip = material.weight_texture.lod(footprint)
        mip_by_region = [float(np.median(mip[witnesses & side])) for side in (region["west"], ~region["west"])]
        require(min(mip_by_region) >= 0.15,
                f"Shared weights must use a coarser mip on both sides: {mip_by_region}")
        alternate = references["two" if variant == "four" else "four"]
        expected_other = alternate.evaluate("baseColor", region["world"], footprint=footprint)
        result = _verify_aov(directory, shot, size, variant, "baseColor", "weightMipFar", region, material, expected_other)
        result["aovImage"] = (directory.relative_to(experiment_root) / result["aovImage"]).as_posix()
        result.update(frame=frame, medianWeightMipByRegion=mip_by_region, pixelsByRegion=region["pixelsByRegion"],
                      adjacentPairs=len(region["first"]), depthSha256=matched_geometry)
        results[variant] = result
    return results
