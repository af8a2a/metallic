#!/usr/bin/env python3
"""Compare the DCC cliff projection with the planar eight-layer material."""

import argparse
import copy
import math
from pathlib import Path
import shutil
from statistics import median
import sys

import numpy as np
from PIL import Image

from AncientRiftValleyWorldAcceptance import SCENE, fixture
from PrepareTerrainStreamingAcceptance import GRAPH, ROOT, read_json, write_json
from RunTerrainStreamingRegression import capture
from TerrainLayerAcceptance import save_aov
from TerrainLayerReference import repeat_linear
from VerifyTerrainStreamingRegression import REPORT_NAME, capture_geometry, raw_pixels, require
from VirtualShadowMapAcceptance import digest, run as run_vsm, verify_package


SOURCE = ROOT / "Asset/Terrain/AncientRiftValley"
VARIANTS = ("planar", "triplanar")
VIEWS = ("final", "baseColor", "normalTexture", "roughness")
PRODUCTION_VIEWS = ("final", "baseColor")
CLIFF_LAYERS = {"red_sandstone", "pale_caprock"}
RESOLUTION = (960, 540)
WARMUP = 72
FREEZE_FRAME = WARMUP - 4
FRAMES = WARMUP + 32
DIAGNOSTIC_FRAMES = WARMUP + 8
TIMEOUT = 600
SHADOW_WORK_LIMIT = 1024 * 1024
GEOMETRY_BUDGET = 256 * 1024 * 1024
SAMPLE_ID = "gpu-driven-ancient-rift-valley-4x4-vsm"
SAMPLE_REGIONS = list(range(16))


def asset_path(directory, uri):
    require(uri.startswith("asset://"), f"Expected a local material resource URI: {uri}")
    path = (directory / uri.removeprefix("asset://")).resolve()
    require(path.is_relative_to(directory) and path.is_file(), f"Missing local material resource: {uri}")
    return path


def camera():
    view = read_json(SOURCE / "Preview/RenderSettings.json")["cameras"]["CanyonVista"]
    # Blender's default horizontal sensor fit uses a 36 mm sensor width.
    focal = view["lensMillimeters"]
    fov_y = 2 * math.degrees(math.atan(18 / focal * RESOLUTION[1] / RESOLUTION[0]))
    return dict(projection="perspective", eye=view["eye"], center=view["target"], up=[0.0, 1.0, 0.0],
                fovDegrees=fov_y, znear=0.5, zfar=3500.0)


def material_variants(directory):
    original_path = SCENE.with_suffix(".metallic_scene.json")
    document = read_json(original_path)
    binding = document["materials"][0]["materialAsset"]
    require(all(entry["materialAsset"] == binding for entry in document["materials"]),
            "Expected one shared terrain material across the 16 regions")
    original_root = (original_path.parent / binding["root"]).resolve()
    original_material = asset_path(original_root, binding["uri"])
    triplanar = read_json(original_material)
    rules = triplanar["terrainLayers"]
    require(len(rules["layers"]) == 8 and len(rules["weightMaps"]) == 2
            and {layer["name"] for layer in rules["layers"] if "triplanar" in layer} == CLIFF_LAYERS,
            "Bake the eight-layer 16-region fixture with both triplanar cliffs first")
    planar = copy.deepcopy(triplanar)
    for layer in planar["terrainLayers"]["layers"]:
        layer.pop("triplanar", None)

    # Only the planar control needs a temporary material library. The triplanar
    # arm uses the registered sample's scene and canonical material assets.
    material_root = directory / "material"
    resources = {triplanar["definition"], *(value["uri"] for value in rules["weightMaps"])}
    resources.update(layer[role]["uri"] for layer in rules["layers"] for role in ("baseColor", "normal", "roughness"))
    source_resources = []
    for uri in sorted(resources):
        source = asset_path(original_root, uri)
        source_resources.append(source)
        output = material_root / source.relative_to(original_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)

    document["source"] = SCENE.as_posix()
    document["terrain"]["heightField"] = (original_path.parent / document["terrain"]["heightField"]).resolve().as_posix()
    environment = document["world"]["environment"]
    environment["path"] = (original_path.parent / environment["path"]).resolve().as_posix()
    material_path = material_root / "planar.material"
    write_json(material_path, planar)
    for entry in document["materials"]:
        entry["materialAsset"] = {"root": material_root.as_posix(), "uri": f"asset://{material_path.name}"}
    path = directory / "planar.metallic_scene.json"
    write_json(path, document)
    common_paths = [GRAPH, SCENE, original_path, original_material, Path(f"{SCENE}.meshstream.bin"),
                    Path(document["terrain"]["heightField"]), Path(environment["path"]), path, *source_resources,
                    *(resource for resource in material_root.rglob("*") if resource.is_file())]
    write_json(directory / "Inputs.json", dict(files={path.as_posix(): digest(path) for path in common_paths},
               sourceMaterial=original_material.as_posix(), sourceMaterialSha256=digest(original_material),
               textureBytesScope="VBuffer tracked resident scene texture allocations in the terrain-only fixture"))
    return {"planar": path.as_posix(), "triplanar": None}


def prepare(directory):
    _, regions, _ = fixture()
    require(len(regions) == len(SAMPLE_REGIONS), "Expected the registered sixteen-region terrain fixture")
    documents = material_variants(directory)
    for variant, scene in documents.items():
        spec = dict(id=variant, graph=GRAPH.as_posix(), builtInSample=SAMPLE_ID, scene=scene,
                    width=RESOLUTION[0], height=RESOLUTION[1], frames=FRAMES, warmupFrames=WARMUP,
                    camera=camera(), temporalJitter=False, metadataOnly=True, requireShadow=True,
                    allowResidentBudgetDeferral=True, requireResidentBudgetDeferral=False,
                    freezeStreamingFrame=FREEZE_FRAME)
        for mode in ("diagnostic", "timing"):
            diagnostic = mode == "diagnostic"
            for view in VIEWS if diagnostic else ("final",):
                case_dir = directory / mode / variant / view
                case = {**spec, "id": f"{variant}-{view}"}
                if diagnostic:
                    case.update(frames=DIAGNOSTIC_FRAMES, deferredDebugView=view,
                                captureFrames=[DIAGNOSTIC_FRAMES - 1],
                                output="FinalBlit.color" if view == "final" else "Deferred.color",
                                comparisonOutput="Deferred.color")
                    if view == "final" and variant == "triplanar":
                        case["comparisonFrames"] = [DIAGNOSTIC_FRAMES - 1]
                        case["shadowWorkDiagnostics"] = True
                write_json(case_dir / "Cases.json", {"version": 1, "mode": mode, "cases": [case]})


def measurements(report, expected):
    case, = report["cases"]
    require(report["protocol"] == "virtual-shadow-map-acceptance-v2" and report["status"] == case["status"] == "passed"
            and case["spec"] == expected and not case["temporalJitter"], "Incomplete or mismatched acceptance receipt")
    require([row["frame"] for row in case["frames"]] == list(range(expected["frames"])), "Expected all rendered frames")
    production = expected.get("builtInSample") == SAMPLE_ID
    if production:
        require_production_graph(case)
    rows = []
    for frame in case["frames"][expected["warmupFrames"]:]:
        stream, = (entry for entry in frame["streaming"] if entry["passName"] == "VBuffer")
        ready = frame["readiness"]
        selection = stream["regionSelection"]
        deferrals = stream["allocationBudgetDeferrals"] if production else 0
        require((frame["actualRenderWidth"], frame["actualRenderHeight"]) == RESOLUTION and not frame["warmup"]
                and ready["ready"] and ready["completedPages"] == ready["requiredPages"] > 0
                and Path(stream["assetPath"]).resolve() == Path(f"{SCENE}.meshstream.bin").resolve()
                and stream["loadFailures"] == stream["requestOverflows"] == 0
                and stream["allocationFailures"] == deferrals
                and stream["activePages"] <= stream["maxResidentPages"]
                and stream["geometryUsedBytes"] <= stream["geometryBudgetBytes"],
                f"Frame {frame['frame']}: mismatched extent or incomplete geometry streaming")
        if production:
            require(ready["requiredPages"] == 256 and selection["publishedRegions"] == SAMPLE_REGIONS,
                    f"Frame {frame['frame']}: incomplete production region cut")
        deferred, = (node for node in frame["nodes"] if node["name"] == "Deferred")
        gpu_ms = deferred["gpuMs"]
        require(gpu_ms is not None and math.isfinite(gpu_ms) and gpu_ms > 0, "Expected a Deferred Vulkan timestamp")
        rows.append(dict(frame=frame["frame"], deferredGpuMs=gpu_ms, textureResidentBytes=stream["textureResidentBytes"],
                         textureIdentity=stream["textureSemanticIdentitySha256"], geometryBytes=stream["geometryUsedBytes"],
                         geometryPages=stream["residentPages"], geometryActivePages=stream["activePages"],
                         geometryPendingPages=stream["pendingPages"], geometryTopology=stream["originalTopologyHash"],
                         primitiveMetadata=stream["primitiveMetadataHash"], regions=selection["publishedRegions"],
                         terminalPages=ready["requiredPages"], geometryBudgetDeferrals=deferrals,
                         geometryPublicationRevision=stream["geometryPublicationRevision"]))
    if production:
        require(len({row["geometryPublicationRevision"] for row in rows}) == 1,
                "Production resident geometry kept publishing after the stationary snapshot froze")
    return case, rows


def require_production_graph(case):
    nodes = {node["name"]: node["properties"] for node in case["resolvedGraph"]["nodes"]}
    vbuffer, shadows = nodes["VBuffer"], nodes["Shadows"]
    require(case["builtInSample"] == case["spec"]["builtInSample"] == SAMPLE_ID
            and Path(case["spec"]["graph"]).resolve() == GRAPH
            and vbuffer["autoLod"] and vbuffer["maxResidentBytes"] == GEOMETRY_BUDGET
            and vbuffer["maxResidentPages"] == 4096 and vbuffer["initialStreamRegions"] == SAMPLE_REGIONS
            and len(vbuffer["streamRegions"]) == len(SAMPLE_REGIONS)
            and shadows["shadowLOD"] and case["shadowLODSettings"]["enabled"] and shadows["packedRasterWork"]
            and shadows["clipmapCount"] == 9 and shadows["physicalPageBudget"] == 1024,
            "CanyonVista must use the registered 16-region budget and automatic shadow LOD")
    require(case["spec"]["camera"] == camera() and not case["spec"].get("events")
            and case["spec"]["freezeStreamingFrame"] == FREEZE_FRAME,
            "CanyonVista camera or stationary workload changed")
    return shadows


def shadow_map(directory, shot):
    resource = "Shadows.shadows"
    layers = shot["rawImages"][resource]["layerCount"]
    return raw_pixels(directory, shot, resource, RESOLUTION, "u1", "R8Unorm", layers)[..., 0]


def production_shadow_summary(case, directory, shot):
    shadows = require_production_graph(case)
    work = [row["shadow"] for row in case["frames"]]
    require(all(entry["totals"][5] == entry["totals"][9] == entry["header"]["budgetDroppedCount"] == 0
                for entry in work), "Production automatic-LOD VSM overflow or invalid coarse fallback")
    first, settled = work[0], work[shot["frame"]]
    lod = settled["lodStatistics"]
    finest_base_level = next((index for index, count in enumerate(first["lodStatistics"][16:]) if count), None)
    require(finest_base_level is not None and lod[1] == 1 and lod[2] == shadows["clipmapCount"] + 1
            and lod[4] > 0 and lod[5] > first["lodStatistics"][5] and sum(lod[16:16 + finest_base_level]) > 0
            and settled["totals"][24] > 0 and settled["totals"][6] > settled["totals"][7],
            "The settled CanyonVista view did not select refined casters and project valid fine shadows")
    _, covered = capture_geometry(directory, shot, camera(), RESOLUTION)
    visibility = shadow_map(directory, shot)[0][covered]
    lit = int(np.count_nonzero(visibility == 255))
    occluded = int(np.count_nonzero(visibility == 0))
    require(lit > 0 and occluded > 0, "The directional CanyonVista capture must include lit and occluded terrain")
    comparisons = case["comparisons"]
    if case["spec"].get("comparisonFrames"):
        require([entry["frame"] for entry in comparisons] == case["spec"]["comparisonFrames"]
                and all(entry["geometryIdentityStable"] and entry["byteEqual"] and entry["changedShadowBytes"] == 0
                        for entry in comparisons), "CanyonVista cache and forced redraw shadow pixels differ")
        require(all(entry[key][5] == entry[key][9] == 0 for entry in comparisons
                    for key in ("cachedShadowTotals", "redrawnShadowTotals")),
                "CanyonVista cache or forced shadow redraw overflowed")
    capacity = shadows.get("maxWorkItems", SHADOW_WORK_LIMIT)
    require(case["gpuWorkDiagnostics"] and all(entry["gpuWork"]["enabled"] for entry in work),
            "The canonical diagnostic omitted packed VSM work counters")
    packed = max(face["counters"]["rasterFineGroups"] for entry in work for face in entry["gpuWork"]["faces"])
    require(0 < packed <= capacity, f"Packed fine work peak {packed:,} is outside capacity {capacity:,}")
    require(max(entry["totals"][3] for entry in work) > capacity,
            "This run never exceeded the old one-logical-pair-per-slot work limit")
    return dict(automaticShadowLod=True, shadowRasterWorkLimit=capacity, peakPackedFineWork=packed,
                peakWorkCandidates=max(entry["totals"][3] for entry in work),
                peakWorkAdmitted=max(entry["totals"][4] for entry in work),
                peakWorkOverflow=0, shadowFrames=len(work), validFinePages=settled["totals"][24],
                fineReceivers=settled["totals"][6] - settled["totals"][7], selectedClusters=lod[4],
                startupTriangleSlots=first["lodStatistics"][5], selectedTriangleSlots=lod[5],
                selectedClustersByLevel=lod[16:], litTerrainPixels=lit, occludedTerrainPixels=occluded,
                cacheComparisons=len(comparisons), cacheChangedShadowBytes=sum(entry["changedShadowBytes"] for entry in comparisons),
                forcedRedrawCandidates=[entry["redrawnShadowTotals"][3] for entry in comparisons],
                forcedRedrawAdmitted=[entry["redrawnShadowTotals"][4] for entry in comparisons],
                freezeStreamingFrame=case["spec"]["freezeStreamingFrame"],
                shadowImage=(directory / shot["rawImages"]["Shadows.shadows"]["previews"][0]["file"]).as_posix())


def read_diagnostic(directory, variant, view):
    case_dir = directory / "diagnostic" / variant / view
    output = case_dir / "capture"
    report = read_json(output / REPORT_NAME)
    expected, = read_json(case_dir / "Cases.json")["cases"]
    validation = report["validationMessages"]
    require(report["mode"] == "diagnostic" and not report["performanceEligible"] and report["validation"]
            and not validation["captureFailed"] and validation["count"] == 0, f"{variant}/{view}: invalid diagnostic")
    case, rows = measurements(report, expected)
    shot, = case["captures"]
    require([shot["frame"]] == expected["captureFrames"], f"{variant}/{view}: missing matched capture")
    return output, shot, rows[-1]


def verify_production_shadow(directory):
    output, shot, row = read_diagnostic(directory, "triplanar", "final")
    case, = read_json(output / REPORT_NAME)["cases"]
    source = str(case["resolvedScene"])
    require(case["spec"]["scene"] is None and Path(source).resolve() in (SCENE, SCENE.with_suffix(".metallic_scene.json")),
            "The triplanar shadow acceptance must use the canonical scene")
    summary = production_shadow_summary(case, output, shot)
    require(summary["cacheComparisons"] > 0, "The canonical sample omitted the forced shadow redraw")
    summary.update(scene=SCENE.as_posix(), sample=SAMPLE_ID, camera=camera(), resolution=RESOLUTION,
                   captureFrame=shot["frame"], capture=(output / shot["file"]).as_posix(),
                   textureResidentBytes=row["textureResidentBytes"], geometryResidentBytes=row["geometryBytes"])
    write_json(directory / "ProductionShadow.json", summary)
    return summary


def cliff_pixels(directory, world, covered):
    # The same exterior stencil on the shared depth image identifies steep surfaces.
    dy, dx = np.gradient(world, axis=(0, 1))
    normal = np.cross(dx, dy)
    lengths = np.linalg.norm(normal, axis=-1)
    slope_ny = np.divide(np.abs(normal[..., 1]), lengths, out=np.ones_like(lengths), where=lengths > 1e-8)
    max_step = np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1))
    # Match source texel centres for the second linear RGBA weight map.
    material = Path(read_json(directory / "Inputs.json")["sourceMaterial"])
    material_root = material.parent.resolve()
    uri = read_json(material)["terrainLayers"]["weightMaps"][1]["uri"]
    with Image.open(asset_path(material_root, uri)) as image:
        map_pixels = np.asarray(image.convert("RGBA"), dtype=np.float32)[..., :2] / 255
    height, width = map_pixels.shape[:2]
    safe = covered & (max_step < 3.0)
    weights = np.zeros(covered.shape, dtype=np.float64)
    uv = np.clip(world[safe][:, [0, 2]] / 1000 + 0.5, 0, 1)
    uv = (uv * [width - 1, height - 1] + 0.5) / [width, height]
    weights[safe] = repeat_linear(map_pixels, uv, clamp=True).sum(axis=-1)
    return safe & (slope_ny < 0.6) & (weights > 0.6), safe & (weights < 0.001)


def pair_image(directory, name, before, after):
    width, height = RESOLUTION
    paired = Image.new("RGB", (width * 2, height))
    with Image.open(before) as image:
        paired.paste(image.convert("RGB"), (0, 0))
    with Image.open(after) as image:
        paired.paste(image.convert("RGB"), (width, 0))
    path = directory / f"{name}-planar-left-triplanar-right.png"
    paired.save(path)
    return path.as_posix()


def matched_material_pixels(a, b):
    world, covered = capture_geometry(a[0], a[1], camera(), RESOLUTION)
    depth = [raw_pixels(path, shot, "VBuffer.depth", RESOLUTION, "<f4", "D32Sfloat", 1)[0, ..., 0]
             for path, shot, _ in (a, b)]
    visibility = [raw_pixels(path, shot, "VBuffer.visibility", RESOLUTION, "<u4", "R32Uint", 1)[0, ..., 0]
                  for path, shot, _ in (a, b)]
    matched = covered & (depth[0] == depth[1]) & (visibility[0] == visibility[1])
    comparable = int(np.count_nonzero(matched))
    terrain = int(np.count_nonzero(covered | ((depth[1] > 0) & (visibility[1] != 0))))
    require(terrain > 0 and comparable >= 0.98 * terrain,
            f"Production LOD cuts shared only {comparable:,}/{terrain:,} identical terrain pixels")
    changed_shadows = int(np.count_nonzero(np.any(shadow_map(a[0], a[1]) != shadow_map(b[0], b[1]), axis=0) & matched))
    return world, matched, dict(terrainPixels=terrain, matchedDepthAndVisibilityPixels=comparable,
                                matchedGeometryShadowDifferences=changed_shadows)


def verify_visual(directory):
    candidate, = read_json(directory / "diagnostic/triplanar/final/Cases.json")["cases"]
    production = candidate.get("builtInSample") == SAMPLE_ID
    shadow_verification = verify_production_shadow(directory) if production else None
    all_views = {}
    geometry = None
    cliff = control = None
    geometry_residency = None
    for view in PRODUCTION_VIEWS if production else VIEWS:
        a, b = [read_diagnostic(directory, variant, view) for variant in VARIANTS]
        identities = [{name: shot["rawImages"][name]["rawSha256"]
                       for name in ("VBuffer.depth", "VBuffer.visibility", "Shadows.shadows")} for _, shot, _ in (a, b)]
        if production:
            world, covered, comparison = matched_material_pixels(a, b)
            cliff, control = cliff_pixels(directory, world, covered)
            require(cliff.sum() >= 500 and control.sum() >= 500, "CanyonVista missed matched cliff or control pixels")
        else:
            require(identities[0] == identities[1], f"{view}: projection changed depth, visibility or VSM")
            comparison = {}
        if geometry is None:
            geometry = identities[0]
            if not production:
                world, covered = capture_geometry(a[0], a[1], camera(), RESOLUTION)
                cliff, control = cliff_pixels(directory, world, covered)
                require(cliff.sum() >= 500 and control.sum() >= 500, "CanyonVista missed a steep cliff or its planar control")
            mask = np.zeros((*cliff.shape, 3), dtype=np.uint8)
            mask[cliff] = (255, 140, 0)
            mask[control] = (40, 160, 255)
            Image.fromarray(mask).save(directory / "CliffAndControlPixels.png")
        elif not production:
            require(identities[0] == geometry, f"{view}: AOV does not use the same depth, visibility and VSM")
        require(a[2]["textureIdentity"] and a[2]["textureIdentity"] == b[2]["textureIdentity"],
                f"{view}: the two projections do not have identical texture and published mip bindings")
        cut_keys = ("regions", "terminalPages", "geometryPages", "geometryActivePages", "geometryBytes")
        cuts = [tuple(entry[key] for key in cut_keys) for entry in (a[2], b[2])]
        require(cuts[0] == cuts[1] and (geometry_residency is None or geometry_residency == cuts[0]),
                f"{view}: measured geometry residency differs across views")
        geometry_residency = cuts[0]
        if view == "final":
            compare = pair_image(directory, view, a[0] / a[1]["file"], b[0] / b[1]["file"])
            all_views[view] = dict(image=compare, **comparison)
            continue
        values = [raw_pixels(path, shot, "Deferred.color", RESOLUTION, "<f4", "RGBA32Sfloat", 1, 4)[0, ..., :3]
                  for path, shot, _ in (a, b)]
        require(all(np.isfinite(rgb).all() for rgb in values), f"{view}: nonfinite shader output")
        difference = np.abs(values[0] - values[1])
        cliff_delta = float(difference[cliff].mean())
        control_delta = float(difference[control].mean())
        require(cliff_delta > 0.001 and control_delta < cliff_delta / 5,
                f"{view}: no distinct cliff signal, cliff {cliff_delta:.6f}, control {control_delta:.6f}")
        if view == "normalTexture":
            require(all(np.max(np.abs(np.linalg.norm(rgb[cliff] * 2 - 1, axis=-1) - 1)) < 0.002 for rgb in values),
                    "Mapped texture normals are not normalized")
        if view == "roughness":
            require(all(np.ptp(rgb[cliff], axis=-1).max() < 1e-6 for rgb in values), "Roughness is not scalar")
        pngs = [path / save_aov(path, f"{variant}-{view}", rgb, view)
                for variant, (path, _, _), rgb in zip(VARIANTS, (a, b), values, strict=True)]
        all_views[view] = dict(image=pair_image(directory, view, *pngs), cliffMeanAbsoluteDelta=cliff_delta,
                               controlMeanAbsoluteDelta=control_delta, **comparison)
    result = dict(passed=True, camera=camera(), resolution=RESOLUTION, captureFrame=DIAGNOSTIC_FRAMES - 1,
                  cliffPixels=int(cliff.sum()), planarControlPixels=int(control.sum()),
                  geometryResidency=dict(zip(cut_keys, geometry_residency, strict=True)),
                  textureSemanticIdentitySha256=a[2]["textureIdentity"], views=all_views)
    if production:
        result.update(builtInSample=SAMPLE_ID, productionShadows=shadow_verification)
    else:
        result["sharedGeometryAndShadow"] = geometry
    write_json(directory / "Visual.json", result)


def measure_verdict(runs, attempted=None):
    attempted = len(runs) if attempted is None else attempted
    a, b = [[run["deferredMedianMs"] for run in runs if run["variant"] == variant] for variant in VARIANTS]
    if min(len(a), len(b)) < 2:
        verdict = "inconclusive" if attempted == 8 else "extend"
        return dict(verdict=verdict, stopReason="fewer-than-two-matched-runs-per-variant")
    spread = max(max(a) - min(a), max(b) - min(b))
    separation = max(min(b) - max(a), min(a) - max(b))
    delta = median(b) - median(a)
    relative = delta / median(a)
    if abs(relative) < 0.01:
        verdict, reason = "no-measurable-difference", "less-than-one-percent"
    elif separation > spread:
        verdict, reason = ("triplanar-slower" if delta > 0 else "triplanar-faster"), "nonoverlapping-beyond-within-variant-spread"
    elif len(runs) == 8:
        verdict, reason = "no-measurable-difference", "eight-runs-not-separated-beyond-within-variant-spread"
    elif attempted == 8:
        verdict, reason = "inconclusive", "attempted-limit-with-overlapping-matched-runs"
    else:
        verdict, reason = "extend", "not-separated-beyond-within-variant-spread"
    return dict(verdict=verdict, stopReason=reason, planarMedianMs=median(a), triplanarMedianMs=median(b),
                deltaMs=delta, relativeDelta=relative, withinVariantSpreadMs=spread, separationMs=separation)


def verified_inputs(directory):
    inputs = read_json(directory / "Inputs.json")
    for filename, expected in inputs["files"].items():
        require(digest(filename) == expected, f"Comparison input changed: {filename}")
    return inputs


def timing_package(directory, cases, executable):
    if not directory.exists():
        run_vsm(argparse.Namespace(cases=cases, exe=executable, output=directory, runs=1, timeout=TIMEOUT))
    manifest, captures = verify_package(directory)
    identity = read_json(directory / "Identity.json")
    require(manifest == read_json(cases) and str(executable) in identity["binaries"],
            f"{directory.name}: archived timing package has different inputs")
    fingerprints = {**identity["inputs"], **identity["binaries"],
                    **{str(ROOT / path): sha for path, sha in identity["sources"].items()}}
    require(all(digest(path) == sha for path, sha in fingerprints.items()),
            f"{directory.name}: renderer, source or input changed since the archived timing process")
    return manifest, captures


def run_measurements(directory, executable):
    visual = read_json(directory / "Visual.json")
    require(visual["passed"], "Verify the comparison images before ordinary timing")
    inputs = verified_inputs(directory)
    production_shadows = visual.get("productionShadows")
    cut = visual["geometryResidency"]
    cut_keys = ("regions", "terminalPages", "geometryActivePages", "geometryBytes") if production_shadows else tuple(cut)
    runs = []
    excluded = []
    identity = workload = None
    for index, variant in enumerate(VARIANTS * 4, 1):
        case_dir = directory / "timing" / variant / "final"
        output = directory / "runs" / f"{index:02d}-{variant}"
        operation = "Verify archived" if output.exists() else "Run"
        print(f"{operation} Deferred timing process {index}: {variant}", flush=True)
        manifest, captures = timing_package(output, case_dir / "Cases.json", executable)
        (receipt, report), = captures
        expected, = manifest["cases"]
        case, rows = measurements(report, expected)
        current = ({key: value for key, value in expected.items() if key not in ("id", "scene")}, case["resolvedGraph"])
        require(workload is None or workload == current, f"{variant}: comparison did not use the same production graph")
        workload = current
        final = rows[-1]
        require(all(row["textureResidentBytes"] == final["textureResidentBytes"] > 0 for row in rows),
                f"{variant}: measured window includes texture transitions or lacks resident texture bytes")
        current = tuple(final[key] for key in ("regions", "terminalPages", "geometryTopology", "primitiveMetadata"))
        require(identity is None or identity == current, f"{variant}: geometry topology or region cut differs")
        identity = current
        for row in rows:
            require(all(row[key] == final[key] for key in (*cut_keys, "geometryPages", "geometryPendingPages")),
                    f"{variant}: geometry allocation or its residency split changed during timing")
        sample = dict(process=index, variant=variant, deferredMedianMs=median(row["deferredGpuMs"] for row in rows),
                      textureResidentBytes=final["textureResidentBytes"], receipt=receipt.relative_to(directory).as_posix(),
                      receiptSha256=digest(receipt), geometryActivePages=final["geometryActivePages"],
                      geometryResidentPages=final["geometryPages"], geometryPendingPages=final["geometryPendingPages"],
                      geometryBytes=final["geometryBytes"], frames=rows)
        allocation_matches = all(final[key] == cut[key] for key in cut_keys)
        if production_shadows and not allocation_matches:
            sample["reason"] = "different geometry active allocation from the matched diagnostics"
            excluded.append(sample)
            print(f"Exclude process {index}: active geometry {final['geometryActivePages']:,} pages / "
                  f"{final['geometryBytes']:,} bytes", flush=True)
        else:
            require(allocation_matches, f"{variant}: measured window uses different geometry from the matched diagnostics")
            if production_shadows:
                require(abs(final["geometryPages"] - cut["geometryPages"]) <= 1,
                        f"{variant}: resident slot count differed by more than the observed one-page completion")
            runs.append(sample)
        if index in (4, 8):
            verified_inputs(directory)
            result = measure_verdict(runs, index)
            texture = {variant: sorted({run["textureResidentBytes"] for run in runs if run["variant"] == variant})
                       for variant in VARIANTS}
            write_json(directory / "Timing.json", dict(result=result, textureResidentBytes=texture, runs=runs,
                       attemptedRuns=index, excludedRuns=excluded,
                       resolution=RESOLUTION, camera=camera(), warmupFrames=WARMUP, measuredFrames=FRAMES - WARMUP,
                       shadowRasterWorkLimit=production_shadows["shadowRasterWorkLimit"] if production_shadows else SHADOW_WORK_LIMIT,
                       shadowCasterCut="automatic" if production_shadows else "terminal",
                       geometryBudgetBytes=GEOMETRY_BUDGET if production_shadows else 512 * 1024 * 1024,
                       stableGeometry=visual["geometryResidency"],
                       freezeStreamingFrame=expected.get("freezeStreamingFrame"),
                       scope="Independent process medians of the Vulkan Deferred graph node, no validation or readback",
                       textureScope=inputs["textureBytesScope"]))
            if result["verdict"] != "extend":
                return


def report(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timing_path = directory / "Timing.json"
    timing = read_json(timing_path)
    result = timing["result"]
    runs = timing["runs"]
    attempted = timing.get("attemptedRuns", len(runs))
    excluded = timing.get("excludedRuns", [])
    require(attempted == len(runs) + len(excluded) and result == measure_verdict(runs, attempted)
            and result["verdict"] != "extend", "The timing comparison is incomplete")
    require(all(digest(directory / run["receipt"]) == run["receiptSha256"] for run in (*runs, *excluded)), "Timing receipts changed")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    width, height = timing["resolution"]
    figure, (gpu, memory) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, constrained_layout=True)
    for variant, color in zip(VARIANTS, ("#447cae", "#ac5938"), strict=True):
        samples = [run for run in runs if run["variant"] == variant]
        if not samples:
            continue
        process = [run["process"] for run in samples]
        times = [run["deferredMedianMs"] for run in samples]
        bytes_mib = [run["textureResidentBytes"] / (1024 ** 2) for run in samples]
        gpu.scatter(process, times, color=color, label=variant, s=40)
        gpu.axhline(median(times), color=color, alpha=0.5, linestyle="--")
        memory.scatter(process, bytes_mib, color=color, s=40)
    figure.suptitle(f"Ancient Rift Valley / CanyonVista / {width} × {height}")
    gpu.set_ylabel("Deferred GPU time (ms)")
    if runs:
        gpu.legend(title="Material projection and median")
    memory.set_ylim(0, max((run["textureResidentBytes"] for run in runs), default=1024 ** 2) / (1024 ** 2) * 1.18)
    memory.set_ylabel("Resident textures (MiB)")
    memory.set_xlabel("Independent process launch")
    memory.set_xticks(range(1, attempted + 1))
    for axis in (gpu, memory):
        axis.grid(axis="y", alpha=0.2)
        for entry in excluded:
            axis.axvline(entry["process"], color="#888888", alpha=0.5, linestyle=":")
    figure.savefig(output / "CliffCost.png", dpi=150)
    plt.close(figure)

    reproducible = Path(__file__).resolve()
    command = f'python -B "{reproducible.as_posix()}" report --output "{directory.as_posix()}"'
    (output / "plot_report.py").write_text(
        "import runpy\nimport sys\nfrom pathlib import Path\n\n"
        f"runner = {str(reproducible)!r}\n"
        "sys.path.insert(0, str(Path(runner).parent))\n"
        "sys.argv = [runner, 'report', '--output', str(Path(__file__).resolve().parents[1])]\n"
        "runpy.run_path(runner, run_name='__main__')\n", encoding="utf-8")
    pages = timing["shadowCasterCut"] == "automatic"
    rows = ["| Run | Projection | Deferred GPU median (ms) | Resident textures (MiB) |" +
            (" Active / resident / pending geometry pages |" if pages else ""),
            "|---|---|---:|---:|" + ("---:|" if pages else "")]
    rows.extend(f"| {run['process']} | {run['variant']} | {run['deferredMedianMs']:.6f} | "
                f"{run['textureResidentBytes'] / (1024 ** 2):.4f} |" +
                (f" {run['geometryActivePages']:,} / {run['geometryResidentPages']:,} / {run['geometryPendingPages']:,} |"
                 if pages else "") for run in runs)
    excluded_text = [f"Process {entry['process']} ({entry['variant']}) was excluded for {entry['reason']}. "
                     f"Its {entry['geometryActivePages']:,} active geometry pages used {entry['geometryBytes']:,} bytes. "
                     f"Its [raw receipt](../{entry['receipt']}) is retained without using its timing."
                     for entry in excluded]
    timing_summary = (f"The observed median changed by {result['deltaMs']:+.6f} ms ({result['relativeDelta']:+.2%}). "
                      if "deltaMs" in result else "")
    caster_cut = {"terminal": "the complete terminal caster cut", "automatic": "the registered sample's automatic caster LOD"}
    freeze = timing.get("freezeStreamingFrame")
    snapshot = (f"Geometry publication and texture feedback freeze at frame {freeze} of the stationary view. "
                "The timed window retains this published snapshot with automatic VSM caster LOD enabled. "
                "At the cap, a one-page completion can move the resident count while the shared active-page allocation "
                "and per-process publication revision remain fixed. The table records the exact split. ") if freeze is not None else ""
    (output / "Report.md").write_text("\n".join([
        "# Ancient Rift Valley cliff projection cost", "",
        f"CanyonVista at {width} × {height}. Each fresh process retains disk caches, warms up for {timing['warmupFrames']} frames "
        f"and measures {timing['measuredFrames']} frames. The independent statistical unit is the process median. "
        "Dashed lines show medians across processes. GPU clocks were left at their defaults. Both variants use the same "
        f"{timing['shadowRasterWorkLimit']:,}-item VSM raster work budget with 1,024 physical pages "
        f"and {caster_cut[timing['shadowCasterCut']]}. Automatic main-view geometry LOD remains active. "
        f"The shared {timing['geometryBudgetBytes'] // (1024 ** 2)} MiB geometry budget reaches a stable "
        f"{timing['stableGeometry'].get('geometryActivePages', timing['stableGeometry']['geometryPages']):,}-page allocation before timing. {snapshot}", "",
        "![Deferred GPU and resident texture cost](CliffCost.png)", "", *rows, "",
        f"{timing_summary}Verdict `{result['verdict']}` after {attempted} launches ({len(runs)} matched), "
        f"with stop reason `{result['stopReason']}`.", "", *excluded_text, "",
        "The Deferred timestamp covers one Vulkan render graph node. Resident texture bytes are the tracked VBuffer scene "
        "texture allocations. Diagnostics and readbacks run in separate processes.", "",
        f"[Source data](../Timing.json), SHA-256 `{digest(timing_path)}`. Raw receipt paths and hashes are recorded per launch. "
        f"Convert `textureResidentBytes` to MiB using 1,048,576 bytes per MiB. Python {sys.version.split()[0]}, "
        f"Matplotlib {matplotlib.__version__}.", "", "```powershell", command, "```", ""]), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "diagnostic", "verify-shadow", "verify", "timing", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--view", choices=VIEWS, help="Capture just one diagnostic pair before the full verification")
    parser.add_argument("--variant", choices=VARIANTS, help="Capture just one diagnostic arm")
    args = parser.parse_args()
    directory = args.output.resolve()
    try:
        if args.action == "prepare":
            directory.mkdir(parents=True, exist_ok=False)
            prepare(directory)
        if args.action in ("diagnostic", "timing"):
            require(args.executable and args.executable.is_file(), "Build and provide MetallicRHITests")
        if args.action == "diagnostic":
            for view in (args.view,) if args.view else PRODUCTION_VIEWS:
                for variant in (args.variant,) if args.variant else VARIANTS:
                    capture(args.executable.resolve(), directory / "diagnostic" / variant / view, timeout=TIMEOUT)
        if args.action == "verify-shadow":
            verify_production_shadow(directory)
        if args.action == "verify" or (args.action == "diagnostic" and not (args.view or args.variant)):
            verify_visual(directory)
        if args.action == "timing":
            run_measurements(directory, args.executable.resolve())
        if args.action == "report":
            report(directory)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"Ancient Rift Valley cliff validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
