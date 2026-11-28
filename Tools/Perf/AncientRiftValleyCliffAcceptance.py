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
CLIFF_LAYERS = {"red_sandstone", "pale_caprock"}
RESOLUTION = (960, 540)
WARMUP = 72
FRAMES = WARMUP + 32
DIAGNOSTIC_FRAMES = WARMUP + 8
TIMEOUT = 360
SHADOW_WORK_LIMIT = 1024 * 1024
GEOMETRY_BUDGET = 512 * 1024 * 1024


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

    # Both variants load the same resource library. Keep the source fixture untouched.
    material_root = directory / "material"
    resources = {triplanar["definition"], *(value["uri"] for value in rules["weightMaps"])}
    resources.update(layer[role]["uri"] for layer in rules["layers"] for role in ("baseColor", "normal", "roughness"))
    for uri in sorted(resources):
        source = asset_path(original_root, uri)
        output = material_root / source.relative_to(original_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)

    document["source"] = SCENE.as_posix()
    document["terrain"]["heightField"] = (original_path.parent / document["terrain"]["heightField"]).resolve().as_posix()
    environment = document["world"]["environment"]
    environment["path"] = (original_path.parent / environment["path"]).resolve().as_posix()
    documents = {}
    for variant, material in zip(VARIANTS, (planar, triplanar), strict=True):
        material_path = material_root / f"{variant}.material"
        write_json(material_path, material)
        for entry in document["materials"]:
            entry["materialAsset"] = {"root": material_root.as_posix(), "uri": f"asset://{material_path.name}"}
        path = directory / f"{variant}.metallic_scene.json"
        write_json(path, document)
        documents[variant] = path
    common_paths = [SCENE, Path(f"{SCENE}.meshstream.bin"), Path(document["terrain"]["heightField"]),
                    Path(environment["path"]), *documents.values(), *(path for path in material_root.rglob("*") if path.is_file())]
    write_json(directory / "Inputs.json", dict(files={path.as_posix(): digest(path) for path in common_paths},
               sourceMaterial=original_material.as_posix(), sourceMaterialSha256=digest(original_material),
               textureBytesScope="VBuffer tracked resident scene texture allocations in the terrain-only fixture"))
    return documents


def prepare(directory):
    stream, regions, _ = fixture()
    documents = material_variants(directory)
    graph = read_json(GRAPH)
    nodes = {node["name"]: node["properties"] for node in graph["nodes"]}
    nodes["VBuffer"].update(path=SCENE.as_posix(), streamAssetPath=stream.as_posix(), streamRegions=regions,
                            initialStreamRegions=list(range(len(regions))), initialLoad=True,
                            geometryRegionSelectionMode="viewAndDirectionalCasters", geometryRegionLoadDistance=44.0,
                            geometryRegionUnloadDistance=56.0, maxResidentPages=4096,
                            maxResidentBytes=GEOMETRY_BUDGET, maxLockedFallbackPages=4096, maxActiveGroups=32768,
                            autoLod=True, lodPixelError=0.05, compactShadingAttributes=False)
    # Fix the complete VSM terminal cut. Automatic caster refinement overflows
    # the fine-work buffer in this view, even when the main material stays planar.
    nodes["Shadows"].update(clipmapCount=9, physicalPageBudget=1024, shadowBias=0.008,
                            maxWorkItems=SHADOW_WORK_LIMIT, shadowLOD=False)
    nodes["Deferred"]["path"] = SCENE.as_posix()

    for variant, document in documents.items():
        spec = dict(id=variant, graph="Valley.metallic_graph.json", scene=document.as_posix(),
                    width=RESOLUTION[0], height=RESOLUTION[1], frames=FRAMES, warmupFrames=WARMUP,
                    camera=camera(), temporalJitter=False, metadataOnly=True, requireShadow=True)
        for mode in ("diagnostic", "timing"):
            diagnostic = mode == "diagnostic"
            for view in VIEWS if diagnostic else ("final",):
                case_dir = directory / mode / variant / view
                nodes["Deferred"]["debugView"] = view
                write_json(case_dir / spec["graph"], graph)
                case = {**spec, "id": f"{variant}-{view}"}
                if diagnostic:
                    case.update(frames=DIAGNOSTIC_FRAMES,
                                captureFrames=[DIAGNOSTIC_FRAMES - 1], streamingEvidenceFrames=[DIAGNOSTIC_FRAMES - 1],
                                output="FinalBlit.color" if view == "final" else "Deferred.color",
                                comparisonOutput="Deferred.color")
                write_json(case_dir / "Cases.json", {"version": 1, "mode": mode, "cases": [case]})


def measurements(report, expected):
    case, = report["cases"]
    require(report["protocol"] == "virtual-shadow-map-acceptance-v2" and report["status"] == case["status"] == "passed"
            and case["spec"] == expected and not case["temporalJitter"], "Incomplete or mismatched acceptance receipt")
    require([row["frame"] for row in case["frames"]] == list(range(expected["frames"])), "Expected all rendered frames")
    rows = []
    for frame in case["frames"][WARMUP:]:
        stream, = (entry for entry in frame["streaming"] if entry["passName"] == "VBuffer")
        ready = frame["readiness"]
        selection = stream["regionSelection"]
        require((frame["actualRenderWidth"], frame["actualRenderHeight"]) == RESOLUTION and not frame["warmup"]
                and ready["ready"] and ready["completedPages"] == ready["requiredPages"] > 0
                and Path(stream["assetPath"]).resolve() == Path(f"{SCENE}.meshstream.bin").resolve()
                and stream["loadFailures"] == stream["requestOverflows"] == stream["allocationFailures"] == 0
                and stream["activePages"] <= stream["maxResidentPages"]
                and stream["geometryUsedBytes"] <= stream["geometryBudgetBytes"],
                f"Frame {frame['frame']}: mismatched extent or incomplete geometry streaming")
        deferred, = (node for node in frame["nodes"] if node["name"] == "Deferred")
        gpu_ms = deferred["gpuMs"]
        require(gpu_ms is not None and math.isfinite(gpu_ms) and gpu_ms > 0, "Expected a Deferred Vulkan timestamp")
        rows.append(dict(frame=frame["frame"], deferredGpuMs=gpu_ms, textureResidentBytes=stream["textureResidentBytes"],
                         textureIdentity=stream["textureSemanticIdentitySha256"], geometryBytes=stream["geometryUsedBytes"],
                         geometryPages=stream["residentPages"], geometryTopology=stream["originalTopologyHash"],
                         primitiveMetadata=stream["primitiveMetadataHash"], regions=selection["publishedRegions"],
                         terminalPages=ready["requiredPages"]))
    return case, rows


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
    require(shot["frame"] == DIAGNOSTIC_FRAMES - 1, f"{variant}/{view}: missing matched capture")
    return output, shot, rows[-1]


def cliff_pixels(directory, world, covered):
    # The same exterior stencil on the shared depth image identifies steep surfaces.
    dy, dx = np.gradient(world, axis=(0, 1))
    normal = np.cross(dx, dy)
    lengths = np.linalg.norm(normal, axis=-1)
    slope_ny = np.divide(np.abs(normal[..., 1]), lengths, out=np.ones_like(lengths), where=lengths > 1e-8)
    max_step = np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1))
    # Match source texel centres for the second linear RGBA weight map.
    material_root = (directory / "material").resolve()
    uri = read_json(material_root / "triplanar.material")["terrainLayers"]["weightMaps"][1]["uri"]
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


def verify_visual(directory):
    all_views = {}
    geometry = None
    cliff = control = None
    geometry_residency = None
    for view in VIEWS:
        a, b = [read_diagnostic(directory, variant, view) for variant in VARIANTS]
        identities = [{name: shot["rawImages"][name]["rawSha256"]
                       for name in ("VBuffer.depth", "VBuffer.visibility", "Shadows.shadows")} for _, shot, _ in (a, b)]
        require(identities[0] == identities[1], f"{view}: projection changed depth, visibility or VSM")
        if geometry is None:
            geometry = identities[0]
            world, covered = capture_geometry(a[0], a[1], camera(), RESOLUTION)
            cliff, control = cliff_pixels(directory, world, covered)
            require(cliff.sum() >= 500 and control.sum() >= 500, "CanyonVista missed a steep cliff or its planar control")
            mask = np.zeros((*cliff.shape, 3), dtype=np.uint8)
            mask[cliff] = (255, 140, 0)
            mask[control] = (40, 160, 255)
            Image.fromarray(mask).save(directory / "CliffAndControlPixels.png")
        else:
            require(identities[0] == geometry, f"{view}: AOV does not use the same depth, visibility and VSM")
        require(a[2]["textureIdentity"] and a[2]["textureIdentity"] == b[2]["textureIdentity"],
                f"{view}: the two projections do not have identical texture and published mip bindings")
        cut_keys = ("regions", "terminalPages", "geometryPages", "geometryBytes")
        cuts = [tuple(entry[key] for key in cut_keys) for entry in (a[2], b[2])]
        require(cuts[0] == cuts[1] and (geometry_residency is None or geometry_residency == cuts[0]),
                f"{view}: measured geometry residency differs across views")
        geometry_residency = cuts[0]
        if view == "final":
            compare = pair_image(directory, view, a[0] / a[1]["file"], b[0] / b[1]["file"])
            all_views[view] = dict(image=compare)
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
                               controlMeanAbsoluteDelta=control_delta)
    write_json(directory / "Visual.json", dict(passed=True, camera=camera(), resolution=RESOLUTION, captureFrame=DIAGNOSTIC_FRAMES - 1,
               cliffPixels=int(cliff.sum()), planarControlPixels=int(control.sum()), sharedGeometryAndShadow=geometry,
               geometryResidency=dict(zip(cut_keys, geometry_residency, strict=True)),
               textureSemanticIdentitySha256=a[2]["textureIdentity"], views=all_views))


def measure_verdict(runs):
    a, b = [[run["deferredMedianMs"] for run in runs if run["variant"] == variant] for variant in VARIANTS]
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
    else:
        verdict, reason = "extend", "not-separated-beyond-within-variant-spread"
    return dict(verdict=verdict, stopReason=reason, planarMedianMs=median(a), triplanarMedianMs=median(b),
                deltaMs=delta, relativeDelta=relative, withinVariantSpreadMs=spread, separationMs=separation)


def verified_inputs(directory):
    inputs = read_json(directory / "Inputs.json")
    for filename, expected in inputs["files"].items():
        require(digest(filename) == expected, f"Comparison input changed: {filename}")
    return inputs


def run_measurements(directory, executable):
    visual = read_json(directory / "Visual.json")
    require(visual["passed"], "Verify the comparison images before ordinary timing")
    inputs = verified_inputs(directory)
    runs = []
    identity = workload = None
    for index, variant in enumerate(VARIANTS * 4, 1):
        case_dir = directory / "timing" / variant / "final"
        output = directory / "runs" / f"{index:02d}-{variant}"
        print(f"Deferred timing process {index}: {variant}", flush=True)
        run_vsm(argparse.Namespace(cases=case_dir / "Cases.json", exe=executable, output=output, runs=1, timeout=TIMEOUT))
        manifest, captures = verify_package(output)
        (receipt, report), = captures
        expected, = manifest["cases"]
        case, rows = measurements(report, expected)
        current = ({key: value for key, value in expected.items() if key not in ("id", "scene")}, case["resolvedGraph"])
        require(workload is None or workload == current, f"{variant}: comparison did not use the same production graph")
        workload = current
        final = rows[-1]
        for row in rows:
            require(all(row[key] == value for key, value in visual["geometryResidency"].items()),
                    f"{variant}: measured window uses different geometry residency than the matched diagnostics")
            require(row["textureResidentBytes"] == final["textureResidentBytes"],
                    f"{variant}: measured window includes texture transitions")
        current = tuple(final[key] for key in ("regions", "terminalPages", "geometryTopology", "primitiveMetadata"))
        require(identity is None or identity == current, f"{variant}: geometry topology or region cut differs")
        identity = current
        require(final["textureResidentBytes"] > 0, f"{variant}: missing resident texture allocation count")
        runs.append(dict(process=index, variant=variant, deferredMedianMs=median(row["deferredGpuMs"] for row in rows),
                         textureResidentBytes=final["textureResidentBytes"], receipt=receipt.relative_to(directory).as_posix(),
                         receiptSha256=digest(receipt), frames=rows))
        if index in (4, 8):
            verified_inputs(directory)
            result = measure_verdict(runs)
            texture = {variant: sorted({run["textureResidentBytes"] for run in runs if run["variant"] == variant})
                       for variant in VARIANTS}
            write_json(directory / "Timing.json", dict(result=result, textureResidentBytes=texture, runs=runs,
                       resolution=RESOLUTION, camera=camera(), warmupFrames=WARMUP, measuredFrames=FRAMES - WARMUP,
                       shadowRasterWorkLimit=SHADOW_WORK_LIMIT, shadowCasterCut="terminal",
                       geometryBudgetBytes=GEOMETRY_BUDGET, stableGeometry=visual["geometryResidency"],
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
    require(result == measure_verdict(runs) and result["verdict"] != "extend", "The timing comparison is incomplete")
    require(all(digest(directory / run["receipt"]) == run["receiptSha256"] for run in runs), "Timing receipts changed")
    output = directory / "report"
    output.mkdir(exist_ok=True)
    width, height = timing["resolution"]
    figure, (gpu, memory) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, constrained_layout=True)
    for variant, color in zip(VARIANTS, ("#447cae", "#ac5938"), strict=True):
        samples = [run for run in runs if run["variant"] == variant]
        process = [run["process"] for run in samples]
        times = [run["deferredMedianMs"] for run in samples]
        bytes_mib = [run["textureResidentBytes"] / (1024 ** 2) for run in samples]
        gpu.scatter(process, times, color=color, label=variant, s=40)
        gpu.axhline(median(times), color=color, alpha=0.5, linestyle="--")
        memory.scatter(process, bytes_mib, color=color, s=40)
    figure.suptitle(f"Ancient Rift Valley / CanyonVista / {width} × {height}")
    gpu.set_ylabel("Deferred GPU time (ms)")
    gpu.legend(title="Material projection and median")
    memory.set_ylim(0, max(run["textureResidentBytes"] for run in runs) / (1024 ** 2) * 1.18)
    memory.set_ylabel("Resident textures (MiB)")
    memory.set_xlabel("Independent process launch")
    memory.set_xticks([run["process"] for run in runs])
    for axis in (gpu, memory):
        axis.grid(axis="y", alpha=0.2)
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
    rows = ["| Run | Projection | Deferred GPU median (ms) | Resident textures (MiB) |", "|---|---|---:|---:|"]
    rows.extend(f"| {run['process']} | {run['variant']} | {run['deferredMedianMs']:.6f} | "
                f"{run['textureResidentBytes'] / (1024 ** 2):.4f} |" for run in runs)
    (output / "Report.md").write_text("\n".join([
        "# Ancient Rift Valley cliff projection cost", "",
        f"CanyonVista at {width} × {height}. Each fresh process retains disk caches, warms up for {timing['warmupFrames']} frames "
        f"and measures {timing['measuredFrames']} frames. The independent statistical unit is the process median. "
        "Dashed lines show medians across processes. GPU clocks were left at their defaults. Both variants use the same "
        f"{timing['shadowRasterWorkLimit']:,}-item VSM raster work budget with 1,024 physical pages "
        "and the complete terminal caster cut. Automatic main-view geometry LOD remains active. "
        f"The shared {timing['geometryBudgetBytes'] // (1024 ** 2)} MiB geometry budget reaches a stable "
        f"{timing['stableGeometry']['geometryPages']:,}-page cut before timing.", "",
        "![Deferred GPU and resident texture cost](CliffCost.png)", "", *rows, "",
        f"The observed median changed by {result['deltaMs']:+.6f} ms ({result['relativeDelta']:+.2%}). "
        f"Verdict `{result['verdict']}` after {len(runs)} runs with stop reason `{result['stopReason']}`.", "",
        "The Deferred timestamp covers one Vulkan render graph node. Resident texture bytes are the tracked VBuffer scene "
        "texture allocations. Diagnostics and readbacks run in separate processes.", "",
        f"[Source data](../Timing.json), SHA-256 `{digest(timing_path)}`. Raw receipt paths and hashes are recorded per launch. "
        f"Convert `textureResidentBytes` to MiB using 1,048,576 bytes per MiB. Python {sys.version.split()[0]}, "
        f"Matplotlib {matplotlib.__version__}.", "", "```powershell", command, "```", ""]), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "diagnostic", "verify", "timing", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--view", choices=VIEWS, help="Capture just one diagnostic pair before the full verification")
    args = parser.parse_args()
    directory = args.output.resolve()
    try:
        if args.action == "prepare":
            directory.mkdir(parents=True, exist_ok=False)
            prepare(directory)
        if args.action in ("diagnostic", "timing"):
            require(args.executable and args.executable.is_file(), "Build and provide MetallicRHITests")
        if args.action == "diagnostic":
            for view in (args.view,) if args.view else VIEWS:
                for variant in VARIANTS:
                    capture(args.executable.resolve(), directory / "diagnostic" / variant / view, timeout=TIMEOUT)
        if args.action == "verify" or (args.action == "diagnostic" and not args.view):
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
