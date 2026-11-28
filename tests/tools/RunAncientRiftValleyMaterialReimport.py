"""Reimport the second Ancient Rift Valley weight map in an isolated production-target build."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from RunTerrainMaterialReimportRegression import (
    OpenEXR, TEXTURE_ROLES, changed_binding, isolated_path, read_json, require, require_unchanged, run, snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Asset/Terrain/AncientRiftValley"
FULL_TARGET = "MetallicAncientRiftValley2x2Sample"
MATERIAL_TARGET = "MetallicAncientRiftValley2x2Materials"


def bindings(sample):
    terrain = read_json(sample.with_suffix(".material"))["terrainLayers"]
    maps = terrain["weightMaps"]
    require(len(maps) == 2 and len(terrain["layers"]) == 8, "Expected both maps and eight material layers")
    result = {f"weightMaps[{index}]": item for index, item in enumerate(maps)}
    for layer in terrain["layers"]:
        result.update({f"{layer['name']}.{role}": layer[role] for role in TEXTURE_ROLES})
    return result


def edit_second_map(path, first, second):
    # Exchange channel values in the eight-layer apron. The source sum is preserved so map 0 can
    # retain its exact packed payload after joint normalization and quantization of both maps.
    region = (slice(624, 630), slice(972, 981))
    with OpenEXR.File(str(path), separate_channels=True) as source:
        header = {name: value for name, value in source.header().items() if name != "channels"}
        pixels = {name: source.channels()[name].pixels.copy() for name in "RGBA"}
    temp = pixels[first][region].copy()
    pixels[first][region] = pixels[second][region]
    pixels[second][region] = temp
    OpenEXR.File(header, pixels).write(str(path))


def verify(args, work, report):
    source = work / "authoring"
    source.mkdir()
    shutil.copy2(SOURCE / "TerrainAsset.json", source)
    shutil.copytree(SOURCE / "Maps", source / "Maps")
    shutil.copytree(SOURCE / "Materials", source / "Materials", ignore=shutil.ignore_patterns("*.hip", "*.jpg"))
    manifest = source / "TerrainAsset.json"
    material_source = read_json(manifest)
    second_map = isolated_path(work, source / material_source["weightMaps"][1]["path"])
    project = work / "fixture"
    project.mkdir()
    helper = (ROOT / "cmake/AncientRiftValleySample.cmake").as_posix()
    (project / "CMakeLists.txt").write_text(
        f'''cmake_minimum_required(VERSION 3.20)
project(AncientRiftValleyMaterialReimport LANGUAGES NONE)
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
add_executable(MetallicMeshletCook IMPORTED)
set_target_properties(MetallicMeshletCook PROPERTIES IMPORTED_LOCATION "{args.cooker.as_posix()}")
include("{helper}")
add_metallic_ancient_rift_valley_sample("{manifest.as_posix()}" "${{CMAKE_CURRENT_BINARY_DIR}}/sample" MetallicMeshletCook)
''', encoding="utf-8")
    build_dir = work / "build"
    run([str(args.cmake), "-S", str(project), "-B", str(build_dir), "-G", "Ninja",
         f"-DCMAKE_MAKE_PROGRAM={args.ninja.as_posix()}"], work / "configure.log")
    build = [str(args.cmake), "--build", str(build_dir), "--parallel", "2", "--target"]
    run(build + [FULL_TARGET], work / "initial-build.log")
    sample = build_dir / "sample/ancient_rift_valley_2x2.gltf"
    geometry = sorted(path for path in sample.parent.iterdir() if path.is_file() and (
        path.suffix in (".gltf", ".bin") or path.name.endswith((".metallic_scene.json", ".meshstream.regions.json",
                                                               ".meshstream.bin.json"))))
    require(sample in geometry and sample.with_suffix(".metallic_scene.json") in geometry
            and sum(path.name.endswith(".gltf.meshstream.bin") for path in geometry) == 5,
            "Expected the aggregate, scene document, and all four independent StreamAssets")
    before = snapshot(geometry, work)
    previous = bindings(sample)
    edits = []
    for target, (first, second) in zip((FULL_TARGET, MATERIAL_TARGET), (("R", "G"), ("B", "A"))):
        edit_second_map(second_map, first, second)
        steps = run(build + [target], work / f"{target}.log")
        require(len(steps) == 1 and "material" in steps[0].lower() and "import" in steps[0].lower(),
                f"{target} must execute exactly one material import, found {steps}")
        after = bindings(sample)
        edited = changed_binding(sample, previous, after, "weightMaps[1]")
        geometry_after = snapshot(geometry, work)
        require_unchanged(before, geometry_after)
        edits.append({"target": target, "steps": steps, "binding": edited, "geometry": geometry_after})
        previous = after
    require(not run(build + [FULL_TARGET], work / "clean-build.log"), "Incremental sample remained dirty")
    report.update(geometryBefore=before, edits=edits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("cmake", "ninja", "cooker", "output-root"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    args.cooker = args.cooker.resolve()
    if not args.cooker.is_file():
        parser.error(f"Build the terrain cooker first: {args.cooker}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    print(f"Ancient Rift Valley material reimport evidence: {work}", flush=True)
    report = {"passed": False}
    try:
        verify(args, work, report)
        report["passed"] = True
        print("Both build targets imported only weight map 1 and preserved the geometry and scene fingerprints.")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        report["error"] = str(error)
        print(f"Ancient Rift Valley material reimport failed: {error}", file=sys.stderr)
    finally:
        (work / "AncientRiftValleyMaterialReimport.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
