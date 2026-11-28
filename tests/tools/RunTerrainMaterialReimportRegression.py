"""Exercise terrain material reimport targets in an isolated Ninja build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

try:
    import Imath
    import numpy as np
    import OpenEXR
    from PIL import Image
except ModuleNotFoundError as error:
    if error.name not in {"Imath", "numpy", "OpenEXR", "PIL"}:
        raise
    print("Terrain material reimport requires Tools/requirements-terrain.txt", file=sys.stderr)
    sys.exit(77)


FULL_TARGET = "MetallicTerrainMultilayer16x16VSMSample"
MATERIAL_TARGET = "MetallicTerrainMultilayer16x16VSMMaterials"
FIXTURE = Path(__file__).resolve().with_name("terrain-material-reimport")
TEXTURE_ROLES = ("baseColor", "normal", "roughness")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command, log):
    environment = {**os.environ, "NINJA_STATUS": "[%f/%t] "}
    result = subprocess.run(command, text=True, errors="replace", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=environment)
    log.write_text(result.stdout, encoding="utf-8")
    require(result.returncode == 0, f"{log.name} exited with code {result.returncode}:\n{result.stdout[-6000:]}")
    return re.findall(r"^\[\d+/\d+\] (.+)$", result.stdout, re.MULTILINE)


def fingerprint(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "mtimeNs": path.stat().st_mtime_ns}


def snapshot(paths, root):
    return {path.relative_to(root).as_posix(): fingerprint(path) for path in paths}


def require_unchanged(before, after):
    changed = [name for name in before if before[name] != after[name]]
    require(not changed, f"Material reimport changed geometry or the scene document: {', '.join(changed)}")


def isolated_path(root, path):
    resolved = path.resolve()
    require(resolved.is_relative_to(root), f"Fixture path outside the isolated build: {resolved}")
    return resolved


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bindings(sample):
    terrain = read_json(sample.with_suffix(".material"))["terrainLayers"]
    result = {"weights": terrain["weights"]}
    for layer in terrain["layers"]:
        result.update({f"{layer['name']}.{role}": layer[role] for role in TEXTURE_ROLES})
    require(len(result) == 13, "Expected the weight map and twelve layer texture bindings")
    return result


def texture_path(sample, uri):
    require(uri.startswith("asset://"), f"Expected a packaged texture URI: {uri}")
    return isolated_path(sample.parent, sample.parent / uri.removeprefix("asset://"))


def changed_binding(sample, before, after, key):
    require(before.keys() == after.keys(), "Reimport changed the material texture binding set")
    changed = {name for name in before if before[name] != after[name]}
    require(changed == {key}, f"Expected only {key} to change, got {sorted(changed)}")
    old_uri, new_uri = before[key]["uri"], after[key]["uri"]
    old_fields = {name: value for name, value in before[key].items() if name != "uri"}
    new_fields = {name: value for name, value in after[key].items() if name != "uri"}
    require(old_fields == new_fields and old_uri != new_uri, f"Reimport did not replace only the URI for {key}")
    old_texture, new_texture = (texture_path(sample, uri) for uri in (old_uri, new_uri))
    old_sha, new_sha = (fingerprint(path)["sha256"] for path in (old_texture, new_texture))
    require(old_sha != new_sha, f"Reimport did not replace the texture payload for {key}")
    return {"oldUri": old_uri, "newUri": new_uri, "oldSha256": old_sha, "newSha256": new_sha}


def edit_weight(path, replacement):
    source = OpenEXR.InputFile(str(path))
    try:
        header = source.header()
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = {name: np.frombuffer(source.channel(name, pixel_type), dtype=np.float32).copy() for name in "RGBA"}
    finally:
        source.close()
    chosen = "G" if channels["R"][0] >= channels["G"][0] else "R"
    for name, pixels in channels.items():
        pixels[0] = float(name == chosen)
    output = OpenEXR.OutputFile(str(replacement), header)
    try:
        output.writePixels({name: pixels.tobytes() for name, pixels in channels.items()})
    finally:
        output.close()
    path.write_bytes(replacement.read_bytes())
    return tuple(255 if name == chosen else 0 for name in "RGBA")


def edit_color(path):
    with Image.open(path) as source:
        pixels = source.convert("RGB")
    pixels.putpixel((0, 0), tuple(255 - component for component in pixels.getpixel((0, 0))))
    pixels.save(path)


def build_incrementally(build, target, log):
    steps = run(build + [target], log)
    require(len(steps) == 1 and "BuildTerrainVSMMaterial.py" in steps[0],
            f"{target} must execute exactly one material import: {log}")
    bake_commands = ("--build-meshstream", "BuildTerrainMesh.py", "BuildTerrainVSMSample.py",
                     "ExportTerrainMultilayerExample.py")
    require(not any(command in steps[0] for command in bake_commands),
            f"{target} chained a geometry or authoring bake to the material import: {log}")
    return steps


def verify(args, work, report):
    fixture_build = work / "build"
    run([str(args.cmake), "-S", str(FIXTURE), "-B", str(fixture_build), "-G", "Ninja",
         f"-DCMAKE_MAKE_PROGRAM={args.ninja}", f"-DMETALLIC_TEST_COOKER={args.cooker}",
         f"-DPython3_EXECUTABLE={sys.executable}"], work / "configure.log")
    build = [str(args.cmake), "--build", str(fixture_build), "--verbose", "--target"]
    steps = run(build + [FULL_TARGET], work / "baseline.log")
    require(any("--build-meshstream" in step for step in steps), "The full baseline did not cook a MeshletStreamAsset")

    source = fixture_build / "TerrainBlocks16x16/terrain_blocks_16x16_height"
    sample = fixture_build / "TerrainMultilayer16x16/terrain_multilayer_16x16.gltf"
    paths = [source.with_suffix(".gltf"), source.with_suffix(".bin"), sample, sample.with_suffix(".bin"),
             Path(str(sample) + ".meshstream.bin"), sample.with_suffix(".metallic_scene.json")]
    before = snapshot(paths, work)
    report["baselineGeometry"] = before
    original = bindings(sample)
    authoring = sample.parent / "source"
    asset = read_json(authoring / "terrain.layers.json")

    weight = isolated_path(work, authoring / asset["weightMap"])
    expected_weight = edit_weight(weight, work / "edited-weights.exr")
    weight_steps = build_incrementally(build, FULL_TARGET, work / "weight-full-build.log")
    weight_bindings = bindings(sample)
    weight_report = changed_binding(sample, original, weight_bindings, "weights")
    grid = read_json(source.with_suffix(".gltf"))["extras"]["metallicTerrainTiles"]
    with Image.open(texture_path(sample, weight_report["newUri"])) as pixels:
        edited_row = pixels.height // 2 - 1 if grid["flipZ"] else 0
        require(pixels.getpixel((0, edited_row)) == expected_weight, "Imported weight texel did not match the edited EXR")
    after_weight = snapshot(paths, work)
    report["weightEdit"] = {"target": FULL_TARGET, "steps": weight_steps,
                            "binding": weight_report, "geometry": after_weight}
    require_unchanged(before, after_weight)
    require(not run(build + [FULL_TARGET], work / "weight-clean-build.log"), "The full target remained dirty after the weight import")

    rock = next(layer for layer in asset["layers"] if layer["name"] == "rock")
    color = isolated_path(work, authoring / rock["baseColor"])
    edit_color(color)
    color_steps = build_incrementally(build, MATERIAL_TARGET, work / "texture-material-build.log")
    color_report = changed_binding(sample, weight_bindings, bindings(sample), "rock.baseColor")
    require(texture_path(sample, color_report["newUri"]).read_bytes() == color.read_bytes(),
            "Packaged rock base color did not match the edited layer texture")
    after_color = snapshot(paths, work)
    report["textureEdit"] = {"target": MATERIAL_TARGET, "steps": color_steps,
                             "binding": color_report, "geometry": after_color}
    require_unchanged(before, after_color)
    require(not run(build + [FULL_TARGET], work / "texture-clean-build.log"), "The full target remained dirty after the texture import")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--ninja", required=True, type=Path)
    parser.add_argument("--cooker", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.cooker = args.cooker.resolve()
    if not args.cooker.is_file():
        parser.error(f"Build the terrain cooker first: {args.cooker}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_root.resolve()))
    print(f"Terrain material reimport regression artifacts: {work}", flush=True)
    report = {"passed": False}
    try:
        verify(args, work, report)
        report["passed"] = True
        print("Both edits ran one material import and preserved all geometry and scene fingerprints.")
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        report["error"] = str(error)
        print(f"Terrain material reimport regression failed: {error}", file=sys.stderr)
    finally:
        (work / "TerrainMaterialReimportRegression.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
