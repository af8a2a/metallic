#!/usr/bin/env python3
"""Create a disposable EXR terrain and verify viewport placement and reimport."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import OpenEXR


ROOT = Path(__file__).resolve().parents[2]


def stage_terrain(directory, height_delta):
    directory.mkdir()
    # Nine source samples span [-8, 8] on each horizontal axis. Their
    # plane makes the click heights and both components of the slope known.
    axis = np.linspace(-8, 8, 9, dtype=np.float32)
    x, z = np.meshgrid(axis, axis)
    height = directory / "height.exr"
    OpenEXR.File({}, {"R": 2.0 + x * 0.25 + z * 0.125 + height_delta}).write(str(height))
    field = directory / "terrain.mthf"
    surface = directory / "terrain_height.gltf"
    scene = directory / "terrain.gltf"
    subprocess.run([sys.executable, "-B", ROOT / "Tools/BuildTerrainMesh.py", height,
                    "--output", surface, "--horizontal-size", "16", "--height-scale", "1",
                    "--tiles-per-axis", "2", "--height-field-output", field], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", ROOT / "Tools/BuildTerrainVSMSample.py",
                    surface, ROOT / "Asset/Terrain/TerrainVSM.metallic_scene.json",
                    ROOT / "Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json", scene,
                    "--height-field", field], cwd=ROOT, check=True)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=args.output.resolve()))
    print(f"Terrain placement smoke artifacts: {run}", flush=True)
    try:
        scene = stage_terrain(run / "base", 0.0)
        stage_terrain(run / "raised", 3.0)
        subprocess.run([executable, "--build-meshstream", scene, "--output",
                        str(scene) + ".meshstream.bin"], cwd=ROOT, check=True)
        environment = os.environ | {
            "METALLIC_SMOKE_TEST_HIDDEN": "1",
            "METALLIC_DEBUG_VALIDATION": "1",
            "METALLIC_SMOKE_TEST_SAMPLE": "gpu-driven-terrain-placement-16x16-vsm",
            "METALLIC_SMOKE_TEST_TERRAIN_PLACEMENT": "1",
            "METALLIC_TERRAIN_PLACEMENT_SMOKE_FIXTURE": str(run),
        }
        log = run / "editor.log"
        with log.open("wb") as stream:
            result = subprocess.run([executable, "--scene", scene, "--smoke-test", "--debug-control"],
                                    cwd=run, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=420)
        contents = log.read_text(encoding="utf-8", errors="replace")
        if (result.returncode or "Vulkan validation:" in contents or "[error]" in contents or
                "[Smoke Terrain Placement] Passed" not in contents):
            print("\n".join(contents.splitlines()[-40:]), file=sys.stderr)
            raise RuntimeError(f"Terrain placement editor smoke failed ({result.returncode}); {log}")
        print(f"Terrain placement editor smoke passed. {log}")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
