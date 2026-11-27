#!/usr/bin/env python3
"""Capture and verify the 4x4 and 8x8 terrain streaming regressions."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PrepareTerrainStreamingAcceptance import ROOT, read_json, write_json


def capture(executable, case):
    output = case / "capture"
    output.mkdir()
    log = output / "acceptance.log"
    environment = {**os.environ, "METALLIC_VSM_ACCEPTANCE_CASES": str(case / "Cases.json")}
    command = [str(executable), "--filter", "virtual_shadow_map_acceptance", "--rhi-bindless",
               "--rhi-validation", "--output-dir", str(output), "--gtest_color=no"]
    print(f"Capturing {case.name}", flush=True)
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or not (output / "VirtualShadowMapAcceptance.json").is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
            print(line, file=sys.stderr)
        raise RuntimeError(f"RHI capture failed for {case.name}. Log: {log}")
    return output


def keep_first_crossing(case):
    targets = read_json(case / "Targets.json")["firstCrossing"]
    path = case / "Cases.json"
    manifest = read_json(path)
    specification, = manifest["cases"]
    frames = targets["captureFrames"][-1] + 1
    specification.update(frames=frames, captureFrames=targets["captureFrames"],
                         streamingEvidenceFrames=list(range(frames)),
                         events=[event for event in specification["events"] if event["frame"] < frames])
    write_json(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles-per-axis", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"Build MetallicRHITests first: {args.executable}")
    if any(importlib.util.find_spec(name) is None for name in ("numpy", "PIL")):
        parser.error("Install Tools/requirements-terrain.txt for the terrain regression")

    args.output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=args.output.resolve()))
    print(f"Terrain regression artifacts: {run}", flush=True)
    scripts = Path(__file__).resolve().parent
    executable = args.executable.resolve()
    four_by_four = args.tiles_per_axis == 4
    verifier = "VerifyTerrainStreamingRegression.py" if four_by_four else "VerifyTerrain8x8StreamingRegression.py"
    command = [sys.executable, "-B", str(scripts / verifier), "--report", str(run / "TerrainStreamingRegression.json")]
    try:
        subprocess.run([sys.executable, "-B", str(scripts / "PrepareTerrainStreamingAcceptance.py"),
                        "--tiles-per-axis", str(args.tiles_per_axis), "--output", str(run)],
                       cwd=ROOT, check=True)
        if four_by_four:
            directed = run / "motion-directed-tight"
            keep_first_crossing(directed)
        else:
            directed = run / "motion-directed-8x8-tight"
            command += ["--motion", str(capture(executable, run / "motion-tight"))]
        command += ["--directed", str(capture(executable, directed))]
        command += ["--fixed", str(capture(executable, run / "fixed-tight"))]
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"Terrain regression could not capture the scene: {error}", file=sys.stderr)
        return 1
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
