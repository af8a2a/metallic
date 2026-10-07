"""Phase 0 fixed material workloads: capture, verify, and compare linear HDR evidence."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "LookDev/MaterialSystem"


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def asset_paths(case_path=None):
    if case_path is not None:
        config = read(case_path)
        return sorted({case_path, *[Path(p) for p in config["assetFiles"]],
                       *[case_path.parent / c["graph"] for c in config["cases"]]})
    files = set(FIXTURES.rglob("*.json"))
    for directory in [ROOT / "Asset/LookDev/OpenPBRDefault"]:
        files.update(p for p in directory.rglob("*") if p.is_file() and not p.name.endswith(".meshlets.bin"))
    groom = ROOT / "External/RTXCR-Assets/Claire/ponyTail_15vtx.gltf"
    files.add(groom)
    model = read(groom)
    for item in model.get("buffers", []) + model.get("images", []):
        uri = item.get("uri", "")
        if uri and not uri.startswith("data:"):
            files.add(groom.parent / uri)
    files.add(ROOT / "External/RTXCR-Assets/EnvironmentMaps/studio_small_09_1k.hdr")
    sidecar = groom.with_suffix(".metallic_scene.json")
    if sidecar.exists():
        files.add(sidecar)
    return sorted(files)


def identity(exe, case_path=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()
    sources = list((ROOT / "Shaders").rglob("*.slang")) + list((ROOT / "Shaders").rglob("*.hlsli"))
    sources += list((ROOT / "Source").rglob("*.cpp")) + list((ROOT / "Source").rglob("*.h"))
    sources += [ROOT / "tests/rhi/MaterialBaselineTests.cpp", Path(__file__)]
    # Inspect superproject metadata without requiring access to every optional
    # submodule checkout. External index entries and workload hashes remain recorded.
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short", "--ignore-submodules=all"),
            "statusScope": "Superproject only; submodule worktrees are not inspected",
            "submodules": git("ls-files", "--stage", "External"), "platform": platform.platform(),
            "python": sys.version, "exe": str(exe),
            "binaries": {str(p): digest(p) for p in [exe, *sorted(exe.parent.glob("*.dll"))]},
            "workload": {(str(p) if case_path else str(p.relative_to(ROOT)).replace("\\", "/")): digest(p)
                         for p in asset_paths(case_path)},
            "sources": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(set(sources))}}


def seal(output):
    save(output / "Manifest.json", {str(p.relative_to(output)).replace("\\", "/"): digest(p)
         for p in sorted(output.rglob("*")) if p.is_file() and p.name != "Manifest.json"})


def images(output, excluded=()):
    import numpy as np
    result = {}
    for report_path in sorted(output.glob("run-*/MaterialBaseline.json")):
        report = read(report_path)
        for case in report["cases"]:
            if case["id"] in excluded or not case.get("validHDR", True):
                continue
            dtype = "<f2" if case["format"] == "RGBA16F" else "<f4"
            pixels = np.fromfile(report_path.parent / case["image"], dtype=dtype).astype(np.float32)
            pixels = pixels.reshape(case["height"], case["width"], 4)
            if not np.isfinite(pixels).all():
                raise ValueError("Nonfinite HDR image")
            result.setdefault(case["id"], []).append(pixels)
    return result


def verify(output):
    manifest = read(output / "Manifest.json")
    actual = {str(p.relative_to(output)).replace("\\", "/"): digest(p)
              for p in output.rglob("*") if p.is_file() and p.name != "Manifest.json"}
    if manifest != actual:
        raise ValueError("Evidence file set or hash changed")
    reports = sorted(output.glob("run-*/MaterialBaseline.json"))
    if len(reports) != 3 or read(output / "Process.json")["exitCodes"] != [0, 0, 0]:
        raise ValueError("Three successful independent processes required")
    config = read(output / "fixtures/Cases.json")
    specifications = config["cases"]
    expected = {c["id"] for c in specifications}
    if not expected or len(expected) != len(specifications):
        raise ValueError("Distinct material cases required")
    invalid_hdr = {}
    first_cases = {c["id"]: c for c in read(reports[0])["cases"]}
    for path in reports:
        report = read(path)
        if report["validation"] or len(report["cases"]) != len(expected) or {c["id"] for c in report["cases"]} != expected:
            raise ValueError("Invalid case set or diagnostic timings")
        for case in report["cases"]:
            spec = next(c for c in specifications if c["id"] == case["id"])
            if any(case.get(key) != value for key, value in spec.items()):
                raise ValueError("Captured case differs from fixed fixture")
            if config.get("version", 1) >= 2:
                first = first_cases[case["id"]]
                if case["resolvedGraph"] != first["resolvedGraph"]:
                    raise ValueError("Resolved graph changed between processes")
                if spec.get("requireEnvironment", True) and not case["environmentTransitions"]:
                    raise ValueError("Missing HDR environment evidence")
            if case["format"] not in ("RGBA16F", "RGBA32F") or Path(case["image"]).name != case["image"]:
                raise ValueError("Invalid HDR format or image path")
            if config.get("version", 1) >= 2:
                import numpy as np
                pixels = np.fromfile(path.parent / case["image"], dtype="<f2" if case["format"] == "RGBA16F" else "<f4")
                count_nonfinite = int(np.count_nonzero(~np.isfinite(pixels)))
                if (pixels.size != case["width"] * case["height"] * 4 or
                    count_nonfinite != case["nonfiniteComponents"] or case["validHDR"] != (count_nonfinite == 0)):
                    raise ValueError("HDR quality metadata mismatch")
                if count_nonfinite:
                    invalid_hdr.setdefault(case["id"], {})[path.parent.name] = count_nonfinite
            frames = case["frames"]
            warmup, count = config.get("warmupFrames", 32), config.get("timingFrames", 64)
            if [f["frame"] for f in frames] != list(range(warmup, warmup + count)):
                raise ValueError("Missing or reordered timing frames")
            if len({f["executionId"] for f in frames}) != count:
                raise ValueError("Repeated GPU execution IDs")
            if any(not math.isfinite(f["graphMs"]) or f["graphMs"] <= 0 for f in frames):
                raise ValueError("Invalid GPU times")
            required = spec.get("requiredTiming") or {"OpenPBRPathTrace": ("Reference", "Path trace shading"),
                        "OpenPBRDeferred": ("Deferred", "Material classification"),
                        "RTXCRChiang": ("PathTrace", "Path trace shading")}[case["id"]]
            for frame in frames:
                if config.get("version", 1) >= 2:
                    # The registered VBuffer graph forks early/late raster work,
                    # then joins before Deferred. Its graphics envelope includes
                    # those waits; it is not the sum of concurrent queue times.
                    expected_branches = spec.get("expectedAsyncComputeBranches", 2 if spec.get("backend") == "Deferred" else 0)
                    if frame["asyncComputeBranches"] != expected_branches or any(n["queue"] != 0 for n in frame["nodes"]):
                        raise ValueError("Unexpected asynchronous timing scope")
                node = next((n for n in frame["nodes"] if n["name"] == required[0]), None)
                if node is None or node["gpuMs"] is None:
                    raise ValueError("Missing material pass timing")
                section = next((s for s in node["sections"] if s["name"] == required[1]), None)
                if required[1] and (section is None or section["gpuMs"] is None):
                    raise ValueError("Missing required GPU scope")
                if not math.isfinite(node["gpuMs"]) or node["gpuMs"] <= 0:
                    raise ValueError("Invalid material pass time")
    import numpy as np
    aa = {}
    for name, runs in images(output, invalid_hdr).items():
        if len(runs) != 3:
            raise ValueError("Missing image repeat")
        aa[name] = {"maxAbsoluteDifference": float(max(np.max(np.abs(x - runs[0])) for x in runs[1:])),
                    "rgbRMSE": [float(np.sqrt(np.mean((x[:, :, :3] - runs[0][:, :, :3]) ** 2))) for x in runs[1:]],
                    "exact": all(np.array_equal(x, runs[0]) for x in runs[1:])}
    return {"verified": True, "meaning": "Evidence integrity only; invalidHDR cases are NOT qualified performance baselines.",
            "invalidHDR": invalid_hdr, "imageAA": aa}


def run(args):
    exe, output = args.exe.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    cache = exe.parent.parent / "CMakeCache.txt"
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache.read_text():
        raise ValueError("Use an existing Release build")
    if (ROOT / "build/shader-experiment.lock").exists():
        raise ValueError("Shader experiment lock exists; do not overlap GPU experiments")
    lock = ROOT / "build/material-phase0.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        output.mkdir(parents=True)
        case_path = args.cases.resolve() if args.cases else None
        before = identity(exe, case_path)
        save(output / "Identity.json", before)
        shutil.copytree(case_path.parent if case_path else FIXTURES, output / "fixtures")
        shutil.copy2(cache, output / "CMakeCache.txt")
        shutil.copy2(__file__, output / "MaterialBaseline.py")
        shutil.copy2(ROOT / "tests/rhi/MaterialBaselineTests.cpp", output / "MaterialBaselineTests.cpp")
        with (output / "WorkingTree.patch").open("wb") as stream:
            subprocess.run(["git", "diff", "--binary", "--ignore-submodules=all", "HEAD"], cwd=ROOT, stdout=stream, check=True)
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu", "--format=csv"], text=True)
        (output / "GPU.csv").write_text(gpu)
        (output / "GPUProcesses.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
        env = {k: v for k, v in os.environ.items() if not k.startswith("METALLIC_")}
        env.update(METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0")
        if env.get("VK_INSTANCE_LAYERS"):
            raise ValueError("Remove injected VK_INSTANCE_LAYERS for normal timing")
        if case_path:
            env["METALLIC_MATERIAL_BASELINE_CASES"] = str(case_path)
        save(output / "Environment.json", {k: v for k, v in env.items() if k.startswith("METALLIC_")})
        processes = {"exitCodes": [], "commands": [], "timeoutSeconds": args.timeout,
                     "cache": "Existing disk caches retained; frames 0-31 excluded; no GPU clock changes"}
        for index in range(3):
            destination = output / f"run-{index}"
            destination.mkdir()
            command = [str(exe), "--gtest_filter=*material_phase0_baseline*", "--rhi-no-validation",
                       "--output-dir", str(destination), f"--gtest_output=xml:{destination / 'Tests.xml'}"]
            processes["commands"].append(command)
            print(f"Capturing process {index + 1}/3: {destination}", flush=True)
            with (destination / "stdout.log").open("wb") as stdout, (destination / "stderr.log").open("wb") as stderr:
                with (destination / "Telemetry.csv").open("wb") as telemetry:
                    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw", "--format=csv", "-l", "1"], stdout=telemetry, stderr=subprocess.DEVNULL)
                    try:
                        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout)
                    finally:
                        monitor.terminate()
                        monitor.wait(timeout=10)
            processes["exitCodes"].append(completed.returncode)
            save(output / "Process.json", processes)
            tests = ET.parse(destination / "Tests.xml").getroot()
            if completed.returncode or tests.get("tests") != "1" or tests.get("failures") != "0" or tests.findall(".//skipped"):
                raise ValueError(f"Baseline process failed or skipped: {destination}")
        after = identity(exe, case_path)
        if any(before[key] != after[key] for key in ("binaries", "workload", "sources", "commit", "submodules")):
            raise ValueError("Inputs changed during capture")
        seal(output)
        print(json.dumps(verify(output), indent=2))
    finally:
        lock.unlink()


def compare(baseline, candidate):
    import numpy as np
    baseline_verification = verify(baseline)
    candidate_verification = verify(candidate)
    excluded = set(baseline_verification["invalidHDR"]) | set(candidate_verification["invalidHDR"])
    left, right = read(baseline / "Identity.json"), read(candidate / "Identity.json")
    if left["workload"] != right["workload"] or (baseline / "GPU.csv").read_text().splitlines()[0:1] != (candidate / "GPU.csv").read_text().splitlines()[0:1]:
        raise ValueError("Workload identity mismatch")
    # Temperature/pstate may change; GPU UUID and driver must remain identical.
    if (baseline / "GPU.csv").read_text().splitlines()[1].split(",")[:3] != (candidate / "GPU.csv").read_text().splitlines()[1].split(",")[:3]:
        raise ValueError("GPU or driver mismatch")
    a, b = images(baseline, excluded), images(candidate, excluded)
    result = {}
    for name in a:
        delta = b[name][0] - a[name][0]
        result[name] = {"maxAbsoluteDifference": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta * delta))),
                        "exact": bool(np.array_equal(a[name][0], b[name][0]))}
        for label, directory in (("baseline", baseline), ("candidate", candidate)):
            medians = []
            for path in sorted(directory.glob("run-*/MaterialBaseline.json")):
                case = next(c for c in read(path)["cases"] if c["id"] == name)
                medians.append(statistics.median(f["graphMs"] for f in case["frames"]))
            result[name][label + "GraphMediansMs"] = medians
    return {"cases": result, "excludedInvalidHDR": sorted(excluded), "baselineAA": baseline_verification["imageAA"],
            "candidateAA": candidate_verification["imageAA"],
            "performance": "Independent-process medians; these runs are not interleaved ABBA. No automatic speedup acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("run")
    capture.add_argument("--exe", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--timeout", type=int, default=600)
    capture.add_argument("--cases", type=Path, help="Explicit Cases.json; defaults to Phase 0 fixtures")
    check = sub.add_parser("verify")
    check.add_argument("output", type=Path)
    diff = sub.add_parser("compare")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args)
    elif args.action == "verify":
        print(json.dumps(verify(args.output), indent=2))
    else:
        print(json.dumps(compare(args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()
