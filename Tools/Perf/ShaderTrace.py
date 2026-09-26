"""Bounded WorkControl Shader Watch: one real production dispatch per process.

Uses the qualified MiniZorah history recipe. Diagnostic records are never timing
samples. Verification replays raw artifacts through the current metallicctl.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import DeepProfile as d
import ExperimentRunner as e
import WorkloadAssets as a
import WorkloadCase as w

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RunShaderTraceP1 import decode, read

ROOT = w.ROOT
PROTOCOL = "metallic.shader-trace.p2.v1"
CASE = ROOT / "Tools/Perf/WorkloadCase.MiniZorahHistory.json"
SITE = "stream.after-triangle-prepare"


def source_inventory():
    sources = w.source_inventory()
    paths = list((ROOT / "Shaders").rglob("*.hlsli")) + [Path(__file__),
        ROOT / "Tools/RunShaderTraceP1.py", *[ROOT / "Tools/Perf" / name for name in
        ("DeepProfile.py", "ExperimentRunner.py", "WorkloadCase.py", "WorkloadAssets.py")]]
    sources.update({p.relative_to(ROOT).as_posix(): w.digest(p) for p in paths})
    return sources


def recipe():
    case = w.validate_case(w.load(CASE))
    # These are output/timing budgets; camera/history/render settings stay fixed.
    return {**case, "rounds": 1, "sampleFrames": 8}


def selection(phase="early", group_x=0, local=0, triangle=None):
    w.require(phase in ("early", "late") and type(group_x) is int and 0 <= group_x <= 65534
              and type(local) is int and 0 <= local <= 127, "Invalid bounded invocation")
    result = {"phase": phase, "group": [group_x, 0, 0], "localIndex": local}
    if triangle is not None:
        w.require(type(triangle) is int and 0 <= triangle <= 0xffffffff, "Invalid triangleId predicate")
        result["predicate"] = {"field": "triangleId", "op": "eq", "value": triangle}
    return result


def schedule(acceptance, selected):
    if not acceptance:
        return [{"name": "watch", "selection": selected}]
    return [{"name": "normal-control", "selection": None}] + [
        {"name": f"{phase}-{i+1}", "selection": selection(phase), "expected": "Matched"}
        for phase in ("early", "late") for i in range(3)] + [
        {"name": "no-match", "selection": selection(triangle=0xffffffff), "expected": "NoMatch"},
        {"name": "site-not-reached", "selection": selection(local=127), "expected": "SiteNotReached"}]


def assets(path):
    graph = "Pipelines/Samples/gpu_driven_realtime.metallic_graph.json"
    scene = "Asset/MiniZorah/zorah_main_public.v2.gltf"
    stream = w.ASSETS["gpu-driven-sample"]
    dependencies = a.dependencies(ROOT / graph, ROOT / scene, ROOT / stream)
    if path:
        result = w.load(path)
        w.require(result["protocol"] == "metallic-declared-assets-v1" and result["stream"] == stream
                  and set(result["files"]) == {p.relative_to(ROOT).as_posix() for p in dependencies}, "Wrong asset closure")
    else:
        records = {}
        for item in dependencies:
            before = item.stat()
            checksum = w.digest(item)
            after = item.stat()
            w.require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Asset changed while hashing")
            records[item.relative_to(ROOT).as_posix()] = {"bytes": after.st_size, "mtimeNs": after.st_mtime_ns, "sha256": checksum}
            print(f"asset: {item.name} ({after.st_size} bytes)", flush=True)
        result = {"protocol": "metallic-declared-assets-v1", "graph": graph, "scene": scene,
                  "stream": stream, "scope": "declared MiniZorah graph/glTF/stream dependency closure", "files": records}
    w.asset_metadata_matches(result)
    return result


def replay(cli, capture):
    command = [str(cli), "--capture", str(capture), "--json", "shader", "verify"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    value = decode(json.loads(result.stdout.decode("utf-8")))
    w.require(result.returncode == 0 and value.get("status") == "ok", f"Offline decode failed: {result.stderr!r}")
    return value["result"]


def normalized_diagnostic(snapshot, baseline, phase, decoded):
    result = copy.deepcopy(snapshot)
    changed = [item for item in result["productionDispatches"] if item["scope"] == "diagnostic-dispatch"]
    w.require(len(changed) == 1 and changed[0]["phase"] == phase, "Wrong/multiple diagnostic bindings")
    item = changed[0]
    w.require(item.pop("instrumentation") == "Printf" and item.pop("dispatchToken") == decoded["dispatch"]["dispatchToken"],
              "Diagnostic token mismatch")
    w.require(item.pop("diagnosticCompilerSpirvSha256") == decoded["dispatch"]["variant"]["compilerSpirvSha256"],
              "Compiled variant differs from actual binding")
    item["spirvFnv1a64"] = item.pop("uninstrumentedSpirvFnv1a64")
    item["scope"] = "production-dispatch"
    w.require(result["productionDispatches"] == baseline["productionDispatches"], "Unrelated production binding changed")
    return result


def inspect(directory, case, row, cli):
    process = w.load(directory / "Process.json")
    w.require(process.get("exitCode") == 0, "Renderer did not exit cleanly")
    app = directory / "app"
    if row["selection"] is None:
        normal = w.analyze_run(app, case)
        w.require(normal["normalTiming"], "Control process was instrumented")
        return {"name": row["name"], "process": process, "selection": None,
                "identity": {k: normal["identity"][k] for k in ("snapshot", "camera", "renderExtent", "graph", "historyInvalidationPolicy")}}
    selected = row["selection"]
    report = w.load(app / "Capture.json")
    config = {**w.engine_config(case), "shaderTrace": selected}
    w.require(report.get("status") == "capture_complete" and report["config"] == config, "Incomplete/wrong workload config")
    w.require(report["workloadCase"] == config["workloadCase"] and report["hidden"] and report["validationRequested"]
              and report["measurementKind"] == "diagnostic" and report["shaderTraceRequested"], "Wrong diagnostic environment")
    w.require(not any(report.get(k) for k in ("graphicsCaptureInjected", "gpuTraceInjected", "renderDocInjected",
                                             "pipelineStatisticsRequested", "nvPerfRequested")), "Competing instrumentation")
    w.require(report["renderExtent"] == [case["renderWidth"], case["renderHeight"]]
              and report["outputExtent"] == [case["width"], case["height"]], "Workload extent drift")
    w.require(len(report["cases"]) == 1 and report["cases"][0]["variant"] == case["variant"], "Wrong round/variant")
    snapshots = report["cases"][0]
    w.require("framesFile" not in snapshots, "Diagnostic frames mislabeled as timing samples")
    trace = replay(cli, app / "shader-trace/capture")
    raw = read(app / "shader-trace/capture/0.bin")
    phase = selected["phase"]
    w.require(trace["selectedScopeComplete"] and trace["outcome"] in ("Matched", "NoMatch", "SiteNotReached")
              and trace["performanceEligible"] is False and trace["backendCollectionStatus"] == "Closed", "Observation incomplete")
    if "expected" in row:
        w.require(trace["outcome"] == row["expected"], "Unexpected observation outcome")
    dispatch, runtime, variant = trace["dispatch"], raw["runtime"], trace["dispatch"]["variant"]
    w.require(trace["site"]["name"] == SITE and trace["site"]["phase"] == dispatch["phase"] == phase
              and dispatch["pass"] == "VBuffer" and dispatch["dispatchOrdinal"] == (0 if phase == "early" else 1)
              and dispatch["execution"] > 0 and dispatch["commandBufferRecording"] > 0, "Dispatch identity mismatch")
    w.require(dispatch["queue"]["type"] == "Graphics" and dispatch["submit"]["execution"] == dispatch["execution"], "Missing real submission")
    timelines = dispatch["submit"]["frameTimelineValues"]
    w.require(timelines and all(type(v) is int and v > 0 for v in timelines)
              and dispatch["submit"]["timelineValue"] == (timelines[0] if len(timelines)==1 else None), "Ambiguous frame timeline evidence")
    w.require(raw["request"]["invocation"] == {k: selected[k] for k in ("group", "localIndex")}
              and raw["request"].get("predicate") == selected.get("predicate"), "Watch selection mismatch")
    w.require(runtime["backendEchoCount"] == 1 and runtime["streamline"] and runtime["targetGpuComplete"]
              and runtime["collectionBoundary"] == "case-process-instance-destroyed", "Backend qualification missing")
    w.require(w.digest(app / "shader-trace/layer-settings/vk_layer_settings.txt") == runtime["settingsFileSha256"], "Layer configuration changed")
    spirv = bytes.fromhex(variant["compilerSpirvHex"])
    w.require(hashlib.sha256(spirv).hexdigest() == variant["compilerSpirvSha256"] and b"NonSemantic.DebugPrintf" in spirv,
              "Missing/corrupt diagnostic compiler SPIR-V")
    w.require(variant["instrumentation"] == "Printf" and variant["shaderDebugMode"] == "Disabled"
              and variant["diskCache"] is False and variant["descriptorHeapMode"] == "mapped", "Unregistered compile policy")
    archive = directory.parent
    original_root = Path(w.load(archive / "Manifest.json")["projectRoot"])
    for path, checksum in variant["dependencyHashes"].items():
        relative = Path(path).relative_to(original_root).as_posix()
        w.require(w.digest(w.file_in(archive, "source/" + relative)) == checksum, "Compiled source differs from archive")
    w.require(runtime["input"]["baseline"] == snapshots["before"] and runtime["input"]["camera"] == report["camera"],
              "Compiled observation input differs from target checkpoint")
    defines = variant["macros"]
    for key, prefix in (("sessionToken", "SESSION"), ("runToken", "RUN"), ("dispatchToken", "DISPATCH")):
        w.require(int(defines[f"TRACE_{prefix}_LO"]) == dispatch[key] & 0xffffffff
                  and int(defines[f"TRACE_{prefix}_HI"]) == dispatch[key] >> 32, "Variant token mismatch")
    for axis, value in zip("XYZ", selected["group"]):
        w.require(int(defines["TRACE_GROUP_" + axis]) == value, "Compiled invocation mismatch")
    w.require(int(defines["TRACE_LOCAL"]) == selected["localIndex"] and defines["TRACE_MAX_RECORDS"] == "16"
              and defines["METALLIC_WORK_CONTROL_TRACE"] == "1", "Compiled watch budget mismatch")
    w.require(defines["TRACE_PREDICATE_ENABLED"] == ("1" if "predicate" in selected else "0")
              and int(defines["TRACE_TRIANGLE_ID"]) == selected.get("predicate", {}).get("value", 0), "Compiled predicate mismatch")
    baseline = w.snapshot_identity(app, snapshots["before"], case)
    diagnostic = normalized_diagnostic(snapshots["diagnostic"], snapshots["before"], phase, trace)
    w.require(w.snapshot_identity(app, diagnostic, case) == baseline
              and w.snapshot_identity(app, snapshots["after"], case) == baseline, "Workload/output restoration mismatch")
    restoration = runtime["restoration"]
    w.require(restoration["before"] == snapshots["before"] and restoration["diagnostic"] == snapshots["diagnostic"]
              and restoration["after"] == snapshots["after"], "Raw artifact and readback evidence disagree")
    w.require(runtime["productionBinding"] == next(i for i in snapshots["before"]["productionDispatches"] if i["phase"] == phase),
              "Target was not the production WorkControl binding")
    identity = {"snapshot": baseline, **{k: report[k] for k in ("camera", "renderExtent", "graph", "historyInvalidationPolicy")}}
    return {"name": row["name"], "process": process, "selection": selected, "identity": identity,
            "session": dispatch["session"], "phase": phase, "outcome": trace["outcome"],
            "fields": [r["fields"] for r in trace["records"]], "summary": trace["summary"],
            "schemaHash": trace["site"]["schemaHash"], "selectedScopeComplete": True,
            "variantSha256": variant["compilerSpirvSha256"]}


def assess(rows, acceptance):
    w.require(rows and e.independent_processes(rows), "Processes were not independent serial runs")
    observations = [r for r in rows if r["selection"] is not None]
    w.require(observations and len({r["session"] for r in observations}) == len(observations), "Reused observation session")
    w.require(all(r["identity"] == rows[0]["identity"] for r in rows), "Cross-process workload/input/output drift")
    w.require(all(r["selectedScopeComplete"] for r in observations), "Incomplete observation")
    if acceptance:
        w.require([r["name"] for r in rows] == [r["name"] for r in schedule(True, None)], "Incomplete acceptance schedule")
        for phase in ("early", "late"):
            group = [r for r in observations if r["name"].startswith(phase + "-")]
            w.require(len(group) == 3 and all(r["phase"] == phase and r["outcome"] == "Matched" for r in group), "Phase not reproduced three times")
            w.require(group[0]["fields"] and all(r["fields"] == group[0]["fields"] and r["schemaHash"] == group[0]["schemaHash"]
                                               for r in group), "Observed value bits/semantics drift")
        w.require(rows[-2]["outcome"] == "NoMatch" and rows[-1]["outcome"] == "SiteNotReached", "Negative controls not distinguished")
        w.require(rows[-2]["summary"]["siteEvaluationCount"] == 1 and rows[-1]["summary"]["siteEvaluationCount"] == 0,
                  "Negative controls lack coverage evidence")
    return {"status": "verified", "p2Acceptance": acceptance, "independentProcesses": len(rows),
            "performanceEligible": False, "correctness": "exact depth/visibility and frozen workload identity",
            "observations": [{k: r[k] for k in ("name", "phase", "outcome", "fields", "summary")} for r in observations]}


def execute(args):
    output, exe, cli = args.output.resolve(), args.exe.resolve(), args.cli.resolve()
    w.require(exe.is_file() and cli.is_file() and args.layer_path.is_dir() and 30 <= args.timeout <= 600, "Invalid executable/layer/run budget")
    output.mkdir(parents=True, exist_ok=False)
    case = recipe()
    acceptance = args.command == "acceptance"
    selected = selection(args.phase, args.group_x, args.local_index, args.triangle_id)
    runs = schedule(acceptance, selected)
    w.save(output / "Case.json", case)
    w.save(output / "Schedule.json", runs)
    manifest = {"protocol": PROTOCOL, "status": "running", "acceptance": acceptance,
                "performanceEligible": False, "projectRoot": str(ROOT), "runs": runs, "rows": []}
    lock = ROOT / "build/shader-experiment.lock"
    lock.parent.mkdir(exist_ok=True)
    lock_handle = None
    try:
        lock_handle = lock.open("x", encoding="utf-8")
        lock_handle.write(json.dumps({"output": str(output), "pid": os.getpid()})); lock_handle.flush()
        declared_assets = assets(args.assets)
        w.save(output / "Assets.json", declared_assets)
        sources = source_inventory()
        manifest["sources"] = sources
        for relative, checksum in sources.items():
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
            w.require(w.digest(target) == checksum, "Source changed during archival")
        binaries = [exe, cli, *sorted(exe.parent.glob("*.dll")), args.layer_path.resolve() / "VkLayer_khronos_validation.dll"]
        manifest["runtime"] = {str(p): w.digest(p) for p in binaries}
        manifest["cachePolicy"] = "production disk/pipeline caches reused; diagnostic Slang variant bypasses disk cache; fixed priming before every checkpoint"
        manifest["assetPolicy"] = "content SHA256 once, metadata checked before/after processes; external originals, no portable copies"
        manifest["inheritedEnvironment"] = {k: v for k, v in os.environ.items() if k.startswith(("VK_", "METALLIC_"))}
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        w.require(gpu.returncode == 0, "Cannot identify GPU")
        manifest["gpu"] = gpu.stdout.strip()
        w.save(output / "Manifest.json", manifest)
        for row in runs:
            directory = output / row["name"]
            directory.mkdir(); (directory / "app").mkdir(); (directory / "empty-layers").mkdir()
            config = w.engine_config(case)
            if row["selection"] is not None:
                config["shaderTrace"] = row["selection"]
            w.save(directory / "Config.json", config)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("METALLIC_", "VK_"))}
            env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"), METALLIC_FULL_ROAM_OUTPUT=str(directory / "app"),
                       METALLIC_FULL_ROAM_HIDDEN="1", METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
                       METALLIC_VK_PIPELINE_STATISTICS="0", METALLIC_SLANG_DESCRIPTOR_MODE="mapped",
                       VK_LAYER_PATH=str(args.layer_path.resolve()), VK_IMPLICIT_LAYER_PATH=str(directory / "empty-layers"),
                       VK_LAYER_SETTINGS_PATH=str(directory / "empty-layers"))
            if row["selection"] is not None:
                env["METALLIC_SHADER_TRACE"] = "1"
            w.save(directory / "Invocation.json", {"selection": row["selection"], "environment": {k: v for k, v in env.items() if k.startswith(("METALLIC_", "VK_"))}})
            w.asset_metadata_matches(declared_assets)
            d.run_process([str(exe), "--sample", case["sampleId"]], env, directory, args.timeout)
            result = inspect(directory, case, row, cli)
            w.save(directory / "Validation.json", result)
            manifest["rows"].append(result)
            w.save(output / "Manifest.json", manifest)
            w.require(source_inventory() == sources, "Source changed during GPU experiment")
            w.asset_metadata_matches(declared_assets)
            print(f"{row['name']}: {result.get('outcome', 'normal control')} verified", flush=True)
        w.require(all(w.digest(Path(p)) == h for p, h in manifest["runtime"].items()), "Runtime changed during experiment")
        verdict = assess(manifest["rows"], acceptance)
        w.save(output / "Result.json", verdict)
        manifest["status"] = "verified"
        return verdict
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["artifacts"] = e.artifact_hashes(output)
        w.save(output / "Manifest.json", manifest)
        if lock_handle is not None:
            lock_handle.close()
            lock.unlink()


def verify(directory, cli):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL and manifest["status"] == "verified", "Unfinished evidence")
    for relative, checksum in manifest["artifacts"].items():
        w.require(w.digest(w.file_in(directory, relative)) == checksum, f"Artifact changed: {relative}")
    case = w.validate_case(w.load(directory / "Case.json"))
    rows = [inspect(w.file_in(directory, row["name"] + "/Process.json").parent, case, row, cli) for row in manifest["runs"]]
    return assess(rows, manifest["acceptance"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "acceptance"):
        run = commands.add_parser(name)
        run.add_argument("--exe", type=Path, default=ROOT / "build-release/Source/MetallicGPUDrivenSample.exe")
        run.add_argument("--cli", type=Path, default=ROOT / "build-release/Source/metallicctl.exe")
        run.add_argument("--layer-path", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--assets", type=Path)
        run.add_argument("--timeout", type=int, default=240)
        run.add_argument("--phase", choices=("early", "late"), default="early")
        run.add_argument("--group-x", type=int, default=0)
        run.add_argument("--local-index", type=int, default=0)
        run.add_argument("--triangle-id", type=int)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--cli", type=Path, default=ROOT / "build-release/Source/metallicctl.exe")
    args = parser.parse_args()
    try:
        result = verify(args.directory.resolve(), args.cli.resolve()) if args.command == "verify" else execute(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
