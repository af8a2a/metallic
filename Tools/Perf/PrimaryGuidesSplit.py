"""Prepare and verify the opt-in primary-guides split experiment.

This tool does not launch GPU work or change installed shaders. Run each Cases.json
through RHIRendering.material_phase0_baseline with the same candidate executable.
Correctness retains frame 96, moving cameras, exact guide bytes and the existing
2e-5 absolute/relative HDR gate. --quick reduces resolution, not temporal coverage.
Native --group takes FUSED PRIMARY CONTINUATION report.json paths per repeat.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import RunShaderPipelineDriver as driver


CASE_IDS = ("Opaque1", "Mixed1", "Sky", "LegacyShadow", "ABeautifulGame")
PROPERTY = "splitPrimaryGuides"
STAGES = ("Primary guides", "Path continuation")
NATIVE_ROLES = {
    "fused": ("Features/PathTracing/OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain"),
    "primary": ("Features/PathTracing/OpenPBRPrimaryGuides", "openPbrPrimaryGuidesMain"),
    "continuation": ("Features/PathTracing/OpenPBRPathContinuation", "openPbrPathContinuationMain"),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, fixtures=None, *, quick=False, phase="correctness"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if fixtures is None:
        from PreparePathTraceQueries import prepare as prepare_queries
        fixtures = output / "source"
        prepare_queries(fixtures)
    fixtures = Path(fixtures).resolve()
    source = read(fixtures / "Correctness.json")
    chosen = [case for case in source["cases"] if case["id"] in CASE_IDS]
    require({case["id"] for case in chosen} == set(CASE_IDS), "Missing required query fixtures")
    for split in (False, True):
        folder = output / ("split" if split else "fused")
        folder.mkdir()
        cases = copy.deepcopy(source)
        cases["cases"] = copy.deepcopy(chosen)
        if phase == "performance":
            cases.update(frames=97, warmupFrames=32, timingFrames=64)
        for case in cases["cases"]:
            graph = read(fixtures / case["graph"])
            nodes = [n for n in graph["nodes"] if n["type"] == "ScenePathTracePass"]
            require(len(nodes) == 1, "Expected one PathTrace pass")
            props = nodes[0]["properties"]
            # The initial split supports exactly one camera sample and SHaRC off.
            props.update(samples=1, cacheMode="off")
            eligible = props.get("bsdf") == "openpbr" and props.get("exportDenoiserGuides") is True
            # LegacyShadow is an unchanged negative control, not a split workload.
            require(eligible or case["id"] == "LegacyShadow", "Unsupported split case")
            props[PROPERTY] = split and eligible
            case["requiredTiming"] = ["PathTrace", "Path continuation" if split and eligible else "Path trace shading"]
            if quick:
                case.update(width=64, height=64)
            elif phase == "performance":
                case.update(width=512, height=512)
            if phase == "performance":
                for key in ("frameCount", "captureFrames", "cameraFrames", "guideCameraStep", "useRenderView"):
                    case.pop(key, None)
            graph_path = folder / (case["id"] + ".json")
            save(graph_path, graph)
            case["graph"] = graph_path.name
        cases["assetFiles"] = list(source.get("assetFiles", [])) + [str(p) for p in folder.glob("*.json")]
        save(folder / "Cases.json", cases)
    return {arm: str(output / arm / "Cases.json") for arm in ("fused", "split")}


def normalize_graph(graph, *, split):
    graph = copy.deepcopy(graph)
    nodes = [n for n in graph["nodes"] if n["type"] == "ScenePathTracePass"]
    require(len(nodes) == 1, "Expected one PathTrace pass")
    props = nodes[0]["properties"]
    eligible = props.get("bsdf") == "openpbr" and props.get("exportDenoiserGuides") is True
    require(props.get("samples") == 1 and props.get("cacheMode") == "off", "Unsupported split settings")
    require(type(props.get(PROPERTY)) is bool and props[PROPERTY] == (split and eligible),
            "Wrong or missing splitPrimaryGuides policy")
    # This is the sole allowed behavioral graph difference.
    del props[PROPERTY]
    return graph, eligible


def compare(fused_path, split_path):
    import PathTraceQueries as queries
    import RunComputePipelineExperiments as experiment
    paths = [Path(fused_path).resolve(), Path(split_path).resolve()]
    reports = [read(path) for path in paths]
    ids = [case["id"] for case in reports[0]["cases"]]
    require(bool(ids) and len(ids) == len(set(ids)), "Missing or duplicate cases")
    require(ids == [c["id"] for c in reports[1]["cases"]], "Missing or reordered cases")
    require(reports[0].get("validation") == reports[1].get("validation"), "Validation policy changed")
    timing = []
    for left, right in zip(reports[0]["cases"], reports[1]["cases"]):
        # Runtime timings/resource revisions are observations, not workload inputs.
        # Reject every other metadata difference, including future case controls.
        observations = {"frames", "warmup", "environmentTransitions", "resolvedGraph", "requiredTiming"}
        require({k: v for k, v in left.items() if k not in observations} ==
                {k: v for k, v in right.items() if k not in observations}, "Workload metadata changed")
        a, eligible = normalize_graph(left["resolvedGraph"], split=False)
        b, _ = normalize_graph(right["resolvedGraph"], split=True)
        require(a == b, "Workload graph changed beyond splitPrimaryGuides")
        for case, split in ((left, False), (right, True)):
            require(case.get("requiredTiming") == ["PathTrace", "Path continuation" if split and eligible else "Path trace shading"],
                    "Unexpected requiredTiming policy")
            require(case.get("validHDR") is True, "Invalid HDR")
            require(bool(case.get("frames")), "Missing GPU timing frames")
            for frame in case["frames"]:
                node = next((n for n in frame["nodes"] if n["name"] == "PathTrace"), None)
                require(node is not None, "Missing PathTrace GPU scope")
                names = [s["name"] for s in node["sections"]]
                require(all(names.count(stage) == (1 if split and eligible else 0) for stage in STAGES),
                        "Split dispatch scopes do not match requested policy")
                for section in node["sections"]:
                    if section["name"] in STAGES:
                        require(isinstance(section.get("gpuMs"), (int, float)) and math.isfinite(section["gpuMs"])
                                and section["gpuMs"] > 0, "Invalid split GPU timestamp")
        require([r["frame"] for r in left["frames"]] == [r["frame"] for r in right["frames"]],
                "Timing frame window changed")
        for arm, case in zip(("fused", "split"), (left, right)):
            guide_frames = [c["frame"] for c in case.get("captures", []) if c["output"] != case["output"]]
            frame_count = case.get("frameCount", min(guide_frames) if guide_frames else
                                   max(case.get("captureFrames", [case["frames"][-1]["frame"]])) + 1)
            configuration = dict(frames=frame_count, warmupFrames=case["frames"][0]["frame"],
                                 timingFrames=len(case["frames"]), cases=[case])
            metrics = experiment.gpu_timings(dict(cases=[case]), configuration)[case["id"]]
            if arm == "split" and eligible:
                metrics["split_sections_ms"] = {name: driver.timings([section["gpuMs"]
                    for frame in case["frames"] for node in frame["nodes"] if node["name"] == "PathTrace"
                    for section in node["sections"] if section["name"] == name]) for name in STAGES}
            timing.append(dict(case=case["id"], arm=arm, metrics=metrics))
        left["resolvedGraph"], right["resolvedGraph"] = a, b
    # Reuse the established exact-guide/HDR comparator without modifying source
    # reports or weakening its checks. Copies are temporary and never evidence.
    with tempfile.TemporaryDirectory(prefix="primary-guides-compare-") as temporary:
        normalized = []
        for index, (path, report) in enumerate(zip(paths, reports)):
            folder = Path(temporary) / str(index)
            folder.mkdir()
            for case in report["cases"]:
                files = [case["image"], *(c["file"] for c in case.get("captures", []))]
                for filename in files:
                    require(Path(filename).name == filename, "Readback must have a simple file name")
                    shutil.copyfile(path.parent / filename, folder / filename)
            normalized.append(folder / "MaterialBaseline.json")
            save(normalized[-1], report)
        differences = queries.compare_reports(*normalized, ids)
    return dict(reports=[dict(path=str(p), sha256=digest(p)) for p in paths], differences=differences,
                timings=timing, scope="One process per arm; pass/graph GPU time, not end-to-end FPS. No speedup acceptance.")


def validate_native_role(sample, arm):
    expected = NATIVE_ROLES[arm]
    request = sample.get("request", {})
    require((request.get("module"), request.get("entry")) == expected, "Wrong Native kernel role: " + arm)
    if sample.get("shader_source") == "recipe_spirv":
        stages = sample.get("shader_stages", [])
        require(sample.get("pipeline_type") == "compute" and len(stages) == 1,
                "Native role requires exactly one compute recipe stage")
        stage = stages[0]
        require(stage.get("stage") == "compute" and stage.get("source_identity_ambiguous") is False and
                (stage.get("module"), stage.get("source_entry")) == expected,
                "Wrong or ambiguous Native recipe kernel role: " + arm)
        return "recipe module/source-entry verified; macro defines are not recorded in replay reports"
    defines = request.get("defines", [])
    require(isinstance(defines, list) and all(isinstance(d, list) and len(d) == 2 for d in defines),
            "Invalid Native request defines")
    single = [d[1] for d in defines if d[0] == "METALLIC_PT_SINGLE_SAMPLE"]
    require(single == ["1"], "Native kernel role requires unambiguous Single=1 macro")
    return "catalog module/entry and Single=1 macro verified"


def aggregate_native(groups):
    require(bool(groups), "At least one native repeat is required")
    identities, device, common, rows, used = {}, None, None, [], set()
    for repeat, group in enumerate(groups):
        require(len(group) == 3, "Each repeat requires fused, primary and continuation reports")
        durations, evidence = {}, []
        for arm, path in zip(("fused", "primary", "continuation"), group):
            path = Path(path).resolve()
            require(path not in used, "Cannot reuse a native process report")
            used.add(path)
            report = read(path)
            sample = driver.single_sample(report, native=True)
            role_evidence = validate_native_role(sample, arm)
            identity = driver.shader_identity(report, sample)
            identity.update(device_spirv_fnv1a64=sample.get("device_spirv_fnv1a64"),
                            device_shader_content_hash=sample.get("device_shader_content_hash"),
                            pso_hash=sample["pipeline"].get("pso_hash"))
            if arm in identities:
                require(identity == identities[arm], "Native input identity changed between repeats")
            identities[arm] = identity
            if device is None:
                device = report.get("device")
            require(isinstance(device, dict) and bool(device), "Missing device identity")
            driver.verify_measurement(report, driver.shader_identity(report, sample), device, disabled=True)
            config = report["configuration"]
            require(config.get("compute_backend", "pipeline") == "pipeline" and
                    not config.get("disable_driver_optimization", False), "Native creation policy changed")
            policy = {key: identity.get(key) for key in ("build_configuration", "slang_build", "shader_target",
                      "debug_mode", "validation", "slang_optimization", "descriptor_mode", "resource_heap_stride", "sampler_heap_stride")}
            policy["postprocess"] = sample.get("effective_spirv_postprocess", "none")
            if common is None:
                common = policy
            require(policy == common, "Native compiler/descriptor policy changed across kernels")
            pipeline = sample["pipeline"]
            require(not any(pipeline.get(key, False) for key in ("native_cache_hit", "metadata_hit",
                    "native_object_reused", "native_binary_cache_hit", "native_binary_driver_rejected")), "Native cache/reuse contaminated sample")
            require(not any(report.get("environment", {}).get(key) for key in
                    ("METALLIC_VK_PIPELINE_STATISTICS", "METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY")),
                    "Diagnostic native sample cannot enter ordinary timings")
            durations[arm] = pipeline["native_create_ms"]
            evidence.append(dict(arm=arm, path=str(path), sha256=digest(path), role_validation=role_evidence))
        require(len({json.dumps(identities[arm], sort_keys=True) for arm in identities}) == 3,
                "Expected three distinct shader/PSO identities")
        rows.append(dict(repeat=repeat, **durations, split_max_ms=max(durations["primary"], durations["continuation"]),
                         split_sum_ms=durations["primary"] + durations["continuation"], reports=evidence))
    return dict(device=device, policy=common, identities=identities, repeats=rows,
                summary={key: driver.timings([row[key] for row in rows]) for key in
                         ("fused", "primary", "continuation", "split_max_ms", "split_sum_ms")},
                scope="Per-repeat native creation max and sum. Driver cache control disabled; hidden compiler/OS state is not guaranteed cold.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("output", type=Path)
    prep.add_argument("--fixtures", type=Path)
    prep.add_argument("--quick", action="store_true")
    prep.add_argument("--phase", choices=("correctness", "performance"), default="correctness")
    comparison = commands.add_parser("compare")
    comparison.add_argument("fused", type=Path)
    comparison.add_argument("split", type=Path)
    native = commands.add_parser("native")
    native.add_argument("--group", nargs=3, type=Path, action="append", required=True,
                        metavar=("FUSED", "PRIMARY", "CONTINUATION"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.fixtures, quick=args.quick, phase=args.phase)
    elif args.command == "compare":
        result = compare(args.fused, args.split)
    else:
        result = aggregate_native(args.group)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
