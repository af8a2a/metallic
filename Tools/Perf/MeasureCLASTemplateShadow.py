"""Fixed three-arm production CLAS experiment with Shadows and CLAS retention gates."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import sys

import MeasureCLASTessellation as base

ROOT = base.ROOT
LEGACY_PROTOCOL = "metallic-clas-tessellation-ab-v2"
PROTOCOL = "metallic-clas-tessellation-ab-v3"
ARMS = {"A": "triangles", "B": "templates", "C": "candidate"}
MODES = {"A": "triangles", "B": "templates", "C": "templates"}
BLOCK = "ABCCBA"
PAIRS = {"B/A": ("A", "B"), "C/A": ("A", "C"), "C/B": ("B", "C")}
METRICS = (*base.METRICS, "shadowTraceMs", "shadowCopyMs")
STRATEGY_KEYS = ("rtTessellationTemplateShape", "rtTessellationInstantiationFlags")
TEMPLATE_SHAPES = ("right", "equilateral")
INSTANTIATION_FLAGS = ("fastTrace", "none")
DEFAULT_STRATEGY = dict(zip(STRATEGY_KEYS, ("right", "fastTrace")))
CANDIDATES = {"C1": dict(zip(STRATEGY_KEYS, ("right", "none"))),
              "C2": dict(zip(STRATEGY_KEYS, ("equilateral", "fastTrace")))}
BOUNDS_KEY = "rtTessellationTemplateBounds"
BOUNDED_DEFAULT = {**DEFAULT_STRATEGY, BOUNDS_KEY: "none"}
BOUNDED_CANDIDATES = {"C3": {**DEFAULT_STRATEGY, BOUNDS_KEY: "geometry"}}
MODULES = ("Features/GPUDriven/StreamCLASTessellation", "Features/Lighting/ScreenSpaceShadows")
POLICY = {"statistic": "abccba-block-log-ratio", "confidence": .95, "imagePolicy": "exact-bytes",
          "gates": [{"pair": "C/A", "metric": "shadowsMs", "minimum": -.02, "strict": False},
                    {"pair": "C/A", "metric": "graphMs", "minimum": -.02, "strict": False},
                    {"pair": "C/A", "metric": "clasMs", "minimum": .50, "strict": False},
                    {"pair": "C/B", "metric": "shadowsMs", "minimum": 0, "strict": True},
                    {"pair": "C/B", "metric": "clasMs", "minimum": -.05, "strict": False}]}
read, save, digest, require = base.read, base.save, base.digest, base.require


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def properties(graph):
    nodes = [n for n in graph["nodes"] if n["name"] == "VBuffer"]
    require(len(nodes) == 1, "Unique VBuffer producer required")
    return nodes[0]["properties"]


def strategy_schema(protocol):
    require(protocol in (LEGACY_PROTOCOL, PROTOCOL), "Unknown strategy protocol")
    if protocol == LEGACY_PROTOCOL:
        return STRATEGY_KEYS, DEFAULT_STRATEGY, CANDIDATES
    return (*STRATEGY_KEYS, BOUNDS_KEY), BOUNDED_DEFAULT, BOUNDED_CANDIDATES


def normalize_graph(graph, protocol=PROTOCOL):
    result = copy.deepcopy(graph)
    p = properties(result)
    for key in ("rtTessellationBuildMode", *strategy_schema(protocol)[0]):
        p.pop(key, None)
    return result


def validate_strategy(strategy, protocol=PROTOCOL):
    require(set(strategy) == set(strategy_schema(protocol)[0]) and
            strategy[STRATEGY_KEYS[0]] in TEMPLATE_SHAPES and
            strategy[STRATEGY_KEYS[1]] in INSTANTIATION_FLAGS and
            (protocol == LEGACY_PROTOCOL or strategy[BOUNDS_KEY] in ("none", "geometry")), "Unregistered template strategy")


def prepare(args):
    candidate = {STRATEGY_KEYS[0]: args.template_shape, STRATEGY_KEYS[1]: args.instantiation_flags,
                 BOUNDS_KEY: args.template_bounds}
    validate_strategy(candidate)
    require(candidate == BOUNDED_CANDIDATES["C3"], "Only C3 right/fastTrace/geometry is registered in v3")
    require(args.native_sizes, "v3 requires native size and actual template-bound diagnostics")
    # Preserve the production v1 scene/camera/native-resolution fixture builder.
    base.prepare(args)
    output = args.output.resolve()
    shutil.copytree(output / "templates", output / "candidate")
    for arm, directory in ARMS.items():
        graph = read(output / directory / "Graph.json")
        properties(graph).update(candidate if arm == "C" else BOUNDED_DEFAULT)
        properties(graph)["benchmarkCLASBuildSizes"] = False
        save(output / directory / "Graph.json", graph)
        config = read(output / directory / "Cases.json")
        diagnostic = copy.deepcopy(config)
        diagnostic["cases"][0]["graph"] = "DiagnosticGraph.json"
        diagnostic_graph = copy.deepcopy(graph)
        if args.native_sizes:
            diagnostic["cases"][0]["captureTessellationNativeSizes"] = True
            properties(diagnostic_graph)["benchmarkCLASBuildSizes"] = True
        save(output / directory / "DiagnosticGraph.json", diagnostic_graph)
        save(output / directory / "DiagnosticCases.json", diagnostic)
    fixture = read(output / "Fixture.json")
    fixture.update(protocol=PROTOCOL, candidate=candidate, baselineStrategy=BOUNDED_DEFAULT,
                   candidateId="C3",
                   nativeSizeDiagnostic=args.native_sizes, blockOrder=BLOCK)
    save(output / "Fixture.json", fixture)
    fixture_configs(output)


def fixture_configs(directory):
    fixture = read(directory / "Fixture.json")
    protocol = fixture["protocol"]
    keys, default, candidates = strategy_schema(protocol)
    require(fixture["blockOrder"] == BLOCK, "Unknown three-arm fixture protocol")
    require(protocol == LEGACY_PROTOCOL or fixture["nativeSizeDiagnostic"] is True, "v3 requires actual bound diagnostics")
    validate_strategy(fixture["candidate"], protocol)
    require(fixture["baselineStrategy"] == default and
            candidates.get(fixture["candidateId"]) == fixture["candidate"],
            "Baseline or declared candidate strategy changed")
    configs = {arm: read(directory / name / "Cases.json") for arm, name in ARMS.items()}
    require(configs["A"] == configs["B"] == configs["C"], "Arms differ in scene, settings or fixed frame sequence")
    config = configs["A"]
    require(len(config["cases"]) == 1 and config["warmupFrames"] >= 4 and config["timingFrames"] > 0 and
            config["frames"] == config["warmupFrames"] + config["timingFrames"] + 1, "Invalid fixed production frame windows")
    spec = config["cases"][0]
    require("captureTessellationNativeSizes" not in spec, "Native size diagnostics leaked into ordinary timing")
    graphs = {arm: read(directory / name / spec["graph"]) for arm, name in ARMS.items()}
    require(normalize_graph(graphs["A"], protocol) == normalize_graph(graphs["B"], protocol) == normalize_graph(graphs["C"], protocol),
            "Graph differs beyond the predeclared CLAS construction strategy")
    diagnostics, diagnostic_graphs = {}, {}
    for arm, graph in graphs.items():
        require(graph["view"]["renderResolution"] == {"width": spec["width"], "height": spec["height"], "adaptive": False},
                "Declared native render resolution changed")
        p = properties(graph)
        strategy = {key: p[key] for key in keys}
        require(p["rtTessellationBuildMode"] == MODES[arm] and p["rtTessellation"] and
                strategy == (fixture["candidate"] if arm == "C" else default) and
                p["benchmarkCLASBuildSizes"] is False, "Graph does not select its declared ordinary arm")
        diagnostic = read(directory / ARMS[arm] / "DiagnosticCases.json")
        expected = copy.deepcopy(config)
        expected["cases"][0]["graph"] = "DiagnosticGraph.json"
        if fixture["nativeSizeDiagnostic"]:
            expected["cases"][0]["captureTessellationNativeSizes"] = True
        require(diagnostic == expected, "Diagnostic case differs beyond native size collection")
        diagnostics[arm] = diagnostic
        diagnostic_graph = read(directory / ARMS[arm] / "DiagnosticGraph.json")
        expected_graph = copy.deepcopy(graph)
        properties(expected_graph)["benchmarkCLASBuildSizes"] = fixture["nativeSizeDiagnostic"]
        require(diagnostic_graph == expected_graph, "Diagnostic graph differs beyond native size recording")
        diagnostic_graphs[arm] = diagnostic_graph
    return config, diagnostics, graphs, diagnostic_graphs


def identity(exe, fixtures):
    value = base.identity(exe, fixtures)
    value["additional"][str(Path(__file__).resolve())] = digest(Path(__file__))
    value["additional"][str(Path(base.__file__).resolve())] = digest(Path(base.__file__))
    return value


def pair_rows(rows, left, right):
    require(len(rows) >= 12 and len(rows) % 6 == 0 and [r["arm"] for r in rows] == list(BLOCK) * (len(rows) // 6),
            "Complete ABCCBA blocks are required")
    require((left, right) in PAIRS.values(), "Unregistered contrast")
    result = []
    for offset in range(0, len(rows), 6):
        selected = [r for r in rows[offset:offset + 6] if r["arm"] in (left, right)]
        require([r["arm"] for r in selected] == [left, right, right, left], "Contrast is not symmetric ABBA")
        result.extend({**r, "sourceArm": r["arm"], "arm": "A" if r["arm"] == left else "B"} for r in selected)
    return result


def assess(rows, stage):
    require(stage in ("discovery", "confirmation"), "Unknown experiment stage")
    require(base.experiment.independent_processes(rows), "Invalid or overlapping independent processes")
    contrasts = {name: {metric: base.experiment.block_gain(pair_rows(rows, *arms), metric) for metric in METRICS}
                 for name, arms in PAIRS.items()}
    gates = []
    for gate in POLICY["gates"]:
        gain = contrasts[gate["pair"]][gate["metric"]]
        passes = lambda value: value > gate["minimum"] if gate["strict"] else value >= gate["minimum"]
        gates.append({**gate, "pointPass": passes(gain["estimate"]), "intervalPass": passes(gain["lower95"]),
                      "intervalFail": not passes(gain["upper95"])})
    status, effect, action = "inconclusive", "uncertain", "retain_candidate"
    if any(g["intervalFail"] for g in gates):
        status, effect, action = "reject", "failed_target", "reject_candidate"
    elif stage == "discovery" and all(g["pointPass"] for g in gates):
        effect, action = "promising", "run_confirmation"
    elif stage == "confirmation" and all(g["intervalPass"] for g in gates):
        status, effect, action = "accept", "qualified_shadow_and_clas_gain", "accept_candidate"
    spread = {arm: {metric: (max(values) - min(values)) / statistics.median(values)
                   for metric in METRICS for values in [[r["result"]["timings"][metric]["median"] for r in rows if r["arm"] == arm]]}
              for arm in ARMS}
    return {"decision": status, "stage": stage, "evidenceValidity": "valid", "effectAssessment": effect,
            "nextAction": action, "contrasts": contrasts, "gates": gates, "withinArmRelativeRange": spread,
            "noiseWarning": any(v > .1 for arm in spread.values() for v in arm.values()),
            "statisticalUnit": "one complete six-process ABCCBA block; contrasts share samples and are not independent replications"}


def native_sizes(directory, case):
    import numpy as np
    receipt = case["nativeCLASSizeReceipt"]
    path = directory / receipt["file"]
    require(path.parent == directory and path.is_file() and receipt["count"] == case["tessellationCounts"][0] and
            path.stat().st_size == receipt["bytes"] == receipt["count"] * 4, "Invalid native CLAS size readback")
    sizes = np.fromfile(path, dtype="<u4")
    require(len(sizes) > 0 and (sizes > 0).all(), "Native CLAS size receipt contains empty work")
    require(receipt["totalBytes"] == int(sizes.sum(dtype=np.uint64)) and receipt["minBytes"] == int(sizes.min()) and
            receipt["maxBytes"] == int(sizes.max()), "Native size summary differs from actual GPU readback")
    return {"file": receipt["file"], "sha256": digest(path), "count": len(sizes),
            "minimum": int(sizes.min()), "maximum": int(sizes.max()), "totalBytes": int(sizes.sum(dtype=np.uint64)),
            "median": float(np.median(sizes)), "scope": "native CLAS destination sizes; separate untimed diagnostic publication"}


def bounded_path_evidence(directory, case, identity, graph, arm, diagnostic):
    mode = properties(graph)[BOUNDS_KEY]
    require(identity["templateBoundsMode"] == mode, "Actual template bounds mode differs from graph")
    count, observed = case["templateBoundFallbacks"], case["templateBoundFallbackObserved"]
    require(type(count) is int and count >= 0 and type(observed) is bool, "Invalid GPU bound fallback receipt")
    generation = case["templatePoolGeneration"]
    require(type(generation) is int and generation >= 0 and generation == identity["templatePoolGeneration"],
            "Final bound fallback receipt came from a different template allocation")
    for frame in [*case["warmup"], *case["frames"]]:
        profiles = [p for p in frame["streaming"] if p["passName"] == "VBuffer"]
        require(len(profiles) == 1, "Missing unique bound-generation producer")
        profile = profiles[0]["tessellationIdentity"]
        if isinstance(profile, str):
            profile = json.loads(profile)
        require(profile["templatePoolGeneration"] == generation,
                "Template pool replaced during warmup/timing; sticky fallback history does not cover the window")
    require(identity["nativeTemplateCount"] == (0 if arm == "A" else 512 if arm == "B" else 1024),
            "Actual template population differs from registered arm")
    require(generation == 0 if arm == "A" else generation > 0, "Invalid template pool generation")
    raw_receipt = case.get("nativeCLASBoundsReceipt")
    if arm != "C":
        require(mode == "none" and identity["templateBounds"] is None and not identity["boundedTemplatesReady"] and
                raw_receipt is None, "Unbounded baseline unexpectedly published bounded templates")
    else:
        require(mode == "geometry" and identity["boundedTemplatesReady"] is True and count == 0 and observed is False,
                "C3 did not exclusively use bounded templates throughout warmup and timing")
        bounds = identity["templateBounds"]
        require(isinstance(bounds, list) and len(bounds) == 6 and all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
                and all(bounds[i + 3] >= bounds[i] for i in range(3)), "Invalid actual template bounds")
        if diagnostic:
            require(isinstance(raw_receipt, dict), "Missing actual template-bound buffer readback")
            path = directory / raw_receipt["file"]
            require(path.parent == directory and path.is_file() and path.stat().st_size == raw_receipt["bytes"] == 24,
                    "Invalid native template-bound byte range")
            data = path.read_bytes()
            require(len(raw_receipt["values"]) == 6 and struct.pack("<6f", *bounds) == data ==
                    struct.pack("<6f", *raw_receipt["values"]), "Native bound buffer disagrees with production identity")
        else:
            require(raw_receipt is None, "Native bound readback leaked into ordinary timing")
    value = {"fallbacksAtFinalFrame": count, "fallbackObservedSinceAllocation": observed,
             "templatePoolGeneration": generation, "coveredFrames": len(case["warmup"]) + len(case["frames"]),
             "scope": "GPU sticky fallback bit since the unchanged allocation, read once after timing"}
    if raw_receipt is not None:
        value["nativeBounds"] = {"sha256": digest(directory / raw_receipt["file"]), "values": raw_receipt["values"], "bytes": 24}
    return value


def analyze_run(directory, config, graph, arm, diagnostic, protocol=LEGACY_PROTOCOL):
    value = base.analyze_run(directory, config, graph, arm, expected_mode=MODES[arm])
    p, receipt = properties(graph), value["identity"]
    require(receipt["templateShape"] == p[STRATEGY_KEYS[0]] and receipt["instantiationFlags"] == p[STRATEGY_KEYS[1]],
            "Actual production template construction strategy differs from graph")
    case = read(directory / "MaterialBaseline.json")["cases"][0]
    for metric, name in (("shadowTraceMs", "Shadow trace"), ("shadowCopyMs", "Publish shadow")):
        values = []
        for frame in case["frames"]:
            shadow = next(n for n in frame["nodes"] if n["name"] == "Shadows")
            matches = [s["gpuMs"] for s in shadow["sections"] if s["name"] == name]
            require(len(matches) == 1, "Missing unique production Shadow GPU scope: " + name)
            values.append(base.finite_time(matches[0], name))
        value["timings"][metric] = {"median": statistics.median(values), "frames": values}
    collect_sizes = config["cases"][0].get("captureTessellationNativeSizes", False)
    require(diagnostic or not collect_sizes, "Native size capture is diagnostic only")
    require(receipt["diagnosticBuildSizes"] is collect_sizes and
            p["benchmarkCLASBuildSizes"] is collect_sizes, "Actual native size instrumentation differs from declared process")
    require(("nativeCLASSizeReceipt" in case) == collect_sizes, "Native size diagnostic receipt does not match its request")
    if collect_sizes:
        value["nativeCLASSizes"] = native_sizes(directory, case)
    if protocol == PROTOCOL:
        value["boundPathEvidence"] = bounded_path_evidence(directory, case, receipt, graph, arm, diagnostic)
    return value


def shader_receipts(directory, result):
    receipts = base.receipts(directory, modules=MODULES)
    production = result["identity"]
    require(any(r["module"] == production["module"] and r["entry"] == production["entry"] and
                r["sha256"] == production["spirvSha256"] for r in receipts), "Bound tessellation SPIR-V differs from production identity")
    shadow = [r for r in receipts if r["module"] == MODULES[1]]
    require(shadow and all(r["entry"] == "rayTracedShadowsMain" and r["stage"] == "compute" for r in shadow),
            "Unexpected bound shadow kernel")
    require(len({r["sha256"] for r in shadow}) == 1, "Ambiguous production shadow shader")
    require(len({canonical(r["device"]) for r in receipts}) == 1, "Diagnostic shaders came from different devices")
    return receipts


def validate_results(rows, diagnostics, protocol=LEGACY_PROTOCOL):
    require([r["arm"] for r in diagnostics] == list("ABC"), "Three diagnostic baselines required")
    reference = diagnostics[0]["result"]
    device = diagnostics[0]["recipes"][0]["device"]
    shadow_hashes = {r["sha256"] for row in diagnostics for r in row["recipes"] if r["module"] == MODULES[1]}
    require(len(shadow_hashes) == 1, "Shadow SPIR-V changed across strategies")
    for row in [*diagnostics, *rows]:
        value = row["result"]
        require(value["inputs"] == reference["inputs"], "Published geometry/material/count identity changed")
        require(value["outputs"] == reference["outputs"], "Exact HDR/depth/shadow mismatch")
        def ordinary_identity(identity):
            result = dict(identity)
            del result["diagnosticBuildSizes"]
            if protocol == PROTOCOL:
                # Allocation generations establish a single process's sticky
                # history; they are not portable identities across processes.
                del result["templatePoolGeneration"]
            return result
        require(ordinary_identity(value["identity"]) == ordinary_identity(
                next(d["result"]["identity"] for d in diagnostics if d["arm"] == row["arm"])),
                "Within-arm backend or tessellation SPIR-V changed")
        if row["stage"] == "diagnostic":
            require(all(r["device"] == device for r in row["recipes"]), "Device changed across diagnostic arms")


def bound_size_diagnostic(diagnostics):
    values = {row["arm"]: row["result"]["nativeCLASSizes"] for row in diagnostics}
    require(set(values) == set(ARMS) and len({v["count"] for v in values.values()}) == 1,
            "Bounded native-size comparison requires the same actual live CLAS count")
    baseline, candidate = values["B"]["totalBytes"], values["C"]["totalBytes"]
    return {"scope": "untimed native live CLAS output; separate from retained template pool capacity",
            "requiredForPerformanceAcceptance": False, "liveCLASCount": values["C"]["count"],
            "originalTemplateBytes": baseline, "boundedTemplateBytes": candidate,
            "reductionFraction": 1 - candidate / baseline, "smallerThanOriginalTemplates": candidate < baseline,
            "classification": "reduced" if candidate < baseline else "unchanged" if candidate == baseline else "larger",
            "arms": {arm: {k: value[k] for k in ("totalBytes", "minimum", "maximum", "median")} for arm, value in values.items()}}


def run(args):
    exe, fixtures, output = args.exe.resolve(), args.fixtures.resolve(), args.output.resolve()
    config, diagnostic_configs, graphs, diagnostic_graphs = fixture_configs(fixtures)
    require(read(fixtures / "Fixture.json")["protocol"] == PROTOCOL, "Only the bounded C3 candidate can start a new experiment")
    require(exe.is_file() and not output.exists(), "Existing executable and fresh evidence directory required")
    require("CMAKE_BUILD_TYPE:STRING=Release" in (exe.parent.parent / "CMakeCache.txt").read_text(), "Existing Release build required")
    locks = [ROOT / "build/shader-experiment.lock", ROOT / "build/material-phase0.lock"]
    require(not any(p.exists() for p in locks), "Another cooperative GPU experiment owns the lock")
    locks[0].parent.mkdir(exist_ok=True)
    with locks[0].open("x") as lock:
        json.dump({"pid": os.getpid(), "output": str(output), "protocol": PROTOCOL}, lock)
    output.mkdir(parents=True)
    manifest = {"protocol": PROTOCOL, "policy": POLICY, "status": "running", "runs": [], "blockOrder": BLOCK,
                "budgets": {"discovery": args.blocks, "confirmation": args.confirmation_blocks},
                "diagnosticsOnly": args.diagnostics_only, "timeoutSeconds": args.timeout, "outputDirectory": str(output),
                "scope": read(fixtures / "Fixture.json")["scope"],
                "cachePolicy": "Existing disk caches retained; fixed warmup; compilation and template initialization excluded",
                "gpuExclusivity": "Cooperative lock and whole-device telemetry; not system GPU exclusivity"}
    try:
        shutil.copytree(fixtures, output / "fixtures")
        initial = identity(exe, fixtures)
        save(output / "Identity.json", initial)
        save(output / "Experiment.json", manifest)
        (output / "GPU.txt").write_text(base.experiment.gpu_identity(), encoding="utf-8")
        for path in (Path(__file__), Path(base.__file__), Path(base.experiment.__file__), Path(base.baseline.__file__)):
            shutil.copy2(path, output / path.name)
        diagnostics = []
        for stage, arms in [("diagnostic", "ABC"), ("discovery", BLOCK * args.blocks), ("confirmation", BLOCK * args.confirmation_blocks)]:
            if stage != "diagnostic" and (args.diagnostics_only or stage == "confirmation" and
                    manifest["discovery"]["nextAction"] != "run_confirmation"):
                break
            stage_rows = []
            for arm in arms:
                require(identity(exe, fixtures) == initial, "Source, asset, fixture or executable changed")
                diagnostic = stage == "diagnostic"
                row = base.collect(exe, fixtures, output, arm, stage, args.timeout,
                                   variant_directory=ARMS[arm], case_filename="DiagnosticCases.json" if diagnostic else "Cases.json")
                row["inputIdentitySha256"] = canonical(initial)
                row["result"] = analyze_run(output / row["directory"], diagnostic_configs[arm] if diagnostic else config,
                                            diagnostic_graphs[arm] if diagnostic else graphs[arm], arm, diagnostic, PROTOCOL)
                if diagnostic:
                    row["recipes"] = shader_receipts(output / row["directory"], row["result"])
                    diagnostics.append(row)
                stage_rows.append(row)
                manifest["runs"].append(row)
                save(output / "Experiment.json", manifest)
                if not diagnostic:
                    validate_results([row], diagnostics, PROTOCOL)
            if stage == "diagnostic":
                validate_results([], diagnostics, PROTOCOL)
                manifest["boundSizeDiagnostic"] = bound_size_diagnostic(diagnostics)
                save(output / "Experiment.json", manifest)
            else:
                manifest[stage] = assess(stage_rows, stage)
                save(output / "Decision.json", manifest[stage])
                save(output / "Experiment.json", manifest)
        require(identity(exe, fixtures) == initial, "Input identity changed during experiment")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save(output / "Experiment.json", manifest)
        base.baseline.seal(output)
        locks[0].unlink()
    print(json.dumps(verify(output), indent=2))


def verify(directory):
    directory = directory.resolve()
    manifest = read(directory / "Experiment.json")
    if manifest["protocol"] == base.PROTOCOL:
        return base.verify(directory)
    require(read(directory / "Manifest.json") == {p.relative_to(directory).as_posix(): digest(p)
            for p in directory.rglob("*") if p.is_file() and p.name != "Manifest.json"}, "Evidence file set/hash changed")
    protocol = manifest["protocol"]
    require(protocol in (LEGACY_PROTOCOL, PROTOCOL) and manifest["policy"] == POLICY and manifest["status"] == "complete" and
            manifest["blockOrder"] == BLOCK, "Unsupported/incomplete three-arm experiment")
    require(set(manifest["budgets"]) == {"discovery", "confirmation"} and
            all(type(n) is int and 2 <= n <= 10 for n in manifest["budgets"].values()), "Invalid fixed stage budgets")
    config, diagnostic_configs, graphs, diagnostic_graphs = fixture_configs(directory / "fixtures")
    require(read(directory / "fixtures/Fixture.json")["protocol"] == protocol, "Experiment and fixture protocols disagree")
    initial = read(directory / "Identity.json")
    require(initial["fixtureFiles"] == {p.relative_to(directory / "fixtures").as_posix(): digest(p)
            for p in (directory / "fixtures").rglob("*.json")}, "Archived fixture differs from captured input identity")
    schedule = [("diagnostic", arm) for arm in "ABC"]
    if not manifest["diagnosticsOnly"]:
        schedule += [("discovery", arm) for arm in BLOCK * manifest["budgets"]["discovery"]]
        if "confirmation" in manifest:
            schedule += [("confirmation", arm) for arm in BLOCK * manifest["budgets"]["confirmation"]]
    rows = manifest["runs"]
    require([(r["stage"], r["arm"]) for r in rows] == schedule and
            len({r["directory"] for r in rows}) == len(rows), "Unexpected stage order or repeated process evidence")
    for row in rows:
        root = directory / row["directory"]
        require(root.parent == directory, "Invalid evidence path")
        process = read(root / "Process.json")
        require(all(process[k] == row[k] for k in ("process", "environment", "directory", "stage", "arm")) and
                row["process"]["exitCode"] == 0 and row["inputIdentitySha256"] == canonical(initial), "Invalid process/input identity receipt")
        diagnostic = row["stage"] == "diagnostic"
        case_name = "DiagnosticCases.json" if diagnostic else "Cases.json"
        env = row["environment"]
        expected_env = {"METALLIC_NSIGHT_GRAPHICS_CAPTURE": "0", "METALLIC_SHADER_CAPTURE_SYMBOLS": "0",
                        "METALLIC_MATERIAL_BASELINE_CASES": str(Path(initial["fixtureRoot"]) / ARMS[row["arm"]] / case_name)}
        recorded_root = Path(manifest["outputDirectory"]) / row["directory"]
        if diagnostic:
            expected_env["METALLIC_PIPELINE_RECIPE_DIRECTORY"] = str(recorded_root / "recipes")
        require({k: v for k, v in env.items() if k.startswith("METALLIC_")} == expected_env and
                not env.get("VK_INSTANCE_LAYERS"), "Instrumented or wrong workload process environment")
        require(diagnostic or not (root / "recipes").exists(), "Diagnostic recipes leaked into ordinary timing")
        expected_command = [initial["exe"], "--gtest_filter=*material_phase0_baseline*", "--rhi-no-validation",
                            "--output-dir", str(recorded_root), "--gtest_output=xml:" + str(recorded_root / "Tests.xml")]
        require(row["process"]["command"] == expected_command, "Process did not run declared executable/workload")
        result = analyze_run(root, diagnostic_configs[row["arm"]] if diagnostic else config,
                             diagnostic_graphs[row["arm"]] if diagnostic else graphs[row["arm"]], row["arm"], diagnostic, protocol)
        require(result == row["result"], "Saved analysis differs from raw frame/image/native-size evidence")
        if diagnostic:
            require(shader_receipts(root, result) == row["recipes"], "Bound production shaders changed")
    require(base.experiment.independent_processes(rows), "Process lifetimes overlap")
    diagnostics = rows[:3]
    validate_results(rows[3:], diagnostics, protocol)
    if protocol == PROTOCOL:
        require(bound_size_diagnostic(diagnostics) == manifest["boundSizeDiagnostic"],
                "Saved bound-size diagnostic differs from actual native readback")
    decisions = {}
    for stage in ("discovery", "confirmation"):
        selected = [r for r in rows if r["stage"] == stage]
        if not selected:
            continue
        require(stage != "confirmation" or decisions["discovery"]["nextAction"] == "run_confirmation", "Unqualified confirmation")
        decisions[stage] = assess(selected, stage)
        require(decisions[stage] == manifest[stage], "Saved decision differs from complete-block statistics")
    require("discovery" not in decisions or decisions["discovery"]["nextAction"] != "run_confirmation" or
            "confirmation" in decisions, "Promising discovery lacks its declared confirmation")
    return {"verified": True, "protocol": protocol, "diagnosticsOnly": manifest["diagnosticsOnly"], "decisions": decisions}


def report(directory, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    require(not output.exists() and not output.resolve().is_relative_to(directory.resolve()), "Report must be new and outside sealed evidence")
    verified = verify(directory)
    require(verified["protocol"] in (LEGACY_PROTOCOL, PROTOCOL), "Use the v1 report command for a v1 bundle")
    manifest = read(directory / "Experiment.json")
    runs = [r for r in manifest["runs"] if r["stage"] != "diagnostic"]
    require(runs, "Diagnostic-only collection has no ordinary timing report")
    output.mkdir(parents=True)
    rows = [{"stage": r["stage"], "arm": r["arm"], "process": i,
             **{m: r["result"]["timings"][m]["median"] for m in METRICS}} for i, r in enumerate(runs)]
    save(output / "PlotData.json", rows)
    colors = {"A": "#355c9a", "B": "#c26432", "C": "#348563"}
    figure, axes = plt.subplots(3, 2, figsize=(13, 11))
    boundaries = [i - .5 for i in range(1, len(rows)) if rows[i]["stage"] != rows[i - 1]["stage"]]
    for metric, axis in zip(METRICS, axes.flat):
        for arm in ARMS:
            selected = [r for r in rows if r["arm"] == arm]
            axis.scatter([r["process"] for r in selected], [r[metric] for r in selected], color=colors[arm], label=ARMS[arm], s=22)
        for boundary in boundaries:
            axis.axvline(boundary, color="#666666", linestyle="--", linewidth=1)
        axis.set(title=metric, xlabel="Independent process in ABCCBA order", ylabel="GPU ms: process median")
        axis.grid(alpha=.2); axis.legend()
    figure.suptitle("Three-arm CLAS/Shadow experiment; dashed line separates discovery and confirmation")
    figure.tight_layout(); figure.savefig(output / "ProcessMedians.png", dpi=150); plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    stage_colors = {"discovery": "#355c9a", "confirmation": "#8053a4"}
    for pair, axis in zip(PAIRS, axes):
        for i, (stage, assessment) in enumerate(verified["decisions"].items()):
            for j, metric in enumerate(METRICS):
                gain = assessment["contrasts"][pair][metric]
                x = j + (i - .5) * .15
                axis.errorbar(x, gain["estimate"] * 100, yerr=[[100*(gain["estimate"]-gain["lower95"])],
                    [100*(gain["upper95"]-gain["estimate"])]], fmt="o", color=stage_colors[stage], capsize=4,
                    label=stage if j == 0 else None)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(title=pair + " gain", xticks=range(len(METRICS)), xticklabels=METRICS, ylabel="Gain % with block 95% CI")
        axis.tick_params(axis="x", labelrotation=55); axis.legend(); axis.grid(alpha=.2)
    figure.tight_layout(); figure.savefig(output / "BlockGains.png", dpi=150); plt.close(figure)
    stage = "confirmation" if "confirmation" in verified["decisions"] else "discovery"
    selected = [r for r in rows if r["stage"] == stage]
    summary = []
    for metric in METRICS:
        times = {arm: statistics.median(r[metric] for r in selected if r["arm"] == arm) for arm in ARMS}
        gains = verified["decisions"][stage]["contrasts"]
        summary.append({"metric": metric, "stage": stage, "processMedianMs": times,
                        "contrasts": {pair: gains[pair][metric] for pair in PAIRS}})
    save(output / "TimingSummary.json", summary)
    diagnostics = [r for r in manifest["runs"] if r["stage"] == "diagnostic"]
    backend = {r["arm"]: {"identity": r["result"]["identity"], "nativeCLASSizes": r["result"].get("nativeCLASSizes"),
                           "boundPathEvidence": r["result"].get("boundPathEvidence")} for r in diagnostics}
    save(output / "BackendAllocations.json", backend)
    bound_text = ""
    if verified["protocol"] == PROTOCOL:
        sizes = bound_size_diagnostic(diagnostics)
        save(output / "BoundSizeDiagnostic.json", sizes)
        evidence = [{"directory": r["directory"], "stage": r["stage"], "arm": r["arm"],
                     **r["result"]["boundPathEvidence"]} for r in manifest["runs"]]
        save(output / "BoundPathEvidence.json", evidence)
        allocation_rows = "\n".join("| " + arm + " | " + " | ".join(str(item["identity"][key]) for key in
            ("nativeTemplateCount", "templateBytes", "templateAuxiliaryBytes", "clasBytes", "scratchBytes")) + " |"
            for arm, item in backend.items())
        bound_text = (f"C3 bounded-path evidence: final fallback count zero and sticky any-fallback bit false in every C process; "
            "each process retained one generation across all warmup/timed frames and the final untimed receipt. Actual diagnostic 24-byte bounds match the production identity.\n\n" +
            f"Native live CLAS size check: **{sizes['classification']}**, C3 {sizes['boundedTemplateBytes']:,} bytes versus "
            f"original templates {sizes['originalTemplateBytes']:,} bytes ({100*sizes['reductionFraction']:+.2f}% reduction, "
            f"{sizes['liveCLASCount']:,} live CLAS). This is a diagnostic observation; it neither adds an acceptance gate nor proves a traversal-cost cause. No bounds-volume threshold is applied.\n\n" +
            "Retained allocation capacity (bytes), including both unbounded and bounded sets in the single C3 template pool:\n\n" +
            "| Arm | Native templates | Template storage | Template auxiliary | Runtime CLAS arena | Runtime scratch |\n"
            "|---|---:|---:|---:|---:|---:|\n" + allocation_rows + "\n\n")
    save(output / "Verification.json", verified)
    save(output / "Versions.json", {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "evidence": str(directory.resolve()), "manifestSha256": digest(directory / "Manifest.json"),
        "runnerSha256": digest(Path(__file__)), "source": "Experiment.json runs[].result.timings; raw MaterialBaseline.json frames"})
    shutil.copy2(__file__, output / Path(__file__).name)
    table = "\n".join("| " + r["metric"] + " | " + " | ".join(f"{r['processMedianMs'][a]:.6f}" for a in ARMS) + " | " +
        " | ".join(f"{100*r['contrasts'][p]['estimate']:+.2f}% [{100*r['contrasts'][p]['lower95']:+.2f}, {100*r['contrasts'][p]['upper95']:+.2f}]" for p in PAIRS) + " |" for r in summary)
    (output / "report.md").write_text("# CLAS template Shadow qualification\n\n" +
        f"{stage.capitalize()}: n = {len(selected)//6} complete six-process ABCCBA blocks. Each pair uses symmetric ABBA observations from the same block; contrasts share samples and are not independent replications. No frames or outliers were removed.\n\n" +
        "Same native-resolution production graph, camera and exact HDR/depth/shadow checks. Compilation, startup, template-pool initialization, final readbacks and native size diagnostics are excluded. Persistent template and runtime allocation capacities remain in BackendAllocations.json; native sizes are actual live CLAS output, not total renderer memory.\n\n" +
        "Milliseconds are medians of process medians; percentage gains and 95% intervals use paired block log ratios. Negative gain is a regression. All five registered CI gates must pass confirmation; discovery never accepts.\n\n" +
        bound_text +
        "| GPU scope | A triangles ms | B original templates ms | C candidate ms | B/A gain [CI] | C/A gain [CI] | C/B gain [CI] |\n|---|---:|---:|---:|---:|---:|---:|\n" + table +
        "\n\n![Process medians](ProcessMedians.png)\n\n![Block gains](BlockGains.png)\n\n```json\n" + json.dumps(verified, indent=2) +
        "\n```\n\nReproduce with the current checkout's `python -X utf8 -B Tools/Perf/MeasureCLASTemplateShadow.py report --input " + str(directory) + " --output NEW_DIRECTORY`.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--sample", type=Path, default=ROOT / "Pipelines/Samples/gpu_driven_tessellation.metallic_graph.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=1444); p.add_argument("--height", type=int, default=694)
    p.add_argument("--warmup", type=int, default=32); p.add_argument("--frames", type=int, default=64)
    p.add_argument("--template-shape", choices=TEMPLATE_SHAPES, default="right")
    p.add_argument("--instantiation-flags", choices=INSTANTIATION_FLAGS, default="fastTrace")
    p.add_argument("--template-bounds", choices=("geometry",), default="geometry")
    p.add_argument("--native-sizes", action="store_true", default=True)
    p = commands.add_parser("run")
    p.add_argument("--exe", type=Path, required=True); p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, choices=range(2, 11), default=3)
    p.add_argument("--confirmation-blocks", type=int, choices=range(2, 11), default=5)
    p.add_argument("--timeout", type=int, default=300); p.add_argument("--diagnostics-only", action="store_true")
    for name in ("verify", "report"):
        p = commands.add_parser(name); p.add_argument("--input", type=Path, required=True)
        if name == "report": p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run(args)
    elif args.command == "verify": print(json.dumps(verify(args.input), indent=2))
    else: report(args.input, args.output)


if __name__ == "__main__":
    main()
