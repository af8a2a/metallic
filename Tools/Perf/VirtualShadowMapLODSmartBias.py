"""Bounded Off/Legacy/Smart shadow-LOD experiment using native acceptance packages.

The original Bunny, reverse light and closer orthographic view share immutable
geometry. Only the original view is eligible for the 36-process timing budget.
Diagnostic quality and ordinary GPU timing are separate, sealed experiments.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

import PrepareVirtualShadowMapLODImpact as fixture
import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapGPUProfile as profile
import VirtualShadowMapLODImpact as impact
import VirtualShadowMapLODSelfShadow as previous
import VirtualShadowMapLODSmartBiasQuality as quality


PROTOCOL = "virtual-shadow-map-lod-smart-bias-v1"
ARMS = {"Off": (False, False), "Legacy": (True, False), "Smart": (True, True)}
WORKLOADS = previous.WORKLOADS
BLOCK = ("Off", "Legacy", "Smart", "Smart", "Legacy", "Off")
METRICS = impact.METRICS
RAY_LENGTH = .015
SCENARIOS = ("Baseline", "ReverseLight", "CloseView")


def scenario_settings(name):
    camera = copy.deepcopy(fixture.CAMERA)
    if name == "CloseView":
        camera["orthoHeight"] = 5.
    return camera, [-2, 0, -1] if name == "ReverseLight" else [2, 0, -1]


def schedule(phase, workloads):
    return [dict(index=i, workload=workload, arm=arm, block=block,
                 package=f"process-{i + 1:02d}-{workload}-{arm}")
            for i, (workload, block, arm) in enumerate(
                (workload, block, arm) for workload in workloads
                for block in ([None] if phase == "diagnostic" else range(3))
                for arm in (ARMS if phase == "diagnostic" else BLOCK))]


def normalize(spec, graph, phase, workload, arm):
    """Reuse the original fixture contract after checking the new arm fields."""
    spec, graph = copy.deepcopy(spec), copy.deepcopy(graph)
    case = spec["cases"][0]
    scenario = next((name for name in SCENARIOS if case["id"] == f"BunnySmartBias{name}{workload}"), None)
    if scenario is None or (phase == "timing" and scenario != "Baseline"):
        raise ValueError("Unknown fixture or undeclared stress-scene timing")
    camera, direction = scenario_settings(scenario)
    if case["camera"] != camera or case["lighting"] != dict(sun=dict(direction=direction, illuminance=3., angularRadius=.00465)):
        raise ValueError("Camera/light differs from the declared scenario")
    props = next(n["properties"] for n in graph["nodes"] if n["name"] == "Shadows")
    enabled, smart = ARMS[arm]
    expected = dict(shadowLOD=enabled, shadowLODDepthGuard=False, shadowLODSmartBias=smart,
                    shadowLODScreenRayLength=RAY_LENGTH, shadowBias=.002, filter="hard")
    if any(type(props.get(k)) is not type(v) or props[k] != v for k, v in expected.items()):
        raise ValueError("LOD, guard, Smart bias, ray length or base shadow bias changed")
    if props.pop("shadowLODSmartBiasRefine", False) is not False:
        raise ValueError("v1 Smart must retain the original unrefined path")
    # The old validator supplies scene, stream, residency and instrumentation
    # checks; these substitutions affect only its private validation copies.
    saved_camera = case["camera"]
    case["camera"] = copy.deepcopy(fixture.CAMERA)
    props.pop("shadowLODSmartBias")
    props.pop("shadowLODScreenRayLength")
    props["shadowLODDepthGuard"] = arm == "Off"
    result = previous.normalize(spec, graph, phase, workload, "Legacy" if arm == "Smart" else arm)
    result["cases"][0]["camera"] = saved_camera
    return result


def prepare(args):
    assets, metrics = fixture.inspect_assets()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    camera, direction = scenario_settings(args.scenario)
    manifests = []
    for phase in (("diagnostic", "timing") if args.scenario == "Baseline" else ("diagnostic",)):
        for workload in WORKLOADS:
            for arm, (enabled, smart) in ARMS.items():
                name = f"{phase.title()}-{workload}-{arm}"
                graph = fixture.base_graph()
                graph["edges"] = [edge for edge in graph["edges"] if edge["dst"] != "Shadows.visibility"]
                graph["view"]["camera"] = copy.deepcopy(camera)
                props = next(n["properties"] for n in graph["nodes"] if n["name"] == "Shadows")
                props.update(shadowLOD=enabled, shadowLODHierarchy=True, shadowLODDepthGuard=False,
                             shadowLODSmartBias=smart, shadowLODScreenRayLength=RAY_LENGTH, shadowLODSmartBiasRefine=False,
                             cacheEnabled=workload == "Warm", benchmarkGPUWork=phase == "diagnostic")
                case = dict(id=f"BunnySmartBias{args.scenario}{workload}", graph=f"{name}.json",
                            scene=fixture.SCENE_PATH.as_posix(), metadataOnly=False, enableStreamline=False,
                            synchronizationValidation=phase == "diagnostic", requireNoRayCapabilities=True,
                            temporalJitter=False, width=768, height=512, output="FinalBlit.color",
                            comparisonOutput="Deferred.color", camera=copy.deepcopy(camera), requireShadow=True,
                            lighting=dict(sun=dict(direction=direction, illuminance=3., angularRadius=.00465)),
                            environment=dict(enabled=False), exposureEV100=0,
                            captureFrames=[32, 47] if phase == "diagnostic" else [],
                            comparisonFrames=previous.comparison_frames(phase, workload, arm),
                            comparisonStartFrame=16, freezeStreamingFrame=16, events=[dict(frame=16, cameraCut=True)])
                spec = dict(version=1, mode=phase, frames=48 if phase == "diagnostic" else 96,
                            warmupFrames=32, cases=[case])
                normalize(spec, graph, phase, workload, arm)
                acceptance.save(output / f"{name}.json", graph)
                acceptance.save(output / f"{name}Cases.json", spec)
                manifests.append(f"{name}Cases.json")
    acceptance.save(output / "Assets.json", assets)
    acceptance.save(output / "Preparation.json", dict(protocol=PROTOCOL, scenario=args.scenario,
        fixtureMetrics=metrics, screenRayLength=RAY_LENGTH, qualityCriteria=quality.CRITERIA,
        templateGraph=dict(path=fixture.GRAPH_PATH.as_posix(), sha256=acceptance.digest(fixture.GRAPH_PATH)),
        sourceAssetsModified=False, gpuExecutionPerformed=False))
    acceptance.save(output / "Catalog.json", dict(caseManifests=manifests))
    print(output)


def read_package(source, plan, scheduled):
    package = source / scheduled["package"]
    spec, captures = acceptance.verify_package(package)
    if len(captures) != 1:
        raise ValueError("Exactly one independent process is required per package")
    report = captures[0][1]
    case = report["cases"][0]
    workload, arm, phase = scheduled["workload"], scheduled["arm"], plan["phase"]
    if spec["cases"][0]["id"] != f"BunnySmartBias{plan['scenario']}{workload}":
        raise ValueError("Package scenario differs from the declared plan")
    key = f"{workload}-{arm}"
    planned = plan["inputs"][key]
    archived_spec, archived_graph = source / "Inputs" / f"{key}Cases.json", source / "Inputs" / f"{key}Graph.json"
    identity = acceptance.read(package / "Identity.json")
    if (acceptance.digest(archived_spec) != planned["casesSHA256"]
            or acceptance.digest(archived_graph) != planned["graphSHA256"]
            or spec != acceptance.read(archived_spec)
            or identity["inputs"] != {planned["cases"]: planned["casesSHA256"], planned["graph"]: planned["graphSHA256"]}
            or plan["exe"] not in identity["binaries"]):
        raise ValueError("Child inputs or executable differ from the archived plan")
    normalize(spec, acceptance.read(archived_graph), phase, workload, arm)
    actual = case.get("shadowLODSettings", {})
    if actual.get("smartBiasRefine", False) is not False:
        raise ValueError("v1 receipt unexpectedly enabled Smart-bias refinement")
    expected = dict(enabled=ARMS[arm][0], hierarchy=True, depthGuard=False, smartBias=ARMS[arm][1], errorTexels=1.)
    if (any(actual.get(k) != v for k, v in expected.items())
            or not math.isclose(actual.get("screenRayLength", -1), RAY_LENGTH, rel_tol=0, abs_tol=1e-8)):
        raise ValueError("Actual LOD/Smart settings differ from the declared arm")
    if phase == "diagnostic":
        messages = report["validationMessages"]
        if not report["validation"] or not case["synchronizationValidation"] or messages["captureFailed"] or messages["count"]:
            raise ValueError("Missing/failed Vulkan synchronization validation receipt")
    elif report["validation"] or not report["performanceEligible"] or case.get("captures") or case.get("comparisons"):
        raise ValueError("Ordinary timing contains diagnostic work")
    if (case.get("gpuWorkDiagnostics") is not (phase == "diagnostic")
            or case.get("cpuProfileDiagnostics") or case.get("buildSubstageDiagnostics")):
        raise ValueError("Actual instrumentation differs from the plan")
    measured = [impact.profile_fixture_frame(f) for f in case["frames"] if not f["warmup"]]
    return dict(**scheduled, identity=impact.identity(package),
                shaders=impact.shader_identity(package / "run-1/Process.log", phase),
                normalized=normalize(spec, case["resolvedGraph"], phase, workload, arm),
                streams=impact.stream_input(case, phase), frames=measured,
                medians={m: statistics.median(f[m] for f in measured) for m in METRICS},
                means={m: statistics.mean(f[m] for f in measured) for m in METRICS},
                diagnostic=previous.diagnose(case, workload, "Legacy" if arm == "Smart" else arm,
                                             captures[0][0].parent) if phase == "diagnostic" else None)


def check_records(records, evidence=None):
    normalized, streams, shaders = {}, {}, {}
    for row in records:
        workload, arm = row["workload"], row["arm"]
        if row["identity"] != records[0]["identity"]:
            raise ValueError("Source/binary changed between arms/processes")
        if arm in shaders and shaders[arm] != row["shaders"]:
            raise ValueError("Logged SPIR-V changed for the same arm")
        shaders[arm] = row["shaders"]
        if workload in normalized and (normalized[workload] != row["normalized"] or streams[workload] != row["streams"]):
            raise ValueError("Workload or streaming identity differs between arms/processes")
        normalized[workload], streams[workload] = row["normalized"], row["streams"]
        if evidence is not None:
            reference = next(r for r in evidence["records"] if r["workload"] == workload and r["arm"] == arm)
            if any(row[k] != reference[k] for k in ("identity", "normalized", "streams")):
                raise ValueError("Timing differs from its diagnostic prerequisite")
    for workload in normalized:
        rows = {r["arm"]: r for r in records if r["workload"] == workload and r["diagnostic"]}
        if not rows:
            continue
        reference = next(iter(rows.values()))["diagnostic"]["images"]
        for row in rows.values():
            for a, b in zip(reference, row["diagnostic"]["images"], strict=True):
                if any(a["images"][k]["rawSha256"] != b["images"][k]["rawSha256"] for k in ("VBuffer.depth", "VBuffer.visibility")):
                    raise ValueError("LOD changed primary receiver geometry/visibility")
        if "Legacy" not in rows or "Smart" not in rows:
            continue
        for a, b in zip(rows["Legacy"]["diagnostic"]["rows"], rows["Smart"]["diagnostic"]["rows"], strict=True):
            if a["lod"] != b["lod"] or a["build"] != b["build"] or a["totals"] != b["totals"]:
                raise ValueError("Smart changed Legacy LOD selection, page work or build diagnostics")
            if any(a["work"][k] != b["work"][k] for k in profile.WORK_COUNTERS if not k.startswith("projection")):
                raise ValueError("Smart changed Legacy Mark/Build/Raster work")
            if b["lod"][5] >= b["lod"][10]:
                raise ValueError("Candidate lost the actual shadow-LOD geometry reduction")
            if (rows["Smart"]["normalized"]["cases"][0]["id"].startswith("BunnySmartBiasBaseline")
                    and b["lod"][4:6] != [146, 17348]):
                raise ValueError("Baseline must retain the original 146 clusters / 17,348 triangles")


def verify(source, diagnostic=None):
    source = Path(source).resolve()
    impact.check_seal(source)
    plan, processes = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    if (plan["protocol"] != PROTOCOL or plan["qualityCriteria"] != quality.CRITERIA
            or plan["processes"] != schedule(plan["phase"], plan["workloads"])):
        raise ValueError("Changed protocol, quality criteria or fixed process schedule")
    for path, digest in plan["toolsSHA256"].items():
        if acceptance.digest(source / "Tools" / Path(path).name) != digest:
            raise ValueError("Archived tool differs from its declared identity")
    if not acceptance.read(source / "Completion.json")["passed"] or len(processes) != len(plan["processes"]):
        raise ValueError("Incomplete experiment")
    records = []
    for expected, actual in zip(plan["processes"], processes, strict=True):
        if any(actual.get(k) != v for k, v in expected.items()) or not actual["passed"]:
            raise ValueError("Incomplete or changed scheduled process")
        records.append(read_package(source, plan, expected))
    evidence = None
    if plan["phase"] == "timing":
        diagnostic = Path(diagnostic or plan["diagnostic"]["path"]).resolve()
        if acceptance.digest(diagnostic / "Manifest.json") != plan["diagnostic"]["manifestSHA256"]:
            raise ValueError("Diagnostic prerequisite was replaced")
        evidence = verify(diagnostic)
        if evidence["phase"] != "diagnostic" or evidence["assets"] != acceptance.read(source / "Assets.json") or not evidence["quality"]["passed"]:
            raise ValueError("Missing accepted quality prerequisite")
    check_records(records, evidence)
    return dict(phase=plan["phase"], scenario=plan["scenario"], workloads=plan["workloads"], records=records,
                assets=acceptance.read(source / "Assets.json"),
                quality=quality.analyze(records) if plan["phase"] == "diagnostic" else evidence["quality"])


def run(args):
    workloads = [args.workload] if args.workload else list(WORKLOADS)
    cases_dir, output = args.cases_dir.resolve(), args.output.resolve()
    assets = acceptance.read(cases_dir / "Assets.json")
    impact.validate_assets(assets)
    preparation = acceptance.read(cases_dir / "Preparation.json")
    if (preparation["protocol"] != PROTOCOL or preparation["qualityCriteria"] != quality.CRITERIA
            or preparation["scenario"] != args.scenario):
        raise ValueError("Prepared protocol/quality criteria changed")
    if args.phase == "timing" and args.scenario != "Baseline":
        raise ValueError("Only the Baseline scenario has a declared timing budget")
    inputs, normalized = {}, {}
    for workload in workloads:
        for arm in ARMS:
            cases = cases_dir / f"{args.phase.title()}-{workload}-{arm}Cases.json"
            spec = acceptance.read(cases)
            if spec["cases"][0]["id"] != f"BunnySmartBias{args.scenario}{workload}":
                raise ValueError("Prepared cases differ from the requested scenario")
            graph = (cases.parent / spec["cases"][0]["graph"]).resolve()
            value = normalize(spec, acceptance.read(graph), args.phase, workload, arm)
            if workload in normalized and normalized[workload] != value:
                raise ValueError("Prepared inputs differ beyond the declared arm")
            normalized[workload] = value
            inputs[f"{workload}-{arm}"] = dict(cases=str(cases), graph=str(graph),
                casesSHA256=acceptance.digest(cases), graphSHA256=acceptance.digest(graph))
    evidence, prerequisite = None, None
    if args.phase == "timing":
        if args.diagnostic is None:
            raise ValueError("Timing requires a verified diagnostic prerequisite")
        evidence = verify(args.diagnostic)
        if (evidence["phase"] != "diagnostic" or evidence["scenario"] != args.scenario
                or evidence["assets"] != assets or not evidence["quality"]["passed"]
                or any(w not in evidence["workloads"] for w in workloads)):
            raise ValueError("Diagnostic prerequisite does not qualify these timing inputs")
        prerequisite = dict(path=str(args.diagnostic.resolve()), manifestSHA256=acceptance.digest(args.diagnostic / "Manifest.json"))
    output.mkdir(parents=True, exist_ok=False)
    tool_paths = [Path(module.__file__) for module in (fixture, acceptance, profile, impact, previous, quality, impact.optimization)] + [Path(__file__)]
    tools = {str(p.resolve()): acceptance.digest(p) for p in tool_paths}
    plan = dict(protocol=PROTOCOL, phase=args.phase, scenario=args.scenario, workloads=workloads, created=impact.utc(), inputs=inputs,
                processes=schedule(args.phase, workloads), diagnostic=prerequisite, exe=str(args.exe.resolve()),
                qualityCriteria=quality.CRITERIA, timeoutSeconds=args.timeout, toolsSHA256=tools,
                statisticalUnit="Three independent ABCCBA blocks per workload; no adaptive samples",
                cache="Disk caches retained; fresh process per sample; first 32 frames excluded",
                gpuExclusivity="Not enforced; applications and clocks unchanged")
    acceptance.save(output / "Plan.json", plan)
    acceptance.save(output / "Assets.json", assets)
    (output / "Inputs").mkdir()
    (output / "Tools").mkdir()
    for key, item in inputs.items():
        for source_key, suffix in (("cases", "Cases"), ("graph", "Graph")):
            (output / "Inputs" / f"{key}{suffix}.json").write_bytes(Path(item[source_key]).read_bytes())
    for path in tool_paths:
        (output / "Tools" / path.name).write_bytes(path.read_bytes())
    processes, records = [], []
    try:
        for scheduled in plan["processes"]:
            impact.validate_assets(assets)
            if any(acceptance.digest(path) != digest for path, digest in tools.items()):
                raise ValueError("Experiment tools changed after plan creation")
            item = inputs[f"{scheduled['workload']}-{scheduled['arm']}"]
            if any(acceptance.digest(item[k]) != item[f"{k}SHA256"] for k in ("cases", "graph")):
                raise ValueError("Prepared inputs changed after plan creation")
            row = dict(**scheduled, started=impact.utc(), beforeGPU=impact.telemetry(), passed=False)
            processes.append(row)
            acceptance.save(output / "Processes.json", processes)
            try:
                acceptance.run(SimpleNamespace(cases=Path(item["cases"]), exe=args.exe,
                    output=output / scheduled["package"], runs=1, timeout=args.timeout))
                records.append(read_package(output, plan, scheduled))
                check_records(records, evidence)
                impact.validate_assets(assets)
                row["passed"] = True
            finally:
                row.update(finished=impact.utc(), afterGPU=impact.telemetry())
                acceptance.save(output / "Processes.json", processes)
        acceptance.save(output / "Completion.json", dict(passed=True, finished=impact.utc()))
        impact.seal(output)
        result = verify(output, args.diagnostic)
    except Exception as error:
        acceptance.save(output / "Completion.json", dict(passed=False, finished=impact.utc(), error=str(error)))
        impact.seal(output)
        raise RuntimeError(f"Experiment stopped; evidence retained at {output}: {error}") from error
    print(f"{output}\nOff-reference image gates passed: {result['quality']['passed']} (independent geometry acceptance is separate)")


def report(args):
    value = verify(args.source, args.diagnostic)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    acceptance.save(output / "Quality.json", value["quality"])
    acceptance.save(output / "ProcessMeasurements.json", [{k: r[k] for k in ("workload", "arm", "block", "package", "medians", "means", "frames")} for r in value["records"]])
    lines = ["# Shadow LOD Smart bias experiment", "", "Fixed resident multi-level Bunny, finest main view, one directional light. Pass/graph GPU scope excludes presentation; no general production FPS claim.", "", f"Off-reference image gates passed: {value['quality']['passed']}. These compare against the full-geometry VSM, which can itself contain errors; independent authored-ray and contact acceptance remain separate. See Quality.json for extra-dark, lost-occlusion, eroded-interior and edge displacement metrics.", ""]
    if value["phase"] == "diagnostic":
        quality.plot(value["records"], output)
        acceptance.save(output / "DiagnosticReceipts.json", [{"workload": r["workload"], "arm": r["arm"], **r["diagnostic"]} for r in value["records"]])
        lines += ["Diagnostic timestamps include validation, readbacks and counters and are not performance evidence.", ""]
        lines += [f"![{w} native shadow comparison]({w}-ShadowQuality.png)" for w in value["workloads"]]
    else:
        # Reuse the established three-block estimator; translate only display labels.
        rows = [dict(r, arm="Guarded" if r["arm"] == "Smart" else r["arm"]) for r in value["records"]]
        comparisons = previous.block_statistics(rows, value["workloads"])
        for row in comparisons:
            row["candidate"] = row["candidate"].replace("Guarded", "Smart")
            row["baseline"] = row["baseline"].replace("Guarded", "Smart")
        acceptance.save(output / "Comparisons.json", comparisons)
        lines += ["Three ABCCBA blocks per workload; 96 frames/process, first 32 excluded. Mean paired log ratios of process medians; t(df=2)=4.303 for 95% CI. Frames are not independent samples. No additional samples or trimming.", "", "| State | Comparison | GPU scope | Change | 95% CI |", "|---|---|---|---:|---:|"]
        for row in comparisons:
            if "changePercent" in row:
                low, high = row["ci95Percent"]
                lines.append(f"| {row['workload']} | {row['candidate']}/{row['baseline']} | {row['metric']} | {row['changePercent']:+.2f}% | [{low:+.2f}, {high:+.2f}]% |")
    lines += ["", "Cache-disabled Redraw changes projection padding. Cache equivalence is tested separately with in-frame forceRedraw under unchanged cacheEnabled. All clocks and concurrent applications remain unchanged."]
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "Report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--scenario", choices=SCENARIOS, default="Baseline")
    runner = commands.add_parser("run")
    runner.add_argument("--cases-dir", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--phase", choices=("diagnostic", "timing"), required=True)
    runner.add_argument("--scenario", choices=SCENARIOS, default="Baseline")
    runner.add_argument("--diagnostic", type=Path)
    runner.add_argument("--workload", choices=WORKLOADS)
    runner.add_argument("--exe", type=Path, default=acceptance.ROOT / "build-release/tests/MetallicRHITests.exe")
    runner.add_argument("--timeout", type=float, default=600)
    for name in ("verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        if name == "report":
            command.add_argument("output", type=Path)
        command.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.source, args.diagnostic)
        print(f"Verified {result['phase']}: {len(result['records'])} processes; quality accepted: {result['quality']['passed']}")
    else:
        {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
