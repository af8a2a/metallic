"""Historical Off/Legacy/Smart shadow-LOD evidence using native acceptance packages.

Live prepare/run is retired: depth agreement and refined receivers are standard.
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
    raise ValueError("Retired VSM LOD reference experiment: hierarchy traversal, depth agreement and refined receiver bias are standard; use verify/report for historical evidence")


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
    raise ValueError("Retired VSM LOD reference experiment: hierarchy traversal, depth agreement and refined receiver bias are standard; use verify/report for historical evidence")


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
