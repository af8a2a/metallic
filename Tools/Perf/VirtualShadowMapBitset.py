"""Fixed-budget VSM dirty-page bitset A/B; never edits shaders or prepares graphs.

run --phase diagnostic|discovery|confirmation --cases-dir DIR --exe EXE --output NEW_DIR
    [--diagnostic DIAGNOSTIC_SOURCE] [--reference DISCOVERY_SOURCE]
report SOURCE NEW_REPORT [--diagnostic DIAGNOSTIC_SOURCE] [--reference DISCOVERY_SOURCE]

Cases are Diagnostic-ACases.json/Diagnostic-BCases.json or Timing-A/B counterparts.
A disables benchmarkClusterPageBitset, B enables it; both use production mark/cluster.
--experiment directional-build instead toggles only benchmarkDirectionalBuild,
with exact build-diagnostic equivalence and unchanged mark/projection SPIR-V.
Timing requires verified diagnostic evidence, and confirmation requires promising discovery.
This protocol reuses acceptance receipts and ABBA mathematics, not the WorkRaster protocol.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

import ExperimentRunner as experiment
import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapGPUProfile as profile
import VirtualShadowMapOptimization as optimization


PROTOCOL = "virtual-shadow-map-bitset-v1"
ARMS = ("A", "B")
BLOCK = ("A", "B", "B", "A")
BLOCKS = {"diagnostic": 0, "discovery": 2, "confirmation": 3}
SWITCH = "benchmarkClusterPageBitset"
DIRECTIONAL_PROTOCOL = "virtual-shadow-map-directional-build-v1"
DIRECTIONAL_SWITCH = "benchmarkDirectionalBuild"
METRICS = optimization.METRICS
POLICY = {"minimumGain": .03, "maximumGraphRegression": .02, "confidence": .95,
          "primaryMetric": "buildMs", "statisticalUnit": "independent four-process ABBA block",
          "statistic": "ABBA log ratios via ExperimentRunner.block_gain", "promotion": "manual",
          "spreadWarning": .10}


def schedule(phase):
    order = ARMS if phase == "diagnostic" else BLOCK * BLOCKS[phase]
    return [{"index": index, "block": None if phase == "diagnostic" else index // 4,
             "arm": arm, "package": f"process-{index + 1:02d}-{arm}"} for index, arm in enumerate(order)]


def normalized_workload(spec, graph, arm, directional=False):
    mode = spec["mode"]
    if mode not in ("diagnostic", "timing") or len(spec["cases"]) != 1:
        raise ValueError("Expected one diagnostic or timing case")
    case = spec["cases"][0]
    count = case.get("frames", spec.get("frames"))
    warmup = case.get("warmupFrames", spec.get("warmupFrames"))
    if (type(count) is not int or count not in ((64,) if mode == "diagnostic" else (64, 96)) or
            type(warmup) is not int or warmup != 32 or optimization.comparison_start(spec) != 16):
        raise ValueError("Fixed budget requires diagnostic 64/timing 64 or 96 frames, warmup 32 and canonical freeze/cut at 16")
    if case.get("comparisonFrames"):
        raise ValueError("comparisonFrames must be empty: extra renders would alter temporal history")
    if mode == "diagnostic":
        if case.get("captureFrames") != [16, 32, 63]:
            raise ValueError("Diagnostic checkpoints must be exactly frames 16, 32, 63")
    elif (case.get("captureFrames") or case.get("comparisonFrames") or
          any(spec.get(key, False) or case.get(key, False) for key in ("validation", "synchronizationValidation"))):
        raise ValueError("Ordinary timing requires validation and capture/readback off")
    graph = copy.deepcopy(graph)
    switch = DIRECTIONAL_SWITCH if directional else SWITCH
    nodes = [node for node in graph["nodes"] if node.get("name") == "Shadows"]
    if len(nodes) != 1 or nodes[0].get("properties", {}).get(switch) is not (arm == "B"):
        raise ValueError(f"{arm}: {switch} must explicitly be the boolean {arm == 'B'}")
    nodes[0]["properties"].pop(switch, None)
    for node in graph["nodes"]:
        if mode == "timing" and node.get("properties", {}).get("benchmarkCaptureSemanticSource", False):
            raise ValueError("Ordinary timing cannot capture semantic texture identity")
    return optimization.normalized_workload(spec, graph, "Both")


def declared_output_workload(workload):
    """Mirror only AcceptanceTests' explicit markOutput calls, preserving other graph fields."""
    result = copy.deepcopy(workload)
    case = result["cases"][0]
    outputs = case["graph"].setdefault("outputs", [])
    required = [case.get("output", "FinalBlit.color")]
    if result["mode"] == "diagnostic" and case.get("captureFrames"):
        required.append(case.get("comparisonOutput", "Deferred.color"))
    for output in required:
        if output not in outputs:
            outputs.append(output)
    return result


def prepare_plan(args):
    mode = "diagnostic" if args.phase == "diagnostic" else "timing"
    directional = getattr(args, "experiment", "cluster-bitset") == "directional-build"
    inputs, previous = {}, None
    for arm in ARMS:
        cases = args.cases_dir.resolve() / f"{mode.title()}-{arm}Cases.json"
        spec = acceptance.read(cases)
        if spec["mode"] != mode:
            raise ValueError("Case mode differs from the phase")
        graph = (cases.parent / spec["cases"][0]["graph"]).resolve()
        workload = normalized_workload(spec, acceptance.read(graph), arm, directional)
        if previous is not None and workload != previous:
            raise ValueError("Cases/graphs differ beyond the declared experiment switch")
        previous = workload
        inputs[arm] = {"cases": str(cases), "casesSHA256": acceptance.digest(cases),
                       "graph": str(graph), "graphSHA256": acceptance.digest(graph)}
    return {"protocol": DIRECTIONAL_PROTOCOL if directional else PROTOCOL, "phase": args.phase, "mode": mode, "created": optimization.utc(),
            "policy": POLICY, "blocks": BLOCKS[args.phase], "timeoutSeconds": 600, "runsPerPackage": 1,
            "exe": str(args.exe.resolve()), "inputs": inputs, "comparisonStartFrame": 16,
            "processes": schedule(args.phase), "toolHashes": tool_hashes(),
            "cache": "Existing disk caches retained; fresh process per sample",
            "gpuExclusivity": "Not enforced; no applications are stopped", "workload": declared_output_workload(previous)}


def shader_identity(log, arm, mode, directional=False):
    result = optimization.shader_identity(log, "Both", mode)
    projection = set()
    for line in log.splitlines():
        if "[VSMShader]" in line:
            fields = dict(re.findall(r"(\w+)=(\S+)", line.split("[VSMShader]", 1)[1]))
            entry = fields.get("entry")
            if directional and entry in (*result, "vsmProjectMain"):
                expected = arm == "B" and entry == "vsmBuildMain"
                if fields.get("lightType") != "0" or fields.get("directionalBuild") != str(expected).lower():
                    raise ValueError("Compiled directionalBuild flag or directional light type mismatch")
                if entry == "vsmProjectMain":
                    value = fields.get("spirvSHA256", "")
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                        raise ValueError("Missing or invalid projection SPIR-V SHA256")
                    projection.add(value.lower())
            elif not directional and entry in result and fields.get("clusterPageBitset") != str(arm == "B").lower():
                raise ValueError("Compiled shader clusterPageBitset flag mismatch or missing")
    if directional:
        if len(projection) != 1:
            raise ValueError("Missing or unstable projection SPIR-V identity")
        result["vsmProjectMain"] = next(iter(projection))
    return result


def check_shaders(processes, directional=False):
    hashes = {}
    for row in processes:
        if row["arm"] in hashes and hashes[row["arm"]] != row["shaders"]:
            raise ValueError("Actual shader identity changed within an arm")
        hashes[row["arm"]] = row["shaders"]
    if hashes["A"]["vsmMarkMain"] != hashes["B"]["vsmMarkMain"]:
        raise ValueError("Unrelated mark shader changed between experiment arms")
    if directional and hashes["A"]["vsmProjectMain"] != hashes["B"]["vsmProjectMain"]:
        raise ValueError("Unrelated projection shader changed between experiment arms")
    if hashes["A"]["vsmBuildMain"] == hashes["B"]["vsmBuildMain"]:
        raise ValueError("Candidate did not change actual build SPIR-V")


def verify_experiment(source):
    source = Path(source).resolve()
    plan, receipts = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    phase = plan["phase"]
    directional = plan.get("protocol") == DIRECTIONAL_PROTOCOL
    expected = schedule(phase)
    mode = "diagnostic" if phase == "diagnostic" else "timing"
    if (plan.get("protocol") not in (PROTOCOL, DIRECTIONAL_PROTOCOL) or plan.get("policy") != POLICY or plan.get("mode") != mode or
            plan.get("processes") != expected or plan.get("blocks") != BLOCKS[phase] or
            plan.get("timeoutSeconds") != 600 or plan.get("runsPerPackage") != 1 or
            plan.get("comparisonStartFrame") != 16 or len(receipts) != len(expected)):
        raise ValueError("Unknown plan/policy or incomplete fixed process schedule")
    # Initial v1 plans stored the raw graph. Apply the same explicit output registration
    # when reading those plans; the sealed plan/package bytes are never rewritten.
    planned_workload = declared_output_workload(plan["workload"])
    rows, frames, work, build, inputs, startup, signatures = [], [], [], [], [], [], []
    identity, workload, canonical, previous_end = None, None, None, None
    for item, receipt in zip(expected, receipts):
        if any(receipt.get(key) != value for key, value in item.items()) or receipt.get("status") != "passed":
            raise ValueError("Failed, reordered or replaced process receipt")
        started, ended = (datetime.datetime.fromisoformat(receipt[key]) for key in ("started", "ended"))
        if started.tzinfo is None or ended.tzinfo is None or ended < started or (previous_end is not None and started < previous_end):
            raise ValueError("Invalid or overlapping serialized process timestamps")
        previous_end = ended
        package = source / item["package"]
        if acceptance.digest(package / "Manifest.json") != receipt["manifestSHA256"]:
            raise ValueError("Child package changed after process receipt")
        spec, captures = acceptance.verify_package(package)
        integrity, children = acceptance.read(package / "Integrity.json"), acceptance.read(package / "Processes.json")
        if (len(captures) != 1 or integrity["requestedRuns"] != 1 or integrity["completedRuns"] != 1 or
                len(children) != 1 or children[0].get("timedOut")):
            raise ValueError("Expected one complete independent process per package")
        path, data = captures[0]
        if data["mode"] != mode:
            raise ValueError("Package mode differs from planned phase")
        case = data["cases"][0]
        current = acceptance.read(package / "Identity.json")
        planned = plan["inputs"][item["arm"]]
        if any(current["inputs"].get(planned[key]) != planned[key + "SHA256"] for key in ("cases", "graph")):
            raise ValueError("Package inputs differ from declared plan")
        current = {key: current[key] for key in ("sources", "binaries")}
        if (not all(current.values()) or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                for values in current.values() for value in values.values()) or
                (identity is not None and current != identity)):
            raise ValueError("Empty or changed source/binary identity")
        identity = current
        current = normalized_workload(spec, case["resolvedGraph"], item["arm"], directional)
        if (workload is not None and current != workload) or current != planned_workload:
            raise ValueError("Cases/graphs differ beyond the bitset switch")
        workload = current
        streams = [optimization.stream_signature(frame, mode, frame["frame"] >= 16) for frame in case["frames"]]
        current = [frame for frame in streams if frame["frame"] >= 16]
        if canonical is not None and current != canonical:
            raise ValueError("Canonical streaming input signature drift between processes/arms")
        canonical = current
        inputs.extend({"process": item["index"], "arm": item["arm"], "comparisonEligible": frame["frame"] >= 16,
                       "frame": frame["frame"], **stream} for frame in streams for stream in frame["streams"])
        startup.append({"process": item["index"], "arm": item["arm"], "frames": [
            {"frame": raw["frame"], "streaming": stream["streams"], "statistics": raw.get("shadow", {}).get("statistics")}
            for raw, stream in zip(case["frames"], streams) if raw["frame"] < 16]})
        timings = [profile.profile_frame(frame) for frame in case["frames"]]
        work_rows = [profile.work_frame(frame, mode) for frame in case["frames"]]
        build_rows = [profile.build_frame(frame, mode) for frame in case["frames"]]
        if mode == "diagnostic":
            if not all(row["gpuWorkAvailable"] for row in work_rows):
                raise ValueError("Missing declared diagnostic workload readback")
            signature = optimization.diagnostic_signature(case, spec["cases"][0], 16)
            stable_work = []
            for raw in case["frames"]:
                if raw["frame"] < 16:
                    continue
                faces = raw["shadow"]["gpuWork"]["faces"]
                if {face["lightIndex"] * 6 + face["face"] for face in faces} != set(range(len(raw["shadow"]["statistics"]))):
                    raise ValueError("Diagnostic workload must cover every legacy-statistics face")
                stable_work.append({"frame": raw["frame"], "faces": [
                    {"lightIndex": face["lightIndex"], "face": face["face"], "counters": {
                        name: face["counters"][name] for name in profile.WORK_COUNTERS if name != "markPageRequests"}}
                    for face in sorted(faces, key=lambda value: (value["lightIndex"], value["face"]))]})
            signature["stableWorkCounters"] = stable_work
            if directional:
                if not all(row["gpuBuildDiagnosticsAvailable"] for row in build_rows):
                    raise ValueError("Directional build requires complete build diagnostic readbacks")
                if any(raw["shadow"]["gpuBuildDiagnostics"]["version"] != 3 for raw in case["frames"]):
                    raise ValueError("Directional build requires version 3 with all 45 build counters")
                signature["exactBuildDiagnostics"] = [{"frame": raw["frame"],
                    "diagnostics": raw["shadow"]["gpuBuildDiagnostics"]}
                    for raw in case["frames"] if raw["frame"] >= 16]
            signatures.append(signature)
        elif data.get("gpuWorkDiagnostics", False) or case.get("gpuWorkDiagnostics", False):
            raise ValueError("Instrumented counters cannot enter ordinary timing")
        declared_build = case.get("gpuBuildDiagnostics", data.get("gpuBuildDiagnostics"))
        if declared_build is True and not all(row["gpuBuildDiagnosticsAvailable"] for row in build_rows):
            raise ValueError("Missing declared build diagnostics")
        if declared_build is False and any(row["gpuBuildDiagnosticsAvailable"] for row in build_rows):
            raise ValueError("Build diagnostic declaration contradicts enabled readback")
        valid = [row for row in timings if not row["warmup"]]
        medians = {metric: profile.describe([row[metric] for row in valid])["median"] for metric in METRICS}
        rows.append({**item, "started": receipt["started"], "ended": receipt["ended"], "capture": str(path),
            "manifestSHA256": receipt["manifestSHA256"], "postWarmupFrames": len(valid), "metrics": medians,
            "shaders": shader_identity((path.parent / "Process.log").read_text(encoding="utf-8", errors="replace"), item["arm"], mode, directional)})
        for output, values in ((frames, timings), (work, work_rows), (build, build_rows)):
            output.extend({"process": item["index"], "arm": item["arm"], "block": item["block"], **row} for row in values)
    check_shaders(rows, directional)
    # All input identities are checked before attributing output differences to correctness.
    if signatures and any(value != signatures[0] for value in signatures[1:]):
        raise ValueError("Canonical PNG/HDR/depth/shadow, legacy 32-statistics or stable-workload mismatch after input verification")
    return {"source": str(source), "phase": phase, "mode": mode, "plan": plan,
        "planSHA256": acceptance.digest(source / "Plan.json"), "processesSHA256": acceptance.digest(source / "Processes.json"),
        "identity": identity, "workload": workload, "processes": rows, "frames": frames, "workFrames": work,
        "buildDiagnosticFrames": build, "inputFrames": inputs, "startupEvidence": startup,
        "canonicalInputSignatures": canonical, "diagnosticReceipts": signatures[0] if signatures else None,
        "exactPNGLinearHDRDepthShadowAndLegacyStatistics": bool(signatures)}


def comparisons(rows, phase):
    if phase == "diagnostic":
        return {"stage": phase, "decision": "diagnostic_passed", "nextAction": "run_discovery"}
    for block in range(BLOCKS[phase]):
        if tuple(row["arm"] for row in rows if row["block"] == block) != BLOCK:
            raise ValueError("Incomplete ABBA block")
    values = [{"result": {"timings": {name: {"median": value} for name, value in row["metrics"].items()}}} for row in rows]
    gains = {metric: experiment.block_gain(values, metric) for metric in METRICS}
    target, graph = gains["buildMs"], gains["graphMs"]
    if target["upper95"] < 0 or graph["upper95"] < -.02:
        decision, assessment, action = "reject", "confirmed_regression", "retain_baseline"
    elif target["upper95"] < .03:
        decision, assessment, action = "reject", "below_target", "retain_baseline"
    elif phase == "confirmation" and target["lower95"] >= .03 and graph["lower95"] >= -.02:
        decision, assessment, action = "accept", "confirmed_gain", "manual_promotion"
    elif phase == "discovery" and target["estimate"] >= .03:
        decision, assessment, action = "inconclusive", "promising_gain", "run_confirmation"
    else:
        decision, assessment, action = "inconclusive", "uncertain", "retain_candidate"
    warnings = []
    for arm in ARMS:
        for metric in ("buildMs", "graphMs"):
            values = [row["metrics"][metric] for row in rows if row["arm"] == arm]
            spread = (max(values) - min(values)) / statistics.median(values)
            if spread > POLICY["spreadWarning"]:
                warnings.append({"arm": arm, "metric": metric, "relativeRange": spread})
    return {"stage": phase, "decision": decision, "effectAssessment": assessment, "nextAction": action,
            "evidenceValidity": "valid", "primaryMetric": "buildMs", "gains": gains, "spreadWarnings": warnings}


def check_diagnostic(result, diagnostic):
    if (diagnostic["phase"] != "diagnostic" or not diagnostic["exactPNGLinearHDRDepthShadowAndLegacyStatistics"] or
            result["plan"]["protocol"] != diagnostic["plan"]["protocol"] or
            result["identity"] != diagnostic["identity"] or
            optimization.scene_workload(result["workload"]) != optimization.scene_workload(diagnostic["workload"])):
        raise ValueError("Diagnostic reference differs in source, binary or scene workload")
    prior = {frame["frame"]: frame for frame in diagnostic["canonicalInputSignatures"]}
    for current in result["canonicalInputSignatures"]:
        if current["frame"] not in prior:
            continue
        left, right = copy.deepcopy(current), copy.deepcopy(prior[current["frame"]])
        for a, b in zip(left["streams"], right["streams"]):
            if not a["textureSemanticIdentityAvailable"]:
                for row in (a, b):
                    for key in ("textureSemanticIdentitySha256", "textureSemanticIdentityBytes", "textureSemanticIdentityAvailable"):
                        row.pop(key, None)
        if left != right:
            raise ValueError("Canonical diagnostic/timing input mismatch")


def check_reference(result, reference):
    if (result["phase"] != "confirmation" or reference["phase"] != "discovery" or
            result["plan"]["protocol"] != reference["plan"]["protocol"] or
            result["identity"] != reference["identity"] or result["workload"] != reference["workload"] or
            result["canonicalInputSignatures"] != reference["canonicalInputSignatures"]):
        raise ValueError("Confirmation requires identical discovery source, binary and timing inputs")
    for arm in ARMS:
        current = next(row["shaders"] for row in result["processes"] if row["arm"] == arm)
        previous = next(row["shaders"] for row in reference["processes"] if row["arm"] == arm)
        if current != previous:
            raise ValueError("Shader identity changed between discovery and confirmation")
    if comparisons(reference["processes"], "discovery")["nextAction"] != "run_confirmation":
        raise ValueError("Discovery did not qualify for independent confirmation")


def load_links(args, phase, plan=None):
    links = {}
    for key, required in (("diagnostic", phase != "diagnostic"), ("reference", phase == "confirmation")):
        pinned = (plan or {}).get("links", {}).get(key)
        path = getattr(args, key, None) or (pinned or {}).get("source")
        if required and not path:
            raise ValueError(f"{phase} requires --{key} evidence")
        if path and not required:
            raise ValueError(f"--{key} is not applicable to {phase}")
        if path:
            result = verify_experiment(path)
            if pinned and any(result[name] != pinned[name] for name in ("planSHA256", "processesSHA256")):
                raise ValueError("Linked evidence changed after planning")
            links[key] = result
    if links:
        if links["diagnostic"]["phase"] != "diagnostic":
            raise ValueError("Expected diagnostic reference")
        if "reference" in links:
            check_diagnostic(links["reference"], links["diagnostic"])
            if links["reference"]["phase"] != "discovery" or comparisons(links["reference"]["processes"], "discovery")["nextAction"] != "run_confirmation":
                raise ValueError("Confirmation requires promising discovery")
    return links


def link_receipts(links):
    return {key: {name: value[name] for name in ("source", "planSHA256", "processesSHA256")}
            for key, value in links.items()}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        plan = prepare_plan(args)
        links = load_links(args, args.phase)
        if any(plan["protocol"] != link["plan"]["protocol"] for link in links.values()):
            raise ValueError("Linked evidence uses a different experiment protocol")
        plan["links"] = link_receipts(links)
        if links and optimization.scene_workload(plan["workload"]) != optimization.scene_workload(links["diagnostic"]["workload"]):
            raise ValueError("Planned scene differs from diagnostic reference")
        if "reference" in links and plan["workload"] != links["reference"]["workload"]:
            raise ValueError("Planned timing workload differs from discovery")
        acceptance.save(output / "Plan.json", plan)
        identity = links["diagnostic"]["identity"] if links else None
        for item in plan["processes"]:
            row = {**item, "started": optimization.utc(), "status": "running"}
            rows.append(row)
            acceptance.save(output / "Processes.json", rows)
            try:
                for paths in plan["inputs"].values():
                    if any(acceptance.digest(paths[key]) != paths[key + "SHA256"] for key in ("cases", "graph")):
                        raise ValueError("Declared input changed after planning")
                package = output / item["package"]
                acceptance.run(SimpleNamespace(cases=Path(plan["inputs"][item["arm"]]["cases"]),
                    exe=args.exe, output=package, runs=1, timeout=600))
                acceptance.verify_package(package)
                current = acceptance.read(package / "Identity.json")
                current = {key: current[key] for key in ("sources", "binaries")}
                if identity is not None and current != identity:
                    raise ValueError("Source/binary identity differs from previous evidence")
                identity = current
                row.update(status="passed", manifestSHA256=acceptance.digest(package / "Manifest.json"))
            except BaseException as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                row["ended"] = optimization.utc()
                acceptance.save(output / "Processes.json", rows)
        result = verify_experiment(output)
        if links:
            check_diagnostic(result, links["diagnostic"])
        if "reference" in links:
            check_reference(result, links["reference"])
        acceptance.save(output / "Completion.json", {"status": "passed", "ended": optimization.utc(), "processes": len(rows)})
    except BaseException as error:
        acceptance.save(output / "Failure.json", {"status": "failed", "ended": optimization.utc(),
                        "error": f"{type(error).__name__}: {error}", "recordedProcesses": len(rows)})
        raise
    print(output)


def tool_hashes():
    return {str(Path(module.__file__).resolve()): acceptance.digest(Path(module.__file__))
            for module in (sys.modules[__name__], optimization, acceptance, profile, experiment)}


def plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if result["mode"] == "diagnostic":
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        directional = result["plan"]["protocol"] == DIRECTIONAL_PROTOCOL
        panels = ((axes[0], "buildDiagnosticFrames", ("clusterLevelTests", "clusterSurvivingGroups")),
            (axes[1], "buildDiagnosticFrames", ("clusterBitsetWordTests", "triangleBitsetWordTests", "triangleBitsetPageHits"))) if directional else (
            (axes[0], "workFrames", ("buildPageTests",)),
            (axes[1], "buildDiagnosticFrames", ("clusterPageTests", "clusterBitsetWordTests")))
        for axis, field, names in panels:
            for index, name in enumerate(names):
                values = [[row[name] for row in result[field] if row["arm"] == arm and not row["warmup"] and name in row] for arm in ARMS]
                if all(values):
                    axis.bar([position + index * .3 for position in range(2)], [statistics.median(value) for value in values], width=.3, label=name)
            axis.set_xticks([0, 1], ARMS)
            axis.set_ylabel("Post-warmup per-frame median count")
            if axis.has_data():
                axis.legend(fontsize=8)
        figure.suptitle("Directional build: identical diagnostic work; counts are not timings" if directional else
            "Diagnostic work only; page reads and bitset word reads have different units")
        figure.savefig(output / "DiagnosticWork.png", dpi=150)
        plt.close(figure)
        return
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, ("buildMs", "allocatorMs", "shadowsMs", "graphMs")):
        for arm, color in (("A", "#4774b7"), ("B", "#dc8338")):
            rows = [row for row in result["processes"] if row["arm"] == arm]
            axis.scatter([row["index"] + 1 for row in rows], [row["metrics"][metric] for row in rows], label=arm, color=color)
        axis.set(title=metric, xlabel="Independent process in fixed ABBA order", ylabel="Post-warmup median (ms)")
        axis.legend()
    figure.savefig(output / "ProcessMedians.png", dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, metric in enumerate(METRICS):
        gain = result["comparison"]["gains"][metric]
        estimate = gain["estimate"] * 100
        axis.errorbar(index, estimate, yerr=[[estimate - gain["lower95"] * 100], [gain["upper95"] * 100 - estimate]], fmt="o", color="#4774b7")
        axis.scatter([index] * len(gain["values"]), [value * 100 for value in gain["values"]], marker="x", color="#dc8338")
    axis.axhline(0, color="gray", linewidth=.7)
    axis.set_xticks(range(len(METRICS)), METRICS, rotation=25)
    axis.set(ylabel="B gain versus A (%)", title=f"{result['phase']}: n = {BLOCKS[result['phase']]} independent ABBA blocks; 95% Student-t CI")
    figure.savefig(output / "BlockGains.png", dpi=150)
    plt.close(figure)


def report(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = verify_experiment(args.source)
        links = load_links(args, result["phase"], result["plan"])
        if links:
            check_diagnostic(result, links["diagnostic"])
        if "reference" in links:
            check_reference(result, links["reference"])
        result["links"] = link_receipts(links)
        result["comparison"] = comparisons(result["processes"], result["phase"])
        for name, key in (("Frames", "frames"), ("WorkFrames", "workFrames"),
                ("BuildDiagnosticFrames", "buildDiagnosticFrames"), ("InputFrames", "inputFrames")):
            acceptance.save(output / f"{name}.json", result[key])
            profile.write_csv(output / f"{name}.csv", result[key])
        medians = [{key: row[key] for key in ("index", "block", "arm", "postWarmupFrames")} | row["metrics"] for row in result["processes"]]
        profile.write_csv(output / "ProcessMedians.csv", medians)
        acceptance.save(output / "ProcessMedians.json", result["processes"])
        acceptance.save(output / "Decision.json", result["comparison"])
        acceptance.save(output / "Summary.json", {key: value for key, value in result.items() if key not in ("frames", "workFrames", "buildDiagnosticFrames", "inputFrames")})
        gains = result["comparison"].get("gains", {})
        profile.write_csv(output / "BlockGains.csv", [{"metric": metric, "block": block, "gain": gain, "logRatio": value["logRatios"][block]}
            for metric, value in gains.items() for block, gain in enumerate(value["values"])])
        acceptance.save(output / "BlockGains.json", gains)
        plots(result, output)
        directional = result["plan"]["protocol"] == DIRECTIONAL_PROTOCOL
        title = "directional build" if directional else "bitset"
        description = ("A: generic build; B: directional-only compiled build. All other graph settings are identical." if directional else
            "A: bitset disabled; B: enabled. Production mark and cluster culling remain enabled.")
        diagnostic_note = ("All 45 build diagnostic counters match exactly for every canonical frame and face. Actual mark/projection SPIR-V is unchanged; build SPIR-V differs." if directional else
            "Build diagnostic counters describe different page/word scanning operations and are not an output-equivalence gate.")
        lines = [f"# VSM {title} {result['phase']}", "", description, "",
            "Canonical comparison starts at frame 16 after same-frame streaming freeze and camera cut; warmup ends at 32. Startup evidence is retained but is not used for arm equivalence.", "",
            "Source, binary, sealed package, actual mark/build SPIR-V and case/graph identities were verified. Ordinary timing has validation and diagnostic readbacks off. Timing texture semantic hashes are unavailable; only published residency/input fields are compared. Diagnostic HDR/depth/shadow/PNG hashes and every canonical frame's legacy 32 statistics match exactly.", "",
            diagnostic_note + " Missing counters are unavailable, never zero. No clock control or global GPU exclusivity is certified.", ""]
        lines.extend(["The 15 stable legacy workload counters match exactly for every canonical frame and face. Executed markPageRequests atomics are excluded because shared-cache scheduling can change duplicate suppression.", ""])
        if gains:
            lines.extend([f"n = {BLOCKS[result['phase']]} independent ABBA blocks; each process contributes one post-warmup median. Confirmation uses only its own blocks; discovery is not pooled.", "",
                f"Decision: **{result['comparison']['decision']} / {result['comparison']['effectAssessment']}**; next: {result['comparison']['nextAction']}.", "",
                "| Metric | Gain | 95% CI |", "|---|---:|---:|"])
            lines.extend(f"| {metric} | {gain['estimate']:.2%} | [{gain['lower95']:.2%}, {gain['upper95']:.2%}] |" for metric, gain in gains.items())
            lines.extend(["", "Primary target: build gain lower bound ≥3%; graph gain lower bound ≥−2% in independent confirmation. Discovery only screens promising candidates.", "",
                "![Process medians](ProcessMedians.png)", "", "![Block confidence intervals](BlockGains.png)"])
        else:
            lines.extend(["Diagnostic only: exact canonical output checks passed. These instrumented timings are not performance evidence.", "", "![Diagnostic counts](DiagnosticWork.png)"])
        lines.extend(["", "Run `python Reproduce.py NEW_REPORT_DIRECTORY` to verify and recreate this report with the recorded tool versions. Source packages are never edited."])
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        hashes = tool_hashes()
        acceptance.save(output / "ReportTools.json", hashes)
        command = [str(Path(__file__).resolve()), "report", str(Path(args.source).resolve()), "{REPORT_OUTPUT}"]
        for name, value in result["links"].items():
            command.extend(["--" + name, value["source"]])
        script = "import hashlib, pathlib, subprocess, sys\n" + f"expected = {hashes!r}\n"
        script += "for path, digest in expected.items():\n    if hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() != digest:\n        raise SystemExit('Report tool changed: ' + path)\n"
        script += f"arguments = {command!r}\nsubprocess.check_call([sys.executable, '-B', *[sys.argv[1] if value == '{{REPORT_OUTPUT}}' else value for value in arguments]])\n"
        (output / "Reproduce.py").write_text(script, encoding="utf-8")
        print(json.dumps(result["comparison"], indent=2))
    except BaseException as error:
        acceptance.save(output / "Failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (output / "report.md").write_text(f"# VSM bitset evidence verification failed\n\n{type(error).__name__}: {error}\n\nNo performance conclusion was produced. Source packages were preserved.\n", encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--phase", choices=tuple(BLOCKS), required=True)
    execute.add_argument("--experiment", choices=("cluster-bitset", "directional-build"), default="cluster-bitset")
    execute.add_argument("--cases-dir", type=Path, required=True)
    execute.add_argument("--exe", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("report")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("output", type=Path)
    for command in (execute, analyze):
        command.add_argument("--diagnostic", type=Path)
        command.add_argument("--reference", type=Path)
    args = parser.parse_args()
    try:
        (run if args.command == "run" else report)(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
