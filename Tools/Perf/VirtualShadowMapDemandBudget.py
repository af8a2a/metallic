"""Demand-aware multi-light VSM allocation: sealed correctness and fixed-budget A/B.

prepare --output NEW_CASES
run --phase diagnostic|timing --cases-dir CASES --exe EXE --output NEW_PACKAGE
    [--diagnostic DIAGNOSTIC_PACKAGE]
verify SOURCE [--diagnostic DIAGNOSTIC_PACKAGE]
report SOURCE NEW_REPORT [--diagnostic DIAGNOSTIC_PACKAGE]

Diagnostic uses five processes: Slack Equal/Demand and Pressure Equal/Demand/Reference.
Reference has ample pages and is an image reference, not an independent geometry oracle.
Timing uses exactly three ABBA blocks per workload (24 independent processes total).
Runtime demand feedback is part of the candidate, including its ordinary GPU copy cost.
Protocol v2 retains an already sufficient physical layout; only pressure redistributes pages.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

import ExperimentRunner as experiment
import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapGPUProfile as profile
import VirtualShadowMapLODSelfShadow as images
import VirtualShadowMapOptimization as optimization


ROOT = acceptance.ROOT
PROTOCOL = "virtual-shadow-map-demand-budget-v2"
TEMPLATE = ROOT / "build/vsm-one-pass-projection/Mixed/Cases/Diagnostic-B.json"
TEMPLATE_CASE = TEMPLATE.with_name("Diagnostic-BCases.json")
SCENE = ROOT / "Asset/LookDev/OpenPbrDefault/OpenPbrVSMQuality.metallic_scene.json"
CAMERA = dict(eye=[0., 1.0478609688580036, 3.6447348222404337],
              center=[-.05485051870346069, 1.0478609688580036, -.06448951363563538],
              up=[0., 1., 0.], fovDegrees=60., projection="perspective", znear=.05, zfar=100.)
WORKLOADS = {"Slack": 2048, "Pressure": 256}
ARMS = ("Equal", "Demand")
BLOCK = ("Equal", "Demand", "Demand", "Equal")
METRICS = (*optimization.METRICS, "otherShadowsMs")
POLICY = {"blocksPerWorkload": 3, "frames": 96, "warmupFrames": 32,
          "confidence": .95, "statisticalUnit": "independent four-process ABBA block",
          "timingPurpose": "cost measurement, not a mandatory speedup gate",
          "slackGate": "all requests satisfied, exact native shadow/HDR, identical per-face allocation and atlas bytes",
          "pressureGate": "more admitted fine pages, fewer rejected pages, no worse per-light MAE against ample-budget reference",
          "promotion": "manual; default remains unchanged", "spreadWarning": .10}
COUNTERS = {"requestedFine": 0, "admittedFine": 1, "rejectedFine": 2,
            "coarseReceivers": 7, "assignedPages": 12, "ancestorReceivers": 13,
            "dirtyPages": 16, "cachedPages": 17, "newPhysicalPages": 20,
            "coarseRedraws": 21, "cacheEpoch": 23, "residentPages": 24, "filteredCoarseTaps": 31}


def schedule(phase):
    if phase not in ("diagnostic", "timing"):
        raise ValueError("Expected diagnostic or timing phase")
    order = [(w, None, a) for w in WORKLOADS for a in (*ARMS, *(("Reference",) if w == "Pressure" else ()))] \
        if phase == "diagnostic" else [(w, b, a) for w in WORKLOADS for b in range(3) for a in BLOCK]
    return [dict(index=i, workload=w, block=b, arm=a, package=f"process-{i + 1:02d}-{w}-{a}")
            for i, (w, b, a) in enumerate(order)]


def tool_hashes():
    return {str(Path(module.__file__).resolve()): acceptance.digest(module.__file__)
            for module in (sys.modules[__name__], acceptance, profile, images, optimization, experiment)}


def asset_manifest():
    scene = acceptance.read(SCENE)
    gltf = (SCENE.parent / scene["source"]).resolve()
    asset = acceptance.read(gltf)
    files = {SCENE, gltf, (SCENE.parent / scene["world"]["environment"]["path"]).resolve()}
    for group in ("buffers", "images"):
        for entry in asset.get(group, []):
            uri = entry.get("uri", "")
            if uri and not uri.startswith("data:"):
                files.add((gltf.parent / uri).resolve())
    return {str(path.relative_to(ROOT)): acceptance.digest(path) for path in sorted(files)}


def normalize(spec, graph, phase, workload, arm):
    spec, graph = copy.deepcopy(spec), copy.deepcopy(graph)
    if spec.get("mode") != phase or len(spec.get("cases", [])) != 1:
        raise ValueError("One case in the declared phase is required")
    case = spec["cases"][0]
    if (spec.get("frames") != (64 if phase == "diagnostic" else 96) or spec.get("warmupFrames") != 32
            or case.get("width") != 1920 or case.get("height") != 1080
            or case.get("captureFrames") != ([32, 63] if phase == "diagnostic" else [])
            or case.get("comparisonFrames") or case.get("temporalJitter") is not False
            or case.get("comparisonStartFrame") != 16 or case.get("freezeStreamingFrame") != 16
            or case.get("events") != [dict(frame=16, cameraCut=True)] or case.get("camera") != CAMERA
            or graph.get("view", {}).get("camera") != CAMERA):
        raise ValueError("Changed fixed frame, resolution, capture or history budget")
    if phase == "timing" and (spec.get("validation", False) or case.get("synchronizationValidation", False)):
        raise ValueError("Ordinary timing must disable validation")
    nodes = [n for n in graph["nodes"] if n["name"] == "Shadows"]
    if len(nodes) != 1 or nodes[0]["type"] != "VirtualShadowMapLightsPass":
        raise ValueError("Expected the production multi-light pass")
    p = nodes[0]["properties"]
    expected = dict(demandDrivenBudget=arm == "Demand", physicalPageBudget=2048 if arm == "Reference" else WORKLOADS[workload],
                    benchmarkGPUProfile=True, benchmarkGPUWork=phase == "diagnostic", benchmarkCPUProfile=False,
                    benchmarkBuildSubstages=False, compactOutputs=True, debugOutput=False, cacheEnabled=True,
                    smrtTemporalEnabled=False, onePassProjection=True, benchmarkOnePassProjection=True,
                    shadowLOD=False, shadowLODSmartBias=False, forceRedraw=False)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected allocator, cache, projection or instrumentation settings")
    if case.get("scene") != SCENE.relative_to(ROOT).as_posix() or case.get("id") != f"MixedDemandBudget{workload}":
        raise ValueError("Changed fixed scene identity")
    if p.get("filter") != "smrt" or p.get("smrtRayCount") != 2 or p.get("smrtSamplesPerRay") != 16:
        raise ValueError("Expected fixed SMRT 2 x 16 filtering")
    for key in ("demandDrivenBudget", "physicalPageBudget", "benchmarkGPUWork"):
        p.pop(key)
    for node in graph["nodes"]:
        if node["name"] == "VBuffer":
            node["properties"].pop("benchmarkCaptureSemanticSource", None)
    for key in ("mode", "frames", "validation"):
        spec.pop(key, None)
    for key in ("captureFrames", "comparisonFrames", "synchronizationValidation"):
        case.pop(key, None)
    graph["outputs"] = [output for output in graph.get("outputs", []) if output != case["comparisonOutput"]]
    if case["output"] not in graph["outputs"]:
        graph["outputs"].append(case["output"])
    case["graph"] = graph
    return spec


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    template, spec_template = acceptance.read(TEMPLATE), acceptance.read(TEMPLATE_CASE)
    catalog = []
    for phase in ("diagnostic", "timing"):
        for workload in WORKLOADS:
            for arm in (*ARMS, *(("Reference",) if phase == "diagnostic" and workload == "Pressure" else ())):
                name = f"{phase.title()}-{workload}-{arm}"
                graph, spec = copy.deepcopy(template), copy.deepcopy(spec_template)
                graph["name"] = "Mixed light demand budget"
                p = next(n["properties"] for n in graph["nodes"] if n["name"] == "Shadows")
                p.update(demandDrivenBudget=arm == "Demand", physicalPageBudget=2048 if arm == "Reference" else WORKLOADS[workload],
                         benchmarkGPUProfile=True, benchmarkGPUWork=phase == "diagnostic", benchmarkCPUProfile=False,
                         benchmarkBuildSubstages=False, compactOutputs=True, debugOutput=False, cacheEnabled=True,
                         smrtTemporalEnabled=False, onePassProjection=True, benchmarkOnePassProjection=True,
                         shadowLOD=False, shadowLODSmartBias=False, forceRedraw=False)
                next(n["properties"] for n in graph["nodes"] if n["name"] == "VBuffer")["benchmarkCaptureSemanticSource"] = phase == "diagnostic"
                spec.update(mode=phase, frames=64 if phase == "diagnostic" else 96, warmupFrames=32,
                            validation=phase == "diagnostic")
                spec["cases"][0].update(id=f"MixedDemandBudget{workload}", graph=f"{name}.json",
                                       captureFrames=[32, 63] if phase == "diagnostic" else [], comparisonFrames=[],
                                       synchronizationValidation=phase == "diagnostic")
                normalize(spec, graph, phase, workload, arm)
                acceptance.save(output / f"{name}.json", graph)
                acceptance.save(output / f"{name}Cases.json", spec)
                catalog.append(f"{name}Cases.json")
    acceptance.save(output / "Assets.json", asset_manifest())
    acceptance.save(output / "Preparation.json", dict(protocol=PROTOCOL, policy=POLICY,
        template={str(p): acceptance.digest(p) for p in (TEMPLATE, TEMPLATE_CASE)},
        assetScope="Authored scene, glTF, all referenced external buffers/images and environment HDR",
        gpuExecuted=False, cases=catalog))
    print(output)


def shader_identity(log, phase):
    result = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[VSMShader]" not in line and "[VSMOnePassShader]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        if fields.get("work") != str(phase == "diagnostic").lower():
            raise ValueError("Shader work instrumentation differs from the phase")
        key = f"{fields['entry']}:{fields.get('lightType', 'all')}"
        value = fields.get("spirvSHA256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", value) or key in result and result[key] != value:
            raise ValueError("Missing or unstable logged SPIR-V identity")
        result[key] = value
    if not all(any(k.startswith(entry + ":") for k in result) for entry in
               ("vsmMarkMain", "vsmBuildMain", "vsmProjectOnePassMain")):
        raise ValueError("Missing mixed-light Mark, Build or one-pass projection program")
    return result


def seal(directory):
    acceptance.save(directory / "Manifest.json", {str(p.relative_to(directory)): acceptance.digest(p)
                    for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "Manifest.json"})


def check_seal(directory):
    actual = {str(p.relative_to(directory)): acceptance.digest(p)
              for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "Manifest.json"}
    if acceptance.read(directory / "Manifest.json") != actual:
        raise ValueError("Sealed evidence file set or contents changed")


def counter_frame(frame):
    shadow = frame["shadow"]
    header, entries, values = shadow["header"], shadow["entries"], shadow["statistics"]
    if header["budgetDroppedCount"] or len(entries) != header["count"]:
        raise ValueError("Fixture must retain all configured coarse-light admissions")
    faces, assigned = [], 0
    for entry in entries:
        if entry["status"] != 1:
            continue
        total = 0
        for face in range(entry["faceCount"]):
            v = values[entry["index"] * 6 + face]
            if not 1 <= v[12] <= 4096 or v[1] + v[2] != v[0] or v[1] > v[12] - 1 or v[5]:
                raise ValueError("Invalid per-face budget, admission or work-overflow evidence")
            total += v[12]
            faces.append(dict(stableId=entry["stableId"], kind=entry["kind"], face=face, layer=entry["baseLayer"],
                              **{name: v[index] for name, index in COUNTERS.items()}))
        if total != entry["pageBudget"]:
            raise ValueError("Published per-light allocation differs from actual face capacities")
        assigned += total
    if assigned > header["physicalPageBudget"] or len(faces) != 9:
        raise ValueError("Invalid total allocation or changed mixed-light face count")
    memory = shadow["resourceMemory"]
    budget = shadow["budgetDiagnostics"]
    if (len(budget["header"]) != 8 or budget["header"][0] != 1 or len(budget["faces"]) != header["capacity"] * 6
            or budget["header"][2] != len(faces) or budget["header"][4] != header["physicalPageBudget"]
            or budget["header"][5] != assigned):
        raise ValueError("Invalid actual budget-manager telemetry")
    result = dict(frame=frame["frame"], warmup=frame["warmup"], faces=faces, entries=entries,
                  atlasBytes=memory[0], atlasPages=memory[5], resourceBufferBytes=memory[2])
    result.update(demandDriven=bool(budget["header"][1]), knownFaces=budget["header"][3],
                  changedFaceCount=budget["header"][6], feedbackBufferCreates=budget["header"][7], budgetFaces=budget["faces"])
    for name in COUNTERS:
        if name != "cacheEpoch":
            result[name] = sum(face[name] for face in faces)
    result["cacheEpochs"] = [face["cacheEpoch"] for face in faces]
    result["deviceLiveBytes"] = frame["memory"]["liveAllocationBytes"]
    return result


def demand_signature(row):
    return [(f["stableId"], f["face"], f["requestedFine"]) for f in row["faces"]]


def quality_gates(records):
    import numpy as np
    lookup = {(r["workload"], r["arm"]): r for r in records}
    gates, errors = {}, []
    for workload in WORKLOADS:
        a, b = (lookup[(workload, arm)] for arm in ARMS)
        for fa, fb in zip(a["counters"], b["counters"], strict=True):
            if fa["frame"] >= 32 and demand_signature(fa) != demand_signature(fb):
                errors.append(f"{workload}: canonical receiver page demand differs")
        stable = b["counters"][32:]
        if any(not row["demandDriven"] or row["knownFaces"] != 9 for row in stable):
            errors.append(f"{workload}: completed per-face demand feedback missing")
        if any(row["changedFaceCount"] or row["feedbackBufferCreates"] for row in stable):
            errors.append(f"{workload}: post-warmup budget or feedback buffer allocation churn")
        budget_vectors = [[f["assignedPages"] for f in row["faces"]] for row in stable]
        if any(v != budget_vectors[0] for v in budget_vectors[1:]):
            errors.append(f"{workload}: post-warmup allocation did not stabilize")
        if any(row["dirtyPages"] or row["coarseRedraws"] for row in stable):
            errors.append(f"{workload}: static post-warmup cache unexpectedly redraws")
        if workload == "Pressure" and not any(all(row["faces"][i]["requestedFine"] == 0 and row["faces"][i]["assignedPages"] == 1
                                                for row in stable) for i in range(9)):
            errors.append(f"{workload}: no zero-demand point face released its fine allocation")
        qa = {r["frame"]: r for r in a["captures"]}
        qb = {r["frame"]: r for r in b["captures"]}
        for frame in (32, 63):
            for resource in ("VBuffer.depth", "VBuffer.visibility"):
                if qa[frame]["images"][resource]["rawSha256"] != qb[frame]["images"][resource]["rawSha256"]:
                    errors.append(f"{workload}/{frame}: primary input changed")
        if workload == "Slack":
            exact = all(qa[f]["images"][r]["rawSha256"] == qb[f]["images"][r]["rawSha256"]
                        for f in (32, 63) for r in ("Shadows.shadows", "Deferred.color"))
            if not exact or any(r["rejectedFine"] for record in (a, b) for r in record["counters"][32:]):
                errors.append("Slack: expected all requested pages and exact native shadow/HDR")
            if not all(fb["assignedPages"] == fa["assignedPages"] and fb["atlasBytes"] == fa["atlasBytes"]
                       and [f["assignedPages"] for f in fb["faces"]] == [f["assignedPages"] for f in fa["faces"]]
                       for fa, fb in zip(a["counters"][32:], b["counters"][32:], strict=True)):
                errors.append("Slack: demand allocation changed an already sufficient physical layout")
            gates[workload] = dict(exactShadowAndHDR=exact)
        else:
            reference = lookup[(workload, "Reference")]
            qr = {r["frame"]: r for r in reference["captures"]}
            if any(r["rejectedFine"] for r in reference["counters"][32:]):
                errors.append("Pressure: ample-budget image reference still drops requested pages")
            if not all(fb["admittedFine"] > fa["admittedFine"] and fb["rejectedFine"] < fa["rejectedFine"]
                       for fa, fb in zip(a["counters"][32:], b["counters"][32:], strict=True)):
                errors.append("Pressure: no useful fine-page admission improvement")
            comparisons = []
            for frame in (32, 63):
                for resource in ("VBuffer.depth", "VBuffer.visibility"):
                    if qr[frame]["images"][resource]["rawSha256"] != qa[frame]["images"][resource]["rawSha256"]:
                        errors.append(f"Pressure/{frame}: ample reference primary input changed")
                arrays = []
                for record, capture in ((a, qa[frame]), (b, qb[frame]), (reference, qr[frame])):
                    desc = capture["images"]["Shadows.shadows"]
                    arrays.append(np.fromfile(Path(record["directory"]) / desc["file"], dtype=np.uint8)
                                  .reshape(desc["layerCount"], desc["height"], desc["width"]).astype(np.int16))
                for entry in b["counters"][frame]["entries"]:
                    if entry["status"] != 1:
                        continue
                    layer = entry["baseLayer"]
                    stats = []
                    for value in arrays[:2]:
                        delta = value[layer] - arrays[2][layer]
                        stats.append(dict(mae=float(np.abs(delta).mean() / 255.), changed=int(np.count_nonzero(delta)),
                                          darker=int(np.count_nonzero(delta < 0)), lighter=int(np.count_nonzero(delta > 0))))
                    if stats[1]["mae"] > stats[0]["mae"] + 1e-12:
                        errors.append(f"Pressure/{frame}/light {entry['index']}: reference MAE regressed")
                    comparisons.append(dict(frame=frame, light=entry["index"], stableId=entry["stableId"], equal=stats[0], demand=stats[1]))
            gates[workload] = dict(reference="Equal policy with 2048 pages; not independent geometric truth", comparisons=comparisons)
    return dict(passed=not errors, errors=sorted(set(errors)), workloads=gates)


def verify(source, diagnostic=None):
    source = Path(source).resolve()
    check_seal(source)
    plan, receipts = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    if (plan.get("protocol") != PROTOCOL or plan.get("policy") != POLICY or plan.get("processes") != schedule(plan["phase"])
            or len(receipts) != len(plan["processes"]) or not acceptance.read(source / "Completion.json")["passed"]):
        raise ValueError("Unknown protocol or incomplete fixed experiment budget")
    normalized, common_identity, common_shaders, stream_inputs, records = {}, None, None, {}, []
    for scheduled, receipt in zip(plan["processes"], receipts, strict=True):
        if any(receipt.get(k) != v for k, v in scheduled.items()) or not receipt.get("passed"):
            raise ValueError("Failed or mismatched process receipt")
        package = source / scheduled["package"]
        spec, captures = acceptance.verify_package(package)
        report = captures[0][1]
        case = report["cases"][0]
        workload, arm = scheduled["workload"], scheduled["arm"]
        key = f"{workload}-{arm}"
        info = plan["inputs"][key]
        archived_case, archived_graph = source / "Inputs" / f"{key}Cases.json", source / "Inputs" / f"{key}Graph.json"
        identity = acceptance.read(package / "Identity.json")
        if (len(captures) != 1 or spec != acceptance.read(archived_case)
                or acceptance.digest(archived_case) != info["casesSHA256"] or acceptance.digest(archived_graph) != info["graphSHA256"]
                or identity["inputs"] != {info["cases"]: info["casesSHA256"], info["graph"]: info["graphSHA256"]}
                or acceptance.digest(package / "Manifest.json") != receipt["manifestSHA256"]):
            raise ValueError("Child receipt differs from predeclared archived inputs")
        normalize(spec, acceptance.read(archived_graph), plan["phase"], workload, arm)
        current = normalize(spec, case["resolvedGraph"], plan["phase"], workload, arm)
        if workload in normalized and normalized[workload] != current:
            raise ValueError("Workload changed beyond the declared page allocation policy/budget")
        normalized[workload] = current
        current_identity = {k: identity[k] for k in ("sources", "binaries")}
        if common_identity is not None and common_identity != current_identity:
            raise ValueError("Source or binary changed between independent processes")
        common_identity = current_identity
        shaders = shader_identity(captures[0][0].parent / "Process.log", plan["phase"])
        if common_shaders is not None and common_shaders != shaders:
            raise ValueError("Actual logged shader identity differs across allocation policies")
        common_shaders = shaders
        if case["cpuProfileDiagnostics"] or case["buildSubstageDiagnostics"] or case["gpuWorkDiagnostics"] != (plan["phase"] == "diagnostic"):
            raise ValueError("Wrong instrumentation scope")
        timings = [profile.profile_frame(f) for f in case["frames"]]
        for raw, timing in zip(case["frames"], timings, strict=True):
            node = next(n for n in raw["nodes"] if n["name"] == "Shadows")
            feedback = [s for s in node["sections"] if s["name"] == "VSM budget feedback"]
            if len(feedback) != int(arm == "Demand") or any(s.get("cpuOnly", False) for s in feedback):
                raise ValueError("Actual feedback-copy GPU scope does not match the allocation policy")
            timing["feedbackMs"] = sum(profile.duration(s["gpuMs"], "VSM budget feedback") for s in feedback)
        if any(f["faceCount"] != 9 or f["projectionMode"] != "one-pass" for f in timings):
            raise ValueError("Expected nine admitted faces with actual one-pass projection")
        inputs = [optimization.stream_signature(f, plan["phase"], f["frame"] >= 16) for f in case["frames"]]
        canonical = inputs[16:]
        if workload in stream_inputs and stream_inputs[workload] != canonical:
            raise ValueError("Canonical streaming/texture input changed between processes")
        stream_inputs[workload] = canonical
        row = dict(**scheduled, directory=str(captures[0][0].parent), timings=timings,
                   metrics={m: statistics.median(f[m] for f in timings if not f["warmup"]) for m in (*METRICS, "feedbackMs")})
        if plan["phase"] == "diagnostic":
            validation = report["validationMessages"]
            if not report["validation"] or not case["synchronizationValidation"] or validation["captureFailed"] or validation["count"]:
                raise ValueError("Diagnostic synchronization validation did not pass")
            if [c["frame"] for c in case["captures"]] != [32, 63]:
                raise ValueError("Missing diagnostic captures")
            row["captures"] = [dict(frame=c["frame"], png=c["file"], images=images.raw_images(c, case, captures[0][0].parent))
                               for c in case["captures"]]
            row["counters"] = [counter_frame(f) for f in case["frames"]]
            if any(f["demandDriven"] != (arm == "Demand") for f in row["counters"]):
                raise ValueError("Actual budget-manager mode differs from the declared arm")
        elif report["validation"] or not report["performanceEligible"] or case["captures"] or case["comparisons"]:
            raise ValueError("Diagnostic work contaminated ordinary timing")
        records.append(row)
    result = dict(source=str(source), phase=plan["phase"], policy=POLICY, records=records, normalized=normalized,
                  identity=common_identity, shaders=common_shaders, tools=plan["tools"], assets=acceptance.read(source / "Assets.json"),
                  inputs=stream_inputs, planSHA256=acceptance.digest(source / "Plan.json"))
    if plan["phase"] == "diagnostic":
        result["quality"] = quality_gates(records)
        if not result["quality"]["passed"]:
            raise ValueError("Diagnostic quality gate failed: " + "; ".join(result["quality"]["errors"]))
    else:
        pinned = plan["diagnostic"]
        prior = verify(diagnostic or pinned["source"])
        if (prior["phase"] != "diagnostic" or prior["planSHA256"] != pinned["planSHA256"]
                or prior["identity"] != common_identity or prior["tools"] != plan["tools"]
                or prior["assets"] != result["assets"] or prior["normalized"] != normalized):
            raise ValueError("Timing differs from sealed diagnostic source, binary, assets or workload")
        for workload in WORKLOADS:
            for left, right in zip(prior["inputs"][workload], stream_inputs[workload]):
                left, right = copy.deepcopy(left), copy.deepcopy(right)
                for a, b in zip(left["streams"], right["streams"], strict=True):
                    if not b["textureSemanticIdentityAvailable"]:
                        for item in (a, b):
                            for key in ("textureSemanticIdentitySha256", "textureSemanticIdentityBytes", "textureSemanticIdentityAvailable"):
                                item.pop(key, None)
                if left != right:
                    raise ValueError("Diagnostic/timing canonical streaming identity differs")
        result["comparisons"] = {}
        for workload in WORKLOADS:
            rows = [r for r in records if r["workload"] == workload]
            values = [dict(result=dict(timings={m: dict(median=r["metrics"][m]) for m in METRICS})) for r in rows]
            result["comparisons"][workload] = {m: experiment.block_gain(values, m) for m in METRICS}
    return result


def run(args):
    output, cases = args.output.resolve(), args.cases_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "Inputs").mkdir()
    prior = verify(args.diagnostic) if args.phase == "timing" and args.diagnostic else None
    if args.phase == "timing" and prior is None:
        raise ValueError("Ordinary timing requires --diagnostic passed evidence")
    assets = acceptance.read(cases / "Assets.json")
    if assets != asset_manifest():
        raise ValueError("Authored scene asset contents changed since preparation")
    plan = dict(protocol=PROTOCOL, phase=args.phase, policy=POLICY, processes=schedule(args.phase),
                created=optimization.utc(), exe=str(args.exe.resolve()), inputs={}, tools=tool_hashes(),
                cache="Disk caches retained; fresh process per sample", gpuExclusivity="Not enforced; clocks unchanged",
                diagnostic=None if prior is None else {k: prior[k] for k in ("source", "planSHA256")})
    if prior is not None and prior["tools"] != plan["tools"]:
        raise ValueError("Timing must use the same declared tools as the diagnostic evidence")
    normalized = {}
    for item in plan["processes"]:
        key = f"{item['workload']}-{item['arm']}"
        if key in plan["inputs"]:
            continue
        case = cases / f"{args.phase.title()}-{key}Cases.json"
        spec = acceptance.read(case)
        graph = (case.parent / spec["cases"][0]["graph"]).resolve()
        value = normalize(spec, acceptance.read(graph), args.phase, item["workload"], item["arm"])
        if item["workload"] in normalized and normalized[item["workload"]] != value:
            raise ValueError("Planned arms differ beyond allocation policy/page budget")
        normalized[item["workload"]] = value
        plan["inputs"][key] = dict(cases=str(case), casesSHA256=acceptance.digest(case), graph=str(graph), graphSHA256=acceptance.digest(graph))
        (output / "Inputs" / f"{key}Cases.json").write_bytes(case.read_bytes())
        (output / "Inputs" / f"{key}Graph.json").write_bytes(graph.read_bytes())
    acceptance.save(output / "Plan.json", plan)
    acceptance.save(output / "Assets.json", assets)
    receipts, identity = [], None if prior is None else prior["identity"]
    try:
        for item in plan["processes"]:
            record = dict(**item, started=optimization.utc(), passed=False, gpuBefore=images.impact.telemetry())
            receipts.append(record)
            acceptance.save(output / "Processes.json", receipts)
            if tool_hashes() != plan["tools"]:
                raise ValueError("Experiment tools changed during collection")
            info = plan["inputs"][f"{item['workload']}-{item['arm']}"]
            if any(acceptance.digest(info[k]) != info[k + "SHA256"] for k in ("cases", "graph")):
                raise ValueError("Declared input changed during collection")
            package = output / item["package"]
            acceptance.run(SimpleNamespace(cases=Path(info["cases"]), exe=args.exe, output=package, runs=1, timeout=600))
            acceptance.verify_package(package)
            current = acceptance.read(package / "Identity.json")
            current = {k: current[k] for k in ("sources", "binaries")}
            if identity is not None and identity != current:
                raise ValueError("Source/binary identity changed from prior evidence")
            identity = current
            record.update(passed=True, ended=optimization.utc(), gpuAfter=images.impact.telemetry(),
                          manifestSHA256=acceptance.digest(package / "Manifest.json"))
            acceptance.save(output / "Processes.json", receipts)
        if assets != asset_manifest():
            raise ValueError("Source asset contents changed during collection")
        acceptance.save(output / "Completion.json", dict(passed=True, ended=optimization.utc()))
        seal(output)
        verify(output, args.diagnostic)
    except BaseException as error:
        acceptance.save(output / "Failure.json", dict(error=f"{type(error).__name__}: {error}", ended=optimization.utc()))
        acceptance.save(output / "Completion.json", dict(passed=False, ended=optimization.utc()))
        seal(output)
        raise
    print(output)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = verify(args.source, args.diagnostic)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    acceptance.save(output / "Summary.json", result)
    text = ["# 多灯 VSM 需求预算验证", "", "同一 OpenPBR/P4 fixture，1920×1080、SMRT 2×16、9 个面，主视图/光源固定，时间平滑关闭。",
            "需求预算只在页面压力下重分配，Slack 保留已足够的逐面物理布局；Pressure 的 Reference 单列使用充足预算，不是独立几何真值。", ""]
    if result["phase"] == "diagnostic":
        text += ["5 个诊断进程，64 帧 / 32 预热；包含 validation、统计和图像读回，不作为普通性能。",
                 "质量、需求输入、预算稳定性、闲置面释放、GPU validation 门槛全部通过。", ""]
        rows = []
        for record in result["records"]:
            for frame in record["counters"]:
                rows.append(dict(workload=record["workload"], arm=record["arm"],
                                 **{k: v for k, v in frame.items() if k not in ("faces", "entries", "cacheEpochs", "budgetFaces")}))
        save_csv(output / "Counters.csv", rows)
        fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
        for col, workload in enumerate(WORKLOADS):
            for row, field in enumerate(("assignedPages", "admittedFine", "rejectedFine", "atlasBytes")):
                axis = axes[row, col]
                for record in result["records"]:
                    if record["workload"] == workload:
                        factor = 1 / 1048576 if field == "atlasBytes" else 1
                        axis.plot([f["frame"] for f in record["counters"]], [f[field] * factor for f in record["counters"]], label=record["arm"])
                axis.axvline(32, color="gray", linestyle=":")
                axis.set(title=f"{workload}: {field}", xlabel="Frame", ylabel="MiB" if field == "atlasBytes" else "Pages")
                axis.legend()
        fig.suptitle("Diagnostic only: allocation/useful pages/memory; not GPU timing")
        fig.savefig(output / "BudgetTimeline.png", dpi=150)
        plt.close(fig)
        text += ["![预算与需求时间序列](BudgetTimeline.png)", "", "| 场景 | 策略 | 请求细页 | 准入细页 | 拒绝细页 | 分配页 | Atlas MiB |", "|---|---|---:|---:|---:|---:|---:|"]
        for record in result["records"]:
            frames = record["counters"][32:]
            v = {k: statistics.median(f[k] for f in frames) for k in ("requestedFine", "admittedFine", "rejectedFine", "assignedPages", "atlasBytes")}
            text.append(f"| {record['workload']} | {record['arm']} | {v['requestedFine']} | {v['admittedFine']} | {v['rejectedFine']} | {v['assignedPages']} | {v['atlasBytes']/1048576:.3f} |")
        acceptance.save(output / "Quality.json", result["quality"])
        text += ["", "逐灯参考 MAE、偏暗/偏亮差异见 Quality.json；这些差异不是独立真值下的真实漏影/误遮挡。"]
    else:
        rows = [dict(workload=r["workload"], arm=r["arm"], process=r["index"], block=r["block"], **r["metrics"]) for r in result["records"]]
        save_csv(output / "ProcessMedians.csv", rows)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        for row, workload in enumerate(WORKLOADS):
            for col, metric in enumerate(("shadowsMs", "graphMs", "otherShadowsMs")):
                axis = axes[row, col]
                for arm, color in (("Equal", "#4774b7"), ("Demand", "#dc8338")):
                    points = [r for r in rows if r["workload"] == workload and r["arm"] == arm]
                    axis.scatter([r["process"] + 1 for r in points], [r[metric] for r in points], label=arm, color=color)
                axis.set(title=f"{workload}: {metric}", xlabel="Independent process", ylabel="Post-warmup median (ms)")
                axis.legend()
        fig.suptitle("24 ordinary processes: 3 ABBA blocks per workload; diagnostics disabled")
        fig.savefig(output / "GPUProcessMedians.png", dpi=150)
        plt.close(fig)
        text += ["24 个普通进程，每场景 3 个完整 ABBA 区组、96 帧 / 32 预热；磁盘缓存保留，未锁 GPU 时钟或隔离其他应用。",
                 "候选的生产反馈 copy 属于实际成本；诊断和 validation 关闭。帧不是独立样本，95% CI 以三个区组 log ratio、t(df=2)=4.303 计算。",
                 "Pressure 可能增加有效页光栅工作，时间变化不等同于相同画质加速；RenderGraph 不含 present。", "", "![GPU进程分布](GPUProcessMedians.png)", "",
                 "| 场景 | GPU 范围 | Equal ms | Demand ms | Demand 变化 | 95% CI |", "|---|---|---:|---:|---:|---|"]
        comparisons = []
        for workload in WORKLOADS:
            for metric, gain in result["comparisons"][workload].items():
                medians = [statistics.median(r[metric] for r in rows if r["workload"] == workload and r["arm"] == arm) for arm in ARMS]
                change, low, high = (-gain["estimate"] * 100, -gain["upper95"] * 100, -gain["lower95"] * 100)
                text.append(f"| {workload} | {metric} | {medians[0]:.6f} | {medians[1]:.6f} | {change:+.2f}% | [{low:+.2f}%, {high:+.2f}%] |")
                comparisons.append(dict(workload=workload, metric=metric, equalMs=medians[0], demandMs=medians[1], changePercent=change, lower95=low, upper95=high))
        save_csv(output / "Comparisons.csv", comparisons)
        text += ["", "反馈复制单独时间（候选六个进程中位数的中位数，Equal 无此 scope；不计算零基线比值）："]
        for workload in WORKLOADS:
            value = statistics.median(r["feedbackMs"] for r in rows if r["workload"] == workload and r["arm"] == "Demand")
            text.append(f"- {workload}: {value:.6f} ms。")
    acceptance.save(output / "ReportTools.json", dict(python=sys.version, matplotlib=matplotlib.__version__, tools=tool_hashes()))
    text += ["", f"原始封存证据：`{result['source']}`。所有结果来自当前 verifier 对原始回执的重验。",
             "", f"重建：`python -B Tools/Perf/VirtualShadowMapDemandBudget.py report \"{result['source']}\" <new-report-directory>`。"]
    (output / "Report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--phase", choices=("diagnostic", "timing"), required=True)
    p.add_argument("--cases-dir", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic", type=Path)
    for command in ("verify", "report"):
        p = sub.add_parser(command)
        p.add_argument("source", type=Path)
        if command == "report":
            p.add_argument("output", type=Path)
        p.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.source, args.diagnostic)
        print(f"Passed: {result['phase']}, {len(result['records'])} processes")
    else:
        {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
