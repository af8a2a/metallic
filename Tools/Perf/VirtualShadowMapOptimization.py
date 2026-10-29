"""Bounded four-arm VSM experiments; this is not the WorkRaster experiment protocol.

run --phase diagnostic|discovery|confirmation --cases-dir CASES --exe EXE --output NEW_DIR
report SOURCE NEW_REPORT [--diagnostic DIAGNOSTIC_SOURCE] [--reference DISCOVERY_SOURCE]
Reports verify sealed acceptance packages offline. No command prepares or edits graphs.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace

import ExperimentRunner as experiment
import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapGPUProfile as profile


PROTOCOL = "virtual-shadow-map-optimization-v1"
ARMS = {"Legacy": (False, False), "Mark": (True, False),
        "Cluster": (False, True), "Both": (True, True)}
BLOCK = ("Legacy", "Mark", "Cluster", "Both", "Both", "Cluster", "Mark", "Legacy")
BLOCKS = {"diagnostic": 0, "discovery": 2, "confirmation": 3}
SWITCHES = ("benchmarkLegacyMark", "benchmarkDisableClusterCull")
PRIMARY = {"Mark": "markMs", "Cluster": "buildMs", "Both": "shadowsMs"}
METRICS = ("graphMs", "shadowsMs", *[f"{stage}Ms" for stage in profile.STAGES])
STREAM_FIELDS = ("residentPages", "geometryUsedBytes", "originalTopologyHash", "primitiveMetadataHash",
                 "textureSemanticIdentitySha256", "textureSemanticIdentityBytes")
STREAM_ERRORS = ("pendingPages", "allocationFailures", "loadFailures", "requestOverflows")
TEXTURE_FIELDS = ("textureResidentBytes", "loadedTextures", "pendingRequests", "texturePendingBytes", "texturePendingImages")
POLICY = {"minimumGain": .03, "maximumGraphRegression": .02, "confidence": .95,
          "statisticalUnit": "independent symmetric eight-process block",
          "statistic": "ABBA log ratios via ExperimentRunner.block_gain", "promotion": "manual"}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def schedule(phase):
    if phase not in BLOCKS:
        raise ValueError("Unknown experiment phase")
    order = tuple(ARMS) if phase == "diagnostic" else BLOCK * BLOCKS[phase]
    return [{"index": index, "block": None if phase == "diagnostic" else index // 8,
             "arm": arm, "package": f"process-{index + 1:02d}-{arm}"}
            for index, arm in enumerate(order)]


def normalized_graph(graph, arm, mode):
    graph = copy.deepcopy(graph)
    nodes = [node for node in graph["nodes"] if node.get("name") == "Shadows"]
    if len(nodes) != 1 or nodes[0]["type"] != "VirtualShadowMapLightsPass":
        raise ValueError("Expected the production VSM Lights pass named Shadows")
    properties = nodes[0].get("properties", {})
    mark, cluster = ARMS[arm]
    expected = {SWITCHES[0]: not mark, SWITCHES[1]: not cluster,
                "benchmarkGPUProfile": True, "benchmarkGPUWork": mode == "diagnostic",
                "benchmarkCPUProfile": False}
    for key, value in expected.items():
        if properties.get(key, False) is not value:
            raise ValueError(f"{arm}: graph property {key} must be {value}")
    for key in SWITCHES:
        properties.pop(key, None)
    return graph


def normalized_workload(spec, resolved_graph, arm):
    if len(spec["cases"]) != 1:
        raise ValueError("One scene/device is required per process")
    result = copy.deepcopy(spec)
    result["cases"][0]["graph"] = normalized_graph(resolved_graph, arm, spec["mode"])
    return result


def comparison_start(spec):
    settings = spec["cases"][0]
    start = settings.get("comparisonStartFrame", 0)
    if type(start) is not int or start < 0:
        raise ValueError("comparisonStartFrame must be a nonnegative frame index")
    if start:
        warmup = settings.get("warmupFrames", spec.get("warmupFrames", 16))
        reset = any(event.get("frame") == start and event.get("cameraCut") is True
                    for event in settings.get("events", []))
        if start >= warmup or settings.get("freezeStreamingFrame") != start or not reset:
            raise ValueError("A nonzero comparisonStartFrame requires same-frame cameraCut and freezeStreamingFrame inside warmup")
    return start


def stream_signature(frame, mode, canonical):
    streams = frame.get("streaming", [])
    if canonical and not streams:
        raise ValueError("Missing canonical streaming input evidence")
    result = []
    for stream in streams:
        row = {key: stream.get(key) for key in ("passName", *STREAM_FIELDS, *STREAM_ERRORS)}
        row.update({key: stream[key] for key in TEXTURE_FIELDS if key in stream})
        if canonical:
            for key in (*STREAM_FIELDS, *STREAM_ERRORS):
                if key != "textureSemanticIdentitySha256":
                    profile.integer_counter(row[key], key)
            for key in TEXTURE_FIELDS:
                if key in row:
                    profile.integer_counter(row[key], key)
            if any(row[key] != 0 for key in STREAM_ERRORS):
                raise ValueError("Canonical streaming must have zero pending pages, allocation/load failures and request overflows")
            available = row["textureSemanticIdentityBytes"] > 0
            if (available and not re.fullmatch(r"[0-9a-f]{64}", row["textureSemanticIdentitySha256"])) or (
                    not available and row["textureSemanticIdentitySha256"] != ""):
                raise ValueError("Inconsistent texture semantic input identity")
            if mode == "diagnostic" and not available:
                raise ValueError("Missing diagnostic texture semantic input identity")
        row["textureSemanticIdentityAvailable"] = bool(row["textureSemanticIdentityBytes"])
        result.append(row)
    return {"frame": frame["frame"], "streams": result}


def prepare_plan(args):
    mode = "diagnostic" if args.phase == "diagnostic" else "timing"
    inputs, normalized = {}, None
    for arm in ARMS:
        cases = args.cases_dir.resolve() / f"{mode.title()}-{arm}Cases.json"
        spec = acceptance.read(cases)
        if spec["mode"] != mode or len(spec["cases"]) != 1:
            raise ValueError(f"Wrong mode or case count: {cases}")
        comparison_start(spec)
        graph = (cases.parent / spec["cases"][0]["graph"]).resolve()
        workload = normalized_workload(spec, acceptance.read(graph), arm)
        if normalized is not None and workload != normalized:
            raise ValueError("Cases/graphs differ beyond the two optimization switches")
        normalized = workload
        inputs[arm] = {"cases": str(cases), "casesSHA256": acceptance.digest(cases),
                       "graph": str(graph), "graphSHA256": acceptance.digest(graph)}
    return {"protocol": PROTOCOL, "phase": args.phase, "mode": mode, "created": utc(),
            "policy": POLICY, "blocks": BLOCKS[args.phase], "timeoutSeconds": 600,
            "runsPerPackage": 1, "exe": str(args.exe.resolve()), "inputs": inputs,
            "comparisonStartFrame": comparison_start(spec),
            "processes": schedule(args.phase), "toolSHA256": acceptance.digest(Path(__file__)),
            "cache": "Existing disk caches retained; fresh process per sample",
            "gpuExclusivity": "Not enforced; no user applications are stopped"}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        plan = prepare_plan(args)
        acceptance.save(output / "Plan.json", plan)
        identity = None
        for item in plan["processes"]:
            row = {**item, "started": utc(), "status": "running"}
            rows.append(row)
            acceptance.save(output / "Processes.json", rows)
            try:
                for paths in plan["inputs"].values():
                    for key in ("cases", "graph"):
                        if acceptance.digest(paths[key]) != paths[key + "SHA256"]:
                            raise ValueError("Declared experiment input changed after planning")
                package = output / item["package"]
                acceptance.run(SimpleNamespace(cases=Path(plan["inputs"][item["arm"]]["cases"]),
                    exe=args.exe, output=package, runs=1, timeout=600))
                acceptance.verify_package(package)
                current = acceptance.read(package / "Identity.json")
                current = {key: current[key] for key in ("sources", "binaries")}
                if identity is not None and current != identity:
                    raise ValueError("Source or binary identity changed between processes")
                identity = current
                row["status"] = "passed"
                row["manifestSHA256"] = acceptance.digest(package / "Manifest.json")
            except BaseException as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                row["ended"] = utc()
                acceptance.save(output / "Processes.json", rows)
        acceptance.save(output / "Completion.json", {"status": "passed", "ended": utc(), "processes": len(rows)})
    except BaseException as error:
        acceptance.save(output / "Failure.json", {"status": "failed", "ended": utc(),
                        "error": f"{type(error).__name__}: {error}", "completedProcesses": len(rows)})
        raise
    print(output)


def shader_identity(log, arm, mode):
    found = {entry: set() for entry in ("vsmMarkMain", "vsmBuildMain")}
    for line in log.splitlines():
        if "[VSMShader]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", line.split("[VSMShader]", 1)[1]))
        entry = fields.get("entry")
        if entry not in found:
            continue
        mark, cluster = ARMS[arm]
        for key, wanted in (("optimizeMark", mark), ("clusterCull", cluster), ("work", mode == "diagnostic")):
            if fields.get(key) != str(wanted).lower():
                raise ValueError(f"{arm}: compiled shader flag mismatch: {key}")
        value = fields.get("spirvSHA256", "")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("Missing or invalid actual SPIR-V SHA256")
        found[entry].add(value.lower())
    if any(len(values) != 1 for values in found.values()):
        raise ValueError("Missing or unstable mark/build SPIR-V identity")
    return {entry: next(iter(values)) for entry, values in found.items()}


def check_shader_pairs(processes):
    hashes = {}
    for row in processes:
        for entry, value in row["shaders"].items():
            key = row["arm"], entry
            if key in hashes and hashes[key] != value:
                raise ValueError("Actual shader hash changed within the same arm/mode/entry")
            hashes[key] = value
    for entry, unchanged, changed in (
        ("vsmMarkMain", (("Legacy", "Cluster"), ("Mark", "Both")), ("Legacy", "Mark")),
        ("vsmBuildMain", (("Legacy", "Mark"), ("Cluster", "Both")), ("Legacy", "Cluster"))):
        if any(hashes[a, entry] != hashes[b, entry] for a, b in unchanged):
            raise ValueError(f"Unrelated switch changed {entry}")
        if hashes[changed[0], entry] == hashes[changed[1], entry]:
            raise ValueError(f"Optimization did not change actual {entry} SPIR-V")


def diagnostic_signature(case, settings, start=0):
    captures = case.get("captures", [])
    expected = sorted(set(settings.get("captureFrames", [])))
    if not expected or [item["frame"] for item in captures] != expected:
        raise ValueError("Diagnostic phase requires all declared capture checkpoints exactly once")
    if not any(frame >= start for frame in expected):
        raise ValueError("Diagnostic phase requires a canonical capture checkpoint")
    receipts = []
    for item in captures:
        if item["frame"] < start:
            continue
        hdr = item["linearHDR"]
        if (hdr["output"] != settings.get("comparisonOutput", "Deferred.color") or
                hdr["format"] not in ("RGBA16Sfloat", "RGBA32Sfloat") or
                hdr["width"] <= 0 or hdr["height"] <= 0 or
                hdr["components"] != hdr["width"] * hdr["height"] * 4 or hdr["nonfiniteComponents"] != 0):
            raise ValueError("Invalid or nonfinite linear HDR checkpoint receipt")
        hashes = [item.get(key, "") for key in ("rawSha256", "depthSha256", "shadowSha256")] + [hdr.get("rawSha256", "")]
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise ValueError("Missing PNG, depth, shadow or linear HDR checkpoint hash")
        receipts.append({key: item[key] for key in ("frame", "rawSha256", "depthSha256", "shadowSha256", "linearHDR")})
    statistics_rows = []
    for frame in case["frames"]:
        if frame["frame"] < start:
            continue
        stats = frame["shadow"]["statistics"]
        if not stats or any(len(face) != 32 for face in stats):
            raise ValueError("Diagnostic phase requires the complete legacy 32-word face statistics")
        for face in stats:
            for value in face:
                profile.integer_counter(value, "legacy statistic")
        if frame["shadow"]["totals"] != [sum(face[index] for face in stats) for index in range(32)]:
            raise ValueError("Legacy face statistics disagree with totals")
        statistics_rows.append({"frame": frame["frame"], "statistics": stats})
    return {"captures": receipts, "legacyStatistics": statistics_rows}


def verify_experiment(source):
    source = Path(source).resolve()
    plan, receipts = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    phase = plan["phase"]
    expected = schedule(phase)
    if (plan.get("protocol") != PROTOCOL or plan.get("policy") != POLICY or
            plan.get("processes") != expected or plan.get("blocks") != BLOCKS[phase] or
            plan.get("timeoutSeconds") != 600 or plan.get("runsPerPackage") != 1 or len(receipts) != len(expected)):
        raise ValueError("Unknown plan, changed policy or incomplete fixed process schedule")
    mode = "diagnostic" if phase == "diagnostic" else "timing"
    if plan["mode"] != mode:
        raise ValueError("Phase/mode mismatch")
    processes, frames, work, input_frames, startup, failures = [], [], [], [], [], []
    identity, workload, signature, input_signature, previous_end = None, None, None, None, None
    for item, receipt in zip(expected, receipts):
        try:
            if any(receipt.get(key) != value for key, value in item.items()) or receipt.get("status") != "passed":
                raise ValueError("Failed, reordered or replaced process receipt")
            started, ended = (datetime.datetime.fromisoformat(receipt[key]) for key in ("started", "ended"))
            if started.tzinfo is None or ended.tzinfo is None or ended < started or (previous_end is not None and started < previous_end):
                raise ValueError("Invalid or overlapping serialized process timestamps")
            previous_end = ended
            package = source / item["package"]
            if acceptance.digest(package / "Manifest.json") != receipt["manifestSHA256"]:
                raise ValueError("Child package no longer matches its process receipt")
            spec, captures = acceptance.verify_package(package)
            start = comparison_start(spec)
            if start != plan.get("comparisonStartFrame", 0):
                raise ValueError("Canonical comparison boundary differs from the plan")
            integrity = acceptance.read(package / "Integrity.json")
            sub_processes = acceptance.read(package / "Processes.json")
            if (len(captures) != 1 or integrity["requestedRuns"] != 1 or integrity["completedRuns"] != 1 or
                    len(sub_processes) != 1 or sub_processes[0].get("timedOut")):
                raise ValueError("Expected exactly one complete independent process per package")
            capture_path, data = captures[0]
            case = data["cases"][0]
            if data["mode"] != mode:
                raise ValueError("Package mode differs from planned phase")
            current_identity = acceptance.read(package / "Identity.json")
            planned = plan["inputs"][item["arm"]]
            if (current_identity["inputs"].get(planned["cases"]) != planned["casesSHA256"] or
                    current_identity["inputs"].get(planned["graph"]) != planned["graphSHA256"]):
                raise ValueError("Child package inputs differ from the declared plan")
            current_identity = {key: current_identity[key] for key in ("sources", "binaries")}
            if not all(current_identity.values()):
                raise ValueError("Empty source or binary identity")
            if identity is not None and current_identity != identity:
                raise ValueError("Source or binary identity changed between processes")
            identity = current_identity
            current_workload = normalized_workload(spec, case["resolvedGraph"], item["arm"])
            if workload is not None and current_workload != workload:
                raise ValueError("Cases/graphs differ beyond the two optimization switches")
            workload = current_workload
            shaders = shader_identity((capture_path.parent / "Process.log").read_text(encoding="utf-8", errors="replace"), item["arm"], mode)
            rows = [profile.profile_frame(frame) for frame in case["frames"]]
            work_rows = [profile.work_frame(frame, mode) for frame in case["frames"]]
            streams = [stream_signature(frame, mode, start > 0 and frame["frame"] >= start) for frame in case["frames"]]
            canonical_streams = [frame for frame in streams if frame["frame"] >= start]
            if start > 0 and input_signature is not None and canonical_streams != input_signature:
                raise ValueError("Canonical streaming input signature differs between processes/arms")
            input_signature = canonical_streams
            input_frames.extend({"process": item["index"], "arm": item["arm"], "frame": frame["frame"],
                "comparisonEligible": frame["frame"] >= start, **stream} for frame in streams for stream in frame["streams"])
            if start:
                startup.append({"process": item["index"], "arm": item["arm"],
                    "frames": [{"frame": frame["frame"], "statistics": frame.get("shadow", {}).get("statistics"),
                                "streaming": streams[index]["streams"]}
                               for index, frame in enumerate(case["frames"]) if frame["frame"] < start],
                    "captures": [capture for capture in case.get("captures", []) if capture["frame"] < start]})
            if mode == "diagnostic":
                if not all(row["gpuWorkAvailable"] for row in work_rows):
                    raise ValueError("Missing declared diagnostic GPU workload")
                current_signature = diagnostic_signature(case, spec["cases"][0], start)
                if signature is not None and current_signature != signature:
                    raise ValueError("Cross-arm PNG/HDR/depth/shadow capture or legacy statistics/requested-page mismatch")
                signature = current_signature
            elif data.get("gpuWorkDiagnostics", False) or case.get("gpuWorkDiagnostics", False):
                raise ValueError("Instrumented counters cannot enter ordinary timings")
            valid = [row for row in rows if not row["warmup"]]
            metrics = {metric: profile.describe([profile.duration(row[metric], metric) for row in valid])["median"] for metric in METRICS}
            processes.append({**item, "started": receipt["started"], "ended": receipt["ended"],
                "capture": str(capture_path), "manifestSHA256": receipt["manifestSHA256"], "shaders": shaders,
                "postWarmupFrames": len(valid), "metrics": metrics})
            frames.extend({"process": item["index"], "arm": item["arm"], "block": item["block"], **row} for row in rows)
            work.extend({"process": item["index"], "arm": item["arm"], **row} for row in work_rows)
        except Exception as error:
            failures.append({**item, "error": f"{type(error).__name__}: {error}"})
    if failures:
        raise ValueError("Invalid process evidence: " + json.dumps(failures))
    check_shader_pairs(processes)
    return {"source": str(source), "phase": phase, "mode": mode, "planSHA256": acceptance.digest(source / "Plan.json"),
            "identity": identity, "workloadSHA256": fingerprint(workload), "workload": workload, "processes": processes,
            "frames": frames, "workFrames": work, "diagnosticReceipts": signature,
            "comparisonStartFrame": plan.get("comparisonStartFrame", 0), "canonicalInputSignatures": input_signature,
            "inputFrames": input_frames, "startupEvidence": startup,
            "exactPNGLinearHDRDepthShadowAndLegacyStatistics": mode == "diagnostic"}


def check_reference(result, reference):
    if (result["phase"] != "confirmation" or reference["phase"] != "discovery" or
            result["identity"] != reference["identity"] or result["workloadSHA256"] != reference["workloadSHA256"] or
            (result["comparisonStartFrame"] > 0 and result["canonicalInputSignatures"] != reference["canonicalInputSignatures"])):
        raise ValueError("Confirmation reference must be discovery with identical source, binary and timing workload")
    for arm in ARMS:
        current = next(row["shaders"] for row in result["processes"] if row["arm"] == arm)
        prior = next(row["shaders"] for row in reference["processes"] if row["arm"] == arm)
        if current != prior:
            raise ValueError("Actual shader identity changed between discovery and confirmation")
    return {"source": reference["source"], "planSHA256": reference["planSHA256"],
            "discoveryComparisons": comparisons(reference["processes"], "discovery")}


def scene_workload(workload):
    """Exclude only mode/readback/sample-budget fields when linking correctness evidence."""
    result = copy.deepcopy(workload)
    for key in ("mode", "validation", "synchronizationValidation", "frames", "warmupFrames"):
        result.pop(key, None)
    case = result["cases"][0]
    for key in ("captureFrames", "comparisonFrames", "synchronizationValidation", "frames", "warmupFrames"):
        case.pop(key, None)
    graph = case["graph"]
    for node in graph["nodes"]:
        if node["name"] == "Shadows":
            node["properties"].pop("benchmarkGPUWork", None)
        if node["name"] == "VBuffer":
            node["properties"].pop("benchmarkCaptureSemanticSource", None)
    comparison = case.get("comparisonOutput", "Deferred.color")
    if comparison != case.get("output", "FinalBlit.color"):
        graph["outputs"] = [output for output in graph.get("outputs", []) if output != comparison]
    return result


def comparisons(processes, phase):
    if phase == "diagnostic":
        return {}
    result = {}
    for arm, primary in PRIMARY.items():
        projected = []
        for block in range(BLOCKS[phase]):
            rows = [row for row in processes if row["block"] == block and row["arm"] in ("Legacy", arm)]
            if [row["arm"] for row in rows] != ["Legacy", arm, arm, "Legacy"]:
                raise ValueError("Incomplete symmetric ABBA projection")
            projected.extend({"result": {"timings": {metric: {"median": value} for metric, value in row["metrics"].items()}}} for row in rows)
        gains = {metric: experiment.block_gain(projected, metric) for metric in METRICS}
        target, graph = gains[primary], gains["graphMs"]
        if phase == "discovery":
            decision = "promising" if target["estimate"] >= .03 and graph["estimate"] >= -.02 else "not_promising"
        elif target["lower95"] >= .03 and graph["lower95"] >= -.02:
            decision = "performance_confirmed"
        elif graph["upper95"] < -.02:
            decision = "graph_regressed"
        elif target["upper95"] < .03:
            decision = "below_target"
        else:
            decision = "inconclusive"
        result[arm] = {"primaryMetric": primary, "decision": decision, "gains": gains}
    return result


def plots(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    processes = result["processes"]
    if result["mode"] == "diagnostic":
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        for axis, counter in zip(axes, ("markPageRequests", "buildDecodedTriangles")):
            values = [statistics.median(row[counter] for row in result["workFrames"]
                      if row["arm"] == arm and not row["warmup"]) for arm in ARMS]
            axis.bar(list(ARMS), values)
            axis.set(title=counter, ylabel="Diagnostic count / frame")
        figure.suptitle("Diagnostic only: n=1 process per arm; no timing inference")
        filename = "DiagnosticWork.png"
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)
        return [filename], matplotlib.__version__
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    for axis, metric in zip(axes, ("markMs", "buildMs", "shadowsMs", "graphMs")):
        for arm in ARMS:
            values = [row for row in processes if row["arm"] == arm]
            axis.scatter([row["index"] + 1 for row in values], [row["metrics"][metric] for row in values], label=arm)
        axis.set(ylabel=f"{metric} (ms)")
        axis.grid(alpha=.25)
    axes[0].legend(ncol=4)
    axes[0].set_title(f"{result['phase']}: independent process medians, n={BLOCKS[result['phase']]} blocks; declared warmup excluded")
    axes[-1].set_xlabel("Process execution order; eight processes per symmetric block")
    figure.savefig(output / "ProcessMedians.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for index, (arm, comparison) in enumerate(result["comparisons"].items()):
        for axis, metric in zip(axes, (comparison["primaryMetric"], "graphMs")):
            gain = comparison["gains"][metric]
            axis.errorbar(index, gain["estimate"] * 100,
                yerr=[[100 * (gain["estimate"] - gain["lower95"])], [100 * (gain["upper95"] - gain["estimate"])]],
                fmt="o", capsize=5)
            axis.scatter([index] * gain["blocks"], [value * 100 for value in gain["values"]], marker="x", alpha=.6)
    for axis, title, threshold in zip(axes, ("Primary: mark / build / Shadows", "Whole graph"), (3, -2)):
        axis.axhline(threshold, color="red", linestyle="--", label=f"Gate: {threshold}%")
        axis.axhline(0, color="grey", linewidth=.6)
        axis.set(xticks=range(3), xticklabels=list(PRIMARY), ylabel="Reduction versus Legacy (%)", title=title)
        axis.grid(alpha=.25)
        axis.legend()
    figure.suptitle(f"{result['phase']}: n={BLOCKS[result['phase']]} blocks; two-sided 95% t CI in log ratio space")
    figure.savefig(output / "BlockGains.png", dpi=150)
    plt.close(figure)
    return ["ProcessMedians.png", "BlockGains.png"], matplotlib.__version__


def report(args):
    source, output = args.source.resolve(), args.output.resolve()
    reference_path = getattr(args, "reference", None)
    sources = [source] + [path.resolve() for path in (args.diagnostic, reference_path) if path]
    if any(path == output or path in output.parents for path in sources):
        raise ValueError("Reports must be outside sealed experiment/package trees")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = verify_experiment(source)
        diagnostic = verify_experiment(args.diagnostic) if args.diagnostic else None
        if diagnostic and (diagnostic["phase"] != "diagnostic" or diagnostic["identity"] != result["identity"]):
            raise ValueError("Diagnostic evidence must have matching source/binary identity")
        if diagnostic and scene_workload(diagnostic["workload"]) != scene_workload(result["workload"]):
            raise ValueError("Diagnostic evidence has a different scene/camera/quality workload")
        if diagnostic and result["comparisonStartFrame"] > 0:
            diagnostic_inputs = {row["frame"]: row["streams"] for row in diagnostic["canonicalInputSignatures"]}
            for row in result["canonicalInputSignatures"]:
                if row["frame"] not in diagnostic_inputs:
                    continue
                observed, expected = copy.deepcopy(row["streams"]), copy.deepcopy(diagnostic_inputs[row["frame"]])
                if not all(stream["textureSemanticIdentityAvailable"] for stream in observed):
                    for stream in (*observed, *expected):
                        for key in ("textureSemanticIdentitySha256", "textureSemanticIdentityBytes", "textureSemanticIdentityAvailable"):
                            stream.pop(key, None)
                if observed != expected:
                    raise ValueError("Diagnostic canonical streaming inputs differ from ordinary timing")
        reference = verify_experiment(reference_path) if reference_path else None
        result["reference"] = check_reference(result, reference) if reference else None
        result["comparisons"] = comparisons(result["processes"], result["phase"])
        result["correctnessEvidence"] = diagnostic["source"] if diagnostic else (str(source) if result["mode"] == "diagnostic" else None)
        if diagnostic:
            result["diagnosticEvidence"] = {key: diagnostic[key] for key in ("source", "planSHA256", "diagnosticReceipts")}
        result["policy"] = POLICY
        acceptance.save(output / "Summary.json", result)
        profile.write_csv(output / "ProcessMedians.csv", [{**{key: row[key] for key in ("index", "block", "arm", "started", "ended")},
                                                          **row["metrics"]} for row in result["processes"]])
        profile.write_csv(output / "Frames.csv", result["frames"])
        profile.write_csv(output / "WorkFrames.csv", result["workFrames"])
        profile.write_csv(output / "InputFrames.csv", result["inputFrames"])
        if result["comparisons"]:
            profile.write_csv(output / "BlockGains.csv", [{"arm": arm, "metric": metric,
                **{key: gain[key] for key in ("blocks", "estimate", "lower95", "upper95")}}
                for arm, comparison in result["comparisons"].items() for metric, gain in comparison["gains"].items()])
        images, version = plots(result, output)
        reproduction = [str(Path(__file__).resolve()), "report", str(source)]
        extras = ["--diagnostic", str(args.diagnostic.resolve())] if args.diagnostic else []
        if reference_path:
            extras.extend(["--reference", str(reference_path.resolve())])
        (output / "Reproduce.py").write_text("\"\"\"python -B Reproduce.py NEW_OUTPUT\"\"\"\nimport subprocess, sys\n"
            f"subprocess.run([sys.executable, '-B', *{reproduction!r}, sys.argv[1], *{extras!r}], check=True)\n", encoding="utf-8")
        acceptance.save(output / "ReportIdentity.json", {"python": sys.version, "matplotlib": version,
                        "toolSHA256": acceptance.digest(Path(__file__)), "planSHA256": result["planSHA256"]})
        text = ["# VSM optimization experiment", "", f"Phase: **{result['phase']}**. Source: `{source}`.", "",
            ("Diagnostic: one process per arm; no inferential timing sample." if result["mode"] == "diagnostic" else
             f"n = **{BLOCKS[result['phase']]} independent blocks**, {len(result['processes'])} processes; n is not the frame count."), "",
            "Packages were verified offline, including file hashes, completeness, source/binary identity, normalized cases/graphs and actual mark/build SPIR-V hashes.",
            "Only declared warmup frames are excluded. Each process contributes one post-warmup median; frames are not independent samples.",
            "Each eight-process block projects to Legacy/candidate/candidate/Legacy. Candidates share baselines and their comparisons are correlated.",
            "Discovery screens promising candidates only; confirmation uses only its own three blocks. No samples are automatically added and no implementation is automatically promoted.",
            "Whole-graph GPU regression must not exceed 2%; the primary stage target is 3%. This is not editor FPS or end-to-end speedup.",
            "Diagnostic counters/timestamps include instrumentation and readback costs and are not pooled with ordinary timing.",
            "Assets retain the acceptance package's path-only identity limitation; GPU exclusivity and clocks are not controlled.", ""]
        start = result["comparisonStartFrame"]
        text.extend([f"Canonical comparison begins at frame **{start}** (inclusive). " +
            ("The manifest declares a camera cut and streaming freeze at that frame inside warmup. Earlier frames remain startup evidence and are not used for cross-arm equivalence."
             if start else "The default policy compares from frame zero."),
            ("Canonical per-frame streaming signatures match across processes/arms, with zero pending pages, allocation/load failures and request overflows."
             if start else "Streaming fields are retained as evidence; no additional strict streaming gate was declared for the legacy frame-zero policy."),
            "Generated asset paths and pointer-bearing diagnosticSourceIdentity are excluded; resident geometry and stable topology/primitive/texture semantic identities are compared.", ""])
        if any(not row["textureSemanticIdentityAvailable"] for row in result["inputFrames"] if row["comparisonEligible"]):
            text.extend(["Ordinary timing does not publish texture semantic hashes (empty hash/zero bytes). Those fields are unavailable, not verified texture equivalence; full texture semantic identity is checked in the separate diagnostic evidence.", ""])
        if result["mode"] == "diagnostic":
            text.extend(["Cross-arm PNG, linear HDR, depth and shadow hashes at every declared canonical checkpoint and every canonical frame's legacy 32-word face statistics (including requested pages) are exact.",
                         "This is equivalence among the four arms, not a reference-renderer quality verdict.", ""])
        else:
            text.extend([f"Correctness package: `{result['correctnessEvidence']}`." if result["correctnessEvidence"] else
                         "Correctness evidence is unavailable; performance decisions alone do not approve adoption.", "",
                         "| Arm | Primary | Reduction / 95% CI | Graph reduction / 95% CI | Decision |",
                         "| --- | --- | --- | --- | --- |"])
            for arm, comparison in result["comparisons"].items():
                def formatted(metric):
                    gain = comparison["gains"][metric]
                    return f"{gain['estimate']:.2%} [{gain['lower95']:.2%}, {gain['upper95']:.2%}]"
                text.append(f"| {arm} | {comparison['primaryMetric']} | {formatted(comparison['primaryMetric'])} | {formatted('graphMs')} | {comparison['decision']} |")
            text.append("")
        text.extend(f"![{name}]({name})\n" for name in images)
        text.extend(["Reproduce: `python -B Reproduce.py NEW_OUTPUT_DIRECTORY`.",
                     "Field mappings: `VirtualShadowMapGPUProfile.profile_frame/work_frame`; process values and frame values are retained in CSV/JSON."])
        (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    except BaseException as error:
        acceptance.save(output / "Failure.json", {"status": "failed", "source": str(source), "ended": utc(),
                        "error": f"{type(error).__name__}: {error}"})
        (output / "report.md").write_text(f"# VSM report failed\n\nEvidence was not accepted: {type(error).__name__}: {error}\n", encoding="utf-8")
        raise
    print(output / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("run")
    capture.add_argument("--phase", choices=tuple(BLOCKS), required=True)
    capture.add_argument("--cases-dir", type=Path, required=True)
    capture.add_argument("--exe", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("report")
    summary.add_argument("source", type=Path)
    summary.add_argument("output", type=Path)
    summary.add_argument("--diagnostic", type=Path)
    summary.add_argument("--reference", type=Path, help="Discovery experiment to verify against confirmation; never pooled")
    args = parser.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
