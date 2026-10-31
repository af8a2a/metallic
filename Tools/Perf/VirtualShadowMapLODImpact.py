"""Bounded, independently launched Off / Flat / Hierarchy shadow LOD GPU replay.

Diagnostic receipts and ordinary GPU timestamps are separate experiments. Timing
uses exactly three ABCCBA blocks per selected workload; no adaptive expansion.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import math
from pathlib import Path
import re
import statistics
import subprocess
from types import SimpleNamespace

import VirtualShadowMapAcceptance as acceptance
import VirtualShadowMapGPUProfile as profile
import VirtualShadowMapOptimization as optimization


PROTOCOL = "virtual-shadow-map-lod-impact-v1"
ARMS = {"Off": (False, False), "Flat": (True, False), "Hierarchy": (True, True)}
WORKLOADS = ("Redraw", "Warm")
BLOCK = ("Off", "Flat", "Hierarchy", "Hierarchy", "Flat", "Off")
METRICS = ("graphMs", "shadowsMs", *[f"{stage}Ms" for stage in profile.STAGES])
LOD_EQUAL = (0, 1, 2, *range(4, 11), *range(16, 32))
HASH = re.compile(r"[0-9a-f]{64}")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def schedule(phase, workloads):
    result = []
    for workload in workloads:
        for block in range(1 if phase == "diagnostic" else 3):
            for arm in ARMS if phase == "diagnostic" else BLOCK:
                index = len(result)
                result.append(dict(index=index, workload=workload, arm=arm,
                                   block=None if phase == "diagnostic" else block,
                                   package=f"process-{index + 1:02d}-{workload}-{arm}"))
    return result


def normalize(spec, graph, phase, workload, arm, cross_mode=False):
    """Validate the declared workload, removing only explicitly allowed differences."""
    spec, graph = copy.deepcopy(spec), copy.deepcopy(graph)
    if spec.get("mode") != phase or len(spec.get("cases", [])) != 1:
        raise ValueError("Expected exactly one case in the declared mode")
    case = spec["cases"][0]
    if case.get("frames", spec.get("frames")) != (48 if phase == "diagnostic" else 96):
        raise ValueError("The predeclared frame budget must not be changed")
    if case.get("warmupFrames", spec.get("warmupFrames")) != 32:
        raise ValueError("Exactly 32 warmup frames are required")
    if case.get("captureFrames", []) != ([32, 47] if phase == "diagnostic" else []):
        raise ValueError("Unexpected capture checkpoints")
    if case.get("comparisonFrames", []) or case.get("metadataOnly", False):
        raise ValueError("This experiment requires resident world geometry and no extra comparison frames")
    nodes = [node for node in graph["nodes"] if node.get("name") == "Shadows"]
    if len(nodes) != 1 or nodes[0]["type"] != "VirtualShadowMapLightsPass":
        raise ValueError("Expected the production Shadows Lights pass")
    props = nodes[0]["properties"]
    enabled, hierarchy = ARMS[arm]
    expected = dict(shadowLOD=enabled, shadowLODHierarchy=hierarchy,
                    benchmarkGPUProfile=True, benchmarkGPUWork=phase == "diagnostic",
                    benchmarkCPUProfile=False, benchmarkBuildSubstages=False,
                    cacheEnabled=workload == "Warm")
    if any(props.get(key, False) is not value for key, value in expected.items()):
        raise ValueError(f"Unexpected LOD, cache or instrumentation switches: {workload}/{arm}")
    if props.get("shadowLODErrorTexels") != 1.0 or props.get("forceRedraw", False):
        raise ValueError("Expected a 1-texel error budget and no forceRedraw override")
    props.pop("shadowLOD")
    props.pop("shadowLODHierarchy")
    case["graph"] = graph
    if cross_mode:
        spec.pop("mode")
        spec.pop("frames", None)
        spec.pop("validation", None)
        case.pop("frames", None)
        case.pop("captureFrames", None)
        case.pop("synchronizationValidation", None)
        props.pop("benchmarkGPUWork")
        # Acceptance adds precisely these diagnostic observations at runtime.
        for node in graph["nodes"]:
            if node.get("name") == "VBuffer":
                node["properties"].pop("benchmarkCaptureSemanticSource", None)
        if phase == "diagnostic":
            graph["outputs"] = [value for value in graph.get("outputs", [])
                                if value != case.get("comparisonOutput", "Deferred.color")]
    return spec


def validate_assets(assets):
    files = assets.get("files", [])
    if not files or len({item["path"] for item in files}) != len(files):
        raise ValueError("Expected a nonempty, distinct asset content manifest")
    for item in files:
        if not HASH.fullmatch(item["sha256"]) or acceptance.digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Asset identity changed: {item['path']}")


def telemetry():
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,utilization.gpu,"
               "clocks.current.graphics,clocks.current.memory,pstate,power.draw", "--format=csv,noheader,nounits"]
    result = dict(timestamp=utc(), command=command)
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        result.update(exitCode=process.returncode, stdout=process.stdout.strip(), stderr=process.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        result["missing"] = str(error)
    return result


def shader_identity(log, phase):
    entries = {}
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        if "[VSMShader]" not in line and "[VSMOnePassShader]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        entry, digest = fields["entry"], fields["spirvSHA256"]
        if not HASH.fullmatch(digest) or fields.get("work") != str(phase == "diagnostic").lower():
            raise ValueError("Invalid SPIR-V digest or instrumentation permutation")
        if entry in entries and entries[entry] != digest:
            raise ValueError(f"Multiple shader identities for {entry}")
        entries[entry] = digest
    if not {"vsmMarkMain", "vsmBuildMain", "vsmProjectMain"}.issubset(entries):
        raise ValueError("Missing VSM mark/build/single-face projection SPIR-V identity")
    return entries


def profile_fixture_frame(frame):
    result = profile.profile_frame(frame)
    # This fixture has one directional light face. The default one-pass option
    # intentionally takes the existing single-face projection fast path.
    if (result["faceCount"] != 1 or result["projectionCount"] != 1
            or result["projectionMode"] != "per-face"):
        raise ValueError("Expected one directional face and its single-face projection fast path")
    return result


def identity(package):
    value = acceptance.read(package / "Identity.json")
    return {key: value[key] for key in ("sources", "binaries")}


def stream_input(case, phase):
    """Freeze input metadata across measured frames, including ordinary timing.

    Diagnostic-only texture content readback is deliberately excluded here;
    diagnose() additionally compares that full receipt between diagnostic arms.
    Resident scenes may have no streaming span, or a span with zero geometry.
    Such cases use the separately hashed asset manifest instead of inventing a
    nonzero streaming identity.
    """
    reference = None
    for frame in case["frames"]:
        if frame["warmup"]:
            continue
        rows = optimization.stream_signature(frame, phase, canonical=False)["streams"]
        for row in rows:
            for key in (*optimization.STREAM_ERRORS, "residentPages", "geometryUsedBytes",
                        "originalTopologyHash", "primitiveMetadataHash"):
                profile.integer_counter(row[key], key)
            if any(row[key] != 0 for key in optimization.STREAM_ERRORS):
                raise ValueError("Pending or failed streaming work in measured frames")
            for key in ("textureSemanticIdentitySha256", "textureSemanticIdentityBytes",
                        "textureSemanticIdentityAvailable"):
                row.pop(key, None)
        if reference is not None and reference != rows:
            raise ValueError("Streaming inputs changed after warmup in the stationary workload")
        reference = rows
    if reference is None:
        raise ValueError("No post-warmup input identity receipts")
    resident = not reference or all(row["geometryUsedBytes"] == 0 for row in reference)
    return dict(streams=reference, residentEmptySpan=resident,
                basis="Hashed asset files and sampled diagnostic receiver depth" if resident else
                      "Hashed asset files and frozen runtime geometry/topology/primitive metadata")


def diagnose(case, arm):
    expected = dict(enabled=ARMS[arm][0], hierarchy=ARMS[arm][1], errorTexels=1.0)
    if case.get("shadowLODSettings") != expected:
        raise ValueError("Actual LOD properties differ from declared arm")
    captures = case.get("captures", [])
    if [c["frame"] for c in captures] != [32, 47]:
        raise ValueError("Incomplete diagnostic image checkpoints")
    images = []
    for capture in captures:
        hdr = capture["linearHDR"]
        if hdr["nonfiniteComponents"] or hdr["components"] != hdr["width"] * hdr["height"] * 4:
            raise ValueError("Invalid diagnostic HDR receipt")
        hashes = {key: capture[key] for key in ("depthSha256", "shadowSha256", "rawSha256")}
        hashes["hdrSha256"] = hdr["rawSha256"]
        if any(not HASH.fullmatch(value) for value in hashes.values()):
            raise ValueError("Invalid image hash")
        images.append(dict(frame=capture["frame"], **hashes))
    rows = []
    for frame in case["frames"]:
        if frame["warmup"]:
            continue
        shadow = frame["shadow"]
        lod = shadow["lodStatistics"]
        if len(lod) != 32 or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in lod) or lod[0] != 2:
            raise ValueError("Invalid LOD statistics v2 receipt")
        if lod[1] != int(ARMS[arm][0]) or lod[10] <= 0:
            raise ValueError("Missing baseline geometry or wrong actual LOD enable state")
        if arm == "Off" and lod[5] != lod[10]:
            raise ValueError("Disabled shadow LOD must retain the complete baseline triangle slots")
        if arm == "Flat" and (lod[12] or lod[13]):
            raise ValueError("Flat selection unexpectedly traversed/pruned hierarchy nodes")
        work = profile.work_frame(frame, "diagnostic")
        build = profile.build_frame(frame, "diagnostic")
        if not work["gpuWorkAvailable"] or not build["gpuBuildDiagnosticsAvailable"]:
            raise ValueError("Missing diagnostic GPU work counters")
        streams = optimization.stream_signature(frame, "diagnostic", canonical=False)
        for stream in streams["streams"]:
            for key in optimization.STREAM_ERRORS:
                if stream[key] != 0:
                    raise ValueError(f"Unsettled streaming input: {key}")
        rows.append(dict(frame=frame["frame"], lod=lod, work=work, build=build,
                         statistics=shadow["statistics"], totals=shadow["totals"], streams=streams))
    return dict(images=images, rows=rows,
                inputIdentity="Asset content manifest + receiver depth; empty resident streaming spans are allowed")


def compare_diagnostics(receipts, workloads):
    for workload in workloads:
        off, flat, hierarchy = (receipts[(workload, arm)] for arm in ARMS)
        if flat["images"] != hierarchy["images"]:
            raise ValueError(f"{workload}: Flat/Hierarchy image hashes differ")
        if [i["depthSha256"] for i in off["images"]] != [i["depthSha256"] for i in flat["images"]]:
            raise ValueError(f"{workload}: Off/LOD receiver depth differs")
        for a, b, c in zip(off["rows"], flat["rows"], hierarchy["rows"], strict=True):
            if a["streams"] != b["streams"] or b["streams"] != c["streams"]:
                raise ValueError(f"{workload}: geometry/texture streaming input differs between arms")
            if not a["lod"][10] == b["lod"][10] == c["lod"][10]:
                raise ValueError(f"{workload}: baseline triangle capacity differs")
            if any(b["lod"][i] != c["lod"][i] for i in LOD_EQUAL):
                raise ValueError(f"{workload}: Flat/Hierarchy selected geometry differs")
            if any(b[key] != c[key] for key in ("work", "build", "statistics", "totals")):
                raise ValueError(f"{workload}: Flat/Hierarchy GPU workload or cache statistics differ")


def seal(output):
    acceptance.save(output / "Manifest.json", {str(p.relative_to(output)): acceptance.digest(p)
                    for p in sorted(output.rglob("*")) if p.is_file() and p != output / "Manifest.json"})


def check_seal(source):
    actual = {str(p.relative_to(source)): acceptance.digest(p) for p in sorted(source.rglob("*"))
              if p.is_file() and p != source / "Manifest.json"}
    if acceptance.read(source / "Manifest.json") != actual:
        raise ValueError("LOD experiment file set or content changed")


def verify(source, diagnostic=None):
    source = Path(source).resolve()
    check_seal(source)
    plan, processes = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    if plan["protocol"] != PROTOCOL or plan["processes"] != schedule(plan["phase"], plan["workloads"]):
        raise ValueError("Unknown protocol or changed fixed process budget")
    if not acceptance.read(source / "Completion.json")["passed"] or len(processes) != len(plan["processes"]):
        raise ValueError("Incomplete experiment is evidence of failure, not accepted performance")
    common_identity, shaders, normalized, receipts, results, stream_inputs = None, None, {}, {}, [], {}
    for scheduled, process in zip(plan["processes"], processes, strict=True):
        if any(process.get(key) != value for key, value in scheduled.items()) or not process.get("passed"):
            raise ValueError("Process result does not match the fixed schedule")
        package = source / scheduled["package"]
        spec, captures = acceptance.verify_package(package)
        if len(captures) != 1:
            raise ValueError("One independent process is required per package")
        case = captures[0][1]["cases"][0]
        key = f"{scheduled['workload']}-{scheduled['arm']}"
        planned = plan["inputs"][key]
        archived_spec = source / "Inputs" / f"{key}Cases.json"
        archived_graph = source / "Inputs" / f"{key}Graph.json"
        package_identity = acceptance.read(package / "Identity.json")
        if (acceptance.digest(archived_spec) != planned["casesSHA256"]
                or acceptance.digest(archived_graph) != planned["graphSHA256"]
                or spec != acceptance.read(archived_spec)
                or package_identity["inputs"] != {planned["cases"]: planned["casesSHA256"],
                                                   planned["graph"]: planned["graphSHA256"]}):
            raise ValueError("Child package inputs differ from the archived predeclared plan")
        normalize(spec, acceptance.read(archived_graph), plan["phase"], scheduled["workload"], scheduled["arm"])
        # Compare serialized resolved graphs with each other, preserving runtime defaults.
        workload, arm = scheduled["workload"], scheduled["arm"]
        value = normalize(spec, case["resolvedGraph"], plan["phase"], workload, arm, cross_mode=True)
        if workload in normalized and normalized[workload] != value:
            raise ValueError("Resolved workload differs beyond the LOD switches")
        normalized[workload] = value
        current_streams = stream_input(case, plan["phase"])
        if workload in stream_inputs and stream_inputs[workload] != current_streams:
            raise ValueError("Measured runtime streaming inputs differ across processes or arms")
        stream_inputs[workload] = current_streams
        current = identity(package)
        if common_identity is not None and current != common_identity:
            raise ValueError("Source or binary identity changed between independent processes")
        common_identity = current
        current_shaders = shader_identity(package / "run-1/Process.log", plan["phase"])
        if shaders is not None and current_shaders != shaders:
            raise ValueError("SPIR-V identity changed within the same instrumentation mode")
        shaders = current_shaders
        frames = [profile_fixture_frame(frame) for frame in case["frames"] if not frame["warmup"]]
        expected_lod = dict(enabled=ARMS[arm][0], hierarchy=ARMS[arm][1], errorTexels=1.0)
        if case.get("shadowLODSettings") != expected_lod:
            raise ValueError("Actual shadow LOD settings mismatch")
        results.append({**scheduled, "medians": {m: statistics.median(f[m] for f in frames) for m in METRICS},
                        "means": {m: statistics.mean(f[m] for f in frames) for m in METRICS}})
        if plan["phase"] == "diagnostic":
            receipts[(workload, arm)] = diagnose(case, arm)
    if plan["phase"] == "diagnostic":
        compare_diagnostics(receipts, plan["workloads"])
    else:
        diagnostic = Path(diagnostic or plan["diagnostic"]["path"]).resolve()
        if acceptance.digest(diagnostic / "Manifest.json") != plan["diagnostic"]["manifestSHA256"]:
            raise ValueError("Timing diagnostic prerequisite was replaced")
        evidence = verify(diagnostic)
        if evidence["phase"] != "diagnostic" or evidence["identity"] != common_identity:
            raise ValueError("Diagnostic and timing source/binary identities differ")
        if evidence["assets"] != acceptance.read(source / "Assets.json"):
            raise ValueError("Diagnostic and timing asset identities differ")
        if any(evidence["normalized"].get(w) != normalized[w] for w in plan["workloads"]):
            raise ValueError("Diagnostic and timing workloads differ beyond instrumentation/capture budget")
        if any(evidence["streamInputs"].get(w) != stream_inputs[w] for w in plan["workloads"]):
            raise ValueError("Diagnostic and timing runtime geometry inputs differ")
    return dict(phase=plan["phase"], workloads=plan["workloads"], identity=common_identity,
                assets=acceptance.read(source / "Assets.json"), normalized=normalized,
                shaders=shaders, processes=results, diagnosticReceipts=receipts, streamInputs=stream_inputs)


def run(args):
    workloads = [args.workload] if args.workload else list(WORKLOADS)
    cases_dir, output = args.cases_dir.resolve(), args.output.resolve()
    assets = acceptance.read(cases_dir / "Assets.json")
    validate_assets(assets)
    inputs, normalized = {}, {}
    for workload in workloads:
        for arm in ARMS:
            cases = cases_dir / f"{args.phase.title()}-{workload}-{arm}Cases.json"
            spec = acceptance.read(cases)
            graph_path = (cases.parent / spec["cases"][0]["graph"]).resolve()
            graph = acceptance.read(graph_path)
            value = normalize(spec, graph, args.phase, workload, arm)
            if workload in normalized and normalized[workload] != value:
                raise ValueError("Prepared arm workloads differ beyond shadow LOD switches")
            normalized[workload] = value
            inputs[f"{workload}-{arm}"] = dict(cases=str(cases), graph=str(graph_path),
                casesSHA256=acceptance.digest(cases), graphSHA256=acceptance.digest(graph_path))
    prerequisite = None
    if args.phase == "timing":
        if args.diagnostic is None:
            raise ValueError("Ordinary timing requires a verified --diagnostic experiment")
        evidence = verify(args.diagnostic)
        if evidence["phase"] != "diagnostic" or evidence["assets"] != assets:
            raise ValueError("Invalid diagnostic prerequisite or asset mismatch")
        if any(w not in evidence["workloads"] for w in workloads):
            raise ValueError("Diagnostic prerequisite does not cover all selected workloads")
        prerequisite = dict(path=str(args.diagnostic.resolve()), manifestSHA256=acceptance.digest(args.diagnostic / "Manifest.json"))
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(protocol=PROTOCOL, phase=args.phase, workloads=workloads, created=utc(), inputs=inputs,
                processes=schedule(args.phase, workloads), diagnostic=prerequisite, exe=str(args.exe.resolve()),
                timeoutSeconds=args.timeout, frames=48 if args.phase == "diagnostic" else 96, warmupFrames=32,
                statisticalUnit="Three predeclared independent symmetric ABCCBA blocks per workload; no adaptive samples",
                cache="Existing disk caches retained; fresh process per sample; first 32 frames excluded",
                gpuExclusivity="Not enforced; user applications are not stopped; GPU clocks are not changed",
                toolSHA256=acceptance.digest(Path(__file__)))
    acceptance.save(output / "Plan.json", plan)
    acceptance.save(output / "Assets.json", assets)
    # Keep declared graphs/specs reviewable even after the preparation directory changes.
    archived = output / "Inputs"
    archived.mkdir()
    for key, item in inputs.items():
        (archived / f"{key}Cases.json").write_bytes(Path(item["cases"]).read_bytes())
        (archived / f"{key}Graph.json").write_bytes(Path(item["graph"]).read_bytes())
    results, common, shaders, resolved, receipts, stream_inputs = [], None, None, {}, {}, {}
    try:
        for scheduled in plan["processes"]:
            validate_assets(assets)
            item = inputs[f"{scheduled['workload']}-{scheduled['arm']}"]
            if any(acceptance.digest(item[k]) != item[f"{k}SHA256"] for k in ("cases", "graph")):
                raise ValueError("Prepared case/graph input changed after plan creation")
            row = {**scheduled, "started": utc(), "beforeGPU": telemetry(), "passed": False}
            results.append(row)
            acceptance.save(output / "Processes.json", results)
            package = output / scheduled["package"]
            try:
                acceptance.run(SimpleNamespace(cases=Path(item["cases"]), exe=args.exe, output=package,
                                               runs=1, timeout=args.timeout))
                captured_spec, captures = acceptance.verify_package(package)
                case = captures[0][1]["cases"][0]
                workload, arm = scheduled["workload"], scheduled["arm"]
                actual_workload = normalize(captured_spec, case["resolvedGraph"], args.phase, workload, arm, cross_mode=True)
                if workload in resolved and resolved[workload] != actual_workload:
                    raise ValueError("Resolved workload changed between arms")
                if args.phase == "timing" and evidence["normalized"][workload] != actual_workload:
                    raise ValueError("Actual timing workload differs from the diagnostic prerequisite")
                resolved[workload] = actual_workload
                current_streams = stream_input(case, args.phase)
                if workload in stream_inputs and stream_inputs[workload] != current_streams:
                    raise ValueError("Runtime geometry input changed across arms or processes")
                if args.phase == "timing" and evidence["streamInputs"][workload] != current_streams:
                    raise ValueError("Timing runtime geometry input differs from diagnostic")
                stream_inputs[workload] = current_streams
                current_shaders = shader_identity(package / "run-1/Process.log", args.phase)
                if shaders is not None and shaders != current_shaders:
                    raise ValueError("SPIR-V changed within the same instrumentation mode")
                shaders = current_shaders
                for frame in case["frames"]:
                    profile_fixture_frame(frame)
                if args.phase == "diagnostic":
                    receipts[(workload, arm)] = diagnose(case, arm)
                    if all((workload, a) in receipts for a in ARMS):
                        compare_diagnostics(receipts, [workload])
                current = identity(package)
                if common is not None and current != common:
                    raise ValueError("Sources or binaries changed during the fixed experiment")
                if args.phase == "timing" and current != evidence["identity"]:
                    raise ValueError("Sources or binaries differ from the diagnostic prerequisite")
                common = current
                validate_assets(assets)
                row["passed"] = True
            finally:
                row.update(finished=utc(), afterGPU=telemetry())
                acceptance.save(output / "Processes.json", results)
        acceptance.save(output / "Completion.json", dict(passed=True, finished=utc()))
        seal(output)
        verify(output, args.diagnostic)
    except Exception as error:
        acceptance.save(output / "Completion.json", dict(passed=False, finished=utc(), error=str(error)))
        if not (output / "Processes.json").exists():
            acceptance.save(output / "Processes.json", results)
        seal(output)
        raise RuntimeError(f"LOD experiment stopped; evidence retained at {output}: {error}") from error
    print(output)


def block_statistics(processes, workloads):
    result = []
    for workload in workloads:
        for candidate, baseline in (("Flat", "Off"), ("Hierarchy", "Flat"), ("Hierarchy", "Off")):
            for metric in METRICS:
                ratios = []
                for block in range(3):
                    rows = [p for p in processes if p["workload"] == workload and p["block"] == block]
                    a = [p["medians"][metric] for p in rows if p["arm"] == candidate]
                    b = [p["medians"][metric] for p in rows if p["arm"] == baseline]
                    if len(a) != 2 or len(b) != 2:
                        raise ValueError("Incomplete ABCCBA block")
                    if min(*a, *b) <= 0:
                        break
                    ratios.append(statistics.mean(map(math.log, a)) - statistics.mean(map(math.log, b)))
                item = dict(workload=workload, candidate=candidate, baseline=baseline, metric=metric,
                            blockLogRatios=ratios, n=len(ratios))
                if len(ratios) == 3:
                    mean = statistics.mean(ratios)
                    delta = 4.303 * statistics.stdev(ratios) / math.sqrt(3)
                    item.update(changePercent=100 * math.expm1(mean),
                                ci95Percent=[100 * math.expm1(mean - delta), 100 * math.expm1(mean + delta)])
                else:
                    item["unavailable"] = "Zero stage durations; a log-ratio is undefined"
                result.append(item)
    return result


def report(args):
    verified = verify(args.source, args.diagnostic)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {key: value for key, value in verified.items() if key not in ("diagnosticReceipts", "normalized", "identity")}
    summary.update(source=str(args.source.resolve()), manifestSHA256=acceptance.digest(args.source / "Manifest.json"))
    lines = ["# Shadow LOD GPU impact", "",
             "Controlled resident Bunny fixture. GPU graph/pass spans exclude presentation. User applications remain running; GPU exclusivity is not enforced and clocks are unchanged.", "",
             "Diagnostic image/counter equality establishes only the sampled Flat/Hierarchy workloads. Off uses full geometry and can differ in shadow quality; only receiver depth and input identity are required to match.", ""]
    if verified["phase"] == "diagnostic":
        receipts = {f"{w}-{a}": receipt for (w, a), receipt in verified["diagnosticReceipts"].items()}
        acceptance.save(output / "DiagnosticReceipts.json", receipts)
        lines += ["All declared diagnostic checks passed. Instrumented timestamps are not performance measurements.", ""]
    else:
        summary["comparisons"] = block_statistics(verified["processes"], verified["workloads"])
        lines += ["Exactly three independent ABCCBA blocks per workload. Each block compares the mean log of two process medians per arm; 95% intervals use Student t(df=2)=4.303. Frames are not independent samples. Negative changes mean lower time; these are fixture-local estimates, not production or end-to-end speedups.", "",
                  "| Workload | Comparison | GPU scope | Change | 95% CI |", "| --- | --- | --- | ---: | ---: |"]
        for item in summary["comparisons"]:
            if "changePercent" in item:
                low, high = item["ci95Percent"]
                lines.append(f"| {item['workload']} | {item['candidate']}/{item['baseline']} | {item['metric']} | {item['changePercent']:+.2f}% | [{low:+.2f}, {high:+.2f}]% |")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summary["matplotlib"] = matplotlib.__version__
        for workload in verified["workloads"]:
            figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            for axis, metric in zip(axes.flat, METRICS, strict=True):
                for index, arm in enumerate(ARMS):
                    rows = [p for p in verified["processes"] if p["workload"] == workload and p["arm"] == arm]
                    values = [p["medians"][metric] for p in rows]
                    axis.scatter([index + (i - 2.5) * .035 for i in range(len(values))], values, alpha=.7)
                    axis.plot([index - .2, index + .2], [statistics.median(values)] * 2, color="black")
                    axis.scatter([index], [statistics.mean(p["means"][metric] for p in rows)], marker="x", color="red")
                axis.set(title=metric, xticks=list(range(3)), xticklabels=list(ARMS), ylabel="GPU ms")
                axis.grid(alpha=.2)
            figure.suptitle(f"{workload}: six process medians per arm; black=median, red=mean of process means")
            filename = f"{workload}-GPU.png"
            figure.savefig(output / filename, dpi=150)
            plt.close(figure)
            lines += ["", f"![{workload} GPU process measurements]({filename})"]
    acceptance.save(output / "Summary.json", summary)
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "Report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--cases-dir", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--phase", choices=("diagnostic", "timing"), required=True)
    run_parser.add_argument("--diagnostic", type=Path)
    run_parser.add_argument("--workload", choices=WORKLOADS)
    run_parser.add_argument("--exe", type=Path, default=acceptance.ROOT / "build-release/tests/MetallicRHITests.exe")
    run_parser.add_argument("--timeout", type=float, default=600)
    for name in ("verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        if name == "report":
            command.add_argument("output", type=Path)
        command.add_argument("--diagnostic", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "report":
        report(args)
    else:
        value = verify(args.source, args.diagnostic)
        print(f"Verified {value['phase']}: {len(value['processes'])} independent processes")


if __name__ == "__main__":
    main()
