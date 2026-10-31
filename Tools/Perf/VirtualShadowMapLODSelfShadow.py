"""Prepare, run and verify bounded shadow LOD depth-guard GPU comparisons.

Off / Legacy / Guarded use the same resident Bunny and receiver inputs. Ordinary
timing is exactly three ABCCBA blocks per workload; diagnostic images, software
counters and same-projection cache comparisons are collected separately.
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


PROTOCOL = "virtual-shadow-map-lod-self-shadow-v1"
ARMS = {"Off": (False, True), "Legacy": (True, False), "Guarded": (True, True)}
WORKLOADS = ("Redraw", "Warm")
BLOCK = ("Off", "Legacy", "Guarded", "Guarded", "Legacy", "Off")
METRICS = impact.METRICS


def schedule(phase, workloads):
    rows = []
    for workload in workloads:
        for block in range(1 if phase == "diagnostic" else 3):
            for arm in ARMS if phase == "diagnostic" else BLOCK:
                index = len(rows)
                rows.append(dict(index=index, workload=workload, arm=arm,
                                 block=None if phase == "diagnostic" else block,
                                 package=f"process-{index + 1:02d}-{workload}-{arm}"))
    return rows


def comparison_frames(phase, workload, arm):
    return [47] if phase == "diagnostic" and workload == "Warm" and arm != "Off" else []


def prepare(args):
    assets, metrics = fixture.inspect_assets()
    template = fixture.base_graph()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for phase in ("diagnostic", "timing"):
        for workload in WORKLOADS:
            for arm, (enabled, guard) in ARMS.items():
                name = f"{phase.title()}-{workload}-{arm}"
                graph = copy.deepcopy(template)
                props = next(n["properties"] for n in graph["nodes"] if n["name"] == "Shadows")
                props.update(shadowLOD=enabled, shadowLODHierarchy=True, shadowLODDepthGuard=guard,
                             cacheEnabled=workload == "Warm", benchmarkGPUWork=phase == "diagnostic")
                acceptance.save(output / f"{name}.json", graph)
                case = dict(id=f"BunnyResident{workload}", graph=f"{name}.json",
                            scene=fixture.SCENE_PATH.as_posix(), metadataOnly=False,
                            enableStreamline=False, synchronizationValidation=phase == "diagnostic",
                            requireNoRayCapabilities=True, temporalJitter=False,
                            width=768, height=512, output="FinalBlit.color", comparisonOutput="Deferred.color",
                            camera=copy.deepcopy(fixture.CAMERA), requireShadow=True,
                            lighting=dict(sun=dict(direction=[2, 0, -1], illuminance=3., angularRadius=.00465)),
                            environment=dict(enabled=False), exposureEV100=0,
                            captureFrames=[32, 47] if phase == "diagnostic" else [],
                            comparisonFrames=comparison_frames(phase, workload, arm),
                            comparisonStartFrame=16, freezeStreamingFrame=16,
                            events=[dict(frame=16, cameraCut=True)])
                spec = dict(version=1, mode=phase, frames=48 if phase == "diagnostic" else 96,
                            warmupFrames=32, cases=[case])
                normalize(spec, graph, phase, workload, arm)
                acceptance.save(output / f"{name}Cases.json", spec)
                manifests.append(f"{name}Cases.json")
    acceptance.save(output / "Assets.json", assets)
    acceptance.save(output / "Preparation.json", dict(protocol=PROTOCOL, fixtureMetrics=metrics,
        templateGraph=dict(path=fixture.GRAPH_PATH.as_posix(), sha256=acceptance.digest(fixture.GRAPH_PATH)),
        scope="Resident full-geometry shadow source, fixed finest main view, one directional light",
        limitation="Cache-disabled redraw changes projection padding; compare cache equality with in-frame forceRedraw",
        sourceAssetsModified=False))
    acceptance.save(output / "Catalog.json", dict(caseManifests=manifests))
    print(output)


def normalize(spec, graph, phase, workload, arm):
    spec, graph = copy.deepcopy(spec), copy.deepcopy(graph)
    if spec.get("mode") != phase or len(spec.get("cases", [])) != 1:
        raise ValueError("Expected one case in the declared mode")
    case = spec["cases"][0]
    if (case.get("width") != 768 or case.get("height") != 512 or case.get("camera") != fixture.CAMERA
            or case.get("scene") != fixture.SCENE_PATH.as_posix() or case.get("temporalJitter") is not False):
        raise ValueError("The declared fixed Bunny scene, camera or resolution changed")
    if (case.get("frames", spec.get("frames")) != (48 if phase == "diagnostic" else 96)
            or case.get("warmupFrames", spec.get("warmupFrames")) != 32
            or case.get("captureFrames", []) != ([32, 47] if phase == "diagnostic" else [])
            or case.get("comparisonFrames", []) != comparison_frames(phase, workload, arm)
            or case.get("metadataOnly", False)):
        raise ValueError("The predeclared frame/capture budget or resident input changed")
    nodes = [n for n in graph["nodes"] if n.get("name") == "Shadows"]
    if len(nodes) != 1 or nodes[0]["type"] != "VirtualShadowMapLightsPass":
        raise ValueError("Expected the production Shadows Lights pass")
    props = nodes[0]["properties"]
    for node in graph["nodes"]:
        if node.get("name") in ("VBuffer", "Deferred"):
            value = node["properties"]
            if (value.get("path") != fixture.SCENE_PATH.as_posix() or value.get("sceneBinding") != "world"
                    or value.get("streamAssetPath") != fixture.STREAM_PATH.as_posix()
                    or value.get("autoBuildStreamAsset") is not False):
                raise ValueError("Shadow/main-view inputs must use the hashed resident fixture and prebuilt stream")
    enabled, guard = ARMS[arm]
    expected = dict(shadowLOD=enabled, shadowLODHierarchy=True, shadowLODDepthGuard=guard,
                    benchmarkGPUProfile=True, benchmarkGPUWork=phase == "diagnostic",
                    benchmarkCPUProfile=False, benchmarkBuildSubstages=False,
                    cacheEnabled=workload == "Warm")
    if any(props.get(k, False) is not v for k, v in expected.items()):
        raise ValueError("Unexpected LOD, guard, cache or instrumentation switches")
    if props.get("shadowLODErrorTexels") != 1.0 or props.get("forceRedraw", False):
        raise ValueError("Expected a 1-texel error budget and ordinary initial cache state")
    for key in ("shadowLOD", "shadowLODDepthGuard", "benchmarkGPUWork"):
        props.pop(key)
    for key in ("mode", "frames", "validation"):
        spec.pop(key, None)
    for key in ("frames", "captureFrames", "comparisonFrames", "synchronizationValidation"):
        case.pop(key, None)
    for node in graph["nodes"]:
        if node.get("name") == "VBuffer":
            node["properties"].pop("benchmarkCaptureSemanticSource", None)
    # Acceptance marks the HDR diagnostic output at runtime. It is absent from
    # ordinary timing and has no shader or scene-setting effect.
    if phase == "diagnostic":
        graph["outputs"] = [v for v in graph.get("outputs", []) if v != case.get("comparisonOutput")]
    case["graph"] = graph
    return spec


def raw_images(capture, case, directory):
    expected = {"Shadows.shadows": ("R8Unorm", 1), "VBuffer.depth": ("D32Sfloat", 4),
                "VBuffer.visibility": ("R32Uint", 4)}
    hdr = capture["linearHDR"]
    comparison_output = case["spec"]["comparisonOutput"]
    if hdr["format"] not in ("RGBA16Sfloat", "RGBA32Sfloat") or hdr["nonfiniteComponents"]:
        raise ValueError("Missing finite floating-point diagnostic HDR")
    expected[comparison_output] = (hdr["format"], 8 if hdr["format"] == "RGBA16Sfloat" else 16)
    images = capture.get("rawImages", {})
    if set(images) != set(expected):
        raise ValueError("Incomplete diagnostic raw-image export")
    for resource, (format_name, pixel_bytes) in expected.items():
        image = images[resource]
        filename = image["file"]
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("Diagnostic image file must be one package-local filename")
        width, height, layers = image["width"], image["height"], image["layerCount"]
        if (image["format"] != format_name or image["bytesPerPixel"] != pixel_bytes
                or width != hdr["width"] or height != hdr["height"] or layers < 1
                or (resource != "Shadows.shadows" and layers != 1)
                or image["rowStrideBytes"] != width * pixel_bytes
                or image["layerStrideBytes"] != width * height * pixel_bytes
                or image["readbackLayerStrideBytes"] < image["layerStrideBytes"]
                or image["byteLength"] != width * height * layers * pixel_bytes
                or image["texelOrder"] != "layer,y,x,component" or image["byteOrder"] != "little"):
            raise ValueError(f"Invalid raw image shape/layout: {resource}")
        path = directory / filename
        if (path.stat().st_size != image["byteLength"] or not impact.HASH.fullmatch(image["rawSha256"])
                or acceptance.digest(path) != image["rawSha256"]):
            raise ValueError(f"Raw diagnostic image content mismatch: {resource}")
    for resource, digest in (("Shadows.shadows", capture["shadowSha256"]),
                             ("VBuffer.depth", capture["depthSha256"]),
                             (comparison_output, hdr["rawSha256"])):
        if images[resource]["rawSha256"] != digest:
            raise ValueError("Exported bytes differ from the existing diagnostic hash receipt")
    if hdr["components"] != hdr["width"] * hdr["height"] * 4:
        raise ValueError("Incorrect HDR component count")
    return copy.deepcopy(images)


def diagnose(case, workload, arm, directory):
    captures = case.get("captures", [])
    if [c["frame"] for c in captures] != [32, 47]:
        raise ValueError("Missing diagnostic image checkpoints")
    images = [dict(frame=c["frame"], images=raw_images(c, case, directory)) for c in captures]
    comparisons = case.get("comparisons", [])
    if [c["frame"] for c in comparisons] != comparison_frames("diagnostic", workload, arm):
        raise ValueError("Missing declared same-projection cache comparison")
    for comparison in comparisons:
        if (comparison["status"] != "passed" or not comparison["geometryIdentityStable"]
                or not comparison["byteEqual"] or not comparison["hdrByteEqual"]
                or comparison["changedShadowBytes"] != 0):
            raise ValueError("Same-projection cache/force-redraw equivalence failed")
    rows = []
    for frame in case["frames"]:
        if frame["warmup"]:
            continue
        shadow = frame["shadow"]
        lod = shadow["lodStatistics"]
        if (len(lod) != 32 or lod[0] != 2 or lod[1] != int(ARMS[arm][0]) or lod[10] <= 0
                or any(type(v) is not int or not 0 <= v <= 0xffffffff for v in lod)):
            raise ValueError("Invalid LOD statistics v2")
        if arm == "Off" and lod[5] != lod[10]:
            raise ValueError("Off must retain the complete resident triangle capacity")
        totals = shadow["totals"]
        if len(totals) != 32 or totals[5] or totals[9] or totals[6] <= 0:
            raise ValueError("Invalid/empty shadow workload or overflow/coarse fallback")
        work, build = profile.work_frame(frame, "diagnostic"), profile.build_frame(frame, "diagnostic")
        if not work["gpuWorkAvailable"] or not build["gpuBuildDiagnosticsAvailable"]:
            raise ValueError("Missing diagnostic GPU work counters")
        rows.append(dict(frame=frame["frame"], lod=lod, work=work, build=build, totals=totals))
    return dict(directory=str(directory), images=images, rows=rows, cacheComparisons=comparisons)


def read_package(source, plan, scheduled):
    package = source / scheduled["package"]
    spec, captures = acceptance.verify_package(package)
    if len(captures) != 1:
        raise ValueError("Exactly one independent process is required per package")
    report = captures[0][1]
    case = report["cases"][0]
    workload, arm, phase = scheduled["workload"], scheduled["arm"], plan["phase"]
    key = f"{workload}-{arm}"
    planned = plan["inputs"][key]
    archived_spec, archived_graph = source / "Inputs" / f"{key}Cases.json", source / "Inputs" / f"{key}Graph.json"
    package_identity = acceptance.read(package / "Identity.json")
    if (acceptance.digest(archived_spec) != planned["casesSHA256"]
            or acceptance.digest(archived_graph) != planned["graphSHA256"]
            or spec != acceptance.read(archived_spec)
            or package_identity["inputs"] != {planned["cases"]: planned["casesSHA256"], planned["graph"]: planned["graphSHA256"]}):
        raise ValueError("Child inputs differ from the archived experiment plan")
    if plan["exe"] not in package_identity["binaries"]:
        raise ValueError("Child package did not use the planned test executable")
    normalize(spec, acceptance.read(archived_graph), phase, workload, arm)
    expected = dict(enabled=ARMS[arm][0], hierarchy=True, depthGuard=ARMS[arm][1], errorTexels=1.0)
    if any(case.get("shadowLODSettings", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Actual LOD/depth-guard settings differ from the declared arm")
    if phase == "diagnostic":
        messages = report["validationMessages"]
        if not report["validation"] or not case["synchronizationValidation"] or messages["captureFailed"] or messages["count"]:
            raise ValueError("Missing/failed Vulkan synchronization validation receipt")
    else:
        if report["validation"] or not report["performanceEligible"] or case.get("captures") or case.get("comparisons"):
            raise ValueError("Ordinary timing contains diagnostic work")
    if (case.get("gpuWorkDiagnostics") is not (phase == "diagnostic")
            or case.get("cpuProfileDiagnostics") or case.get("buildSubstageDiagnostics")):
        raise ValueError("Actual instrumentation mode differs from the experiment plan")
    frames = [impact.profile_fixture_frame(f) for f in case["frames"]]
    measured = [f for f in frames if not f["warmup"]]
    return dict(**scheduled, identity=impact.identity(package),
                shaders=impact.shader_identity(package / "run-1/Process.log", phase),
                normalized=normalize(spec, case["resolvedGraph"], phase, workload, arm),
                streams=impact.stream_input(case, phase), frames=measured,
                medians={m: statistics.median(f[m] for f in measured) for m in METRICS},
                means={m: statistics.mean(f[m] for f in measured) for m in METRICS},
                diagnostic=diagnose(case, workload, arm, package / "run-1") if phase == "diagnostic" else None)


def check_records(records, evidence=None):
    identity, shaders = records[0]["identity"], records[0]["shaders"]
    normalized, streams = {}, {}
    for row in records:
        workload = row["workload"]
        if row["identity"] != identity or row["shaders"] != shaders:
            raise ValueError("Source/binary or logged VSM SPIR-V identity changed between arms/processes")
        if workload in normalized and (normalized[workload] != row["normalized"] or streams[workload] != row["streams"]):
            raise ValueError("Workload or runtime streaming identity differs across arms/processes")
        normalized[workload], streams[workload] = row["normalized"], row["streams"]
        if evidence is not None:
            reference = next(r for r in evidence["records"] if r["workload"] == workload)
            if row["identity"] != reference["identity"] or row["normalized"] != reference["normalized"] or row["streams"] != reference["streams"]:
                raise ValueError("Ordinary timing differs from its diagnostic prerequisite")
    for workload in normalized:
        rows = [r for r in records if r["workload"] == workload and r["diagnostic"]]
        if not rows:
            continue
        reference = rows[0]["diagnostic"]["images"]
        for row in rows[1:]:
            for a, b in zip(reference, row["diagnostic"]["images"], strict=True):
                for resource in ("VBuffer.depth", "VBuffer.visibility"):
                    if a["images"][resource]["rawSha256"] != b["images"][resource]["rawSha256"]:
                        raise ValueError("LOD changed primary receiver geometry or visibility")


def verify(source, diagnostic=None):
    source = Path(source).resolve()
    impact.check_seal(source)
    plan, processes = acceptance.read(source / "Plan.json"), acceptance.read(source / "Processes.json")
    if plan["protocol"] != PROTOCOL or plan["processes"] != schedule(plan["phase"], plan["workloads"]):
        raise ValueError("Changed protocol or fixed process schedule")
    for path, digest in plan["toolsSHA256"].items():
        if acceptance.digest(source / "Tools" / Path(path).name) != digest:
            raise ValueError("Archived experiment tool differs from its declared identity")
    if not acceptance.read(source / "Completion.json")["passed"] or len(processes) != len(plan["processes"]):
        raise ValueError("Incomplete experiment cannot be accepted")
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
        if evidence["phase"] != "diagnostic" or evidence["assets"] != acceptance.read(source / "Assets.json"):
            raise ValueError("Diagnostic phase or asset identities differ")
    check_records(records, evidence)
    return dict(phase=plan["phase"], workloads=plan["workloads"], records=records,
                assets=acceptance.read(source / "Assets.json"))


def run(args):
    workloads = [args.workload] if args.workload else list(WORKLOADS)
    cases_dir, output = args.cases_dir.resolve(), args.output.resolve()
    assets = acceptance.read(cases_dir / "Assets.json")
    impact.validate_assets(assets)
    inputs, normalized = {}, {}
    for workload in workloads:
        for arm in ARMS:
            cases = cases_dir / f"{args.phase.title()}-{workload}-{arm}Cases.json"
            spec = acceptance.read(cases)
            graph = (cases.parent / spec["cases"][0]["graph"]).resolve()
            value = normalize(spec, acceptance.read(graph), args.phase, workload, arm)
            if workload in normalized and normalized[workload] != value:
                raise ValueError("Prepared inputs differ beyond LOD/depth-guard settings")
            normalized[workload] = value
            inputs[f"{workload}-{arm}"] = dict(cases=str(cases), graph=str(graph),
                casesSHA256=acceptance.digest(cases), graphSHA256=acceptance.digest(graph))
    evidence, prerequisite = None, None
    if args.phase == "timing":
        if args.diagnostic is None:
            raise ValueError("Timing requires a verified diagnostic prerequisite")
        evidence = verify(args.diagnostic)
        if evidence["phase"] != "diagnostic" or evidence["assets"] != assets or any(w not in evidence["workloads"] for w in workloads):
            raise ValueError("Diagnostic prerequisite does not cover timing inputs")
        prerequisite = dict(path=str(args.diagnostic.resolve()), manifestSHA256=acceptance.digest(args.diagnostic / "Manifest.json"))
    output.mkdir(parents=True, exist_ok=False)
    tool_paths = (Path(__file__), Path(impact.__file__), Path(fixture.__file__), Path(acceptance.__file__),
                  Path(profile.__file__), Path(impact.optimization.__file__))
    tools = {str(p.resolve()): acceptance.digest(p) for p in tool_paths}
    plan = dict(protocol=PROTOCOL, phase=args.phase, workloads=workloads, created=impact.utc(), inputs=inputs,
                processes=schedule(args.phase, workloads), diagnostic=prerequisite, exe=str(args.exe.resolve()),
                timeoutSeconds=args.timeout, toolsSHA256=tools,
                statisticalUnit="Exactly three independent symmetric ABCCBA blocks per workload; no adaptive samples",
                cache="Existing disk caches retained; fresh process per sample; first 32 frames excluded",
                gpuExclusivity="Not enforced; user applications remain running and clocks unchanged")
    acceptance.save(output / "Plan.json", plan)
    acceptance.save(output / "Assets.json", assets)
    (output / "Inputs").mkdir()
    (output / "Tools").mkdir()
    for key, item in inputs.items():
        (output / "Inputs" / f"{key}Cases.json").write_bytes(Path(item["cases"]).read_bytes())
        (output / "Inputs" / f"{key}Graph.json").write_bytes(Path(item["graph"]).read_bytes())
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
        verify(output, args.diagnostic)
    except Exception as error:
        acceptance.save(output / "Completion.json", dict(passed=False, finished=impact.utc(), error=str(error)))
        if not (output / "Processes.json").exists():
            acceptance.save(output / "Processes.json", processes)
        impact.seal(output)
        raise RuntimeError(f"Experiment stopped; evidence retained at {output}: {error}") from error
    print(output)


def block_statistics(records, workloads):
    output = []
    for workload in workloads:
        for candidate, baseline in (("Legacy", "Off"), ("Guarded", "Legacy"), ("Guarded", "Off")):
            for metric in METRICS:
                ratios = []
                for block in range(3):
                    rows = [r for r in records if r["workload"] == workload and r["block"] == block]
                    a = [r["medians"][metric] for r in rows if r["arm"] == candidate]
                    b = [r["medians"][metric] for r in rows if r["arm"] == baseline]
                    if len(a) != 2 or len(b) != 2:
                        raise ValueError("Incomplete ABCCBA block")
                    if min(*a, *b) <= 0:
                        break
                    ratios.append(statistics.mean(map(math.log, a)) - statistics.mean(map(math.log, b)))
                item = dict(workload=workload, candidate=candidate, baseline=baseline, metric=metric,
                            n=len(ratios), blockLogRatios=ratios)
                if len(ratios) == 3:
                    mean = statistics.mean(ratios)
                    delta = 4.303 * statistics.stdev(ratios) / math.sqrt(3)
                    item.update(changePercent=100 * math.expm1(mean),
                                ci95Percent=[100 * math.expm1(mean - delta), 100 * math.expm1(mean + delta)])
                else:
                    item["unavailable"] = "A zero stage duration makes the log-ratio undefined"
                output.append(item)
    return output


def report(args):
    verified = verify(args.source, args.diagnostic)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = dict(phase=verified["phase"], workloads=verified["workloads"], source=str(args.source.resolve()),
                   manifestSHA256=acceptance.digest(args.source / "Manifest.json"), matplotlib=matplotlib.__version__)
    lines = ["# Shadow LOD depth-guard A/B", "",
             "Fixed resident Bunny fixture, 768 x 512, one directional light. GPU graph/pass spans exclude presentation; no general production/FPS claim. User applications remain running and clocks are unchanged.", "",
             "Redraw disables caching and changes projection padding; Warm keeps caching. Exact cache correctness uses the in-frame forceRedraw comparisons with cacheEnabled unchanged.", ""]
    records = verified["records"]
    acceptance.save(output / "ProcessMeasurements.json", [{k: r[k] for k in ("workload", "arm", "block", "package", "medians", "means", "frames")} for r in records])
    if verified["phase"] == "diagnostic":
        acceptance.save(output / "DiagnosticReceipts.json", [{"workload": r["workload"], "arm": r["arm"], **r["diagnostic"]} for r in records])
        lines += ["Validation, raw image layouts/content and declared same-projection cache comparisons passed. Different LOD images are measured, not required to match. Raw receipts and exported files remain available for separate acne/leak/silhouette quality analysis. These instrumented timestamps are not performance evidence.", ""]
    else:
        summary["comparisons"] = block_statistics(records, verified["workloads"])
        lines += ["Exactly three independent ABCCBA blocks per workload, 96 frames/process with 32 warmup frames. Each block uses the mean log of two process medians per arm; 95% CI uses t(df=2)=4.303. Frames are not independent experiments. No samples beyond the declared warmup are excluded.", "",
                  "| State | Comparison | GPU scope | Change | 95% CI |", "| --- | --- | --- | ---: | ---: |"]
        for item in summary["comparisons"]:
            if "changePercent" in item:
                low, high = item["ci95Percent"]
                lines.append(f"| {item['workload']} | {item['candidate']}/{item['baseline']} | {item['metric']} | {item['changePercent']:+.2f}% | [{low:+.2f}, {high:+.2f}]% |")
        for workload in verified["workloads"]:
            figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
            for axis, metric in zip(axes.flat, METRICS, strict=True):
                for index, arm in enumerate(ARMS):
                    rows = [r for r in records if r["workload"] == workload and r["arm"] == arm]
                    values = [r["medians"][metric] for r in rows]
                    axis.scatter([index + (i - 2.5) * .035 for i in range(len(values))], values, alpha=.7)
                    axis.plot([index - .2, index + .2], [statistics.median(values)] * 2, color="black")
                    axis.scatter([index], [statistics.mean(r["means"][metric] for r in rows)], marker="x", color="red")
                axis.set(title=metric, xticks=list(range(3)), xticklabels=list(ARMS), ylabel="GPU ms")
                axis.grid(alpha=.2)
            figure.suptitle(f"{workload}: six process medians per arm; black=median, red=mean of process means")
            filename = f"{workload}-GPU.png"
            figure.savefig(output / filename, dpi=150)
            plt.close(figure)
            lines += ["", f"![GPU process distributions]({filename})"]
    acceptance.save(output / "Summary.json", summary)
    (output / "plot_report.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    lines += ["", "Reproduce from the maintained tool (not an untrusted archived script):", "", "```powershell",
              f'python -X utf8 -B Tools/Perf/VirtualShadowMapLODSelfShadow.py report "{args.source.resolve()}" "<new-report-directory>"', "```"]
    (output / "Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "Report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--output", type=Path, required=True)
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
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    elif args.command == "report":
        report(args)
    else:
        value = verify(args.source, args.diagnostic)
        print(f"Verified {value['phase']}: {len(value['records'])} independent processes")


if __name__ == "__main__":
    main()
