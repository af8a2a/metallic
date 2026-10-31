"""CPU-only sealed fixtures and mocked process launch; no GPU/build is started."""
import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapBitset as bitset
from TestVirtualShadowMapGPUProfile import frame, add_work, add_build
import TestVirtualShadowMapOptimization as optimization_tests

a = bitset.acceptance


def one_pass_shader_log(arm, mode, role="primary"):
    lines = []
    for kind in ((0, 1, 2) if role == "primary" else (0,)):
        for index, entry in enumerate(("vsmMarkMain", "vsmBuildMain", "vsmProjectMain")):
            flags = {"optimizeMark": True, "clusterCull": True, "clusterPageBitset": True,
                "trianglePageBitset": True, "compactBuild": False, "markSamples": kind != 0,
                "directionalMark": index == 0 and kind == 0, "directionalBuild": index == 1 and kind == 0,
                "work": mode == "diagnostic", "projectionStatistics": False}
            lines.append(f"[VSMShader] entry={entry} lightType={kind} spirvSHA256={str(kind * 3 + index + 1) * 64} " +
                " ".join(f"{key}={str(value).lower()}" for key, value in flags.items()))
    if arm == "B" and role == "primary":
        lines.append(f"[VSMOnePassShader] entry=vsmProjectOnePassMain spirvSHA256={'f' * 64} work={str(mode == 'diagnostic').lower()} projectionStatistics=false")
    return "\n".join(lines)


def graph(arm, mode, resolved=False, directional=False, compact=False, directional_mark=False, one_pass=False, mark_clipmaps=False):
    result = optimization_tests.graph("Both", mode)
    result["nodes"][0]["properties"][bitset.MARK_CLIPMAPS_SWITCH if mark_clipmaps else bitset.ONE_PASS_SWITCH if one_pass else bitset.DIRECTIONAL_MARK_SWITCH if directional_mark else bitset.COMPACT_SWITCH if compact else bitset.DIRECTIONAL_SWITCH if directional else bitset.SWITCH] = arm == "B"
    if mark_clipmaps:
        result["nodes"][0]["properties"].update(benchmarkDirectionalMark=True, clipmapCount=6)
    if resolved:
        result["outputs"] = ["FinalBlit.color"] + (["Deferred.color"] if mode == "diagnostic" else [])
    return result


def shader_log(arm, mode, directional=False, compact=False, directional_mark=False, one_pass=False, role="primary", mark_clipmaps=False):
    if one_pass:
        return one_pass_shader_log(arm, mode, role)
    result = optimization_tests.shader_log("Both", mode)
    if arm == "B":
        result = result.replace(("b" if directional_mark or mark_clipmaps else "d") * 64, "e" * 64)
    if directional_mark or mark_clipmaps:
        result += f"\n[VSMShader] entry=vsmProjectMain spirvSHA256={'f' * 64} work={str(mode == 'diagnostic').lower()}"
        return "\n".join(line + f" lightType=0 compactBuild=false directionalBuild={str('entry=vsmBuildMain ' in line).lower()} directionalMark={str((mark_clipmaps or arm == 'B') and 'entry=vsmMarkMain ' in line).lower()}" +
            (f" markCacheLevels={6 if arm == 'B' and 'entry=vsmMarkMain ' in line else 12}" if mark_clipmaps else "") for line in result.splitlines())
    if compact:
        result += f"\n[VSMShader] entry=vsmProjectMain spirvSHA256={'f' * 64} work={str(mode == 'diagnostic').lower()}"
        if arm == "B":
            result += f"\n[VSMShader] entry=vsmCompactMain spirvSHA256={'9' * 64} work={str(mode == 'diagnostic').lower()}"
        return "\n".join(line + f" compactBuild={str(arm == 'B' and any('entry=' + entry + ' ' in line for entry in ('vsmBuildMain', 'vsmCompactMain'))).lower()}"
            for line in result.splitlines())
    if directional:
        result += f"\n[VSMShader] entry=vsmProjectMain spirvSHA256={'f' * 64}"
        return "\n".join(line + f" lightType=0 directionalBuild={str(arm == 'B' and 'entry=vsmBuildMain ' in line).lower()}"
            for line in result.splitlines())
    return "\n".join(line + f" clusterPageBitset={str(arm == 'B').lower()}" for line in result.splitlines())


def spec(mode, arm="A", frames=64):
    return {"mode": mode, "frames": frames, "warmupFrames": 32, "cases": [{"id": "Scene", "graph": f"/{arm}.json",
        "captureFrames": [16, 32, 63] if mode == "diagnostic" else [], "comparisonFrames": [],
        "comparisonStartFrame": 16, "freezeStreamingFrame": 16, "events": [{"frame": 16, "cameraCut": True}]}]}


class BitsetTests(unittest.TestCase):
    seal = optimization_tests.OptimizationTests.seal
    mutate = optimization_tests.OptimizationTests.mutate

    def test_substage_timestamps_cannot_enter_ordinary_timing(self):
        candidate = graph("B", "timing", compact=True)
        candidate["nodes"][0]["properties"]["benchmarkBuildSubstages"] = True
        with self.assertRaisesRegex(ValueError, "attribution diagnostics"):
            bitset.normalized_workload(spec("timing", "B"), candidate, "B", compact=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def experiment(self, phase, gain=.2, name=None, frames=64, directional=False, compact=False, directional_mark=False, one_pass=False, role="primary", mark_clipmaps=False):
        source = self.root / (name or phase)
        source.mkdir()
        mode = "diagnostic" if phase == "diagnostic" else "timing"
        protocol = bitset.MARK_CLIPMAPS_PROTOCOL if mark_clipmaps else bitset.ONE_PASS_PROTOCOL if one_pass else bitset.DIRECTIONAL_MARK_PROTOCOL if directional_mark else bitset.COMPACT_PROTOCOL if compact else bitset.DIRECTIONAL_PROTOCOL if directional else bitset.PROTOCOL
        plan = {"protocol": protocol, "phase": phase, "mode": mode, "policy": bitset.experiment_policy(protocol, role),
            "processes": bitset.schedule(phase), "blocks": bitset.BLOCKS[phase], "runsPerPackage": 1,
            "timeoutSeconds": 600, "comparisonStartFrame": 16,
            "workload": bitset.normalized_workload(spec(mode, frames=frames), graph("A", mode, directional=directional, compact=compact, directional_mark=directional_mark, one_pass=one_pass, mark_clipmaps=mark_clipmaps), "A", directional, compact, directional_mark, one_pass, mark_clipmaps),
            "inputs": {arm: {"cases": f"/{mode}-{arm}.json", "casesSHA256": arm,
                "graph": f"/{arm}.json", "graphSHA256": arm} for arm in bitset.ARMS}}
        a.save(source / "Plan.json", plan)
        receipts = []
        epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        for item in plan["processes"]:
            package = source / item["package"]
            directory = package / "run-1"
            directory.mkdir(parents=True)
            arm = item["arm"]
            case = {"id": "Scene", "status": "passed", "resolvedGraph": graph(arm, mode, resolved=True, directional=directional, compact=compact, directional_mark=directional_mark, one_pass=one_pass, mark_clipmaps=mark_clipmaps),
                "noRayTracingAllocations": True, "noRayQueryOrRTAS": True, "frames": []}
            for index in range(frames):
                row = frame(index, faces=(9 if role == "primary" else 1) if one_pass else 1 if mark_clipmaps else 2, scale=1000 if index < 32 else (1 if arm == "A" else 1 - gain))
                if one_pass and arm == "B" and role == "primary":
                    sections = row["nodes"][0]["sections"]
                    total = sum(section["gpuMs"] for section in sections if section["name"] == "VSM projection")
                    row["nodes"][0]["sections"] = [section for section in sections if section["name"] != "VSM projection"]
                    row["nodes"][0]["sections"].append({"name": "VSM one-pass projection", "parent": 0, "cpuOnly": False, "gpuMs": total})
                row["warmup"] = index < 32
                row["streaming"] = [{"passName": "VBuffer", "residentPages": 5, "geometryUsedBytes": 1024,
                    "originalTopologyHash": 3, "primitiveMetadataHash": 4,
                    "textureSemanticIdentitySha256": "e" * 64 if mode == "diagnostic" else "",
                    "textureSemanticIdentityBytes": 12 if mode == "diagnostic" else 0, "textureResidentBytes": 4096,
                    "pendingPages": 0, "allocationFailures": 0, "loadFailures": 0, "requestOverflows": 0}]
                if mode == "diagnostic":
                    add_work(row)
                    add_build(row)
                    # Build diagnostics are valid but deliberately differ; legacy/image receipts do not.
                    if directional or compact or directional_mark or mark_clipmaps or one_pass:
                        diagnostic = row["shadow"]["gpuBuildDiagnostics"]
                        diagnostic["version"] = 3
                        for counters in [diagnostic["totals"], *[face["counters"] for face in diagnostic["faces"]]]:
                            counters.update({name: 0 for name in bitset.profile.ALL_BUILD_COUNTERS if name not in counters})
                    elif arm == "B":
                        for face in row["shadow"]["gpuBuildDiagnostics"]["faces"]:
                            face["counters"]["clusterGroups"] += 1
                        row["shadow"]["gpuBuildDiagnostics"]["totals"]["clusterGroups"] += 2
                    if one_pass or mark_clipmaps:
                        shadow = row["shadow"]
                        count = 9 if one_pass and role == "primary" else 1
                        shadow["statistics"] = [shadow["statistics"][0].copy() for _ in range(count)]
                        shadow["totals"] = [value * count for value in shadow["statistics"][0]]
                        for name in ("gpuWork", "gpuBuildDiagnostics"):
                            counters = shadow[name]["faces"][0]["counters"]
                            shadow[name]["faces"] = [{"lightIndex": i // 6, "face": i % 6, "counters": counters.copy()} for i in range(count)]
                            shadow[name]["totals"] = {key: value * count for key, value in counters.items()}
                case["frames"].append(row)
            if mode == "diagnostic":
                case["captures"] = [{"frame": index, "rawSha256": "a" * 64, "depthSha256": "b" * 64,
                    "shadowSha256": "c" * 64, "linearHDR": {"rawSha256": "d" * 64, "output": "Deferred.color",
                    "format": "RGBA16Sfloat", "width": 2, "height": 1, "components": 8, "nonfiniteComponents": 0}}
                    for index in (16, 32, 63)]
            data = {"protocol": "virtual-shadow-map-acceptance-v2", "status": "passed", "mode": mode,
                "validation": mode == "diagnostic", "performanceEligible": mode == "timing", "cases": [case]}
            a.save(directory / "VirtualShadowMapAcceptance.json", data)
            (directory / "Process.log").write_text(shader_log(arm, mode, directional, compact, directional_mark, one_pass, role, mark_clipmaps), encoding="utf-8")
            a.save(package / "Cases.json", spec(mode, arm, frames))
            paths = plan["inputs"][arm]
            a.save(package / "Identity.json", {"sources": {"source": "1" * 64}, "binaries": {"exe": "2" * 64},
                "inputs": {paths["cases"]: paths["casesSHA256"], paths["graph"]: paths["graphSHA256"]}})
            a.save(package / "Integrity.json", {"sourceAndInputsUnchanged": True, "binariesUnchanged": True,
                "requestedRuns": 1, "completedRuns": 1})
            a.save(package / "Processes.json", [{"exitCode": 0, "timedOut": False, "captureValid": True}])
            receipts.append({**item, "status": "passed", "started": (epoch + datetime.timedelta(seconds=item["index"] * 2)).isoformat(),
                "ended": (epoch + datetime.timedelta(seconds=item["index"] * 2 + 1)).isoformat()})
        a.save(source / "Processes.json", receipts)
        for index in range(len(receipts)):
            self.seal(source, index)
        return source

    def test_fixed_budget_and_production_flags(self):
        self.assertEqual(len(bitset.schedule("diagnostic")), 2)
        for phase, count in (("discovery", 8), ("confirmation", 12)):
            rows = bitset.schedule(phase)
            self.assertEqual(len(rows), count)
            self.assertEqual(tuple(row["arm"] for row in rows[:4]), bitset.BLOCK)
        for key in ("benchmarkLegacyMark", "benchmarkDisableClusterCull"):
            value = graph("A", "timing")
            value["nodes"][0]["properties"][key] = True
            with self.assertRaisesRegex(ValueError, key):
                bitset.normalized_workload(spec("timing"), value, "A")
        with self.assertRaisesRegex(ValueError, bitset.SWITCH):
            bitset.normalized_workload(spec("timing"), graph("B", "timing"), "A")

    def test_directional_switch_is_explicit_and_only_difference(self):
        for arm in bitset.ARMS:
            for invalid in (None, 0, 1, "false", "true"):
                value = graph(arm, "timing", directional=True)
                value["nodes"][0]["properties"][bitset.DIRECTIONAL_SWITCH] = invalid
                with self.assertRaisesRegex(ValueError, bitset.DIRECTIONAL_SWITCH):
                    bitset.normalized_workload(spec("timing", arm), value, arm, True)
        left = bitset.normalized_workload(spec("timing"), graph("A", "timing", directional=True), "A", True)
        right = bitset.normalized_workload(spec("timing", "B"), graph("B", "timing", directional=True), "B", True)
        self.assertEqual(left, right)

    def test_directional_mark_switch_and_production_build_contract(self):
        workloads = []
        for arm in bitset.ARMS:
            value = graph(arm, "timing", directional_mark=True)
            workloads.append(bitset.normalized_workload(spec("timing", arm), value, arm, directional_mark=True))
            for invalid in (None, 0, 1, "false", "true"):
                value["nodes"][0]["properties"][bitset.DIRECTIONAL_MARK_SWITCH] = invalid
                with self.assertRaisesRegex(ValueError, bitset.DIRECTIONAL_MARK_SWITCH):
                    bitset.normalized_workload(spec("timing", arm), value, arm, directional_mark=True)
            for key, setting in ((bitset.COMPACT_SWITCH, True), (bitset.DIRECTIONAL_SWITCH, False),
                                 ("benchmarkBuildSubstages", True)):
                value = graph(arm, "timing", directional_mark=True)
                value["nodes"][0]["properties"][key] = setting
                with self.assertRaises(ValueError):
                    bitset.normalized_workload(spec("timing", arm), value, arm, directional_mark=True)
        self.assertEqual(*workloads)

    def test_mark_clipmaps_switch_and_same_directional_workload(self):
        workloads = [bitset.normalized_workload(spec("timing", arm), graph(arm, "timing", mark_clipmaps=True),
            arm, mark_clipmaps=True) for arm in bitset.ARMS]
        self.assertEqual(*workloads)
        for arm in bitset.ARMS:
            for key, setting in ([(bitset.MARK_CLIPMAPS_SWITCH, value) for value in (None, 0, 1, "false", "true")] +
                    [("clipmapCount", value) for value in (None, True, 0, 12, 6.0, "6")] +
                    [(bitset.DIRECTIONAL_MARK_SWITCH, False), (bitset.DIRECTIONAL_SWITCH, False),
                     (bitset.COMPACT_SWITCH, True), ("benchmarkBuildSubstages", True)]):
                value = graph(arm, "timing", mark_clipmaps=True)
                value["nodes"][0]["properties"][key] = setting
                with self.subTest(arm=arm, key=key, setting=setting), self.assertRaises(ValueError):
                    bitset.normalized_workload(spec("timing", arm), value, arm, mark_clipmaps=True)
        with self.assertRaises(ValueError):
            bitset.experiment_policy(bitset.MARK_CLIPMAPS_PROTOCOL, "overhead")

    def test_mark_clipmaps_actual_levels_and_unrelated_shader_identity(self):
        rows = [{"arm": arm, "shaders": bitset.shader_identity(shader_log(arm, "timing", mark_clipmaps=True),
            arm, "timing", mark_clipmaps=True, mark_cache_levels=6)} for arm in bitset.ARMS]
        bitset.check_shaders(rows, mark_clipmaps=True)
        for entry, value in (("vsmMarkMain", rows[0]["shaders"]["vsmMarkMain"]),
                             ("vsmBuildMain", "8" * 64), ("vsmProjectMain", "8" * 64)):
            prior = rows[1]["shaders"][entry]
            rows[1]["shaders"][entry] = value
            with self.assertRaises(ValueError):
                bitset.check_shaders(rows, mark_clipmaps=True)
            rows[1]["shaders"][entry] = prior
        for arm in bitset.ARMS:
            valid = shader_log(arm, "timing", mark_clipmaps=True)
            for invalid in (valid.replace("lightType=0", "lightType=1"),
                    valid.replace("directionalMark=true", "directionalMark=false"),
                    valid.replace("directionalBuild=true", "directionalBuild=false"),
                    valid.replace("compactBuild=false", "compactBuild=true"),
                    valid.replace("work=false", "work=true"), valid.replace("markCacheLevels=", "wrongLevels="),
                    valid.replace("markCacheLevels=12", "markCacheLevels=6"),
                    valid.replace("markCacheLevels=6", "markCacheLevels=12") if arm == "B" else valid.replace("markCacheLevels=12", "markCacheLevels=11", 1),
                    "\n".join(valid.splitlines()[:-1]), shader_log(arm, "timing", directional_mark=True)):
                with self.subTest(arm=arm, log=invalid), self.assertRaises(ValueError):
                    bitset.shader_identity(invalid, arm, "timing", mark_clipmaps=True, mark_cache_levels=6)
        valid = shader_log("B", "timing", mark_clipmaps=True).replace("markCacheLevels=6", "markCacheLevels=4")
        bitset.shader_identity(valid, "B", "timing", mark_clipmaps=True, mark_cache_levels=4)
        with self.assertRaisesRegex(ValueError, "markCacheLevels"):
            bitset.shader_identity(valid, "B", "timing", mark_clipmaps=True, mark_cache_levels=6)

    def test_mark_clipmaps_exact_diagnostic_gates(self):
        source = self.experiment("diagnostic", mark_clipmaps=True)
        result = bitset.verify_experiment(source)
        self.assertEqual(result["processes"][0]["faceCount"], 1)
        receipt = result["diagnosticReceipts"]
        self.assertEqual(len(receipt["exactBuildDiagnostics"][0]["diagnostics"]["totals"]), 45)
        self.assertEqual(len(receipt["stableWorkCounters"][0]["faces"][0]["counters"]), 15)
        def change_counter(data, field, name):
            counters = data["cases"][0]["frames"][16]["shadow"][field]
            counters["faces"][0]["counters"][name] += 1
            counters["totals"][name] += 1
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda data: change_counter(data, "gpuWork", "markPageRequests"))
        bitset.verify_experiment(source)
        mutations = [lambda data: change_counter(data, "gpuWork", "markLevelTests"),
            lambda data: change_counter(data, "gpuBuildDiagnostics", "clusterGroups"),
            lambda data: data["cases"][0]["captures"][0]["linearHDR"].update(rawSha256="9" * 64)]
        mutations.extend(lambda data, key=key: data["cases"][0]["captures"][0].update({key: "9" * 64})
            for key in ("rawSha256", "depthSha256", "shadowSha256"))
        def change_page_demand(data):
            shadow = data["cases"][0]["frames"][16]["shadow"]
            shadow["statistics"][0][0] += 1
            shadow["totals"][0] += 1
        mutations.append(change_page_demand)
        for index, mutation in enumerate(mutations):
            source = self.experiment("diagnostic", name=f"bad-clipmap-output-{index}", mark_clipmaps=True)
            self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", mutation)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "mismatch after input verification"):
                bitset.verify_experiment(source)

    def test_mark_clipmaps_phases_and_primary_are_distinct(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic", mark_clipmaps=True))
        discovery = bitset.verify_experiment(self.experiment("discovery", frames=96, mark_clipmaps=True))
        confirmation = bitset.verify_experiment(self.experiment("confirmation", frames=96, mark_clipmaps=True))
        for result in (discovery, confirmation):
            for row in result["processes"]:
                row["metrics"]["buildMs"] = 1
            bitset.check_diagnostic(result, diagnostic)
            decision = bitset.comparisons(result["processes"], result["phase"], policy=result["plan"]["policy"])
            self.assertEqual(decision["primaryMetric"], "markMs")
            self.assertEqual(decision["gains"]["markMs"]["blocks"], bitset.BLOCKS[result["phase"]])
        bitset.check_reference(confirmation, discovery)
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation", policy=confirmation["plan"]["policy"])["decision"], "accept")
        for row in confirmation["processes"]:
            row["metrics"]["markMs"] = 1 if row["arm"] == "A" else 1.1
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation", policy=confirmation["plan"]["policy"])["effectAssessment"], "confirmed_regression")
        discovery["plan"]["protocol"] = bitset.DIRECTIONAL_MARK_PROTOCOL
        with self.assertRaises(ValueError):
            bitset.check_diagnostic(discovery, diagnostic)
        with self.assertRaises(ValueError):
            bitset.check_reference(confirmation, discovery)

    def test_mark_clipmaps_rejects_changed_identity_or_execution(self):
        for field in ("sources", "binaries"):
            source = self.experiment("diagnostic", name=field, mark_clipmaps=True)
            self.mutate(source, 1, "Identity.json", lambda data: data[field].update(changed="9" * 64))
            with self.assertRaisesRegex(ValueError, "source/binary identity"):
                bitset.verify_experiment(source)
        source = self.experiment("diagnostic", name="wrong-levels", mark_clipmaps=True)
        package = source / a.read(source / "Processes.json")[1]["package"]
        path = package / "run-1/Process.log"
        path.write_text(path.read_text(encoding="utf-8").replace("markCacheLevels=6", "markCacheLevels=12"), encoding="utf-8")
        self.seal(source, 1)
        with self.assertRaisesRegex(ValueError, "markCacheLevels"):
            bitset.verify_experiment(source)
        source = self.experiment("diagnostic", name="wrong-faces", mark_clipmaps=True)
        def change_scope(data):
            row = data["cases"][0]["frames"][16]
            row["nodes"] = frame(16, faces=2)["nodes"]
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change_scope)
        with self.assertRaisesRegex(ValueError, "one directional face"):
            bitset.verify_experiment(source)

    def test_mark_clipmaps_report_and_plan(self):
        source = self.experiment("diagnostic", mark_clipmaps=True)
        timing = self.experiment("discovery", mark_clipmaps=True)
        output = self.root / "clipmaps-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=timing, output=output, diagnostic=source))
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("mark clipmaps", report)
        self.assertIn("12 compiled cache levels", report)
        self.assertIn("markCacheLevels", report)
        self.assertIn("All 45", report)
        self.assertIn("Primary target: markMs", report)
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic", mark_clipmaps=True))
        plan = bitset.prepare_plan(SimpleNamespace(phase="diagnostic", experiment="mark-clipmaps",
            cases_dir=cases, exe=self.root / "fake.exe"))
        self.assertEqual(plan["protocol"], bitset.MARK_CLIPMAPS_PROTOCOL)
        self.assertEqual(plan["policy"]["primaryMetric"], "markMs")
        self.assertNotEqual(plan["protocol"], bitset.DIRECTIONAL_MARK_PROTOCOL)

    def test_one_pass_switch_preserves_production_settings(self):
        workloads = []
        for arm in bitset.ARMS:
            value = graph(arm, "timing", one_pass=True)
            workloads.append(bitset.normalized_workload(spec("timing", arm), value, arm, one_pass=True))
            for key, invalid in ((bitset.ONE_PASS_SWITCH, None), (bitset.ONE_PASS_SWITCH, 1),
                    (bitset.DIRECTIONAL_MARK_SWITCH, False), ("benchmarkMarkSamples", False),
                    (bitset.DIRECTIONAL_SWITCH, False), (bitset.COMPACT_SWITCH, True),
                    ("benchmarkBuildSubstages", True), ("benchmarkProjectionStatistics", True),
                    ("compactOutputs", False)):
                value = graph(arm, "timing", one_pass=True)
                value["nodes"][0]["properties"][key] = invalid
                with self.assertRaises(ValueError):
                    bitset.normalized_workload(spec("timing", arm), value, arm, one_pass=True)
        self.assertEqual(*workloads)

    def test_one_pass_shader_variants_and_single_face_identity(self):
        for role in ("primary", "overhead"):
            rows = [{"arm": arm, "shaders": bitset.shader_identity(one_pass_shader_log(arm, "timing", role),
                arm, "timing", one_pass=True, role=role)} for arm in bitset.ARMS]
            bitset.check_shaders(rows, one_pass=True, role=role)
            self.assertEqual(len(rows[0]["shaders"]["facePrograms"]["vsmMarkMain"]), 3 if role == "primary" else 1)
            rows[1]["shaders"]["facePrograms"]["vsmMarkMain"]["0"] = "a" * 64
            with self.assertRaisesRegex(ValueError, "vsmMarkMain"):
                bitset.check_shaders(rows, one_pass=True, role=role)
        valid = one_pass_shader_log("B", "timing")
        for invalid in (valid.replace("work=false", "work=true"),
                valid.replace("projectionStatistics=false", "projectionStatistics=true"),
                valid.replace("directionalMark=true", "directionalMark=false"),
                "\n".join(valid.splitlines()[:-1]), valid.replace("f" * 64, "invalid")):
            with self.assertRaises(ValueError):
                bitset.shader_identity(invalid, "B", "timing", one_pass=True)
        with self.assertRaisesRegex(ValueError, "only for the multi-face candidate"):
            bitset.shader_identity(valid, "A", "timing", one_pass=True)

    def test_one_pass_execution_scope_is_authoritative(self):
        source = self.experiment("discovery", one_pass=True)
        result = bitset.verify_experiment(source)
        self.assertEqual(result["processes"][0]["faceCount"], 9)
        self.assertEqual({row["projectionMode"] for row in result["frames"] if row["arm"] == "B"}, {"one-pass"})
        def fake_scope(data):
            for part in data["cases"][0]["frames"][16]["nodes"][0]["sections"]:
                if part["name"] == "VSM one-pass projection":
                    part["name"] = "VSM projection"
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", fake_scope)
        with self.assertRaisesRegex(ValueError, "per-face"):
            bitset.verify_experiment(source)

    def test_one_pass_exact_work_and_build_receipts(self):
        source = self.experiment("diagnostic", one_pass=True)
        result = bitset.verify_experiment(source)
        self.assertEqual(len(result["diagnosticReceipts"]["exactBuildDiagnostics"][0]["diagnostics"]["totals"]), 45)
        self.assertEqual(len(result["diagnosticReceipts"]["stableWorkCounters"][0]["faces"][0]["counters"]), 15)
        def change(data):
            work = data["cases"][0]["frames"][16]["shadow"]["gpuWork"]
            work["faces"][0]["counters"]["projectionReceivers"] += 1
            work["totals"]["projectionReceivers"] += 1
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
        with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
            bitset.verify_experiment(source)

    def test_one_pass_primary_and_guard_thresholds(self):
        result = bitset.verify_experiment(self.experiment("confirmation", one_pass=True, role="overhead"))
        primary = bitset.experiment_policy(bitset.ONE_PASS_PROTOCOL)
        overhead = bitset.experiment_policy(bitset.ONE_PASS_PROTOCOL, "overhead")
        for row in result["processes"]:
            row["metrics"]["projectionMs"] = 1 if row["arm"] == "A" else 1.03
            row["metrics"]["graphMs"] = 1
        decision = bitset.comparisons(result["processes"], "confirmation", policy=overhead)
        self.assertEqual(decision["effectAssessment"], "confirmed_overhead_guard")
        self.assertEqual(decision["nextAction"], "require_primary_confirmation")
        self.assertEqual(decision["gains"]["projectionMs"]["blocks"], 3)
        self.assertEqual(bitset.comparisons(result["processes"], "confirmation", policy=primary)["decision"], "reject")
        self.assertEqual(bitset.comparisons(result["processes"][:8], "discovery", policy=overhead)["nextAction"], "run_confirmation")
        for row in result["processes"]:
            row["metrics"]["projectionMs"] = 1 if row["arm"] == "A" else .9
        self.assertEqual(bitset.comparisons(result["processes"], "confirmation", policy=primary)["nextAction"], "require_overhead_confirmation")
        for row in result["processes"]:
            if row["arm"] == "B":
                row["metrics"]["graphMs"] = 1.03
        self.assertEqual(bitset.comparisons(result["processes"], "confirmation", policy=overhead)["decision"], "reject")

    def test_one_pass_stage_roles_and_face_counts_cannot_drift(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic", one_pass=True))
        discovery = bitset.verify_experiment(self.experiment("discovery", one_pass=True))
        confirmation = bitset.verify_experiment(self.experiment("confirmation", one_pass=True))
        bitset.check_diagnostic(discovery, diagnostic)
        bitset.check_reference(confirmation, discovery)
        discovery["processes"][0]["faceCount"] = 8
        with self.assertRaisesRegex(ValueError, "face count"):
            bitset.check_diagnostic(discovery, diagnostic)
        discovery["processes"][0]["faceCount"] = 9
        discovery["plan"]["policy"] = bitset.experiment_policy(bitset.ONE_PASS_PROTOCOL, "overhead")
        with self.assertRaises(ValueError):
            bitset.check_reference(confirmation, discovery)

    def test_one_pass_report_and_plan(self):
        source = self.experiment("diagnostic", one_pass=True, role="overhead")
        timing = self.experiment("discovery", one_pass=True, role="overhead")
        output = self.root / "projection-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=timing, output=output, diagnostic=source))
        text = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("Role: **overhead**", text)
        self.assertIn("No placeholder spans", text)
        self.assertIn("requires both confirmations", text)
        self.assertIn("Primary target: projectionMs", text)
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic", one_pass=True))
        plan = bitset.prepare_plan(SimpleNamespace(phase="diagnostic", experiment="one-pass-projection", role="overhead",
            cases_dir=cases, exe=self.root / "fake.exe"))
        self.assertEqual(plan["protocol"], bitset.ONE_PASS_PROTOCOL)
        self.assertEqual(plan["policy"]["role"], "overhead")

    def test_directional_mark_shader_scope_and_actual_hashes(self):
        rows = [{"arm": arm, "shaders": bitset.shader_identity(shader_log(arm, "timing", directional_mark=True),
            arm, "timing", directional_mark=True)} for arm in bitset.ARMS]
        bitset.check_shaders(rows, directional_mark=True)
        for entry, value in (("vsmMarkMain", rows[0]["shaders"]["vsmMarkMain"]),
                             ("vsmBuildMain", "8" * 64), ("vsmProjectMain", "8" * 64)):
            prior = rows[1]["shaders"][entry]
            rows[1]["shaders"][entry] = value
            with self.assertRaises(ValueError):
                bitset.check_shaders(rows, directional_mark=True)
            rows[1]["shaders"][entry] = prior
        valid = shader_log("B", "timing", directional_mark=True)
        for invalid in (valid.replace("lightType=0", "lightType=1"),
                        valid.replace("directionalMark=true", "directionalMark=false"),
                        valid.replace("directionalMark=false", "directionalMark=true"),
                        valid.replace("compactBuild=false", "compactBuild=true"),
                        valid.replace("directionalBuild=true", "directionalBuild=false"),
                        valid.replace("work=false", "work=true"), "\n".join(valid.splitlines()[:-1]),
                        valid + "\n[VSMShader] entry=vsmCompactMain directionalMark=false"):
            with self.assertRaises(ValueError):
                bitset.shader_identity(invalid, "B", "timing", directional_mark=True)

    def test_directional_mark_exact_diagnostics_and_stable_work(self):
        source = self.experiment("diagnostic", directional_mark=True)
        result = bitset.verify_experiment(source)
        receipt = result["diagnosticReceipts"]
        self.assertEqual(len(receipt["exactBuildDiagnostics"][0]["diagnostics"]["totals"]), 45)
        self.assertEqual(len(receipt["stableWorkCounters"][0]["faces"][0]["counters"]), 15)
        def change(data, field, name):
            counters = data["cases"][0]["frames"][16]["shadow"][field]
            counters["faces"][0]["counters"][name] += 1
            counters["totals"][name] += 1
        # The atomics count is schedule dependent and intentionally not an exact receipt.
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda data: change(data, "gpuWork", "markPageRequests"))
        bitset.verify_experiment(source)
        for index, (field, name) in enumerate((("gpuWork", "markLevelTests"), ("gpuBuildDiagnostics", "clusterGroups"))):
            source = self.experiment("diagnostic", name=f"bad-counter-{index}", directional_mark=True)
            self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", lambda data: change(data, field, name))
            with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
                bitset.verify_experiment(source)

    def test_directional_mark_primary_and_independent_confirmation(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic", directional_mark=True))
        discovery = bitset.verify_experiment(self.experiment("discovery", frames=96, directional_mark=True))
        confirmation = bitset.verify_experiment(self.experiment("confirmation", frames=96, directional_mark=True))
        for result in (discovery, confirmation):
            # Build provides no gain; only the selected Mark primary qualifies this candidate.
            for row in result["processes"]:
                row["metrics"]["buildMs"] = 1
            bitset.check_diagnostic(result, diagnostic)
            decision = bitset.comparisons(result["processes"], result["phase"], "markMs")
            self.assertEqual(decision["primaryMetric"], "markMs")
            self.assertEqual(decision["gains"]["markMs"]["blocks"], bitset.BLOCKS[result["phase"]])
        bitset.check_reference(confirmation, discovery)
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation", "markMs")["decision"], "accept")
        for row in confirmation["processes"]:
            row["metrics"]["markMs"] = 1 if row["arm"] == "A" else 1.1
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation", "markMs")["effectAssessment"], "confirmed_regression")
        discovery["plan"]["protocol"] = bitset.DIRECTIONAL_PROTOCOL
        with self.assertRaises(ValueError):
            bitset.check_diagnostic(discovery, diagnostic)

    def test_directional_mark_report_and_plan(self):
        source = self.experiment("diagnostic", directional_mark=True)
        report = self.root / "mark-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=source, output=report))
        text = (report / "report.md").read_text(encoding="utf-8")
        self.assertIn("directional mark", text)
        self.assertIn("All 45", text)
        self.assertTrue((report / "DiagnosticWork.png").is_file())
        timing = self.experiment("discovery", directional_mark=True)
        timing_report = self.root / "mark-timing-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=timing, output=timing_report, diagnostic=source))
        text = (timing_report / "report.md").read_text(encoding="utf-8")
        self.assertIn("Primary target: markMs", text)
        self.assertEqual(a.read(timing_report / "Decision.json")["primaryMetric"], "markMs")
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic", directional_mark=True))
        plan = bitset.prepare_plan(SimpleNamespace(phase="diagnostic", experiment="directional-mark",
            cases_dir=cases, exe=self.root / "fake.exe"))
        self.assertEqual(plan["protocol"], bitset.DIRECTIONAL_MARK_PROTOCOL)
        self.assertEqual(plan["policy"]["primaryMetric"], "markMs")

    def test_directional_mark_rejects_changed_images_and_wrong_policy(self):
        source = self.experiment("diagnostic", directional_mark=True)
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda data: data["cases"][0]["captures"][0].update(rawSha256="9" * 64))
        with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
            bitset.verify_experiment(source)
        plan = a.read(source / "Plan.json")
        plan["policy"] = bitset.POLICY
        a.save(source / "Plan.json", plan)
        with self.assertRaisesRegex(ValueError, "Unknown plan/policy"):
            bitset.verify_experiment(source)

    def test_directional_shader_scope_and_identity(self):
        rows = [{"arm": arm, "shaders": bitset.shader_identity(shader_log(arm, "timing", True), arm, "timing", True)}
            for arm in bitset.ARMS]
        bitset.check_shaders(rows, True)
        rows[1]["shaders"]["vsmProjectMain"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "projection"):
            bitset.check_shaders(rows, True)
        for value in (shader_log("B", "timing", True).replace("lightType=0", "lightType=1"),
                shader_log("B", "timing", True).replace("directionalBuild=true", "directionalBuild=false"),
                shader_log("B", "timing", True).replace("directionalBuild=false", "directionalBuild=true"),
                "\n".join(shader_log("B", "timing", True).splitlines()[:-1])):
            with self.assertRaises(ValueError):
                bitset.shader_identity(value, "B", "timing", True)

    def test_directional_diagnostics_require_all_45_exact_counters(self):
        source = self.experiment("diagnostic", directional=True)
        result = bitset.verify_experiment(source)
        self.assertEqual(len(result["diagnosticReceipts"]["exactBuildDiagnostics"][0]["diagnostics"]["totals"]), 45)
        def change(data):
            diagnostic = data["cases"][0]["frames"][16]["shadow"]["gpuBuildDiagnostics"]
            diagnostic["faces"][0]["counters"]["clusterGroups"] += 1
            diagnostic["totals"]["clusterGroups"] += 1
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
        with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
            bitset.verify_experiment(source)

    def test_directional_cross_phase_protocol_is_separate(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic", directional=True))
        discovery = bitset.verify_experiment(self.experiment("discovery", frames=96, directional=True))
        confirmation = bitset.verify_experiment(self.experiment("confirmation", frames=96, directional=True))
        bitset.check_diagnostic(discovery, diagnostic)
        bitset.check_reference(confirmation, discovery)
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation")["decision"], "accept")
        discovery["plan"]["protocol"] = bitset.PROTOCOL
        with self.assertRaises(ValueError):
            bitset.check_diagnostic(discovery, diagnostic)
        with self.assertRaises(ValueError):
            bitset.check_reference(confirmation, discovery)

    def test_directional_report_and_plan(self):
        source = self.experiment("diagnostic", directional=True)
        report = self.root / "directional-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=source, output=report))
        text = (report / "report.md").read_text(encoding="utf-8")
        self.assertIn("directional build", text)
        self.assertIn("All 45", text)
        self.assertTrue((report / "DiagnosticWork.png").is_file())
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic", directional=True))
        plan = bitset.prepare_plan(SimpleNamespace(phase="diagnostic", experiment="directional-build",
            cases_dir=cases, exe=self.root / "fake.exe"))
        self.assertEqual(plan["protocol"], bitset.DIRECTIONAL_PROTOCOL)

    def test_compact_switch_is_explicit_and_only_difference(self):
        for arm in bitset.ARMS:
            for invalid in (None, 0, 1, "false", "true"):
                value = graph(arm, "timing", compact=True)
                value["nodes"][0]["properties"][bitset.COMPACT_SWITCH] = invalid
                with self.assertRaisesRegex(ValueError, bitset.COMPACT_SWITCH):
                    bitset.normalized_workload(spec("timing", arm), value, arm, compact=True)
        left = bitset.normalized_workload(spec("timing"), graph("A", "timing", compact=True), "A", compact=True)
        right = bitset.normalized_workload(spec("timing", "B"), graph("B", "timing", compact=True), "B", compact=True)
        self.assertEqual(left, right)

    def test_compact_shader_scope_and_identity(self):
        rows = [{"arm": arm, "shaders": bitset.shader_identity(shader_log(arm, "timing", compact=True), arm, "timing", compact=True)}
            for arm in bitset.ARMS]
        bitset.check_shaders(rows, compact=True)
        self.assertNotIn("vsmCompactMain", rows[0]["shaders"])
        self.assertIn("vsmCompactMain", rows[1]["shaders"])
        for arm, value in (("B", shader_log("B", "timing", compact=True).replace("compactBuild=true", "compactBuild=false")),
                ("B", "\n".join(shader_log("B", "timing", compact=True).splitlines()[:-1])),
                ("B", shader_log("B", "timing", compact=True).replace("compactBuild=false", "compactBuild=true")),
                ("B", shader_log("B", "timing", compact=True).replace("work=false", "work=true")),
                ("A", shader_log("A", "timing", compact=True) + "\n[VSMShader] entry=vsmCompactMain compactBuild=false work=false spirvSHA256=" + "9" * 64)):
            with self.assertRaises(ValueError):
                bitset.shader_identity(value, arm, "timing", compact=True)
        rows[1]["shaders"]["vsmProjectMain"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "projection"):
            bitset.check_shaders(rows, compact=True)

    def test_compact_requires_exact_build_diagnostics(self):
        source = self.experiment("diagnostic", compact=True)
        result = bitset.verify_experiment(source)
        self.assertEqual(len(result["diagnosticReceipts"]["exactBuildDiagnostics"][0]["diagnostics"]["totals"]), 45)
        def change(data):
            diagnostic = data["cases"][0]["frames"][16]["shadow"]["gpuBuildDiagnostics"]
            diagnostic["faces"][0]["counters"]["clusterGroups"] += 1
            diagnostic["totals"]["clusterGroups"] += 1
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
        with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
            bitset.verify_experiment(source)

    def test_compact_protocol_and_compaction_identity_persist_across_stages(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic", compact=True))
        discovery = bitset.verify_experiment(self.experiment("discovery", frames=96, compact=True))
        confirmation = bitset.verify_experiment(self.experiment("confirmation", frames=96, compact=True))
        bitset.check_diagnostic(discovery, diagnostic)
        bitset.check_reference(confirmation, discovery)
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation")["decision"], "accept")
        confirmation["processes"][1]["shaders"]["vsmCompactMain"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "Shader identity changed"):
            bitset.check_reference(confirmation, discovery)
        discovery["plan"]["protocol"] = bitset.DIRECTIONAL_PROTOCOL
        with self.assertRaises(ValueError):
            bitset.check_diagnostic(discovery, diagnostic)

    def test_compact_report_and_plan_preserve_scope(self):
        source = self.experiment("diagnostic", compact=True)
        report = self.root / "compact-report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=source, output=report))
        text = (report / "report.md").read_text(encoding="utf-8")
        self.assertIn("compact build", text)
        self.assertIn("compaction, its synchronization and indirect build", text)
        self.assertIn("do not read back indirect arguments", text)
        self.assertTrue((report / "DiagnosticWork.png").is_file())
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic", compact=True))
        plan = bitset.prepare_plan(SimpleNamespace(phase="diagnostic", experiment="compact-build",
            cases_dir=cases, exe=self.root / "fake.exe"))
        self.assertEqual(plan["protocol"], bitset.COMPACT_PROTOCOL)

    def test_bitset_switch_must_be_explicit_in_both_arms(self):
        for arm in bitset.ARMS:
            with self.subTest(arm=arm):
                value = graph(arm, "timing")
                value["nodes"][0]["properties"].pop(bitset.SWITCH)
                with self.assertRaisesRegex(ValueError, bitset.SWITCH):
                    bitset.normalized_workload(spec("timing", arm), value, arm)

    def test_bitset_switch_rejects_non_booleans_in_both_arms(self):
        for arm in bitset.ARMS:
            for invalid in (None, 0, 1, 0.0, 1.0, "false", "true", [], {}):
                with self.subTest(arm=arm, value=invalid):
                    value = graph(arm, "timing")
                    value["nodes"][0]["properties"][bitset.SWITCH] = invalid
                    with self.assertRaisesRegex(ValueError, bitset.SWITCH):
                        bitset.normalized_workload(spec("timing", arm), value, arm)

    def test_fixed_cases_reject_budget_reset_and_extra_renders(self):
        for change in (lambda value: value.update(frames=128), lambda value: value.update(warmupFrames=16),
                lambda value: value["cases"][0].update(freezeStreamingFrame=15),
                lambda value: value["cases"][0].update(events=[]),
                lambda value: value["cases"][0].update(comparisonFrames=[32])):
            value = spec("diagnostic")
            change(value)
            with self.assertRaises(ValueError):
                bitset.normalized_workload(value, graph("A", "diagnostic"), "A")
        for mode in ("diagnostic", "timing"):
            value = spec(mode)
            value["cases"][0]["comparisonFrames"] = [32]
            with self.assertRaisesRegex(ValueError, "extra renders"):
                bitset.normalized_workload(value, graph("A", mode), "A")

    def test_timing_disallows_capture_validation_and_work(self):
        for key in ("validation", "synchronizationValidation"):
            value = spec("timing")
            value[key] = True
            with self.assertRaisesRegex(ValueError, "off"):
                bitset.normalized_workload(value, graph("A", "timing"), "A")
        value = graph("A", "timing")
        value["nodes"][0]["properties"]["benchmarkGPUWork"] = True
        with self.assertRaisesRegex(ValueError, "benchmarkGPUWork"):
            bitset.normalized_workload(spec("timing"), value, "A")

    def test_medians_and_independent_confirmation(self):
        discovery = bitset.verify_experiment(self.experiment("discovery"))
        self.assertEqual(discovery["processes"][0]["postWarmupFrames"], 32)
        self.assertEqual(discovery["processes"][0]["metrics"]["graphMs"], 100)
        decision = bitset.comparisons(discovery["processes"], "discovery")
        self.assertEqual(decision["nextAction"], "run_confirmation")
        self.assertEqual(decision["gains"]["buildMs"]["blocks"], 2)
        self.assertAlmostEqual(decision["gains"]["buildMs"]["estimate"], .2)
        confirmation = bitset.verify_experiment(self.experiment("confirmation", gain=.01))
        bitset.check_reference(confirmation, discovery)
        self.assertEqual(bitset.comparisons(confirmation["processes"], "confirmation")["effectAssessment"], "below_target")

    def test_confirmation_acceptance_and_graph_gate(self):
        result = bitset.verify_experiment(self.experiment("confirmation"))
        decision = bitset.comparisons(result["processes"], "confirmation")
        self.assertEqual(decision["decision"], "accept")
        self.assertEqual(decision["gains"]["buildMs"]["blocks"], 3)
        for row in result["processes"]:
            if row["arm"] == "B":
                row["metrics"]["graphMs"] = 110
        self.assertEqual(bitset.comparisons(result["processes"], "confirmation")["effectAssessment"], "confirmed_regression")

    def test_spread_warns_without_dropping_samples(self):
        result = bitset.verify_experiment(self.experiment("discovery"))
        result["processes"][0]["metrics"]["buildMs"] *= 2
        decision = bitset.comparisons(result["processes"], "discovery")
        self.assertTrue(decision["spreadWarnings"])
        self.assertEqual(decision["gains"]["buildMs"]["blocks"], 2)

    def test_diagnostic_exact_ignores_changed_work_counters(self):
        result = bitset.verify_experiment(self.experiment("diagnostic"))
        self.assertTrue(result["exactPNGLinearHDRDepthShadowAndLegacyStatistics"])
        self.assertNotEqual(result["buildDiagnosticFrames"][0]["clusterGroups"], result["buildDiagnosticFrames"][64]["clusterGroups"])

    def test_explicit_output_registration_accepts_old_raw_plans_without_masking_changes(self):
        source = self.experiment("diagnostic")
        self.assertEqual(a.read(source / "Plan.json")["workload"]["cases"][0]["graph"]["outputs"], [])
        result = bitset.verify_experiment(source)
        self.assertEqual(result["workload"]["cases"][0]["graph"]["outputs"], ["FinalBlit.color", "Deferred.color"])
        self.assertEqual(a.read(source / "Plan.json")["workload"]["cases"][0]["graph"]["outputs"], [])
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["resolvedGraph"]["outputs"].append("Unexpected.color"))
        with self.assertRaisesRegex(ValueError, "Cases/graphs"):
            bitset.verify_experiment(source)

    def test_input_drift_precedes_output_mismatch(self):
        source = self.experiment("diagnostic")
        def change(data):
            case = data["cases"][0]
            case["frames"][16]["streaming"][0]["residentPages"] += 1
            case["captures"][0]["rawSha256"] = "f" * 64
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
        with self.assertRaisesRegex(ValueError, "input signature drift"):
            bitset.verify_experiment(source)

    def test_legacy_work_gate_excludes_only_scheduling_dependent_mark_atomics(self):
        for counter, accepted in (("projectionAtlasLoads", False), ("markPageRequests", True)):
            source = self.experiment("diagnostic", name=counter)
            def change(data):
                work = data["cases"][0]["frames"][16]["shadow"]["gpuWork"]
                work["faces"][0]["counters"][counter] += 1
                work["totals"][counter] += 1
            self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
            if accepted:
                bitset.verify_experiment(source)
            else:
                with self.assertRaisesRegex(ValueError, "stable-workload mismatch"):
                    bitset.verify_experiment(source)

    def test_pending_and_texture_semantics_are_strict_after_boundary(self):
        for field, value in (("pendingPages", 1), ("textureSemanticIdentitySha256", "")):
            source = self.experiment("diagnostic", name=field)
            self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
                lambda data: data["cases"][0]["frames"][16]["streaming"][0].update({field: value}))
            with self.assertRaises(ValueError):
                bitset.verify_experiment(source)

    def test_startup_retained_but_not_compared(self):
        source = self.experiment("diagnostic")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda data: data["cases"][0]["frames"][0]["streaming"][0].update(pendingPages=99))
        result = bitset.verify_experiment(source)
        self.assertEqual(result["startupEvidence"][1]["frames"][0]["streaming"][0]["pendingPages"], 99)

    def test_hdr_and_legacy_statistics_mismatch_rejected(self):
        for kind in ("HDR", "stats"):
            source = self.experiment("diagnostic", name=kind)
            def change(data):
                case = data["cases"][0]
                if kind == "HDR":
                    case["captures"][0]["linearHDR"]["rawSha256"] = "f" * 64
                else:
                    case["frames"][16]["shadow"]["statistics"][0][0] += 1
                    case["frames"][16]["shadow"]["totals"][0] += 1
            self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", change)
            with self.assertRaisesRegex(ValueError, "mismatch after input verification"):
                bitset.verify_experiment(source)

    def test_hash_identity_graph_and_shader_drift_rejected(self):
        source = self.experiment("discovery")
        self.mutate(source, 1, "Identity.json", lambda value: value["sources"].update(source="3" * 64))
        with self.assertRaisesRegex(ValueError, "source/binary"):
            bitset.verify_experiment(source)
        source = self.experiment("discovery", name="changed-graph")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["resolvedGraph"].update(sampleCount=4))
        with self.assertRaisesRegex(ValueError, "Cases/graphs"):
            bitset.verify_experiment(source)
        (source / "process-01-A/run-1/Process.log").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash/file"):
            bitset.verify_experiment(source)
        with self.assertRaisesRegex(ValueError, "clusterPageBitset"):
            bitset.shader_identity(optimization_tests.shader_log("Both", "timing"), "A", "timing")
        result = bitset.verify_experiment(self.experiment("discovery", name="same-shader"))
        for row in result["processes"]:
            row["shaders"]["vsmBuildMain"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "did not change"):
            bitset.check_shaders(result["processes"])

    def test_cross_phase_links_and_no_timing_texture_claim(self):
        diagnostic = bitset.verify_experiment(self.experiment("diagnostic"))
        discovery = bitset.verify_experiment(self.experiment("discovery", frames=96))
        bitset.check_diagnostic(discovery, diagnostic)
        self.assertFalse(discovery["canonicalInputSignatures"][0]["streams"][0]["textureSemanticIdentityAvailable"])
        confirmation = bitset.verify_experiment(self.experiment("confirmation", frames=96))
        bitset.check_reference(confirmation, discovery)
        confirmation["processes"][0]["shaders"]["vsmBuildMain"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "between discovery"):
            bitset.check_reference(confirmation, discovery)
        with self.assertRaisesRegex(ValueError, "requires --diagnostic"):
            bitset.load_links(SimpleNamespace(), "discovery")

    def test_reference_must_be_promising_and_pinned(self):
        diagnostic = self.experiment("diagnostic")
        discovery = self.experiment("discovery", gain=.01)
        with self.assertRaisesRegex(ValueError, "promising"):
            bitset.load_links(SimpleNamespace(diagnostic=diagnostic, reference=discovery), "confirmation")
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            bitset.load_links(SimpleNamespace(diagnostic=diagnostic), "discovery", {"links": {"diagnostic": {
                "source": str(diagnostic), "planSHA256": "wrong", "processesSHA256": "wrong"}}})

    def test_overlapping_and_incomplete_processes_rejected(self):
        source = self.experiment("discovery")
        receipts = a.read(source / "Processes.json")
        receipts[1]["started"] = receipts[0]["started"]
        a.save(source / "Processes.json", receipts)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            bitset.verify_experiment(source)
        a.save(source / "Processes.json", receipts[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            bitset.verify_experiment(source)

    def test_report_separate_outputs_and_immutable_source(self):
        diagnostic = self.experiment("diagnostic")
        source = self.experiment("discovery")
        before = {str(path): a.digest(path) for root in (diagnostic, source) for path in root.rglob("*") if path.is_file()}
        output = self.root / "report"
        with mock.patch("builtins.print"):
            bitset.report(SimpleNamespace(source=source, output=output, diagnostic=diagnostic, reference=None))
        self.assertTrue((output / "ProcessMedians.png").exists())
        self.assertTrue((output / "BlockGains.png").exists())
        self.assertEqual(a.read(output / "Decision.json")["gains"]["buildMs"]["blocks"], 2)
        self.assertTrue((output / "Reproduce.py").exists())
        self.assertTrue((output / "BuildDiagnosticFrames.json").exists())
        self.assertEqual(before, {path: a.digest(path) for path in before})

    def test_runner_failure_preserves_receipt_without_retry(self):
        cases = self.root / "cases"
        cases.mkdir()
        for arm in bitset.ARMS:
            value = spec("diagnostic", arm)
            value["cases"][0]["graph"] = f"{arm}.json"
            a.save(cases / f"Diagnostic-{arm}Cases.json", value)
            a.save(cases / f"{arm}.json", graph(arm, "diagnostic"))
        args = SimpleNamespace(phase="diagnostic", cases_dir=cases, exe=self.root / "fake.exe", output=self.root / "run")
        with mock.patch.object(a, "run", side_effect=RuntimeError("synthetic failure")) as launch:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                bitset.run(args)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[0].runs, 1)
        self.assertEqual(launch.call_args.args[0].timeout, 600)
        receipts = a.read(args.output / "Processes.json")
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0]["status"], "failed")
        self.assertIn("ended", receipts[0])
        self.assertTrue((args.output / "Failure.json").exists())


if __name__ == "__main__":
    unittest.main()
