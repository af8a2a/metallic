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


def graph(arm, mode, resolved=False, directional=False):
    result = optimization_tests.graph("Both", mode)
    result["nodes"][0]["properties"][bitset.DIRECTIONAL_SWITCH if directional else bitset.SWITCH] = arm == "B"
    if resolved:
        result["outputs"] = ["FinalBlit.color"] + (["Deferred.color"] if mode == "diagnostic" else [])
    return result


def shader_log(arm, mode, directional=False):
    result = optimization_tests.shader_log("Both", mode)
    if arm == "B":
        result = result.replace("d" * 64, "e" * 64)
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

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def experiment(self, phase, gain=.2, name=None, frames=64, directional=False):
        source = self.root / (name or phase)
        source.mkdir()
        mode = "diagnostic" if phase == "diagnostic" else "timing"
        plan = {"protocol": bitset.DIRECTIONAL_PROTOCOL if directional else bitset.PROTOCOL, "phase": phase, "mode": mode, "policy": bitset.POLICY,
            "processes": bitset.schedule(phase), "blocks": bitset.BLOCKS[phase], "runsPerPackage": 1,
            "timeoutSeconds": 600, "comparisonStartFrame": 16,
            "workload": bitset.normalized_workload(spec(mode, frames=frames), graph("A", mode, directional=directional), "A", directional),
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
            case = {"id": "Scene", "status": "passed", "resolvedGraph": graph(arm, mode, resolved=True, directional=directional),
                "noRayTracingAllocations": True, "noRayQueryOrRTAS": True, "frames": []}
            for index in range(frames):
                row = frame(index, scale=1000 if index < 32 else (1 if arm == "A" else 1 - gain))
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
                    if directional:
                        diagnostic = row["shadow"]["gpuBuildDiagnostics"]
                        diagnostic["version"] = 3
                        for counters in [diagnostic["totals"], *[face["counters"] for face in diagnostic["faces"]]]:
                            counters.update({name: 0 for name in bitset.profile.ALL_BUILD_COUNTERS if name not in counters})
                    elif arm == "B":
                        for face in row["shadow"]["gpuBuildDiagnostics"]["faces"]:
                            face["counters"]["clusterGroups"] += 1
                        row["shadow"]["gpuBuildDiagnostics"]["totals"]["clusterGroups"] += 2
                case["frames"].append(row)
            if mode == "diagnostic":
                case["captures"] = [{"frame": index, "rawSha256": "a" * 64, "depthSha256": "b" * 64,
                    "shadowSha256": "c" * 64, "linearHDR": {"rawSha256": "d" * 64, "output": "Deferred.color",
                    "format": "RGBA16Sfloat", "width": 2, "height": 1, "components": 8, "nonfiniteComponents": 0}}
                    for index in (16, 32, 63)]
            data = {"protocol": "virtual-shadow-map-acceptance-v2", "status": "passed", "mode": mode,
                "validation": mode == "diagnostic", "performanceEligible": mode == "timing", "cases": [case]}
            a.save(directory / "VirtualShadowMapAcceptance.json", data)
            (directory / "Process.log").write_text(shader_log(arm, mode, directional), encoding="utf-8")
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
