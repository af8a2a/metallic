"""Synthetic sealed packages and mocked runner only; never starts a renderer/GPU."""
import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapOptimization as optimization
from TestVirtualShadowMapGPUProfile import frame, add_work

a = optimization.acceptance


def graph(arm, mode):
    mark, cluster = optimization.ARMS[arm]
    return {"nodes": [{"name": "Shadows", "type": "VirtualShadowMapLightsPass", "properties": {
        "benchmarkLegacyMark": not mark, "benchmarkDisableClusterCull": not cluster,
        "benchmarkGPUProfile": True, "benchmarkGPUWork": mode == "diagnostic"}}], "outputs": []}


def shader_log(arm, mode):
    mark, cluster = optimization.ARMS[arm]
    return "\n".join(f"[VSMShader] entry={entry} spirvSHA256={character * 64} "
        f"optimizeMark={str(mark).lower()} clusterCull={str(cluster).lower()} work={str(mode == 'diagnostic').lower()}"
        for entry, character in (("vsmMarkMain", "b" if mark else "a"), ("vsmBuildMain", "d" if cluster else "c")))


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def seal(self, source, index):
        rows = a.read(source / "Processes.json")
        package = source / rows[index]["package"]
        a.save(package / "Manifest.json", {str(path.relative_to(package)): a.digest(path)
            for path in sorted(package.rglob("*")) if path.is_file() and path.name != "Manifest.json"})
        rows[index]["manifestSHA256"] = a.digest(package / "Manifest.json")
        a.save(source / "Processes.json", rows)

    def experiment(self, phase, gain=.2, name=None, start=0):
        source = self.root / (name or phase)
        source.mkdir()
        mode = "diagnostic" if phase == "diagnostic" else "timing"
        plan = {"protocol": optimization.PROTOCOL, "phase": phase, "mode": mode,
            "policy": optimization.POLICY, "processes": optimization.schedule(phase),
            "blocks": optimization.BLOCKS[phase], "runsPerPackage": 1, "timeoutSeconds": 600,
            "comparisonStartFrame": start,
            "inputs": {arm: {"cases": f"/{mode}-{arm}.json", "casesSHA256": arm,
                              "graph": f"/{arm}.json", "graphSHA256": arm} for arm in optimization.ARMS}}
        a.save(source / "Plan.json", plan)
        receipts = []
        epoch = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        for item in plan["processes"]:
            package = source / item["package"]
            directory = package / "run-1"
            directory.mkdir(parents=True)
            arm = item["arm"]
            spec = {"mode": mode, "frames": 4, "warmupFrames": start + 1, "cases": [{"id": "Scene", "graph": f"/{arm}.json"}]}
            if start:
                spec["cases"][0].update(comparisonStartFrame=start, freezeStreamingFrame=start,
                    events=[{"frame": start, "cameraCut": True}])
            if mode == "diagnostic":
                spec["cases"][0]["captureFrames"] = [1, 3]
            case = {"id": "Scene", "status": "passed", "resolvedGraph": graph(arm, mode),
                    "noRayTracingAllocations": True, "noRayQueryOrRTAS": True,
                    "frames": [frame(i, scale=1 if arm == "Legacy" else 1 - gain) for i in range(4)]}
            for index, row in enumerate(case["frames"]):
                row["warmup"] = index <= start
                row["streaming"] = [{"passName": "VBuffer", "residentPages": 5, "geometryUsedBytes": 1024,
                    "originalTopologyHash": 3, "primitiveMetadataHash": 4,
                    "textureSemanticIdentitySha256": "e" * 64 if mode == "diagnostic" else "",
                    "textureSemanticIdentityBytes": 12 if mode == "diagnostic" else 0,
                    "textureResidentBytes": 4096,
                    "pendingPages": 0, "allocationFailures": 0, "loadFailures": 0, "requestOverflows": 0}]
            if mode == "diagnostic":
                case["captures"] = [{"frame": i, "rawSha256": "a" * 64, "depthSha256": "b" * 64,
                    "shadowSha256": "c" * 64, "linearHDR": {"rawSha256": "d" * 64, "output": "Deferred.color",
                        "format": "RGBA16Sfloat", "width": 2, "height": 1, "components": 8, "nonfiniteComponents": 0}} for i in (1, 3)]
                for row in case["frames"]:
                    add_work(row)
                    row["shadow"]["statistics"] = [[500 + index for index in range(32)] for _ in range(2)]
                    row["shadow"]["totals"] = [1000 + 2 * index for index in range(32)]
            data = {"protocol": "virtual-shadow-map-acceptance-v2", "status": "passed", "mode": mode,
                    "validation": mode == "diagnostic", "performanceEligible": mode == "timing", "cases": [case]}
            a.save(directory / "VirtualShadowMapAcceptance.json", data)
            (directory / "Process.log").write_text(shader_log(arm, mode), encoding="utf-8")
            a.save(package / "Cases.json", spec)
            paths = plan["inputs"][arm]
            a.save(package / "Identity.json", {"sources": {"source": "1"}, "binaries": {"exe": "2"},
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

    def mutate(self, source, index, filename, change):
        package = source / a.read(source / "Processes.json")[index]["package"]
        path = package / filename
        value = a.read(path)
        change(value)
        a.save(path, value)
        self.seal(source, index)

    def test_bounded_schedules(self):
        self.assertEqual(len(optimization.schedule("diagnostic")), 4)
        for phase, count in (("discovery", 16), ("confirmation", 24)):
            rows = optimization.schedule(phase)
            self.assertEqual(len(rows), count)
            self.assertEqual(tuple(row["arm"] for row in rows[:8]), optimization.BLOCK)

    def test_process_medians_block_unit_and_separate_confirmation(self):
        discovery = optimization.verify_experiment(self.experiment("discovery"))
        result = optimization.comparisons(discovery["processes"], "discovery")
        self.assertEqual(result["Mark"]["decision"], "promising")
        self.assertEqual(result["Mark"]["gains"]["markMs"]["blocks"], 2)
        self.assertAlmostEqual(result["Mark"]["gains"]["markMs"]["estimate"], .2)
        self.assertEqual(discovery["processes"][0]["postWarmupFrames"], 3)
        confirmation = optimization.verify_experiment(self.experiment("confirmation", gain=.01))
        optimization.check_reference(confirmation, discovery)
        self.assertEqual(optimization.comparisons(confirmation["processes"], "confirmation")["Mark"]["decision"], "below_target")

    def test_graph_gate(self):
        result = optimization.verify_experiment(self.experiment("confirmation"))
        for row in result["processes"]:
            if row["arm"] != "Legacy":
                row["metrics"]["graphMs"] = 110
        self.assertEqual(optimization.comparisons(result["processes"], "confirmation")["Both"]["decision"], "graph_regressed")

    def test_graph_quality_change_and_wrong_switch_rejected(self):
        source = self.experiment("discovery")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["resolvedGraph"].update(sampleCount=4))
        with self.assertRaisesRegex(ValueError, "Cases/graphs differ"):
            optimization.verify_experiment(source)
        with self.assertRaisesRegex(ValueError, "benchmarkLegacyMark"):
            optimization.normalized_graph(graph("Legacy", "timing"), "Mark", "timing")

    def test_hash_and_source_identity_drift(self):
        source = self.experiment("discovery")
        self.mutate(source, 1, "Identity.json", lambda value: value["sources"].update(source="other"))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            optimization.verify_experiment(source)
        (source / "process-01-Legacy/run-1/Process.log").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash/file"):
            optimization.verify_experiment(source)

    def test_actual_shaders_and_cross_phase_identity(self):
        with self.assertRaisesRegex(ValueError, "Missing or unstable"):
            optimization.shader_identity("", "Legacy", "timing")
        with self.assertRaisesRegex(ValueError, "flag mismatch"):
            optimization.shader_identity(shader_log("Legacy", "diagnostic"), "Legacy", "timing")
        discovery = optimization.verify_experiment(self.experiment("discovery"))
        confirmation = optimization.verify_experiment(self.experiment("confirmation"))
        confirmation["processes"][0]["shaders"]["vsmMarkMain"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "between discovery"):
            optimization.check_reference(confirmation, discovery)

    def test_noop_and_within_arm_shader_drift_rejected(self):
        result = optimization.verify_experiment(self.experiment("discovery"))
        for row in result["processes"]:
            row["shaders"]["vsmMarkMain"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "did not change"):
            optimization.check_shader_pairs(result["processes"])
        result["processes"][-1]["shaders"]["vsmMarkMain"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "within the same arm"):
            optimization.check_shader_pairs(result["processes"])

    def test_diagnostic_scene_identity_must_match_timing(self):
        diagnostic = optimization.verify_experiment(self.experiment("diagnostic"))
        timing = optimization.verify_experiment(self.experiment("discovery"))
        self.assertEqual(optimization.scene_workload(diagnostic["workload"]), optimization.scene_workload(timing["workload"]))
        diagnostic["workload"]["cases"][0]["camera"] = {"eye": [1, 2, 3]}
        self.assertNotEqual(optimization.scene_workload(diagnostic["workload"]), optimization.scene_workload(timing["workload"]))

    def test_diagnostic_exact_hdr_and_old_statistics(self):
        source = self.experiment("diagnostic")
        self.assertTrue(optimization.verify_experiment(source)["exactPNGLinearHDRDepthShadowAndLegacyStatistics"])
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["captures"][0]["linearHDR"].update(rawSha256="e" * 64))
        with self.assertRaisesRegex(ValueError, "Cross-arm PNG/HDR"):
            optimization.verify_experiment(source)
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["captures"][0]["linearHDR"].update(rawSha256="d" * 64))
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["frames"][0]["shadow"]["statistics"][0].__setitem__(0, 999))
        with self.assertRaisesRegex(ValueError, "statistics disagree"):
            optimization.verify_experiment(source)

    def test_missing_checkpoint_and_nonfinite_hdr(self):
        source = self.experiment("diagnostic")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", lambda value: value["cases"][0]["captures"].pop())
        with self.assertRaisesRegex(ValueError, "checkpoints exactly once"):
            optimization.verify_experiment(source)

    def test_canonical_boundary_requires_warmup_freeze_and_reset(self):
        settings = {"comparisonStartFrame": 2, "freezeStreamingFrame": 2, "events": [{"frame": 2, "cameraCut": True}]}
        spec = {"warmupFrames": 3, "cases": [settings]}
        self.assertEqual(optimization.comparison_start(spec), 2)
        settings["freezeStreamingFrame"] = 1
        with self.assertRaisesRegex(ValueError, "same-frame"):
            optimization.comparison_start(spec)
        settings["freezeStreamingFrame"] = 2
        settings["events"] = []
        with self.assertRaisesRegex(ValueError, "same-frame"):
            optimization.comparison_start(spec)
        settings["events"] = [{"frame": 2, "cameraCut": True}]
        spec["warmupFrames"] = 2
        with self.assertRaisesRegex(ValueError, "inside warmup"):
            optimization.comparison_start(spec)

    def test_startup_drift_retained_but_canonical_input_and_stats_exact(self):
        source = self.experiment("diagnostic", start=1)
        def startup(value):
            first = value["cases"][0]["frames"][0]
            first["streaming"][0].update(pendingPages=9, residentPages=1)
            first["shadow"]["statistics"][0][0] += 1
            first["shadow"]["totals"][0] += 1
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json", startup)
        result = optimization.verify_experiment(source)
        self.assertEqual(len(result["frames"]), 16)
        self.assertEqual(len(result["startupEvidence"]), 4)
        self.assertEqual(result["diagnosticReceipts"]["legacyStatistics"][0]["frame"], 1)
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["frames"][1]["streaming"][0].update(residentPages=1))
        with self.assertRaisesRegex(ValueError, "Canonical streaming input signature"):
            optimization.verify_experiment(source)

    def test_canonical_pending_or_load_error_rejected(self):
        for field in optimization.STREAM_ERRORS:
            with self.subTest(field=field):
                source = self.experiment("diagnostic", start=1, name=field)
                self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
                    lambda value: value["cases"][0]["frames"][1]["streaming"][0].update({field: 1}))
                with self.assertRaisesRegex(ValueError, "zero pending"):
                    optimization.verify_experiment(source)

    def test_default_zero_preserves_legacy_startup_streaming_policy(self):
        source = self.experiment("diagnostic")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["frames"][0]["streaming"][0].update(pendingPages=9, residentPages=1))
        result = optimization.verify_experiment(source)
        self.assertEqual(result["comparisonStartFrame"], 0)
        self.assertEqual(result["diagnosticReceipts"]["legacyStatistics"][0]["frame"], 0)

    def test_timing_empty_texture_identity_is_explicitly_unavailable(self):
        timing = optimization.verify_experiment(self.experiment("discovery", start=1))
        self.assertTrue(all(not row["textureSemanticIdentityAvailable"] for row in timing["inputFrames"]))
        source = self.experiment("diagnostic", start=1)
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["frames"][1]["streaming"][0].update(textureSemanticIdentitySha256="", textureSemanticIdentityBytes=0))
        with self.assertRaisesRegex(ValueError, "Missing diagnostic texture"):
            optimization.verify_experiment(source)
        source = self.experiment("diagnostic", name="nonfinite")
        self.mutate(source, 1, "run-1/VirtualShadowMapAcceptance.json",
            lambda value: value["cases"][0]["captures"][0]["linearHDR"].update(nonfiniteComponents=1))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            optimization.verify_experiment(source)

    def test_overlapping_or_incomplete_processes_rejected(self):
        source = self.experiment("discovery")
        rows = a.read(source / "Processes.json")
        rows[1]["started"] = rows[0]["started"]
        a.save(source / "Processes.json", rows)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            optimization.verify_experiment(source)
        rows.pop()
        a.save(source / "Processes.json", rows)
        with self.assertRaisesRegex(ValueError, "incomplete fixed"):
            optimization.verify_experiment(source)

    def test_retired_live_runner_cannot_plan_or_launch(self):
        output = self.root / "retired"
        args = SimpleNamespace(output=output, exe=Path("never-started.exe"))
        with mock.patch.object(a, "run") as launch, mock.patch.object(a, "read") as read:
            for entry in (optimization.prepare_plan, optimization.run):
                with self.assertRaisesRegex(ValueError, "Retired VSM four-arm"):
                    entry(args)
            launch.assert_not_called()
            read.assert_not_called()
        self.assertFalse(output.exists())

    def test_reports_plot_and_reproduce_without_modifying_sealed_packages(self):
        diagnostic = self.experiment("diagnostic")
        discovery = self.experiment("discovery")
        confirmation = self.experiment("confirmation")
        before = {str(path): a.digest(path) for source in (diagnostic, discovery, confirmation) for path in source.rglob("*") if path.is_file()}
        for source, name in ((diagnostic, "diagnostic-report"), (confirmation, "confirmation-report")):
            output = self.root / name
            optimization.report(SimpleNamespace(source=source, output=output,
                diagnostic=diagnostic if source == confirmation else None,
                reference=discovery if source == confirmation else None))
            self.assertTrue((output / "Reproduce.py").exists())
            self.assertTrue(list(output.glob("*.png")))
            self.assertTrue((output / "Summary.json").exists())
        self.assertTrue(all(a.digest(path) == digest for path, digest in before.items()))

    def test_report_failure_retained_and_nested_output_rejected(self):
        source = self.experiment("discovery")
        with self.assertRaisesRegex(ValueError, "outside sealed"):
            optimization.report(SimpleNamespace(source=source, output=source / "report", diagnostic=None))
        (source / "process-01-Legacy/run-1/Process.log").write_text("tampered", encoding="utf-8")
        output = self.root / "invalid-report"
        with self.assertRaisesRegex(ValueError, "hash/file"):
            optimization.report(SimpleNamespace(source=source, output=output, diagnostic=None))
        self.assertTrue((output / "Failure.json").exists())
        self.assertTrue((output / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
