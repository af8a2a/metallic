"""CPU-only sealed fixtures for optional build attribution; no GPU is launched."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapBuildAttribution as attribution
import TestVirtualShadowMapBitset as fixtures
from TestVirtualShadowMapGPUProfile import add_substages

a = attribution.acceptance


class AttributionTests(unittest.TestCase):
    setUp = fixtures.BitsetTests.setUp
    tearDown = fixtures.BitsetTests.tearDown
    seal = fixtures.BitsetTests.seal
    experiment = fixtures.BitsetTests.experiment

    def mutate(self, source, index, change):
        fixtures.BitsetTests.mutate(self, source, index, "run-1/VirtualShadowMapAcceptance.json", change)

    def source(self):
        source = self.experiment("discovery", frames=96, compact=True)
        plan = a.read(source / "Plan.json")
        plan.update(protocol=attribution.PROTOCOL, frames=96, warmupFrames=32)
        a.save(source / "Plan.json", plan)
        for item in plan["processes"]:
            def change(data):
                case = data["cases"][0]
                case["buildSubstageDiagnostics"] = True
                case["resolvedGraph"]["nodes"][0]["properties"]["benchmarkBuildSubstages"] = True
                for row in case["frames"]:
                    add_substages(row, item["arm"] == "B")
            self.mutate(source, item["index"], change)
        return source

    def test_complete_sealed_attribution_preserves_aggregate_and_missing_branches(self):
        result = attribution.verify(self.source())
        self.assertEqual(len(result["processes"]), 8)
        self.assertEqual(len(result["frames"]), 768)
        self.assertEqual(result["processes"][0]["metrics"]["buildMs"], 2.)
        self.assertNotIn("compactMs", result["processes"][0]["metrics"])
        self.assertNotIn("directMs", result["processes"][1]["metrics"])
        self.assertIn("unavailable", result["performanceAcceptance"])

    def test_requires_declared_split_and_exact_budget(self):
        value = fixtures.graph("A", "timing", compact=True)
        with self.assertRaisesRegex(ValueError, "benchmarkBuildSubstages"):
            attribution.normalized_workload(fixtures.spec("timing", frames=96), value, "A")
        value["nodes"][0]["properties"]["benchmarkBuildSubstages"] = True
        with self.assertRaisesRegex(ValueError, "96 frames"):
            attribution.normalized_workload(fixtures.spec("timing"), value, "A")
        attribution.normalized_workload(fixtures.spec("timing", frames=96), value, "A")

    def test_missing_measured_or_declared_substages_rejected(self):
        source = self.source()
        self.mutate(source, 1, lambda data: data["cases"][0].update(buildSubstageDiagnostics=False))
        with self.assertRaisesRegex(ValueError, "declaration"):
            attribution.verify(source)

    def test_canonical_input_drift_rejected(self):
        source = self.source()
        self.mutate(source, 1, lambda data: data["cases"][0]["frames"][32]["streaming"][0].update(residentPages=6))
        with self.assertRaisesRegex(ValueError, "streaming inputs drift"):
            attribution.verify(source)

    def test_work_counters_rejected(self):
        source = self.source()
        self.mutate(source, 1, lambda data: data["cases"][0].update(gpuWorkDiagnostics=True))
        with self.assertRaisesRegex(ValueError, "workload counter"):
            attribution.verify(source)

    def test_source_identity_drift_rejected(self):
        source = self.source()
        fixtures.BitsetTests.mutate(self, source, 1, "Identity.json", lambda value: value["sources"].update(source="3" * 64))
        with self.assertRaisesRegex(ValueError, "source/binary identity"):
            attribution.verify(source)

    def test_measured_branch_mismatch_rejected(self):
        source = self.source()
        def change(data):
            for row in data["cases"][0]["frames"]:
                sections = row["nodes"][0]["sections"]
                sections[:] = [section for section in sections if section["name"] not in attribution.profile.BUILD_SUBSTAGES]
                add_substages(row, compact=False)
        self.mutate(source, 1, change)
        with self.assertRaisesRegex(ValueError, "Measured build branch"):
            attribution.verify(source)

    def test_report_protects_source_and_emits_reproducible_plot(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, "outside"):
            attribution.report(source, source / "Report")
        before = a.digest(source / "process-01-A/Manifest.json")
        output = self.root / "Report"
        attribution.report(source, output)
        for name in ("BuildAttribution.png", "Frames.csv", "ProcessMedians.csv", "Summary.json", "ReportTools.json", "Reproduce.py"):
            self.assertTrue((output / name).exists(), name)
        self.assertEqual(before, a.digest(source / "process-01-A/Manifest.json"))


if __name__ == "__main__":
    unittest.main()
