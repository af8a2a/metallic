"""CPU-only negative controls for demand-budget experiment evidence; never starts GPU work."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapDemandBudget as budget


class DemandBudgetTests(unittest.TestCase):
    def test_live_replay_rejects_retired_switch_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output=root / "output", cases_dir=root / "cases", phase="diagnostic",
                                   exe=root / "fake.exe", diagnostic=None)
            graph = {"nodes": [{"name": "Shadows", "type": "VirtualShadowMapLightsPass",
                                "properties": {"benchmarkDirectionalProjection": False}}]}

            def read(path):
                if path.name == "Assets.json":
                    return {}
                return graph if path.name == "Graph.json" else {"cases": [{"graph": "Graph.json"}]}

            with mock.patch.object(budget, "asset_manifest", return_value={}), \
                    mock.patch.object(budget.acceptance, "read", side_effect=read), \
                    mock.patch.object(budget.acceptance, "run") as launch:
                with self.assertRaisesRegex(ValueError, "Retired VSM graph properties"):
                    budget.run(args)
                launch.assert_not_called()

    def test_fixed_independent_process_budget(self):
        diagnostic = budget.schedule("diagnostic")
        self.assertEqual(len(diagnostic), 5)
        self.assertEqual([(r["workload"], r["arm"]) for r in diagnostic],
                         [("Slack", "Equal"), ("Slack", "Demand"), ("Pressure", "Equal"),
                          ("Pressure", "Demand"), ("Pressure", "Reference")])
        timing = budget.schedule("timing")
        self.assertEqual(len(timing), 24)
        for workload in budget.WORKLOADS:
            for block in range(3):
                self.assertEqual(tuple(r["arm"] for r in timing if r["workload"] == workload and r["block"] == block), budget.BLOCK)

    def counter_fixture(self, version):
        entries, statistics, records = [], [[0] * 32 for _ in range(54)], [[0] * 8 for _ in range(54)]
        for index in range(9):
            entries.append(dict(index=index, stableId=[index + 1, 0], kind=1, faceCount=1,
                                baseLayer=index, status=1, pageBudget=5))
            values = statistics[index * 6]
            values[0], values[1], values[12], values[17] = 4, 4, 5, 4
            records[index * 6] = [1, 4, 4, 5, 5, 0, 0, 7]
        return dict(frame=7, warmup=False, memory=dict(liveAllocationBytes=2048), shadow=dict(
            header=dict(count=9, capacity=9, budgetDroppedCount=0, physicalPageBudget=64),
            entries=entries, statistics=statistics, resourceMemory=[1024, 0, 256, 0, 9, 64, 0, 0],
            budgetDiagnostics=dict(header=[version, 1, 9, 9, 64, 45, 0, 36 if version == 2 else 0], faces=records)))

    def test_shared_schema_checks_current_totals_and_grants(self):
        raw = self.counter_fixture(2)
        actual = budget.counter_frame(raw)
        self.assertEqual(actual["currentDemandFaces"], 9)
        self.assertEqual(actual["reusedFine"], 36)
        self.assertNotIn("feedbackBufferCreates", actual)
        legacy = budget.counter_frame(self.counter_fixture(1))
        self.assertEqual(legacy["knownFaces"], 9)
        self.assertNotIn("currentDemandFaces", legacy)
        for mutate in (
                lambda f: f["shadow"]["budgetDiagnostics"]["header"].__setitem__(3, 0),
                lambda f: f["shadow"]["budgetDiagnostics"]["header"].__setitem__(6, 1),
                lambda f: f["shadow"]["budgetDiagnostics"]["header"].__setitem__(7, 0),
                lambda f: f["shadow"]["budgetDiagnostics"]["faces"][0].__setitem__(1, 3),
                lambda f: f["shadow"]["budgetDiagnostics"]["faces"][0].__setitem__(3, 6)):
            candidate = copy.deepcopy(raw)
            mutate(candidate)
            with self.assertRaisesRegex(ValueError, "Shared-pool"):
                budget.counter_frame(candidate)

    def test_shared_and_historical_allocation_timing(self):
        shared = dict(name="VSM shared physical pool", gpuMs=.2)
        feedback = dict(name="VSM budget feedback", gpuMs=.1)
        for arm in budget.ARMS:
            measured = budget.allocation_timing(dict(sections=[shared]), arm)
            self.assertEqual(measured["sharedPoolMs"], .2)
            self.assertEqual(measured["feedbackMs"], 0)
        self.assertEqual(budget.allocation_timing(dict(sections=[feedback]), "Demand")["feedbackMs"], .1)
        self.assertFalse(budget.allocation_timing(dict(sections=[]), "Equal")["sharedPoolScope"])
        for sections in ([shared, feedback], [shared, shared], [dict(shared, cpuOnly=True)]):
            with self.assertRaisesRegex(ValueError, "shared physical"):
                budget.allocation_timing(dict(sections=sections), "Demand")

    def test_shared_schema_allows_hot_face_above_4096_pages(self):
        raw = self.counter_fixture(2)
        shadow = raw["shadow"]
        shadow["header"]["physicalPageBudget"] = 6000
        shadow["entries"][0]["pageBudget"] = 5001
        values = shadow["statistics"][0]
        values[0], values[1], values[12], values[17] = 5000, 5000, 5001, 5000
        shadow["budgetDiagnostics"]["header"] = [2, 1, 9, 9, 6000, 5041, 0, 5032]
        shadow["budgetDiagnostics"]["faces"][0] = [1, 5000, 5000, 5001, 667, 0, 0, 7]
        self.assertEqual(budget.counter_frame(raw)["faces"][0]["assignedPages"], 5001)
        shadow["budgetDiagnostics"]["header"][0] = 1
        with self.assertRaisesRegex(ValueError, "Invalid per-face budget"):
            budget.counter_frame(raw)

    def fixture(self, root):
        demands = [35, 33, 101, 0, 0, 59, 0, 50, 120]
        entries = [dict(index=0, stableId=[1, 0], kind=0, faceCount=1, baseLayer=0, status=1),
                   dict(index=1, stableId=[2, 0], kind=0, faceCount=1, baseLayer=1, status=1),
                   dict(index=2, stableId=[3, 0], kind=2, faceCount=6, baseLayer=2, status=1),
                   dict(index=3, stableId=[4, 0], kind=1, faceCount=1, baseLayer=3, status=1)]
        records = []
        for item in budget.schedule("diagnostic"):
            workload, arm = item["workload"], item["arm"]
            directory = root / item["package"]
            directory.mkdir()
            value = 255 if workload == "Slack" or arm == "Reference" else 0 if arm == "Equal" else 128
            (directory / "shadow.raw").write_bytes(bytes([value]) * 16)
            faces, index = [], 0
            for entry in entries:
                for face in range(entry["faceCount"]):
                    faces.append(dict(stableId=entry["stableId"], face=face, requestedFine=demands[index],
                                      assignedPages=1 if workload == "Pressure" and arm == "Demand" and demands[index] == 0 else 51))
                    index += 1
            counters = [dict(frame=i, warmup=i < 32, faces=copy.deepcopy(faces), entries=copy.deepcopy(entries),
                             demandDriven=arm == "Demand", knownFaces=9 if arm == "Demand" else 0,
                             changedFaceCount=0, feedbackBufferCreates=0, assignedPages=256 if workload == "Pressure" and arm == "Demand" else 512,
                             admittedFine=200 if workload == "Slack" or arm != "Equal" else 100,
                             rejectedFine=0 if workload == "Slack" or arm == "Reference" else 100 if arm == "Demand" else 200,
                             dirtyPages=0, coarseRedraws=0, atlasBytes=1024 if workload == "Pressure" and arm == "Demand" else 2048)
                        for i in range(64)]
            captures = [dict(frame=i, images={"VBuffer.depth": dict(rawSha256="depth"),
                        "VBuffer.visibility": dict(rawSha256="visibility"), "Deferred.color": dict(rawSha256="hdr"),
                        "Shadows.shadows": dict(rawSha256="shadow", file="shadow.raw", layerCount=4, width=2, height=2)})
                        for i in (32, 63)]
            records.append(dict(**item, counters=counters, captures=captures, directory=str(directory)))
        return records

    def test_quality_gate_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = self.fixture(Path(temporary))
            self.assertTrue(budget.quality_gates(records)["passed"])
            for mutate, message in (
                (lambda r: r[1]["captures"][0]["images"]["Deferred.color"].update(rawSha256="changed"), "exact native"),
                (lambda r: r[1]["counters"][33]["faces"][0].update(requestedFine=99), "page demand differs"),
                (lambda r: r[1]["counters"][33]["faces"][0].update(assignedPages=999), "did not stabilize"),
                (lambda r: [f.update(atlasBytes=1024) for f in r[1]["counters"]], "sufficient physical layout"),
                (lambda r: [f["faces"][0].update(assignedPages=52) for f in r[1]["counters"]], "sufficient physical layout"),
                (lambda r: r[1]["counters"][33].update(knownFaces=0), "feedback missing"),
                (lambda r: r[1]["counters"][33].update(feedbackBufferCreates=1), "allocation churn"),
                (lambda r: r[3]["counters"][33].update(admittedFine=1), "admission improvement"),
                (lambda r: r[4]["counters"][33].update(rejectedFine=1), "reference still drops"),
                (lambda r: r[3]["captures"][0]["images"]["VBuffer.depth"].update(rawSha256="changed"), "primary input changed"),
            ):
                candidate = copy.deepcopy(records)
                mutate(candidate)
                result = budget.quality_gates(candidate)
                self.assertFalse(result["passed"], message)
                self.assertTrue(any(message in error for error in result["errors"]), result["errors"])

    def test_pressure_image_regression_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = self.fixture(Path(temporary))
            # Baseline now matches the ample reference; candidate still differs.
            (Path(records[2]["directory"]) / "shadow.raw").write_bytes(bytes([255]) * 16)
            result = budget.quality_gates(records)
            self.assertFalse(result["passed"])
            self.assertTrue(any("MAE regressed" in error for error in result["errors"]))

    def test_shared_quality_gate_uses_allocation_not_feedback_churn(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = self.fixture(Path(temporary))
            for record in records:
                for row in record["counters"]:
                    row.update(budgetVersion=2, currentDemandFaces=9, allocatedNewFine=0, reusedFine=200)
                    for name in ("knownFaces", "changedFaceCount", "feedbackBufferCreates"):
                        row.pop(name)
            self.assertTrue(budget.quality_gates(records)["passed"])
            for field, value, message in (("currentDemandFaces", 0, "current-frame shared demand missing"),
                                          ("allocatedNewFine", 1, "shared physical allocation churn")):
                candidate = copy.deepcopy(records)
                candidate[1]["counters"][33][field] = value
                result = budget.quality_gates(candidate)
                self.assertFalse(result["passed"])
                self.assertTrue(any(message in error for error in result["errors"]))

    def test_seal_detects_changed_or_added_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data.txt").write_text("original")
            budget.seal(root)
            budget.check_seal(root)
            (root / "data.txt").write_text("modified")
            with self.assertRaises(ValueError):
                budget.check_seal(root)
            (root / "data.txt").write_text("original")
            (root / "extra.txt").write_text("unsealed")
            with self.assertRaises(ValueError):
                budget.check_seal(root)


if __name__ == "__main__":
    unittest.main()
