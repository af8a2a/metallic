"""CPU-only negative controls for demand-budget experiment evidence; never starts GPU work."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapDemandBudget as budget


class DemandBudgetTests(unittest.TestCase):
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
