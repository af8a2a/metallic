import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/Perf"))
import MaterialBaseline as baseline


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional baseline tooling requires numpy")
class MaterialBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixtures").mkdir()
        cases = []
        for name, node, scope in [("OpenPBRPathTrace", "Reference", "Path trace shading"),
                                  ("OpenPBRDeferred", "Deferred", "Material classification"),
                                  ("RTXCRChiang", "PathTrace", "Path trace shading")]:
            cases.append({"id": name, "width": 1, "height": 1, "format": "RGBA32F", "image": name + ".rgba32f",
                          "frames": [{"frame": f, "executionId": f + 1, "graphMs": 1.0,
                                      "nodes": [{"name": node, "gpuMs": 0.8,
                                                 "sections": [{"name": scope, "gpuMs": 0.7}]}]} for f in range(32, 96)]})
        baseline.save(self.root / "fixtures/Cases.json", {"cases": [{"id": c["id"], "width": 1, "height": 1} for c in cases]})
        baseline.save(self.root / "Process.json", {"exitCodes": [0, 0, 0]})
        for index in range(3):
            run = self.root / f"run-{index}"
            run.mkdir()
            baseline.save(run / "MaterialBaseline.json", {"validation": False, "cases": cases})
            for case in cases:
                (run / case["image"]).write_bytes(struct.pack("<4f", 0.2, 0.4, 0.6, 1))
        baseline.seal(self.root)

    def mutate(self, action):
        path = self.root / "run-0/MaterialBaseline.json"
        report = baseline.read(path)
        action(report)
        baseline.save(path, report)
        baseline.seal(self.root)

    def test_harness_archive_copies_are_not_additional_processes(self):
        target = self.root / "run-0/reports/cases/0"
        target.mkdir(parents=True)
        baseline.save(target / "MaterialBaseline.json", {})
        baseline.seal(self.root)
        self.assertTrue(baseline.verify(self.root)["verified"])

    def test_changed_file_rejected(self):
        (self.root / "run-0/OpenPBRDeferred.rgba32f").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            baseline.verify(self.root)

    def test_missing_frames_rejected_even_with_valid_hashes(self):
        self.mutate(lambda r: r["cases"][0]["frames"].pop())
        with self.assertRaisesRegex(ValueError, "timing frames"):
            baseline.verify(self.root)

    def test_diagnostic_timings_rejected(self):
        self.mutate(lambda r: r.update(validation=True))
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            baseline.verify(self.root)

    def test_absent_scope_not_zero(self):
        self.mutate(lambda r: r["cases"][1]["frames"][0]["nodes"][0].update(sections=[]))
        with self.assertRaisesRegex(ValueError, "GPU scope"):
            baseline.verify(self.root)

    def test_case_configuration_drift_rejected(self):
        self.mutate(lambda r: r["cases"][0].update(width=2))
        with self.assertRaisesRegex(ValueError, "fixed fixture"):
            baseline.verify(self.root)

    def test_nonfinite_hdr_rejected(self):
        (self.root / "run-0/OpenPBRDeferred.rgba32f").write_bytes(struct.pack("<4f", float("nan"), 0, 0, 1))
        baseline.seal(self.root)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            baseline.verify(self.root)


if __name__ == "__main__":
    unittest.main()
