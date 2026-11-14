import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import PathTraceQueries as pt


class PathTraceQueryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = dict(id="Mixed", resolvedGraph={"nodes": [1]}, format="RGBA32F", width=1, height=1,
                         image="color.bin", captures=[dict(output="PathTrace.motionVectors",frame=8,
                             file="motion.bin",bytes=4,format=1)])
        for arm in ("A", "B"):
            folder = self.root / arm
            folder.mkdir()
            (folder / "color.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 1))
            (folder / "motion.bin").write_bytes(struct.pack("<2e", .1, .2))
            (folder / "MaterialBaseline.json").write_text(json.dumps(dict(cases=[self.case])))

    def tearDown(self):
        self.temp.cleanup()

    def compare(self):
        return pt.compare_reports(self.root / "A/MaterialBaseline.json", self.root / "B/MaterialBaseline.json", ["Mixed"])

    def test_html_copy_is_not_an_extra_process(self):
        duplicate = self.root / "A/reports/cases/0"
        duplicate.mkdir(parents=True)
        (duplicate / "MaterialBaseline.json").write_text("{}")
        self.assertEqual(pt.report_path(self.root / "A"), self.root / "A/MaterialBaseline.json")
        self.assertTrue(self.compare()[0]["capturesByteIdentical"])

    def test_motion_vector_change_rejected(self):
        (self.root / "B/motion.bin").write_bytes(struct.pack("<2e", .1, .21))
        with self.assertRaisesRegex(ValueError, "Guide changed"):
            self.compare()

    def test_nonfinite_and_truncated_hdr_rejected(self):
        for blob in (struct.pack("<4f", float("nan"), 2, 3, 1), b"\0\0\0\0"):
            (self.root / "B/color.bin").write_bytes(blob)
            with self.assertRaises(ValueError):
                self.compare()

    def test_changed_camera_rejected(self):
        case = copy.deepcopy(self.case)
        case["resolvedGraph"] = {"nodes": [2]}
        (self.root / "B/MaterialBaseline.json").write_text(json.dumps(dict(cases=[case])))
        with self.assertRaisesRegex(ValueError, "Workload"):
            self.compare()

    def test_missing_case_rejected(self):
        (self.root / "B/MaterialBaseline.json").write_text(json.dumps(dict(cases=[])))
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.compare()

    def verify_performance(self, gains, *, graph_gain=0, thresholds=None):
        # Exercise the verifier through a sealed bundle, including recorded policy.
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            pt.mb.save(root / "Identity.json", dict(phase="performance", orders=pt.ORDERS,
                scope="synthetic policy regression fixture", thresholds=thresholds or pt.THRESHOLDS))
            pt.mb.save(root / "Transaction.json", dict(restored=True))
            pt.mb.save(root / "Cases.json", dict(cases=[dict(id="Mixed")], warmupFrames=2, timingFrames=2))
            processes = []
            for block, order in enumerate(pt.ORDERS):
                for arm in order:
                    folder = root / f"run-{block:02}-{arm}"
                    folder.mkdir()
                    (folder / "color.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 1))
                    (folder / "motion.bin").write_bytes(struct.pack("<2e", .1, .2))
                    case = copy.deepcopy(self.case)
                    pt_time = 1 if arm == "A" else 1 - gains[block]
                    graph_time = 2 if arm == "A" else 2 * (1 - graph_gain)
                    case["frames"] = [dict(frame=i, executionId=i, graphMs=graph_time,
                        nodes=[dict(name="PathTrace", gpuMs=pt_time)]) for i in (2, 3)]
                    pt.mb.save(folder / "MaterialBaseline.json", dict(validation=False, cases=[case]))
                    index = len(processes)
                    processes.append(dict(block=block, arm=arm, exit=0, start=index * 2,
                                          end=index * 2 + 1, shaderCacheLoads=[]))
            pt.mb.save(root / "Processes.json", processes)
            pt.mb.seal(root)
            result = pt.verify(root)
            return next(row for row in result["comparisons"] if row["comparison"] == "AB")

    def test_verified_small_gain_is_improvement(self):
        result = self.verify_performance([.01] * 6)
        self.assertEqual(result["decision"], "improvement")
        self.assertAlmostEqual(result["ptGain"]["low"], .01)

    def test_historical_bundle_preserves_original_threshold(self):
        self.assertEqual(self.verify_performance([.01] * 6, thresholds=pt.LEGACY_THRESHOLDS)["decision"], "inconclusive")
        self.assertEqual(self.verify_performance([.04] * 6, thresholds=pt.LEGACY_THRESHOLDS)["decision"], "improvement")

    def test_zero_or_uncertain_gain_is_inconclusive(self):
        for gains in ([0] * 6, [-.01, .02] * 3):
            with self.subTest(gains=gains):
                self.assertEqual(self.verify_performance(gains)["decision"], "inconclusive")

    def test_regression_and_noise_gates_remain(self):
        self.assertEqual(self.verify_performance([-.01] * 6)["decision"], "regression")
        self.assertEqual(self.verify_performance([.01] * 6, graph_gain=-.03)["decision"], "regression")
        self.assertEqual(self.verify_performance([.1, .3] * 3)["decision"], "inconclusive")

    def test_unknown_recorded_thresholds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported declared thresholds"):
            self.verify_performance([.01] * 6, thresholds={**pt.THRESHOLDS, "gainLower": -.01})


if __name__ == "__main__":
    unittest.main()
