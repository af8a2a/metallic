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


if __name__ == "__main__":
    unittest.main()
