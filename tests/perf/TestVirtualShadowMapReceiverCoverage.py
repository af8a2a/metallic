"""Small CPU set/allocator fixtures for the offline coverage model; no GPU."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import numpy as np
import VirtualShadowMapReceiverCoverage as coverage


class CoverageSets(unittest.TestCase):
    def test_two_pages_cover_and_third_page_rejects(self):
        rects = np.array([[0, 0, 1, 0], [0, 0, 2, 0]], dtype=np.int32)
        witness = np.array([[0, 1], [0, 1]], dtype=np.uint32)
        covered, area, _ = coverage.covered_rectangles(rects, witness, 0, 4)
        self.assertEqual(covered.tolist(), [True, False])
        self.assertEqual(area.tolist(), [2, 3])

    def test_cross_level_and_invalid_witness_cannot_cover(self):
        rects = np.array([[0, 0, 0, 0]] * 3, dtype=np.int32)
        witness = np.array([[16], [coverage.INVALID], [0]], dtype=np.uint32)
        covered, _, _ = coverage.covered_rectangles(rects, witness, 0, 4)
        self.assertEqual(covered.tolist(), [False, False, True])

    def test_duplicate_witness_does_not_cover_second_page(self):
        rects = np.array([[0, 0, 1, 0]], dtype=np.int32)
        witness = np.array([[0, 0, coverage.INVALID]], dtype=np.uint32)
        covered, _, unique = coverage.covered_rectangles(rects, witness, 0, 4)
        self.assertFalse(covered[0])
        self.assertEqual(unique[0], 1)

    def test_reverse_page_rank_reserves_coarse_slot(self):
        requested = np.array([True, False, True, True, False, True])
        self.assertEqual(np.flatnonzero(coverage.descending_admission(requested, 3)).tolist(), [3, 5])
        self.assertFalse(coverage.descending_admission(requested, 1).any())
        candidate = requested.copy()
        candidate[4] = True
        m = coverage.set_metrics(candidate, requested, coverage.descending_admission(requested, 3), 3)
        self.assertEqual(m["baselineAdmittedDisplaced"], 1)

    def test_tiles_preserve_coordinates_and_padded_tail(self):
        values = np.arange(15, dtype=np.int32)
        tiled = coverage.tile_values(values, 5, 3, 2, -1)
        self.assertEqual(tiled[0].tolist(), [0, 1, 5, 6])
        self.assertEqual(tiled[-1].tolist(), [14, -1, -1, -1])
        restored = coverage.untile(np.arange(6), 5, 3, 2)
        self.assertEqual(restored.reshape(3, 5).tolist(), [[0, 0, 1, 1, 2], [0, 0, 1, 1, 2], [3, 3, 4, 4, 5]])

    def test_halo_stays_within_page_level_and_rect_union(self):
        mask = coverage.halo_pages(np.array([0, 31, coverage.INVALID], dtype=np.uint32), 4, 2)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1, 4, 5, 26, 27, 30, 31])
        result = np.zeros(32, dtype=bool)
        coverage.add_rectangles(result, np.array([[0, 0, 1, 0], [1, 0, 2, 1], [4, 4, -1, -1]]), 1, 4)
        self.assertEqual(np.flatnonzero(result).tolist(), [16, 17, 18, 21, 22])

    def test_parent_receipt_selects_same_frame_face_counter(self):
        source = {"cases": [{"receiverCoverageCaptures": [{"frame": 32, "lightIndex": 1}],
                            "frames": [{"frame": 32, "shadow": {"gpuWork": {"enabled": True,
                                "faces": [{"lightIndex": 1, "face": 0, "counters": {
                                    "markLevelTests": 100, "markSamplePositions": 20, "markReceiverDispatches": 2}}]}}}]}]}
        self.assertEqual(list(coverage.captures(source))[0]["markLevelTests"], 100)

    def test_binary_protocol_and_counter_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rec = np.zeros((64, 22), dtype="<u4")
            rec[:, 0], rec[:, 4], rec[:, 6], rec[:, 7], rec[:, 8] = 1, np.float32(1).view("u4"), 1, 2, 10
            rec[:, 9:21] = coverage.INVALID
            rec[:, 9], rec[:, 10], rec[:, 21] = 10, 26, np.float32(10).view("u4")
            proj = np.zeros((3, 28), dtype="<f4")
            for level in range(2):
                proj[level, 0] = proj[level, 5] = .1 / (2 ** level)
                proj[level, 10:12] = [.1, .5]
                proj[level, 17] = .3125 * (2 ** level)
                proj.view("<u4")[level, 20:24] = [level * 16, 4, level, 0]
            pages = np.zeros((33, 4), dtype="<u4")
            pages[[10, 26], 1] = 3
            capture = dict(version=1, frame=32, width=8, height=8, pageSize=16,
                           virtualPagesPerAxis=4, clipmapCount=2, physicalPageCount=3,
                           rayCount=2, samplesPerRay=16, angularRadius=.00465, lightIndex=0,
                           recordWords=22, recordsEnabled=True, markLevelTests=64 * 38,
                           markSamplePositions=64 * 36, markReceiverDispatches=64)
            for name, data in (("records", rec), ("projections", proj), ("pages", pages)):
                path = root / (name + ".bin")
                data.tofile(path)
                capture[name] = dict(file=path.name, byteLength=path.stat().st_size, rawSha256=coverage.digest(path))
            report = coverage.analyze(capture, root)
            self.assertEqual(report["baseline"]["residualSampleLevelTests"], 64 * 34)
            self.assertEqual(len(report["coverage"]), 8)
            self.assertTrue(report["validation"]["markLevelTestsMatch"])
            capture["markLevelTests"] += 1
            with self.assertRaisesRegex(ValueError, "GPU Mark level tests disagree"):
                coverage.analyze(capture, root)


if __name__ == "__main__":
    unittest.main()
