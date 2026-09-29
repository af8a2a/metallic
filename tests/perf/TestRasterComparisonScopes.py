import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools"))
from AnalyzeZorahFullRasterComparison import raster_parent_scopes


class RasterComparisonScopes(unittest.TestCase):
    def test_nested_stage_wrappers_are_not_added_twice(self):
        scopes = [{"path": "Frame/Stream early", "gpuMs": 9.0},
                  {"path": "Frame/Stream early/Stream early", "gpuMs": 8.9},
                  {"path": "Frame/Stream early/Stream early/Software raster", "gpuMs": 7.0},
                  {"path": "Frame/Stream late", "gpuMs": 1.0},
                  {"path": "Frame/Stream late/Stream late", "gpuMs": .9}]
        self.assertEqual(sum(s["gpuMs"] for s in raster_parent_scopes(scopes)), 10.0)

    def test_original_single_level_is_preserved(self):
        scopes = [{"path": "Frame/Stream early", "gpuMs": 9.0},
                  {"path": "Frame/Stream late", "gpuMs": 1.0}]
        self.assertEqual(raster_parent_scopes(scopes), scopes)

    def test_incomplete_gpu_results_are_rejected(self):
        for scopes in ([{"path": "Frame/Stream early", "gpuMs": 1.0}],
                       [{"path": "Frame/Stream early", "gpuMs": None},
                        {"path": "Frame/Stream late", "gpuMs": 1.0}]):
            with self.assertRaises(AssertionError):
                raster_parent_scopes(scopes)
