"""Guard statistical units, matched identities and the measured load scope."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import AnalyzeComputePipelineExperiments as analysis


def gpu_row(block, arm, value, frame_count=64):
    return dict(experiment="fixed", command="dispatch", block=block, arm=arm,
                imported=False, cache_kind="none", case="Opaque", dispatch_scope_ms=value,
                timing_frame_count=frame_count)


class ComputePipelineAnalysisTests(unittest.TestCase):
    def test_gpu_frames_are_not_independent_process_replicates(self):
        rows = [gpu_row(0, "O1-pipeline-normal", 2, 4), gpu_row(1, "O1-pipeline-normal", 4, 100)]
        result = analysis.group_rows(rows)[0]["metrics"]["dispatch_scope_ms"]
        self.assertEqual(result, dict(n=2, min=2, median=3, mean=3, max=4))

    def test_pair_ratios_match_blocks_before_aggregation(self):
        rows = [gpu_row(0, "O1-pipeline-normal", 10), gpu_row(1, "O1-pipeline-normal", 1),
                gpu_row(1, "O0-pipeline-normal", 2), gpu_row(0, "O0-pipeline-normal", 5)]
        result = analysis.paired_ratios(rows)[0]
        self.assertEqual(result["ratios"]["median"], 1.25)
        self.assertEqual({pair["block"]: pair["ratio"] for pair in result["pairs"]}, {0: 0.5, 1: 2})

    def test_separate_experiments_are_not_pooled(self):
        rows = [gpu_row(0, "O1-pipeline-normal", 1), gpu_row(0, "O1-pipeline-normal", 2)]
        rows[1]["experiment"] = "different-device"
        self.assertEqual(len(analysis.group_rows(rows)), 2)
        self.assertEqual(analysis.paired_ratios(rows), [])

    def test_duplicate_process_block_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate process"):
            analysis.group_rows([gpu_row(0, "O1-pipeline-normal", 1), gpu_row(0, "O1-pipeline-normal", 2)])

    def test_import_pair_rejects_different_device_code(self):
        rows = [dict(experiment="fixed", command="compile", block=0, arm="O1-pipeline-normal", imported=False,
                     cache_kind="none", case="", pipeline_load_and_create_wall_ms=10, device_spirv_fnv1a64="abc"),
                dict(experiment="fixed", command="compile", block=0, arm="O1-pipeline-normal", imported=True,
                     cache_kind="application-pipeline-cache", case="", pipeline_load_and_create_wall_ms=1, device_spirv_fnv1a64="def")]
        with self.assertRaisesRegex(ValueError, "different device SPIR-V"):
            analysis.paired_ratios(rows)

    def test_native_does_not_replace_missing_legacy_full_load_scope(self):
        arm = dict(name="O1-pipeline-normal", optimization="O1", backend="pipeline", disable_driver_optimization=False)
        run = dict(block=0, arm=arm, imported=False, report="unused")
        sample = dict(device_spirv_bytes=4, device_spirv_fnv1a64="abc", device_spirv_statistics={},
                      compile_wall_ms=2, pipeline_wall_ms=11, shader_module_wall_ms=3,
                      pipeline=dict(native_create_ms=10, driver_ms=None))
        with patch.object(analysis.experiments, "read", return_value=sample), \
             patch.object(analysis.experiments.driver, "single_sample", side_effect=lambda report, native: report):
            row = analysis.native_rows("unused", dict(runs=[run]))[0]
            self.assertIsNone(row["pipeline_load_and_create_wall_ms"])
            self.assertIsNone(row["native_fraction"])
            sample.update(pipeline_load_and_create_wall_ms=12, pipeline_cache_preparation_wall_ms=1)
            self.assertAlmostEqual(analysis.native_rows("unused", dict(runs=[run]))[0]["native_fraction"], 10 / 12)
            sample["pipeline_load_and_create_wall_ms"] = 11
            with self.assertRaisesRegex(ValueError, "cache preparation"):
                analysis.native_rows("unused", dict(runs=[run]))

    def test_diagnostics_keep_identity_and_driver_defined_values(self):
        lines = """[PipelineStatistics] enabled=true
[PipelineStatisticsBinding] cacheKey=000000000000abcd inputSpirvFnv1a64=123 deviceSpirvFnv1a64=456 shader=Module.main entry=main
[PipelineStatistics] entry=main spirv=000000000000abcd executable=Shader CS subgroup=32 Register Count=64 (Number of registers)
[PipelineStatistics] entry=main spirv=000000000000abcd executable=Shader CS subgroup=32 Local Memory Size=18446744073709551615 (Driver value)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.log"
            path.write_text(lines, encoding="utf-8")
            result = analysis.statistics_log("O1", path)
            self.assertFalse(result["timing_eligible"])
            self.assertEqual(result["records"][1]["value"], "18446744073709551615")
            self.assertEqual(result["records"][0]["device_spirv_fnv1a64"], "456")
            path.write_text(lines.replace("entry=main spirv", "entry=other spirv"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matching shader binding"):
                analysis.statistics_log("O1", path)

    def test_sealed_input_directory_cannot_be_analysis_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "outside sealed"):
                analysis.analyze([root], root / "plots")

    def test_nonfinite_negative_boolean_metrics_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                analysis.distribution([value])

    def test_cost_components_do_not_double_count_driver_time(self):
        rows = []
        for block, native, prep in ((0, 10, 2), (1, 20, 4)):
            rows.append(dict(experiment="fixed", command="compile", block=block, arm="O1-pipeline-normal", imported=False,
                             cache_kind="none", case="", native_create_ms=native, driver_ms=native - 1,
                             pipeline_cache_preparation_wall_ms=prep, native_binary_import_ms=1,
                             native_binary_export_ms=0, other_pipeline_wall_ms=3,
                             pipeline_load_and_create_wall_ms=native + prep + 4))
        result = analysis.cost_breakdown(rows)[0]
        self.assertEqual(result["mean_total_ms"], 22)
        self.assertEqual(sum(result["components"].values()), result["mean_total_ms"])
        self.assertNotIn("driver_ms", result["components"])

    def test_gpu_native_cache_setting_does_not_claim_an_import_hit(self):
        arm = dict(name="O1-shader-object-normal", optimization="O1", backend="shader-object", disable_driver_optimization=False)
        correctness = [dict(case="Opaque", qualified=True, readbacks=[dict(max_absolute_error=0)])]
        run = dict(block=0, arm=arm, report="unused", qualified=True, correctness=correctness,
                   timings=dict(Opaque=dict(dispatch_scope_ms=dict(median=2, count=64))))
        row = analysis.dispatch_rows("unused", dict(runs=[run], native_binary_cache=True))[0]
        self.assertFalse(row["imported"])
        self.assertTrue(row["native_binary_cache_enabled"])
        self.assertEqual(row["cache_kind"], "runtime-configured-native-cache")
        arm["backend"] = "pipeline"
        row = analysis.dispatch_rows("unused", dict(runs=[run], native_binary_cache=True))[0]
        self.assertFalse(row["native_binary_cache_enabled"])
        self.assertEqual(row["cache_kind"], "none")


if __name__ == "__main__":
    unittest.main()
