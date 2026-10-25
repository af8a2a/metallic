import copy
from pathlib import Path
import sys
import struct
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import MeasureCLASTessellation as runner


def rows(gain=.10, graph_gain=.01, blocks=3):
    values = []
    for index, arm in enumerate("ABBA" * blocks):
        # Common drift deliberately exceeds the per-arm spread warning.
        scale = 1 + (index // 4) * .5
        candidate = arm == "B"
        times = {m: {"median": scale * (1 - (graph_gain if m == "graphMs" else gain) if candidate else 1)}
                 for m in runner.METRICS}
        values.append({"arm": arm, "process": {"pid": index + 100, "startedUnix": index * 2,
                                                "endedUnix": index * 2 + 1},
                       "result": {"timings": times, "identity": {"mode": runner.MODES[arm], "spirvSha256": "same-is-legal"},
                                  "outputs": {"HDR": "exact"}}})
    return values


class CLASTessellationABTests(unittest.TestCase):
    def test_runtime_stream_normalizes_only_declared_source_metadata(self):
        data = bytearray(216)
        struct.pack_into("<8sIIQQqQII", data, 0, b"MTLMSTRM", 9, 0x01020304, 216, 70, 123, 456, 0, 0)
        first, provenance = runner.stream_content_identity(data, True)
        changed = bytearray(data)
        struct.pack_into("<qQ", changed, 32, 789, 101112)
        second, changed_provenance = runner.stream_content_identity(changed, True)
        self.assertEqual(first, second)
        self.assertNotEqual(provenance["rawSha256"], changed_provenance["rawSha256"])
        changed[100] = 1
        self.assertNotEqual(first, runner.stream_content_identity(changed, True)[0])
        self.assertNotEqual(runner.stream_content_identity(data, False)[0], runner.stream_content_identity(changed, False)[0])

    def test_runtime_stream_unknown_schema_or_dependencies_fail(self):
        data = bytearray(216)
        struct.pack_into("<8sIIQQqQII", data, 0, b"MTLMSTRM", 9, 0x01020304, 216, 70, 123, 456, 1, 5)
        with self.assertRaisesRegex(ValueError, "no external dependency"):
            runner.stream_content_identity(data, True)
        struct.pack_into("<I", data, 8, 10)
        with self.assertRaisesRegex(ValueError, "schema"):
            runner.stream_content_identity(data, True)

    def test_discovery_never_accepts(self):
        result = runner.decision(rows(), "discovery")
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["nextAction"], "run_confirmation")

    def test_confirmation_uses_blocks_and_allows_same_shader(self):
        result = runner.decision(rows(blocks=5), "confirmation", rows())
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["gains"]["clasMs"]["blocks"], 5)
        self.assertAlmostEqual(result["gains"]["clasMs"]["estimate"], .10)

    def test_noise_warning_does_not_discard_paired_gain(self):
        result = runner.decision(rows(), "discovery")
        self.assertTrue(result["noiseWarning"])
        self.assertAlmostEqual(result["gains"]["clasMs"]["estimate"], .10)

    def test_graph_regression(self):
        result = runner.decision(rows(graph_gain=-.05), "confirmation")
        self.assertEqual(result["effectAssessment"], "confirmed_regression")

    def test_below_target_is_not_no_benefit(self):
        result = runner.decision(rows(gain=.01), "confirmation")
        self.assertEqual(result["effectAssessment"], "below_target")

    def test_cross_stage_identity_drift_is_invalid(self):
        changed = rows(blocks=5)
        changed[1]["result"]["identity"]["spirvSha256"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "identity drift"):
            runner.decision(changed, "confirmation", rows())

    def test_cross_arm_input_drift_is_invalid(self):
        changed = rows()
        changed[1]["result"]["inputs"] = {"tessellationCounts": [0, 0, 0, 0]}
        with self.assertRaisesRegex(ValueError, "input drift"):
            runner.decision(changed, "discovery")

    def test_protocol_run_order(self):
        manifest = {"budgets": {"discovery": 3, "confirmation": 5}, "diagnosticsOnly": False}
        sequence = [("diagnostic", arm) for arm in "AB"] + [("discovery", arm) for arm in "ABBA" * 3]
        runs = [{"stage": stage, "arm": arm, "directory": str(i)} for i, (stage, arm) in enumerate(sequence)]
        runner.validate_run_order(runs, manifest)
        runs[1]["stage"] = "unknown"
        with self.assertRaisesRegex(ValueError, "stage or process order"):
            runner.validate_run_order(runs, manifest)

    def test_diagnostics_only_forbids_ordinary_processes(self):
        manifest = {"budgets": {"discovery": 3, "confirmation": 5}, "diagnosticsOnly": True}
        runs = [{"stage": "diagnostic", "arm": arm, "directory": str(i)} for i, arm in enumerate("AB")]
        runner.validate_run_order(runs, manifest)
        runs.append({"stage": "discovery", "arm": "A", "directory": "2"})
        with self.assertRaisesRegex(ValueError, "stage or process order"):
            runner.validate_run_order(runs, manifest)

    def test_different_output_rejects(self):
        changed = rows()
        changed[1]["result"]["outputs"]["HDR"] = "different"
        self.assertEqual(runner.decision(changed, "discovery")["effectAssessment"], "correctness_failure")

    def test_partial_or_wrong_order_fails(self):
        with self.assertRaisesRegex(ValueError, "ABBA"):
            runner.decision(rows()[:-1], "discovery")
        changed = rows()
        changed[1], changed[3] = changed[3], changed[1]
        with self.assertRaisesRegex(ValueError, "ABBA"):
            runner.decision(changed, "discovery")

    def test_process_overlap_fails(self):
        changed = rows()
        changed[1]["process"]["startedUnix"] = .5
        with self.assertRaisesRegex(ValueError, "processes"):
            runner.decision(changed, "discovery")

    def test_graph_normalization_removes_only_backend(self):
        a = {"nodes": [{"name": "VBuffer", "properties": {"rtTessellationBuildMode": "triangles", "rtTessellationMaxFactor": 8}}]}
        b = copy.deepcopy(a)
        b["nodes"][0]["properties"]["rtTessellationBuildMode"] = "templates"
        self.assertEqual(runner.normalized_graph(a), runner.normalized_graph(b))
        b["nodes"][0]["properties"]["rtTessellationMaxFactor"] = 4
        self.assertNotEqual(runner.normalized_graph(a), runner.normalized_graph(b))

    def test_nonpositive_and_nan_timing_fails(self):
        for value in (0, -1, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                runner.finite_time(value, "CLAS")


if __name__ == "__main__":
    unittest.main()
