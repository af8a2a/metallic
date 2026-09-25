"""Fail-closed joins and native export shape; synthetic tests are not GPU evidence."""
import copy
import csv
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import DeepProfile as d
import TestWorkloadCase as fixtures


class DeepProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkloadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        self.bindings = self.fixture.capture["cases"][0]["before"]["productionDispatches"]
        self.log = "[PipelineStatistics] enabled=true\n"
        self.log += ("[PipelineStatisticsBinding] cacheKey=0123456789abcdef inputSpirvFnv1a64=123456 "
                     "deviceSpirvFnv1a64=654321 shader=Features/GPUDriven/GPUDrivenStreamWorkRaster."
                     "streamClusterRasterWorkControlMain entry=main\n")
        for name, value in (("Register Count", 40), ("Shared Memory Size", 2048), ("Binary Size", 4096)):
            self.log += ("[PipelineStatistics] entry=main spirv=0123456789abcdef executable=CS subgroup=32 "
                         f"{name}={value} (Driver description)\n")
        self.rows = [["flattened_event_name", *[name for name in d.METRICS for _ in range(2)]]]
        self.events = [["event_text", "time_ms", "time_ms"]]
        for marker in d.RANGES.values():
            self.rows.append([marker, *[str(i + 1) for i in range(2 * len(d.METRICS))]])
            self.events.append([marker, ".2", ".21"])

    def parse(self):
        for name, rows in (("GPUTRACE_REGIMES.xls", self.rows), ("D3DPERF_EVENTS.xls", self.events)):
            with (self.path / name).open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream, delimiter="\t").writerows(rows)
        return d.triage(self.path)

    def diagnostic_rows(self):
        self.fixture.capture["measurementKind"] = "diagnostic"
        validation = self.fixture.analyze()
        profile = self.parse()
        return [{"arm": "A", "validation": copy.deepcopy(validation), "profile": copy.deepcopy(profile),
                 "process": {"pid": 10 + i, "startedUnix": i * 10, "endedUnix": i * 10 + 5}}
                for i in range(3)]

    def test_raw_and_device_fingerprints_not_cache_keys(self):
        result = d.resources(self.log, self.bindings)
        self.assertEqual(result["boundSpirvFnv1a64"], "123456")
        self.assertEqual(result["statistics"][0]["deviceSpirvFnv1a64"], "654321")
        self.assertFalse(result["hardwareCounters"])

    def test_unbound_or_ambiguous_resources_rejected(self):
        for bad in (self.log.replace("inputSpirvFnv1a64=123456", "inputSpirvFnv1a64=111"),
                    self.log.replace("enabled=true", "enabled=false"),
                    self.log.replace("Shared Memory Size", "Unknown Size"),
                    self.log.replace("streamClusterRasterWorkControlMain", "anotherShader"),
                    self.log + "[PipelineStatistics] statistics unavailable: 5"):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                d.resources(bad, self.bindings)

    def test_conflicting_pipeline_query_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            d.resources(self.log + self.log.replace("Register Count=40", "Register Count=41"), self.bindings)

    def test_duplicate_columns_retained_with_raw_positions(self):
        result = self.parse()["selections"]["early"]
        self.assertEqual(result["row"], 2)
        self.assertEqual([m["column"] for m in result["metrics"][:2]], [2, 3])
        self.assertEqual([m["value"] for m in result["metrics"][:2]], [1, 2])
        self.assertEqual(result["durationMsColumns"], [.2, .21])

    def test_unknown_duplicate_missing_and_ragged_schema(self):
        original = copy.deepcopy(self.rows)
        for change in (lambda: self.rows[0].__setitem__(1, "wrong.counter"),
                       lambda: self.rows[1].pop(),
                       lambda: self.rows.append(self.rows[1][:])):
            self.rows = copy.deepcopy(original)
            change()
            with self.assertRaises(ValueError):
                self.parse()

    def test_native_indentation_and_names_containing_slashes(self):
        self.events = [["event_text", "time_ms", "time_ms"]]
        self.rows = [self.rows[0]]
        for phase, marker in d.RANGES.items():
            labels = marker.split("/")
            for depth, label in enumerate(labels):
                self.events.append([" " * (depth * 8) + label, ".1", ".1"])
                self.rows.append(["/".join(labels[:depth + 1]), *["1"] * (len(self.rows[0]) - 1)])
        result = self.parse()
        self.assertEqual(result["selections"]["late"]["marker"], d.RANGES["late"])
        # An unrelated label containing '/' must remain a single stack entry.
        self.events.extend([["Other / root", ".1", ".1"], ["        Child / label", ".1", ".1"]])
        self.rows.extend([[name, *["1"] * (len(self.rows[0]) - 1)]
                          for name in ("Other / root", "Other / root/Child / label")])
        self.parse()
        self.events[2][0] = " " + self.events[2][0]
        with self.assertRaisesRegex(ValueError, "indentation"):
            self.parse()

    def test_empty_ambiguous_and_misaligned_markers(self):
        self.rows[1][0] = "Submit"
        with self.assertRaises(ValueError):
            self.parse()
        self.events[1][0] = "Submit"
        with self.assertRaisesRegex(ValueError, "Missing/ambiguous"):
            self.parse()

    def test_nonfinite_sentinel_and_unavailable_values(self):
        for value in ("NaN", "inf", "-1", "N/A", ""):
            self.rows[1][1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse()

    def test_repeats_keep_noisy_counter_inconclusive(self):
        rows = self.diagnostic_rows()
        self.assertTrue(d.assess(rows, "triage")["arms"]["A"]["repeatable"])
        rows[-1]["profile"]["selections"]["late"]["metrics"][0]["value"] *= 2
        result = d.assess(rows, "triage")
        self.assertFalse(result["arms"]["A"]["repeatable"])
        self.assertEqual(result["optimizationDecision"], "not-evaluated")

    def test_process_and_workload_drift_rejected(self):
        for field in ("process", "identity", "output"):
            rows = self.diagnostic_rows()
            if field == "process":
                rows[1]["process"] = rows[0]["process"]
            elif field == "identity":
                rows[1]["validation"]["identity"]["snapshot"]["cutHash"] = "drift"
            else:
                rows[1]["validation"]["identity"]["snapshot"]["VBuffer.depth"]["sha256"] = "drift"
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.assess(rows, "triage")

    def test_normal_timing_cannot_be_reused_as_diagnostics(self):
        rows = self.diagnostic_rows()
        rows[0]["validation"]["normalTiming"] = True
        with self.assertRaisesRegex(ValueError, "non-diagnostic"):
            d.assess(rows, "triage")

    def test_sdk_completion_and_snapshot_must_match(self):
        import shutil
        self.fixture.capture.update(measurementKind="diagnostic", gpuTraceInjected=True)
        self.fixture.analyze()
        app = self.path / "app"
        app.mkdir()
        for p in list(self.path.iterdir()):
            if p.is_file():
                shutil.copyfile(p, app / p.name)
        d.w.save(self.path / "Process.json", {"exitCode": 0})
        (self.path / "stdout.log").write_text("")
        with self.assertRaisesRegex(ValueError, "SDK capture"):
            d.inspect_run(self.path, self.fixture.case, "triage")
        capture = d.w.load(app / "Capture.json")
        snapshot = copy.deepcopy(capture["cases"][0]["before"])
        snapshot["cutHash"] = "drift"
        capture["sdkTrace"] = {"complete": True, "frames": 1,
                                "workloadCase": capture["workloadCase"], "snapshot": snapshot}
        d.w.save(app / "Capture.json", capture)
        with self.assertRaisesRegex(ValueError, "SDK workload differs"):
            d.inspect_run(self.path, self.fixture.case, "triage")

    def test_archive_corruption_rejected_before_parsing(self):
        (self.path / "stdout.log").write_text("changed")
        d.w.save(self.path / "Manifest.json", {"protocol": d.PROTOCOL, "status": "complete",
                  "files": {"stdout.log": "not-the-hash"}})
        with self.assertRaisesRegex(ValueError, "Changed evidence"):
            d.verify(self.path)

    def test_exit_between_poll_and_child_inventory(self):
        import psutil
        from unittest.mock import Mock, patch
        process = Mock(pid=123, returncode=0)
        process.poll.side_effect = [None, 0]
        process.children.side_effect = psutil.NoSuchProcess(123)
        with patch.object(psutil, "Popen", return_value=process), patch.object(psutil, "wait_procs"):
            d.run_process(["fixture"], {}, self.path, 1)
        self.assertEqual(d.w.load(self.path / "Process.json")["exitCode"], 0)
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
