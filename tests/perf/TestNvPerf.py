"""Fail-closed evidence checks; synthetic fixtures are not GPU measurements."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import NvPerf as n
import TestWorkloadCase as fixtures


class NvPerfTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"protocol": "metallic-nvperf-v1", "status": "complete", "backend": "nvperf-vulkan-range",
            "measurementKind": "diagnostic", "clockPolicy": "unaltered", "requiredPasses": 1, "passesCollected": 1,
            "numRangesDropped": 0, "numTraceBytesDropped": 0, "metricNames": n.DEFAULT_METRICS,
            "chip": "test", "queueFamily": 0, "libraryDirectory": "test",
            "vulkanApiVersion": 4206592, "vulkanVersionOfficiallySupported": False,
            "ranges": [{"name": "WorkControl/" + phase, "index": i, "metrics": [
                {"name": name, "value": 10., "dimUnits": []} for name in n.DEFAULT_METRICS]}
                for i, phase in enumerate(("early", "late"))]}

    def test_valid_profile(self):
        n.validate_profile(self.profile, n.DEFAULT_METRICS)

    def test_incomplete_and_overflow_rejected(self):
        for key, value in (("status", "collecting"), ("requiredPasses", 2), ("passesCollected", 0),
                           ("numRangesDropped", 1), ("numTraceBytesDropped", 1), ("clockPolicy", "base")):
            with self.subTest(key=key):
                profile = {**self.profile, key: value}
                with self.assertRaises(ValueError):
                    n.validate_profile(profile, n.DEFAULT_METRICS)

    def test_missing_overflow_evidence(self):
        del self.profile["numRangesDropped"]
        with self.assertRaises(ValueError):
            n.validate_profile(self.profile, n.DEFAULT_METRICS)

    def test_wrong_duplicate_or_missing_range(self):
        for ranges in ([self.profile["ranges"][0]] * 2, [], [{**r, "name": "other"} for r in self.profile["ranges"]]):
            with self.assertRaises(ValueError):
                n.validate_profile({**self.profile, "ranges": ranges}, n.DEFAULT_METRICS)

    def test_missing_metric_or_units(self):
        self.profile["ranges"][0]["metrics"].pop()
        with self.assertRaises(ValueError):
            n.validate_profile(self.profile, n.DEFAULT_METRICS)

    def test_nonfinite_negative_empty_duration(self):
        for value in (float("nan"), float("inf"), -1, 0, True):
            with self.subTest(value=value):
                self.profile["ranges"][0]["metrics"][0]["value"] = value
                with self.assertRaises(ValueError):
                    n.validate_profile(self.profile, n.DEFAULT_METRICS)

    def test_instrumentation_cannot_qualify_timing(self):
        fixture = fixtures.WorkloadTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.capture["nvPerfRequested"] = True
        self.assertFalse(fixture.analyze()["normalTiming"])

    def test_repeatability_keeps_unstable_metrics(self):
        rows = [{"profile": copy.deepcopy(self.profile), "validation": {"identity": {"same": True}},
                 "process": {"pid": 10+i, "startedUnix": 10*i, "endedUnix": 10*i+5}} for i in range(3)]
        rows[2]["profile"]["ranges"][0]["metrics"][0]["value"] = 20
        summary = n.assess(rows)
        self.assertFalse(summary["metrics"][0]["repeatableWithin10Percent"])
        self.assertFalse(summary["candidateAccepted"])
        rows[2]["validation"]["identity"] = {"same": False}
        with self.assertRaises(ValueError):
            n.assess(rows)


if __name__ == "__main__":
    unittest.main()
