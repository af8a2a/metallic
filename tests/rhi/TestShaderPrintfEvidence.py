"""Tests for false-success prevention in the P0 GPU acceptance harness."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools"))
from RunShaderPrintfP0 import check


def fixture():
    return {
        "schema": "metallic.shader-printf.p0.v1", "phase": "finished", "status": "verified",
        "smokeVerified": True, "performanceEligible": False, "instrumentation": True,
        "gpuCompleted": True, "hostDropped": 0, "hostTruncated": 0, "gpuOverflow": False, "warningOrError": False,
        "capabilities": dict.fromkeys(("layerDiscovered", "instanceConfigured", "messengerConfigured", "deviceConfigured"), True),
    }, [{"id": 0x4fe1fef9, "severity": 16, "truncated": False,
         "text": "vkQueueSubmit2(): DebugPrintf:\nMTP0 ordinary seq=0 group=1 lane=3 value=73\n"}]


class EvidenceTests(unittest.TestCase):
    def test_empty_and_duplicate_echo_cannot_pass(self):
        report, raw = fixture()
        self.assertTrue(check("ordinary", report, raw, 0, ""))
        self.assertFalse(check("ordinary", report, [], 0, ""))
        self.assertFalse(check("ordinary", report, raw * 2, 0, ""))

    def test_unrelated_log_text_cannot_impersonate_printf(self):
        report, raw = fixture()
        raw[0]["id"] = 0
        self.assertFalse(check("ordinary", report, raw, 0, ""))

    def test_info_level_gpu_overflow_is_incomplete(self):
        report, raw = fixture()
        report.update(status="incomplete", smokeVerified=False, gpuOverflow=True)
        warning = {"id": 0x4fe1fef9, "severity": 16, "truncated": False,
                   "text": "[WARNING] Debug Printf message was truncated due to the buffer size (128) being too small"}
        self.assertTrue(check("gpu-overflow", report, raw + [warning], 2, ""))
        self.assertFalse(check("gpu-overflow", report, raw, 2, ""))
        wrong_capacity = copy.deepcopy(warning)
        wrong_capacity["text"] = wrong_capacity["text"].replace("(128)", "(1024)")
        self.assertFalse(check("gpu-overflow", report, raw + [wrong_capacity], 2, ""))

    def test_configuration_warning_is_not_gpu_overflow_evidence(self):
        report, raw = fixture()
        report.update(status="incomplete", smokeVerified=False, gpuOverflow=True)
        raw += [{"id": 123, "severity": 256, "text": "printf buffer size setting is invalid"}]
        self.assertFalse(check("gpu-overflow", report, raw, 2, ""))

    def test_heap_requires_readback_not_only_echo(self):
        report, raw = fixture()
        raw[0]["text"] = "MTP0 heap seq=0 group=1 lane=3 value=73 cookie=305397763"
        self.assertFalse(check("heap-mapped", report, raw, 0, ""))
        report["readback"] = {"passed": True, "actual": [74, 305397763, 15, 16], "expected": [74, 305397763, 15, 16]}
        self.assertTrue(check("heap-mapped", report, raw, 0, ""))
        report["readback"]["actual"][0] = 0
        self.assertFalse(check("heap-mapped", report, raw, 0, ""))

    def test_stdout_notice_is_allowed_only_in_redirect_negative_case(self):
        report, raw = fixture()
        report.update(status="incomplete", smokeVerified=False, warningOrError=True)
        notice = {"id": 2132353751, "idName": "VALIDATION-SETTINGS", "severity": 256, "truncated": False,
                  "text": "vkCreateInstance(): The debug callback is already logging to stdout, but printf_to_stdout is also enabled. DebugPrintf will skip the debug callback in favor of a direct stdout write."}
        text = "MTP0 ordinary seq=0 group=1 lane=3 value=73"
        self.assertTrue(check("stdout", report, [notice], 2, text))
        self.assertFalse(check("no-info", report, [notice], 2, text))
        self.assertFalse(check("stdout", report, [notice], 2, ""))

    def test_host_loss_and_crashes_never_pass(self):
        report, raw = fixture()
        for changes in ({"hostDropped": 1}, {"hostTruncated": 1}, {"phase": "submit"}):
            self.assertFalse(check("ordinary", dict(report, **changes), raw, 0, ""))
        self.assertFalse(check("ordinary", report, raw, 3221225477, ""))


if __name__ == "__main__":
    unittest.main()
