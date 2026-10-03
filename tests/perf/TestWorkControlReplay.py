"""Synthetic evidence adversaries; these are not GPU failure-injection tests."""
import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import WorkControlReplay as r
import NvPerf as n
import TestNvPerf as nv_fixtures


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        code = b"\x03\x02\x23\x07" + bytes(16)
        for name in ("Input.spv", "Device.spv"):
            (self.root / name).write_bytes(code)
        indices = {name: i + 1 for i, name in enumerate(sorted(r.BOUND))}
        push = [0] * 34
        for name, slot in {"pages": 0, "groups": 1, "pageTable": 2, "params": 3, "header": 7, "rasterBindings": 27, "bins": 29}.items():
            push[slot] = indices[name]
        (self.root / "Push.bin").write_bytes(struct.pack("<34I", *push))
        stages = ["archive-inputs", "production-before"] + ["restore-inputs", "isolated-dispatch", "compare-output"] * 2 + ["production-after"]
        self.report = {"protocol": "metallic-work-control-replay-v1", "status": "complete",
            "scope": "isolated-correctness-only", "counterEligible": False, "measurementKind": "diagnostic",
            "productionStatePublished": False, "sameRetainedExecution": True,
            "bindingPolicy": "same-typed-slots-private-allocations", "correctnessPasses": 2, "phase": "early",
            "productionShader": {"spirvFnv1a64": r.w.fnv(code)}, "bindings": [],
            "indirect": {"offset": 48, "dimensions": [1, 1, 1]},
            "submissions": [{"stage": s, "accepted": True, "completed": True} for s in stages]}
        for index, name in enumerate(sorted(r.NAMES)):
            data = bytearray(136)
            if name == "bins":
                struct.pack_into("<I", data, 44, indices["pixels"])
            if name == "rasterBindings":
                struct.pack_into("<I", data, 64, indices["instances"])
            if name == "arguments":
                struct.pack_into("<3I", data, 48, 1, 1, 1)
            if name == "pixels":
                data[0] = 7  # Nonzero initial state must be restored, never cleared.
            self.report["bindings"].append({"name": name, "shaderIndex": indices.get(name, 0xffffffff),
                "sourceAddress": str(1000 + index), "scratchAddress": str(2000 + index), "bytes": len(data), "stride": 4})
            for suffix in ("input", "production-before", "production-after"):
                (self.root / f"{name}-{suffix}.bin").write_bytes(data)
            control = bytes([9]) + bytes(data[1:]) if name == "pixels" else bytes(data)
            if name == "pixels":
                (self.root / "Control.bin").write_bytes(control)
            for p in range(2):
                (self.root / f"{p}-{name}-restored.bin").write_bytes(data)
                (self.root / f"{p}-{name}-output.bin").write_bytes(control)
        self.save()

    def save(self):
        r.w.save(self.root / "Replay.json", self.report)

    def reject(self):
        self.save()
        with self.assertRaises((ValueError, KeyError)):
            r.inspect_replay(self.root)

    def test_nonzero_initial_and_two_restored_passes(self):
        r.inspect_replay(self.root)

    def test_second_pass_cannot_skip_restore(self):
        self.report["submissions"].pop(5)
        self.reject()

    def test_second_pass_cannot_clear_initial_pixels(self):
        (self.root / "1-pixels-restored.bin").write_bytes(bytes(136))
        self.reject()

    def test_direct_output_byte_mismatch(self):
        (self.root / "0-pixels-output.bin").write_bytes(bytes(136))
        self.reject()

    def test_readonly_write_rejected(self):
        (self.root / "1-groups-output.bin").write_bytes(b"x" * 136)
        self.reject()

    def test_page_age_request_and_history_leaks_without_depth_change(self):
        for name in ("pageTable", "requests", "hzb0", "lodState", "instanceVisibility"):
            path = self.root / f"{name}-production-after.bin"
            original = path.read_bytes()
            path.write_bytes(b"x" + original[1:])
            self.reject()
            path.write_bytes(original)

    def test_failure_terminals_cannot_be_verified(self):
        for cause in ("cancelled", "timeout", "submit_failed", "device_lost", "restore_failed"):
            self.report.update(status="failed", error=cause)
            self.reject()

    def test_accepted_but_unretired_submit_rejected(self):
        self.report["submissions"][3]["completed"] = False
        self.reject()

    def test_production_alias_rejected(self):
        self.report["bindings"][0]["scratchAddress"] = self.report["bindings"][1]["sourceAddress"]
        self.reject()

    def test_wrong_nested_binding(self):
        path = self.root / "bins-input.bin"
        data = bytearray(path.read_bytes())
        struct.pack_into("<I", data, 44, 999)
        path.write_bytes(data)
        self.reject()

    def test_stale_spirv_and_indirect_identity(self):
        self.report["productionShader"]["spirvFnv1a64"] = "0"
        self.reject()
        self.report["productionShader"]["spirvFnv1a64"] = r.w.fnv((self.root / "Input.spv").read_bytes())
        self.report["indirect"]["offset"] = 0
        self.reject()

    def test_correctness_cannot_promote_to_counter_evidence(self):
        self.report["counterEligible"] = True
        self.reject()

    def test_legacy_profile_cannot_be_promoted(self):
        fixture = nv_fixtures.NvPerfTests()
        fixture.setUp()
        n.validate_profile(fixture.profile, n.DEFAULT_METRICS)
        fixture.profile["scope"] = "dispatch-exclusive-isolated"
        with self.assertRaises(ValueError):
            n.validate_profile(fixture.profile, n.DEFAULT_METRICS)

    def counters(self):
        self.report.update(counterPasses=2, counterEligible=True, scope="isolated-dispatch-RHI-exclusive",
                           submissionIsolation="all-RHI-queues-owner-lease")
        stages = ["archive-inputs", "production-before"] + ["restore-inputs", "isolated-dispatch", "compare-output"] * 2
        stages += ["production-gate"] + ["restore-inputs", "isolated-dispatch", "compare-output"] * 2 + ["production-after"]
        self.report["submissions"] = [{"stage": s, "accepted": True, "completed": True} for s in stages]
        for name in r.NAMES:
            for p in (2, 3):
                for suffix in ("restored", "output"):
                    (self.root / f"{p}-{name}-{suffix}.bin").write_bytes((self.root / f"0-{name}-{suffix}.bin").read_bytes())
            (self.root / f"{name}-production-gate.bin").write_bytes((self.root / f"{name}-production-before.bin").read_bytes())
        (self.root / "nvperf").mkdir()
        for name in n.RAW_FILES:
            (self.root / "nvperf" / name).write_bytes(b"synthetic-not-an-sdk-image")
        self.profile = {"protocol": "metallic-nvperf-isolated-v1", "status": "complete", "scope": self.report["scope"],
            "measurementKind": "diagnostic", "clockPolicy": "unaltered", "requiredPasses": 2, "passesCollected": 2,
            "numRangesDropped": 0, "numTraceBytesDropped": 0, "metricNames": ["gpu__time_duration.sum"],
            "ranges": [{"name": "WorkControl/isolated/early", "index": 0, "metrics": [
                {"name": "gpu__time_duration.sum", "value": 1, "dimUnits": []}]}]}
        r.w.save(self.root / "nvperf/NvPerf.json", self.profile)
        self.save()

    def test_multi_pass_ledger_and_raw_bytes(self):
        self.counters()
        r.inspect_replay(self.root)

    def test_counter_second_pass_reset_missing(self):
        self.counters()
        self.report["submissions"].pop(-4)
        self.reject()

    def test_counter_collection_cannot_skip_production_gate(self):
        self.counters()
        (self.root / "pageTable-production-gate.bin").write_bytes(b"x" * 136)
        self.reject()

    def test_counter_final_output_mismatch(self):
        self.counters()
        (self.root / "3-pixels-output.bin").write_bytes(bytes(136))
        self.reject()

    def test_missing_pass_and_dropped_counter_data(self):
        self.counters()
        for key, value in (("passesCollected", 1), ("numRangesDropped", 1), ("numTraceBytesDropped", 1)):
            profile = {**self.profile, key: value}
            r.w.save(self.root / "nvperf/NvPerf.json", profile)
            self.reject()

    def test_wrong_counter_phase(self):
        self.counters()
        self.profile["ranges"][0]["name"] = "WorkControl/isolated/late"
        r.w.save(self.root / "nvperf/NvPerf.json", self.profile)
        self.reject()

    def test_unowned_submission_scope(self):
        self.counters()
        self.report["submissionIsolation"] = "same-queue-only"
        self.reject()

    def test_failed_injection_not_upgraded(self):
        self.report["faultInjection"] = "state-leak"
        self.reject()


if __name__ == "__main__":
    unittest.main()
