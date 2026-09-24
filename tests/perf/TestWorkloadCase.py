"""Synthetic negative evidence tests; GPU integration results live in build/."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("workload", ROOT / "Tools/Perf/WorkloadCase.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.case = w.load(ROOT / "Tools/Perf/WorkloadCase.ZorahWorkControl.json")
        self.case.update(rounds=1, sampleFrames=8, width=64, height=64, renderWidth=64, renderHeight=64)
        self.shader = {"mode": 5, "module": "Features/GPUDriven/GPUDrivenStreamWorkRaster",
                       "entryPoint": "streamClusterRasterWorkControlMain", "spirvFnv1a64": "123456",
                       "forceHardware": False, "asyncRequested": False, "workloadEnabled": False,
                       "snapshotFrozen": True, "paramsHzbValid": True}
        snapshot = {"activeGroups": 1, "cutHash": "123", "pageMappingsHash": "456",
                    "productionDispatches": [{**self.shader, "phase": phase, "queue": "graphics",
                                              "dispatch": "indirect", "argumentOffsetBytes": 48,
                                              "scope": "production-dispatch"} for phase in ("early", "late")]}
        for phase in w.PHASES:
            snapshot[phase] = {"softwareClusters": 2, "hardwareClusters": 1, "capacity": 16, "candidates": 3,
                               "softwareListHash": "789", "softwareDispatch": [2, 1, 1]}
            snapshot[phase.replace("Bins", "ClusterCull")] = {"candidateOverflow": 0}
        for resource in w.OUTPUTS:
            data = b"\1\0\0\0" * 4096
            (self.path / (resource + ".bin")).write_bytes(data)
            snapshot[resource] = {"file": resource + ".bin", "hash": w.fnv(data), "pixels": 4096}
        self.capture = {"status": "capture_complete", "protocol": self.case["protocol"],
                        "config": w.engine_config(self.case), "workloadCase": w.engine_config(self.case)["workloadCase"],
                        "outputExtent": [64, 64], "renderExtent": [64, 64], "hidden": True,
                        "validationRequested": False, "measurementKind": "normal-timing", "graphicsCaptureInjected": False,
                        "camera": {"temporalJitter": False}, "historyInvalidationPolicy": "reprojection-v1", "graph": [],
                        "cases": [{"round": 1, "variant": "swWorkControl", "fullHardware": False,
                                   "before": snapshot, "after": copy.deepcopy(snapshot), "frames": 8, "framesFile": "frames.json"}]}
        self.frames = [{"frame": i, "streaming": [{"softwareRaster": self.shader, "residentPages": 1,
                                                   "textureUpgrades": 0, "textureDowngrades": 0}],
                        "scopes": [{"path": path, "gpuMs": ms} for path, ms in
                                   (("root/RenderGraph GPU envelope", 4), ("root/Stream early/Software raster", .5),
                                    ("root/Stream late/Software raster", .1))]} for i in range(8)]

    def analyze(self):
        w.save(self.path / "Capture.json", self.capture)
        # Allow negative NaN input to exercise fail-closed parsing.
        (self.path / "frames.json").write_text(json.dumps(self.frames), encoding="utf-8")
        return w.analyze_run(self.path, self.case)

    def test_valid_repeatability_and_independent_unit(self):
        result = self.analyze()
        aa = w.compare_runs([result, copy.deepcopy(result), copy.deepcopy(result)], self.case)
        self.assertEqual(aa["status"], "stable")
        self.assertFalse(aa["candidateAccepted"])
        with self.assertRaises(ValueError):
            w.compare_runs([result], self.case)

    def test_unknown_case_fields_and_invalid_variant(self):
        for patch in ({"variant": "guess"}, {"scope": "dispatch"}, {"width": True},
                      {"warmupSeconds": float("nan")}, {"sampleFrame": 64}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                w.validate_case({**self.case, **patch})

    def test_wrong_shader_or_queue(self):
        for key, value in (("entryPoint", "streamClusterRasterLegacyMain"), ("queue", "compute"),
                           ("spirvFnv1a64", ""), ("paramsHzbValid", False)):
            before = self.capture["cases"][0]["before"]["productionDispatches"][0]
            original = before[key]
            before[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze()
            before[key] = original

    def test_zero_work(self):
        for phase in w.PHASES:
            self.capture["cases"][0]["before"][phase]["softwareClusters"] = 0
        with self.assertRaisesRegex(ValueError, "Zero software"):
            self.analyze()

    def test_frozen_input_drift(self):
        self.capture["cases"][0]["after"][w.PHASES[0]]["softwareListHash"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed within run"):
            self.analyze()

    def test_output_tampering(self):
        (self.path / "VBuffer.depth.bin").write_bytes(b"\0" * 16384)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.analyze()

    def test_missing_phase_and_nan_timestamp(self):
        self.frames[0]["scopes"][1]["gpuMs"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite timing"):
            self.analyze()

    def test_instrumented_cannot_qualify_aa(self):
        self.capture["graphicsCaptureInjected"] = True
        result = self.analyze()
        with self.assertRaisesRegex(ValueError, "instrumented"):
            w.compare_runs([result] * 3, self.case)

    def test_diagnostic_scope_in_measurement(self):
        self.frames[0]["scopes"].append({"path": "root/SW workload replay (diagnostic)", "gpuMs": 0.1})
        with self.assertRaisesRegex(ValueError, "Diagnostic work"):
            self.analyze()

    def test_cross_run_mismatch_and_noisy_timing(self):
        first = self.analyze()
        second = copy.deepcopy(first)
        second["identity"]["snapshot"]["cutHash"] = "different"
        with self.assertRaisesRegex(ValueError, "Cross-run"):
            w.compare_runs([first, first, second], self.case)
        second = copy.deepcopy(first)
        second["timings"]["graphGpuMs"]["median"] *= 2
        self.assertEqual(w.compare_runs([first, first, second], self.case)["status"], "inconclusive")

    def test_evidence_path_escape(self):
        self.capture["cases"][0]["before"]["VBuffer.depth"]["file"] = "../outside.bin"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.analyze()

    def test_failed_manifest_cannot_be_requalified(self):
        w.save(self.path / "Manifest.json", {"protocol": "metallic-workload-evidence-v1", "status": "failed"})
        with self.assertRaisesRegex(ValueError, "did not finish"):
            w.verify(self.path)


if __name__ == "__main__":
    unittest.main()
