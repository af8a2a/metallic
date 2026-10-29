"""Synthetic acceptance packages; no renderer or GPU is launched."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import VirtualShadowMapGPUProfile as profile


def frame(index, faces=2, scale=1.):
    sections = [{"name": "VSM page demand, cache and raster", "parent": profile.ROOT_PARENT,
                 "cpuOnly": False, "gpuMs": 50. * scale}]
    for face in range(faces):
        for stage in profile.STAGES:
            sections.append({"name": f"VSM {stage}", "parent": 0, "cpuOnly": False, "gpuMs": scale})
    return {"frame": index, "warmup": index == 0, "graphMs": 100. * scale,
            "nodes": [{"name": "Shadows", "type": "VirtualShadowMapLightsPass", "gpuMs": 60. * scale,
                       "sections": sections}],
            "memory": {"rayTracingAllocationBytes": 0, "clasAllocationBytes": 0,
                       "domains": {"RayTracing": {"peakAllocationBytes": 0}, "CLAS": {"peakAllocationBytes": 0}}}}


def add_work(row):
    counters = {name: index + 1 for index, name in enumerate(profile.WORK_COUNTERS)}
    row["shadow"] = {"gpuWork": {"version": 1, "enabled": True,
                      "faces": [{"lightIndex": 0, "face": face, "counters": counters.copy()} for face in range(2)],
                      "totals": {name: value * 2 for name, value in counters.items()}},
                     "totals": [1000 + index for index in range(32)]}
    return row


class GPUProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def seal(self, package):
        profile.acceptance.save(package / "Manifest.json", {
            str(path.relative_to(package)): profile.acceptance.digest(path)
            for path in sorted(package.rglob("*")) if path.is_file() and path.name != "Manifest.json"})

    def package(self, mode="timing", scales=(1., 2.)):
        package = self.root / mode
        package.mkdir()
        spec = {"mode": mode, "frames": 4, "warmupFrames": 1, "cases": [{"id": "Sponza"}]}
        profile.acceptance.save(package / "Cases.json", spec)
        profile.acceptance.save(package / "Integrity.json", {"sourceAndInputsUnchanged": True,
            "binariesUnchanged": True, "requestedRuns": len(scales), "completedRuns": len(scales)})
        profile.acceptance.save(package / "Processes.json", [
            {"exitCode": 0, "timedOut": False, "captureValid": True} for _ in scales])
        for process, scale in enumerate(scales, 1):
            run = package / f"run-{process}"
            run.mkdir()
            (run / "Process.log").write_text("Passed\n", encoding="utf-8")
            data = {"protocol": "virtual-shadow-map-acceptance-v2", "status": "passed", "mode": mode,
                    "validation": mode == "diagnostic", "performanceEligible": mode == "timing",
                    "cases": [{"id": "Sponza", "status": "passed", "noRayTracingAllocations": True,
                               "noRayQueryOrRTAS": True, "frames": [frame(index, scale=scale) for index in range(4)]}]}
            profile.acceptance.save(run / "VirtualShadowMapAcceptance.json", data)
        self.seal(package)
        return package

    def mutate_capture(self, package, mutate):
        path = package / "run-1/VirtualShadowMapAcceptance.json"
        data = profile.acceptance.read(path)
        mutate(data)
        profile.acceptance.save(path, data)
        self.seal(package)

    def test_leaf_stages_sum_faces_without_inclusive_parent(self):
        result = profile.profile_frame(frame(1))
        self.assertEqual(result["markMs"], 2.)
        self.assertEqual(result["stageSumMs"], 12.)
        self.assertEqual(result["otherShadowsMs"], 48.)
        self.assertEqual(result["faceCount"], 2)

    def test_missing_nonfinite_or_cpu_timestamps_are_not_zero(self):
        for value in (None, float("nan"), -1., True):
            row = frame(1)
            row["nodes"][0]["sections"][1]["gpuMs"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "timestamp"):
                profile.profile_frame(row)
        row = frame(1)
        row["nodes"][0]["sections"][1]["cpuOnly"] = True
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            profile.profile_frame(row)

    def test_missing_stage_and_nested_stage_are_rejected(self):
        row = frame(1)
        row["nodes"][0]["sections"].pop()
        with self.assertRaisesRegex(ValueError, "per-face"):
            profile.profile_frame(row)
        row = frame(1)
        row["nodes"][0]["sections"][2]["parent"] = 1
        with self.assertRaisesRegex(ValueError, "double count"):
            profile.profile_frame(row)

    def test_stage_sum_cannot_exceed_parent(self):
        row = frame(1)
        row["nodes"][0]["gpuMs"] = 5.
        with self.assertRaisesRegex(ValueError, "exceed"):
            profile.profile_frame(row)

    def test_verified_process_medians_are_independent_units(self):
        package = self.package(scales=(1., 3.))
        with mock.patch.object(profile.acceptance, "validate_capture", wraps=profile.acceptance.validate_capture) as verifier:
            result = profile.summarize_package(package)
        self.assertEqual(verifier.call_count, 2)
        self.assertEqual(result["cases"][0]["independentProcesses"], 2)
        values = result["cases"][0]["processMedianSummary"]["markMs"]
        self.assertEqual(values["count"], 2)
        self.assertEqual(values["median"], 4.)
        self.assertEqual(result["processes"][0]["metrics"]["markMs"]["count"], 3)

    def test_tampered_package_and_runtime_error_rejected(self):
        package = self.package()
        (package / "run-1/Process.log").write_text("[error] SDK failure\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash/file"):
            profile.summarize_package(package)
        self.seal(package)
        with self.assertRaisesRegex(ValueError, "Runtime/SDK"):
            profile.summarize_package(package)

    def test_base_verifier_rejects_skipped_frames_even_if_package_resealed(self):
        package = self.package()
        self.mutate_capture(package, lambda data: data["cases"][0]["frames"].pop())
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            profile.summarize_package(package)

    def test_base_verifier_rejects_diagnostics_disguised_as_timing(self):
        package = self.package()
        self.mutate_capture(package, lambda data: data.update(validation=True, performanceEligible=False))
        with self.assertRaisesRegex(ValueError, "Diagnostic"):
            profile.summarize_package(package)

    def test_no_post_warmup_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "No post-warmup"):
            profile.describe([])

    def test_counters_are_measured_and_page_state_is_separate(self):
        result = profile.work_frame(add_work(frame(1)), "diagnostic")
        self.assertTrue(result["gpuWorkAvailable"])
        self.assertEqual(result["markReceiverDispatches"], 2)
        self.assertEqual(result["requestedFinePages"], 1000)
        self.assertEqual(result["validMappedFinePages"], 1024)
        self.assertNotIn("triangleCapacity", result)

    def test_disabled_or_missing_counters_are_unavailable(self):
        row = frame(1)
        result = profile.work_frame(row, "timing")
        self.assertFalse(result["gpuWorkAvailable"])
        self.assertNotIn("projectionRays", result)
        row["shadow"] = {"gpuWork": {"version": 1, "enabled": False}}
        self.assertEqual(result, profile.work_frame(row, "diagnostic"))

    def test_work_counters_cannot_be_timing(self):
        with self.assertRaisesRegex(ValueError, "ordinary timing"):
            profile.work_frame(add_work(frame(1)), "timing")

    def test_duplicate_or_inconsistent_counter_faces_rejected(self):
        row = add_work(frame(1))
        row["shadow"]["gpuWork"]["faces"][1]["face"] = 0
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            profile.work_frame(row, "diagnostic")
        row = add_work(frame(1))
        row["shadow"]["gpuWork"]["totals"]["projectionRays"] += 1
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            profile.work_frame(row, "diagnostic")

    def test_report_cannot_modify_its_sealed_source(self):
        package = self.package()
        with self.assertRaisesRegex(ValueError, "outside the sealed"):
            profile.report(SimpleNamespace(source=package, output=package / "report", diagnostic=None))
        self.assertFalse((package / "report").exists())

    def test_declared_work_readback_cannot_be_missing(self):
        package = self.package(mode="diagnostic")
        self.mutate_capture(package, lambda data: data["cases"][0].update(gpuWorkDiagnostics=True))
        with self.assertRaisesRegex(ValueError, "missing frame readback"):
            profile.summarize_package(package)


if __name__ == "__main__":
    unittest.main()
