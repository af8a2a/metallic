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


def add_build(row):
    counters = dict.fromkeys(profile.BUILD_COUNTERS, 0)
    counters.update(clusterGroups=3, clusterBoundsValidGroups=2, clusterSurvivingGroups=1,
        clusterLevelTests=6, clusterPageTests=15, clusterPageHits=1, clusterNearPlaneKeeps=1,
        clusterOutsideLevels=1, clusterEmptyLevels=3,
        clusterPageScan0=2, clusterPageScan1=1, clusterPageScan2To3=1, clusterPageScan4To7=1, clusterPageScan8To15=1)
    stats = [0] * 32
    stats[10], stats[11], stats[16] = 192, 3, 1
    shadow = row.setdefault("shadow", {})
    shadow.update(statistics=[stats.copy(), stats.copy()], totals=[value * 2 for value in stats])
    shadow["gpuBuildDiagnostics"] = {"version": 1, "enabled": True,
        "faces": [{"lightIndex": 0, "face": side, "counters": counters.copy()} for side in range(2)],
        "totals": {name: value * 2 for name, value in counters.items()}}
    return row


def retotal_build(row):
    diagnostic = row["shadow"]["gpuBuildDiagnostics"]
    diagnostic["totals"] = {name: sum(face["counters"][name] for face in diagnostic["faces"]) for name in profile.BUILD_COUNTERS}
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

    def test_build_sidecar_is_independent_and_optional(self):
        row = add_build(add_work(frame(1)))
        before = profile.work_frame(row, "diagnostic")
        result = profile.build_frame(row, "diagnostic")
        self.assertTrue(result["gpuBuildDiagnosticsAvailable"])
        self.assertEqual(result["clusterPageTests"], 30)
        del row["shadow"]["gpuBuildDiagnostics"]
        self.assertEqual(before, profile.work_frame(row, "diagnostic"))
        result = profile.build_frame(row, "diagnostic")
        self.assertFalse(result["gpuBuildDiagnosticsAvailable"])
        self.assertEqual(result["gpuBuildDiagnosticsState"], "missing")
        self.assertNotIn("clusterGroups", result)
        row["shadow"]["gpuBuildDiagnostics"] = {"version": 1, "enabled": False}
        result = profile.build_frame(row, "timing")
        self.assertFalse(result["gpuBuildDiagnosticsAvailable"])
        self.assertEqual(result["gpuBuildDiagnosticsState"], "disabled")
        self.assertNotIn("clusterGroups", result)

    def test_old_packages_do_not_claim_zero_build_work(self):
        result = profile.summarize_package(self.package("diagnostic", scales=(1.,)))
        self.assertEqual(result["cases"][0]["buildDiagnosticProcessMedianSummary"], {})
        self.assertTrue(all(not row["gpuBuildDiagnosticsAvailable"] for row in result["buildDiagnosticFrames"]))
        self.assertTrue(all("clusterGroups" not in row for row in result["buildDiagnosticFrames"]))

    def test_enabled_build_requires_faces_statistics_and_all_counter_names(self):
        for missing in ("faces", "totals"):
            row = add_build(frame(1))
            del row["shadow"]["gpuBuildDiagnostics"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "Missing"):
                profile.build_frame(row, "diagnostic")
        row = add_build(frame(1))
        del row["shadow"]["statistics"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            profile.build_frame(row, "diagnostic")
        row = add_build(frame(1))
        del row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"][profile.BUILD_COUNTERS[-1]]
        with self.assertRaisesRegex(ValueError, "malformed"):
            profile.build_frame(row, "diagnostic")

    def test_build_diagnostics_reject_timing_and_stale_disabled_data(self):
        row = add_build(frame(1))
        with self.assertRaisesRegex(ValueError, "ordinary timing"):
            profile.build_frame(row, "timing")
        row["shadow"]["gpuBuildDiagnostics"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "stale"):
            profile.build_frame(row, "diagnostic")
        for invalid in (None, {"version": True, "enabled": True}, {"version": 2, "enabled": True}):
            row["shadow"]["gpuBuildDiagnostics"] = invalid
            with self.subTest(schema=invalid), self.assertRaisesRegex(ValueError, "schema"):
                profile.build_frame(row, "diagnostic")

    def test_build_diagnostic_face_identity_and_totals(self):
        for key, value in (("face", 0), ("face", 6), ("lightIndex", 1)):
            row = add_build(frame(1))
            row["shadow"]["gpuBuildDiagnostics"]["faces"][1][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "malformed"):
                profile.build_frame(row, "diagnostic")
        row = add_build(frame(1))
        row["shadow"]["gpuBuildDiagnostics"]["totals"]["clusterGroups"] += 1
        with self.assertRaisesRegex(ValueError, "totals disagree"):
            profile.build_frame(row, "diagnostic")

    def test_build_unsigned_u64_and_total_overflow(self):
        for value in (-1, True, 1.5, 1 << 64):
            row = add_build(frame(1))
            row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"]["clusterGroups"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unsigned GPU counter"):
                profile.build_frame(row, "diagnostic")
        row = add_build(frame(1))
        for face in row["shadow"]["gpuBuildDiagnostics"]["faces"]:
            face["counters"]["clusterGroups"] = (1 << 32) + 7
        self.assertEqual(profile.build_frame(retotal_build(row), "diagnostic")["clusterGroups"], (1 << 33) + 14)
        for face in row["shadow"]["gpuBuildDiagnostics"]["faces"]:
            face["counters"]["clusterGroups"] = 1 << 63
        with self.assertRaisesRegex(ValueError, "overflow uint64"):
            profile.build_frame(retotal_build(row), "diagnostic")

    def test_build_group_level_and_outcome_relations(self):
        for field, value, error in (("clusterBoundsValidGroups", 4, "surviving <= valid"),
                ("clusterSurvivingGroups", 3, "surviving <= valid"),
                ("clusterLevelTests", 7, "clipmap count"), ("clusterPageHits", 2, "partition"),
                ("clusterPageScan1", 2, "partition")):
            row = add_build(frame(1))
            row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                profile.build_frame(retotal_build(row), "diagnostic")
        row = add_build(frame(1))
        counters = row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"]
        counters["clusterPageScan0"] -= 1
        counters["clusterPageScan1"] += 1
        with self.assertRaisesRegex(ValueError, "zero-scan"):
            profile.build_frame(retotal_build(row), "diagnostic")

    def test_build_histogram_weighted_bounds_and_last_bin_cap(self):
        for value in (14, 27):
            row = add_build(frame(1))
            row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"]["clusterPageTests"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "scan-length bounds"):
                profile.build_frame(retotal_build(row), "diagnostic")
        row = add_build(frame(1))
        for stats, face in zip(row["shadow"]["statistics"], row["shadow"]["gpuBuildDiagnostics"]["faces"]):
            stats[11] = 1
            face["counters"] = dict.fromkeys(profile.BUILD_COUNTERS, 0)
            face["counters"].update(clusterGroups=1, clusterBoundsValidGroups=1, clusterSurvivingGroups=1,
                clusterLevelTests=1, clusterPageHits=1, clusterPageTests=16384, clusterPageScan16384Plus=1)
        self.assertEqual(profile.build_frame(retotal_build(row), "diagnostic")["clusterPageTests"], 32768)
        row["shadow"]["gpuBuildDiagnostics"]["faces"][0]["counters"]["clusterPageTests"] = 16385
        with self.assertRaisesRegex(ValueError, "maximum 16384"):
            profile.build_frame(retotal_build(row), "diagnostic")

    def test_build_inactive_or_zero_dirty_faces_are_zero_but_disabled_culling_is_available(self):
        for index, value in ((10, 0), (14, 1), (16, 0)):
            row = add_build(frame(1))
            row["shadow"]["statistics"][0][index] = value
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Inactive or zero-dirty"):
                profile.build_frame(row, "diagnostic")
        row = add_build(frame(1))
        for face in row["shadow"]["gpuBuildDiagnostics"]["faces"]:
            face["counters"] = dict.fromkeys(profile.BUILD_COUNTERS, 0)
        result = profile.build_frame(retotal_build(row), "diagnostic")
        self.assertTrue(result["gpuBuildDiagnosticsAvailable"])
        self.assertEqual(result["clusterGroups"], 0)

    def test_partially_missing_build_readback_is_rejected(self):
        package = self.package("diagnostic", scales=(1.,))
        self.mutate_capture(package, lambda data: add_build(data["cases"][0]["frames"][0]))
        with self.assertRaisesRegex(ValueError, "missing declared frame readback"):
            profile.summarize_package(package)

    def test_declared_build_readback_cannot_be_entirely_missing(self):
        package = self.package("diagnostic", scales=(1.,))
        self.mutate_capture(package, lambda data: data["cases"][0].update(gpuBuildDiagnostics=True))
        with self.assertRaisesRegex(ValueError, "Declared GPU build diagnostics are missing"):
            profile.summarize_package(package)

    def test_build_report_exports_separate_diagnostic_rows_and_histogram(self):
        package = self.package("diagnostic", scales=(1.,))
        self.mutate_capture(package, lambda data: [add_build(row) for row in data["cases"][0]["frames"]])
        timing = self.package("timing", scales=(1.,))
        result = profile.summarize_package(package)
        self.assertEqual(result["processes"][0]["buildDiagnosticFrames"], 3)
        self.assertEqual(result["cases"][0]["buildDiagnosticProcessMedianSummary"]["clusterPageTests"]["median"], 30)
        self.assertEqual(len(result["buildDiagnosticFaces"]), 8)
        before = profile.acceptance.digest(package / "Manifest.json")
        output = self.root / "report"
        profile.report(SimpleNamespace(source=timing, output=output, diagnostic=package))
        rows = profile.acceptance.read(output / "BuildDiagnosticFrames.json")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["mode"] == "diagnostic" and row["package"] == "Diagnostic" for row in rows))
        self.assertTrue((output / "BuildDiagnosticFrames.csv").exists())
        self.assertTrue((output / "BuildDiagnosticFaces.csv").exists())
        self.assertTrue((output / "Diagnostic-1-build-histogram.png").exists())
        self.assertFalse((output / "Primary-1-build-histogram.png").exists())
        self.assertEqual(before, profile.acceptance.digest(package / "Manifest.json"))


if __name__ == "__main__":
    unittest.main()
