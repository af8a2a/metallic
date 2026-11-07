"""Evidence-contract tests; no renderer, GPU or benchmark process is launched."""

import copy
import unittest
from unittest.mock import patch

from MeasureProductionCPUJobs import benchmark_environment, block_comparisons, effective_identity, summarize_device_memory, summarize_frames, validation_messages
from ProcessResources import summarize_process
from MeasureProductionMeshlets import geometry_identity, source_dependencies
from pathlib import Path


def capture():
    return {"workerCount": 4, "validationRequested": False, "scene": "SuperSponza.usda",
            "camera": {"route": "fixed"}, "graph": [{"name": "Final"}],
            "outputExtent": [1280, 720], "renderExtent": [1280, 720], "recordingWorkerLimit": 4,
            "scopes": [{"id": 0, "path": "Frame"}, {"id": 1, "path": "Frame/Prepare"},
                       {"id": 2, "path": "RenderGraph GPU envelope"}]}


class ProductionCPUJobsTests(unittest.TestCase):
    def test_nested_and_worker_scopes_are_kept_independent(self):
        row = {"frameMs": 12., "scopes": [{"id": 0, "cpuMs": 10.}, {"id": 1, "cpuMs": 3.},
                                           {"id": 2, "cpuMs": 0., "gpuMs": 5.}],
               "scheduling": {"enabled": True, "executeNs": 1000, "workerExecuteNs": 1400}}
        result = summarize_frames([row], capture())
        self.assertEqual(result["cpuScopesMs"]["Frame"]["p50"], 10.)
        self.assertEqual(result["cpuScopesMs"]["Frame/Prepare"]["p50"], 3.)
        self.assertEqual(result["schedulingNs"]["workerExecuteNs"]["p50"], 1400)
        self.assertEqual(result["gpuFrameMs"]["p50"], 5.)
        self.assertNotIn("criticalPathMs", result)

    def test_missing_gpu_memory_stays_null_and_is_not_process_memory(self):
        result = summarize_device_memory([{"privateBytes": 123456}], {})
        self.assertIsNone(result["deviceLocalAllocatedSampledPeakBytes"])
        self.assertIsNone(result["deviceLocalUsageSampledPeakBytes"])
        self.assertTrue(result["missing"])

    def test_device_heap_quantities_and_domain_peaks_are_distinct(self):
        frame = {"deviceMemory": {"heaps": [
            {"deviceLocal": True, "allocationBytes": 12, "blockBytes": 20, "usageBytes": 24},
            {"deviceLocal": False, "allocationBytes": 1000, "blockBytes": 2000, "usageBytes": 3000}],
            "domains": [{"peakAllocationBytes": 30}, {"peakAllocationBytes": 40}], "availableBytes": 8}}
        after = {"deviceMemoryAfterLoad": {"heaps": [
            {"deviceLocal": True, "allocationBytes": 16, "blockBytes": 18, "usageBytes": 20}],
            "domains": {"Geometry": {"peakAllocationBytes": 45}}, "availableBytes": 5}}
        result = summarize_device_memory([frame], after)
        self.assertEqual(result["deviceLocalAllocatedSampledPeakBytes"], 16)
        self.assertEqual(result["deviceLocalBlockSampledPeakBytes"], 20)
        self.assertEqual(result["deviceLocalUsageSampledPeakBytes"], 24)
        self.assertEqual(result["domainAllocationPeakBytes"], {"Other": 30, "Geometry": 45})
        self.assertEqual(result["minimumAvailableBytes"], 5)
        self.assertNotIn("summedDomainPeakBytes", result)
        self.assertIsNone(result["peakTrackedDeviceLocalBytes"])
        frame["deviceMemory"].update(trackedAllocationBytes=20, trackedDeviceLocalBytes=12,
                                    peakTrackedAllocationBytes=60, peakTrackedDeviceLocalBytes=50)
        result = summarize_device_memory([frame], after)
        self.assertEqual(result["peakTrackedDeviceLocalBytes"], 50)
        self.assertEqual(result["peakTrackedAllocationBytes"], 60)

    def test_bad_scope_or_nan_timing_rejects_evidence(self):
        with self.assertRaises(ValueError):
            summarize_frames([{"frameMs": float("nan")}], capture())
        with self.assertRaises(ValueError):
            summarize_frames([{"frameMs": 1., "scopes": [{"id": 100, "cpuMs": 2.}]}], capture())

    def test_measured_frame_range_rejects_duplicates_and_warmup_rows(self):
        rows = [{"frame": 120, "frameMs": 1.}, {"frame": 121, "frameMs": 1.}]
        self.assertEqual(summarize_frames(rows, capture())["measuredFrameRange"], {"first": 120, "last": 121})
        rows[-1]["frame"] = 120
        with self.assertRaises(ValueError):
            summarize_frames(rows, capture())
        with self.assertRaises(ValueError):
            summarize_frames([{"frameMs": 1., "phase": "warmup"}], capture())

    def test_exclusive_segments_require_an_owner_overlap_contract(self):
        row = {"frameMs": 1., "exclusiveCpuSegmentsMs": {"record": .5}}
        with self.assertRaises(ValueError):
            summarize_frames([row], capture())
        definition = capture()
        definition["exclusiveCpuSegmentsDefinition"] = "Non-overlapping coordinator intervals"
        self.assertEqual(summarize_frames([row], definition)["exclusiveCpuSegmentsMs"]["record"]["p50"], .5)

    def test_coordinator_partition_rejects_overlap_and_keeps_unattributed_time(self):
        row = {"frameMs": 5., "cpuFramePathContract": "nonoverlapping-coordinator-wall-segments-v1",
               "cpuFramePathSegmentsMs": [{"name": "wait", "startMs": 0., "endMs": 1.},
                                           {"name": "draw", "startMs": 2., "endMs": 4.}]}
        result = summarize_frames([row], capture())
        self.assertEqual(result["cpuFramePathSegmentsMs"]["Unattributed frame envelope"]["mean"], 2.)
        row["cpuFramePathSegmentsMs"][1]["startMs"] = .5
        with self.assertRaises(ValueError):
            summarize_frames([row], capture())

    def test_effective_workload_rejects_wrong_workers_validation_or_missing_identity(self):
        identity = effective_identity(capture(), 4)
        self.assertEqual(identity["scene"], "SuperSponza.usda")
        for field, value in (("workerCount", 8), ("validationRequested", True), ("camera", None)):
            invalid = copy.deepcopy(capture())
            invalid[field] = value
            with self.assertRaises(ValueError):
                effective_identity(invalid, 4)

    def test_validation_purpose_and_clean_fixed_environment(self):
        validated = capture()
        validated["validationRequested"] = True
        self.assertTrue(effective_identity(validated, 4, purpose="validation")["validationRequested"])
        with patch.dict("os.environ", {"METALLIC_NSIGHT_OLD": "1", "METALLIC_SHADER_TRACE": "1",
                                       "METALLIC_FULL_ROAM_OUTPUT": "stale"}, clear=True):
            environment, record = benchmark_environment(True, True)
        self.assertNotIn("METALLIC_NSIGHT_OLD", environment)
        self.assertNotIn("METALLIC_FULL_ROAM_OUTPUT", environment)
        self.assertEqual(environment["METALLIC_FULL_ROAM_NO_VSYNC"], "1")
        self.assertEqual(environment["METALLIC_REFLEX_MODE"], "off")
        self.assertEqual(environment["METALLIC_SHADER_TRACE"], "0")
        self.assertEqual(environment["METALLIC_DEBUG_VALIDATION"], "1")
        self.assertEqual(environment["METALLIC_RENDER_SCHEDULING_DIAGNOSTICS"], "1")
        self.assertIn("METALLIC_NSIGHT_OLD", record["removedNames"])

    def test_validation_log_stops_on_any_warning_without_double_counting_lines(self):
        messages = validation_messages("startup\nVulkan validation: VUID-Example SYNC-HAZARD\nVUID-Second\n")
        self.assertEqual(messages["matchedLineCount"], 2)
        self.assertEqual(messages["patternLineCounts"]["VUID-"], 2)
        self.assertEqual(messages["firstMessage"], "Vulkan validation: VUID-Example SYNC-HAZARD")
        self.assertEqual(validation_messages("clean log")["matchedLineCount"], 0)

    def test_native_validation_conditions_are_required_in_strict_captures(self):
        value = capture()
        value.update(diagnostics=False, graphicsCaptureInjected=False, shaderDebugMode="disabled", vsyncRequested=False,
                     synchronizationValidationRequested=False, validationEnabled=False,
                     synchronizationValidationEnabled=False, validationMessengerActive=False,
                     reflex={"requestedMode": 0, "frameLimitUs": 0, "available": True})
        self.assertFalse(effective_identity(value, 4, strict_conditions=True)["validationEnabled"])
        value["validationEnabled"] = True
        with self.assertRaises(ValueError):
            effective_identity(value, 4, strict_conditions=True)

    def test_core_only_validation_accepts_disabled_sync_and_rejects_native_mismatch(self):
        value = capture()
        value.update(validationRequested=True, validationEnabled=True, validationMessengerActive=True,
                     config={"synchronizationValidation": False}, synchronizationValidationRequested=False,
                     synchronizationValidationEnabled=False, diagnostics=True, graphicsCaptureInjected=False,
                     shaderDebugMode="disabled", vsyncRequested=False,
                     reflex={"requestedMode": 0, "frameLimitUs": 0, "available": True})
        identity = effective_identity(value, 4, purpose="validation", diagnostics=True, strict_conditions=True)
        self.assertFalse(identity["synchronizationValidationEnabled"])
        value["synchronizationValidationEnabled"] = True
        with self.assertRaises(ValueError):
            effective_identity(value, 4, purpose="validation", diagnostics=True, strict_conditions=True)

    def test_sync_validation_request_must_match_config_and_performance_must_disable_it(self):
        value = capture()
        value.update(validationRequested=True, validationEnabled=True, validationMessengerActive=True,
                     config={"synchronizationValidation": True}, synchronizationValidationRequested=False,
                     synchronizationValidationEnabled=False, diagnostics=True, graphicsCaptureInjected=False,
                     shaderDebugMode="disabled", vsyncRequested=False,
                     reflex={"requestedMode": 0, "frameLimitUs": 0, "available": True})
        with self.assertRaises(ValueError):
            effective_identity(value, 4, purpose="validation", diagnostics=True, strict_conditions=True)
        value.update(synchronizationValidationRequested=True, synchronizationValidationEnabled=True)
        self.assertTrue(effective_identity(value, 4, purpose="validation", diagnostics=True, strict_conditions=True)["synchronizationValidationEnabled"])
        value.update(validationRequested=False, validationEnabled=False, validationMessengerActive=False)
        with self.assertRaises(ValueError):
            effective_identity(value, 4, purpose="performance", diagnostics=True, strict_conditions=True)

    def test_null_scheduling_is_valid_when_diagnostics_are_off(self):
        result = summarize_frames([{"frame": 1, "frameMs": 1, "scheduling": None}], capture())
        self.assertEqual(result["schedulingNs"], {})

    def test_process_os_high_water_is_separate_from_sampled_peak(self):
        rows = [{"elapsedNs": 100, "cpuNs": 30, "privateBytes": 12, "peakPagefileBytes": 24,
                 "workingSetBytes": 8, "peakWorkingSetBytes": 16, "threadCount": 4, "queryWallNs": 2},
                {"elapsedNs": 200, "cpuNs": 60, "privateBytes": None, "peakPagefileBytes": None,
                 "workingSetBytes": None, "peakWorkingSetBytes": None, "threadCount": None,
                 "processLifetimeNs": 180, "queryWallNs": 3, "missing": ["process exited"]}]
        result = summarize_process(rows, 210)
        self.assertEqual(result["lifetimePeakPrivateBytes"], 24)
        self.assertEqual(result["sampledPeakPrivateBytes"], 12)
        self.assertEqual(result["sampledPeakThreads"], 4)
        self.assertEqual(result["processCpuNs"], 60)
        self.assertEqual(result["averageCpuCores"], 60 / 180)
        self.assertEqual(result["samplerQueryWallNs"], 5)

    def test_abba_uses_independent_process_medians_and_rejects_incomplete_blocks(self):
        rows = [{"runId": index, "phase": "measured", "workers": 4, "block": 0,
                 "candidate": candidate, "frameMs": {"p50": value}}
                for index, (candidate, value) in enumerate(
                    (("legacy", 12), ("shared", 8), ("shared", 12), ("legacy", 8)))]
        result = block_comparisons(rows)
        self.assertEqual(result[0]["frameMs"]["legacyOverSharedRatio"], 1.)
        with self.assertRaises(ValueError):
            block_comparisons(rows[:-1])

    def test_cook_identity_excludes_output_path_and_process_statistics(self):
        first = {"status": "complete", "asset": "run-a.bin", "fileBytes": 100, "primitives": 1,
                 "instances": 1, "groups": 2, "pages": 2, "storedPayloadBytes": 80,
                 "elapsedSecondsThisInvocation": 10.}
        second = dict(first, asset="run-b.bin", elapsedSecondsThisInvocation=12.)
        self.assertEqual(geometry_identity(first), geometry_identity(second))
        second["pages"] = 3
        self.assertNotEqual(geometry_identity(first), geometry_identity(second))
        with self.assertRaises(ValueError):
            geometry_identity({"status": "paused"})

    def test_cook_source_dependencies_include_embedded_source_and_reject_remote_buffers(self):
        with patch.object(Path, "read_text", return_value='{"buffers":[{"uri":"data:application/octet-stream;base64,AA=="}]}'):
            self.assertEqual(source_dependencies(Path("input.gltf")), [Path("input.gltf")])
        with patch.object(Path, "read_text", return_value='{"buffers":[{"uri":"https://example.invalid/buffer.bin"}]}'):
            with self.assertRaises(ValueError):
                source_dependencies(Path("input.gltf"))


if __name__ == "__main__":
    unittest.main()
