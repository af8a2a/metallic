"""Reject misleading compile/GPU experiment evidence; fixtures are synthetic."""

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import RunComputePipelineExperiments as experiment


def native_fixture():
    return dict(status="passed", validation_errors=0, build_configuration="Release", slang_build="test",
        shader_target=dict(id="vulkan", revision=1),
        configuration=dict(debug_mode="disabled", validation=False, pipeline_cache="empty", compile_only=False,
                           slang_optimization="O1", compute_backend="pipeline", disable_driver_optimization=False),
        environment={experiment.driver.CACHE_ENVIRONMENT: "disabled"}, device=dict(name="test", driver_version=1),
        samples=[dict(status="passed", pipeline_status="created", request=dict(index=146, module="Guides", entry="main"),
            pipeline_wall_ms=4.5,
            execution_model=5, spirv_fnv1a64="1234", spirv_bytes=512, effective_descriptor_mode="mapped",
            resource_heap_stride=0, sampler_heap_stride=0, shader_cache_hit=True, effective_slang_optimization="O1",
            spirv_statistics=dict(instruction_count=64, function_count=2, basic_block_count=8),
            pipeline=dict(compute_backend="pipeline", native_create_ms=4, driver_ms=3, cache_lock_wait_ms=0,
                          native_binary_cache_hit=False, native_binary_driver_rejected=False))])


def workload_fixture():
    config = dict(warmupFrames=2, timingFrames=2,
                  cases=[dict(id="Opaque", requiredTiming=["PathTrace", "Path trace shading"])])
    case = dict(id="Opaque", width=1, height=1, format="RGBA32F", image="color.bin", validHDR=True,
                resolvedGraph=dict(nodes=[1]), captures=[dict(output="PathTrace.motionVectors", frame=4,
                file="motion.bin", bytes=4, format=25)], frames=[])
    for index in (2, 3):
        case["frames"].append(dict(frame=index, executionId=index, warmup=False, graphMs=3., asyncComputeBranches=0,
            nodes=[dict(name="PathTrace", queue=0, gpuMs=2., sections=[dict(name="Path trace shading", gpuMs=1.)])]))
    return config, dict(validation=False, cases=[case])


class NativeExperimentTests(unittest.TestCase):
    def test_compile_schedule_prefills_each_level_and_requires_warm_measured_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "bench.exe"
            executable.write_bytes(b"fixture only")
            source = root / "Guides.slang"
            source.write_text("fixed source")
            calls = []

            def process(command, environment, destination, timeout):
                calls.append((command, environment))
                destination.mkdir()
                optimization = command[command.index("--slang-optimization") + 1]
                report = native_fixture()
                sample = report["samples"][0]
                compiled_only = "--compile-only" in command
                report["environment"] = environment
                report["configuration"].update(slang_optimization=optimization, compile_only=compiled_only,
                    disable_driver_optimization="--disable-driver-optimization" in command)
                sample.update(effective_slang_optimization=optimization, shader_cache_hit=not compiled_only,
                    spirv_fnv1a64=optimization + "-code", dependencies=[str(source)], compile_wall_ms=3.,
                    device_spirv_bytes=512, device_spirv_fnv1a64=optimization + "-device-code",
                    device_spirv_statistics=sample["spirv_statistics"], device_shader_content_hash=optimization + "-content")
                if compiled_only:
                    sample["pipeline_status"] = "compile_only"
                experiment.save(destination / "report.json", report)
                return dict(exit_code=0, timed_out=False), report

            args = SimpleNamespace(executable=executable, optimizations=["O1", "O0", "O2"], backends=["pipeline"],
                driver_optimization="both", native_binary_cache=False, descriptor_mode="mapped", validation=False,
                pairs=1, request_index=146, timeout=10)
            output = root / "evidence"
            output.mkdir()
            summary = dict(command="compile", runs=[], excluded_runs=[])
            with patch.object(experiment, "input_identity", return_value=experiment.snapshots([source, executable])), \
                 patch.object(experiment.driver, "run_process", side_effect=process):
                experiment.compile_experiment(args, output, summary)
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(summary["excluded_runs"]), 3)
            self.assertEqual(len(summary["runs"]), 6)
            self.assertEqual(set(summary["device_shaders"]), {"O0", "O1", "O2"})
            self.assertTrue(all(environment[experiment.driver.CACHE_ENVIRONMENT] == "disabled" for _, environment in calls))

    def test_pipeline_flag_pairs_share_bytecode_identity(self):
        report = native_fixture()
        shader = experiment.shader_identity(report, report["samples"][0])
        conditions = experiment.arms(["O1"], ["pipeline"], "both")
        experiment.verify_native(report, shader, conditions[0])
        report["configuration"]["disable_driver_optimization"] = True
        experiment.verify_native(report, shader, conditions[1])
        report["samples"][0]["spirv_fnv1a64"] = "drift"
        with self.assertRaisesRegex(ValueError, "SPIR-V"):
            experiment.verify_native(report, shader, conditions[1])

    def test_rejects_backend_settings_cache_and_unknown_timing_drift(self):
        original = native_fixture()
        shader = experiment.shader_identity(original, original["samples"][0])
        arm, = experiment.arms(["O1"], ["pipeline"], "normal")
        mutations = [lambda r: r["samples"][0].update(shader_cache_hit=False),
                     lambda r: r["samples"][0]["spirv_statistics"].update(function_count=3),
                     lambda r: r["configuration"].update(compute_backend="shader-object"),
                     lambda r: r["configuration"].update(pipeline_cache="reuse"),
                     lambda r: r["environment"].update({experiment.driver.CACHE_ENVIRONMENT: "enabled"}),
                     lambda r: r["samples"][0]["pipeline"].update(native_create_ms=float("nan")),
                     lambda r: r["samples"][0]["pipeline"].update(driver_ms=-1),
                     lambda r: r["device"].update(driver_version=2)]
        for mutate in mutations:
            report = copy.deepcopy(original)
            mutate(report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                experiment.verify_native(report, shader, arm, device=original["device"])

    def test_rejected_binary_fallback_cannot_pass_as_import(self):
        report = native_fixture()
        report["configuration"]["compute_backend"] = "shader-object"
        pipeline = report["samples"][0]["pipeline"]
        pipeline.update(compute_backend="shader-object", driver_ms=None, native_binary_cache_hit=True)
        shader = experiment.shader_identity(report, report["samples"][0])
        arm, = experiment.arms(["O1"], ["shader-object"], "normal")
        experiment.verify_native(report, shader, arm, imported=True)
        for mutation in (dict(native_binary_cache_hit=False), dict(native_binary_driver_rejected=True)):
            original = copy.deepcopy(pipeline)
            pipeline.update(mutation)
            with self.assertRaises(ValueError):
                experiment.verify_native(report, shader, arm, imported=True)
            pipeline.update(original)

    def test_excludes_invalid_shader_object_flags_and_duplicate_conditions(self):
        conditions = experiment.arms(["O0", "O1", "O2"], ["shader-object", "pipeline"], "both")
        self.assertEqual(len(conditions), 9)
        self.assertFalse(any(arm["backend"] == "shader-object" and arm["disable_driver_optimization"] for arm in conditions))
        with self.assertRaises(ValueError):
            experiment.arms(["O1"], ["shader-object"], "disabled")
        with self.assertRaises(ValueError):
            experiment.arms(["O1", "O1"], ["pipeline"], "normal")
        for block in range(8):
            ordered = experiment.balanced_order(conditions, block)
            self.assertEqual({arm["name"] for arm in ordered}, {arm["name"] for arm in conditions})

    def test_binary_file_drift_and_empty_extraction_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(ValueError):
                experiment.binary_snapshot(directory)
            file = directory / "native.bin"
            file.write_bytes(b"binary")
            identity = experiment.binary_snapshot(directory)
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                experiment.check_snapshots(identity)


class ProductionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.configuration, self.report = workload_fixture()
        for name in ("A", "B"):
            directory = self.root / name
            directory.mkdir()
            (directory / "color.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 1))
            (directory / "motion.bin").write_bytes(struct.pack("<2e", .1, .2))
            experiment.save(directory / "MaterialBaseline.json", self.report)

    def tearDown(self):
        self.temp.cleanup()

    def compare(self):
        return experiment.compare_readbacks(self.root / "A/MaterialBaseline.json", self.root / "B/MaterialBaseline.json",
                                            self.configuration, 2e-5, 2e-5)

    def test_actual_motion_guide_format_and_predeclared_correctness_gate(self):
        self.assertTrue(self.compare()[0]["qualified"])
        (self.root / "B/motion.bin").write_bytes(struct.pack("<2e", .1, .21))
        result = self.compare()[0]
        self.assertFalse(result["qualified"])
        self.assertFalse(result["readbacks"][1]["within_tolerance"])

    def test_recipe_capture_records_actual_device_and_shader_policy(self):
        directory = self.root / "recipes"
        directory.mkdir()
        raw = struct.pack("<5I", 0x07230203, 0x10600, 0, 1, 0)
        (directory / "input.spv").write_bytes(raw)
        shader = dict(spirv_file="input.spv", spirv_bytes=20, module_name="Guides", source_entry_point="main",
                      slang_optimization="O1", debug_mode="release", input_spirv_hash="0123456789abcdef",
                      descriptor_heap_mode="mapped")
        recipe = dict(schema="metallic_pipeline_recipe", source_device=dict(device_name="fixture GPU", driver_version=1),
                      pipeline_type="compute", state=dict(compute_backend=0, disable_optimization=False),
                      shaders=dict(compute=shader))
        manifest = directory / "compute-fixture.recipe.json"
        experiment.save(manifest, recipe)
        arm, = experiment.arms(["O1"], ["pipeline"], "normal")
        evidence = experiment.recipe_evidence(directory, arm)
        self.assertEqual(evidence["selected_compute_recipe_count"], 1)
        self.assertEqual(evidence["shaders"][0]["spirv_bytes"], 20)
        shader["slang_optimization"] = "O0"
        experiment.save(manifest, recipe)
        with self.assertRaisesRegex(ValueError, "front-end"):
            experiment.recipe_evidence(directory, arm)

    def test_readback_nonfinite_truncated_and_unknown_format_rejected(self):
        for raw in (struct.pack("<4f", float("nan"), 2, 3, 1), b"\0\0\0\0"):
            (self.root / "B/color.bin").write_bytes(raw)
            with self.assertRaises(ValueError):
                self.compare()
        (self.root / "B/color.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 1))
        changed = copy.deepcopy(self.report)
        changed["cases"][0]["captures"][0]["format"] = 1
        for name in ("A", "B"):
            experiment.save(self.root / name / "MaterialBaseline.json", changed)
        with self.assertRaises(ValueError):
            self.compare()

    def test_workload_camera_and_capture_contract_drift_rejected(self):
        for field, value in (("resolvedGraph", dict(nodes=[2])), ("validHDR", False), ("width", 2), ("captures", [])):
            changed = copy.deepcopy(self.report)
            changed["cases"][0][field] = value
            experiment.save(self.root / "B/MaterialBaseline.json", changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.compare()

    def test_missing_duplicate_or_wrong_scope_gpu_timestamps_rejected(self):
        timing = experiment.gpu_timings(self.report, self.configuration)
        self.assertEqual(timing["Opaque"]["dispatch_scope_ms"]["median"], 1.)
        mutations = [lambda r: r["cases"][0]["frames"].pop(),
                     lambda r: r["cases"][0]["frames"][1].update(executionId=2),
                     lambda r: r["cases"][0]["frames"][0].update(warmup=True),
                     lambda r: r["cases"][0]["frames"][0].update(asyncComputeBranches=1),
                     lambda r: r["cases"][0]["frames"][0]["nodes"][0].update(queue=1),
                     lambda r: r["cases"][0]["frames"][0]["nodes"][0]["sections"][0].update(gpuMs=None),
                     lambda r: r["cases"][0]["frames"][0]["nodes"][0]["sections"].clear()]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                experiment.gpu_timings(report, self.configuration)

    def test_runtime_must_confirm_effective_slang_policy(self):
        log = self.root / "stdout.log"
        arm, = experiment.arms(["O1"], ["pipeline"], "normal")
        log.write_text("[Slang] Begin compile Guides.main (debugMode=0, optimization=O1)\n"
                       "[ShaderCache] Loaded Guides.main from 'a.spv' (optimization=O1)\n"
                       "[ComputeCreatePolicy] backend=0 disableOptimization=false pso=0123456789abcdef\n")
        self.assertEqual(experiment.compiler_log_evidence(log, arm)["confirmed_shader_count"], 2)
        for text in ("", "[Slang] Begin compile Guides.main (optimization=O0)",
                     "[Slang] Begin compile Guides.main (optimization=O1)\n"
                     "[ComputeCreatePolicy] backend=0 disableOptimization=true pso=0123456789abcdef\n",
                     "[Slang] Begin compile Guides.main (optimization=O1)\nVUID-Test-validation"):
            log.write_text(text)
            with self.assertRaises(ValueError):
                experiment.compiler_log_evidence(log, arm)

    def test_frame_window_resize_rejects_temporal_camera_fixture(self):
        graph = self.root / "Graph.json"
        experiment.save(graph, dict(outputs=["PathTrace.color", "PathTrace.motionVectors"]))
        cases = self.root / "Original.json"
        configuration = dict(frames=7, warmupFrames=2, timingFrames=4, assetFiles=[],
                             cases=[dict(id="Opaque", graph="Graph.json", width=512, height=512,
                                         output="PathTrace.color", requiredTiming=["PathTrace", "Path trace shading"])])
        experiment.save(cases, configuration)
        output = self.root / "prepared"
        output.mkdir()
        args = SimpleNamespace(cases=cases, case_id=["Opaque"], resolution=64, warmup_frames=2,
                               timing_frames=4, capture_graph_outputs=True)
        path, prepared, _ = experiment.workload_configuration(args, output)
        self.assertEqual(prepared["cases"][0]["width"], 64)
        self.assertEqual(prepared["cases"][0]["captureOutputs"], ["PathTrace.motionVectors"])
        self.assertEqual(path, output / "Cases.json")
        configuration["cases"][0]["cameraFrames"] = [dict(frame=3)]
        experiment.save(cases, configuration)
        with self.assertRaisesRegex(ValueError, "temporal"):
            experiment.workload_configuration(args, output)

    def test_sealed_evidence_recomputes_timings_and_rejects_drift(self):
        # Only one arm avoids a mock subprocess, while exercising the real
        # readback verifier, complete schedule and manifest integrity.
        arm, = experiment.arms(["O1"], ["pipeline"], "normal")
        path = self.root / "A/MaterialBaseline.json"
        run = dict(block=0, arm=arm, report=str(path), process=dict(exit_code=0, timed_out=False),
                   timings=experiment.gpu_timings(self.report, self.configuration),
                   correctness=experiment.compare_readbacks(path, path, self.configuration, 2e-5, 2e-5))
        summary = dict(status="passed", command="dispatch", pairs=1, conditions=[arm], native_binary_cache=False,
                       workload=self.configuration, tolerance=dict(absolute=2e-5, relative=2e-5), runs=[run])
        experiment.save(self.root / "summary.json", summary)
        experiment.seal(self.root)
        experiment.verify(self.root)
        (self.root / "A/motion.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "evidence"):
            experiment.verify(self.root)


if __name__ == "__main__":
    unittest.main()
