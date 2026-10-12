"""PSO experiment integrity checks; synthetic data are never performance evidence."""

import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import PlotShaderPipelineBench as plot
import RunShaderPipelineDriver as driver
import RunShaderPipelineMatrix as matrix
import TestShaderPipelineBench as bench


def fixture():
    return dict(status="passed", validation_errors=0, build_configuration="Release", slang_build="test", shader_target=dict(id="vulkan", revision=1),
        configuration=dict(debug_mode="disabled", validation=False, pipeline_cache="empty", compile_only=False),
        environment={driver.CACHE_ENVIRONMENT: "enabled"}, device=dict(name="test GPU", driver_version=1),
        samples=[dict(status="passed", pipeline_status="created", request=dict(index=2, module="Test", entry="compute"),
            execution_model=5, spirv_fnv1a64="1234", spirv_bytes=512, effective_descriptor_mode="mapped",
            resource_heap_stride=0, sampler_heap_stride=0, shader_cache_hit=True,
            pipeline=dict(native_create_ms=4, driver_ms=3, cache_lock_wait_ms=1, native_cache_hit=False))])


class DriverExperimentTests(unittest.TestCase):
    def test_native_exception_detection_and_failfast_preserve_evidence(self):
        for code in (0xc0000005, 0xc0000409, 0xc000001d, -11):
            self.assertTrue(driver.native_process_crashed(code))
        for code in (0, 1, 2, 3):
            self.assertFalse(driver.native_process_crashed(code))
        with tempfile.TemporaryDirectory() as temporary:
            class FatalTests(bench.ShaderPipelineBenchTests):
                failures_directory = Path(temporary) / "preserved"
                executable = Path("mock-benchmark")

                def test_first(self):
                    (self.root / "report.json").write_text('{"status":"creating_device"}')
                    for worker in (1, 2, 4):
                        with self.subTest(worker=worker):
                            self.run_bench("--workers", worker)

                def test_second(self):
                    raise AssertionError("must never reach the next test after a native crash")

            # No executable is launched: this verifies unittest's actual
            # subTest/stop behavior and durable evidence after tearDown.
            with patch.object(bench.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0xc0000005, stdout="native stdout", stderr="native stderr")) as launched:
                result = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite([
                    FatalTests("test_first"), FatalTests("test_second")]))
            self.assertEqual(launched.call_count, 1)
            self.assertEqual(result.testsRun, 1)
            self.assertTrue(result.shouldStop)
            evidence, = list((Path(temporary) / "preserved").iterdir())
            self.assertEqual(json.loads((evidence / "report.json").read_text())["status"], "creating_device")
            self.assertEqual(json.loads((evidence / "failure.json").read_text())["exit_code"], 0xc0000005)
            self.assertEqual((evidence / "stdout.log").read_text(), "native stdout")
    def test_rejects_shader_device_control_and_cache_drift(self):
        report = fixture()
        identity = driver.shader_identity(report, report["samples"][0])
        driver.verify_measurement(report, identity, report["device"])
        mutations = [
            lambda r: r["samples"][0].update(spirv_fnv1a64="changed"),
            lambda r: r["samples"][0].update(shader_cache_hit=False),
            lambda r: r["samples"][0].update(pipeline_status="skipped_non_compute"),
            lambda r: r["configuration"].update(pipeline_cache="reuse"),
            lambda r: r["configuration"].update(debug_mode="debug"),
            lambda r: r.update(build_configuration="Debug"),
            lambda r: r["device"].update(driver_version=2),
            lambda r: r["environment"].update({driver.CACHE_ENVIRONMENT: "disabled"}),
            lambda r: r.update(validation_errors=1),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    driver.verify_measurement(changed, identity, report["device"])

    def test_unknown_driver_feedback_is_retained(self):
        report = fixture()
        report["samples"][0]["pipeline"].update(driver_ms=None, native_cache_hit=None)
        identity = driver.shader_identity(report, report["samples"][0])
        sample = driver.verify_measurement(report, identity)
        self.assertIsNone(sample["pipeline"]["driver_ms"])
        for field, value in (("native_create_ms", float("nan")), ("driver_ms", -1)):
            changed = copy.deepcopy(report)
            changed["samples"][0]["pipeline"][field] = value
            with self.assertRaises(ValueError):
                driver.verify_measurement(changed, identity)

    def test_graphics_recipe_identity_and_hash_must_match(self):
        report = fixture()
        sample = report["samples"][0]
        sample.update(shader_source="recipe_spirv", pipeline_type="graphics", shader_stages=[dict(
            stage="vertex", module="Graphics", entry="main", source_entry="vertex", spirv_bytes=512, spirv_fnv1a64="1234")])
        sample.pop("shader_cache_hit")
        sample["pipeline"]["recipe_hash_matches"] = True
        identity = driver.shader_identity(report, sample)
        driver.verify_measurement(report, identity)
        sample["shader_stages"][0]["spirv_fnv1a64"] = "changed"
        with self.assertRaises(ValueError):
            driver.verify_measurement(report, copy.deepcopy(identity))
        sample["shader_stages"][0]["spirv_fnv1a64"] = "1234"
        sample["pipeline"]["recipe_hash_matches"] = False
        with self.assertRaises(ValueError):
            driver.verify_measurement(report, identity)

    def test_independent_process_environment_and_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "run"
            code = "import json,os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({'value':os.getenv('METALLIC_TEST_CACHE')}))"
            metadata, report = driver.run_process([sys.executable, "-c", code, str(destination / "report.json")],
                dict(os.environ, METALLIC_TEST_CACHE="disabled"), destination, 10)
            self.assertEqual(report["value"], "disabled")
            self.assertGreater(metadata["pid"], 0)
            self.assertGreaterEqual(metadata["process_wall_ms"], 0)
            self.assertTrue((destination / "process.json").exists())
            failure = Path(temporary) / "failure"
            with self.assertRaises(RuntimeError):
                driver.run_process([sys.executable, "-c", "raise SystemExit(3)"], os.environ.copy(), failure, 10)
            self.assertEqual(json.loads((failure / "process.json").read_text())["exit_code"], 3)

    def test_matrix_and_replicas_have_identical_shader_identity(self):
        cells = list(matrix.matrix())
        self.assertEqual(len(cells), 13)
        for workers in (1, 2, 4, 8):
            self.assertIn(("parallel-shared", workers, "creation"), cells)
            self.assertIn(("parallel-shared", workers, "metadata"), cells)
            self.assertIn(("parallel-sharded", workers, "metadata"), cells)
        report = fixture()
        expected = matrix.identities(report)
        report["samples"].append(copy.deepcopy(report["samples"][0]))
        self.assertEqual(matrix.identities(report), expected)
        report["samples"][1]["spirv_fnv1a64"] = "drift"
        with self.assertRaises(ValueError):
            matrix.identities(report)

    def test_plot_preserves_feedback_and_filters_failed_samples(self):
        report = fixture()
        report["samples"].append(dict(status="failed"))
        row, = list(plot.sample_rows(report, Path("fixture.json")))
        self.assertEqual(row["shader_module"], "Test")
        self.assertEqual(row["entry_point"], "compute")
        self.assertEqual(row["application_pipeline_cache_hit"], "miss")
        self.assertEqual(row["pipeline_type"], "compute")
        self.assertEqual(row["build_configuration"], "Release")
        self.assertEqual(row["driver_cache_control"], "enabled")
        self.assertEqual(row["actual_workers"], 1)
        report["pipeline_batches"] = [dict(actual_workers=4)]
        report["samples"][0].update(pipeline_batch=0, pipeline_worker=2, iteration=1, pipeline_replica=3)
        batch_row = next(plot.sample_rows(report, Path("fixture.json")))
        self.assertEqual((batch_row["actual_workers"], batch_row["pipeline_worker"],
                          batch_row["iteration"], batch_row["pipeline_replica"]), (4, 2, 1, 3))
        report["samples"][0]["pipeline"]["native_cache_hit"] = None
        self.assertEqual(next(plot.sample_rows(report, Path("fixture.json")))["application_pipeline_cache_hit"], "unknown")
        report["samples"][0]["pipeline"]["native_create_ms"] = float("inf")
        with self.assertRaises(ValueError):
            list(plot.sample_rows(report, Path("fixture.json")))

    def test_svg_fallback_escapes_shader_labels(self):
        report = fixture()
        report["samples"][0]["request"]["module"] = "Shader<&>"
        rows = list(plot.sample_rows(report, Path("fixture.json")))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot.svg_distribution(rows, output, 20)
            files = list(output.glob("*.svg"))
            self.assertEqual(len(files), 3)
            for path in files:
                ET.parse(path)

    def test_log_svg_preserves_zeros_and_distinguishes_pipeline_types(self):
        row = next(plot.sample_rows(fixture(), Path("fixture.json")))
        rows = [dict(row, native_create_ms=value) for value in (0, .001, .01)]
        rows.append(dict(row, pipeline_type="graphics", shader_module="Graphics", native_create_ms=10000))
        original = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot.svg_distribution(rows, output, 20, log_scale=True)
            namespace = {"svg": "http://www.w3.org/2000/svg"}
            scatter = ET.parse(output / "PipelineSizeLatency.svg")
            circles = scatter.findall(".//svg:circle", namespace)
            points = [circle for circle in circles if circle.find("svg:title", namespace) is not None]
            self.assertEqual(len(points), 3)
            ys = [float(point.get("cy")) for point in points]
            # Each decade receives the same spacing: 0 displays at 1e-6,
            # three decades below 1e-3, then one decade to 1e-2.
            self.assertAlmostEqual(ys[0] - ys[1], 3 * (ys[1] - ys[2]), places=2)
            graphics = scatter.findall(".//svg:polygon", namespace)
            self.assertEqual(len(graphics), 2)  # One data point and its legend.
            self.assertTrue(all(shape.get("data-pipeline-type") == "graphics" for shape in graphics))
            for path in output.glob("*.svg"):
                ET.parse(path)
                self.assertIn("Base-10 log", path.read_text())
                self.assertIn("1e-6 ms", path.read_text())
        self.assertEqual(rows, original)


if __name__ == "__main__":
    unittest.main()
