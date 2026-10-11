"""CLI/report regression checks; uses mapped compile-only requests, no GPU."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class ShaderPipelineBenchTests(unittest.TestCase):
    executable = None
    gpu = False

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metallic-shader-bench-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_bench(self, *arguments, expected=0):
        result = subprocess.run(
            [str(self.executable), *map(str, arguments)],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_help_and_invalid_arguments(self):
        self.assertIn("--pipeline-cache", self.run_bench("--help").stdout)
        self.run_bench("--iterations", "0", expected=2)
        self.run_bench("--unknown-option", expected=2)
        self.run_bench("--output-dir", self.root / "unselected", expected=2)
        self.assertFalse((self.root / "unselected").exists())

    def test_no_match_does_not_create_output(self):
        destination = self.root / "no-match"
        self.run_bench("--filter", "__nonexistent_metallic_request__",
                       "--output-dir", destination, expected=2)
        self.assertFalse(destination.exists())

    def test_listing_and_exact_selection(self):
        requests = [json.loads(line) for line in self.run_bench(
            "--list", "--filter", "Features/Debug/GPUProbe.probe").stdout.splitlines()
            if line.startswith("{")]
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request["entry"], "probe")
        for field in ("defines", "capabilities", "profile", "search_path"):
            self.assertIn(field, request)
        exact = self.run_bench("--list", "--request-index", request["index"])
        self.assertIn(json.dumps(request["module"]), exact.stdout)

    def compile_report(self, destination, mode, cache=None):
        args = ["--filter", "Features/Debug/GPUProbe.probe", "--compile-only",
                "--descriptor-mode", "mapped", "--iterations", "2",
                "--shader-cache", mode, "--output-dir", destination]
        if cache is not None:
            args.extend(["--cache-dir", cache])
        self.run_bench(*args)
        report = json.loads((destination / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["failures"], 0)
        self.assertNotIn("device", report)
        self.assertEqual(len(report["samples"]), 2)
        for sample in report["samples"]:
            self.assertEqual(sample["status"], "passed")
            self.assertEqual(sample["pipeline_status"], "compile_only")
            self.assertEqual(sample["execution_model"], 5)
            self.assertGreater(sample["spirv_bytes"], 20)
            self.assertGreaterEqual(sample["compile_wall_ms"], 0)
            self.assertNotIn("pipeline_wall_ms", sample)
            self.assertNotIn("pipeline", sample)
        self.assertEqual(report["samples"][0]["spirv_fnv1a64"],
                         report["samples"][1]["spirv_fnv1a64"])
        return report

    def test_disabled_cache_and_report_preservation(self):
        destination = self.root / "disabled"
        report = self.compile_report(destination, "disabled")
        self.assertEqual([s["shader_cache_hit"] for s in report["samples"]], [False, False])
        self.assertFalse(any((destination / "cache").rglob("*.spv")))
        previous = (destination / "report.json").read_bytes()
        self.run_bench("--filter", "GPUProbe", "--compile-only",
                       "--output-dir", destination, expected=2)
        self.assertEqual((destination / "report.json").read_bytes(), previous)

    def test_disk_cache_reuse_within_and_across_runs(self):
        cache = self.root / "shared-cache"
        initial = self.compile_report(self.root / "initial", "reuse", cache)
        self.assertEqual([s["shader_cache_hit"] for s in initial["samples"]], [False, True])
        subsequent = self.compile_report(self.root / "subsequent", "reuse", cache)
        self.assertEqual([s["shader_cache_hit"] for s in subsequent["samples"]], [True, True])
        self.assertEqual(initial["samples"][0]["spirv_fnv1a64"],
                         subsequent["samples"][0]["spirv_fnv1a64"])

    def test_gpu_pipeline_cache_and_timing(self):
        if not self.gpu:
            self.skipTest("enable --gpu for native pipeline checks")
        cache = self.root / "gpu-cache"
        for name, shader_mode, pipeline_mode, expected_shader, expected_metadata in (
            ("first", "reuse", "reuse", [False, True], [False, True]),
            ("reload", "reuse", "reuse", [True, True], [True, True]),
            ("empty", "disabled", "empty", [False, False], [False, False]),
        ):
            destination = self.root / name
            self.run_bench("--filter", "GPUProbe.probe", "--iterations", "2",
                           "--output-dir", destination, "--cache-dir", cache,
                           "--shader-cache", shader_mode, "--pipeline-cache", pipeline_mode)
            report = json.loads((destination / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["pipelines_created"], 2)
            self.assertEqual(report["pipelines_skipped"], 0)
            self.assertEqual(report["validation_errors"], 0)
            self.assertIn("name", report["device"])
            samples = report["samples"]
            self.assertEqual([s["shader_cache_hit"] for s in samples], expected_shader)
            self.assertEqual([s["pipeline"]["metadata_hit"] for s in samples], expected_metadata)
            for sample in samples:
                self.assertEqual(sample["pipeline_status"], "created")
                for field in ("compile_wall_ms", "shader_module_wall_ms", "pipeline_wall_ms"):
                    self.assertGreaterEqual(sample[field], 0)
                pipeline = sample["pipeline"]
                self.assertGreaterEqual(pipeline["native_create_ms"], 0)
                self.assertGreaterEqual(sample["pipeline_wall_ms"], pipeline["native_create_ms"])
                if pipeline["feedback_valid"]:
                    self.assertIsInstance(pipeline["native_cache_hit"], bool)
                    self.assertGreaterEqual(pipeline["driver_ms"], 0)
                else:
                    self.assertIsNone(pipeline["native_cache_hit"])
                    self.assertIsNone(pipeline["driver_ms"])
            self.assertEqual(samples[0]["pipeline"]["pso_hash"], samples[1]["pipeline"]["pso_hash"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    arguments, remaining = parser.parse_known_args()
    ShaderPipelineBenchTests.executable = arguments.executable.resolve(strict=True)
    ShaderPipelineBenchTests.gpu = arguments.gpu
    unittest.main(argv=[__file__, *remaining])
