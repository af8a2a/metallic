"""CLI/report regressions; compile-only by default, optional native GPU checks."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
from RunShaderPipelineDriver import native_process_crashed, suppress_windows_error_dialogs


class ShaderPipelineBenchTests(unittest.TestCase):
    executable = None
    gpu = False
    failures_directory = Path(__file__).resolve().parents[2] / ".cache/ShaderPipelineBenchFailures"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="metallic-shader-bench-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_bench(self, *arguments, expected=0):
        command = [str(self.executable), *map(str, arguments)]
        try:
            with suppress_windows_error_dialogs():
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            evidence = self.preserve_failure(command, None, error.stdout, error.stderr, timeout=120)
            self.stop_after_native_failure(f"Benchmark process timed out; evidence: {evidence}")
        if native_process_crashed(result.returncode):
            evidence = self.preserve_failure(command, result.returncode, result.stdout, result.stderr)
            self.stop_after_native_failure(f"Benchmark native crash 0x{result.returncode & 0xffffffff:08X}; evidence: {evidence}\n{result.stdout}{result.stderr}")
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def preserve_failure(self, command, exit_code, stdout, stderr, **metadata):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        destination = self.failures_directory / name
        shutil.copytree(self.root, destination)
        for filename, output in (("stdout.log", stdout), ("stderr.log", stderr)):
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            (destination / filename).write_text(output or "", encoding="utf-8")
        (destination / "failure.json").write_text(json.dumps(dict(command=command, exit_code=exit_code,
            test=self.id(), source_directory=str(self.root), **metadata), indent=2) + "\n", encoding="utf-8")
        return destination

    def stop_after_native_failure(self, message):
        # unittest subTest normally continues after assertions. A native crash
        # or timeout ends the whole suite so it cannot launch another process.
        if self._outcome is not None and self._outcome.result is not None:
            self._outcome.result.failfast = True
            self._outcome.result.stop()
        self.fail(message)

    def test_help_and_invalid_arguments(self):
        help_text = self.run_bench("--help").stdout
        for option in ("--pipeline-cache", "--mode", "--workers", "--cache-lock-scope", "--recipe-file"):
            self.assertIn(option, help_text)
        self.run_bench("--iterations", "0", expected=2)
        self.run_bench("--workers", "0", expected=2)
        self.run_bench("--mode", "unknown", expected=2)
        self.run_bench("--cache-lock-scope", "unknown", expected=2)
        self.run_bench("--pipeline-replicas", "0", expected=2)
        self.run_bench("--mode", "serial", "--workers", "2", expected=2)
        self.run_bench("--merge-shards", expected=2)
        self.run_bench("--pipeline-workers", "0", expected=2)
        self.run_bench("--pipeline-workers", "2", "--compile-only", expected=2)
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

    def test_gpu_concurrent_pipeline_batches(self):
        if not self.gpu:
            self.skipTest("enable --gpu for concurrent pipeline checks")
        for workers in (1, 3):
            destination = self.root / f"workers-{workers}"
            self.run_bench("--filter", "Features/PostProcess/AutoExposure.", "--iterations", "2",
                           "--pipeline-workers", workers, "--shader-cache", "reuse",
                           "--pipeline-cache", "reuse", "--output-dir", destination)
            report = json.loads((destination / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["pipelines_created"], 6)
            self.assertEqual(report["validation_errors"], 0)
            self.assertEqual(len(report["pipeline_batches"]), 2)
            for iteration, batch in enumerate(report["pipeline_batches"]):
                self.assertEqual(batch["actual_workers"], workers)
                self.assertEqual(batch["successful_pipelines"], 3)
                self.assertEqual(batch["metadata_hits"], 3 if iteration else 0)
                self.assertEqual(batch["metadata_misses"], 0 if iteration else 3)
                samples = [report["samples"][i] for i in batch["sample_indices"]]
                self.assertTrue(all(s["iteration"] == iteration for s in samples))
                self.assertTrue(all(s["pipeline_status"] == "created" for s in samples))
                self.assertAlmostEqual(batch["pipeline_wall_sum_ms"],
                                       sum(s["pipeline_wall_ms"] for s in samples))
                self.assertAlmostEqual(batch["successful_native_create_sum_ms"],
                                       sum(s["pipeline"]["native_create_ms"] for s in samples))
                self.assertAlmostEqual(batch["successful_cache_lock_wait_sum_ms"],
                                       sum(s["pipeline"]["cache_lock_wait_ms"] for s in samples))
                self.assertGreaterEqual(batch["pipeline_batch_wall_ms"],
                                        max(s["pipeline_wall_ms"] for s in samples))

    def test_gpu_explicit_modes_and_worker_matrix(self):
        if not self.gpu:
            self.skipTest("enable --gpu for native pipeline mode checks")
        # Three selected requests times three replicas gives nine jobs, enough
        # to launch all eight workers while preserving the workload per mode.
        cells = [("serial", 1, "metadata")]
        for workers in (1, 2, 4, 8):
            cells.extend((mode, workers, scope) for mode, scope in (
                ("parallel-shared", "metadata"), ("parallel-shared", "creation"),
                ("parallel-sharded", "metadata")))
        for mode, workers, scope in cells:
            with self.subTest(mode=mode, workers=workers, scope=scope):
                destination = self.root / f"{mode}-{workers}-{scope}"
                self.run_bench("--filter", "Features/PostProcess/AutoExposure.", "--iterations", "2",
                               "--mode", mode, "--workers", workers, "--pipeline-replicas", "3",
                               "--cache-lock-scope", scope, "--shader-cache", "reuse",
                               "--pipeline-cache", "reuse", "--output-dir", destination)
                report = json.loads((destination / "report.json").read_text())
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["pipelines_created"], 18)
                self.assertEqual(report["validation_errors"], 0)
                self.assertEqual(report["configuration"]["mode"], mode)
                self.assertEqual(report["configuration"]["workers"], workers)
                self.assertEqual(report["configuration"]["cache_lock_scope"], scope)
                self.assertEqual(len(report["pipeline_batches"]), 2)
                for batch in report["pipeline_batches"]:
                    self.assertEqual(batch["actual_workers"], workers)
                    self.assertEqual(batch["cache_count"], workers if mode == "parallel-sharded" else 1)
                    self.assertEqual(batch["pipeline_count"], 9)
                    self.assertEqual(batch["successful_pipelines"], 9)
                    self.assertEqual(batch["metadata_hits"] + batch["metadata_misses"], 9)
                    samples = [report["samples"][i] for i in batch["sample_indices"]]
                    self.assertEqual(len(samples), 9)
                    native = [sample["pipeline"]["native_create_ms"] for sample in samples]
                    lock = [sample["pipeline"]["cache_lock_wait_ms"] for sample in samples]
                    wall = [sample["pipeline_wall_ms"] for sample in samples]
                    for values in (native, lock, wall):
                        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))
                    self.assertAlmostEqual(batch["successful_native_create_sum_ms"], sum(native))
                    self.assertAlmostEqual(batch["successful_cache_lock_wait_sum_ms"], sum(lock))
                    self.assertAlmostEqual(batch["pipeline_wall_sum_ms"], sum(wall))
                    for field, values in (("pipeline_wall_p95_ms", wall), ("native_create_p95_ms", native),
                                          ("cache_lock_wait_p95_ms", lock)):
                        self.assertGreaterEqual(batch[field], min(values))
                        self.assertLessEqual(batch[field], max(values))
                    self.assertGreater(batch["throughput_pipelines_per_second"], 0)
                    self.assertAlmostEqual(batch["throughput_pipelines_per_second"],
                                           9000 / batch["pipeline_batch_wall_ms"])

    def test_gpu_shard_merge_reloads_metadata(self):
        if not self.gpu:
            self.skipTest("enable --gpu for shard merge/reload checks")
        destination = self.root / "shards"
        self.run_bench("--filter", "Features/PostProcess/AutoExposure.", "--iterations", "1",
                       "--mode", "parallel-sharded", "--workers", "4", "--pipeline-replicas", "3",
                       "--merge-shards", "--pipeline-cache", "empty", "--output-dir", destination)
        report = json.loads((destination / "report.json").read_text())
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["merged_cache"]["stored_pso_count"], 3)
        self.assertGreater(report["merged_cache"]["backend_data_bytes"], 0)
        self.assertGreaterEqual(report["pipeline_cache_merge_wall_ms"], 0)
        self.assertEqual(report["pipeline_cache_merge_wall_ms"],
                         report["pipeline_batches"][0]["pipeline_cache_merge_wall_ms"])
        cache = self.root / "merged-cache"
        cache.mkdir()
        shutil.copyfile(report["merged_cache"]["file"], cache / "pipeline.pso")
        reload = self.root / "merged-reload"
        self.run_bench("--filter", "Features/PostProcess/AutoExposure.", "--iterations", "1",
                       "--pipeline-cache", "reuse", "--cache-dir", cache, "--output-dir", reload)
        reloaded = json.loads((reload / "report.json").read_text())
        self.assertEqual(reloaded["status"], "passed")
        self.assertEqual(reloaded["pipelines_created"], 3)
        self.assertTrue(all(sample["pipeline"]["metadata_hit"] for sample in reloaded["samples"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    arguments, remaining = parser.parse_known_args()
    ShaderPipelineBenchTests.executable = arguments.executable.resolve(strict=True)
    ShaderPipelineBenchTests.gpu = arguments.gpu
    unittest.main(argv=[__file__, *remaining])
