"""Cold-cache, complete-catalog and evidence qualification boundaries; no compiler runs."""
from contextlib import redirect_stdout
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import MeasureShaderWarmup as measure

CATALOG = """[1/2] Features/First.firstMain
[2/2] Features/Second.secondMain TEXT=a b;$()
Shader warmup: 2 requests, 0 existing cache hits, 0 failures (list only)
"""
WARMUP = """Shader warmup workers: 2
  Processing 2/2: Features/Second.secondMain
  Processing 1/2: Features/First.firstMain [resource-stride=16, sampler-stride=16]
  Request 2 timing: total=0.4s, setup=0.0s, module=0.1s, codegen=0.3s
  Request 1 timing: total=0.2s, setup=0.0s, module=0.1s, codegen=0.1s
Shader warmup: 2 requests, 0 existing cache hits, 0 failures
Shader warmup elapsed: 0.6s; summed request time: 0.6s (module=0.2s, codegen=0.4s)
"""
BATCH_WARMUP = """Shader warmup workers: 2
[0/2 0%] Starting validated immutable source batch | cached: 0 | failed: 0 | elapsed: 0.0s
  Request 1 timing: total=0.2s, setup=0.0s, module=0.1s, codegen=0.1s
[1/2 50%] Compiled: Features/First.firstMain | cached: 0 | failed: 0 | elapsed: 0.6s
  Request 2 timing: total=0.4s, setup=0.0s, module=0.1s, codegen=0.3s
[2/2 100%] Compiled: Features/Second.secondMain | cached: 0 | failed: 0 | elapsed: 0.6s
Warmup source batch: 1 groups; static estimate schedule; snapshot=0.01s; boundary=0.01s; cache publication/readback=0.1s; live_fs_atomic=false
Warmup V3 publication phases: proof encode=0.02s; atomic write=0.02s; file read/decode=0.04s; full source validation=0.02s
Shader warmup: 2 requests, 0 existing cache hits, 0 failures
Shader warmup elapsed: 0.6s; summed request time: 0.6s (module=0.2s, codegen=0.4s)
"""


class ShaderWarmupEvidenceTests(unittest.TestCase):
    def test_existing_directories_and_prefilled_caches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            measure.fresh_cache(cache)
            measure.ensure_empty_cache(cache)
            with self.assertRaisesRegex(ValueError, "must be new"):
                measure.fresh_cache(cache)
            (cache / "old.spv").write_bytes(b"existing cache")
            with self.assertRaisesRegex(ValueError, "not empty"):
                measure.ensure_empty_cache(cache)

    def test_complete_catalog_preserves_macro_argument_text(self):
        rows = measure.parse_catalog(CATALOG)
        self.assertEqual(rows[1]["define_text"], "TEXT=a b;$()")
        for invalid in (CATALOG.replace("[2/2]", "[1/2]"), CATALOG.replace("[1/2]", "[1/3]"),
                        CATALOG.replace("2 requests", "3 requests"), CATALOG + CATALOG):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                measure.parse_catalog(invalid)

    def test_full_cold_warmup_requires_every_request_and_consistent_phase_sums(self):
        catalog = measure.parse_catalog(CATALOG)
        result = measure.parse_warmup(WARMUP, catalog, 8)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["existing_cache_hits"], 0)
        self.assertEqual([row["index"] for row in result["request_phases_seconds"]], [1, 2])
        self.assertAlmostEqual(result["summed_phases_seconds"]["codegen"], 0.4)
        for invalid in (WARMUP.replace("0 existing cache hits", "1 existing cache hits"),
                        WARMUP.replace("0 failures", "1 failures"),
                        WARMUP.replace("Request 2 timing", "Request 1 timing"),
                        WARMUP.replace("Features/Second.secondMain", "Features/Other.otherMain"),
                        WARMUP.replace("codegen=0.4s)", "codegen=9.4s)"),
                        WARMUP.replace("total=0.4s", "total=nans")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                measure.parse_warmup(invalid, catalog, 8)

    def test_goal_uses_every_complete_process_and_strict_threshold(self):
        runs = [dict(process=dict(process_wall_seconds=value)) for value in (8.0, 9.0, 10.0)]
        result = measure.aggregate(runs, 10.0)
        self.assertFalse(result["goal_reached"])
        self.assertEqual(result["median_seconds"], 9.0)
        runs[-1]["process"]["process_wall_seconds"] = 9.9
        self.assertTrue(measure.aggregate(runs, 10.0)["goal_reached"])

    def test_batch_log_requires_every_request_and_actual_publication(self):
        catalog = measure.parse_catalog(CATALOG)
        result = measure.parse_warmup(BATCH_WARMUP, catalog, 8)
        self.assertEqual(result["batch"]["session_groups"], 1)
        self.assertAlmostEqual(result["batch"]["publication_phases_seconds"]["file_read_decode"], 0.04)
        for invalid in (BATCH_WARMUP.replace("Features/Second.secondMain", "Features/Other.otherMain"),
                        BATCH_WARMUP.replace("[2/2 100%] Compiled:", "[1/2 50%] Compiled:"),
                        BATCH_WARMUP.replace("1 groups", "0 groups"),
                        BATCH_WARMUP.replace("file read/decode=0.04s", "file read/decode=9.04s"),
                        BATCH_WARMUP.replace("full source validation=0.02s", "full source validation=nans"),
                        BATCH_WARMUP + "Shader warmup used fresh sessions without persistent cache reuse\n"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                measure.parse_warmup(invalid, catalog, 8)

    def test_input_content_and_source_inventory_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "compiler.exe"
            executable.write_bytes(b"compiler")
            (root / "slang.dll").write_bytes(b"sdk")
            (root / "Shaders").mkdir()
            source = root / "Shaders/Probe.slang"
            source.write_text("module Probe;", encoding="utf-8")
            identity = measure.input_identity(executable, root)
            measure.check_identity(identity, executable, root)
            source.write_text("module Other;", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed evidence"):
                measure.check_identity(identity, executable, root)
            source.write_text("module Probe;", encoding="utf-8")
            (root / "Shaders/New.slang").write_text("module New;", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed evidence"):
                measure.check_identity(identity, executable, root)

    def test_output_artifacts_under_a_source_root_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "compiler.exe"
            executable.write_bytes(b"compiler")
            output = root / "Shaders/evidence"
            output.mkdir(parents=True)
            artifact = output / "summary.json"
            artifact.write_text("initial", encoding="utf-8")
            identity = measure.input_identity(executable, root, exclude=output)
            artifact.write_text("updated", encoding="utf-8")
            (output / "new.json").write_text("new artifact", encoding="utf-8")
            measure.check_identity(identity, executable, root, exclude=output)
            self.assertNotIn(str(artifact.resolve()), identity["files"])

    def test_environment_evidence_only_contains_known_compiler_inputs(self):
        environment = {"METALLIC_SLANG_OPTIMIZATION": "O1", "METALLIC_WORKING_COLOR_SPACE": "rec709",
                       "PATH": "compiler path", "METALLIC_API_TOKEN": "secret", "SLANG_API_KEY": "secret",
                       "VK_CUSTOM_TOKEN": "secret"}
        evidence = measure.compiler_environment(environment)
        self.assertEqual(evidence["METALLIC_SLANG_OPTIMIZATION"], "O1")
        self.assertEqual(evidence["METALLIC_WORKING_COLOR_SPACE"], "rec709")
        self.assertNotIn("secret", evidence.values())

    def measurement_fixture(self, root, wall):
        executable = root / "compiler.exe"
        executable.write_bytes(b"fixture compiler")
        output = root / "output"
        calls = []
        environment = {"METALLIC_SLANG_OPTIMIZATION": "O1", "METALLIC_WORKING_COLOR_SPACE": "rec709"}

        def run_process(command, inherited, destination, timeout):
            calls.append(command)
            self.assertEqual(inherited, environment)
            self.assertNotIn("--filter", command)
            self.assertNotIn("--slang-optimization", command)
            destination.mkdir()
            listing = "--list" in command
            if not listing:
                cache = Path(command[command.index("--cache-dir") + 1])
                measure.ensure_empty_cache(cache)
                (cache / "new.spv").write_bytes(b"fixture compilation")
            (destination / "stdout.log").write_text(CATALOG if listing else WARMUP, encoding="utf-8")
            (destination / "stderr.log").write_text("full diagnostics", encoding="utf-8")
            return dict(command=command, exit_code=0, timed_out=False, process_wall_seconds=wall,
                        stdout=str(destination / "stdout.log"), stderr=str(destination / "stderr.log"),
                        memory=dict(provider="unavailable", peak_working_set_bytes=None, peak_private_usage_bytes=None))

        args = ["--executable", str(executable), "--output-dir", str(output), "--jobs", "8", "--repeats", "2"]
        identity = measure.input_identity(executable, root)
        return args, output, calls, environment, run_process, identity

    def test_goal_miss_still_saves_passed_measurement_and_returns_one(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, calls, environment, run_process, identity = self.measurement_fixture(Path(directory), 12.0)
            stdout = io.StringIO()
            with patch.object(measure.os, "environ", environment), \
                 patch.object(measure, "input_identity", return_value=identity), \
                 patch.object(measure, "check_identity"), \
                 patch.object(measure, "run_process", side_effect=run_process), redirect_stdout(stdout):
                self.assertEqual(measure.main(args), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "passed")
            self.assertFalse(summary["goal_reached"])
            self.assertEqual(summary["aggregate"]["count"], 2)
            self.assertEqual(len(calls), 3)  # Listing plus two serial independent processes.
            self.assertNotEqual(calls[1][2], calls[2][2])
            self.assertEqual(len(stdout.getvalue().splitlines()), 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "passed")

    def test_goal_success_and_failed_process_keep_distinct_statuses(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args, output, _, environment, run_process, identity = self.measurement_fixture(Path(directory), 9.0)

                def failing_process(*args):
                    result = run_process(*args)
                    if failure and "--list" not in args[0]:
                        result["exit_code"] = 1
                    return result

                with patch.object(measure.os, "environ", environment), \
                     patch.object(measure, "input_identity", return_value=identity), \
                     patch.object(measure, "check_identity"), \
                     patch.object(measure, "run_process", side_effect=failing_process), redirect_stdout(io.StringIO()):
                    self.assertEqual(measure.main(args), int(failure))
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], "failed" if failure else "passed")
                self.assertEqual(summary["goal_reached"], not failure)
                if failure:
                    self.assertEqual(len(summary["runs"]), 1)
                    self.assertTrue((output / "run-00/stderr.log").exists())

    def test_existing_output_and_non_o1_environment_never_start_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, _, environment, _, _ = self.measurement_fixture(Path(directory), 9.0)
            output.mkdir()
            with patch.object(measure.os, "environ", environment), patch.object(measure, "run_process") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(args), 1)
                run.assert_not_called()
            self.assertEqual(list(output.iterdir()), [])
            output.rmdir()
            environment["METALLIC_SLANG_OPTIMIZATION"] = "O0"
            with patch.object(measure.os, "environ", environment), patch.object(measure, "run_process") as run, redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(args), 1)
                run.assert_not_called()

    def test_timeout_must_be_finite_and_positive(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                measure.positive_finite(value)


if __name__ == "__main__":
    unittest.main()
