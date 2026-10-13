"""Guard exact workload reconstruction and qualification of CLI diagnostics."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import ProfileComputeShaderFrontend as frontend


CODE = struct.pack("<5I", 0x07230203, 0x00010600, 0, 4, 0)


def cache_bytes(code=CODE, path=b"source.slang", count=1, dependency_reserved=0):
    payload = struct.pack("<IIQ", len(path), dependency_reserved, 123) + path + code
    header = struct.pack("<8sIIQIIQQ", b"MTLSPV01", 2, 48, 123, count, 0, len(code), int(frontend.fnv(payload), 16))
    return header + payload


def report(directory, level="O1"):
    directory = Path(directory)
    source = directory / "program root/Other/Actual.slang"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("// exact shader source\n", encoding="utf-8")
    return dict(status="passed", failures=0, slang_build="test-sdk",
                configuration=dict(cache_directory=str(directory / "cache"), debug_mode="disabled", compile_only=True),
                environment=dict(METALLIC_WORKING_COLOR_SPACE=None),
                samples=[dict(status="passed", pipeline_type="compute", effective_descriptor_mode="mapped",
                              effective_slang_optimization=level, spirv_bytes=len(CODE), spirv_fnv1a64=frontend.fnv(CODE),
                              dependencies=[str(source)], request=dict(index=17, module="Other/Actual", entry="evaluateMain",
                              profile="spirv_1_6", search_path=str(source.parents[1]),
                              additional_search_paths=[str(directory / "SDK with spaces")],
                              capabilities=["spvRayQueryKHR", "spvGroupNonUniformBallot"],
                              defines=[["FEATURE_COUNT", "2"], ["TEXT", "a b;$()"]]))])


class ComputeShaderFrontendTests(unittest.TestCase):
    def test_cache_format_and_checksum_are_verified_before_extracting_code(self):
        self.assertEqual(frontend.decode_shader_cache(cache_bytes()), CODE)
        corrupt = bytearray(cache_bytes())
        corrupt[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            frontend.decode_shader_cache(corrupt)

    def test_unknown_headers_and_truncated_dependencies_are_rejected(self):
        for offset, value in ((0, 0), (8, 3), (12, 44), (24, 0), (24, 4097), (28, 1)):
            with self.subTest(offset=offset, value=value):
                corrupt = bytearray(cache_bytes())
                struct.pack_into("<I", corrupt, offset, value)
                with self.assertRaises(ValueError):
                    frontend.decode_shader_cache(corrupt)
        with self.assertRaises(ValueError):
            frontend.decode_shader_cache(cache_bytes()[:40])
        with self.assertRaisesRegex(ValueError, "dependency|payload"):
            frontend.decode_shader_cache(cache_bytes(count=2))
        with self.assertRaisesRegex(ValueError, "dependency"):
            frontend.decode_shader_cache(cache_bytes(dependency_reserved=1))
        with self.assertRaisesRegex(ValueError, "Null byte"):
            frontend.decode_shader_cache(cache_bytes(path=b"evil\0path"))

    def test_matching_cache_payload_requires_the_bench_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = report(directory)
            root = Path(metadata["configuration"]["cache_directory"]) / "spirv"
            root.mkdir(parents=True)
            (root / "junk.spv").write_bytes(b"not a shader")
            path = root / "matching.spv"
            path.write_bytes(cache_bytes())
            self.assertEqual(frontend.bench_bytes(metadata, metadata["samples"][0]), (path.resolve(), CODE))
            metadata["samples"][0]["spirv_fnv1a64"] = "0000000000000000"
            with self.assertRaisesRegex(ValueError, "No validated"):
                frontend.bench_bytes(metadata, metadata["samples"][0])

    def test_cli_preserves_the_actual_request_and_argument_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = report(directory)
            sample = metadata["samples"][0]
            command = frontend.diagnostic_command("slangc.exe", metadata, sample, "output.spv")
            for key, value in (("-module-name", "Other/Actual"), ("-entry", "evaluateMain"),
                               ("-profile", "spirv_1_6"), ("-o", "output.spv")):
                self.assertEqual(command[command.index(key) + 1], value)
            self.assertIn("-matrix-layout-row-major", command)
            self.assertIn("-O1", command)
            self.assertIn("-report-detailed-perf-benchmark", command)
            self.assertIn("TEXT=a b;$()", command) # No shell interpolation or splitting.
            self.assertIn("METALLIC_WORKING_SPACE_ACESCG=1", command)
            self.assertEqual([command[i + 1] for i, argument in enumerate(command) if argument == "-capability"],
                             sample["request"]["capabilities"])
            includes = [command[i + 1] for i, argument in enumerate(command) if argument == "-I"]
            self.assertEqual(includes[:2], [str(Path(sample["request"]["search_path"]).resolve()),
                                           str((Path(directory) / "SDK with spaces").resolve())])
            self.assertEqual(len(includes), len(set(includes)))

    def test_descriptor_debug_and_working_color_policies_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = report(directory)
            sample = metadata["samples"][0]
            metadata["environment"]["METALLIC_WORKING_COLOR_SPACE"] = "rec709"
            self.assertIn("METALLIC_WORKING_SPACE_ACESCG=0", frontend.diagnostic_command("compiler", metadata, sample, "out"))
            metadata["environment"]["METALLIC_WORKING_COLOR_SPACE"] = "invalid"
            with self.assertRaisesRegex(ValueError, "working color"):
                frontend.diagnostic_command("compiler", metadata, sample, "out")
            metadata["environment"]["METALLIC_WORKING_COLOR_SPACE"] = None
            sample["effective_descriptor_mode"] = "native"
            with self.assertRaisesRegex(ValueError, "mapped"):
                frontend.diagnostic_command("compiler", metadata, sample, "out")
            sample["effective_descriptor_mode"] = "mapped"
            metadata["configuration"]["debug_mode"] = "capture"
            with self.assertRaisesRegex(ValueError, "disabled"):
                frontend.diagnostic_command("compiler", metadata, sample, "out")

    def test_bytecode_mismatch_is_ineligible_even_when_size_is_equal(self):
        evidence = {}
        with self.assertRaisesRegex(ValueError, "must not be attributed"):
            frontend.compare_bytecode(CODE, CODE[:-1] + b"\1", evidence)
        self.assertFalse(evidence["matches_bench_byte_for_byte"])
        self.assertEqual(evidence["status"], "bytecode_mismatch")
        self.assertEqual(evidence["emitted_bytes"], len(CODE))
        evidence = {}
        frontend.compare_bytecode(CODE, CODE, evidence)
        self.assertTrue(evidence["matches_bench_byte_for_byte"])
        self.assertEqual(evidence["spirv_sha256"], hashlib.sha256(CODE).hexdigest())

    def prepare_run(self, directory):
        directory = Path(directory)
        sdk = directory / "sdk"
        (sdk / "bin").mkdir(parents=True)
        (sdk / "include").mkdir()
        (sdk / "bin" / ("slangc.exe" if os.name == "nt" else "slangc")).write_bytes(b"fake compiler")
        (sdk / "include/slang-tag-version.h").write_text('#define SLANG_TAG_VERSION "test-sdk"\n', encoding="utf-8")
        metadata = report(directory)
        (directory / "cache/spirv").mkdir(parents=True)
        (directory / "cache/spirv/matching.spv").write_bytes(cache_bytes())
        input_root = directory / "input"
        (input_root / "prefill-O1").mkdir(parents=True)
        (input_root / "prefill-O1/report.json").write_text(json.dumps(metadata), encoding="utf-8")
        output = directory / "output"
        args = ["--input-root", str(input_root), "--output-dir", str(output), "--slang-root", str(sdk), "--optimizations", "O1"]
        return args, output, metadata, input_root

    def test_process_success_with_different_bytecode_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, _, _ = self.prepare_run(directory)
            def run(command, **kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(CODE[:-1] + b"\1")
                return subprocess.CompletedProcess(command, 0, b"profile data", b"diagnostics")
            with patch.object(frontend.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(frontend.main(args), 1)
            evidence = json.loads((output / "summary.json").read_text())
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["runs"][0]["status"], "bytecode_mismatch")
            self.assertEqual((output / "block-00-O1/stdout.log").read_bytes(), b"profile data")
            self.assertFalse((output / "Manifest.json").exists())

    def test_completed_evidence_verifies_exact_bytes_and_detects_artifact_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, _, _ = self.prepare_run(directory)
            def run(command, **kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(CODE)
                return subprocess.CompletedProcess(command, 0, b"profile data", b"")
            with patch.object(frontend.subprocess, "run", side_effect=run), redirect_stdout(io.StringIO()):
                self.assertEqual(frontend.main(args), 0)
            self.assertEqual(frontend.verify(output), dict(status="verified", process_count=1))
            (output / "block-00-O1/shader.spv").write_bytes(CODE[:-1] + b"\1")
            with self.assertRaisesRegex(ValueError, "checksum"):
                frontend.verify(output)

    def test_sdk_build_tag_mismatch_never_starts_a_profile_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, metadata, input_root = self.prepare_run(directory)
            metadata["slang_build"] = "different compiler"
            (input_root / "prefill-O1/report.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch.object(frontend.subprocess, "run") as run, redirect_stderr(io.StringIO()):
                self.assertEqual(frontend.main(args), 1)
            run.assert_not_called()
            self.assertIn("SDK build tag", json.loads((output / "summary.json").read_text())["error"])

    def test_failed_prefill_never_starts_a_profile_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args, output, metadata, input_root = self.prepare_run(directory)
            metadata["samples"][0]["status"] = "failed"
            (input_root / "prefill-O1/report.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch.object(frontend.subprocess, "run") as run, redirect_stderr(io.StringIO()):
                self.assertEqual(frontend.main(args), 1)
            run.assert_not_called()

    def test_existing_performance_lock_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            lock = root / "build/compute-pipeline-experiment.lock"
            lock.write_text("native compilation in progress", encoding="utf-8")
            with patch.object(frontend, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "experiment lock"):
                    with frontend.experiment_lock():
                        self.fail("must not enter while another experiment runs")
            self.assertEqual(lock.read_text(), "native compilation in progress")

    def test_invalid_timeout_and_duplicate_conditions_are_rejected_before_process_creation(self):
        with patch.object(frontend.subprocess, "run") as run, redirect_stderr(io.StringIO()):
            for extra in (["--timeout", "nan"], ["--timeout", "inf"], ["--optimizations", "O1", "O1"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    frontend.main(["--input-root", "unused", "--output-dir", "unused", *extra])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
