"""Qualify narrow material-read compile evidence without Slang or a GPU."""
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import ValidatePathTraceMaterialReads as harness


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code(generator=1):
    return struct.pack("<9I", 0x07230203, 0x00010600, generator, 5, 0, (4 << 16) | 59, 1, 2, 7)


class MaterialReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in harness.overlay.SOURCE_FILES:
            source = ROOT / "Shaders" / name
            target = self.root / "Shaders" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ("Source/Runtime/Render/Core/NamedResourceParameters.h", "Source/Runtime/Render/Core/ColorSpaceMatrices.h",
                "Source/Runtime/Render/Environment/AtmosphereKernelParameters.h", "Source/Runtime/Render/Material/MaterialValueProgram.cpp",
                "Source/Runtime/Render/Material/SceneMaterialData.h", "Tools/CheckShaderModules.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.sdk = self.root / "SDK"
        (self.sdk / "bin").mkdir(parents=True)
        (self.sdk / "include").mkdir()
        (self.sdk / "bin/slangc.exe").write_bytes(b"Synthetic compiler, never executed")
        (self.sdk / "include/slang.h").write_bytes(b"Fixture Slang API")
        (self.sdk / "include/slang-tag-version.h").write_text('#define SLANG_TAG_VERSION "fixture"\n', encoding="utf-8")
        self.validator = self.root / "spirv-val.exe"
        self.validator.write_bytes(b"Synthetic validator, never executed")
        self.cache = self.root / "reference-cache.spv"
        self.cache.write_bytes(b"synthetic cache; decoder is separately covered")
        sample = dict(status="passed", pipeline_status="compile_only", shader_cache_hit=False,
            compilation_phases=dict(compiler_executed=True), effective_slang_optimization="O1",
            effective_descriptor_mode="mapped", effective_spirv_postprocess="none", dependencies=[],
            spirv_bytes=len(code()), spirv_fnv1a64=harness.profile.fnv(code()), request=dict(index=146,
                module="Features/PathTracing/OpenPBRRayQueryPathTraceGuides", entry="openPbrRayQueryPathTraceGuidesMain",
                profile="spirv_1_6", search_path=str(self.root / "Shaders"), additional_search_paths=[],
                capabilities=["spvRayQueryKHR"], defines=[["METALLIC_CUSTOM_MATERIALS", "0"]]))
        self.report = dict(status="passed", build_configuration="Release", slang_build="fixture", environment={},
            configuration=dict(compile_only=True, shader_cache="reuse", slang_optimization="O1", descriptor_mode="mapped",
                spirv_postprocess="none", debug_mode="disabled", validation=False), samples=[sample])
        self.report_path = self.root / "report.json"
        save(self.report_path, self.report)
        self.prepare_args = SimpleNamespace(report=self.report_path, request_index=146, slang_root=self.sdk, output_dir=self.root / "prepared")
        self.counter, self.fail_at, self.variant_drift, self.baseline_mismatch = 0, None, False, False

    def fake_invoke(self, command, directory, name, timeout):
        index = self.counter
        self.counter += 1
        start = datetime.now(timezone.utc).isoformat()
        if "-o" in command:
            path = Path(command[command.index("-o") + 1])
            narrow = "material-narrow-general" in command[1]
            member = "material-member-general" in command[1]
            generator = 3 if member else (2 if narrow else 1)
            if self.baseline_mismatch and not narrow:
                generator = 99
            if self.variant_drift and narrow and index >= 6:
                generator = 3
            if name == "cleanup":
                path.write_bytes(Path(command[command.index("-o") - 1]).read_bytes())
            else:
                path.write_bytes(code(generator))
        (directory / (name + ".stdout.log")).write_text("", encoding="utf-8")
        (directory / (name + ".stderr.log")).write_text("[*] fixturePhase 1 1.25ms\n", encoding="utf-8")
        stage = dict(command=command, pid=900 + index, started_utc=start,
            ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=1.0 + index / 100,
            execution_id=uuid.uuid4().hex, exit_code=1 if index == self.fail_at else 0,
            timed_out=index == self.fail_at, native_crash=False,
            stdout=name + ".stdout.log", stderr=name + ".stderr.log", status="failed" if index == self.fail_at else "passed")
        save(directory / (name + ".process.json"), stage)
        return stage

    def prepared(self):
        with patch.object(harness, "ROOT", self.root), patch.object(harness.profile, "ROOT", self.root), \
             patch.object(harness.legacy, "ROOT", self.root), patch.object(harness.profile, "bench_bytes", return_value=(self.cache, code())), \
             patch.object(harness, "invoke", side_effect=self.fake_invoke):
            harness.prepare(self.prepare_args)
        return self.prepare_args.output_dir

    def compiled(self):
        preparation = self.prepared()
        self.counter = 0
        args = SimpleNamespace(preparation=preparation, output_dir=self.root / "compiled", spirv_val=self.validator, timeout=1)
        with patch.object(harness, "invoke", side_effect=self.fake_invoke), \
             patch.object(harness.profile, "experiment_lock", side_effect=nullcontext):
            result = harness.run(args)
        return args.output_dir, result

    def test_schedule_is_three_independent_ab_ba_ab_source_pairs(self):
        self.assertEqual(harness.schedule(), [(0, "original-general"), (0, "material-narrow-general"),
            (1, "material-narrow-general"), (1, "original-general"), (2, "original-general"), (2, "material-narrow-general")])

    def test_selected_member_pair_keeps_default_raw_pair_and_rejects_unrelated_arms(self):
        self.assertEqual(harness.selected_arms(), harness.ARMS)
        self.assertEqual(harness.selected_arms("material-member-general"), ("original-general", "material-member-general"))
        self.assertEqual(harness.schedule("material-member-general"), [(0, "original-general"), (0, "material-member-general"),
            (1, "material-member-general"), (1, "original-general"), (2, "original-general"), (2, "material-member-general")])
        for arm in ("original-general", "backend-link-no-fiber", "material-member", None, True, ["material-member-general"]):
            with self.subTest(arm=arm), self.assertRaises((ValueError, TypeError)):
                harness.selected_arms(arm)

    def test_reference_rejects_other_request_policy_permutation_cache_and_native_work(self):
        mutations = (lambda report: report["samples"][0]["request"].update(index=149),
            lambda report: report["configuration"].update(slang_optimization="O2"),
            lambda report: report["samples"][0].update(shader_cache_hit=True),
            lambda report: report["samples"][0].update(pipeline_status="created"),
            lambda report: report["samples"][0]["request"].update(defines=[["METALLIC_CUSTOM_MATERIALS", "1"]]),
            lambda report: report["samples"][0]["request"].update(defines=[["METALLIC_CUSTOM_MATERIALS", "0"], ["SHARC_QUERY", "1"]]),
            lambda report: report.update(build_configuration="Debug"),
            lambda report: report["environment"].update(METALLIC_WORKING_COLOR_SPACE="rec709"))
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                harness.reference_sample(report)

    def test_prepare_freezes_exact_original_and_only_planned_general_material_reads(self):
        output = self.prepared()
        preparation = harness.verify_preparation(output)
        self.assertEqual(preparation["arms"], list(harness.ARMS))
        self.assertIn("Source/Runtime/Render/Material/SceneMaterialData.h", preparation["source_files"])
        original = output / "overlays/original-general"
        narrow = output / "overlays/material-narrow-general"
        new_read = Path("Shaders") / harness.overlay.READ_SOURCE
        self.assertFalse((original / new_read).exists())
        self.assertTrue((narrow / new_read).is_file())
        self.assertTrue(all(plan["overlay_metadata"]["feature"] == "general" for plan in preparation["plans"]))
        self.assertTrue(all("-O1" in plan["direct_command"] for plan in preparation["plans"]))

    def test_resealed_arm_source_or_metadata_relabelling_still_rejects(self):
        output = self.prepared()
        path = output / "Preparation.json"
        good = harness.experiment.read(path)
        for mutation in (lambda data: data["plans"][0].update(label="material-narrow-general"),
                lambda data: data["plans"][0].update(overlay_metadata=copy.deepcopy(data["plans"][1]["overlay_metadata"])),
                lambda data: data["plans"][1]["direct_command"].append("-O0")):
            changed = copy.deepcopy(good)
            mutation(changed)
            save(path, changed)
            harness.profile.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                harness.verify_preparation(output)

    def test_raw_statistics_include_actual_variable_storage_classes_and_cfg(self):
        statistics = harness.counts(code())
        self.assertEqual(statistics["variables"], 1)
        self.assertEqual(statistics["variable_storage_classes"], {"7": 1})
        self.assertEqual(statistics["instructions"], 1)
        self.assertEqual(statistics["basic_blocks"], 0)
        with self.assertRaises(ValueError):
            harness.counts(code()[:-1])

    def test_mocked_run_requires_six_source_samples_with_exact_baseline_and_final_validation(self):
        output, summary = self.compiled()
        self.assertEqual(summary["status"], "passed")
        self.assertEqual([(row["block"], row["arm"]) for row in summary["runs"]], harness.schedule())
        self.assertEqual(len({row[name]["execution_id"] for row in summary["runs"] for name in ("compile", "validation")}), 12)
        self.assertEqual(summary["timing_summary"]["denominator_arm"], "original-general")
        self.assertEqual(harness.verify_records(output), summary)
        for row in summary["runs"]:
            self.assertEqual(row["validation"]["command"][1:4], ["--target-env", "vulkan1.4", "--scalar-block-layout"])
            self.assertEqual(row["sdk_phases"]["rows"], [dict(name="fixturePhase", invocations=1, milliseconds=1.25, stream="stderr")])

    def test_member_run_binds_only_selected_source_candidate_and_original_denominator(self):
        self.prepare_args.candidate_arm = "material-member-general"
        output, summary = self.compiled()
        self.assertEqual(summary["arms"], ["original-general", "material-member-general"])
        self.assertEqual(summary["candidate_arm"], "material-member-general")
        self.assertEqual(summary["timing_summary"]["candidate_arm"], "material-member-general")
        self.assertEqual(summary["timing_summary"]["denominator_arm"], "original-general")
        preparation = harness.experiment.read(output / "Preparation/Preparation.json")
        self.assertEqual(preparation["arms"], summary["arms"])
        self.assertFalse((output / "Preparation/overlays/material-narrow-general").exists())
        for row in summary["runs"]:
            expected = code(3 if row["arm"] == "material-member-general" else 1)
            self.assertEqual((output / row["directory"] / "Guides.spv").read_bytes(), expected)
            self.assertIn(row["arm"], row["compile"]["command"][1])

    def test_member_records_cannot_be_relabelled_as_raw_candidate_or_another_preparation(self):
        self.prepare_args.candidate_arm = "material-member-general"
        output, original = self.compiled()
        for mutation in (lambda summary: summary.update(candidate_arm="material-narrow-general"),
                lambda summary: summary.update(arms=list(harness.ARMS)),
                lambda summary: summary["runs"][1].update(arm="material-narrow-general"),
                lambda summary: summary["timing_summary"].update(candidate_arm="material-narrow-general")):
            changed = copy.deepcopy(original)
            mutation(changed)
            save(output / "summary.json", changed)
            harness.profile.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                harness.verify_records(output)

    def test_baseline_mismatch_and_repetition_drift_are_sealed_without_admitting_bad_sample(self):
        preparation = self.prepared()
        for index, variant in enumerate(("baseline", "drift")):
            self.counter = 0
            self.baseline_mismatch, self.variant_drift = variant == "baseline", variant == "drift"
            args = SimpleNamespace(preparation=preparation, output_dir=self.root / f"rejected-{index}", spirv_val=self.validator, timeout=1)
            with patch.object(harness, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext), \
                    self.assertRaises(ValueError):
                harness.run(args)
            summary = harness.experiment.read(args.output_dir / "summary.json")
            self.assertEqual(summary["status"], "failed")
            self.assertLess(len(summary["runs"]), 6)
            harness.legacy.verify_manifest(args.output_dir)

    def test_failed_compiler_records_timeout_pid_logs_and_seal(self):
        preparation = self.prepared()
        self.counter, self.fail_at = 0, 0
        args = SimpleNamespace(preparation=preparation, output_dir=self.root / "failed", spirv_val=self.validator, timeout=1)
        with patch.object(harness, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext), \
                self.assertRaisesRegex(ValueError, "Compile failed"):
            harness.run(args)
        summary = harness.experiment.read(args.output_dir / "summary.json")
        self.assertEqual(summary["runs"], [])
        stage = harness.experiment.read(args.output_dir / "block-00-original-general/compile.process.json")
        self.assertTrue(stage["timed_out"])
        self.assertEqual(stage["pid"], 900)
        harness.legacy.verify_manifest(args.output_dir)

    def test_actual_process_wrapper_preserves_timeout_and_crash_telemetry(self):
        for index, (timed_out, exit_code) in enumerate(((True, 1), (False, -1073741819))):
            directory = self.root / f"process-{index}"
            directory.mkdir()
            waits = []
            process = SimpleNamespace(pid=801 + index, kill=lambda: waits.append("killed"))

            def wait(**kwargs):
                if timed_out and not waits:
                    waits.append("timeout")
                    raise subprocess.TimeoutExpired(["mock compiler"], 1)
                return exit_code

            process.wait = wait
            with self.subTest(timed_out=timed_out), patch.object(harness.backend.subprocess, "Popen", return_value=process), \
                 patch.object(harness.driver, "suppress_windows_error_dialogs", side_effect=nullcontext):
                stage = harness.invoke(["mock compiler"], directory, "compile", 1)
            self.assertIs(stage["timed_out"], timed_out)
            self.assertEqual(stage["pid"], 801 + index)
            self.assertEqual(stage["exit_code"], exit_code)
            self.assertEqual(stage["status"], "failed")
            self.assertEqual(stage, harness.experiment.read(directory / "compile.process.json"))
            self.assertTrue((directory / "compile.stdout.log").is_file())
            self.assertTrue((directory / "compile.stderr.log").is_file())
            self.assertEqual("killed" in waits, timed_out)

    def test_historical_replay_ignores_later_live_sources_and_check_rejects_them(self):
        output, summary = self.compiled()
        (self.sdk / "bin/slangc.exe").write_bytes(b"later compiler SDK")
        self.assertEqual(harness.verify_records(output), summary)
        with self.assertRaisesRegex(ValueError, "input changed"):
            harness.verify_records(output, current_inputs=True)
        with patch.object(harness.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="")) as replay:
            self.assertEqual(harness.verify(output), summary)
        self.assertEqual(replay.call_args.args[0], [sys.executable, "-B", str(output / "Verifier/Tools/Perf/ValidatePathTraceMaterialReads.py"), "_verify-frozen", str(output)])

    def test_resealed_missing_samples_nonfinite_time_or_wrong_original_bytes_reject(self):
        output, original = self.compiled()
        for mutation in (lambda value: value["runs"].pop(), lambda value: value["runs"].reverse(),
                lambda value: value["runs"][0]["compile"].update(wall_seconds=-1),
                lambda value: value["runs"][0].update(arm="material-narrow-general")):
            changed = copy.deepcopy(original)
            mutation(changed)
            save(output / "summary.json", changed)
            harness.profile.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                harness.verify_records(output)
        rows = copy.deepcopy(original["runs"])
        rows[0]["compile"]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            harness.summarize(rows)

    def test_analysis_uses_only_two_same_cleanup_passes_and_retains_raw_arm_identity(self):
        source, _ = self.compiled()
        optimizer = self.root / "spirv-opt.exe"
        optimizer.write_bytes(b"Synthetic optimizer, never executed")
        args = SimpleNamespace(input_dir=source, output_dir=self.root / "analysis", spirv_tools=self.root, timeout=1)
        with patch.object(harness, "verify", side_effect=lambda path: harness.verify_records(path)), \
             patch.object(harness, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext):
            result = harness.analyze(args)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["cleanup_passes"], harness.PASSES)
        for arm in harness.ARMS:
            self.assertEqual(result["outputs"][arm]["raw"], result["outputs"][arm]["cleaned"])
            self.assertEqual(result["outputs"][arm]["cleanup"]["command"][1:5], ["--target-env=vulkan1.4", "--scalar-block-layout", *harness.PASSES])


if __name__ == "__main__":
    unittest.main()
