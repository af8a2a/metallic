"""Qualify a single shadow function boundary without Slang or GPU execution."""
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
import ValidatePathTraceFunctionBoundaries as harness

TARGET = "Modules/OpenPBRPathTracing/OpenPBRPathTrace.slang"
SIGNATURE = "public float3 openPbrShadowTransmittance("


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def instruction(opcode, *operands):
    return [((len(operands) + 1) << 16) | opcode, *operands]


def code(boundary=False, control=2, generator=1):
    # Identity and instruction parsing are tested here. The mocked validator
    # never claims this small binary is a full Vulkan shader.
    words = [0x07230203, 0x00010600, generator, 8 if boundary else 5, 0]
    def string_words(text):
        raw = text.encode("utf-8") + b"\0"
        raw += b"\0" * (-len(raw) % 4)
        return struct.unpack("<" + "I" * (len(raw) // 4), raw)

    words += instruction(15, 5, 3, *string_words("main"))
    if boundary:
        words += instruction(5, 5, *string_words("openPbrShadowTransmittance"))
    words += instruction(19, 1) + instruction(33, 2, 1)
    if boundary:
        words += instruction(54, 1, 5, control, 2) + instruction(248, 6)
        words += instruction(253) + instruction(56)
    words += instruction(54, 1, 3, 0, 2) + instruction(248, 4)
    if boundary:
        words += instruction(57, 1, 7, 5)
    words += instruction(253) + instruction(56)
    return struct.pack("<" + "I" * len(words), *words)


class FunctionBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name in harness.overlay.SOURCE_FILES:
            target = self.root / "Shaders" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "Shaders" / name, target)
        for name in ("Source/Runtime/Render/Core/NamedResourceParameters.h", "Source/Runtime/Render/Core/ColorSpaceMatrices.h",
                "Source/Runtime/Render/Environment/AtmosphereKernelParameters.h", "Source/Runtime/Render/Material/MaterialValueProgram.cpp",
                "Source/Runtime/Render/Material/SceneMaterialData.h", "Tools/CheckShaderModules.py"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.sdk = self.root / "SDK"
        (self.sdk / "bin").mkdir(parents=True)
        (self.sdk / "include").mkdir()
        (self.sdk / "bin/slangc.exe").write_bytes(b"Synthetic compiler; never executed")
        (self.sdk / "include/slang.h").write_bytes(b"Synthetic SDK header")
        (self.sdk / "include/slang-tag-version.h").write_text('#define SLANG_TAG_VERSION "fixture"\n', encoding="utf-8")
        self.validator = self.root / "spirv-val.exe"
        self.validator.write_bytes(b"Synthetic validator; never executed")
        self.cache = self.root / "cache.spv"
        self.cache.write_bytes(b"synthetic cache; decoder has separate tests")
        self.report = dict(status="passed", build_configuration="Release", slang_build="fixture", environment={},
            configuration=dict(compile_only=True, shader_cache="reuse", slang_optimization="O1", descriptor_mode="mapped",
                spirv_postprocess="none", debug_mode="disabled", validation=False), samples=[dict(status="passed",
                pipeline_status="compile_only", shader_cache_hit=False, compilation_phases=dict(compiler_executed=True),
                effective_slang_optimization="O1", effective_descriptor_mode="mapped", effective_spirv_postprocess="none",
                dependencies=[], spirv_bytes=len(code()), spirv_fnv1a64=harness.profile.fnv(code()), request=dict(index=146,
                    module="Features/PathTracing/OpenPBRRayQueryPathTraceGuides", entry="openPbrRayQueryPathTraceGuidesMain",
                    profile="spirv_1_6", search_path=str(self.root / "Shaders"), additional_search_paths=[],
                    capabilities=["spvRayQueryKHR"], defines=[["METALLIC_CUSTOM_MATERIALS", "0"]]))])
        self.report_path = self.root / "report.json"
        save(self.report_path, self.report)
        self.prepare_args = SimpleNamespace(report=self.report_path, request_index=146, slang_root=self.sdk,
            output_dir=self.root / "prepared")
        self.counter, self.fail_at, self.baseline_mismatch, self.variant_drift = 0, None, False, False

    def fake_invoke(self, command, directory, name, timeout):
        index = self.counter
        self.counter += 1
        started = datetime.now(timezone.utc).isoformat()
        if "-o" in command:
            path = Path(command[command.index("-o") + 1])
            boundary = "shadow-noinline-general" in command[1]
            generator = 99 if self.baseline_mismatch and not boundary else 1
            if self.variant_drift and boundary and index >= 6:
                generator = 2
            if name == "cleanup":
                path.write_bytes(Path(command[command.index("-o") - 1]).read_bytes())
            else:
                path.write_bytes(code(boundary, generator=generator))
        (directory / (name + ".stdout.log")).write_text("", encoding="utf-8")
        (directory / (name + ".stderr.log")).write_text("[*] fixturePhase 1 1.25ms\n", encoding="utf-8")
        stage = dict(command=command, pid=1000 + index, started_utc=started,
            ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=1 + index / 100,
            execution_id=uuid.uuid4().hex, exit_code=1 if self.fail_at == index else 0,
            timed_out=self.fail_at == index, native_crash=False, stdout=name + ".stdout.log", stderr=name + ".stderr.log",
            status="failed" if self.fail_at == index else "passed")
        save(directory / (name + ".process.json"), stage)
        return stage

    def prepared(self):
        with patch.object(harness, "ROOT", self.root), patch.object(harness._engine, "ROOT", self.root), patch.object(harness.profile, "ROOT", self.root), \
             patch.object(harness.legacy, "ROOT", self.root), patch.object(harness.profile, "bench_bytes", return_value=(self.cache, code())), \
             patch.object(harness._engine, "invoke", side_effect=self.fake_invoke):
            harness.prepare(self.prepare_args)
        return self.prepare_args.output_dir

    def compiled(self):
        preparation = self.prepared()
        self.counter = 0
        args = SimpleNamespace(preparation=preparation, output_dir=self.root / "compiled", spirv_val=self.validator, timeout=1)
        with patch.object(harness._engine, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext):
            result = harness.run(args)
        return args.output_dir, result

    def test_overlay_changes_only_one_annotation_and_preserves_entire_shadow_body(self):
        source = (self.root / "Shaders" / TARGET).read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in source else "\n"
        original = harness.overlay.plan_overlay(self.root / "Shaders", "original-general")
        candidate = harness.overlay.plan_overlay(self.root / "Shaders", "shadow-noinline-general")
        self.assertEqual(original["files"], {})
        self.assertEqual(set(candidate["files"]), {TARGET})
        emitted = candidate["files"][TARGET]
        self.assertEqual(emitted.count("[noinline]"), source.count("[noinline]") + 1)
        self.assertEqual(emitted.replace("[noinline]" + newline + SIGNATURE, SIGNATURE, 1), source)
        self.assertEqual(emitted.count("implementing OpenPBRPathTracing;"), 1)
        self.assertNotIn("SceneSurfaceMaterialReads", emitted)
        self.assertNotIn("NoFiberOpenPBRBackend", emitted)
        self.assertIn("materialDomain(shadowTransport) == kMaterialDomainFiber", emitted)

    def test_overlay_rejects_unknown_or_preannotated_source_instead_of_combining_experiments(self):
        for arm in ("backend-link-no-fiber", "material-narrow-general", "shadow", None):
            with self.subTest(arm=arm), self.assertRaises((ValueError, TypeError)):
                harness.overlay.plan_overlay(self.root / "Shaders", arm)
        source = self.root / "Shaders" / TARGET
        text = source.read_text(encoding="utf-8")
        source.write_text(text.replace(SIGNATURE, "[noinline]\n" + SIGNATURE, 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            harness.overlay.plan_overlay(self.root / "Shaders", "shadow-noinline-general")

    def test_schedule_is_six_fresh_ab_ba_ab_compiler_processes(self):
        self.assertEqual(harness.schedule(), [(0, "original-general"), (0, "shadow-noinline-general"),
            (1, "shadow-noinline-general"), (1, "original-general"), (2, "original-general"), (2, "shadow-noinline-general")])

    def test_reference_requires_original_general_fresh_o1_without_feature_restriction(self):
        mutations = (lambda report: report["configuration"].update(slang_optimization="O2"),
            lambda report: report["configuration"].update(spirv_postprocess="dead-branches"),
            lambda report: report["samples"][0].update(shader_cache_hit=True),
            lambda report: report["samples"][0]["request"].update(defines=[["METALLIC_CUSTOM_MATERIALS", "0"], ["SHARC_QUERY", "1"]]),
            lambda report: report.update(build_configuration="Debug"))
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                harness._engine.reference_sample(report)

    def test_prepare_freezes_both_owned_source_trees_and_actual_annotation_identity(self):
        output = self.prepared()
        preparation = harness.verify_preparation(output)
        self.assertEqual(preparation["arms"], list(harness.ARMS))
        original = (output / "FrozenSources/Shaders" / TARGET).read_bytes()
        self.assertEqual((output / "overlays/original-general/Shaders" / TARGET).read_bytes(), original)
        candidate = (output / "overlays/shadow-noinline-general/Shaders" / TARGET).read_bytes()
        newline = b"\r\n" if b"\r\n" in original else b"\n"
        self.assertEqual(candidate.replace(b"[noinline]" + newline + SIGNATURE.encode(), SIGNATURE.encode(), 1), original)
        self.assertTrue(all("-O1" in plan["direct_command"] for plan in preparation["plans"]))

    def test_synthetic_function_boundary_counts_real_calls_dontinline_and_cfg(self):
        original = harness.counts(code())
        candidate = harness.counts(code(True))
        self.assertEqual((original["functions"], original["function_calls"], original["basic_blocks"]), (1, 0, 1))
        self.assertEqual((candidate["functions"], candidate["function_calls"], candidate["basic_blocks"]), (2, 1, 2))
        self.assertEqual(candidate["function_boundaries"]["dont_inline_functions"], 1)
        self.assertEqual(original["function_boundaries"]["dont_inline_functions"], 0)
        self.assertEqual(candidate["function_boundaries"]["entry_reachable_call_depth"], 1)
        self.assertEqual(original["function_boundaries"]["entry_reachable_call_depth"], 0)
        self.assertEqual(harness.counts(code(True, control=1))["function_boundaries"]["dont_inline_functions"], 0)
        with self.assertRaises(ValueError):
            harness.counts(code(True)[:-1])

    def test_dontinline_flag_requires_a_called_reachable_boundary_and_cycles_reject(self):
        inline = harness.counts(code(True, control=1))
        with self.assertRaisesRegex(ValueError, "boundary"):
            harness.retained_targets(inline)
        words = list(struct.unpack("<" + "I" * (len(code(True)) // 4), code(True)))
        cursor = 5
        while cursor < len(words):
            size = words[cursor] >> 16
            if words[cursor] & 65535 == 57:
                call_offset = cursor
                break
            cursor += size
        removed = words[:call_offset] + words[call_offset + 4:]
        unreachable = harness.counts(struct.pack("<" + "I" * len(removed), *removed))
        self.assertEqual(unreachable["function_boundaries"]["dont_inline_functions"], 1)
        self.assertEqual(unreachable["function_boundaries"]["entry_reachable_call_depth"], 0)
        with self.assertRaisesRegex(ValueError, "boundary"):
            harness.retained_targets(unreachable)
        words[call_offset + 3] = 3
        with self.assertRaisesRegex(ValueError, "(?i)(recursive|cyclic)"):
            harness.counts(struct.pack("<" + "I" * len(words), *words))

    def test_adapter_does_not_modify_existing_material_read_harness_globals(self):
        import ValidatePathTraceMaterialReads as existing
        self.assertEqual(existing.ARMS, ("original-general", "material-narrow-general"))
        self.assertEqual(existing.selected_arms(), existing.ARMS)
        self.assertEqual(existing.EXPERIMENT, "material-narrow-reads-v1")
        self.assertNotIn("function_boundaries", existing.counts(code()))
        self.assertIn("function_boundaries", harness.counts(code()))

    def test_six_samples_keep_baseline_exact_and_candidate_repeat_bytes_and_actual_process_contract(self):
        output, summary = self.compiled()
        self.assertEqual([(row["block"], row["arm"]) for row in summary["runs"]], harness.schedule())
        self.assertEqual(len({row[name]["execution_id"] for row in summary["runs"] for name in ("compile", "validation")}), 12)
        self.assertEqual(summary["timing_summary"]["denominator_arm"], "original-general")
        for row in summary["runs"]:
            self.assertEqual(row["counts"]["functions"], 2 if row["arm"] == "shadow-noinline-general" else 1)
            self.assertEqual(row["validation"]["command"][1:4], ["--target-env", "vulkan1.4", "--scalar-block-layout"])
        self.assertEqual(harness.verify_records(output), summary)

    def test_baseline_mismatch_drift_and_timeout_seal_failed_evidence_without_sample_admission(self):
        preparation = self.prepared()
        for index, condition in enumerate(("baseline", "drift", "timeout")):
            self.counter = 0
            self.baseline_mismatch, self.variant_drift = condition == "baseline", condition == "drift"
            self.fail_at = 0 if condition == "timeout" else None
            args = SimpleNamespace(preparation=preparation, output_dir=self.root / f"failed-{index}", spirv_val=self.validator, timeout=1)
            with patch.object(harness._engine, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext), \
                    self.assertRaises(ValueError):
                harness.run(args)
            summary = harness.experiment.read(args.output_dir / "summary.json")
            self.assertEqual(summary["status"], "failed")
            self.assertLess(len(summary["runs"]), 6)
            harness.legacy.verify_manifest(args.output_dir)
            if condition == "timeout":
                sidecar = harness.experiment.read(args.output_dir / "block-00-original-general/compile.process.json")
                self.assertTrue(sidecar["timed_out"])
                self.assertEqual(sidecar["pid"], 1000)

    def test_resealed_source_arm_command_counts_and_missing_sample_reject(self):
        output, original = self.compiled()
        mutations = (lambda data: data["runs"].pop(), lambda data: data["runs"].reverse(),
            lambda data: data["runs"][0].update(arm="shadow-noinline-general"),
            lambda data: data["runs"][1]["counts"]["function_boundaries"].update(dont_inline_functions=0),
            lambda data: data["runs"][0]["compile"]["command"].append("-O0"))
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            save(output / "summary.json", changed)
            harness.profile.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                harness.verify_records(output)
        records = copy.deepcopy(original["runs"])
        records[0]["compile"]["wall_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            harness.summarize(records)

    def test_historical_verifier_replays_actual_archive_after_live_sdk_drift(self):
        output, summary = self.compiled()
        (self.sdk / "bin/slangc.exe").write_bytes(b"later compiler")
        self.assertEqual(harness.verify_records(output), summary)
        with self.assertRaisesRegex(ValueError, "input changed"):
            harness.verify_records(output, current_inputs=True)
        with patch.object(harness.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="")) as replay:
            self.assertEqual(harness.verify(output), summary)
        self.assertEqual(replay.call_args.args[0], [sys.executable, "-B", str(output / "Verifier/Tools/Perf/ValidatePathTraceFunctionBoundaries.py"), "_verify-frozen", str(output)])

    def test_uniform_two_pass_cleanup_retains_analyzed_real_function_call_counts(self):
        source, _ = self.compiled()
        (self.root / "spirv-opt.exe").write_bytes(b"Synthetic optimizer; never executed")
        args = SimpleNamespace(input_dir=source, output_dir=self.root / "analysis", spirv_tools=self.root, timeout=1)
        with patch.object(harness._engine, "verify", side_effect=lambda path: harness.verify_records(path)), \
             patch.object(harness._engine, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext):
            summary = harness.analyze(args)
        self.assertEqual(summary["cleanup_passes"], ["--eliminate-dead-branches", "--simplify-instructions"])
        candidate = summary["outputs"]["shadow-noinline-general"]
        self.assertEqual(candidate["cleaned"]["counts"]["function_calls"], 1)
        self.assertEqual(candidate["cleaned"]["counts"]["function_boundaries"]["dont_inline_functions"], 1)
        self.assertEqual(candidate["raw"], candidate["cleaned"])


if __name__ == "__main__":
    unittest.main()
