"""Reject incomplete or changed evidence before attributing link-time benefits."""

import argparse
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import ValidateSlangLinkTimeSpecialization as experiment


CODE = struct.pack("<5I", 0x07230203, 0x00010600, 0, 4, 0)


def riff_module(payload):
    return b"RIFF" + struct.pack("<I", len(payload) + 4) + b"SLmc" + payload


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class SlangLinkTimeSpecializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.preparation = self.root / "preparation"
        self.preparation.mkdir()
        self.compiler = self.root / "compiler.exe"
        self.compiler.write_bytes(b"pinned test compiler")
        (self.preparation / "Reference.spv").write_bytes(CODE)
        overlay = self.preparation / "overlays/original/Shaders/Guide.slang"
        overlay.parent.mkdir(parents=True)
        overlay.write_text("// frozen shader\n", encoding="utf-8")
        self.plans = [
            dict(label="original-general", mechanism="original",
                 feature_mask=experiment.feature_values("general"), provider=None,
                 direct_command=[str(self.compiler), str(overlay), "-module-name", "Guide",
                                 "-entry", "guideMain", "-target", "spirv", "-O1", "-o", "unused.spv"]),
            dict(label="static-opaque-dielectric", mechanism="static",
                 feature_mask=experiment.feature_values("opaque-dielectric"), provider=None,
                 direct_command=[str(self.compiler), str(overlay), "-module-name", "Guide",
                                 "-entry", "guideMain", "-target", "spirv", "-O1", "-o", "unused.spv"]),
        ]
        self.preparation_data = dict(schema_version=1, status="prepared", plans=self.plans,
            compiler=str(self.compiler), compiler_sha256=experiment.profile.sha(self.compiler),
            reference_fnv1a64=experiment.profile.fnv(CODE),
            reference_sha256=hashlib.sha256(CODE).hexdigest())
        save(self.preparation / "Preparation.json", self.preparation_data)
        experiment.profile.seal(self.preparation)

    def tearDown(self):
        self.temporary.cleanup()

    def completed_run(self, repetitions=2):
        output = self.root / "run"
        output.mkdir()
        records = []
        for repetition in range(repetitions):
            ordered = self.plans if repetition % 2 == 0 else list(reversed(self.plans))
            for plan in ordered:
                directory = output / f"block-{repetition:02}-{plan['label']}"
                directory.mkdir()
                shader = directory / "Guides.spv"
                shader.write_bytes(CODE)
                stages = []
                stage_names = (["source-direct"] if plan["mechanism"] == "original" else [])
                stage_names += ["consumer-precompile"]
                if plan["provider"]:
                    stage_names.append("provider-precompile")
                stage_names.append("link")
                for name in stage_names:
                    (directory / (name + ".stdout.log")).write_bytes(b"compiler output")
                    (directory / (name + ".stderr.log")).write_bytes(b"")
                    stages.append(dict(status="passed", exit_code=0, wall_seconds=.1,
                        command=[str(self.compiler), name], stdout=name + ".stdout.log",
                        stderr=name + ".stderr.log"))
                if plan["mechanism"] == "original":
                    (directory / "Direct.spv").write_bytes(CODE)
                records.append(dict(label=plan["label"], mechanism=plan["mechanism"],
                    feature_mask=plan["feature_mask"], repetition=repetition,
                    directory=directory.name, status="passed",
                    stages=stages,
                    spirv_bytes=len(CODE), spirv_fnv1a64=experiment.profile.fnv(CODE),
                    spirv_sha256=hashlib.sha256(CODE).hexdigest(),
                    matches_current_bench_byte_for_byte=True))
        summary = dict(schema_version=1, status="passed", repetitions=repetitions,
            variants=[plan["label"] for plan in self.plans], runs=records,
            preparation=str(self.preparation),
            preparation_manifest_sha256=experiment.profile.sha(self.preparation / "Manifest.json"))
        save(output / "summary.json", summary)
        experiment.profile.seal(output)
        return output, summary

    def assert_invalid_run(self, output, summary):
        save(output / "summary.json", summary)
        experiment.profile.seal(output)
        with self.assertRaises(ValueError):
            experiment.verify_run(output)

    def test_sealed_preparation_detects_changed_shader_or_compiler(self):
        self.assertEqual(experiment.verify_preparation(self.preparation)["status"], "prepared")
        shader = self.preparation / "overlays/original/Shaders/Guide.slang"
        shader.write_text("// different shader\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum|manifest|Manifest"):
            experiment.verify_preparation(self.preparation)
        shader.write_text("// frozen shader\n", encoding="utf-8")
        self.compiler.write_bytes(b"different compiler")
        with self.assertRaisesRegex(ValueError, "compiler"):
            experiment.verify_preparation(self.preparation)

    def test_unlisted_module_is_rejected_even_when_all_listed_hashes_match(self):
        extra = self.preparation / "overlays/original/Shaders/Guide.slang-module"
        extra.write_bytes(b"unexpected precompiled module")
        with self.assertRaises(ValueError):
            experiment.verify_preparation(self.preparation)

    def test_manifest_cannot_reference_outside_preparation(self):
        manifest_path = self.preparation / "Manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"].append(dict(path="../compiler.exe", sha256=experiment.profile.sha(self.compiler)))
        save(manifest_path, manifest)
        with self.assertRaises(ValueError):
            experiment.verify_preparation(self.preparation)

    def test_complete_alternating_schedule_is_verified(self):
        output, summary = self.completed_run()
        self.assertEqual(experiment.verify_run(output)["runs"], summary["runs"])

    def test_failed_or_missing_runs_cannot_be_analyzed_as_success(self):
        output, summary = self.completed_run()
        for mutation in (lambda value: value.update(status="failed"),
                         lambda value: value["runs"].pop(),
                         lambda value: value["runs"][0].update(status="failed"),
                         lambda value: value["runs"][0]["stages"][0].update(exit_code=1),
                         lambda value: value["runs"].clear()):
            changed = copy.deepcopy(summary)
            mutation(changed)
            with self.subTest(mutation=mutation):
                self.assert_invalid_run(output, changed)

    def test_duplicate_reordered_or_wrong_feature_schedule_is_rejected(self):
        output, summary = self.completed_run()
        def reorder(value):
            value["runs"][0], value["runs"][1] = value["runs"][1], value["runs"][0]
        for mutation in (reorder,
                         lambda value: value["runs"].append(copy.deepcopy(value["runs"][0])),
                         lambda value: value["runs"][0].update(repetition=1),
                         lambda value: value["runs"][0].update(mechanism="type"),
                         lambda value: value["runs"][0].update(feature_mask=experiment.feature_values("no-fiber")),
                         lambda value: value["variants"].append("missing-variant")):
            changed = copy.deepcopy(summary)
            mutation(changed)
            with self.subTest(mutation=mutation):
                self.assert_invalid_run(output, changed)

    def test_resealed_shader_must_still_match_recorded_identity(self):
        output, summary = self.completed_run()
        shader = output / summary["runs"][0]["directory"] / "Guides.spv"
        shader.write_bytes(CODE[:-1] + b"\1")
        experiment.profile.seal(output)
        with self.assertRaises(ValueError):
            experiment.verify_run(output)

    def test_recorded_size_fnv_and_sha_are_independently_checked(self):
        output, summary = self.completed_run()
        for field, value in (("spirv_bytes", len(CODE) + 4),
                             ("spirv_fnv1a64", "0000000000000000"),
                             ("spirv_sha256", "0" * 64)):
            changed = copy.deepcopy(summary)
            changed["runs"][0][field] = value
            with self.subTest(field=field):
                self.assert_invalid_run(output, changed)

    def test_original_baseline_cannot_claim_a_different_reference(self):
        output, summary = self.completed_run()
        direct = output / summary["runs"][0]["directory"] / "Direct.spv"
        direct.write_bytes(CODE[:-1] + b"\1")
        experiment.profile.seal(output)
        with self.assertRaises(ValueError):
            experiment.verify_run(output)

    def test_preparation_change_invalidates_recorded_run_binding(self):
        output, _ = self.completed_run()
        self.preparation_data["scope"] = "changed after run"
        save(self.preparation / "Preparation.json", self.preparation_data)
        experiment.profile.seal(self.preparation)
        with self.assertRaises(ValueError):
            experiment.verify_run(output)

    def test_run_cannot_alias_another_variants_shader_directory(self):
        output, summary = self.completed_run()
        changed = copy.deepcopy(summary)
        changed["runs"][0]["directory"] = changed["runs"][1]["directory"]
        self.assert_invalid_run(output, changed)

    def test_compiler_arguments_preserve_literal_macro_and_provider_inputs(self):
        command = ["compiler with spaces.exe", "shader with spaces.slang", "-module-name", "Guide",
                   "-entry", "guideMain", "-target", "spirv", "-emit-spirv-directly",
                   "-profile", "spirv_1_6", "-D", "TEXT=a b;$()", "-O1", "-o", "unused.spv"]
        compiler, source, flags, module, entry = experiment.command_parts(dict(direct_command=command))
        self.assertEqual((compiler, source, module, entry),
                         (command[0], command[1], "Guide", "guideMain"))
        self.assertIn("TEXT=a b;$()", flags)
        self.assertIn("-O1", flags)
        self.assertNotIn("-target", flags)
        self.assertNotIn("-o", flags)

    def test_only_shader_source_and_include_paths_are_rewritten(self):
        overlay = self.root / "overlay"
        compiler = str(ROOT / "External/slang/bin/slangc.exe")
        source = str(ROOT / "Shaders/Entry.slang")
        includes = [str(ROOT / "Shaders/Modules"), str(ROOT / "External/ShaderVendor"), str(ROOT)]
        output = str(ROOT / "build/output.spv")
        module_name = str(ROOT / "Shaders/ModuleName")
        macro = "PATH=" + str(ROOT / "Shaders/Texture")
        command = [compiler, source, "-I", includes[0], "-I", includes[1], "-I", includes[2],
                   "-o", output, "-module-name", module_name, "-D", macro]
        rewritten = experiment.rewrite_command_inputs(command, overlay, [ROOT])
        self.assertEqual(rewritten[1], str(overlay / "Shaders/Entry.slang"))
        self.assertEqual([rewritten[index + 1] for index, token in enumerate(rewritten) if token == "-I"],
                         [str(overlay / "Shaders/Modules"), str(overlay / "External/ShaderVendor"), str(overlay)])
        self.assertEqual(rewritten[0], compiler)
        self.assertEqual(rewritten[rewritten.index("-o") + 1], output)
        self.assertEqual(rewritten[rewritten.index("-module-name") + 1], module_name)
        self.assertEqual(rewritten[rewritten.index("-D") + 1], macro)
        self.assertEqual(command[1], source)

    def test_unsnapshotted_and_relative_include_roots_remain_unchanged(self):
        source = str(ROOT / "Shaders/Entry.slang")
        unrelated = str(self.root / "SDK/Include")
        command = ["compiler.exe", source, "-I", unrelated, "-I", "relative/include"]
        rewritten = experiment.rewrite_command_inputs(command, self.root / "overlay", [ROOT / "Shaders"])
        self.assertEqual(rewritten[2:], command[2:])

    def initial_ir_run(self):
        provider = self.preparation / "providers/opaque.slang"
        provider.parent.mkdir()
        provider.write_text("// fixed provider\n", encoding="utf-8")
        plan = copy.deepcopy(self.plans[1])
        plan.update(label="constant-opaque-dielectric", mechanism="constant", provider=str(provider))
        self.plans.append(plan)
        save(self.preparation / "Preparation.json", self.preparation_data)
        experiment.profile.seal(self.preparation)
        output, summary = self.completed_run(repetitions=1)
        for record in summary["runs"]:
            directory = output / record["directory"]
            (directory / "Consumer.slang-module").write_bytes(riff_module(b"consumer:" + record["label"].encode()))
            if record["mechanism"] == "constant":
                (directory / "Provider.slang-module").write_bytes(riff_module(b"provider:" + record["label"].encode()))
        experiment.profile.seal(output)
        return output

    def relink_args(self, initial):
        return argparse.Namespace(input_dir=initial, output_dir=self.root / "relink",
            variants=["constant-opaque-dielectric"], repetitions=1, timeout=30,
            target_optimization="O2")

    def mock_relink(self, args):
        def invoke(command, directory, name, timeout):
            Path(command[command.index("-o") + 1]).write_bytes(CODE)
            (directory / (name + ".stdout.log")).write_bytes(b"relink output")
            (directory / (name + ".stderr.log")).write_bytes(b"")
            return dict(command=command, exit_code=0, wall_seconds=.05, status="passed",
                        stdout=name + ".stdout.log", stderr=name + ".stderr.log")
        with patch.object(experiment.profile, "experiment_lock", return_value=nullcontext()), \
                patch.object(experiment, "compiler_manifest", return_value=[]), \
                patch.object(experiment, "invoke", side_effect=invoke) as process, redirect_stdout(io.StringIO()):
            result = experiment.relink(args)
        return result, process.call_args_list

    def test_relink_reuses_exact_ir_and_separates_precompile_from_final_optimization(self):
        initial = self.initial_ir_run()
        args = self.relink_args(initial)
        result, calls = self.mock_relink(args)
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 1)
        command = calls[0].args[0]
        self.assertIn("-O2", command)
        self.assertNotIn("-O1", command)
        self.assertNotIn("-no-codegen", command)
        summary = experiment.verify_run(args.output_dir)
        self.assertEqual(summary["mode"], "relink")
        self.assertEqual(summary["final_target_optimization"], "O2")
        record = summary["runs"][0]
        self.assertAlmostEqual(record["initial_precompile_wall_seconds"], .2)
        self.assertEqual(record["stages"][0]["wall_seconds"], .05)
        self.assertEqual(len(record["reused_ir"]), 2)
        for artifact in record["reused_ir"]:
            source = Path(artifact["source"])
            copied = args.output_dir / record["directory"] / artifact["destination"]
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            self.assertEqual(artifact["sha256"], experiment.profile.sha(source))

    def test_relink_evidence_rejects_wrong_provider_origin_duplicate_and_changed_copy(self):
        initial = self.initial_ir_run()
        args = self.relink_args(initial)
        self.mock_relink(args)
        summary = experiment.verify_run(args.output_dir)
        for mutation in (
                lambda value: value["runs"][0]["reused_ir"].append(copy.deepcopy(value["runs"][0]["reused_ir"][0])),
                lambda value: value["runs"][0]["reused_ir"].pop(),
                lambda value: value["runs"][0]["reused_ir"][1].update(source=value["runs"][0]["reused_ir"][0]["source"]),
                lambda value: value["runs"][0]["reused_ir"][1].update(bytes=1)):
            changed = copy.deepcopy(summary)
            mutation(changed)
            with self.subTest(mutation=mutation):
                self.assert_invalid_run(args.output_dir, changed)
        save(args.output_dir / "summary.json", summary)
        provider = args.output_dir / summary["runs"][0]["directory"] / "Provider.slang-module"
        provider.write_bytes(riff_module(b"different provider"))
        experiment.profile.seal(args.output_dir)
        with self.assertRaises(ValueError):
            experiment.verify_run(args.output_dir)

    def test_malformed_frozen_module_never_reaches_relink_process(self):
        initial = self.initial_ir_run()
        module = initial / "block-00-constant-opaque-dielectric/Provider.slang-module"
        args = self.relink_args(initial)
        for index, malformed in enumerate((b"RIFF", b"NOT!" + struct.pack("<I", 4) + b"SLmc",
                                            b"RIFF" + struct.pack("<I", 4) + b"WRNG")):
            module.write_bytes(malformed)
            experiment.profile.seal(initial)
            args.output_dir = self.root / f"bad-relink-{index}"
            with self.subTest(malformed=malformed), \
                    patch.object(experiment.profile, "experiment_lock", return_value=nullcontext()), \
                    patch.object(experiment, "compiler_manifest", return_value=[]), \
                    patch.object(experiment, "invoke") as process, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Slang module"):
                    experiment.relink(args)
                process.assert_not_called()

    def test_exact_original_control_refuses_o2_relink_before_process_creation(self):
        initial = self.initial_ir_run()
        args = self.relink_args(initial)
        args.variants = ["original-general"]
        with patch.object(experiment, "invoke") as process:
            with self.assertRaisesRegex(ValueError, "requires O1"):
                experiment.relink(args)
            process.assert_not_called()

    def test_malformed_spirv_is_rejected_before_structural_counts(self):
        for data in (b"bad", CODE[:19], CODE[:-4] + b"\0\0\0\0" + struct.pack("<I", 0),
                     CODE + struct.pack("<I", (4 << 16) | 54)):
            with self.subTest(data=data), self.assertRaises(ValueError):
                experiment.instruction_counts(data)

    def test_finite_positive_limits_are_required_before_any_process(self):
        for timeout in ("nan", "inf", "0", "-1"):
            argv = ["ValidateSlangLinkTimeSpecialization.py", "run", "--preparation", "unused",
                    "--output-dir", "unused", "--timeout", timeout]
            with self.subTest(timeout=timeout), patch.object(sys, "argv", argv), \
                    patch.object(experiment.subprocess, "run") as process, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    experiment.main()
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
