"""Qualify external SSA bytes and evidence without Slang, SPIR-V tools or GPU."""
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
import ValidatePathTraceSSA as harness
import ValidatePathTraceBackendRendering as rendering
import TestPathTraceBackendRendering as rendering_tests


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def instruction(opcode, *operands):
    return [((len(operands) + 1) << 16) | opcode, *operands]


def string_words(value):
    raw = value.encode() + b"\0"
    raw += b"\0" * (-len(raw) % 4)
    return struct.unpack("<" + "I" * (len(raw) // 4), raw)


def code(*, ssa=False, id_shift=0, entry="main", local_size=8, binding=0, push_offset=0,
         spec_value=4, spec_id=7, float_width=32, precision=True, unknown_pointer=False,
         extra_function=False, call=False, generator=1):
    # These binaries exercise CPU structure/ABI parsers. The process mock does
    # not establish Vulkan validity or the behavior of a real SSA optimizer.
    i = lambda identifier: identifier + id_shift
    words = [0x07230203, 0x00010600, generator, 100 + id_shift, 0]
    words += instruction(17, 1) + instruction(14, 0, 1)
    words += instruction(15, 5, i(13), *string_words(entry), i(8), i(11))
    words += instruction(16, i(13), 17, local_size, 1, 1)
    words += instruction(71, i(6), 2) + instruction(72, i(6), 0, 35, 0)
    words += instruction(71, i(8), 33, binding) + instruction(71, i(8), 34, 0)
    words += instruction(71, i(9), 2) + instruction(72, i(9), 0, 35, push_offset)
    words += instruction(71, i(25), 1, spec_id)
    if precision:
        words += instruction(71, i(8), 0) + instruction(71, i(4), 0)
    words += instruction(19, i(1)) + instruction(20, i(2))
    words += instruction(21, i(3), 32, 0) + instruction(22, i(4), float_width)
    words += instruction(30, i(6), i(3)) + instruction(32, i(7), 12, i(6))
    words += instruction(59, i(7), i(8), 12)
    words += instruction(30, i(9), i(3)) + instruction(32, i(10), 9, i(9))
    words += instruction(59, i(10), i(11), 9)
    words += instruction(33, i(12), i(1))
    words += instruction(43, i(3), i(15), 0) + instruction(43, i(3), i(16), 1)
    words += instruction(32, i(17), 7, i(3)) + instruction(30, i(19), i(3), i(3))
    words += instruction(32, i(18), 7, i(19)) + instruction(32, i(27), 12, i(3))
    words += instruction(50, i(3), i(25), spec_value)
    if ssa:
        words += instruction(1, i(3), i(34))
    if extra_function or call:
        words += instruction(54, i(1), i(40), 0, i(12)) + instruction(248, i(41))
        words += instruction(253) + instruction(56)
    words += instruction(54, i(1), i(13), 0, i(12)) + instruction(248, i(14))
    if not ssa:
        words += instruction(59, i(18), i(26), 7)
        words += instruction(65, i(17), i(21), i(26), i(15))
        words += instruction(62, i(21), i(16)) + instruction(83, i(17), i(22), i(21))
        words += instruction(61, i(3), i(23), i(99) if unknown_pointer else i(22))
    words += instruction(65, i(27), i(28), i(8), i(15))
    words += instruction(61, i(3), i(29), i(28)) + instruction(62, i(28), i(16))
    if call:
        words += instruction(57, i(1), i(42), i(40))
    words += instruction(249, i(30)) + instruction(248, i(30))
    if ssa:
        words += instruction(245, i(3), i(33), i(16), i(14))
    else:
        words += instruction(245, i(17), i(24), i(22), i(14))
        words += instruction(61, i(3), i(31), i(24))
    words += instruction(253) + instruction(56)
    return struct.pack("<" + "I" * len(words), *words)


class SSAEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / harness.SOURCE_RELATIVE
        original = self.source / "original-general"
        original.mkdir(parents=True)
        (original / "Raw.spv").write_bytes(code(generator=2))
        (original / "Cleaned.spv").write_bytes(code())
        self.sdk = self.root / "SDK"
        self.sdk.mkdir()
        for name in ("spirv-opt.exe", "spirv-val.exe", "fixture.dll"):
            (self.sdk / name).write_bytes(b"Synthetic pinned SDK; never executed")
        source_file = self.root / "Shaders/Synthetic.slang"
        source_file.parent.mkdir()
        source_file.write_text("module Synthetic;\n", encoding="utf-8")
        raw_identity = harness.identity(original / "Raw.spv")
        cleaned_identity = harness.identity(original / "Cleaned.spv")
        prep = self.source / "SourceRun/Preparation"
        prep.mkdir(parents=True)
        shutil.copyfile(original / "Raw.spv", prep / "Reference.spv")
        report = rendering_tests.reference_report()
        report.update(build_configuration="Release")
        report["samples"][0].update(spirv_bytes=raw_identity["bytes"], spirv_fnv1a64=raw_identity["fnv1a64"])
        report["samples"][0]["request"]["search_path"] = str(self.root / "Shaders")
        save(prep / "ReferenceReport.json", report)
        reference = self.root / harness.REFERENCE_RELATIVE
        reference.parent.mkdir(parents=True)
        save(reference, report)
        save(prep / "Preparation.json", dict(status="prepared", repository_root=str(self.root),
            reference=raw_identity, live_inputs={str(reference): harness.profile.sha(reference)},
            source_files={"Shaders/Synthetic.slang": harness.profile.sha(source_file)},
            plans=[dict(label="original-general", overlay_metadata=dict(mechanism="original", feature="general",
                requires_no_fiber=False, provider_file=None, provider_module=None, provider_source=None,
                changed_files=[], generated_hashes={}, extra_compiler_flags=[]))]))
        harness.profile.seal(prep)
        save(self.source / "SourceRun/summary.json", dict(status="passed", runs=[
            dict(arm="original-general", identity=raw_identity, block=index) for index in range(3)]))
        harness.profile.seal(self.source / "SourceRun")
        save(self.source / "summary.json", dict(status="passed", command="analyze", experiment=harness.boundary.EXPERIMENT,
            origin=str(self.source), arms=list(harness.boundary.ARMS), cleanup_passes=harness.ANCESTRY_PASSES,
            outputs={"original-general": dict(raw=dict(identity=raw_identity, counts=harness.counts(code(generator=2))),
                cleaned=dict(identity=cleaned_identity, counts=harness.counts(code())))}))
        harness.profile.seal(self.source)
        self.counter = 0
        self.failure, self.drift, self.changed_abi = None, False, False

    def fake_invoke(self, command, directory, name, timeout):
        index = self.counter
        self.counter += 1
        start = datetime.now(timezone.utc).isoformat()
        if "-o" in command:
            path = Path(command[command.index("-o") + 1])
            path.write_bytes(code(ssa=True, binding=1 if self.changed_abi else 0,
                generator=3 if self.drift and index >= 8 else 1))
        (directory / (name + ".stdout.log")).write_text("SPIRV-Tools fixture\n" if "--version" in command else "", encoding="utf-8")
        (directory / (name + ".stderr.log")).write_text("", encoding="utf-8")
        failed = self.failure is not None and index == 3
        stage = dict(command=command, pid=2000 + index, started_utc=start,
            ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=1 + index / 100,
            execution_id=uuid.uuid4().hex, exit_code=1 if failed else 0,
            timed_out=failed and self.failure == "timeout", native_crash=failed and self.failure == "crash",
            status="failed" if failed else "passed", stdout=name + ".stdout.log", stderr=name + ".stderr.log")
        save(directory / (name + ".process.json"), stage)
        return stage

    def prepared(self):
        args = SimpleNamespace(input_dir=self.source, output_dir=self.root / "prepared", spirv_tools=self.sdk, grammar=harness.GRAMMAR_PATH)
        with patch.object(harness, "ROOT", self.root), patch.object(harness.boundary, "verify", side_effect=harness.legacy.verify_manifest):
            harness.prepare(args)
        return args.output_dir

    def run_prepared(self, preparation, output):
        args = SimpleNamespace(preparation=preparation, output_dir=output, timeout=1)
        with patch.object(harness.boundary, "verify", side_effect=harness.legacy.verify_manifest), \
             patch.object(harness, "invoke", side_effect=self.fake_invoke), patch.object(harness.profile, "experiment_lock", side_effect=nullcontext):
            return harness.run(args)

    def compiled(self):
        prepared = self.prepared()
        output = self.root / "run"
        return output, self.run_prepared(prepared, output)

    def verify(self, output, **kwargs):
        with patch.object(harness.boundary, "verify", side_effect=harness.legacy.verify_manifest):
            return harness.verify_records(output, **kwargs)

    def test_schedule_copies_control_and_only_runs_three_independent_ssa_rewrites(self):
        self.assertEqual(harness.schedule(), [(index, arm) for index in range(3) for arm in harness.ARMS])
        command = harness.pass_command("optimizer", "input", "output")
        self.assertEqual(command, ["optimizer", "--target-env=vulkan1.4", "--scalar-block-layout", "--ssa-rewrite", "input", "-o", "output"])

    def test_pointer_result_types_classify_access_chain_copy_and_phi_storage(self):
        metrics = harness.counts(code())
        ssa = metrics["ssa_metrics"]
        self.assertEqual((ssa["function_variables"], ssa["function_loads"], ssa["function_stores"]), (1, 2, 1))
        self.assertEqual(ssa["loads_by_storage_class"], {"Function": 2, "StorageBuffer": 1})
        self.assertEqual(ssa["stores_by_storage_class"], {"Function": 1, "StorageBuffer": 1})
        self.assertEqual((metrics["functions"], metrics["function_calls"], metrics["basic_blocks"], metrics["cfg_edges"]), (1, 0, 2, 1))
        rewritten = harness.counts(code(ssa=True))["ssa_metrics"]
        self.assertEqual((rewritten["function_variables"], rewritten["function_loads"], rewritten["function_stores"], rewritten["phi"], rewritten["undef"]), (0, 0, 0, 1, 1))
        self.assertEqual(rewritten["loads_by_storage_class"], {"StorageBuffer": 1})

    def test_unknown_pointer_extra_function_and_calls_cannot_be_qualified(self):
        for options in ({"unknown_pointer": True}, {"extra_function": True}, {"call": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                harness.qualify_structure(harness.counts(code(**options)))
        for raw in (code()[:-1], struct.pack("<5I", 0x07230203, 0x00010600, 0, 0, 0)):
            with self.assertRaises(ValueError):
                harness.counts(raw)

    def test_abi_is_id_independent_and_ignores_function_body_and_precision(self):
        abi = harness.normalized_abi(code())
        self.assertEqual(abi, harness.normalized_abi(code(id_shift=100, ssa=True, precision=False)))
        metrics = harness.counts(code())["ssa_metrics"]
        reduced = harness.counts(code(precision=False))["ssa_metrics"]
        self.assertEqual(metrics["precision_decorations"], {"RelaxedPrecision": 2})
        self.assertEqual(reduced["precision_decorations"], {})
        self.assertEqual(metrics["floating_point_type_widths"], {"32": 1})
        self.assertEqual(metrics["integer_type_declarations"], {"32-unsigned": 1})

    def test_abi_rejects_entry_execution_resource_push_and_specialization_drift(self):
        abi = harness.normalized_abi(code())
        for options in ({"entry": "other"}, {"local_size": 16}, {"binding": 2}, {"push_offset": 4}, {"spec_id": 9}, {"spec_value": 5}):
            with self.subTest(options=options):
                self.assertNotEqual(abi, harness.normalized_abi(code(**options)))
        # A local numeric type is structural evidence, independently retained
        # even when it is not part of the external binding ABI.
        self.assertEqual(harness.counts(code(float_width=64))["ssa_metrics"]["floating_point_type_widths"], {"64": 1})

    def test_source_provenance_rejects_shadow_overlay_nofiber_and_api_mismatch(self):
        path = self.source / "SourceRun/Preparation/Preparation.json"
        original = harness.experiment.read(path)
        for mutation in (lambda item: item["plans"][0].update(label="shadow-noinline-general"),
                lambda item: item["plans"][0]["overlay_metadata"].update(mechanism="shadow-function-noinline"),
                lambda item: item["plans"][0]["overlay_metadata"].update(requires_no_fiber=True),
                lambda item: item["plans"][0]["overlay_metadata"].update(changed_files=["Shader.slang"]),
                lambda item: item["plans"][0]["overlay_metadata"].update(provider_file="Provider.slang"),
                lambda item: item["plans"][0]["overlay_metadata"].update(extra_compiler_flags=["-O0"])):
            changed = copy.deepcopy(original)
            mutation(changed)
            save(path, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                harness.source_provenance(self.source, verify=False)
        save(path, original)
        (path.parent / "Reference.spv").write_bytes(code(generator=99))
        with self.assertRaisesRegex(ValueError, "API reference"):
            harness.source_provenance(self.source, verify=False)

    def test_prepare_freezes_actual_source_sdk_helpers_and_exact_cleaned_control(self):
        output = self.prepared()
        with patch.object(harness.boundary, "verify", side_effect=harness.legacy.verify_manifest):
            preparation = harness.verify_preparation(output)
        self.assertEqual((output / "Input.spv").read_bytes(), code())
        self.assertEqual(preparation["passes"], {harness.ARMS[0]: [], harness.ARMS[1]: ["--ssa-rewrite"]})
        self.assertEqual(preparation["runtime_postprocess"], "none")
        self.assertTrue((output / "FrozenSDK/spirv-opt.exe").is_file())
        self.assertIn(str(self.root / "Shaders/Synthetic.slang"), preparation["live_inputs"])
        self.assertTrue(any(row["file"] == "Verifier/Tools/Perf/ValidatePathTraceSSA.py" for row in harness.experiment.read(output / "FrozenHelpers.json")))

    def test_three_rewrites_keep_control_exact_full_process_scope_and_pass_provenance(self):
        output, summary = self.compiled()
        self.assertEqual(self.counter, 11)
        self.assertEqual(summary["timing_summary"]["control_optimizer_samples"], 0)
        self.assertAlmostEqual(summary["timing_summary"]["median_ssa_pass_wall_seconds"], 1.06)
        executions = [item["process"]["execution_id"] for item in summary["versions"].values()]
        for row in summary["runs"]:
            if row["arm"] == harness.ARMS[0]:
                self.assertIsNone(row["optimizer"])
                self.assertEqual((output / row["directory"] / "Guides.spv").read_bytes(), code())
            else:
                executions.append(row["optimizer"]["execution_id"])
                self.assertEqual(row["optimizer"]["command"][1:4], ["--target-env=vulkan1.4", "--scalar-block-layout", "--ssa-rewrite"])
            executions.append(row["validation"]["execution_id"])
            self.assertEqual(row["validation"]["command"][1:4], ["--target-env", "vulkan1.4", "--scalar-block-layout"])
        self.assertEqual(len(set(executions)), 11)
        self.assertEqual(len(harness.experiment.read(output / "SnapshotChecks.json")), 16)
        self.assertEqual(summary["outputs"][harness.ARMS[1]]["prepared"]["passes"], ["--ssa-rewrite"])
        self.assertEqual(self.verify(output), summary)

    def test_optimizer_failure_timeout_crash_abi_and_determinism_seal_failed_run(self):
        preparation = self.prepared()
        for condition in ("exit", "timeout", "crash", "abi", "drift"):
            self.counter, self.failure, self.drift, self.changed_abi = 0, condition if condition in ("exit", "timeout", "crash") else None, condition == "drift", condition == "abi"
            output = self.root / ("failed-" + condition)
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                self.run_prepared(preparation, output)
            summary = harness.experiment.read(output / "summary.json")
            self.assertEqual(summary["status"], "failed")
            self.assertNotIn("timing_summary", summary)
            harness.legacy.verify_manifest(output)
            if self.failure:
                sidecar = harness.experiment.read(output / "repeat-00-ssa-general/ssa.process.json")
                self.assertEqual(sidecar["pid"], 2003)
                self.assertEqual(sidecar["timed_out"], condition == "timeout")
                self.assertEqual(sidecar["native_crash"], condition == "crash")

    def test_resealed_missing_samples_wrong_arm_extra_pass_identity_and_overlap_reject(self):
        output, original = self.compiled()
        for mutation in (lambda data: data["runs"].pop(), lambda data: data["runs"].reverse(),
                lambda data: data["runs"][1].update(arm=harness.ARMS[0]),
                lambda data: data["runs"][1]["optimizer"]["command"].append("--simplify-instructions"),
                lambda data: data["outputs"][harness.ARMS[1]]["prepared"].update(passes=["--ssa-rewrite", "--simplify-instructions"]),
                lambda data: data["outputs"][harness.ARMS[1]]["identity"].update(sha256="0" * 64)):
            changed = copy.deepcopy(original)
            mutation(changed)
            save(output / "summary.json", changed)
            harness.profile.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                self.verify(output)
        changed = copy.deepcopy(original)
        stage = changed["runs"][0]["validation"]
        stage["started_utc"] = changed["versions"]["optimizer"]["process"]["started_utc"]
        save(output / changed["runs"][0]["directory"] / "validation.process.json", stage)
        save(output / "summary.json", changed)
        harness.profile.seal(output)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.verify(output)

    def test_nonfinite_zero_boolean_and_missing_optimizer_samples_reject(self):
        records = [dict(arm=harness.ARMS[1], optimizer=dict(wall_seconds=value)) for value in (1, 2, 3)]
        self.assertEqual(harness.summarize(records)["median_ssa_pass_wall_seconds"], 2)
        for value in (float("nan"), float("inf"), 0, -1, True):
            changed = copy.deepcopy(records)
            changed[0]["optimizer"]["wall_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                harness.summarize(changed)
        with self.assertRaises(ValueError):
            harness.summarize(records[:-1])

    def test_historical_replay_uses_actual_archive_and_live_sdk_source_drift_is_optional(self):
        output, summary = self.compiled()
        (self.sdk / "spirv-opt.exe").write_bytes(b"later SDK")
        (self.root / "Shaders/Synthetic.slang").write_text("module Synthetic;\n// later source\n", encoding="utf-8")
        self.assertEqual(self.verify(output), summary)
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.verify(output, current_inputs=True)
        with patch.object(harness.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="")) as replay:
            self.assertEqual(harness.verify(output), summary)
        self.assertEqual(replay.call_args.args[0], [sys.executable, "-B", str(output / "Verifier/Tools/Perf/ValidatePathTraceSSA.py"), "_verify-frozen", str(output)])
        helpers = harness.experiment.read(output / "FrozenHelpers.json")
        owned = next(row for row in helpers if row["file"].endswith("ValidatePathTraceSSA.py"))
        (output / owned["file"]).write_bytes(b"wrong archived helper")
        harness.profile.seal(output)
        with self.assertRaisesRegex(ValueError, "Frozen helper"):
            self.verify(output)

    def test_os_launch_failure_preserves_no_fabricated_pid_and_sealed_logs(self):
        output = self.root / "launch"
        output.mkdir()
        with patch.object(harness.material, "invoke", side_effect=OSError("cannot launch fixture")):
            stage = harness.invoke(["missing.exe"], output, "ssa", 1)
        self.assertEqual(stage["status"], "failed")
        self.assertIsNone(stage["pid"])
        self.assertIsNone(stage["exit_code"])
        self.assertTrue(stage["execution_id"])
        self.assertIn("cannot launch", stage["launch_error"])
        self.assertTrue((output / stage["stdout"]).is_file())
        harness.profile.seal(output)
        harness.legacy.verify_manifest(output)


class PreparedRenderingPolicyTests(unittest.TestCase):
    def setUp(self):
        fixture = rendering_tests.ArtifactRenderingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        manifests, _ = fixture.prepared()
        self.manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        producer = self.root / "producer"
        producer.mkdir()
        nested = producer / "Preparation"
        nested.mkdir()
        (nested / "source.txt").write_bytes(b"frozen source")
        harness.profile.seal(nested)
        control_file, candidate_file = producer / "C.spv", producer / "S.spv"
        control_file.write_bytes(code())
        candidate_file.write_bytes(code(ssa=True))
        control_identity, candidate_identity = harness.identity(control_file), harness.identity(candidate_file)
        outputs = {}
        for arm, artifact in zip(harness.ARMS, (control_identity, candidate_identity)):
            prepared = dict(transform="external-cleaned" if arm == harness.ARMS[0] else "external-ssa",
                input_sha256=control_identity["sha256"], input_bytes=control_identity["bytes"],
                passes=[] if arm == harness.ARMS[0] else ["--ssa-rewrite"], ancestry_passes=harness.ANCESTRY_PASSES,
                tool_version="SPIRV-Tools fixture", source_arm="original-general",
                source_analysis_summary_sha256="a" * 64, preparation_manifest_sha256=harness.profile.sha(nested / "Manifest.json"))
            outputs[arm] = dict(identity=artifact, counts=dict(functions=1, function_calls=0), prepared=prepared)
        self.producer_summary = dict(status="passed", command="run", experiment=harness.EXPERIMENT,
            arms=list(harness.ARMS), runtime_postprocess="none", outputs=outputs)
        save(producer / "summary.json", self.producer_summary)
        harness.profile.seal(producer)
        artifact = self.manifest["artifact"]
        artifact.update(candidate_identity, file=str(candidate_file), spirv_postprocess="none", tool_version="",
            source_tag="Artifacts/PathSSA/ssa-general/SSARewritten", prepared=copy.deepcopy(outputs[harness.ARMS[1]]["prepared"]))
        seal = dict(file=str(producer / "Manifest.json"), sha256=harness.profile.sha(producer / "Manifest.json"))
        artifact["prepared"]["provenance_manifest"] = seal
        self.manifest.update(artifact_id=harness.ARMS[1], requires_no_fiber=False)
        self.manifest["provenance"]["manifests"].append(seal)
        self.producer = producer

    def test_nested_actual_producer_seal_allows_external_ssa_with_runtime_none(self):
        rendering.validate_artifact_postprocess(self.manifest)
        directory = self.root / "runtime"
        report, _, _ = self.fixture.runtime(directory, self.manifest)
        report["shader_artifact"].update(postprocess="none", prepared=self.manifest["artifact"]["prepared"])
        save(directory / "MaterialBaseline.json", report)
        for path in (directory / "recipes").rglob("*.recipe.json"):
            recipe = rendering.experiment.read(path)
            recipe["shaders"]["compute"].update(spirv_postprocess="none")
            save(path, recipe)
        verified = rendering.verify_runtime(directory, self.manifest, self.fixture.config)
        self.assertEqual(verified["report"], str(directory / "MaterialBaseline.json"))
        self.assertEqual(len(verified["recipes"]), 3)

    def test_external_policy_cannot_masquerade_as_cleanup_or_other_arm(self):
        for mutation in (lambda item: item["artifact"].update(spirv_postprocess="dead-branches", tool_version="SPIRV-Tools"),
                lambda item: item.update(artifact_id=harness.ARMS[0]),
                lambda item: item.update(requires_no_fiber=True),
                lambda item: item["artifact"]["prepared"].update(passes=["--ssa-rewrite", "--simplify-instructions"]),
                lambda item: item["artifact"]["prepared"].update(input_sha256="0" * 64),
                lambda item: item["artifact"]["prepared"].update(source_arm="shadow-noinline-general"),
                lambda item: item["artifact"].update(tool_version="SPIRV-Tools")):
            changed = copy.deepcopy(self.manifest)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                rendering.validate_artifact_postprocess(changed)

    def test_runtime_must_report_exact_prepared_provenance_and_actual_recipe_policy(self):
        directory = self.root / "runtime"
        report, _, _ = self.fixture.runtime(directory, self.manifest)
        report["shader_artifact"].update(postprocess="none", prepared=copy.deepcopy(self.manifest["artifact"]["prepared"]))
        save(directory / "MaterialBaseline.json", report)
        with self.assertRaisesRegex(ValueError, "Device artifact provenance/ABI"):
            rendering.verify_runtime(directory, self.manifest, self.fixture.config)
        report["shader_artifact"]["prepared"]["passes"] = []
        save(directory / "MaterialBaseline.json", report)
        with self.assertRaisesRegex(ValueError, "external artifact preparation"):
            rendering.verify_runtime(directory, self.manifest, self.fixture.config)


if __name__ == "__main__":
    unittest.main()
