"""Qualify original-General native PSO controls without a compiler or GPU."""

from contextlib import nullcontext, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import MeasurePathTraceBackendNative as native


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def code(generator=1):
    # Only identity is checked here; real validation is mandatory in frozen
    # compiler/cleanup provenance and is never replaced by this synthetic blob.
    return struct.pack("<5I", 0x07230203, 0x00010600, generator, 5, 0)


class BackendNativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.raw = code()
        self.artifact = self.root / "Cleaned.spv"
        self.artifact.write_bytes(self.raw)
        self.identity = native.blob_identity(self.artifact)
        self.source_device = dict(device_name="Fixture GPU", vendor_id=0x10de, device_id=0x2d03,
            api_version=4206592, driver_version=1, driver_name="Fixture Driver", pipeline_cache_uuid="01" * 16,
            validation_enabled=False, synchronization_validation_enabled=False)
        shader = dict(entry_point="main", module_name="Artifacts/PathBackend/original-general/TwoPassCleaned",
            source_entry_point="openPbrRayQueryPathTraceGuidesMain", spirv_file=self.artifact.name,
            spirv_bytes=self.identity["bytes"], input_spirv_hash=self.identity["fnv1a64"],
            slang_optimization="O1", debug_mode="release", descriptor_heap_mode="mapped",
            source_identity_ambiguous=False, spirv_postprocess="dead-branches",
            spirv_postprocess_tool_version="SPIRV-Tools fixture")
        self.recipe = dict(version=1, pipeline_type="compute", shaders=dict(compute=shader),
            state=dict(uses_bindless_heap=True, indirect_bindable=False, bindless_user_push_data_size=12,
                custom_descriptor_mappings=[], compute_backend=0, disable_optimization=False),
            source_device=self.source_device)
        self.recipe["expected_pso_hash"] = native.pso_hash(self.recipe, self.raw)

    def report(self, native_ms=10.0, driver_ms=9.0, lock_ms=0.0):
        shader = self.recipe["shaders"]["compute"]
        stage = dict(stage="compute", module=shader["module_name"], entry="main", source_entry=shader["source_entry_point"],
            debug_mode="release", descriptor_mode="mapped", slang_optimization="O1", spirv_postprocess="dead-branches",
            spirv_postprocess_tool_version=shader["spirv_postprocess_tool_version"], source_identity_ambiguous=False,
            spirv_bytes=self.identity["bytes"], spirv_fnv1a64=self.identity["fnv1a64"],
            device_spirv_bytes=self.identity["bytes"], device_spirv_fnv1a64=self.identity["fnv1a64"],
            device_shader_content_hash=self.identity["content_hash"],
            spirv_statistics=dict(instructions=5, functions=1), device_spirv_statistics=dict(instructions=5, functions=1))
        pipeline = dict(native_create_ms=native_ms, driver_ms=driver_ms, cache_lock_wait_ms=lock_ms,
            feedback_valid=True, native_cache_hit=False, metadata_hit=False, native_object_reused=False,
            recipe_hash_matches=True, recipe_state_overridden=False, compute_backend="pipeline",
            pso_hash=int(self.recipe["expected_pso_hash"], 16), expected_pso_hash=int(self.recipe["expected_pso_hash"], 16),
            native_binary_cache_hit=False, native_binary_driver_rejected=False, native_binary_persisted=False,
            native_binary_data_bytes=0, native_binary_import_ms=0, native_binary_export_ms=0)
        sample = dict(status="passed", pipeline_status="created", shader_source="recipe_spirv", pipeline_type="compute",
            effective_slang_optimization="O1", effective_descriptor_mode="mapped", effective_spirv_postprocess="dead-branches",
            compilation_phases=dict(compiler_executed=False), compile_wall_ms=0, shader_stages=[stage],
            pipeline=pipeline, pipeline_attempt=copy.deepcopy(pipeline),
            spirv_bytes=self.identity["bytes"], spirv_fnv1a64=self.identity["fnv1a64"],
            device_spirv_bytes=self.identity["bytes"], device_spirv_fnv1a64=self.identity["fnv1a64"],
            device_shader_content_hash=self.identity["content_hash"],
            spirv_statistics=copy.deepcopy(stage["spirv_statistics"]),
            device_spirv_statistics=copy.deepcopy(stage["device_spirv_statistics"]))
        device = dict(self.source_device, name=self.source_device["device_name"], validation_active=False)
        batch = dict(pipeline_count=1, successful_pipelines=1, actual_workers=1, requested_workers=1,
            native_hits=0, native_misses=1, metadata_hits=0, metadata_misses=1, unknown_feedback=0,
            valid_driver_feedback_count=1, validation_errors=0, cache_count=1, mode="serial", sample_indices=[0],
            attempted_native_create_sum_ms=native_ms, successful_native_create_sum_ms=native_ms,
            attempted_cache_lock_wait_sum_ms=lock_ms, successful_cache_lock_wait_sum_ms=lock_ms, driver_duration_sum_ms=driver_ms)
        return dict(status="passed", validation_errors=0, pipelines_created=1, pipelines_skipped=0, failures=0,
            pipeline_batches=[batch], samples=[sample], device=device,
            recipe_source_device_matches=True, build_configuration="Release", slang_build="Fixture Slang 2026.18.2",
            shader_target="spirv_1_6", environment={native.driver.CACHE_ENVIRONMENT: "disabled",
                "METALLIC_VK_COMPUTE_BACKEND": "pipeline", "METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION": "0"},
            configuration=dict(compile_only=False, compute_backend="pipeline", debug_mode="disabled", descriptor_mode="mapped",
                disable_driver_optimization=False, iterations=1, mode="serial", pipeline_cache="empty", pipeline_replicas=1,
                workers=1, shader_cache="reuse", validation=False, native_binary_cache_directory="", application_pipeline_cache_used=True))

    def test_identity_uses_raw_hash_for_input_and_size_framed_device_content_hash(self):
        identity = native.blob_identity(self.artifact)
        self.assertEqual(identity["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(identity["bytes"], len(self.raw))
        self.assertEqual(identity["fnv1a64"], f"{native.fnv(self.raw):016x}")
        self.assertNotEqual(identity["content_hash"], int(identity["fnv1a64"], 16))
        for raw in (b"", self.raw[:-1], b"bad!" + self.raw[4:]):
            self.artifact.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                native.blob_identity(self.artifact)

    def test_original_is_explicit_default_and_only_matching_general_controls_are_allowed(self):
        expected = (("original-general", "backend-link-no-fiber"), ("backend-link-no-fiber", "original-general"),
            ("original-general", "backend-link-no-fiber"))
        self.assertEqual(tuple(tuple(pair) for pair in native.schedule()), expected)
        linked = tuple(tuple(pair) for pair in native.schedule("backend-link-general"))
        self.assertEqual(linked, tuple(tuple("backend-link-general" if arm == "original-general" else arm for arm in pair) for pair in expected))
        for arm in ("backend-concrete-general", "backend-link-no-fiber", "original", "", None, True, ["original-general"]):
            with self.subTest(arm=arm), self.assertRaises((ValueError, TypeError)):
                native.schedule(arm)

    def test_replay_command_creates_one_empty_cache_native_pipeline_without_shader_or_dispatch_work(self):
        destination = self.root / "process"
        command = native.native_command(self.root / "bench.exe", self.root / "Original.recipe.json", destination)
        for option, value in (("--recipe-file", str(self.root / "Original.recipe.json")), ("--iterations", "1"),
                ("--mode", "serial"), ("--workers", "1"), ("--pipeline-cache", "empty"),
                ("--shader-cache", "reuse"), ("--descriptor-mode", "mapped"), ("--debug-mode", "disabled")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertNotIn("--request-index", command)
        self.assertNotIn("--compile-only", command)
        self.assertNotIn("--validation", command)
        self.assertEqual(command[command.index("--cache-dir") + 1], str(destination / "cache"))

    def test_valid_native_metrics_include_exact_actual_source_device_identity_and_zero_lock_wait(self):
        report = self.report()
        values, identity = native.validate_report(report, self.recipe, self.identity)
        self.assertEqual(values, dict(native_create_ms=10.0, driver_ms=9.0, cache_lock_wait_ms=0.0))
        self.assertEqual(identity["device"], report["device"])
        self.assertEqual(identity["shader_stage"], report["samples"][0]["shader_stages"][0])
        self.assertEqual(identity["shader_stage"]["device_shader_content_hash"], self.identity["content_hash"])

    def test_external_prepared_bytes_require_explicit_runtime_none_without_relaxing_identity_or_miss_gates(self):
        self.recipe["shaders"]["compute"].update(spirv_postprocess="none", spirv_postprocess_tool_version="")
        report = self.report()
        report["samples"][0]["effective_spirv_postprocess"] = "none"
        report["samples"][0]["shader_stages"][0]["spirv_postprocess"] = "none"
        values, identity = native.validate_report(report, self.recipe, self.identity, expected_postprocess="none")
        self.assertEqual(values["native_create_ms"], 10.0)
        self.assertEqual(identity["shader_stage"]["spirv_postprocess_tool_version"], "")
        with self.assertRaises(ValueError):
            native.validate_report(report, self.recipe, self.identity)
        mutations = (
            lambda item: item["samples"][0]["shader_stages"][0].update(spirv_postprocess="dead-branches"),
            lambda item: item["samples"][0]["shader_stages"][0].update(device_spirv_fnv1a64="wrong"),
            lambda item: item["samples"][0]["pipeline"].update(native_object_reused=True),
            lambda item: item["samples"][0]["compilation_phases"].update(compiler_executed=True))
        for mutation in mutations:
            changed = copy.deepcopy(report)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_report(changed, self.recipe, self.identity, expected_postprocess="none")

    def test_runtime_postprocess_policy_must_match_recipe_and_reject_unknown_labels(self):
        report = self.report()
        for value in ("external-ssa", "slang-dead-branches", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                native.validate_report(report, self.recipe, self.identity, expected_postprocess=value)
        report["samples"][0]["effective_spirv_postprocess"] = "none"
        report["samples"][0]["shader_stages"][0]["spirv_postprocess"] = "none"
        with self.assertRaisesRegex(ValueError, "Recipe replay postprocess"):
            native.validate_report(report, self.recipe, self.identity, expected_postprocess="none")
        self.recipe["shaders"]["compute"]["spirv_postprocess"] = "none"
        with self.assertRaisesRegex(ValueError, "must not claim a postprocess tool"):
            native.validate_report(report, self.recipe, self.identity, expected_postprocess="none")

    def test_report_rejects_frontend_source_fallback_multiple_calls_and_any_cache_hit(self):
        mutations = (
            lambda report: report.update(status="failed"),
            lambda report: report.update(validation_errors=1),
            lambda report: report["samples"].append(copy.deepcopy(report["samples"][0])),
            lambda report: report["samples"][0].update(pipeline_status="compile_only"),
            lambda report: report["samples"][0].update(shader_source="slang"),
            lambda report: report["samples"][0]["compilation_phases"].update(compiler_executed=True),
            lambda report: report["samples"][0].update(compile_wall_ms=0.1),
            lambda report: report["samples"][0]["pipeline"].update(native_object_reused=True),
            lambda report: report["samples"][0]["pipeline"].update(native_cache_hit=True),
            lambda report: report["samples"][0]["pipeline"].update(metadata_hit=True),
            lambda report: report["samples"][0]["pipeline"].update(feedback_valid=False),
            lambda report: report["configuration"].update(pipeline_cache="reuse"),
            lambda report: report["configuration"].update(workers=2))
        for mutation in mutations:
            report = self.report()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                native.validate_report(report, self.recipe, self.identity)

    def test_report_rejects_requested_cache_override_missing_native_binary_import_and_diagnostics(self):
        mutations = (
            lambda report: report["environment"].clear(),
            lambda report: report["environment"].update({native.driver.CACHE_ENVIRONMENT: "enabled"}),
            lambda report: report["configuration"].update(native_binary_cache_directory="binary-cache"),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_cache_hit=True),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_driver_rejected=True),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_persisted=True),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_import_ms=0.01),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_export_ms=0.01),
            lambda report: report["samples"][0]["pipeline"].update(native_binary_data_bytes=4))
        for name in ("METALLIC_VK_PIPELINE_BINARY", "METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY", "METALLIC_VK_PIPELINE_STATISTICS"):
            mutations += (lambda report, name=name: report["environment"].update({name: "enabled"}),)
        for mutation in mutations:
            report = self.report()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_report(report, self.recipe, self.identity)

    def test_report_rejects_actual_device_validation_shader_bytes_and_abi_mismatches(self):
        mutations = (
            lambda report: report.update(recipe_source_device_matches=False),
            lambda report: report["device"].update(validation_active=True),
            lambda report: report["device"].update(name="another GPU"),
            lambda report: report["device"].update(driver_version=2),
            lambda report: report["device"].update(pipeline_cache_uuid="02" * 16),
            lambda report: report["samples"][0]["shader_stages"][0].update(module="backend-link-general"),
            lambda report: report["samples"][0]["shader_stages"][0].update(source_identity_ambiguous=True),
            lambda report: report["samples"][0]["shader_stages"][0].update(device_spirv_fnv1a64="00" * 8),
            lambda report: report["samples"][0]["shader_stages"][0].update(device_shader_content_hash=int(self.identity["fnv1a64"], 16)),
            lambda report: report["samples"][0].update(device_spirv_bytes=self.identity["bytes"] + 4),
            lambda report: report["samples"][0]["pipeline"].update(recipe_hash_matches=False),
            lambda report: report["samples"][0]["pipeline"].update(recipe_state_overridden=True),
            lambda report: report["samples"][0]["pipeline"].update(pso_hash=0),
            lambda report: report["samples"][0]["pipeline"].update(expected_pso_hash=0))
        for mutation in mutations:
            report = self.report()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_report(report, self.recipe, self.identity)

    def test_native_creation_uses_returned_per_call_metrics_and_rejects_invalid_windows(self):
        for metric, value in (("native_create_ms", 0), ("driver_ms", 0), ("driver_ms", 11),
                ("cache_lock_wait_ms", -1), ("native_create_ms", float("nan")), ("driver_ms", True)):
            report = self.report()
            report["samples"][0]["pipeline"][metric] = value
            report["samples"][0]["pipeline_attempt"][metric] = value
            with self.subTest(metric=metric, value=value), self.assertRaises(ValueError):
                native.validate_report(report, self.recipe, self.identity)
        report = self.report()
        report["samples"][0]["pipeline_attempt"]["native_create_ms"] = 77.0
        with self.assertRaisesRegex(ValueError, "per-call"):
            native.validate_report(report, self.recipe, self.identity)

    def test_native_counters_and_batch_sums_must_prove_exactly_one_actual_miss(self):
        mutations = (
            lambda report: report.update(pipelines_created=2),
            lambda report: report.update(pipelines_created=True),
            lambda report: report.update(pipelines_skipped=1),
            lambda report: report.update(failures=1),
            lambda report: report["pipeline_batches"].clear(),
            lambda report: report["pipeline_batches"][0].update(native_hits=1),
            lambda report: report["pipeline_batches"][0].update(native_misses=0),
            lambda report: report["pipeline_batches"][0].update(unknown_feedback=1),
            lambda report: report["pipeline_batches"][0].update(valid_driver_feedback_count=0),
            lambda report: report["pipeline_batches"][0].update(sample_indices=[1]),
            lambda report: report["pipeline_batches"][0].update(cache_count=2),
            lambda report: report["pipeline_batches"][0].update(driver_duration_sum_ms=88),
            lambda report: report["pipeline_batches"][0].update(successful_native_create_sum_ms=88))
        for mutation in mutations:
            report = self.report()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_report(report, self.recipe, self.identity)

    def records(self, general="original-general"):
        values = ((10, 8), (20, 10), (10, 9))
        records = []
        for pair, order in enumerate(native.schedule(general)):
            for arm in order:
                value = values[pair][0 if arm == general else 1]
                records.append(dict(pair=pair, label=arm, native_create_ms=value, driver_ms=value - 1,
                    cache_lock_wait_ms=0))
        return records

    def test_native_summary_uses_three_paired_ratios_with_explicit_original_denominator(self):
        records = self.records()
        summary = native.aggregate(records)
        self.assertEqual(summary["general_arm"], "original-general")
        self.assertEqual(list(summary["arms"]), ["original-general", "backend-link-no-fiber"])
        self.assertEqual([pair["order"] for pair in summary["paired_ratios"]], [list(pair) for pair in native.schedule()])
        self.assertEqual([pair["native_no_fiber_over_general"] for pair in summary["paired_ratios"]], [0.8, 0.5, 0.9])
        self.assertEqual(summary["median_native_no_fiber_over_general"], 0.8)
        self.assertNotEqual(summary["median_native_no_fiber_over_general"],
            summary["arms"]["backend-link-no-fiber"]["native_create_ms"]["median"] /
            summary["arms"]["original-general"]["native_create_ms"]["median"])
        linked = native.aggregate(self.records("backend-link-general"), "backend-link-general")
        self.assertEqual(linked["general_arm"], "backend-link-general")
        self.assertNotIn("original-general", linked["arms"])

    def test_native_summary_rejects_missing_repeated_wrong_control_nonfinite_or_reordered_samples(self):
        mutations = (
            lambda records: records.pop(),
            lambda records: records.append(copy.deepcopy(records[0])),
            lambda records: records.reverse(),
            lambda records: records[0].update(label="backend-link-general"),
            lambda records: records[2].update(pair=True),
            lambda records: records[0].update(native_create_ms=float("nan")),
            lambda records: records[0].update(cache_lock_wait_ms=-1),
            lambda records: records[0].update(driver_ms=True))
        for mutation in mutations:
            records = self.records()
            mutation(records)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                native.aggregate(records)

    def test_process_identity_requires_fresh_successful_observed_process_and_timezone(self):
        execution = "00000000000040008000000000000000"
        process = dict(execution_id=execution, pid=22, started_utc="2026-10-13T05:00:00+00:00",
            ended_utc="2026-10-13T05:00:01+00:00",
            process_wall_ms=10, exit_code=0, timed_out=False, native_crash=False)
        native.validate_process(process, execution)
        mutations = (
            lambda metadata: metadata.update(execution_id="not-the-process"),
            lambda metadata: metadata.update(pid=True),
            lambda metadata: metadata.update(exit_code=-1073741819),
            lambda metadata: metadata.update(timed_out=True),
            lambda metadata: metadata.update(native_crash=True),
            lambda metadata: metadata.update(started_utc="2026-10-13T05:00:00"),
            lambda metadata: metadata.update(ended_utc="2026-10-13T04:59:00+00:00"),
            lambda metadata: metadata.update(process_wall_ms=float("inf")))
        for mutation in mutations:
            metadata = copy.deepcopy(process)
            mutation(metadata)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native.validate_process(metadata, execution)

    def synthetic_inputs(self, general="original-general"):
        """Stub only preflight; staging, byte identities, run and replay stay real."""
        selected, files, manifests, plans, records, analysis_inputs, analyses = {}, [], [], [], [], {}, {}
        preparation_dir, compile_dir, analysis_dir = (self.root / name for name in ("preparation", "compile", "analysis"))
        for directory in (preparation_dir, compile_dir, analysis_dir):
            directory.mkdir(exist_ok=True)
        arm_directories = []
        for arm, generator in ((general, 1), (native.NO_FIBER_ARM, 2)):
            directory = self.root / arm
            directory.mkdir(exist_ok=True)
            arm_directories.append(directory)
            source = directory / "Cleaned.spv"
            source.write_bytes(code(generator))
            identity = native.blob_identity(source)
            raw_directory = compile_dir / arm
            raw_directory.mkdir()
            raw = raw_directory / "Guides.spv"
            raw.write_bytes(code(generator))
            raw_digest = native.experiment.digest(raw)
            record = dict(label=arm, repetition=0, spirv_sha256=raw_digest)
            records.append(record)
            analysis_inputs[arm] = dict(path=str(raw), sha256=raw_digest, record=copy.deepcopy(record))
            analysis = dict(cleaned=identity, raw=dict(sha256=raw_digest), label=arm)
            analyses[arm] = analysis
            save(directory / "Analysis.json", analysis)
            (directory / "Consumer.slang").write_text(f"module {arm.replace('-', '_')};\n", encoding="utf-8")
            provider = None
            if arm != "original-general":
                name = "PathTraceMaterialBackendLinkNoFiberProvider.slang" if arm == native.NO_FIBER_ARM else "PathTraceMaterialBackendLinkGeneralProvider.slang"
                provider = directory / name
                provider.write_text("module FixtureProvider;\n", encoding="utf-8")
            plan = dict(label=arm, provider=str(provider) if provider else None,
                mechanism="original" if arm == "original-general" else "backend-link")
            plans.append(plan)
            files.extend(path for path in directory.iterdir() if path.is_file())
            files.append(raw)
            selected[arm] = dict(source=str(source), identity=identity, plan=plan, provider=plan["provider"],
                provider_sha256=native.experiment.digest(provider) if provider else None, analysis=analysis,
                artifact_manifest=dict(artifact_id=arm, requires_no_fiber=arm == native.NO_FIBER_ARM,
                    artifact=dict(file=str(source), source_tag=f"Artifacts/PathBackend/{arm}/TwoPassCleaned",
                        entry_main="main", spirv_postprocess="dead-branches", tool_version="SPIRV-Tools fixture", **identity)))
        save(preparation_dir / "Preparation.json", dict(status="prepared", plans=plans))
        save(compile_dir / "summary.json", dict(status="passed", runs=records))
        save(analysis_dir / "Inputs.json", dict(inputs=analysis_inputs))
        save(analysis_dir / "summary.json", dict(status="passed", cleanup_passes=native.PASSES, outputs=analyses))
        files.extend([preparation_dir / "Preparation.json", compile_dir / "summary.json",
            analysis_dir / "Inputs.json", analysis_dir / "summary.json"])
        for directory in [preparation_dir, compile_dir, analysis_dir, *arm_directories]:
            manifest = directory / "Manifest.json"
            save(manifest, dict(schema_version=1, artifacts=[dict(path=path.relative_to(directory).as_posix(),
                sha256=native.experiment.digest(path)) for path in sorted(directory.rglob("*")) if path.is_file()]))
            manifests.append(dict(file=str(manifest), sha256=native.experiment.digest(manifest)))
            files.append(manifest)
        template_path = self.root / "Captured.recipe.json"
        save(template_path, self.recipe)
        executable = self.root / "bench.exe"
        executable.write_bytes(b"mocked benchmark executable")
        files.extend([template_path, self.artifact, executable])
        return dict(schema_version=1, general_arm=general, schedule=[list(row) for row in native.schedule(general)],
            template=str(template_path), template_sha256=native.experiment.digest(template_path),
            template_document=copy.deepcopy(self.recipe), template_original_file=str(self.artifact),
            template_original=copy.deepcopy(self.identity), template_pso_hash_validated=True,
            preparation=str(preparation_dir), compile=str(compile_dir), analysis=str(analysis_dir),
            source_manifests=manifests, postprocess="dead-branches", postprocess_tool_version="SPIRV-Tools fixture",
            inputs=selected, frozen_files=native.experiment.snapshots(files), executable=str(executable))

    def run_mocked(self, *, general="original-general", fail_at=None, timeout=False, exit_code=0, mutation=None):
        inputs = self.synthetic_inputs(general)
        args = SimpleNamespace(output_dir=self.root / "measured", executable=Path(inputs["executable"]),
            timeout=1, general_arm=general)
        processes = []
        testcase = self
        values = ((10, 8), (20, 10), (10, 9))

        class Process:
            def __init__(self, command, **kwargs):
                self.command, self.environment = command, kwargs["env"]
                self.index, self.pid, self.waits, self.killed = len(processes), 700 + len(processes), 0, False
                processes.append(self)
                destination = Path(command[command.index("--output-dir") + 1])
                recipe_path = Path(command[command.index("--recipe-file") + 1])
                recipe = native.experiment.read(recipe_path)
                raw_path = recipe_path.parent / recipe["shaders"]["compute"]["spirv_file"]
                old_recipe, old_identity = testcase.recipe, testcase.identity
                testcase.recipe, testcase.identity = recipe, native.blob_identity(raw_path)
                try:
                    arm = recipe_path.name.removesuffix(".recipe.json")
                    value = values[self.index // 2][0 if arm == general else 1]
                    report = testcase.report(value, value - 1, 0)
                finally:
                    testcase.recipe, testcase.identity = old_recipe, old_identity
                report["configuration"].update(recipe_file=str(recipe_path), cache_directory=str(destination / "cache"))
                if mutation and (fail_at is None or self.index == fail_at):
                    mutation(report, destination)
                save(destination / "report.json", report)

            def wait(self, **kwargs):
                self.waits += 1
                if timeout and self.index == fail_at and self.waits == 1:
                    raise subprocess.TimeoutExpired(self.command, 1)
                return exit_code if self.index == fail_at else 0

            def kill(self):
                self.killed = True

        with patch.object(native, "preflight", return_value=inputs), \
             patch.object(native.frontend, "experiment_lock", side_effect=nullcontext), \
             patch.object(native.driver, "suppress_windows_error_dialogs", side_effect=nullcontext), \
             patch.object(native.driver.subprocess, "Popen", side_effect=Process), \
             patch.object(native, "verify", side_effect=lambda output: native.verify_records(output)), \
             redirect_stdout(io.StringIO()):
            try:
                summary = native.run(args)
            except Exception as error:
                return args.output_dir, processes, None, error
        return args.output_dir, processes, summary, None

    def test_full_mocked_run_stages_selected_original_and_six_independent_native_misses(self):
        output, processes, summary, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        self.assertEqual(len(processes), 6)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(len({row["execution_id"] for row in summary["runs"]}), 6)
        self.assertEqual(summary["aggregation"]["median_native_no_fiber_over_general"], 0.8)
        expected = [arm for order in native.schedule() for arm in order]
        self.assertEqual([row["label"] for row in summary["runs"]], expected)
        for process, arm in zip(processes, expected):
            recipe = Path(process.command[process.command.index("--recipe-file") + 1])
            self.assertEqual(recipe.name, f"{arm}.recipe.json")
            self.assertEqual(process.environment[native.driver.CACHE_ENVIRONMENT], "disabled")
            self.assertNotIn("METALLIC_SPIRV_POSTPROCESS", process.environment)
            self.assertNotIn("METALLIC_SLANG_OPTIMIZATION", process.environment)
        native.validate_seal(output)

    def test_frozen_native_replay_ignores_later_source_builds_and_live_check_rejects_them(self):
        output, _, summary, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        (self.root / "original-general/Consumer.slang").write_text("module Changed;", encoding="utf-8")
        (self.root / "bench.exe").write_bytes(b"later benchmark build")
        self.assertEqual(native.verify_records(output), summary)
        with self.assertRaisesRegex(ValueError, "input changed"):
            native.verify_records(output, current_inputs=True)
        with patch.object(native.subprocess, "run", return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps(summary), stderr="")) as replay:
            self.assertEqual(native.verify(output), summary)
        self.assertEqual(replay.call_args.args[0], [sys.executable, "-B",
            str(output / "Verifier/Tools/Perf/MeasurePathTraceBackendNative.py"), "_verify-frozen", str(output)])

    def test_declared_original_control_cannot_bind_a_linked_artifact_or_provider_plan(self):
        output, _, _, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        original = native.experiment.read(output / "Inputs.json")
        native.validate_arm_provenance(output, original, "original-general")
        linked = original["inputs"][native.NO_FIBER_ARM]
        mutations = (
            lambda item: item.update(artifact_manifest=copy.deepcopy(linked["artifact_manifest"])),
            lambda item: item["artifact_manifest"]["artifact"].update(source_tag=linked["artifact_manifest"]["artifact"]["source_tag"]),
            lambda item: item.update(plan=copy.deepcopy(linked["plan"]), provider=linked["provider"],
                provider_sha256=linked["provider_sha256"]))
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed["inputs"]["original-general"])
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "(?i)(arm|plan|provider)"):
                native.validate_arm_provenance(output, changed, "original-general")

    def test_native_replay_rejects_resealed_control_denominator_raw_metrics_and_process_overlap(self):
        output, _, original, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        for mutation in (
                lambda summary: summary.update(general_arm="backend-link-general"),
                lambda summary: summary["runs"][0].update(label="backend-link-general"),
                lambda summary: summary["runs"][0].update(native_create_ms=999),
                lambda summary: summary["aggregation"].update(median_native_no_fiber_over_general=0.1),
                lambda summary: summary["runs"][1].update(execution_id=summary["runs"][0]["execution_id"])):
            modified = copy.deepcopy(original)
            mutation(modified)
            save(output / "summary.json", modified)
            native.experiment.seal(output)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                native.verify_records(output)
        save(output / "summary.json", original)
        first = native.experiment.read(output / original["runs"][0]["directory"] / "process.json")
        path = output / original["runs"][1]["directory"] / "process.json"
        process = native.experiment.read(path)
        process["started_utc"] = first["started_utc"]
        save(path, process)
        native.experiment.seal(output)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            native.verify_records(output)

    def test_crash_timeout_and_unqualified_native_report_seal_diagnostics_without_sample_admission(self):
        for index, (timed_out, code_value, mutation) in enumerate((
                (True, 1, None), (False, -1073741819, None),
                (False, 0, lambda report, directory: report["samples"][0]["pipeline"].update(native_cache_hit=True)))):
            with self.subTest(timed_out=timed_out, code=code_value):
                original_root = self.root
                self.root = original_root / f"failed-{index}"
                self.root.mkdir()
                output, processes, summary, error = self.run_mocked(fail_at=2, timeout=timed_out,
                    exit_code=code_value, mutation=mutation)
                self.root = original_root
                self.assertIsInstance(error, (ValueError, RuntimeError))
                self.assertIsNone(summary)
                self.assertEqual(len(processes), 3)
                self.assertEqual(processes[-1].killed, timed_out)
                failed = native.experiment.read(output / "summary.json")
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(len(failed["runs"]), 2)
                directory = output / "pair-01-00-backend-link-no-fiber"
                metadata = native.experiment.read(directory / "process.json")
                self.assertEqual(metadata["exit_code"], code_value)
                self.assertIs(metadata["timed_out"], timed_out)
                self.assertTrue((directory / "report.json").is_file())
                native.validate_seal(output)


if __name__ == "__main__":
    unittest.main()
