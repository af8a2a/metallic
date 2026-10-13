"""Qualify paired path-trace dispatch evidence without Slang or GPU execution."""

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
sys.path.insert(0, str(ROOT / "tests/perf"))
import MeasurePathTraceBackendDispatch as dispatch
import TestPathTraceBackendRendering as fixture


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class BackendDispatchTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the synthetic files and report builders, never that suite's
        # tests or an executable. Each test owns an independent temporary tree.
        self.fixture = fixture.ArtifactRenderingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.output = self.fixture.output
        self.args = self.fixture.args
        self.args.pairs = 3
        self.args.width = 512
        self.args.resolution = 512
        self.args.warmup_frames = 16
        self.args.timing_frames = 32

    def workload_inputs(self):
        original = self.fixture.workload_inputs()
        original.update(frames=16, warmupFrames=4, timingFrames=4)
        for case in original["cases"]:
            case.update(frameCount=16, captureFrames=list(range(8, 16)),
                requiredTiming=["PathTrace", "Path trace shading"],
                cameraFrames=[dict(frame=11, node="PathTrace", camera=dict(eye=[0.18, 0.3, 4.5]))],
                captureOutputs=["PathTrace.albedo", "PathTrace.specularAlbedo", "PathTrace.normalRoughness",
                    "PathTrace.motionVectors", "PathTrace.linearDepth", "PathTrace.specularHitDistance", "PathTrace.depth"],
                guideCameraStep=dict(node="PathTrace", delta=[0.02, 0, 0]))
        save(self.args.cases, original)
        return original

    def test_schedule_has_validation_pair_then_fresh_balanced_timing_pairs(self):
        rows = dispatch.schedule(3)
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["phase"] for row in rows], ["correctness"] * 2 + ["timing"] * 6)
        self.assertEqual([row["validation"] for row in rows], [True] * 2 + [False] * 6)
        self.assertEqual([row["arm"] for row in rows[:2]], list(dispatch.ARMS))
        self.assertEqual([row["arm"] for row in rows[2:]],
            [dispatch.ARMS[0], dispatch.ARMS[1], dispatch.ARMS[1], dispatch.ARMS[0], dispatch.ARMS[0], dispatch.ARMS[1]])
        self.assertEqual([row["block"] for row in rows[2:]], [0, 0, 1, 1, 2, 2])
        self.assertEqual(len({row["label"] for row in rows}), len(rows))

    def test_invalid_pair_counts_reject_instead_of_empty_qualification(self):
        for value in (0, -1, True, 1.0, float("nan")):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                dispatch.schedule(value)

    def test_workload_uses_same_512_windows_with_camera_and_readbacks_after_timing(self):
        original = self.workload_inputs()
        with patch.object(dispatch.rendering, "ROOT", self.root):
            path, configuration, inputs, absent = dispatch.prepare_dispatch_workload(self.args.cases, self.output)
        self.assertEqual(configuration["warmupFrames"], 16)
        self.assertEqual(configuration["timingFrames"], 32)
        self.assertGreater(configuration["frames"], 48)
        self.assertEqual([case["id"] for case in configuration["cases"]], ["Opaque1", "Mixed1", "Sky"])
        for case in configuration["cases"]:
            self.assertEqual((case["width"], case["height"]), (512, 512))
            self.assertGreater(case["frameCount"], 48)
            self.assertTrue(case["cameraFrames"])
            self.assertTrue(case["captureFrames"])
            self.assertTrue(case["captureOutputs"])
            self.assertTrue(all(frame >= 48 for frame in case["captureFrames"]))
            self.assertTrue(all(item["frame"] >= 48 for item in case["cameraFrames"]))
            self.assertEqual(case["requiredTiming"], ["PathTrace", "Path trace shading"])
            properties = dispatch.experiment.read(Path(case["graph"]))["nodes"][0]["properties"]
            self.assertEqual((properties["samples"], properties["maxDepth"], properties["cacheMode"]), (2, 12, "off"))
        self.assertEqual(dispatch.experiment.read(self.args.cases), original)
        self.assertIn(path, inputs)
        self.assertEqual(len(absent), 3)
        dispatch.rendering.check_absences(absent)

    def test_dispatch_workload_rejects_invalid_windows_and_dimensions(self):
        self.workload_inputs()
        invalid = [(dict(width=0)), dict(width=True), dict(warmup_frames=-1),
            dict(warmup_frames=True), dict(timing_frames=0), dict(timing_frames=1.5)]
        for index, options in enumerate(invalid):
            destination = self.root / f"invalid-workload-{index}"
            destination.mkdir()
            with self.subTest(options=options), patch.object(dispatch.rendering, "ROOT", self.root), self.assertRaises((ValueError, TypeError)):
                dispatch.prepare_dispatch_workload(self.args.cases, destination, **options)

    def test_implicit_sidecar_and_shader_source_drift_reject_before_measurement(self):
        self.workload_inputs()
        with patch.object(dispatch.rendering, "ROOT", self.root):
            _, _, inputs, absent = dispatch.prepare_dispatch_workload(self.args.cases, self.output)
        identity = dispatch.experiment.snapshots(inputs)
        dispatch.experiment.check_snapshots(identity)
        Path(absent[0]).write_bytes(b"uninspected scene materials")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            dispatch.rendering.check_absences(absent)
        target = self.root / "shader-source.slang"
        target.write_text("module Fixture;", encoding="utf-8")
        frozen = dispatch.experiment.snapshots([target])
        target.write_text("module Fixture; // changed during batch", encoding="utf-8")
        with self.assertRaises(ValueError):
            dispatch.experiment.check_snapshots(frozen)

    def timing_configuration(self):
        configuration = copy.deepcopy(self.fixture.config)
        configuration.update(frames=56, warmupFrames=16, timingFrames=32)
        for case in configuration["cases"]:
            case.update(width=512, height=512, frameCount=56, captureFrames=list(range(48, 56)),
                requiredTiming=["PathTrace", "Path trace shading"],
                cameraFrames=[dict(frame=51, node="PathTrace", camera=dict(eye=[0.18, 0.3, 4.5]))],
                captureOutputs=["PathTrace.albedo", "PathTrace.specularAlbedo", "PathTrace.normalRoughness",
                    "PathTrace.motionVectors", "PathTrace.linearDepth", "PathTrace.specularHitDistance", "PathTrace.depth"],
                guideCameraStep=dict(node="PathTrace", delta=[0.02, 0, 0]))
        return configuration

    def timing_report(self, configuration, scale=1.0):
        cases = []
        for specification in configuration["cases"]:
            case = copy.deepcopy(specification)
            case.update(validHDR=True, format="RGBA32F", image=f"{case['id']}-color.bin",
                resolvedGraph=dict(nodes=[]), captures=[dict(output=case["output"], frame=frame,
                    file=f"{case['id']}-{frame}.bin", bytes=16, format=42) for frame in case["captureFrames"]] +
                    [dict(output=output, frame=case["frameCount"] + index,
                        file=f"{case['id']}-guide-{index}.bin", bytes=16, format=42)
                     for index, output in enumerate(case["captureOutputs"])])
            case["frames"] = [dict(frame=frame, executionId=frame + 1, warmup=False,
                asyncComputeBranches=0, graphMs=scale * 4,
                nodes=[dict(name="PathTrace", queue=0, gpuMs=scale * 2,
                    sections=[dict(name="Path trace shading", gpuMs=scale)])])
                for frame in range(configuration["warmupFrames"], configuration["warmupFrames"] + configuration["timingFrames"])]
            cases.append(case)
        return dict(status="passed", validation=False, cases=cases)

    def timing_runs(self, configuration, general_arm=None):
        runs = []
        values = ((10, 8), (20, 10), (10, 9))
        arm_pair = dispatch.ARMS if general_arm is None else dispatch.selected_arms(general_arm)
        rows = dispatch.schedule(3) if general_arm is None else dispatch.schedule(3, general_arm)
        for index, row in enumerate(rows[2:]):
            run = copy.deepcopy(row)
            execution_id = f"00000000000040008000{index:012d}"
            scale = values[run["block"]][arm_pair.index(run["arm"])]
            run.update(execution_id=execution_id,
                process=dict(execution_id=execution_id, pid=100 + index, started=10.0 * index + 1,
                    ended=10.0 * index + 2, exit_code=0, timed_out=False, native_crash=False),
                timings=dispatch.experiment.gpu_timings(self.timing_report(configuration, scale), configuration))
            runs.append(run)
        return runs

    def test_explicit_original_control_keeps_historical_default_and_rejects_unqualified_arms(self):
        self.assertEqual(dispatch.ARMS, ("backend-link-general", "backend-link-no-fiber"))
        self.assertEqual(dispatch.selected_arms(), dispatch.ARMS)
        self.assertEqual(dispatch.selected_arms("original-general"), ("original-general", "backend-link-no-fiber"))
        self.assertIsInstance(dispatch.selected_arms("original-general"), tuple)
        for arm in ("backend-concrete-general", "backend-link-no-fiber", "original", "", None, True, ["original-general"]):
            with self.subTest(arm=arm), self.assertRaises((ValueError, TypeError)):
                dispatch.selected_arms(arm)

    def test_original_control_schedule_is_fresh_ab_ba_ab_and_validation_excluded(self):
        rows = dispatch.schedule(3, "original-general")
        expected = ["original-general", "backend-link-no-fiber", "original-general", "backend-link-no-fiber",
            "backend-link-no-fiber", "original-general", "original-general", "backend-link-no-fiber"]
        self.assertEqual([row["arm"] for row in rows], expected)
        self.assertEqual([row["validation"] for row in rows], [True, True] + [False] * 6)
        self.assertEqual([row["phase"] for row in rows], ["correctness"] * 2 + ["timing"] * 6)
        self.assertEqual([row["block"] for row in rows[2:]], [0, 0, 1, 1, 2, 2])
        self.assertEqual(len({row["label"] for row in rows}), 8)
        self.assertTrue(all("backend-link-general" not in row["label"] for row in rows))

    def test_original_control_statistics_bind_paired_denominator_and_process_roles(self):
        configuration = self.timing_configuration()
        rows = self.timing_runs(configuration, "original-general")
        result = dispatch.summarize_pairs(rows, configuration, 3, "original-general")
        self.assertEqual(result["arm_pair"], ["original-general", "backend-link-no-fiber"])
        self.assertEqual(result["arm_roles"], {"control": "original-general", "candidate": "backend-link-no-fiber"})
        self.assertNotIn("backend-link-general", result["per_arm"])
        for case in configuration["cases"]:
            for scope in dispatch.SCOPES:
                metric = result["paired"][case["id"]][scope]
                self.assertEqual(metric["ratios"], [0.8, 0.5, 0.9])
                self.assertAlmostEqual(metric["percent_change"], -20)
                self.assertEqual(metric["percent_change_denominator_arm"], "original-general")
                self.assertEqual(metric["percent_change_definition"],
                    "100 * (median(candidate_process_median_ms / control_process_median_ms) - 1)")
                control = result["per_arm"]["original-general"][case["id"]][scope]
                candidate = result["per_arm"]["backend-link-no-fiber"][case["id"]][scope]
                self.assertNotEqual(metric["ratio_statistics"]["median"], candidate["median"] / control["median"])
        with self.assertRaises(ValueError):
            dispatch.summarize_pairs(rows, configuration, 3, "backend-link-general")

    def test_original_artifact_must_match_its_frozen_compiler_control_not_only_a_valid_blob(self):
        manifests, _ = self.fixture.prepared()
        original = dispatch.experiment.read(manifests["original-general"])
        linked = dispatch.experiment.read(manifests["backend-link-general"])
        dispatch.validate_selected_artifact(original, "original-general")
        mutations = (
            lambda manifest: manifest.update(artifact_id="backend-link-general"),
            lambda manifest: manifest.update(requires_no_fiber=True),
            lambda manifest: manifest.update(artifact=copy.deepcopy(linked["artifact"])),
            lambda manifest: manifest["artifact"].update(source_tag=linked["artifact"]["source_tag"]),
            lambda manifest: manifest["reference_request"].update(entry="anotherEntry"),
            lambda manifest: manifest["reference_settings"].update(optimization="O2"),
            lambda manifest: manifest["provenance"]["manifests"].pop())
        for mutation in mutations:
            manifest = copy.deepcopy(original)
            mutation(manifest)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                dispatch.validate_selected_artifact(manifest, "original-general")

    def test_original_control_requires_actual_raw_reference_equivalence(self):
        manifests, _ = self.fixture.prepared()
        original = dispatch.experiment.read(manifests["original-general"])
        dispatch.validate_selected_artifact(original, "original-general")
        reference = self.args.preparation / "Reference.spv"
        reference.write_bytes(fixture.code(999))
        fixture.seal(self.args.preparation)
        # Re-sealing inventories cannot turn unrelated raw code into the measured
        # original consumer. Even matching new manifest hashes must not admit it.
        preparation_manifest = self.args.preparation / "Manifest.json"
        analysis_inputs = dispatch.experiment.read(self.args.analysis / "Inputs.json")
        analysis_inputs["manifests"]["preparation"]["sha256"] = dispatch.experiment.digest(preparation_manifest)
        save(self.args.analysis / "Inputs.json", analysis_inputs)
        compiler_summary = dispatch.experiment.read(self.args.compile / "summary.json")
        compiler_summary["preparation_manifest_sha256"] = dispatch.experiment.digest(preparation_manifest)
        save(self.args.compile / "summary.json", compiler_summary)
        fixture.seal(self.args.compile)
        analysis_inputs["manifests"]["compile"]["sha256"] = dispatch.experiment.digest(self.args.compile / "Manifest.json")
        save(self.args.analysis / "Inputs.json", analysis_inputs)
        fixture.seal(self.args.analysis)
        for row in original["provenance"]["manifests"]:
            row["sha256"] = dispatch.experiment.digest(Path(row["file"]))
        with self.assertRaisesRegex(ValueError, "(?i)(raw|reference|original)"):
            dispatch.validate_selected_artifact(original, "original-general")


    def test_aggregate_uses_paired_process_medians_instead_of_pooled_frames(self):
        configuration = self.timing_configuration()
        result = dispatch.summarize_pairs(self.timing_runs(configuration), configuration, pairs=3)
        self.assertIn("process", result["statistical_unit"].lower())
        self.assertIn("frames", result["statistical_unit"].lower())
        for case in configuration["cases"]:
            for scope, factor in (("graph_ms", 4), ("pass_ms", 2), ("dispatch_scope_ms", 1)):
                general = result["per_arm"][dispatch.ARMS[0]][case["id"]][scope]
                candidate = result["per_arm"][dispatch.ARMS[1]][case["id"]][scope]
                self.assertEqual(general, dispatch.driver.timings([10 * factor, 20 * factor, 10 * factor]))
                self.assertEqual(candidate, dispatch.driver.timings([8 * factor, 10 * factor, 9 * factor]))
                paired = result["paired"][case["id"]][scope]
                self.assertEqual(paired["ratios"], [0.8, 0.5, 0.9])
                self.assertEqual(paired["ratio_statistics"], dispatch.driver.timings([0.8, 0.5, 0.9]))
                self.assertAlmostEqual(paired["percent_change"], -20)
                # Independent medians' ratio is 0.9; pairing gives 0.8.
                self.assertNotEqual(paired["ratio_statistics"]["median"], candidate["median"] / general["median"])

    def test_aggregate_rejects_missing_duplicate_reordered_or_failed_processes(self):
        configuration = self.timing_configuration()
        good = self.timing_runs(configuration)
        mutations = [lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows.reverse(), lambda rows: rows[1].update(execution_id=rows[0]["execution_id"]),
            lambda rows: rows[0]["process"].update(execution_id="unrelated-process"),
            lambda rows: rows[0]["process"].update(exit_code=1),
            lambda rows: rows[0]["process"].update(timed_out=True),
            lambda rows: rows[0]["process"].update(ended=rows[0]["process"]["started"] - 1),
            lambda rows: rows[0].update(validation=True),
            lambda rows: rows[0].update(arm="backend-concrete-general")]
        for mutation in mutations:
            rows = copy.deepcopy(good)
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.summarize_pairs(rows, configuration, pairs=3)

    def test_aggregate_rejects_wrong_case_scope_framecount_and_nonfinite_timestamps(self):
        configuration = self.timing_configuration()
        good = self.timing_runs(configuration)
        case = configuration["cases"][0]["id"]
        mutations = [lambda rows: rows[0]["timings"].pop(case),
            lambda rows: rows[0]["timings"][case].pop("dispatch_scope_ms"),
            lambda rows: rows[0]["timings"][case]["dispatch_scope_ms"].update(count=31),
            lambda rows: rows[0]["timings"][case]["dispatch_scope_ms"].update(median=float("nan")),
            lambda rows: rows[0]["timings"][case]["dispatch_scope_ms"].update(min=0),
            lambda rows: rows[0]["timings"][case]["dispatch_scope_ms"].update(p95=float("inf"))]
        for mutation in mutations:
            rows = copy.deepcopy(good)
            mutation(rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.summarize_pairs(rows, configuration, pairs=3)

    def test_real_timing_parser_rejects_incomplete_rows_ids_and_wrong_scope(self):
        configuration = self.timing_configuration()
        good = self.timing_report(configuration)
        mutations = [lambda report: report["cases"][0]["frames"].pop(),
            lambda report: report["cases"][0]["frames"][1].update(executionId=17),
            lambda report: report["cases"][0]["frames"][0].update(warmup=True),
            lambda report: report["cases"][0]["frames"][0].update(frame=0),
            lambda report: report["cases"][0]["frames"][0].update(asyncComputeBranches=1),
            lambda report: report["cases"][0]["frames"][0]["nodes"][0].update(queue=1),
            lambda report: report["cases"][0]["frames"][0]["nodes"][0]["sections"][0].update(name="unrelated scope"),
            lambda report: report["cases"][0]["frames"][0].update(graphMs=float("nan")),
            lambda report: report["cases"][0]["frames"][0]["nodes"][0].update(gpuMs=0),
            lambda report: report["cases"][0].update(validHDR=False)]
        for mutation in mutations:
            report = copy.deepcopy(good)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.experiment.gpu_timings(report, configuration)

    def runtime_report(self, directory, manifest, configuration, *, validation):
        self.fixture.config = configuration
        report, _, _ = self.fixture.runtime(directory, manifest)
        report["validation"] = validation
        report["cases"] = self.timing_report(configuration)["cases"]
        for path in (directory / "recipes").rglob("*.recipe.json"):
            recipe = dispatch.experiment.read(path)
            recipe["source_device"]["validation_enabled"] = validation
            recipe["source_device"]["synchronization_validation_enabled"] = False
            save(path, recipe)
            ordinary = copy.deepcopy(recipe)
            raw_path = path.parent / "Ordinary.spv"
            raw_path.write_bytes(fixture.code(88))
            identity = dispatch.rendering.blob_identity(raw_path)
            ordinary["shaders"]["compute"].update(module_name="Features/UnchangedRuntimeShader", source_entry_point="unchangedMain",
                spirv_file=raw_path.name, input_spirv_hash=identity["fnv1a64"], spirv_bytes=identity["bytes"],
                spirv_postprocess="none", spirv_postprocess_tool_version="")
            ordinary["expected_pso_hash"] = dispatch.rendering.pso_hash(ordinary, raw_path.read_bytes())
            save(path.parent / "Ordinary.recipe.json", ordinary)
        save(directory / "MaterialBaseline.json", report)
        return report

    def test_runtime_off_requires_actual_off_device_recipes_and_per_case_artifact_proof(self):
        manifests, _ = self.fixture.prepared()
        manifest = dispatch.experiment.read(manifests[dispatch.ARMS[0]])
        directory = self.root / "runtime-off"
        configuration = self.timing_configuration()
        good = self.runtime_report(directory, manifest, configuration, validation=False)
        evidence = dispatch.rendering.verify_runtime(directory, manifest, configuration, expected_validation=False)
        self.assertEqual(len(evidence["recipes"]), 3)
        self.assertFalse(evidence["devices"][0]["validation_enabled"])
        for mutation in (lambda report: report.update(validation=True),
                lambda report: report["artifact_cases"][0].update(consumed=0),
                lambda report: report["artifact_cases"][0].update(actual_pipeline_recipes=[]),
                lambda report: report["cases"][0].update(width=128),
                lambda report: report["cases"][0]["cameraFrames"][0].update(frame=11)):
            report = copy.deepcopy(good)
            mutation(report)
            save(directory / "MaterialBaseline.json", report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.rendering.verify_runtime(directory, manifest, configuration, expected_validation=False)
        save(directory / "MaterialBaseline.json", good)
        recipe_path = next((directory / "recipes").rglob("*.recipe.json"))
        recipe = dispatch.experiment.read(recipe_path)
        recipe["source_device"]["validation_enabled"] = True
        save(recipe_path, recipe)
        with self.assertRaisesRegex(ValueError, "Recipe validation"):
            dispatch.rendering.verify_runtime(directory, manifest, configuration, expected_validation=False)

    def test_validation_on_pair_cannot_share_only_asserted_device_identity(self):
        manifests, _ = self.fixture.prepared()
        manifest = dispatch.experiment.read(manifests[dispatch.ARMS[0]])
        directory = self.root / "runtime-on"
        configuration = self.timing_configuration()
        self.runtime_report(directory, manifest, configuration, validation=True)
        evidence = dispatch.rendering.verify_runtime(directory, manifest, configuration)
        self.assertEqual(len(evidence["devices"]), 1)
        self.assertTrue(evidence["devices"][0]["validation_enabled"])
        (directory / "recipes" / configuration["cases"][0]["id"] / "Consumed.spv").write_bytes(fixture.code(99))
        with self.assertRaisesRegex(ValueError, "Device artifact differs"):
            dispatch.rendering.verify_runtime(directory, manifest, configuration)

    def test_stronger_timing_gate_rejects_invalid_execution_ids_and_nested_scope_durations(self):
        configuration = self.timing_configuration()
        good = self.timing_report(configuration)
        self.assertEqual(dispatch.verify_timing_rows(good, configuration), dispatch.experiment.gpu_timings(good, configuration))
        for mutation in (
                lambda report: report["cases"][0]["frames"][0].update(executionId=True),
                lambda report: report["cases"][0]["frames"][0].update(executionId=0),
                lambda report: report["cases"][0]["frames"][0].update(executionId=-1),
                lambda report: report["cases"][0]["frames"][0].update(graphMs=1),
                lambda report: report["cases"][0]["frames"][0]["nodes"][0]["sections"][0].update(gpuMs=3)):
            report = copy.deepcopy(good)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.verify_timing_rows(report, configuration)

    def test_device_identity_removes_only_validation_toggle(self):
        device = dict(device_name="GPU", driver_version=7, validation_enabled=True,
            synchronization_validation_enabled=False, cache_uuid="fixture", debug_markers=False)
        timing = dict(device, validation_enabled=False)
        self.assertEqual(dispatch.device_identity(dict(devices=[device])), dispatch.device_identity(dict(devices=[timing])))
        self.assertEqual(device["validation_enabled"], True)
        for field, value in (("driver_version", 8), ("cache_uuid", "another"), ("debug_markers", True)):
            changed = dict(timing, **{field: value})
            self.assertNotEqual(dispatch.device_identity(dict(devices=[device])), dispatch.device_identity(dict(devices=[changed])))
        for devices in ([], [device, timing], [dict(device, validation_enabled=1)],
                [dict(device, synchronization_validation_enabled=True)]):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                dispatch.device_identity(dict(devices=devices))

    def run_mocked(self, *, fail_at=None, timeout=False, exit_code=0, drift=False, qualified=True, mutation=None):
        processes = []
        configuration = self.timing_configuration()
        for case in configuration["cases"]:
            graph_path = self.output / f"{case['id']}.json"
            save(graph_path, dict(nodes=[dict(type="ScenePathTracePass", properties=dict(samples=2, maxDepth=12,
                cacheMode="off", bsdf="openpbr", exportDenoiserGuides=True, path=case["scene"]))]))
            case["graph"] = str(graph_path)
        save(self.args.cases, configuration)
        testcase = self
        comparison = [dict(case=case["id"], qualified=qualified,
            readbacks=[dict(output="final HDR", numerically_exact=qualified,
                max_absolute_error=0 if qualified else 1, within_tolerance=qualified)]) for case in configuration["cases"]]

        class Process:
            def __init__(self, command, **kwargs):
                self.pid = 900 + len(processes)
                self.command, self.environment = command, kwargs["env"]
                self.killed, self.waits = False, 0
                self.index = len(processes)
                processes.append(self)
                destination = Path(command[command.index("--output-dir") + 1])
                manifest = dispatch.experiment.read(self.environment["METALLIC_SHADER_ARTIFACT_MANIFEST"])
                report = testcase.runtime_report(destination, manifest, configuration,
                    validation="--rhi-validation" in command)
                if mutation and (fail_at is None or self.index == fail_at):
                    mutation(report, destination)
                    save(destination / "MaterialBaseline.json", report)
                if drift and (fail_at is None or self.index == fail_at):
                    testcase.args.executable.write_bytes(b"changed during process")

            def wait(self, **kwargs):
                self.waits += 1
                if timeout and self.index == fail_at and self.waits == 1:
                    raise subprocess.TimeoutExpired(self.command, 1)
                return exit_code if self.index == fail_at else 0

            def kill(self):
                self.killed = True

        summary = dict(schema_version=1, status="running", runs=[])
        with patch.object(dispatch, "prepare_dispatch_workload", return_value=(self.args.cases, configuration, [self.args.cases], [])), \
             patch.object(dispatch.experiment, "input_identity", side_effect=lambda executable, extra=(), **kwargs:
                          dispatch.experiment.snapshots([executable, self.args.cases, *extra,
                              *[Path(case["graph"]) for case in configuration["cases"]]])), \
             patch.object(dispatch.experiment, "gpu_inventory", return_value=[dict(name="Fixture GPU", driver_version="1")]), \
             patch.object(dispatch.driver, "suppress_windows_error_dialogs", side_effect=nullcontext), \
             patch.object(dispatch.subprocess, "Popen", side_effect=Process), \
             patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison) as comparisons, \
             redirect_stdout(io.StringIO()):
            try:
                dispatch.run(self.args, self.output, summary)
            except Exception as error:
                return processes, comparisons, summary, error
        return processes, comparisons, summary, None

    def test_mocked_run_requires_correctness_pair_then_six_distinct_timing_processes(self):
        processes, comparisons, summary, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        self.assertEqual(len(processes), 8)
        self.assertEqual(len({run["execution_id"] for run in summary["runs"]}), 8)
        self.assertEqual(len({process.pid for process in processes}), 8)
        self.assertEqual(comparisons.call_count, 8)
        self.assertTrue(summary["correctness_qualified"])
        self.assertEqual(summary["timing_summary"], dispatch.summarize_pairs(summary["runs"], summary["workload"], 3))
        for process, row in zip(processes, dispatch.schedule(3)):
            self.assertIn("--rhi-validation" if row["validation"] else "--rhi-no-validation", process.command)
            self.assertEqual(process.environment["METALLIC_SPIRV_POSTPROCESS"], "none")
            self.assertEqual(process.environment["METALLIC_WORKING_COLOR_SPACE"], "acescg")
            self.assertEqual(Path(process.environment["METALLIC_SHADER_ARTIFACT_MANIFEST"]).name, f"{row['arm']}.artifact.json")
            self.assertEqual(Path(process.environment["METALLIC_MATERIAL_BASELINE_CASES"]), self.args.cases)
        for comparison in comparisons.call_args_list:
            self.assertEqual(comparison.kwargs, dispatch.TOLERANCE)
            self.assertEqual(comparison.args[2], summary["workload"])
        self.assertTrue(all("timings" not in run for run in summary["runs"][:2]))
        self.assertTrue(all("timings" in run for run in summary["runs"][2:]))

    def test_mocked_original_control_uses_original_bytes_and_actual_manifest_in_every_process(self):
        self.args.general_arm = "original-general"
        processes, comparisons, summary, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        arms = dispatch.selected_arms(self.args.general_arm)
        self.assertEqual(summary["general_arm"], self.args.general_arm)
        self.assertEqual(summary["arm_pair"], list(arms))
        self.assertEqual(summary["arm_roles"], {"control": arms[0], "candidate": arms[1]})
        self.assertEqual(list(summary["artifacts"]), list(arms))
        self.assertEqual([run["arm"] for run in summary["runs"]], [row["arm"] for row in dispatch.schedule(3, arms[0])])
        self.assertEqual(len(processes), 8)
        self.assertEqual(comparisons.call_count, 8)
        for process, run in zip(processes, summary["runs"]):
            manifest_path = Path(process.environment["METALLIC_SHADER_ARTIFACT_MANIFEST"])
            manifest = dispatch.experiment.read(manifest_path)
            self.assertEqual(manifest["artifact_id"], run["arm"])
            self.assertEqual(Path(manifest["artifact"]["file"]).read_bytes(), self.fixture.raw[run["arm"]])
            self.assertEqual(manifest["artifact"]["source_tag"], f"Artifacts/PathBackend/{run['arm']}/TwoPassCleaned")
            for recipe in run["evidence"]["recipes"]:
                self.assertEqual(recipe["artifact"]["sha256"], manifest["artifact"]["sha256"])
        denominator = summary["timing_summary"]["paired"]["Opaque1"]["dispatch_scope_ms"]["percent_change_denominator_arm"]
        self.assertEqual(denominator, "original-general")

    def test_invalid_original_control_rejects_before_any_runtime_process(self):
        self.args.general_arm = "backend-concrete-general"
        with patch.object(dispatch.subprocess, "Popen") as process, self.assertRaises(ValueError):
            dispatch.run(self.args, self.output, dict(schema_version=1, status="running", runs=[]))
        process.assert_not_called()

    def test_correctness_failure_never_admits_dispatch_metrics(self):
        processes, comparisons, summary, error = self.run_mocked(qualified=False)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(len(processes), 1)
        self.assertEqual(comparisons.call_count, 1)
        self.assertEqual(summary["runs"], [])
        self.assertNotIn("timing_summary", summary)
        self.assertTrue((self.output / dispatch.schedule(3)[0]["label"] / "MaterialBaseline.json").is_file())
        self.assertTrue((self.output / dispatch.schedule(3)[0]["label"] / "process.json").is_file())

    def test_crash_and_timeout_retain_failed_process_reports_without_metric_admission(self):
        for timed_out, code in ((True, 1), (False, -1073741819)):
            with self.subTest(timed_out=timed_out):
                self.output = self.root / f"process-failure-{timed_out}"
                self.output.mkdir()
                processes, comparisons, summary, error = self.run_mocked(fail_at=2, timeout=timed_out, exit_code=code)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(processes), 3)
                self.assertEqual(len(summary["runs"]), 2)
                self.assertTrue(all("timings" not in run for run in summary["runs"]))
                self.assertEqual(processes[-1].killed, timed_out)
                metadata = dispatch.experiment.read(self.output / dispatch.schedule(3)[2]["label"] / "process.json")
                self.assertEqual(metadata["exit_code"], code)
                self.assertIs(metadata["timed_out"], timed_out)
                self.assertEqual(comparisons.call_count, 2)

    def test_input_drift_and_recipe_rejection_prevent_next_process(self):
        for drift in (True, False):
            with self.subTest(drift=drift):
                self.output = self.root / f"drift-or-reject-{drift}"
                self.output.mkdir()
                mutation = None if drift else lambda report, directory: report["artifact_cases"][0].update(consumed=0)
                processes, comparisons, summary, error = self.run_mocked(fail_at=2, drift=drift, mutation=mutation)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(processes), 3)
                self.assertNotIn("timing_summary", summary)
                self.args.executable.write_bytes(b"mocked executable only")

    def sealed_run(self):
        self.args.cases = self.output / "Cases.json"
        _, _, summary, error = self.run_mocked()
        self.assertIsNone(error, str(error))
        summary["status"] = "passed"
        save(self.output / "summary.json", summary)
        dispatch.experiment.seal(self.output)
        return summary, summary["runs"][0]["correctness"]

    def test_historical_record_replay_is_independent_of_current_sources_but_live_check_rejects_drift(self):
        summary, comparison = self.sealed_run()
        self.args.executable.write_bytes(b"later executable build")
        with patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison), \
             patch.object(dispatch.experiment, "gpu_inventory", return_value=summary["gpu_inventory"]):
            verified = dispatch.verify_records(self.output)
            self.assertEqual(verified["timing_summary"], summary["timing_summary"])
            with self.assertRaisesRegex(ValueError, "input changed"):
                dispatch.verify_records(self.output, current_inputs=True)

    def test_original_control_replay_keeps_actual_denominator_and_rejects_relabelled_records(self):
        self.args.general_arm = "original-general"
        original, comparison = self.sealed_run()
        self.args.executable.write_bytes(b"later executable build")
        with patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison):
            verified = dispatch.verify_records(self.output)
        self.assertEqual(verified["general_arm"], "original-general")
        mutations = (
            lambda summary: summary.update(general_arm="backend-link-general"),
            lambda summary: summary.update(arm_pair=list(dispatch.ARMS)),
            lambda summary: summary["arm_roles"].update(control="backend-link-general"),
            lambda summary: summary["timing_summary"]["paired"]["Opaque1"]["dispatch_scope_ms"].update(
                percent_change_denominator_arm="backend-link-general"),
            lambda summary: summary["artifacts"].update({"original-general": summary["artifacts"]["backend-link-no-fiber"]}))
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            save(self.output / "summary.json", changed)
            dispatch.experiment.seal(self.output)
            with self.subTest(mutation=mutation), patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison), \
                    self.assertRaises((ValueError, KeyError)):
                dispatch.verify_records(self.output)


    def test_historical_replay_rejects_resealed_metadata_tolerance_summary_and_recipe_changes(self):
        original, comparison = self.sealed_run()
        mutations = (
            lambda summary: summary["tolerance"].update(absolute=0.1),
            lambda summary: summary["runs"][0]["process"].update(pid=777),
            lambda summary: summary["runs"][2].update(execution_id=summary["runs"][0]["execution_id"]),
            lambda summary: summary["runs"][2]["timings"]["Opaque1"]["dispatch_scope_ms"].update(median=99),
            lambda summary: summary["timing_summary"]["paired"]["Opaque1"]["dispatch_scope_ms"].update(percent_change=-99),
            lambda summary: summary["runs"][0].update(timings={}),
            lambda summary: summary["runs"].pop(1),
            lambda summary: summary.update(statistical_unit="32 independent frames"))
        for mutation in mutations:
            summary = copy.deepcopy(original)
            mutation(summary)
            save(self.output / "summary.json", summary)
            dispatch.experiment.seal(self.output)
            with self.subTest(mutation=mutation), patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison), \
                 self.assertRaises((ValueError, KeyError)):
                dispatch.verify_records(self.output)
        save(self.output / "summary.json", original)
        recipe_path = next((self.output / dispatch.schedule(3)[2]["label"] / "recipes").rglob("*.recipe.json"))
        recipe = dispatch.experiment.read(recipe_path)
        recipe["shaders"]["compute"]["source_identity_ambiguous"] = True
        save(recipe_path, recipe)
        dispatch.experiment.seal(self.output)
        with patch.object(dispatch.experiment, "compare_readbacks", return_value=comparison), self.assertRaises(ValueError):
            dispatch.verify_records(self.output)

    def test_frozen_verifier_copies_actual_source_and_launches_archive_with_bytecode_disabled(self):
        summary, _ = self.sealed_run()
        source = self.output / "Verifier/Tools/Perf/MeasurePathTraceBackendDispatch.py"
        self.assertEqual(source.read_bytes(), Path(dispatch.__file__).read_bytes())
        with patch.object(dispatch.subprocess, "run", return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps(summary), stderr="")) as replay:
            self.assertEqual(dispatch.verify(self.output), summary)
        command = replay.call_args.args[0]
        self.assertEqual(command, [sys.executable, "-B", str(source), "_verify-frozen", str(self.output)])
        self.assertEqual(replay.call_args.kwargs, dict(check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        source.write_text("raise RuntimeError('changed archived verifier')", encoding="utf-8")
        with patch.object(dispatch.subprocess, "run") as replay, self.assertRaisesRegex(ValueError, "Sealed"):
            dispatch.verify(self.output)
        replay.assert_not_called()

    def test_cli_failure_seals_diagnostics_and_releases_only_its_lock(self):
        output = self.root / "failed-cli"
        argv = [str(dispatch.__file__), "run", "--preparation", str(self.args.preparation), "--compile", str(self.args.compile),
            "--analysis", str(self.args.analysis), "--reference", str(self.args.reference), "--cases", str(self.args.cases),
            "--executable", str(self.args.executable), "--output-dir", str(output)]

        def fail(args, destination, summary):
            (destination / "stdout.log").write_text("artifact rejected before measurement", encoding="utf-8")
            save(destination / "process.json", dict(exit_code=1, timed_out=False))
            raise ValueError("actual artifact did not qualify")

        with patch.object(dispatch, "ROOT", self.root), patch.object(dispatch, "run", side_effect=fail), \
             patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "did not qualify"):
            dispatch.main()
        summary = dispatch.experiment.read(output / "summary.json")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["runs"], [])
        self.assertIn("did not qualify", summary["error"])
        self.assertTrue((output / "stdout.log").is_file())
        self.assertTrue((output / "process.json").is_file())
        self.assertFalse((self.root / "build/compute-pipeline-experiment.lock").exists())
        dispatch.validate_seal(output)

    def test_non_target_proof_requires_actual_bytecode_and_unchanged_compiler_abi_in_each_case(self):
        manifests, _ = self.fixture.prepared()
        manifest = dispatch.experiment.read(manifests[dispatch.ARMS[0]])
        directory = self.root / "ordinary-proof"
        configuration = self.timing_configuration()
        self.runtime_report(directory, manifest, configuration, validation=False)
        evidence = dispatch.non_target_shaders(directory, manifest, configuration, expected_validation=False)
        self.assertEqual(list(evidence), [case["id"] for case in configuration["cases"]])
        self.assertTrue(all(len(rows) == 1 for rows in evidence.values()))
        recipe_path = directory / "recipes" / configuration["cases"][0]["id"] / "Ordinary.recipe.json"
        good = dispatch.experiment.read(recipe_path)
        for mutation in (
                lambda recipe: recipe["shaders"]["compute"].update(input_spirv_hash="0" * 16),
                lambda recipe: recipe["shaders"]["compute"].update(spirv_bytes=24),
                lambda recipe: recipe["shaders"]["compute"].update(slang_optimization="O2"),
                lambda recipe: recipe["shaders"]["compute"].update(spirv_postprocess="dead-branches"),
                lambda recipe: recipe["shaders"]["compute"].update(descriptor_heap_mode="native"),
                lambda recipe: recipe["shaders"]["compute"].update(source_identity_ambiguous=1),
                lambda recipe: recipe["shaders"]["compute"].update(module_name=manifest["artifact"]["source_tag"]),
                lambda recipe: recipe["source_device"].update(validation_enabled=True)):
            recipe = copy.deepcopy(good)
            mutation(recipe)
            save(recipe_path, recipe)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                dispatch.non_target_shaders(directory, manifest, configuration, expected_validation=False)
        save(recipe_path, good)
        (recipe_path.parent / "Ordinary.spv").write_bytes(fixture.code(99))
        with self.assertRaisesRegex(ValueError, "SPIR-V differs"):
            dispatch.non_target_shaders(directory, manifest, configuration, expected_validation=False)

    def test_non_target_bytecode_drift_between_validation_and_timing_rejects_candidate(self):
        def change_ordinary(report, directory):
            recipe_path = directory / "recipes" / report["cases"][0]["id"] / "Ordinary.recipe.json"
            recipe = dispatch.experiment.read(recipe_path)
            raw_path = recipe_path.parent / "Ordinary.spv"
            raw_path.write_bytes(fixture.code(123))
            identity = dispatch.rendering.blob_identity(raw_path)
            recipe["shaders"]["compute"].update(input_spirv_hash=identity["fnv1a64"], spirv_bytes=identity["bytes"])
            recipe["expected_pso_hash"] = dispatch.rendering.pso_hash(recipe, raw_path.read_bytes())
            save(recipe_path, recipe)

        processes, comparisons, summary, error = self.run_mocked(fail_at=2, mutation=change_ordinary)
        self.assertIsInstance(error, ValueError)
        self.assertIn("non-target", str(error).lower())
        self.assertEqual(len(processes), 3)
        self.assertEqual(comparisons.call_count, 2)
        self.assertEqual(len(summary["runs"]), 2)
        self.assertNotIn("timing_summary", summary)


if __name__ == "__main__":
    unittest.main()
