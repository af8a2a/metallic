"""Reject unproven artifact rendering evidence without Slang or GPU execution."""

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
import ValidatePathTraceBackendRendering as rendering


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seal(directory):
    save(directory / "Manifest.json", dict(schema_version=1, artifacts=[
        dict(path=path.relative_to(directory).as_posix(), sha256=rendering.experiment.digest(path))
        for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "Manifest.json"]))


def code(generator=1):
    # The runner checks blob identity; full SPIR-V validation belongs to the
    # frozen analysis and the RHI loader, which these CPU tests never invoke.
    return struct.pack("<5I", 0x07230203, 0x00010600, generator, 5, 0)


def reference_report():
    return dict(status="passed", validation_errors=0,
        configuration=dict(compile_only=True, shader_cache="reuse", slang_optimization="O1",
            descriptor_mode="mapped", spirv_postprocess="none", debug_mode="disabled",
            validation=False, selected_requests=[146]), environment={},
        samples=[dict(status="passed", shader_cache_hit=False, pipeline_status="compile_only",
            effective_slang_optimization="O1", effective_descriptor_mode="mapped", effective_spirv_postprocess="none",
            compilation_phases=dict(compiler_executed=True), request=dict(index=146,
                module="Features/PathTracing/OpenPBRRayQueryPathTraceGuides",
                entry="openPbrRayQueryPathTraceGuidesMain", profile="spirv_1_6",
                search_path=str(ROOT / "Shaders"), additional_search_paths=[],
                capabilities=["spvRayQueryKHR"], defines=[["METALLIC_CUSTOM_MATERIALS", "0"]]),
            spirv_bytes=20, spirv_fnv1a64=f"{rendering.fnv(code()):016x}", dependencies=[])])


def configuration():
    return dict(frames=4, cases=[dict(id=name, graph=f"{name}.json", width=1, height=1,
        scene=f"{name}.gltf", environment=dict(path="fixture.exr", intensity=1, rotationDegrees=0),
        output="PathTrace.color", frameCount=4, cameraFrames=[], captureFrames=[0],
        captureOutputs=["PathTrace.motion"], guideCameraStep={}, useRenderView=False, requireEnvironment=True)
        for name in ("Opaque1", "Mixed1", "Sky")])


class ArtifactRenderingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "output"
        self.output.mkdir()
        self.args = SimpleNamespace(preparation=self.root / "preparation", compile=self.root / "compile",
            analysis=self.root / "analysis", reference=self.root / "reference.json", cases=self.root / "Cases.json",
            executable=self.root / "rhi.exe", timeout=1, absolute_tolerance=2e-5, relative_tolerance=3e-5)
        for directory in (self.args.preparation, self.args.compile, self.args.analysis):
            directory.mkdir()
        save(self.args.reference, reference_report())
        save(self.args.preparation / "ReferenceReport.json", reference_report())
        (self.args.preparation / "Reference.spv").write_bytes(code())
        save(self.args.preparation / "Preparation.json", dict(status="prepared"))
        seal(self.args.preparation)
        self.args.executable.write_bytes(b"mocked executable only")
        self.config = configuration()
        save(self.args.cases, self.config)
        outputs = {}
        records, compiler_inputs = [], {}
        self.raw = {}
        for index, arm in enumerate(rendering.ARMS):
            directory = self.args.analysis / arm
            directory.mkdir()
            self.raw[arm] = code(index + 1)
            (directory / "Cleaned.spv").write_bytes(self.raw[arm])
            outputs[arm] = dict(cleaned=rendering.blob_identity(directory / "Cleaned.spv"),
                raw=dict(sha256=hashlib.sha256(self.raw[arm]).hexdigest()))
            raw_directory = self.args.compile / f"block-00-{arm}"
            raw_directory.mkdir()
            raw_path = raw_directory / "Guides.spv"
            raw_path.write_bytes(self.raw[arm])
            record = dict(label=arm, repetition=0, directory=raw_directory.name,
                spirv_sha256=hashlib.sha256(self.raw[arm]).hexdigest())
            records.append(record)
            compiler_inputs[arm] = dict(path=str(raw_path), sha256=record["spirv_sha256"], record=copy.deepcopy(record))
            save(directory / "cleaned-validation.process.json", dict(status="passed", exit_code=0,
                command=[str(self.root / "SDK/spirv-val.exe"), "--target-env", "vulkan1.4", "--scalar-block-layout",
                    str(directory / "Cleaned.spv")]))
        save(self.args.compile / "summary.json", dict(status="passed", variants=list(rendering.ARMS), runs=records,
            preparation_manifest_sha256=rendering.experiment.digest(self.args.preparation / "Manifest.json")))
        seal(self.args.compile)
        save(self.args.analysis / "Inputs.json", dict(manifests={key: dict(path=str(directory / "Manifest.json"),
            sha256=rendering.experiment.digest(directory / "Manifest.json"))
            for key, directory in (("preparation", self.args.preparation), ("compile", self.args.compile))}, inputs=compiler_inputs))
        save(self.args.analysis / "summary.json", dict(status="passed",
            cleanup_passes=["--eliminate-dead-branches", "--simplify-instructions"], outputs=outputs))
        (self.args.analysis / "opt-version.stdout.log").write_text("", encoding="utf-8")
        (self.args.analysis / "opt-version.stderr.log").write_text("SPIRV-Tools fixture validator\n", encoding="utf-8")
        for directory in (self.args.preparation, self.args.compile, self.args.analysis):
            seal(directory)

    def prepared(self):
        manifests, paths = rendering.prepare_artifacts(self.args, self.output)
        return manifests, paths

    def runtime(self, directory, manifest):
        directory.mkdir(parents=True, exist_ok=True)
        recipes = directory / "recipes"
        recipes.mkdir(exist_ok=True)
        artifact = manifest["artifact"]
        raw = Path(artifact["file"]).read_bytes()
        stage = dict(module_name=artifact["source_tag"], source_entry_point=manifest["expected_target"]["entry"],
            entry_point="main", input_spirv_hash=artifact["fnv1a64"], spirv_bytes=artifact["bytes"],
            spirv_file="Consumed.spv", source_identity_ambiguous=False, spirv_postprocess="dead-branches",
            spirv_postprocess_tool_version=artifact["tool_version"], slang_optimization="O1",
            debug_mode="release", descriptor_heap_mode="mapped")
        recipe = dict(pipeline_type="compute", shaders=dict(compute=stage), state=dict(uses_bindless_heap=True,
            indirect_bindable=False, bindless_user_push_data_size=12, disable_optimization=False,
            compute_backend=0, custom_descriptor_mappings=[]),
            source_device=dict(device_name="fixture GPU", driver_version=1, validation_enabled=True))
        recipe["expected_pso_hash"] = rendering.pso_hash(recipe, raw)
        recipe_path = recipes / self.config["cases"][0]["id"] / "Consumed.recipe.json"
        proof = dict(file=str(recipe_path), expected_pso_hash=int(recipe["expected_pso_hash"], 16),
            input_sha256=artifact["sha256"], input_fnv1a64=artifact["fnv1a64"], input_bytes=artifact["bytes"],
            source_tag=artifact["source_tag"], source_identity_ambiguous=False)
        evidence = dict(artifact_id=manifest["artifact_id"], input_sha256=artifact["sha256"],
            input_fnv1a64=artifact["fnv1a64"], input_bytes=artifact["bytes"], source_tag=artifact["source_tag"],
            postprocess="dead-branches", postprocess_tool_version=artifact["tool_version"],
            consumed=3, rejected_target=0, errors=[])
        rows, cases = [], []
        for spec in self.config["cases"]:
            case_recipes = recipes / spec["id"]
            case_recipes.mkdir(exist_ok=True)
            (case_recipes / "Consumed.spv").write_bytes(raw)
            case_recipe = case_recipes / "Consumed.recipe.json"
            save(case_recipe, recipe)
            case_proof = copy.deepcopy(proof)
            case_proof["file"] = str(case_recipe)
            rows.append(dict(id=spec["id"], status="passed", consumed=1, rejected_target=0,
                input_sha256=artifact["sha256"], input_fnv1a64=artifact["fnv1a64"],
                recipe_directory=str(case_recipes),
                scene_eligibility=dict(eligible=True, requires_no_fiber=manifest["requires_no_fiber"],
                    materials_inspected=3, scene_sources_inspected=1, custom_material_programs=0,
                    fiber_materials=0 if manifest["requires_no_fiber"] else None),
                actual_pipeline_recipes=[case_proof]))
            case = copy.deepcopy(spec)
            case.update(validHDR=True, format="RGBA32F", image=f"{spec['id']}-color.bin", resolvedGraph=dict(nodes=[]),
                captures=[dict(output=spec["output"], frame=0, file=f"{spec['id']}-temporal.bin", bytes=16, format=28),
                    dict(output="PathTrace.motion", frame=4, file=f"{spec['id']}-motion.bin", bytes=16, format=28)])
            cases.append(case)
        report = dict(status="passed", validation=True, cases=cases, shader_artifact=evidence, artifact_cases=rows)
        save(directory / "MaterialBaseline.json", report)
        (directory / "stdout.log").write_text("Rendering passed\n", encoding="utf-8")
        (directory / "stderr.log").write_text("", encoding="utf-8")
        return report, recipe, recipe_path

    def test_sealed_paths_reject_escape_duplicate_and_changed_files(self):
        manifest_path = self.args.preparation / "Manifest.json"
        good = rendering.experiment.read(manifest_path)
        outside = self.root / "outside.json"
        outside.write_bytes(b"outside")
        for mutation in (
                lambda data: data["artifacts"].append(dict(path="../outside.json", sha256=rendering.experiment.digest(outside))),
                lambda data: data["artifacts"].append(copy.deepcopy(data["artifacts"][0])),
                lambda data: data["artifacts"][0].update(sha256="0" * 64)):
            modified = copy.deepcopy(good)
            mutation(modified)
            save(manifest_path, modified)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                rendering.validate_manifest(manifest_path)

    def test_missing_inventory_and_unlisted_diagnostic_files_reject(self):
        path = self.args.analysis / "Manifest.json"
        good = rendering.experiment.read(path)
        modified = copy.deepcopy(good)
        modified["artifacts"] = modified["artifacts"][:-1]
        save(path, modified)
        with self.assertRaises(ValueError):
            rendering.validate_manifest(path)
        save(path, good)
        diagnostic = self.args.analysis / "diagnostic-merge-blocks"
        diagnostic.mkdir()
        (diagnostic / "unsealed.json").write_bytes(b"changed diagnostic")
        with self.assertRaises(ValueError):
            rendering.validate_manifest(path)

    def test_prepare_keeps_exact_five_arm_bytes_tags_and_no_fiber_contract(self):
        before = {arm: raw for arm, raw in self.raw.items()}
        manifests, paths = self.prepared()
        self.assertEqual(tuple(manifests), rendering.ARMS)
        for arm, path in manifests.items():
            manifest = rendering.experiment.read(path)
            self.assertEqual(manifest["artifact_id"], arm)
            self.assertIs(manifest["requires_no_fiber"], arm.endswith("no-fiber"))
            self.assertEqual(manifest["artifact"]["source_tag"], f"Artifacts/PathBackend/{arm}/TwoPassCleaned")
            self.assertEqual(manifest["artifact"]["tool_version"], "SPIRV-Tools fixture validator")
            self.assertEqual(manifest["artifact"]["spirv_postprocess"], "dead-branches")
            artifact = Path(manifest["artifact"]["file"])
            self.assertEqual(artifact.read_bytes(), before[arm])
            identity = rendering.blob_identity(artifact)
            self.assertEqual({key: manifest["artifact"][key] for key in identity}, identity)
            self.assertEqual(manifest["reference_request"], reference_report()["samples"][0]["request"])
            self.assertEqual(len(manifest["provenance"]["manifests"]), 3)
            self.assertIn(artifact, paths)

    def test_reference_policy_and_cleaned_analysis_mismatch_reject(self):
        good = reference_report()
        for mutation in (
                lambda report: report["configuration"].update(debug_mode="capture"),
                lambda report: report["configuration"].update(slang_optimization="O2"),
                lambda report: report["samples"][0].update(shader_cache_hit=True),
                lambda report: report["samples"][0]["compilation_phases"].update(compiler_executed=False),
                lambda report: report["samples"][0]["request"].update(index=149)):
            changed = copy.deepcopy(good)
            mutation(changed)
            save(self.args.reference, changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prepared()
        save(self.args.reference, good)
        analysis = rendering.experiment.read(self.args.analysis / "summary.json")
        analysis["outputs"][rendering.ARMS[0]]["cleaned"]["sha256"] = "0" * 64
        save(self.args.analysis / "summary.json", analysis)
        seal(self.args.analysis)
        with self.assertRaisesRegex(ValueError, "artifact mismatch"):
            self.prepared()

    def test_missing_actual_optimizer_version_and_invalid_spirv_reject(self):
        (self.args.analysis / "opt-version.stderr.log").write_text("validator unknown\n", encoding="utf-8")
        seal(self.args.analysis)
        with self.assertRaisesRegex(ValueError, "tool version"):
            self.prepared()
        bad = self.root / "bad.spv"
        for raw in (b"", code()[:-1], bytes(20)):
            bad.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                rendering.blob_identity(bad)

    def test_resealed_analysis_cannot_mix_another_compiler_manifest_or_validation_scope(self):
        foreign = self.root / "foreign-compile"
        foreign.mkdir()
        save(foreign / "summary.json", dict(status="passed"))
        seal(foreign)
        path = self.args.analysis / "Inputs.json"
        original = rendering.experiment.read(path)
        mixed = copy.deepcopy(original)
        mixed["manifests"]["compile"] = dict(path=str(foreign / "Manifest.json"),
            sha256=rendering.experiment.digest(foreign / "Manifest.json"))
        save(path, mixed)
        seal(self.args.analysis)
        with self.assertRaisesRegex(ValueError, "different preparation/compiler"):
            self.prepared()
        save(path, original)
        validation_path = self.args.analysis / rendering.ARMS[0] / "cleaned-validation.process.json"
        validation = rendering.experiment.read(validation_path)
        validation["command"][2] = "vulkan1.2"
        save(validation_path, validation)
        seal(self.args.analysis)
        with self.assertRaisesRegex(ValueError, "validation evidence"):
            self.prepared()

    def workload_inputs(self):
        settings = copy.deepcopy(self.config)
        environment = self.root / "fixture.exr"
        environment.write_bytes(b"frozen environment image")
        mesh = self.root / "mesh.bin"
        mesh.write_bytes(b"frozen geometry")
        image = self.root / "material.png"
        image.write_bytes(b"frozen material image")
        for case in settings["cases"]:
            scene = self.root / f"{case['id']}.gltf"
            save(scene, dict(buffers=[dict(uri=mesh.name)], images=[dict(uri=image.name), dict(uri="data:image/png;base64,AAAA")]))
            Path(str(scene) + ".meshlets.bin").write_bytes(b"prepopulated geometry cache")
            save(self.root / case["graph"], dict(nodes=[dict(type="ScenePathTracePass", properties=dict(
                path=scene.name, samples=1, maxDepth=4, cacheMode="sharc", bsdf="openpbr", exportDenoiserGuides=True))]))
            case["environment"]["path"] = str(environment)
        save(self.args.cases, settings)
        return settings

    def test_workload_keeps_inputs_frozen_and_prepares_fixed_two_sample_guides(self):
        settings = self.workload_inputs()
        before = {path: path.read_bytes() for path in self.root.glob("*.json")}
        with patch.object(rendering, "ROOT", self.root):
            path, config, inputs, absent = rendering.prepare_workload(self.args.cases, self.output)
        self.assertEqual([case["id"] for case in config["cases"]], ["Opaque1", "Mixed1", "Sky"])
        self.assertEqual(len(absent), 3)
        rendering.check_absences(absent)
        for case in config["cases"]:
            self.assertEqual((case["width"], case["height"], case["useRenderView"], case["requireEnvironment"]), (128, 128, False, True))
            props = rendering.experiment.read(Path(case["graph"]))["nodes"][0]["properties"]
            self.assertEqual((props["samples"], props["maxDepth"], props["cacheMode"]), (2, 12, "off"))
            self.assertIn(Path(str(case["scene"]) + ".meshlets.bin"), inputs)
        self.assertIn(self.root / "mesh.bin", inputs)
        self.assertIn(self.root / "material.png", inputs)
        self.assertIn(self.root / "fixture.exr", inputs)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_external_materials_missing_geometry_cache_and_new_sidecars_reject(self):
        settings = self.workload_inputs()
        with patch.object(rendering, "ROOT", self.root):
            _, _, _, absent = rendering.prepare_workload(self.args.cases, self.output)
            Path(absent[0]).write_bytes(b"uninspected implicit material set")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                rendering.check_absences(absent)
            with self.assertRaisesRegex(ValueError, "sidecar"):
                rendering.prepare_workload(self.args.cases, self.output)
            Path(absent[0]).unlink()
            modified = copy.deepcopy(settings)
            modified["cases"][0]["materialAsset"] = "uninspected.material"
            save(self.args.cases, modified)
            with self.assertRaisesRegex(ValueError, "material libraries"):
                rendering.prepare_workload(self.args.cases, self.output)
            save(self.args.cases, settings)
            (self.root / "Opaque1.gltf.meshlets.bin").unlink()
            with self.assertRaisesRegex(ValueError, "geometry caches"):
                rendering.prepare_workload(self.args.cases, self.output)

    def test_runtime_accepts_only_consumed_artifact_with_actual_recipe_proof(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        directory = self.root / "runtime"
        self.runtime(directory, manifest)
        evidence = rendering.verify_runtime(directory, manifest, self.config)
        self.assertEqual(len(evidence["recipes"]), 3)
        self.assertEqual(len(evidence["devices"]), 1)

    def test_runtime_unused_rejected_incomplete_case_and_fallback_evidence_reject(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        directory = self.root / "runtime"
        good, _, _ = self.runtime(directory, manifest)
        mutations = (
            lambda report: report.update(status="failed"),
            lambda report: report.update(error="material fallback"),
            lambda report: report.update(validation=False),
            lambda report: report["shader_artifact"].update(consumed=0),
            lambda report: report["shader_artifact"].update(consumed=True),
            lambda report: report["shader_artifact"].update(rejected_target=1),
            lambda report: report["shader_artifact"].update(rejected_target=False),
            lambda report: report["shader_artifact"].update(errors=["request mismatch"]),
            lambda report: report["shader_artifact"].update(source_tag="OtherProducer"),
            lambda report: report["shader_artifact"].update(postprocess_tool_version="different tool"),
            lambda report: report["artifact_cases"].pop(),
            lambda report: report["artifact_cases"][0].update(consumed=0),
            lambda report: report["artifact_cases"][0].update(rejected_target=1),
            lambda report: report["artifact_cases"][0].update(actual_pipeline_recipes=[]),
            lambda report: report["artifact_cases"][0]["scene_eligibility"].update(eligible=False),
            lambda report: report["cases"][0].update(validHDR=False),
            lambda report: report["cases"][0].update(width=2),
            lambda report: report["cases"][0]["captures"].reverse())
        for mutation in mutations:
            modified = copy.deepcopy(good)
            mutation(modified)
            save(directory / "MaterialBaseline.json", modified)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                rendering.verify_runtime(directory, manifest, self.config)

    def test_no_fiber_and_per_case_recipe_proof_cannot_be_asserted_without_matching_evidence(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests["backend-link-no-fiber"])
        directory = self.root / "runtime"
        good, _, _ = self.runtime(directory, manifest)
        for mutation in (
                lambda report: report["artifact_cases"][0]["scene_eligibility"].update(requires_no_fiber=False),
                lambda report: report["artifact_cases"][0]["scene_eligibility"].update(fiber_materials=1),
                lambda report: report["artifact_cases"][0]["scene_eligibility"].update(custom_material_programs=1),
                lambda report: report["artifact_cases"][0].update(actual_pipeline_recipes=[{}]),
                lambda report: report["artifact_cases"][0]["actual_pipeline_recipes"][0].update(file=str(self.root / "foreign.recipe.json")),
                lambda report: report["artifact_cases"][0]["actual_pipeline_recipes"][0].update(input_sha256="0" * 64)):
            modified = copy.deepcopy(good)
            mutation(modified)
            save(directory / "MaterialBaseline.json", modified)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                rendering.verify_runtime(directory, manifest, self.config)

    def test_recipe_bytes_provenance_abi_state_and_pso_hash_drift_reject(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        directory = self.root / "runtime"
        _, good, path = self.runtime(directory, manifest)
        mutations = (
            lambda recipe: recipe["shaders"]["compute"].update(module_name="WrongProducer"),
            lambda recipe: recipe["shaders"]["compute"].update(source_identity_ambiguous=True),
            lambda recipe: recipe["shaders"]["compute"].update(input_spirv_hash="0" * 16),
            lambda recipe: recipe["shaders"]["compute"].update(source_entry_point="otherMain"),
            lambda recipe: recipe["shaders"]["compute"].update(slang_optimization="O2"),
            lambda recipe: recipe["shaders"]["compute"].update(descriptor_heap_mode="native"),
            lambda recipe: recipe["shaders"]["compute"].update(spirv_postprocess_tool_version="different"),
            lambda recipe: recipe["state"].update(bindless_user_push_data_size=16),
            lambda recipe: recipe["state"].update(compute_backend=1),
            lambda recipe: recipe.update(expected_pso_hash="0" * 16),
            lambda recipe: recipe["source_device"].update(validation_enabled=False))
        for mutation in mutations:
            modified = copy.deepcopy(good)
            mutation(modified)
            save(path, modified)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                rendering.verify_runtime(directory, manifest, self.config)
        save(path, good)
        (path.parent / "Consumed.spv").write_bytes(code(999))
        with self.assertRaises(ValueError):
            rendering.verify_runtime(directory, manifest, self.config)

    def test_runtime_spirv_file_cannot_escape_the_fresh_case_recipe_directory(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        directory = self.root / "runtime"
        _, recipe, path = self.runtime(directory, manifest)
        outside = directory / "recipes" / "outside.spv"
        outside.write_bytes(Path(manifest["artifact"]["file"]).read_bytes())
        recipe["shaders"]["compute"]["spirv_file"] = "../outside.spv"
        save(path, recipe)
        with self.assertRaises(ValueError):
            rendering.verify_runtime(directory, manifest, self.config)

    def test_fallback_and_vulkan_validation_logs_cannot_qualify(self):
        manifests, _ = self.prepared()
        manifest = rendering.experiment.read(manifests[rendering.ARMS[0]])
        directory = self.root / "runtime"
        self.runtime(directory, manifest)
        for text in ("Initial material compilation failed", "Validation Error", "VUID-failed-state",
                     "[ShaderArtifactOverride] Rejected: wrong permutation", "[ShaderArtifactOverride] Resolve failed: exception"):
            (directory / "stderr.log").write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                rendering.verify_runtime(directory, manifest, self.config)

    def run_mocked(self, *, timeout=False, exit_code=0, drift=False, qualified=True):
        processes = []
        compare = patch.object(rendering.experiment, "compare_readbacks", return_value=[dict(qualified=qualified)])
        fixture = self

        class Process:
            def __init__(self, command, **kwargs):
                self.pid = 900 + len(processes)
                self.command, self.environment = command, kwargs["env"]
                self.killed, self.waits = False, 0
                processes.append(self)
                directory = Path(command[command.index("--output-dir") + 1])
                manifest = rendering.experiment.read(self.environment["METALLIC_SHADER_ARTIFACT_MANIFEST"])
                fixture.runtime(directory, manifest)
                if drift:
                    fixture.args.executable.write_bytes(b"changed during process")

            def wait(self, **kwargs):
                self.waits += 1
                if timeout and self.waits == 1:
                    raise subprocess.TimeoutExpired(self.command, 1)
                return exit_code

            def kill(self):
                self.killed = True

        summary = dict(runs=[])
        with patch.object(rendering, "prepare_workload", return_value=(self.args.cases, self.config, [self.args.cases], [])), \
             patch.object(rendering.experiment, "input_identity", side_effect=lambda executable, *args, **kwargs:
                          rendering.experiment.snapshots([executable])), \
             patch.object(rendering.driver, "suppress_windows_error_dialogs", side_effect=nullcontext), \
             patch.object(rendering.subprocess, "Popen", side_effect=Process), compare as comparisons, \
             redirect_stdout(io.StringIO()):
            try:
                rendering.run(self.args, self.output, summary)
            except Exception as error:
                return processes, comparisons, summary, error
        return processes, comparisons, summary, None

    def test_mocked_processes_use_five_fresh_arms_and_delegate_all_hdr_comparisons(self):
        processes, comparisons, summary, error = self.run_mocked()
        self.assertIsNone(error)
        self.assertEqual([run["arm"] for run in summary["runs"]], list(rendering.ARMS))
        self.assertEqual(len({process.pid for process in processes}), 5)
        self.assertEqual(comparisons.call_count, 7) # Five original comparisons, two concrete/link controls.
        for process, arm in zip(processes, rendering.ARMS):
            environment = process.environment
            self.assertIn("--rhi-validation", process.command)
            self.assertEqual(environment["METALLIC_SPIRV_POSTPROCESS"], "none")
            self.assertEqual(environment["METALLIC_WORKING_COLOR_SPACE"], "acescg")
            self.assertEqual(Path(environment["METALLIC_SHADER_ARTIFACT_MANIFEST"]).name, f"{arm}.artifact.json")
            self.assertEqual(Path(environment["METALLIC_PIPELINE_RECIPE_DIRECTORY"]), self.output / arm / "recipes")
        for call in comparisons.call_args_list:
            self.assertEqual(call.args[2:], (self.config, self.args.absolute_tolerance, self.args.relative_tolerance))
        original = self.output / rendering.ARMS[0] / "MaterialBaseline.json"
        self.assertTrue(all(call.args[0] == original for call in comparisons.call_args_list[:5]))
        self.assertEqual(comparisons.call_args_list[-1].args[:2],
            (self.output / "backend-concrete-no-fiber/MaterialBaseline.json", self.output / "backend-link-no-fiber/MaterialBaseline.json"))

    def test_timeout_and_crash_stop_preserve_metadata_and_skip_hdr_comparison(self):
        for timed_out, code_value in ((True, 1), (False, -1073741819)):
            with self.subTest(timed_out=timed_out):
                target = self.root / f"failed-{timed_out}"
                target.mkdir()
                self.output = target
                processes, comparisons, summary, error = self.run_mocked(timeout=timed_out, exit_code=code_value)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(processes), 1)
                self.assertEqual(processes[0].killed, timed_out)
                comparisons.assert_not_called()
                metadata = rendering.experiment.read(target / rendering.ARMS[0] / "process.json")
                self.assertIs(metadata["timed_out"], timed_out)
                self.assertEqual(metadata["exit_code"], code_value)
                self.assertEqual(summary["runs"], [])

    def test_input_drift_and_readback_failure_stop_before_next_arm(self):
        for drift, qualified in ((True, True), (False, False)):
            with self.subTest(drift=drift):
                self.output = self.root / f"gated-{drift}"
                self.output.mkdir()
                processes, comparisons, summary, error = self.run_mocked(drift=drift, qualified=qualified)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(len(processes), 1)


if __name__ == "__main__":
    unittest.main()
