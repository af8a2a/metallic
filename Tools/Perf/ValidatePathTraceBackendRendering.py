"""Execute the frozen five-arm material-backend SPIR-V in real render graphs.

This is a validation-on correctness experiment. It does not qualify GPU timing.
The test-only scoped artifact resolver rejects mismatched target permutations;
exported PSO recipes independently prove which device bytecode was exercised.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver
from ValidatePathTraceBackend import reference_sample
from PathTraceBackendOverlay import ARMS

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counter(value, minimum=0):
    return type(value) is int and value >= minimum


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value


def blob_identity(path):
    raw = Path(path).read_bytes()
    require(len(raw) >= 20 and len(raw) % 4 == 0 and raw[:4] == b"\x03\x02\x23\x07", "Invalid artifact SPIR-V")
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), fnv1a64=f"{fnv(raw):016x}")


def validate_manifest(path):
    manifest = experiment.read(path)
    require(manifest.get("schema_version") == 1 and isinstance(manifest.get("artifacts"), list), "Unsupported sealed manifest")
    paths = [path]
    seen = set()
    for item in manifest["artifacts"]:
        source = (path.parent / item["path"]).resolve(strict=True)
        require(source.is_relative_to(path.parent.resolve()) and source not in seen, "Invalid or duplicate sealed artifact path")
        require(experiment.digest(source) == item["sha256"], f"Sealed artifact changed: {source}")
        seen.add(source)
        paths.append(source)
    require(seen, "Empty sealed provenance")
    actual = {source.resolve() for source in path.parent.rglob("*") if source.is_file() and source != path}
    require(actual == seen, "Sealed manifest inventory differs from its source directory")
    return paths


def prepare_workload(cases_path, output):
    original = experiment.read(cases_path)
    configuration = copy.deepcopy(original)
    configuration["cases"] = []
    paths = [cases_path]
    absent = []
    for source in original["cases"]:
        if source["id"] not in ("Opaque1", "Mixed1", "Sky"):
            continue
        case = copy.deepcopy(source)
        require(not any(key in case for key in ("materialAsset", "materialRoot")), "Artifact fixture rejects external material libraries")
        graph_path = (cases_path.parent / case["graph"]).resolve(strict=True)
        graph = experiment.read(graph_path)
        nodes = [node for node in graph["nodes"] if node["type"] == "ScenePathTracePass"]
        require(len(nodes) == 1, "Expected one path-trace pass")
        properties = nodes[0]["properties"]
        properties.update(samples=2, maxDepth=12, cacheMode="off")
        require(properties.get("bsdf") == "openpbr" and properties.get("exportDenoiserGuides"), "Expected OpenPBR Guides graph")
        scene = (ROOT / properties["path"]).resolve(strict=True)
        require(scene.suffix.lower() == ".gltf", "Artifact fixture requires a single-source glTF scene")
        sidecar = scene.with_suffix(".metallic_scene.json")
        require(not sidecar.exists(), "Artifact fixture requires no implicit scene sidecar")
        absent.append(str(sidecar))
        properties["path"] = str(scene)
        case.update(scene=str(scene), useRenderView=False, requireEnvironment=True)
        case["width"] = case["height"] = 128
        target = output / f"{case['id']}.json"
        experiment.save(target, graph)
        case["graph"] = str(target)
        configuration["cases"].append(case)
        paths += [graph_path, target, scene, Path(case["environment"]["path"])]
        meshlet_cache = Path(str(scene) + ".meshlets.bin")
        require(meshlet_cache.is_file(), "Prepopulate geometry caches before the correctness experiment")
        paths.append(meshlet_cache)
        scene_json = experiment.read(scene)
        for item in scene_json.get("buffers", []) + scene_json.get("images", []):
            uri = item.get("uri", "")
            if uri and not uri.startswith("data:"):
                paths.append((scene.parent / uri).resolve(strict=True))
    require([case["id"] for case in configuration["cases"]] == ["Opaque1", "Mixed1", "Sky"], "Missing fixed correctness scenes")
    target = output / "Cases.json"
    experiment.save(target, configuration)
    return target, configuration, paths + [target], sorted(set(absent))


def check_absences(paths):
    require(all(not Path(path).exists() for path in paths), "An implicit scene sidecar appeared during the experiment")


def prepare_artifacts(args, output):
    paths = []
    provenance = []
    for directory in (args.preparation, args.compile, args.analysis):
        manifest = directory.resolve(strict=True) / "Manifest.json"
        paths += validate_manifest(manifest)
        provenance.append(dict(file=str(manifest), sha256=experiment.digest(manifest)))
    reference = args.reference.resolve(strict=True)
    report = experiment.read(reference)
    require(report["status"] == "passed" and len(report["samples"]) == 1, "Invalid original reference")
    sample = reference_sample(report, 146)
    frozen_sample = reference_sample(experiment.read(args.preparation / "ReferenceReport.json"), 146)
    require(sample == frozen_sample, "Reference differs from the compiled overlay preparation")
    analysis = experiment.read(args.analysis / "summary.json")
    analysis_inputs = experiment.read(args.analysis / "Inputs.json")
    compilation = experiment.read(args.compile / "summary.json")
    for key, directory in (("preparation", args.preparation), ("compile", args.compile)):
        linked = analysis_inputs["manifests"][key]
        manifest_path = (directory / "Manifest.json").resolve()
        require(Path(linked["path"]).resolve() == manifest_path and linked["sha256"] == experiment.digest(manifest_path),
                "Cleanup analysis references a different preparation/compiler manifest")
    require(compilation["status"] == "passed" and compilation["variants"] == list(ARMS) and
            compilation["preparation_manifest_sha256"] == experiment.digest(args.preparation / "Manifest.json"), "Compiler/preparation chain mismatch")
    require(analysis["status"] == "passed" and analysis["cleanup_passes"] ==
            ["--eliminate-dead-branches", "--simplify-instructions"], "Unexpected cleanup pass policy")
    version_text = "\n".join((args.analysis / f"opt-version.{stream}.log").read_text(encoding="utf-8") for stream in ("stdout", "stderr"))
    version = next((line.strip() for line in version_text.splitlines() if line.startswith("SPIRV-Tools")), "")
    require(version.startswith("SPIRV-Tools"), "Missing cleanup tool version")
    manifests = {}
    for arm in ARMS:
        artifact = (args.analysis / arm / "Cleaned.spv").resolve(strict=True)
        identity = blob_identity(artifact)
        expected = analysis["outputs"][arm]["cleaned"]
        require(identity["bytes"] == expected["bytes"] and identity["sha256"] == expected["sha256"], "Cleaned analysis artifact mismatch")
        original_input = analysis_inputs["inputs"][arm]
        raw_path = Path(original_input["path"]).resolve(strict=True)
        require(raw_path.is_relative_to(args.compile.resolve()) and experiment.digest(raw_path) == original_input["sha256"] ==
                analysis["outputs"][arm]["raw"]["sha256"], "Raw compiler/analysis chain mismatch")
        records = [record for record in compilation["runs"] if record["label"] == arm and record["repetition"] == 0]
        require(len(records) == 1 and records[0] == original_input["record"] and records[0]["spirv_sha256"] == original_input["sha256"],
                "Cleanup source differs from its recorded compiler arm")
        validation = experiment.read(args.analysis / arm / "cleaned-validation.process.json")
        command = validation["command"]
        require(validation["status"] == "passed" and validation["exit_code"] == 0 and
                Path(command[0]).name == "spirv-val.exe" and command[1:4] == ["--target-env", "vulkan1.4", "--scalar-block-layout"] and
                len(command) == 5 and Path(command[4]).resolve() == artifact, "Cleaned SPIR-V validation evidence mismatch")
        if arm == "original-general":
            raw = blob_identity(args.preparation / "Reference.spv")
            require(raw["bytes"] == sample["spirv_bytes"] and raw["fnv1a64"] == sample["spirv_fnv1a64"] and
                    raw["sha256"] == analysis["outputs"][arm]["raw"]["sha256"], "Original baseline raw bytes differ from the reference")
        manifest = dict(version=1, artifact_id=arm, requires_no_fiber=arm.endswith("no-fiber"),
                        reference_request=sample["request"], expected_target={key: sample["request"][key] for key in ("module", "entry")},
                        reference_settings=dict(optimization="O1", descriptor="mapped", debug="disabled", working_space="acescg", postprocess="none"),
                        artifact=dict(file=str(artifact), **identity, entry_main="main", source_tag=f"Artifacts/PathBackend/{arm}/TwoPassCleaned",
                                      spirv_postprocess="dead-branches", tool_version=version), provenance=dict(manifests=provenance))
        path = output / f"{arm}.artifact.json"
        experiment.save(path, manifest)
        manifests[arm] = path
        paths += [path, artifact]
    return manifests, paths + [reference]


def pso_hash(recipe, raw):
    state = recipe["state"]
    require(recipe["pipeline_type"] == "compute" and not state.get("custom_descriptor_mappings"), "Unexpected artifact PSO type/mappings")
    entry = recipe["shaders"]["compute"]["entry_point"].encode("utf-8")
    data = struct.pack("<IIQ", 6, 0x4350534f, fnv(struct.pack("<Q", len(raw)) + raw))
    data += struct.pack("<Q", len(entry)) + entry
    data += struct.pack("<BBII", state["uses_bindless_heap"], state["indirect_bindable"], state["bindless_user_push_data_size"], 0)
    if state["disable_optimization"] or state["compute_backend"]:
        data += struct.pack("<IBB", 0x45585031, state["disable_optimization"], state["compute_backend"])
    return f"{fnv(data):016x}"


def verify_runtime(directory, manifest, configuration, *, expected_validation=True):
    report_path = directory / "MaterialBaseline.json"
    report = experiment.read(report_path)
    require(type(expected_validation) is bool and report["validation"] is expected_validation, "Runtime validation configuration mismatch")
    require([case["id"] for case in report["cases"]] == [case["id"] for case in configuration["cases"]], "Missing correctness case")
    artifact = manifest["artifact"]
    require(report.get("status") == "passed" and "error" not in report, "Artifact test did not qualify")
    evidence = report["shader_artifact"]
    require(evidence["artifact_id"] == manifest["artifact_id"] and evidence["input_sha256"] == artifact["sha256"] and
            evidence["input_fnv1a64"] == artifact["fnv1a64"] and evidence["input_bytes"] == artifact["bytes"] and
            counter(evidence["consumed"], 1) and type(evidence["rejected_target"]) is int and evidence["rejected_target"] == 0 and
            not evidence["errors"] and evidence["source_tag"] == artifact["source_tag"] and
            evidence["postprocess"] == artifact["spirv_postprocess"] and evidence["postprocess_tool_version"] == artifact["tool_version"],
            "Artifact unused, rejected or changed")
    require([row["id"] for row in report["artifact_cases"]] == [case["id"] for case in configuration["cases"]], "Missing artifact case evidence")
    for case, specification, row in zip(report["cases"], configuration["cases"], report["artifact_cases"]):
        experiment.validate_requested_case(case, specification, configuration)
        require(row["status"] == "passed" and row["input_sha256"] == artifact["sha256"] and row["input_fnv1a64"] == artifact["fnv1a64"] and
                counter(row["consumed"], 1) and type(row["rejected_target"]) is int and row["rejected_target"] == 0 and row["actual_pipeline_recipes"],
                "Missing actual per-case artifact consumption")
        eligibility = row["scene_eligibility"]
        require(eligibility["eligible"] is True and case["validHDR"] is True and
                eligibility["requires_no_fiber"] is manifest["requires_no_fiber"] and eligibility["custom_material_programs"] == 0 and
                counter(eligibility["materials_inspected"], 1) and
                (not manifest["requires_no_fiber"] or eligibility["fiber_materials"] == 0), "Unqualified scene/artifact HDR")
    log = (directory / "stdout.log").read_text(encoding="utf-8", errors="replace") + (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
    lower = log.lower()
    require(not any(marker in lower for marker in ("initial material compilation failed", "displaying the error material",
            "validation error", "vuid-", "[shaderartifactoverride] rejected", "[shaderartifactoverride] resolve failed")),
            "Material fallback, artifact rejection or validation error")
    recipes = []
    devices = []
    verified = {}
    for path in sorted((directory / "recipes").rglob("*.recipe.json")):
        recipe = experiment.read(path)
        stage = recipe["shaders"].get("compute", {})
        if stage.get("module_name") != artifact["source_tag"]:
            continue
        raw_path = (path.parent / stage["spirv_file"]).resolve(strict=True)
        require(raw_path.is_relative_to(path.parent.resolve()), "Device artifact path escapes its recipe directory")
        identity = blob_identity(raw_path)
        require(identity == {key: artifact[key] for key in identity}, "Device artifact differs from tested SPIR-V")
        require(stage["input_spirv_hash"] == artifact["fnv1a64"] and stage["spirv_bytes"] == artifact["bytes"] and
                stage["entry_point"] == "main" and stage["source_entry_point"] == manifest["expected_target"]["entry"] and
                stage["source_identity_ambiguous"] is False and stage["spirv_postprocess"] == "dead-branches" and
                stage["spirv_postprocess_tool_version"] == artifact["tool_version"] and stage["slang_optimization"] == "O1" and
                stage["debug_mode"] == "release" and stage["descriptor_heap_mode"] == "mapped", "Device artifact provenance/ABI mismatch")
        state = recipe["state"]
        require(state["bindless_user_push_data_size"] == 12 and state["uses_bindless_heap"] and not state["indirect_bindable"] and
                state["compute_backend"] == 0 and not state["disable_optimization"], "Unexpected runtime artifact pipeline state")
        require(recipe["expected_pso_hash"] == pso_hash(recipe, raw_path.read_bytes()), "Actual artifact PSO hash mismatch")
        require(recipe["source_device"]["validation_enabled"] is expected_validation, "Recipe validation mismatch")
        if recipe["source_device"] not in devices:
            devices.append(recipe["source_device"])
        item = dict(file=str(path.resolve()), sha256=experiment.digest(path), pso_hash=recipe["expected_pso_hash"], artifact=identity)
        verified[path.resolve()] = item
        recipes.append(item)
    require(recipes and len(devices) == 1, "Actual artifact device creation was not proven")
    for row in report["artifact_cases"]:
        expected_directory = (directory / "recipes" / row["id"]).resolve()
        require(Path(row["recipe_directory"]).resolve() == expected_directory, "Wrong per-case recipe capture directory")
        for item in row["actual_pipeline_recipes"]:
            path = Path(item["file"]).resolve()
            require(path.parent == expected_directory and path in verified, "Per-case PSO was not independently verified")
            require(item["expected_pso_hash"] == int(verified[path]["pso_hash"], 16) and item["input_sha256"] == artifact["sha256"] and
                    item["input_fnv1a64"] == artifact["fnv1a64"] and item["input_bytes"] == artifact["bytes"] and
                    item["source_tag"] == artifact["source_tag"] and item["source_identity_ambiguous"] is False, "Per-case PSO evidence changed")
    return dict(report=str(report_path), recipes=recipes, devices=devices)


def run(args, output, summary):
    manifests, paths = prepare_artifacts(args, output)
    cases_path, configuration, workload_paths, absent = prepare_workload(args.cases.resolve(strict=True), output)
    executable = args.executable.resolve(strict=True)
    inputs = experiment.input_identity(executable, paths + workload_paths + [Path(__file__),
        ROOT / "Source/Runtime/Render/Core/ShaderRegistry.h", ROOT / "tests/rhi/ShaderArtifactFixture.h"], production=True)
    experiment.save(output / "Inputs.json", inputs)
    experiment.save(output / "AbsentInputs.json", absent)
    summary.update(workload=configuration, tolerance=dict(absolute=args.absolute_tolerance, relative=args.relative_tolerance),
                   scope="Validation-on fixed-scene HDR, temporal and guide equivalence; no GPU timing claim")
    baseline = None
    devices = None
    reports = {}
    for arm, path in manifests.items():
        experiment.check_snapshots(inputs)
        check_absences(absent)
        destination = output / arm
        destination.mkdir()
        environment = experiment.environment_for(dict(optimization="O1", postprocess="none", backend="pipeline", disable_driver_optimization=False),
                                                  experiment.clean_environment(), internal_cache="enabled")
        environment.update(METALLIC_SHADER_ARTIFACT_MANIFEST=str(path), METALLIC_MATERIAL_BASELINE_CASES=str(cases_path),
                           METALLIC_PSO_CACHE_DIRECTORY=str(destination / "pso-cache"),
                           METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / "recipes"), METALLIC_WORKING_COLOR_SPACE="acescg")
        command = [str(executable), "--gtest_filter=RHIRendering.material_phase0_baseline", "--rhi-validation", "--output-dir", str(destination)]
        print(f"Rendering {arm}: exact artifact, validation on", flush=True)
        started = time.time()
        with (destination / "stdout.log").open("w", encoding="utf-8") as stdout, (destination / "stderr.log").open("w", encoding="utf-8") as stderr:
            with driver.suppress_windows_error_dialogs():
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            timed_out = False
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill(); code = process.wait(); timed_out = True
        metadata = dict(pid=process.pid, command=command, started=started, ended=time.time(), exit_code=code, timed_out=timed_out,
                        native_crash=driver.native_process_crashed(code), environment={k: v for k, v in environment.items() if k.startswith("METALLIC_")})
        experiment.save(destination / "process.json", metadata)
        require(code == 0 and not timed_out, f"Artifact render process failed ({code}); preserved {destination}")
        evidence = verify_runtime(destination, experiment.read(path), configuration)
        if devices is None:
            devices = evidence["devices"]
        require(devices == evidence["devices"], "Device/driver configuration changed")
        current = Path(evidence["report"])
        baseline = baseline or current
        comparison = experiment.compare_readbacks(baseline, current, configuration, args.absolute_tolerance, args.relative_tolerance)
        summary["runs"].append(dict(arm=arm, process=metadata, evidence=evidence, comparisons=comparison))
        experiment.save(output / "summary.json", summary)
        require(all(case["qualified"] for case in comparison), f"Artifact {arm} exceeded readback tolerance")
        reports[arm] = current
        experiment.check_snapshots(inputs)
        check_absences(absent)
    summary["static_link_comparisons"] = {feature: experiment.compare_readbacks(reports[f"backend-concrete-{feature}"], reports[f"backend-link-{feature}"],
        configuration, args.absolute_tolerance, args.relative_tolerance) for feature in ("general", "no-fiber")}
    require(all(case["qualified"] for cases in summary["static_link_comparisons"].values() for case in cases), "Static/link variants differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("preparation", "compile", "analysis", "reference", "cases", "executable", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-5)
    parser.add_argument("--relative-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    require(args.timeout > 0 and math.isfinite(args.timeout) and all(math.isfinite(x) and x >= 0 for x in
            (args.absolute_tolerance, args.relative_tolerance)), "Invalid timeout/tolerances")
    for name in experiment.LOCK_NAMES:
        require(not (ROOT / "build" / name).exists(), "Existing GPU experiment lock; inspect its owner")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = ROOT / "build/compute-pipeline-experiment.lock"
    summary = dict(schema_version=1, status="running", runs=[])
    with lock.open("x", encoding="utf-8") as file:
        json.dump(dict(pid=os.getpid(), output=str(output)), file)
    try:
        run(args, output, summary)
        summary["status"] = "passed"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "summary.json", summary)
        lock.unlink()
    experiment.seal(output)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
