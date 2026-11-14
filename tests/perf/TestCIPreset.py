"""CPU CI preset policy and selected-test executable closure (no tests executed)."""
import argparse
import copy
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_LABELS = ("scene", "task", "debug", "build", "tools", "skip-policy")
REQUIRED_TARGETS = ("MetallicSceneTests", "MetallicTaskTests", "MetallicDebugTests",
                    "MetallicCtl", "MetallicRHITests", "MetallicShaderPipelineBench")
REQUIRED_CHECKS = ("MetallicShaderModuleOwnershipAudit", "MetallicShaderModuleOwnershipTests",
                   "MetallicShaderPipelineBenchTests", "MetallicCPUWorkflowTests")
BUILD_DIR = None


def ci_preset(document, section):
    return next(preset for preset in document[section] if preset["name"] == "metallic-ci")


def selected_test(document, test):
    filters = ci_preset(document, "testPresets")["filter"]
    labels = next((prop["value"] for prop in test.get("properties", [])
                   if prop["name"] == "LABELS"), [])
    return (any(re.search(filters["include"]["label"], label) for label in labels)
            and not re.search(filters.get("exclude", {}).get("name", r"(?!)"), test["name"]))


def policy_errors(document):
    errors = []
    pattern = ci_preset(document, "testPresets")["filter"]["include"]["label"]
    for label in REQUIRED_LABELS:
        if not re.search(pattern, label):
            errors.append(f"missing label: {label}")
    for label in ("rhi", "editor", "testbench", "gpu"):
        if re.search(pattern, label):
            errors.append(f"unexpected GPU/editor label: {label}")
    if selected_test(document, {"name": "MetallicGPUPageTests", "properties": [
            {"name": "LABELS", "value": ["scene", "rhi"]}]}):
        errors.append("mixed CPU/GPU page bundle must stay outside the CPU lane")
    targets = ci_preset(document, "buildPresets")["targets"]
    errors.extend(f"missing target: {target}" for target in REQUIRED_TARGETS if target not in targets)
    variables = ci_preset(document, "configurePresets")["cacheVariables"]
    if variables.get("METALLIC_REQUIRE_PYTHON_TESTS") is not True:
        errors.append("CPU tooling requires Python")
    return errors


def missing_required_checks(document, test_model):
    selected = {test["name"] for test in test_model["tests"] if selected_test(document, test)}
    return [name for name in REQUIRED_CHECKS if name not in selected]


def missing_executable_targets(document, test_model):
    """Check generated commands, including CMake -D arguments holding test EXEs.

    Repository executable outputs use their target name (metallicctl differs
    only in case). CMake/Python and fixture-created tools are not build targets.
    """
    targets = {name.casefold() for name in ci_preset(document, "buildPresets")["targets"]}
    missing = []
    for test in test_model["tests"]:
        if not selected_test(document, test):
            continue
        if not test.get("command"):
            missing.append(f"{test['name']}: unavailable CTest command; build selected executables first")
            continue
        for index, argument in enumerate(test["command"]):
            key, separator, value = argument.partition("=")
            cmake_executable = key.startswith("-D") and key.endswith("EXECUTABLE") and bool(separator)
            executable_argument = index > 0 and test["command"][index - 1] == "--executable"
            if not (index == 0 or cmake_executable or executable_argument):
                continue
            path = Path(value if cmake_executable else argument)
            if (path.stem.casefold().startswith("metallic")
                    and path.stem.casefold() not in targets):
                missing.append(f"{test['name']}: {path.stem}")
    return missing


class CIPresetTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        self.valid = copy.deepcopy(self.document)
        ci_preset(self.valid, "buildPresets")["targets"] = list(REQUIRED_TARGETS)
        ci_preset(self.valid, "testPresets")["filter"]["include"]["label"] = "^(" + "|".join(REQUIRED_LABELS) + ")$"
        ci_preset(self.valid, "configurePresets")["cacheVariables"]["METALLIC_REQUIRE_PYTHON_TESTS"] = True
        ci_preset(self.valid, "testPresets")["filter"]["exclude"] = {"name": "^MetallicGPUPageTests$"}

    def test_repository_policy(self):
        self.assertEqual(policy_errors(self.document), [])

    def test_valid_policy(self):
        self.assertEqual(policy_errors(self.valid), [])

    def test_each_missing_target_is_rejected(self):
        for target in REQUIRED_TARGETS:
            with self.subTest(target=target):
                document = copy.deepcopy(self.valid)
                ci_preset(document, "buildPresets")["targets"].remove(target)
                self.assertIn(f"missing target: {target}", policy_errors(document))

    def test_each_missing_label_is_rejected(self):
        for label in REQUIRED_LABELS:
            with self.subTest(label=label):
                document = copy.deepcopy(self.valid)
                ci_preset(document, "testPresets")["filter"]["include"]["label"] = "^(" + "|".join(
                    item for item in REQUIRED_LABELS if item != label) + ")$"
                self.assertIn(f"missing label: {label}", policy_errors(document))

    def test_gpu_label_expansion_is_rejected(self):
        ci_preset(self.valid, "testPresets")["filter"]["include"]["label"] = ".*"
        self.assertIn("unexpected GPU/editor label: rhi", policy_errors(self.valid))

    def test_optional_python_is_rejected(self):
        ci_preset(self.valid, "configurePresets")["cacheVariables"]["METALLIC_REQUIRE_PYTHON_TESTS"] = False
        self.assertIn("CPU tooling requires Python", policy_errors(self.valid))

    def test_missing_command_is_rejected(self):
        model = {"tests": [{"name": "unbuilt", "properties": [
            {"name": "LABELS", "value": ["scene"]}]}]}
        self.assertEqual(missing_executable_targets(self.valid, model), [
            "unbuilt: unavailable CTest command; build selected executables first"])

    def test_gpu_scene_bundle_is_excluded(self):
        test = {"name": "MetallicGPUPageTests", "properties": [
            {"name": "LABELS", "value": ["scene", "rhi"]}]}
        self.assertFalse(selected_test(self.valid, test))
        del ci_preset(self.valid, "testPresets")["filter"]["exclude"]
        self.assertTrue(selected_test(self.valid, test))
        self.assertIn("mixed CPU/GPU page bundle must stay outside the CPU lane", policy_errors(self.valid))

    def test_generated_command_detection(self):
        model = {"tests": [
            {"name": "direct", "command": ["/build/MetallicSceneTests"],
             "properties": [{"name": "LABELS", "value": ["scene"]}]},
            {"name": "cmake-wrapper", "command": ["cmake", "-DMETALLIC_CTL_EXECUTABLE=/build/metallicctl.exe"],
             "properties": [{"name": "LABELS", "value": ["debug", "tools"]}]},
            {"name": "posix-cmake-wrapper", "command": ["cmake", "-DRHI_EXECUTABLE=/build/MetallicRHITests"],
             "properties": [{"name": "LABELS", "value": ["rhi", "tools", "cpu"]}]},
            {"name": "python-wrapper", "command": ["python", "TestBench.py", "--executable", "/build/MetallicShaderPipelineBench"],
             "properties": [{"name": "LABELS", "value": ["tools", "perf"]}]},
            {"name": "not-selected", "command": ["/build/MetallicEditor.exe"],
             "properties": [{"name": "LABELS", "value": ["editor"]}]},
        ]}
        self.assertEqual(missing_executable_targets(self.valid, model), [])
        for target, expected in (("MetallicCtl", "cmake-wrapper: metallicctl"),
                                 ("MetallicSceneTests", "direct: MetallicSceneTests"),
                                 ("MetallicRHITests", "posix-cmake-wrapper: MetallicRHITests"),
                                 ("MetallicShaderPipelineBench", "python-wrapper: MetallicShaderPipelineBench")):
            with self.subTest(target=target):
                document = copy.deepcopy(self.valid)
                ci_preset(document, "buildPresets")["targets"].remove(target)
                self.assertEqual(missing_executable_targets(document, model), [expected])

    def test_required_checks_must_be_registered_and_selected(self):
        model = {"tests": [{"name": name, "properties": [
            {"name": "LABELS", "value": ["tools", "shader"]}]} for name in REQUIRED_CHECKS]}
        self.assertEqual(missing_required_checks(self.valid, model), [])
        for name in REQUIRED_CHECKS:
            for mutation in ("unregistered", "unlabelled", "excluded"):
                with self.subTest(name=name, mutation=mutation):
                    document = copy.deepcopy(self.valid)
                    changed = copy.deepcopy(model)
                    test = next(test for test in changed["tests"] if test["name"] == name)
                    if mutation == "unregistered":
                        changed["tests"].remove(test)
                    elif mutation == "unlabelled":
                        test["properties"][0]["value"] = ["shader", "architecture", "cpu"]
                    else:
                        ci_preset(document, "testPresets")["filter"]["exclude"]["name"] += "|^" + name + "$"
                    self.assertEqual(missing_required_checks(document, changed), [name])

    def test_configured_ctest_executable_closure(self):
        if BUILD_DIR is None:
            self.skipTest("supply --build-dir for generated CTest command checks")
        result = subprocess.run(["ctest", "--test-dir", str(BUILD_DIR), "--show-only=json-v1"],
                                check=True, capture_output=True, text=True)
        model = json.loads(result.stdout)
        self.assertTrue(model["tests"], "configured tree must have registered tests")
        native_result = subprocess.run(
            ["ctest", "--preset", "metallic-ci", "--test-dir", str(BUILD_DIR.resolve()), "--show-only=json-v1"],
            cwd=ROOT, check=True, capture_output=True, text=True)
        native_tests = json.loads(native_result.stdout)["tests"]
        self.assertTrue(native_tests, "the CPU preset must select registered jobs")
        self.assertEqual({test["name"] for test in native_tests},
                         {test["name"] for test in model["tests"] if selected_test(self.document, test)})
        self.assertEqual(missing_required_checks(self.document, model), [])
        self.assertEqual(missing_executable_targets(self.document, model), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    arguments, remaining = parser.parse_known_args()
    BUILD_DIR = arguments.build_dir
    unittest.main(argv=[__file__, *remaining])
