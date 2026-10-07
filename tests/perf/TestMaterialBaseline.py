import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/Perf"))
import MaterialBaseline as baseline


class MaterialBaselineIdentityTests(unittest.TestCase):
    def test_identity_does_not_require_access_to_optional_submodule_worktrees(self):
        def git(command, **kwargs):
            if command[1] == "status" and "--ignore-submodules=all" not in command:
                raise baseline.subprocess.CalledProcessError(128, command, stderr="External/ImGuizmo: Permission denied")
            return {"rev-parse": "commit", "status": " M Shaders/Glass.slang", "ls-files": "160000 gitlink 0\tExternal/ImGuizmo"}[command[1]]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("Shaders", "Source", "tests/rhi"):
                (root / name).mkdir(parents=True)
            with patch.object(baseline, "ROOT", root), patch.object(baseline, "__file__", str(root / "Tools/Perf/MaterialBaseline.py")), \
                    patch.object(baseline, "asset_paths", return_value=[]), \
                    patch.object(baseline, "digest", return_value="hash"), \
                    patch.object(baseline.platform, "platform", return_value="test-platform"), \
                    patch.object(baseline.subprocess, "check_output", side_effect=git):
                result = baseline.identity(root / "tests/MetallicRHITests.exe")
        self.assertEqual(result["status"], "M Shaders/Glass.slang")
        self.assertIn("External/ImGuizmo", result["submodules"])
        self.assertIn("not inspected", result["statusScope"])
        self.assertTrue(result["binaries"])
        self.assertTrue(result["sources"])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional baseline tooling requires numpy")
class MaterialBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixtures").mkdir()
        cases = []
        for name, node, scope in [("OpenPBRPathTrace", "Reference", "Path trace shading"),
                                  ("OpenPBRDeferred", "Deferred", "Material classification"),
                                  ("RTXCRChiang", "PathTrace", "Path trace shading")]:
            cases.append({"id": name, "width": 1, "height": 1, "format": "RGBA32F", "image": name + ".rgba32f",
                          "frames": [{"frame": f, "executionId": f + 1, "graphMs": 1.0,
                                      "nodes": [{"name": node, "gpuMs": 0.8,
                                                 "sections": [{"name": scope, "gpuMs": 0.7}]}]} for f in range(32, 96)]})
        baseline.save(self.root / "fixtures/Cases.json", {"cases": [{"id": c["id"], "width": 1, "height": 1} for c in cases]})
        baseline.save(self.root / "Process.json", {"exitCodes": [0, 0, 0]})
        for index in range(3):
            run = self.root / f"run-{index}"
            run.mkdir()
            baseline.save(run / "MaterialBaseline.json", {"validation": False, "cases": cases})
            for case in cases:
                (run / case["image"]).write_bytes(struct.pack("<4f", 0.2, 0.4, 0.6, 1))
        baseline.seal(self.root)

    def mutate(self, action):
        path = self.root / "run-0/MaterialBaseline.json"
        report = baseline.read(path)
        action(report)
        baseline.save(path, report)
        baseline.seal(self.root)

    def test_harness_archive_copies_are_not_additional_processes(self):
        target = self.root / "run-0/reports/cases/0"
        target.mkdir(parents=True)
        baseline.save(target / "MaterialBaseline.json", {})
        baseline.seal(self.root)
        self.assertTrue(baseline.verify(self.root)["verified"])

    def test_changed_file_rejected(self):
        (self.root / "run-0/OpenPBRDeferred.rgba32f").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            baseline.verify(self.root)

    def test_missing_frames_rejected_even_with_valid_hashes(self):
        self.mutate(lambda r: r["cases"][0]["frames"].pop())
        with self.assertRaisesRegex(ValueError, "timing frames"):
            baseline.verify(self.root)

    def test_diagnostic_timings_rejected(self):
        self.mutate(lambda r: r.update(validation=True))
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            baseline.verify(self.root)

    def test_absent_scope_not_zero(self):
        self.mutate(lambda r: r["cases"][1]["frames"][0]["nodes"][0].update(sections=[]))
        with self.assertRaisesRegex(ValueError, "GPU scope"):
            baseline.verify(self.root)

    def test_case_configuration_drift_rejected(self):
        self.mutate(lambda r: r["cases"][0].update(width=2))
        with self.assertRaisesRegex(ValueError, "fixed fixture"):
            baseline.verify(self.root)

    def test_nonfinite_hdr_rejected(self):
        (self.root / "run-0/OpenPBRDeferred.rgba32f").write_bytes(struct.pack("<4f", float("nan"), 0, 0, 1))
        baseline.seal(self.root)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            baseline.verify(self.root)

    def test_custom_case_count_and_window(self):
        config = baseline.read(self.root / "fixtures/Cases.json")
        config.update(warmupFrames=3, timingFrames=4)
        config["cases"] = [dict(id="Custom", width=1, height=1, requiredTiming=["Reference", None])]
        baseline.save(self.root / "fixtures/Cases.json", config)
        for index in range(3):
            path = self.root / f"run-{index}/MaterialBaseline.json"
            report = baseline.read(path)
            case = report["cases"][0]
            case.update(id="Custom", requiredTiming=["Reference", None])
            case["frames"] = case["frames"][:4]
            for frame, data in enumerate(case["frames"], 3):
                data["frame"] = frame
            report["cases"] = [case]
            baseline.save(path, report)
        baseline.seal(self.root)
        self.assertTrue(baseline.verify(self.root)["verified"])
        self.mutate(lambda r: r["cases"][0]["frames"][0]["nodes"][0].update(gpuMs=-1))
        with self.assertRaisesRegex(ValueError, "Invalid material pass time"):
            baseline.verify(self.root)

    def test_invalid_catalog_hdr_is_preserved_and_excluded(self):
        config = baseline.read(self.root / "fixtures/Cases.json")
        config["version"] = 2
        for spec in config["cases"]:
            spec["backend"] = "Deferred" if spec["id"] == "OpenPBRDeferred" else "PT"
        baseline.save(self.root / "fixtures/Cases.json", config)
        for index in range(3):
            path = self.root / f"run-{index}/MaterialBaseline.json"
            report = baseline.read(path)
            for case in report["cases"]:
                case.update(validHDR=True, nonfiniteComponents=0, resolvedGraph={},
                            environmentTransitions=[dict(frame=0, mapAvailable=True)],
                            backend="Deferred" if case["id"] == "OpenPBRDeferred" else "PT")
                for frame in case["frames"]:
                    frame["asyncComputeBranches"] = 2 if case["backend"] == "Deferred" else 0
                    for node in frame["nodes"]:
                        node["queue"] = 0
            if index == 0:
                report["cases"][0].update(validHDR=False, nonfiniteComponents=1)
                (path.parent / report["cases"][0]["image"]).write_bytes(struct.pack("<4f", float("nan"), 0, 0, 1))
            baseline.save(path, report)
        baseline.seal(self.root)
        result = baseline.verify(self.root)
        self.assertEqual(result["invalidHDR"], {"OpenPBRPathTrace": {"run-0": 1}})
        self.assertNotIn("OpenPBRPathTrace", result["imageAA"])
        baseline.save(self.root / "Identity.json", {"workload": {}})
        (self.root / "GPU.csv").write_text("name,uuid,driver\nGPU,42,1\n")
        baseline.seal(self.root)
        comparison = baseline.compare(self.root, self.root)
        self.assertEqual(comparison["excludedInvalidHDR"], ["OpenPBRPathTrace"])
        self.assertEqual(set(comparison["cases"]), {"OpenPBRDeferred", "RTXCRChiang"})
        self.mutate(lambda r: r["cases"][0].update(validHDR=True))
        with self.assertRaisesRegex(ValueError, "HDR quality metadata mismatch"):
            baseline.verify(self.root)
        self.mutate(lambda r: r["cases"][0].update(validHDR=False))
        self.mutate(lambda r: r["cases"][1]["frames"][0].update(asyncComputeBranches=1))
        with self.assertRaisesRegex(ValueError, "asynchronous timing scope"):
            baseline.verify(self.root)


if __name__ == "__main__":
    unittest.main()
