"""Synthetic evidence tests; these never compile shaders or launch GPU work."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import PrimaryGuidesSplit as split


def graph(enabled=False, legacy=False):
    return dict(nodes=[dict(name="PathTrace", type="ScenePathTracePass", properties=dict(
        bsdf="gltf" if legacy else "openpbr", exportDenoiserGuides=not legacy,
        samples=1, cacheMode="off", maxDepth=12, splitPrimaryGuides=enabled))])


def native(arm, duration):
    return dict(status="passed", validation_errors=0, build_configuration="Release", slang_build="test",
        shader_target=dict(id="vulkan"), device=dict(name="test", driver=1),
        configuration=dict(debug_mode="disabled", validation=False, pipeline_cache="empty", compile_only=False,
                           compute_backend="pipeline", disable_driver_optimization=False),
        environment={split.driver.CACHE_ENVIRONMENT: "disabled"},
        samples=[dict(status="passed", pipeline_status="created", shader_cache_hit=True,
                      request=dict(module=split.NATIVE_ROLES[arm][0], entry=split.NATIVE_ROLES[arm][1],
                                   defines=[["METALLIC_PT_SINGLE_SAMPLE", "1"]]), spirv_bytes=100, spirv_fnv1a64=arm,
                      effective_descriptor_mode="mapped", resource_heap_stride=0, sampler_heap_stride=0,
                      effective_slang_optimization="O1", pipeline=dict(native_create_ms=duration, cache_lock_wait_ms=0,
                      pso_hash=arm, native_cache_hit=False))])


class SplitEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def reports(self):
        paths = []
        for enabled, arm in enumerate(("fused", "split")):
            folder = self.root / arm
            folder.mkdir(exist_ok=True)
            (folder / "color.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 1))
            (folder / "guide.bin").write_bytes(struct.pack("<f", .5))
            sections = [dict(name="Path trace shading", gpuMs=2.)]
            if enabled:
                sections += [dict(name=name, gpuMs=.5) for name in split.STAGES]
            case = dict(id="Opaque1", resolvedGraph=graph(bool(enabled)), format="RGBA32F", width=1, height=1,
                        image="color.bin", output="PathTrace.color", validHDR=True,
                        requiredTiming=["PathTrace", "Path continuation" if enabled else "Path trace shading"],
                        captureOutputs=["PathTrace.linearDepth"],
                        captures=[dict(output="PathTrace.linearDepth", frame=3, file="guide.bin", bytes=4, format=1)],
                        frames=[dict(frame=1, executionId=1, warmup=False, graphMs=3., asyncComputeBranches=0,
                                     nodes=[dict(name="PathTrace", queue=0, gpuMs=2., sections=sections)])])
            path = folder / "MaterialBaseline.json"
            split.save(path, dict(validation=True, cases=[case]))
            paths.append(path)
        return paths

    def test_compare_allows_only_split_policy_and_keeps_sources(self):
        paths = self.reports()
        before = [p.read_bytes() for p in paths]
        result = split.compare(*paths)
        self.assertTrue(result["differences"][0]["capturesByteIdentical"])
        self.assertEqual([p.read_bytes() for p in paths], before)
        self.assertEqual(len(result["timings"]), 2)

    def test_camera_or_path_settings_change_rejected(self):
        paths = self.reports()
        report = split.read(paths[1])
        report["cases"][0]["resolvedGraph"]["nodes"][0]["properties"]["maxDepth"] = 2
        split.save(paths[1], report)
        with self.assertRaisesRegex(ValueError, "beyond"):
            split.compare(*paths)

    def test_fallback_cannot_claim_split_execution(self):
        paths = self.reports()
        report = split.read(paths[1])
        report["cases"][0]["frames"][0]["nodes"][0]["sections"] = []
        split.save(paths[1], report)
        with self.assertRaisesRegex(ValueError, "dispatch scopes"):
            split.compare(*paths)

    def test_unknown_workload_metadata_change_rejected(self):
        paths = self.reports()
        report = split.read(paths[1])
        report["cases"][0]["futureLightingOverride"] = 1
        split.save(paths[1], report)
        with self.assertRaisesRegex(ValueError, "metadata"):
            split.compare(*paths)

    def test_timing_policy_change_rejected(self):
        paths = self.reports()
        report = split.read(paths[1])
        report["cases"][0]["requiredTiming"] = ["PathTrace", "Primary guides"]
        split.save(paths[1], report)
        with self.assertRaisesRegex(ValueError, "requiredTiming"):
            split.compare(*paths)

    def test_changed_guide_and_nonfinite_hdr_rejected(self):
        paths = self.reports()
        (paths[1].parent / "guide.bin").write_bytes(struct.pack("<f", .6))
        with self.assertRaisesRegex(ValueError, "Guide changed"):
            split.compare(*paths)
        (paths[1].parent / "guide.bin").write_bytes(struct.pack("<f", .5))
        (paths[1].parent / "color.bin").write_bytes(struct.pack("<4f", float("nan"), 2, 3, 1))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            split.compare(*paths)

    def test_unsupported_settings_rejected(self):
        for key, value in (("samples", 2), ("cacheMode", "sharc"), ("splitPrimaryGuides", 1)):
            candidate = graph(True)
            candidate["nodes"][0]["properties"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                split.normalize_graph(candidate, split=True)

    def test_prepare_preserves_temporal_regression_and_control(self):
        source = self.root / "fixtures"
        source.mkdir()
        cases = []
        for case_id in split.CASE_IDS:
            split.save(source / (case_id + ".json"), graph(legacy=case_id == "LegacyShadow"))
            case = dict(id=case_id, graph=case_id + ".json", width=128, height=128)
            if case_id == "ABeautifulGame":
                case.update(frameCount=97, captureFrames=[93, 94, 95, 96])
            cases.append(case)
        split.save(source / "Correctness.json", dict(frames=16, warmupFrames=4, timingFrames=4, cases=cases))
        paths = split.prepare(self.root / "prepared", source, quick=True)
        for arm, path in paths.items():
            config = split.read(path)
            self.assertEqual(config["cases"][-1]["captureFrames"][-1], 96)
            for case in config["cases"]:
                self.assertEqual(case["width"], 64)
                props = split.read(Path(path).parent / case["graph"])["nodes"][0]["properties"]
                self.assertEqual(props[split.PROPERTY], arm == "split" and case["id"] != "LegacyShadow")

    def groups(self):
        groups = []
        for repeat, durations in enumerate(((100, 20, 90), (110, 21, 95))):
            group = []
            for arm, duration in zip(("fused", "primary", "continuation"), durations):
                path = self.root / f"{repeat}-{arm}.json"
                split.save(path, native(arm, duration))
                group.append(path)
            groups.append(group)
        return groups

    def test_native_aggregates_per_repeat_max_and_sum(self):
        result = split.aggregate_native(self.groups())
        self.assertEqual(result["summary"]["split_max_ms"]["median"], 92.5)
        self.assertEqual(result["summary"]["split_sum_ms"]["median"], 113)
        self.assertEqual(result["summary"]["fused"]["median"], 105)

    def test_native_rejects_device_shader_and_cache_contamination(self):
        for change in (lambda r: r["device"].update(driver=2),
                       lambda r: r["samples"][0].update(spirv_fnv1a64="changed"),
                       lambda r: r["samples"][0].update(shader_cache_hit=False),
                       lambda r: r["samples"][0]["pipeline"].update(native_cache_hit=True),
                       lambda r: r["configuration"].update(pipeline_cache="reuse"),
                       lambda r: r["environment"].update(METALLIC_VK_INTERNAL_PIPELINE_CACHE="enabled")):
            groups = self.groups()
            report = split.read(groups[1][1])
            change(report)
            split.save(groups[1][1], report)
            with self.assertRaises(ValueError):
                split.aggregate_native(groups)

    def test_native_rejects_duplicate_process_report(self):
        groups = self.groups()
        with self.assertRaisesRegex(ValueError, "reuse"):
            split.aggregate_native([groups[0], groups[0]])

    def test_native_rejects_swapped_roles_even_with_stable_identities(self):
        groups = self.groups()
        for group in groups:
            group[0], group[2] = group[2], group[0]
        with self.assertRaisesRegex(ValueError, "kernel role"):
            split.aggregate_native(groups)

    def test_native_rejects_missing_or_ambiguous_single_macro(self):
        for defines in ([], [["METALLIC_PT_SINGLE_SAMPLE", "0"]],
                        [["METALLIC_PT_SINGLE_SAMPLE", "1"], ["METALLIC_PT_SINGLE_SAMPLE", "1"]]):
            sample = native("fused", 1)["samples"][0]
            sample["request"]["defines"] = defines
            with self.assertRaisesRegex(ValueError, "Single=1"):
                split.validate_native_role(sample, "fused")

    def test_recipe_role_uses_source_entry_and_rejects_ambiguity(self):
        sample = native("fused", 1)["samples"][0]
        sample["request"].pop("defines")
        sample.update(shader_source="recipe_spirv", pipeline_type="compute", shader_stages=[dict(
            stage="compute", module=split.NATIVE_ROLES["fused"][0], entry="main",
            source_entry=split.NATIVE_ROLES["fused"][1], source_identity_ambiguous=False)])
        self.assertIn("not recorded", split.validate_native_role(sample, "fused"))
        for field, value in (("source_entry", "main"), ("source_identity_ambiguous", True),
                             ("module", split.NATIVE_ROLES["continuation"][0])):
            old = sample["shader_stages"][0][field]
            sample["shader_stages"][0][field] = value
            with self.assertRaisesRegex(ValueError, "recipe kernel role"):
                split.validate_native_role(sample, "fused")
            sample["shader_stages"][0][field] = old


if __name__ == "__main__":
    unittest.main()
