import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import MeasureCLASTemplateShadow as runner


def rows(blocks=3, candidate_shadow=.99, candidate_clas=.19, candidate_graph=4.7):
    values = []
    for index, arm in enumerate(runner.BLOCK * blocks):
        scale = 1 + index // 6 * .2
        times = {"clasMs": {"A": 1, "B": .2, "C": candidate_clas}[arm],
                 "shadowsMs": {"A": 1, "B": 1.1, "C": candidate_shadow}[arm],
                 "graphMs": {"A": 5, "B": 4.8, "C": candidate_graph}[arm]}
        for metric in runner.METRICS:
            times.setdefault(metric, {"A": 2, "B": 1.9, "C": 1.8}[arm])
        values.append({"arm": arm, "stage": "confirmation", "directory": str(index),
                       "process": {"pid": index + 100, "startedUnix": index * 2, "endedUnix": index * 2 + 1},
                       "result": {"timings": {m: {"median": v * scale} for m, v in times.items()}}})
    return values


class CLASTemplateShadowTests(unittest.TestCase):
    def make_fixture(self, root, native=True):
        def baseline_prepare(args):
            args.output.mkdir()
            config = {"version": 2, "frames": 97, "warmupFrames": 32, "timingFrames": 64, "assetFiles": [],
                      "cases": [{"id": "CLASTessellation", "graph": "Graph.json", "width": 1444, "height": 694}]}
            for name in ("triangles", "templates"):
                directory = args.output / name
                directory.mkdir()
                graph = {"view": {"renderResolution": {"width": 1444, "height": 694, "adaptive": False}},
                         "nodes": [{"name": "VBuffer", "properties": {"rtTessellationBuildMode": name,
                                     "rtTessellation": True, "rtTessellationMaxFactor": 8}}]}
                runner.save(directory / "Graph.json", graph)
                runner.save(directory / "Cases.json", config)
            runner.save(args.output / "Fixture.json", {"protocol": runner.base.PROTOCOL, "scope": "test"})
        args = SimpleNamespace(output=root, template_shape="right", instantiation_flags="fastTrace",
                               template_bounds="geometry", native_sizes=native)
        with patch.object(runner.base, "prepare", baseline_prepare):
            runner.prepare(args)

    def test_diagnostic_graph_is_separate_and_ordinary_size_writes_are_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            self.make_fixture(root)
            config, diagnostics, graphs, diagnostic_graphs = runner.fixture_configs(root)
            self.assertNotIn("captureTessellationNativeSizes", config["cases"][0])
            for arm in runner.ARMS:
                self.assertFalse(runner.properties(graphs[arm])["benchmarkCLASBuildSizes"])
                self.assertTrue(runner.properties(diagnostic_graphs[arm])["benchmarkCLASBuildSizes"])
                self.assertTrue(diagnostics[arm]["cases"][0]["captureTessellationNativeSizes"])
            changed = graphs["C"]
            runner.properties(changed)["rtTessellationMaxFactor"] = 4
            runner.save(root / "candidate/Graph.json", changed)
            with self.assertRaisesRegex(ValueError, "beyond the predeclared"):
                runner.fixture_configs(root)

    def test_combined_unregistered_strategy_fails_before_creating_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            args = SimpleNamespace(output=root, template_shape="equilateral", instantiation_flags="none",
                                   template_bounds="geometry", native_sizes=True)
            with self.assertRaisesRegex(ValueError, "Only C3"):
                runner.prepare(args)
            self.assertFalse(root.exists())

    def test_legacy_v2_fixture_remains_valid_without_bounds_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            self.make_fixture(root)
            fixture = runner.read(root / "Fixture.json")
            fixture.update(protocol=runner.LEGACY_PROTOCOL, candidateId="C1", candidate=runner.CANDIDATES["C1"],
                           baselineStrategy=runner.DEFAULT_STRATEGY)
            runner.save(root / "Fixture.json", fixture)
            for arm, name in runner.ARMS.items():
                for filename in ("Graph.json", "DiagnosticGraph.json"):
                    path = root / name / filename
                    graph = runner.read(path)
                    p = runner.properties(graph)
                    del p[runner.BOUNDS_KEY]
                    p.update(runner.CANDIDATES["C1"] if arm == "C" else runner.DEFAULT_STRATEGY)
                    runner.save(path, graph)
            runner.fixture_configs(root)

    def bounded_case(self):
        identity = {"templateBoundsMode": "geometry", "templateBounds": [-1, -1, -.1, 1, 1, .1],
                    "templatePoolGeneration": 7, "nativeTemplateCount": 1024, "boundedTemplatesReady": True}
        profile = {"streaming": [{"passName": "VBuffer", "tessellationIdentity": copy.deepcopy(identity)}]}
        case = {"templateBoundFallbacks": 0, "templateBoundFallbackObserved": False, "templatePoolGeneration": 7,
                "warmup": [copy.deepcopy(profile)], "frames": [copy.deepcopy(profile)]}
        graph = {"nodes": [{"name": "VBuffer", "properties": {runner.BOUNDS_KEY: "geometry"}}]}
        return case, identity, graph

    def test_sticky_fallback_rejects_even_when_last_frame_has_none(self):
        case, identity, graph = self.bounded_case()
        runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)
        case["templateBoundFallbackObserved"] = True
        with self.assertRaisesRegex(ValueError, "exclusively"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)

    def test_replacement_cannot_clear_sticky_window_proof(self):
        case, identity, graph = self.bounded_case()
        case["warmup"][0]["streaming"][0]["tessellationIdentity"]["templatePoolGeneration"] = 6
        with self.assertRaisesRegex(ValueError, "replaced"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)
        case, identity, graph = self.bounded_case()
        case["templatePoolGeneration"] = 8
        with self.assertRaisesRegex(ValueError, "different template allocation"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)

    def test_native_bounds_are_actual_f32_bytes_and_must_match_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case, identity, graph = self.bounded_case()
            data = struct.pack("<6f", *identity["templateBounds"])
            (root / "bounds.bin").write_bytes(data)
            case["nativeCLASBoundsReceipt"] = {"file": "bounds.bin", "bytes": 24,
                                               "values": list(struct.unpack("<6f", data))}
            value = runner.bounded_path_evidence(root, case, identity, graph, "C", True)
            self.assertEqual(value["nativeBounds"]["bytes"], 24)
            identity["templateBounds"][0] = -2
            with self.assertRaisesRegex(ValueError, "disagrees"):
                runner.bounded_path_evidence(root, case, identity, graph, "C", True)

    def test_bounds_readback_cannot_enter_ordinary_timing(self):
        case, identity, graph = self.bounded_case()
        case["nativeCLASBoundsReceipt"] = {"file": "unexpected.bin"}
        with self.assertRaisesRegex(ValueError, "leaked"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)

    def test_missing_bounded_template_half_is_invalid(self):
        case, identity, graph = self.bounded_case()
        identity["nativeTemplateCount"] = 512
        with self.assertRaisesRegex(ValueError, "population"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False)

    def test_bound_size_reduction_is_diagnostic_and_does_not_change_policy(self):
        samples = [{"arm": arm, "result": {"nativeCLASSizes": {"count": 2, "totalBytes": total,
                    "minimum": total // 2, "maximum": total // 2, "median": total / 2}}}
                   for arm, total in zip("ABC", (3200, 5120, 4096))]
        value = runner.bound_size_diagnostic(samples)
        self.assertEqual(value["classification"], "reduced")
        self.assertAlmostEqual(value["reductionFraction"], .2)
        self.assertFalse(value["requiredForPerformanceAcceptance"])
        self.assertEqual(len(runner.POLICY["gates"]), 5)
        samples[2]["result"]["nativeCLASSizes"]["totalBytes"] = 5120
        self.assertEqual(runner.bound_size_diagnostic(samples)["classification"], "unchanged")

    def test_each_contrast_uses_one_observation_per_six_process_block(self):
        result = runner.assess(rows(5), "confirmation")
        self.assertEqual(result["decision"], "accept")
        for pair in runner.PAIRS:
            for metric in runner.METRICS:
                self.assertEqual(result["contrasts"][pair][metric]["blocks"], 5)
                self.assertEqual(len(result["contrasts"][pair][metric]["logRatios"]), 5)
        self.assertAlmostEqual(result["contrasts"]["C/A"]["clasMs"]["estimate"], .81)

    def test_discovery_cannot_accept(self):
        result = runner.assess(rows(), "discovery")
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["nextAction"], "run_confirmation")

    def test_shadow_noninferiority_is_mandatory(self):
        result = runner.assess(rows(5, candidate_shadow=1.03), "confirmation")
        self.assertEqual(result["decision"], "reject")
        self.assertTrue(next(g for g in result["gates"] if g["pair"] == "C/A" and g["metric"] == "shadowsMs")["intervalFail"])

    def test_shadow_must_improve_original_templates(self):
        result = runner.assess(rows(5, candidate_shadow=1.11), "confirmation")
        self.assertEqual(result["decision"], "reject")
        self.assertTrue(next(g for g in result["gates"] if g["pair"] == "C/B" and g["metric"] == "shadowsMs")["intervalFail"])

    def test_original_clas_gain_must_be_retained(self):
        result = runner.assess(rows(5, candidate_clas=.22), "confirmation")
        self.assertEqual(result["decision"], "reject")
        self.assertTrue(next(g for g in result["gates"] if g["pair"] == "C/B" and g["metric"] == "clasMs")["intervalFail"])

    def test_triangle_clas_gain_floor_is_fifty_percent(self):
        changed = rows(5, candidate_clas=.6)
        for row in changed:
            if row["arm"] == "B":
                row["result"]["timings"]["clasMs"]["median"] *= 3
        result = runner.assess(changed, "confirmation")
        self.assertEqual(result["decision"], "reject")
        self.assertTrue(next(g for g in result["gates"] if g["pair"] == "C/A" and g["metric"] == "clasMs")["intervalFail"])

    def test_graph_regression_is_mandatory(self):
        self.assertEqual(runner.assess(rows(5, candidate_graph=5.2), "confirmation")["decision"], "reject")

    def test_incomplete_or_wrong_order_is_invalid(self):
        with self.assertRaisesRegex(ValueError, "ABCCBA"):
            runner.assess(rows()[:-1], "discovery")
        changed = rows()
        changed[0]["arm"] = "B"
        with self.assertRaisesRegex(ValueError, "ABCCBA"):
            runner.assess(changed, "discovery")

    def test_process_overlap_is_invalid(self):
        changed = rows()
        changed[1]["process"]["startedUnix"] = .5
        with self.assertRaisesRegex(ValueError, "overlapping"):
            runner.assess(changed, "discovery")

    def test_noise_warning_does_not_discard_blocks(self):
        result = runner.assess(rows(5), "confirmation")
        self.assertTrue(result["noiseWarning"])
        self.assertEqual(result["decision"], "accept")

    def test_only_strategy_fields_are_normalized(self):
        graph = {"nodes": [{"name": "VBuffer", "properties": {"rtTessellationBuildMode": "templates",
            **runner.DEFAULT_STRATEGY, "rtTessellationMaxFactor": 8}}]}
        candidate = copy.deepcopy(graph)
        runner.properties(candidate).update(runner.CANDIDATES["C1"])
        self.assertEqual(runner.normalize_graph(graph), runner.normalize_graph(candidate))
        runner.properties(candidate)["rtTessellationMaxFactor"] = 4
        self.assertNotEqual(runner.normalize_graph(graph), runner.normalize_graph(candidate))

    def test_native_size_summary_recomputed_from_u32_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "sizes.bin").write_bytes(struct.pack("<III", 64, 96, 128))
            receipt = {"file": "sizes.bin", "count": 3, "bytes": 12, "totalBytes": 288, "minBytes": 64, "maxBytes": 128}
            case = {"nativeCLASSizeReceipt": receipt, "tessellationCounts": [3, 1, 0, 0]}
            self.assertEqual(runner.native_sizes(root, case)["median"], 96)
            receipt["totalBytes"] += 1
            with self.assertRaisesRegex(ValueError, "summary"):
                runner.native_sizes(root, case)

    def test_diagnostic_identity_exemption_is_only_size_instrumentation(self):
        diagnostics = []
        for arm in "ABC":
            diagnostics.append({"arm": arm, "stage": "diagnostic", "recipes": [
                {"module": runner.MODULES[0], "sha256": "tess", "device": {"gpu": "fixed"}},
                {"module": runner.MODULES[1], "sha256": "shadow", "device": {"gpu": "fixed"}}],
                "result": {"identity": {"mode": runner.MODES[arm], "diagnosticBuildSizes": True},
                           "inputs": {"geometry": "same"}, "outputs": {"HDR": "same"}}})
        ordinary = copy.deepcopy(diagnostics[2])
        ordinary["stage"] = "discovery"
        ordinary["result"]["identity"]["diagnosticBuildSizes"] = False
        runner.validate_results([ordinary], diagnostics)
        ordinary["result"]["identity"]["shaderHash"] = "changed"
        with self.assertRaisesRegex(ValueError, "Within-arm"):
            runner.validate_results([ordinary], diagnostics)

    def test_shadow_shader_must_match_all_three_arms(self):
        diagnostics = [{"arm": a, "recipes": [{"module": runner.MODULES[1], "sha256": a, "device": {}}],
                        "result": {"inputs": {}, "outputs": {}}} for a in "ABC"]
        with self.assertRaisesRegex(ValueError, "Shadow SPIR-V"):
            runner.validate_results([], diagnostics)


if __name__ == "__main__":
    unittest.main()
