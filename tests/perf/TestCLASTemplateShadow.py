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
    def make_fixture(self, root, native=True, protocol=runner.PROTOCOL):
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
        args = SimpleNamespace(output=root, template_shape="rightCorner", instantiation_flags="fastTrace",
                               template_bounds="none", native_sizes=native)
        with patch.object(runner.base, "prepare", baseline_prepare):
            runner.prepare(args)
        if protocol != runner.PROTOCOL:
            fixture = runner.read(root / "Fixture.json")
            _, default, candidates = runner.strategy_schema(protocol)
            candidate_id, candidate = next(iter(candidates.items()))
            fixture.update(protocol=protocol, candidateId=candidate_id, candidate=candidate, baselineStrategy=default)
            runner.save(root / "Fixture.json", fixture)
            for filename in ("Graph.json", "DiagnosticGraph.json"):
                path = root / "candidate" / filename
                graph = runner.read(path)
                runner.properties(graph).update(candidate)
                runner.save(path, graph)

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
            with self.assertRaisesRegex(ValueError, "Only C5"):
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

    def test_v4_only_changes_guide_and_declares_fixed_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            self.make_fixture(root, protocol=runner.DYADIC_PROTOCOL)
            fixture = runner.read(root / "Fixture.json")
            self.assertEqual(fixture["protocol"], runner.DYADIC_PROTOCOL)
            self.assertEqual(fixture["candidateId"], "C4")
            self.assertEqual(fixture["budgets"], {"discovery": 3, "confirmation": 5})
            _, _, graphs, _ = runner.fixture_configs(root)
            baseline, candidate = copy.deepcopy(graphs["B"]), copy.deepcopy(graphs["C"])
            self.assertEqual(runner.properties(candidate)[runner.BOUNDS_KEY], "none")
            runner.properties(candidate)[runner.STRATEGY_KEYS[0]] = "right"
            self.assertEqual(baseline, candidate)
            fixture["budgets"]["discovery"] = 4
            runner.save(root / "Fixture.json", fixture)
            with self.assertRaisesRegex(ValueError, "3 discovery"):
                runner.fixture_configs(root)

    def test_v3_fixture_keeps_original_bounded_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            self.make_fixture(root)
            fixture = runner.read(root / "Fixture.json")
            del fixture["budgets"]
            fixture.update(protocol=runner.BOUNDED_PROTOCOL, candidateId="C3", candidate=runner.BOUNDED_CANDIDATES["C3"])
            runner.save(root / "Fixture.json", fixture)
            for filename in ("Graph.json", "DiagnosticGraph.json"):
                path = root / "candidate" / filename
                graph = runner.read(path)
                runner.properties(graph).update(runner.BOUNDED_CANDIDATES["C3"])
                runner.save(path, graph)
            runner.fixture_configs(root)

    def test_v5_cannot_disable_native_size_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            with self.assertRaisesRegex(ValueError, "native size diagnostics"):
                self.make_fixture(root, native=False)
            self.assertFalse(root.exists())

    def test_v4_budget_is_fixed_without_changing_old_protocols(self):
        for protocol in (runner.DYADIC_PROTOCOL, runner.PROTOCOL):
            runner.validate_budgets({"discovery": 3, "confirmation": 5}, protocol)
            for budgets in ({"discovery": 4, "confirmation": 5}, {"discovery": 3, "confirmation": 6}):
                with self.subTest(protocol=protocol, budgets=budgets), self.assertRaisesRegex(ValueError, "3 discovery"):
                    runner.validate_budgets(budgets, protocol)
        for protocol in (runner.LEGACY_PROTOCOL, runner.BOUNDED_PROTOCOL):
            runner.validate_budgets({"discovery": 2, "confirmation": 10}, protocol)

    def test_v5_only_changes_corner_guide_bank_and_keeps_fixed_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            self.make_fixture(root)
            fixture = runner.read(root / "Fixture.json")
            self.assertEqual(fixture["protocol"], runner.PROTOCOL)
            self.assertEqual(fixture["candidateId"], "C5")
            self.assertEqual(fixture["candidate"], runner.CORNER_CANDIDATES["C5"])
            self.assertEqual(fixture["budgets"], {"discovery": 3, "confirmation": 5})
            _, _, graphs, _ = runner.fixture_configs(root)
            baseline, candidate = copy.deepcopy(graphs["B"]), copy.deepcopy(graphs["C"])
            runner.properties(candidate)[runner.STRATEGY_KEYS[0]] = "right"
            self.assertEqual(baseline, candidate)
            self.assertEqual(runner.properties(candidate)[runner.BOUNDS_KEY], "none")
            self.assertEqual(runner.properties(candidate)[runner.STRATEGY_KEYS[1]], "fastTrace")
            fixture["nativeSizeDiagnostic"] = False
            runner.save(root / "Fixture.json", fixture)
            with self.assertRaisesRegex(ValueError, "native size diagnostics"):
                runner.fixture_configs(root)

    def test_v5_rejects_other_candidate_shapes_bounds_and_flags(self):
        for shape, flags, bounds in (("right", "fastTrace", "none"), ("rightDyadic", "fastTrace", "none"),
                                    ("rightCorner", "none", "none"), ("rightCorner", "fastTrace", "geometry")):
            with self.subTest(shape=shape, flags=flags, bounds=bounds), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "fixture"
                args = SimpleNamespace(output=root, template_shape=shape, instantiation_flags=flags,
                                       template_bounds=bounds, native_sizes=True)
                with self.assertRaisesRegex(ValueError, "Only C5"):
                    runner.prepare(args)
                self.assertFalse(root.exists())

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

    def unbounded_case(self):
        case, identity, graph = self.bounded_case()
        identity.update(templateBoundsMode="none", templateBounds=None, boundedTemplatesReady=False,
                        nativeTemplateCount=512, patternCount=512)
        runner.properties(graph)[runner.BOUNDS_KEY] = "none"
        return case, identity, graph

    def test_v4_requires_unbounded_512_pool_without_bounds_receipt(self):
        case, identity, graph = self.unbounded_case()
        evidence = runner.bounded_path_evidence(Path("."), case, identity, graph, "C", True, runner.DYADIC_PROTOCOL)
        self.assertNotIn("nativeBounds", evidence)
        identity["nativeTemplateCount"] = 1024
        with self.assertRaisesRegex(ValueError, "population"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", True, runner.DYADIC_PROTOCOL)
        identity["nativeTemplateCount"] = 512
        case["nativeCLASBoundsReceipt"] = {"file": "unexpected.bin"}
        with self.assertRaisesRegex(ValueError, "unexpectedly"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", True, runner.DYADIC_PROTOCOL)

    def test_v4_still_requires_zero_fallback_and_continuous_generation(self):
        for field, value in (("templateBoundFallbacks", 1), ("templateBoundFallbackObserved", True)):
            case, identity, graph = self.unbounded_case()
            case[field] = value
            with self.assertRaisesRegex(ValueError, "fallback"):
                runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False, runner.DYADIC_PROTOCOL)
        case, identity, graph = self.unbounded_case()
        case["warmup"][0]["streaming"][0]["tessellationIdentity"]["templatePoolGeneration"] = 6
        with self.assertRaisesRegex(ValueError, "replaced"):
            runner.bounded_path_evidence(Path("."), case, identity, graph, "C", False, runner.DYADIC_PROTOCOL)

    def corner_case(self, arm):
        case, identity, graph = self.unbounded_case()
        identity.update(templateVariantCount={"A": 0, "B": 1, "C": 3}[arm],
                        nativeTemplateCount={"A": 0, "B": 512, "C": 1536}[arm],
                        patternCount=0 if arm == "A" else 512,
                        templatePoolGeneration=0 if arm == "A" else 7)
        case["templatePoolGeneration"] = identity["templatePoolGeneration"]
        for frame in [*case["warmup"], *case["frames"]]:
            frame["streaming"][0]["tessellationIdentity"] = copy.deepcopy(identity)
        return case, identity, graph

    def test_v5_checks_each_arms_integer_pattern_native_and_variant_counts(self):
        for arm in "ABC":
            case, identity, graph = self.corner_case(arm)
            runner.bounded_path_evidence(Path("."), case, identity, graph, arm, True, runner.PROTOCOL)
            for key in ("patternCount", "nativeTemplateCount", "templateVariantCount"):
                expected = identity[key]
                for invalid in (float(expected), bool(expected), str(expected), expected + 1):
                    with self.subTest(arm=arm, key=key, invalid=invalid), self.assertRaises(ValueError):
                        runner.bounded_path_evidence(Path("."), case, {**identity, key: invalid}, graph, arm, True, runner.PROTOCOL)
                missing = dict(identity)
                del missing[key]
                with self.subTest(arm=arm, missing=key), self.assertRaises(KeyError):
                    runner.bounded_path_evidence(Path("."), case, missing, graph, arm, True, runner.PROTOCOL)

    def test_v5_still_rejects_bounds_fallback_and_generation_changes(self):
        for arm in "ABC":
            for field, value in (("templateBoundFallbacks", 1), ("templateBoundFallbackObserved", True)):
                case, identity, graph = self.corner_case(arm)
                case[field] = value
                with self.subTest(arm=arm, field=field), self.assertRaisesRegex(ValueError, "fallback"):
                    runner.bounded_path_evidence(Path("."), case, identity, graph, arm, False, runner.PROTOCOL)
            case, identity, graph = self.corner_case(arm)
            case["nativeCLASBoundsReceipt"] = {"file": "unexpected.bin"}
            with self.assertRaisesRegex(ValueError, "unexpectedly"):
                runner.bounded_path_evidence(Path("."), case, identity, graph, arm, True, runner.PROTOCOL)
            case, identity, graph = self.corner_case(arm)
            case["warmup"][0]["streaming"][0]["tessellationIdentity"]["templatePoolGeneration"] += 1
            with self.assertRaisesRegex(ValueError, "replaced"):
                runner.bounded_path_evidence(Path("."), case, identity, graph, arm, False, runner.PROTOCOL)

    def test_bound_size_reduction_is_diagnostic_and_does_not_change_policy(self):
        samples = [{"arm": arm, "result": {"nativeCLASSizes": {"count": 2, "totalBytes": total,
                    "minimum": total // 2, "maximum": total // 2, "median": total / 2}}}
                   for arm, total in zip("ABC", (3200, 5120, 4096))]
        value = runner.bound_size_diagnostic(samples)
        self.assertEqual(value["classification"], "reduced")
        self.assertAlmostEqual(value["reductionFraction"], .2)
        self.assertFalse(value["requiredForPerformanceAcceptance"])
        self.assertEqual(len(runner.POLICY["gates"]), 5)
        candidate = runner.native_size_diagnostic(samples)
        self.assertNotIn("boundedTemplateBytes", candidate)
        self.assertEqual(candidate["candidateTemplateBytes"], 4096)
        self.assertEqual(candidate["classification"], value["classification"])
        samples[2]["result"]["nativeCLASSizes"]["totalBytes"] = 5120
        self.assertEqual(runner.bound_size_diagnostic(samples)["classification"], "unchanged")

    def report_case(self, protocol):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "synthetic"
            evidence.mkdir()
            runs = rows()
            for row in runs:
                row["stage"] = "discovery"
                row["result"]["templatePathEvidence"] = {"fallbacksAtFinalFrame": 0}
            diagnostics = []
            for arm, total in zip("ABC", (3200, 5120, 4864)):
                diagnostics.append({"arm": arm, "stage": "diagnostic", "directory": "diagnostic-" + arm,
                    "result": {"identity": {"nativeTemplateCount": 0 if arm == "A" else 1536 if protocol == runner.PROTOCOL and arm == "C" else 512,
                        "templateVariantCount": {"A": 0, "B": 1, "C": 3}[arm], "patternCount": 0 if arm == "A" else 512,
                        "templateBytes": 300 if arm == "C" else 100,
                        "templateAuxiliaryBytes": 600 if arm == "C" else 200, "clasBytes": 300, "scratchBytes": 400},
                        "nativeCLASSizes": {"count": 2, "totalBytes": total, "minimum": total // 2,
                                            "maximum": total // 2, "median": total / 2},
                        "templatePathEvidence": {"fallbacksAtFinalFrame": 0}}})
            runner.save(evidence / "Experiment.json", {"runs": diagnostics + runs})
            runner.save(evidence / "Manifest.json", {})
            verified = {"protocol": protocol, "decisions": {"discovery": runner.assess(runs, "discovery")}}
            with patch.object(runner, "verify", return_value=verified):
                runner.report(evidence, root / "report")
            report = (root / "report/report.md").read_text(encoding="utf-8")
            self.assertIn("C5 rightCorner guide evidence" if protocol == runner.PROTOCOL else "C4 rightDyadic guide evidence", report)
            self.assertIn("Runtime CLAS arena", report)
            self.assertNotIn("C3 bounded-path", report)
            sizes = runner.read(root / "report/NativeSizeDiagnostic.json")
            self.assertEqual(sizes["candidateTemplateBytes"], 4864)
            self.assertNotIn("boundedTemplateBytes", sizes)
            self.assertTrue((root / "report/ProcessMedians.png").is_file())
            self.assertTrue((root / "report/BlockGains.png").is_file())
            allocations = runner.read(root / "report/BackendAllocations.json")
            self.assertEqual(allocations["C"]["identity"]["templateBytes"], 300)
            self.assertEqual(allocations["C"]["identity"]["templateAuxiliaryBytes"], 600)
            self.assertEqual(len(runner.POLICY["gates"]), 5)
            if protocol == runner.PROTOCOL:
                self.assertIn("Persistent bank memory is diagnostic, not an additional acceptance gate", report)
                self.assertEqual(allocations["C"]["identity"]["nativeTemplateCount"], 1536)

    def test_v4_report_uses_candidate_size_and_reports_allocations(self):
        self.report_case(runner.DYADIC_PROTOCOL)

    def test_v5_report_keeps_three_banks_resident_capacity_outside_five_gates(self):
        self.report_case(runner.PROTOCOL)

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

    def test_v5_requires_exact_quality_outputs_and_retains_variant_identity(self):
        diagnostics = [{"arm": arm, "stage": "diagnostic", "recipes": [
            {"module": runner.MODULES[1], "sha256": "shadow", "device": {"gpu": "fixed"}}],
            "result": {"inputs": {"geometry": "same"}, "outputs": {"HDR": "same", "depth": "same", "shadow": "same"},
                       "identity": {"diagnosticBuildSizes": True, "templatePoolGeneration": 7,
                                    "templateVariantCount": {"A": 0, "B": 1, "C": 3}[arm]}}} for arm in "ABC"]
        ordinary = copy.deepcopy(diagnostics[2])
        ordinary["stage"] = "discovery"
        ordinary["result"]["identity"].update(diagnosticBuildSizes=False, templatePoolGeneration=99)
        runner.validate_results([ordinary], diagnostics, runner.PROTOCOL)
        for output in ("HDR", "depth", "shadow"):
            changed = copy.deepcopy(ordinary)
            changed["result"]["outputs"][output] = "different"
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "Exact HDR/depth/shadow"):
                runner.validate_results([changed], diagnostics, runner.PROTOCOL)
        ordinary["result"]["identity"]["templateVariantCount"] = 1
        with self.assertRaisesRegex(ValueError, "Within-arm"):
            runner.validate_results([ordinary], diagnostics, runner.PROTOCOL)


if __name__ == "__main__":
    unittest.main()
