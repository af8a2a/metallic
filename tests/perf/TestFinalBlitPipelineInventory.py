"""CPU-only inventory/presentation checks, not RenderGraph::validate or GPU coverage.

The C++ expectedSources map remains the source of exact pipeline expectations.
This deliberately checks only static inventory and hasFinalOutput's JSON contract;
pass reflection, firstOutputName(), sample overrides and execution need runtime tests.
"""
import copy
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = ROOT / "tests/rhi/FinalBlitTests.cpp"
PHYSICAL_TYPES = {
    "ScenePathTracePass", "SceneRealtimeLightingPass", "VisibilityBufferDeferredPass",
    "SceneRTXDIPass", "RTXDICompositePass",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def parse_expected_sources(text):
    blocks = re.findall(
        r"const\s+std::unordered_map<std::string,\s*std::string>\s+expectedSources\s*\{(.*?)\};",
        text, re.DOTALL)
    require(len(blocks) == 1, "Expected exactly one expectedSources initializer")
    block = blocks[0]
    entry = re.compile(r'\s*\{"([^"\\]+\.metallic_graph\.json)",\s*"([^"\\]*)"\},')
    expected = {}
    position = 0
    while block[position:].strip():
        match = entry.match(block, position)
        require(match is not None, "Unparsed expectedSources initializer entry")
        filename, source = match.groups()
        require(filename not in expected, f"Duplicate expectedSources key: {filename}")
        require(Path(filename).name == filename and "/" not in filename and "\\" not in filename,
                f"Expected a pipeline basename: {filename}")
        expected[filename] = source
        position = match.end()
    require(expected, "Empty expectedSources initializer")
    return expected


def validate_inventory(expected, paths):
    actual = {}
    for path in paths:
        require(path.name not in actual, f"Duplicate pipeline basename: {path.name}")
        actual[path.name] = path
    missing = sorted(actual.keys() - expected.keys())
    stale = sorted(expected.keys() - actual.keys())
    require(not missing and not stale, f"Pipeline inventory mismatch: missing={missing}, stale={stale}")
    return actual


def validate_presentation(graph, expected_source):
    """Mirror the static part of FinalBlitPipelinesTest::hasFinalOutput exactly."""
    nodes = {}
    node_ids = set()
    for node in graph["nodes"]:
        require(node["name"] not in nodes, f"Duplicate node name: {node['name']}")
        require(node["id"] not in node_ids, f"Duplicate node id: {node['id']}")
        nodes[node["name"]] = node
        node_ids.add(node["id"])
    edge_ids = [edge["id"] for edge in graph["edges"]]
    require(len(edge_ids) == len(set(edge_ids)), "Duplicate edge id")

    def incoming(destination):
        return [edge["src"] for edge in graph["edges"] if edge["dst"] == destination]

    physical_nodes = [node for node in nodes.values() if node["type"] in PHYSICAL_TYPES]
    require(all(node.get("properties", {}).get("outputLinear", False) for node in physical_nodes),
            "Physical producers must set outputLinear")
    exposure = nodes.get("AutoExposure")
    require(bool(physical_nodes) == (exposure is not None), "Physical/AutoExposure presence mismatch")
    if exposure is not None:
        require(exposure["type"] == "AutoExposurePass", "AutoExposure has wrong pass type")
        require(incoming("AutoExposure.source") == [expected_source],
                "AutoExposure.source must have exactly the expected source")
        expected_source = "AutoExposure.color"
    nr = nodes.get("DLSSNR")
    if nr is not None:
        require(nr["type"] == "DLSSNRPass", "DLSSNR has wrong pass type")
        require(incoming("DLSSNR.inputColor") == [expected_source],
                "DLSSNR.inputColor must have exactly the expected source")
        expected_source = "DLSSNR.color"
    final = nodes.get("FinalBlit")
    require(final is not None and final["type"] == "FinalBlitPass", "Missing FinalBlitPass terminal")
    require(not graph.get("outputs", []), "Explicit graph outputs bypass automatic presentation")
    if physical_nodes:
        grading = nodes.get("ColorGrading")
        require(grading is not None and grading["type"] == "ColorGradingLUTPass" and
                grading.get("properties", {}).get("toneCurve", "aces2") == "aces2",
                "Physical pipeline needs ACES2 ColorGradingLUTPass")
        require("ColorGrading.lut" in incoming("FinalBlit.lut"), "Missing ColorGrading.lut connection")
    if expected_source:
        require(incoming("FinalBlit.source") == [expected_source],
                "FinalBlit.source must have exactly the expected source")
    else:
        require(not incoming("FinalBlit.source") and
                final.get("properties", {}).get("calibrationPattern", False),
                "Unconnected FinalBlit requires calibrationPattern")


class FinalBlitPipelineInventoryTests(unittest.TestCase):
    def test_repository_pipeline_inventory_and_presentation(self):
        expected = parse_expected_sources(CPP_SOURCE.read_text(encoding="utf-8"))
        actual = validate_inventory(expected, sorted((ROOT / "Pipelines").rglob("*.metallic_graph.json")))
        for filename, path in actual.items():
            with self.subTest(pipeline=filename):
                validate_presentation(json.loads(path.read_text(encoding="utf-8")), expected[filename])

    def initializer(self, entries):
        return "const std::unordered_map<std::string, std::string> expectedSources{\n" + entries + "\n};"

    def fixture(self, physical=False, nr=False):
        graph = {
            "nodes": [
                {"id": 1, "name": "Source", "type": "ScenePathTracePass" if physical else "ClearColorPass",
                 "properties": {"outputLinear": True}},
                {"id": 2, "name": "FinalBlit", "type": "FinalBlitPass", "properties": {}},
            ],
            "edges": [], "outputs": [],
        }
        source = "Source.color"
        if physical:
            graph["nodes"] += [
                {"id": 3, "name": "AutoExposure", "type": "AutoExposurePass"},
                {"id": 4, "name": "ColorGrading", "type": "ColorGradingLUTPass",
                 "properties": {"toneCurve": "aces2"}},
            ]
            graph["edges"] += [
                {"id": 1, "src": source, "dst": "AutoExposure.source"},
                {"id": 2, "src": "ColorGrading.lut", "dst": "FinalBlit.lut"},
            ]
            source = "AutoExposure.color"
        if nr:
            graph["nodes"].append({"id": 5, "name": "DLSSNR", "type": "DLSSNRPass"})
            graph["edges"].append({"id": 3, "src": source, "dst": "DLSSNR.inputColor"})
            source = "DLSSNR.color"
        graph["edges"].append({"id": 4, "src": source, "dst": "FinalBlit.source"})
        return graph

    def test_initializer_parses_exact_sources_and_calibration(self):
        text = self.initializer('{"a.metallic_graph.json", "Source.color"},\n{"b.metallic_graph.json", ""},')
        self.assertEqual(parse_expected_sources(text), {"a.metallic_graph.json": "Source.color",
                                                       "b.metallic_graph.json": ""})

    def test_initializer_rejects_duplicate_key(self):
        entry = '{"a.metallic_graph.json", "Source.color"},'
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_expected_sources(self.initializer(entry + entry))

    def test_initializer_rejects_unparsed_entries(self):
        for entries in ('BROKEN,', '{"a.metallic_graph.json", expression},',
                        '{"a.metallic_graph.json", "Source.color"}, ignored'):
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "Unparsed"):
                parse_expected_sources(self.initializer(entries))

    def test_initializer_rejects_missing_empty_or_multiple_maps(self):
        for text in ('', self.initializer(''), self.initializer('') * 2):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_expected_sources(text)

    def test_initializer_rejects_non_basename_keys(self):
        with self.assertRaisesRegex(ValueError, "basename"):
            parse_expected_sources(self.initializer('{"Samples/a.metallic_graph.json", "Source.color"},'))

    def test_new_pipeline_missing_from_whitelist_fails(self):
        with self.assertRaisesRegex(ValueError, "missing=.*new.metallic_graph.json"):
            validate_inventory({}, [Path("Samples/new.metallic_graph.json")])

    def test_stale_whitelist_entry_fails(self):
        with self.assertRaisesRegex(ValueError, "stale=.*old.metallic_graph.json"):
            validate_inventory({"old.metallic_graph.json": "Source.color"}, [])

    def test_duplicate_asset_basename_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate pipeline"):
            validate_inventory({"a.metallic_graph.json": "Source.color"},
                               [Path("a.metallic_graph.json"), Path("Samples/a.metallic_graph.json")])

    def test_valid_direct_exposure_and_nr_chains(self):
        for physical in (False, True):
            for nr in (False, True):
                with self.subTest(physical=physical, nr=nr):
                    validate_presentation(self.fixture(physical, nr), "Source.color")

    def test_wrong_expected_source_fails(self):
        for physical in (False, True):
            with self.subTest(physical=physical), self.assertRaisesRegex(ValueError, "expected source"):
                validate_presentation(self.fixture(physical), "Wrong.color")

    def test_missing_wrong_or_duplicate_terminal_source_fails(self):
        for mutation in ("missing", "wrong", "duplicate"):
            graph = self.fixture()
            if mutation == "missing":
                graph["edges"].clear()
            elif mutation == "wrong":
                graph["edges"][-1]["src"] = "Wrong.color"
            else:
                graph["edges"].append({"id": 9, "src": "Source.color", "dst": "FinalBlit.source"})
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "FinalBlit.source"):
                validate_presentation(graph, "Source.color")

    def test_explicit_output_or_wrong_final_type_fails(self):
        for mutation in ("output", "type", "missing"):
            graph = self.fixture()
            if mutation == "output":
                graph["outputs"] = ["FinalBlit.color"]
            elif mutation == "type":
                graph["nodes"][1]["type"] = "CopyColorPass"
            else:
                graph["nodes"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_presentation(graph, "Source.color")

    def test_physical_linear_exposure_and_grading_requirements(self):
        base = self.fixture(physical=True)
        for mutation in ("linear", "exposure", "exposure_type", "grading", "tone", "lut"):
            graph = copy.deepcopy(base)
            if mutation == "linear":
                graph["nodes"][0]["properties"]["outputLinear"] = False
            elif mutation == "exposure":
                graph["nodes"] = [node for node in graph["nodes"] if node["name"] != "AutoExposure"]
            elif mutation == "exposure_type":
                graph["nodes"][2]["type"] = "CopyColorPass"
            elif mutation == "grading":
                graph["nodes"][3]["type"] = "CopyColorPass"
            elif mutation == "tone":
                graph["nodes"][3]["properties"]["toneCurve"] = "none"
            else:
                graph["edges"] = [edge for edge in graph["edges"] if edge["dst"] != "FinalBlit.lut"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_presentation(graph, "Source.color")

    def test_exposure_and_nr_input_must_be_unique_and_exact(self):
        for destination in ("AutoExposure.source", "DLSSNR.inputColor"):
            for mutation in ("missing", "wrong", "duplicate"):
                graph = self.fixture(physical=True, nr=True)
                edge = next(edge for edge in graph["edges"] if edge["dst"] == destination)
                if mutation == "missing":
                    graph["edges"].remove(edge)
                elif mutation == "wrong":
                    edge["src"] = "Wrong.color"
                else:
                    graph["edges"].append(dict(edge, id=9))
                with self.subTest(destination=destination, mutation=mutation), self.assertRaises(ValueError):
                    validate_presentation(graph, "Source.color")

    def test_nonphysical_exposure_and_wrong_nr_type_fail(self):
        graph = self.fixture(physical=True)
        graph["nodes"][0]["type"] = "ClearColorPass"
        with self.assertRaisesRegex(ValueError, "presence mismatch"):
            validate_presentation(graph, "Source.color")
        graph = self.fixture(nr=True)
        graph["nodes"][-1]["type"] = "CopyColorPass"
        with self.assertRaisesRegex(ValueError, "DLSSNR has wrong"):
            validate_presentation(graph, "Source.color")

    def test_calibration_requires_pattern_and_no_source(self):
        graph = self.fixture()
        graph["edges"].clear()
        graph["nodes"][1]["properties"]["calibrationPattern"] = True
        validate_presentation(graph, "")
        graph["nodes"][1]["properties"]["calibrationPattern"] = False
        with self.assertRaisesRegex(ValueError, "calibrationPattern"):
            validate_presentation(graph, "")
        graph["nodes"][1]["properties"]["calibrationPattern"] = True
        graph["edges"] = [{"id": 1, "src": "Source.color", "dst": "FinalBlit.source"}]
        with self.assertRaisesRegex(ValueError, "calibrationPattern"):
            validate_presentation(graph, "")

    def test_duplicate_node_names_ids_and_edge_ids_fail(self):
        for mutation in ("name", "id", "edge"):
            graph = self.fixture()
            if mutation == "name":
                graph["nodes"][1]["name"] = "Source"
            elif mutation == "id":
                graph["nodes"][1]["id"] = 1
            else:
                graph["edges"].append(copy.deepcopy(graph["edges"][0]))
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_presentation(graph, "Source.color")


if __name__ == "__main__":
    unittest.main()
