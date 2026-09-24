"""Real Nsight CSV fixture tests; mutations below are explicitly negative tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nsight_source", ROOT / "Tools/Perf/NsightSource.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)
FIXTURE = ROOT / "tests/perf/fixtures/NsightSourceReal.csv"


class NativeNsightTests(unittest.TestCase):
    def setUp(self):
        self.data = FIXTURE.read_bytes()

    def parse(self, data=None):
        data = self.data if data is None else data
        return ns.normalize(data, {"version": 1, "dialect": "nsight-source-il-observed-v1",
                                   "raw_sha256": ns.sha(data)}, {})

    def test_real_fixture_provenance_and_metrics(self):
        provenance = ns.read_json(FIXTURE.with_suffix(".provenance.json"))
        self.assertFalse(provenance["synthetic"])
        self.assertEqual(ns.sha(self.data), provenance["fixture_sha256"])
        self.assertEqual(provenance["source_sha256"], "c14f9367f825029169ecabc115cdf8356a0ceee0d505278df01c2014056869bc")
        result = self.parse()
        self.assertTrue(result["native_dialect_validated"])
        shader = ns.shaders(result)[0]
        self.assertEqual(shader["module"], "comp.10000.spv (4d0378657da7b7d1)")
        self.assertEqual(shader["entry"], "main")
        self.assertEqual(shader["il_self_samples"], 614)
        self.assertEqual(result["modules"][0]["entry_points"][0]["symbol"],
                         "%streamClusterRasterWorkControlMain")
        self.assertEqual(len([r for r in result["rows"] if r["representation"] == "source"]), 65)

    def test_mirrors_and_total_samples_not_added_to_il(self):
        result = self.parse()
        mirrors = [r for r in result["annotations"] if r["kind"] == "source-mirror"]
        self.assertEqual(len(mirrors), 35)
        row = next(r for r in result["rows"] if r["representation"] == "source" and r["line"] == "63")
        self.assertEqual(row["self_samples"], 6)
        self.assertEqual(row["total_samples"], 5165)
        self.assertIsNone(row["inclusive_samples"])
        self.assertEqual(ns.shaders(result)[0]["il_self_samples"], 614)

    def test_stall_lower_bound_and_hotspot(self):
        result = self.parse()
        hot = ns.hotspots(result, "comp.10000.spv (4d0378657da7b7d1)", "main", "il", "self_samples", 3)
        self.assertEqual(hot["rows"][0]["line"], "24316")
        self.assertEqual(hot["rows"][0]["self_samples"], 502)
        self.assertEqual(hot["rows"][0]["stalls"], [{"name": "Barrier", "samples": 502}])
        reasons = {}
        for row in result["rows"]:
            if row["representation"] == "il":
                for item in row["stalls"]:
                    reasons[item["name"]] = reasons.get(item["name"], 0) + item["samples"]
        self.assertEqual(reasons["Barrier"], 517)
        self.assertEqual(sum(reasons.values()), 604)
        self.assertEqual(reasons["Not Selected"], 3)

    def test_unknown_headers_and_damaged_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "header"):
            self.parse(self.data.replace(b"Dependency-Attributed Samples", b"Unknown", 1))
        with self.assertRaisesRegex(ValueError, "row"):
            self.parse(self.data.replace(b",503,503,Barrier", b",503,extra,503,Barrier", 1))

    def test_sample_unit_and_stall_overflow_fail(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            self.parse(self.data.replace(b",503,503,Barrier", b",503,503K,Barrier", 1))
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.parse(self.data.replace(b",503,503,Barrier", b",503,1,Barrier", 1))

    def test_duplicate_rows_deduplicated_and_conflicts_fail(self):
        line = next(line for line in self.data.splitlines(keepends=True) if line.startswith(b"24316,"))
        repeated = self.parse(self.data + line)
        self.assertEqual(ns.shaders(repeated)[0]["il_self_samples"], 614)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.parse(self.data + line.replace(b",502,Barrier,502", b",503,Barrier,502"))

    def test_multiple_modules_leave_source_unattributed(self):
        header = b"#,Source,Samples,"
        start = self.data.index(header)
        extra = self.data[start:].replace(b"4d0378657da7b7d1", b"aaaaaaaaaaaaaaaa")
        result = self.parse(self.data + extra)
        for association in result["source_associations"]:
            self.assertIsNone(association["module"])
        for module in ("comp.10000.spv (4d0378657da7b7d1)", "comp.10000.spv (aaaaaaaaaaaaaaaa)"):
            selected = ns.select(result, module, "main")
            self.assertTrue(all(r["representation"] == "il" for r in selected))
            self.assertEqual(sum(r["self_samples"] or 0 for r in selected), 614)

    def test_missing_entry_is_not_inferred_from_source_function(self):
        data = self.data.replace(b"OpEntryPoint GLCompute", b"OpUnknown GLCompute")
        result = self.parse(data)
        with self.assertRaisesRegex(ValueError, "entry identity"):
            ns.select(result, "comp.10000.spv (4d0378657da7b7d1)")

    def test_multiple_entry_declarations_do_not_select_first(self):
        entry_line = next(line for line in self.data.splitlines(keepends=True) if line.startswith(b"18,") and b"OpEntryPoint" in line)
        extra = entry_line.replace(b"18,", b"99999,", 1).replace(b"GLCompute %streamClusterRasterWorkControlMain", b"GLCompute %anotherEntry")
        result = self.parse(self.data + extra)
        with self.assertRaisesRegex(ValueError, "entry identity"):
            ns.select(result, "comp.10000.spv (4d0378657da7b7d1)")
        self.assertIsNone(result["source_associations"][0]["module"])

    def test_source_context_comes_from_export(self):
        result = self.parse()
        anchor = next(r for r in result["rows"] if r["representation"] == "source" and r["line"] == "17")
        value = ns.source_context(result, anchor["module"], "main", anchor["evidence"][0]["record"], 1)
        self.assertEqual([r["line"] for r in value["rows"]], ["16", "17", "18"])
        self.assertIn("g_StreamSoftwareValid", value["rows"][1]["code"])

    def test_native_bundle_integrity_and_no_fabricated_capture_context(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            ns.import_bundle(FIXTURE, None, None, output, native=True)
            result = ns.load_bundle(output)
            self.assertEqual(result["context"], {})
            with self.assertRaisesRegex(ValueError, "Context requires"):
                ns.compare_repeats([result, copy.deepcopy(result), copy.deepcopy(result)])
            manifest = ns.read_json(output / "manifest.json")
            manifest["native_parser_sha256"] = "0" * 64
            ns.write_json(output / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Native parser changed"):
                ns.load_bundle(output)


if __name__ == "__main__":
    unittest.main()

