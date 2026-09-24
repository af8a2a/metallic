"""Synthetic parser-contract tests; these are NOT native Nsight export fixtures."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "Tools/Perf/NsightSource.py"
spec = importlib.util.spec_from_file_location("nsight_source", SCRIPT)
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)


def fixture():
    header = ["Module", "Entry", "ID", "File", "Line", "Code", "Samples", "Samples", "Dependency", "Live"]
    table = [
        ["SYNTHETIC CONTRACT FIXTURE - NOT AN NSIGHT EXPORT"],
        header,
        ["same.spv (aaa)", "main", "i1", "shader.slang", "10", "load, x\nnext", "5", "9", "2", "36"],
        ["same.spv (aaa)", "main", "i2", "shader.slang", "11", "barrier", "0", "7", "", ""],
        ["same.spv (bbb)", "main", "i1", "shader.slang", "10", "other", "28", "28", "", ""],
        header,
        ["same.spv (aaa)", "main", "i1", "shader.slang", "10", "load, x\nnext", "5", "9", "2", "36"],
        header,
        ["same.spv (aaa)", "main", "s10", "shader.slang", "10", "let x = load();", "500", "1000", "", ""],
        ["same.spv (aaa)", "main", "s11", "shader.slang", "11", "barrier();", "", "", "", ""],
    ]
    output = io.StringIO(newline="")
    csv.writer(output).writerows(table)
    raw = output.getvalue().encode("utf-8")
    columns = dict(zip(ns.TEXT_FIELDS + ns.METRICS, [0, 1, 3, 4, 5, 6, 7, 8, 9]))
    layout = {"version": 1, "raw_sha256": ns.sha(raw), "sections": [
        {"header_record": start, "end_record": end, "header": header,
         "representation": rep, "columns": columns, "identity_columns": [2]}
        for start, end, rep in [(2, 5, "il"), (6, 7, "il"), (8, 10, "source")]]}
    context = {"artifact_sha256": "a" * 64, "nsight_version": "synthetic",
               "export_id": "synthetic-1", "selection": {"id": "range-1",
               "requested_scope": "shader-in-window", "achieved_scope": "shader-in-window"},
               "synthetic": True}
    return raw, layout, context


class NsightSourceTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.layout, self.context = fixture()

    def result(self):
        return ns.normalize(self.raw, self.layout, self.context)

    def edit_raw(self, old, new):
        self.raw = self.raw.replace(old, new)
        self.layout["raw_sha256"] = ns.sha(self.raw)

    def test_multiline_physical_provenance_and_duplicate_headers(self):
        rows = ns.records(self.raw)
        self.assertEqual(rows[2]["physical_lines"], [3, 4])
        self.assertEqual(rows[2]["cells"][5], "load, x\nnext")
        self.assertEqual(rows[1]["cells"].count("Samples"), 2)

    def test_modules_and_representations_never_added_together(self):
        result = self.result()
        self.assertEqual([s["il_self_samples"] for s in ns.shaders(result)], [5, 28])
        self.assertEqual(result["unmapped_nonempty_records"], 1)
        self.assertFalse(result["native_dialect_validated"])
        hot = ns.hotspots(result, "same.spv (aaa)", "main", "il", "self_samples", 10)
        self.assertEqual(len(hot["rows"]), 2)
        self.assertEqual(hot["rows"][0]["inclusive_samples"], 9)

    def test_duplicate_mirrors_retain_both_provenances(self):
        result = self.result()
        self.assertEqual(len(result["rows"]), 5)
        row = result["rows"][0]
        self.assertEqual([e["record"] for e in row["evidence"]], [3, 7])

    def test_conflicting_duplicate_rejected(self):
        # Change only the repeated IL row, retaining a pinned raw hash.
        text = self.raw.decode()
        at = text.rfind("next\",5,9,2,36")
        self.assertGreater(at, 0)
        self.raw = (text[:at] + text[at:].replace("next\",5,9,2,36", "next\",6,9,2,36", 1)).encode()
        self.layout["raw_sha256"] = ns.sha(self.raw)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.result()

    def test_missing_is_not_zero(self):
        row = self.result()["rows"][1]
        self.assertEqual(row["self_samples"], 0)
        self.assertIsNone(row["dependency_samples"])
        self.assertIsNone(row["live_registers"])
        hot = ns.hotspots(self.result(), "same.spv (aaa)", "main", "source", "self_samples", 10)
        self.assertEqual(hot["missing_metric_rows"], 1)

    def test_truncated_stalls_remain_lower_bounds_and_not_selected_is_preserved(self):
        for section in self.layout["sections"]:
            section["stalls"] = [{"name": 5, "samples": 8}]
            section["stall_coverage"] = "top-k-lower-bound"
        self.edit_raw(b",barrier,0", b",Not Selected,0")
        rows = self.result()["rows"]
        self.assertEqual(rows[1]["stalls"][0]["name"], "Not Selected")
        self.assertIsNone(rows[1]["stalls"][0]["samples"])
        self.assertEqual(rows[0]["stall_coverage"], "top-k-lower-bound")
        del self.layout["sections"][0]["stall_coverage"]
        with self.assertRaisesRegex(ValueError, "stall_coverage"):
            self.result()

    def test_numbers_are_exact_no_unit_guessing(self):
        for value in ("-1", "1.2", "1K", "NaN", "1,000", " 2", "1e3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ns.number(value)
        self.assertEqual(ns.number("0"), 0)
        self.assertIsNone(ns.number(""))

    def test_layout_hash_and_header_must_match(self):
        self.layout["raw_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.result()
        self.layout["raw_sha256"] = ns.sha(self.raw)
        self.layout["sections"][0]["header"] = ["guessed"]
        with self.assertRaisesRegex(ValueError, "header"):
            self.result()

    def test_overlap_width_and_column_mistakes_fail(self):
        raw, layout, context = fixture()
        self.layout["sections"][1]["header_record"] = 5
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.result()
        self.layout = copy.deepcopy(layout)
        self.layout["sections"][0]["columns"]["module"] = 99
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.result()
        self.layout = copy.deepcopy(layout)
        self.edit_raw(b",barrier,0", b",barrier,extra,0")
        with self.assertRaisesRegex(ValueError, "width"):
            self.result()

    def test_same_module_multiple_entries_requires_entry(self):
        self.edit_raw(b"same.spv (bbb),main", b"same.spv (aaa),other")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ns.select(self.result(), "same.spv (aaa)")
        self.assertEqual(len(ns.select(self.result(), "same.spv (aaa)", "other")), 1)

    def test_unknown_entry_and_wrong_shader_fail(self):
        self.edit_raw(b"same.spv (bbb),main", b"same.spv (bbb),")
        with self.assertRaisesRegex(ValueError, "entry identity"):
            ns.select(self.result(), "same.spv (bbb)")
        with self.assertRaisesRegex(ValueError, "missing"):
            ns.select(self.result(), "nonexistent")

    def test_wrong_scope_fails_before_analysis(self):
        self.context["selection"]["achieved_scope"] = "whole-device"
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            self.result()

    def test_source_only_uses_exported_text(self):
        result = ns.source_context(self.result(), "same.spv (aaa)", "main", 9, 1)
        self.assertEqual([r["line"] for r in result["rows"]], ["10", "11"])
        with self.assertRaisesRegex(ValueError, "IL/source mapping"):
            ns.source_context(self.result(), "same.spv (aaa)", "main", 3, 2)
        with self.assertRaisesRegex(ValueError, "not part"):
            ns.source_context(self.result(), "same.spv (aaa)", "main", 5, 2)

    def test_empty_or_invalid_csv_and_utf16_tsv(self):
        with self.assertRaises(ValueError):
            ns.records(b"")
        with self.assertRaises(csv.Error):
            ns.records(b'"unterminated')
        with self.assertRaises(UnicodeError):
            ns.records(b"\xff")
        rows = ns.records("Name\tSamples\r\nshader\t3\r\n".encode("utf-16"), "\t", "utf-16")
        self.assertEqual(rows[1]["cells"], ["shader", "3"])

    def make_bundle(self, root):
        raw, layout, context = root / "input.csv", root / "layout.json", root / "context.json"
        raw.write_bytes(self.raw)
        ns.write_json(layout, self.layout)
        ns.write_json(context, self.context)
        output = root / "bundle"
        ns.import_bundle(raw, layout, context, output)
        return output

    def test_bundle_tamper_and_output_reuse_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = self.make_bundle(root)
            self.assertEqual(ns.load_bundle(output), self.result())
            with self.assertRaises(FileExistsError):
                self.make_bundle(root)
            (output / "raw.csv").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                ns.load_bundle(output)

    def test_cli_import_queries_and_machine_readable_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Exercise import through the command line rather than a second implementation.
            (root / "raw.csv").write_bytes(self.raw)
            ns.write_json(root / "layout.json", self.layout)
            ns.write_json(root / "context.json", self.context)
            args = ["import", "--raw", str(root / "raw.csv"), "--layout", str(root / "layout.json"),
                    "--context", str(root / "context.json"), "--output", str(root / "bundle")]
            def run(args):
                return subprocess.run([sys.executable, "-B", str(SCRIPT)] + args,
                                      text=True, capture_output=True, encoding="utf-8")
            proc = run(args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            proc = run(["hotspots", str(root / "bundle"), "--module", "same.spv (aaa)"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["rows"][0]["self_samples"], 5)
            self.assertEqual(json.loads(proc.stdout)["raw_sha256"], ns.sha(self.raw))
            self.assertEqual(json.loads(proc.stdout)["metadata_trust"], "caller-declared")
            proc = run(["source", str(root / "bundle"), "--module", "same.spv (aaa)", "--record", "9"])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            proc = run(["hotspots", str(root / "bundle"), "--module", "wrong"])
            self.assertEqual(proc.returncode, 2)
            self.assertIn("error", json.loads(proc.stderr))

    def test_three_exports_compare_content_and_selection_not_declare_automation(self):
        first = self.result()
        results = [copy.deepcopy(first) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "export_id"):
            ns.compare_repeats(results)
        for i, result in enumerate(results):
            result["context"]["export_id"] = "synthetic-" + str(i)
        compared = ns.compare_repeats(results)
        self.assertTrue(compared["structured_results_equal"])
        self.assertFalse(compared["automation_verified"])
        results[1]["rows"][0]["self_samples"] = 6
        self.assertFalse(ns.compare_repeats(results)["structured_results_equal"])
        results[1]["rows"][0]["self_samples"] = 5
        results[1]["context"]["selection"]["id"] = "another-range"
        self.assertFalse(ns.compare_repeats(results)["structured_results_equal"])


if __name__ == "__main__":
    unittest.main()
