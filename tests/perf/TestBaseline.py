"""Evidence validity tests; no Nsight, GPU, network or desktop needed."""
import importlib.util
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "Tools/Perf/Baseline.py"
spec = importlib.util.spec_from_file_location("metallic_perf_baseline", SOURCE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def test_desktop_does_not_require_child_and_never_implies_verified_export(self):
        self.assertEqual(baseline.source_export_capability("desktop")["status"], "supported-unverified")
        self.assertIsNone(baseline.source_export_capability("desktop")["verified_scope"])
        self.assertEqual(baseline.source_export_capability("child")["status"], "blocked")
        self.assertEqual(baseline.source_export_capability("child", {"workerReady": True})["status"],
                         "supported-unverified")
        self.assertEqual(baseline.source_export_capability("none")["status"], "blocked")

    def test_empty_manifest_is_not_a_valid_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "Baseline.json"
            baseline.write_json(file, {"schema_version": 1, "kind": "metallic.perf.baseline"})
            self.assertEqual(len(baseline.verify_bundle(file)), 3)

    def test_real_fixtures_preserve_hashes_and_duplicate_columns(self):
        directory = Path(__file__).parent / "fixtures"
        manifest = baseline.read_json(directory / "Provenance.json")
        self.assertFalse(manifest["shaderSourceFixture"])
        for item in manifest["artifacts"]:
            self.assertEqual(baseline.digest(directory / item["file"]), item["sha256"])
        with (directory / "GpuTraceRegimes.sample.tsv").open(encoding="utf-8-sig", newline="") as stream:
            header, row = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual(len(header), len(row))
        self.assertGreater(len(header), len(set(header)))
        self.assertEqual(row[0], "RenderGraphPass: VBuffer (VisibilityBufferPass)/Hybrid raster: stable cluster bins")

    def test_missing_empty_and_mismatch_are_not_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "source.csv"
            self.assertEqual(baseline.record_file(file, "raw")["state"], "missing")
            file.touch()
            self.assertEqual(baseline.record_file(file, "raw")["state"], "empty")
            file.write_bytes(b"not the historical export")
            self.assertEqual(baseline.record_file(file, "raw", "a" * 64)["state"], "hash-mismatch")

    def make_bundle(self, directory):
        collector = baseline.Collector(directory, Path(directory) / "run")
        source = Path(directory) / "evidence.csv"
        source.write_bytes(b"Name,Samples\r\nshader,12\r\n")
        collector.artifact("evidence", source, "raw", archive=True)
        collector.command("probe", [sys.executable, "-c", "print('ok')"])
        bundle = {"schema_version": 1, "kind": "metallic.perf.baseline", "artifacts": collector.artifacts,
                  "commands": collector.commands,
                  "capabilities": {"discovery": baseline.capability("verified", "test probe only", "unit test", ["probe"])}}
        target = collector.output / "Baseline.json"
        baseline.write_json(target, bundle)
        return collector, source, bundle, target

    def test_archive_survives_original_removal_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            collector, source, bundle, target = self.make_bundle(directory)
            source.unlink()
            self.assertEqual(baseline.verify_bundle(target), [])
            archive = collector.output / bundle["artifacts"]["evidence"]["archive"]
            archive.write_bytes(b"changed")
            self.assertIn("Artifact missing or changed: evidence", baseline.verify_bundle(target))

    def test_raw_command_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            collector, _, bundle, target = self.make_bundle(directory)
            raw = collector.output / bundle["commands"]["probe"]["stdout"]["path"]
            raw.write_bytes(b"forged success")
            self.assertIn("Command evidence missing or changed: probe.stdout", baseline.verify_bundle(target))

    def test_failed_command_cannot_support_verified_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            collector, _, bundle, target = self.make_bundle(directory)
            collector.command("probe", [sys.executable, "-c", "raise SystemExit(2)"])
            baseline.write_json(target, bundle)
            self.assertIn("Verified claim cites failed command: discovery/probe", baseline.verify_bundle(target))

    def test_verified_claim_requires_scope_and_existing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, bundle, target = self.make_bundle(directory)
            bundle["capabilities"]["discovery"] = baseline.capability("verified", None, "bad", ["unknown"])
            baseline.write_json(target, bundle)
            errors = baseline.verify_bundle(target)
            self.assertTrue(any("lacks scope" in error for error in errors))
            self.assertTrue(any("Unknown evidence" in error for error in errors))

    def test_output_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            baseline.Collector(directory, output)
            with self.assertRaises(FileExistsError):
                baseline.Collector(directory, output)

    def test_timeout_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = baseline.Collector(directory, Path(directory) / "run")
            collector.command("slow", [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
            self.assertTrue(collector.commands["slow"]["timed_out"])
            self.assertIsNone(collector.commands["slow"]["exit_code"])


if __name__ == "__main__":
    unittest.main()
