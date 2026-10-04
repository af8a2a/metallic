"""Evidence rejection tests; synthetic timings do not constitute GPU validation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import ClosureScheduling as c


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        c.w.save(self.directory / "Process.json", {"exitCode": 0})
        (self.directory / "Tests.xml").write_text('<testsuites><testsuite>' + ''.join(
            f'<testcase name="{name}" status="run" result="completed" />' for name in
            ("material_closure_family_classification", "material_closure_fused_split_ab")) + '</testsuite></testsuites>')
        fixtures = ((1, 1, 8, False), (17, 9, 8, True), (257, 129, 8, True), (1024, 512, 2, False),
                    (1024, 512, 2, True), (1024, 512, 8, False), (1024, 512, 8, True))
        self.report = {"validation": False, "pipelineStatistics": False, "validationErrors": 0,
                       "warmupPairs": 2, "measuredPairs": 6, "samples": [], "switchSamples": [],
                       "shaders": [{"label": f'ClosureSchedulingProbe.fusedMain.P{i}', "inputSpirvFnv1a64": i} for i in range(8)] +
                                  [{"label": f'other{i}', "inputSpirvFnv1a64": i + 8} for i in range(21)]}
        for width, height, count, mixed in fixtures:
            for pair in range(8):
                for split in (False, True):
                    self.report["samples"].append({"width": width, "height": height, "programCount": count,
                        "mixed": mixed, "pair": pair, "split": split, "warmup": pair < 2, "closureFamilyCount": 1,
                        "maxOutputDifference": 0, "programClassificationMs": .1, "materialMs": .1,
                        "familyClassificationMs": .1, "lightingMs": .1, "totalMs": .3})
        for count in (2, 8):
            for sample in range(8):
                for distinct in (False, True):
                    self.report["switchSamples"].append({"programCount": count, "sample": sample, "distinct": distinct,
                        "warmup": sample < 2, "dispatches": count * 64, "gpuMs": .1})

    def tearDown(self):
        self.temp.cleanup()

    def inspect(self, report=None):
        c.w.save(self.directory / "ClosureScheduling.json", self.report if report is None else report)
        return c.inspect(self.directory, "timing")

    def test_complete_pairing(self):
        self.inspect()
        broken = copy.deepcopy(self.report)
        broken['samples'][-1] = broken['samples'][-2]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.inspect(broken)

    def test_diagnostics_are_not_normal_timing(self):
        for flag in ('validation', 'pipelineStatistics'):
            broken = copy.deepcopy(self.report); broken[flag] = True
            with self.assertRaisesRegex(ValueError, 'instrumentation'):
                self.inspect(broken)

    def test_bad_output_and_nonfinite_duration(self):
        for field, value in (('maxOutputDifference', .01), ('totalMs', -1)):
            broken = copy.deepcopy(self.report); broken['samples'][0][field] = value
            with self.assertRaises(ValueError): self.inspect(broken)
        self.inspect()
        path = self.directory / 'ClosureScheduling.json'
        path.write_text(path.read_text().replace('"totalMs": 0.3', '"totalMs": NaN', 1))
        with self.assertRaisesRegex(ValueError, 'duration'): c.inspect(self.directory, 'timing')

    def test_missing_switch_control(self):
        broken = copy.deepcopy(self.report); broken['switchSamples'][-1] = broken['switchSamples'][-2]
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.inspect(broken)

    def test_skipped_or_wrong_test_is_not_evidence(self):
        path = self.directory / 'Tests.xml'
        path.write_text(path.read_text().replace('status="run"', 'status="notrun"', 1))
        with self.assertRaisesRegex(ValueError, 'skipped'): self.inspect()

    def test_file_tampering(self):
        data = self.directory / 'payload.json'; data.write_text('{}')
        c.w.save(self.directory / 'Manifest.json', {'protocol': c.PROTOCOL, 'status': 'complete',
                 'files': {'payload.json': c.w.digest(data)}})
        data.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'): c.verify(self.directory)


if __name__ == '__main__':
    unittest.main()
