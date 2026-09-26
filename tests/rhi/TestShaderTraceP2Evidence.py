"""Acceptance gates must reject drift even if cached run status says success."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import ShaderTrace as t


def rows():
    result = []
    for i, item in enumerate(t.schedule(True, None)):
        record = {**item, "process": {"pid": i+1, "startedUnix": i*10+1, "endedUnix": i*10+5},
                  "identity": {"case": "same"}}
        if item["selection"] is not None:
            outcome = item["expected"]
            record.update(session=str(i), phase=item["selection"]["phase"], outcome=outcome,
                          selectedScopeComplete=True, schemaHash=item["selection"]["phase"],
                          fields=[{"depth": {"type": "f32", "bits": 0x80000000}}] if outcome=="Matched" else [],
                          summary={"siteEvaluationCount": 0 if outcome=="SiteNotReached" else 1})
        result.append(record)
    return result


class EvidenceTests(unittest.TestCase):
    def test_full_schedule(self):
        self.assertTrue(t.assess(rows(), True)["p2Acceptance"])

    def test_wrong_bits_phase_and_input(self):
        for fault in range(3):
            data = rows()
            if fault==0: data[2]["fields"][0]["depth"]["bits"] = 0
            if fault==1: data[4]["phase"] = "early"
            if fault==2: data[3]["identity"]["case"] = "changed"
            with self.assertRaises(ValueError): t.assess(data, True)

    def test_stale_or_overlapping_processes(self):
        data=rows(); data[2]["session"]=data[1]["session"]
        with self.assertRaises(ValueError): t.assess(data, True)
        data=rows(); data[2]["process"]["startedUnix"]=1
        with self.assertRaises(ValueError): t.assess(data, True)

    def test_empty_result_is_not_coverage(self):
        for fault in range(3):
            data=rows()
            if fault==0: data[-1]["outcome"]="NoMatch"
            if fault==1: data[-2]["summary"]["siteEvaluationCount"]=0
            if fault==2: data[-1]["selectedScopeComplete"]=False
            with self.assertRaises(ValueError): t.assess(data, True)

    def test_bounded_selectors(self):
        self.assertEqual(t.selection(local=127)["localIndex"],127)
        for args in ({"group_x":65535}, {"local":128}, {"triangle":-1}, {"triangle":True}):
            with self.assertRaises(ValueError): t.selection(**args)


if __name__ == "__main__":
    unittest.main()
