"""Fail-closed P3 evidence gates, separate from real GPU acceptance."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import ShaderDebugger as s


def word(value):
    return {"type":"u32", "bits":value & 0xffffffff, "value":value}


def rows():
    result = []
    for i, spec in enumerate(s.schedule()):
        fault = spec["arm"] == "fault"
        identity = {"snapshot":{k:{"sha256":"fault" if fault else "same","pixels":1} for k in s.w.OUTPUTS},
                    "camera":{},"renderExtent":[1,1],"graph":[],"historyInvalidationPolicy":"fixed"}
        item = {**spec,"identity":identity,"process":{"pid":i+1,"startedUnix":i*10+1,"endedUnix":i*10+5}}
        if spec["selection"] is None:
            item["normalResult"] = {"valid":True,"normalTiming":True,"identity":copy.deepcopy(identity),
                "timings":{m:{"median":1.0} for m in ("softwareTotalMs","graphGpuMs")}}
        else:
            f = {k:word(v) for k,v in {"recordIndex":0,"triangleId":0,"instanceFlags":3}.items()}
            if spec["name"] == "prepare": f["aX"] = word(157365)
            else:
                f.update({k:word(v) for k,v in {"signedArea":0 if fault else -923,"doubleSided":1,
                    "reason":1 if fault else 3,"evaluatedStages":1 if fault else 3,
                    "lowerX":615,"lowerY":235,"upperX":614,"upperY":234,"determinant":0}.items()})
            item.update(session=str(i),fields=[f],outcome="Matched",selectedScopeComplete=True)
        result.append(item)
    return result


class DebuggerTests(unittest.TestCase):
    def test_full_fault_repair_and_no_promotion(self):
        value=s.assess(rows(),s.t.recipe())
        self.assertTrue(value["p3Acceptance"])
        self.assertFalse(value["candidateAccepted"])
        self.assertEqual(value["performanceGate"]["status"],"pass")

    def test_logs_disappearing_are_not_repair(self):
        for change in ({"fields":[]},{"outcome":"NoMatch"},{"selectedScopeComplete":False}):
            data=rows(); data[-1].update(change)
            with self.assertRaises(ValueError): s.assess(data,s.t.recipe())

    def test_print_only_fix_and_missing_normal_proof(self):
        data=rows(); data[-4]["identity"]["snapshot"][s.w.OUTPUTS[0]]["sha256"]="fault"
        with self.assertRaises(ValueError): s.assess(data,s.t.recipe())
        data=rows(); data.pop(-4)
        with self.assertRaises(ValueError): s.assess(data,s.t.recipe())

    def test_uninstrumented_fault_must_change_output(self):
        data=rows()
        for r in data:
            if r["arm"]=="fault": r["identity"]["snapshot"]=copy.deepcopy(data[0]["identity"]["snapshot"])
        with self.assertRaisesRegex(ValueError,"did not reproduce"): s.assess(data,s.t.recipe())

    def test_earlier_divergence_and_wrong_triangle(self):
        for name in ("aX","triangleId"):
            data=rows(); data[6]["fields"][0][name]=word(91)
            with self.assertRaises(ValueError): s.assess(data,s.t.recipe())

    def test_stage_mask_excludes_uncomputed_plane(self):
        row=rows()[4]
        self.assertNotIn("determinant",s.decision_values(row))
        row["fields"][0]["evaluatedStages"]=word(7)
        self.assertIn("determinant",s.decision_values(row))
        row["fields"][0]["evaluatedStages"]=word(0)
        with self.assertRaises(ValueError): s.decision_values(row)

    def test_printf_timing_and_identity_drift_rejected(self):
        data=rows(); data[8]["normalResult"]["normalTiming"]=False
        with self.assertRaisesRegex(ValueError,"Instrumented"): s.assess(data,s.t.recipe())
        data=rows(); data[8]["normalResult"]["identity"]["residentPages"]=4
        with self.assertRaisesRegex(ValueError,"residency"): s.assess(data,s.t.recipe())

    def test_noise_and_regression_not_promoted(self):
        data=rows(); data[8]["normalResult"]["timings"]["graphGpuMs"]["median"]=2
        self.assertEqual(s.assess(data,s.t.recipe())["performanceGate"]["status"],"inconclusive")
        data=rows()
        for r in data[8:11]: r["normalResult"]["timings"]["graphGpuMs"]["median"]=1.2
        self.assertEqual(s.assess(data,s.t.recipe())["performanceGate"]["status"],"reject")

    def test_stale_and_overlapping_processes(self):
        data=rows(); data[-1]["session"]=data[4]["session"]
        with self.assertRaises(ValueError): s.assess(data,s.t.recipe())
        data=rows(); data[8]["process"]["startedUnix"]=1
        with self.assertRaises(ValueError): s.assess(data,s.t.recipe())

    def test_pinned_finite_plan(self):
        original=(s.ROOT/s.e.TARGET).read_bytes()
        plan=s.plan_for(original)
        self.assertNotEqual(s.validate_plan(plan,original),original)
        for mutate in (lambda p:p.update(command="anything"),lambda p:p["selection"].update(localIndex=1),
                       lambda p:p["repair"].update(baseSha256="wrong"),lambda p:p["fault"].update(path="elsewhere")):
            candidate=copy.deepcopy(plan); mutate(candidate)
            with self.assertRaises(ValueError): s.validate_plan(candidate,original)

    def test_transaction_restore_and_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/s.e.TARGET; path.parent.mkdir(parents=True); path.write_bytes(b"base")
            output=root/"evidence"; output.mkdir()
            with patch.object(s.e,"ROOT",root):
                tx=s.e.ShaderTransaction(output,b"base",b"fault")
                tx.install(b"fault"); tx.restore()
                self.assertEqual(path.read_bytes(),b"base")
                tx.install(b"fault"); path.write_bytes(b"external edit")
                with self.assertRaisesRegex(ValueError,"Concurrent"): tx.restore()
                self.assertEqual(path.read_bytes(),b"external edit")

    def test_orphan_renderer_blocks_recovery(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            s.w.save(root/"Manifest.json",{"protocol":s.PROTOCOL})
            child=Mock(); child.name.return_value="MetallicGPUDrivenSample.exe"; child.pid=42
            with patch("psutil.process_iter",return_value=[child]), patch.object(s.e,"recover") as restore:
                with self.assertRaisesRegex(ValueError,"still running"): s.recover(root)
                restore.assert_not_called()
            with patch("psutil.process_iter",return_value=[]), patch.object(s.e,"recover",return_value={"state":"restored"}):
                self.assertEqual(s.recover(root)["state"],"restored")

    def test_unfinished_or_tampered_archive_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            s.w.save(root/"Manifest.json",{"protocol":s.PROTOCOL,"status":"failed"})
            with self.assertRaisesRegex(ValueError,"Unfinished"): s.verify(root,Path("unused"))
            s.w.save(root/"Manifest.json",{"protocol":s.PROTOCOL,"status":"verified","artifacts":{}})
            (root/"extra.txt").write_text("tamper")
            with self.assertRaisesRegex(ValueError,"hash changed"): s.verify(root,Path("unused"))


if __name__=="__main__": unittest.main()
