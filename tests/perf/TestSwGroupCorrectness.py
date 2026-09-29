import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools"))
from AnalyzeZorahFullRasterComparison import analyze


class SwGroupCorrectness(unittest.TestCase):
    def fixture(self, root):
        cases, segments = [], []
        for round_id in (1, 2):
            filename = f"live{round_id}.json"
            shader = {"groupSize": 32, "entryPoint": "streamClusterRasterGroup32Main", "spirvFnv1a64": "32"}
            live = [{"shader": shader, "residentPages": n, "uploads": 1, "evictions": 1,
                     "loadFailures": 0, "requestOverflows": 0, "blasOverflowCount": 0} for n in (10, 11)]
            (root / filename).write_text(json.dumps(live))
            segments.append({"frames": 2, "seconds": 1, "telemetryFile": filename})
            for name, size, fingerprint in (("swWorkControl", 128, "control"), ("swGroup32", 32, "32"),
                                            ("swGroup64", 64, "64"), ("swGroup128", 128, "128")):
                entry = "streamClusterRasterWorkControlMain" if name == "swWorkControl" else f"streamClusterRasterGroup{size}Main"
                stamp = {"cutHash": str(round_id), "pageMappingsHash": str(round_id), "activeGroups": 10,
                         "productionDispatches": [{"groupSize": size, "subgroupSize": 32,
                            "entryPoint": entry, "spirvFnv1a64": fingerprint}] * 2}
                for phase in ("AfterStreamEarlyBins", "AfterStreamLateBins"):
                    stamp[phase] = {"softwareClusters": 2, "hardwareClusters": 1}
                for resource in ("VBuffer.depth", "VBuffer.visibility"):
                    file = f"{round_id}-{resource}.bin"
                    (root / file).write_bytes(struct.pack("<4I", *([round_id] * 4)))
                    stamp[resource] = {"file": file, "hash": str(round_id)}
                frames_file = f"{round_id}-{name}.json"
                rows = [{"editorLoopMs": 1, "streaming": [{"residentPages": round_id}], "scopes": [
                    {"path": "Frame/Stream early", "gpuMs": .1},
                    {"path": "Frame/Stream late", "gpuMs": .1},
                    {"path": "Frame/RenderGraph GPU envelope", "gpuMs": .3}]}]
                (root / frames_file).write_text(json.dumps(rows))
                cases.append({"name": name, "variant": name, "round": round_id, "maxPixels": 8,
                              "fullHardware": False, "framesFile": frames_file, "frames": 1,
                              "before": stamp, "after": stamp})
        capture = {"status": "capture_complete", "protocol": "zorah-full-raster-comparison-v1",
                   "config": {"swGroupComparison": True, "swGroupRoamSeconds": 1, "rounds": 2,
                              "requireNonzeroLate": True}, "cases": cases, "liveRoam": segments,
                   "outputExtent": [2, 2], "renderExtent": [2, 2]}
        return capture

    def check(self, mutate=None, failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.fixture(root)
            if mutate:
                mutate(capture)
            (root / "Capture.json").write_text(json.dumps(capture))
            with contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(AssertionError):
                        analyze(root)
                else:
                    self.assertTrue(analyze(root)["groupOutputsByteEqual"])

    def test_rounds_have_independent_cut_and_reference(self):
        self.check()

    def test_zero_late_is_not_accepted(self):
        def mutate(capture):
            for case in capture["cases"]:
                case["before"]["AfterStreamLateBins"]["softwareClusters"] = 0
        self.check(mutate, failure=True)

    def test_candidate_output_cannot_use_another_round(self):
        def mutate(capture):
            stamp = capture["cases"][1]["before"]
            stamp["VBuffer.depth"] = {"file": "2-VBuffer.depth.bin", "hash": "2"}
        self.check(mutate, failure=True)

    def test_same_cut_must_hold_within_round(self):
        self.check(lambda c: c["cases"][1]["before"].update(cutHash="wrong"), failure=True)

    def test_blank_reference_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.fixture(root)
            for path in root.glob("*-VBuffer.visibility.bin"):
                path.write_bytes(bytes(16))
            (root / "Capture.json").write_text(json.dumps(capture))
            with self.assertRaises(AssertionError), contextlib.redirect_stdout(io.StringIO()):
                analyze(root)
