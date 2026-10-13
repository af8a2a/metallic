"""Validate requested workloads independently of agreement between experiment arms."""

import copy
from contextlib import redirect_stderr
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Tools/Perf"))
import RunComputePipelineExperiments as experiment


class RequestedWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        specification = dict(id="Opaque", graph="/fixture/Graph.json", width=1, height=1,
            scene=None, environment=dict(path="/fixture/sky.hdr", intensity=1, rotationDegrees=25),
            output="PathTrace.color", requiredTiming=["PathTrace", "Path trace shading"],
            captureFrames=[4, 0, 4], captureOutputs=["PathTrace.motionVectors", "PathTrace.depth"],
            cameraFrames=[dict(frame=4, node="PathTrace", camera=dict(eye=[1, 0, 4]))],
            guideCameraStep=dict(node="PathTrace", delta=[.02, 0, 0]), useRenderView=True)
        self.configuration = dict(frames=5, warmupFrames=1, timingFrames=2, cases=[specification])
        case = copy.deepcopy(specification)
        case.update(format="RGBA32F", image="color.bin", validHDR=True,
            resolvedGraph=dict(nodes=[dict(name="PathTrace")]), frames=[], captures=[])
        for frame in (1, 2):
            case["frames"].append(dict(frame=frame, executionId=frame, warmup=False,
                graphMs=3., asyncComputeBranches=0, nodes=[dict(name="PathTrace", queue=0, gpuMs=2.,
                    sections=[dict(name="Path trace shading", gpuMs=1.)])]))
        case["captures"] = [
            dict(output="PathTrace.color", frame=0, file="temporal-0.bin", bytes=16, format=42),
            dict(output="PathTrace.color", frame=4, file="temporal-4.bin", bytes=16, format=42),
            dict(output="PathTrace.motionVectors", frame=5, file="motion.bin", bytes=4, format=25),
            dict(output="PathTrace.depth", frame=6, file="depth.bin", bytes=4, format=33)]
        self.report = dict(validation=False, cases=[case])
        for arm in ("A", "B"):
            directory = self.root / arm
            directory.mkdir()
            for name in ("color.bin", "temporal-0.bin", "temporal-4.bin"):
                (directory / name).write_bytes(struct.pack("<4f", 1, 2, 3, 1))
            (directory / "motion.bin").write_bytes(struct.pack("<2e", .1, .2))
            (directory / "depth.bin").write_bytes(struct.pack("<f", .5))
        self.write_reports(self.report)

    def tearDown(self):
        self.temporary.cleanup()

    def write_reports(self, report):
        for arm in ("A", "B"):
            experiment.save(self.root / arm / "MaterialBaseline.json", report)

    def compare(self):
        return experiment.compare_readbacks(self.root / "A/MaterialBaseline.json",
            self.root / "B/MaterialBaseline.json", self.configuration, 2e-5, 2e-5)

    def test_matching_temporal_and_guide_capture_schedule_passes(self):
        self.assertTrue(self.compare()[0]["qualified"])
        self.assertEqual(experiment.gpu_timings(self.report, self.configuration)["Opaque"]["dispatch_scope_ms"]["median"], 1.)

    def test_identical_arms_with_missing_requested_guides_are_rejected(self):
        for captures in ([], self.report["cases"][0]["captures"][:2], self.report["cases"][0]["captures"][2:]):
            changed = copy.deepcopy(self.report)
            changed["cases"][0]["captures"] = copy.deepcopy(captures)
            self.write_reports(changed)
            with self.subTest(captures=captures), self.assertRaisesRegex(ValueError, "requested readback"):
                self.compare()
            with self.assertRaisesRegex(ValueError, "requested readback"):
                experiment.gpu_timings(changed, self.configuration)

    def test_wrong_duplicate_and_reordered_capture_frames_are_rejected(self):
        for mutation in (lambda captures: captures[2].update(frame=4),
                         lambda captures: captures.append(copy.deepcopy(captures[-1])),
                         lambda captures: captures.reverse()):
            changed = copy.deepcopy(self.report)
            mutation(changed["cases"][0]["captures"])
            self.write_reports(changed)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "requested readback"):
                self.compare()

    def test_identical_arms_with_wrong_requested_workload_are_rejected(self):
        for field, value in (("graph", "/fixture/OtherGraph.json"), ("width", 2), ("height", 2),
                ("output", "Other.color"), ("scene", "Other.gltf"),
                ("environment", dict(path="/fixture/other.hdr", intensity=1, rotationDegrees=25)),
                ("cameraFrames", []), ("guideCameraStep", {}), ("useRenderView", False),
                ("captureOutputs", []), ("captureFrames", [])):
            changed = copy.deepcopy(self.report)
            changed["cases"][0][field] = value
            self.write_reports(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Requested production workload"):
                self.compare()
            with self.assertRaisesRegex(ValueError, "Requested production workload"):
                experiment.gpu_timings(changed, self.configuration)

    def test_case_frame_count_overrides_the_global_guide_start(self):
        configuration = copy.deepcopy(self.configuration)
        configuration["cases"][0]["frameCount"] = 7
        case = copy.deepcopy(self.report["cases"][0])
        case["frameCount"] = 7
        case["captures"][2]["frame"] = 7
        case["captures"][3]["frame"] = 8
        experiment.validate_requested_case(case, configuration["cases"][0], configuration)
        case["captures"][2]["frame"] = 5
        with self.assertRaisesRegex(ValueError, "requested readback"):
            experiment.validate_requested_case(case, configuration["cases"][0], configuration)

    def test_workload_without_differential_captures_requires_no_capture_entries(self):
        configuration = copy.deepcopy(self.configuration)
        for field in ("captureFrames", "captureOutputs"):
            del configuration["cases"][0][field]
        case = copy.deepcopy(self.report["cases"][0])
        case["captures"] = []
        experiment.validate_requested_case(case, configuration["cases"][0], configuration)
        case["captures"] = copy.deepcopy(self.report["cases"][0]["captures"])
        with self.assertRaisesRegex(ValueError, "requested readback"):
            experiment.validate_requested_case(case, configuration["cases"][0], configuration)

    def test_nonfinite_timeout_is_rejected_before_a_process_is_launched(self):
        for timeout in ("nan", "inf", "-inf"):
            argv = ["runner", "compile", "--executable", "not-an-executable.exe", "--request-index", "146",
                    "--output-dir", str(self.root / "must-not-be-created"), f"--timeout={timeout}"]
            with self.subTest(timeout=timeout), patch.object(sys, "argv", argv), \
                 patch.object(experiment.driver, "run_process") as process, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failed:
                    experiment.main()
                self.assertEqual(failed.exception.code, 2)
                process.assert_not_called()
                self.assertFalse((self.root / "must-not-be-created").exists())


if __name__ == "__main__":
    unittest.main()
