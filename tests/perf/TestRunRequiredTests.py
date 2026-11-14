"""CPU-only subprocess and real CTest contracts for the required-run wrapper."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "Tools/RunRequiredTests.py"
CHILD = r'''
from pathlib import Path
import sys
mode, exit_code = sys.argv[1], int(sys.argv[2])
report = next((arg.split("=xml:", 1)[1] for arg in sys.argv if arg.startswith("--gtest_output=xml:")), None)
if report and mode != "missing":
    content = {
        "pass": '<testsuite name="Synthetic"><testcase name="Pass"/></testsuite>',
        "skip": '<testsuite name="Synthetic"><testcase name="Skip"><skipped message="no GPU capability"/></testcase></testsuite>',
        "allskip": '<testsuite name="Synthetic"><testcase name="A"><skipped/></testcase><testcase name="B"><skipped/></testcase></testsuite>',
        "fail": '<testsuite name="Synthetic"><testcase name="Fail"><failure message="bad result"/></testcase></testsuite>',
        "mixed": '<testsuite name="Synthetic"><testcase name="Pass"/><testcase name="Skip"><skipped/></testcase><testcase name="Fail"><failure/></testcase></testsuite>',
        "both": '<testsuite name="Synthetic"><testcase name="Both"><skipped/><failure message="real failure"/></testcase></testsuite>',
        "disabled": '<testsuite name="Synthetic"><testcase name="Disabled" status="notrun" result="suppressed"/></testsuite>',
        "bad": '<testsuite>',
        "empty": '<testsuites tests="1"/>',
        "zero": '<testsuite name="Synthetic" tests="0"/>',
    }[mode]
    Path(report).write_text(content, encoding="utf-8")
print("synthetic child stdout", flush=True)
print("synthetic child stderr", file=sys.stderr, flush=True)
sys.exit(exit_code)
'''


class RequiredRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="required run spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = 0

    def invoke(self, mode="pass", child_exit=0, extra=(), environment=None, kind="gtest", command=None):
        self.index += 1
        output = self.root / f"new output {self.index} 路径"
        if command is None:
            command = [sys.executable, "-B", "-c", CHILD, mode, str(child_exit), "--gtest_filter=Synthetic.*", *extra]
        env = os.environ.copy()
        env.pop("GTEST_REPEAT", None)
        env.update(environment or {})
        process = subprocess.run([sys.executable, "-B", str(SOURCE), "--kind", kind,
                                  "--output-dir", str(output), "--", *map(str, command)],
                                 capture_output=True, text=True, check=False, env=env)
        return process, output

    def report(self, output, name="run.json"):
        return json.loads((output / name).read_text(encoding="utf-8"))

    def test_pass_appends_owned_xml_and_single_iteration_flags(self):
        process, output = self.invoke()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        run = self.report(output)
        self.assertEqual((run["childExit"], run["wrapperExit"]), (0, 0))
        self.assertEqual(run["command"][-3:], ["--gtest_repeat=1", "--gtest_fail_if_no_test_selected",
                                              f"--gtest_output=xml:{output / 'results.xml'}"])
        self.assertIn("synthetic child stdout", process.stdout)
        self.assertIn("synthetic child stderr", process.stderr)
        self.assertEqual(self.report(output, "summary.json")["counts"],
                         {"passed": 1, "skipped": 0, "failed": 0, "total": 1})

    def test_skip_fails_without_changing_observed_status(self):
        process, output = self.invoke("skip")
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        report = self.report(output, "summary.json")
        self.assertEqual(report["counts"]["passed"], 0)
        self.assertEqual(report["counts"]["skipped"], 1)
        self.assertEqual(report["tests"][0]["status"], "skipped")
        self.assertIn("<skipped", (output / "results.xml").read_text(encoding="utf-8"))

    def test_allskip_disabled_failure_mixed_and_failure_priority(self):
        for mode, passed, skipped, failed in (("allskip", 0, 2, 0), ("disabled", 0, 1, 0),
                                               ("fail", 0, 0, 1), ("mixed", 1, 1, 1), ("both", 0, 0, 1)):
            with self.subTest(mode=mode):
                process, output = self.invoke(mode)
                self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
                counts = self.report(output, "summary.json")["counts"]
                self.assertEqual((counts["passed"], counts["skipped"], counts["failed"]), (passed, skipped, failed))

    def test_nonzero_child_is_never_erased_by_passing_xml(self):
        for code in (1, 7, 77):
            with self.subTest(code=code):
                process, output = self.invoke("pass", code)
                self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
                self.assertEqual(self.report(output)["childExit"], code)
                self.assertEqual(self.report(output, "summary.json")["counts"]["passed"], 1)

    def test_missing_bad_empty_and_zero_selected_reports_are_errors(self):
        for mode, code in (("missing", 0), ("missing", 77), ("bad", 0), ("empty", 0), ("zero", 1)):
            with self.subTest(mode=mode, code=code):
                process, output = self.invoke(mode, code)
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertEqual(self.report(output)["childExit"], code)
                self.assertTrue(self.report(output)["errors"])
                self.assertFalse((output / "summary.json").exists())

    def test_unlaunchable_child_is_recorded(self):
        process, output = self.invoke(command=[str(self.root / "missing executable"), "--gtest_filter=*"])
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertIsNone(self.report(output)["childExit"])
        self.assertIn("could not launch", self.report(output)["errors"][0])

    def test_existing_output_directory_is_rejected_without_overwriting(self):
        process, output = self.invoke()
        self.assertEqual(process.returncode, 0, process.stderr)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        process = subprocess.run([sys.executable, "-B", str(SOURCE), "--kind", "gtest", "--output-dir", str(output),
                                  "--", sys.executable, "-c", "raise RuntimeError('must not run')", "--gtest_filter=*"],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_conflicting_gtest_modes_and_repeats_are_rejected_before_launch(self):
        for argument in ("--gtest_repeat=2", "--gtest_repeat=-1", "--gtest_repeat=0", "--gtest_repeat",
                         "--gtest_output=xml:old.xml", "--gtest_output", "--gtest_list_tests", "--tb-run"):
            with self.subTest(argument=argument):
                process, output = self.invoke(extra=[argument])
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertFalse(output.exists())

    def test_repeat_environment_is_checked_even_with_explicit_one(self):
        for value in ("2", "-1", "0", ""):
            with self.subTest(value=value):
                process, output = self.invoke(extra=["--gtest_repeat=1"], environment={"GTEST_REPEAT": value})
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertFalse(output.exists())
        process, _ = self.invoke(extra=["--gtest_repeat=1"], environment={"GTEST_REPEAT": "1"})
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_gtest_requires_one_explicit_nonempty_filter(self):
        for filters in ([], ["--gtest_filter="], ["--gtest_filter=A", "--gtest_filter=B"]):
            with self.subTest(filters=filters):
                process, output = self.invoke(command=[sys.executable, "-c", "raise RuntimeError('must not run')", *filters])
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertFalse(output.exists())

    def test_conflicting_ctest_modes_and_repeats_are_rejected_before_launch(self):
        for argument in ("--repeat", "--repeat=until-pass:2", "--repeat-until-fail", "--repeat-until-fail=2",
                         "--output-junit", "--output-junit=old.xml", "-N", "--show-only=json-v1"):
            with self.subTest(argument=argument):
                process, output = self.invoke(kind="ctest", command=[sys.executable, "-c", "raise RuntimeError('must not run')", argument])
                self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
                self.assertFalse(output.exists())

    def test_separator_and_missing_command_are_errors(self):
        for command in ([], ["--"]):
            process = subprocess.run([sys.executable, "-B", str(SOURCE), "--kind", "gtest", "--output-dir", str(self.root / "unused"), *command],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 2, process.stdout + process.stderr)
        self.assertFalse((self.root / "unused").exists())

    @unittest.skipUnless(shutil.which("ctest"), "CTest is required for the real editor-marker probe")
    def test_real_ctest_editor_markers_return77_pass_fail_disabled_and_zero(self):
        script = self.root / "fake editor.py"
        script.write_text("import sys\nprint(sys.argv[1])\nsys.exit(int(sys.argv[2]))\n", encoding="utf-8")
        # Direct CTest testfile: no configure, compiler, build, renderer, or GPU.
        python = Path(sys.executable).as_posix()
        (self.root / "CTestTestfile.cmake").write_text(
            f'add_test(Pass "{python}" "{script.as_posix()}" "presented" "0")\n'
            f'add_test(Fail "{python}" "{script.as_posix()}" "failed" "9")\n'
            f'add_test(PipelineSkip "{python}" "{script.as_posix()}" "SKIP: no internal-cache control" "0")\n'
            'set_tests_properties(PipelineSkip PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP:")\n'
            f'add_test(ReflexSkip "{python}" "{script.as_posix()}" "Reflex unavailable; skipping" "0")\n'
            'set_tests_properties(ReflexSkip PROPERTIES SKIP_REGULAR_EXPRESSION "Reflex unavailable; skipping")\n'
            f'add_test(Return77 "{python}" "{script.as_posix()}" "no device" "77")\n'
            'set_tests_properties(Return77 PROPERTIES SKIP_RETURN_CODE 77)\n'
            f'add_test(Disabled "{python}" "{script.as_posix()}" "must not run" "9")\n'
            'set_tests_properties(Disabled PROPERTIES DISABLED TRUE)\n', encoding="utf-8")
        for case, expected, status in (("Pass", 0, "passed"), ("Fail", 1, "failed"),
                                       ("PipelineSkip", 1, "skipped"), ("ReflexSkip", 1, "skipped"),
                                       ("Return77", 1, "skipped"), ("Disabled", 1, "skipped"),
                                       ("DoesNotExist", 2, None)):
            with self.subTest(case=case):
                process, output = self.invoke(kind="ctest", command=[shutil.which("ctest"), "--test-dir", str(self.root),
                                                                       "-R", f"^{case}$", "--output-on-failure"])
                self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
                run = self.report(output)
                if status:
                    report = self.report(output, "summary.json")
                    self.assertEqual(report["tests"][0]["status"], status)
                    self.assertEqual(report["counts"]["total"], 1)
                    if status == "skipped" and case != "Disabled":
                        self.assertEqual(run["childExit"], 0)
                    if status == "failed":
                        self.assertNotEqual(run["childExit"], 0)
                else:
                    self.assertNotEqual(run["childExit"], 0)
                self.assertEqual(run["command"][-3:], ["--no-tests=error", "--output-junit", str(output / "results.xml")])


if __name__ == "__main__":
    unittest.main()
