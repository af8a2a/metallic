"""CPU-only XML fixtures for the CTest / GoogleTest result summary CLI."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "Tools/SummarizeTestResults.py"
spec = importlib.util.spec_from_file_location("metallic_test_summary", SOURCE)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

MIXED = """<?xml version="1.0"?>
<testsuites tests="99" failures="0" name="AllTests">
  <testsuite name="GPU" tests="99" skipped="0">
    <testcase name="Works" status="run" result="completed"/>
    <testcase name="NoDevice" status="run" result="skipped">
      <skipped message="Device unavailable"/>
    </testcase>
    <testcase name="NoExtension"><skipped>Extension unavailable</skipped></testcase>
    <testcase name="NoDeviceAgain"><skipped message="Device unavailable"/></testcase>
  </testsuite>
</testsuites>"""


class SummarizeTestResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def fixture(self, content, name="results.xml"):
        path = self.directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = summary.main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_mixed_counts_use_cases_not_root_or_suite_totals(self):
        path = self.fixture(MIXED)
        report = summary.summarize([path])
        self.assertEqual(report["schemaVersion"], 1)
        self.assertEqual(report["counts"], {"passed": 1, "skipped": 3, "failed": 0, "total": 4})
        self.assertEqual(report["skipReasons"], {"Device unavailable": 2, "Extension unavailable": 1})
        self.assertEqual(report["tests"][0], {
            "source": str(path), "suite": "GPU", "name": "Works", "status": "passed", "reason": ""})

    def test_all_skipped_does_not_count_as_passed(self):
        path = self.fixture('<testsuite name="GPU"><testcase name="Only"><skipped/></testcase></testsuite>')
        report = summary.summarize([path])
        self.assertEqual(report["counts"], {"passed": 0, "skipped": 1, "failed": 0, "total": 1})
        self.assertTrue(report["tests"][0]["reason"])
        self.assertEqual(sum(report["skipReasons"].values()), 1)
        self.assertEqual(self.invoke(path)[0], 0)
        self.assertEqual(self.invoke(path, "--require-all")[0], 1)

    def test_failure_and_error_take_priority_over_skip_and_disabled(self):
        path = self.fixture('''<testsuite name="Mixed">
          <testcase name="Failure" status="notrun" result="suppressed">
            <skipped message="not supported"/><failure message="assertion failed"/>
          </testcase>
          <testcase name="Error"><skipped/><error>process crashed</error></testcase>
          <testcase name="Skipped"><skipped message="no device"/></testcase>
        </testsuite>''')
        report = summary.summarize([path])
        self.assertEqual(report["counts"], {"passed": 0, "skipped": 1, "failed": 2, "total": 3})
        self.assertEqual([test["reason"] for test in report["tests"]],
                         ["assertion failed", "process crashed", "no device"])
        self.assertEqual(self.invoke(path)[0], 1)
        self.assertEqual(self.invoke(path, "--require-all")[0], 1)

    def test_disabled_notrun_and_suppressed_are_skipped(self):
        for attributes in ('status="notrun" result="suppressed"', 'status="notrun"',
                           'result="suppressed"', 'status="disabled"'):
            with self.subTest(attributes=attributes):
                path = self.fixture(f'<testsuite><testcase name="DISABLED_Case" {attributes}/></testsuite>')
                report = summary.summarize([path])
                self.assertEqual(report["counts"]["skipped"], 1)
                self.assertIn("disabled", report["tests"][0]["reason"].lower())

    def test_skip_result_without_element_is_not_passed(self):
        path = self.fixture('<testsuite><testcase name="Skip" result="skipped"/></testsuite>')
        self.assertEqual(summary.summarize([path])["counts"]["skipped"], 1)

    def test_ctest_single_suite_and_metadata(self):
        path = self.fixture('''<testsuite name="(empty)" tests="3" failures="1" disabled="0" skipped="1"
            hostname="" time="0" timestamp="2026-11-14T00:00:00">
          <testcase name="Pass" classname="Pass" time="0.01" status="run">
            <properties><property name="cmake_labels" value="unit"/></properties>
            <system-out>ordinary output; not a skip</system-out>
          </testcase>
          <testcase name="Skip" classname="Skip" time="0" status="notrun">
            <skipped message="SKIP_RETURN_CODE=77"/><system-out>no device</system-out>
          </testcase>
          <testcase name="Fail" classname="Fail" time="0" status="fail"><failure message="Failed"/></testcase>
        </testsuite>''')
        report = summary.summarize([path])
        self.assertEqual(report["counts"], {"passed": 1, "skipped": 1, "failed": 1, "total": 3})
        self.assertEqual(report["skipReasons"], {"SKIP_RETURN_CODE=77": 1})

    def test_real_ctest_skip_code_regex_and_disabled_format(self):
        # Structure captured from CTest --output-junit; disabled has no skipped element.
        path = self.fixture('''<?xml version="1.0" encoding="UTF-8"?>
        <testsuite name="(empty)" tests="5" failures="1" disabled="1" skipped="2"
                   hostname="" time="0" timestamp="2026-11-14T08:52:33">
          <testcase name="passed" classname="passed" time="0.020271" status="run">
            <properties/><system-out></system-out>
          </testcase>
          <testcase name="failed" classname="failed" time="0.0187257" status="fail">
            <failure message="Failed"/><properties/><system-out></system-out>
          </testcase>
          <testcase name="skip_code" classname="skip_code" time="0.0398882" status="notrun">
            <skipped message="SKIP_RETURN_CODE=77"/><properties/><system-out></system-out>
          </testcase>
          <testcase name="skip_regex" classname="skip_regex" time="0.0190451" status="notrun">
            <skipped message="SKIP_REGULAR_EXPRESSION_MATCHED"/><properties/>
            <system-out>SKIP: synthetic unavailable capability\n</system-out>
          </testcase>
          <testcase name="disabled" classname="disabled" time="0" status="disabled">
            <properties/><system-out>Disabled</system-out>
          </testcase>
        </testsuite>''')
        report = summary.summarize([path])
        self.assertEqual(report["counts"], {"passed": 1, "skipped": 3, "failed": 1, "total": 5})
        self.assertEqual(report["skipReasons"]["SKIP_RETURN_CODE=77"], 1)
        self.assertEqual(report["skipReasons"]["SKIP_REGULAR_EXPRESSION_MATCHED"], 1)
        self.assertEqual(report["tests"][-1]["status"], "skipped")
        self.assertIn("disabled", report["tests"][-1]["reason"])
        self.assertEqual(self.invoke(path)[0], 1)

    def test_ctest_fail_status_is_not_passed_without_failure_element(self):
        path = self.fixture('<testsuite><testcase name="Fail" status="fail"/></testsuite>')
        self.assertEqual(summary.summarize([path])["counts"]["failed"], 1)

    def test_multiple_files_preserve_identical_test_names(self):
        xml = '<testsuite name="Same"><testcase name="Same"/></testsuite>'
        first, second = self.fixture(xml, "first.xml"), self.fixture(xml, "second.xml")
        report = summary.summarize([first, second])
        self.assertEqual(report["counts"]["passed"], 2)
        self.assertEqual([test["source"] for test in report["tests"]], [str(first), str(second)])

    def test_nested_suites_keep_own_suite_names(self):
        path = self.fixture('''<testsuites><testsuite name="Outer"><testsuite name="Inner">
          <testcase name="Works"/></testsuite></testsuite></testsuites>''')
        self.assertEqual(summary.summarize([path])["tests"][0]["suite"], "Inner")

    def test_invalid_empty_and_unrecognized_xml_return_two(self):
        for xml in ("", "<testsuite>", "<testsuite/>", '<testsuites tests="20"/>',
                    '<report><testcase name="Fake"/></report>',
                    '<testsuite><payload><testcase name="Fake"/></payload></testsuite>',
                    '<testsuites><testcase name="Fake"/></testsuites>'):
            with self.subTest(xml=xml):
                code, stdout, stderr = self.invoke(self.fixture(xml))
                self.assertEqual(code, 2)
                self.assertTrue(stderr)
                self.assertNotIn("passed=", stdout)

    def test_missing_input_and_directory_return_two(self):
        for path in (self.directory / "missing.xml", self.directory):
            with self.subTest(path=path):
                code, _, stderr = self.invoke(path)
                self.assertEqual(code, 2)
                self.assertIn(str(path), stderr)

    def test_one_bad_input_does_not_produce_partial_success_report(self):
        valid = self.fixture(MIXED)
        empty = self.fixture("<testsuite/>", "empty.xml")
        output = self.directory / "summary.json"
        self.assertEqual(self.invoke(valid, empty, "--json", output)[0], 2)
        self.assertFalse(output.exists())

    def test_json_schema_details_and_stdout_histogram(self):
        path = self.fixture(MIXED)
        output = self.directory / "summary.json"
        code, stdout, stderr = self.invoke(path, "--json", output)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary.summarize([path]))
        for text in ("passed=1", "skipped=3", "failed=0", "Device unavailable", "Extension unavailable", "NoDevice"):
            self.assertIn(text, stdout)

    def test_require_all_changes_exit_only_not_status_or_json(self):
        path = self.fixture(MIXED)
        normal, strict = self.directory / "normal.json", self.directory / "strict.json"
        self.assertEqual(self.invoke(path, "--json", normal)[0], 0)
        self.assertEqual(self.invoke(path, "--require-all", "--json", strict)[0], 1)
        self.assertEqual(normal.read_text(encoding="utf-8"), strict.read_text(encoding="utf-8"))

    def test_all_passed_is_zero_even_when_require_all(self):
        path = self.fixture('<testsuite><testcase name="Pass"/></testsuite>')
        self.assertEqual(self.invoke(path, "--require-all")[0], 0)

    def test_report_write_error_returns_two(self):
        path = self.fixture(MIXED)
        code, _, stderr = self.invoke(path, "--json", self.directory)
        self.assertEqual(code, 2)
        self.assertTrue(stderr)

    def test_json_cannot_overwrite_any_input_or_relative_absolute_alias(self):
        path = self.fixture(MIXED)
        other = self.fixture('<testsuite><testcase name="Other"/></testsuite>', "other.xml")
        original = path.read_bytes()
        for inputs, output in (([str(path)], str(path)), ([path.name], str(path)),
                               ([str(path)], path.name), ([other.name, str(path)], path.name)):
            with self.subTest(inputs=inputs, output=output):
                path.write_bytes(original)
                result = subprocess.run([sys.executable, "-B", str(SOURCE), *inputs, "--json", output],
                                        cwd=self.directory, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(path.read_bytes(), original)
                self.assertIn("input", result.stderr.lower())

    def test_json_can_replace_an_existing_distinct_report(self):
        path = self.fixture(MIXED)
        output = self.directory / "summary.json"
        output.write_text("old report", encoding="utf-8")
        self.assertEqual(self.invoke(path, "--json", output)[0], 0)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), summary.summarize([path]))

    def test_json_is_written_for_failed_run(self):
        path = self.fixture('<testsuite><testcase name="Failure"><failure/></testcase></testsuite>')
        output = self.directory / "failed.json"
        self.assertEqual(self.invoke(path, "--json", output)[0], 1)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["counts"]["failed"], 1)

    def test_cli_process_exit_codes(self):
        passed = self.fixture('<testsuite><testcase name="Pass"/></testsuite>', "passed.xml")
        skipped = self.fixture('<testsuite><testcase name="Skip"><skipped/></testcase></testsuite>', "skip.xml")
        failed = self.fixture('<testsuite><testcase name="Fail"><failure/></testcase></testsuite>', "fail.xml")
        for args, expected in (([passed], 0), ([skipped], 0), ([skipped, "--require-all"], 1),
                               ([failed], 1), ([self.directory / "missing.xml"], 2), ([], 2)):
            with self.subTest(args=args):
                result = subprocess.run([sys.executable, "-B", str(SOURCE), *map(str, args)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
