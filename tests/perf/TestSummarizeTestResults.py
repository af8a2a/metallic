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
import xml.etree.ElementTree as ET

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

    def baseline(self, skips=None, name="known-skips.json"):
        if skips is None:
            skips = [{"suite": "GPU", "name": "NoDevice", "reason": "Device unavailable",
                      "category": "capability", "requires": "GPU device",
                      "source": "source-backed, not GPU observed; synthetic CPU fixture"}]
        data = {"schemaVersion": 1, "profiles": {
            "limited": {"description": "Limited source-backed profile, not GPU observed", "skips": skips}}}
        return self.fixture(json.dumps(data), name)

    def test_known_skip_stays_skipped_and_strict_still_fails(self):
        path = self.fixture('<testsuite name="GPU"><testcase name="NoDevice">'
                            '<skipped message="Device unavailable"/></testcase></testsuite>')
        baseline = self.baseline()
        normal, strict = self.directory / "normal.json", self.directory / "strict.json"
        args = (path, "--known-skips", baseline, "--skip-profile", "limited")
        code, stdout, stderr = self.invoke(*args, "--json", normal)
        self.assertEqual((code, stderr), (0, ""))
        report = json.loads(normal.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"passed": 0, "skipped": 1, "failed": 0, "total": 1})
        self.assertEqual(report["skipBaseline"], {"profile": "limited", "known": 1, "unexpected": 0})
        self.assertIn("known=1 unexpected=0", stdout)
        self.assertEqual(self.invoke(*args, "--require-all", "--json", strict)[0], 1)
        self.assertEqual(normal.read_bytes(), strict.read_bytes())

    def test_new_case_suite_or_reason_is_unexpected(self):
        baseline = self.baseline()
        for suite, name, reason in (("GPU", "NewCase", "Device unavailable"),
                                    ("Other", "NoDevice", "Device unavailable"),
                                    ("GPU", "NoDevice", "New reason"),
                                    ("GPU", "NoDevice", "device unavailable"),
                                    ("GPU", "NoDevice", "Device unavailable\nextra detail")):
            with self.subTest(suite=suite, name=name, reason=reason):
                xml = ET.Element("testsuite", name=suite)
                ET.SubElement(ET.SubElement(xml, "testcase", name=name), "skipped", message=reason)
                path = self.fixture(ET.tostring(xml, encoding="unicode"))
                output = self.directory / "unexpected.json"
                code, stdout, _ = self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited",
                                             "--json", output)
                self.assertEqual(code, 1)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["skipBaseline"], {"profile": "limited", "known": 0, "unexpected": 1})
                self.assertIn("unexpected=1", stdout)

    def test_google_test_location_only_is_removed_for_matching(self):
        baseline = self.baseline()
        for location in (r"E:\checkout\tests\rhi\main.cpp:359", "/checkout/main.cc:1",
                         "main.cxx:12", "test.hpp:22", "test.h:5", "test.C:7"):
            with self.subTest(location=location):
                reason = location + "\nDevice unavailable"
                xml = ET.Element("testsuite", name="GPU")
                ET.SubElement(ET.SubElement(xml, "testcase", name="NoDevice"), "skipped", message=reason)
                path = self.fixture(ET.tostring(xml, encoding="unicode"))
                output = self.directory / "location.json"
                code, stdout, _ = self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited",
                                             "--json", output)
                self.assertEqual(code, 0)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(report["tests"][0]["reason"], reason)
                self.assertEqual(report["skipReasons"], {reason: 1})
                self.assertIn(reason, stdout)

    def test_non_location_prefixes_and_later_locations_are_not_removed(self):
        baseline = self.baseline()
        for prefix in ("unavailable:359", "main.txt:359", "main.cpp:no", "main.cpp:0",
                       "main.cpp:359 extra", "main.cpp:359:", "main.cpp(359)",
                       "prefix\nmain.cpp:359"):
            with self.subTest(prefix=prefix):
                xml = ET.Element("testsuite", name="GPU")
                ET.SubElement(ET.SubElement(xml, "testcase", name="NoDevice"), "skipped",
                              message=prefix + "\nDevice unavailable")
                path = self.fixture(ET.tostring(xml, encoding="unicode"))
                self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited")[0], 1)

    def test_baseline_does_not_forgive_failure_even_with_known_skip_marker(self):
        baseline = self.baseline()
        path = self.fixture('''<testsuite name="GPU"><testcase name="NoDevice" status="notrun">
          <skipped message="Device unavailable"/><failure message="real assertion"/>
          </testcase></testsuite>''')
        output = self.directory / "failed.json"
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited",
                                     "--json", output)[0], 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["failed"], 1)
        self.assertEqual(report["skipBaseline"], {"profile": "limited", "known": 0, "unexpected": 0})
        self.assertEqual(report["tests"][0]["reason"], "real assertion")

    def test_known_and_unexpected_counts_preserve_multifile_duplicates(self):
        baseline = self.baseline()
        path = self.fixture(MIXED)
        other = self.fixture(MIXED, "other.xml")
        output = self.directory / "multi.json"
        self.assertEqual(self.invoke(path, other, "--known-skips", baseline, "--skip-profile", "limited",
                                     "--json", output)[0], 1)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"passed": 2, "skipped": 6, "failed": 0, "total": 8})
        self.assertEqual(report["skipBaseline"], {"profile": "limited", "known": 2, "unexpected": 4})

    def test_baseline_does_not_require_every_known_skip_to_appear(self):
        baseline = self.baseline()
        path = self.fixture('<testsuite><testcase name="Pass"/></testsuite>')
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited",
                                     "--require-all")[0], 0)

    def test_baseline_is_opt_in_and_no_baseline_schema_is_unchanged(self):
        path = self.fixture(MIXED)
        output = self.directory / "ordinary.json"
        code, stdout, _ = self.invoke(path, "--json", output)
        self.assertEqual(code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(set(report), {"schemaVersion", "counts", "tests", "skipReasons"})
        self.assertEqual(report, summary.summarize([path]))
        self.assertNotIn("baseline", stdout.lower())

    def test_unknown_profile_and_invalid_baselines_return_two(self):
        path = self.fixture(MIXED)
        good = json.loads(self.baseline().read_text(encoding="utf-8"))
        invalid = [None, [], {}, {**good, "schemaVersion": 2}, {**good, "schemaVersion": True},
                   {"schemaVersion": 1, "profiles": {}}, {"schemaVersion": 1, "profiles": []},
                   {"schemaVersion": 1, "profiles": {"limited": []}},
                   {"schemaVersion": 1, "profiles": {"limited": {"description": "x", "skips": []}}}]
        for key in ("suite", "name", "reason", "category", "requires", "source"):
            for value in (None, "", "  ", [], 1):
                data = json.loads(json.dumps(good))
                data["profiles"]["limited"]["skips"][0][key] = value
                invalid.append(data)
            data = json.loads(json.dumps(good))
            del data["profiles"]["limited"]["skips"][0][key]
            invalid.append(data)
        invalid.append({"schemaVersion": 1, "profiles": {"limited": {"description": "x", "skips": [None]}}})
        duplicate = json.loads(json.dumps(good))
        duplicate["profiles"]["limited"]["skips"] *= 2
        invalid.append(duplicate)
        for data in invalid:
            with self.subTest(data=data):
                baseline = self.fixture(json.dumps(data), "invalid.json")
                code, stdout, stderr = self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited")
                self.assertEqual(code, 2)
                self.assertTrue(stderr)
                self.assertNotIn("passed=", stdout)
        for baseline, profile in ((self.baseline(), "unknown"), (self.directory / "missing.json", "limited"),
                                  (self.directory, "limited"), (self.fixture("{", "bad.json"), "limited")):
            with self.subTest(baseline=baseline, profile=profile):
                self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", profile)[0], 2)

    def test_only_selected_profile_can_allow_a_skip(self):
        path = self.fixture('<testsuite name="GPU"><testcase name="NoDevice">'
                            '<skipped message="Device unavailable"/></testcase></testsuite>')
        baseline = self.baseline()
        data = json.loads(baseline.read_text(encoding="utf-8"))
        other = json.loads(json.dumps(data["profiles"]["limited"]))
        other["skips"][0]["reason"] = "Other unavailable capability"
        data["profiles"]["other"] = other
        baseline.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited")[0], 0)
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "other")[0], 1)

    def test_unselected_profiles_are_validated_too(self):
        path = self.fixture(MIXED)
        baseline = self.baseline()
        data = json.loads(baseline.read_text(encoding="utf-8"))
        data["profiles"]["broken"] = {"skips": []}
        baseline.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited")[0], 2)

    def test_wildcards_are_not_patterns(self):
        path = self.fixture(MIXED)
        baseline = self.baseline([{"suite": "*", "name": "*", "reason": "*",
                                   "category": "opt-in", "requires": "explicit opt in", "source": "fixture"}])
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited")[0], 1)

    def test_json_cannot_overwrite_baseline(self):
        path = self.fixture(MIXED)
        baseline = self.baseline()
        original = baseline.read_bytes()
        self.assertEqual(self.invoke(path, "--known-skips", baseline, "--skip-profile", "limited",
                                     "--json", baseline)[0], 2)
        self.assertEqual(baseline.read_bytes(), original)

    def test_baseline_cli_paired_arguments_and_exit_codes(self):
        path = self.fixture('<testsuite name="GPU"><testcase name="NoDevice">'
                            '<skipped message="Device unavailable"/></testcase></testsuite>')
        baseline = self.baseline()
        paired = ["--known-skips", baseline, "--skip-profile", "limited"]
        for extra, expected in ((paired, 0), (paired + ["--require-all"], 1),
                                (["--known-skips", baseline], 2), (["--skip-profile", "limited"], 2),
                                (["--known-skips", baseline, "--skip-profile", "unknown"], 2)):
            with self.subTest(extra=extra):
                result = subprocess.run([sys.executable, "-B", str(SOURCE), str(path), *map(str, extra)],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_repository_profile_is_limited_source_backed_and_matches_sources(self):
        root = SOURCE.parent.parent
        data = json.loads((root / "tests/KnownSkips.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schemaVersion"], 1)
        profile = data["profiles"]["minizorah-opt-in-disabled"]
        self.assertIn("source-backed", profile["description"])
        self.assertIn("not GPU observed", profile["description"])
        self.assertIn("limited", profile["description"].lower())
        self.assertEqual(len(profile["skips"]), 2)
        expected = [("RHIRendering", "minizorah_typed_final_blit_frames", "Set METALLIC_TEST_MINIZORAH=1",
                     "tests/rhi/FinalBlitTests.cpp"),
                    ("RHIValidation", "minizorah_quality_audit", "Opt in with METALLIC_TEST_MINIZORAH=1",
                     "tests/rhi/MiniZorahQualityTests.cpp")]
        xml = ET.Element("testsuites")
        for entry, (suite, name, reason, source) in zip(profile["skips"], expected):
            self.assertEqual((entry["suite"], entry["name"], entry["reason"]), (suite, name, reason))
            self.assertIn(source, entry["source"])
            self.assertIn("source-backed", entry["source"])
            self.assertIn("not GPU observed", entry["source"])
            text = (root / source).read_text(encoding="utf-8")
            self.assertIn(f'name = "{name}"', text)
            self.assertIn(f'RHITestResult::skip("{reason}")', text)
            case = ET.SubElement(ET.SubElement(xml, "testsuite", name=suite), "testcase", name=name)
            ET.SubElement(case, "skipped", message="tests/rhi/main.cpp:359\n" + reason)
        path = self.fixture(ET.tostring(xml, encoding="unicode"))
        output = self.directory / "repository-profile.json"
        self.assertEqual(self.invoke(path, "--known-skips", root / "tests/KnownSkips.json",
                                     "--skip-profile", "minizorah-opt-in-disabled", "--json", output)[0], 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"], {"passed": 0, "skipped": 2, "failed": 0, "total": 2})
        self.assertEqual(report["skipBaseline"], {
            "profile": "minizorah-opt-in-disabled", "known": 2, "unexpected": 0})

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
