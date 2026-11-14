"""Summarize CTest --output-junit and GoogleTest XML without treating skips as passes.

Exit codes: 0 = no failures; 1 = failures (or skips with --require-all);
2 = input, XML, or report-writing error. --require-all only controls this tool.
Counts come from testcase records, never the aggregate XML attributes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _element_reason(element, fallback):
    return element.get("message", "").strip() or "".join(element.itertext()).strip() or fallback


def _case_result(case):
    # A failure must not be hidden by a simultaneous skipped/disabled marker.
    failures = [child for child in case if child.tag in ("failure", "error")]
    if failures:
        return "failed", "\n".join(_element_reason(child, child.tag) for child in failures)

    status = case.get("status", "").lower()
    result = case.get("result", "").lower()
    if status in ("fail", "failed", "error"):
        return "failed", f"status={status}"

    skipped = case.findall("skipped")
    if skipped:
        return "skipped", "\n".join(_element_reason(child, "skipped (reason not provided)")
                                    for child in skipped)
    if status in ("notrun", "disabled", "suppressed") or result == "suppressed":
        markers = [f"{key}={value}" for key, value in (("status", status), ("result", result)) if value]
        return "skipped", f"disabled/not run ({', '.join(markers)})"
    if status == "skipped" or result == "skipped":
        return "skipped", "skipped (reason not provided)"
    return "passed", ""


def _suite_cases(element):
    """Only accept cases in standard suite containers, not arbitrary XML descendants."""
    for child in element:
        if child.tag == "testcase":
            if element.tag != "testsuite":
                raise ValueError("testcase must be inside a testsuite")
            yield element, child
        elif child.tag in ("testsuite", "testsuites"):
            yield from _suite_cases(child)


def summarize(paths):
    """Return schemaVersion=1 data, retaining every case even if names repeat.

    Each input must independently contain at least one testcase. Parse and I/O
    errors propagate to main, so a bad file cannot produce partial success.
    """
    tests = []
    for path in paths:
        source = str(path)
        try:
            root = ET.parse(path).getroot()
            if root.tag not in ("testsuite", "testsuites"):
                raise ValueError(f"expected testsuite or testsuites root, got {root.tag!r}")
            cases = list(_suite_cases(root))
            if not cases:
                raise ValueError("no testcase records found")
        except (OSError, ET.ParseError, ValueError) as error:
            raise ValueError(f"{source}: {error}") from error
        for suite, case in cases:
            status, reason = _case_result(case)
            tests.append({"source": source, "suite": suite.get("name", ""),
                          "name": case.get("name", ""), "status": status, "reason": reason})

    if not tests:
        raise ValueError("no testcase records found")
    counts = {status: sum(test["status"] == status for test in tests)
              for status in ("passed", "skipped", "failed")}
    counts["total"] = len(tests)
    skip_reasons = Counter(test["reason"] for test in tests if test["status"] == "skipped")
    return {"schemaVersion": 1, "counts": counts, "tests": tests,
            "skipReasons": dict(sorted(skip_reasons.items()))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CTest / GoogleTest XML paths")
    parser.add_argument("--json", type=Path, metavar="PATH", help="write schemaVersion=1 JSON report")
    parser.add_argument("--require-all", action="store_true",
                        help="exit 1 for any skipped case as well as failures; does not change the test harness")
    args = parser.parse_args(argv)
    try:
        if args.json is not None and args.json.resolve() in (path.resolve() for path in args.inputs):
            raise ValueError(f"--json output must not overwrite an input XML: {args.json}")
        report = summarize(args.inputs)
        if args.json is not None:
            args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    counts = report["counts"]
    print(" ".join(f"{key}={value}" for key, value in counts.items()))
    for test in report["tests"]:
        detail = f"[{test['status']}] {test['source']} :: {test['suite']} :: {test['name']}"
        if test["reason"]:
            detail += f" : {test['reason']}"
        print(detail)
    if report["skipReasons"]:
        print("Skip reasons:")
        for reason, count in report["skipReasons"].items():
            print(f"  {count}: {reason}")
    return 1 if counts["failed"] or (args.require_all and counts["skipped"]) else 0


if __name__ == "__main__":
    sys.exit(main())
