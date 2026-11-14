"""Summarize CTest --output-junit and GoogleTest XML without treating skips as passes.

Exit codes: 0 = no failures or unexpected skips; 1 = failures, unexpected
baseline skips, or any skips with --require-all; 2 = input, XML, baseline, or
report-writing error. --require-all only controls this tool. Counts come from
testcase records, never the aggregate XML attributes. Baselines are opt-in,
exact suite/name/reason allowlists, not evidence of GPU validation or xfail.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
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


def _skip_key(test):
    reason = test["reason"]
    location, separator, remainder = reason.partition("\n")
    # GoogleTest XML prepends FormatCompilerIndependentFileLocation + newline.
    # Strip only that first C/C++ file:line, never arbitrary reason prefixes.
    if separator and re.fullmatch(r"(?:[A-Za-z]:)?[^:\r\n]+\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx):[1-9][0-9]*",
                                  location, flags=re.IGNORECASE):
        reason = remainder
    return test["suite"], test["name"], reason


def load_skip_profile(path, name):
    """Validate a schemaVersion=1 baseline and return exact keys for one profile.

    profiles maps profile names to {description, skips}; each nonempty skips
    list contains suite/name/reason plus category/requires/source strings.
    All profiles are validated, not just the selected one. Metadata describes
    the evidence and requirements, and is never used for fuzzy matching.
    """
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict) or type(baseline.get("schemaVersion")) is not int or baseline["schemaVersion"] != 1:
            raise ValueError("expected baseline schemaVersion=1")
        profiles = baseline.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("profiles must be a nonempty object")
        keys_by_profile = {}
        for profile_name, profile in profiles.items():
            if not profile_name.strip() or not isinstance(profile, dict):
                raise ValueError("each named profile must be an object")
            description = profile.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"profile {profile_name!r}: description must be a nonempty string")
            skips = profile.get("skips")
            if not isinstance(skips, list) or not skips:
                raise ValueError(f"profile {profile_name!r}: skips must be a nonempty list")
            keys = set()
            for entry in skips:
                if not isinstance(entry, dict):
                    raise ValueError(f"profile {profile_name!r}: skip must be an object")
                for field in ("suite", "name", "reason", "category", "requires", "source"):
                    value = entry.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"profile {profile_name!r}: {field} must be a nonempty string")
                key = _skip_key(entry)
                if key in keys:
                    raise ValueError(f"profile {profile_name!r}: duplicate skip {key!r}")
                keys.add(key)
            keys_by_profile[profile_name] = keys
        if name not in keys_by_profile:
            raise ValueError(f"unknown skip profile {name!r}")
        return keys_by_profile[name]
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def compare_skip_baseline(report, profile, known_skips):
    """Annotate counts without changing testcase status or original reasons."""
    skipped = [test for test in report["tests"] if test["status"] == "skipped"]
    known = sum(_skip_key(test) in known_skips for test in skipped)
    report["skipBaseline"] = {"profile": profile, "known": known, "unexpected": len(skipped) - known}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="CTest / GoogleTest XML paths")
    parser.add_argument("--json", type=Path, metavar="PATH", help="write schemaVersion=1 JSON report")
    parser.add_argument("--require-all", action="store_true",
                        help="exit 1 for any skipped case as well as failures; does not change the test harness")
    parser.add_argument("--known-skips", type=Path, metavar="PATH",
                        help="schemaVersion=1 exact skip baseline; requires --skip-profile")
    parser.add_argument("--skip-profile", metavar="NAME",
                        help="explicit profile in --known-skips; unexpected skips exit 1")
    args = parser.parse_args(argv)
    try:
        if (args.known_skips is None) != (args.skip_profile is None):
            raise ValueError("--known-skips and --skip-profile must be provided together")
        if args.json is not None and args.json.resolve() in (path.resolve() for path in args.inputs):
            raise ValueError(f"--json output must not overwrite an input XML: {args.json}")
        if args.json is not None and args.known_skips is not None and args.json.resolve() == args.known_skips.resolve():
            raise ValueError(f"--json output must not overwrite the skip baseline: {args.json}")
        known_skips = load_skip_profile(args.known_skips, args.skip_profile) if args.known_skips is not None else None
        report = summarize(args.inputs)
        if known_skips is not None:
            compare_skip_baseline(report, args.skip_profile, known_skips)
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
    unexpected = 0
    if "skipBaseline" in report:
        baseline = report["skipBaseline"]
        unexpected = baseline["unexpected"]
        print(f"Skip baseline {baseline['profile']}: known={baseline['known']} unexpected={unexpected}")
    return 1 if counts["failed"] or unexpected or (args.require_all and counts["skipped"]) else 0


if __name__ == "__main__":
    sys.exit(main())
