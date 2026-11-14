"""Run one required lane without treating XML skips as passes.

Usage: --kind gtest|ctest --output-dir NEWDIR -- COMMAND [ARGS...]
The output directory must not exist; its parent must exist. Child stdout/stderr
are inherited and childExit is retained in run.json, even when XML says pass.
Only one iteration is supported. gtest requires an explicit --gtest_filter=...
and checks inner cases; ctest checks registered jobs only (use for editor smoke,
not to infer inner GoogleTest results). This wrapper does not change a binary's
normal skip policy, the working directory, or child-specific artifact paths.
Exit 0 = all reported cases passed and child exited 0; 1 = child/test/skip failure;
2 = argument, launch, XML, or report error. Wrapper artifacts stay in NEWDIR.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from SummarizeTestResults import summarize


def required_command(kind, command, report):
    if not command:
        raise ValueError("a child command after -- is required")
    arguments = command[1:]
    if kind == "gtest":
        filters = [arg.partition("=")[2] for arg in arguments if arg.startswith("--gtest_filter=")]
        if len(filters) != 1 or not filters[0]:
            raise ValueError("gtest requires exactly one nonempty --gtest_filter=<pattern>")
        if "GTEST_REPEAT" in os.environ and os.environ["GTEST_REPEAT"] != "1":
            raise ValueError("required gtest runs only support GTEST_REPEAT=1 (or unset)")
        for argument in arguments:
            key = argument.partition("=")[0]
            if key in ("--gtest_output", "--gtest_list_tests") or key.startswith("--tb-"):
                raise ValueError(f"conflicting required gtest option: {argument}")
            if key == "--gtest_repeat" and argument != "--gtest_repeat=1":
                raise ValueError("required gtest runs only support --gtest_repeat=1")
        return [*command, "--gtest_repeat=1", "--gtest_fail_if_no_test_selected",
                f"--gtest_output=xml:{report}"]
    for argument in arguments:
        key = argument.partition("=")[0]
        if (key in ("--repeat", "--repeat-until-fail", "--output-junit", "--show-only", "-N")
                or argument.startswith("-N")):
            raise ValueError(f"conflicting required ctest option: {argument}")
    return [*command, "--no-tests=error", "--output-junit", str(report)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", required=True, choices=("gtest", "ctest"))
    parser.add_argument("--output-dir", required=True, type=Path, metavar="NEWDIR")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if not command or command[0] != "--":
        parser.error("separate the child command with --")
    output = args.output_dir.absolute()
    report_path = output / "results.xml"
    try:
        command = required_command(args.kind, command[1:], report_path)
        # A single atomic mkdir rejects old reports and competing invocations.
        output.mkdir()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    run = {"schemaVersion": 1, "kind": args.kind, "command": command,
           "childExit": None, "wrapperExit": 2, "errors": []}
    summary = None
    try:
        run["childExit"] = subprocess.run(command, check=False).returncode
    except OSError as error:
        run["errors"].append(f"could not launch child: {error}")
    if run["childExit"] is not None:
        try:
            summary = summarize([report_path])
        except (OSError, ValueError) as error:
            run["errors"].append(str(error))
    if not run["errors"]:
        counts = summary["counts"]
        run["wrapperExit"] = 1 if run["childExit"] != 0 or counts["skipped"] or counts["failed"] else 0
    try:
        if summary is not None:
            write_json(output / "summary.json", summary)
        write_json(output / "run.json", run)
    except OSError as error:
        print(f"error writing required-run reports: {error}", file=sys.stderr)
        return 2
    print(f"childExit={run['childExit']} wrapperExit={run['wrapperExit']}", flush=True)
    if summary is not None:
        print(" ".join(f"{key}={value}" for key, value in summary["counts"].items()))
        for test in summary["tests"]:
            print(f"[{test['status']}] {test['suite']}.{test['name']} : {test['reason']}")
    for error in run["errors"]:
        print(f"error: {error}", file=sys.stderr)
    return run["wrapperExit"]


if __name__ == "__main__":
    sys.exit(main())
