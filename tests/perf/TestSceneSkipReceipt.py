"""Exercise one native CPU opt-in skip and require its diagnostic receipt.

This intentionally checks a skipped case, not the Zorah asset path itself.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


CASE = "SceneLoading.ZorahFullOpenReportsMissingCook"
REASON = "Set METALLIC_TEST_ZORAH_FULL=1 for the local asset"


def check(executable):
    environment = os.environ.copy()
    environment.pop("METALLIC_TEST_ZORAH_FULL", None)
    # Only the child's opt-in is removed. The parent and machine are unchanged.
    with tempfile.TemporaryDirectory(prefix="MetallicSceneSkipReceipt-") as temporary:
        directory = Path(temporary)
        receipt = directory / "scene-skip.xml"
        result = subprocess.run(
            [str(executable.resolve()), f"--gtest_filter={CASE}", "--gtest_repeat=1",
             "--gtest_color=no", f"--gtest_output=xml:{receipt}"],
            cwd=directory, env=environment, capture_output=True, text=True, timeout=15,
        )
        print(result.stdout, end="")
        print(result.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(f"Native skip probe exited {result.returncode}")
        # The directory is fresh: no stale XML can satisfy the contract.
        document = receipt.read_text(encoding="utf-8")
        print(document)
        cases = list(ET.fromstring(document).iter("testcase"))
        if len(cases) != 1:
            raise RuntimeError(f"Expected exactly one native testcase, got {len(cases)}")
        case = cases[0]
        if (case.get("classname"), case.get("name")) != tuple(CASE.split(".", 1)):
            raise RuntimeError(f"Unexpected native testcase: {case.attrib}")
        if case.findall("failure") or case.findall("error"):
            raise RuntimeError("Native failure must not be hidden by a skip")
        skips = case.findall("skipped")
        if len(skips) != 1 or case.get("result") != "skipped":
            raise RuntimeError("Expected one explicit native opt-in skip, not a pass")
        if REASON not in skips[0].get("message", ""):
            raise RuntimeError(f"Native skip receipt is missing its reason: {REASON}")
        print("Verified one intentional native opt-in skip with its reason; not asset-path acceptance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    arguments = parser.parse_args()
    check(arguments.executable)
