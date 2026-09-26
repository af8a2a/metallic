"""Bounded, serial real-GPU acceptance for Metallic Shader Printf P0 (stdlib only)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(case, report, raw, code, stdout):
    if report.get("schema") != "metallic.shader-printf.p0.v1" or report.get("phase") != "finished":
        return False
    if report.get("performanceEligible") is not False or report.get("instrumentation") is not True:
        return False
    caps = report.get("capabilities", {})
    if case == "missing-layer":
        return (code == 2 and report.get("smokeVerified") is False and report.get("status") == "failed"
                and caps.get("layerDiscovered") is False and caps.get("instanceConfigured") is False)
    if not all(caps.get(key) for key in ("layerDiscovered", "instanceConfigured", "messengerConfigured", "deviceConfigured")):
        return False
    if not report.get("gpuCompleted") or "error" in report:
        return False
    if report.get("hostDropped") != 0 or report.get("hostTruncated") != 0 or any(item.get("truncated") for item in raw):
        return False
    messages = [item["text"] for item in raw if item["id"] == 0x4fe1fef9 and item["severity"] == 16]
    echoes = [line.strip() for msg in messages for line in msg.splitlines() if line.startswith("MTP0 ")]
    for item in raw:
        expected_redirect_notice = (case == "stdout" and item.get("idName") == "VALIDATION-SETTINGS"
            and item["severity"] == 256 and
            item["text"] == "vkCreateInstance(): The debug callback is already logging to stdout, but printf_to_stdout is also enabled. DebugPrintf will skip the debug callback in favor of a direct stdout write.")
        if item["severity"] & (256 | 4096) and not expected_redirect_notice:
            return False
    if case in ("no-info", "stdout"):
        valid = code == 2 and report.get("status") == "incomplete" and not report.get("smokeVerified") and not echoes
        return valid and report.get("gpuOverflow") is False and (case != "stdout" or "MTP0 ordinary seq=0 group=1 lane=3 value=73" in stdout)
    if case == "gpu-overflow":
        # Loss must be reproduced, not inferred from the chosen small capacity.
        warning = any(item["id"] == 0x4fe1fef9 and "[WARNING]" in item["text"] and
                      "truncated due to the buffer size (128)" in item["text"] for item in raw)
        return code == 2 and report.get("status") == "incomplete" and not report.get("smokeVerified") and len(echoes) < 128 and warning and report.get("gpuOverflow") is True
    expected = "MTP0 ordinary seq=0 group=1 lane=3 value=73" if case == "ordinary" else "MTP0 heap seq=0 group=1 lane=3 value=73 cookie=305397763"
    return (code == 0 and report.get("status") == "verified" and report.get("smokeVerified") is True
            and echoes == [expected] and report.get("warningOrError") is False and report.get("gpuOverflow") is False
            and (case == "ordinary" or (report.get("readback", {}).get("passed") is True and
                report.get("readback", {}).get("actual") == report.get("readback", {}).get("expected") and
                len(report["readback"]["actual"]) == 4)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    parser.add_argument("--layer-path", type=Path, help="Explicit validation layer manifest directory for this child process")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    exe, output = args.exe.resolve(), args.output.resolve()
    if not exe.is_file() or output.exists() or not 10 <= args.timeout <= 300:
        parser.error("Require an existing executable, new output directory, and timeout of 10..300 seconds")
    output.mkdir(parents=True)
    empty = output / "empty-layer-path"
    empty.mkdir()
    repository = Path(__file__).resolve().parents[1]
    manifest = {"schema": "metallic.shader-printf.p0-suite.v1", "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "performanceEligible": False, "executable": str(exe), "executableSha256": digest(exe),
                "cases": [], "dependencies": {}, "complete": False}
    # Process-local isolation only. Never edit VkConfig, registry, SDK files, or global environment.
    base_env = os.environ.copy()
    manifest["inheritedVulkanEnvironment"] = {k: v for k, v in base_env.items() if k.startswith(("VK_", "METALLIC_SLANG", "METALLIC_NVPERF"))}
    for key in list(base_env):
        if key.startswith(("VK_LAYER_PRINTF", "VK_LAYER_DUPLICATE_MESSAGE", "VK_LAYER_MESSAGE_ID", "METALLIC_NVPERF")):
            base_env.pop(key)
    base_env["VK_LAYER_SETTINGS_PATH"] = str(empty)
    base_env["VK_IMPLICIT_LAYER_PATH"] = str(empty)
    if args.layer_path:
        base_env["VK_LAYER_PATH"] = str(args.layer_path.resolve())
    manifest["processVulkanEnvironment"] = {k: v for k, v in base_env.items() if k.startswith("VK_")}
    for case in ("ordinary", "heap-mapped", "heap-native", "no-info", "stdout", "gpu-overflow", "missing-layer"):
        mode = case if case.startswith("heap-") else "ordinary"
        fault = case if case in ("no-info", "stdout", "gpu-overflow") else "none"
        env = base_env.copy()
        if case == "missing-layer":
            env["VK_LAYER_PATH"] = str(empty)
            env.pop("VK_ADD_LAYER_PATH", None)
        command = [str(exe), "--mode", mode, "--fault", fault, "--output", str(output / case)]
        start = time.monotonic()
        entry = {"case": case, "command": command, "environmentOverrides": {k: v for k, v in env.items() if base_env.get(k) != v}}
        with (output / (case + ".stdout.log")).open("wb") as stdout, (output / (case + ".stderr.log")).open("wb") as stderr:
            try:
                proc = subprocess.run(command, cwd=repository, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                entry["exitCode"] = proc.returncode
            except subprocess.TimeoutExpired:
                entry["timeout"] = True
        entry["elapsedSeconds"] = round(time.monotonic() - start, 3)
        try:
            report = read(output / case / "Report.json")
            raw = read(output / case / "RawMessages.json")
            stdout_text = (output / (case + ".stdout.log")).read_text(encoding="utf-8", errors="replace")
            entry["accepted"] = not entry.get("timeout") and check(case, report, raw, entry.get("exitCode"), stdout_text)
            entry["probeStatus"] = report.get("status")
            entry["receivedRecords"] = report.get("receivedRecords")
            paths = [report.get("loadedLayerModule"), report.get("slang", {}).get("module"), report.get("slang", {}).get("apiModule")]
            paths += report.get("compiler", {}).get("dependencies", [])
            for path in paths:
                if path and Path(path).is_file():
                    manifest["dependencies"][str(Path(path).resolve())] = digest(path)
        except (OSError, ValueError, KeyError) as error:
            entry["accepted"] = False
            entry["evidenceError"] = str(error)
        manifest["cases"].append(entry)
        write(output / "Suite.json", manifest)
        print(f"{case}: accepted={entry['accepted']} status={entry.get('probeStatus')} exit={entry.get('exitCode')}", flush=True)
        if entry.get("timeout") or entry.get("exitCode") not in (0, 2):
            break  # Do not continue GPU work after a hung/crashed probe.
    manifest["complete"] = len(manifest["cases"]) == 7
    manifest["passed"] = manifest["complete"] and all(item["accepted"] for item in manifest["cases"])
    write(output / "Suite.json", manifest)
    write(output / "Hashes.json", {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
