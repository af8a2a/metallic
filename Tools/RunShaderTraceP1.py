"""Serial, bounded real-GPU/CLI/export/offline acceptance for Shader Trace P1 (stdlib only)."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"$type", "value"} and value["$type"] in ("u64", "i64"):
            return int(value["value"])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def read(path):
    return decode(json.loads(path.read_text(encoding="utf-8")))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-path", type=Path, required=True)
    args = parser.parse_args()
    exe, cli, output = args.exe.resolve(), args.cli.resolve(), args.output.resolve()
    if not exe.is_file() or not cli.is_file() or output.exists():
        parser.error("Require existing probe/CLI and a new output directory")
    output.mkdir(parents=True)
    empty = output / "empty-layer-path"
    empty.mkdir()
    env = os.environ.copy()
    inherited = {k: v for k, v in env.items() if k.startswith(("VK_", "METALLIC_SLANG", "METALLIC_NVPERF"))}
    for key in list(env):
        if key.startswith(("VK_LAYER_PRINTF", "VK_LAYER_DUPLICATE_MESSAGE", "VK_LAYER_MESSAGE_ID", "METALLIC_NVPERF")):
            env.pop(key)
    env.update(VK_LAYER_PATH=str(args.layer_path.resolve()), VK_IMPLICIT_LAYER_PATH=str(empty), VK_LAYER_SETTINGS_PATH=str(empty))
    suite = {"schema": "metallic.shader-trace.p1-suite.v1", "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "performanceEligible": False, "fixtureOnly": True, "exeSha256": digest(exe), "cliSha256": digest(cli),
             "inheritedEnvironment": inherited, "processEnvironment": {k: v for k, v in env.items() if k.startswith("VK_")},
             "dependencies": {}, "modes": [], "passed": False}
    repository = Path(__file__).resolve().parents[1]
    for mode in ("heap-mapped", "heap-native"):
        mode_dir = output / mode
        mode_dir.mkdir()
        entry = {"mode": mode, "cases": [], "passed": False}
        suite["modes"].append(entry)
        command = [str(exe), "--serve", "--mode", mode, "--serve-seconds", "25", "--output", str(mode_dir / "service")]
        entry["command"] = command
        proc = None
        with (mode_dir / "stdout.log").open("wb") as stdout, (mode_dir / "stderr.log").open("wb") as stderr:
            try:
                proc = subprocess.Popen(command, cwd=repository, env=env, stdout=stdout, stderr=stderr,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                deadline = time.monotonic() + 30
                ready = None
                while time.monotonic() < deadline:
                    require(proc.poll() is None, f"Fixture service exited during startup: {proc.returncode}")
                    try:
                        ready = read(mode_dir / "service" / "Server.json")
                        break
                    except (OSError, ValueError):
                        time.sleep(0.05)
                require(ready is not None, "Fixture service startup deadline exceeded")
                session = ready["result"]["session"]
                entry.update(pid=proc.pid, session=session)
                counter = 0

                def invoke(words, *, online=True, expect_error=False):
                    nonlocal counter
                    counter += 1
                    cmd = [str(cli), "--json"]
                    if online:
                        cmd += ["--pid", str(proc.pid), "--session", session]
                    cmd += [str(word) for word in words]
                    run = subprocess.run(cmd, cwd=repository, capture_output=True, timeout=15,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    value = decode(json.loads(run.stdout.decode("utf-8")))
                    write(mode_dir / f"cli-{counter:02}.json", {"command": cmd, "exitCode": run.returncode,
                          "response": value, "stderr": run.stderr.decode("utf-8", errors="replace")})
                    require((run.returncode != 0 and value["status"] == "error") if expect_error else
                            (run.returncode == 0 and value["status"] == "ok"), f"CLI failure: {value}")
                    return value

                caps = invoke(["shader", "capabilities"])["result"]
                require(caps["smokeVerified"] and caps["fixtureOnly"], "Missing backend qualification")
                site = invoke(["shader", "sites"])["result"][0]
                entry["capabilities"] = caps
                runtime = caps["runtime"]
                binary_paths = [runtime.get("loadedLayerModule"), runtime.get("slang", {}).get("module"), runtime.get("slang", {}).get("apiModule")]
                for path in binary_paths:
                    if path and Path(path).is_file():
                        suite["dependencies"][str(Path(path).resolve())] = digest(path)
                previous_token = 0
                for scenario in ("matched", "no-match", "site-not-reached", "missing-end", "quota"):
                    case = {"scenario": scenario, "accepted": False}
                    entry["cases"].append(case)
                    spec = {"version": 1, "generation": 1, "target": {"site": site["name"], "expectedSiteSchemaHash": site["schemaHash"]},
                            "invocation": site["invocation"], "limits": {"targetFrames": 1, "maxRecords": 16, "timeoutMs": 15000},
                            "fixtureScenario": scenario}
                    spec_path = mode_dir / (scenario + ".spec.json")
                    write(spec_path, spec)
                    result = invoke(["shader", "watch", "--spec", spec_path, "--wait"])["result"]
                    job = result["job"]
                    case["job"] = job
                    capture = mode_dir / (scenario + "-capture")
                    invoke(["capture", "export", job, "--out", capture])
                    decoded = invoke(["--capture", capture, "shader", "verify"], online=False)["result"]
                    online = invoke(["eval", "shaderTrace", "--job", job])["result"]["value"]
                    require(online == decoded, "Online/offline analysis diverged")
                    expected = {"matched": "Matched", "no-match": "NoMatch", "site-not-reached": "SiteNotReached",
                                "missing-end": "Incomplete", "quota": "Incomplete"}[scenario]
                    require(decoded["outcome"] == expected, f"Unexpected outcome: {decoded['outcome']}")
                    require(decoded["selectedScopeComplete"] == (scenario in ("matched", "no-match", "site-not-reached")), "Incorrect completeness")
                    require(decoded["readbackStatus"] == "Verified" and decoded["targetExecutionStatus"] == "Completed", "GPU/readback incomplete")
                    require(not decoded["performanceEligible"], "Diagnostic trace must not be timing evidence")
                    require(decoded["dispatch"]["session"] == session and decoded["dispatch"]["dispatchToken"] > previous_token, "Identity reused")
                    previous_token = decoded["dispatch"]["dispatchToken"]
                    if decoded["records"]:
                        fields = decoded["records"][0]["fields"]
                        require(fields["negativeZero"]["bits"] == 0x80000000 and math.copysign(1, fields["negativeZero"]["value"]) < 0, "Lost negative zero")
                        require(fields["nanPayload"]["bits"] == 0x7fc12345 and fields["nanPayload"]["value"] is None, "Lost NaN payload")
                        require(fields["large"]["value"] == 2305843009213693953 and fields["value"]["value"] == 73, "Lost integer precision/heap value")
                    if scenario == "quota":
                        require(decoded["summary"]["budgetExceeded"] and decoded["summary"]["matchedCount"] == 15 and
                                decoded["summary"]["emittedCount"] == 14 and decoded["receivedRecordCount"] == 16, "Quota control records missing")
                    raw_bundle = read(capture / "0.bin")
                    variant = raw_bundle["runtime"]["variant"]
                    require(hashlib.sha256(bytes.fromhex(variant["compilerSpirvHex"])).hexdigest() == variant["compilerSpirvSha256"], "SPIR-V identity mismatch")
                    require(decoded["dispatch"]["variant"]["compilerSpirvSha256"] == variant["compilerSpirvSha256"], "Dispatch/compiled variant mismatch")
                    case.update(accepted=True, outcome=expected, selectedScopeComplete=decoded["selectedScopeComplete"],
                                dispatchToken=previous_token, recordCount=decoded["receivedRecordCount"])
                    print(f"{mode}/{scenario}: {expected}, complete={decoded['selectedScopeComplete']}", flush=True)
                    write(output / "Suite.json", suite)
                tampered = mode_dir / "tampered-copy"
                shutil.copytree(mode_dir / "matched-capture", tampered)
                data = bytearray((tampered / "0.bin").read_bytes())
                data[-1] ^= 1
                (tampered / "0.bin").write_bytes(data)
                error = invoke(["--capture", tampered, "shader", "verify"], online=False, expect_error=True)
                require(error["error"]["code"] == "InvalidShaderTrace", "Tampered evidence was not rejected by hash validation")
                entry["tamperRejected"] = True
                entry["exitCode"] = proc.wait(timeout=35)
                require(entry["exitCode"] == 0, "Fixture service did not shut down cleanly")
                entry["passed"] = True
            except Exception as error:
                entry["error"] = str(error)
                print(f"{mode}: FAILED: {error}", flush=True)
            finally:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
                    entry["forcedTermination"] = True
                if proc is not None:
                    entry["exitCode"] = proc.returncode
        write(output / "Suite.json", suite)
        if not entry["passed"]:
            break  # Keep failure evidence; do not continue GPU work after a failure.
    suite["passed"] = len(suite["modes"]) == 2 and all(mode["passed"] for mode in suite["modes"])
    write(output / "Suite.json", suite)
    write(output / "Hashes.json", {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
