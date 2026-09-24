"""Collect a read-only Nsight capability/evidence baseline; never launch a GPU workload.

Python standard library only. The output directory must be new. Missing evidence
is a recorded gap, not a zero measurement or proof of an unsupported feature.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


STATUSES = {"verified", "supported-unverified", "unsupported", "blocked"}
SOURCE_REPORT = "Documentation/StreamClusterBinProfile20260924.json"
HISTORICAL_RUN = "build-release/full-sw-nsight-profile4"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def record_file(path, role, expected_sha256=None):
    path = Path(path).resolve()
    item = {"path": str(path), "role": role, "state": "missing", "bytes": None,
            "sha256": None, "expected_sha256": expected_sha256}
    if not path.is_file():
        return item
    try:
        size = path.stat().st_size
        sha = digest(path)
        item.update(bytes=size, sha256=sha, state="present" if size else "empty")
        if expected_sha256 and sha != expected_sha256.lower():
            item["state"] = "hash-mismatch"
    except OSError as error:
        item.update(state="unreadable", error=str(error))
    return item


class Collector:
    def __init__(self, root, output):
        self.root = Path(root).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "raw").mkdir()
        (self.output / "archive").mkdir()
        self.commands = {}
        self.artifacts = {}

    def artifact(self, key, path, role, expected=None, archive=False):
        item = record_file(path, role, expected)
        self.artifacts[key] = item
        if archive and item["state"] == "present":
            destination = self.output / "archive" / (key + Path(path).suffix)
            shutil.copyfile(path, destination)
            if digest(destination) != item["sha256"]:
                raise RuntimeError(f"Evidence changed while archiving: {path}")
            item["archive"] = destination.relative_to(self.output).as_posix()
        return item

    def command(self, key, args, timeout=45):
        started = time.monotonic()
        result = {"argv": [str(arg) for arg in args], "cwd": str(self.root),
                  "exit_code": None, "timed_out": False, "error": None}
        stdout = stderr = b""
        try:
            process = subprocess.run(args, cwd=self.root, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=timeout, check=False,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            stdout, stderr = process.stdout, process.stderr
            result["exit_code"] = process.returncode
        except subprocess.TimeoutExpired as error:
            result["timed_out"] = True
            stdout, stderr = error.stdout or b"", error.stderr or b""
        except OSError as error:
            result["error"] = str(error)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        for channel, data in (("stdout", stdout), ("stderr", stderr)):
            path = self.output / "raw" / f"{key}.{channel}.txt"
            path.write_bytes(data)
            result[channel] = {"path": path.relative_to(self.output).as_posix(),
                               "sha256": digest(path), "bytes": len(data)}
        self.commands[key] = result
        return stdout.decode("utf-8-sig", errors="replace")

    def json_command(self, key, args):
        text = self.command(key, args)
        try:
            return json.loads(text) if self.commands[key]["exit_code"] == 0 else None
        except ValueError:
            self.commands[key]["parse_error"] = "Output is not valid JSON"
            return None


def capability(status, scope, reason, evidence, declaration=None):
    if status not in STATUSES:
        raise ValueError(f"Invalid capability status: {status}")
    return {"status": status, "verified_scope": scope, "reason": reason,
            "evidence": evidence, "declaration": declaration}


def verify_bundle(path):
    path = Path(path).resolve()
    bundle = read_json(path)
    errors = []
    if bundle.get("schema_version") != 1 or bundle.get("kind") != "metallic.perf.baseline":
        errors.append("Unsupported baseline schema")
    for section in ("artifacts", "commands", "capabilities"):
        if not isinstance(bundle.get(section), dict) or not bundle[section]:
            errors.append(f"Missing or empty baseline section: {section}")
    if errors:
        return errors
    for key, item in bundle.get("artifacts", {}).items():
        if item["state"] != "present":
            continue
        target = path.parent / item["archive"] if "archive" in item else Path(item["path"])
        if not target.is_file() or digest(target) != item["sha256"]:
            errors.append(f"Artifact missing or changed: {key}")
    for key, command in bundle.get("commands", {}).items():
        for channel in ("stdout", "stderr"):
            raw = command[channel]
            target = path.parent / raw["path"]
            if not target.is_file() or digest(target) != raw["sha256"]:
                errors.append(f"Command evidence missing or changed: {key}.{channel}")
    for key, cap in bundle.get("capabilities", {}).items():
        if cap["status"] not in STATUSES:
            errors.append(f"Invalid status: {key}")
        if cap["status"] == "verified" and (not cap["verified_scope"] or not cap["evidence"]):
            errors.append(f"Verified claim lacks scope/evidence: {key}")
        for reference in cap["evidence"]:
            if reference not in bundle["commands"] and reference not in bundle["artifacts"]:
                errors.append(f"Unknown evidence reference: {key}/{reference}")
            elif cap["status"] == "verified":
                if reference in bundle["commands"]:
                    command = bundle["commands"][reference]
                    if command["exit_code"] != 0 or command.get("timed_out") or command.get("parse_error"):
                        errors.append(f"Verified claim cites failed command: {key}/{reference}")
                elif bundle["artifacts"][reference]["state"] != "present":
                    errors.append(f"Verified claim cites unavailable artifact: {key}/{reference}")
    return errors


def collect(args):
    collector = Collector(args.root, args.output)
    root = collector.root
    wrapper = shutil.which("cli-anything-nsight-graphics")
    if not wrapper:
        raise RuntimeError("cli-anything-nsight-graphics is not on PATH")
    info = collector.json_command("doctor-info", [wrapper, "--json", "doctor", "info"])
    versions = collector.json_command("doctor-versions", [wrapper, "--json", "doctor", "versions"])
    info = info if isinstance(info, dict) else {}
    collector.artifact("wrapper", wrapper, "tool-binary")
    collector.command("wrapper-package", [sys.executable, "-m", "pip", "show", "cli-anything-nsight-graphics"])
    collector.command("wrapper-module-files", [sys.executable, "-c",
        "import cli_anything.nsight_graphics as p; from pathlib import Path; import hashlib,json; "
        "roots=list(p.__path__); files=sorted({f for r in roots for f in Path(r).rglob('*.py')}); "
        "print(json.dumps([{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]))"])

    for name, binary in info.get("binaries", {}).items():
        collector.artifact(name, binary, "tool-binary")
    ngfx = info.get("binaries", {}).get("ngfx")
    if ngfx:
        collector.command("ngfx-help-all", [ngfx, "--help-all"])
    gpu_text = collector.command("gpu", [shutil.which("nvidia-smi") or "nvidia-smi",
        "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
    gpus = [{"name": row[0].strip(), "uuid": row[1].strip(), "driver": row[2].strip()}
            for row in csv.reader(io.StringIO(gpu_text)) if len(row) == 3]
    head = collector.command("git-head", ["git", "rev-parse", "HEAD"]).strip()
    collector.command("git-status", ["git", "status", "--porcelain=v1"])
    collector.command("git-diff", ["git", "diff", "--binary", "HEAD", "--", "Source", "Shaders", "Tools", "tests", "cmake", "CMakeLists.txt"])
    source_files = ["Source/Runtime/Render/SlangCompiler.cpp", "Source/Runtime/Render/SlangCompiler.h",
        "Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp",
        "Source/Editor/EditorRasterComparison.cpp", "Source/Runtime/Render/Profiling/NsightGraphicsCapture.cpp",
        "Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang"]
    for index, relative in enumerate(source_files):
        collector.artifact(f"source-{index}", root / relative, "current-source-snapshot", archive=True)
    collector.artifact("collector", Path(__file__), "baseline-generator", archive=True)
    historical_report = collector.artifact("source-profile-report", root / SOURCE_REPORT,
                                         "historical-derived-analysis", archive=True)
    report = read_json(root / SOURCE_REPORT) if historical_report["state"] == "present" else {}
    expected = report.get("sha256")
    csv_path = Path(args.shader_csv) if args.shader_csv else root / "Captures/NsightGraphics/streamClusterBinMain.csv"
    source_csv = collector.artifact("shader-source-csv", csv_path, "shader-profiler-original",
                                    expected if not args.new_export else None, archive=True)
    source_csv["reported_location"] = args.source_location
    source_csv["location_basis"] = "user-provided" if args.source_location == "remote-only" else "local-file-probe"
    collector.artifact("historical-parser", root / "build/nsight-visibility-20260924/AnalyzeBinProfile.py", "historical-parser")
    if args.new_export and not args.shader_csv:
        raise ValueError("--new-export requires --shader-csv")

    search_results = []
    for index, location in enumerate([str(root)] + args.search_root):
        text = collector.command(f"search-{index}", [shutil.which("rg") or "rg", "--files", "-uuu", location,
            "-g", "*streamClusterBinMain*", "-g", "*AnalyzeBinProfile*", "-g", "!**/.git/**",
            "-g", "!**/node_modules/**", "-g", "!**/.pytest_cache/**"])
        search_results.append({"root": location, "command": f"search-{index}", "matches": text.splitlines()})

    historical = root / HISTORICAL_RUN
    for key, relative in {"old-run": "Manifest.json", "old-ready": "app/ProfileReady.json",
                          "old-log": "trace.log", "old-config": "Config.json"}.items():
        collector.artifact(key, historical / relative, "historical-run-evidence", archive=True)
    for name in ("D3DPERF_EVENTS.xls", "FRAME.xls", "GPUTRACE_FRAME.xls", "GPUTRACE_REGIMES.xls", "REPRO_INFO.xls"):
        collector.artifact("old-" + name, historical / "trace/BASE_UNLOCKED" / name, "historical-gpu-trace-table", archive=True)
    for index, trace in enumerate(sorted((historical / "trace").glob("*.ngfx-gputrace"))):
        collector.artifact(f"old-trace-{index}", trace, "historical-gpu-trace")
    for index, capture in enumerate(sorted((root / "Captures/NsightGraphics").glob("*.ngfx-capture"))):
        collector.artifact(f"capture-{index}", capture, "historical-graphics-capture")

    child_status = None
    child_probe_evidence = []
    if args.child:
        collector.artifact("cua-child", args.child, "tool-binary")
        child_status = collector.json_command("child-status", [args.child, "status"])
        child_probe_evidence.append("child-status")
    if args.child_probe_dir:
        for channel in ("stdout", "stderr"):
            key = "child-initialize-" + channel
            collector.artifact(key, Path(args.child_probe_dir) / f"cua-child-initialize.{channel}.txt",
                               "child-initialize-probe", archive=True)
            child_probe_evidence.append(key)
    options = info.get("activity_options", {}).get("GPU Trace Profiler", [])
    discovered = bool(info.get("ok") and versions and versions.get("ok"))
    cli = ["doctor-info", "doctor-versions"]
    shader_doc = "https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html"
    capabilities = {
        "cli_discovery": capability("verified" if discovered else "blocked", "installed CLI discovery only" if discovered else None,
            "Discovery does not verify injection, counters, or export.", cli),
        "graphics_capture": capability("supported-unverified", None, "No fresh capture attempted in M0.", cli),
        "gpu_trace_metrics_export": capability("supported-unverified" if "--auto-export" in options else "blocked", None,
            "Historical exports are archived; current GPU/driver capture is not verified.", cli),
        "shader_summary_export": capability("supported-unverified", None, "Documented CSV export; current UI export not exercised.", cli, shader_doc),
        "source_hotspots": capability("supported-unverified", None, "Source/function symbols in code are not proof of current Nsight correlation.", ["source-0", "source-profile-report"], shader_doc),
        "dependencies": capability("supported-unverified", None, "Instruction dependency view/export has not been exercised.", cli, shader_doc),
        "full_disassembly": capability("supported-unverified", None, "Edition-specific availability not established.", cli, shader_doc),
        "source_export_automation": capability("supported-unverified" if child_status and child_status.get("workerReady") else "blocked", None,
            "A ready child worker still requires Nsight export validation; no main-desktop fallback.", child_probe_evidence),
    }
    gaps = []
    if not gpus or collector.commands["gpu"]["exit_code"] != 0:
        gaps.append({"code": "GpuIdentityUnavailable", "next_action": "Resolve the GPU/driver discovery failure before a hardware experiment."})

    if source_csv["state"] != "present":
        gaps.append({"code": "SourceCsvUnavailable", "artifact": "shader-source-csv", "state": source_csv["state"],
                     "next_action": "Recover the original matching its recorded SHA256, or produce a fresh Nsight source/IL export with its own manifest."})
    if capabilities["source_export_automation"]["status"] == "blocked":
        gaps.append({"code": "ChildWorkerUnavailable", "next_action": "Restore cua-child session readiness; do not control the main desktop or change login/security settings automatically."})
    if not discovered:
        gaps.append({"code": "DiscoveryFailed", "next_action": "Inspect raw doctor outputs."})

    if source_csv["state"] == "present" and (args.new_export or not expected):
        gaps.append({"code": "SourceExportProvenanceUnverified", "artifact": "shader-source-csv",
                     "next_action": "Associate the replacement export with its capture, shader identity, scope and environment before accepting it as a fixture."})

    case = {"schema_version": 1, "case_id": "zorah-full-legacy-sw-20260920",
            "status": "historical-evidence-only", "capture_environment": read_json(historical / "Manifest.json") if (historical / "Manifest.json").is_file() else None,
            "workload": "GPUDriven.Raster.Early.Software", "entry_point": "streamClusterRasterLegacyMain",
            "entry_identity_basis": "Historical profiler hold selected mode 10; capture bytecode has not been matched here.",
            "requested_scope": "software-dispatch", "achieved_scope": "marker-range-hw-sw-merged",
            "marker": "RenderGraphPass: VBuffer (VisibilityBufferPass)/Hybrid raster: stable cluster bins",
            "queue": None, "dispatch_id": None, "source_hash_at_capture": None,
            "ready_snapshot_artifact": "old-ready", "timing_acceptance_eligible": False,
            "limitations": ["Not the current WorkControl production shader.", "Not a pure SW counter range.",
                            "Historical driver differs from the current baseline.", "No new performance measurement."]}
    write_json(collector.output / "case.json", case)
    collector.artifact("case", collector.output / "case.json", "minimal-historical-case", archive=True)
    bundle = {"schema_version": 1, "kind": "metallic.perf.baseline", "created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"os": platform.platform(), "python": sys.version, "repository": str(root), "git_head": head, "gpus": gpus},
              "tool_version": info.get("version"), "commands": collector.commands, "artifacts": collector.artifacts,
              "searches": search_results, "capabilities": capabilities, "gaps": gaps,
              "m0": {"status": "partial" if gaps else "evidence-ready",
                     "source_fixture": "available-not-parsed" if source_csv["state"] == "present" else "missing",
                     "runtime_capture_verified": False, "source_csv_new_export": args.new_export}}
    write_json(collector.output / "Baseline.json", bundle)
    errors = verify_bundle(collector.output / "Baseline.json")
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({"manifest": str(collector.output / "Baseline.json"), "m0": bundle["m0"],
                      "integrity": "passed", "gaps": gaps}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output")
    parser.add_argument("--verify", help="Check hashes without running tools")
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--child", help="Observed cua-child.exe path; only status is queried")
    parser.add_argument("--child-probe-dir", help="Directory containing separately acquired MCP initialization logs")
    parser.add_argument("--shader-csv")
    parser.add_argument("--source-location", choices=("local-or-unknown", "remote-only"), default="local-or-unknown")
    parser.add_argument("--new-export", action="store_true", help="Do not expect the historical CSV hash")
    args = parser.parse_args()
    if args.verify:
        errors = verify_bundle(args.verify)
        print(json.dumps({"integrity": "failed" if errors else "passed", "errors": errors}, indent=2))
        return 1 if errors else 0
    if not args.output:
        parser.error("--output is required for collection")
    if args.new_export and not args.shader_csv:
        parser.error("--new-export requires --shader-csv")
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
