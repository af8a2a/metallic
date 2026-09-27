"""Declarative, reversible triangle-decision debugger acceptance experiment.

The controlled repair restores pinned source bytes. Printf is never a performance
sample; this diagnostic workflow cannot promote an M3 optimization candidate.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import ShaderTrace as t
import ExperimentRunner as e
import WorkloadCase as w

ROOT = w.ROOT
PROTOCOL = "metallic.shader-debug-plan-v1"
ANCHOR = "        // Shader debugger transaction boundary: prepared vertices -> raster decision."
FAULT = ANCHOR + "\n        if (!reorder) { b = a; }"


def plan_for(original):
    def patch(identity, data, before, after):
        return {"protocol": e.PROTOCOL, "id": identity, "hypothesis": "Collapse prepared edge before the actual triangle decision",
                "path": e.TARGET, "baseSha256": e.sha(data),
                "replacements": [{"before": before, "after": after, "count": 1}]}
    fault = patch("collapse-edge", original, ANCHOR, FAULT)
    changed = e.candidate_bytes(fault, original)
    return {"protocol": PROTOCOL, "id": "triangle-edge-collapse", "caseId": t.recipe()["id"],
            "selection": t.selection(), "observations": [t.SITE, t.DECISION_SITE],
            "fault": fault, "repair": patch("restore-edge", changed, FAULT, ANCHOR)}


def validate_plan(plan, original):
    w.require(set(plan) == {"protocol", "id", "caseId", "selection", "observations", "fault", "repair"}, "Unknown plan fields")
    w.require(plan["protocol"] == PROTOCOL and plan["id"] == "triangle-edge-collapse"
              and plan["caseId"] == t.recipe()["id"] and plan["selection"] == t.selection()
              and plan["observations"] == [t.SITE, t.DECISION_SITE], "Unregistered debugger plan")
    # P3 is a bounded fault-reversal experiment, not an arbitrary source executor.
    expected = plan_for(original)
    w.require(plan == expected, "Plan differs from registered fault/repair contract")
    changed = e.candidate_bytes(plan["fault"], original)
    w.require(e.candidate_bytes(plan["repair"], changed) == original, "Repair does not restore pinned baseline")
    return changed


def schedule():
    rows = []
    for arm in ("healthy", "fault", "repaired"):
        for index in range(1 if arm == "fault" else 3):
            rows.append({"arm": arm, "name": f"normal-{index+1}", "selection": None})
        for site in ([t.DECISION_SITE] if arm == "repaired" else [t.SITE, t.DECISION_SITE]):
            rows.append({"arm": arm, "name": "prepare" if site == t.SITE else "decision",
                         "selection": t.selection(site=site), "expected": "Matched"})
    return rows


def fields(row):
    w.require(row.get("selectedScopeComplete") and row.get("outcome") == "Matched"
              and len(row.get("fields", [])) == 1, "Missing complete selected triangle evidence")
    return row["fields"][0]


def decision_values(row):
    values = fields(row)
    stages = values["evaluatedStages"]["value"]
    names = ["recordIndex", "triangleId", "instanceFlags", "signedArea", "doubleSided", "reason", "evaluatedStages"]
    if stages & 2:
        names += ["lowerX", "lowerY", "upperX", "upperY"]
    if stages & 4:
        names += ["determinant"]
    w.require(stages in (1, 3, 7), "Invalid decision stage coverage")
    return {name: values[name] for name in names}


def diagnose(rows):
    by = {(r["arm"], r["name"]): r for r in rows}
    hp, fp = (fields(by[(arm, "prepare")]) for arm in ("healthy", "fault"))
    hd, fd = (decision_values(by[(arm, "decision")]) for arm in ("healthy", "fault"))
    w.require(hp == fp, "Divergence already exists at preparation; declared diagnosis not established")
    for name in ("recordIndex", "triangleId", "instanceFlags"):
        w.require(hp[name] == hd[name] == fd[name], "Different selected triangles")
    w.require(hd["signedArea"]["value"] != 0 and fd["signedArea"]["value"] == 0
              and fd["reason"]["value"] == 1 and fd["evaluatedStages"]["value"] == 1,
              "Fault signature not observed in the actual decision")
    # History-dependent late lists may change after the declared early fault.
    # Require the camera, graph, dimensions, and selected prepared input instead.
    for key in ("camera", "renderExtent", "graph", "historyInvalidationPolicy"):
        w.require(all(r["identity"][key] == rows[0]["identity"][key] for r in rows), "Fixed workload drift")
    return {"firstObservedDivergence": t.DECISION_SITE, "previousObservationEqual": t.SITE,
            "healthy": hd, "fault": fd, "scope": "selected early triangle, ordered registered observations; not a GPU total order",
            "pixelCausality": "global fault output regression is separate from this selected triangle witness"}


def outputs(row):
    return {key: row["identity"]["snapshot"][key] for key in w.OUTPUTS}


def assess(rows, case):
    expected = schedule()
    w.require([{k: r[k] for k in ("arm", "name", "selection")} for r in rows]
              == [{k: r[k] for k in ("arm", "name", "selection")} for r in expected], "Incomplete/debug schedule changed")
    w.require(e.independent_processes(rows), "Overlapping or missing process evidence")
    watches = [r for r in rows if r["selection"] is not None]
    w.require(len({r["session"] for r in watches}) == len(watches), "Reused watch session")
    diagnosis = diagnose(rows)
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in ("healthy", "fault", "repaired")}
    for arm in arms.values():
        w.require(all(r["identity"] == arm[0]["identity"] for r in arm), "Within-arm workload/output drift")
    baseline = arms["healthy"][0]
    w.require(outputs(arms["fault"][0]) != outputs(baseline), "Fault did not reproduce an uninstrumented output regression")
    w.require(all(r["identity"] == baseline["identity"] for r in arms["repaired"]), "Uninstrumented repair did not restore outputs and binding")
    w.require(fields(arms["healthy"][-1]) == fields(arms["repaired"][-1]), "Repaired decision not restored; missing logs are not a repair")
    normal = {arm: [r["normalResult"] for r in arms[arm] if r["selection"] is None] for arm in arms}
    w.require(all(r["valid"] and r["normalTiming"] for group in normal.values() for r in group), "Instrumented performance sample")
    w.require(all(r["identity"] == normal["healthy"][0]["identity"] for r in normal["repaired"]), "Normal repaired residency/input drift")
    qualification = {arm: w.compare_runs(normal[arm], case) for arm in ("healthy", "repaired")}
    gains = {metric: e.interval([1-b["timings"][metric]["median"]/a["timings"][metric]["median"]
             for a,b in zip(normal["healthy"],normal["repaired"])]) for metric in ("softwareTotalMs", "graphGpuMs")}
    stable = all(q["status"] == "stable" for q in qualification.values())
    status = "pass" if stable and all(g["lower95"] >= -.02 for g in gains.values()) else "inconclusive"
    if stable and any(g["upper95"] < -.02 for g in gains.values()):
        status = "reject"
    return {"status": "verified", "p3Acceptance": True, "independentProcesses": len(rows), "diagnosis": diagnosis,
            "correctness": "fault reproduced without Printf; repaired depth/visibility, production binding and decision exactly restored",
            "performanceGate": {"status": status, "maximumRegression": .02, "gains": gains, "qualification": qualification,
                "scope": "three independent normal runs per arm; sequential screening, no ABBA or GPU competition qualification"},
            "performanceEligible": False, "candidateAccepted": False,
            "optimizationGate": "blocked: debugger screening cannot replace M3 ABBA, competition and confirmation gates"}


def inventory():
    return {**t.source_inventory(), Path(__file__).relative_to(ROOT).as_posix(): w.digest(Path(__file__))}


def archive(output, sources):
    for relative, checksum in sources.items():
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        w.require(w.digest(target) == checksum, "Source changed during archival")
    w.save(output / "Manifest.json", {"projectRoot": str(ROOT), "sources": sources})


def inspect(output, case, row, cli):
    directory = output / row["arm"] / row["name"]
    result = {**t.inspect(directory, case, row, cli), "arm": row["arm"]}
    if row["selection"] is None:
        result["normalResult"] = w.analyze_run(directory / "app", case)
    return result


def collect(output, case, row, args):
    directory = output / row["arm"] / row["name"]
    directory.mkdir(); (directory / "app").mkdir(); (directory / "empty-layers").mkdir()
    config = w.engine_config(case)
    if row["selection"] is not None:
        config["shaderTrace"] = row["selection"]
    w.save(directory / "Config.json", config)
    env = {k: v for k,v in os.environ.items() if not k.startswith(("METALLIC_", "VK_"))}
    env.update(METALLIC_FULL_ROAM_CONFIG=str(directory / "Config.json"), METALLIC_FULL_ROAM_OUTPUT=str(directory / "app"),
               METALLIC_FULL_ROAM_HIDDEN="1", METALLIC_NSIGHT_GRAPHICS_CAPTURE="0", METALLIC_SHADER_CAPTURE_SYMBOLS="0",
               METALLIC_VK_PIPELINE_STATISTICS="0", METALLIC_SLANG_DESCRIPTOR_MODE="mapped",
               VK_LAYER_PATH=str(args.layer_path.resolve()), VK_IMPLICIT_LAYER_PATH=str(directory / "empty-layers"),
               VK_LAYER_SETTINGS_PATH=str(directory / "empty-layers"))
    if row["selection"] is not None:
        env["METALLIC_SHADER_TRACE"] = "1"
    w.save(directory / "Invocation.json", {"environment": {k:v for k,v in env.items() if k.startswith(("METALLIC_", "VK_"))}})
    t.d.run_process([str(args.exe.resolve()), "--sample", case["sampleId"]], env, directory, args.timeout)
    return inspect(output, case, row, args.cli.resolve())


def execute(args):
    w.require(args.exe.is_file() and args.cli.is_file() and args.layer_path.is_dir() and 30 <= args.timeout <= 600, "Invalid runtime/budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": PROTOCOL, "status": "running", "projectRoot": str(ROOT), "runs": schedule(), "rows": []}
    lock = ROOT / "build/shader-experiment.lock"
    lock.parent.mkdir(exist_ok=True)
    handle = transaction = None
    try:
        handle = lock.open("x", encoding="utf-8")
        handle.write(json.dumps({"output": str(output), "pid": os.getpid()})); handle.flush()
        original = (ROOT / e.TARGET).read_bytes()
        plan = w.load(args.plan)
        changed = validate_plan(plan, original)
        assets, case = t.assets(args.assets), t.recipe()
        for name, value in (("Plan.json", plan), ("Case.json", case), ("Assets.json", assets)):
            w.save(output / name, value)
        (output / "Baseline.slang").write_bytes(original)
        (output / "Candidate.slang").write_bytes(changed)
        transaction = e.ShaderTransaction(output, original, changed)
        sources = inventory()
        manifest["sources"] = sources
        binaries = [args.exe.resolve(), args.cli.resolve(), *args.exe.resolve().parent.glob("*.dll"),
                    args.layer_path.resolve() / "VkLayer_khronos_validation.dll"]
        manifest["runtime"] = {str(p): w.digest(p) for p in binaries}
        manifest["gpu"] = e.gpu_identity()
        manifest["cachePolicy"] = "production caches reused; diagnostic bypasses disk cache; fixed priming per checkpoint"
        manifest["assetPolicy"] = "SHA256 manifest plus before/after metadata checks"
        current_arm = None
        for row in schedule():
            if row["arm"] != current_arm:
                if row["arm"] == "repaired":
                    w.save(output / "Diagnosis.json", diagnose(manifest["rows"]))
                transaction.install(changed if row["arm"] == "fault" else original)
                expected = {**sources, e.TARGET: e.sha(changed if row["arm"] == "fault" else original)}
                w.require(inventory() == expected, "Source changed outside declared patch")
                archive(output / row["arm"], expected)
                current_arm = row["arm"]
            w.save(output / "Manifest.json", manifest)
            w.require(inventory() == expected, "Source drift before process")
            w.asset_metadata_matches(assets)
            result = collect(output, case, row, args)
            manifest["rows"].append(result)
            w.save(output / row["arm"] / row["name"] / "Validation.json", result)
            w.require(inventory() == expected, "Source drift during process")
            w.asset_metadata_matches(assets)
            w.require(all(w.digest(Path(p)) == h for p,h in manifest["runtime"].items()), "Runtime drift")
            print(f"{row['arm']}/{row['name']}: {result.get('outcome', 'normal')} verified", flush=True)
        result = assess(manifest["rows"], case)
        w.save(output / "Result.json", result)
        manifest["status"] = "verified"
        return result
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        try:
            if transaction is not None:
                transaction.restore()
        except Exception as error:
            manifest.update(status="recovery-required", restorationError=str(error))
            raise
        finally:
            manifest["artifacts"] = e.artifact_hashes(output)
            w.save(output / "Manifest.json", manifest)
            if handle is not None:
                handle.close()
                if manifest["status"] != "recovery-required":
                    lock.unlink()


def verify(directory, cli):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL and manifest["status"] == "verified", "Unfinished P3 evidence")
    w.require(e.artifact_hashes(directory) == manifest["artifacts"], "Artifact inventory/hash changed")
    baseline = (directory / "Baseline.slang").read_bytes()
    fault = validate_plan(w.load(directory / "Plan.json"), baseline)
    w.require((directory / "Candidate.slang").read_bytes() == fault, "Archived fault differs from plan")
    journal = w.load(directory / "Transaction.json")
    w.require(journal == {"path":e.TARGET,"baselineSha256":e.sha(baseline),"candidateSha256":e.sha(fault),"state":"restored"}, "Source restoration evidence missing")
    case = w.validate_case(w.load(directory / "Case.json"))
    w.require(case == t.recipe() and manifest["runs"] == schedule(), "Workload/schedule drift")
    for arm in ("healthy", "fault", "repaired"):
        expected = {**manifest["sources"], e.TARGET:e.sha(fault if arm == "fault" else baseline)}
        arm_manifest = w.load(directory / arm / "Manifest.json")
        w.require(arm_manifest["sources"] == expected, "Undeclared arm source changes")
        w.require(all(w.digest(w.file_in(directory / arm, "source/"+p)) == h for p,h in expected.items()), "Source archive damaged")
    rows = [inspect(directory, case, row, cli) for row in schedule()]
    return assess(rows, case)



def recover(directory):
    manifest = w.load(directory / "Manifest.json")
    w.require(manifest["protocol"] == PROTOCOL, "Wrong recovery protocol")
    # A killed controller may leave the GPU child alive before Process.json is
    # sealed. Refuse restoration while any Metallic source consumer is running.
    # Do not terminate another process or infer absence from a missing journal.
    import psutil
    consumers = {"metallic.exe", "metallicgpudrivensample.exe", "metallicshadercompiler.exe"}
    for process in psutil.process_iter():
        try:
            w.require(process.name().lower() not in consumers,
                      f"Metallic source consumer still running (pid {process.pid}); recovery refused")
        except psutil.NoSuchProcess:
            continue
    return e.recover(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    for name in ("plan", "assets", "output", "layer-path"):
        run.add_argument("--"+name, type=Path, required=True)
    run.add_argument("--exe", type=Path, default=ROOT / "build-release/Source/MetallicGPUDrivenSample.exe")
    run.add_argument("--cli", type=Path, default=ROOT / "build-release/Source/metallicctl.exe")
    run.add_argument("--timeout", type=int, default=240)
    check = sub.add_parser("verify"); check.add_argument("directory",type=Path)
    check.add_argument("--cli", type=Path, default=ROOT / "build-release/Source/metallicctl.exe")
    recovery = sub.add_parser("recover"); recovery.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            w.require(not args.output.exists(), "Plan already exists")
            result = plan_for((ROOT / e.TARGET).read_bytes()); w.save(args.output, result)
        elif args.command == "run": result = execute(args)
        elif args.command == "verify": result = verify(args.directory.resolve(), args.cli.resolve())
        else: result = recover(args.directory.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status":"failed","error":str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
