"""Compile fixed General Guides with narrowly loaded material fields.

This source-only experiment retains all material features. It is independent
of Backend Type/No-Fiber selection and establishes no rendering qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
import ProfileComputeShaderFrontend as profile
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver
import ValidateSlangLinkTimeSpecialization as legacy
import ValidatePathTraceBackend as backend
import PathTraceMaterialReadOverlay as overlay

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("original-general", "material-narrow-general")
CANDIDATE_ARMS = ("material-narrow-general", "material-member-general")
EXPERIMENT = "material-narrow-reads-v1"
PASSES = ["--eliminate-dead-branches", "--simplify-instructions"]
SCOPE = "Six independent source-to-SPIR-V CLI processes; no native creation or GPU execution."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def selected_arms(candidate_arm=CANDIDATE_ARMS[0]):
    require(candidate_arm in CANDIDATE_ARMS, "Unknown material-read candidate")
    return (ARMS[0], candidate_arm)


def schedule(candidate_arm=CANDIDATE_ARMS[0]):
    arms = selected_arms(candidate_arm)
    return [(block, arm) for block in range(3) for arm in (arms if block != 1 else tuple(reversed(arms)))]


def identity(path):
    raw = Path(path).read_bytes()
    legacy.instruction_counts(raw)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), fnv1a64=profile.fnv(raw))


def counts(raw):
    result = legacy.instruction_counts(raw)
    words = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    storage, cursor = {}, 5
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 65535
        if opcode == 59:
            require(size >= 4, "Truncated OpVariable")
            name = str(words[cursor + 3])
            storage[name] = storage.get(name, 0) + 1
        cursor += size
    opcodes = result["opcode_counts"]
    result.update(variables=sum(storage.values()), variable_storage_classes=storage,
        branches=opcodes.get("249", 0), conditional_branches=opcodes.get("250", 0),
        switches=opcodes.get("251", 0), selection_merges=opcodes.get("247", 0))
    return result


def sdk_phases(directory, stage):
    rows = []
    pattern = re.compile(r"^\[\*\]\s*(\S+)\s+(\d+)\s+([\d.]+)ms\s*$")
    for stream in ("stdout", "stderr"):
        for line in (directory / stage[stream]).read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                milliseconds = float(match[3])
                require(finite(milliseconds) and int(match[2]) > 0, "Invalid SDK phase telemetry")
                rows.append(dict(name=match[1], invocations=int(match[2]), milliseconds=milliseconds, stream=stream))
    return dict(rows=rows, scope="SDK counters may overlap. Preserve the raw reports; never sum these rows into CLI wall time.")


def invoke(command, directory, name, timeout):
    require(finite(timeout, positive=True), "Invalid process timeout")
    stage = backend.invoke(command, directory, name, timeout)
    stage.update(execution_id=uuid.uuid4().hex, ended_utc=datetime.now(timezone.utc).isoformat())
    experiment.save(directory / (name + ".process.json"), stage)
    return stage


def validate_process(stage, command, directory, name):
    require(stage == experiment.read(directory / (name + ".process.json")) and stage["command"] == command and
        stage["stdout"] == name + ".stdout.log" and stage["stderr"] == name + ".stderr.log", "Changed actual process command/sidecar")
    require(stage["status"] == "passed" and type(stage["exit_code"]) is int and stage["exit_code"] == 0 and
        stage["timed_out"] is False and stage["native_crash"] is False and type(stage["pid"]) is int and stage["pid"] > 0 and
        finite(stage["wall_seconds"], positive=True), "Unsuccessful or invalid compiler/tool process")
    execution = stage["execution_id"]
    require(isinstance(execution, str) and len(execution) == 32 and all(value in "0123456789abcdef" for value in execution), "Invalid process identity")
    start, end = (datetime.fromisoformat(stage[name]) for name in ("started_utc", "ended_utc"))
    require(start.tzinfo is not None and end.tzinfo is not None and end >= start, "Invalid process interval")
    return start.timestamp(), end.timestamp()


def command_for(plan, directory):
    command = list(plan["direct_command"])
    command[command.index("-o") + 1] = str(directory / "Guides.spv")
    return command


def validate_command(validator, source):
    return [str(validator), "--target-env", "vulkan1.4", "--scalar-block-layout", str(source)]


def frozen_helpers(output):
    destination = output / "Verifier/Tools/Perf"
    destination.mkdir(parents=True)
    rows = []
    for source in sorted(Path(__file__).parent.glob("*.py")):
        target = destination / source.name
        shutil.copyfile(source, target)
        rows.append(dict(source=str(source.resolve()), file=target.relative_to(output).as_posix(), sha256=profile.sha(target)))
    experiment.save(output / "FrozenHelpers.json", rows)
    return [Path(row["source"]) for row in rows]


def validate_helpers(output):
    rows = experiment.read(output / "FrozenHelpers.json")
    require(any(Path(row["file"]).name == Path(__file__).name for row in rows), "Actual verifier was not archived")
    for row in rows:
        source = (output / row["file"]).resolve(strict=True)
        require(source.is_relative_to((output / "Verifier").resolve()) and profile.sha(source) == row["sha256"], "Frozen verifier changed")


def reference_sample(report, request_index=146):
    require(type(request_index) is int and request_index == 146, "This experiment fixes API reference request 146")
    sample = backend.reference_sample(report, request_index)
    require(report.get("build_configuration") == "Release" and report["environment"].get("METALLIC_WORKING_COLOR_SPACE") in (None, "", "acescg"),
        "Fixed reference requires Release and ACEScg")
    return sample


def verify_preparation(path, *, current_inputs=False):
    path = Path(path).resolve(strict=True)
    legacy.verify_manifest(path)
    preparation = experiment.read(path / "Preparation.json")
    arms = selected_arms(preparation.get("candidate_arm", CANDIDATE_ARMS[0]))
    require(preparation["status"] == "prepared" and preparation["experiment"] == EXPERIMENT and preparation["arms"] == list(arms), "Wrong preparation/arm identity")
    report = experiment.read(path / "ReferenceReport.json")
    sample = reference_sample(report)
    require(identity(path / "Reference.spv") == preparation["reference"] and sample["spirv_bytes"] == preparation["reference"]["bytes"] and
        sample["spirv_fnv1a64"] == preparation["reference"]["fnv1a64"], "Original API reference identity changed")
    require([plan["label"] for plan in preparation["plans"]] == list(arms), "Missing or reordered source arms")
    repository = Path(preparation["repository_root"])
    require(Path(sample["request"]["search_path"]).resolve() == repository / "Shaders", "Wrong repository reference root")
    baseline = path / "FrozenSources"
    source_files = preparation["source_files"]
    require({item.relative_to(baseline).as_posix(): profile.sha(item) for item in baseline.rglob("*") if item.is_file()} == source_files,
        "Frozen original source inventory changed")
    old_profile, old_legacy = profile.ROOT, legacy.ROOT
    profile.ROOT = legacy.ROOT = repository
    try:
        base = profile.diagnostic_command(preparation["compiler"], report, sample, Path(preparation["origin"]) / "unused.spv")
        roots = [repository / "Shaders", *[Path(root) for root in sample["request"]["additional_search_paths"]]]
        for plan, arm in zip(preparation["plans"], arms):
            generated = overlay.plan_overlay(baseline / "Shaders", arm)
            metadata = {key: value for key, value in generated.items() if key != "files"}
            require(generated["label"] == arm and metadata == plan["overlay_metadata"], "Declared arm differs from the frozen overlay generator")
            expected = dict(source_files)
            for relative, text in generated["files"].items():
                target = Path("Shaders") / relative
                require(not target.is_absolute() and ".." not in target.parts, "Overlay escaped owned shader sources")
                expected[target.as_posix()] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            directory = path / "overlays" / arm
            require({item.relative_to(directory).as_posix(): profile.sha(item) for item in directory.rglob("*") if item.is_file()} == expected,
                "Arm source files differ from the exact overlay plan")
            require(plan["direct_command"] == legacy.rewrite_command_inputs(base, Path(preparation["origin"]) / "overlays" / arm, roots),
                "Source command differs from the API settings")
            audit = plan["ownership_audit"]
            expected_audit = [audit["command"][0], "-B", str(repository / "Tools/CheckShaderModules.py"),
                "--root", str(Path(preparation["origin"]) / "overlays" / arm)]
            validate_process(audit, expected_audit, path, arm + "-ownership")
    finally:
        profile.ROOT, legacy.ROOT = old_profile, old_legacy
    if current_inputs:
        experiment.check_snapshots(preparation["live_inputs"])
    return preparation


def prepare(args):
    candidate_arm = getattr(args, "candidate_arm", CANDIDATE_ARMS[0])
    arms = selected_arms(candidate_arm)
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(status="running", command="prepare", experiment=EXPERIMENT)
    try:
        report = experiment.read(args.report)
        sample = reference_sample(report, args.request_index)
        require(Path(sample["request"]["search_path"]).resolve() == ROOT / "Shaders", "Reference must use the current shader root")
        cache, raw = profile.bench_bytes(report, sample)
        compiler = (args.slang_root / "bin/slangc.exe").resolve(strict=True)
        sdk = legacy.compiler_manifest(compiler)
        tag = (compiler.parent.parent / "include/slang-tag-version.h").read_text(encoding="utf-8")
        match = re.search(r'#define\s+SLANG_TAG_VERSION\s+"([^"]+)"', tag)
        require(match and match[1] == report["slang_build"], "SDK differs from the fresh API compiler")
        roots = [ROOT / "Shaders", *[Path(root).resolve(strict=True) for root in sample["request"]["additional_search_paths"]]]
        require(all(root.is_relative_to(ROOT) and not output.is_relative_to(root) for root in roots), "Unfrozen source root or nested output")
        paths = {source.resolve() for root in roots for source in root.rglob("*") if source.is_file()}
        paths.update(Path(source).resolve(strict=True) for source in sample["dependencies"])
        paths.update(ROOT / name for name in ("Source/Runtime/Render/Core/NamedResourceParameters.h",
            "Source/Runtime/Render/Core/ColorSpaceMatrices.h", "Source/Runtime/Render/Environment/AtmosphereKernelParameters.h",
            "Source/Runtime/Render/Material/MaterialValueProgram.cpp", "Source/Runtime/Render/Material/SceneMaterialData.h"))
        require(all(source.is_relative_to(ROOT) for source in paths), "Dependency escaped the frozen repository")
        source_files = {source.relative_to(ROOT).as_posix(): profile.sha(source) for source in sorted(paths)}
        tools = [*sorted(Path(__file__).parent.glob("*.py")), ROOT / "Tools/CheckShaderModules.py"]
        live = experiment.snapshots([args.report, cache, *paths, *tools, *[Path(row["path"]) for row in sdk]])
        baseline = output / "FrozenSources"
        for relative in source_files:
            target = baseline / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        experiment.save(output / "ReferenceReport.json", report)
        (output / "Reference.spv").write_bytes(raw)
        frozen_helpers(output)
        base = profile.diagnostic_command(compiler, report, sample, output / "unused.spv")
        plans = []
        for arm in arms:
            destination = output / "overlays" / arm
            shutil.copytree(baseline, destination)
            modifications = overlay.plan_overlay(baseline / "Shaders", arm)
            for relative, text in modifications["files"].items():
                target = (destination / "Shaders" / relative).resolve()
                require(target.is_relative_to((destination / "Shaders").resolve()), "Overlay path escaped shader ownership")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            audit = invoke([sys.executable, "-B", str(ROOT / "Tools/CheckShaderModules.py"), "--root", str(destination)], output, arm + "-ownership", 30)
            require(audit["status"] == "passed", "Shader ownership audit failed")
            plans.append(dict(label=arm, overlay_metadata={key: value for key, value in modifications.items() if key != "files"},
                direct_command=legacy.rewrite_command_inputs(base, destination, roots), ownership_audit=audit))
        experiment.check_snapshots(live)
        preparation = dict(schema_version=1, status="prepared", command="prepare", experiment=EXPERIMENT, arms=list(arms), candidate_arm=candidate_arm,
            origin=str(output), repository_root=str(ROOT), source_files=source_files, plans=plans, live_inputs=live,
            compiler=str(compiler), sdk_files=sdk, slang_build=match[1], reference=identity(output / "Reference.spv"), scope=SCOPE)
        experiment.save(output / "Preparation.json", preparation)
        summary.update(status="passed")
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "summary.json", summary)
        profile.seal(output)
    verify_preparation(output, current_inputs=True)
    return output


def summarize(records, candidate_arm=CANDIDATE_ARMS[0]):
    arms = selected_arms(candidate_arm)
    require([(row["block"], row["arm"]) for row in records] == schedule(candidate_arm), "Missing, duplicate or reordered compile sample")
    times = {arm: driver.timings([row["compile"]["wall_seconds"] for row in records if row["arm"] == arm]) for arm in arms}
    require(all(finite(row["compile"]["wall_seconds"], positive=True) for row in records), "Invalid compile duration")
    ratios = []
    for block in range(3):
        pair = {row["arm"]: row for row in records if row["block"] == block}
        ratios.append(pair[arms[1]]["compile"]["wall_seconds"] / pair[arms[0]]["compile"]["wall_seconds"])
    ratio = driver.timings(ratios)
    return dict(arms=times, paired_ratios=ratios, ratio_statistics=ratio, percent_change=100 * (ratio["median"] - 1),
        denominator_arm=arms[0], candidate_arm=arms[1], statistical_unit="One independent compiler process per arm/pair; SDK phases overlap.")


def run(args):
    origin = args.preparation.resolve(strict=True)
    preparation = verify_preparation(origin, current_inputs=True)
    candidate_arm = preparation.get("candidate_arm", CANDIDATE_ARMS[0])
    arms = selected_arms(candidate_arm)
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(schema_version=1, status="running", command="run", experiment=EXPERIMENT, arms=list(arms), candidate_arm=candidate_arm,
        runs=[], scope=SCOPE, origin=str(output))
    checks = []
    try:
        shutil.copytree(origin, output / "Preparation")
        helpers = frozen_helpers(output)
        frozen = dict(preparation["live_inputs"])
        frozen.update(experiment.snapshots([*helpers, *[item for item in origin.rglob("*") if item.is_file()]]))
        validator = args.spirv_val.resolve(strict=True)
        frozen.update(experiment.snapshots([validator]))
        experiment.save(output / "MeasuredInputs.json", frozen)
        summary.update(preparation_origin=str(origin), validator=str(validator))
        known, plans = {}, {plan["label"]: plan for plan in preparation["plans"]}
        with profile.experiment_lock():
            for block, arm in schedule(candidate_arm):
                label = f"block-{block:02d}-{arm}"
                directory = output / label
                directory.mkdir()
                experiment.check_snapshots(frozen)
                checks.append(dict(label="before-" + label, inputs=experiment.snapshots(frozen)))
                stage = invoke(command_for(plans[arm], directory), directory, "compile", args.timeout)
                require(stage["status"] == "passed", "Compile failed; process diagnostics retained")
                spirv = directory / "Guides.spv"
                artifact = identity(spirv)
                require(arm != ARMS[0] or spirv.read_bytes() == (origin / "Reference.spv").read_bytes(), "Original source differs from fresh API reference")
                require(arm not in known or artifact == known[arm], "Arm bytecode changed between independent repetitions")
                known[arm] = artifact
                validation = invoke(validate_command(validator, spirv), directory, "validation", args.timeout)
                require(validation["status"] == "passed", "Vulkan 1.4 scalar SPIR-V validation failed")
                experiment.check_snapshots(frozen)
                checks.append(dict(label="after-" + label, inputs=experiment.snapshots(frozen)))
                summary["runs"].append(dict(block=block, arm=arm, directory=label, compile=stage, validation=validation,
                    identity=artifact, counts=counts(spirv.read_bytes()), sdk_phases=sdk_phases(directory, stage)))
                experiment.save(output / "summary.json", summary)
        summary.update(status="passed", timing_summary=summarize(summary["runs"], candidate_arm))
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "SnapshotChecks.json", checks)
        experiment.save(output / "summary.json", summary)
        profile.seal(output)
    return verify_records(output)


def verify_records(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    legacy.verify_manifest(output)
    validate_helpers(output)
    summary = experiment.read(output / "summary.json")
    candidate_arm = summary.get("candidate_arm", CANDIDATE_ARMS[0])
    arms = selected_arms(candidate_arm)
    require(summary["status"] == "passed" and summary["experiment"] == EXPERIMENT and summary["arms"] == list(arms), "Failed or different experiment")
    if summary["command"] == "analyze":
        source = output / "SourceRun"
        original = verify_records(source)
        require(original["arms"] == list(arms), "Analysis declares a different source arm pair")
        require(summary["source_summary_sha256"] == profile.sha(source / "summary.json"), "Analysis run binding changed")
        frozen = experiment.read(output / "MeasuredInputs.json")
        checks = experiment.read(output / "SnapshotChecks.json")
        require(summary["cleanup_passes"] == PASSES and [row["label"] for row in checks] ==
            [prefix + arm for arm in arms for prefix in ("before-", "after-")] and
            all(row["inputs"] == frozen for row in checks), "Analysis source/tool checks are missing or changed")
        if current_inputs:
            experiment.check_snapshots(frozen)
        for arm in arms:
            item = summary["outputs"][arm]
            directory = output / arm
            raw = directory / "Raw.spv"
            cleaned = directory / "Cleaned.spv"
            require(identity(raw) == next(row["identity"] for row in original["runs"] if row["arm"] == arm), "Analysis raw is not the declared compile arm")
            for name, path in (("raw", raw), ("cleaned", cleaned)):
                require(item[name] == dict(identity=identity(path), counts=counts(path.read_bytes())), "Analysis bytecode/structure changed")
            recorded_directory = Path(summary["origin"]) / arm
            expected = [summary["optimizer"], "--target-env=vulkan1.4", "--scalar-block-layout", *PASSES,
                str(recorded_directory / "Raw.spv"), "-o", str(recorded_directory / "Cleaned.spv")]
            validate_process(item["cleanup"], expected, directory, "cleanup")
            validate_process(item["validation"], validate_command(summary["validator"], recorded_directory / "Cleaned.spv"), directory, "validation")
        return summary
    require(summary["command"] == "run" and [(row["block"], row["arm"]) for row in summary["runs"]] == schedule(candidate_arm), "Invalid compile schedule")
    preparation = verify_preparation(output / "Preparation")
    require(preparation["origin"] == summary["preparation_origin"] and preparation["arms"] == list(arms), "Run does not use its archived preparation/arms")
    frozen = experiment.read(output / "MeasuredInputs.json")
    require(all(frozen.get(name) == digest for name, digest in preparation["live_inputs"].items()), "Run input snapshot omitted prepared sources/tools")
    if current_inputs:
        experiment.check_snapshots(frozen)
    checks = experiment.read(output / "SnapshotChecks.json")
    require([row["label"] for row in checks] == [prefix + f"block-{block:02d}-{arm}" for block, arm in schedule(candidate_arm) for prefix in ("before-", "after-")] and
        all(row["inputs"] == frozen for row in checks), "Missing before/after input checks")
    plans = {plan["label"]: plan for plan in preparation["plans"]}
    seen, known, previous_end = set(), {}, None
    for row in summary["runs"]:
        directory = output / f"block-{row['block']:02d}-{row['arm']}"
        require(row["directory"] == directory.name, "Changed compile directory")
        recorded_directory = Path(summary["origin"]) / directory.name
        for name, command in (("compile", command_for(plans[row["arm"]], recorded_directory)),
                ("validation", validate_command(summary["validator"], recorded_directory / "Guides.spv"))):
            start, end = validate_process(row[name], command, directory, name)
            require(row[name]["execution_id"] not in seen and (previous_end is None or start >= previous_end), "Repeated or overlapping process")
            seen.add(row[name]["execution_id"])
            previous_end = end
        spirv = directory / "Guides.spv"
        actual = identity(spirv)
        require(actual == row["identity"] and row["counts"] == counts(spirv.read_bytes()) and row["sdk_phases"] == sdk_phases(directory, row["compile"]), "Changed bytecode/SDK report")
        require(row["arm"] != ARMS[0] or actual == preparation["reference"], "Original baseline bytecode mismatch")
        require(row["arm"] not in known or known[row["arm"]] == actual, "Arm bytes changed between samples")
        known[row["arm"]] = actual
    require(summary["timing_summary"] == summarize(summary["runs"], candidate_arm), "Paired timing summary changed")
    return summary


def analyze(args):
    source = args.input_dir.resolve(strict=True)
    verify(source)
    source_summary = experiment.read(source / "summary.json")
    candidate_arm = source_summary.get("candidate_arm", CANDIDATE_ARMS[0])
    arms = selected_arms(candidate_arm)
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(schema_version=1, status="running", command="analyze", experiment=EXPERIMENT, arms=list(arms), candidate_arm=candidate_arm,
        outputs={}, origin=str(output),
        scope="SPIR-V structure and identical two-pass cleanup; no Driver or GPU timing.")
    try:
        shutil.copytree(source, output / "SourceRun")
        frozen_helpers(output)
        original = experiment.read(source / "summary.json")
        optimizer = (args.spirv_tools / "spirv-opt.exe").resolve(strict=True)
        validator = (args.spirv_tools / "spirv-val.exe").resolve(strict=True)
        frozen = experiment.snapshots([optimizer, validator, *[path for path in source.rglob("*") if path.is_file()],
            *Path(__file__).parent.glob("*.py")])
        experiment.save(output / "MeasuredInputs.json", frozen)
        summary.update(optimizer=str(optimizer), validator=str(validator), source_summary_sha256=profile.sha(output / "SourceRun/summary.json"), cleanup_passes=PASSES)
        checks = []
        with profile.experiment_lock():
            for arm in arms:
                directory = output / arm
                directory.mkdir()
                row = next(row for row in original["runs"] if row["arm"] == arm)
                raw, cleaned = directory / "Raw.spv", directory / "Cleaned.spv"
                shutil.copyfile(source / row["directory"] / "Guides.spv", raw)
                experiment.check_snapshots(frozen)
                checks.append(dict(label="before-" + arm, inputs=experiment.snapshots(frozen)))
                command = [str(optimizer), "--target-env=vulkan1.4", "--scalar-block-layout", *PASSES, str(raw), "-o", str(cleaned)]
                cleanup = invoke(command, directory, "cleanup", args.timeout)
                require(cleanup["status"] == "passed", "Uniform cleanup failed")
                validation = invoke(validate_command(validator, cleaned), directory, "validation", args.timeout)
                require(validation["status"] == "passed", "Cleaned SPIR-V validation failed")
                experiment.check_snapshots(frozen)
                checks.append(dict(label="after-" + arm, inputs=experiment.snapshots(frozen)))
                summary["outputs"][arm] = dict(raw=dict(identity=identity(raw), counts=counts(raw.read_bytes())),
                    cleaned=dict(identity=identity(cleaned), counts=counts(cleaned.read_bytes())), cleanup=cleanup, validation=validation)
        summary.update(status="passed")
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "SnapshotChecks.json", locals().get("checks", []))
        experiment.save(output / "summary.json", summary)
        profile.seal(output)
    return verify_records(output)


def verify(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    legacy.verify_manifest(output)
    source = output / "Verifier/Tools/Perf/ValidatePathTraceMaterialReads.py"
    require(source.is_file(), "Missing actual archived verifier")
    command = [sys.executable, "-B", str(source), "_verify-frozen", str(output)]
    if current_inputs:
        command.append("--current-inputs")
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, "Archived verification failed: " + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--report", type=Path, required=True)
    preparation.add_argument("--request-index", type=int, default=146)
    preparation.add_argument("--candidate-arm", choices=CANDIDATE_ARMS, default=CANDIDATE_ARMS[0])
    preparation.add_argument("--slang-root", type=Path, default=ROOT / "External/slang")
    preparation.add_argument("--output-dir", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--preparation", type=Path, required=True)
    execution.add_argument("--output-dir", type=Path, required=True)
    execution.add_argument("--spirv-val", type=Path, required=True)
    execution.add_argument("--timeout", type=float, default=900)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--input-dir", type=Path, required=True)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--spirv-tools", type=Path, required=True)
    analysis.add_argument("--timeout", type=float, default=120)
    for name in ("verify", "check", "_verify-frozen"):
        child = sub.add_parser(name)
        child.add_argument("output", type=Path)
        child.add_argument("--current-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args))
    elif args.command == "run":
        print(json.dumps(run(args)["timing_summary"], indent=2, allow_nan=False))
    elif args.command == "analyze":
        print(json.dumps(analyze(args)["outputs"], indent=2, allow_nan=False))
    elif args.command == "_verify-frozen":
        print(json.dumps(verify_records(args.output, current_inputs=args.current_inputs), allow_nan=False))
    else:
        if (args.output / "Preparation.json").is_file():
            result = verify_preparation(args.output, current_inputs=args.current_inputs or args.command == "check")
        else:
            result = verify(args.output, current_inputs=args.current_inputs or args.command == "check")
        print(json.dumps(dict(status="verified", experiment=result["experiment"]), indent=2))


if __name__ == "__main__":
    main()
