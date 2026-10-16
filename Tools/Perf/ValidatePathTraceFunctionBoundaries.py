"""Measure a current-source full-feature Guides noinline boundary.

This adapter gives the established material-source harness a private module
instance and a different pure overlay. It never changes that harness or its
public imported module. Six source-to-SPIR-V processes use the same fresh API
reference, settings, validation and AB/BA/AB order. Raw and uniformly cleaned
SPIR-V retain independently inspected function controls, calls and CFG metrics.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
import AnalyzeSPIRVComplexity as complexity
import PathTraceFunctionBoundaryOverlay as overlay

ROOT = Path(__file__).resolve().parents[2]
ARMS = overlay.ARMS
CANDIDATE_ARMS = (ARMS[1],)
EXPERIMENT = "path-trace-shadow-boundary-v1"
SCOPE = "Six independent source-to-SPIR-V CLI processes; no native creation or GPU execution."
PASSES = ["--eliminate-dead-branches", "--simplify-instructions"]
_archived_grammar = ROOT / "SPIRV/spirv.core.grammar.json"
GRAMMAR_PATH = _archived_grammar if _archived_grammar.is_file() else complexity.find_grammar()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_arms(candidate_arm=CANDIDATE_ARMS[0]):
    require(candidate_arm in CANDIDATE_ARMS, "Unknown function-boundary candidate")
    return ARMS[0], candidate_arm


def schedule(candidate_arm=CANDIDATE_ARMS[0]):
    arms = selected_arms(candidate_arm)
    return [(block, arm) for block in range(3)
            for arm in (arms if block != 1 else tuple(reversed(arms)))]


def function_boundaries(raw):
    """Count actual binary function controls/calls; reject recursive call graphs."""
    require(GRAMMAR_PATH is not None and Path(GRAMMAR_PATH).is_file(), "A complete pinned Khronos SPIR-V grammar is required")
    grammar_raw = Path(GRAMMAR_PATH).read_bytes()
    grammar_document = json.loads(grammar_raw)
    grammar = {row["opcode"]: row for row in grammar_document["instructions"]}
    inspected = complexity.inspect_bytes(raw, grammar)
    graph = inspected["call_graph"]
    require(not graph["recursive_components"], "Recursive or cyclic SPIR-V call graph")
    entries = graph["entry_points"]
    require(entries, "Missing SPIR-V entry point")
    words = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    controls, cursor = {}, 5
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 65535
        if opcode == 54:
            require(size == 5, "Malformed OpFunction control operands")
            controls[words[cursor + 2]] = words[cursor + 3]
        cursor += size
    reachable = {entry["function_id"] for entry in entries}
    pending = list(reachable)
    calls = {row["id"]: [] for row in inspected["functions"]}
    for edge in graph["edges"]:
        calls[edge["caller_id"]].append(dict(function_id=edge["callee_id"], static_calls=edge["static_callsite_count"]))
    while pending:
        for edge in calls[pending.pop()]:
            target = edge["function_id"]
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    functions = []
    for row in inspected["functions"]:
        control = controls[row["id"]]
        functions.append(dict(id=row["id"], name=row["name"], control=control,
            dont_inline=bool(control & 2), inline=bool(control & 1),
            instructions=row["instruction_count"], blocks=row["cfg"]["block_count"],
            cfg_edges=row["cfg"]["edge_count"], call_targets=sorted(calls[row["id"]], key=lambda item: item["function_id"]),
            reachable_from_entry=row["id"] in reachable))
    roots = []
    for entry in entries:
        require(type(entry["max_static_call_depth"]) is int and entry["max_static_call_depth"] >= 1,
                "Undefined entry-reachable call depth")
        roots.append(dict(function_id=entry["function_id"], name=entry["name"], execution_model=entry["execution_model"],
            reachable_function_count=entry["reachable_function_count"], max_call_depth=entry["max_static_call_depth"] - 1))
    return dict(function_count=len(functions), function_calls=inspected["summary"]["static_function_calls"],
        dont_inline_functions=sum(row["dont_inline"] for row in functions),
        inline_functions=sum(row["inline"] for row in functions),
        max_function_instructions=max((row["instructions"] for row in functions), default=0),
        max_function_blocks=max((row["blocks"] for row in functions), default=0),
        cfg_edges=sum(row["cfg_edges"] for row in functions),
        entry_reachable_call_depth=max(root["max_call_depth"] for root in roots),
        functions=functions, entry_points=roots,
        grammar_sha256=overlay.sha256(grammar_raw), grammar_revision=grammar_document.get("revision"),
        scope="Binary static structure. Function instruction counts include OpFunction/parameters/end; call depth counts edges. CFG switch targets use the declared 8/16/32/64-bit selector width.")


def _load_engine():
    # Separate globals are essential: no mutation of an imported material harness
    # and no effect on its old arms, defaults, verifier or archived evidence.
    source = Path(__file__).with_name("ValidatePathTraceMaterialReads.py")
    spec = importlib.util.spec_from_file_location("_private_function_boundary_engine", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(Path(__file__).resolve())
    module.__doc__ = __doc__
    module.overlay = overlay
    module.ROOT = ROOT
    module.ARMS = ARMS
    module.CANDIDATE_ARMS = CANDIDATE_ARMS
    module.EXPERIMENT = EXPERIMENT
    module.SCOPE = SCOPE
    module.selected_arms = selected_arms
    module.schedule = schedule
    return module


_engine = _load_engine()
profile, legacy, driver = _engine.profile, _engine.legacy, _engine.driver
_base_experiment = _engine.experiment


class _ExperimentInputs:
    """Pin the analyzer's actual grammar in every private-engine input snapshot."""
    def __getattr__(self, name):
        return getattr(_base_experiment, name)

    def snapshots(self, paths):
        require(GRAMMAR_PATH is not None and Path(GRAMMAR_PATH).is_file(), "Missing complete Khronos grammar")
        return _base_experiment.snapshots([*paths, Path(GRAMMAR_PATH)])


experiment = _engine.experiment = _ExperimentInputs()
_base_counts = _engine.counts
_base_frozen_helpers = _engine.frozen_helpers
_base_verify_records = _engine.verify_records
_base_summarize = _engine.summarize


def counts(raw):
    result = _base_counts(raw)
    boundary = function_boundaries(raw)
    require(result["functions"] == boundary["function_count"] and result["function_calls"] == boundary["function_calls"],
            "Independent function/call parsers disagree")
    result.update(function_boundaries=boundary, dont_inline_functions=boundary["dont_inline_functions"],
        function_controls={str(row["id"]): row["control"] for row in boundary["functions"]},
        max_function_instructions=boundary["max_function_instructions"], max_function_basic_blocks=boundary["max_function_blocks"],
        entry_reachable_call_depth=boundary["entry_reachable_call_depth"])
    return result


def frozen_helpers(output):
    helpers = _base_frozen_helpers(output)
    require(GRAMMAR_PATH is not None and Path(GRAMMAR_PATH).is_file(), "Missing complete Khronos grammar")
    source = Path(GRAMMAR_PATH).resolve(strict=True)
    target = output / "Verifier/SPIRV/spirv.core.grammar.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    digest = profile.sha(source)
    require(profile.sha(target) == digest, "Grammar changed while freezing its actual analyzer input")
    rows = experiment.read(output / "FrozenHelpers.json")
    rows.append(dict(source=str(source), file=target.relative_to(output).as_posix(), sha256=digest))
    experiment.save(output / "FrozenHelpers.json", rows)
    return [*helpers, source]


def retained_targets(metrics):
    boundary = metrics["function_boundaries"]
    targets = {row["id"]: row["control"] for row in boundary["functions"]
               if row["dont_inline"] and row["reachable_from_entry"]}
    called = {edge["function_id"] for row in boundary["functions"] if row["reachable_from_entry"]
              for edge in row["call_targets"] if edge["static_calls"] > 0}
    require(targets and set(targets).issubset(called) and boundary["entry_reachable_call_depth"] >= 1,
            "Candidate lacks an entry-reachable DontInline function call boundary")
    return targets


def _qualified_summarize(records, candidate_arm=CANDIDATE_ARMS[0]):
    for row in records:
        if row["arm"] == ARMS[1]:
            retained_targets(row["counts"])
    return _base_summarize(records, candidate_arm)


def verify_records(output, *, current_inputs=False):
    summary = _base_verify_records(output, current_inputs=current_inputs)
    if summary["command"] == "run":
        for row in summary["runs"]:
            if row["arm"] == ARMS[1]:
                retained_targets(row["counts"])
    elif summary["command"] == "analyze":
        candidate = summary["outputs"][ARMS[1]]
        raw = retained_targets(candidate["raw"]["counts"])
        cleaned = retained_targets(candidate["cleaned"]["counts"])
        require(all(cleaned.get(identifier) == control for identifier, control in raw.items()),
                "Uniform cleanup deleted or inlined the retained noinline boundary")
    return summary


def verify(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    legacy.verify_manifest(output)
    source = output / "Verifier/Tools/Perf/ValidatePathTraceFunctionBoundaries.py"
    require(source.is_file(), "Missing actual archived function-boundary verifier")
    command = [sys.executable, "-B", str(source), "_verify-frozen", str(output)]
    if current_inputs:
        command.append("--current-inputs")
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, "Archived function-boundary verification failed: " + result.stderr.strip())
    return json.loads(result.stdout)


_engine.counts = counts
_engine.frozen_helpers = frozen_helpers
_engine.verify_records = verify_records
_engine.verify = verify
_engine.summarize = _qualified_summarize
prepare, run = _engine.prepare, _engine.run
verify_preparation, identity = _engine.verify_preparation, _engine.identity
invoke, validate_process = _engine.invoke, _engine.validate_process


def summarize(records, candidate_arm=CANDIDATE_ARMS[0]):
    return _base_summarize(records, candidate_arm)


def analyze(args):
    """Reuse the sealed source run; qualify retention before marking it passed."""
    source = args.input_dir.resolve(strict=True)
    verify(source)
    original = experiment.read(source / "summary.json")
    require(original["candidate_arm"] == ARMS[1] and original["arms"] == list(ARMS), "Wrong source boundary experiment")
    require(all(row["counts"]["function_boundaries"]["grammar_sha256"] == profile.sha(GRAMMAR_PATH)
                for row in original["runs"]), "Source run and cleanup analysis use different Khronos grammars")
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(schema_version=1, status="running", command="analyze", experiment=EXPERIMENT,
        arms=list(ARMS), candidate_arm=ARMS[1], outputs={}, origin=str(output),
        scope="SPIR-V function/CFG structure and identical two-pass cleanup; no Driver or GPU timing.")
    checks = []
    try:
        shutil.copytree(source, output / "SourceRun")
        frozen_helpers(output)
        optimizer = (args.spirv_tools / "spirv-opt.exe").resolve(strict=True)
        validator = (args.spirv_tools / "spirv-val.exe").resolve(strict=True)
        frozen = experiment.snapshots([optimizer, validator, *[path for path in source.rglob("*") if path.is_file()],
            *Path(__file__).parent.glob("*.py")])
        experiment.save(output / "MeasuredInputs.json", frozen)
        summary.update(optimizer=str(optimizer), validator=str(validator),
            source_summary_sha256=profile.sha(output / "SourceRun/summary.json"), cleanup_passes=PASSES)
        with profile.experiment_lock():
            for arm in ARMS:
                directory = output / arm
                directory.mkdir()
                row = next(row for row in original["runs"] if row["arm"] == arm)
                raw, cleaned = directory / "Raw.spv", directory / "Cleaned.spv"
                shutil.copyfile(source / row["directory"] / "Guides.spv", raw)
                experiment.check_snapshots(frozen)
                checks.append(dict(label="before-" + arm, inputs=experiment.snapshots(frozen)))
                command = [str(optimizer), "--target-env=vulkan1.4", "--scalar-block-layout", *PASSES,
                    str(raw), "-o", str(cleaned)]
                cleanup = _engine.invoke(command, directory, "cleanup", args.timeout)
                require(cleanup["status"] == "passed", "Uniform cleanup failed")
                validation = _engine.invoke(_engine.validate_command(validator, cleaned), directory, "validation", args.timeout)
                require(validation["status"] == "passed", "Cleaned SPIR-V validation failed")
                experiment.check_snapshots(frozen)
                checks.append(dict(label="after-" + arm, inputs=experiment.snapshots(frozen)))
                item = dict(raw=dict(identity=identity(raw), counts=counts(raw.read_bytes())),
                    cleaned=dict(identity=identity(cleaned), counts=counts(cleaned.read_bytes())),
                    cleanup=cleanup, validation=validation)
                summary["outputs"][arm] = item
                if arm == ARMS[1]:
                    raw_targets = retained_targets(item["raw"]["counts"])
                    cleaned_targets = retained_targets(item["cleaned"]["counts"])
                    require(all(cleaned_targets.get(identifier) == control for identifier, control in raw_targets.items()),
                            "Uniform cleanup deleted or inlined the retained noinline boundary")
        summary.update(status="passed")
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "SnapshotChecks.json", checks)
        experiment.save(output / "summary.json", summary)
        profile.seal(output)
    return verify_records(output)


_engine.analyze = analyze


def main():
    _engine.main()


if __name__ == "__main__":
    main()
