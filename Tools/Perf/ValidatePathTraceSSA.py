"""Qualify one external SSA rewrite of the fixed full-feature General artifact.

The control is the already qualified O1 source -> raw -> two-pass cleaned
Original General, never the shadow/noinline candidate. Three independent
optimizer processes add exactly --ssa-rewrite. This measures artifact-pass
wall time and static structure, not frontend, Driver, native ISA or GPU cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
import AnalyzeSPIRVComplexity as complexity
import ProfileComputeShaderFrontend as profile
import RunComputePipelineExperiments as experiment
import ValidateSlangLinkTimeSpecialization as legacy
import ValidatePathTraceMaterialReads as material
import ValidatePathTraceFunctionBoundaries as boundary

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("original-cleaned-general", "ssa-general")
PASSES = ["--ssa-rewrite"]
ANCESTRY_PASSES = ["--eliminate-dead-branches", "--simplify-instructions"]
SOURCE_RELATIVE = Path("build-release/path-function-boundaries-20261013/analysis")
REFERENCE_RELATIVE = Path("build-release/function-boundary-reference2-20261013/report.json")
EXPERIMENT = "path-trace-external-ssa-v1"
SCOPE = "External SPIR-V pass wall time and static structure only; no frontend, Driver or GPU timing."
_archived_grammar = ROOT / "SPIRV/spirv.core.grammar.json"
GRAMMAR_PATH = _archived_grammar if _archived_grammar.is_file() else complexity.find_grammar()
validate_process = material.validate_process
validate_command = material.validate_command


def require(condition, message):
    if not condition:
        raise ValueError(message)


def invoke(command, directory, name, timeout):
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    try:
        return material.invoke(command, directory, name, timeout)
    except (OSError, subprocess.SubprocessError) as error:
        # A launch failure has no fabricated PID/exit code. Preserve its logs
        # and process identity before the outer finally seals the failed run.
        for stream in ("stdout", "stderr"):
            path = directory / (name + "." + stream + ".log")
            if not path.exists():
                path.write_bytes(b"")
        stage = dict(command=command, pid=None, started_utc=started_utc,
            ended_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
            execution_id=uuid.uuid4().hex, exit_code=None, timed_out=False, native_crash=False,
            stdout=name + ".stdout.log", stderr=name + ".stderr.log", status="failed", launch_error=str(error))
        experiment.save(directory / (name + ".process.json"), stage)
        return stage


def schedule():
    # C is copied and validated; it has no optimizer wall-time sample.
    return [(repetition, arm) for repetition in range(3) for arm in ARMS]


def identity(path):
    raw = Path(path).read_bytes()
    legacy.instruction_counts(raw)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), fnv1a64=profile.fnv(raw))


def _grammar(path=None):
    path = Path(path or GRAMMAR_PATH).resolve(strict=True)
    document = experiment.read(path)
    require(document.get("instructions") and document.get("operand_kinds"), "Complete Khronos grammar required")
    return document, {row["opcode"]: row for row in document["instructions"]}


def _instructions(raw, grammar):
    require(len(raw) >= 20 and len(raw) % 4 == 0, "Invalid raw SPIR-V length")
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    require(words[0] == 0x07230203 and words[3] > 0 and words[4] == 0, "Invalid raw SPIR-V header")
    rows, cursor = [], 5
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 65535
        require(size > 0 and cursor + size <= len(words), "Invalid SPIR-V instruction boundary")
        require(opcode in grammar, f"Opcode {opcode} absent from pinned grammar")
        rows.append((opcode, words[cursor + 1:cursor + size]))
        cursor += size
    return words, rows


def _string(words, start=0):
    raw = struct.pack(f"<{len(words) - start}I", *words[start:])
    require(b"\0" in raw, "Unterminated SPIR-V literal string")
    size = raw.index(0) + 1
    return raw[:size - 1].decode("utf-8", errors="strict"), start + (size + 3) // 4


def counts(raw, grammar_path=None):
    """Pointer result types determine storage, including access chains/Phi/copies."""
    document, grammar = _grammar(grammar_path)
    _, rows = _instructions(raw, grammar)
    result = legacy.instruction_counts(raw)
    inspected = complexity.inspect_bytes(raw, grammar)
    pointer_classes, value_types, result_opcodes = {}, {}, {}
    float_widths, integer_types, arithmetic = Counter(), Counter(), Counter()
    for opcode, operands in rows:
        kinds = [item["kind"] for item in grammar[opcode].get("operands", [])]
        if kinds[:2] == ["IdResultType", "IdResult"]:
            require(len(operands) >= 2, "Missing typed result")
            value_types[operands[1]] = operands[0]
            result_opcodes[operands[1]] = opcode
        if opcode == 32:
            require(len(operands) == 3, "Malformed OpTypePointer")
            pointer_classes[operands[0]] = operands[1]
        elif opcode == 22:
            require(len(operands) >= 2, "Malformed OpTypeFloat")
            float_widths[str(operands[1])] += 1
        elif opcode == 21:
            require(len(operands) == 3, "Malformed OpTypeInt")
            integer_types[f"{operands[1]}-{'signed' if operands[2] else 'unsigned'}"] += 1
        if grammar[opcode].get("class") == "Arithmetic":
            arithmetic[grammar[opcode]["opname"]] += 1
    variables, loads, stores, precision, arithmetic_precision = Counter(), Counter(), Counter(), Counter(), Counter()
    decoration_names = {item["value"]: item["enumerant"] for kind in document["operand_kinds"]
        if kind["kind"] == "Decoration" for item in kind["enumerants"]}
    precision_names = {"RelaxedPrecision", "NoContraction", "FPRoundingMode", "FPFastMathMode"}
    for opcode, operands in rows:
        if opcode == 59:
            require(len(operands) >= 3, "Malformed OpVariable")
            variables[complexity.STORAGE_CLASSES.get(operands[2], str(operands[2]))] += 1
        elif opcode in (61, 62):
            require(len(operands) >= (3 if opcode == 61 else 2), "Malformed load/store")
            pointer = operands[2] if opcode == 61 else operands[0]
            storage = pointer_classes.get(value_types.get(pointer))
            name = complexity.STORAGE_CLASSES.get(storage, str(storage)) if storage is not None else "Unknown"
            (loads if opcode == 61 else stores)[name] += 1
        elif opcode in (71, 72, 332):
            index = 2 if opcode == 72 else 1
            require(len(operands) > index, "Malformed precision decoration")
            name = decoration_names.get(operands[index], str(operands[index]))
            if name in precision_names or name.startswith("FP"):
                precision[name] += 1
                target_opcode = result_opcodes.get(operands[0])
                if target_opcode is not None and grammar[target_opcode].get("class") == "Arithmetic":
                    arithmetic_precision[name] += 1
    result.update(variables=sum(variables.values()), variable_storage_classes=dict(sorted(variables.items())),
        ssa_metrics=dict(function_variables=variables["Function"], function_loads=loads["Function"],
            function_stores=stores["Function"], loads_by_storage_class=dict(sorted(loads.items())),
            stores_by_storage_class=dict(sorted(stores.items())), phi=result["opcode_counts"].get("245", 0),
            undef=result["opcode_counts"].get("1", 0), precision_decorations=dict(sorted(precision.items())),
            arithmetic_precision_decorations=dict(sorted(arithmetic_precision.items())),
            arithmetic_opcode_counts=dict(sorted(arithmetic.items())),
            floating_point_type_widths=dict(sorted(float_widths.items())), integer_type_declarations=dict(sorted(integer_types.items())),
            unknown_pointer_loads=loads["Unknown"], unknown_pointer_stores=stores["Unknown"]),
        max_function_instructions=inspected["summary"]["max_function_instructions"],
        cfg_edges=inspected["summary"]["cfg_edges"],
        function_cfg=[dict(instructions=row["instruction_count"], **row["cfg"]) for row in inspected["functions"]],
        grammar_sha256=profile.sha(grammar_path or GRAMMAR_PATH))
    require(result["functions"] == inspected["summary"]["function_count"] and
        result["function_calls"] == inspected["summary"]["static_function_calls"], "Independent structure parsers disagree")
    return result


def normalized_abi(raw, grammar_path=None):
    """Canonical external types/layout/entry/spec ABI, independent of numeric IDs.

    Function body IDs and debug names are excluded. Unknown ABI operand kinds,
    recursive type/constant references or unresolved IDs are rejected rather
    than converted into numeric identifiers that could mask an ABI change.
    """
    document, grammar = _grammar(grammar_path)
    words, rows = _instructions(raw, grammar)
    definitions, decorations, members = {}, {}, {}
    groups, group_rows, globals_, entries, modes, capabilities, extensions = set(), [], [], [], [], [], []
    memory = []
    for opcode, operands in rows:
        kinds = [item["kind"] for item in grammar[opcode].get("operands", [])]
        if kinds[:2] == ["IdResultType", "IdResult"]:
            require(len(operands) >= 2, "Missing ABI typed result")
            identifier = operands[1]
        elif kinds[:1] == ["IdResult"]:
            require(operands, "Missing ABI result")
            identifier = operands[0]
        else:
            identifier = None
        if identifier is not None:
            require(identifier not in definitions, "Duplicate SPIR-V result ID")
            definitions[identifier] = (opcode, operands)
        if opcode in (71, 332):
            require(len(operands) >= 2, "Malformed ABI decoration")
            decorations.setdefault(operands[0], []).append((opcode, operands[1:]))
        elif opcode == 72:
            require(len(operands) >= 3, "Malformed ABI member decoration")
            members.setdefault(operands[0], {}).setdefault(operands[1], []).append((opcode, operands[2:]))
        elif opcode == 73:
            groups.add(operands[0])
        elif opcode in (74, 75):
            group_rows.append((opcode, operands))
        elif opcode == 59 and operands[2] != 7:
            globals_.append(operands[1])
        elif opcode == 15:
            entries.append(operands)
        elif opcode in (16, 331):
            modes.append((opcode, operands))
        elif opcode == 14:
            memory.append(list(operands))
        elif opcode == 17:
            capabilities.append(list(operands))
        elif opcode == 10:
            value, end = _string(operands)
            require(end == len(operands), "Trailing extension operands")
            extensions.append(value)
        elif opcode in (5632, 5633):
            raise ValueError("String decorations require an explicit ABI normalization implementation")
    for opcode, operands in group_rows:
        require(operands and operands[0] in groups, "Unknown ABI decoration group")
        source = decorations.get(operands[0], [])
        if opcode == 74:
            for identifier in operands[1:]:
                decorations.setdefault(identifier, []).extend(source)
        else:
            require((len(operands) - 1) % 2 == 0, "Incomplete ABI group-member pair")
            for index in range(1, len(operands), 2):
                members.setdefault(operands[index], {}).setdefault(operands[index + 1], []).extend(source)
    memo, pending = {}, set()
    decoration_specs = {item["value"]: item for kind in document["operand_kinds"]
        if kind["kind"] == "Decoration" for item in kind["enumerants"]}

    def ordered(values):
        return sorted(values, key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))

    def deco(items):
        result = []
        for opcode, value in items:
            require(value[0] in decoration_specs, "Unknown ABI decoration enumerant")
            specification = decoration_specs[value[0]]
            # Precision affects numerical qualification, not descriptor/layout
            # ABI. Preserve it separately in ssa_metrics, including locals.
            name = specification["enumerant"]
            if name in ("RelaxedPrecision", "NoContraction", "FPRoundingMode", "FPFastMathMode") or name.startswith("FP"):
                continue
            specifications = specification.get("parameters", [])
            if opcode == 332:
                require(all(row["kind"].startswith("Id") for row in specifications), "Unsupported OpDecorateId parameter signature")
            result.append(dict(decoration=value[0], operands=decode(specifications, value[1:])))
        return ordered(result)

    def operands_for(opcode, operands, *, skip_result=True):
        specifications = grammar[opcode].get("operands", [])
        # SpecConstantOp embeds another opcode with that operation's operands.
        if opcode == 52:
            require(len(operands) >= 3 and operands[2] in grammar, "Unknown specialization operation")
            operation = operands[2]
            specs = grammar[operation].get("operands", [])
            require([item["kind"] for item in specs[:2]] == ["IdResultType", "IdResult"], "Unsupported specialization operation signature")
            return [resolve(operands[0]), operation,
                decode(specs[2:], operands[3:])]
        decoded = decode(specifications, operands, skip_result=skip_result)
        return decoded

    def decode(specifications, operands, *, skip_result=False):
        values, cursor = [], 0
        for index, item in enumerate(specifications):
            kind, quantifier = item["kind"], item.get("quantifier")
            remaining = len(operands) - cursor
            if quantifier == "?" and remaining == 0:
                continue
            repetitions = remaining if quantifier == "*" else 1
            require(quantifier != "*" or index == len(specifications) - 1, "Non-final ABI variadic operand")
            if kind in ("LiteralString", "LiteralContextDependentNumber"):
                require(remaining > 0, "Truncated ABI literal")
                if kind == "LiteralString":
                    value, cursor = _string(operands, cursor)
                else:
                    value, cursor = list(operands[cursor:]), len(operands)
                values.append(value)
                continue
            for _ in range(repetitions):
                require(cursor < len(operands), "Truncated ABI operand")
                value = operands[cursor]
                cursor += 1
                if kind == "IdResult":
                    if not skip_result:
                        raise ValueError("Unexpected ABI result ID")
                    continue
                if kind.startswith("Id"):
                    value = resolve(value)
                else:
                    require(not kind.startswith("Pair"), "Unsupported ABI pair operand")
                    # Enumerants with parameters only occur in decorations or
                    # body instructions, handled separately above/below.
                    require(kind.startswith("Literal") or any(row["kind"] == kind and row["category"] in ("ValueEnum", "BitEnum")
                        for row in document["operand_kinds"]), "Unknown ABI operand kind: " + kind)
                values.append(value)
        require(cursor == len(operands), "Unnormalized trailing ABI operands")
        return values

    def resolve(identifier):
        if identifier in memo:
            return memo[identifier]
        require(identifier in definitions and identifier not in pending, "Unresolved or recursive ABI type/constant")
        opcode, operands = definitions[identifier]
        category = grammar[opcode].get("class")
        require(category in ("Type-Declaration", "Constant-Creation") or opcode == 59,
            "ABI references a function-body ID")
        pending.add(identifier)
        value = dict(op=grammar[opcode]["opname"], operands=operands_for(opcode, operands),
            decorations=deco(decorations.get(identifier, [])),
            member_decorations=[dict(member=member, decorations=deco(items))
                for member, items in sorted(members.get(identifier, {}).items())])
        pending.remove(identifier)
        memo[identifier] = value
        return value

    entry_rows = []
    function_keys = {}
    for operands in entries:
        require(len(operands) >= 3, "Truncated ABI entry point")
        name, end = _string(operands, 2)
        key = dict(execution_model=operands[0], name=name)
        require(operands[1] not in function_keys, "Multiple ABI entries for the same function")
        function_keys[operands[1]] = key
        entry_rows.append(dict(**key, interfaces=ordered([resolve(identifier) for identifier in operands[end:]])))
    require(entry_rows, "Missing ABI entry point")
    mode_rows = []
    for opcode, operands in modes:
        require(len(operands) >= 2 and operands[0] in function_keys, "Execution mode references a non-entry function")
        mode_rows.append(dict(entry=function_keys[operands[0]], mode=operands[1],
            operands=[resolve(identifier) for identifier in operands[2:]] if opcode == 331 else list(operands[2:])))
    constants = [resolve(identifier) for identifier, (opcode, _) in definitions.items() if opcode in (48, 49, 50, 51, 52)]
    return dict(spirv_version=words[1], memory_models=ordered(memory), capabilities=ordered(capabilities),
        extensions=sorted(extensions), entry_points=ordered(entry_rows), execution_modes=ordered(mode_rows),
        global_variables=ordered([resolve(identifier) for identifier in globals_]), specialization_constants=ordered(constants),
        scope="ID-independent entry/execution/interface/global resource/type-layout/spec ABI; excludes debug names and function body.")


def qualify_structure(metrics):
    require(metrics["functions"] == 1 and metrics["function_calls"] == 0,
        "SSA experiment fixes one fully inlined function and zero calls")
    require(metrics["ssa_metrics"]["unknown_pointer_loads"] == 0 and metrics["ssa_metrics"]["unknown_pointer_stores"] == 0,
        "Cannot classify every load/store by actual pointer storage class")


def frozen_helpers(output, grammar_path):
    destination = output / "Verifier/Tools/Perf"
    destination.mkdir(parents=True)
    rows = []
    for source in sorted(Path(__file__).parent.glob("*.py")):
        target = destination / source.name
        shutil.copyfile(source, target)
        rows.append(dict(source=str(source.resolve()), file=target.relative_to(output).as_posix(), sha256=profile.sha(target)))
    for source in sorted(Path(grammar_path).parent.glob("*.grammar.json")):
        target = output / "Verifier/SPIRV" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append(dict(source=str(source.resolve()), file=target.relative_to(output).as_posix(), sha256=profile.sha(target)))
    require(any(Path(row["file"]).name == "spirv.core.grammar.json" for row in rows), "Core grammar not frozen")
    experiment.save(output / "FrozenHelpers.json", rows)
    return [Path(row["source"]) for row in rows]


def validate_helpers(output):
    rows = experiment.read(output / "FrozenHelpers.json")
    require(any(Path(row["file"]).name == Path(__file__).name for row in rows), "Actual SSA verifier not archived")
    for row in rows:
        path = (output / row["file"]).resolve(strict=True)
        require(path.is_relative_to((output / "Verifier").resolve()) and profile.sha(path) == row["sha256"], "Frozen helper/grammar changed")
    return rows


def source_provenance(source, *, verify=True):
    source = Path(source).resolve(strict=True)
    if verify:
        boundary.verify(source)
    analysis = experiment.read(source / "summary.json")
    require(analysis["status"] == "passed" and analysis["command"] == "analyze" and
        analysis["experiment"] == boundary.EXPERIMENT and analysis["arms"] == list(boundary.ARMS) and
        analysis["cleanup_passes"] == ANCESTRY_PASSES, "Wrong or unqualified source/cleanup experiment")
    source_run = source / "SourceRun"
    compilation = experiment.read(source_run / "summary.json")
    preparation = experiment.read(source_run / "Preparation/Preparation.json")
    require(compilation["status"] == "passed" and preparation["status"] == "prepared", "Unqualified source compile/preparation")
    original = [plan for plan in preparation["plans"] if plan["label"] == "original-general"]
    require(len(original) == 1 and original[0]["overlay_metadata"]["mechanism"] == "original",
        "Control is not the original full-feature General source")
    metadata = original[0]["overlay_metadata"]
    require(metadata["feature"] == "general" and metadata["requires_no_fiber"] is False and
        metadata["provider_file"] is None and metadata["provider_module"] is None and metadata["provider_source"] is None and
        metadata["changed_files"] == [] and metadata["generated_hashes"] == {} and
        not metadata.get("extra_compiler_flags", []), "Original control added an overlay/provider or feature restriction")
    records = [row for row in compilation["runs"] if row["arm"] == "original-general"]
    require(len(records) == 3 and all(row["identity"] == preparation["reference"] for row in records), "Original source repeats differ from the API reference")
    report = experiment.read(source_run / "Preparation/ReferenceReport.json")
    sample = material.reference_sample(report)
    repository = Path(preparation["repository_root"])
    reference = repository / REFERENCE_RELATIVE
    require(str(reference.resolve()) in preparation["live_inputs"] and
        preparation["live_inputs"][str(reference.resolve())] == profile.sha(source_run / "Preparation/ReferenceReport.json"),
        "Source did not use the fresh explicit-none reference2")
    item = analysis["outputs"]["original-general"]
    require(item["raw"]["identity"] == preparation["reference"] and item["raw"]["counts"]["functions"] == 1 and
        item["raw"]["counts"]["function_calls"] == 0 and item["cleaned"]["counts"]["functions"] == 1 and
        item["cleaned"]["counts"]["function_calls"] == 0, "Original control mixed another function boundary")
    raw, cleaned = source / "original-general/Raw.spv", source / "original-general/Cleaned.spv"
    require(identity(raw) == item["raw"]["identity"] and identity(cleaned) == item["cleaned"]["identity"], "Original source artifact identity mismatch")
    require(raw.read_bytes() == (source_run / "Preparation/Reference.spv").read_bytes(), "Original raw is not byte-identical to API reference")
    require(Path(analysis["origin"]) == repository / SOURCE_RELATIVE, "Wrong fixed General analysis origin")
    return dict(source_arm="original-general", source_analysis_origin=analysis["origin"],
        source_analysis_summary_sha256=profile.sha(source / "summary.json"),
        source_analysis_manifest_sha256=profile.sha(source / "Manifest.json"),
        source_compile_summary_sha256=profile.sha(source_run / "summary.json"),
        source_preparation_manifest_sha256=profile.sha(source_run / "Preparation/Manifest.json"),
        reference_report_sha256=profile.sha(source_run / "Preparation/ReferenceReport.json"),
        reference_origin=str(reference), request_index=sample["request"]["index"],
        raw_identity=item["raw"]["identity"], cleaned_identity=item["cleaned"]["identity"],
        ancestry_passes=ANCESTRY_PASSES, source_files=preparation["source_files"],
        repository_root=str(repository), slang_optimization="O1", source_requires_no_fiber=False)


def prepare(args):
    source = Path(getattr(args, "input_dir", ROOT / SOURCE_RELATIVE)).resolve(strict=True)
    require(source == (ROOT / SOURCE_RELATIVE).resolve(), "SSA input fixes the current original General analysis")
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(schema_version=1, status="running", command="prepare", experiment=EXPERIMENT, arms=list(ARMS), origin=str(output))
    try:
        provenance = source_provenance(source)
        grammar = Path(getattr(args, "grammar", None) or GRAMMAR_PATH).resolve(strict=True)
        _grammar(grammar)
        optimizer = (args.spirv_tools / "spirv-opt.exe").resolve(strict=True)
        validator = (args.spirv_tools / "spirv-val.exe").resolve(strict=True)
        sdk_paths = sorted({optimizer, validator, *optimizer.parent.glob("*.dll")})
        helper_paths = frozen_helpers(output, grammar)
        live_paths = [*helper_paths, *sdk_paths, *[path for path in source.rglob("*") if path.is_file()]]
        repository = Path(provenance["repository_root"])
        for relative, digest in provenance["source_files"].items():
            path = repository / relative
            require(path.is_file() and profile.sha(path) == digest, "Original source/include changed before SSA preparation")
            live_paths.append(path)
        frozen = experiment.snapshots(live_paths)
        shutil.copytree(source, output / "SourceAnalysis")
        shutil.copyfile(source / "original-general/Cleaned.spv", output / "Input.spv")
        sdk = []
        for path in sdk_paths:
            target = output / "FrozenSDK" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            sdk.append(dict(source=str(path), file=target.relative_to(output).as_posix(), sha256=profile.sha(target)))
        experiment.check_snapshots(frozen)
        input_identity = identity(output / "Input.spv")
        require(input_identity == provenance["cleaned_identity"], "Prepared input differs from original cleaned bytes")
        summary.update(status="prepared", repository_root=str(ROOT), source_origin=str(source),
            provenance=provenance, input=input_identity, optimizer=str(optimizer), validator=str(validator),
            sdk_files=sdk, grammar_file="Verifier/SPIRV/spirv.core.grammar.json", grammar_sha256=profile.sha(grammar),
            passes={ARMS[0]: [], ARMS[1]: PASSES}, runtime_postprocess="none", live_inputs=frozen, scope=SCOPE)
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        experiment.save(output / "Preparation.json", summary)
        profile.seal(output)
    return verify_preparation(output)


def verify_preparation(path, *, current_inputs=False):
    path = Path(path).resolve(strict=True)
    legacy.verify_manifest(path)
    validate_helpers(path)
    preparation = experiment.read(path / "Preparation.json")
    require(preparation["status"] == "prepared" and preparation["command"] == "prepare" and
        preparation["experiment"] == EXPERIMENT and preparation["arms"] == list(ARMS) and
        preparation["passes"] == {ARMS[0]: [], ARMS[1]: PASSES} and preparation["runtime_postprocess"] == "none", "Wrong SSA preparation")
    provenance = source_provenance(path / "SourceAnalysis")
    require(preparation["provenance"] == provenance and preparation["source_origin"] == provenance["source_analysis_origin"] and
        preparation["repository_root"] == provenance["repository_root"] and
        identity(path / "Input.spv") == preparation["input"] == provenance["cleaned_identity"], "SSA preparation source/input binding changed")
    require(preparation["grammar_file"] == "Verifier/SPIRV/spirv.core.grammar.json" and
        profile.sha(path / preparation["grammar_file"]) == preparation["grammar_sha256"], "Prepared grammar binding changed")
    frozen = preparation["live_inputs"]
    for source in (path / "SourceAnalysis").rglob("*"):
        if source.is_file():
            original = Path(preparation["source_origin"]) / source.relative_to(path / "SourceAnalysis")
            require(frozen.get(str(original.resolve())) == profile.sha(source), "Archived source provenance differs from its actual input snapshot")
    for row in [*preparation["sdk_files"], *experiment.read(path / "FrozenHelpers.json")]:
        require(frozen.get(row["source"]) == row["sha256"] and profile.sha(path / row["file"]) == row["sha256"], "Prepared tool/helper snapshot omitted or changed")
    require({Path(row["source"]).name for row in preparation["sdk_files"]}.issuperset({"spirv-opt.exe", "spirv-val.exe"}), "Pinned optimizer/validator missing")
    require(len({row["source"] for row in preparation["sdk_files"]}) == len(preparation["sdk_files"]) and
        all((path / row["file"]).resolve(strict=True).is_relative_to((path / "FrozenSDK").resolve()) and
            Path(row["file"]).name == Path(row["source"]).name for row in preparation["sdk_files"]), "SDK archive paths/identities changed")
    require(all(any(row["source"] == preparation[name] and Path(row["file"]).name == filename for row in preparation["sdk_files"])
        for name, filename in (("optimizer", "spirv-opt.exe"), ("validator", "spirv-val.exe"))), "Declared SDK command differs from its frozen executable")
    if current_inputs:
        experiment.check_snapshots(frozen)
    return preparation


def pass_command(optimizer, source, target):
    return [str(optimizer), "--target-env=vulkan1.4", "--scalar-block-layout", *PASSES, str(source), "-o", str(target)]


def _version(directory, stage):
    text = "\n".join((directory / stage[name]).read_text(encoding="utf-8", errors="replace") for name in ("stdout", "stderr"))
    version = next((line.strip() for line in text.splitlines() if "SPIRV-Tools" in line), None)
    require(version is not None and len(version) < 4096, "Actual SDK version report is missing")
    return version


def _prepared_metadata(preparation, preparation_path, tool_version, arm):
    return dict(transform="external-cleaned" if arm == ARMS[0] else "external-ssa",
        input_sha256=preparation["input"]["sha256"], input_bytes=preparation["input"]["bytes"],
        passes=[] if arm == ARMS[0] else PASSES, ancestry_passes=ANCESTRY_PASSES, tool_version=tool_version,
        source_arm="original-general", source_analysis_summary_sha256=preparation["provenance"]["source_analysis_summary_sha256"],
        preparation_manifest_sha256=profile.sha(preparation_path / "Manifest.json"))


def summarize(records):
    samples = [row["optimizer"]["wall_seconds"] for row in records if row["arm"] == ARMS[1]]
    require(len(samples) == 3 and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples), "Invalid SSA pass samples")
    return dict(ssa_pass_wall_seconds=samples, median_ssa_pass_wall_seconds=statistics.median(samples),
        control_optimizer_samples=0, scope=SCOPE,
        warning="A smaller module or fewer loads/variables does not establish Driver or GPU improvement.")


def structure_observations(outputs):
    control, candidate = (outputs[arm] for arm in ARMS)
    original, rewritten = control["counts"]["ssa_metrics"], candidate["counts"]["ssa_metrics"]
    return dict(output_bytes_changed=control["identity"] != candidate["identity"],
        phi_delta=rewritten["phi"] - original["phi"], undef_delta=rewritten["undef"] - original["undef"],
        precision_decorations_changed=rewritten["precision_decorations"] != original["precision_decorations"],
        arithmetic_precision_decorations_changed=rewritten["arithmetic_precision_decorations"] != original["arithmetic_precision_decorations"],
        arithmetic_opcode_counts_changed=rewritten["arithmetic_opcode_counts"] != original["arithmetic_opcode_counts"],
        numeric_type_declarations_changed=any(rewritten[name] != original[name] for name in ("floating_point_type_widths", "integer_type_declarations")),
        scope="Record structural/precision changes for subsequent GPU correctness qualification; these are not automatic truncation or benefit criteria.")


def run(args):
    source = Path(args.preparation).resolve(strict=True)
    preparation = verify_preparation(source, current_inputs=True)
    output = legacy.fresh_directory(args.output_dir)
    summary = dict(schema_version=1, status="running", command="run", experiment=EXPERIMENT,
        arms=list(ARMS), origin=str(output), runs=[], outputs={}, scope=SCOPE, runtime_postprocess="none")
    checks = []
    try:
        shutil.copytree(source, output / "Preparation")
        grammar = output / "Preparation" / preparation["grammar_file"]
        helpers = frozen_helpers(output, grammar)
        frozen = experiment.snapshots([*[Path(path) for path in preparation["live_inputs"]], *helpers,
            *[path for path in source.rglob("*") if path.is_file()]])
        experiment.save(output / "MeasuredInputs.json", frozen)
        summary.update(preparation_origin=preparation["origin"], optimizer=preparation["optimizer"], validator=preparation["validator"],
            grammar_file="Verifier/SPIRV/spirv.core.grammar.json", grammar_sha256=preparation["grammar_sha256"],
            source_provenance=preparation["provenance"], passes={ARMS[0]: [], ARMS[1]: PASSES}, versions={})
        cache = {}
        def inspect(path):
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest not in cache:
                cache[digest] = dict(identity=identity(path), counts=counts(raw, grammar), abi=normalized_abi(raw, grammar))
            return cache[digest]
        with profile.experiment_lock():
            baseline = inspect(output / "Preparation/Input.spv")
            qualify_structure(baseline["counts"])
            summary["input_structure"] = baseline
            for name, tool in (("optimizer", preparation["optimizer"]), ("validator", preparation["validator"])):
                experiment.check_snapshots(frozen)
                checks.append(dict(label="before-" + name + "-version", inputs=experiment.snapshots(frozen)))
                stage = invoke([tool, "--version"], output, name + "-version", args.timeout)
                require(stage["status"] == "passed", "SDK version process failed")
                validate_process(stage, [tool, "--version"], output, name + "-version")
                summary["versions"][name] = dict(process=stage, version=_version(output, stage))
                experiment.check_snapshots(frozen)
                checks.append(dict(label="after-" + name + "-version", inputs=experiment.snapshots(frozen)))
            for repetition, arm in schedule():
                name = f"repeat-{repetition:02d}-{arm}"
                directory = output / name
                directory.mkdir()
                row = dict(repetition=repetition, arm=arm, directory=name, optimizer=None)
                summary["runs"].append(row)
                experiment.check_snapshots(frozen)
                checks.append(dict(label="before-" + name, inputs=experiment.snapshots(frozen)))
                path = directory / "Guides.spv"
                if arm == ARMS[0]:
                    shutil.copyfile(output / "Preparation/Input.spv", path)
                else:
                    input_path = directory / "Input.spv"
                    shutil.copyfile(output / "Preparation/Input.spv", input_path)
                    row["optimizer"] = invoke(pass_command(preparation["optimizer"], input_path, path), directory, "ssa", args.timeout)
                    require(row["optimizer"]["status"] == "passed", "SSA rewrite failed or timed out")
                    validate_process(row["optimizer"], pass_command(preparation["optimizer"], input_path, path), directory, "ssa")
                row["validation"] = invoke(validate_command(preparation["validator"], path), directory, "validation", args.timeout)
                require(row["validation"]["status"] == "passed", "SPIR-V validation failed")
                validate_process(row["validation"], validate_command(preparation["validator"], path), directory, "validation")
                experiment.check_snapshots(frozen)
                checks.append(dict(label="after-" + name, inputs=experiment.snapshots(frozen)))
                row.update(inspect(path))
                qualify_structure(row["counts"])
                require(row["abi"] == baseline["abi"], "SSA changed normalized entry/descriptor/push/spec ABI")
                known = [prior["identity"] for prior in summary["runs"][:-1] if prior["arm"] == arm and "identity" in prior]
                require(not known or all(item == row["identity"] for item in known), "SSA output is not deterministic")
                require(arm != ARMS[0] or row["identity"] == preparation["input"], "Control copy changed exact input bytes")
                experiment.save(output / "summary.json", summary)
            for arm in ARMS:
                selected = next(row for row in summary["runs"] if row["arm"] == arm)
                target = output / "artifacts" / arm / "Guides.spv"
                target.parent.mkdir(parents=True)
                shutil.copyfile(output / selected["directory"] / "Guides.spv", target)
                summary["outputs"][arm] = dict(file=target.relative_to(output).as_posix(),
                    **inspect(target), prepared=_prepared_metadata(preparation, output / "Preparation", summary["versions"]["optimizer"]["version"], arm))
        summary.update(status="passed", timing_summary=summarize(summary["runs"]),
            structure_observations=structure_observations(summary["outputs"]))
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
    require(summary["status"] == "passed" and summary["command"] == "run" and summary["experiment"] == EXPERIMENT and
        summary["arms"] == list(ARMS) and summary["passes"] == {ARMS[0]: [], ARMS[1]: PASSES} and
        summary["runtime_postprocess"] == "none", "Wrong or failed SSA run")
    preparation = verify_preparation(output / "Preparation")
    require(summary["preparation_origin"] == preparation["origin"] and summary["source_provenance"] == preparation["provenance"], "SSA run source binding changed")
    grammar = output / summary["grammar_file"]
    require(summary["grammar_file"] == "Verifier/SPIRV/spirv.core.grammar.json" and
        summary["grammar_sha256"] == profile.sha(grammar) == preparation["grammar_sha256"], "SSA run grammar changed")
    frozen = experiment.read(output / "MeasuredInputs.json")
    require(all(frozen.get(path) == digest for path, digest in preparation["live_inputs"].items()), "Run omitted prepared tool/source inputs")
    for row in experiment.read(output / "FrozenHelpers.json"):
        require(frozen.get(row["source"]) == row["sha256"], "Run omitted its actual helper/grammar input")
    for source in (output / "Preparation").rglob("*"):
        if source.is_file():
            original = Path(preparation["origin"]) / source.relative_to(output / "Preparation")
            require(frozen.get(str(original.resolve())) == profile.sha(source), "Archived preparation differs from the actual run input snapshot")
    checks = experiment.read(output / "SnapshotChecks.json")
    labels = [prefix + name for name in ("optimizer-version", "validator-version", *[f"repeat-{repetition:02d}-{arm}" for repetition, arm in schedule()])
        for prefix in ("before-", "after-")]
    require([row["label"] for row in checks] == labels and all(row["inputs"] == frozen for row in checks), "Missing or changed before/after input checks")
    if current_inputs:
        experiment.check_snapshots(frozen)
    previous_end, executions = None, set()
    def process(stage, command, directory, name):
        nonlocal previous_end
        start, end = validate_process(stage, command, directory, name)
        require(stage["execution_id"] not in executions and (previous_end is None or start >= previous_end), "Repeated or overlapping tool process")
        executions.add(stage["execution_id"])
        previous_end = end
    recorded = Path(summary["origin"])
    for name in ("optimizer", "validator"):
        version = summary["versions"][name]
        process(version["process"], [summary[name], "--version"], output, name + "-version")
        require(version["version"] == _version(output, version["process"]), "SDK version text changed")
        require(summary[name] == preparation[name], "Run selected another SDK executable")
    cache = {}
    def inspect(path):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in cache:
            cache[digest] = dict(identity=identity(path), counts=counts(raw, grammar), abi=normalized_abi(raw, grammar))
        return cache[digest]
    baseline = inspect(output / "Preparation/Input.spv")
    require(summary["input_structure"] == baseline, "Prepared input structure changed")
    qualify_structure(baseline["counts"])
    require([(row["repetition"], row["arm"]) for row in summary["runs"]] == schedule(), "Incomplete or changed SSA repetition schedule")
    known = {}
    for row in summary["runs"]:
        name = f"repeat-{row['repetition']:02d}-{row['arm']}"
        directory = output / name
        require(row["directory"] == name, "Changed SSA repetition directory")
        path = directory / "Guides.spv"
        if row["arm"] == ARMS[0]:
            require(row["optimizer"] is None and path.read_bytes() == (output / "Preparation/Input.spv").read_bytes(), "Baseline was optimized or changed")
        else:
            require((directory / "Input.spv").read_bytes() == (output / "Preparation/Input.spv").read_bytes(), "SSA input is not the frozen original cleaned control")
            process(row["optimizer"], pass_command(summary["optimizer"], recorded / name / "Input.spv", recorded / name / "Guides.spv"), directory, "ssa")
        process(row["validation"], validate_command(summary["validator"], recorded / name / "Guides.spv"), directory, "validation")
        actual = inspect(path)
        require(all(row[key] == actual[key] for key in actual), "SPIR-V bytes/structure/ABI changed")
        qualify_structure(actual["counts"])
        require(actual["abi"] == baseline["abi"], "SSA altered normalized ABI")
        require(row["arm"] not in known or known[row["arm"]] == actual, "SSA artifact changed between independent processes")
        known[row["arm"]] = actual
    require(set(summary["outputs"]) == set(ARMS), "Missing SSA artifact outputs")
    for arm, item in summary["outputs"].items():
        relative = f"artifacts/{arm}/Guides.spv"
        require(item["file"] == relative and inspect(output / relative) == known[arm] and
            all(item[key] == known[arm][key] for key in known[arm]), "Selected artifact does not match actual repeated output")
        require(item["prepared"] == _prepared_metadata(preparation, output / "Preparation", summary["versions"]["optimizer"]["version"], arm), "External pass provenance changed")
    require(summary["timing_summary"] == summarize(summary["runs"]), "SSA pass timing summary changed")
    require(summary["structure_observations"] == structure_observations(summary["outputs"]), "SSA structural observations changed")
    return summary


def verify(output, *, current_inputs=False):
    output = Path(output).resolve(strict=True)
    legacy.verify_manifest(output)
    source = output / "Verifier/Tools/Perf/ValidatePathTraceSSA.py"
    require(source.is_file(), "Missing actual archived SSA verifier")
    command = [sys.executable, "-B", str(source), "_verify-frozen", str(output)]
    if current_inputs:
        command.append("--current-inputs")
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(result.returncode == 0, "Archived SSA verification failed: " + result.stderr.strip())
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--input-dir", type=Path, default=ROOT / SOURCE_RELATIVE)
    preparation.add_argument("--spirv-tools", type=Path, default=Path("C:/VulkanSDK/1.4.363.0/Bin"))
    preparation.add_argument("--grammar", type=Path)
    preparation.add_argument("--output-dir", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--preparation", type=Path, required=True)
    execution.add_argument("--output-dir", type=Path, required=True)
    execution.add_argument("--timeout", type=float, default=120)
    for name in ("verify", "check", "_verify-frozen"):
        child = sub.add_parser(name)
        child.add_argument("output", type=Path)
        child.add_argument("--current-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args), indent=2, allow_nan=False))
    elif args.command == "run":
        print(json.dumps(run(args)["timing_summary"], indent=2, allow_nan=False))
    elif args.command == "_verify-frozen":
        result = verify_preparation(args.output, current_inputs=args.current_inputs) if (args.output / "Preparation.json").is_file() else verify_records(args.output, current_inputs=args.current_inputs)
        print(json.dumps(result, allow_nan=False))
    else:
        result = verify(args.output, current_inputs=args.current_inputs or args.command == "check")
        print(json.dumps(dict(status="verified", experiment=result["experiment"]), indent=2))


if __name__ == "__main__":
    main()
