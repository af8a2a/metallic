"""Inspect exact SPIR-V bytes without compiling, optimizing or modifying them.

CFG and callsite expansion are static structure proxies, not executed work or
native ISA. OpLine source attribution is emitted only when the module has it.
Use the SDK's Khronos JSON grammar for complete opcode names and classes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import struct


CORE_NAMES = {
    5: "OpName", 7: "OpString", 8: "OpLine", 11: "OpExtInstImport", 12: "OpExtInst",
    15: "OpEntryPoint", 21: "OpTypeInt", 32: "OpTypePointer", 43: "OpConstant", 54: "OpFunction",
    55: "OpFunctionParameter", 56: "OpFunctionEnd", 57: "OpFunctionCall",
    59: "OpVariable", 61: "OpLoad", 62: "OpStore",
    245: "OpPhi", 246: "OpLoopMerge", 247: "OpSelectionMerge", 248: "OpLabel",
    249: "OpBranch", 250: "OpBranchConditional", 251: "OpSwitch", 252: "OpKill",
    253: "OpReturn", 254: "OpReturnValue", 255: "OpUnreachable", 317: "OpNoLine",
}
DEBUG_OPS = {2, 3, 4, 5, 6, 7, 8, 317, 330}
NON_BODY_OPS = DEBUG_OPS | {54, 55, 56}
TERMINATORS = {249, 250, 251, 252, 253, 254, 255, 4416, 4448, 4449}
STORAGE_CLASSES = {0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 4: "Workgroup",
                   5: "CrossWorkgroup", 6: "Private", 7: "Function", 8: "Generic", 9: "PushConstant",
                   10: "AtomicCounter", 11: "Image", 12: "StorageBuffer", 5349: "PhysicalStorageBuffer"}


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def literal_string(words):
    data = struct.pack(f"<{len(words)}I", *words)
    if b"\0" not in data:
        raise ValueError("SPIR-V literal string is not terminated")
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def find_grammar():
    candidates = []
    if os.environ.get("VULKAN_SDK"):
        candidates.append(Path(os.environ["VULKAN_SDK"]) / "Include/spirv/unified1/spirv.core.grammar.json")
    candidates.extend(sorted(Path("C:/VulkanSDK").glob("*/Include/spirv/unified1/spirv.core.grammar.json"), reverse=True))
    return next((path for path in candidates if path.is_file()), None)


def load_grammar(path=None):
    instructions = {opcode: dict(opname=name, **{"class": "Unknown"}, operands=[])
                    for opcode, name in CORE_NAMES.items()}
    metadata = dict(path=None, sha256=None, coverage="minimal built-in control-flow names")
    extended = {}
    if path:
        path = Path(path).resolve()
        data = path.read_bytes()
        grammar = json.loads(data)
        instructions.update({item["opcode"]: item for item in grammar["instructions"]})
        metadata = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                        coverage="Khronos grammar", revision=grammar.get("revision"))
        glsl = path.with_name("extinst.glsl.std.450.grammar.json")
        if glsl.is_file():
            data = glsl.read_bytes()
            extended["GLSL.std.450"] = {item["opcode"]: item["opname"] for item in json.loads(data)["instructions"]}
            metadata["glsl_std_450"] = dict(path=str(glsl), sha256=hashlib.sha256(data).hexdigest())
    return instructions, extended, metadata


def strongly_connected(graph):
    """Iterative Kosaraju; large fully inlined shaders exceed Python DFS depth."""
    visited, order = set(), []
    for start in graph:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(graph[start]))]
        while stack:
            node, edges = stack[-1]
            successor = next(edges, None)
            if successor is None:
                order.append(node)
                stack.pop()
            elif successor not in visited:
                visited.add(successor)
                stack.append((successor, iter(graph[successor])))
    reverse = {node: [] for node in graph}
    for node, edges in graph.items():
        for target in edges:
            reverse[target].append(node)
    assigned, components = set(), []
    for start in reversed(order):
        if start in assigned:
            continue
        component, stack = [], [start]
        assigned.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for predecessor in reverse[node]:
                if predecessor not in assigned:
                    assigned.add(predecessor)
                    stack.append(predecessor)
        components.append(component)
    return components


def reachable(graph, start):
    found, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node not in found:
            found.add(node)
            stack.extend(graph[node])
    return found


def cfg_metrics(blocks):
    graph = {identifier: set(block["targets"]) for identifier, block in blocks.items()}
    for node, targets in graph.items():
        if not targets.issubset(graph):
            raise ValueError(f"Branch in block {node} targets a missing block")
    if not graph:
        return dict(block_count=0, edge_count=0, reachable_blocks=0, unreachable_blocks=0,
                    cyclic_components=0, largest_cyclic_component_blocks=0,
                    largest_cyclic_component_instructions=0, max_predecessors=0,
                    max_successors=0, max_block_instructions=0, top_blocks=[], top_cycles=[])
    entry = next(iter(graph))
    live = reachable(graph, entry)
    predecessors = Counter(target for targets in graph.values() for target in targets)
    cycles = [component for component in strongly_connected(graph)
              if len(component) > 1 or component[0] in graph[component[0]]]
    cycle_sizes = [dict(block_count=len(component),
                        instruction_count=sum(blocks[node]["instructions"] for node in component),
                        first_block_id=min(component)) for component in cycles]
    cycle_sizes.sort(key=lambda item: item["instruction_count"], reverse=True)
    # E - N + 2 for the connected entry-reachable subgraph; parallel switch
    # destinations are collapsed. This is a graph proxy, not path enumeration.
    live_edges = sum(len(graph[node] & live) for node in live)
    result = dict(block_count=len(graph), edge_count=sum(map(len, graph.values())),
                  reachable_blocks=len(live), unreachable_blocks=len(graph) - len(live),
                  entry_reachable_cyclomatic_complexity=live_edges - len(live) + 2,
                  cyclic_components=len(cycles),
                  largest_cyclic_component_blocks=max((len(cycle) for cycle in cycles), default=0),
                  largest_cyclic_component_instructions=max((item["instruction_count"] for item in cycle_sizes), default=0),
                  max_predecessors=max(predecessors.values(), default=0),
                  max_successors=max(map(len, graph.values()), default=0),
                  max_block_instructions=max(block["instructions"] for block in blocks.values()),
                  top_cycles=cycle_sizes[:20])
    result["top_blocks"] = [dict(block_id=node, instruction_count=block["instructions"],
                                 successors=len(graph[node]), predecessors=predecessors[node])
                            for node, block in sorted(blocks.items(), key=lambda item: item[1]["instructions"], reverse=True)[:20]]
    return result


def call_metrics(functions, entries, names):
    graph = {identifier: set(function["calls"]) for identifier, function in functions.items()}
    if any(not targets.issubset(graph) for targets in graph.values()):
        raise ValueError("OpFunctionCall targets a missing function")
    components = strongly_connected(graph)
    recursive = [component for component in components
                 if len(component) > 1 or component[0] in graph[component[0]]]
    # In Kosaraju order, callees follow callers when the condensed graph is a
    # DAG. Reversal computes expansion without recursive Python evaluation.
    estimates, depths = {}, {}
    for component in reversed(components):
        is_cycle = len(component) > 1 or component[0] in graph[component[0]]
        for identifier in component:
            calls = functions[identifier]["calls"]
            unknown = is_cycle or any(estimates.get(callee) is None for callee in calls)
            estimates[identifier] = None if unknown else functions[identifier]["instruction_count"] + \
                sum(count * estimates[callee] for callee, count in calls.items())
            depths[identifier] = None if unknown else 1 + max((depths[callee] for callee in calls), default=0)
    roots = []
    for entry in entries:
        if entry["function_id"] not in graph:
            raise ValueError("OpEntryPoint targets a missing function")
        live = reachable(graph, entry["function_id"])
        occurrences = Counter({entry["function_id"]: 1})
        contributors = None
        if estimates[entry["function_id"]] is not None:
            for component in components:
                for identifier in component:
                    if occurrences[identifier]:
                        for callee, count in functions[identifier]["calls"].items():
                            occurrences[callee] += occurrences[identifier] * count
            total = estimates[entry["function_id"]]
            contributors = [dict(function_id=identifier, name=names.get(identifier),
                                 static_expanded_occurrences=count,
                                 own_instruction_count=functions[identifier]["instruction_count"],
                                 expanded_own_instruction_contribution=count * functions[identifier]["instruction_count"],
                                 contribution_fraction=count * functions[identifier]["instruction_count"] / total if total else 0)
                            for identifier, count in occurrences.items() if count]
            contributors.sort(key=lambda item: item["expanded_own_instruction_contribution"], reverse=True)
        roots.append(dict(**entry, reachable_function_count=len(live),
                          reachable_unique_function_instructions=sum(functions[node]["instruction_count"] for node in live),
                          callsite_expanded_instruction_proxy=estimates[entry["function_id"]],
                          max_static_call_depth=depths[entry["function_id"]], expanded_contributors=contributors))
    edges = [dict(caller_id=caller, caller_name=names.get(caller), callee_id=callee,
                  callee_name=names.get(callee), static_callsite_count=count)
             for caller, function in functions.items() for callee, count in function["calls"].items()]
    return dict(entry_points=roots, recursive_components=[sorted(component) for component in recursive],
                edges=sorted(edges, key=lambda edge: edge["static_callsite_count"], reverse=True)), estimates, depths


def inspect_bytes(data, grammar=None, extended=None):
    if len(data) < 20 or len(data) % 4:
        raise ValueError("Input is not a raw, word-aligned SPIR-V module")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203 or not words[3] or words[4]:
        raise ValueError("Invalid raw SPIR-V header; Metallic shader-cache files need extraction first")
    grammar = grammar or load_grammar()[0]
    extended = extended or {}
    histogram, categories, ext_histogram = Counter(), Counter(), Counter()
    names, strings, ext_sets, integer_widths, value_types = {}, {}, {}, {}, {}
    pointer_classes, memory_storage = {}, defaultdict(Counter)
    functions, entries, source_lines = {}, [], Counter()
    function, block, location = None, None, None
    offset = 5
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 0xffff
        if not size or offset + size > len(words):
            raise ValueError(f"Invalid SPIR-V instruction boundary at word {offset}")
        operands = words[offset + 1:offset + size]
        minimum = {5: 2, 7: 2, 8: 3, 11: 2, 12: 4, 15: 3, 21: 3, 32: 3, 43: 3,
                   54: 4, 55: 2, 57: 3, 59: 3, 61: 3, 62: 2, 245: 4, 246: 3, 247: 2, 248: 1,
                   249: 1, 250: 3, 251: 2, 254: 1}.get(opcode, 0)
        if len(operands) < minimum:
            raise ValueError(f"Truncated SPIR-V opcode {opcode}")
        spec = grammar.get(opcode, {})
        name = spec.get("opname", f"OpUnknown_{opcode}")
        histogram[opcode] += 1
        category = "RayQuery" if "RayQuery" in name else "RayTracing" if "TraceRay" in name else spec.get("class", "Unknown")
        categories[category] += 1
        kinds = [operand["kind"] for operand in spec.get("operands", [])]
        if kinds[:2] == ["IdResultType", "IdResult"] or opcode in {43, 55, 57, 59, 61, 245}:
            value_types[operands[1]] = operands[0]
        if opcode == 5:
            names[operands[0]] = literal_string(operands[1:])
        elif opcode == 7:
            strings[operands[0]] = literal_string(operands[1:])
        elif opcode == 11:
            ext_sets[operands[0]] = literal_string(operands[1:])
        elif opcode == 12:
            ext_histogram[(operands[2], operands[3])] += 1
        elif opcode == 15:
            entries.append(dict(execution_model=operands[0], function_id=operands[1], name=literal_string(operands[2:])))
        elif opcode == 21:
            integer_widths[operands[0]] = operands[1]
        elif opcode == 32:
            pointer_classes[operands[0]] = operands[1]
        elif opcode == 59:
            memory_storage["OpVariable"][STORAGE_CLASSES.get(operands[2], str(operands[2]))] += 1
        elif opcode in {61, 62}:
            pointer_id = operands[2] if opcode == 61 else operands[0]
            storage_class = pointer_classes.get(value_types.get(pointer_id))
            storage_name = STORAGE_CLASSES.get(storage_class, str(storage_class)) if storage_class is not None else "Unknown"
            memory_storage[name][storage_name] += 1
        elif opcode == 54:
            if function is not None or operands[1] in functions:
                raise ValueError("Nested or duplicate SPIR-V function")
            function = dict(id=operands[1], instruction_count=0, body_instruction_count=0,
                            calls=Counter(), blocks={}, counters=Counter(), categories=Counter())
            functions[operands[1]] = function
            block, location = None, None
        elif opcode == 248:
            if function is None or operands[0] in function["blocks"]:
                raise ValueError("OpLabel outside a function or duplicate label")
            if block is not None and not block["terminated"]:
                raise ValueError("SPIR-V block lacks a terminator")
            block = dict(instructions=0, targets=[], terminated=False)
            function["blocks"][operands[0]] = block
        elif opcode == 8:
            location = tuple(operands[:3])
        elif opcode == 317:
            location = None
        if function is not None:
            function["instruction_count"] += 1
            if opcode not in NON_BODY_OPS:
                function["body_instruction_count"] += 1
                function["categories"][category] += 1
                if location:
                    source_lines[location] += 1
            if block is not None and opcode != 56:
                block["instructions"] += 1
            if opcode == 57:
                function["calls"][operands[2]] += 1
            elif opcode == 245:
                if (len(operands) - 2) % 2:
                    raise ValueError("OpPhi has an incomplete incoming pair")
                incoming = (len(operands) - 2) // 2
                function["counters"]["phi_nodes"] += 1
                function["counters"]["phi_incoming_pairs"] += incoming
                function["counters"]["max_phi_incoming_pairs"] = max(incoming, function["counters"]["max_phi_incoming_pairs"])
            elif opcode in {246, 247, 250, 251}:
                counter = {246: "loop_merges", 247: "selection_merges", 250: "conditional_branches", 251: "switches"}[opcode]
                function["counters"][counter] += 1
                if opcode == 246:
                    function["counters"]["loop_unroll_requested"] += bool(operands[2] & 1)
                    function["counters"]["loop_dont_unroll_requested"] += bool(operands[2] & 2)
            if opcode in TERMINATORS:
                if block is None or block["terminated"]:
                    raise ValueError("SPIR-V terminator outside a block or duplicate terminator")
                block["terminated"] = True
                if opcode == 249:
                    block["targets"] = [operands[0]]
                elif opcode == 250:
                    block["targets"] = list(operands[1:3])
                elif opcode == 251:
                    width = integer_widths.get(value_types.get(operands[0]))
                    if width not in {8, 16, 32, 64}:
                        raise ValueError("Cannot determine OpSwitch integer width; provide the full grammar")
                    literal_words = 2 if width == 64 else 1
                    stride = literal_words + 1
                    if (len(operands) - 2) % stride:
                        raise ValueError("OpSwitch has an incomplete literal/target pair")
                    block["targets"] = [operands[1]] + list(operands[2 + literal_words::stride])
                    function["counters"]["switch_cases"] += (len(operands) - 2) // stride
            if opcode == 56:
                if block is not None and not block["terminated"]:
                    raise ValueError("SPIR-V function ends with an unterminated block")
                function, block, location = None, None, None
        elif opcode in {55, 56, 57, 245, 246, 247, 249, 250, 251}:
            raise ValueError(f"Opcode {opcode} appears outside a function")
        offset += size
    if function is not None:
        raise ValueError("Unterminated SPIR-V function")
    calls, estimates, depths = call_metrics(functions, entries, names)
    rows = []
    for identifier, function in functions.items():
        rows.append(dict(id=identifier, name=names.get(identifier),
                         instruction_count=function["instruction_count"],
                         body_instruction_count=function["body_instruction_count"],
                         static_calls=sum(function["calls"].values()), unique_callees=len(function["calls"]),
                         callsite_expanded_instruction_proxy=estimates[identifier], max_static_call_depth=depths[identifier],
                         **{key: function["counters"][key] for key in ["loop_merges", "selection_merges", "conditional_branches", "switches", "switch_cases", "phi_nodes", "phi_incoming_pairs", "max_phi_incoming_pairs", "loop_unroll_requested", "loop_dont_unroll_requested"]},
                         opcode_categories=dict(function["categories"]), cfg=cfg_metrics(function["blocks"])))
    rows.sort(key=lambda row: row["instruction_count"], reverse=True)
    source = [dict(file_id=file_id, file=strings.get(file_id), line=line, column=column, instruction_count=count)
              for (file_id, line, column), count in source_lines.most_common()]
    total_fn = sum(row["instruction_count"] for row in rows)
    summary = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), fnv1a64=fnv(data),
                   spirv_version=f"{(words[1] >> 16) & 255}.{(words[1] >> 8) & 255}",
                   generator=words[2], id_bound=words[3], instruction_count=sum(histogram.values()),
                   function_count=len(rows), function_instruction_count=total_fn,
                   max_function_instructions=max((row["instruction_count"] for row in rows), default=0),
                   largest_function_fraction=max((row["instruction_count"] for row in rows), default=0) / total_fn if total_fn else 0,
                   instruction_weighted_function_size=sum(row["instruction_count"] ** 2 for row in rows) / total_fn if total_fn else 0,
                   basic_blocks=sum(row["cfg"]["block_count"] for row in rows),
                   cfg_edges=sum(row["cfg"]["edge_count"] for row in rows),
                   loop_merges=histogram[246], selection_merges=histogram[247],
                   conditional_branches=histogram[250], switches=histogram[251], phi_nodes=histogram[245],
                   static_function_calls=histogram[57], op_name_count=histogram[5], op_line_count=histogram[8],
                   source_line_attribution_available=bool(source), source_attributed_instructions=sum(source_lines.values()))
    opcodes = [dict(opcode=opcode, name=grammar.get(opcode, {}).get("opname", f"OpUnknown_{opcode}"),
                    category=grammar.get(opcode, {}).get("class", "Unknown"), count=count,
                    fraction=count / summary["instruction_count"])
               for opcode, count in histogram.most_common()]
    ext_rows = [dict(set=ext_sets.get(set_id), set_id=set_id, instruction=number,
                     name=extended.get(ext_sets.get(set_id), {}).get(number), count=count)
                for (set_id, number), count in ext_histogram.most_common()]
    return dict(summary=summary, opcode_histogram=opcodes, opcode_categories=dict(categories),
                memory_operations_by_storage_class={name: dict(counts) for name, counts in memory_storage.items()},
                extended_instruction_histogram=ext_rows, functions=rows, call_graph=calls, source_lines=source,
                limitations=["Static SPIR-V structure is not native ISA or dynamically executed work.",
                             "Callsite expansion visits each static callsite once, including sites inside loops; it is a hypothetical full-inlining size proxy.",
                             "Recursive call graphs have no finite expansion proxy.",
                             "CFG SCCs collapse nested loops; cyclic component counts are not source loop counts.",
                             "Debug/source instructions may be stripped or transformed; absent OpLine cannot attribute code to source lines.",
                             "This reader checks selected structural invariants and is not a substitute for spirv-val."])


def recipe_inputs(directories):
    inputs = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.recipe.json")):
            recipe = json.loads(path.read_text(encoding="utf-8"))
            for stage, shader in recipe.get("shaders", {}).items():
                if not shader.get("spirv_file"):
                    continue
                file = (path.parent / shader["spirv_file"]).resolve()
                if not file.is_relative_to(path.parent.resolve()):
                    raise ValueError("Recipe shader path escapes its directory")
                label = f"{shader.get('slang_optimization', 'default')}:{shader.get('module_name', stage)}.{shader.get('source_entry_point', shader.get('entry_point', 'main'))}"
                inputs.append(dict(label=label, path=file, recipe=str(path.resolve()),
                                   metadata=shader, source_device=recipe.get("source_device")))
    return inputs


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg(modules):
    metrics = [("instruction_count", "SPIR-V instructions"), ("basic_blocks", "Basic blocks"),
               ("max_function_instructions", "Largest function instructions"), ("cfg_edges", "CFG edges")]
    width, left, row_height = 1220, 625, 30
    height = 85 + len(metrics) * (35 + len(modules) * row_height)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#fff"/>',
                '<g font-family="Segoe UI, sans-serif" font-size="12" fill="#20242a">',
                '<text x="20" y="27" font-size="18">Static SPIR-V complexity (exact input bytes)</text>',
                '<text x="20" y="50">Linear scale within each metric; these counts do not measure GPU work or Driver compilation time.</text>']
    y = 80
    for metric, title in metrics:
        elements.append(f'<text x="20" y="{y}" font-size="15">{title}</text>')
        y += 26
        maximum = max((module["summary"][metric] for module in modules), default=1) or 1
        for index, module in enumerate(modules):
            value = module["summary"][metric]
            label = module["label"]
            if len(label) > 82:
                label = label[:79] + "..."
            color = ["#2b75b7", "#d5771a", "#27966b"][index % 3]
            elements.extend([f'<text x="20" y="{y}">{html.escape(label)}</text>',
                             f'<rect x="{left}" y="{y - 12}" width="{max(1, value / maximum * 445):.1f}" height="17" fill="{color}"/>',
                             f'<text x="1080" y="{y}">{value:,}</text>'])
            y += row_height
        y += 14
    return "\n".join(elements + ["</g></svg>"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spirv", action="append", default=[], metavar="LABEL=FILE", help="Raw SPIR-V; repeated")
    parser.add_argument("--recipe-dir", action="append", default=[], type=Path)
    parser.add_argument("--filter", default="", help="Case-insensitive label substring")
    parser.add_argument("--grammar", type=Path, default=find_grammar())
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args(argv)
    try:
        instructions, extended, metadata = load_grammar(options.grammar)
        inputs = recipe_inputs(options.recipe_dir)
        for item in options.spirv:
            if "=" not in item:
                raise ValueError("--spirv must be LABEL=FILE")
            label, path = item.split("=", 1)
            inputs.append(dict(label=label, path=Path(path).resolve(), recipe=None, metadata=None, source_device=None))
        inputs = [item for item in inputs if options.filter.lower() in item["label"].lower()]
        if not inputs:
            raise ValueError("No SPIR-V inputs matched")
        if options.output_dir.exists():
            raise ValueError("Output directory already exists; use a fresh directory")
        modules, seen = [], set()
        for item in inputs:
            data = item["path"].read_bytes()
            result = inspect_bytes(data, instructions, extended)
            identity = (item["label"], result["summary"]["sha256"])
            if identity in seen:
                continue
            seen.add(identity)
            shader = item["metadata"]
            if shader and (result["summary"]["bytes"] != shader.get("spirv_bytes") or result["summary"]["fnv1a64"] != shader.get("input_spirv_hash")):
                raise ValueError("Recipe bytes/hash disagree with the raw SPIR-V file")
            modules.append(dict(label=item["label"], input=str(item["path"]), recipe=item["recipe"],
                                recipe_shader=shader, source_device=item["source_device"], **result))
        options.output_dir.mkdir(parents=True)
        report = dict(schema_version=1, grammar=metadata, modules=modules,
                      scope="Read-only static analysis of exact recorded SPIR-V; no compilation, native ISA or dispatch measurement")
        (options.output_dir / "Complexity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary_rows = [dict(label=module["label"], input=module["input"], **module["summary"]) for module in modules]
        write_csv(options.output_dir / "Modules.csv", summary_rows, list(summary_rows[0]))
        function_rows = [dict(label=module["label"], **{key: value for key, value in row.items() if key not in {"cfg", "opcode_categories"}},
                              **{f"cfg_{key}": value for key, value in row["cfg"].items() if not isinstance(value, list)})
                         for module in modules for row in module["functions"]]
        write_csv(options.output_dir / "Functions.csv", function_rows, list(function_rows[0]) if function_rows else ["label", "id", "name"])
        opcode_rows = [dict(label=module["label"], **row) for module in modules for row in module["opcode_histogram"]]
        write_csv(options.output_dir / "Opcodes.csv", opcode_rows, ["label", "opcode", "name", "category", "count", "fraction"])
        call_rows = [dict(label=module["label"], **row) for module in modules for row in module["call_graph"]["edges"]]
        write_csv(options.output_dir / "Calls.csv", call_rows, ["label", "caller_id", "caller_name", "callee_id", "callee_name", "static_callsite_count"])
        expansion_rows = [dict(label=module["label"], entry=entry["name"], **row)
                          for module in modules for entry in module["call_graph"]["entry_points"]
                          for row in (entry["expanded_contributors"] or [])]
        write_csv(options.output_dir / "Expansion.csv", expansion_rows,
                  ["label", "entry", "function_id", "name", "static_expanded_occurrences", "own_instruction_count", "expanded_own_instruction_contribution", "contribution_fraction"])
        (options.output_dir / "Complexity.svg").write_text(svg(modules), encoding="utf-8")
        print(json.dumps(dict(status="passed", modules=len(modules), report=str((options.output_dir / "Complexity.json").resolve()))))
    except (ValueError, OSError, KeyError, struct.error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
