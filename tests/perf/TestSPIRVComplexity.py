"""Check static graph semantics, input identity and truthful source attribution."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Tools/Perf"))
import AnalyzeSPIRVComplexity as analysis


def instruction(opcode, *operands):
    return [((len(operands) + 1) << 16) | opcode, *operands]


def string_words(text):
    data = text.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    return struct.unpack(f"<{len(data) // 4}I", data)


def module(*instructions):
    words = [0x07230203, 0x00010600, 0, 1000, 0]
    for item in instructions:
        words.extend(item)
    return struct.pack(f"<{len(words)}I", *words)


def simple_function(identifier, label, *body):
    return [instruction(54, 1, identifier, 0, 2), instruction(248, label),
            *body, instruction(253), instruction(56)]


def entry(identifier=10):
    return instruction(15, 5, identifier, *string_words("main"))


class SPIRVComplexityTests(unittest.TestCase):
    def test_callsite_expansion_preserves_multiplicity_and_unique_reachability(self):
        data = module(entry(), instruction(5, 10, *string_words("root")),
                      instruction(5, 40, *string_words("helper")),
                      *simple_function(10, 20, instruction(57, 1, 30, 40), instruction(57, 1, 31, 40)),
                      *simple_function(40, 50))
        result = analysis.inspect_bytes(data)
        root = result["call_graph"]["entry_points"][0]
        self.assertEqual(root["reachable_function_count"], 2)
        self.assertEqual(root["reachable_unique_function_instructions"], 10)
        self.assertEqual(root["callsite_expanded_instruction_proxy"], 14)
        self.assertEqual(root["max_static_call_depth"], 2)
        self.assertEqual(sum(row["expanded_own_instruction_contribution"] for row in root["expanded_contributors"]), 14)
        self.assertEqual(root["expanded_contributors"][0]["static_expanded_occurrences"], 2)
        self.assertEqual(result["call_graph"]["edges"][0]["static_callsite_count"], 2)
        self.assertEqual(result["call_graph"]["edges"][0]["callee_name"], "helper")
        self.assertEqual(result["summary"]["sha256"], hashlib.sha256(data).hexdigest())

    def test_recursive_graph_has_no_finite_expansion(self):
        result = analysis.inspect_bytes(module(entry(),
            *simple_function(10, 20, instruction(57, 1, 30, 40)),
            *simple_function(40, 50, instruction(57, 1, 31, 10))))
        self.assertEqual(result["call_graph"]["recursive_components"], [[10, 40]])
        self.assertIsNone(result["call_graph"]["entry_points"][0]["callsite_expanded_instruction_proxy"])
        self.assertIsNone(result["call_graph"]["entry_points"][0]["expanded_contributors"])

    def test_diamond_cfg_counts_join_edges_and_phi_inputs(self):
        result = analysis.inspect_bytes(module(entry(), instruction(21, 1, 32, 0), instruction(43, 1, 4, 1),
            instruction(54, 1, 10, 0, 2), instruction(248, 20), instruction(247, 23, 0),
            instruction(250, 4, 21, 22), instruction(248, 21), instruction(249, 23),
            instruction(248, 22), instruction(249, 23), instruction(248, 23),
            instruction(245, 1, 5, 4, 21, 4, 22), instruction(253), instruction(56)))
        function = result["functions"][0]
        self.assertEqual(function["cfg"]["edge_count"], 4)
        self.assertEqual(function["cfg"]["entry_reachable_cyclomatic_complexity"], 2)
        self.assertEqual(function["cfg"]["max_predecessors"], 2)
        self.assertEqual(function["phi_incoming_pairs"], 2)
        self.assertEqual(function["selection_merges"], 1)
        self.assertEqual(function["cfg"]["cyclic_components"], 0)

    def test_loop_scc_and_unroll_request_are_structural_not_iterations(self):
        result = analysis.inspect_bytes(module(entry(), instruction(54, 1, 10, 0, 2),
            instruction(248, 20), instruction(246, 22, 21, 1), instruction(249, 21),
            instruction(248, 21), instruction(250, 4, 20, 22),
            instruction(248, 22), instruction(253), instruction(56)))
        function = result["functions"][0]
        self.assertEqual(function["loop_merges"], 1)
        self.assertEqual(function["loop_unroll_requested"], 1)
        self.assertEqual(function["cfg"]["cyclic_components"], 1)
        self.assertEqual(function["cfg"]["largest_cyclic_component_blocks"], 2)
        self.assertEqual(function["cfg"]["reachable_blocks"], 3)

    def test_switch_64_bit_literal_is_not_mistaken_for_target(self):
        result = analysis.inspect_bytes(module(entry(), instruction(21, 1, 64, 0), instruction(43, 1, 4, 0, 0),
            instruction(54, 1, 10, 0, 2), instruction(248, 20), instruction(251, 4, 22, 123, 456, 21),
            instruction(248, 21), instruction(249, 22), instruction(248, 22), instruction(253), instruction(56)))
        self.assertEqual(result["functions"][0]["switch_cases"], 1)
        self.assertEqual(result["functions"][0]["cfg"]["edge_count"], 3)

    def test_unreachable_cfg_blocks_do_not_inflate_entry_complexity(self):
        result = analysis.inspect_bytes(module(entry(), instruction(54, 1, 10, 0, 2),
            instruction(248, 20), instruction(253), instruction(248, 21), instruction(249, 21), instruction(56)))
        cfg = result["functions"][0]["cfg"]
        self.assertEqual(cfg["unreachable_blocks"], 1)
        self.assertEqual(cfg["entry_reachable_cyclomatic_complexity"], 1)
        self.assertEqual(cfg["cyclic_components"], 1)

    def test_source_attribution_requires_real_op_line_and_no_line_clears_it(self):
        result = analysis.inspect_bytes(module(entry(), instruction(7, 90, *string_words("Actual.slang")),
            instruction(54, 1, 10, 0, 2), instruction(248, 20), instruction(8, 90, 12, 2),
            instruction(57, 1, 30, 40), instruction(317), instruction(253), instruction(56),
            *simple_function(40, 50)))
        self.assertTrue(result["summary"]["source_line_attribution_available"])
        self.assertEqual(result["source_lines"], [dict(file_id=90, file="Actual.slang", line=12, column=2, instruction_count=1)])
        stripped = analysis.inspect_bytes(module(entry(), *simple_function(10, 20)))
        self.assertFalse(stripped["summary"]["source_line_attribution_available"])
        self.assertEqual(stripped["source_lines"], [])

    def test_memory_operations_distinguish_function_ir_from_storage_buffer(self):
        result = analysis.inspect_bytes(module(entry(), instruction(21, 1, 32, 0),
            instruction(32, 6, 7, 1), instruction(32, 7, 12, 1), instruction(59, 7, 8, 12),
            *simple_function(10, 20, instruction(59, 6, 9, 7), instruction(61, 1, 30, 9),
                             instruction(61, 1, 31, 8), instruction(62, 9, 31))))
        self.assertEqual(result["memory_operations_by_storage_class"],
                         dict(OpVariable=dict(StorageBuffer=1, Function=1),
                              OpLoad=dict(Function=1, StorageBuffer=1), OpStore=dict(Function=1)))

    def test_iterative_graph_handles_large_inlined_cfg(self):
        graph = {index: {index + 1} if index < 9999 else {5000} for index in range(10000)}
        components = analysis.strongly_connected(graph)
        self.assertEqual(max(map(len, components)), 5000)

    def test_malformed_raw_module_and_graph_rejected(self):
        cases = [b"MTLSPV01" + bytes(40), module(entry(), instruction(54, 1, 10, 0, 2)),
                 module(entry(), *simple_function(10, 20, instruction(249, 99))),
                 module(entry(), *simple_function(10, 20, instruction(57, 1, 30, 99)))]
        for data in cases:
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                analysis.inspect_bytes(data)
        data = bytearray(module(entry(), *simple_function(10, 20)))
        struct.pack_into("<I", data, 20, 0)
        with self.assertRaisesRegex(ValueError, "boundary"):
            analysis.inspect_bytes(data)

    def test_recipe_identity_mismatch_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data = module(entry(), *simple_function(10, 20))
            (directory / "input.spv").write_bytes(data)
            recipe = dict(shaders=dict(compute=dict(spirv_file="input.spv", spirv_bytes=len(data), input_spirv_hash="bad")))
            (directory / "compute.recipe.json").write_text(json.dumps(recipe))
            output = directory / "out"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analysis.main(["--recipe-dir", str(directory), "--output-dir", str(output)])
            self.assertFalse(output.exists())

    def test_recipe_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compute.recipe.json"
            path.write_text(json.dumps(dict(shaders=dict(compute=dict(spirv_file="../outside.spv")))))
            with self.assertRaisesRegex(ValueError, "escapes"):
                analysis.recipe_inputs([directory])

    def test_cli_writes_exact_identity_and_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path, output = directory / "test.spv", directory / "out"
            path.write_bytes(module(entry(), *simple_function(10, 20)))
            arguments = ["--spirv", f"A & B={path}", "--output-dir", str(output)]
            with redirect_stdout(io.StringIO()):
                analysis.main(arguments)
            report = json.loads((output / "Complexity.json").read_text())
            self.assertEqual(report["modules"][0]["summary"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertIn("A &amp; B", (output / "Complexity.svg").read_text())
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                analysis.main(arguments)


if __name__ == "__main__":
    unittest.main()
