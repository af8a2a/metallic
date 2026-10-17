"""CPU-only evidence-policy tests; no graphics device or compiler process."""
from pathlib import Path
import copy
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools/Perf'))
import VkGltfSER as evidence
import VkGltfSERCompile as compiler
import PathTraceTraversal as old


def graph(arm):
    return dict(nodes=[dict(name='PathTrace', type='ScenePathTracePass', properties=dict(
        traversalBackend='rayPipeline', cacheMode='off', samples=1, exportDenoiserGuides=True, **evidence.ARMS[arm]))])


def spirv(ser=False):
    raw = struct.pack('<5I', 0x07230203, 0x00010600, 0, 1, 0)
    if ser:
        name = b'SPV_EXT_shader_invocation_reorder\0'
        name += b'\0' * ((-len(name)) % 4)
        raw += struct.pack('<I', ((1 + len(name) // 4) << 16) | 10) + name
    return raw


class VkGltfSEREvidenceTests(unittest.TestCase):
    def setUp(self):
        self.test_root = ROOT / 'build/vkgltf-ser-tests'
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.test_root)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.test_root.resolve()))
        self.temporary.cleanup()

    def test_declared_model_and_ser_are_the_only_graph_changes(self):
        self.assertEqual(evidence.normalized_graph(graph('O'), 'O'), evidence.normalized_graph(graph('G'), 'G'))
        self.assertEqual(evidence.normalized_graph(graph('G'), 'G'), evidence.normalized_graph(graph('S'), 'S'))
        changed = graph('G')
        changed['nodes'][0]['properties']['samples'] = 4
        self.assertNotEqual(evidence.normalized_graph(graph('O'), 'O'), evidence.normalized_graph(changed, 'G'))
        with self.assertRaisesRegex(ValueError, 'property changed'):
            evidence.normalized_graph(graph('G'), 'S')

    def test_legacy_runner_still_rejects_vkgltf(self):
        with self.assertRaisesRegex(ValueError, 'OpenPBR/cache-off'):
            old.normalized_graph(graph('G'), 'B')
        self.assertEqual((evidence.ABSOLUTE, evidence.RELATIVE), (2e-5, 2e-5))

    def test_balanced_three_arm_performance_schedule(self):
        values = list(evidence.schedule('performance', [{'id': 'Test'}]))
        self.assertEqual([v[2] for v in values], list('OGSGSOSOG'))
        self.assertEqual(len(values), 9)
        for position in range(3):
            self.assertEqual({values[block * 3 + position][2] for block in range(3)}, set('OGS'))
        correctness = list(evidence.schedule('correctness', [{'id': 'Test'}]))
        self.assertEqual([v[2] for v in correctness], list('OGSSGO'))

    def test_spirv_extension_is_decoded_not_searched_as_arbitrary_bytes(self):
        self.assertEqual(evidence.spirv_extensions(spirv()), [])
        self.assertEqual(evidence.spirv_extensions(spirv(True)), ['SPV_EXT_shader_invocation_reorder'])
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            evidence.spirv_extensions(spirv() + struct.pack('<I', 10))

    def write_recipe(self, arm):
        directory = self.directory / arm
        directory.mkdir()
        payload = directory / 'raygen.spv'
        payload.write_bytes(spirv(arm == 'S'))
        hit = directory / 'hit.spv'
        hit.write_bytes(spirv())
        shaders = {}
        for name, suffix in [('ray_generation', 'PathTraceMain'), ('miss', 'Miss'), ('closest_hit', 'ClosestHit'), ('any_hit', 'AnyHit')]:
            raw = payload.read_bytes() if name == 'ray_generation' else hit.read_bytes()
            shaders[name] = dict(spirv_file=payload.name if name == 'ray_generation' else hit.name,
                spirv_bytes=len(raw), input_spirv_hash=old.fnv1a(raw), module_name=evidence.VK_MODULE if name == 'ray_generation' else evidence.VK_HIT_MODULE,
                source_entry_point='vkGltfRayPipeline' + suffix, entry_point='main', debug_mode='release',
                descriptor_heap_mode='mapped', slang_optimization='O1', source_identity_ambiguous=False,
                spirv_postprocess='none', spirv_postprocess_tool_version='', specialization_constants=[])
        recipe = dict(schema='metallic_pipeline_recipe', version=4 if arm == 'S' else 3, pipeline_type='ray_tracing',
            expected_pso_hash='0000000000000001', cache_group='ScenePathTracePass', shaders=shaders,
            state=dict(disable_optimization=False, uses_bindless_heap=True, bindless_user_push_data_size=12,
                max_recursion_depth=1, cluster_acceleration_structure=True, enable_invocation_reorder=arm == 'S'),
            source_device=dict(validation_enabled=True, synchronization_validation_enabled=True,
                enabled_features=dict(ray_tracing_pipeline=True, ray_tracing_invocation_reorder=True)))
        path = directory / 'recipe.recipe.json'
        evidence.save(path, recipe)
        return path, recipe

    def inspect(self, path, arm):
        return evidence.recipe_evidence(path.parent, arm, True, {'id': 'Test'}, graph(arm))

    def test_ser_v4_requires_actual_extension_and_feature(self):
        path, recipe = self.write_recipe('S')
        self.inspect(path, 'S')
        recipe['source_device']['enabled_features']['ray_tracing_invocation_reorder'] = False
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'device feature'):
            self.inspect(path, 'S')

    def test_ser_native_state_cannot_be_mislabeled(self):
        path, recipe = self.write_recipe('G')
        self.inspect(path, 'G')
        recipe['state']['enable_invocation_reorder'] = True
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'Native SER state'):
            self.inspect(path, 'G')

    def test_wrong_ser_payload_is_rejected_even_if_hash_is_updated(self):
        path, recipe = self.write_recipe('S')
        payload = path.parent / 'raygen.spv'
        payload.write_bytes(spirv())
        recipe['shaders']['ray_generation'].update(spirv_bytes=len(spirv()), input_spirv_hash=old.fnv1a(spirv()))
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'SPIR-V SER identity'):
            self.inspect(path, 'S')

    def test_missing_hit_stage_is_rejected(self):
        path, recipe = self.write_recipe('G')
        del recipe['shaders']['any_hit']
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'Incomplete/changed'):
            self.inspect(path, 'G')

    def write_readback(self, arm, value=1.0, guide=0.5, unrelated='same'):
        directory = self.directory / arm
        directory.mkdir()
        (directory / 'image.bin').write_bytes(struct.pack('<4f', value, value, value, 1))
        (directory / 'guide.bin').write_bytes(struct.pack('<4f', guide, guide, guide, 1))
        case = dict(id='Test', width=1, height=1, format='RGBA32F', image='image.bin', output='PathTrace.color',
            resolvedGraph=graph(arm), captures=[dict(output='PathTrace.normalRoughness', frame=1, file='guide.bin', bytes=16, format=42)])
        path = directory / 'MaterialBaseline.json'
        evidence.save(path, {'cases': [case]})
        shared = {stage: dict(hash='same', extensions=[]) for stage in ('miss', 'closest_hit', 'any_hit')}
        return dict(report=str(path), arm=arm, recipes=dict(device={'device': 'same'}, unrelated=[unrelated], target=dict(shaders=shared)))

    def test_different_model_differences_are_descriptive_not_equivalent(self):
        a, b = self.write_readback('O'), self.write_readback('G', value=2, guide=0.3, unrelated='LUT excluded')
        result = evidence.compare(a, b)
        self.assertIsNone(result['qualified'])
        self.assertFalse(result['equivalence_required'])
        self.assertFalse(result['readbacks'][0]['strict_equivalent'])
        self.assertEqual(result['readbacks'][0]['max_absolute_error'], 1)

    def test_ser_numerical_difference_is_failure(self):
        a, b = self.write_readback('G'), self.write_readback('S', value=2)
        result = evidence.compare(a, b)
        self.assertFalse(result['qualified'])
        self.assertTrue(result['equivalence_required'])

    def test_ser_guide_requires_byte_exact_even_below_hdr_tolerance(self):
        a, b = self.write_readback('G'), self.write_readback('S', guide=0.500001)
        result = evidence.compare(a, b)
        self.assertFalse(result['qualified'])
        self.assertTrue(result['readbacks'][0]['strict_equivalent'])

    def test_same_arm_difference_and_nonfinite_are_rejected(self):
        a, b = self.write_readback('G'), self.write_readback('S', value=2)
        report = evidence.read(b['report'])
        report['cases'][0]['resolvedGraph'] = graph('G')
        evidence.save(Path(b['report']), report)
        b['arm'] = 'G'
        self.assertFalse(evidence.compare(a, b)['qualified'])
        (Path(b['report']).parent / 'image.bin').write_bytes(struct.pack('<4f', float('nan'), 1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evidence.compare(a, b)

    def test_ser_non_target_pipeline_changes_are_rejected(self):
        a, b = self.write_readback('G'), self.write_readback('S', unrelated='changed')
        with self.assertRaisesRegex(ValueError, 'non-target'):
            evidence.compare(a, b)

    def test_ser_hit_stage_change_is_rejected(self):
        a, b = self.write_readback('G'), self.write_readback('S')
        b['recipes']['target']['shaders']['closest_hit']['hash'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'shared hit stage'):
            evidence.compare(a, b)

    def test_ser_extension_cannot_enter_shared_hit_stages(self):
        shaders = {stage: dict(extensions=['SPV_EXT_shader_invocation_reorder']) for stage in ('miss', 'closest_hit', 'any_hit')}
        with self.assertRaisesRegex(ValueError, 'must not enter'):
            evidence.require_shared_hit_stages(shaders, copy.deepcopy(shaders))

    def test_three_arm_statistics_separate_material_and_ser(self):
        rows = [dict(role='single', arm=arm, value=value) for block in range(3) for arm, value in [('O', 10), ('G', 5), ('S', 4)]]
        result = compiler.statistics_for(rows, 'value')['single']
        self.assertEqual(result['contrasts']['material']['paired_median_percent'], -50)
        self.assertAlmostEqual(result['contrasts']['ser']['paired_median_percent'], -20)
        rows.pop()
        with self.assertRaisesRegex(ValueError, 'Three positive'):
            compiler.statistics_for(rows, 'value')

    def test_compile_command_uses_recipe_source_and_explicit_request(self):
        stage = dict(module_name=evidence.VK_MODULE, source_entry_point='vkGltfRayPipelinePathTraceMain')
        request = dict(profile='sm_6_6', capabilities=['spvShaderInvocationReorderEXT'], defines=[['METALLIC_VKGLTF_SER', '1']])
        command = compiler.compile_command(Path('slangc'), stage, request, Path('shader.spv'))
        self.assertIn('spvShaderInvocationReorderEXT', command)
        self.assertIn('METALLIC_VKGLTF_SER=1', command)
        self.assertNotIn('-report-detailed-perf-benchmark', command)
        with self.assertRaisesRegex(ValueError, 'Unknown compile request'):
            compiler.compile_command(Path('slangc'), stage, dict(request, arbitrary_flags=['-O0']), Path('shader.spv'))


if __name__ == '__main__':
    unittest.main()
