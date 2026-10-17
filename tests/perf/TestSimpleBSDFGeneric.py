"""CPU-only contract tests for direct/generic same-BSDF evidence."""
from pathlib import Path
import copy
import struct
import sys
import tempfile
import unittest
import shutil
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools/Perf'))
import SimpleBSDFGeneric as evidence
import SimpleBSDFGenericCompile as compiler
import VkGltfSER as previous


def graph(arm):
    return dict(nodes=[dict(name='PathTrace', type='ScenePathTracePass', properties=dict(
        traversalBackend='rayPipeline', cacheMode='off', samples=1, exportDenoiserGuides=True, **evidence.ARMS[arm]))])


def requests(role, arm):
    result = {}
    for stage in evidence.traversal.RT_STAGES:
        raygen = stage == 'ray_generation'
        defines = [['METALLIC_CUSTOM_MATERIALS', '0'], ['METALLIC_GLOBAL_VIEW', '0'], ['METALLIC_HAS_RTXCR', '0'],
            ['METALLIC_VKGLTF_GUIDES', '1' if raygen else '0'],
            ['METALLIC_PT_SINGLE_SAMPLE', '1' if raygen and role == 'single' else '0'], ['METALLIC_VKGLTF_SER', '0']]
        if raygen and arm == 'B':
            defines += [['METALLIC_SIMPLE_BSDF', '1']]
        result[stage] = dict(profile='spirv_1_6', capabilities=['spvRayTracingKHR'], defines=defines, search_paths=[])
    return result


def policy():
    return dict(device={'device': 'same'}, unrelated=[{'shader': 'same'}], target=dict(cache_group='ScenePathTracePass',
        state={'enable_invocation_reorder': False}, shaders={stage: dict(sha256='same', extensions=[])
        for stage in evidence.traversal.RT_STAGES}))


class SimpleBSDFGenericTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'build/simple-bsdf-generic-tests'
        self.root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.root.resolve()))
        self.temp.cleanup()

    def readback(self, arm, value=1., guide=.5):
        directory = self.directory / arm
        directory.mkdir()
        (directory / 'hdr.bin').write_bytes(struct.pack('<4f', value, value, value, 1.))
        (directory / 'guide.bin').write_bytes(struct.pack('<f', guide))
        case = dict(id='Test', width=1, height=1, format='RGBA32F', image='hdr.bin', output='PathTrace.color',
            resolvedGraph=graph(arm), captures=[dict(output='PathTrace.depth', frame=8, file='guide.bin', bytes=4, format=47)])
        path = directory / 'MaterialBaseline.json'
        evidence.save(path, dict(cases=[case]))
        return dict(report=str(path), arm=arm, recipes=policy())

    def test_only_wrapper_selection_may_differ(self):
        self.assertEqual(evidence.normalized_graph(graph('A'), 'A'), evidence.normalized_graph(graph('B'), 'B'))
        changed = graph('B')
        changed['nodes'][0]['properties']['samples'] = 4
        self.assertNotEqual(evidence.normalized_graph(graph('A'), 'A'), evidence.normalized_graph(changed, 'B'))
        changed['nodes'][0]['properties']['shaderExecutionReorder'] = True
        with self.assertRaisesRegex(ValueError, 'property changed'):
            evidence.normalized_graph(changed, 'B')

    def test_old_contract_is_unchanged_and_rejects_simple(self):
        with self.assertRaisesRegex(ValueError, 'property changed'):
            previous.normalized_graph(graph('B'), 'G')
        self.assertEqual(previous.ORDER, ('OGS', 'GSO', 'SOG'))
        self.assertEqual((evidence.ABSOLUTE, evidence.RELATIVE), (2e-5, 2e-5))

    def test_correctness_has_two_independent_repeats_per_arm(self):
        rows = list(evidence.schedule([{'id': name} for name in evidence.CASES]))
        self.assertEqual(len(rows), 12)
        self.assertEqual([r[2] for r in rows], list('ABBA' * 3))

    def test_measurement_has_four_balanced_blocks(self):
        rows = list(compiler.schedule())
        self.assertEqual(len(rows), 16)
        for role in compiler.ROLE_CASES:
            self.assertEqual([r[2] for r in rows if r[1] == role], list('ABBAABBA'))

    def test_same_bsdf_is_always_equivalence_gated(self):
        result = evidence.compare(self.readback('A'), self.readback('B', value=2.))
        self.assertTrue(result['equivalence_required'])
        self.assertFalse(result['qualified'])
        self.assertEqual(result['readbacks'][0]['failed_components'], 3)

    def test_equal_output_passes_despite_different_bsdf_names(self):
        result = evidence.compare(self.readback('A'), self.readback('B'))
        self.assertTrue(result['qualified'])
        self.assertTrue(all(r['byte_exact'] for r in result['readbacks']))

    def test_guide_below_hdr_tolerance_still_fails(self):
        result = evidence.compare(self.readback('A'), self.readback('B', guide=.500001))
        self.assertTrue(result['readbacks'][0]['strict_equivalent'])
        self.assertFalse(result['qualified'])

    def test_signed_zero_guide_requires_byte_equality(self):
        result = evidence.compare(self.readback('A', guide=0.), self.readback('B', guide=-0.))
        self.assertFalse(result['qualified'])
        self.assertEqual(result['readbacks'][1]['max_absolute_error'], 0.)

    def test_nonfinite_output_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evidence.compare(self.readback('A'), self.readback('B', value=float('nan')))

    def test_non_target_or_shared_hit_changes_are_rejected(self):
        a, b = policy(), policy()
        b['unrelated'] = []
        with self.assertRaisesRegex(ValueError, 'Non-target'):
            evidence.require_shared_policy(a, b)
        b = policy()
        b['target']['shaders']['any_hit']['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'shared hit'):
            evidence.require_shared_policy(a, b)

    def test_ser_state_or_extension_is_rejected(self):
        a, b = policy(), policy()
        a['target']['state']['enable_invocation_reorder'] = True
        b['target']['state']['enable_invocation_reorder'] = True
        with self.assertRaisesRegex(ValueError, 'SER must'):
            evidence.require_shared_policy(a, b)
        a, b = policy(), policy()
        b['target']['shaders']['ray_generation']['extensions'] = ['SPV_EXT_shader_invocation_reorder']
        with self.assertRaisesRegex(ValueError, 'SER extension'):
            evidence.require_shared_policy(a, b)

    def test_factory_contract_is_only_one_raygen_define(self):
        for role in compiler.ROLE_CASES:
            for arm in evidence.ARMS:
                compiler.validate_requests(role, arm, requests(role, arm))
        for stage in ('miss', 'closest_hit', 'any_hit'):
            self.assertEqual(requests('single', 'A')[stage], requests('single', 'B')[stage])
        wrong = requests('single', 'B')
        wrong['miss']['defines'].append(['METALLIC_SIMPLE_BSDF', '1'])
        with self.assertRaisesRegex(ValueError, 'Factory request'):
            compiler.validate_requests('single', 'B', wrong)

    def test_ser_capability_duplicate_define_and_wrong_role_are_rejected(self):
        for change in ('capability', 'duplicate', 'single'):
            wrong = requests('general', 'B')
            if change == 'capability':
                wrong['ray_generation']['capabilities'].append('spvShaderInvocationReorderEXT')
            elif change == 'duplicate':
                wrong['ray_generation']['defines'].append(['METALLIC_SIMPLE_BSDF', '1'])
            else:
                wrong['ray_generation']['defines'][4][1] = '1'
            with self.assertRaisesRegex(ValueError, 'Factory request'):
                compiler.validate_requests('general', 'B', wrong)

    def rows(self):
        return [dict(block=block, role=role, arm=arm, value_ms=(10. + block) * (1.1 if arm == 'B' else 1.))
                for block, role, arm in compiler.schedule()]

    def test_statistics_uses_paired_process_values(self):
        result = compiler.statistics_for(self.rows())
        self.assertAlmostEqual(result['single']['paired_median_percent'], 10.)
        self.assertEqual(len(result['single']['values_ms']['A']), 4)
        self.assertEqual(result['general']['medians_ms']['A'], 11.5)

    def test_missing_duplicate_or_nonfinite_samples_are_rejected(self):
        for change in ('missing', 'duplicate', 'nonfinite'):
            rows = self.rows()
            if change == 'missing':
                rows.pop()
            elif change == 'duplicate':
                rows[-1] = copy.deepcopy(rows[0])
            else:
                rows[0]['value_ms'] = float('nan')
            with self.assertRaises(ValueError):
                compiler.statistics_for(rows)

    def test_plan_rejects_wrong_experiment_before_running_verifier(self):
        path = self.directory / 'Plan.json'
        evidence.save(path, dict(schema_version=1, experiment='vkgltf_ser', roles={}))
        with patch.object(evidence, 'verify') as verify:
            with self.assertRaisesRegex(ValueError, 'Wrong experiment'):
                compiler.load_plan(path)
            verify.assert_not_called()

    def test_artifact_and_frozen_input_tampering_is_detected(self):
        payload = self.directory / 'payload.bin'
        payload.write_bytes(b'original')
        frozen = evidence.experiment.snapshots([payload])
        evidence.seal(self.directory)
        payload.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            evidence.verify_seal(self.directory)
        with self.assertRaises((ValueError, RuntimeError)):
            evidence.experiment.check_snapshots(frozen)

    def compile_package(self, mismatch=False):
        output = self.directory / 'compile'
        output.mkdir()
        executable = self.directory / 'fake-compiler.exe'
        executable.write_bytes(b'not executable; no process is launched')
        source = Path(compiler.__file__).resolve()
        identities, roles = {}, {}
        for role in compiler.ROLE_CASES:
            identities[role], roles[role] = {}, {}
            for arm in evidence.ARMS:
                folder = self.directory / f'{role}-{arm}'
                folder.mkdir()
                (folder / 'reference.spv').write_bytes(b'good')
                stages = {name: dict(module_name=evidence.shared.VK_MODULE,
                    source_entry_point='vkGltfRayPipelinePathTraceMain', spirv_file='reference.spv')
                    for name in evidence.traversal.RT_STAGES}
                identities[role][arm] = dict(path=str(folder / 'recipe.json'), recipe=dict(shaders=stages))
                roles[role][arm] = dict(stages=requests(role, arm))
        plan = dict(roles=roles)
        correctness = dict(verified=True, correctness_qualified=True)
        rows = []
        tick = 1
        for block, role, arm in compiler.schedule():
            directory = f'{block:02}-{role}-{arm}'
            stages = {}
            for name in evidence.traversal.RT_STAGES:
                stage_dir = output / directory / name
                stage_dir.mkdir(parents=True)
                emitted = stage_dir / 'shader.spv'
                unequal = mismatch and not rows and not stages
                emitted.write_bytes(b'bad!' if unequal else b'good')
                item = identities[role][arm]
                reference = Path(item['path']).parent / 'reference.spv'
                record = dict(command=compiler.compile_command(executable, item['recipe']['shaders'][name], roles[role][arm]['stages'][name], emitted),
                    started=tick, ended=tick + 1, wall_ms=1000., exitCode=0, timedOut=False,
                    reference_sha256=evidence.digest(reference), emitted_sha256=evidence.digest(emitted),
                    emitted_bytes=4, matches_runtime_spirv=not unequal)
                tick += 2
                evidence.save(stage_dir / 'Process.json', record)
                stages[name] = record
            rows.append(dict(block=block, role=role, arm=arm, directory=directory, stages=stages, value_ms=4000.))
        summary = dict(experiment='simple_bsdf_generic', arms=evidence.ARMS, orders=list(evidence.ORDER),
            mode='compile', status='bytecode_mismatch' if mismatch else 'passed', executable=str(executable),
            frozen=evidence.experiment.snapshots([source, executable]), plan=plan, identities=identities,
            correctness=correctness, rows=rows, statistics=compiler.statistics_for(rows), production_attribution_qualified=not mismatch)
        evidence.save(output / 'Plan.json', plan)
        evidence.save(output / 'Summary.json', summary)
        shutil.copy2(source, output / source.name)
        evidence.seal(output)
        return output, (plan, identities, correctness)

    def test_compile_verifier_keeps_byte_mismatch_diagnostic(self):
        output, loaded = self.compile_package(mismatch=True)
        with patch.object(compiler, 'load_plan', return_value=loaded):
            with self.assertRaisesRegex(ValueError, 'bytecode-mismatched'):
                compiler.verify(output)
            result = compiler.verify(output, allow_bytecode_mismatch=True)
        self.assertTrue(result['verified'])
        self.assertFalse(result['production_attribution_qualified'])
        self.assertTrue(result['correctness_qualified'])

    def test_compile_verifier_recomputes_payload_even_if_package_resealed(self):
        output, loaded = self.compile_package()
        with patch.object(compiler, 'load_plan', return_value=loaded):
            self.assertTrue(compiler.verify(output)['production_attribution_qualified'])
            payload = output / '00-single-A/ray_generation/shader.spv'
            payload.write_bytes(b'tamp')
            evidence.seal(output)
            with self.assertRaisesRegex(ValueError, 'bytecode changed'):
                compiler.verify(output)

    def test_compile_verifier_rechecks_runtime_frozen_input(self):
        output, loaded = self.compile_package()
        (self.directory / 'fake-compiler.exe').write_bytes(b'changed')
        with patch.object(compiler, 'load_plan', return_value=loaded):
            with self.assertRaises((ValueError, RuntimeError)):
                compiler.verify(output)


if __name__ == '__main__':
    unittest.main()
