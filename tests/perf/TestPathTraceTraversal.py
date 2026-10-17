"""CPU-only rejection tests for the traversal A/B evidence boundary."""
from pathlib import Path
import copy
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools/Perf'))
import PathTraceTraversal as traversal
import PathTraceTraversalPSO as native


def graph(arm='A'):
    return dict(version=1, name='Case', nodes=[dict(id=1, name='PathTrace', type='ScenePathTracePass',
        properties=dict(bsdf='openpbr', cacheMode='off', samples=1, exportDenoiserGuides=True,
            traversalBackend=traversal.ARMS[arm], camera={'eye': [0, 0, 1]}))], edges=[], outputs=['PathTrace.color'])


def spec():
    return dict(id='Case', graph='Graph.json', width=1, height=1, scene=None, output='PathTrace.color',
        requiredTiming=['PathTrace', 'Path trace shading'], requireEnvironment=False,
        captureFrames=[1], captureOutputs=['PathTrace.normalRoughness'])


class TraversalEvidenceTests(unittest.TestCase):
    def test_native_recipe_hex_hash_matches_numeric_report(self):
        self.assertEqual(native.expected_pso_hash({'expected_pso_hash': '11c6647a4a683848'}), 1280821620430616648)
        self.assertEqual(native.expected_pso_hash({'expected_pso_hash': '93a06744d878dddf'}), 10637615865236348383)

    def test_native_recipe_rejects_noncanonical_hash(self):
        for value in [1280821620430616648, '0x11c6647a4a683848', '11c6647a4a68384g', '11c6647a4a68384']:
            with self.assertRaisesRegex(ValueError, 'Invalid recipe PSO hash'):
                native.expected_pso_hash({'expected_pso_hash': value})

    def setUp(self):
        self.test_root = (ROOT / 'build/pathtrace-traversal-tests').resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.test_root)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.test_root))
        self.temporary.cleanup()

    def test_only_backend_graph_difference_allowed(self):
        a, b = graph('A'), graph('B')
        self.assertEqual(traversal.normalized_graph(a, 'A'), traversal.normalized_graph(b, 'B'))
        b['nodes'][0]['properties']['samples'] = 2
        self.assertNotEqual(traversal.normalized_graph(a, 'A'), traversal.normalized_graph(b, 'B'))
        with self.assertRaisesRegex(ValueError, 'traversal backend'):
            traversal.normalized_graph(a, 'B')

    def test_resolved_graph_camera_contract_is_explicit(self):
        source = graph()
        resolved = traversal.expected_resolved_graph(source, {'useRenderView': True})
        self.assertNotIn('camera', resolved['nodes'][0]['properties'])
        self.assertIn('camera', source['nodes'][0]['properties'])
        self.assertEqual(resolved['nodes'][0]['position'], {'x': 0.0, 'y': 0.0})
        self.assertIn('camera', traversal.expected_resolved_graph(source, {})['nodes'][0]['properties'])

    def write_rt_recipe(self):
        directory = self.directory / 'recipes'
        directory.mkdir()
        # Minimal byte framing is sufficient for CPU hash/schema tests. These
        # bytes are never submitted to a shader compiler or GPU driver.
        raw = struct.pack('<5I', 0x07230203, 0x00010600, 0, 1, 0)
        payload = directory / 'payload.spv'
        payload.write_bytes(raw)
        shaders = {}
        for stage, entry in [('ray_generation', 'openPbrRayPipelinePathTraceGuidesMain'),
                             ('miss', 'openPbrRayPipelineMiss'), ('closest_hit', 'openPbrRayPipelineClosestHit'),
                             ('any_hit', 'openPbrRayPipelineAnyHit')]:
            shaders[stage] = dict(spirv_file=payload.name, spirv_bytes=len(raw),
                input_spirv_hash=traversal.fnv1a(raw), module_name='Features/PathTracing/' +
                ('OpenPBRRayPipelinePathTraceGuides' if stage == 'ray_generation' else 'OpenPBRRayPipelineHit'),
                source_entry_point=entry, entry_point='main', debug_mode='release', descriptor_heap_mode='mapped',
                slang_optimization='O1', source_identity_ambiguous=False, spirv_postprocess='none',
                spirv_postprocess_tool_version='', specialization_constants=
                    [dict(id=0, type='bool32', size=4, bits=0)] if stage == 'ray_generation' else [])
        recipe = dict(schema='metallic_pipeline_recipe', version=3, pipeline_type='ray_tracing',
            expected_pso_hash='0000000000000001', cache_group='ScenePathTracePass',
            source_device=dict(validation_enabled=True, synchronization_validation_enabled=True,
                enabled_features=dict(ray_tracing_pipeline=True)), shaders=shaders,
            state=dict(disable_optimization=False, uses_bindless_heap=True, bindless_user_push_data_size=256,
                max_recursion_depth=1, cluster_acceleration_structure=True))
        path = directory / 'target.recipe.json'
        traversal.save(path, recipe)
        return path, recipe, payload

    def inspect_recipe(self, directory):
        return traversal.recipe_evidence(directory, 'B', True, spec(), graph('B'))

    def test_complete_rt_recipe_is_retained_and_hashed(self):
        path, _, payload = self.write_rt_recipe()
        evidence = self.inspect_recipe(path.parent)
        self.assertEqual(set(evidence['target']['shaders']), set(traversal.RT_STAGES))
        self.assertEqual(evidence['records'][0]['payloads']['any_hit']['sha256'], traversal.digest(payload))

    def test_missing_rt_stage_is_rejected(self):
        path, recipe, _ = self.write_rt_recipe()
        del recipe['shaders']['any_hit']
        traversal.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'complete CLAS'):
            self.inspect_recipe(path.parent)

    def test_wrong_payload_hash_is_rejected(self):
        path, _, payload = self.write_rt_recipe()
        raw = bytearray(payload.read_bytes())
        raw[-1] ^= 1
        payload.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'FNV differs'):
            self.inspect_recipe(path.parent)

    def test_native_specialization_type_is_not_normalized_away(self):
        path, recipe, _ = self.write_rt_recipe()
        recipe['shaders']['ray_generation']['specialization_constants'][0]['type'] = 'int32'
        traversal.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'cache/guide tuple'):
            self.inspect_recipe(path.parent)

    def test_missing_sync_validation_is_rejected(self):
        path, recipe, _ = self.write_rt_recipe()
        recipe['source_device']['synchronization_validation_enabled'] = False
        traversal.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'synchronization-validation'):
            self.inspect_recipe(path.parent)

    def test_same_arm_target_or_non_target_change_is_rejected(self):
        previous = {'recipes': {'stable': [{'target': 'payload-a'}, {'non_target': 'payload-b'}]}}
        traversal.require_stable_recipes(previous, copy.deepcopy(previous))
        for index in (0, 1):
            changed = copy.deepcopy(previous)
            changed['recipes']['stable'][index]['injected_state_change'] = True
            with self.assertRaisesRegex(ValueError, 'Same-arm'):
                traversal.require_stable_recipes(previous, changed)

    def write_readbacks(self):
        destination = self.directory / 'run'
        destination.mkdir()
        requested = spec()
        configuration = dict(version=2, frames=3, warmupFrames=0, timingFrames=1,
            synchronizationValidation=True, cases=[requested])
        traversal.save(self.directory / 'Graph.json', graph())
        (destination / 'Tests.xml').write_text('<testsuites tests="1" failures="0" errors="0">'
            '<testsuite><testcase classname="RHIRendering" name="material_phase0_baseline"/>'
            '</testsuite></testsuites>')
        traversal.save(destination / 'process.json', {'exit_code': 0, 'timed_out': False})
        (destination / 'stdout.log').write_text('')
        (destination / 'stderr.log').write_text('')
        case = dict(requested, validHDR=True, nonfiniteComponents=0, format='RGBA32F', image='final.bin',
            resolvedGraph=traversal.expected_resolved_graph(graph(), requested), frames=[],
            captures=[dict(output='PathTrace.color', frame=1, file='color.bin', bytes=16, format=42),
                      dict(output='PathTrace.normalRoughness', frame=3, file='guide.bin', bytes=16, format=42)])
        for filename in ('final.bin', 'color.bin', 'guide.bin'):
            (destination / filename).write_bytes(struct.pack('<4f', 0.25, 0.5, 0.75, 1))
        report = {'validation': True, 'synchronizationValidation': True, 'cases': [case]}
        traversal.save(destination / 'MaterialBaseline.json', report)
        return destination, configuration, report

    def inspect_readbacks(self, destination, configuration):
        with patch.object(traversal, 'recipe_evidence', return_value={}), \
             patch.object(traversal.experiment, 'gpu_timings', return_value={}):
            return traversal.inspect_run(destination, configuration, 'A', True, self.directory)

    def test_guide_missing_or_capture_sequence_changed_is_rejected(self):
        destination, configuration, report = self.write_readbacks()
        self.inspect_readbacks(destination, configuration)
        original = copy.deepcopy(report)
        for change in ('missing', 'reordered', 'wrong_frame'):
            report = copy.deepcopy(original)
            captures = report['cases'][0]['captures']
            if change == 'missing':
                captures.pop()
            elif change == 'reordered':
                captures.reverse()
            else:
                captures[-1]['frame'] += 1
            traversal.save(destination / 'MaterialBaseline.json', report)
            with self.assertRaisesRegex(ValueError, 'Missing, duplicated or reordered'):
                self.inspect_readbacks(destination, configuration)

    def test_nonfinite_guide_cannot_pass_finite_hdr_metadata(self):
        destination, configuration, _ = self.write_readbacks()
        (destination / 'guide.bin').write_bytes(struct.pack('<4f', float('nan'), 0.5, 0.75, 1))
        with self.assertRaisesRegex(ValueError, 'Nonfinite readback'):
            self.inspect_readbacks(destination, configuration)

    def write_numeric_failure(self):
        prepared = self.directory / 'prepared'
        output = self.directory / 'evidence'
        prepared.mkdir()
        output.mkdir()
        traversal.save(prepared / 'correctness-A.json', {'cases': [{'id': 'Case'}]})
        traversal.seal(prepared)
        schedule = [('Case', 0, arm) for arm in 'AB']
        analysis = dict(runs=[{'case': 'Case', 'arm': arm} for arm in 'AB'], correctness_qualified=False)
        summary = dict(status='failed', stage='correctness', prepared=str(prepared),
            prepared_manifest_sha256=traversal.digest(prepared / 'Manifest.json'),
            error='ValueError: Strict HDR/guide/lighting gate failed; raw data retained; thresholds unchanged',
            absolute_tolerance=traversal.ABSOLUTE, relative_tolerance=traversal.RELATIVE,
            guide_comparison='raw-byte-exact', selected_cases=['Case'], full_stage_coverage=True,
            schedule=[list(item) for item in schedule],
            process_records=[dict(case=name, block=block, arm=arm) for name, block, arm in schedule], analysis=analysis)
        traversal.save(output / 'Summary.json', summary)
        traversal.seal(output)
        return output, summary

    def verify_boundary(self, output, summary, **kwargs):
        # Mock only the expensive detailed scene/recipe analysis. The new status
        # boundary, sealed files, manifest identity, tolerance and schedule are real.
        with patch.object(traversal, 'verify_prepared'), \
             patch.object(traversal, 'analyze', return_value=summary['analysis']) as analyze:
            result = traversal.verify(output, **kwargs)
            analyze.assert_called_once_with(output, Path(summary['prepared']), 'correctness', ['Case'], summary['analysis'])
            return result

    def test_numeric_failure_is_opt_in_and_stays_unqualified(self):
        output, summary = self.write_numeric_failure()
        with self.assertRaisesRegex(ValueError, 'Failed/incomplete'):
            self.verify_boundary(output, summary)
        result = self.verify_boundary(output, summary, allow_numeric_failure=True)
        self.assertTrue(result['verified'])
        self.assertTrue(result['numeric_failure'])
        self.assertFalse(result['correctness_qualified'])

    def test_numeric_failure_flag_rejects_other_failures_and_incomplete_analysis(self):
        output, original = self.write_numeric_failure()
        mutations = [dict(error='RuntimeError: Child failed'), dict(error=original['error'] + ' '),
                     dict(status='running'), dict(analysis={}), dict(analysis=None)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                summary = dict(original, **mutation)
                traversal.save(output / 'Summary.json', summary)
                traversal.seal(output)
                with self.assertRaisesRegex(ValueError, 'Failed/incomplete'):
                    self.verify_boundary(output, summary, allow_numeric_failure=True)

    def test_numeric_failure_flag_preserves_integrity_and_schedule_checks(self):
        output, original = self.write_numeric_failure()
        mutations = [(dict(prepared_manifest_sha256='changed'), 'Prepared manifest'),
                     (dict(absolute_tolerance=1), 'Fixed comparison'),
                     (dict(schedule=original['schedule'][:-1]), 'Recorded schedule'),
                     (dict(process_records=original['process_records'][:-1]), 'Process schedule')]
        for mutation, message in mutations:
            with self.subTest(mutation=mutation):
                summary = dict(original, **mutation)
                traversal.save(output / 'Summary.json', summary)
                traversal.seal(output)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify_boundary(output, summary, allow_numeric_failure=True)
        (output / 'extra-unsealed.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Evidence changed'):
            self.verify_boundary(output, original, allow_numeric_failure=True)

    def test_numeric_failure_still_recomputes_and_rejects_changed_analysis(self):
        output, summary = self.write_numeric_failure()
        with patch.object(traversal, 'verify_prepared'), \
             patch.object(traversal, 'analyze', side_effect=ValueError('Recomputed evidence differs')):
            with self.assertRaisesRegex(ValueError, 'Recomputed evidence differs'):
                traversal.verify(output, allow_numeric_failure=True)
        with patch.object(traversal, 'verify_prepared'), \
             patch.object(traversal, 'analyze', return_value={'correctness_qualified': True}):
            with self.assertRaisesRegex(ValueError, 'qualification contradicts'):
                traversal.verify(output, allow_numeric_failure=True)

    def test_passed_evidence_keeps_strict_qualification(self):
        output, summary = self.write_numeric_failure()
        summary.update(status='passed', analysis=dict(runs=[], correctness_qualified=True))
        del summary['error']
        traversal.save(output / 'Summary.json', summary)
        traversal.seal(output)
        for allow in (False, True):
            result = self.verify_boundary(output, summary, allow_numeric_failure=allow)
            self.assertFalse(result['numeric_failure'])
            self.assertTrue(result['correctness_qualified'])

    def test_verify_cli_forwards_numeric_failure_opt_in_only(self):
        for flag, expected in (([], False), (['--allow-numeric-failure'], True)):
            with patch.object(sys, 'argv', ['PathTraceTraversal.py', 'verify', 'evidence', *flag]), \
                 patch.object(traversal, 'verify', return_value={}) as verify, contextlib.redirect_stdout(io.StringIO()):
                traversal.main()
                verify.assert_called_once_with(Path('evidence'), allow_numeric_failure=expected)


if __name__ == '__main__':
    unittest.main()
