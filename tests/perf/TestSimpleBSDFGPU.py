"""CPU-only contracts for four-arm direct/generic glTF GPU evidence."""
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
import SimpleBSDFGPU as evidence


def graph(arm):
    return dict(nodes=[dict(name='PathTrace', type='ScenePathTracePass', properties=dict(
        traversalBackend='rayPipeline', cacheMode='off', samples=1, exportDenoiserGuides=True, **evidence.ARMS[arm]))])


def requests():
    return dict(roles={role: {arm: dict(stages={stage: evidence.expected_request(role, arm, stage)
        for stage in evidence.traversal.RT_STAGES}) for arm in evidence.ARMS} for role in ('single', 'general')})


def policy(arm):
    return dict(device={'device': 'same'}, unrelated=[{'shader': 'same'}], target=dict(cache_group='ScenePathTracePass',
        state={'enable_invocation_reorder': evidence.ARMS[arm]['shaderExecutionReorder']},
        shaders={stage: dict(sha256=arm if stage == 'ray_generation' else 'same',
            extensions=['SPV_EXT_shader_invocation_reorder'] if stage == 'ray_generation' and arm.endswith('1') else [])
            for stage in evidence.traversal.RT_STAGES}))


def spirv(extension=None):
    raw = struct.pack('<5I', 0x07230203, 0x00010600, 0, 1, 0)
    if extension:
        name = extension.encode() + b'\0'
        name += b'\0' * ((-len(name)) % 4)
        raw += struct.pack('<I', ((1 + len(name) // 4) << 16) | 10) + name
    return raw


class SimpleBSDFGPUTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'build/simple-bsdf-gpu-tests'
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
            resolvedGraph=graph(arm), captures=[dict(output='PathTrace.depth', frame=97, file='guide.bin', bytes=4, format=47)])
        path = directory / 'MaterialBaseline.json'
        evidence.save(path, dict(cases=[case]))
        return dict(report=str(path), arm=arm, recipes=policy(arm))

    def write_recipe(self, arm):
        directory = self.directory / arm
        directory.mkdir()
        shaders = {}
        for name, request in requests()['roles']['single'][arm]['stages'].items():
            raw = spirv('SPV_EXT_shader_invocation_reorder' if name == 'ray_generation' and arm.endswith('1') else None)
            (directory / f'{name}.spv').write_bytes(raw)
            shaders[name] = dict(spirv_file=f'{name}.spv', spirv_bytes=len(raw), input_spirv_hash=evidence.traversal.fnv1a(raw),
                module_name=request['module'], source_entry_point=request['entry'], entry_point='main', debug_mode='release',
                descriptor_heap_mode='mapped', slang_optimization='O1', source_identity_ambiguous=False,
                spirv_postprocess='none', spirv_postprocess_tool_version='', specialization_constants=[])
        recipe = dict(schema='metallic_pipeline_recipe', version=4 if arm.endswith('1') else 3, pipeline_type='ray_tracing',
            expected_pso_hash='0000000000000001', cache_group='ScenePathTracePass', shaders=shaders,
            state=dict(disable_optimization=False, uses_bindless_heap=True, bindless_user_push_data_size=12,
                max_recursion_depth=1, cluster_acceleration_structure=True, enable_invocation_reorder=arm.endswith('1')),
            source_device=dict(validation_enabled=True, synchronization_validation_enabled=True,
                enabled_features=dict(ray_tracing_pipeline=True, ray_tracing_invocation_reorder=True)))
        path = directory / 'recipe.recipe.json'
        evidence.save(path, recipe)
        return path, recipe

    def inspect_recipe(self, path, arm):
        reference = dict(roles={'single': {arm: dict(stages={stage: dict(sha256=evidence.digest(path.parent / f'{stage}.spv'))
            for stage in evidence.traversal.RT_STAGES})}})
        return evidence.recipe_evidence(path.parent, arm, True, {'id': 'Test'}, graph(arm), requests(), reference)

    def test_wrong_runtime_factory_payload_is_rejected_despite_matching_entry(self):
        path, _ = self.write_recipe('S0')
        reference = dict(roles={'single': {'S0': dict(stages={stage: dict(sha256='wrong')
            for stage in evidence.traversal.RT_STAGES})}})
        with self.assertRaisesRegex(ValueError, 'factory reference'):
            evidence.recipe_evidence(path.parent, 'S0', True, {'id': 'Test'}, graph('S0'), requests(), reference)

    def test_only_declared_arm_properties_are_normalized(self):
        for arm in evidence.ARMS:
            self.assertEqual(evidence.normalized_graph(graph('D0'), 'D0'), evidence.normalized_graph(graph(arm), arm))
        with self.assertRaisesRegex(ValueError, 'property changed'):
            evidence.normalized_graph(graph('S1'), 'D1')
        wrong = graph('S0')
        wrong['nodes'][0]['properties']['samples'] = 4
        self.assertNotEqual(evidence.normalized_graph(wrong, 'S0'), evidence.normalized_graph(graph('D0'), 'D0'))

    def test_williams_order_balances_positions_and_all_adjacent_pairs(self):
        rows = evidence.ORDER
        for position in range(4):
            self.assertEqual({r[position] for r in rows}, set(evidence.ARMS))
        pairs = [(a, b) for row in rows for a, b in zip(row, row[1:])]
        self.assertEqual(len(pairs), 12)
        self.assertEqual(set(pairs), {(a, b) for a in evidence.ARMS for b in evidence.ARMS if a != b})
        self.assertEqual(evidence.CORRECTNESS_ORDER[0], evidence.CORRECTNESS_ORDER[1][::-1])
        cases = [dict(id=c) for c in evidence.CASES]
        self.assertEqual(len(list(evidence.schedule('correctness', cases))), 24)
        self.assertEqual(len(list(evidence.schedule('performance', cases))), 48)
        self.assertEqual(len(list(evidence.schedule('correctness', cases[:1]))), 8)

    def test_all_four_contrasts_require_strict_equivalence(self):
        runs = {arm: self.readback(arm) for arm in evidence.ARMS}
        for left, right in evidence.CONTRASTS.values():
            result = evidence.compare(runs[left], runs[right])
            self.assertTrue(result['equivalence_required'])
            self.assertTrue(result['qualified'])
            self.assertTrue(all(r['byte_exact'] for r in result['readbacks']))

    def test_wrapper_failure_is_not_skipped_for_different_bsdf_strings(self):
        result = evidence.compare(self.readback('D0'), self.readback('S0', value=2.))
        self.assertFalse(result['qualified'])
        self.assertEqual(result['readbacks'][0]['failed_components'], 3)

    def test_tiny_guide_change_fails_without_relaxing_hdr_tolerance(self):
        result = evidence.compare(self.readback('D0'), self.readback('D1', guide=.500001))
        self.assertTrue(result['readbacks'][0]['strict_equivalent'])
        self.assertFalse(result['qualified'])

    def test_signed_zero_guide_requires_raw_byte_equality(self):
        result = evidence.compare(self.readback('D0', guide=0.), self.readback('D1', guide=-0.))
        self.assertFalse(result['qualified'])
        self.assertEqual(result['readbacks'][1]['max_absolute_error'], 0.)

    def test_nonfinite_hdr_is_runtime_invalid(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            evidence.compare(self.readback('D0'), self.readback('S0', value=float('nan')))

    def test_factory_requests_only_change_wrapper_macro_and_ser(self):
        all_requests = requests()
        evidence.validate_requests(all_requests)
        for role in all_requests['roles'].values():
            for stage in ('miss', 'closest_hit', 'any_hit'):
                self.assertEqual(len({evidence.traversal.canonical(role[a]['stages'][stage]) for a in evidence.ARMS}), 1)
            for direct, simple in (('D0', 'S0'), ('D1', 'S1')):
                a, b = [copy.deepcopy(role[arm]['stages']['ray_generation']) for arm in (direct, simple)]
                self.assertEqual(b['defines'].pop(), ['METALLIC_SIMPLE_BSDF', '1'])
                self.assertEqual(a, b)

    def test_wrong_capability_duplicate_define_and_changed_hit_request_rejected(self):
        for mutation in ('nv', 'duplicate', 'hit'):
            wrong = requests()
            stage = wrong['roles']['single']['S1']['stages']['ray_generation']
            if mutation == 'nv':
                stage['capabilities'][-1] = 'spvShaderInvocationReorderNV'
            elif mutation == 'duplicate':
                stage['defines'].append(['METALLIC_SIMPLE_BSDF', '1'])
            else:
                wrong['roles']['general']['S0']['stages']['miss']['defines'].append(['METALLIC_SIMPLE_BSDF', '1'])
            with self.assertRaisesRegex(ValueError, 'Factory request changed'):
                evidence.validate_requests(wrong)

    def test_actual_four_arm_recipes_require_ext_ser_and_feature(self):
        for arm in evidence.ARMS:
            path, recipe = self.write_recipe(arm)
            self.inspect_recipe(path, arm)
            if arm.endswith('1'):
                recipe['source_device']['enabled_features']['ray_tracing_invocation_reorder'] = False
                evidence.save(path, recipe)
                with self.assertRaisesRegex(ValueError, 'device feature'):
                    self.inspect_recipe(path, arm)

    def test_native_ser_state_cannot_be_mislabeled(self):
        path, recipe = self.write_recipe('S0')
        recipe['state']['enable_invocation_reorder'] = True
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'Native SER state'):
            self.inspect_recipe(path, 'S0')

    def test_rehashed_wrong_ser_payload_still_rejected(self):
        path, recipe = self.write_recipe('S1')
        raw = spirv()
        (path.parent / 'ray_generation.spv').write_bytes(raw)
        recipe['shaders']['ray_generation'].update(spirv_bytes=len(raw), input_spirv_hash=evidence.traversal.fnv1a(raw))
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'SPIR-V SER identity'):
            self.inspect_recipe(path, 'S1')

    def test_nv_reorder_is_rejected_even_on_off_arm(self):
        path, recipe = self.write_recipe('D0')
        raw = spirv('SPV_NV_shader_invocation_reorder')
        (path.parent / 'ray_generation.spv').write_bytes(raw)
        recipe['shaders']['ray_generation'].update(spirv_bytes=len(raw), input_spirv_hash=evidence.traversal.fnv1a(raw))
        evidence.save(path, recipe)
        with self.assertRaisesRegex(ValueError, 'Only requested EXT'):
            self.inspect_recipe(path, 'D0')

    def test_shared_hit_non_target_and_native_policy_must_match(self):
        for change in ('hit', 'unrelated', 'state', 'raygen'):
            a, b = policy('D0'), policy('S1')
            if change == 'hit':
                b['target']['shaders']['any_hit']['sha256'] = 'wrong'
            elif change == 'unrelated':
                b['unrelated'] = []
            elif change == 'state':
                b['target']['state']['disable_optimization'] = True
            else:
                b['target']['shaders']['ray_generation']['slang_optimization'] = 'O0'
            with self.assertRaises(ValueError):
                evidence.require_shared_policy(a, b)

    def synthetic_source(self):
        source = self.directory / 'fixtures'
        source.mkdir()
        config = dict(frames=16, warmupFrames=4, timingFrames=4, synchronizationValidation=True,
            assetFiles=['preserved'], cases=[dict(id=name, graph=f'G/{name}.json', scene='same-scene', width=64, height=64,
                environment={'path': 'same-env'}, captureFrames=[8, 9, 10], captureOutputs=evidence.GUIDES,
                cameraFrames=[{'frame': 11}], guideCameraStep={'delta': [.02, 0, 0]}, useRenderView=True,
                requiredTiming=['PathTrace', 'Path trace shading']) for name in evidence.CASES])
        evidence.save(source / 'correctness-G.json', config)
        return source, config

    def test_timing_derivation_keeps_scene_environment_and_seven_post_timing_guides(self):
        source, original = self.synthetic_source()
        config = evidence.derived_config(source, 'S1', 'performance')
        self.assertEqual((config['frames'], config['warmupFrames'], config['timingFrames']), (97, 32, 64))
        self.assertFalse(config['synchronizationValidation'])
        self.assertEqual(config['assetFiles'], original['assetFiles'])
        for before, after in zip(original['cases'], config['cases']):
            self.assertEqual((after['width'], after['height'], after['frameCount']), (512, 512, 97))
            self.assertEqual(after['captureOutputs'], evidence.GUIDES)
            self.assertTrue(all(field not in after for field in evidence.TEMPORAL_FIELDS))
            self.assertEqual(after['scene'], before['scene'])
            self.assertEqual(after['environment'], before['environment'])
        self.assertEqual(evidence.SAMPLES, dict(Opaque1=1, Mixed4=4, ABeautifulGame=2))

    def test_correctness_keeps_camera_temporal_guides_at_512(self):
        source, original = self.synthetic_source()
        config = evidence.derived_config(source, 'D0', 'correctness')
        for before, after in zip(original['cases'], config['cases']):
            self.assertEqual((after['width'], after['height']), (512, 512))
            for field in (*evidence.TEMPORAL_FIELDS, 'captureOutputs'):
                self.assertEqual(after[field], before[field])

    def timing_report(self):
        case = dict(id='Test', warmup=[dict(frame=i, executionId=i, warmup=True) for i in range(32)],
            frames=[dict(frame=i, executionId=i, warmup=False) for i in range(32, 96)])
        return dict(validation=False, synchronizationValidation=False, cases=[case]), dict(
            frames=97, warmupFrames=32, timingFrames=64, synchronizationValidation=False)

    def full_timing_report(self):
        report, config = self.timing_report()
        spec = dict(id='Test', output='PathTrace.color', requiredTiming=['PathTrace', 'Path trace shading'],
            width=512, height=512, frameCount=97, captureOutputs=evidence.GUIDES)
        config['cases'] = [spec]
        case = report['cases'][0]
        case.update(spec, validHDR=True, captures=[dict(output=o, frame=97+i) for i, o in enumerate(evidence.GUIDES)])
        for row in case['frames']:
            row.update(asyncComputeBranches=0, graphMs=3., nodes=[dict(name='PathTrace', queue=0, gpuMs=2.,
                sections=[dict(name='Path trace shading', gpuMs=1.)])])
        return report, config

    def test_raw_timing_has_exact_window_queue_scope_and_positive_timestamps(self):
        report, config = self.full_timing_report()
        self.assertEqual(evidence.gpu_timings(report, config)['Test']['pass_ms']['median'], 2.)
        for mutation in ('missing', 'warmup', 'queue', 'scope', 'zero', 'nan', 'guide-frame'):
            wrong = copy.deepcopy(report)
            case = wrong['cases'][0]
            if mutation == 'missing':
                case['frames'].pop()
            elif mutation == 'warmup':
                case['frames'][0]['warmup'] = True
            elif mutation == 'queue':
                case['frames'][0]['nodes'][0]['queue'] = 1
            elif mutation == 'scope':
                case['frames'][0]['nodes'][0]['sections'] = []
            elif mutation in ('zero', 'nan'):
                case['frames'][0]['graphMs'] = 0. if mutation == 'zero' else float('nan')
            else:
                case['captures'][0]['frame'] = 96
            with self.assertRaises(ValueError):
                evidence.gpu_timings(wrong, config)

    def test_warmup_missing_mislabeled_reordered_and_reused_ids_rejected(self):
        for change in ('missing', 'flag', 'order', 'id'):
            report, config = self.timing_report()
            warmup = report['cases'][0]['warmup']
            if change == 'missing':
                warmup.pop()
            elif change == 'flag':
                warmup[0]['warmup'] = False
            elif change == 'order':
                warmup.reverse()
            else:
                warmup[0]['executionId'] = 32
            with patch.object(evidence.experiment, 'gpu_timings') as timing:
                with self.assertRaises(ValueError):
                    evidence.gpu_timings(report, config)
                timing.assert_not_called()

    def test_validation_for_timing_is_forbidden(self):
        report, config = self.timing_report()
        report['validation'] = True
        with self.assertRaisesRegex(ValueError, 'no validation'):
            evidence.gpu_timings(report, config)
        command = evidence.expected_command(Path('fake.exe'), self.directory, 'performance')
        self.assertIn('--rhi-no-validation', command)
        self.assertNotIn('--rhi-validation', command)

    def analysis(self, failed=()):
        comparisons = [dict(case=case, contrast=label, qualified=label not in failed)
            for case in evidence.CASES for label in evidence.CONTRASTS]
        repeats = [dict(case=case, arm=arm, qualified=True) for case in evidence.CASES for arm in evidence.ARMS]
        per_case = evidence.qualifications(comparisons, repeats, evidence.CASES)
        return dict(case_contrast_qualified=per_case, correctness_qualified=not failed)

    def test_ser_failure_does_not_erase_wrapper_qualification(self):
        local = self.analysis()
        linked = self.analysis(('ser_direct', 'ser_simple'))
        result = evidence.qualification(local, linked, 'performance', True)
        self.assertFalse(result['correctness_qualified'])
        self.assertFalse(result['performance_qualified'])
        self.assertTrue(result['diagnostic_timing'])
        self.assertEqual(result['contrast_qualified'], dict(wrapper_ser_off=True, wrapper_ser_on=True, ser_direct=False, ser_simple=False))

    def test_same_arm_failure_disqualifies_its_contrasts(self):
        comparisons = [dict(case='Test', contrast=c, qualified=True) for c in evidence.CONTRASTS]
        repeats = [dict(case='Test', arm=a, qualified=a != 'S1') for a in evidence.ARMS]
        self.assertEqual(evidence.qualifications(comparisons, repeats, ['Test'])['Test'],
            dict(wrapper_ser_off=True, wrapper_ser_on=False, ser_direct=True, ser_simple=False))

    def test_performance_cannot_qualify_with_partial_coverage(self):
        result = evidence.qualification(self.analysis(), self.analysis(), 'performance', False)
        self.assertTrue(result['correctness_qualified'])
        self.assertFalse(result['performance_qualified'])

    def link(self):
        capture, prepared = self.directory / 'correctness', self.directory / 'prepared'
        capture.mkdir()
        prepared.mkdir()
        evidence.save(prepared / 'Manifest.json', {})
        inputs = dict(snapshots={'source': 'hash', 'renderer': 'binary'})
        evidence.save(capture / 'Inputs.json', inputs)
        summary = dict(experiment='simple_bsdf_gpu', stage='correctness', status='passed', full_stage_coverage=True,
            selected_cases=list(evidence.CASES), prepared_manifest_sha256=evidence.digest(prepared / 'Manifest.json'),
            analysis={'correctness_qualified': True}, capture_path=str(capture))
        return summary, inputs, prepared

    def test_link_requires_full_coverage_same_binary_and_explicit_failure_override(self):
        summary, inputs, prepared = self.link()
        evidence.check_correctness_link(summary, inputs, prepared, False)
        summary['analysis']['correctness_qualified'] = False
        summary['status'] = 'numeric_failure'
        with self.assertRaisesRegex(ValueError, 'explicit diagnostic'):
            evidence.check_correctness_link(summary, inputs, prepared, False)
        evidence.check_correctness_link(summary, inputs, prepared, True)
        summary['full_stage_coverage'] = False
        with self.assertRaisesRegex(ValueError, 'Complete sealed'):
            evidence.check_correctness_link(summary, inputs, prepared, True)
        summary['full_stage_coverage'] = True
        inputs['snapshots']['renderer'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identical frozen'):
            evidence.check_correctness_link(summary, inputs, prepared, True)

    def test_failed_runtime_is_never_override_eligible(self):
        summary, inputs, prepared = self.link()
        summary['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'Complete sealed'):
            evidence.check_correctness_link(summary, inputs, prepared, True)

    def test_cross_stage_actual_identity_must_match_beyond_validation(self):
        prior = dict(case='Opaque1', arm='S1', block=0, recipes=dict(stable=[{'sha256': 'same'}],
            device=dict(validation_enabled=True, synchronization_validation_enabled=True, driver='same')))
        evidence.save(self.directory / 'Summary.json', dict(analysis={'runs': [prior]}))
        current = copy.deepcopy(prior)
        current['recipes']['device'].update(validation_enabled=False, synchronization_validation_enabled=False)
        evidence.require_linked_identities({'runs': [current]}, {'path': str(self.directory)})
        current['recipes']['stable'][0]['sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            evidence.require_linked_identities({'runs': [current]}, {'path': str(self.directory)})

    def capture(self, failure=False):
        output, prepared = self.directory / 'capture', self.directory / 'prepared'
        output.mkdir()
        prepared.mkdir()
        source = Path(evidence.__file__).resolve()
        executable = self.directory / 'not-executable.exe'
        executable.write_bytes(b'CPU test only; never executed')
        shutil.copy2(source, output / source.name)
        evidence.save(output / 'Inputs.json', dict(snapshots=evidence.experiment.snapshots([source, executable])))
        evidence.save(prepared / 'Manifest.json', {})
        cases = [dict(id='Opaque1')]
        evidence.save(prepared / 'correctness-D0.json', {'cases': cases})
        env = {k: v for k, v in evidence.traversal.clean_environment()[0].items() if k.startswith(('METALLIC_', 'VK_'))}
        evidence.save(output / 'Environment.json', dict(fixed=env))
        rows = []
        for index, (case, block, arm) in enumerate(evidence.schedule('correctness', cases)):
            destination = output / case / f'block-{block:02}-{arm}'
            destination.mkdir(parents=True)
            process = dict(command=evidence.expected_command(executable, destination, 'correctness'), cwd=str(ROOT),
                timeout_seconds=1800, pid=index + 1, exit_code=0, timed_out=False,
                started_utc=f'2026-10-17T00:00:{index*2:02}+00:00', ended_utc=f'2026-10-17T00:00:{index*2+1:02}+00:00',
                environment=dict(env, METALLIC_MATERIAL_BASELINE_CASES=str(destination / 'Cases.json'),
                    METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / 'recipes'),
                    METALLIC_PSO_CACHE_DIRECTORY=str(output / 'application-pso-cache' / arm)))
            evidence.save(destination / 'process.json', process)
            rows.append(dict(case=case, block=block, arm=arm, process=process))
        analysis = dict(runs=rows, case_contrast_qualified={'Opaque1': {label: not (failure and label.startswith('ser_'))
            for label in evidence.CONTRASTS}})
        summary = dict(schema_version=1, experiment='simple_bsdf_gpu', status='numeric_failure' if failure else 'passed',
            stage='correctness', arms=evidence.ARMS, executable=str(executable), prepared=str(prepared),
            prepared_manifest_sha256=evidence.digest(prepared / 'Manifest.json'), selected_cases=['Opaque1'],
            full_stage_coverage=False, schedule=list(evidence.schedule('correctness', cases)), process_records=rows,
            correctness=None, allow_unqualified_correctness=False, absolute_tolerance=evidence.ABSOLUTE,
            relative_tolerance=evidence.RELATIVE, guide_comparison='raw-byte-exact', analysis=analysis,
            qualification=evidence.qualification(analysis, None, 'correctness', False))
        evidence.save(output / 'Summary.json', summary)
        evidence.seal(output)
        return output, summary, analysis

    def test_verifier_numeric_failure_remains_diagnostic_and_strict_by_default(self):
        output, summary, analysis = self.capture(failure=True)
        with patch.object(evidence, 'verify_prepared'), patch.object(evidence, 'analyze', return_value=analysis):
            with self.assertRaisesRegex(ValueError, 'numerically unqualified'):
                evidence.verify(output)
            result = evidence.verify(output, allow_numeric_failure=True)
            self.assertTrue(result['verified'])
            self.assertFalse(result['correctness_qualified'])
            self.assertTrue(result['contrast_qualified']['wrapper_ser_on'])
            summary['qualification']['correctness_qualified'] = True
            evidence.save(output / 'Summary.json', summary)
            evidence.seal(output)
            with self.assertRaisesRegex(ValueError, 'Qualification contradicts'):
                evidence.verify(output, allow_numeric_failure=True)

    def test_verifier_rejects_resealed_process_overlap_and_command_change(self):
        output, summary, analysis = self.capture()
        with patch.object(evidence, 'verify_prepared'), patch.object(evidence, 'analyze', return_value=analysis):
            self.assertTrue(evidence.verify(output)['verified'])
            row = summary['process_records'][1]
            row['process']['started_utc'] = summary['process_records'][0]['process']['started_utc']
            destination = output / row['case'] / f'block-{row["block"]:02}-{row["arm"]}'
            evidence.save(destination / 'process.json', row['process'])
            evidence.save(output / 'Summary.json', summary)
            evidence.seal(output)
            with self.assertRaisesRegex(ValueError, 'Overlapping'):
                evidence.verify(output)
            row['process']['command'][2] = '--rhi-no-validation'
            evidence.save(destination / 'process.json', row['process'])
            evidence.save(output / 'Summary.json', summary)
            evidence.seal(output)
            with self.assertRaisesRegex(ValueError, 'Process record/command'):
                evidence.verify(output)

    def rows(self):
        return [dict(case=case, block=block, arm=arm, timings={case: {metric: dict(median=(10. + block) * factor)
            for metric in evidence.METRICS}}) for case in evidence.CASES for block in range(4)
            for arm, factor in [('D0', 1), ('S0', 1.1), ('D1', 2), ('S1', 2.1)]]

    def test_statistics_use_four_process_medians_with_paired_dispersion(self):
        result = evidence.statistics_for(self.rows(), evidence.CASES)['Mixed4']['pass_ms']
        self.assertEqual(result['medians_ms']['D0'], 11.5)
        self.assertEqual(len(result['values_ms']['S1']), 4)
        self.assertAlmostEqual(result['contrasts']['wrapper_ser_off']['paired_median_percent'], 10.)
        self.assertAlmostEqual(result['contrasts']['wrapper_ser_on']['paired_median_percent'], 5.)
        self.assertAlmostEqual(result['contrasts']['ser_direct']['paired_median_percent'], 100.)

    def test_missing_duplicate_nonfinite_process_samples_rejected(self):
        for mutation in ('missing', 'duplicate', 'nan'):
            rows = self.rows()
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'duplicate':
                rows[-1] = copy.deepcopy(rows[0])
            else:
                rows[0]['timings']['Opaque1']['pass_ms']['median'] = float('nan')
            with self.assertRaises(ValueError):
                evidence.statistics_for(rows, evidence.CASES)

    def test_changed_sealed_payload_or_frozen_input_is_rejected(self):
        payload = self.directory / 'payload.bin'
        payload.write_bytes(b'original')
        frozen = evidence.experiment.snapshots([payload])
        evidence.seal(self.directory)
        payload.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            evidence.verify_seal(self.directory)
        with self.assertRaises((ValueError, RuntimeError)):
            evidence.experiment.check_snapshots(frozen)


if __name__ == '__main__':
    unittest.main()
