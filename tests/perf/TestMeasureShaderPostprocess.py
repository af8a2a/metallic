import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Tools/Perf'))
import MeasureShaderPostprocess as measure
import RunComputePipelineExperiments as experiment


class ShaderPostprocessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.conditions = experiment.arms(['O1'], ['pipeline'], 'normal', ['none'])

    def evidence(self, report):
        report_path = self.root / 'report.json'
        experiment.save(report_path, report)
        summary = dict(status='passed', request_indices=[146], pairs=1, conditions=self.conditions,
            runs=[dict(request_index=146, block=0, arm=self.conditions[0], report=str(report_path),
                process=dict(exit_code=0, timed_out=False, native_crash=False))])
        experiment.save(self.root / 'summary.json', summary)
        experiment.seal(self.root)
        return summary

    def sample(self):
        return dict(status='passed', configuration=dict(compile_only=True, shader_cache='reuse'),
            samples=[dict(status='passed', shader_cache_hit=False,
                compilation_phases=dict(compiler_executed=True))])

    def complete_evidence(self, pairs=2):
        """Build synthetic v2 cache/report evidence without invoking a compiler."""
        source = self.root / 'EvidenceFixture.slang'
        source.write_text('module EvidenceFixture;\n', encoding='utf-8')
        raw = struct.pack('<5I', 0x07230203, 0x00010600, 0, 4, 0)
        conditions = experiment.arms(['O1'], ['pipeline'], 'normal', experiment.POSTPROCESSES)

        def report(directory, mode):
            directory.mkdir()
            cache = directory / 'shader-cache'
            (cache / 'spirv').mkdir(parents=True)
            dependency = str(source.resolve()).encode('utf-8')
            payload = struct.pack('<IIQ', len(dependency), 0,
                int(measure.frontend.fnv(source.read_bytes()), 16)) + dependency + raw
            header = struct.pack('<8sIIQIIQQ', b'MTLSPV01', 2, 48, 1, 1, 0,
                len(raw), int(measure.frontend.fnv(payload), 16))
            (cache / 'spirv/cache.spv').write_bytes(header + payload)
            sample = dict(status='passed', shader_cache_hit=False, pipeline_status='compile_only',
                request=dict(index=146, module='EvidenceFixture', entry='main', defines=[]),
                dependencies=[str(source.resolve())], compile_wall_ms=2., spirv_bytes=len(raw),
                spirv_fnv1a64=measure.frontend.fnv(raw), effective_descriptor_mode='mapped',
                effective_slang_optimization='O1', effective_spirv_postprocess=mode,
                resource_heap_stride=0, sampler_heap_stride=0, spirv_statistics=dict(instruction_count=0),
                compilation_phases=dict(compiler_executed=True, cache_hit=False,
                    code_generation_ms=1., postprocess_ms=0. if mode == 'none' else .1,
                    raw_spirv_bytes=len(raw), output_spirv_bytes=len(raw)))
            data = dict(status='passed', validation_errors=0, build_configuration='Release',
                slang_build='fixture SDK', shader_target=dict(id='vulkan', revision=1), samples=[sample],
                configuration=dict(compile_only=True, shader_cache='reuse', cache_directory=str(cache),
                    slang_optimization='O1', debug_mode='disabled', validation=False,
                    compute_backend='pipeline', disable_driver_optimization=False,
                    spirv_postprocess=mode, spirv_postprocess_tool_version='fixture ' + mode))
            filename = directory / 'report.json'
            experiment.save(filename, data)
            return filename, data, sample

        process = dict(exit_code=0, timed_out=False, native_crash=False)
        discovery, _, _ = report(self.root / 'discovery', 'none')
        summary = dict(schema_version=2, status='passed', request_indices=[146], pairs=pairs,
            conditions=conditions, inputs=experiment.snapshots([source]),
            excluded_runs=[dict(request_index=146, process=copy.deepcopy(process), report=str(discovery))], runs=[])
        for block in range(pairs):
            for arm in measure.order(conditions, block):
                filename, data, sample = report(self.root / f"block-{block}-{arm['postprocess']}", arm['postprocess'])
                spirv = filename.parent / 'Output.spv'
                spirv.write_bytes(raw)
                summary['runs'].append(dict(request_index=146, block=block, arm=arm,
                    report=str(filename), spirv=str(spirv), process=copy.deepcopy(process),
                    compile_wall_ms=sample['compile_wall_ms'], phases=copy.deepcopy(sample['compilation_phases']),
                    shader_identity=experiment.shader_identity(data, sample),
                    output=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), fnv1a64=measure.frontend.fnv(raw))))
        self.reseal(summary)
        return summary

    def reseal(self, summary):
        summary['aggregate'] = measure.summarize(summary['runs'])
        # Malformed numeric fixtures intentionally need NaN/Inf in both raw and
        # summary JSON, so verification rather than the test writer rejects them.
        (self.root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=True), encoding='utf-8')
        experiment.seal(self.root)

    def update_report(self, summary, index, mutate, *, refresh_identity=True):
        run = summary['runs'][index]
        path = Path(run['report'])
        report = experiment.read(path)
        mutate(report)
        sample = report['samples'][0]
        path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')
        run['compile_wall_ms'] = sample['compile_wall_ms']
        run['phases'] = copy.deepcopy(sample['compilation_phases'])
        if refresh_identity:
            run['shader_identity'] = experiment.shader_identity(report, sample)
        self.reseal(summary)

    def test_complete_v2_evidence_and_legacy_v1_remain_verifiable(self):
        summary = self.complete_evidence()
        verified = measure.verify(self.root)
        self.assertEqual(verified['status'], 'passed')
        self.assertEqual(len(verified['runs']), 6)
        summary.pop('schema_version')
        summary.pop('inputs')
        summary.pop('excluded_runs')
        self.reseal(summary)
        self.assertEqual(measure.verify(self.root)['status'], 'passed')

    def test_v2_requires_complete_successful_dependency_discovery(self):
        summary = self.complete_evidence()
        for mutation in (
                lambda data: data['excluded_runs'].clear(),
                lambda data: data['excluded_runs'].append(copy.deepcopy(data['excluded_runs'][0])),
                lambda data: data['excluded_runs'][0]['process'].update(exit_code=1),
                lambda data: data['excluded_runs'][0]['process'].update(timed_out=True),
                lambda data: data['excluded_runs'][0]['process'].update(native_crash=True)):
            broken = copy.deepcopy(summary)
            mutation(broken)
            self.reseal(broken)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'dependency discovery'):
                measure.verify(self.root)

    def test_v2_discovery_and_measured_dependencies_must_be_frozen(self):
        summary = self.complete_evidence()
        for filename in (summary['excluded_runs'][0]['report'], summary['runs'][0]['report']):
            path = Path(filename)
            original = path.read_bytes()
            report = experiment.read(path)
            report['samples'][0]['dependencies'].append(str(self.root / 'UnfrozenDependency.slang'))
            experiment.save(path, report)
            self.reseal(summary)
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, 'dependency was not frozen'):
                measure.verify(self.root)
            path.write_bytes(original)

    def test_policy_and_repeated_tool_identity_drift_are_rejected_even_for_equal_bytes(self):
        summary = self.complete_evidence()
        index = next(index for index, run in enumerate(summary['runs'])
                     if run['block'] == 1 and run['arm']['postprocess'] == 'slang-dead-branches')
        self.update_report(summary, index, lambda data: data['configuration'].update(
            spirv_postprocess_tool_version='different Slang/validator'))
        with self.assertRaisesRegex(ValueError, 'tool identity'):
            measure.verify(self.root)
        self.update_report(summary, index, lambda data: data['configuration'].update(
            spirv_postprocess='dead-branches'), refresh_identity=False)
        with self.assertRaisesRegex(ValueError, 'postprocess policy'):
            measure.verify(self.root)

    def test_workload_identity_is_fixed_across_modes_and_against_discovery(self):
        summary = self.complete_evidence(pairs=1)
        index = next(index for index, run in enumerate(summary['runs'])
                     if run['arm']['postprocess'] == 'dead-branches')
        original_summary = copy.deepcopy(summary)
        path = Path(summary['runs'][index]['report'])
        original_report = path.read_bytes()
        for mutate in (
                lambda data: data.update(slang_build='different SDK'),
                lambda data: data['shader_target'].update(revision=99),
                lambda data: data['samples'][0]['request'].update(defines=[['CHANGED_WORKLOAD', '1']]),
                lambda data: data['samples'][0].update(resource_heap_stride=64)):
            path.write_bytes(original_report)
            changed = copy.deepcopy(original_summary)
            self.update_report(changed, index, mutate)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, 'identity drifted'):
                measure.verify(self.root)
        path.write_bytes(original_report)
        discovery = Path(summary['excluded_runs'][0]['report'])
        report = experiment.read(discovery)
        report['slang_build'] = 'different discovery SDK'
        experiment.save(discovery, report)
        self.reseal(original_summary)
        with self.assertRaisesRegex(ValueError, 'identity drifted'):
            measure.verify(self.root)

    def test_nonfinite_negative_and_non_numeric_metrics_are_rejected(self):
        summary = self.complete_evidence(pairs=1)
        original_summary = copy.deepcopy(summary)
        path = Path(summary['runs'][0]['report'])
        original_report = path.read_bytes()
        for field in ('compile_wall_ms', 'code_generation_ms'):
            for value in (float('nan'), float('inf'), -1., True, 'unknown'):
                path.write_bytes(original_report)
                changed = copy.deepcopy(original_summary)
                def mutate(data):
                    sample = data['samples'][0]
                    if field == 'compile_wall_ms':
                        sample[field] = value
                    else:
                        sample['compilation_phases'][field] = value
                self.update_report(changed, 0, mutate)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'finite'):
                    measure.verify(self.root)

    def test_three_blocks_balance_every_mode_in_each_position(self):
        conditions = experiment.arms(['O1'], ['pipeline'], 'normal', ['none', 'dead-branches', 'slang-dead-branches'])
        sequences = [measure.order(conditions, block) for block in range(3)]
        for position in range(3):
            self.assertEqual({row[position]['postprocess'] for row in sequences},
                {'none', 'dead-branches', 'slang-dead-branches'})
        two = conditions[:2]
        self.assertEqual(measure.order(two, 0), two)
        self.assertEqual(measure.order(two, 1), list(reversed(two)))

    def test_cached_compile_does_not_qualify_as_source_cost(self):
        report = self.sample()
        report['samples'][0]['shader_cache_hit'] = True
        self.evidence(report)
        with self.assertRaisesRegex(ValueError, 'fresh source compilation'):
            measure.verify(self.root)

    def test_native_creation_does_not_qualify_as_frontend_cost(self):
        report = self.sample()
        report['configuration']['compile_only'] = False
        self.evidence(report)
        with self.assertRaisesRegex(ValueError, 'fresh source compilation'):
            measure.verify(self.root)

    def test_missing_or_duplicate_sample_is_rejected(self):
        summary = self.evidence(self.sample())
        for runs in ([], summary['runs'] + summary['runs']):
            broken = copy.deepcopy(summary)
            broken['runs'] = runs
            experiment.save(self.root / 'summary.json', broken)
            experiment.seal(self.root)
            with self.assertRaisesRegex(ValueError, 'Missing, duplicated or reordered'):
                measure.verify(self.root)

    def test_changed_inventory_is_rejected(self):
        self.evidence(self.sample())
        (self.root / 'unexpected.txt').write_text('unrecorded evidence')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            measure.verify(self.root)


if __name__ == '__main__':
    unittest.main()
