"""Check backend overlay contracts and reject incomplete compiler evidence."""

import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools/Perf'))
import PathTraceBackendOverlay as overlay
import ValidatePathTraceBackend as experiment


CODE = struct.pack('<5I', 0x07230203, 0x00010600, 0, 4, 0)


def save(path, value):
    # Malformed metric fixtures deliberately include JSON NaN/Infinity.
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def module_bytes(payload=b'fixture'):
    return b'RIFF' + struct.pack('<I', len(payload) + 4) + b'SLmc' + payload


def reference_report():
    return dict(status='passed', validation_errors=0,
        configuration=dict(compile_only=True, shader_cache='reuse',
            slang_optimization='O1', descriptor_mode='mapped', spirv_postprocess='none',
            debug_mode='disabled', validation=False, selected_requests=[146]), environment={},
        samples=[dict(status='passed', shader_cache_hit=False, pipeline_status='compile_only',
            effective_slang_optimization='O1', effective_descriptor_mode='mapped',
            effective_spirv_postprocess='none', compilation_phases=dict(compiler_executed=True),
            request=dict(index=146, module='Features/PathTracing/OpenPBRRayQueryPathTraceGuides',
                entry='openPbrRayQueryPathTraceGuidesMain', profile='spirv_1_6',
                search_path=str(ROOT / 'Shaders'), additional_search_paths=[],
                capabilities=['spvRayQueryKHR'], defines=[['METALLIC_CUSTOM_MATERIALS', '0'],
                    ['SHARC_UPDATE', '0'], ['SHARC_QUERY', '0'], ['LITERAL', 'a b;$()']]),
            spirv_bytes=len(CODE), spirv_fnv1a64=experiment.profile.fnv(CODE), dependencies=[])])


class BackendReferenceTests(unittest.TestCase):
    def test_current_scope_accepts_only_the_fixed_fresh_reference(self):
        report = reference_report()
        self.assertIs(experiment.reference_sample(report, 146), report['samples'][0])

    def test_cache_native_or_requested_effective_policy_drift_is_rejected(self):
        mutations = (
            lambda r: r.update(status='failed'),
            lambda r: r['samples'].append(copy.deepcopy(r['samples'][0])),
            lambda r: r['configuration'].update(compile_only=False),
            lambda r: r['configuration'].update(shader_cache='disabled'),
            lambda r: r['samples'][0].update(shader_cache_hit=True),
            lambda r: r['samples'][0]['compilation_phases'].update(compiler_executed=False),
            lambda r: r['samples'][0].update(pipeline_status='created'),
            lambda r: r['configuration'].update(slang_optimization='O2'),
            lambda r: r['configuration'].update(descriptor_mode='native'),
            lambda r: r['configuration'].update(spirv_postprocess='slang-dead-branches'),
            lambda r: r['samples'][0].update(effective_slang_optimization='O0'),
            lambda r: r['samples'][0].update(effective_descriptor_mode='native'),
            lambda r: r['samples'][0].update(effective_spirv_postprocess='dead-branches'),
            lambda r: r['configuration'].update(debug_mode='capture'),
            lambda r: r['configuration'].update(validation=True),
            lambda r: r['samples'][0]['request'].update(index=149),
            lambda r: r['samples'][0]['request'].update(entry='otherMain'),
            lambda r: r['samples'][0]['request'].update(module='Features/ReSTIR/SceneRTXDI'),
        )
        for mutation in mutations:
            report = reference_report()
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                experiment.reference_sample(report, 146)

    def test_custom_material_and_sharc_requests_are_outside_initial_scope(self):
        for name in ('METALLIC_CUSTOM_MATERIALS', 'SHARC_UPDATE', 'SHARC_QUERY'):
            report = reference_report()
            report['samples'][0]['request']['defines'] = [[name, '1'], ['METALLIC_CUSTOM_MATERIALS', '0']] \
                if name != 'METALLIC_CUSTOM_MATERIALS' else [[name, '1']]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'SHARC|CUSTOM_MATERIALS'):
                experiment.reference_sample(report, 146)


class BackendOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.shader_root = Path(self.temporary.name).resolve() / 'Shaders'
        for relative in overlay.SOURCE_FILES:
            destination = self.shader_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'Shaders' / relative, destination)
        self.before = {relative: (self.shader_root / relative).read_bytes()
            for relative in overlay.SOURCE_FILES}
        self.plans = {arm: overlay.plan_overlay(self.shader_root, arm) for arm in overlay.ARMS}

    def test_five_plans_are_pure_and_original_is_unchanged(self):
        self.assertEqual(tuple(self.plans), tuple(label for label, _, _ in experiment.ARMS))
        original = self.plans['original-general']
        self.assertEqual(original['files'], {})
        self.assertIsNone(original['provider_file'])
        for plan in self.plans.values():
            self.assertEqual(plan['source_hashes'], {relative: hashlib.sha256(raw).hexdigest()
                for relative, raw in self.before.items()})
            self.assertEqual(plan['generated_hashes'], {relative: hashlib.sha256(text.encode('utf-8')).hexdigest()
                for relative, text in plan['files'].items()})
            self.assertEqual(plan['extra_compiler_flags'], [])
            self.assertTrue(plan['preserves_report_macros'])
            self.assertTrue(plan['preserves_resource_layout'])
            self.assertTrue(plan['retains_conditional_slab'])
        self.assertEqual(self.before, {relative: (self.shader_root / relative).read_bytes()
            for relative in overlay.SOURCE_FILES})
        self.assertEqual(len(list(self.shader_root.rglob('*.slang'))), len(overlay.SOURCE_FILES))

    def test_aliases_normalize_and_unknown_arm_fails(self):
        for alias, canonical in overlay.ARM_ALIASES.items():
            self.assertEqual(overlay.plan_overlay(self.shader_root, alias), self.plans[canonical])
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            overlay.plan_overlay(self.shader_root, 'backend-link-no-metallic')

    def test_concrete_link_controls_match_and_link_consumers_are_identical(self):
        for feature in ('general', 'no-fiber'):
            concrete = self.plans['backend-concrete-' + feature]
            linked = self.plans['backend-link-' + feature]
            self.assertEqual(concrete['feature_mask'], experiment.legacy.feature_values(feature))
            self.assertEqual(concrete['feature_mask'], linked['feature_mask'])
            self.assertEqual(concrete['transformation'], linked['transformation'])
        general, restricted = (self.plans['backend-link-' + feature] for feature in ('general', 'no-fiber'))
        consumers = lambda plan: {path: text for path, text in plan['files'].items()
            if path != plan['provider_file']}
        self.assertEqual(consumers(general), consumers(restricted))
        self.assertNotEqual(general['provider_source'], restricted['provider_source'])
        self.assertIn('= GeneralScenePathTraceMaterialBackend;', general['provider_source'])
        self.assertIn('= NoFiberScenePathTraceMaterialBackend;', restricted['provider_source'])

    def test_generated_sources_have_owned_modules_and_one_selected_provider(self):
        for arm, plan in self.plans.items():
            for relative, text in plan['files'].items():
                with self.subTest(arm=arm, source=relative):
                    path = Path(relative)
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn('..', path.parts)
                    owners = re.findall(r'^\s*(?:module|implementing)\s+[\w.]+\s*;', text, re.M)
                    self.assertEqual(len(owners), 1)
                    self.assertNotRegex(text, r'#\s*include\s*["<][^">]+\.slang[">]')
            if plan['mechanism'] == 'original':
                continue
            owner = plan['files'][overlay.OWNER_SOURCE]
            self.assertIn('__include "OpenPBRPathTracing/' + Path(overlay.BACKEND_SOURCE).name + '";', owner)
            self.assertIn('implementing OpenPBRPathTracing;', plan['files'][overlay.BACKEND_SOURCE])
            if plan['provider_file']:
                provider = plan['files'][plan['provider_file']]
                self.assertEqual(provider, plan['provider_source'])
                self.assertIn('module ' + plan['provider_module'] + ';', provider)
                self.assertEqual(provider.count('export struct ScenePathTraceMaterialBackend'), 1)
                self.assertLess(provider.index('import OpenPBRPathTracing;'),
                    provider.index('namespace Metallic.OpenPBRPathTracing'))
                self.assertIn('extern struct ScenePathTraceMaterialBackend', plan['files'][overlay.BACKEND_SOURCE])
            else:
                self.assertIn('typealias ScenePathTraceMaterialBackend =', plan['files'][overlay.BACKEND_SOURCE])

    def test_common_vertex_prefix_rng_slab_and_general_bodies_are_preserved(self):
        vertex = self.before[overlay.VERTEX_SOURCE].decode('utf-8').replace('\r\n', '\n')
        surface = self.before[overlay.SURFACE_SOURCE].decode('utf-8').replace('\r\n', '\n')
        _, _, _, body = overlay.function_parts(vertex, overlay.VERTEX_SIGNATURE)
        boundary = body.index(overlay.TRANSPORT_LOAD)
        prefix, general = body[:boundary], body[boundary:]
        fiber_start, fiber_end = general.index(overlay.FIBER_BRANCH), general.index(overlay.COVERAGE_BRANCH)
        no_fiber = general[:fiber_start] + general[fiber_end:]
        _, _, _, guide_body = overlay.function_parts(surface, overlay.GUIDE_SIGNATURE)
        guide_start = guide_body.index(overlay.FIBER_BRANCH)
        guide_end = guide_body.index('    let context = sceneSurfaceMaterialContextForRay(')
        guide_no_fiber = guide_body[:guide_start] + guide_body[guide_end:]
        for arm, plan in self.plans.items():
            if plan['mechanism'] == 'original':
                continue
            with self.subTest(arm=arm):
                changed = plan['files'][overlay.VERTEX_SOURCE]
                _, _, _, retained = overlay.function_parts(changed, overlay.VERTEX_SIGNATURE)
                self.assertTrue(retained.startswith(prefix))
                for name, expected in (('shadeOpenPBRFoundMaterialGeneral', general),
                        ('shadeOpenPBRFoundMaterialNoFiber', no_fiber)):
                    signature = overlay.VERTEX_SIGNATURE.replace('shadeOpenPBRPathVertex', name)
                    _, _, _, actual = overlay.function_parts(changed, signature)
                    self.assertEqual(actual, '\n' + expected)
                _, _, _, original_guides = overlay.function_parts(plan['files'][overlay.SURFACE_SOURCE], overlay.GUIDE_SIGNATURE)
                self.assertEqual(original_guides, guide_body)
                _, _, _, restricted_guides = overlay.function_parts(plan['files'][overlay.SURFACE_SOURCE],
                    overlay.GUIDE_SIGNATURE.replace('evaluateSceneSurfaceGuides', 'evaluateSceneSurfaceGuidesNoFiber'))
                self.assertEqual(restricted_guides, guide_no_fiber)
                self.assertNotIn(overlay.ENTRY_SOURCE, plan['files'])

    def test_changed_or_duplicate_layout_anchors_fail_before_source_writes(self):
        cases = (
            (overlay.OWNER_SOURCE, lambda s: s.replace('module OpenPBRPathTracing;', 'module WrongOwner;')),
            (overlay.VERTEX_SOURCE, lambda s: s.replace(overlay.VERTEX_SIGNATURE, overlay.VERTEX_SIGNATURE + '\n' + overlay.VERTEX_SIGNATURE)),
            (overlay.VERTEX_SOURCE, lambda s: s.replace(overlay.COVERAGE_BRANCH, '    if (false) {')),
            (overlay.GUIDES_SOURCE, lambda s: s.replace(overlay.GUIDE_CALL, overlay.GUIDE_CALL + '\n' + overlay.GUIDE_CALL)),
            (overlay.ENTRY_SOURCE, lambda s: s.replace('[numthreads(8, 8, 1)]', '[numthreads(1, 1, 1)]')),
        )
        for relative, change in cases:
            path = self.shader_root / relative
            original = path.read_bytes()
            path.write_text(change(original.decode('utf-8').replace('\r\n', '\n')), encoding='utf-8')
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                overlay.plan_overlay(self.shader_root, 'original-general')
            path.write_bytes(original)
        existing = self.shader_root / overlay.BACKEND_SOURCE
        existing.write_text('implementing OpenPBRPathTracing;\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already contains'):
            overlay.plan_overlay(self.shader_root, 'backend-link-general')

    def test_branch_scanner_ignores_literal_and_comment_braces(self):
        source = '{ /* } */ // }\nlet text = "{\\\"}"; if (ok) { return; } }tail'
        self.assertEqual(source[:overlay.brace_end(source, 0)], source[:-4])
        for malformed in ('{ /* unfinished', '{ "unfinished', '{ if (ok) { }'):
            with self.subTest(source=malformed), self.assertRaises(ValueError):
                overlay.brace_end(malformed, 0)


class BackendEvidenceTests(unittest.TestCase):
    # Small sealed fixtures exercise execution provenance. Real shader source
    # preservation is checked separately by BackendOverlayTests and prepare.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.preparation = self.root / 'preparation'
        self.preparation.mkdir()
        self.compiler = self.root / 'compiler with spaces.exe'
        self.compiler.write_bytes(b'pinned compiler fixture')
        (self.preparation / 'Reference.spv').write_bytes(CODE)
        report = reference_report()
        save(self.preparation / 'ReferenceReport.json', report)
        base = experiment.profile.diagnostic_command(self.compiler, report, report['samples'][0],
            self.preparation / 'unused.spv')
        self.plans = []
        for label, arm, feature in experiment.ARMS:
            directory = self.preparation / 'overlays' / label
            directory.mkdir(parents=True)
            source = directory / 'Shaders' / overlay.ENTRY_SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('module OpenPBRRayQueryPathTraceGuides;\n', encoding='utf-8')
            provider_file = 'Modules/SelectedProvider.slang' if arm.startswith('link-') else None
            provider = str(directory / 'Shaders' / provider_file) if provider_file else None
            generated = {}
            if provider:
                provider_path = Path(provider)
                provider_path.parent.mkdir(parents=True)
                provider_path.write_text('module SelectedProvider;\n', encoding='utf-8')
                generated[provider_file] = experiment.profile.sha(provider_path)
            metadata = dict(feature_mask=experiment.legacy.feature_values(feature),
                provider_file=provider_file, extra_compiler_flags=[], generated_hashes=generated)
            self.plans.append(dict(label=label, backend_arm=arm,
                mechanism='original' if arm == 'original' else ('backend-link' if provider else 'backend-concrete'),
                feature_mask=experiment.legacy.feature_values(feature), provider=provider,
                overlay_metadata=metadata,
                direct_command=experiment.legacy.rewrite_command_inputs(base, directory, [ROOT / 'Shaders'])))
        self.preparation_data = dict(status='prepared', experiment='material-backend-type-v1',
            compiler=str(self.compiler), compiler_sha256=experiment.profile.sha(self.compiler),
            reference_sha256=hashlib.sha256(CODE).hexdigest(), reference_fnv1a64=experiment.profile.fnv(CODE),
            sdk_files=[], tool_inputs={}, plans=self.plans, scope='Mock compiler processes; no GPU work.')
        save(self.preparation / 'Preparation.json', self.preparation_data)
        experiment.profile.seal(self.preparation)
        self.process_count = 0

    def fake_invoke(self, command, directory, name, timeout):
        self.process_count += 1
        if '-o' in command:
            destination = Path(command[command.index('-o') + 1])
            destination.write_bytes(module_bytes(name.encode()) if destination.suffix == '.slang-module' else CODE)
        (directory / (name + '.stdout.log')).write_bytes(b'fixture stdout')
        (directory / (name + '.stderr.log')).write_bytes(b'')
        record = dict(command=command, pid=1000 + self.process_count,
            started_utc='2026-10-13T00:00:00+00:00', exit_code=0,
            timed_out=False, native_crash=False, wall_seconds=.05,
            stdout=name + '.stdout.log', stderr=name + '.stderr.log', status='passed')
        experiment.save(directory / (name + '.process.json'), record)
        return record

    def execute(self, repetitions=1, invoke=None, output_name='run'):
        args = argparse.Namespace(preparation=self.preparation, output_dir=self.root / output_name,
            repetitions=repetitions, timeout=30)
        with patch.object(experiment.profile, 'experiment_lock', return_value=nullcontext()), \
                patch.object(experiment, 'invoke', side_effect=invoke or self.fake_invoke) as process, \
                redirect_stdout(io.StringIO()):
            result = experiment.run(args)
        return args.output_dir, result, process

    def invalidate(self, output, summary, mutate_process=True):
        if mutate_process:
            for record in summary['runs']:
                directory = output / record['directory']
                for stage, (name, _) in zip(record['stages'], experiment.stage_commands(
                        next(p for p in self.plans if p['label'] == record['label']), directory)):
                    save(directory / (name + '.process.json'), stage)
        save(output / 'summary.json', summary)
        experiment.profile.seal(output)
        with self.assertRaises(ValueError):
            experiment.verify(output)

    def test_stage_commands_bind_one_provider_preserve_flags_and_separate_phases(self):
        for plan in self.plans:
            stages = experiment.stage_commands(plan, self.root / plan['label'])
            names = [name for name, _ in stages]
            expected = (['source-direct'] if plan['mechanism'] == 'original' else []) + ['consumer-precompile']
            expected += ['provider-precompile'] if plan['provider'] else []
            self.assertEqual(names, expected + ['link'])
            for name, command in stages:
                self.assertIn('-O1', command)
                self.assertIn('LITERAL=a b;$()', command)
                self.assertNotIn('-O2', command)
                self.assertNotIn('-Xspirv-opt', command)
                if name.endswith('precompile'):
                    self.assertIn('-no-codegen', command)
                    self.assertIn('-emit-ir', command)
                    self.assertNotIn('-entry', command)
                elif name == 'link':
                    self.assertNotIn('-no-codegen', command)
                    self.assertEqual(command[command.index('-entry') + 1], 'openPbrRayQueryPathTraceGuidesMain')
                    self.assertEqual(sum(Path(arg).name == 'Provider.slang-module' for arg in command), bool(plan['provider']))

    def test_complete_five_arm_alternating_schedule_and_processes_verify(self):
        output, result, process = self.execute(repetitions=2)
        self.assertEqual(result, 0)
        self.assertEqual(process.call_count, 26)
        summary = experiment.verify(output)
        self.assertEqual(len(summary['runs']), 10)
        expected = [(block, label) for block in range(2) for label in
            ([p['label'] for p in self.plans] if block == 0 else [p['label'] for p in reversed(self.plans)])]
        self.assertEqual([(r['repetition'], r['label']) for r in summary['runs']], expected)

    def test_missing_duplicate_reordered_or_wrong_arm_runs_fail(self):
        output, _, _ = self.execute()
        original = experiment.read(output / 'summary.json')
        for change in (lambda s: s['runs'].pop(), lambda s: s['runs'].append(copy.deepcopy(s['runs'][0])),
                lambda s: s['runs'].reverse(), lambda s: s['runs'][1].update(feature_mask=experiment.legacy.feature_values('no-fiber')),
                lambda s: s['runs'][1].update(directory=s['runs'][0]['directory'])):
            summary = copy.deepcopy(original)
            change(summary)
            self.invalidate(output, summary, mutate_process=False)

    def test_missing_wrong_or_extra_stage_and_provider_are_rejected(self):
        output, _, _ = self.execute()
        original = experiment.read(output / 'summary.json')
        for change in (lambda r: r['stages'].pop(), lambda r: r['stages'].append(copy.deepcopy(r['stages'][0])),
                lambda r: r['stages'][1]['command'].append('-O2'),
                lambda r: r['stages'][-1]['command'].insert(-2, str(self.root / 'WrongProvider.slang-module')),
                lambda r: r['stages'].reverse()):
            summary = copy.deepcopy(original)
            change(summary['runs'][2]) # The linked general arm needs its own provider.
            self.invalidate(output, summary)

    def test_raw_process_record_cannot_disagree_with_summary(self):
        output, _, _ = self.execute()
        summary = experiment.read(output / 'summary.json')
        process_file = output / summary['runs'][0]['directory'] / 'source-direct.process.json'
        process = experiment.read(process_file)
        process['command'].append('-O2')
        save(process_file, process)
        experiment.profile.seal(output)
        with self.assertRaisesRegex(ValueError, 'sealed backend plan'):
            experiment.verify(output)

    def test_nonfinite_nonpositive_timeout_and_crash_stages_never_qualify(self):
        output, _, _ = self.execute()
        original = experiment.read(output / 'summary.json')
        for field, value in [('wall_seconds', v) for v in (True, 0, -1, float('inf'), float('nan'))] + \
                [('timed_out', True), ('native_crash', True)]:
            summary = copy.deepcopy(original)
            summary['runs'][0]['stages'][0][field] = value
            with self.subTest(field=field, value=value):
                self.invalidate(output, summary)

    def test_one_process_cannot_be_recounted_as_another_stage(self):
        output, _, _ = self.execute()
        summary = experiment.read(output / 'summary.json')
        first, second = summary['runs'][0]['stages'][:2]
        second.update(pid=first['pid'], started_utc=first['started_utc'])
        self.invalidate(output, summary)

    def test_wrong_ir_container_is_rejected_after_resealing(self):
        output, _, _ = self.execute()
        summary = experiment.read(output / 'summary.json')
        provider = output / summary['runs'][2]['directory'] / 'Provider.slang-module'
        provider.write_bytes(b'RIFF' + struct.pack('<I', 4) + b'WRNG')
        experiment.profile.seal(output)
        with self.assertRaisesRegex(ValueError, 'serialized Slang module'):
            experiment.verify(output)

    def test_baseline_mismatch_stops_before_all_transformed_arms(self):
        def mismatch(command, directory, name, timeout):
            stage = self.fake_invoke(command, directory, name, timeout)
            if name == 'source-direct':
                (directory / 'Direct.spv').write_bytes(CODE + struct.pack('<I', 1 << 16))
            return stage
        output, result, process = self.execute(invoke=mismatch)
        self.assertEqual(result, 1)
        self.assertEqual(process.call_count, 3)
        summary = experiment.read(output / 'summary.json')
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(len(summary['runs']), 1)
        self.assertEqual(summary['runs'][0]['status'], 'baseline_bytecode_mismatch')
        experiment.legacy.verify_manifest(output)
        with self.assertRaises(ValueError):
            experiment.verify(output)

    def test_failed_stage_stops_and_seals_evidence(self):
        def fail(command, directory, name, timeout):
            stage = self.fake_invoke(command, directory, name, timeout)
            stage.update(exit_code=1, status='failed')
            experiment.save(directory / (name + '.process.json'), stage)
            return stage
        output, result, process = self.execute(invoke=fail)
        self.assertEqual(result, 1)
        self.assertEqual(process.call_count, 1)
        self.assertEqual(experiment.read(output / 'summary.json')['status'], 'failed')
        experiment.legacy.verify_manifest(output)

    def test_compiler_or_helper_change_is_rejected_before_invocation(self):
        helper = self.root / 'helper.py'
        helper.write_text('original helper\n', encoding='utf-8')
        self.preparation_data['tool_inputs'] = {str(helper): experiment.profile.sha(helper)}
        save(self.preparation / 'Preparation.json', self.preparation_data)
        experiment.profile.seal(self.preparation)
        helper.write_text('different helper\n', encoding='utf-8')
        with patch.object(experiment, 'invoke') as process, self.assertRaisesRegex(ValueError, 'helper changed'):
            experiment.run(argparse.Namespace(preparation=self.preparation, output_dir=self.root / 'changed', repetitions=1, timeout=30))
        process.assert_not_called()

    def test_resealed_preparation_cannot_change_arm_provider_or_compiler_policy(self):
        original = copy.deepcopy(self.preparation_data)
        for change in (
                lambda p: p['plans'].reverse(),
                lambda p: p['plans'][2].update(provider=None),
                lambda p: p['plans'][2].update(provider=p['plans'][4]['provider']),
                lambda p: p['plans'][1].update(provider=p['plans'][2]['provider']),
                lambda p: p['plans'][2]['overlay_metadata']['extra_compiler_flags'].append('-O2'),
                lambda p: p['plans'][2]['direct_command'].extend(['-Xspirv-opt', '--eliminate-dead-branches']),
                lambda p: p['plans'][2].update(feature_mask=experiment.legacy.feature_values('no-fiber'))):
            preparation = copy.deepcopy(original)
            change(preparation)
            save(self.preparation / 'Preparation.json', preparation)
            experiment.profile.seal(self.preparation)
            with self.subTest(change=change), self.assertRaises(ValueError):
                experiment.check_inputs(self.preparation)

    def test_resealed_generated_provider_must_match_its_sealed_digest(self):
        provider = Path(self.plans[2]['provider'])
        provider.write_text('module ChangedProvider;\n', encoding='utf-8')
        experiment.profile.seal(self.preparation)
        with self.assertRaisesRegex(ValueError, 'Generated backend shader changed'):
            experiment.check_inputs(self.preparation)

    def test_prepare_refuses_an_unrelated_shader_root_before_ownership_process(self):
        report = reference_report()
        report['samples'][0]['request']['search_path'] = str(self.root / 'OtherShaders')
        report_path = self.root / 'unrelated-report.json'
        save(report_path, report)
        with patch.object(experiment, 'invoke') as process, self.assertRaisesRegex(ValueError, 'shader root'):
            experiment.prepare(argparse.Namespace(output_dir=self.root / 'unrelated-output', report=report_path,
                request_index=146, slang_root=self.root / 'unused-sdk'))
        process.assert_not_called()

    def test_timeout_kills_mock_child_and_persists_process_evidence(self):
        directory = self.root / 'timeout'
        directory.mkdir()
        child = Mock(pid=987)
        child.wait.side_effect = [subprocess.TimeoutExpired('fixture compiler', .1), -9]
        with patch.object(experiment.subprocess, 'Popen', return_value=child), \
                patch.object(experiment.driver, 'suppress_windows_error_dialogs', return_value=nullcontext()):
            record = experiment.invoke([str(self.compiler)], directory, 'consumer-precompile', .1)
        child.kill.assert_called_once()
        self.assertTrue(record['timed_out'])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(experiment.read(directory / 'consumer-precompile.process.json'), record)
        self.assertTrue((directory / 'consumer-precompile.stdout.log').is_file())
        self.assertTrue((directory / 'consumer-precompile.stderr.log').is_file())

    def test_invalid_limits_are_rejected_without_compiler_processes(self):
        for timeout in ('nan', 'inf', '0', '-1'):
            args = ['ValidatePathTraceBackend.py', 'run', '--preparation', 'unused',
                '--output-dir', 'unused', '--timeout', timeout]
            with self.subTest(timeout=timeout), patch.object(sys, 'argv', args), \
                    patch.object(experiment.subprocess, 'Popen') as process, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    experiment.main()
                process.assert_not_called()

    def test_prepare_freezes_real_sources_and_five_owned_plans_without_compiling(self):
        sdk = self.root / 'sdk'
        compiler = sdk / 'bin/slangc.exe'
        compiler.parent.mkdir(parents=True)
        compiler.write_bytes(b'unexecuted pinned compiler fixture')
        report_path = self.root / 'ReferenceReport.json'
        save(report_path, reference_report())
        output = self.root / 'real-preparation'
        args = argparse.Namespace(output_dir=output, report=report_path, request_index=146, slang_root=sdk)
        with patch.object(experiment.profile, 'bench_bytes', return_value=(self.root / 'cache.spv', CODE)), \
                patch.object(experiment.legacy, 'compiler_manifest', return_value=[]), \
                patch.object(experiment, 'invoke', side_effect=self.fake_invoke) as process, \
                patch.object(experiment.subprocess, 'Popen') as compiler_process, redirect_stdout(io.StringIO()):
            experiment.prepare(args)
        compiler_process.assert_not_called()
        self.assertEqual(process.call_count, 5)
        self.assertTrue(all(call.args[0][1:3] == ['-B', str(ROOT / 'Tools/CheckShaderModules.py')]
            for call in process.call_args_list))
        prepared = experiment.check_inputs(output)
        self.assertEqual([p['label'] for p in prepared['plans']], list(overlay.ARMS))
        for plan in prepared['plans']:
            self.assertEqual(plan['feature_mask'], plan['overlay_metadata']['feature_mask'])
            self.assertTrue(Path(plan['direct_command'][1]).is_relative_to(output))
            self.assertIsNotNone(plan['provider'] if plan['mechanism'] == 'backend-link' else True)
        original = output / 'overlays/original-general/Shaders'
        for relative in overlay.SOURCE_FILES:
            self.assertEqual((original / relative).read_bytes(), (ROOT / 'Shaders' / relative).read_bytes())


if __name__ == '__main__':
    unittest.main()
