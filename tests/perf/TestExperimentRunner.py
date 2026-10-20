"""Decision/recovery negative tests; synthetic timings never serve as GPU evidence."""
import copy
import datetime
import io
import math
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Tools/Perf'))
import ExperimentRunner as e
import TestWorkloadCase as fixtures


class LegacyExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkloadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.analyze()
        self.case = self.fixture.case
        self.policy = copy.deepcopy(e.LEGACY_POLICY)
        self.rows = []
        for index, arm in enumerate('ABBAABBA'):
            r = copy.deepcopy(self.base)
            if arm == 'B':
                for d in r['identity']['snapshot']['productionDispatches']:
                    d['spirvFnv1a64'] = '654321'
                r['timings']['softwareTotalMs']['median'] *= .9
            self.rows.append({'arm': arm, 'result': r, 'process': {'exitCode': 0, 'pid':1000+index,
                              'startedUnix':index*10, 'endedUnix':index*10+5}, 'competition': {'covered': True}})

    def assess(self):
        return e.assess(self.rows, self.case, self.policy, 8)

    def test_gain_requires_independent_pairs(self):
        r = self.assess()
        self.assertEqual(r['decision'], 'accept')
        self.assertEqual(r['gains']['softwareTotalMs']['pairs'], 4)
        self.assertNotIn('stage',r)
        self.assertNotIn('effectAssessment',r)
        self.assertNotIn('noiseWarning',r)

    def test_reused_process_is_not_an_independent_sample(self):
        self.rows[1]['process'] = copy.deepcopy(self.rows[0]['process'])
        self.assertEqual(self.assess()['reason'],'independent_process_evidence_invalid')

    def test_no_gain_rejected(self):
        for row in self.rows:
            row['result']['timings']['softwareTotalMs']['median'] = .6
        self.assertEqual(self.assess()['decision'], 'reject')

    def test_correctness_failure_rejects_even_fast_candidate(self):
        self.rows[1]['result']['identity']['snapshot']['VBuffer.depth']['sha256'] = 'wrong'
        self.assertEqual(self.assess()['reason'], 'exact_depth_visibility_mismatch')

    def test_input_or_shader_drift_cannot_accept(self):
        for field, value in [('softwareListHash', 'changed'), ('softwareDispatch', [1,1,1])]:
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[1]['result']['identity']['snapshot'][e.w.PHASES[0]][field] = value
                self.assertEqual(e.assess(rows, self.case, self.policy, 8)['decision'], 'inconclusive')
        self.rows[2]['result']['identity']['snapshot']['productionDispatches'][0]['spirvFnv1a64'] = '999'
        self.assertEqual(self.assess()['reason'], 'within_arm_identity_drift')

    def test_same_compiled_shader_cannot_be_speedup(self):
        for row in self.rows:
            for d in row['result']['identity']['snapshot']['productionDispatches']:
                d['spirvFnv1a64'] = '123456'
        self.assertEqual(self.assess()['reason'], 'candidate_compiled_to_same_shader')

    def test_instrumentation_and_missing_monitor_fail_closed(self):
        self.rows[1]['result']['normalTiming'] = False
        self.assertEqual(self.assess()['decision'], 'inconclusive')
        self.rows[1]['result']['normalTiming'] = True
        self.rows[1]['competition']['covered'] = False
        self.assertEqual(self.assess()['reason'], 'process_or_environment_evidence_missing')

    def test_monitor_requires_each_window_and_retains_zero_samples(self):
        root = self.fixture.path
        e.w.save(root/'Capture.json', {'cases':[
            {'measurementBeginUnixMs':1000,'measurementEndUnixMs':2000},
            {'measurementBeginUnixMs':5000,'measurementEndUnixMs':6000}]})
        header = 'timestamp,pid,process,engine,utilization\n'
        first = '1970-01-01T00:00:02+00:00,42,Metallic,gpu,0\n'
        second = '1970-01-01T00:00:06+00:00,42,Metallic,gpu,1\n'
        (root/'GpuProcesses.csv').write_text(header+first)
        self.assertFalse(e.monitor_coverage(root,42)['covered'])
        (root/'GpuProcesses.csv').write_text(header+first+second)
        self.assertTrue(e.monitor_coverage(root,42)['covered'])

    def test_incomplete_and_wrong_order(self):
        self.rows.pop()
        self.assertEqual(self.assess()['reason'], 'incomplete_schedule')
        self.setUp()
        self.rows.sort(key=lambda r:r['arm'])
        self.assertEqual(self.assess()['reason'], 'invalid_interleaving')

    def test_noise_and_graph_guard(self):
        for row in self.rows:
            if row['arm'] == 'B':
                row['result']['timings']['graphGpuMs']['median'] *= 1.1
        self.assertEqual(self.assess()['reason'], 'graph_regression')
        self.rows[0]['result']['timings']['graphGpuMs']['median'] *= 2
        self.assertEqual(self.assess()['reason'], 'within_arm_timing_noise')

    def test_uncertainty_does_not_become_acceptance(self):
        b = [r for r in self.rows if r['arm'] == 'B']
        for row, gain in zip(b, (.015,.025,.035,.045)):
            row['result']['timings']['softwareTotalMs']['median'] = .6*(1-gain)
        self.assertEqual(self.assess()['reason'], 'confidence_interval_crosses_gate')

    def test_patch_hash_context_and_scope(self):
        original = b'hello\r\n'
        spec = {'protocol':e.PROTOCOL,'id':'test','hypothesis':'test only','path':e.TARGET,
                'baseSha256':e.sha(original),'replacements':[{'before':'hello','after':'world','count':1}]}
        self.assertEqual(e.candidate_bytes(spec, original), b'world\r\n')
        for change in ({'path':'../outside.slang'}, {'baseSha256':'bad'},
                       {'replacements':[{'before':'missing','after':'x','count':1}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                e.candidate_bytes({**spec, **change}, original)

    def test_transaction_restores_and_preserves_concurrent_edit(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'ROOT', Path(temp)):
            root = Path(temp)
            target = root / e.TARGET
            target.parent.mkdir(parents=True)
            target.write_bytes(b'baseline')
            transaction = e.ShaderTransaction(root, b'baseline', b'candidate')
            transaction.install(b'candidate')
            transaction.restore()
            self.assertEqual(target.read_bytes(), b'baseline')
            transaction.install(b'candidate')
            target.write_bytes(b'user change')
            with self.assertRaisesRegex(ValueError, 'Concurrent'):
                transaction.restore()
            self.assertEqual(target.read_bytes(), b'user change')

    def test_recover_candidate_and_damaged_backup(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'ROOT', Path(temp)):
            root = Path(temp)
            target = root / e.TARGET
            target.parent.mkdir(parents=True)
            target.write_bytes(b'a')
            (root/'Baseline.slang').write_bytes(b'a')
            transaction = e.ShaderTransaction(root,b'a',b'b')
            transaction.install(b'b')
            e.recover(root)
            self.assertEqual(target.read_bytes(), b'a')
            (root/'Baseline.slang').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'damaged'):
                e.recover(root)

    def test_execution_exception_archives_failure_and_restores(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'ROOT', Path(temp)):
            root = Path(temp)
            target = root / e.TARGET
            target.parent.mkdir(parents=True)
            target.write_bytes(b'baseline')
            case = {**self.case, 'sampleId':'gpu-driven-sample', 'primeCameraOffset':[0,0,3]}
            candidate = {'protocol':e.PROTOCOL,'id':'test','hypothesis':'synthetic recovery test',
                         'path':e.TARGET,'baseSha256':e.sha(b'baseline'),
                         'replacements':[{'before':'baseline','after':'candidate','count':1}]}
            assets = {'protocol':'metallic-declared-assets-v1','stream':e.w.ASSETS[case['sampleId']]}
            for name, value in [('Case.json',case),('Candidate.json',candidate),('Assets.json',assets)]:
                e.w.save(root/name,value)
            args = SimpleNamespace(case=root/'Case.json',candidate=root/'Candidate.json',assets=root/'Assets.json',
                                   output=root/'result',blocks=2,confirmation_blocks=2,timeout=30,
                                   build_dir=root/'build',exe=root/'unused.exe')
            with patch.object(e.w,'asset_metadata_matches'), patch.object(e,'validate_build'), patch.object(e.w,'source_inventory',side_effect=RuntimeError('test failure')):
                result = e.execute(args)
            self.assertEqual(result['decision'],'inconclusive')
            self.assertEqual(target.read_bytes(),b'baseline')
            self.assertFalse((root/'build/shader-experiment.lock').exists())
            self.assertEqual(e.verify(root/'result')['integrity'],'passed')
            (root/'result/Baseline.slang').write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError,'Artifact changed'):
                e.verify(root/'result')

    def test_build_cannot_use_debug_wrong_tree_or_stale_other_exe(self):
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp)
            cache = build/'CMakeCache.txt'
            good = f'CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={e.ROOT}\n'
            cache.write_text(good)
            exe = build/'Source/MetallicGPUDrivenSample.exe'
            self.assertEqual(e.validate_build(build,exe)['configuration'],'Release')
            with self.assertRaisesRegex(ValueError,'Executable'):
                e.validate_build(build,build/'other.exe')
            cache.write_text(good.replace('=Release','=Debug'))
            with self.assertRaisesRegex(ValueError,'Release'):
                e.validate_build(build,exe)
            cache.write_text(good.replace(str(e.ROOT),str(build)))
            with self.assertRaisesRegex(ValueError,'another'):
                e.validate_build(build,exe)

    def test_offline_acceptance_recomputed_and_confirmation_cannot_be_omitted(self):
        # Entirely synthetic evidence; discarded with this temporary directory.
        root = self.fixture.path/'synthetic-experiment'
        root.mkdir()
        original_capture, original_frames = copy.deepcopy(self.fixture.capture), copy.deepcopy(self.fixture.frames)
        buffers = {key:(self.fixture.path/(key+'.bin')).read_bytes() for key in e.w.OUTPUTS}
        spec = {'protocol':e.PROTOCOL,'id':'synthetic','hypothesis':'unit test only','path':e.TARGET,
                'baseSha256':e.sha(b'a'),'replacements':[{'before':'a','after':'b','count':1}]}
        e.w.save(root/'Candidate.json',spec)
        e.w.save(root/'Case.json',self.case)
        (root/'Baseline.slang').write_bytes(b'a'); (root/'Candidate.slang').write_bytes(b'b')
        e.w.save(root/'Build.json',{'exitCode':0})
        (root/'CMakeCache.txt').write_bytes(b'synthetic cache')
        e.w.save(root/'Transaction.json',{'state':'restored'})
        manifest = {'protocol':e.PROTOCOL,'status':'complete','policy':e.LEGACY_POLICY,'blocks':2,
                    'confirmationBlocks':2,'restored':True,'runs':[],
                    'buildIdentity':{'configuration':'Release','cacheSha256':e.w.digest(root/'CMakeCache.txt')},
                    'toolHashes':{}}
        for name in ('ExperimentRunner.py','WorkloadCase.py','MeasureExperimentGPU.ps1'):
            data = Path(e.__file__).with_name(name).read_bytes()
            (root/name).write_bytes(data); manifest['toolHashes'][name] = e.sha(data)
        for stage_index, stage in enumerate(('discovery','confirmation')):
            rows = []
            for index, arm in enumerate('ABBAABBA'):
                directory = f'{stage}-{index+1:02d}-{arm}'
                run = root/directory; run.mkdir()
                for key,data in buffers.items(): (run/(key+'.bin')).write_bytes(data)
                self.fixture.path = run
                self.fixture.capture = copy.deepcopy(original_capture)
                self.fixture.frames = copy.deepcopy(original_frames)
                seconds = (stage_index*8+index)*10
                pid = 1000+stage_index*8+index
                measured = self.fixture.capture['cases'][0]
                measured.update(measurementBeginUnixMs=seconds*1000+1000,measurementEndUnixMs=seconds*1000+2000)
                if arm == 'B':
                    for side in ('before','after'):
                        for d in measured[side]['productionDispatches']: d['spirvFnv1a64']='654321'
                    for frame in self.fixture.frames:
                        frame['streaming'][0]['softwareRaster']['spirvFnv1a64']='654321'
                        for scope in frame['scopes']:
                            if 'Software raster' in scope['path']: scope['gpuMs'] *= .9
                result = self.fixture.analyze()
                process = {'pid':pid,'exitCode':0,'startedUnix':seconds,'endedUnix':seconds+5}
                e.w.save(run/'Process.json',process)
                timestamp = datetime.datetime.fromtimestamp(seconds+2,datetime.timezone.utc).isoformat()
                (run/'GpuProcesses.csv').write_text(f'timestamp,pid,process,engine,utilization\n{timestamp},{pid},synthetic,gpu,0\n')
                entry = {'directory':directory,'stage':stage,'arm':arm}
                manifest['runs'].append(entry)
                rows.append({**entry,'result':result,'process':process,'competition':e.monitor_coverage(run,pid)})
            decision = e.assess(rows,self.case,e.LEGACY_POLICY,8)
            self.assertEqual(decision['decision'],'accept')
            e.w.save(root/(stage+'-Decision.json'),decision)
        e.w.save(root/'Decision.json',decision)
        manifest['artifacts'] = e.artifact_hashes(root)
        e.w.save(root/'Manifest.json',manifest)
        self.assertEqual(e.verify(root)['decision'],'accept')
        # Archived experiments retain their original producer file names and hashes.
        monitor = root/'MeasureExperimentGPU.ps1'
        temporary_monitor = root/'ArchivedMonitor.tmp'
        monitor.rename(temporary_monitor)
        temporary_monitor.rename(root/'MeasureExperimentGpu.ps1')
        manifest['toolHashes']['MeasureExperimentGpu.ps1'] = manifest['toolHashes'].pop('MeasureExperimentGPU.ps1')
        manifest['artifacts'] = e.artifact_hashes(root)
        e.w.save(root/'Manifest.json',manifest)
        self.assertEqual(e.verify(root)['decision'],'accept')
        manifest['runs'] = manifest['runs'][:8]
        e.w.save(root/'Manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'confirmation'):
            e.verify(root)


class CurrentExperimentTests(unittest.TestCase):
    def setUp(self):
        LegacyExperimentTests.setUp(self)
        self.policy = copy.deepcopy(e.POLICY)
        self.capture = copy.deepcopy(self.fixture.capture)
        self.frames = copy.deepcopy(self.fixture.frames)
        self.buffers = {key:(self.fixture.path/(key+'.bin')).read_bytes() for key in e.w.OUTPUTS}

    def assess(self, stage='confirmation'):
        return e.assess(self.rows, self.case, self.policy, len(self.rows), stage=stage)

    def block_timings(self, gains, scales=None, graph_gains=None):
        scales = scales or [1] * len(gains)
        graph_gains = graph_gains or [0] * len(gains)
        self.rows = []
        for gain, scale, graph_gain in zip(gains, scales, graph_gains):
            for arm in 'ABBA':
                index = len(self.rows)
                result = copy.deepcopy(self.base)
                if arm == 'B':
                    for dispatch in result['identity']['snapshot']['productionDispatches']:
                        dispatch['spirvFnv1a64'] = '654321'
                result['timings']['softwareTotalMs']['median'] = .6 * scale * (1-gain if arm == 'B' else 1)
                result['timings']['graphGpuMs']['median'] = 4 * scale * (1-graph_gain if arm == 'B' else 1)
                self.rows.append({'arm':arm,'result':result,
                    'process':{'pid':1000+index,'exitCode':0,'startedUnix':index*10,'endedUnix':index*10+5},
                    'competition':{'covered':True}})

    def write_run(self, directory, arm, index, gain, scale=1, graph_gain=0):
        """Produce raw synthetic captures so verification exercises the real parser."""
        directory.mkdir()
        for resource, data in self.buffers.items():
            (directory/(resource+'.bin')).write_bytes(data)
        saved = self.fixture.path, self.fixture.capture, self.fixture.frames
        try:
            self.fixture.path = directory
            self.fixture.capture = copy.deepcopy(self.capture)
            self.fixture.frames = copy.deepcopy(self.frames)
            self.fixture.capture['config'] = e.w.engine_config(self.case)
            self.fixture.capture['workloadCase'] = e.w.engine_config(self.case)['workloadCase']
            measured = self.fixture.capture['cases'][0]
            seconds, pid = index*10, 1000+index
            measured.update(measurementBeginUnixMs=seconds*1000+1000,measurementEndUnixMs=seconds*1000+2000)
            if arm == 'B':
                for checkpoint in ('before','after'):
                    for dispatch in measured[checkpoint]['productionDispatches']:
                        dispatch['spirvFnv1a64'] = '654321'
            for frame in self.fixture.frames:
                if arm == 'B':
                    frame['streaming'][0]['softwareRaster']['spirvFnv1a64'] = '654321'
                for scope in frame['scopes']:
                    if 'Software raster' in scope['path']:
                        scope['gpuMs'] *= scale * (1-gain if arm == 'B' else 1)
                    elif 'RenderGraph GPU envelope' in scope['path']:
                        scope['gpuMs'] *= scale * (1-graph_gain if arm == 'B' else 1)
            result = self.fixture.analyze()
        finally:
            self.fixture.path, self.fixture.capture, self.fixture.frames = saved
        process = {'pid':pid,'exitCode':0,'startedUnix':seconds,'endedUnix':seconds+5}
        e.w.save(directory/'Process.json',process)
        timestamp = datetime.datetime.fromtimestamp(seconds+2,datetime.timezone.utc).isoformat()
        (directory/'GpuProcesses.csv').write_text(
            f'timestamp,pid,process,engine,utilization\n{timestamp},{pid},synthetic,gpu,0\n')
        return {'arm':arm,'result':result,'process':process,'competition':e.monitor_coverage(directory,pid)}

    def archive(self, discovery=(.1,.1), confirmation=(.1,.1), scales=None):
        root = self.fixture.path/'current-experiment'
        root.mkdir()
        candidate = {'protocol':e.PROTOCOL,'id':'synthetic','hypothesis':'unit test only','path':e.TARGET,
                     'baseSha256':e.sha(b'a'),'replacements':[{'before':'a','after':'b','count':1}]}
        e.w.save(root/'Candidate.json',candidate)
        e.w.save(root/'Case.json',self.case)
        (root/'Baseline.slang').write_bytes(b'a')
        (root/'Candidate.slang').write_bytes(b'b')
        e.w.save(root/'Build.json',{'exitCode':0})
        (root/'CMakeCache.txt').write_bytes(b'synthetic cache')
        e.w.save(root/'Transaction.json',{'state':'restored'})
        manifest = {'protocol':e.CURRENT_PROTOCOL,'status':'complete','policy':self.policy,
                    'blocks':len(discovery),'confirmationBlocks':len(confirmation),'restored':True,'runs':[],
                    'buildIdentity':{'configuration':'Release','cacheSha256':e.w.digest(root/'CMakeCache.txt')},
                    'toolHashes':{}}
        for name in ('ExperimentRunner.py','WorkloadCase.py','MeasureExperimentGPU.ps1'):
            data = Path(e.__file__).with_name(name).read_bytes()
            (root/name).write_bytes(data)
            manifest['toolHashes'][name] = e.sha(data)
        reference_rows = []
        for stage, gains in (('discovery',discovery),('confirmation',confirmation)):
            rows = []
            for local_index, arm in enumerate('ABBA'*len(gains)):
                directory = f'{stage}-{local_index+1:02d}-{arm}'
                block = local_index//4
                scale = scales[block] if scales else 1
                row = self.write_run(root/directory,arm,len(manifest['runs']),gains[block],scale)
                entry = {'directory':directory,'stage':stage,'arm':arm}
                manifest['runs'].append(entry)
                rows.append({**row,**entry})
            decision = e.assess(rows,self.case,self.policy,len(gains)*4,stage=stage,reference_rows=reference_rows)
            e.w.save(root/(stage+'-Decision.json'),decision)
            if stage == 'discovery':
                reference_rows = rows
        e.w.save(root/'Decision.json',decision)
        self.save_manifest(root,manifest)
        return root,manifest

    @staticmethod
    def save_manifest(root, manifest):
        manifest['artifacts'] = e.artifact_hashes(root)
        e.w.save(root/'Manifest.json',manifest)

    def execute_synthetic(self, discovery, confirmation):
        root = self.fixture.path/'execution'
        root.mkdir()
        target = root/e.TARGET
        target.parent.mkdir(parents=True)
        target.write_bytes(b'baseline')
        self.case = self.fixture.case = {**self.case,'sampleId':'gpu-driven-sample','primeCameraOffset':[0,0,3]}
        candidate = {'protocol':e.PROTOCOL,'id':'synthetic','hypothesis':'unit test only','path':e.TARGET,
                     'baseSha256':e.sha(b'baseline'),'replacements':[{'before':'baseline','after':'candidate','count':1}]}
        assets = {'protocol':'metallic-declared-assets-v1','stream':e.w.ASSETS[self.case['sampleId']]}
        for name, value in (('Case.json',self.case),('Candidate.json',candidate),('Assets.json',assets)):
            e.w.save(root/name,value)
        build = root/'build'
        executable = build/'Source/MetallicGPUDrivenSample.exe'
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b'synthetic executable')
        (build/'CMakeCache.txt').write_text(
            f'CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={root}\n')
        args = SimpleNamespace(case=root/'Case.json',candidate=root/'Candidate.json',assets=root/'Assets.json',
                               output=root/'result',blocks=len(discovery),confirmation_blocks=len(confirmation),
                               timeout=30,build_dir=build,exe=executable)
        calls = []

        def collect(case, exe, directory, timeout):
            stage, number, arm = directory.name.split('-')
            gains = discovery if stage == 'discovery' else confirmation
            row = self.write_run(directory,arm,len(calls),gains[(int(number)-1)//4])
            calls.append(stage)
            return row

        with patch.object(e,'ROOT',root), patch.object(e.w,'asset_metadata_matches'), \
             patch.object(e.w,'source_inventory',side_effect=lambda:{e.TARGET:e.sha(target.read_bytes())}), \
             patch.object(e,'gpu_identity',return_value='synthetic GPU'), \
             patch.object(e.subprocess,'run',return_value=SimpleNamespace(returncode=0)), \
             patch.object(e,'collect',side_effect=collect), redirect_stdout(io.StringIO()):
            decision = e.execute(args)
        self.assertEqual(target.read_bytes(),b'baseline')
        self.assertFalse((build/'shader-experiment.lock').exists())
        return decision,calls,args.output

    def test_common_drift_retains_gain_and_warns_without_early_exit(self):
        self.block_timings([.1,.1,.1],[.8,1,1.2])
        decision = self.assess()
        self.assertEqual(decision['decision'],'accept')
        self.assertTrue(decision['noiseWarning'])
        self.assertTrue(any(q['status'] != 'stable' for q in decision['qualification'].values()))
        gain = decision['gains']['softwareTotalMs']
        self.assertEqual(gain['blocks'],3)
        for value in gain['values']:
            self.assertAlmostEqual(value,.1)
        self.assertAlmostEqual(gain['estimate'],.1)
        self.assertAlmostEqual(gain['lower95'],.1)
        legacy = e.assess(self.rows,self.case,e.LEGACY_POLICY,len(self.rows))
        self.assertEqual(legacy['reason'],'within_arm_timing_noise')
        self.assertNotIn('gains',legacy)

    def test_block_log_ratio_interval_uses_blocks_not_process_pairs(self):
        self.block_timings([.05,.1,.15])
        decision = self.assess()
        gain = decision['gains']['softwareTotalMs']
        logs = [math.log(.95),math.log(.9),math.log(.85)]
        mean = sum(logs)/3
        variance = sum((value-mean)**2 for value in logs)/2
        margin = 4.303*math.sqrt(variance/3)
        self.assertEqual(gain['blocks'],3)
        self.assertEqual(len(gain['values']),3)
        for actual, expected in zip(gain['logRatios'],logs):
            self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(gain['meanLogRatio'],mean)
        self.assertAlmostEqual(gain['estimate'],1-math.exp(mean))
        self.assertAlmostEqual(gain['lower95'],1-math.exp(mean+margin))
        self.assertAlmostEqual(gain['upper95'],1-math.exp(mean-margin))

    def test_abba_cancels_shared_log_linear_drift_inside_each_block(self):
        self.block_timings([.1,.1,.1])
        for index,row in enumerate(self.rows):
            for metric in ('softwareTotalMs','graphGpuMs'):
                row['result']['timings'][metric]['median'] *= math.exp(.08*index)
        decision = self.assess()
        self.assertTrue(decision['noiseWarning'])
        self.assertEqual(decision['decision'],'accept')
        self.assertAlmostEqual(decision['gains']['softwareTotalMs']['estimate'],.1)
        self.assertAlmostEqual(decision['gains']['graphGpuMs']['estimate'],0)

    def test_discovery_promising_gain_schedules_confirmation_without_acceptance(self):
        self.block_timings([.01,.09,.17])
        decision = self.assess('discovery')
        self.assertEqual(decision['decision'],'inconclusive')
        self.assertFalse(decision['candidateAccepted'])
        self.assertEqual(decision['reason'],'promising_gain')
        self.assertEqual(decision['nextAction'],'run_confirmation')
        self.assertEqual(decision['stage'],'discovery')
        self.assertEqual(decision['evidenceValidity'],'valid')
        self.assertEqual(decision['effectAssessment'],'promising')
        self.assertLess(decision['gains']['softwareTotalMs']['lower95'],self.policy['minimumGain'])

    def test_discovery_below_point_target_with_wide_interval_retains_candidate(self):
        self.block_timings([-.08,.01,.08])
        decision = self.assess('discovery')
        self.assertEqual(decision['decision'],'inconclusive')
        self.assertEqual(decision['nextAction'],'retain_candidate')
        self.assertLess(decision['gains']['softwareTotalMs']['estimate'],self.policy['minimumGain'])
        self.assertGreater(decision['gains']['softwareTotalMs']['upper95'],self.policy['minimumGain'])

    def test_zero_gain_and_small_reliable_gain_are_below_target(self):
        for gain in (0,.02):
            with self.subTest(gain=gain):
                self.block_timings([gain]*3)
                decision = self.assess('discovery')
                self.assertEqual(decision['decision'],'reject')
                self.assertEqual(decision['reason'],'gain_below_threshold')
                self.assertEqual(decision['effectAssessment'],'below_target')
                self.assertAlmostEqual(decision['gains']['softwareTotalMs']['estimate'],gain)

    def test_primary_and_graph_regressions_are_explicit(self):
        for gains, graph_gains, reason in (([-.1]*3,[0]*3,'primary_regression'),
                                          ([.1]*3,[-.1]*3,'graph_regression')):
            with self.subTest(reason=reason):
                self.block_timings(gains,graph_gains=graph_gains)
                decision = self.assess('discovery')
                self.assertEqual(decision['decision'],'reject')
                self.assertEqual(decision['reason'],reason)
                self.assertEqual(decision['effectAssessment'],'confirmed_regression')
                self.assertFalse(decision['candidateAccepted'])

    def test_correctness_and_identity_remain_hard_boundaries(self):
        for field, reason, status in (('output','exact_depth_visibility_mismatch','reject'),
                                      ('input','workload_input_or_binding_drift','inconclusive'),
                                      ('input_and_output','workload_input_or_binding_drift','inconclusive')):
            with self.subTest(field=field):
                self.block_timings([.1,.1,.1],[.8,1,1.2])
                snapshot = self.rows[1]['result']['identity']['snapshot']
                if field in ('output','input_and_output'):
                    snapshot['VBuffer.depth']['sha256'] = 'wrong'
                if field in ('input','input_and_output'):
                    snapshot[e.w.PHASES[0]]['softwareListHash'] = 'wrong'
                decision = self.assess('discovery')
                self.assertEqual(decision['decision'],status)
                self.assertEqual(decision['reason'],reason)
                self.assertNotEqual(decision['nextAction'],'run_confirmation')
                self.assertEqual(decision['evidenceValidity'],'valid' if field == 'output' else 'invalid')
                self.assertEqual(decision['effectAssessment'],'correctness_failure' if field == 'output' else 'not_assessed')

    def test_invalid_collection_is_not_reported_as_uncertain_gain(self):
        for problem in ('instrumented','uncovered','failed_process'):
            with self.subTest(problem=problem):
                self.block_timings([.1,.1,.1])
                if problem == 'instrumented':
                    self.rows[0]['result']['normalTiming'] = False
                elif problem == 'uncovered':
                    self.rows[0]['competition']['covered'] = False
                else:
                    self.rows[0]['process']['exitCode'] = 1
                decision = self.assess('discovery')
                self.assertEqual(decision['evidenceValidity'],'invalid')
                self.assertEqual(decision['effectAssessment'],'not_assessed')
                self.assertEqual(decision['nextAction'],'repair_evidence')
                self.assertNotIn('gains',decision)

    def test_confirmation_uncertainty_retains_candidate_without_more_sampling(self):
        self.block_timings([.01,.09,.17])
        decision = self.assess()
        self.assertEqual(decision['decision'],'inconclusive')
        self.assertEqual(decision['nextAction'],'retain_candidate')
        self.assertEqual(decision['effectAssessment'],'uncertain')
        self.assertFalse(decision['candidateAccepted'])

    def test_candidate_graph_tail_prevents_confirmation_acceptance(self):
        self.block_timings([.1,.1,.1],graph_gains=[0,0,-.15])
        decision = self.assess()
        self.assertTrue(decision['noiseWarning'])
        self.assertEqual(decision['decision'],'inconclusive')
        self.assertEqual(decision['nextAction'],'retain_candidate')
        self.assertLess(decision['gains']['graphGpuMs']['lower95'],-self.policy['maximumGraphRegression'])

    def test_confirmation_checks_discovery_contract_without_pooling_its_timings(self):
        self.block_timings([.1,.1,.1])
        reference = self.rows
        self.block_timings([0,0,0])
        for row in self.rows:
            row['process']['startedUnix'] += 120
            row['process']['endedUnix'] += 120
            row['process']['pid'] += 12
        decision = e.assess(self.rows,self.case,self.policy,12,reference_rows=reference)
        self.assertEqual(decision['decision'],'reject')
        self.assertEqual(decision['effectAssessment'],'below_target')
        self.assertAlmostEqual(decision['gains']['softwareTotalMs']['estimate'],0)

    def test_confirmation_rejects_cross_stage_output_or_identity_changes(self):
        self.block_timings([.1,.1,.1])
        reference = self.rows
        for change in ('output','candidate_output','input','shader'):
            with self.subTest(change=change):
                self.block_timings([.1,.1,.1])
                for row in self.rows:
                    row['process']['startedUnix'] += 120
                    row['process']['endedUnix'] += 120
                    row['process']['pid'] += 12
                    snapshot = row['result']['identity']['snapshot']
                    if change == 'output':
                        snapshot['VBuffer.depth']['sha256'] = 'different'
                    elif change == 'candidate_output' and row['arm'] == 'B':
                        snapshot['VBuffer.depth']['sha256'] = 'different'
                    elif change == 'input':
                        snapshot[e.w.PHASES[0]]['softwareListHash'] = 'different'
                    elif row['arm'] == 'B':
                        for dispatch in snapshot['productionDispatches']:
                            dispatch['spirvFnv1a64'] = '999999'
                decision = e.assess(self.rows,self.case,self.policy,12,reference_rows=reference)
                self.assertFalse(decision['candidateAccepted'])
                self.assertEqual(decision['evidenceValidity'],'valid' if change == 'candidate_output' else 'invalid')
                if change == 'candidate_output':
                    self.assertEqual(decision['effectAssessment'],'correctness_failure')
                self.assertNotIn('gains',decision)

    def test_execute_promising_discovery_runs_fixed_independent_confirmation(self):
        decision,calls,root = self.execute_synthetic([.01,.09,.17],[.1]*5)
        self.assertEqual(calls,['discovery']*12+['confirmation']*20)
        self.assertEqual(decision['decision'],'accept')
        discovery = e.w.load(root/'discovery-Decision.json')
        self.assertEqual(discovery['decision'],'inconclusive')
        self.assertEqual(discovery['nextAction'],'run_confirmation')
        self.assertEqual(e.verify(root)['decision'],'accept')

    def test_execute_confirmation_failure_does_not_reuse_discovery_gain(self):
        decision,calls,root = self.execute_synthetic([.1]*3,[0]*5)
        self.assertEqual(calls,['discovery']*12+['confirmation']*20)
        self.assertEqual(decision['decision'],'reject')
        self.assertAlmostEqual(decision['gains']['softwareTotalMs']['estimate'],0)
        self.assertEqual(e.verify(root)['decision'],'reject')

    def test_execute_does_not_confirm_rejected_discovery(self):
        decision,calls,root = self.execute_synthetic([0]*3,[.1]*5)
        self.assertEqual(calls,['discovery']*12)
        self.assertEqual(decision['decision'],'reject')
        self.assertFalse((root/'confirmation-Decision.json').exists())
        self.assertEqual(e.verify(root)['decision'],'reject')

    def test_execute_uncertain_confirmation_stops_at_registered_budget(self):
        decision,calls,root = self.execute_synthetic([.1]*3,[.01,.09,.17])
        self.assertEqual(calls,['discovery']*12+['confirmation']*12)
        self.assertEqual(decision['decision'],'inconclusive')
        self.assertEqual(decision['nextAction'],'retain_candidate')
        self.assertEqual(e.verify(root)['decision'],'inconclusive')

    def test_cli_predeclares_three_discovery_and_five_confirmation_blocks(self):
        argv = ['ExperimentRunner.py','run']
        for option in ('case','candidate','assets','exe','build-dir','output'):
            argv.extend(['--'+option,'unused'])
        with patch.object(sys,'argv',argv), patch.object(e,'execute',return_value={'decision':'inconclusive'}) as run, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(e.main(),2)
        args = run.call_args.args[0]
        self.assertEqual(args.blocks,3)
        self.assertEqual(args.confirmation_blocks,5)

    def test_offline_current_evidence_recomputes_noise_and_classification(self):
        root,manifest = self.archive(discovery=(.1,.1,.1),confirmation=(.1,.1,.1),scales=(.8,1,1.2))
        result = e.verify(root)
        self.assertEqual(result['decision'],'accept')
        self.assertTrue(result['noiseWarning'])
        decision = e.w.load(root/'confirmation-Decision.json')
        decision['effectAssessment'] = 'forged_classification'
        e.w.save(root/'confirmation-Decision.json',decision)
        e.w.save(root/'Decision.json',decision)
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'decision|classification'):
            e.verify(root)

    def test_offline_promising_discovery_cannot_omit_confirmation(self):
        root,manifest = self.archive()
        manifest['runs'] = manifest['runs'][:8]
        e.w.save(root/'Decision.json',e.w.load(root/'discovery-Decision.json'))
        # Keep all raw evidence and rehash: completed promising discovery must still confirm.
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'confirmation|schedule'):
            e.verify(root)

    def test_offline_confirmation_without_eligible_discovery_is_invalid(self):
        root,manifest = self.archive(discovery=(0,0))
        with self.assertRaisesRegex(ValueError,'discovery|confirmation'):
            e.verify(root)

    def test_offline_stage_reordering_is_invalid(self):
        root,manifest = self.archive()
        manifest['runs'] = manifest['runs'][8:]+manifest['runs'][:8]
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'schedule|order|process|discovery'):
            e.verify(root)

    def test_offline_protocol_policy_mismatch_is_invalid(self):
        root,manifest = self.archive()
        manifest['policy'] = copy.deepcopy(e.LEGACY_POLICY)
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'policy'):
            e.verify(root)

    def test_offline_confirmation_cannot_change_workload_between_stages(self):
        root,manifest = self.archive()
        for entry in manifest['runs']:
            if entry['stage'] == 'confirmation':
                path = root/entry['directory']/'frames.json'
                frames = e.w.load(path)
                for frame in frames:
                    frame['streaming'][0]['residentPages'] = 2
                e.w.save(path,frames)
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'decision|identity|workload'):
            e.verify(root)

    def test_offline_confirmation_cannot_change_candidate_shader_between_stages(self):
        root,manifest = self.archive()
        for entry in manifest['runs']:
            if entry['stage'] != 'confirmation' or entry['arm'] != 'B':
                continue
            directory = root/entry['directory']
            capture = e.w.load(directory/'Capture.json')
            for measured in capture['cases']:
                for checkpoint in ('before','after'):
                    for dispatch in measured[checkpoint]['productionDispatches']:
                        dispatch['spirvFnv1a64'] = '999999'
            e.w.save(directory/'Capture.json',capture)
            frames = e.w.load(directory/'frames.json')
            for frame in frames:
                frame['streaming'][0]['softwareRaster']['spirvFnv1a64'] = '999999'
            e.w.save(directory/'frames.json',frames)
        self.save_manifest(root,manifest)
        with self.assertRaisesRegex(ValueError,'decision|identity|shader'):
            e.verify(root)


if __name__ == '__main__':
    unittest.main()
