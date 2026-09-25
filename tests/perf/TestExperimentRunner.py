"""Decision/recovery negative tests; synthetic timings never serve as GPU evidence."""
import copy
import datetime
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Tools/Perf'))
import ExperimentRunner as e
import TestWorkloadCase as fixtures


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkloadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.analyze()
        self.case = self.fixture.case
        self.policy = {'minimumGain': .03, 'maximumGraphRegression': .02}
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
        manifest = {'protocol':e.PROTOCOL,'status':'complete','policy':e.POLICY,'blocks':2,
                    'confirmationBlocks':2,'restored':True,'runs':[],
                    'buildIdentity':{'configuration':'Release','cacheSha256':e.w.digest(root/'CMakeCache.txt')},
                    'toolHashes':{}}
        for name in ('ExperimentRunner.py','WorkloadCase.py','MeasureExperimentGpu.ps1'):
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
            decision = e.assess(rows,self.case,e.POLICY,8)
            self.assertEqual(decision['decision'],'accept')
            e.w.save(root/(stage+'-Decision.json'),decision)
        e.w.save(root/'Decision.json',decision)
        manifest['artifacts'] = e.artifact_hashes(root)
        e.w.save(root/'Manifest.json',manifest)
        self.assertEqual(e.verify(root)['decision'],'accept')
        manifest['runs'] = manifest['runs'][:8]
        e.w.save(root/'Manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'confirmation'):
            e.verify(root)


if __name__ == '__main__':
    unittest.main()
