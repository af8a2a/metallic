"""Compare captured RayQuery and ray-tracing PSOs in balanced, fresh processes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import shutil
import time

import MaterialBaseline as mb
import ProfileComputeShaderFrontend as frontend
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver

ROOT = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise ValueError(message)


def expected_pso_hash(recipe):
    # Recipes encode the uint64 hash as hex; benchmark JSON uses a uint64 number.
    value = recipe['expected_pso_hash']
    require(isinstance(value, str) and len(value) == 16 and
            all(c in '0123456789abcdef' for c in value), 'Invalid recipe PSO hash')
    return int(value, 16)


def recipe_identity(path, pipeline_type):
    recipe = mb.read(path)
    require(recipe['pipeline_type'] == pipeline_type, 'Wrong captured pipeline family')
    expected = {'compute'} if pipeline_type == 'compute' else {'ray_generation', 'miss', 'closest_hit', 'any_hit'}
    stages = {k: v for k, v in recipe['shaders'].items() if v['spirv_file'] is not None}
    require(set(stages) == expected, 'Missing or unexpected shader stage')
    files = {str(path): mb.digest(path)}
    for name, stage in stages.items():
        blob = (path.parent / stage['spirv_file']).resolve(strict=True)
        require(blob.parent == path.parent, 'Recipe blob escaped sibling directory')
        raw = blob.read_bytes()
        require(len(raw) == stage['spirv_bytes'] and frontend.fnv(raw) == stage['input_spirv_hash'], 'Changed recipe blob')
        require(stage['slang_optimization'] == 'O1' and stage['spirv_postprocess'] == 'none' and
                stage['descriptor_heap_mode'] == 'mapped' and not stage['source_identity_ambiguous'], 'Unqualified shader policy')
        files[str(blob)] = mb.digest(blob)
    require(not recipe['state']['disable_optimization'], 'Driver optimization disabled')
    return dict(path=str(path), recipe=recipe, files=files)


def run_process(command, environment, destination, timeout):
    destination.mkdir(parents=True, exist_ok=False)
    record = dict(command=command, started=time.time(), exitCode=None, timedOut=False)
    process = None
    with (destination / 'stdout.log').open('wb') as stdout, (destination / 'stderr.log').open('wb') as stderr:
        try:
            with driver.suppress_windows_error_dialogs():
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            record['pid'] = process.pid
            mb.save(destination / 'Process.json', record)
            try:
                record['exitCode'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record['timedOut'] = True
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    record['terminatedByRunner'] = True
                record['exitCode'] = process.wait(timeout=15)
            record['ended'] = time.time()
            mb.save(destination / 'Process.json', record)
    require(record['exitCode'] == 0 and not record['timedOut'], f'Native process failed: {destination}')
    return record, mb.read(destination / 'report.json')


def summarize(rows):
    result = {}
    for role in sorted({r['role'] for r in rows}):
        values = {arm: [r['nativeCreateMs'] for r in rows if r['role'] == role and r['arm'] == arm] for arm in 'AB'}
        changes = [100 * (b / a - 1) for a, b in zip(values['A'], values['B'])]
        result[role] = dict(valuesMs=values, mediansMs={a: statistics.median(v) for a, v in values.items()},
                            pairedChangesPercent=changes, pairedMedianChangePercent=statistics.median(changes))
    return result


def verify(output):
    output = Path(output).resolve(strict=True)
    require(mb.read(output / 'Manifest.json') == {p.relative_to(output).as_posix(): mb.digest(p)
        for p in output.rglob('*') if p.is_file() and p.name != 'Manifest.json'}, 'Evidence manifest changed')
    summary = mb.read(output / 'Summary.json')
    require(summary['status'] == 'passed', 'Incomplete native comparison')
    require(summary['orders'] == ['AB', 'BA', 'AB'], 'Native comparison order changed')
    # The capture freezes its runner while measuring. Verification uses the
    # current maintained implementation, which may fix verifier-only defects.
    runner = Path(__file__).resolve()
    require(all(mb.digest(Path(p)) == h for p, h in summary['frozenFiles'].items()
                if Path(p).resolve() != runner), 'Native inputs changed')
    archived_runner = output / 'PathTraceTraversalPSO.py'
    if archived_runner.exists():
        require(mb.digest(archived_runner) == summary['frozenFiles'][str(runner)], 'Captured runner changed')
    for role, arms in summary['inputs'].items():
        for arm, kind in [('A', 'compute'), ('B', 'ray_tracing')]:
            require(recipe_identity(Path(arms[arm]['path']), kind) == arms[arm], 'Captured recipe identity changed')
    expected = [(block, role, arm) for block, order in enumerate(summary['orders']) for role in summary['inputs'] for arm in order]
    require([(r['block'], r['role'], r['arm']) for r in summary['rows']] == expected, 'Incomplete process sequence')
    identities = {}
    previous_end = 0
    for row in summary['rows']:
        destination = output / row['directory']
        process, report = mb.read(destination / 'Process.json'), mb.read(destination / 'report.json')
        require(process['exitCode'] == 0 and not process['timedOut'] and process['started'] >= previous_end, 'Invalid/overlapping process')
        previous_end = process['ended']
        sample = driver.single_sample(report, native=True)
        key = row['role'], row['arm']
        identity = driver.shader_identity(report, sample)
        identities.setdefault(key, identity)
        driver.verify_measurement(report, identities[key], summary['device'], disabled=True)
        pipeline = sample['pipeline']
        require(math.isfinite(pipeline['native_create_ms']) and pipeline['native_create_ms'] > 0, 'Invalid native duration')
        require(pipeline['native_create_ms'] == row['nativeCreateMs'] and pipeline['pso_hash'] == row['psoHash'], 'Native result changed')
        require(pipeline['pso_hash'] == expected_pso_hash(summary['inputs'][row['role']][row['arm']]['recipe']), 'Wrong replayed PSO')
        require(not any(pipeline.get(k, False) for k in ('native_cache_hit', 'metadata_hit', 'native_object_reused',
            'native_binary_cache_hit', 'native_binary_driver_rejected')), 'Cache contamination')
        require(not report['configuration']['validation'] and not report['configuration']['disable_driver_optimization'], 'Diagnostic timing')
    require(summarize(summary['rows']) == summary['statistics'], 'Statistics differ from raw reports')
    return dict(status='passed', independentProcesses=len(summary['rows']), statistics=summary['statistics'],
                capturedRunnerSha256=summary['frozenFiles'][str(runner)], currentVerifierSha256=mb.digest(runner),
                runnerArchivePresent=archived_runner.exists())


def run(args):
    output = args.output.resolve()
    require(not output.exists() and math.isfinite(args.timeout) and args.timeout > 0, 'Use a new directory and positive timeout')
    executable = args.exe.resolve(strict=True)
    selected, frozen = {}, {str(executable): mb.digest(executable), str(Path(__file__)): mb.digest(Path(__file__))}
    for dependency in (*executable.parent.glob('*.dll'), Path(mb.__file__), Path(driver.__file__), Path(frontend.__file__), Path(experiment.__file__)):
        frozen[str(dependency)] = mb.digest(dependency)
    for role, a, b in args.pair:
        require(role not in selected and role.replace('-', '').isalnum(), 'Invalid/duplicate role')
        selected[role] = {arm: recipe_identity(Path(p).resolve(strict=True), kind)
                          for arm, p, kind in [('A', a, 'compute'), ('B', b, 'ray_tracing')]}
        left, right = [selected[role][arm]['recipe'] for arm in 'AB']
        for field in ('uses_bindless_heap', 'bindless_user_push_data_size', 'disable_optimization'):
            require(left['state'][field] == right['state'][field], 'Shared pipeline ABI/policy changed')
        require(left['shaders']['compute'].get('specialization_constants', []) ==
                right['shaders']['ray_generation'].get('specialization_constants', []), 'Integrator specialization differs')
        require(left['source_device'] == right['source_device'], 'Capture device/features differ')
        for identity in selected[role].values():
            frozen.update(identity['files'])
    environment = experiment.clean_environment()
    environment['METALLIC_VK_INTERNAL_PIPELINE_CACHE'] = 'disabled'
    summary = dict(status='running', inputs=selected, frozenFiles=frozen, rows=[], orders=['AB', 'BA', 'AB'],
        scope='One native vkCreateComputePipelines (A) or vkCreateRayTracingPipelinesKHR (B) call per process. No Slang compile/GPU dispatch.',
        cachePolicy='Empty application cache; driver internal cache control disabled. Compiler and OS state are not guaranteed cold.')
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / 'PathTraceTraversalPSO.py')
    mb.save(output / 'Summary.json', summary)
    try:
        with frontend.experiment_lock():
            identities = {}
            for block, order in enumerate(summary['orders']):
                for role, arms in selected.items():
                    for arm in order:
                        require(all(mb.digest(Path(p)) == h for p, h in frozen.items()), 'Native inputs changed')
                        destination = output / f'{block:02}-{role}-{arm}'
                        command = [str(executable), '--recipe-file', arms[arm]['path'], '--iterations', '1',
                            '--pipeline-cache', 'empty', '--debug-mode', 'disabled', '--output-dir', str(destination)]
                        print(f'Native {block} {role} {arm}', flush=True)
                        process, report = run_process(command, environment, destination, args.timeout)
                        sample = driver.single_sample(report, native=True)
                        identities.setdefault((role, arm), driver.shader_identity(report, sample))
                        summary.setdefault('device', report['device'])
                        driver.verify_measurement(report, identities[(role, arm)], summary['device'], disabled=True)
                        pipeline = sample['pipeline']
                        summary['rows'].append(dict(block=block, role=role, arm=arm, directory=destination.name,
                            nativeCreateMs=pipeline['native_create_ms'], driverMs=pipeline.get('driver_ms'),
                            psoHash=pipeline['pso_hash'], process=process))
                        mb.save(output / 'Summary.json', summary)
            summary['statistics'] = summarize(summary['rows'])
            summary['status'] = 'passed'
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        mb.save(output / 'Summary.json', summary)
        mb.seal(output)
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--pair', nargs=3, action='append', required=True, metavar=('ROLE', 'RQ_RECIPE', 'RT_RECIPE'))
    runner.add_argument('--exe', type=Path, required=True)
    runner.add_argument('--output', type=Path, required=True)
    runner.add_argument('--timeout', type=float, default=300)
    checker = sub.add_parser('verify')
    checker.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args) if args.command == 'run' else verify(args.output.resolve()), indent=2))
