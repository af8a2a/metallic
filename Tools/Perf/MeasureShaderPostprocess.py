"""Measure fresh shader-cache misses independently of native Pipeline creation.

Every process compiles one catalog request into a fresh cache. Exact output
bytes, compiler phases, inputs and the real schedule are retained. Three-arm
blocks rotate their starting position; two-arm blocks alternate AB/BA.
"""
import argparse
import math
from pathlib import Path
import shutil

import ProfileComputeShaderFrontend as frontend
import RunComputePipelineExperiments as experiment


def order(conditions, block):
    if len(conditions) == 2:
        return list(reversed(conditions)) if block % 2 else list(conditions)
    shift = block % len(conditions)
    return conditions[shift:] + conditions[:shift]


def summarize(runs):
    result = {}
    for run in runs:
        key = (run['request_index'], run['arm']['postprocess'])
        result.setdefault(key, []).append(run)
    return [dict(request_index=request, postprocess=mode,
        compile_wall_ms=experiment.driver.timings([run['compile_wall_ms'] for run in rows]),
        code_generation_ms=experiment.driver.timings([run['phases']['code_generation_ms'] for run in rows]),
        postprocess_ms=experiment.driver.timings([run['phases']['postprocess_ms'] for run in rows]),
        output=rows[0]['output']) for (request, mode), rows in result.items()]


def finite_metric(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or \
        (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')


def frontend_sample(report, request_index, postprocess):
    sample = experiment.driver.single_sample(report, native=False)
    configuration = report['configuration']
    phases = sample.get('compilation_phases', {})
    if not configuration['compile_only'] or configuration['shader_cache'] != 'reuse' or \
        sample.get('shader_cache_hit') is not False or phases.get('compiler_executed') is not True:
        raise ValueError('Expected a fresh source compilation without native creation')
    if sample['request']['index'] != request_index or sample['pipeline_status'] != 'compile_only':
        raise ValueError('Wrong request or native scope')
    if configuration['slang_optimization'] != 'O1' or sample['effective_slang_optimization'] != 'O1' or \
        configuration['debug_mode'] != 'disabled' or configuration['validation'] or \
        sample['effective_descriptor_mode'] != 'mapped':
        raise ValueError('Frontend optimization, debug, validation or descriptor settings differ')
    experiment.postprocess_policy(report, sample, postprocess)
    finite_metric(sample.get('compile_wall_ms'), 'compile_wall_ms', positive=True)
    for name in ('code_generation_ms', 'postprocess_ms'):
        finite_metric(phases.get(name), name)
    return sample


def workload_identity(shader):
    # Output and policy/tool fields deliberately differ between transforms.
    # Request, SDK/build, target, debug/validation and descriptor ABI must match.
    variable_fields = {'spirv_bytes', 'spirv_fnv1a64', 'spirv_statistics',
                       'spirv_postprocess', 'effective_spirv_postprocess', 'spirv_postprocess_tool_version'}
    return {name: value for name, value in shader.items() if name not in variable_fields}


def verify(path):
    path = Path(path).resolve(strict=True)
    expected = experiment.read(path / 'Manifest.json')
    actual = {str(file.relative_to(path)).replace('\\', '/'): experiment.digest(file)
        for file in path.rglob('*') if file.is_file() and file != path / 'Manifest.json'}
    if actual != expected:
        raise ValueError('Frontend evidence inventory changed')
    summary = experiment.read(path / 'summary.json')
    if summary['status'] != 'passed':
        raise ValueError('Frontend experiment incomplete')
    workloads = {}
    if summary.get('schema_version', 1) >= 2:
        discoveries = summary.get('excluded_runs', [])
        if [run['request_index'] for run in discoveries] != summary['request_indices']:
            raise ValueError('Missing or duplicated dependency discovery')
        for discovery in discoveries:
            process = discovery['process']
            if process['exit_code'] or process['timed_out'] or process['native_crash']:
                raise ValueError('Unsuccessful dependency discovery')
            report = experiment.read(discovery['report'])
            sample = frontend_sample(report, discovery['request_index'], 'none')
            dependencies = {str(Path(path).resolve()) for path in sample['dependencies']}
            if sample['request']['index'] != discovery['request_index'] or not dependencies.issubset(summary['inputs']):
                raise ValueError('Discovered source dependency was not frozen')
            workloads[discovery['request_index']] = workload_identity(experiment.shader_identity(report, sample))
    schedule = [(request, block, arm) for request in summary['request_indices']
        for block in range(summary['pairs']) for arm in order(summary['conditions'], block)]
    runs = summary['runs']
    if [(run['request_index'], run['block'], run['arm']) for run in runs] != schedule:
        raise ValueError('Missing, duplicated or reordered compiler sample')
    outputs = {}
    compiler_identities = {}
    for run in runs:
        process = run['process']
        if process['exit_code'] or process['timed_out'] or process['native_crash']:
            raise ValueError('Unsuccessful compiler process')
        report = experiment.read(run['report'])
        sample = frontend_sample(report, run['request_index'], run['arm']['postprocess'])
        if summary.get('schema_version', 1) >= 2:
            dependencies = {str(Path(path).resolve()) for path in sample['dependencies']}
            if not dependencies.issubset(summary['inputs']):
                raise ValueError('Measured source dependency was not frozen')
        if sample['compile_wall_ms'] != run['compile_wall_ms'] or sample['compilation_phases'] != run['phases']:
            raise ValueError('Compiler metric summary differs from raw report')
        shader = experiment.shader_identity(report, sample)
        if shader != run['shader_identity']:
            raise ValueError('Compiler settings or bytecode identity changed')
        key = (run['request_index'], run['arm']['postprocess'])
        if compiler_identities.setdefault(key, shader) != shader:
            raise ValueError('Repeated compilation changed compiler policy or tool identity')
        workload = workload_identity(shader)
        if workloads.setdefault(run['request_index'], workload) != workload:
            raise ValueError('Compiler request, SDK, target or descriptor identity drifted across modes/discovery')
        _, raw = frontend.bench_bytes(report, sample)
        output = dict(bytes=len(raw), sha256=frontend.sha(Path(run['spirv'])), fnv1a64=frontend.fnv(raw))
        if raw != Path(run['spirv']).read_bytes() or output != run['output']:
            raise ValueError('Captured output differs from shader cache payload')
        if outputs.setdefault(key, output) != output:
            raise ValueError('Repeated compilation produced different output bytes')
    if summarize(runs) != summary['aggregate']:
        raise ValueError('Compiler aggregate differs from raw samples')
    return summary


def run(args):
    executable = args.executable.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    conditions = experiment.arms(['O1'], ['pipeline'], 'normal', args.postprocesses)
    identity = experiment.input_identity(executable, [Path(__file__), Path(frontend.__file__)], production=True)
    summary = dict(schema_version=2, status='running', pairs=args.pairs, conditions=conditions,
        request_indices=args.request_indices, inputs=identity, runs=[], excluded_runs=[],
        scope='Independent fresh shader-cache misses. Compile-only, mapped O1, debug/validation disabled; no device, native Pipeline or GPU work.')
    experiment.save(output / 'summary.json', summary)
    try:
        with frontend.experiment_lock():
            base = experiment.clean_environment()
            for request in args.request_indices:
                # Discover compiler-reported vendor/interop dependencies before
                # measured work. This compilation is excluded from all timings.
                discovery = output / f'request-{request}-dependency-discovery'
                baseline = experiment.arms(['O1'], ['pipeline'], 'normal', ['none'])[0]
                command = [str(executable), '--request-index', str(request), '--iterations', '1', '--compile-only',
                    '--descriptor-mode', 'mapped', '--debug-mode', 'disabled', '--slang-optimization', 'O1',
                    '--spirv-postprocess', 'none', '--shader-cache', 'reuse', '--pipeline-cache', 'empty',
                    '--cache-dir', str(discovery / 'shader-cache'), '--output-dir', str(discovery)]
                process, report = experiment.driver.run_process(command, experiment.environment_for(baseline, base), discovery, args.timeout)
                sample = experiment.driver.single_sample(report, native=False)
                experiment.check_snapshots(identity)
                identity.update(experiment.snapshots(map(Path, sample['dependencies'])))
                summary['excluded_runs'].append(dict(request_index=request, reason='dependency discovery', process=process,
                    report=str(discovery / 'report.json')))
                experiment.save(output / 'summary.json', summary)
                for block in range(args.pairs):
                    for arm in order(conditions, block):
                        experiment.check_snapshots(identity)
                        destination = output / f"request-{request}-block-{block:02}-{arm['name']}"
                        cache = destination / 'shader-cache'
                        if destination.exists():
                            raise ValueError('Compiler sample directory must be fresh')
                        command = [str(executable), '--request-index', str(request), '--iterations', '1', '--compile-only',
                            '--descriptor-mode', 'mapped', '--debug-mode', 'disabled', '--slang-optimization', 'O1',
                            '--spirv-postprocess', arm['postprocess'], '--shader-cache', 'reuse', '--pipeline-cache', 'empty',
                            '--cache-dir', str(cache), '--output-dir', str(destination)]
                        process, report = experiment.driver.run_process(command, experiment.environment_for(arm, base), destination, args.timeout)
                        sample = experiment.driver.single_sample(report, native=False)
                        dependencies = {str(Path(path).resolve()) for path in sample['dependencies']}
                        if not dependencies.issubset(identity):
                            raise ValueError('Measured compilation introduced an unfrozen source dependency')
                        if sample['shader_cache_hit']:
                            raise ValueError('Fresh compiler sample unexpectedly hit a shader cache')
                        experiment.postprocess_policy(report, sample, arm['postprocess'])
                        _, raw = frontend.bench_bytes(report, sample)
                        spirv = destination / 'Output.spv'
                        spirv.write_bytes(raw)
                        row = dict(request_index=request, block=block, arm=arm, process=process,
                            report=str(destination / 'report.json'), spirv=str(spirv),
                            compile_wall_ms=sample['compile_wall_ms'], phases=sample['compilation_phases'],
                            shader_identity=experiment.shader_identity(report, sample),
                            output=dict(bytes=len(raw), sha256=frontend.sha(spirv), fnv1a64=frontend.fnv(raw)))
                        summary['runs'].append(row)
                        experiment.check_snapshots(identity)
                        experiment.save(output / 'summary.json', summary)
                        print(request, arm['postprocess'], sample['compile_wall_ms'], len(raw), flush=True)
            summary['aggregate'] = summarize(summary['runs'])
            summary['status'] = 'passed'
            experiment.save(output / 'summary.json', summary)
            experiment.seal(output)
            verify(output)
    except Exception as error:
        summary.update(status='failed', error=str(error))
        experiment.save(output / 'summary.json', summary)
        raise
    print(output / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    measure = sub.add_parser('run')
    measure.add_argument('--executable', type=Path, required=True)
    measure.add_argument('--output-dir', type=Path, required=True)
    measure.add_argument('--request-indices', nargs='+', type=int, required=True)
    measure.add_argument('--postprocesses', nargs='+', choices=experiment.POSTPROCESSES,
        default=['none', 'dead-branches', 'slang-dead-branches'])
    measure.add_argument('--pairs', type=int, default=3)
    measure.add_argument('--timeout', type=float, default=600)
    check = sub.add_parser('verify')
    check.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        summary = verify(args.output)
        print(summary['status'], len(summary['runs']), 'compiler processes verified')
        return
    if args.pairs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(index < 0 for index in args.request_indices) or \
        len(set(args.request_indices)) != len(args.request_indices):
        parser.error('Positive pairs/timeout and distinct nonnegative request indices are required')
    run(args)


if __name__ == '__main__':
    main()
