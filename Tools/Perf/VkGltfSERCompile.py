"""Measure isolated RT material/SER compiler or native-PSO costs.

The plan maps role -> O/G/S -> {recipe, stages}. Native mode needs only recipe;
compile mode also needs per-stage {profile, capabilities, defines, search_paths}.
CLI compile wall time is attributable only when all four outputs exactly match
the actual captured PSO's finalized SPIR-V. Native mode excludes Slang and SBT.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import VkGltfSER as experiment

ROOT = experiment.ROOT
read, save, digest, require = experiment.read, experiment.save, experiment.digest, experiment.require


def statistics_for(rows, field):
    result = {}
    for role in sorted({r['role'] for r in rows}):
        values = {arm: [r[field] for r in rows if r['role'] == role and r['arm'] == arm] for arm in experiment.ARMS}
        require(all(len(v) == 3 and all(math.isfinite(x) and x > 0 for x in v) for v in values.values()),
                'Three positive independent samples required per arm')
        contrasts = {}
        for label, left, right in [('material', 'O', 'G'), ('ser', 'G', 'S')]:
            delta = [100 * (b / a - 1) for a, b in zip(values[left], values[right])]
            contrasts[label] = dict(left=left, right=right, paired_percent=delta,
                                   paired_median_percent=statistics.median(delta))
        result[role] = dict(values_ms=values, medians_ms={a: statistics.median(v) for a, v in values.items()}, contrasts=contrasts)
    return result


def load_plan(path):
    plan = read(path)
    require(plan.get('schema_version') == 1 and isinstance(plan.get('roles'), dict) and plan['roles'], 'Invalid plan')
    identities = {}
    for role, arms in plan['roles'].items():
        require(role.replace('-', '').isalnum() and set(arms) == set(experiment.ARMS), 'Role requires exactly O/G/S')
        identities[role] = {}
        for arm, item in arms.items():
            identity = experiment.native.recipe_identity(Path(item['recipe']).resolve(strict=True), 'ray_tracing')
            main = identity['recipe']['shaders']['ray_generation']
            require(main['module_name'] in experiment.traversal.RT_MODULES if arm == 'O' else
                    main['module_name'] == experiment.VK_MODULE, 'Wrong recipe source for arm')
            raw = (Path(identity['path']).parent / main['spirv_file']).read_bytes()
            ser = 'SPV_EXT_shader_invocation_reorder' in experiment.spirv_extensions(raw)
            require(ser is experiment.ARMS[arm]['shaderExecutionReorder'], 'Actual captured SER arm differs from plan')
            require(identity['recipe']['state'].get('enable_invocation_reorder', False) is ser,
                    'Native SER state differs from SPIR-V')
            if ser:
                require(identity['recipe']['source_device']['enabled_features'].get('ray_tracing_invocation_reorder') is True,
                        'EXT SER device feature not enabled')
            identities[role][arm] = identity
        sources = [identities[role][arm]['recipe']['source_device'] for arm in experiment.ARMS]
        require(sources[0] == sources[1] == sources[2], 'Capture device/features differ')
        for field in ('uses_bindless_heap', 'bindless_user_push_data_size', 'disable_optimization', 'max_recursion_depth',
                      'cluster_acceleration_structure'):
            values = [identities[role][arm]['recipe']['state'][field] for arm in experiment.ARMS]
            require(values[0] == values[1] == values[2], f'Shared pipeline ABI/policy changed: {field}')
        shared = []
        for arm in ('G', 'S'):
            shaders = {}
            for stage in ('miss', 'closest_hit', 'any_hit'):
                shader = dict(identities[role][arm]['recipe']['shaders'][stage])
                payload = Path(identities[role][arm]['path']).parent / shader.pop('spirv_file')
                shader.update(sha256=digest(payload), extensions=experiment.spirv_extensions(payload.read_bytes()))
                shaders[stage] = shader
            shared.append(shaders)
        experiment.require_shared_hit_stages(*shared)
    return plan, identities


def compile_command(compiler, stage, request, destination):
    require(set(request) <= {'profile', 'capabilities', 'defines', 'search_paths'}, 'Unknown compile request field')
    paths = [ROOT / 'Shaders', *[Path(p).resolve(strict=True) for p in request.get('search_paths', [])],
             ROOT / 'Shaders/Modules', ROOT / 'Shaders/Interop']
    command = [str(compiler), str(ROOT / 'Shaders' / (stage['module_name'] + '.slang')),
        '-module-name', stage['module_name'], '-entry', stage['source_entry_point'], '-target', 'spirv',
        '-profile', request.get('profile', 'spirv_1_6'), '-emit-spirv-directly', '-matrix-layout-row-major',
        '-O1', '-o', str(destination)]
    for path in paths:
        command += ['-I', str(path)]
    for capability in request.get('capabilities', []):
        require(isinstance(capability, str) and capability, 'Invalid capability')
        command += ['-capability', capability]
    for name, value in request.get('defines', []):
        require(isinstance(name, str) and name and isinstance(value, str), 'Invalid define')
        command += ['-D', f'{name}={value}']
    if not any(name == 'METALLIC_WORKING_SPACE_ACESCG' for name, _ in request.get('defines', [])):
        command += ['-D', 'METALLIC_WORKING_SPACE_ACESCG=1']
    return command


def compile_stage(command, destination, timeout, reference, environment):
    destination.mkdir(parents=True)
    record = dict(command=command, started=time.time(), exitCode=None, timedOut=False,
                  reference_sha256=digest(reference), reference_file=str(reference))
    begin = time.perf_counter()
    process = None
    try:
        with (destination / 'stdout.log').open('wb') as stdout, (destination / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
            record['pid'] = process.pid
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
        record.update(wall_ms=(time.perf_counter() - begin) * 1000, ended=time.time())
        save(destination / 'Process.json', record)
    require(record['exitCode'] == 0 and not record['timedOut'], f'Compiler failed: {destination}')
    emitted = destination / 'shader.spv'
    record.update(emitted_sha256=digest(emitted), emitted_bytes=emitted.stat().st_size,
                  matches_runtime_spirv=emitted.read_bytes() == reference.read_bytes())
    save(destination / 'Process.json', record)
    return record


def run(args):
    output = args.output.resolve()
    require(not output.exists() and math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Fresh output/bounded timeout required')
    executable = args.executable.resolve(strict=True)
    plan_path = args.plan.resolve(strict=True)
    plan, identities = load_plan(plan_path)
    paths = {executable, plan_path, Path(__file__).resolve(), Path(experiment.__file__).resolve(),
             Path(experiment.native.__file__).resolve(), Path(experiment.driver.__file__).resolve(),
             Path(experiment.frontend.__file__).resolve(), Path(experiment.baseline.__file__).resolve(),
             Path(experiment.experiment.__file__).resolve(), Path(experiment.traversal.__file__).resolve()}
    paths.update(executable.parent.glob('*.dll'))
    for arms in identities.values():
        for identity in arms.values():
            paths.update(Path(p) for p in identity['files'])
    if args.mode == 'compile':
        paths.update((ROOT / 'Shaders').rglob('*.slang'))
        paths.update((ROOT / 'Shaders').rglob('*.hlsli'))
        paths.update((ROOT / 'Shaders').rglob('*.h'))
        paths.update((ROOT / 'Shaders').rglob('*.inl'))
        paths.update((ROOT / 'Source').rglob('*.inl'))
        paths.add(ROOT / 'Source/Runtime/Render/Core/ShaderRequests.h')
        for role, arms in plan['roles'].items():
            for arm, item in arms.items():
                require(set(item.get('stages', {})) == set(experiment.traversal.RT_STAGES), 'Compile plan requires all four stage requests')
                for request in item['stages'].values():
                    for folder in request.get('search_paths', []):
                        paths.update(p for p in Path(folder).rglob('*') if p.is_file())
    frozen = experiment.experiment.snapshots(paths)
    environment, removed = experiment.traversal.clean_environment()
    environment['METALLIC_VK_INTERNAL_PIPELINE_CACHE'] = 'disabled'
    summary = dict(schema_version=1, mode=args.mode, status='running', plan=plan, identities=identities,
        frozen=frozen, executable=str(executable), orders=experiment.ORDER, rows=[],
        scope='Pure native vkCreateRayTracingPipelinesKHR; no Slang or SBT.' if args.mode == 'native' else
            'Sum of four sequential fresh slangc process wall times; includes process/module loading, not native PSO.',
        cache_policy='Native: empty application VkPipelineCache, controllable driver cache disabled. Compiler/OS state not guaranteed cold.',
        environment={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))}, removed_environment=removed)
    with experiment.traversal.experiment_lock(output):
        output.mkdir(parents=True)
        shutil.copy2(__file__, output / 'VkGltfSERCompile.py')
        save(output / 'Plan.json', plan)
        save(output / 'Summary.json', summary)
        try:
            for block, order in enumerate(experiment.ORDER):
                for role, arms in identities.items():
                    for arm in order:
                        experiment.experiment.check_snapshots(frozen)
                        destination = output / f'{block:02}-{role}-{arm}'
                        print(f'{args.mode}/{block}/{role}/{arm}', flush=True)
                        if args.mode == 'native':
                            command = [str(executable), '--recipe-file', arms[arm]['path'], '--iterations', '1',
                                '--pipeline-cache', 'empty', '--debug-mode', 'disabled', '--output-dir', str(destination)]
                            process, report = experiment.native.run_process(command, environment, destination, args.timeout)
                            sample = experiment.driver.single_sample(report, native=True)
                            summary.setdefault('device', report['device'])
                            experiment.driver.verify_measurement(report, experiment.driver.shader_identity(report, sample),
                                summary['device'], disabled=True)
                            row = dict(process=process, value_ms=sample['pipeline']['native_create_ms'])
                        else:
                            stages = {}
                            for stage_name in experiment.traversal.RT_STAGES:
                                stage = arms[arm]['recipe']['shaders'][stage_name]
                                stage_dir = destination / stage_name
                                request = plan['roles'][role][arm]['stages'][stage_name]
                                command = compile_command(executable, stage, request, stage_dir / 'shader.spv')
                                reference = Path(arms[arm]['path']).parent / stage['spirv_file']
                                stages[stage_name] = compile_stage(command, stage_dir, args.timeout, reference, environment)
                            row = dict(stages=stages, value_ms=sum(s['wall_ms'] for s in stages.values()))
                        summary['rows'].append(dict(block=block, role=role, arm=arm, directory=destination.name, **row))
                        save(output / 'Summary.json', summary)
                        experiment.experiment.check_snapshots(frozen)
            summary['statistics'] = statistics_for(summary['rows'], 'value_ms')
            matching = args.mode == 'native' or all(s['matches_runtime_spirv'] for r in summary['rows'] for s in r['stages'].values())
            summary.update(status='passed' if matching else 'bytecode_mismatch', production_attribution_qualified=matching)
        except BaseException as error:
            summary.update(status='failed', error=repr(error))
            raise
        finally:
            save(output / 'Summary.json', summary)
            experiment.seal(output)
    return verify(output, allow_bytecode_mismatch=True)


def verify(output, allow_bytecode_mismatch=False):
    output = Path(output).resolve(strict=True)
    experiment.verify_seal(output)
    summary = read(output / 'Summary.json')
    require(summary['status'] == 'passed' or (allow_bytecode_mismatch and summary['status'] == 'bytecode_mismatch'),
            'Failed/incomplete or unmatched production bytecode')
    require(summary['orders'] == list(experiment.ORDER), 'Process order changed')
    current_verifier = Path(__file__).resolve()
    experiment.experiment.check_snapshots({p: h for p, h in summary['frozen'].items() if Path(p).resolve() != current_verifier})
    plan, identities = load_plan(output / 'Plan.json')
    require(plan == summary['plan'] and identities == summary['identities'], 'Captured inputs changed')
    expected = [(block, role, arm) for block, order in enumerate(experiment.ORDER) for role in identities for arm in order]
    require([(r['block'], r['role'], r['arm']) for r in summary['rows']] == expected, 'Incomplete process sequence')
    last_end, stable, bytecode_match = 0, {}, True
    for row in summary['rows']:
        destination = output / row['directory']
        if summary['mode'] == 'native':
            process, report = read(destination / 'Process.json'), read(destination / 'report.json')
            require(process == row['process'], 'Process record changed')
            processes = [process]
            sample = experiment.driver.single_sample(report, native=True)
            key = row['role'], row['arm']
            stable.setdefault(key, experiment.driver.shader_identity(report, sample))
            experiment.driver.verify_measurement(report, stable[key], summary['device'], disabled=True)
            pipeline = sample['pipeline']
            require(pipeline['native_create_ms'] == row['value_ms'] and pipeline['pso_hash'] ==
                experiment.native.expected_pso_hash(identities[row['role']][row['arm']]['recipe']), 'Wrong PSO/timing')
            require(not any(pipeline.get(k, False) for k in ('native_cache_hit', 'metadata_hit', 'native_object_reused',
                'native_binary_cache_hit', 'native_binary_driver_rejected')), 'Native cache contamination')
            require(not report['configuration']['validation'] and not report['configuration']['disable_driver_optimization'],
                    'Diagnostic timing contamination')
        else:
            processes = []
            for stage_name in experiment.traversal.RT_STAGES:
                stage_dir = destination / stage_name
                process = read(stage_dir / 'Process.json')
                require(process == row['stages'][stage_name], 'Compiler process record changed')
                item = identities[row['role']][row['arm']]
                stage = item['recipe']['shaders'][stage_name]
                command = compile_command(Path(summary['executable']), stage,
                    plan['roles'][row['role']][row['arm']]['stages'][stage_name], stage_dir / 'shader.spv')
                require(command == process['command'], 'Compiler command differs from frozen plan')
                reference = Path(item['path']).parent / stage['spirv_file']
                emitted = stage_dir / 'shader.spv'
                equal = emitted.read_bytes() == reference.read_bytes()
                require(digest(reference) == process['reference_sha256'] and digest(emitted) == process['emitted_sha256'] and
                    emitted.stat().st_size == process['emitted_bytes'] and equal is process['matches_runtime_spirv'], 'Compiler bytecode changed')
                bytecode_match &= equal
                processes.append(process)
            require(sum(p['wall_ms'] for p in processes) == row['value_ms'], 'Compiler timing aggregation changed')
        for process in processes:
            require(process['exitCode'] == 0 and not process['timedOut'] and process['started'] >= last_end and
                    process['ended'] >= process['started'], 'Failed/overlapping child process')
            last_end = process['ended']
    require(statistics_for(summary['rows'], 'value_ms') == summary['statistics'], 'Statistics differ from raw processes')
    require(bytecode_match is summary['production_attribution_qualified'] and bytecode_match is (summary['status'] == 'passed'),
            'Production attribution status contradicts bytecode')
    archived = output / 'VkGltfSERCompile.py'
    require(digest(archived) == summary['frozen'][str(Path(__file__).resolve())], 'Archived runner changed')
    return dict(verified=True, mode=summary['mode'], production_attribution_qualified=bytecode_match,
        independent_samples=len(summary['rows']), statistics=summary['statistics'], manifest_sha256=digest(output / 'Manifest.json'),
        captured_runner_sha256=summary['frozen'][str(current_verifier)], current_verifier_sha256=digest(current_verifier))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('run')
    capture.add_argument('--mode', choices=('compile', 'native'), required=True)
    capture.add_argument('--plan', type=Path, required=True)
    capture.add_argument('--executable', type=Path, required=True, help='slangc for compile; MetallicShaderPipelineBench for native')
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--timeout', type=float, default=600)
    check = sub.add_parser('verify')
    check.add_argument('output', type=Path)
    check.add_argument('--allow-bytecode-mismatch', action='store_true')
    args = parser.parse_args()
    result = run(args) if args.command == 'run' else verify(args.output, args.allow_bytecode_mismatch)
    print(json.dumps(result, indent=2))
    if not result['production_attribution_qualified']:
        parser.exit(2, 'CLI bytecode differs from production; timings cannot be attributed to the captured production PSO.\n')


if __name__ == '__main__':
    main()
