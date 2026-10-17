"""Four-block native PSO/compiler comparison for direct vs generic glTF BSDF."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

import SimpleBSDFGeneric as evidence
import VkGltfSERCompile as shared_compile

ROOT = evidence.ROOT
read, save, digest, require = evidence.read, evidence.save, evidence.digest, evidence.require
compile_command, compile_stage = shared_compile.compile_command, shared_compile.compile_stage
ROLE_CASES = {'single': 'Opaque1', 'general': 'Mixed4'}


def statistics_for(rows):
    result = {}
    for role in ROLE_CASES:
        values = {}
        for arm in evidence.ARMS:
            selected = [r for r in rows if r['role'] == role and r['arm'] == arm]
            require([r['block'] for r in selected] == list(range(len(evidence.ORDER))), 'Four ordered independent samples required')
            values[arm] = [r['value_ms'] for r in selected]
            require(all(math.isfinite(v) and v > 0 for v in values[arm]), 'Finite positive times required')
        changes = [100 * (b / a - 1) for a, b in zip(values['A'], values['B'])]
        result[role] = dict(values_ms=values, medians_ms={a: statistics.median(v) for a, v in values.items()},
            paired_percent=changes, paired_median_percent=statistics.median(changes),
            paired_min_percent=min(changes), paired_max_percent=max(changes))
    require(len(rows) == len(ROLE_CASES) * len(evidence.ARMS) * len(evidence.ORDER), 'Unexpected measurement rows')
    return result


def validate_requests(role, arm, stages):
    require(set(stages) == set(evidence.traversal.RT_STAGES), 'Four stage requests required')
    for stage, request in stages.items():
        raygen = stage == 'ray_generation'
        defines = [['METALLIC_CUSTOM_MATERIALS', '0'], ['METALLIC_GLOBAL_VIEW', '0'], ['METALLIC_HAS_RTXCR', '0'],
                   ['METALLIC_VKGLTF_GUIDES', '1' if raygen else '0'],
                   ['METALLIC_PT_SINGLE_SAMPLE', '1' if raygen and role == 'single' else '0'], ['METALLIC_VKGLTF_SER', '0']]
        if arm == 'B' and raygen:
            defines.append(['METALLIC_SIMPLE_BSDF', '1'])
        require(request == dict(profile='spirv_1_6', capabilities=['spvRayTracingKHR'], defines=defines, search_paths=[]),
                f'Factory request differs from fixed same-integrator/SER-off policy: {role}/{arm}/{stage}')


def load_plan(path):
    plan = read(path)
    require(plan.get('schema_version') == 1 and plan.get('experiment') == 'simple_bsdf_generic' and
            set(plan.get('roles', {})) == set(ROLE_CASES), 'Wrong experiment or single/general roles')
    capture = Path(plan['correctness']).resolve(strict=True)
    verification = evidence.verify(capture, allow_numeric_failure=True)
    summary = read(capture / 'Summary.json')
    require(verification['full_stage_coverage'], 'Plan requires complete declared correctness coverage')
    identities = {}
    for role, case in ROLE_CASES.items():
        arms = plan['roles'][role]
        require(set(arms) == set(evidence.ARMS), 'Exactly direct A and generic B are required')
        identities[role] = {}
        runs = []
        for arm, item in arms.items():
            validate_requests(role, arm, item['stages'])
            run = next(r for r in summary['analysis']['runs'] if (r['case'], r['block'], r['arm']) == (case, 0, arm))
            target = next(r for r in run['recipes']['records'] if r['target'])
            path = Path(item['recipe']).resolve(strict=True)
            require(path == Path(target['manifest']).resolve() and digest(path) == target['manifest_sha256'],
                    'Plan recipe is not the actual captured role/arm target')
            identity = evidence.native.recipe_identity(path, 'ray_tracing')
            require(identity['recipe']['expected_pso_hash'] == target['identity']['expected_pso_hash'], 'Target PSO hash changed')
            identities[role][arm] = identity
            runs.append(run)
        evidence.require_shared_policy(runs[0]['recipes'], runs[1]['recipes'])
    return plan, identities, verification


def schedule():
    for block, order in enumerate(evidence.ORDER):
        for role in ROLE_CASES:
            for arm in order:
                yield block, role, arm


def frozen_paths(executable, plan_path, plan, identities):
    paths = {executable, plan_path, Path(__file__).resolve()}
    paths.update(executable.parent.glob('*.dll'))
    paths.update(Path(m.__file__).resolve() for m in (evidence, shared_compile, evidence.shared,
        evidence.traversal, evidence.baseline, evidence.experiment, evidence.driver, evidence.native, evidence.frontend))
    capture = Path(plan['correctness']).resolve()
    paths.update(capture / name for name in ('Manifest.json', 'Summary.json', 'Inputs.json'))
    paths.update(Path(p) for p in read(capture / 'Inputs.json')['snapshots'])
    for arms in identities.values():
        for identity in arms.values():
            paths.update(Path(p) for p in identity['files'])
    return paths


def run(args):
    output, executable, plan_path = args.output.resolve(), args.executable.resolve(strict=True), args.plan.resolve(strict=True)
    require(not output.exists(), 'Measurement requires a fresh directory')
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Bounded timeout required')
    plan, identities, correctness = load_plan(plan_path)
    frozen = evidence.experiment.snapshots(frozen_paths(executable, plan_path, plan, identities))
    environment, removed = evidence.traversal.clean_environment()
    environment['METALLIC_VK_INTERNAL_PIPELINE_CACHE'] = 'disabled'
    summary = dict(schema_version=1, experiment='simple_bsdf_generic', status='running', mode=args.mode,
        executable=str(executable), plan_path=str(plan_path), plan=plan, identities=identities, frozen=frozen,
        orders=list(evidence.ORDER), arms=evidence.ARMS, rows=[], correctness=correctness,
        scope='Pure native vkCreateRayTracingPipelinesKHR; no Slang or SBT.' if args.mode == 'native' else
              'Sum of four sequential fresh slangc process wall times; includes process/module loading, not native PSO.',
        cache_policy='Empty application VkPipelineCache, controllable driver cache disabled; OS state may be warm.' if args.mode == 'native' else
                     'Four fresh compiler processes and output files per sample; compiler/OS cache state not established cold.',
        environment={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))}, removed_environment=removed)
    with evidence.traversal.experiment_lock(output):
        output.mkdir(parents=True)
        shutil.copy2(__file__, output / 'SimpleBSDFGenericCompile.py')
        save(output / 'Plan.json', plan)
        save(output / 'Summary.json', summary)
        try:
            for block, role, arm in schedule():
                evidence.experiment.check_snapshots(frozen)
                destination = output / f'{block:02}-{role}-{arm}'
                item = identities[role][arm]
                print(f'{args.mode}/{block}/{role}/{arm}', flush=True)
                if args.mode == 'native':
                    command = [str(executable), '--recipe-file', item['path'], '--iterations', '1',
                               '--pipeline-cache', 'empty', '--debug-mode', 'disabled', '--output-dir', str(destination)]
                    process, report = evidence.native.run_process(command, environment, destination, args.timeout)
                    sample = evidence.driver.single_sample(report, native=True)
                    summary.setdefault('device', report['device'])
                    evidence.driver.verify_measurement(report, evidence.driver.shader_identity(report, sample), summary['device'], disabled=True)
                    row = dict(process=process, value_ms=sample['pipeline']['native_create_ms'])
                else:
                    stages = {}
                    for stage_name in evidence.traversal.RT_STAGES:
                        stage = item['recipe']['shaders'][stage_name]
                        stage_dir = destination / stage_name
                        command = compile_command(executable, stage, plan['roles'][role][arm]['stages'][stage_name], stage_dir / 'shader.spv')
                        stages[stage_name] = compile_stage(command, stage_dir, args.timeout,
                            Path(item['path']).parent / stage['spirv_file'], environment)
                    row = dict(stages=stages, value_ms=sum(s['wall_ms'] for s in stages.values()))
                summary['rows'].append(dict(block=block, role=role, arm=arm, directory=destination.name, **row))
                save(output / 'Summary.json', summary)
                evidence.experiment.check_snapshots(frozen)
            summary['statistics'] = statistics_for(summary['rows'])
            matching = args.mode == 'native' or all(s['matches_runtime_spirv'] for r in summary['rows'] for s in r['stages'].values())
            summary.update(status='passed' if matching else 'bytecode_mismatch', production_attribution_qualified=matching)
        except BaseException as error:
            summary.update(status='failed', error=repr(error))
            raise
        finally:
            save(output / 'Summary.json', summary)
            evidence.seal(output)
    return verify(output, allow_bytecode_mismatch=True)


def verify(output, allow_bytecode_mismatch=False):
    output = Path(output).resolve(strict=True)
    evidence.verify_seal(output)
    summary = read(output / 'Summary.json')
    require(summary['status'] == 'passed' or (allow_bytecode_mismatch and summary['status'] == 'bytecode_mismatch'),
            'Failed/incomplete or bytecode-mismatched evidence')
    require(summary['experiment'] == 'simple_bsdf_generic' and summary['arms'] == evidence.ARMS and
            summary['orders'] == list(evidence.ORDER) and summary['mode'] in ('native', 'compile'), 'Measurement contract changed')
    current = Path(__file__).resolve()
    require(digest(output / current.name) == summary['frozen'][str(current)], 'Archived runner changed')
    evidence.experiment.check_snapshots({p: h for p, h in summary['frozen'].items() if Path(p).resolve() != current})
    plan, identities, correctness = load_plan(output / 'Plan.json')
    require(plan == summary['plan'] and identities == summary['identities'] and correctness == summary['correctness'],
            'Captured source identity or correctness result changed')
    require([(r['block'], r['role'], r['arm']) for r in summary['rows']] == list(schedule()), 'Incomplete or changed process sequence')
    last_end, stable, matching = 0, {}, True
    for row in summary['rows']:
        require(row['directory'] == f'{row["block"]:02}-{row["role"]}-{row["arm"]}', 'Process directory changed')
        destination = output / row['directory']
        item = identities[row['role']][row['arm']]
        if summary['mode'] == 'native':
            process, report = read(destination / 'Process.json'), read(destination / 'report.json')
            require(process == row['process'], 'Native process record changed')
            expected_command = [summary['executable'], '--recipe-file', item['path'], '--iterations', '1',
                                '--pipeline-cache', 'empty', '--debug-mode', 'disabled', '--output-dir', str(destination)]
            require(process['command'] == expected_command, 'Native command/cache policy changed')
            processes = [process]
            sample = evidence.driver.single_sample(report, native=True)
            key = row['role'], row['arm']
            stable.setdefault(key, evidence.driver.shader_identity(report, sample))
            evidence.driver.verify_measurement(report, stable[key], summary['device'], disabled=True)
            pipeline = sample['pipeline']
            require(pipeline['native_create_ms'] == row['value_ms'] and pipeline['pso_hash'] ==
                    evidence.native.expected_pso_hash(item['recipe']), 'Wrong PSO/timing')
            require(not any(pipeline.get(k, False) for k in ('native_cache_hit', 'metadata_hit', 'native_object_reused',
                    'native_binary_cache_hit', 'native_binary_driver_rejected')), 'Native cache contamination')
            require(not report['configuration']['validation'] and not report['configuration']['disable_driver_optimization'],
                    'Diagnostic timing contamination')
        else:
            processes = []
            require(set(row['stages']) == set(evidence.traversal.RT_STAGES), 'Missing compiler stage')
            for stage_name in evidence.traversal.RT_STAGES:
                stage_dir = destination / stage_name
                process = read(stage_dir / 'Process.json')
                require(process == row['stages'][stage_name], 'Compiler process record changed')
                stage = item['recipe']['shaders'][stage_name]
                command = compile_command(Path(summary['executable']), stage,
                    plan['roles'][row['role']][row['arm']]['stages'][stage_name], stage_dir / 'shader.spv')
                require(command == process['command'], 'Compiler request changed')
                reference, emitted = Path(item['path']).parent / stage['spirv_file'], stage_dir / 'shader.spv'
                equal = emitted.read_bytes() == reference.read_bytes()
                require(digest(reference) == process['reference_sha256'] and digest(emitted) == process['emitted_sha256'] and
                        emitted.stat().st_size == process['emitted_bytes'] and equal is process['matches_runtime_spirv'], 'Compiler bytecode changed')
                require(math.isfinite(process['wall_ms']) and process['wall_ms'] > 0, 'Invalid compiler elapsed time')
                matching &= equal
                processes.append(process)
            require(sum(p['wall_ms'] for p in processes) == row['value_ms'], 'Compiler aggregate changed')
        for process in processes:
            require(process['exitCode'] == 0 and not process['timedOut'] and process['started'] >= last_end and
                    process['ended'] >= process['started'], 'Failed or overlapping child process')
            last_end = process['ended']
    require(statistics_for(summary['rows']) == summary['statistics'], 'Statistics changed')
    require(matching is summary['production_attribution_qualified'] and matching is (summary['status'] == 'passed'),
            'Production bytecode attribution contradicts status')
    return dict(verified=True, mode=summary['mode'], production_attribution_qualified=matching,
        correctness_qualified=correctness['correctness_qualified'], independent_samples=len(summary['rows']),
        statistics=summary['statistics'], manifest_sha256=digest(output / 'Manifest.json'),
        captured_runner_sha256=summary['frozen'][str(current)], current_verifier_sha256=digest(current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('run')
    capture.add_argument('--mode', choices=('native', 'compile'), required=True)
    capture.add_argument('--plan', type=Path, required=True)
    capture.add_argument('--executable', type=Path, required=True)
    capture.add_argument('--output', type=Path, required=True)
    capture.add_argument('--timeout', type=float, default=600)
    check = sub.add_parser('verify')
    check.add_argument('output', type=Path)
    check.add_argument('--allow-bytecode-mismatch', action='store_true')
    args = parser.parse_args()
    result = run(args) if args.command == 'run' else verify(args.output, args.allow_bytecode_mismatch)
    print(json.dumps(result, indent=2))
    if not result['production_attribution_qualified']:
        parser.exit(2, 'Runtime bytecode mismatch; compiler timing is diagnostic only.\n')


if __name__ == '__main__':
    main()
