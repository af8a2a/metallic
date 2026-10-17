"""Strict direct-vs-generic glTF BSDF evidence; same integrator, SER disabled."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil

import VkGltfSER as shared

ROOT = shared.ROOT
TASK = ROOT / 'build-release/simple-bsdf-generic-20261017'
ARMS = {'A': dict(bsdf='vkGltf', shaderExecutionReorder=False),
        'B': dict(bsdf='simple', shaderExecutionReorder=False)}
CASES = ('Opaque1', 'Mixed4', 'ABeautifulGame')
ORDER = ('AB', 'BA', 'AB', 'BA')
CORRECTNESS_ORDER = ('AB', 'BA')
ABSOLUTE, RELATIVE = shared.ABSOLUTE, shared.RELATIVE
traversal, baseline, experiment = shared.traversal, shared.baseline, shared.experiment
native, driver, frontend = shared.native, shared.driver, shared.frontend
read, save, digest, require = shared.read, shared.save, shared.digest, shared.require
seal, verify_seal = shared.seal, shared.verify_seal


def normalized_graph(graph, arm):
    result = copy.deepcopy(graph)
    props = traversal.path_node(result)['properties']
    require(props.get('traversalBackend') == 'rayPipeline' and props.get('cacheMode') == 'off' and
            props.get('debugView', 'final') == 'final', 'Only rayPipeline/cache-off/final is supported')
    for name, expected in ARMS[arm].items():
        require(props.pop(name, None) == expected, f'Arm property changed: {name}')
    return result


def schedule(cases):
    for case in cases:
        for block, order in enumerate(CORRECTNESS_ORDER):
            for arm in order:
                yield case['id'], block, arm


def prepare(args):
    source = args.fixtures.resolve(strict=True)
    shared.verify_prepared(source)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Prepare requires a fresh directory')
    source_inputs = experiment.snapshots(p for p in source.rglob('*') if p.is_file())
    output.mkdir(parents=True)
    for arm, properties in ARMS.items():
        (output / arm).mkdir()
        config = read(source / 'correctness-G.json')
        config['cases'] = [c for c in config['cases'] if c['id'] in CASES]
        require([c['id'] for c in config['cases']] == list(CASES), 'Missing declared correctness cases')
        for spec in config['cases']:
            graph = read(source / spec['graph'])
            traversal.path_node(graph)['properties'].update(properties)
            spec['graph'] = f'{arm}/{spec["id"]}.json'
            save(output / spec['graph'], graph)
        save(output / f'correctness-{arm}.json', config)
    save(output / 'Prepared.json', dict(schema_version=1, experiment='simple_bsdf_generic', arms=ARMS,
        cases=CASES, source=str(source), source_inputs=source_inputs, correctness_order=CORRECTNESS_ORDER,
        measurement_order=ORDER, absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE,
        guide_requirement='raw-byte-exact', comparison_contract='Same BSDF and integrator; A/B and A/A always require strict equivalence.',
        exclusions=['No GPU timing. No OpenPBR or SER arm. Only the three declared correctness cases.']))
    seal(output)
    return verify_prepared(output)


def verify_prepared(directory):
    directory = Path(directory)
    verify_seal(directory)
    metadata = read(directory / 'Prepared.json')
    require(metadata['experiment'] == 'simple_bsdf_generic' and metadata['arms'] == ARMS and
            metadata['cases'] == list(CASES) and metadata['correctness_order'] == list(CORRECTNESS_ORDER) and
            metadata['measurement_order'] == list(ORDER), 'Prepared experiment contract changed')
    require(metadata['absolute_tolerance'] == ABSOLUTE and metadata['relative_tolerance'] == RELATIVE and
            metadata['guide_requirement'] == 'raw-byte-exact', 'Fixed numeric contract changed')
    experiment.check_snapshots(metadata['source_inputs'])
    reference = None
    for arm in ARMS:
        config = read(directory / f'correctness-{arm}.json')
        require(config['synchronizationValidation'] is True and
                [c['id'] for c in config['cases']] == list(CASES), 'Correctness coverage/validation changed')
        for spec in config['cases']:
            spec['graph'] = normalized_graph(read(directory / spec['graph']), arm)
            require(spec['captureFrames'] and len(spec['captureOutputs']) == 7, 'Temporal/seven-guide coverage missing')
        if reference is None:
            reference = config
        require(config == reference, 'Arms differ beyond declared BSDF wrapper selection')
    return metadata


def recipe_evidence(directory, arm, validation, spec, graph):
    normalized_graph(graph, arm)
    # Both arms deliberately share the exact vkGltf entrypoints and hit stages.
    # G selects the existing non-SER shader-family validator, not the experiment arm.
    return shared.recipe_evidence(directory, 'G', validation, spec, graph)


def require_shared_policy(left, right):
    require(left['device'] == right['device'], 'Device/features/validation changed')
    require(left['unrelated'] == right['unrelated'], 'Non-target PSO changed')
    require(left['target']['state'] == right['target']['state'], 'Native target state changed')
    require(left['target']['cache_group'] == right['target']['cache_group'], 'Target cache group changed')
    shared.require_shared_hit_stages(left['target']['shaders'], right['target']['shaders'])
    for item in (left, right):
        require(not item['target']['state'].get('enable_invocation_reorder', False), 'SER must be disabled')
        for shader in item['target']['shaders'].values():
            require(not any('invocation_reorder' in e for e in shader.get('extensions', [])), 'SER extension is forbidden')


def inspect_run(destination, configuration, arm, prepared):
    process = read(destination / 'process.json')
    require(process['exit_code'] == 0 and not process['timed_out'], 'Failed or timed-out process')
    traversal.validate_xml(destination / 'Tests.xml')
    path = destination / 'MaterialBaseline.json'
    report = read(path)
    require(report['validation'] is True and report.get('synchronizationValidation') is True and
            len(report['cases']) == 1, 'Correctness requires core/sync validation and one case')
    spec, case = configuration['cases'][0], report['cases'][0]
    experiment.validate_requested_case(case, spec, configuration)
    require(case['id'] == spec['id'] and case.get('validHDR') is True and case.get('nonfiniteComponents') == 0,
            'Invalid case or nonfinite HDR')
    graph = read(prepared / spec['graph'])
    require(normalized_graph(case['resolvedGraph'], arm) ==
            normalized_graph(traversal.expected_resolved_graph(graph, spec), arm), 'Resolved graph differs from request')
    if spec.get('requireEnvironment', True):
        require(case.get('environmentTransitions') and all(t['mapAvailable'] for t in case['environmentTransitions']),
                'Environment was not resident')
    require(Path(case['image']).name == case['image'], 'Readback escaped sibling directory')
    experiment.readback(path, case)
    for capture in case.get('captures', []):
        require(Path(capture['file']).name == capture['file'], 'Capture escaped sibling directory')
        experiment.readback(path, case, capture['file'], capture['bytes'], capture['format'])
    log = '\n'.join((destination / n).read_text(errors='replace').lower() for n in ('stdout.log', 'stderr.log'))
    for marker in ('initial material compilation failed', 'displaying the error material', 'validation error:',
                   'validation error [', 'validation error (', 'validation error count: 1', 'vuid-'):
        require(marker not in log, f'Runtime error: {marker}')
    return dict(report=str(path), report_sha256=digest(path), xml_sha256=digest(destination / 'Tests.xml'), process=process,
                recipes=recipe_evidence(destination / 'recipes', arm, True, spec, graph))


def compare(left, right):
    import numpy as np
    lp, rp = Path(left['report']), Path(right['report'])
    a, b = read(lp)['cases'][0], read(rp)['cases'][0]
    require(normalized_graph(a['resolvedGraph'], left['arm']) == normalized_graph(b['resolvedGraph'], right['arm']),
            'Compared graph changed beyond wrapper selection')
    for field in ('id', 'width', 'height', 'format', 'environment', 'scene', 'output', 'captures'):
        require(a.get(field) == b.get(field), f'Readback workload changed: {field}')
    require_shared_policy(left['recipes'], right['recipes'])
    comparisons = []
    for capture in [None, *a.get('captures', [])]:
        args = (capture['file'], capture['bytes'], capture['format']) if capture else ()
        av, bv = experiment.readback(lp, a, *args), experiment.readback(rp, b, *args)
        require(av.shape == bv.shape, 'Readback shape changed')
        error = np.abs(av - bv)
        guide = capture is not None and capture['output'] != a['output']
        filename = capture['file'] if capture else a['image']
        exact = (lp.parent / filename).read_bytes() == (rp.parent / filename).read_bytes()
        bound = 0 if guide else ABSOLUTE + RELATIVE * np.abs(av)
        comparisons.append(dict(output=f'{capture["output"]}@{capture["frame"]}' if capture else 'final HDR',
            guide=guide, byte_exact=exact, strict_equivalent=exact if guide else bool(np.all(error <= bound)),
            max_absolute_error=float(error.max(initial=0)), rmse=float(np.sqrt(np.mean(error * error))),
            differing_components=int(np.count_nonzero(error)), failed_components=int(np.count_nonzero(error > bound))))
    return dict(qualified=all(c['strict_equivalent'] for c in comparisons), equivalence_required=True,
                readbacks=comparisons, non_target_pipeline_identity_equal=True)


def analyze(output, prepared, selected, recorded=None):
    configs = {arm: read(prepared / f'correctness-{arm}.json') for arm in ARMS}
    cases = [c for c in configs['A']['cases'] if c['id'] in selected]
    runs, repeats, comparisons, first = [], [], [], {}
    for name, block, arm in schedule(cases):
        destination = output / name / f'block-{block:02}-{arm}'
        expected = copy.deepcopy(configs[arm])
        expected['cases'] = [c for c in expected['cases'] if c['id'] == name]
        expected['cases'][0]['graph'] = str((prepared / expected['cases'][0]['graph']).resolve())
        require(read(destination / 'Cases.json') == expected, 'Captured configuration changed')
        run = dict(case=name, block=block, arm=arm, **inspect_run(destination, expected, arm, prepared))
        if (name, arm) in first:
            traversal.require_stable_recipes(first[name, arm], run)
            repeats.append(dict(case=name, block=block, arm=arm, **compare(first[name, arm], run)))
        else:
            first[name, arm] = run
        runs.append(run)
    for case in cases:
        for block in range(len(CORRECTNESS_ORDER)):
            pair = [next(r for r in runs if (r['case'], r['block'], r['arm']) == (case['id'], block, arm)) for arm in ARMS]
            comparisons.append(dict(case=case['id'], block=block, **compare(*pair)))
    result = dict(runs=runs, comparisons=comparisons, same_arm_readbacks=repeats,
                  correctness_qualified=all(r['qualified'] for r in comparisons + repeats))
    if recorded is not None:
        require(result == recorded, 'Recomputed analysis differs from recorded evidence')
    return result


def frozen_inputs(executable, prepared):
    identity = baseline.identity(executable, prepared / 'correctness-A.json')
    paths = {Path(p) for p in identity['binaries']} | {ROOT / p for p in identity['sources']}
    paths.update(Path(p) for p in identity['workload'])
    paths.update(p for p in prepared.rglob('*') if p.is_file())
    for extension in ('*.h', '*.inl'):
        paths.update((ROOT / 'Shaders').rglob(extension))
    paths.update((ROOT / 'Source').rglob('*.inl'))
    paths.update(Path(m.__file__).resolve() for m in (shared, traversal, baseline, experiment, driver, native, frontend))
    paths.add(Path(__file__).resolve())
    return dict(metadata=identity, snapshots=experiment.snapshots(paths))


def run(args):
    prepared, executable = args.prepared.resolve(strict=True), args.executable.resolve(strict=True)
    verify_prepared(prepared)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Run requires a fresh directory')
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Bounded timeout required')
    require('CMAKE_BUILD_TYPE:STRING=Release' in (executable.parent.parent / 'CMakeCache.txt').read_text(), 'Release build required')
    configs = {arm: read(prepared / f'correctness-{arm}.json') for arm in ARMS}
    selected = args.case or list(CASES)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(CASES), 'Unknown/repeated case')
    cases = [c for c in configs['A']['cases'] if c['id'] in selected]
    selected = [c['id'] for c in cases]
    environment, removed = traversal.clean_environment()
    with traversal.experiment_lock(output):
        output.mkdir(parents=True)
        inputs = frozen_inputs(executable, prepared)
        save(output / 'Inputs.json', inputs)
        shutil.copy2(__file__, output / 'SimpleBSDFGeneric.py')
        save(output / 'Environment.json', dict(removed=removed,
            fixed={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))}))
        summary = dict(schema_version=1, experiment='simple_bsdf_generic', status='running', stage='correctness', arms=ARMS,
            prepared=str(prepared), prepared_manifest_sha256=digest(prepared / 'Manifest.json'), selected_cases=selected,
            full_stage_coverage=selected == list(CASES), schedule=list(schedule(cases)), process_records=[],
            absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_comparison='raw-byte-exact',
            scope='Core/sync validation, temporal HDR and guides. No GPU performance attribution.')
        save(output / 'Summary.json', summary)
        try:
            for name, block, arm in schedule(cases):
                experiment.check_snapshots(inputs['snapshots'])
                destination = output / name / f'block-{block:02}-{arm}'
                destination.mkdir(parents=True)
                config = copy.deepcopy(configs[arm])
                config['cases'] = [c for c in config['cases'] if c['id'] == name]
                config['cases'][0]['graph'] = str((prepared / config['cases'][0]['graph']).resolve())
                save(destination / 'Cases.json', config)
                env = dict(environment, METALLIC_MATERIAL_BASELINE_CASES=str(destination / 'Cases.json'),
                    METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / 'recipes'),
                    METALLIC_PSO_CACHE_DIRECTORY=str(output / 'application-pso-cache' / arm))
                command = [str(executable), '--gtest_filter=RHIRendering.material_phase0_baseline', '--rhi-validation',
                    '--output-dir', str(destination), f'--gtest_output=xml:{destination / "Tests.xml"}']
                print(f'correctness/{name}/block-{block:02}-{arm}', flush=True)
                process = traversal.child(command, env, destination, args.timeout)
                summary['process_records'].append(dict(case=name, block=block, arm=arm, process=process))
                save(output / 'Summary.json', summary)
                inspect_run(destination, config, arm, prepared)
                experiment.check_snapshots(inputs['snapshots'])
            summary['analysis'] = analyze(output, prepared, selected)
            summary['status'] = 'passed' if summary['analysis']['correctness_qualified'] else 'numeric_failure'
        except BaseException as error:
            summary.update(status='failed', error=repr(error))
            raise
        finally:
            save(output / 'Summary.json', summary)
            seal(output)
    return verify(output, allow_numeric_failure=True)


def verify(output, allow_numeric_failure=False):
    output = Path(output).resolve(strict=True)
    verify_seal(output)
    summary = read(output / 'Summary.json')
    require(summary['status'] == 'passed' or (allow_numeric_failure and summary['status'] == 'numeric_failure'),
            'Incomplete/failed or numerically unqualified capture')
    require(summary['experiment'] == 'simple_bsdf_generic' and summary['stage'] == 'correctness' and summary['arms'] == ARMS,
            'Experiment policy changed')
    require(summary['absolute_tolerance'] == ABSOLUTE and summary['relative_tolerance'] == RELATIVE and
            summary['guide_comparison'] == 'raw-byte-exact', 'Numeric policy changed')
    prepared = Path(summary['prepared'])
    verify_prepared(prepared)
    require(digest(prepared / 'Manifest.json') == summary['prepared_manifest_sha256'], 'Prepared identity changed')
    inputs = read(output / 'Inputs.json')
    current = Path(__file__).resolve()
    require(digest(output / current.name) == inputs['snapshots'][str(current)], 'Archived runner changed')
    experiment.check_snapshots({p: h for p, h in inputs['snapshots'].items() if Path(p).resolve() != current})
    configs = read(prepared / 'correctness-A.json')
    selected = summary['selected_cases']
    cases = [c for c in configs['cases'] if c['id'] in selected]
    require(selected and selected == [c['id'] for c in cases] and
            summary['full_stage_coverage'] == (selected == list(CASES)), 'Coverage changed')
    expected = list(schedule(cases))
    require(summary['schedule'] == [list(v) for v in expected] and
            [(r['case'], r['block'], r['arm']) for r in summary['process_records']] == expected, 'Incomplete/changed schedule')
    last_end = None
    for row in summary['process_records']:
        path = output / row['case'] / f'block-{row["block"]:02}-{row["arm"]}' / 'process.json'
        process = read(path)
        require(process == row['process'], 'Process record changed')
        start, end = [datetime.fromisoformat(process[k]) for k in ('started_utc', 'ended_utc')]
        require(end >= start and (last_end is None or start >= last_end), 'Overlapping/invalid process timing')
        last_end = end
    analysis = analyze(output, prepared, selected, summary['analysis'])
    require(analysis['correctness_qualified'] is (summary['status'] == 'passed'), 'Qualification contradicts status')
    return dict(verified=True, stage='correctness', correctness_qualified=analysis['correctness_qualified'],
        numeric_failure=summary['status'] == 'numeric_failure', independent_processes=len(analysis['runs']),
        full_stage_coverage=summary['full_stage_coverage'], manifest_sha256=digest(output / 'Manifest.json'),
        captured_runner_sha256=inputs['snapshots'][str(current)], current_verifier_sha256=digest(current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--fixtures', type=Path, default=shared.TASK / 'prepared-final')
    prep.add_argument('--output-dir', type=Path, default=TASK / 'prepared')
    capture = sub.add_parser('run')
    capture.add_argument('--prepared', type=Path, default=TASK / 'prepared')
    capture.add_argument('--executable', type=Path, default=ROOT / 'build-pass-stages-nrd/tests/MetallicRHITests.exe')
    capture.add_argument('--output-dir', type=Path, required=True)
    capture.add_argument('--timeout', type=float, default=1800)
    capture.add_argument('--case', action='append')
    check = sub.add_parser('verify')
    check.add_argument('output_dir', type=Path)
    check.add_argument('--allow-numeric-failure', action='store_true')
    args = parser.parse_args()
    result = prepare(args) if args.command == 'prepare' else run(args) if args.command == 'run' else verify(args.output_dir, args.allow_numeric_failure)
    print(json.dumps(result, indent=2))
    if result.get('numeric_failure'):
        parser.exit(2, 'Strict direct/generic correctness gate failed; intact diagnostics retained.\n')


if __name__ == '__main__':
    main()
