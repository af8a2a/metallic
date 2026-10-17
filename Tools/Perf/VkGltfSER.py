"""Isolated OpenPBR RT / vkGltf RT / vkGltf EXT SER measurements.

O/G intentionally compare different material models. Only G/S and same-arm
repeats have an equivalence gate. This does not relax PathTraceTraversal.py.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import MaterialBaseline as baseline
import PathTraceTraversal as traversal
import PathTraceTraversalPSO as native
import ProfileComputeShaderFrontend as frontend
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver

ROOT = traversal.ROOT
TASK = ROOT / 'build-release/vkgltf-ser-20261017'
ARMS = {'O': dict(bsdf='openpbr', shaderExecutionReorder=False),
        'G': dict(bsdf='vkGltf', shaderExecutionReorder=False),
        'S': dict(bsdf='vkGltf', shaderExecutionReorder=True)}
ORDER = ('OGS', 'GSO', 'SOG')
CORRECTNESS_ORDER = ('OGS', 'SGO')
ABSOLUTE, RELATIVE = traversal.ABSOLUTE, traversal.RELATIVE
VK_MODULE = 'Features/PathTracing/VkGltfRayPipelinePathTrace'
VK_HIT_MODULE = 'Features/PathTracing/VkGltfRayPipelineHit'
read, save, digest, require = traversal.read, traversal.save, traversal.digest, traversal.require
seal, verify_seal = traversal.seal, traversal.verify_seal


def normalized_graph(graph, arm):
    result = copy.deepcopy(graph)
    props = traversal.path_node(result)['properties']
    require(props.get('traversalBackend') == 'rayPipeline' and props.get('cacheMode') == 'off',
            'Only rayPipeline/cache-off workloads are supported')
    for field, expected in ARMS[arm].items():
        require(props.pop(field, None) == expected, f'Arm {arm} property changed: {field}')
    return result


def prepare(args):
    source = args.fixtures.resolve(strict=True)
    # Reuse sealed workload data, never execute an archived runner.
    traversal.verify_prepared(source)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Prepare requires a fresh output directory')
    source_inputs = experiment.snapshots(p for p in source.rglob('*') if p.is_file())
    output.mkdir(parents=True)
    exclusions = []
    for arm, properties in ARMS.items():
        (output / arm).mkdir()
        for stage in ('correctness', 'performance'):
            config = read(source / f'{stage}-B.json')
            supported = []
            for spec in config['cases']:
                graph = read(source / spec['graph'])
                if traversal.path_node(graph)['properties'].get('debugView', 'final') != 'final':
                    record = dict(stage=stage, case=spec['id'], reason='vkGltf supports debugView=final only')
                    if record not in exclusions:
                        exclusions.append(record)
                    continue
                traversal.path_node(graph)['properties'].update(properties)
                spec['graph'] = f'{arm}/{spec["id"]}.json'
                save(output / spec['graph'], graph)
                supported.append(spec)
            config['cases'] = supported
            save(output / f'{stage}-{arm}.json', config)
    save(output / 'Prepared.json', dict(schema_version=1, arms=ARMS, source=str(source),
        source_inputs=source_inputs, performance_order=ORDER, correctness_order=CORRECTNESS_ORDER, exclusions=exclusions,
        absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_requirement='raw-byte-exact',
        comparison_contract='O/G material models differ: descriptive only. G/S and same-arm repeats: strict HDR and byte-exact guides.',
        source_runner_sha256=digest(__file__)))
    seal(output)
    return verify_prepared(output)


def verify_prepared(directory):
    verify_seal(directory)
    metadata = read(directory / 'Prepared.json')
    experiment.check_snapshots(metadata['source_inputs'])
    require(metadata['arms'] == ARMS and metadata['performance_order'] == list(ORDER) and
            metadata['correctness_order'] == list(CORRECTNESS_ORDER), 'Experiment arm/order contract changed')
    require(metadata['absolute_tolerance'] == ABSOLUTE and metadata['relative_tolerance'] == RELATIVE and
            metadata['guide_requirement'] == 'raw-byte-exact', 'Fixed numeric contract changed')
    for stage in ('correctness', 'performance'):
        reference = None
        for arm in ARMS:
            config = read(directory / f'{stage}-{arm}.json')
            require(config['synchronizationValidation'] is (stage == 'correctness'), 'Diagnostic timing contamination')
            normalized = copy.deepcopy(config)
            for spec in normalized['cases']:
                spec['graph'] = normalized_graph(read(directory / spec['graph']), arm)
                require(traversal.path_node(spec['graph'])['properties'].get('debugView', 'final') == 'final',
                        'Unsupported debugView in prepared workload')
            if reference is None:
                reference = normalized
            require(reference == normalized, 'Arms differ beyond the declared BSDF/SER properties')
    return metadata


def spirv_extensions(raw):
    import struct
    require(len(raw) >= 20 and len(raw) % 4 == 0 and raw[:4] == b'\x03\x02\x23\x07', 'Invalid SPIR-V')
    words = struct.unpack('<' + 'I' * (len(raw) // 4), raw)
    extensions, offset = [], 5
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 0xffff
        require(size > 0 and offset + size <= len(words), 'Malformed SPIR-V instruction')
        if opcode == 10:  # OpExtension, decoded from actual captured payload.
            extensions.append(raw[(offset + 1) * 4:(offset + size) * 4].split(b'\0', 1)[0].decode('ascii'))
        offset += size
    return extensions


def recipe_evidence(directory, arm, validation, specification, graph):
    entries, targets, devices = [], [], []
    target_modules = set(traversal.RT_MODULES) | {VK_MODULE}
    for path in sorted(directory.glob('*.recipe.json')):
        recipe = read(path)
        require(recipe['schema'] == 'metallic_pipeline_recipe' and recipe['version'] in (1, 2, 3, 4),
                'Unsupported captured recipe')
        device = recipe['source_device']
        require(device['validation_enabled'] is validation and
                device['synchronization_validation_enabled'] is validation, 'Recipe validation state changed')
        if device not in devices:
            devices.append(device)
        shaders, payloads = {}, {}
        for name, stage in recipe['shaders'].items():
            if stage['spirv_file'] is None:
                continue
            require(Path(stage['spirv_file']).name == stage['spirv_file'], 'Recipe payload escaped sibling directory')
            payload = directory / stage['spirv_file']
            raw = payload.read_bytes()
            extensions = spirv_extensions(raw)
            require(len(raw) == stage['spirv_bytes'] and traversal.fnv1a(raw) == stage['input_spirv_hash'], 'Recipe SPIR-V changed')
            require(stage['slang_optimization'] == 'O1' and stage['debug_mode'] == 'release' and
                    stage['descriptor_heap_mode'] == 'mapped' and stage['spirv_postprocess'] == 'none' and
                    stage['source_identity_ambiguous'] is False, 'Unqualified shader policy')
            identity = copy.deepcopy(stage)
            identity.pop('spirv_file')
            identity['specialization_constants'] = traversal.specialization(stage, recipe['version'])
            identity['sha256'] = digest(payload)
            identity['extensions'] = extensions
            shaders[name] = identity
            payloads[name] = dict(file=str(payload), bytes=len(raw), sha256=digest(payload))
        identity = dict(pipeline_type=recipe['pipeline_type'], state=recipe['state'], shaders=shaders,
                        expected_pso_hash=recipe['expected_pso_hash'], cache_group=recipe['cache_group'])
        target = any(s['module_name'] in target_modules for s in shaders.values())
        record = dict(manifest=str(path), manifest_sha256=digest(path), payloads=payloads, identity=identity, target=target)
        entries.append(record)
        if target:
            targets.append(identity)
    require(len(devices) == 1 and len(targets) == 1, 'One target RT PSO/device is required')
    target = targets[0]
    shaders, state = target['shaders'], target['state']
    require(target['pipeline_type'] == 'ray_tracing' and set(shaders) == set(traversal.RT_STAGES) and
            state['uses_bindless_heap'] and not state['disable_optimization'] and
            state['max_recursion_depth'] == 1 and state['cluster_acceleration_structure'], 'Incomplete/changed RT PSO')
    require(devices[0]['enabled_features']['ray_tracing_pipeline'], 'RT pipeline feature not enabled')
    require(state.get('enable_invocation_reorder', False) is ARMS[arm]['shaderExecutionReorder'],
            'Native SER state differs from requested arm')
    if arm == 'S':
        require(devices[0]['enabled_features'].get('ray_tracing_invocation_reorder') is True,
                'EXT invocation reorder device feature not enabled')
    props = traversal.path_node(graph)['properties']
    single = props.get('samples', 1) == 1 and props.get('exportDenoiserGuides', False)
    main = shaders['ray_generation']
    if arm == 'O':
        expected_module = traversal.RT_MODULES[1 if single else 0]
        expected_entry = 'openPbrRayPipelinePathTraceGuidesMain' if single else 'openPbrRayPipelinePathTraceMain'
        expected_specialization = [dict(id=0, type='bool32', size=4, bits=0)]
        if not single:
            expected_specialization.append(dict(id=1, type='bool32', size=4, bits=int(props.get('exportDenoiserGuides', False))))
        hit_module, hit_prefix = 'Features/PathTracing/OpenPBRRayPipelineHit', 'openPbrRayPipeline'
    else:
        expected_module, expected_entry, expected_specialization = VK_MODULE, 'vkGltfRayPipelinePathTraceMain', []
        hit_module, hit_prefix = VK_HIT_MODULE, 'vkGltfRayPipeline'
    require(main['module_name'] == expected_module and main['source_entry_point'] == expected_entry and
            main['specialization_constants'] == expected_specialization, 'Target shader family/entry/specialization changed')
    for stage, suffix in [('miss', 'Miss'), ('closest_hit', 'ClosestHit'), ('any_hit', 'AnyHit')]:
        require(shaders[stage]['module_name'] == hit_module and
                shaders[stage]['source_entry_point'] == hit_prefix + suffix and
                not shaders[stage]['specialization_constants'], 'Hit stage policy changed')
        if arm in ('G', 'S'):
            require('SPV_EXT_shader_invocation_reorder' not in shaders[stage]['extensions'],
                    'Invocation reorder must not enter the shared hit stages')
    ser = 'SPV_EXT_shader_invocation_reorder' in main['extensions']
    require(ser is ARMS[arm]['shaderExecutionReorder'], 'Actual raygen SPIR-V SER identity differs from requested arm')
    return dict(records=entries, target=target, device=devices[0],
        unrelated=sorted((r['identity'] for r in entries if not r['target']), key=traversal.canonical),
        stable=sorted((r['identity'] for r in entries), key=traversal.canonical))


def inspect_run(destination, configuration, arm, validation, prepared):
    process = read(destination / 'process.json')
    require(process['exit_code'] == 0 and not process['timed_out'], 'Failed/timeout process is not evidence')
    traversal.validate_xml(destination / 'Tests.xml')
    path = destination / 'MaterialBaseline.json'
    report = read(path)
    require(report['validation'] is validation and report.get('synchronizationValidation', False) is validation and
            len(report['cases']) == 1, 'Runtime validation/case set changed')
    spec, case = configuration['cases'][0], report['cases'][0]
    require(case['id'] == spec['id'], 'Runtime case identity changed')
    experiment.validate_requested_case(case, spec, configuration)
    require(case.get('validHDR') is True and case.get('nonfiniteComponents') == 0, 'Nonfinite/invalid HDR')
    graph = read(prepared / spec['graph'])
    require(normalized_graph(case['resolvedGraph'], arm) ==
            normalized_graph(traversal.expected_resolved_graph(graph, spec), arm), 'Runtime graph differs from prepared graph')
    if spec.get('requireEnvironment', True):
        require(case.get('environmentTransitions') and all(t['mapAvailable'] for t in case['environmentTransitions']),
                'Environment was not resident')
    require(Path(case['image']).name == case['image'], 'HDR readback escaped sibling directory')
    experiment.readback(path, case)
    for capture in case.get('captures', []):
        require(Path(capture['file']).name == capture['file'], 'Guide capture escaped sibling directory')
        experiment.readback(path, case, capture['file'], capture['bytes'], capture['format'])
    log = '\n'.join((destination / name).read_text(errors='replace').lower() for name in ('stdout.log', 'stderr.log'))
    for marker in ('initial material compilation failed', 'displaying the error material', 'validation error:',
                   'validation error [', 'validation error (', 'validation error count: 1', 'vuid-'):
        require(marker not in log, f'Runtime error marker: {marker}')
    return dict(report=str(path), report_sha256=digest(path), xml_sha256=digest(destination / 'Tests.xml'), process=process,
        recipes=recipe_evidence(destination / 'recipes', arm, validation, spec, graph),
        timings=experiment.gpu_timings(report, configuration))


def compare(left, right):
    import numpy as np
    lp, rp = Path(left['report']), Path(right['report'])
    a, b = read(lp)['cases'][0], read(rp)['cases'][0]
    require(normalized_graph(a['resolvedGraph'], left['arm']) == normalized_graph(b['resolvedGraph'], right['arm']),
            'Compared graph changed beyond arm properties')
    for field in ('id', 'width', 'height', 'format', 'environment', 'scene', 'output', 'captures'):
        require(a.get(field) == b.get(field), f'Readback workload mismatch: {field}')
    same_model = ARMS[left['arm']]['bsdf'] == ARMS[right['arm']]['bsdf']
    comparisons = []
    for capture in [None, *a.get('captures', [])]:
        args = (capture['file'], capture['bytes'], capture['format']) if capture else ()
        av, bv = experiment.readback(lp, a, *args), experiment.readback(rp, b, *args)
        require(av.shape == bv.shape, 'Readback shape mismatch')
        error = np.abs(av - bv)
        guide = capture is not None and capture['output'] != a['output']
        filename = capture['file'] if capture else a['image']
        exact = (lp.parent / filename).read_bytes() == (rp.parent / filename).read_bytes()
        strict = exact if guide else bool(np.all(error <= ABSOLUTE + RELATIVE * np.abs(av)))
        comparisons.append(dict(output=f'{capture["output"]}@{capture["frame"]}' if capture else 'final HDR',
            guide=guide, byte_exact=exact, strict_equivalent=strict, max_absolute_error=float(error.max(initial=0)),
            rmse=float(np.sqrt(np.mean(error * error))), differing_components=int(np.count_nonzero(error)),
            failed_components=int(np.count_nonzero(error > (0 if guide else ABSOLUTE + RELATIVE * np.abs(av))))))
    require(left['recipes']['device'] == right['recipes']['device'], 'Device/driver/validation changed')
    unrelated_equal = left['recipes']['unrelated'] == right['recipes']['unrelated']
    if same_model:
        require(unrelated_equal, 'SER/same-arm comparison changed non-target PSOs')
        if {left['arm'], right['arm']} == {'G', 'S'}:
            require_shared_hit_stages(left['recipes']['target']['shaders'], right['recipes']['target']['shaders'])
    return dict(qualified=all(c['strict_equivalent'] for c in comparisons) if same_model else None,
        equivalence_required=same_model, readbacks=comparisons, non_target_pipeline_identity_equal=unrelated_equal)


def require_shared_hit_stages(left, right):
    for name in ('miss', 'closest_hit', 'any_hit'):
        require(left[name] == right[name], f'SER comparison changed shared hit stage: {name}')
        require('SPV_EXT_shader_invocation_reorder' not in left[name].get('extensions', []),
                'Invocation reorder must not enter the shared hit stages')


def schedule(stage, cases):
    for case in cases:
        for block, order in enumerate(ORDER if stage == 'performance' else CORRECTNESS_ORDER):
            for arm in order:
                yield case['id'], block, arm


def lighting_controls(runs):
    import numpy as np
    results = []
    for arm in ARMS:
        by_case = {r['case']: r for r in runs if r['arm'] == arm and r['block'] == 0}
        for label, lit, dark in [('local', 'MixedNoEnvironment', 'MixedNoEnvironmentNoDirect'),
                                 ('environment', 'MixedEnvironmentOnly', 'MixedEnvironmentOnlyNoDirect')]:
            if lit not in by_case or dark not in by_case:
                continue
            lp, rp = [Path(by_case[name]['report']) for name in (lit, dark)]
            a, b = [read(p)['cases'][0] for p in (lp, rp)]
            delta = np.abs(experiment.readback(lp, a) - experiment.readback(rp, b))
            results.append(dict(arm=arm, control=label, maximum_difference=float(delta.max(initial=0)),
                changed_components=int(np.count_nonzero(delta)), qualified=bool(delta.max(initial=0) > 1e-6)))
    return results


def analyze(output, prepared, stage, selected, recorded=None):
    configs = {arm: read(prepared / f'{stage}-{arm}.json') for arm in ARMS}
    cases = [c for c in configs['O']['cases'] if c['id'] in selected]
    runs, same_arm, comparisons, stable = [], [], [], {}
    for name, block, arm in schedule(stage, cases):
        destination = output / name / f'block-{block:02}-{arm}'
        expected = copy.deepcopy(configs[arm])
        expected['cases'] = [c for c in expected['cases'] if c['id'] == name]
        expected['cases'][0]['graph'] = str((prepared / expected['cases'][0]['graph']).resolve())
        config = read(destination / 'Cases.json')
        require(config == expected, 'Cases.json differs from prepared workload')
        run = dict(case=name, block=block, arm=arm, **inspect_run(destination, config, arm, stage == 'correctness', prepared))
        if (name, arm) in stable:
            previous = stable[name, arm]
            traversal.require_stable_recipes(previous, run)
            same_arm.append(dict(case=name, block=block, arm=arm, **compare(previous, run)))
        else:
            stable[name, arm] = run
        runs.append(run)
    for name in [c['id'] for c in cases]:
        for block in range(len(ORDER if stage == 'performance' else CORRECTNESS_ORDER)):
            for left, right, label in [('O', 'G', 'material'), ('G', 'S', 'ser')]:
                a, b = [next(r for r in runs if r['case'] == name and r['block'] == block and r['arm'] == arm)
                        for arm in (left, right)]
                metrics = {}
                if stage == 'performance':
                    for metric in ('graph_ms', 'pass_ms', 'dispatch_scope_ms'):
                        av, bv = [r['timings'][name][metric]['median'] for r in (a, b)]
                        metrics[metric] = dict(left=av, right=bv, percent=100 * (bv / av - 1))
                comparisons.append(dict(case=name, block=block, contrast=label, left=left, right=right,
                    **compare(a, b), metrics=metrics))
    controls = lighting_controls(runs) if stage == 'correctness' else []
    qualified = all(c['qualified'] for c in same_arm + comparisons if c['equivalence_required']) and all(c['qualified'] for c in controls)
    result = dict(runs=runs, comparisons=comparisons, same_arm_readbacks=same_arm,
        lighting_controls=controls, correctness_qualified=qualified, different_material_models_equivalent=False)
    if recorded is not None:
        require(result == recorded, 'Recomputed analysis differs from saved evidence')
    return result


def frozen_inputs(executable, prepared):
    identity = baseline.identity(executable, prepared / 'correctness-O.json')
    paths = {Path(p) for p in identity['binaries']} | {ROOT / p for p in identity['sources']}
    paths.update(Path(p) for p in identity['workload'])
    paths.update(p for p in prepared.rglob('*') if p.is_file())
    for extension in ('*.h', '*.inl'):
        paths.update((ROOT / 'Shaders').rglob(extension))
    paths.update((ROOT / 'Source').rglob('*.inl'))
    paths.update(Path(m.__file__).resolve() for m in (traversal, baseline, experiment, driver, native, frontend))
    paths.add(Path(__file__).resolve())
    return dict(metadata=identity, snapshots=experiment.snapshots(paths))


def run(args):
    prepared, executable = args.prepared.resolve(strict=True), args.executable.resolve(strict=True)
    verify_prepared(prepared)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Run requires a fresh output directory')
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Bounded timeout required')
    require('CMAKE_BUILD_TYPE:STRING=Release' in (executable.parent.parent / 'CMakeCache.txt').read_text(), 'Release build required')
    configs = {arm: read(prepared / f'{args.stage}-{arm}.json') for arm in ARMS}
    all_cases = [c['id'] for c in configs['O']['cases']]
    selected = args.case or all_cases
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(all_cases), 'Unknown/repeated case')
    cases = [c for c in configs['O']['cases'] if c['id'] in selected]
    selected = [c['id'] for c in cases]
    environment, removed = traversal.clean_environment()
    with traversal.experiment_lock(output):
        output.mkdir(parents=True)
        inputs = frozen_inputs(executable, prepared)
        save(output / 'Inputs.json', inputs)
        shutil.copy2(__file__, output / 'VkGltfSER.py')
        save(output / 'Environment.json', dict(removed=removed,
            fixed={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))}))
        summary = dict(schema_version=1, status='running', stage=args.stage, prepared=str(prepared),
            prepared_manifest_sha256=digest(prepared / 'Manifest.json'), arms=ARMS, selected_cases=selected,
            full_stage_coverage=selected == all_cases, schedule=list(schedule(args.stage, cases)), process_records=[],
            absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_comparison='raw-byte-exact',
            scope='GPU graph/pass/shading timestamps. One independent process median is one sample.',
            cache_policy='Separate fresh application PSO directories per arm; compiler/OS/driver caches may be warm.',
            limitations=['Cooperative lock does not prove system-wide GPU exclusivity.',
                'O/G are intentionally different material models; speed is not an equivalent-renderer speedup.',
                'Correctness runs enable core/sync validation and are not ordinary timing samples.'])
        save(output / 'Summary.json', summary)
        try:
            for name, block, arm in schedule(args.stage, cases):
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
                command = [str(executable), '--gtest_filter=RHIRendering.material_phase0_baseline',
                    '--rhi-validation' if args.stage == 'correctness' else '--rhi-no-validation',
                    '--output-dir', str(destination), f'--gtest_output=xml:{destination / "Tests.xml"}']
                print(f'{args.stage}/{name}/block-{block:02}-{arm}', flush=True)
                process = traversal.child(command, env, destination, args.timeout)
                summary['process_records'].append(dict(case=name, block=block, arm=arm, process=process))
                save(output / 'Summary.json', summary)
                inspect_run(destination, config, arm, args.stage == 'correctness', prepared)
                experiment.check_snapshots(inputs['snapshots'])
            summary['analysis'] = analyze(output, prepared, args.stage, selected)
            summary['status'] = 'passed' if summary['analysis']['correctness_qualified'] else 'numeric_failure'
        except BaseException as error:
            summary.update(status='failed', error=f'{type(error).__name__}: {error}')
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
            'Failed/incomplete or unqualified evidence')
    prepared = Path(summary['prepared'])
    verify_prepared(prepared)
    require(digest(prepared / 'Manifest.json') == summary['prepared_manifest_sha256'], 'Prepared manifest changed')
    require(summary['arms'] == ARMS and summary['absolute_tolerance'] == ABSOLUTE and
            summary['relative_tolerance'] == RELATIVE and summary['guide_comparison'] == 'raw-byte-exact', 'Numeric/arm policy changed')
    inputs = read(output / 'Inputs.json')
    require(digest(output / 'VkGltfSER.py') == inputs['snapshots'][str(Path(__file__).resolve())], 'Archived runner changed')
    # A maintained verifier may fix evidence-only defects after a capture.
    # Its archived original remains hash checked; runtime inputs may not drift.
    current_verifier = Path(__file__).resolve()
    experiment.check_snapshots({p: h for p, h in inputs['snapshots'].items() if Path(p).resolve() != current_verifier})
    configs = read(prepared / f'{summary["stage"]}-O.json')
    all_cases = [c['id'] for c in configs['cases']]
    selected = summary['selected_cases']
    cases = [c for c in configs['cases'] if c['id'] in selected]
    require(selected and selected == [c['id'] for c in cases] and
            summary['full_stage_coverage'] == (selected == all_cases), 'Selected coverage changed')
    expected = list(schedule(summary['stage'], cases))
    require(summary['schedule'] == [list(v) for v in expected] and
            [(r['case'], r['block'], r['arm']) for r in summary['process_records']] == expected, 'Process schedule incomplete/changed')
    analysis = analyze(output, prepared, summary['stage'], selected, summary['analysis'])
    require(analysis['correctness_qualified'] is (summary['status'] == 'passed'), 'Qualification contradicts status')
    return dict(verified=True, stage=summary['stage'], correctness_qualified=analysis['correctness_qualified'],
        numeric_failure=summary['status'] == 'numeric_failure', independent_processes=len(analysis['runs']),
        selected_cases=selected, full_stage_coverage=summary['full_stage_coverage'], manifest_sha256=digest(output / 'Manifest.json'),
        captured_runner_sha256=inputs['snapshots'][str(current_verifier)], current_verifier_sha256=digest(current_verifier))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--fixtures', type=Path, default=traversal.DEFAULT_TASK / 'prepared-v2')
    prep.add_argument('--output-dir', type=Path, default=TASK / 'prepared-final')
    capture = sub.add_parser('run')
    capture.add_argument('--stage', choices=('correctness', 'performance'), required=True)
    capture.add_argument('--prepared', type=Path, default=TASK / 'prepared-final')
    capture.add_argument('--executable', type=Path, default=ROOT / 'build-pass-stages-nrd/tests/MetallicRHITests.exe')
    capture.add_argument('--output-dir', type=Path, required=True)
    capture.add_argument('--timeout', type=float, default=1800)
    capture.add_argument('--case', action='append')
    check = sub.add_parser('verify')
    check.add_argument('output_dir', type=Path)
    check.add_argument('--allow-numeric-failure', action='store_true')
    args = parser.parse_args()
    result = prepare(args) if args.command == 'prepare' else run(args) if args.command == 'run' else \
        verify(args.output_dir, allow_numeric_failure=args.allow_numeric_failure)
    print(json.dumps(result, indent=2))
    if result.get('numeric_failure'):
        parser.exit(2, 'Captured evidence is intact but strict SER/same-arm numerical qualification failed.\n')


if __name__ == '__main__':
    main()
