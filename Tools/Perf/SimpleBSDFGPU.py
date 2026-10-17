"""Four-arm GPU evidence for direct/generic glTF BSDF, with EXT SER off/on."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import statistics

import VkGltfSER as shared

ROOT = shared.ROOT
TASK = ROOT / 'build-release/simple-bsdf-gpu-20261017'
ARMS = {'D0': dict(bsdf='vkGltf', shaderExecutionReorder=False),
        'S0': dict(bsdf='simple', shaderExecutionReorder=False),
        'D1': dict(bsdf='vkGltf', shaderExecutionReorder=True),
        'S1': dict(bsdf='simple', shaderExecutionReorder=True)}
CASES = ('Opaque1', 'Mixed4', 'ABeautifulGame')
SAMPLES = {'Opaque1': 1, 'Mixed4': 4, 'ABeautifulGame': 2}
CONTRASTS = {'wrapper_ser_off': ('D0', 'S0'), 'wrapper_ser_on': ('D1', 'S1'),
             'ser_direct': ('D0', 'D1'), 'ser_simple': ('S0', 'S1')}
# Even Williams design: each arm occupies each position once, and each ordered
# adjacent pair occurs once. Correctness uses only two reversed blocks.
ORDER = (('D0', 'S0', 'S1', 'D1'), ('S0', 'D1', 'D0', 'S1'),
         ('D1', 'S1', 'S0', 'D0'), ('S1', 'D0', 'D1', 'S0'))
CORRECTNESS_ORDER = (('D0', 'S0', 'D1', 'S1'), ('S1', 'D1', 'S0', 'D0'))
METRICS = ('graph_ms', 'pass_ms', 'dispatch_scope_ms')
ABSOLUTE, RELATIVE = shared.ABSOLUTE, shared.RELATIVE
TEMPORAL_FIELDS = ('captureFrames', 'guideCameraStep', 'cameraFrames', 'useRenderView')
GUIDES = ['PathTrace.' + s for s in ('albedo', 'specularAlbedo', 'normalRoughness',
          'motionVectors', 'linearDepth', 'specularHitDistance', 'depth')]
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


def schedule(stage, cases):
    require(stage in ('correctness', 'performance'), 'Unknown stage')
    for case in cases:
        for block, order in enumerate(ORDER if stage == 'performance' else CORRECTNESS_ORDER):
            for arm in order:
                yield case['id'], block, arm


def expected_request(role, arm, stage):
    raygen = stage == 'ray_generation'
    ser = ARMS[arm]['shaderExecutionReorder'] and raygen
    defines = [['METALLIC_CUSTOM_MATERIALS', '0'], ['METALLIC_GLOBAL_VIEW', '0'], ['METALLIC_HAS_RTXCR', '0'],
        ['METALLIC_VKGLTF_GUIDES', '1' if raygen else '0'],
        ['METALLIC_PT_SINGLE_SAMPLE', '1' if raygen and role == 'single' else '0'],
        ['METALLIC_VKGLTF_SER', '1' if ser else '0']]
    if raygen and ARMS[arm]['bsdf'] == 'simple':
        defines.append(['METALLIC_SIMPLE_BSDF', '1'])
    suffix = {'ray_generation': 'PathTraceMain', 'miss': 'Miss', 'closest_hit': 'ClosestHit', 'any_hit': 'AnyHit'}[stage]
    return dict(module=shared.VK_MODULE if raygen else shared.VK_HIT_MODULE, entry='vkGltfRayPipeline' + suffix,
        profile='spirv_1_6', capabilities=['spvRayTracingKHR'] + (['spvShaderInvocationReorderEXT'] if ser else []),
        defines=defines, search_paths=[])


def validate_requests(requests):
    require(set(requests.get('roles', {})) == {'single', 'general'}, 'Both factory roles required')
    for role, arms in requests['roles'].items():
        require(set(arms) == set(ARMS), 'All four factory arms required')
        for arm, item in arms.items():
            require(set(item) == {'stages'} and set(item['stages']) == set(traversal.RT_STAGES), 'Four factory stages required')
            for stage, request in item['stages'].items():
                require(request == expected_request(role, arm, stage), f'Factory request changed: {role}/{arm}/{stage}')


def verify_reference(reference, requests_path):
    require(reference['schema_version'] == 1 and reference['factory_requests_sha256'] == digest(requests_path) and
            digest(reference['requests_path']) == digest(requests_path), 'Reference factory request identity changed')
    require(set(reference['roles']) == {'single', 'general'}, 'Reference requires both roles')
    paths = {Path(reference['requests_path'])}
    experiment.check_snapshots(reference['tools'])
    paths.update(Path(p) for p in reference['tools'])
    hit_hashes = {stage: set() for stage in ('miss', 'closest_hit', 'any_hit')}
    for role, arms in reference['roles'].items():
        require(set(arms) == set(ARMS), 'Reference requires all four arms')
        for arm, item in arms.items():
            require(set(item['stages']) == set(traversal.RT_STAGES), 'Reference requires all four stages')
            for stage, payload in item['stages'].items():
                path = Path(payload['path'])
                require(path.is_absolute() and digest(path) == payload['sha256'], 'Reference SPIR-V hash/path changed')
                expected = ['SPV_EXT_shader_invocation_reorder'] if stage == 'ray_generation' and ARMS[arm]['shaderExecutionReorder'] else []
                extensions = [e for e in shared.spirv_extensions(path.read_bytes()) if 'invocation_reorder' in e]
                require(extensions == expected, 'Reference EXT SER identity changed')
                if stage in hit_hashes:
                    hit_hashes[stage].add(payload['sha256'])
                paths.add(path)
    require(all(len(hashes) == 1 for hashes in hit_hashes.values()), 'Reference shared hit stages differ')
    return paths


def derived_config(source, arm, stage):
    config = read(source / 'correctness-G.json')
    config['cases'] = [c for c in config['cases'] if c['id'] in CASES]
    require([c['id'] for c in config['cases']] == list(CASES), 'Missing declared cases')
    if stage == 'performance':
        config.update(frames=97, warmupFrames=32, timingFrames=64, synchronizationValidation=False)
    for spec in config['cases']:
        spec.update(graph=f'{arm}/{spec["id"]}.json', width=512, height=512)
        if stage == 'performance':
            for key in TEMPORAL_FIELDS:
                spec.pop(key, None)
            spec['frameCount'] = 97
        else:
            require(spec.get('captureFrames') and spec.get('captureOutputs') == GUIDES,
                    'Temporal/seven-guide coverage missing')
    return config


def prepare(args):
    source, requests_path = args.fixtures.resolve(strict=True), args.requests.resolve(strict=True)
    identity_path = (args.request_identity or requests_path.with_name('ExporterIdentity.json')).resolve(strict=True)
    shared.verify_prepared(source)
    requests, identity = read(requests_path), read(identity_path)
    validate_requests(requests)
    require(isinstance(identity, dict) and identity, 'Exporter identity snapshots required')
    experiment.check_snapshots(identity)
    reference_path = args.reference_spirv.resolve(strict=True)
    reference = read(reference_path)
    reference_paths = verify_reference(reference, requests_path)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Prepare requires a fresh directory')
    source_inputs = experiment.snapshots([*(p for p in source.rglob('*') if p.is_file()),
        requests_path, identity_path, *(Path(p) for p in identity), reference_path, *reference_paths])
    output.mkdir(parents=True)
    for arm, properties in ARMS.items():
        (output / arm).mkdir()
        for spec in read(source / 'correctness-G.json')['cases']:
            if spec['id'] not in CASES:
                continue
            graph = read(source / spec['graph'])
            traversal.path_node(graph)['properties'].update(properties)
            save(output / arm / f'{spec["id"]}.json', graph)
        for stage in ('correctness', 'performance'):
            save(output / f'{stage}-{arm}.json', derived_config(source, arm, stage))
    save(output / 'FactoryRequests.json', requests)
    save(output / 'ExporterIdentity.json', identity)
    save(output / 'Reference.json', reference)
    save(output / 'Prepared.json', dict(schema_version=1, experiment='simple_bsdf_gpu', arms=ARMS, cases=CASES,
        contrasts=CONTRASTS, source=str(source), source_inputs=source_inputs, correctness_order=CORRECTNESS_ORDER,
        performance_order=ORDER, requests_source=str(requests_path), request_identity_source=str(identity_path),
        reference_source=str(reference_path),
        absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_requirement='raw-byte-exact',
        transformation='Both stages 512x512. Correctness preserves temporal/camera/seven guides. Performance removes temporal capture/view controls, times frames 32..95, captures final HDR at 96, and rerenders seven guide readbacks at 97..103 outside timing.',
        exclusions=['No OpenPBR arm. No native PSO/compiler timing. No local/environment contribution-control cases.']))
    seal(output)
    return verify_prepared(output)


def verify_prepared(directory):
    directory = Path(directory)
    verify_seal(directory)
    metadata = read(directory / 'Prepared.json')
    require(metadata['schema_version'] == 1 and metadata['experiment'] == 'simple_bsdf_gpu' and
            metadata['arms'] == ARMS and metadata['cases'] == list(CASES) and
            metadata['contrasts'] == {k: list(v) for k, v in CONTRASTS.items()} and
            metadata['correctness_order'] == [list(v) for v in CORRECTNESS_ORDER] and
            metadata['performance_order'] == [list(v) for v in ORDER], 'Prepared contract changed')
    require(metadata['absolute_tolerance'] == ABSOLUTE and metadata['relative_tolerance'] == RELATIVE and
            metadata['guide_requirement'] == 'raw-byte-exact', 'Fixed numeric contract changed')
    experiment.check_snapshots(metadata['source_inputs'])
    source = Path(metadata['source'])
    shared.verify_prepared(source)
    requests = read(directory / 'FactoryRequests.json')
    validate_requests(requests)
    require(requests == read(metadata['requests_source']) and
            read(directory / 'ExporterIdentity.json') == read(metadata['request_identity_source']), 'Factory provenance changed')
    reference = read(directory / 'Reference.json')
    require(reference == read(metadata['reference_source']), 'Reference provenance changed')
    verify_reference(reference, directory / 'FactoryRequests.json')
    for arm in ARMS:
        for stage in ('correctness', 'performance'):
            config = read(directory / f'{stage}-{arm}.json')
            require(config == derived_config(source, arm, stage), 'Prepared workload differs from declared derivation')
        for spec in read(source / 'correctness-G.json')['cases']:
            if spec['id'] not in CASES:
                continue
            graph = read(directory / arm / f'{spec["id"]}.json')
            require(normalized_graph(graph, arm) == shared.normalized_graph(read(source / spec['graph']), 'G'),
                    'Prepared graph differs beyond declared arm')
            props = traversal.path_node(graph)['properties']
            require(props.get('exportDenoiserGuides') is True and
                    props.get('samples', 1) == SAMPLES[spec['id']], 'Single/general guide workload changed')
    return metadata


def recipe_evidence(directory, arm, validation, spec, graph, requests, reference):
    normalized_graph(graph, arm)
    result = shared.recipe_evidence(directory, 'S' if ARMS[arm]['shaderExecutionReorder'] else 'G', validation, spec, graph)
    role = 'single' if traversal.path_node(graph)['properties'].get('samples', 1) == 1 else 'general'
    for stage, shader in result['target']['shaders'].items():
        request = requests['roles'][role][arm]['stages'][stage]
        require(shader['module_name'] == request['module'] and shader['source_entry_point'] == request['entry'],
                'Actual shader family differs from exported factory')
        require(shader['sha256'] == reference['roles'][role][arm]['stages'][stage]['sha256'],
                f'Actual SPIR-V differs from factory reference: {role}/{arm}/{stage}')
        reorder_extensions = [e for e in shader['extensions'] if 'invocation_reorder' in e]
        expected = ['SPV_EXT_shader_invocation_reorder'] if stage == 'ray_generation' and ARMS[arm]['shaderExecutionReorder'] else []
        require(reorder_extensions == expected, 'Only requested EXT SER is permitted; NV/shared-hit SER is forbidden')
    return result


def require_shared_policy(left, right):
    require(left['device'] == right['device'], 'Device/features/validation changed')
    require(left['unrelated'] == right['unrelated'], 'Non-target PSO changed')
    states = [copy.deepcopy(r['target']['state']) for r in (left, right)]
    for state in states:
        state.pop('enable_invocation_reorder', None)
    require(states[0] == states[1], 'Native target state changed beyond SER')
    require(left['target']['cache_group'] == right['target']['cache_group'], 'Target cache group changed')
    shared.require_shared_hit_stages(left['target']['shaders'], right['target']['shaders'])
    # Runtime payload/hash/size and declared EXT may differ; compiler/resource policy may not.
    policies = []
    for result in (left, right):
        policy = copy.deepcopy(result['target']['shaders']['ray_generation'])
        for field in ('sha256', 'input_spirv_hash', 'spirv_bytes', 'extensions'):
            policy.pop(field, None)
        policies.append(policy)
    require(policies[0] == policies[1], 'Raygen policy changed beyond bytecode/declared SER')


def gpu_timings(report, configuration):
    require(configuration['warmupFrames'] == 32 and configuration['timingFrames'] == 64 and
            configuration['frames'] == 97 and configuration['synchronizationValidation'] is False and
            report['validation'] is False and report['synchronizationValidation'] is False,
            'Ordinary timing requires 32 warmup/64 timed/97 frames and no validation')
    for case in report['cases']:
        warmup, timed = case['warmup'], case['frames']
        require([r['frame'] for r in warmup] == list(range(32)) and all(r['warmup'] is True for r in warmup),
                'Incomplete/reordered/mislabeled warmup window')
        require(len({r['executionId'] for r in warmup + timed}) == 96,
                'Warmup/timed execution IDs must be unique across both windows')
    return experiment.gpu_timings(report, configuration)


def inspect_run(destination, configuration, arm, validation, prepared, requests, reference):
    process = read(destination / 'process.json')
    require(process['exit_code'] == 0 and not process['timed_out'], 'Failed or timed-out process')
    traversal.validate_xml(destination / 'Tests.xml')
    path = destination / 'MaterialBaseline.json'
    report = read(path)
    require(report['validation'] is validation and report.get('synchronizationValidation') is validation and
            len(report['cases']) == 1, 'Runtime validation/case set changed')
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
        recipes=recipe_evidence(destination / 'recipes', arm, validation, spec, graph, requests, reference),
        timings=gpu_timings(report, configuration) if not validation else None)


def compare(left, right):
    import numpy as np
    lp, rp = Path(left['report']), Path(right['report'])
    a, b = read(lp)['cases'][0], read(rp)['cases'][0]
    require(normalized_graph(a['resolvedGraph'], left['arm']) == normalized_graph(b['resolvedGraph'], right['arm']),
            'Compared graph changed beyond declared arms')
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


def qualifications(comparisons, repeats, selected):
    result = {}
    for case in selected:
        result[case] = {}
        for contrast, arms in CONTRASTS.items():
            pairs = [c for c in comparisons if c['case'] == case and c['contrast'] == contrast]
            same = [r for r in repeats if r['case'] == case and r['arm'] in arms]
            require(pairs and same and set(r['arm'] for r in same) == set(arms), 'Missing contrast/repeat qualification')
            result[case][contrast] = all(r['qualified'] for r in pairs + same)
    return result


def statistics_for(runs, selected):
    require(len(runs) == len(selected) * len(ARMS) * len(ORDER), 'Unexpected measurement row count')
    result = {}
    for case in selected:
        result[case] = {}
        for metric in METRICS:
            values = {}
            for arm in ARMS:
                rows = [r for r in runs if r['case'] == case and r['arm'] == arm]
                require([r['block'] for r in rows] == list(range(len(ORDER))), 'Four ordered independent process samples required')
                values[arm] = [r['timings'][case][metric]['median'] for r in rows]
                require(all(math.isfinite(v) and v > 0 for v in values[arm]), 'Finite positive timing samples required')
            contrasts = {}
            for label, (left, right) in CONTRASTS.items():
                changes = [100 * (b / a - 1) for a, b in zip(values[left], values[right])]
                contrasts[label] = dict(left=left, right=right, paired_percent=changes,
                    paired_median_percent=statistics.median(changes), paired_min_percent=min(changes), paired_max_percent=max(changes))
            result[case][metric] = dict(values_ms=values, medians_ms={a: statistics.median(v) for a, v in values.items()},
                contrasts=contrasts, unit='One independent process median; 64 frames are not independent samples.')
    return result


def analyze(output, prepared, stage, selected, recorded=None):
    configs = {arm: read(prepared / f'{stage}-{arm}.json') for arm in ARMS}
    requests = read(prepared / 'FactoryRequests.json')
    reference = read(prepared / 'Reference.json')
    cases = [c for c in configs['D0']['cases'] if c['id'] in selected]
    runs, repeats, comparisons, first = [], [], [], {}
    for name, block, arm in schedule(stage, cases):
        destination = output / name / f'block-{block:02}-{arm}'
        expected = copy.deepcopy(configs[arm])
        expected['cases'] = [c for c in expected['cases'] if c['id'] == name]
        expected['cases'][0]['graph'] = str((prepared / expected['cases'][0]['graph']).resolve())
        require(read(destination / 'Cases.json') == expected, 'Captured configuration changed')
        run = dict(case=name, block=block, arm=arm,
            **inspect_run(destination, expected, arm, stage == 'correctness', prepared, requests, reference))
        if (name, arm) in first:
            traversal.require_stable_recipes(first[name, arm], run)
            repeats.append(dict(case=name, block=block, arm=arm, **compare(first[name, arm], run)))
        else:
            first[name, arm] = run
        runs.append(run)
    indexed = {(r['case'], r['block'], r['arm']): r for r in runs}
    for name in selected:
        for block in range(len(ORDER if stage == 'performance' else CORRECTNESS_ORDER)):
            for contrast, (left, right) in CONTRASTS.items():
                comparisons.append(dict(case=name, block=block, contrast=contrast, left=left, right=right,
                    **compare(indexed[name, block, left], indexed[name, block, right])))
    per_case = qualifications(comparisons, repeats, selected)
    per_contrast = {label: all(per_case[case][label] for case in selected) for label in CONTRASTS}
    result = dict(runs=runs, comparisons=comparisons, same_arm_readbacks=repeats, case_contrast_qualified=per_case,
        contrast_qualified=per_contrast, correctness_qualified=all(per_contrast.values()),
        statistics=statistics_for(runs, selected) if stage == 'performance' else {})
    if recorded is not None:
        require(result == recorded, 'Recomputed analysis differs from recorded evidence')
    return result


def frozen_inputs(executable, prepared):
    identity = baseline.identity(executable, prepared / 'correctness-D0.json')
    paths = {Path(p) for p in identity['binaries']} | {ROOT / p for p in identity['sources']}
    paths.update(Path(p) for p in identity['workload'])
    paths.update(p for p in prepared.rglob('*') if p.is_file())
    paths.update(Path(p) for p in read(prepared / 'Prepared.json')['source_inputs'])
    for extension in ('*.h', '*.inl'):
        paths.update((ROOT / 'Shaders').rglob(extension))
    paths.update((ROOT / 'Source').rglob('*.inl'))
    paths.update(Path(m.__file__).resolve() for m in (shared, traversal, baseline, experiment, driver, native, frontend))
    paths.add(Path(__file__).resolve())
    return dict(metadata=identity, snapshots=experiment.snapshots(paths))


def check_correctness_link(summary, inputs, prepared, allow_unqualified):
    require(summary['experiment'] == 'simple_bsdf_gpu' and summary['stage'] == 'correctness' and
            summary['status'] in ('passed', 'numeric_failure') and summary['full_stage_coverage'] and
            summary['selected_cases'] == list(CASES), 'Complete sealed correctness capture required')
    require(summary['prepared_manifest_sha256'] == digest(prepared / 'Manifest.json'), 'Correctness prepared workload differs')
    require(summary['analysis']['correctness_qualified'] or allow_unqualified,
            'Correctness is unqualified; explicit diagnostic override required')
    require(inputs['snapshots'] == read(Path(summary['capture_path']) / 'Inputs.json')['snapshots'],
            'Correctness and timing must use identical frozen sources, tools, fixtures and binaries')


def load_correctness(path, inputs, prepared, allow_unqualified):
    path = Path(path).resolve(strict=True)
    verification = verify(path, allow_numeric_failure=allow_unqualified)
    summary = read(path / 'Summary.json')
    check_correctness_link(dict(summary, capture_path=str(path)), inputs, prepared, allow_unqualified)
    return dict(path=str(path), manifest_sha256=digest(path / 'Manifest.json'), summary_sha256=digest(path / 'Summary.json'),
        correctness_qualified=verification['correctness_qualified'], contrast_qualified=verification['contrast_qualified'],
        case_contrast_qualified=summary['analysis']['case_contrast_qualified'],
        diagnostic_override=bool(allow_unqualified), full_stage_coverage=True)


def require_linked_identities(analysis, correctness):
    previous = read(Path(correctness['path']) / 'Summary.json')['analysis']['runs']
    by_key = {(r['case'], r['arm']): r for r in previous if r['block'] == 0}
    for run in analysis['runs']:
        a, b = by_key[run['case'], run['arm']]['recipes'], run['recipes']
        require(a['stable'] == b['stable'], 'Timing actual shader/native identity differs from correctness')
        devices = [copy.deepcopy(r['device']) for r in (a, b)]
        for device in devices:
            device.pop('validation_enabled', None)
            device.pop('synchronization_validation_enabled', None)
        require(devices[0] == devices[1], 'Timing device/features differ from correctness')


def qualification(analysis, correctness, stage, complete):
    per_case = copy.deepcopy(analysis['case_contrast_qualified'])
    if stage == 'performance':
        for case, contrasts in per_case.items():
            for label in contrasts:
                contrasts[label] = contrasts[label] and correctness['case_contrast_qualified'][case][label]
    per_contrast = {label: all(v[label] for v in per_case.values()) for label in CONTRASTS}
    globally = all(per_contrast.values()) and (stage == 'correctness' or correctness['correctness_qualified'])
    return dict(correctness_qualified=globally, contrast_qualified=per_contrast, case_contrast_qualified=per_case,
        performance_qualified=stage == 'performance' and complete and globally,
        diagnostic_timing=stage == 'performance' and not globally)


def expected_command(executable, destination, stage):
    return [str(executable), '--gtest_filter=RHIRendering.material_phase0_baseline',
        '--rhi-validation' if stage == 'correctness' else '--rhi-no-validation',
        '--output-dir', str(destination), f'--gtest_output=xml:{destination / "Tests.xml"}']


def run(args):
    prepared, executable = args.prepared.resolve(strict=True), args.executable.resolve(strict=True)
    verify_prepared(prepared)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Run requires a fresh directory')
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Bounded timeout required')
    require('CMAKE_BUILD_TYPE:STRING=Release' in (executable.parent.parent / 'CMakeCache.txt').read_text(), 'Release build required')
    require(args.stage == 'performance' or (args.correctness is None and not args.allow_unqualified_correctness),
            'Correctness does not accept a correctness link/override')
    require(args.stage != 'performance' or args.correctness is not None, 'Performance requires a linked correctness capture')
    configs = {arm: read(prepared / f'{args.stage}-{arm}.json') for arm in ARMS}
    selected = args.case or list(CASES)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(CASES), 'Unknown/repeated case')
    cases = [c for c in configs['D0']['cases'] if c['id'] in selected]
    selected = [c['id'] for c in cases]
    environment, removed = traversal.clean_environment()
    with traversal.experiment_lock(output):
        inputs = frozen_inputs(executable, prepared)
        correctness = load_correctness(args.correctness, inputs, prepared, args.allow_unqualified_correctness) if args.stage == 'performance' else None
        output.mkdir(parents=True)
        save(output / 'Inputs.json', inputs)
        shutil.copy2(__file__, output / 'SimpleBSDFGPU.py')
        save(output / 'Environment.json', dict(removed=removed,
            fixed={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))}))
        summary = dict(schema_version=1, experiment='simple_bsdf_gpu', status='running', stage=args.stage, arms=ARMS,
            executable=str(executable), prepared=str(prepared), prepared_manifest_sha256=digest(prepared / 'Manifest.json'),
            selected_cases=selected, full_stage_coverage=selected == list(CASES), schedule=list(schedule(args.stage, cases)),
            process_records=[], correctness=correctness, allow_unqualified_correctness=bool(args.allow_unqualified_correctness),
            absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_comparison='raw-byte-exact',
            scope='GPU graph/pass/shading timestamps, no validation for timing. One independent process median is one sample.',
            cache_policy='Fresh application PSO directory per arm; shared by that arm across blocks. Compiler/OS/driver caches may be warm.',
            limitations=['Cooperative lock does not establish system-wide GPU exclusivity.',
                'Correctness has core/sync validation and is not a timing sample.',
                'Four independent timing samples per arm/case; no end-to-end, PSO or compiler attribution.',
                'Cross-SER failure does not erase independently qualified same-SER wrapper contrasts.'])
        save(output / 'Summary.json', summary)
        requests = read(prepared / 'FactoryRequests.json')
        reference = read(prepared / 'Reference.json')
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
                print(f'{args.stage}/{name}/block-{block:02}-{arm}', flush=True)
                process = traversal.child(expected_command(executable, destination, args.stage), env, destination, args.timeout)
                summary['process_records'].append(dict(case=name, block=block, arm=arm, process=process))
                save(output / 'Summary.json', summary)
                inspect_run(destination, config, arm, args.stage == 'correctness', prepared, requests, reference)
                experiment.check_snapshots(inputs['snapshots'])
            analysis = analyze(output, prepared, args.stage, selected)
            if correctness:
                require_linked_identities(analysis, correctness)
            summary['analysis'] = analysis
            summary['qualification'] = qualification(analysis, correctness, args.stage, summary['full_stage_coverage'])
            summary['status'] = 'passed' if summary['qualification']['correctness_qualified'] else 'numeric_failure'
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
            'Incomplete/failed or numerically unqualified capture')
    require(summary['schema_version'] == 1 and summary['experiment'] == 'simple_bsdf_gpu' and
            summary['stage'] in ('correctness', 'performance') and summary['arms'] == ARMS, 'Experiment policy changed')
    require(summary['absolute_tolerance'] == ABSOLUTE and summary['relative_tolerance'] == RELATIVE and
            summary['guide_comparison'] == 'raw-byte-exact', 'Numeric policy changed')
    prepared, stage = Path(summary['prepared']), summary['stage']
    verify_prepared(prepared)
    require(digest(prepared / 'Manifest.json') == summary['prepared_manifest_sha256'], 'Prepared identity changed')
    inputs, current = read(output / 'Inputs.json'), Path(__file__).resolve()
    require(digest(output / current.name) == inputs['snapshots'][str(current)], 'Archived runner changed')
    experiment.check_snapshots({p: h for p, h in inputs['snapshots'].items() if Path(p).resolve() != current})
    correctness = None
    if stage == 'performance':
        correctness = load_correctness(summary['correctness']['path'], inputs, prepared, summary['allow_unqualified_correctness'])
        require(correctness == summary['correctness'], 'Correctness link changed')
    else:
        require(summary['correctness'] is None and summary['allow_unqualified_correctness'] is False, 'Unexpected correctness override')
    config = read(prepared / f'{stage}-D0.json')
    selected = summary['selected_cases']
    cases = [c for c in config['cases'] if c['id'] in selected]
    require(selected and selected == [c['id'] for c in cases] and
            summary['full_stage_coverage'] == (selected == list(CASES)), 'Coverage changed')
    expected = list(schedule(stage, cases))
    require(summary['schedule'] == [list(v) for v in expected] and
            [(r['case'], r['block'], r['arm']) for r in summary['process_records']] == expected, 'Incomplete/changed schedule')
    last_end = None
    environment = read(output / 'Environment.json')['fixed']
    expected_env = {k: v for k, v in traversal.clean_environment()[0].items() if k.startswith(('METALLIC_', 'VK_'))}
    require(environment == expected_env, 'Controlled environment changed')
    for row in summary['process_records']:
        destination = output / row['case'] / f'block-{row["block"]:02}-{row["arm"]}'
        process = read(destination / 'process.json')
        env = dict(environment, METALLIC_MATERIAL_BASELINE_CASES=str(destination / 'Cases.json'),
            METALLIC_PIPELINE_RECIPE_DIRECTORY=str(destination / 'recipes'),
            METALLIC_PSO_CACHE_DIRECTORY=str(output / 'application-pso-cache' / row['arm']))
        require(process == row['process'] and process['command'] == expected_command(Path(summary['executable']), destination, stage) and
                process['environment'] == env and process['cwd'] == str(ROOT), 'Process record/command/environment changed')
        require(math.isfinite(process['timeout_seconds']) and 0 < process['timeout_seconds'] <= 7200, 'Invalid process timeout')
        start, end = [datetime.fromisoformat(process[k]) for k in ('started_utc', 'ended_utc')]
        require(isinstance(process.get('pid'), int) and process['pid'] > 0 and end > start and
                (last_end is None or start >= last_end), 'Overlapping/invalid process timing/PID')
        last_end = end
    analysis = analyze(output, prepared, stage, selected, summary['analysis'])
    if correctness:
        require_linked_identities(analysis, correctness)
    qualified = qualification(analysis, correctness, stage, summary['full_stage_coverage'])
    require(qualified == summary['qualification'] and qualified['correctness_qualified'] is (summary['status'] == 'passed'),
            'Qualification contradicts recorded evidence')
    return dict(verified=True, stage=stage, **qualified, numeric_failure=summary['status'] == 'numeric_failure',
        independent_processes=len(analysis['runs']), selected_cases=selected, full_stage_coverage=summary['full_stage_coverage'],
        manifest_sha256=digest(output / 'Manifest.json'), captured_runner_sha256=inputs['snapshots'][str(current)],
        current_verifier_sha256=digest(current))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--fixtures', type=Path, default=shared.TASK / 'prepared-final')
    prep.add_argument('--requests', type=Path, required=True)
    prep.add_argument('--request-identity', type=Path)
    prep.add_argument('--reference-spirv', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, default=TASK / 'prepared')
    capture = sub.add_parser('run')
    capture.add_argument('--stage', choices=('correctness', 'performance'), required=True)
    capture.add_argument('--prepared', type=Path, default=TASK / 'prepared')
    capture.add_argument('--executable', type=Path, default=ROOT / 'build-pass-stages-nrd/tests/MetallicRHITests.exe')
    capture.add_argument('--output-dir', type=Path, required=True)
    capture.add_argument('--correctness', type=Path)
    capture.add_argument('--allow-unqualified-correctness', action='store_true')
    capture.add_argument('--timeout', type=float, default=1800)
    capture.add_argument('--case', action='append')
    check = sub.add_parser('verify')
    check.add_argument('output_dir', type=Path)
    check.add_argument('--allow-numeric-failure', action='store_true')
    args = parser.parse_args()
    result = prepare(args) if args.command == 'prepare' else run(args) if args.command == 'run' else verify(args.output_dir, args.allow_numeric_failure)
    print(json.dumps(result, indent=2))
    if result.get('numeric_failure'):
        parser.exit(2, 'Strict numeric qualification failed; intact diagnostics retained, thresholds unchanged.\n')


if __name__ == '__main__':
    main()
