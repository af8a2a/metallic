"""Prepare, run, and verify production OpenPBR ray-query / ray-pipeline A/B evidence.

Only prepare is non-GPU. Runs use one independent RHITests process per case/arm,
actual captured pipeline recipes, fixed readback gates, and cooperative GPU locks.
No source swaps, artifact overrides, or archived experiment scripts are executed.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

import MaterialBaseline as baseline
import RunComputePipelineExperiments as experiment
import RunShaderPipelineDriver as driver

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / 'build-release/openpbr-ray-pipeline-20261017'
DEFAULT_INPUT = ROOT / 'build-release/openpbr-nee-shared-20261017/FixturesV2'
ARMS = {'A': 'rayQuery', 'B': 'rayPipeline'}
ORDER = (('A', 'B'), ('B', 'A'), ('A', 'B'))
ABSOLUTE, RELATIVE = 2e-5, 2e-5
RT_STAGES = ('ray_generation', 'miss', 'closest_hit', 'any_hit')
QUERY_MODULES = ('Features/PathTracing/OpenPBRRayQueryPathTrace',
                 'Features/PathTracing/OpenPBRRayQueryPathTraceGuides')
RT_MODULES = ('Features/PathTracing/OpenPBRRayPipelinePathTrace',
              'Features/PathTracing/OpenPBRRayPipelinePathTraceGuides')
read, save, digest = experiment.read, experiment.save, experiment.digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_manifest(directory):
    return {str(p.relative_to(directory)).replace('\\', '/'): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and p != directory / 'Manifest.json'}


def seal(directory):
    save(directory / 'Manifest.json', file_manifest(directory))


def verify_seal(directory):
    require(read(directory / 'Manifest.json') == file_manifest(directory), f'Evidence changed: {directory}')


def path_node(graph):
    nodes = [n for n in graph['nodes'] if n['type'] == 'ScenePathTracePass']
    require(len(nodes) == 1, 'Each workload must contain exactly one ScenePathTracePass')
    require(nodes[0]['name'] == 'PathTrace', 'Unexpected path-tracing node name')
    return nodes[0]


def normalized_graph(graph, arm):
    graph = copy.deepcopy(graph)
    props = path_node(graph)['properties']
    require(props.pop('traversalBackend', None) == ARMS[arm], 'Requested traversal backend missing or changed')
    require(props.get('cacheMode') == 'off' and props.get('bsdf') == 'openpbr', 'Only OpenPBR/cache-off is qualified')
    return graph


def expected_resolved_graph(graph, specification):
    # Mirror only the existing MaterialBaseline serialization contract: default
    # editor positions are emitted and RenderView takes ownership of the camera.
    result = copy.deepcopy(graph)
    for node in result['nodes']:
        node.setdefault('position', {'x': 0.0, 'y': 0.0})
        if specification.get('useRenderView', False) and node['name'] == specification.get('renderViewCameraNode', 'PathTrace'):
            node['properties'].pop('camera')
    return result


def prepare(args):
    source = args.fixtures.resolve(strict=True)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Prepare requires a new output directory')
    original = read(source / 'Correctness.json')
    performance_source = read(source / 'Performance.json')
    source_paths = {source / 'Correctness.json', source / 'Performance.json', source / 'Coverage.json'}
    source_paths.update(Path(p).resolve(strict=True) for p in original['assetFiles'])
    source_paths.update(source / c['graph'] for c in original['cases'])
    before = experiment.snapshots(source_paths)
    correctness, exclusions, graphs = [], [], {}
    for spec in original['cases']:
        graph = read(source / spec['graph'])
        props = path_node(graph)['properties']
        reason = 'SHARC is outside the initial RT backend' if 'sharc' in spec['id'].lower() else \
            'Legacy glTF BSDF remains on the original traversal path' if props.get('bsdf') != 'openpbr' else None
        if reason:
            exclusions.append({'case': spec['id'], 'reason': reason})
            continue
        graph = copy.deepcopy(graph)
        path_node(graph)['properties']['cacheMode'] = 'off'
        graphs[spec['id']] = graph
        correctness.append(copy.deepcopy(spec))
    ids = {c['id'] for c in correctness}
    require({'Opaque1', 'Mixed1', 'Mixed4', 'MixedNoGuides', 'ABeautifulGame', 'MixedNoEnvironment',
             'MixedNoEnvironmentNoDirect', 'MixedEnvironmentOnly', 'MixedEnvironmentOnlyNoDirect'} <= ids,
            'Required coverage cases are missing')
    require(any(c.get('cameraFrames') and len(c.get('captureOutputs', [])) == 7 for c in correctness),
            'Moving camera and seven-guide coverage are required')
    materials = read(source / 'Queries/Mixed/Scene.gltf')['materials']
    require({'MASK', 'BLEND'} <= {m.get('alphaMode', 'OPAQUE') for m in materials}, 'MASK/BLEND fixture missing')
    iors = {m.get('extensions', {}).get('KHR_materials_ior', {}).get('ior') for m in materials}
    require({1.5, 1.33} <= iors, 'Nested-IOR material fixture missing')
    performance = []
    for name in ('Opaque1', 'Mixed1', 'ABeautifulGame'):
        collection = performance_source['cases'] if name != 'ABeautifulGame' else correctness
        spec = copy.deepcopy(next(c for c in collection if c['id'] == name))
        for field in ('cameraFrames', 'captureFrames', 'captureOutputs', 'guideCameraStep', 'useRenderView'):
            spec.pop(field, None)
        spec.update(width=512, height=512, frameCount=97, requiredTiming=['PathTrace', 'Path trace shading'])
        performance.append(spec)
    output.mkdir(parents=True)
    for arm in ARMS:
        (output / arm).mkdir()
        for name, graph in graphs.items():
            actual = copy.deepcopy(graph)
            path_node(actual)['properties']['traversalBackend'] = ARMS[arm]
            save(output / arm / f'{name}.json', actual)
        for stage, specs in (('correctness', correctness), ('performance', performance)):
            config = dict(version=2, frames=16 if stage == 'correctness' else 97,
                          warmupFrames=4 if stage == 'correctness' else 32,
                          timingFrames=4 if stage == 'correctness' else 64,
                          synchronizationValidation=stage == 'correctness',
                          assetFiles=original['assetFiles'], cases=copy.deepcopy(specs))
            for spec in config['cases']:
                spec['graph'] = f'{arm}/{spec["id"]}.json'
            save(output / f'{stage}-{arm}.json', config)
    for name in graphs:
        require(normalized_graph(read(output / 'A' / f'{name}.json'), 'A') ==
                normalized_graph(read(output / 'B' / f'{name}.json'), 'B'), 'A/B graph differs beyond traversalBackend')
    shutil.copy2(source / 'Coverage.json', output / 'OriginalCoverage.json')
    save(output / 'Prepared.json', dict(schema_version=1, source=str(source), source_inputs=before,
        exclusions=exclusions, correctness_cases=[c['id'] for c in correctness],
        performance_cases=[c['id'] for c in performance], arms=ARMS,
        graph_difference='Only PathTrace.properties.traversalBackend differs between A and B; both cacheMode=off.',
        performance_order=[list(v) for v in ORDER], absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE,
        guide_requirement='Raw byte exact; no threshold adjustment after capture.',
        coverage='Authored MASK/BLEND, IOR 1.5/1.33, colored transmission, moving camera, seven guides, local/environment isolation. SHARC and legacy glTF excluded.',
        script_sha256=digest(__file__)))
    experiment.check_snapshots(before)
    seal(output)
    verify_prepared(output)
    print(output / 'Prepared.json')


def verify_prepared(directory):
    verify_seal(directory)
    metadata = read(directory / 'Prepared.json')
    experiment.check_snapshots(metadata['source_inputs'])
    require(metadata['arms'] == ARMS and metadata['absolute_tolerance'] == ABSOLUTE and
            metadata['relative_tolerance'] == RELATIVE, 'Prepared experiment policy changed')
    for stage in ('correctness', 'performance'):
        a, b = (read(directory / f'{stage}-{arm}.json') for arm in ARMS)
        require(a['synchronizationValidation'] is (stage == 'correctness') and
                b['synchronizationValidation'] is (stage == 'correctness'), 'Synchronization-validation policy changed')
        require(len(a['cases']) == len(b['cases']), 'Prepared case count mismatch')
        for left, right in zip(a['cases'], b['cases']):
            ll, rr = copy.deepcopy(left), copy.deepcopy(right)
            ga, gb = ll.pop('graph'), rr.pop('graph')
            require(ll == rr, 'A/B case differs beyond graph filename')
            require(normalized_graph(read(directory / ga), 'A') == normalized_graph(read(directory / gb), 'B'),
                    'A/B graph differs beyond traversal backend')
    return metadata


@contextmanager
def experiment_lock(output):
    directory = ROOT / 'build'
    directory.mkdir(exist_ok=True)
    own = directory / 'compute-pipeline-experiment.lock'
    token = {'pid': os.getpid(), 'token': uuid.uuid4().hex, 'output': str(output), 'owner': str(Path(__file__).resolve())}
    for name in experiment.LOCK_NAMES:
        require(not (directory / name).exists(), f'GPU experiment lock exists: {directory / name}')
    with own.open('x', encoding='utf-8') as stream:
        json.dump(token, stream)
    try:
        for name in experiment.LOCK_NAMES:
            require(name == own.name or not (directory / name).exists(), f'Concurrent GPU experiment: {name}')
        yield
    finally:
        if own.exists() and read(own) == token:
            own.unlink()


def clean_environment():
    prefixes = ('METALLIC_', 'VK_', 'NV_', '__GL_', 'NSIGHT_', 'NVPERF_')
    environment = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    removed = sorted(k for k in os.environ if k.startswith(prefixes))
    environment.update(METALLIC_SHADER_DEBUG_MODE='disabled', METALLIC_SLANG_OPTIMIZATION='O1',
        METALLIC_SPIRV_POSTPROCESS='none', METALLIC_VK_COMPUTE_BACKEND='pipeline',
        METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION='0', METALLIC_VK_INTERNAL_PIPELINE_CACHE='enabled',
        METALLIC_NSIGHT_GRAPHICS_CAPTURE='0', METALLIC_SHADER_CAPTURE_SYMBOLS='0',
        METALLIC_WORKING_COLOR_SPACE='acescg')
    return environment, removed


def child(command, environment, destination, timeout):
    record = dict(command=command, cwd=str(ROOT), timeout_seconds=timeout,
        started_utc=datetime.now(timezone.utc).isoformat(), timed_out=False,
        environment={k: v for k, v in environment.items() if k.startswith(('METALLIC_', 'VK_'))})
    process = None
    try:
        with (destination / 'stdout.log').open('wb') as stdout, (destination / 'stderr.log').open('wb') as stderr:
            with driver.suppress_windows_error_dialogs():
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                try:
                    record['pid'] = process.pid
                    try:
                        record['exit_code'] = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        record['timed_out'] = True
                        process.kill()
                        record['exit_code'] = process.wait(timeout=30)
                except BaseException:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=30)
                    raise
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        if process is not None:
            record['exit_code'] = process.poll()
        save(destination / 'process.json', record)
    require(record['exit_code'] == 0 and not record['timed_out'], f'Child failed; preserve {destination}')
    validate_xml(destination / 'Tests.xml')
    return record


def validate_xml(path):
    tests = ET.parse(path).getroot()
    require(tests.get('tests') == '1' and tests.get('failures') == '0' and tests.get('errors', '0') == '0' and
            not tests.findall('.//skipped'), f'RHITests failed, skipped, or incomplete: {path}')
    cases = tests.findall('.//testcase')
    require(len(cases) == 1 and cases[0].get('classname') == 'RHIRendering' and
            cases[0].get('name') == 'material_phase0_baseline', 'Wrong registered test was executed')


def fnv1a(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'{value:016x}'


def specialization(stage, version):
    values = stage.get('specialization_constants', [])
    require(version == 1 or 'specialization_constants' in stage, 'Missing native specialization vector')
    require(isinstance(values, list), 'Specialization vector must be an array')
    ids = []
    for value in values:
        require(set(value) == {'id', 'type', 'size', 'bits'} and value['type'] in ('bool32', 'int32') and
                value['size'] == 4 and type(value['id']) is int and type(value['bits']) is int and
                0 <= value['id'] < 2**32 and 0 <= value['bits'] < 2**32, 'Invalid specialization record')
        require(value['type'] != 'bool32' or value['bits'] in (0, 1), 'Noncanonical Bool32 specialization')
        ids.append(value['id'])
    require(ids == sorted(set(ids)), 'Specialization IDs must be sorted and unique')
    return values


def recipe_evidence(directory, arm, validation, specification, graph):
    entries, targets, unrelated, devices = [], [], [], []
    for path in sorted(directory.glob('*.recipe.json')):
        recipe = read(path)
        require(recipe.get('schema') == 'metallic_pipeline_recipe' and recipe.get('version') in (1, 2, 3),
                'Unsupported runtime recipe schema')
        require(recipe['pipeline_type'] in ('compute', 'graphics', 'ray_tracing'), 'Unsupported pipeline type')
        require(recipe['pipeline_type'] != 'ray_tracing' or recipe['version'] == 3, 'RT recipes require version 3')
        device = recipe['source_device']
        require(device['validation_enabled'] is validation, 'Recipe validation state does not match run')
        require(device['synchronization_validation_enabled'] is validation, 'Recipe synchronization-validation state does not match run')
        if device not in devices:
            devices.append(device)
        shaders, payloads = {}, {}
        for name, stage in recipe['shaders'].items():
            if stage['spirv_file'] is None:
                continue
            filename = stage['spirv_file']
            require(Path(filename).name == filename, 'Recipe payload must be a sibling file')
            payload = directory / filename
            raw = payload.read_bytes()
            require(len(raw) == stage['spirv_bytes'] and len(raw) >= 20 and len(raw) % 4 == 0 and
                    raw[:4] == b'\x03\x02\x23\x07', 'Invalid captured SPIR-V payload')
            require(fnv1a(raw) == stage['input_spirv_hash'].lower(), 'Captured SPIR-V FNV differs from recipe')
            require(stage['slang_optimization'] == 'O1' and stage['debug_mode'] == 'release' and
                    stage['source_identity_ambiguous'] is False and stage['spirv_postprocess'] == 'none',
                    'Runtime shader policy changed or provenance is ambiguous')
            normalized = copy.deepcopy(stage)
            normalized.pop('spirv_file')
            normalized['specialization_constants'] = specialization(stage, recipe['version'])
            normalized['sha256'] = hashlib.sha256(raw).hexdigest()
            shaders[name] = normalized
            payloads[name] = dict(file=str(payload), bytes=len(raw), sha256=normalized['sha256'])
        identity = dict(pipeline_type=recipe['pipeline_type'], state=recipe['state'], shaders=shaders,
                        expected_pso_hash=recipe['expected_pso_hash'], cache_group=recipe['cache_group'])
        modules = {s['module_name'] for s in shaders.values()}
        target = bool(modules.intersection(QUERY_MODULES + RT_MODULES))
        record = dict(manifest=str(path), manifest_sha256=digest(path), payloads=payloads, identity=identity)
        entries.append(record)
        (targets if target else unrelated).append(identity)
    require(len(devices) == 1 and len(targets) == 1 and entries, 'One actual target PSO and one device are required per process')
    target = targets[0]
    shaders, state = target['shaders'], target['state']
    require(state['disable_optimization'] is False and state['uses_bindless_heap'] is True, 'Target PSO policy changed')
    if arm == 'A':
        require(target['pipeline_type'] == 'compute' and set(shaders) == {'compute'} and state['compute_backend'] == 0,
                'A did not create the production compute/RQ PSO')
        main = shaders['compute']
        require(main['module_name'] in QUERY_MODULES, 'A target source changed')
    else:
        require(target['pipeline_type'] == 'ray_tracing' and set(shaders) == set(RT_STAGES) and
                state['max_recursion_depth'] == 1 and state['cluster_acceleration_structure'] is True,
                'B did not create the complete CLAS ray-tracing PSO')
        require(devices[0]['enabled_features']['ray_tracing_pipeline'] is True, 'RT pipeline feature was not enabled')
        main = shaders['ray_generation']
        require(main['module_name'] in RT_MODULES, 'B target source changed')
        for name, entry in (('miss', 'openPbrRayPipelineMiss'), ('closest_hit', 'openPbrRayPipelineClosestHit'),
                            ('any_hit', 'openPbrRayPipelineAnyHit')):
            require(shaders[name]['module_name'] == 'Features/PathTracing/OpenPBRRayPipelineHit' and
                    shaders[name]['source_entry_point'] == entry and not shaders[name]['specialization_constants'],
                    'Hit-stage source or native policy changed')
    props = path_node(graph)['properties']
    single = props.get('samples', 1) == 1 and props.get('exportDenoiserGuides', False)
    expected = [dict(id=0, type='bool32', size=4, bits=0)]
    if not single:
        expected.append(dict(id=1, type='bool32', size=4, bits=int(props.get('exportDenoiserGuides', False))))
    require(main['specialization_constants'] == expected, f'Native cache/guide tuple changed for {specification["id"]}')
    require(main['module_name'].endswith('Guides') is bool(single), 'Canonical Single/General shader family changed')
    expected_entry = ('openPbrRayQuery' if arm == 'A' else 'openPbrRayPipeline') + ('PathTraceGuidesMain' if single else 'PathTraceMain')
    require(main['source_entry_point'] == expected_entry and main['entry_point'] == 'main', 'Target entry point changed')
    return dict(records=entries, device=devices[0], target=target,
                unrelated=sorted(unrelated, key=canonical), stable=sorted((e['identity'] for e in entries), key=canonical))


def inspect_run(destination, configuration, arm, validation, prepared):
    process = read(destination / 'process.json')
    require(process['exit_code'] == 0 and process['timed_out'] is False, 'Failed/timeout process is not evidence')
    validate_xml(destination / 'Tests.xml')
    report_path = destination / 'MaterialBaseline.json'
    report = read(report_path)
    require(report['validation'] is validation and len(report['cases']) == 1, 'Wrong validation state or case set')
    require(report.get('synchronizationValidation', False) is validation, 'Runtime synchronization-validation configuration changed')
    spec, case = configuration['cases'][0], report['cases'][0]
    require(case['id'] == spec['id'], 'Runtime case identity changed')
    experiment.validate_requested_case(case, spec, configuration)
    require(case.get('validHDR') is True and case.get('nonfiniteComponents') == 0, 'Nonfinite/invalid HDR is excluded')
    graph = read(prepared / spec['graph'])
    require(normalized_graph(case['resolvedGraph'], arm) ==
            normalized_graph(expected_resolved_graph(graph, spec), arm), 'Runtime graph changed from prepared graph')
    if spec.get('requireEnvironment', True):
        require(case.get('environmentTransitions') and all(t['mapAvailable'] for t in case['environmentTransitions']),
                'HDR environment was not ready during the workload')
    require(Path(case['image']).name == case['image'], 'Final HDR must be a sibling readback')
    experiment.readback(report_path, case)
    for capture in case.get('captures', []):
        require(Path(capture['file']).name == capture['file'], 'Guide/temporal capture must be a sibling readback')
        experiment.readback(report_path, case, capture['file'], capture['bytes'], capture['format'])
    log = (destination / 'stdout.log').read_text(errors='replace') + (destination / 'stderr.log').read_text(errors='replace')
    for marker in ('initial material compilation failed', 'displaying the error material', 'validation error:', 'validation error [',
                   'validation error (', 'validation error count: 1', 'vuid-'):
        require(marker not in log.lower(), f'Runtime diagnostic failure marker: {marker}')
    recipes = recipe_evidence(destination / 'recipes', arm, validation, spec, graph)
    timings = experiment.gpu_timings(report, configuration)
    return dict(report=str(report_path), process=process, recipes=recipes, timings=timings,
                report_sha256=digest(report_path), xml_sha256=digest(destination / 'Tests.xml'))


def compare(left, right, arm_left='A', arm_right='B'):
    import numpy as np
    lp, rp = Path(left['report']), Path(right['report'])
    a, b = read(lp)['cases'][0], read(rp)['cases'][0]
    require(normalized_graph(a['resolvedGraph'], arm_left) == normalized_graph(b['resolvedGraph'], arm_right),
            'Compared runtime graphs differ beyond traversal backend')
    for field in ('id', 'width', 'height', 'format', 'environment', 'scene', 'output'):
        require(a.get(field) == b.get(field), f'Readback workload mismatch: {field}')
    require(a['captures'] == b['captures'], 'Capture contract changed')
    pairs = [('final HDR', None)] + [(f'{c["output"]}@{c["frame"]}', c) for c in a['captures']]
    comparisons = []
    for name, capture in pairs:
        args = (capture['file'], capture['bytes'], capture['format']) if capture else ()
        av, bv = experiment.readback(lp, a, *args), experiment.readback(rp, b, *args)
        require(av.shape == bv.shape, 'Readback shape mismatch')
        error = np.abs(av - bv)
        guide = capture is not None and capture['output'] != a['output']
        filename = capture['file'] if capture else a['image']
        exact = (lp.parent / filename).read_bytes() == (rp.parent / filename).read_bytes()
        passed = exact if guide else bool(np.all(error <= ABSOLUTE + RELATIVE * np.abs(av)))
        comparisons.append(dict(output=name, guide=guide, byte_exact=exact, qualified=passed,
            max_absolute_error=float(error.max(initial=0)), rmse=float(np.sqrt(np.mean(error * error))),
            differing_components=int(np.count_nonzero(error)), failed_components=int(np.count_nonzero(
                error > (0 if guide else ABSOLUTE + RELATIVE * np.abs(av))))))
    require(left['recipes']['device'] == right['recipes']['device'], 'Device/driver/validation identity changed')
    require(left['recipes']['unrelated'] == right['recipes']['unrelated'], 'A/B changed non-target pipeline state/SPIR-V')
    if arm_left != arm_right:
        a_main = left['recipes']['target']['shaders']['compute']
        b_main = right['recipes']['target']['shaders']['ray_generation']
        for field in ('specialization_constants', 'debug_mode', 'descriptor_heap_mode', 'slang_optimization',
                      'spirv_postprocess', 'spirv_postprocess_tool_version'):
            require(a_main[field] == b_main[field], f'A/B target policy changed: {field}')
        for field in ('uses_bindless_heap', 'bindless_user_push_data_size', 'disable_optimization'):
            require(left['recipes']['target']['state'][field] == right['recipes']['target']['state'][field],
                    f'A/B target resource/native policy changed: {field}')
    return dict(qualified=all(c['qualified'] for c in comparisons), readbacks=comparisons,
                non_target_pipeline_identity_equal=True)


def lighting_controls(runs):
    import numpy as np
    results = []
    for arm in ARMS:
        by_case = {r['case']: r for r in runs if r['arm'] == arm and r['block'] == 0}
        for label, lit, dark in (('local', 'MixedNoEnvironment', 'MixedNoEnvironmentNoDirect'),
                                 ('environment', 'MixedEnvironmentOnly', 'MixedEnvironmentOnlyNoDirect')):
            if lit not in by_case or dark not in by_case:
                continue
            lp, rp = (Path(by_case[name]['report']) for name in (lit, dark))
            aa, bb = read(lp)['cases'][0], read(rp)['cases'][0]
            a, b = experiment.readback(lp, aa), experiment.readback(rp, bb)
            delta = np.abs(a - b)
            results.append(dict(arm=arm, control=label, maximum_difference=float(delta.max(initial=0)),
                changed_components=int(np.count_nonzero(delta)), qualified=bool(delta.max(initial=0) > 1e-6)))
    return results


def schedule(stage, cases):
    for case in cases:
        for block, order in enumerate(ORDER if stage == 'performance' else ORDER[:1]):
            for arm in order:
                yield case['id'], block, arm


def require_stable_recipes(previous, current):
    require(current['recipes']['stable'] == previous['recipes']['stable'],
            'Same-arm actual PSO/SPIR-V identity changed')


def analyze(output, prepared, stage, selected_cases, recorded=None):
    configs = {arm: read(prepared / f'{stage}-{arm}.json') for arm in ARMS}
    for config in configs.values():
        config['cases'] = [c for c in config['cases'] if c['id'] in selected_cases]
    validation = stage == 'correctness'
    runs, pairs, same_arm = [], [], []
    stable = {}
    for name, block, arm in schedule(stage, configs['A']['cases']):
        destination = output / name / f'block-{block:02}-{arm}'
        config = read(destination / 'Cases.json')
        expected = copy.deepcopy(configs[arm])
        expected['cases'] = [c for c in expected['cases'] if c['id'] == name]
        # A one-case process uses absolute graph paths so its output directory
        # cannot affect asset resolution. All graph bytes remain sealed inputs.
        expected['cases'][0]['graph'] = str((prepared / expected['cases'][0]['graph']).resolve())
        require(config == expected, 'Child Cases.json differs from prepared workload')
        run = dict(case=name, block=block, arm=arm, **inspect_run(destination, config, arm, validation, prepared))
        key = (name, arm)
        if key in stable:
            previous = stable[key]
            require_stable_recipes(previous, run)
            same_arm.append(dict(case=name, arm=arm, block=block, **compare(previous, run, arm, arm)))
        else:
            stable[key] = run
        runs.append(run)
    for name in [c['id'] for c in configs['A']['cases']]:
        for block in range(3 if stage == 'performance' else 1):
            a = next(r for r in runs if r['case'] == name and r['block'] == block and r['arm'] == 'A')
            b = next(r for r in runs if r['case'] == name and r['block'] == block and r['arm'] == 'B')
            comparison = compare(a, b)
            metrics = {}
            if stage == 'performance':
                for metric in ('graph_ms', 'pass_ms', 'dispatch_scope_ms'):
                    av, bv = a['timings'][name][metric]['median'], b['timings'][name][metric]['median']
                    metrics[metric] = dict(A=av, B=bv, percent=100.0 * (bv / av - 1.0))
            pairs.append(dict(case=name, block=block, **comparison, metrics=metrics))
    controls = lighting_controls(runs) if stage == 'correctness' else []
    qualified = all(c['qualified'] for c in pairs + same_arm + controls)
    result = dict(runs=runs, comparisons=pairs, same_arm_readbacks=same_arm, lighting_controls=controls,
                  correctness_qualified=qualified)
    if recorded is not None:
        require(result == recorded, 'Recomputed evidence differs from saved analysis')
    return result


def frozen_inputs(executable, prepared):
    # Use the maintained full-renderer identity, including shared runtime/RHI
    # sources and every shader; no measured run may continue across a source edit.
    identity = baseline.identity(executable, prepared / 'correctness-A.json')
    paths = {Path(p) for p in identity['binaries']} | {ROOT / p for p in identity['sources']}
    paths.update(Path(p) for p in identity['workload'])
    paths.update(p for p in prepared.rglob('*') if p.is_file())
    paths.update((Path(__file__).resolve(), Path(experiment.__file__).resolve(), Path(driver.__file__).resolve()))
    return dict(metadata=identity, snapshots=experiment.snapshots(paths))


def run(args):
    prepared, executable = args.prepared.resolve(strict=True), args.executable.resolve(strict=True)
    verify_prepared(prepared)
    output = args.output_dir.resolve()
    require(not output.exists(), 'Run requires a fresh output directory; never overwrite prior evidence')
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 7200, 'Timeout must be bounded to (0,7200] seconds')
    require('CMAKE_BUILD_TYPE:STRING=Release' in (executable.parent.parent / 'CMakeCache.txt').read_text(),
            'Use the configured Release RHITests executable')
    configs = {arm: read(prepared / f'{args.stage}-{arm}.json') for arm in ARMS}
    all_cases = [c['id'] for c in configs['A']['cases']]
    selected = args.case or all_cases
    require(len(selected) == len(set(selected)) and set(selected) <= set(all_cases), 'Unknown or repeated --case selection')
    for config in configs.values():
        config['cases'] = [c for c in config['cases'] if c['id'] in selected]
    selected = [c['id'] for c in configs['A']['cases']]
    environment, removed = clean_environment()
    with experiment_lock(output):
        output.mkdir(parents=True)
        inputs = frozen_inputs(executable, prepared)
        save(output / 'Inputs.json', inputs)
        shutil.copy2(__file__, output / 'PathTraceTraversal.py')
        save(output / 'Environment.json', dict(removed_environment_keys=removed,
            fixed={k: v for k, v in environment.items() if k.startswith('METALLIC_')}))
        summary = dict(schema_version=1, status='running', stage=args.stage, prepared=str(prepared),
            prepared_manifest_sha256=digest(prepared / 'Manifest.json'), arms=ARMS,
            validation=args.stage == 'correctness', schedule=list(schedule(args.stage, configs['A']['cases'])),
            synchronization_validation=args.stage == 'correctness', selected_cases=selected,
            full_stage_coverage=selected == all_cases,
            absolute_tolerance=ABSOLUTE, relative_tolerance=RELATIVE, guide_comparison='raw-byte-exact',
            performance_order=[list(v) for v in ORDER], process_records=[],
            scope='Independent-process GPU graph/pass/shading-scope medians; native creation and readbacks are outside GPU timing.',
            cache_policy='Separate fresh application PSO directories per arm; existing compiler/OS/driver caches allowed warm; no clock changes.',
            statistical_unit='One independent-process median. Pair percent = 100 * (B / A - 1); positive means slower.',
            limitations=['Cooperative locks do not prove system-wide GPU exclusivity.',
                         'Strict fixed-workload HDR/guide equality is not proof of correctness for every scene.',
                         'Correctness runs use validation and are not performance measurements.'])
        save(output / 'Summary.json', summary)
        try:
            for name, block, arm in schedule(args.stage, configs['A']['cases']):
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
                process = child(command, env, destination, args.timeout)
                summary['process_records'].append(dict(case=name, block=block, arm=arm, process=process))
                save(output / 'Summary.json', summary)
                # Fail closed on process/identity/finite-data errors immediately.
                # Numeric differences are retained and summarized after all pairs.
                inspect_run(destination, config, arm, args.stage == 'correctness', prepared)
                experiment.check_snapshots(inputs['snapshots'])
            summary['analysis'] = analyze(output, prepared, args.stage, selected)
            require(summary['analysis']['correctness_qualified'], 'Strict HDR/guide/lighting gate failed; raw data retained; thresholds unchanged')
            summary['status'] = 'passed'
        except BaseException as error:
            summary['status'] = 'failed'
            summary['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            save(output / 'Summary.json', summary)
            seal(output)
    print(json.dumps(verify(output), indent=2))


def verify(output, allow_numeric_failure=False):
    output = Path(output).resolve(strict=True)
    verify_seal(output)
    summary = read(output / 'Summary.json')
    numeric_failure = (summary.get('status') == 'failed' and summary.get('error') ==
        'ValueError: Strict HDR/guide/lighting gate failed; raw data retained; thresholds unchanged' and
        isinstance(summary.get('analysis'), dict) and summary['analysis'].get('correctness_qualified') is False)
    require(summary['status'] == 'passed' or (allow_numeric_failure and numeric_failure),
            'Failed/incomplete evidence cannot pass verification')
    prepared = Path(summary['prepared'])
    verify_prepared(prepared)
    require(digest(prepared / 'Manifest.json') == summary['prepared_manifest_sha256'], 'Prepared manifest changed')
    require(summary['absolute_tolerance'] == ABSOLUTE and summary['relative_tolerance'] == RELATIVE and
            summary['guide_comparison'] == 'raw-byte-exact', 'Fixed comparison contract changed')
    configs = read(prepared / f'{summary["stage"]}-A.json')
    all_cases = [c['id'] for c in configs['cases']]
    require(set(summary['selected_cases']) <= set(all_cases) and summary['selected_cases'], 'Invalid selected case set')
    configs['cases'] = [c for c in configs['cases'] if c['id'] in summary['selected_cases']]
    require(summary['selected_cases'] == [c['id'] for c in configs['cases']] and
            summary['full_stage_coverage'] == (summary['selected_cases'] == all_cases), 'Selected coverage changed')
    require(summary['schedule'] == [list(v) for v in schedule(summary['stage'], configs['cases'])], 'Recorded schedule changed')
    require([(r['case'], r['block'], r['arm']) for r in summary['process_records']] ==
            list(schedule(summary['stage'], configs['cases'])), 'Process schedule incomplete or reordered')
    result = analyze(output, prepared, summary['stage'], summary['selected_cases'], summary['analysis'])
    require(result['correctness_qualified'] is (not numeric_failure), 'Correctness qualification contradicts recorded status')
    return dict(verified=True, stage=summary['stage'], summary_sha256=digest(output / 'Summary.json'),
                manifest_sha256=digest(output / 'Manifest.json'), processes=len(result['runs']),
                cases=summary['selected_cases'], full_stage_coverage=summary['full_stage_coverage'],
                correctness_qualified=not numeric_failure, numeric_failure=numeric_failure,
                identity='Actual complete compute/RT recipes and every SPIR-V payload rehashed; same-arm stable; non-target A/B equal.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Copy and seal graph JSON; no GPU execution')
    prep.add_argument('--fixtures', type=Path, default=DEFAULT_INPUT)
    prep.add_argument('--output-dir', type=Path, default=DEFAULT_TASK / 'prepared-v2')
    capture = commands.add_parser('run', help='Execute bounded independent GPU processes')
    capture.add_argument('--stage', choices=('correctness', 'performance'), required=True)
    capture.add_argument('--prepared', type=Path, default=DEFAULT_TASK / 'prepared-v2')
    capture.add_argument('--executable', type=Path, default=ROOT / 'build-release/tests/MetallicRHITests.exe')
    capture.add_argument('--output-dir', type=Path, required=True)
    capture.add_argument('--timeout', type=float, default=1800)
    capture.add_argument('--case', action='append', help='Optional exact case filter; repeat for a bounded validation subset')
    check = commands.add_parser('verify', help='Recompute integrity, identity and readback gates; no GPU execution')
    check.add_argument('output_dir', type=Path)
    check.add_argument('--allow-numeric-failure', action='store_true',
        help='Verify sealed numeric-gate failures for diagnosis only; correctness remains unqualified')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args)
        elif args.command == 'run':
            run(args)
        else:
            print(json.dumps(verify(args.output_dir, allow_numeric_failure=args.allow_numeric_failure), indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
