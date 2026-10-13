"""Compare concrete and Slang link-time material implementations on real Guides.

Only frozen shader overlays are changed. No-Fiber is an explicit scene
restriction, not a claim of general equivalence or a production default.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import ProfileComputeShaderFrontend as profile
import RunShaderPipelineDriver as driver
import ValidateSlangLinkTimeSpecialization as legacy
import PathTraceBackendOverlay as overlay

ROOT = Path(__file__).resolve().parents[2]
ARMS = (
    ('original-general', 'original', 'general'),
    ('backend-concrete-general', 'concrete-general', 'general'),
    ('backend-link-general', 'link-general', 'general'),
    ('backend-concrete-no-fiber', 'concrete-no-fiber', 'no-fiber'),
    ('backend-link-no-fiber', 'link-no-fiber', 'no-fiber'),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def reference_sample(report, request_index):
    sample = driver.single_sample(report, native=False)
    cfg = report['configuration']
    if not cfg['compile_only'] or cfg['shader_cache'] != 'reuse' or sample['shader_cache_hit'] or \
        not sample['compilation_phases']['compiler_executed']:
        raise ValueError('Need one fresh compile-only shader-cache miss')
    if sample['pipeline_status'] != 'compile_only':
        raise ValueError('Reference unexpectedly performed native creation')
    if sample['request']['index'] != request_index or \
        sample['request']['module'] != 'Features/PathTracing/OpenPBRRayQueryPathTraceGuides' or \
        sample['request']['entry'] != 'openPbrRayQueryPathTraceGuidesMain':
        raise ValueError('Need the fixed real Guides request')
    if cfg['slang_optimization'] != 'O1' or cfg['descriptor_mode'] != 'mapped' or \
        cfg.get('spirv_postprocess', 'none') != 'none' or \
        sample['effective_slang_optimization'] != 'O1' or sample['effective_descriptor_mode'] != 'mapped' or \
        sample.get('effective_spirv_postprocess', 'none') != 'none' or \
        cfg['debug_mode'] != 'disabled' or cfg['validation']:
        raise ValueError('Need unchanged mapped O1 with debug/validation disabled')
    definitions = dict(sample['request']['defines'])
    if definitions.get('METALLIC_CUSTOM_MATERIALS') != '0' or \
        any(definitions.get(name, '0') != '0' for name in ('SHARC_UPDATE', 'SHARC_QUERY')):
        raise ValueError('First backend experiment requires CUSTOM_MATERIALS=0 and SHARC off')
    return sample


def stage_commands(plan, directory):
    compiler, source, flags, module_name, entry = legacy.command_parts(plan)
    stages = []
    if plan['mechanism'] == 'original':
        direct = list(plan['direct_command'])
        direct[direct.index('-o') + 1] = str(directory / 'Direct.spv')
        stages.append(('source-direct', direct))
    consumer = directory / 'Consumer.slang-module'
    provider = directory / 'Provider.slang-module'
    stages.append(('consumer-precompile', [compiler, '-target', 'spirv', *flags,
        '-no-codegen', '-emit-ir', source, '-module-name', module_name, '-o', str(consumer)]))
    if plan['provider']:
        stages.append(('provider-precompile', [compiler, '-target', 'spirv', *flags,
            '-no-codegen', '-emit-ir', plan['provider'], '-o', str(provider)]))
    providers = [str(provider)] if plan['provider'] else []
    stages.append(('link', [compiler, '-target', 'spirv', *flags, *providers, str(consumer),
        '-emit-spirv-directly', '-entry', entry, '-o', str(directory / 'Guides.spv')]))
    return stages


def invoke(command, directory, name, timeout):
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    with (directory / (name + '.stdout.log')).open('wb') as stdout, \
        (directory / (name + '.stderr.log')).open('wb') as stderr:
        with driver.suppress_windows_error_dialogs():
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait()
            timed_out = True
    record = dict(command=command, pid=process.pid, started_utc=started_utc,
        exit_code=code, timed_out=timed_out, native_crash=driver.native_process_crashed(code),
        wall_seconds=time.perf_counter() - started,
        stdout=name + '.stdout.log', stderr=name + '.stderr.log',
        status='passed' if code == 0 and not timed_out else 'failed')
    save(directory / (name + '.process.json'), record)
    return record


def check_inputs(preparation_path):
    preparation = legacy.verify_preparation(preparation_path)
    validate_plans(Path(preparation_path).resolve(), preparation)
    for path, expected in preparation['tool_inputs'].items():
        if profile.sha(path) != expected:
            raise ValueError('Experiment helper changed after preparation: ' + path)
    return preparation


def validate_plans(path, preparation):
    if preparation.get('experiment') != 'material-backend-type-v1' or \
        [plan['label'] for plan in preparation['plans']] != [label for label, _, _ in ARMS]:
        raise ValueError('Incomplete or reordered backend preparation')
    report = read(path / 'ReferenceReport.json')
    sample = reference_sample(report, report['configuration']['selected_requests'][0])
    reference = (path / 'Reference.spv').read_bytes()
    if hashlib.sha256(reference).hexdigest() != preparation['reference_sha256'] or \
        profile.fnv(reference) != preparation['reference_fnv1a64'] or \
        sample['spirv_bytes'] != len(reference) or sample['spirv_fnv1a64'] != profile.fnv(reference):
        raise ValueError('Reference report or bytecode identity changed')
    roots = [ROOT / 'Shaders', *[Path(name).resolve() for name in sample['request']['additional_search_paths']]]
    base = profile.diagnostic_command(preparation['compiler'], report, sample, path / 'unused.spv')
    for plan, (label, arm, feature) in zip(preparation['plans'], ARMS):
        directory = path / 'overlays' / label
        metadata = plan['overlay_metadata']
        expected_mechanism = 'original' if arm == 'original' else \
            ('backend-link' if arm.startswith('link-') else 'backend-concrete')
        if plan['backend_arm'] != arm or plan['mechanism'] != expected_mechanism or \
            plan['feature_mask'] != legacy.feature_values(feature) or \
            metadata['feature_mask'] != plan['feature_mask'] or metadata['extra_compiler_flags']:
            raise ValueError('Backend mechanism or scene restriction changed')
        expected_provider = str(directory / 'Shaders' / metadata['provider_file']) if arm.startswith('link-') else None
        if plan['provider'] != expected_provider or bool(metadata['provider_file']) != arm.startswith('link-'):
            raise ValueError('Wrong or missing backend provider')
        if plan['direct_command'] != legacy.rewrite_command_inputs(base, directory, roots):
            raise ValueError('Backend compiler settings differ from the fixed reference')
        for relative, digest in metadata['generated_hashes'].items():
            if profile.sha(directory / 'Shaders' / relative) != digest:
                raise ValueError('Generated backend shader changed')


def prepare(args):
    output = legacy.fresh_directory(args.output_dir)
    report = read(args.report)
    sample = reference_sample(report, args.request_index)
    if Path(sample['request']['search_path']).resolve() != ROOT / 'Shaders':
        raise ValueError('Reference must use the current repository shader root')
    cache, raw = profile.bench_bytes(report, sample)
    compiler = (args.slang_root / 'bin/slangc.exe').resolve(strict=True)
    shader_root = ROOT / 'Shaders'
    roots = [shader_root, *[Path(path).resolve(strict=True)
        for path in sample['request']['additional_search_paths']]]
    if any(not path.is_relative_to(ROOT) or output.is_relative_to(path) for path in roots):
        raise ValueError('Source roots must be repository paths outside the output')
    paths = {path.resolve() for parent in roots for path in parent.rglob('*') if path.is_file()}
    paths.update(Path(path).resolve(strict=True) for path in sample['dependencies'])
    paths.update(ROOT / path for path in (
        'Source/Runtime/Render/Core/NamedResourceParameters.h',
        'Source/Runtime/Render/Core/ColorSpaceMatrices.h',
        'Source/Runtime/Render/Environment/AtmosphereKernelParameters.h',
        'Source/Runtime/Render/Material/MaterialValueProgram.cpp'))
    if any(not path.is_relative_to(ROOT) for path in paths):
        raise ValueError('Unfrozen dependency outside the repository')
    sources = {str(path.relative_to(ROOT)): profile.sha(path) for path in sorted(paths)}
    tool_paths = [Path(__file__).resolve(), Path(overlay.__file__).resolve(),
        Path(legacy.__file__).resolve(), Path(profile.__file__).resolve(), Path(driver.__file__).resolve(),
        ROOT / 'Tools/CheckShaderModules.py']
    tools = {str(path): profile.sha(path) for path in tool_paths}
    copies = output / 'HarnessSources'
    copies.mkdir()
    for path in tool_paths:
        shutil.copy2(path, copies / path.name)
    save(output / 'ReferenceReport.json', report)
    (output / 'Reference.spv').write_bytes(raw)
    base = profile.diagnostic_command(compiler, report, sample, output / 'unused.spv')
    plans = []
    for label, arm, feature in ARMS:
        destination = output / 'overlays' / label
        for relative, expected in sources.items():
            source = ROOT / relative
            if profile.sha(source) != expected:
                raise ValueError('Source changed while freezing: ' + relative)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        modifications = overlay.plan_overlay(destination / 'Shaders', arm)
        for relative, text in modifications['files'].items():
            target = (destination / 'Shaders' / relative).resolve()
            if not target.is_relative_to(destination / 'Shaders'):
                raise ValueError('Overlay modification escaped shader root')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode('utf-8'))
        audit = invoke([sys.executable, '-B', str(ROOT / 'Tools/CheckShaderModules.py'),
            '--root', str(destination)], output, label + '-ownership', 30)
        if audit['status'] != 'passed':
            raise ValueError('Generated shader module ownership failed: ' + label)
        provider_file = modifications.get('provider_file')
        provider = str(destination / 'Shaders' / provider_file) if provider_file else None
        command = legacy.rewrite_command_inputs(base, destination, roots)
        plans.append(dict(label=label, mechanism='original' if arm == 'original' else
            ('backend-link' if provider else 'backend-concrete'), feature_mask=legacy.feature_values(feature),
            provider=provider, direct_command=command, backend_arm=arm,
            overlay_metadata={name: value for name, value in modifications.items() if name != 'files'},
            ownership_audit=audit))
    for relative, expected in sources.items():
        if profile.sha(ROOT / relative) != expected:
            raise ValueError('Source changed during preparation')
    preparation = dict(schema_version=1, status='prepared', experiment='material-backend-type-v1',
        report=str(args.report.resolve()), report_sha256=profile.sha(args.report),
        shader_cache=str(cache), reference_fnv1a64=profile.fnv(raw),
        reference_sha256=hashlib.sha256(raw).hexdigest(), source_files=sources,
        compiler=str(compiler), compiler_sha256=profile.sha(compiler),
        sdk_files=legacy.compiler_manifest(compiler), tool_inputs=tools, plans=plans,
        scope='Frozen mapped O1 Guides. Concrete versus linked complete material dispatch/guide implementations. No GPU or Driver work. No-Fiber is a scene restriction.')
    save(output / 'Preparation.json', preparation)
    profile.seal(output)
    check_inputs(output)
    print(output / 'Preparation.json')


def run(args):
    preparation_path = args.preparation.resolve()
    preparation = check_inputs(preparation_path)
    output = legacy.fresh_directory(args.output_dir)
    variants = [label for label, _, _ in ARMS]
    summary = dict(schema_version=1, status='running', experiment=preparation['experiment'],
        preparation=str(preparation_path), preparation_manifest_sha256=profile.sha(preparation_path / 'Manifest.json'),
        repetitions=args.repetitions, variants=variants, mode='source-and-link', runs=[],
        scope=preparation['scope'] + ' Initial Consumer/Provider source-to-IR costs and IR-to-target linking are separate; imported modules may parse source again.')
    save(output / 'summary.json', summary)
    with profile.experiment_lock():
        for block in range(args.repetitions):
            plans = preparation['plans'] if block % 2 == 0 else list(reversed(preparation['plans']))
            for plan in plans:
                directory = output / f"block-{block:02}-{plan['label']}"
                directory.mkdir()
                stages = []
                for name, command in stage_commands(plan, directory):
                    check_inputs(preparation_path)
                    stage = invoke(command, directory, name, args.timeout)
                    stages.append(stage)
                    check_inputs(preparation_path)
                    if stage['status'] != 'passed':
                        break
                record = dict(label=plan['label'], mechanism=plan['mechanism'], feature_mask=plan['feature_mask'],
                    repetition=block, directory=directory.name, stages=stages,
                    status='passed' if len(stages) == len(stage_commands(plan, directory)) and
                        all(stage['status'] == 'passed' for stage in stages) else 'failed')
                spirv = directory / 'Guides.spv'
                if record['status'] == 'passed':
                    raw = spirv.read_bytes()
                    legacy.instruction_counts(raw)
                    record.update(spirv_bytes=len(raw), spirv_fnv1a64=profile.fnv(raw),
                        spirv_sha256=profile.sha(spirv), matches_current_bench_byte_for_byte=
                            raw == (preparation_path / 'Reference.spv').read_bytes())
                    if plan['mechanism'] == 'original' and (not record['matches_current_bench_byte_for_byte'] or
                        (directory / 'Direct.spv').read_bytes() != raw):
                        record['status'] = 'baseline_bytecode_mismatch'
                summary['runs'].append(record)
                save(output / 'summary.json', summary)
                print(record['label'], record['status'], record.get('spirv_bytes'), flush=True)
                if record['status'] != 'passed':
                    summary['status'] = 'failed'
                    save(output / 'summary.json', summary)
                    profile.seal(output)
                    return 1
    summary['status'] = 'passed'
    save(output / 'summary.json', summary)
    profile.seal(output)
    verify(output)
    return 0


def verify(path):
    summary = legacy.verify_run(path)
    path = Path(path).resolve()
    preparation = read(Path(summary['preparation']) / 'Preparation.json')
    validate_plans(Path(summary['preparation']).resolve(), preparation)
    if summary.get('experiment') != 'material-backend-type-v1' or \
        summary['variants'] != [label for label, _, _ in ARMS]:
        raise ValueError('Wrong backend experiment or arm set')
    plans = {plan['label']: plan for plan in preparation['plans']}
    processes = set()
    for record in summary['runs']:
        directory = path / record['directory']
        commands = stage_commands(plans[record['label']], directory)
        if len(record['stages']) != len(commands):
            raise ValueError('Missing or extra backend compiler stage')
        for stage, (name, command) in zip(record['stages'], commands):
            if stage['command'] != command or stage != read(directory / (name + '.process.json')) or \
                stage['stdout'] != name + '.stdout.log' or stage['stderr'] != name + '.stderr.log':
                raise ValueError('Compiler stage did not execute its sealed backend plan')
            duration = stage['wall_seconds']
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or \
                not math.isfinite(duration) or duration <= 0 or stage['timed_out'] or stage['native_crash']:
                raise ValueError('Invalid backend compiler duration or process')
            identity = (stage['pid'], stage['started_utc'])
            if identity in processes:
                raise ValueError('Compiler process was reused as another sample')
            processes.add(identity)
        for name in ('Consumer.slang-module', 'Provider.slang-module') if plans[record['label']]['provider'] else ('Consumer.slang-module',):
            raw = (directory / name).read_bytes()
            if len(raw) < 12 or raw[:4] != b'RIFF' or raw[8:12] != b'SLmc':
                raise ValueError('Compiler IR is not a serialized Slang module')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('prepare')
    preparation.add_argument('--report', type=Path, required=True)
    preparation.add_argument('--request-index', type=int, default=146)
    preparation.add_argument('--slang-root', type=Path, default=ROOT / 'External/slang')
    preparation.add_argument('--output-dir', type=Path, required=True)
    execution = sub.add_parser('run')
    execution.add_argument('--preparation', type=Path, required=True)
    execution.add_argument('--output-dir', type=Path, required=True)
    execution.add_argument('--repetitions', type=int, default=1)
    execution.add_argument('--timeout', type=float, default=900)
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--input-dir', type=Path, required=True)
    analysis.add_argument('--output-dir', type=Path, required=True)
    analysis.add_argument('--spirv-opt', type=Path, default=Path('C:/VulkanSDK/1.4.363.0/Bin/spirv-opt.exe'))
    analysis.add_argument('--spirv-val', type=Path, default=Path('C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe'))
    check = sub.add_parser('verify')
    check.add_argument('input_dir', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'run':
        if args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error('Positive repetitions and timeout are required')
        return run(args)
    elif args.command == 'analyze':
        verify(args.input_dir)
        legacy.analyze(args)
    else:
        print(verify(args.input_dir)['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
