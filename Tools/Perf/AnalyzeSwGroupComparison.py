"""Audit Full wave32 group-size experiments; ordinary timing is separate from diagnostics."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Tools'))
from AnalyzeZorahFullRasterComparison import analyze, load


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def review(directory, pilot):
    manifest = load(directory / 'Manifest.json')
    assert manifest['runs'] == 3
    runs = []
    fingerprints = None
    for index in range(1, 4):
        folder = directory / f'run{index}'
        capture = load(folder / 'Capture.json')
        assert capture['outputExtent'] == [1797, 660] and capture['renderExtent'] == [1198, 440]
        assert len(capture['cases']) == 12 and all(case['frames'] == 128 for case in capture['cases'])
        assert capture['measurementKind'] == 'normal-timing'
        for key in ('validationRequested', 'pipelineStatisticsRequested', 'graphicsCaptureInjected',
                    'gpuTraceInjected', 'renderDocInjected', 'nvPerfRequested', 'shaderTraceRequested',
                    'workControlReplayRequested'):
            assert not capture.get(key, False), key
        assert not (folder / 'Teardown.txt').exists(), 'Process required forced teardown'
        assert '[Benchmark shutdown] complete' in (folder / 'stdout.log').read_text(encoding='utf-8')
        summary = analyze(folder)
        assert summary['groupOutputsByteEqual']
        if fingerprints is None:
            fingerprints = summary['pipelineFingerprints']
        assert fingerprints == summary['pipelineFingerprints']
        windows = [(c['measurementBeginUnixMs'], c['measurementEndUnixMs']) for c in capture['cases']]
        background = {}
        with (folder / 'GpuProcesses.csv').open(encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                timestamp = datetime.datetime.fromisoformat(row['timestamp']).timestamp() * 1000
                if not any(start - 1000 <= timestamp <= end for start, end in windows):
                    continue
                if row['process'].lower() == 'metallicgpudrivensample':
                    continue
                key = (row['pid'], row['process'], row['engine'])
                background[key] = max(background.get(key, 0), float(row['utilization']))
        runs.append({'run': index, 'cutHash': summary['cutHash'], 'pageMappingsHash': summary['pageMappingsHash'],
                     'activeGroups': summary['activeGroups'], 'residentState': summary['residentState'],
                     'timingsMs': summary['pooled'],
                     'otherEnginePeaks': [{'pid': key[0], 'process': key[1], 'engine': key[2], 'percent': value}
                                          for key, value in sorted(background.items(), key=lambda x: -x[1])]})
    comparisons = {}
    for candidate in ('swGroup32', 'swGroup64', 'swGroup128'):
        comparisons[candidate] = {}
        for reference in ('swWorkControl', 'swGroup128'):
            if candidate == reference:
                continue
            metrics = {}
            for metric in ('software', 'graphGpu', 'rasterTotal'):
                gains = [1 - r['timingsMs'][candidate][metric]['mean'] /
                         r['timingsMs'][reference][metric]['mean'] for r in runs]
                mean = statistics.mean(gains)
                margin = 4.30265273 * statistics.stdev(gains) / math.sqrt(len(gains))
                metrics[metric] = {'independentProcessGains': gains, 'meanGain': mean,
                                   'lower95': mean - margin, 'upper95': mean + margin}
            comparisons[candidate][reference] = metrics
    pilot_capture = load(pilot / 'Capture.json')
    assert pilot_capture['status'] == 'capture_complete' and pilot_capture['pipelineStatisticsRequested']
    analyze(pilot)
    log = (pilot / 'stdout.log').read_text(encoding='utf-8')
    resources = {}
    for case in pilot_capture['cases']:
        variant = case['variant']
        binding = case['after']['productionDispatches'][0]
        assert binding['spirvFnv1a64'] == fingerprints[variant]
        mapping = re.search(r'\[PipelineStatisticsBinding\] cacheKey=(\w+) inputSpirvFnv1a64=' +
                            fingerprints[variant] + r' deviceSpirvFnv1a64=(\d+) shader=(\S+) entry=(\S+)', log)
        assert mapping and mapping[2] == fingerprints[variant]
        stats = re.findall(r'\[PipelineStatistics\] entry=' + re.escape(mapping[4]) + r' spirv=' +
                           mapping[1] + r' executable=\S+ subgroup=(\d+) (.*?)=(.*?) \(', log)
        assert stats and all(int(row[0]) == 32 for row in stats)
        resources[variant] = {row[1]: row[2] for row in stats}
        resources[variant]['subgroup'] = 32
    result = {'protocol': 'metallic-full-sw-group-review-v1', 'scope': 'frozen Full editor, graphics queue, early+late SW',
              'runs': runs, 'comparisons': comparisons, 'resourcesDiagnosticOnly': resources,
              'pipelineFingerprints': fingerprints, 'normalFrames': 3 * 3 * 4 * 128,
              'sameOutputWithinEachRun': True, 'defaultChanged': False,
              'qualification': 'Three process comparisons at one camera; no M3 promotion or live-roaming acceptance',
              'counterCaveat': 'Pilot coverage replay uses 128 diagnostic lanes. Its lane/row sums are not physical 32/64-thread kernel utilization.'}
    save(directory / 'Review.json', result)
    return result


def seal(directory):
    # The source snapshot is taken after measurement, not a claimed before-run inventory.
    # Original runner checks shader digest and executable SHA around the entire batch.
    snapshot = directory / 'SourceSnapshot'
    snapshot.mkdir(exist_ok=False)
    paths = list((ROOT / 'Shaders').rglob('*.slang')) + [
        ROOT / p for p in ('Source/Editor/EditorRasterComparison.cpp',
                          'Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp',
                          'Tools/RunZorahFullRoam.ps1', 'Tools/AnalyzeZorahFullRasterComparison.py',
                          'Tools/Perf/AnalyzeSwGroupComparison.py')]
    for source in paths:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    executable = ROOT / 'build-scheduling-release/Source/MetallicGPUDrivenSample.exe'
    assert digest(executable).lower() == load(directory / 'Manifest.json')['executableSha256'].lower()
    shutil.copyfile(executable, directory / executable.name)
    hashes = {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob('*')
              if p.is_file() and p.name != 'EvidenceHashes.json'}
    save(directory / 'EvidenceHashes.json', hashes)


def verify(directory):
    hashes = load(directory / 'EvidenceHashes.json')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and p.name != 'EvidenceHashes.json'}
    assert actual == set(hashes), 'Evidence inventory changed'
    for name, value in hashes.items():
        path = (directory / name).resolve()
        assert path.is_relative_to(directory.resolve()) and digest(path) == value, name
    for index in range(1, 4):
        folder = directory / f'run{index}'
        summary = load(folder / 'Summary.json')
        capture = load(folder / 'Capture.json')
        reference = capture['cases'][0]['before']
        assert summary['groupOutputsByteEqual'] and capture['status'] == 'capture_complete'
        for case in capture['cases']:
            for point in ('before', 'after'):
                for resource in ('VBuffer.visibility', 'VBuffer.depth'):
                    assert (folder / case[point][resource]['file']).read_bytes() == (folder / reference[resource]['file']).read_bytes()
    return {'verified': True, 'files': len(hashes), 'scope': 'archived hashes and raw image equality; no rerun'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.directory.resolve())))
    else:
        assert args.pilot
        review(args.directory.resolve(), args.pilot.resolve())
        if args.seal:
            seal(args.directory.resolve())
