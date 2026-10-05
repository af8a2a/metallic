"""Run/verify the scoped native OpenPBR GPU benchmark; never edits shaders.

GPU buffers are device-local. Timings exclude uploads/readback/compilation.
This is a BSDF + exact buffer-LUT microbenchmark, not a frame benchmark.
"""
from __future__ import annotations
import argparse
import array
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root / 'Manifest.json').read_text())
    for name, digest in manifest['artifacts'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Artifact changed: {name}')
    summary = []
    for run in sorted(root.glob('run-*')):
        if not run.is_dir():
            continue
        process = json.loads((root / (run.name + '.json')).read_text())
        if process['exit'] != 0:
            raise ValueError(f'{run}: process failed')
        rows = list(csv.DictReader((run / 'OpenPBRNativeTiming.csv').open()))
        if len(rows) != 40 or any(r['validation'] != '0' or r['dispatches'] != '8' for r in rows):
            raise ValueError(f'{run}: invalid timing contract')
        for mode in (0, 1):
            outputs = []
            for side in ('Vendor', 'Native'):
                values = array.array('f')
                values.frombytes((run / f'{side}-{mode}.bin').read_bytes())
                if sys.byteorder != 'little':
                    values.byteswap()
                if len(values) != 32768 * 32:
                    raise ValueError('Unexpected readback length')
                outputs.append(values)
            max_error = 0.0
            for i, (a, b) in enumerate(zip(*outputs)):
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError('Non-finite readback')
                error = abs(a - b) / max(1.0, abs(a))
                max_error = max(max_error, error)
                if error > 2e-4 or (i % 32 == 11 and a != b):
                    raise ValueError(f'{run}: numerical/event mismatch at {i}')
            means = []
            for side in (0, 1):
                selected = [r for r in rows if int(r['mode']) == mode and int(r['side']) == side and r['warmup'] == '0']
                if sorted(int(r['block']) for r in selected) != list(range(2, 10)):
                    raise ValueError('Missing or duplicated timing blocks')
                times = [float(r['gpu_ms']) for r in selected]
                if any(not math.isfinite(t) or t <= 0 for t in times):
                    raise ValueError('Invalid GPU time')
                means.append(statistics.mean(times))
            summary.append(dict(run=run.name, mode=mode, vendor_ms=means[0], native_ms=means[1],
                                reduction=1 - means[1] / means[0], max_scaled_error=max_error))
    if len(summary) != manifest['runs'] * 2:
        raise ValueError('Missing run')
    return summary


def run(args):
    repo = Path(__file__).resolve().parents[2]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    sources = [*repo.glob('Shaders/Modules/OpenPBR/*.slang'), repo / 'Shaders/Modules/OpenPBR.slang',
               repo / 'Shaders/Interop/OpenPBRModule.hlsli', repo / 'tests/rhi/OpenPBRClosureTests.cpp',
               repo / 'tests/rhi/shaders/OpenPBRNativeProbe.slang',
               *repo.glob('External/openpbr-bsdf/**/*.h')]
    identity = {str(p.relative_to(repo)): sha(p) for p in sources}
    env = {k: v for k, v in os.environ.items() if not k.startswith('METALLIC_')}
    if env.get('VK_INSTANCE_LAYERS'):
        raise RuntimeError('Remove injected VK_INSTANCE_LAYERS for ordinary timing')
    metadata = dict(runs=args.runs, sources=identity, executable=str(exe), executable_sha256=sha(exe),
                    scope='32768 BSDF cases; eight dispatches plus WAW barriers / 8; exact buffer LUTs',
                    validation=False, clocks='unaltered', profiling=False,
                    environment={k: v for k, v in env.items() if k.startswith('VK_')})
    (root / 'Metadata.json').write_text(json.dumps(metadata, indent=2))
    for index in range(args.runs):
        out = root / f'run-{index}'
        command = [str(exe), '--gtest_filter=RHIRendering.material_openpbr_native_equivalence',
                   '--rhi-no-validation', '--output-dir', str(out)]
        start = time.time()
        code = None
        try:
            with (root / f'run-{index}.log').open('w') as log:
                code = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
        finally:
            (root / f'run-{index}.json').write_text(json.dumps(dict(command=command, exit=code,
                seconds=time.time()-start), indent=2))
        if code != 0:
            raise RuntimeError(f'Run {index} failed: {code}; preserve its log')
        print(f'run-{index}: passed', flush=True)
    if identity != {str(p.relative_to(repo)): sha(p) for p in sources} or sha(exe) != metadata['executable_sha256']:
        raise RuntimeError('Source or executable changed during measurement')
    metadata['artifacts'] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    (root / 'Manifest.json').write_text(json.dumps(metadata, indent=2))
    summary = verify(root)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    live = sub.add_parser('run')
    live.add_argument('--exe', type=Path, required=True)
    live.add_argument('--output', type=Path, required=True)
    live.add_argument('--runs', type=int, choices=range(3, 11), default=5)
    live.add_argument('--timeout', type=int, default=180)
    check = sub.add_parser('verify')
    check.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        repo = Path(__file__).resolve().parents[2]
        shared = repo / 'build/shader-experiment.lock'
        if shared.exists():
            raise RuntimeError(f'Existing GPU experiment lock: {shared}; inspect its owner first')
        lock = repo / 'build/openpbr-benchmark.lock'
        lock.parent.mkdir(exist_ok=True)
        with lock.open('x') as file:
            json.dump(dict(pid=os.getpid(), output=str(args.output.resolve())), file)
        try:
            run(args)
        finally:
            lock.unlink()
    else:
        print(json.dumps(verify(args.output), indent=2))


if __name__ == '__main__':
    main()
