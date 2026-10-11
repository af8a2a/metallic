"""Differential PT query validation and balanced independent-process A/B/C timings.

A=original, B=primary hit reuse, C=primary reuse plus shadow classification.
Uses the production MaterialBaseline RenderGraph harness. Restores the installed
candidate after every experiment, including failures. No quality settings change.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

import numpy as np
import psutil
from scipy.stats import t
import MaterialBaseline as mb

ROOT = Path(__file__).resolve().parents[2]
SHADERS = ROOT / "Shaders/Features/PathTracing"
FILES = ["ScenePathTrace.slang", "StreamSceneRayQuery.slang", "OpenPBRRayQueryPathTrace.slang",
         "OpenPBRRayQueryPathTraceGuides.slang"]
ORDERS = ["ABC", "CBA", "BAC", "CAB", "ACB", "BCA"]
LOCK = ROOT / "build/shader-experiment.lock"


def report_path(folder):
    # The HTML harness also copies evidence into reports/.../cases/0.
    # The producer's root report is authoritative, not that duplicate.
    if (folder / "MaterialBaseline.json").is_file():
        return folder / "MaterialBaseline.json"
    paths = list(folder.rglob("MaterialBaseline.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one completed production report: {folder}: {paths}")
    return paths[0]


def compare_reports(a_path, b_path, expected):
    a, b = mb.read(a_path), mb.read(b_path)
    if [c['id'] for c in a['cases']] != expected or [c['id'] for c in b['cases']] != expected:
        raise ValueError("Missing differential case")
    results = []
    for ca, cb in zip(a['cases'], b['cases']):
        if ca['resolvedGraph'] != cb['resolvedGraph'] or ca['format'] != cb['format']:
            raise ValueError("Workload or output format changed")
        dtype = '<f4' if ca['format'] == 'RGBA32F' else '<f2'
        pa = np.fromfile(a_path.parent / ca['image'], dtype=dtype).astype(np.float64)
        pb = np.fromfile(b_path.parent / cb['image'], dtype=dtype).astype(np.float64)
        if pa.size != ca['width'] * ca['height'] * 4 or pa.shape != pb.shape:
            raise ValueError("Truncated HDR")
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            raise ValueError("Nonfinite HDR")
        error = np.abs(pa-pb)
        # Declared before measurement; never adjusted to pass a candidate.
        if not np.all(error <= 2e-5 + 2e-5*np.abs(pa)):
            raise ValueError(f"HDR changed: {ca['id']}: max={error.max()}")
        if len(ca.get('captures',[])) != len(cb.get('captures',[])):
            raise ValueError("Missing guide/temporal captures")
        exact = True
        for xa, xb in zip(ca.get('captures',[]),cb.get('captures',[])):
            if xa != xb:
                raise ValueError("Differential capture sequence changed")
            ba, bb = (a_path.parent / xa['file']).read_bytes(), (b_path.parent / xb['file']).read_bytes()
            if len(ba) != xa['bytes'] or len(bb) != xb['bytes']:
                raise ValueError("Truncated differential capture")
            exact &= ba == bb
            if ba != bb:
                if not xa['output'].endswith('.color'):
                    raise ValueError(f"Guide changed: {ca['id']} {xa['output']}")
                va, vb = np.frombuffer(ba,dtype=dtype).astype(float),np.frombuffer(bb,dtype=dtype).astype(float)
                if not np.isfinite(va).all() or not np.isfinite(vb).all() or not np.all(np.abs(va-vb)<=2e-5+2e-5*np.abs(va)):
                    raise ValueError(f"Temporal HDR changed: {ca['id']} frame {xa['frame']}")
        results.append(dict(case=ca['id'],hdrByteIdentical=(a_path.parent/ca['image']).read_bytes()==(b_path.parent/cb['image']).read_bytes(),
                            maxAbsoluteError=float(error.max()),capturesByteIdentical=exact,captures=len(ca.get('captures',[]))))
    return results


def restore(root):
    journal = mb.read(root / 'Transaction.json')
    for name in FILES:
        known = {mb.digest(root / 'sources' / arm / name) for arm in 'ABC'}
        if mb.digest(SHADERS / name) not in known:
            raise RuntimeError(f"Concurrent shader edit preserved: {name}; manual reconciliation required")
    for name in FILES:
        shutil.copyfile(root/'sources/C'/name,SHADERS/name)
    journal['restored'] = True
    mb.save(root/'Transaction.json',journal)


def run(args):
    root, exe, cases = args.output.resolve(), args.exe.resolve(), args.cases.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source_dirs = dict(A=args.baseline.resolve(),B=args.primary.resolve(),C=SHADERS)
    for arm, folder in source_dirs.items():
        (root/'sources'/arm).mkdir(parents=True)
        for name in FILES:
            shutil.copyfile(folder/name,root/'sources'/arm/name)
    mb.save(root/'Transaction.json',dict(pid=os.getpid(),restored=False,source=str(SHADERS)))
    identity = mb.identity(exe,cases)
    identity.update(phase=args.phase,clocks='unaltered',orders=['ABC'] if args.phase=='correctness' else ORDERS,
                    statisticalUnit='independent-process median; one A/B/C triplet per block',
                    scope='production PathTrace pass and render-graph GPU timestamps; excludes denoiser/upscaler/presentation',
                    thresholds=dict(hdrAbsolute=2e-5,hdrRelative=2e-5,guides='byte exact',gainLower=.03,graphGainLower=-.02))
    identity['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    mb.save(root/'Identity.json',identity)
    shutil.copyfile(cases,root/'Cases.json')
    env = {k:v for k,v in os.environ.items() if not k.startswith('METALLIC_')}
    if env.get('VK_INSTANCE_LAYERS'):
        raise RuntimeError('Remove injected layers before ordinary measurements')
    env['METALLIC_MATERIAL_BASELINE_CASES'] = str(cases)
    env['METALLIC_SHADER_DEBUG_MODE'] = 'disabled'
    for name in ['PreparePathTraceQueries.py','PathTraceQueries.py','MaterialBaseline.py']:
        shutil.copyfile(Path(__file__).parent/name,root/name)
    mutable = {'Shaders/Features/PathTracing/'+n for n in FILES}
    def check_identity():
        for path,digest in identity['sources'].items():
            if path not in mutable and mb.digest(ROOT/path)!=digest:
                raise RuntimeError(f'Source changed during experiment: {path}')
        for group in ('workload','binaries'):
            for path,digest in identity[group].items():
                if mb.digest(Path(path))!=digest:
                    raise RuntimeError(f'{group} changed: {path}')
    processes = []
    try:
        for block,order in enumerate(identity['orders']):
            for arm in order:
                check_identity()
                for name in FILES:
                    if mb.digest(SHADERS/name) not in {mb.digest(root/'sources'/a/name) for a in 'ABC'}:
                        raise RuntimeError(f'Concurrent shader change: {name}')
                    shutil.copyfile(root/'sources'/arm/name,SHADERS/name)
                folder = root / f'run-{block:02}-{arm}'
                cmd = [str(exe),'--gtest_filter=RHIRendering.material_phase0_baseline',
                       '--rhi-validation' if args.phase=='correctness' else '--rhi-no-validation',
                       '--output-dir',str(folder)]
                record = dict(block=block,arm=arm,command=cmd,start=time.time(),exit=None,
                              shaders={n:mb.digest(SHADERS/n) for n in FILES})
                print(f'block={block} arm={arm} start',flush=True)
                with (root/(folder.name+'.log')).open('w',encoding='utf-8') as log:
                    process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    record['pid'] = process.pid
                    try:
                        record['exit'] = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                        raise
                    finally:
                        record['end'] = time.time()
                        processes.append(record)
                        mb.save(root/'Processes.json',processes)
                if record['exit'] != 0:
                    raise RuntimeError(f'{folder.name}: exit {record["exit"]}; see log')
                report = report_path(folder)
                if not all(c.get('validHDR') for c in mb.read(report)['cases']):
                    raise RuntimeError('Invalid HDR, not a performance result')
                # Archive cached production SPIR-V referenced by the runtime log.
                log_text = (root/(folder.name+'.log')).read_text(encoding='utf-8')
                bound = []
                for entry,path in re.findall(r"Loaded (Features/PathTracing/\S+) from '([^']+\.spv)'",log_text):
                    cache=Path(path); digest=mb.digest(cache); dst=root/'spirv'/(digest+'.cache')
                    dst.parent.mkdir(exist_ok=True)
                    if not dst.exists(): shutil.copyfile(cache,dst)
                    blob=cache.read_bytes()
                    size=int.from_bytes(blob[32:40],'little')
                    payload=blob[-size:]
                    if len(blob)<48 or size<20 or payload[:4]!=b'\x03\x02\x23\x07':
                        raise ValueError('Invalid shader-cache SPIR-V payload')
                    bound.append(dict(entry=entry,file='spirv/'+dst.name,sha256=digest,
                                      payloadBytes=size,spirvSha256=hashlib.sha256(payload).hexdigest()))
                record['shaderCacheLoads']=bound
                mb.save(root/'Processes.json',processes)
                check_identity()
                print(f'{folder.name}: passed ({record["end"]-record["start"]:.1f}s)',flush=True)
    finally:
        restore(root)
    mb.seal(root)
    summary=verify(root)
    print(json.dumps(summary,indent=2),flush=True)


def verify(root):
    manifest=mb.read(root/'Manifest.json')
    actual={str(p.relative_to(root)).replace('\\','/'):mb.digest(p) for p in root.rglob('*') if p.is_file() and p.name!='Manifest.json'}
    if manifest!=actual or not mb.read(root/'Transaction.json')['restored']:
        raise ValueError('Evidence integrity/restoration failed')
    identity,config,processes=mb.read(root/'Identity.json'),mb.read(root/'Cases.json'),mb.read(root/'Processes.json')
    expected=[c['id'] for c in config['cases']]
    sequence=[(i,a) for i,o in enumerate(identity['orders']) for a in o]
    if [(p['block'],p['arm']) for p in processes]!=sequence or any(p['exit']!=0 for p in processes):
        raise ValueError('Incomplete process sequence')
    if any(a['end']>b['start'] for a,b in itertools.pairwise(processes)):
        raise ValueError('Overlapping GPU runs')
    first=report_path(root/'run-00-A')
    differences={}
    medians={}
    for p in processes:
        label=f'run-{p["block"]:02}-{p["arm"]}'
        path=report_path(root/label)
        differences[label]=compare_reports(first,path,expected)
        report=mb.read(path)
        for shader in p['shaderCacheLoads']:
            cache=root/shader['file']; blob=cache.read_bytes()
            if mb.digest(cache)!=shader['sha256'] or hashlib.sha256(blob[-shader['payloadBytes']:]).hexdigest()!=shader['spirvSha256']:
                raise ValueError('Shader payload identity changed')
        if report['validation'] != (identity['phase']=='correctness'):
            raise ValueError('Validation/timing mode mismatch')
        for case in report['cases']:
            rows=case['frames']
            if [r['frame'] for r in rows]!=list(range(config['warmupFrames'],config['warmupFrames']+config['timingFrames'])):
                raise ValueError('Wrong timing window')
            if len({r['executionId'] for r in rows})!=len(rows): raise ValueError('Duplicate frames')
            pt=[]; graph=[]
            for row in rows:
                nodes=[n for n in row['nodes'] if n['name']=='PathTrace']
                if len(nodes)!=1: raise ValueError('Missing production PT timestamp')
                pt.append(nodes[0]['gpuMs']); graph.append(row['graphMs'])
            if any(v is None or not math.isfinite(v) or v<=0 for v in pt+graph): raise ValueError('Invalid timestamp')
            medians[(p['block'],p['arm'],case['id'])]=(statistics.median(pt),statistics.median(graph))
    result=dict(correctness=differences,scope=identity['scope'],comparisons=[])
    if identity['phase']=='performance':
        n=len(identity['orders'])
        for case in expected:
            for a,b in [('A','B'),('B','C'),('A','C')]:
                values=[]
                for metric in (0,1):
                    gains=[1-medians[(i,b,case)][metric]/medians[(i,a,case)][metric] for i in range(n)]
                    mean=statistics.mean(gains)
                    margin=float(t.ppf(.975,n-1))*statistics.stdev(gains)/math.sqrt(n)
                    values.append(dict(mean=mean,low=mean-margin,high=mean+margin,pairs=gains))
                decision='improvement' if values[0]['low']>=.03 and values[1]['low']>=-.02 else 'inconclusive'
                if values[0]['high']<0 or values[1]['high']<-.02: decision='regression'
                spread=[]
                for arm in (a,b):
                    times=[medians[(i,arm,case)][0] for i in range(n)]
                    spread.append((max(times)-min(times))/statistics.median(times))
                if max(spread)>.1: decision='inconclusive'
                result['comparisons'].append(dict(case=case,comparison=a+b,n=n,ptGain=values[0],graphGain=values[1],decision=decision,
                    processMedianSpread=spread,
                    aMs=[medians[(i,a,case)][0] for i in range(n)],bMs=[medians[(i,b,case)][0] for i in range(n)]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    live=sub.add_parser('run')
    for name in ['exe','cases','baseline','primary','output']: live.add_argument('--'+name,type=Path,required=True)
    live.add_argument('--phase',choices=['correctness','performance'],required=True)
    live.add_argument('--timeout',type=int,default=600)
    check=sub.add_parser('verify'); check.add_argument('output',type=Path)
    recover=sub.add_parser('recover'); recover.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.command=='run':
        with LOCK.open('x') as lock: json.dump(dict(pid=os.getpid(),output=str(args.output.resolve())),lock)
        try: run(args)
        finally:
            if (args.output/'Transaction.json').exists() and mb.read(args.output/'Transaction.json')['restored']:
                LOCK.unlink()
    elif args.command=='recover':
        owner=mb.read(LOCK)
        if psutil.pid_exists(owner['pid']) or Path(owner['output']).resolve()!=args.output.resolve():
            raise RuntimeError('Active or unrelated experiment owner')
        restore(args.output.resolve()); LOCK.unlink()
    else:
        print(json.dumps(verify(args.output.resolve()),indent=2))
