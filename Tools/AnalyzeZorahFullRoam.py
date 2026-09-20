"""Summarize Full editor replay. Nested scopes and independent submits are never summed."""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def distribution(values):
    if not values:
        return None
    data = sorted(values)
    def percentile(p):
        return data[max(0, math.ceil(len(data)*p)-1)]
    return dict(count=len(data), mean=sum(data)/len(data), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), max=data[-1])


def frame_stats(rows):
    values = [r['frameMs'] for r in rows]
    run = longest = 0
    for v in values:
        run = run+1 if v > 1000/30 else 0
        longest = max(run, longest)
    return dict(frameMs=distribution(values), overBudget=sum(v > 1000/30 for v in values),
                longestOverBudgetRun=longest, sampledRouteMeets30Fps=bool(values) and max(values) <= 1000/30)


def analyze(root):
    capture = json.loads((root/'Capture.json').read_text(encoding='utf-8-sig'))
    if capture['status'] != 'capture_complete':
        raise ValueError('Capture incomplete')
    rows = [json.loads(line) for line in (root/'Frames.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    if len(rows) != capture['frames'] or not rows or len({r['frame'] for r in rows}) != len(rows):
        raise ValueError('Frame count/identity mismatch')
    if any(not math.isfinite(r['frameMs']) or r['frameMs'] <= 0 for r in rows):
        raise ValueError('Invalid frame times')
    definitions = {s['id']: s for s in capture['scopes']}
    stages = defaultdict(list)
    stage_timers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    timers = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stages[row['stage']].append(row)
        for s in row['scopes']:
            timers[s['id']]['cpu'].append(s['cpuMs'])
            stage_timers[row['stage']][s['id']]['cpu'].append(s['cpuMs'])
            if s['gpuMs'] is not None:
                timers[s['id']]['gpu'].append(s['gpuMs'])
                stage_timers[row['stage']][s['id']]['gpu'].append(s['gpuMs'])
    scopes = [dict(**definitions[i], cpuMs=distribution(v['cpu']), gpuMs=distribution(v['gpu'])) for i,v in timers.items()]
    required = ['BLAS reset','BLAS cut compare','BLAS count','BLAS setup','BLAS insert','BLAS build','TLAS input','TLAS build']
    missing = [name for name in required if not any(s['path'].endswith('/'+name) and s['gpuMs'] and s['gpuMs']['count']==len(rows) for s in scopes)]
    if missing:
        raise ValueError(f'Missing RTAS timing coverage: {missing}')
    if not any(s['path'].endswith('/Texture streaming') and s['cpuMs']['count']==len(rows) for s in scopes):
        raise ValueError('Missing texture CPU timing coverage')
    uploads = [json.loads(line) for line in (root/'Uploads.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    executions = {s['executionId'] for r in rows for s in r['scopes'] if s['executionId'] is not None}
    uploads = [s for s in uploads if s['submitFrame'] in executions]
    slowest = sorted(rows, key=lambda r:r['frameMs'], reverse=True)[:20]
    result = dict(protocol=capture['protocol'], outputExtent=capture['outputExtent'], renderExtent=capture['renderExtent'],
                  validation=capture['validationRequested'], diagnosticRun=capture.get("diagnosticRun",False),
                  diagnosticFrames=sum(r.get("diagnostic",False) for r in rows), **frame_stats(rows),
                  stages={k:frame_stats(v) for k,v in stages.items()}, scopes=scopes,
                  stageScopes={stage:[dict(**definitions[i], cpuMs=distribution(v['cpu']), gpuMs=distribution(v['gpu']))
                                      for i,v in counts.items()] for stage,counts in stage_timers.items()},
                  uploadGpuMs=distribution([s['gpuMs'] for s in uploads if s['gpuMs'] is not None]),
                  uploadMissingGpu=sum(s['gpuMs'] is None for s in uploads),
                  slowest=[{k:r[k] for k in ('frame','seconds','stage','frameMs','availableBytes','streaming')} for r in slowest])
    (root/'Summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    gpu = sorted((s for s in scopes if s['gpuMs']), key=lambda s:s['gpuMs']['mean'], reverse=True)
    lines = ['# Full editor roam', '', f"Output {capture['outputExtent']}; render {capture['renderExtent']}; validation={capture['validationRequested']}.",
             '', '| Stage | Frames | p50 ms | p95 ms | p99 ms | Max ms | >33.33 ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for stage, data in [('all',result), *result['stages'].items()]:
        d=data['frameMs']
        lines.append(f"| {stage} | {d['count']} | {d['p50']:.3f} | {d['p95']:.3f} | {d['p99']:.3f} | {d['max']:.3f} | {data['overBudget']} |")
    if result['diagnosticRun']:
        lines += ['', '**Diagnostic run: workload replay/copies perturb timing; not a performance acceptance sample.**']
    lines += ['', 'GPU scopes are inclusive; do not sum parents and children. Scope means use recorded occurrences. Independent uploads are separate submissions.', '', '| Scope | GPU mean ms | GPU p99 ms |', '|---|---:|---:|']
    for s in gpu[:25]:
        lines.append(f"| {s['path']} | {s['gpuMs']['mean']:.3f} | {s['gpuMs']['p99']:.3f} |")
    (root/'Summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('frameMs','overBudget','sampledRouteMeets30Fps','uploadGpuMs')},ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('capture',type=Path)
    root=parser.parse_args().capture
    if (root/'Capture.json').exists():
        analyze(root)
    else:
        runs=sorted(p for p in root.glob('run*') if p.is_dir())
        if not runs:
            raise ValueError('No captures found')
        results=[]
        conditions=None
        for run in runs:
            c=json.loads((run/'Capture.json').read_text(encoding='utf-8-sig'))
            current={k:c[k] for k in ('outputExtent','renderExtent','validationRequested','config','absoluteKeyframes','graph','hidden','vsync')}
            if conditions is not None and current!=conditions:
                raise ValueError('Runs have different capture conditions')
            conditions=current
            summary=json.loads((run/'Summary.json').read_text(encoding='utf-8'))
            results.append(dict(run=run.name,frameMs=summary['frameMs'],overBudget=summary['overBudget'],
                                longestOverBudgetRun=summary['longestOverBudgetRun'],uploadGpuMs=summary['uploadGpuMs'],
                                sampledRouteMeets30Fps=summary['sampledRouteMeets30Fps']))
        aggregate=dict(conditions=conditions,runs=results,
                       allSampledRoutesMeet30Fps=all(r['sampledRouteMeets30Fps'] for r in results))
        (root/'Comparison.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2),encoding='utf-8')
        lines=['# Full replay repetitions','',f"Output {conditions['outputExtent']}; render {conditions['renderExtent']}; hidden={conditions['hidden']}.",
               '', '| Run | Frames | p50 ms | p95 ms | p99 ms | Max ms | >33.33 ms |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in results:
            d=r['frameMs']
            lines.append(f"| {r['run']} | {d['count']} | {d['p50']:.3f} | {d['p95']:.3f} | {d['p99']:.3f} | {d['max']:.3f} | {r['overBudget']} |")
        (root/'Comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps(results,ensure_ascii=False))
