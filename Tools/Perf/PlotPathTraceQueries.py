"""Verify PT query evidence and plot independent-process timings and image checks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--correctness',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    repo,perf,correct,out=[x.resolve() for x in (args.repo,args.performance,args.correctness,args.output)]
    sys.path.insert(0,str(repo/'Tools/Perf'))
    import PathTraceQueries as pt
    import MaterialBaseline as mb
    summary=pt.verify(perf); validation=pt.verify(correct)
    checks=[case for label,rows in validation['correctness'].items() if not label.endswith('-A') for case in rows]
    out.mkdir(parents=True,exist_ok=False)
    mb.save(out/'Summary.json',summary)
    mb.save(out/'Correctness.json',validation)
    shutil.copyfile(__file__,out/'plot_report.py')
    data=[]
    for process in mb.read(perf/'Processes.json'):
        path=pt.report_path(perf/f'run-{process["block"]:02}-{process["arm"]}')
        for case in mb.read(path)['cases']:
            for row in case['warmup']+case['frames']:
                node=next(n for n in row['nodes'] if n['name']=='PathTrace')
                data.append(dict(block=process['block'],arm=process['arm'],case=case['id'],frame=row['frame'],
                                 warmup=row['warmup'],pt_ms=node['gpuMs'],graph_ms=row['graphMs']))
    with (out/'Frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader();writer.writerows(data)
    cases=[c['id'] for c in mb.read(perf/'Cases.json')['cases']]
    colors=dict(A='#64748b',B='#2b7aab',C='#cc6b38')
    labels=dict(A='Original',B='Primary reuse',C='Primary + shadow')
    fig,axes=plt.subplots(2,len(cases),figsize=(14,8),layout='constrained')
    for col,case in enumerate(cases):
        ax=axes[0,col]
        for arm in 'ABC':
            values=[np.median([r['pt_ms'] for r in data if r['case']==case and r['arm']==arm and
                              r['block']==i and not r['warmup']]) for i in range(6)]
            ax.plot(range(1,7),values,'o-',color=colors[arm],label=labels[arm],lw=1.2)
        ax.set(title=case,xlabel='Independent process block',ylabel='PathTrace GPU time (ms)',xticks=range(1,7))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axes[1,col]
        rows=[r for r in summary['comparisons'] if r['case']==case]
        for i,row in enumerate(rows):
            g=row['ptGain']; lo,hi=g['low']*100,g['high']*100
            ax.errorbar(i,g['mean']*100,yerr=[[g['mean']*100-lo],[hi-g['mean']*100]],fmt='o',capsize=5,color='#2b7aab')
            ax.scatter(np.full(6,i)+np.linspace(-.12,.12,6),np.array(g['pairs'])*100,s=14,color='#94a3b8')
        ax.axhline(0,color='#444',lw=.8)
        ax.axhline(3,color='#888',ls='--',lw=.8)
        ax.set(xticks=range(3),xticklabels=['Primary / original','Shadow / primary','Combined / original'],
               ylabel='Reduction in PT time (%)',title='Mean paired gain; 95% Student-t interval (n=6)')
        ax.tick_params(axis='x',labelrotation=15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('512 x 512 | 2 spp/frame | maxDepth 12 | cache off | 32 warmup + 64 timed frames/process\n'
                 'Points above are process medians; frames are not independent experiments. Positive gain = faster.',fontsize=11)
    fig.savefig(out/'Performance.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(cases),figsize=(14,3.5),layout='constrained')
    for ax,case in zip(axes,cases):
        for arm in 'ABC':
            values=[np.median([r['pt_ms'] for r in data if r['case']==case and r['arm']==arm and r['frame']==f]) for f in range(96)]
            ax.plot(range(96),values,color=colors[arm],label=labels[arm],lw=1)
        ax.axvspan(0,31,color='#ccc',alpha=.25)
        ax.set(title=case,xlabel='Frame (shaded = excluded warmup)',ylabel='PT time (ms), median of 6 processes')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'Warmup.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout='constrained')
    for row,caseid in enumerate(['Mixed4','MixedSharc']):
        captures=[]
        for arm in ('A','C'):
            path=pt.report_path(correct/f'run-00-{arm}')
            case=next(c for c in mb.read(path)['cases'] if c['id']==caseid)
            image=np.fromfile(path.parent/case['image'],dtype='<f4').reshape(case['height'],case['width'],4)[:,:,:3]
            captures.append(image)
        for col,image in enumerate(captures):
            axes[row,col].imshow(np.clip(image/(1+image),0,1)**(1/2.2))
            axes[row,col].set_title(caseid+(' original' if col==0 else ' combined'))
        diff=np.max(np.abs(captures[0]-captures[1]),axis=2)
        im=axes[row,2].imshow(diff,vmin=0,vmax=max(1e-5,float(diff.max())),cmap='magma')
        axes[row,2].set_title(f'Linear HDR max error = {diff.max():.3g}')
        fig.colorbar(im,ax=axes[row,2],shrink=.7)
    for ax in axes.flat: ax.axis('off')
    fig.suptitle('Single-frame noisy PT; identical display transform x/(1+x), gamma 1/2.2')
    fig.savefig(out/'Correctness.png',dpi=180);plt.close(fig)
    lines=['# Path tracing query optimization', '',
           'Production RenderGraph GPU measurements on '+mb.read(perf/'Identity.json')['gpu']+'.', '',
           'Scope: PathTrace pass and graph only. DLSS-RR/NRD, presentation and end-to-end FPS were not measured.', '',
           'All arms use 512 x 512, 2 samples/frame, depth 12, SHARC off and a fixed studio HDR environment. '
           'ABeautifulGame uses the sample camera with that fixed lighting. Six balanced A/B/C blocks; 18 serial processes. '
           'Each process warms 32 frames and measures 64. Cache/driver compilation and final readback are excluded. '
           'Clocks unaltered; other applications were not closed and exclusive GPU ownership is not certified.', '',
           '![Performance](Performance.png)', '',
           '| Case | Comparison | Mean PT gain | 95% paired interval | Decision |',
           '|---|---|---:|---:|---|']
    for row in summary['comparisons']:
        g=row['ptGain']
        lines.append(f"| {row['case']} | {row['comparison']} | {g['mean']:.2%} | [{g['low']:.2%}, {g['high']:.2%}] | {row['decision']} |")
    lines+=['','A = original; B = primary hit reuse; C = primary reuse plus shadow classification. '
            'Intervals use six process-paired gains, never individual frames. No samples were filtered. '
            'Improvement requires PT lower bound >=3%, graph lower bound >=-2%, and per-arm process-median spread <=10%. '
            'Inconclusive is not an accepted speedup.', '', '![Warmup](Warmup.png)', '',
            'Correctness: nine fixed scenarios including ABeautifulGame, 1/2/4 samples, MASK, BLEND, nested textured transmission/absorption, '
            'normal mapping, sky misses, seven guides, moving RenderView, SHARC and legacy Boolean shadows. '
            'The verifier requires exact guide bytes; HDR absolute/relative tolerances were fixed at 2e-5 before measurement.', '',
            f"Observed: {sum(c['hdrByteIdentical'] for c in checks)}/{len(checks)} candidate/case final HDR comparisons "
            f"are byte-identical; {sum(c['captures'] for c in checks if c['capturesByteIdentical'])} "
            f"of {sum(c['captures'] for c in checks)} temporal/guide captures are byte-identical. "
            f"Maximum final HDR error: {max(c['maxAbsoluteError'] for c in checks):.8g}.", '',
            '![Correctness](Correctness.png)', '',
            f"Correctness evidence: [{correct}]({correct.as_posix()})",'',
            f"Performance evidence: [{perf}]({perf.as_posix()})",'',
            'Raw field mapping: `MaterialBaseline.json` → cases[].frames/warmup[].nodes[name=PathTrace].gpuMs '
            'and graphMs, both milliseconds without conversion. Process medians and paired statistics are recomputed '
            'by the current repository verifier. Frames.csv retains every warmup/timed sample.', '',
            'Reproduce:', '', '```powershell',
            f'python -B "{Path(__file__).resolve()}" --repo "{repo}" --performance "{perf}" --correctness "{correct}" --output "<new-report-directory>"',
            '```','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    mb.save(out/'Provenance.json',dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
        performanceManifest=mb.digest(perf/'Manifest.json'),correctnessManifest=mb.digest(correct/'Manifest.json'),
        scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__': main()
