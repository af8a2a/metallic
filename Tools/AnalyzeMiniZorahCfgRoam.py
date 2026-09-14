"""Export phase/scope evidence from the native reference and optional Metallic replay."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('--metallic',type=Path)
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--output',type=Path,help='Write derived artifacts here, preserving an earlier analysis')
    args=parser.parse_args()
    output=args.output or args.reference
    output.mkdir(parents=True,exist_ok=True)
    route=read(args.reference/'Replay.json')
    result={'protocol':route['protocol'],'referenceManifest':read(args.reference/'Manifest.json'),
            'route':{k:v for k,v in route.items() if k!='frames'},'reference':{},'metallic':{},
            'limitations':['Reference exports rounded integer-microsecond aggregates, not per-frame P99.',
                'Native cfg uses RT + DLSS; aligned reference remains RT, Metallic remains VBuffer + CLAS production.',
                f'Metallic uses {route["phaseFrames"]} full frames per segment; reference discards reset/delayed-query frames. Means are segment comparisons, not exact timestamp pairs.',
                'Scope gpuAvgMs/cpuAvgMs are amortized over phaseFrames; recorded averages cover only samples where a conditional Metallic scope exists. Native reports already include inactive frames.',
                'Memory is the engine-reported endpoint allocation with engine-specific accounting. Device-wide telemetry includes other applications.']}
    scopes,memory=[],[]
    for case in ('cfg1','aligned1','aligned2','cfg2','verify'):
        data=read(args.reference/case/'Profile.json')
        if data['status']!='capture_complete':
            raise ValueError(f'Incomplete {case}')
        if case!='verify' and data['routeSha256']!=result['referenceManifest']['replaySha256']:
            raise ValueError('Reference camera route differs')
        result['reference'][case]=data
        for phase in data['profiles']:
            for t in phase['timers']:
                scopes.append(dict(engine='vk',case=case,phase=phase['name'],path=t['path'],samples=t['samples'],phaseFrames=t['samples'],cpuOnly=False,
                    gpuRecordedAvgMs=t['gpuMs']['avg'],cpuRecordedAvgMs=t['cpuMs']['avg'],
                    gpuAvgMs=t['gpuMs']['avg'],gpuMinMs=t['gpuMs']['min'],gpuMaxMs=t['gpuMs']['max'],
                    cpuAvgMs=t['cpuMs']['avg'],cpuMinMs=t['cpuMs']['min'],cpuMaxMs=t['cpuMs']['max']))
            for group,items in phase['memory'].items():
                for name,values in items.items():
                    memory.append(dict(engine='vk',case=case,phase=phase['name'],group=group,name=name,**values))
    if args.metallic:
        manifest=read(args.metallic/'Manifest.json')
        if manifest['replaySha256'].lower()!=result['referenceManifest']['replaySha256'].lower():
            raise ValueError('Metallic input replay hash differs')
        result['metallicManifest']=manifest
        for case in ('m1','m2','quality'):
            directory=args.metallic/case
            baseline=read(directory/'Baseline.json')
            if baseline['status']!='passed' or baseline['frameCount']!=len(route['frames']):
                raise ValueError(f'Metallic {case} failed')
            cameras=read(directory/'Cameras.json')
            if cameras!=[f['camera'] for f in route['frames']]:
                raise ValueError('Executed Metallic camera data differs')
            data={'phases':baseline['phases'],'quality':baseline['quality'],
                  'memoryAfterReplay':baseline['memoryAfterReplay'],'scopes':{}}
            rows=[]
            group={}
            with (directory/'Frames.jsonl').open() as source:
                for line in source:
                    frame=json.loads(line)
                    index=frame['frame']
                    if index!=len(rows) or frame['phase']!=route['frames'][index]['phase']:
                        raise ValueError('Metallic frame/phase mismatch')
                    for node in frame.pop('nodes'):
                        timers=[(node['name'],node)]
                        paths=[]
                        for t in node['sections']:
                            parent=node['name'] if t['parent']==0xffffffff else paths[t['parent']]
                            path=parent+'/'+t['name']
                            paths.append(path)
                            timers.append((path+' ['+t['queue']+']',t))
                        for name,t in timers:
                            group.setdefault((frame['phase'],name),[]).append((t['gpuMs'],t['cpuMs']))
                    rows.append(frame)
            if len(rows)!=len(route['frames']):
                raise ValueError('Missing Metallic timing samples')
            for (phase,path),times in group.items():
                gpu,cpu=zip(*times)
                phase_frames=sum(f['phase']==phase for f in rows)
                cpu_only=path.endswith(' [cpu]')
                timer=dict(engine='metallic',case=case,phase=phase,path=path,samples=len(times),phaseFrames=phase_frames,cpuOnly=cpu_only,
                    gpuRecordedAvgMs=statistics.mean(gpu),cpuRecordedAvgMs=statistics.mean(cpu),
                    gpuAvgMs=sum(gpu)/phase_frames,gpuMinMs=min(gpu) if len(times)==phase_frames else 0,gpuMaxMs=max(gpu),
                    cpuAvgMs=sum(cpu)/phase_frames,cpuMinMs=min(cpu) if len(times)==phase_frames else 0,cpuMaxMs=max(cpu))
                if cpu_only:
                    for key in ('gpuRecordedAvgMs','gpuAvgMs','gpuMinMs','gpuMaxMs'):
                        timer[key]=None
                data['scopes'].setdefault(phase,{})[path]=timer
                scopes.append(timer)
            data['phaseMeans']={p['name']:{metric:statistics.mean(f[metric] for f in rows if f['phase']==p['name'])
                for metric in ('gpuMs','cpuRecordMs','cpuExecuteMs','hostFrameMs')} for p in route['phases']}
            data['endpoints']={f['phase']:f['stream'] for f in rows}
            data['peaks']={k:max(f['stream'][k] for f in rows) for k in ('geometryBytes','clasBytes','clasScratchBytes','clasPending')}
            data['final']=rows[-1]['stream']
            result['metallic'][case]=data
            for phase,values in data['endpoints'].items():
                for name,key in (('Geometry','geometryBytes'),('CLAS','clasBytes')):
                    memory.append(dict(engine='metallic',case=case,phase=phase,group='Memory',name=name,actual=values[key],
                        reserved=values['geometryBudgetBytes' if name=='Geometry' else 'clasCapacityBytes']))
    for name,rows,columns in [('Scopes.csv',scopes,list(scopes[0])),
                              ('Memory.csv',memory,['engine','case','phase','group','name','actual','reserved'])]:
        with (output/name).open('w',newline='',encoding='utf-8-sig') as target:
            writer=csv.DictWriter(target,fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    (output/'Evidence.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        phase_names=[p['name'] for p in route['phases'] if p['name']!='warmup']
        fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
        for case,color in [('cfg1','#4586bd'),('cfg2','#9cbcdc'),('aligned1','#229674'),('aligned2','#a0cbb8')]:
            data={p['name']:p for p in result['reference'][case]['profiles']}
            axes[0].plot(phase_names,[next(t['gpuMs']['avg'] for t in data[p]['timers'] if t['name']=='Frame') for p in phase_names],label='vk '+case,color=color,marker='.')
            for axis,name in ((axes[1],'Geometry'),(axes[2],'CLAS')):
                axis.plot(phase_names,[data[p]['memory']['Memory'][name]['actual']/1048576 for p in phase_names],label='vk '+case,color=color,marker='.')
        for case,color in [('m1','#c37736'),('m2','#e4b580')]:
            if case not in result['metallic']:
                continue
            data=result['metallic'][case]
            axes[0].plot(phase_names,[data['phaseMeans'][p]['gpuMs'] for p in phase_names],label='Metallic '+case,color=color,marker='.')
            for axis,key in ((axes[1],'geometryBytes'),(axes[2],'clasBytes')):
                axis.plot(phase_names,[data['endpoints'][p][key]/1048576 for p in phase_names],color=color,marker='.')
        axes[0].set_title('MiniZorah: same camera route, different rendering paths\nNative cfg: RT + DLSS | Aligned: RT, 1080p / 1.5 px | Metallic: VBuffer + CLAS')
        axes[0].set_ylabel('GPU frame mean (ms)')
        axes[1].set_ylabel('Geometry endpoint (MiB)')
        axes[2].set_ylabel('CLAS endpoint (MiB)')
        axes[0].legend(ncol=3)
        for axis in axes:
            axis.grid(alpha=.2)
        axes[-1].tick_params(axis='x',rotation=25)
        fig.savefig(output/'Comparison.png',dpi=160)
        plt.close(fig)
        if result['metallic'] and any(t['cpuOnly'] for t in scopes):
            fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
            categories=['Residency completion','CLAS completion / expiry','GPU request feedback',
                        'Joint cold page reclaim','Prepare page uploads']
            for axis,case in zip(axes,('m1','m2')):
                data=result['metallic'][case]['scopes']
                totals=[data[p]['GPUDriven/Stream Begin [graphics]']['cpuAvgMs'] for p in phase_names]
                bottom=[0.0]*len(phase_names)
                for category in categories:
                    values=[data[p].get('GPUDriven/Stream Begin/'+category+' [cpu]',{}).get('cpuAvgMs',0) for p in phase_names]
                    axis.bar(phase_names,values,bottom=bottom,label=category)
                    bottom=[a+b for a,b in zip(bottom,values)]
                axis.bar(phase_names,[max(0,a-b) for a,b in zip(totals,bottom)],bottom=bottom,label='Other / instrumentation')
                axis.set_title(case+' | CPU Stream Begin')
                axis.set_ylabel('Mean CPU time (ms)')
                axis.grid(axis='y',alpha=.2)
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside upper center',ncol=3)
            axes[-1].tick_params(axis='x',rotation=25)
            fig.savefig(output/'CpuStreamBegin.png',dpi=160)
            plt.close(fig)
    print(f'Validated routes and exported {len(scopes)} scope rows, {len(memory)} memory/counter rows.')


if __name__=='__main__':
    main()
