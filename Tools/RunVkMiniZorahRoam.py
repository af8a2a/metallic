"""Replay the MiniZorah .cfg with vk_lod_clusters' native camera/profiler sequencer.

Runs headless from a private runtime directory; never controls the desktop or
edits the reference checkout. The original .cfg is parsed by the application.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_quiet(arguments, **kwargs):
    return subprocess.run(arguments, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                          check=True, **kwargs)


def make_route(cfg, distance, frames):
    match = re.search(r'--camerastring\s+"([^"]+)"', cfg.read_text(encoding="utf-8-sig"))
    if not match:
        raise ValueError("The config must contain a quoted --camerastring")
    values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", match[1])]
    if len(values) != 12:
        raise ValueError("Expected eye, center, up, FOV and clip planes")
    original = dict(eye=values[:3], center=values[3:6], up=values[6:9], fovDegrees=values[9],
                    znear=values[10], zfar=values[11], reversedZ=True)
    direction = [original['center'][i] - original['eye'][i] for i in (0, 2)]
    length = math.hypot(*direction)
    if length < 1e-6:
        raise ValueError("Camera must have a horizontal direction")
    direction = [direction[0] / length, 0., direction[1] / length]

    def camera_at(offset):
        camera = dict(original)
        for key in ('eye', 'center'):
            camera[key] = [original[key][i] + direction[i] * offset for i in range(3)]
        return camera

    def key(camera):
        return ', '.join('{' + ', '.join(format(v, '.17g') for v in values) + '}'
                         for values in (camera['eye'], camera['center'], camera['up'], [camera['fovDegrees']]))

    d = distance / 3
    phases = [('warmup', 0, 0), ('start_hold', 0, 0), ('forward_1', 0, d),
              ('forward_2', d, 2*d), ('forward_3', 2*d, distance), ('far_hold', distance, distance),
              ('return_1', distance, 2*d), ('return_2', 2*d, d), ('return_3', d, 0), ('return_hold', 0, 0)]
    paths, replay, sequence = [], [], []
    for index, (name, start, end) in enumerate(phases):
        paths.append(f"smooth 0 loop 0 dur {frames/60:.17g} ; {key(camera_at(start))} ; {key(camera_at(end))}")
        sequence.append(f'SEQUENCE "{name}"\n--sequenceframes {frames}\n--sequenceaverages 0\n--sequenceresetframes 8\n--runcamerapath {index} {frames}\n')
        for f in range(frames):
            replay.append({'phase': name, 'camera': camera_at(start + (end-start)*f/(frames-1))})
    return dict(protocol='minizorah-cfg-roam-v1', originalCamera=original, horizontalDirection=direction,
                forwardDistance=distance, totalTravelDistance=2*distance, phaseFrames=frames, paced=False,
                nominalStepSeconds=1/60, phases=[dict(name=n, startOffset=a, endOffset=b) for n,a,b in phases],
                frames=replay), '\n'.join(paths)+'\n', '\n'.join(sequence)


def parse_report(log, route):
    timer_re = re.compile(r'Timeline "([^"]+)"; level (-?\d+); Timer "([^"]+)"; GPU; avg (\d+); min (\d+); max (\d+); last (\d+); CPU; avg (\d+); min (\d+); max (\d+); last (\d+); samples (\d+);')
    reports = []
    for index, name, text in re.findall(r'ParameterSequence (\d+) "([^"]+)" = \{\n(.*?)\n\}', log, re.S):
        if name == 'capture_flush':
            continue
        timers, parents = [], {}
        for fields in timer_re.findall(text):
            timeline, level, timer = fields[:3]
            level = int(level)
            path = '/'.join(parents[i] for i in range(level) if i in parents) + '/' + timer
            parents[level] = timer
            parents = {i:n for i,n in parents.items() if i <= level}
            vals = list(map(int, fields[3:]))
            timers.append(dict(timeline=timeline, level=level, name=timer, path=path.lstrip('/'),
                gpuMs=dict(zip(('avg','min','max','last'), (v/1000 for v in vals[:4]))),
                cpuMs=dict(zip(('avg','min','max','last'), (v/1000 for v in vals[4:8]))), samples=vals[8]))
        memory_match = re.search(rf'MemoryReport {index} "{re.escape(name)}" = \{{\s*\n(.*?)\n\}}', log, re.S)
        if not memory_match:
            raise ValueError(f"Missing memory report for {name}")
        memory = {}
        section = None
        for line in memory_match[1].splitlines():
            cells = [x.strip() for x in line.split(';') if x.strip()]
            if len(cells) >= 2 and cells[1] == 'Actual':
                section = cells[0]
                memory[section] = {}
            elif section and len(cells) >= 2:
                memory[section][cells[0]] = {'actual': int(cells[1]), **({'reserved': int(cells[2])} if len(cells)>2 else {})}
        if not timers or min(t['samples'] for t in timers) < route['phaseFrames']//2:
            raise ValueError(f"Insufficient profiler samples in {name}")
        if memory.get('Resident', {}).get('Groups', {}).get('actual', 0) == 0:
            raise ValueError(f"No resident scene in {name}")
        reports.append(dict(index=int(index), name=name, timers=timers, memory=memory))
    if [r['name'] for r in reports] != [p['name'] for p in route['phases']]:
        raise ValueError("Missing/reordered camera sequences")
    for bad in ('VK_ERROR_DEVICE_LOST', 'VUID-', 'out of range', 'could not parse camera path', 'could not open'):
        if bad in log:
            raise ValueError(f"Invalid run: {bad}")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'Asset/MiniZorah/zorah_main_public.v2.cfg')
    parser.add_argument('--reference', type=Path, default=Path('E:/vk_lod_clusters'))
    parser.add_argument('--nvpro', type=Path, default=Path('E:/nvpro_core2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distance', type=float, default=12., help='Horizontal scene units forward; then return')
    parser.add_argument('--phase-frames', type=int, default=300)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--exit-grace-seconds', type=float, default=8,
                        help='Owned process cleanup grace after all required profiler reports are complete')
    parser.add_argument('--cases', nargs='+', choices=['cfg1','cfg2','aligned1','aligned2','verify'],
                        default=['cfg1','aligned1','aligned2','cfg2','verify'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; existing evidence is never overwritten')
    if args.phase_frames < 32 or not 0 < args.distance <= 100 or min(args.width, args.height) < 64:
        raise ValueError('Invalid route/resolution')
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    # Require an existing cooked reference scene; avoid an accidental all-scene cook.
    config_text = args.config.read_text(encoding='utf-8-sig')
    scene_name = config_text.splitlines()[0].strip().strip('"')
    scene_path = args.config.parent/scene_name
    cache = Path(str(scene_path)+'.nvsngeo')
    if not scene_path.exists() or not cache.exists():
        raise ValueError(f'Reference model/cache missing: {cache}')
    args.output.mkdir(parents=True)
    route, paths, sequence = make_route(args.config, args.distance, args.phase_frames)
    write_json(args.output/'Replay.json', route)
    (args.output/'CameraPaths.txt').write_text(paths, encoding='utf-8')
    (args.output/'Sequences.txt').write_text(sequence, encoding='utf-8')
    shutil.copy2(args.config, args.output/'Original.cfg')
    runtime = args.output/'runtime'
    runtime.mkdir()
    binaries = args.reference/'_bin/Release'
    for path in binaries.iterdir():
        if path.suffix.lower() in ('.exe','.dll'):
            shutil.copy2(path, runtime/path.name)
    shutil.copytree(args.reference/'shaders', runtime/'vk_lod_clusters_files/shaders')
    shutil.copytree(args.nvpro/'nvshaders', runtime/'nvshaders')
    exe = runtime/'vk_lod_clusters.exe'
    manifest = dict(protocol=route['protocol'], config=str(args.config), configSha256=sha(args.config),
        reference=str(args.reference), referenceHead=run_quiet(['git','-C',str(args.reference),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
        executableSha256=sha(exe), scriptSha256=sha(__file__), replaySha256=sha(args.output/'Replay.json'),
        cache=dict(path=str(cache), bytes=cache.stat().st_size, modifiedNs=cache.stat().st_mtime_ns),
        sourceHashes={str(p.relative_to(args.reference)):sha(p) for p in (args.reference/'src/camera_path.cpp',args.reference/'src/lodclusters.cpp',args.reference/'src/main.cpp')},
        viewport=[args.width,args.height], cases=args.cases,
        notes=['Native profiler exports integer microseconds: averages/min/max, no per-frame distribution or P99.',
               'Sequences discard initial reset frames and use four-frame delayed GPU queries; memory is an endpoint snapshot.',
               'cfg cases retain .cfg plus application defaults; aligned cases explicitly change resolution scaling, DLSS, LOD error and budgets.',
               'No desktop input; isolated .ini/log/shader runtime. Screenshots only in separate verify case.'])
    write_json(args.output/'Manifest.json', manifest)
    if args.prepare_only:
        print(f'Prepared {args.output}', flush=True)
        return
    gpu_fields='timestamp,name,driver_version,utilization.gpu,memory.used,memory.total,clocks.gr,temperature.gpu,power.draw'
    for case in args.cases:
        output = args.output/case
        output.mkdir()
        # nvapp has no size CLI in this binary; its headless viewport reads Application/State.
        (runtime/'vk_lod_clusters.ini').write_text(f'[Application][State]\nSize={args.width},{args.height}\nPos=0,0\n',encoding='utf-8')
        active_route, active_paths, active_sequence = (make_route(args.config,args.distance,120)
                                                      if case=='verify' else (route,paths,sequence))
        (output/'CameraPaths.txt').write_text(active_paths,encoding='utf-8')
        # A final untabulated sequence flushes the reference logger's buffered
        # endpoint report. This also lets us distinguish complete captures from
        # a timeout while rendering if this binary stalls during destruction.
        guard=f'\nSEQUENCE "capture_flush"\n--runcamerapath 9 {active_route["phaseFrames"]}\n'
        (output/'Sequences.txt').write_text(active_sequence+guard,encoding='utf-8')
        command=[str(exe),'--configfile',str(args.config),'--headless','--headlessframes',str(len(active_route['frames'])+active_route['phaseFrames']+16),
                 '--vsync','0','--validation','0','--autosavecache','0','--loadcamerapaths',str(output/'CameraPaths.txt'),
                 '--sequencefile',str(output/'Sequences.txt'),'--sequencescreenshot','2' if case=='verify' else '0']
        if case.startswith('aligned'):
            command += ['--supersample','1','--dlss','0','--loderror','1.5','--maxgeomegabytes','1024','--maxclasmegabytes','512']
        write_json(output/'Command.json',command)
        print(f'Starting {case}: {len(active_route["frames"])} frames, {args.distance:g} units forward and back',flush=True)
        started=time.time()
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        with (output/'Gpu.csv').open('w',encoding='utf-8') as gpu, (output/'stdout.log').open('w',encoding='utf-8') as stdout, (output/'stderr.log').open('w',encoding='utf-8') as stderr:
            monitor=subprocess.Popen(['nvidia-smi',f'--query-gpu={gpu_fields}','--format=csv','-l','1'],stdout=gpu,stderr=subprocess.DEVNULL,creationflags=flags)
            try:
                process=subprocess.Popen(command,cwd=output,stdout=stdout,stderr=stderr,creationflags=flags)
                complete_time=None
                forced_cleanup=False
                try:
                    while process.poll() is None:
                        if time.time()-started > 900:
                            raise TimeoutError('Rendering did not complete within 900 seconds')
                        if complete_time is None:
                            text=(output/'stdout.log').read_text(encoding='utf-8')
                            if re.search(r'MemoryReport 9 "return_hold" = \{.*?\n\}',text,re.S):
                                parse_report(text,active_route)
                                complete_time=time.time()
                        elif time.time()-complete_time > args.exit_grace_seconds:
                            forced_cleanup=True
                            process.kill()
                            break
                        time.sleep(1)
                    exit_code=process.wait(timeout=15)
                    if exit_code and not forced_cleanup:
                        raise RuntimeError(f'Reference process failed with exit code {exit_code}')
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=15)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
        log=(output/'stdout.log').read_text(encoding='utf-8')
        log+=(output/'stderr.log').read_text(encoding='utf-8')
        reports=parse_report(log,active_route)
        if case!='verify' and any('DLSS' in t['name'] for r in reports for t in r['timers']) == case.startswith('aligned'):
            raise ValueError('Unexpected DLSS scopes: requested mode not active')
        write_json(output/'Profile.json',dict(status='capture_complete',case=case,wallSeconds=time.time()-started,
            processExitCode=exit_code, forcedCleanupAfterCapture=forced_cleanup,
            captureCompleteSeconds=complete_time-started if complete_time is not None else time.time()-started,
            routeSha256=sha(args.output/'Replay.json') if case!='verify' else None,phaseFrames=active_route['phaseFrames'],
            profiles=reports))
        print(f'{case}: {len(reports)} complete reports; forced cleanup after capture={forced_cleanup}',flush=True)
    if sha(args.config)!=manifest['configSha256'] or sha(exe)!=manifest['executableSha256']:
        raise ValueError('Inputs changed during capture')
    if cache.stat().st_size!=manifest['cache']['bytes'] or cache.stat().st_mtime_ns!=manifest['cache']['modifiedNs']:
        raise ValueError('Reference cook changed during capture')


if __name__=='__main__':
    main()
