"""Build the prepared suite in a running Painter 12.1+ via its official Python API."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PainterRemote import call, wait_ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--iray-glass',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    profile=json.loads((root/'PainterProfile.json').read_text(encoding='utf-8'))
    summary=[]
    for case in manifest['cases']:
        if args.only and case['id'] not in args.only:
            continue
        for mode in case['modes']:
            directory=root/case['id']/mode
            done=directory/'BuildReceipt.json'
            if done.exists():
                previous=json.loads(done.read_text())
                saved=directory/'source'/f'{case["id"]}_{mode}.spp'
                if not saved.is_file() or hashlib.sha256(saved.read_bytes()).hexdigest()!=previous['sppSHA256']:
                    raise RuntimeError('Existing project differs from its receipt: '+str(saved))
                summary.append(previous)
                print('EXISTS',case['id'],mode,flush=True)
                continue
            request={'root':str(root),'case':case['id'],'mode':mode,'profile':profile}
            print('START',case['id'],mode,flush=True)
            for action in ['create','author','save','export']:
                call(dict(request,action=action))
                wait_ready(60041)
                print('  '+action,flush=True)
            view=call(dict(request,action='reference'))
            wait_ready(60041)
            time.sleep(2)
            capture=call(dict(request,action='capture'))
            references=[capture]
            if args.iray_glass and case['id']=='M06_SolidGlass':
                call(dict(request,action='reference',referenceMode='Visualisation'))
                time.sleep(15)
                references.append(call(dict(request,action='capture')))
                call(dict(request,action='reference',referenceMode='Edition'))
                wait_ready(60041)
            call(dict(request,action='save'))
            saved=directory/'source'/f'{case["id"]}_{mode}.spp'
            receipt={'case':case['id'],'mode':mode,'status':'created_saved_exported',
                     'spp':str(saved),'sppSHA256':hashlib.sha256(saved.read_bytes()).hexdigest(),
                     'camera':view['camera'],'references':references,
                     'renderCorrectness':'requires image review and matched Metallic rendering'}
            done.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            summary.append(receipt)
            call(dict(request,action='close'))
            (root/'BuildProgress.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print('DONE',case['id'],mode,flush=True)
    print('Finished',len(summary),'projects',flush=True)


if __name__=='__main__':
    main()
