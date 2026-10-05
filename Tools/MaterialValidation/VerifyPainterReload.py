"""Reopen each generated SPP and verify public API values and embedded resources."""
import argparse
import json
from pathlib import Path

from PainterRemote import call, wait_ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    results=[]
    for case in manifest['cases']:
        for mode in case['modes']:
            request={'root':str(root),'case':case['id'],'mode':mode}
            call(dict(request,action='open'))
            wait_ready(60041)
            result=call(dict(request,action='verify_reloaded'))
            results.append(result)
            (root/'ReloadValidation.json').write_text(json.dumps({
                'passed':False,'complete':False,'projects':results},indent=2),encoding='utf-8')
            call(dict(request,action='close'))
            print('PASS',case['id'],mode,result['sourceChecks'],flush=True)
    (root/'ReloadValidation.json').write_text(json.dumps({
        'passed':True,'complete':True,'projects':results},indent=2),encoding='utf-8')
    print('Verified',len(results),'saved projects',flush=True)


if __name__=='__main__':
    main()
