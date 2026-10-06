"""Loopback client for Adobe's --enable-remote-scripting /run.json API.

No UI automation, process termination, plug-in installation, or remote host access.
"""
import argparse
import base64
import http.client
import json
from pathlib import Path
import time


def call(request, port=60041):
    script=Path(__file__).with_name('PainterAPI.py').resolve().as_posix()
    expression=f"__import__('json').dumps(__import__('runpy').run_path({script!r})['safe_dispatch']({request!r}))"
    payload=json.dumps({'python':base64.b64encode(expression.encode()).decode()})
    connection=http.client.HTTPConnection('localhost',port,timeout=300)
    try:
        connection.request('POST','/run.json',payload,{'Content-Type':'application/json','Accept':'application/json'})
        response=connection.getresponse()
        text=response.read().decode('utf-8').strip()
        if response.status!=200:
            raise RuntimeError(f'Painter HTTP {response.status}: {text[:1000]}')
    finally:
        connection.close()
    data=json.loads(text)
    if isinstance(data,str):
        data=json.loads(data)
    if not isinstance(data,dict) or not data.get('ok'):
        raise RuntimeError(str(data))
    return data['result']


def wait_ready(port):
    end=time.monotonic()+180
    while time.monotonic()<end:
        if call({'action':'ready'},port)['ready']:
            return
        time.sleep(.5)
    raise TimeoutError('Painter did not enter an idle edition state; project retained for diagnosis')


def make_profile(root, output):
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    keys=set()
    for case in manifest['cases']:
        keys.update(case['texturedOverrides'])
        for var in case['variants']:
            keys.update(var['parameters'])
    profile=dict(status='draft_requires_Painter_12_1_inventory',verifiedPainterVersion=None,
                 ocioConfig=None,acescgColorSpace='ACEScg',templates={'default':None},
                 bindings={k:None for k in sorted(keys)},coverageEvidence=None,
                 exportPreset=None,exportColorEvidence=None,
                 bindingExample={'channel':'exact ChannelType member from inventory','scale':1.0,
                                 'evidence':'installed OpenPBR shader/template source and parameter semantics'})
    with output.open('x',encoding='utf-8') as stream:
        json.dump(profile,stream,indent=2)
    return {'status':profile['status'],'path':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inventory','profile','probe','build'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--case',default='M01_NeutralDielectric')
    parser.add_argument('--mode',choices=['uniform','textured'],default='uniform')
    parser.add_argument('--template',type=Path)
    parser.add_argument('--ocio',type=Path)
    parser.add_argument('--port',type=int,default=60041)
    parser.add_argument('--close',action='store_true',help='Close only the generated, saved project after success')
    args=parser.parse_args()
    if args.action=='profile':
        if not args.root or not args.output:
            parser.error('profile needs --root and --output')
        result=make_profile(args.root.resolve(),args.output.resolve())
    elif args.action=='inventory':
        result=call({'action':'inventory'},args.port)
        if args.output:
            with args.output.open('x',encoding='utf-8') as f:
                json.dump(result,f,indent=2)
    else:
        if not args.root:
            parser.error('--root is required')
        request={'root':str(args.root.resolve()),'case':args.case,'mode':args.mode}
        if args.action=='probe':
            if not args.template or not args.ocio:
                parser.error('probe needs --template and --ocio')
            call(dict(request,action='probe',template=str(args.template.resolve()),ocio=str(args.ocio.resolve())),args.port)
            wait_ready(args.port)
            result=call({'action':'inventory'},args.port)
            if args.output:
                with args.output.open('x',encoding='utf-8') as f:
                    json.dump(result,f,indent=2)
        else:
            if not args.profile:
                parser.error('build needs a verified --profile')
            request['profile']=json.loads(args.profile.read_text(encoding='utf-8'))
            for action in ['create','author','save','export']:
                result=call(dict(request,action=action),args.port)
                print(json.dumps({'step':action,'result':result},ensure_ascii=False),flush=True)
                wait_ready(args.port)
            if args.close:
                result=call(dict(request,action='close'),args.port)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
