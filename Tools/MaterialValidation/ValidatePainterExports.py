"""Check actual Painter EXR data against authored numeric inputs (not BSDF images)."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import OpenEXR


LABELS = {'BaseColor':'Base_color','BaseWeight':'Base_weight','BaseMetalness':'Base_metalness',
          'SpecularWeight':'Specular_weight','SpecularRoughness':'Specular_roughness',
          'SpecularRoughnessAnisotropy':'Specular_anisotropy','CoatWeight':'Coat_weight',
          'CoatColor':'Coat_color','CoatRoughness':'Coat_roughness','CoatNormal':'Coat_normal',
          'FuzzWeight':'Fuzz_weight','FuzzRoughness':'Fuzz_roughness','FuzzColor':'Fuzz_color',
          'TransmissionWeight':'Transmission_weight','TransmissionColor':'Transmission_color',
          'SubsurfaceWeight':'Subsurface_weight','ThinFilmWeight':'Thin-film_weight',
          'EmissionColor':'Emission_color','Opacity':'Opacity','Normal':'Normal','Tangent':'Tangent'}


def pixels(path):
    with OpenEXR.File(str(path)) as image:
        channels=image.channels()
        if 'RGB' in channels:
            return channels['RGB'].pixels.copy()
        return next(iter(channels.values())).pixels.copy()


def input_png(path):
    """Decode the unfiltered RGB16 PNG format emitted by PreparePainter.py."""
    data=path.read_bytes()
    if data[:8]!=b'\x89PNG\r\n\x1a\n':
        raise ValueError('Not PNG')
    offset=8
    compressed=[]
    while offset<len(data):
        count=struct.unpack_from('>I',data,offset)[0]
        kind=data[offset+4:offset+8]
        payload=data[offset+8:offset+8+count]
        if kind==b'IHDR':
            width,height,bits,color,compression,filter_method,interlace=struct.unpack('>IIBBBBB',payload)
            if (bits,color,compression,filter_method,interlace)!=(16,2,0,0,0):
                raise ValueError('Unexpected generated PNG format')
        if kind==b'IDAT':
            compressed.append(payload)
        offset+=12+count
    rows=np.frombuffer(zlib.decompress(b''.join(compressed)),dtype=np.uint8).reshape(height,1+width*6)
    if rows[:,0].any():
        raise ValueError('Expected generated PNG with filter type zero')
    return np.frombuffer(rows[:,1:].tobytes(),dtype='>u2').reshape(height,width,3).astype(np.float32)/65535.


def validate(root):
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    report={'scope':'Painter authoring, shader readback, EXR data; not renderer equivalence',
            'projects':[],'failures':[],'files':0,'constantChecks':0,'maxConstantError':0.0,
            'textureChecks':0,'maxTextureError':0.0}
    for case in manifest['cases']:
        for mode in case['modes']:
            directory=root/case['id']/mode
            receipt=json.loads((directory/'AuthoringReceipt.json').read_text(encoding='utf-8'))
            build=json.loads((directory/'BuildReceipt.json').read_text(encoding='utf-8'))
            spp=directory/'source'/f'{case["id"]}_{mode}.spp'
            if hashlib.sha256(spp.read_bytes()).hexdigest()!=build['sppSHA256']:
                report['failures'].append(str(spp)+': saved project changed after receipt')
            files=list((directory/'textures').glob('*.exr'))
            for file in files:
                data=pixels(file)
                if not np.isfinite(data).all():
                    report['failures'].append(str(file)+': nonfinite')
            report['files']+=len(files)
            for variant in case['variants']:
                for key,value in variant['parameters'].items():
                    binding=receipt['profile']['bindings'][key]
                    if 'shaderParameter' in binding:
                        shader=receipt['shaderReadback']['shaders']['Validation_'+variant['id']]
                        flat={k:v for g in shader['parameters'].values() for k,v in g.items()}
                        expected=float(value)*binding.get('scale',1.)
                        if abs(float(flat[binding['shaderParameter']])-expected)>1e-5:
                            report['failures'].append(case['id']+'/'+key+': shader mismatch')
                        continue
                    if mode=='textured' and key in case['texturedOverrides']:
                        continue
                    label=LABELS[binding['channel']]
                    matches=list((directory/'textures').glob(variant['id']+'_'+label+'_*.exr'))
                    matches=[p for p in matches if p.stem in [variant['id']+'_'+label+'_Raw',variant['id']+'_'+label+'_ACEScg']]
                    if len(matches)!=1:
                        report['failures'].append(f'{case["id"]}/{variant["id"]}/{key}: missing/ambiguous map')
                        continue
                    data=pixels(matches[0])
                    # UV rotations can leave unused corners; test the center region
                    # present in each generated UV chart, not dilated outside pixels.
                    data=data[data.shape[0]//3:2*data.shape[0]//3,data.shape[1]//3:2*data.shape[1]//3]
                    expected=np.asarray(value)*binding.get('scale',1.)
                    error=float(np.max(np.abs(data-expected)))
                    report['constantChecks']+=1
                    report['maxConstantError']=max(report['maxConstantError'],error)
                    if error>2e-5:
                        report['failures'].append(f'{matches[0].name}: constant error={error}')
                if mode=='textured':
                    for key,texture in case['texturedOverrides'].items():
                        binding=receipt['profile']['bindings'][key]
                        label=LABELS[binding['channel']]
                        candidates=[p for p in files if p.stem in [
                            variant['id']+'_'+label+'_Raw',variant['id']+'_'+label+'_ACEScg']]
                        if len(candidates)!=1:
                            report['failures'].append(variant['id']+'/'+key+': missing texture export')
                            continue
                        actual=pixels(candidates[0])
                        expected=input_png(root/manifest['textures'][texture]['path'])
                        if actual.ndim==2:
                            expected=expected[:,:,0]
                        if actual.shape!=expected.shape:
                            report['failures'].append(variant['id']+'/'+key+': texture dimensions differ')
                            continue
                        # The central UV square is covered even for rotated charts.
                        region=(slice(actual.shape[0]//4,3*actual.shape[0]//4),
                                slice(actual.shape[1]//4,3*actual.shape[1]//4))
                        error=float(np.max(np.abs(actual[region]-expected[region])))
                        report['textureChecks']+=1
                        report['maxTextureError']=max(report['maxTextureError'],error)
                        if error>1e-4:
                            report['failures'].append(candidates[0].name+': input texture mismatch='+str(error))
            report['projects'].append({'case':case['id'],'mode':mode,'exrCount':len(files),'spp':str(spp)})
    report['passed']=not report['failures']
    (root/'ExportValidation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    result=validate(args.root.resolve())
    print(json.dumps({k:v for k,v in result.items() if k!='projects'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
