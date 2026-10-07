"""Translate the verified Painter package into selectable Metallic LookDev scenes.

EXR originals remain untouched. Resident Metallic textures currently decode to
RGBA8, so preview PNGs are explicitly quantized; the receipt records this limit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import zlib

import numpy as np
from ValidatePainterExports import pixels, LABELS

REPO=Path(__file__).resolve().parents[2]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def png(path,data):
    if not np.isfinite(data).all() or data.min() < -1e-5 or data.max()>1.00001:
        raise ValueError('Preview texture outside UNORM range: '+str(path))
    encoded=np.rint(np.clip(data,0,1)*255).astype(np.uint8)
    height,width,channels=encoded.shape
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload))
    rows=b''.join(b'\0'+row.tobytes() for row in encoded)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,6 if channels==4 else 2,0,0,0))+
                     chunk(b'IDAT',zlib.compress(rows,9))+chunk(b'IEND',b''))
    return float(np.max(np.abs(encoded.astype(np.float32)/255-data)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=REPO/'build/MaterialValidation/PainterLookDev')
    args=parser.parse_args()
    root=args.root.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory; existing scenes are preserved')
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    validation=json.loads((root/'ExportValidation.json').read_text(encoding='utf-8'))
    if not validation['passed']:
        raise ValueError('Painter export validation must pass first')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(REPO/'Asset/Materials/OpenPBR.materialdef',out/'OpenPBR.materialdef')
    matrices=(REPO/'Source/Runtime/Render/Core/ColorSpaceMatrices.h').read_text()
    def matrix(name):
        raw=re.search(r'METALLIC_COLOR_MATRIX\('+name+r',([^)]*)\)',matrices).group(1)
        return np.array([float(x) for x in raw.split(',')]).reshape(3,3)
    ap1_to_709=matrix('kXYZToRec709')@matrix('kD60ToD65')@matrix('kAP1ToXYZ')
    def color(rgb): return (ap1_to_709@np.asarray(rgb)).tolist()+[0]
    def sample(slot,component=None):
        value={'op':'textureSample','texture':slot,'footprint':'RayCone','args':[{'op':'uv'},0]}
        return {'op':'swizzle','components':component*4,'args':[value]} if component else value
    # Radiance RGBE environment is supported by the existing environment loader.
    env=pixels(root/'NeutralStudio.exr')
    maximum=np.max(env,axis=2)
    mantissa,exponent=np.frexp(maximum)
    rgbe=np.zeros((*maximum.shape,4),np.uint8)
    rgbe[:,:,:3]=np.clip(env*np.where(maximum>1e-32,mantissa*256/np.maximum(maximum,1e-32),0)[:,:,None],0,255).astype(np.uint8)
    rgbe[:,:,3]=np.where(maximum>1e-32,exponent+128,0).astype(np.uint8)
    (out/'NeutralStudio.hdr').write_bytes(f'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {env.shape[0]} +X {env.shape[1]}\n'.encode()+rgbe.tobytes())
    template=json.loads((REPO/'Pipelines/Samples/lookdev_vbuffer.metallic_graph.json').read_text())
    catalog=[]; texture_checks=[]
    for case in manifest['cases']:
        for mode in case['modes']:
            directory=out/case['id']/mode; directory.mkdir(parents=True)
            original=root/case['id']/mode
            receipt=json.loads((original/'BuildReceipt.json').read_text())
            spp=original/'source'/f'{case["id"]}_{mode}.spp'
            if hashlib.sha256(spp.read_bytes()).hexdigest()!=receipt['sppSHA256']:
                raise ValueError('Painter project changed since validation: '+str(spp))
            model=json.loads((root/case['gltf']).read_text())
            overrides=[]
            for index,var in enumerate(case['variants']):
                p=var['parameters']; name=var['id']; resources={}
                binding=f'{case["id"]}/{mode}'
                outputs={'baseColor':color(p['base_color'])}
                params={'baseColor':[1,1,1],'metalness':p['base_metalness'],'roughness':p['specular_roughness'],
                        'ior':p['specular_ior'],'specularWeight':p['specular_weight'],
                        'transmission':p['transmission_weight'],'occlusionStrength':0,
                        'emission':[1,1,1] if p['emission_luminance']>0 else [0,0,0]}
                flags={'alphaMode':'mask' if case['id']=='M08_MaskedCard' else 'opaque',
                       'doubleSided':case['id']=='M08_MaskedCard'}
                for key,target in [('coat_weight','coatWeight'),('coat_roughness','coatRoughness'),('coat_ior','coatIOR'),
                                   ('fuzz_weight','fuzzWeight'),('fuzz_roughness','fuzzRoughness'),
                                   ('specular_roughness_anisotropy','specularAnisotropy')]:
                    if key in p: outputs[target]=p[key]
                if 'fuzz_color' in p: outputs['fuzzColor']=color(p['fuzz_color'])
                if p['transmission_weight']>0 or 'transmission_color' in p:
                    params.update(thickness=var.get('thicknessMeters',.01),attenuationDistance=p.get('transmission_depth',.01))
                    outputs['attenuationColor']=color(p.get('transmission_color',[1,1,1]))
                scale=p['emission_luminance']/1000
                outputs['emissive']=color(np.asarray(p.get('emission_color',[0,0,0]))*scale)
                def exported(channel):
                    matches=[f for f in (original/'textures').glob(name+'_'+LABELS[channel]+'_*.exr')
                             if f.stem in [name+'_'+LABELS[channel]+'_Raw',name+'_'+LABELS[channel]+'_ACEScg']]
                    if len(matches)!=1: raise ValueError((case['id'],name,channel,matches))
                    return pixels(matches[0])
                def texture(label,data,slot,color_space=None):
                    path=directory/(name+'_'+label+'.png')
                    error=png(path,data)
                    texture_checks.append({'path':str(path.relative_to(out)),'maxQuantizationError':error})
                    resource={'uri':'asset://'+binding+'/'+path.name}
                    if color_space:resource['colorSpace']=color_space
                    resources[slot]=resource
                if mode=='textured':
                    maps=case['texturedOverrides']
                    if 'specular_roughness' in maps or 'base_metalness' in maps:
                        packed=np.ones((512,512,3),np.float32)
                        if 'specular_roughness' in maps:
                            packed[:,:,1]=exported('SpecularRoughness');params['roughness']=1
                        if 'base_metalness' in maps:
                            packed[:,:,2]=exported('BaseMetalness');params['metalness']=1
                        texture('MetalRough',packed,'metallicRoughnessTexture')
                    for key,channel,target in [('coat_weight','CoatWeight','coatWeight'),('fuzz_weight','FuzzWeight','fuzzWeight')]:
                        if key in maps:
                            texture(target,np.repeat(exported(channel)[:,:,None],3,axis=2),'occlusionTexture')
                            outputs[target]=sample('occlusion','x')
                    if 'diagnostic_tangent' in maps:
                        texture('Tangent',exported('Tangent'),'occlusionTexture')
                        outputs['anisotropyTangent']=sample('occlusion')
                    if 'transmission_color' in maps:
                        texture('Absorption',exported('TransmissionColor'),'emissiveTexture','acescg')
                        outputs['attenuationColor']=sample('emissive')
                    if 'emission_color' in maps:
                        texture('Emission',exported('EmissionColor'),'emissiveTexture','acescg')
                        outputs['emissive']=[scale,scale,scale,0]
                    if 'geometry_opacity' in maps:
                        rgba=np.ones((512,512,4),np.float32);rgba[:,:,3]=exported('Opacity')
                        texture('Coverage',rgba,'baseColorTexture','lin_rec709')
                    if 'diagnostic_normal' in maps:
                        texture('Normal',exported('Normal'),'normalTexture')
                asset={'type':'Metallic.MaterialInstance','version':1,'definitionVersion':1,
                       'definition':'asset://OpenPBR.materialdef','parent':None,'parameters':params,
                       'resources':resources,'features':flags}
                write(directory/(name+'.material'),asset)
                overrides.append({'sourceId':'main','materialIndex':index,'sourceName':name,
                    'materialAsset':{'uri':'asset://'+binding+'/'+name+'.material','root':'../..'},
                    'properties':{'valueProgram':json.dumps({'version':2,'nodes':{},'outputs':outputs},separators=(',',':'))}})
            scene=directory/'Scene.gltf';write(scene,model)
            write(directory/'Scene.metallic_scene.json',{'version':3,'source':'Scene.gltf','sceneIndex':0,'nodes':[],
                  'materials':overrides,'world':{'environment':{'enabled':True,'path':'../../NeutralStudio.hdr','intensity':1.,'rotationDegrees':0,'visible':True},
                   'lighting':{'exposureEV100':0,'autoExposure':{'enabled':False,'compensation':0},'lights':[]}}})
            columns=case['columns'];rows=math.ceil(len(case['variants'])/columns)
            size=max((columns-1)*.14+.1,(rows-1)*.14+.1)
            distance=size*.5/math.tan(math.radians(25))*1.3
            camera={'eye':[0,size*.20,distance],'center':[0,0,0],'up':[0,1,0],'fovDegrees':50,
                    'projection':'perspective','znear':.001,'zfar':100}
            graph=json.loads(json.dumps(template));graph['name']='Painter / '+case['id']+' / '+mode
            for node in graph['nodes']:
                props=node['properties']
                if 'path' in props:props['path']=scene.as_posix()
                if 'camera' in props:props['camera']=camera
                if node['name']=='Deferred':props.update(samples=64,materialBinning=False)
            graph_path=directory/'LookDev.metallic_graph.json';write(graph_path,graph)
            catalog.append({'id':'painter-'+case['id']+'-'+mode,'name':case['id']+' / '+mode,
                            'scenePath':scene.as_posix(),'graphPath':graph_path.as_posix(),
                            'environment':(out/'NeutralStudio.hdr').as_posix(),
                            'description':case['purpose']+' Preview PNGs use RGBA8; original EXRs retained in Painter package.'})
    write(out/'Catalog.json',{'version':1,'samples':catalog})
    write(out/'ImportReceipt.json',{'source':str(root),'projects':len(catalog),'textures':texture_checks,
          'colorContract':'ACEScg colors; Value constants converted to linear Rec709 without clipping. Texture colorSpace is explicit.',
          'limitations':['PNG preview textures use 8-bit UNORM; original EXR data are not overwritten.',
                         'Emission luminance divided by 1000 to match Painter shader convention.',
                         'Glass uses Metallic volume transport; Painter absolute scale remains uncalibrated.',
                         'Comparison framing is fitted to source metre-scale mesh, not copied from normalized Painter camera.']})
    print('Generated',len(catalog),'scenes and',len(texture_checks),'textures in',out)


if __name__=='__main__':main()
