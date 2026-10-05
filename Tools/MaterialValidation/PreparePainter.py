"""Generate deterministic, metre-scale Painter/OpenPBR diagnostic inputs.

No Painter, third-party Python package, or renderer is required for preparation.
The output is authored input, never a rendered reference or a passing BSDF test.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


BASE = dict(base_weight=1.0, base_color=[0.18] * 3, base_metalness=0.0,
            specular_weight=1.0, specular_roughness=0.35, specular_ior=1.5,
            specular_roughness_anisotropy=0.0, coat_weight=0.0,
            fuzz_weight=0.0, transmission_weight=0.0, subsurface_weight=0.0,
            thin_film_weight=0.0, emission_luminance=0.0, geometry_opacity=1.0)
BASE.update(diagnostic_normal=[.5,.5,1.],diagnostic_coat_normal=[.5,.5,1.],coat_color=[1.,1.,1.])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def png(path, width, height, pixel):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            values = pixel(x / (width - 1), y / (height - 1))
            rows.extend(struct.pack('>3H', *(round(max(0, min(1, c)) * 65535) for c in values)))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 16, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(rows, 9)) + chunk(b'IEND', b''))


def patterns(directory, size):
    directory.mkdir(parents=True, exist_ok=True)
    checker = lambda u, v: float((min(int(u * 8), 7) + min(int(v * 8), 7)) % 2)
    def normal(u, v):
        x, y = 0.35 * math.sin(2 * math.pi * u * 4), 0.35 * math.sin(2 * math.pi * v * 2)
        return (0.5 + x / 2, 0.5 + y / 2, 0.5 + math.sqrt(1 - x*x - y*y) / 2)
    definitions = {
        'roughness_gradient': (lambda u, v: [0.05 + 0.75*u]*3, 'data'),
        'metalness_split': (lambda u, v: [float(u >= 0.5)]*3, 'data'),
        'coat_mask': (lambda u, v: [checker(u, v)]*3, 'data'),
        'fuzz_mask': (lambda u, v: [u]*3, 'data'),
        'anisotropy_turns': (lambda u, v: [0 if u < 1/3 else 0.125 if u < 2/3 else 0.25]*3, 'data'),
        'tangent_directions': (lambda u, v: [0.5+0.5*math.cos((0 if u < 1/3 else math.pi/4 if u < 2/3 else math.pi/2)),
                                            0.5+0.5*math.sin((0 if u < 1/3 else math.pi/4 if u < 2/3 else math.pi/2)),1.0], 'data'),
        'emission_checker': (lambda u, v: ([1, 0.03, 0.01] if checker(u, v) else [0, 0, 0]), 'ACEScg'),
        'transmission_tint': (lambda u, v: [0.2 + 0.6*u, 0.7, 0.85], 'ACEScg'),
        'opacity_thresholds': (lambda u, v: [([0, 0.25, 0.499, 0.501, 0.75, 1][min(int(u*6), 5)]
                                                if v < 0.5 else checker(u, v))]*3, 'data'),
        'normal_opengl': (normal, 'normal_opengl'),
        'base_uv': (lambda u, v: [0.08 + 0.55*u, 0.08 + 0.4*v, 0.12], 'ACEScg'),
    }
    result = {}
    for name, (fn, space) in definitions.items():
        png(directory / f'{name}.png', size, size, fn)
        result[name] = dict(path=f'textures/{name}.png', colorSpace=space, bitDepth=16,
                            channels='RGB', encoding='linear UNORM; no embedded color profile')
    return result


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def geometry(kind, center, thickness=0.01, uv_turns=0.0):
    positions, normals, uvs, triangles = [], [], [], []
    def vertex(p, n, uv):
        positions.append(tuple(p[i]+center[i] for i in range(3)))
        normals.append(n)
        # Rotate UVs to rotate the actual derived tangent, not an invented OpenPBR scalar.
        a = -2*math.pi*uv_turns
        u, v = uv[0]-0.5, uv[1]-0.5
        uvs.append((0.5+math.cos(a)*u-math.sin(a)*v, 0.5+math.sin(a)*u+math.cos(a)*v))
    if kind == 'sphere':
        lat, lon = 24, 48
        for j in range(lat+1):
            t = math.pi*j/lat
            for i in range(lon+1):
                p = 2*math.pi*i/lon
                n = (math.sin(t)*math.cos(p), math.cos(t), math.sin(t)*math.sin(p))
                vertex(tuple(0.05*x for x in n), n, (i/lon, 1-j/lat))
        for j in range(lat):
            for i in range(lon):
                a, b = j*(lon+1)+i, (j+1)*(lon+1)+i
                if j > 0:
                    triangles.append((a, b, a+1))
                if j < lat-1:
                    triangles.append((a+1, b, b+1))
    else:
        # Independent face UVs/normals, geometrically watertight for solid glass.
        faces = [((0,0,1), (-.05,-.05,thickness/2), (.1,0,0), (0,.1,0))]
        if kind == 'back_card':
            faces = [((0,0,-1), (.05,-.05,0), (-.1,0,0), (0,.1,0))]
        elif kind == 'box':
            faces += [((0,0,-1), (.05,-.05,-thickness/2), (-.1,0,0), (0,.1,0)),
                      ((1,0,0), (.05,-.05,thickness/2), (0,0,-thickness), (0,.1,0)),
                      ((-1,0,0), (-.05,-.05,-thickness/2), (0,0,thickness), (0,.1,0)),
                      ((0,1,0), (-.05,.05,thickness/2), (.1,0,0), (0,0,-thickness)),
                      ((0,-1,0), (-.05,-.05,-thickness/2), (.1,0,0), (0,0,thickness))]
        for n, origin, du, dv in faces:
            start = len(positions)
            for u, v in [(0,0),(1,0),(1,1),(0,1)]:
                vertex(tuple(origin[k]+u*du[k]+v*dv[k] for k in range(3)), n, (u,v))
            triangles.extend([(start,start+1,start+2),(start,start+2,start+3)])
    for i, (a,b,c) in enumerate(triangles):
        if dot(cross(sub(positions[b], positions[a]), sub(positions[c], positions[a])), normals[a]) < 0:
            triangles[i] = a,c,b
    return positions, normals, uvs, triangles


def write_mesh(directory, variants, columns):
    directory.mkdir(parents=True, exist_ok=True)
    obj, mtl, binary, views, accessors, primitives = ['mtllib Validation.mtl'], [], bytearray(), [], [], []
    offset = 1
    meshes = []
    def accessor(values, code, components, target):
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        flat = [x for v in values for x in (v if isinstance(v, (tuple,list)) else [v])]
        binary.extend(struct.pack('<' + code*len(flat), *flat))
        views.append(dict(buffer=0, byteOffset=start, byteLength=len(binary)-start, target=target))
        spec = dict(bufferView=len(views)-1, componentType=5126 if code=='f' else 5125,
                    count=len(values), type={1:'SCALAR',2:'VEC2',3:'VEC3'}[components])
        if components == 3:
            spec.update(min=[min(v[k] for v in values) for k in range(3)], max=[max(v[k] for v in values) for k in range(3)])
        accessors.append(spec)
        return len(accessors)-1
    rows = math.ceil(len(variants)/columns)
    for index, var in enumerate(variants):
        center = ((index%columns-(columns-1)/2)*.14, ((rows-1)/2-index//columns)*.14, 0)
        p,n,uv,tris = geometry(var.get('geometry','sphere'), center, var.get('thicknessMeters',.01), var.get('uvRotationTurns',0))
        meshes.append((p,n,uv,tris,var))
        obj += [f'o {var["id"]}', f'usemtl {var["id"]}']
        obj += ['v '+' '.join(f'{x:.9g}' for x in q) for q in p]
        obj += ['vt '+' '.join(f'{x:.9g}' for x in q) for q in uv]
        obj += ['vn '+' '.join(f'{x:.9g}' for x in q) for q in n]
        obj += ['f '+' '.join(f'{v+offset}/{v+offset}/{v+offset}' for v in t) for t in tris]
        offset += len(p)
        mtl += [f'newmtl {var["id"]}', 'Kd 0.18 0.18 0.18', 'd 1', '']
        primitives.append(dict(attributes={'POSITION':accessor(p,'f',3,34962), 'NORMAL':accessor(n,'f',3,34962),
                                            'TEXCOORD_0':accessor([(u,1-v) for u,v in uv],'f',2,34962)},
                               indices=accessor([v for t in tris for v in t],'I',1,34963), material=index))
    (directory/'Validation.obj').write_text('\n'.join(obj)+'\n',encoding='utf-8')
    (directory/'Validation.mtl').write_text('\n'.join(mtl),encoding='utf-8')
    # glTF is a geometry transport counterpart, not an OpenPBR material translation.
    gltf = dict(asset={'version':'2.0','generator':'Metallic Painter validation input generator'},
                buffers=[dict(byteLength=len(binary),uri='data:application/octet-stream;base64,'+base64.b64encode(binary).decode())],
                bufferViews=views,accessors=accessors,materials=[dict(name=v['id'],extras={'openpbrReference':v['parameters']}) for v in variants],
                meshes=[dict(primitives=primitives)],nodes=[dict(mesh=0)],scenes=[dict(nodes=[0])],scene=0)
    write_json(directory/'GeometryOnly.gltf',gltf)
    return meshes


def variant(name, updates=None, **kwargs):
    return dict(id=name, parameters=BASE | (updates or {}), **kwargs)


def cases():
    return [
        ('M01_NeutralDielectric',[variant(f'R{r:03d}',{'specular_roughness':r/100}) for r in [5,20,50,80]],
         {'specular_roughness':'roughness_gradient'}, 'Neutral RGB; roughness changes highlight width, not hue.'),
        ('M02_SaturatedMetal',[variant(f'R{r:03d}',{'base_metalness':1.,'base_color':[.75,.22,.06],'specular_roughness':r/100}) for r in [10,35,70]],
         {'base_metalness':'metalness_split'}, 'Metal side has colored specular; dielectric side retains diffuse.'),
        ('M03_CoatedPaint',[variant(n,{'base_color':[.5,.025,.015],'coat_weight':w,'coat_roughness':r,'coat_ior':1.5})
                             for n,w,r in [('NoCoat',0.,.03),('Coat003',1.,.03),('Coat025',1.,.25)]],
         {'coat_weight':'coat_mask'}, 'Separate coat/base lobes and attenuation; not equivalent to base roughness.'),
        ('M04_BrushedMetal',[variant(f'Tangent{deg:03d}',{'base_metalness':1.,'base_color':[.65]*3,'specular_roughness':.3,
                                    'specular_roughness_anisotropy':.8,'diagnostic_tangent':[1.,.5,1.]},uvRotationTurns=deg/360) for deg in [0,45,90]],
         {'diagnostic_tangent':'tangent_directions'}, 'Uniform variants rotate mesh UV/tangent; texture encodes tangent XY, not an OpenPBR scalar angle.'),
        ('M05_Fuzz',[variant(f'Fuzz{w:03d}',{'base_color':[.025,.012,.018],'specular_roughness':.8,'fuzz_weight':w/100,
                                           'fuzz_color':[.3,.06,.12],'fuzz_roughness':.5}) for w in [0,50,100]],
         {'fuzz_weight':'fuzz_mask'}, 'Grazing response and base energy; do not substitute glTF sheen.'),
        ('M06_SolidGlass',[variant(f'T{mm:02d}mm_R{r:02d}',{'base_color':[1.]*3,'transmission_weight':1.,'specular_roughness':r/100,
                                  'transmission_color':[.35,.7,.85],'transmission_depth':.01,'geometry_thin_walled':False},
                                  geometry='box',thicknessMeters=mm/1000) for mm in [10,50] for r in [0,15]],
         {'transmission_color':'transmission_tint'}, 'Beer-Lambert interior attenuation; compare PT, not unsupported VBuffer volume transport.'),
        ('M07_Emission',[variant(f'Nits{v:03d}',{'base_color':[0.]*3,'emission_color':[1.,.03,.01],'emission_luminance':float(v)}) for v in [1,10,100]],
         {'emission_color':'emission_checker'}, 'OpenPBR luminance is nits; derive radiance using model luminance normalization, not direct RGB multiplication.'),
        ('M08_MaskedCard',[variant(n,{'base_color':[.12,.35,.08]},geometry=g,
                                   renderState={'alphaMode':'MASK','alphaCutoff':.5,'doubleSided':True})
                           for n,g in [('Front','card'),('Back','back_card')]],
         {'geometry_opacity':'opacity_thresholds','diagnostic_normal':'normal_opengl'}, 'Uniform is solid control; textured threshold bands and holes must affect winner, RT and shadow.'),
        ('H01_HDREmission',[variant(f'Nits{v:06d}',{'base_color':[0.]*3,'emission_color':[1.,.03,.01],
                                    'emission_luminance':float(v)}) for v in [1000,10000,100000]],
         {'emission_color':'emission_checker'}, 'Supplement to M07: Painter shader divides luminance by 1000; these values exercise emitted RGB above one. Display screenshots are not linear HDR measurements.'),
    ]


def sweeps():
    definitions = [
        ('S01_Metal_Roughness','base_metalness',[0,.25,.5,.75,1],'specular_roughness',[.05,.2,.4,.6,.8],{}),
        ('S02_IOR_Roughness','specular_ior',[1.,1.25,1.5,2.,2.5],'specular_roughness',[.05,.2,.4,.6,.8],{}),
        ('S03_Coat','coat_weight',[0,.25,.5,.75,1],'coat_roughness',[.03,.1,.25,.5,.8],{'base_color':[.5,.025,.015]}),
        ('S04_Anisotropy','specular_roughness_anisotropy',[0,.2,.4,.6,.8],'uvRotationTurns',[0,.125,.25,.375,.5],{'base_metalness':1.,'diagnostic_tangent':[1.,.5,1.]}),
        ('S05_Transmission','transmission_weight',[0,.25,.5,.75,1],'specular_roughness',[.0,.05,.15,.3,.5],{'transmission_color':[.35,.7,.85],'transmission_depth':.01,'geometry_thin_walled':False}),
        ('S06_Fuzz','fuzz_weight',[0,.25,.5,.75,1],'fuzz_roughness',[.1,.25,.5,.75,1],{'base_color':[.025,.012,.018],'fuzz_color':[.3,.06,.12]}),
    ]
    for name,y,ys,x,xs,base in definitions:
        values=[]
        for row,yv in enumerate(ys):
            for col,xv in enumerate(xs):
                extra = {'uvRotationTurns':xv} if x=='uvRotationTurns' else {}
                params=base | {y:yv} | ({} if extra else {x:xv})
                if name=='S05_Transmission':
                    extra |= {'geometry':'box','thicknessMeters':.01}
                values.append(variant(f'Row{row}_Col{col}',params,**extra))
        yield name,values,{},f'Rows top-to-bottom: {y}={ys}; columns left-to-right: {x}={xs}.'


def prepare(root, size):
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new empty output directory; existing evidence is not overwritten.')
    root.mkdir(parents=True,exist_ok=True)
    textures=patterns(root/'textures',size)
    manifest=dict(schemaVersion=1,status='prepared_inputs_not_painter_validated',model='OpenPBR',modelVersion='1.1',
                  workingColorSpace='ACEScg',meshUnit='meter',painterMeshUnitScale=100.0,
                  normalConvention='OpenGL +Y',textures=textures,cases=[])
    checks=[]
    for name,variants,maps,purpose in cases()+list(sweeps()):
        columns=5 if name.startswith('S') else len(variants)
        meshes=write_mesh(root/name/'mesh',variants,columns)
        for p,n,uv,tris,v in meshes:
            for a,b,c in tris:
                area=cross(sub(p[b],p[a]),sub(p[c],p[a]))
                assert dot(area,area)>1e-20 and dot(area,n[a])>0, (name,v['id'],'bad winding')
            if v.get('geometry')=='box':
                edges={}
                for t in tris:
                    for a,b in zip(t,t[1:]+t[:1]):
                        key=tuple(sorted((tuple(round(x,8) for x in p[a]),tuple(round(x,8) for x in p[b]))))
                        edges[key]=edges.get(key,0)+1
                assert all(count==2 for count in edges.values()), 'Non-watertight solid'
                assert abs(max(q[2] for q in p)-min(q[2] for q in p)-v['thicknessMeters'])<1e-9
            checks.append(dict(case=name,material=v['id'],vertices=len(p),triangles=len(tris),winding='outward'))
        entry=dict(id=name,mesh=f'{name}/mesh/Validation.obj',gltf=f'{name}/mesh/GeometryOnly.gltf',columns=columns,
                   variants=variants,texturedOverrides=maps,purpose=purpose,modes=['uniform','textured'] if maps else ['uniform'])
        manifest['cases'].append(entry)
        write_json(root/name/'Reference.json',dict(model='OpenPBR',version='1.1',workingColorSpace='ACEScg',meshUnit='meter',
                   status='authored_input_only',comparison={'Painter':'appearance_reference','Metallic':'not_run'},**entry))
    write_json(root/'Manifest.json',manifest)
    hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}
    write_json(root/'PreparationChecks.json',dict(status='input_geometry_checks_passed',checks=checks,sha256=hashes,
                  painterExecuted=False,rendererExecuted=False))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--texture-size',type=int,default=512,choices=[256,512,1024])
    args=parser.parse_args()
    result=prepare(args.output.resolve(),args.texture_size)
    print(json.dumps(dict(output=str(args.output.resolve()),cases=len(result['cases']),projects=sum(len(c['modes']) for c in result['cases']),
                          materials=sum(len(c['variants']) for c in result['cases']),status=result['status']),indent=2))
