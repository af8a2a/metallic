"""Build White Studio 02 lighting and photographic ColorChecker references.

The photograph is radiance, never material albedo. Original files remain intact.
Requires numpy, OpenEXR and Pillow; generates optional local LookDev assets.
"""
import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFont
from PreparePainter import geometry

REPO = Path(__file__).resolve().parents[2]
PATCH_NAMES = ['Dark skin','Light skin','Blue sky','Foliage','Blue flower','Bluish green',
               'Orange','Purplish blue','Moderate red','Purple','Yellow green','Orange yellow',
               'Blue','Green','Red','Yellow','Magenta','Cyan',
               'White','Neutral 8','Neutral 6.5','Neutral 5','Neutral 3.5','Black']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def rgb(path):
    with OpenEXR.File(str(path)) as image:
        channels = image.channels()
        data = channels['RGB' if 'RGB' in channels else 'RGBA'].pixels[:, :, :3].copy()
    if not np.isfinite(data).all() or data.min() < 0:
        raise ValueError('Expected finite non-negative linear radiance: ' + str(path))
    return data


def srgb(values):
    values = np.clip(values, 0, 1)
    return np.where(values <= .0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - .055)


def radiance(path, data):
    maximum = data.max(axis=2)
    mantissa, exponent = np.frexp(maximum)
    encoded = np.zeros((*maximum.shape, 4), np.uint8)
    scale = np.where(maximum > 1e-32, mantissa * 256 / np.maximum(maximum, 1e-32), 0)
    encoded[:, :, :3] = np.clip(data * scale[:, :, None], 0, 255).astype(np.uint8)
    encoded[:, :, 3] = np.where(maximum > 1e-32, exponent + 128, 0).astype(np.uint8)
    # Flat RGBE is supported by stb's Radiance decoder; no LDR clipping or gamma.
    path.write_bytes(f'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {data.shape[0]} +X {data.shape[1]}\n'.encode() + encoded.tobytes())
    decoded = encoded[:, :, :3].astype(np.float32) * np.exp2(encoded[:, :, 3].astype(np.float32)-136)[:, :, None]
    return {'maximumAbsoluteError': float(np.max(np.abs(decoded-data))),
            'relativeRMSError': float(np.sqrt(np.mean((decoded-data)**2)/np.mean(data**2))),
            'sourceMax': float(data.max()), 'decodedMax': float(decoded.max())}


class Mesh:
    def __init__(self):
        self.binary = bytearray()
        self.views, self.accessors, self.primitives, self.materials = [], [], [], []
        self.images, self.textures = [], []

    def accessor(self, values, code, count, target):
        start = len(self.binary)
        values = np.asarray(values, dtype='<f4' if code == 'f' else '<u4')
        self.binary.extend(values.tobytes())
        self.views.append(dict(buffer=0, byteOffset=start, byteLength=len(self.binary)-start, target=target))
        spec = dict(bufferView=len(self.views)-1, componentType=5126 if code == 'f' else 5125,
                    count=len(values), type={1:'SCALAR',2:'VEC2',3:'VEC3'}[count])
        if count == 3:
            spec.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        self.accessors.append(spec)
        return len(self.accessors)-1

    def add(self, name, center, scale, color, metal=0., rough=.5, kind='sphere', texture=None, unlit=False, extensions=None):
        p,n,uv,t = geometry(kind, (0,0,0))
        p = (np.asarray(p)*np.asarray(scale)+np.asarray(center)).tolist()
        material = dict(name=name, pbrMetallicRoughness=dict(baseColorFactor=[*color,1], metallicFactor=metal, roughnessFactor=rough))
        if texture:
            self.images.append({'uri':texture})
            self.textures.append({'source':len(self.images)-1})
            material['pbrMetallicRoughness']['baseColorTexture'] = {'index':len(self.textures)-1}
        material['extensions'] = extensions or {}
        if unlit:
            material['extensions']['KHR_materials_unlit'] = {}
        index = len(self.materials)
        self.materials.append(material)
        self.primitives.append(dict(attributes={'POSITION':self.accessor(p,'f',3,34962),
            'NORMAL':self.accessor(n,'f',3,34962),'TEXCOORD_0':self.accessor([(u,1-v) for u,v in uv],'f',2,34962)},
            indices=self.accessor(np.asarray(t).flatten(),'I',1,34963),material=index))
        return index

    def save(self, path):
        write(path, dict(asset={'version':'2.0','generator':'Metallic White Studio LookDev'},
            extensionsUsed=sorted({k for m in self.materials for k in m['extensions']}),
            buffers=[{'byteLength':len(self.binary),'uri':'data:application/octet-stream;base64,'+base64.b64encode(self.binary).decode()}],
            bufferViews=self.views,accessors=self.accessors,materials=self.materials,images=self.images,textures=self.textures,
            meshes=[{'primitives':self.primitives}],nodes=[{'mesh':0}],scenes=[{'nodes':[0]}],scene=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdri', type=Path, required=True)
    parser.add_argument('--chart-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=REPO/'build/MaterialValidation/WhiteStudio02')
    args = parser.parse_args()
    out = args.output.resolve()
    if (out/'Catalog.json').exists():
        raise FileExistsError('Choose a fresh output directory; existing scene edits are preserved')
    out.mkdir(parents=True, exist_ok=True)
    source = out/'source'; source.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.chart_zip) as archive:
        # Exact known members only; never follow archive paths or embedded commands.
        for name in ['PH_Studio_2-3_colorchart.exr','_DNG6501.tif','_DNG6501.NEF']:
            (source/name).write_bytes(archive.read(name))
    env = rgb(args.hdri)
    conversion = radiance(out/'WhiteStudio02.hdr', env)
    chart = rgb(source/'PH_Studio_2-3_colorchart.exr')
    # Central, scratch-free ROIs inside the lower 6x4 classic chart, in source pixels.
    xs = [390,655,918,1181,1444,1707]
    ys = [1798,2060,2323,2588]
    patches = []
    for row,y in enumerate(ys):
        for col,x in enumerate(xs):
            samples = chart[y-36:y+36,x-36:x+36].reshape(-1,3)
            patches.append({'index':len(patches)+1,'name':PATCH_NAMES[len(patches)],'roi':[x-36,y-36,72,72],
                'medianLinearRec709':np.median(samples,axis=0).tolist(), 'stddev':samples.std(axis=0).tolist()})
    gray = np.asarray(patches[21]['medianLinearRec709'])
    gain = .18/float(gray @ [.2126,.7152,.0722])
    write(out/'ChartMeasurements.json', {'source':'source/PH_Studio_2-3_colorchart.exr','colorSpace':'lin_rec709',
        'meaning':'Photographed radiance, NOT reflectance. No automatic HDRI calibration.',
        'displayGain':gain,'displayGainMeaning':'Presentation-only scalar: photographed Neutral 5 luminance maps to 0.18. No white balance matrix.',
        'patches':patches})
    reference = Image.fromarray(np.uint8(np.rint(srgb(chart*gain)*255)))
    reference.thumbnail((1536,1536)); reference.save(out/'PhotographicReference.png')
    marked = Image.fromarray(np.uint8(np.rint(srgb(chart*gain)*255)))
    draw = ImageDraw.Draw(marked)
    for p in patches:
        x,y,w,h = p['roi']; draw.rectangle((x,y,x+w,y+h),outline='yellow',width=4)
    marked.thumbnail((1000,1000)); marked.save(out/'ChartROIs.png')
    template = json.loads((REPO/'Pipelines/Samples/lookdev_vbuffer.metallic_graph.json').read_text())
    environment = {'enabled':True,'path':str(out/'WhiteStudio02.hdr'),'colorSpace':'lin_rec709',
                   'intensity':1.,'rotationDegrees':0.,'visible':True}
    catalog = []

    def finish(key, name, scene, camera, overrides=None, sidecar=None):
        if sidecar is None:
            sidecar = {'version':3,'source':scene.name,'sceneIndex':0,'nodes':[],'materials':overrides or []}
        sidecar['world'] = {'environment':environment,'lighting':{'exposureEV100':0,
            'autoExposure':{'enabled':False,'compensation':0},'lights':[]}}
        write(scene.with_suffix('.metallic_scene.json'),sidecar)
        graph = copy.deepcopy(template); graph['name']=name
        for node in graph['nodes']:
            props=node['properties']
            if 'path' in props: props['path']=scene.as_posix()
            if 'camera' in props: props['camera']=camera
            if node['name']=='Reference': props.update(samples=4,maxDepth=12,accumulate=True)
            if node['name']=='Deferred': props.update(materialBinning=False)
            if node['name']=='Slider': props['splitPosition']=1.0
        graph_path=scene.parent/'LookDev.metallic_graph.json'; write(graph_path,graph)
        catalog.append({'id':'studio-white-'+key,'name':name,'scenePath':scene.as_posix(),'graphPath':graph_path.as_posix(),
                        'environment':str(out/'WhiteStudio02.hdr'),
                        'description':'White Studio 02 / linear Rec.709 / intensity 1 / manual EV 0. Photo chart is radiance, not albedo.'})

    def camera(eye,center):
        return {'eye':eye,'center':center,'up':[0,1,0],'fovDegrees':40,'projection':'perspective',
                'znear':.01,'zfar':100}

    mesh=Mesh(); overrides=[]
    spheres=[('18% gray',[.18]*3,0,.5,{}),('80% white',[.8]*3,0,.5,{}),('Mirror 90%',[.9]*3,1,.025,{}),
             ('Rough metal',[.65]*3,1,.3,{}),('Copper',[.72,.28,.10],1,.2,{}),
             ('Red coat',[.5,.025,.015],0,.3,{'coatWeight':1.,'coatRoughness':.04}),
             ('Anisotropic',[.65]*3,1,.3,{'specularAnisotropy':.8}),
             ('Fuzz',[.025,.012,.018],0,.8,{'fuzzWeight':1.,'fuzzColor':[.3,.06,.12,0]}),
             ('Dielectric gloss',[.07,.2,.4],0,.08,{})]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    for i,(name,color,metal,rough,inputs) in enumerate(spheres):
        x=(i%3-1)*.5; y=(1-i//3)*.53
        index=mesh.add(name,[x,y,0],3.,color,metal,rough)
        if inputs:
            overrides.append({'sourceId':'main','materialIndex':index,'sourceName':name,
                'properties':{'valueProgram':json.dumps({'version':2,'nodes':{},'outputs':inputs})}})
        label=Image.new('RGB',(512,64),(28,28,28)); d=ImageDraw.Draw(label)
        d.text((256,32),name,font=font,anchor='mm',fill=(220,220,220))
        label.save(out/f'Label{i}.png')
        mesh.add(name+' label',[x,y-.2,.025],[4.2,.525,1],[1,1,1],rough=1,kind='card',texture=f'../Label{i}.png')
    overview=out/'Overview'/'Scene.gltf'; overview.parent.mkdir(exist_ok=True); mesh.save(overview)
    finish('overview','White Studio / Material Overview',overview,camera([0,.08,3.25],[0,-.05,0]),overrides)

    mesh=Mesh()
    mesh.add('Photographic ColorChecker - display normalized',[-.47,0,0],[8.5,11.7,1],[1,1,1],
             kind='card',texture='../PhotographicReference.png',unlit=True)
    for i,p in enumerate(patches):
        c=np.asarray(p['medianLinearRec709'])*gain
        mesh.add(f'{i+1:02d} {p["name"]} - radiance swatch',[.07+(i%6)*.155,.28-(i//6)*.19,0],
                 [1.35,1.6,1],c.tolist(),kind='card',unlit=True)
    scene=out/'ChartReference'/'Scene.gltf'; scene.parent.mkdir(exist_ok=True); mesh.save(scene)
    finish('chart','White Studio / Photographic Chart (Unlit)',scene,camera([0,0,2.9],[0,0,0]))

    painter=REPO/'build/MaterialValidation/PainterLookDev/Catalog.json'
    if painter.exists():
        for item in json.loads(painter.read_text())['samples']:
            original=Path(item['scenePath']); key=item['id'].removeprefix('painter-')
            scene=out/key/'Scene.gltf'; scene.parent.mkdir(exist_ok=True)
            model=json.loads(original.read_text())
            for collection in ['buffers','images']:
                for value in model.get(collection,[]):
                    uri=value.get('uri','')
                    if uri and not uri.startswith('data:'):
                        value['uri']=(original.parent/uri).resolve().as_posix()
            write(scene,model)
            sidecar=json.loads(original.with_suffix('.metallic_scene.json').read_text())
            sidecar['source']='Scene.gltf'
            for material in sidecar.get('materials',[]):
                asset=material.get('materialAsset')
                if asset: asset['root']=(original.parent/asset['root']).resolve().as_posix()
            source_graph=json.loads(Path(item['graphPath']).read_text())
            view=next(n['properties']['camera'] for n in source_graph['nodes'] if 'camera' in n['properties'])
            finish(key,'White Studio / '+item['name'],scene,view,sidecar=sidecar)
    write(out/'Catalog.json',{'version':1,'samples':catalog})
    write(out/'ImportReceipt.json',{'sourceHDRI':str(args.hdri.resolve()),'sourceChartZIP':str(args.chart_zip.resolve()),
        'hdriSHA256':hashlib.sha256(args.hdri.read_bytes()).hexdigest(),
        'chartZipSHA256':hashlib.sha256(args.chart_zip.read_bytes()).hexdigest(),
        'sourceURL':'https://polyhaven.com/a/white_studio_02','license':'CC0','author':'Grzegorz Wronkowski',
        'HDRIColorSpace':'Assumed linear Rec.709/D65; source EXR has no chromaticities attribute',
        'chartColorSpace':'linear Rec.709/D65, confirmed by EXR chromaticities',
        'environmentConversion':conversion,'sceneCount':len(catalog),
        'limitations':['RGBE conversion is lossy; original EXR is unchanged.',
            'Photo preview uses sRGB PNG8; measured patch medians remain float constants.',
            'Chart display gain is not a calibrated relation between camera exposure and HDRI.',
            'No reflectance, delta-E, spectral or absolute photometric accuracy claim.']})
    print(f'Built {len(catalog)} scenes; chart display gain {gain:.6f}; HDR conversion {conversion}')


if __name__=='__main__':
    main()
