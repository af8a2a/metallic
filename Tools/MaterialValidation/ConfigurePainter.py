"""Configure a fresh package for the locally verified Painter 12.1.4 installation."""
import argparse
import json
from pathlib import Path

import numpy as np
import OpenEXR


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--painter-root',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    resources=args.painter_root.resolve()/'resources'
    if not (root/'Manifest.json').is_file():
        raise FileNotFoundError('Run PreparePainter.py first')
    destinations=[root/n for n in ['PainterProfile.json','NeutralStudio.exr','NeutralStudio.json']]
    if any(p.exists() for p in destinations):
        raise FileExistsError('Configuration already exists; existing evidence is preserved')
    profile=json.loads(Path(__file__).with_name('Painter12.1.4Profile.json').read_text(encoding='utf-8'))
    profile['ocioConfig']=str(resources/'ocio/aces_2.0/config.ocio')
    profile['templates']={'default':str(resources/'starter_assets/templates/OpenPBR - Coat.spt')}
    for path in [profile['ocioConfig'],profile['templates']['default']]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    profile['environment']=str(root/'NeutralStudio.exr')
    height,width=512,1024
    yy,xx=np.mgrid[0:height,0:width]
    u,v=xx/width,yy/height
    pixels=np.full((height,width,3),.12,dtype=np.float32)
    pixels[(u>=.10)&(u<.16)&(v>=.18)&(v<.45)]=8.
    pixels[(u>=.56)&(u<.66)&(v>=.25)&(v<.50)]=3.
    OpenEXR.File({}, {'RGB':pixels}).write(profile['environment'])
    metadata={'type':'authored neutral analytic environment','projection':'latlong',
              'colorSpace':'linear neutral RGB','baseRadiance':.12,'keyRadiance':8.,
              'fillRadiance':3.,'units':'relative linear radiance','notMeasuredHDRI':True,
              'keyUVBounds':[.10,.16,.18,.45],'fillUVBounds':[.56,.66,.25,.50]}
    (root/'NeutralStudio.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (root/'PainterProfile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    print('Configured',root,'for Painter',profile['verifiedPainterVersion'])


if __name__=='__main__':
    main()
