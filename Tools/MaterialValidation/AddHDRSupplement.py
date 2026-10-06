"""Append the HDR supplement to an existing package without modifying its cases."""
import argparse
import hashlib
import json
from pathlib import Path

from PreparePainter import cases, write_json, write_mesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest=json.loads((root/'Manifest.json').read_text(encoding='utf-8'))
    name,variants,maps,purpose=next(c for c in cases() if c[0]=='H01_HDREmission')
    if any(c['id']==name for c in manifest['cases']) or (root/name).exists():
        raise FileExistsError('HDR supplement already exists')
    write_mesh(root/name/'mesh',variants,len(variants))
    entry=dict(id=name,mesh=f'{name}/mesh/Validation.obj',gltf=f'{name}/mesh/GeometryOnly.gltf',
               columns=len(variants),variants=variants,texturedOverrides=maps,purpose=purpose,
               modes=['uniform','textured'])
    write_json(root/name/'Reference.json',dict(model='OpenPBR',version='1.1',workingColorSpace='ACEScg',
               meshUnit='meter',status='authored_input_only',comparison={'Painter':'appearance_reference','Metallic':'not_run'},**entry))
    manifest['cases'].append(entry)
    write_json(root/'Manifest.json',manifest)
    write_json(root/name/'SupplementChecks.json',{'sha256':{
        str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/name).rglob('*') if p.is_file()},
        'note':'Added after original PreparationChecks.json; original manifest hash is historical.'})


if __name__=='__main__':
    main()
