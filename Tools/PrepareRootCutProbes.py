"""Prepare isolated root-heavy primitive probes from a MeshletCook inspection report.

Reuses the Zorah probe remapper, retains source material/attribute semantics and
references existing payload files. Never cooks or changes the production asset.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from PrepareZorahFullProbes import make_probe, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    source = args.source.resolve(strict=True)
    raw = source.read_bytes()
    scene = json.loads(raw)
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    # The cook records the flattened source primitive index. This is the same
    # mapping used by validateMeshletStreamAttributes, including uninstanced meshes.
    indices = [(mi, pi) for mi, mesh in enumerate(scene['meshes'])
               for pi in range(len(mesh['primitives']))]
    candidates = audit['rootCutAudit']['primitivesByRootBytes'][:args.count]
    args.directory.mkdir(parents=True, exist_ok=True)
    probes = []
    for candidate in candidates:
        mi, pi = indices[candidate['sourcePrimitive']]
        primitive = scene['meshes'][mi]['primitives'][pi]
        if primitive.get('material', 0) != candidate['material']:
            raise ValueError('Source material differs from cook; verify the audit/source pair')
        isolated = copy.deepcopy(scene)
        isolated['meshes'][mi]['primitives'] = [primitive]
        probe, info = make_probe(isolated, source, [mi])
        name = f"RootPrimitive{candidate['primitive']}"
        path = args.directory / (name + '.gltf')
        write_json(path, probe)
        probes.append(dict(name=name, path=str(path.resolve()), sourceMesh=mi,
                           sourceMeshPrimitive=pi, meshName=scene['meshes'][mi].get('name'),
                           audit=candidate, probeInfo=info))
    write_json(args.directory / 'root-probes.json', dict(
        source=str(source), sourceSha256=hashlib.sha256(raw).hexdigest(),
        audit=str(args.audit.resolve()), probes=probes,
        note='Metadata probes only; production payloads are referenced read-only; no recook or simplification changes.'))
    print(json.dumps([dict(name=p['name'], path=p['path'], triangles=p['probeInfo']['sourceTriangles'])
                      for p in probes], indent=2))


if __name__ == '__main__':
    main()
