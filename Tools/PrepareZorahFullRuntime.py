"""Generate ZorahFull's bounded first-frame graph from a fully verified cook."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def power_of_two(value):
    return 1 << (max(1, value) - 1).bit_length()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'build-release/zorah-z5/Full.runtime.json')
    parser.add_argument('--geometry-mib', type=int, default=3584)
    parser.add_argument('--clas-mib', type=int, default=2048)
    parser.add_argument('--output', type=Path, default=ROOT / 'Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['payloadValidation'] != 'all-pages':
        raise ValueError('A completed cook with all payloads validated is required')
    budget = args.geometry_mib * 1024**2
    if not manifest['minimumResidentBytesIncludingOneStreamPage'] <= budget < 2**32:
        raise ValueError('Geometry budget must hold the complete terminal cut plus a streaming page, below 4 GiB')
    active_groups = min(power_of_two(manifest['terminalInstanceGroups'] * 2), (2**25-1)//32)
    if manifest['terminalInstanceGroups'] > active_groups:
        raise ValueError('The terminal instance cut exceeds the visibility record capacity')
    if args.clas_mib <= 0:
        raise ValueError('CLAS budget must be positive')
    source = Path(manifest['source'])
    if not source.is_absolute():
        source = ROOT / source
    graph = json.loads((ROOT / 'Pipelines/Samples/gpu_driven_realtime.metallic_graph.json').read_text())
    graph['name'] = 'ZorahFull Streamed Realtime'
    for node in graph['nodes']:
        props = node['properties']
        if node['name'] in ('VBuffer', 'Deferred', 'Shadows'):
            props.update(path=source.relative_to(ROOT).as_posix(),
                         materialTextureMaxDimension=128, materialTextureMaskMaxDimension=512, materialTextureBudgetMiB=512,
                         materialTextureStreaming=True, materialTextureRefineDimension=512, materialTextureColdFrames=180)
        if node['name'] == 'VBuffer':
            props.update(streamAssetPath=Path(manifest['asset']).relative_to(ROOT).as_posix(),
                         maxResidentBytes=budget, maxClasBytes=args.clas_mib * 1024**2,
                         maxLockedFallbackPages=power_of_two(manifest['terminalPageCount']),
                         maxActiveGroups=active_groups,
                         compactShadingAttributes=manifest.get('compactShadingAttributes', False),
                         maxResidentPages=0, maxBlasBytes=512 * 1024**2)
    cfg = source.with_suffix('.cfg').read_text()
    camera_text = re.search(r'--camerastring\s+"([^"]+)"', cfg).group(1)
    camera = [[float(v) for v in group.split(',')] for group in re.findall(r'\{([^}]+)\}', camera_text)]
    graph['view']['camera'] = dict(eye=camera[0], center=camera[1], up=camera[2],
                                   fovDegrees=camera[3][0], znear=camera[4][0], zfar=camera[4][1], reversedZ=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    current = None
    if args.output.exists():
        try:
            current = json.loads(args.output.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            pass
    changed = current != graph
    # Avoid unnecessary editor file-watch reloads for identical settings.
    if changed:
        args.output.write_text(json.dumps(graph, indent=4) + '\n', encoding='utf-8')
    print(json.dumps(dict(graph=str(args.output), changed=changed, rootPages=manifest['terminalPageCount'],
                         rootBytes=manifest['terminalPageBytesAligned256'],
                         rootInstanceGroups=manifest['terminalInstanceGroups'], geometryBudgetBytes=budget)))


if __name__ == '__main__':
    main()
