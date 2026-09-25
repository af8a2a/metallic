"""Hash the declared scene/graph dependency closure, streaming large files."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def dependencies(graph, scene, stream):
    paths = {graph.resolve(), scene.resolve(), stream.resolve()}
    for text in strings(json.loads(graph.read_text(encoding='utf-8-sig'))):
        if text.replace('\\', '/').startswith('Asset/'):
            path = (ROOT / text).resolve()
            if not path.is_file():
                raise ValueError(f'Missing graph asset: {path}')
            paths.add(path)
    document = json.loads(scene.read_text(encoding='utf-8-sig'))
    for group in ('images', 'buffers'):
        for item in document.get(group, []):
            uri = item.get('uri', '')
            if uri and not uri.startswith('data:'):
                path = (scene.parent / unquote(uri)).resolve()
                if not path.is_file():
                    raise ValueError(f'Missing glTF dependency: {path}')
                paths.add(path)
    for suffix in ('.scene.json', '.nodes.bin', '.cfg'):
        candidate = scene.with_suffix(suffix)
        if candidate.is_file():
            paths.add(candidate.resolve())
    if any(not path.is_relative_to(ROOT) for path in paths):
        raise ValueError('Asset closure leaves repository')
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, graph, scene, stream in (
        ('mini', 'Pipelines/Samples/gpu_driven_realtime.metallic_graph.json', 'Asset/MiniZorah/zorah_main_public.v2.gltf',
         'Asset/MeshletCache/MiniZorahCook/MiniZorah.meshstream.bin'),
        ('full', 'Pipelines/Samples/gpu_driven_zorah_full.metallic_graph.json', 'Asset/ZorahFull/zorah_textured_public.v1.gltf',
         'Asset/ZorahFull/zorah_textured_public.v1.gltf.meshstream.bin')):
        records = {}
        files = dependencies(ROOT / graph, ROOT / scene, ROOT / stream)
        for index, path in enumerate(files):
            before = path.stat()
            with path.open('rb') as source:
                checksum = hashlib.file_digest(source, 'sha256').hexdigest()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Asset changed during hashing: {path}')
            records[path.relative_to(ROOT).as_posix()] = {'bytes': after.st_size, 'mtimeNs': after.st_mtime_ns, 'sha256': checksum}
            if index % 100 == 0 or after.st_size > 1024**3:
                print(f'{name}: {index+1}/{len(files)} {path.name} {after.st_size} bytes', flush=True)
        result = {'protocol': 'metallic-declared-assets-v1', 'graph': graph, 'scene': scene, 'stream': stream,
                  'scope': 'graph Asset paths, glTF images/buffers and existing scene sidecars', 'files': records}
        (args.output / f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
