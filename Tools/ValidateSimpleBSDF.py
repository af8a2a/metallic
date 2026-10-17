"""Compare the concrete SimpleBSDF generic facade with direct VkGltfBSDF calls.

Runs Slang CPU C++ kernels only; does not start a GPU or create a native PSO.
The existing full-BSDF cases and compiler adapter are reused without modification.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import random
import sys

import ValidateVkGltfBSDF as reference

ROOT = reference.ROOT
OUTPUT_FLOATS = 32


def execute(dll: Path, materials, directions, count: int):
    output = (ct.c_float * (count * OUTPUT_FLOATS))()
    globals_ = reference.Globals(reference.Buffer(ct.addressof(materials), count),
        reference.Buffer(ct.addressof(directions), count * 3),
        reference.Buffer(ct.addressof(output), count * 8))
    dispatch = reference.Dispatch((ct.c_uint32 * 3)(0, 0, 0), (ct.c_uint32 * 3)(count // 64, 1, 1))
    library = ct.CDLL(str(dll))
    kernel = library.simpleBSDFProbeMain
    kernel.argtypes = [ct.POINTER(reference.Dispatch), ct.c_void_p, ct.POINTER(reference.Globals)]
    kernel.restype = None
    kernel(ct.byref(dispatch), None, ct.byref(globals_))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory')
    parser.add_argument('--vcvars64', type=Path)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args()
    if args.samples < 1024 or args.samples % 64:
        parser.error('--samples must be >= 1024 and divisible by 64')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = ROOT / 'External/slang/bin/slangc.exe'
    shader = ROOT / 'tests/rhi/shaders/SimpleBSDFProbe.slang'
    dlls = {}
    commands = {}
    for name, generic in [('direct', 0), ('generic', 1)]:
        cpp = output / (name + '.cpp')
        command = [str(compiler), str(shader), '-entry', 'simpleBSDFProbeMain', '-target', 'cpp',
            '-I', str(ROOT / 'Shaders/Modules'),
            f'-DMETALLIC_SIMPLE_BSDF_PROBE_GENERIC={generic}', '-o', str(cpp)]
        commands[name] = command
        reference.run(command, output, name + '_slang')
        dlls[name] = reference.compile_dll(cpp, output, name, args.vcvars64)

    variants = reference.cases()
    emission_case = reference.default_material()
    emission_case.update(metallic=0.3, emissive=[0.125, 2.0, 5.0], roughness=[0.2, 0.2])
    variants.append(('emission_passthrough', emission_case))
    rng = random.Random(0x20261017)
    material_data, direction_data = [], []
    for name, material in variants:
        flat = []
        for field, size in reference.FIELDS:
            value = material[field]
            flat.extend(value if size > 1 else [value])
        for _ in range(args.samples):
            material_data.extend(flat)
            z = 0.002 if name == 'grazing_rough' else rng.uniform(0.05, 1.0)
            phi = rng.random() * math.tau
            wo = [math.sqrt(1.0-z*z)*math.cos(phi), math.sqrt(1.0-z*z)*math.sin(phi), z]
            z = rng.uniform(-1.0, 1.0)
            phi = rng.random() * math.tau
            wi = [math.sqrt(1.0-z*z)*math.cos(phi), math.sqrt(1.0-z*z)*math.sin(phi), z]
            direction_data.extend(wo + [0.0] + wi + [0.0] + [rng.random() for _ in range(3)] + [0.0])
    count = len(variants) * args.samples
    materials = (ct.c_float * len(material_data))(*material_data)
    directions = (ct.c_float * len(direction_data))(*direction_data)
    (output / 'Materials.f32').write_bytes(bytes(materials))
    (output / 'Directions.f32').write_bytes(bytes(directions))
    results = {name: execute(dll, materials, directions, count) for name, dll in dlls.items()}
    for name, data in results.items():
        (output / (name + '.f32')).write_bytes(bytes(data))
    comparisons = []
    for index, (name, _) in enumerate(variants):
        start, stop = index * args.samples * OUTPUT_FLOATS, (index + 1) * args.samples * OUTPUT_FLOATS
        direct, generic = results['direct'][start:stop], results['generic'][start:stop]
        errors = [abs(a-b) for a, b in zip(direct, generic)]
        comparisons.append(dict(name=name, finite=all(math.isfinite(x) for x in direct + generic),
            max_abs_error=max(errors), changed_components=sum(a != b for a, b in zip(direct, generic))))
    data = results['generic']
    importance_rejected = all(data[lane * OUTPUT_FLOATS + 28] == 1.0 for lane in range(count))
    source_state_preserved = all(data[lane * OUTPUT_FLOATS + 24] == 1.0 and
        data[lane * OUTPUT_FLOATS + 25] == float((lane & 255) + 1) and
        data[lane * OUTPUT_FLOATS + 26] == 40.0 and
        data[lane * OUTPUT_FLOATS + 27] == lane + 0.25 for lane in range(count))
    exact = bytes(results['direct']) == bytes(results['generic'])
    passed = exact and importance_rejected and source_state_preserved and all(x['finite'] for x in comparisons)
    sources = [ROOT / 'Shaders/Modules/SimpleBSDF.slang', shader,
        ROOT / 'Shaders/Modules/VkGltfBSDF.slang', ROOT / 'Shaders/Modules/MaterialProgram.slang',
        Path(__file__), Path(reference.__file__)]
    sources += list((ROOT / 'Shaders/Modules/VkGltfBSDF').glob('*.slang'))
    sources += list((ROOT / 'Shaders/Modules/MaterialProgram').glob('*.slang'))
    summary = dict(passed=passed, scope='CPU Slang direct-vs-generic facade validation; no GPU/PSO execution',
        reference_cases=19, additional_cases=['emission_passthrough'], samples_per_case=args.samples,
        total_samples=count, byte_identical=exact, comparisons=comparisons,
        source_resolve_once_and_mutations_preserved=source_state_preserved,
        unsupported_importance_rejected=importance_rejected,
        native_events=sorted(set(int(data[lane * OUTPUT_FLOATS + 7]) for lane in range(count))),
        native_dirac_samples=sum(data[lane * OUTPUT_FLOATS + 11] == -1.0 for lane in range(count)),
        material_stride=reference.MATERIAL_FLOATS * 4, output_float_stride=OUTPUT_FLOATS,
        compiler_sha256=reference.sha(compiler), commands=commands,
        source_sha256={str(path.relative_to(ROOT)): reference.sha(path) for path in sources},
        python=sys.version)
    (output / 'Inputs.json').write_text(json.dumps(dict(cases=variants, seed=0x20261017,
        samples_per_case=args.samples), indent=2) + '\n', encoding='utf-8')
    (output / 'Summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    manifest = {path.name: reference.sha(path) for path in output.iterdir() if path.is_file()}
    (output / 'Manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=passed, total_samples=count, byte_identical=exact,
        source_resolve_once_and_mutations_preserved=source_state_preserved,
        unsupported_importance_rejected=importance_rejected), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
