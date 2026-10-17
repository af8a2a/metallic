"""Compile and compare the isolated full glTF BSDF against its upstream Slang.

This is CPU numerical validation, not GPU/runtime or scene-render validation.
The upstream source must be provided explicitly and match the vendored manifest.
Run from a Visual Studio x64 developer shell, or pass --vcvars64 on Windows.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    ('baseColor', 3), ('opacity', 1), ('roughness', 2), ('metallic', 1),
    ('emissive', 3), ('occlusion', 1), ('N', 3), ('T', 3), ('B', 3), ('Ng', 3),
    ('ior1', 1), ('ior2', 1), ('dispersion', 1), ('specular', 1),
    ('specularColor', 3), ('transmission', 1), ('transmissionTint', 3),
    ('attenuationColor', 3), ('attenuationDistance', 1), ('thickness', 1),
    ('scatterCoefficient', 3), ('scatterAnisotropy', 1), ('clearcoat', 1),
    ('clearcoatRoughness', 1), ('Nc', 3), ('iridescence', 1),
    ('iridescenceIor', 1), ('iridescenceThickness', 1), ('sheenColor', 3),
    ('sheenRoughness', 1), ('diffuseTransmissionFactor', 1),
    ('diffuseTransmissionColor', 3), ('retroreflection', 1),
]
MATERIAL_FLOATS = sum(n for _, n in FIELDS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], output: Path, name: str) -> None:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    (output / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); see {output / (name + ".log")}')


def default_material() -> dict:
    return dict(baseColor=[1., 1., 1.], opacity=1., roughness=[1., 1.], metallic=1.,
        emissive=[0., 0., 0.], occlusion=1., N=[0., 0., 1.], T=[1., 0., 0.],
        B=[0., 1., 0.], Ng=[0., 0., 1.], ior1=1., ior2=1.5, dispersion=0.,
        specular=1., specularColor=[1., 1., 1.], transmission=0.,
        transmissionTint=[1., 1., 1.], attenuationColor=[1., 1., 1.],
        attenuationDistance=1., thickness=0., scatterCoefficient=[0., 0., 0.],
        scatterAnisotropy=0., clearcoat=0., clearcoatRoughness=.01,
        Nc=[0., 0., 1.], iridescence=0., iridescenceIor=1.5,
        iridescenceThickness=.1, sheenColor=[0., 0., 0.], sheenRoughness=0.,
        diffuseTransmissionFactor=0., diffuseTransmissionColor=[1., 1., 1.],
        retroreflection=0.)


def cases() -> list[tuple[str, dict]]:
    variants = [
        ('lambert', dict(metallic=0., specular=0., baseColor=[.2, .5, .8])),
        ('diffuse_transmission', dict(metallic=0., specular=0.,
            diffuseTransmissionFactor=1., diffuseTransmissionColor=[.7, .3, .1])),
        ('dielectric', dict(metallic=0., roughness=[.16, .16])),
        ('metal', dict(baseColor=[.9, .6, .2], roughness=[.2, .2])),
        ('anisotropic', dict(baseColor=[.8, .9, 1.], roughness=[.8, .08])),
        ('thin_transmission', dict(metallic=0., transmission=1., roughness=[.1, .1])),
        ('volume_transmission', dict(metallic=0., transmission=1., thickness=.4,
            roughness=[.2, .2], attenuationColor=[.8, .2, .4], attenuationDistance=.7)),
        ('inside_volume', dict(metallic=0., transmission=1., thickness=.4,
            roughness=[.2, .2], ior1=1.5, ior2=1.)),
        ('clearcoat', dict(metallic=.2, clearcoat=1., clearcoatRoughness=.25)),
        ('sheen', dict(metallic=0., sheenColor=[.4, .6, .9], sheenRoughness=.5)),
        ('iridescent_dielectric', dict(metallic=0., iridescence=1.,
            iridescenceThickness=450., roughness=[.3, .3])),
        ('iridescent_metal', dict(iridescence=.8, iridescenceIor=1.3,
            iridescenceThickness=300., roughness=[.25, .25])),
        ('dispersion', dict(metallic=0., transmission=1., thickness=.3,
            dispersion=.7, roughness=[.2, .2])),
        ('retroreflection', dict(metallic=.5, retroreflection=.6, roughness=[.3, .3])),
        ('transmission_tint', dict(metallic=0., transmission=.8,
            transmissionTint=[.2, .6, .9], roughness=[.2, .2])),
        ('phase_forward', dict(scatterAnisotropy=.8)),
        ('phase_backward', dict(scatterAnisotropy=-.8)),
        ('mixed', dict(metallic=.3, transmission=.2, clearcoat=.5,
            clearcoatRoughness=.2, sheenColor=[.3, .2, .5], sheenRoughness=.4,
            iridescence=.4, iridescenceThickness=350., diffuseTransmissionFactor=.3,
            diffuseTransmissionColor=[.3, .7, .1], roughness=[.6, .15])),
        ('grazing_rough', dict(metallic=.2, roughness=[.9, .7])),
    ]
    result = []
    for name, override in variants:
        material = default_material()
        material.update(override)
        result.append((name, material))
    return result


def reference_shader(upstream: Path) -> str:
    return '''#language slang 2026
module VkGltfBSDFUpstreamProbe;
#include "''' + (upstream / 'bsdf_functions.h.slang').as_posix() + '''"
StructuredBuffer<PbrMaterial> probeMaterials;
StructuredBuffer<float4> probeDirections;
RWStructuredBuffer<float4> probeOutput;
[shader("compute")]
[numthreads(64, 1, 1)]
void vkGltfBSDFProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    PbrMaterial mat = probeMaterials[lane];
    float3 k1 = probeDirections[lane * 3].xyz;
    float3 k2 = probeDirections[lane * 3 + 1].xyz;
    float3 xi = probeDirections[lane * 3 + 2].xyz;
    BsdfEvaluateData eval = {};
    eval.k1 = k1; eval.k2 = k2; eval.xi = xi;
    bsdfEvaluate(eval, mat);
    BsdfSampleData sample = {};
    sample.k1 = k1; sample.xi = xi;
    bsdfSample(sample, mat);
    probeOutput[lane * 5] = float4(eval.bsdf_diffuse, eval.pdf);
    probeOutput[lane * 5 + 1] = float4(eval.bsdf_glossy, float(sample.event_type));
    probeOutput[lane * 5 + 2] = float4(sample.k2, sample.pdf);
    probeOutput[lane * 5 + 3] = float4(sample.bsdf_over_pdf, 0);
    probeOutput[lane * 5 + 4] = float4(volumeExtinctionCoefficient(mat),
        henyeyGreenstein(dot(k1, k2), mat.scatterAnisotropy));
}
'''


def compile_dll(cpp: Path, output: Path, name: str, vcvars: Path | None) -> Path:
    source = cpp.read_text(encoding='utf-8')
    material_type = re.search(r'StructuredBuffer<([^>]+)> probeMaterials_', source).group(1)
    # The static assertion prevents a changed shader layout from silently invalidating ctypes input.
    adapter = output / (name + '_adapter.cpp')
    adapter.write_text('#include "' + cpp.as_posix() + '"\n'
        + f'static_assert(sizeof({material_type}) == {MATERIAL_FLOATS * 4});\n', encoding='utf-8')
    dll = output / (name + '.dll')
    command = ['cl', '/nologo', '/LD', '/O2', '/EHsc', '/fp:precise',
        str(adapter), '/Fo' + str(output / (name + '.obj')), '/link',
        '/OUT:' + str(dll), '/IMPLIB:' + str(output / (name + '.lib'))]
    if vcvars:
        # All paths are quoted by Windows' subprocess implementation; no user shell text is evaluated.
        batch = output / (name + '_build.cmd')
        batch.write_text('@echo off\ncall "' + str(vcvars) + '" >nul\n'
            + subprocess.list2cmdline(command) + '\n', encoding='utf-8')
        command = ['cmd', '/c', str(batch)]
    run(command, output, name + '_build')
    return dll


class Buffer(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('count', ct.c_size_t)]


class Globals(ct.Structure):
    _fields_ = [('materials', Buffer), ('directions', Buffer), ('output', Buffer)]


class Dispatch(ct.Structure):
    _fields_ = [('start', ct.c_uint32 * 3), ('end', ct.c_uint32 * 3)]


def execute(dll: Path, materials, directions, count: int):
    output = (ct.c_float * (count * 20))()
    globals_ = Globals(Buffer(ct.addressof(materials), count),
        Buffer(ct.addressof(directions), count * 3), Buffer(ct.addressof(output), count * 5))
    dispatch = Dispatch((ct.c_uint32 * 3)(0, 0, 0), (ct.c_uint32 * 3)(count // 64, 1, 1))
    library = ct.CDLL(str(dll))
    kernel = library.vkGltfBSDFProbeMain
    kernel.argtypes = [ct.POINTER(Dispatch), ct.c_void_p, ct.POINTER(Globals)]
    kernel.restype = None
    kernel(ct.byref(dispatch), None, ct.byref(globals_))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True, help='nvpro_core2/nvshaders directory')
    parser.add_argument('--output', type=Path, required=True, help='new evidence directory')
    parser.add_argument('--vcvars64', type=Path)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args()
    if args.samples < 1024 or args.samples % 64:
        parser.error('--samples must be >= 1024 and divisible by 64')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = args.upstream.resolve()
    manifest = json.loads((ROOT / 'Shaders/Modules/VkGltfBSDF/SourceManifest.json').read_text())
    for entry in manifest['files'] + manifest.get('validationDependencies', []):
        source = upstream / Path(entry['source']).name
        if sha(source) != entry['sha256']:
            raise RuntimeError(f'Upstream snapshot mismatch: {source}')
    reference = output / 'UpstreamProbe.slang'
    reference.write_text(reference_shader(upstream), encoding='utf-8')
    compiler = ROOT / 'External/slang/bin/slangc.exe'
    shaders = dict(port=ROOT / 'tests/rhi/shaders/VkGltfBSDFProbe.slang', upstream=reference)
    dlls = {}
    for name, shader in shaders.items():
        cpp = output / (name + '.cpp')
        run([str(compiler), str(shader), '-entry', 'vkGltfBSDFProbeMain', '-target', 'cpp',
            '-I', str(ROOT / 'Shaders/Modules'), '-o', str(cpp)], output, name + '_slang')
        dlls[name] = compile_dll(cpp, output, name, args.vcvars64)
    rng = random.Random(0x20261017)
    variants = cases()
    count = len(variants) * args.samples
    material_data = []
    directions_data = []
    for name, material in variants:
        flat = []
        for field, size in FIELDS:
            value = material[field]
            flat.extend(value if size > 1 else [value])
        for _ in range(args.samples):
            material_data.extend(flat)
            z = .002 if name == 'grazing_rough' else rng.uniform(.05, 1.)
            phi = rng.random() * math.tau
            view = [math.sqrt(1. - z*z)*math.cos(phi), math.sqrt(1. - z*z)*math.sin(phi), z]
            z = rng.uniform(-1., 1.)
            phi = rng.random() * math.tau
            light = [math.sqrt(1. - z*z)*math.cos(phi), math.sqrt(1. - z*z)*math.sin(phi), z]
            directions_data.extend(view + [0.] + light + [0.] + [rng.random() for _ in range(3)] + [0.])
    materials = (ct.c_float * len(material_data))(*material_data)
    directions = (ct.c_float * len(directions_data))(*directions_data)
    result = {name: execute(dll, materials, directions, count) for name, dll in dlls.items()}
    for name, data in result.items():
        (output / (name + '.f32')).write_bytes(bytes(data))
    comparisons = []
    for index, (name, material) in enumerate(variants):
        start, stop = index * args.samples * 20, (index + 1) * args.samples * 20
        reference_values = result['upstream'][start:stop]
        port_values = result['port'][start:stop]
        finite = all(math.isfinite(x) for x in reference_values + port_values)
        errors = [abs(a-b) for a,b in zip(reference_values, port_values)]
        tolerance = all(e <= 2e-6 + 2e-6 * abs(a) for e,a in zip(errors, reference_values))
        comparisons.append(dict(name=name, finite=finite, matched=tolerance,
            max_abs_error=max(errors), changed_components=sum(e != 0 for e in errors)))
    # Independent analytic checks, separate from upstream/port agreement.
    analytic = []
    for case_index, case_name, color, sign in [(0, 'lambert', [.2,.5,.8], 1.),
                                               (1, 'diffuse_transmission', [.7,.3,.1], -1.)]:
        max_error = 0.
        for sample in range(args.samples):
            lane = case_index * args.samples + sample
            offset = lane * 20
            z = directions[lane*12 + 6]
            projected = max(0., z * sign) / math.pi
            for c in range(3):
                max_error = max(max_error, abs(result['port'][offset+c] - projected * color[c]))
                max_error = max(max_error, abs(result['port'][offset+12+c] - color[c]))
            max_error = max(max_error, abs(result['port'][offset+3] - projected))
            event = int(result['port'][offset+7])
            if event != (9 if sign > 0 else 17):
                raise AssertionError(f'{case_name}: unexpected sample event {event}')
        analytic.append(dict(name=case_name, max_abs_error=max_error, passed=max_error < 2e-6))
    for case_index, (name, material) in enumerate(variants):
        expected = [-math.log(max(c,.001))/max(material['attenuationDistance'],.001)
                    for c in material['attenuationColor']]
        offset = case_index * args.samples * 20 + 16
        error = max(abs(result['port'][offset+i]-expected[i]) for i in range(3))
        analytic.append(dict(name=name+'_extinction', max_abs_error=error, passed=error < 2e-6))
        phase_error = 0.
        phase_matches = True
        g = material['scatterAnisotropy']
        for sample in range(args.samples):
            lane = case_index * args.samples + sample
            cosine = sum(directions[lane*12+i] * directions[lane*12+4+i] for i in range(3))
            expected_phase = (1.-g*g)/(4.*math.pi*(1.+g*g-2.*g*cosine)**1.5)
            error = abs(result['port'][lane*20+19] - expected_phase)
            phase_error = max(phase_error, error)
            # The HG denominator approaches (1-|g|)^2. Its float32 dot/denominator
            # roundoff is amplified by d(log(x^-3/2))/dx = -1.5/x. Use that
            # condition bound for the double-precision analytic reference;
            # upstream/port comparison above retains its original strict gate.
            denom = 1.+g*g-2.*g*cosine
            roundoff_bound = (2e-6 + 2e-6*abs(expected_phase)
                + abs(expected_phase)*1.5*(8.*2.**-24)/denom)
            phase_matches &= error < roundoff_bound
        analytic.append(dict(name=name+'_hg_formula', max_abs_error=phase_error,
            tolerance='2e-6 + 2e-6*abs(reference) + abs(reference)*1.5*(8*2^-24)/denominator',
            passed=phase_matches))
    # HG normalization integral over cos(theta), independently using trapezoid integration.
    for g in [0., .8, -.8]:
        steps = 100000
        values = [(1.-g*g)/(4.*math.pi*(1.+g*g-2.*g*(-1.+2.*i/steps))**1.5)
                  for i in range(steps+1)]
        integral = math.tau * (2./steps) * (sum(values) - (values[0]+values[-1])*.5)
        analytic.append(dict(name=f'hg_normalization_g{g}', value=integral,
            passed=abs(integral-1.) < 1e-6))
    passed = all(x['finite'] and x['matched'] for x in comparisons) and all(x['passed'] for x in analytic)
    summary = dict(passed=passed, scope='CPU Slang numerical validation; no GPU execution',
        material_cases=len(variants), samples_per_case=args.samples, total_samples=count,
        abs_tolerance=2e-6, rel_tolerance=2e-6, comparisons=comparisons, analytic_checks=analytic,
        shader_compiler_sha256=sha(compiler), upstream_manifest=manifest,
        material_fields=FIELDS, material_stride=MATERIAL_FLOATS*4,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'Shaders/Modules/VkGltfBSDF.slang', shaders['port']]
            + list((ROOT/'Shaders/Modules/VkGltfBSDF').glob('*.slang'))},
        command=sys.argv, python=sys.version)
    summary['byte_identical'] = bytes(result['port']) == bytes(result['upstream'])
    (output/'Inputs.json').write_text(json.dumps(dict(cases=variants, seed=0x20261017,
        samples_per_case=args.samples),indent=2)+'\n',encoding='utf-8')
    (output/'Summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    evidence_files = {str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file()}
    (output/'Manifest.json').write_text(json.dumps(evidence_files,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=passed, samples=count, cases=len(variants), output=str(output))))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
