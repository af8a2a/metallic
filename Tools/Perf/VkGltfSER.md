# Isolated material and SER experiment

`VkGltfSER.py` is separate from the existing strict OpenPBR traversal experiment.
It never modifies or executes scripts from the historical evidence package.

| Arm | BSDF | Traversal | EXT invocation reorder |
| --- | --- | --- | --- |
| O | OpenPBR | RT pipeline | off |
| G | vk_gltf_renderer port | RT pipeline | off |
| S | vk_gltf_renderer port | RT pipeline | on |

O/G measures a material-model change. Its HDR and guides differences are
reported descriptively and are **not** an equivalence or quality gate. G/S
isolates SER and keeps the existing HDR threshold `2e-5 + 2e-5 * abs(reference)`
and raw-byte-exact guides. Same-arm repeated processes use those same strict
gates. Every readback must be finite, every actual PSO/SPIR-V is captured and
hashed, and same-arm PSOs must remain identical. SER identity requires both
the native recipe flag and `SPV_EXT_shader_invocation_reorder` in the actual
raygen payload. These checks do not prove a particular driver reorder rate.

All arms have cache off and use the same workload/camera/resolution. The
Preview device enables the optional EXT feature consistently in every arm;
the shader and native pipeline flag activate it only in S. Changed material
models may omit OpenPBR LUT preparation PSOs. G/S cannot change unrelated PSOs.
Preparation excludes unsupported non-final debug views and records each
excluded case and reason. Local/environment on/off controls remain included;
their readbacks must demonstrate a nonzero lighting contribution in every arm.

## GPU measurements

```powershell
python -B Tools/Perf/VkGltfSER.py prepare
python -B Tools/Perf/VkGltfSER.py run --stage correctness --output-dir build-release/vkgltf-ser-20261017/correctness
python -B Tools/Perf/VkGltfSER.py run --stage performance --output-dir build-release/vkgltf-ser-20261017/performance
python -B Tools/Perf/VkGltfSER.py verify build-release/vkgltf-ser-20261017/correctness
```

`--case` selects a bounded subset and can be repeated; the summary records
incomplete coverage. Correctness runs use core/sync validation and two
independent processes per arm, ordered OGS/SGO. Performance uses three
independent processes per arm, ordered OGS/GSO/SOG, at the prepared 512²,
32-warmup/64-timed-frame workload. One process median is one sample. Ordinary
GPU timing excludes validation, native creation and readback. Driver/OS
warmth is not guaranteed, and the cooperative lock does not prove system-wide
GPU exclusivity. All samples are retained without filtering.

A strict numeric failure seals the full data and exits 2. `verify
--allow-numeric-failure` permits evidence diagnosis but retains
`correctness_qualified=false`; it cannot turn failure into accepted evidence.
Runtime errors, missing stages, nonfinite outputs and identity drift fail
immediately. Do not edit source, assets, binaries or runners during a capture.

## Native PSO and compiler measurements

Create a JSON plan from the actual captured RT recipes. Each role contains
exactly O/G/S; for example `single` may use Opaque1 and `general` Mixed4. Native
mode needs the recipe path. Compile mode additionally needs the exact ordered
production request for each of `ray_generation`, `miss`, `closest_hit`,
`any_hit`:

```json
{
  "schema_version": 1,
  "roles": {
    "single": {
      "O": {
        "recipe": "<actual O recipe path>",
        "stages": {
          "ray_generation": {
            "profile": "spirv_1_6",
            "capabilities": ["<exact production capabilities>"],
            "defines": [["<name>", "<value>"]],
            "search_paths": ["<generated material include directory>"]
          }
        }
      },
      "G": {"recipe": "<actual G recipe path>"},
      "S": {"recipe": "<actual S recipe path>"}
    }
  }
}
```

The abbreviated example must be completed with all four stage requests for
each arm before compiler measurement. Read the current `ShaderRequests.h`
factory and actual generated-material paths; do not reuse stale macro lists.

```powershell
python -B Tools/Perf/VkGltfSERCompile.py run --mode native --plan <plan.json> --executable build-pass-stages-nrd/Source/MetallicShaderPipelineBench.exe --output <new-native-directory>
python -B Tools/Perf/VkGltfSERCompile.py run --mode compile --plan <plan.json> --executable External/slang/bin/slangc.exe --output <new-compile-directory>
python -B Tools/Perf/VkGltfSERCompile.py verify <new-native-directory>
```

Native mode uses one new process and empty application VkPipelineCache per
sample, disables the controllable driver cache, and measures only
`vkCreateRayTracingPipelinesKHR`. Slang and SBT creation are excluded. It does
not claim every compiler/OS cache is cold.

Compile mode sums four separate sequential `slangc` process wall times per
sample; this includes process/module loading. It disables instrumentation and
uses O1/mapped production policy. **Every emitted stage must match the captured
runtime SPIR-V byte for byte before the costs are attributed to production.**
A mismatch remains sealed and unqualified, exits 2, and requires an explicit
`verify --allow-bytecode-mismatch` for diagnosis. It must not be relabeled as
an equivalent production compile measurement. Both modes use three balanced
OGS/GSO/SOG blocks and report material (G/O) and SER (S/G) contrasts separately.

The maintained current verifier recomputes hashes, process order, raw timings,
statistics and readback gates. Never execute archived runner copies from the
evidence packages. Generate graphs in a new report directory outside sealed
raw evidence, retaining the plotting script and source-data provenance.
