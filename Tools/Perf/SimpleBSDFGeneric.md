# Direct vs generic glTF BSDF experiment

This runner measures only the wrapper change: A uses `bsdf=vkGltf`, B uses
`bsdf=simple`; both use `rayPipeline`, `cacheMode=off`, `debugView=final`, and
`shaderExecutionReorder=false`. No OpenPBR or SER arm and no GPU performance
stage are included. Existing VkGltfSER and PathTraceTraversal evidence is unchanged.

`SimpleBSDFGeneric.py prepare` copies the sealed three-arm fixture inputs into a
new directory, retaining only Opaque1, Mixed4, and ABeautifulGame. Correctness
uses AB/BA: 12 independent processes, core and synchronization validation,
temporal HDR, camera steps where declared, and all seven guides. HDR requires
`abs(B-A) <= 2e-5 + 2e-5*abs(A)`; guides require raw byte equality. Both cross-arm
and same-arm comparisons always require equivalence, even though the BSDF
property strings differ. A complete numerical failure is sealed and returns
exit 2; `verify --allow-numeric-failure` can inspect its integrity without
converting it to acceptance.

The measurement plan has `schema_version: 1`, `experiment: simple_bsdf_generic`,
`correctness: <current capture directory>`, and `roles.single/general.A/B`.
Each arm has `recipe: <actual captured target recipe>` and `stages` containing
ray_generation, miss, closest_hit, and any_hit requests with profile,
capabilities, defines, and search_paths. Generate these requests from the
current C++ factories; no archived executable or script is an authority.
Single recipes must be the Opaque1 block-0 target; general uses Mixed4 block 0.
The plan requires the complete declared correctness set, but numerical status
is reported separately from bytecode/scope attribution.

Both arms use `Features/PathTracing/VkGltfRayPipelinePathTrace` /
`vkGltfRayPipelinePathTraceMain`. Only B's raygen request adds
`METALLIC_SIMPLE_BSDF=1`. All other factory request fields are fixed and equal.
Hit-stage bytecode, native state, non-target PSOs, and source device/features
must match. SER native state and invocation-reorder extensions are rejected.

Native and compile runs use four blocks, AB/BA/AB/BA, for both single/general:
16 independent native samples or 16 compile samples comprising 64 fresh slangc
processes. Native measures only `vkCreateRayTracingPipelinesKHR`, with empty
application cache and controllable driver caching disabled; it excludes Slang
and SBT. Compile measures the sum of four sequential fresh compiler-process
wall times and includes startup/module loading. Compiler and OS caches are not
claimed cold. Every compiler stage must match the actual runtime SPIR-V byte
for byte before production attribution; mismatch remains diagnostic evidence.

```powershell
python -B Tools/Perf/SimpleBSDFGeneric.py prepare
python -B Tools/Perf/SimpleBSDFGeneric.py run --output-dir build-release/simple-bsdf-generic-20261017/correctness
python -B Tools/Perf/SimpleBSDFGenericCompile.py run --mode native --plan <plan.json> --executable <MetallicShaderPipelineBench.exe> --output <fresh-native-directory>
python -B Tools/Perf/SimpleBSDFGenericCompile.py run --mode compile --plan <plan.json> --executable <slangc.exe> --output <fresh-compile-directory>
python -B Tools/Perf/SimpleBSDFGeneric.py verify <correctness-directory>
python -B Tools/Perf/SimpleBSDFGenericCompile.py verify <measurement-directory>
python -B tests/perf/TestSimpleBSDFGeneric.py
```

All runs share the existing cooperative experiment lock, have bounded child
timeouts, preserve raw outputs and process state, and freeze source/binary/
fixture/helper identities. Only controlled environment fields are recorded.
The lock does not prove system-wide GPU exclusivity. Run stages serially after
freezing sources and binaries. Summaries retain all four process samples,
paired percentage changes, their median and range. A small change relative to
dispersion is not proof of non-regression; no samples are discarded. Generate
charts outside sealed raw directories and report correctness separately.
