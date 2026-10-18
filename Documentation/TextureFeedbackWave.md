# Texture feedback wave aggregation

`ScenePathTracePass.textureFeedbackWave` is an opt-in experiment (default `false`).
It applies to unsplit compute path tracing, including OpenPBR/standard and guide
variants. It requires compute subgroup arithmetic and enabled
`VK_KHR_shader_maximal_reconvergence`. Other devices, ray pipelines, deferred,
realtime, and split-primary paths retain per-lane atomics.

The global `METALLIC_TEXTURE_FEEDBACK_WAVE` shader permutation and
`[MaximallyReconverges]` compute entry attribute are selected together. They enter
the shader cache identity; native warmup also checks the emitted execution mode
before attempting pipeline creation on another device.

The shared `recordTextureDemand` helper groups participating lanes by texture,
reduces their requested mip, and sums their hit counts. One elected lane performs
one atomic min and one atomic add per texture group. Both normal and mip0 requests
use it. Existing record bounds, capture/tile gating, and sampling mip floors stay
in place. Each caller must validate the record and use the same feedback buffer
across participating lanes. The helper does not pre-read the atomically updated
minimum through a non-atomic access.

Compute plus RayQuery alone does not guarantee reconvergence of divergent wave
operations. The [Khronos maximal reconvergence examples](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_maximal_reconvergence.html)
explain the required execution guarantee. A group with many distinct texture keys
needs many reduction iterations and can be slower than per-lane atomics; the
required execution mode can also affect code generation outside this helper.

## Focused validation

Build `MetallicRHITests` and `MetallicVulkanDeviceFeaturesTests` in a compatible
configured build tree. Then run:

```powershell
.\build-pass-stages-nrd\tests\MetallicVulkanDeviceFeaturesTests.exe --gtest_filter=DeviceFeatures.MaximalReconvergenceIsOptionalAndRequiresExtensionAndFeature
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-validation --gtest_filter=RHIRendering.texture_feedback_wave
$env:METALLIC_RUN_TEXTURE_FEEDBACK_WAVE_BENCHMARK = '1'
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --rhi-bindless --rhi-no-validation --gtest_filter=RHIRendering.texture_feedback_wave_benchmark --output-dir build/wave-feedback-measurement
Remove-Item Env:METALLIC_RUN_TEXTURE_FEEDBACK_WAVE_BENCHMARK
```

The probe compares every record word against a CPU oracle across nine key/activity
patterns. Its manual benchmark measures isolated feedback dispatches, not texture
sampling or ray traversal. Production acceptance additionally requires the same
scene/camera/settings, active streaming demand, exact HDR/temporal/guide readbacks,
and repeated process-level timing. A frozen demand-off workload cannot establish
the benefit of aggregation. Fewer authored atomics are not a measured frame gain.

## Initial experiment (2026-10-18)

On RTX 5070 Ti / driver 617.42, the nine-pattern probe passed exact record
comparison and SPIR-V validation. The isolated two/three-key microbenchmarks
improved by about 93%/90%, but distinct keys per lane regressed by about 30%.
The single-key case was essentially unchanged. These are four ABBA batches in
one process, not independent production samples.

A separate four-process ABBA used 512x512, 2 spp, depth 12, cache off, and fully
resident KTX textures with demand collection enabled by the unfrozen host policy.
All final HDR, temporal color, and guide readbacks were byte-identical. The two
paired shading-time changes were +16.6%/+36.5% (Opaque), -1.8%/+28.1% (Mixed),
and +15.2%/+15.6% (Sky, about 20 microseconds). Opaque/Mixed also showed substantial
baseline drift. This does not establish a production gain; keep the experiment
disabled by default. No new Nsight warp/stall or real LookDev-scene claim is made.

Local measurements, hashes, scripts, and charts are retained under
`build/texture-feedback-wave-20261018/Report.html`; the valid production batch is
`production-ab-v3`. Earlier interrupted harness-validation batches are excluded.
