# RTXCR Material Showcase

`MetallicRtxcrSample` integrates the shader-only portions of NVIDIA RTX Character
Rendering (RTXCR) with Metallic's Slang-to-SPIR-V and Vulkan RenderGraph path.
The procedural overview renders three spheres so the material responses can be
compared without downloading the large Claire asset package:

- RTXCR Chiang near-field hair BCSDF
- RTXCR analytical far-field hair BCSDF
- RTXCR Burley diffusion-profile subsurface scattering and volume coefficients

The sample uses the upstream RTXCR Material Library as an external dependency;
the NVIDIA sources are not copied into this repository.

## Configure

Point `RTXCR_ROOT` at either a full RTXCR checkout or the standalone RTXCR
Material Library:

```powershell
cmake -S . -B build -DRTXCR_ROOT=E:/RTXCR -DMETALLIC_BUILD_TESTS=ON
```

The default discovery also checks these locations:

- `External/RTXCR-Material`
- a sibling `../RTXCR/libraries/rtxcr/material` checkout

The standalone dependency can be obtained from
`https://github.com/NVIDIA-RTX/RTXCR-Material-Library.git`.

## Build and run

```powershell
cmake --build build --target MetallicRtxcrSample --config Debug
# Visual Studio / other multi-config generators
build\Source\Debug\MetallicRtxcrSample.exe
# Ninja / other single-config generators
build\Source\MetallicRtxcrSample.exe
```

For a one-frame integration check:

```powershell
build\Source\Debug\MetallicRtxcrSample.exe --smoke-test
```

For a single-config build, omit the `Debug` directory in the smoke-test path.

The runtime settings panel exposes the overview and individual material views,
hair melanin/redness, longitudinal and azimuthal roughness, cuticle angle, IOR,
subsurface scale/anisotropy/sample radius, light azimuth, and exposure.

## Architecture

`RtxcrMaterialSamplePass` compiles `Shaders/RtxcrMaterialSample.slang` with two
Slang search roots: Metallic's `Shaders/` directory and the detected RTXCR
`shaders/include` directory. Imported RTXCR files are included in Metallic's
shader dependency tracking and invalidate the SPIR-V disk cache when changed.
The pass writes a RenderGraph storage image through Metallic's Vulkan compute
program wrapper.

The Burley view evaluates the upstream RTXCR diffusion-profile sampler over a
procedural tangent-plane neighborhood and combines it with RTXCR volume
coefficients for a compact transmission demonstration. It is intended to make
the library functions and parameter response easy to validate; it is not a
replacement for a production mesh-space BSSRDF integrator.

## Scope

This integration covers the RTXCR Material Library. RTXCR Geometry Library
features such as DOTS, Linear-Swept Spheres (LSS), morph-target animation, and
hardware LSS intersection require additional scene import and Vulkan
acceleration-structure work and are not enabled by this sample.
