# RTXCR Claire Ponytail Sample

`MetallicRtxcrSample` renders an NVIDIA Claire reference groom through Metallic's
Vulkan ray-query path. The sample works directly from the pinned upstream
repositories instead of copying NVIDIA source or assets into Metallic:

- [`External/RTXCR-Material`](https://github.com/NVIDIA-RTX/RTXCR-Material-Library)
  (`v1.2.0`) supplies the RTXCR Chiang hair BSDF.
- [`External/RTXCR-Geometry`](https://github.com/NVIDIA-RTX/RTXCR-Geometry-Library)
  (`v1.1.0`) supplies the CPU DOTS conversion.
- [`External/RTXCR-Assets`](https://github.com/NVIDIA-RTX/RTXCR-Assets)
  (`v1.1.0`) supplies the Claire ponytail and studio HDR.

The gitlinks, rather than branch names, define the reproducible revisions used by
the project.

## License acceptance and checkout

Read and accept the licenses before initializing or using these submodules:

- [`External/RTXCR-Material/License.txt`](../External/RTXCR-Material/License.txt)
- [`External/RTXCR-Geometry/License.txt`](../External/RTXCR-Geometry/License.txt)
- [`External/RTXCR-Assets/License.txt`](../External/RTXCR-Assets/License.txt)
- [Claire asset license](../External/RTXCR-Assets/Claire/NVIDIA%20License%20for%20Claire%20Assets%20(2024.11.18)%20%5BFINAL%5D.pdf)

The Claire license is restricted. In particular, it permits the asset solely for
use with NVIDIA Avatar Cloud Engine technologies when building and deploying game
characters and interactive avatars. It does not permit distributing Claire as a
stand-alone asset. Preserve all copyright and proprietary notices, do not imply
NVIDIA sponsorship or endorsement, and review the complete license rather than
relying on this summary.

This sample confines the asset to RTXCR character rendering. NVIDIA lists RTX
Character Rendering under ACE's
[Digital Human Rendering Technologies](https://developer.nvidia.com/ace).

After accepting the terms, initialize the exact revisions recorded by Metallic:

```powershell
git submodule update --init --recursive -- `
    External/RTXCR-Material `
    External/RTXCR-Geometry `
    External/RTXCR-Assets
git -C External/RTXCR-Assets lfs pull
git -C External/RTXCR-Assets lfs ls-files
```

The CMake check does not enable the Claire asset integration when its `.bin` is
still a small Git LFS pointer, and emits a configuration warning with the recovery
commands instead of deferring the problem to an opaque runtime import error.

## Configure, build, and run

The default configuration discovers the three `External/RTXCR-*` submodules:

```powershell
cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON
cmake --build build --target MetallicRtxcrSample --config Debug
build\Source\Debug\MetallicRtxcrSample.exe
```

For a one-frame Vulkan integration check:

```powershell
build\Source\Debug\MetallicRtxcrSample.exe --smoke-test
```

Single-config generators place the executable directly under `build\Source`.
External checkouts can instead be selected with `RTXCR_ROOT` and
`RTXCR_ASSETS_ROOT` at CMake configure time.

## Integration path

The default sample loads
`External/RTXCR-Assets/Claire/ponyTail_15vtx.gltf`. This is the real, reduced
Claire reference groom: 30,108 line endpoints form 15,054 segments. The importer
reads its `_RADIUS` attribute and `NV_materials_hair` extension, then calls the
upstream `rtxcr::geometry::convertToDisjointOrthogonalTriangleStrips` function.
The result contains 180,648 vertices and 60,216 triangles suitable for Metallic's
existing Vulkan triangle BLAS path.

`ScenePathTrace.slang` keeps authored/world-space normal and tangent data stable
while constructing the hair interaction frame. It evaluates and importance
samples the upstream RTXCR Chiang BSDF using the material values authored in the
Claire glTF. The sample uses the repository's
`EnvironmentMaps/studio_small_09_1k.hdr` reference environment.

Generated meshlet data is written under `.cache/scenes/rtxcr-assets`, not beside
the licensed source asset, so running the sample does not dirty the asset
submodule.

This first reference-asset integration deliberately uses DOTS over the portable
triangle acceleration-structure path. RTXCR hardware Linear-Swept Sphere
intersection, animation/morph targets, and the full Claire body/skin scene remain
future work.

## NVIDIA notice

This software contains source code provided by NVIDIA Corporation.

The NVIDIA sources and assets remain in their separately licensed submodules.
Any redistribution must satisfy the applicable SDK and asset licenses, including
the source-notice and application-distribution requirements.

## Focused validation

```powershell
build\tests\Debug\MetallicSceneTests.exe `
    --gtest_filter=SceneImport.RtxcrClairePonytailDots
build\tests\Debug\MetallicRhiTests.exe --filter rtxcr
```

For a single-config build, omit the `Debug` path component.
