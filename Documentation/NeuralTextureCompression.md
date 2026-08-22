# Neural Texture Compression

Metallic can sample NVIDIA Neural Texture Compression (NTC) texture sets directly in the material visualization, path-tracing, and RTXDI scene shaders. At runtime it keeps the compact latent textures and inference weights resident instead of uploading the replaced RGBA8 mip chains. On supported NVIDIA GPUs it prefers FP8 inference through `VK_NV_cooperative_vector`; other devices retain the Generic INT8/DP4a path.

## Build setup

NTC support is enabled by default when `RTXNTC-Library` can be found in one of these locations:

- `-DRTXNTC_ROOT=<RTXNTC checkout or RTXNTC-Library path>`
- `External/RTXNTC-Library`
- a sibling checkout at `../RTXNTC/libraries/RTXNTC-Library`

For the local reference checkout:

```powershell
cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON -DRTXNTC_ROOT=E:/RTXNTC
cmake --build build --target MetallicMaterialVisualizationSample MetallicPathTracingSample MetallicRtxdiSample --config Debug
```

Use `-DMETALLIC_ENABLE_NTC=OFF` to build without LibNTC. CUDA, DX12, and LibNTC's prebuilt decompression shaders are disabled. The LibNTC Vulkan backend is enabled for cooperative-vector capability discovery and weight-layout conversion; inference itself still runs through Metallic's Vulkan RHI and Slang shaders. A Vulkan SDK that provides `vulkan/vulkan.hpp` is required when NTC is enabled.

## Asset contract

The glTF file must expose each neural source as an image with MIME type `image/vnd-nvidia.ntc` or a `.ntc` URI. Logical glTF textures select that source and its output channels through `NV_texture_swizzle`:

```json
{
  "extensions": {
    "NV_texture_swizzle": {
      "options": [
        { "source": 5, "channels": [0, 1, 2, 3] }
      ]
    }
  }
}
```

`source` is the glTF image index. `channels` contains one to four NTC output-channel indices in RGBA order. The current implementation accepts channel indices 0-15 and uses the first valid NTC option. A texture without a valid neural option continues through the conventional image path.

## Runtime behavior

For every referenced NTC texture set, Metallic:

1. Reads metadata, packed BGRA4 latent mip arrays, constants, and the best supported inference weights through LibNTC.
2. Selects CoopVec FP8 when both the Vulkan device and texture set support it; otherwise selects Generic INT8 when integer dot products are available.
3. Converts CoopVec weights to the device-optimized layout with `vkCmdConvertCooperativeVectorMatrixNV` during the scene upload. This upload uses the graphics queue because the conversion command is not valid on a transfer-only queue.
4. Uploads the latent arrays and shared inference buffers with the normal asynchronous scene upload batches.
5. Omits replaced conventional material textures from GPU residency.
6. Reconstructs requested material channels at the shader sample site.

The implementation currently supports at most 64 NTC texture sets per scene and reserves descriptor bindings 29-33 for latent arrays, constants, weights, per-set offsets/modes, and the latent sampler. Cooperative-vector shaders are compiled with Slang's `spvCooperativeVectorNV` capability. Vulkan 1.3 integer dot-product support is required only for texture sets that fall back to Generic INT8.

If LibNTC is unavailable, the GPU supports neither cooperative vectors nor integer dot products, an NTC stream cannot be opened, or metadata/inference payload validation fails, the loader logs the reason and keeps the conventional textures. A malformed neural reference therefore does not make an otherwise valid scene unusable.

## Running and validating

All three sample executables accept a scene override:

```powershell
build\Source\MetallicMaterialVisualizationSample.exe --smoke-test --scene <scene.gltf>
build\Source\MetallicPathTracingSample.exe --smoke-test --scene <scene.gltf>
build\Source\MetallicRtxdiSample.exe --smoke-test --scene <scene.gltf>
```

The NTC resource log reports texture-set count, the CoopVec/Generic split, replaced logical textures, conventional byte estimate, resident neural bytes, saved bytes, and reduction percentage. On an RTX 5070 Ti with the RTXNTC FlightHelmet scene, all five texture sets selected CoopVec FP8. Reported material-texture residency changed from 285,212,652 bytes (272.0 MiB) to 11,969,568 bytes (11.4 MiB), saving 273,243,084 bytes (260.6 MiB, 95.8%). This measures texture payloads replaced by NTC; acceleration structures, geometry, render targets, and other renderer allocations are outside that figure.

## Current limits

- CoopVec FP8 is selected per texture set. Mixed scenes can use CoopVec FP8 and Generic INT8 in the same packed buffers and shader permutation.
- Material visualization and RTXDI currently reconstruct mip 0. The path tracer selects neural mip levels from its existing ray-cone/STF sampling path.
- NTC is integrated into the shared ray-query scene resources used by material visualization, path tracing, and RTXDI. The separate GPU-driven raster texture system is unchanged.
- NTC encoding and glTF conversion are asset-pipeline tasks and are not performed at runtime.
