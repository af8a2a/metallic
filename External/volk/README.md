# Vendored volk

Source: https://github.com/zeux/volk
Commit: `0dc3ce00bf98b9f0b6fe708ca0f7eb74e2830173` (upstream master)
`VOLK_HEADER_VERSION`: 365

`volk.c`, `volk.h`, and `LICENSE.md` are unmodified upstream files.
Update them together from the same commit and retain the MIT license.

Metallic builds volk through `External/CMakeLists.txt`, using SDL3's bundled
Vulkan headers when available. Extension declarations and loaders are gated by
those headers; updating volk does not enable additional device features.
