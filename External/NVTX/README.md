# NVTX (NVIDIA Tools Extension Library) v3

Vendored header-only dependency, pinned to the official release tag
[`v3.6.0`](https://github.com/NVIDIA/NVTX/releases/tag/v3.6.0)
(commit `029d6076946076d5af0196d59e8e0e7bcd37c4d9`) of
<https://github.com/NVIDIA/NVTX>.

Only the C/C++ header subset under `c/include/nvtx3/` is vendored, exactly as
published upstream, plus the upstream `LICENSE.txt` (Apache License v2.0 with
LLVM Exceptions). NVTX v3 requires no import library on any platform: on
Windows/MSVC the link-once globals use `__declspec(selectany)`, so simply
including `<nvtx3/nvToolsExt.h>` and adding `c/include` to the include path is
enough. See the official integration notes at <https://nvidia.github.io/NVTX/>.

`cmake/SetupNsight.cmake` picks this directory up automatically; set
`METALLIC_NVTX_ROOT` to use a different NVTX checkout instead.

To update, copy the new release's `c/include/nvtx3/` over this directory and
update the version above.
