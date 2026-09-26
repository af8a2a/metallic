# WorkControl shader watch

`ShaderTrace.py run` starts one bounded production observation; `acceptance` runs the fixed P2 schedule;
`verify` replays raw artifacts using the current metallicctl and rechecks readback/source/process evidence.
See [the P2 contract, commands and validation](../../Documentation/AgenticShaderPrintfP2.md).

This is a shader debugger, with `performanceEligible=false`. It does not replace uninstrumented M3 timing
or NvPerf counters. Live collection needs Python plus the existing `psutil` process helper, the built
sample/CLI, the configured Slang runtime, Vulkan validation layer and MiniZorah assets. Offline
verification needs Python and the current built CLI; it does not start a GPU process or rehash scene assets.
