# CPU AABB kernels

`MetallicCPUKernels` is independent of TaskGraph, Scene, RenderGraph and Vulkan.
It owns no worker threads. `transformAABBs()` transforms valid AABBs using
row-major affine 3x4 matrices in separate contiguous arrays. A caller can invoke
it on each range supplied by `parallelFor`; input and output storage must remain
alive until that range operation joins.

The three implementations use the same interval-bounds algorithm. `Scalar`
disables loop vectorization for a fair code-generation baseline;
`AutoVectorized` uses ordinary optimized C++; `ISPC` uses an exported SPMD kernel
with no `launch` or `sync`. The tests use an independent double-precision
eight-corner oracle. This does not replace the scene's existing bounds algorithm
or establish that repacking scene data into SoA improves scene-update time.
`Automatic` retains the C++ auto-vectorized baseline; ISPC is selected explicitly
for comparison, rather than enabled on the assumption that a wider ISA is faster.

Inputs must contain ordered min/max values and finite transforms, with finite
intermediates. Invalid bounds and NaN/Inf values are outside this kernel's input
contract and must be handled at the caller's input boundary. Results use the
scale-sensitive absolute tolerance specified in `AABBTransform.h`; bitwise
equality and conservative outward rounding are not promised. This distinction
matters before adopting the result for a visibility test.

Output arrays must not overlap inputs or each other. Range endpoints are
unsigned 32-bit values; empty ranges require no storage. ISPC's lane loop and
`--addressing=64` use 64-bit arithmetic so a tail close to `UINT32_MAX` cannot
wrap and byte offsets larger than 4 GiB retain their high bits. Nonzero starts,
partial four/eight-lane tails and output sentinels are exercised by tests.

## Build

The regular build includes this target. `METALLIC_ENABLE_ISPC=ON` probes an
installed compiler; a missing compiler keeps the C++ fallback available. Use
`-DMETALLIC_ENABLE_ISPC=OFF` to build that path explicitly. No compiler is
downloaded automatically and no host `/arch:AVX2` flag is added.

The default `METALLIC_ISPC_INSTRUCTION_SETS` is
`sse4.2-i32x4;avx2-i32x8`. ISPC's generated dispatcher selects the supported
variant. The host checks the minimum compiled ISA first, falling back to C++
on unsupported CPUs rather than entering a dispatcher with no usable variant.
For the AVX2-only diagnostic build, that gate checks CPUID and XCR0 OS state.
Single-target SSE4.2 or AVX2 builds allow separate ISA measurements.
An ISPC coherent tail branch provides an unmasked full-gang path and a masked
final gang. The ordinary C++ kernel needs no minimum SIMD width or alignment.

Tests and the one-thread benchmark can also be configured without the renderer:

```powershell
cmake -S tests/cpu -B build-cpu-kernels -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release
cmake --build build-cpu-kernels --target MetallicCPUKernelTests MetallicCPUKernelBench
ctest --test-dir build-cpu-kernels --output-on-failure
build-cpu-kernels/MetallicCPUKernelBench.exe --count 65536 --samples 31
```

Standalone configuration uses `find_package(GTest)` or the repository's locked
GoogleTest source. `METALLIC_GTEST_SOURCE` can point at an existing checkout.
`CMAKE_ISPC_COMPILER` can select a user-managed ISPC compiler. Set
`METALLIC_CPU_KERNEL_VECTOR_REPORT=ON` for C++ vectorization diagnostics.

The benchmark emits raw CSV, using deterministic seed `0x73b01`, one caller
thread, two untimed warmup rounds and rotated backend order. `first_call` means
the first invocation of each backend in the process; it does not claim a cold
OS page cache or cold CPU caches. `warm` samples exclude allocation, data setup
and AoS packing. A separately labeled `aos-to-soa` row records repacking time;
allocation and output repacking are still excluded. These kernel timings do
not measure scheduling gain or whole-scene performance.
The timed kernel's output is consumed using a front/middle/back checksum from
each output stream; numerical correctness is established by the separate oracle
and tail tests, not by treating that sampled checksum as a full-array hash.

`MeasureCPUKernels.py` can rotate separate executable configurations across runs
and write raw samples, nearest-rank percentile summaries and source/binary
SHA256 metadata. For example:

```powershell
python -B tests/cpu/MeasureCPUKernels.py `
  --executable multi=build-cpu-kernels/MetallicCPUKernelBench.exe `
  --output build-cpu-kernels/evidence --runs 3 --samples 31
```

The regular renderer test build also adds a TaskGraph producer/range/consumer
test when `MetallicRuntimeTask` exists. It checks 1/2/4/8 workers, all three
partition strategies and every kernel backend, while standalone kernel builds
remain independent of the task runtime.
