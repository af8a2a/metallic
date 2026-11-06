# CPU AABB kernel evidence

This experiment isolates one-thread CPU kernel code generation from the range
scheduler. `Automatic` retains the C++ auto-vectorized implementation. ISPC is
an explicit comparison backend; these results do not change scene bounds,
culling, frame scheduling or the renderer's ISA requirements.

## Workload and provenance

The input is a deterministic (`0x73b01`) SoA batch of valid AABBs and affine
3x4 transforms. Scalar C++, auto-vectorized C++, and ISPC use the same interval
bounds algorithm; an independent double-precision eight-corner oracle checks
numeric correctness. Finite inputs/intermediates, ordered bounds and disjoint
output storage are the caller's contract. NaN/Inf, invalid bounds and
conservative outward rounding are outside this experimental kernel contract.

The host is an AMD Ryzen 7 7800X3D (16 logical processors), identified from the
read-only Windows hardware registry; the exact key is recorded in metadata.
Windows reports build 10.0.26300. The benchmark has one caller thread and no
worker pool, ISPC `launch/sync`,
Vulkan instance or validation layer. It uses MSVC 19.44.35229 Release `/O2` with
the ordinary x64 host baseline, and ISPC 1.32.0 / LLVM 23.1.2. ISPC variants are
SSE4.2 (four lanes), AVX2 (eight lanes), and generated SSE4.2/AVX2 runtime dispatch.
ISPC FMA contraction is disabled and addressing is explicitly 64-bit so large
valid array offsets retain their high bits. No AVX-512 build was evaluated.

Three processes per configuration and size each record one `first_call`, two
untimed warmup rounds and 31 warm samples. Backend order rotates inside each
process, and executable order rotates between runs. Sizes are 31, 65,536 and
1,048,576 AABBs. Other task benchmarks, builds and test processes are paused
during the measurement window; normal OS activity remains. There is no affinity
pinning or cache flush. `first_call` is a first invocation on initialized arrays,
not a claim of cold CPU/OS caches.

Allocation, deterministic input setup, output validation and packing are outside
kernel timing. A separately labeled `aos-to-soa` sample measures the simple
18-float AoS-to-SoA repacking loop; it excludes allocation and output repacking.
The sampled output checksum is not a full-array validation hash. Numerical
oracle and range tests are run separately.

Final raw samples, percentile summaries, CPU identity source, source SHA256s and
executable SHA256s are recorded by `tests/cpu/MeasureCPUKernels.py` in
[Raw.csv](Raw.csv), [Summary.csv](Summary.csv) and [Metadata.json](Metadata.json).
Source/binary hashes are checked before and after sampling. Summaries aggregate
93 warm samples per configuration/backend/size, with nearest-rank P95/P99.

## Results

The final run contains 3,456 raw samples (3,348 warm and 108 first-call samples).
The runtime-dispatch build selects AVX2 on this host. Warm timings below are in
microseconds; each row aggregates 93 samples.

| AABBs | Backend, runtime-dispatch build | P50 us | P95 us | P99 us |
| ---: | --- | ---: | ---: | ---: |
| 65,536 | Scalar interval | 295.4 | 479.4 | 572.3 |
| 65,536 | C++ auto-vectorized | 132.1 | 247.2 | 310.3 |
| 65,536 | ISPC AVX2 | 76.9 | 114.7 | 213.2 |
| 1,048,576 | Scalar interval | 11,341.4 | 13,272.1 | 15,708.6 |
| 1,048,576 | C++ auto-vectorized | 3,958.1 | 5,412.7 | 6,978.7 |
| 1,048,576 | ISPC AVX2 | 5,038.8 | 6,113.7 | 12,172.6 |

For 65,536 elements, C++ auto vectorization is 2.24x faster than the scalar
interval baseline by P50, and ISPC AVX2 is another 1.72x faster than C++ auto.
At 1,048,576 elements, C++ auto is 2.87x faster than scalar while ISPC AVX2 takes
27% longer than C++ auto. These are one-thread kernel ratios, with no scheduling
gain included. AVX2-only measurements show the same size-dependent result
(77.4 us and 4,683.8 us). SSE4.2-only ISPC records 117.9 us and 11,778.5 us;
its matched C++ auto baselines are 131.9 us and 4,020.4 us. The raw summaries
retain the corresponding first-call and tail distributions for every build.

The 31-element C++/ISPC medians are 0.1 us at the timer's observed quantization;
this case establishes no useful speedup. The runtime-dispatch build's simple
AoS-to-SoA packing medians are 335.0 us (65,536) and 61,141.4 us (1,048,576),
already larger than either kernel. This is a measured packing prototype, not
an optimized transpose or a claim that every production layout needs packing.
Do not add these independently timed samples into an asserted scene speedup.

An initial exploratory ordinary-`if` ISPC candidate was slower than C++ auto
vectorization (65,536 elements: about 275 vs 134 microseconds, runtime AVX2).
Its generated AVX2 code used masked contiguous loads/stores for full gangs as
well as the tail. One small candidate changes that branch to ISPC `cif`; the
final generated code has both unmasked `vmovups` full-gang accesses and a masked
`vmaskmovps` tail path. The final CSV files refer to this candidate and the
final C++ default policy, rather than the exploratory binary.

MSVC reports C5001 for the C++ inner loop, and disassembly contains packed
`mulps/minps/maxps` on XMM registers. Scalar loops report C5002/reason 1400
(`no_vector`). ISPC AVX2 contains packed YMM arithmetic. The inspected ISPC code
uses contiguous memory operations, without gather/scatter instructions.
No hardware SIMD utilization, CPU utilization or cache/performance counters were
collected.
The theoretical lane occupancy for a single 31-element call is 31/32; both
larger sizes are full multiples of four/eight. This is arithmetic lane coverage,
not measured hardware utilization.

Kernel-only measurements and the separately measured repacking prototype do
not establish scene-update, meshlet-cooking, critical-path or FPS gains. They
also do not include a required output conversion or a production consumer's
numeric-margin policy. Retaining the C++ default avoids turning a single-host
kernel experiment into a production backend policy change.

## Correctness and reproduction

All five standalone kernel tests pass in C++-only, runtime-dispatch, SSE4.2-only
and AVX2-only configurations. They cover deterministic random transforms,
reflections, shear, degenerate bounds, signed zero, small/large finite values,
nonzero starts, masked tails, empty ranges and backend reporting. A Windows
sparse-virtual-array test reserves 16 GiB per stream and commits only one page,
then checks real offsets above 4 GiB and a nonempty range ending at UINT32_MAX,
including adjacent output sentinels, separately for each backend.

The task-integrated test executable also passes a sixth test: an asynchronous
TaskGraph producer -> `addParallelFor` -> consumer chain checks complete output
publication for 1/2/4/8 workers, all three partition strategies and every kernel
backend. Standalone kernel tests remain independent of TaskSystem.

Run from an x64 Visual Studio developer shell, with an optional ISPC compiler on
PATH. A missing compiler keeps the C++ build available:

```powershell
cmake -S tests/cpu -B build-cpu-kernels -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DMETALLIC_ENABLE_ISPC=OFF
cmake --build build-cpu-kernels --target MetallicCPUKernelTests MetallicCPUKernelBench
ctest --test-dir build-cpu-kernels --output-on-failure

cmake -S tests/cpu -B build-cpu-kernels-ispc -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DMETALLIC_ENABLE_ISPC=ON `
  '-DMETALLIC_ISPC_INSTRUCTION_SETS=sse4.2-i32x4;avx2-i32x8'
cmake -S tests/cpu -B build-cpu-kernels-sse4 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DMETALLIC_ENABLE_ISPC=ON `
  -DMETALLIC_ISPC_INSTRUCTION_SETS=sse4.2-i32x4
cmake -S tests/cpu -B build-cpu-kernels-avx2 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release -DMETALLIC_ENABLE_ISPC=ON `
  -DMETALLIC_ISPC_INSTRUCTION_SETS=avx2-i32x8
# Build/test each configured target before measurement.
python -B tests/cpu/MeasureCPUKernels.py `
  --executable multi=build-cpu-kernels-ispc/MetallicCPUKernelBench.exe `
  --executable sse4=build-cpu-kernels-sse4/MetallicCPUKernelBench.exe `
  --executable avx2=build-cpu-kernels-avx2/MetallicCPUKernelBench.exe `
  --output build-cpu-kernels/evidence --runs 3 --samples 31
```

The optional compiler was obtained from the official
[ISPC 1.32.0 release](https://github.com/ispc/ispc/releases/tag/v1.32.0), with the
published Windows archive SHA256 checked before extraction. No compiler is
automatically downloaded or globally installed by the build. ISPC's
[user guide](https://ispc.github.io/ispc.html) documents multi-target dispatch,
64-bit addressing and coherent control flow; the host gate falls back to C++
if the minimum compiled ISA is unavailable. AVX2-only builds additionally check
OS XCR0 state. Unsupported-ISA hardware was not available for physical testing.
