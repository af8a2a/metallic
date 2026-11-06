#pragma once

#include <array>
#include <cstdint>

namespace metallic::cpu {

enum class CPUKernelBackend : uint8_t {
    Automatic,
    Scalar,
    AutoVectorized,
    ISPC,
};

// Each pointer addresses a separate contiguous array with at least end entries.
// affine is row-major 3x4: m00, m01, m02, m03, m10, ..., m23.
// Inputs are finite affine transforms and valid ordered AABBs; intermediates
// must remain finite. The caller establishes these conditions at its input
// boundary before dispatching this internal numeric kernel.
struct AABBTransformInput {
    std::array<const float*, 3> minimum;
    std::array<const float*, 3> maximum;
    std::array<const float*, 12> affine;
};

struct AABBTransformOutput {
    std::array<float*, 3> minimum;
    std::array<float*, 3> maximum;
};

// begin <= end. Output arrays must not overlap one another or the input arrays.
// Distinct calls may write disjoint ranges. No allocation, thread launch or join
// takes place here; the caller owns range scheduling and input/output lifetimes.
// All backends use interval bounds; the scalar baseline disables loop SIMD so
// comparisons isolate code generation rather than a different bounds algorithm.
// For each coordinate, absolute error against the double-precision
// corner result is at most 1e-6 + 8*FLT_EPSILON*(sum(abs(mj)*max(abs(lo_j),
// abs(hi_j))) + abs(translation)); bitwise scene-bounds identity is not promised.
// Automatic retains the measured C++ baseline. ISPC is an explicit A/B option;
// an unavailable ISPC request falls back to AutoVectorized. The return value
// names what ran, without changing the caller's worker or ISA requirements.
CPUKernelBackend transformAABBs(
    const AABBTransformInput& input,
    const AABBTransformOutput& output,
    uint32_t begin,
    uint32_t end,
    CPUKernelBackend backend = CPUKernelBackend::Automatic);

bool cpuKernelBackendAvailable(CPUKernelBackend backend) noexcept;
const char* cpuKernelBackendName(CPUKernelBackend backend) noexcept;

} // namespace metallic::cpu
