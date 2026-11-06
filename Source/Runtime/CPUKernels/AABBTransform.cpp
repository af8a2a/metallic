#include "Runtime/CPUKernels/AABBTransformInternal.h"

#include <cassert>

#if METALLIC_HAS_ISPC
#include "AABBTransform_ispc.h"
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace metallic::cpu {
namespace {

bool supportsISPC() noexcept
{
#if METALLIC_HAS_ISPC
    static const bool supported = [] {
#if defined(_MSC_VER)
        int registers[4];
        __cpuid(registers, 1);
#if METALLIC_ISPC_REQUIRES_AVX2
        constexpr int kAVXAndOSXSAVE = (1 << 28) | (1 << 27);
        if ((registers[2] & kAVXAndOSXSAVE) != kAVXAndOSXSAVE || (_xgetbv(0) & 6) != 6) {
            return false;
        }
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
#else
        return (registers[2] & (1 << 20)) != 0;
#endif
#elif defined(__GNUC__) || defined(__clang__)
#if METALLIC_ISPC_REQUIRES_AVX2
        return static_cast<bool>(__builtin_cpu_supports("avx2"));
#else
        return static_cast<bool>(__builtin_cpu_supports("sse4.2"));
#endif
#else
        return false;
#endif
    }();
    return supported;
#else
    return false;
#endif
}

} // namespace

bool cpuKernelBackendAvailable(CPUKernelBackend backend) noexcept
{
    return backend != CPUKernelBackend::ISPC || supportsISPC();
}

const char* cpuKernelBackendName(CPUKernelBackend backend) noexcept
{
    switch (backend) {
    case CPUKernelBackend::Automatic: return "automatic";
    case CPUKernelBackend::Scalar: return "scalar-interval";
    case CPUKernelBackend::AutoVectorized: return "cpp-auto-vectorized";
    case CPUKernelBackend::ISPC:
#if METALLIC_HAS_ISPC
        if (supportsISPC()) {
            return ispc::metallicAABBTransformISA() == 2 ? "ispc-avx2" : "ispc-sse4.2";
        }
#endif
        return "ispc-unavailable";
    }
    return "unknown";
}

CPUKernelBackend transformAABBs(const AABBTransformInput& input,
    const AABBTransformOutput& output, uint32_t begin, uint32_t end, CPUKernelBackend backend)
{
    assert(begin <= end);
    if (backend == CPUKernelBackend::Scalar) {
        detail::transformAABBsScalar(input, output, begin, end);
        return CPUKernelBackend::Scalar;
    }
#if METALLIC_HAS_ISPC
    if (backend == CPUKernelBackend::ISPC && supportsISPC()) {
        ispc::metallicTransformAABBs(input.minimum[0], input.minimum[1], input.minimum[2],
            input.maximum[0], input.maximum[1], input.maximum[2],
            input.affine[0], input.affine[1], input.affine[2], input.affine[3],
            input.affine[4], input.affine[5], input.affine[6], input.affine[7],
            input.affine[8], input.affine[9], input.affine[10], input.affine[11],
            output.minimum[0], output.minimum[1], output.minimum[2],
            output.maximum[0], output.maximum[1], output.maximum[2], begin, end);
        return CPUKernelBackend::ISPC;
    }
#endif
    detail::transformAABBsAutoVectorized(input, output, begin, end);
    return CPUKernelBackend::AutoVectorized;
}

} // namespace metallic::cpu
