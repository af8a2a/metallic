#include "Runtime/CPUKernels/AABBTransformInternal.h"

#if defined(_MSC_VER)
#define METALLIC_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define METALLIC_RESTRICT __restrict__
#else
#define METALLIC_RESTRICT
#endif

namespace metallic::cpu::detail {
namespace {

void transformAxis(const float* METALLIC_RESTRICT minX, const float* METALLIC_RESTRICT minY,
    const float* METALLIC_RESTRICT minZ, const float* METALLIC_RESTRICT maxX,
    const float* METALLIC_RESTRICT maxY, const float* METALLIC_RESTRICT maxZ,
    const float* METALLIC_RESTRICT m0, const float* METALLIC_RESTRICT m1,
    const float* METALLIC_RESTRICT m2, const float* METALLIC_RESTRICT translation,
    float* METALLIC_RESTRICT outMin, float* METALLIC_RESTRICT outMax, uint32_t begin, uint32_t end)
{
    for (uint32_t index = begin; index < end; ++index) {
        const float x0 = m0[index] * minX[index], x1 = m0[index] * maxX[index];
        const float y0 = m1[index] * minY[index], y1 = m1[index] * maxY[index];
        const float z0 = m2[index] * minZ[index], z1 = m2[index] * maxZ[index];
        outMin[index] = (x0 < x1 ? x0 : x1) + (y0 < y1 ? y0 : y1) + (z0 < z1 ? z0 : z1) + translation[index];
        outMax[index] = (x0 > x1 ? x0 : x1) + (y0 > y1 ? y0 : y1) + (z0 > z1 ? z0 : z1) + translation[index];
    }
}

} // namespace

void transformAABBsAutoVectorized(const AABBTransformInput& input,
    const AABBTransformOutput& output, uint32_t begin, uint32_t end)
{
    // Hoist the SoA pointers for each row; restrict follows the non-overlapping
    // output contract and lets the compiler vectorize contiguous index batches.
    for (uint32_t row = 0; row < 3; ++row) {
        transformAxis(input.minimum[0], input.minimum[1], input.minimum[2],
            input.maximum[0], input.maximum[1], input.maximum[2],
            input.affine[row * 4], input.affine[row * 4 + 1], input.affine[row * 4 + 2],
            input.affine[row * 4 + 3], output.minimum[row], output.maximum[row], begin, end);
    }
}

} // namespace metallic::cpu::detail
