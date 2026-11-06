#include "Runtime/CPUKernels/AABBTransformInternal.h"

#include <algorithm>

namespace metallic::cpu::detail {

void transformAABBsScalar(const AABBTransformInput& input,
    const AABBTransformOutput& output, uint32_t begin, uint32_t end)
{
#if defined(_MSC_VER) && !defined(__clang__)
#pragma loop(no_vector)
#endif
    for (uint32_t row = 0; row < 3; ++row) {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma loop(no_vector)
#endif
        for (uint32_t index = begin; index < end; ++index) {
            const float x0 = input.affine[row * 4][index] * input.minimum[0][index];
            const float x1 = input.affine[row * 4][index] * input.maximum[0][index];
            const float y0 = input.affine[row * 4 + 1][index] * input.minimum[1][index];
            const float y1 = input.affine[row * 4 + 1][index] * input.maximum[1][index];
            const float z0 = input.affine[row * 4 + 2][index] * input.minimum[2][index];
            const float z1 = input.affine[row * 4 + 2][index] * input.maximum[2][index];
            output.minimum[row][index] = std::min(x0, x1) + std::min(y0, y1) + std::min(z0, z1) +
                input.affine[row * 4 + 3][index];
            output.maximum[row][index] = std::max(x0, x1) + std::max(y0, y1) + std::max(z0, z1) +
                input.affine[row * 4 + 3][index];
        }
    }
}

} // namespace metallic::cpu::detail
