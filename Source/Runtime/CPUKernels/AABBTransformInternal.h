#pragma once

#include "Runtime/CPUKernels/AABBTransform.h"

namespace metallic::cpu::detail {

void transformAABBsScalar(const AABBTransformInput& input,
    const AABBTransformOutput& output, uint32_t begin, uint32_t end);
void transformAABBsAutoVectorized(const AABBTransformInput& input,
    const AABBTransformOutput& output, uint32_t begin, uint32_t end);

} // namespace metallic::cpu::detail
