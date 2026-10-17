#pragma once

// Shared wire layout for the ComputeProgram resource and constant spans.
// It is either uploaded behind ParameterRoot or pushed directly as 24 bytes.
#ifdef __cplusplus
#include "Runtime/Render/Core/ShaderResourceABI.h"
#include <cstddef>
namespace metallic::render {
#define METALLIC_COMPUTE_ROOT_PUBLIC
#define METALLIC_COMPUTE_ROOT_SPAN GPUBufferSpan
#else
#define METALLIC_COMPUTE_ROOT_PUBLIC public
#define METALLIC_COMPUTE_ROOT_SPAN BufferSpan<uint>
#endif

METALLIC_COMPUTE_ROOT_PUBLIC struct ComputeResourceParameters
{
    METALLIC_COMPUTE_ROOT_PUBLIC METALLIC_COMPUTE_ROOT_SPAN resources;
    METALLIC_COMPUTE_ROOT_PUBLIC METALLIC_COMPUTE_ROOT_SPAN constants;
};

#undef METALLIC_COMPUTE_ROOT_PUBLIC
#undef METALLIC_COMPUTE_ROOT_SPAN

#ifdef __cplusplus
static_assert(sizeof(ComputeResourceParameters) == 24);
static_assert(alignof(ComputeResourceParameters) == 4);
static_assert(offsetof(ComputeResourceParameters, resources) == 0);
static_assert(offsetof(ComputeResourceParameters, constants) == 12);
static_assert(std::is_trivially_copyable_v<ComputeResourceParameters>);
} // namespace metallic::render
#endif
