#language slang 2026
module InlineComputeRoot;

import ComputeResourceTypes;

#ifndef METALLIC_INLINE_COMPUTE_ROOT
#error InlineComputeRoot requires METALLIC_INLINE_COMPUTE_ROOT=1
#elif METALLIC_INLINE_COMPUTE_ROOT != 1
#error InlineComputeRoot requires METALLIC_INLINE_COMPUTE_ROOT=1
#endif

namespace Metallic
{
// Direct ComputeProgram transport: two descriptor-relative spans, no root buffer.
[[vk::push_constant]] public ConstantBuffer<ComputeResourceParameters> gInlineComputeRoot;
} // namespace Metallic
