#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render::detail {

uint64_t shaderContentHash(const ShaderModuleDesc& desc);
// PSO identities combine final device shader-module content, entry points,
// explicit state and the versioned implicit RHI layout/state ABI. They exclude
// binary-cache ownership and diagnostic names. Native reuse is device-local.
uint64_t graphicsPipelineStateHash(const GraphicsPipelineDesc& desc);
uint64_t computePipelineStateHash(const ComputePipelineDesc& desc);

} // namespace metallic::render::detail
