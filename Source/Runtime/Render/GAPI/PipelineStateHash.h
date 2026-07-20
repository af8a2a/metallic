#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

namespace metallic::render::detail {

uint64_t shaderContentHash(const ShaderModuleDesc& desc);
uint64_t graphicsPipelineStateHash(const GraphicsPipelineDesc& desc);
uint64_t computePipelineStateHash(const ComputePipelineDesc& desc);

} // namespace metallic::render::detail
