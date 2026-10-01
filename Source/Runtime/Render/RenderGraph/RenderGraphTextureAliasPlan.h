#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphResourceAliasPlan.h"

namespace metallic::render::detail {

// Preserve the texture planner API while buffers and textures use one strict
// happens-before algorithm. Their native compatibility groups remain separate.
inline constexpr size_t kNoGraphTextureAliasSlot = kNoGraphResourceAliasSlot;
using GraphTextureAliasCandidate = GraphResourceAliasCandidate;
using GraphTextureAliasDependency = GraphResourceAliasDependency;
using GraphTextureAliasSlot = GraphResourceAliasSlot;
using GraphTextureAliasHandoff = GraphResourceAliasHandoff;
using GraphTextureAliasPlan = GraphResourceAliasPlan;

Result<GraphTextureAliasPlan> buildGraphTextureAliasPlan(size_t passCount,
    std::span<const GraphTextureAliasCandidate> candidates,
    std::span<const GraphTextureAliasDependency> dependencies);

} // namespace metallic::render::detail
