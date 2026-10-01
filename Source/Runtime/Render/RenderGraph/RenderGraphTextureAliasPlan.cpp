#include "Runtime/Render/RenderGraph/RenderGraphTextureAliasPlan.h"

namespace metallic::render::detail {

Result<GraphTextureAliasPlan> buildGraphTextureAliasPlan(size_t passCount,
    std::span<const GraphTextureAliasCandidate> candidates,
    std::span<const GraphTextureAliasDependency> dependencies)
{
    return buildGraphResourceAliasPlan(passCount, candidates, dependencies);
}

} // namespace metallic::render::detail
