#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Scene/Scene.h"

namespace metallic::render {

struct MeshletStreamShadowGeometryView;

struct VirtualShadowMapCasterResources
{
    Buffer* parameters = nullptr;
    uint32_t triangleCapacity = 0;
    scene::Bounds bounds;
    std::vector<ResourceLease> leases;
    std::shared_ptr<void> owner;
    bool streamed = false;
};

// Captures the complete scene, independent of primary-view visibility. Resident
// scenes upload world-space triangles; streamed scenes decode the complete
// locked terminal cut through the shared page pool. Uploads are immutable and
// retained by this recording. No shadow cache or stream traversal is performed.
Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasters(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log);

} // namespace metallic::render
