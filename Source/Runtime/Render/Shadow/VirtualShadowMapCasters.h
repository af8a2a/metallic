#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Scene/Scene.h"

namespace metallic::render {

struct MeshletStreamShadowGeometryView;

struct VirtualShadowMapCasterState
{
    // Render-node identity is stable within the published structural revision.
    uint64_t id = 0;
    scene::Bounds bounds;
    uint64_t contentKey = 0;
};

struct VirtualShadowMapCasterResources
{
    Buffer* parameters = nullptr;
    uint32_t triangleCapacity = 0;
    scene::Bounds bounds;
    std::vector<ResourceLease> leases;
    std::shared_ptr<void> owner;
    bool streamed = false;
    uint64_t sceneIdentity = 0;
    uint64_t structuralRevision = 0;
    // Immutable terminal geometry publication; the pass combines the current
    // material texture publication when Coverage samples an alpha texture.
    uint64_t dependencyKey = 0;
    // Sorted by id. Hidden/deleted casters are absent so the cache can invalidate
    // their previous bounds when comparing committed snapshots.
    std::vector<VirtualShadowMapCasterState> states;
};

// Captures the complete scene, independent of primary-view visibility. Resident
// scenes upload world-space triangles; streamed scenes decode the complete
// locked terminal cut through the shared page pool. Uploads are immutable and
// retained by this recording. No shadow cache or stream traversal is performed.
Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasters(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log);

} // namespace metallic::render
