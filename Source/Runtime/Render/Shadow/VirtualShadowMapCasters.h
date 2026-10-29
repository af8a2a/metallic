#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Scene/Scene.h"

namespace metallic::render {

struct MeshletStreamShadowGeometryView;

// Matches the fine-work compute group. Stream candidates retain their native
// 128-triangle cluster; its bounds serve two adjacent groups.
inline constexpr uint32_t kVirtualShadowMapCasterClusterSize = 64;

struct VirtualShadowMapCasterSettings
{
    bool tessellationEnabled = false; // Must match the receiver producer's explicit mode.
    // Independent shadow-domain subdivision. Uses the raster displacement
    // evaluator, but does not borrow the primary camera's visible patch list.
    uint32_t tessellationFactor = 4;
    // Uniform dyadic patch splits, independent of the primary camera. Depth 2
    // emits sixteen times as many microtriangles as depth 0 at the same factor.
    uint32_t subdivisionDepth = 0;
    // Empty or one published height view per material, already in ShaderRead.
    std::span<TextureView* const> displacementTextures;
    // Immutable physical image/mip publication, including alpha and height.
    // Zero is appropriate only when no shadow material samples a texture.
    uint64_t texturePublicationRevision = 0;
};

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
    bool displacedGeometry = false;
    uint64_t sceneIdentity = 0;
    uint64_t structuralRevision = 0;
    // Immutable terminal geometry and shadow material texture publication.
    uint64_t dependencyKey = 0;
    // Sorted by id. Hidden/deleted casters are absent so the cache can invalidate
    // their previous bounds when comparing committed snapshots.
    std::vector<VirtualShadowMapCasterState> states;
};

// Captures the complete scene, independent of primary-view visibility. Resident
// scenes upload world-space triangles and conservative 64-slot cluster bounds;
// streamed scenes decode the complete locked terminal cut through the shared
// page pool. Uploads are immutable and retained by this recording. No shadow
// cache or stream traversal is performed.
Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasters(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, std::string& log,
    const VirtualShadowMapCasterSettings& settings = {});

// Pass-owned resident uploads, independent of the view and frame. The cache
// must be destroyed before its device, like the pass's other GPU resources.
// Streamed preparation remains per recording to preserve residency and sync.
class VirtualShadowMapCasterCache
{
public:
    Result<VirtualShadowMapCasterResources> prepare(
        Device& device, CommandBuffer& commands, const scene::Scene& scene,
        const MeshletStreamShadowGeometryView* stream, std::string& log,
        const VirtualShadowMapCasterSettings& settings = {});

private:
    struct Resident;
    std::shared_ptr<const Resident> resident_;
};

} // namespace metallic::render
