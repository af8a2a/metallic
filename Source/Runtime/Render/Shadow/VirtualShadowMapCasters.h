#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"
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
    uint64_t transformKey = 0;
    uint64_t materialKey = 0;
    // Independent of authored content: a cut change invalidates depth without
    // classifying an otherwise static caster as moving. Zero is the base cut.
    uint64_t lodKey = 0;
    // Complete contiguous geometry slot range, including displaced expansion
    // and the streamed terminal clusters' 128-slot padding.
    uint32_t triangleFirst = 0;
    uint32_t triangleCount = 0;
};

struct VirtualShadowMapShadowLODDiagnostics
{
    uint32_t selectedClusters = 0;
    uint32_t selectedTriangleSlots = 0; // Stream clusters have 128 padded slots.
    uint32_t requestedPages = 0;
    uint32_t blockedPages = 0;
    uint32_t fallbackCasters = 0;
    uint32_t protectedCasters = 0; // MASK and displaced retain their base geometry.
    uint32_t testedGroups = 0;
    std::vector<uint32_t> selectedLevels;
    // (render node, primitive-local group, local cluster), including base cuts.
    std::vector<std::array<uint32_t, 3>> selectedClusterIDs;
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
    uint64_t geometryRevision = 0;
    uint64_t texturePublicationRevision = 0;
    uint64_t tessellationKey = 0;
    // This layer variant's mask and parameters; excludes retained source uploads.
    uint64_t layerBufferBytes = 0;
    VirtualShadowMapShadowLODDiagnostics shadowLOD;
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

// Selects a shared conservative cut for all admitted light projections. The
// source snapshot owns immutable authored keys and complete caster bounds.
// Zero error forces finest available geometry; streaming can retain a complete
// resident ancestor cut while requesting refinement for later frames.
Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapShadowLOD(
    Device& device, CommandBuffer& commands, const scene::Scene& scene,
    const MeshletStreamShadowGeometryView* stream, const VirtualShadowMapCasterResources& base,
    const VirtualShadowMapCasterSettings& settings,
    std::span<const VirtualShadowMapLODProjection> projections, float maxErrorTexels, std::string& log);

// Selects one layer from a complete producer snapshot. IDs are sorted and use
// the source structural revision. Geometry slot IDs/capacity and complete scene
// bounds stay unchanged; states contains only the selected layer's casters.
// The immutable variant retains the source uploads and descriptor leases.
Result<VirtualShadowMapCasterResources> prepareVirtualShadowMapCasterLayer(
    Device& device, CommandBuffer& commands, const VirtualShadowMapCasterResources& source,
    std::span<const uint64_t> dynamicCasterIds, bool staticLayer, std::string& log);

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
