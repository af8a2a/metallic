#pragma once

#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GAPI/RHI.h"

#include <memory>
#include <array>
#include <span>
#include <string>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::render {

struct VirtualShadowMapCasterResources;

enum class VirtualShadowMapLightType : uint32_t { Directional, Spot, PointFace };
enum class VirtualShadowMapFilter : uint32_t { Hard, PCF, SMRT };

// Point faces use +X, -X, +Y, -Y, +Z, -Z. Position and the complete light
// projection identify depth content; source identities belong to the manager.
struct VirtualShadowMapLight
{
    VirtualShadowMapLightType type = VirtualShadowMapLightType::Directional;
    std::array<float, 3> position{};
    std::array<float, 3> direction{0, -1, 0};
    float outerConeAngle = 0.7853981634f;
    float range = 0; // Zero means unbounded influence; scene bounds determine far.
    float nearPlane = 0.001f;
    float angularRadius = 0; // Directional source half-angle in radians.
    float sourceRadius = 0; // Local disk source radius in world meters.
    uint32_t face = 0;
    bool enabled = true;
};

// One projection view: directional clipmaps, a spot frustum or one point face.
struct VirtualShadowMapSettings
{
    uint32_t pageSize = 128;
    uint32_t virtualPagesPerAxis = 16;
    uint32_t clipmapCount = 4;
    uint32_t physicalPageCount = 256;
    uint32_t maxWorkItems = 65535;
    float firstClipmapExtent = 4.0f; // Half-width in world meters, doubles per level.
    float shadowBias = 0.002f; // World meters toward the light.
    VirtualShadowMapFilter filter = VirtualShadowMapFilter::Hard;
    float pcfRadius = 1.0f; // Radius in selected virtual texels.
    uint32_t smrtRayCount = 8;
    uint32_t smrtSamplesPerRay = 8;
    float smrtMaxRayLength = 10.0f; // World meters; local rays also stop at the source.
    bool receiverFootprint = true; // Filtered queries select a level from the receiver pixel footprint.
    bool enabled = true;
    bool cacheEnabled = false;
    bool forceRedraw = false;
};

struct VirtualShadowMapResult
{
    Texture* texture = nullptr;
    TextureView* shadow = nullptr; // Linear visibility. Only Hard is compatible with legacy sqrt encoding.
    Texture* debugTexture = nullptr;
    TextureView* debug = nullptr;
    Buffer* statistics = nullptr;
    Texture* atlas = nullptr;
    TextureView* atlasView = nullptr;
    Buffer* pageTable = nullptr;
    Buffer* projections = nullptr;
    uint32_t physicalPagesX = 0;
    uint32_t clipmapCount = 0;
    uint32_t pageSize = 0;
};

// A complete coarse projection plus directional clipmaps or local mip levels.
// Physical page zero is reserved for complete coarse coverage. Fine allocation
// failure or work overflow queries that rendered coarse page, never uninitialized depth.
class VirtualShadowMaps
{
public:
    VirtualShadowMaps();
    ~VirtualShadowMaps();
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false);
    Result<VirtualShadowMapResult> record(Device& device, CommandBuffer& commands,
        TextureView& depth, const ViewConstants& view, const VirtualShadowMapLight& light,
        const VirtualShadowMapCasterResources& casters,
        std::span<const scene::RenderMaterial> materials, std::span<TextureView* const> alphaTextures,
        const VirtualShadowMapSettings& settings, std::string& log, bool deferProjection = false);

    // A filtered point light records all six faces before projecting any face.
    // The ordered face results keep each tap's page table, projection and atlas
    // explicit; no tap may wrap into an unrelated physical atlas tile.
    Result<> project(CommandBuffer& commands, std::span<const VirtualShadowMapResult> pointFaces, std::string& log);

private:
    struct Programs;
    struct State;
    struct Cache;
    std::shared_ptr<Programs> programs_;
    std::shared_ptr<State> state_;
    std::shared_ptr<Cache> cache_;
    std::shared_ptr<Cache> pending_;
};

} // namespace metallic::render
