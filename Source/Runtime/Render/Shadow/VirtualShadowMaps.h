#pragma once

#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GAPI/RHI.h"

#include <memory>
#include <span>
#include <string>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::render {

struct VirtualShadowMapCasterResources;

// One directional-light view. P1 keeps caching off; P2 enables it per celestial.
struct VirtualShadowMapSettings
{
    uint32_t pageSize = 128;
    uint32_t virtualPagesPerAxis = 16;
    uint32_t clipmapCount = 4;
    uint32_t physicalPageCount = 256;
    uint32_t maxWorkItems = 65535;
    float firstClipmapExtent = 4.0f; // Half-width in world meters, doubles per level.
    float shadowBias = 0.002f; // World meters toward the light.
    bool enabled = true;
    bool cacheEnabled = false;
    bool forceRedraw = false;
};

struct VirtualShadowMapResult
{
    Texture* texture = nullptr;
    TextureView* shadow = nullptr; // Hard 0/1: also compatible with legacy sqrt encoding.
    Texture* debugTexture = nullptr;
    TextureView* debug = nullptr;
    Buffer* statistics = nullptr;
    Texture* atlas = nullptr;
    TextureView* atlasView = nullptr;
    Buffer* pageTable = nullptr;
    Buffer* projections = nullptr;
};

// P1 full-scene coarse projection plus nested camera-centered clipmaps.
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
        const VirtualShadowMapSettings& settings, std::string& log);

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
