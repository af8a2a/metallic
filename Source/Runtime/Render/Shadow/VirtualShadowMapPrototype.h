#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <array>
#include <memory>
#include <span>
#include <string>

namespace metallic::scene { struct RenderMaterial; }

namespace metallic::render {

struct VirtualShadowMapPrototypeVertex
{
    std::array<float, 4> position{0, 0, 0, 1};
    std::array<float, 4> uv{};
};

// World-space geometry, independent of primary-camera visibility.
struct VirtualShadowMapPrototypeTriangle
{
    std::array<VirtualShadowMapPrototypeVertex, 3> vertices;
    uint32_t materialIndex = 0;
    std::array<uint32_t, 3> reserved{};
};
static_assert(sizeof(VirtualShadowMapPrototypeTriangle) == 112);

struct VirtualShadowMapPrototypePage
{
    uint32_t physicalPage = UINT32_MAX;
    uint32_t dirty = 1;
    std::array<uint32_t, 2> reserved{};
};
static_assert(sizeof(VirtualShadowMapPrototypePage) == 16);

// Row-major orthographic world -> Vulkan clip coordinates (w = 1).
// Shadow depth is reversed: near = 1, far = 0, clear = 0.
struct VirtualShadowMapPrototypeProjection
{
    std::array<float, 16> rows{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

struct VirtualShadowMapPrototypeReceiver
{
    std::array<float, 4> positionBias{}; // world xyz, non-negative clip-depth bias
};

enum class VirtualShadowMapPrototypeQueryStatus : uint32_t
{
    Valid = 0,
    Unmapped = 1,
    OutsideProjection = 2,
    WorkOverflow = 3,
};

struct VirtualShadowMapPrototypeQuery
{
    float visibility = 0;
    float depth = 0;
    uint32_t virtualPage = UINT32_MAX;
    VirtualShadowMapPrototypeQueryStatus status = VirtualShadowMapPrototypeQueryStatus::Unmapped;
    uint32_t physicalPage = UINT32_MAX;
    std::array<uint32_t, 3> reserved{};
};
static_assert(sizeof(VirtualShadowMapPrototypeQuery) == 32);

struct VirtualShadowMapPrototypeDesc
{
    uint32_t pageSize = 128;
    uint32_t virtualPagesX = 2;
    uint32_t virtualPagesY = 2;
    uint32_t physicalPagesX = 2;
    uint32_t physicalPagesY = 2;
    uint32_t maxWorkItems = 4096;
};

struct VirtualShadowMapPrototypeScene
{
    std::span<const VirtualShadowMapPrototypeTriangle> triangles;
    std::span<const scene::RenderMaterial> materials;
    // Empty or one optional sampled RGBA alpha texture per material. Views must
    // be in ShaderRead when record executes. Their leases retain allocations.
    std::span<TextureView* const> alphaTextures;
    std::span<const VirtualShadowMapPrototypePage> pages;
    std::span<const VirtualShadowMapPrototypeReceiver> receivers;
    VirtualShadowMapPrototypeProjection projection;
};

// Fixed-page P0 raster prototype; no allocator, scene streaming or temporal cache.
// Caller completes prior GPU use before updating inputs or reinitializing.
// record does not wait. Caller supplies the actual atlas layout, so discarded
// recordings never silently advance a CPU layout/history state.
class VirtualShadowMapPrototype
{
public:
    VirtualShadowMapPrototype();
    ~VirtualShadowMapPrototype();
    Result<> initialize(Device& device, const VirtualShadowMapPrototypeDesc& desc,
        const VirtualShadowMapPrototypeScene& scene, std::string& log);
    // Caller dirties old/new caster footprints after geometry edits, all mapped
    // pages after projection edits, and every newly assigned physical mapping.
    // P0 tests this explicit page input; automatic invalidation belongs to P2.
    Result<> updatePages(std::span<const VirtualShadowMapPrototypePage> pages);
    Result<> updateTriangles(std::span<const VirtualShadowMapPrototypeTriangle> triangles);
    Result<> updateProjection(const VirtualShadowMapPrototypeProjection& projection);
    // clearAtlas initializes the entire pool and redraws all mapped pages;
    // otherwise only dirty mapped pages are cleared and drawn.
    // Returns atlas in ShaderRead; receiver results and
    // the eight uint32 statistics are available for transfer after barriers.
    Result<> record(CommandBuffer& commands, TextureLayout previousAtlasLayout, bool clearAtlas);
    Texture& atlas() const;
    TextureView& atlasView() const;
    Buffer& receiverResults() const;
    // raster dispatch xyz, clear dispatch xyz, requested work, overflow count.
    Buffer& statistics() const;
    const VirtualShadowMapPrototypeDesc& desc() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render
