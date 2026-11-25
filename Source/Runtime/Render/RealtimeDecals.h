#pragma once

#include "Runtime/Render/Material/DeferredDecalData.h"

#include <span>
#include <vector>

namespace metallic::scene { struct RenderDecal; }
namespace metallic::render::builtin_pass { struct ScenePathTracePush; }

namespace metallic::render {

class ScenePathTraceResources;

struct DecalTileRange {
    uint32_t first = 0;
    uint32_t count = 0;
};
static_assert(sizeof(DecalTileRange) == 8);

struct RealtimeDecalView {
    std::vector<DeferredDecalData> records;
    std::vector<DeferredDecalMaterial> materials;
    std::vector<DecalTileRange> tileRanges;
    std::vector<uint32_t> tileIndices;
    // Zero selects a full scan of records and requires no tile buffers.
    uint32_t tilesX = 0;
};

// Both modes blend in ascending priority, then persistent projector ID. The
// full scan includes offscreen projectors so it can serve as a culling oracle.
RealtimeDecalView buildRealtimeDecalView(std::span<const scene::RenderDecal> decals,
    const builtin_pass::ScenePathTracePush& camera, const ScenePathTraceResources& resources,
    bool fullScan = false);

} // namespace metallic::render
