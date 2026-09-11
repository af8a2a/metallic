#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <vector>

namespace metallic::render {

struct BakedOpacityMicromap {
    std::vector<uint8_t> data;
    std::vector<OpacityMicromapTriangle> triangles;
    std::vector<OpacityMicromapUsage> usages;
    // Transparent, opaque, unknown-transparent, unknown-opaque microtriangles.
    std::array<uint64_t, 4> stateCounts{};
};

// Conservative coverage of mip-0 nearest/wrap sampling used by scene ray queries.
// A microtriangle is classified only if every texel in its UV bounds agrees.
class OpacityMicromapBaker {
public:
    OpacityMicromapBaker(const scene::RenderMaterial& material, const scene::RenderImage::Mip* image);
    bool bake(const scene::RenderPrimitive& primitive, uint32_t subdivisionLevel, BakedOpacityMicromap& output) const;

private:
    uint8_t classify(const std::array<float2, 3>& uv) const;
    scene::RenderTextureInfo textureInfo_;
    uint32_t width_ = 1, height_ = 1;
    bool valid_ = false;
    std::vector<uint32_t> opaquePrefix_, unknownPrefix_;
};

struct ScenePrimitiveOpacity {
    // -2: no instances; -1: divergent materials cannot share a baked micromap.
    int32_t materialIndex = -2;
    bool usesAlpha = false;
};
std::vector<ScenePrimitiveOpacity> scenePrimitiveOpacity(const scene::Scene& scene);
std::vector<BakedOpacityMicromap> bakeSceneOpacityMicromaps(
    const scene::Scene& scene, std::span<const ScenePrimitiveOpacity> opacity, uint32_t subdivisionLevel);

} // namespace metallic::render
