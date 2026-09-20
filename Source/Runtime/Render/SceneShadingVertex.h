#pragma once
#include "Runtime/Scene/GeometryEncoding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace metallic::render {

// Shader ABI: Modules/Core/SceneShadingVertex.slang. UVs remain float32
// to preserve tiling and large/negative texture coordinates.
struct SceneShadingVertex {
    uint32_t normal = 0x80008000u;
    uint32_t tangent = 0;
    float texcoord[2] = {};
};
static_assert(sizeof(SceneShadingVertex) == 16);
static_assert(offsetof(SceneShadingVertex, texcoord) == 8);

inline constexpr uint32_t kSceneFallbackPositionsBinding = 54;

using scene::packSceneNormal;
using scene::packSceneTangent;

} // namespace metallic::render
