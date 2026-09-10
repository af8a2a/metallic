#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace metallic::render {

// Shader ABI: Libraries/Scene/SceneShadingVertex.slang. UVs remain float32
// to preserve tiling and large/negative texture coordinates.
struct SceneShadingVertex {
    uint32_t normal = 0x80008000u;
    uint32_t tangent = 0;
    float texcoord[2] = {};
};
static_assert(sizeof(SceneShadingVertex) == 16);
static_assert(offsetof(SceneShadingVertex, texcoord) == 8);

inline constexpr uint32_t kSceneFallbackPositionsBinding = 54;

namespace detail {

inline bool encodeSceneOctahedron(float x, float y, float z, float& u, float& v)
{
    const float length1 = std::abs(x) + std::abs(y) + std::abs(z);
    if (!std::isfinite(length1) || length1 <= 1e-20f) { return false; }
    u = x / length1;
    v = y / length1;
    if (z < 0.0f) {
        const float oldU = u;
        u = (1.0f - std::abs(v)) * (oldU < 0.0f ? -1.0f : 1.0f);
        v = (1.0f - std::abs(oldU)) * (v < 0.0f ? -1.0f : 1.0f);
    }
    return true;
}

inline uint32_t packSceneSnorm(float value, uint32_t bits)
{
    const int32_t maximum = (1 << (bits - 1)) - 1;
    const int32_t quantized = static_cast<int32_t>(std::round(std::clamp(value, -1.0f, 1.0f) * maximum));
    return static_cast<uint32_t>(quantized) & ((1u << bits) - 1u);
}

} // namespace detail

inline uint32_t packSceneNormal(float x, float y, float z)
{
    float u, v;
    if (!detail::encodeSceneOctahedron(x, y, z, u, v)) {
        // -32768 is never emitted by snorm quantization. Preserve missing/zero
        // normals so the shader can fall back to the geometric normal.
        return 0x80008000u;
    }
    return detail::packSceneSnorm(u, 16) | (detail::packSceneSnorm(v, 16) << 16);
}

inline uint32_t packSceneTangent(float x, float y, float z, float sign)
{
    const uint32_t handedness = sign < 0.0f ? 0x40000000u : 0u;
    float u, v;
    if (!detail::encodeSceneOctahedron(x, y, z, u, v)) { return handedness; }
    // Octahedral snorm15 x 2, handedness, direction-valid bit.
    return detail::packSceneSnorm(u, 15) | (detail::packSceneSnorm(v, 15) << 15) |
        handedness | 0x80000000u;
}

} // namespace metallic::render
