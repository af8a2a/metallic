#pragma once

#include "Runtime/Render/TessellationPatterns.h"
#include "Runtime/Scene/Scene.h"
#include <bit>
#include <span>

namespace metallic::render {

inline float tessellationDisplacementBound(std::span<const scene::RenderMaterial> materials)
{
    float bound = 0.0f;
    for (const auto& material : materials) {
        if (material.displacementTexture.textureIndex < 0) { continue; }
        bound = std::max(bound, std::abs(material.displacementMagnitude) *
            std::max(material.displacementCenter, 1.0f - material.displacementCenter));
    }
    return bound;
}

inline std::vector<uint32_t> buildTessellationData(std::span<const scene::RenderMaterial> materials,
    std::span<const uint32_t> textureDescriptors)
{
    std::vector<uint32_t> data(16 + materials.size() * 16);
    data[0] = static_cast<uint32_t>(data.size());
    data[1] = static_cast<uint32_t>(materials.size());
    // Remaining header words are reserved. Per-draw quality settings live in
    // push constants, independent of this immutable material/pattern table.
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& material = materials[i];
        const auto& texture = material.displacementTexture;
        const size_t base = 16 + i * 16;
        const bool valid = texture.textureIndex >= 0 && size_t(texture.textureIndex) < textureDescriptors.size() &&
            texture.texCoord == 0 && textureDescriptors[texture.textureIndex] != UINT32_MAX;
        data[base] = valid ? textureDescriptors[texture.textureIndex] : UINT32_MAX;
        data[base + 1] = std::bit_cast<uint32_t>(material.displacementMagnitude);
        data[base + 2] = std::bit_cast<uint32_t>(material.displacementCenter);
        data[base + 3] = valid && material.displacementMagnitude != 0.0f ? 1u : 0u;
        // RenderTextureInfo stores the same row-major 2x3 transform as GPUScene.
        for (size_t j = 0; j < 3; ++j) {
            data[base + 4 + j] = std::bit_cast<uint32_t>(texture.uvTransform[j]);
            data[base + 8 + j] = std::bit_cast<uint32_t>(texture.uvTransform[j + 3]);
        }
    }
    static const auto patterns = buildTessellationPatterns();
    data.insert(data.end(), patterns.begin(), patterns.end());
    return data;
}

} // namespace metallic::render
