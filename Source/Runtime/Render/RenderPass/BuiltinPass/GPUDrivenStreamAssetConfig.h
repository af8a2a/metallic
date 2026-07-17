#pragma once

#include "Runtime/Render/MeshletStreamPageLoader.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace metallic::render::builtin_pass {

inline uint32_t pageLoadConcurrencyFromProperties(
    const RenderGraphProperties& properties,
    uint32_t fallback = 2)
{
    const auto readProperty = [&properties](const char* key, uint32_t propertyFallback) {
        auto iter = properties.find(key);
        if (iter == properties.end() || !iter->is_number_integer()) {
            return propertyFallback;
        }
        const int64_t value = iter->get<int64_t>();
        return value < 0 || value > std::numeric_limits<uint32_t>::max()
            ? propertyFallback
            : static_cast<uint32_t>(value);
    };

    const uint32_t legacyValue = readProperty("pageLoadWorkerCount", fallback);
    return std::min(
        readProperty("pageLoadConcurrency", legacyValue),
        kMeshletStreamMaxPageLoadConcurrency);
}

} // namespace metallic::render::builtin_pass
