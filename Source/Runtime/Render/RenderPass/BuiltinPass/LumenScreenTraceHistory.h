#pragma once

#include "Runtime/Render/Core/HistoryResources.h"

#include <array>
#include <string>

namespace metallic::render::builtin_pass {

inline std::array<std::string, 2> lumenScreenTraceHistoryNames(std::string_view producer)
{
    const auto prefix = "LumenScreenTrace." + std::string(producer) + ".";
    return {prefix + "depth", prefix + "color"};
}

inline Result<> ensureLumenScreenTraceHistory(HistoryResourceManager& history,
    const std::array<std::string, 2>& names, uint32_t width, uint32_t height)
{
    constexpr std::array formats{Format::R32Sfloat, Format::RGBA32Sfloat};
    for (size_t index = 0; index < names.size(); ++index) {
        const TextureDesc desc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = formats[index],
            .width = width, .height = height, .depth = 1, .mipCount = 1, .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        };
        if (auto result = history.ensureTexture(names[index], desc); !result) { return result; }
    }
    return {};
}

} // namespace metallic::render::builtin_pass
