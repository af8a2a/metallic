#include "Runtime/Render/Core/ColorGrading.h"
#include "Runtime/Render/Core/ACESTables.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace metallic::render {
namespace {
float number(const RenderGraphProperties& properties, const char* key, float fallback, float low, float high)
{
    const auto it = properties.find(key);
    if (it == properties.end() || !it->is_number()) {
        return fallback;
    }
    const float value = it->get<float>();
    return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
}
void vector(const RenderGraphProperties& properties, const char* key, std::array<float, 4>& values, float minimum)
{
    const auto it = properties.find(key);
    if (it == properties.end() || !it->is_array() || it->size() != 4) {
        return;
    }
    for (size_t i = 0; i < 4; ++i) {
        if ((*it)[i].is_number()) {
            const float value = (*it)[i].get<float>();
            if (std::isfinite(value)) {
                values[i] = std::clamp(value, minimum, 16.0f);
            }
        }
    }
}
} // namespace

ColorGradingParameters colorGradingParameters(const RenderGraphProperties& properties)
{
    ColorGradingParameters p;
    vector(properties, "colorSaturation", p.saturation, 0.0f);
    vector(properties, "colorContrast", p.contrast, 0.01f);
    vector(properties, "colorGamma", p.gamma, 0.01f);
    vector(properties, "colorGain", p.gain, 0.0f);
    vector(properties, "colorOffset", p.offset, -16.0f);
    p.film = {number(properties, "filmSlope", 0.88f, 0.01f, 1), number(properties, "filmToe", 0.55f, 0, 0.99f),
              number(properties, "filmShoulder", 0.26f, 0, 0.99f), number(properties, "filmBlackClip", 0, 0, 1)};
    p.filmExtra = {number(properties, "filmWhiteClip", 0.04f, 0, 1), number(properties, "blueCorrection", 0.6f, 0, 1),
                   number(properties, "expandGamut", 1, 0, 1), number(properties, "toneCurveAmount", 1, 0, 1)};
    float total = 0;
    for (size_t i = 0; i < 4; ++i) {
        const std::string key = "lut" + std::to_string(i + 1);
        const auto path = properties.find(key);
        if (path != properties.end() && path->is_string() && !path->get<std::string>().empty()) {
            p.lutWeights[i] = number(properties, (key + "Weight").c_str(), i == 0 ? 1.0f : 0.0f, 0, 1);
        }
        total += p.lutWeights[i];
    }
    // The editor's four slots imply the remaining neutral contribution. Match
    // UE GenerateFinalTable: keep the strongest occurrence of each texture,
    // discard contributors below 1/512, then normalize including neutral.
    const float neutral = std::max(1.0f - total, 0.0f);
    for (size_t i = 0; i < 4; ++i) {
        if (p.lutWeights[i] < 1.0f / 512.0f) {
            p.lutWeights[i] = 0;
            continue;
        }
        for (size_t j = i + 1; j < 4; ++j) {
            if (p.lutWeights[j] > 0 && properties.value("lut" + std::to_string(i + 1), "") ==
                                           properties.value("lut" + std::to_string(j + 1), "")) {
                if (p.lutWeights[i] > p.lutWeights[j]) {
                    p.lutWeights[j] = 0;
                } else {
                    p.lutWeights[i] = 0;
                    break;
                }
            }
        }
    }
    total = neutral;
    for (float weight : p.lutWeights) {
        total += weight;
    }
    if (total > 0.001f) {
        for (float& weight : p.lutWeights) {
            weight /= total;
        }
    } else {
        p.lutWeights.fill(0);
    }
    return p;
}

std::vector<RenderGraphRuntimeSetting> colorGradingSettings()
{
    using namespace builtin_pass;
    std::vector<RenderGraphRuntimeSetting> settings;
    for (const auto* key : {"colorSaturation", "colorContrast", "colorGamma", "colorGain", "colorOffset"}) {
        const float value = std::string_view(key) == "colorOffset" ? 0.0f : 1.0f;
        settings.push_back({.key = key,
                            .label = std::string(key) + " (RGB / master)",
                            .type = RenderGraphRuntimeSettingType::Float4,
                            .defaultValue = {value, value, value, value}});
    }
    settings.push_back(runtimeFloatSetting("filmSlope", "UE Film Slope (SDR)", 0.88f, 0.01f, 1));
    settings.push_back(runtimeFloatSetting("filmToe", "UE Film Toe (SDR)", 0.55f, 0, 0.99f));
    settings.push_back(runtimeFloatSetting("filmShoulder", "UE Film Shoulder (SDR)", 0.26f, 0, 0.99f));
    settings.push_back(runtimeFloatSetting("filmBlackClip", "UE Film Black Clip (SDR)", 0, 0, 1));
    settings.push_back(runtimeFloatSetting("filmWhiteClip", "UE Film White Clip (SDR)", 0.04f, 0, 1));
    settings.push_back(runtimeFloatSetting("blueCorrection", "UE Blue Correction (SDR)", 0.6f, 0, 1));
    settings.push_back(runtimeFloatSetting("expandGamut", "UE Expand Gamut", 1, 0, 1));
    settings.push_back(runtimeFloatSetting("toneCurveAmount", "UE Tone Curve Amount (SDR)", 1, 0, 1));
    for (int i = 1; i <= 4; ++i) {
        const auto key = "lut" + std::to_string(i);
        settings.push_back({.key = key,
                            .label = "UE custom LUT " + std::to_string(i) + " (SDR, 256x16 image; Enter to load)",
                            .type = RenderGraphRuntimeSettingType::String,
                            .defaultValue = "",
                            .rebuildGraph = true});
        settings.push_back(
            runtimeFloatSetting(key + "Weight", "LUT " + std::to_string(i) + " weight", i == 1 ? 1.0f : 0.0f, 0, 1));
    }
    return settings;
}

std::array<TextureView*, 7> ColorGradingResources::views() const
{
    std::array<TextureView*, 7> result{};
    for (size_t i = 0; i < 7; ++i) {
        result[i] = views_[i].get();
    }
    return result;
}

Result<> ColorGradingResources::initialize(Device& device, const RenderGraphProperties& properties, float peakNits,
                                           bool aces2, std::string& log)
{
    std::array<std::vector<std::byte>, 7> bytes;
    std::array<uint32_t, 7> widths{256, 256, 256, 256, 1, 1, 1}, heights{16, 16, 16, 16, 1, 1, 1};
    for (size_t i = 0; i < 4; ++i) {
        bytes[i].resize(256 * 16 * 4);
        auto* rgba = reinterpret_cast<unsigned char*>(bytes[i].data());
        // Neutral UE unwrapped 16^3 LUT; uploaded as UNORM without sRGB decoding.
        for (uint32_t b = 0; b < 16; ++b) {
            for (uint32_t g = 0; g < 16; ++g) {
                for (uint32_t r = 0; r < 16; ++r) {
                    const uint32_t offset = (g * 256 + b * 16 + r) * 4;
                    rgba[offset] = static_cast<unsigned char>(r * 17);
                    rgba[offset + 1] = static_cast<unsigned char>(g * 17);
                    rgba[offset + 2] = static_cast<unsigned char>(b * 17);
                    rgba[offset + 3] = 255;
                }
            }
        }
        const auto it = properties.find("lut" + std::to_string(i + 1));
        if (it == properties.end() || !it->is_string() || it->get<std::string>().empty()) {
            continue;
        }
        std::filesystem::path path = it->get<std::string>();
        if (path.is_relative()) {
            path = std::filesystem::path(PROJECT_SOURCE_DIR) / path;
        }
        int w = 0, h = 0, channels = 0;
        unsigned char* pixels = stbi_load(path.string().c_str(), &w, &h, &channels, 4);
        if (!pixels || w != 256 || h != 16) {
            stbi_image_free(pixels);
            log = "UE custom LUT must be a readable 256x16 image: " + path.string();
            return makeError(Error::InvalidArgument);
        }
        std::memcpy(bytes[i].data(), pixels, bytes[i].size());
        stbi_image_free(pixels);
    }
    for (size_t i = 4; i < 7; ++i) {
        bytes[i].resize(sizeof(float));
    }
    if (aces2) {
        const auto tables = makeACESTables(peakNits);
        const auto copy = [&](size_t i, const auto& values, uint32_t w, uint32_t h) {
            widths[i] = w;
            heights[i] = h;
            bytes[i].resize(values.size() * sizeof(float));
            std::memcpy(bytes[i].data(), values.data(), bytes[i].size());
        };
        copy(4, tables.reach, 360, 1);
        copy(5, tables.gamut, 3, 362);
        copy(6, tables.gamma, 362, 1);
    }
    auto* queue = device.getQueue(QueueType::Graphics);
    if (!queue) {
        return makeError(Error::Unsupported);
    }
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> command;
    std::unique_ptr<Fence> fence;
    auto result = device.createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); });
    if (!result) {
        return result;
    }
    result = pool->createCommandBuffer().transform([&](auto value) { command = std::move(value); });
    if (!result) {
        return result;
    }
    result = device.createFence(false).transform([&](auto value) { fence = std::move(value); });
    if (!result) {
        return result;
    }
    result = command->begin();
    if (!result) {
        return result;
    }
    std::array<std::unique_ptr<Buffer>, 7> uploads;
    for (size_t i = 0; i < 7; ++i) {
        result = device
                     .createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                                     .format = i < 4 ? Format::RGBA8Unorm : Format::R32Sfloat,
                                     .width = widths[i],
                                     .height = heights[i],
                                     .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute})
                     .transform([&](auto value) { textures_[i] = std::move(value); });
        if (!result) {
            return result;
        }
        result =
            device.createTextureView(*textures_[i], {}).transform([&](auto value) { views_[i] = std::move(value); });
        if (!result) {
            return result;
        }
        result = device
                     .createBuffer({.size = bytes[i].size(),
                                    .usage = BufferUsageBits::TransferSource,
                                    .memoryLocation = MemoryLocation::HostUpload})
                     .transform([&](auto value) { uploads[i] = std::move(value); });
        if (!result) {
            return result;
        }
        void* mapped = uploads[i]->map();
        if (!mapped) {
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, bytes[i].data(), bytes[i].size());
        uploads[i]->flush();
        uploads[i]->unmap();
        TextureBarrierDesc barrier{.texture = textures_[i].get(),
                                   .oldLayout = TextureLayout::Undefined,
                                   .newLayout = TextureLayout::TransferDestination,
                                   .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
        result = command->synchronize({.textures = {&barrier, 1}});
        if (!result) {
            return result;
        }
        command->copyBufferToTexture({.buffer = uploads[i].get(),
                                      .texture = textures_[i].get(),
                                      .bufferRowPitch = widths[i] * 4,
                                      .bufferSlicePitch = widths[i] * heights[i] * 4,
                                      .width = widths[i],
                                      .height = heights[i]});
        barrier.oldLayout = TextureLayout::TransferDestination;
        barrier.newLayout = TextureLayout::ShaderRead;
        barrier.before = {PipelineStageBits::Transfer, AccessBits::TransferWrite};
        barrier.after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead};
        result = command->synchronize({.textures = {&barrier, 1}});
        if (!result) {
            return result;
        }
    }
    result = command->end();
    if (!result) {
        return result;
    }
    CommandBuffer* commands = command.get();
    result = queue->submit({.commandBuffers = {&commands, 1}, .signalFence = fence.get()});
    if (!result) {
        return result;
    }
    return fence->wait();
}
} // namespace metallic::render
