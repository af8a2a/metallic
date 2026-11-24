#pragma once

#include "Runtime/Render/Core/StaticTextureUpload.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace metallic::render::builtin_pass {

enum class LumenBlueNoise { Vec2, Scalar };

inline Result<std::unique_ptr<TextureView>> loadLumenBlueNoise(
    Device& device, Queue& queue, LumenBlueNoise kind, std::string& log, Queue* computeConsumer = nullptr)
{
    constexpr uint32_t width = 128, height = 128 * 64;
    const bool scalar = kind == LumenBlueNoise::Scalar;
    const uint32_t byteSize = width * height * (scalar ? 1u : 2u);
    const auto path = std::filesystem::path(PROJECT_SOURCE_DIR "/Asset/Lumen") /
        (scalar ? "FastBlueNoiseScalar.r8" : "FastBlueNoiseVec2.rg8");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != byteSize) {
        log = "Expected a " + std::to_string(byteSize) + "-byte blue-noise texture at " + path.string() +
            ". Restore the texture from the project assets.";
        return makeError(Error::InvalidArgument);
    }
    auto staging = device.createBuffer({.size = byteSize, .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload});
    if (!staging) { return makeError(staging.error()); }
    auto* mapped = static_cast<char*>((*staging)->map());
    if (!mapped) { return makeError(Error::Failure); }
    file.seekg(0).read(mapped, byteSize);
    if (!file) {
        (*staging)->unmap();
        log = "Failed to read blue noise from " + path.string();
        return makeError(Error::Failure);
    }
    (*staging)->flush();
    (*staging)->unmap();
    auto texture = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
        .format = scalar ? Format::R8Unorm : Format::RG8Unorm, .width = width, .height = height,
        .queueAccess = computeConsumer ? QueueAccessBits::Graphics | QueueAccessBits::Compute : QueueAccessBits::Graphics});
    if (!texture) { return makeError(texture.error()); }
    auto upload = uploadStaticComputeTexture(device, queue, **texture, **staging, computeConsumer);
    if (!upload) { return makeError(upload.error()); }
    return device.createTextureView(**texture, {});
}

} // namespace metallic::render::builtin_pass
