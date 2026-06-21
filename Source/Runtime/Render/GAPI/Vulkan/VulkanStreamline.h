#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <cstdint>
#include <string>

namespace metallic::render::vulkan {

struct StreamlineDlssRrTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
};

enum class StreamlineDlssRrMode : uint32_t {
    Off,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
    UltraQuality,
};

struct StreamlineDlssRrDesc {
    StreamlineDlssRrTextureRef inputColor;
    StreamlineDlssRrTextureRef outputColor;
    StreamlineDlssRrTextureRef albedo;
    StreamlineDlssRrTextureRef specularAlbedo;
    StreamlineDlssRrTextureRef normalRoughness;
    StreamlineDlssRrTextureRef motionVectors;
    StreamlineDlssRrTextureRef linearDepth;
    StreamlineDlssRrTextureRef specularHitDistance;
    uint32_t width = 0;
    uint32_t height = 0;
    StreamlineDlssRrMode mode = StreamlineDlssRrMode::Balanced;
    bool reset = false;
};

const char* streamlineVulkanLibraryName();
bool streamlineSdkAvailable();
bool streamlineInitialized();
bool streamlineDlssRrSupported();
Result initializeStreamlinePreDevice(std::string& log);
Result setStreamlineVulkanDevice(
    const NativeDevice& device,
    const NativeQueue& graphicsQueue,
    const NativeQueue& computeQueue,
    std::string& log);
void shutdownStreamline();
Result evaluateStreamlineDlssRr(CommandBuffer& commandBuffer, const StreamlineDlssRrDesc& desc, std::string& log);

} // namespace metallic::render::vulkan
