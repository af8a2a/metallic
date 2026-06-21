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

struct StreamlineDlssRrCamera {
    float eye[3] = {0.0f, 0.0f, 0.0f};
    float center[3] = {0.0f, 0.0f, -1.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float previousEye[3] = {0.0f, 0.0f, 0.0f};
    float previousCenter[3] = {0.0f, 0.0f, -1.0f};
    float previousUp[3] = {0.0f, 1.0f, 0.0f};
    float fovRadians = 0.87266463f;
    float previousFovRadians = 0.87266463f;
    float aspectRatio = 1.0f;
    float previousAspectRatio = 1.0f;
    float zNear = 0.001f;
    float previousZNear = 0.001f;
    float zFar = 10000.0f;
    float previousZFar = 10000.0f;
    float orthoHeight = 1.0f;
    float previousOrthoHeight = 1.0f;
    bool orthographic = false;
    bool previousOrthographic = false;
    bool previousValid = false;
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
    StreamlineDlssRrCamera camera;
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
