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

struct StreamlineDlssRrOptimalSettings {
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t renderWidthMin = 0;
    uint32_t renderHeightMin = 0;
    uint32_t renderWidthMax = 0;
    uint32_t renderHeightMax = 0;
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
    float jitterOffset[2] = {0.0f, 0.0f};
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
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssRrCamera camera;
    StreamlineDlssRrMode mode = StreamlineDlssRrMode::Balanced;
    bool reset = false;
};

using StreamlineDlssSrTextureRef = StreamlineDlssRrTextureRef;
using StreamlineDlssSrMode = StreamlineDlssRrMode;
using StreamlineDlssSrOptimalSettings = StreamlineDlssRrOptimalSettings;
using StreamlineDlssSrCamera = StreamlineDlssRrCamera;

struct StreamlineDlssSrDesc {
    StreamlineDlssSrTextureRef inputColor;
    StreamlineDlssSrTextureRef outputColor;
    StreamlineDlssSrTextureRef motionVectors;
    StreamlineDlssSrTextureRef depth;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssSrCamera camera;
    StreamlineDlssSrMode mode = StreamlineDlssSrMode::Balanced;
    bool reset = false;
};

const char* streamlineVulkanLibraryName();
bool streamlineSdkAvailable();
bool streamlineInitialized();
bool streamlineDlssSrSupported();
bool streamlineDlssRrSupported();
Result getStreamlineDlssSrOptimalSettings(
    StreamlineDlssSrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssSrOptimalSettings& settings,
    std::string& log);
Result getStreamlineDlssRrOptimalSettings(
    StreamlineDlssRrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssRrOptimalSettings& settings,
    std::string& log);
Result initializeStreamlinePreDevice(std::string& log);
Result setStreamlineVulkanDevice(
    const NativeDevice& device,
    const NativeQueue& graphicsQueue,
    const NativeQueue& computeQueue,
    std::string& log);
void shutdownStreamline();
Result evaluateStreamlineDlssSr(CommandBuffer& commandBuffer, const StreamlineDlssSrDesc& desc, std::string& log);
Result evaluateStreamlineDlssRr(CommandBuffer& commandBuffer, const StreamlineDlssRrDesc& desc, std::string& log);

} // namespace metallic::render::vulkan
