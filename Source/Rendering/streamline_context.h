#pragma once

// Streamline / DLSS Super Resolution integration for Vulkan.
// This is a Windows-only, Vulkan-only integration layer.
// When METALLIC_HAS_STREAMLINE is not defined, all functions are stubs.

#ifdef _WIN32

#include <cstdint>
#include <string>
#include <vector>

#include "rhi_interop.h"

// Forward declare Vulkan handles without pulling in vulkan.h.
// These are opaque pointer types defined by the Vulkan spec.
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkImage_T;
struct VkImageView_T;
typedef VkInstance_T* VkInstance;
typedef VkPhysicalDevice_T* VkPhysicalDevice;
typedef VkDevice_T* VkDevice;
typedef VkQueue_T* VkQueue;
typedef VkImage_T* VkImage;
typedef VkImageView_T* VkImageView;

class RhiTexture;

// DLSS quality presets (mirrors sl::DLSSMode ordering)
enum class DlssPreset : uint32_t {
    Off = 0,
    MaxPerformance,
    Balanced,
    MaxQuality,
    UltraPerformance,
    UltraQuality,
    DLAA,
    Count
};

const char* dlssPresetName(DlssPreset preset);

// Requirements that Streamline/DLSS needs from the Vulkan device
struct StreamlineVulkanRequirements {
    std::vector<const char*> instanceExtensions;
    std::vector<const char*> deviceExtensions;
    bool needsTimelineSemaphore = false;
};

// Per-frame data needed by Streamline evaluate
struct StreamlineDlssFrameData {
    struct VulkanTextureInfo {
        uint32_t state = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t nativeFormat = 0;
        uint32_t mipLevels = 1;
        uint32_t arrayLayers = 1;
        uint32_t flags = 0;
        uint32_t usage = 0;
    };

    // Vulkan resources (native handles)
    VkImage colorInput = nullptr;       // render-res HDR color (scaling input)
    VkImageView colorInputView = nullptr;
    VulkanTextureInfo colorInputInfo;
    VkImage colorOutput = nullptr;      // display-res HDR color (scaling output)
    VkImageView colorOutputView = nullptr;
    VulkanTextureInfo colorOutputInfo;
    VkImage depth = nullptr;
    VkImageView depthView = nullptr;
    VulkanTextureInfo depthInfo;
    VkImage motionVectors = nullptr;
    VkImageView motionVectorsView = nullptr;
    VulkanTextureInfo motionVectorsInfo;

    // Dimensions
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth = 0;
    uint32_t displayHeight = 0;

    // Camera / temporal
    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;
    float mvecScaleX = 1.0f;
    float mvecScaleY = 1.0f;
    bool motionVectorsJittered = false;
    bool depthInverted = false;
    bool reset = false;
    bool motionVectors3D = false;

    // Matrices (column-major, unjittered)
    float cameraViewToClip[16] = {};
    float clipToCameraView[16] = {};
    float clipToPrevClip[16] = {};
    float prevClipToClip[16] = {};

    float cameraPos[3] = {};
    float cameraUp[3] = {};
    float cameraRight[3] = {};
    float cameraForward[3] = {};
    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    float cameraFov = 0.0f;
    float cameraAspectRatio = 1.0f;

    uint32_t frameIndex = 0;
    void* commandBuffer = nullptr; // VkCommandBuffer
};

// Streamline context — manages SDK lifecycle and DLSS feature
class StreamlineContext : public IUpscalerIntegration {
public:
    StreamlineContext();
    ~StreamlineContext();

    // Non-copyable
    StreamlineContext(const StreamlineContext&) = delete;
    StreamlineContext& operator=(const StreamlineContext&) = delete;

    // Phase 1: Initialize Streamline before Vulkan device creation.
    // Returns false if SL is unavailable (non-NVIDIA, missing DLLs, etc.)
    bool init(const char* projectRoot, uint32_t applicationId = 073432);

    // Phase 1b: Query Vulkan extensions/features required by DLSS.
    // Call after init(), before device creation.
    bool queryVulkanRequirements(StreamlineVulkanRequirements& out) const;

    // Phase 2: Provide Vulkan handles after device creation.
    bool setVulkanDevice(VkInstance instance,
                         VkPhysicalDevice physicalDevice,
                         VkDevice device,
                         VkQueue graphicsQueue,
                         uint32_t graphicsQueueFamily,
                         uint32_t graphicsQueueIndex = 0);

    // Query DLSS availability on current adapter
    bool isDlssAvailable() const { return m_dlssAvailable; }
    bool isAvailable() const override { return m_dlssAvailable; }
    bool isEnabled() const override { return m_currentPreset != DlssPreset::Off; }
    bool isInitialized() const { return m_initialized; }
    void* vulkanDeviceProcAddrProxy() const { return m_vkGetDeviceProcAddrProxy; }
    void* vulkanBeginCommandBufferHook() const { return m_vkBeginCommandBufferHook; }
    void* vulkanCmdBindPipelineHook() const { return m_vkCmdBindPipelineHook; }
    void* vulkanCmdBindDescriptorSetsHook() const { return m_vkCmdBindDescriptorSetsHook; }
    void setImageLayoutTracker(class VulkanResourceStateTracker* tracker) { m_imageLayoutTracker = tracker; }

    // Get optimal render resolution for a given display size and preset
    bool getOptimalRenderSize(DlssPreset preset,
                              uint32_t displayWidth, uint32_t displayHeight,
                              uint32_t& outRenderWidth, uint32_t& outRenderHeight) const;
    bool getOptimalRenderSize(uint32_t displayWidth,
                              uint32_t displayHeight,
                              uint32_t& outRenderWidth,
                              uint32_t& outRenderHeight) const override;

    // Set DLSS mode for the default viewport
    bool setDlssOptions(DlssPreset preset, uint32_t outputWidth, uint32_t outputHeight);

    // Evaluate DLSS for the current frame
    bool evaluate(const StreamlineDlssFrameData& data);
    bool evaluate(const UpscalerEvaluateInputs& inputs,
                  RhiComputeCommandEncoder& encoder) override;

    // Reset temporal history (call on resize, camera cut, pipeline reload, preset change)
    void resetHistory() override;

    // Shutdown
    void shutdown();

    // Status string for UI
    std::string statusString() const override;

private:
    bool m_initialized = false;
    bool m_vulkanSet = false;
    bool m_dlssAvailable = false;
    DlssPreset m_currentPreset = DlssPreset::Off;
    uint32_t m_currentOutputWidth = 0;
    uint32_t m_currentOutputHeight = 0;
    bool m_needsReset = false;
    void* m_vkGetDeviceProcAddrProxy = nullptr;
    void* m_vkBeginCommandBufferHook = nullptr;
    void* m_vkCmdBindPipelineHook = nullptr;
    void* m_vkCmdBindDescriptorSetsHook = nullptr;
    class VulkanResourceStateTracker* m_imageLayoutTracker = nullptr;
};

#endif // _WIN32
