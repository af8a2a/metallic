#pragma once

// Streamline / DLSS Super Resolution integration for Vulkan.
// This is a Windows-only, Vulkan-only integration layer.
// When METALLIC_HAS_STREAMLINE is not defined, all functions are stubs.

#ifdef _WIN32

#include <cstdint>
#include <string>

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

// Per-frame data needed by Streamline evaluate
struct StreamlineDlssFrameData {
    // Vulkan resources (native handles)
    VkImage colorInput = nullptr;       // render-res HDR color (scaling input)
    VkImageView colorInputView = nullptr;
    VkImage colorOutput = nullptr;      // display-res HDR color (scaling output)
    VkImageView colorOutputView = nullptr;
    VkImage depth = nullptr;
    VkImageView depthView = nullptr;
    VkImage motionVectors = nullptr;
    VkImageView motionVectorsView = nullptr;

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

    // Matrices (column-major, unjittered)
    float clipToPrevClip[16] = {};

    uint32_t frameIndex = 0;
    void* commandBuffer = nullptr; // VkCommandBuffer
};

// Streamline context — manages SDK lifecycle and DLSS feature
class StreamlineContext {
public:
    StreamlineContext();
    ~StreamlineContext();

    // Non-copyable
    StreamlineContext(const StreamlineContext&) = delete;
    StreamlineContext& operator=(const StreamlineContext&) = delete;

    // Phase 1: Initialize Streamline before Vulkan device creation.
    // Returns false if SL is unavailable (non-NVIDIA, missing DLLs, etc.)
    bool init(const char* projectRoot, uint32_t applicationId = 0);

    // Phase 2: Provide Vulkan handles after device creation.
    bool setVulkanDevice(VkInstance instance,
                         VkPhysicalDevice physicalDevice,
                         VkDevice device,
                         VkQueue graphicsQueue,
                         uint32_t graphicsQueueFamily,
                         uint32_t graphicsQueueIndex = 0);

    // Query DLSS availability on current adapter
    bool isDlssAvailable() const { return m_dlssAvailable; }
    bool isInitialized() const { return m_initialized; }

    // Get optimal render resolution for a given display size and preset
    bool getOptimalRenderSize(DlssPreset preset,
                              uint32_t displayWidth, uint32_t displayHeight,
                              uint32_t& outRenderWidth, uint32_t& outRenderHeight) const;

    // Set DLSS mode for the default viewport
    bool setDlssOptions(DlssPreset preset, uint32_t outputWidth, uint32_t outputHeight);

    // Evaluate DLSS for the current frame
    bool evaluate(const StreamlineDlssFrameData& data);

    // Reset temporal history (call on resize, camera cut, pipeline reload, preset change)
    void resetHistory();

    // Shutdown
    void shutdown();

    // Status string for UI
    std::string statusString() const;

private:
    bool m_initialized = false;
    bool m_vulkanSet = false;
    bool m_dlssAvailable = false;
    DlssPreset m_currentPreset = DlssPreset::Off;
    uint32_t m_currentOutputWidth = 0;
    uint32_t m_currentOutputHeight = 0;
    bool m_needsReset = false;
};

#endif // _WIN32
