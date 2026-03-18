#ifdef _WIN32

#include "streamline_context.h"
#include <spdlog/spdlog.h>

#ifdef METALLIC_HAS_STREAMLINE

#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>
#include <sl_consts.h>

// Streamline SDK version used at link time
static constexpr uint64_t kSlSdkVersion = sl::kSDKVersion;

static void slLogCallback(sl::LogType type, const char* msg) {
    switch (type) {
    case sl::LogType::eInfo:  spdlog::info("[Streamline] {}", msg); break;
    case sl::LogType::eWarn:  spdlog::warn("[Streamline] {}", msg); break;
    case sl::LogType::eError: spdlog::error("[Streamline] {}", msg); break;
    default: break;
    }
}

static sl::DLSSMode toSlDlssMode(DlssPreset preset) {
    switch (preset) {
    case DlssPreset::Off:              return sl::DLSSMode::eOff;
    case DlssPreset::MaxPerformance:   return sl::DLSSMode::eMaxPerformance;
    case DlssPreset::Balanced:         return sl::DLSSMode::eBalanced;
    case DlssPreset::MaxQuality:       return sl::DLSSMode::eMaxQuality;
    case DlssPreset::UltraPerformance: return sl::DLSSMode::eUltraPerformance;
    case DlssPreset::UltraQuality:     return sl::DLSSMode::eUltraQuality;
    case DlssPreset::DLAA:             return sl::DLSSMode::eDLAA;
    default:                           return sl::DLSSMode::eOff;
    }
}

// Build a 4x4 clipToPrevClip matrix from two column-major 4x4 arrays
// clipToPrevClip = prevViewProj * inverse(currentViewProj)
// The caller is expected to provide this pre-computed.

static sl::float4x4 toSlMatrix(const float* m) {
    sl::float4x4 out{};
    memcpy(&out, m, sizeof(float) * 16);
    return out;
}

#endif // METALLIC_HAS_STREAMLINE

const char* dlssPresetName(DlssPreset preset) {
    switch (preset) {
    case DlssPreset::Off:              return "Off";
    case DlssPreset::MaxPerformance:   return "Max Performance";
    case DlssPreset::Balanced:         return "Balanced";
    case DlssPreset::MaxQuality:       return "Max Quality";
    case DlssPreset::UltraPerformance: return "Ultra Performance";
    case DlssPreset::UltraQuality:     return "Ultra Quality";
    case DlssPreset::DLAA:             return "DLAA";
    default:                           return "Unknown";
    }
}

StreamlineContext::StreamlineContext() = default;

StreamlineContext::~StreamlineContext() {
    shutdown();
}

#ifdef METALLIC_HAS_STREAMLINE

bool StreamlineContext::init(const char* projectRoot, uint32_t applicationId) {
    if (m_initialized) return true;

    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = slLogCallback;
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking
               | sl::PreferenceFlags::eUseManualHooking;
    pref.featuresToLoad = &sl::kFeatureDLSS;
    pref.numFeaturesToLoad = 1;
    pref.renderAPI = sl::RenderAPI::eVulkan;
    pref.applicationId = applicationId;

    // Set plugin search paths to the executable directory
    // (Streamline DLLs are copied there by CMake post-build)
    wchar_t pluginPath[1] = {0};
    pref.pathsToPlugins = nullptr;
    pref.numPathsToPlugins = 0;

    sl::Result result = slInit(pref, kSlSdkVersion);
    if (result != sl::Result::eOk) {
        spdlog::warn("Streamline init failed (result={}); DLSS will be unavailable",
                      static_cast<int>(result));
        return false;
    }

    m_initialized = true;
    spdlog::info("Streamline SDK initialized");
    return true;
}

bool StreamlineContext::setVulkanDevice(VkInstance instance,
                                        VkPhysicalDevice physicalDevice,
                                        VkDevice device,
                                        VkQueue graphicsQueue,
                                        uint32_t graphicsQueueFamily,
                                        uint32_t graphicsQueueIndex) {
    if (!m_initialized) return false;
    if (m_vulkanSet) return true;

    sl::VulkanInfo vkInfo{};
    vkInfo.device = device;
    vkInfo.instance = instance;
    vkInfo.physicalDevice = physicalDevice;
    vkInfo.graphicsQueueFamily = graphicsQueueFamily;
    vkInfo.graphicsQueueIndex = graphicsQueueIndex;
    // We don't allocate extra compute/optical-flow queues in the MVP.
    // SL will use the graphics queue for DLSS compute work.
    vkInfo.computeQueueFamily = graphicsQueueFamily;
    vkInfo.computeQueueIndex = graphicsQueueIndex;

    sl::Result result = slSetVulkanInfo(vkInfo);
    if (result != sl::Result::eOk) {
        spdlog::warn("Streamline setVulkanInfo failed (result={})", static_cast<int>(result));
        return false;
    }
    m_vulkanSet = true;

    // Check DLSS availability
    sl::AdapterInfo adapterInfo{};
    adapterInfo.vkPhysicalDevice = physicalDevice;
    sl::Result dlssResult = slIsFeatureSupported(sl::kFeatureDLSS, adapterInfo);
    m_dlssAvailable = (dlssResult == sl::Result::eOk);

    if (m_dlssAvailable) {
        spdlog::info("DLSS Super Resolution is available");
    } else {
        spdlog::info("DLSS Super Resolution is NOT available (result={})",
                      static_cast<int>(dlssResult));
    }

    return true;
}

bool StreamlineContext::getOptimalRenderSize(DlssPreset preset,
                                             uint32_t displayWidth, uint32_t displayHeight,
                                             uint32_t& outRenderWidth, uint32_t& outRenderHeight) const {
    if (!m_dlssAvailable) return false;

    sl::DLSSOptions options{};
    options.mode = toSlDlssMode(preset);
    options.outputWidth = displayWidth;
    options.outputHeight = displayHeight;

    sl::DLSSOptimalSettings settings{};
    sl::Result result = slDLSSGetOptimalSettings(options, settings);
    if (result != sl::Result::eOk || settings.optimalRenderWidth == 0) {
        return false;
    }

    outRenderWidth = settings.optimalRenderWidth;
    outRenderHeight = settings.optimalRenderHeight;
    return true;
}

bool StreamlineContext::setDlssOptions(DlssPreset preset, uint32_t outputWidth, uint32_t outputHeight) {
    if (!m_dlssAvailable) return false;

    sl::DLSSOptions options{};
    options.mode = toSlDlssMode(preset);
    options.outputWidth = outputWidth;
    options.outputHeight = outputHeight;
    options.colorBuffersHDR = sl::Boolean::eTrue;
    options.preExposure = 1.0f;

    sl::ViewportHandle viewport{0};
    sl::Result result = slDLSSSetOptions(viewport, options);
    if (result != sl::Result::eOk) {
        spdlog::error("slDLSSSetOptions failed (result={})", static_cast<int>(result));
        return false;
    }

    m_currentPreset = preset;
    m_currentOutputWidth = outputWidth;
    m_currentOutputHeight = outputHeight;
    return true;
}

bool StreamlineContext::evaluate(const StreamlineDlssFrameData& data) {
    if (!m_dlssAvailable || m_currentPreset == DlssPreset::Off) return false;

    sl::FrameToken* frameToken = nullptr;
    sl::Result result = slGetNewFrameToken(frameToken, &data.frameIndex);
    if (result != sl::Result::eOk || !frameToken) {
        spdlog::error("slGetNewFrameToken failed");
        return false;
    }

    sl::ViewportHandle viewport{0};

    // Set per-frame constants
    sl::Constants constants{};
    constants.mvecScale = {data.mvecScaleX, data.mvecScaleY};
    constants.jitterOffset = {data.jitterOffsetX, data.jitterOffsetY};
    constants.depthInverted = data.depthInverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = data.motionVectorsJittered ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.reset = (data.reset || m_needsReset) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    memcpy(&constants.clipToPrevClip, data.clipToPrevClip, sizeof(float) * 16);

    m_needsReset = false;

    result = slSetConstants(constants, *frameToken, viewport);
    if (result != sl::Result::eOk) {
        spdlog::error("slSetConstants failed (result={})", static_cast<int>(result));
        return false;
    }

    // Tag resources
    sl::Resource colorIn{sl::ResourceType::eTex2d, data.colorInput, nullptr,
                         data.colorInputView, 0};
    sl::Resource colorOut{sl::ResourceType::eTex2d, data.colorOutput, nullptr,
                          data.colorOutputView, 0};
    sl::Resource depthRes{sl::ResourceType::eTex2d, data.depth, nullptr,
                          data.depthView, 0};
    sl::Resource mvecRes{sl::ResourceType::eTex2d, data.motionVectors, nullptr,
                         data.motionVectorsView, 0};

    sl::Extent renderExtent{};
    renderExtent.width = data.renderWidth;
    renderExtent.height = data.renderHeight;

    sl::ResourceTag tags[] = {
        {&colorIn,  sl::kBufferTypeScalingInputColor,  sl::ResourceLifecycle::eValidUntilPresent, &renderExtent},
        {&colorOut, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilPresent},
        {&depthRes, sl::kBufferTypeDepth,              sl::ResourceLifecycle::eValidUntilPresent, &renderExtent},
        {&mvecRes,  sl::kBufferTypeMotionVectors,      sl::ResourceLifecycle::eValidUntilPresent, &renderExtent},
    };

    result = slSetTagForFrame(*frameToken, viewport, tags, _countof(tags),
                              static_cast<sl::CommandBuffer*>(data.commandBuffer));
    if (result != sl::Result::eOk) {
        spdlog::error("slSetTagForFrame failed (result={})", static_cast<int>(result));
        return false;
    }

    // Evaluate DLSS
    const sl::BaseStructure* inputs[] = {nullptr};
    result = slEvaluateFeature(sl::kFeatureDLSS, *frameToken, inputs, 0,
                               static_cast<sl::CommandBuffer*>(data.commandBuffer));
    if (result != sl::Result::eOk) {
        spdlog::error("slEvaluateFeature(DLSS) failed (result={})", static_cast<int>(result));
        return false;
    }

    return true;
}

void StreamlineContext::resetHistory() {
    m_needsReset = true;
}

void StreamlineContext::shutdown() {
    if (!m_initialized) return;

    if (m_dlssAvailable) {
        sl::ViewportHandle viewport{0};
        slFreeResources(sl::kFeatureDLSS, viewport);
    }

    slShutdown();
    m_initialized = false;
    m_vulkanSet = false;
    m_dlssAvailable = false;
    spdlog::info("Streamline shutdown");
}

std::string StreamlineContext::statusString() const {
    if (!m_initialized) return "Streamline: Not initialized";
    if (!m_vulkanSet) return "Streamline: Vulkan not configured";
    if (!m_dlssAvailable) return "DLSS: Unavailable (non-NVIDIA or unsupported GPU)";
    return "DLSS: Available";
}

#else // !METALLIC_HAS_STREAMLINE — stub implementation

bool StreamlineContext::init(const char*, uint32_t) {
    spdlog::info("Streamline SDK not available in this build");
    return false;
}

bool StreamlineContext::setVulkanDevice(VkInstance, VkPhysicalDevice, VkDevice,
                                        VkQueue, uint32_t, uint32_t) {
    return false;
}

bool StreamlineContext::getOptimalRenderSize(DlssPreset, uint32_t, uint32_t,
                                             uint32_t&, uint32_t&) const {
    return false;
}

bool StreamlineContext::setDlssOptions(DlssPreset, uint32_t, uint32_t) {
    return false;
}

bool StreamlineContext::evaluate(const StreamlineDlssFrameData&) {
    return false;
}

void StreamlineContext::resetHistory() {}

void StreamlineContext::shutdown() {}

std::string StreamlineContext::statusString() const {
    return "Streamline: Not included in build";
}

#endif // METALLIC_HAS_STREAMLINE

#endif // _WIN32
